# -*- coding: utf-8 -*-
"""
mqtt_handler.py — Callbacks MQTT RoboCare v2.6
Corrections v2.6 :
  - on_message : ne bloque plus les topics valve/pump (< 7 parts)
  - on_connect : subscribe aux topics valve/pump pour logs
  - auth MQTT supprimée (Mosquitto allow_anonymous)
"""

import json
import time
import threading

from firebase_admin import firestore

import config as cfg
from services.firebase_service import update_sensor_data, deactivate_sensor
from services.notification_service import (
    send_critical_alert,
    send_irrigation_prediction,
    send_thresholds_summary_alert,
    send_irrigation_finished_alert,
    send_sensor_failure_alert,
    send_pump_forgotten_alert,
)
from services.chatbot_service import handle_chatbot_async
from ml.predictor import predict_stress_risk, predict_irrigation_combined
from utils.logger import get_logger

logger = get_logger("robocare.mqtt")


# ── État interne ─────────────────────────────────────────────────────────────

_command_watchers:   dict  = {}
_zone_watchers:      dict  = {}
_processed_msg_ids:  set   = set()
_watcher_start_time: float = time.time()

VALVE_DELAY_S = 0.3

mqtt_client = None
db          = None


def init(client, firestore_db):
    global mqtt_client, db
    mqtt_client = client
    db          = firestore_db


# ── Helpers irrigation ────────────────────────────────────────────────────────

def _publish_irrigation_start(uid: str, zone_num: str) -> None:
    def _do_start():
        logger.info("Zone %s → Irrigation START (vanne puis pompe)", zone_num)

        mqtt_client.publish(
            f"robocare/{uid}/valve/control/{zone_num}",
            "1",
            qos=1,
        )

        time.sleep(VALVE_DELAY_S)

        mqtt_client.publish(
            f"robocare/{uid}/pump/control",
            "1",
            qos=1,
        )

        logger.info(
            "Zone %s → Vanne + Pompe ON (délai %.0fms respecté)",
            zone_num,
            VALVE_DELAY_S * 1000,
        )

    threading.Thread(
        target=_do_start,
        daemon=True,
        name=f"irrig-start-z{zone_num}",
    ).start()


def _publish_irrigation_stop(uid: str, zone_num: str) -> None:
    def _do_stop():
        logger.info("Zone %s → Irrigation STOP (pompe puis vanne)", zone_num)

        mqtt_client.publish(
            f"robocare/{uid}/pump/control",
            "0",
            qos=1,
        )

        time.sleep(VALVE_DELAY_S)

        mqtt_client.publish(
            f"robocare/{uid}/valve/control/{zone_num}",
            "0",
            qos=1,
        )

        logger.info(
            "Zone %s → Pompe + Vanne OFF (délai %.0fms respecté)",
            zone_num,
            VALVE_DELAY_S * 1000,
        )

    threading.Thread(
        target=_do_stop,
        daemon=True,
        name=f"irrig-stop-z{zone_num}",
    ).start()


# ── Surveillance capteurs silencieux ─────────────────────────────────────────

def _stale_sensor_watcher(interval_seconds: int = 60):
    if not cfg.SENSOR_STALE_SECONDS:
        logger.info(
            "Surveillance capteurs silencieux désactivée "
            "(SENSOR_STALE_SECONDS=None)"
        )
        return

    logger.info(
        "Thread surveillance capteurs démarré "
        "(vérif. toutes %ds, seuil=%ds)",
        interval_seconds,
        cfg.SENSOR_STALE_SECONDS,
    )

    while True:
        time.sleep(interval_seconds)
        try:
            users = db.collection("users").get()
            for user_doc in users:
                uid   = user_doc.id
                zones = (
                    db.collection("users")
                    .document(uid)
                    .collection("zones")
                    .get()
                )
                for zone_doc in zones:
                    zone_num = zone_doc.id.replace("zone", "")
                    _check_stale_sensors_in_zone(uid, zone_num)
        except Exception as exc:
            logger.error("Erreur thread surveillance : %s", exc)


def _check_stale_sensors_in_zone(uid: str, zone_num: str) -> None:
    now = time.time()
    try:
        sensor_docs = (
            db.collection("users")
            .document(uid)
            .collection("zones")
            .document(f"zone{zone_num}")
            .collection("sensors")
            .where("active", "==", True)
            .get()
        )

        for doc in sensor_docs:
            data = doc.to_dict()
            if not data:
                continue

            ts = data.get("timestamp")
            if ts is None:
                continue

            if hasattr(ts, "timestamp"):
                last_seen = ts.timestamp()
            elif hasattr(ts, "_seconds"):
                last_seen = ts._seconds
            else:
                continue

            age = now - last_seen

            if age > cfg.SENSOR_STALE_SECONDS:
                sensor_id = doc.id
                logger.warning(
                    "Zone %s | Capteur %s silencieux depuis %.0fs → désactivé",
                    zone_num, sensor_id, age,
                )
                deactivate_sensor(db, uid, zone_num, sensor_id)
                send_sensor_failure_alert(db, uid, zone_num, sensor_id) # Notification PANNE
                mqtt_client.publish(
                    f"robocare/{uid}/zone/{zone_num}/sensor/{sensor_id}/status",
                    "offline",
                    qos=1,
                    retain=True,
                )
            elif age > 600: # 10 minutes
                # État intermédiaire : Sommeil (Deep Sleep)
                mqtt_client.publish(
                    f"robocare/{uid}/zone/{zone_num}/sensor/{sensor_id}/status",
                    "sleeping",
                    qos=1,
                    retain=True,
                )

    except Exception as exc:
        logger.error(
            "Erreur vérification capteurs zone %s : %s", zone_num, exc,
        )


def start_stale_watcher(interval_seconds: int = 60) -> None:
    t = threading.Thread(
        target=_stale_sensor_watcher,
        args=(interval_seconds,),
        daemon=True,
        name="stale-sensor-watcher",
    )
    t.start()


def _pump_safety_watcher(interval_seconds: int = 60):
    """Vérifie si une pompe tourne en manuel depuis trop longtemps (ex: 15 min)."""
    MAX_PUMP_MINUTES = 15
    logger.info("Thread surveillance pompe démarré (seuil=%d min)", MAX_PUMP_MINUTES)
    
    while True:
        time.sleep(interval_seconds)
        try:
            users = db.collection("users").get()
            for user_doc in users:
                uid = user_doc.id
                zones = db.collection("users").document(uid).collection("zones").where("enabled", "==", True).get()
                for zone_doc in zones:
                    data = zone_doc.to_dict()
                    start_time = data.get("pump_start_time")
                    if not start_time:
                        continue
                    
                    # Calcul de la durée
                    now = time.time()
                    if hasattr(start_time, "timestamp"):
                        start_ts = start_time.timestamp()
                    elif hasattr(start_time, "_seconds"):
                        start_ts = start_time._seconds
                    else:
                        continue
                        
                    duration_min = (now - start_ts) / 60
                    if duration_min > MAX_PUMP_MINUTES:
                        zone_num = zone_doc.id.replace("zone", "")
                        # On envoie une notification toutes les 15 minutes d'oubli
                        if int(duration_min) % MAX_PUMP_MINUTES == 0:
                            send_pump_forgotten_alert(db, uid, zone_num, int(duration_min))
                            
        except Exception as exc:
            logger.error("Erreur thread surveillance pompe : %s", exc)


def start_pump_watcher(interval_seconds: int = 60) -> None:
    t = threading.Thread(
        target=_pump_safety_watcher,
        args=(interval_seconds,),
        daemon=True,
        name="pump-safety-watcher",
    )
    t.start()


# ── Lecture seuils Firebase ───────────────────────────────────────────────────

def _get_zone_config(uid: str, zone_num: str) -> dict:
    defaults = {
        "minHumidity": cfg.HUMIDITY_ALERT,
        "maxHumidity": 70.0,
        "duration":    15,
        "zone_name":   f"Zone {zone_num}",
        "plant_type":  None,
        "crop_stage":  None,
    }

    try:
        zone_ref = (
            db.collection("users")
            .document(uid)
            .collection("zones")
            .document(f"zone{zone_num}")
        )

        zone_doc = zone_ref.get()
        if zone_doc.exists:
            zone_data = zone_doc.to_dict() or {}
            defaults.update({
                "zone_name":  zone_data.get("zone_name",
                              zone_data.get("name", defaults["zone_name"])),
                "plant_type": zone_data.get("plant_type"),
                "crop_stage": zone_data.get("crop_stage"),
            })

        plant_current_doc = (
            zone_ref.collection("plante").document("current").get()
        )
        if plant_current_doc.exists:
            plant_data = plant_current_doc.to_dict() or {}
            plant_thresholds = plant_data.get("thresholds", {})
            humidity_limits = (
                plant_thresholds.get("humidity", {})
                if isinstance(plant_thresholds, dict)
                else {}
            )
            if isinstance(humidity_limits, dict):
                defaults.update({
                    "minHumidity": float(
                        humidity_limits.get("min", defaults["minHumidity"])
                    ),
                    "maxHumidity": float(
                        humidity_limits.get("max", defaults["maxHumidity"])
                    ),
                })
                logger.info(
                    "Seuils humidité zone %s lus depuis plante/current : min=%.1f max=%.1f",
                    zone_num,
                    defaults["minHumidity"],
                    defaults["maxHumidity"],
                )

        thresholds_doc = zone_ref.collection("config").document("thresholds").get()
        if thresholds_doc.exists:
            t = thresholds_doc.to_dict() or {}
            defaults.update({
                "minHumidity": float(t.get("minHumidity", defaults["minHumidity"])),
                "maxHumidity": float(t.get("maxHumidity", defaults["maxHumidity"])),
                "duration":    int(t.get("duration",      defaults["duration"])),
            })
            logger.info(
                "Seuils zone %s : min=%.1f max=%.1f durée=%d min",
                zone_num,
                defaults["minHumidity"],
                defaults["maxHumidity"],
                defaults["duration"],
            )
        else:
            logger.warning(
                "Aucune config thresholds pour UID=%s zone=%s — valeurs défaut",
                uid, zone_num,
            )

    except Exception as exc:
        logger.error("Erreur lecture config zone %s : %s", zone_num, exc)

    return defaults


def _to_float(value, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_plant_thresholds(uid: str, zone_num: str) -> dict:
    """
    Lit users/{uid}/zones/zone{zone_num}/plante/current
    ET users/{uid}/zones/zone{zone_num}/config/thresholds pour fusionner les seuils.
    """
    thresholds = {}
    try:
        zone_ref = (
            db.collection("users")
            .document(uid)
            .collection("zones")
            .document(f"zone{zone_num}")
        )

        # 1. Seuils spécifiques à la plante (recherche agronomique / JSON structuré)
        current_snap = zone_ref.collection("plante").document("current").get()
        if current_snap.exists:
            data = current_snap.to_dict() or {}
            raw_thresholds = data.get("thresholds", {})
            if isinstance(raw_thresholds, dict):
                for field in ("humidity", "temperature", "ec", "ph", "n", "p", "k"):
                    limits = raw_thresholds.get(field, {})
                    if isinstance(limits, dict):
                        min_v = _to_float(limits.get("min"))
                        max_v = _to_float(limits.get("max"))
                        if min_v is not None and max_v is not None:
                            thresholds[field] = {"min": min_v, "max": max_v}

        # 2. Seuils configurés manuellement par l'admin (config/thresholds / Format Plat)
        # Ils ont priorité sur les seuils agronomiques par défaut.
        config_snap = zone_ref.collection("config").document("thresholds").get()
        if config_snap.exists:
            t = config_snap.to_dict() or {}
            mapping = {
                "humidity":    ("minHumidity", "maxHumidity"),
                "temperature": ("minTemp",     "maxTemp"),
                "ph":          ("minPh",       "maxPh"),
                "ec":          ("minEc",       "maxEc"),
                "n":           ("minN",        "maxN"),
                "p":           ("minP",        "maxP"),
                "k":           ("minK",        "maxK"),
            }
            
            for field, (min_key, max_key) in mapping.items():
                admin_min = _to_float(t.get(min_key))
                admin_max = _to_float(t.get(max_key))
                
                if admin_min is not None and admin_max is not None:
                    thresholds[field] = {"min": admin_min, "max": admin_max}
                    logger.info("Seuils admin pour %s zone %s : [%.1f, %.1f]", field, zone_num, admin_min, admin_max)

    except Exception as exc:
        logger.error(
            "Erreur lecture seuils fusionnés zone %s : %s", zone_num, exc,
        )

    return thresholds


def _get_plant_type(uid: str, zone_num: str) -> str | None:
    """Retourne le type de plante depuis plante/current si disponible."""
    try:
        snap = (
            db.collection("users")
            .document(uid)
            .collection("zones")
            .document(f"zone{zone_num}")
            .collection("plante")
            .document("current")
            .get()
        )
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        value = data.get("plant_type", data.get("plantType"))
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception as exc:
        logger.warning("Erreur lecture plant_type zone %s : %s", zone_num, exc)
    return None


def _check_thresholds_and_notify(
    uid: str,
    zone_num: str,
    humidity: float,
    temperature: float,
    ph: float,
    ec: float,
    n: float,
    p: float,
    k: float,
) -> None:
    thresholds = _get_plant_thresholds(uid, zone_num)
    if not thresholds:
        return

    issues = []
    values = {
        "humidity": humidity,
        "temperature": temperature,
        "ph": ph,
        "ec": ec,
        "n": n,
        "p": p,
        "k": k,
    }

    for field, value in values.items():
        limits = thresholds.get(field)
        if not limits:
            continue

        min_v = limits["min"]
        max_v = limits["max"]
        if value < min_v or value > max_v:
            logger.warning(
                "Zone %s | %s=%.1f hors seuil [%.1f, %.1f]",
                zone_num, field, value, min_v, max_v,
            )
            issues.append({
                "field": field,
                "value": round(value, 2),
                "min": round(min_v, 2),
                "max": round(max_v, 2),
            })

    if issues:
        send_thresholds_summary_alert(
            db=db,
            uid=uid,
            zone_num=zone_num,
            plant_type=_get_plant_type(uid, zone_num),
            issues=issues,
        )


# ── Contrôle automatique irrigation ──────────────────────────────────────────

def _control_valve_auto(uid: str, zone_num: str, humidity: float) -> None:
    thresholds   = _get_zone_config(uid, zone_num)
    min_humidity = thresholds["minHumidity"]
    max_humidity = thresholds["maxHumidity"]
    duration     = thresholds["duration"]

    if humidity < min_humidity:
        logger.info(
            "Zone %s | H moy. %.1f%% < min %.1f%% → Irrigation ON",
            zone_num, humidity, min_humidity,
        )
        _publish_irrigation_start(uid, zone_num)

        # ✅ SUPERVISION — Mise à jour statut + notification démarrage AUTO
        (
            db.collection("users")
            .document(uid)
            .collection("zones")
            .document(f"zone{zone_num}")
            .update({
                "enabled": True,
                "irrigation_status": "STARTED",
                "irrigation_mode": "auto",
                "irrigation_started_at": firestore.SERVER_TIMESTAMP,
            })
        )
        from services.notification_service import send_auto_irrigation_started_alert
        send_auto_irrigation_started_alert(db, uid, zone_num, round(humidity, 1), round(min_humidity, 1))

        def auto_close():
            time.sleep(duration * 60)
            try:
                zone_doc = (
                    db.collection("users")
                    .document(uid)
                    .collection("zones")
                    .document(f"zone{zone_num}")
                    .get()
                )
                if not zone_doc.exists:
                    return

                current_h = zone_doc.to_dict().get("humidity", 0)
                if current_h >= max_humidity:
                    logger.info(
                        "Zone %s | H moy. %.1f%% ≥ max %.1f%% → OFF auto",
                        zone_num, current_h, max_humidity,
                    )
                    _publish_irrigation_stop(uid, zone_num)
                    # ✅ SUPERVISION — Mise à jour statut + notification arrêt AUTO (timer)
                    (
                        db.collection("users")
                        .document(uid)
                        .collection("zones")
                        .document(f"zone{zone_num}")
                        .update({
                            "enabled": False,
                            "irrigation_status": "FINISHED",
                            "irrigation_mode": "auto",
                        })
                    )
                    from services.notification_service import send_irrigation_finished_alert
                    send_irrigation_finished_alert(db, uid, zone_num)
            except Exception as exc:
                logger.error(
                    "Erreur fermeture auto zone %s : %s", zone_num, exc,
                )

        threading.Thread(
            target=auto_close,
            daemon=True,
            name=f"auto-close-z{zone_num}",
        ).start()

    elif humidity >= max_humidity:
        logger.info(
            "Zone %s | H moy. %.1f%% ≥ max %.1f%% → Irrigation OFF",
            zone_num, humidity, max_humidity,
        )
        _publish_irrigation_stop(uid, zone_num)
        # ✅ SUPERVISION — Mise à jour statut arrêt AUTO (humidité suffisante)
        (
            db.collection("users")
            .document(uid)
            .collection("zones")
            .document(f"zone{zone_num}")
            .update({
                "enabled": False,
                "irrigation_status": "FINISHED",
                "irrigation_mode": "auto",
            })
        )


# ── Callbacks MQTT ───────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, reason_code, properties):
    logger.info("Connecté au broker MQTT (code %s)", reason_code)

    # Topics data capteurs
    client.subscribe(cfg.MQTT_TOPIC_DATA)
    client.subscribe("robocare/discovery")

    # ✅ AJOUT v2.6 — subscribe valve/pump/status pour logs et supervision
    client.subscribe("robocare/+/valve/control/+")
    client.subscribe("robocare/+/pump/control")
    client.subscribe("robocare/+/zone/+/irrigation/status")

    logger.info("Subscriptions actives :")
    logger.info("  → %s", cfg.MQTT_TOPIC_DATA)
    logger.info("  → robocare/discovery")
    logger.info("  → robocare/+/valve/control/+")
    logger.info("  → robocare/+/pump/control")
    logger.info("  → robocare/+/zone/+/irrigation/status")


def on_message(client, userdata, msg):
    try:
        # ── Discovery ──
        if msg.topic == "robocare/discovery":
            _handle_discovery(msg)
            return

        parts = msg.topic.split("/")

        # ✅ CORRIGÉ v2.6 — topics valve/pump gérés par ESP32
        # Le bridge les reçoit juste pour les logger
        if len(parts) >= 5 and parts[2] == "valve" and parts[3] == "control":
            payload = msg.payload.decode("utf-8")
            logger.info(
                "Commande VANNE reçue → zone=%s payload=%s "
                "(transmise directement à ESP32)",
                parts[4], payload,
            )
            return

        if len(parts) >= 4 and parts[2] == "pump" and parts[3] == "control":
            payload = msg.payload.decode("utf-8")
            logger.info(
                "Commande POMPE reçue → payload=%s "
                "(transmise directement à ESP32)",
                payload,
            )
            return

        # ✅ SUPERVISION — robocare/{uid}/zone/{zone}/irrigation/status
        if len(parts) >= 6 and parts[4] == "irrigation" and parts[5] == "status":
            uid = parts[1]
            zone_num = parts[3]
            status = msg.payload.decode("utf-8")
            logger.info("Statut irrigation Zone %s : %s", zone_num, status)
            
            # Mise à jour Firestore pour supervision Flutter
            db.collection("users").document(uid).collection("zones").document(f"zone{zone_num}").update({
                "irrigation_status": status,
                "enabled": (status == "STARTED")
            })

            if status == "FINISHED":
                send_irrigation_finished_alert(db, uid, zone_num)
            return

        # ── Topics data — minimum 7 parts ──
        if len(parts) < 7:
            logger.warning(
                "Topic inattendu (attendu ≥ 7 parts, reçu %d) : %s",
                len(parts), msg.topic,
            )
            return

        uid       = parts[1]
        zone_num  = parts[3]
        sensor_id = parts[5]

        payload = json.loads(msg.payload.decode("utf-8"))

        logger.info(
            "← MQTT Zone %s | Capteur %s | UID: %s",
            zone_num, sensor_id, uid,
        )

        result = update_sensor_data(db, uid, zone_num, payload, sensor_id)

        # ✅ Remise à l'état ONLINE dès qu'on reçoit une donnée
        mqtt_client.publish(
            f"robocare/{uid}/zone/{zone_num}/sensor/{sensor_id}/status",
            "online",
            qos=1,
            retain=True,
        )

        if result[0] is None:
            return

        humidity, temperature, ph, ec, n, p, k = result

        meta = payload.get("meta", {})
        if isinstance(meta, dict) and meta:
            try:
                (
                    db.collection("users")
                    .document(uid)
                    .collection("zones")
                    .document(f"zone{zone_num}")
                    .collection("signal")
                    .add({
                        "sensor_id": sensor_id,
                        "node_id":   meta.get("node_id"),
                        "mac":       meta.get("mac"),
                        "rssi":      meta.get("rssi"),
                        "snr":       meta.get("snr"),
                        "timestamp": firestore.SERVER_TIMESTAMP,
                    })
                )
                logger.info(
                    "Zone %s | Capteur %s | MAC: %s | RSSI: %s | SNR: %s",
                    zone_num, sensor_id,
                    meta.get("mac"), meta.get("rssi"), meta.get("snr"),
                )
            except Exception as exc:
                logger.warning(
                    "Erreur sauvegarde métadonnées zone %s : %s",
                    zone_num, exc,
                )

        # On récupère la config complète pour avoir les seuils à jour
        zone_cfg = _get_zone_config(uid, zone_num)
        min_humidity_threshold = zone_cfg.get("minHumidity", cfg.HUMIDITY_ALERT)

        _control_valve_auto(uid, zone_num, humidity)
        _check_thresholds_and_notify(
            uid=uid,
            zone_num=zone_num,
            humidity=humidity,
            temperature=temperature,
            ph=ph,
            ec=ec,
            n=n,
            p=p,
            k=k,
        )

        # Utilisation du seuil configuré par l'admin au lieu du hardcodé
        if humidity < min_humidity_threshold:
            send_critical_alert(db, uid, zone_num, humidity)

        if predict_stress_risk(db, uid, zone_num, humidity):
            logger.warning("Stress hydrique prédit — Zone %s", zone_num)

        ml_result = predict_irrigation_combined(
            db, uid, zone_num,
            humidity, temperature, ec, n, p, k,
        )

        (
            db.collection("users")
            .document(uid)
            .collection("zones")
            .document(f"zone{zone_num}")
            .collection("history")
            .add(ml_result.to_firestore_dict())
        )

        if ml_result.is_dangerous:
            send_irrigation_prediction(db, uid, zone_num, ml_result)

    except Exception as exc:
        logger.exception("Erreur on_message : %s", exc)


# ── Gestion discovery ────────────────────────────────────────────────────────

def _handle_discovery(msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        mac  = data.get("mac", "").strip()

        if not mac:
            logger.warning("Discovery : MAC manquante dans payload")
            return

        logger.info("Discovery reçu : MAC=%s", mac)

        doc_ref  = db.collection("pending_devices").document(mac)
        doc_snap = doc_ref.get()

        if doc_snap.exists:
            doc_data = doc_snap.to_dict()
            status   = doc_data.get("status", "waiting")
            uid      = doc_data.get("uid", "")

            if status == "claimed" and uid:
                config_topic = f"robocare/config/{mac}"
                mqtt_client.publish(config_topic, uid, qos=1, retain=True)
                logger.info("Device connu → UID envoyé : %s → %s", mac, uid)
                return

        doc_ref.set(
            {
                "mac":       mac,
                "status":    "waiting",
                "timestamp": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        logger.info("Nouveau device enregistré en attente : MAC=%s", mac)

    except Exception as exc:
        logger.error("Erreur _handle_discovery : %s", exc)


# ── Listeners Firestore ──────────────────────────────────────────────────────

def _on_variateur_snapshot(uid, doc_snapshot, changes, read_time):
    for doc in doc_snapshot:
        if not doc.exists:
            continue

        data  = doc.to_dict()
        is_on = bool(data.get("isOn", False))
        freq  = int(data.get("frequency", 0))

        mqtt_client.publish(
            f"robocare/{uid}/pump/control",
            "1" if is_on else "0",
            qos=1,
        )

        if freq > 0:
            mqtt_client.publish(
                f"robocare/{uid}/pump/frequency",
                str(freq),
                qos=1,
            )


def _on_zones_snapshot(uid, col_snapshot, changes, read_time):
    for change in changes:
        if change.type.name in ("ADDED", "MODIFIED"):
            doc      = change.document
            data     = doc.to_dict() or {}
            enabled  = bool(data.get("enabled", False))
            zone_num = doc.id.replace("zone", "")

            # ✅ NOUVEAU — Irrigation temporisée (Timed)
            timed_seconds = data.get("timed_order_seconds", 0)
            if timed_seconds > 0:
                logger.info("App mobile → Zone %s TIMED START (%d s)", zone_num, timed_seconds)
                mqtt_client.publish(
                    f"robocare/{uid}/irrigation/timed/{zone_num}",
                    str(timed_seconds),
                    qos=1
                )
                # Reset order and force ENABLED + STARTED status for UI feedback
                doc.reference.update({
                    "timed_order_seconds": 0,
                    "enabled": True,
                    "irrigation_status": "STARTED",
                    "pump_start_time": firestore.SERVER_TIMESTAMP # Pour la sécurité
                })
                return 

            if enabled:
                logger.info("App mobile → Zone %s ON (manuel)", zone_num)
                # On enregistre l'heure de démarrage si pas déjà fait
                if "pump_start_time" not in data:
                    doc.reference.update({"pump_start_time": firestore.SERVER_TIMESTAMP})
                _publish_irrigation_start(uid, zone_num)
            else:
                logger.info("App mobile → Zone %s OFF (manuel)", zone_num)
                doc.reference.update({"pump_start_time": firestore.DELETE_FIELD})
                _publish_irrigation_stop(uid, zone_num)

        elif change.type.name == "REMOVED":
            zone_num = change.document.id.replace("zone", "")
            logger.info(
                "Zone %s supprimée — arrêt irrigation sécurisé", zone_num,
            )
            _publish_irrigation_stop(uid, zone_num)


def on_msg_snapshot(col_snap, changes, read_time):
    for change in changes:
        if change.type.name != "ADDED":
            continue

        doc  = change.document
        data = doc.to_dict()

        if doc.id in _processed_msg_ids:
            continue

        path_parts = doc.reference.path.split("/")

        if len(path_parts) == 4 and path_parts[2] == "messages":
            _processed_msg_ids.add(doc.id)
            continue
        elif len(path_parts) == 6 and path_parts[4] == "messages":
            uid     = path_parts[1]
            chat_id = path_parts[3]
        else:
            logger.warning(
                "Chemin inconnu ignoré : %s", doc.reference.path,
            )
            _processed_msg_ids.add(doc.id)
            continue

        msg_time = data.get("timestamp")
        if msg_time is not None:
            ts = msg_time
            if hasattr(ts, "timestamp"):
                ts = ts.timestamp()
            elif hasattr(ts, "_seconds"):
                ts = ts._seconds
            try:
                if float(ts) < _watcher_start_time - 30:
                    _processed_msg_ids.add(doc.id)
                    continue
            except Exception:
                pass

        if data.get("sender") == "user":
            _processed_msg_ids.add(doc.id)
            handle_chatbot_async(db, uid, chat_id, data.get("text", ""))


# ── Démarrage watchers ────────────────────────────────────────────────────────

def start_command_listener(uid: str) -> None:
    if uid in _command_watchers:
        return

    ref = (
        db.collection("users")
        .document(uid)
        .collection("commands")
        .document("variateur")
    )

    _command_watchers[uid] = ref.on_snapshot(
        lambda ds, c, rt: _on_variateur_snapshot(uid, ds, c, rt)
    )

    z_ref = (
        db.collection("users")
        .document(uid)
        .collection("zones")
    )

    _zone_watchers[uid] = z_ref.on_snapshot(
        lambda cs, c, rt: _on_zones_snapshot(uid, cs, c, rt)
    )

    logger.info("Listeners démarrés pour UID : %s", uid)


def start_all_watchers() -> None:
    users = db.collection("users").get()

    for u in users:
        start_command_listener(u.id)

    db.collection("users").on_snapshot(
        lambda s, c, r: [
            start_command_listener(ch.document.id)
            for ch in c
            if ch.type.name == "ADDED"
        ]
    )

    db.collection_group("messages").on_snapshot(on_msg_snapshot)

    def on_pending(snapshots, changes, read_time):
        for change in changes:
            if change.type.name not in ("ADDED", "MODIFIED"):
                continue

            d      = change.document.to_dict()
            status = d.get("status", "")
            mac    = d.get("mac", "")
            uid    = d.get("uid", "")

            if status == "claimed" and mac and uid:
                config_topic = f"robocare/config/{mac}"
                mqtt_client.publish(
                    config_topic, uid, qos=1, retain=True,
                )
                logger.info(
                    "Device claimed → UID publié : %s → topic %s",
                    uid, config_topic,
                )
                start_command_listener(uid)

    db.collection("pending_devices").on_snapshot(on_pending)

    logger.info("Tous les watchers Firestore démarrés")