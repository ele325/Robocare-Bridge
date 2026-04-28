# -*- coding: utf-8 -*-
"""
mqtt_handler.py — Callbacks MQTT RoboCare v2.5
Corrections v2.5 :
  - _control_valve_auto() : vanne d'abord (200ms) puis pompe
  - _on_zones_snapshot()  : commande vanne + pompe dans le bon ordre
  - auto_close()          : arrêt pompe avant fermeture vanne
  - Commande manuelle app mobile : ordre correct via Firestore watcher
"""

import json
import time
import threading

from firebase_admin import firestore

import config as cfg
from services.firebase_service import update_sensor_data, deactivate_sensor
from services.notification_service import send_critical_alert, send_irrigation_prediction
from services.chatbot_service import handle_chatbot_async
from ml.predictor import predict_stress_risk, predict_irrigation_combined
from utils.logger import get_logger

logger = get_logger("robocare.mqtt")


# ── État interne ─────────────────────────────────────────────────────────────

_command_watchers:    dict  = {}
_zone_watchers:       dict  = {}
_processed_msg_ids:   set   = set()
_watcher_start_time:  float = time.time()

# Délai mécanique ouverture/fermeture électrovanne (secondes)
VALVE_DELAY_S = 0.3   # 300 ms — temps ouverture mécanique vanne

mqtt_client = None
db          = None


def init(client, firestore_db):
    global mqtt_client, db
    mqtt_client = client
    db          = firestore_db


# ── Helpers irrigation ────────────────────────────────────────────────────────

def _publish_irrigation_start(uid: str, zone_num: str) -> None:
    """
    Démarre l'irrigation d'une zone :
      1. Publie vanne ON
      2. Attend VALVE_DELAY_S (ouverture mécanique)
      3. Publie pompe ON
    Exécuté dans un thread séparé pour ne pas bloquer le callback MQTT.
    """
    def _do_start():
        logger.info("Zone %s → Irrigation START (vanne puis pompe)", zone_num)

        # 1. Ouvrir la vanne
        mqtt_client.publish(
            f"robocare/{uid}/valve/control/{zone_num}",
            "1",
            qos=1,
        )

        # 2. Attendre ouverture mécanique
        time.sleep(VALVE_DELAY_S)

        # 3. Démarrer la pompe
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

    threading.Thread(target=_do_start, daemon=True,
                     name=f"irrig-start-z{zone_num}").start()


def _publish_irrigation_stop(uid: str, zone_num: str) -> None:
    """
    Arrête l'irrigation d'une zone :
      1. Publie pompe OFF
      2. Attend VALVE_DELAY_S (arrêt complet pompe)
      3. Publie vanne OFF
    Exécuté dans un thread séparé.
    """
    def _do_stop():
        logger.info("Zone %s → Irrigation STOP (pompe puis vanne)", zone_num)

        # 1. Arrêter la pompe d'abord (évite surpression)
        mqtt_client.publish(
            f"robocare/{uid}/pump/control",
            "0",
            qos=1,
        )

        # 2. Attendre arrêt complet
        time.sleep(VALVE_DELAY_S)

        # 3. Fermer la vanne
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

    threading.Thread(target=_do_stop, daemon=True,
                     name=f"irrig-stop-z{zone_num}").start()


# ── Surveillance des capteurs silencieux ─────────────────────────────────────

def _stale_sensor_watcher(interval_seconds: int = 60):
    """
    Thread daemon — vérifie toutes les interval_seconds les capteurs actifs.
    Si un capteur dépasse SENSOR_STALE_SECONDS → désactivé.
    """
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
    """Désactive les capteurs silencieux d'une zone."""
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

                mqtt_client.publish(
                    f"robocare/{uid}/zone/{zone_num}/sensor/{sensor_id}/status",
                    "offline",
                    qos=1,
                    retain=True,
                )

    except Exception as exc:
        logger.error(
            "Erreur vérification capteurs zone %s : %s", zone_num, exc,
        )


def start_stale_watcher(interval_seconds: int = 60) -> None:
    """Démarre le thread de surveillance des capteurs silencieux."""
    t = threading.Thread(
        target=_stale_sensor_watcher,
        args=(interval_seconds,),
        daemon=True,
        name="stale-sensor-watcher",
    )
    t.start()


# ── Lecture des seuils depuis Firebase ───────────────────────────────────────

def _get_zone_config(uid: str, zone_num: str) -> dict:
    """Lit la configuration d'irrigation d'une zone depuis Firestore."""

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
                "zone_name": zone_data.get(
                    "zone_name",
                    zone_data.get("name", defaults["zone_name"]),
                ),
                "plant_type": zone_data.get("plant_type"),
                "crop_stage": zone_data.get("crop_stage"),
            })

        thresholds_doc = (
            zone_ref.collection("config").document("thresholds").get()
        )

        if thresholds_doc.exists:
            t = thresholds_doc.to_dict() or {}
            defaults.update({
                "minHumidity": float(t.get("minHumidity", defaults["minHumidity"])),
                "maxHumidity": float(t.get("maxHumidity", defaults["maxHumidity"])),
                "duration":    int(t.get("duration",    defaults["duration"])),
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


# ── Contrôle automatique de l'irrigation ─────────────────────────────────────

def _control_valve_auto(uid: str, zone_num: str, humidity: float) -> None:
    """
    CORRECTION v2.5 — Active/désactive irrigation avec le bon ordre :
      ON  : vanne d'abord → délai → pompe
      OFF : pompe d'abord → délai → vanne

    Si humidity < minHumidity → irrigation ON
    Si humidity >= maxHumidity → irrigation OFF
    """

    thresholds   = _get_zone_config(uid, zone_num)
    min_humidity = thresholds["minHumidity"]
    max_humidity = thresholds["maxHumidity"]
    duration     = thresholds["duration"]

    if humidity < min_humidity:
        logger.info(
            "Zone %s | H moy. %.1f%% < min %.1f%% → Irrigation ON",
            zone_num, humidity, min_humidity,
        )

        # Démarrer : vanne → délai → pompe
        _publish_irrigation_start(uid, zone_num)

        # Mise à jour Firestore
        (
            db.collection("users")
            .document(uid)
            .collection("zones")
            .document(f"zone{zone_num}")
            .update({"enabled": True})
        )

        # Fermeture automatique après durée configurée
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
                        "Zone %s | H moy. %.1f%% ≥ max %.1f%% → Irrigation OFF auto",
                        zone_num, current_h, max_humidity,
                    )

                    # Arrêter : pompe → délai → vanne
                    _publish_irrigation_stop(uid, zone_num)

                    (
                        db.collection("users")
                        .document(uid)
                        .collection("zones")
                        .document(f"zone{zone_num}")
                        .update({"enabled": False})
                    )

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

        # Arrêter : pompe → délai → vanne
        _publish_irrigation_stop(uid, zone_num)

        (
            db.collection("users")
            .document(uid)
            .collection("zones")
            .document(f"zone{zone_num}")
            .update({"enabled": False})
        )


# ── Callbacks MQTT ───────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, reason_code, properties):
    logger.info("Connecté au broker MQTT (code %s)", reason_code)

    client.subscribe(cfg.MQTT_TOPIC_DATA)
    client.subscribe("robocare/discovery")

    logger.info("Subscriptions actives :")
    logger.info("  → %s", cfg.MQTT_TOPIC_DATA)
    logger.info("  → robocare/discovery")


def on_message(client, userdata, msg):
    """
    Callback principal MQTT.
    Topic : robocare/{uid}/zone/{zone_num}/sensor/{sensor_id}/data
    """

    try:
        if msg.topic == "robocare/discovery":
            _handle_discovery(msg)
            return

        parts = msg.topic.split("/")

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

        # Mise à jour Firestore
        result = update_sensor_data(db, uid, zone_num, payload, sensor_id)

        if result[0] is None:
            return

        humidity, temperature, ph, ec, n, p, k = result

        # Métadonnées réseau
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
                    "Zone %s | Capteur %s | MAC: %s | RSSI: %s dBm | SNR: %s",
                    zone_num, sensor_id,
                    meta.get("mac"), meta.get("rssi"), meta.get("snr"),
                )

            except Exception as exc:
                logger.warning(
                    "Erreur sauvegarde métadonnées zone %s : %s", zone_num, exc,
                )

        # ── CONTRÔLE AUTOMATIQUE IRRIGATION ──
        _control_valve_auto(uid, zone_num, humidity)

        # Alerte critique
        if humidity < cfg.HUMIDITY_ALERT:
            send_critical_alert(db, uid, zone_num, humidity)

        # ML stress hydrique
        if predict_stress_risk(db, uid, zone_num, humidity):
            logger.warning("Stress hydrique prédit — Zone %s", zone_num)

        # ML combinée
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

                logger.info(
                    "Device connu → UID envoyé : %s → %s", mac, uid,
                )
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
    """Watcher pompe manuelle (variateur)."""
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
    """
    CORRECTION v2.5 — Watcher Firestore zones.
    Quand l'utilisateur change enabled dans une zone (via app mobile) :
      ON  → vanne d'abord → délai → pompe
      OFF → pompe d'abord → délai → vanne
    """

    for change in changes:
        if change.type.name in ("ADDED", "MODIFIED"):
            doc      = change.document
            data     = doc.to_dict() or {}
            enabled  = bool(data.get("enabled", False))
            zone_num = doc.id.replace("zone", "")

            if enabled:
                logger.info(
                    "App mobile → Zone %s ON (vanne puis pompe)", zone_num,
                )
                # Commande manuelle : vanne → délai → pompe
                _publish_irrigation_start(uid, zone_num)

            else:
                logger.info(
                    "App mobile → Zone %s OFF (pompe puis vanne)", zone_num,
                )
                # Commande manuelle : pompe → délai → vanne
                _publish_irrigation_stop(uid, zone_num)

        elif change.type.name == "REMOVED":
            zone_num = change.document.id.replace("zone", "")

            logger.info(
                "Zone %s supprimée — arrêt irrigation sécurisé", zone_num,
            )

            # Sécurité : arrêt propre
            _publish_irrigation_stop(uid, zone_num)


def on_msg_snapshot(col_snap, changes, read_time):
    """Watcher messages chatbot."""
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
            logger.warning("Chemin inconnu ignoré : %s", doc.reference.path)
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


# ── Démarrage des watchers ────────────────────────────────────────────────────

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