# -*- coding: utf-8 -*-
"""
mqtt_handler.py — Callbacks MQTT RoboCare v2.3
Multi-capteurs par zone avec adresses fixes.

Topic attendu : robocare/{uid}/zone/{zone_num}/sensor/{sensor_id}/data
Index parts   :     [0]   [1]  [2]    [3]        [4]      [5]      [6]

Principe de remplacement hardware :
  Si le capteur à l'adresse "11" tombe en panne, l'agriculteur configure
  le nouveau hardware avec l'adresse "11". Le serveur ne voit aucune
  différence — le topic et le sensor_id sont identiques, les données
  continuent d'arriver normalement.
"""

import json
import time
import threading

from firebase_admin import firestore

import config as cfg
from services.firebase_service     import update_sensor_data, deactivate_sensor
from services.notification_service import send_critical_alert, send_irrigation_prediction
from services.chatbot_service      import handle_chatbot_async
from ml.predictor                  import predict_stress_risk, predict_irrigation_combined
from utils.logger                  import get_logger

logger = get_logger("robocare.mqtt")

# ── État interne ─────────────────────────────────────────────────────────────
_command_watchers   : dict  = {}
_zone_watchers      : dict  = {}
_processed_msg_ids  : set   = set()
_watcher_start_time : float = time.time()

mqtt_client = None
db          = None


def init(client, firestore_db):
    global mqtt_client, db
    mqtt_client = client
    db          = firestore_db


# ── Surveillance des capteurs silencieux ──────────────────────────────────────

def _stale_sensor_watcher(interval_seconds: int = 60):
    """
    Thread daemon — tourne en arrière-plan toutes les `interval_seconds`.

    Pour chaque capteur dont le dernier message dépasse SENSOR_STALE_SECONDS :
      → publie "offline" sur le topic de statut MQTT
      → appelle deactivate_sensor() pour l'exclure des moyennes

    Dès que le hardware de remplacement (même adresse) envoie son premier
    message, update_sensor_data() remet active=True automatiquement.
    Aucune intervention supplémentaire n'est nécessaire.
    """
    if not cfg.SENSOR_STALE_SECONDS:
        logger.info("Surveillance capteurs silencieux désactivée (SENSOR_STALE_SECONDS=None)")
        return

    logger.info(
        "Thread surveillance capteurs démarré (vérif. toutes %ds, seuil=%ds)",
        interval_seconds, cfg.SENSOR_STALE_SECONDS,
    )

    while True:
        time.sleep(interval_seconds)
        try:
            users = db.collection("users").get()
            for user_doc in users:
                uid   = user_doc.id
                zones = (
                    db.collection("users").document(uid)
                      .collection("zones").get()
                )
                for zone_doc in zones:
                    zone_num = zone_doc.id.replace("zone", "")
                    _check_stale_sensors_in_zone(uid, zone_num)
        except Exception as exc:
            logger.error("Erreur thread surveillance : %s", exc)


def _check_stale_sensors_in_zone(uid: str, zone_num: str) -> None:
    """
    Vérifie tous les capteurs d'une zone et désactive ceux qui sont silencieux.
    """
    now = time.time()
    try:
        sensor_docs = (
            db.collection("users").document(uid)
              .collection("zones").document(f"zone{zone_num}")
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
                # Marquer inactif → exclu des moyennes
                deactivate_sensor(db, uid, zone_num, sensor_id)
                # Notifier via MQTT
                mqtt_client.publish(
                    f"robocare/{uid}/zone/{zone_num}/sensor/{sensor_id}/status",
                    "offline",
                    qos=1,
                    retain=True,
                )

    except Exception as exc:
        logger.error(
            "Erreur vérification capteurs zone %s : %s", zone_num, exc
        )


def start_stale_watcher(interval_seconds: int = 60) -> None:
    """Démarre le thread de surveillance (une seule fois)."""
    t = threading.Thread(
        target=_stale_sensor_watcher,
        args=(interval_seconds,),
        daemon=True,
        name="stale-sensor-watcher",
    )
    t.start()


# ── Lecture des seuils depuis Firebase ───────────────────────────────────────

def _get_zone_config(uid: str, zone_num: str) -> dict:
    """
    Lit la config complète d'une zone depuis Firestore :
    seuils + identité métier (plante, stade, nom).
    Les constantes système (HUMIDITY_ALERT, etc.) restent dans config.py.
    """
    defaults = {
        "minHumidity": cfg.HUMIDITY_ALERT,
        "maxHumidity": 70.0,
        "duration":    15,
        "zone_name":   f"Zone {zone_num}",
        "plant_type":  None,
        "crop_stage":  None,
    }
    try:
        doc = (
            db.collection("users").document(uid)
              .collection("zones").document(f"zone{zone_num}")
              .get()
        )
        if doc.exists:
            d = doc.to_dict()
            defaults.update({
                "minHumidity": d.get("minHumidity", defaults["minHumidity"]),
                "maxHumidity": d.get("maxHumidity", defaults["maxHumidity"]),
                "duration":    d.get("duration",    defaults["duration"]),
                "zone_name":   d.get("zone_name",   defaults["zone_name"]),
                "plant_type":  d.get("plant_type"),
                "crop_stage":  d.get("crop_stage"),
            })
    except Exception as exc:
        logger.error("Erreur lecture config zone %s : %s", zone_num, exc)
    return defaults

# ── Contrôle automatique de la vanne ─────────────────────────────────────────

def _control_valve_auto(uid: str, zone_num: str, humidity: float) -> None:
    """
    Ouvre ou ferme la vanne selon les seuils configurés.
    `humidity` est la moyenne agrégée des capteurs actifs de la zone.
    """
    thresholds = _get_zone_config(uid, zone_num)   
    min_humidity = thresholds["minHumidity"]
    max_humidity = thresholds["maxHumidity"]
    duration     = thresholds["duration"]
    topic_valve  = f"robocare/{uid}/valve/control/{zone_num}"

    if humidity < min_humidity:
        logger.info(
            "Zone %s | H moy. %.1f%% < min %.1f%% → Vanne OUVERTE",
            zone_num, humidity, min_humidity,
        )
        mqtt_client.publish(topic_valve, "1", qos=1)
        db.collection("users").document(uid) \
          .collection("zones").document(f"zone{zone_num}") \
          .update({"enabled": True})

        def auto_close():
            time.sleep(duration * 60)
            zone_doc = (
                db.collection("users").document(uid)
                  .collection("zones").document(f"zone{zone_num}")
                  .get()
            )
            if zone_doc.exists:
                current_h = zone_doc.to_dict().get("humidity", 0)
                if current_h >= max_humidity:
                    logger.info(
                        "Zone %s | H moy. %.1f%% ≥ max %.1f%% → Vanne FERMÉE (auto)",
                        zone_num, current_h, max_humidity,
                    )
                    mqtt_client.publish(topic_valve, "0", qos=1)
                    db.collection("users").document(uid) \
                      .collection("zones").document(f"zone{zone_num}") \
                      .update({"enabled": False})

        threading.Thread(target=auto_close, daemon=True).start()

    elif humidity >= max_humidity:
        logger.info(
            "Zone %s | H moy. %.1f%% ≥ max %.1f%% → Vanne FERMÉE",
            zone_num, humidity, max_humidity,
        )
        mqtt_client.publish(topic_valve, "0", qos=1)
        db.collection("users").document(uid) \
          .collection("zones").document(f"zone{zone_num}") \
          .update({"enabled": False})


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
    Callback principal.

    Topic : robocare/{uid}/zone/{zone_num}/sensor/{sensor_id}/data
    parts :     [0]   [1]  [2]    [3]        [4]      [5]      [6]

    Remplacement hardware transparent :
      Le capteur à l'adresse "11" tombe en panne → l'agriculteur configure
      le nouveau capteur avec l'adresse "11" → ce callback reçoit exactement
      le même topic → update_sensor_data() écrase les données au même
      endroit Firestore → active passe automatiquement à True.
      Aucune modification serveur requise.
    """
    try:
        # ── Discovery ────────────────────────────────────────────────────────
        if msg.topic == "robocare/discovery":
            _handle_discovery(msg)
            return

        # ── Décodage du topic ─────────────────────────────────────────────────
        parts = msg.topic.split("/")
        if len(parts) < 7:
            logger.warning(
                "Topic inattendu (attendu ≥ 7 parts, reçu %d) : %s",
                len(parts), msg.topic,
            )
            return

        uid       = parts[1]
        zone_num  = parts[3]   # adresse fixe de la zone  (ex. "1")
        sensor_id = parts[5]   # adresse fixe du capteur  (ex. "11")

        payload = json.loads(msg.payload.decode("utf-8"))
        logger.info(
            "← MQTT Zone %s | Capteur %s | UID: %s",
            zone_num, sensor_id, uid,
        )
        logger.debug("  Payload : %s", payload)

        # ── Mise à jour Firebase + calcul moyenne zone ────────────────────────
        result = update_sensor_data(db, uid, zone_num, payload, sensor_id)
        if result[0] is None:
            return

        humidity, temperature, ph, ec, n, p, k = result

        # ── Métadonnées réseau ────────────────────────────────────────────────
        meta = payload.get("meta", {})
        if meta:
            try:
                db.collection("users").document(uid) \
                  .collection("zones").document(f"zone{zone_num}") \
                  .collection("signal") \
                  .add({
                      "sensor_id": sensor_id,
                      "node_id":   meta.get("node_id"),
                      "mac":       meta.get("mac"),
                      "rssi":      meta.get("rssi"),
                      "snr":       meta.get("snr"),
                      "timestamp": firestore.SERVER_TIMESTAMP,
                  })
                logger.info(
                    "Zone %s | Capteur %s | MAC: %s | RSSI: %d dBm | SNR: %.1f",
                    zone_num, sensor_id,
                    meta.get("mac"), meta.get("rssi", 0), meta.get("snr", 0),
                )
            except Exception as exc:
                logger.warning(
                    "Erreur sauvegarde métadonnées zone %s / capteur %s : %s",
                    zone_num, sensor_id, exc,
                )

        # ── Contrôle automatique de la vanne ─────────────────────────────────
        _control_valve_auto(uid, zone_num, humidity)

        # ── Alerte critique ───────────────────────────────────────────────────
        if humidity < cfg.HUMIDITY_ALERT:
            send_critical_alert(db, uid, zone_num, humidity)

        # ── Stress hydrique ───────────────────────────────────────────────────
        if predict_stress_risk(db, uid, zone_num, humidity):
            logger.warning("Stress hydrique prédit — Zone %s", zone_num)

        # ── ML combinée ───────────────────────────────────────────────────────
        ml_result = predict_irrigation_combined(
            db, uid, zone_num, humidity, temperature, ec, n, p, k
        )

        # ── Historique Firestore ──────────────────────────────────────────────
        db.collection("users").document(uid) \
          .collection("zones").document(f"zone{zone_num}") \
          .collection("history") \
          .add(ml_result.to_firestore_dict())

        if ml_result.is_dangerous:
            send_irrigation_prediction(db, uid, zone_num, ml_result)

    except Exception as exc:
        logger.exception("Erreur on_message : %s", exc)


# ── Gestion discovery ─────────────────────────────────────────────────────────

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


# ── Listeners Firestore ───────────────────────────────────────────────────────

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
    """
    Écoute les changements sur la collection zones/.
    La suppression d'une zone n'affecte pas les autres zones
    (elles conservent leurs numéros).
    """
    for change in changes:
        if change.type.name in ("ADDED", "MODIFIED"):
            doc      = change.document
            data     = doc.to_dict() or {}
            enabled  = bool(data.get("enabled", False))
            zone_num = doc.id.replace("zone", "")

            mqtt_client.publish(
                f"robocare/{uid}/valve/control/{zone_num}",
                "1" if enabled else "0",
                qos=1,
            )

        elif change.type.name == "REMOVED":
            # Zone supprimée — les autres zones ne changent PAS de numéro
            zone_num = change.document.id.replace("zone", "")
            logger.info(
                "Zone %s supprimée pour UID %s — autres zones inchangées",
                zone_num, uid,
            )
            # Fermer la vanne de cette zone par sécurité
            mqtt_client.publish(
                f"robocare/{uid}/valve/control/{zone_num}",
                "0",
                qos=1,
                retain=True,
            )


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
    if uid not in _command_watchers:
        ref = (
            db.collection("users")
              .document(uid)
              .collection("commands")
              .document("variateur")
        )
        _command_watchers[uid] = ref.on_snapshot(
            lambda ds, c, rt: _on_variateur_snapshot(uid, ds, c, rt)
        )

        z_ref = db.collection("users").document(uid).collection("zones")
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
            for ch in c if ch.type.name == "ADDED"
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
                mqtt_client.publish(config_topic, uid, qos=1, retain=True)
                logger.info(
                    "Device claimed → UID publié : %s → topic %s", uid, config_topic
                )
                start_command_listener(uid)

    db.collection("pending_devices").on_snapshot(on_pending)
    logger.info("Tous les watchers Firestore démarrés")