# -*- coding: utf-8 -*-
"""
mqtt_handler.py — Callbacks MQTT RoboCare v2.2
Corrections :
  - on_connect() souscrit à "robocare/+/zone/+/data" (aligné sur config.py)
  - _handle_discovery() publie l'UID sur "robocare/config/<mac>" quand claimed
  - on_pending() corrigé : publie UID dès que status == "claimed"
"""

import json
import time
import threading

from firebase_admin import firestore

import config as cfg
from services.firebase_service      import update_sensor_data
from services.notification_service  import send_critical_alert, send_irrigation_prediction
from services.chatbot_service       import handle_chatbot_async
from ml.predictor                   import predict_stress_risk, predict_irrigation_combined
from utils.logger                   import get_logger

logger = get_logger("robocare.mqtt")

# ── État interne ─────────────────────────────────────────────────────────────
_command_watchers    : dict  = {}
_zone_watchers       : dict  = {}
_processed_msg_ids   : set   = set()
_watcher_start_time  : float = time.time()

# Référence au client MQTT et Firestore (injectés depuis main.py)
mqtt_client = None
db          = None


def init(client, firestore_db):
    """Injecte le client MQTT et Firestore dans ce module."""
    global mqtt_client, db
    mqtt_client = client
    db          = firestore_db


# ── Callbacks MQTT ───────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, reason_code, properties):
    logger.info("Connecté au broker MQTT (code %s)", reason_code)

    # Données capteurs — aligné sur network_manager.c
    # Topic publié par la réceptrice : "robocare/<uid>/zone/<node_id>/data"
    client.subscribe(cfg.MQTT_TOPIC_DATA)   # "robocare/+/zone/+/data"

    # Discovery des nouveaux devices
    client.subscribe("robocare/discovery")

    logger.info("Subscriptions actives :")
    logger.info("  → %s", cfg.MQTT_TOPIC_DATA)
    logger.info("  → robocare/discovery")


def on_message(client, userdata, msg):
    try:
        # ── Discovery ────────────────────────────────────────────────────
        if msg.topic == "robocare/discovery":
            _handle_discovery(msg)
            return

        # ── Données capteur ───────────────────────────────────────────────
        # Format topic : "robocare/<uid>/zone/<node_id>/data"
        parts = msg.topic.split("/")
        if len(parts) < 5:
            logger.warning("Topic inattendu (< 5 parts) : %s", msg.topic)
            return

        uid      = parts[1]
        zone_num = parts[3]
        payload  = json.loads(msg.payload.decode("utf-8"))

        logger.info("← MQTT Zone %s | UID: %s", zone_num, uid)
        logger.debug("  Payload : %s", payload)

        humidity, temperature, ec, n, p, k = update_sensor_data(
            db, uid, zone_num, payload
        )
        if humidity is None:
            return

        # Alerte critique immédiate
        if humidity < cfg.HUMIDITY_ALERT:
            send_critical_alert(db, uid, zone_num, humidity)

        # Stress hydrique (pente)
        if predict_stress_risk(db, uid, zone_num, humidity):
            logger.warning("Stress hydrique prédit — Zone %s", zone_num)

        # ML combinée
        result = predict_irrigation_combined(
            db, uid, zone_num, humidity, temperature, ec, n, p, k
        )

        # Historique Firestore
        db.collection("users") \
          .document(uid) \
          .collection("zones") \
          .document("zone{}".format(zone_num)) \
          .collection("history") \
          .add(result.to_firestore_dict())

        if result.is_dangerous:
            send_irrigation_prediction(db, uid, zone_num, result)

    except Exception as exc:
        logger.exception("Erreur on_message : %s", exc)


def _handle_discovery(msg):
    """
    Reçoit {"mac": "AA:BB:CC:DD:EE:FF"} depuis la réceptrice.
    Écrit dans Firestore "pending_devices/<mac>" avec status="waiting".
    Si le device est déjà "claimed" (connu), publie l'UID directement.
    """
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        mac  = data.get("mac", "").strip()
        if not mac:
            logger.warning("Discovery : MAC manquante dans payload")
            return

        logger.info("Discovery reçu : MAC=%s", mac)

        # Vérifier si le device est déjà "claimed"
        doc_ref  = db.collection("pending_devices").document(mac)
        doc_snap = doc_ref.get()

        if doc_snap.exists:
            doc_data = doc_snap.to_dict()
            status   = doc_data.get("status", "waiting")
            uid      = doc_data.get("uid", "")

            if status == "claimed" and uid:
                # Device déjà connu → envoyer l'UID immédiatement
                config_topic = "robocare/config/{}".format(mac)
                mqtt_client.publish(config_topic, uid, qos=1, retain=True)
                logger.info("Device connu → UID envoyé : %s → %s", mac, uid)
                return

        # Nouveau device ou pas encore claimed → enregistrer en attente
        doc_ref.set(
            {
                "mac":       mac,
                "status":    "waiting",
                "timestamp": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        logger.info("Nouveau device enregistré en attente : MAC=%s", mac)
        logger.info("  → L'utilisateur doit 'claimer' ce device dans l'app")

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

        # Commande pompe → topic aligné sur network_manager.c
        mqtt_client.publish(
            "robocare/{}/pump/control".format(uid),
            "1" if is_on else "0",
            qos=1,
        )
        if freq > 0:
            mqtt_client.publish(
                "robocare/{}/pump/frequency".format(uid),
                str(freq),
                qos=1,
            )


def _on_zones_snapshot(uid, col_snapshot, changes, read_time):
    for change in changes:
        if change.type.name in ("ADDED", "MODIFIED"):
            doc      = change.document
            data     = doc.to_dict()
            enabled  = bool(data.get("enabled", False))
            zone_num = str(data.get("zone_num",
                           doc.id.replace("zone", "")))

            # Commande vanne → topic aligné sur network_manager.c
            mqtt_client.publish(
                "robocare/{}/valve/control/{}".format(uid, zone_num),
                "1" if enabled else "0",
                qos=1,
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


def start_command_listener(uid: str) -> None:
    if uid not in _command_watchers:
        # Écoute commandes variateur
        ref = (
            db.collection("users")
              .document(uid)
              .collection("commands")
              .document("variateur")
        )
        _command_watchers[uid] = ref.on_snapshot(
            lambda ds, c, rt: _on_variateur_snapshot(uid, ds, c, rt)
        )

        # Écoute commandes zones (vannes)
        z_ref = db.collection("users").document(uid).collection("zones")
        _zone_watchers[uid] = z_ref.on_snapshot(
            lambda cs, c, rt: _on_zones_snapshot(uid, cs, c, rt)
        )

        logger.info("Listeners démarrés pour UID : %s", uid)


def start_all_watchers() -> None:
    """Démarre tous les listeners Firestore."""

    # Listeners pour les utilisateurs existants
    users = db.collection("users").get()
    for u in users:
        start_command_listener(u.id)

    # Listener pour les nouveaux utilisateurs
    db.collection("users").on_snapshot(
        lambda s, c, r: [
            start_command_listener(ch.document.id)
            for ch in c if ch.type.name == "ADDED"
        ]
    )

    # Listener messages chatbot
    db.collection_group("messages").on_snapshot(on_msg_snapshot)

    # Listener pending_devices : envoie l'UID quand status == "claimed"
    def on_pending(snapshots, changes, read_time):
        for change in changes:
            if change.type.name not in ("ADDED", "MODIFIED"):
                continue
            d      = change.document.to_dict()
            status = d.get("status", "")
            mac    = d.get("mac", "")
            uid    = d.get("uid", "")

            if status == "claimed" and mac and uid:
                # Publier l'UID sur le topic de config de ce device
                config_topic = "robocare/config/{}".format(mac)
                mqtt_client.publish(config_topic, uid, qos=1, retain=True)
                logger.info("Device claimed → UID publié : %s → topic %s",
                            uid, config_topic)

                # Démarrer les listeners pour cet utilisateur
                start_command_listener(uid)

    db.collection("pending_devices").on_snapshot(on_pending)
    logger.info("Tous les watchers Firestore démarrés")