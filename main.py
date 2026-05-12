# -*- coding: utf-8 -*-
"""
main.py — Point d'entrée RoboCare v6.0
"""

import sys
import threading
import os
import json
from firebase_admin import credentials, firestore
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Logging ──────────────────────────────────────────────────────────────────
from utils.logger import setup_logging, get_logger
setup_logging("robocare.log")
logger = get_logger("robocare.main")

# ── Validation configuration ─────────────────────────────────────────────────
import config as cfg
if not cfg.GROQ_API_KEY:
    logger.critical(
        "GROQ_API_KEY manquante ! Définissez-la avec : "
        "$env:GROQ_API_KEY='votre_cle'"
    )
    sys.exit(1)

# ── Firebase ─────────────────────────────────────────────────────────────────
import firebase_admin

# Priorité : fichier local serviceAccountKey.json, sinon variable d'env
key_path = "serviceAccountKey.json"
if os.path.exists(key_path):
    logger.info("Chargement des credentials Firebase depuis %s", key_path)
    cred = credentials.Certificate(key_path)
else:
    env_creds = os.getenv("FIREBASE_CREDENTIALS")
    if env_creds:
        logger.info("Chargement des credentials Firebase depuis FIREBASE_CREDENTIALS")
        firebase_json = json.loads(env_creds)
        cred = credentials.Certificate(firebase_json)
    else:
        logger.critical("Credentials Firebase manquants ! (serviceAccountKey.json ou FIREBASE_CREDENTIALS)")
        sys.exit(1)

firebase_admin.initialize_app(cred)
db = firestore.client()

# ── MQTT ─────────────────────────────────────────────────────────────────────
import paho.mqtt.client as mqtt
import mqtt_handler

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = mqtt_handler.on_connect
mqtt_client.on_message = mqtt_handler.on_message

# ✅ Auth désactivée — Mosquitto en allow_anonymous
# mqtt_client.username_pw_set(cfg.MQTT_USER, cfg.MQTT_PASSWORD)

mqtt_handler.init(mqtt_client, db)


# ── Démarrage ────────────────────────────────────────────────────────────────
def main():
    logger.info("RoboCare Bridge — démarrage")
    logger.info(
        "Adressage fixe : le remplacement hardware se fait en "
        "reconfigurant le nouveau capteur/zone avec la même adresse."
    )

    try:
        mqtt_client.connect(cfg.MQTT_BROKER, cfg.MQTT_PORT)

        threading.Thread(
            target=mqtt_handler.start_all_watchers,
            daemon=True,
            name="firestore-watchers",
        ).start()

        mqtt_handler.start_stale_watcher(interval_seconds=60)
        mqtt_handler.start_pump_watcher(interval_seconds=60)

        logger.info(
            "En attente de données MQTT sur %s:%d ...",
            cfg.MQTT_BROKER, cfg.MQTT_PORT,
        )
        mqtt_client.loop_forever()

    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur.")
    except Exception as exc:
        logger.critical("Erreur fatale : %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()