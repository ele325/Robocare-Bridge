# -*- coding: utf-8 -*-
"""
main.py — Point d'entrée RoboCare v6.0
"""

import sys
import threading

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Logging (initialiser en premier) ────────────────────────────────────────
from utils.logger import setup_logging, get_logger
setup_logging("robocare.log")
logger = get_logger("robocare.main")

# ── Validation de la configuration ──────────────────────────────────────────
import config as cfg
if not cfg.GROQ_API_KEY:
    logger.critical("GROQ_API_KEY manquante ! Définissez-la avec : $env:GROQ_API_KEY='votre_cle'")
    sys.exit(1)

# ── Firebase ────────────────────────────────────────────────────────────────
from firebase_admin import credentials, firestore
import firebase_admin

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── MQTT ────────────────────────────────────────────────────────────────────
import paho.mqtt.client as mqtt
import mqtt_handler

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = mqtt_handler.on_connect
mqtt_client.on_message = mqtt_handler.on_message

# Injection des dépendances
mqtt_handler.init(mqtt_client, db)


# ── Démarrage ────────────────────────────────────────────────────────────────
def main():
    logger.info("═══════════════════════════════════════")
    logger.info("  RoboCare Bridge v6.0  — démarrage")
    logger.info("═══════════════════════════════════════")

    try:
        mqtt_client.connect(cfg.MQTT_BROKER, cfg.MQTT_PORT)
        threading.Thread(target=mqtt_handler.start_all_watchers, daemon=True).start()
        logger.info("En attente de données MQTT sur %s:%d ...", cfg.MQTT_BROKER, cfg.MQTT_PORT)
        mqtt_client.loop_forever()
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur.")
    except Exception as exc:
        logger.critical("Erreur fatale : %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()