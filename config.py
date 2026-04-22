# -*- coding: utf-8 -*-
"""
config.py — Centralisation de toutes les constantes RoboCare
"""

import os

# ── Groq ────────────────────────────────────────────────────────────────────
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS  = 300
GROQ_TEMPERATURE = 0.3
GROQ_MAX_RETRIES = 3

# ── MQTT ────────────────────────────────────────────────────────────────────
MQTT_BROKER = os.environ.get("MQTT_BROKER", "80.75.212.179")
MQTT_PORT     = int(os.environ.get("MQTT_PORT", 1883))
MQTT_USER     = "root"
MQTT_PASSWORD = "A6k0tYRThA7UjNNH"
MQTT_TOPIC_DATA = "robocare/+/zone/+/sensor/+/data"
# ── ML ──────────────────────────────────────────────────────────────────────
ML_HISTORY_LIMIT        = 30   # nb relevés pour régression
ML_MIN_HISTORY          = 10   # minimum requis
ML_STRESS_HISTORY_LIMIT = 5
ML_STRESS_MIN_HISTORY   = 3
ML_POLY_DEGREE          = 2    # degré polynomial
ML_RIDGE_ALPHA          = 1.0  # régularisation Ridge
ML_ZSCORE_THRESHOLD     = 2.5  # seuil détection anomalie
DANGEROUS_SCORE         = 50   # score à partir duquel on notifie

# ── Seuils agronomiques ──────────────────────────────────────────────────────
HUMIDITY_CRITICAL = 25.0
HUMIDITY_LOW      = 35.0
HUMIDITY_LIMIT    = 45.0
HUMIDITY_ALERT    = 30.0   # seuil notification immédiate
TEMP_EXCESSIVE    = 35.0
TEMP_HIGH         = 28.0
TEMP_NORMAL       = 22.0
EC_CRITICAL       = 100.0
EC_LOW            = 300.0
TREND_H_FAST_DROP = -3.0
TREND_H_DROP      = -1.5
TREND_EC_DROP     = -50.0
N_CRITICAL        = 10.0
N_LOW             = 20.0
TREND_N_DROP      = -2.0
P_CRITICAL        = 5.0
P_LOW             = 10.0
K_CRITICAL        = 10.0
K_LOW             = 20.0

# ── Santé du sol ─────────────────────────────────────────────────────────────
HEALTH_HUMIDITY_MIN = 35.0
HEALTH_HUMIDITY_MAX = 85.0
HEALTH_PH_MIN       = 5.5
HEALTH_PH_MAX       = 8.0
HEALTH_EC_MIN       = 300.0
HEALTH_N_MIN        = 20.0
HEALTH_P_MIN        = 10.0
HEALTH_K_MIN        = 20.0

# ── Multi-capteurs ───────────────────────────────────────────────────────────
# Durée (en secondes) au-delà de laquelle un capteur est considéré inactif
# et exclu du calcul de moyenne. Mettre à None pour désactiver.
SENSOR_STALE_SECONDS = 300   # 5 minutes