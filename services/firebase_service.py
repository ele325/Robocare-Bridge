# -*- coding: utf-8 -*-
"""
services/firebase_service.py — Accès Firestore centralisé RoboCare
"""

from firebase_admin import firestore
from ml.health_score import calculate_health_score
from utils.logger import get_logger

logger = get_logger("robocare.services.firebase")


def update_sensor_data(db, uid: str, zone_num: str | int, payload: dict):
    """
    Parse le payload MQTT, calcule le score santé,
    persiste dans Firestore (zone courante + historique).

    Retourne (humidity, temperature, ec, n, p, k) ou (None,)*6 si erreur.
    """
    user_ref = db.collection("users").document(uid)
    if not user_ref.get().exists:
        logger.warning("UID introuvable dans Firestore : %s", uid)
        return None, None, None, None, None, None

    zone_id  = "zone{}".format(zone_num)
    doc_ref  = user_ref.collection("zones").document(zone_id)

    m         = payload.get("measurements", {})
    nutrients = m.get("nutrients_mg_per_kg", {})

    humidity    = float(m.get("moisture_percent",       0.0))
    temperature = float(m.get("temperature_celsius",    0.0))
    ph          = float(m.get("ph",                     0.0))
    ec          = float(m.get("conductivity_uS_per_cm", 0.0))
    n           = float(nutrients.get("nitrogen",        0.0))
    p           = float(nutrients.get("phosphorus",      0.0))
    k           = float(nutrients.get("potassium",       0.0))

    health_score = calculate_health_score(humidity, ph, ec, n, p, k)

    data_fields = {
        "humidity":    humidity,
        "temperature": temperature,
        "ph":          ph,
        "ec":          ec,
        "azote":       n,
        "phosphore":   p,
        "potassium":   k,
        "sante":       health_score,
        "zone_num":    str(zone_num),
    }

    doc_ref.set({**data_fields, "last_update": firestore.SERVER_TIMESTAMP}, merge=True)
    doc_ref.collection("history").add({**data_fields, "timestamp": firestore.SERVER_TIMESTAMP})

    logger.info(
        "Zone %s → H:%.1f%% T:%.1f°C EC:%.0f N:%.1f P:%.1f K:%.1f Santé:%d/10",
        zone_num, humidity, temperature, ec, n, p, k, health_score,
    )
    return humidity, temperature, ec, n, p, k