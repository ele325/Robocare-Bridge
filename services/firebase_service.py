# -*- coding: utf-8 -*-
"""
services/firebase_service.py — Lecture / écriture Firestore RoboCare v2.6
Architecture : 
 - 'measures' : Collection pour l'historique des graphiques (Flutter).
 - 'sensors'  : État de chaque capteur (incluant MAC et Node ID).
 - 'zones'    : Document principal pour l'affichage temps réel.
"""

import time
from firebase_admin import firestore
import config as cfg
from ml.health_score import calculate_health_score
from utils.logger import get_logger

logger = get_logger("robocare.firebase")

# ── Helpers ──────────────────────────────────────────────────────────────────

def _float(value, default: float = 0.0) -> float:
    """Conversion sécurisée en float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _parse_payload(payload: dict, zone_num: str, sensor_id: str):
    """Extrait les mesures depuis le payload structuré."""
    m = payload.get("measurements")
    if not isinstance(m, dict):
        raise ValueError(f"Clé 'measurements' absente (zone={zone_num}, capteur={sensor_id})")

    nut = m.get("nutrients_mg_per_kg", {})
    return (
        _float(m.get("moisture_percent")),
        _float(m.get("temperature_celsius")),
        _float(m.get("ph")),
        _float(m.get("conductivity_uS_per_cm")),
        _float(nut.get("nitrogen")),
        _float(nut.get("phosphorus")),
        _float(nut.get("potassium")),
    )

def _is_sensor_fresh(sensor_data: dict) -> bool:
    """Vérifie si le capteur n'est pas considéré comme hors-ligne."""
    if cfg.SENSOR_STALE_SECONDS is None:
        return True
    ts = sensor_data.get("timestamp")
    if ts is None:
        return True
    
    if hasattr(ts, "timestamp"):
        sensor_time = ts.timestamp()
    elif hasattr(ts, "_seconds"):
        sensor_time = ts._seconds
    else:
        return True
    return (time.time() - sensor_time) <= cfg.SENSOR_STALE_SECONDS

# ── Fonction principale ───────────────────────────────────────────────────────

def update_sensor_data(
    db,
    uid: str,
    zone_num: str,
    payload: dict,
    sensor_id: str = "default",
):
    """
    Traite un message MQTT et met à jour Firestore sur 3 niveaux.
    """

    # 1. Parsing des données
    try:
        humidity, temperature, ph, ec, n, p, k = _parse_payload(payload, zone_num, sensor_id)
    except Exception as exc:
        logger.error("Payload invalide zone %s : %s", zone_num, exc)
        return (None,) * 7

    zone_ref = db.collection("users").document(uid).collection("zones").document(f"zone{zone_num}")
    meta = payload.get("meta", {})

    # 2. Mise à jour du capteur individuel (avec MAC et Node ID)
    try:
        sensor_info = {
            "sensor_id":   sensor_id,
            "humidity":    humidity,
            "temperature": temperature,
            "ph":          ph,
            "ec":          ec,
            "n":           n,
            "p":           p,
            "k":           k,
            "active":      True,
            "timestamp":   firestore.SERVER_TIMESTAMP,
        }
        
        # Ajout des données techniques si présentes
        if meta:
            sensor_info.update({
                "mac":     meta.get("mac"),
                "node_id": meta.get("node_id"),
                "rssi":    meta.get("rssi"),
                "snr":     meta.get("snr")
            })

        zone_ref.collection("sensors").document(sensor_id).set(sensor_info)
    except Exception as exc:
        logger.error("Erreur sauvegarde capteur %s : %s", sensor_id, exc)

    # 3. Calcul de la moyenne de la zone (Agrégation)
    count = 1
    try:
        sensor_docs = zone_ref.collection("sensors").get()
        readings = [d.to_dict() for d in sensor_docs if d.to_dict() and d.to_dict().get("active") and _is_sensor_fresh(d.to_dict())]
        
        if readings:
            count = len(readings)
            def avg(f): return sum(_float(r.get(f)) for r in readings) / count
            humidity, temperature, ph, ec, n, p, k = avg("humidity"), avg("temperature"), avg("ph"), avg("ec"), avg("n"), avg("p"), avg("k")
    except Exception as exc:
        logger.error("Erreur calcul moyenne zone %s : %s", zone_num, exc)

    # 4. ARCHIVAGE DANS 'MEASURES' (Pour les graphiques Flutter)
    try:
        zone_ref.collection("measures").add({
            "timestamp":   firestore.SERVER_TIMESTAMP,
            "humidity":    humidity,
            "temperature": temperature,
            "ph":          ph,
            "ec":          ec,
            "n":           n,
            "p":           p,
            "k":           k
        })
    except Exception as exc:
        logger.error("Erreur archivage 'measures' zone %s : %s", zone_num, exc)

    # 5. Mise à jour du document Zone (Temps réel et Santé)
    try:
        health = calculate_health_score(humidity, ph, ec, n, p, k)
        zone_ref.set({
            "humidity":     humidity,
            "temperature":  temperature,
            "ph":           ph,
            "ec":           ec,
            "n":            n,
            "p":            p,
            "k":            k,
            "sante":        health,
            "sensor_count": count,
            "last_updated": firestore.SERVER_TIMESTAMP,
        }, merge=True)
    except Exception as exc:
        logger.error("Erreur mise à jour zone %s : %s", zone_num, exc)

    return humidity, temperature, ph, ec, n, p, k

# ── Désactivation ───────────────────────────────────────────────────────────

def deactivate_sensor(db, uid: str, zone_num: str, sensor_id: str) -> None:
    """Marque un capteur comme inactif s'il ne répond plus."""
    try:
        db.collection("users").document(uid)\
          .collection("zones").document(f"zone{zone_num}")\
          .collection("sensors").document(sensor_id)\
          .update({"active": False})
    except Exception as exc:
        logger.error("Erreur désactivation capteur %s : %s", sensor_id, exc)