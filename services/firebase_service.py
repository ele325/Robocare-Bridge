# -*- coding: utf-8 -*-
"""
services/firebase_service.py — Lecture / écriture Firestore RoboCare v2.7
Architecture :
 - 'measures' : Collection pour l'historique des graphiques Flutter.
 - 'sensors'  : État de chaque capteur.
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
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_number(source: dict, keys: list[str], default: float = 0.0) -> float:
    """
    Retourne la première valeur numérique trouvée dans source selon une liste de clés.
    Exemple: _first_number(m, ["humidity", "moisture_percent"])
    """
    for key in keys:
        if key in source and source.get(key) is not None:
            return _float(source.get(key), default)
    return default


def _parse_payload(payload: dict, zone_num: str, sensor_id: str):
    """
    Extrait les mesures depuis le payload MQTT.

    Format simple accepté :
    {
      "measurements": {
        "temperature": 25.5,
        "humidity": 40.0,
        "ph": 6.8,
        "ec": 500,
        "n": 30,
        "p": 20,
        "k": 25
      }
    }

    Ancien format accepté aussi :
    {
      "measurements": {
        "temperature_celsius": 25.5,
        "moisture_percent": 40.0,
        "ph": 6.8,
        "conductivity_uS_per_cm": 500,
        "nutrients_mg_per_kg": {
          "nitrogen": 30,
          "phosphorus": 20,
          "potassium": 25
        }
      }
    }
    """
    m = payload.get("measurements")

    if not isinstance(m, dict):
        raise ValueError(
            f"Clé 'measurements' absente ou invalide "
            f"(zone={zone_num}, capteur={sensor_id})"
        )

    nut = m.get("nutrients_mg_per_kg", {})
    if not isinstance(nut, dict):
        nut = {}

    humidity = _first_number(m, [
        "humidity",
        "moisture",
        "moisture_percent",
        "soil_moisture",
        "soil_moisture_percent",
    ])

    temperature = _first_number(m, [
        "temperature",
        "temp",
        "temperature_celsius",
        "temperature_c",
    ])

    ph = _first_number(m, [
        "ph",
        "pH",
    ])

    ec = _first_number(m, [
        "ec",
        "conductivity",
        "conductivity_uS_per_cm",
        "ec_us_cm",
    ])

    n = _first_number(m, [
        "n",
        "nitrogen",
        "azote",
    ], default=_first_number(nut, ["n", "nitrogen", "azote"]))

    p = _first_number(m, [
        "p",
        "phosphorus",
        "phosphore",
    ], default=_first_number(nut, ["p", "phosphorus", "phosphore"]))

    k = _first_number(m, [
        "k",
        "potassium",
    ], default=_first_number(nut, ["k", "potassium"]))

    logger.info(
        "Payload parsé zone %s capteur %s : H=%.1f T=%.1f pH=%.1f EC=%.1f N=%.1f P=%.1f K=%.1f",
        zone_num, sensor_id, humidity, temperature, ph, ec, n, p, k
    )

    return humidity, temperature, ph, ec, n, p, k


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
    Traite un message MQTT et met à jour Firestore sur 3 niveaux :
    1. users/{uid}/zones/zone{zone_num}/sensors/{sensor_id}
    2. users/{uid}/zones/zone{zone_num}/measures
    3. users/{uid}/zones/zone{zone_num}
    """

    # 1. Parsing des données
    try:
        humidity, temperature, ph, ec, n, p, k = _parse_payload(
            payload, zone_num, sensor_id
        )
    except Exception as exc:
        logger.error("Payload invalide zone %s : %s", zone_num, exc)
        return (None,) * 7

    zone_ref = (
        db.collection("users")
        .document(uid)
        .collection("zones")
        .document(f"zone{zone_num}")
    )

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    # 2. Mise à jour du capteur individuel
    try:
        sensor_info = {
            "sensor_id": sensor_id,
            "humidity": humidity,
            "temperature": temperature,
            "ph": ph,
            "ec": ec,
            "n": n,
            "p": p,
            "k": k,
            "active": True,
            "timestamp": firestore.SERVER_TIMESTAMP,
        }

        if meta:
            sensor_info.update({
                "mac": meta.get("mac"),
                "node_id": meta.get("node_id"),
                "rssi": meta.get("rssi"),
                "snr": meta.get("snr"),
            })

        zone_ref.collection("sensors").document(sensor_id).set(
            sensor_info,
            merge=True,
        )

    except Exception as exc:
        logger.error("Erreur sauvegarde capteur %s : %s", sensor_id, exc)

    # 3. Calcul de la moyenne de la zone
    count = 1
    try:
        sensor_docs = zone_ref.collection("sensors").get()

        readings = []
        for doc in sensor_docs:
            sensor_data = doc.to_dict()
            if not sensor_data:
                continue
            if not sensor_data.get("active"):
                continue
            if not _is_sensor_fresh(sensor_data):
                continue
            readings.append(sensor_data)

        if readings:
            count = len(readings)

            def avg(field: str) -> float:
                return sum(_float(r.get(field)) for r in readings) / count

            humidity = avg("humidity")
            temperature = avg("temperature")
            ph = avg("ph")
            ec = avg("ec")
            n = avg("n")
            p = avg("p")
            k = avg("k")

    except Exception as exc:
        logger.error("Erreur calcul moyenne zone %s : %s", zone_num, exc)

    # 4. Archivage dans 'measures'
    try:
        zone_ref.collection("measures").add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "humidity": humidity,
            "temperature": temperature,
            "ph": ph,
            "ec": ec,
            "n": n,
            "p": p,
            "k": k,
            "sensor_id": sensor_id,
        })

    except Exception as exc:
        logger.error("Erreur archivage 'measures' zone %s : %s", zone_num, exc)

    # 5. Mise à jour du document Zone principal
    try:
        health = calculate_health_score(humidity, ph, ec, n, p, k)

        zone_ref.set({
            "humidity": humidity,
            "temperature": temperature,
            "ph": ph,
            "ec": ec,
            "n": n,
            "p": p,
            "k": k,
            "sante": health,
            "sensor_count": count,
            "last_updated": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        logger.info(
            "Zone %s mise à jour Firestore : H=%.1f T=%.1f pH=%.1f EC=%.1f N=%.1f P=%.1f K=%.1f santé=%s",
            zone_num, humidity, temperature, ph, ec, n, p, k, health
        )

    except Exception as exc:
        logger.error("Erreur mise à jour zone %s : %s", zone_num, exc)

    return humidity, temperature, ph, ec, n, p, k


# ── Désactivation ───────────────────────────────────────────────────────────

def deactivate_sensor(db, uid: str, zone_num: str, sensor_id: str) -> None:
    """Marque un capteur comme inactif s'il ne répond plus."""
    try:
        (
            db.collection("users")
            .document(uid)
            .collection("zones")
            .document(f"zone{zone_num}")
            .collection("sensors")
            .document(sensor_id)
            .update({"active": False})
        )
    except Exception as exc:
        logger.error("Erreur désactivation capteur %s : %s", sensor_id, exc)