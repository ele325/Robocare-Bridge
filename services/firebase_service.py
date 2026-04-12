# -*- coding: utf-8 -*-
"""
services/firebase_service.py — Lecture / écriture Firestore RoboCare v2.3

Principe d'adressage (SIMPLE) :
─────────────────────────────────────────────────────────────────────────────
  • L'adresse (sensor_id, zone_num) est FIXE et configurée par l'agriculteur.
  • Si un capteur hardware tombe en panne, on le remplace par un nouveau
    hardware configuré avec la MÊME adresse → zéro impact sur le reste.
  • Si une zone est supprimée, les autres zones ne changent PAS de numéro.

Structure Firestore :
  users/{uid}/zones/zone{N}/
    ├── humidity, temperature, ph, ec, n, p, k   ← moyennes agrégées
    ├── sensor_count                               ← nb capteurs actifs
    ├── last_updated
    ├── sensors/
    │     ├── {sensor_id} → { humidity, temperature, ph, ec, n, p, k,
    │     │                   timestamp, active }
    │     └── ...
    └── history/ → documents horodatés (valeurs agrégées)

Payload MQTT attendu :
  {
    "measurements": {
      "moisture_percent":        42.5,
      "temperature_celsius":     23.1,
      "ph":                       6.8,
      "conductivity_uS_per_cm": 380.0,
      "nutrients_mg_per_kg": {
        "nitrogen":   25.0,
        "phosphorus": 12.0,
        "potassium":  18.0
      }
    },
    "meta": {
      "node_id": "node_01",
      "mac":     "AA:BB:CC:DD:EE:FF",
      "rssi":    -72,
      "snr":      8.5
    }
  }
"""

import time

from firebase_admin import firestore

import config as cfg
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
    """
    Extrait les mesures depuis le payload structuré.

    Retourne (humidity, temperature, ph, ec, n, p, k)
    ou lève ValueError si 'measurements' est absent.
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
    """
    Retourne True si le capteur a envoyé des données récemment.
    Si SENSOR_STALE_SECONDS est None → tous considérés actifs.
    """
    if cfg.SENSOR_STALE_SECONDS is None:
        return True

    ts = sensor_data.get("timestamp")
    if ts is None:
        return True   # document tout neuf, pas encore de timestamp serveur

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
    Traite un message MQTT d'un capteur identifié par son adresse fixe.

    Si le capteur hardware a été remplacé par un nouveau configuré
    avec la même adresse (sensor_id), cette fonction fonctionne
    exactement comme avant — aucune action spéciale requise.

    Étapes :
      1. Parse le payload (measurements / nutrients_mg_per_kg).
      2. Écrase le document sensors/{sensor_id} avec les nouvelles valeurs
         (set() sans merge → les données de l'ancien hardware sont remplacées).
      3. Relit tous les capteurs de la zone, exclut les périmés.
      4. Calcule la moyenne et met à jour le document zone.
      5. Retourne (humidity, temperature, ph, ec, n, p, k) agrégées,
         ou (None,)*7 si le payload est invalide.

    Args:
        db         : client Firestore
        uid        : identifiant utilisateur
        zone_num   : numéro de zone tel que configuré (ex. "1", "2")
        payload    : dict JSON reçu via MQTT
        sensor_id  : adresse fixe du capteur (ex. "11", "12")
                     → si le hardware est remplacé, le nouveau reçoit
                       la même adresse : rien ne change côté serveur.
    """

    # ── 1. Parse du payload ──────────────────────────────────────────────────
    try:
        humidity, temperature, ph, ec, n, p, k = _parse_payload(
            payload, zone_num, sensor_id
        )
    except (ValueError, Exception) as exc:
        logger.error(
            "Payload invalide — zone %s / capteur %s : %s",
            zone_num, sensor_id, exc,
        )
        return (None,) * 7

    zone_ref   = (
        db.collection("users")
          .document(uid)
          .collection("zones")
          .document(f"zone{zone_num}")
    )
    sensor_ref = zone_ref.collection("sensors").document(sensor_id)

    # ── 2. Écraser les données du capteur à son adresse fixe ─────────────────
    #
    #  set() sans merge=True remplace complètement le document.
    #  → Si c'est un nouveau hardware sur la même adresse, les anciennes
    #    données du capteur défaillant sont proprement écrasées.
    #  → Le champ 'active' est toujours True : le capteur vient d'envoyer.
    #
    try:
        sensor_ref.set({
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
        })
        logger.debug(
            "Zone %s | Capteur %s → H=%.1f T=%.1f pH=%.2f EC=%.1f N=%.1f P=%.1f K=%.1f",
            zone_num, sensor_id, humidity, temperature, ph, ec, n, p, k,
        )
    except Exception as exc:
        logger.error(
            "Erreur sauvegarde capteur %s / zone %s : %s",
            sensor_id, zone_num, exc,
        )

    # ── 3. Relire tous les capteurs et calculer la moyenne zone ───────────────
    #
    #  On lit tous les documents de sensors/, on exclut :
    #    - les capteurs marqués active=False (désactivés manuellement)
    #    - les capteurs dont le timestamp dépasse SENSOR_STALE_SECONDS
    #
    count = 1
    try:
        sensor_docs    = zone_ref.collection("sensors").get()
        fresh_readings = []

        for doc in sensor_docs:
            data = doc.to_dict()
            if not data:
                continue

            # Exclusion explicite si désactivé manuellement
            if not data.get("active", True):
                logger.debug(
                    "Zone %s | Capteur %s exclu (désactivé manuellement)",
                    zone_num, doc.id,
                )
                continue

            # Exclusion si données périmées (capteur silencieux)
            if not _is_sensor_fresh(data):
                logger.debug(
                    "Zone %s | Capteur %s exclu (silence > %ds)",
                    zone_num, doc.id, cfg.SENSOR_STALE_SECONDS,
                )
                continue

            fresh_readings.append(data)

        if fresh_readings:
            count = len(fresh_readings)

            def avg(field: str) -> float:
                return sum(_float(r.get(field)) for r in fresh_readings) / count

            humidity    = avg("humidity")
            temperature = avg("temperature")
            ph          = avg("ph")
            ec          = avg("ec")
            n           = avg("n")
            p           = avg("p")
            k           = avg("k")

            logger.info(
                "Zone %s | %d capteur(s) actif(s) → "
                "H=%.1f T=%.1f pH=%.2f EC=%.1f N=%.1f P=%.1f K=%.1f",
                zone_num, count, humidity, temperature, ph, ec, n, p, k,
            )
        else:
            logger.warning(
                "Zone %s | Aucun capteur actif/frais — utilisation payload brut",
                zone_num,
            )

    except Exception as exc:
        logger.error(
            "Erreur lecture capteurs zone %s : %s — utilisation payload brut",
            zone_num, exc,
        )

    # ── 4. Mise à jour du document zone avec les valeurs agrégées ─────────────
    try:
        zone_ref.set(
            {
                "humidity":     humidity,
                "temperature":  temperature,
                "ph":           ph,
                "ec":           ec,
                "n":            n,
                "p":            p,
                "k":            k,
                "sensor_count": count,
                "last_updated": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
    except Exception as exc:
        logger.error("Erreur mise à jour zone %s : %s", zone_num, exc)

    return humidity, temperature, ph, ec, n, p, k


# ── Désactivation manuelle d'un capteur ──────────────────────────────────────

def deactivate_sensor(db, uid: str, zone_num: str, sensor_id: str) -> None:
    """
    Marque un capteur comme inactif (active=False).
    Il sera exclu des calculs de moyenne jusqu'à ce qu'il renvoie des données
    (ce qui remettra active=True automatiquement via update_sensor_data).

    Cas d'usage : capteur confirmé en panne, en attente de remplacement.
    Dès que le nouveau hardware envoie son premier message sur la même
    adresse, il redevient actif automatiquement — aucune autre action requise.
    """
    try:
        (
            db.collection("users")
              .document(uid)
              .collection("zones")
              .document(f"zone{zone_num}")
              .collection("sensors")
              .document(sensor_id)
              .update({
                  "active":       False,
                  "deactivated_at": firestore.SERVER_TIMESTAMP,
              })
        )
        logger.info(
            "Zone %s | Capteur %s marqué inactif (en attente de remplacement)",
            zone_num, sensor_id,  
        )
    except Exception as exc:
        logger.error(
            "Erreur désactivation capteur %s / zone %s : %s",
            sensor_id, zone_num, exc,
        )