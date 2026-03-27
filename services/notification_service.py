# -*- coding: utf-8 -*-
"""
services/notification_service.py — Notifications FCM RoboCare
"""

from firebase_admin import firestore, messaging
from ml.predictor import PredictionResult
from utils.logger import get_logger
from utils.decorators import retry

logger = get_logger("robocare.services.notifications")


@retry(max_attempts=3, delay=5)
def send_critical_alert(db, uid: str, zone_num: str | int, humidity: float) -> None:
    """Alerte immédiate humidité critique (< seuil)."""
    topic   = "user_{}".format(uid)
    message = messaging.Message(
        notification=messaging.Notification(
            title="🔴 Alerte Zone {}".format(zone_num),
            body="Humidité critique : {:.1f}%. Irrigation activée.".format(humidity),
        ),
        data={"zone": str(zone_num), "click_action": "FLUTTER_NOTIFICATION_CLICK"},
        topic=topic,
    )
    messaging.send(message)

    db.collection("users").document(uid).collection("alerts").add({
        "type":      "low_humidity",
        "level":     "critique",
        "zone_num":  str(zone_num),
        "humidity":  round(humidity, 1),
        "timestamp": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Notification critique envoyée — UID: %s Zone: %s H: %.1f%%", uid, zone_num, humidity)


@retry(max_attempts=3, delay=5)
def send_irrigation_prediction(db, uid: str, zone_num: str | int, result: PredictionResult) -> None:
    """
    Notification basée sur la prédiction ML combinée.
    Inclut l'intervalle de confiance et le R² dans les données.
    """
    if result.score >= 80:
        niveau = "🔴 URGENT"
    elif result.score >= 65:
        niveau = "🟠 Important"
    else:
        niveau = "🟡 Recommandé"

    body = (
        "Score: {}/100 | H→{:.1f}% [IC: {:.1f}–{:.1f}] | "
        "T→{:.1f}°C | EC→{:.0f}µS | N:{:.1f} P:{:.1f} K:{:.1f} | R²={:.2f}"
    ).format(
        result.score,
        result.pred_humidity, result.ci_humidity_low, result.ci_humidity_high,
        result.pred_temp,
        result.pred_ec,
        result.pred_n, result.pred_p, result.pred_k,
        result.r2_humidity,
    )

    message = messaging.Message(
        notification=messaging.Notification(
            title="{} Irrigation Zone {}".format(niveau, zone_num),
            body=body,
        ),
        data={
            "zone":         str(zone_num),
            "score":        str(result.score),
            "click_action": "FLUTTER_NOTIFICATION_CLICK",
        },
        topic="user_{}".format(uid),
    )
    messaging.send(message)

    doc = result.to_firestore_dict()
    doc["zone_num"] = str(zone_num)
    db.collection("users").document(uid).collection("predictions").add(doc)

    logger.info(
        "Notification ML envoyée — Zone %s Score %d/100 IC:[%.1f, %.1f] R²=%.2f",
        zone_num, result.score, result.ci_humidity_low, result.ci_humidity_high, result.r2_humidity,
    )