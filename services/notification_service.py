# -*- coding: utf-8 -*-
"""
services/notification_service.py — Notifications FCM RoboCare
"""

import json

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


@retry(max_attempts=3, delay=5)
def send_threshold_breach_alert(
    db,
    uid: str,
    zone_num: str | int,
    field: str,
    value: float,
    minimum: float,
    maximum: float,
) -> None:
    """Notification quand une mesure sort des seuils plante/current."""
    topic = "user_{}".format(uid)
    message = messaging.Message(
        notification=messaging.Notification(
            title="Alerte seuil Zone {}".format(zone_num),
            body=(
                "{} hors seuil: {:.1f} (min {:.1f} / max {:.1f})".format(
                    field.upper(), value, minimum, maximum
                )
            ),
        ),
        data={
            "zone": str(zone_num),
            "field": field,
            "value": str(round(value, 2)),
            "min": str(round(minimum, 2)),
            "max": str(round(maximum, 2)),
            "click_action": "FLUTTER_NOTIFICATION_CLICK",
        },
        topic=topic,
    )
    messaging.send(message)

    db.collection("users").document(uid).collection("alerts").add({
        "type": "threshold_breach",
        "level": "warning",
        "zone_num": str(zone_num),
        "field": field,
        "value": round(value, 2),
        "min": round(minimum, 2),
        "max": round(maximum, 2),
        "timestamp": firestore.SERVER_TIMESTAMP,
    })

    logger.info(
        "Alerte seuil envoyée — UID: %s Zone: %s Champ: %s Valeur: %.2f [%.2f-%.2f]",
        uid, zone_num, field, value, minimum, maximum,
    )


@retry(max_attempts=3, delay=5)
def send_thresholds_summary_alert(
    db,
    uid: str,
    zone_num: str | int,
    plant_type: str | None,
    issues: list[dict],
) -> None:
    """
    Envoie une notification unique résumant tous les paramètres hors seuil.
    """
    if not issues:
        return

    topic = "user_{}".format(uid)
    issue_count = len(issues)
    has_humidity = any(i.get("field") == "humidity" for i in issues)

    if has_humidity and issue_count >= 2:
        severity = "critical"
        badge = "🔴"
    elif issue_count >= 2:
        severity = "warning"
        badge = "🟠"
    else:
        severity = "info"
        badge = "🟡"

    plant_label = (plant_type or "Plante").strip()
    title = "{} Zone {} — Alerte {}".format(badge, zone_num, plant_label)

    max_lines = 2
    details = []
    for issue in issues[:max_lines]:
        field = str(issue.get("field", "")).upper()
        value = float(issue.get("value", 0.0))
        minimum = float(issue.get("min", 0.0))
        maximum = float(issue.get("max", 0.0))
        if value < minimum:
            details.append("• {}: {:.1f} (min {:.1f})".format(field, value, minimum))
        else:
            details.append("• {}: {:.1f} (max {:.1f})".format(field, value, maximum))

    extra = ""
    if issue_count > max_lines:
        extra = "\n• +{} autre(s) paramètre(s)".format(issue_count - max_lines)

    body = "{} paramètre(s) hors plage\n{}{}\nConseil: vérifier irrigation et nutriments.".format(
        issue_count,
        "\n".join(details),
        extra,
    )

    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        data={
            "zone": str(zone_num),
            "plant_type": plant_label,
            "severity": severity,
            "issues_count": str(issue_count),
            "issues_json": json.dumps(issues, ensure_ascii=True),
            "click_action": "FLUTTER_NOTIFICATION_CLICK",
        },
        topic=topic,
    )
    messaging.send(message)

    db.collection("users").document(uid).collection("alerts").add({
        "type": "threshold_breach_summary",
        "level": severity,
        "zone_num": str(zone_num),
        "plant_type": plant_label,
        "issues_count": issue_count,
        "issues": issues,
        "timestamp": firestore.SERVER_TIMESTAMP,
    })

    logger.info(
        "Notification résumé seuils envoyée — UID: %s Zone: %s Nb: %d Niveau: %s",
        uid, zone_num, issue_count, severity,
    )


@retry(max_attempts=3, delay=5)
def send_irrigation_finished_alert(db, uid: str, zone_num: str | int) -> None:
    """Notification quand l'arrosage temporisé est terminé."""
    topic = "user_{}".format(uid)
    message = messaging.Message(
        notification=messaging.Notification(
            title="✅ Arrosage Terminé - Zone {}".format(zone_num),
            body="L'arrosage de la zone {} est fini. La pompe est maintenant fermée.".format(zone_num),
        ),
        data={
            "zone": str(zone_num),
            "type": "irrigation_finished",
            "click_action": "FLUTTER_NOTIFICATION_CLICK"
        },
        android=messaging.AndroidConfig(
            priority='high',
            notification=messaging.AndroidNotification(
                channel_id='smartfarm_alerts',
                vibrate_timings_millis=[0, 500, 200, 500], # Vibration attirante
            ),
        ),
        topic=topic,
    )
    messaging.send(message)
    logger.info("Notification FINISHED envoyée — UID: %s Zone: %s", uid, zone_num)


@retry(max_attempts=3, delay=5)
def send_sensor_failure_alert(db, uid: str, zone_num: str | int, sensor_id: str) -> None:
    """Notification quand un capteur tombe en panne (stale)."""
    topic = "user_{}".format(uid)
    message = messaging.Message(
        notification=messaging.Notification(
            title="⚠️ Panne Capteur - Zone {}".format(zone_num),
            body="Le capteur {} ne répond plus. Veuillez vérifier l'installation.".format(sensor_id),
        ),
        data={
            "zone": str(zone_num),
            "sensor": sensor_id,
            "type": "sensor_failure",
            "click_action": "FLUTTER_NOTIFICATION_CLICK"
        },
        android=messaging.AndroidConfig(
            priority='high',
            notification=messaging.AndroidNotification(
                channel_id='smartfarm_alerts',
            ),
        ),
        topic=topic,
    )
    messaging.send(message)
    logger.info("Notification PANNE envoyée — UID: %s Zone: %s Sensor: %s", uid, zone_num, sensor_id)