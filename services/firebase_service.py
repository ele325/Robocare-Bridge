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
    Solution Optimisée PFE :
    1. Auto-création de l'arborescence (User > Zone).
    2. Calcul du score Santé (IA).
    3. Double archivage : Temps réel + Historique intelligent.
    4. Sous-document Analytics pour les prédictions futures.
    """
    try:
        # --- 1. RÉFÉRENCES (Architecture NoSQL) ---
        user_ref = db.collection("users").document(uid)
        zone_id = f"zone{zone_num}"
        doc_ref = user_ref.collection("zones").document(zone_id)
        ai_ref = doc_ref.collection("analytics").document("ai_results")

        # --- 2. EXTRACTION DES DONNÉES (MQTT Payload) ---
        m = payload.get("measurements", {})
        nutrients = m.get("nutrients_mg_per_kg", {})

        # Conversion forcée en float pour éviter les erreurs de type
        h   = float(m.get("moisture_percent",       0.0))
        t   = float(m.get("temperature_celsius",    0.0))
        ph  = float(m.get("ph",                     0.0))
        ec  = float(m.get("conductivity_uS_per_cm", 0.0))
        n   = float(nutrients.get("nitrogen",        0.0))
        p   = float(nutrients.get("phosphorus",      0.0))
        k   = float(nutrients.get("potassium",       0.0))

        # --- 3. INTELLIGENCE ARTIFICIELLE ---
        health_score = calculate_health_score(h, ph, ec, n, p, k)

        # Préparation du snapshot complet
        full_data = {
            "humidity":    h,
            "temperature": t,
            "ph":          ph,
            "ec":          ec,
            "azote":       n,
            "phosphore":   p,
            "potassium":   k,
            "health_score": health_score,  # Intégré directement
            "zone_num":    str(zone_num),
            "last_update": firestore.SERVER_TIMESTAMP
        }

        # --- 4. PERSISTENCE MULTI-NIVEAUX ---
        
        # A. Update User (pour savoir quand il s'est connecté la dernière fois)
        user_ref.set({"last_seen": firestore.SERVER_TIMESTAMP}, merge=True)

        # B. Update Zone (Valeurs "Live" pour l'écran d'accueil de l'app)
        doc_ref.set(full_data, merge=True)
        
        # C. Archive Historique (Pour tracer des graphiques Humidité/Santé dans Flutter)
        doc_ref.collection("history").add({
            **full_data,
            "timestamp": firestore.SERVER_TIMESTAMP 
        })

        # D. Update Analytics (Le "cerveau" pour les conseils d'arrosage)
        ai_ref.set({
            "current_health": health_score,
            "status": "Optimal" if health_score > 70 else "Alerte",
            "recommendation": "Arrosage requis" if h < 30 else "Conditions stables",
            "last_analysis": firestore.SERVER_TIMESTAMP
        }, merge=True)

        logger.info(f"✅ Synchronisation réussie : Zone {zone_num} | Score: {health_score}%")

        # --- 5. RETOUR (Strictement 6 valeurs pour mqtt_handler) ---
        return h, t, ec, n, p, k

    except Exception as e:
        logger.error(f"❌ Erreur critique Firebase : {e}")
        return None, None, None, None, None, None