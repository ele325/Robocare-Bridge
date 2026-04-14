# -*- coding: utf-8 -*-
"""
services/chatbot_service.py — Chatbot IA multilingue RoboCare
Correction v6.1 : champs n/p/k harmonisés + sante valeur par défaut 0
"""

import time
import re
import threading

from groq import Groq
from firebase_admin import firestore

import config as cfg
from utils.logger import get_logger

logger = get_logger("robocare.services.chatbot")

groq_client = Groq(api_key=cfg.GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Détection de langue
# ─────────────────────────────────────────────────────────────────────────────

_FR_WORDS = {
    "bonjour","salut","bonsoir","merci","oui","non","allo",
    "je","tu","il","elle","nous","vous","ils","elles",
    "mon","ton","son","notre","votre","leur",
    "le","la","les","un","une","des","du",
    "mais","donc","avec","pour","dans","sur","par","sans",
    "que","qui","quoi","dont",
    "est","sont","avoir","faire","aller","venir",
    "aime","parle","aide","veux","peux","dois",
    "pas","plus","tres","très","bien","mal","vite",
    "toujours","jamais","aussi","trop","encore",
    "comment","pourquoi","quelle","quel","quand","combien",
    "zone","humidite","humidité","plante","arrosage",
    "irrigation","culture","sante","conseil",
    "temperature","capteur","sol","engrais",
    "francais","anglais","langue",
    "depeche","toi","moi","lui",
    "switch","reponds","repond",
    "etat","affiche","montre","donne","explique",
}

_SYSTEM_PROMPTS = {
    "ar": (
        "أنت RoboCare AI، خبير في الزراعة الذكية.\n"
        "قاعدة صارمة: أجب باللغة العربية فقط، 2 إلى 3 جمل قصيرة.\n"
        "لا تستخدم الفرنسية أو الإنجليزية أبداً."
    ),
    "fr": (
        "Tu es RoboCare AI, expert en agriculture intelligente.\n"
        "RÈGLE ABSOLUE : réponds UNIQUEMENT en français, 2 à 3 phrases max.\n"
        "N'utilise jamais l'anglais ni l'arabe."
    ),
    "en": (
        "You are RoboCare AI, smart agriculture expert.\n"
        "ABSOLUTE RULE: respond ONLY in English, 2 to 3 sentences max.\n"
        "Never use French or Arabic."
    ),
}


def detect_language(text: str) -> tuple[str, str]:
    if any("\u0600" <= c <= "\u06FF" for c in text):
        return "ar", "Arabic"
    if any(w in text.lower().split() for w in _FR_WORDS):
        return "fr", "Français"
    return "en", "English"


# ─────────────────────────────────────────────────────────────────────────────
# Appel Groq
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(prompt: str, system_prompt: str) -> str | None:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]
    for attempt in range(cfg.GROQ_MAX_RETRIES):
        try:
            response = groq_client.chat.completions.create(
                model       = cfg.GROQ_MODEL,
                messages    = messages,
                max_tokens  = cfg.GROQ_MAX_TOKENS,
                temperature = cfg.GROQ_TEMPERATURE,
            )
            logger.info("Groq a répondu (tentative %d)", attempt + 1)
            return response.choices[0].message.content
        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate" in err.lower():
                wait  = 10
                match = re.search(r"retry[^\d]*(\d+)", err)
                if match:
                    wait = int(match.group(1)) + 2
                logger.warning(
                    "Quota Groq — attente %ds (tentative %d/%d)",
                    wait, attempt + 1, cfg.GROQ_MAX_RETRIES,
                )
                time.sleep(wait)
            else:
                logger.error("Erreur Groq : %s", exc)
                break
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Logique principale du chatbot
# ─────────────────────────────────────────────────────────────────────────────

def handle_chatbot_logic(db, uid: str, chat_id: str, user_message: str) -> None:
    time.sleep(0.5)
    try:
        lang_code, lang_name = detect_language(user_message)
        logger.info("Langue détectée : %s pour → '%s'", lang_name, user_message)

        # Contexte capteurs
        zones_ref = db.collection("users").document(uid).collection("zones").get()

        lines = []
        for z in zones_ref:
            d = z.to_dict()
            # ── CORRECTION v6.1 : champs n/p/k + sante avec valeur par défaut 0
            lines.append(
                "- {}: humidity={}%, pH={}, EC={} uS/cm, "
                "N={}, P={}, K={}, health={}/10".format(
                    z.id,
                    d.get("humidity",    "--"),
                    d.get("ph",          "--"),
                    d.get("ec",          "--"),
                    d.get("n",           "--"),   # corrigé : était azote
                    d.get("p",           "--"),   # corrigé : était phosphore
                    d.get("k",           "--"),   # corrigé : était potassium
                    d.get("sante",       0),      # corrigé : valeur par défaut 0
                )
            )

        context = "\n".join(lines) if lines else "No sensor data available."

        user_prompt = "Current sensor data:\n{}\n\nUser question: {}".format(
            context, user_message
        )
        reply = _call_groq(user_prompt, _SYSTEM_PROMPTS[lang_code])

        if reply and len(reply.strip()) > 2:
            db.collection("users").document(uid) \
              .collection("chats").document(chat_id) \
              .collection("messages").add({
                  "text":      reply.strip(),
                  "sender":    "ai",
                  "timestamp": firestore.SERVER_TIMESTAMP,
              })
            logger.info("Réponse envoyée en : %s", lang_name)
        else:
            logger.warning("Groq n'a pas retourné de réponse utilisable")

    except Exception as exc:
        logger.exception("Erreur chatbot : %s", exc)


def handle_chatbot_async(db, uid: str, chat_id: str, user_message: str) -> None:
    threading.Thread(
        target=handle_chatbot_logic,
        args=(db, uid, chat_id, user_message),
        daemon=True,
    ).start()