# -*- coding: utf-8 -*-
"""
ml/health_score.py — Score de santé agronomique du sol
"""

from utils.logger import get_logger
import config as cfg

logger = get_logger("robocare.ml.health_score")


def calculate_health_score(
    h: float, ph: float, ec: float,
    n: float, p: float, k: float,
) -> int:
    """
    Calcule un score de santé du sol entre 1 et 10.

    Paramètres
    ----------
    h   : humidité (%)
    ph  : pH
    ec  : conductivité électrique (µS/cm)
    n   : azote (mg/kg)
    p   : phosphore (mg/kg)
    k   : potassium (mg/kg)

    Retourne
    --------
    int : score [1–10]
    """
    score = 10.0

    if h  < cfg.HEALTH_HUMIDITY_MIN or h  > cfg.HEALTH_HUMIDITY_MAX: score -= 2.0
    if ph < cfg.HEALTH_PH_MIN       or ph > cfg.HEALTH_PH_MAX:       score -= 2.0
    if ec < cfg.HEALTH_EC_MIN:                                        score -= 1.5
    if n  < cfg.HEALTH_N_MIN:                                         score -= 1.5
    if p  < cfg.HEALTH_P_MIN:                                         score -= 1.5
    if k  < cfg.HEALTH_K_MIN:                                         score -= 1.5

    final = max(1, round(score))
    logger.debug(
        "Score santé calculé : %d/10 (H=%.1f%% pH=%.1f EC=%.0f N=%.1f P=%.1f K=%.1f)",
        final, h, ph, ec, n, p, k,
    )
    return final