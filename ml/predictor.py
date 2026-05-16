# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from firebase_admin import firestore

import config as cfg
from utils.logger import get_logger

logger = get_logger("robocare.ml.predictor")

# 1. Structures de données
@dataclass
class AnomalyReport:
    variable:        str
    n_anomalies:     int
    anomaly_indices: List[int]
    cleaned_values:  np.ndarray

    @property
    def has_anomalies(self) -> bool:
        return self.n_anomalies > 0


@dataclass
class PredictionResult:
    is_dangerous:    bool
    score:           int
    reason:          str
    pred_humidity:   float = 0.0
    pred_temp:       float = 0.0
    pred_ec:         float = 0.0
    pred_n:          float = 0.0
    pred_p:          float = 0.0
    pred_k:          float = 0.0
    trend_h:         float = 0.0
    trend_t:         float = 0.0
    trend_ec:        float = 0.0
    trend_n:         float = 0.0
    trend_p:         float = 0.0
    trend_k:         float = 0.0
    ci_humidity_low:  float = 0.0
    ci_humidity_high: float = 0.0
    r2_humidity:     float = 0.0
    anomalies:       List[AnomalyReport] = field(default_factory=list)

    def to_firestore_dict(self) -> dict:
        return {
            "type":            "irrigation_combined",
            "score":           self.score,
            "raison":          self.reason,
            "humidity":        self.pred_humidity,
            "temperature":     self.pred_temp,
            "ec":              self.pred_ec,
            "pred_humidity":   self.pred_humidity,
            "pred_temp":       self.pred_temp,
            "pred_ec":         self.pred_ec,
            "trend_humidity":  self.trend_h,
            "trend_temp":      self.trend_t,
            "trend_ec":        self.trend_ec,
            "pred_n":          self.pred_n,
            "pred_p":          self.pred_p,
            "pred_k":          self.pred_k,
            "trend_n":         self.trend_n,
            "trend_p":         self.trend_p,
            "trend_k":         self.trend_k,
            "ci_humidity_low":  self.ci_humidity_low,
            "ci_humidity_high": self.ci_humidity_high,
            "r2_humidity":     self.r2_humidity,
            "n_anomalies":     sum(a.n_anomalies for a in self.anomalies),
            "timestamp":       firestore.SERVER_TIMESTAMP,
        }


# 2. detect_anomalies
def detect_anomalies(values: np.ndarray, variable: str) -> AnomalyReport:
    if len(values) < 4:
        return AnomalyReport(variable, 0, [], values.copy())

    # Vérifier l'écart-type pour éviter division par zéro (séries constantes)
    std = np.std(values)
    if std == 0:
        return AnomalyReport(variable, 0, [], values.copy())

    z_scores = np.abs(stats.zscore(values))
    anomaly_indices = list(np.where(z_scores > cfg.ML_ZSCORE_THRESHOLD)[0])
    cleaned = values.copy()

    if not anomaly_indices:
        return AnomalyReport(variable, 0, [], cleaned)

    median = float(np.median(values))
    n_anomalies = len(anomaly_indices)
    consecutive = False
    for i in range(1, n_anomalies):
        if anomaly_indices[i] == anomaly_indices[i-1] + 1:
            consecutive = True
            break

    ratio = n_anomalies / len(values)

    if n_anomalies == 1:
        cleaned[anomaly_indices[0]] = median
        logger.info("[ANOMALIE] %s : valeur inhabituelle (indice %d)", variable, anomaly_indices[0])
    elif consecutive or ratio > 0.2:
        for idx in anomaly_indices:
            cleaned[idx] = median
        logger.warning("[ANOMALIE CRITIQUE] %s : %d anomalies répétées → capteur suspect", variable, n_anomalies)
    else:
        for idx in anomaly_indices:
            cleaned[idx] = median
        logger.info("[ANOMALIE] %s : anomalies ponctuelles (%d)", variable, n_anomalies)

    return AnomalyReport(variable, n_anomalies, anomaly_indices, cleaned)


# 3. _fit_predict  ← ICI AVANT predict_irrigation_combined
def _fit_predict(
    X: np.ndarray,
    y: np.ndarray,
    next_idx: int,
    degree: int = 2,
    alpha: float = 1.0,
) -> Tuple[float, float, float]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
    model.fit(X, y)
    pred  = float(model.predict([[next_idx]])[0])
    trend = float(model.predict([[next_idx]])[0] - model.predict([[next_idx - 1]])[0])
    r2    = float(r2_score(y, model.predict(X)))
    return pred, trend, r2


# 4. _bootstrap_ci  ← ICI AVANT predict_irrigation_combined
def _bootstrap_ci(
    X: np.ndarray,
    y: np.ndarray,
    next_idx: int,
    n_boot: int = 100,
    ci: float = 95.0,
    degree: int = 2,
    alpha: float = 1.0,
) -> Tuple[float, float]:
    preds = []
    n     = len(y)
    for _ in range(n_boot):
        idx    = np.random.choice(n, n, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        model  = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
        model.fit(X_boot, y_boot)
        preds.append(float(model.predict([[next_idx]])[0]))
    low  = float(np.percentile(preds, (100 - ci) / 2))
    high = float(np.percentile(preds, 100 - (100 - ci) / 2))
    return low, high


# 5. predict_irrigation_combined
def predict_irrigation_combined(
    db,
    uid:              str,
    zone_num:         str | int,
    current_humidity: float,
    current_temp:     float,
    current_ec:       float,
    current_n:        float = 0.0,
    current_p:        float = 0.0,
    current_k:        float = 0.0,
) -> PredictionResult:
    zone_id = "zone{}".format(zone_num)
    try:
        history = (
            db.collection("users").document(uid)
              .collection("zones").document(zone_id)
              .collection("measures")
              .order_by("timestamp", direction=firestore.Query.DESCENDING)
              .limit(cfg.ML_HISTORY_LIMIT)
              .get()
        )
        nb = len(history)
        logger.info("[ML] Zone %s — %d relevés trouvés", zone_num, nb)

        if nb < cfg.ML_MIN_HISTORY:
            logger.warning("[ML] Données insuffisantes (%d/%d)", nb, cfg.ML_MIN_HISTORY)
            return PredictionResult(False, 0, "Données insuffisantes ({}/{})".format(nb, cfg.ML_MIN_HISTORY))

        records = [d.to_dict() for d in history]
        records.reverse()

        raw_h  = np.array([r.get("humidity",    current_humidity) for r in records], dtype=float)
        raw_t  = np.array([r.get("temperature", current_temp)     for r in records], dtype=float)
        raw_ec = np.array([r.get("ec",          current_ec)       for r in records], dtype=float)
        raw_n  = np.array([r.get("n",           current_n)        for r in records], dtype=float)
        raw_p  = np.array([r.get("p",           current_p)        for r in records], dtype=float)
        raw_k  = np.array([r.get("k",           current_k)        for r in records], dtype=float)

        rep_h  = detect_anomalies(raw_h,  "humidity")
        rep_t  = detect_anomalies(raw_t,  "temperature")
        rep_ec = detect_anomalies(raw_ec, "ec")
        rep_n  = detect_anomalies(raw_n,  "n")
        rep_p  = detect_anomalies(raw_p,  "p")
        rep_k  = detect_anomalies(raw_k,  "k")

        anomaly_reports = [r for r in [rep_h, rep_t, rep_ec, rep_n, rep_p, rep_k] if r.has_anomalies]

        X        = np.arange(nb, dtype=float).reshape(-1, 1)
        next_idx = nb

        pred_h,  trend_h,  r2_h  = _fit_predict(X, rep_h.cleaned_values,  next_idx)
        pred_t,  trend_t,  _     = _fit_predict(X, rep_t.cleaned_values,  next_idx)
        pred_ec, trend_ec, _     = _fit_predict(X, rep_ec.cleaned_values, next_idx)
        pred_n,  trend_n,  _     = _fit_predict(X, rep_n.cleaned_values,  next_idx)
        pred_p,  trend_p,  _     = _fit_predict(X, rep_p.cleaned_values,  next_idx)
        pred_k,  trend_k,  _     = _fit_predict(X, rep_k.cleaned_values,  next_idx)

        ci_low, ci_high = _bootstrap_ci(X, rep_h.cleaned_values, next_idx)

        logger.info("[ML] Zone %s → H: %.1f%% (IC 95%%: [%.1f, %.1f] trend:%+.2f R²=%.2f)",
                    zone_num, pred_h, ci_low, ci_high, trend_h, r2_h)
        logger.info("[ML] Zone %s → T: %.1f°C | EC: %.0f | N: %.1f | P: %.1f | K: %.1f",
                    zone_num, pred_t, pred_ec, pred_n, pred_p, pred_k)

        score   = 0
        reasons = []

        if pred_h < cfg.HUMIDITY_CRITICAL:
            score += 50; reasons.append("💧 H prédite {:.1f}% critique".format(pred_h))
        elif pred_h < cfg.HUMIDITY_LOW:
            score += 35; reasons.append("💧 H prédite {:.1f}% basse".format(pred_h))
        elif pred_h < cfg.HUMIDITY_LIMIT:
            score += 15; reasons.append("💧 H prédite {:.1f}% limite".format(pred_h))

        if ci_low < cfg.HUMIDITY_CRITICAL and pred_h >= cfg.HUMIDITY_CRITICAL:
            score += 10; reasons.append("⚠️ IC bas {:.1f}% sous seuil critique".format(ci_low))

        if trend_h < cfg.TREND_H_FAST_DROP:
            score += 15; reasons.append("📉 Chute rapide {:.1f}%/cycle".format(trend_h))
        elif trend_h < cfg.TREND_H_DROP:
            score += 8;  reasons.append("📉 Baisse {:.1f}%/cycle".format(trend_h))

        if pred_t > cfg.TEMP_EXCESSIVE:
            score += 30; reasons.append("🌡️ T prédite {:.1f}°C excessive".format(pred_t))
        elif pred_t > cfg.TEMP_HIGH:
            score += 20; reasons.append("🌡️ T prédite {:.1f}°C élevée".format(pred_t))
        elif pred_t > cfg.TEMP_NORMAL:
            score += 10; reasons.append("🌡️ T prédite {:.1f}°C normale-haute".format(pred_t))

        if pred_ec < cfg.EC_CRITICAL:
            score += 20; reasons.append("⚡ EC prédite {:.0f} µS/cm critique".format(pred_ec))
        elif pred_ec < cfg.EC_LOW:
            score += 10; reasons.append("⚡ EC prédite {:.0f} µS/cm faible".format(pred_ec))
        if trend_ec < cfg.TREND_EC_DROP:
            score += 10; reasons.append("📉 EC chute {:.0f}/cycle".format(trend_ec))

        if pred_n < cfg.N_CRITICAL:
            score += 15; reasons.append("🌱 Azote prédit {:.1f} critique".format(pred_n))
        elif pred_n < cfg.N_LOW:
            score += 10; reasons.append("🌱 Azote prédit {:.1f} bas".format(pred_n))
        if trend_n < cfg.TREND_N_DROP:
            score += 5;  reasons.append("📉 Azote chute {:.1f}/cycle".format(trend_n))

        if pred_p < cfg.P_CRITICAL:
            score += 10; reasons.append("🌿 Phosphore prédit {:.1f} critique".format(pred_p))
        elif pred_p < cfg.P_LOW:
            score += 5;  reasons.append("🌿 Phosphore prédit {:.1f} bas".format(pred_p))

        if pred_k < cfg.K_CRITICAL:
            score += 10; reasons.append("🍃 Potassium prédit {:.1f} critique".format(pred_k))
        elif pred_k < cfg.K_LOW:
            score += 5;  reasons.append("🍃 Potassium prédit {:.1f} bas".format(pred_k))

        if anomaly_reports:
            score += 5
            for r in anomaly_reports:
                if r.n_anomalies == 1:
                    reasons.append("⚠️ Valeur inhabituelle ({})".format(r.variable))
                elif r.n_anomalies >= 3:
                    reasons.append("🚨 Capteur suspect ({})".format(r.variable))
                else:
                    reasons.append("⚠️ Anomalies ponctuelles ({})".format(r.variable))

        score        = min(score, 100)
        is_dangerous = score >= cfg.DANGEROUS_SCORE
        reason_str   = " | ".join(reasons) if reasons else "Conditions normales"

        logger.info("[SCORE] Zone %s → %d/100 | %s", zone_num, score, reason_str)

        return PredictionResult(
            is_dangerous     = is_dangerous,
            score            = score,
            reason           = reason_str,
            pred_humidity    = pred_h,
            pred_temp        = pred_t,
            pred_ec          = pred_ec,
            pred_n           = pred_n,
            pred_p           = pred_p,
            pred_k           = pred_k,
            trend_h          = trend_h,
            trend_t          = trend_t,
            trend_ec         = trend_ec,
            trend_n          = trend_n,
            trend_p          = trend_p,
            trend_k          = trend_k,
            ci_humidity_low  = ci_low,
            ci_humidity_high = ci_high,
            r2_humidity      = r2_h,
            anomalies        = anomaly_reports,
        )

    except Exception as exc:
        logger.exception("[ML] Erreur predict_irrigation_combined : %s", exc)
        return PredictionResult(False, 0, "Erreur interne ML")


# 6. predict_stress_risk
def predict_stress_risk(db, uid: str, zone_num: str | int, current_humidity: float) -> bool:
    try:
        zone_id = "zone{}".format(zone_num)
        history = (
            db.collection("users").document(uid)
              .collection("zones").document(zone_id)
              .collection("measures")
              .order_by("timestamp", direction=firestore.Query.DESCENDING)
              .limit(cfg.ML_STRESS_HISTORY_LIMIT)
              .get()
        )
        if len(history) < cfg.ML_STRESS_MIN_HISTORY:
            return False

        h_vals     = [d.to_dict().get("humidity", current_humidity) for d in history]
        h_vals.reverse() # Remettre en ordre chronologique (ancien -> récent)

        total_drop = sum(h_vals[i] - h_vals[i + 1] for i in range(len(h_vals) - 1))
        avg_drop   = total_drop / (len(h_vals) - 1)
        pred_next  = current_humidity - avg_drop

        if avg_drop > 1.0 and pred_next < cfg.HUMIDITY_LOW:
            logger.warning("[STRESS] Zone %s — tendance stress hydrique (drop moyen %.2f%%/cycle)", zone_num, avg_drop)
            return True
        return False

    except Exception as exc:
        logger.debug("[ML] Erreur predict_stress_risk : %s", exc)
        return False