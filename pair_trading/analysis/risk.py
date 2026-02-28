from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_extremes(zscore: pd.Series) -> dict:
    z = zscore.dropna()
    if len(z) == 0:
        return {"z_max": 0.0, "z_min": 0.0, "max_excursion": 0.0, "percentile_95": 0.0, "percentile_5": 0.0}
    return {
        "z_max": float(z.max()),
        "z_min": float(z.min()),
        "max_excursion": float(max(abs(z.max()), abs(z.min()))),
        "percentile_95": float(np.percentile(z, 95)),
        "percentile_5": float(np.percentile(z, 5)),
    }


def longest_extreme_streak(zscore: pd.Series, threshold: float = 2.0) -> int:
    flags = (zscore.abs() > threshold).fillna(False).astype(int).tolist()
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def max_adverse_move(zscore: pd.Series, current_signal: str) -> float:
    z = zscore.dropna()
    if len(z) == 0:
        return 0.0
    if current_signal in {"BUY", "BUY_WEAK", "WATCH_BUY"}:
        return float(z.min())
    if current_signal in {"SELL", "SELL_WEAK", "WATCH_SELL"}:
        return float(z.max())
    return float(max(abs(z.max()), abs(z.min())))


def estimated_max_loss(zscore: pd.Series, current_z: float, signal: str) -> float:
    adv = max_adverse_move(zscore, signal)
    return float(abs(current_z - adv))


def risk_score(metrics: dict) -> str:
    risk_points = 0
    if metrics.get("sharpe", 0) < 0:
        risk_points += 2
    elif metrics.get("sharpe", 0) < 0.5:
        risk_points += 1

    if metrics.get("win_rate", 0) < 0.45:
        risk_points += 2
    elif metrics.get("win_rate", 0) < 0.55:
        risk_points += 1

    if metrics.get("max_dd", 0) < -0.2:
        risk_points += 2
    elif metrics.get("max_dd", 0) < -0.1:
        risk_points += 1

    if metrics.get("streak", 0) > 20:
        risk_points += 2
    elif metrics.get("streak", 0) > 10:
        risk_points += 1

    if abs(metrics.get("adverse_move", 0)) > 4:
        risk_points += 2
    elif abs(metrics.get("adverse_move", 0)) > 3:
        risk_points += 1

    if risk_points >= 8:
        return "EXTREME"
    if risk_points >= 5:
        return "HIGH"
    if risk_points >= 3:
        return "MEDIUM"
    return "LOW"
