from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_extremes(zscore: pd.Series) -> dict:
    z = zscore.dropna()
    if z.empty:
        return {"z_max": 0, "z_min": 0, "max_excursion": 0, "percentile_95": 0, "percentile_5": 0}
    return {
        "z_max": float(z.max()),
        "z_min": float(z.min()),
        "max_excursion": float(max(abs(z.max()), abs(z.min()))),
        "percentile_95": float(np.percentile(z, 95)),
        "percentile_5": float(np.percentile(z, 5)),
    }


def longest_extreme_streak(zscore: pd.Series, threshold: float = 2.0) -> int:
    streak = best = 0
    for val in zscore.fillna(0.0):
        if abs(val) > threshold:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def max_adverse_move(zscore: pd.Series, current_signal: str) -> float:
    z = zscore.dropna()
    if z.empty:
        return 0.0
    if current_signal.startswith("BUY"):
        return float(z.min())
    if current_signal.startswith("SELL"):
        return float(z.max())
    return float(max(abs(z.max()), abs(z.min())))


def estimated_max_loss(zscore: pd.Series, current_z: float, signal: str) -> float:
    adverse = max_adverse_move(zscore, signal)
    return float(abs(current_z - adverse))


def risk_score(metrics: dict) -> str:
    risk_points = 0
    if metrics.get("sharpe", 0) < 0:
        risk_points += 1
    if metrics.get("win_rate", 0) < 0.45:
        risk_points += 1
    if metrics.get("max_dd", 0) < -0.2:
        risk_points += 1
    if metrics.get("streak", 0) > 8:
        risk_points += 1
    if abs(metrics.get("adverse_move", 0)) > 3.5:
        risk_points += 1
    return ["LOW", "MEDIUM", "HIGH", "EXTREME"][min(3, risk_points)]
