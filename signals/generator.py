from __future__ import annotations

from config import Config


def generate_signal(pair_metrics: dict) -> str:
    pval = pair_metrics["pval"]
    zscore = pair_metrics["zscore"]
    recently_cointegrated = pair_metrics["recent_coint_stable"]

    if pval <= 0.10 and recently_cointegrated:
        if zscore < -Config.ENTRY_ZSCORE:
            return "BUY"
        if zscore > Config.ENTRY_ZSCORE:
            return "SELL"
        if zscore < -1.5:
            return "WATCH_BUY"
        if zscore > 1.5:
            return "WATCH_SELL"
        return "NEUTRAL"
    if pval <= 0.10:
        if zscore < -Config.ENTRY_ZSCORE:
            return "BUY_WEAK"
        if zscore > Config.ENTRY_ZSCORE:
            return "SELL_WEAK"
        return "NEUTRAL"
    return "WEAK"
