from __future__ import annotations

import numpy as np
import pandas as pd


def calc_ols_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    joined = pd.concat([y, x], axis=1).dropna()
    yv = joined.iloc[:, 0].values
    xv = joined.iloc[:, 1].values
    x_design = np.column_stack([np.ones(len(xv)), xv])
    alpha, beta = np.linalg.lstsq(x_design, yv, rcond=None)[0]
    return float(alpha), float(beta)


def calc_spread(p1: pd.Series, p2: pd.Series, beta: float, alpha: float) -> pd.Series:
    joined = pd.concat([p1, p2], axis=1).dropna()
    return joined.iloc[:, 0] - beta * joined.iloc[:, 1] - alpha


def calc_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std().replace(0, np.nan)
    return ((spread - mean) / std).replace([np.inf, -np.inf], np.nan)


def calc_half_life(spread: pd.Series) -> float:
    s = spread.dropna()
    if len(s) < 20:
        return 999.0
    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    joined = pd.concat([delta, lag], axis=1).dropna()
    if len(joined) < 20:
        return 999.0
    y = joined.iloc[:, 0].values
    x = joined.iloc[:, 1].values
    x_design = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(x_design, y, rcond=None)[0]
    lam = coef[1]
    if lam >= 0:
        return 999.0
    hl = -np.log(2) / lam
    if not np.isfinite(hl) or hl <= 0:
        return 999.0
    return float(min(999.0, hl))


def rolling_beta(p1: pd.Series, p2: pd.Series, window: int = 120, step: int = 20) -> pd.DataFrame:
    joined = pd.concat([p1, p2], axis=1).dropna()
    rows = []
    for i in range(window, len(joined) + 1, step):
        seg = joined.iloc[i - window : i]
        alpha, beta = calc_ols_hedge_ratio(seg.iloc[:, 0], seg.iloc[:, 1])
        rows.append({"date": seg.index[-1], "beta": beta, "alpha": alpha})
    return pd.DataFrame(rows)


def beta_stability(p1: pd.Series, p2: pd.Series, window: int = 120) -> float:
    rb = rolling_beta(p1, p2, window=window, step=max(1, window // 6))
    if rb.empty:
        return 0.0
    return float(max(0.0, 1 - min(1.0, rb["beta"].std() * 2)))


def rolling_half_life(spread: pd.Series, window: int = 120, step: int = 20) -> pd.DataFrame:
    spread = spread.dropna()
    rows = []
    for i in range(window, len(spread) + 1, step):
        seg = spread.iloc[i - window : i]
        rows.append({"date": seg.index[-1], "half_life": calc_half_life(seg)})
    return pd.DataFrame(rows)


def half_life_trend(rhl_series: list) -> str:
    if len(rhl_series) < 4:
        return "stable"
    last = rhl_series[-1]
    prev = rhl_series[-4]
    if prev == 0:
        return "stable"
    change = (last - prev) / abs(prev)
    if change > 0.2:
        return "increasing"
    if change < -0.2:
        return "decreasing"
    return "stable"
