from __future__ import annotations

import numpy as np
import pandas as pd


def calc_ols_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 5:
        return 0.0, 1.0
    Y = df.iloc[:, 0].values
    X = np.column_stack([np.ones(len(df)), df.iloc[:, 1].values])
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    return float(coef[0]), float(coef[1])


def calc_spread(p1: pd.Series, p2: pd.Series, beta: float, alpha: float) -> pd.Series:
    df = pd.concat([p1, p2], axis=1).dropna()
    spread = df.iloc[:, 0] - beta * df.iloc[:, 1] - alpha
    return spread


def calc_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std(ddof=0).replace(0, np.nan)
    z = (spread - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def calc_half_life(spread: pd.Series) -> float:
    s = spread.dropna()
    if len(s) < 20:
        return 999.0
    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    df = pd.concat([delta, lag], axis=1).dropna()
    if len(df) < 10:
        return 999.0
    Y = df.iloc[:, 0].values
    X = np.column_stack([np.ones(len(df)), df.iloc[:, 1].values])
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    lam = float(coef[1])
    if lam >= 0:
        return 999.0
    hl = -np.log(2) / lam
    if not np.isfinite(hl) or hl <= 0:
        return 999.0
    return float(hl)


def rolling_beta(p1: pd.Series, p2: pd.Series, window: int = 120, step: int = 20) -> pd.DataFrame:
    df = pd.concat([p1, p2], axis=1).dropna()
    rows = []
    for end in range(window, len(df) + 1, step):
        chunk = df.iloc[end - window : end]
        alpha, beta = calc_ols_hedge_ratio(chunk.iloc[:, 0], chunk.iloc[:, 1])
        rows.append({"date": str(chunk.index[-1].date()), "beta": beta, "alpha": alpha})
    return pd.DataFrame(rows)


def beta_stability(p1: pd.Series, p2: pd.Series, window: int = 120) -> float:
    rb = rolling_beta(p1, p2, window=window, step=max(1, window // 6))
    if len(rb) == 0:
        return 0.0
    return max(0.0, 1 - min(1.0, float(rb["beta"].std(ddof=0) * 2)))


def rolling_half_life(spread: pd.Series, window: int = 120, step: int = 20) -> pd.DataFrame:
    s = spread.dropna()
    rows = []
    for end in range(window, len(s) + 1, step):
        chunk = s.iloc[end - window : end]
        rows.append({"date": str(chunk.index[-1].date()), "half_life": calc_half_life(chunk)})
    return pd.DataFrame(rows)


def half_life_trend(rhl_series: list) -> str:
    vals = [v for v in rhl_series if pd.notna(v)]
    if len(vals) < 4:
        return "stable"
    prev = vals[-4]
    curr = vals[-1]
    if prev <= 0:
        return "stable"
    change = (curr - prev) / prev
    if change > 0.2:
        return "increasing"
    if change < -0.2:
        return "decreasing"
    return "stable"
