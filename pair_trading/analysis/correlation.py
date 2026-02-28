from __future__ import annotations

import pandas as pd


def full_correlation(r1: pd.Series, r2: pd.Series) -> float:
    df = pd.concat([r1, r2], axis=1).dropna()
    if len(df) < 2:
        return 0.0
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def rolling_correlation(r1: pd.Series, r2: pd.Series, window: int = 250) -> pd.Series:
    df = pd.concat([r1, r2], axis=1)
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1]).dropna()


def correlation_stability(r1: pd.Series, r2: pd.Series, window: int = 250) -> float:
    rc = rolling_correlation(r1, r2, window=window)
    if len(rc) == 0:
        return 0.0
    stability = 1 - min(1.0, float(rc.std(ddof=0) * 2))
    return max(0.0, stability)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def filter_high_correlation_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.50) -> list[tuple]:
    cols = list(corr_matrix.columns)
    pairs: list[tuple] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr_matrix.iloc[i, j]
            if pd.notna(c) and c >= threshold:
                pairs.append((cols[i], cols[j], float(c)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs
