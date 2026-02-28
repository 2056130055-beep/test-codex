from __future__ import annotations

import pandas as pd


def full_correlation(r1: pd.Series, r2: pd.Series) -> float:
    joined = pd.concat([r1, r2], axis=1).dropna()
    if len(joined) < 3:
        return 0.0
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def rolling_correlation(r1: pd.Series, r2: pd.Series, window: int = 250) -> pd.Series:
    joined = pd.concat([r1, r2], axis=1).dropna()
    return joined.iloc[:, 0].rolling(window).corr(joined.iloc[:, 1]).dropna()


def correlation_stability(r1: pd.Series, r2: pd.Series, window: int = 250) -> float:
    rc = rolling_correlation(r1, r2, window=window)
    if rc.empty:
        return 0.0
    return float(max(0.0, 1 - min(1.0, rc.std() * 2)))


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr().fillna(0.0)


def filter_high_correlation_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.50) -> list[tuple]:
    cols = corr_matrix.columns.tolist()
    pairs = []
    for i, t1 in enumerate(cols):
        for t2 in cols[i + 1 :]:
            corr = float(corr_matrix.loc[t1, t2])
            if corr >= threshold:
                pairs.append((t1, t2, corr))
    return sorted(pairs, key=lambda x: x[2], reverse=True)
