from __future__ import annotations

import numpy as np
import pandas as pd


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(beta[0]), float(beta[1])


def adf_test(series: pd.Series) -> tuple[float, float]:
    s = series.dropna().astype(float)
    if len(s) < 20:
        return 0.0, 0.5

    y = s.values
    dy = np.diff(y)
    y_lag = y[:-1]
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    coef, *_ = np.linalg.lstsq(X, dy, rcond=None)
    resid = dy - X @ coef
    dof = max(1, len(dy) - X.shape[1])
    sigma2 = (resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    tau = float(coef[1] / se[1]) if se[1] > 0 else 0.0

    if tau < -3.43:
        p = 0.01
    elif tau < -2.86:
        p = 0.05
    elif tau < -2.57:
        p = 0.10
    elif tau < -2.00:
        p = 0.20
    else:
        p = 0.50
    return tau, p


def engle_granger_test(y: pd.Series, x: pd.Series) -> dict:
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]
    if len(df) < 20:
        return {"pval": 0.5, "tau": 0.0, "beta": np.nan, "intercept": np.nan, "spread": pd.Series(dtype=float)}
    alpha, beta = _ols(df["y"].values, df["x"].values)
    spread = df["y"] - (alpha + beta * df["x"])
    tau, pval = adf_test(spread)
    return {"pval": pval, "tau": tau, "beta": beta, "intercept": alpha, "spread": spread}


def rolling_adf(spread: pd.Series, window: int, step: int) -> pd.DataFrame:
    s = spread.dropna()
    rows = []
    for end in range(window, len(s) + 1, step):
        chunk = s.iloc[end - window : end]
        tau, pval = adf_test(chunk)
        rows.append(
            {
                "date": str(chunk.index[-1].date()) if hasattr(chunk.index[-1], "date") else str(chunk.index[-1]),
                "tau": tau,
                "pval": pval,
                "is_cointegrated": pval <= 0.10,
            }
        )
    return pd.DataFrame(rows)


def is_recently_cointegrated(spread: pd.Series, window: int = 120, n_recent: int = 3) -> bool:
    rdf = rolling_adf(spread, window=window, step=max(1, window // 6))
    if len(rdf) < n_recent:
        return False
    return bool((rdf.tail(n_recent)["pval"] <= 0.10).all())
