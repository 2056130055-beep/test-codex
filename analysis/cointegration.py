from __future__ import annotations

import numpy as np
import pandas as pd


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    x_design = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(x_design, y, rcond=None)[0]
    return float(alpha), float(beta)


def adf_test(series: pd.Series) -> tuple[float, float]:
    s = series.dropna().values
    if len(s) < 20:
        return 0.0, 0.5
    y = s[1:] - s[:-1]
    x = s[:-1]
    x_design = np.column_stack([np.ones(len(x)), x])
    coef, residuals, _, _ = np.linalg.lstsq(x_design, y, rcond=None)
    if len(residuals) == 0:
        return 0.0, 0.5
    sigma2 = residuals[0] / max(1, len(x) - x_design.shape[1])
    cov = sigma2 * np.linalg.inv(x_design.T @ x_design)
    se = np.sqrt(np.diag(cov))
    tau = coef[1] / se[1] if se[1] > 0 else 0.0

    if tau < -3.43:
        pval = 0.01
    elif tau < -2.86:
        pval = 0.05
    elif tau < -2.57:
        pval = 0.10
    elif tau < -2.00:
        pval = 0.20
    else:
        pval = 0.50
    return float(tau), pval


def engle_granger_test(y: pd.Series, x: pd.Series) -> dict:
    aligned = pd.concat([y, x], axis=1).dropna()
    aligned.columns = ["y", "x"]
    alpha, beta = _ols(aligned["y"].values, aligned["x"].values)
    spread = aligned["y"] - (alpha + beta * aligned["x"])
    tau, pval = adf_test(spread)
    return {
        "pval": pval,
        "tau": tau,
        "beta": beta,
        "intercept": alpha,
        "spread": spread,
    }


def rolling_adf(spread: pd.Series, window: int, step: int) -> pd.DataFrame:
    spread = spread.dropna()
    rows = []
    for i in range(window, len(spread) + 1, step):
        seg = spread.iloc[i - window : i]
        tau, pval = adf_test(seg)
        rows.append(
            {
                "date": seg.index[-1],
                "tau": tau,
                "pval": pval,
                "is_cointegrated": pval <= 0.10,
            }
        )
    return pd.DataFrame(rows)


def is_recently_cointegrated(spread: pd.Series, window: int = 120, n_recent: int = 3) -> bool:
    rdf = rolling_adf(spread, window=window, step=max(1, window // n_recent))
    if len(rdf) < n_recent:
        return False
    return bool((rdf.tail(n_recent)["pval"] <= 0.10).all())
