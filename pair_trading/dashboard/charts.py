from __future__ import annotations

import pandas as pd


def downsample_series(series: pd.Series, step: int = 3) -> tuple[list[str], list[float]]:
    s = series.dropna()
    s = s.iloc[::step]
    labels = [str(i.date()) if hasattr(i, "date") else str(i) for i in s.index]
    vals = [float(v) for v in s.values]
    return labels, vals


def prepare_heatmap_data(corr_matrix: pd.DataFrame, top_n: int = 20) -> dict:
    cols = list(corr_matrix.columns[:top_n])
    data = corr_matrix.loc[cols, cols].fillna(0.0).round(3).values.tolist()
    return {"labels": cols, "data": data}
