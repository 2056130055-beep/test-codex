from __future__ import annotations

import pandas as pd


def downsample_series(series: pd.Series, step: int) -> tuple[list[str], list[float]]:
    s = series.dropna().iloc[::step]
    return [d.strftime("%Y-%m-%d") for d in s.index], [float(v) for v in s.values]
