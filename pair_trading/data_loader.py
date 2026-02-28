from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def get_valid_tickers(df: pd.DataFrame, min_obs: int) -> list[str]:
    counts = df.groupby("ticker")["date"].nunique()
    return counts[counts >= min_obs].index.tolist()


def get_price_matrix(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    valid_tickers = get_valid_tickers(df, min_obs)
    sub = df[df["ticker"].isin(valid_tickers)]
    px = (
        sub.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        .sort_index()
        .ffill(limit=3)
    )
    px = px.dropna(axis=1, how="all")
    return np.log(px)


def get_returns_matrix(log_prices: pd.DataFrame) -> pd.DataFrame:
    return log_prices.diff().dropna(how="all")
