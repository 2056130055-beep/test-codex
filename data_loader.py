from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "ticker", "open", "high", "low", "close", "volume"}


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"]).copy()
    df = df[df["close"] > 0]
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def get_valid_tickers(df: pd.DataFrame, min_obs: int) -> list[str]:
    counts = df.groupby("ticker")["close"].size()
    return counts[counts >= min_obs].index.tolist()


def get_price_matrix(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    valid_tickers = get_valid_tickers(df, min_obs)
    prices = (
        df[df["ticker"].isin(valid_tickers)]
        .pivot(index="date", columns="ticker", values="close")
        .sort_index()
        .ffill(limit=3)
    )
    prices = prices.dropna(axis=1, how="any")
    prices = prices.dropna(axis=0, how="any")
    return np.log(prices)


def get_returns_matrix(log_prices: pd.DataFrame) -> pd.DataFrame:
    return log_prices.diff().dropna(how="all")
