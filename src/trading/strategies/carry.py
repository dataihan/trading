"""Cross-sectional funding carry (round 5): short the highest trailing funding, long the lowest."""
from __future__ import annotations

import pandas as pd


def weekly(w: pd.DataFrame, every: int = 7) -> pd.DataFrame:
    return w.iloc[::every].reindex(w.index).ffill()


def fcarry_daily(close: pd.DataFrame, fund: pd.DataFrame, lookback: int = 7, frac: float = 0.2,
                 min_hist: int = 60, min_coins: int = 8) -> pd.DataFrame:
    """Target weights for EVERY day (no rebalance schedule): +0.5 gross long the lowest `frac`
    trailing funding, -0.5 gross short the highest. Row t uses data up to the close of day t."""
    hist = close.notna().cumsum()
    score = fund.rolling(lookback).sum().where(hist >= min_hist)
    n = score.notna().sum(axis=1)
    k = (n * frac).round().clip(lower=1)
    hi = score.rank(axis=1, ascending=False).le(k, axis=0)
    lo = score.rank(axis=1, ascending=True).le(k, axis=0)
    w = lo.astype(float).div(k, axis=0) * 0.5 - hi.astype(float).div(k, axis=0) * 0.5
    return w.where(n >= min_coins, 0.0)


def fcarry(close: pd.DataFrame, fund: pd.DataFrame, lookback: int = 7, frac: float = 0.2, every: int = 7,
           min_hist: int = 60) -> pd.DataFrame:
    return weekly(fcarry_daily(close, fund, lookback, frac, min_hist), every)
