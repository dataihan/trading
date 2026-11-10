"""Round 7: crowding signals for the cross-sectional carry edge (R7-1 premium, R7-2 composite, R7-3 resid reversal).

All functions return DAILY target weights (row t uses only data <= close of day t, no negative shifts);
wrap with carry.weekly(...) for the 7-day rebalance schedule. Legs: +0.5 gross long the bottom `frac`,
-0.5 gross short the top `frac`, equal weight (same convention as carry.fcarry_daily).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading.strategies.carry import _legs, ew_return, weekly


def _ls(score: pd.DataFrame, frac: float, min_coins: int) -> pd.DataFrame:
    """Long lowest-score `frac`, short highest-score `frac`, 0.5 gross per leg."""
    lo, hi, k = _legs(score, frac, min_coins)
    return (lo.astype(float).div(k, axis=0) * 0.5 - hi.astype(float).div(k, axis=0) * 0.5).fillna(0.0)


def premium(close: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    """perp_close / spot_close - 1 at the same daily close (NaN where spot is missing)."""
    sp = spot.reindex(index=close.index, columns=close.columns)
    return close / sp - 1.0


def _spot_ok(close: pd.DataFrame, spot: pd.DataFrame, min_hist: int) -> pd.DataFrame:
    """Eligible at t: >= min_hist days of history on BOTH perp and spot, and spot close present at t."""
    sp = spot.reindex(index=close.index, columns=close.columns)
    return (close.notna().cumsum() >= min_hist) & (sp.notna().cumsum() >= min_hist) & sp.notna()


def premium_daily(close, spot, lookback: int = 7, frac: float = 0.2, min_hist: int = 60,
                  min_coins: int = 8) -> pd.DataFrame:
    """R7-1: short top `frac` by 7d mean premium (t-6..t), long bottom `frac`."""
    s = premium(close, spot).rolling(lookback).mean().where(_spot_ok(close, spot, min_hist))
    return _ls(s, frac, min_coins)


def premium_carry(close, spot, lookback=7, frac=0.2, every=7, min_hist=60) -> pd.DataFrame:
    return weekly(premium_daily(close, spot, lookback, frac, min_hist), every)


def composite_daily(close, spot, fund, lookback: int = 7, frac: float = 0.2, min_hist: int = 60,
                    z_window: int = 60, min_coins: int = 8) -> pd.DataFrame:
    """R7-2: mean of cross-sectional pct-ranks of (a) 7d funding sum, (b) 7d mean premium,
    (c) z of the 7d funding sum vs its own trailing `z_window` mean/std (window ends at t).
    Ranked only among coins with all three present (and R7-1 eligibility)."""
    f7 = fund.reindex(close.index).rolling(lookback).sum()
    mu = f7.rolling(z_window).mean()
    sd = f7.rolling(z_window).std().replace(0.0, np.nan)
    z = (f7 - mu) / sd
    prem = premium(close, spot).rolling(lookback).mean()
    ok = _spot_ok(close, spot, min_hist) & f7.notna() & prem.notna() & z.notna()
    a, b, c = f7.where(ok), prem.where(ok), z.where(ok)
    score = (a.rank(axis=1, pct=True) + b.rank(axis=1, pct=True) + c.rank(axis=1, pct=True)) / 3.0
    return _ls(score, frac, min_coins)


def composite_carry(close, spot, fund, lookback=7, frac=0.2, every=7, min_hist=60) -> pd.DataFrame:
    return weekly(composite_daily(close, spot, fund, lookback, frac, min_hist, z_window=min_hist), every)


def resid_reversal_daily(close, lookback: int = 7, frac: float = 0.2, min_hist: int = 60,
                         beta_window: int = 60, min_coins: int = 8) -> pd.DataFrame:
    """R7-3: beta_i,t = OLS beta of r_i on EW return over t-59..t; resid = sum_{t-6..t}(r_i - beta_i,t * r_EW).
    Long the bottom `frac` residual (losers), short the top `frac` (winners)."""
    r = close.pct_change(fill_method=None)
    m = ew_return(close)
    beta = r.rolling(beta_window).cov(m).div(m.rolling(beta_window).var(), axis=0)
    rs = r.rolling(lookback).sum()
    ms = m.rolling(lookback).sum()
    resid = rs - beta.mul(ms, axis=0)                    # = sum(r_i) - beta_i,t * sum(r_EW), beta fixed at t
    score = resid.where(close.notna().cumsum() >= min_hist)
    return _ls(score, frac, min_coins)


def resid_reversal(close, lookback=7, frac=0.2, every=7, min_hist=60) -> pd.DataFrame:
    return weekly(resid_reversal_daily(close, lookback, frac, min_hist), every)
