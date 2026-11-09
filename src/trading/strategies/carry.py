"""Cross-sectional funding carry (round 5): short the highest trailing funding, long the lowest."""
from __future__ import annotations

import numpy as np
import pandas as pd


def weekly(w: pd.DataFrame, every: int = 7, offset: int = 0) -> pd.DataFrame:
    """Hold the weights of every `every`-th row (starting at `offset`). offset=0 is the original behaviour;
    for offset>0 the rows before the first rebalance are flat (0)."""
    out = w.iloc[offset::every].reindex(w.index).ffill()
    return out if offset == 0 else out.fillna(0.0)


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


# ---------------------------------------------------------------- Round 6 (robustness, nothing tuned)
def tranche(w_daily: pd.DataFrame, every: int = 7) -> pd.DataFrame:
    """R6-1 helper: average of `every` sleeves, each rebalanced on a different phase (offset 0..every-1)."""
    return sum(weekly(w_daily, every, o) for o in range(every)) / every


def fcarry_tranched(close: pd.DataFrame, fund: pd.DataFrame, lookback: int = 7, frac: float = 0.2,
                    min_hist: int = 60, every: int = 7) -> pd.DataFrame:
    """R6-1: phase-tranched fcarry = (1/7)·Σ_o weekly(fcarry_daily, 7, offset o)."""
    return tranche(fcarry_daily(close, fund, lookback, frac, min_hist), every)


def eligible(close: pd.DataFrame, qv: pd.DataFrame, fund: pd.DataFrame, min_hist: int = 60, top_n: int = 20,
             liq_window: int = 30, zero_max: float = 0.2) -> pd.DataFrame:
    """R6-2 mask at t (trailing data only): hist >= min_hist AND top `top_n` by trailing 30d median
    quote_volume AND share of days with funding == 0 exactly over trailing 30d <= zero_max."""
    hist = close.notna().cumsum()
    liq = qv.rolling(liq_window).median()
    top = liq.rank(axis=1, ascending=False).le(top_n)
    zero = (fund == 0).astype(float).where(fund.notna()).rolling(liq_window).mean()
    return (hist >= min_hist) & top & zero.le(zero_max)


def _legs(score: pd.DataFrame, frac: float, min_coins: int):
    n = score.notna().sum(axis=1)
    k = (n * frac).round().clip(lower=1)
    hi = score.rank(axis=1, ascending=False).le(k, axis=0)
    lo = score.rank(axis=1, ascending=True).le(k, axis=0)
    ok = n >= min_coins
    return lo & ok.values[:, None], hi & ok.values[:, None], k


def fcarry_filtered_daily(close, qv, fund, lookback: int = 7, frac: float = 0.2, min_hist: int = 60,
                          min_coins: int = 8, top_n: int = 20) -> pd.DataFrame:
    """Daily (un-tranched) R6-2 weights: fcarry ranked only within the eligible set, ±0.5 gross per leg."""
    score = fund.rolling(lookback).sum().where(eligible(close, qv, fund, min_hist, top_n))
    lo, hi, k = _legs(score, frac, min_coins)
    return lo.astype(float).div(k, axis=0) * 0.5 - hi.astype(float).div(k, axis=0) * 0.5


def fcarry_filtered(close, qv, fund, lookback: int = 7, frac: float = 0.2, min_hist: int = 60,
                    every: int = 7, top_n: int = 20) -> pd.DataFrame:
    """R6-2: liquidity + data-hygiene filter on R6-1 (tranched)."""
    return tranche(fcarry_filtered_daily(close, qv, fund, lookback, frac, min_hist, top_n=top_n), every)


def ew_return(close: pd.DataFrame) -> pd.Series:
    """Equal-weight universe daily return (close t-1 -> t), over coins with a price at both ends."""
    return close.pct_change(fill_method=None).mean(axis=1)


def fcarry_beta_neutral_daily(close, qv, fund, lookback: int = 7, frac: float = 0.2, min_hist: int = 60,
                              min_coins: int = 8, top_n: int = 20, vol_window: int = 60,
                              leg_cap: float = 0.75) -> pd.DataFrame:
    """Daily R6-3 weights. Legs = R6-2 legs. Within a leg w ∝ 1/σ60. β_i = 60d OLS beta to EW return.
    Leg gross: short share s solves s·β_S = (1-s)·β_L (Σw·β = 0), clipped to [1-cap, cap], total gross 1.
    Falls back to 0.5/0.5 when a leg beta is non-positive or missing. All rolling stats end at t."""
    score = fund.rolling(lookback).sum().where(eligible(close, qv, fund, min_hist, top_n))
    lo, hi, _ = _legs(score, frac, min_coins)
    r = close.pct_change(fill_method=None)
    m = ew_return(close)
    mp = max(vol_window // 2, 2)
    sig = r.rolling(vol_window, min_periods=mp).std()
    beta = r.rolling(vol_window, min_periods=mp).cov(m).div(m.rolling(vol_window, min_periods=mp).var(), axis=0)
    inv = (1.0 / sig.replace(0, np.nan))
    inv = inv.apply(lambda row: row.fillna(row.median()), axis=1)   # missing vol -> cross-sectional median

    def leg(mask):
        x = inv.where(mask)
        return x.div(x.sum(axis=1), axis=0).fillna(0.0)             # sums to 1 within the leg (or 0)

    wl, ws = leg(lo), leg(hi)
    bl = (wl * beta.fillna(1.0)).sum(axis=1).where(lo.any(axis=1))
    bs = (ws * beta.fillna(1.0)).sum(axis=1).where(hi.any(axis=1))
    s = (bl / (bl + bs)).where((bl > 0) & (bs > 0), 0.5).clip(1 - leg_cap, leg_cap)
    w = wl.mul(1 - s, axis=0) - ws.mul(s, axis=0)
    return w.where(lo.any(axis=1) & hi.any(axis=1), 0.0).fillna(0.0)


def fcarry_beta_neutral(close, qv, fund, lookback: int = 7, frac: float = 0.2, min_hist: int = 60,
                        every: int = 7, top_n: int = 20) -> pd.DataFrame:
    """R6-3: beta-neutral, inverse-vol fcarry on R6-2 (tranched)."""
    return tranche(fcarry_beta_neutral_daily(close, qv, fund, lookback, frac, min_hist, top_n=top_n), every)
