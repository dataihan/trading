"""Strategy library. Every function takes an OHLCV DataFrame (UTC index) + params
and returns a position Series in [-1, 1] decided at each bar's CLOSE
(the engine applies it to the NEXT bar's return). Some strategies also return a
custom per-bar return series via the `bar_ret` key.

Rule: only use information available at bar close t. No .shift(-k) on inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _hold(signal: pd.Series, bars: int) -> pd.Series:
    """Keep each non-zero entry signal active for `bars` bars (latest signal wins)."""
    s = signal.replace(0, np.nan)
    return s.ffill(limit=max(bars - 1, 0)).fillna(0.0)


def _vol_scale(pos: pd.Series, close: pd.Series, target_ann: float | None, lookback: int,
               ppy: float, cap: float = 2.0) -> pd.Series:
    if not target_ann:
        return pos
    vol = close.pct_change().rolling(lookback).std() * np.sqrt(ppy)
    return (pos * (target_ann / vol).clip(upper=cap)).fillna(0.0)


# ---------------------------------------------------------------- trend
def buy_hold(df):
    return {"position": pd.Series(1.0, index=df.index)}


def tsmom(df, lookback=24 * 30, long_only=False, rebalance=24, vol_target=None,
          vol_lb=24 * 30, ppy=8760, threshold=0.0):
    r = df.close / df.close.shift(lookback) - 1
    pos = np.sign(r).where(r.abs() > threshold, 0.0)
    if long_only:
        pos = pos.clip(lower=0)
    pos = pos.iloc[::rebalance].reindex(df.index).ffill().fillna(0.0)
    return {"position": _vol_scale(pos, df.close, vol_target, vol_lb, ppy)}


def ma_cross(df, fast=24 * 10, slow=24 * 50, long_only=True, vol_target=None, ppy=8760):
    f, s = df.close.rolling(fast).mean(), df.close.rolling(slow).mean()
    pos = pd.Series(np.where(f > s, 1.0, -1.0), index=df.index)
    pos[s.isna()] = 0.0
    if long_only:
        pos = pos.clip(lower=0)
    return {"position": _vol_scale(pos, df.close, vol_target, slow, ppy)}


def donchian(df, entry=24 * 20, exit=24 * 10, long_only=False):
    hi = df.high.rolling(entry).max().shift(1)
    lo = df.low.rolling(entry).min().shift(1)
    xlo = df.low.rolling(exit).min().shift(1)
    xhi = df.high.rolling(exit).max().shift(1)
    c = df.close.values
    pos = np.zeros(len(df))
    p = 0.0
    for i in range(len(df)):
        if p == 0:
            if c[i] > hi.iat[i]:
                p = 1.0
            elif not long_only and c[i] < lo.iat[i]:
                p = -1.0
        elif p > 0 and c[i] < xlo.iat[i]:
            p = 0.0
        elif p < 0 and c[i] > xhi.iat[i]:
            p = 0.0
        pos[i] = p
    return {"position": pd.Series(pos, index=df.index)}


# ---------------------------------------------------------------- mean reversion
def zscore_revert(df, lookback=48, entry=2.0, exit=0.0, long_only=False, trend_filter=None):
    m = df.close.rolling(lookback).mean()
    sd = df.close.rolling(lookback).std()
    z = ((df.close - m) / sd).values
    tf = None if not trend_filter else (df.close > df.close.rolling(trend_filter).mean()).values
    pos = np.zeros(len(df))
    p = 0.0
    for i in range(len(df)):
        zi = z[i]
        if np.isnan(zi):
            pos[i] = 0
            continue
        if p == 0:
            if zi < -entry and (tf is None or tf[i]):
                p = 1.0
            elif zi > entry and not long_only and (tf is None or not tf[i]):
                p = -1.0
        elif p > 0 and zi > exit:
            p = 0.0
        elif p < 0 and zi < -exit:
            p = 0.0
        pos[i] = p
    return {"position": pd.Series(pos, index=df.index)}


def ibs(df, low=0.2, high=None, long_only=True, trend_filter=200):
    """Internal Bar Strength (daily): long when close near the day's low."""
    ibs_ = (df.close - df.low) / (df.high - df.low).replace(0, np.nan)
    pos = (ibs_ < low).astype(float)
    if trend_filter:
        pos = pos * (df.close > df.close.rolling(trend_filter).mean())
    if not long_only and high is not None:
        pos = pos - (ibs_ > high).astype(float)
    return {"position": pos.fillna(0.0)}


# ---------------------------------------------------------------- calendar / session
def hour_of_day(df, long_hours=(), short_hours=(), weekdays=None):
    """Hold position during bar t+1 if bar (t+1) starts in a chosen UTC hour."""
    nxt_hour = pd.Series(df.index.hour, index=df.index).shift(-1)  # known schedule, not data
    nxt_wd = pd.Series(df.index.dayofweek, index=df.index).shift(-1)
    pos = nxt_hour.isin(long_hours).astype(float) - nxt_hour.isin(short_hours).astype(float)
    if weekdays is not None:
        pos = pos * nxt_wd.isin(weekdays)
    return {"position": pos.fillna(0.0)}


def overnight(df, cost_bps=1.0):
    """Daily bars: buy at close, sell next open (equity indices/ETFs)."""
    ret = df.open.shift(-1) / df.close - 1 - 2 * cost_bps / 1e4
    return {"position": pd.Series(1.0, index=df.index), "bar_ret": ret, "no_turnover_cost": True}


def turn_of_month(df, before=1, after=3):
    idx = df.index
    month = idx.tz_localize(None).to_period("M")
    pos_in_month = pd.Series(1, index=idx).groupby(month).cumcount().values
    n_in_month = pd.Series(1, index=idx).groupby(month).transform("count").values
    from_end = n_in_month - pos_in_month  # 1 = last day
    hold = (from_end <= before) | (pos_in_month < after)
    # decide at close t to hold bar t+1
    pos = pd.Series(hold.astype(float), index=idx).shift(-1).fillna(0.0)
    return {"position": pos}


def prevday_sweep(df, hold=6, side="both", reclaim=True):
    """ICT-style liquidity sweep of previous UTC day's high/low.
    Long: bar trades below prev-day low and closes back above it -> long `hold` bars.
    Short: mirror at prev-day high."""
    day = df.index.floor("D")
    dh = df.high.groupby(day).max()
    dl = df.low.groupby(day).min()
    pdh = dh.shift(1).reindex(day).values
    pdl = dl.shift(1).reindex(day).values
    pdh = pd.Series(pdh, index=df.index)
    pdl = pd.Series(pdl, index=df.index)
    if reclaim:
        long_sig = (df.low < pdl) & (df.close > pdl)
        short_sig = (df.high > pdh) & (df.close < pdh)
    else:  # breakout continuation
        long_sig = df.close > pdh
        short_sig = df.close < pdl
    sig = long_sig.astype(float) * (side in ("both", "long")) - \
        short_sig.astype(float) * (side in ("both", "short"))
    return {"position": _hold(sig, hold)}


def session_breakout(df, range_start=0, range_end=7, trade_until=21, mode="breakout", side="both"):
    """Asia-range (UTC hours [range_start, range_end)) breakout or fade, flat by `trade_until`."""
    h = df.index.hour
    day = df.index.floor("D")
    in_rng = (h >= range_start) & (h < range_end)
    rh = df.high.where(in_rng).groupby(day).transform("max")
    rl = df.low.where(in_rng).groupby(day).transform("min")
    active = (h >= range_end) & (h < trade_until)
    up = (df.close > rh) & active
    dn = (df.close < rl) & active
    if mode == "fade":
        up, dn = dn, up
    sig = up.astype(float) * (side in ("both", "long")) - dn.astype(float) * (side in ("both", "short"))
    sig = sig.where(sig != 0)
    # first signal of the day sticks until trade_until
    first = sig.groupby(day).transform(lambda s: s.ffill())
    pos = first.where(active | (h == trade_until - 1)).fillna(0.0)
    pos[h >= trade_until - 1] = 0.0  # be flat for the bar starting at trade_until
    return {"position": pos}


REGISTRY = {f.__name__: f for f in [buy_hold, tsmom, ma_cross, donchian, zscore_revert, ibs, hour_of_day,
                                    overnight, turn_of_month, prevday_sweep, session_breakout]}


def ml_gbm(df, symbol=None, **kw):
    """Walk-forward LightGBM direction model (see trading.ml.walkforward)."""
    from trading.ml.walkforward import walk_forward
    funding = None
    if symbol:
        from trading.data.loaders import binance_funding
        funding = binance_funding(symbol)
    pos, _ = walk_forward(df, funding=funding, **kw)
    return {"position": pos}


REGISTRY["ml_gbm"] = ml_gbm


def trend_ensemble(df, lookbacks=(7, 14, 30, 60, 90, 180), unit=24, long_only=True,
                   vol_target=0.4, vol_lb=30, rebalance=24, ppy=8760, cap=1.5):
    """Average of sign(momentum) over several lookbacks (in `unit` bars), vol-targeted.
    Averaging lookbacks avoids picking a single 'best' parameter (less overfit)."""
    sig = sum(np.sign(df.close / df.close.shift(L * unit) - 1) for L in lookbacks) / len(lookbacks)
    if long_only:
        sig = sig.clip(lower=0)
    vol = df.close.pct_change().rolling(vol_lb * unit).std() * np.sqrt(ppy)
    pos = (sig * (vol_target / vol).clip(upper=cap)).fillna(0.0)
    pos = pos.iloc[::rebalance].reindex(df.index).ffill().fillna(0.0)
    return {"position": pos}


REGISTRY["trend_ensemble"] = trend_ensemble


# ---------------------------------------------------------------- round 2: ML filters
def _primary(df, primary, primary_params):
    return REGISTRY[primary](df, **(primary_params or {}))["position"]


def meta_label(df, primary="trend_ensemble", primary_params=None, symbol=None, horizon=24,
               rebalance=24, retrain=24 * 30, min_train=24 * 365, daily=False, **kw):
    """Primary rule picks the side; walk-forward LightGBM decides whether to take the bet."""
    from trading.ml.meta import meta_filter
    pos = _primary(df, primary, primary_params)
    funding = None
    if symbol:
        from trading.data.loaders import binance_funding
        funding = binance_funding(symbol)
    decision = (np.arange(len(df)) % rebalance) == 0
    out, _ = meta_filter(df, pos, horizon, decision, retrain, min_train, funding=funding,
                         daily=daily, **kw)
    return {"position": out}


def regime_filter(df, primary="trend_ensemble", primary_params=None, n_states=3, refit_every=63,
                  min_train=500, hard=False):
    """Scale a primary strategy by a walk-forward HMM regime multiplier computed on daily closes."""
    from trading.ml.meta import hmm_regime
    pos = _primary(df, primary, primary_params)
    daily_close = df.close.groupby(df.index.floor("D")).last()
    mult = hmm_regime(daily_close, n_states=n_states, refit_every=refit_every, min_train=min_train)
    # multiplier known at end of day D -> apply from the first bar after day D closes
    m = mult.shift(1).reindex(df.index.floor("D")).values
    m = pd.Series(m, index=df.index).fillna(1.0)
    if hard:
        m = (m > 0.5).astype(float)
    return {"position": pos * m}


REGISTRY["meta_label"] = meta_label
REGISTRY["regime_filter"] = regime_filter


# ---------------------------------------------------------------- round 3d: next-H-bars ML veto on trend
def trend_ml_filter(df, symbol, h=24, margin=0.0, **trend_kw):
    """Trend ensemble (long/flat) picks exposure; a walk-forward LightGBM forecast of the next `h` bars
    can only veto it: if P(up) < 0.5 - margin at a decision bar, be flat until the next decision bar."""
    from trading.ml.multibar import proba_1h
    base = trend_ensemble(df, long_only=True, **trend_kw)["position"]
    p = proba_1h(df, symbol, h)
    dec = pd.Series((np.arange(len(df)) % h) == 0, index=df.index)
    gate = (~(p < 0.5 - margin)).astype(float).where(dec & p.notna()).ffill().fillna(1.0)
    return {"position": base * gate}


REGISTRY["trend_ml_filter"] = trend_ml_filter
