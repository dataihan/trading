"""Vectorised bar backtester.

Convention (no look-ahead):
  position[t] is decided with information up to and including the CLOSE of bar t,
  and earns the return of bar t+1 (close[t] -> close[t+1]) unless a custom
  `bar_ret` is supplied (e.g. overnight close->open returns).
Costs: `cost_bps` charged per unit of turnover (|pos[t]-pos[t-1]|), each side.
Funding (perps): `funding` series aligned to bars (rate paid by longs at that bar).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


def periods_per_year(index: pd.DatetimeIndex) -> float:
    dt = pd.Series(index).diff().median()
    hours = dt / pd.Timedelta(hours=1)
    if hours >= 20:  # daily bars: assume trading-day calendar if gaps on weekends
        span_days = (index[-1] - index[0]).days or 1
        return len(index) / span_days * 365.25
    return 365.25 * 24 / hours


@dataclass
class Result:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    n_trades: int
    exposure: float
    win_rate: float
    profit_factor: float
    avg_trade_bps: float
    bars: int

    def as_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def run(close: pd.Series, position: pd.Series, cost_bps: float = 6.0,
        bar_ret: pd.Series | None = None, funding: pd.Series | None = None):
    """Return (pnl_series, equity_series, Result)."""
    pos = position.reindex(close.index).fillna(0.0).clip(-3, 3)
    if bar_ret is None:
        bar_ret = close.pct_change().shift(-1)  # return earned AFTER decision at t
    gross = pos * bar_ret.fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * cost_bps / 1e4
    fund = 0.0
    if funding is not None:
        fund = pos * funding.reindex(close.index).shift(-1).fillna(0.0)
    pnl = (gross - cost - fund).fillna(0.0)
    equity = (1 + pnl).cumprod()
    return pnl, equity, stats(pnl, pos, close.index)


def _trades(pos: pd.Series, pnl: pd.Series) -> np.ndarray:
    """Group pnl into trades: a trade = consecutive bars with same non-zero sign."""
    sgn = np.sign(pos.values)
    trade_id = np.cumsum(np.r_[1, (sgn[1:] != sgn[:-1]).astype(int)])
    df = pd.DataFrame({"id": trade_id, "s": sgn, "p": pnl.values})
    df = df[df.s != 0]
    if df.empty:
        return np.array([])
    return df.groupby("id")["p"].apply(lambda x: np.prod(1 + x) - 1).values


def stats(pnl: pd.Series, pos: pd.Series, index: pd.DatetimeIndex) -> Result:
    ppy = periods_per_year(index)
    eq = (1 + pnl).cumprod()
    total = eq.iloc[-1] - 1 if len(eq) else 0.0
    years = max(len(pnl) / ppy, 1e-9)
    cagr = (eq.iloc[-1]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    sd = pnl.std()
    sharpe = pnl.mean() / sd * np.sqrt(ppy) if sd > 0 else 0.0
    dsd = pnl[pnl < 0].std()
    sortino = pnl.mean() / dsd * np.sqrt(ppy) if dsd and dsd > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
    tr = _trades(pos, pnl)
    wins, losses = tr[tr > 0], tr[tr < 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf") if len(wins) else 0.0
    return Result(
        total_return=float(total), cagr=float(cagr), sharpe=float(sharpe), sortino=float(sortino),
        max_dd=float(dd), calmar=float(cagr / -dd) if dd < 0 else 0.0, n_trades=int(len(tr)),
        exposure=float((pos != 0).mean()), win_rate=float(len(wins) / len(tr)) if len(tr) else 0.0,
        profit_factor=float(pf), avg_trade_bps=float(tr.mean() * 1e4) if len(tr) else 0.0,
        bars=int(len(pnl)),
    )


# ---- splits ---------------------------------------------------------------
SPLITS_INTRADAY = {  # crypto / 1h
    "train": ("2020-01-01", "2023-12-31"),
    "valid": ("2024-01-01", "2025-06-30"),
    "holdout": ("2025-07-01", "2100-01-01"),
}
SPLITS_DAILY = {  # long-history daily
    "train": ("1985-01-01", "2014-12-31"),
    "valid": ("2015-01-01", "2021-12-31"),
    "holdout": ("2022-01-01", "2100-01-01"),
}


def split_stats(pnl: pd.Series, pos: pd.Series, splits: dict) -> dict:
    out = {}
    for name, (a, b) in splits.items():
        m = (pnl.index >= pd.Timestamp(a, tz="UTC")) & (pnl.index <= pd.Timestamp(b, tz="UTC"))
        if m.sum() < 50:
            continue
        out[name] = stats(pnl[m], pos[m], pnl.index[m]).as_dict()
    return out
