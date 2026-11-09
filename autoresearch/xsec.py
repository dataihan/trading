"""Round 5: edges other than candle forecasting, on a 28-coin daily universe.

1. xsmom   — cross-sectional momentum: rank coins by past return, long top / short bottom (or long-only top),
             rebalanced weekly.
2. fcarry  — cross-sectional funding carry: short the highest trailing funding, long the lowest, weekly.
3. cashcarry — per-coin delta-neutral cash-and-carry: long spot + short perp while trailing funding is high.

Timing: weights decided at the daily close t from data <= t, earn close[t+1]/close[t]; funding of day t+1.
Costs: perp 6 bps/side, spot 10 bps/side, on turnover. Funding: longs pay positive funding, shorts receive.
Universe is survivorship-biased (coins still listed today). Train + valid only.

    python3 autoresearch/xsec.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from next_bar import LOG, ROOT
from portfolio import combine
from trading.backtest import engine
from trading.data.universe import load_universe
from trading.strategies.carry import fcarry as _fcarry, weekly

PERP_BPS, SPOT_BPS = 6.0, 10.0
SPLITS = {k: v for k, v in engine.SPLITS_INTRADAY.items() if k != "holdout"}


def stats(pnl: pd.Series, extra: dict | None = None) -> dict:
    out = {}
    for name, (a, b) in SPLITS.items():
        x = pnl[(pnl.index >= pd.Timestamp(a, tz="UTC")) & (pnl.index <= pd.Timestamp(b, tz="UTC"))].dropna()
        if len(x) < 60:
            continue
        eq = (1 + x).cumprod()
        yrs = len(x) / 365
        out[name] = {"sharpe": round(float(x.mean() / x.std() * np.sqrt(365)), 3) if x.std() > 0 else 0.0,
                     "cagr": round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4),
                     "max_dd": round(float((eq / eq.cummax() - 1).min()), 4),
                     **({k: round(float(v[x.index].mean()), 4) for k, v in (extra or {}).items()})}
    return out


def run_weights(w: pd.DataFrame, close: pd.DataFrame, fund: pd.DataFrame, bps: float = PERP_BPS):
    """Daily pnl of perp weights w (decided at close t), net of costs and funding."""
    ret = close.pct_change().shift(-1)                      # t -> t+1
    w = w.reindex(close.index).fillna(0.0)
    gross = (w * ret).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    fpay = (w * fund.shift(-1)).sum(axis=1)                 # longs pay positive funding
    pnl = gross - turnover * bps / 1e4 - fpay
    return pnl.shift(1).fillna(0.0), turnover                # label by the day it is realised


def xsmom(u, lookbacks=(30,), mode="ls", frac=0.2, every=7, min_hist=120):
    close = u["close"]
    hist = close.notna().cumsum()
    score = sum(close.pct_change(L).rank(axis=1, pct=True) for L in lookbacks) / len(lookbacks)
    score = score.where(hist >= min_hist)
    n = score.notna().sum(axis=1)
    k = (n * frac).round().clip(lower=1)
    rk = score.rank(axis=1, ascending=False)
    top = rk.le(k, axis=0)
    bot = score.rank(axis=1, ascending=True).le(k, axis=0)
    if mode == "ls":
        w = top.astype(float).div(k, axis=0) * 0.5 - bot.astype(float).div(k, axis=0) * 0.5
    else:
        w = top.astype(float).div(k, axis=0)
    w = w.where(n >= 8, 0.0)
    return weekly(w, every)


def fcarry(u, lookback=7, frac=0.2, every=7, min_hist=60):
    return _fcarry(u["close"], u["funding"], lookback, frac, every, min_hist)


def cashcarry(u, coins=None, thresh_ann=0.10, lookback=7):
    """Long spot + short perp (equal notional) while trailing funding (annualised) > threshold.
    Equal weight across active coins; pnl per unit of notional."""
    close, spot, fund = u["close"], u["spot_close"], u["funding"]
    coins = [c for c in (coins or spot.columns) if c in spot.columns]
    ann = fund[coins].rolling(lookback).mean() * 365
    on = (ann > thresh_ann) & spot[coins].notna() & close[coins].notna()
    n = on.sum(axis=1).replace(0, np.nan)
    w = on.astype(float).div(n, axis=0).fillna(0.0)          # weight per coin (each = spot long + perp short)
    pr = close[coins].pct_change().shift(-1)
    sr = spot[coins].pct_change().shift(-1)
    gross = (w * (sr - pr)).sum(axis=1) + (w * fund[coins].shift(-1)).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    pnl = gross - turn * (PERP_BPS + SPOT_BPS) / 1e4
    return pnl.shift(1).fillna(0.0), w.sum(axis=1)


if __name__ == "__main__":
    u = load_universe()
    v1 = combine(json.loads((ROOT / "strategies" / "portfolio_v1.json").read_text()))["portfolio"]
    results = {}

    def report(name, pnl, extra=None):
        st = stats(pnl, extra)
        both = pnl.index.intersection(v1.index)
        va = both[(both >= pd.Timestamp("2024-01-01", tz="UTC")) & (both <= pd.Timestamp("2025-06-30", tz="UTC"))]
        corr = round(float(pnl[va].corr(v1[va])), 2) if len(va) > 60 else None
        results[name] = {**st, "corr_v1_valid": corr}
        t, v = st.get("train", {}), st.get("valid", {})
        print(f"{name:34s} train sh={t.get('sharpe', np.nan):+.2f} cagr={t.get('cagr', np.nan):+.1%} "
              f"dd={t.get('max_dd', np.nan):+.1%} | valid sh={v.get('sharpe', np.nan):+.2f} "
              f"cagr={v.get('cagr', np.nan):+.1%} dd={v.get('max_dd', np.nan):+.1%} | corr(v1)={corr}", flush=True)
        with LOG.open("a") as f:
            f.write(json.dumps({"ts": pd.Timestamp.now().isoformat(timespec="seconds"), "holdout_look": False,
                                "market": "binance:28-coin universe:1d", "strategy": name, "params": {},
                                "cost_bps": "perp 6 / spot 10", "funding": True,
                                "note": "R5 cross-section / carry (survivorship-biased universe)",
                                "results": results[name]}) + "\n")

    for lbs in [(14,), (30,), (60,), (14, 30, 60)]:
        for mode in ["ls", "lo"]:
            pnl, to = run_weights(xsmom(u, lbs, mode), u["close"], u["funding"])
            report(f"xsmom_{mode}_L{'-'.join(map(str, lbs))}", pnl, {"turnover": to})
    for lb in [3, 7, 30]:
        pnl, to = run_weights(fcarry(u, lb), u["close"], u["funding"])
        report(f"fcarry_ls_L{lb}", pnl, {"turnover": to})
    for coins, tag in [(["BTCUSDT", "ETHUSDT"], "btceth"), (None, "all")]:
        for th in [0.05, 0.10, 0.20]:
            pnl, expo = cashcarry(u, coins, th)
            report(f"cashcarry_{tag}_th{int(th * 100)}", pnl, {"exposure": expo})
    (ROOT / "results" / "xsec.json").write_text(json.dumps(results, indent=1))
