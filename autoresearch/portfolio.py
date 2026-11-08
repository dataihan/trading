"""Combine strategy sleeves into one daily-rebalanced portfolio and report per split.

Each sleeve's per-bar pnl (net of fees + funding) is compounded to calendar days,
then sleeves are combined with fixed weights (no look-ahead weight fitting).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "autoresearch"))

from evaluate import DEFAULT_COST, _data  # noqa: E402
from trading.backtest import engine  # noqa: E402
from trading.data.loaders import binance_funding  # noqa: E402
from trading.strategies.library import REGISTRY  # noqa: E402

PORTFOLIO = ROOT / "strategies" / "portfolio_v1.json"


def sleeve_daily_pnl(cfg: dict, cost_mult: float = 1.0) -> pd.Series:
    market = cfg["market"]
    src = market.split(":")[0]
    df = _data(market)
    out = REGISTRY[cfg["strategy"]](df, **cfg.get("params", {}))
    funding = binance_funding(market.split(":")[1]) if src == "binance" else None
    cost = cfg.get("cost_bps", DEFAULT_COST[src]) * cost_mult
    pnl, _, _ = engine.run(df.close, out["position"], cost_bps=cost,
                           bar_ret=out.get("bar_ret"), funding=funding)
    # pnl at bar t is earned during bar t+1 -> label it by the bar it is realised in
    pnl = pnl.shift(1).fillna(0.0)
    daily = (1 + pnl).groupby(pnl.index.floor("D")).prod() - 1
    daily.index = daily.index.tz_convert("UTC") if daily.index.tz else daily.index.tz_localize("UTC")
    return daily


def combine(spec: dict, cost_mult: float = 1.0) -> pd.DataFrame:
    cols = {s["name"]: sleeve_daily_pnl(s, cost_mult) for s in spec["sleeves"]}
    raw = pd.DataFrame(cols).sort_index()
    start = max(raw[c].first_valid_index() for c in raw)  # all sleeves live
    raw = raw[raw.index >= start].fillna(0.0)
    base_w = pd.Series({s["name"]: s["weight"] for s in spec["sleeves"]})
    if spec.get("weighting") == "inverse_vol":
        # weights from TRAILING vol only (shift 1 day), rescaled to the same gross weight sum
        lb = spec.get("vol_lookback", 60)
        vol = raw.rolling(lb, min_periods=20).std().shift(1)
        iv = (base_w / vol).replace([np.inf, -np.inf], np.nan)
        w = iv.div(iv.sum(axis=1), axis=0).mul(base_w.sum()).fillna(base_w)
        if spec.get("target_vol"):
            port_vol = (raw * w).sum(axis=1).rolling(lb, min_periods=20).std().shift(1) * np.sqrt(365)
            w = w.mul((spec["target_vol"] / port_vol).clip(upper=spec.get("max_lev", 2.0)).fillna(1.0), axis=0)
        df = raw * w
    else:
        df = raw * base_w
    df["portfolio"] = df.sum(axis=1)
    return df


def report(spec: dict, holdout: bool, cost_mult: float = 1.0) -> dict:
    df = combine(spec, cost_mult)
    splits = dict(engine.SPLITS_INTRADAY)
    if not holdout:
        splits.pop("holdout")
    res = {}
    for col in df.columns:
        pnl = df[col]
        pos = pd.Series(1.0, index=pnl.index)
        res[col] = engine.split_stats(pnl, pos, splits)
    return res


if __name__ == "__main__":
    holdout = "--holdout" in sys.argv
    cost_mult = 2.0 if "--2xcost" in sys.argv else 0.0 if "--nocost" in sys.argv else 1.0
    path = next((Path(a) for a in sys.argv[1:] if a.endswith(".json")), PORTFOLIO)
    spec = json.loads(path.read_text())
    res = report(spec, holdout, cost_mult)
    for name, r in res.items():
        for split, m in r.items():
            print(f"{name:14s} {split:8s} ret={m['total_return']:+.1%} cagr={m['cagr']:+.1%} "
                  f"sharpe={m['sharpe']:.2f} maxDD={m['max_dd']:.1%}")
    tag = "_2xcost" if cost_mult > 1 else "_nocost" if cost_mult == 0 else ""
    out = ROOT / "results" / f"{spec['name']}{'_holdout' if holdout else ''}{tag}.json"
    out.write_text(json.dumps(res, indent=1))
