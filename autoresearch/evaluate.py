"""Evaluate one experiment config and append the result to results/experiments.jsonl.

Config (JSON / dict):
  {"market": "binance:BTCUSDT:1h", "strategy": "tsmom", "params": {...},
   "cost_bps": 6, "funding": true, "note": "H2 ..."}

Holdout split is NOT reported unless --holdout is passed (it must be touched
once per final candidate; every holdout look is logged).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading.backtest import engine  # noqa: E402
from trading.data.loaders import binance_funding, load  # noqa: E402
from trading.strategies.library import REGISTRY  # noqa: E402

LOG = ROOT / "results" / "experiments.jsonl"
LOG.parent.mkdir(exist_ok=True)

DEFAULT_COST = {"binance": 6.0, "spot": 12.0, "yahoo": 2.0}  # bps per side incl. slippage

_cache: dict = {}


def _data(market):
    if market not in _cache:
        _cache[market] = load(market)
    return _cache[market]


def evaluate(cfg: dict, holdout: bool = False, log: bool = True) -> dict:
    market = cfg["market"]
    src = market.split(":")[0]
    df = _data(market)
    out = REGISTRY[cfg["strategy"]](df, **cfg.get("params", {}))
    pos = out["position"]
    cost = cfg.get("cost_bps", DEFAULT_COST[src])
    funding = None
    if src == "binance" and cfg.get("funding", True):
        funding = binance_funding(market.split(":")[1])
    pnl, _, _ = engine.run(df.close, pos, cost_bps=0.0 if out.get("no_turnover_cost") else cost,
                           bar_ret=out.get("bar_ret"), funding=funding)
    splits = engine.SPLITS_DAILY if market.endswith("1d") else engine.SPLITS_INTRADAY
    sp = dict(splits)
    if not holdout:
        sp.pop("holdout")
    res = engine.split_stats(pnl, pos.reindex(pnl.index).fillna(0), sp)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "id": hashlib.md5(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10], "holdout_look": holdout,
        **cfg, "cost_bps": cost, "results": res}
    if log:
        with LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="JSON string or path to .json")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()) if a.config.endswith(".json") else json.loads(a.config)
    print(json.dumps(evaluate(cfg, a.holdout), indent=1))
