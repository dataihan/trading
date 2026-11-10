import sys, json, time, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from trading.data.loaders import load, binance_funding
from trading.backtest import engine
from trading.strategies.library import REGISTRY
sp = {k: v for k, v in engine.SPLITS_INTRADAY.items() if k != "holdout"}
var = [("rsi", {"mode": "revert"}), ("rsi", {"mode": "trend", "high": None}), ("macd", {}),
       ("bollinger", {"mode": "revert"}), ("bollinger", {"mode": "breakout"})]
log = open(ROOT / "results" / "experiments.jsonl", "a")
for m in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    h = load(f"binance:{m}:1h")
    d = h.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    f = binance_funding(m)
    f = f.resample("1D").sum()  # daily sum of 8h funding rates
    for s, p in var:
        pos = REGISTRY[s](d, **p)["position"]
        pnl, _, _ = engine.run(d.close, pos, cost_bps=6.0, funding=f)
        r = engine.split_stats(pnl, pos.reindex(pnl.index).fillna(0), sp)
        cfg = {"market": f"binance:{m}:1h->1d resample", "strategy": s, "params": p, "cost_bps": 6.0,
               "note": "classic TA on real daily candles"}
        log.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "id": hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10],
                              "holdout_look": False, **cfg, "results": r}) + "\n")
        t, v = r["train"], r["valid"]
        print(f"{m[:3]} {s}-{p.get('mode','')}".ljust(22),
              f"train SR {t['sharpe']:+.2f} ret {t['total_return']*100:+.0f}% | valid SR {v['sharpe']:+.2f} ret {v['total_return']*100:+.0f}% DD {v['max_dd']*100:.0f}% n{v.get('trades', v.get('n_trades'))}")
