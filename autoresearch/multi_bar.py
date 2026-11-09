"""Round 3c: predict the direction of the next H candles instead of the next one.

One decision every H bars (non-overlapping), position held for H bars -> far fewer fee payments, and each
correct call earns a bigger move than a single candle does.

    python3 autoresearch/multi_bar.py                      # train + valid
    python3 autoresearch/multi_bar.py --only btc_1h_h24_m0.05 --holdout   # risk-auditor only, once

Walk-forward LightGBM, expanding window, retrained every 30 days, trained on decision bars only.
Purge: a training row i is used only if its label is known, i.e. i + H <= prediction bar.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from next_bar import ROOT, load_market, log
from trading.backtest import engine
from trading.ml.multibar import wf_proba_h
from trading.ml.walkforward import features

GRID = {"btc_1h": [4, 12, 24], "eth_1h": [4, 12, 24], "btc_4h": [3, 6], "eth_4h": [3, 6]}
MARGINS = [0.0, 0.05, 0.10]


def run(key: str, h: int, holdout: bool = False) -> dict:
    df, fund, cost, splits, daily = load_market(key)
    bpd = {"1h": 24, "4h": 6}[key.split("_")[1]]
    disk = ROOT / "data" / "processed" / f"multibar_proba_{key}_h{h}.parquet"
    fwd = np.log(df.close.shift(-h) / df.close)
    y = (fwd > 0).astype(float).where(fwd.notna() & (fwd != 0))
    dec = (np.arange(len(df)) % h) == 0
    if disk.exists():
        p = pd.read_parquet(disk)["p"]
    else:
        X = features(df, fund, daily=False)
        p = wf_proba_h(X, y, dec, h, min_train=365 * bpd, retrain=30 * bpd)
        disk.parent.mkdir(parents=True, exist_ok=True)
        p.to_frame("p").to_parquet(disk)
    dec_s = pd.Series(dec, index=df.index)
    out = {}
    for margin in MARGINS:
        sig = pd.Series(0.0, index=df.index)
        sig[p > 0.5 + margin] = 1.0
        sig[p < 0.5 - margin] = -1.0
        sig[p.isna()] = 0.0
        pos = sig.where(dec_s).ffill().fillna(0.0)     # decide on decision bars, hold H bars
        pnl, _, _ = engine.run(df.close, pos, cost_bps=cost, funding=fund)
        sp = dict(splits)
        if not holdout:
            sp.pop("holdout")
        res = {}
        for name, (a, b) in sp.items():
            per = (df.index >= pd.Timestamp(a, tz="UTC")) & (df.index <= pd.Timestamp(b, tz="UTC"))
            d = per & dec & p.notna().values & y.notna().values
            if d.sum() < 30:
                continue
            u = y[d].astype(bool)
            pp = p[d]
            conf = (pp - 0.5).abs() > margin
            hit = ((pp > 0.5) == u)
            base = float(u.mean())
            n_conf = int(conf.sum())
            acc_conf = float(hit[conf].mean()) if n_conf else float("nan")
            st = engine.stats(pnl[per], pos[per], df.index[per]).as_dict()  # all bars of the split
            res[name] = {"decisions": int(d.sum()), "base_up": round(base, 4),
                         "majority": round(max(base, 1 - base), 4), "acc": round(float(hit.mean()), 4),
                         "acc_conf": round(acc_conf, 4), "coverage": round(n_conf / d.sum(), 4),
                         "z_conf": round((acc_conf - 0.5) / np.sqrt(0.25 / max(n_conf, 1)), 2),
                         "sharpe": st["sharpe"], "total_return": st["total_return"],
                         "max_dd": st["max_dd"], "n_trades": st["n_trades"], "exposure": st["exposure"]}
        out[f"{key}_h{h}_m{margin}"] = res
        log(key, f"multibar_h{h}", margin, res, holdout, "R3c multi-bar horizon, hold H bars, taker 6bps")
    return out


if __name__ == "__main__":
    holdout = "--holdout" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    rows = []
    for key, hs in GRID.items():
        for h in hs:
            if only and not only.startswith(f"{key}_h{h}_"):
                continue
            for name, res in run(key, h, holdout).items():
                if only and name != only:
                    continue
                for split, r in res.items():
                    rows.append({"config": name, "split": split, **r})
                    print(f"{name:20s} {split:7s} dec={r['decisions']:5d} acc={r['acc']:.3f} "
                          f"maj={r['majority']:.3f} acc_conf={r['acc_conf']:.3f} cov={r['coverage']:.2f} "
                          f"z={r['z_conf']:+.1f} trades={r['n_trades']:4d} sharpe={r['sharpe']:+.2f} "
                          f"ret={r['total_return']:+.1%} dd={r['max_dd']:.1%}", flush=True)
    (ROOT / "results" / f"multi_bar{'_holdout' if holdout else ''}.json").write_text(json.dumps(rows, indent=1))
