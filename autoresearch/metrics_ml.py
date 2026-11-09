"""Round 4: do open interest / long-short ratio features improve the next-H-bars forecast?

A/B on identical rows: arm A = price/volume/funding features ("base_late"), arm B = A + metrics.
Part 1: forecast quality + standalone trading (hold H bars, taker 6 bps + funding), 4 coins x H in {4, 24}.
Part 2: H=24 forecast as a veto on the portfolio_v1 crypto trend sleeves (round 3d setup).
Train + valid only.

    python3 autoresearch/metrics_ml.py
"""
import copy
import json

import numpy as np
import pandas as pd

from next_bar import LOG, ROOT
from portfolio import combine
from trading.backtest import engine
from trading.data.loaders import binance_funding, binance_klines
from trading.ml.multibar import proba_1h

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
SPLITS = {k: v for k, v in engine.SPLITS_INTRADAY.items() if k != "holdout"}
ARMS = ["base_late", "metrics"]


def part1():
    rows = []
    for sym in SYMS:
        df = binance_klines(sym, "1h")
        fund = binance_funding(sym)
        for h in (4, 24):
            fwd = np.log(df.close.shift(-h) / df.close)
            y = (fwd > 0).where(fwd.notna() & (fwd != 0))
            dec = pd.Series((np.arange(len(df)) % h) == 0, index=df.index)
            ps = {a: proba_1h(df, sym, h, a) for a in ARMS}
            both = ps["base_late"].notna() & ps["metrics"].notna() & y.notna() & dec
            for arm, p in ps.items():
                for margin in (0.0, 0.05, 0.10):
                    sig = pd.Series(0.0, index=df.index)
                    sig[p > 0.5 + margin] = 1.0
                    sig[p < 0.5 - margin] = -1.0
                    pos = sig.where(dec & both).ffill().fillna(0.0)
                    pos[~both.cummax()] = 0.0
                    pnl, _, _ = engine.run(df.close, pos, cost_bps=6.0, funding=fund)
                    res = {}
                    for split, (a, b) in SPLITS.items():
                        per = (df.index >= pd.Timestamp(a, tz="UTC")) & (df.index <= pd.Timestamp(b, tz="UTC"))
                        d = per & both.values
                        if d.sum() < 30:
                            continue
                        u, pp = y[d].astype(bool), p[d]
                        conf = (pp - 0.5).abs() > margin
                        hit = (pp > 0.5) == u
                        live = per & both.cummax().values
                        st = engine.stats(pnl[live], pos[live], df.index[live]).as_dict()
                        res[split] = {"decisions": int(d.sum()), "acc": round(float(hit.mean()), 4),
                                      "majority": round(float(max(u.mean(), 1 - u.mean())), 4),
                                      "acc_conf": round(float(hit[conf].mean()), 4) if conf.any() else None,
                                      "coverage": round(float(conf.mean()), 4),
                                      "sharpe": st["sharpe"], "total_return": st["total_return"],
                                      "max_dd": st["max_dd"]}
                    rows.append({"sym": sym, "h": h, "arm": arm, "margin": margin, **res})
                    with LOG.open("a") as f:
                        f.write(json.dumps({"ts": pd.Timestamp.now().isoformat(timespec="seconds"),
                                            "holdout_look": False, "market": f"binance:{sym}:1h",
                                            "strategy": f"multibar_{arm}", "params": {"h": h, "margin": margin},
                                            "cost_bps": 6, "funding": True, "note": "R4 OI/long-short A/B",
                                            "results": res}) + "\n")
                    t, v = res.get("train", {}), res.get("valid", {})
                    print(f"{sym:8s} h={h:2d} {arm:9s} m={margin:.2f} | train n={t.get('decisions', 0):4d} "
                          f"acc={t.get('acc', float('nan')):.3f} conf={t.get('acc_conf') or float('nan'):.3f} "
                          f"sh={t.get('sharpe', float('nan')):+.2f} | valid n={v.get('decisions', 0):4d} "
                          f"acc={v.get('acc', float('nan')):.3f} conf={v.get('acc_conf') or float('nan'):.3f} "
                          f"sh={v.get('sharpe', float('nan')):+.2f}", flush=True)
    return rows


def part2():
    v1 = json.loads((ROOT / "strategies" / "portfolio_v1.json").read_text())

    def daily_stats(x):
        out = {}
        for split, (a, b) in SPLITS.items():
            s = x[(x.index >= pd.Timestamp(a, tz="UTC")) & (x.index <= pd.Timestamp(b, tz="UTC"))]
            eq = (1 + s).cumprod()
            out[split] = {"sharpe": round(float(s.mean() / s.std() * np.sqrt(365)), 3),
                          "max_dd": round(float((eq / eq.cummax() - 1).min()), 4)}
        return out

    base = combine(v1)
    res = {"v1": daily_stats(base.portfolio)}
    for arm in ARMS:
        s = copy.deepcopy(v1)
        for sl in s["sleeves"]:
            if sl["strategy"] == "trend_ensemble":
                sl["strategy"] = "trend_ml_filter"
                sl["params"] = {"symbol": sl["market"].split(":")[1], "h": 24, "margin": 0.0, "extra": arm}
        df = combine(s)
        res[f"veto_h24_{arm}"] = daily_stats(df.portfolio[df.index >= base.index[0]])
    for k, r in res.items():
        print(f"portfolio {k:20s} train sharpe={r['train']['sharpe']:+.2f} dd={r['train']['max_dd']:+.1%} | "
              f"valid sharpe={r['valid']['sharpe']:+.2f} dd={r['valid']['max_dd']:+.1%}")
    with LOG.open("a") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now().isoformat(timespec="seconds"), "holdout_look": False,
                            "market": "portfolio_v1 crypto sleeves", "strategy": "trend_ml_filter h24 A/B",
                            "params": {}, "cost_bps": 6, "funding": True, "note": "R4 OI/long-short veto",
                            "results": res}) + "\n")
    return res


if __name__ == "__main__":
    r1 = part1()
    r2 = part2()
    (ROOT / "results" / "metrics_ml.json").write_text(json.dumps({"forecast": r1, "veto": r2}, indent=1))
