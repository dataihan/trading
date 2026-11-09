"""Round 3d: next-H-bars LightGBM forecast as a veto on the crypto trend sleeves of portfolio_v1.

Compares each crypto sleeve (base vs filtered) and the whole portfolio (v1 vs v1 with filtered crypto),
on train + valid. KEEP only if the portfolio improves on BOTH train and valid and most sleeves improve.

    python3 autoresearch/trend_filter.py
"""
import copy
import json

import numpy as np
import pandas as pd

from next_bar import ROOT, LOG
from portfolio import combine
from trading.backtest import engine

V1 = json.loads((ROOT / "strategies" / "portfolio_v1.json").read_text())
VARIANTS = [(4, 0.0), (4, 0.05), (24, 0.0), (24, 0.05)]
SPLITS = {k: v for k, v in engine.SPLITS_INTRADAY.items() if k != "holdout"}


def stats(daily: pd.Series) -> dict:
    out = {}
    for name, (a, b) in SPLITS.items():
        x = daily[(daily.index >= pd.Timestamp(a, tz="UTC")) & (daily.index <= pd.Timestamp(b, tz="UTC"))]
        sh = float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0.0
        eq = (1 + x).cumprod()
        out[name] = {"sharpe": round(sh, 3), "ret": round(float(eq.iloc[-1] - 1), 4),
                     "max_dd": round(float((eq / eq.cummax() - 1).min()), 4)}
    return out


def spec_with(h, margin):
    s = copy.deepcopy(V1)
    s["name"] = f"v1_trendfilter_h{h}_m{margin}"
    for sl in s["sleeves"]:
        if sl["strategy"] == "trend_ensemble":
            sl["strategy"] = "trend_ml_filter"
            sl["params"] = {"symbol": sl["market"].split(":")[1], "h": h, "margin": margin}
    return s


base = combine(V1)
start = base.index[0]
rows = {"v1": {c: stats(base[c]) for c in base.columns}}
for h, m in VARIANTS:
    df = combine(spec_with(h, m))
    df = df[df.index >= start]
    rows[f"h{h}_m{m}"] = {c: stats(df[c]) for c in df.columns}

cols = [c for c in base.columns if c.endswith("_trend")] + ["portfolio"]
print(f"{'variant':12s} " + " ".join(f"{c:>21s}" for c in cols))
for v, r in rows.items():
    print(f"{v:12s} " + " ".join(f"{r[c]['train']['sharpe']:+6.2f}/{r[c]['valid']['sharpe']:+5.2f}"
                                  f" dd{r[c]['valid']['max_dd']:+.0%}" for c in cols))
for v, r in rows.items():
    if v == "v1":
        continue
    with LOG.open("a") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now().isoformat(timespec="seconds"), "holdout_look": False,
                            "market": "portfolio_v1 crypto sleeves", "strategy": "trend_ml_filter",
                            "params": {"variant": v}, "cost_bps": 6, "funding": True,
                            "note": "R3d ML veto on trend", "results": r}) + "\n")
(ROOT / "results" / "trend_filter.json").write_text(json.dumps(rows, indent=1))
