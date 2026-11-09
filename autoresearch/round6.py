"""Round 6: make fcarry robust (R6-1 tranching, R6-2 liquidity/hygiene filter, R6-3 beta-neutral inverse-vol).

Base = weekly fcarry L=7, frac=0.2, min_hist=60. Each config run at 1x (6 bps/side) and 2x (12 bps) perp cost
via xsec.run_weights (funding on). Train + valid only; no holdout.

    python3 autoresearch/round6.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from next_bar import LOG, ROOT  # noqa: F401  (sets sys.path)
from xsec import PERP_BPS, SPLITS, run_weights, stats
from trading.data.universe import load_universe
from trading.strategies.carry import (ew_return, fcarry, fcarry_beta_neutral, fcarry_filtered,
                                      fcarry_tranched)


def realised_beta(pnl: pd.Series, mkt: pd.Series) -> dict:
    out = {}
    for name, (a, b) in SPLITS.items():
        idx = pnl.index[(pnl.index >= pd.Timestamp(a, tz="UTC")) & (pnl.index <= pd.Timestamp(b, tz="UTC"))]
        x, y = pnl[idx], mkt.reindex(idx)
        ok = x.notna() & y.notna()
        if ok.sum() < 60:
            continue
        out[name] = round(float(np.cov(x[ok], y[ok])[0, 1] / y[ok].var()), 3)
    return out


if __name__ == "__main__":
    u = load_universe()
    close, qv, fund = u["close"], u["quote_volume"], u["funding"]
    mkt = ew_return(close)                                   # labelled by the day it is realised, like pnl
    configs = {
        "fcarry_base_L7": fcarry(close, fund, 7, 0.2, 7, 60),
        "R6-1_fcarry_tranched": fcarry_tranched(close, fund),
        "R6-2_fcarry_filtered": fcarry_filtered(close, qv, fund),
        "R6-3_fcarry_beta_neutral": fcarry_beta_neutral(close, qv, fund),
    }
    results = {}
    print(f"{'config':34s} {'cost':>4s} | {'train sh':>8s} {'cagr':>7s} {'dd':>7s} {'to':>6s} {'beta':>6s} | "
          f"{'valid sh':>8s} {'cagr':>7s} {'dd':>7s} {'to':>6s} {'beta':>6s} | cost/yr fund/yr (valid)")
    for name, w in configs.items():
        w = w.reindex(close.index).fillna(0.0)
        fund_rx = -(w * fund.shift(-1)).sum(axis=1).shift(1).fillna(0.0)   # funding received, per day
        for mult in (1, 2):
            bps = PERP_BPS * mult
            pnl, to = run_weights(w, close, fund, bps)
            cost = (to * bps / 1e4).shift(1).fillna(0.0)
            st = stats(pnl, {"turnover": to, "cost_drag_ann": cost * 365, "funding_ann": fund_rx * 365})
            rb = realised_beta(pnl, mkt)
            for sp in st:
                st[sp]["beta_ew"] = rb.get(sp)
            key = f"{name}_{mult}x"
            results[key] = st
            t, v = st.get("train", {}), st.get("valid", {})
            print(f"{name:34s} {mult}x   | {t.get('sharpe', np.nan):+8.2f} {t.get('cagr', np.nan):+7.1%} "
                  f"{t.get('max_dd', np.nan):+7.1%} {t.get('turnover', np.nan):6.3f} {t.get('beta_ew', np.nan):+6.2f} | "
                  f"{v.get('sharpe', np.nan):+8.2f} {v.get('cagr', np.nan):+7.1%} {v.get('max_dd', np.nan):+7.1%} "
                  f"{v.get('turnover', np.nan):6.3f} {v.get('beta_ew', np.nan):+6.2f} | "
                  f"{v.get('cost_drag_ann', np.nan):.1%} {v.get('funding_ann', np.nan):.1%}", flush=True)
            with LOG.open("a") as f:
                f.write(json.dumps({"ts": pd.Timestamp.now().isoformat(timespec="seconds"), "holdout_look": False,
                                    "market": "binance:28-coin universe:1d", "strategy": name,
                                    "params": {"lookback": 7, "frac": 0.2, "min_hist": 60, "every": 7,
                                               "cost_mult": mult},
                                    "cost_bps": f"perp {bps:g}", "funding": True, "note": "R6",
                                    "results": st}) + "\n")
    (ROOT / "results" / "round6.json").write_text(json.dumps(results, indent=1))
