"""Round 7: crowding signals (R7-1 perp-spot premium, R7-2 crowding composite, R7-3 beta-residual reversal).

Base = weekly fcarry L=7, frac=0.2, every=7, min_hist=60, phase 0. Each config run at 1x (6 bps/side) and
2x (12 bps) perp cost via xsec.run_weights (funding on). Spot is a signal only, never traded.
Diagnostics: valid daily-pnl corr with base fcarry (same cost mult); valid Sharpe min/median across the
7 weekday rebalance phases. Train + valid only; no holdout.

    python3 autoresearch/round7.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from next_bar import LOG, ROOT  # noqa: F401  (sets sys.path)
from xsec import PERP_BPS, SPLITS, run_weights, stats
from trading.data.universe import load_universe
from trading.strategies.carry import fcarry_daily, weekly
from trading.strategies.crowding import composite_daily, premium_daily, resid_reversal_daily

VA = SPLITS["valid"]


def _valid(x: pd.Series) -> pd.Series:
    return x[(x.index >= pd.Timestamp(VA[0], tz="UTC")) & (x.index <= pd.Timestamp(VA[1], tz="UTC"))]


def _sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0.0


if __name__ == "__main__":
    u = load_universe(spot=True)
    close, spot, fund = u["close"], u["spot_close"], u["funding"]
    daily = {
        "fcarry_base_L7": fcarry_daily(close, fund, 7, 0.2, 60),
        "R7-1_premium_carry": premium_daily(close, spot, 7, 0.2, 60),
        "R7-2_crowding_composite": composite_daily(close, spot, fund, 7, 0.2, 60, z_window=60),
        "R7-3_resid_reversal": resid_reversal_daily(close, 7, 0.2, 60),
    }
    params = {"lookback": 7, "frac": 0.2, "min_hist": 60, "every": 7}
    base_pnl, results = {}, {}
    print(f"{'config':26s} {'cost':>4s} | {'train sh':>8s} {'cagr':>7s} {'dd':>7s} {'to':>6s} | "
          f"{'valid sh':>8s} {'cagr':>7s} {'dd':>7s} {'to':>6s} | {'corr':>5s} | phase min/med (valid)")
    for name, wd in daily.items():
        wd = wd.reindex(close.index).fillna(0.0)
        w = weekly(wd, 7)
        for mult in (1, 2):
            bps = PERP_BPS * mult
            pnl, to = run_weights(w, close, fund, bps)
            st = stats(pnl, {"turnover": to})
            if name == "fcarry_base_L7":
                base_pnl[mult] = pnl
            pv, bv = _valid(pnl), _valid(base_pnl[mult])
            corr = round(float(pv.corr(bv)), 3)
            ph = [_sharpe(_valid(run_weights(weekly(wd, 7, o), close, fund, bps)[0])) for o in range(7)]
            diag = {"corr_base_valid": corr, "phase_sharpe_valid_min": round(min(ph), 3),
                    "phase_sharpe_valid_median": round(float(np.median(ph)), 3)}
            st.setdefault("valid", {}).update(diag)
            key = f"{name}_{mult}x"
            results[key] = st
            t, v = st.get("train", {}), st["valid"]
            print(f"{name:26s} {mult}x   | {t.get('sharpe', np.nan):+8.2f} {t.get('cagr', np.nan):+7.1%} "
                  f"{t.get('max_dd', np.nan):+7.1%} {t.get('turnover', np.nan):6.3f} | "
                  f"{v.get('sharpe', np.nan):+8.2f} {v.get('cagr', np.nan):+7.1%} {v.get('max_dd', np.nan):+7.1%} "
                  f"{v.get('turnover', np.nan):6.3f} | {corr:+5.2f} | "
                  f"{diag['phase_sharpe_valid_min']:+.2f}/{diag['phase_sharpe_valid_median']:+.2f}", flush=True)
            with LOG.open("a") as f:
                f.write(json.dumps({"ts": pd.Timestamp.now().isoformat(timespec="seconds"), "holdout_look": False,
                                    "market": "binance:28-coin universe:1d", "strategy": name,
                                    "params": {**params, "cost_mult": mult},
                                    "cost_bps": f"perp {bps:g}", "funding": True, "note": "R7",
                                    "results": st}) + "\n")
    (ROOT / "results" / "round7.json").write_text(json.dumps(results, indent=1))
