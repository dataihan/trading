"""Round 5b: portfolio v2 = v1 + new sleeves with fixed risk-parity weights from TRAIN volatility only.

Each sleeve is scaled so its train-period daily vol equals v1's train vol, then all sleeves get equal weight.
Variants: with and without cash-and-carry. Train + valid only (holdout: risk-auditor, once).

    python3 autoresearch/portfolio_v2.py
"""
import json

import numpy as np
import pandas as pd

from next_bar import LOG, ROOT
from portfolio import combine
from trading.data.universe import load_universe
from xsec import cashcarry, fcarry, run_weights, stats, xsmom

TRAIN = (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC"))


def build(holdout: bool = False):
    u = load_universe()
    sleeves = {
        "v1": combine(json.loads((ROOT / "strategies" / "portfolio_v1.json").read_text()))["portfolio"],
        "xsmom_ls_L14": run_weights(xsmom(u, (14,), "ls"), u["close"], u["funding"])[0],
        "fcarry_ls_L7": run_weights(fcarry(u, 7), u["close"], u["funding"])[0],
        "cashcarry_btceth_th10": cashcarry(u, ["BTCUSDT", "ETHUSDT"], 0.10)[0],
    }
    df = pd.DataFrame(sleeves).dropna()
    df = df[df.index >= sleeves["v1"].index[0]]
    tr = df[(df.index >= TRAIN[0]) & (df.index <= TRAIN[1])]
    scale = tr["v1"].std() / tr.std()                       # train-only risk scaling
    return df, scale


if __name__ == "__main__":
    import sys
    holdout = "--holdout" in sys.argv
    df, scale = build()
    if holdout:
        import xsec
        xsec.SPLITS = {"train": xsec.SPLITS["train"], "valid": xsec.SPLITS["valid"],
                       "holdout": ("2025-07-01", "2100-01-01")}
    variants = {
        "v1": ["v1"],
        "v2a_all": ["v1", "xsmom_ls_L14", "fcarry_ls_L7", "cashcarry_btceth_th10"],
        "v2b_no_cashcarry": ["v1", "xsmom_ls_L14", "fcarry_ls_L7"],
    }
    out = {"scale": scale.round(3).to_dict()}
    print("train-vol scale per sleeve:", scale.round(3).to_dict())
    print("valid correlation:\n", df[(df.index >= "2024-01-01") & (df.index <= "2025-06-30")].corr().round(2))
    from xsec import stats as st
    for name, cols in variants.items():
        pnl = (df[cols] * scale[cols]).mean(axis=1)
        r = st(pnl)
        out[name] = r
        print(name, {k: (v["sharpe"], v["cagr"], v["max_dd"]) for k, v in r.items()})
        with LOG.open("a") as f:
            f.write(json.dumps({"ts": pd.Timestamp.now().isoformat(timespec="seconds"), "holdout_look": holdout,
                                "market": "portfolio", "strategy": name, "params": {"sleeves": cols,
                                "weights": "equal risk, train vol"}, "cost_bps": "perp 6 / spot 10",
                                "funding": True, "note": "R5b portfolio v2", "results": r}) + "\n")
    (ROOT / "results" / f"portfolio_v2{'_holdout' if holdout else ''}.json").write_text(json.dumps(out, indent=1))
