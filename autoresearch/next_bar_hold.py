"""Round 3b: next-bar LightGBM with position hold (flip only on confident signals)."""
import json
import sys

from next_bar import ROOT, evaluate, log

holdout = "--holdout" in sys.argv
only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
rows = []
for key in ["btc_1h", "eth_1h", "btc_4h", "eth_4h", "qqq_1d"]:
    for margin in [0.02, 0.05, 0.08]:
        for cost in ([None] if key == "qqq_1d" else [None, 2.0]):
            name = f"{key}_lgbm_hold_m{margin}" + ("_maker" if cost == 2.0 else "")
            if only and name != only:
                continue
            res = evaluate(key, "lgbm", margin, holdout, {}, hold=True, cost_override=cost)
            log(key, "lgbm_hold", margin, res, holdout, f"R3b hold, cost={'maker 2bps' if cost else 'default'}")
            for split, r in res.items():
                rows.append({"config": name, "split": split, **r})
                print(f"{name:30s} {split:7s} acc_conf={r['acc_conf']:.3f} cov={r['coverage']:.2f} "
                      f"trades={r['n_trades']:5d} sharpe={r['sharpe']:+.2f} ret={r['total_return']:+.1%} "
                      f"dd={r['max_dd']:.1%}", flush=True)
(ROOT / "results" / f"next_bar_hold{'_holdout' if holdout else ''}.json").write_text(json.dumps(rows, indent=1))
