"""Round 3: next-candle direction indicator.

Question: can we predict whether the NEXT candle closes up or down, better than always guessing the
majority direction? Measured out-of-sample (walk-forward), then traded with costs.

    python3 autoresearch/next_bar.py               # train + valid only
    python3 autoresearch/next_bar.py --only btc_4h_lgbm --holdout   # risk-auditor only, once

Models
- lgbm: LightGBM, walk-forward on an expanding window, retrained every `retrain` bars. A prediction for
  bar t uses only data up to the close of t. Label = sign(close[t+1] / close[t]).
- mom1 / rev1: next candle = same / opposite direction as the last candle.
- ibs: (close-low)/(high-low) < 0.2 -> up, > 0.8 -> down (daily only).

Metrics per split
- acc: hit rate on all bars; majority: hit rate of always guessing the more common direction;
  edge = acc - majority (the honest number).
- acc_conf / coverage: hit rate when |p - 0.5| > margin, and the share of bars where that happens.
- trading: long if p > 0.5 + margin, short if p < 0.5 - margin, else flat; costs per side as below.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading.backtest import engine  # noqa: E402
from trading.data.loaders import binance_funding, binance_klines, yahoo  # noqa: E402
from trading.ml.walkforward import features  # noqa: E402

LOG = ROOT / "results" / "experiments.jsonl"
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
       "quote_volume": "sum", "trades": "sum", "taker_buy_base": "sum"}

# market -> (loader, bars per day, cost bps per side, splits)
MARKETS = {
    "btc_1h": ("BTCUSDT", "1h"), "btc_4h": ("BTCUSDT", "4h"), "btc_1d": ("BTCUSDT", "1d"),
    "eth_1h": ("ETHUSDT", "1h"), "eth_4h": ("ETHUSDT", "4h"), "eth_1d": ("ETHUSDT", "1d"),
    "qqq_1d": ("QQQ", "1d"),
}


def load_market(key: str) -> tuple[pd.DataFrame, pd.Series | None, float, dict, bool]:
    sym, tf = MARKETS[key]
    if sym == "QQQ":
        return yahoo("QQQ", "1d", "max"), None, 2.0, engine.SPLITS_DAILY, True
    df = binance_klines(sym, "1h")
    if tf != "1h":
        df = df.resample(tf.replace("h", "h").replace("1d", "1D"), label="left", closed="left").agg(AGG).dropna()
    fund = binance_funding(sym)
    fund = fund.groupby(fund.index.floor({"1h": "1h", "4h": "4h", "1d": "1D"}[tf])).sum()
    return df, fund, 6.0, engine.SPLITS_INTRADAY, tf == "1d"


def wf_proba(X: pd.DataFrame, y: pd.Series, min_train: int, retrain: int, seed: int = 0) -> pd.Series:
    import lightgbm as lgb
    n = len(X)
    proba = pd.Series(np.nan, index=X.index)
    start = min_train
    while start < n:
        end = min(start + retrain, n)
        # label[i] uses close[i+1] <= close[start], known when predicting bar `start`
        Xtr, ytr = X.iloc[:start], y.iloc[:start]
        ok = ytr.notna() & Xtr.notna().all(axis=1)
        if ok.sum() > 300:
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7,
                                   min_child_samples=max(50, int(ok.sum() / 50)), subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                   random_state=seed, verbose=-1)
            m.fit(Xtr[ok], ytr[ok])
            Xte = X.iloc[start:end]
            good = Xte.notna().all(axis=1).values
            if good.any():
                proba.iloc[np.arange(start, end)[good]] = m.predict_proba(Xte[good])[:, 1]
        start = end
    return proba


def model_proba(key: str, model: str, df: pd.DataFrame, fund, daily: bool) -> pd.Series:
    r = np.log(df.close).diff()
    if model == "mom1":
        return 0.5 + 0.5 * np.sign(r)
    if model == "rev1":
        return 0.5 - 0.5 * np.sign(r)
    if model == "ibs":
        ibs = (df.close - df.low) / (df.high - df.low).replace(0, np.nan)
        return pd.Series(np.where(ibs < 0.2, 1.0, np.where(ibs > 0.8, 0.0, 0.5)), index=df.index)
    if model == "lgbm":
        X = features(df, fund, daily=daily)
        fwd = np.log(df.close.shift(-1) / df.close)
        y = (fwd > 0).astype(float).where(fwd.notna() & (fwd != 0))
        bpd = {"1h": 24, "4h": 6, "1d": 1}[MARKETS[key][1]]
        min_train = 252 * 5 if key == "qqq_1d" else 365 * bpd
        retrain = 63 if key == "qqq_1d" else 30 * bpd
        return wf_proba(X, y, min_train, retrain)
    raise ValueError(model)


def evaluate(key: str, model: str, margin: float, holdout: bool = False, cache: dict | None = None) -> dict:
    df, fund, cost, splits, daily = load_market(key)
    ck = (key, model)
    if cache is not None and ck in cache:
        p = cache[ck]
    else:
        p = model_proba(key, model, df, fund, daily)
        if cache is not None:
            cache[ck] = p
    fwd = df.close.shift(-1) / df.close - 1
    up = (fwd > 0).where(fwd.notna() & (fwd != 0))
    pos = pd.Series(0.0, index=df.index)
    pos[p > 0.5 + margin] = 1.0
    if key != "qqq_1d" or model != "ibs":
        pos[p < 0.5 - margin] = -1.0
    pos[p.isna()] = 0.0
    pnl, _, _ = engine.run(df.close, pos, cost_bps=cost, funding=fund)
    sp = dict(splits)
    if not holdout:
        sp.pop("holdout")
    out = {}
    for name, (a, b) in sp.items():
        m = (df.index >= pd.Timestamp(a, tz="UTC")) & (df.index <= pd.Timestamp(b, tz="UTC")) & p.notna() & up.notna()
        if m.sum() < 50:
            continue
        u, pp = up[m].astype(bool), p[m]
        base = float(u.mean())
        pred_up = pp > 0.5
        decided = pp != 0.5
        acc = float((pred_up[decided] == u[decided]).mean()) if decided.any() else float("nan")
        conf = (pp - 0.5).abs() > margin
        acc_conf = float((pred_up[conf] == u[conf]).mean()) if conf.any() else float("nan")
        n_conf = int(conf.sum())
        maj = max(base, 1 - base)
        st = engine.stats(pnl[m], pos[m], df.index[m]).as_dict()
        out[name] = {"n": int(m.sum()), "base_up": round(base, 4), "majority": round(maj, 4),
                     "acc": round(acc, 4), "edge": round(acc - maj, 4),
                     "acc_conf": round(acc_conf, 4), "coverage": round(n_conf / m.sum(), 4),
                     # z-score of acc_conf vs 50% coin flip
                     "z_conf": round((acc_conf - 0.5) / np.sqrt(0.25 / max(n_conf, 1)), 2),
                     "sharpe": st["sharpe"], "total_return": st["total_return"], "max_dd": st["max_dd"],
                     "n_trades": st["n_trades"]}
    return out


def log(key, model, margin, res, holdout, note):
    rec = {"ts": datetime.now().isoformat(timespec="seconds"),
           "id": hashlib.sha1(f"{key}{model}{margin}{holdout}".encode()).hexdigest()[:10],
           "holdout_look": holdout, "market": key, "strategy": f"next_bar_{model}",
           "params": {"margin": margin}, "cost_bps": "6 crypto / 2 QQQ", "funding": True,
           "note": note, "results": res}
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    holdout = "--holdout" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    cache: dict = {}
    rows = []
    for key in MARKETS:
        models = ["mom1", "rev1", "lgbm"] + (["ibs"] if key.endswith("1d") else [])
        for model in models:
            for margin in ([0.0] if model in ("mom1", "rev1", "ibs") else [0.0, 0.02, 0.05]):
                name = f"{key}_{model}" + (f"_m{margin}" if model == "lgbm" else "")
                if only and not name.startswith(only):
                    continue
                res = evaluate(key, model, margin, holdout, cache)
                log(key, model, margin, res, holdout, "R3 next-bar direction")
                for split, r in res.items():
                    rows.append({"config": name, "split": split, **r})
                    print(f"{name:22s} {split:7s} n={r['n']:6d} up={r['base_up']:.3f} acc={r['acc']:.3f} "
                          f"edge={r['edge']:+.3f} acc_conf={r['acc_conf']:.3f} cov={r['coverage']:.2f} "
                          f"z={r['z_conf']:+.1f} sharpe={r['sharpe']:+.2f} ret={r['total_return']:+.1%}",
                          flush=True)
    out = ROOT / "results" / f"next_bar{'_holdout' if holdout else ''}.json"
    out.write_text(json.dumps(rows, indent=1))
