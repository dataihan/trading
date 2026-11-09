"""Walk-forward LightGBM probability that close[t+H] > close[t] (round 3c), with disk cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "processed"


def wf_proba_h(X: pd.DataFrame, y: pd.Series, dec: np.ndarray, h: int, min_train: int, retrain: int,
               seed: int = 0) -> pd.Series:
    """Expanding window, retrained every `retrain` bars, trained and predicted on decision bars only.
    Purge: training row i is used only if its label is known at prediction time (i + h <= start)."""
    import lightgbm as lgb
    n = len(X)
    proba = pd.Series(np.nan, index=X.index)
    idx_dec = np.flatnonzero(dec)
    start = min_train
    while start < n:
        end = min(start + retrain, n)
        tr = idx_dec[idx_dec + h <= start]
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        ok = (ytr.notna() & Xtr.notna().all(axis=1)).values
        if ok.sum() > 200:
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7,
                                   min_child_samples=max(30, int(ok.sum() / 50)), subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                   random_state=seed, verbose=-1)
            m.fit(Xtr[ok], ytr[ok])
            te = idx_dec[(idx_dec >= start) & (idx_dec < end)]
            Xte = X.iloc[te]
            good = Xte.notna().all(axis=1).values
            if good.any():
                proba.iloc[te[good]] = m.predict_proba(Xte[good])[:, 1]
        start = end
    return proba


def proba_1h(df: pd.DataFrame, symbol: str, h: int) -> pd.Series:
    """P(up over next h bars) on decision bars (every h bars from the first bar); NaN elsewhere."""
    from trading.data.loaders import binance_funding
    from trading.ml.walkforward import features
    key = symbol.lower().replace("usdt", "") + "_1h"
    disk = CACHE / f"multibar_proba_{key}_h{h}.parquet"
    if disk.exists():
        p = pd.read_parquet(disk)["p"]
        if p.index.equals(df.index):
            return p
    fund = binance_funding(symbol)
    fund = fund.groupby(fund.index.floor("1h")).sum()
    fwd = np.log(df.close.shift(-h) / df.close)
    y = (fwd > 0).astype(float).where(fwd.notna() & (fwd != 0))
    dec = (np.arange(len(df)) % h) == 0
    p = wf_proba_h(features(df, fund, daily=False), y, dec, h, min_train=365 * 24, retrain=30 * 24)
    CACHE.mkdir(parents=True, exist_ok=True)
    p.to_frame("p").to_parquet(disk)
    return p
