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


def metric_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Open interest / long-short / taker-ratio features (round 4). Value at bar t = last 5-min stamp
    inside bar t (see loaders.binance_metrics), so it is known at the close of t."""
    from trading.data.loaders import binance_metrics
    m = binance_metrics(symbol).reindex(df.index).ffill(limit=3)
    X = pd.DataFrame(index=df.index)
    oi = np.log(m.sum_open_interest_value)
    for k in (1, 4, 24, 168):
        X[f"oi_chg_{k}"] = oi.diff(k)
    X["oi_z_720"] = (oi - oi.rolling(720).mean()) / oi.rolling(720).std()
    ret24 = np.log(df.close).diff(24)
    X["oi_price_div_24"] = np.sign(X.oi_chg_24) * np.sign(ret24)
    for col, name in [("sum_toptrader_long_short_ratio", "top_pos_ls"),
                      ("count_toptrader_long_short_ratio", "top_acct_ls"),
                      ("count_long_short_ratio", "global_ls")]:
        v = np.log(m[col])
        X[name] = v
        X[f"{name}_chg_24"] = v.diff(24)
        X[f"{name}_z_720"] = (v - v.rolling(720).mean()) / v.rolling(720).std()
    tk = np.log(m.sum_taker_long_short_vol_ratio)
    X["taker_ls_4"] = tk.rolling(4).mean()
    X["taker_ls_24"] = tk.rolling(24).mean()
    return X.replace([np.inf, -np.inf], np.nan)


def proba_1h(df: pd.DataFrame, symbol: str, h: int, extra: str | None = None) -> pd.Series:
    """P(up over next h bars) on decision bars (every h bars from the first bar); NaN elsewhere.

    extra=None: price/volume/funding features (round 3c).
    extra="metrics": + open interest / long-short features; rows before metrics exist are unusable.
    extra="base_late": price features only, but restricted to the same rows as "metrics" (fair A/B).
    """
    from trading.data.loaders import binance_funding
    from trading.ml.walkforward import features
    key = symbol.lower().replace("usdt", "") + "_1h"
    disk = CACHE / f"multibar_proba_{key}_h{h}{'_' + extra if extra else ''}.parquet"
    if disk.exists():
        p = pd.read_parquet(disk)["p"]
        if p.index.equals(df.index):
            return p
    fund = binance_funding(symbol)
    fund = fund.groupby(fund.index.floor("1h")).sum()
    fwd = np.log(df.close.shift(-h) / df.close)
    y = (fwd > 0).astype(float).where(fwd.notna() & (fwd != 0))
    dec = (np.arange(len(df)) % h) == 0
    X = features(df, fund, daily=False)
    min_train = 365 * 24
    if extra:
        mx = metric_features(df, symbol)
        first = mx.dropna().index.min()
        if extra == "metrics":
            X = X.join(mx)
        X = X[X.index >= first].reindex(X.index)      # same usable rows for both arms
        min_train = int((df.index < first).sum()) + 365 * 24
    p = wf_proba_h(X, y, dec, h, min_train=min_train, retrain=30 * 24)
    CACHE.mkdir(parents=True, exist_ok=True)
    p.to_frame("p").to_parquet(disk)
    return p
