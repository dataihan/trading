"""Walk-forward gradient-boosting signal (H12 / H8).

- Features use only data up to bar close t.
- Label: sign of forward `horizon`-bar return (t -> t+horizon).
- Retrain every `retrain` bars on an expanding window, with a purge gap of
  `horizon` bars so no training label overlaps the prediction period.
- Decisions only every `step` bars (e.g. once a day) to limit turnover; the
  position is held until the next decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def features(df: pd.DataFrame, funding: pd.Series | None = None, daily: bool = False) -> pd.DataFrame:
    c = df.close
    r = np.log(c).diff()
    X = pd.DataFrame(index=df.index)
    lags = [1, 2, 5, 10, 21, 63, 126, 252] if daily else [1, 4, 12, 24, 72, 168, 336, 720]
    for k in lags:
        X[f"ret_{k}"] = np.log(c / c.shift(k))
    vw = 21 if daily else 24 * 7
    X["vol_s"] = r.rolling(vw).std()
    X["vol_l"] = r.rolling(vw * 4).std()
    X["vol_ratio"] = X.vol_s / X.vol_l
    for k in ([10, 50, 200] if daily else [24, 24 * 7, 24 * 30]):
        m = c.rolling(k).mean()
        X[f"dist_ma_{k}"] = (c - m) / (r.rolling(k).std() * c)
    rng = (df.high - df.low).replace(0, np.nan)
    X["ibs"] = (c - df.low) / rng
    X["range_rel"] = rng / c / X.vol_s
    hh = df.high.rolling(20 if daily else 24).max()
    ll = df.low.rolling(20 if daily else 24).min()
    X["pos_in_range"] = (c - ll) / (hh - ll).replace(0, np.nan)
    if "taker_buy_base" in df:
        imb = df.taker_buy_base / df.volume.replace(0, np.nan) - 0.5
        X["taker_imb_24"] = imb.rolling(24).mean()
        X["taker_imb_168"] = imb.rolling(168).mean()
        X["vol_z"] = np.log(df.quote_volume).diff(24)
    if funding is not None:
        f = funding.reindex(df.index).fillna(0.0)
        X["funding_7d"] = f.rolling(24 * 7).sum()
        X["funding_30d"] = f.rolling(24 * 30).sum()
    if not daily:
        X["dow"] = df.index.dayofweek
    else:
        X["dow"] = df.index.dayofweek
        X["dom"] = df.index.day
    return X.replace([np.inf, -np.inf], np.nan)


def walk_forward(df: pd.DataFrame, horizon: int = 24, step: int = 24, retrain: int = 24 * 30,
                 min_train: int = 24 * 365, margin: float = 0.03, long_only: bool = False,
                 funding: pd.Series | None = None, daily: bool = False, seed: int = 0,
                 n_estimators: int = 300, learning_rate: float = 0.03, num_leaves: int = 15,
                 min_child_samples: int = 200, decision_hour: int = 0) -> tuple[pd.Series, pd.Series]:
    import lightgbm as lgb

    X = features(df, funding, daily)
    fwd = np.log(df.close.shift(-horizon) / df.close)
    y = (fwd > 0).astype(float).where(fwd.notna())
    n = len(df)
    # decision bars
    if daily:
        is_dec = np.zeros(n, bool)
        is_dec[::step] = True
    else:
        is_dec = (df.index.hour == decision_hour)
    proba = pd.Series(np.nan, index=df.index)
    start = min_train
    while start < n:
        end = min(start + retrain, n)
        tr_end = start - horizon  # purge
        tr_idx = np.arange(0, tr_end)
        tr_idx = tr_idx[is_dec[tr_idx]]  # one sample per decision -> less overlap
        Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
        ok = ytr.notna() & Xtr.notna().all(axis=1)
        if ok.sum() > 200:
            model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                       num_leaves=num_leaves, min_child_samples=min(min_child_samples, int(ok.sum() / 5)),
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                       reg_lambda=1.0, random_state=seed, verbose=-1)
            model.fit(Xtr[ok], ytr[ok])
            te = np.arange(start, end)
            te = te[is_dec[te]]
            Xte = X.iloc[te]
            good = Xte.notna().all(axis=1).values
            if good.any():
                proba.iloc[te[good]] = model.predict_proba(Xte[good])[:, 1]
        start = end
    sig = pd.Series(0.0, index=df.index)
    sig[proba > 0.5 + margin] = 1.0
    if not long_only:
        sig[proba < 0.5 - margin] = -1.0
    # hold decision until next decision bar
    held = sig.where(pd.Series(is_dec, index=df.index)).ffill().fillna(0.0)
    held[proba.isna() & pd.Series(is_dec, index=df.index)] = 0.0
    held = held.where(pd.Series(is_dec, index=df.index)).ffill().fillna(0.0)
    return held, proba
