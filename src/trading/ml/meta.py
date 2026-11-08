"""Round 2: ML as a FILTER, not a predictor.

1. Meta-labeling (Lopez de Prado, AFML ch.3): a primary rule decides the SIDE;
   a walk-forward LightGBM model predicts whether that bet will be profitable
   over `horizon` bars (net of cost) and only lets confident bets through.
2. HMM regime filter: Gaussian HMM on (daily return, rolling vol), refit
   walk-forward; FORWARD-filtered state probabilities only (no smoothing,
   which would leak the future). Position is scaled by 1 - P(worst state).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading.ml.walkforward import features


def meta_filter(df: pd.DataFrame, primary: pd.Series, horizon: int, decision: np.ndarray,
                retrain: int, min_train: int, threshold: float = 0.5, mode: str = "filter",
                cost: float = 0.0012, funding: pd.Series | None = None, daily: bool = False,
                seed: int = 0, n_estimators: int = 200, learning_rate: float = 0.03,
                num_leaves: int = 7, min_child_samples: int = 50) -> tuple[pd.Series, pd.Series]:
    import lightgbm as lgb

    X = features(df, funding, daily)
    X["primary"] = primary.values
    side = np.sign(primary)
    fwd = np.log(df.close.shift(-horizon) / df.close)
    y = ((side * fwd) > cost).astype(float).where(fwd.notna() & (side != 0))
    n = len(df)
    event = decision & (side.values != 0)
    proba = pd.Series(np.nan, index=df.index)
    start = min_train
    while start < n:
        end = min(start + retrain, n)
        tr = np.arange(0, max(start - horizon, 0))  # purge overlap with label horizon
        tr = tr[event[tr]]
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        ok = (ytr.notna() & Xtr.notna().all(axis=1)).values
        te = np.arange(start, end)
        te = te[event[te]]
        if ok.sum() >= 100 and len(te):
            m = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                   num_leaves=num_leaves, min_child_samples=min_child_samples,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   reg_lambda=1.0, random_state=seed, verbose=-1)
            m.fit(Xtr[ok], ytr[ok])
            Xte = X.iloc[te]
            g = Xte.notna().all(axis=1).values
            if g.any():
                proba.iloc[te[g]] = m.predict_proba(Xte[g])[:, 1]
        start = end
    dec = pd.Series(decision, index=df.index)
    if mode == "filter":
        mult = (proba > threshold).astype(float)
    else:  # "scale": linear sizing around 0.5, capped to [0, 1]
        mult = ((proba - 0.5) * 4 + 1).clip(0, 1)
    mult = mult.where(proba.notna(), 1.0)  # before the model exists: pass primary through
    mult = mult.where(dec).ffill().fillna(1.0)
    return primary * mult, proba


# ------------------------------------------------------------------ HMM regime
def _forward_filter(model, X: np.ndarray) -> np.ndarray:
    """Normalised forward recursion -> P(state_t | x_1..x_t). No smoothing."""
    from scipy.stats import multivariate_normal

    k = model.n_components
    covs = model.covars_
    lik = np.column_stack([multivariate_normal(model.means_[j], covs[j], allow_singular=True).pdf(X)
                           for j in range(k)]) + 1e-300
    A = model.transmat_
    out = np.zeros((len(X), k))
    a = model.startprob_ * lik[0]
    out[0] = a / a.sum()
    for t in range(1, len(X)):
        a = (out[t - 1] @ A) * lik[t]
        out[t] = a / a.sum()
    return out


def hmm_regime(close: pd.Series, n_states: int = 3, vol_win: int = 20, refit_every: int = 63,
               min_train: int = 500, seed: int = 0) -> pd.Series:
    """Daily close -> multiplier in [0,1] = 1 - P(state with lowest mean return | info up to t)."""
    from hmmlearn.hmm import GaussianHMM

    r = np.log(close).diff()
    feat = pd.DataFrame({"r": r, "v": r.rolling(vol_win).std()}).dropna()
    Z = ((feat - feat.expanding(250).mean()) / feat.expanding(250).std()).dropna()
    vals = Z.values
    mult = pd.Series(np.nan, index=Z.index)
    start = min_train
    while start < len(Z):
        end = min(start + refit_every, len(Z))
        m = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=seed)
        m.fit(vals[:start])
        post = _forward_filter(m, vals[:end])[start:end]
        worst = int(np.argmin(m.means_[:, 0]))
        mult.iloc[start:end] = 1 - post[:, worst]
        start = end
    return mult.reindex(close.index).ffill()
