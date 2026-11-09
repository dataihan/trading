"""Multi-coin universe for cross-sectional research (round 5).

Coins: liquid USDT-M perps listed on Binance by ~2021 that still trade today. This is a SURVIVORSHIP-BIASED
list (delisted coins are missing), which flatters long-the-winners strategies; keep that in mind.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from trading.data.loaders import binance_funding, binance_klines

UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT",
            "LINKUSDT", "DOTUSDT", "BCHUSDT", "TRXUSDT", "ETCUSDT", "XLMUSDT", "ATOMUSDT", "AVAXUSDT",
            "UNIUSDT", "FILUSDT", "AAVEUSDT", "NEARUSDT", "ALGOUSDT", "XTZUSDT", "VETUSDT", "SANDUSDT",
            "MANAUSDT", "AXSUSDT", "EOSUSDT", "THETAUSDT"]


def _one(sym: str, spot: bool):
    perp = binance_klines(sym, "1d")
    sp = None
    if spot:
        try:
            sp = binance_klines(sym, "1d", market="spot")
        except RuntimeError:
            sp = None
    fund = binance_funding(sym)
    return sym, perp, sp, fund


def load_universe(spot: bool = True, workers: int = 4) -> dict[str, pd.DataFrame]:
    """Wide daily frames: perp close/open, spot close, daily funding sum (fraction, paid by longs)."""
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda s: _one(s, spot), UNIVERSE))
    close = pd.DataFrame({s: p.close for s, p, _, _ in res}).sort_index()
    qv = pd.DataFrame({s: p.quote_volume for s, p, _, _ in res}).sort_index()
    spot_close = pd.DataFrame({s: sp.close for s, _, sp, _ in res if sp is not None}).sort_index()
    fund = pd.DataFrame({s: f.groupby(f.index.floor("1D")).sum() for s, _, _, f in res}).sort_index()
    return {"close": close, "quote_volume": qv, "spot_close": spot_close,
            "funding": fund.reindex(close.index).fillna(0.0)}
