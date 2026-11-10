"""Score TradingAgents weekly decisions with costs (6 bps/side) and funding, on Binance USDT-M perps.

    PYTHONPATH=src python3 tools/tradingagents_bt/evaluate.py

Entry at the first daily close AFTER the decision date (the framework reads news up to that day),
held until the next decision's entry. Two mappings:
  long/short: Buy, Overweight -> +1; Hold -> 0; Underweight, Sell -> -1
  long-only : Buy, Overweight -> +1; else 0
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from trading.data.loaders import binance_funding, binance_klines, binance_klines_daily  # noqa: E402

LS = {"Buy": 1, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -1}
SYM = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}


def daily(sym):
    d = binance_klines(sym, "1d")
    extra = binance_klines_daily(sym, "1d", since=d.index.max() + pd.Timedelta(days=1))
    if not extra.empty:
        d = pd.concat([d, extra])
    return d[~d.index.duplicated(keep="last")].sort_index().close


def score(pos, close, fund, bps=6.0):
    ret = close.pct_change().shift(-1)
    pnl = pos * ret - pos.diff().abs().fillna(pos.abs()) * bps / 1e4 - pos * fund.shift(-1)
    pnl = pnl.dropna()
    eq = (1 + pnl).cumprod()
    return {"return": round(float(eq.iloc[-1] - 1), 4),
            "sharpe": round(float(pnl.mean() / pnl.std() * np.sqrt(365)), 2) if pnl.std() > 0 else 0.0,
            "max_dd": round(float((eq / eq.cummax() - 1).min()), 4), "days": len(pnl)}


rows = [json.loads(x) for x in (ROOT / "results" / "tradingagents_bt.jsonl").read_text().splitlines() if x.strip()]
df = pd.DataFrame(rows)
print(f"decisions: {len(df)}  errors/REVIEW: {int(df.signal.isna().sum() + (df.signal == 'REVIEW').sum())}  "
      f"cost: ${df.cost_usd.sum():.2f}  avg/run: ${df.cost_usd.mean():.3f}")
print(df.signal.value_counts().to_dict())
for tk, g in df.groupby("ticker"):
    sym = SYM[tk]
    close = daily(sym)
    f = binance_funding(sym)
    fund = f.groupby(f.index.floor("1D")).sum().reindex(close.index).fillna(0.0)
    g = g.sort_values("date")
    start = pd.Timestamp(g.date.iloc[0], tz="UTC") + pd.Timedelta(days=1)
    end = pd.Timestamp(g.date.iloc[-1], tz="UTC") + pd.Timedelta(days=8)
    idx = close.index[(close.index >= start) & (close.index <= end)]
    for name, mp in [("long/short", LS), ("long-only", {k: max(v, 0) for k, v in LS.items()})]:
        sig = pd.Series({pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=1): mp.get(s, 0)
                         for d, s in zip(g.date, g.signal)})
        pos = sig.reindex(idx).ffill().fillna(0.0)
        print(f"{tk} {name:10s}", score(pos, close.reindex(idx), fund.reindex(idx)))
    print(f"{tk} buy&hold  ", score(pd.Series(1.0, index=idx), close.reindex(idx), fund.reindex(idx)))
    # hit rate: direction of the following week
    nxt = []
    for d, s in zip(g.date, g.signal):
        a = pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=1)
        if a in close.index and a + pd.Timedelta(days=7) in close.index and LS.get(s, 0) != 0:
            nxt.append(np.sign(close[a + pd.Timedelta(days=7)] / close[a] - 1) == LS[s])
    if nxt:
        print(f"{tk} directional hit rate on non-Hold calls: {np.mean(nxt):.1%} (n={len(nxt)})")
