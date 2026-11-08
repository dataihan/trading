"""Live (paper) vs backtest comparison + go/no-go check.

    python -m trading.live.report

For the paper-trading period it:
1. re-runs the backtest of the same frozen portfolio over the same dates,
2. reads actual paper equity snapshots from data/live/equity.jsonl,
3. prints tracking gap and checks the rules from docs/paper_trading.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "autoresearch"))

EQUITY = ROOT / "data" / "live" / "equity.jsonl"

# pre-registered thresholds (see docs/paper_trading.md) — do not edit after paper trading starts
KILL_DD = -0.15          # 1.5x the worst backtest drawdown (-10.5%)
MAX_GAP_ANN = 0.05       # live may lag backtest by at most 5%/yr (execution costs)
MIN_DAYS = 90


def snapshot_equity():
    """Append current paper equity (Binance testnet + Alpaca paper) — run daily."""
    from datetime import datetime, timezone

    from trading.live.clients import AlpacaPaper, BinanceFutures
    from trading.live.runner import load_env
    load_env()
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for name, fn in [("binance", lambda: BinanceFutures().balance_usdt()),
                     ("alpaca", lambda: AlpacaPaper().equity())]:
        try:
            rec[name] = fn()
        except Exception as e:
            rec[name + "_err"] = str(e)[:120]
    with EQUITY.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec)


def main():
    from portfolio import combine
    import os
    spec = json.loads(Path(os.getenv("PORTFOLIO", ROOT / "strategies" / "portfolio_v1.json")).read_text())
    if not EQUITY.exists():
        print("no paper equity yet — run `python -m trading.live.report snapshot` daily")
        return
    eq = pd.DataFrame([json.loads(l) for l in EQUITY.read_text().splitlines()])
    eq["ts"] = pd.to_datetime(eq.ts)
    venues = ["binance", "alpaca"] if any(s["market"].startswith("yahoo:") for s in spec["sleeves"]) else ["binance"]
    eq = eq.set_index("ts").filter(venues).ffill().dropna()
    capital = float(os.getenv("PAPER_CAPITAL_USD", "10000"))
    # accounts hold more than the strategy capital (testnet/paper defaults), so measure
    # PnL in USD relative to the capital the bot actually allocates
    live_total = capital + (eq.sum(axis=1) - eq.sum(axis=1).iloc[0])
    live_ret = live_total.iloc[-1] / capital - 1
    days = (eq.index[-1] - eq.index[0]).days
    bt = combine(spec)["portfolio"]
    bt = bt[(bt.index >= eq.index[0].floor("D")) & (bt.index <= eq.index[-1])]
    bt_ret = float((1 + bt).prod() - 1)
    live_curve = live_total / live_total.iloc[0]
    dd = float((live_curve / live_curve.cummax() - 1).min())
    gap_ann = (live_ret - bt_ret) * 365 / max(days, 1)
    print(f"days={days}  live={live_ret:+.2%}  backtest(same dates)={bt_ret:+.2%}  "
          f"gap/yr={gap_ann:+.2%}  live maxDD={dd:.2%}")
    if dd <= KILL_DD:
        print("KILL: drawdown beyond pre-registered limit -> stop bot, back to research (docs/paper_trading.md §3)")
    elif days < MIN_DAYS:
        print(f"WAIT: only {days} days (< {MIN_DAYS}); too early to judge")
    elif gap_ann < -MAX_GAP_ANN:
        print("EXECUTION PROBLEM: live lags backtest on the same dates -> fix execution (§2A)")
    elif bt_ret < 0:
        print("STRATEGY FLAT/DOWN but live tracks backtest -> regime issue, check §2B before changing anything")
    else:
        print("OK: live tracks backtest and is profitable")


if __name__ == "__main__":
    snapshot_equity() if sys.argv[1:] == ["snapshot"] else main()
