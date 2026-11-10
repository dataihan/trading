"""Weekly TradingAgents decisions for BTC, gold and Nasdaq 100 over the last year (research backtest, no orders).

Run with the TradingAgents venv:  /home/user/tauricresearch/venv/bin/python tools/tradingagents_bt/run.py [--one]
Needs ANTHROPIC_API_KEY. Model: Claude Sonnet 5.5 for both tiers; analysts: market + news.
Appends one line per (ticker, date) to results/tradingagents_bt.jsonl; resumes where it stopped.
Stops before the next call once the measured spend passes BUDGET_USD (default 50).
Runs newest week first, so a budget stop still leaves the most recent (least model-contaminated) weeks.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tradingagents_bt.jsonl"
TICKERS = {"BTC-USD": "crypto", "GC=F": "stock", "^NDX": "stock"}   # ticker -> asset_type
MODEL = os.getenv("TA_MODEL", "claude-sonnet-5-5")
PRICE_IN, PRICE_OUT = 2.0, 10.0          # USD per 1M tokens, Claude Sonnet 5.5
BUDGET = float(os.getenv("BUDGET_USD", "50"))


def mondays(start: date, end: date):
    d = start + timedelta(days=(7 - start.weekday()) % 7)
    while d <= end:
        yield d
        d += timedelta(days=7)


def done_keys():
    if not OUT.exists():
        return set(), 0.0
    rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
    return {(r["ticker"], r["date"]) for r in rows}, sum(r.get("cost_usd", 0.0) for r in rows)


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set: add it to the cloud environment variables, then open a new session")
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from tradingagents.default_config import build_default_config
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    cfg = build_default_config()
    cfg.update({"llm_provider": "anthropic", "deep_think_llm": MODEL, "quick_think_llm": MODEL,
                "anthropic_effort": os.getenv("TA_EFFORT", "medium"), "max_debate_rounds": 1,
                "max_risk_discuss_rounds": 1, "checkpoint_enabled": False})
    end = date.today() - timedelta(days=7)               # need a week of outcome after the decision
    weeks = sorted(mondays(end - timedelta(days=365), end), reverse=True)
    grid = [(t, d.isoformat()) for d in weeks for t in TICKERS]
    if "--one" in sys.argv:
        grid = grid[:1]
    seen, spent = done_keys()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ticker, d in grid:
        if (ticker, d) in seen:
            continue
        if spent >= BUDGET:
            print(f"budget reached: ${spent:.2f} >= ${BUDGET:.2f}; stopping")
            break
        usage = UsageMetadataCallbackHandler()
        ta = TradingAgentsGraph(selected_analysts=["market", "news"], config=cfg, callbacks=[usage])
        t0 = time.time()
        try:
            _, signal = ta.propagate(ticker, d, asset_type=TICKERS[ticker])
            err = None
        except Exception as e:  # keep going; record the failure
            signal, err = None, str(e)[:300]
        tin = sum(v.get("input_tokens", 0) for v in usage.usage_metadata.values())
        tout = sum(v.get("output_tokens", 0) for v in usage.usage_metadata.values())
        cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
        spent += cost
        rec = {"ticker": ticker, "date": d, "signal": signal, "error": err, "model": MODEL,
               "tokens_in": tin, "tokens_out": tout, "cost_usd": round(cost, 4),
               "seconds": round(time.time() - t0, 1)}
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), f"| total ${spent:.2f}", flush=True)


if __name__ == "__main__":
    main()
