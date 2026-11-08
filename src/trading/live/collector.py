"""Hourly market-microstructure collector (data for autoresearch round 3).

Binance does not offer free history for order book depth, and open interest /
long-short ratios only go back ~30 days via the API, so we record them ourselves.

    python -m trading.live.collector            # run hourly (systemd timer)

Writes one JSON line per symbol per run to data/live/micro/YYYY-MM-DD.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "live" / "micro"
OUT.mkdir(parents=True, exist_ok=True)
BASE = os.getenv("COLLECT_FAPI_BASE", "https://fapi.binance.com")
SYMBOLS = os.getenv("COLLECT_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XAUUSDT").split(",")


def get(path, **params):
    r = requests.get(f"{BASE}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def snapshot(sym: str) -> dict:
    rec = {"symbol": sym}
    prem = get("/fapi/v1/premiumIndex", symbol=sym)
    rec.update(mark=float(prem["markPrice"]), index=float(prem["indexPrice"]),
               funding=float(prem["lastFundingRate"]))
    rec["open_interest"] = float(get("/fapi/v1/openInterest", symbol=sym)["openInterest"])
    book = get("/fapi/v1/depth", symbol=sym, limit=100)
    bids = [(float(p), float(q)) for p, q in book["bids"]]
    asks = [(float(p), float(q)) for p, q in book["asks"]]
    mid = (bids[0][0] + asks[0][0]) / 2
    rec["spread_bps"] = (asks[0][0] - bids[0][0]) / mid * 1e4
    for bps in (10, 50, 100):  # notional depth within +-bps of mid
        b = sum(p * q for p, q in bids if p >= mid * (1 - bps / 1e4))
        a = sum(p * q for p, q in asks if p <= mid * (1 + bps / 1e4))
        rec[f"depth_bid_{bps}"], rec[f"depth_ask_{bps}"] = round(b), round(a)
        rec[f"imb_{bps}"] = round((b - a) / (b + a), 4) if a + b else 0.0
    for name, path in [("taker_ls", "/futures/data/takerlongshortRatio"),
                       ("global_ls", "/futures/data/globalLongShortAccountRatio"),
                       ("top_ls", "/futures/data/topLongShortPositionRatio")]:
        try:
            d = get(path, symbol=sym, period="1h", limit=1)[-1]
            rec[name] = float(d.get("buySellRatio", d.get("longShortRatio", 0)))
        except Exception as e:
            rec[name + "_err"] = str(e)[:80]
    return rec


def main():
    now = datetime.now(timezone.utc)
    path = OUT / f"{now:%Y-%m-%d}.jsonl"
    ok = 0
    with path.open("a") as f:
        for s in SYMBOLS:
            try:
                rec = {"ts": now.isoformat(timespec="seconds"), **snapshot(s)}
                ok += 1
            except Exception as e:
                rec = {"ts": now.isoformat(timespec="seconds"), "symbol": s, "error": str(e)[:200]}
            f.write(json.dumps(rec) + "\n")
    print(f"collected {ok}/{len(SYMBOLS)} -> {path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
