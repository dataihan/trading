"""Paper-trading runner for a frozen portfolio (default strategies/portfolio_v1.json).

    python -m trading.live.runner crypto [--dry-run]   # run daily at 00:01 UTC
    python -m trading.live.runner qqq    [--dry-run]   # run on trading days at 15:45 ET

Signals reuse the SAME strategy functions as the backtest (trading.strategies.library),
so live and backtest cannot drift apart silently. Every decision is appended to
data/live/ledger.jsonl for the live-vs-backtest comparison (trading.live.report).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from trading.data.loaders import KLINE_COLS, binance_klines, binance_klines_daily  # noqa: E402
from trading.live.clients import AlpacaPaper, BinanceFutures, round_step  # noqa: E402
from trading.strategies.library import REGISTRY  # noqa: E402

LIVE_DIR = ROOT / "data" / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
LEDGER = LIVE_DIR / "ledger.jsonl"


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def log(rec: dict):
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **rec}
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


def portfolio(path: str | None = None) -> dict:
    return json.loads(Path(path or os.getenv("PORTFOLIO") or ROOT / "strategies" / "portfolio_v1.json").read_text())


# ------------------------------------------------------------------ crypto
def recent_hourly(symbol: str, client: BinanceFutures | None, days: int = 220) -> pd.DataFrame:
    """Last `days` of closed 1h bars. Source order: real Binance fapi -> archive cache.
    Signals always use REAL market prices (testnet prices are not real)."""
    try:
        import requests
        base = "https://fapi.binance.com" if os.getenv("SIGNAL_SOURCE", "real") == "real" else client.base
        rows, end = [], None
        while len(rows) < days * 24:
            p = {"symbol": symbol, "interval": "1h", "limit": 1500, **({"endTime": end} if end else {})}
            r = requests.get(f"{base}/fapi/v1/klines", params=p, timeout=30)  # public, read-only
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows = batch + rows
            end = batch[0][0] - 1
        df = pd.DataFrame([r[:12] for r in rows], columns=KLINE_COLS)
        df.index = pd.to_datetime(df.open_time.astype("int64"), unit="ms", utc=True)
        df = df[["open", "high", "low", "close", "volume", "quote_volume", "trades",
                 "taker_buy_base"]].astype(float)
        df = df[~df.index.duplicated()].sort_index()
        source = "fapi"
    except Exception as e:  # geo-blocked (HTTP 451) etc. -> archive (lags ~1 day)
        print(f"[warn] live klines failed for {symbol}: {e}; using data.binance.vision archive")
        df = binance_klines(symbol, "1h", start=(pd.Timestamp.now() - pd.Timedelta(days=days + 40)).strftime("%Y-%m"))
        # monthly archive ends at the last full month -> top up with daily files (lag ~1 day)
        recent = binance_klines_daily(symbol, "1h", since=df.index.max() + pd.Timedelta(hours=1))
        if not recent.empty:
            df = pd.concat([df, recent])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        source = "archive"
    now = pd.Timestamp.now(tz="UTC").floor("h")
    df = df[df.index < now]  # drop the still-open bar
    df.attrs["source"] = source
    return df


def run_crypto(spec: dict, dry: bool):
    client = None if dry else BinanceFutures()
    capital = float(os.getenv("PAPER_CAPITAL_USD", "10000"))
    for sl in spec["sleeves"]:
        src, sym, _ = sl["market"].split(":")
        if src != "binance":
            continue
        df = recent_hourly(sym, client)
        params = {**sl.get("params", {}), "rebalance": 1}  # take the signal at the latest close
        pos = float(REGISTRY[sl["strategy"]](df, **params)["position"].iloc[-1])
        px = float(df.close.iloc[-1])
        target_usd = capital * sl["weight"] * pos
        rec = {"venue": "binance", "sleeve": sl["name"], "symbol": sym, "signal_pos": round(pos, 4),
               "target_usd": round(target_usd, 2), "signal_close": px, "bar": str(df.index[-1]),
               "data_source": df.attrs.get("source"), "dry_run": dry}
        if not dry:
            f = client.filters(sym)
            mark = client.price(sym)
            cur = client.position_qty(sym)
            target_qty = round_step(target_usd / mark, f["step"])
            delta = round_step(target_qty - cur, f["step"])
            rec.update({"exec_price_ref": mark, "current_qty": cur, "target_qty": target_qty, "delta_qty": delta})
            if abs(delta) * mark >= max(f["min_notional"], 5):
                try:
                    client.set_leverage(sym, int(os.getenv("LEVERAGE", "2")))
                    o = client.market_order(sym, delta, reduce_only=abs(target_qty) < abs(cur) and target_qty * cur >= 0)
                    rec.update({"order_id": o.get("orderId"), "avg_price": o.get("avgPrice"), "status": o.get("status")})
                except Exception as e:
                    rec["error"] = str(e)[:300]
            else:
                rec["skipped"] = "delta below min notional"
        log(rec)


# ------------------------------------------------------------------ QQQ
def run_qqq(spec: dict, dry: bool):
    sleeves = [s for s in spec["sleeves"] if s["market"].startswith("yahoo:QQQ")]
    if not sleeves:
        return
    capital = float(os.getenv("PAPER_CAPITAL_USD", "10000"))
    if dry:  # dry-run from cached Yahoo daily data: last completed bar
        from trading.data.loaders import yahoo
        hist = yahoo("QQQ", "1d", "max")
        df = hist.copy()
        px = float(df.close.iloc[-1])
        # append future business days so calendar rules (turn-of-month) know what comes next
        fut = pd.bdate_range(df.index[-1].floor("D") + pd.Timedelta(days=1), periods=30, tz="UTC")
        df = pd.concat([df, pd.DataFrame({"open": px, "high": px, "low": px, "close": px}, index=fut)])
        df.attrs["last_real"] = len(hist) - 1
    else:
        al = AlpacaPaper()
        if not al.clock()["is_open"]:
            log({"venue": "alpaca", "skipped": "market closed"})
            return
        bar, px = al.today_bar("QQQ")
        today = pd.Timestamp(bar["t"]).tz_convert("UTC").floor("D")
        # build a calendar-accurate frame: past trading days of this+next month + today's partial bar
        start = (today - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        end = (today + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        days = [pd.Timestamp(d["date"], tz="UTC") for d in al.calendar(start, end)]
        df = pd.DataFrame(index=pd.DatetimeIndex(days), data={"open": px, "high": px, "low": px, "close": px})
        df.loc[today, ["open", "high", "low", "close"]] = [bar["o"], max(bar["h"], px), min(bar["l"], px), px]
        df = df[df.index <= today + pd.Timedelta(days=40)]
    total_usd, parts = 0.0, {}
    for sl in sleeves:
        out = REGISTRY[sl["strategy"]](df, **sl.get("params", {}))["position"]
        if dry:
            pos = float(out.iloc[df.attrs["last_real"]])
        else:
            pos = float(out.loc[df.index[df.index <= pd.Timestamp.now(tz="UTC").floor("D")][-1]])
        parts[sl["name"]] = pos
        total_usd += capital * sl["weight"] * pos
    rec = {"venue": "alpaca", "symbol": "QQQ", "signals": parts, "target_usd": round(total_usd, 2),
           "ref_price": px, "dry_run": dry}
    if not dry:
        cur = al.position_qty("QQQ")
        target = int(total_usd // px)
        delta = target - cur
        rec.update({"current_qty": cur, "target_qty": target, "delta_qty": delta})
        if abs(delta) >= 1:
            try:
                o = al.order_on_close("QQQ", delta)
                rec.update({"order_id": o.get("id"), "status": o.get("status")})
            except Exception as e:
                rec["error"] = str(e)[:300]
    log(rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("venue", choices=["crypto", "qqq"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--portfolio")
    a = ap.parse_args()
    load_env()
    spec = portfolio(a.portfolio)
    (run_crypto if a.venue == "crypto" else run_qqq)(spec, a.dry_run)
