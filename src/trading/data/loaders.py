"""Market data loaders with local parquet cache.

Sources (no API key needed):
- Binance public archive (data.binance.vision): USDT-M futures + spot klines.
  Note: api.binance.com is geo-blocked from some regions (HTTP 451); the archive is not.
- Yahoo Finance chart API: gold (GC=F), silver (SI=F), oil (CL=F), Nasdaq (^NDX, QQQ), US stocks.
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "raw"
CACHE.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (trading-research)"}
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def _months(start: str, end: str | None = None):
    s = pd.Period(start, "M")
    e = pd.Period(end or date.today().isoformat()[:7], "M")
    while s <= e:
        yield s
        s += 1


def binance_klines(symbol: str, interval: str = "1h", start: str = "2020-01",
                   end: str | None = None, market: str = "futures/um") -> pd.DataFrame:
    """Download monthly kline zips from data.binance.vision, cached as parquet.

    market: 'futures/um' (USDT-M perpetual) or 'spot'.
    Returns UTC-indexed OHLCV DataFrame.
    """
    tag = market.replace("/", "_")
    out = CACHE / f"binance_{tag}_{symbol}_{interval}.parquet"
    cached = pd.read_parquet(out) if out.exists() else None
    months = list(_months(start, end))
    if cached is not None and len(cached):
        first = pd.Period(cached.index.min().strftime("%Y-%m"), "M")
        last = pd.Period(cached.index.max().strftime("%Y-%m"), "M")
        fresh = (pd.Timestamp.now(tz="UTC") - cached.index.max()).days < 40
        # fetch only what the cache lacks: history before it, and (if stale) its last month onward
        months = [m for m in months if m < first or (not fresh and m >= last)]
        if not months:
            return cached
    frames = [] if cached is None else [cached.reset_index(drop=True).assign(
        open_time=(cached.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1ms"))]
    for m in months:
        url = (f"https://data.binance.vision/data/{market}/monthly/klines/"
               f"{symbol}/{interval}/{symbol}-{interval}-{m}.zip")
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code != 200:
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = z.read(z.namelist()[0]).decode()
        first_line = raw.split("\n", 1)[0]
        has_header = not first_line[:1].isdigit()
        d = pd.read_csv(io.StringIO(raw), header=0 if has_header else None)
        d = d.iloc[:, :12]
        d.columns = KLINE_COLS
        frames.append(d)
    if not frames:
        raise RuntimeError(f"no data for {symbol} {interval}")
    df = pd.concat(frames, ignore_index=True)
    ot = pd.to_numeric(df["open_time"])
    # spot archive switched to microseconds in 2025
    ot = ot.where(ot < 1e14, ot // 1000)
    df.index = pd.to_datetime(ot, unit="ms", utc=True)
    df = df[["open", "high", "low", "close", "volume", "quote_volume", "trades",
             "taker_buy_base"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(out)
    return df


def binance_klines_daily(symbol: str, interval: str = "1h", since: pd.Timestamp | None = None,
                         market: str = "futures/um") -> pd.DataFrame:
    """Daily kline zips from data.binance.vision, from `since` (UTC date) up to yesterday.

    Fills the gap after the last monthly archive. A day's file is published around 09:00 UTC
    the next day, so missing (404) days are skipped. Not cached.
    """
    today = pd.Timestamp.now(tz="UTC").normalize()
    day = pd.Timestamp(since) if since is not None else today - pd.Timedelta(days=35)
    day = (day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")).normalize()
    frames = []
    while day < today:
        d = day.strftime("%Y-%m-%d")
        url = (f"https://data.binance.vision/data/{market}/daily/klines/"
               f"{symbol}/{interval}/{symbol}-{interval}-{d}.zip")
        day += pd.Timedelta(days=1)
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code != 200:
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = z.read(z.namelist()[0]).decode()
        has_header = not raw[:1].isdigit()
        x = pd.read_csv(io.StringIO(raw), header=0 if has_header else None).iloc[:, :12]
        x.columns = KLINE_COLS
        frames.append(x)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    ot = pd.to_numeric(df["open_time"])
    ot = ot.where(ot < 1e14, ot // 1000)
    df.index = pd.to_datetime(ot, unit="ms", utc=True)
    df = df[["open", "high", "low", "close", "volume", "quote_volume", "trades",
             "taker_buy_base"]].astype(float)
    return df[~df.index.duplicated()].sort_index()


def binance_funding(symbol: str, start: str = "2020-01") -> pd.Series:
    """Perp funding rate history (fraction, paid by longs when positive), cached."""
    out = CACHE / f"binance_funding_{symbol}.parquet"
    if out.exists() and (pd.Timestamp.now(tz="UTC") - pd.read_parquet(out).index.max()).days < 40:
        return pd.read_parquet(out)["rate"]
    frames = []
    for m in _months(start):
        url = (f"https://data.binance.vision/data/futures/um/monthly/fundingRate/"
               f"{symbol}/{symbol}-fundingRate-{m}.zip")
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code != 200:
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            d = pd.read_csv(z.open(z.namelist()[0]))
        d.columns = [c.strip().lower() for c in d.columns]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    tcol = [c for c in df.columns if "time" in c][0]
    rcol = [c for c in df.columns if "rate" in c and "interval" not in c][0]
    s = pd.Series(df[rcol].astype(float).values,
                  index=pd.to_datetime(pd.to_numeric(df[tcol]), unit="ms", utc=True), name="rate")
    s.index = s.index.floor("h")
    s = s.groupby(level=0).sum().sort_index()
    s.to_frame().to_parquet(out)
    return s


def yahoo(ticker: str, interval: str = "1d", rng: str = "max") -> pd.DataFrame:
    """Yahoo chart API. interval 1d (rng=max) or 1h (rng<=730d)."""
    safe = ticker.replace("^", "IDX_").replace("=", "_")
    out = CACHE / f"yahoo_{safe}_{interval}.parquet"
    if out.exists() and (time.time() - out.stat().st_mtime) < 86400:
        return pd.read_parquet(out)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    # range=max silently downsamples daily bars to monthly; use explicit epoch window instead
    params = ({"interval": interval, "period1": 0, "period2": int(time.time())}
              if rng == "max" else {"interval": interval, "range": rng})
    for attempt in range(4):
        r = requests.get(url, params=params, headers=UA, timeout=60)
        if r.status_code == 200:
            break
        time.sleep(2 ** (attempt + 1))
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ["open", "high", "low", "close", "volume"]},
                      index=pd.to_datetime(res["timestamp"], unit="s", utc=True)).astype(float)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated()].sort_index()
    df.to_parquet(out)
    return df


def load(spec: str) -> pd.DataFrame:
    """Unified loader. spec examples:
    'binance:BTCUSDT:1h', 'binance:XAUUSDT:1h', 'spot:PAXGUSDT:1h', 'yahoo:GC=F:1d'.
    """
    src, sym, itv = spec.split(":")
    if src == "binance":
        return binance_klines(sym, itv)
    if src == "spot":
        return binance_klines(sym, itv, market="spot")
    if src == "yahoo":
        return yahoo(sym, itv, "730d" if itv.endswith("h") else "max")
    raise ValueError(spec)
