"""Paper trading for the funding-carry sleeve (round 5, part of frozen portfolio v2b).

Two tracks:
1. SHADOW (the evaluation): target weights from real Binance archive data; PnL recomputed by `report` from
   real daily closes + real funding (monthly archive, falling back to the premium-index estimate for the
   most recent days). Testnet funding is synthetic, so testnet PnL says nothing about a carry edge.
2. TESTNET (execution check only): orders on the Binance Futures testnet for CARRY_CAPITAL_USD gross.
   Coins traded by the v1 bot (BTC/ETH/SOL/BNB) are NOT ordered here, because one-way positions are shared
   per symbol and the v1 runner would fight them. They stay in the shadow book.

Rebalance: Mondays (decision on data up to Sunday's close), as fixed before paper trading started.
Universe: 27 coins of trading.data.universe (EOSUSDT excluded: not on testnet).

    python -m trading.live.carry run [--dry-run]
    python -m trading.live.carry report
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trading.data.loaders import binance_funding_estimate
from trading.data.universe import UNIVERSE, load_universe
from trading.strategies.carry import fcarry_daily

ROOT = Path(__file__).resolve().parents[3]
LIVE = ROOT / "data" / "live"
TARGETS = LIVE / "carry_targets.jsonl"
LEDGER = LIVE / "carry_ledger.jsonl"
COINS = [c for c in UNIVERSE if c != "EOSUSDT"]
V1_COINS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}
PERP_BPS = 6.0


def _log(path: Path, rec: dict):
    LIVE.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **rec}
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


def data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Daily perp closes (archive, ~1 day lag) and best-available daily funding per coin.
    Returns (close, funding, is_estimated_last_date per coin)."""
    from trading.data.loaders import binance_klines_daily
    u = load_universe(spot=False)
    close = u["close"][COINS]
    # top up closes with daily kline files (monthly archive ends at the last full month)
    extra = {}
    for c in COINS:
        last = close[c].dropna().index.max()
        d = binance_klines_daily(c, "1d", since=last + pd.Timedelta(days=1))
        if not d.empty:
            extra[c] = d.close
    if extra:
        close = close.combine_first(pd.DataFrame(extra)).sort_index()
    fund = u["funding"][COINS].reindex(close.index)
    est_from = {}
    for c in COINS:
        f_real = u["funding"][c]
        last_real = f_real[f_real != 0].index.max()
        est = binance_funding_estimate(c, last_real + pd.Timedelta(days=1))
        if len(est):
            fund.loc[est.index.intersection(fund.index), c] = est.reindex(fund.index).dropna()
        est_from[c] = str((last_real + pd.Timedelta(days=1)).date())
    return close, fund.fillna(0.0), pd.Series(est_from)


def last_target() -> dict | None:
    if not TARGETS.exists():
        return None
    rows = [json.loads(x) for x in TARGETS.read_text().splitlines() if x.strip()]
    return rows[-1] if rows else None


def run(dry: bool):
    close, fund, est_from = data()
    asof = close.dropna(how="all").index.max()
    w_all = fcarry_daily(close, fund)
    w = w_all.loc[asof]
    prev = last_target()
    today = pd.Timestamp.now(tz="UTC").normalize()
    rebalance = prev is None or (today.dayofweek == 0 and prev["asof"] < str(asof.date()))
    if rebalance:
        weights = {c: round(float(v), 6) for c, v in w.items() if v != 0}
        _log(TARGETS, {"asof": str(asof.date()), "weights": weights, "dry_run": dry,
                       "funding_estimated_from": est_from.to_dict()})
    else:
        weights = prev["weights"]
        print(f"[carry] no rebalance today (Mondays only); holding targets from {prev['asof']}")
    if dry:
        return
    from trading.live.clients import BinanceFutures, round_step
    cap = float(os.getenv("CARRY_CAPITAL_USD", "2000"))
    client = BinanceFutures()
    for c in COINS:
        if c in V1_COINS:
            continue
        tgt_usd = cap * weights.get(c, 0.0)
        rec = {"symbol": c, "target_w": weights.get(c, 0.0), "target_usd": round(tgt_usd, 2)}
        try:
            f = client.filters(c)
            mark = client.price(c)
            cur = client.position_qty(c)
            tq = round_step(tgt_usd / mark, f["step"])
            delta = round_step(tq - cur, f["step"])
            rec.update({"mark": mark, "current_qty": cur, "target_qty": tq, "delta_qty": delta})
            if abs(delta) * mark >= max(f["min_notional"], 5):
                client.set_leverage(c, 2)
                o = client.market_order(c, delta, reduce_only=abs(tq) < abs(cur) and tq * cur >= 0)
                rec.update({"order_id": o.get("orderId"), "status": o.get("status"), "avg_price": o.get("avgPrice")})
            elif delta != 0:
                rec["skipped"] = "below min notional"
        except Exception as e:
            rec["error"] = str(e)[:200]
        if rec.get("target_usd") or rec.get("current_qty") or rec.get("error"):
            _log(LEDGER, rec)


def carry_income(client=None) -> float:
    """Realised PnL + funding + commission booked on the carry sleeve's testnet symbols (USDT).
    The v1 equity snapshot subtracts this, so the shared testnet account still measures v1 alone."""
    from trading.live.clients import BinanceFutures
    client = client or BinanceFutures()
    mine = set(COINS) - V1_COINS
    total, start = 0.0, 0
    while True:
        rows = client._req("GET", "/fapi/v1/income", {"startTime": start, "limit": 1000}, signed=True)
        total += sum(float(r["income"]) for r in rows if r.get("symbol") in mine and r.get("asset") == "USDT")
        if len(rows) < 1000:
            return total
        start = int(rows[-1]["time"]) + 1


def report():
    """Shadow PnL of the logged targets, from real closes and best-available funding."""
    if not TARGETS.exists():
        print("no carry targets yet")
        return
    rows = [json.loads(x) for x in TARGETS.read_text().splitlines() if x.strip()]
    close, fund, est_from = data()
    start = pd.Timestamp(rows[0]["asof"], tz="UTC")
    w = pd.DataFrame({pd.Timestamp(r["asof"], tz="UTC"): r["weights"] for r in rows}).T.reindex(columns=COINS)
    w = w.fillna(0.0).reindex(close.index).ffill().fillna(0.0)
    w = w[w.index >= start]
    ret = close.pct_change().shift(-1).reindex(w.index)
    gross = (w * ret).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    fpay = (w * fund.shift(-1).reindex(w.index)).sum(axis=1)
    pnl = (gross - turn * PERP_BPS / 1e4 - fpay).shift(1).dropna()
    pnl = pnl[pnl.index <= close.dropna(how="all").index.max()]
    days = len(pnl)
    if days < 2:
        print(f"carry shadow: {days} day(s) of PnL so far; first targets as of {rows[0]['asof']}")
        return
    eq = (1 + pnl).cumprod()
    sh = pnl.mean() / pnl.std() * np.sqrt(365) if pnl.std() > 0 else float("nan")
    print(f"carry shadow ({days} days from {pnl.index[0].date()}): return={eq.iloc[-1] - 1:+.2%} "
          f"sharpe={sh:+.2f} maxDD={(eq / eq.cummax() - 1).min():.2%} "
          f"funding_part={-fpay.reindex(pnl.index).sum():+.2%} "
          f"(funding estimated from {min(est_from.values)} on; re-run later for real funding)")
    if days < 90:
        print("WAIT: < 90 days, too early to judge (backtest valid Sharpe 0.98, holdout 1.50)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from trading.live.runner import load_env
    load_env()
    run(a.dry_run) if a.cmd == "run" else report()
