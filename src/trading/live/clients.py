"""Minimal REST clients for paper trading. No third-party SDK needed.

Binance USDT-M Futures — default base URL is the TESTNET. Live trading is refused unless
ALLOW_LIVE=yes is set AND the base URL is changed by hand (two deliberate steps).
Alpaca — default base URL is the PAPER endpoint.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import os
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

TESTNET_FAPI = "https://testnet.binancefuture.com"
LIVE_FAPI = "https://fapi.binance.com"
ALPACA_PAPER = "https://paper-api.alpaca.markets"
ALPACA_DATA = "https://data.alpaca.markets"


def sign(secret: str, query: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def sign_ed25519(private_key_pem: bytes, query: str) -> str:
    """Binance 'self-generated' Ed25519 keys: base64 signature, URL-encoded."""
    import base64
    from urllib.parse import quote

    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    key = load_pem_private_key(private_key_pem, password=None)
    return quote(base64.b64encode(key.sign(query.encode())).decode(), safe="")


def round_step(qty: float, step: float) -> float:
    """Round quantity DOWN to the exchange step size."""
    if step <= 0:
        return qty
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(math.floor(abs(qty) / step + 1e-9) * step, decimals) * (1 if qty >= 0 else -1)


class BinanceFutures:
    def __init__(self, key: str | None = None, secret: str | None = None, base: str | None = None):
        self.key = key or os.getenv("BINANCE_API_KEY", "")
        self.secret = secret or os.getenv("BINANCE_API_SECRET", "")
        # Ed25519 alternative: path to the private key PEM (Binance "Self-generated" key type)
        pk = os.getenv("BINANCE_PRIVATE_KEY_PATH")
        self.ed25519_pem = Path(pk).expanduser().read_bytes() if pk else None
        self.base = (base or os.getenv("BINANCE_FAPI_BASE", TESTNET_FAPI)).rstrip("/")
        if self.base == LIVE_FAPI and os.getenv("ALLOW_LIVE") != "yes":
            raise RuntimeError("Refusing to trade live: set ALLOW_LIVE=yes explicitly (Han's approval needed).")
        self.s = requests.Session()
        self.s.headers["X-MBX-APIKEY"] = self.key
        self._filters: dict = {}

    def _req(self, method: str, path: str, params: dict | None = None, signed: bool = False):
        params = dict(params or {})
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            q = urlencode(params)
            q += "&signature=" + (sign_ed25519(self.ed25519_pem, q) if self.ed25519_pem
                                  else sign(self.secret, q))
        else:
            q = urlencode(params)
        r = self.s.request(method, f"{self.base}{path}?{q}", timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"Binance {method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    # --- public
    def klines(self, symbol: str, interval: str = "1h", limit: int = 1500, end_time: int | None = None):
        p = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time:
            p["endTime"] = end_time
        return self._req("GET", "/fapi/v1/klines", p)

    def price(self, symbol: str) -> float:
        return float(self._req("GET", "/fapi/v1/ticker/price", {"symbol": symbol})["price"])

    def filters(self, symbol: str) -> dict:
        if not self._filters:
            info = self._req("GET", "/fapi/v1/exchangeInfo")
            for s in info["symbols"]:
                f = {x["filterType"]: x for x in s["filters"]}
                self._filters[s["symbol"]] = {
                    "step": float(f["MARKET_LOT_SIZE"]["stepSize"] if "MARKET_LOT_SIZE" in f else f["LOT_SIZE"]["stepSize"]),
                    "min_notional": float(f.get("MIN_NOTIONAL", {}).get("notional", 5)),
                }
        return self._filters[symbol]

    # --- private
    def position_qty(self, symbol: str) -> float:
        for p in self._req("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, signed=True):
            if p["symbol"] == symbol:
                return float(p["positionAmt"])
        return 0.0

    def balance_usdt(self) -> float:
        for b in self._req("GET", "/fapi/v2/balance", signed=True):
            if b["asset"] == "USDT":
                return float(b["balance"])
        return 0.0

    def set_leverage(self, symbol: str, leverage: int = 2):
        return self._req("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    def market_order(self, symbol: str, qty: float, reduce_only: bool = False):
        side = "BUY" if qty > 0 else "SELL"
        p = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": abs(qty)}
        if reduce_only:
            p["reduceOnly"] = "true"
        return self._req("POST", "/fapi/v1/order", p, signed=True)


class AlpacaPaper:
    def __init__(self, key: str | None = None, secret: str | None = None, base: str | None = None):
        self.base = (base or os.getenv("ALPACA_BASE", ALPACA_PAPER)).rstrip("/")
        if "paper" not in self.base and os.getenv("ALLOW_LIVE") != "yes":
            raise RuntimeError("Refusing non-paper Alpaca endpoint without ALLOW_LIVE=yes.")
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key or os.getenv("ALPACA_KEY", ""),
                               "APCA-API-SECRET-KEY": secret or os.getenv("ALPACA_SECRET", "")})

    def _get(self, url, params=None):
        r = self.s.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def calendar(self, start: str, end: str):
        return self._get(f"{self.base}/v2/calendar", {"start": start, "end": end})

    def clock(self):
        return self._get(f"{self.base}/v2/clock")

    def today_bar(self, symbol: str):
        """Today's partial daily bar (IEX feed, free)."""
        d = self._get(f"{ALPACA_DATA}/v2/stocks/{symbol}/snapshot", {"feed": "iex"})
        return d["dailyBar"], float(d["latestTrade"]["p"])

    def position_qty(self, symbol: str) -> float:
        r = self.s.get(f"{self.base}/v2/positions/{symbol}", timeout=30)
        return float(r.json()["qty"]) if r.status_code == 200 else 0.0

    def equity(self) -> float:
        return float(self._get(f"{self.base}/v2/account")["equity"])

    def order_on_close(self, symbol: str, qty: float):
        """Market-on-close order (must be sent before ~15:50 ET). Whole shares only:
        Alpaca does not accept fractional quantities for MOC orders."""
        body = {"symbol": symbol, "qty": str(int(abs(qty))), "side": "buy" if qty > 0 else "sell",
                "type": "market", "time_in_force": "cls"}
        r = self.s.post(f"{self.base}/v2/orders", json=body, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca order -> {r.status_code}: {r.text[:300]}")
        return r.json()
