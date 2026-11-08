import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from trading.live.clients import BinanceFutures, round_step, sign  # noqa: E402


def test_binance_signature_matches_official_example():
    # example from Binance API docs (SIGNED endpoint, HMAC SHA256)
    secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
    q = "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559"
    assert sign(secret, q) == "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"


def test_round_step():
    assert round_step(0.123456, 0.001) == 0.123
    assert round_step(-0.123456, 0.001) == -0.123
    assert round_step(12.7, 1) == 12
    assert round_step(0.0009, 0.001) == 0


def test_live_endpoint_refused_without_flag(monkeypatch):
    monkeypatch.delenv("ALLOW_LIVE", raising=False)
    with pytest.raises(RuntimeError):
        BinanceFutures(key="k", secret="s", base="https://fapi.binance.com")


def test_default_is_testnet(monkeypatch):
    monkeypatch.delenv("BINANCE_FAPI_BASE", raising=False)
    assert "testnet" in BinanceFutures(key="k", secret="s").base


def test_ed25519_signature_verifies():
    import base64
    from urllib.parse import unquote

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from trading.live.clients import sign_ed25519
    k = Ed25519PrivateKey.generate()
    pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                          serialization.NoEncryption())
    q = "symbol=BTCUSDT&side=BUY&type=MARKET&quantity=0.001&timestamp=1"
    sig = base64.b64decode(unquote(sign_ed25519(pem, q)))
    k.public_key().verify(sig, q.encode())  # raises if invalid
