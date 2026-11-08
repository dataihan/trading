"""Offline check: live signal == backtest signal on the same bar (same code path)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from trading.data.loaders import CACHE  # noqa: E402
from trading.strategies.library import trend_ensemble  # noqa: E402


def test_live_signal_matches_backtest_at_rebalance_bars():
    f = CACHE / "binance_futures_um_BTCUSDT_1h.parquet"
    if not f.exists():
        import pytest
        pytest.skip("no cached data")
    df = pd.read_parquet(f).iloc[-24 * 400:]
    bt = trend_ensemble(df, long_only=True)["position"]          # rebalanced every 24 bars
    for i in range(24 * 200, len(df), 24 * 17):                    # sample rebalance bars
        live = trend_ensemble(df.iloc[: i + 1], long_only=True, rebalance=1)["position"].iloc[-1]
        assert abs(live - bt.iloc[i]) < 1e-12
