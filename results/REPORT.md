# REPORT — autoresearch round 1 (2026-11-07)

## Result: portfolio_v1 is profitable on train, valid AND holdout (after fees + funding)

| split | period | return | CAGR | Sharpe | max DD |
|---|---|---|---|---|---|
| train | 2020-09 → 2023-12 | +83.7% | +20.3% | 1.90 | -10.5% |
| valid | 2024-01 → 2025-06 | +16.4% | +10.7% | 1.05 | -8.9% |
| **holdout** (opened once) | 2025-07 → 2026-10 | **+13.9%** | **+10.1%** | **1.05** | **-7.1%** |
| holdout @ 2× costs | same | +12.2% | +8.9% | 0.93 | -7.5% |

Comparison: in the holdout, BTC buy & hold returned -32% (DD -54%) and ETH -13% (DD -70%).
Calendar years: 2020 (from Sep) +17.3%, 2021 +25.5%, 2022 -1.7%, 2023 +26.9%, 2024 +14.5%, 2025 +7.9%, 2026 YTD +7.4%.

![equity](portfolio_v1_equity.png)

## What is in it (`strategies/portfolio_v1.json`)
| sleeve | weight | rule | evidence |
|---|---|---|---|
| BTC/ETH/SOL/BNB trend | 15/15/10/10% | Binance USDT-M perps, long/flat. Average of sign(return) over 7/14/30/60/90/180 days, vol-targeted to 40%/yr, rebalanced daily | Time-series momentum (Moskowitz–Ooi–Pedersen 2012; crypto TSMOM literature) |
| QQQ IBS | 25% | Daily candle: if (close−low)/(high−low) < 0.2, hold QQQ the next day | Internal Bar Strength mean reversion (Pagonidis 2013, Quantpedia) |
| QQQ turn-of-month | 25% | Hold the last 2 trading days + first 3 trading days of each month | Ariel 1987, Lakonishok & Smidt 1988 |

All 6 sleeves were positive in the holdout.

## What failed (logged in `experiments.jsonl`, ~100 configs)
- ICT-style rules on crypto 1h: previous-day high/low sweep, Asia-range breakout/fade. Mostly negative after 6 bps fees.
- BTC 22–23h UTC seasonality: strongly negative after fees.
- Single-lookback TSMOM and Donchian: positive in train, broke in valid (overfit to one parameter).
- LightGBM walk-forward direction models: around Sharpe 0 to 0.7, unstable across coins. Not kept.
- Gold/silver/oil trend: weak in valid (2015–2021).

## Honest caveats
- Returns are modest by design (crypto sleeves use small weights, about 50% gross exposure). This is
  a low-drawdown edge, not a get-rich engine. Leverage scales both return and DD.
- About 100 configs were tried. The holdout was opened once for the strategy portfolio (plus 3 buy & hold baselines).
  Some selection bias remains (e.g. IBS without trend filter was chosen after seeing valid).
- QQQ fills assume trades at the close. Real execution needs MOC orders. Crypto assumes taker fills at the hourly close.
- Funding is included for perps. Borrow, tax and FX are not.
- Past performance ≠ future results. Paper-trade before any real money.

## Next rounds (for the loop)
1. Add gold (PAXG/XAUUSDT) and Nasdaq trend sleeves; test inverse-vol weights.
2. Use ML as a filter on top of the trend sleeve (not as a standalone predictor).
3. Paper-trading bot on Binance testnet for the crypto sleeve.

---

# Round 2 (2026-11-07): ML as a filter + regime + portfolio construction

**Verdict: nothing beat portfolio_v1 on train AND valid, so v1 stays the champion. No holdout look was spent on v2.**

| idea | where | result (valid Sharpe vs base) | verdict |
|---|---|---|---|
| Meta-labeling (LightGBM, walk-forward, purged) on crypto trend | BTC/ETH/SOL/BNB | BTC 1.35 vs 0.97, SOL 0.91 vs 0.45, **ETH -0.36 vs 0.79, BNB 0.03 vs 1.06** | REJECT (unstable across coins) |
| Meta-labeling, "scale" sizing | same | mixed, worse on ETH/BNB | REJECT |
| Meta-labeling on QQQ IBS / TOM | QQQ | IBS 1.01–1.05 vs 1.09, TOM 0.19–0.45 vs 0.52 | REJECT (no gain) |
| HMM regime filter (3 states, forward-filtered, walk-forward refit) | crypto | BTC/ETH ↑, SOL/BNB ↓ | REJECT |
| HMM regime filter (2 states) | crypto | ETH 1.58 ↑, others flat or ↓ | REJECT |
| HMM regime filter | QQQ IBS / TOM | 0.30 vs 1.09, -0.29 vs 0.52 | REJECT |
| Gold trend (PAXG 1h, GC=F daily) | gold | train Sharpe < 0 / 0.39 | REJECT |
| QQQ trend ensemble | QQQ | train 0.31 | REJECT |
| Inverse-vol sleeve weights (v2a) | portfolio | valid 0.63 vs 1.05 | REJECT |
| Inverse-vol + 10% vol target (v2b) | portfolio | valid 0.49 vs 1.05 | REJECT |

**Fees matter less for v1 than for intraday rules:** v1 with 0 fees: train Sharpe 2.01 / valid 1.17
(vs 1.90 / 1.05 with fees). v1 trades rarely, so fees cost only about 1–2%/yr. The ICT intraday rules lost money even before fees.

**Lesson (same as the ML literature):** on daily/hourly OHLCV, gradient boosting and HMMs add little on
top of simple, well-documented effects. They fit the training period but do not generalise consistently.
Better ML inputs would need new data: order book, liquidations, open interest, on-chain, news sentiment.

Total configs logged: see `experiments.jsonl`. Holdout looks: unchanged (v1 only).
