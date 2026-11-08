# Trading Hypotheses — Evidence-Ranked Backlog

Research date: 2026-11-07. Author: research agent. Scope: data in `data/` (Binance USDT-M perps 1h, PAXGUSDT 1h, Yahoo daily/1h).

## Conventions (apply to every hypothesis)

- **Bar timestamps.** Binance klines are stamped with the bar's **open** time in UTC. A signal computed from bar `t` (known at its close) trades at the **open of bar t+1**. Never use bar `t`'s close to fill a signal that bar `t` produced.
- **Yahoo 1h bars** are in exchange-local time (America/New_York for QQQ, CL=F, GC=F, SI=F). Convert to UTC with a timezone-aware function, because US session hours in UTC move with DST (cash open is 13:30 UTC in summer and 14:30 UTC in winter). The first QQQ 1h bar runs 09:30–10:30 ET and the last runs 15:30–16:00 ET.
- **Costs (baseline).** Binance futures taker 0.05%/side, so a round trip (RT) costs **10 bp**. Add slippage of 1 bp/side for BTC/ETH and 2–3 bp/side for SOL/BNB. US ETF: 1–2 bp/side all-in. Futures on Yahoo continuous series: roughly 1 bp/side, but **roll gaps in GC=F/SI=F/CL=F continuous series are not returns** and must be removed or the series back-adjusted.
- **[Update] Funding is now loaded** from data.binance.vision (`binance_funding()`) and charged in every Binance backtest. Original note: Longs on USDT-M perps historically paid ~0.01%/8h on average, roughly 10%/yr in bull regimes, and funding can be much higher. Any multi-day crypto hypothesis must be re-checked with funding history (Binance `/fapi/v1/fundingRate`, which is public). Intraday flat-by-funding-time strategies are mostly unaffected; funding is charged at 00:00, 08:00 and 16:00 UTC.
- **Evidence grades.** **A** = peer-reviewed and replicated across markets. **B** = peer-reviewed or reputable working paper, limited replication. **C** = working paper or blog only, or the effect is known to decay. **D** = folklore or vendor claims, with no credible statistical evidence.

---

## 1. Top 12 hypotheses (ranked by evidence × data fit × cost robustness)

### H1. Volatility-scaled time-series momentum (TSMOM) on daily commodities and Nasdaq — Grade A (decayed post-2009)
- **Market/TF:** Daily GC=F, SI=F, CL=F, ^NDX (QQQ for the tradable version), 2000–present (NDX from 1985).
- **Rule:**
  - Compute at each close `t`: `sig = sign(P_t / P_{t-252} - 1)`. Also test lookbacks of 21, 63 and 126 days, plus an equal-weight blend of 21/63/252, *chosen in train only*.
  - `sigma_t` = EWMA std of daily returns with center-of-mass 60 days, annualized (×√252).
  - `w_t = sig × (0.40 / sigma_t) / N_assets`. MOP use 40% per asset; to target ~10% portfolio vol, cap |w| ≤ 2.
  - Rebalance monthly on the last trading day, filled at the next open. Daily rebalancing with a 10% no-trade buffer is a variant.
- **Expected sign:** Positive. Strongest in crises (crisis alpha).
- **Evidence:** Moskowitz, Ooi & Pedersen (2012, JFE). Hurst, Ooi & Pedersen, "A Century of Evidence on Trend-Following". Kim, Tse & Wald (2016) show much of the alpha comes from **vol scaling**, not the sign: alpha falls from 1.27% to 0.41%/mo without scaling. Replications report weaker results after 2009, especially in commodities.
- **Required control:** Run the same vol-scaled weights with `sig = +1` always (a long-only vol-scaled benchmark) to separate the trend effect from the vol-scaling effect.
- **Costs:** Very low sensitivity (~12 rebalances/yr). The main risk is roll handling in continuous futures.
- **Caveat:** Four assets make a thin cross-section, so expect a Sharpe of 0.3–0.6 at best.

### H2. Daily/weekly TSMOM on crypto perps — Grade B
- **Market/TF:** BTC, ETH, SOL, BNB perps. Aggregate 1h bars to daily bars at 00:00 UTC.
- **Rule:**
  - At 00:00 UTC on day `t`: `sig = sign(close_{t-1} / close_{t-1-L} - 1)`, with L ∈ {7, 14, 28} days chosen in train.
  - Weight `w = sig × min(2, target_vol / sigma_30d)`, with target_vol = 50% annualized per coin.
  - Hold 7 days and rebalance weekly. Variant: daily rebalancing with a 25% buffer.
  - Test long/flat as well as long/short. Short perps *receive* funding in bull regimes, which matters here.
- **Expected sign:** Positive.
- **Evidence:** Liu & Tsyvinski (2021, RFS), "Risks and Returns of Cryptocurrency": significant TSMOM at daily and weekly horizons for BTC, ETH and XRP. Padyšák & Vojtko (SSRN 4081000) report trend-following (MAX/MIN breakout) working on BTC.
- **Costs:** Low to moderate. About 52 RT/yr × 10 bp ≈ 5%/yr per coin at full turnover. Buffering helps. **Funding is the larger unknown**, so fetch it.
- **Caveat:** The sample is dominated by 2020–21 and 2023–24 bull runs. Report results by regime and with funding included.

### H3. Overnight vs intraday decomposition, QQQ/NDX ("night effect") — Grade A for the pattern, C for net tradability
- **Market/TF:** Daily QQQ (1999–present). ^NDX daily for long history, but check whether early-year `Open` values are just the prior close (stale-open problem in index data).
- **Rule:**
  - Overnight leg: buy at close `t-1`, sell at open `t`, so `r_on = O_t / C_{t-1} - 1`.
  - Intraday leg: `r_id = C_t / O_t - 1`.
  - Test 3 variants: (a) long overnight only; (b) long overnight, short intraday; (c) long overnight only when the 200-day trend filter is up.
- **Expected sign:** `r_on` > 0 and `r_id` ≈ 0 or < 0 on average.
- **Evidence:** Cliff, Cooper & Gulen (2008), "Return Differences between Trading and Non-trading Hours: Like Night and Day" (effect holds across indexes and futures, NYSE and Nasdaq). Berkman et al. (2012). Lou, Polk & Skouras (2019, JFE), "A Tug of War". Practitioner replications: RobotWealth (SPY, effect still present) and R-bloggers (QQQ 2000–2013: summed overnight ≈ +93%, intraday ≈ -84%).
- **Costs:** **Highly cost-sensitive.** 2 trades/day ≈ 252 RT/yr. At 2 bp RT that is about 5%/yr, and at 4 bp about 10%/yr, against a gross overnight drift of roughly 8–12%/yr. Opening-auction slippage is real. A useful implementation is as a **timing overlay** for an existing long (e.g. hold QQQ overnight in leverage and de-risk intraday) rather than as a standalone strategy.
- **Caveat:** The effect is well known, which raises the risk of decay. Report results by decade.

### H4. Turn-of-the-month (TOM), Nasdaq-100 — Grade B (equities only)
- **Market/TF:** Daily ^NDX (1985+) and QQQ (1999+).
- **Rule:** Long from the close of trading day **-2** of month m (the second-to-last trading day) to the close of trading day **+3** of month m+1. Flat otherwise. Also test the narrower window -1 → +3 and the wider window -4 → +3. Recent evidence suggests the effect has **moved earlier**. Pre-register the three windows only.
- **Expected sign:** Positive excess return per day in the window versus non-TOM days.
- **Evidence:** Lakonishok & Smidt (1988) and McConnell & Xu (2008). Maberly & Waggoner (Atlanta Fed WP 2000-11) found the effect disappeared in S&P futures after 1990. Quantseeker ("Turn-of-the-Month Strategies: Do They Still Work?") finds it significant in SPY and QQQ with wider windows, but **no effect outside equities**. Do not expect it in gold, oil or crypto. A crypto TOM test is a placebo check only.
- **Costs:** 12 RT/yr, so negligible.
- **Caveat:** The main benefit is lower exposure (~25% time in market) at similar return. Judge it on return per unit of exposure and t-stat, not on CAGR.

### H5. Crypto hour-of-day seasonality (BTC 21:00/22:00–00:00 UTC) — Grade C
- **Market/TF:** BTC, ETH (SOL, BNB as out-of-sample assets) perps, 1h.
- **Rule:**
  - (a) Descriptive first: mean and t-stat (Newey-West) of the return for each of the 24 UTC hours, for each symbol and each year.
  - (b) Strategy: go long at the open of the 22:00 UTC bar and exit at the open of the 00:00 UTC bar (hold 2 bars). Variant: 21:00–23:00.
  - (c) Weekend/day-of-week: the mean return of each weekday (UTC) is purely descriptive.
- **Expected sign:** Positive for 21–23 UTC per the cited paper, and negative around 03–04 UTC per a Quantpedia summary.
- **Evidence:** Padyšák & Vojtko, "Seasonality, Trend-following, and Mean reversion in Bitcoin" (SSRN 4081000; Quantpedia: ~33% p.a., Sharpe 1.58 on Gemini 2015–21, **costs not reported**). Baur, Cahill, Godfrey & Liu (2019, FRL), using 15M observations, find time-specific anomalies but **no persistent effects across time**. Aalipour et al. question whether crypto day-of-week effects are real.
- **Costs:** **Fatal as a standalone taker strategy.** 365 RT/yr × 10 bp ≈ 36.5%/yr, which is about equal to the claimed gross edge. It is viable only (i) with maker fills, (ii) as an overlay that shifts the timing of entries and exits for other strategies, or (iii) if the in-sample gross mean exceeds ~15 bp per trade.
- **Multiple testing:** 24 hours × 4 coins = 96 tests. Require Bonferroni/Holm or deflated-Sharpe treatment.

### H6. Intraday momentum, "first period predicts last period" — Grade B (equities), C (crypto)
- **(a) QQQ 1h (2023-12+, short sample):**
  - `r1` = previous close → close of the first 1h bar (10:30 ET).
  - At the open of the last full bar (15:00 ET, or 15:30 if a half-bar exists), go long if `r1 > 0` and short if `r1 < 0`. Exit at the 16:00 close.
  - Gao, Han, Li & Zhou (2018, JFE) use 30-minute windows. The 1h adaptation is untested.
- **(b) Crypto 1h, UTC day:**
  - `r1` = return of the 00:00 bar (or the 22:00 previous close → 01:00).
  - `r_mid` = return from 01:00 to 23:00.
  - At the 23:00 bar open, take `sign(r1)` (paper) or `sign(r_mid)` (variant). Exit at 00:00.
  - Also test US-session anchoring: the first hour after the US cash open predicts the last hour before 20:00/21:00 UTC.
- **Expected sign:** Positive (continuation).
- **Evidence:** Gao et al. (2018, JFE): S&P ETF 1993–2013, stronger on high-vol, high-volume and news days. CXO replication: R² only 1–2%, gross ~6.3% p.a. QuantConnect: weaker in recent periods. Shen, Urquhart & Wang (2022), "Bitcoin intraday time-series momentum" (driven by liquidity provision).
- **Costs:** One RT per day. Crypto: 10 bp vs a typical |1h return| of ~50 bp, so the hit rate must be ≳ 55%. QQQ: 2–4 bp RT, which is fine. Condition on high-vol days (|r1| > 1σ) to raise gross edge per trade.

### H7. Noise-area intraday breakout ("Beat the Market"), adapted to 1h — Grade C
- **Market/TF:** QQQ 1h (2023-12+) and crypto 1h. For crypto, anchor the "session open" at 00:00 UTC, and test 13:30/14:30 UTC (US open) as an alternative anchor.
- **Rule (Zarattini, Aziz & Barbon 2024):**
  - For each hour-of-session `h`: `move_h` = mean over the last 14 sessions of `|P(open + h) / P(open) - 1|`.
  - Upper band = `Open_t × (1 + move_h)` and lower band = `Open_t × (1 - move_h)`. Use `max(Open_t, C_{t-1})` and `min(...)` for gap handling, per the paper.
  - At each bar close, enter long if `close > upper` and short if `close < lower`.
  - Trailing exit when price crosses back inside the band or crosses VWAP (computed from the bar's quote_volume / volume). Flat at session end.
  - Size = target_daily_vol / sigma_14d.
- **Expected sign:** Positive (trend days persist).
- **Evidence:** SFI Research Paper N°24-97: SPY 2007–2024, 19.6% p.a. **net**, Sharpe 1.33 (as summarized by CXO; verify against the paper). Not peer-reviewed. Crypto has no published test, so this is a transfer hypothesis.
- **Costs:** Moderate. Roughly 0.5–1 RT/day in crypto ≈ 18–36%/yr at 10 bp, so it needs a large gross edge. QQQ is fine cost-wise but has only about 3 years of 1h history, giving low power.

### H8. Taker-buy order-flow imbalance predicts 4–12h returns (crypto) — Grade B-/C (new evidence, sign uncertain)
- **Market/TF:** BTC, ETH, SOL, BNB perps, 1h.
- **Features:**
  - `ofi_k = Σ_{last k bars}(2·taker_buy_base − volume) / Σ_{last k bars} volume`, for k ∈ {1, 4, 12, 24}.
  - z-score each over a trailing 30-day window.
  - Also `trades` per unit of volume (average trade size), as a proxy for retail vs algorithmic flow.
- **Rule:** At each 4h boundary (00, 04, …, 20 UTC), if `z(ofi_k) > +1.5` go long, and if `< -1.5` go short. Hold H ∈ {4, 8, 12} hours. **Pre-register the sign as continuation** (the order-imbalance literature). Because the short-horizon reversal paper (H9) finds reversals after taker-driven moves at 15 minutes, treat the opposite sign as a *separate* pre-declared test and count both in the multiple-testing tally.
- **Evidence:** "The Quarter-Hour Effect: Periodic Algorithmic Trading and Return Predictability in Cryptocurrency Futures" (arXiv 2607.09426, 2026, Binance perps): order imbalance forecasts 4–12h returns, weaker at shorter horizons. Order-flow imbalance nowcasts BTC crashes (Springer, Rev. Quant. Fin. Acc. 2023). GBM on 1-second Binance perp data shows a monotone OFI effect with concavity at the extremes (arXiv 2602.00776).
- **Costs:** At most 6 RT/day if always on. Must gate to extreme z-scores; aim for ≥ 20 bp gross per trade.
- **Caveat:** Our 1h-aggregated taker volume is much coarser than the tick data in these papers.

### H9. Short-term reversal after extreme hourly moves (crypto) — Grade B for existence, C for net profitability
- **Market/TF:** BTC, ETH, SOL, BNB perps, 1h.
- **Rule:**
  - `sigma_h` = std of 1h returns over the trailing 720 bars.
  - If `r_t < -k·sigma_h` (k ∈ {3, 4}), go long at the next open. If `r_t > +k·sigma_h`, go short (test separately: the evidence is mainly for *negative* shocks).
  - Hold H ∈ {6, 12, 24} bars, or exit early when price recovers 50% of the shock.
  - No overlapping trades per symbol.
- **Expected sign:** Positive for fading negative shocks; positive shocks mixed.
- **Evidence:** De Nicola, "On the Intraday Behavior of Bitcoin" (Ledger): negative autocorrelation at 1, 2 and 4h, stronger for large moves. Bitcoin overreaction literature (J. Applied Econ. 2022): overreaction after negative hourly shocks builds over 6–24h. **Counter-evidence:** "Short-horizon mean reversion in cryptocurrency markets" (arXiv 2608.21888, 2026, 183 Binance pairs at 15 minutes) finds the reversal is real but gross ≈ 1.3 bp/trade, **below a 5 bp RT cost**.
- **Costs:** Critical. Only large-threshold, longer-hold variants have a chance against 10–14 bp RT. Expect few trades (likely < 100 per coin per k), so pool across coins.

### H10. Previous-day high/low break: continuation (Osler) vs "liquidity sweep" reversal (ICT) — Grade B for continuation in FX, D for ICT reversal
- **Market/TF:** BTC, ETH perps and PAXGUSDT, 1h. PDH/PDL = high/low of the previous UTC day. Test also the previous New York 17:00 ET day and the "Asia range" (00:00–08:00 UTC) high/low.
- **Rule A (continuation, evidence-backed):** First 1h **close** above PDH in day `t` → long at the next open. Stop at PDH − 0.5·ATR(24) or exit after H ∈ {4, 8, 24} bars or at 23:00 UTC. Mirror the rule for PDL.
- **Rule B (sweep reversal, folklore):** A 1h bar trades **above PDH (high > PDH) but closes back below PDH** → short at the next open. Stop at that bar's high + 0.1·ATR, target PDL or the midpoint, time exit at 12 bars. Optional "killzone" filter: only signals between 07:00–10:00 UTC (London) or 12:00–15:00 UTC (NY). Mirror the rule for PDL.
- **Expected sign:** A > 0 (stop-loss cascades make trends unusually rapid after clustered levels are breached; effect lasts hours, not days). B: no credible prior. Blog tests found sweeps lean toward continuation.
- **Evidence:** Osler, "Stop-Loss Orders and Price Cascades in Currency Markets" (NY Fed Staff Report 150): stop clusters at salient levels; rapid trends after they are hit; significant over hours, not days. For ICT/SMC there is **no peer-reviewed evidence**: Thor Trade Copier found no journal article on Crossref, and the only positive claims are vendor backtests (e.g. Backtrex PF 1.68, unverifiable).
- **Costs:** ≤ 1–2 trades/day/symbol, moderate. The sweep variant uses tight stops, so slippage on stop fills matters. Model a stop fill at stop price + 1 bp, or at the next bar open if price gapped through.
- **Fair value gaps:** One independent blog found the apparent edge was a close-based stop/target artifact. After the fix, the best variant had PF 1.11, which is not tradable. The killzone filter is a parameter, not a hypothesis. Count each filter in the multiple-testing tally.

### H11. Gold time-of-day: Asia-up / West-down drift — Grade C/D
- **Market/TF:** PAXGUSDT 1h (2020-08+). Confirmation on GC=F 1h (2024-06+) and XAUUSDT perp (2025-12+, too short except as a sanity check).
- **Rule:**
  - (a) Descriptive hour-of-day mean returns, by year.
  - (b) Long from 23:00 UTC to 07:00 UTC (Asian session), flat otherwise.
  - (c) Short (or flat) across the London fix windows: 10:00–11:00 UTC and 14:00–16:00 UTC (adjust for UK DST).
- **Expected sign:** (b) > 0 and (c) < 0, *if* the documented pattern persists.
- **Evidence:** Master's theses only. CBS "Gold Price Dynamics Around the Clock" finds a hat-shaped intraday pattern with gold rising in Eastern hours and falling around the London fix in alleged-manipulation years. A UZH thesis finds the most negative returns around the AM and PM fixes. A 2026 commentary notes gold up in Asian hours and down in Western hours year-to-date. Other work (Banks & Bank Systems 2016) found **London** had the highest mean, which contradicts the others.
- **Costs:** PAXG spot taker is 0.1%/side and the book is thin, so it is not tradable standalone. Treat it as a regime/timing feature or test it on the XAUUSDT perp later.

### H12. Gradient-boosted classifier on lagged features with purged walk-forward (crypto 1h) — Grade C (method, not an anomaly)
- **Market/TF:** Pooled BTC, ETH, SOL, BNB perps, 1h. Decision every 4h.
- **Features (all computed at bar close t):**
  - Returns over 1, 4, 12, 24, 72 and 168 bars.
  - Realized vol over 24 and 168 bars, and their ratio.
  - OFI z-scores (H8).
  - log(volume) and trades z-scores.
  - Hour-of-day and day-of-week (one-hot).
  - Distance to PDH/PDL and to the Asia-range high/low, in ATR units.
  - Distance to rolling 20/100-day highs and lows.
  - BTC features as cross-asset inputs for the alts.
- **Target:** `sign(r_{t+1 → t+1+H})` with H = 8 bars. Or a 3-class label with a "no-trade" band of |r| < 2×cost. The triple-barrier label is an alternative.
- **Model:** LightGBM/XGBoost with shallow trees (depth ≤ 4), strong regularization, ≤ 300 trees, early stopping on an inner validation fold.
- **Validation:** Expanding walk-forward retrained every 3 months. **Purge** H bars plus the feature lookback at each boundary, and **embargo** 1 day (López de Prado, *Advances in Financial Machine Learning*, ch. 7).
- **Trading rule:** Trade only when predicted P(up) > 0.55 or < 0.45, with the threshold fixed in train.
- **Expected sign:** Small positive information coefficient (IC) at best. Use as a **combiner** of H5, H8 and H9 features, not as a separate source of alpha.
- **Pitfalls:**
  - Lookahead through rolling z-scores or normalization fitted on the full sample.
  - Using the bar-t close to fill.
  - Leakage through overlapping labels.
  - Hyperparameter search over the holdout.
  - Survivorship bias: only 4 coins, all survivors.
  - Regime dependence: the 2021 bull run dominates.
- **Benchmark:** Must beat a logistic regression on the same features and a simple H2 TSMOM rule. Otherwise reject.

### Deprioritized / not in top 12
- **5-minute opening range breakout on QQQ** (Zarattini & Aziz 2023, SSRN 4416622: 33% CAGR, zero slippage). An independent 2026 replication on NQ and four other indices found **gross reproduced, net ≈ 0** (+0.131R gross, +0.002R net). Our QQQ data is 1h only, so a true 5-minute ORB cannot be tested. A 1h "first-bar breakout" is a much coarser and different hypothesis. Low priority.
- **Fair value gaps, Asia-range sweeps, killzones as standalone strategies:** Grade D. Use them only as features inside H10/H12.
- **CME weekend-gap fill in BTC:** trader folklore. CME moved to 24/7 crypto trading in 2026 per press, so the effect ended structurally anyway.
- **Crypto turn-of-the-month:** no evidence (Quantseeker finds no TOM effect outside equities). Placebo only.

---

## 2. Validation protocol

1. **Splits (chronological, no shuffling).**
   - *Crypto 1h (2020-01 → 2026-10):* Train/design 2020-01 → 2023-12. Validation (walk-forward, parameter selection) 2024-01 → 2025-06. **Holdout 2025-07 → 2026-10, touched once per hypothesis**, after the rule is frozen and committed to git.
   - *Daily commodities/NDX:* Train 2000 (1985 for NDX) → 2014. Validation 2015 → 2020. Holdout 2021 → 2026-10.
   - *QQQ/GC/SI/CL 1h (≈2.5 years):* too short for a split. Use the data only to confirm hypotheses already validated on daily data or on crypto, and say so explicitly.
2. **Walk-forward.** Use an expanding or rolling window, refit every 3 months (crypto) or 12 months (daily). Purge plus embargo whenever labels overlap.
3. **Minimum sample.**
   - At least 100 independent trades in train and 30 in holdout, pooled across symbols if needed.
   - For daily/monthly-rebalanced strategies, at least 10 years in train.
   - Report Newey-West t-stats, because returns from overlapping holds are autocorrelated.
4. **Multiple testing.**
   - Keep a **trial ledger** (every variant run, including failures). The ledger count is N for the **Deflated Sharpe Ratio** (Bailey & López de Prado 2014). Accept only if DSR > 0.95 on validation.
   - Require t > 3 (Harvey, Liu & Zhu 2016) for anything found by scanning hours, days or thresholds.
   - When many variants are tried, report the **Probability of Backtest Overfitting** via CSCV (Bailey, Borwein, López de Prado & Zhu 2015), for example with the R `pbo` package or a Python port.
5. **Costs and robustness gates (all must pass).**
   - (a) Net of baseline cost; still positive at **2× baseline cost**.
   - (b) Positive in at least 2 of 3 sub-periods.
   - (c) Same sign on at least 3 of 4 crypto symbols, or on at least 2 of 3 commodities.
   - (d) Parameter neighbourhood stable: ±25% in lookback or threshold keeps at least half of the Sharpe.
   - (e) Beats the relevant naive benchmark (buy-and-hold vol-scaled; random-entry with the same holding period and exposure).
6. **Implementation hygiene.**
   - Signals use data available at the bar close only; fills at the next bar open.
   - Use UTC everywhere; convert Yahoo series with a timezone-aware function.
   - Back-adjust futures rolls.
   - Include perp funding before any multi-day crypto conclusion.
   - No full-sample normalization.

---

## 3. Sources

Time-series momentum and volatility scaling
- Moskowitz, Ooi, Pedersen (2012), Time Series Momentum — https://papers.ssrn.com/abstract=2089463 ; https://econpapers.repec.org/RePEc:eee:jfinec:v:104:y:2012:i:2:p:228-250
- Kim, Tse, Wald (2016), Time series momentum and volatility scaling — https://econpapers.repec.org/RePEc:eee:finmar:v:30:y:2016:i:c:p:103-124 ; summary https://quantpedia.com/deconstructing-the-time-series-momentum-strategy/
- Hurst, Ooi, Pedersen, A Century of Evidence on Trend-Following Investing (AQR) — https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing
- Moreira & Muir (2017), Volatility-Managed Portfolios — https://www.nber.org/papers/w22208
- Liu & Tsyvinski (2021, RFS), Risks and Returns of Cryptocurrency — https://www.nber.org/papers/w24877 ; https://papers.ssrn.com/abstract=3226806

Overnight / intraday / time-of-day
- R-bloggers, Overnight vs. Intraday ETF Returns — https://www.r-bloggers.com/overnight-vs-intraday-etf-returns/
- RobotWealth, Revisiting overnight vs intraday equity returns — https://robotwealth.com/revisiting-overnight-vs-intraday-equity-returns/
- Lou, Polk, Skouras, A tug of war: overnight versus intraday expected returns — https://www.scinapse.io/papers/2797315895
- Gao, Han, Li, Zhou (2018, JFE), Market intraday momentum — https://econpapers.repec.org/RePEc:eee:jfinec:v:129:y:2018:i:2:p:394-414 ; https://alphaarchitect.com/2014/08/attention-prop-traders-the-first-half-hour-of-trading-predicts-the-last-half-hour/ ; replication https://www.quantconnect.com/learning/articles/investment-strategy-library/intraday-etf-momentum
- Shen, Urquhart, Wang (2022), Bitcoin intraday time-series momentum — https://centaur.reading.ac.uk/100181 ; https://research.birmingham.ac.uk/en/publications/bitcoin-intraday-time-series-momentum/
- Zarattini, Aziz, Barbon (2024), Beat the Market (SPY intraday momentum) — https://www.sfi.ch/de/publications/n-24-97-beat-the-market-an-effective-intraday-momentum-strategy-for-s-p500-etf-spy ; https://www.cxoadvisory.com/momentum-investing/complex-intraday-time-series-momentum-strategy-applied-to-spy/
- Zarattini & Aziz (2023), Can Day Trading Really Be Profitable? (ORB) — https://www.cxoadvisory.com/technical-trading/day-trading-with-an-opening-range-breakout-strategy ; net-of-cost replication https://www.mql5.com/en/blogs/post/776235

Crypto seasonality, flow, reversal
- Quantpedia, Intraday Seasonality in Bitcoin (Padyšák & Vojtko, SSRN 4081000) — https://quantpedia.com/strategies/intraday-seasonality-in-bitcoin
- Baur, Cahill, Godfrey, Liu (2019, FRL), Bitcoin time-of-day, day-of-week and month-of-year effects — https://econpapers.repec.org/RePEc:eee:finlet:v:31:y:2019:i:c:p:78-92
- Periodicity in crypto volatility/liquidity — https://arxiv.org/pdf/2109.12142v2.pdf
- The Quarter-Hour Effect (Binance perps, order imbalance 4–12h) — https://arxiv.org/abs/2607.09426
- Explainable Patterns in Cryptocurrency Microstructure (GBM, OFI) — https://arxiv.org/html/2602.00776
- Order-flow imbalance and bitcoin crashes — https://link.springer.com/article/10.1007/s11156-023-01148-1
- Short-horizon mean reversion in crypto (net-of-cost negative) — https://arxiv.org/pdf/2608.21888
- De Nicola, On the Intraday Behavior of Bitcoin — https://ledger.pitt.edu/ojs/ledger/article/view/213
- Bitcoin overreaction to hourly shocks — https://informahealthcare.com/doi/abs/10.1080/15140326.2022.2151253

Calendar
- Quantseeker, Turn-of-the-Month Strategies: Do They Still Work? — https://www.quantseeker.com/p/turn-of-the-month-strategies-do-they
- Maberly & Waggoner, Atlanta Fed WP 2000-11 (TOM disappearing in S&P futures) — https://www.atlantafed.org/research/publications/wp/2000/11.aspx

Gold time-of-day
- CBS thesis, Gold Price Dynamics Around the Clock — https://research.cbs.dk/en/studentProjects/gold-price-dynamics-around-the-clock/
- UZH thesis (gold/silver sessions, fixes) — https://www.merlin.uzh.ch/publication/show/11299
- Gold up in Asia, down in the West (2026 commentary) — https://www.moneymetals.com/news/2026/07/12/a-strange-dichotomy-gold-up-on-the-year-in-asian-markets-down-big-in-the-west-005059

Stop cascades / ICT
- Osler, Stop-Loss Orders and Price Cascades in Currency Markets (NY Fed SR 150) — https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr150.pdf
- Does ICT / Smart Money Concepts work? (no peer-reviewed evidence) — https://thortradecopier.com/blog/does-ict-smart-money-concepts-work
- FVG magnetism test (artifact-driven PF) — https://curupira-blog.pages.dev/blog/fvg-magnetism-fair-value-gaps
- Vendor sweep backtest (unverified, for reference only) — https://backtrex.com/en/blog/liquidity-sweep-strategy-backtest-results

Overfitting / validation
- Bailey & López de Prado (2014), The Deflated Sharpe Ratio — https://papers.ssrn.com/abstract=2460551
- Bailey, Borwein, López de Prado, Zhu, The Probability of Backtest Overfitting — https://papers.ssrn.com/abstract=2326253 ; R package https://cran.r-project.org/package=pbo
- Harvey, Liu, Zhu (2016), …and the Cross-Section of Expected Returns (t > 3 hurdle) — https://www.nber.org/papers/w20592
- López de Prado (2018), Advances in Financial Machine Learning (Wiley): purged/embargoed CV, triple-barrier labels.

Note: some figures above come from search snippets and secondary summaries (CXO, Quantpedia, blogs), not from the full papers. Verify the key numbers against the originals before citing them in any report.
