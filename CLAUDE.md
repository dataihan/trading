# CLAUDE.md

Trading research repo (crypto, Binance perps, gold/silver/oil, Nasdaq). Owner: Han.

- Start with `program.md` (autoresearch loop + rules) and `results/REPORT.md` (current best).
- Subagents: `.claude/agents/` (researcher, strategy-builder, backtester, risk-auditor).
- Evaluate a config: `python3 autoresearch/evaluate.py '{"market":"binance:BTCUSDT:1h","strategy":"trend_ensemble","params":{}}'`
- Portfolio: `python3 autoresearch/portfolio.py` (add `--holdout` only via the risk-auditor).
- Data is cached in `data/raw/*.parquet` (gitignored) and downloads automatically. `api.binance.com` may be geo-blocked, so use the `data.binance.vision` archive.
- Never place live orders. Reply to Han briefly, in Vietnamese if Han writes in Vietnamese.
