# program.md — Autoresearch loop for trading

Inspired by Karpathy's `autoresearch` and ATLAS (chrisworsey55/atlas-gic): **git is the memory, the
backtest is the loss function, and a change survives only if it improves the out-of-sample score.**
The difference from ATLAS: our "weights" are code and config (rules + ML), not LLM prompts, so
every result is reproducible and cheap to re-run.

## Goal
Find strategies that are profitable **after fees, slippage and funding** on data they were not tuned on.

## Data splits (fixed — do not change)
| split | intraday / crypto | daily long history |
|---|---|---|
| train | 2020-01 → 2023-12 | 1985 → 2014 |
| valid | 2024-01 → 2025-06 | 2015 → 2021 |
| holdout | 2025-07 → now | 2022 → now |

The holdout is opened only by the risk-auditor, once per frozen portfolio version.

## Team
| agent | file | role |
|---|---|---|
| orchestrator | (main Claude Code session) | runs the loop below and decides when to stop |
| researcher | `.claude/agents/researcher.md` | evidence → hypotheses (`research/hypotheses.md`) |
| strategy-builder | `.claude/agents/strategy-builder.md` | hypotheses → code (`src/trading/`) |
| backtester | `.claude/agents/backtester.md` | code → metrics, KEEP/REJECT (`results/`) |
| risk-auditor | `.claude/agents/risk-auditor.md` | look-ahead review, multiple testing, holdout, report |

## Loop (one round)
1. **researcher**: add ≤3 new hypotheses, or refine failed ones.
2. **strategy-builder**: implement them, then smoke-test.
3. **backtester**: run on train and valid with 2× cost and parameter robustness checks, then write KEEP/REJECT.
4. If there is a KEEP: add it to a NEW portfolio version `strategies/portfolio_vN.json` (fixed weights),
   run `python3 autoresearch/portfolio.py` (train and valid). If the portfolio's valid Sharpe beats the previous
   version → `git commit`. Otherwise → `git reset --hard` (Darwinian keep/revert).
5. **risk-auditor**: when a version has been kept → review → holdout once → `results/REPORT.md`.
6. Stop when the holdout is profitable at 2× cost with Sharpe > 0.8 and max DD < 20%, OR after 5 rounds without
   improvement (then report honestly that nothing better was found).

## Hard rules
- No look-ahead (see strategy-builder). No deleting lines from `results/experiments.jsonl`.
- Costs are always on: Binance perps 6 bps/side + funding; US ETFs 2 bps/side.
- Prefer simple, documented effects and ensembles over finely tuned parameters.
- Research only. Nothing here places real orders. Live trading needs Han's explicit approval and a separate paper-trading step.

## How to run unattended
In Claude Code inside this repo:
> "Read program.md and run the autoresearch loop for N rounds using the subagents in .claude/agents."
