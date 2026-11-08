#!/usr/bin/env bash
# Paper trading from a Claude cloud session (no persistent disk).
# Logs (data/live/*.jsonl) live on git branch `paper-log`: restore -> run -> commit -> push.
#
#   bash deploy/cloud_run.sh          # daily: crypto signals + orders on testnet, equity snapshot
#   bash deploy/cloud_run.sh report   # weekly: live vs backtest (read-only, no push)
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
LOG_BRANCH=${LOG_BRANCH:-paper-log}
export PYTHONPATH=src
export PORTFOLIO=${PORTFOLIO:-strategies/portfolio_v1_crypto.json}

python3 -c "import pandas, pyarrow, requests, cryptography" 2>/dev/null ||
  pip install -q pandas pyarrow requests cryptography numpy

# 1. restore logs
WT=$(mktemp -d)
trap 'git worktree remove --force "$WT" 2>/dev/null || true' EXIT
mkdir -p data/live
if git fetch -q origin "$LOG_BRANCH" 2>/dev/null; then
  git worktree add -q -f -B "$LOG_BRANCH" "$WT" "origin/$LOG_BRANCH"
  cp "$WT"/*.jsonl data/live/ 2>/dev/null || true
elif git show-ref -q --verify "refs/heads/$LOG_BRANCH"; then
  git worktree add -q -f "$WT" "$LOG_BRANCH"
  cp "$WT"/*.jsonl data/live/ 2>/dev/null || true
else
  git worktree add -q -f --detach "$WT"
  git -C "$WT" checkout -q --orphan "$LOG_BRANCH"
  git -C "$WT" rm -rq --cached . >/dev/null
  find "$WT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
fi

if [ "$MODE" = report ]; then
  python3 -m trading.live.report
  echo "--- last ledger rows"; tail -n 8 data/live/ledger.jsonl 2>/dev/null || echo "(no ledger yet)"
  exit 0
fi

# 2. run
python3 -m trading.live.runner crypto
python3 -m trading.live.report snapshot

# 3. save logs
cp data/live/*.jsonl "$WT"/
git -C "$WT" add -A
git -C "$WT" -c user.name="paper-bot" -c user.email="paper-bot@users.noreply.github.com" \
  commit -q -m "paper log $(date -u +%F)" || echo "nothing new to commit"
[ -n "${NO_PUSH:-}" ] && { echo "NO_PUSH set, skip push"; exit 0; }
for i in 1 2 3 4; do
  git -C "$WT" push -q origin "$LOG_BRANCH" && break
  sleep $((2 ** i))
done
