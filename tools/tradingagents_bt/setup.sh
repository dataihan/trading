#!/usr/bin/env bash
# Install TauricResearch/TradingAgents in its own venv (outside this repo).
set -euo pipefail
DIR=${TA_DIR:-/home/user/tauricresearch}
mkdir -p "$DIR"
[ -d "$DIR/tradingagents/.git" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 https://github.com/TauricResearch/tradingagents "$DIR/tradingagents"
[ -x "$DIR/venv/bin/python" ] || python3 -m venv "$DIR/venv"
"$DIR/venv/bin/pip" install -q "$DIR/tradingagents"
"$DIR/venv/bin/python" -c "import tradingagents; print('tradingagents ok')"
