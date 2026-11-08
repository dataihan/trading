#!/usr/bin/env bash
# Install the paper-trading bot + data collector as systemd timers (Ubuntu 22.04+).
# Usage (on the server, inside the cloned repo):  sudo bash deploy/install.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="${SUDO_USER:-$(whoami)}"
PY="$REPO/.venv/bin/python"

echo "== repo: $REPO  user: $USER_NAME"

# 1. venv + deps
if [ ! -x "$PY" ]; then
  sudo -u "$USER_NAME" python3 -m venv "$REPO/.venv"
fi
sudo -u "$USER_NAME" "$REPO/.venv/bin/pip" install -q --upgrade pip
sudo -u "$USER_NAME" "$REPO/.venv/bin/pip" install -q numpy pandas requests pyarrow scipy

# 2. .env check
if [ ! -f "$REPO/.env" ]; then
  cp "$REPO/.env.example" "$REPO/.env"; chown "$USER_NAME" "$REPO/.env"; chmod 600 "$REPO/.env"
  echo "!! Fill in $REPO/.env (testnet + Alpaca paper keys), then re-run this script."; exit 1
fi
chmod 600 "$REPO/.env"

# 3. connectivity check (Binance returns HTTP 451 from restricted locations)
for u in https://fapi.binance.com/fapi/v1/ping https://testnet.binancefuture.com/fapi/v1/ping https://paper-api.alpaca.markets; do
  printf "%-50s %s\n" "$u" "$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$u")"
done

# 4. systemd units
unit() {  # name, schedule, command
  cat > "/etc/systemd/system/trading-$1.service" <<EOF
[Unit]
Description=trading $1
After=network-online.target
[Service]
Type=oneshot
User=$USER_NAME
WorkingDirectory=$REPO
Environment=PYTHONPATH=$REPO/src
ExecStart=$PY -m $3
EOF
  cat > "/etc/systemd/system/trading-$1.timer" <<EOF
[Unit]
Description=trading $1 schedule
[Timer]
OnCalendar=$2
Persistent=true
[Install]
WantedBy=timers.target
EOF
  systemctl daemon-reload
  systemctl enable --now "trading-$1.timer"
}

unit crypto    "*-*-* 00:01:00 UTC"                         "trading.live.runner crypto"
unit qqq       "Mon..Fri *-*-* 15:45:00 America/New_York"   "trading.live.runner qqq"
unit collector "*-*-* *:00:30 UTC"                          "trading.live.collector"
unit equity    "*-*-* 00:30:00 UTC"                         "trading.live.report snapshot"

systemctl list-timers 'trading-*' --no-pager
echo "== done. Logs: journalctl -u 'trading-*' -f   Ledger: $REPO/data/live/ledger.jsonl"
