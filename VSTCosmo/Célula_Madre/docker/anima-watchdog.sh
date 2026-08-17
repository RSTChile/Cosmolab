#!/usr/bin/env bash
# Relanza el organismo si cae o deja de responder.
set -uo pipefail

CM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$CM_DIR/docker/anima-watchdog.log"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo "[watchdog] $(date -Iseconds) iniciado"

while true; do
  sleep 30
  if ! "$CM_DIR/docker/run_native_pi.sh" status >/dev/null 2>&1; then
    echo "[watchdog] $(date -Iseconds) organismo caído — relanzando"
    "$CM_DIR/docker/run_native_pi.sh" start || true
  fi
done