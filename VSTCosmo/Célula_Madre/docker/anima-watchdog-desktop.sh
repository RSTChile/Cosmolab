#!/usr/bin/env bash
# Relanza el organismo desktop si cae o deja de responder.
set -uo pipefail

CM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$CM_DIR/docker/anima-watchdog-desktop.log"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo "[watchdog] $(date -Iseconds 2>/dev/null || date) iniciado"

while true; do
  sleep 30
  if ! "$CM_DIR/docker/run_native_desktop.sh" status >/dev/null 2>&1; then
    echo "[watchdog] $(date -Iseconds 2>/dev/null || date) organismo caído — relanzando"
    "$CM_DIR/docker/run_native_desktop.sh" start || true
  fi
done