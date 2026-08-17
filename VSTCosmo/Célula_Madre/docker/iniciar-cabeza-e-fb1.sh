#!/usr/bin/env bash
set -euo pipefail

BASE="/home/ubuntu/anima/celula_madre/docker"
CONF="xorg-fb1-cabeza-e.conf"
APP="$BASE/abrir-cabeza-e-fb1.sh"
LOG="$BASE/cabeza-e-fb1.log"
PIDFILE="$BASE/cabeza-e-fb1.pid"

if [ ! -e /dev/fb1 ]; then
  echo "No existe /dev/fb1; revisa el overlay SPI de la pantalla." >&2
  exit 1
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Cabeza E ya corre en fb1 (PID $(cat "$PIDFILE"))"
  exit 0
fi

pkill -f "Xorg :1 .*xorg-fb1-cabeza-e.conf" 2>/dev/null || true
pkill -f "anima-cabeza-e-fb1" 2>/dev/null || true

nohup xinit "$APP" -- /usr/bin/Xorg :1 -config "$CONF" -nolisten tcp -noreset -novtswitch -sharevts vt3 >"$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "Cabeza E fb1 lanzada (PID $(cat "$PIDFILE")); log: $LOG"
