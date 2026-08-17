#!/usr/bin/env bash
# VST_AudioServer — mismo puente que en el Mac: captura Rode → TCP → organismo.
# Uso: ./anima-audio-server.sh list|start|stop|status|test
set -euo pipefail

CM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="$CM_DIR/docker/anima-audioserver.pid"
LOG="$CM_DIR/docker/anima-audioserver.log"
# shellcheck source=/dev/null
[ -f "$CM_DIR/docker/.env.pi" ] && set -a && source "$CM_DIR/docker/.env.pi" && set +a

DEVICE="${ANIMA_AUDIO_DEVICE:-Main Multitrack}"
HOST="${VST_AUDIO_BIND:-0.0.0.0}"
PORT="${VST_SERVIDOR_PORT:-8765}"
PYTHON="${CM_DIR}/.venv-pi/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

cmd="${1:-status}"

case "$cmd" in
  list)
    cd "$CM_DIR"
    PYTHONPATH="$CM_DIR:$CM_DIR/audio" "$PYTHON" audio/VST_AudioServer.py --list
    ;;
  test)
    cd "$CM_DIR"
    PYTHONPATH="$CM_DIR:$CM_DIR/audio" "$PYTHON" audio/VST_AudioServer.py --test --port "$PORT"
    ;;
  start)
    mkdir -p "$(dirname "$LOG")"
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "AudioServer ya corre (PID $(cat "$PIDFILE"))"
      exit 0
    fi
    cd "$CM_DIR"
    PYTHONPATH="$CM_DIR:$CM_DIR/audio" nohup "$PYTHON" audio/VST_AudioServer.py \
      --device "$DEVICE" --host "$HOST" --port "$PORT" >>"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    echo "AudioServer PID $(cat "$PIDFILE") · $DEVICE · :$PORT"
    ;;
  stop)
    [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null && rm -f "$PIDFILE" && echo "Detenido" || echo "No corría"
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Vivo PID $(cat "$PIDFILE") · puerto $PORT"
      ss -ltn 2>/dev/null | grep ":$PORT " || netstat -ltn 2>/dev/null | grep ":$PORT " || true
    else
      echo "No corre"
      exit 1
    fi
    ;;
  *)
    echo "Uso: $0 [list|start|stop|status|test]"
    exit 1
    ;;
esac