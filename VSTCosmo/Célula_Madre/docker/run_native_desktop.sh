#!/usr/bin/env bash
# Arranca un organismo ANIMA nativamente en PC (sin Docker).
# Uso: ./run_native_desktop.sh [start|stop|status]
set -euo pipefail

CM_DIR="$(cd "$(dirname "$0")/.." && pwd)"

load_env_file() {
  local f="$1"
  [ -f "$f" ] || return 0
  set -a
  # shellcheck source=/dev/null
  source "$f"
  set +a
}

load_env_file "$HOME/.config/anima/organismo.env"
load_env_file /etc/anima/organismo.env
load_env_file "$CM_DIR/docker/.env.desktop"

case "$(uname -s)" in
  Darwin)
    DATA_DIR="${ANIMA_DATA_DIR:-$HOME/Library/Application Support/ANIMA}"
    ;;
  *)
    DATA_DIR="${ANIMA_DATA_DIR:-$HOME/.local/share/anima}"
    ;;
esac

VENV="$CM_DIR/.venv-desktop"
PIDFILE="$CM_DIR/docker/anima-desktop.pid"
LOG="$CM_DIR/docker/anima-desktop.log"
HISTORY="${VST_HISTORY_DIR:-$DATA_DIR/history}"

export VST_PUERTO="${VST_PUERTO:-7788}"
export VST_ORGANISMO_ID="${VST_ORGANISMO_ID:-ANIMA_LOCAL_PC}"
export VST_ORGANISMO_NOMBRE="${VST_ORGANISMO_NOMBRE:-${VST_ORGANISMO_LABEL:-Animalito}}"
export VST_ORGANISMO_LABEL="${VST_ORGANISMO_LABEL:-$VST_ORGANISMO_NOMBRE}"
export ANIMA_AUTOSTART=1
export ANIMA_FUENTE_DEFECTO="${ANIMA_FUENTE_DEFECTO:-demo:silencio}"
export ANIMA_ESCUCHAR_PAR="${ANIMA_ESCUCHAR_PAR:-0}"
export ANIMA_ESCUCHAR_TODOS="${ANIMA_ESCUCHAR_TODOS:-0}"
export ANIMA_BIND=0.0.0.0
export ANIMA_AUDIO_MODE="${ANIMA_AUDIO_MODE:-local}"
export ANIMA_AUDIO_LOCAL_POLICY="${ANIMA_AUDIO_LOCAL_POLICY:-prefer-system-input}"
export ANIMA_AUDIO_LOCAL_MATCH="${ANIMA_AUDIO_LOCAL_MATCH:-}"
export ANIMA_AUDIO_SERVER_OPTIONAL="${ANIMA_AUDIO_SERVER_OPTIONAL:-1}"
export VST_DISABLE_DIRECT_AUDIO="${VST_DISABLE_DIRECT_AUDIO:-0}"
export ANIMA_SDR_ENABLE="${ANIMA_SDR_ENABLE:-0}"
export ANIMA_PRESENCE_MODE="${ANIMA_PRESENCE_MODE:-local}"
export ANIMA_VISIBILITY="${ANIMA_VISIBILITY:-local}"
export ANIMA_AISLAMIENTO_TIEMPO_SATURACION="${ANIMA_AISLAMIENTO_TIEMPO_SATURACION:-300}"
export ANIMA_DISPLAY="${ANIMA_DISPLAY:-desktop}"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export VST_LIFE_MODE=continuous
export VST_HISTORY_ENABLE=true
export VST_HISTORY_FORMAT=csv
export VST_HISTORY_DIR="$HISTORY"
export VST_HISTORY_ROTATE_SECONDS=3600
export VST_SNAPSHOT_SECONDS=600
export VST_RECORD_VOICE_WAV=true
export VST_RECORD_EXTERNAL_AUDIO=false
export PYTHONPATH="${CM_DIR}:${CM_DIR}/audio${PYTHONPATH:+:$PYTHONPATH}"
export ANIMA_AUTOSTART_DELAY="${ANIMA_AUTOSTART_DELAY:-5}"

cmd="${1:-start}"

venv_ok() {
  [ -x "$VENV/bin/python" ] && [ -x "$VENV/bin/pip" ] && \
    "$VENV/bin/python" -c "import numpy" 2>/dev/null
}

pick_python() {
  if venv_ok; then
    echo "$VENV/bin/python"
  else
    echo "python3"
  fi
}

wait_audio_server() {
  [ "${ANIMA_WAIT_AUDIOSERVER:-0}" = "1" ] || return 0
  [ -z "${ANIMA_RODE_CH_L:-}" ] && [ -z "${ANIMA_MUNDO_CANAL:-}" ] && return 0
  local host="${VST_SERVIDOR_HOST:-127.0.0.1}"
  local port="${VST_SERVIDOR_PORT:-8765}"
  echo "[anima] esperando AudioServer en ${host}:${port}…"
  local i
  for i in $(seq 1 90); do
    if python3 -c "import socket; s=socket.create_connection(('${host}',${port}),2); s.close()" 2>/dev/null; then
      echo "[anima] AudioServer listo (intento $i)"
      return 0
    fi
    sleep 2
  done
  echo "[anima] AVISO: AudioServer no respondió en 3 min — el organismo intentará igual"
}

setup_venv() {
  PYTHON_BIN="python3"
  if venv_ok; then
    "$VENV/bin/pip" install -q -r "$CM_DIR/requirements-desktop.txt" 2>/dev/null || true
    PYTHON_BIN="$(pick_python)"
    return
  fi
  if [ -d "$VENV" ] && ! venv_ok; then
    echo "[anima] venv incompleto — recreando"
    rm -rf "$VENV" 2>/dev/null || true
  fi
  if command -v python3 >/dev/null 2>&1 && python3 -m venv "$VENV" >/dev/null 2>&1 && venv_ok; then
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q -r "$CM_DIR/requirements-desktop.txt"
    PYTHON_BIN="$(pick_python)"
    return
  fi
  echo "[anima] venv no disponible — pip --user + python3"
  pip3 install -q --user --upgrade pip 2>/dev/null || true
  pip3 install -q --user -r "$CM_DIR/requirements-desktop.txt" 2>/dev/null || true
  PYTHON_BIN="$(pick_python)"
}

case "$cmd" in
  start)
    mkdir -p "$HISTORY" "$(dirname "$LOG")" "$DATA_DIR"
    setup_venv
    PYTHON_BIN="$(pick_python)"
    wait_audio_server
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Ya corre (PID $(cat "$PIDFILE"))"
      exit 0
    fi
    cd "$CM_DIR"
    nohup "$PYTHON_BIN" web/VST_CelulaMadre_WebLive_A.py >>"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    echo "Arrancado PID $(cat "$PIDFILE") — log: $LOG"
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then
      kill "$(cat "$PIDFILE")" 2>/dev/null || true
      rm -f "$PIDFILE"
      echo "Detenido"
    else
      echo "No hay PIDfile"
    fi
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Vivo PID $(cat "$PIDFILE")"
      curl -sf "http://127.0.0.1:${VST_PUERTO}/estado" | head -c 400 || echo "(/estado aún no responde)"
      echo ""
    else
      echo "No corre"
      exit 1
    fi
    ;;
  *)
    echo "Uso: $0 [start|stop|status]"
    exit 1
    ;;
esac