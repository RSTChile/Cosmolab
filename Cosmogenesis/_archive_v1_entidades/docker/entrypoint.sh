#!/bin/sh
# Punto de entrada CG001 — un mismo cuerpo, tres roles (análogo a ANIMA).
# Los universos corren bajo watchdog: si /estado deja de responder, se relanza.

run_with_watchdog() {
  SCRIPT="$1"; PORT="$2"
  while true; do
    python "$SCRIPT" &
    PID=$!
    echo "[watchdog] CG001 arrancado (PID=$PID, puerto=$PORT)"
    sleep 30
    fails=0
    while kill -0 "$PID" 2>/dev/null; do
      if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/estado', timeout=5)" >/dev/null 2>&1; then
        fails=0
      else
        fails=$((fails + 1))
        echo "[watchdog] /estado no responde ($fails/3)"
        if [ "$fails" -ge 3 ]; then
          echo "[watchdog] proceso COLGADO → matando y relanzando"
          kill "$PID" 2>/dev/null; sleep 3; kill -9 "$PID" 2>/dev/null
          break
        fi
      fi
      sleep 20
    done
    wait "$PID" 2>/dev/null
    echo "[watchdog] WebLive terminó → relanzando en 3s"
    sleep 3
  done
}

case "${CG_ROLE:-lab}" in
  lab)
    echo "[cg001] rol=LAB (ε=${CG_EPSILON:-0.00001}, ${CG_EXPERIMENT_ID:-CG001-B}, puerto ${CG_PUERTO:-7888})"
    run_with_watchdog /app/CG001/server/cg001_weblive.py "${CG_PUERTO:-7888}"
    ;;
  a)
    echo "[cg001] rol=A legado (ε=0)"
    export CG_EPSILON=0
    export CG_EXPERIMENT_ID=CG001-A
    run_with_watchdog /app/CG001/server/cg001_weblive.py "${CG_PUERTO:-7888}"
    ;;
  b)
    echo "[cg001] rol=B legado (ε>0)"
    export CG_EPSILON="${CG_EPSILON:-0.00001}"
    export CG_EXPERIMENT_ID=CG001-B
    run_with_watchdog /app/CG001/server/cg001_weblive.py "${CG_PUERTO:-7889}"
    ;;
  observatorio)
    echo "[cg001] rol=OBSERVATORIO legado (puerto ${CG_OBS_PORT:-7900})"
    exec python /app/CG001/observatorio/cg001_observatorio.py
    ;;
  *)
    echo "CG_ROLE desconocido: '${CG_ROLE}' (usa lab|a|b|observatorio)"
    exit 1
    ;;
esac