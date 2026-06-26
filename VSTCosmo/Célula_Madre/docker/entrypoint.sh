#!/bin/sh
# Punto de entrada: un mismo cuerpo, tres roles. ANIMA_ROLE elige quién despierta en este contenedor.
# Los organismos (a/b) corren bajo un WATCHDOG: si el proceso muere O se cuelga (PID vivo pero /estado
# deja de responder), se mata y se relanza al instante. Antes el proceso se colgaba cada ~2 h y, como
# python era PID 1 pero seguía "vivo", restart:unless-stopped no lo detectaba → huecos en la biografía.
# NOTA: sin `set -e` aquí (el chequeo de salud devuelve no-cero cuando el server está caído — es esperado).

run_with_watchdog() {
  SCRIPT="$1"; PORT="$2"
  while true; do
    python "$SCRIPT" &
    PID=$!
    echo "[watchdog] WebLive arrancado (PID=$PID, puerto=$PORT)"
    sleep 45                         # margen para arranque + autoarranque acoplado (delay 6s)
    fails=0
    while kill -0 "$PID" 2>/dev/null; do
      if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/estado', timeout=5)" >/dev/null 2>&1; then
        fails=0
      else
        fails=$((fails + 1))
        echo "[watchdog] /estado no responde ($fails/3)"
        if [ "$fails" -ge 3 ]; then  # ~3 fallos consecutivos (~60s colgado) → reinicio duro
          echo "[watchdog] proceso COLGADO → matando y relanzando"
          kill "$PID" 2>/dev/null; sleep 3; kill -9 "$PID" 2>/dev/null
          break
        fi
      fi
      sleep 20
    done
    wait "$PID" 2>/dev/null
    echo "[watchdog] WebLive terminó/colgó → relanzando en 3s"
    sleep 3
  done
}

case "${ANIMA_ROLE:-a}" in
  a)   echo "[anima] rol=A (organismo, puerto ${VST_PUERTO:-7788}) — bajo watchdog"
       run_with_watchdog /app/celula_madre/web/VST_CelulaMadre_WebLive_A.py "${VST_PUERTO:-7788}" ;;
  b)   echo "[anima] rol=B (organismo, puerto ${VST_PUERTO:-7799}) — bajo watchdog"
       run_with_watchdog /app/celula_madre/web/VST_CelulaMadre_WebLive_B.py "${VST_PUERTO:-7799}" ;;
  mcp) echo "[anima] rol=MCP (membrana de la díada, HTTP)"
       exec python /app/celula_madre/mcp/vst_mcp_diada.py --http ;;
  conversacion) echo "[anima] rol=CONVERSACION (observatorio permanente)"
       exec python /app/celula_madre/conversacion/vst_conversacion.py ;;
  *)   echo "ANIMA_ROLE desconocido: '${ANIMA_ROLE}' (usa a|b|mcp|conversacion)"; exit 1 ;;
esac
