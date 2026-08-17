#!/usr/bin/env bash
# Al login: cabeza 3D (Three.js) en ventana de aplicación. NO kiosk salvo ANIMA_DISPLAY=hdmi|both.
set -euo pipefail

CM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$CM_DIR/docker/anima-boot.log"
PID_KIOSK="$CM_DIR/docker/anima-kiosk.pid"
PID_HEADLESS="$CM_DIR/docker/anima-pi-headless.pid"
ANIMA_DISPLAY="${ANIMA_DISPLAY:-desktop}"
CABEZA_URL="${CABEZA_URL:-http://127.0.0.1:7788/cabeza}"
# shellcheck source=/dev/null
[ -f "$CM_DIR/docker/.env.pi" ] && set -a && source "$CM_DIR/docker/.env.pi" && set +a

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo "=== $(date -Iseconds) anima-boot (display=$ANIMA_DISPLAY) ==="

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"

"$CM_DIR/docker/run_native_pi.sh" start || true

for _ in $(seq 1 90); do
  curl -sf http://127.0.0.1:7788/estado >/dev/null 2>&1 && break
  sleep 2
done

start_hdmi_kiosk() {
  local url="${1:-$CABEZA_URL}"
  pgrep -f '127.0.0.1:7788' | xargs -r kill 2>/dev/null || true
  sleep 1
  if command -v firefox >/dev/null 2>&1; then
    firefox --kiosk "$url" >>"$LOG" 2>&1 &
  else
    /snap/bin/chromium --kiosk --disable-gpu --disable-webgl "$url" >>"$LOG" 2>&1 &
  fi
  echo $! >"$PID_KIOSK"
  echo "[boot] kiosk HDMI → $url PID $(cat "$PID_KIOSK")"
}

firefox_pi_env() {
  # PiScreen + drm: GLX falla; WebGL vía llvmpipe para Three.js en /cabeza
  export MOZ_WEBRENDER=0
  export MOZ_ACCELERATED=0
  export MOZ_X11_EGL_FORCE_SLOW=1
  export LIBGL_ALWAYS_SOFTWARE=1
}

start_pi_headless() {
  # Three.js exacto vía Chromium headless → fb1 (WebGL del navegador en X falla en PiScreen).
  if pgrep -f 'anima-pi-headless.py' >/dev/null 2>&1; then
    echo "[boot] pi-headless ya activo"
    return 0
  fi
  if systemctl --user is-active anima-pi-headless.service >/dev/null 2>&1; then
    echo "[boot] anima-pi-headless.service activo"
    return 0
  fi
  command -v chromium >/dev/null 2>&1 || command -v /snap/bin/chromium >/dev/null 2>&1 || {
    echo "[boot] AVISO: chromium no instalado — sin cabeza en fb1"
    return 1
  }
  python3 "$CM_DIR/docker/anima-pi-headless.py" >>"$LOG" 2>&1 &
  echo $! >"$PID_HEADLESS"
  echo "[boot] pi-headless 3D → fb1 PID $(cat "$PID_HEADLESS")"
}

start_cabeza_window() {
  # HDMI / Mac: ventana con barra × − + (Three.js en navegador).
  if pgrep -f 'firefox.*cabeza' >/dev/null 2>&1; then
    echo "[boot] firefox /cabeza ya activo"
    return 0
  fi
  command -v firefox >/dev/null || return 0
  pgrep -f 'firefox.*7788' | xargs -r kill 2>/dev/null || true
  sleep 1
  firefox_pi_env
  firefox --new-window "$CABEZA_URL" >>"$LOG" 2>&1 &
  echo "[boot] firefox ventana → $CABEZA_URL PID $!"
}

# Detener render 2D / firefox kiosk en pantalla pequeña
systemctl --user stop anima-pi-screen.service 2>/dev/null || true
pkill -f anima-pi-screen.py 2>/dev/null || true
pgrep -f 'firefox.*7788' | xargs -r kill 2>/dev/null || true

case "$ANIMA_DISPLAY" in
  desktop)
    systemctl --user stop anima-pi-headless.service 2>/dev/null || true
    systemctl --user stop anima-pi-screen.service 2>/dev/null || true
    echo "[boot] modo escritorio — HDMI libre; organismo en :7788"
    ;;
  small)
    if [ -e /dev/fb1 ]; then
      start_pi_headless
      echo "[boot] pantalla pequeña = cabeza 3D headless (fb1); HDMI = escritorio GNOME"
    else
      start_cabeza_window
      echo "[boot] sin fb1 — cabeza en ventana Firefox"
    fi
    ;;
  hdmi) start_hdmi_kiosk "$CABEZA_URL" ;;
  both)
    [ -e /dev/fb1 ] && start_pi_headless || true
    start_cabeza_window
    ;;
esac
echo "[boot] listo"