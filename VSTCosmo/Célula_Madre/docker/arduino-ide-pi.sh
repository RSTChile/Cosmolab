#!/usr/bin/env bash
# Arduino IDE 2 en Raspberry Pi (AppImage ARM64). Electron necesita flags en Pi.
set -euo pipefail

APP="${ARDUINO_APPIMAGE:-$HOME/Descargas/arduino-ide_2.3.9_arm64.AppImage}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
[ -f "/run/user/$(id -u)/gdm/Xauthority" ] && XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
export MOZ_WEBRENDER=0
export LIBGL_ALWAYS_SOFTWARE=1

if [ ! -x "$APP" ]; then
  command -v notify-send >/dev/null 2>&1 && \
    notify-send "Arduino IDE" "No encuentro AppImage en $APP" || true
  exit 1
fi

exec "$APP" --no-sandbox --disable-gpu --disable-dev-shm-usage "$@"