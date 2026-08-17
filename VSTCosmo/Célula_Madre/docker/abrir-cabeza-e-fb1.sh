#!/usr/bin/env bash
set -euo pipefail

# Abre solamente la cabeza del Organismo E en la pantalla GPIO/SPI (/dev/fb1).
# Esta sesion X es independiente del escritorio principal/VNC.

ENV_FILE="${ANIMA_CABEZA_ENV:-/home/ubuntu/anima/celula_madre/docker/cabeza-e-fb1.env}"
[ -f "$ENV_FILE" ] && set -a && source "$ENV_FILE" && set +a

export DISPLAY="${DISPLAY:-:1}"
ROTATE="${ANIMA_CABEZA_ROTATE:-180}"
URL_VERSION="${ANIMA_CABEZA_URL_VERSION:-fb1}"
URL="${ANIMA_CABEZA_URL:-http://127.0.0.1:7788/cabeza?fb1=1&rotate=${ROTATE}&v=${URL_VERSION}}"
PROFILE="${ANIMA_CABEZA_PROFILE:-$HOME/.mozilla/anima-cabeza-e-fb1}"
CHROME_PROFILE="${ANIMA_CABEZA_CHROME_PROFILE:-$HOME/snap/chromium/common/anima-cabeza-e-fb1}"

mkdir -p "$PROFILE"
mkdir -p "$CHROME_PROFILE"
mkdir -p "$CHROME_PROFILE/Default"
CURSOR_DIR="${ANIMA_CABEZA_CURSOR_DIR:-$HOME/.cache/anima-cabeza-e}"
mkdir -p "$CURSOR_DIR"
cat > "$CURSOR_DIR/blank.xbm" <<'EOF'
#define blank_width 16
#define blank_height 16
static unsigned char blank_bits[] = {
  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
  0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
};
EOF
# Pantalla SPI: sin apagado por idle. Xorg DPMS (Standby/Suspend/Off=600s) dejaba
# "Monitor is Off" y la cabeza desaparecía a los ~10 min aunque Chromium siguiera vivo.
# Nota: Xorg -config relativo solo mira /etc/X11 (a menudo conf vieja sin BlankTime 0).
# CLI -s 0 -dpms (ejecutar-cabeza-e-fb1.sh) + xset + keepalive cubren el caso real.
anti_blank() {
  xset s off 2>/dev/null || true
  xset s noblank 2>/dev/null || true
  xset s 0 0 2>/dev/null || true
  xset dpms 0 0 0 2>/dev/null || true
  xset -dpms 2>/dev/null || true
  xset dpms force on 2>/dev/null || true
  # Desblank del framebuffer SPI si el driver lo expone (video group).
  if [ -w /sys/class/graphics/fb1/blank ]; then
    echo 0 >/sys/class/graphics/fb1/blank 2>/dev/null || true
  fi
}
anti_blank
# Keepalive: re-aplicar cada 45s por si algo re-habilita DPMS o blankea el fb.
(
  while true; do
    sleep 45
    anti_blank
  done
) >/dev/null 2>&1 &
xsetroot -cursor "$CURSOR_DIR/blank.xbm" "$CURSOR_DIR/blank.xbm" 2>/dev/null || true
xdotool mousemove 479 319 2>/dev/null || true
unclutter -display "$DISPLAY" -idle 0.2 -root >/dev/null 2>&1 &
cat > "$PROFILE/user.js" <<'EOF'
user_pref("signon.rememberSignons", false);
user_pref("signon.autofillForms", false);
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.shell.checkDefaultBrowser", false);
EOF
python3 - "$CHROME_PROFILE/Default/Preferences" <<'PY' 2>/dev/null || true
import json, os, sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        prefs = json.load(f)
except Exception:
    prefs = {}
prefs.setdefault("translate", {})["enabled"] = False
prefs.setdefault("browser", {})["enable_spellchecking"] = False
prefs.setdefault("intl", {})["accept_languages"] = "es-CL,es"
os.makedirs(os.path.dirname(path), exist_ok=True)
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(prefs, f, separators=(",", ":"))
os.replace(tmp, path)
PY

CHROME="$(command -v chromium-browser || command -v chromium || command -v /snap/bin/chromium || true)"
if [[ -n "$CHROME" ]]; then
  exec "$CHROME" \
    --kiosk "$URL" \
    --no-first-run \
    --password-store=basic \
    --disable-session-crashed-bubble \
    --disable-infobars \
    --disable-translate \
    --disable-features=Translate,TranslateUI,TFLiteLanguageDetectionEnabled \
    --lang=es-CL \
    --hide-cursor \
    --disable-dev-shm-usage \
    --enable-unsafe-swiftshader \
    --use-gl=swiftshader \
    --window-position=0,0 \
    --window-size=480,320 \
    --user-data-dir="$CHROME_PROFILE"
fi

export MOZ_DISABLE_GNOME_KEYRING=1
exec firefox --new-instance --profile "$PROFILE" --kiosk "$URL"
