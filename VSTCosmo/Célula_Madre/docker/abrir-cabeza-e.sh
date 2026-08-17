#!/usr/bin/env bash
set -euo pipefail

# Pantalla auxiliar del organismo E en la Raspberry Pi.
# No abre la pagina completa del organismo: solo la cabeza de E.

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

URL="${ANIMA_CABEZA_URL:-http://127.0.0.1:7788/cabeza}"
FIREFOX="$(command -v firefox || true)"
CHROME="$(command -v chromium-browser || command -v chromium || command -v /snap/bin/chromium || true)"

if [[ -n "$FIREFOX" ]]; then
  export MOZ_DISABLE_GNOME_KEYRING=1
  PROFILE="${ANIMA_CABEZA_PROFILE:-$HOME/.mozilla/anima-cabeza-e}"
  mkdir -p "$PROFILE"
  cat > "$PROFILE/user.js" <<'EOF'
user_pref("signon.rememberSignons", false);
user_pref("signon.autofillForms", false);
user_pref("signon.management.page.breach-alerts.enabled", false);
user_pref("browser.sessionstore.resume_from_crash", false);
EOF
  exec "$FIREFOX" \
    --new-instance \
    --profile "$PROFILE" \
    --kiosk "$URL"
fi

exec "$CHROME" \
  --app="$URL" \
  --start-fullscreen \
  --no-first-run \
  --password-store=basic \
  --disable-session-crashed-bubble \
  --disable-infobars \
  --window-size="${ANIMA_CABEZA_SIZE:-480,800}" \
  --user-data-dir="${ANIMA_CABEZA_PROFILE:-$HOME/.config/anima-cabeza-e}"
