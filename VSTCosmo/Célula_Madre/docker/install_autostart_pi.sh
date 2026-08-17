#!/usr/bin/env bash
# Instala arranque automático en la Pi. Ejecutar EN la Pi o vía ssh.
set -euo pipefail

CM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AUTOSTART="$HOME/.config/autostart"
SYSTEMD_USER="$HOME/.config/systemd/user"

mkdir -p "$AUTOSTART" "$SYSTEMD_USER"
chmod +x "$CM_DIR/docker/anima-boot.sh" "$CM_DIR/docker/anima-watchdog.sh" \
  "$CM_DIR/docker/anima-pi-headless.py" "$CM_DIR/docker/anima-iniciar.sh" \
  "$CM_DIR/docker/install-desktop-launcher.sh" \
  "$CM_DIR/docker/run_native_pi.sh" "$CM_DIR/docker/anima-pi-screen.py" "$CM_DIR/docker/fix_piscreen.sh"

bash "$CM_DIR/docker/install-desktop-launcher.sh" 2>/dev/null || true

cp "$CM_DIR/docker/anima-boot.desktop" "$AUTOSTART/anima-boot.desktop"
chmod 644 "$AUTOSTART/anima-boot.desktop"

cp "$CM_DIR/docker/systemd/"*.service "$SYSTEMD_USER/"
systemctl --user daemon-reload
systemctl --user enable anima-organismo.service anima-watchdog.service
systemctl --user disable anima-pi-screen.service 2>/dev/null || true
# cabeza 3D headless → fb1 (Three.js exacto)
if [ -e /dev/fb1 ]; then
  systemctl --user enable anima-pi-headless.service
  systemctl --user start anima-pi-headless.service 2>/dev/null || true
fi

loginctl enable-linger "$(whoami)" 2>/dev/null || true

mkdir -p "$HOME/.config/environment.d"
cat >"$HOME/.config/environment.d/anima.conf" <<EOF
ANIMA_DISPLAY=small
EOF

echo "OK — arranque automático instalado:"
echo "  [boot]  systemd: organismo + watchdog (+ pi-headless si fb1)"
echo "  [login] GNOME:   cabeza 3D en fb1; HDMI = escritorio extendido"
echo ""
echo "  Si la pantalla pequeña no funciona:"
echo "    sudo bash $CM_DIR/docker/fix_piscreen.sh && sudo reboot"
echo ""
systemctl --user is-enabled anima-organismo.service anima-watchdog.service 2>/dev/null || true