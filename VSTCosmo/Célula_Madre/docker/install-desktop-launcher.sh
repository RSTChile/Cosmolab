#!/usr/bin/env bash
# Instala el icono «Iniciar ANIMA» en el escritorio de Ubuntu (Desktop y/o Escritorio).
set -euo pipefail

CM="$(cd "$(dirname "$0")/.." && pwd)"
NAME="Iniciar ANIMA.desktop"
SRC="$CM/docker/ANIMA.desktop"
APP_DIR="$HOME/.local/share/applications"

chmod +x "$CM/docker/anima-iniciar.sh"

# GNOME en español usa ~/Escritorio (xdg-user-dir), no ~/Desktop.
DESKS=()
if command -v xdg-user-dir >/dev/null 2>&1; then
  XDG_DESK="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
  [ -n "$XDG_DESK" ] && [ -d "$XDG_DESK" ] && DESKS+=("$XDG_DESK")
fi
for d in "$HOME/Escritorio" "$HOME/Desktop" "${XDG_DESKTOP_DIR:-}"; do
  [ -n "$d" ] && [ -d "$d" ] && DESKS+=("$d")
done
# Sin duplicados
mapfile -t DESKS < <(printf '%s\n' "${DESKS[@]}" | awk '!seen[$0]++')

install_desktop() {
  local dest="$1"
  cp "$SRC" "$dest/$NAME"
  chmod +x "$dest/$NAME"
  if command -v gio >/dev/null 2>&1; then
    gio set "$dest/$NAME" metadata::trusted true 2>/dev/null || true
  fi
  echo "  → $dest/$NAME"
}

echo "Instalando icono ANIMA…"
for desk in "${DESKS[@]}"; do
  install_desktop "$desk"
done

mkdir -p "$APP_DIR"
cp "$SRC" "$APP_DIR/anima-iniciar.desktop"
chmod +x "$APP_DIR/anima-iniciar.desktop"
echo "  → $APP_DIR/anima-iniciar.desktop (menú Aplicaciones)"

echo "OK — debería verse en el escritorio. Si no, busca «Iniciar ANIMA» en Aplicaciones."