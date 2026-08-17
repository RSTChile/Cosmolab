#!/usr/bin/env bash
# Instalador ANIMA Desktop Runtime para macOS (sin sudo).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERSION="${ANIMA_VERSION:-unknown}"
CM_DEST="$HOME/Library/Application Support/ANIMA/celula_madre"
CONFIG_DIR="$HOME/.config/anima"
LOG_DIR="$HOME/Library/Logs/ANIMA"
BIN_DIR="${ANIMA_BIN_DIR:-$HOME/.local/bin}"
LAUNCH_DIR="$HOME/Library/LaunchAgents"

echo "ANIMA Desktop Runtime — instalador macOS ($VERSION)"
echo "Destino: $CM_DEST"
echo

mkdir -p "$CM_DEST" "$CONFIG_DIR" "$LOG_DIR" "$BIN_DIR" "$LAUNCH_DIR" "$HOME/.anima"

if [ -d "$SCRIPT_DIR/celula_madre" ]; then
  rsync -a --delete \
    --exclude '.DS_Store' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.venv-desktop' \
    "$SCRIPT_DIR/celula_madre/" "$CM_DEST/"
else
  echo "ERROR: no se encontró celula_madre/ en el paquete." >&2
  exit 1
fi

chmod +x \
  "$CM_DEST/docker/run_native_desktop.sh" \
  "$CM_DEST/docker/anima-watchdog-desktop.sh" 2>/dev/null || true

for bin in anima anima-config anima-status; do
  install -m 0755 "$SCRIPT_DIR/bin/$bin" "$BIN_DIR/$bin"
done

export ANIMA_CM_DIR="$CM_DEST"
export PATH="$BIN_DIR:$PATH"

if [ ! -f "$CONFIG_DIR/identity.configured" ]; then
  if [ -t 0 ]; then
    anima setup --perfil "${ANIMA_PERFIL:-limpio}"
  else
    anima setup --perfil "${ANIMA_PERFIL:-limpio}" --nombre "${ANIMA_NOMBRE:-Animalito}"
  fi
fi

echo "[install] creando venv Python…"
python3 -m venv "$CM_DEST/.venv-desktop"
"$CM_DEST/.venv-desktop/bin/pip" install -q --upgrade pip
"$CM_DEST/.venv-desktop/bin/pip" install -q -r "$CM_DEST/requirements-desktop.txt"

install_launchagent() {
  local src="$1" label="$2"
  local dst="$LAUNCH_DIR/$label.plist"
  sed \
    -e "s|__CM_DIR__|$CM_DEST|g" \
    -e "s|__LOG_DIR__|$LOG_DIR|g" \
    "$src" > "$dst"
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$dst"
  echo "  → $dst"
}

echo "[install] LaunchAgents…"
install_launchagent "$SCRIPT_DIR/launchd/com.vstcosmo.anima-organismo.plist" "com.vstcosmo.anima-organismo"
install_launchagent "$SCRIPT_DIR/launchd/com.vstcosmo.anima-watchdog.plist" "com.vstcosmo.anima-watchdog"

anima start || true

DESKTOP="$HOME/Desktop"
if [ -f "$SCRIPT_DIR/Iniciar ANIMA.command" ]; then
  cp "$SCRIPT_DIR/Iniciar ANIMA.command" "$DESKTOP/"
  chmod +x "$DESKTOP/Iniciar ANIMA.command"
  echo "  → $DESKTOP/Iniciar ANIMA.command"
fi

echo
echo "Instalación completa."
echo "  Comandos: anima status | anima open | anima restart"
echo "  Observatorio: http://127.0.0.1:7788/"
echo "  Logs: $LOG_DIR"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  echo
  echo "Añade a ~/.zshrc:"
  echo "  export PATH=\"$BIN_DIR:\$PATH\""
fi