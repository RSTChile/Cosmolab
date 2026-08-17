#!/usr/bin/env bash
set -euo pipefail

PKG_NAME="anima-desktop-runtime"
VERSION="${VERSION:-0.1.0-dev}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAGE_DIR="$CM_DIR/build/${PKG_NAME}_${VERSION}_macos"
OUT_DIR="$CM_DIR/dist"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/celula_madre" "$OUT_DIR"

runtime_paths=(
  "audio"
  "campo"
  "conversacion"
  "diada"
  "docker"
  "genoma"
  "lexico_comun"
  "mcp"
  "organelos"
  "voces_r2d2"
  "web"
  "schemas"
  "CAPA_BIOLOGICA_vs_CAPA_TECNOLOGICA.md"
  "ESPECIFICACION_TECNICA_PLAZA_PRESENCIA_AUDIO_2026-07-07.md"
  "requirements.txt"
  "requirements-desktop.txt"
  "audio/audio_library_minimal.json"
  "audio/anima_audio_catalogo.py"
)

for runtime_path in "${runtime_paths[@]}"; do
  if [ -e "$CM_DIR/$runtime_path" ]; then
    rsync -a \
      --exclude '.DS_Store' \
      --exclude '__pycache__' \
      --exclude '*.pyc' \
      --exclude '*.bak' \
      --exclude '*.zip' \
      "$CM_DIR/$runtime_path" "$STAGE_DIR/celula_madre/"
  fi
done

if [ -f "$STAGE_DIR/celula_madre/campo/Célula_Madre_Funcional_001.py" ]; then
  cp "$STAGE_DIR/celula_madre/campo/Célula_Madre_Funcional_001.py" \
     "$STAGE_DIR/celula_madre/campo/Celula_Madre_Funcional_001.py"
fi

chmod +x \
  "$STAGE_DIR/celula_madre/docker/run_native_desktop.sh" \
  "$STAGE_DIR/celula_madre/docker/anima-watchdog-desktop.sh" 2>/dev/null || true

rsync -a "$SCRIPT_DIR/bin/" "$STAGE_DIR/bin/"
rsync -a "$SCRIPT_DIR/config/" "$STAGE_DIR/config/"
rsync -a "$SCRIPT_DIR/launchd/" "$STAGE_DIR/launchd/"
chmod 0755 "$STAGE_DIR/bin/"* "$STAGE_DIR/install_mac.sh" "$STAGE_DIR/Iniciar ANIMA.command" 2>/dev/null || true
cp "$SCRIPT_DIR/install_mac.sh" "$STAGE_DIR/"
cp "$SCRIPT_DIR/Iniciar ANIMA.command" "$STAGE_DIR/"
chmod 0755 "$STAGE_DIR/install_mac.sh" "$STAGE_DIR/Iniciar ANIMA.command"
cp "$SCRIPT_DIR/README.md" "$STAGE_DIR/"

OUT_TGZ="$OUT_DIR/${PKG_NAME}_${VERSION}_macos.tar.gz"
tar -C "$STAGE_DIR/.." -czf "$OUT_TGZ" "$(basename "$STAGE_DIR")"

echo "[build] staging: $STAGE_DIR"
du -sh "$STAGE_DIR"
echo "[build] listo: $OUT_TGZ"
echo
echo "Instalar en Mac:"
echo "  tar -xzf $OUT_TGZ"
echo "  cd ${PKG_NAME}_${VERSION}_macos && ./install_mac.sh"