#!/usr/bin/env bash
set -euo pipefail

PI_HOST="${PI_HOST:-ubuntu@192.168.86.36}"
PKG_NAME="anima-pi-runtime"
VERSION="${VERSION:-0.2.8-dev}"
ARCH="${ARCH:-arm64}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$CM_DIR/build/${PKG_NAME}_${VERSION}_${ARCH}"
OUT_DIR="$CM_DIR/dist"
REMOTE_DIR="/tmp/${PKG_NAME}-build"

rm -rf "$BUILD_DIR"
mkdir -p \
  "$BUILD_DIR/DEBIAN" \
  "$BUILD_DIR/opt/anima/celula_madre" \
  "$BUILD_DIR/etc/anima" \
  "$BUILD_DIR/usr/bin" \
  "$BUILD_DIR/usr/share/anima-pi-runtime/config" \
  "$BUILD_DIR/usr/share/anima-pi-runtime/systemd-user" \
  "$BUILD_DIR/usr/share/doc/anima-pi-runtime" \
  "$BUILD_DIR/var/lib/anima"
mkdir -p "$OUT_DIR"

rsync -a "$SCRIPT_DIR/DEBIAN/" "$BUILD_DIR/DEBIAN/"
sed -i.bak "s/^Version:.*/Version: ${VERSION}/" "$BUILD_DIR/DEBIAN/control"
rm -f "$BUILD_DIR/DEBIAN/control.bak"
chmod 0755 "$BUILD_DIR/DEBIAN/postinst" "$BUILD_DIR/DEBIAN/prerm" "$BUILD_DIR/DEBIAN/postrm"

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
  "arduino"
  "schemas"
  "CAPA_BIOLOGICA_vs_CAPA_TECNOLOGICA.md"
  "ESPECIFICACION_TECNICA_PLAZA_PRESENCIA_AUDIO_2026-07-07.md"
  "requirements.txt"
  "requirements-pi.txt"
)

for runtime_path in "${runtime_paths[@]}"; do
  if [ -e "$CM_DIR/$runtime_path" ]; then
    rsync -a \
      --exclude '.DS_Store' \
      --exclude '__pycache__' \
      --exclude '*.pyc' \
      --exclude '*.bak' \
      --exclude '*.zip' \
      "$CM_DIR/$runtime_path" "$BUILD_DIR/opt/anima/celula_madre/"
  fi
done

rsync -a "$SCRIPT_DIR/bin/" "$BUILD_DIR/usr/bin/"
chmod 0755 "$BUILD_DIR/usr/bin/anima" "$BUILD_DIR/usr/bin/anima-status" "$BUILD_DIR/usr/bin/anima-config"

rsync -a "$SCRIPT_DIR/config/" "$BUILD_DIR/usr/share/anima-pi-runtime/config/"
rsync -a "$SCRIPT_DIR/systemd-user/" "$BUILD_DIR/usr/share/anima-pi-runtime/systemd-user/"
install -m 0644 "$SCRIPT_DIR/README.md" "$BUILD_DIR/usr/share/doc/anima-pi-runtime/README.md"
if [ -f "$SCRIPT_DIR/changelog" ]; then
  gzip -9c "$SCRIPT_DIR/changelog" > "$BUILD_DIR/usr/share/doc/anima-pi-runtime/changelog.gz"
fi

find "$BUILD_DIR/opt/anima/celula_madre/docker" -type f -name '*.sh' -exec chmod 0755 {} + 2>/dev/null || true
chmod 0755 "$BUILD_DIR/opt/anima/celula_madre/docker/run_native_pi.sh" 2>/dev/null || true
chmod 0755 "$BUILD_DIR/opt/anima/celula_madre/docker/anima-watchdog.sh" 2>/dev/null || true
chmod 0755 "$BUILD_DIR/opt/anima/celula_madre/docker/abrir-cabeza-e-fb1.sh" 2>/dev/null || true
chmod 0755 "$BUILD_DIR/opt/anima/celula_madre/docker/ejecutar-cabeza-e-fb1.sh" 2>/dev/null || true
chmod 0755 "$BUILD_DIR/opt/anima/celula_madre/docker/ejecutar-cabeza-e-fb1-vnc.sh" 2>/dev/null || true
chmod 0755 "$BUILD_DIR/opt/anima/celula_madre/docker/configurar-cabeza-e-fb1.sh" 2>/dev/null || true
if [ -f "$BUILD_DIR/opt/anima/celula_madre/campo/Célula_Madre_Funcional_001.py" ]; then
  cp "$BUILD_DIR/opt/anima/celula_madre/campo/Célula_Madre_Funcional_001.py" \
     "$BUILD_DIR/opt/anima/celula_madre/campo/Celula_Madre_Funcional_001.py"
fi

echo "[build] staging: $BUILD_DIR"
du -sh "$BUILD_DIR"

ssh "$PI_HOST" "rm -rf '$REMOTE_DIR' && mkdir -p '$REMOTE_DIR'"
rsync -a --delete "$BUILD_DIR/" "$PI_HOST:$REMOTE_DIR/pkg/"
ssh "$PI_HOST" "dpkg-deb --root-owner-group --build '$REMOTE_DIR/pkg' '$REMOTE_DIR/${PKG_NAME}_${VERSION}_${ARCH}.deb'"
rsync -a "$PI_HOST:$REMOTE_DIR/${PKG_NAME}_${VERSION}_${ARCH}.deb" "$OUT_DIR/"

echo "[build] listo: $OUT_DIR/${PKG_NAME}_${VERSION}_${ARCH}.deb"
