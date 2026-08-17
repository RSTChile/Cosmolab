#!/usr/bin/env bash
set -euo pipefail

PKG_NAME="anima-desktop-runtime"
VERSION="${VERSION:-0.1.0-dev}"
ARCH="${ARCH:-amd64}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$CM_DIR/build/${PKG_NAME}_${VERSION}_${ARCH}"
OUT_DIR="$CM_DIR/dist"

rm -rf "$BUILD_DIR"
mkdir -p \
  "$BUILD_DIR/DEBIAN" \
  "$BUILD_DIR/opt/anima/celula_madre" \
  "$BUILD_DIR/etc/anima" \
  "$BUILD_DIR/usr/bin" \
  "$BUILD_DIR/usr/share/anima-desktop-runtime/config" \
  "$BUILD_DIR/usr/share/anima-desktop-runtime/systemd-user" \
  "$BUILD_DIR/usr/share/doc/anima-desktop-runtime" \
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
  "schemas"
  "CAPA_BIOLOGICA_vs_CAPA_TECNOLOGICA.md"
  "ESPECIFICACION_TECNICA_PLAZA_PRESENCIA_AUDIO_2026-07-07.md"
  "requirements.txt"
  "requirements-desktop.txt"
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

rsync -a "$SCRIPT_DIR/config/" "$BUILD_DIR/usr/share/anima-desktop-runtime/config/"
rsync -a "$SCRIPT_DIR/systemd-user/" "$BUILD_DIR/usr/share/anima-desktop-runtime/systemd-user/"
install -m 0644 "$SCRIPT_DIR/README.md" "$BUILD_DIR/usr/share/doc/anima-desktop-runtime/README.md"
if [ -f "$SCRIPT_DIR/changelog" ]; then
  gzip -9c "$SCRIPT_DIR/changelog" > "$BUILD_DIR/usr/share/doc/anima-desktop-runtime/changelog.gz"
fi

chmod +x \
  "$BUILD_DIR/opt/anima/celula_madre/docker/run_native_desktop.sh" \
  "$BUILD_DIR/opt/anima/celula_madre/docker/anima-watchdog-desktop.sh" 2>/dev/null || true

if [ -f "$BUILD_DIR/opt/anima/celula_madre/campo/Célula_Madre_Funcional_001.py" ]; then
  cp "$BUILD_DIR/opt/anima/celula_madre/campo/Célula_Madre_Funcional_001.py" \
     "$BUILD_DIR/opt/anima/celula_madre/campo/Celula_Madre_Funcional_001.py"
fi

echo "[build] staging: $BUILD_DIR"
du -sh "$BUILD_DIR"

TMP_DIR="$CM_DIR/build/${PKG_NAME}_${VERSION}_${ARCH}.debparts"
OUT_DEB="$OUT_DIR/${PKG_NAME}_${VERSION}_${ARCH}.deb"
rm -rf "$TMP_DIR" "$OUT_DEB"
mkdir -p "$TMP_DIR"
printf '2.0\n' > "$TMP_DIR/debian-binary"
(
  cd "$BUILD_DIR/DEBIAN"
  COPYFILE_DISABLE=1 tar --format ustar --uid 0 --gid 0 --uname root --gname root -czf "$TMP_DIR/control.tar.gz" .
)
(
  cd "$BUILD_DIR"
  COPYFILE_DISABLE=1 tar --format ustar --uid 0 --gid 0 --uname root --gname root --exclude './DEBIAN' -czf "$TMP_DIR/data.tar.gz" .
)
python3 - "$TMP_DIR" "$OUT_DEB" <<'PY'
import sys
from pathlib import Path

parts = Path(sys.argv[1])
out = Path(sys.argv[2])

def header(name: str, data: bytes) -> bytes:
    n = (name + "/").encode("ascii")
    if len(n) > 16:
        raise ValueError(name)
    return b"".join([
        n.ljust(16, b" "),
        b"0".ljust(12, b" "),
        b"0".ljust(6, b" "),
        b"0".ljust(6, b" "),
        b"100644".ljust(8, b" "),
        str(len(data)).encode("ascii").ljust(10, b" "),
        b"`\n",
    ])

with out.open("wb") as f:
    f.write(b"!<arch>\n")
    for name in ("debian-binary", "control.tar.gz", "data.tar.gz"):
        data = (parts / name).read_bytes()
        f.write(header(name, data))
        f.write(data)
        if len(data) % 2:
            f.write(b"\n")
PY

echo "[build] listo: $OUT_DEB"