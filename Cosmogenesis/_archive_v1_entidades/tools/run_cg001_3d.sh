#!/bin/sh
# Visor 3D nativo CG001 (VisPy + PyQt6) — corre en el Mac, no en Docker.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/venv_viz"
if [ ! -x "$VENV/bin/python" ]; then
  echo "[cg001-3d] Creando venv_viz e instalando PyQt6 + pyqtgraph (OpenGL)…"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -r "$ROOT/requirements-viz.txt"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CG_LAB_URL="${CG_LAB_URL:-http://127.0.0.1:7888}"

exec "$VENV/bin/python" "$ROOT/CG001/visualization/cg001_desktop_3d.py" --live --url "$CG_LAB_URL" "$@"