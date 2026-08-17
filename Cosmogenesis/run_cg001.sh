#!/usr/bin/env bash
# CG001 campo — atajos de ejecucion (Omega_posible, sin entidades v1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if [[ -d "$ROOT/venv" ]]; then
  PY="$ROOT/venv/bin/python3"
fi

cmd="${1:-demo}"
case "$cmd" in
  demo)
    "$PY" cg001_field.py
    ;;
  grueso)
    shift || true
    "$PY" cg001_barrido_grueso.py "$@"
    ;;
  fino)
    shift || true
    "$PY" cg001_barrido_fino.py "$@"
    ;;
  grueso-quick)
    "$PY" cg001_barrido_grueso.py --quick
    ;;
  fino-quick)
    "$PY" cg001_barrido_fino.py --quick
    ;;
  localizacion)
    shift || true
    "$PY" cg001_test_localizacion.py "$@"
    ;;
  localizacion-compare)
    "$PY" cg001_test_localizacion.py --compare
    ;;
  compuerta)
    shift || true
    "$PY" cg001_test_localizacion.py --compare "$@"
    ;;
  compuerta-quick)
    "$PY" cg001_test_localizacion.py --ruidos 0.007,0.001 --semillas 1-3 --pasos 100
    ;;
  causalidad)
    shift || true
    "$PY" cg001_test_causalidad.py "$@"
    ;;
  causalidad-quick)
    "$PY" cg001_test_causalidad.py --quick
    ;;
  protocolo)
    bash "$ROOT/run_protocolo_completo.sh"
    ;;
  *)
    echo "Uso: $0 {demo|grueso|fino|grueso-quick|fino-quick|compuerta|causalidad|protocolo}"
    exit 1
    ;;
esac