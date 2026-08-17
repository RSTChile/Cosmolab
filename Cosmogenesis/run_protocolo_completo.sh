#!/usr/bin/env bash
# CG001 — protocolo completo (solo datos, sin veredictos)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
LOGROOT="$ROOT/logs/protocolo_completo_$TS"
mkdir -p "$LOGROOT"

run_step() {
  local name="$1"
  shift
  echo ""
  echo "========== $name =========="
  echo "$(date -Iseconds) $name" | tee "$LOGROOT/00_indice.log"
  ./run_cg001.sh "$@" 2>&1 | tee "$LOGROOT/${name}.log"
}

echo "Protocolo completo CG001 — $TS" | tee "$LOGROOT/00_indice.log"

# 1. Causalidad RUIDO intermedio (30 semillas)
run_step "01_causalidad_ruido_intermedio" causalidad \
  --ruidos "0.074,0.02,0.007,0.003" --tag ruido_intermedio

# 2. Causalidad ε escalado en densidad máxima
run_step "02_causalidad_eps_escalado" causalidad \
  --ruido 1.0 --eps-list "0.05,0.5,5.0" --tag eps_escalado

# 3. Compuerta ABC producción
run_step "03_compuerta_production" compuerta --production

# 4. Barrido grueso producción
run_step "04_barrido_grueso_production" grueso --production

# 5. Barrido fino producción
run_step "05_barrido_fino_production" fino

echo ""
echo "Protocolo completo — logs en $LOGROOT"
echo "$(date -Iseconds) FIN" >> "$LOGROOT/00_indice.log"