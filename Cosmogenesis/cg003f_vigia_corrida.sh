#!/bin/bash
# Vigía de la corrida de fondo cg003f v1 (instrucción de Claude Science, 2-jul-2026).
# NO pelea la máquina en caliente: espera a que haya "aire" (que `import numpy` deje
# de hacer deadlock) y RECIÉN AHÍ lanza el barrido de los 3 brazos.
#   BRAZO REGLA  = λ_H > 0   |  BRAZO CONTROL = λ_H = 0  |  BRAZO AZAR = shuffle
# El main() de cg003f_planitud_exergia.py ya recorre esos tres y mide δ(Gromov),
# diámetro vs N (pendiente log-log), dimensión (emergente) y % componente gigante,
# para Dtan=2 y Dtan=3. Salida completa -> RESULTS; latido de estado -> STATUS.
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis || exit 1
PY=./venv/bin/python
RESULTS=cg003f_fondo_resultados.txt
STATUS=cg003f_fondo_STATUS.txt
CAPS="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1"

echo "[$(date '+%H:%M:%S')] vigía iniciado; esperando aire (numpy sin deadlock)…" > "$STATUS"

for i in $(seq 1 160); do            # ~160 intentos * ~105s ≈ 4.5 h de ventana
  # Sonda: importar numpy con límite de 15 s. Si deadlockea, no hay aire.
  ( eval $CAPS $PY -c "'import numpy'" ) >/dev/null 2>&1 &
  p=$!; ready=0
  for s in $(seq 1 15); do kill -0 "$p" 2>/dev/null || { ready=1; break; }; sleep 1; done

  if [ "$ready" = 1 ] && wait "$p" 2>/dev/null; then
    load=$(uptime | sed 's/.*load averages*: //')
    echo "[$(date '+%H:%M:%S')] AIRE en intento $i (load $load) -> lanzando barrido 3 brazos" >> "$STATUS"
    eval $CAPS $PY cg003f_planitud_exergia.py > "$RESULTS" 2>&1
    rc=$?
    echo "[$(date '+%H:%M:%S')] BARRIDO TERMINADO (rc=$rc). Resultados en $RESULTS" >> "$STATUS"
    exit $rc
  fi
  kill -9 "$p" 2>/dev/null
  load=$(uptime | sed 's/.*load averages*: //')
  echo "[$(date '+%H:%M:%S')] intento $i: sin aire aún (load $load)" >> "$STATUS"
  sleep 90
done

echo "[$(date '+%H:%M:%S')] RENDICIÓN: 4.5 h sin aire; no se lanzó. Revisar carga/ANIMA." >> "$STATUS"
exit 2
