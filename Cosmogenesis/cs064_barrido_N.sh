#!/bin/zsh
# CS064 — barrido de N (régimen tractable). Cada N a su CSV, checkpointeado, secuencial.
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis
PY=/Users/alexis/.venvs/vstcosmo/bin/python
export CS064_STEPS=20 CS064_WORKERS=6 CS064_REGLA=nematico
run() {  # N  PATCHES
  echo "=== $(date '+%H:%M:%S') N=$1 patches=$2 ==="
  CS064_N=$1 CS064_PATCHES=$2 CS064_OUT=cs064_N$1.csv $PY cs064_sistema_completo.py
}
run 1500 120
run 2500 100
run 3500 60
echo "=== $(date '+%H:%M:%S') BARRIDO COMPLETO ==="
