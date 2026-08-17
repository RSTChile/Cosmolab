#!/bin/zsh
# CS065 — barrido de N (exclusión de Pauli). Cada N a su CSV, checkpointeado, secuencial.
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis
PY=/Users/alexis/.venvs/vstcosmo/bin/python
export CS065_STEPS=20 CS065_WORKERS=6
run() {
  echo "=== $(date '+%H:%M:%S') N=$1 patches=$2 ==="
  CS065_N=$1 CS065_PATCHES=$2 CS065_OUT=cs065_N$1.csv $PY cs065_exclusion_pauli.py
}
run 1500 100
run 2500 100
run 3500 60
echo "=== $(date '+%H:%M:%S') BARRIDO CS065 COMPLETO ==="
