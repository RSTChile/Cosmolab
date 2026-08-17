#!/bin/zsh
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis
PY=/Users/alexis/.venvs/vstcosmo/bin/python
export CS065B_STEPS=20 CS065B_WORKERS=6
run() {
  echo "=== $(date '+%H:%M:%S') N=$1 patches=$2 ==="
  CS065B_N=$1 CS065B_PATCHES=$2 CS065B_OUT=cs065b_N$1.csv $PY cs065b_exclusion_ortogonalizante.py
}
run 1500 100
run 2500 100
run 3500 60
echo "=== $(date '+%H:%M:%S') BARRIDO CS065b COMPLETO ==="
