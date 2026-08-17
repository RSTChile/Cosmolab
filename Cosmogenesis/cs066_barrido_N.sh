#!/bin/zsh
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis
PY=/Users/alexis/.venvs/vstcosmo/bin/python
export CS066_STEPS=20 CS066_WORKERS=6
run() {
  echo "=== $(date '+%H:%M:%S') N=$1 patches=$2 ==="
  CS066_N=$1 CS066_PATCHES=$2 CS066_OUT=cs066_N$1.csv $PY cs066_localidad_geometrogenesis.py
}
run 1500 100
run 2500 100
run 3500 60
echo "=== $(date '+%H:%M:%S') BARRIDO CS066 COMPLETO ==="
