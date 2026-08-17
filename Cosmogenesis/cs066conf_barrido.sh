#!/bin/zsh
# CS066 confirmatorio Nivel 1 — malla FIJA k_local × N, ≥40 parches/celda (DISENO_CS066_confirmatorio_Nivel1_CS.md)
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis
PY=/Users/alexis/.venvs/vstcosmo/bin/python
export CS066_STEPS=20 CS066_WORKERS=6
cell() {  # k N patches
  echo "=== $(date '+%H:%M:%S') CELDA k_local=$1 N=$2 patches=$3 ==="
  CS066_KFIX=$1 CS066_N=$2 CS066_PATCHES=$3 CS066_OUT=cs066conf_k$1_N$2.csv $PY cs066_localidad_geometrogenesis.py
}
# régimen decisivo (localidad fuerte) PRIMERO, luego el laxo — así el exponente que importa queda temprano
for k in 3 4 5 6 8 10; do
  for N in 1500 2500 3500 5000; do
    cell $k $N 40
  done
done
echo "=== $(date '+%H:%M:%S') CONFIRMATORIO CS066 COMPLETO ==="
