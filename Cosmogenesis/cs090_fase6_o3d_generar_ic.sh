#!/bin/bash
# cs090_fase6_o3d_generar_ic.sh -- FASE VI, tarea O3-D: genera EN PARALELO las condiciones iniciales de
# Phantom de las 32 reglas del barrido de kcap (una por línea de cs090_fase6_o3d_trabajos.txt).
#
# POR QUÉ EN PARALELO ACÁ Y NO EN PHANTOM: este paso es Python monohilo (el 95 % del costo es
# `layout_resortes`, Fruchterman-Reingold O(N²) por 100 iteraciones ~ 130 s por regla) y cada worker
# escribe en su propia carpeta, sin ningún estado compartido -- el resultado es bit a bit el mismo se
# corra solo o acompañado, porque cada regla deriva su rng de su propia semilla. Phantom, en cambio, va
# SERIAL (ver cabecera de cs090_fase6_o3d_barrido_kcap.py): está compilado con OpenMP sobre los 16 hilos
# y correr instancias simultáneas cambiaría el orden de las reducciones respecto de la línea base.
#
# Se fijan las variables de hilos de BLAS a 1 para que los 6 workers no se peleen por los mismos núcleos.
#
# Uso: bash cs090_fase6_o3d_generar_ic.sh [n_paralelo]
set -u
NP="${1:-6}"
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

awk '{print $1, $2, $3}' cs090_fase6_o3d_trabajos.txt \
  | xargs -P "$NP" -n 3 sh -c 'python3.9 cs090_fase6_o3d_barrido_kcap.py worker "$0" "$1" "$2"'
echo "FIN_IC_O3D"
