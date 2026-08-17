#!/bin/bash
# cs090_fase6_o3d_generar_controles.sh -- FASE VI, tarea O3-D: controles SIN ESTRUCTURA para la
# pregunta 3 ("con kcap=7 dando 0 % de geometría extendida, ¿la masa acretada cae al nivel de un
# control sin estructura?").
#
# Un control es un grafo Erdős-Rényi puro (sin dinámica, sin recableo, sin kcap) pasado por EXACTAMENTE
# el mismo pipeline: layout_resortes -> dilatación estática -> turbulencia Mach=3 seed=42 -> masa fija
# 18800 en N=2000. Lo único que cambia respecto de una regla es de dónde salió el grafo.
#
# EMPAREJADOS EN ARISTAS, en los dos extremos del barrido: el número de aristas del grafo final está
# fuertemente atado a kcap (kcap=4 -> mediana 2321 aristas; kcap=7 -> mediana 4608, medido sobre las 16
# reglas de esos dos grupos ANTES de generar nada). Un control único no serviría: parecería "más denso"
# que kcap=4 y "menos denso" que kcap=7, y la comparación mezclaría densidad con estructura. Por eso van
# 3 controles a 2321 aristas y 3 a 4608 -- cada extremo del barrido tiene su propio espejo sin estructura.
#
# Espera a que termine la generación de IC de las reglas (FIN_IC_O3D) para no pelearse por los núcleos.
set -u
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

while ! grep -q FIN_IC_O3D cs090_fase6_o3d_generar_ic.log; do sleep 15; done

printf '%s\n' \
  "3000001 2321" "3000002 2321" "3000003 2321" \
  "3000011 4608" "3000012 4608" "3000013 4608" \
  | xargs -P 6 -n 2 sh -c 'python3.9 cs090_fase6_o3d_barrido_kcap.py control "$0" "$1"'
echo "FIN_CONTROLES_O3D"
