#!/bin/bash
# cs090_fase6_o3a_correr_phantom.sh -- FASE VI, tarea O3-A: corre Phantom, UNA CORRIDA POR VEZ, sobre las
# condiciones iniciales ya generadas por `cs090_fase6_o3a_convergencia_resolucion.py worker ... solo_ic`.
#
# POR QUÉ SERIAL Y NO EN PARALELO: el binario de Phantom de este proyecto está compilado con OpenMP y usa
# los 16 hilos de la máquina ("Running in openMP on 16 threads" en cualquier run.log de CS073). Correr
# varias instancias a la vez las haría pelearse por los mismos hilos y, peor, cambiaría el orden de las
# sumas en las reducciones OpenMP respecto de cómo se corrieron los puntos de N=2000 con los que se
# compara. Serial = mismas condiciones que la línea base.
#
# POR QUÉ SE PODAN LOS DUMPS INTERMEDIOS: cada corrida escribe 500 dumps (tmax=0.500, dtmax=0.001). A
# N=4000 eso son ~54 MB por corrida y el disco de esta Mac está al 99% (7.8 GiB libres al empezar la
# tarea). El análisis (`cs090_fase5b_analizar.analizar_carpeta`) sólo usa el PRIMER dump (para contar el
# gas inicial), el ÚLTIMO (masa en sumideros vs. gas) y el archivo `.sink` (κ_V, t del primer sumidero) --
# los 498 del medio no se leen nunca. Se borran DESPUÉS de que la corrida ya fue analizada y su
# `resultado_o3a.json` está escrito, nunca antes.
#
# Uso:  bash cs090_fase6_o3a_correr_phantom.sh <archivo_de_jobs> <N>
#       (archivo de jobs: una línea por regla, "rule_id seed clase")

set -u
JOBS="$1"
N="$2"
BASE="/Users/alexis/phantom_cs073/bateria_fase6_o3a_resolucion/N${N}"
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis || exit 1

while read -r RID SEED CLASE; do
  [ -z "${RID:-}" ] && continue
  CARPETA="${BASE}/${RID}_${CLASE}"
  if [ ! -f "${CARPETA}/cosmogenesis_ic.txt" ]; then
    echo "[SALTO] ${RID} ${CLASE} N=${N}: todavía no tiene condición inicial"
    continue
  fi
  if [ -f "${CARPETA}/resultado_o3a.json" ]; then
    echo "[YA] ${RID} ${CLASE} N=${N}: ya analizada"
    continue
  fi
  ./venv/bin/python cs090_fase6_o3a_convergencia_resolucion.py worker "$RID" "$SEED" "$CLASE" "$N"
  # poda de dumps intermedios (ver cabecera): se conservan el primero, el último y el .sink
  if [ -f "${CARPETA}/resultado_o3a.json" ]; then
    ULT=$(ls "${CARPETA}"/cosmog_0* 2>/dev/null | sort | tail -1)
    PRI=$(ls "${CARPETA}"/cosmog_0* 2>/dev/null | sort | head -1)
    for D in "${CARPETA}"/cosmog_0*; do
      [ "$D" = "$ULT" ] && continue
      [ "$D" = "$PRI" ] && continue
      rm -f "$D"
    done
    echo "[PODA] ${RID} ${CLASE} N=${N}: conservados $(basename "$PRI") y $(basename "$ULT")"
  fi
done < "$JOBS"
echo "FIN_PHANTOM_N${N}"
