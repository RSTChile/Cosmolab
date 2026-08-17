#!/bin/bash
# cs090_fase8_f806_lanzador.sh — F8-06: encadena Phantom detrás de cada condición inicial
# ============================================================================================
# QUÉ HACE Y POR QUÉ EXISTE
# -------------------------
# A N=4000 el paso caro es el LAYOUT (suma exacta O(N²), ~4-10 min por brazo según la carga de la
# máquina), y la corrida de Phantom es barata (18-57 s medidos en O3-A). Si se esperara a que los
# 24 layouts terminen para recién ahí lanzar Phantom, se perdería tiempo de pared gratis.
#
# Este lanzador revisa cada `INTERVALO` segundos qué carpetas ya tienen su condición inicial COMPLETA
# (el runner de Python exige `meta_regla.json` — que se escribe DESPUÉS del IC — y las 4002 líneas
# exactas) y corre Phantom sobre las que falten, N_PARALELO a la vez. Nunca recomputa una carpeta que
# ya tenga `cosmog_00500`.
#
# USO:  ./cs090_fase8_f806_lanzador.sh [n_esperadas] [n_paralelo] [intervalo_s]
set -u
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis

BASE=/Users/alexis/phantom_cs073/bateria_fase8_f806_n4000
ESPERADAS=${1:-24}
N_PARALELO=${2:-4}
INTERVALO=${3:-90}

while :; do
  listas=$(ls -d "$BASE"/*/ 2>/dev/null | wc -l | tr -d ' ')
  hechas=$(ls "$BASE"/*/cosmog_00500 2>/dev/null | wc -l | tr -d ' ')
  echo "[lanzador] $(date +%H:%M:%S)  carpetas=$listas  con_dump_final=$hechas / $ESPERADAS"
  if [ "$hechas" -ge "$ESPERADAS" ]; then
    echo "[lanzador] listo"; break
  fi
  # una tanda: el runner de Python filtra solo lo que está completo y salta lo ya corrido
  ls -d "$BASE"/*/ 2>/dev/null | sed 's:.*/\([^/]*\)/$:\1:' \
    | xargs -P "$N_PARALELO" -I{} python3.9 cs090_fase8_f806_correr.py {} >/dev/null 2>&1
  sleep "$INTERVALO"
done
