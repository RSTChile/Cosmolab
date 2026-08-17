#!/bin/bash
# cs090_fase7_f701_generar_ic.sh — generación de las condiciones iniciales de F7-01 EN PARALELO.
#
# Por qué en paralelo: el paso caro del worker es `layout_resortes` (Fruchterman-Reingold O(N^2), 100
# iteraciones, monohilo en Python) — ~85-95 % del costo por regla. Cada worker escribe en SU PROPIA
# carpeta y deriva todo su azar de su propia semilla, así que el resultado es bit a bit idéntico al
# que daría la corrida serial. Phantom, en cambio, va SERIAL (está compilado con OpenMP y usa los 16
# hilos; varias instancias a la vez cambiarían el orden de las reducciones respecto de la línea base).
#
# Uso: bash cs090_fase7_f701_generar_ic.sh [n_paralelo]
set -u
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis
NPAR="${1:-8}"
TRABAJOS=cs090_fase7_f701_trabajos.txt
LOG=cs090_fase7_f701_ic.log
: > "$LOG"
echo "[ic] $(wc -l < $TRABAJOS) trabajos, $NPAR en paralelo — $(date)" | tee -a "$LOG"
while read -r rid seed kcap M Mobj; do
  [ -z "$rid" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do sleep 2; done
  ( python3.9 cs090_fase7_f701_factorial.py worker "$rid" "$seed" "$kcap" "$M" >> "$LOG" 2>&1 ) &
done < "$TRABAJOS"
wait
echo "[ic] FIN — $(date)" | tee -a "$LOG"
