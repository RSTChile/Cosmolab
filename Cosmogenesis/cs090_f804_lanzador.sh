#!/bin/zsh
# cs090_f804_lanzador.sh — F8-04: encadena Phantom detrás de cada layout apenas su IC está escrita.
# No espera a que terminen los 16 layouts: cada réplica arranca sola. Cap de pared por argumento.
BASE=/Users/alexis/phantom_cs073/f804_grano_n8000/N8000
CS=/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis
CAP=${1:-5400}
mkdir -p /tmp/f804log
typeset -A lanzado
pendientes=16
while (( pendientes > 0 )); do
  for g in r23_I r10_III; do
    for r in 0 1 2 3 4 5 6 7; do
      key="${g}_rep0${r}"
      [[ -n "${lanzado[$key]}" ]] && continue
      if [[ -s "$BASE/$key/meta_regla.json" && -s "$BASE/$key/cosmogenesis_ic.txt" ]]; then
        OMP_NUM_THREADS=1 nohup $CS/venv/bin/python $CS/cs090_f804_grano_n8000.py run $g $r $CAP \
          > /tmp/f804log/run_${g}_${r}.log 2>&1 &
        lanzado[$key]=1
        pendientes=$((pendientes-1))
        echo "[LANZADO] $key  (quedan $pendientes)"
      fi
    done
  done
  sleep 15
done
echo "[TODOS LANZADOS]"
wait
echo "[TODOS TERMINADOS]"
