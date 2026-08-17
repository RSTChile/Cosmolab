#!/bin/bash
# Cadena de producción F1-1: N=400 -> N=800 -> N=1600 -> resumen.
# N=200 ya se completó (F1_1_produccion_N200_resultado.json existe).
cd "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/BATERIA_FUNDAMENTOS/F1_1_forma_magnitud" || exit 1
PY="/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/venv/bin/python3"

"$PY" -u F1_1_motor.py produccion --N 400 > resultados/F1_1_stdout_N400.log 2>&1
"$PY" -u F1_1_motor.py produccion --N 800 > resultados/F1_1_stdout_N800.log 2>&1
"$PY" -u F1_1_motor.py produccion --N 1600 > resultados/F1_1_stdout_N1600.log 2>&1
"$PY" -u F1_1_motor.py resumen > resultados/F1_1_stdout_resumen.log 2>&1
echo "CADENA COMPLETA" >> resultados/F1_1_log_ejecucion.txt
