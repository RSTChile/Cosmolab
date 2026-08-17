# Script GUARDADO y reproducible de las dos corridas grandes de CS073 (antes sólo existían como
# invocaciones sueltas de `python -c`, sin quedar en el repo -- ver ARQUITECTURA_EXPERIMENTO_CS.md,
# hallazgo del 19-jul: los resultados vivían en /tmp, no en el proyecto). Correr con:
#   PYTHONPATH=. venv/bin/python verificar_cs073_holistico.py
# Reproduce EXACTAMENTE los parámetros de cs073_holistico_real_vs_null_f5_output.txt y
# cs073_control_positivo_output.txt (ya en el repo, resultados ya obtenidos el 19-jul).
import json
from cs073_cierre_holistico import correr_real_vs_null, correr_control_positivo

print("=== 1) REAL vs NULL (f=5, 250 átomos H reales, masa pesada por #23) ===")
r1 = correr_real_vs_null(nq=1500, naq=1050, ne=500, npos=350, pasos_basal=150,
                          n_null=8, n_pasos_estructura=60)
print(json.dumps({k: v for k, v in r1.items() if k != "real_detalle"}, indent=2, default=str))

print("\n=== 2) CONTROL POSITIVO (f=5, masa real SIN pesar por #23, 120 pasos) ===")
r2 = correr_control_positivo(nq=1500, naq=1050, ne=500, npos=350, pasos_basal=150,
                              n_pasos_estructura=120)
print(json.dumps({k: v for k, v in r2.items() if k != "detalle"}, indent=2, default=str))
