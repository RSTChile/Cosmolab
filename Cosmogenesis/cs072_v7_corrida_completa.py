"""
CS072 v7.2 -- CORRIDA COMPLETA, UN SOLO PASE (a pedido explícito del director: no fragmentar el experimento,
correr TODO junto y presentar UN resultado consolidado). Incluye, en orden:
  1) Los 4 brazos de control (A/B/C/D) con las 23 piezas.
  2) Invariancia dura (permutación + atol 1e-9) sobre el brazo D.
  3) Auditoría de las 23: apagar cada pieza una por una y confirmar que el resultado CAMBIA.
  4) Validación de que cada hidrógeno contado es un protón (uud) discreto + electrón.
  5) Barrido de N (3 escalas) sobre los 4 brazos.
  6) Barrido de sensibilidad de los 3 parámetros copiados del toy (alpha/tasa_exp/amplitud).
Un solo log real con todo, guardado en cs072_v7_corrida_completa_log.txt.
"""
import sys, time, json
sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
import cs072_fold_completo as m

t0 = time.time()
resultados = {}

print("=" * 100, flush=True)
print("CS072 v7.2 -- CORRIDA COMPLETA (30q/21aq/10e/7p base, pasos=300)", flush=True)
print("=" * 100, flush=True)

print("\n--- (1) LOS 4 BRAZOS DE CONTROL ---", flush=True)
r4 = m.test_cuatro_brazos(30, 21, 10, 7, pasos=300)
resultados["cuatro_brazos"] = r4
print(f"  (t={(time.time()-t0)/60:.2f} min)", flush=True)

print("\n--- (2) INVARIANCIA DURA (brazo D) ---", flush=True)
rinv = m.test_invariancia_dura(30, 21, 10, 7, pasos=300)
resultados["invariancia_dura"] = rinv
print(f"  (t={(time.time()-t0)/60:.2f} min)", flush=True)

print("\n--- (3) AUDITORÍA DE LAS 23 PIEZAS (apagar una a una, brazo D) ---", flush=True)
raud = m.auditoria_piezas_activas(30, 21, 10, 7, pasos=300)
resultados["auditoria_piezas"] = raud
print(f"  (t={(time.time()-t0)/60:.2f} min)", flush=True)

print("\n--- (4) VALIDACIÓN DE HIDRÓGENO DISCRETO (brazo D) ---", flush=True)
rD = m.corre_proceso_unico(30, 21, 10, 7, pasos=300, homogeneo=False, expansion=True)
atD = m.cuenta_bariones_e_hidrogeno(rD)
rvalH = m.valida_hidrogeno_discreto(rD, atD)
for det in rvalH["detalle"]:
    print(f"  H: trio={det['trio']} colores={det['colores']} cargas_quarks={det['cargas_quarks']} "
          f"carga_e={det['carga_electron']} -- {'VÁLIDO (uud+e discreto)' if det['valido_discreto'] else '*** INVÁLIDO ***'}",
          flush=True)
print(f"  TODOS VÁLIDOS: {rvalH['todos_validos']}", flush=True)
resultados["validacion_hidrogeno"] = rvalH
print(f"  (t={(time.time()-t0)/60:.2f} min)", flush=True)

print("\n--- (5) BARRIDO DE N (3 escalas, 4 brazos, pasos=300) ---", flush=True)
escalas = [(30, 21, 10, 7), (60, 42, 20, 14), (120, 84, 40, 28)]
rb = m.barrido_N_diametro(escalas, pasos=300)
resultados["barrido_N"] = rb
print(f"  (t={(time.time()-t0)/60:.2f} min)", flush=True)

print("\n--- (6) BARRIDO DE SENSIBILIDAD DE PARÁMETROS (27 combinaciones, pasos=150) ---", flush=True)
rsens = m.barrido_sensibilidad_parametros_termicos(30, 21, 10, 7, pasos=150)
resultados["sensibilidad_parametros"] = dict(robusto=rsens["robusto"], n_empates_con_c=rsens["n_empates_con_c"])
print(f"  (t={(time.time()-t0)/60:.2f} min)", flush=True)

print("\n" + "=" * 100, flush=True)
print(f"TIEMPO TOTAL: {(time.time()-t0)/60:.2f} min", flush=True)
print("=" * 100, flush=True)

with open("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs072_v7_corrida_completa_resultados.json", "w") as f:
    json.dump(resultados, f, indent=2, default=str)
