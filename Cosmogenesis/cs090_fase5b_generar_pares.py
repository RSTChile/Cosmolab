"""
cs090_fase5b_generar_pares.py — FASE V-B Paso 1+2: elegir PARES emparejados (Clase III + Clase I) de
`cs090_fase5_profundizar_a2b0c2_resumen.csv` y generar sus condiciones iniciales de Phantom (masa fija,
N=2000) con `cs090_fase5b_phantom_adaptador.py`.

CÓMO SE ELIGIERON LOS PARES (documentado, no arbitrario): de las 20 reglas nuevas del CSV resumen, se
excluyen r0 (clase "intermedio") y r15 (Clase II) -- quedan 18 reglas utilizables (8 Clase I, 10 Clase
III). Para cada regla Clase I se buscó, a mano, la regla Clase III con distancia más chica en sus
parámetros NO-clasificatorios (K, J, noise, meandeg, kcap — el objetivo 2 de
FASE5A_profundizar_A2B0C2_resultado_CS.md mostró que NINGUNO separa limpiamente por sí solo, con
solape total de rangos entre clases; kcap/K tienen correlación moderada -0.43/+0.45 pero no un umbral
duro). Los 3 pares con menor distancia combinada, sin reusar ninguna regla dos veces:

  PAR A: r9 (I) vs r19 (III) — K=5/5 (igual), kcap=7/7 (igual), meandeg=4.25/4.02 (Δ0.23),
         J=0.493/0.341 (Δ0.152), noise=0.244/0.326 (Δ0.082). El par MÁS cercano de los 18 (K y kcap
         coinciden exactamente, meandeg casi idéntico).
  PAR B: r1 (I) vs r17 (III) — K=5/5 (igual), meandeg=7.63/7.35 (Δ0.28), noise=0.234/0.211 (Δ0.023),
         J=0.659/0.475 (Δ0.184), kcap=6/5 (Δ1).
  PAR C: r6 (I) vs r14 (III) — kcap=5/5 (igual), J=0.632/0.567 (Δ0.065), noise=0.119/0.198 (Δ0.079),
         meandeg=5.38/5.76 (Δ0.38), K=7/8 (Δ1).

N=2000 para TODAS las reglas de los 3 pares (piso de resolución SPH válido, lección de CS073 -- ver
MISTERIO_N500_vs_N2000_CS.md/cosmogenesis-masa-fija-resultado-7ago2026 -- N<1000 no es confiable).
`seed_layout` fijo (12345, mismo valor "original" que bateria_n2000/ic_real) para las 6 reglas -- la
ÚNICA diferencia entre reglas del mismo par es el GRAFO de entrada (Clase I vs Clase III, ya
determinado por su dinámica A2-B0-C2 propia), no la realización del layout.

Escribe en `/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto/<rule_id>_<clase>/
cosmogenesis_ic.txt`. No corre Phantom (ver cs090_fase5b_correr.py). No toca ningún script congelado.
"""
from __future__ import annotations
import csv
import json
import os
import time

from cs090_fase5b_phantom_adaptador import reconstruir_regla_a2b0c2, generar_ic_masa_fija_desde_grafo

RESUMEN_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_profundizar_a2b0c2_resumen.csv"
BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto"
N_PILOTO = 2000
SEED_LAYOUT = 12345   # mismo "original" que bateria_n2000/ic_real -- único grado de libertad es el grafo

# Los 3 pares elegidos (rule_id -> seed), leídos directamente del CSV resumen (no hardcodeados a mano
# por separado -- se buscan por rule_id+origen para evitar transcribir mal un seed).
PARES = [
    ("PAR_A", "A2-B0-C2-r9", "A2-B0-C2-r19"),   # I vs III, el más cercano (K y kcap iguales)
    ("PAR_B", "A2-B0-C2-r1", "A2-B0-C2-r17"),   # I vs III
    ("PAR_C", "A2-B0-C2-r6", "A2-B0-C2-r14"),   # I vs III
]


def cargar_resumen():
    filas = {}
    with open(RESUMEN_CSV) as f:
        for row in csv.DictReader(f):
            if row["origen"] == "nueva_profundizar":
                filas[row["rule_id"]] = row
    return filas


def generar_una_regla(rule_id, row, N=N_PILOTO, seed_layout=SEED_LAYOUT):
    seed = int(row["seed"])
    clase = row["clase"]
    t0 = time.time()
    p, m = reconstruir_regla_a2b0c2(seed=seed, N=N, n_sweeps=14)
    t_reconstruir = time.time() - t0

    carpeta = f"{BASE_SALIDA}/{rule_id}_{clase}"
    os.makedirs(carpeta, exist_ok=True)
    ruta_ic = f"{carpeta}/cosmogenesis_ic.txt"

    t1 = time.time()
    info_ic = generar_ic_masa_fija_desde_grafo(m["adj_final"], N=N, seed_layout=seed_layout,
                                                ruta_salida=ruta_ic)
    t_ic = time.time() - t1

    meta = dict(rule_id=rule_id, clase=clase, seed=seed, N=N, seed_layout=seed_layout,
                K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                sim_thr_frac=p["sim_thr_frac"],
                diam_del_barrido=float(row["diam_real"]) if "diam_real" in row else None,
                n_aristas_grafo_final=m["n_aristas"], diam_grafo_final=m["diam"],
                giant_grafo_final=m["giant"], holon_grafo_final=m["holonomia"],
                grado_medio_grafo_final=2.0 * m["n_aristas"] / N,
                masa_total_ic=info_ic["masa_total"], carpeta=carpeta, ruta_ic=ruta_ic,
                t_reconstruir_grafo_s=round(t_reconstruir, 2), t_generar_ic_s=round(t_ic, 2))
    with open(f"{carpeta}/meta_regla.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    resumen = cargar_resumen()
    os.makedirs(BASE_SALIDA, exist_ok=True)
    resultados = []
    for par_nombre, rid_I, rid_III in PARES:
        for rule_id in (rid_I, rid_III):
            row = resumen[rule_id]
            assert row["clase"] in ("I", "III"), f"{rule_id}: clase inesperada {row['clase']}"
            print(f"[{par_nombre}] generando IC para {rule_id} (clase {row['clase']}, "
                  f"seed={row['seed']})...", flush=True)
            meta = generar_una_regla(rule_id, row)
            meta["par"] = par_nombre
            resultados.append(meta)
            print(f"  -> {meta['carpeta']} (n_aristas grafo final={meta['n_aristas_grafo_final']}, "
                  f"masa_total_ic={meta['masa_total_ic']:.6g}, "
                  f"t_reconstruir+ic={meta['t_reconstruir_grafo_s']+meta['t_generar_ic_s']:.2f}s)",
                  flush=True)

    ruta_resumen = f"{BASE_SALIDA}/pares_resumen.json"
    with open(ruta_resumen, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"\n[TOTAL] {len(resultados)} reglas (3 pares) -- resumen en {ruta_resumen}")
    return resultados


if __name__ == "__main__":
    main()
