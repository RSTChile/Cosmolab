"""
cs090_fase5b_generar_pares_v2.py — FASE V-B, escalado de la bateria de pares (10-ago-2026).

Continua cs090_fase5b_generar_pares.py (NO se edita ese archivo). Piloto anterior emparejo por
"distancia combinada minima" en (K,J,noise,meandeg,kcap); esta version aplica el criterio MAS
ESTRICTO que recomendo el propio informe del piloto (§6): emparejar SOLO reglas Clase I y Clase III
con K Y kcap EXACTAMENTE IGUALES, para no confundir "efecto de clase" con parametros residuales.

Paso 1 -- revisar las 18 reglas ya clasificadas de cs090_fase5_profundizar_a2b0c2_resumen.csv (origen
"nueva_profundizar", excluyendo r0 intermedio y r15 Clase II) por combinacion exacta (K,kcap):
  encontrado por inspeccion (ver informe): (K=5,kcap=7): r9(I), r12(I), r19(III) -- ya usado r9 vs r19
  en el piloto; r12 vs r19 es un par NUEVO con match exacto no usado antes.
                (K=7,kcap=5): r6(I) [ya usado en piloto contra r14, match solo en kcap] vs r2(III) --
  r6 vs r2 es un par NUEVO con match EXACTO en K y kcap (mejor que el emparejamiento r6-r14 del piloto).
  Ningun otro par exacto existe entre las 18 filas ya clasificadas (verificado exhaustivamente).

Paso 2 -- como solo salen 2 pares nuevos de las reglas YA clasificadas, se generan reglas ADICIONALES
con el generador congelado (`cs090_fase5_generador.generar_reglas_clase`, mismos ejes A2-B0-C2, mismo
metodo P1-P5), se corren con el motor congelado (`cs090_fase5_motor.correr_regla_coarse`, MISMOS
parametros de ejecucion que profundizar: N=2000, n_sweeps=14, escalas_b=(1,2,4,8,16),
n_seeds_null_topo=3) y se clasifican con el clasificador congelado
(`cs090_fase5_clasificador.clasificar_regla`) -- exactamente el mismo pipeline de
cs090_fase5_profundizar_a2b0c2.py, con un SEED_BASE nuevo para no repetir ninguna regla ya usada. Se
agrupan por (K,kcap) y se buscan pares I/III con match exacto entre las reglas nuevas.

Ningun script congelado se modifica (cs090_fase5_generador.py, cs090_fase5_motor.py,
cs090_fase5_clasificador.py, cs090_fase5b_phantom_adaptador.py -- todos solo se importan). No corre
Phantom (eso es un paso aparte, reusando cs090_fase5b_correr.main sobre las carpetas nuevas). No
declara cierre ni veredicto.
"""
from __future__ import annotations
import csv
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")

import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
from cs090_fase5_clasificador import clasificar_regla
from cs090_fase5b_phantom_adaptador import reconstruir_regla_a2b0c2, generar_ic_masa_fija_desde_grafo

RESUMEN_CSV_V1 = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_profundizar_a2b0c2_resumen.csv"
BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v2"
N_PILOTO = 2000
SEED_LAYOUT = 12345

EJE_A, EJE_B, EJE_C = "A2", "B0", "C2"
SEED_BASE_NUEVAS = 371828   # distinto de 271828 (profundizar) y de cualquier seed ya usado -- nuevas reglas
N_CANDIDATAS_OBJETIVO = 40  # reglas ADICIONALES a generar y clasificar (barato, ~1.8s/regla en el motor)
N_SWEEPS = 14
ESCALAS_B = (1, 2, 4, 8, 16)
N_SEEDS_NULL_TOPO = 3

RUTA_CANDIDATAS_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_candidatas_v2.csv"
RUTA_PARES_JSON = f"{BASE_SALIDA}/pares_v2_resumen.json"


# =====================================================================================
# Paso 1 -- pares exactos ya disponibles entre las 18 reglas clasificadas del piloto v1
# =====================================================================================
def cargar_v1():
    filas = {}
    with open(RESUMEN_CSV_V1) as f:
        for row in csv.DictReader(f):
            if row["origen"] == "nueva_profundizar" and row["clase"] in ("I", "III"):
                row["K"] = int(row["K"]); row["kcap"] = int(row["kcap"])
                filas[row["rule_id"]] = row
    return filas


def pares_exactos_v1(filas):
    """Empareja por (K,kcap) EXACTO entre las reglas ya clasificadas de v1, evitando duplicar pares
    ya corridos en el piloto (r9-r19, r1-r17, r6-r14) -- se listan explicitamente para no repetir
    trabajo pero SI se permite reusar una regla de un lado (p.ej. r19 o r6) en un par NUEVO, porque
    el objetivo es sumar observaciones independientes de pares con match exacto, no evitar toda
    reutilizacion de nodo."""
    ya_pareadas_piloto = {("A2-B0-C2-r9", "A2-B0-C2-r19"), ("A2-B0-C2-r1", "A2-B0-C2-r17"),
                           ("A2-B0-C2-r6", "A2-B0-C2-r14")}
    por_bucket = defaultdict(lambda: {"I": [], "III": []})
    for rid, row in filas.items():
        por_bucket[(row["K"], row["kcap"])][row["clase"]].append(rid)

    pares = []
    for (K, kcap), grupo in por_bucket.items():
        for rid_i in grupo["I"]:
            for rid_iii in grupo["III"]:
                par = (rid_i, rid_iii)
                if par in ya_pareadas_piloto or (par[1], par[0]) in ya_pareadas_piloto:
                    continue
                pares.append(dict(rid_I=rid_i, rid_III=rid_iii, K=K, kcap=kcap, origen="v1_exacto"))
    return pares


# =====================================================================================
# Paso 2 -- generar reglas NUEVAS candidatas, clasificarlas, buscar pares exactos entre ellas
# =====================================================================================
def generar_y_clasificar_candidatas():
    print(f"Generando {N_CANDIDATAS_OBJETIVO} reglas candidatas NUEVAS ({EJE_A}-{EJE_B}-{EJE_C}, "
          f"seed_base={SEED_BASE_NUEVAS})...", flush=True)
    admitidas, descartadas = GEN.generar_reglas_clase(
        EJE_A, EJE_B, EJE_C, n_reglas=N_CANDIDATAS_OBJETIVO, seed_base=SEED_BASE_NUEVAS, max_intentos=150)
    print(f"  admitidas={len(admitidas)} descartadas(filtro P1-P5)={len(descartadas)}", flush=True)

    filas_resumen = []
    t0 = time.time()
    for i, p in enumerate(admitidas):
        try:
            filas_regla = MOT.correr_regla_coarse(p, N=N_PILOTO, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                                    n_seeds_null_topo=N_SEEDS_NULL_TOPO)
        except Exception as e:
            print(f"  {p['rule_id']}: FALLO DEL MOTOR (no se toca) {type(e).__name__}: {e}", flush=True)
            continue
        r = clasificar_regla(filas_regla)
        fila = dict(rule_id=p["rule_id"], clase=r["clase"], seed=p["seed"], K=p["K"], J=p["J"],
                    noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                    sim_thr_frac=p["sim_thr_frac"], pendiente=r["pendiente_real"], z_agg=r["z_agg"],
                    holon_ratio=r["holon_ratio"])
        filas_resumen.append(fila)
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(admitidas)} clasificadas ({time.time()-t0:.0f}s)", flush=True)

    with open(RUTA_CANDIDATAS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_resumen[0].keys()))
        w.writeheader()
        w.writerows(filas_resumen)
    print(f"[candidatas nuevas] {len(filas_resumen)} clasificadas en {time.time()-t0:.0f}s -> "
          f"{RUTA_CANDIDATAS_CSV}", flush=True)

    cnt = defaultdict(int)
    for f in filas_resumen:
        cnt[f["clase"]] += 1
    print(f"  distribucion de clases entre candidatas nuevas: {dict(cnt)}", flush=True)
    return filas_resumen


def pares_exactos_nuevas(filas_resumen):
    por_bucket = defaultdict(lambda: {"I": [], "III": []})
    for f in filas_resumen:
        if f["clase"] in ("I", "III"):
            por_bucket[(f["K"], f["kcap"])][f["clase"]].append(f)

    pares = []
    for (K, kcap), grupo in por_bucket.items():
        n = min(len(grupo["I"]), len(grupo["III"]))
        for k in range(n):
            fi, fiii = grupo["I"][k], grupo["III"][k]
            pares.append(dict(rid_I=fi["rule_id"], rid_III=fiii["rule_id"], K=K, kcap=kcap,
                               origen="v2_generado_nuevo",
                               seed_I=fi["seed"], seed_III=fiii["seed"]))
    return pares


# =====================================================================================
# Paso 3 -- generar condiciones iniciales de Phantom para los pares elegidos
# =====================================================================================
def generar_ic_para_regla(rule_id, seed, clase, N=N_PILOTO, seed_layout=SEED_LAYOUT):
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
                sim_thr_frac=p["sim_thr_frac"], n_aristas_grafo_final=m["n_aristas"],
                diam_grafo_final=m["diam"], giant_grafo_final=m["giant"],
                holon_grafo_final=m["holonomia"], grado_medio_grafo_final=2.0 * m["n_aristas"] / N,
                masa_total_ic=info_ic["masa_total"], carpeta=carpeta, ruta_ic=ruta_ic,
                t_reconstruir_grafo_s=round(t_reconstruir, 2), t_generar_ic_s=round(t_ic, 2))
    with open(f"{carpeta}/meta_regla.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main(n_pares_objetivo=10):
    filas_v1 = cargar_v1()
    pares_v1 = pares_exactos_v1(filas_v1)
    print(f"\nPares exactos NUEVOS entre reglas YA clasificadas (v1): {len(pares_v1)}")
    for par in pares_v1:
        print(f"  {par['rid_I']} (I) vs {par['rid_III']} (III) -- K={par['K']} kcap={par['kcap']}")

    faltan = max(0, n_pares_objetivo - len(pares_v1))
    pares_v2 = []
    filas_nuevas = []
    if faltan > 0:
        filas_nuevas = generar_y_clasificar_candidatas()
        pares_v2 = pares_exactos_nuevas(filas_nuevas)
        print(f"\nPares exactos entre candidatas NUEVAS (v2): {len(pares_v2)}")
        for par in pares_v2:
            print(f"  {par['rid_I']} (I) vs {par['rid_III']} (III) -- K={par['K']} kcap={par['kcap']}")

    todos_pares = (pares_v1 + pares_v2)[:n_pares_objetivo]
    print(f"\n[TOTAL PARES SELECCIONADOS] {len(todos_pares)} (objetivo={n_pares_objetivo})")

    # -------------------- generar condiciones iniciales de Phantom para cada par --------------------
    os.makedirs(BASE_SALIDA, exist_ok=True)
    resultados = []
    seeds_conocidos = {rid: int(row["seed"]) for rid, row in filas_v1.items()}
    seeds_conocidos.update({f["rule_id"]: int(f["seed"]) for f in filas_nuevas})

    for par in todos_pares:
        for rule_id, clase in ((par["rid_I"], "I"), (par["rid_III"], "III")):
            carpeta_esperada = f"{BASE_SALIDA}/{rule_id}_{clase}"
            if os.path.exists(f"{carpeta_esperada}/cosmogenesis_ic.txt"):
                print(f"[{rule_id}] IC ya existe -- se salta", flush=True)
                continue
            # si la regla es de las 6 YA generadas en el piloto v1 (r9,r19,r1,r17,r6,r14), reusar esa
            # carpeta/IC directamente en vez de regenerar -- ahorra ~85s de layout_resortes por regla.
            carpeta_piloto_v1 = f"/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto/{rule_id}_{clase}"
            if os.path.exists(f"{carpeta_piloto_v1}/cosmog_00500"):
                print(f"[{rule_id}] YA CORRIDO en el piloto v1 -- se reusa {carpeta_piloto_v1} "
                      f"(no se regenera ni se recorre)", flush=True)
                continue
            seed = seeds_conocidos[rule_id]
            print(f"[{rule_id}] generando IC (clase={clase}, seed={seed})...", flush=True)
            meta = generar_ic_para_regla(rule_id, seed, clase)
            meta["par"] = f"{par['rid_I']}_vs_{par['rid_III']}"
            resultados.append(meta)
            print(f"  -> {meta['carpeta']} (n_aristas={meta['n_aristas_grafo_final']}, "
                  f"t={meta['t_reconstruir_grafo_s']+meta['t_generar_ic_s']:.1f}s)", flush=True)

    with open(RUTA_PARES_JSON, "w") as f:
        json.dump(dict(pares=todos_pares, reglas_generadas=resultados), f, indent=2)
    print(f"\n[TOTAL] {len(todos_pares)} pares -- {len(resultados)} reglas nuevas generadas -- "
          f"resumen en {RUTA_PARES_JSON}")
    return todos_pares, resultados


if __name__ == "__main__":
    n_obj = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(n_obj)
