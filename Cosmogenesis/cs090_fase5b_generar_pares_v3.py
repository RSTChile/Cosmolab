"""
cs090_fase5b_generar_pares_v3.py -- FASE V-B, escalado a ~20 pares TOTALES (10-ago-2026, tarea "escala
20 pares"). Continua cs090_fase5b_generar_pares.py y cs090_fase5b_generar_pares_v2.py (NINGUNO de los
dos se edita). Objetivo: sumar ~12 pares NUEVOS con K Y kcap EXACTAMENTE IGUALES entre regla Clase I y
Clase III a los 8 pares ya corridos (5 de ellos ya exactos: A,D,E,F,G de la tarea anterior).

LECCION DE LA TAREA ANTERIOR (no repetir): el generador congelado `cs090_fase5_generador.generar_regla`
etiqueta SIEMPRE `rule_id=f"{eje_A}-{eje_B}-{eje_C}-r{idx}"` con idx reiniciando en 0 en cada llamada a
`generar_reglas_clase` -- NO incluye el seed_base en el nombre. Si dos lotes generados en momentos
distintos usan el mismo rango de idx (r0..r39), sus nombres COLISIONAN aunque las reglas sean FISICAMENTE
DISTINTAS (distinto seed). Eso paso en la tarea anterior con r2, r9, r12 y produjo 3 corridas de Phantom
con el grafo EQUIVOCADO hasta que se detecto comparando meta_regla.json contra el CSV de origen.

FIX de esta tarea (dos capas, no una sola):
  1. `seed_base` NUEVO: 471828 -- fuera de los rangos ya usados (271828 profundizar-original,
     371828 escala_v2). Nunca colisiona con un seed ya usado en ningun lote anterior.
  2. Convencion de RENOMBRADO explicito e inmediato tras generar cada regla: el `rule_id` que devuelve
     el generador (siempre "A2-B0-C2-r{idx}", reincide con v1 y v2) se SOBRESCRIBE de inmediato, antes
     de correr el motor o guardar nada, por "A2-B0-C2-batch3-r{idx}" -- prefijo "batch3" nuevo, no usado
     por v1 (sin prefijo) ni v2 (sin prefijo, mismo rango 0-39 que colisiono). Verificado por grep: el
     motor (`correr_regla_coarse`) solo usa `p["rule_id"]` para ETIQUETAR filas de salida, nunca para
     derivar semilla ni fisica (linea 433: `rng = np.random.default_rng(p["seed"]*5000+N)`, no usa
     rule_id) -- renombrar rule_id ANTES de llamar al motor es seguro y no cambia ningun numero fisico.

Ademas, esta tarea recupera UN par que la tarea anterior generó correctamente pero nunca llegó a correr
por culpa del mismo bug de colision: dentro de las 40 candidatas v2 (`cs090_fase5b_candidatas_v2.csv`),
la propia r9 de v2 (seed=372702, I, K=5, kcap=6) tiene match EXACTO de (K,kcap) con la propia r39 de v2
(seed=375612, III, K=5, kcap=6) -- un sexto par exacto que quedó "escondido" porque el codigo de v2
reusó por error la carpeta YA CORRIDA de la r9 del PILOTO v1 (K=5, kcap=7, seed=272702, un grafo
DISTINTO) en vez de generar la IC de la r9 de v2. Se reconstruye acá, con nombre sin colision
("A2-B0-C2-r9v2fix"), y se aparea con la r39 de v2 YA CORRIDA (se reusa esa carpeta de Phantom, no se
recorre) -- pareja "gratis", sin generar ninguna regla nueva.

No se modifica ningun script congelado (`cs090_fase5_generador.py`, `cs090_fase5_motor.py`,
`cs090_fase5_clasificador.py`, `cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_correr.py`,
`cs090_fase5b_analizar.py`, `cs090_fase5b_generar_pares.py`, `cs090_fase5b_generar_pares_v2.py`) -- todos
solo se importan. No corre Phantom por si mismo (ver `cs090_fase5b_correr_v3.py`). No declara cierre ni
veredicto.
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

N_PILOTO = 2000
SEED_LAYOUT = 12345
EJE_A, EJE_B, EJE_C = "A2", "B0", "C2"

# -------------------------------------------------------------------------------------------------
# seed_base NUEVO (nunca usado antes: ni 271828 ni 371828 ni ningun seed individual de esos lotes,
# que caen en los rangos 271829-273672 y 371829-375612 respectivamente -- 471828 esta muy lejos)
# -------------------------------------------------------------------------------------------------
SEED_BASE_V3 = 471828
PREFIJO_RULE_ID_V3 = "A2-B0-C2-batch3-r"   # convencion de lote NUEVA, sin colision con r0-r19, r0-r39
N_CANDIDATAS_OBJETIVO = 150   # generosa -- tasa historica de pares exactos fue ~3/40 (7.5%), se busca
                              # margen para ~11-12 pares nuevos sin tener que hacer una segunda tanda

BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v3"
RUTA_CANDIDATAS_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_candidatas_v3.csv"
RUTA_PARES_JSON = f"{BASE_SALIDA}/pares_v3_resumen.json"

# carpeta v2 (tarea anterior) de donde se reusa la corrida YA HECHA de r39 (III, seed=375612)
CARPETA_V2 = "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v2"
RUTA_CANDIDATAS_V2_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_candidatas_v2.csv"


# =====================================================================================
# Paso 0 -- recuperar el par "escondido" r9(v2) vs r39(v2), perdido por el bug de v2
# =====================================================================================
def par_recuperado_v2():
    """La r9 de v2 (seed=372702) nunca se corrio con su propio grafo -- el bug de colision de nombre
    hizo que se reusara la carpeta de la r9 del PILOTO v1 (seed=272702, grafo DISTINTO). Se reconstruye
    acá bajo un nombre sin colision ('A2-B0-C2-r9v2fix') y se compara su (K,kcap) contra la fila real
    de r39 en cs090_fase5b_candidatas_v2.csv antes de aceptar el par -- chequeo cruzado obligatorio."""
    filas_v2 = {}
    with open(RUTA_CANDIDATAS_V2_CSV) as f:
        for row in csv.DictReader(f):
            filas_v2[row["rule_id"]] = row

    fila_r9 = filas_v2["A2-B0-C2-r9"]
    fila_r39 = filas_v2["A2-B0-C2-r39"]
    assert fila_r9["clase"] == "I" and fila_r39["clase"] == "III", "esperaba r9(v2)=I, r39(v2)=III"
    assert int(fila_r9["K"]) == int(fila_r39["K"]) and int(fila_r9["kcap"]) == int(fila_r39["kcap"]), (
        f"r9(v2) K={fila_r9['K']} kcap={fila_r9['kcap']} vs r39(v2) K={fila_r39['K']} "
        f"kcap={fila_r39['kcap']} -- NO son match exacto, no se arma el par")
    print(f"[recuperado v2] r9(v2) seed={fila_r9['seed']} K={fila_r9['K']} kcap={fila_r9['kcap']} (I) "
          f"vs r39(v2) seed={fila_r39['seed']} K={fila_r39['K']} kcap={fila_r39['kcap']} (III) "
          f"-- match exacto confirmado", flush=True)
    return dict(rid_I="A2-B0-C2-r9v2fix", seed_I=int(fila_r9["seed"]), clase_I="I",
                rid_III="A2-B0-C2-r39", seed_III=int(fila_r39["seed"]), clase_III="III",
                K=int(fila_r9["K"]), kcap=int(fila_r9["kcap"]), origen="v2_recuperado_bug_colision",
                carpeta_III_ya_corrida=f"{CARPETA_V2}/A2-B0-C2-r39_III")


# =====================================================================================
# Paso 1 -- generar reglas NUEVAS (seed_base+prefijo sin colision), clasificar, buscar match exacto
# =====================================================================================
def generar_y_clasificar_candidatas_v3(n_candidatas=N_CANDIDATAS_OBJETIVO):
    print(f"Generando {n_candidatas} reglas candidatas NUEVAS ({EJE_A}-{EJE_B}-{EJE_C}, "
          f"seed_base={SEED_BASE_V3}, prefijo rule_id='{PREFIJO_RULE_ID_V3}')...", flush=True)
    admitidas, descartadas = GEN.generar_reglas_clase(
        EJE_A, EJE_B, EJE_C, n_reglas=n_candidatas, seed_base=SEED_BASE_V3, max_intentos=n_candidatas * 4)
    print(f"  admitidas={len(admitidas)} descartadas(filtro P1-P5)={len(descartadas)}", flush=True)

    # renombrar rule_id INMEDIATAMENTE, antes de correr el motor -- ver docstring del modulo.
    for idx, p in enumerate(admitidas):
        p["rule_id"] = f"{PREFIJO_RULE_ID_V3}{idx}"

    filas_resumen = []
    t0 = time.time()
    for i, p in enumerate(admitidas):
        try:
            filas_regla = MOT.correr_regla_coarse(p, N=N_PILOTO, n_sweeps=14, escalas_b=(1, 2, 4, 8, 16),
                                                    n_seeds_null_topo=3)
        except Exception as e:
            print(f"  {p['rule_id']}: FALLO DEL MOTOR (no se toca) {type(e).__name__}: {e}", flush=True)
            continue
        r = clasificar_regla(filas_regla)
        fila = dict(rule_id=p["rule_id"], clase=r["clase"], seed=p["seed"], K=p["K"], J=p["J"],
                    noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                    sim_thr_frac=p["sim_thr_frac"], pendiente=r["pendiente_real"], z_agg=r["z_agg"],
                    holon_ratio=r["holon_ratio"])
        filas_resumen.append(fila)
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(admitidas)} clasificadas ({time.time()-t0:.0f}s)", flush=True)

    with open(RUTA_CANDIDATAS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_resumen[0].keys()))
        w.writeheader()
        w.writerows(filas_resumen)
    print(f"[candidatas v3] {len(filas_resumen)} clasificadas en {time.time()-t0:.0f}s -> "
          f"{RUTA_CANDIDATAS_CSV}", flush=True)

    cnt = defaultdict(int)
    for f in filas_resumen:
        cnt[f["clase"]] += 1
    print(f"  distribucion de clases entre candidatas v3: {dict(cnt)}", flush=True)

    # verificacion cruzada barata: todos los rule_id deben empezar con el prefijo v3, cero colision
    # posible con "A2-B0-C2-r{n}" (v1/v2) ni con "A2-B0-C2-r{n}v1fix"/"r9v2fix".
    for f in filas_resumen:
        assert f["rule_id"].startswith(PREFIJO_RULE_ID_V3), f"rule_id sin prefijo v3: {f['rule_id']}"
    return filas_resumen


def pares_exactos_v3(filas_resumen):
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
                               origen="v3_generado_nuevo",
                               seed_I=fi["seed"], seed_III=fiii["seed"]))
    return pares


# =====================================================================================
# Paso 2 -- generar condiciones iniciales de Phantom para los pares elegidos
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


def main(n_pares_objetivo=12):
    os.makedirs(BASE_SALIDA, exist_ok=True)
    resultados = []
    todos_pares = []

    # ---- Paso 0: par recuperado (gratis, sin generar nada nuevo) ----
    par0 = par_recuperado_v2()
    todos_pares.append(par0)

    # ---- Paso 1: generar candidatas v3 y buscar pares exactos entre ellas ----
    faltan = max(0, n_pares_objetivo - len(todos_pares))
    filas_v3 = generar_y_clasificar_candidatas_v3()
    pares_v3 = pares_exactos_v3(filas_v3)
    print(f"\nPares exactos entre candidatas v3: {len(pares_v3)}")
    for par in pares_v3:
        print(f"  {par['rid_I']} (I) vs {par['rid_III']} (III) -- K={par['K']} kcap={par['kcap']}")

    todos_pares += pares_v3
    todos_pares = todos_pares[:n_pares_objetivo] if len(todos_pares) > n_pares_objetivo else todos_pares
    print(f"\n[TOTAL PARES SELECCIONADOS v3] {len(todos_pares)} (objetivo={n_pares_objetivo}, "
          f"incluye 1 recuperado + {len(todos_pares)-1} nuevos de candidatas v3)")

    # seeds conocidos: par0 (nombres especiales) + candidatas v3 (prefijo batch3, sin colision)
    seeds_conocidos = {par0["rid_I"]: par0["seed_I"]}
    seeds_conocidos.update({f["rule_id"]: int(f["seed"]) for f in filas_v3})

    for par in todos_pares:
        for rule_id, clase in ((par["rid_I"], "I"), (par["rid_III"], "III")):
            carpeta_esperada = f"{BASE_SALIDA}/{rule_id}_{clase}"
            if os.path.exists(f"{carpeta_esperada}/cosmogenesis_ic.txt"):
                print(f"[{rule_id}] IC ya existe -- se salta", flush=True)
                continue
            # el lado III del par recuperado ya esta corrido en escala_v2 -- se reusa esa carpeta, NO
            # se regenera ni se recorre (misma logica de "no recomputar" que v1/v2).
            if par.get("carpeta_III_ya_corrida") and rule_id == par["rid_III"]:
                print(f"[{rule_id}] YA CORRIDO en escala_v2 -- se reusa {par['carpeta_III_ya_corrida']} "
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
    n_obj = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    main(n_obj)
