"""
cs090_fase5b_generar_pares_v4.py -- FASE V-B, escalado de 20 a ~40 pares TOTALES (10-ago-2026, tarea
"escala 40 pares"). Continua cs090_fase5b_generar_pares.py, _v2.py y _v3.py (NINGUNO de los tres se
edita). Objetivo: sumar ~20 pares NUEVOS con K Y kcap EXACTAMENTE IGUALES entre regla Clase I y Clase III
a los 20 pares ya corridos y consolidados en cs090_fase5b_TOTAL_20pares.csv.

Dos fuentes de pares nuevos, en este orden (mismo principio de "lo gratis primero" que ya uso v3):

  1. LOS 6 PARES SOBRANTES DE v3 -- de las 150 candidatas generadas en la tarea anterior
     (cs090_fase5b_candidatas_v3.csv, seed_base=471828, prefijo batch3) se habian encontrado 17 pares
     exactos (K,kcap) pero solo se corrieron 11 (mas 1 recuperado del bug de v2 = 12 corridos en total).
     Quedaron 6 pares exactos YA GENERADOS Y CLASIFICADOS (el motor relacional ya corrio sobre ellos, ya
     tienen K/kcap/clase en el CSV) pero SIN condicion inicial de Phantom todavia -- se pueden correr sin
     generar ninguna regla nueva. Confirmado programaticamente comparando la lista completa de pares
     exactos de v3 contra PARES_NUEVOS_V3 (definidos en cs090_fase5b_correr_v3.py, congelado):

       sobrantes = pares_exactos_v3(candidatas_v3.csv) - PARES_NUEVOS_V3(correr_v3.py)
       => 6 pares: (r59,r58) (r107,r71) (r112,r108) (r120,r111) (r76,r26) (r143,r70)

  2. CANDIDATAS NUEVAS -- si 6 no alcanzan para llegar a ~20 pares nuevos (probable: la tarea pide
     agregar ~20 para llegar a n~40 total), se genera un lote v4 con seed_base y prefijo NUEVOS, mismas
     dos capas de proteccion anti-colision que v2/v3:
       - seed_base=571828 (patron de la linea: 271828 -> 371828 -> 471828 -> 571828; nunca usado antes,
         fuera de los rangos ya ocupados por v1/v2/v3)
       - prefijo rule_id "A2-B0-C2-batch4-r{idx}" (nunca usado antes)

No se modifica ningun script congelado (`cs090_fase5_generador.py`, `cs090_fase5_motor.py`,
`cs090_fase5_clasificador.py`, `cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_generar_pares.py`,
`cs090_fase5b_generar_pares_v2.py`, `cs090_fase5b_generar_pares_v3.py`, `cs090_fase5b_correr.py`,
`cs090_fase5b_correr_v3.py`, `cs090_fase5b_analizar.py`) -- todos solo se importan. No corre Phantom por
si mismo (ver cs090_fase5b_correr_v4.py). No declara cierre ni veredicto.
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
# NOTA: NO se reusa generar_ic_para_regla de cs090_fase5b_generar_pares_v3.py -- esa funcion usa el
# BASE_SALIDA GLOBAL definido en el modulo v3 (".../escala_v3"), asi que llamarla desde aca escribiria
# las IC nuevas de v4 adentro de la carpeta de la tarea ANTERIOR por error de scope, no por bug de
# nombres esta vez sino por reuso descuidado de una funcion que cierra sobre una constante ajena. Se
# reimplementa aca (misma logica exacta, ver abajo) apuntando al BASE_SALIDA de ESTE modulo
# (".../escala_v4"). v3 sigue sin tocarse.

N_PILOTO = 2000
SEED_LAYOUT = 12345
EJE_A, EJE_B, EJE_C = "A2", "B0", "C2"

# -------------------------------------------------------------------------------------------------
# seed_base NUEVO: sigue el patron de la linea (271828 -> 371828 -> 471828 -> 571828). Los rangos
# individuales ya usados son 271829-273672 (v1), 371829-375612 (v2), 471829-~485700 (v3, 150 candidatas
# con seed=seed_base+idx aprox) -- 571828 esta muy lejos de todos.
# -------------------------------------------------------------------------------------------------
SEED_BASE_V4 = 571828
PREFIJO_RULE_ID_V4 = "A2-B0-C2-batch4-r"   # prefijo nuevo, sin colision con r0-r19 (v1), r0-r39 (v2),
                                            # batch3-r0..r149 (v3), ni con los fix (r9v2fix, etc.)
N_CANDIDATAS_OBJETIVO_V4 = 220   # generoso: tasa historica de pares exactos 7.5%-11.3%; con 220 se
                                  # espera entre ~16 y ~25 pares exactos, mas que suficiente para los
                                  # ~14 que hacen falta si los 6 sobrantes de v3 no alcanzan solos

BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v4"
RUTA_CANDIDATAS_V3_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_candidatas_v3.csv"
RUTA_CANDIDATAS_V4_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_candidatas_v4.csv"
RUTA_PARES_JSON = f"{BASE_SALIDA}/pares_v4_resumen.json"

# Pares nuevos ya corridos en v3 (PARES_NUEVOS_V3 mas el recuperado r9v2fix/r39) -- se usa SOLO para
# verificar por diferencia de conjuntos que los 6 sobrantes abajo son, en efecto, los que faltan (no se
# importa el modulo correr_v3 para no arrastrar dependencias de Phantom en este paso de generacion).
RULE_IDS_YA_CORRIDOS_V3 = {
    "A2-B0-C2-r9v2fix", "A2-B0-C2-r39",
    "A2-B0-C2-batch3-r100", "A2-B0-C2-batch3-r0",
    "A2-B0-C2-batch3-r1", "A2-B0-C2-batch3-r69",
    "A2-B0-C2-batch3-r5", "A2-B0-C2-batch3-r114",
    "A2-B0-C2-batch3-r44", "A2-B0-C2-batch3-r10",
    "A2-B0-C2-batch3-r86", "A2-B0-C2-batch3-r12",
    "A2-B0-C2-batch3-r50", "A2-B0-C2-batch3-r21",
    "A2-B0-C2-batch3-r48", "A2-B0-C2-batch3-r25",
    "A2-B0-C2-batch3-r35", "A2-B0-C2-batch3-r31",
    "A2-B0-C2-batch3-r9", "A2-B0-C2-batch3-r83",
    "A2-B0-C2-batch3-r53", "A2-B0-C2-batch3-r23",
    "A2-B0-C2-batch3-r104", "A2-B0-C2-batch3-r60",
}

# Los 6 pares exactos "sobrantes" de v3 -- confirmados por script (ver docstring) comparando TODOS los
# pares exactos hallados entre las 150 candidatas v3 contra los 11 efectivamente corridos.
PARES_SOBRANTES_V3 = [
    dict(rid_I="A2-B0-C2-batch3-r59",  rid_III="A2-B0-C2-batch3-r58",  K=7, kcap=5),
    dict(rid_I="A2-B0-C2-batch3-r107", rid_III="A2-B0-C2-batch3-r71",  K=6, kcap=5),
    dict(rid_I="A2-B0-C2-batch3-r112", rid_III="A2-B0-C2-batch3-r108", K=6, kcap=5),
    dict(rid_I="A2-B0-C2-batch3-r120", rid_III="A2-B0-C2-batch3-r111", K=6, kcap=5),
    dict(rid_I="A2-B0-C2-batch3-r76",  rid_III="A2-B0-C2-batch3-r26",  K=8, kcap=5),
    dict(rid_I="A2-B0-C2-batch3-r143", rid_III="A2-B0-C2-batch3-r70",  K=4, kcap=5),
]


def verificar_pares_sobrantes_v3():
    """Verifica los 6 pares sobrantes contra el CSV real de candidatas v3 (clase, K, kcap) Y confirma
    que ninguno de los 12 rule_id que componen esos 6 pares ya fue corrido en v3 (interseccion vacia con
    RULE_IDS_YA_CORRIDOS_V3) -- doble chequeo antes de tocar nada."""
    filas_v3 = {}
    with open(RUTA_CANDIDATAS_V3_CSV) as f:
        for row in csv.DictReader(f):
            filas_v3[row["rule_id"]] = row

    for par in PARES_SOBRANTES_V3:
        assert par["rid_I"] not in RULE_IDS_YA_CORRIDOS_V3, f"{par['rid_I']} ya fue corrido en v3!"
        assert par["rid_III"] not in RULE_IDS_YA_CORRIDOS_V3, f"{par['rid_III']} ya fue corrido en v3!"
        fi, fiii = filas_v3[par["rid_I"]], filas_v3[par["rid_III"]]
        assert fi["clase"] == "I", f"{par['rid_I']} no es clase I en CSV v3 (es {fi['clase']})"
        assert fiii["clase"] == "III", f"{par['rid_III']} no es clase III en CSV v3 (es {fiii['clase']})"
        assert int(fi["K"]) == par["K"] and int(fi["kcap"]) == par["kcap"], (
            f"{par['rid_I']}: CSV dice K={fi['K']} kcap={fi['kcap']}, esperado K={par['K']} kcap={par['kcap']}")
        assert int(fiii["K"]) == par["K"] and int(fiii["kcap"]) == par["kcap"], (
            f"{par['rid_III']}: CSV dice K={fiii['K']} kcap={fiii['kcap']}, esperado K={par['K']} kcap={par['kcap']}")
        par["seed_I"] = int(fi["seed"])
        par["seed_III"] = int(fiii["seed"])
        par["origen"] = "v3_sobrante_ya_clasificado"
    print(f"[sobrantes v3] {len(PARES_SOBRANTES_V3)} pares verificados contra "
          f"{RUTA_CANDIDATAS_V3_CSV} -- clase/K/kcap OK, sin solape con los 12 ya corridos en v3", flush=True)
    return PARES_SOBRANTES_V3


# =====================================================================================
# Generar IC de Phantom para una regla -- MISMA logica que generar_ic_para_regla de v3, pero apuntando
# al BASE_SALIDA de v4 (ver nota arriba sobre por que no se reusa la de v3 directamente)
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


# =====================================================================================
# Generar candidatas NUEVAS v4 (si los 6 sobrantes de v3 no alcanzan) -- mismo patron que v3
# =====================================================================================
def generar_y_clasificar_candidatas_v4(n_candidatas=N_CANDIDATAS_OBJETIVO_V4):
    print(f"Generando {n_candidatas} reglas candidatas NUEVAS ({EJE_A}-{EJE_B}-{EJE_C}, "
          f"seed_base={SEED_BASE_V4}, prefijo rule_id='{PREFIJO_RULE_ID_V4}')...", flush=True)
    admitidas, descartadas = GEN.generar_reglas_clase(
        EJE_A, EJE_B, EJE_C, n_reglas=n_candidatas, seed_base=SEED_BASE_V4, max_intentos=n_candidatas * 4)
    print(f"  admitidas={len(admitidas)} descartadas(filtro P1-P5)={len(descartadas)}", flush=True)

    for idx, p in enumerate(admitidas):
        p["rule_id"] = f"{PREFIJO_RULE_ID_V4}{idx}"

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

    with open(RUTA_CANDIDATAS_V4_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_resumen[0].keys()))
        w.writeheader()
        w.writerows(filas_resumen)
    print(f"[candidatas v4] {len(filas_resumen)} clasificadas en {time.time()-t0:.0f}s -> "
          f"{RUTA_CANDIDATAS_V4_CSV}", flush=True)

    cnt = defaultdict(int)
    for f in filas_resumen:
        cnt[f["clase"]] += 1
    print(f"  distribucion de clases entre candidatas v4: {dict(cnt)}", flush=True)

    for f in filas_resumen:
        assert f["rule_id"].startswith(PREFIJO_RULE_ID_V4), f"rule_id sin prefijo v4: {f['rule_id']}"
    return filas_resumen


def pares_exactos_v4(filas_resumen):
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
                               origen="v4_generado_nuevo",
                               seed_I=fi["seed"], seed_III=fiii["seed"]))
    return pares


if __name__ == "__main__":
    # modo standalone: solo genera y clasifica candidatas v4 + busca pares exactos, no corre Phantom.
    verificar_pares_sobrantes_v3()
    n_obj = int(sys.argv[1]) if len(sys.argv) > 1 else N_CANDIDATAS_OBJETIVO_V4
    filas = generar_y_clasificar_candidatas_v4(n_obj)
    pares = pares_exactos_v4(filas)
    print(f"\nPares exactos entre candidatas v4: {len(pares)}")
    buckets = defaultdict(list)
    for par in pares:
        buckets[(par["K"], par["kcap"])].append(par)
        print(f"  {par['rid_I']} (I) vs {par['rid_III']} (III) -- K={par['K']} kcap={par['kcap']}")
    print(f"\nbuckets distintos: {len(buckets)} -> {sorted(buckets.keys())}")
