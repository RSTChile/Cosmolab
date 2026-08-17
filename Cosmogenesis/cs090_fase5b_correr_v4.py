"""
cs090_fase5b_correr_v4.py -- FASE V-B, escalado de 20 a ~40 pares TOTALES: genera IC + corre Phantom +
analiza para los pares NUEVOS de esta tarea: los 6 pares "sobrantes" de v3 (candidatas ya generadas y
clasificadas en la tarea anterior, sin Phantom todavia) + 14 pares elegidos de las 25 candidatas exactas
encontradas entre las 220 reglas nuevas v4 (seed_base=571828, prefijo "A2-B0-C2-batch4-r").

Verificacion cruzada OBLIGATORIA (misma disciplina de v3, no se salta): antes de aceptar cualquier par
como valido, se comparan K/kcap/seed/clase leidos del meta_regla.json real (escrito tras reconstruir el
grafo) contra la fila correspondiente del CSV de candidatas (cs090_fase5b_candidatas_v3.csv para los 6
sobrantes, cs090_fase5b_candidatas_v4.csv para los 14 nuevos) -- si algo no coincide, AssertionError, no
se sigue en silencio.

No se modifica ningun script congelado (`cs090_fase5b_generar_pares_v4.py`,
`cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_correr.py`, `cs090_fase5b_analizar.py`) -- todos solo
se importan. No declara cierre ni veredicto.
"""
from __future__ import annotations
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")

from cs090_fase5b_generar_pares_v4 import (verificar_pares_sobrantes_v3, generar_ic_para_regla,
                                             BASE_SALIDA, PARES_SOBRANTES_V3)
from cs090_fase5b_correr import correr_una
from cs090_fase5b_analizar import analizar_carpeta

RUTA_CANDIDATAS_V4_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_candidatas_v4.csv"
RUTA_METRICAS_V4_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_escala_v4_metricas.csv"

# ---------------------------------------------------------------------------------------------------
# 14 pares NUEVOS elegidos de las 25 candidatas exactas encontradas en las 220 reglas v4 (ver stdout de
# cs090_fase5b_generar_pares_v4.py, log cs090_fase5b_generar_pares_v4.log), priorizando diversidad de
# bucket (K,kcap) primero (los 8 buckets distintos que aparecieron) y despues repeticiones en los
# buckets con mas de un candidato disponible -- mismo criterio que uso v3 para elegir sus 11.
# ---------------------------------------------------------------------------------------------------
PARES_NUEVOS_V4 = [
    dict(rid_I="A2-B0-C2-batch4-r11",  rid_III="A2-B0-C2-batch4-r0",   K=6, kcap=6),
    dict(rid_I="A2-B0-C2-batch4-r31",  rid_III="A2-B0-C2-batch4-r1",   K=6, kcap=5),
    dict(rid_I="A2-B0-C2-batch4-r3",   rid_III="A2-B0-C2-batch4-r9",   K=5, kcap=6),
    dict(rid_I="A2-B0-C2-batch4-r23",  rid_III="A2-B0-C2-batch4-r10",  K=5, kcap=5),
    dict(rid_I="A2-B0-C2-batch4-r15",  rid_III="A2-B0-C2-batch4-r62",  K=8, kcap=5),
    dict(rid_I="A2-B0-C2-batch4-r51",  rid_III="A2-B0-C2-batch4-r36",  K=8, kcap=4),
    dict(rid_I="A2-B0-C2-batch4-r38",  rid_III="A2-B0-C2-batch4-r72",  K=4, kcap=5),
    dict(rid_I="A2-B0-C2-batch4-r57",  rid_III="A2-B0-C2-batch4-r43",  K=7, kcap=5),
    dict(rid_I="A2-B0-C2-batch4-r13",  rid_III="A2-B0-C2-batch4-r94",  K=6, kcap=6),
    dict(rid_I="A2-B0-C2-batch4-r41",  rid_III="A2-B0-C2-batch4-r12",  K=6, kcap=5),
    dict(rid_I="A2-B0-C2-batch4-r18",  rid_III="A2-B0-C2-batch4-r19",  K=5, kcap=6),
    dict(rid_I="A2-B0-C2-batch4-r39",  rid_III="A2-B0-C2-batch4-r26",  K=5, kcap=5),
    dict(rid_I="A2-B0-C2-batch4-r70",  rid_III="A2-B0-C2-batch4-r47",  K=7, kcap=5),
    dict(rid_I="A2-B0-C2-batch4-r44",  rid_III="A2-B0-C2-batch4-r28",  K=6, kcap=5),
]


def cargar_csv_v4():
    filas = {}
    with open(RUTA_CANDIDATAS_V4_CSV) as f:
        for row in csv.DictReader(f):
            filas[row["rule_id"]] = row
    return filas


def verificar_par_contra_csv(par, filas_v4):
    fi = filas_v4[par["rid_I"]]
    fiii = filas_v4[par["rid_III"]]
    assert fi["clase"] == "I", f"{par['rid_I']} no es clase I en el CSV (es {fi['clase']})"
    assert fiii["clase"] == "III", f"{par['rid_III']} no es clase III en el CSV (es {fiii['clase']})"
    assert int(fi["K"]) == par["K"] and int(fi["kcap"]) == par["kcap"], (
        f"{par['rid_I']}: CSV dice K={fi['K']} kcap={fi['kcap']}, par declara K={par['K']} kcap={par['kcap']}")
    assert int(fiii["K"]) == par["K"] and int(fiii["kcap"]) == par["kcap"], (
        f"{par['rid_III']}: CSV dice K={fiii['K']} kcap={fiii['kcap']}, par declara K={par['K']} kcap={par['kcap']}")
    return int(fi["seed"]), int(fiii["seed"])


def verificar_meta_contra_csv(carpeta, seed_esperado, K_esperado, kcap_esperado, fuente_csv_nombre):
    meta = json.loads((Path(carpeta) / "meta_regla.json").read_text())
    assert meta["seed"] == seed_esperado, (
        f"{carpeta}: meta_regla.json seed={meta['seed']} != esperado {seed_esperado} (fuente {fuente_csv_nombre}) "
        f"-- POSIBLE COLISION DE NOMBRE, abortando")
    assert meta["K"] == K_esperado and meta["kcap"] == kcap_esperado, (
        f"{carpeta}: meta_regla.json K={meta['K']} kcap={meta['kcap']} != esperado K={K_esperado} "
        f"kcap={kcap_esperado} -- POSIBLE COLISION DE NOMBRE, abortando")
    return meta


def main(n_pares_v4_objetivo=14):
    t_inicio = time.time()

    # -------------------- Paso 1: verificar los 6 pares sobrantes de v3 --------------------
    sobrantes = verificar_pares_sobrantes_v3()   # ya verificados adentro (clase/K/kcap contra CSV v3)
    todos = [dict(rid_I=p["rid_I"], rid_III=p["rid_III"], K=p["K"], kcap=p["kcap"],
                  seed_I=p["seed_I"], seed_III=p["seed_III"], origen="v3_sobrante") for p in sobrantes]

    # -------------------- Paso 2: verificar los pares v4 elegidos contra el CSV ANTES de generar nada --------------------
    filas_v4 = cargar_csv_v4()
    elegidos = PARES_NUEVOS_V4[:n_pares_v4_objetivo]
    for par in elegidos:
        seed_I, seed_III = verificar_par_contra_csv(par, filas_v4)
        par["seed_I"] = seed_I
        par["seed_III"] = seed_III
        par["origen"] = "v4_generado_nuevo"
        todos.append(par)
    print(f"[verificacion previa] {len(todos)} pares ({len(sobrantes)} sobrantes v3 + {len(elegidos)} "
          f"nuevos v4) verificados contra CSV de origen -- OK, ningun mismatch antes de generar IC", flush=True)

    # -------------------- Paso 3: generar IC (grafo + layout) para cada regla que falte --------------------
    carpetas_por_par = []
    for i, par in enumerate(todos):
        carpeta_I = Path(f"{BASE_SALIDA}/{par['rid_I']}_I")
        carpeta_III = Path(f"{BASE_SALIDA}/{par['rid_III']}_III")
        fuente = "candidatas_v3" if par["origen"] == "v3_sobrante" else "candidatas_v4"

        if not (carpeta_I / "cosmogenesis_ic.txt").exists():
            print(f"[{i+1}/{len(todos)}] generando IC {par['rid_I']} (I, seed={par['seed_I']})...", flush=True)
            t0 = time.time()
            meta = generar_ic_para_regla(par["rid_I"], par["seed_I"], "I")
            verificar_meta_contra_csv(meta["carpeta"], par["seed_I"], par["K"], par["kcap"], fuente)
            print(f"  -> ok, t={time.time()-t0:.0f}s, meta verificado contra CSV", flush=True)
        else:
            print(f"[{i+1}/{len(todos)}] IC de {par['rid_I']} ya existe -- se salta", flush=True)

        if not (carpeta_III / "cosmogenesis_ic.txt").exists():
            print(f"[{i+1}/{len(todos)}] generando IC {par['rid_III']} (III, seed={par['seed_III']})...", flush=True)
            t0 = time.time()
            meta = generar_ic_para_regla(par["rid_III"], par["seed_III"], "III")
            verificar_meta_contra_csv(meta["carpeta"], par["seed_III"], par["K"], par["kcap"], fuente)
            print(f"  -> ok, t={time.time()-t0:.0f}s, meta verificado contra CSV", flush=True)
        else:
            print(f"[{i+1}/{len(todos)}] IC de {par['rid_III']} ya existe -- se salta", flush=True)

        carpetas_por_par.append((carpeta_I, carpeta_III))

    print(f"\n[TOTAL IC] generadas/verificadas para {len(todos)} pares en {time.time()-t_inicio:.0f}s", flush=True)

    # -------------------- Paso 4: correr Phantom (reusando correr_una, SIN modificar el script congelado) --------------------
    t_phantom0 = time.time()
    for i, (cI, cIII) in enumerate(carpetas_por_par):
        for carpeta in (cI, cIII):
            print(f"[{i+1}/{len(todos)}] Phantom en {carpeta.name}...", flush=True)
            info = correr_una(carpeta)
            if info.get("ya_corrida"):
                print(f"  ya tenia cosmog_00500 -- se salta", flush=True)
            else:
                print(f"  exit_setup={info['exit_setup']} t_setup={info['t_setup']}s "
                      f"exit_run={info['exit_run']} t_run={info['t_run']}s", flush=True)
                if info["exit_run"] != 0:
                    print(f"  AVISO: exit_run != 0 en {carpeta}", flush=True)
    print(f"\n[TOTAL PHANTOM] {time.time()-t_phantom0:.0f}s", flush=True)

    # -------------------- Paso 5: analizar (reusando analizar_carpeta, SIN modificar el script congelado) --------------------
    filas_metricas = []
    for par, (cI, cIII) in zip(todos, carpetas_por_par):
        fila_I = analizar_carpeta(cI)
        fila_III = analizar_carpeta(cIII)
        assert fila_I["clase"] == "I", f"{cI}: clase real={fila_I['clase']} != I esperado"
        assert fila_III["clase"] == "III", f"{cIII}: clase real={fila_III['clase']} != III esperado"
        assert fila_I["K"] == par["K"] and fila_I["kcap"] == par["kcap"], f"{cI}: K/kcap no coincide con el par"
        assert fila_III["K"] == par["K"] and fila_III["kcap"] == par["kcap"], f"{cIII}: K/kcap no coincide con el par"
        for fila, rol in ((fila_I, "I"), (fila_III, "III")):
            fila["par"] = f"{par['rid_I']}_vs_{par['rid_III']}"
            fila["rol"] = rol
            fila["match_exacto_K_kcap"] = True
            fila["origen_par"] = par["origen"]
            filas_metricas.append(fila)

    campos = list(filas_metricas[0].keys())
    for f in filas_metricas:
        for c in f:
            if c not in campos:
                campos.append(c)
    with open(RUTA_METRICAS_V4_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas_metricas)
    print(f"\n[TOTAL] {len(todos)} pares nuevos ({len(filas_metricas)} corridas) -- metricas en "
          f"{RUTA_METRICAS_V4_CSV} -- tiempo total script={time.time()-t_inicio:.0f}s", flush=True)
    return todos, filas_metricas


if __name__ == "__main__":
    n_obj = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    main(n_obj)
