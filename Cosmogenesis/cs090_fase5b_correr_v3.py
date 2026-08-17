"""
cs090_fase5b_correr_v3.py -- FASE V-B, escalado a ~20 pares TOTALES: genera IC + corre Phantom +
analiza para los 12 pares NUEVOS elegidos de `cs090_fase5b_generar_pares_v3.py` (1 recuperado del bug
de colision de v2 + 11 nuevos de las 150 candidatas v3, seed_base=471828, prefijo "A2-B0-C2-batch3-r").

Verificacion cruzada OBLIGATORIA (leccion de la tarea anterior): antes de aceptar cualquier par como
valido, se comparan K/kcap/seed/clase leidos del meta_regla.json real (escrito por
`generar_ic_para_regla` a partir del grafo YA RECONSTRUIDO) contra la fila correspondiente del CSV de
candidatas (cs090_fase5b_candidatas_v3.csv o cs090_fase5b_candidatas_v2.csv para el par recuperado) --
si algo no coincide, se aborta con AssertionError en vez de seguir con datos silenciosamente erroneos.

No se modifica ningun script congelado (`cs090_fase5b_generar_pares_v3.py`,
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

from cs090_fase5b_generar_pares_v3 import (par_recuperado_v2, generar_ic_para_regla, BASE_SALIDA,
                                             CARPETA_V2)
from cs090_fase5b_correr import correr_una
from cs090_fase5b_analizar import analizar_carpeta

RUTA_CANDIDATAS_V3_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_candidatas_v3.csv"
RUTA_METRICAS_V3_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_escala_v3_metricas.csv"

# ---------------------------------------------------------------------------------------------------
# 11 pares NUEVOS elegidos de las 17 candidatas exactas encontradas en las 150 reglas v3 (ver stdout de
# generar_y_clasificar_candidatas_v3), priorizando diversidad de bucket (K,kcap) -- 8 buckets distintos
# cubiertos + 3 repeticiones de bucket para sumar observaciones. Todos verificados a mano contra
# cs090_fase5b_candidatas_v3.csv antes de escribir esta lista (ver informe).
# ---------------------------------------------------------------------------------------------------
PARES_NUEVOS_V3 = [
    dict(rid_I="A2-B0-C2-batch3-r100", rid_III="A2-B0-C2-batch3-r0",   K=5, kcap=4),
    dict(rid_I="A2-B0-C2-batch3-r1",   rid_III="A2-B0-C2-batch3-r69",  K=7, kcap=6),
    dict(rid_I="A2-B0-C2-batch3-r5",   rid_III="A2-B0-C2-batch3-r114", K=6, kcap=6),
    dict(rid_I="A2-B0-C2-batch3-r44",  rid_III="A2-B0-C2-batch3-r10",  K=7, kcap=5),
    dict(rid_I="A2-B0-C2-batch3-r86",  rid_III="A2-B0-C2-batch3-r12",  K=5, kcap=5),
    dict(rid_I="A2-B0-C2-batch3-r50",  rid_III="A2-B0-C2-batch3-r21",  K=6, kcap=5),
    dict(rid_I="A2-B0-C2-batch3-r48",  rid_III="A2-B0-C2-batch3-r25",  K=8, kcap=5),
    dict(rid_I="A2-B0-C2-batch3-r35",  rid_III="A2-B0-C2-batch3-r31",  K=4, kcap=5),
    dict(rid_I="A2-B0-C2-batch3-r9",   rid_III="A2-B0-C2-batch3-r83",  K=7, kcap=6),
    dict(rid_I="A2-B0-C2-batch3-r53",  rid_III="A2-B0-C2-batch3-r23",  K=7, kcap=5),
    dict(rid_I="A2-B0-C2-batch3-r104", rid_III="A2-B0-C2-batch3-r60",  K=6, kcap=5),
]


def cargar_csv_v3():
    filas = {}
    with open(RUTA_CANDIDATAS_V3_CSV) as f:
        for row in csv.DictReader(f):
            filas[row["rule_id"]] = row
    return filas


def verificar_par_contra_csv(par, filas_v3):
    """Chequeo cruzado obligatorio: (K,kcap,clase) del par declarado deben coincidir EXACTO con la fila
    real del CSV de candidatas para ambos rule_id -- si no, AssertionError (no se sigue en silencio)."""
    fi = filas_v3[par["rid_I"]]
    fiii = filas_v3[par["rid_III"]]
    assert fi["clase"] == "I", f"{par['rid_I']} no es clase I en el CSV (es {fi['clase']})"
    assert fiii["clase"] == "III", f"{par['rid_III']} no es clase III en el CSV (es {fiii['clase']})"
    assert int(fi["K"]) == par["K"] and int(fi["kcap"]) == par["kcap"], (
        f"{par['rid_I']}: CSV dice K={fi['K']} kcap={fi['kcap']}, par declara K={par['K']} kcap={par['kcap']}")
    assert int(fiii["K"]) == par["K"] and int(fiii["kcap"]) == par["kcap"], (
        f"{par['rid_III']}: CSV dice K={fiii['K']} kcap={fiii['kcap']}, par declara K={par['K']} kcap={par['kcap']}")
    return int(fi["seed"]), int(fiii["seed"])


def verificar_meta_contra_csv(carpeta, seed_esperado, K_esperado, kcap_esperado, fuente_csv_nombre):
    """Chequeo cruzado obligatorio #2: el meta_regla.json REAL escrito en disco tras generar la IC debe
    coincidir con lo que el CSV de origen decia ANTES de generar nada -- detecta el mismo tipo de bug de
    colision de nombre que afecto a la tarea anterior (r2/r9/r12), esta vez verificado por script, no a
    mano."""
    meta = json.loads((Path(carpeta) / "meta_regla.json").read_text())
    assert meta["seed"] == seed_esperado, (
        f"{carpeta}: meta_regla.json seed={meta['seed']} != esperado {seed_esperado} (fuente {fuente_csv_nombre}) "
        f"-- POSIBLE COLISION DE NOMBRE, abortando")
    assert meta["K"] == K_esperado and meta["kcap"] == kcap_esperado, (
        f"{carpeta}: meta_regla.json K={meta['K']} kcap={meta['kcap']} != esperado K={K_esperado} "
        f"kcap={kcap_esperado} -- POSIBLE COLISION DE NOMBRE, abortando")
    return meta


def main():
    t_inicio = time.time()
    filas_v3 = cargar_csv_v3()

    # -------------------- Paso 1: par recuperado (v2, "gratis") --------------------
    par0 = par_recuperado_v2()
    todos = [dict(rid_I=par0["rid_I"], rid_III=par0["rid_III"], K=par0["K"], kcap=par0["kcap"],
                  seed_I=par0["seed_I"], seed_III=par0["seed_III"], origen="v2_recuperado",
                  carpeta_III_ya_corrida=par0["carpeta_III_ya_corrida"])]

    # -------------------- Paso 2: verificar los 11 pares v3 contra el CSV ANTES de generar nada --------------------
    for par in PARES_NUEVOS_V3:
        seed_I, seed_III = verificar_par_contra_csv(par, filas_v3)
        par["seed_I"] = seed_I
        par["seed_III"] = seed_III
        par["origen"] = "v3_generado_nuevo"
        todos.append(par)
    print(f"[verificacion previa] {len(todos)} pares (1 recuperado + {len(todos)-1} v3) verificados "
          f"contra CSV de origen -- OK, ningun mismatch antes de generar IC", flush=True)

    # -------------------- Paso 3: generar IC (grafo + layout) para cada regla que falte --------------------
    carpetas_por_par = []
    for i, par in enumerate(todos):
        carpeta_I = Path(f"{BASE_SALIDA}/{par['rid_I']}_I")
        carpeta_III_default = Path(f"{BASE_SALIDA}/{par['rid_III']}_III")
        carpeta_III = Path(par["carpeta_III_ya_corrida"]) if par.get("carpeta_III_ya_corrida") else carpeta_III_default

        if not (carpeta_I / "cosmogenesis_ic.txt").exists():
            print(f"[{i+1}/{len(todos)}] generando IC {par['rid_I']} (I, seed={par['seed_I']})...", flush=True)
            t0 = time.time()
            meta = generar_ic_para_regla(par["rid_I"], par["seed_I"], "I")
            verificar_meta_contra_csv(meta["carpeta"], par["seed_I"], par["K"], par["kcap"], "candidatas_v3/v2")
            print(f"  -> ok, t={time.time()-t0:.0f}s, meta verificado contra CSV", flush=True)
        else:
            print(f"[{i+1}/{len(todos)}] IC de {par['rid_I']} ya existe -- se salta", flush=True)

        if par.get("carpeta_III_ya_corrida"):
            print(f"[{i+1}/{len(todos)}] {par['rid_III']} (III) YA CORRIDO en {carpeta_III} -- se reusa, "
                  f"no se regenera ni se recorre", flush=True)
        elif not (carpeta_III_default / "cosmogenesis_ic.txt").exists():
            print(f"[{i+1}/{len(todos)}] generando IC {par['rid_III']} (III, seed={par['seed_III']})...", flush=True)
            t0 = time.time()
            meta = generar_ic_para_regla(par["rid_III"], par["seed_III"], "III")
            verificar_meta_contra_csv(meta["carpeta"], par["seed_III"], par["K"], par["kcap"], "candidatas_v3/v2")
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
        # chequeo cruzado #3: la clase leida del meta_regla.json real debe coincidir con el rol asignado
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
    with open(RUTA_METRICAS_V3_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas_metricas)
    print(f"\n[TOTAL] {len(todos)} pares nuevos ({len(filas_metricas)} corridas) -- metricas en "
          f"{RUTA_METRICAS_V3_CSV} -- tiempo total script={time.time()-t_inicio:.0f}s", flush=True)
    return todos, filas_metricas


if __name__ == "__main__":
    main()
