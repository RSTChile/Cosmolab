#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F7-05 · PASO 1 — Unificación de TODAS las corridas de Phantom acumuladas
========================================================================

QUÉ HACE ESTE SCRIPT (a nivel módulo)
-------------------------------------
Lee, sin modificarlos, los CSV crudos de todos los experimentos de Fase V-B y
Fase VI que pasaron grafos por Phantom, y arma UNA sola tabla donde cada fila es
UNA corrida de Phantom (un grafo → una nube de gas → un resultado gravitatorio).

Analogía: cada experimento anterior es una libreta de campo distinta, con su
propia letra y sus propias columnas. Acá se copian todas las libretas a un mismo
cuaderno, poniendo en cada renglón de qué libreta salió, y anotando cuando un
renglón ya estaba copiado desde otra libreta (para no contarlo dos veces).

REGLA DE ORO (bug documentado en FASE6_O3B_control_rewiring_CS.md)
------------------------------------------------------------------
La clave de identidad de una corrida es la PAREJA (rule_id, seed), NUNCA rule_id
solo: hay lotes distintos (v1/v2, mapa de transición, etc.) que reciclan el mismo
patrón de nombre `A2-B0-C2-rN` para reglas genuinamente distintas.
Este script verifica esa colisión numéricamente y la deja registrada.

QUÉ NO HACE
-----------
No corre Phantom. No genera reglas ni grafos. No toca ningún archivo existente.
Sólo lee CSV y escribe uno nuevo.

SALIDA
------
  cs090_fase7_f705_dataset_unificado.csv   (la tabla)
  cs090_fase7_f705_unificar.log            (el registro de qué entró y qué no)
"""

import os
import sys
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(BASE, "cs090_fase7_f705_dataset_unificado.csv")
LOG = os.path.join(BASE, "cs090_fase7_f705_unificar.log")

_log_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _log_lines.append(s)


def leer(nombre):
    """Lee un CSV del directorio base; devuelve None si no existe."""
    p = os.path.join(BASE, nombre)
    if not os.path.exists(p):
        log(f"  [FALTA] {nombre}")
        return None
    d = pd.read_csv(p)
    log(f"  [OK] {nombre}: {len(d)} filas")
    return d


# ---------------------------------------------------------------------------
# BLOQUE 1 — Las corridas de Phantom, experimento por experimento
# ---------------------------------------------------------------------------
# Columnas canónicas del cuaderno unificado. Cada lector traduce a estas.
CANON = [
    "exp",            # experimento de origen (covariable de diseño)
    "diseno",         # 'pareado' / 'no_pareado'  (advertencia de Simpson)
    "rule_id", "seed",
    "clase",          # etiqueta histórica I/III/IV (NO se usa como endpoint)
    "brazo",          # variante dentro del experimento (orig/rewire, cond, mem, ...)
    "N_nodos",        # nodos del grafo = partículas de gas iniciales
    "K", "kcap", "J", "noise", "meandeg",
    "n_aristas",      # DENSIDAD (nº de aristas del grafo final)
    "grado_medio",    # 2*E/N  (densidad normalizada por tamaño)
    "clustering",     # clustering local medio (sólo donde fue medido)
    "transitividad",
    "n_triangulos",
    "holon",          # holonomía del grafo final (cierre de ciclos, proxy parcial)
    "giant",          # fracción en la componente gigante
    "diam",           # diámetro corregido de la gigante
    "pendiente",      # pendiente corregida del coarse-graining (la "geometría")
    "frac_masa",      # DESENLACE principal: fracción de masa en sumideros
    "masa_acretada",
    "kappa_v",
    "n_sumideros",
    "t_primer_sumidero",
    "carpeta",
]


def marco_vacio(n):
    return pd.DataFrame({c: [np.nan] * n for c in CANON})


def desde(d, mapa, exp, diseno, extras=None):
    """Traduce un CSV crudo a las columnas canónicas."""
    out = marco_vacio(len(d))
    for dst, src in mapa.items():
        if src is not None and src in d.columns:
            out[dst] = d[src].values
    out["exp"] = exp
    out["diseno"] = diseno
    if extras:
        for k, v in extras.items():
            out[k] = v
    return out


bloques = []
log("=" * 78)
log("F7-05 · Unificación de corridas de Phantom")
log("=" * 78)
log("\n--- Lectura de fuentes ---")

# --- 1a) Fase V-B: los 40 pares (diseño PAREADO, densidad libre) -------------
f5b = leer("cs090_fase5b_TOTAL_40pares.csv")
# n_aristas de Fase V-B vive en los CSV de métricas por tanda
met = []
for nm in ["cs090_fase5b_escala_v2_metricas_FINAL.csv",
           "cs090_fase5b_escala_v3_metricas.csv",
           "cs090_fase5b_escala_v4_metricas.csv",
           "cs090_fase5b_metricas.csv"]:
    dd = leer(nm)
    if dd is not None:
        met.append(dd[["rule_id", "seed", "n_aristas_grafo_final", "diam_grafo_final",
                       "J", "noise", "meandeg"]])
met = pd.concat(met, ignore_index=True).drop_duplicates(subset=["rule_id", "seed"])
# pendiente corregida y giant: de la remedición de las 430 y del CSV de outliers
rem430 = leer("cs090_fase6_remedicion_430.csv")
outdg = leer("cs090_fase6_outliers_diam_gigante.csv")

if f5b is not None:
    b = desde(f5b, {
        "rule_id": "rule_id", "seed": "seed", "clase": "clase",
        "K": "K", "kcap": "kcap",
        "frac_masa": "fraccion_masa_en_sumideros",
        "masa_acretada": "masa_acretada_total",
        "kappa_v": "kappa_v_agregado",
        "n_sumideros": "n_sumideros",
        "t_primer_sumidero": "t_primer_sumidero",
        "carpeta": "carpeta",
    }, exp="F5B_40pares", diseno="pareado", extras={"N_nodos": 2000, "brazo": "unico"})
    b = b.merge(met.rename(columns={"n_aristas_grafo_final": "_ar",
                                    "diam_grafo_final": "_di",
                                    "J": "_J", "noise": "_no", "meandeg": "_md"}),
                on=["rule_id", "seed"], how="left")
    for dst, src in [("n_aristas", "_ar"), ("diam", "_di"), ("J", "_J"),
                     ("noise", "_no"), ("meandeg", "_md")]:
        b[dst] = b[dst].fillna(b[src])
    b = b.drop(columns=[c for c in b.columns if c.startswith("_")])
    if rem430 is not None:
        r = rem430[["rule_id", "seed", "pendiente_corregida", "tam_gigante"]].drop_duplicates(
            subset=["rule_id", "seed"])
        b = b.merge(r, on=["rule_id", "seed"], how="left")
        b["pendiente"] = b["pendiente"].fillna(b["pendiente_corregida"])
        # NO se usa `tam_gigante` como `giant`: en ese CSV es un CONTEO de nodos,
        # no la fracción; mezclarlo con la fracción de otros CSV daría una columna
        # con dos unidades distintas. Se deja NaN.
        b = b.drop(columns=["pendiente_corregida", "tam_gigante"])
    if outdg is not None:
        o = outdg[["rule_id", "seed", "n_aristas_b1", "giant_b1", "pend_corr"]].drop_duplicates(
            subset=["rule_id", "seed"])
        b = b.merge(o, on=["rule_id", "seed"], how="left")
        b["n_aristas"] = b["n_aristas"].fillna(b["n_aristas_b1"])
        b["giant"] = b["giant"].fillna(b["giant_b1"])   # fracción, misma unidad
        b["pendiente"] = b["pendiente"].fillna(b["pend_corr"])
        b = b.drop(columns=["n_aristas_b1", "giant_b1", "pend_corr"])
    bloques.append(b)

# --- 1b) O3-A: la misma familia a N=4000 (resolución) -----------------------
o3a = leer("cs090_fase6_o3a_resolucion_crudo.csv")
if o3a is not None:
    o3a4 = o3a[o3a["N"] == 4000].copy()
    b = desde(o3a4, {
        "rule_id": "rule_id", "seed": "seed", "clase": "clase",
        "K": "K", "kcap": "kcap",
        "n_aristas": "n_aristas_grafo_final",
        "grado_medio": "grado_medio_grafo_final",
        "diam": "diam_grafo_final_OFICIAL",
        "holon": "holon_grafo_final",
        "frac_masa": "fraccion_masa_en_sumideros",
        "masa_acretada": "masa_acretada_total",
        "kappa_v": "kappa_v_agregado",
        "n_sumideros": "n_sumideros",
        "t_primer_sumidero": "t_primer_sumidero",
    }, exp="O3A_N4000", diseno="pareado", extras={"N_nodos": 4000, "brazo": "unico"})
    bloques.append(b)

# --- 1c) O3-B: rewiring con grados idénticos (PAREADO, grados fijos) --------
o3b = leer("cs090_fase6_o3b_phantom_crudo.csv")
o3be = leer("cs090_fase6_o3b_estructura.csv")
if o3b is not None:
    b = desde(o3b, {
        "rule_id": "rule_id", "seed": "seed", "clase": "clase", "brazo": "variante",
        "K": "K", "kcap": "kcap", "J": "J", "noise": "noise", "meandeg": "meandeg",
        "n_aristas": "n_aristas_grafo_final",
        "diam": "diam_grafo_final",
        "frac_masa": "fraccion_masa_en_sumideros",
        "masa_acretada": "masa_acretada_total",
        "kappa_v": "kappa_v_agregado",
        "n_sumideros": "n_sumideros",
        "t_primer_sumidero": "t_primer_sumidero",
        "carpeta": "carpeta",
    }, exp="O3B_rewiring", diseno="pareado", extras={"N_nodos": 2000})
    if o3be is not None:
        # el clustering está por (rule_id, seed) y por brazo (orig / rewire)
        e = o3be.set_index(["rule_id", "seed"])
        for i, row in b.iterrows():
            k = (row["rule_id"], row["seed"])
            if k in e.index:
                r = e.loc[k]
                suf = "orig" if row["brazo"] == "orig" else "rewire"
                b.at[i, "clustering"] = r[f"clustering_local_{suf}"]
                b.at[i, "transitividad"] = r[f"transitividad_{suf}"]
                b.at[i, "n_triangulos"] = r[f"n_triangulos_{suf}"]
                b.at[i, "pendiente"] = r[f"pendiente_corr_{suf}"]
                b.at[i, "grado_medio"] = r["grado_medio"]
    bloques.append(b)

# --- 1d) O3-C: factorial mecanístico 4 condiciones (PAREADO dentro de regla) -
o3c = leer("cs090_fase6_o3c_crudo.csv")
if o3c is not None:
    b = desde(o3c, {
        "rule_id": "rule_id", "seed": "seed", "clase": "clase", "brazo": "cond_id",
        "K": "K", "kcap": "kcap", "J": "J", "noise": "noise", "meandeg": "meandeg",
        "n_aristas": "n_aristas_grafo_final",
        "grado_medio": "grado_medio_grafo_final",
        "diam": "diam_b1_corregido",
        "holon": "holon_grafo_final",
        "giant": "giant_grafo_final",
        "pendiente": "pendiente_corregida",
        "frac_masa": "fraccion_masa_en_sumideros",
        "masa_acretada": "masa_acretada_total",
        "kappa_v": "kappa_v_agregado",
        "n_sumideros": "n_sumideros",
        "t_primer_sumidero": "t_primer_sumidero",
        "carpeta": "carpeta",
    }, exp="O3C_factorial", diseno="pareado", extras={"N_nodos": 2000})
    bloques.append(b)

# --- 1e) O3-D: barrido de kcap + controles Erdos-Renyi (NO pareado) ---------
o3d = leer("cs090_fase6_o3d_crudo.csv")
if o3d is not None:
    b = desde(o3d, {
        "rule_id": "rule_id", "seed": "seed", "clase": "clase",
        "K": "K", "kcap": "kcap", "J": "J", "noise": "noise", "meandeg": "meandeg",
        "n_aristas": "n_aristas_grafo_final",
        "grado_medio": "grado_medio_grafo_final",
        "diam": "diam_grafo_final",
        "holon": "holon_ratio",
        "giant": "giant_grafo_final",
        "pendiente": "pendiente_corregida",
        "frac_masa": "fraccion_masa_en_sumideros",
        "masa_acretada": "masa_acretada_total",
        "kappa_v": "kappa_v_agregado",
        "n_sumideros": "n_sumideros",
        "t_primer_sumidero": "t_primer_sumidero",
        "carpeta": "carpeta",
    }, exp="O3D_kcap", diseno="no_pareado", extras={"N_nodos": 2000})
    # marcar el brazo: control Erdos-Renyi vs regla real
    b["brazo"] = np.where(b["rule_id"].astype(str).str.startswith("CONTROL-ER"),
                          "control_ER", "regla")
    bloques.append(b)

o3dh = leer("cs090_fase6_o3d_control_historico.csv")
if o3dh is not None:
    b = desde(o3dh, {
        "rule_id": "rule_id", "seed": "seed", "clase": "clase",
        "K": "K", "kcap": "kcap", "J": "J", "noise": "noise", "meandeg": "meandeg",
        "n_aristas": "n_aristas_grafo_final",
        "grado_medio": "grado_medio_grafo_final",
        "diam": "diam_grafo_final",
        "frac_masa": "fraccion_masa_en_sumideros",
        "masa_acretada": "masa_acretada_total",
        "kappa_v": "kappa_v_agregado",
        "n_sumideros": "n_sumideros",
        "t_primer_sumidero": "t_primer_sumidero",
        "carpeta": "carpeta",
    }, exp="O3D_control_hist", diseno="no_pareado",
        extras={"N_nodos": 2000, "brazo": "control_historico"})
    bloques.append(b)

# --- 1f) O3-E: memoria vs sin memoria (PAREADO) -----------------------------
o3e = leer("cs090_fase6_o3e_memoria_crudo.csv")
if o3e is not None:
    b = desde(o3e, {
        "rule_id": "rule_id", "seed": "seed", "clase": "clase_corregida_historica",
        "brazo": "brazo",
        "K": "K", "kcap": "kcap", "J": "J", "noise": "noise", "meandeg": "meandeg",
        "n_aristas": "n_aristas_grafo_final",
        "grado_medio": "grado_medio_grafo_final",
        "diam": "diam_grafo_final",
        "holon": "holon_grafo_final",
        "giant": "giant_grafo_final",
        "pendiente": "pendiente_corregida",
        "frac_masa": "fraccion_masa_en_sumideros",
        "masa_acretada": "masa_acretada_total",
        "kappa_v": "kappa_v_agregado",
        "n_sumideros": "n_sumideros",
        "t_primer_sumidero": "t_primer_sumidero",
        "carpeta": "carpeta",
    }, exp="O3E_memoria", diseno="pareado", extras={"N_nodos": 2000})
    bloques.append(b)

# --- 1g) Outliers de pendiente negativa (NO pareado) ------------------------
outp = leer("cs090_fase6_outliers_phantom_metricas.csv")
if outp is not None:
    b = desde(outp, {
        "rule_id": "rule_id", "seed": "seed", "clase": "clase",
        "K": "K", "kcap": "kcap", "J": "J", "noise": "noise", "meandeg": "meandeg",
        "n_aristas": "n_aristas_grafo_final",
        "grado_medio": "grado_medio_grafo_final",
        "diam": "diam_grafo_final",
        "giant": "giant_grafo_final",
        "pendiente": "pendiente",
        "frac_masa": "fraccion_masa_en_sumideros",
        "masa_acretada": "masa_acretada_total",
        "kappa_v": "kappa_v_agregado",
        "n_sumideros": "n_sumideros",
        "t_primer_sumidero": "t_primer_sumidero",
        "carpeta": "carpeta",
    }, exp="OUT_pendNEG", diseno="no_pareado",
        extras={"N_nodos": 2000, "brazo": "unico"})
    bloques.append(b)

D = pd.concat(bloques, ignore_index=True)
log(f"\nFilas leídas en bruto (todas las fuentes): {len(D)}")

# ---------------------------------------------------------------------------
# BLOQUE 2 — Verificación de la clave (rule_id, seed) y deduplicación
# ---------------------------------------------------------------------------
log("\n--- Verificación de la clave (rule_id, seed) ---")

# 2a) ¿existe realmente la colisión de rule_id con seeds distintos?
col = D.groupby("rule_id")["seed"].nunique()
colisiones = col[col > 1]
log(f"rule_id con MÁS DE UN seed dentro del propio dataset de Phantom: {len(colisiones)}")
if len(colisiones):
    log("  ejemplos:", ", ".join(f"{k}({v} seeds)" for k, v in colisiones.head(8).items()))

# 2b) la colisión contra los lotes históricos de grafos (mapa de transición)
mt = leer("cs090_fase5_mapa_transicion_grid.csv")
if mt is not None:
    por_rid = len(set(D.rule_id) & set(mt.rule_id))
    por_par = len(set(zip(D.rule_id, D.seed)) & set(zip(mt.rule_id, mt.seed)))
    log(f"Cruce con cs090_fase5_mapa_transicion_grid: {por_rid} coinciden por rule_id SOLO, "
        f"{por_par} por (rule_id, seed).")
    log("  -> confirma el bug documentado: unir por rule_id solo pegaría "
        f"{por_rid} filas de grafos EQUIVOCADOS.")

# 2c) duplicados: la misma corrida presente en dos fuentes
D["_clave"] = list(zip(D.rule_id, D.seed, D.brazo.astype(str), D.N_nodos))
dups = D[D.duplicated("_clave", keep=False)].sort_values("_clave")
log(f"\nFilas con clave (rule_id, seed, brazo, N) repetida: {len(dups)}")
if len(dups):
    for k, g in dups.groupby("_clave"):
        log(f"  {k[0]} seed={k[1]} brazo={k[2]} N={k[3]} -> exp {list(g.exp)} "
            f"frac_masa {list(np.round(g.frac_masa.values, 5))}")

# Política: se conserva la primera aparición según prioridad de fuente
# (la fuente con más métricas de grafo gana).
PRIO = {"O3C_factorial": 0, "O3D_kcap": 1, "O3E_memoria": 2, "O3B_rewiring": 3,
        "O3A_N4000": 4, "OUT_pendNEG": 5, "F5B_40pares": 6, "O3D_control_hist": 7}
D["_prio"] = D["exp"].map(PRIO).fillna(9)
D = D.sort_values(["_clave", "_prio"]).reset_index(drop=True)
antes = len(D)
D["_dup"] = D.duplicated("_clave", keep="first")
n_desc = int(D["_dup"].sum())
D_desc = D[D["_dup"]].copy()
D = D[~D["_dup"]].copy()
log(f"Descartadas por duplicación exacta de corrida: {n_desc} "
    f"(quedan {len(D)} de {antes})")

# ---------------------------------------------------------------------------
# BLOQUE 3 — Variables derivadas y control de calidad
# ---------------------------------------------------------------------------
log("\n--- Variables derivadas ---")
# grado medio donde falte: 2E/N
falta_gm = D["grado_medio"].isna() & D["n_aristas"].notna()
D.loc[falta_gm, "grado_medio"] = 2.0 * D.loc[falta_gm, "n_aristas"] / D.loc[falta_gm, "N_nodos"]
log(f"grado_medio reconstruido como 2E/N en {int(falta_gm.sum())} filas")

# ATENCIÓN: en los CSV de O3C/O3D/O3E la columna 'grado_medio_grafo_final' vale
# n_aristas/1000, es decir E/(N/2) = 2E/N con N=2000. Coincide con la definición
# de arriba; se verifica.
chk = D[(D.n_aristas.notna()) & (D.grado_medio.notna()) & (D.N_nodos == 2000)]
dif = (chk.grado_medio - 2 * chk.n_aristas / 2000).abs()
log(f"Chequeo grado_medio == 2E/N (N=2000): max |dif| = {dif.max():.3e} sobre {len(chk)} filas")

# densidad de aristas normalizada (aristas / aristas posibles)
D["densidad"] = 2.0 * D["n_aristas"] / (D["N_nodos"] * (D["N_nodos"] - 1))

# ---------------------------------------------------------------------------
# BLOQUE 4 — Geometría inicial (aglomeración FoF sobre las condiciones iniciales)
# ---------------------------------------------------------------------------
log("\n--- Geometría inicial (FoF sobre las IC) ---")
g3a = leer("cs090_fase6_o3a_geometria_ic.csv")
if g3a is not None:
    g = g3a[["rule_id", "N", "sep_media", "fof_b0.20", "ngrupos_b0.20",
             "fof_b0.30", "ngrupos_b0.30", "fof_b0.50", "dens_k8_cv"]].copy()
    g = g.rename(columns={"N": "N_nodos", "fof_b0.20": "geo_fof020",
                          "fof_b0.30": "geo_fof030", "fof_b0.50": "geo_fof050",
                          "ngrupos_b0.20": "geo_ngrupos020",
                          "ngrupos_b0.30": "geo_ngrupos030",
                          "dens_k8_cv": "geo_densk8cv", "sep_media": "geo_sepmedia"})
    g = g.drop_duplicates(subset=["rule_id", "N_nodos"])
    D = D.merge(g, on=["rule_id", "N_nodos"], how="left")
    log(f"geometría IC (O3-A) pegada en {int(D['geo_fof030'].notna().sum())} filas")

o4 = leer("cs090_fase6_o4a_corridas_nbody.csv")
if o4 is not None:
    q = o4[o4["dt"] == o4["dt"].min()] if "dt" in o4.columns else o4
    q = q.rename(columns={"regla": "rule_id"})
    cols = {"ini_fof_ell0.6_nmin3": "geo_o4a_fof06",
            "ini_fof_ell1.0_nmin3": "geo_o4a_fof10",
            "ini_ngrupos_ell0.6_nmin3": "geo_o4a_ngrupos06",
            "ini_knn8_frac_rho_mayor_100x": "geo_o4a_knn100x",
            "ini_knn8_frac_rho_mayor_1000x": "geo_o4a_knn1000x"}
    q = q[["rule_id"] + list(cols)].rename(columns=cols).drop_duplicates(subset=["rule_id"])
    # OJO: O4-A no guarda seed; se pega por rule_id y se marca la fila como
    # 'geometría pegada sin verificar seed'.
    D = D.merge(q, on="rule_id", how="left")
    D["geo_o4a_sin_verificar_seed"] = D["geo_o4a_fof06"].notna()
    log(f"geometría IC (O4-A, pegada SÓLO por rule_id, sin seed) en "
        f"{int(D['geo_o4a_fof06'].notna().sum())} filas — marcada como no verificada")

# ---------------------------------------------------------------------------
# BLOQUE 5 — Resumen de cobertura y escritura
# ---------------------------------------------------------------------------
D = D.drop(columns=[c for c in ["_clave", "_prio", "_dup"] if c in D.columns])
D = D.sort_values(["exp", "rule_id", "brazo"]).reset_index(drop=True)

log("\n--- Cobertura por experimento ---")
res = D.groupby("exp").agg(
    n=("frac_masa", "size"),
    con_masa=("frac_masa", "count"),
    con_aristas=("n_aristas", "count"),
    con_grado=("grado_medio", "count"),
    con_pendiente=("pendiente", "count"),
    con_clustering=("clustering", "count"),
    con_holon=("holon", "count"),
    con_geoIC=("geo_fof030", "count") if "geo_fof030" in D.columns else ("frac_masa", "count"),
)
log(res.to_string())

log("\n--- Cobertura global de cada variable ---")
for c in ["frac_masa", "masa_acretada", "n_aristas", "grado_medio", "densidad",
          "pendiente", "clustering", "transitividad", "holon", "giant", "diam",
          "kappa_v", "geo_fof030", "geo_o4a_fof06"]:
    if c in D.columns:
        log(f"  {c:22s} {int(D[c].notna().sum()):4d} / {len(D)}")

# filas sin desenlace: no sirven para nada aguas abajo
sin_masa = D["frac_masa"].isna().sum()
log(f"\nFilas sin desenlace (frac_masa NaN): {sin_masa}")

D.to_csv(SALIDA, index=False)
log(f"\nESCRITO: {SALIDA}  ({len(D)} filas x {len(D.columns)} columnas)")

if len(D_desc):
    p = os.path.join(BASE, "cs090_fase7_f705_filas_descartadas.csv")
    D_desc.drop(columns=[c for c in ["_clave", "_prio", "_dup"] if c in D_desc.columns]
                ).to_csv(p, index=False)
    log(f"ESCRITO (auditoría): {p}  ({len(D_desc)} filas descartadas por duplicación)")

with open(LOG, "w") as fh:
    fh.write("\n".join(_log_lines) + "\n")
print(f"\nlog -> {LOG}")
