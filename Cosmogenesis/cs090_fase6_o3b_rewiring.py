"""
cs090_fase6_o3b_rewiring.py — FASE VI, tarea O3-B: CONTROL DE RECONFIGURACIÓN (double-edge-swap)
=================================================================================================

QUÉ PREGUNTA CONTESTA (propuesta F6-04 de GPT-5.6 Sol)
------------------------------------------------------
En Fase V-B se vio que los grafos "geométricamente extendidos" (Clase III) acretan más masa en Phantom
que sus pares "compactos" (Clase I) con MISMO K y MISMO kcap (40 pares, test de signos p<0.001). La
objeción dura es: *¿esa ventaja viene de la ORGANIZACIÓN relacional del grafo, o de propiedades
triviales que el grafo extendido arrastra de paso (cuántos nodos, cuántas aristas, cuántos vecinos por
nodo)?*

Este archivo construye el control que separa esas dos cosas: a cada grafo Clase III se le fabrica un
**gemelo reconfigurado** que conserva, EXACTAMENTE:

  - N (mismo número de nodos, 2000),
  - el número total de aristas,
  - la SECUENCIA DE GRADOS nodo por nodo (el nodo 17 tiene los mismos 4 vecinos-en-cantidad que antes,
    aunque sean otros vecinos),

pero con las conexiones intercambiadas, de modo que se destruyen las vecindades, el clustering
(triángulos) y la estructura mesoscópica.

Analogía simple: es la misma cantidad de gente en la misma sala, y cada persona sigue teniendo el mismo
número de amigos que antes — pero ahora los amigos son otros, sorteados. Si la "ventaja" del grafo
extendido sobrevive a ese sorteo, entonces la ventaja nunca fue de la trama de amistades, sino de
cuánta gente hay y cuántos amigos tiene cada uno.

CÓMO SE HACE EL GEMELO (y por qué esta operación y no otra)
------------------------------------------------------------
`double-edge-swap` de Maslov-Sneppen: se toman dos aristas (a-b) y (c-d) y se reconectan como (a-d) y
(c-b). Cada nodo pierde un vecino y gana otro => su grado NO cambia. Repetido `factor_swaps` veces el
número de aristas, el grafo queda "barajado" pero con la misma secuencia de grados. Esa operación ya
existe implementada y validada en el proyecto: `cs072_modulos/piezas/p_semilla_causal.py::barajar_aristas`
(la usa NULL-3 de la Fase II de CS073 vía `null3_generar_ic.py`). AQUÍ SE IMPORTA TAL CUAL — no se
copia ni se modifica. La variante CON filtro de longitud de `null3_investigacion_preliminar.py` NO se
usa a propósito: acá queremos destruir la estructura local, no preservarla.

Se verifica NUMÉRICAMENTE, nodo por nodo, que la secuencia de grados sea idéntica (no se asume).

QUÉ MIDE ANTES DE PHANTOM (para poder decir "cuánto cambió efectivamente")
---------------------------------------------------------------------------
  - solapamiento de aristas original vs gemelo (|E_orig ∩ E_gemelo| / |E|),
  - clustering: coeficiente local medio (Watts-Strogatz) y transitividad global (3·triángulos/tríadas),
  - número de triángulos,
  - tamaño de la componente gigante,
  - pendiente corregida log(diám)-vs-log(N_cajas) del coarse-graining, con la medición OFICIAL de
    diámetro `cs090_diam_corregido.diam_gigante` (REGLA VIGENTE 11-ago-2026: NO se usa el `_diam` de
    cs055, que puede apoyarse en un pedacito suelto del grafo),
  - z_agg vs NULL_topo (los MISMOS 3 grafos Erdős-Rényi de la regla original, que dependen sólo de la
    semilla y de la densidad de aristas — y la densidad es idéntica entre original y gemelo, así que la
    comparación es literalmente contra la misma vara).

QUÉ NO HACE
-----------
No corre Phantom (eso es `cs090_fase6_o3b_correr.py`). No modifica ningún script congelado
(`cs090_fase5_generador.py`, `cs090_fase5_motor.py`, `cs090_fase5_clasificador.py`,
`cs090_fase5b_phantom_adaptador.py`, `cs090_diam_corregido.py`, `p_semilla_causal.py`,
`null3_generar_ic.py`, `null3_investigacion_preliminar.py`) — todos sólo se importan. No declara cierre
ni veredicto: sólo fabrica los gemelos, los mide y escribe las condiciones iniciales.

CRITERIO DE SELECCIÓN DE LOS 12 GRAFOS (documentado, reproducible desde CSV)
-----------------------------------------------------------------------------
Universo: las 40 reglas con rol "III" de `cs090_fase5b_TOTAL_40pares.csv` (las que ya pasaron por
Phantom en Fase V-B). Se les une la pendiente CORREGIDA de `cs090_fase6_remedicion_430.csv`
(1 de las 40, `A2-B0-C2-r2v1fix`, no está en esa remedición y queda fuera; las 39 restantes siguen
siendo Clase III con la medición corregida).

Se eligen las **3 de mayor pendiente corregida dentro de cada uno de los 4 lotes** de `seed_base`
(271828 / 371828 / 471828 / 571828) = 12 grafos. Se reparte por lote a propósito, siguiendo la
observación de `FASE6_O2F_N_efectivo_fase5b_CS.md` de que el techo del diseño de Fase V-B es
experimental (pocos `seed_base` distintos) y no estadístico: 12 grafos concentrados en un solo lote
serían menos independientes entre sí que 3+3+3+3.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase5_generador as GEN                     # sólo import
import cs090_fase5_motor as MOT                         # sólo import
import cs080_renormalizacion as CS80                    # sólo import (cajas_bfs / grafo_grueso)
import cg003_diagnostico_gromov as GR                   # sólo import (ER aleatorio para NULL_topo)
import cs090_diam_corregido as DC                       # medición OFICIAL de diámetro
from cs090_fase5_clasificador import _pendiente_loglog  # sólo import
from cs090_fase5b_phantom_adaptador import (            # sólo import (pipeline validado Fase V-B)
    reconstruir_regla_a2b0c2, generar_ic_masa_fija_desde_grafo,
)
from cs072_modulos.piezas.p_semilla_causal import barajar_aristas   # double-edge-swap, sólo import

# ---------------------------------------------------------------------------------------------
# Parámetros — todos heredados de la línea Fase V-B, ninguno nuevo salvo los del rewiring
# ---------------------------------------------------------------------------------------------
N_NODOS = 2000
N_SWEEPS = 14
SEED_LAYOUT = 12345                 # MISMA semilla de layout que toda la Fase V-B
ESCALAS_B = (1, 2, 4, 8, 16)
N_SEEDS_NULL_TOPO = 3

FACTOR_SWAPS = 10                   # convención estándar del método (10 × nº de aristas intentos),
                                    # el mismo default de `barajar_aristas` y de NULL-3
MULT_SEED_REWIRE = 9100             # multiplicador NUEVO para derivar la semilla del swap desde el
                                    # seed de la regla. Los ya usados en la línea son 1000/2000/5000/
                                    # 6000/7000/7500/8000 (cs090_fase5_motor + cs090_diam_corregido):
                                    # 9100 no colisiona con ninguno.

BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_fase6_o3b_rewiring"
SUFIJO_ORIG = "orig"
SUFIJO_REWIRE = "rewire"            # prefijo/sufijo de nombre NUEVO, jamás usado en la línea
                                    # (lección del bug de colisión de nombres de Fase V-B)

RUTA_TOTAL_40 = f"{HERE}/cs090_fase5b_TOTAL_40pares.csv"
RUTA_REMEDICION = f"{HERE}/cs090_fase6_remedicion_430.csv"
RUTA_SELECCION = f"{HERE}/cs090_fase6_o3b_seleccion.csv"
RUTA_ESTRUCTURA = f"{HERE}/cs090_fase6_o3b_estructura.csv"


# =============================================================================================
# 1) SELECCIÓN de los 12 grafos Clase III (criterio documentado en el docstring)
# =============================================================================================
def lote_de_seed(seed: int) -> str:
    """Cada tanda de Fase V-B usó un `seed_base` distinto y sus seeds individuales caen en un rango
    disjunto: v1 271829-273672, v2 371829-375612, v3 471829-~485700, v4 571829-~581000."""
    seed = int(seed)
    if seed < 300000:
        return "271828"
    if seed < 400000:
        return "371828"
    if seed < 500000:
        return "471828"
    return "571828"


def seleccionar_grafos(n_por_lote: int = 3):
    with open(RUTA_TOTAL_40) as f:
        total = list(csv.DictReader(f))
    # OJO -- COLISIÓN DE rule_id REAL, detectada al construir esta tarea: los lotes v1 y v2 de Fase V-B
    # usaron el MISMO patrón de nombre `A2-B0-C2-r{idx}` sin prefijo de lote, así que existen dos reglas
    # DISTINTAS llamadas, por ejemplo, `A2-B0-C2-r14` (seed 273187 del lote 271828 y seed 373187 del lote
    # 371828). Indexar por rule_id solo hace que una pise a la otra y se lea la pendiente equivocada. La
    # clave de unión correcta es (rule_id, seed) -- misma lección que el bug de colisión de nombres de
    # Fase V-B, ahora del lado del análisis.
    with open(RUTA_REMEDICION) as f:
        remed = {(r["rule_id"], int(r["seed"])): r for r in csv.DictReader(f)}

    vistos, filas = set(), []
    for r in total:
        if r["rol"] != "III":
            continue
        rid = r["rule_id"]
        clave = (rid, int(r["seed"]))
        if clave in vistos:          # el CSV de 40 pares repite alguna regla usada en 2 pares
            continue
        vistos.add(clave)
        m = remed.get(clave)
        if m is None:
            print(f"   [selección] {rid} no está en la remedición corregida -- queda fuera", flush=True)
            continue
        filas.append(dict(
            rule_id=rid, seed=int(r["seed"]), lote=lote_de_seed(r["seed"]),
            K=int(r["K"]), kcap=int(r["kcap"]),
            pendiente_corregida=float(m["pendiente_corregida"]),
            pendiente_vieja=float(m["pendiente_vieja"]),
            clase_corregida=m["clase_corregida"],
            frac_masa_fase5b=float(r["fraccion_masa_en_sumideros"]),
            kappa_v_fase5b=(float(r["kappa_v_agregado"]) if r["kappa_v_agregado"] else None),
            par_fase5b=r["par"], carpeta_fase5b=r["carpeta"],
        ))

    por_lote = defaultdict(list)
    for f in filas:
        por_lote[f["lote"]].append(f)

    elegidos = []
    for lote in sorted(por_lote):
        grupo = sorted(por_lote[lote], key=lambda d: -d["pendiente_corregida"])[:n_por_lote]
        elegidos.extend(grupo)
        print(f"   [selección] lote {lote}: {len(por_lote[lote])} disponibles -> elijo "
              f"{[ (g['rule_id'], round(g['pendiente_corregida'], 4)) for g in grupo ]}", flush=True)

    with open(RUTA_SELECCION, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(elegidos[0].keys()))
        w.writeheader()
        w.writerows(elegidos)
    print(f"   [selección] {len(elegidos)} grafos -> {RUTA_SELECCION.split('/')[-1]}", flush=True)
    return elegidos


# =============================================================================================
# 2) MÉTRICAS ESTRUCTURALES de un grafo (formato nativo: lista/dict de sets)
# =============================================================================================
def _a_lista(adj, N):
    """Normaliza a lista-de-sets indexable por entero (barajar_aristas devuelve dict).

    OJO: reconstruir los sets NO es inocuo para `cs080.cajas_bfs`, que recorre `for v in adj[u]` — el
    ORDEN DE ITERACIÓN de un `set` de Python depende de cómo se llenó, así que un set con los mismos
    elementos pero armado en otro orden puede dar una partición en cajas distinta (verificado: en
    A2-B0-C2-r14 la pendiente pasaba de 0.7729 a 0.7517 sólo por eso). Por eso el grafo ORIGINAL se mide
    con el objeto nativo que devuelve el motor (reproduce el histórico bit a bit) y, aparte, se mide
    también en forma CANÓNICA — ver `canonicalizar`."""
    if isinstance(adj, dict):
        return [set(adj.get(i, ())) for i in range(N)]
    return [set(a) for a in adj]


def canonicalizar(adj, N):
    """Devuelve una lista de sets equivalente pero con un orden de llenado FIJO (vecinos insertados de
    menor a mayor). Sirve para que original y gemelo reciban EXACTAMENTE el mismo trato en el
    coarse-graining: cualquier diferencia de pendiente entre ambos queda atribuible al grafo y no al
    orden accidental en que quedaron armados los conjuntos de vecinos en memoria."""
    out = []
    for i in range(N):
        s = set()
        for v in sorted(adj[i]):
            s.add(v)
        out.append(s)
    return out


def aristas_set(adj, N):
    return {(i, j) for i in range(N) for j in adj[i] if j > i}


def grados(adj, N):
    return np.array([len(adj[i]) for i in range(N)], dtype=int)


def clustering(adj, N):
    """Devuelve (clustering local medio, transitividad global, nº de triángulos).

    - clustering local del nodo i = triángulos que lo tocan / pares de vecinos suyos (C_i=0 si grado<2),
      promediado sobre TODOS los nodos (definición Watts-Strogatz).
    - transitividad global = 3·triángulos / nº de tríadas conectadas (definición de Newman).
    Cálculo exacto por enumeración: con kcap≈5 y N=2000 el coste es despreciable."""
    tri_por_nodo = np.zeros(N, dtype=float)
    n_tri_total = 0
    n_triadas = 0
    for i in range(N):
        vec = sorted(adj[i])
        d = len(vec)
        n_triadas += d * (d - 1) // 2
        if d < 2:
            continue
        t = 0
        for a_idx in range(d):
            va = vec[a_idx]
            for b_idx in range(a_idx + 1, d):
                if vec[b_idx] in adj[va]:
                    t += 1
        tri_por_nodo[i] = t
        n_tri_total += t
    n_tri_total //= 3            # cada triángulo contado una vez por cada uno de sus 3 vértices
    c_local = np.zeros(N, dtype=float)
    for i in range(N):
        d = len(adj[i])
        if d >= 2:
            c_local[i] = tri_por_nodo[i] / (d * (d - 1) / 2.0)
    transitividad = (3.0 * n_tri_total / n_triadas) if n_triadas > 0 else float("nan")
    return float(c_local.mean()), float(transitividad), int(n_tri_total)


def tam_gigante(adj, N):
    comps = DC.componentes(adj, N)
    return max((len(c) for c in comps), default=0)


# =============================================================================================
# 3) PENDIENTE CORREGIDA de un grafo cualquiera (mismo coarse-graining que la línea)
# =============================================================================================
def nulls_topo_de(seed, N, n_aristas):
    """Los MISMOS 3 grafos Erdős-Rényi que usa `cs090_diam_corregido.correr_regla_coarse_doble` para
    esta regla: dependen sólo de (seed, N, densidad de aristas). Original y gemelo tienen exactamente
    la misma densidad, así que ambos se comparan contra LA MISMA vara -- no se genera una vara nueva
    para el gemelo."""
    meandeg_equiv = max(0.5, 2.0 * n_aristas / max(1, N))
    out = []
    for s in range(N_SEEDS_NULL_TOPO):
        adj0, _ = GR.aleatorio(N, meandeg=meandeg_equiv, seed=int(seed * 6000 + N * 13 + s))
        out.append([set(a.tolist()) for a in adj0])
    return out


def pendiente_corregida_de_grafo(adj, N, seed, adjs_null):
    """Repite la cadena de coarse-graining de `cs090_diam_corregido.correr_regla_coarse_doble` (mismas
    semillas derivadas: cajas del REAL con seed*7000+b*31, cajas de los NULL con seed*7500+b*37+k) pero
    sobre un grafo ARBITRARIO -- el original o su gemelo reconfigurado. El diámetro se mide SIEMPRE con
    `DC.diam_gigante` (medición oficial vigente). Devuelve pendiente, z por escala, z_agg y detalle."""
    Ns, diams, zs = [], [], []
    detalle = []
    for b in ESCALAS_B:
        if b == 1:
            adj_g, n_cajas = adj, N
        else:
            rng_b = np.random.default_rng(seed * 7000 + b * 31)
            asign, n_cajas = CS80.cajas_bfs(adj, N, b, rng_b)
            adj_g = CS80.grafo_grueso(adj, N, asign, n_cajas)
        d_real = float(DC.diam_gigante(adj_g, n_cajas)) if n_cajas > 1 else float("nan")

        d_nulls = []
        for k_null, adj_n in enumerate(adjs_null):
            if b == 1:
                adj_ng, n_cajas_n = adj_n, N
            else:
                rng_bn = np.random.default_rng(seed * 7500 + b * 37 + k_null)
                asign_n, n_cajas_n = CS80.cajas_bfs(adj_n, N, b, rng_bn)
                adj_ng = CS80.grafo_grueso(adj_n, N, asign_n, n_cajas_n)
            d_nulls.append(float(DC.diam_gigante(adj_ng, n_cajas_n)) if n_cajas_n > 1 else float("nan"))
        d_null_m = float(np.nanmean(d_nulls))
        d_null_s = float(np.nanstd(d_nulls)) + 1e-9

        Ns.append(n_cajas); diams.append(d_real)
        zs.append(abs(d_real - d_null_m) / d_null_s if d_null_s > 1e-9 else 0.0)
        detalle.append(dict(b=b, n_cajas=n_cajas, diam=d_real, diam_null=d_null_m))

    return dict(pendiente=_pendiente_loglog(Ns, diams), z_agg=float(np.mean(zs)),
                z_sostenido=bool(all(z > 3.0 for z in zs)), diams=diams, n_cajas=Ns,
                z_por_escala=[round(z, 2) for z in zs], detalle=detalle)


# =============================================================================================
# 4) FABRICAR el par (original, gemelo reconfigurado) + medir + escribir IC de Phantom
# =============================================================================================
def procesar_una(sel, generar_ic=True):
    rid, seed = sel["rule_id"], sel["seed"]
    t_ini = time.time()

    # --- original: MISMA reconstrucción bit a bit que usó Fase V-B ---
    p, m = reconstruir_regla_a2b0c2(seed=seed, N=N_NODOS, n_sweeps=N_SWEEPS)
    adj_orig_nativo = m["adj_final"]        # objeto TAL CUAL lo devuelve el motor (orden de sets nativo)
    adj_orig = _a_lista(adj_orig_nativo, N_NODOS)
    E_orig = aristas_set(adj_orig, N_NODOS)
    g_orig = grados(adj_orig, N_NODOS)

    # --- gemelo reconfigurado: double-edge-swap importado tal cual ---
    seed_rewire = int(seed * MULT_SEED_REWIRE + 7)
    t0 = time.time()
    adj_rw = _a_lista(barajar_aristas(adj_orig, N_NODOS, seed=seed_rewire, factor_swaps=FACTOR_SWAPS),
                      N_NODOS)
    t_swap = time.time() - t0
    E_rw = aristas_set(adj_rw, N_NODOS)
    g_rw = grados(adj_rw, N_NODOS)

    # --- VERIFICACIÓN NUMÉRICA, nodo por nodo (no se asume) ---
    grados_identicos = bool(np.array_equal(g_orig, g_rw))
    n_nodos_grado_distinto = int((g_orig != g_rw).sum())
    assert grados_identicos, (
        f"{rid}: el swap NO preservó la secuencia de grados ({n_nodos_grado_distinto} nodos difieren) "
        "-- no se sigue adelante con este par")
    assert len(E_orig) == len(E_rw), f"{rid}: cambió el nº de aristas {len(E_orig)} -> {len(E_rw)}"
    # sin bucles ni aristas repetidas (el swap las rechaza por construcción; se comprueba igual)
    assert all(i not in adj_rw[i] for i in range(N_NODOS)), f"{rid}: el gemelo tiene un bucle i-i"

    solapamiento = len(E_orig & E_rw) / len(E_orig) if E_orig else float("nan")

    cl_o, tr_o, ntri_o = clustering(adj_orig, N_NODOS)
    cl_r, tr_r, ntri_r = clustering(adj_rw, N_NODOS)

    adjs_null = nulls_topo_de(seed, N_NODOS, len(E_orig))
    # (a) medición HISTÓRICA del original: objeto nativo del motor -> reproduce cs090_fase6_remedicion_430
    pc_hist = pendiente_corregida_de_grafo(adj_orig_nativo, N_NODOS, seed, adjs_null)
    # (b) medición CANÓNICA de los dos: mismo trato exacto, comparable entre sí (ver `canonicalizar`)
    pc_o = pendiente_corregida_de_grafo(canonicalizar(adj_orig, N_NODOS), N_NODOS, seed, adjs_null)
    pc_r = pendiente_corregida_de_grafo(canonicalizar(adj_rw, N_NODOS), N_NODOS, seed, adjs_null)

    fila = dict(
        rule_id=rid, seed=seed, lote=sel["lote"], K=sel["K"], kcap=sel["kcap"],
        n_nodos=N_NODOS, n_aristas=len(E_orig), grado_medio=2.0 * len(E_orig) / N_NODOS,
        seed_rewire=seed_rewire, factor_swaps=FACTOR_SWAPS, t_swap_s=round(t_swap, 2),
        grados_identicos=grados_identicos, n_nodos_grado_distinto=n_nodos_grado_distinto,
        n_aristas_rewire=len(E_rw), solapamiento_aristas=solapamiento,
        clustering_local_orig=cl_o, clustering_local_rewire=cl_r,
        transitividad_orig=tr_o, transitividad_rewire=tr_r,
        n_triangulos_orig=ntri_o, n_triangulos_rewire=ntri_r,
        gigante_orig=tam_gigante(adj_orig, N_NODOS), gigante_rewire=tam_gigante(adj_rw, N_NODOS),
        pendiente_corr_orig=pc_o["pendiente"], pendiente_corr_rewire=pc_r["pendiente"],
        z_agg_orig=pc_o["z_agg"], z_agg_rewire=pc_r["z_agg"],
        z_sost_orig=pc_o["z_sostenido"], z_sost_rewire=pc_r["z_sostenido"],
        diams_orig=pc_o["diams"], diams_rewire=pc_r["diams"], n_cajas=pc_o["n_cajas"],
        pendiente_corr_orig_historica=pc_hist["pendiente"], diams_orig_historica=pc_hist["diams"],
        pendiente_corr_csv_o3b_ref=sel["pendiente_corregida"],
        frac_masa_fase5b=sel["frac_masa_fase5b"], carpeta_fase5b=sel["carpeta_fase5b"],
    )

    # --- verificación cruzada: la pendiente corregida recalculada acá (medición histórica, objeto
    #     nativo) tiene que coincidir con la de cs090_fase6_remedicion_430.csv ---
    d_pend = abs(fila["pendiente_corr_orig_historica"] - sel["pendiente_corregida"])
    fila["pendiente_orig_reproduce_remedicion"] = bool(d_pend < 1e-9)
    fila["delta_pendiente_vs_remedicion"] = d_pend
    if d_pend >= 1e-9:
        print(f"   !! {rid}: la pendiente corregida recalculada "
              f"({fila['pendiente_corr_orig_historica']:.6f}) NO coincide con la de la remedición "
              f"({sel['pendiente_corregida']:.6f}), Δ={d_pend:.2e}", flush=True)

    if generar_ic:
        # El ORIGINAL se manda a Phantom con el objeto NATIVO del motor -> su condición inicial sale
        # idéntica a la de Fase V-B (se verifica byte a byte en `verificar_ic_original_vs_fase5b`), lo
        # que convierte a la mitad "original" de cada par en una réplica exacta de un resultado ya
        # conocido: si no reprodujera, el par no sería confiable.
        for variante, adj in ((SUFIJO_ORIG, adj_orig_nativo), (SUFIJO_REWIRE, adj_rw)):
            carpeta = f"{BASE_SALIDA}/{rid}_{variante}"
            os.makedirs(carpeta, exist_ok=True)
            t1 = time.time()
            info = generar_ic_masa_fija_desde_grafo(adj, N=N_NODOS, seed_layout=SEED_LAYOUT,
                                                    ruta_salida=f"{carpeta}/cosmogenesis_ic.txt")
            t_ic = time.time() - t1
            meta = dict(
                tarea="FASE6_O3B_control_rewiring", variante=variante, rule_id=rid, clase="III",
                seed=seed, lote=sel["lote"], N=N_NODOS, seed_layout=SEED_LAYOUT,
                K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                sim_thr_frac=p["sim_thr_frac"],
                n_aristas_grafo_final=(len(E_orig) if variante == SUFIJO_ORIG else len(E_rw)),
                grado_medio_grafo_final=2.0 * len(E_orig) / N_NODOS,
                seed_rewire=(seed_rewire if variante == SUFIJO_REWIRE else None),
                factor_swaps=(FACTOR_SWAPS if variante == SUFIJO_REWIRE else None),
                solapamiento_aristas=(solapamiento if variante == SUFIJO_REWIRE else 1.0),
                clustering_local=(cl_o if variante == SUFIJO_ORIG else cl_r),
                transitividad=(tr_o if variante == SUFIJO_ORIG else tr_r),
                pendiente_corregida=(fila["pendiente_corr_orig"] if variante == SUFIJO_ORIG
                                     else fila["pendiente_corr_rewire"]),
                masa_total_ic=info["masa_total"], carpeta=carpeta,
                t_generar_ic_s=round(t_ic, 2),
            )
            with open(f"{carpeta}/meta_regla.json", "w") as f:
                json.dump(meta, f, indent=2)
            fila[f"carpeta_{variante}"] = carpeta
            fila[f"t_ic_{variante}_s"] = round(t_ic, 2)

    if generar_ic:
        fila.update(verificar_ic_original_vs_fase5b(fila, sel))

    fila["t_total_s"] = round(time.time() - t_ini, 2)
    return fila


BATERIAS_FASE5B = [
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto",
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v2",
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v3",
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v4",
]


def resolver_carpeta_fase5b(valor: str) -> str:
    """La columna `carpeta` de `cs090_fase5b_TOTAL_40pares.csv` a veces trae la ruta absoluta (lotes v1 y
    v2) y a veces sólo el NOMBRE de la carpeta (lotes v3 y v4, consolidados con otro script). Se resuelve
    buscando ese nombre en las 4 baterías conocidas de Fase V-B."""
    if not valor:
        return ""
    if os.path.isabs(valor):
        return valor
    for base in BATERIAS_FASE5B:
        cand = os.path.join(base, valor)
        if os.path.isdir(cand):
            return cand
    return valor


def verificar_ic_original_vs_fase5b(fila, sel):
    """VERIFICACIÓN CRUZADA OBLIGATORIA (lección del bug de colisión de nombres de Fase V-B): antes de
    confiar en un par, se comprueba que (a) el `meta_regla.json` de la carpeta de Fase V-B que dice
    corresponder a esta regla tenga el MISMO rule_id y el MISMO seed que el que estamos usando, y (b)
    que el archivo de condición inicial que acabamos de escribir para el ORIGINAL sea idéntico, línea
    por línea de coordenadas, al que se usó en Fase V-B. Si (b) coincide, la mitad "original" de este
    par es literalmente la misma corrida que ya está informada, y cualquier diferencia con el gemelo
    viene del gemelo."""
    out = dict(fase5b_meta_coincide=None, ic_orig_identico_a_fase5b=None,
               ic_orig_max_dif_coordenada=None)
    carpeta_vieja = resolver_carpeta_fase5b(sel.get("carpeta_fase5b") or "")
    meta_vieja = os.path.join(carpeta_vieja, "meta_regla.json")
    ic_vieja = os.path.join(carpeta_vieja, "cosmogenesis_ic.txt")
    if not os.path.exists(meta_vieja) or not os.path.exists(ic_vieja):
        return out
    mv = json.load(open(meta_vieja))
    out["fase5b_meta_coincide"] = bool(mv.get("rule_id") == fila["rule_id"]
                                       and int(mv.get("seed", -1)) == fila["seed"])
    a = open(ic_vieja).read().splitlines()
    b = open(os.path.join(fila["carpeta_orig"], "cosmogenesis_ic.txt")).read().splitlines()
    # la línea 0 es un comentario con texto libre; se comparan las líneas numéricas (1..N)
    if len(a) != len(b):
        out["ic_orig_identico_a_fase5b"] = False
        return out
    out["ic_orig_identico_a_fase5b"] = bool(a[1:] == b[1:])
    va = np.array([[float(x) for x in l.split()] for l in a[2:]])
    vb = np.array([[float(x) for x in l.split()] for l in b[2:]])
    out["ic_orig_max_dif_coordenada"] = float(np.abs(va - vb).max())
    return out


def main(indices=None, generar_ic=True, sufijo_csv=""):
    elegidos = seleccionar_grafos()
    if indices is not None:
        elegidos = [elegidos[i] for i in indices]
    print(f"\n[o3b] procesando {len(elegidos)} grafos (generar_ic={generar_ic})", flush=True)

    filas, t0 = [], time.time()
    for k, sel in enumerate(elegidos):
        print(f"[{k+1}/{len(elegidos)}] {sel['rule_id']} (lote {sel['lote']}, "
              f"pend_corr={sel['pendiente_corregida']:.4f})...", flush=True)
        fila = procesar_una(sel, generar_ic=generar_ic)
        filas.append(fila)
        print(f"    aristas={fila['n_aristas']} grados_identicos={fila['grados_identicos']} "
              f"solape={fila['solapamiento_aristas']:.4f} "
              f"clust {fila['clustering_local_orig']:.4f}->{fila['clustering_local_rewire']:.4f} "
              f"trans {fila['transitividad_orig']:.4f}->{fila['transitividad_rewire']:.4f} "
              f"pend {fila['pendiente_corr_orig']:.4f}->{fila['pendiente_corr_rewire']:.4f} "
              f"({fila['t_total_s']}s)", flush=True)

    ruta = RUTA_ESTRUCTURA.replace(".csv", f"{sufijo_csv}.csv")
    with open(ruta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\n[o3b] {len(filas)} filas -> {ruta.split('/')[-1]}  (total {time.time()-t0:.0f}s)")
    return filas


if __name__ == "__main__":
    # uso: python3.9 cs090_fase6_o3b_rewiring.py [idx0,idx1,...] [--sin-ic] [--sufijo=_shard0]
    idxs, gen_ic, suf = None, True, ""
    for arg in sys.argv[1:]:
        if arg == "--sin-ic":
            gen_ic = False
        elif arg.startswith("--sufijo="):
            suf = arg.split("=", 1)[1]
        else:
            idxs = [int(x) for x in arg.split(",")]
    main(indices=idxs, generar_ic=gen_ic, sufijo_csv=suf)
