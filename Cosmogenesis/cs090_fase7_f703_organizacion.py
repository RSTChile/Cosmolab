"""
cs090_fase7_f703_organizacion.py — FASE VII, tarea F7-03: GRADOS **Y** TRIÁNGULOS FIJOS,
                                    lo único que cambia es CÓMO ESTÁN ORGANIZADOS LOS TRIÁNGULOS
=================================================================================================

QUÉ PREGUNTA CONTESTA
---------------------
F7-02 (`FASE7_F702_escalera_clustering_CS.md`) dejó establecido que el clustering es una PALANCA
CAUSAL: con la secuencia de grados clavada nodo por nodo, subir el clustering de C=0 a C≈0.24-0.60
subió la masa acretada por Phantom en los 12/12 grafos (Page L p=1.3e-11, +37.2% en todo el rango).

Queda una ambigüedad: el clustering se movió subiendo el NÚMERO de triángulos. Entonces, ¿el mecanismo
son los triángulos (su cantidad), o el clustering era un correlato de algo más fino — de CÓMO están
repartidos esos triángulos por el grafo?

Esta tarea es el control natural: se fabrican variantes del mismo grafo con
  (a) la secuencia de grados IDÉNTICA nodo por nodo (verificado con `np.array_equal`, no se asume), y
  (b) el número total de triángulos IGUAL o lo más cercano posible (se reporta la diferencia REAL),
pero con esos triángulos ORGANIZADOS de maneras contrastantes.

Analogía simple: en F7-02 comparábamos maquetas con pocos y con muchos triangulitos, y las de muchos
juntaban más arena. Ahora todas las maquetas tienen EXACTAMENTE la misma cantidad de triangulitos y las
mismas varillas por poste; lo único distinto es dónde se pusieron: todos amontonados en unos pocos
barrios, o repartidos parejo por todo el pueblo; pegados unos a otros compartiendo varilla, o
separados sin tocarse. Si la arena responde igual en las cuatro, lo que importa es CUÁNTOS triangulitos
hay. Si responde distinto, "clustering" era el nombre grueso de algo más fino.

LAS CINCO ORGANIZACIONES (todas parten del MISMO ancestro común)
-----------------------------------------------------------------
Todas las variantes de un mismo grafo salen del MISMO "piso" E0: el original al que se le destruyeron
los triángulos con swaps de Maslov-Sneppen (la misma función `bajar_clustering` de F7-02, importada,
no reimplementada). Desde ese piso común, se vuelven a construir triángulos con swaps DIRIGIDOS — el
mismo movimiento elemental de F7-02, (x-p),(y-q) -> (x-y),(p-q), que conserva el grado de los cuatro
nodos involucrados — pero con un criterio de aceptación distinto en cada brazo:

  - **libre**   : se acepta cualquier cierre con balance neto de triángulos positivo. Es EXACTAMENTE el
                  criterio de F7-02, y sirve de referencia: la organización "natural" del proceso.
  - **conc**    : concentrada. El vértice-ápice se sortea de una bolsa donde cada nodo aparece tantas
                  veces como triángulos ya ayudó a cerrar (atracción preferencial: rico se hace más
                  rico), y una vez arrancado se exige que el trío nuevo toque un nodo que ya participa
                  de >=3 triángulos. Resultado: los triángulos se apilan en pocos barrios densos.
  - **disp**    : dispersa. Se exige que NINGUNO de los tres vértices del triángulo nuevo supere un cupo
                  de triángulos por nodo (cupo que arranca en 1 y sube de a uno sólo cuando el proceso
                  se atasca, tipo "llenar por capas parejas"). Resultado: los triángulos se reparten
                  sobre la mayor cantidad posible de nodos distintos.
  - **solap**   : solapada. Se exige que el triángulo nuevo COMPARTA UNA ARISTA con un triángulo ya
                  existente. Resultado: "libros" — muchos triángulos apoyados en la misma arista.
  - **disj**    : no solapada (disjunta en aristas). Se exige lo contrario: que cada una de las aristas
                  tocadas quede en exactamente UN triángulo. Resultado: un empaquetamiento de triángulos
                  que no se tocan por ninguna arista.

Los dos primeros pares recorren dos ejes distintos de "organización": conc/disp es concentración a
nivel de NODOS; solap/disj es solapamiento a nivel de ARISTAS.

CÓMO SE IGUALA EL NÚMERO DE TRIÁNGULOS ENTRE BRAZOS
----------------------------------------------------
Cada brazo se corre hasta su techo (con la MISMA restricción de conectividad que F7-02: no se admite
un snapshot que rompa más del 3% de la componente gigante del original). De cada brazo se guarda la
LISTA DE SWAPS ACEPTADOS, no fotos del grafo: así reconstruir el grafo en cualquier punto del recorrido
es simplemente volver a aplicar los primeros k swaps sobre el piso (barato y sin gastar memoria).

Después se toma **T\\* = el mínimo de los cinco techos** y se rebobina cada brazo hasta el swap cuyo
número de triángulos queda MÁS CERCA de T\\*. Como cada swap aceptado suma +1 o +2 triángulos, los cinco
brazos quedan a 0-2 triángulos unos de otros; el número exacto conseguido se reporta por brazo
(columna `n_triangulos`) y la diferencia máxima entre brazos también (`dif_max_triangulos`).

CÓMO SE MIDE "ORGANIZACIÓN"
----------------------------
Con los triángulos enumerados explícitamente se calcula, para cada variante:
  - `frac_nodos_en_triangulo`, `frac_aristas_en_triangulo` — cuánto del grafo participa (soporte).
  - `frac_aristas_multi_tri` — de las aristas que están en algún triángulo, qué fracción está en más de
    uno (la medida directa de SOLAPAMIENTO pedida).
  - `gini_tri_nodo` — desigualdad de Gini de los triángulos por nodo (0 = todos igual, 1 = todo en uno):
    la medida directa de CONCENTRACIÓN.
  - `n_comp_tri`, `frac_mayor_comp_tri` — componentes del "grafo de triángulos" (dos triángulos están
    conectados si comparten al menos un nodo): en cuántos cúmulos separados vive el conjunto.
  - `modularidad_tri` — modularidad de Newman de la partición que esos cúmulos inducen sobre los nodos.
  - `dist_media_tri` y `dist_media_azar` — distancia media (en saltos del grafo) entre triángulos
    distintos, y la misma distancia entre nodos al azar como vara de comparación.
  - clustering local medio, transitividad, asortatividad, componentes, gigante, y la **pendiente
    corregida** log(diám)-vs-log(N_cajas) con la medición oficial (`cs090_diam_corregido`), para poder
    distinguir si un eventual efecto pasa por la geometría o no.

QUÉ REUSA Y QUÉ NO TOCA
------------------------
Sólo importa, nunca modifica: `cs090_fase7_f702_escalera` (motor de swaps y selección de grafos base,
para que la batería se apoye EXACTAMENTE sobre los mismos 12 grafos de F7-02),
`cs090_fase6_o3b_rewiring` (mediciones validadas), `cs090_fase5b_phantom_adaptador` (reconstrucción de
la regla + condición inicial de masa fija), `cs090_diam_corregido`, `cs080_renormalizacion`.
Prefijo `f703`, jamás usado antes. No corre Phantom (eso es `cs090_fase7_f703_correr.py`). No declara
cierre ni veredicto.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase6_o3b_rewiring as O3B                   # sólo import (mediciones validadas)
import cs090_fase7_f702_escalera as F702                 # sólo import (motor de swaps de F7-02)
from cs090_fase5b_phantom_adaptador import (             # sólo import (pipeline validado Fase V-B)
    reconstruir_regla_a2b0c2, generar_ic_masa_fija_desde_grafo,
)

# ---------------------------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------------------------
N_NODOS = 2000
N_SWEEPS = 14
SEED_LAYOUT = 12345                  # MISMA semilla de layout que toda la Fase V-B / O3-B / F7-02

MULT_SEED_ORG = 9800                 # multiplicador NUEVO para las semillas de esta tarea.
                                     # Ya usados en la línea: 1000/2000/5000/6000/7000/7500/8000/9100/9700.

FACTOR_INTENTOS_BAJAR = F702.FACTOR_INTENTOS_BAJAR      # 10 x aristas (mismo piso que F7-02)
FACTOR_INTENTOS_SUBIR = 60           # 60 x aristas: más alto que los 40 de F7-02 porque los brazos
                                     # restringidos rechazan muchas más propuestas por atender su
                                     # criterio de organización. El techo real es el que este
                                     # presupuesto alcanza, y se reporta como tal.
PARADA_SIN_ACEPTAR = 40000           # si pasan tantos intentos seguidos sin aceptar ninguno, el brazo
                                     # saturó: ese es su techo
PACIENCIA_CUPO = 12000               # brazo `disp`: cada tantos intentos seguidos sin aceptar, se sube
                                     # el cupo de triángulos por nodo en 1 (llenado por capas)
CUPO_INICIAL_DISP = 1
CUPO_MAX_DISP = 8
PROB_BOLSA_CONC = 0.85               # brazo `conc`: probabilidad de sortear el ápice de la bolsa de
                                     # nodos "calientes" (atracción preferencial)
PROB_BOLSA_SOLAP = 0.75              # brazo `solap`: idem, para encontrar triángulos donde apoyarse
MIN_TRI_CONC = 1                     # brazo `conc`: umbral de "barrio ya caliente". Se probó con 3 en
                                     # el piloto y el brazo SATURA en 38 triángulos: con grado medio
                                     # ~3.3 un nodo entra como mucho en d(d-1)/2 triángulos, así que
                                     # exigir vecindarios ya muy densos agota el material enseguida.
                                     # Con 1 la exigencia es "pegate a un triángulo que ya existe":
                                     # sigue creciendo por barrios (cúmulos grandes y conectados) pero
                                     # llega mucho más alto. Queda registrado como techo estructural.
ARRANQUE_LIBRE = 12                  # los primeros N triángulos de conc/solap se aceptan sin exigir
                                     # el criterio (hace falta que exista algo donde apoyarse)

CHECK_GIGANTE_CADA = 20              # cada cuántas aceptaciones se mide la componente gigante
FRAC_GIGANTE_MINIMA = F702.FRAC_GIGANTE_MINIMA          # 0.97, misma restricción que F7-02

BRAZOS = ("libre", "conc", "disp", "solap", "disj")

BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_fase7_f703_organizacion"
RUTA_ESTRUCTURA = f"{HERE}/cs090_fase7_f703_estructura.csv"


# =============================================================================================
# 1) Utilidades locales de conteo (baratas: los grados son <= kcap <= 8)
# =============================================================================================
def _tri_arista(adj, a, b):
    """Nº de triángulos que contienen la arista a-b = nº de vecinos comunes."""
    return len(adj[a] & adj[b])


def _tri_nodo(adj, u):
    """Nº de triángulos que tocan al nodo u."""
    vec = sorted(adj[u])
    t = 0
    for i in range(len(vec)):
        ai = adj[vec[i]]
        for j in range(i + 1, len(vec)):
            if vec[j] in ai:
                t += 1
    return t


def _ok_disjunto(adj, x, y, p, q):
    """True si, con el swap YA aplicado, las dos aristas nuevas (x-y) y (p-q) quedan cada una en a lo
    sumo UN triángulo, y ese triángulo no comparte arista con ningún otro.

    Basta mirar las aristas nuevas y las de su entorno inmediato: sacar aristas nunca puede AUMENTAR el
    nº de triángulos de ninguna arista, así que el resto del grafo no puede violar la condición si no la
    violaba antes."""
    for a, b in ((x, y), (p, q)):
        com = adj[a] & adj[b]
        if len(com) > 1:
            return False
        for w in com:
            if len(adj[w] & adj[a]) > 1 or len(adj[w] & adj[b]) > 1:
                return False
    return True


# =============================================================================================
# 2) MOTOR: cerrar triángulos con un criterio de ORGANIZACIÓN
# =============================================================================================
def cerrar_triangulos(adj, edges, idx, N, rng, modo, t_inicial, n_intentos):
    """Corre el brazo `modo` desde el piso hasta su techo.

    Devuelve (swaps, t_por_swap, hist) donde:
      - `swaps` es la lista ordenada de swaps aceptados, cada uno la tupla (x, p, q, y) que se le pasa a
        `F702._delta_triangulos(adj, x, p, q, y)`  [saca x-p y q-y, pone x-y y q-p],
      - `t_por_swap[k]` es el nº total de triángulos DESPUÉS del swap k,
      - `hist` son los controles periódicos de conectividad: (k_swap, t, tam_gigante).

    OJO: `adj/edges/idx` quedan en el estado FINAL del brazo (el techo). El llamador reconstruye
    cualquier punto intermedio con `replay` sobre una copia limpia del piso."""
    t = t_inicial
    swaps, t_por_swap, hist = [], [], [(0, t_inicial, None)]
    bolsa = []                     # nodos "calientes": cada nodo aparece 1 vez por triángulo cerrado
    cupo = CUPO_INICIAL_DISP
    sin_aceptar = 0
    cand = np.array([i for i in range(N) if len(adj[i]) >= 2], dtype=int)
    if len(cand) == 0:
        return swaps, t_por_swap, hist
    n_prop_ok = 0

    for _ in range(n_intentos):
        sin_aceptar += 1
        if sin_aceptar > PARADA_SIN_ACEPTAR:
            break
        if modo == "disp" and sin_aceptar > PACIENCIA_CUPO and cupo < CUPO_MAX_DISP:
            cupo += 1
            sin_aceptar = 0

        # ---- elegir el ápice u ----
        if modo in ("conc", "solap") and bolsa and rng.random() < (
                PROB_BOLSA_CONC if modo == "conc" else PROB_BOLSA_SOLAP):
            u = int(bolsa[int(rng.integers(0, len(bolsa)))])
            if len(adj[u]) < 2:
                continue
        else:
            u = int(cand[int(rng.integers(0, len(cand)))])
        if modo == "disp" and _tri_nodo(adj, u) >= cupo:
            continue

        vec_u = list(adj[u])
        if len(vec_u) < 2:
            continue
        i1, i2 = rng.integers(0, len(vec_u), size=2)
        if i1 == i2:
            continue
        x, y = int(vec_u[int(i1)]), int(vec_u[int(i2)])
        if y in adj[x]:
            continue                                  # ya estaba cerrado
        if modo == "disp" and (_tri_nodo(adj, x) >= cupo or _tri_nodo(adj, y) >= cupo):
            continue

        vx = [v for v in adj[x] if v != u and v != y]
        vy = [v for v in adj[y] if v != u and v != x]
        if not vx or not vy:
            continue
        p = int(vx[int(rng.integers(0, len(vx)))])
        q = int(vy[int(rng.integers(0, len(vy)))])
        if p == q or len({x, p, y, q}) < 4:
            continue
        if (min(p, q), max(p, q)) in idx:
            continue                                  # p-q ya existe: crearía arista doble
        n_prop_ok += 1

        delta = F702._delta_triangulos(adj, x, p, q, y)   # APLICA el swap
        aceptar = delta > 0
        if aceptar and modo == "conc" and t >= ARRANQUE_LIBRE:
            aceptar = max(_tri_nodo(adj, u), _tri_nodo(adj, x), _tri_nodo(adj, y)) >= MIN_TRI_CONC
        elif aceptar and modo == "solap" and t >= ARRANQUE_LIBRE:
            aceptar = max(_tri_arista(adj, x, y), _tri_arista(adj, u, x), _tri_arista(adj, u, y)) >= 2
        elif aceptar and modo == "disj":
            aceptar = _ok_disjunto(adj, x, y, p, q)
        # `disp` ya filtró por cupo antes del swap; `libre` no filtra nada

        if aceptar:
            F702._confirmar_en_listas(edges, idx, x, p, q, y)
            t += delta
            swaps.append((x, p, q, y))
            t_por_swap.append(t)
            sin_aceptar = 0
            if modo in ("conc", "solap"):
                bolsa.extend((u, x, y))
            if len(swaps) % CHECK_GIGANTE_CADA == 0:
                hist.append((len(swaps), t, O3B.tam_gigante(adj, N)))
        else:
            F702._revertir(adj, x, p, q, y)

    hist.append((len(swaps), t, O3B.tam_gigante(adj, N)))
    return swaps, t_por_swap, hist


def replay(adj_piso, swaps, k):
    """Reconstruye el grafo del brazo tras sus primeros `k` swaps, partiendo de una copia del piso."""
    adj = [set(s) for s in adj_piso]
    for (x, p, q, y) in swaps[:k]:
        adj[x].discard(p); adj[p].discard(x)
        adj[q].discard(y); adj[y].discard(q)
        adj[x].add(y); adj[y].add(x)
        adj[q].add(p); adj[p].add(q)
    return adj


def techo_conectado(hist, gigante_minima):
    """Del historial de controles, el (k, t) más alto que todavía conserva la componente gigante."""
    mejor = (0, hist[0][1])
    for k, t, gig in hist:
        if gig is None or gig >= gigante_minima:
            if t >= mejor[1]:
                mejor = (k, t)
    return mejor


# =============================================================================================
# 3) MEDIR LA ORGANIZACIÓN de los triángulos
# =============================================================================================
def enumerar_triangulos(adj, N):
    """Lista de tríos (i<j<k) que forman triángulo. Cada triángulo aparece UNA sola vez."""
    tris = []
    for i in range(N):
        vec = sorted(v for v in adj[i] if v > i)
        for a in range(len(vec)):
            va = vec[a]
            for b in range(a + 1, len(vec)):
                if vec[b] in adj[va]:
                    tris.append((i, va, vec[b]))
    return tris


def _gini(v):
    """Coeficiente de Gini de un vector no negativo (0 = perfectamente parejo, ->1 = todo en uno)."""
    v = np.sort(np.asarray(v, dtype=float))
    n = len(v)
    s = v.sum()
    if n == 0 or s <= 0:
        return float("nan")
    return float((2.0 * np.sum((np.arange(1, n + 1)) * v) - (n + 1) * s) / (n * s))


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _bfs_dist(adj, N, origen):
    d = np.full(N, -1, dtype=int)
    d[origen] = 0
    cola = [origen]
    while cola:
        nueva = []
        for u in cola:
            du = d[u] + 1
            for v in adj[u]:
                if d[v] < 0:
                    d[v] = du
                    nueva.append(v)
        cola = nueva
    return d


def medir_organizacion(adj, N, rng, n_bfs=20, n_pares_azar=400):
    """Batería de medidas de CÓMO están repartidos los triángulos (ver docstring del módulo)."""
    tris = enumerar_triangulos(adj, N)
    n_tri = len(tris)
    m = sum(len(adj[i]) for i in range(N)) // 2
    out = dict(n_triangulos_medido=n_tri)

    tri_nodo = np.zeros(N, dtype=float)
    tri_ar = {}
    for (a, b, c) in tris:
        tri_nodo[a] += 1; tri_nodo[b] += 1; tri_nodo[c] += 1
        for e in ((a, b), (a, c), (b, c)):
            tri_ar[e] = tri_ar.get(e, 0) + 1

    nodos_sop = int((tri_nodo > 0).sum())
    ar_sop = len(tri_ar)
    ar_multi = sum(1 for v in tri_ar.values() if v >= 2)
    out["frac_nodos_en_triangulo"] = nodos_sop / N
    out["frac_aristas_en_triangulo"] = ar_sop / m if m else float("nan")
    out["frac_aristas_multi_tri"] = (ar_multi / ar_sop) if ar_sop else float("nan")
    out["tri_por_arista_media"] = (sum(tri_ar.values()) / ar_sop) if ar_sop else float("nan")
    out["tri_por_nodo_max"] = float(tri_nodo.max()) if n_tri else 0.0
    out["gini_tri_nodo"] = _gini(tri_nodo) if n_tri else float("nan")

    # --- cúmulos de triángulos: dos triángulos conectados si comparten al menos un NODO ---
    if n_tri:
        uf = _UF(n_tri)
        por_nodo = {}
        for ti, (a, b, c) in enumerate(tris):
            for v in (a, b, c):
                por_nodo.setdefault(v, []).append(ti)
        for v, lst in por_nodo.items():
            for ti in lst[1:]:
                uf.union(lst[0], ti)
        raiz = {}
        for ti in range(n_tri):
            raiz.setdefault(uf.find(ti), []).append(ti)
        tam = sorted((len(v) for v in raiz.values()), reverse=True)
        out["n_comp_tri"] = len(tam)
        out["tam_mayor_comp_tri"] = tam[0]
        out["frac_mayor_comp_tri"] = tam[0] / n_tri
        out["tam_medio_comp_tri"] = float(np.mean(tam))

        # --- modularidad de Newman de la partición que inducen esos cúmulos sobre los NODOS ---
        com_de_nodo = {}
        for r, lst in raiz.items():
            for ti in lst:
                for v in tris[ti]:
                    com_de_nodo[v] = r
        L_c, d_c = {}, {}
        for i in range(N):
            ci = com_de_nodo.get(i)
            if ci is None:
                continue
            d_c[ci] = d_c.get(ci, 0) + len(adj[i])
            for j in adj[i]:
                if j > i and com_de_nodo.get(j) == ci:
                    L_c[ci] = L_c.get(ci, 0) + 1
        Q = 0.0
        for ci in d_c:
            Q += L_c.get(ci, 0) / m - (d_c[ci] / (2.0 * m)) ** 2
        out["modularidad_tri"] = float(Q)

        # --- distancia media entre triángulos distintos (en saltos del grafo) ---
        k = min(n_bfs, n_tri)
        sel = rng.choice(n_tri, size=k, replace=False)
        dists = []
        for ti in sel:
            o = tris[int(ti)][0]
            d = _bfs_dist(adj, N, o)
            otros = [tris[j][0] for j in range(n_tri) if j != int(ti)]
            if len(otros) > 200:
                otros = [otros[int(z)] for z in rng.choice(len(otros), size=200, replace=False)]
            vals = [d[v] for v in otros if d[v] >= 0]
            if vals:
                dists.append(float(np.mean(vals)))
        out["dist_media_tri"] = float(np.mean(dists)) if dists else float("nan")
    else:
        for c in ("n_comp_tri", "tam_mayor_comp_tri", "frac_mayor_comp_tri", "tam_medio_comp_tri",
                  "modularidad_tri", "dist_media_tri"):
            out[c] = float("nan")

    # vara de comparación: distancia media entre nodos al azar del mismo grafo
    sel = rng.choice(N, size=min(12, N), replace=False)
    dz = []
    for o in sel:
        d = _bfs_dist(adj, N, int(o))
        vals = d[d >= 0]
        if len(vals) > 1:
            dz.append(float(vals.sum() / (len(vals) - 1)))
    out["dist_media_azar"] = float(np.mean(dz)) if dz else float("nan")
    return out


# =============================================================================================
# 4) PROCESAR un grafo base: piso común -> 5 brazos -> igualar triángulos -> medir -> IC
# =============================================================================================
def procesar_una(sel, generar_ic=True):
    rid, seed = sel["rule_id"], sel["seed"]
    t_ini = time.time()

    p, m = reconstruir_regla_a2b0c2(seed=seed, N=N_NODOS, n_sweeps=N_SWEEPS)
    adj_orig = O3B._a_lista(m["adj_final"], N_NODOS)
    E_orig = O3B.aristas_set(adj_orig, N_NODOS)
    g_orig = O3B.grados(adj_orig, N_NODOS)
    gig_orig = O3B.tam_gigante(adj_orig, N_NODOS)
    c_orig, tr_orig, t_orig = O3B.clustering(adj_orig, N_NODOS)
    piso_gigante = FRAC_GIGANTE_MINIMA * gig_orig

    # ---- piso común E0: se destruyen los triángulos (misma función y dosis que F7-02) ----
    semilla_base = int(seed) * MULT_SEED_ORG + 17
    rng_piso = np.random.default_rng(semilla_base)
    adj_p, edges_p, idx_p = F702._estado_desde_adj(adj_orig, N_NODOS)
    n_aristas = len(edges_p)
    t0 = time.time()
    d_baja, acc_baja = F702.bajar_clustering(adj_p, edges_p, idx_p, N_NODOS, rng_piso,
                                             FACTOR_INTENTOS_BAJAR * n_aristas)
    t_piso = t_orig + d_baja
    adj_piso = [set(s) for s in adj_p]
    t_bajar_s = time.time() - t0
    print(f"    piso: tri {t_orig} -> {t_piso}  ({acc_baja} swaps, {t_bajar_s:.1f}s)", flush=True)

    # ---- cada brazo, desde el MISMO piso, hasta su techo ----
    brazos = {}
    for j, modo in enumerate(BRAZOS):
        rng_b = np.random.default_rng(semilla_base + 101 * (j + 1))
        adj_b, edges_b, idx_b = F702._estado_desde_adj(adj_piso, N_NODOS)
        tb = time.time()
        swaps, t_por_swap, hist = cerrar_triangulos(
            adj_b, edges_b, idx_b, N_NODOS, rng_b, modo, t_piso,
            FACTOR_INTENTOS_SUBIR * n_aristas)
        k_techo, t_techo = techo_conectado(hist, piso_gigante)
        brazos[modo] = dict(swaps=swaps, t_por_swap=t_por_swap, hist=hist,
                            k_techo=k_techo, t_techo=t_techo,
                            t_techo_sin_restriccion=(t_por_swap[-1] if t_por_swap else t_piso),
                            n_aceptados=len(swaps), t_swaps_s=round(time.time() - tb, 1))
        print(f"    {modo:>5}: techo_conectado tri={t_techo} (k={k_techo}/{len(swaps)}) "
              f"techo_libre={brazos[modo]['t_techo_sin_restriccion']} [{brazos[modo]['t_swaps_s']}s]",
              flush=True)

    # ---- T* = el mínimo de los cinco techos; cada brazo se rebobina al swap más cercano ----
    T_obj = min(brazos[mo]["t_techo"] for mo in BRAZOS)
    filas = []
    tris_conseguidos = {}
    for modo in BRAZOS:
        b = brazos[modo]
        tps, k_max = b["t_por_swap"], b["k_techo"]
        if k_max == 0:
            k_mejor = 0
        else:
            k_mejor = min(range(1, k_max + 1), key=lambda k: (abs(tps[k - 1] - T_obj), k))
        adj_v = replay(adj_piso, b["swaps"], k_mejor)
        b["k_elegido"], b["adj"] = k_mejor, adj_v
        tris_conseguidos[modo] = (tps[k_mejor - 1] if k_mejor else t_piso)
    dif_max = max(tris_conseguidos.values()) - min(tris_conseguidos.values())
    print(f"    T*={T_obj}  conseguidos={tris_conseguidos}  dif_max={dif_max}", flush=True)

    adjs_null = O3B.nulls_topo_de(seed, N_NODOS, len(E_orig))
    rng_med = np.random.default_rng(semilla_base + 7)

    for modo in BRAZOS:
        b = brazos[modo]
        adj_v = b["adj"]
        # --- VERIFICACIÓN NUMÉRICA nodo por nodo (no se asume) ---
        g_v = O3B.grados(adj_v, N_NODOS)
        iguales = bool(np.array_equal(g_orig, g_v))
        n_dif = int((g_orig != g_v).sum())
        assert iguales, f"{rid}/{modo}: la secuencia de grados NO se conservó ({n_dif} nodos difieren)"
        E_v = O3B.aristas_set(adj_v, N_NODOS)
        assert len(E_v) == len(E_orig), f"{rid}/{modo}: cambió el nº de aristas"
        assert all(i not in adj_v[i] for i in range(N_NODOS)), f"{rid}/{modo}: hay un bucle i-i"

        cl, tr, ntri = O3B.clustering(adj_v, N_NODOS)
        org = medir_organizacion(adj_v, N_NODOS, rng_med)
        assert org["n_triangulos_medido"] == ntri, f"{rid}/{modo}: dos conteos de triángulos distintos"
        pc = O3B.pendiente_corregida_de_grafo(O3B.canonicalizar(adj_v, N_NODOS), N_NODOS, seed, adjs_null)

        fila = dict(
            rule_id=rid, seed=seed, lote=sel["lote"], K=sel["K"], kcap=sel["kcap"],
            brazo=modo, n_nodos=N_NODOS, n_aristas=len(E_v),
            grado_medio=2.0 * len(E_v) / N_NODOS,
            grados_identicos=iguales, n_nodos_grado_distinto=n_dif,
            solapamiento_aristas=len(E_orig & E_v) / len(E_orig),
            clustering_local=cl, transitividad=tr, n_triangulos=ntri,
            T_objetivo=T_obj, dif_max_triangulos=dif_max,
            gigante=O3B.tam_gigante(adj_v, N_NODOS),
            n_componentes=F702.n_componentes(adj_v, N_NODOS),
            asortatividad=F702.asortatividad_grados(adj_v, N_NODOS),
            pendiente_corr=pc["pendiente"], z_agg=pc["z_agg"], diams=pc["diams"], n_cajas=pc["n_cajas"],
            clustering_original=c_orig, transitividad_original=tr_orig, n_triangulos_original=t_orig,
            t_piso=t_piso, t_techo_brazo=b["t_techo"],
            t_techo_brazo_sin_restriccion=b["t_techo_sin_restriccion"],
            n_swaps_aceptados=b["n_aceptados"], k_elegido=b["k_elegido"],
            t_swaps_s=b["t_swaps_s"], acc_baja=acc_baja, t_bajar_s=round(t_bajar_s, 1),
            gigante_original=gig_orig, semilla_base=semilla_base,
            pendiente_corr_csv_ref=sel["pendiente_corregida"],
            frac_masa_fase5b=sel["frac_masa_fase5b"],
        )
        fila.update(org)

        if generar_ic:
            carpeta = f"{BASE_SALIDA}/{rid}_s{seed}_f703_{modo}"
            os.makedirs(carpeta, exist_ok=True)
            t1 = time.time()
            info = generar_ic_masa_fija_desde_grafo(adj_v, N=N_NODOS, seed_layout=SEED_LAYOUT,
                                                    ruta_salida=f"{carpeta}/cosmogenesis_ic.txt")
            fila["t_ic_s"] = round(time.time() - t1, 2)
            meta = dict(
                tarea="FASE7_F703_organizacion_triangulos", brazo=modo, rule_id=rid, clase="III",
                seed=seed, lote=sel["lote"], N=N_NODOS, seed_layout=SEED_LAYOUT,
                K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                sim_thr_frac=p["sim_thr_frac"],
                n_aristas_grafo_final=len(E_v), grado_medio_grafo_final=2.0 * len(E_v) / N_NODOS,
                grados_identicos_al_original=iguales,
                semilla_base=semilla_base, solapamiento_aristas=fila["solapamiento_aristas"],
                clustering_local=cl, transitividad=tr, n_triangulos=ntri,
                T_objetivo=T_obj, dif_max_triangulos=dif_max,
                gigante=fila["gigante"], pendiente_corregida=pc["pendiente"],
                frac_aristas_multi_tri=org["frac_aristas_multi_tri"],
                gini_tri_nodo=org["gini_tri_nodo"], modularidad_tri=org["modularidad_tri"],
                masa_total_ic=info["masa_total"], carpeta=carpeta,
            )
            with open(f"{carpeta}/meta_regla.json", "w") as f:
                json.dump(meta, f, indent=2)
            fila["carpeta"] = carpeta

        filas.append(fila)
        print(f"    {modo:>5}  tri={ntri:<5} C={cl:.5f} trans={tr:.5f} multi={org['frac_aristas_multi_tri']:.3f} "
              f"gini={org['gini_tri_nodo']:.3f} Q={org['modularidad_tri']:.4f} "
              f"nodos={org['frac_nodos_en_triangulo']:.3f} comp={org['n_comp_tri']} "
              f"gig={fila['gigante']} pend={pc['pendiente']:.4f}"
              + (f" ic={fila.get('t_ic_s')}s" if generar_ic else ""), flush=True)

    for f in filas:
        f["t_total_grafo_s"] = round(time.time() - t_ini, 1)
    return filas


def main(indices=None, generar_ic=True, sufijo_csv="", n_por_lote=3):
    elegidos = F702.seleccionar_grafos(n_por_lote=n_por_lote)   # MISMOS 12 grafos base que F7-02
    if indices is not None:
        elegidos = [elegidos[i] for i in indices]
    print(f"[f703] {len(elegidos)} grafos base (generar_ic={generar_ic})", flush=True)

    todas, t0 = [], time.time()
    for k, sel in enumerate(elegidos):
        print(f"[{k+1}/{len(elegidos)}] {sel['rule_id']} seed={sel['seed']} lote={sel['lote']}",
              flush=True)
        todas.extend(procesar_una(sel, generar_ic=generar_ic))

    ruta = RUTA_ESTRUCTURA.replace(".csv", f"{sufijo_csv}.csv")
    with open(ruta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(todas[0].keys()))
        w.writeheader()
        w.writerows(todas)
    print(f"\n[f703] {len(todas)} filas -> {ruta.split('/')[-1]}  (total {time.time()-t0:.0f}s)")
    return todas


if __name__ == "__main__":
    # uso: python3.9 cs090_fase7_f703_organizacion.py [idx0,idx1,...] [--sin-ic] [--sufijo=_s0]
    idxs, gen_ic, suf = None, True, ""
    for arg in sys.argv[1:]:
        if arg == "--sin-ic":
            gen_ic = False
        elif arg.startswith("--sufijo="):
            suf = arg.split("=", 1)[1]
        else:
            idxs = [int(x) for x in arg.split(",")]
    main(indices=idxs, generar_ic=gen_ic, sufijo_csv=suf)
