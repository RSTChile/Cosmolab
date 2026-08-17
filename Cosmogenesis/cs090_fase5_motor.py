"""
CS090 — FASE V-A (piloto): MOTOR DE EJECUCIÓN (liviano, N=500-2000)
======================================================================
QUIÉN SOY: ejecuta UNA regla generada por cs090_fase5_generador.py — construye el sustrato según el
Eje A, corre la dinámica según el Eje B, aplica costo/poda según el Eje C, y mide diámetro/componente
gigante/holonomía en REAL y en dos controles NULL emparejados (nunca se clasifica sin comparar contra
null, disciplina anti-Shannon del proyecto):

  - NULL_topo: misma N y misma densidad de aristas, pero grafo Erdős-Rényi fresco (independiente) —
    aísla si la TOPOLOGÍA que produjo la dinámica es distinta de azar con la misma densidad. Mismo
    patrón que el brazo "er_null" de cs080_renormalizacion.py.
  - NULL_valor: MISMA topología final que REAL, pero estados reemplazados por ruido i.i.d. en Z_K —
    aísla si los VALORES se organizaron más allá del azar. Mismo patrón que `null_de` de
    cs082_fase4_4sustratos.py.

Piezas reusadas TAL CUAL de los 7 scripts congelados (sólo import, nunca se tocan):
  `_circ_mean_update`, `_linea_adyacencia`, `_holonomia_triangulos` (cs082) · `_diam`, `_giant` (cs055,
  vía import directo) · `aleatorio()` (cg003).

SIMPLIFICACIONES DECLARADAS respecto a los scripts de referencia (motor "aligerado" para el piloto,
tal como pide el plan aprobado — se documentan para que no se confundan con descuido):
  - Costo C1/C2 usa 2 de los 4 componentes de cs081 (inconsistencia histórica + conflicto de
    holonomía), no los 4 — pesos iguales 1/2. Poda a UN percentil fijo (P70), no barrido P50/70/90.
  - Para A0 (campo en anillo, sin grafo) la medición de diám/giant se hace sobre un grafo DERIVADO por
    similitud de estado (dos sitios se conectan si su diferencia circular < umbral), muestreado sobre
    un candidato acotado por nodo (no todos los pares — evita O(N²)). Este grafo es SÓLO para medir/
    clasificar, nunca alimenta la dinámica del campo (que sigue siendo difusión local en anillo).
  - A2 (grafo dinámico co-emergente) implementa una regla de recableo LIGERA: cada pocos sweeps, una
    muestra de aristas "caras" (estados muy distintos en sus extremos) se cae y se reemplaza por
    aristas entre nodos con estados más compatibles — sin coordenadas, sólo diferencia de estado.

Guardián no-hornear: en ningún punto de este archivo hay un arreglo de coordenadas (x,y,z); los
vecinos se leen siempre de adyacencia o de offsets fijos de anillo. No se toca ningún script congelado.
No declara cierre — reporta números, veredicto es de Alexis.
"""
from __future__ import annotations
import sys, time
import numpy as np
from collections import defaultdict, deque

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cg003_diagnostico_gromov as GR              # aleatorio() -- ER, reusado tal cual
import cs082_fase4_4sustratos as C82               # _circ_mean_update, _linea_adyacencia, _holonomia_triangulos


def _extraer_funciones(archivo, nombres):
    """cs055_proceso_acoplado.py llama a main() SIN guardia `if __name__=="__main__"` (línea 278,
    verificado por inspección) -- un `import` normal correría TODA su batería de experimento como
    efecto secundario. Para reusar _diam/_giant TAL CUAL sin tocar el archivo congelado ni disparar
    esa corrida, se extraen sólo esas dos funciones por AST y se ejecutan en un namespace aislado
    (mismo bytecode que tendría un import normal de esas funciones, ningún carácter del archivo
    original se modifica)."""
    import ast
    ruta = f"{_HERE}/{archivo}"
    with open(ruta) as fh:
        fuente = fh.read()
    arbol = ast.parse(fuente)
    ns = {"np": np, "deque": deque}
    for nodo in arbol.body:
        if isinstance(nodo, ast.FunctionDef) and nodo.name in nombres:
            modulo_fn = ast.Module(body=[nodo], type_ignores=[])
            exec(compile(modulo_fn, ruta, "exec"), ns)
    return ns


_ns55 = _extraer_funciones("cs055_proceso_acoplado.py", {"_diam", "_giant"})
_diam, _giant = _ns55["_diam"], _ns55["_giant"]


# ============================================================================================
# 1) CONSTRUCCIÓN DE SUSTRATO por Eje A
# ============================================================================================
def construir_A0(N, rng, p):
    """A0 — SIN GRAFO: campo continuo de fase en un anillo fijo (offsets left/right, ningún objeto de
    adyacencia tipo grafo en ningún punto). Representación no-grafo genuina (§2 salvaguarda)."""
    S = rng.uniform(0, p["K"], N)
    left = (np.arange(N) - 1) % N
    right = (np.arange(N) + 1) % N
    return dict(kind="A0", N=N, S=S, left=left, right=right)


def construir_A1(N, rng, p):
    """A1 — grafo fijo generado al azar (Erdős-Rényi vía aleatorio(), reusada tal cual)."""
    adj0, _ = GR.aleatorio(N, meandeg=p["meandeg"], seed=int(rng.integers(1 << 30)))
    adj = [set(a.tolist()) for a in adj0]
    return dict(kind="A1", N=N, adj=adj)


def construir_A2(N, rng, p):
    """A2 — grafo dinámico: mismo generador inicial que A1, pero se marca para co-evolucionar con los
    estados durante la dinámica (ver _recablear_A2 en dinamica_B0/B1)."""
    adj0, _ = GR.aleatorio(N, meandeg=p["meandeg"], seed=int(rng.integers(1 << 30)))
    adj = [set(a.tolist()) for a in adj0]
    return dict(kind="A2", N=N, adj=adj)


CONSTRUCTORES_A = {"A0": construir_A0, "A1": construir_A1, "A2": construir_A2}


# ============================================================================================
# 2) RECABLEO A2 (co-emergencia liviana, sin coordenadas — sólo diferencia de estado)
# ============================================================================================
def _recablear_A2(adj, S_por_nodo, K, rng, n_intentos=None):
    """Cada llamada: muestrea una fracción chica de aristas existentes; si el estado de sus extremos
    difiere mucho (más de K/3), la arista se cae; se intenta reponer con una arista nueva entre dos
    nodos al azar cuyos estados sean más compatibles (se prueban unos pocos candidatos y se queda con
    el mejor) — la topología emerge de la diferencia de estado, nunca de coordenadas."""
    N = len(adj)
    edges = [(i, j) for i in range(N) for j in adj[i] if j > i]
    if not edges:
        return
    n_intentos = n_intentos or max(1, len(edges) // 20)
    idx = rng.choice(len(edges), size=min(n_intentos, len(edges)), replace=False)
    for t in idx:
        i, j = edges[t]
        d = abs(S_por_nodo[i] - S_por_nodo[j]) % K
        d = min(d, K - d)
        if d > K / 3.0:
            adj[i].discard(j); adj[j].discard(i)
            mejor, mejor_d = None, 1e9
            cand = rng.integers(0, N, size=6)
            a = cand[0]
            for b in cand[1:]:
                if a == b or b in adj[a]:
                    continue
                db = abs(S_por_nodo[a] - S_por_nodo[b]) % K
                db = min(db, K - db)
                if db < mejor_d:
                    mejor, mejor_d = b, db
            if mejor is not None:
                adj[a].add(mejor); adj[mejor].add(a)


def _enforce_kcap(adj, N, kcap):
    """C2 — límite de escala duro: ningún nodo conserva más de `kcap` vecinos; si excede, se quedan
    los `kcap` de mayor soporte local (vecinos compartidos), MISMO criterio de ranking que
    gate_localidad/cs081 (ya validado), aligerado."""
    for i in range(N):
        nb = list(adj[i])
        if len(nb) <= kcap:
            continue
        sup = sorted(((len(adj[i] & adj[j]), j) for j in nb), reverse=True)
        mantener = set(j for _, j in sup[:kcap])
        for j in nb:
            if j not in mantener:
                adj[i].discard(j); adj[j].discard(i)


# ============================================================================================
# 3) COSTO LIVIANO (C1/C2) — 2 de los 4 componentes de cs081, pesos iguales, poda a P70 fijo
# ============================================================================================
def _costo_y_podar(edges, flip_count, E_estado, K, triangles, P=70.0):
    """Costo = 0.5*z(inconsistencia histórica) + 0.5*z(conflicto de holonomía). Poda las aristas con
    costo por encima del percentil P. Devuelve el set de aristas a CONSERVAR."""
    if not edges:
        return set()
    hist = np.array([flip_count.get(e, 0) for e in edges], dtype=float)
    hol_por_arista = defaultdict(list)
    for (i, j, k) in triangles:
        eij = E_estado.get((i, j)) if i < j else E_estado.get((j, i))
        ejk = E_estado.get((j, k)) if j < k else E_estado.get((k, j))
        eki = E_estado.get((k, i)) if k < i else E_estado.get((i, k))
        if eij is None or ejk is None or eki is None:
            continue
        h = (eij + ejk + eki) % K
        h = abs(h - K if h > K / 2 else h)
        for e in ((i, j) if i < j else (j, i), (j, k) if j < k else (k, j), (i, k) if i < k else (k, i)):
            hol_por_arista[e].append(h)
    media_global = float(np.mean([v for vs in hol_por_arista.values() for v in vs])) if hol_por_arista else 0.0
    hol = np.array([float(np.mean(hol_por_arista[e])) if e in hol_por_arista else media_global for e in edges])

    def z(a):
        mu, sd = a.mean(), a.std()
        return (a - mu) / sd if sd > 1e-9 else np.zeros_like(a)

    costo = 0.5 * z(hist) + 0.5 * z(hol)
    umbral = np.percentile(costo, P)
    conservar = set(e for e, c in zip(edges, costo) if c <= umbral)
    return conservar


# ============================================================================================
# 4) DINÁMICA — Eje B0 (entidad-entidad, estado en NODOS o en el campo de anillo A0)
# ============================================================================================
def dinamica_B0(sustrato, p, rng, n_sweeps, costo_nivel):
    kind = sustrato["kind"]
    K, J, noise = p["K"], p["J"], p["noise"]
    if kind == "A0":
        S, left, right = sustrato["S"], sustrato["left"], sustrato["right"]
        for _ in range(n_sweeps):
            ang = 2 * np.pi * S / K
            z = np.exp(1j * ang)
            z_vec = (1 - J) * z + J * 0.5 * (z[left] + z[right])
            S = (np.angle(z_vec) / (2 * np.pi) * K) % K
            S = (S + rng.normal(0, noise, len(S))) % K
        sustrato["S"] = S
        return sustrato
    # A1 / A2: estado en NODOS, vecinos = adyacencia del grafo
    adj = sustrato["adj"]; N = sustrato["N"]
    S = rng.uniform(0, K, N)
    flip_count = defaultdict(int)
    prev_edges = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])
    for step in range(n_sweeps):
        vecinos = [list(a) for a in adj]
        S = C82._circ_mean_update(S, vecinos, J, noise, rng)
        if kind == "A2" and step % 3 == 0:
            _recablear_A2(adj, S, K, rng)
        if costo_nivel == "C2" and step % 4 == 0:
            _enforce_kcap(adj, N, p["kcap"])
        if costo_nivel in ("C1", "C2"):
            cur = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])
            for e in prev_edges ^ cur:
                flip_count[e] += 1
            prev_edges = cur
    sustrato["S"] = S; sustrato["adj"] = adj; sustrato["flip_count"] = flip_count
    if costo_nivel in ("C1", "C2"):
        edges = sorted(prev_edges)
        E_estado = {}
        for (i, j) in edges:
            E_estado[(i, j)] = abs(S[i] - S[j]) % K
        triangles = _muestrear_triangulos(adj, N, rng)
        conservar = _costo_y_podar(edges, flip_count, E_estado, K, triangles)
        for (i, j) in edges:
            if (i, j) not in conservar:
                adj[i].discard(j); adj[j].discard(i)
    return sustrato


# ============================================================================================
# 5) DINÁMICA — Eje B1 (relación-sobre-relación: estado en ARISTAS, vecindad = línea de adyacencia)
# ============================================================================================
def dinamica_B1(sustrato, p, rng, n_sweeps, costo_nivel):
    K, J, noise = p["K"], p["J"], p["noise"]
    adj = sustrato["adj"]; N = sustrato["N"]; kind = sustrato["kind"]
    edges = sorted(set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i]))
    if not edges:
        sustrato["E"] = {}; sustrato["adj"] = adj
        return sustrato
    idx_edge = {e: k for k, e in enumerate(edges)}
    s = rng.uniform(0, K, len(edges))
    flip_count = defaultdict(int)
    prev_edges = set(edges)
    for step in range(n_sweeps):
        vecinos = C82._linea_adyacencia(edges)
        s = C82._circ_mean_update(s, vecinos, J, noise, rng)
        if kind == "A2" and step % 3 == 0:
            S_nodo = np.zeros(N); cnt = np.zeros(N)
            for k, (i, j) in enumerate(edges):
                S_nodo[i] += s[k]; cnt[i] += 1; S_nodo[j] += s[k]; cnt[j] += 1
            S_nodo = np.divide(S_nodo, np.maximum(cnt, 1))
            _recablear_A2(adj, S_nodo, K, rng)
            nuevos_edges = sorted(set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i]))
            nuevo_idx = {}
            s_nuevo = np.zeros(len(nuevos_edges))
            for k2, e in enumerate(nuevos_edges):
                if e in idx_edge:
                    s_nuevo[k2] = s[idx_edge[e]]
                else:
                    s_nuevo[k2] = rng.uniform(0, K)
                nuevo_idx[e] = k2
            edges, idx_edge, s = nuevos_edges, nuevo_idx, s_nuevo
        if costo_nivel in ("C1", "C2"):
            cur = set(edges)
            for e in prev_edges ^ cur:
                flip_count[e] += 1
            prev_edges = cur
    E_estado = {e: s[idx_edge[e]] for e in edges}
    sustrato["E"] = E_estado; sustrato["edges"] = edges; sustrato["flip_count"] = flip_count
    if costo_nivel in ("C1", "C2"):
        triangles = _muestrear_triangulos(adj, N, rng)
        conservar = _costo_y_podar(edges, flip_count, E_estado, K, triangles)
        adj2 = [set() for _ in range(N)]
        for (i, j) in conservar:
            adj2[i].add(j); adj2[j].add(i)
        sustrato["adj"] = adj2
        sustrato["E"] = {e: v for e, v in E_estado.items() if e in conservar}
        sustrato["edges"] = sorted(conservar)
    else:
        sustrato["adj"] = adj
    return sustrato


DINAMICAS_B = {"B0": dinamica_B0, "B1": dinamica_B1}


def _muestrear_triangulos(adj, N, rng, max_tri=1500):
    """Lista de triángulos, muestreada (no enumeración exhaustiva) para mantener el motor liviano a
    N=2000: recorre nodos en orden al azar, revisa pares de vecinos hasta juntar max_tri triángulos."""
    tri = []
    orden = list(range(N)); rng.shuffle(orden)
    for i in orden:
        vs = sorted(adj[i])
        for a in range(len(vs)):
            if len(tri) >= max_tri:
                return tri
            for b in range(a + 1, len(vs)):
                if vs[b] in adj[vs[a]]:
                    tri.append((i, vs[a], vs[b]))
    return tri


# ============================================================================================
# 6) MEDICIÓN — grafo de análisis (derivado para A0) + diám/giant/holonomía
# ============================================================================================
def _grafo_medicion_A0(S, K, thr_frac, rng, n_cand=15):
    """A0 no tiene grafo de dinámica -- para PODER clasificar (Clase I-IV hablan de diám vs N), se
    deriva un grafo de MEDICIÓN por similitud de estado: cada sitio se compara contra n_cand
    candidatos al azar (no todos los pares, evita O(N²)) y se conecta si la diferencia circular está
    bajo el umbral. Este grafo es sólo para medir -- la dinámica del campo nunca lo usa."""
    N = len(S)
    adj = [set() for _ in range(N)]
    thr = thr_frac * K
    cand = rng.integers(0, N, size=(N, n_cand))
    for i in range(N):
        for j in cand[i]:
            j = int(j)
            if j == i:
                continue
            d = abs(S[i] - S[j]) % K
            d = min(d, K - d)
            if d < thr:
                adj[i].add(j); adj[j].add(i)
    return adj


def medir(sustrato, p, rng):
    """Devuelve dict(diam, giant, holonomia, n_aristas, adj_final) del sustrato YA evolucionado."""
    kind = sustrato["kind"]; N = sustrato["N"]
    if kind == "A0":
        adj = _grafo_medicion_A0(sustrato["S"], p["K"], p["sim_thr_frac"], rng)
        E_estado = {}
        edges = sorted(set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i]))
        for (i, j) in edges:
            E_estado[(i, j)] = abs(sustrato["S"][i] - sustrato["S"][j]) % p["K"]
    else:
        adj = sustrato["adj"]
        edges = sorted(set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i]))
        if "E" in sustrato:      # B1: estado nativo en aristas
            E_estado = sustrato["E"]
        else:                    # B0: estado en nodos, se proyecta a diferencia de arista
            S = sustrato["S"]
            E_estado = {(i, j): abs(S[i] - S[j]) % p["K"] for (i, j) in edges}
    diam = float(_diam(adj, N)) if edges else float("nan")
    giant = float(_giant(adj, N)) if edges else 0.0
    triangles = _muestrear_triangulos(adj, N, rng) if edges else []
    hol = C82._holonomia_triangulos(E_estado, triangles) if triangles else np.array([0.0])
    return dict(diam=diam, giant=giant, holonomia=float(np.mean(hol)), n_aristas=len(edges),
                adj_final=adj, E_final=E_estado, n_triangulos=len(triangles))


# ============================================================================================
# 7) NULLS EMPAREJADOS
# ============================================================================================
def medir_null_topo(N, p, rng, n_aristas_objetivo):
    """NULL_topo: Erdős-Rényi fresco con MISMA N y MISMA densidad de aristas que REAL (independiente).
    Aísla si la topología producida por la dinámica es distinta de azar con igual densidad."""
    meandeg_equiv = max(0.5, 2.0 * n_aristas_objetivo / max(1, N))
    adj0, _ = GR.aleatorio(N, meandeg=meandeg_equiv, seed=int(rng.integers(1 << 30)))
    adj = [set(a.tolist()) for a in adj0]
    diam = float(_diam(adj, N)) if any(adj) else float("nan")
    giant = float(_giant(adj, N))
    return dict(diam=diam, giant=giant, n_aristas=sum(len(a) for a in adj) // 2)


def medir_null_valor(sustrato_medido, p, rng):
    """NULL_valor: MISMA topología final que REAL, estados reemplazados por ruido i.i.d en Z_K (mismo
    patrón que null_de de cs082). Aísla si los VALORES (holonomía) se organizaron más allá del azar."""
    E_real = sustrato_medido["E_final"]
    E_null = {e: rng.uniform(0, p["K"]) for e in E_real}
    triangles = _muestrear_triangulos(sustrato_medido["adj_final"], len(sustrato_medido["adj_final"]), rng)
    hol = C82._holonomia_triangulos(E_null, triangles) if triangles else np.array([0.0])
    return dict(holonomia=float(np.mean(hol)))


# ============================================================================================
# 8) CORRER UNA REGLA COMPLETA a través de las 4 N del piloto (500/1000/1500/2000)
# ============================================================================================
def correr_regla(p, Ns=(500, 1000, 1500, 2000), n_sweeps=14, n_seeds_null_topo=3):
    filas = []
    for N in Ns:
        rng = np.random.default_rng(p["seed"] * 1000 + N)
        sustrato = CONSTRUCTORES_A[p["eje_A"]](N, rng, p)
        t0 = time.time()
        sustrato = DINAMICAS_B[p["eje_B"]](sustrato, p, rng, n_sweeps, p["eje_C"])
        m = medir(sustrato, p, rng)
        dt = time.time() - t0

        # NULL_topo: varios seeds independientes, misma N y densidad -> media y std para z-score
        nt = [medir_null_topo(N, p, np.random.default_rng(p["seed"] * 2000 + N * 7 + s), m["n_aristas"])
              for s in range(n_seeds_null_topo)]
        diam_null_mean = float(np.nanmean([x["diam"] for x in nt]))
        diam_null_std = float(np.nanstd([x["diam"] for x in nt])) + 1e-9

        # NULL_valor: misma topología, valores ruido -> holonomía null
        nv = medir_null_valor(m, p, np.random.default_rng(p["seed"] * 3000 + N))

        filas.append(dict(
            rule_id=p["rule_id"], N=N, dt=round(dt, 2),
            diam_real=m["diam"], giant_real=m["giant"], holon_real=m["holonomia"],
            n_aristas=m["n_aristas"], n_triangulos=m["n_triangulos"],
            diam_null_topo=diam_null_mean, diam_null_topo_std=diam_null_std,
            holon_null_valor=nv["holonomia"],
        ))
    return filas


# ============================================================================================
# 9) MÉTODO CORREGIDO — pendiente por COARSE-GRAINING de UN sólo grafo (reusa cs080, sin tocarlo)
# ============================================================================================
# HALLAZGO (ver FASE5A_piloto_resultado_CS.md §4): correr_regla() de arriba mide diám vs N sobre
# grafos INDEPENDIENTES generados a cada N (500/1000/1500/2000) -- pero los umbrales 0.35-0.45 (Clase
# II) / >0.7-0.8 (Clase III) de la especificación se calibraron en cs080_renormalizacion.py agrupando
# (coarse-graining, box-counting BFS) UN MISMO grafo en escalas sucesivas b=1,2,4,8,16, midiendo diám
# vs N_b (nº de supernodos). Es una vara distinta -> pendientes distintas del mismo sustrato (0.11-0.36
# por N-independiente vs 0.459 por coarse-graining, diagnóstico verificado sobre A1-B0-C0 N=2000).
# correr_regla_coarse() reemplaza el método de medición: UN grafo por regla al N más grande razonable
# (default 2000), coarse-graneado con `cajas_bfs`/`grafo_grueso` de cs080_renormalizacion.py (import
# directo, ESE ARCHIVO NO SE TOCA) a las mismas escalas con que se calibraron los umbrales. Devuelve
# "filas" con el MISMO esquema de campos que correr_regla() (rule_id, N, diam_real, diam_null_topo,
# diam_null_topo_std, holon_real, holon_null_valor, ...) -- así cs090_fase5_clasificador.py se reusa
# SIN NINGÚN CAMBIO (Ns pasa a ser N_b por escala, no N por regeneración).
#
# SIMPLIFICACIÓN DECLARADA (no disfrazada): la holonomía REAL/NULL_valor se mide UNA sola vez, a la
# resolución nativa (b=1, sin agrupar), y se replica en las filas de las demás escalas. El bug
# diagnosticado es específico de diám-vs-N (que sí se corrige en cada escala, incluido el z-score
# REAL-vs-NULL_topo que de eso depende); la holonomía no formaba parte del diagnóstico y cambiar
# también su método sin evidencia de que esté roto sería un cambio no pedido.
def correr_regla_coarse(p, N=2000, n_sweeps=14, escalas_b=(1, 2, 4, 8, 16), n_seeds_null_topo=3):
    import cs080_renormalizacion as CS80  # sólo import -- cajas_bfs/grafo_grueso TAL CUAL, sin tocar

    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = CONSTRUCTORES_A[p["eje_A"]](N, rng, p)
    sustrato = DINAMICAS_B[p["eje_B"]](sustrato, p, rng, n_sweeps, p["eje_C"])
    m = medir(sustrato, p, rng)              # medición nativa (b=1): diam/giant/holonomía/adj/E
    adj_real = m["adj_final"]

    nv = medir_null_valor(m, p, np.random.default_rng(p["seed"] * 8000 + N))
    holon_real_native = m["holonomia"]
    holon_null_native = nv["holonomia"]

    # NULL_topo: n_seeds_null_topo grafos ER frescos, MISMA N y MISMA densidad que REAL a b=1,
    # coarse-graneados a las MISMAS escalas que el real para poder comparar diám por escala.
    meandeg_equiv = max(0.5, 2.0 * m["n_aristas"] / max(1, N))
    adjs_null = []
    for s in range(n_seeds_null_topo):
        seed_n = int(p["seed"] * 6000 + N * 13 + s)
        adj0, _ = GR.aleatorio(N, meandeg=meandeg_equiv, seed=seed_n)
        adjs_null.append([set(a.tolist()) for a in adj0])

    filas = []
    t0 = time.time()
    for b in escalas_b:
        if b == 1:
            adj_g, n_cajas = adj_real, N
        else:
            rng_b = np.random.default_rng(p["seed"] * 7000 + b * 31)
            asign, n_cajas = CS80.cajas_bfs(adj_real, N, b, rng_b)
            adj_g = CS80.grafo_grueso(adj_real, N, asign, n_cajas)
        diam_g = float(_diam(adj_g, n_cajas)) if n_cajas > 1 else float("nan")
        giant_g = float(_giant(adj_g, n_cajas)) if n_cajas > 1 else 0.0
        n_aristas_g = sum(len(a) for a in adj_g) // 2

        diam_nulls = []
        for k_null, adj_n in enumerate(adjs_null):
            if b == 1:
                adj_ng, n_cajas_n = adj_n, N
            else:
                rng_bn = np.random.default_rng(p["seed"] * 7500 + b * 37 + k_null)
                asign_n, n_cajas_n = CS80.cajas_bfs(adj_n, N, b, rng_bn)
                adj_ng = CS80.grafo_grueso(adj_n, N, asign_n, n_cajas_n)
            diam_nulls.append(float(_diam(adj_ng, n_cajas_n)) if n_cajas_n > 1 else float("nan"))
        diam_null_mean = float(np.nanmean(diam_nulls))
        diam_null_std = float(np.nanstd(diam_nulls)) + 1e-9

        filas.append(dict(
            rule_id=p["rule_id"], N=n_cajas, escala_b=b, dt=round(time.time() - t0, 2),
            diam_real=diam_g, giant_real=giant_g, holon_real=holon_real_native,
            n_aristas=n_aristas_g, n_triangulos=m["n_triangulos"],
            diam_null_topo=diam_null_mean, diam_null_topo_std=diam_null_std,
            holon_null_valor=holon_null_native,
        ))
    return filas


if __name__ == "__main__":
    import cs090_fase5_generador as GEN
    print("CS090 — motor: smoke test rápido (N chico) de las 5 combinaciones del piloto")
    combos = [("A0", "B0", "C0"), ("A1", "B0", "C0"), ("A2", "B1", "C1"), ("A1", "B1", "C0"), ("A2", "B0", "C2")]
    for (a, b, c) in combos:
        admitidas, _ = GEN.generar_reglas_clase(a, b, c, n_reglas=1, seed_base=1)
        p = admitidas[0]
        t0 = time.time()
        filas = correr_regla(p, Ns=(60, 120), n_sweeps=6, n_seeds_null_topo=2)
        print(f"  {p['rule_id']}: ok ({time.time()-t0:.2f}s) -> {filas[-1]}")
