"""
CS072-II -- módulo de FILTRACIÓN + JUECES CONTINUOS (§7.1-7.3 de PROPUESTA_CODEX_CS072_II..., adjudicado en
ADJUDICACION_CS072_II_transicion_sin_sustrato_CS.md R-LECTURA). Es el LECTOR de topología para el sustrato
sin-grafo: nunca usa un umbral único elegido -- ordena TODOS los pares por peso, avanza por BLOQUES COMPLETOS
DE EMPATE (en W uniforme, TODOS entran juntos -- nunca se desempata por índice/RNG), y mide cómo evolucionan
los jueces a lo largo de TODA la filtración. Una región cuenta como topología sólo si persiste en un
intervalo NO NULO de niveles.

Construido también para pasar S8 (control positivo: detecta métrica conocida) y S9 (W uniforme no adquiere
topología) de la Puerta S -- ver cs072_ii_puerta_s.py.

Codea/ejecuta: CC. Diseño/ruling: CS + Codex (§7).
"""
from __future__ import annotations
import sys
from collections import deque
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
import cs071_histeresis as S71
import cs064_smoke as SM

RNG = np.random.default_rng


# ============================ §7.1 JUECES CONTINUOS, SIN UMBRAL ============================
def jueces_continuos_sin_umbral(W, N):
    """Ninguno necesita binarizar W. dispersión de log(W/mediana); concentración nodal h_i=s_i/Sum(s) y su
    máximo (juez de hub); grado efectivo por participación k_eff(i)=(Sum_j W_ij)^2/Sum_j W_ij^2 (mide
    cuántos vecinos 'cuentan' realmente, sin umbral); espectro del Laplaciano ponderado (rango efectivo =
    cuántos autovalores no-triviales tiene, otro juez sin umbral de estructura)."""
    iu = np.triu_indices(N, k=1)
    vals = W[iu]
    vals_pos = vals[vals > 0]
    mediana = float(np.median(vals_pos)) if vals_pos.size else 0.0
    if mediana > 0 and vals_pos.size:
        log_disp = float(np.std(np.log(vals_pos / mediana)))
    else:
        log_disp = float("nan")

    s = W.sum(axis=1)
    suma_s = float(s.sum())
    h = s / suma_s if suma_s > 0 else np.zeros(N)
    max_h = float(h.max()) if N else float("nan")

    s2 = (W ** 2).sum(axis=1)
    k_eff = np.where(s2 > 0, (s ** 2) / np.maximum(s2, 1e-300), 0.0)
    k_eff_medio = float(k_eff.mean())
    k_eff_max = float(k_eff.max())

    L = np.diag(s) - W   # Laplaciano ponderado
    autovals = np.linalg.eigvalsh(L)
    autovals = np.clip(autovals, 0, None)
    rango_efectivo = int(np.sum(autovals > 1e-9 * max(autovals.max(), 1e-300)))

    return dict(log_dispersion=log_disp, max_h=max_h, k_eff_medio=k_eff_medio, k_eff_max=k_eff_max,
                rango_efectivo_laplaciano=rango_efectivo, n_autovalores=N)


# ============================ §7.2 FILTRACIÓN POR BLOQUES DE EMPATE ============================
def _bloques_de_empate(W, N, tol_rel=1e-9):
    """Ordena TODOS los pares (i<j) por W_ij DESCENDENTE, agrupa en bloques donde la diferencia relativa
    entre valores consecutivos es < tol_rel -- un bloque se procesa COMPLETO, nunca se desempata por
    índice ni RNG. Devuelve lista [(valor_representativo, [(i,j),...]), ...] de mayor a menor peso."""
    iu = np.triu_indices(N, k=1)
    vals = W[iu]
    orden = np.argsort(-vals, kind="stable")
    vals_ord = vals[orden]
    ii_ord = iu[0][orden]; jj_ord = iu[1][orden]
    n = len(vals_ord)
    bloques = []
    i = 0
    escala = max(float(np.max(vals_ord)), 1e-300) if n else 1.0
    while i < n:
        v = vals_ord[i]
        j = i
        while j < n and abs(vals_ord[j] - v) <= tol_rel * escala:
            j += 1
        pares = list(zip(ii_ord[i:j].tolist(), jj_ord[i:j].tolist()))
        bloques.append((float(v), pares))
        i = j
    return bloques


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n)); self.r = [0] * n; self.tam = [1] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra; self.tam[ra] += self.tam[rb]
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1

    def tam_max(self):
        return max(self.tam[self.find(i)] for i in range(len(self.p)))


def curva_filtracion(W, N, n_checkpoints_judges=15, rng_judges=None, tol_rel=1e-9):
    """Recorre TODA la filtración (bloques de empate). frac_gigante se registra en CADA bloque (barato,
    union-find incremental). diam/d_s/delta_gromov (caros, BFS) sólo en ~n_checkpoints_judges niveles
    reprartidos por FRACCIÓN DE PARES ya incluidos -- declarado, no es un umbral elegido para favorecer
    ningún resultado, es un muestreo UNIFORME de toda la curva."""
    if rng_judges is None:
        rng_judges = RNG(0)
    bloques = _bloques_de_empate(W, N, tol_rel=tol_rel)
    total_pares = N * (N - 1) // 2
    uf = _UnionFind(N)
    adj = [set() for _ in range(N)]
    pares_incluidos = 0
    siguiente_checkpoint = 0
    curva = []   # cada item: dict con nivel, frac_pares, frac_gigante, y (si checkpoint) diam/d_s/delta
    for nivel_idx, (valor, pares) in enumerate(bloques):
        for (i, j) in pares:
            uf.union(i, j)
            adj[i].add(j); adj[j].add(i)
        pares_incluidos += len(pares)
        frac_pares = pares_incluidos / total_pares if total_pares else 1.0
        frac_gig = uf.tam_max() / N
        item = dict(nivel=nivel_idx, valor=valor, frac_pares=frac_pares, frac_gigante=frac_gig)
        if frac_pares >= siguiente_checkpoint or nivel_idx == len(bloques) - 1:
            diam = S71._diam_robusto(adj, N, rng_judges)
            ds = SM.dim_volumen(adj, N, rng=rng_judges)
            item["diam"] = diam; item["d_s"] = ds
            siguiente_checkpoint += 1.0 / n_checkpoints_judges
        curva.append(item)
    return curva, adj  # adj final = grafo completo (todos los pares, W>0) -- para diagnóstico


def persistencia_conectividad(curva, frac_umbral=0.9, min_ancho_frac=0.02):
    """Ancla P-COHESIÓN/P-BORDE/P-DISOLUCIÓN (Codex §8): busca el INTERVALO de fracción-de-pares-incluidos
    donde frac_gigante>=frac_umbral se sostiene, y exige que ese intervalo tenga ancho NO NULO
    (>=min_ancho_frac de la filtración total) -- si sólo se cumple en un punto aislado, no cuenta como
    'persistente'."""
    tramos = []
    inicio = None
    for item in curva:
        if item["frac_gigante"] >= frac_umbral:
            if inicio is None:
                inicio = item["frac_pares"]
            fin = item["frac_pares"]
        else:
            if inicio is not None:
                tramos.append((inicio, fin))
            inicio = None
    if inicio is not None:
        tramos.append((inicio, curva[-1]["frac_pares"]))
    tramos_anchos = [(a, b) for (a, b) in tramos if (b - a) >= min_ancho_frac]
    return tramos_anchos


# ============================ §7.3 SEGUNDO SELLO: métrica ponderada, dos transformaciones ============================
def _delta_gromov_de_matriz(D, rng, n_landmarks=40, n_quad=300):
    n = D.shape[0]
    cand = [i for i in range(n) if np.isfinite(D[i]).sum() > 1]
    if len(cand) < 4:
        return float("nan")
    if n_landmarks > len(cand):
        n_landmarks = len(cand)
    landmarks = rng.choice(cand, size=n_landmarks, replace=False)
    sub = D[np.ix_(landmarks, landmarks)]
    deltas = []
    for _ in range(n_quad):
        a, b, c, d = rng.choice(n_landmarks, size=4, replace=False)
        dab, dcd = sub[a, b], sub[c, d]
        dac, dbd = sub[a, c], sub[b, d]
        dad, dbc = sub[a, d], sub[b, c]
        vals = [dab, dcd, dac, dbd, dad, dbc]
        if not all(np.isfinite(v) for v in vals):
            continue
        s1 = dab + dcd; s2 = dac + dbd; s3 = dad + dbc
        s = sorted([s1, s2, s3], reverse=True)
        deltas.append((s[0] - s[1]) / 2.0)
    return float(np.median(deltas)) if deltas else float("nan")


def segundo_sello(W, N, rng, n_landmarks=40, n_quad=300):
    """Métrica ponderada preinscrita d_ij=-log(W_ij/maxW) (Van Raamsdonk/Ryu-Takayanagi, elemento 14 ya
    canónico) + UNA transformación monótona alternativa d_ij=1/W_ij (§7.3: robustez, ninguna se elige por
    dar dimensión preferida). Dijkstra sobre el grafo COMPLETO ponderado (sin binarizar) + delta-Gromov
    muestreado. Devuelve ambos deltas -- la conclusión debe ser robusta a los dos."""
    maxW = float(W.max()) if W.max() > 0 else 1.0
    W_safe = np.where(W > 0, W, 1e-300)

    d_log = -np.log(np.clip(W_safe / maxW, 1e-300, None))
    np.fill_diagonal(d_log, 0.0)
    D_log = shortest_path(csr_matrix(d_log), method="D", directed=False)
    delta_log = _delta_gromov_de_matriz(D_log, rng, n_landmarks, n_quad)

    d_inv = 1.0 / W_safe
    np.fill_diagonal(d_inv, 0.0)
    D_inv = shortest_path(csr_matrix(d_inv), method="D", directed=False)
    delta_inv = _delta_gromov_de_matriz(D_inv, rng, n_landmarks, n_quad)

    return dict(delta_gromov_log=delta_log, delta_gromov_inv=delta_inv)


# ============================ control positivo declarado (instrumento, no origen) ============================
def W_control_positivo_2d(side, xi=1.5, ruido_rel=0.0, rng=None):
    """SUSTRATO MÉTRICO CONOCIDO, declarado sólo como prueba del INSTRUMENTO (Codex §8, brazo 5) -- NO
    participa de la afirmación de origen sin-sustrato-previo. Coordenadas 2D genuinas (grilla side x side),
    W_ij = exp(-dist_euclidea(i,j)/xi). Continua, decae con la distancia real -- exactamente lo que un
    lector honesto debe poder detectar."""
    N = side * side
    coords = np.array([(r, c) for r in range(side) for c in range(side)], dtype=np.float64)
    dif = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((dif ** 2).sum(axis=-1))
    W = np.exp(-dist / xi)
    np.fill_diagonal(W, 0.0)
    if ruido_rel > 0 and rng is not None:
        W = W * (1.0 + ruido_rel * rng.standard_normal(W.shape))
        W = np.clip((W + W.T) / 2.0, 0.0, None)
        np.fill_diagonal(W, 0.0)
    return W, N


if __name__ == "__main__":
    print("cs072_ii_filtracion.py -- lector de filtracion + jueces continuos + segundo sello. "
          "Correr cs072_ii_puerta_s.py (S8/S9 usan este modulo).", flush=True)
