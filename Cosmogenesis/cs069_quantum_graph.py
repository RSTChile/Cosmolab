"""
CS069 — El frente cuántico: ¿la dirección emerge de una superposición de grafos?
==============================================================================
Diseño/ruling: CS (DISENO_CS069_frente_cuantico_CS.md, 17-jul-2026), adjudicando y corrigiendo una
propuesta de Gemini con un hueco de Shannon cazado por código. Cierra sobre el veredicto convergente del
arco del espacio (CS066 B / CS067 B / CS068 Mundo B): con grafos CLÁSICOS definidos, la distancia emerge de
la diferencia pero la DIRECCIÓN no. Pregunta de Alexis (16-jul): si la dirección no vive en ningún grafo
DEFINIDO, ¿vive en la SUPERPOSICIÓN de todos?

MECANISMO
Matriz de amplitud relacional A_ij = ρ_ij · e^(iφ_ij) por enlace del grafo real (motor de los 17, adj de
E._sustrato). ρ_ij sale de la MISMA cantidad del motor (ingrediente 14, H._pesos_correlacion): w_ij∈(0,1]
alto=soporte local, →0=atajo; d_ij=-log(w_ij) es la distancia de correlación clásica. Amplitud =
e^{-distancia} (convención Feynman euclídeo/Boltzmann): ρ_ij := e^{-d_ij} = w_ij EXACTAMENTE. ρ_ij hace
DOBLE función (G-MECANISMO-UNIFICADO, no hay ingrediente nuevo aislado): amplitud del camino Y peso del
acoplamiento de fase entre nodos (ver DINÁMICA DE FASE).

DINÁMICA DE FASE — v2, CORREGIDA por CS (ADJUDICACION_CS069_smoke_regla_fase_CS.md, 17-jul): la v1 (φ del
ENLACE tira hacia la media de los enlaces vecinos) arrastraba atajo y local por igual hacia sincronía local
-- no podía distinguirlos (sospecha de CC, confirmada por CS con código: AUC 0.80 en juguete con la v2).
G-FASE-CIEGA (auditable: la función de update NUNCA recibe es_atajo, distancia de anillo, ni ninguna verdad
de fondo; solo la topología y ρ_ij, ambos del motor):
  θ_i(t+1) = θ_i(t) + η · Σ_j ρ_ij·sin(θ_j−θ_i) / Σ_j ρ_ij        (Kuramoto POR NODO, ponderado por ρ)
  φ_ij := θ_i − θ_j  (centrado en [-π,π])  -- la FRUSTRACIÓN entre los dos extremos del enlace
Vecindarios unidos por enlaces baratos (ρ alto) sincronizan rápido y fuerte; dos vecindarios unidos solo
por un atajo (ρ bajo) sincronizan cada uno por separado, sin que el atajo tenga fuerza para juntarlos. Un
atajo que puentea dos dominios que sincronizaron aparte queda FRUSTRADO (|φ| grande) → decoherente; un
local dentro de un dominio queda coherente (|φ| chico) -- sin que la regla sepa nunca cuál es cuál.

INTEGRAL DE CAMINO — vía potencias de la matriz de amplitud (todos los caminos de cada longitud ≤L,
incluye caminos que revisitan nodos, convención estándar de propagador):
  K_ij(L) = Σ_{l=1}^{L} (A^l)_{ij}          D_q(i,j) = −log|K_ij(L)|
L fijo (no se tunea tras ver resultados): L=8 — margen sobre el diámetro típico mundo-pequeño (~4-6,
CS066/067/068) sin llevar la matriz a densidad completa muchas veces (costo computacional en grafos de
N~2500).

Codea/ejecuta: CC. Diseño/ruling: CS.
"""
from __future__ import annotations
import os, sys, time, math
from collections import deque
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)
import cs067_habitacion_completa as H
import cs068_inflacion_estirar_enfriar as E
import cs068_paso2_mundo_ab as P2
import cs068_paso1_sintetico as S1

RNG = np.random.default_rng

L_CAMINO = 8            # longitud máx. de la integral de camino. Fijo -- G-NO-CALIBRAR.
ETA = 0.5                # peso del acoplamiento de fase (Kuramoto por nodo, ponderado por ρ)
T_PASOS = 40              # pasos de evolución de fase antes de leer K_ij
GAMMA_CORR = 1.0          # gamma de H._pesos_correlacion (ingrediente 14), sin tocar el default del motor


# ============================ ρ_ij y costo (motor de los 17, ingrediente 14, una sola fuente) ============================
def _rho_y_costo(adj, N, rng, gamma=GAMMA_CORR):
    """w_ij del motor (H._pesos_correlacion): ρ_ij=w_ij (amplitud=e^-distancia), costo=d_ij=-log(w_ij).
    Misma cantidad, dos lecturas -- no hay un ingrediente nuevo aislado (G-MECANISMO-UNIFICADO)."""
    wmap = H._pesos_correlacion(adj, N, gamma, "real", rng)
    edges = list(wmap.keys())
    rho = {e: float(w) for e, w in wmap.items()}
    costo = {e: float(-math.log(max(w, 1e-12))) for e, w in wmap.items()}
    return edges, rho, costo


# ============================ dinámica de fase — G-FASE-CIEGA ============================
def _evoluciona_fase(edges, rho, N, rng, t_pasos=T_PASOS, eta=ETA):
    """CORREGIDO por CS (ADJUDICACION_CS069_smoke_regla_fase_CS.md, 17-jul): la v1 ("φ del enlace tira
    hacia la media de los enlaces VECINOS") arrastraba atajo y local por igual hacia sincronía local -- no
    podía distinguirlos (sospecha de CC, confirmada por CS con código: AUC 0.80 en juguete con la v2).

    v2 -- Kuramoto POR NODO (ciego), acoplamiento PONDERADO por ρ_ij (el mismo w_ij del motor -- un enlace
    caro/atajo con ρ bajo tira poco de sus dos extremos hacia sincronía mutua; no hay costo aditivo nuevo,
    el costo YA está en el peso del acoplamiento, G-MECANISMO-UNIFICADO):
      θ_i(t+1) = θ_i(t) + eta · Σ_j ρ_ij·sin(θ_j−θ_i) / Σ_j ρ_ij
    Con esto, vecindarios unidos por enlaces baratos (ρ alto) sincronizan RÁPIDO y FUERTE entre sí; dos
    vecindarios unidos solo por un atajo (ρ bajo) sincronizan cada uno por separado, sin que el atajo tenga
    fuerza para juntarlos. La FRUSTRACIÓN entre los dos extremos de un enlace, φ_ij=θ_i−θ_j (centrada en
    [-π,π]), es la fase que entra en la matriz de amplitud: un atajo que puentea dos dominios que
    sincronizaron aparte queda frustrado (|φ| grande) -> decoherente; un local dentro de un dominio queda
    coherente (|φ| chico). Auditable: SOLO recibe topología+ρ (motor), nunca es_atajo()."""
    vecinos_pesos = [[] for _ in range(N)]
    for (i, j) in edges:
        r = rho[(i, j)]
        vecinos_pesos[i].append((j, r))
        vecinos_pesos[j].append((i, r))
    theta = rng.uniform(0, 2 * np.pi, size=N)
    for _ in range(t_pasos):
        nueva = theta.copy()
        for i in range(N):
            vs = vecinos_pesos[i]
            if not vs:
                continue
            num = 0.0; den = 0.0
            for (j, w) in vs:
                num += w * math.sin(theta[j] - theta[i]); den += w
            nueva[i] = theta[i] + eta * num / max(den, 1e-12)
        theta = nueva % (2 * np.pi)
    phi = {}
    for (i, j) in edges:
        d = theta[i] - theta[j]
        d = (d + np.pi) % (2 * np.pi) - np.pi
        phi[(i, j)] = float(d)
    return phi


# ============================ K_ij(L) y D_q ============================
def _matriz_amplitud(N, edges, rho, phi, grados):
    """CORRECCIÓN encontrada al smoke-testear (no estaba en el diseño de CS, es un detalle de
    implementación): sin normalizar, |K_ij(L)| explota combinatoriamente -- el nº de caminos entre dos
    nodos crece ~grado^l mucho más rápido de lo que ρ<1 lo aplasta, así que D_q=-log|K| salía NEGATIVO
    para casi todo par (más caminos que "distancia"). Normalización estándar de operador de caminata
    cuántica/adyacencia normalizada (D^-1/2 A D^-1/2, la misma que el Laplaciano normalizado de grafos):
    ρ_ij_norm = ρ_ij / sqrt(deg_i·deg_j). Es una cantidad puramente local (grado propio de cada nodo, ya
    conocido por el motor) -- no lee es_atajo ni ninguna verdad de fondo, no rompe G-FASE-CIEGA."""
    filas, cols, datos = [], [], []
    for (i, j) in edges:
        norm = 1.0 / math.sqrt(max(grados[i], 1) * max(grados[j], 1))
        a = rho[(i, j)] * norm * np.exp(1j * phi[(i, j)])
        filas += [i, j]; cols += [j, i]; datos += [a, a]
    return sp.csr_matrix((datos, (filas, cols)), shape=(N, N), dtype=np.complex128)


def _K_y_Dq(N, edges, rho, phi, L=L_CAMINO):
    """K_ij(L) = suma de potencias de la matriz de amplitud hasta L (integral de camino discreta, incluye
    caminos que revisitan nodos). D_q=-log|K|. Usa matriz DENSA desde el principio -- en estos grafos
    mundo-pequeño (diam~4-6) K satura a casi-densa en pocas potencias, así que no hay ahorro real en
    mantener sparse más allá de la primera potencia."""
    grados = np.zeros(N)
    for (i, j) in edges:
        grados[i] += 1; grados[j] += 1
    A = _matriz_amplitud(N, edges, rho, phi, grados).toarray()
    K = np.zeros((N, N), dtype=np.complex128)
    Al = np.eye(N, dtype=np.complex128)
    for _ in range(L):
        Al = Al @ A
        K += Al
    mag = np.abs(K)
    no_alcanzado = mag < 1e-280  # ningún camino de longitud <=L conecta ese par -- NO es una "distancia grande" real
    mag = np.clip(mag, 1e-300, None)
    Dq = -np.log(mag)
    Dq[no_alcanzado] = np.nan
    np.fill_diagonal(Dq, 0.0)
    return Dq


# ============================ Juez B — pendiente log-log de diám_q(N) (G-PENDIENTE-NO-VALOR) ============================
def diam_q_robusto(Dq, N, rng, n_src=12, percentil=90):
    """Análogo cuántico de H._diam_robusto: D_q YA es distancia entre TODOS los pares (no hace falta BFS/
    Dijkstra -- la integral de camino ya sumó todos los caminos). Percentil robusto de D_q(s,*) sobre
    fuentes muestreadas, ignorando la diagonal y pares no-finitos."""
    eccs = []
    for s in rng.integers(0, N, size=min(n_src, N)):
        s = int(s)
        fila = np.delete(Dq[s], s)
        fila = fila[np.isfinite(fila)]
        if len(fila) > 0.3 * N:
            eccs.append(np.percentile(fila, percentil))
    return float(np.median(eccs)) if eccs else float("nan")


def _pendiente_loglog(Ns, vals):
    x = np.log(np.array(Ns, float)); y = np.log(np.array(vals, float))
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m), float(b)


# ============================ Juez A — cedazo de π cuántico ============================
def cedazo_pi(Dq, N, rng, n_fuentes=6, n_bins=12):
    """π_local(r) = tamaño_del_cascarón(r) / (2r), el análogo discreto de circunferencia/(2*radio) para un
    plano euclídeo (C=2πr). Cascarón = nodos con D_q(fuente,·) en [r,r+dr). Si el espacio es 2D-plano
    coherente, π_local(r) debe converger (CV bajo) a un valor estable al crecer r ("se congela"); si es
    mundo-pequeño/decoherente, el conteo por cascarón no crece de forma métrica y π_local oscila sin
    converger ("estalla"). Devuelve (media, CV) agregando sobre varias fuentes y bins intermedios (se
    descartan el primer y último bin -- efectos de borde de la fuente y del alcance de D_q)."""
    valores = []
    fuentes = rng.integers(0, N, size=min(n_fuentes, N))
    for s in fuentes:
        s = int(s)
        fila = np.delete(Dq[s], s)
        fila = fila[np.isfinite(fila) & (fila > 0)]
        if len(fila) < 20:
            continue
        r_max = np.percentile(fila, 95)
        bordes = np.linspace(1e-6, r_max, n_bins + 1)
        for b in range(1, n_bins - 1):  # descarta primer/último bin (borde)
            r0, r1 = bordes[b], bordes[b + 1]
            r_medio = 0.5 * (r0 + r1)
            cascaron = np.sum((fila >= r0) & (fila < r1))
            if cascaron > 0 and r_medio > 1e-6:
                valores.append(cascaron / (2.0 * r_medio))
    if len(valores) < 5:
        return float("nan"), float("nan")
    v = np.array(valores, float)
    media = float(v.mean())
    cv = float(v.std() / (abs(media) + 1e-12))
    return media, cv


# ============================ Juez C — gap espectral (MDS sobre D_q + candado picado-por-nodo CS067) ============================
def _mds_embedding(Dq, N, K=8):
    """Escalado multidimensional clásico: doble-centrado de -D_q^2/2 y top-K autovectores (eigsh, solo
    los K mayores -- evita la eigendescomposición completa O(N^3) a N grande). V_i = posición del nodo i
    en el embedding K-dim (escalada por sqrt(autovalor)).

    MDS clásico necesita una matriz de distancias COMPLETA -- los pares "no alcanzados dentro de L" (NaN
    en D_q) se rellenan con un horizonte fijo (percentil 99 de lo finito × 1.5, tope superior razonable de
    "lejos, más allá de lo que la integral de camino pudo ver"), no con infinito."""
    Dq_lleno = Dq.copy()
    finito = Dq_lleno[np.isfinite(Dq_lleno)]
    tope = float(np.percentile(finito, 99)) * 1.5 if len(finito) else 1.0
    Dq_lleno[~np.isfinite(Dq_lleno)] = tope
    np.fill_diagonal(Dq_lleno, 0.0)
    D2 = Dq_lleno ** 2
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * (J @ D2 @ J)
    B = 0.5 * (B + B.T)  # simetriza (ruido numérico)
    k_eff = min(K, N - 2)
    vals, vecs = eigsh(B, k=k_eff, which="LA")
    orden = np.argsort(vals)[::-1]
    vals = np.clip(vals[orden], 0, None)
    vecs = vecs[:, orden]
    V = vecs * np.sqrt(vals + 1e-12)[None, :]
    return V, vals


def juez_gap_espectral(Dq, N, K=8):
    """Reusa el juez de CS067 tal cual: T=(1/N)ΣV_iV_i^T sobre el embedding MDS, cuenta_ejes_gap sobre sus
    autovalores (n_ejes, gap_interno), y el CANDADO obligatorio picado_por_nodo(V) (umbral 0.85) para
    certificar dominios reales y no smear -- exactamente el mismo candado que cerró CS067."""
    V, _ = _mds_embedding(Dq, N, K=K)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    T = (Vn.T @ Vn) / N
    ev = np.linalg.eigvalsh(T)[::-1]
    n_ejes, PR, gap_interno, r_thr = H.cuenta_ejes_gap(ev)
    pico_medio, frac_picados = H.picado_por_nodo(V)
    return dict(n_ejes=n_ejes, PR=PR, gap_interno=gap_interno, pico_medio=pico_medio,
                frac_picados=frac_picados, certificado=(pico_medio > 0.85))


# ============================ LOS 4 BRAZOS ============================
def brazo_completo(adj, N, rng, L=L_CAMINO):
    edges, rho, _costo = _rho_y_costo(adj, N, rng)
    phi = _evoluciona_fase(edges, rho, N, rng)
    return _K_y_Dq(N, edges, rho, phi, L=L)


def brazo_null_fase_topo(adj, N, rng):
    """Fases al azar CADA lectura, pero sobre un REWIRING config-model del MISMO grafo (misma secuencia de
    grados, misma redundancia topológica) -- mata la coherencia CONSERVANDO el clustering, para no repetir
    la lección CS068 (ganarle a fases-al-azar sobre el MISMO grafo sería solo clustering)."""
    n_edges = sum(len(a) for a in adj) // 2
    adj2 = P2._double_edge_swap(adj, N, 10 * n_edges, rng)
    edges, rho, _costo = _rho_y_costo(adj2, N, rng)
    phi = {e: float(rng.uniform(0, 2 * np.pi)) for e in edges}
    return _K_y_Dq(N, edges, rho, phi)


def brazo_null_fase_azar(adj, N, rng):
    """Control débil de Gemini: fases al azar sobre el grafo REAL (sin rewiring) -- mide cuánto separa el
    clustering solo, sin siquiera cambiar la topología."""
    edges, rho, _costo = _rho_y_costo(adj, N, rng)
    phi = {e: float(rng.uniform(0, 2 * np.pi)) for e in edges}
    return _K_y_Dq(N, edges, rho, phi)


def brazo_null_clasico(adj, N, rng):
    """φ≡0 -- se reduce a un propagador puramente REAL sobre ρ=w_ij. Línea base: debe reproducir el
    colapso mundo-pequeño de CS068 (Mundo B)."""
    edges, rho, _costo = _rho_y_costo(adj, N, rng)
    phi = {e: 0.0 for e in edges}
    return _K_y_Dq(N, edges, rho, phi)


BRAZOS = dict(completo=brazo_completo, null_fase_topo=brazo_null_fase_topo,
              null_fase_azar=brazo_null_fase_azar, null_clasico=brazo_null_clasico)


if __name__ == "__main__":
    print("cs069_quantum_graph.py — módulo de mecanismo. Correr cs069_smoke.py para las 3 anclas obligatorias.")
