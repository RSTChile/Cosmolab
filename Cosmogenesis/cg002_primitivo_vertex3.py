"""
CG002 — Primitivo extendido: compatibilidad pareada + vértices de 3 puntos
==========================================================================
Decisión Mesa (Alexis, 30-jun-2026): extender el marco pareado c_ij con
acoplamiento ternario v_ijk en campo medio.

Ecuación de campo (por partícula i):
  coop_i = Σ_j c_ij m_j  +  Σ_{j,k} v_ijk m_j m_k
  dS_i   = α·η·m_i·coop_i − α·η·m_i²

Convención del vértice (i,j,k,w): contribuye a las tres patas
  dS_i += α·η·w·m_i·m_j·m_k  (idem j, k — simétrico en el triplete).

Tipos implementados:
  · gauge: quark_i — gluon_a — quark_j  con w = Re(v_i^T λ_a v_j)
  · yukawa: fermion_i — higgs — fermion_j  con w = YUK (uniforme, sin masa en semilla)

USO: importado por cg002_r7g_vertex3.py y fases posteriores.
"""
from __future__ import annotations

import numpy as np

ETA, MU, KAPPA_S, S0, S_BAND = 0.05, 0.01, 1e-6, 1.0, 8.0
PASOS, ALPHA = 240, 1.0

# Gell-Mann corregido (PDG): incluye λ₃ = diag(1,-1,0); todas Hermitianas
LAMBDA = np.array(
    [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], float) / np.sqrt(3),
    ],
    dtype=complex,
)


def sat(S: np.ndarray) -> np.ndarray:
    return S / (1.0 + S / S_BAND)


def apply_triplet(dS: np.ndarray, m: np.ndarray, alive: np.ndarray, i: int, j: int, k: int, w: float):
    if not (alive[i] and alive[j] and alive[k]) or w == 0.0:
        return
    t = ALPHA * ETA * w
    dS[i] += t * m[i] * m[j] * m[k]
    dS[j] += t * m[j] * m[i] * m[k]
    dS[k] += t * m[k] * m[i] * m[j]


def coop_from_triplets(m: np.ndarray, alive: np.ndarray, triplets: list[tuple[int, int, int, float]]) -> np.ndarray:
    """Cooperación efectiva en i: Σ_{jk} v_ijk m_j m_k (para combinar con c_ij)."""
    N = len(m)
    coop = np.zeros(N)
    for i, j, k, w in triplets:
        if not (alive[i] and alive[j] and alive[k]) or w == 0.0:
            continue
        coop[i] += w * m[j] * m[k]
        coop[j] += w * m[i] * m[k]
        coop[k] += w * m[i] * m[j]
    return coop


def evolucion_vertex3(
    C_pair: np.ndarray,
    triplets: list[tuple[int, int, int, float]],
    N: int,
    pasos: int = PASOS,
) -> tuple[np.ndarray, np.ndarray, int]:
    S = np.full(N, S0)
    alive = np.ones(N, bool)
    tau = 0
    for _ in range(pasos):
        S = np.where(alive, S * (1 - MU), S)
        m = np.where(alive, np.sqrt(sat(S)), 0.0)
        coop = C_pair @ m + coop_from_triplets(m, alive, triplets)
        dS = ALPHA * ETA * m * coop - ALPHA * ETA * m * m
        d_struct = float(np.abs(dS[alive]).sum()) if alive.any() else 0.0
        S = S + dS
        S = np.minimum(S, 1e12)
        S = np.where(S < 0, 0, S)
        alive = S > KAPPA_S
        S = np.where(alive, S, 0.0)
        if d_struct > 1e-4:
            tau += 1
    return S, alive, tau


def build_gauge_triplets(
    V: np.ndarray,
    labels: np.ndarray,
    idx_quark: np.ndarray,
    idx_gluon: np.ndarray,
    gluon_label_fn,
    scale: float = 1.0,
    eps: float = 1e-6,
    max_per_gluon: int = 2000,
) -> list[tuple[int, int, int, float]]:
    """Vértice q_i — gluon_a — q_j con w = Re(v_i^T λ_a v_j), i≠j."""
    triplets: list[tuple[int, int, int, float]] = []
    Vq = V[idx_quark].astype(complex)
    labs_g = labels[idx_gluon]
    for a in range(8):
        jg = idx_gluon[labs_g == gluon_label_fn(a)]
        if len(jg) == 0:
            continue
        g_slot = int(jg[0])
        La = LAMBDA[a]
        W = np.real(np.einsum("pi,ij,qj->pq", Vq.conj(), La, Vq))
        np.fill_diagonal(W, 0.0)
        pq = np.argwhere(np.abs(W) > eps)
        cand = []
        for p, q in pq:
            if p >= q:
                continue
            cand.append((abs(W[p, q]), p, q, float(W[p, q])))
        cand.sort(reverse=True)
        for _, p, q, w in cand[:max_per_gluon]:
            i, j = int(idx_quark[p]), int(idx_quark[q])
            triplets.append((i, g_slot, j, scale * w))
    return triplets


def build_yukawa_triplets(
    idx_fermion: np.ndarray,
    idx_higgs: np.ndarray,
    yuk: float = 0.08,
    max_pairs: int | None = None,
) -> list[tuple[int, int, int, float]]:
    """Vértice f_i — H — f_j (i<j); yuk uniforme — sin jerarquía en semilla."""
    if len(idx_higgs) == 0:
        return []
    h = int(idx_higgs[0])
    triplets: list[tuple[int, int, int, float]] = []
    ferm = list(idx_fermion)
    n_pairs = 0
    for p in range(len(ferm)):
        for q in range(p + 1, len(ferm)):
            if max_pairs is not None and n_pairs >= max_pairs:
                return triplets
            i, j = int(ferm[p]), int(ferm[q])
            triplets.append((i, h, j, yuk))
            n_pairs += 1
    return triplets


def vertex_coupling_strength(triplets: list[tuple[int, int, int, float]], idx: np.ndarray) -> float:
    """Σ|w| de vértices que tocan algún índice del conjunto idx."""
    idx_set = set(int(i) for i in idx)
    s = 0.0
    for i, j, k, w in triplets:
        if i in idx_set or j in idx_set or k in idx_set:
            s += abs(w)
    return s