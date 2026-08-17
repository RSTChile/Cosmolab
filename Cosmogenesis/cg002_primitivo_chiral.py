"""
CG002 — Primitivo quiral: vértice 3 + R_θ (EIT3 ciclo límite)
=============================================================
Trasplante de la receta EIT3 probada en ciclo límite (abr-2026):
  u_mix = α·R_θ(u) + β·M_k + ε   →   M_k ← safeClamp(u_mix)

Caveat (CC Msg 44): EIT3 gira un campo espacial 2D; aquí giramos un vector
de acoplamiento entre dos patas [Q·m_i, Q·m_j]. La receta se trasplanta;
la geometría espacial no. Validación: controles θ=0 y θ→−θ.

Quiralidad = estructura de paridad (mano), NO masa. YUK uniforme en magnitud;
el signo viene de u_mix[0] tras el giro.

Parámetros por defecto = ciclo límite EIT3 (no Fase I):
  α_loop=0.573, β_mem=0.446, θ=1.74°, σ_E=0.0012, blur_pairs=True
"""
from __future__ import annotations

import numpy as np

from cg002_primitivo_vertex3 import (
    ALPHA,
    ETA,
    KAPPA_S,
    MU,
    PASOS,
    S0,
    apply_triplet,
    sat,
)

# Ciclo límite EIT3 (CC Msg 44 corrección #2)
ALPHA_LOOP = 0.573
BETA_MEM = 0.446
THETA_DEG = 1.74
SIGMA_E = 0.0012
USE_BLUR = True


def rot2(theta_deg: float) -> np.ndarray:
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]], dtype=float)


def safe_clamp_vec(v: np.ndarray, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Análogo EIT3 safeClamp[0,1], adaptado a paridad firmada."""
    return np.clip(v, lo, hi)


def yukawa_plane(Q: np.ndarray, m: np.ndarray, i: int, j: int) -> np.ndarray:
    return np.array([Q[i] * m[i], Q[j] * m[j]], dtype=float)


def blur_pairs(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros(2)
    return np.mean(vectors, axis=0)


def build_mediator_pairs(
    triplets: list[tuple[int, int, int, float]],
) -> dict[int, list[tuple[int, int]]]:
    """k → lista de pares (i,j) que pasan por el mediador."""
    mp: dict[int, list[tuple[int, int]]] = {}
    for i, k, j, _ in triplets:
        mp.setdefault(int(k), []).append((int(i), int(j)))
    return mp


def evolucion_chiral_vertex3(
    C_pair: np.ndarray,
    triplets: list[tuple[int, int, int, float]],
    Q: np.ndarray,
    N: int,
    *,
    yuk_scale: float = 0.06,
    alpha_loop: float = ALPHA_LOOP,
    beta_mem: float = BETA_MEM,
    theta_deg: float = THETA_DEG,
    sigma_e: float = SIGMA_E,
    use_blur: bool = USE_BLUR,
    shuffle_signs: bool = False,
    rng: np.random.Generator | None = None,
    pasos: int = PASOS,
) -> tuple[np.ndarray, np.ndarray, int, dict[int, np.ndarray]]:
    """
    Evolución con mediadores que acumulan memoria M_k y tuercen w con R_θ.
    triplets: plantilla (i,k,j,_) — el peso se recalcula cada paso.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    R = rot2(theta_deg)
    mediator_pairs = build_mediator_pairs(triplets)
    M: dict[int, np.ndarray] = {k: np.zeros(2) for k in mediator_pairs}

    S = np.full(N, S0)
    alive = np.ones(N, bool)
    tau = 0

    for _ in range(pasos):
        S = np.where(alive, S * (1 - MU), S)
        m = np.where(alive, np.sqrt(sat(S)), 0.0)
        coop = C_pair @ m

        dS = ALPHA * ETA * m * (C_pair @ m) - ALPHA * ETA * m * m
        for k, pairs in mediator_pairs.items():
            if not alive[k]:
                continue
            vecs = []
            alive_pairs = [(i, j) for i, j in pairs if alive[i] and alive[j]]
            if not alive_pairs:
                continue
            for i, j in alive_pairs:
                vecs.append(yukawa_plane(Q, m, i, j))
            u_ref = blur_pairs(vecs) if use_blur else vecs[0]

            u_mix_list = []
            for i, j in alive_pairs:
                u_ij = yukawa_plane(Q, m, i, j)
                if use_blur:
                    u_ij = 0.5 * u_ij + 0.5 * u_ref
                u_rot = R @ u_ij
                eps = rng.normal(0.0, sigma_e, size=2) if sigma_e > 0 else np.zeros(2)
                u_mix = alpha_loop * u_rot + beta_mem * M[k] + eps
                u_mix = safe_clamp_vec(u_mix)
                u_mix_list.append(u_mix)
                w = yuk_scale * u_mix[0]
                if shuffle_signs:
                    w = abs(w) * rng.choice([-1.0, 1.0])
                apply_triplet(dS, m, alive, i, k, j, w)

            if u_mix_list:
                M[k] = safe_clamp_vec(np.mean(u_mix_list, axis=0))

        d_struct = float(np.abs(dS[alive]).sum()) if alive.any() else 0.0
        S = S + dS
        S = np.minimum(S, 1e12)
        S = np.where(S < 0, 0, S)
        alive = S > KAPPA_S
        S = np.where(alive, S, 0.0)
        if d_struct > 1e-4:
            tau += 1

    return S, alive, tau, M