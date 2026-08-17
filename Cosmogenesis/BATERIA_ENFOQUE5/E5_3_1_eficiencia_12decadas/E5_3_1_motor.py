#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5_3_1_motor.py — TEMA 3 (eficiencia emergente) · E5.3-1
==========================================================

Implementa EXACTAMENTE la definición congelada en PROTOCOLO_E5.3-1_PREREGISTRO.md
(leer ese archivo primero — este motor no redefine nada, solo ejecuta lo pre-registrado).

Resumen de la física (ver protocolo §2 para la justificación completa):
  - Anillo de N=200 sitios, campo φ.
  - φ(0) = 1 + ε·pert (pert = 5 modos seno, fase aleatoria por semilla, normalizado a std=1).
  - Cada paso: difusión local (solo aristas activas) -> swap dinámico (ruido, conserva
    exactamente Σ(φ-1)² del anillo completo) -> expansión (corte Bernoulli de aristas
    activas, prob H=min(r·D_eps,1)).
  - D_eps = difusividad de UN paso medida en el propio campo (H=0), promediada en semillas.
  - Al final: dominios = componentes conexas del anillo con las aristas que sobrevivieron;
    dominio "ligado" si 1<=tamaño<N (quedó aislado).
  - E_total = Σ(φ(0)-1)² (presupuesto declarado, axioma E1).
  - E_ligada = Σ_{sitios en dominios aislados} (φ(final)-1)².
  - eficiencia = E_ligada / E_total  ∈ [0,1] por construcción (ver protocolo §2.9).
  - NULL: misma topología de dominios, φ(final) permutado antes de sumar.

Barrido: ε en 12 décadas (13 pts, np.logspace(-12,0,13)) + control ε=0;
         r en 6 décadas (13 pts, np.logspace(-3,3,13));
         20 semillas por celda (vectorizadas como batch S=20).

T7 (perturbación dinámica): el swap dinámico actúa en TODOS los pasos de TODAS las
corridas — no es cosmético, es parte de la dinámica que compite con la expansión.

Nada aquí se ajusta hacia 4.9%/31.5% — esos números no aparecen en ninguna constante de
este archivo. La distancia a esos blancos se calcula DESPUÉS, en el análisis (E5_3_1_analisis.py).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constantes metodológicas (fijas, documentadas en el pre-registro, NO apuntan
# a 4.9%/31.5% — son puramente de calibración numérica).
# ---------------------------------------------------------------------------
N = 200                     # tamaño del anillo (fijo; N-scaling es tarea de E5.6-3)
S = 20                      # semillas por celda (>=16 exigidas)
P_SWAP = 0.02                # prob. de swap dinámico por arista activa y por paso
K_PASOS = 5.0                # "vidas medias" de corte que se buscan cubrir
PASOS_MIN, PASOS_MAX = 100, 3000
GLOBAL_SEED = 20260724        # fecha de pre-registro, solo para reproducibilidad determinista

EPS_GRID = np.logspace(-12, 0, 13).tolist()   # 12 décadas, 13 puntos
R_GRID = np.logspace(-3, 3, 13).tolist()      # 6 décadas, 13 puntos


def rng_for(*keys: int) -> np.random.Generator:
    """RNG determinista derivado de una tupla de claves enteras (reproducible)."""
    return np.random.default_rng([GLOBAL_SEED] + list(keys))


def campo_inicial_batch(eps: float, rng: np.random.Generator):
    """phi0 shape (S,N). pert = 5 modos seno, fase aleatoria POR FILA (semilla)."""
    x = np.linspace(0.0, 1.0, N, endpoint=False)  # (N,)
    if eps <= 0.0:
        return np.ones((S, N), dtype=float)
    fases = rng.uniform(0, 2 * np.pi, size=(S, 5))  # (S,5)
    m = np.arange(1, 6, dtype=float)                # (5,)
    # pert[s,i] = sum_m sin(2*pi*m*x_i + fase[s,m]) / m
    ang = 2 * np.pi * m[None, :, None] * x[None, None, :] + fases[:, :, None]  # (S,5,N)
    pert = (np.sin(ang) / m[None, :, None]).sum(axis=1)  # (S,N)
    pert = pert - pert.mean(axis=1, keepdims=True)
    std = pert.std(axis=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    pert = pert / std
    return 1.0 + eps * pert


def paso_difusion_batch(phi: np.ndarray, activo: np.ndarray) -> np.ndarray:
    """Difusión local vectorizada, solo por aristas activas. phi,activo shape (S,N)."""
    left = np.roll(phi, 1, axis=1)
    right = np.roll(phi, -1, axis=1)
    e_left = np.roll(activo, 1, axis=1)
    e_right = activo
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def swap_step_batch(phi: np.ndarray, activo: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Ruido dinámico: swap de valores en aristas activas, prob P_SWAP, en dos fases
    (aristas pares / impares) para evitar conflictos de solapamiento en la
    vectorización. Un swap es una permutación -> preserva EXACTAMENTE Sum (phi-1)^2
    del anillo completo (ver protocolo §2.9).
    """
    phi = phi.copy()
    for parity in (0, 1):
        idx = np.arange(parity, N, 2)
        j_idx = (idx + 1) % N
        sel_active = activo[:, idx]
        u = rng.random(sel_active.shape)
        do_swap = sel_active & (u < P_SWAP)
        phi_i = phi[:, idx]
        phi_j = phi[:, j_idx]
        new_i = np.where(do_swap, phi_j, phi_i)
        new_j = np.where(do_swap, phi_i, phi_j)
        phi[:, idx] = new_i
        phi[:, j_idx] = new_j
    return phi


def paso_expansion_batch(activo: np.ndarray, H: float, rng: np.random.Generator) -> np.ndarray:
    if H <= 0.0:
        return activo
    activo = activo.copy()
    if H >= 1.0:
        activo[:] = False
        return activo
    u = rng.random(activo.shape)
    cortar = activo & (u < H)
    activo[cortar] = False
    return activo


def medir_D_eps(phi0: np.ndarray) -> float:
    """Difusividad de UN paso (H=0), promediada sobre las S semillas. phi0 shape (S,N)."""
    activo = np.ones((S, N), dtype=bool)
    c0 = phi0.std(axis=1)
    phi1 = paso_difusion_batch(phi0.copy(), activo)
    c1 = phi1.std(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(c0 > 0, (c0 - c1) / np.where(c0 > 0, c0, 1.0), 0.0)
    return float(np.mean(np.clip(d, 0.0, None)))


def ring_domains(activo_row: np.ndarray):
    """
    Componentes conexas del anillo dado un vector booleano de aristas activas
    (activo_row[i] = arista entre sitio i y sitio (i+1)%N). Devuelve lista de listas
    de índices de sitio. Si no hay cortes -> un solo dominio = el anillo completo.
    """
    cortes = np.where(~activo_row)[0]
    if cortes.size == 0:
        return [list(range(N))]
    ultimo_corte = int(cortes[-1])
    orden = [(ultimo_corte + 1 + k) % N for k in range(N)]
    doms = []
    actual = [orden[0]]
    for k in range(1, N):
        prev_node = orden[k - 1]
        node = orden[k]
        if not activo_row[prev_node]:
            doms.append(actual)
            actual = [node]
        else:
            actual.append(node)
    doms.append(actual)
    return doms


def pasos_para_eps(D_eps: float) -> int:
    """
    Ventana de observación FIJA por epsilon (NO depende de r): ~K_PASOS "tiempos de
    difusión" propios del campo. r solo determina H=r*D_eps, la velocidad de corte
    DENTRO de esa misma ventana -- así frac_exp varía con r en vez de saturar siempre
    al mismo nivel (ver protocolo §2.1, corrección de diseño detectada en prueba de humo).
    """
    if D_eps <= 0:
        return PASOS_MIN
    return int(np.clip(np.ceil(K_PASOS / D_eps), PASOS_MIN, PASOS_MAX))


def correr_celda(eps: float, phi0: np.ndarray, D_eps: float, r: float, pasos: int, eps_idx: int, r_idx: int):
    H = float(min(r * D_eps, 1.0)) if D_eps > 0 else 0.0

    rng = rng_for(eps_idx, r_idx, 1)  # rng dedicado a esta celda (ruido+corte+null)
    phi = phi0.copy()
    activo = np.ones((S, N), dtype=bool)

    E_total_row = np.sum((phi0 - 1.0) ** 2, axis=1)  # (S,) — presupuesto declarado por semilla

    E_total_ring_0 = E_total_row.copy()  # para el guardián T6 (no debe crecer)

    for _ in range(pasos):
        phi = paso_difusion_batch(phi, activo)
        phi = swap_step_batch(phi, activo, rng)
        activo = paso_expansion_batch(activo, H, rng)

    E_total_ring_final = np.sum((phi - 1.0) ** 2, axis=1)
    # Guardián T6: la energía total del anillo NO puede haber crecido (difusión
    # contractiva + swap neutro + corte no toca valores). Tolerancia numérica 1e-9.
    viol = E_total_ring_final - E_total_ring_0
    guardian_ok = bool(np.all(viol <= 1e-9 * np.maximum(E_total_ring_0, 1e-300) + 1e-15))

    E_ligada_row = np.zeros(S)
    E_ligada_null_row = np.zeros(S)
    n_dominios_row = np.zeros(S, dtype=int)
    tam_max_dominio_row = np.zeros(S, dtype=int)

    for s in range(S):
        doms = ring_domains(activo[s])
        perm = rng.permutation(phi[s])
        if len(doms) <= 1:
            # 0 o 1 dominio = el anillo entero sigue conectado -> nada "ligado"
            E_ligada_row[s] = 0.0
            E_ligada_null_row[s] = 0.0
            n_dominios_row[s] = 0
            tam_max_dominio_row[s] = N
            continue
        # excluir la componente gigante (mayor tamaño; empate -> menor índice de sitio
        # inicial, convención determinista sin efecto material salvo empate exacto)
        tamanos = [len(d) for d in doms]
        idx_gigante = int(np.argmax(tamanos))  # primer máximo en orden de aparición
        e_lig = 0.0
        e_lig_null = 0.0
        ndom = 0
        for k_dom, dom in enumerate(doms):
            if k_dom == idx_gigante:
                continue
            idxs = np.array(dom, dtype=int)
            e_lig += float(np.sum((phi[s, idxs] - 1.0) ** 2))
            e_lig_null += float(np.sum((perm[idxs] - 1.0) ** 2))
            ndom += 1
        E_ligada_row[s] = e_lig
        E_ligada_null_row[s] = e_lig_null
        n_dominios_row[s] = ndom  # dominios ligados (excluye la gigante)
        tam_max_dominio_row[s] = tamanos[idx_gigante]  # tamaño del remanente/gigante

    with np.errstate(invalid="ignore", divide="ignore"):
        eff_row = np.where(E_total_row > 0, E_ligada_row / np.where(E_total_row > 0, E_total_row, 1.0), np.nan)
        eff_null_row = np.where(E_total_row > 0, E_ligada_null_row / np.where(E_total_row > 0, E_total_row, 1.0), np.nan)

    frac_exp_row = 1.0 - activo.mean(axis=1)

    return {
        "eps": eps,
        "r": r,
        "H": H,
        "D_eps": D_eps,
        "pasos": pasos,
        "guardian_conservacion_ok": guardian_ok,
        "eficiencia_real": eff_row.tolist(),
        "eficiencia_null": eff_null_row.tolist(),
        "E_total": E_total_row.tolist(),
        "E_ligada_real": E_ligada_row.tolist(),
        "E_ligada_null": E_ligada_null_row.tolist(),
        "n_dominios": n_dominios_row.tolist(),
        "tam_max_dominio": tam_max_dominio_row.tolist(),
        "frac_exp": frac_exp_row.tolist(),
    }


def main():
    t0 = time.time()
    filas = []
    guardian_violaciones = []

    eps_list = [0.0] + EPS_GRID  # control + grid de 12 décadas

    for eps_idx, eps in enumerate(eps_list):
        ic_rng = rng_for(eps_idx, 0)
        phi0 = campo_inicial_batch(eps, ic_rng)
        D_eps = medir_D_eps(phi0) if eps > 0 else 0.0
        pasos = pasos_para_eps(D_eps)
        print(f"[eps_idx={eps_idx} eps={eps:.3e}] D_eps={D_eps:.6f} pasos={pasos}", file=sys.stderr, flush=True)

        for r_idx, r in enumerate(R_GRID):
            fila = correr_celda(eps, phi0, D_eps, r, pasos, eps_idx, r_idx)
            filas.append(fila)
            if not fila["guardian_conservacion_ok"]:
                guardian_violaciones.append((eps, r))

        elapsed = time.time() - t0
        print(f"  ... {len(R_GRID)} celdas de r listas (acumulado {elapsed:.1f}s)", file=sys.stderr, flush=True)

    result = {
        "experimento": "E5.3-1",
        "N": N,
        "S_semillas": S,
        "P_SWAP": P_SWAP,
        "K_PASOS": K_PASOS,
        "PASOS_MIN": PASOS_MIN,
        "PASOS_MAX": PASOS_MAX,
        "GLOBAL_SEED": GLOBAL_SEED,
        "eps_grid_12dec": EPS_GRID,
        "eps_control": 0.0,
        "r_grid_6dec": R_GRID,
        "filas": filas,
        "guardian_violaciones": guardian_violaciones,
        "guardian_todas_ok": len(guardian_violaciones) == 0,
        "elapsed_s": time.time() - t0,
    }

    out_json = OUT / "E5_3_1_resultado_crudo.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[guardian_todas_ok] {result['guardian_todas_ok']} (violaciones={len(guardian_violaciones)})", file=sys.stderr)
    print(f"[elapsed] {result['elapsed_s']:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
