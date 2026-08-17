#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.5-1 — Barrido fino de eps->0: curvas E, X, S_ent en el limite
==================================================================

Ver pre-registro: PROTOCOLO_E5.5-1_PREREGISTRO.md (definiciones exactas, ANTES de
correr). Firmado 2026-07-25T00:43:36Z.

Reusa el motor fisico de cs074_rcruz.py SIN EDITARLO (import directo de sus
funciones): paso_difusion, campo_inicial, medir_pasos_lavado, P_LAVADO.

Definiciones (ver protocolo para justificacion completa):
  X(t)     -- HEREDADA verbatim de E5_2_2_anticorrelacion_X_S/E5_2_2_motor.py
              X(t) = (1/N) * sum_i (phi_i(t) - 1)^2
  S_ent(t) -- HEREDADA verbatim de E5_2_2_anticorrelacion_X_S/E5_2_2_motor.py
              p_i = phi_i(t)^2 / sum_j phi_j(t)^2 ; S_ent = -sum p_i ln p_i
  E(t)     -- PROPIA de este experimento (E5_2_1 esta vacio en disco, nada que
              reusar de ahi): E(t) = sum_i phi_i(t)^2 -- es exactamente la
              constante de normalizacion que ya usa S_ent heredada, cierre
              interno del trio (E,X,S_ent).

Barrido MUY FINO cerca de 0 (regla de oro de esta tarea): eps en {0} union 23
puntos log-espaciados en [1e-6, 1e-2] (24 puntos totales) x 16 semillas x r=0
(difusion pura, sin expansion -- eleccion declarada en el protocolo).

No se edita cs074_rcruz.py. No topologia (mismo anillo N nodos del motor base).
Sin commits.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent.parent  # Cosmogenesis/
sys.path.insert(0, str(BASE_DIR))

from cs074_rcruz import (  # noqa: E402  (import tras sys.path fix, motor NO editado)
    paso_difusion,
    campo_inicial,
    medir_pasos_lavado,
    P_LAVADO,
)

# ---------------------------------------------------------------------------
# Pre-registro (congelado ANTES de correr, ver .md hermano)
# ---------------------------------------------------------------------------
N = 200
EPS_LOG_LO, EPS_LOG_HI, N_LOG = 1e-6, 1e-2, 23
EPS_LIST = [0.0] + list(np.geomspace(EPS_LOG_LO, EPS_LOG_HI, N_LOG))
N_SEMILLAS = 16
SEED_BASE = 6000
EPS_CALIBRACION = 1e-2  # extremo superior de ESTE barrido (representativo)
H_FIJO = 0.0  # r=0: difusion pura, sin expansion (declarado en protocolo)

# eps de referencia para trayectoria completa (t=0..pasos, cada paso)
EPS_TRAYECTORIA = sorted({EPS_LIST[0], EPS_LIST[1], EPS_LIST[len(EPS_LIST) // 2], EPS_LIST[-1]})


def energia_E(phi: np.ndarray) -> float:
    """E(t) = sum_i phi_i(t)^2 -- propia, cierre con S_ent heredada (misma normalizacion)."""
    return float(np.sum(phi.astype(np.float64) ** 2))


def exergia_X(phi: np.ndarray) -> float:
    """X(t) = (1/N) sum (phi_i - 1)^2 -- HEREDADA verbatim de E5_2_2_motor.py."""
    return float(np.mean((phi - 1.0) ** 2))


def entropia_S(phi: np.ndarray) -> float:
    """S_ent(t) = -sum p_i ln p_i, p_i = phi_i^2/sum(phi^2) -- HEREDADA verbatim de E5_2_2_motor.py."""
    e = phi.astype(np.float64) ** 2
    total = e.sum()
    if total <= 0:
        return float(np.log(phi.size))
    p = e / total
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def medir(phi: np.ndarray) -> dict:
    return {"E": energia_E(phi), "X": exergia_X(phi), "S": entropia_S(phi)}


def correr_endpoint(N, eps, pasos, seed):
    """Evoluciona difusion pura (H=0) y devuelve (medida_t0, medida_final)."""
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    m0 = medir(phi)
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        # H_FIJO=0 => sin expansion; activo permanece todo True siempre
    mf = medir(phi)
    return m0, mf


def correr_trayectoria(N, eps, pasos, seed):
    """Evoluciona difusion pura y registra (E,X,S) en cada paso, incluyendo t=0."""
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    Es = np.empty(pasos + 1, dtype=np.float64)
    Xs = np.empty(pasos + 1, dtype=np.float64)
    Ss = np.empty(pasos + 1, dtype=np.float64)
    m0 = medir(phi)
    Es[0], Xs[0], Ss[0] = m0["E"], m0["X"], m0["S"]
    for t in range(1, pasos + 1):
        phi = paso_difusion(phi, activo)
        m = medir(phi)
        Es[t], Xs[t], Ss[t] = m["E"], m["X"], m["S"]
    return Es, Xs, Ss


def main():
    t0 = time.time()

    # --- calibracion de pasos (medida, no impuesta), a eps=1e-2 (extremo superior) ---
    cal = medir_pasos_lavado(N, EPS_CALIBRACION, N_SEMILLAS, P_thr=P_LAVADO)
    pasos = cal["pasos"]
    print(
        f"[calibracion] N={N} eps_cal={EPS_CALIBRACION} mediana_lavado={cal['mediana']} "
        f"pasos={pasos} lavo_todas={cal['lavo_todas']}",
        file=sys.stderr, flush=True,
    )

    filas = []
    n_total = len(EPS_LIST) * N_SEMILLAS
    contador = 0

    for eps in EPS_LIST:
        m0_E, m0_X, m0_S = [], [], []
        mf_E, mf_X, mf_S = [], [], []
        derivas = []
        for s in range(N_SEMILLAS):
            seed = SEED_BASE + s
            m0, mf = correr_endpoint(N, eps, pasos, seed)
            m0_E.append(m0["E"]); m0_X.append(m0["X"]); m0_S.append(m0["S"])
            mf_E.append(mf["E"]); mf_X.append(mf["X"]); mf_S.append(mf["S"])
            deriva = abs(mf["E"] - m0["E"]) / m0["E"] if m0["E"] > 0 else float("nan")
            derivas.append(deriva)
            contador += 1
            if contador % 100 == 0:
                print(f"[progreso endpoint] {contador}/{n_total}", file=sys.stderr, flush=True)

        def resumen(arr):
            a = np.array(arr, dtype=np.float64)
            finite = a[np.isfinite(a)]
            if finite.size == 0:
                return {"media": None, "std": None, "n": 0}
            return {"media": float(np.mean(finite)), "std": float(np.std(finite)), "n": int(finite.size)}

        filas.append({
            "eps": eps,
            "pasos": pasos,
            "n_semillas": N_SEMILLAS,
            "t0": {"E": resumen(m0_E), "X": resumen(m0_X), "S_ent": resumen(m0_S)},
            "final": {"E": resumen(mf_E), "X": resumen(mf_X), "S_ent": resumen(mf_S)},
            "deriva_E_relativa": resumen(derivas),
            "ln_N": float(np.log(N)),
        })

    # --- trayectorias completas para eps de referencia ---
    trayectorias = []
    for eps in EPS_TRAYECTORIA:
        E_runs, X_runs, S_runs = [], [], []
        for s in range(N_SEMILLAS):
            seed = SEED_BASE + s
            Es, Xs, Ss = correr_trayectoria(N, eps, pasos, seed)
            E_runs.append(Es); X_runs.append(Xs); S_runs.append(Ss)
        E_runs = np.array(E_runs); X_runs = np.array(X_runs); S_runs = np.array(S_runs)
        # submuestreo de puntos de tiempo (log-espaciado) para no inflar el JSON
        n_pts = min(pasos + 1, 60)
        idx = np.unique(np.round(np.geomspace(1, pasos + 1, n_pts)).astype(int) - 1)
        idx = np.clip(idx, 0, pasos)
        trayectorias.append({
            "eps": eps,
            "pasos": pasos,
            "n_semillas": N_SEMILLAS,
            "t_idx": [int(i) for i in idx],
            "E_media": [round(float(v), 8) for v in E_runs[:, idx].mean(axis=0)],
            "E_std": [round(float(v), 8) for v in E_runs[:, idx].std(axis=0)],
            "X_media": [round(float(v), 8) for v in X_runs[:, idx].mean(axis=0)],
            "X_std": [round(float(v), 8) for v in X_runs[:, idx].std(axis=0)],
            "S_media": [round(float(v), 8) for v in S_runs[:, idx].mean(axis=0)],
            "S_std": [round(float(v), 8) for v in S_runs[:, idx].std(axis=0)],
        })
        print(f"[trayectoria] eps={eps} lista", file=sys.stderr, flush=True)

    result = {
        "experimento": "E5.5-1 barrido fino eps->0: curvas E, X, S_ent",
        "definiciones": {
            "E_t": "sum_i phi_i(t)^2  [PROPIA, cierre con S_ent heredada]",
            "X_t": "(1/N) * sum_i (phi_i(t) - 1)^2  [HEREDADA verbatim de E5.2-2]",
            "S_ent_t": "-sum_i p_i ln p_i, p_i = phi_i(t)^2 / sum_j phi_j(t)^2  [HEREDADA verbatim de E5.2-2]",
        },
        "reuso_declarado": {
            "E5_2_1_balance_deriva": "VACIO en disco al momento de escribir (nada que reusar)",
            "E5_2_2_anticorrelacion_X_S": "X y S_ent reusadas verbatim de E5_2_2_motor.py",
        },
        "N": N,
        "eps_list": EPS_LIST,
        "n_semillas": N_SEMILLAS,
        "H_fijo": H_FIJO,
        "pasos": pasos,
        "calibracion_lavado": cal,
        "filas": filas,
        "trayectorias_referencia": trayectorias,
        "elapsed_s": time.time() - t0,
    }

    out_json = HERE / "E5_5_1_resultados.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[elapsed] {result['elapsed_s']:.1f}s", file=sys.stderr)

    # resumen corto a stdout
    print(f"[resumen] eps_puntos={len(EPS_LIST)} semillas={N_SEMILLAS} pasos={pasos}")
    e0 = filas[0]
    print(f"[eps=0] E_final={e0['final']['E']['media']:.6f} X_final={e0['final']['X']['media']:.10f} "
          f"S_final={e0['final']['S_ent']['media']:.10f} ln_N={e0['ln_N']:.6f}")
    eN = filas[-1]
    print(f"[eps={eN['eps']:.2e}] E_final={eN['final']['E']['media']:.6f} X_final={eN['final']['X']['media']:.10f} "
          f"S_final={eN['final']['S_ent']['media']:.10f} deriva_E={eN['deriva_E_relativa']['media']:.3e}")


if __name__ == "__main__":
    main()
