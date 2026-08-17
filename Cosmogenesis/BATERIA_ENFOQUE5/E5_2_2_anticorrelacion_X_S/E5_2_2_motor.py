#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.2-2 — Anticorrelacion exergia<->entropia: ¿X baja exactamente lo que S_ent sube?
====================================================================================

Ver pre-registro: E5_2_2_PROTOCOLO_PREREGISTRO.md (definiciones exactas, ANTES de correr).

Reusa el motor fisico de cs074_rcruz.py SIN EDITARLO (import directo de sus funciones):
paso_difusion, paso_expansion, campo_inicial, medir_D, medir_pasos_lavado.

Define, POR CADA PASO de cada corrida (eps, r, semilla):

  X(t)     = (1/N) * sum_i (phi_i(t) - 1)^2                    [exergia, momentos, ref fija]
  S_ent(t) = -sum_i p_i(t) ln p_i(t),  p_i = phi_i(t)^2/sum(phi^2)   [entropia Shannon espacial]

Mide la correlacion de Pearson temporal REAL entre X(t) y S_ent(t) dentro de cada corrida, y el
NULL (barajado temporal: se permuta el orden de X(t) manteniendo S_ent(t) en su orden original;
se reporta tambien el barajado inverso como verificacion secundaria).

Barrido sobredimensionado: eps en 12 decadas [1e-12..1] (+0), r en 6 decadas [1e-3..1e3] (+0),
16 semillas por celda. Ver PROTOCOLO para el detalle completo.

No se edita cs074_rcruz.py. No topologia (mismo anillo N nodos del motor base). Sin commits.
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
    paso_expansion,
    campo_inicial,
    medir_D,
    medir_pasos_lavado,
    P_LAVADO,
    MARGEN_LAVADO,
)

# ---------------------------------------------------------------------------
# Pre-registro (congelado ANTES de correr, ver .md hermano)
# ---------------------------------------------------------------------------
N = 200
EPS_LIST = [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
R_LIST = [0.0, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
N_SEMILLAS = 16
SEED_BASE = 5000
EPS_CALIBRACION = 1e-2  # representativo, ni extremo ni degenerado
PASS_UMBRAL = -0.9


def exergia_X(phi: np.ndarray) -> float:
    """X(t) = (1/N) sum (phi_i - 1)^2 -- desviacion cuadratica del equilibrio uniforme (phi_eq=1)."""
    return float(np.mean((phi - 1.0) ** 2))


def entropia_S(phi: np.ndarray) -> float:
    """S_ent(t) = -sum p_i ln p_i, p_i = phi_i^2 / sum(phi_j^2) -- Shannon espacial de la densidad phi^2."""
    e = phi.astype(np.float64) ** 2
    total = e.sum()
    if total <= 0:
        # campo identicamente nulo (degenerado): distribucion uniforme por convencion -> entropia maxima
        return float(np.log(phi.size))
    p = e / total
    # evitar log(0): p_i=0 no contribuye (limite x ln x -> 0)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def correr_trayectoria(N, eps, H, pasos, seed):
    """Evoluciona una corrida y registra X(t), S_ent(t) en cada paso (sin permutar al final:
    aqui NO aplicamos el NULL de cs074_rcruz -- ese NULL es otro experimento. El NULL de ESTE
    experimento es el barajado temporal de las series X/S, hecho despues, fuera de esta funcion)."""
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    xs = np.empty(pasos, dtype=np.float64)
    ss = np.empty(pasos, dtype=np.float64)
    for t in range(pasos):
        phi = paso_difusion(phi, activo)
        activo = paso_expansion(activo, H, rng)
        xs[t] = exergia_X(phi)
        ss[t] = entropia_S(phi)
    return xs, ss


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa <= 1e-15 or sb <= 1e-15:
        return float("nan")
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def null_barajado(xs: np.ndarray, ss: np.ndarray, seed: int):
    """Barajado temporal: permuta el orden de xs (manteniendo ss fijo) y viceversa (verificacion
    secundaria). Reordena los pasos, rompe la relacion causal instantanea, conserva los valores."""
    rng = np.random.default_rng(seed)
    T = xs.size
    perm1 = rng.permutation(T)
    r_null_x = pearson(xs[perm1], ss)
    perm2 = rng.permutation(T)
    r_null_s = pearson(xs, ss[perm2])
    return r_null_x, r_null_s


def main():
    t0 = time.time()

    # --- calibracion de pasos (medida, no impuesta) ---
    cal = medir_pasos_lavado(N, EPS_CALIBRACION, N_SEMILLAS, P_thr=P_LAVADO)
    pasos = cal["pasos"]
    print(
        f"[calibracion] N={N} eps_cal={EPS_CALIBRACION} mediana_lavado={cal['mediana']} "
        f"pasos={pasos} lavo_todas={cal['lavo_todas']}",
        file=sys.stderr, flush=True,
    )

    # --- D medido por eps (para H = min(r*D,1)) ---
    D_por_eps = {}
    for eps in EPS_LIST:
        if eps <= 0:
            D_por_eps[eps] = 0.0
            continue
        Ds = [medir_D(N, eps, SEED_BASE + s) for s in range(N_SEMILLAS)]
        D_por_eps[eps] = float(np.mean(Ds))
    print(f"[D_medido] {D_por_eps}", file=sys.stderr, flush=True)

    filas = []
    n_total = len(EPS_LIST) * len(R_LIST) * N_SEMILLAS
    contador = 0
    ejemplos = []  # trayectorias completas para inspeccion visual (r bajo, ~1, alto)
    r_ejemplo_targets = {0.1, 1.0, 100.0}
    eps_ejemplo = 1e-1

    for eps in EPS_LIST:
        D = D_por_eps[eps]
        for r_tgt in R_LIST:
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
                r_eff = H / D
            else:
                H = 0.0 if r_tgt == 0 else 1.0
                r_eff = 0.0 if r_tgt == 0 else float("inf")

            reales, nulos_x, nulos_s = [], [], []
            for s in range(N_SEMILLAS):
                seed = SEED_BASE + s
                xs, ss = correr_trayectoria(N, eps, H, pasos, seed)
                r_real = pearson(xs, ss)
                r_null_x, r_null_s = null_barajado(xs, ss, seed=seed + 900_000)
                reales.append(r_real)
                nulos_x.append(r_null_x)
                nulos_s.append(r_null_s)

                if eps == eps_ejemplo and r_tgt in r_ejemplo_targets and s == 0:
                    ejemplos.append({
                        "eps": eps, "r_target": r_tgt, "H": H, "seed": seed,
                        "X_t": [round(float(v), 8) for v in xs],
                        "S_t": [round(float(v), 8) for v in ss],
                        "r_pearson_real": r_real,
                    })

                contador += 1
                if contador % 200 == 0:
                    print(f"[progreso] {contador}/{n_total}", file=sys.stderr, flush=True)

            reales_a = np.array(reales, dtype=np.float64)
            nx_a = np.array(nulos_x, dtype=np.float64)
            ns_a = np.array(nulos_s, dtype=np.float64)

            def resumen(arr):
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return {"media": None, "mediana": None, "std": None, "n_validos": 0}
                return {
                    "media": float(np.mean(finite)),
                    "mediana": float(np.median(finite)),
                    "std": float(np.std(finite)),
                    "n_validos": int(finite.size),
                }

            res_real = resumen(reales_a)
            res_nx = resumen(nx_a)
            res_ns = resumen(ns_a)

            pass_real = (res_real["media"] is not None) and (res_real["media"] < PASS_UMBRAL)
            null_ausente = (res_nx["media"] is None) or (res_nx["media"] >= PASS_UMBRAL)

            filas.append({
                "eps": eps,
                "r_target": r_tgt,
                "H": round(H, 8),
                "D": round(D, 8),
                "r_efectivo": r_eff,
                "pasos": pasos,
                "n_semillas": N_SEMILLAS,
                "r_pearson_real": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res_real.items()},
                "r_pearson_null_shuffleX": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res_nx.items()},
                "r_pearson_null_shuffleS": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res_ns.items()},
                "pass_real_lt_-0.9": bool(pass_real),
                "null_ausente": bool(null_ausente),
                "pass_experimento_celda": bool(pass_real and null_ausente),
                "valores_real": [round(v, 6) if np.isfinite(v) else None for v in reales],
            })

    result = {
        "experimento": "E5.2-2 anticorrelacion X<->S_ent",
        "definiciones": {
            "X_t": "(1/N) * sum_i (phi_i(t) - 1)^2",
            "S_ent_t": "-sum_i p_i ln p_i, p_i = phi_i(t)^2 / sum_j phi_j(t)^2",
        },
        "N": N,
        "eps_list": EPS_LIST,
        "r_list": R_LIST,
        "n_semillas": N_SEMILLAS,
        "pasos": pasos,
        "calibracion_lavado": cal,
        "D_por_eps": D_por_eps,
        "pass_umbral": PASS_UMBRAL,
        "filas": filas,
        "ejemplos_trayectorias": ejemplos,
        "elapsed_s": time.time() - t0,
    }

    out_json = HERE / "E5_2_2_resultados.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[elapsed] {result['elapsed_s']:.1f}s", file=sys.stderr)

    # resumen corto a stdout
    n_pass = sum(1 for f in filas if f["pass_experimento_celda"])
    print(f"[resumen] celdas totales={len(filas)} celdas PASS(r<-0.9 real, null ausente)={n_pass}")


if __name__ == "__main__":
    main()
