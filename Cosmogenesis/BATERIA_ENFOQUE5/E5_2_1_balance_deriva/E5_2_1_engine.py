#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.2-1 — Balance de energía paso a paso: deriva del total sobre corridas largas
=================================================================================
Motor propio (prefijo E5_2_1_), física IDÉNTICA a cs074_rcruz.py (leído, NO editado),
reimplementada en LOTE (vectorizada sobre columnas = combinaciones r×semilla) para poder
auditar la deriva de E_total EN CADA PASO de corridas de hasta 1e5 pasos, sobre la grilla
completa pre-registrada en PROTOCOLO_E5.2-1_PREREGISTRO.md.

Definición de E_total (idéntica a §3 del protocolo, no se repite el álgebra aquí, solo
se implementa):

    E_campo(t) = N * mean(phi(t))**2
    X(t)       = N * Var(phi(t))
    S_ent(t)   = X(0) - X(t)                          [forma "estado", O(1) por paso]
    E_total(t) = E_campo(t) + X(t) + S_ent(t) = E_campo(t) + X(0)   [forma reducida]

Verificación cruzada #2 (protocolo §7.2): además de S_ent "estado" (X0-X(t)), se lleva un
S_ent "ledger" acumulado paso a paso (suma de las caídas de varianza observadas), que en
aritmética exacta debe coincidir con la forma estado — la diferencia entre ambas expone
error de acumulación de punto flotante sobre corridas largas, no un error conceptual.

NOTA DE EFICIENCIA (protocolo §5): se corre UNA sola trayectoria de 1e5 pasos por columna
(r,semilla) y se extraen checkpoints en pasos∈{1e2,1e3,1e4,1e5} de esa misma trayectoria.
Esto es matemáticamente idéntico a correr 4 corridas independientes de esas longitudes con
la misma semilla, porque el consumo del generador de números aleatorios en cada paso no
depende de cuántos pasos falten (Markoviano dado el estado + el stream de RNG).

Esquema de números aleatorios (declarado, T1): un único Generator (PCG64) por valor de eps,
consumido en orden estándar de numpy sobre arreglos (N, ncols) — no son streams
independientes por columna en el sentido criptográfico, pero es la práctica estándar para
vectorizar Monte Carlo con numpy y estadísticamente válido para las 12 semillas por r que
pide el barrido. Las condiciones iniciales SÍ usan semillas independientes explícitas
(1000+s, igual que la base) vía la función `campo_inicial` ORIGINAL de cs074_rcruz.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cs074_rcruz import campo_inicial as campo_inicial_base
from cs074_rcruz import medir_D as medir_D_base

OUT = Path(__file__).resolve().parent

N = 200
EPS_LIST = [0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 1.0]
R_LIST = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3]
N_SEEDS = 12
PASOS_MAX = 100_000
CHECKPOINTS = [100, 1_000, 10_000, 100_000]
UMBRAL_DERIVA = 1e-6
SEED_BASE_INICIAL = 1000  # mismo esquema que corrida() de la base: seed = 1000+s


def paso_difusion_batch(phi: np.ndarray, activo: np.ndarray) -> np.ndarray:
    """Idéntica a paso_difusion() de cs074_rcruz.py, vectorizada sobre columnas (axis=0 es el anillo)."""
    left = np.roll(phi, 1, axis=0)
    right = np.roll(phi, -1, axis=0)
    e_left = np.roll(activo, 1, axis=0)
    e_right = activo
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def paso_expansion_batch(activo: np.ndarray, H_row: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Idéntica a paso_expansion() de cs074_rcruz.py, vectorizada; H_row shape (1,ncols)."""
    u = rng.random(activo.shape)
    cortar = activo & (u < H_row)
    activo2 = activo.copy()
    activo2[cortar] = False
    full = (H_row[0] >= 1.0)
    if np.any(full):
        activo2[:, full] = False
    return activo2


def medir_D_prom(N, eps, n_seeds, seed_base=SEED_BASE_INICIAL):
    Ds = [medir_D_base(N, eps, seed_base + s) for s in range(n_seeds)]
    return float(np.mean(Ds)), Ds


def construir_columnas(N, eps, r_list, n_seeds):
    """
    Devuelve phi0 (N, n_r*n_seeds), H_row (1, n_r*n_seeds), y metadatos por columna
    (r_target, seed). Columnas ordenadas: r varía lento, semilla varía rápido
    (col = r_idx*n_seeds + s).
    """
    D, D_list = medir_D_prom(N, eps, n_seeds)

    # condición inicial: depende SOLO de (eps, semilla), no de r -> se genera una vez por
    # semilla y se repite (tile) para cada r.
    phi0_por_seed = np.empty((N, n_seeds))
    for s in range(n_seeds):
        rng0 = np.random.default_rng(SEED_BASE_INICIAL + s)
        phi_s, _ = campo_inicial_base(N, eps, rng0)
        phi0_por_seed[:, s] = phi_s

    n_r = len(r_list)
    ncols = n_r * n_seeds
    phi0 = np.tile(phi0_por_seed, (1, n_r))  # (N, n_r*n_seeds)

    H_list = []
    meta = []
    for r_tgt in r_list:
        if D > 0:
            H = float(min(r_tgt * D, 1.0))
        else:
            H = 0.0 if r_tgt == 0.0 else 1.0
        H_list.append(H)
        for s in range(n_seeds):
            meta.append({"r_target": r_tgt, "H": H, "seed": SEED_BASE_INICIAL + s, "seed_idx": s})

    H_row = np.repeat(np.array(H_list, dtype=float), n_seeds).reshape(1, ncols)
    return phi0, H_row, meta, D, D_list


def correr_eps(eps, r_list=R_LIST, n_seeds=N_SEEDS, pasos_max=PASOS_MAX,
               checkpoints=CHECKPOINTS, umbral=UMBRAL_DERIVA, rng_seed_step=None):
    phi0, H_row, meta, D, D_list = construir_columnas(N, eps, r_list, n_seeds)
    ncols = phi0.shape[1]

    phi = phi0.copy()
    activo = np.ones((N, ncols), dtype=bool)

    mean0 = phi.mean(axis=0)
    var0 = phi.var(axis=0)
    E_campo0 = N * mean0 ** 2
    X0 = N * var0
    S_ent0 = np.zeros(ncols)
    E_total0 = E_campo0 + X0 + S_ent0  # = E_campo0 + X0

    # trackers acumulados (running), sin guardar la trayectoria completa
    deriva_max = np.zeros(ncols)
    paso_primer_cruce = np.full(ncols, -1, dtype=np.int64)
    S_ent_state_min = np.zeros(ncols)  # empieza en 0 (S_ent(0)=0)
    S_ent_ledger = np.zeros(ncols)
    ledger_vs_state_max_diff = np.zeros(ncols)
    var_prev = var0.copy()

    checkpoint_records = {}  # cp -> dict de arrays

    if rng_seed_step is None:
        rng_seed_step = 900_000
    rng = np.random.default_rng(rng_seed_step)

    cp_set = set(checkpoints)
    cp_max = max(checkpoints)

    for t in range(1, pasos_max + 1):
        phi = paso_difusion_batch(phi, activo)
        activo = paso_expansion_batch(activo, H_row, rng)

        mean_t = phi.mean(axis=0)
        var_t = phi.var(axis=0)
        E_campo_t = N * mean_t ** 2
        X_t = N * var_t

        S_ent_state_t = X0 - X_t
        S_ent_ledger = S_ent_ledger + N * (var_prev - var_t)
        var_prev = var_t

        E_total_t_reducido = E_campo_t + X0
        E_total_t_ledger = E_campo_t + X_t + S_ent_ledger

        deriva_t = np.abs(E_total_t_reducido - E_total0) / np.abs(E_total0)
        deriva_max = np.maximum(deriva_max, deriva_t)

        cruzo_ahora = (deriva_t >= umbral) & (paso_primer_cruce == -1)
        paso_primer_cruce[cruzo_ahora] = t

        S_ent_state_min = np.minimum(S_ent_state_min, S_ent_state_t)

        diff_ledger = np.abs(E_total_t_ledger - E_total_t_reducido) / np.abs(E_total0)
        ledger_vs_state_max_diff = np.maximum(ledger_vs_state_max_diff, diff_ledger)

        if t in cp_set:
            checkpoint_records[t] = {
                "deriva_max_hasta_aqui": deriva_max.copy(),
                "paso_primer_cruce_hasta_aqui": paso_primer_cruce.copy(),
                "S_ent_state_min_hasta_aqui": S_ent_state_min.copy(),
                "ledger_vs_state_max_diff_hasta_aqui": ledger_vs_state_max_diff.copy(),
                "deriva_en_este_paso": deriva_t.copy(),
                "S_ent_state_en_este_paso": S_ent_state_t.copy(),
                "mean_en_este_paso": mean_t.copy(),
                "n_activas_en_este_paso": activo.sum(axis=0).copy(),
            }
        if t >= cp_max:
            break

    filas = []
    for col in range(ncols):
        m = meta[col]
        fila = {
            "eps": eps,
            "r_target": m["r_target"],
            "H": m["H"],
            "D": D,
            "r_eff": (m["H"] / D) if D > 0 else float("nan"),
            "seed": m["seed"],
            "N": N,
            "E_campo0": float(E_campo0[col]),
            "X0": float(X0[col]),
            "S_ent0": float(S_ent0[col]),
            "E_total0": float(E_total0[col]),
            "checkpoints": {},
        }
        for cp in checkpoints:
            rec = checkpoint_records.get(cp)
            if rec is None:
                fila["checkpoints"][str(cp)] = None
                continue
            deriva_max_cp = float(rec["deriva_max_hasta_aqui"][col])
            fila["checkpoints"][str(cp)] = {
                "deriva_max": deriva_max_cp,
                "deriva_en_paso": float(rec["deriva_en_este_paso"][col]),
                "PASS": bool(deriva_max_cp < umbral),
                "paso_primer_cruce": (int(rec["paso_primer_cruce_hasta_aqui"][col])
                                       if rec["paso_primer_cruce_hasta_aqui"][col] >= 0 else None),
                "S_ent_state_min_hasta_aqui": float(rec["S_ent_state_min_hasta_aqui"][col]),
                "S_ent_state_en_paso": float(rec["S_ent_state_en_este_paso"][col]),
                "ledger_vs_state_max_diff": float(rec["ledger_vs_state_max_diff_hasta_aqui"][col]),
                "mean_phi_en_paso": float(rec["mean_en_este_paso"][col]),
                "n_activas_en_paso": int(rec["n_activas_en_este_paso"][col]),
            }
        fila["deriva_max_global"] = float(deriva_max[col])
        fila["paso_primer_cruce_global"] = (int(paso_primer_cruce[col]) if paso_primer_cruce[col] >= 0 else None)
        fila["S_ent_state_min_global"] = float(S_ent_state_min[col])
        fila["ledger_vs_state_max_diff_global"] = float(ledger_vs_state_max_diff[col])
        fila["PASS_global"] = bool(deriva_max[col] < umbral)
        filas.append(fila)

    return filas, {"eps": eps, "D": D, "D_por_semilla": D_list, "ncols": ncols}


def main():
    t0 = time.time()
    todas_filas = []
    metas = []
    for eps_idx, eps in enumerate(EPS_LIST):
        t_eps0 = time.time()
        filas, meta = correr_eps(eps, rng_seed_step=900_000 + eps_idx)
        todas_filas.extend(filas)
        meta["elapsed_s"] = time.time() - t_eps0
        metas.append(meta)
        n_pass = sum(1 for f in filas if f["PASS_global"])
        print(f"[eps={eps:g}] filas={len(filas)} PASS_global={n_pass}/{len(filas)} "
              f"D={meta['D']:.6g} elapsed={meta['elapsed_s']:.1f}s", file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    resultado = {
        "experimento": "E5.2-1",
        "N": N,
        "eps_list": EPS_LIST,
        "r_list": R_LIST,
        "n_seeds": N_SEEDS,
        "pasos_max": PASOS_MAX,
        "checkpoints": CHECKPOINTS,
        "umbral_deriva": UMBRAL_DERIVA,
        "meta_por_eps": metas,
        "filas": todas_filas,
        "elapsed_s": elapsed,
    }
    out_path = OUT / "E5_2_1_resultado_crudo.json"
    out_path.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[archivo] {out_path}", file=sys.stderr)
    print(f"[elapsed total] {elapsed:.1f}s", file=sys.stderr)

    n_pass_100000 = sum(1 for f in todas_filas if f["checkpoints"]["100000"] and f["checkpoints"]["100000"]["PASS"])
    n_pass_100 = sum(1 for f in todas_filas if f["checkpoints"]["100"] and f["checkpoints"]["100"]["PASS"])
    print(f"[resumen] filas={len(todas_filas)} PASS@100={n_pass_100} PASS@100000={n_pass_100000}", file=sys.stderr)


if __name__ == "__main__":
    main()
