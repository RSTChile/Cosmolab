#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2-3 — Irreversibilidad del corte: ¿el congelamiento requiere cortes permanentes?
===================================================================================

Clon fiel de la física de cs074_rcruz.py (NO se edita el original). Añade UNA
sola cosa: probabilidad de RECONEXIÓN de aristas cortadas, barrida de 0
(irreversible = modelo original) a 1 (totalmente reversible). Ver protocolo
pre-registrado en esta misma carpeta:
  PROTOCOLO_F2-3_PREREGISTRO.md

Todo lo demás (campo_inicial, paso_difusion, persistencia, medir_D,
medir_pasos_lavado, NULL por permutación) es copia byte-a-byte de la lógica
de cs074_rcruz.py, para que en prob_reconexion=0 este motor sea idéntico al
modelo base (verificación de identidad del mecanismo, T2).

Uso:
  python3 F2_3_motor_reconexion.py            # corre la grilla principal + control eps=0
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent

# ---- idéntico a cs074_rcruz.py -------------------------------------------
P_LAVADO = 0.05
MARGEN_LAVADO = 1.15
R_TARGETS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]

# ---- eje nuevo de F2-3, pre-registrado -------------------------------------
P_RECON_TARGETS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 1.0]


def campo_inicial(N, eps, rng):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    fondo = np.ones(N, dtype=float)
    if eps <= 0.0:
        return fondo, x
    pert = np.zeros(N, dtype=float)
    for m in range(1, 6):
        fase = rng.uniform(0, 2 * np.pi)
        pert += np.sin(2 * np.pi * m * x + fase) / m
    pert -= pert.mean()
    if pert.std() > 0:
        pert = pert / pert.std()
    return fondo + eps * pert, x


def paso_difusion(phi, activo):
    """Idéntico a cs074_rcruz.py — difusión solo por aristas vivas."""
    left = np.roll(phi, 1)
    right = np.roll(phi, -1)
    e_left = np.roll(activo, 1)
    e_right = activo
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def paso_expansion(activo, H, rng):
    """Idéntico a cs074_rcruz.py — corte Bernoulli por arista viva, prob H."""
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


def paso_reconexion(activo, p_recon, rng):
    """
    ÚNICA ADICIÓN de F2-3 sobre cs074_rcruz.py.

    Reconecta aristas inactivas (cortadas) con probabilidad p_recon,
    independiente por arista, DESPUÉS del corte del mismo paso (incluye
    aristas recién cortadas ese mismo paso). p_recon=0 -> no-op (idéntico al
    modelo base). p_recon=1 -> todas las inactivas se reconectan cada paso
    (reversibilidad total: nunca queda un corte que sobreviva al paso
    siguiente).
    """
    if p_recon <= 0.0:
        return activo
    inactivas = ~activo
    if not inactivas.any():
        return activo
    if p_recon >= 1.0:
        return activo | inactivas
    u = rng.random(activo.shape)
    reconectar = inactivas & (u < p_recon)
    return activo | reconectar


def paso_expansion_reconexion(activo, H, p_recon, rng):
    activo = paso_expansion(activo, H, rng)
    activo = paso_reconexion(activo, p_recon, rng)
    return activo


def persistencia(phi, contraste0):
    """Idéntico a cs074_rcruz.py — observable PRINCIPAL."""
    if contraste0 <= 0 or phi.std() <= 1e-12:
        return 0.0
    c = np.corrcoef(phi, np.roll(phi, 1))[0, 1]
    if not np.isfinite(c):
        c = 0.0
    c = max(0.0, float(c))
    v = float(phi.var() / (contraste0 ** 2))
    return float(c * v)


def medir_D(N, eps, seed):
    """Idéntico a cs074_rcruz.py."""
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    c0 = phi.std()
    if c0 <= 0:
        return 0.0
    phi1 = paso_difusion(phi, activo)
    c1 = phi1.std()
    return max(0.0, float((c0 - c1) / c0))


def medir_pasos_lavado(N, eps, semillas, P_thr=P_LAVADO, max_steps=200000, check_every=50):
    """Idéntico a cs074_rcruz.py (calibración a H=0, sin reconexión: no aplica)."""
    tiempos = []
    for s in range(semillas):
        rng = np.random.default_rng(10_000 + s)
        phi, _ = campo_inicial(N, eps, rng)
        activo = np.ones(N, dtype=bool)
        c0 = float(phi.std())
        if c0 <= 0:
            tiempos.append(0)
            continue
        t_hit = None
        for t in range(1, max_steps + 1):
            phi = paso_difusion(phi, activo)
            if t % check_every == 0:
                if persistencia(phi, c0) < P_thr:
                    t_hit = t
                    break
        if t_hit is None:
            t_hit = max_steps
        tiempos.append(t_hit)
    med = int(np.median(tiempos))
    pasos = int(np.ceil(med * MARGEN_LAVADO))
    return {
        "tiempos": tiempos,
        "mediana": med,
        "pasos": pasos,
        "P_thr": P_thr,
        "lavo_todas": all(t < max_steps for t in tiempos),
    }


def evolucionar(phi, activo, H, p_recon, pasos, rng, null=False):
    contraste0 = float(phi.std())
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        activo = paso_expansion_reconexion(activo, H, p_recon, rng)
    if null:
        phi = rng.permutation(phi)
    return phi, activo, contraste0


def corrida(N, eps, H, p_recon, pasos, seed, null=False):
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    phi, activo, c0 = evolucionar(phi, activo, H, p_recon, pasos, rng, null=null)
    P = persistencia(phi, c0)
    frac_exp = 1.0 - float(activo.mean())  # fracción de aristas cortadas al final (observable mecanístico)
    std_ratio = float(phi.std() / c0) if c0 > 0 else 0.0  # observable secundario independiente
    return {"P": P, "frac_exp": frac_exp, "std_ratio": std_ratio}


def barrido_principal(N, eps, r_targets, p_recon_targets, semillas, pasos_fijo, D):
    filas = []
    for r_tgt in r_targets:
        H = float(min(r_tgt * D, 1.0)) if D > 0 else (0.0 if r_tgt == 0 else 1.0)
        r_eff = H / D if D > 0 else (float("inf") if r_tgt > 0 else 0.0)
        for p_recon in p_recon_targets:
            Preal, Pnull, fexp_real, fexp_null, srr, srn = [], [], [], [], [], []
            for s in range(semillas):
                seed = 1000 + s
                rr = corrida(N, eps, H, p_recon, pasos_fijo, seed=seed, null=False)
                nn = corrida(N, eps, H, p_recon, pasos_fijo, seed=seed, null=True)
                Preal.append(rr["P"]); Pnull.append(nn["P"])
                fexp_real.append(rr["frac_exp"]); fexp_null.append(nn["frac_exp"])
                srr.append(rr["std_ratio"]); srn.append(nn["std_ratio"])
            Preal = np.array(Preal); Pnull = np.array(Pnull)
            sd = np.sqrt((Preal.var() + Pnull.var()) / 2.0)
            sd = max(sd, 1.0 / max(len(Preal), 1))
            z = float((Preal.mean() - Pnull.mean()) / sd)
            filas.append({
                "eps": eps, "r_target": r_tgt, "H": H, "D": D, "r": r_eff,
                "p_recon": p_recon, "pasos": pasos_fijo,
                "P_real": float(Preal.mean()), "P_null": float(Pnull.mean()),
                "P_real_std": float(Preal.std()), "P_null_std": float(Pnull.std()),
                "P_real_semillas": [float(v) for v in Preal],
                "P_null_semillas": [float(v) for v in Pnull],
                "z": z,
                "frac_exp_real_mean": float(np.mean(fexp_real)),
                "frac_exp_null_mean": float(np.mean(fexp_null)),
                "std_ratio_real_mean": float(np.mean(srr)),
                "std_ratio_null_mean": float(np.mean(srn)),
            })
    return filas


def barrido_control_eps0(N, r_targets, p_recon_targets, semillas, pasos_fijo, D_eps0):
    """Control sanity: eps=0 estricto -> P debe ser 0 sin importar p_recon."""
    filas = []
    eps = 0.0
    for r_tgt in r_targets:
        H = float(min(r_tgt * D_eps0, 1.0)) if D_eps0 > 0 else (0.0 if r_tgt == 0 else 1.0)
        for p_recon in p_recon_targets:
            Preal = []
            for s in range(semillas):
                seed = 2000 + s
                rr = corrida(N, eps, H, p_recon, pasos_fijo, seed=seed, null=False)
                Preal.append(rr["P"])
            filas.append({
                "eps": eps, "r_target": r_tgt, "H": H, "p_recon": p_recon,
                "P_real_mean": float(np.mean(Preal)), "P_real_max": float(np.max(Preal)),
            })
    return filas


def main():
    t0 = time.time()
    ts_inicio = time.strftime("%Y-%m-%d %H:%M:%S %z")

    N = 200
    EPS = 0.1
    SEMILLAS = 12
    SEMILLAS_CONTROL = 4
    R_CONTROL = [0.0, 1.0, 100.0]
    P_RECON_CONTROL = [0.0, 0.5, 1.0]

    print(f"[F2-3] inicio {ts_inicio}", file=sys.stderr, flush=True)

    # Calibración única (idéntica en espíritu a cs074_rcruz "produccion"):
    # H=0, sin reconexión (no aplica a H=0), eps=0.1, 12 semillas.
    cal = medir_pasos_lavado(N, EPS, SEMILLAS)
    pasos_fijo = cal["pasos"]
    print(f"[calibracion] N={N} eps={EPS} mediana_lavado={cal['mediana']} "
          f"pasos={pasos_fijo} lavo_todas={cal['lavo_todas']}", file=sys.stderr, flush=True)

    D = float(np.mean([medir_D(N, EPS, s) for s in range(SEMILLAS)]))
    print(f"[D medida] eps={EPS} D={D}", file=sys.stderr, flush=True)

    t_grid0 = time.time()
    filas = barrido_principal(N, EPS, R_TARGETS, P_RECON_TARGETS, SEMILLAS, pasos_fijo, D)
    t_grid = time.time() - t_grid0
    print(f"[grilla principal] {len(filas)} celdas en {t_grid:.1f}s", file=sys.stderr, flush=True)

    D_eps0 = float(np.mean([medir_D(N, 0.0, s) for s in range(SEMILLAS_CONTROL)]))
    t_ctrl0 = time.time()
    filas_control = barrido_control_eps0(N, R_CONTROL, P_RECON_CONTROL, SEMILLAS_CONTROL, pasos_fijo, D_eps0)
    t_ctrl = time.time() - t_ctrl0
    print(f"[control eps=0] {len(filas_control)} celdas en {t_ctrl:.1f}s", file=sys.stderr, flush=True)

    ts_fin = time.strftime("%Y-%m-%d %H:%M:%S %z")
    elapsed = time.time() - t0

    result = {
        "experimento": "F2-3_irreversibilidad_corte",
        "base_no_editada": "cs074_rcruz.py",
        "protocolo": "PROTOCOLO_F2-3_PREREGISTRO.md",
        "ts_inicio": ts_inicio,
        "ts_fin": ts_fin,
        "elapsed_s": elapsed,
        "N": N,
        "eps_fijo": EPS,
        "semillas": SEMILLAS,
        "pasos_fijo": pasos_fijo,
        "calibracion_lavado": cal,
        "D_medida": D,
        "r_targets": R_TARGETS,
        "p_recon_targets": P_RECON_TARGETS,
        "filas": filas,
        "control_eps0": {
            "D_eps0": D_eps0,
            "semillas_control": SEMILLAS_CONTROL,
            "r_control": R_CONTROL,
            "p_recon_control": P_RECON_CONTROL,
            "filas": filas_control,
        },
    }

    out_json = OUT / "F2_3_resultado_crudo.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[archivo] {out_json}", file=sys.stderr, flush=True)
    print(f"[elapsed total] {elapsed:.1f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
