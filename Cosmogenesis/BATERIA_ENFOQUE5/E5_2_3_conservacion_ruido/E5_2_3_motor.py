#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5_2_3_motor.py — Conservación bajo forzamiento estocástico (Enfoque 5, TEMA 2, exp 3)
========================================================================================
Pregunta (documento madre): con ruido dinámico fuerte, ¿sigue cuadrando la cuenta de
energía, o el ruido fabrica/destruye presupuesto en vez de sólo redistribuirlo (E2)?

Ver PROTOCOLO_E5.2-3_PREREGISTRO.md (escrito ANTES de este motor, T3) para la derivación
completa de las definiciones. Resumen:

  E_total(t) = (1/N)·Σ φ_i(t)²  =  1 + 2·D̄(t) + X(t)     [identidad algebraica exacta]
    D̄(t) = mean(φ(t)) − 1                 (desplazamiento del promedio vs equilibrio)
    X(t) = (1/N)·Σ (φ_i(t) − 1)²           (exergía — MISMA fórmula que E5_2_2, reutilizada
                                             verbatim para comparabilidad; E5.2-1 no tenía
                                             protocolo en disco al momento de escribir esto)
  deriva(t) = |E_total(t) − E_total(0)| / E_total(0)

Física base reutilizada de cs074_rcruz.py SIN editarlo (import): campo_inicial,
paso_difusion, paso_expansion, medir_D. Dos mecanismos de ruido dinámico añadidos aquí,
aplicados DESPUÉS de cada paso de difusión+expansión:

  (a) aditivo:      φ_i += amplitud·N(0,1) i.i.d.           (forzamiento externo genuino)
  (b) intercambio:  transferencia δ~N(0,amplitud²) entre vecinos por arista viva,
                     φ_i -= δ_i ; φ_{i+1} += δ_i             (conserva Σφ_i EXACTO por diseño)

Grid principal a H=0 (sin expansión, aísla la pregunta del ruido) + grid suplementario a
H=H(r=1) (expansión activa, chequeo de robustez). Ver protocolo para todos los parámetros.
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

import cs074_rcruz as base  # noqa: E402  (física reutilizada, NO editada)

UMBRAL_DERIVA = 1e-6
N_DEFAULT = 200
EPS_DEFAULT = 1e-3


# --------------------------------------------------------------------------------------
# E_total y componentes (definición propia E5.2-3, extiende X(t) de E5_2_2)
# --------------------------------------------------------------------------------------
def medir_E(phi):
    """E_total, D_bar, X — identidad algebraica exacta (1/N)Σφ² = 1 + 2·D_bar + X."""
    N = phi.size
    E_total = float(np.sum(phi * phi)) / N
    D_bar = float(phi.mean() - 1.0)
    X = float(np.sum((phi - 1.0) ** 2)) / N
    return E_total, D_bar, X


# --------------------------------------------------------------------------------------
# Mecanismos de ruido dinámico
# --------------------------------------------------------------------------------------
def ruido_aditivo(phi, amplitud, rng):
    if amplitud <= 0.0:
        return phi
    eta = rng.standard_normal(phi.size) * amplitud
    return phi + eta


def ruido_intercambio(phi, activo, amplitud, rng):
    """Transferencia conservativa (Sigma phi_i EXACTA) solo por aristas vivas."""
    if amplitud <= 0.0:
        return phi
    delta = rng.standard_normal(phi.size) * amplitud
    delta = np.where(activo, delta, 0.0)
    return phi - delta + np.roll(delta, 1)


# --------------------------------------------------------------------------------------
# Trayectoria con checkpoints (evita recorridas redundantes por cada 'pasos')
# --------------------------------------------------------------------------------------
def correr_trayectoria(N, eps, H, amplitud, mecanismo, pasos_max, checkpoints, seed):
    rng = np.random.default_rng(seed)
    phi, _ = base.campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)

    E0, D0, X0 = medir_E(phi)
    cps = set(checkpoints)
    registros = []
    for t in range(1, pasos_max + 1):
        phi = base.paso_difusion(phi, activo)
        activo = base.paso_expansion(activo, H, rng)
        if mecanismo == "aditivo":
            phi = ruido_aditivo(phi, amplitud, rng)
        elif mecanismo == "intercambio":
            phi = ruido_intercambio(phi, activo, amplitud, rng)
        elif mecanismo == "ninguno":
            pass
        else:
            raise ValueError(f"mecanismo desconocido: {mecanismo}")

        if t in cps:
            Et, Dt, Xt = medir_E(phi)
            deriva = abs(Et - E0) / abs(E0) if E0 != 0 else float("nan")
            registros.append(
                {
                    "t": t,
                    "E_total": Et,
                    "D_bar": Dt,
                    "X": Xt,
                    "deriva": deriva,
                    "frac_exp": float(1.0 - activo.mean()),
                }
            )
    return registros, E0, X0


def trayectoria_fina(N, eps, H, amplitud, mecanismo, pasos_max, seed):
    """Igual que arriba pero registra TODOS los pasos (testigo de localización)."""
    rng = np.random.default_rng(seed)
    phi, _ = base.campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    E0, _, _ = medir_E(phi)
    ts, derivas = [], []
    for t in range(1, pasos_max + 1):
        phi = base.paso_difusion(phi, activo)
        activo = base.paso_expansion(activo, H, rng)
        if mecanismo == "aditivo":
            phi = ruido_aditivo(phi, amplitud, rng)
        elif mecanismo == "intercambio":
            phi = ruido_intercambio(phi, activo, amplitud, rng)
        Et, _, _ = medir_E(phi)
        ts.append(t)
        derivas.append(abs(Et - E0) / abs(E0) if E0 != 0 else float("nan"))
    return {"t": ts, "deriva": derivas, "E0": E0}


# --------------------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------------------
CHECKPOINTS_MAIN = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
CHECKPOINTS_SUP = [10, 30, 100, 300, 1000, 3000, 10000]
SEMILLAS_MAIN = list(range(20000, 20012))  # 12 semillas
SEMILLAS_SUP = list(range(20000, 20012))


def amplitudes_principal():
    return [0.0] + list(np.logspace(-6, 0, 19))


def amplitudes_suplementario():
    return list(np.logspace(-6, 0, 7))


def correr_grid_principal(N=N_DEFAULT, eps=EPS_DEFAULT, pasos_max=20000, log=print):
    filas = []
    amps = amplitudes_principal()
    mecanismos = ["aditivo", "intercambio"]
    total = len(amps) * len(mecanismos) * len(SEMILLAS_MAIN)
    done = 0
    t0 = time.time()
    for mec in mecanismos:
        for amp in amps:
            for s in SEMILLAS_MAIN:
                regs, E0, X0 = correr_trayectoria(
                    N, eps, 0.0, amp, mec, pasos_max, CHECKPOINTS_MAIN, seed=s
                )
                filas.append(
                    {"mecanismo": mec, "amplitud": amp, "seed": s, "E0": E0, "X0": X0, "reg": regs}
                )
                done += 1
                if done % 40 == 0:
                    dt = time.time() - t0
                    log(f"[principal] {done}/{total} corridas, {dt:.1f}s", flush=True)
    return filas


def correr_grid_suplementario(N=N_DEFAULT, eps=EPS_DEFAULT, pasos_max=10000, log=print):
    D = float(np.mean([base.medir_D(N, eps, s) for s in range(4)]))
    H_r1 = float(min(1.0 * D, 1.0))
    filas = []
    amps = amplitudes_suplementario()
    total = len(amps) * len(SEMILLAS_SUP)
    done = 0
    t0 = time.time()
    for amp in amps:
        for s in SEMILLAS_SUP:
            regs, E0, X0 = correr_trayectoria(
                N, eps, H_r1, amp, "aditivo", pasos_max, CHECKPOINTS_SUP, seed=s + 500
            )
            filas.append(
                {"amplitud": amp, "seed": s + 500, "E0": E0, "X0": X0, "H": H_r1, "D": D, "reg": regs}
            )
            done += 1
            if done % 20 == 0:
                dt = time.time() - t0
                log(f"[suplementario] {done}/{total} corridas, {dt:.1f}s", flush=True)
    return filas, {"D": D, "H_r1": H_r1}


def correr_testigos(N=N_DEFAULT, eps=EPS_DEFAULT, pasos_max=20000, log=print):
    testigos = {}
    for amp in [1e-6, 1.0]:
        key = f"amp={amp:g}"
        testigos[key] = trayectoria_fina(N, eps, 0.0, amp, "aditivo", pasos_max, seed=99000)
        log(f"[testigo] {key} listo", flush=True)
    return testigos


def main():
    t0 = time.time()
    log_path = HERE / "E5_2_3_run.log"
    logf = open(log_path, "a", encoding="utf-8")

    def log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(msg, file=sys.stderr, **{k: v for k, v in kwargs.items() if k != "flush"})
        logf.write(msg + "\n")
        logf.flush()

    log(f"=== E5.2-3 motor iniciado {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")

    log("[fase 1/3] grid principal (H=0, aditivo + intercambio)...")
    grid_principal = correr_grid_principal(log=log)

    log("[fase 2/3] grid suplementario (H=H(r=1), aditivo)...")
    grid_sup, meta_sup = correr_grid_suplementario(log=log)

    log("[fase 3/3] testigos de trayectoria fina...")
    testigos = correr_testigos(log=log)

    elapsed = time.time() - t0
    result = {
        "experimento": "E5.2-3",
        "definicion_E_total": "(1/N) Sum phi_i^2 = 1 + 2*D_bar + X ; X=(1/N)Sum(phi_i-1)^2 reutilizada de E5_2_2",
        "umbral_deriva": UMBRAL_DERIVA,
        "N": N_DEFAULT,
        "eps": EPS_DEFAULT,
        "checkpoints_principal": CHECKPOINTS_MAIN,
        "checkpoints_suplementario": CHECKPOINTS_SUP,
        "semillas_principal": SEMILLAS_MAIN,
        "meta_suplementario": meta_sup,
        "grid_principal": grid_principal,
        "grid_suplementario": grid_sup,
        "testigos": testigos,
        "elapsed_s": elapsed,
    }
    out_json = HERE / "E5_2_3_resultado.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    log(f"[listo] {out_json} elapsed={elapsed:.1f}s")
    logf.close()


if __name__ == "__main__":
    main()
