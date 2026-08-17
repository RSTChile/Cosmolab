#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.1-1 — Supervivencia de exergía frente a la razón expansión/difusión, rango extremo
=======================================================================================

Experimento 1 de 30, Enfoque 5 (Energía · Exergía · Entropía), Tema 1
(Persistencia de la exergía). Ejecutado por el agente de archivos-prefijo E5_1_1_,
en paralelo con otros 29 agentes de la misma batería.

Pre-registro (leer ANTES que este archivo, describe el diseño completo, congelado
antes de escribir este motor):
    PROTOCOLO_E5.1-1_PREREGISTRO.md  (mismo directorio)

Documento madre (spec exacta de este experimento, sección "E5.1-1"):
    ../../BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md

Código base LEÍDO (comprendido, NO editado, NO importado — este motor es una
reimplementación propia bajo mi prefijo, fiel a la física del original):
    ../../cs074_rcruz.py

Pregunta: ¿sobrevive la exergía (capacidad de hacer trabajo) cuando el sistema se
expande, y a partir de qué r=H/D emerge esa supervivencia?

Modelo (idéntico en física a cs074_rcruz.py):
  - Campo φ en anillo de N sitios. Fondo=1 + ε·(5 armónicos, fase aleatoria).
  - Difusión SOLO por aristas vivas (relajación local hacia promedio de vecinos).
  - Expansión = corte Bernoulli de aristas vivas, probabilidad H por paso.
  - D = fracción de contraste borrada en un paso de difusión pura (H=0), MEDIDA.
  - r = H/D es el eje primario; H = min(r_target·D, 1.0).
  - Ruido dinámico (T7): en cada paso se suma ruido gaussiano de amplitud
    NOISE_REL·ε al campo (además de las 16 semillas independientes). Con ε=0 el
    ruido dinámico es exactamente 0.

Observable X_final (exergía) = c · v donde:
    c = autocorrelación a un paso (corr(φ, roll(φ,1)), clip a ≥0)
    v = Var(φ_final) / Var(φ_inicial)
Misma fórmula que `persistencia()` en cs074_rcruz.py (ahí se llama P). Se renombra a
X porque este experimento la interpreta como fracción de energía capaz de hacer
trabajo: la varianza SOLA sobrevive a una permutación (mismo conjunto de valores,
otro orden) — el factor c es lo que la permutación SÍ destruye, y es lo que separa
REAL de NULL.

NULL: permutar φ al final. Optimización de cómputo (documentada, no es cambio de
método): dado que con el mismo seed la trayectoria REAL y la trayectoria NULL de
cs074_rcruz.py son IDÉNTICAS hasta el paso final (misma rng, mismos draws, difieren
solo en la permutación última), aquí se corre la evolución UNA vez por semilla y
el NULL se deriva permutando el φ final con una permutación independiente — resultado
estadísticamente equivalente, mitad del cómputo.

Axiomas (declarados, no física real):
  E1 = conservación del presupuesto declarado (Σφ). Se AUDITA (inicio→fin), no se
       fuerza. Reportado por fila (deriva relativa).
  E2 = la expansión redistribuye E latente en exergía (marco interpretativo de por
       qué r alto debería preservar X); no se fuerza en el motor.

Barrido (sobredimensionado):
  r  = {0} ∪ logspace(1e-3, 1e3, 25)   → 26 puntos (6 décadas + control r=0)
  eps = {0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.3, 1.0}  → 9 puntos (12 décadas + 0)
  semillas = 16 (0..15), cada una con rng propio (np.random.default_rng)
  N = 200, pasos = calibrado UNA vez (lavado a P<0.05 en eps=1e-3, H=0, ×1.15 margen)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent

# ---- Constantes de diseño, congeladas en el pre-registro (T1: nunca ajustadas) ----
N = 200
SEMILLAS = 16
NOISE_REL = 0.02          # amplitud de ruido dinámico por paso, relativa a eps
P_LAVADO = 0.05            # umbral de "lavado" para calibrar pasos
MARGEN_LAVADO = 1.15
EPS_REF_CALIBRACION = 1e-3  # eps de referencia para calibrar pasos (linealidad → generaliza a todo eps>0)
MAX_STEPS_CALIBRACION = 200_000
CHECK_EVERY = 50

EPS_LIST = [0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.3, 1.0]
R_TARGETS = [0.0] + list(np.logspace(-3, 3, 25))


# ---------------------------------------------------------------------------
# Física (fiel a cs074_rcruz.py, reimplementada bajo este prefijo)
# ---------------------------------------------------------------------------

def campo_inicial(N, eps, rng):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    fondo = np.ones(N, dtype=float)
    if eps <= 0.0:
        return fondo.copy(), x
    pert = np.zeros(N, dtype=float)
    for m in range(1, 6):
        fase = rng.uniform(0, 2 * np.pi)
        pert += np.sin(2 * np.pi * m * x + fase) / m
    pert -= pert.mean()
    if pert.std() > 0:
        pert = pert / pert.std()
    return fondo + eps * pert, x


def paso_difusion(phi, activo):
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


def medir_D(N, eps, seed):
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    c0 = phi.std()
    if c0 <= 0:
        return 0.0
    phi1 = paso_difusion(phi, activo)
    c1 = phi1.std()
    return max(0.0, float((c0 - c1) / c0))


def exergia(phi, var0):
    """X_final = c * v  (autocorrelación a un paso * fracción de varianza retenida)."""
    if var0 <= 0 or phi.std() <= 1e-14:
        return 0.0
    c = np.corrcoef(phi, np.roll(phi, 1))[0, 1]
    if not np.isfinite(c):
        c = 0.0
    c = max(0.0, float(c))
    v = float(phi.var() / var0)
    return float(c * v)


def medir_pasos_lavado(N, eps, semillas, P_thr=P_LAVADO, max_steps=MAX_STEPS_CALIBRACION,
                        check_every=CHECK_EVERY):
    """Tiempo medido (pasos) a H=0 para que X < P_thr. Igual método que cs074_rcruz.py."""
    tiempos = []
    for s in range(semillas):
        rng = np.random.default_rng(90_000 + s)
        phi, _ = campo_inicial(N, eps, rng)
        activo = np.ones(N, dtype=bool)
        var0 = float(phi.var())
        if var0 <= 0:
            tiempos.append(0)
            continue
        t_hit = None
        for t in range(1, max_steps + 1):
            phi = paso_difusion(phi, activo)
            if t % check_every == 0:
                if exergia(phi, var0) < P_thr:
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


def evolucionar_con_ruido(phi, activo, H, eps, pasos, rng):
    """
    Evolución REAL con ruido dinámico (T7): en cada paso, además de difusión+expansión,
    se suma ruido gaussiano de amplitud NOISE_REL*eps (0 si eps=0 — preserva el control
    puro). Devuelve trayectoria final (NULL se deriva afuera, permutando).
    """
    var0 = float(phi.var())
    e_decl_0 = float(phi.sum())
    noise_amp = NOISE_REL * eps
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        if noise_amp > 0:
            phi = phi + noise_amp * rng.standard_normal(phi.shape)
        activo = paso_expansion(activo, H, rng)
    e_decl_1 = float(phi.sum())
    deriva_E = abs(e_decl_1 - e_decl_0) / (abs(e_decl_0) + 1e-300)
    return phi, activo, var0, deriva_E


def corrida_celda(N, eps, H, pasos, seed):
    """Una semilla: evoluciona REAL, deriva NULL permutando el final."""
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    phi_f, activo_f, var0, deriva_E = evolucionar_con_ruido(phi, activo, H, eps, pasos, rng)

    X_real = exergia(phi_f, var0)
    std_ratio_real = float(phi_f.std() / np.sqrt(var0)) if var0 > 0 else 0.0

    # NULL: permutación independiente del campo final (misma física hasta este punto,
    # difiere solo en el orden espacial — igual que evolucionar(...,null=True) de la base)
    phi_null = rng.permutation(phi_f)
    X_null = exergia(phi_null, var0)
    std_ratio_null = float(phi_null.std() / np.sqrt(var0)) if var0 > 0 else 0.0

    frac_exp = 1.0 - float(activo_f.mean())

    return {
        "X_real": X_real,
        "X_null": X_null,
        "std_ratio_real": std_ratio_real,
        "std_ratio_null": std_ratio_null,
        "deriva_E": deriva_E,
        "frac_exp": frac_exp,
    }


def barrido(N, eps_list, r_targets, semillas, pasos_fijo):
    filas = []
    meta_por_eps = []
    for eps in eps_list:
        D = float(np.mean([medir_D(N, eps, 70_000 + s) for s in range(semillas)]))
        meta_por_eps.append({"eps": eps, "D": D, "pasos": pasos_fijo})

        for r_tgt in r_targets:
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
                r_eff = H / D
            else:
                H = 0.0 if r_tgt == 0 else 1.0
                r_eff = 0.0 if r_tgt == 0 else float("inf")

            Xr, Xn, srr, srn, deriva, fracs = [], [], [], [], [], []
            for s in range(semillas):
                seed = 1_000_000 + int(round(r_tgt * 1000)) * 100 + s + hash((eps,)) % 97
                # seed determinista y reproducible por (r,eps,seed) sin colisiones triviales
                seed = abs(seed) % (2**32 - 1)
                res = corrida_celda(N, eps, H, pasos_fijo, seed=seed)
                Xr.append(res["X_real"])
                Xn.append(res["X_null"])
                srr.append(res["std_ratio_real"])
                srn.append(res["std_ratio_null"])
                deriva.append(res["deriva_E"])
                fracs.append(res["frac_exp"])

            Xr = np.array(Xr)
            Xn = np.array(Xn)
            sd = np.sqrt((Xr.var() + Xn.var()) / 2.0)
            sd = max(sd, 1e-9)
            z = float((Xr.mean() - Xn.mean()) / sd)

            filas.append({
                "eps": eps,
                "r_target": float(r_tgt),
                "H": H,
                "D": D,
                "r_eff": r_eff,
                "pasos": pasos_fijo,
                "X_real_mean": float(Xr.mean()),
                "X_real_std": float(Xr.std()),
                "X_real_per_seed": [float(v) for v in Xr],
                "X_null_mean": float(Xn.mean()),
                "X_null_std": float(Xn.std()),
                "X_null_per_seed": [float(v) for v in Xn],
                "z": z,
                "std_ratio_real_mean": float(np.mean(srr)),
                "std_ratio_null_mean": float(np.mean(srn)),
                "deriva_E_max": float(np.max(deriva)),
                "deriva_E_mean": float(np.mean(deriva)),
                "frac_exp_mean": float(np.mean(fracs)),
            })
    return filas, meta_por_eps


def control_r0_ok(filas, X_max=0.15):
    rows = [f for f in filas if f["r_target"] == 0.0 and f["eps"] > 0]
    if not rows:
        return False, {}
    mean_X = float(np.mean([f["X_real_mean"] for f in rows]))
    return mean_X < X_max, {"mean_X_r0_eps_gt0": mean_X, "n": len(rows), "X_max": X_max}


def control_eps0_ok(filas, X_max=1e-9):
    rows = [f for f in filas if f["eps"] == 0.0]
    if not rows:
        return False, {}
    mean_X = float(np.mean([f["X_real_mean"] for f in rows]))
    max_X = float(np.max([f["X_real_mean"] for f in rows]))
    return max_X < X_max, {"mean_X_eps0": mean_X, "max_X_eps0": max_X, "n": len(rows)}


def main():
    t0 = time.time()
    print(f"[E5.1-1] inicio {time.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr, flush=True)

    cal = medir_pasos_lavado(N, EPS_REF_CALIBRACION, SEMILLAS)
    pasos_fijo = cal["pasos"]
    print(f"[calibracion] N={N} eps_ref={EPS_REF_CALIBRACION} mediana_lavado={cal['mediana']} "
          f"pasos={pasos_fijo} lavo_todas={cal['lavo_todas']}", file=sys.stderr, flush=True)
    print(f"[grid] r_targets={len(R_TARGETS)} eps={len(EPS_LIST)} semillas={SEMILLAS} "
          f"celdas={len(R_TARGETS)*len(EPS_LIST)} corridas={len(R_TARGETS)*len(EPS_LIST)*SEMILLAS}",
          file=sys.stderr, flush=True)

    filas, meta = barrido(N, EPS_LIST, R_TARGETS, SEMILLAS, pasos_fijo)

    ok_r0, ctrl_r0 = control_r0_ok(filas)
    ok_eps0, ctrl_eps0 = control_eps0_ok(filas)

    elapsed = time.time() - t0

    result = {
        "experimento": "E5.1-1",
        "titulo": "Supervivencia de exergia frente a la razon expansion/difusion, rango extremo",
        "timestamp_inicio": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "N": N,
        "semillas": SEMILLAS,
        "noise_rel": NOISE_REL,
        "r_targets": [float(r) for r in R_TARGETS],
        "eps_list": EPS_LIST,
        "pasos_fijo": pasos_fijo,
        "calibracion_lavado": cal,
        "meta_por_eps": meta,
        "control_r0_lava": ok_r0,
        "control_r0_detail": ctrl_r0,
        "control_eps0_nulo": ok_eps0,
        "control_eps0_detail": ctrl_eps0,
        "filas": filas,
        "elapsed_s": elapsed,
        "pre_inscrito": {
            "eps0": "X_final=0 a todo r",
            "r0_eps_gt0": "X_final bajo (difusion lava), control de validez",
            "r_ll_1": "X_final bajo",
            "r_approx_1": "zona de transicion",
            "r_gg_1": "X_final alto, separado de NULL",
            "null": "debe caer cerca de 0 en todo el rango (T4)",
        },
    }

    out_json = OUT / "E5_1_1_resultado_crudo.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[archivo] {out_json}", file=sys.stderr, flush=True)
    print(f"[control_r0_lava] {ok_r0} {ctrl_r0}", file=sys.stderr, flush=True)
    print(f"[control_eps0_nulo] {ok_eps0} {ctrl_eps0}", file=sys.stderr, flush=True)
    print(f"[elapsed] {elapsed:.1f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
