#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.5-2 — Tiempo a la muerte termica: cuanto tarda X->0 segun eps y r
=====================================================================

Experimento 20 de 30, Enfoque 5 (Energia . Exergia . Entropia), Tema 5
(Muerte termica vs Nada). Ejecutado por el agente de archivos-prefijo E5_5_2_,
en paralelo con otros 29 agentes de la misma bateria.

Pre-registro (leer ANTES que este archivo, describe el diseno completo, congelado
antes de escribir este motor):
    PROTOCOLO_E5.5-2_PREREGISTRO.md  (mismo directorio)

Documento madre (spec exacta de este experimento, seccion "E5.5-2"):
    ../../BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md

Codigo base LEIDO (comprendido, NO editado, NO importado -- este motor es una
reimplementacion propia bajo mi prefijo, fiel a la fisica del original, mismo
metodo que uso E5.1-1):
    ../../cs074_rcruz.py

Protocolo hermano leido en disco para heredar la definicion de X (E5.5-1 NO
estaba en disco al momento del pre-registro, verificado explicitamente):
    ../E5_1_1_supervivencia_exergia/PROTOCOLO_E5.1-1_PREREGISTRO.md
    ../E5_1_1_supervivencia_exergia/E5_1_1_engine.py

Pregunta: cuantos pasos tarda X en caer bajo el umbral de muerte termica
(X_UMBRAL=0.05, heredado de P_LAVADO de la base), en funcion de eps y r=H/D,
en la zona donde r NO alcanza a congelar (r in [1e-3 .. 1], cruzando el umbral
de congelamiento ya conocido ~0.1).

Modelo (identico en fisica a cs074_rcruz.py y a E5.1-1):
  - Campo phi en anillo de N sitios. Fondo=1 + eps*(5 armonicos, fase aleatoria).
  - Difusion SOLO por aristas vivas (relajacion local hacia promedio de vecinos).
  - Expansion = corte Bernoulli IRREVERSIBLE de aristas vivas, probabilidad H por paso.
  - D = fraccion de contraste borrada en un paso de difusion pura (H=0), MEDIDA
    UNA VEZ (no depende de eps por linealidad) y reusada en todo el barrido.
  - r = H/D es el eje del barrido en la zona sub-congelamiento; H = min(r*D, 1.0).
  - Ruido dinamico (T7): en cada paso se suma ruido gaussiano de amplitud
    NOISE_REL*eps al campo (ademas de las 12 semillas independientes). Con
    eps=0 el ruido dinamico es exactamente 0.

Observable X (exergia) = c * v donde:
    c = autocorrelacion a un paso (corr(phi, roll(phi,1)), clip a >=0)
    v = Var(phi_actual) / Var(phi_inicial)
Identica formula a `persistencia()` en cs074_rcruz.py y a `exergia()` en
E5_1_1_engine.py. Evaluada EN EL TIEMPO (no solo al final): t_muerte(eps,r,semilla)
es el primer paso (multiplo de CHECK_EVERY=50) en que X(t) < X_UMBRAL=0.05.
Si no cruza dentro de MAX_STEPS, la corrida se marca CENSURADA (no convergio).

NULL: ninguno (declarado explicitamente por el documento madre para E5.5-2: "NULL: --").

Axiomas (declarados, no fisica real):
  E1 = conservacion del presupuesto declarado (Sum phi). Se AUDITA (inicio -> instante
       de muerte/tope), no se fuerza. Reportado por fila (deriva relativa).
  E2 = la expansion redistribuye E latente en exergia (marco interpretativo de por
       que r creciente deberia alargar/impedir la muerte termica); no se fuerza.

Barrido (sobredimensionado en eps por regla del director; r acotado a la zona
sub-congelamiento pre-registrada por el documento madre para E5.5-2):
  eps = {0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0}  -> 11 puntos
  r   = {0} U logspace(1e-3, 1, 13)                                    -> 14 puntos
  semillas = 12 (0..11), evolucionadas en batch (12,N) por celda (eps,r)
  N = 200, MAX_STEPS calibrado por medicion (mediana_lavado_r0_ref * CAP_MULT=10)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent

# ---- Constantes de diseno, congeladas en el pre-registro (T1: nunca ajustadas) ----
N = 200
SEMILLAS = 12
NOISE_REL = 0.02             # amplitud de ruido dinamico por paso, relativa a eps (T7)
X_UMBRAL = 0.05              # heredado literal de P_LAVADO en cs074_rcruz.py y E5.1-1
CHECK_EVERY = 50             # misma cadencia que medir_pasos_lavado de la base
CAP_MULT = 10                # margen de MAX_STEPS sobre el lavado puro medido (r=0)
D_REF_EPS = 1e-3             # eps de referencia para medir D (no depende de eps: linealidad)
D_REF_SEMILLAS = 20
LAVADO_REF_EPS = 1e-3        # eps de referencia para calibrar MAX_STEPS (r=0)
LAVADO_REF_SEMILLAS = 16
MAX_STEPS_SAFETY = 400_000   # tope absoluto de seguridad para la propia calibracion

EPS_LIST = [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
R_LIST = [0.0] + [float(v) for v in np.logspace(-3, 0, 13)]

CONOCIDO_UMBRAL_CONGELAMIENTO_R = 0.1  # de cs074_rcruz_produccion_resultado.json


# ---------------------------------------------------------------------------
# Fisica (fiel a cs074_rcruz.py, reimplementada bajo este prefijo, batched)
# ---------------------------------------------------------------------------

def campo_inicial_1d(N, eps, rng):
    """Version escalar (1 semilla), usada solo para medir D y para la calibracion
    de lavado a r=0 (identica formula a campo_inicial de la base)."""
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


def paso_difusion_1d(phi, activo):
    left = np.roll(phi, 1)
    right = np.roll(phi, -1)
    e_left = np.roll(activo, 1)
    e_right = activo
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def exergia_1d(phi, var0):
    if var0 <= 0 or phi.std() <= 1e-14:
        return 0.0
    c = np.corrcoef(phi, np.roll(phi, 1))[0, 1]
    if not np.isfinite(c):
        c = 0.0
    c = max(0.0, float(c))
    v = float(phi.var() / var0)
    return float(c * v)


def medir_D(N, eps, semillas):
    """D = fraccion de contraste borrada en UN paso de difusion pura (H=0).
    Medida UNA vez (no depende de eps por linealidad de la difusion), promediada
    sobre `semillas` corridas independientes."""
    vals = []
    for s in range(semillas):
        rng = np.random.default_rng(555_000 + s)
        phi, _ = campo_inicial_1d(N, eps, rng)
        activo = np.ones(N, dtype=bool)
        c0 = phi.std()
        if c0 <= 0:
            vals.append(0.0)
            continue
        phi1 = paso_difusion_1d(phi, activo)
        c1 = phi1.std()
        vals.append(max(0.0, float((c0 - c1) / c0)))
    return float(np.mean(vals))


def medir_lavado_r0(N, eps, semillas, X_thr=X_UMBRAL, max_steps=MAX_STEPS_SAFETY,
                     check_every=CHECK_EVERY):
    """Tiempo medido (pasos) a H=0 (r=0 puro) para que X < X_thr. Mismo metodo que
    medir_pasos_lavado en cs074_rcruz.py y E5.1-1 -- usado SOLO para calibrar
    MAX_STEPS del barrido principal (T0/T1: no puesto a mano)."""
    tiempos = []
    for s in range(semillas):
        rng = np.random.default_rng(777_000 + s)
        phi, _ = campo_inicial_1d(N, eps, rng)
        activo = np.ones(N, dtype=bool)
        var0 = float(phi.var())
        if var0 <= 0:
            tiempos.append(0)
            continue
        t_hit = None
        for t in range(1, max_steps + 1):
            phi = paso_difusion_1d(phi, activo)
            if t % check_every == 0:
                if exergia_1d(phi, var0) < X_thr:
                    t_hit = t
                    break
        if t_hit is None:
            t_hit = max_steps
        tiempos.append(t_hit)
    med = int(np.median(tiempos))
    return {"tiempos": tiempos, "mediana": med, "X_thr": X_thr,
            "lavo_todas": all(t < max_steps for t in tiempos)}


# ---------------------------------------------------------------------------
# Motor batched (S semillas evolucionadas juntas como array (S,N)) -- optimizacion
# de computo declarada en el pre-registro Sec.5: aritmeticamente identica a S
# corridas 1D separadas (misma formula, draws independientes por fila), solo
# evita el overhead de bucle Python por semilla.
# ---------------------------------------------------------------------------

def campo_inicial_batch(N, S, eps, rng):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    fondo = np.ones((S, N), dtype=float)
    if eps <= 0.0:
        return fondo.copy()
    pert = np.zeros((S, N), dtype=float)
    for m in range(1, 6):
        fase = rng.uniform(0, 2 * np.pi, size=(S, 1))
        pert += np.sin(2 * np.pi * m * x[None, :] + fase) / m
    pert -= pert.mean(axis=1, keepdims=True)
    std = pert.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    pert = pert / std
    return fondo + eps * pert


def paso_difusion_batch(phi, activo):
    left = np.roll(phi, 1, axis=1)
    right = np.roll(phi, -1, axis=1)
    e_left = np.roll(activo, 1, axis=1)
    e_right = activo
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def paso_expansion_batch(activo, H, rng):
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


def exergia_batch(phi, var0):
    """X por fila (semilla). var0: array (S,) o escalar."""
    S = phi.shape[0]
    out = np.zeros(S, dtype=float)
    phistd = phi.std(axis=1)
    v0 = np.broadcast_to(var0, (S,))
    for i in range(S):
        if v0[i] <= 0 or phistd[i] <= 1e-14:
            out[i] = 0.0
            continue
        c = np.corrcoef(phi[i], np.roll(phi[i], 1))[0, 1]
        if not np.isfinite(c):
            c = 0.0
        c = max(0.0, float(c))
        v = float(phi[i].var() / v0[i])
        out[i] = c * v
    return out


def corrida_celda(N, S, eps, H, max_steps, check_every, rng_seed):
    """Evoluciona S semillas en batch hasta que TODAS crucen X<X_UMBRAL o se
    alcance max_steps. Devuelve t_muerte por semilla (None=censurada), estado
    final para auditoria (std_ratio, deriva E1, frac_exp)."""
    rng = np.random.default_rng(rng_seed)
    phi = campo_inicial_batch(N, S, eps, rng)
    activo = np.ones((S, N), dtype=bool)
    var0 = phi.var(axis=1)  # (S,)
    e_decl_0 = phi.sum(axis=1).copy()  # (S,)
    noise_amp = NOISE_REL * eps

    t_muerte = np.full(S, -1, dtype=np.int64)  # -1 = aun no muere
    if eps <= 0.0:
        # Trivial: contraste0=0 en todas las filas -> X=0 desde t=0 (T0: caso
        # analitico de la base, no se simula).
        t_muerte[:] = 0

    t = 0
    while t < max_steps and np.any(t_muerte < 0):
        phi = paso_difusion_batch(phi, activo)
        if noise_amp > 0:
            phi = phi + noise_amp * rng.standard_normal(phi.shape)
        activo = paso_expansion_batch(activo, H, rng)
        t += 1
        if t % check_every == 0:
            X_now = exergia_batch(phi, var0)
            recien_muertas = (t_muerte < 0) & (X_now < X_UMBRAL)
            t_muerte[recien_muertas] = t

    censuradas = (t_muerte < 0)
    t_muerte_reportado = np.where(censuradas, max_steps, t_muerte)

    e_decl_1 = phi.sum(axis=1)
    deriva_E = np.abs(e_decl_1 - e_decl_0) / (np.abs(e_decl_0) + 1e-300)
    std_ratio_final = np.divide(
        phi.std(axis=1), np.sqrt(np.where(var0 > 0, var0, 1.0)),
        out=np.zeros(S), where=var0 > 0,
    )
    frac_exp = 1.0 - activo.mean(axis=1)

    return {
        "t_muerte_per_seed": [int(v) for v in t_muerte_reportado],
        "censurada_per_seed": [bool(v) for v in censuradas],
        "n_censuradas": int(censuradas.sum()),
        "pasos_corridos": int(t),
        "std_ratio_final_mean": float(std_ratio_final.mean()),
        "deriva_E_mean": float(deriva_E.mean()),
        "deriva_E_max": float(deriva_E.max()),
        "frac_exp_mean": float(frac_exp.mean()),
    }


def barrido(N, eps_list, r_list, semillas, D, max_steps, check_every):
    filas = []
    for eps in eps_list:
        for r_tgt in r_list:
            H = float(min(r_tgt * D, 1.0)) if D > 0 else (0.0 if r_tgt == 0 else 1.0)
            r_eff = (H / D) if D > 0 else (0.0 if r_tgt == 0 else float("inf"))

            seed = 2_000_000 + int(round(r_tgt * 100000)) + int(round(abs(np.log10(max(eps, 1e-300))) * 1000)) if eps > 0 else 2_000_000 + int(round(r_tgt * 100000))
            seed = abs(seed) % (2**32 - 1)

            res = corrida_celda(N, semillas, eps, H, max_steps, check_every, rng_seed=seed)

            t_arr = np.array(res["t_muerte_per_seed"], dtype=float)
            filas.append({
                "eps": eps,
                "r_target": float(r_tgt),
                "H": H,
                "D": D,
                "r_eff": r_eff,
                "max_steps": max_steps,
                "t_muerte_mediana": float(np.median(t_arr)),
                "t_muerte_media": float(np.mean(t_arr)),
                "t_muerte_std": float(np.std(t_arr)),
                "t_muerte_min": float(np.min(t_arr)),
                "t_muerte_max": float(np.max(t_arr)),
                "t_muerte_per_seed": res["t_muerte_per_seed"],
                "censurada_per_seed": res["censurada_per_seed"],
                "n_censuradas": res["n_censuradas"],
                "frac_censurada": res["n_censuradas"] / semillas,
                "pasos_corridos": res["pasos_corridos"],
                "std_ratio_final_mean": res["std_ratio_final_mean"],
                "deriva_E_mean": res["deriva_E_mean"],
                "deriva_E_max": res["deriva_E_max"],
                "frac_exp_mean": res["frac_exp_mean"],
            })
            print(
                f"[celda] eps={eps:g} r_target={r_tgt:.4g} H={H:.6g} "
                f"t_muerte_med={np.median(t_arr):.0f} frac_censurada={res['n_censuradas']}/{semillas} "
                f"pasos_corridos={res['pasos_corridos']}",
                file=sys.stderr, flush=True,
            )
    return filas


def control_eps0_ok(filas):
    rows = [f for f in filas if f["eps"] == 0.0]
    if not rows:
        return False, {}
    ok = all(f["t_muerte_mediana"] == 0.0 for f in rows)
    return ok, {"n": len(rows), "max_t_muerte_mediana": max(f["t_muerte_mediana"] for f in rows)}


def control_r0_ok(filas, max_steps_ref):
    rows = [f for f in filas if f["r_target"] == 0.0 and f["eps"] > 0]
    if not rows:
        return False, {}
    fracs = [f["frac_censurada"] for f in rows]
    mean_frac = float(np.mean(fracs))
    return mean_frac < 0.5, {"mean_frac_censurada_r0": mean_frac, "n": len(rows)}


def divergencia_en_umbral_conocido(filas, r_umbral=CONOCIDO_UMBRAL_CONGELAMIENTO_R):
    """Compara fraccion censurada justo debajo vs justo encima (o igual) del
    umbral de congelamiento ya conocido (~0.1), agregando sobre eps>0."""
    below = [f for f in filas if f["eps"] > 0 and 0 < f["r_target"] < r_umbral]
    at_above = [f for f in filas if f["eps"] > 0 and f["r_target"] >= r_umbral]
    fb = float(np.mean([f["frac_censurada"] for f in below])) if below else None
    fa = float(np.mean([f["frac_censurada"] for f in at_above])) if at_above else None
    return {
        "frac_censurada_media_r_menor_umbral": fb,
        "frac_censurada_media_r_mayor_igual_umbral": fa,
        "n_below": len(below),
        "n_at_above": len(at_above),
        "diverge_como_se_predijo": (fb is not None and fa is not None and fa > fb),
    }


def main():
    t0 = time.time()
    ts_inicio = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[E5.5-2] inicio {time.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr, flush=True)

    D = medir_D(N, D_REF_EPS, D_REF_SEMILLAS)
    print(f"[calibracion D] eps_ref={D_REF_EPS} semillas={D_REF_SEMILLAS} D={D:.8f}",
          file=sys.stderr, flush=True)

    cal_lavado = medir_lavado_r0(N, LAVADO_REF_EPS, LAVADO_REF_SEMILLAS)
    max_steps = int(np.ceil(cal_lavado["mediana"] * CAP_MULT))
    print(
        f"[calibracion MAX_STEPS] eps_ref={LAVADO_REF_EPS} semillas={LAVADO_REF_SEMILLAS} "
        f"mediana_lavado_r0={cal_lavado['mediana']} CAP_MULT={CAP_MULT} MAX_STEPS={max_steps} "
        f"lavo_todas={cal_lavado['lavo_todas']} tiempos={cal_lavado['tiempos']}",
        file=sys.stderr, flush=True,
    )

    n_celdas = len(EPS_LIST) * len(R_LIST)
    n_celdas_simuladas = (len(EPS_LIST) - 1) * len(R_LIST)  # eps=0 es analitico
    print(
        f"[grid] eps={len(EPS_LIST)} r={len(R_LIST)} semillas={SEMILLAS} "
        f"celdas={n_celdas} celdas_simuladas={n_celdas_simuladas} "
        f"corridas_evolucion={n_celdas_simuladas * SEMILLAS}",
        file=sys.stderr, flush=True,
    )

    filas = barrido(N, EPS_LIST, R_LIST, SEMILLAS, D, max_steps, CHECK_EVERY)

    ok_eps0, ctrl_eps0 = control_eps0_ok(filas)
    ok_r0, ctrl_r0 = control_r0_ok(filas, max_steps)
    diverg = divergencia_en_umbral_conocido(filas)

    elapsed = time.time() - t0
    ts_fin = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    result = {
        "experimento": "E5.5-2",
        "titulo": "Tiempo a la muerte termica: cuanto tarda X->0 segun eps y r",
        "timestamp_inicio": ts_inicio,
        "timestamp_fin": ts_fin,
        "N": N,
        "semillas": SEMILLAS,
        "noise_rel": NOISE_REL,
        "X_UMBRAL": X_UMBRAL,
        "check_every": CHECK_EVERY,
        "cap_mult": CAP_MULT,
        "D_medido": D,
        "D_ref_eps": D_REF_EPS,
        "D_ref_semillas": D_REF_SEMILLAS,
        "calibracion_lavado_r0": cal_lavado,
        "max_steps": max_steps,
        "eps_list": EPS_LIST,
        "r_list": R_LIST,
        "umbral_congelamiento_conocido_r": CONOCIDO_UMBRAL_CONGELAMIENTO_R,
        "control_eps0_trivial": ok_eps0,
        "control_eps0_detail": ctrl_eps0,
        "control_r0_lava_mayoria": ok_r0,
        "control_r0_detail": ctrl_r0,
        "divergencia_en_umbral_conocido": diverg,
        "filas": filas,
        "elapsed_s": elapsed,
        "pre_inscrito": {
            "eps0": "t_muerte=0 a todo r (trivial, sin estructura inicial)",
            "r0": "t_muerte finito, del orden de la calibracion de lavado",
            "r_menor_umbral": "t_muerte finito, similar al de r=0",
            "pass_central": "frac_censurada debe subir al cruzar r~0.1 (congelamiento conocido)",
        },
    }

    out_json = OUT / "E5_5_2_resultado_crudo.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[archivo] {out_json}", file=sys.stderr, flush=True)
    print(f"[control_eps0_trivial] {ok_eps0} {ctrl_eps0}", file=sys.stderr, flush=True)
    print(f"[control_r0_lava_mayoria] {ok_r0} {ctrl_r0}", file=sys.stderr, flush=True)
    print(f"[divergencia_en_umbral_conocido] {diverg}", file=sys.stderr, flush=True)
    print(f"[elapsed] {elapsed:.1f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
