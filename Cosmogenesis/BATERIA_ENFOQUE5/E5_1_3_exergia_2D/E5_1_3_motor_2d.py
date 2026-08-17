#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.1-3 — Exergia persistente en 2D: es artefacto del anillo 1D?
=================================================================

Ejecutor: CC. Diseno: CS (BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md,
Tema 1, E5.1-3). Pre-registro: PROTOCOLO_E5.1-3_PREREGISTRO.md (leer antes que este
archivo -- este motor implementa exactamente lo pre-registrado ahi, nada mas, nada
ajustado despues de ver resultados).

Sustrato fisico: el mismo motor de cs074_rcruz.py (campo continuo + mancha eps,
difusion SOLO por aristas vivas, expansion que corta aristas, NULL = permuta phi al
final) adaptado a malla 2D toroidal. Estilo de adaptacion np.roll en 2 ejes tomado
como referencia de suite_epocas_masa_v6_mass_linaje.py (solo el patron de aristas
horizontales/verticales -- NO se copia fisica de masa/linaje de ese archivo). Tambien
se leyo (solo lectura) F1_6_motor_2d.py -- un motor 2D previo del mismo sustrato para
OTRO observable (persistencia de forma, Enfoque F1) -- como referencia de estilo de
la adaptacion 2D; NO se copia su fisica de masa (no tiene) ni se edita ese archivo.

Ninguno de cs074_rcruz.py, F1_6_motor_2d.py, suite_epocas_masa_v6_mass_linaje.py se
edita. Este es un archivo nuevo, prefijo E5_1_3_, en su propia carpeta E5_1_3_exergia_2D/.

Observable central -- EXERGIA X_final (fraccion de E capaz de hacer trabajo, PROTOCOLO
Sec.3):
  X_final = rho * v
  rho = max(0, promedio isotropico de la autocorrelacion a primer vecino en los 2 ejes)
        (2D) o al unico eje (1D, control interno)
  v   = Var(phi_final) / Var(phi_inicial)
Segundo observable (diagnostico, NO discrimina NULL por construccion): X_var = v sola.

NULL: permutacion espacial de phi al final (aplanar -> permutar -> reformar). Destruye
rho (correlacion), conserva v (varianza) -- por eso X_final,NULL ~ 0 pero X_var,NULL ~
X_var,REAL: la disociacion ES la verificacion de que el observable mide orden, no solo
magnitud (PROTOCOLO Sec.3).

Ademas del motor 2D (entregable central), este archivo incluye un CONTROL INTERNO 1D
(mismo sustrato, misma formula de X_final, un solo eje) para verificacion cruzada
1D<->2D propia y autocontenida, dado que al momento de escribir este motor el
directorio de E5.1-1 (otro agente, mismo dia) esta vacio -- no hay resultados 1D
publicados con los que comparar todavia (ver PROTOCOLO Sec.1). La comparacion final
contra los numeros reales de E5.1-1 queda pendiente para CS/auditoria posterior.

D y pasos_lavado: MEDIDOS del propio campo, no impuestos (T1). r = H/D barre el grid
pre-registrado que cruza r=1 (identico a R_TARGETS/EPS_LIST de cs074_rcruz.py /
F1_6_motor_2d.py -- eleccion deliberada de comparabilidad directa, PROTOCOLO Sec.5).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent

# Umbral de "lavado" para calibrar pasos (observable del propio campo) -- misma
# convencion que cs074_rcruz.py / F1_6_motor_2d.py, aplicada aqui a X_final en vez de P.
EXERGIA_LAVADO = 0.05
MARGEN_LAVADO = 1.15

# Grid pre-registrado (PROTOCOLO Sec.5) -- identico a R_TARGETS/EPS_LIST de
# cs074_rcruz.py y F1_6_motor_2d.py, para comparabilidad directa 1D<->2D.
R_TARGETS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
EPS_LIST = [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0]


# ============================================================================
# 2D toroidal (entregable central)
# ============================================================================

def campo_inicial_2d(L, eps, rng):
    """Fondo uniforme + perturbacion multi-modo 2D (5 modos, numero de onda entero
    aleatorio (kx,ky) en {1,2,3}^2, fase aleatoria), normalizada a std=1 antes de
    escalar por eps. Generalizacion de campo_inicial() 1D de cs074_rcruz.py."""
    x = np.linspace(0.0, 1.0, L, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    fondo = np.ones((L, L), dtype=float)
    if eps <= 0.0:
        return fondo
    pert = np.zeros((L, L), dtype=float)
    for m in range(1, 6):
        kx = int(rng.integers(1, 4))
        ky = int(rng.integers(1, 4))
        fase = rng.uniform(0, 2 * np.pi)
        pert += np.sin(2 * np.pi * (kx * X + ky * Y) + fase) / m
    pert -= pert.mean()
    if pert.std() > 0:
        pert = pert / pert.std()
    return fondo + eps * pert


def paso_difusion_2d(phi, ar, ad):
    """Difusion solo por aristas vivas, malla 4-conexa toroidal. ar[i,j] = arista
    horizontal (i,j)--(i,j+1 mod L); ad[i,j] = arista vertical (i,j)--(i+1 mod L,j).
    Vectorizada; generaliza paso_difusion() 1D de cs074_rcruz.py a 2 ejes."""
    right_active = ar
    left_active = np.roll(ar, 1, axis=1)
    down_active = ad
    up_active = np.roll(ad, 1, axis=0)

    right_val = np.roll(phi, -1, axis=1)
    left_val = np.roll(phi, 1, axis=1)
    down_val = np.roll(phi, -1, axis=0)
    up_val = np.roll(phi, 1, axis=0)

    cnt = (
        right_active.astype(np.float64)
        + left_active.astype(np.float64)
        + down_active.astype(np.float64)
        + up_active.astype(np.float64)
    )
    s = (
        np.where(right_active, right_val, 0.0)
        + np.where(left_active, left_val, 0.0)
        + np.where(down_active, down_val, 0.0)
        + np.where(up_active, up_val, 0.0)
    )
    media = np.divide(s, cnt, out=phi.copy(), where=cnt > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(cnt > 0, nuevo, phi)


def paso_expansion_2d(ar, ad, H, rng):
    """Expansion = cortar aristas vivas (horiz+vert), Bernoulli(H) independiente por
    arista. Misma correccion que cs074_rcruz.paso_expansion frente a round(H*N) roto:
    esperanza de fraccion cortada/paso = H, valida tambien para H*L^2 << 1."""
    if H <= 0.0:
        return ar, ad
    ar = ar.copy()
    ad = ad.copy()
    if H >= 1.0:
        ar[:] = False
        ad[:] = False
        return ar, ad
    u_ar = rng.random(ar.shape)
    u_ad = rng.random(ad.shape)
    ar[ar & (u_ar < H)] = False
    ad[ad & (u_ad < H)] = False
    return ar, ad


def evolucionar_2d(phi, ar, ad, H, pasos, rng, null=False):
    contraste0 = float(phi.std())
    for _ in range(pasos):
        phi = paso_difusion_2d(phi, ar, ad)
        ar, ad = paso_expansion_2d(ar, ad, H, rng)
    if null:
        flat = phi.reshape(-1)
        flat = rng.permutation(flat)
        phi = flat.reshape(phi.shape)
    return phi, ar, ad, contraste0


def medir_D_2d(L, eps, seed):
    rng = np.random.default_rng(seed)
    phi = campo_inicial_2d(L, eps, rng)
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)
    c0 = phi.std()
    if c0 <= 0:
        return 0.0
    phi1 = paso_difusion_2d(phi, ar, ad)
    c1 = phi1.std()
    return max(0.0, float((c0 - c1) / c0))


def exergia_2d(phi, contraste0):
    """X_final = rho * v (PROTOCOLO Sec.3). Devuelve tambien v solo (X_var, segundo
    observable de diagnostico) y rho (coherencia) por separado."""
    if contraste0 <= 0 or phi.std() <= 1e-12:
        return {"X_final": 0.0, "X_var": 0.0, "rho": 0.0}
    flat = phi.reshape(-1)
    flat_h = np.roll(phi, 1, axis=1).reshape(-1)
    flat_v = np.roll(phi, 1, axis=0).reshape(-1)
    c_h = np.corrcoef(flat, flat_h)[0, 1]
    c_v = np.corrcoef(flat, flat_v)[0, 1]
    if not np.isfinite(c_h):
        c_h = 0.0
    if not np.isfinite(c_v):
        c_v = 0.0
    rho = max(0.0, float(0.5 * (c_h + c_v)))
    v = float(phi.var() / (contraste0 ** 2))
    return {"X_final": float(rho * v), "X_var": v, "rho": rho}


def medir_pasos_lavado_2d(L, eps, semillas, X_thr=EXERGIA_LAVADO, max_steps=20000, check_every=25):
    """Tiempo medido (pasos) a H=0 para que X_final < X_thr. Sale del propio campo
    (T1). max_steps mas chico que en 1D porque en 2D cada celda tiene 4 vecinos (vs 2
    en 1D) -> difusion decorrela mas rapido; se valida con 'lavo_todas' en el JSON."""
    tiempos = []
    for s in range(semillas):
        rng = np.random.default_rng(30_000 + s)
        phi = campo_inicial_2d(L, eps, rng)
        ar = np.ones((L, L), dtype=bool)
        ad = np.ones((L, L), dtype=bool)
        c0 = float(phi.std())
        if c0 <= 0:
            tiempos.append(0)
            continue
        t_hit = None
        for t in range(1, max_steps + 1):
            phi = paso_difusion_2d(phi, ar, ad)
            if t % check_every == 0:
                if exergia_2d(phi, c0)["X_final"] < X_thr:
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
        "X_thr": X_thr,
        "lavo_todas": all(t < max_steps for t in tiempos),
    }


def corrida_2d(L, eps, H, pasos, seed, null=False):
    rng = np.random.default_rng(seed)
    phi = campo_inicial_2d(L, eps, rng)
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)
    phi, ar, ad, c0 = evolucionar_2d(phi, ar, ad, H, pasos, rng, null=null)
    ex = exergia_2d(phi, c0)
    n_edges = 2 * L * L
    frac_exp = 1.0 - float((ar.sum() + ad.sum()) / n_edges)
    return {
        "X_final": ex["X_final"],
        "X_var": ex["X_var"],
        "rho": ex["rho"],
        "frac_exp": frac_exp,
        "std_ratio": float(phi.std() / c0) if c0 > 0 else 0.0,
    }


def barrido_2d(L, eps_list, r_targets, semillas, pasos_fijo=None):
    filas = []
    meta_por_eps = []
    for eps in eps_list:
        D = float(np.mean([medir_D_2d(L, eps, s) for s in range(semillas)]))
        if eps <= 0:
            cal = {"tiempos": [], "mediana": 0, "pasos": pasos_fijo or 50, "X_thr": EXERGIA_LAVADO, "lavo_todas": True}
            pasos = pasos_fijo or 50
        else:
            if pasos_fijo is not None:
                cal = {"tiempos": [], "mediana": pasos_fijo, "pasos": pasos_fijo, "X_thr": EXERGIA_LAVADO, "lavo_todas": True, "fijo": True}
                pasos = pasos_fijo
            else:
                cal = medir_pasos_lavado_2d(L, eps, max(semillas, 4))
                pasos = cal["pasos"]
        meta_por_eps.append({"eps": eps, "D": D, "calibracion_lavado": cal, "pasos": pasos})

        for r_tgt in r_targets:
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
                r_eff = H / D if D > 0 else float("inf")
            else:
                H = 0.0 if r_tgt == 0 else 1.0
                r_eff = float("inf") if D <= 0 and r_tgt > 0 else 0.0

            Xreal, Xnull, Vreal, Vnull, srr, srn, fracs = [], [], [], [], [], [], []
            for s in range(semillas):
                rr = corrida_2d(L, eps, H, pasos, seed=4000 + s, null=False)
                nn = corrida_2d(L, eps, H, pasos, seed=4000 + s, null=True)
                Xreal.append(rr["X_final"])
                Xnull.append(nn["X_final"])
                Vreal.append(rr["X_var"])
                Vnull.append(nn["X_var"])
                srr.append(rr["std_ratio"])
                srn.append(nn["std_ratio"])
                fracs.append(rr["frac_exp"])
            Xreal = np.array(Xreal)
            Xnull = np.array(Xnull)
            sd = np.sqrt((Xreal.var() + Xnull.var()) / 2.0)
            sd = max(sd, 1.0 / max(len(Xreal), 1))
            z = float((Xreal.mean() - Xnull.mean()) / sd)
            filas.append(
                {
                    "eps": eps,
                    "r_target": r_tgt,
                    "H": H,
                    "D": D,
                    "r": r_eff,
                    "pasos": pasos,
                    "X_final_real": float(Xreal.mean()),
                    "X_final_null": float(Xnull.mean()),
                    "X_final_real_std": float(Xreal.std()),
                    "X_final_null_std": float(Xnull.std()),
                    "X_var_real": float(np.mean(Vreal)),
                    "X_var_null": float(np.mean(Vnull)),
                    "z": z,
                    "std_ratio_real": float(np.mean(srr)),
                    "std_ratio_null": float(np.mean(srn)),
                    "frac_exp_mean": float(np.mean(fracs)),
                }
            )
    return filas, meta_por_eps


def control_r0_ok(filas, X_max=0.15):
    rows = [f for f in filas if f["r_target"] == 0.0 and f["eps"] > 0]
    if not rows:
        return False, {}
    mean_X = float(np.mean([f["X_final_real"] for f in rows]))
    return mean_X < X_max, {"mean_X_r0_eps_gt0": mean_X, "n": len(rows), "X_max": X_max}


# ============================================================================
# Control interno 1D (verificacion cruzada, ver PROTOCOLO Sec.1) -- MISMA formula
# de X_final que el 2D (rho de un eje * v), MISMO grid EPS_LIST/R_TARGETS. NO es
# una copia de cs074_rcruz.py: se reimplementa aqui, standalone, dentro de este
# archivo, sin editar ni importar cs074_rcruz.py.
# ============================================================================

def campo_inicial_1d(N, eps, rng):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    fondo = np.ones(N, dtype=float)
    if eps <= 0.0:
        return fondo
    pert = np.zeros(N, dtype=float)
    for m in range(1, 6):
        fase = rng.uniform(0, 2 * np.pi)
        pert += np.sin(2 * np.pi * m * x + fase) / m
    pert -= pert.mean()
    if pert.std() > 0:
        pert = pert / pert.std()
    return fondo + eps * pert


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


def paso_expansion_1d(activo, H, rng):
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


def exergia_1d(phi, contraste0):
    if contraste0 <= 0 or phi.std() <= 1e-12:
        return {"X_final": 0.0, "X_var": 0.0, "rho": 0.0}
    c = np.corrcoef(phi, np.roll(phi, 1))[0, 1]
    if not np.isfinite(c):
        c = 0.0
    rho = max(0.0, float(c))
    v = float(phi.var() / (contraste0 ** 2))
    return {"X_final": float(rho * v), "X_var": v, "rho": rho}


def medir_D_1d(N, eps, seed):
    rng = np.random.default_rng(seed)
    phi = campo_inicial_1d(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    c0 = phi.std()
    if c0 <= 0:
        return 0.0
    phi1 = paso_difusion_1d(phi, activo)
    c1 = phi1.std()
    return max(0.0, float((c0 - c1) / c0))


def medir_pasos_lavado_1d(N, eps, semillas, X_thr=EXERGIA_LAVADO, max_steps=200000, check_every=50):
    tiempos = []
    for s in range(semillas):
        rng = np.random.default_rng(50_000 + s)
        phi = campo_inicial_1d(N, eps, rng)
        activo = np.ones(N, dtype=bool)
        c0 = float(phi.std())
        if c0 <= 0:
            tiempos.append(0)
            continue
        t_hit = None
        for t in range(1, max_steps + 1):
            phi = paso_difusion_1d(phi, activo)
            if t % check_every == 0:
                if exergia_1d(phi, c0)["X_final"] < X_thr:
                    t_hit = t
                    break
        if t_hit is None:
            t_hit = max_steps
        tiempos.append(t_hit)
    med = int(np.median(tiempos))
    pasos = int(np.ceil(med * MARGEN_LAVADO))
    return {
        "tiempos": tiempos, "mediana": med, "pasos": pasos, "X_thr": X_thr,
        "lavo_todas": all(t < max_steps for t in tiempos),
    }


def corrida_1d(N, eps, H, pasos, seed, null=False):
    rng = np.random.default_rng(seed)
    phi = campo_inicial_1d(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    c0 = float(phi.std())
    for _ in range(pasos):
        phi = paso_difusion_1d(phi, activo)
        activo = paso_expansion_1d(activo, H, rng)
    if null:
        phi = rng.permutation(phi)
    ex = exergia_1d(phi, c0)
    frac_exp = 1.0 - float(activo.mean())
    return {"X_final": ex["X_final"], "X_var": ex["X_var"], "rho": ex["rho"], "frac_exp": frac_exp}


def barrido_1d(N, eps_list, r_targets, semillas, pasos_fijo=None):
    filas = []
    meta_por_eps = []
    for eps in eps_list:
        D = float(np.mean([medir_D_1d(N, eps, s) for s in range(semillas)]))
        if eps <= 0:
            cal = {"tiempos": [], "mediana": 0, "pasos": pasos_fijo or 100, "X_thr": EXERGIA_LAVADO, "lavo_todas": True}
            pasos = pasos_fijo or 100
        else:
            if pasos_fijo is not None:
                cal = {"tiempos": [], "mediana": pasos_fijo, "pasos": pasos_fijo, "X_thr": EXERGIA_LAVADO, "lavo_todas": True, "fijo": True}
                pasos = pasos_fijo
            else:
                cal = medir_pasos_lavado_1d(N, eps, max(semillas, 4))
                pasos = cal["pasos"]
        meta_por_eps.append({"eps": eps, "D": D, "calibracion_lavado": cal, "pasos": pasos})

        for r_tgt in r_targets:
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
                r_eff = H / D if D > 0 else float("inf")
            else:
                H = 0.0 if r_tgt == 0 else 1.0
                r_eff = float("inf") if D <= 0 and r_tgt > 0 else 0.0

            Xreal, Xnull, Vreal, Vnull, fracs = [], [], [], [], []
            for s in range(semillas):
                rr = corrida_1d(N, eps, H, pasos, seed=6000 + s, null=False)
                nn = corrida_1d(N, eps, H, pasos, seed=6000 + s, null=True)
                Xreal.append(rr["X_final"])
                Xnull.append(nn["X_final"])
                Vreal.append(rr["X_var"])
                Vnull.append(nn["X_var"])
                fracs.append(rr["frac_exp"])
            Xreal = np.array(Xreal)
            Xnull = np.array(Xnull)
            sd = np.sqrt((Xreal.var() + Xnull.var()) / 2.0)
            sd = max(sd, 1.0 / max(len(Xreal), 1))
            z = float((Xreal.mean() - Xnull.mean()) / sd)
            filas.append(
                {
                    "eps": eps, "r_target": r_tgt, "H": H, "D": D, "r": r_eff, "pasos": pasos,
                    "X_final_real": float(Xreal.mean()), "X_final_null": float(Xnull.mean()),
                    "X_final_real_std": float(Xreal.std()), "X_final_null_std": float(Xnull.std()),
                    "X_var_real": float(np.mean(Vreal)), "X_var_null": float(np.mean(Vnull)),
                    "z": z, "frac_exp_mean": float(np.mean(fracs)),
                }
            )
    return filas, meta_por_eps


# ============================================================================
# main
# ============================================================================

def rnd_row(f):
    out = dict(f)
    for k in ("D", "H", "r", "X_final_real", "X_final_null", "X_final_real_std",
              "X_final_null_std", "X_var_real", "X_var_null", "std_ratio_real", "std_ratio_null"):
        if k in out and isinstance(out[k], float):
            out[k] = round(out[k], 6)
    out["z"] = round(out["z"], 3)
    return out


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "smoke_L32"
    t0 = time.time()

    if modo == "smoke_L32":
        L = 32
        semillas = 4
        eps_list = [0.0, 1e-3, 0.1, 1.0]
        r_targets = [0.0, 0.1, 1.0, 10.0, 100.0]
        cal_ref = medir_pasos_lavado_2d(L, 1e-3, semillas)
        pasos_fijo = cal_ref["pasos"]
        filas, meta = barrido_2d(L, eps_list, r_targets, semillas, pasos_fijo=pasos_fijo)
        ok, ctrl = control_r0_ok(filas)
        result = {
            "experimento": "E5.1-3", "sub": "2D", "modo": modo, "L": L, "semillas": semillas,
            "eps_list": eps_list, "r_targets": r_targets, "pasos_fijo": pasos_fijo,
            "calibracion_ref": cal_ref, "meta_por_eps": meta,
            "control_r0_lava": ok, "control_r0_detail": ctrl,
            "filas": [rnd_row(f) for f in filas],
            "elapsed_s": time.time() - t0,
        }
        out_json = OUT / f"E5_1_3_resultado_{modo}.json"

    elif modo in ("prod_L32", "prod_L64", "prod_L128", "prod_L256"):
        L = {"prod_L32": 32, "prod_L64": 64, "prod_L128": 128, "prod_L256": 256}[modo]
        semillas = 8
        eps_list = list(EPS_LIST)
        r_targets = list(R_TARGETS)
        cal_ref = medir_pasos_lavado_2d(L, 1e-3, semillas)
        pasos_fijo = cal_ref["pasos"]
        print(f"[calibracion] modo={modo} L={L} eps=1e-3 mediana_lavado={cal_ref['mediana']} "
              f"pasos={pasos_fijo} lavo_todas={cal_ref['lavo_todas']} tiempos={cal_ref['tiempos']}",
              file=sys.stderr, flush=True)
        filas, meta = barrido_2d(L, eps_list, r_targets, semillas, pasos_fijo=pasos_fijo)
        ok, ctrl = control_r0_ok(filas)
        result = {
            "experimento": "E5.1-3", "sub": "2D", "modo": modo, "L": L, "semillas": semillas,
            "eps_list": eps_list, "r_targets": r_targets, "pasos_fijo": pasos_fijo,
            "calibracion_ref": cal_ref, "meta_por_eps": meta,
            "control_r0_lava": ok, "control_r0_detail": ctrl,
            "filas": [rnd_row(f) for f in filas],
            "elapsed_s": time.time() - t0,
        }
        out_json = OUT / f"E5_1_3_resultado_{modo}.json"

    elif modo == "smoke_L256":
        # Sonda de costo a L=256: mismo grid reducido que smoke_L32, para medir el
        # costo real por punto a esta escala y decidir si prod_L256 es viable
        # (PROTOCOLO Sec.8, paso 3).
        L = 256
        semillas = 4
        eps_list = [0.0, 1e-3, 0.1, 1.0]
        r_targets = [0.0, 0.1, 1.0, 10.0, 100.0]
        cal_ref = medir_pasos_lavado_2d(L, 1e-3, semillas)
        pasos_fijo = cal_ref["pasos"]
        print(f"[calibracion] modo={modo} L={L} eps=1e-3 mediana_lavado={cal_ref['mediana']} "
              f"pasos={pasos_fijo} lavo_todas={cal_ref['lavo_todas']} tiempos={cal_ref['tiempos']}",
              file=sys.stderr, flush=True)
        filas, meta = barrido_2d(L, eps_list, r_targets, semillas, pasos_fijo=pasos_fijo)
        ok, ctrl = control_r0_ok(filas)
        result = {
            "experimento": "E5.1-3", "sub": "2D-smoke-costo", "modo": modo, "L": L, "semillas": semillas,
            "eps_list": eps_list, "r_targets": r_targets, "pasos_fijo": pasos_fijo,
            "calibracion_ref": cal_ref, "meta_por_eps": meta,
            "control_r0_lava": ok, "control_r0_detail": ctrl,
            "filas": [rnd_row(f) for f in filas],
            "elapsed_s": time.time() - t0,
            "n_celdas_smoke": len(eps_list) * len(r_targets),
            "n_celdas_prod": len(EPS_LIST) * len(R_TARGETS),
        }
        out_json = OUT / f"E5_1_3_resultado_{modo}.json"

    elif modo == "control_1d":
        # Control interno 1D (PROTOCOLO Sec.1/5) -- N=256 para comparar de forma
        # directa (mismo "lado" nominal) contra L=256 2D si ese corre; si no,
        # sigue siendo comparable contra L=32/64/128.
        N = 256
        semillas = 8
        eps_list = list(EPS_LIST)
        r_targets = list(R_TARGETS)
        cal_ref = medir_pasos_lavado_1d(N, 1e-3, semillas)
        pasos_fijo = cal_ref["pasos"]
        print(f"[calibracion] modo={modo} N={N} eps=1e-3 mediana_lavado={cal_ref['mediana']} "
              f"pasos={pasos_fijo} lavo_todas={cal_ref['lavo_todas']} tiempos={cal_ref['tiempos']}",
              file=sys.stderr, flush=True)
        filas, meta = barrido_1d(N, eps_list, r_targets, semillas, pasos_fijo=pasos_fijo)
        ok, ctrl = control_r0_ok(filas)
        result = {
            "experimento": "E5.1-3", "sub": "control_1D_interno", "modo": modo, "N": N, "semillas": semillas,
            "eps_list": eps_list, "r_targets": r_targets, "pasos_fijo": pasos_fijo,
            "calibracion_ref": cal_ref, "meta_por_eps": meta,
            "control_r0_lava": ok, "control_r0_detail": ctrl,
            "filas": [rnd_row(f) for f in filas],
            "elapsed_s": time.time() - t0,
        }
        out_json = OUT / f"E5_1_3_resultado_{modo}.json"

    else:
        raise SystemExit(
            f"modo desconocido: {modo} "
            "(usa smoke_L32|prod_L32|prod_L64|prod_L128|smoke_L256|prod_L256|control_1d)"
        )

    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "filas"}, ensure_ascii=False))
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[control_r0_lava] {ok} {ctrl}", file=sys.stderr)
    print(f"[elapsed] {result['elapsed_s']:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
