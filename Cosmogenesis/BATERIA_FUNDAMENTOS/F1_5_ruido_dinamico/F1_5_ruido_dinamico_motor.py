#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1-5 — Robustez frente a ruido dinámico (no cosmético de semilla)
===================================================================

Quién soy: motor de UN experimento de la batería de 24 (F1-5, Enfoque 1).
Pregunta: ¿la persistencia de una diferencia ínfima ε sobrevive si se inyecta
forzamiento estocástico en CADA PASO de la dinámica (no solo en la condición
inicial)? Ésta es la prueba de robustez dinámica que CF-1/CF-2 no tenían
(lección grabada en el documento madre: "10/10 semillas" con una EDP casi
determinista NO es robustez; robustez = perturbar la dinámica).

Protocolo congelado ANTES de este archivo:
  BATERIA_FUNDAMENTOS/F1_5_ruido_dinamico/PROTOCOLO_F1-5_PREREGISTRO.md

Reutiliza SIN EDITAR las primitivas físicas validadas de cs074_rcruz.py
(campo_inicial, paso_difusion, paso_expansion, medir_pasos_lavado,
temperatura_fisica, detectar_cuantizacion). La ÚNICA física nueva que agrega
este archivo es la inyección de ruido gaussiano blanco en cada paso:

    phi = phi + amplitud_ruido * rng.standard_normal(N)

aplicada DESPUÉS de difusión+expansión, en cada uno de los `pasos` de cada
corrida. Todo lo demás (difusión por aristas vivas, corte de aristas por
expansión, medición de D, calibración de pasos_lavado) es exactamente el
código de rcruz, importado tal cual.

Salida: JSON crudo con las ~3840 corridas (REAL+NULL) sin curar, para
auditoría en disco por quien no escribió este código (regla de la batería).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COSMOGENESIS_DIR = HERE.parent.parent  # .../Cosmogenesis
sys.path.insert(0, str(COSMOGENESIS_DIR))

from cs074_rcruz import (  # noqa: E402  (import tras ajustar sys.path, intencional)
    campo_inicial,
    paso_difusion,
    paso_expansion,
    medir_D,
    medir_pasos_lavado,
    temperatura_fisica,
    detectar_cuantizacion,
)

OUT = HERE

# ---------------------------------------------------------------------------
# Grilla pre-registrada (ver PROTOCOLO_F1-5_PREREGISTRO.md, congelada antes
# de correr este motor; no se cambia tras ver resultados).
# ---------------------------------------------------------------------------
N = 200
AMPLITUD_RUIDO = [float(v) for v in np.logspace(-6, -1, 8)]
EPS_LIST = [0.0, 1e-3, 1e-1]          # 0.0 = control obligatorio "ruido sin señal"
R_LIST = [0.0, 0.3, 1.0, 3.0, 10.0]   # cruza r≈1
N_SEMILLAS = 16
CAL_EPS = 1e-2                        # eps de calibración de pasos (punto medio, no se elige por resultado)
CAL_SEMILLAS = 12
MARGEN_LAVADO = 1.15
SALTO_FACTOR = 3.0                    # umbral de "salto artificial" en unidades de std inter-semilla


def persistencia_detallada(phi: np.ndarray, contraste0: float):
    """
    Réplica LITERAL de la fórmula de persistencia() en cs074_rcruz.py, pero
    devuelve también c (autocorrelación sola, acotada [0,1], no normalizada
    por contraste0). Necesario porque a eps=0, contraste0=0 y la fórmula P de
    rcruz da 0.0 SIEMPRE por definición — eso volvería decorativo (T6) el
    control "ruido sin señal" si solo mirásemos P. Decisión de método
    registrada en el protocolo ANTES de correr, no post-hoc.
    """
    if phi.std() <= 1e-12:
        c = 0.0
    else:
        c = float(np.corrcoef(phi, np.roll(phi, 1))[0, 1])
        if not np.isfinite(c):
            c = 0.0
        c = max(0.0, c)
    if contraste0 <= 0 or phi.std() <= 1e-12:
        P = 0.0
    else:
        v = float(phi.var() / (contraste0 ** 2))
        P = c * v
    return float(P), float(c)


def evolucionar_con_ruido(phi, activo, H, pasos, amplitud_ruido, rng, null=False):
    """
    Idéntica a evolucionar() de rcruz (difusión por aristas vivas + corte de
    aristas por expansión), MÁS la única física nueva de F1-5: ruido gaussiano
    blanco inyectado en CADA PASO (no solo en la condición inicial). Ésta es
    la perturbación DINÁMICA que T7 exige.
    """
    contraste0 = float(phi.std())
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        activo = paso_expansion(activo, H, rng)
        if amplitud_ruido > 0.0:
            phi = phi + amplitud_ruido * rng.standard_normal(phi.shape[0])
    if null:
        phi = rng.permutation(phi)
    return phi, activo, contraste0


def corrida_ruido(N, eps, H, pasos, amplitud_ruido, seed, null=False):
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    phi, activo, c0 = evolucionar_con_ruido(
        phi, activo, H, pasos, amplitud_ruido, rng, null=null
    )
    P, c = persistencia_detallada(phi, c0)
    cuantos = detectar_cuantizacion(phi, activo)
    frac_exp = 1.0 - float(activo.mean())
    T_fin = temperatura_fisica(frac_exp)
    return {
        "P": P,
        "c": c,
        "cuantos": cuantos,
        "frac_exp": frac_exp,
        "T_fin_K": T_fin,
        "std_ratio": float(phi.std() / c0) if c0 > 0 else float(phi.std()),
    }


def barrido_f1_5(N, amplitudes, eps_list, r_list, semillas, pasos_fijo, D_por_eps):
    filas = []
    for eps in eps_list:
        D = D_por_eps[eps]
        for r_tgt in r_list:
            if D > 0:
                H = float(min(r_tgt * D, 1.0))
                r_eff = H / D
            else:
                H = 0.0 if r_tgt == 0 else 1.0
                r_eff = 0.0 if r_tgt == 0 else float("inf")

            for amp in amplitudes:
                Preal, Pnull, creal, cnull = [], [], [], []
                Tfin, fracs = [], []
                hist_real = {}
                for s in range(semillas):
                    seed = 1000 + s
                    rr = corrida_ruido(N, eps, H, pasos_fijo, amp, seed=seed, null=False)
                    nn = corrida_ruido(N, eps, H, pasos_fijo, amp, seed=seed, null=True)
                    Preal.append(rr["P"])
                    Pnull.append(nn["P"])
                    creal.append(rr["c"])
                    cnull.append(nn["c"])
                    Tfin.append(rr["T_fin_K"])
                    fracs.append(rr["frac_exp"])
                    for k, cnt in rr["cuantos"].items():
                        hist_real[k] = hist_real.get(k, 0) + cnt
                Preal = np.array(Preal)
                Pnull = np.array(Pnull)
                creal = np.array(creal)
                cnull = np.array(cnull)
                sdP = np.sqrt((Preal.var() + Pnull.var()) / 2.0)
                sdP = max(sdP, 1.0 / max(len(Preal), 1))
                zP = float((Preal.mean() - Pnull.mean()) / sdP)
                sdc = np.sqrt((creal.var() + cnull.var()) / 2.0)
                sdc = max(sdc, 1.0 / max(len(creal), 1))
                zc = float((creal.mean() - cnull.mean()) / sdc)
                filas.append({
                    "eps": eps,
                    "r_target": r_tgt,
                    "r_eff": r_eff,
                    "H": H,
                    "D": D,
                    "amplitud_ruido": amp,
                    "pasos": pasos_fijo,
                    "semillas": semillas,
                    "P_real_mean": float(Preal.mean()),
                    "P_real_std": float(Preal.std()),
                    "P_null_mean": float(Pnull.mean()),
                    "P_null_std": float(Pnull.std()),
                    "z_P": zP,
                    "c_real_mean": float(creal.mean()),
                    "c_real_std": float(creal.std()),
                    "c_null_mean": float(cnull.mean()),
                    "c_null_std": float(cnull.std()),
                    "z_c": zc,
                    "frac_exp_mean": float(np.mean(fracs)),
                    "T_fin_K_mean": float(np.mean(Tfin)),
                    "cuantos_k": {int(k): int(v) for k, v in sorted(hist_real.items())},
                })
    return filas


def analizar_saltos(filas):
    """
    Verificación cruzada (b) del protocolo: a (eps, r) fijos, P_real(amplitud)
    debe decaer SUAVE. Detecta saltos = |ΔP| entre amplitudes log-consecutivas
    > SALTO_FACTOR * dispersión inter-semilla local. No se alisa nada: se
    reporta la lista de saltos tal cual, cruda.
    """
    grupos = {}
    for f in filas:
        key = (f["eps"], f["r_target"])
        grupos.setdefault(key, []).append(f)
    saltos = []
    curvas = {}
    for key, pts in grupos.items():
        pts = sorted(pts, key=lambda f: f["amplitud_ruido"])
        curvas[f"eps={key[0]}_r={key[1]}"] = [
            {"amplitud_ruido": p["amplitud_ruido"], "P_real_mean": p["P_real_mean"],
             "P_real_std": p["P_real_std"], "P_null_mean": p["P_null_mean"]}
            for p in pts
        ]
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            dP = abs(b["P_real_mean"] - a["P_real_mean"])
            disp_local = max(a["P_real_std"], b["P_real_std"], 1e-9)
            if dP > SALTO_FACTOR * disp_local:
                saltos.append({
                    "eps": key[0], "r_target": key[1],
                    "amplitud_ruido_desde": a["amplitud_ruido"],
                    "amplitud_ruido_hasta": b["amplitud_ruido"],
                    "P_desde": a["P_real_mean"], "P_hasta": b["P_real_mean"],
                    "delta_P": dP, "dispersion_local": disp_local,
                    "razon_delta_sobre_disp": dP / disp_local,
                })
    return curvas, saltos


def main():
    t0 = time.time()
    ts_inicio = datetime.now(timezone.utc).isoformat()
    print(f"[F1-5] inicio {ts_inicio}", file=sys.stderr, flush=True)

    # Calibración de pasos: UNA vez, sin ruido dinámico (igual que rcruz),
    # con eps=CAL_EPS (punto medio de la grilla, no elegido por resultado).
    cal = medir_pasos_lavado(N, CAL_EPS, CAL_SEMILLAS)
    cal["pasos"] = int(np.ceil(cal["mediana"] * MARGEN_LAVADO)) if cal["mediana"] > 0 else cal["pasos"]
    pasos_fijo = cal["pasos"]
    print(
        f"[F1-5][calibracion] N={N} eps_cal={CAL_EPS} mediana_lavado={cal['mediana']} "
        f"pasos_fijo={pasos_fijo} lavo_todas={cal['lavo_todas']} tiempos={cal['tiempos']}",
        file=sys.stderr, flush=True,
    )

    # D medido por eps (igual metodología que rcruz: D en un paso de difusión pura).
    D_por_eps = {}
    for eps in EPS_LIST:
        if eps <= 0.0:
            D_por_eps[eps] = 0.0
            continue
        D_por_eps[eps] = float(np.mean([medir_D(N, eps, s) for s in range(N_SEMILLAS)]))
    print(f"[F1-5][D_por_eps] {D_por_eps}", file=sys.stderr, flush=True)

    filas = barrido_f1_5(N, AMPLITUD_RUIDO, EPS_LIST, R_LIST, N_SEMILLAS, pasos_fijo, D_por_eps)
    curvas, saltos = analizar_saltos(filas)

    # Control eps=0 (ruido sin señal): c_real vs c_null en toda la grilla de amplitud.
    ctrl_eps0 = [f for f in filas if f["eps"] == 0.0]
    ctrl_eps0_resumen = {
        "n_puntos": len(ctrl_eps0),
        "max_abs_z_c": float(max((abs(f["z_c"]) for f in ctrl_eps0), default=0.0)),
        "max_c_real_mean": float(max((f["c_real_mean"] for f in ctrl_eps0), default=0.0)),
        "max_c_null_mean": float(max((f["c_null_mean"] for f in ctrl_eps0), default=0.0)),
    }

    # Rango de ruido bajo el cual sobrevive la persistencia (por (eps>0, r)):
    # amplitud máxima con z_P >= 2 y P_real_mean claramente > P_null_mean.
    supervivencia = {}
    for f in filas:
        if f["eps"] <= 0.0:
            continue
        key = f"eps={f['eps']}_r={f['r_target']}"
        supervivencia.setdefault(key, [])
        supervivencia[key].append((f["amplitud_ruido"], f["z_P"], f["P_real_mean"], f["P_null_mean"]))
    rango_supervivencia = {}
    for key, pts in supervivencia.items():
        pts = sorted(pts)
        sobrevive = [p[0] for p in pts if p[1] >= 2.0]
        rango_supervivencia[key] = {
            "amplitud_max_con_z_P_geq_2": max(sobrevive) if sobrevive else None,
            "todas_amplitudes_z_P": [{"amplitud_ruido": p[0], "z_P": p[1]} for p in pts],
        }

    ts_fin = datetime.now(timezone.utc).isoformat()
    elapsed = time.time() - t0

    resultado = {
        "experimento": "F1-5_ruido_dinamico",
        "protocolo": "PROTOCOLO_F1-5_PREREGISTRO.md",
        "base_fisica_no_editada": "cs074_rcruz.py",
        "timestamp_inicio_utc": ts_inicio,
        "timestamp_fin_utc": ts_fin,
        "elapsed_s": elapsed,
        "N": N,
        "amplitud_ruido_grid": AMPLITUD_RUIDO,
        "eps_grid": EPS_LIST,
        "r_grid": R_LIST,
        "n_semillas": N_SEMILLAS,
        "pasos_fijo": pasos_fijo,
        "calibracion_pasos": cal,
        "D_por_eps": D_por_eps,
        "salto_factor_umbral": SALTO_FACTOR,
        "n_corridas_total": len(filas) * N_SEMILLAS * 2,
        "filas": filas,
        "curvas_por_eps_r": curvas,
        "saltos_detectados": saltos,
        "hay_saltos": len(saltos) > 0,
        "control_eps0_ruido_sin_senal": ctrl_eps0_resumen,
        "rango_supervivencia_por_eps_r": rango_supervivencia,
        "pre_inscrito": {
            "a_NULL": "P_real(eps>0) debe superar a P_null en banda r>=1; si P_real~P_null en toda la grilla, es NULO",
            "b_decaimiento_suave": "P(amplitud_ruido) debe decaer sin salto > 3x dispersion inter-semilla; saltos se reportan crudos",
            "c_disco": "este JSON crudo, sin curar",
            "d_control_eps0": "c_real(eps=0) ~ c_null(eps=0) en toda la grilla; si no, ruido dinamico contamina el metodo",
        },
    }

    out_json = OUT / "F1_5_ruido_dinamico_resultado.json"
    out_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[F1-5][archivo] {out_json}", file=sys.stderr, flush=True)
    print(f"[F1-5][elapsed] {elapsed:.1f}s", file=sys.stderr, flush=True)
    print(f"[F1-5][hay_saltos] {len(saltos) > 0} (n={len(saltos)})", file=sys.stderr, flush=True)
    print(f"[F1-5][control_eps0] {ctrl_eps0_resumen}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
