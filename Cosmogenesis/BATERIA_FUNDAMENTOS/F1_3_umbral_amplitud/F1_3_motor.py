#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1-3 — Umbral de amplitud: ¿existe un ε mínimo bajo el cual nada persiste?
===========================================================================

QUIÉN SOY: motor de producción del experimento F1-3 de la batería de
fundamentos (BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md). Pregunta: barriendo ε en
MUCHAS décadas (1e-15 … 1e-1), ¿la persistencia es scale-free (basta S>0) o hay
un piso ε* por debajo del cual nada persiste? Protocolo congelado ANTES de esta
corrida en PROTOCOLO_F1-3_PREREGISTRO.md (mismo directorio) — léalo primero.

QUÉ HAGO:
  1. Reutilizo SIN MODIFICAR la física de cs074_rcruz.py (campo_inicial,
     paso_difusion, paso_expansion, persistencia, medir_D, medir_pasos_lavado)
     importándolas directamente del módulo original (no reimplemento la
     dinámica en float64 — solo agrego instrumentación).
  2. Corro el barrido de producción: 24 ε (log, 1e-15..1e-1) × 16 semillas ×
     6 niveles de ruido dinámico × {real, NULL-permutación} en float64, más el
     NULL primario ε=0 estricto sobre la misma grilla semillas×ruido.
  3. Calculo DOS observables sobre el MISMO estado final φ de cada corrida:
     (a) P = persistencia forma×magnitud (idéntica a cs074_rcruz, importada);
     (b) P_mi = información mutua espacial entre las dos mitades del anillo
         (estimador independiente, ver mi_espacial()) — verificación cruzada
         con el método de F1-2.
  4. Corro un subgrid en numpy.longdouble (misma física, reimplementada aquí
     SOLO para el cross-check de precisión — ver longdouble_*() abajo) con el
     mismo grid de ε × 16 semillas, ruido=0, real + NULL ε=0. Esto es lo que
     decide si un eventual "umbral" en float64 es físico o de redondeo.
  5. Ruido dinámico: forzamiento gaussiano aditivo aplicado a φ DESPUÉS de cada
     paso de difusión (antes de expansión), amplitud σ_ruido barrida — la
     perturbación de la DINÁMICA que evita la trampa T7 (no solo semilla).
  6. Vuelco todo crudo a JSON (una fila por corrida) para auditoría en disco.

NO TOCO cs074_rcruz.py. NO auto-adjudico veredicto — eso lo hace CS con la
curva P(ε) completa que este script entrega.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# --- Importar la física del código base SIN MODIFICARLO --------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cs074_rcruz import (  # noqa: E402
    campo_inicial,
    paso_difusion,
    paso_expansion,
    persistencia,
    medir_D,
    medir_pasos_lavado,
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "resultados"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Parámetros pre-registrados (PROTOCOLO_F1-3_PREREGISTRO.md, secciones 3-4)
# ---------------------------------------------------------------------------
N = 200
R_FIJO = 100.0
N_EPS = 24
EPS_MIN, EPS_MAX = 1e-15, 1e-1
SEMILLAS = list(range(2000, 2016))  # 16 semillas, sin colisión con base/otros F1
RUIDOS = [0.0, 1e-16, 1e-14, 1e-12, 1e-8, 1e-4]  # sigma del forzamiento dinámico
EPS_CAL_PASOS = 1e-3     # eps de referencia para calibrar pasos (como cs074 producción)
SEMILLAS_CAL = 16
P_LAVADO = 0.05
MARGEN_LAVADO = 1.15
N_BINS_MI = 8

# Longdouble cross-check: mismo grid de eps, mismas 16 semillas, ruido=0 solamente
LD_DTYPE = np.longdouble


def eps_grid(n=N_EPS, lo=EPS_MIN, hi=EPS_MAX):
    return list(np.logspace(np.log10(lo), np.log10(hi), n))


# ---------------------------------------------------------------------------
# Observable independiente (verificación cruzada F1-2): información mutua
# espacial entre las dos mitades del anillo, vectorizada con histograma 2D.
# ---------------------------------------------------------------------------
def mi_espacial(phi, n_bins=N_BINS_MI):
    N_ = phi.size
    h = N_ // 2
    A = np.asarray(phi[:h], dtype=np.float64)
    B = np.asarray(phi[h:2 * h], dtype=np.float64)
    if A.std() <= 0 or B.std() <= 0:
        return 0.0
    if A.max() <= A.min() or B.max() <= B.min():
        return 0.0
    joint, _, _ = np.histogram2d(A, B, bins=n_bins)
    total = joint.sum()
    if total <= 0:
        return 0.0
    joint = joint / total
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    denom = pa * pb
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(joint, denom, out=np.zeros_like(joint), where=(joint > 0) & (denom > 0))
        terms = np.where((joint > 0) & (denom > 0), joint * np.log(ratio), 0.0)
    mi = float(terms.sum())
    return max(0.0, mi)


# ---------------------------------------------------------------------------
# Corrida float64 (motor de producción) con ruido dinámico + doble observable
# ---------------------------------------------------------------------------
def corrida_f13(N_, eps, H, pasos, seed, sigma_ruido, null=False, permute=False):
    """
    Una corrida completa: difusión + expansión + (opcional) forzamiento
    estocástico dinámico en cada paso, terminando en NULL de permutación si
    permute=True. null=True/False solo etiqueta la fila (no cambia física);
    permute=True es el NULL-permutación secundario (estilo CS074).
    """
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N_, eps, rng)
    activo = np.ones(N_, dtype=bool)
    c0 = float(phi.std())
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        if sigma_ruido > 0:
            phi = phi + sigma_ruido * rng.standard_normal(N_)
        activo = paso_expansion(activo, H, rng)
    if permute:
        phi = rng.permutation(phi)
    P = persistencia(phi, c0)
    Pmi = mi_espacial(phi)
    frac_exp = 1.0 - float(activo.mean())
    return {"P": P, "P_mi": Pmi, "frac_exp": frac_exp, "std_ratio": float(phi.std() / c0) if c0 > 0 else 0.0}


# ---------------------------------------------------------------------------
# Cross-check de precisión: reimplementación fiel en numpy.longdouble.
# Se reimplementa (no se puede importar directo: cs074_rcruz fuerza float64
# internamente vía np.ones/np.zeros por defecto) SOLO para este cross-check,
# preservando exactamente la misma lógica física, verificada equivalente a
# float64 en los puntos donde float64 no satura (ver protocolo, nota de diseño).
# ---------------------------------------------------------------------------
def ld_campo_inicial(N_, eps, rng, dtype=LD_DTYPE):
    x = np.linspace(0.0, 1.0, N_, endpoint=False).astype(dtype)
    fondo = np.ones(N_, dtype=dtype)
    if eps <= 0.0:
        return fondo, x
    pert = np.zeros(N_, dtype=dtype)
    for m in range(1, 6):
        fase = rng.uniform(0, 2 * np.pi)
        pert += np.sin(2 * np.pi * m * x + fase) / m
    pert -= pert.mean()
    if pert.std() > 0:
        pert = pert / pert.std()
    return fondo + dtype(eps) * pert, x


def ld_paso_difusion(phi, activo):
    left = np.roll(phi, 1)
    right = np.roll(phi, -1)
    e_left = np.roll(activo, 1)
    e_right = activo
    n_nb = e_left.astype(phi.dtype) + e_right.astype(phi.dtype)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + 0.5 * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def ld_paso_expansion(activo, H, rng):
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


def ld_persistencia(phi, c0):
    if c0 <= 0 or float(phi.std()) <= 1e-12:
        return 0.0
    a64 = np.asarray(phi, dtype=np.float64)
    c = np.corrcoef(a64, np.roll(a64, 1))[0, 1]
    if not np.isfinite(c):
        c = 0.0
    c = max(0.0, float(c))
    v = float(phi.var()) / (c0 ** 2)
    return float(c * v)


def corrida_f13_ld(N_, eps, H, pasos, seed, permute=False):
    rng = np.random.default_rng(seed)
    phi, _ = ld_campo_inicial(N_, eps, rng)
    activo = np.ones(N_, dtype=bool)
    c0 = float(phi.std())
    for _ in range(pasos):
        phi = ld_paso_difusion(phi, activo)
        activo = ld_paso_expansion(activo, H, rng)
    if permute:
        phi = rng.permutation(phi)
    P = ld_persistencia(phi, c0)
    Pmi = mi_espacial(phi)
    frac_exp = 1.0 - float(activo.mean())
    return {"P": P, "P_mi": Pmi, "frac_exp": frac_exp}


# ---------------------------------------------------------------------------
# Producción
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log_lines = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        log_lines.append(line)

    log("F1-3 — inicio de la corrida de producción (protocolo congelado 2026-07-24)")

    # --- 1) Calibración de D y pasos (una vez, sobre eps de referencia) ----
    D_cal = float(np.mean([medir_D(N, EPS_CAL_PASOS, s) for s in range(SEMILLAS_CAL)]))
    cal = medir_pasos_lavado(N, EPS_CAL_PASOS, SEMILLAS_CAL, P_thr=P_LAVADO)
    pasos_fijo = cal["pasos"]
    log(f"calibracion: N={N} eps_cal={EPS_CAL_PASOS} D={D_cal:.6e} "
        f"mediana_lavado={cal['mediana']} pasos_fijo={pasos_fijo} lavo_todas={cal['lavo_todas']}")

    eps_list = eps_grid()
    log(f"grid eps: {len(eps_list)} puntos log en [{EPS_MIN:.0e},{EPS_MAX:.0e}]")
    log(f"semillas: {len(SEMILLAS)}  ruidos: {RUIDOS}  r_fijo={R_FIJO}")

    # --- 2) D por cada eps (medido, no impuesto) ----------------------------
    D_por_eps = {}
    for eps in eps_list:
        D_por_eps[eps] = float(np.mean([medir_D(N, eps, s) for s in SEMILLAS[:8]]))

    filas = []

    # --- 3) Barrido principal float64: eps x semillas x ruido --------------
    n_total = len(eps_list) * len(SEMILLAS) * len(RUIDOS)
    contador = 0
    for eps in eps_list:
        D = D_por_eps[eps]
        H = float(min(R_FIJO * D, 1.0)) if D > 0 else 0.0
        r_eff = (H / D) if D > 0 else float("nan")
        for sigma in RUIDOS:
            for seed in SEMILLAS:
                contador += 1
                rr = corrida_f13(N, eps, H, pasos_fijo, seed, sigma, null=False, permute=False)
                nn = corrida_f13(N, eps, H, pasos_fijo, seed, sigma, null=True, permute=True)
                filas.append({
                    "precision": "float64", "eps": eps, "seed": seed, "sigma_ruido": sigma,
                    "D": D, "H": H, "r_eff": r_eff, "pasos": pasos_fijo,
                    "P_real": rr["P"], "P_mi_real": rr["P_mi"], "frac_exp": rr["frac_exp"],
                    "std_ratio_real": rr["std_ratio"],
                    "P_null_perm": nn["P"], "P_mi_null_perm": nn["P_mi"],
                })
                if contador % 500 == 0 or contador == n_total:
                    log(f"barrido principal: {contador}/{n_total} corridas (pares real+null_perm)")

    # --- 4) NULL primario pre-registrado: eps=0 estricto, semillas x ruido -
    log("NULL primario (eps=0 estricto) sobre semillas x ruido")
    filas_null_eps0 = []
    D0 = 0.0
    H0 = 0.0
    for sigma in RUIDOS:
        for seed in SEMILLAS:
            rr = corrida_f13(N, 0.0, H0, pasos_fijo, seed, sigma, null=True, permute=False)
            filas_null_eps0.append({
                "precision": "float64", "eps": 0.0, "seed": seed, "sigma_ruido": sigma,
                "D": D0, "H": H0, "pasos": pasos_fijo,
                "P_eps0": rr["P"], "P_mi_eps0": rr["P_mi"], "frac_exp": rr["frac_exp"],
            })
    log(f"NULL eps=0 estricto: {len(filas_null_eps0)} filas")

    # --- 5) Cross-check de precisión (longdouble), ruido=0, mismas semillas -
    log("cross-check de precision (longdouble): mismo grid de eps, ruido=0")
    filas_ld = []
    # D en longdouble se aproxima con el D de float64 (mismo D físico esperado;
    # el objetivo del cross-check es la persistencia de la señal, no re-medir D
    # con más precisión — H se fija igual que en el barrido float64 para que la
    # comparación sea a igualdad de r_fijo, no de instrumento).
    for i, eps in enumerate(eps_list):
        D = D_por_eps[eps]
        H = float(min(R_FIJO * D, 1.0)) if D > 0 else 0.0
        for seed in SEMILLAS:
            rr = corrida_f13_ld(N, eps, H, pasos_fijo, seed, permute=False)
            filas_ld.append({
                "precision": "longdouble", "eps": eps, "seed": seed, "sigma_ruido": 0.0,
                "D": D, "H": H, "pasos": pasos_fijo,
                "P_real": rr["P"], "P_mi_real": rr["P_mi"], "frac_exp": rr["frac_exp"],
            })
        if (i + 1) % 4 == 0 or (i + 1) == len(eps_list):
            log(f"longdouble: {i+1}/{len(eps_list)} valores de eps completados")

    # NULL eps=0 en longdouble
    filas_ld_null = []
    for seed in SEMILLAS:
        rr = corrida_f13_ld(N, 0.0, 0.0, pasos_fijo, seed, permute=False)
        filas_ld_null.append({
            "precision": "longdouble", "eps": 0.0, "seed": seed, "sigma_ruido": 0.0,
            "D": 0.0, "H": 0.0, "pasos": pasos_fijo,
            "P_eps0": rr["P"], "P_mi_eps0": rr["P_mi"], "frac_exp": rr["frac_exp"],
        })
    log(f"NULL eps=0 longdouble: {len(filas_ld_null)} filas")

    elapsed = time.time() - t0
    log(f"corrida completa en {elapsed:.1f}s ({elapsed/60:.1f} min)")

    resultado = {
        "experimento": "F1-3_umbral_amplitud",
        "protocolo": "PROTOCOLO_F1-3_PREREGISTRO.md",
        "fecha_ejecucion": time.strftime("%Y-%m-%d %H:%M:%S"),
        "codigo_base": "cs074_rcruz.py (no modificado, importado)",
        "parametros": {
            "N": N, "r_fijo": R_FIJO, "n_eps": N_EPS, "eps_min": EPS_MIN, "eps_max": EPS_MAX,
            "semillas": SEMILLAS, "ruidos_dinamicos": RUIDOS,
            "pasos_fijo": pasos_fijo, "D_cal": D_cal, "calibracion_lavado": cal,
            "n_bins_mi": N_BINS_MI,
        },
        "D_por_eps": {f"{k:.3e}": v for k, v in D_por_eps.items()},
        "filas_float64": filas,
        "filas_null_eps0_float64": filas_null_eps0,
        "filas_longdouble": filas_ld,
        "filas_null_eps0_longdouble": filas_ld_null,
        "elapsed_s": elapsed,
    }

    out_json = OUT_DIR / "F1_3_resultado_produccion.json"
    out_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "F1_3_log_ejecucion.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"escrito: {out_json}")
    print(f"\n[F1-3] listo. JSON: {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
