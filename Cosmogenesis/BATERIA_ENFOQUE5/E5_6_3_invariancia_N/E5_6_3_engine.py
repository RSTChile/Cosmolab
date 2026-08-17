#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.6-3 — Invariancia de X a la escala del sistema (N barrido amplio)
=======================================================================================

Experimento 3 de 5, Tema 6 (Definicion y verificacion cruzada de la exergia), Enfoque 5.
Ejecutado por el agente de archivos-prefijo E5_6_3_, en paralelo con otros 29 agentes de
la misma bateria.

Pre-registro (leer ANTES que este archivo, describe el diseno completo, congelado antes de
escribir este motor, incluye el sondeo de factibilidad de computo que fijo el diseno):
    PROTOCOLO_E5.6-3_PREREGISTRO.md  (mismo directorio)

Documento madre (spec exacta, seccion "E5.6-3"):
    ../../BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md

Codigo base LEIDO (comprendido, NO editado, NO importado):
    ../../cs074_rcruz.py
Definicion de X REUTILIZADA verbatim (misma formula, mismo motor fisico) de:
    ../E5_1_1_supervivencia_exergia/E5_1_1_engine.py  (agente E5.1-1, ya en disco)

Pregunta: X/N (exergia por sitio) es intensiva (no depende de N) o depende de N
(efecto de tamano finito / borde)?

Modelo: identico en fisica a cs074_rcruz.py / E5.1-1, con N como eje barrido (E5.1-1 lo
fijaba en N=200). Campo phi en anillo de N sitios, difusion local por aristas vivas,
expansion = corte Bernoulli de aristas, D(N) medido, r=H/D.

Observable X_final (exergia) = c * v, IDENTICO a E5_1_1_engine.py:
    c = autocorrelacion a un paso (corr(phi, roll(phi,1)), clip a >=0)
    v = Var(phi_final) / Var(phi_inicial)
Observable PRIMARIO de este experimento: X/N vs N (mas X crudo en paralelo).

NULL: permutar phi al final (misma optimizacion documentada que E5.1-1: se corre la
evolucion UNA vez por semilla, el NULL se deriva permutando el phi final).

Axiomas (declarados, no fisica real): E1 = conservacion del presupuesto declarado (Sigma
phi), auditada no forzada. E2 = la expansion redistribuye E latente en exergia (marco
interpretativo).

Presupuesto de computo (ver seccion 6 del pre-registro): D(N) escala ~1/N^2 (confirmado
empiricamente, error <2% entre N=64,128,256: K_ref = t_hit*D(N) es practicamente
constante). pasos_fijo(N) se deriva de esa constante SIN fuerza bruta cara en N grande:
    K_ref = mediana(t_hit(N)*D(N))  medido por fuerza bruta SOLO en N in {64,128,256}
    pasos_fijo(N) = ceil(K_ref * MARGEN_LAVADO / D(N))   para TODO N (incluye 2048,4096)
El grid de celdas (eps,r) se reduce en el tramo caro de N (2048,4096) — NUNCA por debajo
de 8 semillas para la celda de señal estocastica (eps=1.0, r=100.0), que es la unica con
cobertura completa en los 7 valores de N (la curva primaria X/N vs N). Ver seccion 6 del
pre-registro para la justificacion completa, declarada ANTES de correr.

*** CORREGIDO 2026-07-25 (ARREGLO 2 + ARREGLO 3, ver ADENDA en
PROTOCOLO_E5.6-3_PREREGISTRO.md) ***
Este motor detectó el bug de ruido dinámico (Arreglo 2: `noise_amp` constante por paso,
sin escalar con `pasos_fijo`, rompía la conservación de energía hasta 71% y colapsaba el
NULL a N>=2048 en la corrida original). Se corrige aquí importando `ruido_por_paso()` de
`_ruido_calibrado.py` (amplitud por paso escalada por 1/sqrt(pasos_fijo), varianza
acumulada total constante). En la misma pasada se agrega la definición canónica de
exergía (`Xh`, Arreglo 3) importada de `_observables_homologadas.py`, calculada EN
PARALELO a la definición vieja de persistencia (`X = c·v`), sobre el mismo φ. El
resultado crudo anterior (con el bug de ruido activo) se conserva en disco como
`E5_6_3_resultado_crudo_DEFINICION_VIEJA_pre_ARREGLOS_2_3.json`.
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

from BATERIA_ENFOQUE5._observables_homologadas import exergia_X as exergia_X_canonica  # noqa: E402
from BATERIA_ENFOQUE5._ruido_calibrado import ruido_por_paso  # noqa: E402

OUT = Path(__file__).resolve().parent

# ---- Constantes de diseño, congeladas en el pre-registro (T1: nunca ajustadas) ----
NOISE_REL = 0.02            # idéntico a E5.1-1
P_LAVADO = 0.05              # idéntico a E5.1-1
MARGEN_LAVADO = 1.15         # idéntico a E5.1-1
EPS_REF_CALIBRACION = 1e-3   # idéntico a E5.1-1
MAX_STEPS_CALIBRACION = 200_000
CHECK_EVERY_CALIB = 50

N_LIST = [64, 128, 256, 512, 1024, 2048, 4096]     # 7 puntos, 6 duplicaciones
N_CALIB_BRUTE = [64, 128, 256]                     # calibración por fuerza bruta (barata)

SEMILLAS_SENAL = 8    # celda (1.0, 100.0) — cobertura completa en los 7 N
SEMILLAS_CTRL_LAVADO = 2   # celda (1.0, 0.0) en tramo caro — chequeo puntual
SEMILLAS_CTRL_CERO = 1     # celda (0.0, 0.0) en tramo caro — determinista, chequeo puntual
SEMILLAS_TRAMO_BARATO = 8  # todas las celdas del tramo N<=1024

N_TRAMO_CARO = {2048, 4096}

# Celdas (eps, r_target) del tramo barato (N<=1024): 5 celdas x 8 semillas
CELDAS_BARATO = [
    (1.0, 0.0),      # control de lavado (sin expansión)
    (1.0, 1.0),      # cerca de la transición nominal r~1
    (1.0, 100.0),    # señal principal (misma celda que en el tramo caro)
    (1.0, 1000.0),   # extremo del rango r
    (0.0, 0.0),      # control cero (determinista)
]

# Celdas del tramo caro (N=2048,4096): reducidas, semillas declaradas por celda
CELDAS_CARO = [
    (1.0, 100.0, SEMILLAS_SENAL),
    (1.0, 0.0, SEMILLAS_CTRL_LAVADO),
    (0.0, 0.0, SEMILLAS_CTRL_CERO),
]


# ---------------------------------------------------------------------------
# Física (fiel a cs074_rcruz.py y E5_1_1_engine.py, reimplementada bajo este prefijo)
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
    """X_final = c * v — IDÉNTICA a E5_1_1_engine.py / persistencia() de la base."""
    if var0 <= 0 or phi.std() <= 1e-14:
        return 0.0
    c = np.corrcoef(phi, np.roll(phi, 1))[0, 1]
    if not np.isfinite(c):
        c = 0.0
    c = max(0.0, float(c))
    v = float(phi.var() / var0)
    return float(c * v)


def medir_pasos_lavado_bruto(N, eps, semillas, P_thr=P_LAVADO, max_steps=MAX_STEPS_CALIBRACION,
                              check_every=CHECK_EVERY_CALIB):
    """Fuerza bruta: tiempo (pasos) a H=0 para que X<P_thr. Solo se usa en N_CALIB_BRUTE
    (barato). Igual método que E5.1-1 / cs074_rcruz.py."""
    tiempos = []
    for s in range(semillas):
        rng = np.random.default_rng(91_000 + s)
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
    return {"tiempos": tiempos, "mediana": med}


def calibrar_K_ref():
    """K_ref = mediana(t_hit(N) * D(N)) sobre N_CALIB_BRUTE. Constante adimensional que
    caracteriza cuántos 'tiempos de difusión locales' hacen falta para lavar el campo a
    P<P_LAVADO — validado empíricamente como ~N-independiente (ver pre-registro sec.6)."""
    productos = []
    detalle = []
    for N in N_CALIB_BRUTE:
        D = float(np.mean([medir_D(N, EPS_REF_CALIBRACION, 70_000 + s) for s in range(8)]))
        cal = medir_pasos_lavado_bruto(N, EPS_REF_CALIBRACION, semillas=8)
        prod = cal["mediana"] * D
        productos.append(prod)
        detalle.append({"N": N, "D": D, "t_hit_mediana": cal["mediana"], "K": prod,
                         "tiempos_bruto": cal["tiempos"]})
    K_ref = float(np.median(productos))
    return K_ref, detalle


def pasos_fijo_de_N(N, D_N, K_ref):
    if D_N <= 0:
        return 1
    return int(np.ceil(K_ref * MARGEN_LAVADO / D_N))


def evolucionar_con_ruido(phi, activo, H, eps, pasos, rng):
    """*** ARREGLO 2 (2026-07-25): noise_amp ya NO es NOISE_REL*eps constante por paso
    (bug detectado por este mismo experimento) -- se calibra con ruido_por_paso() de
    _ruido_calibrado.py para que la varianza acumulada TOTAL sobre `pasos` quede
    constante, independiente de N/pasos_fijo. Ver ADENDA del pre-registro. ***"""
    var0 = float(phi.var())
    e_decl_0 = float(phi.sum())
    noise_amp = ruido_por_paso(NOISE_REL, eps, pasos)
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        if noise_amp > 0:
            phi = phi + noise_amp * rng.standard_normal(phi.shape)
        activo = paso_expansion(activo, H, rng)
    e_decl_1 = float(phi.sum())
    deriva_E = abs(e_decl_1 - e_decl_0) / (abs(e_decl_0) + 1e-300)
    return phi, activo, var0, deriva_E


def corrida_celda(N, eps, H, pasos, seed, guardar_array=False):
    """*** ARREGLO 3 (2026-07-25): además de X_real/X_null (definición vieja de
    persistencia, c*v), calcula EN PARALELO Xh_real/Xh_null (definición canónica,
    exergia_X de _observables_homologadas.py), sobre el MISMO phi_f/phi_null. ***
    Guarda también sum(phi) y sum(phi**2) por corrida (suficiente, junto con N, para
    reconstruir tanto E canónica como X canónica a futuro sin re-simular) y, si
    guardar_array=True (solo para la celda de señal, la más importante), el array phi_f
    crudo completo."""
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    phi_f, activo_f, var0, deriva_E = evolucionar_con_ruido(phi, activo, H, eps, pasos, rng)

    X_real = exergia(phi_f, var0)
    std_ratio_real = float(phi_f.std() / np.sqrt(var0)) if var0 > 0 else 0.0
    Xh_real = exergia_X_canonica(phi_f)

    phi_null = rng.permutation(phi_f)
    X_null = exergia(phi_null, var0)
    std_ratio_null = float(phi_null.std() / np.sqrt(var0)) if var0 > 0 else 0.0
    Xh_null = exergia_X_canonica(phi_null)

    frac_exp = 1.0 - float(activo_f.mean())

    out = {
        "X_real": X_real, "X_null": X_null,
        "Xh_real": Xh_real, "Xh_null": Xh_null,
        "std_ratio_real": std_ratio_real, "std_ratio_null": std_ratio_null,
        "deriva_E": deriva_E, "frac_exp": frac_exp,
        "sum_phi_real": float(phi_f.sum()), "sum_phi2_real": float(np.sum(phi_f ** 2)),
        "sum_phi_null": float(phi_null.sum()), "sum_phi2_null": float(np.sum(phi_null ** 2)),
    }
    if guardar_array:
        out["phi_real_arr"] = [float(v) for v in phi_f]
        out["phi_null_arr"] = [float(v) for v in phi_null]
    return out


def correr_celda(N, eps, r_tgt, D, pasos_fijo, semillas, seed_base, guardar_array=False):
    if D > 0:
        H = float(min(r_tgt * D, 1.0))
        r_eff = H / D
    else:
        H = 0.0 if r_tgt == 0 else 1.0
        r_eff = 0.0 if r_tgt == 0 else float("inf")

    Xr, Xn, Xhr, Xhn, srr, srn, deriva, fracs = [], [], [], [], [], [], [], []
    sum_phi_r, sum_phi2_r, sum_phi_n, sum_phi2_n = [], [], [], []
    phi_arrs_real, phi_arrs_null = [], []
    for s in range(semillas):
        seed = abs(seed_base + int(round(r_tgt * 1000)) * 100 + s + hash((N, eps)) % 997) % (2**32 - 1)
        res = corrida_celda(N, eps, H, pasos_fijo, seed=seed, guardar_array=guardar_array)
        Xr.append(res["X_real"]); Xn.append(res["X_null"])
        Xhr.append(res["Xh_real"]); Xhn.append(res["Xh_null"])
        srr.append(res["std_ratio_real"]); srn.append(res["std_ratio_null"])
        deriva.append(res["deriva_E"]); fracs.append(res["frac_exp"])
        sum_phi_r.append(res["sum_phi_real"]); sum_phi2_r.append(res["sum_phi2_real"])
        sum_phi_n.append(res["sum_phi_null"]); sum_phi2_n.append(res["sum_phi2_null"])
        if guardar_array:
            phi_arrs_real.append(res["phi_real_arr"])
            phi_arrs_null.append(res["phi_null_arr"])

    Xr = np.array(Xr); Xn = np.array(Xn)
    sd = max(np.sqrt((Xr.var() + Xn.var()) / 2.0), 1e-9)
    z = float((Xr.mean() - Xn.mean()) / sd)

    Xhr = np.array(Xhr); Xhn = np.array(Xhn)
    sd_h = max(np.sqrt((Xhr.var() + Xhn.var()) / 2.0), 1e-9)
    z_h = float((Xhr.mean() - Xhn.mean()) / sd_h)

    fila = {
        "N": N, "eps": eps, "r_target": float(r_tgt), "H": H, "D": D, "r_eff": r_eff,
        "pasos": pasos_fijo, "semillas": semillas,
        "X_real_mean": float(Xr.mean()), "X_real_std": float(Xr.std()),
        "X_real_per_seed": [float(v) for v in Xr],
        "X_null_mean": float(Xn.mean()), "X_null_std": float(Xn.std()),
        "X_null_per_seed": [float(v) for v in Xn],
        "z": z,
        "X_por_N_real_mean": float(Xr.mean() / N),
        "X_por_N_real_per_seed": [float(v / N) for v in Xr],
        "Xh_real_mean": float(Xhr.mean()), "Xh_real_std": float(Xhr.std()),
        "Xh_real_per_seed": [float(v) for v in Xhr],
        "Xh_null_mean": float(Xhn.mean()), "Xh_null_std": float(Xhn.std()),
        "Xh_null_per_seed": [float(v) for v in Xhn],
        "z_h": z_h,
        "Xh_por_N_real_mean": float(Xhr.mean() / N),
        "Xh_por_N_real_per_seed": [float(v / N) for v in Xhr],
        "std_ratio_real_mean": float(np.mean(srr)),
        "std_ratio_null_mean": float(np.mean(srn)),
        "deriva_E_max": float(np.max(deriva)), "deriva_E_mean": float(np.mean(deriva)),
        "frac_exp_mean": float(np.mean(fracs)),
        "sum_phi_real_per_seed": sum_phi_r, "sum_phi2_real_per_seed": sum_phi2_r,
        "sum_phi_null_per_seed": sum_phi_n, "sum_phi2_null_per_seed": sum_phi2_n,
    }
    if guardar_array:
        fila["phi_real_arr_per_seed"] = phi_arrs_real
        fila["phi_null_arr_per_seed"] = phi_arrs_null
    return fila


def main():
    t0 = time.time()
    log = []

    def p(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        log.append(line)

    p(f"[E5.6-3] inicio")

    K_ref, detalle_K = calibrar_K_ref()
    p(f"[calibracion K_ref] N_calib={N_CALIB_BRUTE} K_ref={K_ref:.4f} detalle={detalle_K}")

    filas = []
    meta_por_N = []

    for N in N_LIST:
        t_N0 = time.time()
        D = float(np.mean([medir_D(N, EPS_REF_CALIBRACION, 70_000 + s) for s in range(8)]))
        pasos_fijo = pasos_fijo_de_N(N, D, K_ref)
        meta_por_N.append({"N": N, "D": D, "pasos_fijo": pasos_fijo})
        p(f"[N={N}] D={D:.6e} pasos_fijo={pasos_fijo} (derivado de K_ref)")

        if N in N_TRAMO_CARO:
            celdas = CELDAS_CARO
            for (eps, r_tgt, semillas) in celdas:
                es_senal = (eps == 1.0 and r_tgt == 100.0)
                fila = correr_celda(N, eps, r_tgt, D, pasos_fijo, semillas,
                                     seed_base=2_000_000, guardar_array=es_senal)
                filas.append(fila)
                p(f"  celda(eps={eps},r={r_tgt},sem={semillas}) "
                  f"X_real={fila['X_real_mean']:.6f} X_null={fila['X_null_mean']:.6f} "
                  f"z={fila['z']:.2f} X/N={fila['X_por_N_real_mean']:.8f} "
                  f"Xh_real={fila['Xh_real_mean']:.6f} Xh_null={fila['Xh_null_mean']:.6f} "
                  f"z_h={fila['z_h']:.2f} Xh/N={fila['Xh_por_N_real_mean']:.8f}")
        else:
            for (eps, r_tgt) in CELDAS_BARATO:
                es_senal = (eps == 1.0 and r_tgt == 100.0)
                fila = correr_celda(N, eps, r_tgt, D, pasos_fijo, SEMILLAS_TRAMO_BARATO,
                                     seed_base=1_000_000, guardar_array=es_senal)
                filas.append(fila)
                p(f"  celda(eps={eps},r={r_tgt},sem={SEMILLAS_TRAMO_BARATO}) "
                  f"X_real={fila['X_real_mean']:.6f} X_null={fila['X_null_mean']:.6f} "
                  f"z={fila['z']:.2f} X/N={fila['X_por_N_real_mean']:.8f} "
                  f"Xh_real={fila['Xh_real_mean']:.6f} Xh_null={fila['Xh_null_mean']:.6f} "
                  f"z_h={fila['z_h']:.2f} Xh/N={fila['Xh_por_N_real_mean']:.8f}")

        dt_N = time.time() - t_N0
        p(f"[N={N}] completado en {dt_N:.1f}s")

        # Guardado incremental (resiliencia ante interrupciones en corridas largas)
        parcial = {
            "experimento": "E5.6-3",
            "en_progreso": True,
            "N_LIST": N_LIST,
            "K_ref": K_ref,
            "detalle_K_ref": detalle_K,
            "meta_por_N": meta_por_N,
            "filas": filas,
            "elapsed_s_hasta_ahora": time.time() - t0,
        }
        (OUT / "E5_6_3_resultado_crudo.json").write_text(
            json.dumps(parcial, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t0
    p(f"[fin] elapsed={elapsed:.1f}s ({elapsed/3600:.2f}h)")

    # ---- Análisis de intensividad para la celda de señal (eps=1.0, r=100.0) ----
    # X (definición vieja, persistencia c*v) -- ahora con el ruido YA ARREGLADO (Arreglo 2)
    senal = [f for f in filas if f["eps"] == 1.0 and f["r_target"] == 100.0]
    senal_sorted = sorted(senal, key=lambda f: f["N"])
    logN = np.log(np.array([f["N"] for f in senal_sorted], dtype=float))
    logXN = np.log(np.array([max(f["X_por_N_real_mean"], 1e-300) for f in senal_sorted]))
    if len(logN) >= 2:
        slope, intercept = np.polyfit(logN, logXN, 1)
        residuos = logXN - (slope * logN + intercept)
        slope_se = float(np.std(residuos) / (np.std(logN) * np.sqrt(len(logN)))) if np.std(logN) > 0 else float("nan")
    else:
        slope, intercept, slope_se = float("nan"), float("nan"), float("nan")

    # ---- Análisis de intensividad, MISMA celda de señal, definición Xh (canónica, Arreglo 3) ----
    logXhN = np.log(np.array([max(f["Xh_por_N_real_mean"], 1e-300) for f in senal_sorted]))
    if len(logN) >= 2:
        slope_h, intercept_h = np.polyfit(logN, logXhN, 1)
        residuos_h = logXhN - (slope_h * logN + intercept_h)
        slope_h_se = float(np.std(residuos_h) / (np.std(logN) * np.sqrt(len(logN)))) if np.std(logN) > 0 else float("nan")
    else:
        slope_h, intercept_h, slope_h_se = float("nan"), float("nan"), float("nan")

    resultado = {
        "experimento": "E5.6-3",
        "titulo": "Invariancia de X a la escala del sistema (N barrido amplio)",
        "timestamp_inicio": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "N_LIST": N_LIST,
        "N_CALIB_BRUTE": N_CALIB_BRUTE,
        "K_ref": K_ref,
        "detalle_K_ref": detalle_K,
        "P_LAVADO": P_LAVADO,
        "MARGEN_LAVADO": MARGEN_LAVADO,
        "NOISE_REL": NOISE_REL,
        "meta_por_N": meta_por_N,
        "celdas_barato": CELDAS_BARATO,
        "celdas_caro": CELDAS_CARO,
        "N_tramo_caro": sorted(N_TRAMO_CARO),
        "filas": filas,
        "analisis_intensividad_celda_senal": {
            "celda": {"eps": 1.0, "r_target": 100.0},
            "definicion": "X vieja (persistencia, c*v) -- ruido YA ARREGLADO (Arreglo 2)",
            "N_usados": [int(f["N"]) for f in senal_sorted],
            "X_por_N": [f["X_por_N_real_mean"] for f in senal_sorted],
            "pendiente_loglog_alpha": float(slope),
            "pendiente_se_aprox": slope_se,
            "intercepto_loglog": float(intercept),
            "criterio_pass_intensiva": "abs(alpha) < 0.1",
        },
        "analisis_intensividad_celda_senal_Xh": {
            "celda": {"eps": 1.0, "r_target": 100.0},
            "definicion": "Xh canonica (_observables_homologadas.py, Arreglo 3) -- ruido YA ARREGLADO (Arreglo 2)",
            "N_usados": [int(f["N"]) for f in senal_sorted],
            "Xh_por_N": [f["Xh_por_N_real_mean"] for f in senal_sorted],
            "pendiente_loglog_alpha": float(slope_h),
            "pendiente_se_aprox": slope_h_se,
            "intercepto_loglog": float(intercept_h),
            "criterio_pass_intensiva": "abs(alpha) < 0.1",
        },
        "elapsed_s": elapsed,
        "log": log,
    }

    out_json = OUT / "E5_6_3_resultado_crudo.json"
    out_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    p(f"[archivo] {out_json}")
    p(f"[alpha loglog X/N vs N, celda señal, ruido arreglado] {slope:.4f} (SE~{slope_se:.4f})")
    p(f"[alpha loglog Xh/N vs N, celda señal, ruido arreglado, def. canonica] {slope_h:.4f} (SE~{slope_h_se:.4f})")

    (OUT / "E5_6_3_run.log").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
