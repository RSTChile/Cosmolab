#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quién soy / qué hago (código autodescriptivo):
  Verificación del ARREGLO 2 (ver `_ruido_calibrado.py` en esta misma carpeta y
  INSTRUCCION_ARREGLOS_antes_de_seguir_PARA_CC.md, sección "ARREGLO 2").

  Repite, a escala reducida (N hasta 2048, no 4096; menos semillas), el barrido de
  N de E5_6_3_invariancia_N/E5_6_3_engine.py -- el experimento que DETECTÓ el
  problema -- pero corriendo la MISMA celda de señal (eps=1.0, r_target=100.0) y
  los mismos controles (r=0.0 lavado; eps=0.0,r=0.0 cero) con DOS mecanismos de
  ruido en paralelo, para comparar directamente:

    RUIDO_VIEJO: noise_amp = NOISE_REL * eps constante por paso (el bug original,
                 reimplementado aquí verbatim solo para la comparación -- NO se usa
                 en ningún experimento nuevo).
    RUIDO_NUEVO: aplicar_ruido() de _ruido_calibrado.py (la corrección: amplitud
                 por paso escalada por 1/sqrt(pasos_fijo)).

  Criterio de éxito (declarado ANTES de correr, T3): con RUIDO_NUEVO, la deriva de
  conservación (E1, |E_final-E_inicial|/|E_inicial| sobre Sigma phi) debe permanecer
  ACOTADA (no crecer sin control) al aumentar N, y el NULL (permutación espacial de
  phi_final) debe seguir discriminando de REAL (|z| no debe colapsar a ~0) en el N
  más grande probado -- exactamente los dos síntomas que E5_6_3 documentó rotos con
  RUIDO_VIEJO a N>=2048.

  Física (difusión, expansión, campo inicial, exergía) reimplementada aquí bajo
  este prefijo, fiel a cs074_rcruz.py / E5_6_3_engine.py (leídos, NO editados, NO
  importados -- misma convención que el resto de la batería). K_ref se recalibra
  aquí mismo (barato, N<=256) en vez de reusar el de E5_6_3, para que este script
  sea autocontenido.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ruido_calibrado import ruido_por_paso  # noqa: E402

OUT = Path(__file__).resolve().parent

NOISE_REL = 0.02
P_LAVADO = 0.05
MARGEN_LAVADO = 1.15
EPS_REF_CALIBRACION = 1e-3
MAX_STEPS_CALIBRACION = 200_000
CHECK_EVERY_CALIB = 50

N_LIST = [64, 128, 256, 512, 1024, 2048]     # reducido de E5_6_3 (sin 4096, escala verificacion)
N_CALIB_BRUTE = [64, 128, 256]

SEMILLAS_SENAL = 4     # reducido de 8 (E5_6_3) -- solo se necesita ver la tendencia, no estadistica fina
SEMILLAS_CTRL = 2

EPS_SENAL, R_SENAL = 1.0, 100.0
EPS_LAVADO, R_LAVADO = 1.0, 0.0
EPS_CERO, R_CERO = 0.0, 0.0


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
    productos = []
    detalle = []
    for N in N_CALIB_BRUTE:
        D = float(np.mean([medir_D(N, EPS_REF_CALIBRACION, 70_000 + s) for s in range(8)]))
        cal = medir_pasos_lavado_bruto(N, EPS_REF_CALIBRACION, semillas=8)
        prod = cal["mediana"] * D
        productos.append(prod)
        detalle.append({"N": N, "D": D, "t_hit_mediana": cal["mediana"], "K": prod})
    K_ref = float(np.median(productos))
    return K_ref, detalle


def pasos_fijo_de_N(D_N, K_ref):
    if D_N <= 0:
        return 1
    return int(np.ceil(K_ref * MARGEN_LAVADO / D_N))


def evolucionar(phi, activo, H, eps, pasos, rng, modo_ruido):
    """modo_ruido: 'viejo' (bug original, constante) | 'nuevo' (calibrado, Arreglo 2) | 'sin'."""
    e_decl_0 = float(phi.sum())
    if modo_ruido == "viejo":
        amp_paso = NOISE_REL * eps
    elif modo_ruido == "nuevo":
        amp_paso = ruido_por_paso(NOISE_REL, eps, pasos)
    else:
        amp_paso = 0.0
    for _ in range(pasos):
        phi = paso_difusion(phi, activo)
        if amp_paso > 0:
            phi = phi + amp_paso * rng.standard_normal(phi.shape)
        activo = paso_expansion(activo, H, rng)
    e_decl_1 = float(phi.sum())
    deriva_E = abs(e_decl_1 - e_decl_0) / (abs(e_decl_0) + 1e-300)
    return phi, activo, deriva_E


def corrida(N, eps, H, pasos, seed, modo_ruido):
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    var0 = float(phi.var())
    phi_f, activo_f, deriva_E = evolucionar(phi, activo, H, eps, pasos, rng, modo_ruido)
    X_real = exergia(phi_f, var0)
    phi_null = rng.permutation(phi_f)
    X_null = exergia(phi_null, var0)
    return {"X_real": X_real, "X_null": X_null, "deriva_E": deriva_E}


def correr_celda(N, eps, H, pasos, semillas, seed_base, modo_ruido):
    Xr, Xn, deriva = [], [], []
    for s in range(semillas):
        seed = abs(seed_base + s + hash((N, modo_ruido)) % 997) % (2**32 - 1)
        r = corrida(N, eps, H, pasos, seed, modo_ruido)
        Xr.append(r["X_real"]); Xn.append(r["X_null"]); deriva.append(r["deriva_E"])
    Xr = np.array(Xr); Xn = np.array(Xn)
    sd = max(np.sqrt((Xr.var() + Xn.var()) / 2.0), 1e-9)
    z = float((Xr.mean() - Xn.mean()) / sd)
    return {
        "N": N, "modo_ruido": modo_ruido,
        "X_real_mean": float(Xr.mean()), "X_null_mean": float(Xn.mean()), "z": z,
        "deriva_E_max": float(np.max(deriva)), "deriva_E_mean": float(np.mean(deriva)),
    }


def main():
    t0 = time.time()
    log = []

    def p(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        log.append(line)

    p("[verificacion Arreglo 2] inicio")
    K_ref, detalle_K = calibrar_K_ref()
    p(f"[calibracion K_ref] N_calib={N_CALIB_BRUTE} K_ref={K_ref:.4f}")

    filas = []
    for N in N_LIST:
        t_N0 = time.time()
        D = float(np.mean([medir_D(N, EPS_REF_CALIBRACION, 70_000 + s) for s in range(8)]))
        pasos = pasos_fijo_de_N(D, K_ref)
        H = float(min(R_SENAL * D, 1.0))
        p(f"[N={N}] D={D:.6e} pasos_fijo={pasos} H_senal={H:.6e}")

        for modo in ("viejo", "nuevo"):
            fila_senal = correr_celda(N, EPS_SENAL, H, pasos, SEMILLAS_SENAL, 1_000_000, modo)
            fila_lavado = correr_celda(N, EPS_LAVADO, 0.0, pasos, SEMILLAS_CTRL, 2_000_000, modo)
            fila_cero = correr_celda(N, EPS_CERO, 0.0, pasos, SEMILLAS_CTRL, 3_000_000, modo)
            filas.append({"celda": "senal", **fila_senal})
            filas.append({"celda": "lavado", **fila_lavado})
            filas.append({"celda": "cero", **fila_cero})
            p(f"  [{modo}] senal: z={fila_senal['z']:.3f} deriva_E_max={fila_senal['deriva_E_max']:.4e} "
              f"| lavado deriva_E_max={fila_lavado['deriva_E_max']:.4e} "
              f"| cero deriva_E_max={fila_cero['deriva_E_max']:.4e}")

        dt_N = time.time() - t_N0
        p(f"[N={N}] completado en {dt_N:.1f}s")

        parcial = {"N_LIST": N_LIST, "K_ref": K_ref, "filas": filas,
                   "elapsed_s_hasta_ahora": time.time() - t0}
        (OUT / "_verificacion_arreglo2_resultado.json").write_text(
            json.dumps(parcial, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.time() - t0
    p(f"[fin] elapsed={elapsed:.1f}s")

    # ---- Veredicto declarado (T3, fijado en el docstring de este archivo antes de correr) ----
    senal_viejo = [f for f in filas if f["celda"] == "senal" and f["modo_ruido"] == "viejo"]
    senal_nuevo = [f for f in filas if f["celda"] == "senal" and f["modo_ruido"] == "nuevo"]
    senal_viejo_sorted = sorted(senal_viejo, key=lambda f: f["N"])
    senal_nuevo_sorted = sorted(senal_nuevo, key=lambda f: f["N"])

    deriva_viejo_ultimo = senal_viejo_sorted[-1]["deriva_E_max"]
    deriva_nuevo_ultimo = senal_nuevo_sorted[-1]["deriva_E_max"]
    deriva_nuevo_primero = senal_nuevo_sorted[0]["deriva_E_max"]
    z_nuevo_ultimo = senal_nuevo_sorted[-1]["z"]
    z_nuevo_primero = senal_nuevo_sorted[0]["z"]

    # criterio: la deriva con ruido NUEVO en el N mas grande no debe ser mucho mayor que en el
    # N mas chico (acotada, no creciendo sin control como con el ruido viejo), y z debe seguir
    # siendo detectable (|z|>2) en el N mas grande.
    creciente_sin_control = deriva_nuevo_ultimo > 10 * max(deriva_nuevo_primero, 1e-12)
    null_sigue_discriminando = abs(z_nuevo_ultimo) > 2.0

    veredicto = {
        "deriva_E_max_ruido_VIEJO_en_N_mas_grande": deriva_viejo_ultimo,
        "deriva_E_max_ruido_NUEVO_en_N_mas_chico": deriva_nuevo_primero,
        "deriva_E_max_ruido_NUEVO_en_N_mas_grande": deriva_nuevo_ultimo,
        "z_ruido_NUEVO_en_N_mas_chico": z_nuevo_primero,
        "z_ruido_NUEVO_en_N_mas_grande": z_nuevo_ultimo,
        "deriva_NUEVO_crece_mas_de_10x_entre_N_extremos": bool(creciente_sin_control),
        "NULL_sigue_discriminando_en_N_grande_nuevo_ruido": bool(null_sigue_discriminando),
        "ARREGLO_2_VERIFICADO": bool((not creciente_sin_control) and null_sigue_discriminando),
    }

    resultado = {
        "experimento": "verificacion_arreglo2_N_sweep",
        "N_LIST": N_LIST,
        "N_CALIB_BRUTE": N_CALIB_BRUTE,
        "K_ref": K_ref,
        "detalle_K_ref": detalle_K,
        "NOISE_REL": NOISE_REL,
        "filas": filas,
        "veredicto": veredicto,
        "elapsed_s": elapsed,
        "log": log,
    }
    out_json = OUT / "_verificacion_arreglo2_resultado.json"
    out_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    p(f"[archivo] {out_json}")
    p(f"[veredicto] {json.dumps(veredicto, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
