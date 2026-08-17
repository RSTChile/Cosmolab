#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.5-3 — Reversibilidad de la muerte térmica: ¿re-inyectar ε la revierte?
===========================================================================

Quién soy / qué hago (código autodescriptivo):
  Este motor pertenece a la batería ENFOQUE 5 (energía/exergía/entropía),
  experimento E5.5-3 (Tema 5 — muerte térmica vs Nada). Pregunta: una vez que
  el campo llegó a equilibrio (muerte térmica, X≈0 medido, no asumido), si le
  inyecto una NUEVA perturbación ε de distinta amplitud en distintos momentos
  (distinto tiempo llevando muerto), ¿revive la exergía (recuperable) o el
  equilibrio ya no responde (absorbente)?

  Reutiliza la MISMA definición de X que E5.1-1 (única pieza de Tema 1/5 en
  disco al congelar el pre-registro):
      X = c · v ,  c = corr(φ, roll(φ,1)) clip≥0 ,  v = Var(φ)/Var(φ_inicial)
  y la MISMA física de difusión de cs074_rcruz.py (paso_difusion:
  nuevo = φ + 0.5·(media_vecinos−φ)).

  Optimización de implementación (NO cambia la física ni las fórmulas, solo
  cómo se calculan — verificado bit-a-bit contra las funciones escalares
  originales de cs074_rcruz.py antes de usarse en producción, ver
  `_verificar_equivalencia_con_base()`): como el régimen bajo prueba es H=0
  fijo en TODA la corrida (protocolo §2 — sin expansión, todas las aristas
  siempre activas), la máscara de aristas de `paso_difusion` es trivial
  (n_nb=2 en todo sitio, todo el tiempo) y la difusión se puede vectorizar en
  un eje extra de "lote" (muchas semillas/ramas evolucionando en paralelo con
  la misma física, cada una su propio φ). Esto reduce el tiempo de cómputo en
  ~1-2 órdenes de magnitud sin tocar una sola fórmula.

  Protocolo completo, predicción pre-registrada y grillas: ver
  PROTOCOLO_E5.5-3_PREREGISTRO.md (congelado ANTES de escribir este archivo).

  *** CORREGIDO 2026-07-25 (ARREGLO 1, ver ADENDA en PROTOCOLO_E5.5-3_PREREGISTRO.md
  y INSTRUCCION_ARREGLOS_antes_de_seguir_PARA_CC.md) ***
  La versión original de este motor "revivía" X sumando un patrón de perturbación
  NUEVO (independiente, std=1) a φ_muerto -- eso importa varianza de afuera del
  presupuesto declarado, viola el axioma de conservación del proyecto (la energía no
  se crea), y el chequeo de Σφ que este mismo motor ya hacía NO lo detectaba (el
  patrón inyectado tiene media≈0, así que Σφ casi no cambiaba aunque Σφ² sí).
  Corrección: la re-inyección ahora es `redistribuir_energia()` -- una permutación
  PARCIAL de los valores que YA existen en φ_muerto (se reordena una fracción de los
  sitios entre sí). Conserva Σφ Y Σφ² EXACTAMENTE por construcción algebraica (son
  los mismos valores, solo reordenados) -- no hay de dónde crear energía. El eje
  `amplitud_reinyectada` (perturbación externa) se reinterpreta como
  `fraccion_redistribuida` (qué fracción del campo ya existente se reordena), mismo
  rango numérico [1e-6, 1] y mismo número de puntos que el pre-registro original.

No se importa nada de otros experimentos paralelos (aislamiento del agente).
No se edita cs074_rcruz.py.
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

from cs074_rcruz import campo_inicial, paso_difusion, persistencia  # noqa: E402
from BATERIA_ENFOQUE5._observables_homologadas import exergia_X as exergia_X_canonica_escalar  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes declaradas ANTES de correr (protocolo §9) — no se tocan tras ver
# resultados (T1/T3).
# ---------------------------------------------------------------------------
N = 200
EPS_INICIAL = 1.0
H_MUERTE = 0.0  # régimen bajo prueba: difusión pura, sin expansión (protocolo §2)
THR_MUERTE = 0.02
CHECK_EVERY = 50
MAX_CAL_STEPS = 200_000
CONFIRM_CHECKS = 3  # nº de chequeos consecutivos por debajo del umbral para declarar muerte

MOMENTO_FACTORS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
FRACCIONES = np.logspace(-6, 0, 13).tolist()  # ex-AMPLITUDES; misma grilla, Arreglo 1
SEMILLAS = list(range(16))  # >=12 pedido; 16 para robustez extra
CHECKPOINTS_POST = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]

SEED_BASE_FASEA = 1000        # semilla de la corrida de vida/muerte
SEED_BASE_REINY = 90_000      # semilla del patrón de re-inyección (protocolo §9)
SEED_BASE_CAL = 500_000       # semillas piloto de calibración (separadas de las de producción)
N_SEMILLAS_CAL = 6


# ---------------------------------------------------------------------------
# Física vectorizada (lote) — equivalente exacta a cs074_rcruz bajo H=0 fijo
# ---------------------------------------------------------------------------
def batch_paso_difusion(phi_batch):
    """phi_batch: shape (n_filas, N). Todas las aristas activas siempre
    (H=0) => n_nb=2 en cada sitio siempre => misma fórmula de paso_difusion
    pero sin la máscara/división (que sería no-operación de todos modos)."""
    left = np.roll(phi_batch, 1, axis=-1)
    right = np.roll(phi_batch, -1, axis=-1)
    media = 0.5 * (left + right)
    return phi_batch + 0.5 * (media - phi_batch)


def batch_persistencia(phi_batch, c0_vec):
    """Versión vectorizada de persistencia() de cs074_rcruz, una fila por
    corrida. c0_vec: shape (n_filas,) — contraste0 (std de φ en t=0) de la
    corrida ORIGINAL a la que pertenece cada fila (protocolo §3: X mide
    cuánta capacidad ORIGINAL se recuperó)."""
    c0_vec = np.asarray(c0_vec, dtype=float)
    mean = phi_batch.mean(axis=-1, keepdims=True)
    x = phi_batch - mean
    num = np.sum(x * np.roll(x, 1, axis=-1), axis=-1)
    den = np.sum(x * x, axis=-1)
    c = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    c = np.clip(c, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.var(phi_batch, axis=-1) / (c0_vec ** 2)
    X = c * v
    malo = (c0_vec <= 0) | (phi_batch.std(axis=-1) <= 1e-12)
    X = np.where(malo, 0.0, X)
    return X


def batch_exergia_X_canonica(phi_batch):
    """ARREGLO 3 (definicion comun, ver INSTRUCCION_recorrer_5_definicion_comun_PARA_CC.md):
    version vectorizada (una fila por corrida) de exergia_X() de
    BATERIA_ENFOQUE5/_observables_homologadas.py -- X(t) = (1/N) sum_i (phi_i(t)-1)^2.
    Misma formula exacta que la version escalar (verificada en
    _verificar_equivalencia_con_base), solo vectorizada por rendimiento. Se calcula EN
    PARALELO a X_boost/curva_X (definicion persistencia c*v, heredada de E5.1-1) para
    poder comparar veredicto viejo vs veredicto nuevo, tal como pidio el director."""
    return np.mean((phi_batch - 1.0) ** 2, axis=-1)


def _verificar_equivalencia_definicion_canonica():
    """Sanity check: la version vectorizada de X canonica debe coincidir EXACTO con la
    funcion escalar importada de _observables_homologadas.py sobre los mismos datos."""
    rng = np.random.default_rng(999)
    phi_batch = rng.normal(loc=1.0, scale=0.3, size=(5, N))
    batch_vals = batch_exergia_X_canonica(phi_batch)
    escalar_vals = np.array([exergia_X_canonica_escalar(row) for row in phi_batch])
    max_diff = float(np.max(np.abs(batch_vals - escalar_vals)))
    assert max_diff < 1e-12, f"definicion canonica: mismatch batch vs escalar ({max_diff})"
    return {"max_diff_Xcanonica": max_diff}


def _verificar_equivalencia_con_base():
    """Sanity check bit-a-bit: la física vectorizada debe reproducir EXACTO
    (hasta error de punto flotante) a las funciones escalares originales de
    cs074_rcruz.py. Se corre una vez al inicio de main(); si falla, el motor
    aborta (no se corre nada con física no verificada)."""
    rng = np.random.default_rng(12345)
    phi = rng.normal(size=N)
    activo = np.ones(N, dtype=bool)
    p_scalar = phi.copy()
    for _ in range(37):
        p_scalar = paso_difusion(p_scalar, activo)
    X_scalar = persistencia(p_scalar, float(phi.std()))

    p_batch = np.tile(phi, (3, 1))
    for _ in range(37):
        p_batch = batch_paso_difusion(p_batch)
    X_batch = batch_persistencia(p_batch, np.full(3, phi.std()))

    max_diff_phi = float(np.max(np.abs(p_scalar - p_batch[0])))
    max_diff_X = float(np.max(np.abs(X_batch - X_scalar)))
    assert max_diff_phi < 1e-12, f"physics mismatch phi: {max_diff_phi}"
    assert max_diff_X < 1e-12, f"physics mismatch X: {max_diff_X}"
    return {"max_diff_phi": max_diff_phi, "max_diff_X": max_diff_X}


def redistribuir_energia(base_row, frac, rng):
    """ARREGLO 1: re-inyección por REDISTRIBUCIÓN, no por suma de patrón externo.
    Elige round(frac*N) sitios de base_row y permuta sus valores ENTRE SÍ (mismo
    multiconjunto de valores, otro orden espacial). Conserva Σφ y Σφ² EXACTAMENTE
    (hasta error de punto flotante) por construcción -- no importa varianza de
    afuera del presupuesto ya existente en base_row. frac=0 (o <2 sitios) -> sin
    cambio (idéntico al NULL). frac=1 -> permutación completa (barajado total)."""
    N_local = base_row.size
    n_mover = int(round(frac * N_local))
    fila = base_row.copy()
    if n_mover >= 2:
        idx = rng.choice(N_local, size=n_mover, replace=False)
        fila[idx] = base_row[rng.permutation(idx)]
    return fila


# ---------------------------------------------------------------------------
# Calibración: mide t_muerte (pasos a H=0 para que X<THR_MUERTE, sostenido)
# ---------------------------------------------------------------------------
def calibrar_t_muerte():
    """Batch de N_SEMILLAS_CAL semillas piloto evolucionando juntas."""
    rngs = [np.random.default_rng(SEED_BASE_CAL + s) for s in range(N_SEMILLAS_CAL)]
    phi0_list = []
    c0_list = []
    for rng in rngs:
        phi0, _ = campo_inicial(N, EPS_INICIAL, rng)
        phi0_list.append(phi0)
        c0_list.append(float(phi0.std()))
    phi_batch = np.stack(phi0_list, axis=0)
    c0_vec = np.array(c0_list)

    seguidos = np.zeros(N_SEMILLAS_CAL, dtype=int)
    t_hit = np.full(N_SEMILLAS_CAL, -1, dtype=int)
    t = 0
    while t < MAX_CAL_STEPS and np.any(t_hit < 0):
        phi_batch = batch_paso_difusion(phi_batch)
        t += 1
        if t % CHECK_EVERY == 0:
            X = batch_persistencia(phi_batch, c0_vec)
            bajo = X < THR_MUERTE
            seguidos = np.where(bajo, seguidos + 1, 0)
            recien = (seguidos >= CONFIRM_CHECKS) & (t_hit < 0)
            t_hit = np.where(recien, t - CHECK_EVERY * (CONFIRM_CHECKS - 1), t_hit)
    t_hit = np.where(t_hit < 0, MAX_CAL_STEPS, t_hit)
    tiempos = t_hit.tolist()
    mediana = int(np.median(tiempos))
    return {"tiempos": tiempos, "mediana": mediana, "THR_MUERTE": THR_MUERTE,
            "CHECK_EVERY": CHECK_EVERY, "CONFIRM_CHECKS": CONFIRM_CHECKS,
            "todas_murieron": bool(all(t < MAX_CAL_STEPS for t in tiempos))}


# ---------------------------------------------------------------------------
# Fase A: vida y muerte + snapshots en la grilla de momentos (batch = semillas)
# ---------------------------------------------------------------------------
def fase_a_snapshots_batch(seeds, momento_steps_sorted):
    phi0_list = []
    c0_list = []
    suma0_list = []
    for seed in seeds:
        rng = np.random.default_rng(SEED_BASE_FASEA + seed)
        phi0, _ = campo_inicial(N, EPS_INICIAL, rng)
        phi0_list.append(phi0)
        c0_list.append(float(phi0.std()))
        suma0_list.append(float(phi0.sum()))
    phi_batch = np.stack(phi0_list, axis=0)
    c0_vec = np.array(c0_list)

    snapshots = {}  # t_step -> array shape (n_seeds, N)
    if 0 in momento_steps_sorted:
        snapshots[0] = phi_batch.copy()

    t = 0
    for t_target in [s for s in momento_steps_sorted if s > 0]:
        pasos = t_target - t
        for _ in range(pasos):
            phi_batch = batch_paso_difusion(phi_batch)
        t = t_target
        snapshots[t_target] = phi_batch.copy()

    X_por_snapshot = {
        tt: batch_persistencia(snapshots[tt], c0_vec).tolist() for tt in momento_steps_sorted
    }
    Xh_por_snapshot = {
        tt: batch_exergia_X_canonica(snapshots[tt]).tolist() for tt in momento_steps_sorted
    }
    return {
        "c0_vec": c0_vec,
        "suma0_list": suma0_list,
        "snapshots": snapshots,       # t_step -> (n_seeds, N)
        "X_por_snapshot": X_por_snapshot,  # t_step -> [X por seed], definicion persistencia (vieja)
        "Xh_por_snapshot": Xh_por_snapshot,  # t_step -> [X por seed], definicion canonica (Arreglo 3)
    }


# ---------------------------------------------------------------------------
# Fases B+C, en lote: para un momento dado, TODAS las semillas × TODAS las
# ramas (NULL + 13 amplitudes) evolucionan juntas en un solo tensor.
# ---------------------------------------------------------------------------
def fase_bc_momento(phi_muertos, c0_vec, seeds, momento_idx, fracciones, post_pasos,
                     checkpoints_frac):
    """phi_muertos: shape (n_seeds, N) — snapshot muerto de cada semilla en
    este momento. Construye lote de (n_seeds * (1+n_frac), N): fila 0 de cada
    semilla = NULL (frac=0, sin redistribuir), filas 1..n_frac = fracciones.
    ARREGLO 1: la re-inyección es redistribuir_energia() (permutación parcial),
    no suma de patrón externo -- conserva Σφ y Σφ² exactamente por construcción."""
    n_seeds = len(seeds)
    n_frac = len(fracciones)
    n_ramas = 1 + n_frac  # NULL + fracciones
    N_local = phi_muertos.shape[1]

    lote = np.zeros((n_seeds, n_ramas, N_local), dtype=float)
    suma_pre = np.zeros((n_seeds, n_ramas))
    suma_post_iny = np.zeros((n_seeds, n_ramas))
    sumasq_pre = np.zeros((n_seeds, n_ramas))
    sumasq_post_iny = np.zeros((n_seeds, n_ramas))

    for si, seed in enumerate(seeds):
        base_row = phi_muertos[si]
        suma_pre[si, :] = float(base_row.sum())
        sumasq_pre[si, :] = float(np.sum(base_row ** 2))
        # rama 0: NULL, sin redistribuir
        lote[si, 0, :] = base_row
        suma_post_iny[si, 0] = float(base_row.sum())
        sumasq_post_iny[si, 0] = float(np.sum(base_row ** 2))
        # ramas 1..n_frac: redistribución por (seed, momento, fraccion)
        for fi, frac in enumerate(fracciones):
            rng_reiny = np.random.default_rng(
                SEED_BASE_REINY + 1000 * seed + 10 * momento_idx + fi
            )
            fila = redistribuir_energia(base_row, frac, rng_reiny)
            lote[si, 1 + fi, :] = fila
            suma_post_iny[si, 1 + fi] = float(fila.sum())
            sumasq_post_iny[si, 1 + fi] = float(np.sum(fila ** 2))

    # ARREGLO 1, guardia de conservación (T6): la redistribución NUNCA debe cambiar
    # Σφ ni Σφ² más allá de error de punto flotante -- si esto falla, el motor aborta
    # (no se corre nada con una "redistribución" que en realidad crea/destruye energía).
    max_dev_suma = float(np.max(np.abs(suma_post_iny - suma_pre)))
    max_dev_sumasq = float(np.max(np.abs(sumasq_post_iny - sumasq_pre)))
    assert max_dev_suma < 1e-8 * (1.0 + float(np.max(np.abs(suma_pre)))), \
        f"ARREGLO 1 violado: Sigma phi cambio en la redistribucion (max_dev={max_dev_suma})"
    assert max_dev_sumasq < 1e-8 * (1.0 + float(np.max(np.abs(sumasq_pre)))), \
        f"ARREGLO 1 violado: Sigma phi^2 cambio en la redistribucion (max_dev={max_dev_sumasq})"

    lote_flat = lote.reshape(n_seeds * n_ramas, N_local)
    c0_flat = np.repeat(c0_vec, n_ramas)

    # phi crudo justo tras la redistribucion, ANTES de evolucionar mas (Arreglo 3, regla
    # "guardar detalle suficiente" -- permite recomputar cualquier definicion de X sobre
    # este punto sin volver a simular).
    phi_post_iny_flat = lote_flat.copy()

    X_boost_flat = batch_persistencia(lote_flat, c0_flat)
    Xh_boost_flat = batch_exergia_X_canonica(lote_flat)
    std_ratio_boost_flat = np.where(
        c0_flat > 0, np.std(lote_flat, axis=-1) / np.where(c0_flat > 0, c0_flat, 1.0), 0.0
    )

    checkpoints_pasos = sorted(set(int(round(f * post_pasos)) for f in checkpoints_frac))
    curva_X = {}
    curva_Xh = {}
    curva_std_ratio = {}
    t = 0
    for t_target in checkpoints_pasos:
        pasos = t_target - t
        for _ in range(pasos):
            lote_flat = batch_paso_difusion(lote_flat)
        t = t_target
        curva_X[t_target] = batch_persistencia(lote_flat, c0_flat)
        curva_Xh[t_target] = batch_exergia_X_canonica(lote_flat)
        curva_std_ratio[t_target] = np.where(
            c0_flat > 0, np.std(lote_flat, axis=-1) / np.where(c0_flat > 0, c0_flat, 1.0), 0.0
        )

    suma_final_flat = lote_flat.sum(axis=-1)

    return {
        "n_seeds": n_seeds, "n_ramas": n_ramas,
        "suma_pre": suma_pre, "suma_post_iny": suma_post_iny,
        "X_boost_flat": X_boost_flat, "Xh_boost_flat": Xh_boost_flat,
        "std_ratio_boost_flat": std_ratio_boost_flat,
        "curva_X": curva_X, "curva_Xh": curva_Xh, "curva_std_ratio": curva_std_ratio,
        "suma_final_flat": suma_final_flat,
        "checkpoints_pasos": checkpoints_pasos,
        "phi_post_iny_flat": phi_post_iny_flat,       # (n_seeds*n_ramas, N) crudo
        "phi_final_flat": lote_flat,                  # (n_seeds*n_ramas, N) crudo, ultimo checkpoint
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    print("[verificacion] comparando física vectorizada vs escalar (cs074_rcruz)...",
          file=sys.stderr, flush=True)
    chk = _verificar_equivalencia_con_base()
    print(f"[verificacion] OK max_diff_phi={chk['max_diff_phi']:.2e} "
          f"max_diff_X={chk['max_diff_X']:.2e}", file=sys.stderr, flush=True)

    chk_canon = _verificar_equivalencia_definicion_canonica()
    print(f"[verificacion ARREGLO 3] OK max_diff_Xcanonica={chk_canon['max_diff_Xcanonica']:.2e}",
          file=sys.stderr, flush=True)

    print("[calibracion] midiendo t_muerte (pasos a H=0 para X<THR_MUERTE sostenido)...",
          file=sys.stderr, flush=True)
    cal = calibrar_t_muerte()
    t_muerte_cal = cal["mediana"]
    print(f"[calibracion] t_muerte_cal(mediana)={t_muerte_cal} tiempos={cal['tiempos']} "
          f"todas_murieron={cal['todas_murieron']}", file=sys.stderr, flush=True)
    if t_muerte_cal <= 0:
        t_muerte_cal = CHECK_EVERY  # salvaguarda mínima, se reporta igual

    post_pasos = int(2 * t_muerte_cal)
    momento_steps_map = {mf: int(round(mf * t_muerte_cal)) for mf in MOMENTO_FACTORS}
    momento_steps_sorted = sorted(set(momento_steps_map.values()))
    print(f"[grid] post_pasos={post_pasos} momento_steps={momento_steps_map}",
          file=sys.stderr, flush=True)

    print("[fase A] evolucionando 16 semillas en lote hasta T_MAX...",
          file=sys.stderr, flush=True)
    tA0 = time.time()
    fa = fase_a_snapshots_batch(SEMILLAS, momento_steps_sorted)
    print(f"[fase A] hecha en {time.time()-tA0:.1f}s", file=sys.stderr, flush=True)

    filas_real = []
    filas_null = []

    for mi, mf in enumerate(MOMENTO_FACTORS):
        t_step = momento_steps_map[mf]
        tM0 = time.time()
        phi_muertos = fa["snapshots"][t_step]  # (n_seeds, N)
        res = fase_bc_momento(
            phi_muertos, fa["c0_vec"], SEMILLAS, mi, FRACCIONES, post_pasos,
            CHECKPOINTS_POST,
        )
        n_ramas = res["n_ramas"]
        for si, seed in enumerate(SEMILLAS):
            X_en_momento = fa["X_por_snapshot"][t_step][si]
            Xh_en_momento = fa["Xh_por_snapshot"][t_step][si]
            for rama in range(n_ramas):
                flat_idx = si * n_ramas + rama
                curva_X_row = {str(cp): float(res["curva_X"][cp][flat_idx])
                                for cp in res["checkpoints_pasos"]}
                curva_Xh_row = {str(cp): float(res["curva_Xh"][cp][flat_idx])
                                 for cp in res["checkpoints_pasos"]}
                curva_std_row = {str(cp): float(res["curva_std_ratio"][cp][flat_idx])
                                  for cp in res["checkpoints_pasos"]}
                fila_common = {
                    "seed": seed, "momento_factor": mf, "momento_step": t_step,
                    "X_en_momento": X_en_momento,
                    "Xh_en_momento": Xh_en_momento,
                    "X_boost": float(res["X_boost_flat"][flat_idx]),
                    "Xh_boost": float(res["Xh_boost_flat"][flat_idx]),
                    "std_ratio_boost": float(res["std_ratio_boost_flat"][flat_idx]),
                    "curva_X_post": curva_X_row,
                    "curva_Xh_post": curva_Xh_row,
                    "curva_std_ratio_post": curva_std_row,
                    "suma_pre_inyeccion": float(res["suma_pre"][si, rama]),
                    "suma_post_inyeccion": float(res["suma_post_iny"][si, rama]),
                    "suma_final": float(res["suma_final_flat"][flat_idx]),
                    "phi_post_iny": [round(float(x), 8) for x in res["phi_post_iny_flat"][flat_idx]],
                    "phi_final": [round(float(x), 8) for x in res["phi_final_flat"][flat_idx]],
                }
                if rama == 0:
                    filas_null.append(fila_common)
                else:
                    fila_common["fraccion_redistribuida"] = FRACCIONES[rama - 1]
                    filas_real.append(fila_common)
        print(f"[momento {mf} (step {t_step})] hecho en {time.time()-tM0:.1f}s "
              f"({mi+1}/{len(MOMENTO_FACTORS)})", file=sys.stderr, flush=True)

    elapsed = time.time() - t0

    result = {
        "experimento": "E5.5-3 — Reversibilidad de la muerte térmica",
        "N": N,
        "EPS_INICIAL": EPS_INICIAL,
        "H_MUERTE": H_MUERTE,
        "THR_MUERTE": THR_MUERTE,
        "verificacion_fisica_vectorizada": chk,
        "verificacion_definicion_canonica_ARREGLO_3": chk_canon,
        "definiciones_X": {
            "X (vieja, familia persistencia, heredada de E5.1-1)":
                "c*v, c=corr(phi,roll(phi,1)) clip>=0, v=Var(phi)/Var(phi0)",
            "Xh (nueva, canonica, Arreglo 3, de _observables_homologadas.py)":
                "(1/N) * sum_i (phi_i - 1)^2",
        },
        "calibracion_t_muerte": cal,
        "t_muerte_cal": t_muerte_cal,
        "post_pasos": post_pasos,
        "momento_factors": MOMENTO_FACTORS,
        "momento_steps_map": momento_steps_map,
        "fracciones_redistribuidas": FRACCIONES,
        "semillas": SEMILLAS,
        "checkpoints_post_frac": CHECKPOINTS_POST,
        "meta_por_semilla": [
            {"seed": s, "c0": float(fa["c0_vec"][i]), "suma0": fa["suma0_list"][i],
             "X_por_snapshot": {str(tt): fa["X_por_snapshot"][tt][i] for tt in momento_steps_sorted},
             "Xh_por_snapshot": {str(tt): fa["Xh_por_snapshot"][tt][i] for tt in momento_steps_sorted}}
            for i, s in enumerate(SEMILLAS)
        ],
        "filas_real": filas_real,
        "filas_null": filas_null,
        "elapsed_s": elapsed,
        "pre_inscrito_ARREGLO_1": {
            "pregunta_corregida": "¿se puede recuperar exergia redistribuyendo lo que "
                                   "YA hay en el sistema, SIN meter nada de afuera? "
                                   "(ver ADENDA en PROTOCOLO_E5.5-3_PREREGISTRO.md)",
            "prediccion": "el estado muerto es casi homogeneo (todos los phi_i ~ iguales) "
                           "-> reordenar valores casi identicos entre si deberia dar "
                           "revival ~0 en casi todo el rango de fraccion_redistribuida; "
                           "si aun asi X revive de forma clara y separada del NULL, es "
                           "hallazgo genuino, se reporta tal cual, no se descarta",
            "absorbente_si": "X_boost y X_post ~ NULL (~0) para todas las fracciones",
            "recuperable_si": "X_boost sube con fraccion_redistribuida, separado del "
                               "NULL, y decae de nuevo en Fase C con escala propia",
            "guardia_conservacion": "cada corrida verifica Sigma phi y Sigma phi^2 "
                                     "identicos (hasta 1e-8 relativo) antes/despues de "
                                     "la redistribucion -- el motor ABORTA si no (ver "
                                     "assert en fase_bc_momento)",
        },
    }

    out_json = HERE / "E5_5_3_resultado_crudo.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[filas_real] {len(filas_real)} [filas_null] {len(filas_null)}", file=sys.stderr)
    print(f"[elapsed] {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
