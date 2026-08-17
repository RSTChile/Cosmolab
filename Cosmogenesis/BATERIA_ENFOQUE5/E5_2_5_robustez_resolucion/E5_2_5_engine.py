#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5_2_5_engine.py — Robustez del balance de energia a la resolucion (dt, N)
============================================================================
Ver E5_2_5_PROTOCOLO_PREREGISTRO.md (escrito y fechado ANTES de este motor,
T3) para la justificacion completa de cada eleccion. Resumen:

  - Modelo base: cs074_rcruz.py (Cosmogenesis/cs074_rcruz.py), NO editado.
    Se IMPORTA solo para (a) verificar bit-a-bit que nuestra generalizacion
    de la difusion a dt=0.5 reproduce exactamente paso_difusion original, y
    (b) reusar campo_inicial/paso_expansion con la MISMA logica (copiadas
    aqui verbatim, sin alterar la fisica, solo para no depender de un import
    fragil si el otro archivo cambia).

  - E_total(t) := suma(phi_i(t))  (primer momento del campo). Bajo difusion
    simetrica en el anillo COMPLETO (todas las aristas vivas) esta cantidad
    se conserva analiticamente (matriz doblemente estocastica) para
    cualquier dt -> es la cantidad conservada NATURAL del propio esquema,
    no una construccion ad-hoc. E1 (axioma del documento madre) declara que
    el presupuesto se conserva; aqui se pone a prueba con esa cantidad.

  - dt generaliza el coeficiente 0.5 hardcodeado en el paso de difusion
    original (nuevo = phi + dt*(media-phi); dt=0.5 -> identico al original).

  - H = min(r_target * D_medido(dt,N,eps), 1.0), exactamente la logica de
    cs074_rcruz, con D medido fresco en cada (dt,N,eps). r_target in {0,1}.

  - pasos(dt) = round(T_total/dt), T_total fijo (5.0) para que "refinar dt"
    compare la MISMA duracion fisica a mas resolucion (estudio de
    convergencia estandar).

Salida: JSON crudo con deriva_final y deriva_max por cada
(dt, N, seed, r_target, eps), mas agregados (mediana/percentiles por seed).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # Cosmogenesis/

# ---------------------------------------------------------------------------
# 1) Piezas del modelo base (copiadas verbatim de cs074_rcruz.py salvo la
#    generalizacion documentada de dt en la difusion). NO se edita el
#    original; se importa solo para el chequeo de equivalencia bit-a-bit.
# ---------------------------------------------------------------------------

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


def paso_difusion_dt(phi, activo, dt):
    """Generalizacion de paso_difusion (cs074_rcruz.py) con coeficiente dt
    explicito en vez del 0.5 hardcodeado. A dt=0.5 es identica bit-a-bit
    (verificado en verificar_equivalencia())."""
    left = np.roll(phi, 1)
    right = np.roll(phi, -1)
    e_left = np.roll(activo, 1)
    e_right = activo
    n_nb = e_left.astype(np.float64) + e_right.astype(np.float64)
    s = e_left * left + e_right * right
    media = np.divide(s, n_nb, out=phi.copy(), where=n_nb > 0)
    nuevo = phi + dt * (media - phi)
    return np.where(n_nb > 0, nuevo, phi)


def paso_expansion(activo, H, rng):
    """Identica a cs074_rcruz.paso_expansion (copiada verbatim)."""
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


def medir_D_dt(N, eps, dt, seed):
    """Analogo a cs074_rcruz.medir_D pero con paso_difusion_dt(dt)."""
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    c0 = phi.std()
    if c0 <= 0:
        return 0.0
    phi1 = paso_difusion_dt(phi, activo, dt)
    c1 = phi1.std()
    return max(0.0, float((c0 - c1) / c0))


def verificar_equivalencia():
    """T0: nuestra generalizacion a dt=0.5 debe reproducir bit-a-bit el
    paso_difusion ORIGINAL de cs074_rcruz.py (import de solo lectura)."""
    sys.path.insert(0, str(ROOT))
    import cs074_rcruz as base  # noqa: E402  (import diferido a proposito)

    rng = np.random.default_rng(12345)
    N = 64
    phi, _ = campo_inicial(N, 1e-2, rng)
    activo = np.ones(N, dtype=bool)
    # corta algunas aristas para probar el caso irregular tambien
    activo[3::7] = False
    a_orig = base.paso_difusion(phi.copy(), activo.copy())
    a_mio = paso_difusion_dt(phi.copy(), activo.copy(), 0.5)
    if not np.array_equal(a_orig, a_mio):
        maxdiff = float(np.max(np.abs(a_orig - a_mio)))
        raise SystemExit(
            f"[FALLO EQUIVALENCIA] paso_difusion_dt(dt=0.5) != paso_difusion original "
            f"(cs074_rcruz.py). max|diff|={maxdiff:.3e}. PARAR — no continuar."
        )
    print("[equivalencia] paso_difusion_dt(dt=0.5) == paso_difusion original (cs074_rcruz.py): OK bit-a-bit",
          file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 2) Corrida individual: mide deriva de E_total = sum(phi) a lo largo de la
#    trayectoria completa (T6: se verifica cada paso, no solo al final).
# ---------------------------------------------------------------------------

def corrida(N, eps, dt, r_target, T_total, seed):
    rng = np.random.default_rng(seed)
    phi, _ = campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)

    D = medir_D_dt(N, eps, dt, seed=90_000_000 + seed)
    if D > 0:
        H = float(min(r_target * D, 1.0))
    else:
        H = 0.0 if r_target == 0.0 else 1.0

    pasos = max(1, round(T_total / dt))
    E0 = float(phi.sum())
    denom = max(abs(E0), 1e-300)
    deriva_max = 0.0
    for _t in range(pasos):
        phi = paso_difusion_dt(phi, activo, dt)
        activo = paso_expansion(activo, H, rng)
        Et = float(phi.sum())
        d = abs(Et - E0) / denom
        if d > deriva_max:
            deriva_max = d
    Efin = float(phi.sum())
    deriva_final = abs(Efin - E0) / denom
    frac_exp = 1.0 - float(activo.mean())
    return {
        "D": D,
        "H": H,
        "pasos": pasos,
        "E0": E0,
        "Efin": Efin,
        "deriva_final": deriva_final,
        "deriva_max": deriva_max,
        "frac_exp_final": frac_exp,
    }


# ---------------------------------------------------------------------------
# 3) Barrido completo
# ---------------------------------------------------------------------------

DT_LIST = [1e-4, 10 ** -3.5, 1e-3, 10 ** -2.5, 1e-2, 10 ** -1.5, 1e-1]
N_LIST = [128, 256, 512, 1024, 2048]
SEEDS = list(range(8))  # seed = 2000 + s
R_CONDS = [("aislado", 0.0), ("expansion_r1", 1.0)]
EPS_CONDS = [0.0, 1e-2]
T_TOTAL = 5.0


def main():
    verificar_equivalencia()
    t0_wall = time.time()
    timestamp_inicio = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[inicio] {timestamp_inicio}", file=sys.stderr, flush=True)

    filas = []
    total_combos = len(DT_LIST) * len(N_LIST) * len(SEEDS) * len(R_CONDS) * len(EPS_CONDS)
    done = 0
    for eps in EPS_CONDS:
        for cond_name, r_target in R_CONDS:
            for N in N_LIST:
                for dt in DT_LIST:
                    for s in SEEDS:
                        seed = 2000 + s
                        r = corrida(N, eps, dt, r_target, T_TOTAL, seed)
                        filas.append({
                            "eps": eps,
                            "cond": cond_name,
                            "r_target": r_target,
                            "N": N,
                            "dt": dt,
                            "seed": seed,
                            **r,
                        })
                        done += 1
                    if done % 80 == 0 or done == total_combos:
                        elapsed = time.time() - t0_wall
                        print(f"[progreso] {done}/{total_combos} combos "
                              f"eps={eps} cond={cond_name} N={N} dt={dt:.2e} "
                              f"elapsed={elapsed:.1f}s", file=sys.stderr, flush=True)

    timestamp_fin = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    elapsed_total = time.time() - t0_wall

    result = {
        "experimento": "E5.2-5 robustez del balance a la resolucion",
        "preregistro": "E5_2_5_PROTOCOLO_PREREGISTRO.md",
        "modelo_base": "cs074_rcruz.py (no editado)",
        "E_total_def": "sum(phi_i(t))",
        "T_total": T_TOTAL,
        "dt_list": DT_LIST,
        "N_list": N_LIST,
        "seeds": [2000 + s for s in SEEDS],
        "r_conds": R_CONDS,
        "eps_conds": EPS_CONDS,
        "timestamp_inicio_utc": timestamp_inicio,
        "timestamp_fin_utc": timestamp_fin,
        "elapsed_s": elapsed_total,
        "filas": filas,
    }
    out_json = HERE / "E5_2_5_resultado.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[fin] {timestamp_fin}  elapsed={elapsed_total:.1f}s  filas={len(filas)}",
          file=sys.stderr, flush=True)
    print(f"[archivo] {out_json}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
