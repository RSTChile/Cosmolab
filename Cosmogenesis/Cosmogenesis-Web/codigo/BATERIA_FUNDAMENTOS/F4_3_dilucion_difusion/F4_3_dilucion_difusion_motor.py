#!/usr/bin/env python3
"""
F4_3_dilucion_difusion_motor.py — BATERÍA DE FUNDAMENTOS, Enfoque 4, experimento F4-3

"Dilución y reabsorción: ¿la caída de densidad apaga la difusión?"

Pregunta EXACTA del pre-registro (PROTOCOLO_F4-3_PREREGISTRO.md, congelado ANTES de este
archivo — verificar mtime): medir la difusividad EFECTIVA D como función de la densidad ρ
(alta, media, baja), sin tocar nada más del modelo — pregunta de instrumento/mecanismo pura.

Distinto de F4-2 (desacople expansión×dilución para persistencia) y de F3-3 (barrido del
exponente n de ρ∝a⁻ⁿ dentro del esquema de expansión de CF2): aquí NO hay expansión acoplada
(a≡1 fijo). ρ es la palanca única, barrida directamente. D_eff se MIDE desde la dinámica del
campo por DOS métodos independientes (T2: la cantidad medida ≠ la variable que la juzga) —
nunca se lee D0·ρ/ρ0 como si fuera el resultado.

Sustrato heredado SIN CAMBIO de CF2_estiramiento_motor.py (Cosmogenesis-Web/codigo/
CF2_estiramiento/CF2_estiramiento_motor.py): perfil inicial tanh + ruido, difusión isótropa de
4 vecinos por Euler explícito con N_SUB subiteraciones, L=64, W0=1.2, RHO0=1.0, D0=0.12,
DT=0.25, N_SUB=2. No se retoca ningún valor del sello (T1). No se edita CF2_estiramiento_motor.py
ni TEST_RHO_DISPERSION.py.

Este script NO se auto-adjudica el veredicto físico de la batería. Entrega números crudos
(D_eff por los dos métodos, R² de cada ajuste, correlación cruzada); la adjudicación final es
de CS (Alexis) después.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Sello heredado de CF2_estiramiento_motor.py (idéntico; T1: no se retoca)
# ============================================================
L = 64
RHO0 = 1.0
D0 = 0.12
W0 = 1.2            # ancho comóvil inicial del salto (celdas)
DT = 0.25            # paso "externo" (idéntico a CF2)
N_SUB = 2            # subiteraciones de difusión por paso externo (idéntico a CF2)

# ============================================================
# Barrido pre-registrado (PROTOCOLO_F4-3_PREREGISTRO.md, sección 3)
# ============================================================
RHO_RATIO_GRID = np.geomspace(1e-4, 2.0, 12)
SEEDS_STANDARD = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 271828, 161803]
N_STEPS_PRODUCCION = 2400
N_CKPT_PRODUCCION = 25

# ============================================================
# Constantes de método (sección 5)
# ============================================================
EIG_K1 = 2.0 - 2.0 * np.cos(2.0 * np.pi * 1.0 / L)   # autovalor EXACTO del operador discreto, k=1
AMPLITUDE_FLOOR = 1e-10          # piso de punto flotante para el ajuste espectral
MIN_FIT_POINTS = 3               # mínimo de checkpoints usables para un ajuste

# ============================================================
# Criterio de PASS pre-registrado (sección 6)
# ============================================================
R2_MIN = 0.8
SLOPE_MIN = 0.5
MONO_TOL = 1e-6
CORR_MIN = 0.8
MIN_RESOLVABLE_PTS = 4
PASS_RATE_MIN = 0.55

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "BATERIA_FUNDAMENTOS" / "F4_3_dilucion_difusion"


def initial_T(L: int, w0: float) -> np.ndarray:
    """Salto abrupto vertical: T≈1 a la izquierda, T≈0 a la derecha (frente plano en y).
    Idéntico a CF2_estiramiento_motor.py::initial_T (no se retoca, T1)."""
    x = np.arange(L) - (L - 1) / 2.0
    profile = 0.5 * (1.0 - np.tanh(x / w0))
    return np.tile(profile, (L, 1))


def diffuse(T: np.ndarray, D: float, dt: float, n_sub: int) -> np.ndarray:
    """Idéntica a CF2_estiramiento_motor.py::diffuse (no se retoca, T1)."""
    if D <= 0:
        return T
    out = T
    for _ in range(n_sub):
        lap = (
            np.roll(out, -1, 1)
            + np.roll(out, 1, 1)
            + np.roll(out, -1, 0)
            + np.roll(out, 1, 0)
            - 4.0 * out
        )
        out = out + (dt / n_sub) * D * lap
    return out


def band_slice(n: int) -> slice:
    return slice(n // 8, 7 * n // 8)


def profile_x(T: np.ndarray) -> np.ndarray:
    """perfil(x,t) = promedio sobre filas. Para el operador discreto de diffuse(), el término
    vertical (roll axis=0) se cancela EXACTAMENTE al promediar sobre una columna periódica
    completa (suma telescópica) -> perfil(x,t) evoluciona bajo difusión 1D pura con la MISMA D
    (no es una aproximación, ver PROTOCOLO_F4-3_PREREGISTRO.md sección 5)."""
    return T.mean(axis=0)


def spectral_amp_k1(profile: np.ndarray) -> float:
    centered = profile - profile.mean()
    coeffs = np.fft.rfft(centered)
    return float(np.abs(coeffs[1])) if len(coeffs) > 1 else 0.0


def peak_grad_band(profile: np.ndarray, L: int) -> float:
    dprof = 0.5 * (np.roll(profile, -1) - np.roll(profile, 1))
    band = band_slice(L)
    g = np.abs(dprof[band])
    return float(g.max()) if g.size else 0.0


def linfit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Regresión lineal simple y = slope*x + intercept. Devuelve (slope, intercept, R^2)."""
    if len(x) < 2:
        return 0.0, 0.0, 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res < 1e-30 else 0.0)
    return float(slope), float(intercept), float(r2)


def run_rho_seed(rho_ratio: float, seed: int, n_steps: int, n_ckpt: int) -> dict:
    """Corre la difusión a densidad FIJA rho_ratio (sin expansión, a≡1) y devuelve las series
    temporales de los dos observables (amplitud espectral k=1, ancho de frente) en checkpoints
    equiespaciados."""
    rng = np.random.default_rng(seed)
    T = initial_T(L, W0)
    T = T + 1e-4 * rng.normal(size=T.shape)
    T = np.clip(T, 0.0, 1.0)

    D = D0 * rho_ratio

    ckpt_steps = sorted(set(np.linspace(0, n_steps, n_ckpt).round().astype(int).tolist()))

    times = []
    a1_vals = []
    peakgrad_vals = []
    width_vals = []

    def record(step_idx: int):
        t = step_idx * DT
        prof = profile_x(T)
        a1 = spectral_amp_k1(prof)
        pg = peak_grad_band(prof, L)
        w = 0.5 / pg if pg > 0 else float("inf")
        times.append(float(t))
        a1_vals.append(a1)
        peakgrad_vals.append(pg)
        width_vals.append(w)

    ckpt_set = set(ckpt_steps)
    if 0 in ckpt_set:
        record(0)

    for step in range(1, n_steps + 1):
        T = diffuse(T, D, DT, N_SUB)
        T = np.clip(T, 0.0, 1.0)
        if step in ckpt_set:
            record(step)

    return {
        "rho_ratio": float(rho_ratio),
        "D_input": float(D),
        "seed": seed,
        "t": times,
        "A1": a1_vals,
        "peak_grad": peakgrad_vals,
        "width": width_vals,
    }


def fit_spectral(series: dict) -> dict:
    t = np.array(series["t"])
    a1 = np.array(series["A1"])
    mask = a1 > AMPLITUDE_FLOOR
    n_used = int(mask.sum())
    if n_used < MIN_FIT_POINTS:
        return {"n_used": n_used, "slope": None, "r2": None, "D_eff": None, "resoluble": False}
    y = np.log(a1[mask])
    x = t[mask]
    slope, _intercept, r2 = linfit_r2(x, y)
    D_eff = -slope / EIG_K1
    resoluble = (r2 >= R2_MIN) and (n_used >= MIN_FIT_POINTS)
    return {"n_used": n_used, "slope": slope, "r2": r2, "D_eff": D_eff, "resoluble": bool(resoluble)}


def fit_front(series: dict) -> dict:
    t = np.array(series["t"])
    w = np.array(series["width"])
    mask = np.isfinite(w) & (w > 0)
    n_used = int(mask.sum())
    if n_used < MIN_FIT_POINTS:
        return {"n_used": n_used, "slope": None, "r2": None, "D_eff": None, "resoluble": False}
    y = (w[mask]) ** 2
    x = t[mask]
    slope, _intercept, r2 = linfit_r2(x, y)
    D_eff = slope / 2.0
    resoluble = (r2 >= R2_MIN) and (n_used >= MIN_FIT_POINTS)
    return {"n_used": n_used, "slope": slope, "r2": r2, "D_eff": D_eff, "resoluble": bool(resoluble)}


def evaluate_seed(rho_grid: np.ndarray, per_rho: list[dict]) -> dict:
    """per_rho: lista ordenada por rho_grid creciente, cada elemento con fit_spectral/fit_front."""
    idx_res_both = [
        i for i, rec in enumerate(per_rho)
        if rec["spectral"]["resoluble"] and rec["front"]["resoluble"]
    ]

    if len(idx_res_both) < MIN_RESOLVABLE_PTS:
        return {
            "n_resolvable": len(idx_res_both),
            "seed_pass": None,
            "reason": "sin_rango_resoluble",
        }

    rho_res = rho_grid[idx_res_both]
    D_spec = np.array([per_rho[i]["spectral"]["D_eff"] for i in idx_res_both])
    D_front = np.array([per_rho[i]["front"]["D_eff"] for i in idx_res_both])

    # cond1: pendiente log-log positiva y >= SLOPE_MIN en AMBOS métodos
    valid_spec = D_spec > 0
    valid_front = D_front > 0
    if valid_spec.sum() >= 2:
        slope_spec, _, _ = linfit_r2(np.log(rho_res[valid_spec]), np.log(D_spec[valid_spec]))
    else:
        slope_spec = float("nan")
    if valid_front.sum() >= 2:
        slope_front, _, _ = linfit_r2(np.log(rho_res[valid_front]), np.log(D_front[valid_front]))
    else:
        slope_front = float("nan")

    cond1 = bool(
        np.isfinite(slope_spec) and np.isfinite(slope_front)
        and slope_spec >= SLOPE_MIN and slope_front >= SLOPE_MIN
    )

    # cond2: D_eff_spectral no decreciente en rho (dentro de MONO_TOL)
    mono = True
    for i in range(len(D_spec) - 1):
        if D_spec[i + 1] < D_spec[i] * (1.0 - MONO_TOL) - 1e-15:
            mono = False
            break
    cond2 = bool(mono)

    # cond3: correlacion de Pearson log-log entre los dos metodos
    both_valid = valid_spec & valid_front
    if both_valid.sum() >= 3:
        corr = float(np.corrcoef(np.log(D_spec[both_valid]), np.log(D_front[both_valid]))[0, 1])
    else:
        corr = float("nan")
    cond3 = bool(np.isfinite(corr) and corr >= CORR_MIN)

    seed_pass = bool(cond1 and cond2 and cond3)

    return {
        "n_resolvable": len(idx_res_both),
        "idx_resolvable": idx_res_both,
        "slope_loglog_spectral": None if not np.isfinite(slope_spec) else float(slope_spec),
        "slope_loglog_front": None if not np.isfinite(slope_front) else float(slope_front),
        "cond1_slope_min": cond1,
        "cond2_monotonic": cond2,
        "cond3_corr_cross_method": None if not np.isfinite(corr) else float(corr),
        "cond3_pass": cond3,
        "seed_pass": seed_pass,
        "reason": None,
    }


def sanity_control_D0(seed: int, n_steps: int, n_ckpt: int) -> dict:
    """Control de cordura del arnés (sección 4): D=0 exacto debe dejar el campo BIT-A-BIT
    idéntico en todos los checkpoints, porque diffuse() hace return T sin modificar nada."""
    series = run_rho_seed(0.0, seed, n_steps, n_ckpt)
    a1 = np.array(series["A1"])
    w = np.array(series["width"])
    a1_identical = bool(np.allclose(a1, a1[0], rtol=0, atol=0))
    w_identical = bool(np.allclose(w, w[0], rtol=0, atol=0))
    return {
        "seed": seed,
        "a1_bit_identical_all_ckpt": a1_identical,
        "width_bit_identical_all_ckpt": w_identical,
        "a1_first": float(a1[0]) if len(a1) else None,
        "width_first": float(w[0]) if len(w) else None,
        "harness_ok": bool(a1_identical and w_identical),
    }


def run_production(seeds: list[int], rho_grid: np.ndarray, n_steps: int, n_ckpt: int, tag: str) -> dict:
    t0 = time.time()

    # --- control de cordura del arnés (sección 4), 1 semilla ---
    control = sanity_control_D0(seeds[0], n_steps, n_ckpt)

    resultados_por_semilla = {}
    n_pass = 0
    n_fail = 0
    n_sin_rango = 0

    for seed in seeds:
        per_rho = []
        for rho_ratio in rho_grid:
            series = run_rho_seed(float(rho_ratio), seed, n_steps, n_ckpt)
            spec = fit_spectral(series)
            front = fit_front(series)
            per_rho.append({
                "rho_ratio": float(rho_ratio),
                "D_input": series["D_input"],
                "series": series,
                "spectral": spec,
                "front": front,
            })
        ev = evaluate_seed(rho_grid, per_rho)
        resultados_por_semilla[str(seed)] = {
            "per_rho": per_rho,
            "evaluation": ev,
        }
        if ev["seed_pass"] is True:
            n_pass += 1
        elif ev["seed_pass"] is False:
            n_fail += 1
        else:
            n_sin_rango += 1

    n_validas = n_pass + n_fail
    if n_validas == 0:
        rate = None
        verdict = "F4_3_SIN_SEÑAL_MEDIBLE"
    else:
        rate = n_pass / n_validas
        verdict = "F4_3_D_CAE_CON_RHO" if rate >= PASS_RATE_MIN else "F4_3_D_INDEPENDIENTE_O_NO_CONCLUYENTE"

    payload = {
        "experimento": "F4-3 dilución y reabsorción (D_eff vs rho, instrumento)",
        "tag": tag,
        "sello": {
            "L": L, "RHO0": RHO0, "D0": D0, "W0": W0, "DT": DT, "N_SUB": N_SUB,
            "EIG_K1": EIG_K1,
        },
        "barrido": {
            "rho_ratio_grid": rho_grid.tolist(),
            "seeds": seeds,
            "n_seeds": len(seeds),
            "n_steps": n_steps,
            "n_ckpt": n_ckpt,
            "T_total": n_steps * DT,
        },
        "criterio_preregistrado": {
            "R2_MIN": R2_MIN,
            "SLOPE_MIN": SLOPE_MIN,
            "MONO_TOL": MONO_TOL,
            "CORR_MIN": CORR_MIN,
            "MIN_RESOLVABLE_PTS": MIN_RESOLVABLE_PTS,
            "PASS_RATE_MIN": PASS_RATE_MIN,
            "AMPLITUDE_FLOOR": AMPLITUDE_FLOOR,
        },
        "control_cordura_D0": control,
        "resultados_por_semilla": resultados_por_semilla,
        "resumen": {
            "n_seeds_pass": n_pass,
            "n_seeds_fail": n_fail,
            "n_seeds_sin_rango_resoluble": n_sin_rango,
            "n_seeds_validas": n_validas,
            "rate": rate,
            "verdict": verdict,
        },
        "runtime_seconds": time.time() - t0,
        "generated_at_unix": time.time(),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "produccion"], default="produccion", nargs="?")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        seeds = SEEDS_STANDARD[:3]
        rho_grid = np.geomspace(1e-3, 1.0, 5)
        n_steps = 200
        n_ckpt = 10
        tag = "smoke"
    else:
        seeds = SEEDS_STANDARD
        rho_grid = RHO_RATIO_GRID
        n_steps = N_STEPS_PRODUCCION
        n_ckpt = N_CKPT_PRODUCCION
        tag = "produccion"

    print(f"=== F4-3 dilución y reabsorción — modo={args.mode} ===")
    print(f"seeds={seeds}")
    print(f"rho_ratio_grid={rho_grid.tolist()}")
    print(f"n_steps={n_steps} n_ckpt={n_ckpt} T_total={n_steps*DT}")

    payload = run_production(seeds, rho_grid, n_steps, n_ckpt, tag)

    print("\n=== CONTROL DE CORDURA D=0 ===")
    print(payload["control_cordura_D0"])

    print("\n=== RESUMEN CRUDO POR SEMILLA (sin adjudicar) ===")
    for seed_str, rec in payload["resultados_por_semilla"].items():
        ev = rec["evaluation"]
        if ev["seed_pass"] is None:
            print(f"  seed={seed_str:>6}  SIN_RANGO_RESOLUBLE (n_resolvable={ev['n_resolvable']})")
        else:
            print(
                f"  seed={seed_str:>6}  n_resolvable={ev['n_resolvable']:>2}  "
                f"slope_spec={ev['slope_loglog_spectral']}  slope_front={ev['slope_loglog_front']}  "
                f"corr={ev['cond3_corr_cross_method']}  seed_pass={ev['seed_pass']}"
            )

    resumen = payload["resumen"]
    print(f"\nn_pass={resumen['n_seeds_pass']} n_fail={resumen['n_seeds_fail']} "
          f"n_sin_rango={resumen['n_seeds_sin_rango_resoluble']} n_validas={resumen['n_seeds_validas']}")
    print(f"rate={resumen['rate']}  verdict={resumen['verdict']}")
    print(f"(umbral pre-registrado PASS_RATE_MIN={PASS_RATE_MIN})")

    # JSON crudo completo (auditoría en disco, sección 5/regla 3) sin las series completas de
    # A1/width por checkpoint sería insuficiente para auditar el ajuste -> se guardan.
    out_json = OUT_DIR / f"F4_3_dilucion_difusion_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")
    print(f"runtime_seconds={payload['runtime_seconds']:.2f}")


if __name__ == "__main__":
    main()
