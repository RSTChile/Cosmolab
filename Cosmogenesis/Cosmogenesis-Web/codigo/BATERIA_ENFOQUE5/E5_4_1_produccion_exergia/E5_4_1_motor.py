#!/usr/bin/env python3
"""
E5_4_1_motor.py — BATERÍA ENFOQUE 5, experimento E5.4-1

"Producción de exergía vs enfriamiento medido, expansión de 1 a 1e4"

Pregunta (ver PROTOCOLO_E5.4-1_PREREGISTRO.md, congelado ANTES de este script,
T3): ¿la exergía (X) aparece porque el campo se enfría al expandirse? T(a) se
MIDE del propio estado del campo, nunca impuesta como fórmula cerrada de a
(T2). X se define como una estadística DISTINTA (varianza espacial) sobre el
mismo campo, no como función de T ni del criterio de PASS. Se prueba si X
correlaciona con la caída de T en REAL, y si esa correlación (y la propia
producción de X) desaparece sin expansión (NULL_SIN_EXPANSION).

Hereda el "sello" físico de CF2_estiramiento_motor.py (leído completo, NO
editado, NO importado — se copian aquí las mismas constantes y la misma
dinámica para no acoplar archivos ni tocar el original): grilla 2D periódica,
campo T(x,y)∈[0,1], condición inicial tanh, difusión Laplaciana por vecinos
(np.roll), dilución D(a) = D0·ρ(a)/ρ0 con ρ(a)=ρ0/a³ (axioma de diseño E2).

Extiende CF2 en lo que pide el pre-registro:
  1) a ∈ [1, 1e4] (CF2 llega solo a 1000).
  2) Ruido dinámico ε inyectado en CADA paso (T7 — perturbación además de
     semilla, no solo condición inicial).
  3) Dos observables medidos e INDEPENDIENTES entre sí: T_meas (energía de
     gradiente, "temperatura") y X (varianza espacial, "exergía") — en vez
     del único gradiente máximo de CF2.
  4) Segundo modo NULL_SIN_EXPANSION: congela TODA la expansión (ρ y D fijos
     todo el tiempo), no solo ρ como en CF2; mismo reloj genético que REAL
     para comparación en igualdad de condiciones.
  5) Diagnóstico de conservación de energía comóvil (T6): sum(T) debe
     conservarse exactamente bajo difusión pura (ε=0); con ε>0 el ruido rompe
     la conservación de forma medible y se reporta la deriva.

No edita CF2_estiramiento_motor.py ni ningún archivo fuera de esta carpeta.
No se auto-adjudica el veredicto de la batería: entrega crudo a CS. El
criterio de PASS está congelado en el PREREGISTRO y no se toca después de
correr (regla 7 de EJECUCIÓN).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Sello físico heredado de CF2_estiramiento_motor.py (idéntico, T1: no se
# retoca ningún número para favorecer el resultado)
# ============================================================
L = 64
H_EXP = 6.0
RHO0 = 1.0
D0 = 0.12
W0 = 1.2                  # ancho comóvil inicial del salto (celdas)
DT = 0.25                 # subpaso de difusión (idéntico a CF2)
N_SUB = 2                 # subiteraciones de difusión por paso de reloj genético
ORIGINAL_STEPS_PER_TG = 399

# ============================================================
# Barrido pre-registrado (PROTOCOLO_E5.4-1_PREREGISTRO.md, sección 4)
# ============================================================
A_GRID = np.geomspace(1.0, 1e4, 31)          # 31 pts, 4 décadas, pedido 1..1e4
EPS_GRID = [0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1, 1.0]   # 12 décadas + control 0
SEEDS = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 271828, 161803]  # >=12
MODES = ["REAL", "NULL_SIN_EXPANSION"]

# ============================================================
# Criterio de PASS pre-registrado (sección 7 del PREREGISTRO). Congelado.
# ============================================================
P1_MEDIAN_R_MIN = 0.6
P1_FRAC_R_GE_0P5_MIN = 0.7
P2_NULL_COLLAPSE_MAX = 0.05
P2_NULL_VS_REAL_MAX = 0.10
P3_MEDIAN_R_NULL_MAX = 0.3
P3_FRAC_UNDEFINED_MIN = 0.5
DEGENERATE_VAR_TOL = 1e-12

CODE_DIR = Path(__file__).resolve().parent
OUT_DIR = CODE_DIR / "results"


# ============================================================
# Física (idéntica a CF2, ver docstring)
# ============================================================
def initial_T(L: int, w0: float) -> np.ndarray:
    """Salto abrupto tipo tanh, mismo perfil que CF2_estiramiento_motor.py."""
    x = np.arange(L) - (L - 1) / 2.0
    profile = 0.5 * (1.0 - np.tanh(x / w0))
    return np.tile(profile, (L, 1))


def field_metrics(T: np.ndarray) -> dict:
    """
    Observables MEDIDOS del propio campo (sección 5 del PREREGISTRO):
      T_meas = energía de gradiente (promedio de (dT/dx)^2+(dT/dy)^2)
               -> juega el rol de "temperatura"/agitación.
      X      = varianza espacial de T -> juega el rol de "exergía"/estructura.
    Son dos estadísticas DISTINTAS del mismo array; ninguna se define en
    términos de la otra ni de a (T2).
    """
    dTx = 0.5 * (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1))
    dTy = 0.5 * (np.roll(T, -1, axis=0) - np.roll(T, 1, axis=0))
    T_meas = float(np.mean(dTx ** 2 + dTy ** 2))
    X = float(np.var(T))
    E_total = float(np.sum(T))
    return {"T_meas": T_meas, "X": X, "E_total": E_total}


def diffuse(T: np.ndarray, D: float, dt: float, n_sub: int) -> np.ndarray:
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


def run_trajectory(mode: str, seed: int, eps: float, a_grid: np.ndarray) -> dict:
    """
    Integra difusión (+ ruido dinámico ε en cada paso) desde t_g=0 y
    muestrea el campo EXACTAMENTE en los checkpoints t_g(a)=ln(a)/H_EXP.
    Misma lógica de checkpointing markoviano que CF2 (una sola trayectoria,
    equivalente a re-simular desde cero hasta cada t_g objetivo).
    """
    rng = np.random.default_rng(seed)
    T = initial_T(L, W0)
    T = T + 1e-4 * rng.normal(size=T.shape)
    T = np.clip(T, 0.0, 1.0)

    dtg = 1.0 / ORIGINAL_STEPS_PER_TG
    tg_targets = np.log(a_grid) / H_EXP
    tg_max = float(tg_targets[-1])
    n_steps = max(int(np.ceil(tg_max / dtg)), 1)

    checkpoints: list[dict] = []
    next_ckpt_idx = 0

    def record(tg_now, a_now):
        m = field_metrics(T)
        checkpoints.append({
            "a": float(a_now), "tg": float(tg_now),
            "T_meas": m["T_meas"], "X": m["X"], "E_total": m["E_total"],
        })

    if tg_targets[0] <= 1e-15:
        record(0.0, float(a_grid[0]))
        next_ckpt_idx = 1

    for step in range(1, n_steps + 1):
        tg = step * dtg
        a = float(np.exp(H_EXP * tg))

        if mode == "NULL_SIN_EXPANSION":
            rho = RHO0
            D = D0
        else:  # REAL
            rho = RHO0 / (a ** 3)
            D = D0 * (rho / RHO0)

        T = diffuse(T, D, DT, N_SUB)
        if eps > 0.0:
            T = T + eps * rng.normal(size=T.shape)
        T = np.clip(T, 0.0, 1.0)

        while next_ckpt_idx < len(tg_targets) and tg >= tg_targets[next_ckpt_idx] - 1e-9:
            record(tg, float(np.exp(H_EXP * tg_targets[next_ckpt_idx])))
            next_ckpt_idx += 1

    while next_ckpt_idx < len(tg_targets):
        a_last = float(a_grid[next_ckpt_idx])
        record(tg_targets[next_ckpt_idx], a_last)
        next_ckpt_idx += 1

    a_vals = np.array([c["a"] for c in checkpoints])
    T_meas = np.array([c["T_meas"] for c in checkpoints])
    X_comov = np.array([c["X"] for c in checkpoints])
    E_total = np.array([c["E_total"] for c in checkpoints])

    # Secundarios/diagnóstico: conversión de unidad física bajo expansión
    # (axioma E2), NO usados en el criterio de PASS (sección 5 PREREGISTRO).
    a_safe = np.clip(a_vals, 1e-12, None)
    T_phys = T_meas / (a_safe ** 2)
    X_phys = X_comov / (a_safe ** 4)

    return {
        "mode": mode, "seed": seed, "eps": eps,
        "a": a_vals.tolist(),
        "T_meas_comov": T_meas.tolist(),
        "X_comov": X_comov.tolist(),
        "T_phys": T_phys.tolist(),
        "X_phys": X_phys.tolist(),
        "E_total_comov": E_total.tolist(),
    }


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if np.var(x) < DEGENERATE_VAR_TOL or np.var(y) < DEGENERATE_VAR_TOL:
        return float("nan")
    c = np.corrcoef(x, y)
    return float(c[0, 1])


def evaluate_trajectory(traj: dict) -> dict:
    T_meas = np.array(traj["T_meas_comov"])
    X = np.array(traj["X_comov"])
    E_total = np.array(traj["E_total_comov"])
    r = pearson_r(X, T_meas)
    cooling_final = float(T_meas[0] - T_meas[-1])
    collapse_ratio = float(X[-1] / X[0]) if X[0] > 0 else float("nan")
    # T6: deriva de conservación de energía comóvil relativa al total inicial
    e0 = E_total[0] if E_total[0] != 0 else 1.0
    e_drift_max = float(np.max(np.abs(E_total - E_total[0])) / abs(e0))
    return {
        "r_X_Tmeas": r,
        "r_defined": bool(not np.isnan(r)),
        "cooling_final": cooling_final,
        "X_first": float(X[0]),
        "X_final": float(X[-1]),
        "collapse_ratio_X": collapse_ratio,
        "T_meas_first": float(T_meas[0]),
        "T_meas_final": float(T_meas[-1]),
        "E_drift_max_rel": e_drift_max,
    }


def nanmedian(vals):
    arr = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def run_production(seeds: list[int], eps_grid: list[float], a_grid: np.ndarray, tag: str) -> dict:
    t0 = time.time()
    trajectories = {}     # key "mode|seed|eps" -> {"traj":..., "eval":...}

    for mode in MODES:
        for seed in seeds:
            for eps in eps_grid:
                traj = run_trajectory(mode, seed, eps, a_grid)
                ev = evaluate_trajectory(traj)
                key = f"{mode}|{seed}|{eps}"
                trajectories[key] = {"traj": traj, "eval": ev}

    # ---- Agregados por modo (P1, P3) ----
    r_real = [trajectories[f"REAL|{s}|{e}"]["eval"]["r_X_Tmeas"] for s in seeds for e in eps_grid]
    r_null = [trajectories[f"NULL_SIN_EXPANSION|{s}|{e}"]["eval"]["r_X_Tmeas"] for s in seeds for e in eps_grid]

    r_real_defined = [r for r in r_real if not np.isnan(r)]
    r_null_defined = [r for r in r_null if not np.isnan(r)]

    median_abs_r_real = nanmedian([abs(r) for r in r_real_defined])
    frac_r_real_ge_0p5 = (
        float(np.mean([abs(r) >= 0.5 for r in r_real_defined])) if r_real_defined else 0.0
    )
    median_abs_r_null = nanmedian([abs(r) for r in r_null_defined])
    frac_null_undefined = float(np.mean([np.isnan(r) for r in r_null]))

    # ---- Agregados P2: colapso de X en NULL, y NULL vs REAL apareado ----
    collapse_null = [
        trajectories[f"NULL_SIN_EXPANSION|{s}|{e}"]["eval"]["collapse_ratio_X"]
        for s in seeds for e in eps_grid
    ]
    median_collapse_null = nanmedian(collapse_null)

    paired_ratio = []
    for s in seeds:
        for e in eps_grid:
            xn = trajectories[f"NULL_SIN_EXPANSION|{s}|{e}"]["eval"]["X_final"]
            xr = trajectories[f"REAL|{s}|{e}"]["eval"]["X_final"]
            if xr > 0:
                paired_ratio.append(xn / xr)
    median_null_vs_real = nanmedian(paired_ratio)

    # ---- Criterio de PASS congelado (sección 7 PREREGISTRO) ----
    P1 = bool(median_abs_r_real >= P1_MEDIAN_R_MIN and frac_r_real_ge_0p5 >= P1_FRAC_R_GE_0P5_MIN)
    P2 = bool(
        (not np.isnan(median_collapse_null) and median_collapse_null <= P2_NULL_COLLAPSE_MAX)
        and (not np.isnan(median_null_vs_real) and median_null_vs_real <= P2_NULL_VS_REAL_MAX)
    )
    P3 = bool(
        (not np.isnan(median_abs_r_null) and median_abs_r_null <= P3_MEDIAN_R_NULL_MAX)
        or (frac_null_undefined >= P3_FRAC_UNDEFINED_MIN)
    )
    verdict = "E5_4_1_PASS" if (P1 and P2 and P3) else "E5_4_1_FAIL"

    # ---- Diagnóstico T6: deriva de conservación por eps ----
    drift_by_eps = {}
    for e in eps_grid:
        drifts = [trajectories[f"{m}|{s}|{e}"]["eval"]["E_drift_max_rel"] for m in MODES for s in seeds]
        drift_by_eps[str(e)] = {"median": nanmedian(drifts), "max": float(np.max(drifts))}

    payload = {
        "experimento": "E5.4-1 Producción de exergía vs enfriamiento medido",
        "tag": tag,
        "sello": {
            "L": L, "H_EXP": H_EXP, "RHO0": RHO0, "D0": D0, "W0": W0,
            "DT": DT, "N_SUB": N_SUB, "ORIGINAL_STEPS_PER_TG": ORIGINAL_STEPS_PER_TG,
        },
        "barrido": {
            "a_grid": a_grid.tolist(),
            "eps_grid": eps_grid,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "modes": MODES,
            "n_trayectorias": len(trajectories),
        },
        "criterio_preregistrado": {
            "P1_MEDIAN_R_MIN": P1_MEDIAN_R_MIN,
            "P1_FRAC_R_GE_0P5_MIN": P1_FRAC_R_GE_0P5_MIN,
            "P2_NULL_COLLAPSE_MAX": P2_NULL_COLLAPSE_MAX,
            "P2_NULL_VS_REAL_MAX": P2_NULL_VS_REAL_MAX,
            "P3_MEDIAN_R_NULL_MAX": P3_MEDIAN_R_NULL_MAX,
            "P3_FRAC_UNDEFINED_MIN": P3_FRAC_UNDEFINED_MIN,
        },
        "resumen": {
            "median_abs_r_REAL": median_abs_r_real,
            "frac_r_REAL_ge_0.5": frac_r_real_ge_0p5,
            "n_r_REAL_defined": len(r_real_defined),
            "n_r_REAL_total": len(r_real),
            "median_abs_r_NULL": median_abs_r_null,
            "frac_r_NULL_undefined": frac_null_undefined,
            "n_r_NULL_defined": len(r_null_defined),
            "n_r_NULL_total": len(r_null),
            "median_collapse_ratio_X_NULL": median_collapse_null,
            "median_X_NULL_vs_X_REAL_final": median_null_vs_real,
            "P1_real_signal": P1,
            "P2_null_no_produce_X": P2,
            "P3_null_correlacion_debil_o_indefinida": P3,
            "verdict": verdict,
        },
        "diagnostico_conservacion_T6_drift_relativo_por_eps": drift_by_eps,
        "trayectorias": {k: {"traj": v["traj"], "eval": v["eval"]} for k, v in trajectories.items()},
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
        seeds = SEEDS[:3]
        eps_grid = [0.0, 1e-3, 1.0]
        a_grid = np.geomspace(1.0, 100.0, 7)
        tag = "smoke"
    else:
        seeds = SEEDS
        eps_grid = EPS_GRID
        a_grid = A_GRID
        tag = "produccion"

    print(f"=== E5.4-1 producción de exergía — modo={args.mode} ===")
    print(f"seeds={seeds}")
    print(f"eps_grid={eps_grid}")
    print(f"a_grid: {a_grid[0]:.3g} .. {a_grid[-1]:.3g} ({len(a_grid)} pts)")

    payload = run_production(seeds, eps_grid, a_grid, tag)

    r = payload["resumen"]
    print("\n=== RESUMEN CRUDO (sin adjudicar) ===")
    print(f"  median|r| REAL = {r['median_abs_r_REAL']:.4f}  (frac|r|>=0.5 = {r['frac_r_REAL_ge_0.5']:.3f}, "
          f"n_def={r['n_r_REAL_defined']}/{r['n_r_REAL_total']})")
    print(f"  median|r| NULL = {r['median_abs_r_NULL']:.4f}  (frac indefinido = {r['frac_r_NULL_undefined']:.3f}, "
          f"n_def={r['n_r_NULL_defined']}/{r['n_r_NULL_total']})")
    print(f"  median collapse_ratio_X NULL (X_final/X_first) = {r['median_collapse_ratio_X_NULL']:.6f}")
    print(f"  median X_NULL_final / X_REAL_final = {r['median_X_NULL_vs_X_REAL_final']:.6f}")
    print(f"  P1={r['P1_real_signal']}  P2={r['P2_null_no_produce_X']}  P3={r['P3_null_correlacion_debil_o_indefinida']}")
    print(f"  VERDICT = {r['verdict']}")
    print(f"\nruntime = {payload['runtime_seconds']:.2f} s")

    out_json = OUT_DIR / f"E5_4_1_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
