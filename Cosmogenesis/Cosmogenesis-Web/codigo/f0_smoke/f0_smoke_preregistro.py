"""
F0 smoke — preregistro Fase 5 (doc 14)
Solo tríada: a(t), T∝a^{-1}, ρ∝a^{-3}
PASS: |Ta/T0-1|<=5% y |ρ a^3/ρ0-1|<=5%
FAIL: >20%

Constantes selladas (no retocar para 1/1836).
Corrección respecto a Meta f0_triada_holistica.py:
  - T = T_amp / a  (no T /= a cada paso)
  - a(t) = exp(H_inf * t) continuo en el tramo
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --- sello F0 (unidades del modelo; mapeo K solo si se desea reporte) ---
CONST = {
    "w": 1.0 / 3.0,
    "H_inf": 1e35,  # s^-1 (misma familia Meta; tramo corto)
    "T0": 1e15,  # K
    "rho0": 1e30,  # kg/m^3
    "a0": 1.0,
    "L": 30,
    "dt_gen": 1e-38,  # s
    "pasos": 200,
    "eps": 1e-9,
    "seed": 2025,
    "pass_thr": 0.05,
    "fail_thr": 0.20,
}


def campo_T_amp(L: int, eps: float, rng: np.random.Generator) -> np.ndarray:
    xs = np.linspace(0, 1, L)
    xx, yy = np.meshgrid(xs, xs, indexing="xy")
    pert = np.zeros((L, L))
    for mx in range(1, 4):
        for my in range(1, 4):
            fase = rng.uniform(0, 2 * np.pi)
            pert += np.sin(2 * np.pi * (mx * xx + my * yy) + fase) / (mx + my)
    pert -= pert.mean()
    if pert.std() > 0:
        pert /= pert.std()
    return CONST["T0"] * (1.0 + eps * pert)


def run() -> dict:
    c = CONST
    rng = np.random.default_rng(c["seed"])
    T_amp = campo_T_amp(c["L"], c["eps"], rng)  # fijo: no se reescala mal
    rows = []
    for step in range(c["pasos"]):
        t_g = step * c["dt_gen"]
        a = c["a0"] * np.exp(c["H_inf"] * t_g)
        T = T_amp / a
        rho = c["rho0"] / (a**3)
        T_mean = float(T.mean())
        rho_mean = float(rho.mean()) if hasattr(rho, "mean") else float(rho)
        # rho is scalar here
        rho_mean = c["rho0"] / (a**3)
        Ta = T_mean * a
        rho_a3 = rho_mean * (a**3)
        rows.append(
            {
                "step": step,
                "t_g": t_g,
                "a": float(a),
                "T_mean": T_mean,
                "rho_mean": rho_mean,
                "Ta": float(Ta),
                "rho_a3": float(rho_a3),
                "Ta_over_T0_m1": float(Ta / c["T0"] - 1.0),
                "rho_a3_over_rho0_m1": float(rho_a3 / c["rho0"] - 1.0),
            }
        )

    Tas = np.array([r["Ta"] for r in rows])
    ra = np.array([r["rho_a3"] for r in rows])
    max_err_Ta = float(np.max(np.abs(Tas / c["T0"] - 1.0)))
    max_err_rho = float(np.max(np.abs(ra / c["rho0"] - 1.0)))

    if max_err_Ta <= c["pass_thr"] and max_err_rho <= c["pass_thr"]:
        verdict = "F0-PASS"
    elif max_err_Ta > c["fail_thr"] or max_err_rho > c["fail_thr"]:
        verdict = "F0-FAIL"
    else:
        verdict = "F0-MARGINAL"

    return {
        "verdict": verdict,
        "max_err_Ta": max_err_Ta,
        "max_err_rho_a3": max_err_rho,
        "constants": c,
        "samples": [rows[i] for i in range(0, c["pasos"], 40)] + [rows[-1]],
        "note": "Smoke dilucion solo; sin Euler/Poisson/clusters (preregistro §7)",
    }


def main() -> None:
    out = run()
    print("F0 smoke preregistro")
    print(f"verdict={out['verdict']}")
    print(f"max|Ta/T0-1|={out['max_err_Ta']:.6%}")
    print(f"max|rho a^3/rho0-1|={out['max_err_rho_a3']:.6%}")
    print("step | a | T | Ta/T0-1 | rho_a3/rho0-1")
    for r in out["samples"]:
        print(
            f"{r['step']:4d} | {r['a']:.6f} | {r['T_mean']:.4e} | "
            f"{r['Ta_over_T0_m1']:+.3e} | {r['rho_a3_over_rho0_m1']:+.3e}"
        )
    root = Path(__file__).resolve().parents[2]
    out_path = root / "results" / "f0_smoke_preregistro_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    # constants seal
    seal = root / "codigo" / "f0_smoke" / "f0_constants.json"
    seal.write_text(json.dumps(out["constants"], indent=2))
    print(f"wrote {out_path}")
    print(f"wrote {seal}")


if __name__ == "__main__":
    main()
