"""
kappaDelta_refino.py — κ_Δ: dos arreglos sobre el barrido B
===========================================================
1) La perturbación δ se inyecta AHORA en la precisión de destino. En el barrido B
   original ω se construía en float64 y se le sumaba δ ahí: cualquier δ < ulp(ω)≈1,55e−15
   se perdía ANTES de entrar al motor, así que float80 y mpmath heredaban el piso de
   float64. Eso invalidaba la guarda F1-3 para κ_Δ y se corrige acá.
2) Rejilla fina alrededor de la transición OPERABLE (que en el barrido grueso caía
   entre δ=0,32 y δ=1), y comparación contra la predicción estructural: el estado
   operable cambia cuando el δ acerca alguna pareja al cambio de SIGNO de la
   compatibilidad, o sea a distancia π/2 en fase → δ* = |ω_i−ω_j| más cercano a K/4=2.
"""
import json, math
from pathlib import Path
import numpy as np
from kappas_mortalidad_barrido import motor, motor_mp, K_FIRMA, PREC

R = Path(__file__).resolve().parent
SEMILLAS = list(range(1, 6))
K_STEPS = 250


def corrida(dtipo, seed, om):
    return motor(N=8, alpha=1.0, seed=seed, dtype=dtipo, omega_override=om,
                 k_max=K_STEPS, tau_cap=10**9)


def perturbar(om0, d, dt):
    """ω y δ construidos DIRECTAMENTE en la precisión de destino."""
    om = np.asarray(om0, dtype=dt).copy()
    om[0] = om[0] + dt(d)
    return om


def main():
    out = {"nota": __doc__.strip(), "distinguible": [], "operable": []}

    # ---- (1) piso de DISTINGUIBILIDAD por precisión, con δ inyectado bien ----
    deltas = [10.0 ** e for e in range(-30, 0)]
    for dt in (np.float64, np.longdouble):
        for seed in SEMILLAS:
            om0 = np.random.default_rng(seed).integers(0, K_FIRMA, size=8)
            base = corrida(dt, seed, np.asarray(om0, dtype=dt))
            Sb = np.asarray(base["S_final"], float)
            dmin = float("nan")
            for d in deltas:
                r = corrida(dt, seed, perturbar(om0, d, dt))
                Sr = np.asarray(r["S_final"], float)
                div = np.abs(Sb - Sr).max() / max(np.abs(Sb).max(), 1e-320)
                if div > 0:
                    dmin = d
                    break
            out["distinguible"].append({"precision": PREC[dt], "semilla": seed,
                                        "delta_min": dmin,
                                        "ulp_de_omega": float(np.spacing(np.asarray(float(om0[0]), dtype=dt)))})
    for dps in (30, 60):
        for seed in SEMILLAS[:3]:
            om0 = np.random.default_rng(seed).integers(0, K_FIRMA, size=8)
            base = motor_mp(N=8, alpha=1.0, seed=seed, k_max=K_STEPS, dps=dps,
                            omega_override=[float(o) for o in om0])
            Sb = np.asarray(base["S_final_norm"], float)
            dmin = float("nan")
            for d in deltas:
                # ω como string exacto: la suma la hace mpmath, no float64
                om = [float(o) for o in om0]
                base_om = om[0]
                r = motor_mp(N=8, alpha=1.0, seed=seed, k_max=K_STEPS, dps=dps,
                             omega_override=[base_om + d] + om[1:])
                Sr = np.asarray(r["S_final_norm"], float)
                if np.abs(Sb - Sr).max() > 0:
                    dmin = d
                    break
            out["distinguible"].append({"precision": f"mpmath_dps{dps}", "semilla": seed,
                                        "delta_min": dmin, "ulp_de_omega": 10.0 ** (-dps)})

    # ---- (2) piso OPERABLE, rejilla fina, y su predicción estructural ----
    finos = [round(x, 4) for x in np.arange(0.02, 2.001, 0.02)]
    for eps_tau in (1e-12, 1e-4, 1e-1):
        for seed in SEMILLAS:
            om0 = np.random.default_rng(seed).integers(0, K_FIRMA, size=8)
            base = motor(N=8, alpha=1.0, seed=seed, omega_override=om0.astype(float),
                         k_max=K_STEPS, tau_cap=10**9, EPS_TAU=eps_tau)
            eb = set(map(tuple, base["aristas"]))
            dmin = float("nan")
            for d in finos:
                om = om0.astype(float).copy(); om[0] += d
                r = motor(N=8, alpha=1.0, seed=seed, omega_override=om,
                          k_max=K_STEPS, tau_cap=10**9, EPS_TAU=eps_tau)
                if (set(map(tuple, r["aristas"])) != eb or r["n_vivos"] != base["n_vivos"]
                        or r["tau"] != base["tau"]):
                    dmin = d
                    break
            # predicción: distancia de ω_0 al cambio de signo de cos(2π(ω_0−ω_j)/K)
            dif = np.array([abs(om0[0] - om0[j]) % K_FIRMA for j in range(1, 8)], float)
            dist_signo = np.min(np.abs(np.minimum(dif, K_FIRMA - dif) - K_FIRMA / 4.0))
            out["operable"].append({"eps_tau": eps_tau, "semilla": seed,
                                    "delta_min_operable": dmin,
                                    "prediccion_dist_a_cambio_de_signo": float(dist_signo)})

    with (R / "kappaDelta_refino.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out["distinguible"], indent=1))
    print(json.dumps(out["operable"], indent=1))


if __name__ == "__main__":
    main()
