#!/usr/bin/env python3
"""
F4_6_control_dilucion_motor.py — BATERÍA DE FUNDAMENTOS, experimento F4-6

"Control de dilución: ρ∝a⁻ⁿ barrido + doble NULL"

Cross-check de F3-3: F3-3 barre n en ρ∝a⁻ⁿ midiendo el GRADIENTE físico. Aquí se
barre el MISMO n∈{1,2,3,4,5}, pero el observable declarado es la PERSISTENCIA de
la diferencia (forma×magnitud, receta de F1-1 generalizada a 2D), verificada
contra DOS NULLs independientes corriendo en paralelo sobre los mismos casos:

  1) NULL_BARAJADO_ACOPLE — permutación espacial del campo en cada checkpoint
     (destruye la forma/coherencia, conserva el histograma exacto). Calculado
     in-situ sobre los campos ya generados (5 permutaciones promediadas).
  2) NULL_DENSIDAD_FIJA — ρ≡ρ0, D≡D0 (sin dilución), idéntico al NULL_RHO_FIXED
     de CF2 y al n=0 de F3-3. No depende de n: se corre una vez por semilla.

Criterio de PASS congelado en PROTOCOLO_F4-6_PREREGISTRO.md (escrito y con mtime
ANTES de este script — verificar). Este motor NO se auto-adjudica el veredicto
de la batería: entrega números crudos (curvas completas por n, ambos NULLs,
dispersión entre semillas); la lectura final es de CS (Alexis).

Hereda sin retocar el sello físico de CF2_estiramiento_motor.py (Cosmogenesis-Web/
codigo/CF2_estiramiento/): L=64, W0=1.2, H_EXP=6.0, RHO0=1.0, D0=0.12, DT=0.25,
N_SUB=2, ORIGINAL_STEPS_PER_TG=399. No edita CF2_estiramiento_motor.py ni ningún
archivo de F3-3 ni de otro experimento de la batería.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Sello heredado (idéntico a CF2_estiramiento_motor.py; T1: no se
# retoca para favorecer el resultado)
# ============================================================
L = 64
H_EXP = 6.0
RHO0 = 1.0
D0 = 0.12
W0 = 1.2
DT = 0.25
N_SUB = 2
ORIGINAL_STEPS_PER_TG = 399

# ============================================================
# Barrido pre-registrado (PROTOCOLO_F4-6_PREREGISTRO.md, sección 3)
# ============================================================
A_GRID = np.geomspace(1.0, 1000.0, 7)   # misma grilla exacta que CF2 y F3-3
N_VALUES = [1, 2, 3, 4, 5]
SEEDS_STANDARD = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321,
                  271828, 161803]        # 10 de CF2 + 2 de F3-3 (comparabilidad)

# ============================================================
# Criterio de PASS pre-registrado (sección 6)
# ============================================================
Z_NULL_A = 3.0
PASS_RATE_MIN = 0.55
SINGULARITY_FACTOR = 3.0
SINGULARITY_FLOOR = 1e-4
N_PERM = 5

CODE_DIR = Path(__file__).resolve().parent
WEB_ROOT = CODE_DIR.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "BATERIA_FUNDAMENTOS" / "F4_6_control_dilucion"


def initial_T(L: int, w0: float, rng: np.random.Generator) -> np.ndarray:
    """Salto abrupto vertical: T≈1 a la izquierda, T≈0 a la derecha (frente plano en y).
    Idéntico a CF2/F3-3: perfil tanh + ruido gaussiano de amplitud 1e-4 en la IC."""
    x = np.arange(L) - (L - 1) / 2.0
    profile = 0.5 * (1.0 - np.tanh(x / w0))
    T = np.tile(profile, (L, 1))
    T = T + 1e-4 * rng.normal(size=T.shape)
    return np.clip(T, 0.0, 1.0)


def grad_metrics(T: np.ndarray, a: float) -> dict:
    """Observable secundario (verificación cruzada de método): gradiente físico,
    idéntico a grad_metrics() de CF2/F3-3. No decide el PASS de F4-6."""
    dTx = 0.5 * (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1))
    n = T.shape[1]
    band = slice(n // 8, 7 * n // 8)
    g = np.abs(dTx[:, band])
    A_comov = float(g.max()) if g.size else 0.0
    A_phys = A_comov / max(a, 1e-12)
    return {"A_comov": A_comov, "A_phys": A_phys}


def persistence_score(T: np.ndarray) -> dict:
    """Observable primario Π(T) = autocorr_nn(T) · Var(T) — forma × magnitud,
    receta de F1-1 generalizada a la grilla 2D nativa de este motor (periodic BC)."""
    Tc = T - T.mean()
    var = float(np.mean(Tc * Tc))
    if var < 1e-300:
        return {"pi": 0.0, "var": var, "autocorr_nn": 0.0}
    corr_x = float(np.mean(Tc * np.roll(Tc, -1, axis=1)))
    corr_y = float(np.mean(Tc * np.roll(Tc, -1, axis=0)))
    autocorr_nn = (corr_x + corr_y) / (2.0 * var)
    pi = autocorr_nn * var
    return {"pi": float(pi), "var": var, "autocorr_nn": float(autocorr_nn)}


def permuted_score(T: np.ndarray, rng: np.random.Generator, n_perm: int = N_PERM) -> dict:
    """NULL_BARAJADO_ACOPLE: n_perm permutaciones espaciales independientes de T,
    persistence_score de cada una, media y desvío reportados."""
    flat = T.ravel()
    scores = []
    for _ in range(n_perm):
        perm = rng.permutation(flat)
        Tp = perm.reshape(T.shape)
        scores.append(persistence_score(Tp)["pi"])
    scores = np.array(scores)
    return {"pi_mean": float(scores.mean()), "pi_std": float(scores.std())}


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


def run_trajectory(mode: str, seed: int, n_exp: int | None, a_grid: np.ndarray) -> dict:
    """
    Integra la difusión desde t_g=0 y muestrea checkpoints en t_g(a)=ln(a)/H_EXP.
    mode="REAL": rho=RHO0/a^n_exp, D=D0/a^n_exp (n_exp requerido).
    mode="NULL_DENSIDAD_FIJA": rho≡RHO0, D≡D0 (n_exp ignorado, no depende de n).
    En cada checkpoint calcula: persistence_score (Π), permuted_score (NULL_A,
    5 permutaciones, rng dedicado y determinístico por (seed,n_exp,checkpoint)),
    y grad_metrics (observable secundario A_phys).
    """
    rng_ic = np.random.default_rng(seed)
    T = initial_T(L, W0, rng_ic)

    # rng dedicado para permutaciones, independiente de la dinámica física,
    # derivado determinísticamente de (seed, n_exp) vía SeedSequence (T1: no
    # se elige a mano para favorecer ningún resultado).
    n_tag = n_exp if n_exp is not None else 999  # sentinel no-negativo para NULL_DENSIDAD_FIJA
    perm_rng = np.random.default_rng(np.random.SeedSequence([seed, n_tag, 777]))

    dtg = 1.0 / ORIGINAL_STEPS_PER_TG
    tg_targets = np.log(a_grid) / H_EXP
    tg_max = float(tg_targets[-1])
    n_steps = max(int(np.ceil(tg_max / dtg)), 1)

    checkpoints = []
    next_ckpt_idx = 0

    def record(tg_now, a_now):
        pi = persistence_score(T)
        pn = permuted_score(T, perm_rng, N_PERM)
        gm = grad_metrics(T, a_now)
        checkpoints.append({
            "a": float(a_now),
            "tg": float(tg_now),
            "pi_real": pi["pi"],
            "var": pi["var"],
            "autocorr_nn": pi["autocorr_nn"],
            "pi_null_a_mean": pn["pi_mean"],
            "pi_null_a_std": pn["pi_std"],
            "A_comov": gm["A_comov"],
            "A_phys": gm["A_phys"],
        })

    if tg_targets[0] <= 1e-15:
        record(0.0, float(a_grid[0]))
        next_ckpt_idx = 1

    for step in range(1, n_steps + 1):
        tg = step * dtg
        a = float(np.exp(H_EXP * tg))

        if mode == "NULL_DENSIDAD_FIJA":
            rho = RHO0
            D = D0
        else:  # REAL
            assert n_exp is not None
            rho = RHO0 / (a ** n_exp)
            D = D0 * (rho / RHO0)  # = D0 / a^n_exp

        T = diffuse(T, D, DT, N_SUB)
        T = np.clip(T, 0.0, 1.0)

        while next_ckpt_idx < len(tg_targets) and tg >= tg_targets[next_ckpt_idx] - 1e-9:
            record(tg, float(np.exp(H_EXP * tg_targets[next_ckpt_idx])))
            next_ckpt_idx += 1

    while next_ckpt_idx < len(tg_targets):
        a_last = float(a_grid[next_ckpt_idx])
        record(tg_targets[next_ckpt_idx], a_last)
        next_ckpt_idx += 1

    return {
        "mode": mode,
        "n_exp": n_exp,
        "seed": seed,
        "checkpoints": checkpoints,
    }


def evaluate(real_by_n: dict, null_b_by_seed: dict, seeds: list[int], n_values: list[int]) -> dict:
    """Aplica el criterio de PASS congelado (sección 6 del pre-registro) en a_final,
    y reporta las curvas completas por n para verificación (T5)."""
    a_final = None
    per_n = {}

    for n in n_values:
        seed_records = []
        gaps_at_final = []
        for seed in seeds:
            real_traj = real_by_n[n][seed]
            null_b_traj = null_b_by_seed[seed]
            ckpts_real = real_traj["checkpoints"]
            ckpts_nb = null_b_traj["checkpoints"]
            assert len(ckpts_real) == len(ckpts_nb)
            a_final = ckpts_real[-1]["a"]

            c_real = ckpts_real[-1]
            c_nb = ckpts_nb[-1]

            pi_real = c_real["pi_real"]
            pi_na_mean = c_real["pi_null_a_mean"]
            pi_na_std = c_real["pi_null_a_std"]
            pi_nb = c_nb["pi_real"]

            bite_A = bool(pi_real > pi_na_mean + Z_NULL_A * pi_na_std)
            bite_B = bool(pi_real > pi_nb)
            seed_pass = bool(bite_A and bite_B)
            gap = pi_real - pi_nb
            gaps_at_final.append(gap)

            # curva completa a-por-a (para reporte, T5 — no solo el punto final)
            curve = []
            for cr, cn in zip(ckpts_real, ckpts_nb):
                curve.append({
                    "a": cr["a"],
                    "pi_real": cr["pi_real"],
                    "pi_null_a_mean": cr["pi_null_a_mean"],
                    "pi_null_a_std": cr["pi_null_a_std"],
                    "pi_null_b": cn["pi_real"],
                    "A_phys_real": cr["A_phys"],
                    "A_phys_null_b": cn["A_phys"],
                    "bite_A": bool(cr["pi_real"] > cr["pi_null_a_mean"] + Z_NULL_A * cr["pi_null_a_std"]),
                    "bite_B": bool(cr["pi_real"] > cn["pi_real"]),
                })

            seed_records.append({
                "seed": seed,
                "pi_real_final": pi_real,
                "pi_null_a_mean_final": pi_na_mean,
                "pi_null_a_std_final": pi_na_std,
                "pi_null_b_final": pi_nb,
                "gap_final": gap,
                "bite_A": bite_A,
                "bite_B": bite_B,
                "seed_pass": seed_pass,
                "curve": curve,
            })

        n_pass = sum(1 for r in seed_records if r["seed_pass"])
        n_bite_A = sum(1 for r in seed_records if r["bite_A"])
        n_bite_B = sum(1 for r in seed_records if r["bite_B"])
        n_total = len(seeds)

        per_n[str(n)] = {
            "n": n,
            "seed_records": seed_records,
            "rate": n_pass / n_total if n_total else 0.0,
            "rate_A": n_bite_A / n_total if n_total else 0.0,
            "rate_B": n_bite_B / n_total if n_total else 0.0,
            "n_seeds_pass": n_pass,
            "n_seeds_total": n_total,
            "gap_mean": float(np.mean(gaps_at_final)),
            "gap_std": float(np.std(gaps_at_final)),
            "verdict": "F4_6_PASS" if (n_pass / n_total if n_total else 0.0) >= PASS_RATE_MIN else "F4_6_FAIL",
        }

    # curvatura discreta de gap(n) para chequeo de singularidad de n=3
    gap_by_n = {n: per_n[str(n)]["gap_mean"] for n in n_values}
    singularity = {}
    if all(k in gap_by_n for k in (2, 3, 4)):
        curv3 = gap_by_n[2] - 2 * gap_by_n[3] + gap_by_n[4]
        neighbor_curvs = []
        if all(k in gap_by_n for k in (1, 2, 3)):
            neighbor_curvs.append(abs(gap_by_n[1] - 2 * gap_by_n[2] + gap_by_n[3]))
        if all(k in gap_by_n for k in (3, 4, 5)):
            neighbor_curvs.append(abs(gap_by_n[3] - 2 * gap_by_n[4] + gap_by_n[5]))
        ref = max(neighbor_curvs + [SINGULARITY_FLOOR])
        is_singular = bool(abs(curv3) > SINGULARITY_FACTOR * ref)
        singularity = {
            "curv_n2": (gap_by_n[1] - 2 * gap_by_n[2] + gap_by_n[3]) if all(k in gap_by_n for k in (1, 2, 3)) else None,
            "curv_n3": curv3,
            "curv_n4": (gap_by_n[3] - 2 * gap_by_n[4] + gap_by_n[5]) if all(k in gap_by_n for k in (3, 4, 5)) else None,
            "reference_curvature": ref,
            "n3_singular": is_singular,
        }

    # monotonicidad de gap(n): ¿crece con n como predice la física (mayor n ⇒
    # dilución apaga antes la difusión ⇒ mayor persistencia relativa)?
    ns_sorted = sorted(gap_by_n.keys())
    gaps_sorted = [gap_by_n[n] for n in ns_sorted]
    mono_nondecreasing = all(
        gaps_sorted[i + 1] >= gaps_sorted[i] - 1e-9 for i in range(len(gaps_sorted) - 1)
    )

    return {
        "a_final": a_final,
        "per_n": per_n,
        "gap_by_n": gap_by_n,
        "singularity_n3": singularity,
        "gap_monotonic_nondecreasing_in_n": bool(mono_nondecreasing),
    }


def run_production(seeds: list[int], n_values: list[int], a_grid: np.ndarray, tag: str) -> dict:
    t0 = time.time()

    real_by_n: dict[int, dict[int, dict]] = {n: {} for n in n_values}
    for n in n_values:
        for seed in seeds:
            real_by_n[n][seed] = run_trajectory("REAL", seed, n, a_grid)

    null_b_by_seed: dict[int, dict] = {}
    for seed in seeds:
        null_b_by_seed[seed] = run_trajectory("NULL_DENSIDAD_FIJA", seed, None, a_grid)

    evaluation = evaluate(real_by_n, null_b_by_seed, seeds, n_values)

    payload = {
        "experimento": "F4-6 control de dilución (ρ∝a⁻ⁿ, doble NULL, observable=persistencia)",
        "tag": tag,
        "sello": {
            "L": L, "H_EXP": H_EXP, "RHO0": RHO0, "D0": D0, "W0": W0, "DT": DT,
            "N_SUB": N_SUB, "ORIGINAL_STEPS_PER_TG": ORIGINAL_STEPS_PER_TG,
        },
        "barrido": {
            "n_values": n_values,
            "a_grid": a_grid.tolist(),
            "seeds": seeds,
            "n_seeds": len(seeds),
        },
        "criterio_preregistrado": {
            "Z_NULL_A": Z_NULL_A,
            "PASS_RATE_MIN": PASS_RATE_MIN,
            "SINGULARITY_FACTOR": SINGULARITY_FACTOR,
            "SINGULARITY_FLOOR": SINGULARITY_FLOOR,
            "N_PERM": N_PERM,
            "descripcion": (
                "bite_A = pi_real(a_final) > pi_null_A_mean(a_final) + Z_NULL_A*pi_null_A_std(a_final); "
                "bite_B = pi_real(a_final) > pi_null_B(a_final); "
                "seed_pass(n,s) = bite_A AND bite_B; rate(n) = seeds_pass/n_seeds; "
                "PASS(n) si rate(n) >= PASS_RATE_MIN"
            ),
        },
        "trayectorias_real": {
            str(n): {str(seed): real_by_n[n][seed] for seed in seeds} for n in n_values
        },
        "trayectorias_null_densidad_fija": {str(seed): null_b_by_seed[seed] for seed in seeds},
        "evaluacion": evaluation,
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
        n_values = [1, 3, 5]
        a_grid = np.geomspace(1.0, 10.0, 3)
        tag = "smoke"
    else:
        seeds = SEEDS_STANDARD
        n_values = N_VALUES
        a_grid = A_GRID
        tag = "produccion"

    print(f"=== F4-6 control de dilución — modo={args.mode} ===")
    print(f"n_values={n_values}")
    print(f"seeds={seeds}")
    print(f"a_grid={a_grid.tolist()}")

    payload = run_production(seeds, n_values, a_grid, tag)

    print("\n=== RESUMEN CRUDO (sin adjudicar) ===")
    ev = payload["evaluacion"]
    for n in n_values:
        rec = ev["per_n"][str(n)]
        print(
            f"  n={n}  rate={rec['rate']:.3f} (A={rec['rate_A']:.3f} B={rec['rate_B']:.3f})  "
            f"gap_mean={rec['gap_mean']:.6f}  gap_std={rec['gap_std']:.6f}  verdict={rec['verdict']}"
        )
    print(f"\ngap_by_n = {ev['gap_by_n']}")
    print(f"gap monotónico no-decreciente en n: {ev['gap_monotonic_nondecreasing_in_n']}")
    print(f"singularidad n=3: {ev['singularity_n3']}")

    out_json = OUT_DIR / f"F4_6_control_dilucion_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")
    print(f"runtime_seconds={payload['runtime_seconds']:.2f}")


if __name__ == "__main__":
    main()
