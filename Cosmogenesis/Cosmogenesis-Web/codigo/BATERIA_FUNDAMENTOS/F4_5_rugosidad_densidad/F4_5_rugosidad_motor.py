#!/usr/bin/env python3
"""
F4_5_rugosidad_motor.py — BATERIA_FUNDAMENTOS, Enfoque 4, experimento F4-5

"Rugosidad de densidad: ¿importa el gradiente de ρ, no solo su valor medio?"

QUE ES ESTO (para retomar sin releer todo):
  Siembra un campo de densidad rho(x,y) con ESTRUCTURA ESPACIAL (parches
  suaves, "rugosidad") de amplitud barrida, deja difundir un campo T
  (la "diferencia") con difusividad local D(x,y) = D0*rho(x,y)/RHO0, y
  mide si la PERSISTENCIA final de T por zona se correlaciona con la
  densidad de esa zona -- comparando siempre contra un NULL donde la
  MISMA rho (mismo histograma exacto) se barajó celda a celda, destruyendo
  la estructura espacial pero conservando la estadística puntual.

  Protocolo pre-registrado y CONGELADO antes de este archivo:
  PROTOCOLO_F4-5_PREREGISTRO.md (leer primero -- fija observables, barrido,
  NULL y criterio de PASS; no se toca después de correr, T3).

  Motor 100% autocontenido: NO importa ni edita CF2_estiramiento_motor.py
  ni CF4_confinamiento.py ni ningún otro archivo de la batería (aunque
  reutiliza convenciones de estilo: reloj genético normalizado tg, a(tg)
  = exp(H*tg), fronteras periódicas via np.roll, semillas estándar del
  proyecto).

  Este script NO se auto-adjudica el veredicto de la batería. Entrega
  números crudos (incluidos los 64 valores de rho_zona/persistencia por
  corrida, para auditoría en disco por quien no escribió el código); la
  adjudicación final es de CS/director.

  Convención de carpeta: F4_5_* en
  codigo/BATERIA_FUNDAMENTOS/F4_5_rugosidad_densidad/, resultados en
  results/BATERIA_FUNDAMENTOS/F4_5_rugosidad_densidad/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Rutas
# ============================================================
HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parents[2]  # .../Cosmogenesis-Web
OUT_DIR = WEB_ROOT / "results" / "BATERIA_FUNDAMENTOS" / "F4_5_rugosidad_densidad"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROTOCOL_ID = "F4-5_RUGOSIDAD_DENSIDAD_2026-07-24"

# ============================================================
# Parametros fijos (sec. 2.6 del protocolo) -- produccion
# ============================================================
L_PROD = 64
B_PROD = 8                 # tamano de bloque/zona (LxL -> (L/B)x(L/B) zonas)
RHO0 = 1.0
D0 = 0.12
D_FLOOR_RATIO = 0.05
SIGMA_SUAVIZADO = 3.0
T_NOISE_STD = 0.10
SIGMA_DINAMICO = 0.01
TG_MAX = 1.0
N_STEPS_PROD = 240
N_SUB = 2
DT_SUB = 0.125

# ============================================================
# Barrido pre-registrado (sec. 4)
# ============================================================
AMPLITUD_GRID = (0.0, 0.05, 0.15, 0.30, 0.60, 1.00, 1.80, 3.00)
R_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
SEEDS_STANDARD = (7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 161803, 271828)
MODES = ("REAL", "NULL_BARAJADO")

# ============================================================
# Smoke (sec. 7.1)
# ============================================================
L_SMOKE = 32
B_SMOKE = 8
N_STEPS_SMOKE = 60
AMPLITUD_SMOKE = (0.0, 0.3, 3.0)
R_SMOKE = (0.3, 3.0)
SEEDS_SMOKE = (7, 42)

# ============================================================
# Criterio de PASS pre-registrado (sec. 6)
# ============================================================
MARGEN_CORR = 0.10
MARGEN_BRECHA = 0.01
PASS_RATE_MIN = 0.55


# ------------------------------------------------------------
# Generacion de campos
# ------------------------------------------------------------
def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def smooth_periodic(field: np.ndarray, sigma: float) -> np.ndarray:
    """Suavizado gaussiano separable con frontera periodica (np.roll), sin
    dependencias externas (scipy no se asume disponible)."""
    k = _gaussian_kernel_1d(sigma)
    radius = (len(k) - 1) // 2
    out = np.zeros_like(field)
    # eje x (columnas)
    for i, w in enumerate(k):
        shift = i - radius
        out += w * np.roll(field, shift, axis=1)
    tmp = out
    out = np.zeros_like(field)
    # eje y (filas)
    for i, w in enumerate(k):
        shift = i - radius
        out += w * np.roll(tmp, shift, axis=0)
    return out


def make_rho_real(L: int, amplitud: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """rho REAL: ruido blanco suavizado (parches espacialmente correlacionados),
    normalizado a media 0 / std 1, luego escalado por amplitud_rugosidad
    alrededor de RHO0. amplitud=0 -> rho uniforme por construccion."""
    if amplitud <= 0.0:
        return np.full((L, L), RHO0, dtype=float)
    w = rng.normal(size=(L, L))
    w_smooth = smooth_periodic(w, sigma)
    w_smooth = (w_smooth - w_smooth.mean()) / (w_smooth.std() + 1e-12)
    rho = RHO0 * (1.0 + amplitud * w_smooth)
    rho = np.clip(rho, RHO0 * D_FLOOR_RATIO, None)
    return rho


def shuffle_field(field: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """NULL: misma matriz de valores, posiciones permutadas al azar
    (mismo histograma exacto, estructura espacial destruida)."""
    flat = field.flatten()
    perm = rng.permutation(flat.size)
    return flat[perm].reshape(field.shape)


# ------------------------------------------------------------
# Dinamica: difusion con D espacialmente variable (volumenes finitos,
# D promediada en cada cara, conservativo, frontera periodica)
# ------------------------------------------------------------
def diffuse_step(T: np.ndarray, D: np.ndarray, dt: float, sigma_dinamico: float,
                  rng: np.random.Generator) -> np.ndarray:
    D_right = 0.5 * (D + np.roll(D, -1, axis=1))
    D_down = 0.5 * (D + np.roll(D, -1, axis=0))
    flux_x = D_right * (np.roll(T, -1, axis=1) - T)
    flux_y = D_down * (np.roll(T, -1, axis=0) - T)
    div = flux_x - np.roll(flux_x, 1, axis=1) + flux_y - np.roll(flux_y, 1, axis=0)
    T_new = T + dt * div
    if sigma_dinamico > 0:
        T_new = T_new + sigma_dinamico * np.sqrt(dt) * rng.normal(size=T.shape)
    return T_new


# ------------------------------------------------------------
# Corrida individual
# ------------------------------------------------------------
def run_single(mode: str, amplitud: float, r_hd: float, seed: int,
                L: int, B: int, n_steps: int) -> dict:
    """
    Devuelve rho_zona[] y persistencia[] (L/B x L/B zonas, aplanado) para
    UNA combinacion (modo, amplitud_rugosidad, r, seed).
    """
    rng_rho = np.random.default_rng(seed)
    rng_T = np.random.default_rng(seed + 10_000_000)

    rho_init_real = make_rho_real(L, amplitud, SIGMA_SUAVIZADO, rng_rho)
    if mode == "NULL_BARAJADO":
        rho_init = shuffle_field(rho_init_real, rng_rho)
    else:
        rho_init = rho_init_real

    T = T_NOISE_STD * rng_T.normal(size=(L, L))

    D0_local = D0
    H = r_hd * D0_local
    dtg = TG_MAX / n_steps

    for step in range(1, n_steps + 1):
        tg = step * dtg
        a = float(np.exp(H * tg))
        rho_t = rho_init / (a ** 3)
        D = D0_local * np.clip(rho_t / RHO0, D_FLOOR_RATIO, None)
        for _ in range(N_SUB):
            T = diffuse_step(T, D, DT_SUB / N_SUB, SIGMA_DINAMICO, rng_T)

    # agregacion por zona B x B
    nz = L // B
    rho_zona = rho_init.reshape(nz, B, nz, B).mean(axis=(1, 3)).flatten()
    persistencia = np.abs(T).reshape(nz, B, nz, B).mean(axis=(1, 3)).flatten()

    return {
        "rho_zona": rho_zona.tolist(),
        "persistencia": persistencia.tolist(),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def quartile_gap(rho_zona: np.ndarray, persistencia: np.ndarray) -> float:
    n = len(rho_zona)
    q = max(n // 4, 1)
    order = np.argsort(rho_zona)
    diluted_idx = order[:q]      # menor rho = mas diluido
    dense_idx = order[-q:]       # mayor rho = mas denso
    return float(persistencia[diluted_idx].mean() - persistencia[dense_idx].mean())


def evaluate_point(real: dict, null: dict) -> dict:
    rho_r = np.array(real["rho_zona"])
    per_r = np.array(real["persistencia"])
    rho_n = np.array(null["rho_zona"])
    per_n = np.array(null["persistencia"])

    corr_real = spearman(rho_r, per_r)
    corr_null = spearman(rho_n, per_n)
    gap_real = quartile_gap(rho_r, per_r)
    gap_null = quartile_gap(rho_n, per_n)

    sign_corr_ok = (not np.isnan(corr_real)) and (corr_real < 0)
    sign_gap_ok = gap_real > 0
    corr_margin_ok = (not np.isnan(corr_real)) and (not np.isnan(corr_null)) and \
        (abs(corr_real) > abs(corr_null) + MARGEN_CORR)
    gap_margin_ok = gap_real > gap_null + MARGEN_BRECHA

    seed_bite = bool(sign_corr_ok and sign_gap_ok and corr_margin_ok and gap_margin_ok)

    return {
        "corr_A_REAL": corr_real,
        "corr_A_NULL": corr_null,
        "brecha_B_REAL": gap_real,
        "brecha_B_NULL": gap_null,
        "sign_corr_ok": bool(sign_corr_ok),
        "sign_gap_ok": bool(sign_gap_ok),
        "corr_margin_ok": bool(corr_margin_ok),
        "gap_margin_ok": bool(gap_margin_ok),
        "seed_bite": seed_bite,
    }


# ------------------------------------------------------------
# Barrido completo
# ------------------------------------------------------------
def run_production(amplitud_grid, r_grid, seeds, L, B, n_steps, tag, keep_raw_zonas=True):
    t0 = time.time()
    resultados = {}  # key "amp|r" -> {"per_seed": {...}, "resumen": {...}}

    for amp in amplitud_grid:
        for r in r_grid:
            key = f"{amp:.4f}|{r:.4f}"
            per_seed = {}
            n_bite = 0
            corr_reals, corr_nulls, gap_reals, gap_nulls = [], [], [], []
            for seed in seeds:
                real = run_single("REAL", amp, r, seed, L, B, n_steps)
                null = run_single("NULL_BARAJADO", amp, r, seed, L, B, n_steps)
                ev = evaluate_point(real, null)
                entry = {"evaluation": ev}
                if keep_raw_zonas:
                    entry["REAL"] = real
                    entry["NULL_BARAJADO"] = null
                per_seed[str(seed)] = entry
                if ev["seed_bite"]:
                    n_bite += 1
                if not np.isnan(ev["corr_A_REAL"]):
                    corr_reals.append(ev["corr_A_REAL"])
                if not np.isnan(ev["corr_A_NULL"]):
                    corr_nulls.append(ev["corr_A_NULL"])
                gap_reals.append(ev["brecha_B_REAL"])
                gap_nulls.append(ev["brecha_B_NULL"])

            bite_rate = n_bite / len(seeds) if seeds else 0.0

            def pctl(vals, p):
                return float(np.percentile(vals, p)) if vals else None

            resumen = {
                "amplitud_rugosidad": amp,
                "r": r,
                "n_seeds": len(seeds),
                "n_seeds_bite": n_bite,
                "bite_rate": bite_rate,
                "pass_amplitud_r": bite_rate >= PASS_RATE_MIN,
                "corr_A_REAL_mean": float(np.mean(corr_reals)) if corr_reals else None,
                "corr_A_REAL_median": float(np.median(corr_reals)) if corr_reals else None,
                "corr_A_REAL_p10_p90": [pctl(corr_reals, 10), pctl(corr_reals, 90)] if corr_reals else None,
                "corr_A_NULL_mean": float(np.mean(corr_nulls)) if corr_nulls else None,
                "corr_A_NULL_median": float(np.median(corr_nulls)) if corr_nulls else None,
                "corr_A_NULL_p10_p90": [pctl(corr_nulls, 10), pctl(corr_nulls, 90)] if corr_nulls else None,
                "brecha_B_REAL_mean": float(np.mean(gap_reals)),
                "brecha_B_REAL_median": float(np.median(gap_reals)),
                "brecha_B_REAL_p10_p90": [pctl(gap_reals, 10), pctl(gap_reals, 90)],
                "brecha_B_NULL_mean": float(np.mean(gap_nulls)),
                "brecha_B_NULL_median": float(np.median(gap_nulls)),
                "brecha_B_NULL_p10_p90": [pctl(gap_nulls, 10), pctl(gap_nulls, 90)],
            }
            resultados[key] = {"per_seed": per_seed, "resumen": resumen}
            print(
                f"  [{tag}] amp={amp:6.3f} r={r:7.3f}  bite_rate={bite_rate:.3f}  "
                f"corr_REAL(mean)={resumen['corr_A_REAL_mean']}  "
                f"corr_NULL(mean)={resumen['corr_A_NULL_mean']}  "
                f"gap_REAL(mean)={resumen['brecha_B_REAL_mean']:.5f}  "
                f"gap_NULL(mean)={resumen['brecha_B_NULL_mean']:.5f}  "
                f"elapsed={time.time()-t0:.1f}s"
            )

    payload = {
        "experimento": "F4-5 rugosidad de densidad",
        "protocol_id": PROTOCOL_ID,
        "tag": tag,
        "sello": {
            "L": L, "B": B, "RHO0": RHO0, "D0": D0,
            "D_FLOOR_RATIO": D_FLOOR_RATIO, "SIGMA_SUAVIZADO": SIGMA_SUAVIZADO,
            "T_NOISE_STD": T_NOISE_STD, "SIGMA_DINAMICO": SIGMA_DINAMICO,
            "TG_MAX": TG_MAX, "n_steps": n_steps, "N_SUB": N_SUB, "DT_SUB": DT_SUB,
        },
        "barrido": {
            "amplitud_grid": list(amplitud_grid),
            "r_grid": list(r_grid),
            "seeds": list(seeds),
            "n_seeds": len(seeds),
            "modes": list(MODES),
        },
        "criterio_preregistrado": {
            "MARGEN_CORR": MARGEN_CORR,
            "MARGEN_BRECHA": MARGEN_BRECHA,
            "PASS_RATE_MIN": PASS_RATE_MIN,
            "descripcion": (
                "seed_bite = signo(corr_A_REAL)<0 AND signo(brecha_B_REAL)>0 AND "
                "|corr_A_REAL| > |corr_A_NULL| + MARGEN_CORR AND "
                "brecha_B_REAL > brecha_B_NULL + MARGEN_BRECHA; "
                "PASS(amplitud,r) = bite_rate over seeds >= PASS_RATE_MIN"
            ),
        },
        "resultados_por_punto": resultados,
        "runtime_seconds": time.time() - t0,
        "generated_at_unix": time.time(),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "produccion"], default="produccion", nargs="?")
    parser.add_argument("--no-raw-zonas", action="store_true",
                         help="omite guardar los arrays crudos de rho_zona/persistencia por semilla (reduce tamano del JSON)")
    args = parser.parse_args()

    if args.mode == "smoke":
        print(f"=== F4-5 rugosidad de densidad — modo=smoke ({PROTOCOL_ID}) ===")
        payload = run_production(
            AMPLITUD_SMOKE, R_SMOKE, SEEDS_SMOKE, L_SMOKE, B_SMOKE, N_STEPS_SMOKE,
            tag="smoke", keep_raw_zonas=True,
        )
        out_json = OUT_DIR / "F4_5_smoke_result.json"
    else:
        print(f"=== F4-5 rugosidad de densidad — modo=produccion ({PROTOCOL_ID}) ===")
        print(f"amplitud_grid={AMPLITUD_GRID}")
        print(f"r_grid={R_GRID}")
        print(f"seeds={SEEDS_STANDARD}")
        payload = run_production(
            AMPLITUD_GRID, R_GRID, SEEDS_STANDARD, L_PROD, B_PROD, N_STEPS_PROD,
            tag="produccion", keep_raw_zonas=not args.no_raw_zonas,
        )
        out_json = OUT_DIR / "F4_5_produccion_result.json"

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")
    print(f"runtime_seconds={payload['runtime_seconds']:.1f}")


if __name__ == "__main__":
    main()
