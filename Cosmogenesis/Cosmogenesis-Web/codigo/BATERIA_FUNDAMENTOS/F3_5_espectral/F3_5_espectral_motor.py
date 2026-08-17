#!/usr/bin/env python3
"""
F3_5_espectral_motor.py — BATERÍA_FUNDAMENTOS, Enfoque 3, experimento F3-5

"Enfriamiento en el espectro: ¿qué escalas se congelan primero?"

Lente de verificación TOTALMENTE distinta a F3-1 (que mide gradiente físico
∇_fis=∇_comov/a). Aquí el observable es puramente ESPECTRAL: se sigue la FFT
1D del campo comóvil T a lo largo del eje del salto de temperatura, agrupada
en 6 bandas logarítmicas de número de onda, y se detecta en qué valor de `a`
cada banda deja de perder potencia ("se congela"). La predicción física
pre-registrada (PROTOCOLO_F3-5_PREREGISTRO.md, escrito ANTES de este script)
es que las escalas GRANDES (k chico, λ grande) se congelan PRIMERO —a un `a`
más chico— porque no alcanzan a re-homogeneizarse a través del horizonte
creciente; las escalas CHICAS (k grande) siguen cambiando hasta un `a` mayor.

Reutiliza (importa, NO copia ni edita) el sello físico exacto de
CF2_estiramiento_motor.py: constantes L,H_EXP,RHO0,D0,W0,DT,N_SUB,
ORIGINAL_STEPS_PER_TG y las funciones initial_T(), diffuse(). Esto garantiza
que la lente espectral mira la MISMA dinámica que CF-2/F3-1, no una réplica
que pudo haber divergido. No se usa grad_metrics() de CF-2 en ningún punto:
el observable de este experimento es exclusivamente espectral.

No edita CF2_estiramiento_motor.py, F3_1_estiramiento_motor.py ni ningún otro
archivo existente. Este script NO se auto-adjudica el hallazgo: entrega
números crudos (curvas R_banda(a) completas + freeze_a por banda); el
veredicto final de la batería es de CS/Alexis.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Import directo (no copia) del sello físico de CF-2, por ruta de archivo
# para no depender de que el paquete esté en sys.path.
# ============================================================
CODE_DIR = Path(__file__).resolve().parent
BATERIA_DIR = CODE_DIR.parent          # .../codigo/BATERIA_FUNDAMENTOS
CODIGO_DIR = BATERIA_DIR.parent        # .../codigo
WEB_ROOT = CODIGO_DIR.parent           # .../Cosmogenesis-Web

CF2_PATH = CODIGO_DIR / "CF2_estiramiento" / "CF2_estiramiento_motor.py"

_spec = importlib.util.spec_from_file_location("cf2_estiramiento_motor", CF2_PATH)
_cf2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cf2)  # type: ignore[union-attr]

# Sello físico reutilizado tal cual (T1: no se retoca)
L = _cf2.L
H_EXP = _cf2.H_EXP
RHO0 = _cf2.RHO0
D0 = _cf2.D0
W0 = _cf2.W0
DT = _cf2.DT
N_SUB = _cf2.N_SUB
ORIGINAL_STEPS_PER_TG = _cf2.ORIGINAL_STEPS_PER_TG
initial_T = _cf2.initial_T
diffuse = _cf2.diffuse

# ============================================================
# Barrido pre-registrado (PROTOCOLO_F3-5_PREREGISTRO.md, sección 4)
# ============================================================
A_GRID_PROD = np.geomspace(1.0, 1e4, 30)
A_GRID_SMOKE = np.geomspace(1.0, 100.0, 8)

SEEDS_STANDARD = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321]
SEEDS_F3_5 = SEEDS_STANDARD + [13, 271828, 161803, 31415, 90210, 20260724]
SEEDS_SMOKE = [7, 42]

NOISE_LEVELS_PROD = [0.0, 1e-3, 1e-2]
NOISE_LEVELS_SMOKE = [0.0]

MODES = ["REAL", "NULL_RHO_FIXED"]

# Bandas espectrales log (PROTOCOLO §3). n = índice de armónico FFT real, 1..L/2.
NYQ = L // 2  # 32 para L=64
BAND_DEFS = [
    ("B0", [1]),
    ("B1", [2]),
    ("B2", list(range(3, 5))),      # 3-4
    ("B3", list(range(5, 9))),      # 5-8
    ("B4", list(range(9, 17))),     # 9-16
    ("B5", list(range(17, NYQ + 1))),  # 17-32
]

# ============================================================
# Criterio de detección de congelamiento y PASS pre-registrados
# (PROTOCOLO_F3-5_PREREGISTRO.md, secciones 3 y 6 — no se tocan tras correr)
# ============================================================
FREEZE_SLOPE_TOL = 0.02
R_FLOOR = 0.05
RHO_MIN = 0.6
PASS_RATE_MIN = 0.55
MIN_BANDS_FOR_ORDER = 3

OUT_DIR = WEB_ROOT / "results" / "BATERIA_FUNDAMENTOS" / "F3_5_espectral"


# ============================================================
# Dinámica física: misma trayectoria markoviana con checkpoints en a,
# igual método de muestreo que CF-2/F3-1, más ruido dinámico opcional
# (regla general de la batería §2, T7) y medición espectral por banda
# en vez de grad_metrics.
# ============================================================
def band_power(T: np.ndarray) -> dict:
    """FFT real 1D a lo largo de x (axis=1) por fila, promediada en potencia
    sobre las L filas, agregada en las 6 bandas logarítmicas de n (§3)."""
    F = np.fft.rfft(T, axis=1)          # (L, L/2+1) complejo
    P = (np.abs(F) ** 2).mean(axis=0)   # potencia media por armónico n=0..L/2
    out = {}
    for name, ns in BAND_DEFS:
        out[name] = float(P[ns].sum())
    return out


def run_sweep(mode: str, seed: int, a_grid: np.ndarray, sigma_din: float) -> dict:
    rng = np.random.default_rng(seed)
    T = initial_T(L, W0)
    T = T + 1e-4 * rng.normal(size=T.shape)
    T = np.clip(T, 0.0, 1.0)

    dtg = 1.0 / ORIGINAL_STEPS_PER_TG
    tg_targets = np.log(a_grid) / H_EXP
    tg_max = float(tg_targets[-1])
    n_steps = max(int(np.ceil(tg_max / dtg)), 1)

    checkpoints = []
    next_ckpt_idx = 0

    def record(tg_now, a_now):
        bp = band_power(T)
        checkpoints.append({"a": float(a_now), "tg": float(tg_now), "bandas": bp})

    if tg_targets[0] <= 1e-15:
        record(0.0, float(a_grid[0]))
        next_ckpt_idx = 1

    dt_sub = DT / N_SUB
    for step in range(1, n_steps + 1):
        tg = step * dtg
        a = float(np.exp(H_EXP * tg))

        if mode == "NULL_RHO_FIXED":
            rho = RHO0
            D = D0
        else:  # REAL
            rho = RHO0 / (a**3)
            D = D0 * (rho / RHO0)

        T = diffuse(T, D, DT, N_SUB)
        if sigma_din > 0.0:
            # ruido dinámico tipo Wiener, misma amplitud en cada subpaso
            # (mecanismo idéntico al de F3_1_estiramiento_motor.py; se aplica
            # UNA vez por paso de reloj genético con el sqrt(dt) agregado de
            # los N_SUB subpasos, dt_efectivo=DT en cada paso de registro)
            T = T + sigma_din * np.sqrt(dt_sub * N_SUB) * rng.normal(size=T.shape)
        T = np.clip(T, 0.0, 1.0)

        while next_ckpt_idx < len(tg_targets) and tg >= tg_targets[next_ckpt_idx] - 1e-9:
            record(tg, float(np.exp(H_EXP * tg_targets[next_ckpt_idx])))
            next_ckpt_idx += 1

    while next_ckpt_idx < len(tg_targets):
        a_last = float(a_grid[next_ckpt_idx])
        record(tg_targets[next_ckpt_idx], a_last)
        next_ckpt_idx += 1

    a_vals = [c["a"] for c in checkpoints]
    banda_series = {name: [c["bandas"][name] for c in checkpoints] for name, _ in BAND_DEFS}

    return {"mode": mode, "seed": seed, "sigma_din": sigma_din,
            "a_grid": a_vals, "banda_potencia": banda_series}


# ============================================================
# Análisis espectral: R_banda(a), freeze_a por banda, orden Spearman
# ============================================================
def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Correlación de rango de Spearman sin dependencias externas (scipy no
    garantizado en el venv de este proyecto)."""
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze_run(run: dict) -> dict:
    a_vals = np.array(run["a_grid"])
    band_names = [n for n, _ in BAND_DEFS]
    band_centers = {n: float(np.mean(ns)) for n, ns in BAND_DEFS}

    freeze_a = {}
    r_frozen = {}
    frozen_type = {}
    r_curves = {}

    for name in band_names:
        P = np.array(run["banda_potencia"][name])
        P0 = P[0] if P[0] > 0 else 1e-300
        R = P / P0
        r_curves[name] = R.tolist()

        if len(a_vals) < 2:
            freeze_a[name] = float("nan")
            r_frozen[name] = float("nan")
            frozen_type[name] = "indeterminado"
            continue

        logR = np.log(np.clip(R, 1e-300, None))
        logA = np.log(a_vals)
        slopes = (logR[1:] - logR[:-1]) / (logA[1:] - logA[:-1])

        i_star = None
        for i in range(len(slopes)):
            if np.all(np.abs(slopes[i:]) < FREEZE_SLOPE_TOL):
                i_star = i  # slope_i es entre a[i] y a[i+1]; congela en a[i]
                break

        if i_star is None:
            freeze_a[name] = float("nan")
            r_frozen[name] = float("nan")
            frozen_type[name] = "no_congelada_en_rango"
        else:
            freeze_a[name] = float(a_vals[i_star])
            rf = float(R[i_star])
            r_frozen[name] = rf
            frozen_type[name] = "frozen_depleted" if rf < R_FLOOR else "frozen_preserved"

    # test de orden: bandas no censuradas, centro de banda vs freeze_a
    valid_names = [n for n in band_names if not np.isnan(freeze_a[n])]
    if len(valid_names) >= MIN_BANDS_FOR_ORDER:
        xs = np.array([band_centers[n] for n in valid_names])
        ys = np.array([freeze_a[n] for n in valid_names])
        rho_orden = spearman_rho(xs, ys)
        indeterminado = bool(np.isnan(rho_orden))
    else:
        rho_orden = float("nan")
        indeterminado = True

    orden_ok = (not indeterminado) and (rho_orden >= RHO_MIN)

    return {
        "r_curves": r_curves,
        "freeze_a": freeze_a,
        "r_frozen": r_frozen,
        "frozen_type": frozen_type,
        "n_bandas_validas": len(valid_names),
        "rho_orden": rho_orden,
        "indeterminado": indeterminado,
        "orden_ok": orden_ok,
    }


def run_production(seeds: list[int], a_grid: np.ndarray, sigma_levels: list[float], tag: str) -> dict:
    t0 = time.time()
    per_combo = {}
    resumen_por_sigma = {}

    for sigma in sigma_levels:
        n_pass = 0
        n_indet = 0
        n_total = 0
        rho_orden_real_list = []
        orden_null_true = 0
        for seed in seeds:
            real_run = run_sweep("REAL", seed, a_grid, sigma)
            null_run = run_sweep("NULL_RHO_FIXED", seed, a_grid, sigma)
            real_an = analyze_run(real_run)
            null_an = analyze_run(null_run)

            key = f"sigma={sigma}|seed={seed}"
            per_combo[key] = {
                "sigma_din": sigma,
                "seed": seed,
                "REAL": {"run": real_run, "analisis": real_an},
                "NULL_RHO_FIXED": {"run": null_run, "analisis": null_an},
            }

            n_total += 1
            # BUGFIX post-smoke (pre-producción, no toca ningún umbral congelado):
            # la combinación solo queda INDETERMINADA si el lado REAL no se puede
            # evaluar (no hay suficientes bandas congeladas para juzgar el orden
            # central de la predicción). Que el NULL tenga <3 bandas congeladas
            # NO es indeterminación: es exactamente el resultado esperado por la
            # sección 5 del protocolo (sin freno D∝a⁻³, las bandas del NULL no
            # deberían asentarse) y ya lo captura correctamente `orden_ok(NULL)
            # = False` vía el flag `indeterminado` interno de analyze_run(). La
            # fórmula de combo_pass pre-registrada (sección 6:
            # "combo_pass = orden_ok(REAL) AND NOT orden_ok(NULL)") no cambia.
            if real_an["indeterminado"]:
                n_indet += 1
            else:
                combo_pass = bool(real_an["orden_ok"] and (not null_an["orden_ok"]))
                per_combo[key]["combo_pass"] = combo_pass
                if combo_pass:
                    n_pass += 1
                if null_an["orden_ok"]:
                    orden_null_true += 1
                if not np.isnan(real_an["rho_orden"]):
                    rho_orden_real_list.append(real_an["rho_orden"])

        n_evaluables = n_total - n_indet
        rate = (n_pass / n_evaluables) if n_evaluables else float("nan")
        median_rho_real = float(np.median(rho_orden_real_list)) if rho_orden_real_list else float("nan")
        null_bite_rate = 1.0 - (orden_null_true / n_evaluables) if n_evaluables else float("nan")

        resumen_por_sigma[str(sigma)] = {
            "n_total": n_total,
            "n_indeterminado": n_indet,
            "n_evaluable": n_evaluables,
            "n_combo_pass": n_pass,
            "rate": rate,
            "mediana_rho_orden_REAL": median_rho_real,
            "tasa_NULL_muerde": null_bite_rate,
        }

    sigma0_key = str(sigma_levels[0])  # variante principal = primer nivel (0.0 en producción)
    r0 = resumen_por_sigma[sigma0_key]

    if r0["n_evaluable"] == 0 or np.isnan(r0["rate"]):
        verdict = "INDETERMINADO"
    elif np.isnan(r0["mediana_rho_orden_REAL"]):
        verdict = "INDETERMINADO"
    elif r0["mediana_rho_orden_REAL"] < 0:
        verdict = "FAIL_INVERSO"
    elif r0["tasa_NULL_muerde"] < 0.55:
        verdict = "FAIL_NULL_NO_MUERDE"
    elif r0["rate"] >= PASS_RATE_MIN and r0["mediana_rho_orden_REAL"] >= RHO_MIN:
        verdict = "F35_PASS"
    else:
        verdict = "F35_FAIL"

    payload = {
        "experimento": "F3-5 enfriamiento en el espectro",
        "tag": tag,
        "sello_fisico_heredado_de_CF2": {
            "L": L, "H_EXP": H_EXP, "RHO0": RHO0, "D0": D0, "W0": W0,
            "DT": DT, "N_SUB": N_SUB, "ORIGINAL_STEPS_PER_TG": ORIGINAL_STEPS_PER_TG,
            "cf2_path": str(CF2_PATH),
        },
        "bandas_espectrales": {name: ns for name, ns in BAND_DEFS},
        "barrido": {
            "a_grid": a_grid.tolist(),
            "seeds": seeds,
            "n_seeds": len(seeds),
            "sigma_din_levels": sigma_levels,
        },
        "criterio_preregistrado": {
            "FREEZE_SLOPE_TOL": FREEZE_SLOPE_TOL,
            "R_FLOOR": R_FLOOR,
            "RHO_MIN": RHO_MIN,
            "PASS_RATE_MIN": PASS_RATE_MIN,
            "MIN_BANDS_FOR_ORDER": MIN_BANDS_FOR_ORDER,
            "descripcion": (
                "combo_pass = orden_ok(REAL) AND NOT orden_ok(NULL); "
                "orden_ok = rho_Spearman(centro_banda, freeze_a) >= RHO_MIN "
                "sobre bandas con freeze_a finito (>=3 requeridas)."
            ),
        },
        "resumen_por_sigma_din": resumen_por_sigma,
        "veredicto_variante_principal_sigma0": verdict,
        "resultados_por_combo": per_combo,
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
        seeds = SEEDS_SMOKE
        a_grid = A_GRID_SMOKE
        sigma_levels = NOISE_LEVELS_SMOKE
        tag = "smoke"
    else:
        seeds = SEEDS_F3_5
        a_grid = A_GRID_PROD
        sigma_levels = NOISE_LEVELS_PROD
        tag = "produccion"

    print(f"=== F3-5 espectral — modo={args.mode} ===")
    print(f"seeds={seeds}")
    print(f"a_grid[0]={a_grid[0]:.4g} a_grid[-1]={a_grid[-1]:.4g} n_puntos={len(a_grid)}")
    print(f"sigma_din_levels={sigma_levels}")

    payload = run_production(seeds, a_grid, sigma_levels, tag)

    print("\n=== RESUMEN CRUDO POR SIGMA_DIN (sin adjudicar más allá del criterio congelado) ===")
    for sigma_str, r in payload["resumen_por_sigma_din"].items():
        print(
            f"  sigma_din={sigma_str:>8}  n_eval={r['n_evaluable']:>3}/{r['n_total']:<3}  "
            f"indet={r['n_indeterminado']:>2}  rate={r['rate']:.3f}  "
            f"mediana_rho_REAL={r['mediana_rho_orden_REAL']:.3f}  "
            f"tasa_NULL_muerde={r['tasa_NULL_muerde']:.3f}"
        )
    print(f"\nveredicto (variante principal, sigma_din={sigma_levels[0]}): "
          f"{payload['veredicto_variante_principal_sigma0']}")
    print(f"(umbrales pre-registrados: PASS_RATE_MIN={PASS_RATE_MIN}, RHO_MIN={RHO_MIN})")

    out_json = OUT_DIR / f"F3_5_espectral_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
