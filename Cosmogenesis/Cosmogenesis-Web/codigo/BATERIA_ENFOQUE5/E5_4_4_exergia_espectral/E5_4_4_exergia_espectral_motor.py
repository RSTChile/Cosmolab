#!/usr/bin/env python3
"""
E5_4_4_exergia_espectral_motor.py — Enfoque 5, Tema 4, experimento E5.4-4

"Exergía por escalas espectrales: ¿qué longitudes de onda la retienen?"

Spec (BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md, Tema 4):
  Barrido: a × banda espectral completa × ≥12 semillas.
  Observable: exergía por escala vs a.
  NULL: densidad fija.
  PASS: espectro de retención de exergía reportado; se compara con
        "escalas grandes primero" SIN imponerlo.

Reutiliza la metodología del experimento espectral previo casi idéntico
(codigo/BATERIA_FUNDAMENTOS/F3_5_espectral/F3_5_espectral_motor.py, que tuvo
éxito claro: orden de congelamiento perfecto, escalas grandes primero) —
misma FFT por 6 bandas logarítmicas del campo T, mismo detector de
punto-de-no-retorno ("congelamiento") por banda vía pendiente log-log — pero
aplicada al observable pedido aquí: EXERGÍA por banda, no potencia espectral
cruda. La exergía se define en PROTOCOLO_E5.4-4_PREREGISTRO.md §3.1 como la
disponibilidad termodinámica de orden cuadrático de la desviación del campo
respecto de un estado de referencia fijo T_ref (medido en a=1, no impuesto),
y se verifica por un segundo método independiente en espacio real (§3.2,
identidad de Parseval) antes de usarse como observable de congelamiento.

Reutiliza (importa, NO copia ni edita) el sello físico exacto de
CF2_estiramiento_motor.py: L,H_EXP,RHO0,D0,W0,DT,N_SUB,
ORIGINAL_STEPS_PER_TG, initial_T(), diffuse(). Las 6 bandas espectrales
logarítmicas (BAND_DEFS) se redefinen aquí por VALOR (idénticas a las de
F3_5_espectral_motor.py) en vez de importarse desde ese archivo, para no
crear una dependencia entre carpetas de agentes distintos corriendo en
paralelo (regla de aislamiento de este batch).

No edita CF2_estiramiento_motor.py, F3_5_espectral_motor.py ni ningún otro
archivo existente. Este script NO se auto-adjudica el hallazgo: entrega
números crudos (curvas R_X_banda(a), frac_banda(a), freeze_a por banda,
diagnóstico de conservación de T_ref, discrepancia del cross-check en
espacio real); el veredicto final de la batería es de CS/Alexis.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

# ============================================================
# Import directo (no copia) del sello físico de CF-2, por ruta de archivo,
# igual que hizo F3_5_espectral_motor.py.
# ============================================================
CODE_DIR = Path(__file__).resolve().parent
BATERIA_DIR = CODE_DIR.parent              # .../codigo/BATERIA_ENFOQUE5
CODIGO_DIR = BATERIA_DIR.parent            # .../codigo
WEB_ROOT = CODIGO_DIR.parent               # .../Cosmogenesis-Web

CF2_PATH = CODIGO_DIR / "CF2_estiramiento" / "CF2_estiramiento_motor.py"

_spec = importlib.util.spec_from_file_location("cf2_estiramiento_motor_e544", CF2_PATH)
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
# Barrido pre-registrado (PROTOCOLO_E5.4-4_PREREGISTRO.md, sección 4)
# ============================================================
A_GRID_PROD = np.geomspace(1.0, 1e4, 30)
A_GRID_SMOKE = np.geomspace(1.0, 100.0, 8)

SEEDS_STANDARD = [7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321]
SEEDS_E5_4_4 = SEEDS_STANDARD + [13, 271828, 161803, 31415, 90210, 20260724]
SEEDS_SMOKE = [7, 42]

NOISE_LEVELS_PROD = [0.0, 1e-3, 1e-2]
NOISE_LEVELS_SMOKE = [0.0]

MODES = ["REAL", "NULL_RHO_FIXED"]

# Bandas espectrales log (idénticas por valor a F3_5_espectral_motor.py §3).
NYQ = L // 2  # 32 para L=64
BAND_DEFS = [
    ("B0", [1]),
    ("B1", [2]),
    ("B2", list(range(3, 5))),         # 3-4
    ("B3", list(range(5, 9))),         # 5-8
    ("B4", list(range(9, 17))),        # 9-16
    ("B5", list(range(17, NYQ + 1))),  # 17-32
]

# ============================================================
# Criterio de detección de congelamiento y PASS pre-registrados
# (PROTOCOLO_E5.4-4_PREREGISTRO.md, secciones 3 y 6 — no se tocan tras correr)
# ============================================================
FREEZE_SLOPE_TOL = 0.02
R_FLOOR = 0.05
RHO_MIN = 0.6
PASS_RATE_MIN = 0.55
MIN_BANDS_FOR_ORDER = 3

OUT_DIR = WEB_ROOT / "results" / "BATERIA_ENFOQUE5" / "E5_4_4_exergia_espectral"


# ============================================================
# Observable: potencia espectral por banda (frecuencia) + cross-check en
# espacio real (identidad de Parseval, PROTOCOLO §3.2).
# ============================================================
def band_power_and_realcheck(T: np.ndarray) -> tuple[dict, dict, float]:
    """
    Devuelve:
      P_banda: dict nombre_banda -> potencia (Sum_n |rFFT(T fila)|^2, promediada
               en filas), idéntico en construcción a F3_5.band_power().
      realcheck: dict nombre_banda -> {"actual": ..., "predicted_from_freq": ...}
               reconstrucción pasa-banda en espacio real (irFFT con máscara)
               vs la predicción por identidad de Parseval a partir de P(n,a).
               Deben coincidir a precisión numérica; una discrepancia grande
               indica un error de implementación, no un hallazgo físico.
      T_mean: media espacial del campo en este checkpoint (diagnóstico E1,
               PROTOCOLO §3.4).
    """
    n_cols = T.shape[1]
    F = np.fft.rfft(T, axis=1)              # (L, L/2+1) complejo
    P_full = (np.abs(F) ** 2).mean(axis=0)  # potencia media por armónico n=0..L/2

    P_banda = {name: float(P_full[ns].sum()) for name, ns in BAND_DEFS}

    realcheck = {}
    for name, ns in BAND_DEFS:
        mask = np.zeros(F.shape[1], dtype=complex)
        mask[ns] = 1.0
        F_band = F * mask[np.newaxis, :]
        T_band = np.fft.irfft(F_band, n=n_cols, axis=1)
        actual = float(np.mean(np.sum(T_band ** 2, axis=1)))
        # identidad de Parseval por fila: sum(row^2) = (1/L) * sum_n w_n |X[n]|^2
        # con w_n=2 para 0<n<Nyquist, w_n=1 en Nyquist exacto (n=L/2).
        w = np.array([1.0 if n == n_cols // 2 else 2.0 for n in ns])
        predicted = float((1.0 / n_cols) * np.sum(w * P_full[ns]))
        realcheck[name] = {"actual": actual, "predicted_from_freq": predicted}

    T_mean = float(T.mean())
    return P_banda, realcheck, T_mean


def run_sweep(mode: str, seed: int, a_grid: np.ndarray, sigma_din: float) -> dict:
    """Misma técnica de muestreo markoviano que CF-2/F3-1/F3-5: una sola
    trayectoria por semilla, muestreada en los t_g(a) objetivo."""
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
        P_banda, realcheck, T_mean = band_power_and_realcheck(T)
        checkpoints.append({
            "a": float(a_now), "tg": float(tg_now),
            "bandas_potencia": P_banda, "realcheck": realcheck, "T_mean": T_mean,
        })

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
            rho = RHO0 / (a ** 3)
            D = D0 * (rho / RHO0)

        T = diffuse(T, D, DT, N_SUB)
        if sigma_din > 0.0:
            # ruido dinámico tipo Wiener, mismo mecanismo que F3_5/F3_1.
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
    banda_potencia_series = {name: [c["bandas_potencia"][name] for c in checkpoints] for name, _ in BAND_DEFS}
    T_mean_series = [c["T_mean"] for c in checkpoints]
    realcheck_series = {
        name: [c["realcheck"][name] for c in checkpoints] for name, _ in BAND_DEFS
    }

    return {
        "mode": mode, "seed": seed, "sigma_din": sigma_din,
        "a_grid": a_vals,
        "banda_potencia": banda_potencia_series,
        "T_mean_series": T_mean_series,
        "realcheck_series": realcheck_series,
    }


# ============================================================
# Análisis: exergía por banda, retención, fracción del total, freeze_a,
# orden Spearman, diagnóstico de conservación y de cross-check Parseval.
# ============================================================
def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Correlación de rango de Spearman sin dependencias externas."""
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

    T_ref = float(run["T_mean_series"][0])
    T_ref_safe = T_ref if abs(T_ref) > 1e-300 else 1e-300

    # exergía por banda: X_banda(a) = P_banda(a) / (2*T_ref)  (PROTOCOLO §3.1)
    x_curves = {}
    for name in band_names:
        P = np.array(run["banda_potencia"][name])
        x_curves[name] = (P / (2.0 * T_ref_safe)).tolist()

    x_total = np.sum(np.array([x_curves[n] for n in band_names]), axis=0)
    frac_banda = {name: (np.array(x_curves[name]) / np.clip(x_total, 1e-300, None)).tolist()
                  for name in band_names}

    # diagnóstico E1: deriva de la media espacial respecto de T_ref (§3.4)
    T_mean_series = np.array(run["T_mean_series"])
    deriva_media_rel = (np.abs(T_mean_series - T_ref) / T_ref_safe).tolist()

    # diagnóstico del cross-check en espacio real (§3.2, identidad de Parseval)
    max_discrepancia_rel = 0.0
    for name in band_names:
        for rc in run["realcheck_series"][name]:
            denom = max(abs(rc["predicted_from_freq"]), 1e-300)
            disc = abs(rc["actual"] - rc["predicted_from_freq"]) / denom
            if disc > max_discrepancia_rel:
                max_discrepancia_rel = disc

    freeze_a = {}
    r_frozen = {}
    frozen_type = {}
    r_curves = {}  # retención relativa R_X_banda(a) = X_banda(a)/X_banda(a=1)

    for name in band_names:
        X = np.array(x_curves[name])
        X0 = X[0] if X[0] > 0 else 1e-300
        R = X / X0
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
                i_star = i
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
        "T_ref": T_ref,
        "x_curves": x_curves,
        "x_total": x_total.tolist(),
        "frac_banda": frac_banda,
        "deriva_media_rel": deriva_media_rel,
        "max_discrepancia_realspace_rel": max_discrepancia_rel,
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
    max_discrepancia_global = 0.0
    max_deriva_global = 0.0

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

            max_discrepancia_global = max(
                max_discrepancia_global,
                real_an["max_discrepancia_realspace_rel"],
                null_an["max_discrepancia_realspace_rel"],
            )
            max_deriva_global = max(
                max_deriva_global,
                max(real_an["deriva_media_rel"]) if real_an["deriva_media_rel"] else 0.0,
                max(null_an["deriva_media_rel"]) if null_an["deriva_media_rel"] else 0.0,
            )

            key = f"sigma={sigma}|seed={seed}"
            per_combo[key] = {
                "sigma_din": sigma,
                "seed": seed,
                "REAL": {"run": real_run, "analisis": real_an},
                "NULL_RHO_FIXED": {"run": null_run, "analisis": null_an},
            }

            n_total += 1
            # Igual convención que F3_5: la combinación queda INDETERMINADA solo si
            # el lado REAL no se puede evaluar (menos de 3 bandas congeladas). Que el
            # NULL no tenga orden evaluable es justamente lo que predice §5 del
            # protocolo (sin freno D∝a⁻³, el NULL no debería congelar ordenadamente);
            # se captura vía orden_ok(NULL)=False, no como indeterminación.
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
        verdict = "E544_PASS"
    else:
        verdict = "E544_FAIL"

    payload = {
        "experimento": "E5.4-4 exergía por escalas espectrales",
        "tag": tag,
        "sello_fisico_heredado_de_CF2": {
            "L": L, "H_EXP": H_EXP, "RHO0": RHO0, "D0": D0, "W0": W0,
            "DT": DT, "N_SUB": N_SUB, "ORIGINAL_STEPS_PER_TG": ORIGINAL_STEPS_PER_TG,
            "cf2_path": str(CF2_PATH),
        },
        "bandas_espectrales": {name: ns for name, ns in BAND_DEFS},
        "definicion_exergia": "X_banda(a) = P_banda(a) / (2*T_ref); T_ref = T_mean medido en a=1 por corrida (PROTOCOLO §3.1)",
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
                "sobre bandas con freeze_a finito (>=3 requeridas). freeze_a "
                "calculado sobre R_X_banda(a) = X_banda(a)/X_banda(a=1)."
            ),
        },
        "diagnosticos_globales": {
            "max_discrepancia_realspace_vs_freq_rel": max_discrepancia_global,
            "max_deriva_media_espacial_rel_vs_T_ref": max_deriva_global,
            "nota": (
                "max_discrepancia_realspace_vs_freq_rel = mayor discrepancia relativa "
                "observada entre el cómputo en frecuencia (P_banda) y la reconstrucción "
                "independiente en espacio real (identidad de Parseval, PROTOCOLO §3.2), "
                "sobre TODAS las combinaciones/checkpoints/bandas corridas. Debería ser "
                "~1e-12 (precisión de punto flotante) si no hay error de implementación. "
                "max_deriva_media_espacial_rel_vs_T_ref = mayor deriva relativa de la "
                "media espacial del campo respecto de T_ref, diagnóstico de la validez "
                "de usar una referencia de exergía FIJA (PROTOCOLO §3.4); el único "
                "mecanismo de ruptura esperado es el clip(T,0,1) heredado de CF2."
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
        seeds = SEEDS_E5_4_4
        a_grid = A_GRID_PROD
        sigma_levels = NOISE_LEVELS_PROD
        tag = "produccion"

    print(f"=== E5.4-4 exergía espectral — modo={args.mode} ===")
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
    diag = payload["diagnosticos_globales"]
    print(f"\ndiagnóstico Parseval (cross-check espacio real): "
          f"max_discrepancia_rel={diag['max_discrepancia_realspace_vs_freq_rel']:.3e}")
    print(f"diagnóstico E1 (deriva media espacial vs T_ref): "
          f"max_deriva_rel={diag['max_deriva_media_espacial_rel_vs_T_ref']:.3e}")
    print(f"\nveredicto (variante principal, sigma_din={sigma_levels[0]}): "
          f"{payload['veredicto_variante_principal_sigma0']}")
    print(f"(umbrales pre-registrados: PASS_RATE_MIN={PASS_RATE_MIN}, RHO_MIN={RHO_MIN})")

    out_json = OUT_DIR / f"E5_4_4_exergia_espectral_{args.mode}_result.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
