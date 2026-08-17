#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2_2_D_multipaso_engine.py — "D emergente multi-paso: ¿el r* nominal es el crítico real?"
============================================================================================

Ver PROTOCOLO_F2-2_PREREGISTRO.md (mismo directorio) — fechado ANTES de este motor,
formula de r* y criterios de PASS congelados ahí, no se tocan aquí.

Reusa (import, sin editar) la física de cs074_rcruz.py:
  campo_inicial, paso_difusion, paso_expansion, persistencia, medir_D,
  medir_pasos_lavado, corrida, evolucionar, temperatura_fisica, T_SING, T_FIN.

Hecho que se explota (declarado en el protocolo): la dinámica acoplada depende SOLO de H,
nunca de r ni D directamente. Por eso el barrido caro (H × semillas) se corre una vez;
D_k (k=1,2,5,10,50) se mide aparte (difusión pura, barata) y r_k = H/D_k es una
RE-ETIQUETA del eje x sobre los mismos puntos — no una nueva dinámica.

Salida: F2_2_resultado_<tag>.json por combinación (N, eps), con:
  - filas: H, r_nominal, D_1, P_real, P_null, dispersión entre semillas
  - D_k por escala y por método (A=compuesto, B=tasa local)
  - r*_k por escala k, con la fórmula congelada en el protocolo
  - metadatos de calibración (pasos, lavado) y timestamps
"""
from __future__ import annotations

import json
import sys
import time
import datetime
from pathlib import Path

import numpy as np

# --- import directo del código base (NO se edita, NO se reimplementa la física) ---
BASE_DIR = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
sys.path.insert(0, str(BASE_DIR))
import cs074_rcruz as base  # noqa: E402

OUT = Path(__file__).resolve().parent
K_SCALES = [1, 2, 5, 10, 50]
SEED_DYN_OFFSET = 7000   # semillas para la dinámica acoplada (real/null)
SEED_D_OFFSET = 8000     # semillas para la medición de D_k (independiente)


def ts():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print(f"[{ts()}] {msg}", file=sys.stderr, flush=True)


def medir_Dk_trayectoria(N, eps, seed, k_max=50):
    """
    Corre difusión PURA (H=0, activo=todo vivo) k_max pasos desde una condición inicial
    fresca, grabando c_t = std(phi) en cada paso t=0..k_max.
    Devuelve el array c_t completo; D_k (métodos A y B) se derivan de él fuera de esta
    función (así un solo trayecto sirve para TODAS las escalas k, sin recomputar).
    """
    rng = np.random.default_rng(seed)
    phi, _ = base.campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)
    c = np.empty(k_max + 1, dtype=float)
    c[0] = phi.std()
    for t in range(1, k_max + 1):
        phi = base.paso_difusion(phi, activo)
        c[t] = phi.std()
    return c


def Dk_de_trayectoria(c, k):
    """Método A (compuesto) y B (tasa local en el paso k) a partir de la trayectoria c_t."""
    c0 = c[0]
    if c0 <= 0:
        return 0.0, 0.0
    ck = c[k]
    ratio = max(ck / c0, 0.0)
    D_A = 1.0 - (ratio ** (1.0 / k))
    ck_1 = c[k - 1]
    D_B = 1.0 - (ck / ck_1) if ck_1 > 0 else 0.0
    D_B = max(0.0, float(D_B))
    return max(0.0, float(D_A)), D_B


def medir_D_multiescala(N, eps, semillas, k_scales=K_SCALES, seed_offset=SEED_D_OFFSET):
    """
    Para 'semillas' trayectorias independientes, mide D_k (A y B) en cada escala k.
    También verifica D_1^A contra medir_D() del código base (cross-check de
    autoconsistencia, T3/T6: el instrumento se audita a sí mismo).
    """
    k_max = max(k_scales)
    trayectorias = [medir_Dk_trayectoria(N, eps, seed_offset + s, k_max) for s in range(semillas)]
    out = {}
    for k in k_scales:
        A_vals = []
        B_vals = []
        for c in trayectorias:
            A, B = Dk_de_trayectoria(c, k)
            A_vals.append(A)
            B_vals.append(B)
        out[k] = {
            "D_A_mean": float(np.mean(A_vals)),
            "D_A_std": float(np.std(A_vals)),
            "D_B_mean": float(np.mean(B_vals)),
            "D_B_std": float(np.std(B_vals)),
            "D_A_vals": [round(float(v), 8) for v in A_vals],
        }
    # cross-check contra medir_D() original del código base (debe coincidir con D_1^A)
    D1_original = [base.medir_D(N, eps, seed_offset + s) for s in range(semillas)]
    out["_crosscheck_D1_original_vs_metodoA"] = {
        "D1_original_mean": float(np.mean(D1_original)),
        "D1_metodoA_mean": out[1]["D_A_mean"],
        "abs_diff": abs(float(np.mean(D1_original)) - out[1]["D_A_mean"]),
    }
    return out


def r_nominal_grid(n_log=24, lo=-2.0, hi=2.0):
    pos = np.logspace(lo, hi, n_log)
    return [0.0] + [float(x) for x in pos]


def barrido_H(N, eps, semillas, r_nom, D1_ref, pasos, seed_dyn_offset=SEED_DYN_OFFSET):
    """
    Corre la dinámica acoplada UNA vez por punto de r_nominal (usando H = r_nom*D1_ref),
    real y null, sobre 'semillas' semillas. Devuelve filas con H, P_real, P_null y
    dispersión — la parte cara del experimento.
    """
    filas = []
    for r_t in r_nom:
        H = float(min(r_t * D1_ref, 1.0)) if D1_ref > 0 else (0.0 if r_t == 0 else 1.0)
        Preal, Pnull = [], []
        for s in range(semillas):
            seed = seed_dyn_offset + s
            rr = base.corrida(N, eps, H, pasos, seed=seed, null=False)
            nn = base.corrida(N, eps, H, pasos, seed=seed, null=True)
            Preal.append(rr["P"])
            Pnull.append(nn["P"])
        Preal = np.array(Preal)
        Pnull = np.array(Pnull)
        sd = np.sqrt((Preal.var() + Pnull.var()) / 2.0)
        sd = max(sd, 1.0 / max(len(Preal), 1))
        z = float((Preal.mean() - Pnull.mean()) / sd)
        filas.append({
            "r_nominal": r_t,
            "H": H,
            "P_real_mean": float(Preal.mean()),
            "P_real_std": float(Preal.std()),
            "P_null_mean": float(Pnull.mean()),
            "P_null_std": float(Pnull.std()),
            "z": round(z, 3),
        })
    return filas


def estimar_rstar(filas, D_k, label):
    """
    Fórmula congelada en PROTOCOLO_F2-2_PREREGISTRO.md sección 4. Reetiqueta r usando
    D_k (no re-corre nada): r_k = H / D_k para cada fila con H>0; la fila H=0 queda en
    r=0 siempre. floor/ceiling se calculan sobre P_real, ordenando por H (invariante).
    """
    filas_ordenadas = sorted(filas, key=lambda f: f["H"])
    floor_rows = [f for f in filas_ordenadas if f["H"] == 0.0]
    if not floor_rows:
        return {"r_star": None, "motivo": "sin punto H=0 (floor indefinido)"}
    P_floor = floor_rows[0]["P_real_mean"]
    resto = [f for f in filas_ordenadas if f["H"] > 0.0]
    if len(resto) < 3:
        return {"r_star": None, "motivo": "grid insuficiente para ceiling"}
    top3 = resto[-3:]
    P_ceiling = float(np.mean([f["P_real_mean"] for f in top3]))
    objetivo = (P_floor + P_ceiling) / 2.0

    if D_k <= 0:
        return {"r_star": None, "motivo": f"D_k<=0 para {label}", "P_floor": P_floor, "P_ceiling": P_ceiling, "objetivo": objetivo}

    puntos = [(f["H"] / D_k, f["P_real_mean"]) for f in resto]
    puntos.sort(key=lambda p: p[0])

    r_star = None
    for i in range(len(puntos) - 1):
        r0, p0 = puntos[i]
        r1, p1 = puntos[i + 1]
        if (p0 - objetivo) * (p1 - objetivo) <= 0 and p1 != p0:
            if r0 <= 0 or r1 <= 0:
                continue
            lo0, lo1 = np.log10(r0), np.log10(r1)
            frac = (objetivo - p0) / (p1 - p0)
            r_star = float(10 ** (lo0 + frac * (lo1 - lo0)))
            break

    motivo = "cruce encontrado" if r_star is not None else "P_real nunca cruza el objetivo (curva plana o floor≈ceiling)"
    return {
        "r_star": r_star,
        "D_k": D_k,
        "P_floor": P_floor,
        "P_ceiling": P_ceiling,
        "objetivo": objetivo,
        "motivo": motivo,
    }


def correr_combinacion(tag, N, eps, semillas, n_log_grid, pasos_fijo=None):
    log(f"=== combinación {tag}: N={N} eps={eps} semillas={semillas} n_log_grid={n_log_grid} ===")
    t0 = time.time()

    log(f"{tag}: midiendo D_1 (referencia, media sobre {semillas} semillas)...")
    D1_vals = [base.medir_D(N, eps, SEED_DYN_OFFSET + s) for s in range(semillas)]
    D1_ref = float(np.mean(D1_vals))
    log(f"{tag}: D1_ref={D1_ref:.6g}")

    if pasos_fijo is None:
        log(f"{tag}: calibrando pasos de lavado (medir_pasos_lavado, P_thr=0.05)...")
        cal = base.medir_pasos_lavado(N, eps, max(semillas, 4))
        pasos = cal["pasos"]
    else:
        cal = {"pasos": pasos_fijo, "fijo": True}
        pasos = pasos_fijo
    log(f"{tag}: pasos={pasos} calibracion={cal}")

    r_nom = r_nominal_grid(n_log=n_log_grid)
    log(f"{tag}: grid r_nominal ({len(r_nom)} puntos) = {[round(x,4) for x in r_nom]}")

    log(f"{tag}: barrido H (dinámica acoplada real+null, {len(r_nom)} puntos x {semillas} semillas)...")
    t_dyn0 = time.time()
    filas = barrido_H(N, eps, semillas, r_nom, D1_ref, pasos)
    log(f"{tag}: barrido H terminado en {time.time()-t_dyn0:.1f}s")

    log(f"{tag}: midiendo D_k multiescala k={K_SCALES} ({semillas} trayectorias, k_max={max(K_SCALES)})...")
    t_Dk0 = time.time()
    Dk = medir_D_multiescala(N, eps, semillas)
    log(f"{tag}: D_k terminado en {time.time()-t_Dk0:.1f}s -> {Dk['_crosscheck_D1_original_vs_metodoA']}")

    log(f"{tag}: estimando r*_k para cada escala (fórmula congelada en preregistro)...")
    rstars = {}
    rstars["k1_D1_original"] = estimar_rstar(filas, D1_ref, "D1_original(referencia del grid)")
    for k in K_SCALES:
        Dk_A = Dk[k]["D_A_mean"]
        rstars[f"k{k}"] = estimar_rstar(filas, Dk_A, f"D_{k}_metodoA")

    for k, r in rstars.items():
        log(f"{tag}: r*[{k}] = {r.get('r_star')}  ({r.get('motivo')})")

    result = {
        "tag": tag,
        "N": N,
        "eps": eps,
        "semillas": semillas,
        "pasos": pasos,
        "calibracion_lavado": cal,
        "D1_ref_grid": D1_ref,
        "D1_ref_vals": [round(float(v), 8) for v in D1_vals],
        "r_nominal_grid": r_nom,
        "filas": filas,
        "D_k_multiescala": Dk,
        "r_star_por_escala": rstars,
        "elapsed_s": time.time() - t0,
        "timestamp_fin": ts(),
    }
    out_path = OUT / f"F2_2_resultado_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"{tag}: guardado {out_path} ({time.time()-t0:.1f}s total)")
    return result


def main():
    log("=== F2-2 D multipaso: inicio de corrida completa ===")
    resultados = {}

    # Primario: N=200, eps=1e-3, 16 semillas, grid fino de 25 puntos
    resultados["primario_eps1e-3"] = correr_combinacion(
        "primario_N200_eps1e-3", N=200, eps=1e-3, semillas=16, n_log_grid=24
    )

    # Robustez de eps: mismo N, eps=0.1
    resultados["robustez_eps0.1"] = correr_combinacion(
        "robustez_N200_eps0.1", N=200, eps=0.1, semillas=16, n_log_grid=24
    )

    # Robustez de N: N=400, eps=1e-3, grid reducido, 12 semillas
    resultados["robustez_N400"] = correr_combinacion(
        "robustez_N400_eps1e-3", N=400, eps=1e-3, semillas=12, n_log_grid=11
    )

    log("=== F2-2: todas las combinaciones terminadas ===")
    resumen = {
        tag: {
            "r_star_por_escala": {k: v.get("r_star") for k, v in r["r_star_por_escala"].items()},
            "elapsed_s": r["elapsed_s"],
        }
        for tag, r in resultados.items()
    }
    (OUT / "F2_2_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"resumen: {json.dumps(resumen, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
