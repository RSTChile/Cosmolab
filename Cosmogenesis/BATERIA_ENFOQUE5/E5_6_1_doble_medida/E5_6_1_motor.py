#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.6-1 — Doble medida: exergía termodinámica vs informacional, mismo barrido
=============================================================================

Pre-registro congelado ANTES de correr este motor:
  E5_6_1_PROTOCOLO_PREREGISTRO.md (mismo directorio)

Este motor NO modifica cs074_rcruz.py. Importa directamente su física
(campo_inicial, paso_difusion, paso_expansion, medir_D, medir_pasos_lavado,
persistencia, R_TARGETS) y añade:

  X_termo = persistencia(phi, contraste0)              [YA existe en el motor base,
                                                          es el X de E5.1-1, sin tocar]
  X_info  = 1 - H_spec/H_max  (entropía de Shannon del espectro de Fourier
            normalizado, DC excluido)                   [NUEVO, vía independiente]

Ambas se miden sobre el MISMO phi final, real y barajado (NULL), en el mismo
barrido eps x r x >=16 semillas. Se reporta correlación global, por-eps, y el
comportamiento de ambas bajo NULL. No se ajusta nada hacia el resultado.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent  # .../Cosmogenesis
BASE_PATH = PROJECT_ROOT / "cs074_rcruz.py"

# --- import del motor base SIN editarlo (import por ruta explícita, no se toca el archivo) ---
spec = importlib.util.spec_from_file_location("cs074_rcruz", str(BASE_PATH))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

OUT = HERE


# ------------------------- X_info: entropía espectral -------------------------

def entropia_espectral_X_info(phi: np.ndarray) -> float:
    """
    X_info = 1 - H_spec/H_max
    H_spec = entropía de Shannon del espectro de potencia normalizado (FFT real,
             DC excluido). H_max = log(N/2) (espectro plano = ruido blanco).
    Vía independiente de X_termo: dominio de frecuencias, sin usar varianza
    absoluta (normalizado a suma=1) ni autocorrelación de vecino inmediato.
    """
    n = phi.size
    centrado = phi - phi.mean()
    f = np.fft.rfft(centrado)
    # excluir DC (índice 0); usar todos los modos restantes hasta Nyquist
    potencia = np.abs(f[1:]) ** 2
    total = potencia.sum()
    n_modos = potencia.size
    if total <= 0 or n_modos <= 1:
        return 0.0
    p = potencia / total
    p_nz = p[p > 0]
    H = float(-np.sum(p_nz * np.log(p_nz)))
    Hmax = float(np.log(n_modos))
    if Hmax <= 0:
        return 0.0
    Hnorm = min(max(H / Hmax, 0.0), 1.0)
    return 1.0 - Hnorm


# ------------------------- corrida doble medida -------------------------

def corrida_doble(N, eps, H, pasos, seed):
    """
    Evoluciona el campo (misma física que cs074_rcruz.evolucionar, pero sin
    aplicar el barajado dentro de la función: aquí generamos phi_real y luego
    derivamos phi_null barajando el mismo phi_real final, exactamente como
    hace base.evolucionar(..., null=True) pero recuperando ambos resultados
    de UNA sola evolución física para no gastar el doble de pasos).
    """
    rng = np.random.default_rng(seed)
    phi, _ = base.campo_inicial(N, eps, rng)
    contraste0 = float(phi.std())
    activo = np.ones(N, dtype=bool)
    for _ in range(pasos):
        phi = base.paso_difusion(phi, activo)
        activo = base.paso_expansion(activo, H, rng)
    phi_real = phi
    phi_null = rng.permutation(phi_real)  # NULL = barajado espacial, igual que motor base

    Xt_real = base.persistencia(phi_real, contraste0)
    Xi_real = entropia_espectral_X_info(phi_real)
    Xt_null = base.persistencia(phi_null, contraste0)
    Xi_null = entropia_espectral_X_info(phi_null)

    return {
        "Xt_real": Xt_real, "Xi_real": Xi_real,
        "Xt_null": Xt_null, "Xi_null": Xi_null,
    }


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    t0 = time.time()
    ts_inicio = time.strftime("%Y-%m-%d %H:%M:%S %Z")

    N = 200
    SEMILLAS = 16
    EPS_LIST = [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0]  # mismo grid que modo produccion base
    R_GRID = sorted(set([0.0] + list(np.logspace(-3, 3, 25))))  # 26 pts, 6 décadas, cruza r=1

    # calibración de pasos: mismo criterio que motor base, medido, no puesto a mano
    cal_ref = base.medir_pasos_lavado(N, 1e-3, SEMILLAS)
    pasos_fijo = cal_ref["pasos"]
    print(
        f"[calibracion] N={N} eps=1e-3 mediana_lavado={cal_ref['mediana']} "
        f"pasos={pasos_fijo} lavo_todas={cal_ref['lavo_todas']}",
        file=sys.stderr, flush=True,
    )

    filas = []
    meta_por_eps = []
    for eps in EPS_LIST:
        D = float(np.mean([base.medir_D(N, eps, s) for s in range(SEMILLAS)]))
        meta_por_eps.append({"eps": eps, "D": D, "pasos": pasos_fijo})
        for r_tgt in R_GRID:
            if D > 0:
                Hh = float(min(r_tgt * D, 1.0))
                r_eff = Hh / D
            else:
                Hh = 0.0 if r_tgt == 0 else 1.0
                r_eff = 0.0 if r_tgt == 0 else float("inf")

            Xt_r, Xi_r, Xt_n, Xi_n = [], [], [], []
            for s in range(SEMILLAS):
                res = corrida_doble(N, eps, Hh, pasos_fijo, seed=2000 + s)
                Xt_r.append(res["Xt_real"]); Xi_r.append(res["Xi_real"])
                Xt_n.append(res["Xt_null"]); Xi_n.append(res["Xi_null"])

            fila = {
                "eps": eps, "r_target": r_tgt, "H": Hh, "D": D, "r_eff": r_eff,
                "pasos": pasos_fijo,
                "Xt_real_mean": float(np.mean(Xt_r)), "Xt_real_std": float(np.std(Xt_r)),
                "Xi_real_mean": float(np.mean(Xi_r)), "Xi_real_std": float(np.std(Xi_r)),
                "Xt_null_mean": float(np.mean(Xt_n)), "Xt_null_std": float(np.std(Xt_n)),
                "Xi_null_mean": float(np.mean(Xi_n)), "Xi_null_std": float(np.std(Xi_n)),
                "corr_real_within": pearson(Xt_r, Xi_r),
                "corr_null_within": pearson(Xt_n, Xi_n),
                "Xt_real_seeds": Xt_r, "Xi_real_seeds": Xi_r,
                "Xt_null_seeds": Xt_n, "Xi_null_seeds": Xi_n,
            }
            filas.append(fila)
        print(f"[eps={eps}] D={D:.6g} listo ({len(R_GRID)} r-puntos x {SEMILLAS} semillas)",
              file=sys.stderr, flush=True)

    # --- análisis agregado ---
    all_Xt_real = [v for f in filas for v in f["Xt_real_seeds"]]
    all_Xi_real = [v for f in filas for v in f["Xi_real_seeds"]]
    all_Xt_null = [v for f in filas for v in f["Xt_null_seeds"]]
    all_Xi_null = [v for f in filas for v in f["Xi_null_seeds"]]

    corr_global_real = pearson(all_Xt_real, all_Xi_real)
    corr_global_null = pearson(all_Xt_null, all_Xi_null)

    corr_por_eps = []
    for eps in EPS_LIST:
        filas_eps = [f for f in filas if f["eps"] == eps]
        xt = [v for f in filas_eps for v in f["Xt_real_seeds"]]
        xi = [v for f in filas_eps for v in f["Xi_real_seeds"]]
        xtn = [v for f in filas_eps for v in f["Xt_null_seeds"]]
        xin = [v for f in filas_eps for v in f["Xi_null_seeds"]]
        corr_por_eps.append({
            "eps": eps,
            "corr_real": pearson(xt, xi),
            "corr_null": pearson(xtn, xin),
            "Xt_real_mean": float(np.mean(xt)), "Xi_real_mean": float(np.mean(xi)),
            "Xt_null_mean": float(np.mean(xtn)), "Xi_null_mean": float(np.mean(xin)),
        })

    # NULL "muerde" si real >> null en magnitud para ambas medidas (T4)
    null_muerde_termo = float(np.mean(all_Xt_real)) > float(np.mean(all_Xt_null))
    null_muerde_info = float(np.mean(all_Xi_real)) > float(np.mean(all_Xi_null))

    elapsed = time.time() - t0
    ts_fin = time.strftime("%Y-%m-%d %H:%M:%S %Z")

    # redondeo legible (se guardan también las seeds crudas para auditoría)
    def rnd_fila(f):
        out = dict(f)
        for k in ("H", "D", "r_eff", "Xt_real_mean", "Xt_real_std", "Xi_real_mean",
                  "Xi_real_std", "Xt_null_mean", "Xt_null_std", "Xi_null_mean",
                  "Xi_null_std", "corr_real_within", "corr_null_within"):
            if isinstance(out.get(k), float):
                out[k] = round(out[k], 6)
        return out

    result = {
        "experimento": "E5.6-1 doble medida (X_termo vs X_info)",
        "ts_inicio": ts_inicio,
        "ts_fin": ts_fin,
        "elapsed_s": elapsed,
        "N": N,
        "semillas": SEMILLAS,
        "eps_list": EPS_LIST,
        "r_grid_n": len(R_GRID),
        "r_grid": R_GRID,
        "pasos_fijo": pasos_fijo,
        "calibracion_ref": {k: v for k, v in cal_ref.items() if k != "tiempos"},
        "meta_por_eps": meta_por_eps,
        "corr_global_real": corr_global_real,
        "corr_global_null": corr_global_null,
        "null_muerde_Xtermo": null_muerde_termo,
        "null_muerde_Xinfo": null_muerde_info,
        "Xt_real_mean_global": float(np.mean(all_Xt_real)),
        "Xi_real_mean_global": float(np.mean(all_Xi_real)),
        "Xt_null_mean_global": float(np.mean(all_Xt_null)),
        "Xi_null_mean_global": float(np.mean(all_Xi_null)),
        "corr_por_eps": corr_por_eps,
        "veredicto_pass_umbral_0.9": bool(corr_global_real > 0.9),
        "filas": [rnd_fila(f) for f in filas],
    }

    out_json = OUT / "E5_6_1_resultado.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[archivo] {out_json}", file=sys.stderr)
    print(f"[corr_global_real] {corr_global_real:.4f}  [corr_global_null] {corr_global_null:.4f}",
          file=sys.stderr)
    print(f"[null_muerde_Xtermo]={null_muerde_termo} [null_muerde_Xinfo]={null_muerde_info}",
          file=sys.stderr)
    print(f"[elapsed] {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
