#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5.5-4 — Muerte térmica vs Nada operativa: caracterización de los dos estados
==============================================================================

Motor de este experimento. Pre-registro congelado ANTES de escribir este archivo en:
  PROTOCOLO_E5.5-4_PREREGISTRO.md (mismo directorio).

Reutiliza SIN EDITAR las funciones de bajo nivel de la base de código
`/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py` (campo_inicial,
paso_difusion, paso_expansion, medir_D, medir_pasos_lavado) vía importlib — no se
modifica ni una línea de ese archivo.

Pregunta: en el equilibrio de muerte térmica (difusión pura, sin expansión que aísle
nada, r=0, evolución larga), ¿el vector (E, X, S_ent) confirma EMPÍRICAMENTE — no
supuesto — que el estado tiene E>0 (energía intacta) simultáneamente con X=0 (sin
capacidad de trabajo) y S_ent=máx (entropía máxima)? Y es esto categóricamente distinto
de la Nada operativa ∅ (E=0, el campo no existe)?

Definiciones del vector (E, X, S_ent), congeladas en el pre-registro §4:
  E     = mean(phi)                                    — energía declarada (E1)
  X     = corr(phi, roll(phi,1))_clip0 · Var(phi)/Var(phi_inicial)   — exergía (idéntica
          a persistencia() de la base / E5.1-1)
  S_ent = -sum(p_i log p_i) / log(N), p_i = phi_i / sum(phi)  — entropía de Gibbs/Shannon
          de la DISTRIBUCIÓN de energía por sitio (independiente, método distinto de X)
  W_res = mean(|phi_i - mean(phi)|)                     — capacidad de trabajo residual
          (tercer observable, unidades absolutas)

Implementación de las "dos duraciones" del §5/§8 del pre-registro: en vez de dos
corridas INDEPENDIENTES con el mismo seed (que es también válido), se corre UNA sola
evolución continua hasta pasos_largo y se toma una FOTO del vector en pasos_base (punto
intermedio) y otra al final en pasos_largo. Esto es al menos tan riguroso como el
re-arranque independiente para la pregunta de convergencia ("¿seguir evolucionando
cambia el vector?") y es más eficiente computacionalmente (evita recomputar el primer
tramo dos veces). Se documenta aquí y en el reporte final como aclaración de
implementación, no como desviación del protocolo (el protocolo pedía "dos duraciones,
se reporta si el vector ya no cambia entre ambas" — este método responde exactamente
esa pregunta).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_CODE = HERE.parent.parent / "cs074_rcruz.py"

spec = importlib.util.spec_from_file_location("cs074_rcruz_base", str(BASE_CODE))
cs074 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs074)  # type: ignore

# ---- Parámetros congelados en el pre-registro (§5) --------------------------------
N = 200
EPS_LIST = [0.0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 1.0]
R_LIST = [0.0, 1000.0]
N_SEMILLAS = 20  # >= 16 pedido
DURACION_FACTOR_EXTRA = 4

# Umbrales de PASS (§7), congelados ANTES de correr
UMBRAL_DERIVA_E = 1e-6
UMBRAL_X_MUERTE = 0.05
UMBRAL_SENT_MUERTE = 0.99


def calcular_vector(phi, phi_inicial_std, phi_inicial_var, E_inicial):
    """Calcula (E, X, S_ent, W_res) para un estado phi dado. Fórmulas congeladas §4."""
    E = float(phi.mean())
    deriva_E = float((E - E_inicial) / E_inicial) if E_inicial != 0 else float("nan")

    # X (exergía) — idéntica a persistencia() de la base
    if phi_inicial_var <= 0 or phi.std() <= 1e-12:
        X = 0.0
    else:
        c = np.corrcoef(phi, np.roll(phi, 1))[0, 1]
        if not np.isfinite(c):
            c = 0.0
        c = max(0.0, float(c))
        v = float(phi.var() / phi_inicial_var)
        X = float(c * v)

    # S_ent (entropía Gibbs/Shannon de la distribución de energía por sitio)
    min_phi = float(phi.min())
    clip_aplicado = min_phi < 0.0
    phi_pos = np.clip(phi, 0.0, None) if clip_aplicado else phi
    total = phi_pos.sum()
    if total <= 0:
        S_ent = float("nan")
    else:
        p = phi_pos / total
        mask = p > 0
        S_ent = float(-np.sum(p[mask] * np.log(p[mask])) / np.log(N))

    # W_res (capacidad de trabajo residual, unidades absolutas)
    W_res = float(np.mean(np.abs(phi - phi.mean())))

    return {
        "E": E,
        "deriva_E": deriva_E,
        "X": X,
        "S_ent": S_ent,
        "W_res": W_res,
        "min_phi": min_phi,
        "clip_aplicado": bool(clip_aplicado),
    }


def corrida_dos_duraciones(N, eps, r, seed, pasos_base, pasos_largo):
    """
    Una evolución continua de pasos_largo pasos, con foto del vector en pasos_base
    (checkpoint intermedio) y en pasos_largo (final). H se fija UNA vez al inicio de
    la corrida a partir de D medido con este mismo seed (H=min(r*D,1)).
    """
    rng = np.random.default_rng(seed)
    phi, _ = cs074.campo_inicial(N, eps, rng)
    activo = np.ones(N, dtype=bool)

    E_inicial = float(phi.mean())
    var_inicial = float(phi.var())
    std_inicial = float(phi.std())

    # D medido del propio campo con este seed (independiente del rng de evolución,
    # igual convención que la base: medir_D crea su propio rng interno)
    D = cs074.medir_D(N, eps, seed)
    if D > 0:
        H = float(min(r * D, 1.0))
    else:
        H = 0.0 if r == 0.0 else 1.0

    snapshot_base = None
    for paso in range(1, pasos_largo + 1):
        phi = cs074.paso_difusion(phi, activo)
        activo = cs074.paso_expansion(activo, H, rng)
        if paso == pasos_base:
            snapshot_base = calcular_vector(phi.copy(), std_inicial, var_inicial, E_inicial)

    snapshot_largo = calcular_vector(phi, std_inicial, var_inicial, E_inicial)

    return {
        "eps": eps,
        "r": r,
        "seed": seed,
        "D": D,
        "H": H,
        "frac_exp_final": 1.0 - float(activo.mean()),
        "pasos_base": pasos_base,
        "pasos_largo": pasos_largo,
        "vector_base": snapshot_base,
        "vector_largo": snapshot_largo,
    }


def vector_nada():
    """Comparador Nada (∅): phi≡0, sin evolución, sin campo. §2/§5/§7."""
    phi0 = np.zeros(N, dtype=float)
    E = float(phi0.mean())
    return {
        "E": E,
        "deriva_E": 0.0,
        "X": 0.0,
        "S_ent": None,  # indefinido: sin presupuesto que repartir, no es un límite
                         # numérico de la rama muerte-térmica, es un régimen distinto
        "W_res": 0.0,
        "min_phi": 0.0,
        "clip_aplicado": False,
        "nota": "Nada operativa: ausencia total de campo, E=0 por construcción del "
                "estado de referencia, no por evolución dinámica.",
    }


def main():
    t0 = time.time()
    print(f"[E5.5-4] inicio {time.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr, flush=True)

    cal = cs074.medir_pasos_lavado(N, 1e-3, 8)
    pasos_base = cal["pasos"]
    pasos_largo = pasos_base * DURACION_FACTOR_EXTRA
    print(
        f"[calibracion] pasos_base={pasos_base} (mediana={cal['mediana']}) "
        f"pasos_largo={pasos_largo} lavo_todas={cal['lavo_todas']}",
        file=sys.stderr,
        flush=True,
    )

    filas = []
    total_combos = len(EPS_LIST) * len(R_LIST) * N_SEMILLAS
    i = 0
    for eps in EPS_LIST:
        for r in R_LIST:
            for seed in range(N_SEMILLAS):
                i += 1
                fila = corrida_dos_duraciones(N, eps, r, seed, pasos_base, pasos_largo)
                filas.append(fila)
                if i % 40 == 0 or i == total_combos:
                    dt = time.time() - t0
                    print(
                        f"[progreso] {i}/{total_combos} combos "
                        f"({100.0*i/total_combos:.1f}%) t={dt:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )

    nada = vector_nada()

    result = {
        "experimento": "E5.5-4_muerte_vs_nada",
        "base_code": str(BASE_CODE),
        "N": N,
        "eps_list": EPS_LIST,
        "r_list": R_LIST,
        "n_semillas": N_SEMILLAS,
        "pasos_base": pasos_base,
        "pasos_largo": pasos_largo,
        "calibracion_lavado": cal,
        "umbrales": {
            "deriva_E_max": UMBRAL_DERIVA_E,
            "X_muerte_max": UMBRAL_X_MUERTE,
            "S_ent_muerte_min": UMBRAL_SENT_MUERTE,
        },
        "filas": filas,
        "nada_referencia": nada,
        "elapsed_s": time.time() - t0,
        "timestamp_fin": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_json = HERE / "E5_5_4_resultado_crudo.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[archivo] {out_json}", file=sys.stderr)
    print(f"[elapsed] {result['elapsed_s']:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
