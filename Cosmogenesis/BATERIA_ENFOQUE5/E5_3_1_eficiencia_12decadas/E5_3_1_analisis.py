#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E5_3_1_analisis.py — agrega E5_3_1_resultado_crudo.json (salida del motor) en:
  - curva eficiencia(eps,r): media/std sobre 20 semillas, real y null
  - histograma de TODOS los valores individuales de eficiencia (real), excluyendo eps=0
  - distancia de cada celda (media real) a 4.9% y 31.5% -- SOLO reporte, no selección
  - comparación real vs null (T4): diferencia pareada por semilla, agregada
  - dispersión entre semillas (std) por celda
  - chequeo del guardián de conservación (T6)

No ajusta nada. Se corre DESPUÉS del motor, sobre su salida cruda.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
TARGETS = {"ordinaria_4.9pct": 0.049, "materia_total_31.5pct": 0.315}
CERCA_UMBRAL = 0.02  # 2 puntos porcentuales, umbral de REPORTE (no de seleccion), T2/T3


def main():
    raw_path = OUT / "E5_3_1_resultado_crudo.json"
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    filas = data["filas"]

    # --- separar control eps=0 del grid de 12 decadas ---
    filas_grid = [f for f in filas if f["eps"] > 0]
    filas_ctrl = [f for f in filas if f["eps"] == 0.0]

    curva = []
    todos_valores_real = []
    todos_valores_null = []
    diffs_pareadas = []  # (real - null) por semilla, agregando todo el grid
    n_celdas_cerca = {k: 0 for k in TARGETS}
    celdas_cerca_detalle = {k: [] for k in TARGETS}

    # --- chequeo de degeneracion NULL pedido por CS (via hallazgo de E5.3-5) ---
    # cuenta pares (real,null) por semilla EXACTAMENTE iguales (float ==), y celdas
    # donde z=(mean_real-mean_null)/std_pooled sale exactamente 0 o el pool completo
    # de 20 semillas es degenerado (real==null en TODAS las semillas de la celda).
    n_pares_exactos = 0
    n_celdas_z_exacto_cero = 0
    n_celdas_todas_semillas_exactas = 0
    celdas_degeneradas_detalle = []

    for f in filas_grid:
        eff_r = np.array(f["eficiencia_real"], dtype=float)
        eff_n = np.array(f["eficiencia_null"], dtype=float)
        mask = np.isfinite(eff_r) & np.isfinite(eff_n)
        eff_r_v = eff_r[mask]
        eff_n_v = eff_n[mask]
        todos_valores_real.extend(eff_r_v.tolist())
        todos_valores_null.extend(eff_n_v.tolist())
        diffs_pareadas.extend((eff_r_v - eff_n_v).tolist())

        exact_mask = eff_r_v == eff_n_v
        n_exact_this_cell = int(np.sum(exact_mask))
        n_pares_exactos += n_exact_this_cell
        todas_exactas = bool(len(eff_r_v) > 0 and n_exact_this_cell == len(eff_r_v))
        if todas_exactas:
            n_celdas_todas_semillas_exactas += 1

        mean_r, mean_n = float(np.mean(eff_r_v)), float(np.mean(eff_n_v))
        std_r, std_n = float(np.std(eff_r_v)), float(np.std(eff_n_v))
        sd_pool = np.sqrt((std_r ** 2 + std_n ** 2) / 2.0)
        sd_pool = max(sd_pool, 1.0 / max(len(eff_r_v), 1))
        z_cell = (mean_r - mean_n) / sd_pool if sd_pool > 0 else 0.0
        if z_cell == 0.0:
            n_celdas_z_exacto_cero += 1
        if todas_exactas or n_exact_this_cell > 0:
            celdas_degeneradas_detalle.append(
                {
                    "eps": f["eps"], "r": f["r"], "n_dominios_media": float(np.mean(f["n_dominios"])),
                    "frac_exp_media": float(np.mean(f["frac_exp"])),
                    "n_pares_exactos": n_exact_this_cell, "n_semillas": int(len(eff_r_v)),
                    "todas_semillas_exactas": todas_exactas, "z_cell": float(z_cell),
                }
            )

        fila_res = {
            "eps": f["eps"],
            "r": f["r"],
            "H": f["H"],
            "pasos": f["pasos"],
            "eficiencia_real_media": mean_r if len(eff_r_v) else None,
            "eficiencia_real_std": std_r if len(eff_r_v) else None,
            "eficiencia_null_media": mean_n if len(eff_n_v) else None,
            "eficiencia_null_std": std_n if len(eff_n_v) else None,
            "z_cell": float(z_cell),
            "n_pares_exactos_real_eq_null": n_exact_this_cell,
            "n_semillas_validas": int(len(eff_r_v)),
            "frac_exp_media": float(np.mean(f["frac_exp"])),
            "n_dominios_media": float(np.mean(f["n_dominios"])),
            "guardian_conservacion_ok": f["guardian_conservacion_ok"],
        }
        for k, target in TARGETS.items():
            if fila_res["eficiencia_real_media"] is not None:
                dist = abs(fila_res["eficiencia_real_media"] - target)
                fila_res[f"distancia_{k}"] = dist
                if dist < CERCA_UMBRAL:
                    n_celdas_cerca[k] += 1
                    celdas_cerca_detalle[k].append(
                        {"eps": f["eps"], "r": f["r"], "eficiencia_media": fila_res["eficiencia_real_media"], "distancia": dist}
                    )
        curva.append(fila_res)

    todos_valores_real = np.array(todos_valores_real)
    todos_valores_null = np.array(todos_valores_null)
    diffs_pareadas = np.array(diffs_pareadas)

    # histograma (20 bins entre 0 y 1)
    bins = np.linspace(0, 1, 21)
    hist_counts, _ = np.histogram(todos_valores_real, bins=bins)
    histograma = [
        {"bin_lo": float(bins[i]), "bin_hi": float(bins[i + 1]), "n": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    # comparacion real vs null agregada (T4)
    mean_diff = float(np.mean(diffs_pareadas))
    std_diff = float(np.std(diffs_pareadas))
    se_diff = std_diff / np.sqrt(len(diffs_pareadas)) if len(diffs_pareadas) else float("nan")
    t_stat = mean_diff / se_diff if se_diff > 0 else float("nan")

    # control eps=0
    ctrl_res = []
    for f in filas_ctrl:
        eff_r = np.array(f["eficiencia_real"], dtype=float)
        ctrl_res.append(
            {
                "r": f["r"],
                "eficiencia_real_todas_nan": bool(np.all(np.isnan(eff_r))),
                "E_total_media": float(np.mean(f["E_total"])),
            }
        )

    guardian_ok_global = data["guardian_todas_ok"]

    resumen = {
        "experimento": "E5.3-1",
        "n_celdas_grid_12dec": len(filas_grid),
        "n_valores_individuales": int(len(todos_valores_real)),
        "eficiencia_real_global": {
            "min": float(np.min(todos_valores_real)),
            "max": float(np.max(todos_valores_real)),
            "media": float(np.mean(todos_valores_real)),
            "mediana": float(np.median(todos_valores_real)),
            "std": float(np.std(todos_valores_real)),
        },
        "histograma_20bins": histograma,
        "null_vs_real": {
            "diff_media_real_menos_null": mean_diff,
            "diff_std": std_diff,
            "diff_se": se_diff,
            "t_stat_aprox": t_stat,
            "n_pares": int(len(diffs_pareadas)),
            "nota": "diferencia pareada por semilla, agregada sobre TODA la grilla 13x13; "
            "positivo => REAL > NULL en promedio (los dominios aislados retienen algo mas "
            "de energia estructural que un subconjunto aleatorio del mismo tamano)",
        },
        "degeneracion_null_chequeo": {
            "nota": "Pedido por CS (24-jul, tras hallazgo de E5.3-5): verificar si con >=2 "
            "cortes el 100% de los nodos queda 'aislado' y REAL==NULL exacto en muchas "
            "celdas (esto le paso a la reconstruccion de E5.3-5, hecha ANTES de que este "
            "motor apareciera en disco, sobre una version mas temprana/simplificada de la "
            "definicion -- 'todo dominio con tamano<N es ligado' -- que efectivamente cubre "
            "el 100% del anillo apenas hay 1 corte y por eso el NULL no puede morder. La "
            "version que corrio en este motor ya excluye el dominio mas grande [componente "
            "gigante/percolante] antes de sumar, ver PROTOCOLO_E5.3-1_PREREGISTRO.md #2.4. "
            "Aun asi, en el regimen de fragmentacion alta (muchos dominios de tamano similar) "
            "excluir solo 1 dominio deja una cobertura MUY cercana al 100%, lo que puede "
            "seguir produciendo z~0 aunque no forzosamente exact. Se reporta la cifra real, "
            "sin esconder nada.",
            "n_pares_seed_exactos_real_eq_null": n_pares_exactos,
            "n_pares_seed_total": int(len(diffs_pareadas)),
            "fraccion_pares_exactos": (n_pares_exactos / len(diffs_pareadas)) if diffs_pareadas.size else None,
            "n_celdas_con_al_menos_1_par_exacto": len(celdas_degeneradas_detalle),
            "n_celdas_TODAS_semillas_exactas": n_celdas_todas_semillas_exactas,
            "n_celdas_z_exacto_cero": n_celdas_z_exacto_cero,
            "n_celdas_total_grid": len(filas_grid),
            "detalle_celdas_con_exactos_o_z0": celdas_degeneradas_detalle,
        },
        "cerca_de_blancos": {
            k: {"umbral_pp": CERCA_UMBRAL, "n_celdas_cerca": n_celdas_cerca[k], "detalle": celdas_cerca_detalle[k]}
            for k in TARGETS
        },
        "control_eps0": ctrl_res,
        "guardian_conservacion_todas_ok": guardian_ok_global,
        "curva_completa": curva,
    }

    out_path = OUT / "E5_3_1_resumen.json"
    out_path.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[archivo] {out_path}")
    print(f"eficiencia global: min={resumen['eficiencia_real_global']['min']:.4f} "
          f"max={resumen['eficiencia_real_global']['max']:.4f} "
          f"media={resumen['eficiencia_real_global']['media']:.4f}")
    print(f"guardian_conservacion_todas_ok={guardian_ok_global}")
    print(f"null_vs_real: diff_media={mean_diff:.5f} t~{t_stat:.2f} n_pares={len(diffs_pareadas)}")
    print(f"DEGENERACION: pares exactos real==null = {n_pares_exactos}/{len(diffs_pareadas)} "
          f"| celdas con >=1 par exacto = {len(celdas_degeneradas_detalle)}/{len(filas_grid)} "
          f"| celdas TODAS semillas exactas = {n_celdas_todas_semillas_exactas} "
          f"| celdas z==0 exacto = {n_celdas_z_exacto_cero}")
    for k in TARGETS:
        print(f"cerca de {k}: {n_celdas_cerca[k]} celdas (umbral {CERCA_UMBRAL})")


if __name__ == "__main__":
    main()
