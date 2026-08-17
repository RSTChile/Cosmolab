#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1-4 — Análisis post-hoc del barrido de producción (lee el JSON crudo, no
recalcula ninguna corrida). Aplica el criterio de PASS pre-registrado
(PROTOCOLO_F1-4_PREREGISTRO.md §8) y calcula la dispersión entre familias
(§8, "métrica central") en el grid común (eps,r).

No se ajusta ningún umbral aquí: los umbrales (z>=3, P<0.05, P<0.15, frac>=0.5)
son los mismos fijados en el pre-registro, congelados antes de correr.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "resultados"

Z_THR = 3.0
FRAC_BANDA_THR = 0.5
R_BANDA_MIN = 10.0
EPS_BANDA_MIN = 1e-4


def cargar(modo="produccion"):
    p = RES / f"F1_4_{modo}_resultado.json"
    return json.loads(p.read_text(encoding="utf-8"))


def evaluar_familia(filas):
    banda = [f for f in filas if f["r_target"] >= R_BANDA_MIN and f["eps"] > EPS_BANDA_MIN]
    if not banda:
        return {"z_median_banda": None, "frac_z_ge_3": None, "pass_banda": False, "n_banda": 0}
    zs = np.array([f["z"] for f in banda])
    frac = float(np.mean(zs >= Z_THR))
    return {
        "z_median_banda": float(np.median(zs)),
        "z_mean_banda": float(np.mean(zs)),
        "frac_z_ge_3": frac,
        "pass_banda": frac >= FRAC_BANDA_THR,
        "n_banda": len(banda),
    }


def dispersión_inter_familia(por_familia, familias):
    """Para cada punto (eps,r_target) del grid común, dispersión de P_real
    entre las 6 familias."""
    # index filas por familia -> dict (eps,r_target) -> P_real
    tabla = {}
    for fam in familias:
        for f in por_familia[fam]["filas"]:
            key = (f["eps"], f["r_target"])
            tabla.setdefault(key, {})[fam] = f["P_real"]

    filas_disp = []
    for key, dvals in sorted(tabla.items()):
        if len(dvals) < len(familias):
            continue  # punto incompleto, no se compara
        vals = np.array([dvals[fam] for fam in familias])
        filas_disp.append({
            "eps": key[0],
            "r_target": key[1],
            "P_real_por_familia": dvals,
            "media": float(vals.mean()),
            "std": float(vals.std()),
            "rango": float(vals.max() - vals.min()),
            "min_familia": familias[int(np.argmin(vals))],
            "max_familia": familias[int(np.argmax(vals))],
        })
    return filas_disp


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "produccion"
    d = cargar(modo)
    familias = d["familias"]
    por_familia = d["por_familia"]

    veredicto_por_familia = {}
    for fam in familias:
        info = por_familia[fam]
        banda_eval = evaluar_familia(info["filas"])
        control_r0 = info["control_r0_lava"]
        control_eps0 = info["control_eps0_ok"]
        pass_familia = bool(control_r0 and control_eps0 and banda_eval["pass_banda"])
        veredicto_por_familia[fam] = {
            "control_r0_lava": control_r0,
            "control_r0_detail": info["control_r0_detail"],
            "control_eps0_ok": control_eps0,
            "control_eps0_detail": info["control_eps0_detail"],
            "banda_congelada": banda_eval,
            "pasos_fijo": info["pasos_fijo"],
            "calibracion_tiempos": info["calibracion"]["tiempos"],
            "calibracion_lavo_todas": info["calibracion"]["lavo_todas"],
            "PASS_familia": pass_familia,
        }

    todas_pasan = all(v["PASS_familia"] for v in veredicto_por_familia.values())

    disp = dispersión_inter_familia(por_familia, familias)
    disp_banda = [r for r in disp if r["r_target"] >= R_BANDA_MIN and r["eps"] > EPS_BANDA_MIN]
    if disp_banda:
        stds = np.array([r["std"] for r in disp_banda])
        rangos = np.array([r["rango"] for r in disp_banda])
        disp_resumen = {
            "n_puntos_banda": len(disp_banda),
            "std_media_banda": float(stds.mean()),
            "std_max_banda": float(stds.max()),
            "rango_medio_banda": float(rangos.mean()),
            "rango_max_banda": float(rangos.max()),
            "punto_max_dispersion": max(disp_banda, key=lambda r: r["rango"]),
        }
    else:
        disp_resumen = {}

    analisis = {
        "experimento": "F1-4",
        "modo": modo,
        "criterio_congelado": {
            "Z_THR": Z_THR,
            "FRAC_BANDA_THR": FRAC_BANDA_THR,
            "R_BANDA_MIN": R_BANDA_MIN,
            "EPS_BANDA_MIN": EPS_BANDA_MIN,
        },
        "veredicto_por_familia": veredicto_por_familia,
        "PASS_global_invarianza": todas_pasan,
        "dispersion_inter_familia_banda_resumen": disp_resumen,
        "dispersion_inter_familia_todos_los_puntos": disp,
    }

    out = RES / f"F1_4_{modo}_analisis.json"
    out.write_text(json.dumps(analisis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[analisis] escrito en {out}")
    print(f"PASS_global_invarianza = {todas_pasan}")
    for fam, v in veredicto_por_familia.items():
        print(f"  {fam:18s} PASS={v['PASS_familia']!s:5s} "
              f"r0_lava={v['control_r0_lava']!s:5s} eps0_ok={v['control_eps0_ok']!s:5s} "
              f"frac_z>=3={v['banda_congelada']['frac_z_ge_3']}")
    if disp_resumen:
        print(f"  dispersion (banda r>=10): std_media={disp_resumen['std_media_banda']:.4f} "
              f"rango_medio={disp_resumen['rango_medio_banda']:.4f} "
              f"rango_max={disp_resumen['rango_max_banda']:.4f}")


if __name__ == "__main__":
    main()
