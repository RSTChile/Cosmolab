#!/usr/bin/env python3
"""
CG001 — BARRIDO GRUESO del eje singularidad (RUIDO 1.0 -> 0.02)

Recorre el eje liso<->rugoso con multi-semilla por punto.
Observable principal del barrido: divergencia B-A estable en concentracion y convertido.
NO decide que es la singularidad — el campo responde.

288 corridas tipicas: 24 puntos x 6 semillas x A/B.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from cg001_field import FieldConfig, PRODUCTION, correr, signo_estable

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"

# Barrido grueso (primer contacto del eje)
RUIDOS = np.geomspace(1.0, 0.02, 24)
SEMILLAS = list(range(1, 7))
UMBRAL_SIGNO = 0.83


def main() -> None:
    parser = argparse.ArgumentParser(description="CG001 barrido grueso RUIDO 1.0->0.02")
    parser.add_argument("--quick", action="store_true", help="Smoke: 4 puntos, 2 semillas, L=48")
    parser.add_argument("--production", action="store_true", help="L=64, pasos=400")
    args = parser.parse_args()

    cfg = FieldConfig() if not args.production else PRODUCTION
    ruidos = RUIDOS
    semillas = SEMILLAS
    if args.quick:
        ruidos = np.geomspace(1.0, 0.02, 4)
        semillas = [1, 2]
        print(">>> modo quick\n")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = LOGS / f"barrido_grueso_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== CG001 — barrido GRUESO (eje singularidad) ===")
    print(f"L={cfg.L} pasos={cfg.pasos} EPS={cfg.eps} puntos={len(ruidos)} semillas={len(semillas)}")
    print(f"corridas totales: {len(ruidos)*len(semillas)*2}\n")
    print(f"{'RUIDO':>8} | {'dif_conc':>10} {'signo':>6} | {'dif_conv':>10} {'signo':>6} | {'dif_exerg':>10}")
    print("-" * 72)

    filas_csv = []
    banda_conc = []

    for ruido in ruidos:
        dc, dv, de = [], [], []
        for s in semillas:
            a = correr(False, seed=s, cfg=cfg, ruido=float(ruido))
            b = correr(True, seed=s, cfg=cfg, ruido=float(ruido))
            dc.append(b["concentracion"] - a["concentracion"])
            dv.append(b["convertido"] - a["convertido"])
            de.append(b["exergia"] - a["exergia"])

        mc, sc = signo_estable(dc)
        mv, sv = signo_estable(dv)
        me = float(np.mean(de))
        opera = sc >= UMBRAL_SIGNO and abs(mc) > 1e-3
        marca = " <-- OPERA" if opera else ""
        print(f"{ruido:>8.4f} | {mc:>+10.4f} {sc:>6.2f} | {mv:>+10.4f} {sv:>6.2f} | {me:>+10.4f}{marca}")

        filas_csv.append({
            "ruido": float(ruido),
            "dif_conc_mean": mc,
            "dif_conc_signo": sc,
            "dif_conv_mean": mv,
            "dif_conv_signo": sv,
            "dif_exerg_mean": me,
            "opera_conc": opera,
        })
        if opera:
            banda_conc.append(float(ruido))

    print("-" * 72)
    resumen = {
        "tipo": "barrido_grueso",
        "cfg": cfg.__dict__,
        "ruidos": [float(x) for x in ruidos],
        "semillas": semillas,
        "umbral_signo": UMBRAL_SIGNO,
        "banda_concentracion": banda_conc,
        "filas": filas_csv,
    }
    if banda_conc:
        print(f"BANDA (B!=A estable en concentracion): RUIDO en [{min(banda_conc):.4f}, {max(banda_conc):.4f}]")
        print("-> Siguiente: barrido FINO en la cola lisa (cg001_barrido_fino.py).")
    else:
        print("NO se detecto banda estable en concentracion en [1.0, 0.02].")
        print("Revisar observable o extender hacia RUIDO < 0.02.")

    json_path = out_dir / "resultado.json"
    csv_path = out_dir / "resultado.csv"
    json_path.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_csv[0].keys()))
        w.writeheader()
        w.writerows(filas_csv)
    print(f"\nGuardado: {json_path}\n         {csv_path}")


if __name__ == "__main__":
    main()