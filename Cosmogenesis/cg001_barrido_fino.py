#!/usr/bin/env python3
"""
CG001 — BARRIDO FINO en la COLA LISA (RUIDO 0.02 -> 0.001)

Enfocado donde el barrido grueso senalo senal en concentracion.
30 semillas por punto (#109), certificacion de signo estable.
L=64, PASOS=400 (produccion iMac/LaCie).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from cg001_field import PRODUCTION, correr, signo_estable

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"

RUIDOS = np.geomspace(0.02, 0.001, 16)
SEMILLAS = list(range(1, 31))
UMBRAL_SIGNO = 0.83


def main() -> None:
    parser = argparse.ArgumentParser(description="CG001 barrido fino cola lisa")
    parser.add_argument("--quick", action="store_true", help="Smoke: 4 puntos, 3 semillas, L=48 pasos=100")
    args = parser.parse_args()

    from cg001_field import FieldConfig

    cfg = PRODUCTION
    ruidos = RUIDOS
    semillas = SEMILLAS
    if args.quick:
        cfg = FieldConfig(L=48, pasos=100)
        ruidos = np.geomspace(0.02, 0.001, 4)
        semillas = [1, 2, 3]
        print(">>> modo quick\n")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = LOGS / f"barrido_fino_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_corr = len(ruidos) * len(semillas) * 2
    print("=== CG001 — barrido FINO COLA LISA ===")
    print(f"L={cfg.L} pasos={cfg.pasos} EPS={cfg.eps} GAMMA={cfg.gamma}")
    print(f"puntos={len(ruidos)} semillas={len(semillas)} corridas={n_corr}\n")
    print(f"{'RUIDO':>8} | {'dif_conc':>10} {'signo':>6} | {'dif_conv':>10} {'signo':>6} | {'dif_exerg':>10}")
    print("-" * 72)

    filas_csv = []
    banda = []

    for i, ruido in enumerate(ruidos):
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
            "opera": opera,
        })
        if opera:
            banda.append(float(ruido))

        if (i + 1) % 4 == 0:
            print(f"  ... {i+1}/{len(ruidos)} puntos")

    print("-" * 72)
    resumen = {
        "tipo": "barrido_fino_cola_lisa",
        "cfg": cfg.__dict__,
        "ruidos": [float(x) for x in ruidos],
        "semillas": semillas,
        "umbral_signo": UMBRAL_SIGNO,
        "banda": banda,
        "filas": filas_csv,
    }
    if banda:
        print(f"BANDA (B!=A estable en concentracion): RUIDO en [{min(banda):.6f}, {max(banda):.6f}]")
        print("Siguiente: verificar que estructura de B esta localizada en la arruga (no global).")
    else:
        print("NO se detecto banda en [0.02, 0.001].")
        print("Honesto: revisar senal del barrido grueso o observable antes de seguir.")

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