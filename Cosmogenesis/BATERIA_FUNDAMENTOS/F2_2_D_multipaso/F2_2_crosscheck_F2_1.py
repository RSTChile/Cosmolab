#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2_2_crosscheck_F2_1.py — Verificación cruzada literal pedida por el documento:
"el r* de F2-1 recalculado con D multi-paso debe acercarse o no a 1"

Este script NO edita F2_1_rstar_fino/ (solo lee su JSON de resultado y su función
estimar_rstar, importada, no reimplementada). Al momento del pre-registro de F2-2
(09:30 UTC) F2-1 no tenía resultado en disco (declarado en PROTOCOLO_F2-2_PREREGISTRO.md
sec.7); terminó mientras F2-2 corría. Este script hace la comparación literal ahora que
existe: mismo N=200, eps=1e-3, semillas=16, pasos_fijo=6095 en ambos (physically
idéntico), D_medido de F2-1 (0.0008413667864351009) coincide con mi D_1 (diff ~1e-14),
confirmando que ambos agentes midieron LA MISMA cantidad de la misma manera.

Método: se toman las filas crudas de F2-1 (r_target, H, P_real_media) SIN TOCAR los
valores de P (no se re-corre nada — H ya define la dinámica, igual que en el motor de
F2-2). Se re-eqtiqueta r = H / D_k usando las D_k (k=1,2,5,10,50) que F2-2 midió para
N=200, eps=1e-3 (mismo valor exacto que la combinación 'primario_N200_eps1e-3' de F2-2,
archivo F2_2_resultado_primario_N200_eps1e-3.json). Se reaplica ESTIMAR_RSTAR de F2-1
tal cual (import, no reimplementado) a cada re-etiquetado.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

F2_1_DIR = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/BATERIA_FUNDAMENTOS/F2_1_rstar_fino")
F2_2_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(F2_1_DIR))
from F2_1_motor import estimar_rstar  # noqa: E402  (import de solo lectura, no se edita F2-1)

K_SCALES = [1, 2, 5, 10, 50]


def main():
    f2_1 = json.loads((F2_1_DIR / "F2_1_N200_eps0p001_resultado.json").read_text())
    f2_2 = json.loads((F2_2_DIR / "F2_2_resultado_primario_N200_eps1e-3.json").read_text())

    D1_f2_1 = f2_1["D_medido"]
    D1_f2_2 = f2_2["D1_ref_grid"]
    Dk_f2_2 = {int(k): v["D_A_mean"] for k, v in f2_2["D_k_multiescala"].items() if k.isdigit()}

    print(f"[cross-check] D1 de F2-1 = {D1_f2_1!r}")
    print(f"[cross-check] D1 de F2-2 (mismo N,eps) = {D1_f2_2!r}")
    print(f"[cross-check] |diff| = {abs(D1_f2_1 - D1_f2_2):.3e}  (deben coincidir: misma cantidad, misma física)")
    print(f"[cross-check] N F2-1={f2_1['N']} eps F2-1={f2_1['eps']} semillas F2-1={f2_1['semillas']} pasos F2-1={f2_1['pasos_fijo']}")
    print(f"[cross-check] N F2-2={f2_2['N']} eps F2-2={f2_2['eps']} semillas F2-2={f2_2['semillas']} pasos F2-2={f2_2['pasos']}")
    print()

    filas_f2_1 = f2_1["filas"]
    print(f"[cross-check] F2-1 grid: {len(filas_f2_1)} puntos, r_target hasta {max(f['r_target'] for f in filas_f2_1)}")
    print(f"[cross-check] F2-1 rstar_metricas ORIGINALES (k=1, D de un paso): {json.dumps(f2_1['rstar_metricas'], indent=2)}")
    print()

    resultado = {"D1_f2_1": D1_f2_1, "D1_f2_2_mismo_NEps": D1_f2_2,
                 "rstar_original_F2_1_k1": f2_1["rstar_metricas"], "por_escala": {}}

    for k in K_SCALES:
        Dk = Dk_f2_2[k]
        # re-etiquetar: mismas P_real_media/H de F2-1, nuevo r = H/Dk
        filas_relabel = []
        for f in filas_f2_1:
            H = f["H"]
            r_new = 0.0 if f["r_target"] == 0.0 else (H / Dk if Dk > 0 else float("inf"))
            filas_relabel.append({
                "r_target": r_new,
                "H": H,
                "r_eff": r_new,
                "P_real_media": f["P_real_media"],
            })
        rstar_k = estimar_rstar(filas_relabel)
        resultado["por_escala"][k] = {"D_k": Dk, "rstar_metricas": rstar_k}
        print(f"[cross-check] k={k:>2} D_{k}={Dk:.8g}  r_half_rise={rstar_k['r_half_rise']}  "
              f"r_P_gt_0.5={rstar_k['r_P_gt_0.5']}  r_pendiente_maxima={rstar_k['r_pendiente_maxima']}")

    out = F2_2_DIR / "F2_2_crosscheck_F2_1_resultado.json"
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[cross-check] guardado {out}")


if __name__ == "__main__":
    main()
