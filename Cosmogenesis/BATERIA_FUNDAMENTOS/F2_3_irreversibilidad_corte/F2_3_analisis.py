#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2-3 — Análisis (solo lee F2_3_resultado_crudo.json, no re-corre el motor).

Aplica el criterio de PASS congelado en PROTOCOLO_F2-3_PREREGISTRO.md §6:
  - Spearman(p_recon, P_real) <= -0.8 (rho), evaluado por r >= 1.
  - P_real(p_recon=1.0) dentro de 2 std (entre semillas) de P_null(p_recon=1.0).
  - Veredicto global: PASS si se cumple en >=4 de los 6 r>=1.
Reporta también el observable secundario (std_ratio) y el mecanístico
(frac_exp), y la dispersión entre semillas.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "F2_3_resultado_crudo.json").read_text(encoding="utf-8"))

filas = d["filas"]
r_targets = d["r_targets"]
p_targets = d["p_recon_targets"]

print("=" * 100)
print(f"F2-3 — análisis | corrida {d['ts_inicio']} -> {d['ts_fin']} ({d['elapsed_s']:.1f}s)")
print(f"N={d['N']} eps={d['eps_fijo']} semillas={d['semillas']} pasos_fijo={d['pasos_fijo']} D={d['D_medida']:.6g}")
print("=" * 100)

by_r = {r: [f for f in filas if f["r_target"] == r] for r in r_targets}

r_congelamiento = [1.0, 2.0, 5.0, 10.0, 30.0, 100.0]
veredictos = {}

for r in r_targets:
    rows = sorted(by_r[r], key=lambda f: f["p_recon"])
    p_arr = np.array([f["p_recon"] for f in rows])
    Preal = np.array([f["P_real"] for f in rows])
    Pnull = np.array([f["P_null"] for f in rows])
    Preal_std = np.array([f["P_real_std"] for f in rows])
    Pnull_std = np.array([f["P_null_std"] for f in rows])
    frac = np.array([f["frac_exp_real_mean"] for f in rows])
    sr = np.array([f["std_ratio_real_mean"] for f in rows])

    rho, pval = stats.spearmanr(p_arr, Preal)
    rho_frac, _ = stats.spearmanr(p_arr, frac)
    rho_sr, _ = stats.spearmanr(p_arr, sr)

    # P en p_recon=1 vs NULL en p_recon=1 (criterio de 2 std)
    P1 = Preal[-1]; P1_std = Preal_std[-1]
    N1 = Pnull[-1]; N1_std = Pnull_std[-1]
    combined_std = np.sqrt(P1_std**2 + N1_std**2) / np.sqrt(d["semillas"])
    combined_std = max(combined_std, 1e-6)
    dentro_2std = abs(P1 - N1) <= 2 * combined_std

    monotona_fuerte = (rho <= -0.8) and (pval < 0.01)

    print(f"\n--- r_target={r} (r_eff={rows[0]['r']:.4g}, H={rows[0]['H']:.4g}) ---")
    print(f"  Spearman(p_recon,P_real)  rho={rho:+.4f}  p={pval:.4g}  {'MONOTONA' if monotona_fuerte else 'no-monotona-fuerte'}")
    print(f"  Spearman(p_recon,frac_exp) rho={rho_frac:+.4f}   (frac_exp = fracción de aristas cortadas, mecanismo literal)")
    print(f"  Spearman(p_recon,std_ratio) rho={rho_sr:+.4f}   (observable secundario independiente)")
    print(f"  P_real(0)={Preal[0]:.4f}  P_real(1)={P1:.4f}  P_null(0)={Pnull[0]:.4f}  P_null(1)={N1:.4f}  |P(1)-N(1)|={abs(P1-N1):.4f} vs 2*sd_comb={2*combined_std:.4f} -> {'DENTRO' if dentro_2std else 'FUERA'}")
    print(f"  curva P_real(p_recon): " + " ".join(f"{p:.4g}:{v:.4f}" for p, v in zip(p_arr, Preal)))
    print(f"  curva frac_exp(p_recon): " + " ".join(f"{p:.4g}:{v:.4f}" for p, v in zip(p_arr, frac)))
    print(f"  dispersión entre semillas P_real_std por punto: " + " ".join(f"{v:.4f}" for v in Preal_std))

    if r in r_congelamiento:
        pass_r = monotona_fuerte and dentro_2std
        veredictos[r] = {
            "rho": rho, "pval": pval, "monotona_fuerte": monotona_fuerte,
            "dentro_2std_en_1": dentro_2std, "PASS": pass_r,
        }

print("\n" + "=" * 100)
print("VEREDICTO POR r (régimen r>=1, criterio pre-registrado §6):")
n_pass = 0
for r, v in veredictos.items():
    print(f"  r={r:>6}: rho={v['rho']:+.3f} (p={v['pval']:.3g}) monotona={v['monotona_fuerte']}  "
          f"P(1)~=NULL(1) dentro 2std={v['dentro_2std_en_1']}  -> {'PASS' if v['PASS'] else 'FAIL'}")
    n_pass += int(v["PASS"])

print(f"\n{n_pass}/{len(veredictos)} r>=1 cumplen PASS fuerte (umbral pre-registrado: >=4/6 para PASS global).")
veredicto_global = "PASS" if n_pass >= 4 else "FAIL DEL MECANISMO"
print(f"VEREDICTO GLOBAL F2-3: {veredicto_global}")

print("\n" + "=" * 100)
print("CONTROL r=0 (nada que reconectar -> P debe ser plano en p_recon):")
rows0 = sorted(by_r[0.0], key=lambda f: f["p_recon"])
print("  " + " ".join(f"{f['p_recon']:.4g}:{f['P_real']:.4f}" for f in rows0))
p0 = np.array([f["P_real"] for f in rows0])
print(f"  rango P_real en r=0: [{p0.min():.4f}, {p0.max():.4f}]  (esperado: plano/ruidoso alrededor de ~0.03, SIN tendencia)")

print("\n" + "=" * 100)
print("CONTROL eps=0 (sanity, sin diferencia inicial):")
ctrl = d["control_eps0"]
for f in ctrl["filas"]:
    flag = "OK(=0)" if f["P_real_max"] < 1e-9 else "FALLA"
    print(f"  r={f['r_target']:>6} p_recon={f['p_recon']:.2g}  P_real_mean={f['P_real_mean']:.3e}  P_real_max={f['P_real_max']:.3e}  {flag}")

print("\n" + "=" * 100)
print("Verificación identidad del mecanismo en p_recon=0 vs cs074_rcruz_produccion_resultado.json:")
try:
    ref = json.loads(Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz_produccion_resultado.json").read_text())
    ref_rows = {round(f["r_target"], 6): f for f in ref["filas"] if abs(f["eps"] - 0.1) < 1e-12}
    for r in r_targets:
        rows = by_r[r]
        row0 = [f for f in rows if f["p_recon"] == 0.0][0]
        rr = ref_rows.get(round(r, 6))
        if rr is None:
            print(f"  r={r}: sin fila de referencia (eps=0.1) en el JSON base")
            continue
        dP = abs(row0["P_real"] - rr["P_real"])
        print(f"  r={r}: F2-3(p_recon=0) P_real={row0['P_real']:.6f}  vs  cs074_rcruz P_real={rr['P_real']:.6f}  |diff|={dP:.2e}  {'IDENTICO' if dP < 1e-9 else 'DIFIERE'}")
except FileNotFoundError:
    print("  (no se encontró el JSON de referencia; se omite)")
