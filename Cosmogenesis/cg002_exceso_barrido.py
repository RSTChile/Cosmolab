"""
CG002 — Barrido del exceso dinámico vs nulo geométrico
======================================================
Pregunta única: ¿orden_medido − nulo_geom(d) sobrevive al cambiar (η, μ, S_BAND, d)?

Nulo geométrico (hemisferio uniforme en S^{d-1}):
  nulo(d) = Γ(d/2) / (√π · Γ((d+1)/2))   →  d=3 ⇒ 0.500

USO:
  python3 cg002_exceso_barrido.py
  python3 cg002_exceso_barrido.py --semillas 200

Salida:
  cg002_exceso_barrido.csv
  cg002_exceso_barrido_resumen.txt
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import gamma as gammaf

# ---- baseline (reproducir arco) ----
ETA0, MU0, SBAND0, D0 = 0.05, 0.01, 8.0, 3
KAPPA_S, S0, PASOS, THETA_CP, ALPHA = 1e-6, 1.0, 240, 0.0, 1.0
N_DIF = 2000
N_SEMILLAS = 100

CSV_PATH = Path(__file__).with_name("cg002_exceso_barrido.csv")
RESUMEN_PATH = Path(__file__).with_name("cg002_exceso_barrido_resumen.txt")

FIELDS = [
    "config_id", "param_vary", "eta", "mu", "s_band", "d",
    "n_semillas", "nulo_geom", "f_mean", "f_cv_pct",
    "orden_mean", "exceso_abs", "exceso_rel_pct", "exceso_cv_pct",
    "frac_pos_mean", "elapsed_s",
]


def nulo_geom(d: int) -> float:
    return float(gammaf(d / 2) / (math.sqrt(math.pi) * gammaf((d + 1) / 2)))


def sat(S, s_band):
    return S / (1.0 + S / s_band)


def rot(d, th):
    R = np.eye(d)
    c, s = np.cos(th), np.sin(th)
    R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return R


def firmas(N, d, seed):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((N, d))
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def evolucion(U, eta, mu, s_band, pasos=PASOS):
    N, d = U.shape
    R = rot(d, THETA_CP)
    UR = U @ R
    selfd = np.einsum("ij,ij->i", UR, U)
    S = np.full(N, S0)
    alive = np.ones(N, bool)
    for _ in range(pasos):
        S = np.where(alive, S * (1 - mu), S)
        m = np.where(alive, np.sqrt(sat(S, s_band)), 0.0)
        v = (m[:, None] * U).sum(0)
        dS = ALPHA * eta * m * (UR @ v) - ALPHA * eta * m * m * selfd
        S = S + dS
        S = np.minimum(S, 1e12)
        S = np.where(S < 0, 0, S)
        alive = S > KAPPA_S
        S = np.where(alive, S, 0.0)
    return S, alive


def medir_cosmos(seed, d, eta, mu, s_band):
    U = firmas(N_DIF, d, seed)
    S, alive = evolucion(U, eta, mu, s_band)
    idx = np.where(alive)[0]
    n = len(idx)
    f = n / N_DIF
    if n == 0:
        return f, 0.0, 0.0
    Us = U[idx]
    v = Us.mean(0)
    orden = float(np.linalg.norm(v))
    vhat = v / (orden + 1e-30)
    frac_pos = float((Us @ vhat > 0).mean())
    return f, orden, frac_pos


def cv_pct(a):
    m = np.mean(a)
    if abs(m) < 1e-15:
        return float("nan")
    return 100.0 * np.std(a) / abs(m)


def configs_barrido():
    """OAT desde baseline + baseline explícito."""
    base = dict(eta=ETA0, mu=MU0, s_band=SBAND0, d=D0)
    out = [("baseline", "none", base.copy())]
    for eta in (0.02, 0.05, 0.10):
        c = base.copy()
        c["eta"] = eta
        out.append((f"eta_{eta}", "eta", c))
    for mu in (0.005, 0.01, 0.02):
        c = base.copy()
        c["mu"] = mu
        out.append((f"mu_{mu}", "mu", c))
    for sb in (4.0, 8.0, 16.0):
        c = base.copy()
        c["s_band"] = sb
        out.append((f"sband_{sb}", "s_band", c))
    for d in (2, 3, 4, 5):
        c = base.copy()
        c["d"] = d
        out.append((f"d_{d}", "d", c))
    # dedupe (baseline ya cubre varios valores nominales)
    seen = set()
    deduped = []
    for cid, pv, c in out:
        key = (c["eta"], c["mu"], c["s_band"], c["d"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append((cid, pv, c))
    return deduped


def cargar_hechas():
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open(newline="") as f:
        return {row["config_id"] for row in csv.DictReader(f)}


def correr_config(config_id, param_vary, cfg, n_semillas):
    t0 = time.monotonic()
    d = int(cfg["d"])
    ng = nulo_geom(d)
    fs, ordens, fracps = [], [], []
    for seed in range(1, n_semillas + 1):
        f, orden, fp = medir_cosmos(seed, d, cfg["eta"], cfg["mu"], cfg["s_band"])
        fs.append(f)
        ordens.append(orden)
        fracps.append(fp)
    fs = np.array(fs)
    ordens = np.array(ordens)
    fracps = np.array(fracps)
    excesos = ordens - ng
    row = dict(
        config_id=config_id,
        param_vary=param_vary,
        eta=cfg["eta"],
        mu=cfg["mu"],
        s_band=cfg["s_band"],
        d=d,
        n_semillas=n_semillas,
        nulo_geom=ng,
        f_mean=float(fs.mean()),
        f_cv_pct=cv_pct(fs),
        orden_mean=float(ordens.mean()),
        exceso_abs=float(excesos.mean()),
        exceso_rel_pct=100.0 * float(excesos.mean() / ng) if ng > 0 else float("nan"),
        exceso_cv_pct=cv_pct(excesos),
        frac_pos_mean=float(fracps.mean()),
        elapsed_s=time.monotonic() - t0,
    )
    return row


def escribir_fila(row, header):
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if header:
            w.writeheader()
        w.writerow(row)


def resumen():
    rows = list(csv.DictReader(CSV_PATH.open(newline="")))
    if not rows:
        return "Sin datos."
    lines = [
        "CG002 — Exceso dinámico vs nulo geométrico",
        f"Semillas por config: {rows[0]['n_semillas']}",
        f"Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"{'config':<14} {'d':>2} {'eta':>5} {'mu':>6} {'S_B':>4} "
        f"{'nulo':>6} {'orden':>7} {'exceso':>8} {'ex%':>6} {'f':>6} {'frac+':>6}",
    ]
    excesos = []
    for r in rows:
        lines.append(
            f"{r['config_id']:<14} {int(float(r['d'])):>2} {float(r['eta']):>5.3f} "
            f"{float(r['mu']):>6.4f} {float(r['s_band']):>4.0f} "
            f"{float(r['nulo_geom']):>6.4f} {float(r['orden_mean']):>7.4f} "
            f"{float(r['exceso_abs']):>8.5f} {float(r['exceso_rel_pct']):>5.1f}% "
            f"{float(r['f_mean']):>6.4f} {float(r['frac_pos_mean']):>6.4f}"
        )
        excesos.append(float(r["exceso_abs"]))

    ex = np.array(excesos)
    base = next((r for r in rows if r["config_id"] == "baseline"), rows[0])
    lines += [
        "",
        "--- Veredicto ---",
        f"Baseline exceso: {float(base['exceso_abs']):.5f} ({float(base['exceso_rel_pct']):.1f}% rel)",
        f"Rango exceso en barrido: [{ex.min():.5f}, {ex.max():.5f}]",
        f"CV del exceso entre configs (no semillas): {cv_pct(ex):.1f}%",
        "",
    ]
    if cv_pct(ex) < 10 and ex.min() > 0:
        lines.append("→ Exceso POSITIVO y ESTABLE bajo barrido OAT → candidato a constante de la regla.")
    elif ex.min() > 0 and ex.max() / max(ex.min(), 1e-9) < 2:
        lines.append("→ Exceso positivo, variación moderada → posible constante con dependencia débil de régimen.")
    else:
        lines.append("→ Exceso se MUEVE con parámetros → coeficiente de régimen, no universal fija.")

    lines += [
        "",
        "Controles: frac_pos ≈ 1 (hemisferio); f ≈ ½ salvo cambio fuerte de régimen.",
        "f, orden, f×orden reportados como derivados del hemisferio — no invariantes independientes.",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=N_SEMILLAS)
    args = ap.parse_args()

    hechas = cargar_hechas()
    configs = configs_barrido()
    pendientes = [(cid, pv, c) for cid, pv, c in configs if cid not in hechas]
    nuevo = not CSV_PATH.exists()

    print(f"CG002 exceso barrido | configs={len(configs)} pendientes={len(pendientes)} "
          f"semillas={args.semillas}")
    print(f"nulo_geom: d=2→{nulo_geom(2):.4f} d=3→{nulo_geom(3):.4f} "
          f"d=4→{nulo_geom(4):.4f} d=5→{nulo_geom(5):.4f}")

    for i, (cid, pv, cfg) in enumerate(pendientes, 1):
        row = correr_config(cid, pv, cfg, args.semillas)
        escribir_fila(row, nuevo and len(hechas) == 0 and i == 1)
        hechas.add(cid)
        print(
            f"  [{i}/{len(pendientes)}] {cid}: "
            f"exceso={row['exceso_abs']:+.5f} ({row['exceso_rel_pct']:+.1f}%) "
            f"f={row['f_mean']:.4f} frac_pos={row['frac_pos_mean']:.4f}"
        )

    texto = resumen()
    RESUMEN_PATH.write_text(texto, encoding="utf-8")
    print("\n" + texto)
    print(f"\nCSV: {CSV_PATH}")


if __name__ == "__main__":
    main()