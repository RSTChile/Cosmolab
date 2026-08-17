"""
CG002 — Barrido de constantes emergentes (N cosmos independientes)
==================================================================
Minería de invariantes vs historia. Misma física que cg002_experimentos_arco.py
(campo medio O(N), θ_CP=0, sin semillas impuestas en layout ni spin).

USO:
  python3 cg002_constantes_1000.py
  python3 cg002_constantes_1000.py --desde 501 --hasta 1000   # reanudar / repartir

Salida:
  cg002_constantes_1000.csv   (una fila por cosmos, append incremental)
  cg002_constantes_1000_resumen.txt
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---- modelo (idéntico a cg002_experimentos_arco.py) ----
ETA, MU, KAPPA_S, S0, S_BAND, K = 0.05, 0.01, 1e-6, 1.0, 8.0, 8
N_DIF, D_FIRMA, PASOS = 2000, 3, 240
THETA_CP, ALPHA = 0.0, 1.0

CSV_PATH = Path(__file__).with_name("cg002_constantes_1000.csv")
RESUMEN_PATH = Path(__file__).with_name("cg002_constantes_1000_resumen.txt")

FIELDS = [
    "seed", "f_persiste", "orden_a1", "dir_x", "dir_y", "dir_z",
    "S_mean_vivos", "S_max_vivos", "S_total_vivos",
    "entropia_orient", "dim_participation", "rango_C",
    "ratio_orden_inicial", "frac_coop_inicial",
    "elapsed_s",
]


def sat(S):
    return S / (1.0 + S / S_BAND)


def rot(d, th):
    R = np.eye(d)
    c, s = np.cos(th), np.sin(th)
    R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return R


def firmas(N, d, seed):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((N, d))
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def evolucion_campo_medio(U, theta=0.0, alpha=1.0, pasos=PASOS, mu=MU, eta=ETA):
    N, d = U.shape
    R = rot(d, theta)
    UR = U @ R
    selfd = np.einsum("ij,ij->i", UR, U)
    S = np.full(N, S0)
    alive = np.ones(N, bool)
    for _ in range(pasos):
        S = np.where(alive, S * (1 - mu), S)
        m = np.where(alive, np.sqrt(sat(S)), 0.0)
        v = (m[:, None] * U).sum(0)
        dS = alpha * eta * m * (UR @ v) - alpha * eta * m * m * selfd
        S = S + dS
        S = np.minimum(S, 1e12)
        S = np.where(S < 0, 0, S)
        alive = S > KAPPA_S
        S = np.where(alive, S, 0.0)
    return S, alive


def entropia_orientacion(U_alive, nb=16):
    if len(U_alive) < 2:
        return 0.0
    th = np.arccos(np.clip(U_alive[:, 2], -1, 1))
    ph = np.arctan2(U_alive[:, 1], U_alive[:, 0])
    h, _, _ = np.histogram2d(th, ph, bins=[nb, nb])
    p = h[h > 0] / h.sum()
    return float(-(p * np.log(p + 1e-30)).sum())


def dim_participation(C):
    ev = np.abs(np.linalg.eigvalsh((C + C.T) / 2))
    ev = ev[ev > 1e-9]
    if ev.size == 0:
        return 0.0, 0
    d_eff = float((ev.sum() ** 2) / (np.sum(ev ** 2)))
    rango = int(np.sum(ev > 1e-9))
    return d_eff, rango


def cosmos(seed):
    t0 = time.monotonic()
    U = firmas(N_DIF, D_FIRMA, seed)
    C = U @ U.T
    np.fill_diagonal(C, 0)
    frac_coop = float((C > 0).sum() / (N_DIF * (N_DIF - 1)))
    orden_ini = float(np.linalg.norm(U.mean(0)))

    S, alive = evolucion_campo_medio(U, THETA_CP, ALPHA)
    idx = np.where(alive)[0]
    n_vivos = len(idx)
    f = n_vivos / N_DIF

    if n_vivos == 0:
        row = dict(
            seed=seed, f_persiste=0.0, orden_a1=0.0,
            dir_x=0.0, dir_y=0.0, dir_z=0.0,
            S_mean_vivos=0.0, S_max_vivos=0.0, S_total_vivos=0.0,
            entropia_orient=0.0, dim_participation=0.0, rango_C=0,
            ratio_orden_inicial=0.0, frac_coop_inicial=frac_coop,
            elapsed_s=time.monotonic() - t0,
        )
        return row

    Us = U[idx]
    Ss = S[idx]
    v = Us.mean(0)
    a1 = float(np.linalg.norm(v))
    vhat = v / (a1 + 1e-30)
    Cs = Us @ Us.T
    np.fill_diagonal(Cs, 0)
    d_eff, rango = dim_participation(Cs)

    row = dict(
        seed=seed,
        f_persiste=f,
        orden_a1=a1,
        dir_x=float(vhat[0]),
        dir_y=float(vhat[1]),
        dir_z=float(vhat[2]),
        S_mean_vivos=float(Ss.mean()),
        S_max_vivos=float(Ss.max()),
        S_total_vivos=float(Ss.sum()),
        entropia_orient=entropia_orientacion(Us),
        dim_participation=d_eff,
        rango_C=rango,
        ratio_orden_inicial=a1 / (orden_ini + 1e-30),
        frac_coop_inicial=frac_coop,
        elapsed_s=time.monotonic() - t0,
    )
    return row


def cargar_hechas():
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open(newline="") as f:
        r = csv.DictReader(f)
        return {int(row["seed"]) for row in r}


def escribir_fila(row, nuevo_archivo):
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if nuevo_archivo:
            w.writeheader()
        w.writerow({k: row[k] for k in FIELDS})


def cv_pct(a):
    m = np.mean(a)
    if abs(m) < 1e-15:
        return float("nan")
    return 100.0 * np.std(a) / abs(m)


def angulo_medio_dirs(dirs):
    n = len(dirs)
    if n < 2:
        return float("nan")
    angs = []
    for i in range(n):
        for j in range(i + 1, n):
            angs.append(np.degrees(np.arccos(np.clip(dirs[i] @ dirs[j], -1, 1))))
    return float(np.mean(angs))


def resumen_desde_csv():
    rows = []
    with CSV_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(row[k]) if k != "seed" else int(row[k]) for k in FIELDS})
    if not rows:
        return "Sin datos."

    seeds = np.array([r["seed"] for r in rows])
    lines = [
        f"CG002 constantes emergentes — barrido N={len(rows)} cosmos",
        f"Modelo: N_dif={N_DIF}, d_firma={D_FIRMA}, pasos={PASOS}, theta_CP={THETA_CP}",
        f"CSV: {CSV_PATH.name}",
        f"Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    numeric = [
        "f_persiste", "orden_a1", "S_mean_vivos", "S_max_vivos", "S_total_vivos",
        "entropia_orient", "dim_participation", "rango_C", "ratio_orden_inicial",
        "frac_coop_inicial",
    ]
    lines.append("--- Cantidades agregables (ley vs historia) ---")
    lines.append(f"{'cantidad':<22} {'media':>10} {'std':>10} {'CV%':>8}  veredicto")
    for key in numeric:
        a = np.array([r[key] for r in rows])
        cv = cv_pct(a)
        ver = "CONSTANTE" if cv < 5 else ("moderada" if cv < 15 else "contingente")
        lines.append(f"{key:<22} {np.mean(a):10.5f} {np.std(a):10.5f} {cv:7.2f}%  {ver}")

    dirs = np.array([[r["dir_x"], r["dir_y"], r["dir_z"]] for r in rows if r["f_persiste"] > 0])
    if len(dirs) > 1:
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / norms
        ang = angulo_medio_dirs(dirs)
        lines += [
            "",
            "--- Dirección (historia contingente) ---",
            f"ángulo medio entre cosmos (vivos): {ang:.1f}°  (azar teórico ≈ 90°)",
            f"componentes media: x={dirs[:,0].mean():.4f} y={dirs[:,1].mean():.4f} z={dirs[:,2].mean():.4f}",
        ]

    # Razones entre magnitudes (minería π-style)
    f_arr = np.array([r["f_persiste"] for r in rows])
    a1_arr = np.array([r["orden_a1"] for r in rows])
    lines += [
        "",
        "--- Razones (candidatas a invariantes) ---",
        f"f * orden_a1: media={np.mean(f_arr*a1_arr):.5f} CV={cv_pct(f_arr*a1_arr):.2f}%",
        f"orden_a1 / f:  media={np.mean(a1_arr/(f_arr+1e-9)):.5f} CV={cv_pct(a1_arr/(f_arr+1e-9)):.2f}%",
    ]

    # Correlaciones fuertes
    lines += ["", "--- Correlaciones (|r|>0.5) ---"]
    mat = np.column_stack([np.array([r[k] for r in rows]) for k in numeric])
    names = numeric
    found = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = np.corrcoef(mat[:, i], mat[:, j])[0, 1]
            if abs(r) > 0.5:
                lines.append(f"  {names[i]} ↔ {names[j]}: r={r:.3f}")
                found = True
    if not found:
        lines.append("  (ninguna |r|>0.5 entre pares listados)")

    lines += [
        "",
        "--- Criterio ---",
        "CV < 5%  → candidata a CONSTANTE (ley entre universos)",
        "CV ≥ 15% o dirección ~90° entre semillas → HISTORIA (contingente)",
        "Nada post-hoc: cantidades definidas antes del barrido.",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=1)
    ap.add_argument("--hasta", type=int, default=1000)
    args = ap.parse_args()

    hechas = cargar_hechas()
    nuevo = not CSV_PATH.exists()
    pendientes = [s for s in range(args.desde, args.hasta + 1) if s not in hechas]
    print(f"CG002 constantes 1000 | hechas={len(hechas)} pendientes={len(pendientes)}")
    print(f"CSV: {CSV_PATH}")

    t_start = time.monotonic()
    for i, seed in enumerate(pendientes, 1):
        row = cosmos(seed)
        escribir_fila(row, nuevo and i == 1 and len(hechas) == 0)
        nuevo = False
        if i % 50 == 0 or i == len(pendientes):
            elapsed = time.monotonic() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            print(
                f"  [{i}/{len(pendientes)}] seed={seed} "
                f"f={row['f_persiste']:.4f} orden={row['orden_a1']:.4f} "
                f"({rate:.1f} cosmos/s)"
            )

    texto = resumen_desde_csv()
    RESUMEN_PATH.write_text(texto, encoding="utf-8")
    print("\n" + texto)
    print(f"\nResumen guardado: {RESUMEN_PATH}")


if __name__ == "__main__":
    main()