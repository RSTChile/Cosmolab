"""
CG001 — BARRIDO de ALTA POTENCIA (hasta 1000 semillas) · iPad
=============================================================
Misma fisica que cg001_field.py — NO se toca la dinamica. Pensado para corridas
largas (cientos/miles de semillas) en iPad, robusto a interrupciones:

  - GUARDA CADA SEMILLA al instante (CSV por fila). Si Carnets se suspende o se
    cae, NO se pierde lo hecho.
  - SE REANUDA SOLO: re-ejecuta la misma celda con el MISMO nombre de LOG y salta
    las (ruido, semilla) ya hechas. Sigue donde quedo.
  - RANGO de semillas: corre 1..500 en el iPad y 501..1000 en el iMac (mismo
    archivo, LOG distinto) y luego juntas los CSV. Las semillas son independientes.
  - Memoria constante: NO acumula campos (a diferencia del test de causalidad).

USO: edita CONFIG, pega en una celda de Carnets, Run All. Si se corta, vuelve a
ejecutar la misma celda: retoma. El veredicto se recalcula desde el CSV cada vez.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import json
import os
import time

import numpy as np

try:
    from scipy.ndimage import gaussian_filter as _gf
    _BACKEND = "scipy"

    def gaussian_filter(phi, sigma):
        return _gf(phi, sigma=sigma, mode="wrap")
except Exception:  # pragma: no cover
    _BACKEND = "numpy"

    def gaussian_filter(phi, sigma, truncate=4.0):
        r = int(truncate * sigma + 0.5)
        d = np.arange(-r, r + 1)
        w = np.exp(-0.5 * (d / sigma) ** 2)
        w /= w.sum()
        out = phi
        for ax in range(phi.ndim):
            acc = np.zeros_like(out)
            for k, dd in enumerate(d):
                acc += w[k] * np.roll(out, -int(dd), axis=ax)
            out = acc
        return out


# ============================ CONFIG (editar aqui) ============================
PRODUCCION    = True       # True -> L=64, pasos=400. False -> L=48, pasos=200
SEMILLA_DESDE = 1          # rango de semillas (para repartir iPad/iMac)
SEMILLA_HASTA = 1000       # inclusive
RUIDOS        = "cola"     # "grueso" (1.0->0.02) | "fino" (0.02->0.001) | "cola" (0.005->0.001, foco)
LOG           = "cg001_1000"   # MISMO nombre para reanudar. Cambia solo si quieres empezar de cero.
# =============================================================================


@dataclass(frozen=True)
class FieldConfig:
    L: int = 48
    pasos: int = 300
    lam: float = 0.50
    sigma: float = 1.0
    gamma: float = 8.0
    decay: float = 0.97
    eps: float = 0.05
    ruido: float = 1.0
    q_nicho: float = 0.999


def correr(con_epsilon, seed, cfg, ruido):
    cfg = replace(cfg, ruido=ruido)
    rng = np.random.default_rng(seed)
    phi = rng.normal(0.0, cfg.ruido, size=(cfg.L, cfg.L, cfg.L)).astype(np.float64)
    if con_epsilon:
        c = cfg.L // 2
        phi[c, c, c] += cfg.eps
    m = np.zeros_like(phi)
    for _ in range(cfg.pasos):
        a = phi - gaussian_filter(phi, cfg.sigma)
        abs_a = np.abs(a)
        m = cfg.decay * m + abs_a
        lam_eff = cfg.lam / (1.0 + cfg.gamma * m)
        phi = phi - lam_eff * a
    thr = float(np.quantile(m, cfg.q_nicho))
    return {
        "concentracion": float(m.max() / (m.mean() + 1e-12)),
        "convertido": float(abs_a[m > thr].sum()),
        "exergia": float(abs_a.sum()),
    }


def signo_estable(difs):
    arr = np.asarray(difs, dtype=np.float64)
    mu = float(arr.mean())
    if abs(mu) < 1e-12:
        return mu, 0.0
    return mu, float((np.sign(arr) == np.sign(mu)).mean())


def ruidos_de(nombre):
    if nombre == "grueso":
        return np.geomspace(1.0, 0.02, 24)
    if nombre == "fino":
        return np.geomspace(0.02, 0.001, 16)
    if nombre == "cola":
        return np.geomspace(0.005, 0.001, 9)
    raise SystemExit(f"RUIDOS desconocido: {nombre!r}")


def main():
    cfg = FieldConfig(L=64, pasos=400) if PRODUCCION else FieldConfig(L=48, pasos=200)
    ruidos = ruidos_de(RUIDOS)
    seeds = list(range(SEMILLA_DESDE, SEMILLA_HASTA + 1))
    csv_path = f"{LOG}.csv"
    cols = ["i", "ruido", "seed", "dif_conc", "dif_conv", "dif_exerg"]

    # --- reanudacion: leer lo ya hecho ---
    acc = {i: {"dc": [], "dv": [], "de": []} for i in range(len(ruidos))}
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                i = int(row["i"]); s = int(row["seed"])
                if SEMILLA_DESDE <= s <= SEMILLA_HASTA and i < len(ruidos):
                    done.add((i, s))
                    acc[i]["dc"].append(float(row["dif_conc"]))
                    acc[i]["dv"].append(float(row["dif_conv"]))
                    acc[i]["de"].append(float(row["dif_exerg"]))

    total = len(ruidos) * len(seeds)
    falta = total - len(done)
    print(f"=== CG001 — barrido alta potencia (backend {_BACKEND}) ===")
    print(f"L={cfg.L} pasos={cfg.pasos} RUIDOS={RUIDOS} ({len(ruidos)} pts) "
          f"semillas={SEMILLA_DESDE}..{SEMILLA_HASTA} ({len(seeds)})")
    print(f"corridas totales={total*2} | ya hechas={len(done)*2} | faltan={falta*2}")
    if falta:
        print(f"ETA aprox a ~3s/corrida: ~{falta*2*3/3600:.1f} h  (se puede cortar y reanudar)")
    print(f"LOG={csv_path}  (reanuda con el mismo nombre)\n")

    nuevo = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline="")
    w = csv.writer(fcsv)
    if nuevo:
        w.writerow(cols)
        fcsv.flush()

    t0 = time.time()
    hechas = 0
    for i, ruido in enumerate(ruidos):
        for s in seeds:
            if (i, s) in done:
                continue
            a = correr(False, s, cfg, float(ruido))
            b = correr(True, s, cfg, float(ruido))
            dc = b["concentracion"] - a["concentracion"]
            dv = b["convertido"] - a["convertido"]
            de = b["exergia"] - a["exergia"]
            w.writerow([i, float(ruido), s, dc, dv, de])
            fcsv.flush()                      # durabilidad: cada semilla en disco
            acc[i]["dc"].append(dc); acc[i]["dv"].append(dv); acc[i]["de"].append(de)
            hechas += 1
            if hechas % 25 == 0:
                el = time.time() - t0
                eta = el / hechas * (falta - hechas)
                print(f"  {hechas}/{falta} nuevas · ruido={ruido:.4f} · "
                      f"{el/60:.1f} min · ETA ~{eta/60:.1f} min")
    fcsv.close()

    # --- veredicto desde TODO el CSV (incluye lo reanudado) ---
    print("\n" + "-" * 70)
    print(f"{'RUIDO':>9} | {'dif_conc':>11} {'signo':>6} | n | {'OPERA':>6}")
    print("-" * 70)
    filas, banda = [], []
    for i, ruido in enumerate(ruidos):
        dc = acc[i]["dc"]
        if not dc:
            continue
        mc, sc = signo_estable(dc)
        opera = sc >= 0.83 and abs(mc) > 1e-3
        print(f"{ruido:>9.4f} | {mc:>+11.4f} {sc:>6.2f} | {len(dc):>3} | "
              f"{'  OPERA' if opera else ''}")
        filas.append({"ruido": float(ruido), "dif_conc_mean": mc, "signo": sc,
                      "n": len(dc), "opera": opera})
        if opera:
            banda.append(float(ruido))
    print("-" * 70)
    print("BANDA estable en concentracion:" ,
          f"[{min(banda):.5f}, {max(banda):.5f}]" if banda else "NINGUNA")
    with open(f"{LOG}_resumen.json", "w") as f:
        json.dump({"cfg": cfg.__dict__, "ruidos": RUIDOS,
                   "semillas": [SEMILLA_DESDE, SEMILLA_HASTA], "filas": filas}, f, indent=2)
    print(f"\nGuardado: {csv_path} (crudo, por semilla) y {LOG}_resumen.json (agregado)")
    print("Files -> On My iPad -> Carnets")


if __name__ == "__main__":
    main()
