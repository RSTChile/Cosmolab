"""
CG001 — INSTRUMENTO DE CAMPO · VERSION AUTOCONTENIDA PARA iPad (Carnets / Pyto)
==============================================================================
Transcribe la dinamica de `cg001_field.py` (Grok) TAL CUAL — no cambia la fisica.
Junta engine + barridos en UN solo archivo, sin imports relativos ni bash, para
correr en una app de Python del iPad (M1) sobre el chip rapido.

Anexos respecto del original (orquestacion, NO dinamica):
  - gaussian_filter con scipy si existe; si no, equivalente puro-numpy (wrap).
  - guardado INCREMENTAL tras cada punto RUIDO (si el iPad suspende, no se pierde).
  - progreso + tiempos visibles (corrida larga en pantalla tactil).
  - rango de semillas configurable -> permite repartir trabajo iPad <-> iMac.

COMO USAR: editar el bloque CONFIG de abajo y ejecutar todo. Por defecto corre
una DEMO instantanea para verificar que el entorno funciona; luego cambia MODO.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, replace

import numpy as np

# ---- gaussian_filter: scipy si esta (Carnets lo trae); si no, puro-numpy ----
try:
    from scipy.ndimage import gaussian_filter as _gf
    _BACKEND = "scipy"

    def gaussian_filter(phi, sigma):
        return _gf(phi, sigma=sigma, mode="wrap")
except Exception:  # pragma: no cover  (fallback iPad sin scipy)
    _BACKEND = "numpy"

    def gaussian_filter(phi, sigma, truncate=4.0):
        # Gaussiano separable, circular (== scipy mode="wrap"). Kernel simetrico.
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
MODO = "demo"           # "demo" | "grueso" | "fino"
PRODUCCION = True       # True -> L=64, pasos=400 (la corrida de 3h). False -> L=48/300
SEMILLA_DESDE = 1       # para repartir con el iMac: p.ej. iPad 1..15, iMac 16..30
SEMILLA_HASTA = 30      # inclusive
# =============================================================================


@dataclass(frozen=True)
class FieldConfig:
    L: int = 48
    pasos: int = 300
    lam: float = 0.50       # relajacion = costo de sostener diferencia (R0, #119,#128)
    sigma: float = 1.0      # vecindad del kernel isotropico
    gamma: float = 8.0      # memoria protege asimetria (nicho, #131, C-N2.6)
    decay: float = 0.97     # memoria ambiental no eterna (#129)
    eps: float = 0.05       # UNICA arruga inicial (#44, #133)
    ruido: float = 1.0      # amplitud singularidad: eje liso<->rugoso
    q_nicho: float = 0.999  # cuantil para ICES (solo medicion)
    log_every: int = 10


PRODUCTION = FieldConfig(L=64, pasos=400)


def _inicializar_phi(rng, cfg, con_epsilon):
    # SINGULARIDAD (#43,#7): densidad de asimetrias; amplitud = ruido.
    phi = rng.normal(0.0, cfg.ruido, size=(cfg.L, cfg.L, cfg.L)).astype(np.float64)
    if con_epsilon:
        c = cfg.L // 2
        phi[c, c, c] += cfg.eps
    return phi


def _paso(phi, m, cfg):
    # DIFERENCIA = asimetria local (C-N4). Isotropica.
    a = phi - gaussian_filter(phi, cfg.sigma)
    abs_a = np.abs(a)
    exergia = float(abs_a.sum())                       # asimetria total disponible
    m = cfg.decay * m + abs_a                           # historia ambiental (#126)
    lam_eff = cfg.lam / (1.0 + cfg.gamma * m)           # nicho history-dep (#131)
    disipado = float((lam_eff * abs_a).sum())           # IDES
    thr = float(np.quantile(m, cfg.q_nicho))
    mask_nicho = m > thr
    convertido = float(abs_a[mask_nicho].sum())         # ICES
    phi = phi - lam_eff * a                             # relajacion/estiramiento (#119,#128)
    met = {
        "exergia": exergia,
        "disipado": disipado,
        "convertido": convertido,
        "memoria_max": float(m.max()),
        "concentracion": float(m.max() / (m.mean() + 1e-12)),
        "n_nicho": int(mask_nicho.sum()),
    }
    return phi, m, met


def correr(con_epsilon, seed, cfg, ruido=None):
    if ruido is not None:
        cfg = replace(cfg, ruido=ruido)
    rng = np.random.default_rng(seed)
    phi = _inicializar_phi(rng, cfg, con_epsilon)
    m = np.zeros_like(phi)
    entropia = 0.0
    final = None
    for t in range(cfg.pasos):
        phi, m, met = _paso(phi, m, cfg)
        entropia += met["disipado"]
        final = {"t": t, "entropia": entropia, **met}
    return final


def signo_estable(difs):
    arr = np.asarray(difs, dtype=np.float64)
    mu = float(arr.mean())
    if abs(mu) < 1e-12:
        return mu, 0.0
    return mu, float((np.sign(arr) == np.sign(mu)).mean())


# ------------------------------- demo ----------------------------------------
def demo():
    cfg = FieldConfig()
    print(f"=== CG001 campo — demo (backend gaussiano: {_BACKEND}) ===")
    print(f"L={cfg.L} pasos={cfg.pasos} RUIDO={cfg.ruido} EPS={cfg.eps}\n")
    t0 = time.time()
    A = correr(False, 1, cfg)
    B = correr(True, 1, cfg)
    dt = time.time() - t0
    print(f"FLECHA: exergia final A={A['exergia']:.0f}  entropia A={A['entropia']:.0f}")
    print(f"DIVERGENCIA A/B  concentracion: {B['concentracion'] - A['concentracion']:+.4f}")
    print(f"\n2 corridas en {dt:.1f}s  ->  ~{dt/2:.2f}s por corrida en este equipo.")
    print("Si esto corre, el entorno esta listo. Cambia MODO a 'grueso' o 'fino'.")


# ------------------------------ barridos -------------------------------------
def barrido(modo, cfg, ruidos, semillas):
    n = len(ruidos) * len(semillas) * 2
    print(f"=== CG001 — barrido {modo.upper()} (backend {_BACKEND}) ===")
    print(f"L={cfg.L} pasos={cfg.pasos} EPS={cfg.eps} puntos={len(ruidos)} "
          f"semillas={semillas[0]}..{semillas[-1]} corridas={n}")
    print(f"{'RUIDO':>9} | {'dif_conc':>10} {'signo':>6} | {'dif_conv':>10} {'signo':>6} | {'dif_exerg':>10}")
    print("-" * 74)

    ruta = f"cg001_{modo}_{time.strftime('%Y%m%d_%H%M%S')}_s{semillas[0]}-{semillas[-1]}"
    filas = []
    t_ini = time.time()
    for i, ruido in enumerate(ruidos):
        dc, dv, de = [], [], []
        for s in semillas:
            a = correr(False, s, cfg, ruido=float(ruido))
            b = correr(True, s, cfg, ruido=float(ruido))
            dc.append(b["concentracion"] - a["concentracion"])
            dv.append(b["convertido"] - a["convertido"])
            de.append(b["exergia"] - a["exergia"])
        mc, sc = signo_estable(dc)
        mv, sv = signo_estable(dv)
        me = float(np.mean(de))
        opera = sc >= 0.83 and abs(mc) > 1e-3
        print(f"{ruido:>9.4f} | {mc:>+10.4f} {sc:>6.2f} | {mv:>+10.4f} {sv:>6.2f} | "
              f"{me:>+10.4f}{'  <-- OPERA' if opera else ''}")
        filas.append({"ruido": float(ruido), "dif_conc_mean": mc, "dif_conc_signo": sc,
                      "dif_conv_mean": mv, "dif_conv_signo": sv, "dif_exerg_mean": me,
                      "opera": opera})

        # --- guardado INCREMENTAL: si el iPad suspende, lo hecho queda ---
        with open(ruta + ".json", "w", encoding="utf-8") as f:
            json.dump({"modo": modo, "cfg": cfg.__dict__, "filas": filas}, f, indent=2)
        with open(ruta + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)

        hechas = (i + 1) * len(semillas) * 2
        el = time.time() - t_ini
        eta = el / hechas * (n - hechas)
        print(f"   {hechas}/{n} corridas · {el/60:.1f} min · ETA ~{eta/60:.1f} min")

    print("-" * 74)
    banda = [r["ruido"] for r in filas if r["opera"]]
    if banda:
        print(f"BANDA (B!=A estable en concentracion): RUIDO en [{min(banda):.5f}, {max(banda):.5f}]")
    else:
        print("NO se detecto banda estable en concentracion en el rango.")
    print(f"\nGuardado: {ruta}.json / {ruta}.csv  (Files -> On My iPad -> Carnets)")


def main():
    if MODO == "demo":
        demo()
        return
    cfg = PRODUCTION if PRODUCCION else FieldConfig()
    semillas = list(range(SEMILLA_DESDE, SEMILLA_HASTA + 1))
    if MODO == "grueso":
        ruidos = np.geomspace(1.0, 0.02, 24)
    elif MODO == "fino":
        ruidos = np.geomspace(0.02, 0.001, 16)
    else:
        raise SystemExit(f"MODO desconocido: {MODO!r} (usa demo|grueso|fino)")
    barrido(MODO, cfg, ruidos, semillas)


if __name__ == "__main__":
    main()
