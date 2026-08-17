"""
CG001 — CONTROL: ¿el nicho premia la PERSISTENCIA o un PICO COHERENTE? · iPad
=============================================================================
Misma fisica que cg001_field.py — NO se toca la dinamica.
Cierra la duda que dejo el test de causalidad (INFORME_CAUSALIDAD_EPSILON.md):
el nicho sostiene a epsilon, ¿pero por ser PERSISTENTE o solo por ser un pico
coherente de una celda?

DISEÑO: epsilon se pone en el campo inicial y se REMUEVE tras t_remove pasos
(se resincroniza phi_B = phi_A; solo queda en la memoria m lo que deposito mientras
estuvo). Se barre t_remove = 1, 5, 20, 100, y "persistente" (nunca removida).
A = sin epsilon (misma semilla emparejada, lockstep).

OBSERVABLE: huella = max|m_B - m_A| al t final, y su localizacion (dist a epsilon).

LECTURA (fijada antes del numero):
  - huella(t_remove=1) ~ huella(persistente)  -> el nicho se traba con UN pico coherente;
    la persistencia NO es necesaria. (premia el pico)
  - huella crece con t_remove y solo satura cerca de persistente con t_remove grande
    -> el nicho NECESITA que epsilon persista en phi. (premia la persistencia)
  - La curva huella(t_remove) da el "tiempo de persistencia" que el nicho requiere.

Solo gamma=8 (con gamma=0 ya sabemos que todo se borra).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
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
PRODUCCION = True       # True -> L=64, pasos=400. False -> L=48, pasos=200 (rapida)
N_SEMILLAS = 30         # (>=30, #109)
T_REMOVES = [1, 5, 20, 100, None]   # None = persistente (nunca removida)
# =============================================================================


@dataclass(frozen=True)
class FieldConfig:
    L: int = 48
    pasos: int = 300
    lam: float = 0.50
    sigma: float = 1.0
    gamma: float = 8.0      # nicho ON (el control es sobre persistencia, no sobre gamma)
    decay: float = 0.97
    eps: float = 0.05
    ruido: float = 1.0


def _step(phi, m, cfg):
    """Un paso de la dinamica — identico para A y para cada B (sin sesgo)."""
    a = phi - gaussian_filter(phi, cfg.sigma)
    m = cfg.decay * m + np.abs(a)
    lam_eff = cfg.lam / (1.0 + cfg.gamma * m)
    phi = phi - lam_eff * a
    return phi, m


def correr_curva(seed, cfg, pos_eps, t_removes):
    """
    Lockstep: A (sin eps) y varias B (con eps removida a distintos t_remove), TODAS sobre
    el MISMO ruido inicial (misma semilla). Devuelve m_A y {t_remove: m_B}.
    """
    L = cfg.L
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, cfg.ruido, size=(L, L, L)).astype(np.float64)  # ruido comun
    phiA = base.copy()
    mA = np.zeros_like(base)
    phiB = {}
    mB = {}
    for tr in t_removes:
        b = base.copy()
        b[pos_eps[0], pos_eps[1], pos_eps[2]] += cfg.eps   # epsilon (#44,#133)
        phiB[tr] = b
        mB[tr] = np.zeros_like(base)

    for t in range(cfg.pasos):
        phiA, mA = _step(phiA, mA, cfg)
        for tr in t_removes:
            phiB[tr], mB[tr] = _step(phiB[tr], mB[tr], cfg)
        # REMOCION de epsilon: resync phi_B = phi_A (solo queda la memoria ya depositada).
        for tr in t_removes:
            if tr is not None and t == tr - 1:
                phiB[tr] = phiA.copy()
    return mA, mB


def dist_toroidal(p, q, L):
    s = 0.0
    for a, b in zip(p, q):
        d = abs(int(a) - int(b))
        d = min(d, L - d)
        s += d * d
    return s ** 0.5


def persistencia(cfg, n_seeds, t_removes):
    L = cfg.L
    pos = (L // 2, L // 2, L // 2)   # centro (ya mostramos que sigue a eps en las 4 posiciones)
    seeds = list(range(1, n_seeds + 1))
    print(f"=== CG001 — CONTROL persistencia vs pico coherente (backend {_BACKEND}) ===")
    print(f"L={L} pasos={cfg.pasos} RUIDO={cfg.ruido} EPS={cfg.eps} gamma={cfg.gamma} "
          f"semillas={n_seeds} eps en {pos}")
    print(f"t_remove barridos: {['persist' if x is None else x for x in t_removes]}\n")

    huellas = {tr: [] for tr in t_removes}
    dists = {tr: [] for tr in t_removes}
    ruta = f"cg001_persistencia_{time.strftime('%Y%m%d_%H%M%S')}_n{n_seeds}"
    t0 = time.time()
    for i, s in enumerate(seeds):
        mA, mB = correr_curva(s, cfg, pos, t_removes)
        for tr in t_removes:
            D = np.abs(mB[tr] - mA)
            peak = np.unravel_index(int(np.argmax(D)), D.shape)
            huellas[tr].append(float(D.max()))
            dists[tr].append(dist_toroidal(peak, pos, L))
        el = time.time() - t0
        eta = el / (i + 1) * (n_seeds - i - 1)
        print(f"  semilla {i+1}/{n_seeds} · {el/60:.1f} min · ETA ~{eta/60:.1f} min")

    h_persist = float(np.median(huellas[None]))
    print("\n" + "-" * 64)
    print(f"{'t_remove':>9} | {'huella (mediana)':>18} | {'huella/persist':>14} | {'dist pico->eps':>13}")
    print("-" * 64)
    resumen = {}
    for tr in t_removes:
        hmed = float(np.median(huellas[tr]))
        dmed = float(np.median(dists[tr]))
        rel = hmed / (h_persist + 1e-300)
        etq = "persist" if tr is None else str(tr)
        resumen[etq] = {"huella_mediana": hmed, "huella_rel_persist": rel, "dist_mediana": dmed}
        print(f"{etq:>9} | {hmed:>18.4g} | {rel:>14.3f} | {dmed:>13.2f}")
    print("-" * 64)

    # ---- veredicto ----
    primero = [x for x in t_removes if x is not None][0]
    rel1 = float(np.median(huellas[primero])) / (h_persist + 1e-300)
    print("VEREDICTO:")
    if rel1 >= 0.7:
        print(f"  PICO COHERENTE — con eps presente solo {primero} paso(s) la huella ya es "
              f"{rel1:.0%} de la persistente. El nicho se TRABA con un pico coherente; "
              f"la persistencia de eps en phi NO es necesaria.")
    elif rel1 <= 0.2:
        print(f"  PERSISTENCIA — con eps {primero} paso(s) la huella es solo {rel1:.0%} de la "
              f"persistente, y crece con t_remove. El nicho NECESITA que eps persista en phi. "
              f"'El nicho premia la persistencia' queda clavado.")
    else:
        print(f"  INTERMEDIO — huella(t_remove={primero}) = {rel1:.0%} de persistente. "
              f"Mira la curva: el nicho se traba en algun tiempo de persistencia finito.")

    with open(ruta + ".json", "w", encoding="utf-8") as f:
        json.dump({"cfg": cfg.__dict__, "pos_eps": list(pos), "resumen": resumen}, f, indent=2)
    print(f"\nGuardado: {ruta}.json  (Files -> On My iPad -> Carnets)")


def main():
    cfg = FieldConfig(L=64, pasos=400) if PRODUCCION else FieldConfig(L=48, pasos=200)
    persistencia(cfg, N_SEMILLAS, T_REMOVES)


if __name__ == "__main__":
    main()
