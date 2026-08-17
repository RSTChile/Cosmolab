"""
CG001 — TEST DE CAUSALIDAD DE epsilon · VERSION iPad (Carnets with scipy)
========================================================================
Misma fisica que cg001_field.py (Grok) — NO se toca la dinamica.
Responde la pregunta que el barrido fino NO podia: en regimen de DENSIDAD MAXIMA
(RUIDO=1.0, miles de asimetrias), ¿epsilon es CAUSAL o es solo deriva caotica?

DISEÑO (pre-registrado con Claude/GPT, no se decide post-hoc):
  - Regimen: RUIDO=1.0 fijo. NO se barre hacia liso (ahi concentracion = artefacto ratio).
  - Mover la arruga: epsilon en distintas posiciones (centro, esquina, 2 random).
    A = sin epsilon (misma semilla emparejada). B = con epsilon en la posicion p.
  - Observable causal: campo de diferencia D = |m_B - m_A| (misma semilla -> difieren
    SOLO por epsilon). ¿El PICO de D cae cerca de donde se puso epsilon, y se MUEVE
    cuando epsilon se mueve?
  - Baseline de caos: |m_A(s1) - m_A(s2)| (dos semillas distintas, SIN epsilon). Da el
    nivel de azar: donde cae el pico cuando algo que NO es epsilon cambia el campo.

CRITERIO DE LECTURA (fijado ANTES del numero):
  CAUSAL : mediana dist(pico|m_B-m_A|, epsilon) < RADIO en >=83% de semillas (#109),
           Y mucho menor que la dist del baseline de caos.  -> epsilon OPERA, siembra localizable.
  CAOS   : dist ~ baseline (el pico cae lejos de epsilon, como al cambiar de semilla).
           -> sensibilidad a condiciones iniciales, mas debil que "epsilon opera".
  Se lee al t FINAL (a t bajo el pico esta pegado a epsilon trivialmente, aun no propaga).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
PRODUCCION = True       # True -> L=64, pasos=400. False -> L=48, pasos=200 (prueba rapida)
N_SEMILLAS = 30         # por posicion (>=30, #109). Sube a 100 si quieres mas potencia.
RADIO = 3.0             # celdas: "el pico esta EN la arruga" si dist < RADIO
GAMMA = 8.0             # nicho (#131). Corre 1ro con 8.0; despues con 0.0 = CONTROL nicho-OFF:
                        #   si con 0.0 el pico SIGUE pegado a eps -> era localidad trivial.
                        #   si con 0.0 el pico se dispersa/desaparece -> el nicho hace que eps PERSISTA (causal fuerte).
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
    ruido: float = 1.0      # DENSIDAD MAXIMA — fijo, no se barre
    q_nicho: float = 0.999


def _inicializar_phi(rng, cfg, con_epsilon, pos_eps):
    phi = rng.normal(0.0, cfg.ruido, size=(cfg.L, cfg.L, cfg.L)).astype(np.float64)
    if con_epsilon:
        if pos_eps is None:
            c = cfg.L // 2
            pos_eps = (c, c, c)
        phi[pos_eps[0], pos_eps[1], pos_eps[2]] += cfg.eps
    return phi


def correr_m(con_epsilon, seed, cfg, pos_eps=None):
    """Corre la dinamica y devuelve el campo de memoria FINAL m (L^3)."""
    rng = np.random.default_rng(seed)
    phi = _inicializar_phi(rng, cfg, con_epsilon, pos_eps)
    m = np.zeros_like(phi)
    for _ in range(cfg.pasos):
        a = phi - gaussian_filter(phi, cfg.sigma)          # asimetria local (C-N4)
        abs_a = np.abs(a)
        m = cfg.decay * m + abs_a                            # historia (#126)
        lam_eff = cfg.lam / (1.0 + cfg.gamma * m)            # nicho (#131)
        phi = phi - lam_eff * a                              # relajacion (#119,#128)
    return m


def dist_toroidal(p, q, L):
    """Distancia en el toro (mode='wrap' -> sin bordes privilegiados)."""
    s = 0.0
    for a, b in zip(p, q):
        d = abs(int(a) - int(b))
        d = min(d, L - d)
        s += d * d
    return s ** 0.5


def causalidad(cfg, n_seeds):
    L = cfg.L
    rng_pos = np.random.default_rng(12345)  # posiciones reproducibles
    posiciones = {
        "centro":  (L // 2, L // 2, L // 2),
        "esquina": (3, 3, 3),
        "rand1":   tuple(int(x) for x in rng_pos.integers(0, L, 3)),
        "rand2":   tuple(int(x) for x in rng_pos.integers(0, L, 3)),
    }
    seeds = list(range(1, n_seeds + 1))
    total = n_seeds * (1 + len(posiciones))  # 1 corrida A + 1 B por posicion, por semilla
    print(f"=== CG001 — TEST DE CAUSALIDAD DE epsilon (backend {_BACKEND}) ===")
    print(f"L={L} pasos={cfg.pasos} RUIDO={cfg.ruido} EPS={cfg.eps} "
          f"semillas={n_seeds} posiciones={list(posiciones)} RADIO={RADIO}")
    print(f"~{total} corridas\n")
    print("CRITERIO (fijado): CAUSAL si mediana dist(pico,eps)<RADIO en >=83% semillas Y << baseline caos.")
    print("-" * 74)

    ruta = f"cg001_causalidad_{time.strftime('%Y%m%d_%H%M%S')}_n{n_seeds}"
    t0 = time.time()
    hechas = 0

    # 1) m_A por semilla (sin epsilon) — se reusa para TODAS las posiciones y el baseline.
    mA = {}
    for s in seeds:
        mA[s] = correr_m(False, s, cfg).astype(np.float32)
        hechas += 1
    el = time.time() - t0
    print(f"m_A (sin eps) listo: {n_seeds} campos · {el/60:.1f} min · "
          f"ETA total ~{el/hechas*total/60:.1f} min\n")

    # 2) por posicion: D = |m_B - m_A|, pico, distancia a la posicion de epsilon.
    res = {}
    for nombre, p in posiciones.items():
        dists, mags, dentro = [], [], 0
        for s in seeds:
            mB = correr_m(True, s, cfg, pos_eps=p)
            D = np.abs(mB - mA[s])
            peak = np.unravel_index(int(np.argmax(D)), D.shape)
            d = dist_toroidal(peak, p, L)
            dists.append(d)
            mags.append(float(D.max()))
            if d < RADIO:
                dentro += 1
            hechas += 1
        med = float(np.median(dists))
        frac = dentro / n_seeds
        res[nombre] = {"pos": list(p), "mediana_dist": med, "frac_en_arruga": frac,
                       "max_D_medio": float(np.mean(mags))}
        marca = "  <-- SIGUE A epsilon" if (frac >= 0.83 and med < RADIO) else ""
        print(f"{nombre:>8} eps={str(p):<14} | mediana dist pico->eps = {med:6.2f} | "
              f"en arruga(<{RADIO}) = {frac:4.0%} | max|D| = {np.mean(mags):.4g}{marca}")
        el = time.time() - t0
        print(f"   {hechas}/{total} corridas · {el/60:.1f} min · ETA ~{(el/hechas*(total-hechas))/60:.1f} min")
        with open(ruta + ".json", "w", encoding="utf-8") as f:
            json.dump({"cfg": cfg.__dict__, "radio": RADIO, "posiciones": res}, f, indent=2)

    # 3) baseline de caos: |m_A(s1) - m_A(s2)|, pico, distancia al centro (referencia neutra).
    print("-" * 74)
    half = n_seeds // 2
    base_d, base_mag = [], []
    centro = posiciones["centro"]
    for i in range(half):
        D = np.abs(mA[seeds[i]].astype(np.float64) - mA[seeds[i + half]].astype(np.float64))
        peak = np.unravel_index(int(np.argmax(D)), D.shape)
        base_d.append(dist_toroidal(peak, centro, L))
        base_mag.append(float(D.max()))
    base_med = float(np.median(base_d))
    print(f"BASELINE caos |m_A1-m_A2| (sin eps): mediana dist pico->centro = {base_med:6.2f} | "
          f"max|D| = {np.mean(base_mag):.4g}")
    print(f"   (referencia de azar: ~{L*0.43:.0f} si el pico cae al azar en el volumen)")

    # ---- veredicto ----
    print("-" * 74)
    sigue = [n for n, r in res.items() if r["frac_en_arruga"] >= 0.83 and r["mediana_dist"] < RADIO]
    dist_eps_med = float(np.median([r["mediana_dist"] for r in res.values()]))
    razon = float(np.mean([r["max_D_medio"] for r in res.values()])) / (np.mean(base_mag) + 1e-12)
    print("VEREDICTO (parcial — el decisivo es comparar esta corrida gamma vs gamma=0):")
    if sigue and dist_eps_med < 0.5 * base_med:
        print(f"  LOCALIZADO — el pico de |m_B-m_A| sigue a epsilon ({len(sigue)}/{len(res)} pos, "
              f"dist~{dist_eps_med:.1f} << baseline {base_med:.1f}).")
        print(f"  razon huella(eps)/huella(caos) = {razon:.4f}")
        print("  OJO: 'el pico sigue a epsilon' puede ser LOCALIDAD TRIVIAL (pasa hasta con gamma=0).")
        print("  CAUSAL FUERTE (nicho) solo si esta razon es MAYOR que con gamma=0. Corre el control.")
    elif dist_eps_med > 0.5 * base_med:
        print(f"  CAOS — el pico cae lejos de epsilon (dist~{dist_eps_med:.1f} ~ baseline {base_med:.1f}). "
              f"Sensibilidad caotica gatillada por epsilon, NO siembra localizada.")
    else:
        print(f"  AMBIGUO — dist_eps~{dist_eps_med:.1f}, baseline~{base_med:.1f}, "
              f"posiciones que siguen: {sigue or 'ninguna'}. Subir semillas / mirar fino.")
    print(f"\nGuardado: {ruta}.json  (Files -> On My iPad -> Carnets)")


def main():
    base = FieldConfig(L=64, pasos=400) if PRODUCCION else FieldConfig(L=48, pasos=200)
    cfg = replace(base, gamma=GAMMA)
    if GAMMA == 0.0:
        print(">>> CONTROL nicho-OFF (gamma=0): si el pico sigue en eps, era localidad trivial.\n")
    causalidad(cfg, N_SEMILLAS)


if __name__ == "__main__":
    main()
