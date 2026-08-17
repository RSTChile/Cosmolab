"""
CG001 — INSTRUMENTO COSMOSEMIOTICO DE CAMPO
Derivacion nodo-por-nodo. Cada termino atado a su nodo. NINGUN resultado programado.

PRINCIPIO DE LA TRADUCCION:
- NO hay entidades con atributos. NO se almacena S ni Delta en ningun objeto.
- Hay UN campo continuo (= Omega_posible) y UNA dinamica.
- Una "diferencia" = asimetria local del campo respecto de su vecindad (C-N4, relacional).
- NO se pone probabilidad a mano. La dinamica es determinista; la probabilidad EMERGE
  como lectura a posteriori de lo que persistio.
- Lo UNICO puesto a mano: densidad inicial de asimetrias (singularidad, eje RUIDO) + epsilon.
- GEOMETRIA: kernel isotropico (gaussiano). El 3D es Omega_posible, sin ejes privilegiados.

MARCO TERMODINAMICO (CAPA DE MEDICION — no cambia la dinamica):
- SINGULARIDAD = exergia maxima + entropia cero (campo rugoso; amplitud = RUIDO).
- EXERGIA(t) = Sum|a| — asimetria total disponible.
- RELAJACION convierte exergia en entropia (irreversible).
- FLECHA: emerge de partir en exergia-maxima/entropia-cero (C-N2.5.6).
- ES = ICES + IDES: reparto medido, NO impuesto (#99: "todo a entropia" debe poder salir).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class FieldConfig:
    """Parametros calibrables. NO se ajustan para 'que salga estructura': se barren."""

    L: int = 48
    pasos: int = 300
    lam: float = 0.50  # relajacion = costo de sostener diferencia (R0, #119, #128)
    sigma: float = 1.0  # vecindad del kernel isotropico
    gamma: float = 8.0  # memoria protege asimetria (nicho, #131, C-N2.6)
    decay: float = 0.97  # memoria ambiental no eterna (tau_env, #129)
    eps: float = 0.05  # UNICA arruga inicial (#44, #133)
    ruido: float = 1.0  # amplitud singularidad: eje liso<->rugoso
    q_nicho: float = 0.999  # cuantil para ICES (solo medicion)
    log_every: int = 10


# Produccion (barrido fino en iMac/LaCie)
PRODUCTION = FieldConfig(L=64, pasos=400)


def _inicializar_phi(
    rng: np.random.Generator,
    cfg: FieldConfig,
    con_epsilon: bool,
    eps_pos: tuple[int, int, int] | None = None,
) -> np.ndarray:
    # SINGULARIDAD (#43, #7): densidad de asimetrias; amplitud = ruido (eje experimental).
    phi = rng.normal(0.0, cfg.ruido, size=(cfg.L, cfg.L, cfg.L)).astype(np.float64)
    if con_epsilon:
        if eps_pos is None:
            c = cfg.L // 2
            eps_pos = (c, c, c)
        z, y, x = eps_pos
        phi[z, y, x] += cfg.eps
    return phi


def _paso(phi: np.ndarray, m: np.ndarray, cfg: FieldConfig) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    # DIFERENCIA = asimetria local (C-N4). Isotropica por construccion.
    vecindad = gaussian_filter(phi, sigma=cfg.sigma, mode="wrap")
    a = phi - vecindad
    abs_a = np.abs(a)

    # MEDICION termodinamica (lee; NO modifica dinamica)
    exergia = float(abs_a.sum())

    # HISTORIA AMBIENTAL (#126, C-N2.5.2)
    m = cfg.decay * m + abs_a

    # NICHO (#131, C-N2.6): medio history-dependiente; estructura EMERGE o no
    lam_eff = cfg.lam / (1.0 + cfg.gamma * m)

    # REPARTO ES = ICES + IDES (Bloque 23) — medicion
    disipado = float((lam_eff * abs_a).sum())
    thr = float(np.quantile(m, cfg.q_nicho))
    mask_nicho = m > thr
    convertido = float(abs_a[mask_nicho].sum())

    # RELAJACION / estiramiento (#119, #128)
    phi = phi - lam_eff * a

    metricas = {
        "exergia": exergia,
        "disipado": disipado,
        "convertido": convertido,
        "memoria_max": float(m.max()),
        "concentracion": float(m.max() / (m.mean() + 1e-12)),
        "n_nicho": int(mask_nicho.sum()),
    }
    return phi, m, metricas


def correr(
    con_epsilon: bool,
    seed: int = 1,
    cfg: FieldConfig | None = None,
    *,
    ruido: float | None = None,
    eps_pos: tuple[int, int, int] | None = None,
    retornar_campos: bool = False,
) -> dict[str, Any]:
    """
    Una corrida A (epsilon=0) o B (epsilon>0).
    eps_pos: celda de la arruga en B; default centro L//2.
    Retorna metricas finales; opcionalmente phi, m e historia muestreada.
    """
    cfg = cfg or FieldConfig()
    if ruido is not None:
        cfg = replace(cfg, ruido=ruido)

    rng = np.random.default_rng(seed)
    phi = _inicializar_phi(rng, cfg, con_epsilon, eps_pos=eps_pos)
    m = np.zeros_like(phi)
    entropia_acum = 0.0
    hist: list[dict[str, Any]] = []

    for t in range(cfg.pasos):
        phi, m, met = _paso(phi, m, cfg)
        entropia_acum += met["disipado"]
        if t % cfg.log_every == 0 or t == cfg.pasos - 1:
            hist.append({"t": t, "entropia": entropia_acum, **met})

    final = hist[-1]
    out: dict[str, Any] = {
        "con_epsilon": con_epsilon,
        "seed": seed,
        "ruido": cfg.ruido,
        "eps_pos": eps_pos if con_epsilon else None,
        "exergia": final["exergia"],
        "entropia": final["entropia"],
        "convertido": final["convertido"],
        "concentracion": final["concentracion"],
        "memoria_max": final["memoria_max"],
        "n_nicho": final["n_nicho"],
        "hist": hist,
    }
    if retornar_campos:
        out["phi"] = phi
        out["m"] = m
    return out


def signo_estable(difs: list[float] | np.ndarray) -> tuple[float, float]:
    """Media de divergencias B-A y fraccion de semillas con el mismo signo."""
    arr = np.asarray(difs, dtype=np.float64)
    mu = float(arr.mean())
    if abs(mu) < 1e-12:
        return mu, 0.0
    return mu, float((np.sign(arr) == np.sign(mu)).mean())


def resumen_tag(tag: str, hist: list[dict]) -> dict:
    h0, hf = hist[0], hist[-1]
    print(
        f" [{tag}] t={h0['t']:>3}->{hf['t']:<3} "
        f"exergia {h0['exergia']:.0f}->{hf['exergia']:.0f} "
        f"entropia 0->{hf['entropia']:.0f} "
        f"convertido {h0['convertido']:.1f}->{hf['convertido']:.1f} "
        f"concentracion {h0['concentracion']:.2f}->{hf['concentracion']:.2f}"
    )
    return hf


def demo() -> None:
    cfg = FieldConfig()
    print("=== CG001 campo — primer contacto (1 semilla, demostracion de forma) ===")
    print(
        f"L={cfg.L} pasos={cfg.pasos} LAM={cfg.lam} SIGMA={cfg.sigma} "
        f"GAMMA={cfg.gamma} DECAY={cfg.decay} EPS={cfg.eps} RUIDO={cfg.ruido}\n"
    )

    hA = correr(False, seed=1, cfg=cfg)["hist"]
    hB = correr(True, seed=1, cfg=cfg)["hist"]
    print("CG001-A (epsilon = 0):")
    fA = resumen_tag("A", hA)
    print("CG001-B (epsilon > 0):")
    fB = resumen_tag("B", hB)

    print("\n--- Lectura termodinamica honesta (1 semilla, NO veredicto) ---")
    pct = 100 * (1 - fA["exergia"] / hA[0]["exergia"])
    print(
        f"FLECHA: exergia cae {hA[0]['exergia']:.0f}->{fA['exergia']:.0f} "
        f"(gastada {pct:.0f}%), entropia sube 0->{fA['entropia']:.0f}"
    )
    print(
        f"REPARTO ES=ICES+IDES: convertido={fA['convertido']:.1f} "
        f"disipado_total~entropia={fA['entropia']:.0f}"
    )
    print("DIVERGENCIA A/B (efecto de epsilon):")
    print(f" exergia restante : A={fA['exergia']:.1f} B={fB['exergia']:.1f} (dif {fB['exergia']-fA['exergia']:+.3f})")
    print(f" convertido       : A={fA['convertido']:.2f} B={fB['convertido']:.2f} (dif {fB['convertido']-fA['convertido']:+.3f})")
    print(
        f" concentracion    : A={fA['concentracion']:.2f} B={fB['concentracion']:.2f} "
        f"(dif {fB['concentracion']-fA['concentracion']:+.3f})"
    )


if __name__ == "__main__":
    demo()