"""
C-N2.5.6 — TEST DE REVERSIBILIDAD del sustrato cg001_field.py

Tesis a probar: el paso _paso() es una BIYECCION del estado completo (phi, m) sobre si mismo,
invertible analiticamente. Si lo es, "flecha temporal" medida sobre este sustrato no puede
distinguir adelante/atras por perdida de informacion — solo por redondeo de maquina.

DERIVACION DEL INVERSO (a mano, del codigo):
  vecindad = G phi                (G = gaussian_filter circular, lineal, simetrica, autoval en (0,1])
  a        = (I - G) phi
  m'       = decay*m + |a|
  Lam      = diag( lam / (1 + gamma*m') )      <-- depende de m' (el m YA actualizado), que es estado nuevo
  phi'     = phi - Lam*(I-G) phi = [I - Lam(I-G)] phi

Dado el estado NUEVO (phi', m'):
  1) Lam se conoce EXACTAMENTE (solo necesita m').
  2) phi = [I - Lam(I-G)]^{-1} phi'  -> punto fijo phi_{k+1} = phi' + Lam*((I-G) phi_k).
     Converge porque ||Lam(I-G)|| <= max(lam)*1 = 0.5 < 1 (G es PSD con autoval en (0,1]).
  3) a = (I-G) phi ; m = (m' - |a|) / decay.
Ningun paso pierde informacion: |.| se aplica a una cantidad DERIVADA (recuperable de phi),
no al estado. El umbral np.quantile/mask_nicho es SOLO medicion (no realimenta phi ni m).
"""
from __future__ import annotations
import sys
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
import cg001_field as F


def paso_adelante(phi, m, cfg):
    phi2, m2, _met = F._paso(phi, m, cfg)
    return phi2, m2


def paso_atras(phi2, m2, cfg, n_iter=200, tol=1e-15):
    lam_eff = cfg.lam / (1.0 + cfg.gamma * m2)
    phi = phi2.copy()
    for _ in range(n_iter):
        a = phi - gaussian_filter(phi, sigma=cfg.sigma, mode="wrap")
        nuevo = phi2 + lam_eff * a
        d = np.max(np.abs(nuevo - phi))
        phi = nuevo
        if d < tol:
            break
    a = phi - gaussian_filter(phi, sigma=cfg.sigma, mode="wrap")
    m = (m2 - np.abs(a)) / cfg.decay
    return phi, m


def test(L=32, pasos=25, seed=1, gamma=8.0, lam=0.50):
    cfg = F.FieldConfig(L=L, gamma=gamma, lam=lam)
    rng = np.random.default_rng(seed)
    phi0 = F._inicializar_phi(rng, cfg, con_epsilon=True)
    m0 = np.zeros_like(phi0)

    phi, m = phi0.copy(), m0.copy()
    for _ in range(pasos):
        phi, m = paso_adelante(phi, m, cfg)
    phi_k, m_k = phi.copy(), m.copy()

    for _ in range(pasos):
        phi, m = paso_atras(phi, m, cfg)

    err_phi = float(np.max(np.abs(phi - phi0)))
    err_m = float(np.max(np.abs(m - m0)))
    esc_phi = float(np.max(np.abs(phi0)))
    print(f"L={L} pasos={pasos} seed={seed} gamma={gamma} lam={lam}")
    print(f"  |phi_recuperado - phi_0|_inf = {err_phi:.3e}   (escala |phi_0|_inf = {esc_phi:.3f}"
          f" -> error relativo {err_phi/esc_phi:.3e})")
    print(f"  |m_recuperado   - m_0|_inf   = {err_m:.3e}   (m_0 = 0 exacto)")
    print(f"  eps de maquina (float64)     = {np.finfo(np.float64).eps:.3e}")

    # sanidad: un paso adelante desde el estado recuperado reproduce el estado k=1?
    p1, m1 = paso_adelante(phi0, m0, cfg)
    pb, mb = paso_atras(p1, m1, cfg)
    print(f"  ida-y-vuelta de UN paso: |dphi|={np.max(np.abs(pb-phi0)):.3e}  |dm|={np.max(np.abs(mb-m0)):.3e}")
    return err_phi, err_m


def jacobiano_volumen(L=16, seed=1, gamma=8.0, lam=0.50, pasos=10):
    """Contraccion de volumen del bloque phi: log|det[I - Lam(I-G)]| = sum log(1 - autoval).
    Se estima por traza de la serie de Neumann sobre vectores de Hutchinson (barato)."""
    cfg = F.FieldConfig(L=L, gamma=gamma, lam=lam)
    rng = np.random.default_rng(seed)
    phi = F._inicializar_phi(rng, cfg, con_epsilon=True)
    m = np.zeros_like(phi)
    for _ in range(pasos):
        phi, m, _ = F._paso(phi, m, cfg)
    lam_eff = cfg.lam / (1.0 + cfg.gamma * m)
    # log det = tr log(I - M),  M = Lam(I-G);  tr log(I-M) = -sum_k tr(M^k)/k
    def aplicar_M(v):
        return lam_eff * (v - gaussian_filter(v, sigma=cfg.sigma, mode="wrap"))
    n_probe, K = 8, 25
    total = 0.0
    for j in range(n_probe):
        z = rng.choice([-1.0, 1.0], size=phi.shape)
        v = z.copy()
        for k in range(1, K + 1):
            v = aplicar_M(v)
            total += -float((z * v).sum()) / k
    logdet = total / n_probe
    n = phi.size
    print(f"\nJacobiano del bloque phi (L={L}, tras {pasos} pasos):")
    print(f"  log|det| ~= {logdet:.2f} sobre {n} celdas -> {logdet/n:.5f} por celda (NEGATIVO = contrae volumen)")
    print("  contrae volumen de fase (dinamica disipativa) pero sigue siendo BIYECTIVA: det != 0.")


if __name__ == "__main__":
    test(L=32, pasos=25, seed=1)
    print()
    test(L=48, pasos=50, seed=7)
    print()
    test(L=32, pasos=25, seed=3, gamma=0.0)   # regimen anti-Shannon de CS076
    jacobiano_volumen()
