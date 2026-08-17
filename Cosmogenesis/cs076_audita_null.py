"""Auditoria del NULL de orden barajado de CS076 aplicado al estadistico KL de balance detallado.

El informe de CS076 ya cazo que para SKEWNESS ese NULL es una identidad (momento marginal, invariante al
orden). Pregunta aqui: para el KL de (x_t, x_t+1) vs su transpuesta, el mismo NULL es realmente informativo,
o esta casi degenerado tambien? El NULL reconstruye x_null = x[0] + cumsum(perm(incrementos)): conserva
EXACTAMENTE x[0] y x[-1] y el rango recorrido. Si los incrementos son casi todos del mismo signo (campo que
relaja monotonamente), la serie barajada tambien es monotona con el mismo recorrido -> el histograma 2D
queda casi igual y el KL casi igual, por construccion.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
import cg001_field as F
import cs076_direccion_temporal as C76


def trayectorias(L=32, pasos=300, seed=1, n_celdas=10, gamma=8.0):
    cfg = F.FieldConfig(L=L, pasos=pasos, gamma=gamma)
    rng = np.random.default_rng(seed)
    phi = F._inicializar_phi(rng, cfg, con_epsilon=True)
    m = np.zeros_like(phi)
    idx = [tuple(int(v) for v in rng.integers(0, L, size=3)) for _ in range(n_celdas)]
    tray = {c: [float(phi[c])] for c in idx}
    for _ in range(cfg.pasos):
        phi, m, _ = F._paso(phi, m, cfg)
        for c in idx:
            tray[c].append(float(phi[c]))
    return {str(c): {"phi": v} for c, v in tray.items()}


def main():
    for gamma in (8.0, 0.0):
        ser = trayectorias(gamma=gamma)
        # cuanto de la serie es monotona (signo constante de los incrementos)
        fr = []
        for s in ser.values():
            inc = np.diff(np.asarray(s["phi"]))
            fr.append(max((inc > 0).mean(), (inc < 0).mean()))
        kl_real = C76.violacion_balance_detallado(ser)
        rng = np.random.default_rng(99)
        kls = [C76.violacion_balance_detallado(C76.null_orden_barajado(ser, rng)) for _ in range(200)]
        kls = np.array(kls)
        # el NULL conserva x[0] y x[-1]?
        s0 = list(ser.values())[0]["phi"]
        n0 = list(C76.null_orden_barajado(ser, np.random.default_rng(5)).values())[0]["phi"]
        print(f"\ngamma={gamma}")
        print(f"  frac. de incrementos con signo dominante (media sobre 10 celdas) = {np.mean(fr):.4f}")
        print(f"  NULL conserva x[0]: {abs(s0[0]-n0[0]):.2e}   conserva x[-1]: {abs(s0[-1]-n0[-1]):.2e}")
        print(f"  KL real = {kl_real:.6f}")
        print(f"  KL NULL barajado: media={kls.mean():.6f}  sd={kls.std(ddof=1):.2e}  "
              f"min={kls.min():.6f} max={kls.max():.6f}")
        z = (kl_real - kls.mean()) / (kls.std(ddof=1) + 1e-300)
        print(f"  z = {z:+.3f}   |  desviacion relativa NULL/real = {kls.std(ddof=1)/kl_real:.2e}")
        print(f"  => el NULL reproduce el {100*kls.mean()/kl_real:.2f}% del KL real")


if __name__ == "__main__":
    main()
