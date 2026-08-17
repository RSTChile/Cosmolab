"""
freeze_out.py — PIEZA #5: FUERZA DÉBIL / FREEZE-OUT DEL NEUTRÓN.

Qué hace, en simple: mientras hace calor, la débil convierte protón<->neutrón en ambos sentidos (equilibrio).
Al enfriar bajo la diferencia de masa, sólo queda n->p (el neutrón pesa más). El ratio p:n se CONGELA.

El 7:1 NO se impone: emerge de la competencia entre la expansión (H~h*T^2) y la débil (Gamma~G*T^5), más el
decaimiento del neutrón libre hasta la nucleosíntesis. El 7:1 real cae en una banda de expansión concreta
-- es la firma de la tasa de expansión de NUESTRO universo, un dato de realidad, no una perilla.
"""
import numpy as np

DELTA_M = 1.293      # MeV, diferencia real neutrón-protón (estructural)
G_WEAK  = 1.0        # fuerza débil (estructural)
TAU_N   = 880.0      # vida del neutrón libre (s)
T_NUC_S = 180.0      # tiempo a la nucleosíntesis (s)

def freeze_out_neutron(tasa_expansion):
    """Ratio p:n emergente. Devuelve (ratio, T_freeze)."""
    h = max(tasa_expansion*20.0, 1e-6)          # escala la expansión del motor a la competencia física
    T_freeze = (h/G_WEAK)**(1.0/3.0)
    np_freeze = np.exp(-DELTA_M/T_freeze)        # n/p al congelar (Boltzmann)
    frac_n = np_freeze * np.exp(-T_NUC_S/TAU_N)  # decaimiento del neutrón libre hasta nucleosíntesis
    ratio = (1.0/frac_n) if frac_n > 0 else float("inf")
    return ratio, round(T_freeze, 3)
