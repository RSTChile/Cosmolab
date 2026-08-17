"""
p03_fuerte.py — PIEZA #3: FUERZA FUERTE / CONFINAMIENTO.

Qué hace, en simple: es la ÚNICA fuerza que pega quark con quark. Cuando el universo se enfría bajo T_CONF,
liga quarks de colores distintos (R,G,B) para formar bariones neutros de color. NADIE MÁS toca la ligadura
quark-quark -> por eso apagar esta pieza da CERO bariones (prueba de admisibilidad).

Observable: nº de bariones. Nivel: quark. Época: T < T_CONF (la más caliente tras el plasma).
"""
import numpy as np
from cs072_modulos.pieza_base import Pieza

R_STRONG = 0.30
T_CONF   = 1.0

class FuerzaFuerte(Pieza):
    numero = 3
    nombre = "fuerte/confinamiento"
    nivel = "quark"
    T_umbral = T_CONF
    observable = "bariones"

    def actua(self, estado, step):
        # liga SÓLO pares quark-quark de color distinto y misma clase (materia con materia).
        e = estado
        dq = R_STRONG * (e.cd & e.me).astype(float) * np.sqrt(np.outer(e.viva, e.viva))
        e.Bq = e.Bq + dq
        np.fill_diagonal(e.Bq, 0.0)
        if "confinamiento" not in e.epocas:
            e.epocas["confinamiento"] = round(float(e.T), 3)
