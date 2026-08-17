"""
p08_aniquilacion.py — PIEZA #8: ANIQUILACIÓN MATERIA-ANTIMATERIA.

Qué hace, en simple: cuando materia y antimateria del mismo tipo se encuentran, se destruyen (van a luz).
NO es un porcentaje ni una tasa (constraint del director): es una RESTA de poblaciones -- por cada clase y
color, min(materia, antimateria) desaparece, sobrevive el EXCEDENTE. Invariante al orden (indistinguibles).
Es lo que deja el pequeño excedente de materia (asimetría bariónica) del que sale todo lo demás.

Observable: nº de bariones (sin aniquilación sobran quarks -> más bariones). Nivel: quark. Época: siempre.
"""
import numpy as np
from cs072_modulos.pieza_base import Pieza

class Aniquilacion(Pieza):
    numero = 8
    nombre = "aniquilación materia-antimateria"
    nivel = "quark"
    T_umbral = None            # actúa en todo el enfriamiento
    observable = "bariones"

    def actua(self, estado, step):
        # Aniquilación por (color, SABOR): partir también por carga (up=+2, down=-1, leptón). Antes se partía sólo
        # por color -> qué sabores (up/down) sobrevivían dentro de un color dependía del orden = Shannon residual
        # en la composición p/n. Al partir por sabor, la PROPORCIÓN de sabores supervivientes es invariante al orden.
        e = estado
        cargas_por_tipo = {True: (2, -1), False: (-3, 3)}   # quarks: up=+2/down=-1 ; leptones: e=-3/e+=+3
        for eq in (True, False):
            for c in (0, 1, 2, -1):
                for q in cargas_por_tipo[eq]:
                    mat = np.where((~e.es_anti)&(e.es_quark==eq)&(e.color==c)&(e.carga==q)&(e.viva>0.5))[0]
                    ant = np.where(( e.es_anti)&(e.es_quark==eq)&(e.color==c)&(e.carga==-q)&(e.viva>0.5))[0]
                    k = min(len(mat), len(ant))
                    if k > 0:
                        e.viva[mat[:k]] = 0.0
                        e.viva[ant[:k]] = 0.0
