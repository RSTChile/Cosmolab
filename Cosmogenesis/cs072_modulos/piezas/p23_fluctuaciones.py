"""
p23_fluctuaciones.py — PIEZA #23: FLUCTUACIONES CUÁNTICAS / ASIMETRÍA DE DISTRIBUCIÓN.

Qué hace, en simple: el plasma inicial no era perfectamente liso -- era RUGOSO, con regiones más densas y
otras menos (la misma rugosidad que hoy vemos en el fondo de microondas: manchas más y menos luminosas).
Esta pieza asigna a cada partícula una DENSIDAD local, tomada de una distribución multiescala de amplitudes.
Esa densidad es lo que la gravedad usa para discriminar: sin rugosidad todo es uniforme y no hay estructura;
con rugosidad, las regiones densas atraen más y la red de átomos deja de ser trivial.

HONESTIDAD (anti-Shannon): la rugosidad es DETERMINISTA (cero azar) pero es HETEROGENEIDAD EXTERNA IMPUESTA
-- se declara, no se esconde. Lo que se BARRE es su AMPLITUD (amp_rugosidad). Lo que NO se impone es la forma
del resultado: se mide si emerge estructura y a qué amplitud. El test crítico es invariancia a permutación:
lo que importa es la DISTRIBUCIÓN de densidades (el histograma), no qué índice recibió cuál.

Observable: densidad (campo) -> habilita el diámetro no-trivial de la red. Nivel: global. Época: siempre (condición inicial).
"""
import numpy as np
from cs072_modulos.pieza_base import Pieza

class Fluctuaciones(Pieza):
    numero = 23
    nombre = "fluctuaciones/asimetría de distribución"
    nivel = "global"
    T_umbral = None
    observable = "diametro_red"   # su efecto se ve en la estructura que la gravedad puede tejer

    def __init__(self, amp_rugosidad=0.5):
        self.amp = amp_rugosidad

    def actua(self, estado, step):
        # La densidad ya es INTRÍNSECA: la genera el catálogo (densidad_intrinseca) y se permuta con la partícula.
        # Esta pieza sólo REGISTRA que la rugosidad está activa; no reconstruye el campo desde el índice (eso era
        # el bug Shannon: densidad como función de linspace(N) no se permutaba). Aquí no hace falta actuar cada paso.
        return
