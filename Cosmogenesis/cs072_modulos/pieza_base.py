"""
pieza_base.py — LA INTERFAZ ESTÁNDAR que cada una de las 23 piezas cumple.

Cada módulo-pieza es una subclase que declara:
  numero, nombre   : identidad canónica (del manifiesto)
  nivel            : 'quark' | 'nucleon' | 'atomo' | 'global'  (sobre qué opera)
  T_umbral         : temperatura de activación de su época (None = actúa siempre)
  observable       : nombre del observable que SU acción cambia (para la prueba de admisibilidad)
  actua(estado, step): modifica SÓLO su nivel del Estado. Nada más.

REGLA DE ADMISIBILIDAD (anti-Shannon): apagar una pieza DEBE cambiar su observable. Si no, la pieza está
declarada pero no actúa (reserva A). El núcleo verifica esto apagando cada pieza y midiendo.
REGLA DE ÉPOCA: una pieza sólo actúa cuando estado.T < T_umbral (o siempre si T_umbral is None).
"""

class Pieza:
    numero = 0
    nombre = "base"
    nivel = "global"
    T_umbral = None
    observable = None

    def activa(self, estado):
        return self.T_umbral is None or estado.T < self.T_umbral

    def actua(self, estado, step):
        raise NotImplementedError(f"pieza {self.numero} ({self.nombre}) no implementa actua()")

    def __repr__(self):
        u = "siempre" if self.T_umbral is None else f"T<{self.T_umbral}"
        return f"<#{self.numero} {self.nombre} [{self.nivel}, {u}] -> {self.observable}>"
