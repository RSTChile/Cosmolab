"""
p_expansion.py — PIEZA: EXPANSIÓN (régimen macroscópico, post-fósil).

Qué hace, en simple: la métrica ya existe fosilizada en el átomo (Bohr, escala cuántica) desde CS072.
Esta pieza NO crea espacio -- lo DESPLIEGA a escala macroscópica. Adjudicación v3 (INSTRUCCION_CC_
cierre_holistico.md, Q3): "el 3D cuántico se vuelve 3D macroscópico dimensionable sólo con la expansión
y la consolidación" -- por eso la expansión es imprescindible en el bucle, no un extra.

factor de escala a(t): NO se inventa una ley nueva -- se deriva del propio reloj de enfriamiento que el
motor YA tiene (T cae con el step, `Estado.enfria`). En cosmología real T∝1/a; aquí, adimensional:
a(t) = T0 / T(t). a(0)=1 (por T(0)=T0), crece monótono según T cae -- mismo reloj, ninguna constante nueva.

Efecto: estira TODAS las posiciones por igual (isótropo -- G-EXPANSION-ISOTROPA: no impone dirección ni
rejilla) en cada paso. Compite contra la gravedad general: donde la sobredensidad es fuerte, la atracción
local vence al estiramiento; donde es débil, la expansión diluye.
"""


class Expansion:
    numero = "exp"
    nombre = "expansión macroscópica (post-fósil)"
    nivel = "estructura"

    def __init__(self, T0, activa=True):
        self.T0 = float(T0)
        self.activa_flag = activa
        self._a_prev = 1.0

    def activa(self):
        return self.activa_flag

    def factor_escala(self, T_actual):
        """a(t) = T0/T(t), derivado del reloj de enfriamiento del motor. Nunca decrece (T cae monótono)."""
        return float(self.T0 / max(float(T_actual), 1e-9))

    def paso_de_estiramiento(self, T_actual):
        """Factor MULTIPLICATIVO a aplicar a las posiciones ESTE paso: a(t)/a(t-1). Isótropo -- mismo
        factor en las 3 coordenadas, ningún eje privilegiado."""
        if not self.activa_flag:
            return 1.0
        a_t = self.factor_escala(T_actual)
        paso = a_t / self._a_prev if self._a_prev > 0 else 1.0
        self._a_prev = a_t
        return float(paso)
