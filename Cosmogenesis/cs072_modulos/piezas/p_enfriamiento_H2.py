"""
p_enfriamiento_H2.py — PIEZA: PRESIÓN TÉRMICA (EM) + ENFRIAMIENTO MOLECULAR H₂.

Qué hace, en simple: el colapso gravitacional comprime el gas y lo calienta; si nada libera ese calor,
la presión térmica sube y DETIENE el colapso (un solo grumo difuso, sin fragmentar -- el diagnóstico de
CS de por qué los 4 prototipos anteriores dieron grumo/hub). El canal H₂ (H+e⁻→H⁻+γ; H⁻+H→H₂+e⁻) es el
único refrigerante disponible sin metales: irradia el calor de compresión y PERMITE que el colapso
continúe y se fragmente en varios núcleos en vez de uno.

SÓLO actúa sobre partículas con `siente_em=True` (bariones) -- la materia oscura (`p_materia_oscura_
halo.py`, siente_em=False) nunca pasa por este módulo: ésa es la asimetría de acoplamiento que la hace
colapsar primero (adjudicación Q2), no una posición o masa plantada.

MODELADO (declarado, no oculto -- es una simplificación de juguete, no un solver hidrodinámico
completo; se documenta explícitamente en vez de esconderla):
  - Densidad local DINÁMICA (para calor de compresión) vía k-vecino más cercano en las posiciones
    ACTUALES (no el campo #23 estático -- ese es la SEMILLA de masa; esto es la respuesta dinámica al
    colapso). k=6 (convención SPH típica, no ajustada).
  - Calentamiento por compresión: ΔT ∝ max(ρ_local/ρ_media - 1, 0) -- sólo sube donde hay sobredensidad
    dinámica, nunca inyecta calor de la nada (G-SIN-ENERGIA-NUEVA: el calor viene de la propia
    compresión gravitacional, no de una fuente externa).
  - Enfriamiento H₂ (si el módulo está ON): relajación exponencial de T hacia un piso, SÓLO donde hay
    sobredensidad (el gatillo -- necesita compresión para que el canal opere, como en el mecanismo real).
  - Soporte de presión: representado como agitación térmica ISÓTROPA (kick de velocidad ~√T, dirección
    determinista-no-correlacionada con la densidad -- no un solver de gradiente de presión SPH completo;
    es la forma más simple honesta de dar soporte contra el colapso sin imponer una dirección
    (G-EXPANSION-ISOTROPA aplica también aquí: el soporte térmico no privilegia ningún eje).
"""
import numpy as np
from scipy.spatial import cKDTree


class EnfriamientoH2:
    numero = "h2"
    nombre = "presión térmica (EM) + enfriamiento H₂"
    nivel = "estructura"

    def __init__(self, n, T_inicial, activa_cooling=True, k_vecinos=6,
                 tasa_calentamiento=0.5, tasa_enfriamiento=0.3, T_piso=None, seed=9000, softening=0.3):
        self.n = n
        self.T = np.full(n, float(T_inicial))
        # softening: MISMO piso de distancia que usa GravedadGeneral (no un valor nuevo inventado) --
        # sin esto, un encuentro cercano en la integración N-cuerpos (regularizado en la fuerza por el
        # softening, pero NO en este estimador) hace que rho_local se dispare varios órdenes de magnitud
        # (volumen casi nulo -> densidad artificial) y M_J colapse a ~0, dando una razón masa/M_J
        # espuria. Hallado el 19-jul auditando un salto de la serie de escala (rho_local=4e5 vs 0.05-7
        # en el resto de los clusters de la misma corrida).
        self.softening = softening
        self.activa_cooling_flag = activa_cooling   # el interruptor: SIN esto, sólo calienta -> un grumo
        self.k = k_vecinos
        self.tasa_calentamiento = tasa_calentamiento
        self.tasa_enfriamiento = tasa_enfriamiento
        self.T_piso = T_piso if T_piso is not None else 0.1 * T_inicial
        self._rng = np.random.default_rng(seed)

    def activa(self):
        return True   # la presión térmica SIEMPRE actúa (es EM, no un módulo apagable); el interruptor
                       # real es activa_cooling_flag (el canal H2), para aislar su efecto (Regla 3)

    def _densidad_local_dinamica(self, pos):
        if self.n < self.k + 1:
            return np.ones(self.n)
        tree = cKDTree(pos)
        dist, _ = tree.query(pos, k=self.k + 1)
        r_k = dist[:, -1]
        vol = (4.0 / 3.0) * np.pi * np.maximum(r_k, self.softening) ** 3
        return self.k / vol

    def actualizar(self, pos, rho_externo=None):
        """Un paso: mide densidad local dinámica, calienta por compresión, enfría vía H2 si está ON.
        Devuelve la presión (T actual) para que el orquestador la use como soporte.

        rho_externo=None (default, TODOS los experimentos ya reportados): usa el estimador propio
        (_densidad_local_dinamica, piso=self.softening). Aditivo, no cambia nada existente.
        rho_externo=array (DISENO_CS073_ignicion_PARA_CC.md, adjudicado): usa la MISMA rho_i por
        partícula que ya calculó la gravedad adaptativa (aceleraciones_adaptativas) -- una sola
        resolución compartida entre gravedad y Jeans, en vez de dos pisos que puedan desincronizarse."""
        rho_local = self._densidad_local_dinamica(pos) if rho_externo is None else rho_externo
        rho_media = float(rho_local.mean()) if len(rho_local) else 1.0
        sobredensidad = np.clip(rho_local / max(rho_media, 1e-9) - 1.0, 0.0, None)
        self.T = self.T + self.tasa_calentamiento * sobredensidad
        if self.activa_cooling_flag:
            gatillo = sobredensidad > 0
            self.T = np.where(gatillo,
                               self.T - self.tasa_enfriamiento * (self.T - self.T_piso),
                               self.T)
        self.T = np.maximum(self.T, self.T_piso)
        return self.T

    def kick_termico(self, escala=1.0):
        """Agitación isótropa ~sqrt(T): el soporte de presión. Dirección determinista (semilla fija del
        propio objeto, no correlacionada con la densidad) -- no privilegia ningún eje ni región."""
        dirs = self._rng.normal(size=(self.n, 3))
        norm = np.linalg.norm(dirs, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        dirs = dirs / norm
        return dirs * np.sqrt(np.maximum(self.T, 0.0))[:, None] * escala
