"""
p_materia_oscura_halo.py — PIEZA: MATERIA OSCURA COMO ANDAMIO DINÁMICO (CDM).

Adjudicación Q2 (INSTRUCCION_CC_cierre_holistico.md v3): CDM EMERGE, no se planta. G-SIN-SIEMBRA
restaurado -- NADA de función de masa de halos (Press-Schechter) ni estadística ΛCDM importada: eso
sería sembrar centros de colapso a mano. La materia oscura es una SEGUNDA ESPECIE que sale del MISMO
generador de campo #23 (`densidad_intrinseca`, catalogo.py -- misma receta, reusada, no reimplementada),
en el MISMO escenario 3D, bajo la MISMA gravedad general -- pero DESACOPLADA de EM: no siente presión
térmica ni enfriamiento (`p_enfriamiento_H2.py` nunca actúa sobre ella). La asimetría que la hace
colapsar ANTES está en el ACOPLAMIENTO (ausencia de soporte de presión), no en una posición o masa
plantada de antemano.

Masa: 1.0 por partícula (unidad del sistema) -- NO se importa la razón real ΩCDM/Ωb≈5.4 (sería otro
número de nuestro universo bajo la adjudicación Q1); se deja que el MECANISMO (desacople), no una
proporción de masa importada, sea lo que hace la diferencia. n_cdm = n_bariones (1:1), la comparación
más conservadora posible.
"""
import numpy as np

from cs072_modulos.catalogo import densidad_intrinseca
from cs072_modulos.piezas.p_gravedad_general import posiciones_escenario


class MateriaOscuraHalo:
    numero = "cdm"
    nombre = "materia oscura (CDM, desacoplada de EM)"
    nivel = "estructura"
    siente_em = False   # la marca que la distingue: nunca recibe presión ni enfriamiento

    def __init__(self, n_cdm, amp_rugosidad, lado_escenario, activa=True, seed_pos=54321, seed_dens=7000):
        self.activa_flag = activa
        self.n = n_cdm
        if activa:
            self.densidad = densidad_intrinseca(n_cdm, amp_rugosidad)   # MISMA receta que #23, instancia propia
            self.masa = np.ones(n_cdm)                                    # unidad de masa del sistema
            self.pos, _ = posiciones_escenario(n_cdm, lado=lado_escenario, seed=seed_pos)
        else:
            self.densidad = np.zeros(0)
            self.masa = np.zeros(0)
            self.pos = np.zeros((0, 3))

    def activa(self):
        return self.activa_flag

    def barajar_densidad(self, seed):
        """Para el control NULL: misma distribución, coherencia destruida (misma receta que el resto
        del arco -- NULL = el campo real barajado, no un campo distinto)."""
        if self.n == 0:
            return self.densidad
        return self.densidad[np.random.default_rng(seed).permutation(self.n)]
