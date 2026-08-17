"""
campo_velocidad_turbulento.py — BRAZO 1: turbulencia estándar (referencia), INSTRUCCION CS 20-jul
("dos brazos de campo de velocidades inicial").

DATO IMPORTADO -- se declara como tal (no se deriva de la malla causal, a propósito: es la referencia
externa contra la que se compara el Brazo 2). Espectro de potencias E(k) ~ k^-2 (Burgers, el estándar de
campo para turbulencia supersónica en nubes moleculares -- Mac Low & Klessen 2004; Federrath et al.).

Método: ruido blanco Gaussiano por componente de velocidad en una grilla Ng^3, filtrado en Fourier por
k^-(pendiente/2) en amplitud (=> espectro de energía por cáscara ~k^-pendiente), modo k=0 anulado (sin
momento de bulk impuesto a mano), transformada inversa a espacio real, interpolación TRILINEAL a las
posiciones reales de las partículas (que no caen en la grilla). El método estándar de generación de
turbulencia sintética en SPH (Dubinski 1995 y sucesores), no inventado para este experimento.

Normalización: v_rms objetivo = Mach * c_s, con c_s=sqrt(polyk) (MISMO polyk que ya usa el resto del
arco, leído del objeto real, no hand-typed). Mach=3 -- valor típico/de libro para turbulencia supersónica
en nubes moleculares (Larson 1981; revisión Mac Low & Klessen 2004 cita Mach 3-10 como rango típico) --
ELEGIDO, no ajustado para producir un resultado; se declara aquí explícitamente.
"""
import numpy as np
from scipy.ndimage import map_coordinates

MACH_OBJETIVO = 3.0   # elegido, típico de literatura -- ver docstring del módulo
PENDIENTE_ESPECTRO = 2.0   # Burgers, E(k) ~ k^-2, el estándar para gas supersónico


def _campo_grilla(ng, pendiente, seed):
    """3 campos Gaussianos con espectro de energía ~k^-pendiente, grilla Ng^3, periódica."""
    kfreq = np.fft.fftfreq(ng) * ng
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    kmag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    kmag_segura = np.where(kmag == 0, 1.0, kmag)
    amp = kmag_segura ** (-pendiente / 2.0)
    amp[kmag == 0] = 0.0   # sin componente de bulk (momento neto no se impone a mano)

    rng = np.random.default_rng(seed)
    campo = np.empty((ng, ng, ng, 3))
    for c in range(3):
        ruido = rng.normal(size=(ng, ng, ng)) + 1j * rng.normal(size=(ng, ng, ng))
        campo[..., c] = np.fft.ifftn(ruido * amp).real
    return campo


def campo_turbulento(pos, cs, seed, ng=32, mach=MACH_OBJETIVO, pendiente=PENDIENTE_ESPECTRO):
    """Genera el campo y lo interpola a `pos` (ya dilatadas por la expansión). La escala de la grilla
    periódica se toma del propio rango ocupado por `pos` (pos.ptp()) -- no requiere que el llamador
    conozca lado*a_final por separado."""
    lado_ef = float(np.ptp(pos))
    campo = _campo_grilla(ng, pendiente, seed)

    coords = ((pos - pos.min(axis=0)) % lado_ef) / lado_ef * ng
    vel = np.empty_like(pos)
    for c in range(3):
        vel[:, c] = map_coordinates(campo[..., c], coords.T, order=1, mode="wrap")

    v_rms_actual = np.sqrt(np.mean(np.sum(vel ** 2, axis=1)))
    v_rms_objetivo = mach * cs
    if v_rms_actual > 0:
        vel *= v_rms_objetivo / v_rms_actual
    return vel


def factory(cs, seed, ng=32, mach=MACH_OBJETIVO, pendiente=PENDIENTE_ESPECTRO):
    """vel_generador(pos, adj, dens_bar) -> vel, para pasar a traducir_pool. adj/dens_bar se ignoran a
    propósito -- el Brazo 1 es un campo importado, independiente de la malla causal (esa independencia
    ES el punto: REAL y NULL comparten la MISMA realización de turbulencia, seed fija, sólo cambia la
    posición/topología subyacente, igual que en el resto del arco)."""
    def _gen(pos, adj, dens_bar):
        return campo_turbulento(pos, cs, seed, ng=ng, mach=mach, pendiente=pendiente)
    return _gen
