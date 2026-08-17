"""
null2_zeldovich_generar_ic.py — mejora del método de conversión campo->partícula de NULL-2.

Contexto (ver NULL2_piloto_espectro_potencia_CS.md, el informe del agente anterior): el método
NULL-2 original (`null2_generar_ic.py`) preserva P(k) EXACTO a nivel de grilla (diferencia ~1e-16),
pero al convertir el campo sintético en partículas por MUESTREO POR RECHAZO/INVERSIÓN (cada
partícula se resamplea de la PMF de densidad, independiente de sus vecinas), el catálogo de
partículas resultante NO preserva ξ(r) de REAL (KS=0.495, p≈0) -- porque a N=500-2000 con una
grilla razonable, la ocupación media es de sólo 0.03-0.5 partículas/celda: el ruido de Poisson de
MUESTREAR partículas independientes domina sobre la señal de dos puntos que se quería trasladar.

Este módulo implementa la alternativa estándar de la literatura de condiciones iniciales
cosmológicas (N-GenIC, 2LPTic y sucesores): el DESPLAZAMIENTO DE ZEL'DOVICH. En vez de resamplear
partículas desde cero, se parte de un conjunto de partículas YA HOMOGÉNEO (sin estructura -- una
grilla regular jitterizada, ver `grilla_no_perturbada`) y se las DESPLAZA suavemente según un campo
de desplazamiento continuo Psi(q) calculado a partir de la MISMA delta-hat de fase aleatorizada que
ya usa `null2_generar_ic.aleatorizar_fases` (reusada tal cual, no reescrita):

    laplaciano(phi) = delta   =>   phi_hat(k) = -delta_hat(k) / |k|^2
    Psi = -grad(phi)          =>   Psi_hat_i(k) = i*k_i/|k|^2 * delta_hat(k)

(el modo k=0 se fija a 0: es sólo un desplazamiento global de toda la nube, sin efecto sobre ξ(r)
relativo al centro de masa). Cada partícula se mueve a x = q + Psi(q), interpolando el campo
CONTINUO de desplazamiento en su posición q (trilineal, `scipy.ndimage.map_coordinates`).

Por qué esto debería preservar mejor ξ(r) a nivel de partícula: cada partícula se desplaza según un
campo SUAVE que varía poco entre puntos vecinos -- dos partículas que arrancan cerca (mismo q) reciben
desplazamientos casi idénticos, así que su distancia relativa cambia POCO. Eso es exactamente lo
opuesto al muestreo por rechazo, donde dos partículas "vecinas" en el campo final no tienen ninguna
relación entre sí (cada una se resampleó de forma independiente). El campo se mueve, no se re-tira.

Punto de partida NO perturbado (`grilla_no_perturbada`): grilla regular jitterizada dentro de la
misma caja que ocupa REAL (mismo origin/half_extent que `gridizar`), recortada a exactamente n
partículas. Se eligió la grilla regular -- la opción MÁS SIMPLE de las dos que sugería el encargo --
en vez de "radios reales + ángulos uniformes" (que sería reusar la cáscara de NULL-1) porque
Zel'dovich asume que el punto de partida q es HOMOGÉNEO: si q ya trae la forma de cáscara de REAL,
cualquier ξ(r) parecido a REAL que resultara sería un artefacto de esa herencia, no una prueba de que
el desplazamiento por sí solo reproduce la estructura de dos puntos -- justo lo que este método
necesita demostrar honestamente.

Advertencia de método documentada (no oculta): el campo de Fourier es periódico por construcción
(FFT), pero la caja de REAL (una cáscara finita) no lo es en sentido físico estricto -- se interpola
con `mode="wrap"` (condición de contorno periódica), igual que asume cualquier generador de tipo
N-GenIC sobre su caja de simulación. A este N tan chico y sin caja verdaderamente periódica, el
"wrap" es una aproximación, no una verdad; se deja constancia.

No toca null2_generar_ic.py, null2_disenar_verificar.py, null2_piloto_generar.py,
null2_piloto_correr.py, ni ningún archivo bajo bateria_n2000/ / piloto_null1/ / piloto_null2/ (sólo
los importa/lee). El método anterior queda intacto como referencia de lo que no funcionó bien a
este N.
"""
import numpy as np
from scipy.ndimage import map_coordinates

from null2_generar_ic import gridizar, aleatorizar_fases, verificar_dos_puntos_particulas


# ------------------------------------------------------------------------------------------------
# 1) Campo de desplazamiento de Zel'dovich a partir del campo sintético (misma delta-hat de fase
#    aleatorizada que ya calcula aleatorizar_fases -- no se reinventa esa parte).
# ------------------------------------------------------------------------------------------------
def campo_desplazamiento_zeldovich(campo_sint, cell_size):
    """Devuelve (psi_x, psi_y, psi_z): campos reales del mismo shape que campo_sint con el
    desplazamiento de Zel'dovich Psi(q) = -grad(phi), laplaciano(phi) = delta, resuelto en Fourier
    como Psi_hat_i(k) = i*k_i/|k|^2 * delta_hat(k). El modo k=0 (traslación global de toda la nube,
    irrelevante para cualquier estadística relativa al centro de masa) se fija a 0 explícitamente."""
    ngrid = campo_sint.shape[0]
    media = campo_sint.mean()
    delta = campo_sint / media - 1.0 if media != 0 else campo_sint * 0.0
    delta_hat = np.fft.fftn(delta)

    kfreq = np.fft.fftfreq(ngrid, d=cell_size) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    k2_seguro = np.where(k2 == 0, 1.0, k2)  # evita división por cero en k=0 (se pisa el modo abajo)

    psi_hat_x = 1j * kx / k2_seguro * delta_hat
    psi_hat_y = 1j * ky / k2_seguro * delta_hat
    psi_hat_z = 1j * kz / k2_seguro * delta_hat
    psi_hat_x[0, 0, 0] = 0
    psi_hat_y[0, 0, 0] = 0
    psi_hat_z[0, 0, 0] = 0

    psi_x = np.fft.ifftn(psi_hat_x).real
    psi_y = np.fft.ifftn(psi_hat_y).real
    psi_z = np.fft.ifftn(psi_hat_z).real
    return psi_x, psi_y, psi_z


# ------------------------------------------------------------------------------------------------
# 2) Punto de partida sin estructura (homogéneo) + interpolación trilineal del campo continuo.
# ------------------------------------------------------------------------------------------------
def grilla_no_perturbada(n, origin, cell_size, ngrid, seed):
    """Grilla regular jitterizada de lado ~n^(1/3) dentro de la misma caja que gridizar() usó para
    REAL, recortada a exactamente n partículas. Ver docstring del módulo para por qué NO se usa la
    cáscara radial de REAL/NULL-1 como punto de partida."""
    rng = np.random.default_rng(seed)
    lado = int(np.ceil(n ** (1.0 / 3.0)))
    extent = ngrid * cell_size
    ejes = (np.arange(lado) + 0.5) * (extent / lado)
    qx, qy, qz = np.meshgrid(ejes, ejes, ejes, indexing="ij")
    q_local = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)
    if len(q_local) > n:
        idx = rng.choice(len(q_local), size=n, replace=False)
        q_local = q_local[idx]
    jitter_escala = extent / lado
    q_local = q_local + rng.uniform(-0.5, 0.5, size=q_local.shape) * jitter_escala * 0.9
    return q_local + origin  # coordenadas absolutas, mismo sistema que origin/gridizar


def interpolar_trilineal(campo, q_local, cell_size):
    """Interpola trilinealmente (scipy.ndimage.map_coordinates, order=1) un campo de grilla en las
    posiciones q_local (coordenadas relativas al origin del cubo, unidades físicas). mode='wrap'
    porque el campo de Fourier es periódico por construcción -- ver advertencia de método en el
    docstring del módulo."""
    idx_frac = q_local / cell_size  # coordenadas de índice de grilla (float, no enteras)
    coords = idx_frac.T  # (3, n): formato que exige map_coordinates
    return map_coordinates(campo, coords, order=1, mode="wrap")


# ------------------------------------------------------------------------------------------------
# 3) Orquestador: grilla REAL -> aleatoriza fases (reusado) -> campo de desplazamiento -> grilla
#    no perturbada -> desplaza.
# ------------------------------------------------------------------------------------------------
def generar_null2_zeldovich(pos_real, n_salida, ngrid, seed, pad=1.05, seed_q=None):
    """Orquesta el método Zel'dovich completo. Devuelve dict con las posiciones finales `pos`, el
    punto de partida `q`, el desplazamiento aplicado, y diagnósticos (residuo imaginario de la
    aleatorización de fase, RMS del desplazamiento)."""
    if seed_q is None:
        seed_q = seed + 100000  # separada de la semilla de fase, misma convención del resto del proyecto
    campo, cell_size, origin, centro, half_extent = gridizar(pos_real, ngrid, pad=pad)
    campo_sint, residuo_imag = aleatorizar_fases(campo, seed)
    psi_x, psi_y, psi_z = campo_desplazamiento_zeldovich(campo_sint, cell_size)

    q = grilla_no_perturbada(n_salida, origin, cell_size, ngrid, seed_q)
    q_local = q - origin

    dx = interpolar_trilineal(psi_x, q_local, cell_size)
    dy = interpolar_trilineal(psi_y, q_local, cell_size)
    dz = interpolar_trilineal(psi_z, q_local, cell_size)
    desplazamiento = np.stack([dx, dy, dz], axis=1)

    pos_final = q + desplazamiento
    return dict(pos=pos_final, q=q, desplazamiento=desplazamiento, cell_size=cell_size,
                origin=origin, centro=centro, half_extent=half_extent, residuo_imag=residuo_imag,
                ngrid=ngrid, seed=seed, seed_q=seed_q,
                desplazamiento_rms=float(np.sqrt((desplazamiento ** 2).sum(axis=1).mean())))


if __name__ == "__main__":
    import sys
    print("Uso como módulo -- ver null2_zeldovich_disenar_verificar.py (verificación) y "
          "null2_zeldovich_piloto_generar.py (piloto Phantom, si la verificación lo justifica).")
    sys.exit(0)
