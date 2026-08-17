"""
control_nube_metrica.py — CONTROL POSITIVO: nube métrica con nuestros números (INSTRUCCION CS 20-jul,
"nube métrica con nuestros números").

Aísla UNA sola variable frente a fase1_traducir_a_phantom.py: la GEOMETRÍA del sustrato. Todo lo demás es
IDÉNTICO a la corrida real -- masa por partícula (leída del pool real, no hand-typed), masa total, misma
dilatación isótropa (a_final=√60, MISMO mecanismo p_expansion.Expansion + mismo reloj _T_reloj que el
resto del arco, nunca reimplementado), mismo polyk (leído del MISMO objeto EnfriamientoH2.T_piso), mismo
h uniforme (HFACT), mismos dt/tolv/tmax de Phantom (no se tocan), mismo criterio de núcleo (FoF con
linking_length=0.2*a_final, umbral 116).

Lo único que cambia: las posiciones NO vienen de la malla causal proyectada (p_semilla_causal.py) sino de
una esfera de densidad UNIFORME en el espacio métrico -- lo que Phantom presupone por diseño (un sistema
YA dado en una métrica, no emergente). Radio elegido para que el VOLUMEN de la esfera iguale lado^3 (el
mismo volumen característico que ocupa el layout de resortes antes de dilatar) -- no es una elección
libre, es la misma convención de escala que ya usa traducir_pool.

Dos lecturas pre-inscritas (INSTRUCCION CS 20-jul):
  - Si la nube métrica SÍ converge a tmax y cruza Jeans donde el sustrato causal falló en el primer paso
    => la diferencia es la naturaleza no-métrica/emergente del sustrato, no Phantom ni la resolución.
  - Si la nube métrica TAMBIÉN falla a los mismos N con las mismas tolerancias => es un límite de
    Phantom/paso de tiempo a esta escala, y la Teoría no queda implicada -- se reabre el análisis.

Es un control POSITIVO (¿colapsa cuando la métrica está dada?), no un intento de "lograr la estrella".
"""
import numpy as np

from cs073_cierre_holistico import T0, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion
from cs072_modulos.piezas.p_enfriamiento_H2 import EnfriamientoH2

HFACT = 1.2   # misma convención SPH que fase1_traducir_a_phantom.py
POLYK = EnfriamientoH2(n=1, T_inicial=T0).T_piso   # MISMO objeto, MISMO valor que la corrida real


def _esfera_uniforme(n, radio, seed):
    """N puntos uniformes en el VOLUMEN de una esfera de radio `radio` (método de Marsaglia: dirección
    gaussiana normalizada -- uniforme en la esfera unitaria -- por radio*u^(1/3) -- uniforme en volumen,
    no en superficie). El estándar de campo, no un método inventado para este control."""
    rng = np.random.default_rng(seed)
    direcciones = rng.normal(size=(n, 3))
    direcciones = direcciones / np.linalg.norm(direcciones, axis=1, keepdims=True)
    u = rng.uniform(0.0, 1.0, size=n)
    r = radio * u ** (1.0 / 3.0)
    return direcciones * r[:, None]


def _a_final(n_pasos=60):
    """MISMO mecanismo que traducir_pool (fase1_traducir_a_phantom.py) -- p_expansion.Expansion, mismo
    reloj T0/_T_reloj, nunca reimplementado ni reajustado para este control."""
    expansion = Expansion(T0=T0)
    for step in range(n_pasos):
        expansion.paso_de_estiramiento(_T_reloj(step))
    return expansion._a_prev


def escribir_ic_control(n, masa_particula, seed=12345, ruta_salida="cosmogenesis_ic.txt"):
    """masa_particula se recibe (no se hand-typea) -- léela del mismo pool real usado en la corrida
    causal, para que la masa total/por partícula sea IDÉNTICA, no una aproximación."""
    lado = float(n) ** (1.0 / 3.0)   # mismo volumen característico que ocupa layout_resortes
    radio = lado * (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)   # volumen esfera == lado**3
    pos = _esfera_uniforme(n, radio, seed=seed)

    a_fin = _a_final()
    pos = pos * a_fin   # MISMA dilatación isótropa estática que traducir_pool -- mismo a_final=√60

    vel = np.zeros_like(pos)
    h_guess = np.full(n, HFACT)

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 -- CONTROL nube metrica (esfera uniforme) -- npart={n} "
                 f"masa_particula={masa_particula:.17g} hfact={HFACT} polyk={POLYK:.17g}\n")
        f.write(f"{n} {masa_particula:.17g} {HFACT} {POLYK:.17g}\n")
        for i in range(n):
            f.write(f"{float(pos[i, 0]):.17g} {float(pos[i, 1]):.17g} {float(pos[i, 2]):.17g} "
                     f"{float(vel[i, 0]):.17g} {float(vel[i, 1]):.17g} {float(vel[i, 2]):.17g} "
                     f"{float(h_guess[i]):.17g}\n")
    return dict(ruta=ruta_salida, n=n, masa_particula=masa_particula, lado=lado, radio=radio,
                polyk=POLYK, a_final=a_fin, seed=seed)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    masa_particula = float(sys.argv[2])
    ruta = sys.argv[3] if len(sys.argv) > 3 else "cosmogenesis_ic.txt"
    info = escribir_ic_control(n, masa_particula, ruta_salida=ruta)
    print(info)
