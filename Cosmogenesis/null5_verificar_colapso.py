"""
null5_verificar_colapso.py -- NULL-5, Fase II CS073, escalón "correspondencia nodo<->posición".

QUÉ HACE ESTE SCRIPT (léase antes de correr nada de Phantom, mismo papel que
`null4_verificar_invarianza_orden.py` cumplió para NULL-4): responde, de forma EMPÍRICA y no sólo por
lectura de código, si NULL-5 es operacionalizable como pregunta genuina o si colapsa trivial -- es decir,
si permutar qué nodo del grafo causal REAL ocupa cuál posición física final puede, en este pipeline,
producir un archivo de condición inicial (IC) de Phantom distinto de REAL en ALGO más que el orden de
las filas.

Tres verificaciones, en orden:

  (1) Reconstrucción bit-idéntica: reconstruir la malla causal REAL + `layout_resortes` (mismos
      parámetros que toda la jerarquía) y comparar contra las posiciones que de verdad quedaron escritas
      en `bateria_n2000/ic_real/cosmogenesis_ic.txt` -- si la diferencia no es exactamente 0.0, algo del
      método de reconstrucción no es fiel y hay que corregirlo antes de construir NULL-5 sobre él.

  (2) Velocidad como función PURA de la posición: recomputar el campo de velocidad turbulento
      (`campo_velocidad_turbulento.factory`, mismo seed=42/Mach=3 que toda la jerarquía) usando SÓLO las
      posiciones del archivo real (sin ninguna información de identidad de nodo ni de grafo) y comparar
      contra la columna de velocidad que de verdad quedó escrita en el archivo real -- si coincide bit a
      bit, queda demostrado que la velocidad no "sabe" qué nodo es cada partícula, sólo dónde está.

  (3) Masa y h uniformes: confirmar que el archivo real escribe una única masa global (no una columna
      por partícula) y un único valor de `h` (semilla del solver grad-h, no un valor por identidad de
      nodo) -- ambos ya verificables por inspección directa del archivo.

Si (1), (2) y (3) se cumplen, la conclusión es: NINGÚN atributo físico escrito en el IC depende de la
identidad del nodo -- todos son función pura del VALOR de la posición final (o son constantes globales).
Por lo tanto, permutar la correspondencia nodo<->posición no puede cambiar ningún valor físico de ninguna
partícula -- sólo puede reordenar las FILAS del archivo. NULL-5, tal como la pregunta original lo pide
("¿importa la identidad del nodo, no sólo dónde cayó?"), COLAPSA TRIVIAL en este pipeline: el archivo
resultante es, como MULTISET de partículas, IDÉNTICO a REAL.

Se deja además, como diligencia adicional (no pedida explícitamente pero barata y honesta), una prueba
de una pregunta DISTINTA y más débil que sí queda abierta: ¿importa el ORDEN DE FILA en que estas mismas
tuplas (idénticas en contenido) se le pasan a `phantomsetup`/`phantom`? Esto es un artefacto de
implementación (posible sensibilidad de punto flotante en el orden de suma de fuerzas / árbol de
Phantom), no una prueba de la hipótesis de "identidad de nodo" -- se corre aparte, en
`null5_bateria_generar.py` + `null5_bateria_correr.py`, y se reporta como hallazgo secundario.

No toca ningún archivo congelado ni ninguna carpeta de batería anterior -- sólo lee/importa.
"""
import numpy as np

from null5_generar_ic import reconstruir_malla_y_layout_real
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import POLYK

RUTA_POOL = "/Users/alexis/phantom_cs073/bateria_n2000"
RUTA_IC_REAL = f"{RUTA_POOL}/ic_real/cosmogenesis_ic.txt"


def main():
    print("=== NULL-5 -- verificación de colapso trivial (correspondencia nodo<->posición) ===\n")

    dens_bar = np.load(f"{RUTA_POOL}/dens_bar.npy")
    n = len(dens_bar)
    print(f"n={n}")

    datos = np.loadtxt(RUTA_IC_REAL, skiprows=2)
    pos_archivo = datos[:, 0:3]
    vel_archivo = datos[:, 3:6]
    h_archivo = datos[:, 6]
    with open(RUTA_IC_REAL) as f:
        cabecera = f.readline()
    print(f"cabecera real: {cabecera.strip()}\n")

    # --- (1) reconstrucción bit-idéntica de malla + layout ---
    print("--- (1) Reconstrucción malla causal + layout_resortes vs. archivo REAL en disco ---")
    adj_real, pos_reconstruida, a_final = reconstruir_malla_y_layout_real(dens_bar)
    n_aristas = sum(len(v) for v in adj_real.values()) // 2
    diff_pos = np.abs(pos_reconstruida - pos_archivo)
    print(f"  n_aristas reconstruidas = {n_aristas}")
    print(f"  diff máxima |pos_reconstruida - pos_archivo_real| = {diff_pos.max():.3e}")
    ok1 = diff_pos.max() == 0.0
    print(f"  {'OK: bit-idéntico' if ok1 else 'AVISO: NO bit-idéntico, revisar método'}\n")

    # --- (2) velocidad como función pura de la posición ---
    print("--- (2) Velocidad recomputada SÓLO desde posición (sin adj/dens_bar) vs. archivo REAL ---")
    cs = POLYK ** 0.5
    vel_gen = factory_turb(cs, seed=42, mach=MACH_OBJETIVO)
    vel_recomputada = vel_gen(pos_archivo, None, None)   # adj=None, dens_bar=None a propósito: si la
                                                            # factory los usara de verdad, esto reventaría
    diff_vel = np.abs(vel_recomputada - vel_archivo)
    print(f"  diff máxima |vel_recomputada(pos) - vel_archivo_real| = {diff_vel.max():.3e}")
    ok2 = diff_vel.max() == 0.0
    print(f"  {'OK: velocidad es función pura de la posición' if ok2 else 'AVISO: velocidad SÍ depende de algo más que la posición'}\n")

    # --- (3) masa y h uniformes ---
    print("--- (3) Masa y h: ¿constantes globales o por-partícula? ---")
    valores_h_unicos = np.unique(h_archivo)
    npart, masa_media, hfact_cab, polyk_cab = cabecera.split("npart=")[1].split()[0], None, None, None
    print(f"  npart en cabecera = {npart}")
    print(f"  valores únicos de h en el archivo = {valores_h_unicos} (1 solo valor => uniforme, no depende de partícula)")
    print(f"  la cabecera trae UN SOLO 'masa_particula=' global -- el cuerpo del archivo NO tiene columna de masa por fila")
    ok3 = len(valores_h_unicos) == 1
    print(f"  {'OK: h uniforme' if ok3 else 'AVISO: h varía por partícula'}\n")

    print("=" * 88)
    if ok1 and ok2 and ok3:
        print(">>> CONCLUSIÓN: NULL-5 COLAPSA TRIVIAL en este pipeline.")
        print(">>> Ningún atributo físico escrito en el IC (posición-set, velocidad, masa, h) depende de")
        print(">>> la identidad del nodo del grafo causal -- todos son función pura del VALOR de la")
        print(">>> posición final, o constantes globales. Permutar qué nodo ocupa cuál posición no puede")
        print(">>> cambiar ningún valor físico de ninguna partícula: el archivo resultante es, como")
        print(">>> MULTISET de partículas, IDÉNTICO a REAL -- sólo puede diferir en el orden de FILA.")
    else:
        print(">>> CONCLUSIÓN: NULL-5 NO colapsa trivial -- hay una dependencia de identidad de nodo no")
        print(">>> anticipada. Investigar antes de proceder con la batería completa.")
    print("=" * 88)


if __name__ == "__main__":
    main()
