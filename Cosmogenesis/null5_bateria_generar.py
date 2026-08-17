"""
null5_bateria_generar.py -- genera las condiciones iniciales NULL-5 (N=2000), a partir de la conclusión
de `null5_verificar_colapso.py`: como NULL-5 colapsa trivial (ningún atributo físico del IC depende de
la identidad del nodo, ver ese script), lo único que la permutación nodo<->posición puede todavía
cambiar es el ORDEN DE FILA en que las mismas tuplas (posición,velocidad,masa,h) -- idénticas en
contenido a REAL -- se escriben en el archivo ASCII que lee `phantomsetup`.

Por eso esta batería es DELIBERADAMENTE chica (2 semillas de permutación, no 8 ni 3): no está poniendo a
prueba la hipótesis original de NULL-5 (ya resuelta analítica + empíricamente: colapsa trivial, "sí
importa la identidad de nodo" queda refutado por construcción del pipeline) -- está poniendo a prueba
una pregunta SECUNDARIA y más débil que queda abierta como diligencia honesta: ¿el orden de FILA en el
archivo ASCII (sin cambiar ningún valor físico) le importa a `phantomsetup`/`phantom` por algún motivo
de implementación (orden de suma en punto flotante, construcción del árbol, etc.)? Si la respuesta es
"no" (esperado), confirma la trivialidad end-to-end, no sólo en el archivo de texto sino en el
comportamiento real de Phantom. Si la respuesta es "sí", es un hallazgo de artefacto de implementación
interesante mencionado aparte -- NUNCA una confirmación de la hipótesis de identidad de nodo, que ya
quedó descartada por construcción.

Escribe en `/Users/alexis/phantom_cs073/bateria_null5_n2000/ic_null5_s{801,802}/cosmogenesis_ic.txt`
(carpeta NUEVA). No toca ninguna carpeta de batería anterior ni ningún script congelado -- sólo importa/lee.
"""
import os
import time

import numpy as np

from null5_generar_ic import generar_null5
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import POLYK

RUTA_DENS_BAR = "/Users/alexis/phantom_cs073/bateria_n2000/dens_bar.npy"
RUTA_MASA_BAR = "/Users/alexis/phantom_cs073/bateria_n2000/masa_bar.npy"
BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_null5_n2000"
SEMILLAS = [801, 802]   # semillas de PERMUTACIÓN nodo<->posición (no de reordenamiento de aristas como
                          # NULL-4 -- acá la malla/layout es la reconstrucción REAL exacta bit a bit,
                          # sólo cambia qué fila del archivo recibe cuál posición)

CS_SONORA = POLYK ** 0.5
TURB_SEED = 42


def main():
    t0 = time.time()
    print("[pool] leyendo pool N=2000 ya en disco (sólo lectura)...", flush=True)
    dens_bar = np.load(RUTA_DENS_BAR)
    masa_bar = np.load(RUTA_MASA_BAR)
    print(f"[pool] n={len(dens_bar)} tiempo={time.time()-t0:.2f}s", flush=True)

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)
    os.makedirs(BASE_SALIDA, exist_ok=True)

    for seed in SEMILLAS:
        t1 = time.time()
        carpeta = f"{BASE_SALIDA}/ic_null5_s{seed}"
        os.makedirs(carpeta, exist_ok=True)
        info = generar_null5(masa_bar, dens_bar, seed_permutacion=seed, vel_generador=vel_gen,
                              ruta_salida=f"{carpeta}/cosmogenesis_ic.txt")
        pos_n5 = info["pos"]
        r_n5 = np.linalg.norm(pos_n5 - pos_n5.mean(axis=0), axis=1)
        print(f"[ic_null5_s{seed}] n_aristas={info['n_aristas']} (assert multiset pos+vel==REAL paso OK) "
              f"r_mean={r_n5.mean():.3f} r_std={r_n5.std():.3f} "
              f"tiempo={time.time()-t1:.1f}s -> {carpeta}/cosmogenesis_ic.txt", flush=True)

    print(f"\n[TOTAL generacion IC] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
