"""
null1_bateria_generar.py — Fase II CS073, NULL-1 escalado a la batería completa (N=2000, 8 semillas).

Qué problema resuelve: el piloto (NULL1_piloto_distribucion_radial_CS.md) validó a N=500/3 semillas
que NULL-1 (mismo multiconjunto de radios que REAL, ángulo isótropo aleatorio) es el control aislado
que la jerarquía de 6 pide. Alexis autorizó escalar al mismo diseño que usó CS073 originalmente:
N=2000, 8 semillas NULL-1, comparado contra la corrida ic_real YA EXISTENTE de bateria_n2000 (no hace
falta re-correrla).

Qué hace, distinto del piloto: en vez de RE-GENERAR la corrida REAL con `traducir_pool` (como hacía
`radios_desde_real` en el piloto -- correcto mientras REAL no existía todavía a esa escala), este
script LEE DIRECTO las posiciones REALES ya escritas en
`/Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmogenesis_ic.txt` (sólo lectura, nunca se toca
ese archivo ni la carpeta) con `leer_ic_txt` (misma función congelada de `null1_generar_ic.py`).
De ahí calcula r_i = |pos_i - COM| -- el multiconjunto EXACTO de radios que NULL-1 debe heredar --
sin volver a correr `traducir_pool`/`malla_causal_atomos`/`layout_resortes` (evita cualquier duda
de si una segunda llamada con la misma seed_layout reproduce bit-a-bit la primera; leer el archivo ya
escrito no tiene ese riesgo).

Genera 8 condiciones iniciales NULL-1 (semillas angulares 201..208 -- deliberadamente distintas de las
101-103 ya usadas en el piloto, para no confundir corridas de ambas escalas), mismo campo de velocidad
turbulento que REAL y que el piloto (Mach=3, semilla=42 -- TURB_SEED, aísla la posición como único
grado de libertad que cambia), y las escribe en
`/Users/alexis/phantom_cs073/bateria_null1_n2000/ic_null1_s{1..8}/cosmogenesis_ic.txt` (carpeta nueva,
bateria_n2000/ NO se toca).

No reimplementa nada: importa `leer_ic_txt`/`generar_null1` de `null1_generar_ic.py` (congelado, no
se edita) y `factory`/`MACH_OBJETIVO` de `campo_velocidad_turbulento.py` (congelado, ya usado en la
batería original y en el piloto).

No corre Phantom -- eso es un paso aparte (`null1_bateria_correr.py`).
"""
import os
import time

import numpy as np

from null1_generar_ic import leer_ic_txt, generar_null1
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import POLYK

RUTA_IC_REAL = "/Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmogenesis_ic.txt"
BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_null1_n2000"
SEMILLAS_ANGULARES = list(range(201, 209))  # 8 semillas, distintas de 101-103 (piloto)

CS_SONORA = POLYK ** 0.5   # misma convención que null1_piloto_generar.py / scratch_generar_ic_velocidades.py
TURB_SEED = 42             # MISMA semilla de turbulencia que REAL (bateria_n2000) y que el piloto


def main():
    t0 = time.time()
    print(f"[real] leyendo posiciones REALES ya existentes de {RUTA_IC_REAL} (sólo lectura)...",
          flush=True)
    pos_real, vel_real, h_real, masa_particula, n = leer_ic_txt(RUTA_IC_REAL)
    com = pos_real.mean(axis=0)
    r = np.linalg.norm(pos_real - com, axis=1)
    print(f"[real] n={n} masa_particula={masa_particula:.6g} r_mean={r.mean():.3f} "
          f"r_std={r.std():.3f} com={com}", flush=True)

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)

    for i, seed in enumerate(SEMILLAS_ANGULARES, start=1):
        t1 = time.time()
        carpeta = f"{BASE_SALIDA}/ic_null1_s{i}"
        os.makedirs(carpeta, exist_ok=True)
        info = generar_null1(r, com, masa_particula, n, seed=seed, vel_generador=vel_gen,
                              ruta_salida=f"{carpeta}/cosmogenesis_ic.txt")
        print(f"[null1_s{i}] seed={seed} r_mean={info['r_mean']:.3f} r_std={info['r_std']:.3f} "
              f"tiempo={time.time()-t1:.2f}s -> {carpeta}/cosmogenesis_ic.txt", flush=True)

    print(f"[TOTAL generacion IC] {time.time()-t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
