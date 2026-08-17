"""
null4_bateria_generar.py — NULL-4 (topología completa idéntica a REAL, orden de inserción rebarajado)
escalado a N=2000, 3 semillas de reordenamiento, Fase II CS073, escalón 5 de 6.

Por qué directo a N=2000, sin piloto N=500 (a diferencia de NULL-1/2/3): el propio plan aprobado por
Alexis lo indica explícitamente -- ya se sabe por `MISTERIO_N500_vs_N2000_CS.md` que N=500 falla por
resolución, no aporta señal nueva sobre la pregunta de NULL-4 (orden de formación), y esta batería ya
viene precedida de su propia verificación empírica dedicada (`null4_verificar_invarianza_orden.py`, que
cumple el papel que cumplió el piloto N=500 para NULL-3: confirmar que el mecanismo produce una
diferencia no trivial antes de gastar cómputo Phantom).

Qué hace, mismo patrón que `null3_bateria_generar.py` (no reinventado): lee el pool N=2000 YA EXTRAÍDO
(`bateria_n2000/dens_bar.npy`/`masa_bar.npy`, sólo lectura). Para cada semilla de reordenamiento en
{601, 602, 603}: reconstruye la malla causal REAL exacta (`malla_causal_atomos`, D=3/k=4/seed_ejes=2000,
IDÉNTICA en las 3 -- nunca cambia el conjunto de aristas), rebaraja el ORDEN de inserción
(`construir_adj_en_orden`, la semilla distingue QUÉ permutación se usa), corre `layout_resortes`
(MISMA función y parámetros que REAL/NULL-1/2/3) con `seed_layout=12345` (default, idéntico a toda la
jerarquía), aplica la MISMA dilatación estática y el MISMO campo de velocidad turbulento (Mach=3,
TURB_SEED=42) que el resto de la jerarquía -- así NULL-4 sólo difiere de REAL en la única variable que
se quiere aislar (orden de formación), no en ningún otro parámetro físico.

Escribe en `/Users/alexis/phantom_cs073/bateria_null4_n2000/ic_null4_s{601,602,603}/cosmogenesis_ic.txt`
(carpeta NUEVA). No toca `bateria_n2000/`, `bateria_null1_n2000/`, `bateria_null2_n2000/`,
`bateria_null3_n2000/`, `bateria_real_extra_n2000/`, ni ningún script congelado -- sólo los importa/lee.

No corre Phantom (`null4_bateria_correr.py`, paso aparte).
"""
import os
import time

import numpy as np

from null4_generar_ic import generar_null4
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import POLYK

RUTA_DENS_BAR = "/Users/alexis/phantom_cs073/bateria_n2000/dens_bar.npy"
RUTA_MASA_BAR = "/Users/alexis/phantom_cs073/bateria_n2000/masa_bar.npy"
BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_null4_n2000"
SEMILLAS = [601, 602, 603]   # 3 semillas de REORDENAMIENTO (no de barajado de aristas -- el conjunto
                              # de aristas es el mismo en las 3, sólo cambia el orden de inserción)

CS_SONORA = POLYK ** 0.5
TURB_SEED = 42


def main():
    t0 = time.time()
    print(f"[pool] leyendo pool N=2000 ya en disco (sólo lectura)...", flush=True)
    dens_bar = np.load(RUTA_DENS_BAR)
    masa_bar = np.load(RUTA_MASA_BAR)
    print(f"[pool] n={len(dens_bar)} tiempo={time.time()-t0:.2f}s", flush=True)

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)
    os.makedirs(BASE_SALIDA, exist_ok=True)

    for seed in SEMILLAS:
        t1 = time.time()
        carpeta = f"{BASE_SALIDA}/ic_null4_s{seed}"
        os.makedirs(carpeta, exist_ok=True)
        info = generar_null4(masa_bar, dens_bar, seed_reorden=seed, vel_generador=vel_gen,
                              ruta_salida=f"{carpeta}/cosmogenesis_ic.txt")
        pos_n4 = info["pos"]
        r_n4 = np.linalg.norm(pos_n4 - pos_n4.mean(axis=0), axis=1)
        print(f"[ic_null4_s{seed}] n_aristas={info['n_aristas']} "
              f"r_mean={r_n4.mean():.3f} r_std={r_n4.std():.3f} "
              f"tiempo={time.time()-t1:.1f}s -> {carpeta}/cosmogenesis_ic.txt", flush=True)

    print(f"\n[TOTAL generacion IC] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
