"""
null3_dosis_piloto_generar.py — Parte B, piloto Phantom (N=500) para la curva dosis-respuesta de
`tol_relativa`. Mismo patrón EXACTO que `null3_piloto_generar.py` (congelado, no se toca): extrae el
mismo pool N=500 determinista (nq=3000,naq=2100,ne=1000,npos=700), reusa
`piloto_null1/real/cosmogenesis_ic.txt` ya en disco como REAL de referencia y como `pos_referencia` del
filtro de longitud, y genera condiciones NULL-3 con `generar_null3` -- pero para dos niveles de
`tol_relativa` MÁS PERMISIVOS que el 0.2 ya piloteado (0.4 y 0.8), con semillas nuevas que no chocan con
ninguna otra rama (701/702 para tol=0.4, 801/802 para tol=0.8).

El nivel "sin filtro" (Maslov-Sneppen puro) de la curva de dosis-respuesta NO se pilotea aquí: ya existe
un dato conocido y directamente comparable en la jerarquía -- los NULL1-8 originales de CS073
(`bateria_n2000/ic_null1..8`), que usan exactamente ese mecanismo (swap sin restricción de longitud
sobre la misma malla causal) y ya dieron masa promedio en sumideros ≈680-770 (parcial, no cero, pero muy
por debajo de REAL≈2190) con KS<1e-113 en perfil radial. Repetirlo aquí a N=500 gastaría presupuesto sin
agregar información nueva sobre el mecanismo.

Escribe en /Users/alexis/phantom_cs073/piloto_null3_dosis/tol{0.4,0.8}_s{1,2}/cosmogenesis_ic.txt
(carpeta NUEVA). No toca piloto_null1/, piloto_null3/, ni ningún script/carpeta congelados -- sólo los
importa/lee.

No corre Phantom (ver null3_dosis_piloto_correr.py).
"""
import os
import time

import numpy as np

from cs073_cierre_holistico import _extraer_bariones
from null1_generar_ic import leer_ic_txt
from null3_generar_ic import generar_null3
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import POLYK

BASE = "/Users/alexis/phantom_cs073/piloto_null3_dosis"
RUTA_REAL_N500 = "/Users/alexis/phantom_cs073/piloto_null1/real/cosmogenesis_ic.txt"
CS_SONORA = POLYK ** 0.5
TURB_SEED = 42

# (tol_relativa, [semillas]) -- una semilla por nivel para mantenerse dentro del presupuesto de tiempo;
# se puede ampliar a 2-3 si el tiempo alcanza (ver SEMILLAS_EXTRA más abajo, comentado).
NIVELES = [(0.4, [701]), (0.8, [801])]


def main():
    t0 = time.time()
    print("[pool] extrayendo bariones (nq=3000,naq=2100,ne=1000,npos=700) -- MISMOS parámetros que "
          "null3_piloto_generar.py, determinista...", flush=True)
    masa_bar, dens_bar, obs = _extraer_bariones(3000, 2100, 1000, 700, 150, 1.5)
    print(f"[pool] n_atomos={len(masa_bar)} tiempo={time.time()-t0:.1f}s", flush=True)

    print(f"\n[real] leyendo REAL N=500 ya en disco: {RUTA_REAL_N500}", flush=True)
    pos_real, vel_real, h_real, masa_particula, n = leer_ic_txt(RUTA_REAL_N500)
    assert n == len(masa_bar), (
        f"n de piloto_null1/real ({n}) != n del pool recién extraído ({len(masa_bar)})")
    com_real = pos_real.mean(axis=0)
    r_real = np.linalg.norm(pos_real - com_real, axis=1)
    print(f"    n={n} r_mean={r_real.mean():.3f} r_std={r_real.std():.3f}", flush=True)

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)

    for tol, semillas in NIVELES:
        for i, seed in enumerate(semillas, start=1):
            t1 = time.time()
            carpeta = f"{BASE}/tol{tol}_s{i}"
            os.makedirs(carpeta, exist_ok=True)
            info = generar_null3(masa_bar, dens_bar, pos_real, seed_null3=seed,
                                  tol_relativa=tol, vel_generador=vel_gen,
                                  ruta_salida=f"{carpeta}/cosmogenesis_ic.txt")
            pos_n3 = info["pos"]
            r_n3 = np.linalg.norm(pos_n3 - pos_n3.mean(axis=0), axis=1)
            print(f"[tol={tol} s{i}] seed={seed} swap={info['swap_aceptados']}/{info['swap_intentos']} "
                  f"({100*info['swap_aceptados']/info['swap_intentos']:.1f}%) "
                  f"r_mean={r_n3.mean():.3f} r_std={r_n3.std():.3f} "
                  f"(REAL: {r_real.mean():.3f}/{r_real.std():.3f}) "
                  f"tiempo={time.time()-t1:.1f}s -> {carpeta}/cosmogenesis_ic.txt", flush=True)

    print(f"\n[TOTAL] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
