"""
grafo_random_piloto_generar.py — orquestador del PILOTO chico del control grafo-random (Erdős-Rényi,
independiente de REAL), N=500, 3 semillas -- mismo patrón que `null3_piloto_generar.py`/
`null1_piloto_generar.py`/`null2_piloto_generar.py`.

Extrae el pool de bariones a N=500 con los MISMOS parámetros que `null1_piloto_generar.py`/
`null3_piloto_generar.py` (nq=3000, naq=2100, ne=1000, npos=700 -- determinista, reproduce el mismo
`masa_bar`/`dens_bar` que ya generó `/Users/alexis/phantom_cs073/piloto_null1/real/cosmogenesis_ic.txt`),
cuenta nodos/aristas de la malla causal REAL a esta escala (SÓLO para dimensionar el grafo random --
mismo n, misma m aproximada), y genera 3 condiciones de control con semillas NUEVAS (901, 902, 903 --
no chocan con 101-103 de NULL-1, 201-203/301-303 de NULL-2, 601-603 de NULL-3) con el MISMO campo de
velocidad turbulento (Mach=3, seed=42) que REAL/NULL-1/NULL-2/NULL-3.

Escribe en /Users/alexis/phantom_cs073/piloto_grafo_random/random_s{1,2,3}/cosmogenesis_ic.txt (carpeta
NUEVA). No toca ninguna carpeta de piloto/batería anterior ni ningún script congelado -- sólo los
importa/lee. No corre Phantom (ver grafo_random_piloto_correr.py).
"""
import os
import time

import numpy as np

from cs073_cierre_holistico import _extraer_bariones
from null1_generar_ic import leer_ic_txt
from grafo_random_layout_generar_ic import contar_aristas_malla_real, generar_control_random
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import POLYK

BASE = "/Users/alexis/phantom_cs073/piloto_grafo_random"
RUTA_REAL_N500 = "/Users/alexis/phantom_cs073/piloto_null1/real/cosmogenesis_ic.txt"
CS_SONORA = POLYK ** 0.5
TURB_SEED = 42
SEMILLAS = (901, 902, 903)


def main():
    t0 = time.time()
    print("[pool] extrayendo bariones (nq=3000,naq=2100,ne=1000,npos=700) -- MISMOS parámetros que "
          "null1_piloto_generar.py/null3_piloto_generar.py, determinista...", flush=True)
    masa_bar, dens_bar, obs = _extraer_bariones(3000, 2100, 1000, 700, 150, 1.5)
    print(f"[pool] n_atomos={len(masa_bar)} H={obs.get('hidrogeno')} He={obs.get('helio')} "
          f"tiempo={time.time()-t0:.1f}s", flush=True)

    print(f"\n[real] leyendo REAL N=500 ya en disco (sólo para r_mean/r_std de referencia, no se usa "
          f"para construir el grafo random): {RUTA_REAL_N500}", flush=True)
    pos_real, vel_real, h_real, masa_particula, n = leer_ic_txt(RUTA_REAL_N500)
    assert n == len(masa_bar), (
        f"n de piloto_null1/real ({n}) != n del pool recién extraído ({len(masa_bar)}) -- el pool NO "
        "es determinista como se asumía, no seguir a ciegas.")
    com_real = pos_real.mean(axis=0)
    r_real = np.linalg.norm(pos_real - com_real, axis=1)
    print(f"    n={n} masa_particula={masa_particula:.6g} r_mean={r_real.mean():.3f} "
          f"r_std={r_real.std():.3f}", flush=True)

    print(f"\n[malla REAL a N=500] contando nodos/aristas SÓLO para dimensionar el grafo random "
          f"(no se usa ninguna arista de acá)...", flush=True)
    n_malla, m_malla, _adj_real = contar_aristas_malla_real(dens_bar)
    print(f"    n={n_malla} n_aristas={m_malla} grado_medio={2*m_malla/n_malla:.3f}", flush=True)

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)
    os.makedirs(BASE, exist_ok=True)

    for i, seed in enumerate(SEMILLAS, start=1):
        t1 = time.time()
        os.makedirs(f"{BASE}/random_s{i}", exist_ok=True)
        info = generar_control_random(masa_bar, dens_bar, n_aristas=m_malla, seed_random=seed,
                                       vel_generador=vel_gen,
                                       ruta_salida=f"{BASE}/random_s{i}/cosmogenesis_ic.txt")
        pos_r = info["pos"]
        r_r = np.linalg.norm(pos_r - pos_r.mean(axis=0), axis=1)
        print(f"[random_s{i}] seed_random={seed} seed_layout={info['seed_layout']} "
              f"n_aristas={info['n_aristas']} grado_medio={info['grado_medio']:.3f} "
              f"r_mean={r_r.mean():.3f} r_std={r_r.std():.3f} (REAL: {r_real.mean():.3f}/"
              f"{r_real.std():.3f}) tiempo={time.time()-t1:.1f}s -> "
              f"{BASE}/random_s{i}/cosmogenesis_ic.txt", flush=True)

    print(f"\n[TOTAL] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
