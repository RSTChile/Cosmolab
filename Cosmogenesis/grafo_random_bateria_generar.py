"""
grafo_random_bateria_generar.py — control grafo-random (Erdős-Rényi, independiente de REAL) escalado a
la batería completa (N=2000, 8 semillas nuevas 701-708), Fase II CS073.

Mismo patrón EXACTO que `null3_bateria_generar.py`: lee directo el pool N=2000 YA EXTRAÍDO
(`bateria_n2000/dens_bar.npy`/`masa_bar.npy`, sólo lectura) y las posiciones REALES ya escritas (sólo
para r_mean/r_std de referencia en el log -- NUNCA usadas para construir el grafo random, a diferencia
de NULL-3 donde `pos_referencia` SÍ se usa para el filtro de longitud del swap). Para cada semilla en
701-708: cuenta la malla causal REAL UNA vez (n=2000, m=4945, siempre la misma -- sólo para dimensionar
el grafo random), genera un grafo Erdős-Rényi G(n,m) NUEVO por semilla (`generar_grafo_erdos_renyi`,
`seed_random=seed`), corre `layout_resortes` (MISMA función/parámetros que REAL/NULL-1/NULL-2/NULL-3,
`seed_layout=seed` -- variable por semilla, pedido explícito de la tarea), aplica la MISMA dilatación
estática y el MISMO campo de velocidad turbulento (Mach=3, TURB_SEED=42) que el resto de la jerarquía.

Escribe en
`/Users/alexis/phantom_cs073/bateria_grafo_random_n2000/ic_random_s{701..708}/cosmogenesis_ic.txt`
(carpeta NUEVA). No toca `bateria_n2000/`, `bateria_null1_n2000/`, `bateria_null2_n2000/`,
`bateria_null3_n2000/`, `bateria_real_extra_n2000/`, `piloto_grafo_random/`, ni ningún script congelado
-- sólo los importa/lee.

No corre Phantom (`grafo_random_bateria_correr.py`, paso aparte).
"""
import os
import time

import numpy as np

from null1_generar_ic import leer_ic_txt
from grafo_random_layout_generar_ic import contar_aristas_malla_real, generar_control_random
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import POLYK

RUTA_DENS_BAR = "/Users/alexis/phantom_cs073/bateria_n2000/dens_bar.npy"
RUTA_MASA_BAR = "/Users/alexis/phantom_cs073/bateria_n2000/masa_bar.npy"
RUTA_IC_REAL = "/Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmogenesis_ic.txt"
BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_grafo_random_n2000"
SEMILLAS = list(range(701, 709))   # 8 semillas nuevas, patrón pedido por la tarea

CS_SONORA = POLYK ** 0.5
TURB_SEED = 42


def main():
    t0 = time.time()
    print(f"[pool] leyendo pool N=2000 ya en disco (sólo lectura)...", flush=True)
    dens_bar = np.load(RUTA_DENS_BAR)
    masa_bar = np.load(RUTA_MASA_BAR)
    print(f"[pool] n={len(dens_bar)} tiempo={time.time()-t0:.2f}s", flush=True)

    print(f"[real] leyendo posiciones REALES ya existentes de {RUTA_IC_REAL} (sólo lectura, "
          f"referencia de comparación radial en el log -- NUNCA usada para construir el grafo "
          f"random)...", flush=True)
    pos_real, vel_real, h_real, masa_particula, n = leer_ic_txt(RUTA_IC_REAL)
    com_real = pos_real.mean(axis=0)
    r_real = np.linalg.norm(pos_real - com_real, axis=1)
    print(f"[real] n={n} masa_particula={masa_particula:.6g} r_mean={r_real.mean():.3f} "
          f"r_std={r_real.std():.3f}", flush=True)

    print(f"\n[malla REAL] contando n/aristas de la malla causal REAL (SÓLO para dimensionar el grafo "
          f"random -- misma n, misma m aproximada; ninguna arista se reutiliza)...", flush=True)
    n_malla, m_malla, _adj_real = contar_aristas_malla_real(dens_bar)
    print(f"[malla REAL] n={n_malla} n_aristas={m_malla} grado_medio={2*m_malla/n_malla:.3f}\n",
          flush=True)

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)
    os.makedirs(BASE_SALIDA, exist_ok=True)

    for seed in SEMILLAS:
        t1 = time.time()
        carpeta = f"{BASE_SALIDA}/ic_random_s{seed}"
        os.makedirs(carpeta, exist_ok=True)
        info = generar_control_random(masa_bar, dens_bar, n_aristas=m_malla, seed_random=seed,
                                       vel_generador=vel_gen,
                                       ruta_salida=f"{carpeta}/cosmogenesis_ic.txt")
        pos_r = info["pos"]
        r_r = np.linalg.norm(pos_r - pos_r.mean(axis=0), axis=1)
        print(f"[ic_random_s{seed}] seed_layout={info['seed_layout']} n_aristas={info['n_aristas']} "
              f"grado_medio={info['grado_medio']:.3f} "
              f"r_mean={r_r.mean():.3f} r_std={r_r.std():.3f} "
              f"(REAL: {r_real.mean():.3f}/{r_real.std():.3f}) "
              f"tiempo={time.time()-t1:.1f}s -> {carpeta}/cosmogenesis_ic.txt", flush=True)

    print(f"\n[TOTAL generacion IC] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
