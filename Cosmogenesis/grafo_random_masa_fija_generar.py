"""
grafo_random_masa_fija_generar.py — orquestador que genera condiciones iniciales de MASA TOTAL FIJA
(~18800, ver `grafo_random_layout_generar_ic_masa_fija.py`) para varios N, usando el mismo control
grafo-random (Erdős-Rényi + `layout_resortes`) que ya usa toda la jerarquía CS073. Script NUEVO, no
toca ningún congelado — sólo importa.

De dónde sale `dens_bar` para cada N (atajo deliberado, documentado, sólo para esta validación de
infraestructura — no es un pool físico nuevo por N):
  El pool REAL de N=2000 ya extraído (`bateria_n2000/dens_bar.npy`, motor basal ya corrido y validado)
  se SUBMUESTREA de forma determinista (rng fija, sin reemplazo) para obtener `dens_bar` a N<2000. Se
  eligió este atajo en vez de re-correr `_extraer_bariones` a cada N nuevo por presupuesto de tiempo:
  una prueba de re-extracción real a N=500 con los mismos parámetros que usa
  `grafo_random_piloto_generar.py` (nq=3000,naq=2100,ne=1000,npos=700) tardó 145.7s -- repetirlo para
  3-4 valores de N hubiera consumido la mayor parte del presupuesto de la tarea completa. Es válido acá
  porque `dens_bar` en este script SÓLO alimenta al campo de velocidad turbulento (pesa por densidad
  relativa) -- la masa y la caja, que es lo que este experimento corrige, ya NO dependen de `dens_bar`
  (ver `grafo_random_layout_generar_ic_masa_fija.py`). El SUBMUESTREO preserva la distribución de
  densidades del pool real, sólo reduce cuántas veces se la muestrea -- no inventa densidades.
  ADVERTENCIA para cualquier uso futuro más allá de esta validación de "¿fixea la masa el confound de
  sumideros?": si algún día importa la identidad/topología exacta del pool a cada N (no sólo su efecto
  sobre la velocidad turbulenta), este atajo NO alcanza -- haría falta re-extraer con
  `_extraer_bariones` como hace `grafo_random_piloto_generar.py`.

Escribe en
`/Users/alexis/phantom_cs073/bateria_grafo_random_masa_fija/ic_masaFija_N{n}_s{seed}/cosmogenesis_ic.txt`
(carpeta NUEVA). No corre Phantom (ver `grafo_random_masa_fija_correr.py`).
"""
import os
import time

import numpy as np

from grafo_random_layout_generar_ic import contar_aristas_malla_real
from grafo_random_layout_generar_ic_masa_fija import generar_control_random_masa_fija, LADO_FIJO, \
    MASA_TOTAL_OBJETIVO
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import POLYK

RUTA_DENS_BAR_N2000 = "/Users/alexis/phantom_cs073/bateria_n2000/dens_bar.npy"
BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_grafo_random_masa_fija"

VALORES_N = (200, 500, 1000, 2000)
SEMILLAS = (1, 2)          # 2 semillas por N -- corridas de Phantom son baratas (4-8s c/u, ver logs de
                             # piloto_grafo_random/bateria_grafo_random_n2000), alcanza el presupuesto
SEED_SUBMUESTREO = 12345    # rng fija para el submuestreo determinista de dens_bar por N

CS_SONORA = POLYK ** 0.5
TURB_SEED = 42


def submuestrear_dens_bar(dens_bar_n2000, n, seed=SEED_SUBMUESTREO):
    """Submuestreo determinista SIN reemplazo de las densidades del pool real N=2000 -- ver docstring
    del módulo para por qué esto es válido acá (dens_bar ya no determina masa ni caja, sólo alimenta el
    campo de velocidad turbulento)."""
    if n >= len(dens_bar_n2000):
        return dens_bar_n2000.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dens_bar_n2000), size=n, replace=False)
    return dens_bar_n2000[idx]


def main():
    t0 = time.time()
    dens_bar_n2000 = np.load(RUTA_DENS_BAR_N2000)
    print(f"[pool base] N2000 dens_bar cargado, n={len(dens_bar_n2000)}", flush=True)

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)
    os.makedirs(BASE_SALIDA, exist_ok=True)

    resumen = []
    for n in VALORES_N:
        dens_bar = submuestrear_dens_bar(dens_bar_n2000, n)
        n_malla, m_malla, _adj_real = contar_aristas_malla_real(dens_bar)
        print(f"\n[N={n}] malla REAL (para dimensionar el grafo random): n={n_malla} "
              f"n_aristas={m_malla} grado_medio={2*m_malla/n_malla:.3f}", flush=True)

        for seed in SEMILLAS:
            t1 = time.time()
            carpeta = f"{BASE_SALIDA}/ic_masaFija_N{n}_s{seed}"
            os.makedirs(carpeta, exist_ok=True)
            info = generar_control_random_masa_fija(
                n, dens_bar, n_aristas=m_malla, seed_random=1000 * n + seed,
                vel_generador=vel_gen, ruta_salida=f"{carpeta}/cosmogenesis_ic.txt")
            print(f"  [N={n} s{seed}] seed_random={info['seed_random']} lado={info['lado']:.4f} "
                  f"masa_particula={info['masa_particula']:.6g} "
                  f"masa_total={info['masa_total']:.6g} (objetivo={MASA_TOTAL_OBJETIVO}) "
                  f"tiempo={time.time()-t1:.1f}s -> {carpeta}/cosmogenesis_ic.txt", flush=True)
            resumen.append(dict(n=n, seed=seed, masa_total=info["masa_total"], lado=info["lado"]))

    print(f"\n[TOTAL generacion IC] {time.time()-t0:.1f}s", flush=True)
    print(f"[chequeo rápido] lado usado en TODOS los N: {LADO_FIJO:.6f} (constante, no depende de n)")
    for r in resumen:
        print(f"  N={r['n']} seed={r['seed']} masa_total={r['masa_total']:.6g}")
    return resumen


if __name__ == "__main__":
    main()
