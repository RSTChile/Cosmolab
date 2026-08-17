"""
null3_piloto_generar.py — orquestador del PILOTO chico de NULL-3 (Paso 2 del encargo), N=500, 3
semillas -- mismo patrón que `null1_piloto_generar.py`/`null2_piloto_generar.py`/
`null2_zeldovich_piloto_generar.py`.

Justificación para pasar al piloto: Paso 1 (`null3_paso1_verificar_perfil_radial.py`, N=2000) mostró
que el double-edge-swap con filtro geométrico de longitud (tol_relativa=0.2), después de
`layout_resortes`, produce un perfil radial (distancia al centro de masa) prácticamente indistinguible
de REAL -- KS=0.0295, p=0.349, diff de r_mean=+0.7% -- muy lejos del KS<1e-113 que rompía a los NULL1-8
originales (swap SIN restricción de longitud). Confirma la hipótesis de trabajo: la escala local de
conexión (longitud de arista) sí determina la escala global de la relajación de resortes.

Qué hace: extrae el pool de bariones a N=500 con los MISMOS parámetros que usó
`null1_piloto_generar.py` (nq=3000, naq=2100, ne=1000, npos=700 -- determinista, el motor basal no
depende de ninguna semilla de NULL, así que reproduce exactamente el mismo `masa_bar`/`dens_bar` que ya
generó `/Users/alexis/phantom_cs073/piloto_null1/real/cosmogenesis_ic.txt`), reutiliza ESE archivo REAL
ya en disco (sólo lectura) como `pos_referencia` para el filtro de longitud del swap (misma convención
que Paso 1 a N=2000: medir longitud sobre las posiciones REAL YA ESCRITAS), y genera 3 condiciones
NULL-3 (semillas 601, 602, 603 -- nuevas, no chocan con 101-103 de NULL-1 ni 201-203/301-303 de NULL-2)
con el MISMO campo de velocidad turbulento (Mach=3, seed=42) que REAL/NULL-1/NULL-2.

Escribe en /Users/alexis/phantom_cs073/piloto_null3/null3_s{1,2,3}/cosmogenesis_ic.txt (carpeta NUEVA).
No toca piloto_null1/ ni ningún script congelado -- sólo los importa/lee.

No corre Phantom (ver null3_piloto_correr.py).
"""
import os
import time

from cs073_cierre_holistico import _extraer_bariones
from null1_generar_ic import leer_ic_txt
from null3_generar_ic import generar_null3
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import POLYK

BASE = "/Users/alexis/phantom_cs073/piloto_null3"
RUTA_REAL_N500 = "/Users/alexis/phantom_cs073/piloto_null1/real/cosmogenesis_ic.txt"
CS_SONORA = POLYK ** 0.5
TURB_SEED = 42
SEMILLAS = (601, 602, 603)
TOL_RELATIVA = 0.2   # mismo valor que Paso 1 (N=2000), sin barrer -- consistencia entre escalas


def main():
    t0 = time.time()
    print("[pool] extrayendo bariones (nq=3000,naq=2100,ne=1000,npos=700) -- MISMOS parámetros que "
          "null1_piloto_generar.py, determinista, debe reproducir masa_bar/dens_bar idénticos a los "
          "que ya generaron piloto_null1/real/...", flush=True)
    masa_bar, dens_bar, obs = _extraer_bariones(3000, 2100, 1000, 700, 150, 1.5)
    print(f"[pool] n_atomos={len(masa_bar)} H={obs.get('hidrogeno')} He={obs.get('helio')} "
          f"tiempo={time.time()-t0:.1f}s", flush=True)

    print(f"\n[real] leyendo REAL N=500 ya en disco: {RUTA_REAL_N500}", flush=True)
    pos_real, vel_real, h_real, masa_particula, n = leer_ic_txt(RUTA_REAL_N500)
    assert n == len(masa_bar), (
        f"n de piloto_null1/real ({n}) != n del pool recién extraído ({len(masa_bar)}) -- el pool NO "
        "es determinista como se asumía, no seguir a ciegas.")
    import numpy as np
    com_real = pos_real.mean(axis=0)
    r_real = np.linalg.norm(pos_real - com_real, axis=1)
    print(f"    n={n} masa_particula={masa_particula:.6g} r_mean={r_real.mean():.3f} "
          f"r_std={r_real.std():.3f}", flush=True)

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)

    for i, seed in enumerate(SEMILLAS, start=1):
        t1 = time.time()
        os.makedirs(f"{BASE}/null3_s{i}", exist_ok=True)
        info = generar_null3(masa_bar, dens_bar, pos_real, seed_null3=seed,
                              tol_relativa=TOL_RELATIVA, vel_generador=vel_gen,
                              ruta_salida=f"{BASE}/null3_s{i}/cosmogenesis_ic.txt")
        pos_n3 = info["pos"]
        r_n3 = np.linalg.norm(pos_n3 - pos_n3.mean(axis=0), axis=1)
        print(f"[null3_s{i}] seed={seed} swap={info['swap_aceptados']}/{info['swap_intentos']} "
              f"({100*info['swap_aceptados']/info['swap_intentos']:.1f}%) "
              f"r_mean={r_n3.mean():.3f} r_std={r_n3.std():.3f} "
              f"(REAL: {r_real.mean():.3f}/{r_real.std():.3f}) "
              f"tiempo={time.time()-t1:.1f}s -> {BASE}/null3_s{i}/cosmogenesis_ic.txt", flush=True)

    print(f"\n[TOTAL] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
