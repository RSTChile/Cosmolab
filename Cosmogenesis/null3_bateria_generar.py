"""
null3_bateria_generar.py — NULL-3 (double-edge-swap con filtro geométrico de longitud) escalado a la
batería completa (N=2000, 8 semillas), Fase II CS073, escalón 3 de 6.

Por qué se escala directo (sin otro piloto intermedio): Paso 1 (`null3_paso1_verificar_perfil_radial.py`,
N=2000, seed=501) dio KS(r_real,r_null3)=0.0295, p=0.349 -- perfil radial prácticamente indistinguible
de REAL, muy lejos del KS<1e-113 de los NULL1-8 originales. El piloto (`null3_piloto_generar.py` +
`null3_piloto_correr.py`, N=500, semillas 601-603) corrió LIMPIO: 3/3 exit_setup=0, 3/3 exit_run=0, sin
abortos de conservación, sumideros formados en las 3 corridas (5, 3, 4 -- comparable a los 4 de REAL a
esa escala), masas 347.8/235.0/272.6 vs REAL=282.0. Alexis autorizó escalar directo si el piloto salía
limpio -- este es ese caso.

Qué hace, mismo patrón que `null2_bateria_generar.py` (no reinventado): lee directo el pool N=2000 YA
EXTRAÍDO (`bateria_n2000/dens_bar.npy`/`masa_bar.npy`, sólo lectura, no se re-extrae) y las posiciones
REALES YA ESCRITAS (`bateria_n2000/ic_real/cosmogenesis_ic.txt`, sólo lectura -- usadas como
`pos_referencia` del filtro de longitud, MISMA convención que Paso 1/piloto). Para cada semilla en
501-508: reconstruye la malla causal REAL (`malla_causal_atomos`, D=3/k=4/seed_ejes=2000, IDÉNTICA en
las 8 -- nunca cambia), aplica `barajar_aristas_preservando_longitud` (tol_relativa=0.2, la semilla
distingue QUÉ intercambios se aceptan), corre `layout_resortes` (Fruchterman-Reingold, MISMA función y
parámetros que usa REAL) con `seed_layout=12345` (el default, IDÉNTICO al de REAL/NULL-1/NULL-2 -- la
única diferencia entre las 8 corridas NULL-3 es `seed_null3`, igual que `seed_null` es la única
diferencia REAL/NULL en `traducir_pool`), aplica la MISMA dilatación estática y el MISMO campo de
velocidad turbulento (Mach=3, TURB_SEED=42) que REAL/NULL-1/NULL-2.

Escribe en `/Users/alexis/phantom_cs073/bateria_null3_n2000/ic_null3_s{501..508}/cosmogenesis_ic.txt`
(carpeta NUEVA). No toca `bateria_n2000/`, `bateria_null1_n2000/`, `bateria_null2_n2000/`,
`bateria_real_extra_n2000/`, `piloto_null3/`, ni ningún script congelado -- sólo los importa/lee.

No corre Phantom (`null3_bateria_correr.py`, paso aparte).
"""
import os
import time

import numpy as np

from null1_generar_ic import leer_ic_txt
from null3_generar_ic import generar_null3
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import POLYK

RUTA_DENS_BAR = "/Users/alexis/phantom_cs073/bateria_n2000/dens_bar.npy"
RUTA_MASA_BAR = "/Users/alexis/phantom_cs073/bateria_n2000/masa_bar.npy"
RUTA_IC_REAL = "/Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmogenesis_ic.txt"
BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_null3_n2000"
SEMILLAS = list(range(501, 509))   # 8 semillas, patrón pedido por Alexis
TOL_RELATIVA = 0.2                 # mismo valor que Paso 1 y el piloto -- consistencia entre escalas

CS_SONORA = POLYK ** 0.5
TURB_SEED = 42


def main():
    t0 = time.time()
    print(f"[pool] leyendo pool N=2000 ya en disco (sólo lectura)...", flush=True)
    dens_bar = np.load(RUTA_DENS_BAR)
    masa_bar = np.load(RUTA_MASA_BAR)
    print(f"[pool] n={len(dens_bar)} tiempo={time.time()-t0:.2f}s", flush=True)

    print(f"[real] leyendo posiciones REALES ya existentes de {RUTA_IC_REAL} (sólo lectura, referencia "
          f"de longitud y de comparación radial)...", flush=True)
    pos_real, vel_real, h_real, masa_particula, n = leer_ic_txt(RUTA_IC_REAL)
    com_real = pos_real.mean(axis=0)
    r_real = np.linalg.norm(pos_real - com_real, axis=1)
    print(f"[real] n={n} masa_particula={masa_particula:.6g} r_mean={r_real.mean():.3f} "
          f"r_std={r_real.std():.3f}", flush=True)

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)
    os.makedirs(BASE_SALIDA, exist_ok=True)

    for seed in SEMILLAS:
        t1 = time.time()
        carpeta = f"{BASE_SALIDA}/ic_null3_s{seed}"
        os.makedirs(carpeta, exist_ok=True)
        info = generar_null3(masa_bar, dens_bar, pos_real, seed_null3=seed,
                              tol_relativa=TOL_RELATIVA, vel_generador=vel_gen,
                              ruta_salida=f"{carpeta}/cosmogenesis_ic.txt")
        pos_n3 = info["pos"]
        r_n3 = np.linalg.norm(pos_n3 - pos_n3.mean(axis=0), axis=1)
        print(f"[ic_null3_s{seed}] swap={info['swap_aceptados']}/{info['swap_intentos']} "
              f"({100*info['swap_aceptados']/info['swap_intentos']:.1f}%) "
              f"r_mean={r_n3.mean():.3f} r_std={r_n3.std():.3f} "
              f"(REAL: {r_real.mean():.3f}/{r_real.std():.3f}) "
              f"tiempo={time.time()-t1:.1f}s -> {carpeta}/cosmogenesis_ic.txt", flush=True)

    print(f"\n[TOTAL generacion IC] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
