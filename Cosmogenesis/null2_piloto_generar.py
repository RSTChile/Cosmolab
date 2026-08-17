"""
null2_piloto_generar.py — orquestador del PILOTO chico de NULL-2 (Paso 2 del encargo).

Qué hace: NO vuelve a extraer el pool de bariones ni a correr REAL de nuevo -- reutiliza la condición
REAL de N=500 que ya está en disco en /Users/alexis/phantom_cs073/piloto_null1/real/cosmogenesis_ic.txt
(generada y corrida para el piloto de NULL-1: 4 sumideros, masa 282.0 -- ver
NULL1_piloto_distribucion_radial_CS.md). Sobre esas MISMAS posiciones REAL de N=500 se aplica el
método NULL-2 (null2_generar_ic.gridizar + aleatorizar_fases + muestrear_particulas_de_campo) para
producir 3 condiciones iniciales NULL-2 (semillas 201, 202, 203 -- distintas de las 101-103 usadas por
NULL-1, mismo patrón de nomenclatura), con el MISMO campo de velocidad turbulento (Mach=3, semilla=42)
que usaron REAL y NULL-1, interpolado en las posiciones nuevas -- ningún otro parámetro físico cambia.

Grilla elegida (ngrid=14 para N=500): barrido de diagnóstico (ver null2_disenar_verificar.py, hecho
sobre N=2000) mostró que el resultado de la reconstrucción NO es muy sensible a ngrid en el rango
16-40 -- r_mean reconstruido se mantiene ~48-51 en todos los casos frente a un r_mean REAL de 72.78,
así que no hay un "punto óptimo" claro; se eligió ngrid=14 para N=500 manteniendo una ocupación media
similar (~500/14^3=0.18 part/celda) a la usada en el diagnóstico N=2000 (~0.25 part/celda).

No corre Phantom (ver null2_piloto_correr.py) -- sólo prepara las 3 condiciones iniciales NULL-2 en
/Users/alexis/phantom_cs073/piloto_null2/null2_s{1,2,3}/. No toca piloto_null1/ (sólo lo LEE).
"""
import time

import numpy as np

from null1_generar_ic import leer_ic_txt
from null2_generar_ic import gridizar, aleatorizar_fases, muestrear_particulas_de_campo, escribir_ic_txt
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import HFACT, POLYK

RUTA_REAL_N500 = "/Users/alexis/phantom_cs073/piloto_null1/real/cosmogenesis_ic.txt"
BASE_SALIDA = "/Users/alexis/phantom_cs073/piloto_null2"
NGRID = 14
SEMILLAS = (201, 202, 203)
TURB_SEED = 42  # MISMA semilla de turbulencia que REAL y NULL-1 (aísla la posición, no la velocidad)
CS_SONORA = POLYK ** 0.5


def main():
    t0 = time.time()
    print(f"[1] leyendo REAL N=500 ya en disco: {RUTA_REAL_N500}", flush=True)
    pos_real, vel_real, h_real, masa_particula, n = leer_ic_txt(RUTA_REAL_N500)
    print(f"    n={n} masa_particula={masa_particula:.6g}", flush=True)

    com_real = pos_real.mean(axis=0)
    r_real = np.linalg.norm(pos_real - com_real, axis=1)
    print(f"    REAL (referencia): r_mean={r_real.mean():.2f} r_std={r_real.std():.2f}", flush=True)

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)

    print(f"[2] gridizando en {NGRID}^3 celdas...", flush=True)
    campo, cell_size, origin, centro, half_extent = gridizar(pos_real, NGRID)
    print(f"    ocupación_media={campo.mean():.3f} part/celda cell_size={cell_size:.3f}", flush=True)

    for i, seed in enumerate(SEMILLAS, start=1):
        t1 = time.time()
        campo_sint, residuo_imag = aleatorizar_fases(campo, seed)
        frac_neg = -campo_sint[campo_sint < 0].sum() / campo_sint[campo_sint > 0].sum()
        pos_n2 = muestrear_particulas_de_campo(campo_sint, n, cell_size, origin, seed)
        r_n2 = np.linalg.norm(pos_n2 - pos_n2.mean(axis=0), axis=1)
        vel_n2 = vel_gen(pos_n2, None, None)
        h_n2 = np.full(n, HFACT)

        carpeta = f"{BASE_SALIDA}/null2_s{i}"
        ruta_salida = f"{carpeta}/cosmogenesis_ic.txt"
        escribir_ic_txt(
            ruta_salida, pos_n2, vel_n2, h_n2, masa_particula, HFACT, POLYK,
            comentario=(f"cosmogenesis_ic v2 NULL-2 (aleatorizacion de fases, ngrid={NGRID}) -- "
                        f"npart={n} masa_particula={masa_particula:.17g} hfact={HFACT} "
                        f"polyk={POLYK:.17g} seed={seed}"),
        )
        print(f"[null2_s{i}] seed={seed} residuo_imag={residuo_imag:.2e} "
              f"frac_masa_negativa_clip={frac_neg:.3f} r_mean={r_n2.mean():.2f} "
              f"r_std={r_n2.std():.2f} (REAL: {r_real.mean():.2f}/{r_real.std():.2f}) "
              f"tiempo={time.time()-t1:.2f}s -> {ruta_salida}", flush=True)

    print(f"\n[TOTAL] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
