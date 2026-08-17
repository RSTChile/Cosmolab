"""
null2_bateria_generar.py — NULL-2 (método Zel'dovich) escalado a la batería completa (N=2000,
8 semillas), Fase II CS073, escalón 2 de 6.

Qué problema resuelve: `NULL2_mejora_zeldovich_CS.md` validó a N=2000 (verificación de dos puntos
sin Phantom, seed de diseño 9001, ngrid=20) que el desplazamiento de Zel'dovich baja el KS de 0.495
(método de rechazo/inversión) a 0.220 frente a REAL -- mejora robusta (0.19-0.39 en 10 combinaciones
ngrid/semilla) -- y corrió un piloto en Phantom a N=500/3 semillas que terminó limpio pero sin formar
sumideros en ninguna de las 3. Alexis autorizó escalar a la batería completa del mismo diseño que usó
NULL-1: N=2000, 8 semillas, mismo REAL de referencia (`bateria_n2000/ic_real`), esta vez con semillas
NUEVAS (401-408, para no confundir con las 1-3 del piloto Zel'dovich en `piloto_null2_zeldovich/`).

Qué hace, mismo patrón que `null1_bateria_generar.py` (no reinventado): lee directo las posiciones
REALES YA ESCRITAS en `/Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmogenesis_ic.txt` (sólo
lectura, `leer_ic_txt` de `null1_generar_ic.py`, congelado) -- no vuelve a correr `traducir_pool`.
Sobre esas 2000 posiciones REAL aplica el método Zel'dovich completo
(`generar_null2_zeldovich`, de `null2_zeldovich_generar_ic.py`, congelado, no se toca): gridizar REAL
-> aleatorizar fases (semilla de fase = la semilla de la corrida) -> campo de desplazamiento de
Zel'dovich -> grilla no perturbada (punto de partida homogéneo, `seed_q` interno = seed+100000) ->
desplazar por interpolación trilineal. ngrid=20: la misma resolución de grilla que usó la comparación
"antes/después" ya reportada como número de cabecera en NULL2_mejora_zeldovich_CS.md (KS=0.220 vs
REAL, seed de diseño 9001) -- se mantiene la MISMA resolución para que esta batería sea directamente
comparable a ese número ya validado, no una resolución nueva sin verificar.

Mismo campo de velocidad turbulento que REAL/NULL-1 (Mach=3, TURB_SEED=42) interpolado en las
posiciones finales desplazadas -- ningún otro parámetro físico cambia entre REAL/NULL-1/NULL-2.

Escribe en `/Users/alexis/phantom_cs073/bateria_null2_n2000/ic_null2_s{401..408}/cosmogenesis_ic.txt`
(carpeta NUEVA). No toca `bateria_n2000/`, `bateria_null1_n2000/`, `bateria_real_extra_n2000/`,
`piloto_null2_zeldovich/`, ni ningún script congelado (`null1_generar_ic.py`, `null2_generar_ic.py`,
`null2_zeldovich_generar_ic.py`, `campo_velocidad_turbulento.py`, `fase1_traducir_a_phantom.py`)
-- sólo los importa/lee.

No corre Phantom -- eso es un paso aparte (`null2_bateria_correr.py`).
"""
import os
import time

import numpy as np

from null1_generar_ic import leer_ic_txt
from null2_generar_ic import escribir_ic_txt, verificar_dos_puntos_particulas
from null2_zeldovich_generar_ic import generar_null2_zeldovich
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import HFACT, POLYK

RUTA_IC_REAL = "/Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmogenesis_ic.txt"
BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_null2_n2000"
SEMILLAS = list(range(401, 409))  # 8 semillas, distintas de 301-303 (piloto Zel'dovich N=500)
NGRID = 20  # misma resolución que el número de cabecera KS=0.220 de NULL2_mejora_zeldovich_CS.md

CS_SONORA = POLYK ** 0.5   # misma convención que null1_bateria_generar.py / piloto Zel'dovich
TURB_SEED = 42             # MISMA semilla de turbulencia que REAL/NULL-1


def main():
    t0 = time.time()
    print(f"[real] leyendo posiciones REALES ya existentes de {RUTA_IC_REAL} (sólo lectura)...",
          flush=True)
    pos_real, vel_real, h_real, masa_particula, n = leer_ic_txt(RUTA_IC_REAL)
    com_real = pos_real.mean(axis=0)
    r_real = np.linalg.norm(pos_real - com_real, axis=1)
    print(f"[real] n={n} masa_particula={masa_particula:.6g} r_mean={r_real.mean():.3f} "
          f"r_std={r_real.std():.3f} com={com_real}", flush=True)

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)

    os.makedirs(BASE_SALIDA, exist_ok=True)

    for i, seed in enumerate(SEMILLAS, start=1):
        t1 = time.time()
        resultado = generar_null2_zeldovich(pos_real, n, NGRID, seed)
        pos_z = resultado["pos"]
        r_z = np.linalg.norm(pos_z - pos_z.mean(axis=0), axis=1)
        resumen_ks = verificar_dos_puntos_particulas(pos_real, pos_z)
        vel_z = vel_gen(pos_z, None, None)
        h_z = np.full(n, HFACT)

        carpeta = f"{BASE_SALIDA}/ic_null2_s{seed}"
        os.makedirs(carpeta, exist_ok=True)
        ruta_salida = f"{carpeta}/cosmogenesis_ic.txt"
        escribir_ic_txt(
            ruta_salida, pos_z, vel_z, h_z, masa_particula, HFACT, POLYK,
            comentario=(f"cosmogenesis_ic v2 NULL-2-Zeldovich (desplazamiento Lagrangiano, "
                        f"ngrid={NGRID}) -- npart={n} masa_particula={masa_particula:.17g} "
                        f"hfact={HFACT} polyk={POLYK:.17g} seed={seed}"),
        )
        print(f"[null2_s{seed}] residuo_imag={resultado['residuo_imag']:.2e} "
              f"desplazamiento_rms={resultado['desplazamiento_rms']:.2f} "
              f"r_mean={r_z.mean():.2f} r_std={r_z.std():.2f} "
              f"(REAL: {r_real.mean():.2f}/{r_real.std():.2f}) "
              f"KS={resumen_ks['ks_stat']:.4f} p={resumen_ks['ks_p']:.2e} "
              f"tiempo={time.time()-t1:.2f}s -> {ruta_salida}", flush=True)

    print(f"\n[TOTAL generacion IC] {time.time()-t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
