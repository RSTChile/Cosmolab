"""
null2_zeldovich_piloto_generar.py — orquestador del PILOTO chico en Phantom del método Zel'dovich
(paso 4 del encargo "mejorar el método antes de escalar").

Se justificó correr este piloto porque la verificación de dos puntos ANTES de Phantom
(`null2_zeldovich_disenar_verificar.py`) mostró una mejora sustancial y robusta sobre el método de
rechazo de `null2_generar_ic.py`: KS bajó de 0.495 (rechazo, N=2000, ngrid=20, seed=9001) a 0.22
(Zel'dovich, MISMOS N/ngrid/seed), y se mantiene en el rango 0.19-0.39 al barrer ngrid=16..40 y
5 semillas de fase distintas -- mejora consistente, no un accidente de una sola semilla (ver
NULL2_mejora_zeldovich_CS.md para el detalle completo del barrido).

Igual que `null2_piloto_generar.py`: NO vuelve a extraer el pool de bariones ni corre REAL de nuevo
-- reutiliza la condición REAL de N=500 ya en disco en
/Users/alexis/phantom_cs073/piloto_null1/real/cosmogenesis_ic.txt (4 sumideros, masa 282.0). Sobre
esas MISMAS posiciones REAL de N=500 se aplica el método Zel'dovich (grilla -> aleatorizar fases ->
campo de desplazamiento -> grilla no perturbada -> desplazar) para producir 3 condiciones iniciales
(semillas 301, 302, 303 -- distintas de las 201-203 del método de rechazo, mismo patrón de
nomenclatura), con el MISMO campo de velocidad turbulento (Mach=3, semilla=42) que REAL/NULL-1/
NULL-2-rechazo, interpolado en las posiciones nuevas -- ningún otro parámetro físico cambia.

Grilla: ngrid=14 para N=500, MISMA que usó null2_piloto_generar.py (ocupación media ~500/14^3=0.18
part/celda) -- para que la única diferencia entre este piloto y el de NULL-2-rechazo sea el método
de conversión campo->partícula, no la resolución de grilla.

Escribe en /Users/alexis/phantom_cs073/piloto_null2_zeldovich/ (carpeta NUEVA, no toca
piloto_null2/ ni piloto_null1/ -- sólo LEE piloto_null1/real/). No corre Phantom (ver
null2_zeldovich_piloto_correr.py).
"""
import time

import numpy as np

from null1_generar_ic import leer_ic_txt
from null2_generar_ic import escribir_ic_txt, verificar_dos_puntos_particulas
from null2_zeldovich_generar_ic import generar_null2_zeldovich
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import HFACT, POLYK

RUTA_REAL_N500 = "/Users/alexis/phantom_cs073/piloto_null1/real/cosmogenesis_ic.txt"
BASE_SALIDA = "/Users/alexis/phantom_cs073/piloto_null2_zeldovich"
NGRID = 14
SEMILLAS = (301, 302, 303)
TURB_SEED = 42  # MISMA semilla de turbulencia que REAL/NULL-1/NULL-2-rechazo
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

    for i, seed in enumerate(SEMILLAS, start=1):
        t1 = time.time()
        resultado = generar_null2_zeldovich(pos_real, n, NGRID, seed)
        pos_z = resultado["pos"]
        r_z = np.linalg.norm(pos_z - pos_z.mean(axis=0), axis=1)
        resumen_ks = verificar_dos_puntos_particulas(pos_real, pos_z)
        vel_z = vel_gen(pos_z, None, None)
        h_z = np.full(n, HFACT)

        carpeta = f"{BASE_SALIDA}/null2z_s{i}"
        import os
        os.makedirs(carpeta, exist_ok=True)
        ruta_salida = f"{carpeta}/cosmogenesis_ic.txt"
        escribir_ic_txt(
            ruta_salida, pos_z, vel_z, h_z, masa_particula, HFACT, POLYK,
            comentario=(f"cosmogenesis_ic v2 NULL-2-Zeldovich (desplazamiento Lagrangiano, "
                        f"ngrid={NGRID}) -- npart={n} masa_particula={masa_particula:.17g} "
                        f"hfact={HFACT} polyk={POLYK:.17g} seed={seed}"),
        )
        print(f"[null2z_s{i}] seed={seed} residuo_imag={resultado['residuo_imag']:.2e} "
              f"desplazamiento_rms={resultado['desplazamiento_rms']:.2f} "
              f"r_mean={r_z.mean():.2f} r_std={r_z.std():.2f} "
              f"(REAL: {r_real.mean():.2f}/{r_real.std():.2f}) "
              f"KS={resumen_ks['ks_stat']:.4f} p={resumen_ks['ks_p']:.2e} "
              f"tiempo={time.time()-t1:.2f}s -> {ruta_salida}", flush=True)

    print(f"\n[TOTAL] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
