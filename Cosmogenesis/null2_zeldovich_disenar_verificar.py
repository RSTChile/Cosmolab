"""
null2_zeldovich_disenar_verificar.py — verificación ANTES de gastar cómputo en Phantom del método
Zel'dovich (`null2_zeldovich_generar_ic.py`), en el MISMO formato y sobre los MISMOS datos REAL
N=2000 que usó `null2_disenar_verificar.py` para el método de rechazo, para poder comparar KS
antes/después en igualdad de condiciones (misma fuente REAL, mismo ngrid de diseño, misma semilla
de diseño).

Qué hace:
  1. Lee REAL N=2000 (bateria_n2000/ic_real/cosmog_00000, sólo lectura -- mismo dump que usó el
     agente anterior).
  2. Genera el catálogo NULL-2-Zel'dovich con `generar_null2_zeldovich` (grilla -> fase aleatoria
     -> campo de desplazamiento -> grilla no perturbada -> desplazar).
  3. Corre la MISMA verificación de dos puntos por partícula (`verificar_dos_puntos_particulas`,
     reusada tal cual de null2_generar_ic.py -- ni el test ni el criterio cambian, sólo el método
     de generación) y reporta KS + r_mean/r_std, para comparar directo contra KS=0.495 del método
     de rechazo.

No escribe nada bajo bateria_n2000/. No toca null2_generar_ic.py ni null2_disenar_verificar.py.
"""
import time

import numpy as np

from leer_volcado_phantom import leer_dump
from null2_generar_ic import verificar_dos_puntos_particulas
from null2_zeldovich_generar_ic import generar_null2_zeldovich

RUTA_REAL_N2000 = "/Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmog_00000"
NGRID_DISENO = 20   # MISMO ngrid de diseño que usó null2_disenar_verificar.py, para comparar parejo
SEED_DISENO = 9001  # MISMA semilla de diseño que usó null2_disenar_verificar.py


def main():
    t0 = time.time()
    print(f"[1] leyendo {RUTA_REAL_N2000} (sólo lectura, N=2000 original)...", flush=True)
    gas, sinks = leer_dump(RUTA_REAL_N2000)
    pos_real = gas[["x", "y", "z"]].to_numpy()
    n = len(pos_real)
    com_real = pos_real.mean(axis=0)
    r_real = np.linalg.norm(pos_real - com_real, axis=1)
    print(f"    n={n} r_mean={r_real.mean():.2f} r_std={r_real.std():.2f} "
          f"t={time.time()-t0:.1f}s", flush=True)

    print(f"[2] generando NULL-2-Zel'dovich (ngrid={NGRID_DISENO}, seed={SEED_DISENO})...",
          flush=True)
    t1 = time.time()
    resultado = generar_null2_zeldovich(pos_real, n, NGRID_DISENO, SEED_DISENO)
    pos_z = resultado["pos"]
    com_z = pos_z.mean(axis=0)
    r_z = np.linalg.norm(pos_z - com_z, axis=1)
    print(f"    residuo_imag={resultado['residuo_imag']:.3e} "
          f"desplazamiento_rms={resultado['desplazamiento_rms']:.3f} "
          f"(cell_size={resultado['cell_size']:.3f}) "
          f"r_mean={r_z.mean():.2f} r_std={r_z.std():.2f} "
          f"(REAL: {r_real.mean():.2f}/{r_real.std():.2f}) tiempo={time.time()-t1:.2f}s",
          flush=True)

    print(f"[3] verificación de dos puntos por partícula (ξ vía KS, MISMO test que el método de "
          f"rechazo)...", flush=True)
    t2 = time.time()
    resumen = verificar_dos_puntos_particulas(pos_real, pos_z)
    print(f"    KS(d_real, d_zeldovich): stat={resumen['ks_stat']:.4f} p={resumen['ks_p']:.3e} "
          f"d_mean real={resumen['d_real_mean']:.3f} zeldovich={resumen['d_null2_mean']:.3f} "
          f"d_std real={resumen['d_real_std']:.3f} zeldovich={resumen['d_null2_std']:.3f} "
          f"tiempo={time.time()-t2:.1f}s", flush=True)
    print(f"\n    [REFERENCIA método anterior, mismo N/ngrid/seed de diseño, de "
          f"NULL2_piloto_espectro_potencia_CS.md]: KS=0.495 p≈0 "
          f"d_mean real=97.07 rechazo=66.08 d_std real=36.22 rechazo=25.62", flush=True)

    print(f"\n[TOTAL] {time.time()-t0:.1f}s", flush=True)
    return dict(n=n, r_real=(float(r_real.mean()), float(r_real.std())),
                r_zeldovich=(float(r_z.mean()), float(r_z.std())),
                desplazamiento_rms=resultado["desplazamiento_rms"],
                verificacion_particulas=resumen)


if __name__ == "__main__":
    main()
