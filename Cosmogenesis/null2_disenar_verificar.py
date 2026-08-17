"""
null2_disenar_verificar.py — Paso 1 del encargo NULL-2: diseño + verificación ANTES de gastar cómputo
en Phantom.

Qué hace:
  1. Lee las posiciones de gas de la corrida REAL original de la batería N=2000
     (/Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmog_00000 -- el primer volcado, t=0, la
     condición inicial misma antes de que la gravedad de Phantom actúe) con leer_volcado_phantom.py
     (sólo lectura -- no se toca ni un byte de bateria_n2000/).
  2. Calcula P(k) del campo gridizado de REAL (diagnóstico de diseño) y documenta por qué la
     verificación FINAL de "¿NULL-2 preserva la estadística de dos puntos?" se hace con ξ(r) por pares
     de partícula (KS de la distribución de distancias) y no con P(k) de partícula discreta.
  3. Genera un campo NULL-2 (aleatorización de fases) a partir del campo gridizado de REAL, y verifica
     dos cosas:
       (a) por construcción, el ESPECTRO DE POTENCIA DE LA GRILLA debe coincidir exactamente
           (mismo |F(k)| modo a modo) -- esto es una verificación de que la implementación de
           aleatorizar_fases() no tiene bugs, no una verificación estadística interesante en sí misma.
       (b) la verificación que SÍ importa: ¿el catálogo de partículas muestreado del campo NULL-2
           preserva la distribución de distancias par-a-par de las partículas REALES? (test KS).

Por qué ξ(r)/distancias-por-pares y no P(k) de partícula discreta para la verificación final: con
N=2000 partículas, estimar P(k) directamente de un catálogo de puntos discreto exige volver a grillar
esas partículas en una malla FFT -- lo que reintroduce exactamente el mismo ruido de Poisson (shot
noise) que ya limita la resolución del campo NULL-2 en sí (~0.25 partículas/celda con ngrid=20,
ver docstring de null2_generar_ic.py). ξ(r) vía conteo de pares (pdist) NO requiere grillar de nuevo:
usa las posiciones EXACTAS de cada partícula, así que el único ruido que queda es el de tamaño de
muestra (N=2000 pares -> ~2 millones de distancias), mucho menor que el de re-grillar. Por eso ξ(r)/
distancias-por-pares es la métrica de MENOR riesgo de artefacto de muestreo a este N, y es la elegida
para la verificación (Paso 5 del encargo).

Salida: imprime un resumen; no escribe nada bajo bateria_n2000/.
"""
import time

from leer_volcado_phantom import leer_dump
from null2_generar_ic import (
    gridizar, pk_radial, aleatorizar_fases, muestrear_particulas_de_campo,
    verificar_dos_puntos_particulas,
)

RUTA_REAL_N2000 = "/Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmog_00000"
NGRID_DISENO = 20   # ~2000/20^3 = 0.25 particulas/celda de media -- ver limitaciones en el docstring
SEED_DISENO = 9001  # semilla de diseño, distinta de las semillas del piloto (201,202,203)


def main():
    t0 = time.time()
    print(f"[1] leyendo {RUTA_REAL_N2000} (sólo lectura, N=2000 original)...", flush=True)
    gas, sinks = leer_dump(RUTA_REAL_N2000)
    pos_real = gas[["x", "y", "z"]].to_numpy()
    n = len(pos_real)
    print(f"    n={n} sinks={'ninguno' if sinks is None else len(sinks)} "
          f"t={time.time()-t0:.1f}s", flush=True)

    print(f"[2] gridizando en {NGRID_DISENO}^3 celdas y calculando P(k) de diseño...", flush=True)
    campo, cell_size, origin, centro, half_extent = gridizar(pos_real, NGRID_DISENO)
    ocupacion_media = campo.mean()
    k_real, pk_real = pk_radial(campo, cell_size)
    print(f"    cell_size={cell_size:.3f} ocupación_media={ocupacion_media:.3f} part/celda "
          f"(shot-noise dominante si << 1, documentado)", flush=True)

    print(f"[3] aleatorizando fases (seed={SEED_DISENO})...", flush=True)
    campo_sint, residuo_imag = aleatorizar_fases(campo, SEED_DISENO)
    escala_campo = campo.max()
    print(f"    residuo imaginario máx={residuo_imag:.3e} (escala del campo={escala_campo:.3g}, "
          f"razón={residuo_imag/max(escala_campo,1e-30):.3e} -- debe ser ~1e-14, error de "
          f"punto flotante, no señal)", flush=True)

    print(f"[4] verificación (a) -- P(k) de la GRILLA debe coincidir exacto (por construcción)...",
          flush=True)
    k_sint, pk_sint = pk_radial(campo_sint, cell_size)
    import numpy as np
    validos = ~(np.isnan(pk_real) | np.isnan(pk_sint))
    diff_rel = np.abs(pk_sint[validos] - pk_real[validos]) / np.maximum(pk_real[validos], 1e-30)
    print(f"    diferencia relativa de P(k) grilla real vs sintético: "
          f"máx={diff_rel.max():.3e} mediana={np.median(diff_rel):.3e} "
          f"({validos.sum()}/{len(pk_real)} bins con datos) -- debe ser ~0 (coincide por "
          f"construcción, |F(k)| no se tocó)", flush=True)

    print(f"[5] verificación (b) -- muestreando N={n} partículas del campo NULL-2 y comparando "
          f"distribución de distancias par-a-par contra REAL (ξ vía KS, no P(k) de partícula)...",
          flush=True)
    pos_null2 = muestrear_particulas_de_campo(campo_sint, n, cell_size, origin, SEED_DISENO)
    t1 = time.time()
    resumen = verificar_dos_puntos_particulas(pos_real, pos_null2)
    print(f"    KS(d_real, d_null2): stat={resumen['ks_stat']:.4f} p={resumen['ks_p']:.3e} "
          f"(n_pares por lado ~{resumen['n_real']*(resumen['n_real']-1)//2}) "
          f"d_mean real={resumen['d_real_mean']:.3f} null2={resumen['d_null2_mean']:.3f} "
          f"d_std real={resumen['d_real_std']:.3f} null2={resumen['d_null2_std']:.3f} "
          f"tiempo={time.time()-t1:.1f}s", flush=True)

    print(f"\n[TOTAL] {time.time()-t0:.1f}s", flush=True)
    return dict(n=n, ocupacion_media=float(ocupacion_media), residuo_imag=residuo_imag,
                diff_rel_pk_max=float(diff_rel.max()), diff_rel_pk_mediana=float(np.median(diff_rel)),
                verificacion_particulas=resumen)


if __name__ == "__main__":
    main()
