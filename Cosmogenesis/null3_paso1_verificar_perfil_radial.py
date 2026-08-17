"""
null3_paso1_verificar_perfil_radial.py — Paso 1 del encargo NULL-3: correr `layout_resortes` sobre el
grafo NULL-3 (double-edge-swap + filtro geométrico de longitud, ya verificado a nivel de grafo en
`null3_investigacion_preliminar.py`, KS(L,L_real)=0.004) y verificar el perfil RADIAL resultante contra
REAL -- el mismo tipo de chequeo (KS de la distancia al centro de masa) que hizo el piloto de NULL-1
antes de escalar, y que ya mostró que el double-edge-swap SIN restringir (los NULL1-8 originales de
CS073) rompe el perfil radial completo (KS<1e-113 en las 8 comparaciones, ver docstring de
`null1_generar_ic.py`).

Corre sobre el pool N=2000 ya en disco (`bateria_n2000/dens_bar.npy`/`masa_bar.npy`, sólo lectura --
NO se re-extrae el pool ni se re-corre `traducir_pool`), usando como `pos_referencia` para el filtro de
longitud las posiciones REAL YA ESCRITAS en `bateria_n2000/ic_real/cosmogenesis_ic.txt` -- exactamente
la misma convención que usó `null3_investigacion_preliminar.py`.

No corre Phantom. No escribe nada bajo `bateria_n2000/`. No toca ningún script congelado.
"""
import time

import numpy as np
from scipy.stats import ks_2samp

from null1_generar_ic import leer_ic_txt
from null3_generar_ic import generar_null3

RUTA_DENS_BAR = "/Users/alexis/phantom_cs073/bateria_n2000/dens_bar.npy"
RUTA_MASA_BAR = "/Users/alexis/phantom_cs073/bateria_n2000/masa_bar.npy"
RUTA_IC_REAL = "/Users/alexis/phantom_cs073/bateria_n2000/ic_real/cosmogenesis_ic.txt"
RUTA_SALIDA_TMP = "/tmp/_null3_paso1_verificacion_ic.txt"

SEED_NULL3 = 501       # primera semilla de la futura batería NULL-3 (501-508) -- reproducible, no ad hoc
TOL_RELATIVA = 0.2     # mismo valor que usó null3_investigacion_preliminar.py (punto de partida razonable)


def main():
    t0 = time.time()
    print("[1] leyendo pool N=2000 ya en disco (dens_bar.npy, masa_bar.npy) -- sólo lectura...",
          flush=True)
    dens_bar = np.load(RUTA_DENS_BAR)
    masa_bar = np.load(RUTA_MASA_BAR)
    print(f"    n={len(dens_bar)} tiempo={time.time()-t0:.2f}s", flush=True)

    print(f"\n[2] leyendo REAL ya en disco ({RUTA_IC_REAL}) -- referencia de longitud Y de perfil radial",
          flush=True)
    t1 = time.time()
    pos_real, vel_real, h_real, masa_particula_real, n = leer_ic_txt(RUTA_IC_REAL)
    com_real = pos_real.mean(axis=0)
    r_real = np.linalg.norm(pos_real - com_real, axis=1)
    print(f"    n={n} r_mean={r_real.mean():.3f} r_std={r_real.std():.3f} tiempo={time.time()-t1:.2f}s",
          flush=True)

    print(f"\n[3] generando NULL-3 (seed={SEED_NULL3}, tol_relativa={TOL_RELATIVA}): malla causal REAL "
          f"-> swap con filtro de longitud -> layout_resortes (Fruchterman-Reingold, MISMA función que "
          f"usa REAL) -> dilatación estática...", flush=True)
    t2 = time.time()
    info = generar_null3(masa_bar, dens_bar, pos_real, seed_null3=SEED_NULL3,
                          tol_relativa=TOL_RELATIVA, ruta_salida=RUTA_SALIDA_TMP)
    pos_null3 = info["pos"]
    com_null3 = pos_null3.mean(axis=0)
    r_null3 = np.linalg.norm(pos_null3 - com_null3, axis=1)
    print(f"    swap aceptados/intentados={info['swap_aceptados']}/{info['swap_intentos']} "
          f"({100*info['swap_aceptados']/info['swap_intentos']:.1f}%)  a_final={info['a_final']:.4f}  "
          f"tiempo={time.time()-t2:.1f}s", flush=True)

    print(f"\n[4] perfil radial NULL-3 vs REAL (distancia al centro de masa, post-layout+expansión):",
          flush=True)
    print(f"    REAL   r_mean={r_real.mean():8.3f}  r_std={r_real.std():8.3f}")
    print(f"    NULL-3 r_mean={r_null3.mean():8.3f}  r_std={r_null3.std():8.3f}  "
          f"(diff mean = {100*(r_null3.mean()-r_real.mean())/r_real.mean():+.1f}%)")

    ks = ks_2samp(r_real, r_null3)
    print(f"\n    KS(r_real, r_null3) = {ks.statistic:.4f}  p={ks.pvalue:.3e}")

    print(f"\n[REFERENCIA -- lo que se busca comparar contra]:")
    print(f"    NULL1-8 ORIGINALES (swap sin restricción de longitud, re-layout completo): "
          f"KS<1e-113 en las 8 comparaciones (docstring null1_generar_ic.py) -- perfil radial ROTO.")
    print(f"    NULL-3-grafo puro (sin layout, sólo distribución de longitudes de arista, "
          f"null3_investigacion_preliminar.py): KS(L,L_real)=0.0040 -- prácticamente indistinguible.")
    print(f"    Esta verificación (arriba) es la primera vez que se mide el perfil radial de las "
          f"POSICIONES post-layout_resortes -- no se puede asumir que el KS de longitudes de arista se")
    print(f"    traduzca 1:1 al KS del perfil radial (FR es una relajación no lineal de 100 "
          f"iteraciones); se reporta el número real, no una extrapolación.")

    print(f"\n[TOTAL] {time.time()-t0:.1f}s", flush=True)
    return dict(n=n, r_real=(float(r_real.mean()), float(r_real.std())),
                r_null3=(float(r_null3.mean()), float(r_null3.std())),
                ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                swap_aceptados=info["swap_aceptados"], swap_intentos=info["swap_intentos"])


if __name__ == "__main__":
    main()
