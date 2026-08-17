"""
null1_piloto_generar.py — orquestador del PILOTO chico de NULL-1 (Paso 3, Fase II CS073).

Qué hace: extrae el pool de bariones UNA vez a escala reducida (N=500, el extremo superior del rango
250-500 pedido para el piloto -- no 2000, eso es la batería completa que vendrá después si Alexis lo
autoriza), genera 1 condición REAL (pieza congelada `traducir_pool`, sin tocarla) y 3 condiciones NULL-1
(semillas angulares 1, 2, 3 -- el "2-3 semillas" pedido para el piloto) con `null1_generar_ic.py`, y
escribe cada `cosmogenesis_ic.txt` directamente en su carpeta de corrida dentro de
/Users/alexis/phantom_cs073/piloto_null1/ (carpeta NUEVA, no toca bateria_n2000/).

Escala elegida (nq=3000, naq=2100, ne=1000, npos=700, dando N=500 átomos de H): se probó primero N=300
(nq=1800) -- ese tamaño SÍ corre y NO forma sumideros en tmax=0.5, y al extender tmax a 3.0 para darle
tiempo a colapsar, Phantom aborta con "Large error in linear momentum conservation" a t=0.885
(densidad~215, aún por debajo de rho_crit_cgs=1000) -- señal de que a N=300 el ruido de dos-cuerpos entre
partículas discretas (masa/partícula más alta que a N=2000) revienta la conservación antes de que el gas
llegue a formar un sumidero. N=500 (el techo del rango pedido) tiene partículas más livianas y llega más
lejos antes de ese límite numérico -- ver NULL1_piloto_distribucion_radial_CS.md para el detalle completo
de ambos intentos.

No corre Phantom (eso es un paso aparte, con el binario ya compilado en phantom_cs073/phantom/bin/) --
sólo prepara las 4 condiciones iniciales.
"""
import time

from cs073_cierre_holistico import _extraer_bariones
from null1_generar_ic import radios_desde_real, generar_null1
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from fase1_traducir_a_phantom import POLYK

BASE = "/Users/alexis/phantom_cs073/piloto_null1"
CS_SONORA = POLYK ** 0.5   # c_s = sqrt(polyk), misma convención que scratch_generar_ic_velocidades.py
TURB_SEED = 42             # MISMA semilla de turbulencia en REAL y en las 3 NULL-1 -- aísla la posición


def main():
    t0 = time.time()
    print(f"[pool] extrayendo bariones (nq=3000,naq=2100,ne=1000,npos=700)...", flush=True)
    masa_bar, dens_bar, obs = _extraer_bariones(3000, 2100, 1000, 700, 150, 1.5)
    print(f"[pool] n_atomos={len(masa_bar)} H={obs.get('hidrogeno')} He={obs.get('helio')} "
          f"tiempo={time.time()-t0:.1f}s", flush=True)

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)

    # REAL: pieza congelada, sin tocar -- se escribe directo en la carpeta de corrida del piloto.
    t1 = time.time()
    real = radios_desde_real(masa_bar, dens_bar, seed_layout=12345,
                              ruta_tmp=f"{BASE}/real/cosmogenesis_ic.txt", vel_generador=vel_gen)
    print(f"[real] n={real['n']} r_mean={real['r'].mean():.3f} r_std={real['r'].std():.3f} "
          f"tiempo={time.time()-t1:.1f}s -> {BASE}/real/cosmogenesis_ic.txt", flush=True)

    # NULL-1 x 3 semillas angulares -- mismo r (heredado exacto de REAL), mismo campo de velocidad
    # turbulento (misma semilla 42), sólo cambia la dirección angular aleatoria de cada partícula.
    for i, seed in enumerate((101, 102, 103), start=1):
        t2 = time.time()
        info = generar_null1(real["r"], real["com"], real["masa_particula"], real["n"], seed=seed,
                              vel_generador=vel_gen, ruta_salida=f"{BASE}/null1_s{i}/cosmogenesis_ic.txt")
        print(f"[null1_s{i}] seed={seed} r_mean={info['r_mean']:.3f} r_std={info['r_std']:.3f} "
              f"tiempo={time.time()-t2:.1f}s -> {BASE}/null1_s{i}/cosmogenesis_ic.txt", flush=True)

    print(f"[TOTAL] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
