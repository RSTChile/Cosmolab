"""
scratch_generar_ic_velocidades.py -- generador de los 8 IC (2 brazos x {N=2000,8550} x {REAL,NULL})
para INSTRUCCION CS 20-jul (dos brazos de campo de velocidades inicial). Script de corrida puntual, no
parte del arco congelado -- reusa exclusivamente piezas ya validadas (traducir_pool, ambos módulos de
campo de velocidad).
"""
import numpy as np

from fase1_traducir_a_phantom import traducir_pool, POLYK
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
from campo_velocidad_heredado import factory as factory_hered

CS_SONORA = float(np.sqrt(POLYK))
V_RMS_OBJETIVO = MACH_OBJETIVO * CS_SONORA
SEED_TURB = 4242
SEED_NULL = 101
SEED_LAYOUT = 12345

masa_pool = np.load('/tmp/pool_masa_N8000.npy')
dens_pool = np.load('/tmp/pool_dens_N8000.npy')

print(f"c_s={CS_SONORA:.6f}  Mach={MACH_OBJETIVO}  v_rms_objetivo={V_RMS_OBJETIVO:.6f}")

configs = []
for N in (2000, 8550):
    for brazo in ("turb", "hered"):
        for rama, seed_null in (("real", None), ("null", SEED_NULL)):
            d = f"/Users/alexis/phantom_cs073/run_vel_{brazo}_N{N}_{rama}"
            configs.append((N, brazo, rama, seed_null, d))

for N, brazo, rama, seed_null, d in configs:
    masa = masa_pool[:N]
    dens = dens_pool[:N]
    if brazo == "turb":
        vg = factory_turb(CS_SONORA, seed=SEED_TURB, mach=MACH_OBJETIVO)
    else:
        vg = factory_hered(V_RMS_OBJETIVO)
    ruta = f"{d}/cosmogenesis_ic.txt"
    info = traducir_pool(masa, dens, seed_null=seed_null, seed_layout=SEED_LAYOUT,
                          vel_generador=vg, ruta_salida=ruta)
    vel_check = np.array([[float(x) for x in l.split()[3:6]]
                           for l in open(ruta).readlines()[2:]])
    v_rms_final = float(np.sqrt(np.mean(np.sum(vel_check**2, axis=1))))
    print(f"N={N} brazo={brazo} rama={rama}: n={info['n']} a_final={info['a_final']:.4f} "
          f"v_rms_escrito={v_rms_final:.6f}")
