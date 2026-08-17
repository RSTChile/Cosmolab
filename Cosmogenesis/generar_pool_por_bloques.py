"""
generar_pool_por_bloques.py -- pool de ~N_min átomos SIN materializar Bq a esa escala.

INSTRUCCION_CC_memoria_N2_PARA_CC.md, Opción B. El motor basal (congelado, nucleo.py) NO se toca.

CORRECCIÓN (encontrada al implementar, antes de correr nada): el motor es determinista -- CERO azar, por
diseño anti-Shannon (verificado: dos llamadas a corre() con los MISMOS nq/naq/ne/npos dan resultados
BIT-IDÉNTICOS). Repetir la MISMA escala N_BLOQUES veces habría dado N_BLOQUES copias idénticas de los
mismos átomos, no un pool más grande. La corrección: cada bloque usa una escala f DISTINTA (f=1..K) --
el catálogo y la densidad #23 (densidad_intrinseca, catalogo.py) dependen de N=nq+naq+ne+npos, así que
f distintos dan construcciones deterministas GENUINAMENTE distintas (no duplicados). Efecto colateral
favorable: el costo por átomo crece superlinealmente con f, así que muchos bloques CHICOS (f=1..18) es
más barato en total que pocos bloques grandes -- ~2200s estimados vs correr todo de una sola vez.

El pool final es la CONCATENACIÓN de masa/densidad de todos los bloques -- una lista O(N), nunca una
matriz N² a escala de ignición (Bq se libera por el GC de Python al salir de cada llamada a corre()).
"""
import resource
import time
import numpy as np

from cs072_modulos.nucleo import corre

NQ_BASE, NAQ_BASE, NE_BASE, NPOS_BASE = 300, 210, 100, 70
PASOS_BASAL = 150
AMP_RUGOSIDAD = 1.5


def pico_ram_gb():
    """Pico de RSS del PROCESO hasta ahora (macOS: ru_maxrss en bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def generar_bloque(f):
    """Un bloque = una corrida basal independiente a escala f (DISTINTA de todos los demás bloques --
    ver corrección arriba). Determinista, sin semilla: la escala f ES lo que lo hace distinto."""
    obs, e = corre(NQ_BASE * f, NAQ_BASE * f, NE_BASE * f, NPOS_BASE * f,
                    tasa_expansion=0.02, pasos=PASOS_BASAL, amp_rugosidad=AMP_RUGOSIDAD,
                    devolver_estado=True)
    atomos_H = [n for (n, _) in e.Bem]
    masa = np.array([e.masa_trio.get(a, 1.0) for a in atomos_H], float)
    densidad = np.array([e.densidad[a] for a in atomos_H], float)
    return masa, densidad, obs.get("hidrogeno"), obs.get("helio")


if __name__ == "__main__":
    import sys
    f_max = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    fs = list(range(1, f_max + 1))
    print(f"bloques f=1..{f_max}  (H esperado = 50*sum(f) = {50*sum(fs)})", flush=True)

    masas, densidades = [], []
    t0 = time.time()
    for f in fs:
        tb = time.time()
        masa, dens, H, He = generar_bloque(f)
        masas.append(masa)
        densidades.append(dens)
        print(f"bloque f={f}: H={H} He={He} t={time.time()-tb:.1f}s "
              f"pico_RAM={pico_ram_gb():.2f}GB acumulado_atomos={sum(len(m) for m in masas)}", flush=True)

    masa_pool = np.concatenate(masas)
    dens_pool = np.concatenate(densidades)
    # verificación explícita: NO hay bloques duplicados (cada f es distinto -> arrays distintos)
    n_bloques_unicos = len(set(tuple(d) for d in densidades))
    print(f"\nbloques con densidad ÚNICA: {n_bloques_unicos} de {len(densidades)} (debe ser {len(fs)}/{len(fs)})")
    np.save("/tmp/pool_masa_N8000.npy", masa_pool)
    np.save("/tmp/pool_dens_N8000.npy", dens_pool)
    print(f"POOL FINAL: {len(masa_pool)} átomos, tiempo total={time.time()-t0:.1f}s, "
          f"pico RAM del proceso={pico_ram_gb():.2f}GB")
