"""
ON77_sistemaA_corregido.py — Sistema A del nodo O-N7.7 con el observable EXTENSIVO real (masa en
sumideros vía Phantom), corrigiendo el problema honesto que dejó `ON77_piloto_sistemaAB.py` (ver §8/§10
de `ON77_diseno_falsacion_sistemaAB_CS.md`, sección "Caveat honesto"): el piloto midió
`clustering_global/N`, un proxy de grafo que es INTENSIVO (acotado 0-1, no crece con la masa del
sistema) -- dividir un número casi-constante por un N creciente produce una curva decreciente "casi por
construcción aritmética", sin distinguir "saturación genuina" de "artefacto del proxy elegido".

Qué hace, en simple: para cada N (mismo rango que el piloto, N≈50/100/200/400), arma la MISMA regla
generativa de siempre (malla causal en UN SOLO pase de vecinos-más-cercanos, sin poda -- Sistema A no
cambia de definición, sólo de OBSERVABLE) y esta vez la traduce a una condición inicial de Phantom real
(`traducir_pool`, de `fase1_traducir_a_phantom.py` -- pieza ya validada de toda la jerarquía CS073, NO
reescrita aquí, sólo importada) y corre Phantom con los MISMOS parámetros físicos de sumideros que usa
`bateria_n2000/ic_real/cosmog.in` (rho_crit_cgs=1000, icreate_sinks=1, r_crit=0.6, h_acc=0.3,
tmax=0.500, dtmax=0.001 -- ver `grafo_random_bateria_correr.py`, mismo patrón, no tocado). La masa
acretada en sumideros AL FINAL de la corrida SÍ es una magnitud extensiva (crece con la masa total
disponible) -- el observable correcto para medir "ganancia marginal por recurso".

Observable reportado por N: masa_total en sumideros, masa_total/N (ganancia por partícula, análogo al
proxy anterior pero con la magnitud correcta en el numerador) y el delta MARGINAL entre puntos
consecutivos: (masa(N_i) - masa(N_{i-1})) / (N_i - N_{i-1}) -- cuánta masa adicional aporta CADA
partícula extra al pasar de un N al siguiente, que es la forma más directa de leer "ganancia marginal
decreciente" (predicción O-N7.7(a)) sin pasar por ningún cociente que pueda venir sesgado por un
denominador que crece más rápido que el numerador.

No toca ningún archivo/carpeta congelados (`ON77_piloto_sistemaAB.py`, `bateria_n2000/`, ningún otro
`bateria_*`/`piloto_*`) -- sólo los LEE como plantilla de parámetros físicos. Las corridas de Phantom de
este script viven en una carpeta NUEVA: `/Users/alexis/phantom_cs073/ON77_sistemaA_corregido/`.

No declara cierre ni veredicto sobre O-N7.7 ni CS073 -- sólo reporta números. La lectura es de Alexis.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cs073_cierre_holistico import _extraer_bariones
from fase1_traducir_a_phantom import traducir_pool, POLYK
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO

# campo de velocidad turbulento (Mach=3, TURB_SEED=42) -- MISMO que usa el resto de la jerarquía
# (grafo_random_bateria_generar.py, null1/2/3_bateria_generar.py, todos congelados). SIN esto, Phantom
# arranca con L_inicial=0 EXACTO (v=0 en todas las partículas por construcción) y el guardián de
# conservación de momento angular aborta la corrida (`ERROR! evolve: Large error in angular momentum
# conservation`) -- comprobado empíricamente en el primer intento de este script (ver logs), no una
# suposición: confirma el mismo artefacto ya documentado en fase1_traducir_a_phantom.py.
CS_SONORA = POLYK ** 0.5
TURB_SEED = 42

BASE = Path("/Users/alexis/phantom_cs073/ON77_sistemaA_corregido")
PHANTOMSETUP = "/Users/alexis/phantom_cs073/phantom/bin/phantomsetup_cosmogenesis_backup"
PHANTOM = "/Users/alexis/phantom_cs073/phantom/bin/phantom_cosmogenesis_backup"

# MISMO bloque de sumideros + tmax/dtmax que toda la jerarquía CS073 (bateria_n2000/ic_real/cosmog.in),
# copiado literal de `grafo_random_bateria_correr.py` (congelado, no tocado -- sólo se replica el texto
# aquí porque ese script vive fuera de este directorio y no se importa entre proyectos).
BLOQUE_SINKS_DEFAULT = """# options controlling sink particles
     isink_potential =           0    ! sink potential (0=1/r,1=surf)
       icreate_sinks =           0    ! allow automatic sink particle creation
     h_soft_sinksink =       0.000    ! softening length between sink particles
               f_acc =       0.800    ! particles < f_acc*h_acc accreted without checks"""

BLOQUE_SINKS_CS073 = """# options controlling sink particles
     isink_potential =           0    ! sink potential (0=1/r,1=surf)
       icreate_sinks =           1    ! allow automatic sink particle creation
        rho_crit_cgs =       1000.    ! density above which sink particles are created (g/cm^3)
              r_crit =       0.600    ! critical radius for point mass creation (no new sinks < r_crit from existing sink)
               h_acc =       0.300    ! accretion radius for new sink particles
      h_soft_sinkgas =       0.000    ! softening length for new sink particles
     h_soft_sinksink =       0.000    ! softening length between sink particles
               f_acc =       0.800    ! particles < f_acc*h_acc accreted without checks
      r_merge_uncond =       0.000    ! sinks will unconditionally merge within this separation
        r_merge_cond =       0.000    ! sinks will merge if bound within this radius"""

TMAX = "0.500"
DTMAX = "0.001"

# columnas del .sink -- MISMA convención que `null3_bateria_comparar.py` (congelado, no tocado, sólo
# replicado el patrón: col 0 = tiempo, col 11 = masa acretada acumulada, col 18 = ID del sumidero).
COL_T = 0
COL_MACC = 11
COL_SINKID = 18

# mismo rango de N que el piloto (ON77_piloto_sistemaAB.py, params_a) -- f=0.5/1/2/4 sobre
# (nq,naq,ne,npos)=(600,420,200,140) -- reusado tal cual para poder comparar directamente con el piloto
# de grafo (mismos pools de bariones, misma malla causal, sólo cambia el observable final).
PARAMS_A = {
    0.5: (300, 210, 100, 70),
    1.0: (600, 420, 200, 140),
    2.0: (1200, 840, 400, 280),
    4.0: (2400, 1680, 800, 560),
}


def editar_cosmog_in(ruta: Path) -> None:
    texto = ruta.read_text()
    assert BLOQUE_SINKS_DEFAULT in texto, (
        f"{ruta}: el bloque de sumideros por defecto no coincide con lo esperado -- "
        "¿cambió el binario phantomsetup_cosmogenesis_backup? no se edita a ciegas.")
    texto = texto.replace(BLOQUE_SINKS_DEFAULT, BLOQUE_SINKS_CS073)
    texto = re.sub(r"(?m)^(\s*tmax\s*=\s*)\S+(\s*!)", rf"\g<1>{TMAX}   \g<2>", texto)
    texto = re.sub(r"(?m)^(\s*dtmax\s*=\s*)\S+(\s*!)", rf"\g<1>{DTMAX}   \g<2>", texto)
    ruta.write_text(texto)


def correr_phantom(carpeta: Path) -> dict:
    ic = carpeta / "cosmogenesis_ic.txt"
    assert ic.exists(), f"falta {ic}"

    t0 = time.time()
    with open(carpeta / "setup.log", "w") as f:
        r_setup = subprocess.run([PHANTOMSETUP, "cosmog"], cwd=carpeta, stdin=subprocess.DEVNULL,
                                  stdout=f, stderr=subprocess.STDOUT)
    t_setup = time.time() - t0
    assert r_setup.returncode == 0, f"phantomsetup falló en {carpeta} (ver setup.log)"

    editar_cosmog_in(carpeta / "cosmog.in")

    t1 = time.time()
    with open(carpeta / "run.log", "w") as f:
        r_run = subprocess.run([PHANTOM, "cosmog.in"], cwd=carpeta, stdin=subprocess.DEVNULL,
                                stdout=f, stderr=subprocess.STDOUT)
    t_run = time.time() - t1

    return dict(exit_setup=r_setup.returncode, t_setup=t_setup, exit_run=r_run.returncode, t_run=t_run)


def masa_y_n_sumideros(carpeta: Path) -> dict:
    """Masa total acretada en sumideros vivos al final de la corrida (0/0 si no formó ninguno) --
    reimplementación línea a línea de `null3_bateria_comparar.py` (congelado, no tocado)."""
    ruta = carpeta / "cosmog01.sink"
    if not ruta.exists():
        return dict(masa_total=0.0, n_sumideros=0)
    data = np.loadtxt(ruta, skiprows=2)
    if data.ndim == 1:
        data = data[None, :]
    t_final = data[:, COL_T].max()
    en_t_final = data[np.isclose(data[:, COL_T], t_final)]
    sink_ids = np.unique(en_t_final[:, COL_SINKID].astype(int))
    masas = [float(en_t_final[en_t_final[:, COL_SINKID].astype(int) == sid, COL_MACC][0])
             for sid in sink_ids]
    return dict(masa_total=float(sum(masas)), n_sumideros=len(masas))


def sweep_sistema_a_phantom(params_por_f=PARAMS_A, pasos_basal=150, amp_rugosidad=1.5):
    filas = []
    pools = {}
    for f, params in params_por_f.items():
        carpeta = BASE / f"ic_f{f}"
        carpeta.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        masa_bar, dens_bar, obs = _extraer_bariones(*params, pasos_basal, amp_rugosidad)
        pools[f] = (masa_bar, dens_bar)
        n = len(masa_bar)
        t_extraccion = time.time() - t0

        # Sistema A = malla causal en UN SOLO pase, sin barajado (seed_null=None -> REAL exacto) --
        # traducir_pool es la pieza YA VALIDADA de toda la jerarquía CS073 (fase1_traducir_a_phantom.py,
        # NO tocada aquí, sólo importada), mismos D_causal=3/k_causal=4/seed_ejes=2000 que usaba el
        # piloto de grafo -- así el pool de bariones y la malla causal son IDÉNTICOS al piloto anterior,
        # sólo cambia que ahora se traduce a Phantom en vez de quedarse en clustering de grafo.
        vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)
        info_ic = traducir_pool(masa_bar, dens_bar, seed_null=None, vel_generador=vel_gen,
                                 ruta_salida=str(carpeta / "cosmogenesis_ic.txt"))

        info_run = correr_phantom(carpeta)
        r_sink = masa_y_n_sumideros(carpeta)
        t_total = time.time() - t0

        fila = dict(f=f, N=n, masa_total=r_sink["masa_total"], n_sumideros=r_sink["n_sumideros"],
                    masa_por_N=r_sink["masa_total"] / n if n else 0.0,
                    t_extraccion_s=round(t_extraccion, 2), t_setup_s=round(info_run["t_setup"], 2),
                    t_run_s=round(info_run["t_run"], 2), t_total_s=round(t_total, 2),
                    exit_setup=info_run["exit_setup"], exit_run=info_run["exit_run"])
        filas.append(fila)
        print(f"[A-Phantom] f={f} N={n} masa_total={r_sink['masa_total']:.2f} "
              f"n_sumideros={r_sink['n_sumideros']} masa/N={fila['masa_por_N']:.4f} "
              f"(extracción={t_extraccion:.1f}s setup={info_run['t_setup']:.1f}s "
              f"run={info_run['t_run']:.1f}s)", flush=True)
        if info_run["exit_run"] != 0:
            tail = (carpeta / "run.log").read_text().splitlines()[-15:]
            print(f"  AVISO: exit_run != 0 en f={f}. Tail de run.log:\n  " + "\n  ".join(tail),
                  flush=True)

    # ganancia marginal REAL entre puntos consecutivos de N: (masa_i - masa_{i-1}) / (N_i - N_{i-1}).
    filas_ordenadas = sorted(filas, key=lambda r: r["N"])
    for idx, fila in enumerate(filas_ordenadas):
        if idx == 0:
            fila["ganancia_marginal_delta_N"] = None
        else:
            prev = filas_ordenadas[idx - 1]
            dN = fila["N"] - prev["N"]
            fila["ganancia_marginal_delta_N"] = ((fila["masa_total"] - prev["masa_total"]) / dN
                                                   if dN else None)

    return filas_ordenadas, pools


def main():
    t_total = time.time()
    print("=== Sistema A corregido: N variable, regla fija (H=1), observable EXTENSIVO real "
          "(masa en sumideros vía Phantom) ===", flush=True)
    filas, pools = sweep_sistema_a_phantom()

    resultado = dict(sistema_a_phantom=filas, tiempo_total_s=round(time.time() - t_total, 1))
    with open(BASE / "ON77_sistemaA_corregido_resultado.json", "w") as fp:
        json.dump(resultado, fp, indent=2)

    print(f"\n[TOTAL] {resultado['tiempo_total_s']}s -> "
          f"{BASE / 'ON77_sistemaA_corregido_resultado.json'}", flush=True)
    print("\nN       masa_total   masa/N      ganancia_marginal(delta_masa/delta_N)")
    for fila in filas:
        gm = fila["ganancia_marginal_delta_N"]
        gm_str = f"{gm:+.4f}" if gm is not None else "   --   "
        print(f"{fila['N']:<8d}{fila['masa_total']:<13.2f}{fila['masa_por_N']:<12.4f}{gm_str}")

    return resultado


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
