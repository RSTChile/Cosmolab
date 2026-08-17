"""
ON77_sistemaA_cierre.py — Sistema A del nodo O-N7.7, REENFOCADO a escala grande con masa total FIJA
(encargo de Alexis, 7-ago-2026, tras `ON77_sistemaAB_corregido_CS.md` -- ese intento anterior usó
N=50..400, todo por debajo del piso de resolución real (que sabemos ahora está entre N=500 y N=1000,
ver `INFRA_masa_fija_generador_CS.md`) -- dio CERO sumideros en las 4 corridas, sin poder probar nada.

Qué cambia acá respecto de `ON77_sistemaA_corregido.py` (NO tocado, sólo se replica el patrón de correr
Phantom porque ese script vive en este mismo proyecto pero es un intento anterior, no una pieza
congelada de la jerarquía CS073):

  1. El barrido de N ahora va HACIA ARRIBA desde el punto ya sabido limpio (N=2000, 8 sumideros ambas
     semillas en la batería original Y en la validación de masa fija) -- N ∈ {2000, 4000, 8000, (16000
     si el presupuesto de tiempo alcanza)} -- en vez de hacia abajo desde 50.
  2. La masa TOTAL del sistema queda FIJA en 18800 para TODOS los N (en vez de escalar con el pool de
     bariones extraído, que era el confound real que enmascaraba la pregunta). Se importan las
     constantes YA VALIDADAS de `grafo_random_layout_generar_ic_masa_fija.py` (LADO_FIJO,
     MASA_TOTAL_OBJETIVO, N_REFERENCIA_LADO) -- ese archivo NO se modifica, sólo se leen sus constantes.
     Consecuencia importante para la lectura: a partir de N=2000, subir N YA NO significa "más masa" --
     significa MÁS RESOLUCIÓN (partículas más livianas, mismo total de 18800). La pregunta de
     O-N7.7(a) (ganancia marginal decreciente) pasa a leerse como "¿más resolución sobre el MISMO
     presupuesto de masa sigue encontrando sumideros nuevos al mismo ritmo, o la ganancia marginal por
     partícula adicional cae?" -- no "¿más masa forma más estructura?" (eso sería trivial).
  3. La REGLA GENERATIVA de Sistema A no cambia: `malla_causal_atomos` (de `p_semilla_causal.py`,
     congelada, no tocada), kNN en UN SOLO pase sobre D=3 ejes derivados de la densidad, sin barajado
     (seed_null=None -> REAL exacto) -- "juntá todo y conectá cada nodo con sus k vecinos, una vez,
     listo", igual que el diseño original y que `ON77_sistemaA_corregido.py`.

De dónde sale `dens_bar` a cada N (ATAJO DOCUMENTADO, análogo al de `INFRA_masa_fija_generador_CS.md`
que submuestreó desde N=2000 hacia abajo): re-extraer un pool físico independiente por N con
`_extraer_bariones` a N=8000 tomaría un tiempo no acotado (el motor basal escala con los mismos
parámetros que ya eran caros a N~400) y el presupuesto de esta corrida es ~45-55 min TOTAL, incluyendo
Phantom real a N grande. En su lugar, se usa el pool YA EXTRAÍDO y validado de
`bateria_n2000/dens_bar.npy` (2000 valores REALES de densidad de átomos H+He, el mismo pool que usa
`bateria_n2000/ic_real`, la referencia física de toda la jerarquía CS073) y se lo RE-MUESTREA con
reemplazo (bootstrap, semilla fija) hasta alcanzar cada N >= 2000 pedido. Esto preserva la FORMA de la
distribución real de densidades (no es ruido uniforme inventado) pero ya NO es una extracción física
independiente por N -- es una simplificación de presupuesto de tiempo, documentada aquí con la misma
franqueza que el atajo de INFRA. `dens_bar` sólo alimenta (a) los ejes D=3 que ve `malla_causal_atomos`
y (b) el campo de velocidad turbulento -- NO determina la masa ni la caja (eso ahora es FIJO, ver punto
2), así que el atajo no reintroduce el confound que se está corrigiendo.

El layout de resortes (Fruchterman-Reingold, `layout_resortes`, congelado en `p_semilla_causal.py`,
NO tocado) es O(N^2) por iteración -- a N=8000 con el default de 100 iteraciones sería, por benchmark
propio hecho antes de escribir este script, demasiado lento para el presupuesto total. Se reduce
`ITERS_LAYOUT` (ver constante abajo, medida empíricamente) de forma UNIFORME para todos los N de este
sweep (mismo valor en N=2000/4000/8000/16000) -- así la comparación entre puntos de N sigue siendo
limpia (ningún N recibe más "esfuerzo de relajación" que otro), sólo se sacrifica cuánto converge la
semilla de posición ANTES de que Phantom la evolucione con su propia física real (gravedad, expansión,
enfriamiento) -- el layout es sólo la SIEMBRA, no la dinámica que produce los sumideros.

No toca ningún archivo/carpeta congelados (`grafo_random_layout_generar_ic_masa_fija.py`,
`leer_volcado_phantom.py`, ningún `bateria_*`/`piloto_*`/`ON77_*` existente) -- sólo IMPORTA de ellos.
Corridas de Phantom nuevas en `/Users/alexis/phantom_cs073/ON77_sistemaA_cierre/`.

No declara cierre ni veredicto sobre O-N7.7 ni CS073 -- sólo reporta números. La lectura es de Alexis.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cs073_cierre_holistico import T0, _T_reloj
from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos, layout_resortes
from fase1_traducir_a_phantom import HFACT, POLYK
from cs072_modulos.piezas.p_expansion import Expansion
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
# constantes YA VALIDADAS de la infra de masa fija -- import SOLAMENTE, ese archivo no se toca.
from grafo_random_layout_generar_ic_masa_fija import (
    LADO_FIJO, MASA_TOTAL_OBJETIVO, N_REFERENCIA_LADO,
)

CS_SONORA = POLYK ** 0.5
TURB_SEED = 42
D_CAUSAL = 3
K_CAUSAL = 4
SEED_EJES = 2000                 # mismo default que malla_causal_atomos / traducir_pool
SEED_LAYOUT = 12345
SEED_DENS_BOOTSTRAP = 4242       # semilla del re-muestreo con reemplazo de dens_bar (ver docstring)
ITERS_LAYOUT = 25                # reducido de 100 (default) por presupuesto de tiempo a N grande --
                                  # UNIFORME para todo el sweep, medido empíricamente antes de correr
                                  # (ver ON77_sistemaAB_cierre_CS.md, nota de benchmark).
N_PASOS_EXPANSION = 60           # mismo default que toda la jerarquía (a_final=√60≈7.75)

BASE = Path("/Users/alexis/phantom_cs073/ON77_sistemaA_cierre")
PHANTOMSETUP = "/Users/alexis/phantom_cs073/phantom/bin/phantomsetup_cosmogenesis_backup"
PHANTOM = "/Users/alexis/phantom_cs073/phantom/bin/phantom_cosmogenesis_backup"
DENS_BAR_N2000 = "/Users/alexis/phantom_cs073/bateria_n2000/dens_bar.npy"

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

COL_T = 0
COL_MACC = 11
COL_SINKID = 18

N_LIST = [2000, 4000, 8000]      # 16000 se agrega en runtime sólo si el presupuesto de tiempo alcanza


def dens_bar_para_n(n, seed=SEED_DENS_BOOTSTRAP):
    """Atajo documentado (ver docstring del módulo): re-muestrea CON reemplazo, semilla fija, desde el
    pool REAL de bateria_n2000 (N=2000) hasta alcanzar `n` valores. Para n<=2000 es un submuestreo SIN
    reemplazo (mismo patrón que INFRA_masa_fija_generador_CS.md, determinista)."""
    base = np.load(DENS_BAR_N2000)
    rng = np.random.default_rng(seed + n)   # semilla distinta por N, pero determinista y reproducible
    if n <= len(base):
        idx = rng.choice(len(base), size=n, replace=False)
    else:
        idx = rng.choice(len(base), size=n, replace=True)
    return base[idx]


def escribir_ic_masa_fija(n, dens_bar, ruta_salida):
    """Traduce la malla causal de UN SOLO pase (Sistema A, regla fija) a una IC de Phantom con masa
    total y caja FIJAS (independientes de n) -- mismo patrón de escritura ASCII que `traducir_pool`
    (fase1_traducir_a_phantom.py) y que `generar_control_random_masa_fija`
    (grafo_random_layout_generar_ic_masa_fija.py), combinando: la REGLA de Sistema A (malla_causal_atomos,
    sin barajar) con el TRATAMIENTO de masa fija (LADO_FIJO/MASA_TOTAL_OBJETIVO, importados sin tocar el
    archivo de origen)."""
    adj, _m = malla_causal_atomos(dens_bar, D=D_CAUSAL, k=K_CAUSAL, seed_ejes=SEED_EJES)  # REAL, sin barajar

    pos = layout_resortes(adj, n, lado=LADO_FIJO, iters=ITERS_LAYOUT, seed=SEED_LAYOUT)

    expansion = Expansion(T0=T0)
    for step in range(N_PASOS_EXPANSION):
        expansion.paso_de_estiramiento(_T_reloj(step))
    a_final = expansion._a_prev
    pos = pos * a_final

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)
    vel = vel_gen(pos, adj, dens_bar)
    h_guess = np.full(n, HFACT)
    masa_particula = MASA_TOTAL_OBJETIVO / n

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 Sistema-A-cierre (malla causal 1 pase, MASA FIJA) -- "
                f"npart={n} masa_particula={masa_particula:.17g} lado_fijo={LADO_FIJO:.6f} "
                f"(N_referencia={N_REFERENCIA_LADO}) masa_total_objetivo={MASA_TOTAL_OBJETIVO:.6g} "
                f"hfact={HFACT} polyk={POLYK:.17g} iters_layout={ITERS_LAYOUT}\n")
        f.write(f"{n} {masa_particula:.17g} {HFACT} {POLYK:.17g}\n")
        for i in range(n):
            f.write(f"{float(pos[i, 0]):.17g} {float(pos[i, 1]):.17g} {float(pos[i, 2]):.17g} "
                    f"{float(vel[i, 0]):.17g} {float(vel[i, 1]):.17g} {float(vel[i, 2]):.17g} "
                    f"{float(h_guess[i]):.17g}\n")

    return dict(ruta=ruta_salida, n=n, masa_particula=masa_particula, masa_total=masa_particula * n,
                lado=LADO_FIJO, a_final=a_final)


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
    """Misma lectura que `null3_bateria_comparar.py` (congelado, no tocado, sólo replicado el patrón):
    masa acretada acumulada en sumideros VIVOS al último tiempo del .sink -- 0/0 si nunca se creó
    ninguno, o si la corrida abortó antes de producir el archivo."""
    ruta = carpeta / "cosmog01.sink"
    if not ruta.exists():
        return dict(masa_total=0.0, n_sumideros=0, abortado=None)
    data = np.loadtxt(ruta, skiprows=2)
    if data.ndim == 1:
        data = data[None, :]
    t_final = data[:, COL_T].max()
    en_t_final = data[np.isclose(data[:, COL_T], t_final)]
    sink_ids = np.unique(en_t_final[:, COL_SINKID].astype(int))
    masas = [float(en_t_final[en_t_final[:, COL_SINKID].astype(int) == sid, COL_MACC][0])
             for sid in sink_ids]
    abortado = t_final < float(TMAX) - 1e-6
    return dict(masa_total=float(sum(masas)), n_sumideros=len(masas), abortado=bool(abortado),
                t_final=float(t_final))


def correr_un_n(n):
    carpeta = BASE / f"ic_N{n}"
    carpeta.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    dens_bar = dens_bar_para_n(n)
    info_ic = escribir_ic_masa_fija(n, dens_bar, str(carpeta / "cosmogenesis_ic.txt"))
    t_ic = time.time() - t0

    info_run = correr_phantom(carpeta)
    r_sink = masa_y_n_sumideros(carpeta)
    t_total = time.time() - t0

    fila = dict(N=n, masa_total=r_sink["masa_total"], n_sumideros=r_sink["n_sumideros"],
                masa_por_N=r_sink["masa_total"] / n if n else 0.0,
                abortado=r_sink["abortado"], t_final_sink=r_sink.get("t_final"),
                masa_total_ic=info_ic["masa_total"],
                t_ic_s=round(t_ic, 2), t_setup_s=round(info_run["t_setup"], 2),
                t_run_s=round(info_run["t_run"], 2), t_total_s=round(t_total, 2),
                exit_setup=info_run["exit_setup"], exit_run=info_run["exit_run"])
    print(f"[A-cierre] N={n:<6d} masa_total={r_sink['masa_total']:.2f} "
          f"n_sumideros={r_sink['n_sumideros']} masa/N={fila['masa_por_N']:.4f} "
          f"abortado={r_sink['abortado']} (ic={t_ic:.1f}s setup={info_run['t_setup']:.1f}s "
          f"run={info_run['t_run']:.1f}s total={t_total:.1f}s)", flush=True)
    if info_run["exit_run"] != 0 and not r_sink["abortado"]:
        tail = (carpeta / "run.log").read_text().splitlines()[-15:]
        print(f"  AVISO: exit_run != 0 en N={n}. Tail de run.log:\n  " + "\n  ".join(tail), flush=True)
    return fila


def main(n_list=N_LIST, extra_n16000=False):
    t_total = time.time()
    print(f"=== Sistema A cierre: N variable {n_list} (+16000 si alcanza el tiempo), regla fija (H=1), "
          f"MASA TOTAL FIJA ({MASA_TOTAL_OBJETIVO:.0f}), caja FIJA (lado={LADO_FIJO:.4f}) ===",
          flush=True)
    filas = []
    for n in n_list:
        filas.append(correr_un_n(n))
        with open(BASE / "ON77_sistemaA_cierre_resultado_parcial.json", "w") as fp:
            json.dump(dict(sistema_a=filas, tiempo_hasta_ahora_s=round(time.time() - t_total, 1)), fp,
                       indent=2)
        if time.time() - t_total > 33 * 60 and extra_n16000:
            print("[AVISO] presupuesto de tiempo (33 min) excedido dentro de Sistema A -- "
                  "no se agrega N=16000.", flush=True)
            extra_n16000 = False

    if extra_n16000 and 16000 not in n_list:
        filas.append(correr_un_n(16000))

    filas_ordenadas = sorted(filas, key=lambda r: r["N"])
    for idx, fila in enumerate(filas_ordenadas):
        if idx == 0:
            fila["ganancia_marginal_delta_N"] = None
        else:
            prev = filas_ordenadas[idx - 1]
            dN = fila["N"] - prev["N"]
            fila["ganancia_marginal_delta_N"] = ((fila["masa_total"] - prev["masa_total"]) / dN
                                                   if dN else None)

    resultado = dict(sistema_a_cierre=filas_ordenadas, iters_layout=ITERS_LAYOUT,
                      masa_total_objetivo=MASA_TOTAL_OBJETIVO, lado_fijo=LADO_FIJO,
                      tiempo_total_s=round(time.time() - t_total, 1))
    with open(BASE / "ON77_sistemaA_cierre_resultado.json", "w") as fp:
        json.dump(resultado, fp, indent=2)

    print(f"\n[TOTAL] {resultado['tiempo_total_s']}s -> "
          f"{BASE / 'ON77_sistemaA_cierre_resultado.json'}", flush=True)
    print("\nN       masa_total   masa/N      ganancia_marginal(delta_masa/delta_N)  abortado")
    for fila in filas_ordenadas:
        gm = fila["ganancia_marginal_delta_N"]
        gm_str = f"{gm:+.4f}" if gm is not None else "   --   "
        print(f"{fila['N']:<8d}{fila['masa_total']:<13.2f}{fila['masa_por_N']:<12.4f}{gm_str:<12}"
              f"{fila['abortado']}")
    return resultado


if __name__ == "__main__":
    extra = "--con-16000" in sys.argv
    sys.exit(0 if main(extra_n16000=extra) is not None else 1)
