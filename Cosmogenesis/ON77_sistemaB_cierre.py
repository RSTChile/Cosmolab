"""
ON77_sistemaB_cierre.py — Sistema B del nodo O-N7.7, REENFOCADO a la escala confiable (N=2000 FIJO, en
vez de N=200 como en `ON77_sistemaB_corregido.py`), reusando TAL CUAL el mecanismo de "reorganización de
presupuesto acotado" ya validado en ese intento anterior (memoria genuina de grafo: Jaccard se aleja de
1.0 de forma monótona con H, ver `ON77_sistemaAB_corregido_CS.md` §Resultado 1) -- lo único que cambia es
la escala N y el generador de condición inicial (masa fija en vez de masa heredada del pool físico).

Por qué N=2000 y no N=200 (encargo de Alexis, mismo antecedente que Sistema A cierre): a N=200 todas las
corridas de Phantom del intento anterior dieron CERO sumideros -- N=200 está muy por debajo del piso de
resolución real (documentado entre N=500 y N=1000 en `INFRA_masa_fija_generador_CS.md`), así que
cualquier efecto de H quedaba sin poder medirse en el observable extensivo (masa en sumideros). N=2000 es
el punto donde CS073 YA SABE que se forman sumideros de forma limpia y consistente (8 sumideros, ambas
semillas, tanto en la batería original como en la validación de masa fija) -- es el punto "confiable"
que pidió Alexis para este cierre.

Qué NO cambia respecto de `ON77_sistemaB_corregido.py` (NO tocado, sólo se replica su mecanismo porque
ese script vive en este mismo proyecto y no es una pieza congelada importable entre archivos sin
duplicar código -- ver su propio docstring para la justificación completa del mecanismo):
  - `construir_malla_historica_acotada`: universo de N nodos revelado en H tandas acumulativas (orden
    fijo, SEED_ORDEN_HISTORIA); nodos NUEVOS compiten libremente por sus k vecinos entre todo lo
    revelado; nodos YA colocados reciben UNA oportunidad ACOTADA de reconsiderar contra una MUESTRA
    aleatoria de tamaño fijo (TAM_MUESTRA_RECONSIDERACION) -- si el mejor candidato de esa muestra es
    estrictamente más cercano que el vecino actual más lejano, se poda y reconecta. H=1 = sin historia
    (ancla). Esto es lo que ya demostró memoria genuina (Jaccard != 1.0 para H>1) en el intento anterior.
  - El sweep de grafo (barato) corre SIEMPRE antes de Phantom, con la misma verificación obligatoria: si
    el Jaccard vuelve a salir degenerado (todo 1.0) a esta escala, NO se escala a Phantom.

Qué SÍ cambia (la corrección de escala que pidió Alexis):
  1. N fijo = 2000 (no 200) -- el pool de densidad usado para construir la malla histórica es
     `bateria_n2000/dens_bar.npy`, el mismo pool REAL de 2000 átomos que usa toda la jerarquía CS073
     (ic_real, la validación de masa fija de INFRA, etc.) -- no un pool extraído ad hoc con
     `_extraer_bariones` como hacía el intento anterior. Esto además ahorra la extracción (que a N~200
     tomaba tiempo no trivial) reusando un artefacto ya en disco.
  2. La traducción a IC de Phantom (`escribir_ic_desde_adj` en el script anterior) pasa a masa TOTAL fija
     (`MASA_TOTAL_OBJETIVO=18800`, MISMA constante que usa Sistema A cierre y la infra de masa fija) y
     caja fija (`LADO_FIJO`, el mismo lado congelado que usa toda la jerarquía de masa fija) -- en vez de
     `masa_bar.mean()` heredado del pool y `lado = n**(1/3)`. A N=2000 exacto, `LADO_FIJO` YA ES
     `2000**(1/3)` (el generador de masa fija fija el lado usando N=2000 como referencia) -- así que el
     cambio de caja es un NO-OP a esta escala. La masa por partícula también coincide con el valor por
     defecto de toda la jerarquía a N=2000 (18800/2000=9.4, el mismo `masa_bar.mean()` real medido en
     `MISTERIO_N500_vs_N2000_CS.md`) -- el cambio es mínimo, tal como anticipó el encargo.
  3. `iters_layout` del FR se reduce de 100 a 25 (mismo valor y misma justificación empírica que Sistema
     A cierre, ver ese script: a N=2000, iters=100 mide ~50s por corrida: con 5 puntos de H eso son
     ~250s sólo de layout, que igual entra en el presupuesto -- pero se reduce igual por CONSISTENCIA con
     Sistema A cierre, para que ambos sistemas de este cierre reciban el mismo esfuerzo de relajación de
     semilla y ninguna diferencia entre A y B pueda atribuirse a un layout más o menos convergido).

No toca ningún archivo/carpeta congelados (`ON77_sistemaB_corregido.py`, `p_semilla_causal.py`,
`cs072_modulos/proceso_sucesivo.py`, `grafo_random_layout_generar_ic_masa_fija.py`, ningún
`bateria_*`/`piloto_*`/`ON77_*` existente) -- sólo IMPORTA de ellos. Corridas de Phantom nuevas en
`/Users/alexis/phantom_cs073/ON77_sistemaB_cierre/`.

No declara cierre ni veredicto sobre O-N7.7 ni CS073 -- sólo reporta números. La lectura es de Alexis.
"""
import collections
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cs073_cierre_holistico import T0, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion
from cs072_modulos.piezas.p_semilla_causal import _ejes_desde_densidad, layout_resortes, barajar_aristas
from fase1_traducir_a_phantom import HFACT, POLYK
from null3_investigacion_preliminar import aristas_de
from null3_motivos_directos import contar_triangulos_y_clustering
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO
# constantes YA VALIDADAS de la infra de masa fija -- import SOLAMENTE, ese archivo no se toca.
from grafo_random_layout_generar_ic_masa_fija import LADO_FIJO, MASA_TOTAL_OBJETIVO

CS_SONORA = POLYK ** 0.5
TURB_SEED = 42

D_CAUSAL = 3
K_CAUSAL = 4
SEED_EJES = 2000
SEED_ORDEN_HISTORIA = 777        # MISMA semilla que ON77_sistemaB_corregido.py (mismo mecanismo)
SEED_RECONSIDERACION = 555
SEED_NULL = 909
SEED_LAYOUT = 12345
TAM_MUESTRA_RECONSIDERACION = 3  # MISMO valor ya validado (memoria genuina, Jaccard != 1.0 para H>1)
ITERS_LAYOUT = 25                # reducido de 100, ver docstring -- mismo criterio que Sistema A cierre
N_PASOS_EXPANSION = 60

BASE = Path("/Users/alexis/phantom_cs073/ON77_sistemaB_cierre")
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

H_LIST = [1, 2, 4, 8, 16]


# ----------------------------------------------------------------------------------------------
# Mecanismo de reorganización ACOTADA -- reimplementado línea a línea de `ON77_sistemaB_corregido.py`
# (NO tocado, sólo replicado porque no es un módulo importable entre archivos de este proyecto).
# ----------------------------------------------------------------------------------------------
def construir_malla_historica_acotada(dens_bar, n_batches, D=D_CAUSAL, k=K_CAUSAL, seed_ejes=SEED_EJES,
                                       seed_orden=SEED_ORDEN_HISTORIA, tam_muestra=TAM_MUESTRA_RECONSIDERACION,
                                       seed_reconsideracion=SEED_RECONSIDERACION):
    n = len(dens_bar)
    V = _ejes_desde_densidad(dens_bar, D, seed_ejes)

    diffs = V[:, None, :] - V[None, :, :]
    dist = np.sqrt((diffs ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)

    orden = np.random.default_rng(seed_orden).permutation(n)
    cortes = np.array_split(orden, min(n_batches, n))

    adj = collections.defaultdict(set)
    revelados = []
    rng_recon = np.random.default_rng(seed_reconsideracion)
    n_reorg = 0

    for batch in cortes:
        nuevos = [int(x) for x in batch]
        revelados_antes = list(revelados)
        revelados.extend(nuevos)

        for i in nuevos:
            candidatos = [j for j in revelados if j != i]
            if not candidatos:
                continue
            cand_arr = np.array(candidatos)
            kk = min(k, len(candidatos))
            vecinos_idx = np.argsort(dist[i, cand_arr])[:kk]
            for idx in vecinos_idx:
                j = int(cand_arr[idx])
                adj[i].add(j)
                adj[j].add(i)

        for p in revelados_antes:
            vecinos_p = adj.get(p, set())
            if not vecinos_p:
                continue
            vecinos_arr = np.array(list(vecinos_p))
            d_vecinos = dist[p, vecinos_arr]
            idx_lejano = int(np.argmax(d_vecinos))
            vecino_lejano = int(vecinos_arr[idx_lejano])
            d_lejano = float(d_vecinos[idx_lejano])

            pool_candidatos = [j for j in revelados if j != p and j not in vecinos_p]
            if not pool_candidatos:
                continue
            m = min(tam_muestra, len(pool_candidatos))
            muestra = rng_recon.choice(np.array(pool_candidatos), size=m, replace=False)
            d_muestra = dist[p, muestra]
            idx_mejor = int(np.argmin(d_muestra))
            mejor_cand = int(muestra[idx_mejor])
            d_mejor = float(d_muestra[idx_mejor])

            if d_mejor < d_lejano:
                adj[p].discard(vecino_lejano)
                adj[vecino_lejano].discard(p)
                adj[p].add(mejor_cand)
                adj[mejor_cand].add(p)
                n_reorg += 1

    return dict(adj), n, n_reorg


def _jaccard_aristas(adj_a, adj_b, n):
    ea = set(aristas_de(adj_a, n))
    eb = set(aristas_de(adj_b, n))
    if not ea and not eb:
        return 1.0
    return len(ea & eb) / len(ea | eb)


def _real_y_null(adj, n, seed_null=SEED_NULL):
    r_real = contar_triangulos_y_clustering(adj, n)
    adj_null = barajar_aristas(adj, n, seed=seed_null)
    r_null = contar_triangulos_y_clustering(adj_null, n)
    delta = r_real["clustering_global"] - r_null["clustering_global"]
    return r_real, r_null, delta


def sweep_grafo(dens_bar, n_batches_list):
    n = len(dens_bar)
    adj_h1, _n1, _r1 = construir_malla_historica_acotada(dens_bar, n_batches=1)
    filas = []
    for hb in n_batches_list:
        t0 = time.time()
        adj_h, _n, n_reorg = construir_malla_historica_acotada(dens_bar, n_batches=hb)
        r_real, r_null, delta = _real_y_null(adj_h, n)
        jac = _jaccard_aristas(adj_h, adj_h1, n)
        t = time.time() - t0
        fila = dict(H=hb, N=n, clustering_real=r_real["clustering_global"],
                    clustering_null=r_null["clustering_global"], delta=delta,
                    jaccard_vs_H1=jac, n_reorganizaciones=n_reorg, tiempo_s=round(t, 3),
                    n_triangulos_real=r_real["n_triangulos"], n_triangulos_null=r_null["n_triangulos"],
                    adj=adj_h)
        filas.append(fila)
        print(f"[B-grafo] H={hb:<3d} N={n} C_real={r_real['clustering_global']:.5f} "
              f"C_null={r_null['clustering_global']:.5f} delta={delta:+.5f} "
              f"jaccard_vs_H1={jac:.4f} n_reorg={n_reorg} t={t:.3f}s", flush=True)
    return filas


def escribir_ic_masa_fija_desde_adj(dens_bar, adj, ruta_salida, n_pasos_expansion=N_PASOS_EXPANSION):
    """Misma construcción que `escribir_ic_desde_adj` de `ON77_sistemaB_corregido.py`, con el
    tratamiento de MASA FIJA (LADO_FIJO/MASA_TOTAL_OBJETIVO) en vez de masa/lado heredados de n."""
    n = len(dens_bar)
    pos = layout_resortes(adj, n, lado=LADO_FIJO, iters=ITERS_LAYOUT, seed=SEED_LAYOUT)

    expansion = Expansion(T0=T0)
    for step in range(n_pasos_expansion):
        expansion.paso_de_estiramiento(_T_reloj(step))
    a_final = expansion._a_prev
    pos = pos * a_final

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)
    vel = vel_gen(pos, adj, dens_bar)
    h_guess = np.full(n, HFACT)
    masa_particula = MASA_TOTAL_OBJETIVO / n

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 Sistema-B-cierre (reorganizacion acotada, MASA FIJA) -- npart={n} "
                f"masa_particula={masa_particula:.17g} lado_fijo={LADO_FIJO:.6f} "
                f"masa_total_objetivo={MASA_TOTAL_OBJETIVO:.6g} hfact={HFACT} polyk={POLYK:.17g} "
                f"iters_layout={ITERS_LAYOUT}\n")
        f.write(f"{n} {masa_particula:.17g} {HFACT} {POLYK:.17g}\n")
        for i in range(n):
            f.write(f"{float(pos[i, 0]):.17g} {float(pos[i, 1]):.17g} {float(pos[i, 2]):.17g} "
                    f"{float(vel[i, 0]):.17g} {float(vel[i, 1]):.17g} {float(vel[i, 2]):.17g} "
                    f"{float(h_guess[i]):.17g}\n")
    return dict(ruta=ruta_salida, n=n, masa_total=masa_particula * n, a_final=a_final)


def editar_cosmog_in(ruta: Path) -> None:
    texto = ruta.read_text()
    assert BLOQUE_SINKS_DEFAULT in texto, (
        f"{ruta}: bloque de sumideros por defecto inesperado -- no se edita a ciegas.")
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
    ruta = carpeta / "cosmog01.sink"
    if not ruta.exists():
        return dict(masa_total=0.0, n_sumideros=0, abortado=None, t_final=None)
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


def sweep_phantom(dens_bar, filas_grafo):
    filas_out = []
    for fila in filas_grafo:
        hb = fila["H"]
        carpeta = BASE / f"ic_H{hb}"
        carpeta.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        escribir_ic_masa_fija_desde_adj(dens_bar, fila["adj"], str(carpeta / "cosmogenesis_ic.txt"))
        info_run = correr_phantom(carpeta)
        r_sink = masa_y_n_sumideros(carpeta)
        t_total = time.time() - t0
        fila_out = dict(H=hb, N=fila["N"], masa_total=r_sink["masa_total"],
                         n_sumideros=r_sink["n_sumideros"], abortado=r_sink["abortado"],
                         t_final_sink=r_sink["t_final"], t_total_s=round(t_total, 2),
                         exit_setup=info_run["exit_setup"], exit_run=info_run["exit_run"])
        filas_out.append(fila_out)
        print(f"[B-Phantom] H={hb:<3d} masa_total={r_sink['masa_total']:.2f} "
              f"n_sumideros={r_sink['n_sumideros']} abortado={r_sink['abortado']} t={t_total:.1f}s",
              flush=True)
        with open(BASE / "ON77_sistemaB_cierre_resultado_parcial.json", "w") as fp:
            json.dump(dict(sistema_b_phantom_hasta_ahora=filas_out), fp, indent=2)
        if info_run["exit_run"] != 0 and not r_sink["abortado"]:
            tail = (carpeta / "run.log").read_text().splitlines()[-15:]
            print(f"  AVISO: exit_run != 0 en H={hb}. Tail:\n  " + "\n  ".join(tail), flush=True)
    return filas_out


def main():
    t_total = time.time()
    print("=== Sistema B cierre: N=2000 FIJO (real, bateria_n2000/dens_bar.npy), mecanismo de "
          "reorganizacion ACOTADA, H variable, MASA TOTAL FIJA ===", flush=True)

    dens_bar = np.load(DENS_BAR_N2000)
    n = len(dens_bar)
    print(f"[1] N={n} (pool REAL de bateria_n2000, sin re-extraer)", flush=True)

    print(f"\n[2] sweep de GRAFO (barato, obligatorio ANTES de Phantom) H={H_LIST}...", flush=True)
    filas_grafo = sweep_grafo(dens_bar, n_batches_list=H_LIST)

    jaccards = [f["jaccard_vs_H1"] for f in filas_grafo if f["H"] != 1]
    degenerado = all(abs(j - 1.0) < 1e-9 for j in jaccards)

    filas_grafo_json = [{k: v for k, v in f.items() if k != "adj"} for f in filas_grafo]

    if degenerado:
        print("\n[VEREDICTO DE VERIFICACIÓN] Jaccard SIGUE siendo 1.0000 en TODO H a N=2000 -- el "
              "mecanismo, tal como está, es degenerado A ESTA ESCALA. NO se escala a Phantom.",
              flush=True)
        resultado = dict(sistema_b_grafo=filas_grafo_json, sistema_b_phantom=None,
                          degenerado=True, tiempo_total_s=round(time.time() - t_total, 1))
    else:
        print(f"\n[VERIFICACIÓN OK] Jaccard vs H=1: {[round(j,4) for j in jaccards]} -- el mecanismo "
              "tiene memoria genuina del camino también a N=2000. Se procede a Phantom.", flush=True)
        print("\n[3] traduciendo cada H a IC de Phantom (masa fija) y corriendo...", flush=True)
        filas_phantom = sweep_phantom(dens_bar, filas_grafo)
        resultado = dict(sistema_b_grafo=filas_grafo_json, sistema_b_phantom=filas_phantom,
                          degenerado=False, tiempo_total_s=round(time.time() - t_total, 1))

    resultado["iters_layout"] = ITERS_LAYOUT
    resultado["masa_total_objetivo"] = MASA_TOTAL_OBJETIVO
    resultado["lado_fijo"] = LADO_FIJO

    with open(BASE / "ON77_sistemaB_cierre_resultado.json", "w") as fp:
        json.dump(resultado, fp, indent=2)
    print(f"\n[TOTAL] {resultado['tiempo_total_s']}s -> "
          f"{BASE / 'ON77_sistemaB_cierre_resultado.json'}", flush=True)
    return resultado


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
