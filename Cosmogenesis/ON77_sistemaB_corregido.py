"""
ON77_sistemaB_corregido.py — Sistema B del nodo O-N7.7 con MEMORIA GENUINA, corrigiendo el hallazgo
metodológico honesto del piloto (`ON77_piloto_sistemaAB.py` §9 del .md de diseño): la implementación
anterior (`construir_malla_historica`) recalculaba `_malla_causal` DESDE CERO sobre TODO lo revelado en
cada tanda -- y como el kNN sobre un embedding estático es una operación SIN memoria (converge al mismo
óptimo global sin importar el camino), la última tanda SIEMPRE reconstruía exactamente el mismo grafo
que un solo pase (H=1). Resultado: Jaccard=1.0000 en H=1,2,4,8 -- degenerado, "más pasadas de la MISMA
regla óptima" no operacionaliza historia.

Mecanismo NUEVO (`construir_malla_historica_acotada`), implementando la propuesta ya diseñada en §9 del
.md ("reorganización de presupuesto ACOTADO"):

  1. El universo de N nodos (N FIJO, igual que el piloto) se revela en H tandas acumulativas, mismo
     orden fijo para cualquier H (control limpio, igual que el mecanismo anterior).
  2. Los nodos NUEVOS de cada tanda compiten LIBREMENTE por sus k vecinos más cercanos entre TODO lo
     revelado hasta el momento (razonable: un nodo recién aparecido necesita conectarse con algo).
  3. Los nodos YA colocados en tandas anteriores NO se recalculan por completo -- cada uno recibe UNA
     oportunidad ACOTADA de reconsiderar: se lo compara contra una MUESTRA ALEATORIA de tamaño FIJO
     (`tam_muestra`, no exhaustiva) de nodos ya revelados, y sólo si el mejor candidato de esa muestra es
     ESTRICTAMENTE más cercano que su vecino actualmente más lejano, se poda ese vecino lejano y se
     reconecta con el candidato.

Por qué esto SÍ tiene memoria genuina: como la muestra de reconsideración es acotada (no exhaustiva), un
nodo colocado temprano puede quedarse con una conexión sub-óptima PARA SIEMPRE si nunca le tocó revisar
al candidato correcto en su muestra aleatoria -- el resultado final depende de en qué tanda apareció cada
nodo y qué muestras le tocaron en el camino, no sólo de la identidad final del conjunto. H=1 (un solo
lote, todos "nuevos", nadie tiene oportunidad de reconsiderar porque no hay "ya colocados" antes de la
única tanda) es el ancla de "cero historia" -- mismo rol que en el mecanismo anterior, aunque esta vez NO
se garantiza reproducir byte a byte la malla de `malla_causal_atomos` (esa función usa un desempate físico
adicional que aquí no se reimplementa) -- lo que importa es que H=1 sea el punto de comparación INTERNO
de este sweep (Jaccard de cada H contra su propio H=1), no una igualdad externa con Sistema A.

Salvaguarda anti-artefacto (mismo aviso que el piloto, cruzado con `on77-eta-lf-datos-existentes`): se
reportan SIEMPRE tres números juntos -- clustering crudo REAL/NULL, delta REAL-NULL, y Jaccard vs el
ancla H=1 -- nunca el delta aislado. Un H con delta alto pero Jaccard≈0 es ruido, no historia.

Verificación OBLIGATORIA antes de escalar a Phantom: si el Jaccard sigue siendo 1.0 en todo H (mecanismo
todavía degenerado), este script lo REPORTA y NO corre Phantom -- no tiene sentido gastar cómputo caro en
un mecanismo que no tiene la propiedad que se quiere testear.

No toca ningún archivo/carpeta congelados (`ON77_piloto_sistemaAB.py`, `p_semilla_causal.py`,
`cs072_modulos/proceso_sucesivo.py`, ningún `bateria_*`/`piloto_*`) -- sólo los IMPORTA/lee. Las corridas
de Phantom de este script (si el mecanismo pasa la verificación) viven en una carpeta NUEVA:
`/Users/alexis/phantom_cs073/ON77_sistemaB_corregido/`.

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

from cs073_cierre_holistico import _extraer_bariones, T0, _T_reloj
from cs072_modulos.piezas.p_expansion import Expansion
from cs072_modulos.piezas.p_semilla_causal import _ejes_desde_densidad, layout_resortes, barajar_aristas
from fase1_traducir_a_phantom import HFACT, POLYK
from null3_investigacion_preliminar import aristas_de
from null3_motivos_directos import contar_triangulos_y_clustering
from campo_velocidad_turbulento import factory as factory_turb, MACH_OBJETIVO

# MISMO campo de velocidad turbulento que el resto de la jerarquía (Mach=3, TURB_SEED=42) -- necesario
# para no disparar el guardián de conservación de momento angular de Phantom con v=0 exacto (ver el
# mismo problema encontrado y corregido en ON77_sistemaA_corregido.py).
CS_SONORA = POLYK ** 0.5
TURB_SEED = 42

D_CAUSAL = 3
K_CAUSAL = 4
SEED_EJES = 2000
SEED_ORDEN_HISTORIA = 777       # mismo orden de revelado que el piloto anterior, para comparabilidad
SEED_RECONSIDERACION = 555      # nueva fuente de azar: qué muestra le toca revisar a cada nodo colocado
SEED_NULL = 909
TAM_MUESTRA_RECONSIDERACION = 3  # "tamaño fijo, no todos" -- el parámetro que vuelve la reconsideración
                                  # ACOTADA (no exhaustiva); con N=200 y k=4, revisar sólo 3 candidatos
                                  # al azar (no los ~196 posibles) deja margen real para quedar atascado.

BASE = Path("/Users/alexis/phantom_cs073/ON77_sistemaB_corregido")
PHANTOMSETUP = "/Users/alexis/phantom_cs073/phantom/bin/phantomsetup_cosmogenesis_backup"
PHANTOM = "/Users/alexis/phantom_cs073/phantom/bin/phantom_cosmogenesis_backup"

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

# N fijo del sweep de Sistema B: mismo pool f=2.0 (N≈200) que usó el piloto y que usa Sistema A
# corregido -- reextraído aquí (determinista, sin semilla propia, mismo resultado, cero riesgo de
# desincronizarse de un JSON externo).
PARAMS_N_FIJO = (1200, 840, 400, 280)   # f=2.0 sobre (600,420,200,140)


# ----------------------------------------------------------------------------------------------
# Mecanismo NUEVO: reorganización de presupuesto ACOTADO (sección 9 del .md de diseño).
# ----------------------------------------------------------------------------------------------
def construir_malla_historica_acotada(dens_bar, n_batches, D=D_CAUSAL, k=K_CAUSAL, seed_ejes=SEED_EJES,
                                       seed_orden=SEED_ORDEN_HISTORIA, tam_muestra=TAM_MUESTRA_RECONSIDERACION,
                                       seed_reconsideracion=SEED_RECONSIDERACION):
    """Devuelve (adj, n, n_reorganizaciones) -- ver docstring del módulo para el mecanismo completo.
    V (el espacio de D distinciones) es ESTÁTICO, igual en cualquier H -- lo único que cambia con H es
    CUÁNTAS tandas se usan para revelar el mismo orden fijo, y por lo tanto cuántas oportunidades de
    reconsideración ACOTADA (no exhaustiva) recibe cada nodo en el camino."""
    n = len(dens_bar)
    V = _ejes_desde_densidad(dens_bar, D, seed_ejes)

    # matriz de distancias completa: trivial a N~200 (200x200), evita recalcular distancias par a par
    # en cada tanda -- las POSICIONES en V no cambian con H, sólo la visibilidad de los nodos.
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
        revelados_antes = list(revelados)   # "ya colocados" -- sólo ellos tienen derecho a reconsiderar
        revelados.extend(nuevos)

        # Paso 1 -- nodos NUEVOS compiten LIBREMENTE por sus k vecinos entre TODO lo revelado hasta ahora
        # (incluye a otros nuevos del mismo lote). Sin restricción de muestra: un nodo recién aparecido
        # necesita un punto de partida razonable, no uno artificialmente pobre.
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

        # Paso 2 -- nodos YA colocados: UNA oportunidad ACOTADA de reconsiderar (muestra chica, no todo
        # el universo revelado) por tanda. Esto es lo que introduce memoria genuina del camino: si el
        # candidato correcto nunca cae en la muestra de un nodo dado, ese nodo se queda con la conexión
        # sub-óptima de cuando fue colocado -- para siempre, no sólo "por ahora".
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


# ----------------------------------------------------------------------------------------------
# Nivel grafo: sweep barato H=[1,2,4,8,16], SIEMPRE antes de gastar Phantom.
# ----------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------
# Nivel Phantom (sólo si el sweep de grafo NO salió degenerado): traduce el adj de cada H a una IC,
# corre Phantom, mide masa en sumideros -- mismo patrón de escritura ASCII que `null3_generar_ic.py`
# (congelado, no tocado, sólo replicado el patrón porque necesitamos un adj ARBITRARIO, no el que arma
# `malla_causal_atomos`+`barajar_aristas` internamente).
# ----------------------------------------------------------------------------------------------
def escribir_ic_desde_adj(masa_bar, dens_bar, adj, ruta_salida, seed_layout=12345, iters_layout=100,
                           n_pasos_expansion=60):
    n = len(masa_bar)
    lado = float(n) ** (1.0 / 3.0)
    pos = layout_resortes(adj, n, lado=lado, iters=iters_layout, seed=seed_layout)

    expansion = Expansion(T0=T0)
    for step in range(n_pasos_expansion):
        expansion.paso_de_estiramiento(_T_reloj(step))
    a_final = expansion._a_prev
    pos = pos * a_final

    vel_gen = factory_turb(CS_SONORA, seed=TURB_SEED, mach=MACH_OBJETIVO)
    vel = vel_gen(pos, adj, dens_bar)
    h_guess = np.full(n, HFACT)
    masa_media = float(masa_bar.mean())

    with open(ruta_salida, "w") as f:
        f.write(f"# cosmogenesis_ic v2 Sistema-B-corregido (reorganizacion acotada) -- npart={n} "
                 f"masa_particula={masa_media:.17g} hfact={HFACT} polyk={POLYK:.17g}\n")
        f.write(f"{n} {masa_media:.17g} {HFACT} {POLYK:.17g}\n")
        for i in range(n):
            f.write(f"{float(pos[i,0]):.17g} {float(pos[i,1]):.17g} {float(pos[i,2]):.17g} "
                     f"{float(vel[i,0]):.17g} {float(vel[i,1]):.17g} {float(vel[i,2]):.17g} "
                     f"{float(h_guess[i]):.17g}\n")
    return dict(ruta=ruta_salida, n=n, a_final=a_final)


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


def sweep_phantom(masa_bar, dens_bar, filas_grafo):
    filas_out = []
    for fila in filas_grafo:
        hb = fila["H"]
        carpeta = BASE / f"ic_H{hb}"
        carpeta.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        escribir_ic_desde_adj(masa_bar, dens_bar, fila["adj"], str(carpeta / "cosmogenesis_ic.txt"))
        info_run = correr_phantom(carpeta)
        r_sink = masa_y_n_sumideros(carpeta)
        t_total = time.time() - t0
        fila_out = dict(H=hb, N=fila["N"], masa_total=r_sink["masa_total"],
                         n_sumideros=r_sink["n_sumideros"], t_total_s=round(t_total, 2),
                         exit_setup=info_run["exit_setup"], exit_run=info_run["exit_run"])
        filas_out.append(fila_out)
        print(f"[B-Phantom] H={hb:<3d} masa_total={r_sink['masa_total']:.2f} "
              f"n_sumideros={r_sink['n_sumideros']} t={t_total:.1f}s", flush=True)
        if info_run["exit_run"] != 0:
            tail = (carpeta / "run.log").read_text().splitlines()[-15:]
            print(f"  AVISO: exit_run != 0 en H={hb}. Tail:\n  " + "\n  ".join(tail), flush=True)
    return filas_out


def main():
    t_total = time.time()
    print("=== Sistema B corregido: N fijo, mecanismo de reorganizacion ACOTADA, H variable ===",
          flush=True)

    print("[1] extrayendo pool de bariones N fijo (f=2.0, mismo pool que Sistema A corregido)...",
          flush=True)
    masa_bar, dens_bar, obs = _extraer_bariones(*PARAMS_N_FIJO, 150, 1.5)
    n = len(masa_bar)
    print(f"    N={n}", flush=True)

    print("\n[2] sweep de GRAFO (barato, obligatorio ANTES de Phantom) H=[1,2,4,8,16]...", flush=True)
    filas_grafo = sweep_grafo(dens_bar, n_batches_list=[1, 2, 4, 8, 16])

    jaccards = [f["jaccard_vs_H1"] for f in filas_grafo if f["H"] != 1]
    degenerado = all(abs(j - 1.0) < 1e-9 for j in jaccards)

    filas_grafo_json = [{k: v for k, v in f.items() if k != "adj"} for f in filas_grafo]

    if degenerado:
        print("\n[VEREDICTO DE VERIFICACIÓN] Jaccard SIGUE siendo 1.0000 en TODO H -- el mecanismo de "
              "reorganización acotada, tal como está implementado (tam_muestra="
              f"{TAM_MUESTRA_RECONSIDERACION}), TODAVÍA es degenerado. NO se escala a Phantom (no tiene "
              "sentido gastar cómputo caro en un mecanismo que no tiene la propiedad que se quiere "
              "testear). Se reporta el problema tal cual, sin forzar la corrida.", flush=True)
        resultado = dict(sistema_b_grafo=filas_grafo_json, sistema_b_phantom=None,
                          degenerado=True, tiempo_total_s=round(time.time() - t_total, 1))
    else:
        print(f"\n[VERIFICACIÓN OK] Jaccard vs H=1: {[round(j,4) for j in jaccards]} -- NO todo 1.0, "
              "el mecanismo tiene memoria genuina del camino. Se procede a Phantom.", flush=True)
        print("\n[3] traduciendo cada H a IC de Phantom y corriendo...", flush=True)
        filas_phantom = sweep_phantom(masa_bar, dens_bar, filas_grafo)
        resultado = dict(sistema_b_grafo=filas_grafo_json, sistema_b_phantom=filas_phantom,
                          degenerado=False, tiempo_total_s=round(time.time() - t_total, 1))

    with open(BASE / "ON77_sistemaB_corregido_resultado.json", "w") as fp:
        json.dump(resultado, fp, indent=2)
    print(f"\n[TOTAL] {resultado['tiempo_total_s']}s -> "
          f"{BASE / 'ON77_sistemaB_corregido_resultado.json'}", flush=True)
    return resultado


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
