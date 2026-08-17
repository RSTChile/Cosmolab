"""
ON77_piloto_sistemaAB.py — PILOTO chico del diseño Sistema A vs Sistema B (nodo teórico O-N7.7,
régimen de escala: acumulación adaptativa vs condensación exaptativa). Ver protocolo completo en
`ON77_diseno_falsacion_sistemaAB_CS.md` (mismo directorio) — este script es SOLO el piloto barato
(Paso 1, nivel grafo, sin Phantom), no el experimento completo.

Qué hace, en simple: construye dos "brazos" del mismo sustrato (la malla causal de CS073) que
difieren en UNA sola cosa cada uno.

  Sistema A ("más recursos, misma regla"): más partículas N, la malla causal se arma en UN SOLO
  pase de vecinos-más-cercanos (kNN, `malla_causal_atomos`, función YA VALIDADA de
  `p_semilla_causal.py`, importada tal cual). Es la receta más simple posible: "junta todo y conectá
  cada nodo con sus k vecinos" — nunca revisa ni poda nada de lo que ya conectó.

  Sistema B ("misma cantidad de recursos, más historia"): N FIJO, pero la malla se arma en H pasos
  incrementales — en cada paso se revela un pedazo más del universo y se RECALCULA desde cero el
  kNN sobre lo revelado hasta ahora (`_malla_causal`, la MISMA función de vecinos-más-cercanos que
  usa Sistema A, importada de `cs072_modulos.proceso_sucesivo`, NO reescrita). Recalcular desde cero
  cada vez significa que una conexión hecha en el paso 1 puede desaparecer en el paso 3 si aparece un
  vecino mejor — eso es la poda/reconexión: "historia" operacionalizada como oportunidades de
  revisar lo ya construido, no como más material.

  H=1 (un solo paso, todo revelado de una vez) es matemáticamente IDÉNTICO a la regla de Sistema A
  — es el punto de anclaje donde A y B coinciden por construcción (control de consistencia interno).

Proxy de estructura (Ω_proc/LF), por qué éste y no masa-en-sumideros real: correr Phantom para cada
punto del sweep excede el presupuesto de un piloto (~35-100s SOLO para extraer un pool de bariones a
N~200-400, ver tabla de tiempos en el .md). `NULL3_robustecido_motivos_dosis_CS.md` (ya en disco)
validó que el conteo de triángulos/clustering del grafo causal es un proxy barato (segundos, no
minutos) que se mueve en el mismo sentido que la masa en sumideros real (REAL con más motivos forma
más masa que NULL-3 con motivos degradados, que a su vez forma más que el swap sin filtro con
motivos casi nulos) — se reusa aquí ese mismo proxy (`contar_triangulos_y_clustering`, importado de
`null3_motivos_directos.py`, NO reescrito) como sustituto barato de LF para el piloto. La escalada al
proxy real (masa en sumideros vía Phantom) es el Paso 2 del protocolo completo, pendiente de
autorización de Alexis.

Salvaguarda anti-artefacto (aviso cruzado del agente `on77-eta-lf-datos-existentes`, misma sesión):
un denominador de estructura casi-cero puede venir de dos causas MUY distintas — (a) poda histórica
genuina, o (b) simple ausencia de relación con el sustrato real (ruido). Un ratio ingenuo no las
distingue (su hallazgo: el grafo random independiente de la jerarquía CS073 da el η MÁS ALTO de los
5 sistemas comparados, precisamente por (b), no por (a)). Este piloto reporta el proxy CRUDO y el
delta REAL−NULL en cada punto (nunca sólo el ratio), y agrega Jaccard de aristas contra el ancla H=1
como diagnóstico de si la reorganización con historia es genuina (Jaccard intermedio + clustering
sube = firma esperada de "historia") o degeneración por ruido (Jaccard≈0 + clustering baja, igual
que el caso random-independiente ya medido: 12/4945 aristas ≈0.24% de solape con REAL, ver
`TEST_layout_vs_identidad_grafo_CS.md`).

NULL de cada punto: `barajar_aristas` (double-edge-swap Maslov-Sneppen, `p_semilla_causal.py`,
importado sin modificar) sobre EL MISMO grafo que se acaba de construir en ese punto — preserva
grado exacto, destruye topología específica. Mismo patrón anti-Shannon que toda la jerarquía CS073.

No toca ningún archivo congelado ni ninguna carpeta de batería/piloto existente -- sólo importa/lee.
No corre Phantom. No declara cierre ni veredicto sobre O-N7.7 ni CS073 -- sólo reporta números.
"""
import collections
import json
import time

import numpy as np

from cs072_modulos.proceso_sucesivo import _malla_causal
from cs072_modulos.piezas.p_semilla_causal import (
    _ejes_desde_densidad, malla_causal_atomos, barajar_aristas,
)
from cs073_cierre_holistico import _extraer_bariones
from null3_investigacion_preliminar import aristas_de
from null3_motivos_directos import contar_triangulos_y_clustering

D_CAUSAL = 3
K_CAUSAL = 4
SEED_EJES = 2000          # MISMO default que malla_causal_atomos / traducir_pool (consistencia)
SEED_ORDEN_HISTORIA = 777  # orden de revelado fijo para Sistema B (ver construir_malla_historica)
SEED_NULL = 909


# ----------------------------------------------------------------------------------------------
# Sistema B: la única pieza NUEVA de este script -- todo lo demás es orquestación de piezas ya
# validadas (_malla_causal, _ejes_desde_densidad, barajar_aristas, contar_triangulos_y_clustering).
# ----------------------------------------------------------------------------------------------
def construir_malla_historica(dens_bar, n_batches, D=D_CAUSAL, k=K_CAUSAL, seed_ejes=SEED_EJES,
                               seed_orden=SEED_ORDEN_HISTORIA):
    """Sistema B: mismo universo de N partículas, revelado en `n_batches` tandas ACUMULATIVAS (orden
    fijo, `seed_orden`, el MISMO orden para cualquier n_batches -- lo único que cambia entre puntos
    del sweep es en cuántos cortes se trocea ese mismo orden, no el orden en sí: control limpio).

    En cada tanda se recalcula `_malla_causal` DESDE CERO sobre todo lo revelado hasta ahora (mismos
    D/k/seed_ejes que Sistema A) y se SOBRESCRIBEN las conexiones de esos nodos -- así un nodo que
    quedó ligado a un vecino en la tanda 1 puede perder esa arista en la tanda 3 si aparece un vecino
    más cercano: poda (se cae la vieja) + reconexión (entra la nueva), no acumulación pura.

    n_batches=1 (todo revelado de una vez, sin historia) reconstruye EXACTAMENTE el mismo grafo que
    `malla_causal_atomos` -- ancla de consistencia entre Sistema A y Sistema B."""
    n = len(dens_bar)
    V = _ejes_desde_densidad(dens_bar, D, seed_ejes)  # MISMOS ejes que ve Sistema A, no cambian con H
    orden = np.random.default_rng(seed_orden).permutation(n)
    cortes = np.array_split(orden, min(n_batches, n))

    adj = collections.defaultdict(set)
    revelados = []
    for batch in cortes:
        revelados.extend(int(x) for x in batch)
        idx_rev = np.array(revelados)
        m_sub = len(idx_rev)
        adj_sub, _m, _arr = _malla_causal(V[idx_rev], min(k, max(m_sub - 1, 1)))
        for gi in idx_rev:            # poda: se resetean las conexiones de TODO lo revelado hasta
            adj[int(gi)] = set()      # ahora antes de reconstruirlas con la info nueva de esta tanda
        for local_i, vecinos_local in adj_sub.items():
            gi = int(idx_rev[local_i])
            for local_j in vecinos_local:
                gj = int(idx_rev[local_j])
                adj[gi].add(gj)
                adj[gj].add(gi)
    return dict(adj), n


def _jaccard_aristas(adj_a, adj_b, n):
    ea = set(aristas_de(adj_a, n))
    eb = set(aristas_de(adj_b, n))
    if not ea and not eb:
        return 1.0
    return len(ea & eb) / len(ea | eb)


def _real_y_null(adj, n, seed_null=SEED_NULL):
    """Estructura cruda REAL + su control NULL (double-edge-swap, mismo grado exacto) -- devuelve
    ambos dicts de contar_triangulos_y_clustering + el delta REAL-NULL en clustering_global (la
    métrica que reporta el sweep: nunca el ratio crudo solo, por el aviso de denominador casi-cero)."""
    r_real = contar_triangulos_y_clustering(adj, n)
    adj_null = barajar_aristas(adj, n, seed=seed_null)
    r_null = contar_triangulos_y_clustering(adj_null, n)
    delta = r_real["clustering_global"] - r_null["clustering_global"]
    return r_real, r_null, delta


# ----------------------------------------------------------------------------------------------
# Sistema A: N variable, regla fija (H=1 siempre) -- reusa malla_causal_atomos tal cual.
# ----------------------------------------------------------------------------------------------
def sweep_sistema_a(params_por_f, pasos_basal=150, amp_rugosidad=1.5):
    """Devuelve (filas, pools) -- `pools` guarda dens_bar por f para que Sistema B pueda REUSAR el
    mismo pool ya extraído (el motor basal es determinista, sin semilla, así que reextraerlo sólo
    gastaría cómputo de más sin cambiar el resultado -- se evita aquí)."""
    filas = []
    pools = {}
    for f, params in params_por_f.items():
        t0 = time.time()
        masa_bar, dens_bar, obs = _extraer_bariones(*params, pasos_basal, amp_rugosidad)
        pools[f] = dens_bar
        n = len(masa_bar)
        adj, _m = malla_causal_atomos(dens_bar, D=D_CAUSAL, k=K_CAUSAL, seed_ejes=SEED_EJES)
        r_real, r_null, delta = _real_y_null(adj, n)
        ganancia_marginal = delta / n
        t = time.time() - t0
        fila = dict(f=f, N=n, clustering_real=r_real["clustering_global"],
                    clustering_null=r_null["clustering_global"], delta=delta,
                    ganancia_marginal_por_recurso=ganancia_marginal, tiempo_s=round(t, 2),
                    n_triangulos_real=r_real["n_triangulos"], n_triangulos_null=r_null["n_triangulos"])
        filas.append(fila)
        print(f"[A] f={f} N={n} C_real={r_real['clustering_global']:.5f} "
              f"C_null={r_null['clustering_global']:.5f} delta={delta:+.5f} "
              f"ganancia_marginal={ganancia_marginal:+.6f} t={t:.1f}s", flush=True)
    return filas, pools


# ----------------------------------------------------------------------------------------------
# Sistema B: N fijo (un solo pool reusado), H variable -- usa construir_malla_historica.
# ----------------------------------------------------------------------------------------------
def sweep_sistema_b(dens_bar, n_batches_list):
    n = len(dens_bar)
    adj_h1, _ = construir_malla_historica(dens_bar, n_batches=1)
    filas = []
    for hb in n_batches_list:
        t0 = time.time()
        adj_h, _n = construir_malla_historica(dens_bar, n_batches=hb)
        r_real, r_null, delta = _real_y_null(adj_h, n)
        jac = _jaccard_aristas(adj_h, adj_h1, n)
        t = time.time() - t0
        fila = dict(H=hb, N=n, clustering_real=r_real["clustering_global"],
                    clustering_null=r_null["clustering_global"], delta=delta,
                    eta_LF_proxy=delta, jaccard_vs_H1=jac, tiempo_s=round(t, 2),
                    n_triangulos_real=r_real["n_triangulos"], n_triangulos_null=r_null["n_triangulos"])
        filas.append(fila)
        print(f"[B] H={hb} N={n} C_real={r_real['clustering_global']:.5f} "
              f"C_null={r_null['clustering_global']:.5f} delta={delta:+.5f} "
              f"jaccard_vs_H1={jac:.4f} t={t:.1f}s", flush=True)
    return filas


def main():
    t_total = time.time()
    # Sistema A: 4 escalas de N, extraídas una sola vez cada una (ya cronometrado en el diseño:
    # ~1.4s/4.7s/23.8s/98.2s -- total ~128s, dentro del presupuesto de piloto).
    params_a = {
        0.5: (300, 210, 100, 70),
        1.0: (600, 420, 200, 140),
        2.0: (1200, 840, 400, 280),
        4.0: (2400, 1680, 800, 560),
    }
    print("=== Sistema A: N variable, regla fija (H=1) ===", flush=True)
    filas_a, pools_a = sweep_sistema_a(params_a)

    # Sistema B: reusa el pool f=2.0 (N≈200) YA EXTRAÍDO arriba -- mismo dens_bar, cero cómputo extra
    # del motor basal (determinista, sin semilla -- reextraerlo daría exactamente el mismo array).
    print("\n=== Sistema B: N fijo (pool f=2.0), historia H variable ===", flush=True)
    dens_b = pools_a[2.0]
    filas_b = sweep_sistema_b(dens_b, n_batches_list=[1, 2, 4, 8])

    resultado = dict(sistema_a=filas_a, sistema_b=filas_b, tiempo_total_s=round(time.time() - t_total, 1))
    with open("ON77_piloto_resultado.json", "w") as fp:
        json.dump(resultado, fp, indent=2)
    print(f"\n[TOTAL] {resultado['tiempo_total_s']}s -> ON77_piloto_resultado.json", flush=True)


if __name__ == "__main__":
    main()
