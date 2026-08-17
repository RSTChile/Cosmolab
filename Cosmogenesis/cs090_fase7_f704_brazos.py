"""
cs090_fase7_f704_brazos.py — FASE VII, tarea F7-04: ¿IMPORTA *CUÁLES* ARISTAS SE PIERDEN,
O SÓLO *CUÁNTAS*?
==========================================================================================

QUÉ PREGUNTA CONTESTA
---------------------
La Fase VI dejó esta tensión (INFORME_EQUIPO_FASE6_11ago2026_CS.md, Partes 3 y 3.bis): casi todo el
"efecto de clase" que veníamos midiendo era **densidad de aristas** — descontándola, la ventaja de
clase es indistinguible de cero (p=0.22 / p=0.41). Lo único que sobrevivió fue un **residual chico
(~5%, O3-B)** ligado al **clustering** (ρ=0.77) y no a la pendiente (ρ=0.04). Y
`FASE6_O3E_memoria_vs_sinmemoria_CS.md` dejó explícitamente pendiente el control que aísla eso:
*"podar las mismas ~80 aristas al azar (separa 'elegir bien' de 'cortar cualquier cosa')"*.

Este archivo construye ese control, pero llevado al límite y al lugar donde el mecanismo C2 realmente
decide: **el mismo grafo inicial, el mismo déficit exacto de aristas, tres (cuatro) criterios distintos
para elegir cuáles se van.**

Analogía simple: a un pueblo ya construido hay que cerrarle exactamente 1.000 calles. Un brazo cierra
las que el urbanista habría cerrado (las más "caras"). Otro cierra 1.000 calles sacadas de un sombrero.
Otro cierra justo las 1.000 que el urbanista habría defendido primero. Al final los tres pueblos tienen
**exactamente el mismo número de calles**. Si el primero sigue juntando más arena que los otros dos,
entonces importa *cuáles* calles se pierden. Si los tres empatan, sólo importaba *cuántas*.

DÓNDE SE BIFURCA (el punto exacto del fork)
--------------------------------------------
`cs090_fase5_motor.dinamica_B0` (CONGELADO, sólo se importa) corre 14 sweeps de dinámica sobre el grafo
A2, con recableo co-emergente cada 3 pasos y `_enforce_kcap` cada 4 pasos, y **al final** aplica UNA
poda global: `_costo_y_podar(edges, flip_count, E_estado, K, triangles)`, que conserva las aristas con
costo ≤ percentil 70. Esa poda final es el único punto donde C2 elige, de una sola vez, un ~30% de las
aristas para tirar.

Este archivo **re-ejecuta esa misma cadena paso por paso** (`reproducir_dinamica_hasta_pre_poda`),
llamando a las MISMAS funciones del motor congelado, en el MISMO orden, con el MISMO generador
aleatorio — pero se detiene **justo antes** de la poda final, y expone `edges`, `flip_count`,
`E_estado` y `triangles`. Desde ahí se abren los brazos. Nada del motor se modifica ni se copia: el
único código replicado es el bucle de orquestación (14 líneas), y se **verifica numéricamente**, grafo
por grafo, que el brazo C2 reproduce **exactamente el mismo conjunto de aristas** que devuelve
`MOT.dinamica_B0` corrido de punta a punta (`verificar_c2_reproduce_motor`).

LOS BRAZOS (todos parten del MISMO grafo pre-poda y quitan el MISMO número de aristas, n_cut)
-----------------------------------------------------------------------------------------------
  1. `c2`          — quita exactamente lo que C2 quita: las aristas de costo > percentil 70.
  2. `azar`        — quita n_cut aristas al azar (uniforme, sin reemplazo).
  3. `anticosto`   — quita las n_cut de **menor** costo: literalmente las que C2 conservó *primero*.
                     Es la inversión del ranking que C2 usa **en la poda final** (costo = 0.5·z(flips
                     históricos) + 0.5·z(conflicto de holonomía)).
  4. `soporte`     — quita las n_cut de **menor soporte local** (nº de vecinos compartidos por sus dos
                     extremos). Es el criterio que C2 usa en su OTRO mecanismo, `_enforce_kcap` (línea
                     144 del motor: ante exceso de grado "se quedan los kcap de mayor soporte local"),
                     aplicado como poda: conserva TODAS las aristas que cierran triángulo.
  5. `antisoporte` — quita las n_cut de **mayor soporte local**: la inversión de (4). Destruye todos
                     los triángulos del grafo.

**Por qué DOS ejes de criterio y no uno.** La consigna pedía justificar cómo se define "opuesto". C2 no
tiene un único criterio: tiene dos, y actúan en momentos distintos. `_costo_y_podar` decide la poda
final por *costo* (historia de parpadeo + conflicto de holonomía); `_enforce_kcap` decide, durante la
dinámica, por *soporte local* (triángulos). Invertir sólo el costo dejaría intacto el criterio que la
Fase VI señaló como el correlato del residual — el clustering vive en el soporte local (O3-B: ρ=0.77
con clustering, ρ=0.04 con la pendiente). Se hacen los dos ejes, y el eje de soporte se hace **con su
par positivo** (`soporte`) para que `antisoporte` tenga contra qué compararse en el mismo eje: el par
`soporte` ↔ `antisoporte` es una manipulación de clustering **pura**, con el mismo N, el mismo nº de
aristas y el mismo grafo de partida — exactamente el control que O3-B dejó pendiente ("grados **y
triángulos** fijos: aislaría si el clustering es mecanismo o correlato").

DEGENERACIÓN DE LOS DOS CRITERIOS (medida, no supuesta — condiciona la lectura)
--------------------------------------------------------------------------------
Los dos rankings de C2 resultan estar MUY empatados en estos grafos, y el archivo lo mide y lo informa
por grafo en vez de esconderlo:
  · el **costo** toma sólo ~14-40 valores distintos sobre ~3.000-4.200 aristas, con el 95-97% de las
    aristas en un único valor modal. Por eso la poda final de C2 (`costo <= percentil 70`, con `<=` y
    empates masivos) **no corta el 30% que sugiere el nombre P70: corta 2,3-5,7%** (83-237 aristas).
  · el **soporte local** es prácticamente binario (0 ó 1): con kcap 4-7 a N=2000 el grafo es casi sin
    triángulos, sólo 33-108 aristas cierran alguno.
Consecuencia: cuando el ranking no alcanza a llenar el cupo n_cut, el resto se completa **al azar
dentro del grupo empatado** (no por índice de nodo, que introduciría un sesgo sistemático hacia los
nodos de índice bajo). Cada brazo informa cuántas de sus n_cut aristas salieron del ranking estricto y
cuántas del relleno de empate (`n_estrictas_*` / `n_relleno_empate_*`), para que la potencia real de
cada contraste quede a la vista.

Los tres brazos rankeados se aplican en el MISMO punto (la poda final) para que el número de aristas
quede exactamente igual: intervenir `_enforce_kcap` dentro del bucle cambiaría la dinámica aguas abajo
y con ella el número final de aristas — se perdería justamente la variable que este experimento fija.

QUÉ SE VERIFICA NUMÉRICAMENTE (no se asume nada de esto)
---------------------------------------------------------
  a) el brazo `c2` reproduce, arista por arista, el grafo final de `MOT.dinamica_B0` (conjunto igual);
  b) la réplica local del vector de costos reproduce el `conservar` de `MOT._costo_y_podar`;
  c) los cuatro brazos terminan con **el mismo número exacto de aristas** (se compara e informa el
     número de los cuatro, no se asume — ése es el punto entero del experimento);
  d) el nº de aristas del brazo `c2` coincide con `n_aristas_grafo_final` del `meta_regla.json` que
     Fase V-B escribió para esa misma regla (verificación cruzada obligatoria contra disco);
  e) el `meta_regla.json` de Fase V-B tiene el mismo `rule_id` y el mismo `seed` que estamos usando.

TRATO IDÉNTICO ENTRE BRAZOS
---------------------------
Los cuatro grafos se **canonicalizan** (`canonicalizar` de `cs090_fase6_o3b_rewiring`, reusada tal
cual) antes de medir y antes de escribir la condición inicial: el orden de iteración de un `set` de
Python depende de cómo se llenó, y `cs080.cajas_bfs` recorre `for v in adj[u]`, así que dos grafos con
las mismas aristas pero armadas en otro orden pueden dar particiones distintas (verificado en O3-B). Se
mide, aparte, la pendiente del brazo `c2` con el objeto nativo del motor, para poder comparar contra la
historia.

QUÉ NO HACE
-----------
No corre Phantom (eso es `cs090_fase7_f704_correr.py`). No modifica ningún archivo existente: importa
`cs090_fase5_generador`, `cs090_fase5_motor`, `cs090_fase5_clasificador`, `cs080_renormalizacion`,
`cg003_diagnostico_gromov`, `cs090_diam_corregido`, `cs090_fase5b_phantom_adaptador` y
`cs090_fase6_o3b_rewiring` — todos sólo se importan. No declara cierre ni veredicto: sólo fabrica los
brazos, los mide y escribe números.

USO
---
    python3.9 cs090_fase7_f704_brazos.py --seleccion            # sólo imprime/escribe la selección
    python3.9 cs090_fase7_f704_brazos.py 0 --sufijo=_p          # piloto: grafo 0, cronometrado
    python3.9 cs090_fase7_f704_brazos.py 0,1,2 --sufijo=_shard0 # un shard (para correr en paralelo)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase5_generador as GEN                      # sólo import
import cs090_fase5_motor as MOT                          # sólo import (motor CONGELADO)
import cs082_fase4_4sustratos as C82                     # sólo import (_circ_mean_update)
from cs090_fase5b_phantom_adaptador import generar_ic_masa_fija_desde_grafo   # sólo import
# piezas de medición ya validadas en O3-B — se REUSAN tal cual, no se reimplementan
from cs090_fase6_o3b_rewiring import (
    canonicalizar, aristas_set, grados, clustering, tam_gigante,
    nulls_topo_de, pendiente_corregida_de_grafo, lote_de_seed,
    N_NODOS, N_SWEEPS, SEED_LAYOUT, ESCALAS_B,
)

# ---------------------------------------------------------------------------------------------
# Parámetros — heredados de la línea; los únicos nuevos son los del sorteo de brazos
# ---------------------------------------------------------------------------------------------
BRAZOS = ("c2", "azar", "anticosto", "soporte", "antisoporte")
PERCENTIL_PODA = 70.0        # el MISMO P=70 que usa MOT._costo_y_podar por defecto

MULT_SEED_AZAR = 9704        # multiplicador NUEVO para la semilla del brazo `azar`. Los ya usados en
                             # la línea son 1000/2000/3000/5000/6000/7000/7500/8000/9100(O3-B):
                             # 9704 no colisiona con ninguno.

BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_fase7_f704_cortar_bien"
SUFIJO_CARPETA = "f704"      # prefijo/sufijo NUEVO (lección del bug de colisión de nombres de Fase V-B)

RUTA_TOTAL_40 = f"{HERE}/cs090_fase5b_TOTAL_40pares.csv"
RUTA_REMEDICION = f"{HERE}/cs090_fase6_remedicion_430.csv"
RUTA_SELECCION = f"{HERE}/cs090_fase7_f704_seleccion.csv"
RUTA_ESTRUCTURA = f"{HERE}/cs090_fase7_f704_estructura.csv"

BATERIAS_FASE5B = [
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto",
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v2",
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v3",
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v4",
]


# =============================================================================================
# 1) SELECCIÓN de los grafos base
# =============================================================================================
def seleccionar_grafos(n_por_lote: int = 3):
    """Universo: las reglas de `cs090_fase5b_TOTAL_40pares.csv` (las que YA pasaron por Phantom en
    Fase V-B, así que su carpeta y su `meta_regla.json` existen en disco para la verificación cruzada),
    unidas con la pendiente CORREGIDA de `cs090_fase6_remedicion_430.csv`.

    Unión por `(rule_id, seed)`, NUNCA por `rule_id` solo: los lotes v1 y v2 comparten el patrón de
    nombre `A2-B0-C2-r{idx}` y hay reglas DISTINTAS con el mismo id (bug documentado en O3-B).

    Criterio: dentro de cada uno de los 4 lotes de `seed_base` (271828/371828/471828/571828) se toman
    la de pendiente corregida MÁXIMA, la MEDIANA y la MÍNIMA => 3 por lote, 12 grafos. Se reparte por
    lote (independencia experimental, lección de O2-F) y se abre en abanico dentro del lote a propósito:
    el endpoint de esta tarea es continuo (pendiente y clustering), así que interesa cubrir rango, no
    concentrarse en un extremo — y "% Clase III" NO se usa como endpoint en ningún punto de este archivo.
    """
    with open(RUTA_TOTAL_40) as f:
        total = list(csv.DictReader(f))
    with open(RUTA_REMEDICION) as f:
        remed = {(r["rule_id"], int(r["seed"])): r for r in csv.DictReader(f)}

    vistos, filas = set(), []
    for r in total:
        clave = (r["rule_id"], int(r["seed"]))
        if clave in vistos:
            continue
        vistos.add(clave)
        m = remed.get(clave)
        if m is None:
            print(f"   [selección] {r['rule_id']} (seed {r['seed']}) no está en la remedición "
                  f"corregida -- queda fuera", flush=True)
            continue
        filas.append(dict(
            rule_id=r["rule_id"], seed=int(r["seed"]), lote=lote_de_seed(r["seed"]),
            rol_fase5b=r["rol"], K=int(r["K"]), kcap=int(r["kcap"]),
            pendiente_corregida=float(m["pendiente_corregida"]),
            frac_masa_fase5b=float(r["fraccion_masa_en_sumideros"]),
            carpeta_fase5b=r["carpeta"],
        ))

    por_lote = defaultdict(list)
    for f in filas:
        por_lote[f["lote"]].append(f)

    elegidos = []
    for lote in sorted(por_lote):
        grupo = sorted(por_lote[lote], key=lambda d: d["pendiente_corregida"])
        if len(grupo) <= n_por_lote:
            sel = grupo
        else:
            idxs = sorted({0, len(grupo) // 2, len(grupo) - 1})
            sel = [grupo[i] for i in idxs]
        for s in sel:
            s["motivo_seleccion"] = ("min" if s is sel[0] else
                                     "max" if s is sel[-1] else "mediana") + f"_pendiente_lote_{lote}"
        elegidos.extend(sel)
        print(f"   [selección] lote {lote}: {len(grupo)} disponibles -> "
              f"{[(g['rule_id'], round(g['pendiente_corregida'], 3)) for g in sel]}", flush=True)

    with open(RUTA_SELECCION, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(elegidos[0].keys()))
        w.writeheader()
        w.writerows(elegidos)
    print(f"   [selección] {len(elegidos)} grafos -> {RUTA_SELECCION.split('/')[-1]}", flush=True)
    return elegidos


# =============================================================================================
# 2) FORK: re-ejecutar la dinámica del motor y detenerse JUSTO ANTES de la poda final
# =============================================================================================
def reproducir_dinamica_hasta_pre_poda(seed, N=N_NODOS, n_sweeps=N_SWEEPS):
    """Réplica del bucle de orquestación de `MOT.dinamica_B0` para el caso A2/B0/C2 (líneas 200-224 del
    motor), llamando a las MISMAS funciones del motor congelado en el MISMO orden y consumiendo el MISMO
    generador aleatorio — pero devolviendo el estado JUSTO ANTES de `_costo_y_podar`.

    Se replica el bucle, no la física: `_circ_mean_update`, `_recablear_A2`, `_enforce_kcap` y
    `_muestrear_triangulos` se llaman importadas. `verificar_c2_reproduce_motor` comprueba después, con
    una corrida independiente del motor completo, que el resultado coincide arista por arista.

    Devuelve dict(p, adj_pre, edges, flip_count, E_estado, triangles, S, N).
    """
    p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=seed)
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)

    K, J, noise = p["K"], p["J"], p["noise"]
    adj = sustrato["adj"]
    S = rng.uniform(0, K, N)
    flip_count = defaultdict(int)
    prev_edges = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])
    for step in range(n_sweeps):
        vecinos = [list(a) for a in adj]
        S = C82._circ_mean_update(S, vecinos, J, noise, rng)
        if step % 3 == 0:
            MOT._recablear_A2(adj, S, K, rng)
        if step % 4 == 0:
            MOT._enforce_kcap(adj, N, p["kcap"])
        cur = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])
        for e in prev_edges ^ cur:
            flip_count[e] += 1
        prev_edges = cur

    edges = sorted(prev_edges)
    E_estado = {(i, j): abs(S[i] - S[j]) % K for (i, j) in edges}
    triangles = MOT._muestrear_triangulos(adj, N, rng)
    return dict(p=p, adj_pre=adj, edges=edges, flip_count=flip_count, E_estado=E_estado,
                triangles=triangles, S=S, N=N)


def verificar_c2_reproduce_motor(seed, adj_c2, N=N_NODOS, n_sweeps=N_SWEEPS):
    """Corre `MOT.dinamica_B0` COMPLETO (de punta a punta, sin fork) sobre la misma regla y compara su
    grafo final, arista por arista, con el del brazo `c2`. Si difieren en una sola arista, el fork no es
    fiel y no se puede seguir. Devuelve (identico, n_aristas_motor, n_dif)."""
    p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=seed)
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)
    sustrato = MOT.dinamica_B0(sustrato, p, rng, n_sweeps, "C2")
    m = MOT.medir(sustrato, p, rng)
    E_motor = aristas_set(m["adj_final"], N)
    E_c2 = aristas_set(adj_c2, N)
    return bool(E_motor == E_c2), len(E_motor), len(E_motor ^ E_c2)


# =============================================================================================
# 3) COSTOS Y RANKINGS — réplica local del vector de costo de MOT._costo_y_podar
# =============================================================================================
def vector_de_costos(edges, flip_count, E_estado, K, triangles):
    """Réplica LITERAL del cálculo interno de `MOT._costo_y_podar` (líneas 159-178 del motor), que
    devuelve sólo el conjunto a conservar y no expone el vector. Se necesita el vector para poder
    ordenar las aristas y construir los brazos `anticosto`. La fidelidad de esta réplica NO se asume:
    `procesar_una` comprueba que el set derivado de este vector con el mismo percentil 70 es idéntico
    al `conservar` que devuelve la función del motor."""
    hist = np.array([flip_count.get(e, 0) for e in edges], dtype=float)
    hol_por_arista = defaultdict(list)
    for (i, j, k) in triangles:
        eij = E_estado.get((i, j)) if i < j else E_estado.get((j, i))
        ejk = E_estado.get((j, k)) if j < k else E_estado.get((k, j))
        eki = E_estado.get((k, i)) if k < i else E_estado.get((i, k))
        if eij is None or ejk is None or eki is None:
            continue
        h = (eij + ejk + eki) % K
        h = abs(h - K if h > K / 2 else h)
        for e in ((i, j) if i < j else (j, i), (j, k) if j < k else (k, j),
                  (i, k) if i < k else (k, i)):
            hol_por_arista[e].append(h)
    media_global = (float(np.mean([v for vs in hol_por_arista.values() for v in vs]))
                    if hol_por_arista else 0.0)
    hol = np.array([float(np.mean(hol_por_arista[e])) if e in hol_por_arista else media_global
                    for e in edges])

    def z(a):
        mu, sd = a.mean(), a.std()
        return (a - mu) / sd if sd > 1e-9 else np.zeros_like(a)

    return 0.5 * z(hist) + 0.5 * z(hol)


def soporte_local_de_aristas(adj, edges):
    """Para cada arista (i,j): nº de vecinos compartidos por i y j en el grafo PRE-PODA. Es exactamente
    la cantidad por la que `MOT._enforce_kcap` ordena para decidir qué vecinos conserva un nodo que
    excede kcap (`len(adj[i] & adj[j])`, línea 144 del motor)."""
    return np.array([len(adj[i] & adj[j]) for (i, j) in edges], dtype=float)


def construir_adj_desde_aristas(edges_conservadas, N):
    adj = [set() for _ in range(N)]
    for (i, j) in edges_conservadas:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _quitar_por_ranking(clave, n_cut, rng):
    """Quita las `n_cut` aristas de MENOR `clave` (para 'mayor', pasar -clave). Los empates NO se
    rompen por índice de arista —eso sesgaría sistemáticamente hacia los nodos de índice bajo— sino
    por una permutación aleatoria del propio brazo. Devuelve (indices_a_quitar, diagnóstico), donde el
    diagnóstico cuenta cuántas salieron del ranking estricto y cuántas del relleno de empate."""
    M = len(clave)
    jitter = rng.permutation(M)
    orden = np.lexsort((jitter, clave))          # lexsort ordena por la ÚLTIMA clave
    idx = orden[:n_cut]
    frontera = float(clave[orden[n_cut - 1]]) if n_cut > 0 else float("nan")
    n_estrictas = int((clave < frontera).sum())
    diag = dict(valor_frontera=frontera, n_estrictas=n_estrictas,
                n_relleno_empate=int(n_cut - n_estrictas),
                n_empatados_en_frontera=int((clave == frontera).sum()))
    return idx, diag


def brazos_de_corte(edges, costo, soporte, conservar_c2, seed):
    """Devuelve (conservadas_por_brazo, n_cut, indices_quitados, diagnóstico). Los CINCO brazos quitan
    exactamente el mismo número de aristas `n_cut` (= las que C2 quita en su poda final).

    Cada brazo rankeado usa su PROPIO generador aleatorio (derivado del seed de la regla con offsets
    distintos) sólo para romper empates; el brazo `azar` usa el suyo para el sorteo completo."""
    M = len(edges)
    idx_quitar_c2 = np.array([k for k, e in enumerate(edges) if e not in conservar_c2], dtype=int)
    n_cut = len(idx_quitar_c2)

    base = int(seed) * MULT_SEED_AZAR
    rng_azar = np.random.default_rng(base + 13)
    idx_azar = rng_azar.choice(M, size=n_cut, replace=False)

    idx_anticosto, d_anticosto = _quitar_por_ranking(costo, n_cut, np.random.default_rng(base + 23))
    idx_soporte, d_soporte = _quitar_por_ranking(soporte, n_cut, np.random.default_rng(base + 33))
    idx_antisoporte, d_antisop = _quitar_por_ranking(-soporte, n_cut, np.random.default_rng(base + 43))

    quitar = dict(c2=idx_quitar_c2, azar=idx_azar, anticosto=idx_anticosto,
                  soporte=idx_soporte, antisoporte=idx_antisoporte)
    diag = dict(anticosto=d_anticosto, soporte=d_soporte, antisoporte=d_antisop)
    out = {}
    for brazo, idxs in quitar.items():
        fuera = set(int(k) for k in idxs)
        out[brazo] = [e for k, e in enumerate(edges) if k not in fuera]
    return out, n_cut, {b: np.asarray(v) for b, v in quitar.items()}, diag


# =============================================================================================
# 4) VERIFICACIÓN CRUZADA contra el meta_regla.json de Fase V-B
# =============================================================================================
def resolver_carpeta_fase5b(valor: str) -> str:
    if not valor:
        return ""
    if os.path.isabs(valor):
        return valor
    for base in BATERIAS_FASE5B:
        cand = os.path.join(base, valor)
        if os.path.isdir(cand):
            return cand
    return valor


def verificar_contra_meta_fase5b(sel, n_aristas_c2):
    """Comprueba contra disco (no contra el CSV) que el `meta_regla.json` escrito por Fase V-B para esta
    regla (a) existe, (b) tiene el mismo rule_id y seed, y (c) registra el MISMO número de aristas del
    grafo final que el que produce nuestro brazo `c2`. Si (c) falla, el fork no está reproduciendo la
    historia y el par no es utilizable."""
    out = dict(meta5b_existe=False, meta5b_rule_id_ok=None, meta5b_seed_ok=None,
               meta5b_n_aristas=None, meta5b_n_aristas_coincide=None, meta5b_ruta=None)
    carpeta = resolver_carpeta_fase5b(sel.get("carpeta_fase5b") or "")
    ruta = os.path.join(carpeta, "meta_regla.json")
    out["meta5b_ruta"] = ruta
    if not os.path.exists(ruta):
        return out
    mv = json.load(open(ruta))
    out["meta5b_existe"] = True
    out["meta5b_rule_id_ok"] = bool(mv.get("rule_id") == sel["rule_id"])
    out["meta5b_seed_ok"] = bool(int(mv.get("seed", -1)) == int(sel["seed"]))
    out["meta5b_n_aristas"] = mv.get("n_aristas_grafo_final")
    out["meta5b_n_aristas_coincide"] = bool(out["meta5b_n_aristas"] == n_aristas_c2)
    return out


# =============================================================================================
# 5) PROCESAR UN GRAFO BASE: 4 brazos, medir, escribir IC
# =============================================================================================
def procesar_una(sel, generar_ic=True, verificar_motor=True, medir_pendiente=True):
    rid, seed = sel["rule_id"], int(sel["seed"])
    N = N_NODOS
    t_ini = time.time()
    cron = {}

    t0 = time.time()
    est = reproducir_dinamica_hasta_pre_poda(seed, N=N, n_sweeps=N_SWEEPS)
    cron["t_dinamica_s"] = round(time.time() - t0, 2)
    p, adj_pre, edges = est["p"], est["adj_pre"], est["edges"]
    M_pre = len(edges)

    # --- costo (réplica) y `conservar` (función del motor, tal cual) ---
    t0 = time.time()
    costo = vector_de_costos(edges, est["flip_count"], est["E_estado"], p["K"], est["triangles"])
    conservar_motor = MOT._costo_y_podar(edges, est["flip_count"], est["E_estado"], p["K"],
                                         est["triangles"], P=PERCENTIL_PODA)
    umbral = float(np.percentile(costo, PERCENTIL_PODA))
    conservar_replica = set(e for e, c in zip(edges, costo) if c <= umbral)
    replica_costo_fiel = bool(conservar_replica == conservar_motor)
    soporte = soporte_local_de_aristas(adj_pre, edges)
    cron["t_costos_s"] = round(time.time() - t0, 2)
    assert replica_costo_fiel, (f"{rid}: la réplica del vector de costos NO reproduce el `conservar` "
                                f"de MOT._costo_y_podar -- no se sigue")

    # --- los brazos ---
    conservadas, n_cut, idx_quitadas, diag_rank = brazos_de_corte(edges, costo, soporte,
                                                                  conservar_motor, seed)
    adjs_nativos = {b: construir_adj_desde_aristas(v, N) for b, v in conservadas.items()}
    adjs = {b: canonicalizar(a, N) for b, a in adjs_nativos.items()}      # MISMO trato a los cuatro

    n_aristas = {b: len(aristas_set(a, N)) for b, a in adjs.items()}
    todos_iguales = len(set(n_aristas.values())) == 1
    assert todos_iguales, f"{rid}: los brazos NO terminaron con el mismo nº de aristas: {n_aristas}"
    M_final = n_aristas["c2"]

    # --- verificación (a): el brazo c2 == el motor corrido de punta a punta ---
    if verificar_motor:
        t0 = time.time()
        c2_ok, n_motor, n_dif = verificar_c2_reproduce_motor(seed, adjs_nativos["c2"], N=N)
        cron["t_verif_motor_s"] = round(time.time() - t0, 2)
    else:
        c2_ok, n_motor, n_dif = None, None, None

    # --- solapamiento entre brazos (cuánto se parecen los conjuntos que cada uno CONSERVÓ) ---
    sets_cons = {b: set(v) for b, v in conservadas.items()}
    solapes = {}
    for b in BRAZOS:
        if b == "c2":
            continue
        solapes[f"solape_c2_{b}"] = len(sets_cons["c2"] & sets_cons[b]) / M_final if M_final else float("nan")

    # --- métricas estructurales por brazo ---
    t0 = time.time()
    metricas = {}
    for b, a in adjs.items():
        cl, tr, ntri = clustering(a, N)
        metricas[b] = dict(clustering_local=cl, transitividad=tr, n_triangulos=ntri,
                           gigante=tam_gigante(a, N), grado_medio=2.0 * n_aristas[b] / N)
    cron["t_clustering_s"] = round(time.time() - t0, 2)

    # --- pendiente corregida por brazo (misma vara NULL_topo para los 4: misma densidad exacta) ---
    if medir_pendiente:
        t0 = time.time()
        adjs_null = nulls_topo_de(seed, N, M_final)
        for b, a in adjs.items():
            pc = pendiente_corregida_de_grafo(a, N, seed, adjs_null)
            metricas[b].update(pendiente_corregida=pc["pendiente"], z_agg=pc["z_agg"],
                               z_sostenido=pc["z_sostenido"], diams=pc["diams"], n_cajas=pc["n_cajas"])
        cron["t_pendiente_s"] = round(time.time() - t0, 2)

    # --- fila de salida ---
    fila = dict(
        rule_id=rid, seed=seed, lote=sel["lote"], motivo_seleccion=sel.get("motivo_seleccion"),
        rol_fase5b=sel.get("rol_fase5b"), K=p["K"], J=p["J"], noise=p["noise"],
        meandeg=p["meandeg"], kcap=p["kcap"], n_nodos=N,
        M_pre_poda=M_pre, n_cut=n_cut, frac_cortada=n_cut / M_pre if M_pre else float("nan"),
        M_identico_en_todos=todos_iguales, M_final=M_final,
        # degeneración de los criterios (condiciona la potencia de cada contraste)
        n_valores_costo_distintos=int(len(np.unique(np.round(costo, 12)))),
        frac_costo_modal=float(np.max(np.unique(np.round(costo, 12), return_counts=True)[1]) / M_pre),
        n_aristas_soporte_ge1=int((soporte >= 1).sum()), soporte_max=float(soporte.max()),
        replica_costo_fiel=replica_costo_fiel, umbral_costo_p70=umbral,
        c2_reproduce_motor=c2_ok, n_aristas_motor=n_motor, n_aristas_dif_vs_motor=n_dif,
        seed_azar=int(seed) * MULT_SEED_AZAR + 13,
        pendiente_corregida_fase5b_ref=sel["pendiente_corregida"],
        frac_masa_fase5b=sel["frac_masa_fase5b"],
        **solapes,
    )
    for b in BRAZOS:
        fila[f"M_final_{b}"] = n_aristas[b]
        for k, v in metricas[b].items():
            fila[f"{k}_{b}"] = v
    for b, d in diag_rank.items():
        for k, v in d.items():
            fila[f"{k}_{b}"] = v
    fila.update(verificar_contra_meta_fase5b(sel, n_aristas["c2"]))
    fila.update(cron)

    # --- condiciones iniciales de Phantom ---
    if generar_ic:
        t0 = time.time()
        for b in BRAZOS:
            carpeta = f"{BASE_SALIDA}/{rid}_{SUFIJO_CARPETA}_{b}"
            os.makedirs(carpeta, exist_ok=True)
            info = generar_ic_masa_fija_desde_grafo(adjs[b], N=N, seed_layout=SEED_LAYOUT,
                                                    ruta_salida=f"{carpeta}/cosmogenesis_ic.txt")
            meta = dict(
                tarea="FASE7_F704_cortar_bien_vs_azar", brazo=b, rule_id=rid, seed=seed,
                lote=sel["lote"], N=N, seed_layout=SEED_LAYOUT,
                K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                sim_thr_frac=p["sim_thr_frac"],
                M_pre_poda=M_pre, n_cut=n_cut, n_aristas_grafo_final=n_aristas[b],
                grado_medio_grafo_final=metricas[b]["grado_medio"],
                clustering_local=metricas[b]["clustering_local"],
                transitividad=metricas[b]["transitividad"],
                pendiente_corregida=metricas[b].get("pendiente_corregida"),
                masa_total_ic=info["masa_total"], carpeta=carpeta,
                seed_azar=(fila["seed_azar"] if b == "azar" else None),
            )
            with open(f"{carpeta}/meta_regla.json", "w") as f:
                json.dump(meta, f, indent=2)
            fila[f"carpeta_{b}"] = carpeta
        cron["t_ic_s"] = round(time.time() - t0, 2)
        fila["t_ic_s"] = cron["t_ic_s"]

    fila["t_total_s"] = round(time.time() - t_ini, 2)
    return fila


def main(indices=None, generar_ic=True, verificar_motor=True, medir_pendiente=True, sufijo_csv=""):
    elegidos = seleccionar_grafos()
    if indices is not None:
        elegidos = [elegidos[i] for i in indices]
    print(f"\n[f704] procesando {len(elegidos)} grafos base x {len(BRAZOS)} brazos "
          f"(generar_ic={generar_ic}, verificar_motor={verificar_motor})", flush=True)

    filas, t0 = [], time.time()
    for k, sel in enumerate(elegidos):
        print(f"[{k+1}/{len(elegidos)}] {sel['rule_id']} (lote {sel['lote']}, "
              f"{sel.get('motivo_seleccion')})...", flush=True)
        fila = procesar_una(sel, generar_ic=generar_ic, verificar_motor=verificar_motor,
                            medir_pendiente=medir_pendiente)
        filas.append(fila)
        m_txt = ", ".join("{}:{}".format(b, fila["M_final_" + b]) for b in BRAZOS)
        print(f"    M_pre={fila['M_pre_poda']} n_cut={fila['n_cut']} M_final={{{m_txt}}} "
              f"[identico={fila['M_identico_en_todos']}] "
              f"c2==motor:{fila['c2_reproduce_motor']} meta5b_ok:{fila['meta5b_n_aristas_coincide']}",
              flush=True)
        print("    clustering " + " ".join(f"{b}={fila[f'clustering_local_{b}']:.5f}" for b in BRAZOS),
              flush=True)
        print("    pendiente  " + " ".join(f"{b}={fila.get(f'pendiente_corregida_{b}')}" for b in BRAZOS),
              flush=True)
        print(f"    {fila['t_total_s']}s  cron="
              f"{ {kk: vv for kk, vv in fila.items() if kk.startswith('t_')} }", flush=True)

    ruta = RUTA_ESTRUCTURA.replace(".csv", f"{sufijo_csv}.csv")
    campos = []
    for f in filas:
        for c in f:
            if c not in campos:
                campos.append(c)
    with open(ruta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"\n[f704] {len(filas)} filas -> {ruta.split('/')[-1]}  (total {time.time()-t0:.0f}s)")
    return filas


if __name__ == "__main__":
    idxs, gen_ic, verif, pend, suf = None, True, True, True, ""
    for arg in sys.argv[1:]:
        if arg == "--seleccion":
            seleccionar_grafos()
            sys.exit(0)
        elif arg == "--sin-ic":
            gen_ic = False
        elif arg == "--sin-verif-motor":
            verif = False
        elif arg == "--sin-pendiente":
            pend = False
        elif arg.startswith("--sufijo="):
            suf = arg.split("=", 1)[1]
        else:
            idxs = [int(x) for x in arg.split(",")]
    main(indices=idxs, generar_ic=gen_ic, verificar_motor=verif, medir_pendiente=pend, sufijo_csv=suf)
