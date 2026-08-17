"""
cs090_fase7_f702_escalera.py — FASE VII, tarea F7-02: ESCALERA DE CLUSTERING
=================================================================================================

QUÉ PREGUNTA CONTESTA
---------------------
En `FASE6_O3B_control_rewiring_CS.md` se vio algo llamativo: cuando a un grafo Clase III se le baraja
todo el cableado conservando la secuencia de grados (double-edge-swap), el original conserva una
ventaja de ~+5% en masa acretada (9/12, Wilcoxon p=0.0103), y **lo que predice cuál gana no es la
pendiente (ρ=0.04) sino el CLUSTERING** (ρ=0.77, p=0.0034).

Pero eso es una CORRELACIÓN sobre 12 puntos: se observó que covarían. Esta tarea intenta convertirla en
una INTERVENCIÓN: se agarra el clustering como si fuera una perilla, se lo sube y se lo baja a mano
dejando TODO lo demás fijo (mismo N, mismo nº de aristas, misma secuencia de grados nodo por nodo), y se
mira si la masa acretada por Phantom sigue o no a la perilla.

Analogía simple: hasta ahora habíamos notado que las maquetas con más "triangulitos" juntan más arena.
Podría ser que los triangulitos causen algo, o podría ser que las maquetas con triangulitos tengan
también otra cosa que es la que importa. Acá le sacamos y le ponemos triangulitos a la MISMA maqueta —
misma gente, cada uno con la misma cantidad exacta de amigos que antes— y vemos si la arena responde.

CÓMO SE MUEVE LA PERILLA SIN TOCAR NADA MÁS
--------------------------------------------
Toda la escalera se construye con la MISMA operación elemental que usa NULL-3 y el control O3-B: el
`double-edge-swap` de Maslov-Sneppen, que toma dos aristas (a-b) y (c-d) y las reconecta como (a-d) y
(c-b). Cada nodo pierde un vecino y gana otro => **su grado no cambia jamás**. Lo único que cambia es
con quién está conectado. Repitiendo swaps se puede llevar el grafo a donde uno quiera dentro del
espacio de grafos con esa secuencia de grados exacta.

La diferencia con O3-B es que allá los swaps eran CIEGOS (se aceptaban todos) y acá son DIRIGIDOS:

  - **bajar clustering**: se proponen swaps al azar y se aceptan sólo los que destruyen triángulos
    (Δtriángulos < 0), más los neutros con probabilidad 1/2 para que el grafo igual se mezcle bien.
  - **subir clustering**: en vez de proponer al azar (casi nunca cerraría un triángulo), se propone
    dirigido: se elige un nodo u con al menos 2 vecinos, se eligen dos vecinos suyos x,y que NO estén
    conectados entre sí, y se hace el swap (x-p),(y-q) -> (x-y),(p-q). La arista nueva x-y cierra el
    triángulo u-x-y. Se acepta si el balance neto de triángulos es positivo.

La escalera se arma en UN SOLO recorrido desde un ancestro común, para que todos los escalones sean
comparables entre sí:

    original --(swaps que destruyen triángulos)--> E0 (piso)  --(swaps que crean triángulos)-->  ...
    ... snapshot cuando el nº de triángulos cruza cada objetivo --> E1, E2, E3, E4 (techo)

Los objetivos son fracciones del máximo REALMENTE alcanzado por el proceso de subida (no un número
inventado de antemano): FRACCIONES_OBJETIVO. Por eso el script **reporta el clustering efectivamente
logrado en cada escalón** — la escalera real, no la pretendida.

Se verifica NUMÉRICAMENTE con `np.array_equal`, escalón por escalón contra el original, que la
secuencia de grados de los 2000 nodos sea idéntica (no se asume: si falla, aborta ese grafo).

QUÉ MIDE EN CADA ESCALÓN
-------------------------
clustering local medio (Watts-Strogatz), transitividad global, nº de triángulos, solapamiento de
aristas contra el original, tamaño de la componente gigante, y la **pendiente corregida**
log(diám)-vs-log(N_cajas) con la medición OFICIAL de diámetro (`cs090_diam_corregido.diam_gigante`,
regla vigente) — esta última para poder distinguir si el clustering mueve la masa POR la geometría
(entonces la pendiente debería moverse también) o por otra vía.

QUÉ REUSA Y QUÉ NO TOCA
------------------------
Sólo importa, nunca modifica: `cs090_fase6_o3b_rewiring` (funciones de medición ya validadas:
`clustering`, `grados`, `aristas_set`, `canonicalizar`, `tam_gigante`, `nulls_topo_de`,
`pendiente_corregida_de_grafo`, `lote_de_seed`), `cs090_fase5b_phantom_adaptador`
(`reconstruir_regla_a2b0c2`, `generar_ic_masa_fija_desde_grafo`), `cs090_diam_corregido`,
`cs080_renormalizacion`, `cs090_fase5_generador/motor/clasificador`. La selección de grafos se
reimplementa acá (mismo criterio que O3-B) en vez de llamar a `O3B.seleccionar_grafos` porque esa
función SOBREESCRIBE `cs090_fase6_o3b_seleccion.csv`, y en esta tarea no se toca ningún archivo ajeno.

No corre Phantom (eso es `cs090_fase7_f702_correr.py`). No declara cierre ni veredicto.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

import cs090_fase6_o3b_rewiring as O3B                  # sólo import (funciones de medición validadas)
from cs090_fase5b_phantom_adaptador import (            # sólo import (pipeline validado Fase V-B)
    reconstruir_regla_a2b0c2, generar_ic_masa_fija_desde_grafo,
)

# ---------------------------------------------------------------------------------------------
# Parámetros — heredados de la línea; los únicos nuevos son los del control de clustering
# ---------------------------------------------------------------------------------------------
N_NODOS = 2000
N_SWEEPS = 14
SEED_LAYOUT = 12345                  # MISMA semilla de layout que toda la Fase V-B / O3-B

MULT_SEED_ESCALERA = 9700            # multiplicador NUEVO para derivar la semilla de los swaps.
                                     # Ya usados en la línea: 1000/2000/5000/6000/7000/7500/8000/9100.

FACTOR_INTENTOS_BAJAR = 10           # intentos de swap "destructor de triángulos" = 10 × nº de aristas
                                     # (misma convención de dosis que `barajar_aristas`)
FACTOR_INTENTOS_SUBIR = 40           # intentos de swap dirigido "cerrador de triángulos" = 40 × aristas.
                                     # Es más alto porque cada propuesta dirigida es barata y muchas se
                                     # rechazan por balance neto; el techo real de la escalera es el que
                                     # este presupuesto alcanza, y se reporta como tal.
PARADA_SIN_MEJORA = 30000            # si pasan tantos intentos seguidos sin aceptar ninguno, se corta
                                     # (el proceso saturó: ese es el techo alcanzable)

# fracciones del máximo de triángulos REALMENTE alcanzado, para elegir los escalones
FRACCIONES_OBJETIVO = (0.0, 0.06, 0.18, 0.45, 1.0)
NOMBRES_ESCALON = ("e0", "e1", "e2", "e3", "e4")

# TECHO CON RESTRICCIÓN DE CONECTIVIDAD. Si se deja subir el clustering sin freno, el grafo termina
# rompiéndose en camarillas y la componente gigante colapsa (medido en el piloto: a C=0.577 la gigante
# cae de 1926 a 1236 nodos). Eso metería un segundo cambio grande además del clustering y arruinaría la
# intervención. Por eso el techo de la escalera se pone en el snapshot de MÁS triángulos que todavía
# conserva al menos esta fracción de la componente gigante del original. Se reporta también el techo
# SIN restricción, para dejar constancia de dónde se cortó y por qué.
FRAC_GIGANTE_MINIMA = 0.97

BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_fase7_f702_escalera"

RUTA_TOTAL_40 = f"{HERE}/cs090_fase5b_TOTAL_40pares.csv"
RUTA_REMEDICION = f"{HERE}/cs090_fase6_remedicion_430.csv"
RUTA_ESTRUCTURA = f"{HERE}/cs090_fase7_f702_estructura.csv"


# =============================================================================================
# 1) SELECCIÓN de los grafos base — MISMO criterio que O3-B (así la escalera se apoya
#    exactamente sobre los 12 grafos donde se midió la correlación ρ=0.77 que se quiere testear)
# =============================================================================================
def seleccionar_grafos(n_por_lote: int = 3):
    """Universo: las reglas con rol 'III' de `cs090_fase5b_TOTAL_40pares.csv`, unidas con la pendiente
    corregida de `cs090_fase6_remedicion_430.csv`. Se eligen las `n_por_lote` de mayor pendiente
    corregida dentro de cada uno de los 4 lotes de `seed_base`.

    LA UNIÓN ES POR (rule_id, seed), NO por rule_id solo: los lotes v1 y v2 de Fase V-B reusaron el
    mismo patrón de nombre `A2-B0-C2-r{idx}`, así que hay dos reglas DISTINTAS llamadas, por ejemplo,
    `A2-B0-C2-r14` (seed 273187 y seed 373187). Indexar por rule_id hace que una pise a la otra
    (bug real, documentado en FASE6_O3B_control_rewiring_CS.md §2.1)."""
    with open(RUTA_TOTAL_40) as f:
        total = list(csv.DictReader(f))
    with open(RUTA_REMEDICION) as f:
        remed = {(r["rule_id"], int(r["seed"])): r for r in csv.DictReader(f)}

    vistos, filas = set(), []
    for r in total:
        if r["rol"] != "III":
            continue
        clave = (r["rule_id"], int(r["seed"]))
        if clave in vistos:
            continue
        vistos.add(clave)
        m = remed.get(clave)
        if m is None:
            continue
        filas.append(dict(
            rule_id=r["rule_id"], seed=int(r["seed"]), lote=O3B.lote_de_seed(r["seed"]),
            K=int(r["K"]), kcap=int(r["kcap"]),
            pendiente_corregida=float(m["pendiente_corregida"]),
            frac_masa_fase5b=float(r["fraccion_masa_en_sumideros"]),
            carpeta_fase5b=r["carpeta"],
        ))

    por_lote = defaultdict(list)
    for f in filas:
        por_lote[f["lote"]].append(f)
    elegidos = []
    for lote in sorted(por_lote):
        grupo = sorted(por_lote[lote], key=lambda d: -d["pendiente_corregida"])[:n_por_lote]
        elegidos.extend(grupo)
    return elegidos


# =============================================================================================
# 2) MOTOR DE SWAPS DIRIGIDOS — la perilla del clustering, con grados clavados
# =============================================================================================
def _estado_desde_adj(adj, N):
    """Estructura de trabajo: lista de sets `adj`, lista `edges` (para muestrear aristas al azar en
    O(1)) y `idx` (arista ordenada -> posición en `edges`, para poder actualizar en O(1) tras un swap)."""
    adj = [set(adj[i]) for i in range(N)]
    edges = [(i, j) for i in range(N) for j in adj[i] if j > i]
    idx = {e: k for k, e in enumerate(edges)}
    return adj, edges, idx


def _n_triangulos(adj, N):
    """Nº total de triángulos, por enumeración exacta (barato con kcap<=8 y N=2000)."""
    t = 0
    for i in range(N):
        vec = sorted(adj[i])
        for a in range(len(vec)):
            for b in range(a + 1, len(vec)):
                if vec[b] in adj[vec[a]]:
                    t += 1
    return t // 3


def _delta_triangulos(adj, a, b, c, d):
    """Cambio EXACTO en el nº total de triángulos al hacer el swap (a-b),(c-d) -> (a-d),(c-b), con los
    4 nodos distintos.

    Por qué es exacto: quitar la arista a-b destruye exactamente |N(a) ∩ N(b)| triángulos, y quitar c-d
    destruye |N(c) ∩ N(d)|; ningún triángulo puede contener las DOS aristas a la vez (haría falta que
    compartieran un nodo, y los 4 son distintos), así que no hay doble conteo. Lo mismo del lado de las
    aristas nuevas. Requiere aplicar el swap para medir el lado nuevo, y por eso esta función lo APLICA
    y deja al llamador la decisión de revertirlo."""
    t_viejo = len(adj[a] & adj[b]) + len(adj[c] & adj[d])
    adj[a].discard(b); adj[b].discard(a)
    adj[c].discard(d); adj[d].discard(c)
    adj[a].add(d); adj[d].add(a)
    adj[c].add(b); adj[b].add(c)
    t_nuevo = len(adj[a] & adj[d]) + len(adj[c] & adj[b])
    return t_nuevo - t_viejo


def _revertir(adj, a, b, c, d):
    """Deshace exactamente lo que aplicó `_delta_triangulos`."""
    adj[a].discard(d); adj[d].discard(a)
    adj[c].discard(b); adj[b].discard(c)
    adj[a].add(b); adj[b].add(a)
    adj[c].add(d); adj[d].add(c)


def _confirmar_en_listas(edges, idx, a, b, c, d):
    """Sincroniza `edges`/`idx` con un swap ya aplicado sobre `adj`."""
    e1, e2 = (min(a, b), max(a, b)), (min(c, d), max(c, d))
    n1, n2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
    k1, k2 = idx.pop(e1), idx.pop(e2)
    edges[k1], edges[k2] = n1, n2
    idx[n1], idx[n2] = k1, k2


def bajar_clustering(adj, edges, idx, N, rng, n_intentos):
    """Swaps al azar aceptando sólo los que NO aumentan triángulos (los que bajan siempre; los neutros
    con probabilidad 1/2, para que el grafo igual se mezcle y no quede pegado al original). Devuelve el
    cambio acumulado de triángulos."""
    delta_total = 0
    n_aceptados = 0
    for _ in range(n_intentos):
        k1, k2 = int(rng.integers(0, len(edges))), int(rng.integers(0, len(edges)))
        if k1 == k2:
            continue
        a, b = edges[k1]
        c, d = edges[k2]
        if len({a, b, c, d}) < 4:
            continue
        n1, n2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if n1 in idx or n2 in idx or n1 == n2:
            continue
        delta = _delta_triangulos(adj, a, b, c, d)
        if delta < 0 or (delta == 0 and rng.random() < 0.5):
            _confirmar_en_listas(edges, idx, a, b, c, d)
            delta_total += delta
            n_aceptados += 1
        else:
            _revertir(adj, a, b, c, d)
    return delta_total, n_aceptados


def subir_clustering(adj, edges, idx, N, rng, n_intentos, t_actual,
                     al_cruzar=None, parada_sin_mejora=PARADA_SIN_MEJORA):
    """Swaps DIRIGIDOS a cerrar triángulos, aceptando sólo balance neto positivo.

    Propuesta: se elige un nodo u con grado>=2, dos vecinos suyos x,y no conectados entre sí (cerrar
    x-y crearía el triángulo u-x-y), y se buscan dos aristas para "pagar" el cambio: (x-p) y (y-q) con
    p,q distintos de u y entre sí. El swap (x-p),(y-q) -> (x-y),(p-q) conserva los grados de los cuatro.

    `al_cruzar(t_nuevo)` se llama después de cada aceptación, para que el llamador pueda guardar una
    foto del grafo cuando el nº de triángulos cruza un objetivo."""
    t = t_actual
    n_aceptados = 0
    sin_mejora = 0
    candidatos = [i for i in range(N) if len(adj[i]) >= 2]
    if not candidatos:
        return t, 0
    cand = np.array(candidatos, dtype=int)
    for _ in range(n_intentos):
        sin_mejora += 1
        if sin_mejora > parada_sin_mejora:
            break
        u = int(cand[rng.integers(0, len(cand))])
        vec_u = list(adj[u])
        if len(vec_u) < 2:
            continue
        i1, i2 = rng.integers(0, len(vec_u), size=2)
        if i1 == i2:
            continue
        x, y = vec_u[int(i1)], vec_u[int(i2)]
        if y in adj[x]:
            continue                      # ya estaba cerrado
        vx = [v for v in adj[x] if v != u and v != y]
        vy = [v for v in adj[y] if v != u and v != x]
        if not vx or not vy:
            continue
        p = vx[int(rng.integers(0, len(vx)))]
        q = vy[int(rng.integers(0, len(vy)))]
        if p == q or len({x, p, y, q}) < 4:
            continue
        n_pq = (min(p, q), max(p, q))
        if n_pq in idx:
            continue                      # p-q ya existe: el swap crearía una arista doble
        # swap (x-p),(y-q) -> (x-y),(p-q):  en la notación de _delta_triangulos, (a,b)=(x,p),
        # (c,d)=(q,y)  ->  (a,d)=(x,y) y (c,b)=(q,p)
        delta = _delta_triangulos(adj, x, p, q, y)
        if delta > 0:
            _confirmar_en_listas(edges, idx, x, p, q, y)
            t += delta
            n_aceptados += 1
            sin_mejora = 0
            if al_cruzar is not None:
                al_cruzar(t)
        else:
            _revertir(adj, x, p, q, y)
    return t, n_aceptados


def construir_escalera(adj_orig_lista, N, seed):
    """Devuelve la escalera completa: lista de dicts con la foto del grafo en cada escalón.

    Recorrido: original -> (bajar) -> piso E0 -> (subir, guardando fotos) -> techo E4.
    Las fotos intermedias se guardan sobre una rejilla fina de nº de triángulos y recién al final se
    eligen los 5 escalones, como fracciones del máximo REALMENTE alcanzado."""
    rng = np.random.default_rng(int(seed) * MULT_SEED_ESCALERA + 13)
    adj, edges, idx = _estado_desde_adj(adj_orig_lista, N)
    n_aristas = len(edges)
    t_orig = _n_triangulos(adj, N)

    t0 = time.time()
    d_baja, acc_baja = bajar_clustering(adj, edges, idx, N, rng, FACTOR_INTENTOS_BAJAR * n_aristas)
    t_piso = t_orig + d_baja
    t_bajar_s = time.time() - t0

    piso = dict(adj=[set(s) for s in adj], t=t_piso)

    # rejilla fina de fotos: una cada vez que el nº de triángulos crece un ~12%
    fotos = [(t_piso, [set(s) for s in adj])]
    proximo = [max(1, int(t_piso * 1.12) + 1)]

    def al_cruzar(t_nuevo):
        if t_nuevo >= proximo[0]:
            fotos.append((t_nuevo, [set(s) for s in adj]))
            proximo[0] = max(t_nuevo + 1, int(t_nuevo * 1.12) + 1)

    t1 = time.time()
    t_techo, acc_sube = subir_clustering(adj, edges, idx, N, rng,
                                         FACTOR_INTENTOS_SUBIR * n_aristas, t_piso, al_cruzar=al_cruzar)
    t_subir_s = time.time() - t1
    if not fotos or fotos[-1][0] != t_techo:
        fotos.append((t_techo, [set(s) for s in adj]))

    # --- techo con restricción de conectividad (ver FRAC_GIGANTE_MINIMA) ---
    gig_orig = O3B.tam_gigante(adj_orig_lista, N)
    piso_gigante = FRAC_GIGANTE_MINIMA * gig_orig
    k_techo = 0
    for j in range(len(fotos)):
        if O3B.tam_gigante(fotos[j][1], N) >= piso_gigante:
            k_techo = j
    fotos_ok = fotos[:k_techo + 1]
    t_techo_conectado = fotos_ok[-1][0]

    # elegir los escalones: la foto cuyo nº de triángulos está más cerca del objetivo
    rango = t_techo_conectado - t_piso
    escalones = []
    usados = set()
    for nombre, frac in zip(NOMBRES_ESCALON, FRACCIONES_OBJETIVO):
        objetivo = t_piso + frac * rango
        k = min(range(len(fotos_ok)), key=lambda j: abs(fotos_ok[j][0] - objetivo))
        # evitar repetir exactamente la misma foto en dos escalones distintos
        while k in usados and k + 1 < len(fotos_ok):
            k += 1
        usados.add(k)
        escalones.append(dict(escalon=nombre, frac_objetivo=frac, t_objetivo=objetivo,
                              n_triangulos=fotos_ok[k][0], adj=fotos_ok[k][1]))

    return dict(escalones=escalones, t_orig=t_orig, t_piso=t_piso,
                t_techo_sin_restriccion=t_techo, t_techo=t_techo_conectado,
                gigante_orig=gig_orig, n_fotos=len(fotos), n_fotos_conectadas=len(fotos_ok),
                acc_baja=acc_baja, acc_sube=acc_sube,
                t_bajar_s=round(t_bajar_s, 1), t_subir_s=round(t_subir_s, 1),
                n_aristas=n_aristas, piso=piso)


# =============================================================================================
# 3) COVARIABLES que podrían moverse "de arrastre" al mover el clustering
# =============================================================================================
def asortatividad_grados(adj, N):
    """Correlación de Pearson entre los grados de los dos extremos de cada arista (Newman). Se mide
    porque los swaps dirigidos podrían, sin querer, empujar también esta propiedad: si la masa siguiera
    a la asortatividad y no al clustering, se vería acá."""
    xs, ys = [], []
    for i in range(N):
        for j in adj[i]:
            if j > i:
                xs.append(len(adj[i])); ys.append(len(adj[j]))
    if not xs:
        return float("nan")
    # cada arista aporta en los dos sentidos (definición simétrica estándar)
    a = np.array(xs + ys, dtype=float)
    b = np.array(ys + xs, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def n_componentes(adj, N):
    return len(O3B.DC.componentes(adj, N))


# =============================================================================================
# 4) PROCESAR un grafo base: escalera + medición + condiciones iniciales de Phantom
# =============================================================================================
def procesar_una(sel, generar_ic=True, con_original=True):
    rid, seed = sel["rule_id"], sel["seed"]
    t_ini = time.time()

    p, m = reconstruir_regla_a2b0c2(seed=seed, N=N_NODOS, n_sweeps=N_SWEEPS)
    adj_orig_nativo = m["adj_final"]                    # objeto nativo del motor (orden de sets nativo)
    adj_orig = O3B._a_lista(adj_orig_nativo, N_NODOS)
    E_orig = O3B.aristas_set(adj_orig, N_NODOS)
    g_orig = O3B.grados(adj_orig, N_NODOS)

    esc = construir_escalera(adj_orig, N_NODOS, seed)
    adjs_null = O3B.nulls_topo_de(seed, N_NODOS, len(E_orig))

    variantes = []
    if con_original:
        variantes.append(("orig", adj_orig, adj_orig_nativo))
    for e in esc["escalones"]:
        variantes.append((e["escalon"], e["adj"], e["adj"]))

    filas = []
    for nombre, adj_v, adj_para_ic in variantes:
        # --- VERIFICACIÓN NUMÉRICA nodo por nodo (no se asume) ---
        g_v = O3B.grados(adj_v, N_NODOS)
        iguales = bool(np.array_equal(g_orig, g_v))
        n_dif = int((g_orig != g_v).sum())
        assert iguales, (f"{rid}/{nombre}: la secuencia de grados NO se conservó "
                         f"({n_dif} nodos difieren) -- no se sigue adelante con este grafo")
        E_v = O3B.aristas_set(adj_v, N_NODOS)
        assert len(E_v) == len(E_orig), f"{rid}/{nombre}: cambió el nº de aristas"
        assert all(i not in adj_v[i] for i in range(N_NODOS)), f"{rid}/{nombre}: hay un bucle i-i"

        cl, tr, ntri = O3B.clustering(adj_v, N_NODOS)
        pc = O3B.pendiente_corregida_de_grafo(O3B.canonicalizar(adj_v, N_NODOS), N_NODOS, seed, adjs_null)

        fila = dict(
            rule_id=rid, seed=seed, lote=sel["lote"], K=sel["K"], kcap=sel["kcap"],
            escalon=nombre, n_nodos=N_NODOS, n_aristas=len(E_v),
            grado_medio=2.0 * len(E_v) / N_NODOS,
            grados_identicos=iguales, n_nodos_grado_distinto=n_dif,
            solapamiento_aristas=len(E_orig & E_v) / len(E_orig),
            clustering_local=cl, transitividad=tr, n_triangulos=ntri,
            gigante=O3B.tam_gigante(adj_v, N_NODOS), n_componentes=n_componentes(adj_v, N_NODOS),
            asortatividad=asortatividad_grados(adj_v, N_NODOS),
            pendiente_corr=pc["pendiente"], z_agg=pc["z_agg"], diams=pc["diams"],
            n_cajas=pc["n_cajas"],
            clustering_original=None, n_triangulos_original=esc["t_orig"],
            t_piso=esc["t_piso"], t_techo=esc["t_techo"],
            t_techo_sin_restriccion=esc["t_techo_sin_restriccion"],
            n_fotos=esc["n_fotos"], n_fotos_conectadas=esc["n_fotos_conectadas"],
            acc_baja=esc["acc_baja"], acc_sube=esc["acc_sube"],
            t_bajar_s=esc["t_bajar_s"], t_subir_s=esc["t_subir_s"],
            seed_escalera=int(seed) * MULT_SEED_ESCALERA + 13,
            pendiente_corr_csv_ref=sel["pendiente_corregida"],
            frac_masa_fase5b=sel["frac_masa_fase5b"],
        )

        if generar_ic:
            # nombre de carpeta con SEED incluido: los rule_id de los lotes v1/v2 se repiten entre lotes
            # (A2-B0-C2-r14 existe con seed 273187 y con 373187) -- sin el seed dos grafos distintos se
            # pisarían la carpeta. Prefijo `f702` jamás usado antes en la línea.
            carpeta = f"{BASE_SALIDA}/{rid}_s{seed}_f702_{nombre}"
            os.makedirs(carpeta, exist_ok=True)
            t1 = time.time()
            info = generar_ic_masa_fija_desde_grafo(adj_para_ic, N=N_NODOS, seed_layout=SEED_LAYOUT,
                                                    ruta_salida=f"{carpeta}/cosmogenesis_ic.txt")
            fila["t_ic_s"] = round(time.time() - t1, 2)
            meta = dict(
                tarea="FASE7_F702_escalera_clustering", escalon=nombre, rule_id=rid, clase="III",
                seed=seed, lote=sel["lote"], N=N_NODOS, seed_layout=SEED_LAYOUT,
                K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"],
                sim_thr_frac=p["sim_thr_frac"],
                n_aristas_grafo_final=len(E_v), grado_medio_grafo_final=2.0 * len(E_v) / N_NODOS,
                grados_identicos_al_original=iguales,
                seed_escalera=fila["seed_escalera"],
                solapamiento_aristas=fila["solapamiento_aristas"],
                clustering_local=cl, transitividad=tr, n_triangulos=ntri,
                gigante=fila["gigante"], pendiente_corregida=pc["pendiente"],
                masa_total_ic=info["masa_total"], carpeta=carpeta,
            )
            with open(f"{carpeta}/meta_regla.json", "w") as f:
                json.dump(meta, f, indent=2)
            fila["carpeta"] = carpeta

        filas.append(fila)
        print(f"    {nombre:>4}  tri={ntri:<5} C={cl:.5f}  trans={tr:.5f}  solape={fila['solapamiento_aristas']:.3f} "
              f"gigante={fila['gigante']}  asort={fila['asortatividad']:+.3f}  pend={pc['pendiente']:.4f}"
              + (f"  ic={fila.get('t_ic_s')}s" if generar_ic else ""), flush=True)

    c_orig = next((f["clustering_local"] for f in filas if f["escalon"] == "orig"), None)
    if c_orig is None:
        c_orig, _, _ = O3B.clustering(adj_orig, N_NODOS)
    for f in filas:
        f["clustering_original"] = c_orig
        f["t_total_grafo_s"] = round(time.time() - t_ini, 1)
    return filas


def main(indices=None, generar_ic=True, sufijo_csv="", con_original=True, n_por_lote=3):
    elegidos = seleccionar_grafos(n_por_lote=n_por_lote)
    if indices is not None:
        elegidos = [elegidos[i] for i in indices]
    print(f"[f702] {len(elegidos)} grafos base (generar_ic={generar_ic}, con_original={con_original})",
          flush=True)

    todas, t0 = [], time.time()
    for k, sel in enumerate(elegidos):
        print(f"[{k+1}/{len(elegidos)}] {sel['rule_id']} seed={sel['seed']} lote={sel['lote']} "
              f"pend_corr={sel['pendiente_corregida']:.4f}", flush=True)
        todas.extend(procesar_una(sel, generar_ic=generar_ic, con_original=con_original))

    ruta = RUTA_ESTRUCTURA.replace(".csv", f"{sufijo_csv}.csv")
    with open(ruta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(todas[0].keys()))
        w.writeheader()
        w.writerows(todas)
    print(f"\n[f702] {len(todas)} filas -> {ruta.split('/')[-1]}  (total {time.time()-t0:.0f}s)")
    return todas


if __name__ == "__main__":
    # uso: python3.9 cs090_fase7_f702_escalera.py [idx0,idx1,...] [--sin-ic] [--sin-orig] [--sufijo=_s0]
    idxs, gen_ic, suf, con_orig = None, True, "", True
    for arg in sys.argv[1:]:
        if arg == "--sin-ic":
            gen_ic = False
        elif arg == "--sin-orig":
            con_orig = False
        elif arg.startswith("--sufijo="):
            suf = arg.split("=", 1)[1]
        else:
            idxs = [int(x) for x in arg.split(",")]
    main(indices=idxs, generar_ic=gen_ic, sufijo_csv=suf, con_original=con_orig)
