"""
cs090_fase7_f706_cupos.py — FASE VII, tarea F7-06: CON LA MISMA DISTRIBUCIÓN DE CAPACIDADES,
¿IMPORTA *A QUÉ NODO* LE TOCA CADA CUPO?
============================================================================================

QUÉ PREGUNTA CONTESTA
---------------------
`FASE6_O2C_kcap_capacidad_finita_CS.md` (Prueba 3) repartió capacidad relacional desigual entre los
nodos manteniendo la MISMA media, y encontró dos cosas: (a) la heterogeneidad mueve la geometría en
dosis-respuesta con el CV del reparto (Spearman +0.745), pero (b) el 86-88% de eso pasa por un canal
trivial — los repartos desiguales desperdician cupo y la red termina con MENOS aristas.

Y dejó una anomalía sin explicar: **`HET-grado` es la única distribución que va en dirección
CONTRARIA** (más aristas, menos pendiente en crudo, pero +0.109 POR ENCIMA de la curva de referencia
del grado alcanzado). Es también la única cuyo cupo está **correlacionado con el grado inicial del
nodo** — o sea, la única donde la capacidad está ALINEADA con la estructura que ya existía. El propio
informe marcó el control que falta (§10, último ítem): *"sortear cupos con la misma distribución que
HET-grado pero permutados entre nodos, rompiendo la correlación con el grado inicial y conservando
exactamente la distribución"*.

Este archivo corre ese control, y le agrega el tercer punto del eje (anti-alineado), que es el análogo
directo del hallazgo de `FASE6_O1D_factorial_trio_CS.md`: allí, con alineación el trío real AYUDA y sin
alineación ESTORBA (interacción z=−6.36). Si acá pasa lo mismo, sería el segundo lugar del proyecto
donde lo que decide es la **alineación con la estructura previa**, no la magnitud.

LOS TRES (CUATRO) BRAZOS — mismo multiconjunto de cupos, distinta asignación
----------------------------------------------------------------------------
Todos parten del MISMO grafo Erdős-Rényi inicial (mismo `seed`, mismo `construir_A2`), consumen el
MISMO flujo de números aleatorios en la dinámica y usan el MISMO `kcap` medio (el `p["kcap"]` de la
regla). Lo único que cambia es **qué nodo recibe qué cupo**:

  1. `unif`      — REFERENCIA/VERIFICACIÓN: cupo constante = `p["kcap"]` para todos. Con vector
                   constante, `MA.dinamica_B0_hibrido` es el C2-hard del motor congelado (verificado
                   en O2-C, VERIFICACIÓN 1). Acá se vuelve a verificar arista por arista contra
                   `MOT.dinamica_B0` y contra el `n_aristas_grafo_final` del `meta_regla.json` que
                   Fase V-B escribió para esta misma regla. **No se manda a Phantom** (su número ya
                   está en disco desde Fase V-B); es el control de que el pipeline es fiel.
  2. `alineado`  — el cupo que le corresponde a cada nodo por su grado inicial: `MA._cupo_variable`
                   TAL CUAL, la misma función que O2-C usa para `HET-grado` (se importa, no se copia).
  3. `permutado` — **EL MISMO MULTICONJUNTO EXACTO** de cupos de (2), barajado entre nodos con una
                   permutación aleatoria. Misma distribución, misma media, mismo CV, mismo mínimo y
                   máximo: cambia sólo la asignación. Se verifica con
                   `np.array_equal(np.sort(cupo_alineado), np.sort(cupo_permutado))` — no se asume.
  4. `anti`      — el mismo multiconjunto, asignado **al revés**: el cupo más alto al nodo de grado
                   inicial más bajo y viceversa. Los empates de grado (que son masivos en un ER: el
                   grado es un entero chico) se rompen **al azar**, no por índice de nodo, que
                   sesgaría sistemáticamente hacia los nodos de índice bajo. También se verifica el
                   multiconjunto.

CONFOUND DECLARADO — Y MEDIDO ANTES DE MANDAR NADA A PHANTOM
-------------------------------------------------------------
A diferencia de F7-04 —donde los brazos terminaban con el MISMO nº de aristas por construcción— acá la
intervención es **dentro de la dinámica**, así que permutar los cupos PUEDE cambiar cuántas aristas
sobreviven. Y en este sistema la densidad domina todo (masa vs aristas ρ=−0.97,
`INFORME_EQUIPO_FASE6_11ago2026_CS.md`).

Se corrió primero la parte estructural sola (12 grafos, sin Phantom) y el confound resultó **enorme y
unidireccional: `alineado` > `permutado` > `anti` en 12 de 12 grafos**, con Δ de 244 a 2 083 aristas.
Tiene sentido mecánico: si el cupo grande le toca justo al nodo que ya tenía muchos vecinos, casi no
hay que podar; si le toca a un nodo de grado bajo, el cupo se desperdicia y el vecino saturado pierde
aristas igual. Con esa diferencia de densidad, comparar la masa cruda entre brazos mediría densidad,
no alineación.

Por eso se agregan **dos brazos de densidad igualada**, en el espíritu exacto de F7-04:

  5. `alin_dil`  — el grafo `alineado` con aristas quitadas **al azar** hasta tener EXACTAMENTE el
                   mismo nº de aristas que `anti` (el más ralo de los tres, siempre).
  6. `perm_dil`  — ídem con `permutado`.

Así el trío {`alin_dil`, `perm_dil`, `anti`} tiene, grafo por grafo, **el mismo nº exacto de aristas**,
y cualquier diferencia de masa entre ellos ya no puede ser densidad. La asimetría que queda —a `anti`
no hubo que quitarle nada, a los otros dos sí— se declara: el dilución al azar es en sí una
perturbación estructural (es el brazo `azar` de F7-04), así que el contraste **limpio** de esta tarea
es `alin_dil` vs `perm_dil`, que recibieron el MISMO tratamiento y difieren sólo en el grafo de origen.
El contraste crudo (`alineado` vs `permutado`) se reporta igual, etiquetado como confundido con la
densidad, y el analizador además lo ajusta por regresión contra el nº de aristas.

QUÉ SE VERIFICA NUMÉRICAMENTE (nada de esto se asume)
------------------------------------------------------
  a) el multiconjunto de cupos es IDÉNTICO en los tres brazos heterogéneos (vectores ordenados,
     `np.array_equal`), y con él la media, el CV, el mínimo y el máximo;
  b) la correlación cupo↔grado_inicial (Spearman) es ≈+1 en `alineado`, ≈0 en `permutado` y ≈−1 en
     `anti` — o sea, la manipulación hizo lo que dice hacer;
  c) el brazo `unif` reproduce, arista por arista, el grafo final de `MOT.dinamica_B0` corrido de
     punta a punta con `_enforce_kcap` escalar;
  d) ese mismo nº de aristas coincide con el `n_aristas_grafo_final` del `meta_regla.json` de Fase V-B
     para esa regla (verificación cruzada contra DISCO, con rule_id y seed comprobados).

TRATO IDÉNTICO ENTRE BRAZOS
---------------------------
Los cuatro grafos se **canonicalizan** (`canonicalizar` de `cs090_fase6_o3b_rewiring`, reusada tal
cual) antes de medir y antes de escribir la condición inicial: el orden de iteración de un `set` de
Python depende de cómo se llenó y `cs080.cajas_bfs` recorre `for v in adj[u]`, así que dos grafos con
las mismas aristas armadas en otro orden pueden dar particiones distintas (verificado en O3-B). La
pendiente corregida de cada brazo se mide contra NULLs de SU PROPIA densidad (`nulls_topo_de`), que es
el trato estándar de la línea cuando los brazos no comparten nº de aristas.

QUÉ NO HACE
-----------
No modifica ningún archivo existente: importa `cs090_fase5_generador`, `cs090_fase5_motor`,
`cs090_fase5_mecanismo_aislado` (de ahí sale `_cupo_variable`, la lógica de HET-grado),
`cs090_fase6_o3b_rewiring`, `cs090_fase5b_phantom_adaptador`, `cs080_renormalizacion` y
`cs090_diam_corregido` — todos SÓLO se importan. No corre Phantom (eso es
`cs090_fase7_f706_correr.py`). No declara cierre ni veredicto: fabrica los brazos, los mide y escribe
números.

USO
---
    python3.9 cs090_fase7_f706_cupos.py --seleccion              # sólo la selección
    python3.9 cs090_fase7_f706_cupos.py 0 --sufijo=_piloto       # piloto cronometrado (1 grafo)
    python3.9 cs090_fase7_f706_cupos.py 0,3,6,9 --sufijo=_shard0 # un shard, para paralelizar
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
import cs090_fase5_mecanismo_aislado as MA               # sólo import (_cupo_variable + dinámica híbrida)
from cs090_fase5b_phantom_adaptador import generar_ic_masa_fija_desde_grafo   # sólo import
from cs090_fase6_o3b_rewiring import (                   # piezas de medición ya validadas en O3-B
    canonicalizar, aristas_set, clustering, tam_gigante,
    nulls_topo_de, pendiente_corregida_de_grafo, lote_de_seed,
    N_NODOS, N_SWEEPS, SEED_LAYOUT,
)

# ---------------------------------------------------------------------------------------------
# Parámetros — heredados de la línea; los únicos nuevos son los del sorteo de asignación de cupos
# ---------------------------------------------------------------------------------------------
BRAZOS_DINAMICA = ("unif", "alineado", "permutado", "anti")   # salen de correr la dinámica
BRAZOS_DILUIDOS = ("alin_dil", "perm_dil")                    # densidad igualada a `anti`
BRAZOS = BRAZOS_DINAMICA + BRAZOS_DILUIDOS
# `unif` no va a Phantom: es el control de fidelidad y su número ya está en disco desde Fase V-B
BRAZOS_PHANTOM = ("alineado", "permutado", "anti", "alin_dil", "perm_dil")

MULT_SEED_CUPO = 9706        # multiplicador NUEVO para las semillas de asignación de cupo. Los ya
                             # usados en la línea son 1000/2000/3000/5000/6000/7000/7500/8000/
                             # 9100(O3-B)/9704(F7-04): 9706 no colisiona con ninguno.

BASE_SALIDA = "/Users/alexis/phantom_cs073/bateria_fase7_f706_cupos_alineados"
SUFIJO_CARPETA = "f706"      # prefijo/sufijo NUEVO (lección del bug de colisión de nombres de Fase V-B)

RUTA_TOTAL_40 = f"{HERE}/cs090_fase5b_TOTAL_40pares.csv"
RUTA_REMEDICION = f"{HERE}/cs090_fase6_remedicion_430.csv"
RUTA_SELECCION = f"{HERE}/cs090_fase7_f706_seleccion.csv"
RUTA_ESTRUCTURA = f"{HERE}/cs090_fase7_f706_estructura.csv"

BATERIAS_FASE5B = [
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto",
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v2",
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v3",
    "/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v4",
]


# =============================================================================================
# 0) utilidades chicas
# =============================================================================================
def _rangos(x):
    """Rangos promediados para empates (equivalente a scipy.stats.rankdata, sin la dependencia)."""
    x = np.asarray(x, dtype=float)
    orden = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    xs = x[orden]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[orden[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rangos(a), _rangos(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 1e-12 else float("nan")


# =============================================================================================
# 1) SELECCIÓN de los grafos base — mismo universo y mismo criterio que F7-04
# =============================================================================================
def seleccionar_grafos(n_por_lote: int = 3):
    """Universo: las reglas de `cs090_fase5b_TOTAL_40pares.csv` (las que YA pasaron por Phantom en
    Fase V-B, así que su `meta_regla.json` existe en disco para la verificación cruzada), unidas con la
    pendiente CORREGIDA de `cs090_fase6_remedicion_430.csv`.

    Unión por `(rule_id, seed)`, NUNCA por `rule_id` solo: los lotes v1 y v2 comparten el patrón de
    nombre `A2-B0-C2-r{idx}` y hay reglas DISTINTAS con el mismo id (bug documentado en O3-B).

    Criterio: dentro de cada uno de los 4 lotes de `seed_base` (271828/371828/471828/571828) se toman
    la de pendiente corregida MÍNIMA, la MEDIANA y la MÁXIMA => 3 por lote, 12 grafos. Se reparte por
    lote (independencia experimental, lección de O2-F) y se abre en abanico dentro del lote a propósito:
    el endpoint es continuo, así que interesa cubrir rango. "% Clase III" no se usa en ningún punto.
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
            continue
        filas.append(dict(
            rule_id=r["rule_id"], seed=int(r["seed"]), lote=lote_de_seed(r["seed"]),
            rol_fase5b=r["rol"], K=int(r["K"]), kcap=int(r["kcap"]),
            pendiente_corregida=float(m["pendiente_corregida"]),
            frac_masa_fase5b=float(r["fraccion_masa_en_sumideros"]),
            kappa_v_fase5b=(float(r["kappa_v_agregado"]) if r["kappa_v_agregado"] else None),
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
              f"{[(g['rule_id'], g['kcap'], round(g['pendiente_corregida'], 3)) for g in sel]}",
              flush=True)

    with open(RUTA_SELECCION, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(elegidos[0].keys()))
        w.writeheader()
        w.writerows(elegidos)
    print(f"   [selección] {len(elegidos)} grafos -> {RUTA_SELECCION.split('/')[-1]}", flush=True)
    return elegidos


# =============================================================================================
# 2) LOS VECTORES DE CUPO — mismo multiconjunto, tres asignaciones
# =============================================================================================
def fabricar_cupos(grado_inicial, kcap_base, seed):
    """Devuelve dict(brazo -> vector de cupo entero) + un dict de verificación.

    `alineado` es EXACTAMENTE `MA._cupo_variable(grado_inicial, kcap_base)`, la función que O2-C usa
    para `HET-grado` — importada, no copiada. Los otros dos brazos son PERMUTACIONES de ese mismo
    vector, así que su multiconjunto de valores es idéntico por construcción; igual se comprueba con
    `np.array_equal` sobre los vectores ordenados, porque "por construcción" es exactamente el tipo de
    afirmación que conviene no creerse.

    `anti`: cupos ordenados de mayor a menor asignados a los nodos ordenados de menor a mayor grado
    inicial. Los empates de grado se rompen con una permutación aleatoria (`jitter` en el `lexsort`),
    no por índice de nodo: en un ER el grado es un entero chico y hay cientos de nodos empatados en
    cada valor, así que romper por índice metería un sesgo sistemático hacia los nodos de índice bajo.
    """
    N = len(grado_inicial)
    cupo_alineado = np.asarray(MA._cupo_variable(grado_inicial, kcap_base), dtype=int)

    base = int(seed) * MULT_SEED_CUPO
    perm = np.random.default_rng(base + 17).permutation(N)
    cupo_permutado = cupo_alineado[perm]

    jitter = np.random.default_rng(base + 29).permutation(N)
    nodos_por_grado_asc = np.lexsort((jitter, np.asarray(grado_inicial, dtype=float)))
    cupos_desc = np.sort(cupo_alineado)[::-1]
    cupo_anti = np.empty(N, dtype=int)
    cupo_anti[nodos_por_grado_asc] = cupos_desc

    cupo_unif = np.full(N, int(round(kcap_base)), dtype=int)

    cupos = dict(unif=cupo_unif, alineado=cupo_alineado, permutado=cupo_permutado, anti=cupo_anti)

    orden_a = np.sort(cupo_alineado)
    verif = dict(
        multiconjunto_alineado_vs_permutado=bool(np.array_equal(orden_a, np.sort(cupo_permutado))),
        multiconjunto_alineado_vs_anti=bool(np.array_equal(orden_a, np.sort(cupo_anti))),
        seed_perm=base + 17, seed_anti=base + 29,
    )
    for b, v in cupos.items():
        verif[f"cupo_media_{b}"] = float(v.mean())
        verif[f"cupo_cv_{b}"] = float(v.std() / v.mean()) if v.mean() > 0 else float("nan")
        verif[f"cupo_min_{b}"] = int(v.min())
        verif[f"cupo_max_{b}"] = int(v.max())
        verif[f"cupo_suma_{b}"] = int(v.sum())
        verif[f"rho_cupo_grado_{b}"] = spearman(v, grado_inicial)
    return cupos, verif


# =============================================================================================
# 3) CORRER UN BRAZO: dinámica A2-B0 con cupo por nodo (la pieza ya escrita de F5-C2-C3)
# =============================================================================================
def correr_brazo(seed, cupo, N=N_NODOS, n_sweeps=N_SWEEPS):
    """Mismo camino que `MA.correr_regla_coarse_hibrido` y que `MOT.correr_regla_coarse`: misma regla,
    mismo rng derivado de `seed*5000+N`, mismo `construir_A2`, misma dinámica de 14 sweeps con recableo
    co-emergente cada 3 pasos y enforcement de cupo cada 4, misma poda final por costo P70. El ÚNICO
    cambio entre brazos es el vector `cupo`. Se reconstruye el sustrato desde cero en cada brazo (y no
    se reusa uno mutado) para que los cuatro consuman el MISMO flujo de números aleatorios."""
    p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=seed)
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)
    sustrato = MA.dinamica_B0_hibrido(sustrato, p, rng, n_sweeps, np.asarray(cupo, dtype=int), "soporte")
    m = MOT.medir(sustrato, p, rng)
    return p, m["adj_final"]


def grado_inicial_de(seed, N=N_NODOS):
    """Grado del grafo ER RECIÉN NACIDO, antes del primer sweep — la misma lectura que usa
    `MA.correr_regla_coarse_hibrido` (se captura inmediatamente después de `construir_A2`)."""
    p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=seed)
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)
    return p, np.array([len(sustrato["adj"][i]) for i in range(N)], dtype=float)


def diluir_a(adj, N, M_objetivo, rng):
    """Quita aristas AL AZAR (uniforme, sin reemplazo) hasta dejar exactamente `M_objetivo`. La lista de
    aristas se arma ordenada (`aristas_set` + `sorted`) para que el sorteo no dependa del orden de
    iteración de los `set` de Python, que no es reproducible entre construcciones distintas del mismo
    grafo (misma lección que la canonicalización de O3-B). Devuelve (adj_diluido, n_quitadas)."""
    aristas = sorted(aristas_set(adj, N))
    M = len(aristas)
    n_quitar = M - int(M_objetivo)
    assert n_quitar >= 0, f"no se puede diluir de {M} a {M_objetivo}: el objetivo es mayor"
    fuera = set(int(k) for k in rng.choice(M, size=n_quitar, replace=False)) if n_quitar else set()
    out = [set() for _ in range(N)]
    for k, (i, j) in enumerate(aristas):
        if k in fuera:
            continue
        out[i].add(j)
        out[j].add(i)
    return canonicalizar(out, N), n_quitar


def verificar_unif_reproduce_motor(seed, adj_unif, N=N_NODOS, n_sweeps=N_SWEEPS):
    """Corre `MOT.dinamica_B0` COMPLETO (C2-hard escalar, sin cupo por vector) y compara arista por
    arista con el brazo `unif`. Si difieren, el camino híbrido no es fiel al motor congelado y nada de
    lo que sigue es comparable con la historia de la línea."""
    p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=seed)
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)
    sustrato = MOT.dinamica_B0(sustrato, p, rng, n_sweeps, "C2")
    m = MOT.medir(sustrato, p, rng)
    E_motor = aristas_set(m["adj_final"], N)
    E_unif = aristas_set(adj_unif, N)
    return bool(E_motor == E_unif), len(E_motor), len(E_motor ^ E_unif)


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


def verificar_contra_meta_fase5b(sel, n_aristas_unif):
    """Contra DISCO (no contra el CSV): que el `meta_regla.json` de Fase V-B (a) exista, (b) tenga el
    mismo rule_id y seed, y (c) registre el mismo nº de aristas del grafo final que nuestro `unif`."""
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
    out["meta5b_n_aristas_coincide"] = bool(out["meta5b_n_aristas"] == n_aristas_unif)
    return out


# =============================================================================================
# 5) PROCESAR UN GRAFO BASE
# =============================================================================================
def procesar_una(sel, generar_ic=True, verificar_motor=True, medir_pendiente=True):
    rid, seed = sel["rule_id"], int(sel["seed"])
    N = N_NODOS
    t_ini = time.time()
    cron = {}

    t0 = time.time()
    p, grado_inicial = grado_inicial_de(seed, N=N)
    kcap_base = int(p["kcap"])
    cupos, verif_cupo = fabricar_cupos(grado_inicial, kcap_base, seed)
    cron["t_cupos_s"] = round(time.time() - t0, 2)
    assert verif_cupo["multiconjunto_alineado_vs_permutado"], \
        f"{rid}: el multiconjunto de cupos de `permutado` NO es idéntico al de `alineado`"
    assert verif_cupo["multiconjunto_alineado_vs_anti"], \
        f"{rid}: el multiconjunto de cupos de `anti` NO es idéntico al de `alineado`"

    t0 = time.time()
    adjs_nativos = {}
    for b in BRAZOS_DINAMICA:
        _, adj = correr_brazo(seed, cupos[b], N=N)
        adjs_nativos[b] = adj
    cron["t_dinamica_s"] = round(time.time() - t0, 2)
    adjs = {b: canonicalizar(a, N) for b, a in adjs_nativos.items()}     # MISMO trato a los cuatro
    n_aristas = {b: len(aristas_set(a, N)) for b, a in adjs.items()}

    # --- brazos de densidad igualada: alineado y permutado diluidos al azar hasta el nº de `anti` ---
    M_objetivo = min(n_aristas["alineado"], n_aristas["permutado"], n_aristas["anti"])
    base_dil = int(seed) * MULT_SEED_CUPO
    adjs["alin_dil"], n_quit_alin = diluir_a(adjs["alineado"], N, M_objetivo,
                                             np.random.default_rng(base_dil + 41))
    adjs["perm_dil"], n_quit_perm = diluir_a(adjs["permutado"], N, M_objetivo,
                                             np.random.default_rng(base_dil + 53))
    for b in BRAZOS_DILUIDOS:
        n_aristas[b] = len(aristas_set(adjs[b], N))
    assert n_aristas["alin_dil"] == n_aristas["perm_dil"] == M_objetivo, \
        f"{rid}: la dilución no dejó el nº de aristas exacto: {n_aristas}"

    if verificar_motor:
        t0 = time.time()
        unif_ok, n_motor, n_dif = verificar_unif_reproduce_motor(seed, adjs_nativos["unif"], N=N)
        cron["t_verif_motor_s"] = round(time.time() - t0, 2)
    else:
        unif_ok, n_motor, n_dif = None, None, None

    # --- métricas estructurales por brazo ---
    t0 = time.time()
    metricas = {}
    for b, a in adjs.items():
        cl, tr, ntri = clustering(a, N)
        metricas[b] = dict(clustering_local=cl, transitividad=tr, n_triangulos=ntri,
                           gigante=tam_gigante(a, N), grado_medio=2.0 * n_aristas[b] / N)
    cron["t_clustering_s"] = round(time.time() - t0, 2)

    # --- pendiente corregida: cada brazo contra NULLs de SU PROPIA densidad ---
    # (a diferencia de F7-04, acá los brazos NO comparten nº de aristas — ver el confound declarado
    #  en el encabezado — así que compartir una única vara NULL introduciría una asimetría)
    if medir_pendiente:
        t0 = time.time()
        for b, a in adjs.items():
            adjs_null = nulls_topo_de(seed, N, n_aristas[b])
            pc = pendiente_corregida_de_grafo(a, N, seed, adjs_null)
            metricas[b].update(pendiente_corregida=pc["pendiente"], z_agg=pc["z_agg"],
                               z_sostenido=pc["z_sostenido"], diams=pc["diams"], n_cajas=pc["n_cajas"])
        cron["t_pendiente_s"] = round(time.time() - t0, 2)

    # --- solapamiento de aristas entre brazos (¿los grafos son realmente distintos?) ---
    sets_ar = {b: aristas_set(a, N) for b, a in adjs.items()}
    solapes = {}
    for b in ("permutado", "anti", "unif"):
        inter = len(sets_ar["alineado"] & sets_ar[b])
        union = len(sets_ar["alineado"] | sets_ar[b])
        solapes[f"jaccard_alineado_{b}"] = inter / union if union else float("nan")
    inter = len(sets_ar["alin_dil"] & sets_ar["perm_dil"])
    union = len(sets_ar["alin_dil"] | sets_ar["perm_dil"])
    solapes["jaccard_alin_dil_perm_dil"] = inter / union if union else float("nan")

    fila = dict(
        rule_id=rid, seed=seed, lote=sel["lote"], motivo_seleccion=sel.get("motivo_seleccion"),
        rol_fase5b=sel.get("rol_fase5b"), K=p["K"], J=p["J"], noise=p["noise"],
        meandeg=p["meandeg"], kcap=kcap_base, n_nodos=N,
        grado_inicial_medio=float(grado_inicial.mean()),
        grado_inicial_cv=float(grado_inicial.std() / grado_inicial.mean()),
        unif_reproduce_motor=unif_ok, n_aristas_motor=n_motor, n_aristas_dif_vs_motor=n_dif,
        pendiente_corregida_fase5b_ref=sel["pendiente_corregida"],
        frac_masa_fase5b=sel["frac_masa_fase5b"], kappa_v_fase5b=sel.get("kappa_v_fase5b"),
        **verif_cupo, **solapes,
    )
    for b in BRAZOS:
        fila[f"n_aristas_{b}"] = n_aristas[b]
        for k, v in metricas[b].items():
            fila[f"{k}_{b}"] = v
    fila["aristas_identicas_en_brazos_dinamica"] = len(
        {n_aristas[b] for b in ("alineado", "permutado", "anti")}) == 1
    fila["d_aristas_alineado_menos_permutado"] = n_aristas["alineado"] - n_aristas["permutado"]
    fila["d_aristas_alineado_menos_anti"] = n_aristas["alineado"] - n_aristas["anti"]
    fila["M_objetivo_diluido"] = int(M_objetivo)
    fila["n_quitadas_alin_dil"] = int(n_quit_alin)
    fila["n_quitadas_perm_dil"] = int(n_quit_perm)
    fila["densidad_igualada_en_el_trio"] = bool(
        n_aristas["alin_dil"] == n_aristas["perm_dil"] == n_aristas["anti"])
    fila.update(verificar_contra_meta_fase5b(sel, n_aristas["unif"]))

    # --- condiciones iniciales de Phantom (sólo los 3 brazos heterogéneos) ---
    origen_de = dict(alin_dil="alineado", perm_dil="permutado")
    if generar_ic:
        t0 = time.time()
        for b in BRAZOS_PHANTOM:
            carpeta = f"{BASE_SALIDA}/{rid}_{SUFIJO_CARPETA}_{b}"
            os.makedirs(carpeta, exist_ok=True)
            info = generar_ic_masa_fija_desde_grafo(adjs[b], N=N, seed_layout=SEED_LAYOUT,
                                                    ruta_salida=f"{carpeta}/cosmogenesis_ic.txt")
            bo = origen_de.get(b, b)      # los diluidos heredan el cupo de su brazo de origen
            meta = dict(
                tarea="FASE7_F706_cupos_alineados", brazo=b, brazo_origen=bo,
                diluido=bool(b in BRAZOS_DILUIDOS), rule_id=rid, seed=seed,
                lote=sel["lote"], N=N, seed_layout=SEED_LAYOUT,
                K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=kcap_base,
                sim_thr_frac=p["sim_thr_frac"],
                cupo_media=verif_cupo[f"cupo_media_{bo}"], cupo_cv=verif_cupo[f"cupo_cv_{bo}"],
                cupo_min=verif_cupo[f"cupo_min_{bo}"], cupo_max=verif_cupo[f"cupo_max_{bo}"],
                cupo_suma=verif_cupo[f"cupo_suma_{bo}"],
                rho_cupo_grado=verif_cupo[f"rho_cupo_grado_{bo}"],
                n_aristas_grafo_final=n_aristas[b],
                M_objetivo_diluido=int(M_objetivo),
                n_quitadas_al_azar=(int(n_quit_alin) if b == "alin_dil" else
                                    int(n_quit_perm) if b == "perm_dil" else 0),
                grado_medio_grafo_final=metricas[b]["grado_medio"],
                clustering_local=metricas[b]["clustering_local"],
                transitividad=metricas[b]["transitividad"],
                pendiente_corregida=metricas[b].get("pendiente_corregida"),
                masa_total_ic=info["masa_total"], carpeta=carpeta,
            )
            with open(f"{carpeta}/meta_regla.json", "w") as f:
                json.dump(meta, f, indent=2)
            fila[f"carpeta_{b}"] = carpeta
        cron["t_ic_s"] = round(time.time() - t0, 2)

    fila.update(cron)
    fila["t_total_s"] = round(time.time() - t_ini, 2)
    return fila


def main(indices=None, generar_ic=True, verificar_motor=True, medir_pendiente=True, sufijo_csv=""):
    elegidos = seleccionar_grafos()
    if indices is not None:
        elegidos = [elegidos[i] for i in indices]
    print(f"\n[f706] procesando {len(elegidos)} grafos base x {len(BRAZOS)} brazos "
          f"(IC para {BRAZOS_PHANTOM}; generar_ic={generar_ic})", flush=True)

    filas, t0 = [], time.time()
    for k, sel in enumerate(elegidos):
        print(f"[{k+1}/{len(elegidos)}] {sel['rule_id']} (lote {sel['lote']}, kcap={sel['kcap']}, "
              f"{sel.get('motivo_seleccion')})...", flush=True)
        fila = procesar_una(sel, generar_ic=generar_ic, verificar_motor=verificar_motor,
                            medir_pendiente=medir_pendiente)
        filas.append(fila)
        print("    multiconjunto idéntico: perm={} anti={}   |  rho(cupo,grado) "
              .format(fila["multiconjunto_alineado_vs_permutado"],
                      fila["multiconjunto_alineado_vs_anti"])
              + " ".join(f"{b}={fila[f'rho_cupo_grado_{b}']:+.3f}" for b in BRAZOS_DINAMICA),
              flush=True)
        print("    aristas    " + " ".join(f"{b}={fila[f'n_aristas_{b}']}" for b in BRAZOS)
              + f"   unif==motor:{fila['unif_reproduce_motor']} "
                f"meta5b_ok:{fila['meta5b_n_aristas_coincide']} "
                f"dens_igualada:{fila['densidad_igualada_en_el_trio']}", flush=True)
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
    print(f"\n[f706] {len(filas)} filas -> {ruta.split('/')[-1]}  (total {time.time()-t0:.0f}s)")
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
