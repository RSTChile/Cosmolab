"""
CS083c — FASE IV / tarea O1-D: FACTORIAL SISTEMÁTICO DEL TRÍO (n=30)
=====================================================================
QUIÉN SOY: sigue directo a `cs083b_fase4_control_local_global.py` (informe
`FASE4_control_local_global_CS.md`). Ese informe dejó UN hallazgo raro sin explicar:

    NULL-LOCAL-ROTO (el trío "coherente pero equivocado": las 3 aristas de OTRO triángulo real)
    aplana MENOS que NULL-REWIRE (3 aristas sueltas al azar, sin relación entre sí),
    z = -15.62 — en la dirección OPUESTA a la intuición ("parecerse más a un triángulo real
    debería imitar mejor al REAL").

Ahí se especuló —sin comprobarlo— que la causa podría ser una diferencia de COBERTURA DE ARISTAS
entre los dos esquemas de aleatorización (LOCAL-ROTO concentraría los empujones en las aristas de
zonas densas y dejaría aristas periféricas sin tocar; REWIRE los repartiría más parejo).

Este script pone esa hipótesis a prueba con un factorial de 7 brazos × 30 semillas, midiendo en
CADA brazo, además de la holonomía, tres familias de descriptores nuevos:
  (i)   COBERTURA de aristas (cuántas aristas distintas reciben empujón, y qué tan dispareja es la
        distribución de empujones por arista: Gini, entropía normalizada, fracción de aristas con
        cero empujones, conteo máximo).
  (ii)  VARIACIÓN INTRA-TRÍO (¿cuánto difieren entre sí las 3 aristas de cada trío al final de la
        corrida?) — la hipótesis alternativa del "estrés estructural".
  (iii) SOLAPE DE NODOS del trío destino (descriptor puramente estructural, 0..3 pares del trío que
        comparten un nodo: 3 = trío geométricamente coherente, ~0 = 3 aristas sueltas).

────────────────────────────────────────────────────────────────────────────────────────────────
OBSERVACIÓN ANALÍTICA QUE REDEFINE EL DISEÑO (encontrada al leer el código de cs083b, ANTES de
correr nada — se verifica numéricamente en el modo `pilot`):

  El destino de NULL-LOCAL-ROTO se elige con un DERANGEMENT (permutación sin puntos fijos) sobre el
  conjunto de triángulos. Una permutación es una BIYECCIÓN: cada triángulo real es destino de
  exactamente un empujón por sweep, igual que en REAL. Por lo tanto el MULTICONJUNTO de "cuántos
  empujones recibe cada arista" es IDÉNTICO entre REAL y LOCAL-ROTO (arista por arista, no sólo en
  promedio). La cobertura NO puede explicar la diferencia REAL vs LOCAL-ROTO.

  Segunda consecuencia, más fuerte: si en LOCAL-ROTO el defecto empujado se calculara sobre el trío
  DESTINO (en vez del trío propio), la biyección haría que la suma de correcciones del sweep fuera
  EXACTAMENTE la misma que en REAL (la corrección se acumula y se aplica al final del sweep, así que
  el orden dentro del sweep es irrelevante). O sea: "trío coherente, triángulo equivocado, defecto
  del destino" ES REAL, bit a bit. Se verifica en el piloto (`_verificar_biyeccion_equivale_a_real`).

  Conclusión de diseño: lo único que separa a LOCAL-ROTO de REAL NO es "el trío equivocado" — es la
  DESALINEACIÓN entre la fuente del defecto (las 3 aristas PROPIAS de T) y el destino del empujón
  (las 3 aristas de T'). Ese es el factor que el diseño de cs083b no tenía cruzado. Por eso este
  factorial lo cruza explícitamente:

        FACTOR 1 — DESTINO del empujón:  trío-real-ajeno   /   3 aristas sueltas al azar
        FACTOR 2 — ALINEACIÓN fuente↔destino:  alineado (el defecto es el del trío que se empuja)
                                               desalineado (el defecto es el de OTRO trío)

        alineado   × trío-real-ajeno   =  REAL           (probado idéntico: brazo A)
        alineado   × 3-sueltas         =  NULL-REWIRE    (brazo C)
        desalineado× trío-real-ajeno   =  NULL-LOCAL-ROTO(brazo E, el del hallazgo raro)
        desalineado× 3-sueltas         =  NUEVO          (brazo F, la celda que faltaba)

────────────────────────────────────────────────────────────────────────────────────────────────
LOS 7 BRAZOS (todos: mismo N=110, mismo grafo base por semilla, mismo K=6, J=0.6, J_FACE=0.5,
ruido=0.25, mismo COMPUTE_BUDGET=60 000 → mismos sweeps; lo único que cambia es a QUÉ aristas va el
empujón y DE QUÉ trío sale el defecto):

  A  correcto            destino = sus propias 3 aristas          | defecto = del destino  | = REAL (cs082, bit a bit)
  B  casi_correcto       destino = 1 arista propia + 2 sueltas    | defecto = del destino  | CONDICIÓN NUEVA pedida (llena el hueco correcto↔equivocado)
  C  azar                destino = 3 aristas sueltas al azar      | defecto = del destino  | = NULL-REWIRE (cs083, bit a bit)
  D  sin_trios           destino = TODAS las aristas, cada sweep  | defecto = vs media global | = NULL-GLOBAL (cs083b, bit a bit)
  E  equivocado          destino = las 3 aristas de OTRO triáng.  | defecto = del trío PROPIO | = NULL-LOCAL-ROTO (cs083b, bit a bit) — el hallazgo raro
  F  azar_desalineado    destino = 3 aristas sueltas al azar      | defecto = del trío PROPIO | DIAGNÓSTICO NUEVO: cierra el 2×2
  G  trio_real_azar      destino = trío de un triángulo real al   | defecto = del destino     | DIAGNÓSTICO NUEVO: coherencia SIN biyección
                          azar CON reemplazo (T'≠T)                                            (cobertura más concentrada que REAL, trío coherente, alineado)
  H  azarTri_desalin     destino = 3 aristas sueltas sorteadas    | defecto = del trío PROPIO | DIAGNÓSTICO NUEVO: igual PUNTERÍA que E, SIN trío coherente
                          SÓLO entre aristas de algún triángulo
  I  azarTri_alineado    ídem H                                   | defecto = del destino     | DIAGNÓSTICO NUEVO: igual PUNTERÍA que A, SIN trío coherente

  (+ referencias no dinámicas por semilla, importadas de cs082: NULL = ruido puro, SHUFFLED.)

  Por qué H e I: en C y F las 3 aristas-destino se sortean sobre TODO el grafo, y ~41% de las aristas
  del grafo no pertenecen a NINGÚN triángulo — o sea, ~41% de esos empujones cae en lugares que la
  holonomía ni siquiera mide. En A y E, en cambio, el 100% de los empujones cae sobre aristas de
  triángulo. Esa diferencia de PUNTERÍA es la segunda variable de la familia "cobertura" (distinta del
  Gini: no dice si el reparto es parejo, dice si apunta o no al subconjunto medido) y estaba
  confundida con la coherencia del trío en el contraste C-vs-E. H e I la igualan.

Los brazos A, C, D y E se implementan de modo que reproducen BIT A BIT las funciones ya auditadas de
cs082/cs083/cs083b (mismo stream de RNG, mismo orden de consumo). El modo `pilot` verifica esa
identidad numéricamente contra las funciones importadas — si algún brazo se desviara, el piloto lo
grita antes de gastar las 30 semillas.

REGLA DE LA CASA: NO se modifica cs082_fase4_4sustratos.py, cs083_fase4_robustecer.py ni
cs083b_fase4_control_local_global.py — se importan sus piezas tal cual. Código nuevo: el motor
genérico de brazos, los brazos B/F/G, los descriptores de cobertura / variación intra-trío / solape,
y el análisis de correlación cobertura↔holonomía.

No se declara cierre ni veredicto — se reportan números. La lectura final es de Alexis.

USO:
    ./venv/bin/python cs083c_fase4_factorial_trio.py pilot   # 5 semillas + verificaciones bit a bit
    ./venv/bin/python cs083c_fase4_factorial_trio.py full    # 30 semillas + CSV + PNG
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from cs082_fase4_4sustratos import (  # noqa: E402
    N, K, J, NOISE, J_FACE, COMPUTE_BUDGET,
    construir_base, _linea_adyacencia, _circ_mean_update, _holonomia_triangulos,
    _n_sweeps_para_presupuesto, null_de, shuffled_de,
    correr_sustrato_4_2complejo,
)
from cs083_fase4_robustecer import (  # noqa: E402
    correr_sustrato_4_control_fino,
    z_score_pareado, test_permutacion_signo_pareado,
)
from cs083b_fase4_control_local_global import (  # noqa: E402
    _derangement,
    correr_sustrato_4_null_local_roto,
    correr_sustrato_4_null_global,
)

# ============================ CONFIG ============================
N_SEEDS_FULL = 30                      # pedido explícito de la tarea O1-D
SEEDS_FULL   = list(range(1, N_SEEDS_FULL + 1))
SEEDS_PILOT  = [1, 2, 3, 4, 5]         # mismas 5 del piloto de cs083b, para poder comparar
N_PERM       = 20_000
RNG_MASTER_SEED = 779                  # stream propio de este script (778 fue cs083b, 777 cs083)

# streams de RNG por brazo — los 4 primeros COINCIDEN con los de los scripts originales para que la
# reproducción sea bit a bit; los nuevos usan streams libres.
SEMILLA_BASE = {
    "A_correcto":         40_000,   # = cs082 correr_sustrato_4_2complejo
    "B_casi_correcto":    48_000,   # nuevo
    "C_azar":             45_000,   # = cs083 correr_sustrato_4_control_fino
    "D_sin_trios":        47_000,   # = cs083b correr_sustrato_4_null_global
    "E_equivocado":       46_000,   # = cs083b correr_sustrato_4_null_local_roto
    "F_azar_desalineado": 50_000,   # nuevo
    "G_trio_real_azar":   51_000,   # nuevo
    "H_azarTri_desalin":  52_000,   # nuevo — puntería igualada a E, sin coherencia de trío
    "I_azarTri_alineado": 53_000,   # nuevo — puntería igualada a A, sin coherencia de trío
}

ETIQUETA = {
    "A_correcto":         "A trío CORRECTO (=REAL)",
    "B_casi_correcto":    "B trío CASI-correcto (1+2)",
    "C_azar":             "C trío AL AZAR (=REWIRE)",
    "D_sin_trios":        "D SIN TRÍOS (=GLOBAL)",
    "E_equivocado":       "E trío EQUIVOCADO (=LOC.ROTO)",
    "F_azar_desalineado": "F azar DESALINEADO (nuevo)",
    "G_trio_real_azar":   "G trío real AZAR c/reempl.",
    "H_azarTri_desalin":  "H sueltas-DE-triáng. desalin.",
    "I_azarTri_alineado": "I sueltas-DE-triáng. alineado",
}

# los 5 brazos que la tarea pidió explícitamente (el resto son diagnósticos añadidos)
BRAZOS_PEDIDOS = ["A_correcto", "B_casi_correcto", "C_azar", "D_sin_trios", "E_equivocado"]
BRAZOS_TODOS = list(SEMILLA_BASE.keys())
# ================================================================


# ══════════════════════════ DESCRIPTORES NUEVOS ══════════════════════════
def gini(conteos):
    """Coeficiente de Gini sobre el vector de 'cuántos empujones recibe cada arista' (incluyendo las
    aristas con cero). 0 = todas las aristas reciben exactamente lo mismo (cobertura perfectamente
    pareja); →1 = casi todo el empuje cae sobre unas pocas aristas (cobertura concentrada).
    Es el descriptor central de la hipótesis a poner a prueba en esta tarea."""
    c = np.sort(np.asarray(conteos, dtype=float))
    n = len(c)
    total = c.sum()
    if n == 0 or total <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * c)) / (n * total) - (n + 1.0) / n)


def entropia_normalizada(conteos):
    """Entropía de Shannon de la distribución 'proporción de empujones que recibe cada arista',
    normalizada por log(n_aristas). 1 = reparto perfectamente parejo; 0 = todo a una sola arista.
    Es el mismo fenómeno que mide el Gini, en otra escala — se reportan los dos porque tienen
    sensibilidades distintas a la cola de aristas con CERO empujones (la entropía las ignora, el
    Gini las castiga). Acá 'entropía' es sólo un descriptor de dispersión, no una tesis."""
    c = np.asarray(conteos, dtype=float)
    n = len(c)
    total = c.sum()
    if n <= 1 or total <= 0:
        return 0.0
    p = c[c > 0] / total
    return float(-np.sum(p * np.log(p)) / np.log(n))


def descriptores_cobertura(destinos, n_edges, aristas_de_triangulo=None):
    """A partir de la lista de tríos-destino (fija durante toda la corrida), cuenta cuántos empujones
    por sweep recibe cada arista del grafo y resume la forma de esa distribución.

    `frac_punteria` es la fracción de los empujones que cae sobre aristas que pertenecen a al menos un
    triángulo — o sea, sobre las aristas que la holonomía efectivamente MIDE. Es la segunda variable de
    la familia 'cobertura', distinta del Gini: no habla de si el reparto es parejo, sino de si apunta o
    no al subconjunto de aristas donde se lee el resultado."""
    conteos = np.zeros(n_edges)
    for trio in destinos:
        for idx in trio:
            conteos[idx] += 1
    tocadas = int(np.count_nonzero(conteos))
    total = conteos.sum()
    if aristas_de_triangulo is not None and total > 0:
        mascara = np.zeros(n_edges, dtype=bool)
        mascara[list(aristas_de_triangulo)] = True
        punteria = float(conteos[mascara].sum() / total)
    else:
        punteria = float("nan")
    return dict(
        aristas_tocadas=tocadas,
        frac_cobertura=tocadas / n_edges if n_edges else 0.0,
        frac_aristas_cero=1.0 - (tocadas / n_edges if n_edges else 0.0),
        gini_cobertura=gini(conteos),
        entropia_cobertura=entropia_normalizada(conteos),
        conteo_max=float(conteos.max()) if n_edges else 0.0,
        conteo_medio=float(conteos.mean()) if n_edges else 0.0,
        frac_punteria=punteria,
    )


def dispersion_intra_trio(s, trios):
    """Variación INTRA-TRÍO al final de la corrida: para cada trío de 3 aristas se calcula el módulo
    de la media circular de sus 3 orientaciones (R∈[0,1]) y se devuelve 1-R promediado sobre tríos.
    0 = las 3 aristas del trío terminaron apuntando al mismo lado (trío 'en paz'); →1 = las 3 tiran
    en direcciones incompatibles ('estrés estructural', la hipótesis alternativa del analista)."""
    if not trios:
        return 0.0
    ang = 2 * np.pi * np.asarray(s) / K
    z = np.exp(1j * ang)
    idx = np.asarray(trios)                      # (n_trios, 3)
    R = np.abs(z[idx].mean(axis=1))
    return float(np.mean(1.0 - R))


def solape_nodos_trio(destinos, edges):
    """Descriptor puramente estructural del trío-destino: de los 3 pares posibles entre sus 3
    aristas, ¿cuántos comparten al menos un nodo? 3 = las 3 aristas son mutuamente incidentes (un
    triángulo genuino o al menos una estrella); 0 = 3 aristas totalmente sueltas. Sirve para verificar
    que los brazos 'coherentes' realmente lo son y los 'sueltos' realmente no."""
    if not destinos:
        return 0.0
    ee = [set(e) for e in edges]
    tot = 0.0
    for (a, b, c) in destinos:
        pares = ((a, b), (a, c), (b, c))
        tot += sum(1 for (x, y) in pares if ee[x] & ee[y])
    return tot / (len(destinos))


# ══════════════════════════ MOTOR GENÉRICO DE BRAZOS ══════════════════════════
def _tri_edges_de(triangles, idx_edge):
    """Pre-indexa, para cada triángulo, los 3 índices de sus aristas de borde (misma construcción que
    cs082 — copiada, no modificada, para poder reusarla desde el motor genérico)."""
    out = []
    for (i, j, k) in triangles:
        e1 = (i, j) if i < j else (j, i)
        e2 = (j, k) if j < k else (k, j)
        e3 = (i, k) if i < k else (k, i)
        out.append((idx_edge[e1], idx_edge[e2], idx_edge[e3]))
    return out


def _construir_destinos(modo, tri_edges, n_edges, n_tri, rng):
    """Construye, UNA vez al arrancar la corrida (no por sweep), la lista de tríos-destino de cada
    triángulo, según el brazo. El ORDEN de consumo del RNG está elegido para reproducir bit a bit los
    scripts originales en los brazos A/C/E."""
    if modo == "propios":                      # A
        return list(tri_edges)
    if modo == "casi":                         # B — 1 arista propia (al azar de las 3) + 2 sueltas
        dest = []
        for (a, b, c) in tri_edges:
            propia = (a, b, c)[int(rng.integers(3))]
            sueltas = rng.choice(n_edges, size=2, replace=False)
            dest.append((int(propia), int(sueltas[0]), int(sueltas[1])))
        return dest
    if modo == "azar":                         # C y F — 3 aristas sueltas al azar (mismo draw que cs083)
        if n_edges < 3:
            return []
        return [tuple(int(x) for x in rng.choice(n_edges, size=3, replace=False)) for _ in range(n_tri)]
    if modo == "derangement":                  # E — el trío de OTRO triángulo real, por biyección
        perm = _derangement(n_tri, rng)
        return [tri_edges[perm[t]] for t in range(n_tri)]
    if modo == "azar_tri":                     # H e I — 3 aristas sueltas al azar, pero sorteadas SÓLO
                                               # entre las aristas que pertenecen a ≥1 triángulo. Iguala
                                               # la "puntería" (qué proporción de los empujones cae sobre
                                               # aristas que la holonomía mide) con los brazos de trío
                                               # real, dejando la COHERENCIA del trío como única
                                               # diferencia restante.
        aristas_tri = np.array(sorted({i for trio in tri_edges for i in trio}))
        if len(aristas_tri) < 3:
            return []
        return [tuple(int(x) for x in rng.choice(aristas_tri, size=3, replace=False)) for _ in range(n_tri)]
    if modo == "trio_real_azar":               # G — trío real al azar CON reemplazo (T'≠T), sin biyección
        dest = []
        for t in range(n_tri):
            tp = int(rng.integers(n_tri))
            while n_tri > 1 and tp == t:
                tp = int(rng.integers(n_tri))
            dest.append(tri_edges[tp])
        return dest
    raise ValueError(f"modo de destino desconocido: {modo}")


def correr_brazo(edges, triangles, seed, modo_destino, alineado, semilla_base):
    """Motor genérico del sustrato 4 con destino de empujón configurable.

    Cada sweep hace exactamente lo mismo que `correr_sustrato_4_2complejo` de cs082:
      1) un paso de alineación circular entre aristas vecinas (idéntico, importado),
      2) por cada triángulo real T: se calcula un defecto de holonomía h y se reparte el empujón
         -J_FACE*h/3 sobre 3 aristas; las correcciones se acumulan y se aplican al final del sweep
         (promediadas por la cantidad de empujones que recibió cada arista).

    Lo único configurable:
      · modo_destino: A QUÉ 3 aristas va el empujón de T (ver `_construir_destinos`).
      · alineado=True  -> h se calcula sobre las MISMAS 3 aristas que se empujan (el defecto que se
                          intenta cerrar es el del trío que recibe la corrección: es una fuerza
                          auto-consistente, "cerrá tu propio lazo").
        alineado=False -> h se calcula sobre las 3 aristas PROPIAS de T pero se empuja al trío
                          destino: la corrección que llega a un trío no tiene nada que ver con el
                          estado de ese trío (fuerza desalineada, "te corrijo por el error de otro").

    Devuelve el campo de aristas final, los tríos-destino usados (para los descriptores de cobertura)
    y los metadatos de equiparación.
    """
    rng = np.random.default_rng(semilla_base + seed)
    n_edges = len(edges)
    idx_edge = {e: i for i, e in enumerate(edges)}
    vecinos = _linea_adyacencia(edges)
    s = rng.uniform(0, K, n_edges)                        # 1er consumo de RNG (igual que los originales)
    dof = n_edges + len(triangles)
    n_sweeps = _n_sweeps_para_presupuesto(n_edges, COMPUTE_BUDGET)
    n_tri = len(triangles)
    tri_edges = _tri_edges_de(triangles, idx_edge)
    destinos = _construir_destinos(modo_destino, tri_edges, n_edges, n_tri, rng)
    fuentes = destinos if alineado else tri_edges

    t0 = time.time()
    for _ in range(n_sweeps):
        s = _circ_mean_update(s, vecinos, J, NOISE, rng)
        if destinos:
            correccion = np.zeros(n_edges)
            cuenta = np.zeros(n_edges)
            for t_idx in range(len(destinos)):
                fa, fb, fc = fuentes[t_idx]
                h = (s[fa] + s[fb] + s[fc]) % K
                h = h - K if h > K / 2 else h
                for idx in destinos[t_idx]:
                    correccion[idx] += -J_FACE * h / 3.0
                    cuenta[idx] += 1
            mask = cuenta > 0
            s[mask] = (s[mask] + correccion[mask] / cuenta[mask]) % K
    dt = time.time() - t0
    E = {e: s[idx_edge[e]] for e in edges}
    return dict(E=E, s=s, destinos=destinos, tri_edges=tri_edges,
                dof=dof, n_sweeps=n_sweeps, dt=dt, eventos_por_sweep=3 * len(destinos))


def correr_brazo_sin_trios(edges, triangles, seed, semilla_base):
    """Brazo D (=NULL-GLOBAL de cs083b): no hay ningún trío. Cada sweep, TODAS las aristas del grafo
    reciben un empujón hacia la media circular global del campo, con la misma constante de fuerza por
    evento (J_FACE/3) que recibe una arista real de su cara. Es la cobertura máxima y perfectamente
    pareja posible: Gini=0, entropía=1, ninguna arista sin tocar. Se delega en la función original de
    cs083b (importada, no modificada) y acá sólo se le agregan los descriptores nuevos."""
    E, dof, n_sweeps, dt, ev = correr_sustrato_4_null_global(None, edges, triangles, seed)
    s = np.array([E[e] for e in edges])
    return dict(E=E, s=s, destinos=None, tri_edges=None,
                dof=dof, n_sweeps=n_sweeps, dt=dt, eventos_por_sweep=ev)


CONFIG_BRAZO = {
    "A_correcto":         dict(modo="propios",        alineado=True),
    "B_casi_correcto":    dict(modo="casi",           alineado=True),
    "C_azar":             dict(modo="azar",           alineado=True),
    "D_sin_trios":        None,                                        # motor aparte
    "E_equivocado":       dict(modo="derangement",    alineado=False),
    "F_azar_desalineado": dict(modo="azar",           alineado=False),
    "G_trio_real_azar":   dict(modo="trio_real_azar", alineado=True),
    "H_azarTri_desalin":  dict(modo="azar_tri",       alineado=False),
    "I_azarTri_alineado": dict(modo="azar_tri",       alineado=True),
}


# ══════════════════════════ VERIFICACIONES BIT A BIT ══════════════════════════
def _verificar_reproduccion(seed):
    """Comprueba que el motor genérico reproduce EXACTAMENTE (bit a bit) las funciones ya auditadas de
    cs082/cs083/cs083b en los brazos A, C, E y D. Si algo se desvía, es un bug del motor genérico y
    hay que verlo antes de gastar las 30 semillas."""
    adj, edges, triangles = construir_base(seed)
    ok = {}

    E_ref, _, _, _ = correr_sustrato_4_2complejo(adj, edges, triangles, seed)
    r = correr_brazo(edges, triangles, seed, "propios", True, SEMILLA_BASE["A_correcto"])
    ok["A_correcto == cs082.sustrato_4_real"] = all(
        np.isclose(E_ref[e], r["E"][e], rtol=0, atol=0) for e in edges)

    E_ref, _, _, _, _ = correr_sustrato_4_control_fino(adj, edges, triangles, seed)
    r = correr_brazo(edges, triangles, seed, "azar", True, SEMILLA_BASE["C_azar"])
    ok["C_azar == cs083.control_fino (REWIRE)"] = all(
        np.isclose(E_ref[e], r["E"][e], rtol=0, atol=0) for e in edges)

    E_ref, _, _, _, _ = correr_sustrato_4_null_local_roto(adj, edges, triangles, seed)
    r = correr_brazo(edges, triangles, seed, "derangement", False, SEMILLA_BASE["E_equivocado"])
    ok["E_equivocado == cs083b.null_local_roto"] = all(
        np.isclose(E_ref[e], r["E"][e], rtol=0, atol=0) for e in edges)

    E_ref, _, _, _, _ = correr_sustrato_4_null_global(adj, edges, triangles, seed)
    r = correr_brazo_sin_trios(edges, triangles, seed, SEMILLA_BASE["D_sin_trios"])
    ok["D_sin_trios == cs083b.null_global"] = all(
        np.isclose(E_ref[e], r["E"][e], rtol=0, atol=0) for e in edges)

    return ok


def _verificar_biyeccion_equivale_a_real(seed):
    """Prueba numérica de la observación analítica del encabezado: si el destino se permuta por una
    BIYECCIÓN sobre los triángulos y el defecto se toma del trío DESTINO (alineado), el resultado es
    idéntico a REAL — porque la corrección se acumula sobre todo el sweep y el orden es irrelevante.
    Se usa una rotación cíclica (biyección sin puntos fijos que NO consume RNG, para que el stream
    quede idéntico al de REAL y la comparación sea bit a bit)."""
    adj, edges, triangles = construir_base(seed)
    idx_edge = {e: i for i, e in enumerate(edges)}
    tri_edges = _tri_edges_de(triangles, idx_edge)
    n_tri = len(triangles)
    destinos_rotados = [tri_edges[(t + 1) % n_tri] for t in range(n_tri)]

    # --- REAL, con el motor genérico
    r_real = correr_brazo(edges, triangles, seed, "propios", True, SEMILLA_BASE["A_correcto"])

    # --- misma corrida pero con destino rotado y defecto DEL DESTINO (alineado)
    rng = np.random.default_rng(SEMILLA_BASE["A_correcto"] + seed)
    n_edges = len(edges)
    vecinos = _linea_adyacencia(edges)
    s = rng.uniform(0, K, n_edges)
    n_sweeps = _n_sweeps_para_presupuesto(n_edges, COMPUTE_BUDGET)
    for _ in range(n_sweeps):
        s = _circ_mean_update(s, vecinos, J, NOISE, rng)
        correccion = np.zeros(n_edges)
        cuenta = np.zeros(n_edges)
        for t_idx in range(n_tri):
            a, b, c = destinos_rotados[t_idx]
            h = (s[a] + s[b] + s[c]) % K
            h = h - K if h > K / 2 else h
            for idx in (a, b, c):
                correccion[idx] += -J_FACE * h / 3.0
                cuenta[idx] += 1
        mask = cuenta > 0
        s[mask] = (s[mask] + correccion[mask] / cuenta[mask]) % K
    E_rot = {e: s[idx_edge[e]] for e in edges}
    iguales = all(np.isclose(r_real["E"][e], E_rot[e], rtol=0, atol=0) for e in edges)

    # y de paso: ¿la cobertura de E (derangement) es idéntica a la de A?
    rA = descriptores_cobertura(r_real["destinos"], n_edges)
    rng2 = np.random.default_rng(SEMILLA_BASE["E_equivocado"] + seed)
    _ = rng2.uniform(0, K, n_edges)
    perm = _derangement(n_tri, rng2)
    destinos_E = [tri_edges[perm[t]] for t in range(n_tri)]
    rE = descriptores_cobertura(destinos_E, n_edges)
    cobertura_identica = (abs(rA["gini_cobertura"] - rE["gini_cobertura"]) < 1e-12
                          and rA["aristas_tocadas"] == rE["aristas_tocadas"])
    return iguales, cobertura_identica, rA, rE


# ══════════════════════════ ESTADÍSTICA / SALIDA ══════════════════════════
def resumen(nombre, vals):
    v = np.asarray(vals)
    return (f"{nombre:<32} n={len(v):>2}  media={v.mean():.4f}  DE={v.std(ddof=1):.4f}  "
            f"min={v.min():.4f}  max={v.max():.4f}")


def _reporte_par(nombre, a, b, n_perm, rng_perm):
    z = z_score_pareado(a, b)
    t = test_permutacion_signo_pareado(a, b, n_perm, rng_perm, direccion="a<b")
    n_neg = int(np.sum(np.asarray(a) < np.asarray(b)))     # semillas en que 'a' aplana más que 'b'
    print(f"  {nombre:<46} z={z:+8.2f}  diff_obs={t['obs_diff']:+.4f}  "
          f"p_1cola={t['p_una_cola']:.5f}  p_2colas={t['p_dos_colas']:.5f}  signos a<b={n_neg}/{len(a)}")
    return dict(z=z, signos_a_menor=n_neg, n=len(a), **t)


def spearman(x, y):
    """Correlación de Spearman (rangos) — implementada a mano con numpy para no depender de scipy en
    esta máquina; con empates usa el promedio de rangos, igual que scipy."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")

    def rangos(v):
        orden = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[orden] = np.arange(1, len(v) + 1, dtype=float)
        # promediar rangos de empates
        vals, inv, cuentas = np.unique(v, return_inverse=True, return_counts=True)
        for k in np.where(cuentas > 1)[0]:
            m = inv == k
            r[m] = r[m].mean()
        return r

    rx, ry = rangos(x), rangos(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def correr_bateria(seeds, tag=""):
    print("CS083c — FASE IV / O1-D: FACTORIAL SISTEMÁTICO DEL TRÍO " + tag)
    print("=" * 110)
    print(f"N={N} nodos · K={K} · J={J} · J_FACE={J_FACE} · ruido={NOISE} · presupuesto={COMPUTE_BUDGET} "
          f"op-relación/brazo · n_semillas={len(seeds)}")
    print("Brazos: " + " | ".join(ETIQUETA[b] for b in BRAZOS_TODOS) + "\n")

    filas = []
    t0_bat = time.time()
    for seed in seeds:
        adj, edges, triangles = construir_base(seed)
        n_e, n_t = len(edges), len(triangles)
        # referencias no dinámicas (mismas funciones de cs082 que usaron cs083/cs083b)
        r_ref = correr_brazo(edges, triangles, seed, "propios", True, SEMILLA_BASE["A_correcto"])
        h_null = _holonomia_triangulos(null_de(r_ref["E"], seed), triangles).mean()
        h_shuf = _holonomia_triangulos(shuffled_de(r_ref["E"], seed), triangles).mean()

        aristas_tri = sorted({i for trio in r_ref["tri_edges"] for i in trio})

        for brazo in BRAZOS_TODOS:
            cfg = CONFIG_BRAZO[brazo]
            if cfg is None:
                r = correr_brazo_sin_trios(edges, triangles, seed, SEMILLA_BASE[brazo])
                # cobertura de D: TODAS las aristas, exactamente 1 empujón cada una, cada sweep
                cob = descriptores_cobertura([(i,) for i in range(n_e)], n_e, aristas_tri)
                cob["conteo_max"] = 1.0
                solape = float("nan")            # no hay tríos: el descriptor no aplica
                disp_dest = float("nan")
            else:
                r = r_ref if brazo == "A_correcto" else correr_brazo(
                    edges, triangles, seed, cfg["modo"], cfg["alineado"], SEMILLA_BASE[brazo])
                cob = descriptores_cobertura(r["destinos"], n_e, aristas_tri)
                solape = solape_nodos_trio(r["destinos"], edges)
                disp_dest = dispersion_intra_trio(r["s"], r["destinos"])

            h = _holonomia_triangulos(r["E"], triangles).mean()
            # variación intra-trío sobre los triángulos REALES (comparable entre TODOS los brazos,
            # incluido D que no tiene tríos-destino)
            tri_edges_reales = _tri_edges_de(triangles, {e: i for i, e in enumerate(edges)})
            disp_real = dispersion_intra_trio(r["s"], tri_edges_reales)

            filas.append(dict(
                seed=seed, brazo=brazo, etiqueta=ETIQUETA[brazo],
                n_edges=n_e, n_tri=n_t,
                dof=r["dof"], n_sweeps=r["n_sweeps"], eventos_por_sweep=r["eventos_por_sweep"],
                dt=round(r["dt"], 3),
                h=h, h_null=h_null, h_shuf=h_shuf,
                **cob,
                solape_nodos_trio=solape,
                disp_intra_trio_destino=disp_dest,
                disp_intra_trio_real=disp_real,
            ))
        print(f"  seed {seed:>2}/{max(seeds)}  aristas={n_e} triáng={n_t}  " +
              "  ".join(f"{b.split('_')[0]}={[f['h'] for f in filas if f['seed']==seed and f['brazo']==b][0]:.3f}"
                        for b in BRAZOS_TODOS))
    dt_bat = time.time() - t0_bat
    print(f"\nTiempo total de la batería: {dt_bat:.1f}s\n")

    por_brazo = {b: np.array([f["h"] for f in filas if f["brazo"] == b]) for b in BRAZOS_TODOS}
    h_null = np.array([f["h_null"] for f in filas if f["brazo"] == "A_correcto"])
    h_shuf = np.array([f["h_shuf"] for f in filas if f["brazo"] == "A_correcto"])

    # ---------------- CONTROL DE EQUIPARACIÓN ----------------
    print("── CONTROL DE EQUIPARACIÓN ──")
    f0 = {b: [f for f in filas if f["brazo"] == b][0] for b in BRAZOS_TODOS}
    print(f"  {'brazo':<32} {'DoF':>6} {'sweeps':>7} {'eventos/sweep':>14}")
    for b in BRAZOS_TODOS:
        print(f"  {ETIQUETA[b]:<32} {f0[b]['dof']:>6} {f0[b]['n_sweeps']:>7} {f0[b]['eventos_por_sweep']:>14}")
    mismos = all(len({f["n_sweeps"] for f in filas if f["seed"] == s}) == 1 for s in seeds)
    print(f"  ¿mismos sweeps en los {len(BRAZOS_TODOS)} brazos, las {len(seeds)} semillas? "
          f"{'sí' if mismos else 'NO — revisar'}")
    print("  (D SIN TRÍOS toca n_aristas por sweep en vez de 3×n_tri: es la variable manipulada,")
    print("   igual que en cs083b — se reporta, no se esconde.)\n")

    # ---------------- ORDENAMIENTO EN HOLONOMÍA ----------------
    print(f"── ORDENAMIENTO EN HOLONOMÍA |h| sobre {len(seeds)} semillas (más bajo = más aplanado) ──")
    orden = sorted(BRAZOS_TODOS, key=lambda b: por_brazo[b].mean())
    print(f"  {'#':>2} {'brazo':<32} {'h media':>9} {'DE':>8} {'h mediana':>10} {'min':>8} {'max':>8}"
          f"  {'>A?':>4}  {'pedido?':>8}")
    for i, b in enumerate(orden, 1):
        v = por_brazo[b]
        gana = int(np.sum(v > por_brazo["A_correcto"]))   # nº de semillas en que este brazo aplana MENOS que REAL
        print(f"  {i:>2} {ETIQUETA[b]:<32} {v.mean():>9.4f} {v.std(ddof=1):>8.4f} {np.median(v):>10.4f} "
              f"{v.min():>8.4f} {v.max():>8.4f}  {gana:>2}/{len(v):<2} "
              f"{'sí' if b in BRAZOS_PEDIDOS else 'extra':>8}")
    print("  ('>A?' = en cuántas semillas ese brazo quedó POR ENCIMA de REAL, o sea aplanó menos —")
    print("   test de signos crudo, inmune a semillas atípicas.)")
    print(f"     {'NULL (ruido puro, referencia)':<32} {h_null.mean():>9.4f} {h_null.std(ddof=1):>8.4f}")
    print(f"     {'SHUFFLED (referencia cs082)':<32} {h_shuf.mean():>9.4f} {h_shuf.std(ddof=1):>8.4f}\n")

    # ---------------- COBERTURA Y VARIACIÓN INTRA-TRÍO ----------------
    print("── COBERTURA DE ARISTAS y VARIACIÓN INTRA-TRÍO por brazo (promedio sobre semillas) ──")
    print(f"  {'brazo':<32} {'ar.tocadas':>10} {'%cobert':>8} {'Gini':>7} {'entrop':>7} "
          f"{'%cero':>7} {'max':>5} {'%punter':>8} {'solape':>7} {'disp_dest':>10} {'disp_real':>10}")
    cob_brazo = {}
    for b in orden:
        sub = [f for f in filas if f["brazo"] == b]

        def m(k, _sub=sub):
            vals = [f[k] for f in _sub if not (isinstance(f[k], float) and np.isnan(f[k]))]
            return float(np.mean(vals)) if vals else float("nan")

        cob_brazo[b] = dict(gini=m("gini_cobertura"), ent=m("entropia_cobertura"),
                            cob=m("frac_cobertura"), cero=m("frac_aristas_cero"),
                            disp_real=m("disp_intra_trio_real"), disp_dest=m("disp_intra_trio_destino"))
        print(f"  {ETIQUETA[b]:<32} {m('aristas_tocadas'):>10.1f} {100*m('frac_cobertura'):>7.1f}% "
              f"{m('gini_cobertura'):>7.3f} {m('entropia_cobertura'):>7.3f} "
              f"{100*m('frac_aristas_cero'):>6.1f}% {m('conteo_max'):>5.1f} "
              f"{100*m('frac_punteria'):>7.1f}% "
              f"{m('solape_nodos_trio'):>7.2f} {m('disp_intra_trio_destino'):>10.4f} "
              f"{m('disp_intra_trio_real'):>10.4f}")
    print("  (%punter = 'puntería': qué % de los empujones cae sobre aristas que pertenecen a ≥1")
    print("   triángulo, o sea sobre las aristas que la holonomía efectivamente mide.)")
    print("  (solape: nº de pares —de 3— del trío destino que comparten un nodo. 3=trío coherente,")
    print("   ~0=3 aristas sueltas. disp_dest/disp_real: 1-|media circular| dentro del trío al final")
    print("   de la corrida — 0=las 3 aristas en paz, →1=tiran en direcciones incompatibles.)\n")

    # ---------------- TESTS PAREADOS ----------------
    rng_perm = np.random.default_rng(RNG_MASTER_SEED)
    print(f"── TESTS PAREADOS POR SEMILLA (n={len(seeds)}, {N_PERM} permutaciones sign-flip) ──")
    print("   [bloque 1] cada brazo contra REAL y contra NULL(ruido)")
    tests = {}
    for b in BRAZOS_TODOS:
        if b == "A_correcto":
            continue
        tests[f"A_vs_{b}"] = _reporte_par(f"A(REAL) − {ETIQUETA[b]}", por_brazo["A_correcto"], por_brazo[b],
                                          N_PERM, rng_perm)
    print()
    for b in BRAZOS_TODOS:
        tests[f"{b}_vs_null"] = _reporte_par(f"{ETIQUETA[b]} − NULL(ruido)", por_brazo[b], h_null,
                                             N_PERM, rng_perm)

    print("\n   [bloque 2] EL HALLAZGO RARO: ¿el trío EQUIVOCADO aplana menos que el trío SUELTO?")
    print("   (cs083b con n=20 dio z=−15.62 con C aplanando MÁS que E; se re-testea con n=30)")
    tests["C_vs_E"] = _reporte_par("C(azar) − E(equivocado)   [<0 ⇒ raro se sostiene]",
                                   por_brazo["C_azar"], por_brazo["E_equivocado"], N_PERM, rng_perm)

    print("\n   [bloque 3] EL 2×2 QUE RESUELVE EL RARO — destino(trío real / sueltas) × alineación")
    print("   A=alineado+trío-real · C=alineado+sueltas · E=desalineado+trío-real · F=desalineado+sueltas")
    tests["A_vs_C_"] = _reporte_par("efecto DESTINO con alineación  (A − C)",
                                    por_brazo["A_correcto"], por_brazo["C_azar"], N_PERM, rng_perm)
    tests["E_vs_F_"] = _reporte_par("efecto DESTINO sin alineación  (E − F)",
                                    por_brazo["E_equivocado"], por_brazo["F_azar_desalineado"], N_PERM, rng_perm)
    tests["A_vs_E_"] = _reporte_par("efecto ALINEACIÓN con trío real (A − E)",
                                    por_brazo["A_correcto"], por_brazo["E_equivocado"], N_PERM, rng_perm)
    tests["C_vs_F_"] = _reporte_par("efecto ALINEACIÓN con sueltas   (C − F)",
                                    por_brazo["C_azar"], por_brazo["F_azar_desalineado"], N_PERM, rng_perm)
    inter = ((por_brazo["A_correcto"] - por_brazo["C_azar"]) -
             (por_brazo["E_equivocado"] - por_brazo["F_azar_desalineado"]))
    print(f"  {'INTERACCIÓN (A−C)−(E−F)':<46} media={inter.mean():+.4f}  "
          f"z={inter.mean()/(inter.std(ddof=1)/np.sqrt(len(inter))):+8.2f}")
    tests["interaccion"] = dict(media=float(inter.mean()),
                                z=float(inter.mean() / (inter.std(ddof=1) / np.sqrt(len(inter)))))

    print("\n   [bloque 4] biyección vs sin biyección con trío coherente y alineado (A − G)")
    print("   A y G difieren SÓLO en la cobertura (A: cada trío real destino exactamente 1 vez;")
    print("   G: tríos reales sorteados con reemplazo ⇒ cobertura más dispareja). Mismo mecanismo.")
    tests["A_vs_G_"] = _reporte_par("A(biyección) − G(con reemplazo)",
                                    por_brazo["A_correcto"], por_brazo["G_trio_real_azar"], N_PERM, rng_perm)

    print("\n   [bloque 5] 2×2 CON LA PUNTERÍA IGUALADA — la pregunta residual del bloque 3")
    print("   En C y F las 3 aristas-destino se sortean sobre TODO el grafo, así que ~40% de los")
    print("   empujones cae en aristas que no pertenecen a ningún triángulo (empujones que la holonomía")
    print("   ni siquiera mide). En E, en cambio, el 100% cae sobre aristas de triángulo. H e I repiten")
    print("   C y F pero sorteando SÓLO entre aristas de triángulo: misma puntería que A/E, sin trío")
    print("   coherente. Si E ≈ H, el resto del hallazgo raro es puntería; si E sigue peor que H, hay")
    print("   un efecto propio de 'meterle ruido ajeno a un trío coherente'.")
    tests["I_vs_A_"] = _reporte_par("alineado: A(trío real) − I(sueltas-de-triáng.)",
                                    por_brazo["A_correcto"], por_brazo["I_azarTri_alineado"], N_PERM, rng_perm)
    tests["H_vs_E_"] = _reporte_par("desalineado: H(sueltas-de-triáng.) − E(trío real)",
                                    por_brazo["H_azarTri_desalin"], por_brazo["E_equivocado"], N_PERM, rng_perm)
    tests["C_vs_I_"] = _reporte_par("efecto PUNTERÍA con alineación (C − I)",
                                    por_brazo["C_azar"], por_brazo["I_azarTri_alineado"], N_PERM, rng_perm)
    tests["F_vs_H_"] = _reporte_par("efecto PUNTERÍA sin alineación (F − H)",
                                    por_brazo["F_azar_desalineado"], por_brazo["H_azarTri_desalin"], N_PERM, rng_perm)

    print("\n   [bloque 6] DESCOMPOSICIÓN ADITIVA del hallazgo raro  h(E) − h(C) = "
          f"{por_brazo['E_equivocado'].mean() - por_brazo['C_azar'].mean():+.4f}")
    d_alin = por_brazo["F_azar_desalineado"].mean() - por_brazo["C_azar"].mean()
    d_punt = por_brazo["H_azarTri_desalin"].mean() - por_brazo["F_azar_desalineado"].mean()
    d_coh = por_brazo["E_equivocado"].mean() - por_brazo["H_azarTri_desalin"].mean()
    d_tot = por_brazo["E_equivocado"].mean() - por_brazo["C_azar"].mean()
    for etq, d in (("(1) DESALINEACIÓN fuente↔destino  C→F", d_alin),
                   ("(2) PUNTERÍA (dejar de desperdiciar F→H", d_punt),
                   ("     empujones en aristas sin triángulo)", None),
                   ("(3) COHERENCIA del trío destino   H→E", d_coh)):
        if d is None:
            print(f"  {etq}")
            continue
        print(f"  {etq:<42} Δh={d:+.4f}   {100*d/d_tot:>6.1f}% del hallazgo raro")
    print(f"  {'TOTAL C→E':<42} Δh={d_tot:+.4f}   100.0%")
    tests["descomposicion_raro"] = dict(desalineacion=float(d_alin), punteria=float(d_punt),
                                        coherencia=float(d_coh), total=float(d_tot))

    # ---------------- DESCOMPOSICIÓN ----------------
    gap_total = h_null.mean() - por_brazo["A_correcto"].mean()
    print(f"\n── DESCOMPOSICIÓN (gap_total = h_NULL − h_REAL = {gap_total:+.4f}) ──")
    print(f"  {'brazo':<32} {'h − h_REAL':>11} {'% del gap perdido':>18} {'% que sobrevive':>17}")
    for b in orden:
        if b == "A_correcto":
            continue
        d = por_brazo[b].mean() - por_brazo["A_correcto"].mean()
        print(f"  {ETIQUETA[b]:<32} {d:>11.4f} {100*d/gap_total:>17.1f}% {100*(1-d/gap_total):>16.1f}%")

    # ---------------- CORRELACIÓN COBERTURA ↔ HOLONOMÍA ----------------
    print("\n── ¿LA COBERTURA EXPLICA EL ORDENAMIENTO? (Spearman sobre todas las filas brazo×semilla) ──")
    def col(k, brazos=None, excl=None):
        return [f[k] for f in filas
                if (brazos is None or f["brazo"] in brazos) and (excl is None or f["brazo"] not in excl)]
    variables = ["gini_cobertura", "entropia_cobertura", "frac_cobertura", "frac_aristas_cero",
                 "frac_punteria", "solape_nodos_trio", "disp_intra_trio_real", "disp_intra_trio_destino"]
    print(f"  {'variable':<26} {'ρ (todos)':>14} {'ρ (sin D)':>12} {'ρ (sólo A,C,E,F)':>18}")
    correls = {}
    for v in variables:
        r_all = spearman(col(v), col("h"))
        sin_d = [f for f in filas if f["brazo"] != "D_sin_trios"]
        r_sd = spearman([f[v] for f in sin_d], [f["h"] for f in sin_d])
        cuatro = [f for f in filas if f["brazo"] in ("A_correcto", "C_azar", "E_equivocado", "F_azar_desalineado")]
        r_4 = spearman([f[v] for f in cuatro], [f["h"] for f in cuatro])
        correls[v] = dict(todos=r_all, sin_D=r_sd, cuatro=r_4)
        print(f"  {v:<26} {r_all:>14.3f} {r_sd:>12.3f} {r_4:>18.3f}")
    print("  (ρ sobre filas brazo×semilla: mezcla variación ENTRE brazos —que es lo que interesa acá—")
    print("   con la variación entre semillas dentro de cada brazo, que es chica.)")

    print("\n── contraste directo de la hipótesis de COBERTURA sobre el par del hallazgo raro ──")
    gA, gC, gE, gF = (cob_brazo["A_correcto"]["gini"], cob_brazo["C_azar"]["gini"],
                      cob_brazo["E_equivocado"]["gini"], cob_brazo["F_azar_desalineado"]["gini"])
    print(f"  Gini de cobertura:  A={gA:.4f}  C={gC:.4f}  E={gE:.4f}  F={gF:.4f}")
    print(f"  ¿Gini(A) == Gini(E) exactamente? "
          f"{'sí — la biyección da la MISMA cobertura que REAL, arista por arista' if abs(gA-gE) < 1e-12 else 'NO'}")
    print(f"  ¿Gini(C) ≈ Gini(F)? {'sí — mismo esquema de sorteo' if abs(gC-gF) < 0.01 else 'NO'}"
          f"  (|Δ|={abs(gC-gF):.4f}; difieren sólo por el sorteo concreto, no por el esquema)")
    print("  Si E aplana menos que C pero F (misma cobertura que C) aplana ~igual que E, entonces la")
    print("  cobertura NO es la causa: lo es la desalineación fuente↔destino.")

    return filas, tests, correls


# ══════════════════════════ CSV / GRÁFICO ══════════════════════════
def guardar_csv(filas, path):
    campos = list(filas[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for f in filas:
            w.writerow(f)
    print(f"\nCSV crudo guardado en: {path}")


# El color codifica el FACTOR que resultó explicativo (la alineación fuente↔destino), no la identidad
# de cada brazo: con 9 brazos, 9 colores serían indistinguibles. La identidad va SIEMPRE por la letra
# del brazo, escrita directamente sobre el gráfico — nunca sólo por color. Paleta verificada con el
# validador del proyecto (bandas de luminosidad, piso de croma, separación para daltonismo y contraste
# contra el fondo: los 3 colores pasan en modo claro y oscuro).
FAMILIA = {
    "A_correcto": "alineado", "B_casi_correcto": "alineado", "C_azar": "alineado",
    "G_trio_real_azar": "alineado", "I_azarTri_alineado": "alineado",
    "E_equivocado": "desalineado", "F_azar_desalineado": "desalineado",
    "H_azarTri_desalin": "desalineado",
    "D_sin_trios": "sin tríos",
}
COLOR_FAMILIA = {"alineado": "#2563eb", "desalineado": "#dc2626", "sin tríos": "#7c3aed"}
TINTA = "#1f2937"
TINTA_SUAVE = "#6b7280"


def graficar(filas, path):
    """4 paneles: (1) holonomía por brazo; (2) Gini de cobertura vs holonomía; (3) puntería vs
    holonomía; (4) variación intra-trío vs holonomía. Los paneles 2-4 son el test visual de la
    hipótesis de esta tarea: si el ordenamiento de los brazos siguiera a la cobertura, los puntos de
    esos paneles caerían sobre una línea; si no, la cobertura no es la explicación."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:                       # pragma: no cover
        print(f"(sin gráfico: matplotlib no disponible — {exc})")
        return

    brazos = BRAZOS_TODOS
    letra = {b: b.split("_")[0] for b in brazos}
    col = {b: COLOR_FAMILIA[FAMILIA[b]] for b in brazos}
    n_seeds = len({f["seed"] for f in filas})

    def vals(b, k):
        return np.array([f[k] for f in filas if f["brazo"] == b], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    fig.patch.set_facecolor("white")
    rng = np.random.default_rng(0)

    # ── panel 1: holonomía por brazo, ordenado ──────────────────────────────────────────────
    ax = axes[0][0]
    orden = sorted(brazos, key=lambda b: vals(b, "h").mean())
    for i, b in enumerate(orden):
        v = vals(b, "h")
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.14, 0.14, len(v)), v,
                   s=13, alpha=0.45, color=col[b], edgecolors="none", zorder=2)
        ax.hlines(v.mean(), i - 0.32, i + 0.32, color=col[b], lw=2.8, zorder=3)
    hn = vals("A_correcto", "h_null").mean()
    ax.axhline(hn, ls="--", lw=1.1, color=TINTA_SUAVE, zorder=1)
    ax.text(len(orden) - 0.4, hn - 0.09, "NULL — ruido puro", fontsize=8.5,
            color=TINTA_SUAVE, ha="right")
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels([letra[b] for b in orden], fontsize=10, color=TINTA)
    ax.set_ylabel("holonomía media |h|   (más bajo = más aplanado)", fontsize=9.5, color=TINTA)
    ax.set_title(f"1 · Ordenamiento de los brazos  (n={n_seeds} semillas, punto = 1 semilla)",
                 fontsize=11, color=TINTA, loc="left")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    # ── paneles 2-4: descriptor vs holonomía, media por brazo con barras de dispersión ───────
    def panel_dispersion(ax, campo, xlabel, titulo, incluir_D=True):
        for b in brazos:
            if not incluir_D and b == "D_sin_trios":
                continue
            x, y = vals(b, campo), vals(b, "h")
            if np.all(np.isnan(x)):
                continue
            ax.scatter(x, y, s=11, alpha=0.28, color=col[b], edgecolors="none", zorder=2)
            mx, my = np.nanmean(x), np.nanmean(y)
            ax.errorbar(mx, my, yerr=np.nanstd(y, ddof=1), xerr=np.nanstd(x, ddof=1),
                        fmt="o", ms=8, color=col[b], ecolor=col[b], elinewidth=1.1,
                        capsize=2.5, mec="white", mew=1.6, zorder=4)
            ax.annotate(letra[b], (mx, my), textcoords="offset points", xytext=(9, 6),
                        fontsize=10.5, fontweight="bold", color=TINTA, zorder=5)
        ax.set_xlabel(xlabel, fontsize=9.5, color=TINTA)
        ax.set_ylabel("holonomía media |h|", fontsize=9.5, color=TINTA)
        ax.set_title(titulo, fontsize=11, color=TINTA, loc="left")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)

    panel_dispersion(
        axes[0][1], "gini_cobertura",
        "Gini de la cobertura de aristas   (0 = todas las aristas reciben lo mismo)",
        "2 · ¿La cobertura explica el orden?  (A…I casi apilados en x ⇒ no)")
    panel_dispersion(
        axes[1][0], "frac_punteria",
        "puntería: fracción de empujones que cae sobre aristas de algún triángulo",
        "3 · Puntería vs holonomía", incluir_D=True)
    panel_dispersion(
        axes[1][1], "disp_intra_trio_real",
        "variación intra-trío al final  (1 − |media circular| de las 3 aristas)",
        "4 · 'Estrés estructural' intra-trío vs holonomía")

    handles = [Line2D([], [], marker="o", ls="", ms=8, mfc=c, mec="white", mew=1.4, label=f)
               for f, c in COLOR_FAMILIA.items()]
    axes[0][0].legend(handles=handles, fontsize=9, loc="upper left", framealpha=0.95,
                      title="fuente del defecto ↔ destino del empujón", title_fontsize=8.5)

    partes = [f"{letra[b]} = {ETIQUETA[b][2:]}" for b in brazos]
    fig.suptitle("CS083c / O1-D — factorial del trío sobre el sustrato 4 "
                 "(2-complejo con retroalimentación cara→arista)",
                 fontsize=13, color=TINTA, y=0.988)
    fig.text(0.5, 0.960, "   ·   ".join(partes[:5]), ha="center", fontsize=8.2, color=TINTA_SUAVE)
    fig.text(0.5, 0.940, "   ·   ".join(partes[5:]), ha="center", fontsize=8.2, color=TINTA_SUAVE)
    fig.tight_layout(rect=(0, 0, 1, 0.928))
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"Gráfico guardado en: {path}")


def _leer_csv(path):
    """Relee un CSV ya generado (modo `replot`) para poder rehacer el gráfico sin volver a correr las
    simulaciones — los números no cambian, sólo el dibujo."""
    filas = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            fila = {}
            for k, v in r.items():
                if k in ("brazo", "etiqueta"):
                    fila[k] = v
                else:
                    try:
                        fila[k] = float(v)
                    except ValueError:
                        fila[k] = float("nan")
            fila["seed"] = int(fila["seed"])
            filas.append(fila)
    return filas


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "full"

    if modo == "replot":
        filas = _leer_csv(str(RAIZ / "cs083c_resultados.csv"))
        graficar(filas, str(RAIZ / "cs083c_factorial_trio.png"))
        return

    if modo == "pilot":
        print("── VERIFICACIÓN BIT A BIT del motor genérico contra cs082/cs083/cs083b (seed=1) ──")
        for k, v in _verificar_reproduccion(1).items():
            print(f"  {k:<44} {'OK (idéntico)' if v else '*** DISTINTO — BUG ***'}")
        print("\n── VERIFICACIÓN de la observación analítica (biyección + defecto del destino = REAL) ──")
        ig, cob_ig, rA, rE = _verificar_biyeccion_equivale_a_real(1)
        print(f"  destino rotado (biyección) + defecto del destino == REAL bit a bit:  "
              f"{'sí' if ig else 'NO'}")
        print(f"  cobertura de E(derangement) idéntica a la de A(REAL):                "
              f"{'sí' if cob_ig else 'NO'}")
        print(f"    A: aristas_tocadas={rA['aristas_tocadas']} Gini={rA['gini_cobertura']:.4f} "
              f"entropía={rA['entropia_cobertura']:.4f}")
        print(f"    E: aristas_tocadas={rE['aristas_tocadas']} Gini={rE['gini_cobertura']:.4f} "
              f"entropía={rE['entropia_cobertura']:.4f}")
        print()
        filas, _, _ = correr_bateria(SEEDS_PILOT, tag="(PILOTO, 5 semillas)")
        guardar_csv(filas, str(RAIZ / "cs083c_resultados_piloto.csv"))
    else:
        filas, _, _ = correr_bateria(SEEDS_FULL, tag=f"(COMPLETO, {N_SEEDS_FULL} semillas)")
        guardar_csv(filas, str(RAIZ / "cs083c_resultados.csv"))
        graficar(filas, str(RAIZ / "cs083c_factorial_trio.png"))

    print("\nFin. Ver FASE6_O1D_factorial_trio_CS.md para la interpretación (lectura de Alexis pendiente).")


if __name__ == "__main__":
    main()
