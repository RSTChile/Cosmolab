"""
CS082 — FASE IV: RELACIÓN DE ORDEN SUPERIOR — 4 sustratos (grafo / hipergrafo / simplicial / 2-complejo)
=========================================================================================================
QUIÉN SOY: primer ataque de Fase IV del roadmap consolidado (5 analistas IA, 5-ago-2026). Pregunta:
¿la ARIDAD de la relación (de-a-dos vs de-a-tres-o-más) es lo que estuvo bloqueado en la Pared R7, y
no la física concreta (gluón/Higgs)? Se prueba con una batería IDÉNTICA en 4 sustratos relacionales
puros — sin grupos gauge, sin geometría conocida de antemano, sin coordenadas de embedding (guardián
NO-HORNEAR de siempre). No es un motor físico tipo Phantom: es sobre si la ESTRUCTURA relacional pura
(holonomía, ejes, conexión) cambia cuando una relación puede tocar 3+ nodos a la vez, o relacionarse
con OTRA relación (no sólo con nodos).

────────────────────────────────────────────────────────────────────────────────────────────────────
RECORDATORIO — LA PARED R7 (no repetir el mismo camino; ver PUERTA_R7_espin_como_marco.md,
SINTESIS_ARCO_R7.md, GROK_TO_CC.md Msg 38-41, REGISTRO_EXPERIMENTOS_CS.md CS032):
  · El motor CG002 usa compatibilidad PAREADA c_ij (producto interno de a dos). Lo abeliano (color r7a,
    carga r7d/e) funciona ahí — polarización emergente real, ley+historia, sin CP inyectada.
  · El gluón-entidad (r7b) y el Higgs/Yukawa (r7f) son vértices DE 3 PUNTOS (q→q'+g ; f-H-f'). Un
    vértice de 3 puntos NO se reduce a un escalar pareado c_ij → colapso diagonal (r7b: gluones 100%
    inertes) / desacople (r7f: Higgs Σ|c|=0). Es LA MISMA puerta bloqueada, tocada dos veces.
  · CS032/R7g YA INTENTÓ extender esto: añadió un término v_ijk·m_j·m_k SUMADO al c_ij pareado, sobre
    la MISMA estructura de grafo (el "3-cuerpo" era una fuerza extra, no un objeto-relación distinto).
    Resultado (Grok Msg 38-40, retractado y re-auditado): "la pared se MOVIÓ, no CAYÓ" — hubo acople
    donde antes había cero, pero el octeto seguía muriendo (~0-1% superviv.), f≈0.30 sin cambio real,
    y el ratio de masa quedó en 1.001 (sin jerarquía). CONCLUSIÓN DEL EQUIPO: no repetir "sumar un
    término 3-cuerpo a un grafo pareado" — hace falta un OBJETO-RELACIÓN cuya aridad nativa sea 3+, no
    una fuerza extra sobre el mismo objeto de 2.
  · Por eso Fase IV NO reintenta v_ijk sobre grafo. Construye estructuras de datos donde la relación
    MISMA vive sobre 3+ nodos (hiperarista) o donde una relación se relaciona con otra relación (cara↔
    arista) — la distinción de aridad que el roadmap pide, no un parche de fuerza.

────────────────────────────────────────────────────────────────────────────────────────────────────
LOS 4 SUSTRATOS (misma base combinatoria N nodos + mismo grafo aleatorio, para que la única variable
manipulada sea la ARIDAD/estructura de la relación, no la topología subyacente):

  1) GRAFO DIÁDICO (línea base, de-a-pares): la relación es la ARISTA (i,j). Cada arista lleva un
     estado de orientación s∈Z_K (el "hacia dónde" del espín, sin dirección espacial — es un marco
     LOCAL, no una coordenada). Una arista sólo puede interactuar con OTRA arista sola a la vez
     (comparten un nodo) — el mismo límite estructural que c_ij pareado.

  2) HIPERGRAFO (aridad 3, de-a-tres SIMULTÁNEO): la relación es un TRIÁNGULO {i,j,k} del mismo grafo
     base, tratado como UN SOLO objeto-relación que toca 3 nodos A LA VEZ (no 3 aristas separadas). Su
     estado s∈Z_K se actualiza mirando el campo conjunto de los 3 nodos en UNA sola relación — algo que
     un objeto-arista (que sólo tiene 2 extremos) no puede representar por construcción.

  3) COMPLEJO SIMPLICIAL (triángulos = caras/2-simplices, medición pasiva de borde): mismas aristas y
     misma dinámica que (1), pero además se computa la holonomía de CADA cara (triángulo) como la suma
     de sus 3 aristas de borde mod K (igual que el "brazo C" de cs052_v1/cg004f3 — Burgers/déficit). La
     cara MIDE la relación-entre-relaciones (sus 3 aristas) pero no actúa sobre ellas — control de
     "sólo estructura, sin retroalimentación".

  4) 2-COMPLEJO CON ESTADOS EN ENLACES Y EN CARAS (retroalimentación cara→arista): igual que (3), pero
     el defecto de cada cara SÍ empuja de vuelta a sus 3 aristas de borde (intenta cerrarlas). Acá una
     relación (la cara, que agrega 3 aristas) actúa CAUSALMENTE sobre otras relaciones (esas 3 aristas)
     — la forma más completa de "una relación se relaciona con otra relación" del roadmap.

MEDICIÓN COMÚN (para que los 4 sean comparables): se proyecta el estado de cada sustrato sobre un
"campo de aristas" E[i,j] —nativo en 1/3/4; en 2 se promedian las hiperaristas que contienen (i,j)—, y
se mide la holonomía de los MISMOS triángulos (misma lista, mismo K) en los 4 sustratos: holon(t) =
(E_ij+E_jk+E_ki) mod K, centrada a [-K/2,K/2]. Igual que Burgers/déficit de cg004f3, pero sin caras
euclídeas prefijadas — el K=6 y la convención de sextante es la misma del proyecto (cs052/cg004f3), no
una física importada.

GUARDIÁN NO-HORNEAR: en ningún punto del código hay una coordenada (x,y) que alimente la dinámica o la
medición — sólo adyacencia (quién-con-quién) y estado relacional (Z_K). Verificable por inspección: no
existe ningún arreglo de posiciones en este archivo.

CONTROLES (anti-Shannon, disciplina del proyecto):
  · NULL   = mecanismo apagado: el campo de aristas final se reemplaza por ruido uniforme iid en Z_K
             (misma distribución marginal, cero dinámica).
  · SHUFFLED = estructura barajada: se usan los MISMOS valores finales reales, pero permutados al azar
             entre aristas (mismos valores, topología rota).
  Si REAL no separa de NULL/SHUFFLED, no hay estructura relacional genuina — sólo ruido con la forma de
  la distribución.

CONTROL DE EQUIPARACIÓN (parámetros iguales entre los 4, reportado en tabla, ver imprimir_tabla_control):
  mismo N, mismo grafo base (mismo seed → misma adyacencia), mismo alfabeto K=6, misma constante de
  acople J, mismo ruido por sweep, mismo PRESUPUESTO DE CÓMPUTO total (nº de actualizaciones de
  relación) — el número de sweeps se ajusta por sustrato para que el TOTAL de operaciones-relación sea
  igual, así ningún sustrato gana sólo por "más recursos". Los grados de libertad (nº de variables Z_K)
  SÍ difieren entre sustratos —es la variable manipulada (aridad/estructura), no un descuido— y se
  reportan explícitamente para que la comparación sea auditable, no escondida.

Código nuevo, autodescriptivo, numpy-only. No toca ningún script existente del proyecto (cg002_r7*,
cs052_v1_coemergencia.py, cg004f3_cinta_eisenstein.py se leen sólo como referencia de convención, no se
importan ni se modifican). No declara cierre — reporta números; veredicto es de Alexis.
"""
from __future__ import annotations

import time
import numpy as np
from collections import defaultdict

# ============================ CONFIG (igual entre los 4 sustratos) ============================
N            = 110          # nodos base (igual en los 4)
P_EDGE       = 0.09         # prob. arista Erdos-Renyi del grafo base (igual; da ~540 aristas, ~200 triáng.)
K            = 6            # alfabeto cíclico de orientación (convención sextante del proyecto, cs052/cg004f3)
J            = 0.6          # constante de acople (igual en los 4)
NOISE        = 0.25         # ruido gaussiano por sweep, en unidades de K (igual en los 4)
J_FACE       = 0.5          # sólo sustrato 4: fuerza de la retroalimentación cara→arista
COMPUTE_BUDGET = 60_000     # operaciones-relación TOTALES por sustrato (igual: controla complejidad algorítmica)
SEEDS        = [1, 2, 3, 4, 5]
# ===============================================================================================


def construir_base(seed):
    """Grafo aleatorio base (Erdos-Renyi) + lista de triángulos. SIN coordenadas — sólo adyacencia
    (guardián no-hornear). Idéntico para los 4 sustratos en un mismo seed: la única variable que
    cambia entre sustratos es qué OBJETO-RELACIÓN se define sobre esta misma base."""
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(N)]
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < P_EDGE:
                adj[i].add(j); adj[j].add(i); edges.append((i, j))
    triangles = []
    for i in range(N):
        vs = sorted(x for x in adj[i] if x > i)
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                if vs[b] in adj[vs[a]]:
                    triangles.append((i, vs[a], vs[b]))
    return adj, edges, triangles


def _linea_adyacencia(relaciones):
    """Line-graph: dos relaciones son 'vecinas' si comparten AL MENOS UN nodo. Mismo criterio para
    aristas (sustrato 1) y para hiperaristas/triángulos (sustrato 2) — la única diferencia entre
    sustratos es cuántos nodos toca CADA relación (2 vs 3), no la regla de vecindad entre relaciones."""
    por_nodo = defaultdict(list)
    for idx, r in enumerate(relaciones):
        for v in r:
            por_nodo[v].append(idx)
    vecinos = [set() for _ in relaciones]
    for v, idxs in por_nodo.items():
        for a in idxs:
            for b in idxs:
                if a != b:
                    vecinos[a].add(b)
    return [list(v) for v in vecinos]


def _circ_mean_update(s, vecinos, J, noise, rng):
    """Un sweep de alineación circular (Kuramoto/Potts discreto en Z_K): cada relación se mueve hacia
    la media circular de sus vecinas relacionales, con ruido. MISMA fórmula para los 4 sustratos —
    la única diferencia entre sustratos es SOBRE QUÉ OBJETOS se aplica (aristas vs hiperaristas) y si
    hay retroalimentación cara→arista extra (sustrato 4). No importa ningún grupo gauge: es la
    operación mínima "las relaciones vecinas tienden a ponerse de acuerdo en su orientación"."""
    n = len(s)
    ang = 2 * np.pi * s / K
    z = np.exp(1j * ang)
    s_new = s.copy()
    for i in range(n):
        vs = vecinos[i]
        if not vs:
            continue
        z_vec = (1 - J) * z[i] + J * np.mean(z[vs])
        ang_new = np.angle(z_vec)
        s_new[i] = (ang_new / (2 * np.pi) * K) % K
    s_new = (s_new + rng.normal(0, noise, n)) % K
    return s_new


def _holonomia_triangulos(E, triangles):
    """Holonomía de CADA triángulo, medida SIEMPRE de la misma forma en los 4 sustratos: suma de los
    3 tramos de arista mod K, centrada a [-K/2,K/2] (misma convención que brazo-C de cs052_v1 /
    Burgers de cg004f3, sin caras euclídeas prefijadas — acá K=6 es sólo el alfabeto de orientación).
    E es un dict {(i,j): valor} — 'campo de aristas' nativo o proyectado según el sustrato."""
    vals = []
    for (i, j, k) in triangles:
        eij = E.get((i, j)) if i < j else E.get((j, i))
        ejk = E.get((j, k)) if j < k else E.get((k, j))
        eki = E.get((k, i)) if k < i else E.get((i, k))
        if eij is None or ejk is None or eki is None:
            continue
        h = (eij + ejk + eki) % K
        h = h - K if h > K / 2 else h
        vals.append(abs(h))
    return np.array(vals) if vals else np.array([0.0])


def _contar_ejes(valores, k=K, n_bins=12, umbral_rel=0.15):
    """Cuenta 'ejes' estabilizados = modos (picos) del histograma circular de orientaciones finales.
    Un eje = un bin cuya densidad supera umbral_rel * densidad_máxima Y es un máximo local (más alto
    que sus 2 vecinos circulares). Mide si el sustrato converge a UNA orientación dominante, a varias
    (multi-eje) o a ninguna (uniforme/ruido)."""
    if len(valores) == 0:
        return 0
    hist, _ = np.histogram(np.mod(valores, k), bins=n_bins, range=(0, k))
    if hist.sum() == 0:
        return 0
    dens = hist / hist.sum()
    picos = 0
    for i in range(n_bins):
        izq = dens[(i - 1) % n_bins]; der = dens[(i + 1) % n_bins]
        if dens[i] >= umbral_rel * dens.max() and dens[i] >= izq and dens[i] >= der and dens[i] > 0:
            picos += 1
    return picos


def _n_sweeps_para_presupuesto(n_relaciones, presupuesto):
    return max(5, presupuesto // max(1, n_relaciones))


# ============================ SUSTRATO 1 — GRAFO DIÁDICO ============================
def correr_sustrato_1_grafo(adj, edges, triangles, seed):
    rng = np.random.default_rng(10_000 + seed)
    n_edges = len(edges)
    idx_edge = {e: i for i, e in enumerate(edges)}
    vecinos = _linea_adyacencia(edges)
    s = rng.uniform(0, K, n_edges)
    n_sweeps = _n_sweeps_para_presupuesto(n_edges, COMPUTE_BUDGET)
    t0 = time.time()
    for _ in range(n_sweeps):
        s = _circ_mean_update(s, vecinos, J, NOISE, rng)
    dt = time.time() - t0
    E = {e: s[idx_edge[e]] for e in edges}
    dof = n_edges
    return E, dof, n_sweeps, dt


# ============================ SUSTRATO 2 — HIPERGRAFO (aridad 3 nativa) ============================
def correr_sustrato_2_hipergrafo(adj, edges, triangles, seed):
    rng = np.random.default_rng(20_000 + seed)
    n_tri = len(triangles)
    if n_tri == 0:
        return {}, 0, 0, 0.0
    vecinos = _linea_adyacencia(triangles)   # hiperaristas vecinas si comparten >=1 nodo
    s = rng.uniform(0, K, n_tri)
    n_sweeps = _n_sweeps_para_presupuesto(n_tri, COMPUTE_BUDGET)
    t0 = time.time()
    for _ in range(n_sweeps):
        s = _circ_mean_update(s, vecinos, J, NOISE, rng)
    dt = time.time() - t0
    # proyección a campo de aristas para medición común: promedio circular de las hiperaristas
    # que contienen ambos extremos de (i,j) — una arista puede recibir aporte de VARIAS hiperaristas
    # que comparten un tercer nodo distinto, algo que el sustrato 1 no puede representar en un solo
    # objeto-relación.
    acumulador = defaultdict(list)
    for idx, (i, j, k) in enumerate(triangles):
        acumulador[(i, j)].append(s[idx]); acumulador[(j, k)].append(s[idx]); acumulador[(i, k)].append(s[idx])
    E = {}
    for e, vals in acumulador.items():
        z = np.mean(np.exp(1j * 2 * np.pi * np.array(vals) / K))
        E[e] = (np.angle(z) / (2 * np.pi) * K) % K
    dof = n_tri
    return E, dof, n_sweeps, dt


# ============================ SUSTRATO 3 — COMPLEJO SIMPLICIAL (cara mide, no empuja) ============
def correr_sustrato_3_simplicial(adj, edges, triangles, seed):
    rng = np.random.default_rng(30_000 + seed)
    n_edges = len(edges)
    idx_edge = {e: i for i, e in enumerate(edges)}
    vecinos = _linea_adyacencia(edges)
    s = rng.uniform(0, K, n_edges)
    dof = n_edges + len(triangles)     # aristas + caras (caras son medición derivada, pero cuentan
                                        # como variable de estado del sustrato para la tabla de DoF)
    n_sweeps = _n_sweeps_para_presupuesto(n_edges, COMPUTE_BUDGET)
    t0 = time.time()
    for _ in range(n_sweeps):
        s = _circ_mean_update(s, vecinos, J, NOISE, rng)
    dt = time.time() - t0
    E = {e: s[idx_edge[e]] for e in edges}
    return E, dof, n_sweeps, dt


# ============================ SUSTRATO 4 — 2-COMPLEJO (cara↔arista, retroalimentación) ============
def correr_sustrato_4_2complejo(adj, edges, triangles, seed):
    rng = np.random.default_rng(40_000 + seed)
    n_edges = len(edges)
    idx_edge = {e: i for i, e in enumerate(edges)}
    vecinos = _linea_adyacencia(edges)
    s = rng.uniform(0, K, n_edges)
    dof = n_edges + len(triangles)
    n_sweeps = _n_sweeps_para_presupuesto(n_edges, COMPUTE_BUDGET)
    # pre-indexar qué aristas bordean cada triángulo, para la retroalimentación cara→arista
    tri_edges = []
    for (i, j, k) in triangles:
        e1 = (i, j) if i < j else (j, i)
        e2 = (j, k) if j < k else (k, j)
        e3 = (i, k) if i < k else (k, i)
        tri_edges.append((idx_edge[e1], idx_edge[e2], idx_edge[e3]))
    t0 = time.time()
    for _ in range(n_sweeps):
        s = _circ_mean_update(s, vecinos, J, NOISE, rng)
        if tri_edges:
            # defecto de cada cara = holonomía de sus 3 aristas (mod K, centrada) — LA RELACIÓN
            # (cara) agrega a sus 3 relaciones-borde y empuja de vuelta para reducir el defecto:
            # esto es lo que ningún sustrato pareado puede hacer (una arista no puede "escuchar"
            # a una cara, sólo a otra arista).
            correccion = np.zeros(n_edges)
            cuenta = np.zeros(n_edges)
            for (a, b, c) in tri_edges:
                h = (s[a] + s[b] + s[c]) % K
                h = h - K if h > K / 2 else h     # defecto centrado en [-K/2,K/2]
                for idx in (a, b, c):
                    correccion[idx] += -J_FACE * h / 3.0
                    cuenta[idx] += 1
            mask = cuenta > 0
            s[mask] = (s[mask] + correccion[mask] / cuenta[mask]) % K
    dt = time.time() - t0
    E = {e: s[idx_edge[e]] for e in edges}
    return E, dof, n_sweeps, dt


# ============================ NULL / SHUFFLED ============================
def null_de(E, seed_extra):
    rng = np.random.default_rng(90_000 + seed_extra)
    return {e: rng.uniform(0, K) for e in E}


def shuffled_de(E, seed_extra):
    rng = np.random.default_rng(95_000 + seed_extra)
    keys = list(E.keys())
    vals = [E[k] for k in keys]
    rng.shuffle(vals)
    return {k: v for k, v in zip(keys, vals)}


# ============================ BATERÍA PRINCIPAL ============================
SUSTRATOS = {
    "1_grafo_diadico":      correr_sustrato_1_grafo,
    "2_hipergrafo":         correr_sustrato_2_hipergrafo,
    "3_simplicial":         correr_sustrato_3_simplicial,
    "4_2complejo_feedback": correr_sustrato_4_2complejo,
}


def main():
    print("CS082 — FASE IV: 4 sustratos de relación de orden superior")
    print("=" * 100)
    print(f"N={N} nodos · K={K} (alfabeto orientación) · J={J} · ruido={NOISE} · presupuesto cómputo={COMPUTE_BUDGET} "
          f"op-relación/sustrato · seeds={SEEDS}")
    print("Guardián no-hornear: sin coordenadas en ningún punto del código (verificable por inspección).\n")

    filas = []       # para tabla final
    dof_reportado = {}
    for seed in SEEDS:
        adj, edges, triangles = construir_base(seed)
        n_e, n_t = len(edges), len(triangles)
        for nombre, fn in SUSTRATOS.items():
            E, dof, n_sweeps, dt = fn(adj, edges, triangles, seed)
            if not E:
                print(f"  seed={seed} {nombre}: sin relaciones (grafo demasiado disperso) — salteado")
                continue
            h_real = _holonomia_triangulos(E, triangles)
            E_null = null_de(E, seed)
            h_null = _holonomia_triangulos(E_null, triangles)
            E_shuf = shuffled_de(E, seed)
            h_shuf = _holonomia_triangulos(E_shuf, triangles)

            ejes_real = _contar_ejes(np.array(list(E.values())))
            ejes_null = _contar_ejes(np.array(list(E_null.values())))

            filas.append(dict(
                seed=seed, sustrato=nombre, dof=dof, n_sweeps=n_sweeps, dt=dt,
                n_edges_base=n_e, n_tri_base=n_t,
                h_real=h_real.mean(), h_null=h_null.mean(), h_shuf=h_shuf.mean(),
                ejes_real=ejes_real, ejes_null=ejes_null,
            ))
            dof_reportado[nombre] = dof

    # ---------------- TABLA DE CONTROL DE EQUIPARACIÓN ----------------
    print("── CONTROL DE EQUIPARACIÓN (parámetros iguales salvo la variable manipulada = aridad) ──")
    print(f"  {'sustrato':<22} {'DoF(Z_K)':>9} {'sweeps':>7} {'op-relación total':>18} {'runtime(s) prom':>16}")
    for nombre in SUSTRATOS:
        sub = [f for f in filas if f["sustrato"] == nombre]
        if not sub:
            continue
        dof = sub[0]["dof"]; sw = int(np.mean([f["n_sweeps"] for f in sub]))
        op_total = dof * sw
        dt_prom = np.mean([f["dt"] for f in sub])
        print(f"  {nombre:<22} {dof:>9} {sw:>7} {op_total:>18} {dt_prom:>16.3f}")
    print("  (DoF difiere por diseño — es la aridad, la variable manipulada. El presupuesto de cómputo\n"
          "   [op-relación total] se mantiene ~igual entre sustratos vía nº de sweeps, para que ningún\n"
          "   sustrato gane sólo por 'más recursos'. J, K, ruido y grafo-base [mismo seed] son idénticos.)\n")

    # ---------------- TABLA COMPARATIVA DE RESULTADOS (promedio sobre seeds) ----------------
    print("── HOLONOMÍA (|h| promedio sobre triángulos y seeds) — REAL vs NULL vs SHUFFLED ──")
    print(f"  {'sustrato':<22} {'h_REAL':>8} {'h_NULL':>8} {'h_SHUF':>8} {'REAL/NULL':>10} {'ejes_REAL':>10} {'ejes_NULL':>10}")
    resumen = {}
    for nombre in SUSTRATOS:
        sub = [f for f in filas if f["sustrato"] == nombre]
        if not sub:
            continue
        hr = np.mean([f["h_real"] for f in sub]); hn = np.mean([f["h_null"] for f in sub])
        hs = np.mean([f["h_shuf"] for f in sub])
        er = np.mean([f["ejes_real"] for f in sub]); en = np.mean([f["ejes_null"] for f in sub])
        hr_std_seed = np.std([f["h_real"] for f in sub])   # estabilidad entre seeds
        ratio = hr / hn if hn > 1e-9 else float("nan")
        resumen[nombre] = dict(h_real=hr, h_null=hn, h_shuf=hs, ratio=ratio, ejes_real=er, ejes_null=en,
                                h_real_std_seed=hr_std_seed)
        print(f"  {nombre:<22} {hr:>8.3f} {hn:>8.3f} {hs:>8.3f} {ratio:>10.3f} {er:>10.1f} {en:>10.1f}")

    print("\n── ESTABILIDAD ENTRE SEEDS de h_REAL (proxy de 'conexión independiente de la métrica':")
    print("   si la estructura fuera ruido de embedding, variaría mucho seed a seed; si es relacional")
    print("   genuina, debería repetirse) ──")
    for nombre, r in resumen.items():
        print(f"  {nombre:<22} std_entre_seeds={r['h_real_std_seed']:.4f}  (h_REAL prom={r['h_real']:.3f})")

    # ---------------- VEREDICTO NUMÉRICO (sin cerrar el experimento) ----------------
    print("\n── LECTURA (números, sin cierre — veredicto lo da Alexis) ──")
    print("  Nota: 'separa_de_NULL' mira |h_REAL-h_NULL| en CUALQUIER dirección — ratio<1 significa que")
    print("  la dinámica APLANA (cierra los lazos de 3 más que el azar); ratio>1 significa que los RUGOSA")
    print("  (más frustración que el azar). Ambas son señal real si el efecto supera el ruido entre seeds.\n")
    base = resumen.get("1_grafo_diadico")
    for nombre, r in resumen.items():
        d_holon = r["h_real"] - base["h_real"] if nombre != "1_grafo_diadico" else 0.0
        d_ejes = r["ejes_real"] - base["ejes_real"] if nombre != "1_grafo_diadico" else 0.0
        efecto = abs(r["h_real"] - r["h_null"])
        separa = efecto > 2 * max(r["h_real_std_seed"], 1e-6)
        direccion = "APLANA (h_REAL<h_NULL)" if r["h_real"] < r["h_null"] else "RUGOSA (h_REAL>h_NULL)"
        etiqueta = f" | vs diádico: Δholon={d_holon:+.3f} Δejes={d_ejes:+.1f}" if nombre != "1_grafo_diadico" else ""
        print(f"  {nombre:<22} ratio={r['ratio']:.3f}  {direccion:<24} separa_de_NULL={'sí' if separa else 'no'}{etiqueta}")

    print("\nFin de la batería. Ver FASE4_orden_superior_resultado_CS.md para interpretación completa.")


if __name__ == "__main__":
    main()
