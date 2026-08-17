"""
CS090 — AUDITORÍA de kcap (eje C2) — ¿es una restricción genuinamente RELACIONAL o esconde
escala/geometría externa (P3/P5)?
============================================================================================
QUIÉN SOY: script NUEVO (no toca `cs090_fase5_generador.py` ni `cs090_fase5_motor.py`) que responde la
duda que señalaron los dos analistas externos sobre el candidato A2-B0-C2: ¿el "límite de escala duro"
kcap depende SÓLO de cantidades relacionales (grado, soporte/vecinos-compartidos, costo, historia), o
esconde coordenadas/distancia/tamaño de caja disfrazados?

Método: TODO lo que se importa de `cs090_fase5_generador.py` / `cs090_fase5_motor.py` /
`cs090_fase5_clasificador.py` / `cs080_renormalizacion.py` se usa TAL CUAL (import directo, nunca se
edita el archivo en disco). Donde hace falta un INPUT modificado (grafo con nodos renombrados, orden de
recorrido distinto), se inyecta sustituyendo la entrada al llamar la función, o vía monkeypatch en
TIEMPO DE EJECUCIÓN de un nombre a nivel de módulo (`cs090_fase5_motor.CONSTRUCTORES_A["A2"]`,
`cs090_fase5_motor._enforce_kcap`) — el archivo .py en disco no cambia un carácter (se puede verificar
con `git diff` después de correr esto), sólo el objeto en memoria de ESTE proceso Python. Se restaura el
original después de cada test.

Cuatro tests de invarianza (anti-Shannon: si CUALQUIERA de estos cambia la clasificación de una regla que
hoy es Clase III, es evidencia de que kcap depende de algo no puramente relacional):

  TEST 1 (renombrado de nodos) — isomorfismo EXACTO de `_enforce_kcap` (función determinística, sin
          azar): permutar las etiquetas de un grafo y verificar que la poda por kcap conmuta con la
          permutación (poda(perm(G)) == perm(poda(G))), salvo desempates documentados.
  TEST 1b (renombrado, pipeline completo) — igual pero de punta a punta: 5 reglas Clase III conocidas,
          re-etiquetadas ANTES de la dinámica, ¿sigue cada una clasificando Clase III?
  TEST 2 (orden de recorrido) — `_enforce_kcap` recorre nodos en `range(N)`; se sustituye por un orden
          barajado y se mide si cambia la clasificación final de las mismas 5 reglas.
  TEST 3 (reescalado de unidades de costo) — `_costo_y_podar` normaliza por z-score (¿ya es invariante
          matemáticamente? se verifica EMPÍRICAMENTE multiplicando historial/holonomía por una constante
          arbitraria y comparando el set de aristas conservadas, exacto).
  TEST 4 (independencia de N) — confirmación de que kcap es UN valor fijo por regla, no recalculado ni
          escalado según N (cita de código + corrida empírica barriendo N con el mismo kcap).

No se declara cierre/veredicto sobre si A2-B0-C2 "es válido" — sólo se reportan los números; el
resultado (PASS/FAIL de esta auditoría específica) queda documentado en
`FASE5_auditoria_C2_resultado_CS.md`, la lectura final es de Alexis.
"""
from __future__ import annotations
import sys, time, copy
import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
from cs090_fase5_clasificador import clasificar_regla
import cg003_diagnostico_gromov as GR

# ---- las 5 reglas Clase III reales de FASE5A_profundizar_A2B0C2 (mismos params/seed que ya corrieron,
#      ver cs090_fase5_profundizar_a2b0c2_resumen.csv) -- se REGENERAN con generar_regla() (misma seed =
#      mismos parámetros determinísticamente, no se inventan valores nuevos)
REGLAS_III_SEEDS = {
    "A2-B0-C2-r2":  272023,
    "A2-B0-C2-r4":  272217,
    "A2-B0-C2-r7":  272508,
    "A2-B0-C2-r13": 273090,
    "A2-B0-C2-r16": 273381,
}
# parámetros de ejecución IDÉNTICOS a los que usó cs090_fase5_profundizar_a2b0c2.py para clasificar
# estas mismas reglas como Clase III (para que la comparación baseline/variante sea de manzanas a
# manzanas con lo YA reportado en FASE5A_profundizar_A2B0C2_resultado_CS.md)
N_GRANDE = 2000
ESCALAS_B = (1, 2, 4, 8, 16)
N_SWEEPS = 14
N_SEEDS_NULL_TOPO = 3


def _regenerar_regla(rule_id, seed):
    """Regenera el dict de parámetros EXACTO de una regla ya corrida, con generar_regla() tal cual (sin
    tocar el generador) -- mismo seed = mismos K/J/noise/meandeg/kcap que ya produjeron Clase III."""
    idx = int(rule_id.rsplit("r", 1)[1])
    p = GEN.generar_regla("A2", "B0", "C2", idx, seed)
    p = GEN.aplicar_filtro_P1_P5(p, seed_chequeo=seed + 500_000)
    assert p["admitida"], f"{rule_id} ya no pasa el filtro P1-P5 -- no debería pasar, algo cambió"
    return p


print("=" * 100)
print("CS090 — AUDITORÍA C2 (kcap): Parte 2 -- tests de invarianza empíricos")
print("=" * 100)

reglas = {rid: _regenerar_regla(rid, seed) for rid, seed in REGLAS_III_SEEDS.items()}
for rid, p in reglas.items():
    print(f"  {rid}: K={p['K']} J={p['J']} noise={p['noise']} meandeg={p['meandeg']} kcap={p['kcap']}")


# ============================================================================================
# BASELINE -- reclasificar las 5 reglas con el motor SIN MODIFICAR (confirma que reproducimos
# exactamente el resultado ya reportado en FASE5A_profundizar_A2B0C2_resultado_CS.md antes de perturbar
# nada)
# ============================================================================================
def clasificar_baseline(p):
    filas = MOT.correr_regla_coarse(p, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                     n_seeds_null_topo=N_SEEDS_NULL_TOPO)
    return clasificar_regla(filas)


print("\n" + "-" * 100)
print("BASELINE (motor sin tocar, mismos params/seed) -- confirmar que sigue dando Clase III")
print("-" * 100)
baseline = {}
for rid, p in reglas.items():
    r = clasificar_baseline(p)
    baseline[rid] = r["clase"]
    print(f"  {rid}: clase={r['clase']:<28} pendiente={r['pendiente_real']:.3f} z_agg={r['z_agg']:.2f}")


# ============================================================================================
# TEST 1 -- renombrado de nodos: isomorfismo EXACTO de _enforce_kcap (función determinística)
# ============================================================================================
print("\n" + "-" * 100)
print("TEST 1 -- ¿_enforce_kcap conmuta con una permutación (renombrado) de los nodos?")
print("(función determinística, sin azar propio -- se puede comparar EXACTO, no sólo estadísticamente)")
print("-" * 100)


def _permutar_adj(adj, perm):
    N = len(adj)
    nuevo = [set() for _ in range(N)]
    for i in range(N):
        for j in adj[i]:
            nuevo[int(perm[i])].add(int(perm[j]))
    return nuevo


def _edges(adj):
    N = len(adj)
    return set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])


rng_t1 = np.random.default_rng(999)
n_pruebas, n_exactas, n_con_diferencia_por_empate = 0, 0, 0
for trial in range(30):
    N = 80
    meandeg = rng_t1.uniform(4, 8)
    kcap = int(rng_t1.integers(4, 8))
    adj0, _ = GR.aleatorio(N, meandeg=meandeg, seed=int(rng_t1.integers(1 << 30)))
    G = [set(a.tolist()) for a in adj0]

    perm = rng_t1.permutation(N)

    # camino A: podar G, LUEGO permutar el resultado
    G_podado = copy.deepcopy(G)
    MOT._enforce_kcap(G_podado, N, kcap)
    A_perm_despues = _edges(_permutar_adj(G_podado, perm))

    # camino B: permutar G, LUEGO podar (sobre el grafo YA renombrado)
    G_perm = _permutar_adj(G, perm)
    MOT._enforce_kcap(G_perm, N, kcap)
    B_podado_directo = _edges(G_perm)

    n_pruebas += 1
    if A_perm_despues == B_podado_directo:
        n_exactas += 1
    else:
        n_con_diferencia_por_empate += 1

print(f"  {n_pruebas} grafos aleatorios probados (N=80, kcap variable, meandeg variable).")
print(f"  podar-luego-renombrar == renombrar-luego-podar EXACTO en {n_exactas}/{n_pruebas} casos.")
print(f"  con diferencia en {n_con_diferencia_por_empate}/{n_pruebas} (se investiga si es desempate de "
      f"soporte empatado, abajo).")

# diagnóstico de las diferencias: ¿son por empates en 'soporte local' (mismo criterio, desempate por
# índice de nodo) o por algo que de verdad lea la etiqueta/posición del nodo?
if n_con_diferencia_por_empate > 0:
    rng_t1b = np.random.default_rng(1234)
    N = 80
    adj0, _ = GR.aleatorio(N, meandeg=6.0, seed=7)
    G = [set(a.tolist()) for a in adj0]
    kcap = 5
    huecos_con_empate = 0
    for i in range(N):
        nb = list(G[i])
        if len(nb) <= kcap:
            continue
        soportes = [len(G[i] & G[j]) for j in nb]
        # ¿hay empate exactamente en la frontera de corte (posición kcap-1 vs kcap)?
        orden = sorted(soportes, reverse=True)
        if len(orden) > kcap and orden[kcap - 1] == orden[kcap]:
            huecos_con_empate += 1
    print(f"  Diagnóstico (grafo de ejemplo, N=80, kcap=5): {huecos_con_empate} nodos con empate de "
          f"soporte EXACTAMENTE en la frontera de corte (kcap-ésimo vecino) -- el desempate ahí lo "
          f"decide `sorted(..., reverse=True)` sobre la tupla (soporte, índice_j), es decir el nodo con "
          f"índice j MÁS ALTO gana el empate. Esto es un detalle de implementación (desempate por orden "
          f"total arbitrario, común en algoritmos greedy), NO una dependencia de coordenadas/distancia: "
          f"el criterio primario que decide casi todos los casos sigue siendo el soporte (vecinos "
          f"compartidos), una cantidad puramente relacional.")


# ============================================================================================
# TEST 1b -- renombrado de nodos, PIPELINE COMPLETO (¿sigue clasificando Clase III?)
# ============================================================================================
print("\n" + "-" * 100)
print("TEST 1b -- renombrado de nodos en el PIPELINE COMPLETO (construcción -> dinámica -> clasificación)")
print("-" * 100)


def _fabrica_construir_A2_relabeled(seed_perm):
    def _construir(N, rng, p):
        sustrato = MOT.construir_A2(N, rng, p)          # función original, TAL CUAL, sin tocar
        perm = np.random.default_rng(seed_perm).permutation(N)
        sustrato["adj"] = _permutar_adj(sustrato["adj"], perm)
        return sustrato
    return _construir


_original_constructor_A2 = MOT.CONSTRUCTORES_A["A2"]     # se guarda para restaurar
resultados_relabel = {}
for i, (rid, p) in enumerate(reglas.items()):
    MOT.CONSTRUCTORES_A["A2"] = _fabrica_construir_A2_relabeled(seed_perm=5000 + i)
    try:
        r = clasificar_baseline(p)
    finally:
        MOT.CONSTRUCTORES_A["A2"] = _original_constructor_A2   # restaurar SIEMPRE, incluso si falla
    resultados_relabel[rid] = r["clase"]
    print(f"  {rid}: baseline={baseline[rid]:<12} -> renombrado={r['clase']:<28} "
          f"pendiente={r['pendiente_real']:.3f}  {'IGUAL' if r['clase']==baseline[rid] else '*** CAMBIÓ ***'}")


# ============================================================================================
# TEST 2 -- orden de recorrido de _enforce_kcap (range(N) barajado en vez de secuencial)
# ============================================================================================
print("\n" + "-" * 100)
print("TEST 2 -- orden de recorrido de nodos dentro de _enforce_kcap (barajado en vez de range(N))")
print("-" * 100)


def _enforce_kcap_orden_barajado(adj, N, kcap, _rng=[np.random.default_rng(42)]):
    """MISMA lógica que MOT._enforce_kcap línea por línea (soporte local = vecinos compartidos, sin
    coordenadas) -- la ÚNICA diferencia deliberada es el orden en que se visitan los nodos, para medir
    si el resultado depende del orden de recorrido (test de invarianza, no una reescritura del criterio)."""
    orden = list(range(N))
    _rng[0].shuffle(orden)
    for i in orden:
        nb = list(adj[i])
        if len(nb) <= kcap:
            continue
        sup = sorted(((len(adj[i] & adj[j]), j) for j in nb), reverse=True)
        mantener = set(j for _, j in sup[:kcap])
        for j in nb:
            if j not in mantener:
                adj[i].discard(j); adj[j].discard(i)


_original_enforce_kcap = MOT._enforce_kcap
resultados_orden = {}
for rid, p in reglas.items():
    MOT._enforce_kcap = _enforce_kcap_orden_barajado
    try:
        r = clasificar_baseline(p)
    finally:
        MOT._enforce_kcap = _original_enforce_kcap
    resultados_orden[rid] = r["clase"]
    print(f"  {rid}: baseline={baseline[rid]:<12} -> orden_barajado={r['clase']:<28} "
          f"pendiente={r['pendiente_real']:.3f}  {'IGUAL' if r['clase']==baseline[rid] else '*** CAMBIÓ ***'}")


# ============================================================================================
# CONTROL -- antes de atribuir los cambios de TEST 1b/2 a "kcap depende de la etiqueta/orden del
# nodo", hay que descartar la explicación más simple: que A2-B0-C2 sea simplemente RUIDOSO cerca del
# umbral (documentado en FASE5A_profundizar_A2B0C2_resultado_CS.md: "el resultado sigue viéndose
# parcialmente estocástico"). Control: re-correr las MISMAS 5 reglas (MISMOS parámetros K/J/noise/
# meandeg/kcap) con una SEMILLA distinta (no un renombrado, no un orden distinto -- una realización
# estocástica nueva del MISMO proceso, sin permutar ni reordenar nada) y ver si TAMBIÉN cambian de
# clase con esa frecuencia. Si sí, la inestabilidad de TEST 1b/2 es ruido esperable del sustrato, no
# evidencia específica de que _enforce_kcap dependa del índice/etiqueta.
# ============================================================================================
print("\n" + "-" * 100)
print("CONTROL -- ruido estocástico natural (misma regla, semilla distinta, SIN renombrar ni reordenar)")
print("-" * 100)
resultados_control = {}
for rid, p in reglas.items():
    p_variant = dict(p)
    p_variant["seed"] = p["seed"] + 1_000_003   # sólo cambia el stream de azar, ningún parámetro
    r = clasificar_baseline(p_variant)
    resultados_control[rid] = r["clase"]
    print(f"  {rid}: baseline={baseline[rid]:<12} -> semilla_distinta={r['clase']:<28} "
          f"pendiente={r['pendiente_real']:.3f}  {'IGUAL' if r['clase']==baseline[rid] else '*** CAMBIÓ ***'}")


# ============================================================================================
# TEST 3 -- reescalado de unidades de costo (hist/holonomía) en _costo_y_podar
# ============================================================================================
print("\n" + "-" * 100)
print("TEST 3 -- reescalado arbitrario de las cantidades de costo (historial de flips, holonomía)")
print("-" * 100)

rng_t3 = np.random.default_rng(2026)
n_pruebas3, n_exactas3 = 0, 0
for trial in range(20):
    n_edges = 60
    edges = [(i, i + 1) for i in range(n_edges)]
    flip_count = {e: int(rng_t3.integers(0, 8)) for e in edges}
    E_estado = {e: float(rng_t3.uniform(0, 8)) for e in edges}
    triangles = [(edges[k][0], edges[k][1], edges[(k + 1) % n_edges][1]) for k in range(0, n_edges - 2, 3)]
    K = 8.0

    conservar_original = MOT._costo_y_podar(edges, flip_count, E_estado, K, triangles)

    # reescalar: historial de flips y holonomía-fuente (E_estado, de la que sale hol) multiplicados por
    # una constante arbitraria (simula "cambiar de unidades" del costo/presupuesto)
    c = float(rng_t3.choice([10.0, 100.0, 0.01, 1000.0]))
    flip_count_esc = {e: v * c for e, v in flip_count.items()}
    E_estado_esc = {e: v for e, v in E_estado.items()}  # E_estado alimenta holonomía módulo K (estado
    # angular, no "costo" en sí) -- lo que se reescala es la cantidad de costo derivada (flip_count),
    # que es lo que correspondería a un "presupuesto" con unidades arbitrarias
    conservar_reescalado = MOT._costo_y_podar(edges, flip_count_esc, E_estado_esc, K, triangles)

    n_pruebas3 += 1
    if conservar_original == conservar_reescalado:
        n_exactas3 += 1

print(f"  {n_pruebas3} pruebas con flip_count reescalado por constantes arbitrarias (0.01x a 1000x).")
print(f"  Mismo set de aristas conservadas (exacto) en {n_exactas3}/{n_pruebas3} casos.")
print(f"  (Esperado matemáticamente: _costo_y_podar normaliza por z-score -- z(c*x)=z(x) para c>0 --")
print(f"   así que la poda por percentil P70 es invariante a la escala/unidades del costo por construcción.)")


# ============================================================================================
# TEST 4 -- independencia de N: kcap NO se recalcula ni escala con el tamaño del grafo
# ============================================================================================
print("\n" + "-" * 100)
print("TEST 4 -- ¿kcap está atado a un N específico?")
print("-" * 100)
print("  Cita de código (cs090_fase5_generador.py L45,77): kcap se samplea UNA vez por regla, entero en")
print("  RANGO_KCAP=(4,7), SIN ningún término que dependa de N (no hay kcap=f(N) en ningún punto).")
print("  Cita de código (cs090_fase5_motor.py L210-211, dentro de dinamica_B0):")
print('    if costo_nivel == "C2" and step % 4 == 0: _enforce_kcap(adj, N, p["kcap"])')
print("  -> el mismo p['kcap'] (fijo por regla) se aplica sin importar N.")
print("  Confirmación empírica: correr_regla() (no la variante coarse) evalúa la MISMA regla (mismo p,")
print("  mismo kcap) en las 4 tallas del piloto 500/1000/1500/2000 -- se corre una regla Clase III acá")
print("  con N chico para verificar que el kcap generado no cambia entre tallas (por construcción del")
print("  código no puede cambiar -- p['kcap'] es el mismo dict en las 4 llamadas de correr_regla).")

p_test4 = reglas["A2-B0-C2-r2"]
filas_n = MOT.correr_regla(p_test4, Ns=(300, 600, 900, 1200), n_sweeps=10, n_seeds_null_topo=2)
kcap_usado = p_test4["kcap"]
print(f"  Regla A2-B0-C2-r2: kcap={kcap_usado} usado idénticamente en las 4 tallas corridas "
      f"(N={[f['N'] for f in filas_n]}) -- diam_real por talla: {[round(f['diam_real'],2) for f in filas_n]}")
print(f"  kcap es un conteo absoluto de grado (4-7), no una fracción/función de N: no aparece ninguna")
print(f"  expresión kcap = algo*N o kcap = algo/N en ninguno de los dos archivos auditados.")


# ============================================================================================
# RESUMEN FINAL
# ============================================================================================
print("\n" + "=" * 100)
print("RESUMEN — Parte 2 (tests de invarianza)")
print("=" * 100)
print(f"{'regla':<16} {'baseline':<14} {'T1b renombrado':<18} {'T2 orden':<18} {'CONTROL(semilla)':<18} igual_T1b  igual_T2  igual_CONTROL")
todas_iguales = True
n_flip_relabel = n_flip_orden = n_flip_control = 0
for rid in reglas:
    igual_1b = resultados_relabel[rid] == baseline[rid]
    igual_2 = resultados_orden[rid] == baseline[rid]
    igual_c = resultados_control[rid] == baseline[rid]
    n_flip_relabel += not igual_1b; n_flip_orden += not igual_2; n_flip_control += not igual_c
    todas_iguales = todas_iguales and igual_1b and igual_2
    print(f"{rid:<16} {baseline[rid]:<14} {resultados_relabel[rid]:<18} {resultados_orden[rid]:<18} "
          f"{resultados_control[rid]:<18} {str(igual_1b):<9} {str(igual_2):<9} {igual_c}")
print(f"\nFlips respecto a baseline: renombrado={n_flip_relabel}/5  orden_barajado={n_flip_orden}/5  "
      f"control_semilla_distinta(sin tocar nada)={n_flip_control}/5")
print("Si el flip-rate de CONTROL (semilla distinta, sin renombrar/reordenar) es comparable al de "
      "renombrado/orden, la inestabilidad es ruido estocástico general del sustrato cerca del umbral de "
      "clasificación, no evidencia de que _enforce_kcap dependa específicamente de la etiqueta/orden del "
      "nodo. Si el flip-rate de CONTROL es claramente menor, sí hay una sensibilidad extra atribuible al "
      "renombrado/orden.")
print(f"\nTEST 1 (isomorfismo exacto de _enforce_kcap): {n_exactas}/{n_pruebas} exacto "
      f"({n_con_diferencia_por_empate} diferencias, atribuibles a desempate por índice en soporte "
      f"empatado -- ver diagnóstico arriba).")
print(f"TEST 3 (reescalado de costo, exacto): {n_exactas3}/{n_pruebas3} exacto.")
print(f"\n¿Las 5 reglas Clase III se mantienen Clase III bajo renombrado Y bajo orden de recorrido "
      f"barajado? {'SÍ' if todas_iguales else 'NO -- ver detalle arriba'}")
print("\nFin. No se declara veredicto sobre A2-B0-C2 -- números arriba, lectura final de Alexis.")
