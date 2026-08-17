"""
cs090_diam_corregido.py — MEDICIÓN OFICIAL DEL DIÁMETRO (versión corregida, 11-ago-2026)
=========================================================================================

QUÉ ES ESTE MÓDULO
------------------
Contiene la medición de diámetro de grafo que, a partir del 11-ago-2026 y por autorización explícita
de Alexis ("medición corregida reemplaza a la oficial si estaba mala"), pasa a ser **la medición
oficial** de la línea Cosmogénesis para todo cálculo NUEVO. La versión anterior sigue existiendo, sin
un carácter cambiado, dentro de `cs055_proceso_acoplado.py` — ver "POR QUÉ UN MÓDULO NUEVO".

QUÉ BUG CORRIGE
---------------
`_diam(adj, N)` de `cs055_proceso_acoplado.py` (congelado, importado por cs057/cs080/cs090_fase5_motor
y por ahí por toda la línea Fase III/IV/V) hace un doble-BFS que arranca así:

    src = next((i for i in range(N) if adj[i]), 0)     # PRIMER nodo, POR ÍNDICE, que tenga aristas

El doble-BFS sólo puede recorrer la componente conexa donde cae `src`. Si el grafo está fragmentado y
ese nodo de índice más bajo quedó en un **pedacito suelto** (típicamente un par de 2 nodos colgando),
la función devuelve el diámetro DE ESE PEDACITO — no el del grafo. Documentado con evidencia en
`FASE6_outliers_pendiente_negativa_CS.md` §1.3-§1.4: en 15 de 430 reglas de la línea A2-B0-C2 la
medición nativa (b=1) cayó sobre una componente de 2 nodos (diámetro reportado = 1) mientras la
componente gigante tenía 1.689-1.970 nodos y diámetro real 12-28.

Consecuencia aguas abajo: la pendiente log(diám) vs log(N_cajas) del coarse-graining se calcula sobre
5 escalas (b=1,2,4,8,16); si sólo el punto b=1 está mal (y muy abajo), el ajuste lineal sale NEGATIVO
— "el diámetro crece al agrupar", que es geométricamente imposible: contraer cajas conexas nunca puede
alargar un camino. Esa pendiente negativa es lo que se leía como "disolución" (Clase I) o como
"intermedio (sin clase clara)".

Analogía: es como medir la altura de un edificio apoyando el metro en el buzón de la vereda. Da 30 cm.
No es que el edificio sea raro — es que el metro se apoyó en otra cosa.

QUÉ HACE LA CORRECCIÓN
----------------------
Exactamente el MISMO algoritmo de doble-BFS del original — mismo recorrido, mismo criterio de "nodo más
lejano", mismo resultado entero — pero eligiendo el nodo de arranque **dentro de la componente conexa
más grande** (la "gigante"), en vez de "el de índice más bajo del grafo entero". La elección de la
gigante es determinista y no depende de cómo estén numerados los nodos:

    componentes en orden de descubrimiento (escaneo i=0..N-1 de nodos no aislados)
    gigante = max(componentes, key=len)      # ante empate, la primera descubierta (índice más bajo)
    arranque = gigante[0]                    # = el nodo de índice más bajo DE la gigante

EQUIVALENCIA CON LA VERSIÓN VIEJA (importante)
----------------------------------------------
Cuando el nodo de arranque original ya caía en la componente gigante, las dos versiones dan
**exactamente el mismo entero** — y no por casualidad, por construcción: si `src` (el nodo no aislado
de índice más bajo del grafo) pertenece a la gigante, entonces `src` es también el nodo de índice más
bajo de la gigante, o sea `gigante[0] == src`, y a partir de ahí el algoritmo es idéntico carácter por
carácter. `verificar_equivalencia()` en este mismo archivo lo comprueba numéricamente sobre una muestra
grande de grafos reales del proyecto (ver el bloque `__main__`), en vez de asumirlo.

Es decir: la corrección **no re-escribe la historia** donde la historia estaba bien. Sólo toca los
casos en que el metro se había apoyado en el buzón.

POR QUÉ UN MÓDULO NUEVO Y NO UN PARCHE EN `cs055`
--------------------------------------------------
`cs055_proceso_acoplado.py` está congelado y lo importan (directa o indirectamente) decenas de scripts
de experimentos ya informados: cs057, cs066, cs080 (Fase III), cs082 (Fase IV), cs090_fase5_motor
(Fase V-A/V-B), etc. Si se edita `_diam` en el lugar, cualquiera de esos scripts, al volver a correrse,
daría números distintos de los que están en su informe — y se perdería la capacidad de **reproducir**
los resultados históricos, que es justamente lo que permite auditar si un cambio de método movió algo.
La regla de la casa ("no editar experimentos cerrados") apunta a lo mismo.

Entonces: la versión vieja se conserva en `cs055` SÓLO para reproducir resultados históricos; la
versión de este módulo es la oficial para cálculo nuevo. Los dos números conviven y se pueden comparar
lado a lado, que es exactamente lo que hace `cs090_fase6_remedir_430.py`.

QUÉ NO HACE
-----------
No cambia ningún umbral de clasificación (los de `cs090_fase5_clasificador.py` siguen intactos), no
toca el coarse-graining (`cajas_bfs`/`grafo_grueso` de cs080), no toca la dinámica ni la holonomía, y
no declara ningún veredicto: sólo mide el diámetro apoyando el metro donde corresponde.
"""
from __future__ import annotations

import sys
from collections import deque

import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# =====================================================================================
# 1) LA VERSIÓN VIEJA, IMPORTADA SIN TOCAR EL ARCHIVO CONGELADO
# =====================================================================================
def _cargar_diam_original():
    """Trae `_diam` de cs055_proceso_acoplado.py TAL CUAL, extrayéndola por AST (ese archivo llama a
    main() sin guardia `if __name__`, así que un import normal correría toda su batería como efecto
    secundario — mismo truco que ya usa cs090_fase5_motor._extraer_funciones, verificado)."""
    import ast
    ruta = f"{_HERE}/cs055_proceso_acoplado.py"
    with open(ruta) as fh:
        fuente = fh.read()
    arbol = ast.parse(fuente)
    ns = {"np": np, "deque": deque}
    for nodo in arbol.body:
        if isinstance(nodo, ast.FunctionDef) and nodo.name in {"_diam", "_giant"}:
            exec(compile(ast.Module(body=[nodo], type_ignores=[]), ruta, "exec"), ns)
    return ns["_diam"], ns["_giant"]


diam_original, giant = _cargar_diam_original()
diam_original.__doc__ = ("Medición HISTÓRICA (cs055, congelada). Arranca el doble-BFS en el nodo no "
                         "aislado de índice más bajo del grafo entero — si ése cayó en un fragmento, "
                         "mide el fragmento. Se conserva SÓLO para reproducir resultados históricos.")


# =====================================================================================
# 2) LA VERSIÓN CORREGIDA (oficial desde el 11-ago-2026)
# =====================================================================================
def componentes(adj, N):
    """Lista de componentes conexas (cada una = lista de nodos), ignorando los nodos AISLADOS (sin
    ninguna arista), igual que hace el original al elegir su nodo de arranque. Las componentes salen
    en orden de descubrimiento por escaneo i=0..N-1, así que el orden es determinista y el primer
    elemento de cada componente es su nodo de índice más bajo."""
    visto = np.zeros(N, bool)
    comps = []
    for s in range(N):
        if visto[s] or not adj[s]:
            continue
        q = deque([s])
        visto[s] = True
        c = [s]
        while q:
            u = q.popleft()
            for w in adj[u]:
                if not visto[w]:
                    visto[w] = True
                    q.append(int(w))
                    c.append(int(w))
        comps.append(c)
    return comps


def _doble_bfs(adj, arranque):
    """El MISMO doble-BFS del original (dos BFS: uno para hallar un nodo lejano, otro para medir desde
    él), pero con el nodo de arranque pasado por parámetro."""
    def lejano(s):
        d = {s: 0}
        q = deque([s])
        f = s
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in d:
                    d[w] = d[u] + 1
                    q.append(int(w))
                    if d[w] > d[f]:
                        f = w
        return f, d[f]
    a, _ = lejano(arranque)
    _, dd = lejano(a)
    return dd


def diam_gigante(adj, N):
    """MEDICIÓN OFICIAL (desde 11-ago-2026): diámetro por doble-BFS arrancando en la componente conexa
    MÁS GRANDE. Devuelve 0 si no hay ninguna arista (mismo valor que devolvería el original)."""
    comps = componentes(adj, N)
    if not comps:
        return 0
    gig = max(comps, key=len)          # ante empate, la primera descubierta -> determinista
    if len(gig) < 2:
        return 0
    return _doble_bfs(adj, gig[0])


def diagnostico(adj, N):
    """Devuelve las dos mediciones lado a lado más el diagnóstico de por qué difieren, para poder
    auditar caso por caso en vez de confiar en el número final:
       diam_orig  — lo que devuelve la versión histórica
       diam_corr  — lo que devuelve la versión corregida
       tam_comp_medida  — tamaño de la componente donde cayó el arranque de la versión histórica
       tam_gigante      — tamaño de la componente gigante
       n_componentes / n_aislados
       descarrila — True si la histórica midió una componente < 10% de la gigante (mismo criterio
                    con el que se detectaron los 15 casos en FASE6_outliers_pendiente_negativa_CS.md)
    """
    comps = componentes(adj, N)
    if not comps:
        return dict(diam_orig=float(diam_original(adj, N)), diam_corr=0.0, tam_comp_medida=0,
                    tam_gigante=0, n_componentes=0, n_aislados=N, descarrila=False)
    gig = max(comps, key=len)
    src = next((i for i in range(N) if adj[i]), None)
    tam_src = next((len(c) for c in comps if src in set(c)), 0)
    n_no_aislados = sum(len(c) for c in comps)
    return dict(
        diam_orig=float(diam_original(adj, N)),
        diam_corr=float(diam_gigante(adj, N)),
        tam_comp_medida=tam_src,
        tam_gigante=len(gig),
        n_componentes=len(comps),
        n_aislados=N - n_no_aislados,
        descarrila=bool(len(gig) > 0 and tam_src < 0.10 * len(gig)),
    )


# =====================================================================================
# 3) VERIFICACIÓN DE EQUIVALENCIA (no se asume: se comprueba)
# =====================================================================================
def verificar_equivalencia(grafos, verboso=True):
    """`grafos` = iterable de (etiqueta, adj, N). Comprueba, sobre todos ellos, que
    diam_corr == diam_orig EXACTAMENTE en los casos donde el arranque original ya caía en la gigante,
    y reporta cuántos difieren y por qué. Devuelve dict con el recuento."""
    n_tot = n_no_desc = n_igual_no_desc = n_desc = 0
    difs_indebidas = []
    for etq, adj, N in grafos:
        d = diagnostico(adj, N)
        n_tot += 1
        if d["descarrila"]:
            n_desc += 1
        else:
            n_no_desc += 1
            if d["diam_orig"] == d["diam_corr"]:
                n_igual_no_desc += 1
            else:
                difs_indebidas.append((etq, d))
    if verboso:
        print(f"[equivalencia] grafos probados: {n_tot}")
        print(f"   sin descarrilamiento: {n_no_desc}  ->  idénticos orig==corr: {n_igual_no_desc} "
              f"({100.0*n_igual_no_desc/max(1,n_no_desc):.2f}%)")
        print(f"   con descarrilamiento: {n_desc}  (acá SÍ deben diferir: es el bug)")
        for etq, d in difs_indebidas[:10]:
            print(f"   !! difieren sin descarrilar: {etq} orig={d['diam_orig']} corr={d['diam_corr']} "
                  f"comp_medida={d['tam_comp_medida']} gigante={d['tam_gigante']}")
    return dict(n_total=n_tot, n_sin_descarrilar=n_no_desc, n_identicos=n_igual_no_desc,
                n_descarrilados=n_desc, diferencias_indebidas=difs_indebidas)


# =====================================================================================
# 4) RE-CORRIDA DE UNA REGLA DE FASE V CON LAS DOS MEDICIONES A LA VEZ
# =====================================================================================
def correr_regla_coarse_doble(p, N=2000, n_sweeps=14, escalas_b=(1, 2, 4, 8, 16),
                              n_seeds_null_topo=3):
    """Copia EXACTA de la cadena de `cs090_fase5_motor.correr_regla_coarse` (mismos rng, mismas
    semillas derivadas, mismo coarse-graining de cs080) pero midiendo el diámetro de las DOS maneras
    en cada escala, tanto en REAL como en los NULL_topo. Devuelve (filas_orig, filas_corr, diagnos):

      filas_orig — con el MISMO esquema que correr_regla_coarse (sirve para verificar bit a bit que la
                   reconstrucción reproduce el CSV histórico)
      filas_corr — idénticas salvo diam_real/diam_null_topo/diam_null_topo_std medidos con diam_gigante
      diagnos    — lista por escala con tamaño de componente medida vs gigante (auditoría)

    Nota deliberada: los NULL_topo también se re-miden con la corrección. El z-score REAL-vs-NULL que
    usa el clasificador compara las dos, así que corregir sólo un lado inventaría una asimetría.
    """
    import cs090_fase5_generador as GEN          # noqa: F401  (documenta de dónde sale `p`)
    import cs090_fase5_motor as MOT
    import cs080_renormalizacion as CS80
    import cg003_diagnostico_gromov as GR
    import time as _t

    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.CONSTRUCTORES_A[p["eje_A"]](N, rng, p)
    sustrato = MOT.DINAMICAS_B[p["eje_B"]](sustrato, p, rng, n_sweeps, p["eje_C"])
    m = MOT.medir(sustrato, p, rng)
    adj_real = m["adj_final"]

    nv = MOT.medir_null_valor(m, p, np.random.default_rng(p["seed"] * 8000 + N))
    holon_real_native, holon_null_native = m["holonomia"], nv["holonomia"]

    meandeg_equiv = max(0.5, 2.0 * m["n_aristas"] / max(1, N))
    adjs_null = []
    for s in range(n_seeds_null_topo):
        adj0, _ = GR.aleatorio(N, meandeg=meandeg_equiv, seed=int(p["seed"] * 6000 + N * 13 + s))
        adjs_null.append([set(a.tolist()) for a in adj0])

    filas_orig, filas_corr, diagnos = [], [], []
    t0 = _t.time()
    for b in escalas_b:
        if b == 1:
            adj_g, n_cajas = adj_real, N
        else:
            rng_b = np.random.default_rng(p["seed"] * 7000 + b * 31)
            asign, n_cajas = CS80.cajas_bfs(adj_real, N, b, rng_b)
            adj_g = CS80.grafo_grueso(adj_real, N, asign, n_cajas)

        dg = diagnostico(adj_g, n_cajas) if n_cajas > 1 else None
        diam_o = float(MOT._diam(adj_g, n_cajas)) if n_cajas > 1 else float("nan")
        diam_c = float(dg["diam_corr"]) if dg else float("nan")
        giant_g = float(giant(adj_g, n_cajas)) if n_cajas > 1 else 0.0
        n_aristas_g = sum(len(a) for a in adj_g) // 2

        d_nulls_o, d_nulls_c = [], []
        for k_null, adj_n in enumerate(adjs_null):
            if b == 1:
                adj_ng, n_cajas_n = adj_n, N
            else:
                rng_bn = np.random.default_rng(p["seed"] * 7500 + b * 37 + k_null)
                asign_n, n_cajas_n = CS80.cajas_bfs(adj_n, N, b, rng_bn)
                adj_ng = CS80.grafo_grueso(adj_n, N, asign_n, n_cajas_n)
            d_nulls_o.append(float(MOT._diam(adj_ng, n_cajas_n)) if n_cajas_n > 1 else float("nan"))
            d_nulls_c.append(float(diam_gigante(adj_ng, n_cajas_n)) if n_cajas_n > 1 else float("nan"))

        base = dict(rule_id=p["rule_id"], N=n_cajas, escala_b=b, dt=round(_t.time() - t0, 2),
                    giant_real=giant_g, holon_real=holon_real_native,
                    n_aristas=n_aristas_g, n_triangulos=m["n_triangulos"],
                    holon_null_valor=holon_null_native)
        filas_orig.append(dict(base, diam_real=diam_o,
                               diam_null_topo=float(np.nanmean(d_nulls_o)),
                               diam_null_topo_std=float(np.nanstd(d_nulls_o)) + 1e-9))
        filas_corr.append(dict(base, diam_real=diam_c,
                               diam_null_topo=float(np.nanmean(d_nulls_c)),
                               diam_null_topo_std=float(np.nanstd(d_nulls_c)) + 1e-9))
        diagnos.append(dict(escala_b=b, n_cajas=n_cajas, **(dg or {})))

    return filas_orig, filas_corr, diagnos


# =====================================================================================
# 5) AUTO-TEST
# =====================================================================================
if __name__ == "__main__":
    import cs090_fase5_generador as GEN
    import cs090_fase5_motor as MOT
    import cs080_renormalizacion as CS80

    print(__doc__.split("QUÉ ES ESTE MÓDULO")[0].strip() or "cs090_diam_corregido — auto-test")
    print("\n=== 1) casos de juguete ===")
    #  a) grafo conexo simple: camino de 5 nodos -> las dos mediciones deben dar 4
    adj = [set() for _ in range(5)]
    for i in range(4):
        adj[i].add(i + 1); adj[i + 1].add(i)
    print(f"   camino de 5 nodos: orig={diam_original(adj,5)} corr={diam_gigante(adj,5)}  (esperado 4/4)")
    #  b) el bug en su forma mínima: par suelto con índices bajos + camino largo con índices altos
    adj = [set() for _ in range(32)]
    adj[0].add(1); adj[1].add(0)                       # par suelto (índices 0-1)
    for i in range(2, 31):
        adj[i].add(i + 1); adj[i + 1].add(i)           # camino de 30 nodos (índices 2-31)
    d = diagnostico(adj, 32)
    print(f"   par suelto + camino de 30: orig={d['diam_orig']} corr={d['diam_corr']} "
          f"(comp medida={d['tam_comp_medida']} vs gigante={d['tam_gigante']}, "
          f"descarrila={d['descarrila']})  (esperado 1 vs 29)")

    print("\n=== 2) equivalencia sobre grafos REALES del proyecto ===")
    print("   (30 reglas A2-B0-C2 tomadas del barrido de 430, × 5 escalas de coarse-graining,")
    print("    + los 3 NULL_topo de cada una × 5 escalas = 600 grafos)")
    import csv as _csv
    seeds = []
    with open(f"{_HERE}/cs090_fase6_outliers_430_todas.csv") as fh:
        filas = list(_csv.DictReader(fh))
    # muestra estratificada: 25 al azar-determinista del cuerpo principal + 5 del grupo descarrilado
    filas.sort(key=lambda r: float(r["pendiente"]))
    seeds = [int(filas[i]["seed"]) for i in range(0, 5)]                       # los más negativos
    seeds += [int(filas[i]["seed"]) for i in range(20, len(filas), len(filas) // 25)][:25]

    import cg003_diagnostico_gromov as GR
    grafos = []
    for s in seeds:
        p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=s)
        rng = np.random.default_rng(p["seed"] * 5000 + 2000)
        sus = MOT.construir_A2(2000, rng, p)
        sus = MOT.dinamica_B0(sus, p, rng, 14, "C2")
        mm = MOT.medir(sus, p, rng)
        adjr = mm["adj_final"]
        for b in (1, 2, 4, 8, 16):
            if b == 1:
                ag, nc = adjr, 2000
            else:
                asg, nc = CS80.cajas_bfs(adjr, 2000, b, np.random.default_rng(p["seed"] * 7000 + b * 31))
                ag = CS80.grafo_grueso(adjr, 2000, asg, nc)
            grafos.append((f"seed{s}-real-b{b}", ag, nc))
        md = max(0.5, 2.0 * mm["n_aristas"] / 2000)
        for k in range(3):
            a0, _ = GR.aleatorio(2000, meandeg=md, seed=int(p["seed"] * 6000 + 2000 * 13 + k))
            an = [set(a.tolist()) for a in a0]
            for b in (1, 2, 4, 8, 16):
                if b == 1:
                    ag, nc = an, 2000
                else:
                    asg, nc = CS80.cajas_bfs(an, 2000, b,
                                             np.random.default_rng(p["seed"] * 7500 + b * 37 + k))
                    ag = CS80.grafo_grueso(an, 2000, asg, nc)
                grafos.append((f"seed{s}-null{k}-b{b}", ag, nc))
    verificar_equivalencia(grafos)
