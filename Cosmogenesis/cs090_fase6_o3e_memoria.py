"""
cs090_fase6_o3e_memoria.py — FASE VI, tarea O3-E: A2-B0-C2 CON MEMORIA vs SIN MEMORIA
=========================================================================================================

QUÉ PREGUNTA CONTESTA ESTE MÓDULO
---------------------------------
El 2do analista (propuesta 4.2) señaló que la regla A2-B0-C2 tiene "memoria" por construcción: parte de
lo que decide qué relaciones sobreviven no es el estado presente del sistema, sino su HISTORIA. La
pregunta operativa es si esa historia sirve para algo físico: ¿acreta más masa en Phantom un universo
cuya poda recuerda, que uno idéntico cuya poda sólo mira el presente?

Esto prueba SÓLO el ANTECEDENTE de O-N7.7 ("restricción histórica que reduce sin destruir capacidad
futura") tal como lo fijó la decisión formal ya adoptada (FASE5_especificacion_universalidad_CS.md §6 y
el informe del equipo). NO prueba Libertad Funcional, que pertenece a otro plano teórico
(ANIMA/Célula Madre). Este archivo no cruza ese límite ni declara cierre: produce números.

DÓNDE ESTÁ LA MEMORIA EN A2-B0-C2 (leído de cs090_fase5_motor.py, congelado)
----------------------------------------------------------------------------
En `dinamica_B0` (líneas 200-228 del motor congelado) hay un contador `flip_count`: en CADA sweep se
compara la topología actual contra la del sweep anterior y toda arista que apareció o desapareció suma
+1. Ese contador se ACUMULA a lo largo de los 14 sweeps y, al final, entra como uno de los dos
componentes del costo con el que C1/C2 poda el 30% de aristas más caras (`_costo_y_podar`, línea 154):

    costo = 0.5 * z(inconsistencia HISTÓRICA)  +  0.5 * z(conflicto de holonomía INSTANTÁNEO)
            └────── memoria: cuántas veces ──┘     └── presente: el estado final de los ──┘
                    esa arista cambió                   triángulos que la contienen
                    en TODA la corrida

Analogía: al final de la partida, el sistema decide qué relaciones cortar mirando dos cosas — cuánto
conflicto hay AHORA (holonomía) y cuántas veces esa relación fue inestable A LO LARGO DE TODA LA
PARTIDA (historia). La variante "sin memoria" le saca lo segundo: sólo puede mirar el ahora.

QUÉ SE QUITÓ EXACTAMENTE (y qué NO se tocó)
-------------------------------------------
Un ÚNICO cambio, parametrizado por `ventana_memoria`:

  * `ventana_memoria=None`  → CON MEMORIA. El término histórico del costo es `flip_count` acumulado
    sobre los 14 sweeps. Es, carácter por carácter, lo que hace el motor congelado.
  * `ventana_memoria=1`     → SIN MEMORIA. El término histórico se calcula sobre una ventana de UN solo
    sweep (el último): "cuántas veces cambió esta arista en el sweep que acaba de pasar", sin ninguna
    acumulación previa. Es la "actualización instantánea" que pedía la propuesta.

TODO lo demás es idéntico y así se verifica en `verificar_identidad_con_motor_congelado()`:
  - mismo `p` (misma regla, mismos K/J/noise/meandeg/kcap), misma semilla, mismo `np.random.default_rng`
  - mismo `construir_A2`, mismo `_circ_mean_update`, mismo calendario de recableo (`step % 3 == 0`) y de
    límite de escala kcap (`step % 4 == 0`), mismos 14 sweeps
  - mismo `_costo_y_podar` (mismo percentil P70, mismos pesos 0.5/0.5, mismo z-score)
  - mismo `_muestrear_triangulos`, mismo `medir()`
  - **mismo consumo del generador aleatorio**: contar flips no consume azar, y el costo/poda tampoco.
    Por construcción los dos brazos tienen trayectorias BIT A BIT IDÉNTICAS hasta el instante final de
    la poda; lo único que difiere es CUÁLES aristas condena el costo.

HECHO EMPÍRICO QUE CONVIENE SABER DE ANTEMANO (y que el script mide y reporta, no asume)
-----------------------------------------------------------------------------------------
Con el calendario del motor congelado (recableo cada 3 sweeps, kcap cada 4) y n_sweeps=14, en el ÚLTIMO
sweep (step=13) no hay ni recableo (13%3=1) ni kcap (13%4=1): la topología no cambia. Es decir, la
ventana instantánea de un sweep no contiene ningún cambio, el término histórico queda idénticamente 0 y
z(0)=0, de modo que el costo del brazo SIN MEMORIA es puramente el conflicto de holonomía del presente.
`diagnostico_memoria` de este módulo lo verifica corrida por corrida (`hist_suma`, `hist_no_cero`) en vez
de darlo por supuesto.

Consecuencia que hay que vigilar (el confound que pidió Alexis): el corte es a un PERCENTIL fijo (P70),
así que en principio los dos brazos podan la misma FRACCIÓN de aristas — pero si hay empates de costo
justo en el umbral (y los hay: todas las aristas que no caen en ningún triángulo muestreado comparten el
mismo valor de holonomía), la fracción efectivamente conservada puede diferir. Por eso cada corrida
reporta `n_aristas` y `grado_medio` de los DOS brazos, y el informe compara las densidades antes de
interpretar cualquier diferencia de masa.

NO SE MODIFICA NINGÚN ARCHIVO EXISTENTE. `cs090_fase5_motor.py`, `cs090_fase5_generador.py`,
`cs090_fase5b_phantom_adaptador.py`, `cs090_diam_corregido.py`, `cs080_renormalizacion.py` y
`cs082_fase4_4sustratos.py` sólo se IMPORTAN. La única función copiada (no importada) es `dinamica_B0`,
porque hay que insertarle el parámetro nuevo — la copia está marcada línea por línea abajo.
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN          # generar_regla() -- sólo import
import cs090_fase5_motor as MOT              # construir_A2/_recablear_A2/_enforce_kcap/medir -- sólo import
import cs082_fase4_4sustratos as C82         # _circ_mean_update -- sólo import
import cs080_renormalizacion as CS80         # cajas_bfs/grafo_grueso -- sólo import
import cs090_diam_corregido as DC            # diam_gigante (medición OFICIAL de diámetro) -- sólo import
import cg003_diagnostico_gromov as GR        # aleatorio() (ER para NULL_topo) -- sólo import
from cs090_fase5_clasificador import _pendiente_loglog   # sólo import


# =========================================================================================
# 1) COSTO Y PODA — copia de MOT._costo_y_podar con el término histórico como ARGUMENTO
# =========================================================================================
def _costo_y_podar_con_hist(edges, hist_por_arista, E_estado, K, triangles, P=70.0):
    """Copia EXACTA de `cs090_fase5_motor._costo_y_podar` (líneas 154-181 del motor congelado) salvo un
    detalle: en vez de leer el contador acumulado `flip_count` adentro, recibe el vector histórico ya
    armado (`hist_por_arista`, un dict arista->número). Así el MISMO código de costo sirve para el brazo
    con memoria (contador acumulado) y para el brazo sin memoria (contador de ventana de 1 sweep), y
    queda evidente que lo único que cambia entre brazos es ESE vector.

    Devuelve (conservar, diag): `conservar` es el set de aristas que sobreviven (igual que el original),
    `diag` agrega los números que el informe necesita para auditar el contraste (suma del histórico,
    cuántas aristas tienen histórico no nulo, cuántos valores distintos toma el costo, fracción
    efectivamente conservada — que puede no ser exactamente 30% si hay empates en el umbral).
    """
    if not edges:
        return set(), dict(n_edges=0, hist_suma=0.0, hist_no_cero=0, n_costos_distintos=0,
                           frac_conservada=float("nan"))
    hist = np.array([hist_por_arista.get(e, 0) for e in edges], dtype=float)

    # ---- desde acá hasta `conservar` es idéntico al motor congelado ----
    hol_por_arista = defaultdict(list)
    for (i, j, k) in triangles:
        eij = E_estado.get((i, j)) if i < j else E_estado.get((j, i))
        ejk = E_estado.get((j, k)) if j < k else E_estado.get((k, j))
        eki = E_estado.get((k, i)) if k < i else E_estado.get((i, k))
        if eij is None or ejk is None or eki is None:
            continue
        h = (eij + ejk + eki) % K
        h = abs(h - K if h > K / 2 else h)
        for e in ((i, j) if i < j else (j, i), (j, k) if j < k else (k, j), (i, k) if i < k else (k, i)):
            hol_por_arista[e].append(h)
    media_global = float(np.mean([v for vs in hol_por_arista.values() for v in vs])) if hol_por_arista else 0.0
    hol = np.array([float(np.mean(hol_por_arista[e])) if e in hol_por_arista else media_global for e in edges])

    def z(a):
        mu, sd = a.mean(), a.std()
        return (a - mu) / sd if sd > 1e-9 else np.zeros_like(a)

    costo = 0.5 * z(hist) + 0.5 * z(hol)
    umbral = np.percentile(costo, P)
    conservar = set(e for e, c in zip(edges, costo) if c <= umbral)
    # ---- fin de la parte idéntica ----

    diag = dict(n_edges=len(edges), hist_suma=float(hist.sum()), hist_no_cero=int((hist > 0).sum()),
                n_costos_distintos=int(len(np.unique(np.round(costo, 12)))),
                frac_conservada=len(conservar) / len(edges),
                umbral_costo=float(umbral))
    return conservar, diag


# =========================================================================================
# 2) DINÁMICA B0 con ventana de memoria — copia de MOT.dinamica_B0 (rama A1/A2)
# =========================================================================================
def dinamica_B0_ventana(sustrato, p, rng, n_sweeps, costo_nivel, ventana_memoria=None):
    """Copia de `cs090_fase5_motor.dinamica_B0` (rama de grafo, líneas 200-228) con UN solo agregado: el
    parámetro `ventana_memoria`.

      ventana_memoria=None → el término histórico del costo es el contador ACUMULADO sobre los
                             n_sweeps sweeps. Comportamiento idéntico al motor congelado.
      ventana_memoria=w    → el término histórico sólo cuenta los flips ocurridos en los ÚLTIMOS w
                             sweeps (w=1 → sólo el último: "actualización instantánea, sin acumulación").

    Nótese que AMBOS contadores se llevan siempre (cuesta nada y no consume azar); lo único que cambia
    es cuál de los dos alimenta el costo. Esto garantiza que las dos ramas recorran EXACTAMENTE la misma
    trayectoria de estados y de topología hasta la poda final, y que consuman el generador aleatorio en
    el mismo orden. La rama A0 del original no se copia: esta tarea es sólo A2-B0-C2.
    """
    assert sustrato["kind"] in ("A1", "A2"), "esta variante sólo cubre sustratos con grafo (A1/A2)"
    K, J, noise = p["K"], p["J"], p["noise"]
    adj = sustrato["adj"]; N = sustrato["N"]; kind = sustrato["kind"]

    S = rng.uniform(0, K, N)
    flip_acumulado = defaultdict(int)     # memoria: TODA la historia (lo que usa el motor congelado)
    flip_ventana = defaultdict(int)       # sin memoria: sólo los últimos `ventana_memoria` sweeps
    prev_edges = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])

    primer_step_ventana = None if ventana_memoria is None else max(0, n_sweeps - int(ventana_memoria))

    for step in range(n_sweeps):
        vecinos = [list(a) for a in adj]
        S = C82._circ_mean_update(S, vecinos, J, noise, rng)
        if kind == "A2" and step % 3 == 0:
            MOT._recablear_A2(adj, S, K, rng)
        if costo_nivel == "C2" and step % 4 == 0:
            MOT._enforce_kcap(adj, N, p["kcap"])
        if costo_nivel in ("C1", "C2"):
            cur = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])
            cambios = prev_edges ^ cur
            for e in cambios:
                flip_acumulado[e] += 1
                if primer_step_ventana is not None and step >= primer_step_ventana:
                    flip_ventana[e] += 1
            prev_edges = cur

    sustrato["S"] = S; sustrato["adj"] = adj; sustrato["flip_count"] = flip_acumulado
    diag = dict(ventana_memoria=ventana_memoria)
    if costo_nivel in ("C1", "C2"):
        edges = sorted(prev_edges)
        E_estado = {}
        for (i, j) in edges:
            E_estado[(i, j)] = abs(S[i] - S[j]) % K
        triangles = MOT._muestrear_triangulos(adj, N, rng)
        hist_usado = flip_acumulado if ventana_memoria is None else flip_ventana
        conservar, diag_costo = _costo_y_podar_con_hist(edges, hist_usado, E_estado, K, triangles)
        for (i, j) in edges:
            if (i, j) not in conservar:
                adj[i].discard(j); adj[j].discard(i)
        diag.update(diag_costo)
        diag["hist_acumulado_suma"] = float(sum(flip_acumulado.values()))
        diag["hist_acumulado_no_cero"] = int(len(flip_acumulado))
        diag["n_triangulos_costo"] = len(triangles)
    return sustrato, diag


# =========================================================================================
# 3) RECONSTRUIR UNA REGLA A2-B0-C2 EN CUALQUIERA DE LOS DOS BRAZOS
# =========================================================================================
def reconstruir(seed, ventana_memoria=None, N=2000, n_sweeps=14):
    """Misma cadena determinista que `cs090_fase5b_phantom_adaptador.reconstruir_regla_a2b0c2`
    (mismo `p` desde `seed`, mismo rng `seed*5000+N`, mismo `construir_A2`, mismo `medir`), con la
    dinámica reemplazada por `dinamica_B0_ventana`. Devuelve (p, m, diag)."""
    p = GEN.generar_regla("A2", "B0", "C2", idx=0, seed=seed)
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)
    sustrato, diag = dinamica_B0_ventana(sustrato, p, rng, n_sweeps, "C2", ventana_memoria)
    m = MOT.medir(sustrato, p, rng)
    return p, m, diag


# =========================================================================================
# 4) GEOMETRÍA: pendiente CORREGIDA log(diám) vs log(N_cajas) del grafo de cada brazo
# =========================================================================================
def pendiente_corregida(adj_real, N, seed, n_aristas, escalas_b=(1, 2, 4, 8, 16), n_seeds_null_topo=3):
    """Mide la pendiente del coarse-graining con la medición OFICIAL de diámetro
    (`cs090_diam_corregido.diam_gigante`, adoptada el 11-ago-2026), sobre las MISMAS 5 escalas y con las
    MISMAS semillas de cajas que usa la cadena congelada (`seed*7000+b*31` para el real, `seed*7500+b*37+k`
    para los NULL, `seed*6000+N*13+k` para el ER de cada NULL). Devuelve pendiente real, pendiente null,
    z_agg y las series por escala — mismos ingredientes que consume `clasificar_regla`.

    Ojo con el emparejamiento: los NULL_topo se generan con la densidad del grafo REAL DE ESE BRAZO, así
    que cada brazo tiene su propio null a igual densidad; el z compara cada brazo contra su propio azar.
    """
    diam_real, Ns_cajas, giants = [], [], []
    meandeg_equiv = max(0.5, 2.0 * n_aristas / max(1, N))
    adjs_null = []
    for s in range(n_seeds_null_topo):
        adj0, _ = GR.aleatorio(N, meandeg=meandeg_equiv, seed=int(seed * 6000 + N * 13 + s))
        adjs_null.append([set(a.tolist()) for a in adj0])

    diam_null_mean, diam_null_std_l = [], []
    for b in escalas_b:
        if b == 1:
            adj_g, n_cajas = adj_real, N
        else:
            rng_b = np.random.default_rng(seed * 7000 + b * 31)
            asign, n_cajas = CS80.cajas_bfs(adj_real, N, b, rng_b)
            adj_g = CS80.grafo_grueso(adj_real, N, asign, n_cajas)
        diam_real.append(float(DC.diam_gigante(adj_g, n_cajas)) if n_cajas > 1 else float("nan"))
        giants.append(float(DC.giant(adj_g, n_cajas)) if n_cajas > 1 else 0.0)
        Ns_cajas.append(n_cajas)

        dn = []
        for k_null, adj_n in enumerate(adjs_null):
            if b == 1:
                adj_ng, n_cajas_n = adj_n, N
            else:
                rng_bn = np.random.default_rng(seed * 7500 + b * 37 + k_null)
                asign_n, n_cajas_n = CS80.cajas_bfs(adj_n, N, b, rng_bn)
                adj_ng = CS80.grafo_grueso(adj_n, N, asign_n, n_cajas_n)
            dn.append(float(DC.diam_gigante(adj_ng, n_cajas_n)) if n_cajas_n > 1 else float("nan"))
        diam_null_mean.append(float(np.nanmean(dn)))
        diam_null_std_l.append(float(np.nanstd(dn)) + 1e-9)

    pend_real = _pendiente_loglog(Ns_cajas, diam_real)
    pend_null = _pendiente_loglog(Ns_cajas, diam_null_mean)
    z_por_N = [abs(dr - dnu) / ds if ds > 1e-9 else 0.0
               for dr, dnu, ds in zip(diam_real, diam_null_mean, diam_null_std_l)]
    return dict(pendiente_corregida=pend_real, pendiente_null=pend_null,
                z_agg=float(np.mean(z_por_N)), z_sostenido=all(z > 3.0 for z in z_por_N),
                diam_por_escala=diam_real, n_cajas_por_escala=Ns_cajas,
                diam_null_por_escala=diam_null_mean, giant_b1=giants[0])


# =========================================================================================
# 5) VERIFICACIÓN: el brazo CON MEMORIA reproduce el motor congelado bit a bit
# =========================================================================================
def verificar_identidad_con_motor_congelado(seeds, N=2000, n_sweeps=14, verboso=True):
    """Para cada seed corre (a) la cadena congelada tal cual
    (`cs090_fase5b_phantom_adaptador.reconstruir_regla_a2b0c2`) y (b) este módulo con
    ventana_memoria=None, y exige IGUALDAD EXACTA de la lista de adyacencia final (no sólo del número de
    aristas: nodo por nodo, vecino por vecino) y del estado S final. Si algo difiere, la copia no es fiel
    y el contraste entero no valdría nada."""
    from cs090_fase5b_phantom_adaptador import reconstruir_regla_a2b0c2
    ok = 0
    for s in seeds:
        p_ref, m_ref = reconstruir_regla_a2b0c2(seed=s, N=N, n_sweeps=n_sweeps)
        p_new, m_new, _ = reconstruir(s, ventana_memoria=None, N=N, n_sweeps=n_sweeps)
        iguales_adj = all(m_ref["adj_final"][i] == m_new["adj_final"][i] for i in range(N))
        iguales_num = (m_ref["n_aristas"] == m_new["n_aristas"] and m_ref["diam"] == m_new["diam"]
                       and abs(m_ref["holonomia"] - m_new["holonomia"]) < 1e-12)
        assert iguales_adj, f"seed {s}: la adyacencia final NO coincide con el motor congelado"
        assert iguales_num, f"seed {s}: n_aristas/diam/holonomía NO coinciden con el motor congelado"
        ok += 1
        if verboso:
            print(f"  seed {s}: idéntico al motor congelado (aristas={m_ref['n_aristas']}, "
                  f"diam_viejo={m_ref['diam']})", flush=True)
    return ok


if __name__ == "__main__":
    import time
    print(__doc__.split("QUÉ PREGUNTA CONTESTA")[0].strip())
    print("\n=== verificación de fidelidad (brazo CON memoria == motor congelado) ===")
    t0 = time.time()
    verificar_identidad_con_motor_congelado([272702], N=400, n_sweeps=14)
    print(f"  ({time.time()-t0:.1f}s a N=400)")

    print("\n=== smoke: los dos brazos a N=400, misma regla ===")
    for vent in (None, 1):
        p, m, d = reconstruir(272702, ventana_memoria=vent, N=400, n_sweeps=14)
        etiqueta = "CON memoria " if vent is None else "SIN memoria "
        print(f"  {etiqueta}: aristas={m['n_aristas']} frac_conservada={d['frac_conservada']:.4f} "
              f"hist_suma={d['hist_suma']} hist_no_cero={d['hist_no_cero']} "
              f"hist_acumulado_suma={d['hist_acumulado_suma']} costos_distintos={d['n_costos_distintos']}")
