"""
CS090 — FASE V-C2-C3: AISLAR EL MECANISMO — ¿ES LA RIGIDEZ DEL CORTE O LA UNIFORMIDAD DEL NÚMERO?
=====================================================================================================
QUIÉN SOY: archivo NUEVO (ningún script congelado ni los 2 archivos de las tareas anteriores se tocan)
que sigue de `FASE5_presupuesto_emergente_CS.md` (F5-C2-C) y `FASE5_presupuesto_soporte_local_CS.md`
(F5-C2-C2). Ambos compararon C2-hard (cupo fijo `kcap`, corte estricto vía `MOT._enforce_kcap`) contra
variantes de PRESUPUESTO ACUMULADO (knapsack greedy con `c_ij` de 3 o 4 señales) y encontraron el mismo
patrón las dos veces: C2-hard ~45% Clase III, todo lo demás (budget-original, budget-soporte, random)
agrupado en 10-20%, indistinguible entre sí. La lectura alternativa #3 del segundo informe (§6) plantea
la hipótesis no probada de esta tarea: puede que no sea la SEÑAL de costo (soporte, historia,
holonomía...) sino el MECANISMO -- `_enforce_kcap` es un corte ESTRICTO DE CONTEO (exactamente kcap
aristas, sin excepción, en un paso), mientras que el presupuesto es un límite de COSTO ACUMULADO
(knapsack greedy), una forma estructuralmente más "elástica" de decidir el corte. Alexis pidió
explícitamente aislar esa variable con "una regla única que atraviese por todo".

—— LAS DOS DIMENSIONES QUE LOS EXPERIMENTOS ANTERIORES CAMBIARON JUNTAS ——
  1. MECANISMO de aplicación: corte estricto de conteo exacto (ranking, top-N, sin excepción) vs.
     presupuesto acumulado elástico (knapsack greedy hasta agotar B_i).
  2. UNIFORMIDAD del número de cupo: idéntico para todos los nodos (kcap fijo) vs. variable por nodo.

  C2-hard    = mecanismo ESTRICTO + número UNIFORME   (el único punto ya medido de este lado).
  C2-budget* = mecanismo ELÁSTICO + número YA VARIABLE por diseño (el presupuesto en sí produce grado
               efectivo distinto por nodo según cuánto cuesten SUS aristas -- eso es justamente lo que
               "presupuesto" significa) -- el otro punto ya medido, pero mezclando las 2 dimensiones.
  Faltaba: mecanismo ESTRICTO + número VARIABLE -- la pieza que aísla si la rigidez importa por sí sola.

—— C2-HIBRIDO: EL BRAZO NUEVO QUE COMPLETA LA MATRIZ 2X2 ——
Mecanismo: EXACTAMENTE el de `MOT._enforce_kcap` (motor congelado, línea 136-148) -- ranking por soporte
local (`len(adj[i] & adj[j])`, vecinos compartidos, MISMO cálculo, sin aproximar) y corte a los N mejores,
sin excepción, en un solo paso. Único cambio: el N ya no es `p["kcap"]` fijo para todos los nodos, sino
`kcap_por_nodo[i]`, derivado de una cantidad YA EXISTENTE en el motor (ver fórmula abajo, sección 1).

—— CANTIDAD REUSADA PARA EL CUPO VARIABLE: GRADO INICIAL EN EL GRAFO ER (honestidad explícita) ——
`MOT.construir_A2(N, rng, p)` ya construye un grafo Erdős-Rényi fresco vía `GR.aleatorio(N,
meandeg=p["meandeg"], ...)` (motor, línea 91-96) ANTES de que arranque cualquier dinámica o poda. El
grado de cada nodo en ESE grafo recién construido (`grado_inicial_i = len(adj[i])`, leído inmediatamente
después de `construir_A2`, antes del primer sweep) es una cantidad que el motor YA CALCULA como
consecuencia de generar el sustrato -- no se inventa nada nuevo, sólo se LEE y se guarda antes de que la
dinámica la modifique. En expectación todos los nodos tienen el mismo grado (`meandeg`, parámetro de
`GR.aleatorio`), pero la REALIZACIÓN concreta de un grafo ER finito no es regular: por puro azar de
muestreo unos nodos nacen con más vecinos que otros (varianza tipo Poisson). Se usa esa variación NATURAL
del grafo ya generado como la señal de "capacidad" por nodo -- es la única cantidad en todo el motor que
(a) existe antes de cualquier dinámica/poda (no depende circularmente de decisiones de poda anteriores,
a diferencia de p.ej. el grado LUEGO de podar), y (b) varía nodo a nodo sin que se haya elegido a mano
ningún criterio nuevo.

Fórmula (`_cupo_variable`, sección 1 abajo):

    kcap_i = max(1, round(kcap_base * grado_inicial_i / mean(grado_inicial)))

donde `kcap_base = p["kcap"]` (el mismo entero ya sampleado por el generador, `RANGO_KCAP=(4,7)`,
REINTERPRETADO como cupo PROMEDIO en vez de cupo fijo -- mismo patrón de reinterpretación que
`cs090_fase5_presupuesto_emergente.py` ya usó con `kcap` como presupuesto `B_i`) y `mean(grado_inicial)`
es la media EMPÍRICA sobre los N nodos de ESTA realización concreta del grafo (no `p["meandeg"]`
teórico) -- se eligió la media empírica sobre la teórica porque garantiza, por construcción algebraica
exacta (no sólo en expectación), que el cupo promedio efectivo de la corrida ≈ `kcap_base`, la MISMA
"masa total de cupo" que tendría C2-hard con ese mismo `kcap` -- así la comparación C2-hibrido vs
C2-hard no está confundida por tener, en promedio, más o menos cupo total, sólo por CÓMO se reparte ese
cupo entre nodos (uniforme vs proporcional al grado inicial). Piso de `max(1, ...)`: un nodo que por azar
nace con grado_inicial=0 (raro pero posible en un ER disperso) recibiría kcap_i=0 con la fórmula pura, lo
que lo dejaría sin ninguna arista para siempre tras la primera poda -- se le da un piso de 1 para que siga
siendo un nodo relacional válido, documentado como la única desviación de la fórmula pura.

APROXIMACIÓN DECLARADA (honestidad): "grado inicial en el grafo ER" no es lo mismo que "actividad" o
"importancia" real del nodo en la dinámica -- es sólo la cantidad de vecinos con que nació por azar de
muestreo. Se eligió por ser la ÚNICA cantidad ya presente en el motor que cumple las dos condiciones de
arriba (pre-existe a la dinámica, varía por nodo sin elección manual). No se afirma que sea la mejor
noción posible de "capacidad" -- se documenta como la elección más simple y menos manipulada disponible.

—— LOS 5 BRAZOS ——
  1. C2-hard            — `MOT._enforce_kcap`, sin cambios (control/baseline, ESTRICTO+UNIFORME).
  2. C2-hibrido          — NUEVO: `_enforce_kcap_variable` (mismo ranking por soporte, mismo corte
                            estricto en un paso) con `kcap_por_nodo` variable (ESTRICTO+VARIABLE).
  3. C2-budget-soporte   — reusa `cs090_fase5_presupuesto_soporte.correr_regla_coarse_presupuesto_soporte`
                            TAL CUAL (c_ij de 4 componentes, knapsack greedy) -- recalculado fresco en
                            esta misma corrida, mismas 20 reglas (ELÁSTICO+VARIABLE-por-costo).
  4. C2-random           — NUEVO: `_enforce_kcap_variable_random` -- MISMA magnitud de poda por nodo que
                            C2-hibrido (mismo `kcap_por_nodo`), pero la arista que se suelta se elige al
                            azar en vez de por soporte (ESTRICTO+VARIABLE, sin criterio -- aísla si lo
                            que importa en C2-hibrido es el CRITERIO o sólo la rigidez+variabilidad).
  5. C0                  — sin límite de escala, sin cambios.

Diseño de control: MISMO lote de reglas admitidas A2-B0-C2 (filtro P1-P5 real) que las 2 tareas
anteriores usaron en su corrida completa -- mismo `seed_base` (`SEED_BASE` para piloto, `SEED_BASE+1`
para completo, ver `cs090_fase5_presupuesto_emergente.SEED_BASE=90210`) para que las reglas
(K,J,noise,meandeg,kcap,seed) sean IDÉNTICAS a las de esas dos corridas -- comparabilidad directa entre
las 3 tareas F5-C2-C/C2/C3. No se reusan números archivados de los CSV anteriores para ninguna
comparación cuantitativa dentro de este script -- C2-hard, C2-budget-soporte y C0 se recalculan frescos
acá, en el mismo momento que C2-hibrido y C2-random.

No se corre Phantom. No se declara cierre ni veredicto -- se reportan números, la lectura final es de
Alexis. No se modifica ningún script congelado ni `cs090_fase5_presupuesto_emergente.py` ni
`cs090_fase5_presupuesto_soporte.py` (verificable con git diff / mtime). No se hacen commits de git.
"""
from __future__ import annotations
import csv, sys, time
import numpy as np
from collections import Counter, defaultdict

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
import cs082_fase4_4sustratos as C82                        # _circ_mean_update -- reusado tal cual
import cs090_fase5_presupuesto_emergente as PE               # SEED_BASE, N_GRANDE, ESCALAS_B, etc. -- NO SE TOCA
import cs090_fase5_presupuesto_soporte as PS                 # brazo C2-budget-soporte, reusado TAL CUAL -- NO SE TOCA
from cs090_fase5_clasificador import clasificar_regla

EJE_A, EJE_B = "A2", "B0"
N_GRANDE = PE.N_GRANDE                 # 2000
ESCALAS_B = PE.ESCALAS_B               # (1,2,4,8,16)
N_SWEEPS = PE.N_SWEEPS                 # 14
N_SEEDS_NULL_TOPO = PE.N_SEEDS_NULL_TOPO  # 3
SEED_BASE = PE.SEED_BASE               # 90210 -- MISMA base que las 2 tareas anteriores (comparabilidad)


# ============================================================================================
# 1) CUPO VARIABLE POR NODO -- derivado del grado inicial en el grafo ER (ver docstring del módulo)
# ============================================================================================
def _cupo_variable(grado_inicial, kcap_base, minimo=1):
    """kcap_i = max(minimo, round(kcap_base * grado_inicial_i / mean(grado_inicial))).
    `mean(grado_inicial)` es la media EMPÍRICA de esta realización concreta del grafo (no el meandeg
    teórico) -- garantiza que el cupo total promedio de la corrida sea ≈ kcap_base, igual "masa" que
    C2-hard, y que la ÚNICA diferencia sea CÓMO se reparte esa masa entre nodos."""
    media = float(np.mean(grado_inicial))
    if media <= 1e-9:
        return np.full(len(grado_inicial), max(minimo, int(round(kcap_base))), dtype=int)
    escala = grado_inicial / media
    kcap_i = np.round(kcap_base * escala).astype(int)
    return np.maximum(kcap_i, minimo)


# ============================================================================================
# 2) ENFORCEMENT -- 2 variantes, mismo `kcap_por_nodo` (misma magnitud de poda), difieren SÓLO en el
#    criterio de qué arista se suelta cuando un nodo excede su cupo.
# ============================================================================================
def _enforce_kcap_variable(adj, N, kcap_por_nodo):
    """C2-hibrido -- copia EXACTA del mecanismo de `MOT._enforce_kcap` (corte estricto de conteo,
    ranking por soporte local `len(adj[i] & adj[j])`, se quedan los kcap_i de mayor soporte, sin
    excepción, en un solo paso) -- único cambio: `kcap_i = kcap_por_nodo[i]` en vez de `p["kcap"]` fijo
    para todos. Ver `MOT._enforce_kcap`, motor congelado línea 136-148, para el original sin tocar."""
    for i in range(N):
        nb = list(adj[i])
        kcap_i = int(kcap_por_nodo[i])
        if len(nb) <= kcap_i:
            continue
        sup = sorted(((len(adj[i] & adj[j]), j) for j in nb), reverse=True)
        mantener = set(j for _, j in sup[:kcap_i])
        for j in nb:
            if j not in mantener:
                adj[i].discard(j); adj[j].discard(i)


def _enforce_kcap_variable_random(adj, N, kcap_por_nodo, rng):
    """C2-random -- MISMO corte estricto de conteo, MISMO `kcap_por_nodo` (misma magnitud de poda, nodo
    por nodo) que C2-hibrido, pero la(s) arista(s) a soltar cuando el nodo excede su cupo se elige(n) AL
    AZAR en vez de por ranking de soporte -- aísla si lo que separa a C2-hibrido de C2-hard/C0 es el
    CRITERIO (soporte) o sólo la combinación rigidez+variabilidad del número, sin ningún criterio."""
    for i in range(N):
        nb = list(adj[i])
        kcap_i = int(kcap_por_nodo[i])
        if len(nb) <= kcap_i:
            continue
        n_soltar = len(nb) - kcap_i
        soltar = set(rng.choice(nb, size=n_soltar, replace=False).tolist())
        for j in soltar:
            adj[i].discard(j); adj[j].discard(i)


# ============================================================================================
# 3) DINÁMICA A2-B0 con cupo variable -- copia adaptada de MOT.dinamica_B0 (rama A1/A2), ÚNICO cambio
#    real respecto al original: dónde `dinamica_B0` llama a `_enforce_kcap(adj, N, p["kcap"])` bajo C2,
#    acá se llama a `_enforce_kcap_variable`/`_enforce_kcap_variable_random` con `kcap_por_nodo`. Todo lo
#    demás (actualización de estado, recableo A2, tracking de flip_count, poda final por costo/percentil
#    P70) es EL MISMO CÓDIGO que `MOT.dinamica_B0`, reusando las piezas congeladas sin tocarlas.
# ============================================================================================
def dinamica_B0_hibrido(sustrato, p, rng, n_sweeps, kcap_por_nodo, modo):
    K, J, noise = p["K"], p["J"], p["noise"]
    adj = sustrato["adj"]; N = sustrato["N"]
    S = rng.uniform(0, K, N)
    flip_count = defaultdict(int)
    prev_edges = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])
    for step in range(n_sweeps):
        vecinos = [list(a) for a in adj]
        S = C82._circ_mean_update(S, vecinos, J, noise, rng)
        if step % 3 == 0:
            MOT._recablear_A2(adj, S, K, rng)
        if step % 4 == 0:
            if modo == "soporte":
                _enforce_kcap_variable(adj, N, kcap_por_nodo)
            elif modo == "azar":
                _enforce_kcap_variable_random(adj, N, kcap_por_nodo, rng)
            else:
                raise ValueError(f"modo desconocido: {modo}")
        cur = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])
        for e in prev_edges ^ cur:
            flip_count[e] += 1
        prev_edges = cur
    sustrato["S"] = S; sustrato["adj"] = adj; sustrato["flip_count"] = flip_count
    # poda final por costo/percentil P70 -- MISMA pieza congelada que usa C2-hard al final de dinamica_B0
    # (motor, líneas 218-227), reusada tal cual, no reinventada acá.
    edges = sorted(prev_edges)
    E_estado = {(i, j): abs(S[i] - S[j]) % K for (i, j) in edges}
    triangles = MOT._muestrear_triangulos(adj, N, rng)
    conservar = MOT._costo_y_podar(edges, flip_count, E_estado, K, triangles)
    for (i, j) in edges:
        if (i, j) not in conservar:
            adj[i].discard(j); adj[j].discard(i)
    return sustrato


# ============================================================================================
# 4) CORRER-REGLA-COARSE con cupo variable -- copia adaptada de MOT.correr_regla_coarse: MISMA
#    construcción de sustrato (`MOT.construir_A2`), MISMO coarse-graining (`CS80.cajas_bfs`/
#    `grafo_grueso`, vía PE.CS80 -- import directo del mismo módulo congelado que usa PE), MISMOS nulls
#    emparejados. El grado inicial se captura INMEDIATAMENTE después de `construir_A2`, antes del primer
#    sweep de dinámica -- es la lectura "recién nacido" del grafo, no contaminada por ninguna poda.
# ============================================================================================
def correr_regla_coarse_hibrido(p, modo, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                 n_seeds_null_topo=N_SEEDS_NULL_TOPO):
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)                          # MISMO constructor que usa C2-hard/C0
    grado_inicial = np.array([len(sustrato["adj"][i]) for i in range(N)], dtype=float)
    kcap_por_nodo = _cupo_variable(grado_inicial, p["kcap"])
    sustrato = dinamica_B0_hibrido(sustrato, p, rng, n_sweeps, kcap_por_nodo, modo)
    m = MOT.medir(sustrato, p, rng)                                 # MISMA función de medición congelada
    adj_real = m["adj_final"]

    nv = MOT.medir_null_valor(m, p, np.random.default_rng(p["seed"] * 8000 + N))
    holon_real_native = m["holonomia"]
    holon_null_native = nv["holonomia"]

    meandeg_equiv = max(0.5, 2.0 * m["n_aristas"] / max(1, N))
    adjs_null = []
    for s in range(n_seeds_null_topo):
        seed_n = int(p["seed"] * 6000 + N * 13 + s)
        adj0, _ = MOT.GR.aleatorio(N, meandeg=meandeg_equiv, seed=seed_n)
        adjs_null.append([set(a.tolist()) for a in adj0])

    filas = []
    t0 = time.time()
    for b in escalas_b:
        if b == 1:
            adj_g, n_cajas = adj_real, N
        else:
            rng_b = np.random.default_rng(p["seed"] * 7000 + b * 31)
            asign, n_cajas = PE.CS80.cajas_bfs(adj_real, N, b, rng_b)
            adj_g = PE.CS80.grafo_grueso(adj_real, N, asign, n_cajas)
        diam_g = float(MOT._diam(adj_g, n_cajas)) if n_cajas > 1 else float("nan")
        giant_g = float(MOT._giant(adj_g, n_cajas)) if n_cajas > 1 else 0.0
        n_aristas_g = sum(len(a) for a in adj_g) // 2

        diam_nulls = []
        for k_null, adj_n in enumerate(adjs_null):
            if b == 1:
                adj_ng, n_cajas_n = adj_n, N
            else:
                rng_bn = np.random.default_rng(p["seed"] * 7500 + b * 37 + k_null)
                asign_n, n_cajas_n = PE.CS80.cajas_bfs(adj_n, N, b, rng_bn)
                adj_ng = PE.CS80.grafo_grueso(adj_n, N, asign_n, n_cajas_n)
            diam_nulls.append(float(MOT._diam(adj_ng, n_cajas_n)) if n_cajas_n > 1 else float("nan"))
        diam_null_mean = float(np.nanmean(diam_nulls))
        diam_null_std = float(np.nanstd(diam_nulls)) + 1e-9

        filas.append(dict(
            rule_id=p["rule_id"], N=n_cajas, escala_b=b, dt=round(time.time() - t0, 2),
            diam_real=diam_g, giant_real=giant_g, holon_real=holon_real_native,
            n_aristas=n_aristas_g, n_triangulos=m["n_triangulos"],
            diam_null_topo=diam_null_mean, diam_null_topo_std=diam_null_std,
            holon_null_valor=holon_null_native,
        ))
    return filas


# ============================================================================================
# 5) DRIVER -- 5 brazos sobre el mismo lote de reglas
# ============================================================================================
BRAZOS = ("C2-hard", "C2-hibrido", "C2-budget-soporte", "C2-random", "C0")


def _correr_brazo(brazo, p):
    if brazo == "C2-hard":
        p2 = dict(p); p2["eje_C"] = "C2"
        return MOT.correr_regla_coarse(p2, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                        n_seeds_null_topo=N_SEEDS_NULL_TOPO)
    elif brazo == "C0":
        p2 = dict(p); p2["eje_C"] = "C0"
        return MOT.correr_regla_coarse(p2, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                        n_seeds_null_topo=N_SEEDS_NULL_TOPO)
    elif brazo == "C2-hibrido":
        return correr_regla_coarse_hibrido(p, modo="soporte")
    elif brazo == "C2-random":
        return correr_regla_coarse_hibrido(p, modo="azar")
    elif brazo == "C2-budget-soporte":
        return PS.correr_regla_coarse_presupuesto_soporte(p, modo="costo")       # PS sin tocar
    else:
        raise ValueError(brazo)


def correr_lote(n_seeds, seed_base, presupuesto_seg, etiqueta=""):
    admitidas, descartadas = GEN.generar_reglas_clase(EJE_A, EJE_B, "C2", n_reglas=n_seeds,
                                                        seed_base=seed_base, max_intentos=max(80, n_seeds * 4))
    print(f"[{etiqueta}] reglas admitidas={len(admitidas)}  descartadas(P1-P5)={len(descartadas)}")

    filas_raw, filas_resumen = [], []
    t_inicio = time.time()
    for p in admitidas:
        for brazo in BRAZOS:
            t0 = time.time()
            try:
                filas = _correr_brazo(brazo, p)
            except Exception as e:
                print(f"  *** FALLO {brazo} en {p['rule_id']}: {type(e).__name__}: {e} ***")
                continue
            r = clasificar_regla(filas)
            dt = time.time() - t0
            fila_b1 = next(f for f in filas if f["escala_b"] == 1)
            grado_medio_b1 = 2 * fila_b1["n_aristas"] / fila_b1["N"] if fila_b1["N"] else float("nan")
            for f in filas:
                f2 = dict(f); f2.update(brazo=brazo, clase_final=r["clase"])
                filas_raw.append(f2)
            filas_resumen.append(dict(
                rule_id=p["rule_id"], brazo=brazo, clase=r["clase"],
                pendiente=r["pendiente_real"], z_agg=r["z_agg"], holon_ratio=r["holon_ratio"],
                n_aristas_b1=fila_b1["n_aristas"], grado_medio_b1=round(grado_medio_b1, 3),
                diam_b1=fila_b1["diam_real"], giant_b1=fila_b1["giant_real"], holon_b1=fila_b1["holon_real"],
                K=p["K"], J=p["J"], noise=p["noise"], meandeg=p["meandeg"], kcap=p["kcap"], seed=p["seed"],
            ))
            print(f"  [{etiqueta}] {p['rule_id']:<16} {brazo:<20} clase={r['clase']:<24} "
                  f"pendiente={r['pendiente_real']:.3f} grado_medio={grado_medio_b1:.2f} "
                  f"n_aristas={fila_b1['n_aristas']:<5} ({dt:.2f}s) [t_total={time.time()-t_inicio:.0f}s]")
        if time.time() - t_inicio > presupuesto_seg:
            print(f"  *** [{etiqueta}] SALVAGUARDA DE TIEMPO alcanzada, se corta el lote ***")
            break
    return filas_raw, filas_resumen, admitidas, descartadas


def resumen_por_brazo(filas_resumen):
    out = {}
    for brazo in BRAZOS:
        fb = [f for f in filas_resumen if f["brazo"] == brazo]
        cnt = Counter(f["clase"] for f in fb)
        n = len(fb)
        out[brazo] = dict(n=n, cnt=dict(cnt),
                           frac_III=cnt.get("III", 0) / n if n else float("nan"),
                           frac_IIImasIV=(cnt.get("III", 0) + cnt.get("IV", 0)) / n if n else float("nan"),
                           grado_medio=float(np.mean([f["grado_medio_b1"] for f in fb])) if fb else float("nan"),
                           n_aristas_medio=float(np.mean([f["n_aristas_b1"] for f in fb])) if fb else float("nan"),
                           diam_medio=float(np.mean([f["diam_b1"] for f in fb])) if fb else float("nan"),
                           pendiente_media=float(np.mean([f["pendiente"] for f in fb])) if fb else float("nan"),
                           pendiente_mediana=float(np.median([f["pendiente"] for f in fb])) if fb else float("nan"))
    return out


def imprimir_resumen(titulo, resumen):
    print("\n" + "=" * 110)
    print(titulo)
    print("=" * 110)
    for brazo in BRAZOS:
        r = resumen[brazo]
        print(f"  {brazo:<20} n={r['n']:<4} clases={r['cnt']}  frac_III={r['frac_III']*100:5.1f}%  "
              f"frac_III+IV={r['frac_IIImasIV']*100:5.1f}%  grado_medio={r['grado_medio']:.2f}  "
              f"n_aristas_medio={r['n_aristas_medio']:.1f}  diam_medio={r['diam_medio']:.2f}  "
              f"pendiente_media={r['pendiente_media']:.3f}  pendiente_mediana={r['pendiente_mediana']:.3f}")


def comparaciones_pareadas(filas_resumen):
    """Compara, regla por regla (mismo rule_id), la pendiente continua entre pares de brazos. Devuelve
    dict {(brazoA,brazoB): dict(gana_A, gana_B, empate, media_diff, mediana_diff, n)}."""
    por_regla = {}
    for f in filas_resumen:
        por_regla.setdefault(f["rule_id"], {})[f["brazo"]] = f["pendiente"]
    pares = [
        ("C2-hard", "C2-hibrido"), ("C2-hard", "C2-budget-soporte"), ("C2-hard", "C2-random"),
        ("C2-hibrido", "C2-budget-soporte"), ("C2-hibrido", "C2-random"),
        ("C2-budget-soporte", "C2-random"),
        ("C2-random", "C0"), ("C2-hibrido", "C0"),
    ]
    out = {}
    for a, b in pares:
        diffs = []
        for rid, d in por_regla.items():
            if a in d and b in d and not (np.isnan(d[a]) or np.isnan(d[b])):
                diffs.append(d[a] - d[b])
        diffs = np.array(diffs)
        if len(diffs) == 0:
            continue
        out[(a, b)] = dict(n=len(diffs), gana_A=int(np.sum(diffs > 0)), gana_B=int(np.sum(diffs < 0)),
                            empate=int(np.sum(diffs == 0)), media_diff=float(np.mean(diffs)),
                            mediana_diff=float(np.median(diffs)))
    return out


def imprimir_comparaciones(pareadas):
    print("\n" + "=" * 110)
    print("COMPARACIONES PAREADAS (pendiente continua, misma regla en ambos brazos)")
    print("=" * 110)
    for (a, b), r in pareadas.items():
        print(f"  {a:<20} vs {b:<20} n={r['n']:<3} {a} gana={r['gana_A']:<3} {b} gana={r['gana_B']:<3} "
              f"empate={r['empate']:<2} media_diff={r['media_diff']:+.3f} mediana_diff={r['mediana_diff']:+.3f}")


def guardar_csv(filas, ruta, campos=None):
    if not filas:
        print(f"(sin filas para {ruta})")
        return
    campos = campos or list(filas[0].keys())
    with open(ruta, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=campos)
        wr.writeheader()
        for f in filas:
            wr.writerow(f)
    print(f"CSV: {ruta}  ({len(filas)} filas)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["piloto", "completo"], default="piloto")
    args = ap.parse_args()

    if args.modo == "piloto":
        t0 = time.time()
        filas_raw, filas_resumen, admitidas, descartadas = correr_lote(
            n_seeds=3, seed_base=SEED_BASE, presupuesto_seg=10 * 60, etiqueta="PILOTO")
        resumen = resumen_por_brazo(filas_resumen)
        imprimir_resumen("PILOTO -- resumen por brazo", resumen)
        pareadas = comparaciones_pareadas(filas_resumen)
        imprimir_comparaciones(pareadas)
        guardar_csv(filas_raw, f"{_HERE}/cs090_fase5_mecanismo_aislado_piloto_raw.csv")
        guardar_csv(filas_resumen, f"{_HERE}/cs090_fase5_mecanismo_aislado_piloto_resumen.csv")
        print(f"\nPILOTO terminado en {(time.time()-t0)/60:.1f} min")
    else:
        t0 = time.time()
        filas_raw, filas_resumen, admitidas, descartadas = correr_lote(
            n_seeds=20, seed_base=SEED_BASE + 1, presupuesto_seg=50 * 60, etiqueta="COMPLETO")
        resumen = resumen_por_brazo(filas_resumen)
        imprimir_resumen("COMPLETO -- resumen por brazo", resumen)
        pareadas = comparaciones_pareadas(filas_resumen)
        imprimir_comparaciones(pareadas)
        guardar_csv(filas_raw, f"{_HERE}/cs090_fase5_mecanismo_aislado_resultados.csv")
        guardar_csv(filas_resumen, f"{_HERE}/cs090_fase5_mecanismo_aislado_resumen.csv")
        print(f"\nCOMPLETO terminado en {(time.time()-t0)/60:.1f} min")
        print("Fin. No se declara cierre ni veredicto -- números arriba, lectura final de Alexis.")
