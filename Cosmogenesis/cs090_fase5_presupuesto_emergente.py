"""
CS090 — FASE V-C2-C: PRESUPUESTO RELACIONAL EMERGENTE (¿puede el sistema descubrir su propio límite?)
=======================================================================================================
QUIÉN SOY: archivo NUEVO (ningún script congelado se toca) que responde la pregunta que el equipo marcó
como la más importante del roadmap de A2-B0-C2 (`equipo-analisis-fase5-10ago2026`, memoria del proyecto):
¿puede el sistema DESCUBRIR por sí mismo que no puede relacionarse con todo, en vez de que se lo
impongamos con un número fijo (`kcap`)?

En `cs090_fase5_motor.py`, `_enforce_kcap(adj, N, kcap)` (línea 136) impone un GRADO MÁXIMO idéntico a
todo nodo, todo el tiempo: si un nodo excede `kcap` vecinos, conserva los `kcap` de mayor soporte local
(vecinos compartidos) y descarta el resto. Es un límite de ENTRADA (un número elegido para el sistema).

Este archivo construye una alternativa: un PRESUPUESTO relacional `B_i` por nodo y un COSTO por arista
`c_ij`, ambos derivados de cantidades que YA EXISTEN en el motor congelado (documentado exactamente cuáles
abajo, sin inventar parámetros nuevos elegidos a mano) — de modo que el GRADO MÁXIMO EFECTIVO de cada
nodo sea una SALIDA de "cuántas aristas caben bajo el presupuesto dado su costo real", no un número de
entrada.

—— CANTIDADES REALES REUSADAS DEL MOTOR (honestidad explícita, ninguna inventada) ——
  1. `flip_count[e]` — cuántas veces esa arista exacta se prendió/apagó durante la corrida. Ya existe en
     `dinamica_B0` (se acumula siempre que `costo_nivel in ("C1","C2")`) y ya es uno de los dos
     componentes de `_costo_y_podar` (motor, línea 154) bajo el nombre "inconsistencia histórica". Se
     reusa TAL CUAL como componente de "historia" de `c_ij`.
  2. Conflicto de holonomía por arista — el MISMO cálculo que hace `_costo_y_podar` internamente
     (holonomía promedio de los triángulos que tocan la arista, vía `C82._holonomia_triangulos`). La
     lógica de agregación por-arista está embebida dentro de `_costo_y_podar` y no expuesta como función
     aparte — se REPLICA aquí verbatim (mismas ~10 líneas, sin modificar el original) porque el motor no
     ofrece un punto de reuso más fino.
  3. Diferencia circular de estado entre extremos, `|S_i - S_j| mod K` — EXACTAMENTE la métrica que
     `_recablear_A2` (motor, línea 105-133) ya usa para decidir si una arista es "cara" (`d > K/3` en la
     línea 120) y debe caerse durante el recableo co-emergente. Aquí se reusa esa MISMA señal, ya presente
     y ya interpretada como "costo de incompatibilidad" en el motor, como el tercer componente de `c_ij`
     — se documenta como aproximación honesta de "reciprocidad": el motor no trackea una cuenta explícita
     de interacción mutua i↔j, así que se usa la compatibilidad de estado (que sí determina si la relación
     SIGUE siendo mutuamente favorable) como proxy, no se inventa un contador nuevo.
  4. `p["kcap"]` — el propio parámetro ya sampleado por el generador congelado (`RANGO_KCAP=(4,7)`,
     `cs090_fase5_generador.py` línea 45/77) para CADA regla. Aquí se REINTERPRETA como el presupuesto
     `B_i` (igual para todo nodo, "B_i puede ser fijo" según la tarea) — pero cambia la UNIDAD: en
     C2-hard, kcap cuenta ARISTAS; en C2-budget, kcap cuenta UNIDADES DE COSTO (cada arista de costo
     promedio consume ~1 unidad, así que un nodo con relaciones todas "normales" conserva ~kcap aristas,
     igual que antes — pero un nodo con relaciones baratas puede sostener MÁS de kcap aristas, y uno con
     relaciones caras debe sostener MENOS). No se inventó un valor nuevo: se reusa el mismo entero ya
     sampleado por regla, sólo se le da otro significado.

—— LOS 4 BRAZOS ——
  1. C2-hard   — el actual (`MOT._enforce_kcap`, kcap cuenta aristas) — control/baseline, reusado TAL
                 CUAL vía `cs090_fase5_motor.correr_regla_coarse(p)` con `p["eje_C"]="C2"`.
  2. C2-budget — nuevo: `_enforce_relacional(..., modo="costo")` — kcap cuenta UNIDADES DE COSTO, se
                 conservan las aristas MÁS BARATAS hasta agotar el presupuesto.
  3. C2-random — control: usa el MISMO cálculo de "cuántas aristas debe soltar cada nodo bajo el
                 presupuesto" que C2-budget (misma magnitud de poda, arista por arista, nodo por nodo),
                 pero decide CUÁLES soltar al AZAR en vez de por costo.
  4. C0        — sin límite de escala — ya existe en el generador/motor, reusado TAL CUAL vía
                 `correr_regla_coarse(p)` con `p["eje_C"]="C0"`.

Diseño de control: para que los 4 brazos comparen la MISMA familia de reglas, se genera UN solo lote de
`N_SEEDS` reglas admitidas (filtro P1-P5 real, sin tocar el generador) con `eje_C="C2"` (el filtro P1-P5
no depende de eje_C — ver `_construir_sustrato_chequeo`/`_paso_chequeo` en el generador, que nunca leen
`eje_C`), y se REUTILIZAN los mismos parámetros (K,J,noise,meandeg,kcap,seed) en los 4 brazos, cambiando
sólo el mecanismo de límite de escala. Esto es un control MÁS estricto que el usado en
`cs090_fase5_profundizar_a2b0c2.py` (que generaba lotes independientes por combinación de ejes): acá los
4 brazos parten del MISMO grafo ER inicial (mismo seed → mismo `construir_A2`) y divergen sólo en qué
aristas se podan y por qué. Nota honesta: las trayectorias NO quedan bit-a-bit idénticas después del
primer paso de poda, porque `_enforce_relacional` consume números aleatorios adicionales (para
`_muestrear_triangulos`) que `_enforce_kcap` no consume — el control es "misma regla, mismo punto de
partida", no "mismo trazo de números aleatorios completo".

No se corre Phantom. No se declara cierre ni veredicto — se reportan números, la lectura final es de
Alexis. No se modifica ningún script congelado (verificable con git diff). No se hacen commits de git.
"""
from __future__ import annotations
import csv, sys, time
import numpy as np
from collections import defaultdict, Counter

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
import cs082_fase4_4sustratos as C82          # _circ_mean_update, _holonomia_triangulos -- reusado tal cual
import cs080_renormalizacion as CS80          # cajas_bfs, grafo_grueso -- reusado tal cual (mismo import
                                               # que hace MOT.correr_regla_coarse internamente)
from cs090_fase5_clasificador import clasificar_regla

EJE_A, EJE_B = "A2", "B0"
N_GRANDE = 2000
ESCALAS_B = (1, 2, 4, 8, 16)
N_SWEEPS = 14
N_SEEDS_NULL_TOPO = 3
SEED_BASE = 90210      # reproducible, distinto de los usados en tareas previas de Fase V


# ============================================================================================
# 1) COSTO RELACIONAL c_ij — 3 señales YA PRESENTES en el motor (ver docstring del módulo)
# ============================================================================================
def _costos_relacionales(adj, N, S, K, flip_count, rng, max_tri=1500, eps=1e-9):
    """Devuelve (edges, c_ij) para el grafo VIVO actual.
      hist   = flip_count[e]                              (motor: _costo_y_podar, "hist")
      hol    = holonomía media de triángulos que tocan e   (motor: _costo_y_podar, "hol" -- MISMO cálculo,
                                                              replicado acá porque no está expuesto aparte)
      compat = dist. circular |S_i-S_j| mod K              (motor: _recablear_A2, criterio "arista cara")
    Cada componente se normaliza dividiendo por su propia media (+eps) sobre las aristas VIVAS, así queda
    calibrado a "1.0 = arista de costo promedio" (ver nota de unidades en el docstring del módulo).
    c_ij = promedio simple de los 3 componentes normalizados (pesos iguales, mismo estilo de pesos
    iguales que ya usa `_costo_y_podar` para sus 2 componentes)."""
    edges = sorted(set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i]))
    if not edges:
        return edges, {}
    hist = np.array([flip_count.get(e, 0) for e in edges], dtype=float)

    E_estado = {(i, j): abs(S[i] - S[j]) % K for (i, j) in edges}
    compat = np.array([min(v, K - v) for v in E_estado.values()], dtype=float)

    # --- bloque replicado verbatim de la lógica interna de MOT._costo_y_podar (agregación de holonomía
    #     por arista) -- el motor congelado no la expone como función aparte, así que se reproduce acá
    #     sin alterar el archivo original.
    triangles = MOT._muestrear_triangulos(adj, N, rng, max_tri=max_tri)
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

    def _norm(a):
        m = a.mean()
        return a / (m + eps) if m > eps else np.ones_like(a)   # todo-igual -> costo neutro 1.0 c/u

    c = (_norm(hist) + _norm(hol) + _norm(compat)) / 3.0
    return edges, {e: float(v) for e, v in zip(edges, c)}


# ============================================================================================
# 2) ENFORCEMENT — mismo patrón secuencial nodo-por-nodo de MOT._enforce_kcap (para comparación
#    justa: procesa nodos en orden 0..N-1, un nodo puede podar una arista que otro querría conservar,
#    igual que el original), pero decide CUÁNTAS y CUÁLES aristas soltar por PRESUPUESTO DE COSTO en
#    vez de por cuenta fija.
# ============================================================================================
def _enforce_relacional(adj, N, S, K, flip_count, rng, budget, modo):
    """modo='costo'  -> C2-budget: conserva las aristas MÁS BARATAS hasta agotar `budget` (greedy
                         knapsack por orden de costo ascendente -- mismo estilo "aligerado" que el resto
                         del motor, no optimización exacta).
       modo='azar'   -> C2-random: calcula la MISMA cantidad de aristas a soltar por nodo que 'costo'
                         (mismo cálculo de presupuesto, ver más abajo), pero elige cuáles soltar
                         UNIFORMEMENTE AL AZAR -- control de "misma magnitud de poda, sin criterio"."""
    edges, costos = _costos_relacionales(adj, N, S, K, flip_count, rng)
    if not edges:
        return
    c_por_dirigida = {}
    for (i, j), c in costos.items():
        c_por_dirigida[(i, j)] = c
        c_por_dirigida[(j, i)] = c

    for i in range(N):
        nb = list(adj[i])
        if not nb:
            continue
        costos_nb = sorted(((c_por_dirigida[(i, j)], j) for j in nb))   # ascendente: barato primero
        acumulado, n_keep = 0.0, 0
        for c, _j in costos_nb:
            if acumulado + c > budget:
                break
            acumulado += c
            n_keep += 1
        if n_keep >= len(nb):
            continue  # el nodo ya cabe entero en su presupuesto -- nada que soltar (grado efectivo=salida)
        n_soltar = len(nb) - n_keep
        if modo == "costo":
            soltar = set(j for _c, j in costos_nb[n_keep:])             # las n_soltar MÁS CARAS
        elif modo == "azar":
            soltar = set(rng.choice(nb, size=n_soltar, replace=False).tolist())   # mismas CANTIDAD, azar
        else:
            raise ValueError(f"modo desconocido: {modo}")
        for j in soltar:
            adj[i].discard(j); adj[j].discard(i)


# ============================================================================================
# 3) DINÁMICA A2-B0 con límite relacional emergente -- copia adaptada de MOT.dinamica_B0 (rama A1/A2),
#    ÚNICO cambio real respecto al original: dónde el original llama a `_enforce_kcap` bajo C2, acá se
#    llama a `_enforce_relacional`. Todo lo demás (actualización de estado, recableo A2, tracking de
#    flip_count, poda final por costo/percentil P70) es EL MISMO CÓDIGO que `MOT.dinamica_B0`, reusando
#    las piezas congeladas (`C82._circ_mean_update`, `MOT._recablear_A2`, `MOT._muestrear_triangulos`,
#    `MOT._costo_y_podar`) sin tocarlas.
# ============================================================================================
def dinamica_B0_presupuesto(sustrato, p, rng, n_sweeps, modo):
    K, J, noise = p["K"], p["J"], p["noise"]
    adj = sustrato["adj"]; N = sustrato["N"]
    S = rng.uniform(0, K, N)
    flip_count = defaultdict(int)
    prev_edges = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])
    for step in range(n_sweeps):
        vecinos = [list(a) for a in adj]
        S = C82._circ_mean_update(S, vecinos, J, noise, rng)
        if step % 3 == 0:
            MOT._recablear_A2(adj, S, K, rng)                       # reusado tal cual (A2 co-emergencia)
        if step % 4 == 0:
            _enforce_relacional(adj, N, S, K, flip_count, rng, budget=p["kcap"], modo=modo)
        cur = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])
        for e in prev_edges ^ cur:
            flip_count[e] += 1
        prev_edges = cur
    sustrato["S"] = S; sustrato["adj"] = adj; sustrato["flip_count"] = flip_count
    # poda final por costo/percentil P70 -- MISMA pieza congelada que usa el C2 original al final de
    # dinamica_B0 (motor, líneas 218-227), reusada tal cual, no reinventada acá.
    edges = sorted(prev_edges)
    E_estado = {(i, j): abs(S[i] - S[j]) % K for (i, j) in edges}
    triangles = MOT._muestrear_triangulos(adj, N, rng)
    conservar = MOT._costo_y_podar(edges, flip_count, E_estado, K, triangles)
    for (i, j) in edges:
        if (i, j) not in conservar:
            adj[i].discard(j); adj[j].discard(i)
    return sustrato


# ============================================================================================
# 4) CORRER-REGLA-COARSE con presupuesto -- copia adaptada de MOT.correr_regla_coarse: MISMA
#    construcción de sustrato (`MOT.construir_A2`), MISMO coarse-graining (`CS80.cajas_bfs`/
#    `grafo_grueso`), MISMOS nulls emparejados (`MOT.medir_null_valor`, ER fresco vía `MOT.GR`), sólo
#    cambia qué función de dinámica se llama.
# ============================================================================================
def correr_regla_coarse_presupuesto(p, modo, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                     n_seeds_null_topo=N_SEEDS_NULL_TOPO):
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)                          # MISMO constructor que usa C2-hard/C0
    sustrato = dinamica_B0_presupuesto(sustrato, p, rng, n_sweeps, modo)
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
            asign, n_cajas = CS80.cajas_bfs(adj_real, N, b, rng_b)
            adj_g = CS80.grafo_grueso(adj_real, N, asign, n_cajas)
        diam_g = float(MOT._diam(adj_g, n_cajas)) if n_cajas > 1 else float("nan")
        giant_g = float(MOT._giant(adj_g, n_cajas)) if n_cajas > 1 else 0.0
        n_aristas_g = sum(len(a) for a in adj_g) // 2

        diam_nulls = []
        for k_null, adj_n in enumerate(adjs_null):
            if b == 1:
                adj_ng, n_cajas_n = adj_n, N
            else:
                rng_bn = np.random.default_rng(p["seed"] * 7500 + b * 37 + k_null)
                asign_n, n_cajas_n = CS80.cajas_bfs(adj_n, N, b, rng_bn)
                adj_ng = CS80.grafo_grueso(adj_n, N, asign_n, n_cajas_n)
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
# 5) DRIVER -- genera N_SEEDS reglas admitidas UNA vez, corre los 4 brazos sobre CADA regla,
#    clasifica, guarda CSV crudo (todas las filas, todas las escalas, los 4 brazos) y CSV resumen
#    (una fila por regla x brazo, con clase + observables continuos nativos b=1).
# ============================================================================================
BRAZOS = ("C2-hard", "C2-budget", "C2-random", "C0")


def _correr_brazo(brazo, p):
    if brazo == "C2-hard":
        p2 = dict(p); p2["eje_C"] = "C2"
        return MOT.correr_regla_coarse(p2, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                        n_seeds_null_topo=N_SEEDS_NULL_TOPO)
    elif brazo == "C0":
        p2 = dict(p); p2["eje_C"] = "C0"
        return MOT.correr_regla_coarse(p2, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                        n_seeds_null_topo=N_SEEDS_NULL_TOPO)
    elif brazo == "C2-budget":
        return correr_regla_coarse_presupuesto(p, modo="costo")
    elif brazo == "C2-random":
        return correr_regla_coarse_presupuesto(p, modo="azar")
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
            print(f"  [{etiqueta}] {p['rule_id']:<16} {brazo:<10} clase={r['clase']:<24} "
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
                           grado_medio=float(np.mean([f["grado_medio_b1"] for f in fb])) if fb else float("nan"),
                           n_aristas_medio=float(np.mean([f["n_aristas_b1"] for f in fb])) if fb else float("nan"),
                           diam_medio=float(np.mean([f["diam_b1"] for f in fb])) if fb else float("nan"),
                           pendiente_media=float(np.mean([f["pendiente"] for f in fb])) if fb else float("nan"))
    return out


def imprimir_resumen(titulo, resumen):
    print("\n" + "=" * 100)
    print(titulo)
    print("=" * 100)
    for brazo in BRAZOS:
        r = resumen[brazo]
        print(f"  {brazo:<12} n={r['n']:<4} clases={r['cnt']}  frac_III={r['frac_III']*100:5.1f}%  "
              f"grado_medio={r['grado_medio']:.2f}  n_aristas_medio={r['n_aristas_medio']:.1f}  "
              f"diam_medio={r['diam_medio']:.2f}  pendiente_media={r['pendiente_media']:.3f}")


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
        guardar_csv(filas_raw, f"{_HERE}/cs090_fase5_presupuesto_emergente_piloto_raw.csv")
        guardar_csv(filas_resumen, f"{_HERE}/cs090_fase5_presupuesto_emergente_piloto_resumen.csv")
        print(f"\nPILOTO terminado en {(time.time()-t0)/60:.1f} min")
    else:
        t0 = time.time()
        filas_raw, filas_resumen, admitidas, descartadas = correr_lote(
            n_seeds=20, seed_base=SEED_BASE + 1, presupuesto_seg=45 * 60, etiqueta="COMPLETO")
        resumen = resumen_por_brazo(filas_resumen)
        imprimir_resumen("COMPLETO -- resumen por brazo", resumen)
        guardar_csv(filas_raw, f"{_HERE}/cs090_fase5_presupuesto_emergente_resultados.csv")
        guardar_csv(filas_resumen, f"{_HERE}/cs090_fase5_presupuesto_emergente_resumen.csv")
        print(f"\nCOMPLETO terminado en {(time.time()-t0)/60:.1f} min")
        print("Fin. No se declara cierre ni veredicto -- números arriba, lectura final de Alexis.")
