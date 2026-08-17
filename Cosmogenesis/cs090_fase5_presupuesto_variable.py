"""
CS090 — FASE V-C2-C4: LA CELDA QUE FALTABA — PRESUPUESTO ELÁSTICO CON B_i VARIABLE POR NODO
=================================================================================================
QUIÉN SOY: archivo NUEVO (ningún script congelado ni los 3 archivos de las 3 tareas anteriores de esta
línea se tocan: `cs090_fase5_presupuesto_emergente.py`, `cs090_fase5_presupuesto_soporte.py`,
`cs090_fase5_mecanismo_aislado.py`). Sigue de `FASE5_mecanismo_aislado_CS.md` (F5-C2-C3), que dejó la
matriz 2×2 con 3 celdas medidas y una sin probar:

                        MECANISMO ESTRICTO           MECANISMO ELÁSTICO (presupuesto/knapsack)
    NÚMERO UNIFORME     C2-hard        45.0%          C2-budget-soporte      10-15%
    NÚMERO VARIABLE     C2-hibrido     35.0%          C2-presupuesto-variable   ???  <- ESTA TAREA

Los presupuestos usados hasta ahora (`C2-budget-original`, `C2-budget-soporte`) usan `B_i = p["kcap"]`,
EL MISMO número de entrada para TODOS los nodos — la variabilidad que producen es de SALIDA (grado final
distinto según cuánto cuesten las aristas de cada nodo), no de ENTRADA. Esta tarea agrega la pieza que
faltaba: un presupuesto donde `B_i` YA VARÍA por nodo desde la entrada, con EXACTAMENTE el mismo criterio
de variabilidad que ya validó C2-hibrido (`_cupo_variable` de `cs090_fase5_mecanismo_aislado.py`, grado
inicial en el grafo ER recién construido, ANTES de cualquier poda) — mismo criterio, para que la
comparación entre las 4 celdas de la matriz sea limpia y no se confunda "variabilidad" con "qué cantidad
concreta se usó para variar".

—— CÓMO SE INTEGRÓ B_i VARIABLE EN EL MECANISMO ELÁSTICO ——
Se reusan, sin modificar ninguna línea:
  1. `cs090_fase5_mecanismo_aislado._cupo_variable(grado_inicial, kcap_base)` — la MISMA fórmula
     `kcap_i = max(1, round(kcap_base * grado_inicial_i / mean(grado_inicial)))` que ya usó C2-hibrido
     para su cupo estricto por nodo. Acá el mismo número `kcap_i` se reinterpreta como PRESUPUESTO `B_i`
     en vez de tope de conteo — el mismo patrón de reinterpretación que las 3 tareas anteriores de esta
     línea ya usaron repetidamente con `p["kcap"]` (fijo, tope de conteo en C2-hard; presupuesto fijo en
     C2-budget-original/soporte).
  2. `cs090_fase5_presupuesto_soporte._costos_relacionales_soporte(...)` — el `c_ij` de 4 componentes
     (historia + holonomía + compatibilidad + soporte local) tal cual, sin tocar una línea.

Único código nuevo: `_enforce_relacional_variable`, una copia de
`cs090_fase5_presupuesto_soporte._enforce_relacional_soporte` (modo='costo': knapsack greedy por nodo,
conserva las aristas más baratas hasta agotar el presupuesto) donde el ÚNICO cambio real es
`budget=budget_por_nodo[i]` en vez de `budget` fijo para todos los nodos. Todo lo demás — cálculo de
`c_ij`, orden de recorrido, criterio de corte dentro del presupuesto de cada nodo — es exactamente el
mecanismo ya validado en F5-C2-C2.

—— LOS 4-5 BRAZOS ——
  1. C2-hard                 — `MOT._enforce_kcap`, sin cambios (ESTRICTO + UNIFORME, control).
  2. C2-hibrido               — reusa `MA.correr_regla_coarse_hibrido(p, modo="soporte")` tal cual,
                                 recalculado fresco en esta corrida (ESTRICTO + VARIABLE).
  3. C2-budget-soporte        — reusa `PS.correr_regla_coarse_presupuesto_soporte(p, modo="costo")` tal
                                 cual, recalculado fresco en esta corrida (ELÁSTICO + UNIFORME-en-la-
                                 entrada).
  4. C2-presupuesto-variable  — NUEVO, la celda que faltaba (ELÁSTICO + VARIABLE-en-la-entrada).
  5. C0                       — sin límite de escala, sin cambios.

Diseño de control: MISMO lote de reglas admitidas A2-B0-C2 (filtro P1-P5 real) que las 3 tareas anteriores
de esta línea — mismo `seed_base` (`SEED_BASE` para piloto, `SEED_BASE+1` para completo, ver
`cs090_fase5_presupuesto_emergente.SEED_BASE=90210`) para que las reglas (K,J,noise,meandeg,kcap,seed)
sean IDÉNTICAS a las de esas 3 corridas — comparabilidad directa entre las 4 tareas F5-C2-C/C2/C3/C4.
Ningún número de CSVs anteriores se reusa para comparación cuantitativa dentro de este script — C2-hard,
C2-hibrido, C2-budget-soporte y C0 se recalculan frescos acá, en el mismo momento que
C2-presupuesto-variable.

No se corre Phantom. No se declara cierre ni veredicto — se reportan números, la lectura final es de
Alexis. No se modifica ningún script congelado ni `cs090_fase5_presupuesto_emergente.py`,
`cs090_fase5_presupuesto_soporte.py`, ni `cs090_fase5_mecanismo_aislado.py` (verificable con git diff /
mtime). No se hacen commits de git.
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
import cs090_fase5_presupuesto_emergente as PE               # SEED_BASE, N_GRANDE, ESCALAS_B, CS80 -- NO SE TOCA
import cs090_fase5_presupuesto_soporte as PS                 # c_ij de 4 componentes, brazo C2-budget-soporte -- NO SE TOCA
import cs090_fase5_mecanismo_aislado as MA                   # _cupo_variable, brazo C2-hibrido -- NO SE TOCA
from cs090_fase5_clasificador import clasificar_regla

EJE_A, EJE_B = "A2", "B0"
N_GRANDE = PE.N_GRANDE                 # 2000
ESCALAS_B = PE.ESCALAS_B               # (1,2,4,8,16)
N_SWEEPS = PE.N_SWEEPS                 # 14
N_SEEDS_NULL_TOPO = PE.N_SEEDS_NULL_TOPO  # 3
SEED_BASE = PE.SEED_BASE               # 90210 -- MISMA base que las 3 tareas anteriores (comparabilidad)


# ============================================================================================
# 1) ENFORCEMENT NUEVO -- copia de PS._enforce_relacional_soporte (modo='costo'), único cambio real:
#    `budget_por_nodo[i]` en vez de `budget` fijo. El c_ij (4 componentes) se calcula con la MISMA
#    función `PS._costos_relacionales_soporte`, sin tocar una línea.
# ============================================================================================
def _enforce_relacional_variable(adj, N, S, K, flip_count, rng, budget_por_nodo):
    """Knapsack greedy por nodo, IDÉNTICO a `PS._enforce_relacional_soporte(..., modo='costo')`: para
    cada nodo i, ordena sus aristas vivas por c_ij ascendente (más barata primero) y conserva las más
    baratas hasta agotar el presupuesto. La ÚNICA diferencia real: el presupuesto ya NO es el mismo
    `budget` para todos los nodos -- es `budget_por_nodo[i]`, calculado afuera con
    `MA._cupo_variable(grado_inicial, p["kcap"])`, el mismo criterio de variabilidad que ya usó
    C2-hibrido para su cupo estricto por nodo (grado inicial en el grafo ER, antes de cualquier poda)."""
    edges, costos = PS._costos_relacionales_soporte(adj, N, S, K, flip_count, rng)
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
        budget_i = float(budget_por_nodo[i])
        costos_nb = sorted(((c_por_dirigida[(i, j)], j) for j in nb))
        acumulado, n_keep = 0.0, 0
        for c, _j in costos_nb:
            if acumulado + c > budget_i:
                break
            acumulado += c
            n_keep += 1
        if n_keep >= len(nb):
            continue
        n_soltar = len(nb) - n_keep
        soltar = set(j for _c, j in costos_nb[n_keep:])
        for j in soltar:
            adj[i].discard(j); adj[j].discard(i)


# ============================================================================================
# 2) DINÁMICA A2-B0 con presupuesto variable -- copia adaptada de
#    PS.dinamica_B0_presupuesto_soporte: único cambio real es llamar a `_enforce_relacional_variable`
#    con `budget_por_nodo` en vez de `PS._enforce_relacional_soporte` con `budget` fijo.
# ============================================================================================
def dinamica_B0_presupuesto_variable(sustrato, p, rng, n_sweeps, budget_por_nodo):
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
            _enforce_relacional_variable(adj, N, S, K, flip_count, rng, budget_por_nodo)
        cur = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])
        for e in prev_edges ^ cur:
            flip_count[e] += 1
        prev_edges = cur
    sustrato["S"] = S; sustrato["adj"] = adj; sustrato["flip_count"] = flip_count
    # poda final por costo/percentil P70 -- MISMA pieza congelada que usan los otros brazos al final.
    edges = sorted(prev_edges)
    E_estado = {(i, j): abs(S[i] - S[j]) % K for (i, j) in edges}
    triangles = MOT._muestrear_triangulos(adj, N, rng)
    conservar = MOT._costo_y_podar(edges, flip_count, E_estado, K, triangles)
    for (i, j) in edges:
        if (i, j) not in conservar:
            adj[i].discard(j); adj[j].discard(i)
    return sustrato


# ============================================================================================
# 3) CORRER-REGLA-COARSE con presupuesto variable -- copia adaptada de
#    PS.correr_regla_coarse_presupuesto_soporte / MA.correr_regla_coarse_hibrido: MISMA construcción de
#    sustrato (`MOT.construir_A2`), MISMO coarse-graining, MISMOS nulls emparejados. El grado inicial se
#    captura INMEDIATAMENTE después de `construir_A2`, antes del primer sweep -- igual que en C2-hibrido,
#    para que `budget_por_nodo` se calcule sobre la MISMA cantidad "recién nacida" en ambas celdas VARIABLE
#    de la matriz.
# ============================================================================================
def correr_regla_coarse_presupuesto_variable(p, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                              n_seeds_null_topo=N_SEEDS_NULL_TOPO):
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)                          # MISMO constructor que usan todos los brazos
    grado_inicial = np.array([len(sustrato["adj"][i]) for i in range(N)], dtype=float)
    budget_por_nodo = MA._cupo_variable(grado_inicial, p["kcap"])   # MA sin tocar -- mismo criterio que C2-hibrido
    sustrato = dinamica_B0_presupuesto_variable(sustrato, p, rng, n_sweeps, budget_por_nodo)
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
# 4) DRIVER -- 5 brazos sobre el mismo lote de reglas (matriz 2x2 + C0)
# ============================================================================================
BRAZOS = ("C2-hard", "C2-hibrido", "C2-budget-soporte", "C2-presupuesto-variable", "C0")


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
        return MA.correr_regla_coarse_hibrido(p, modo="soporte")                  # MA sin tocar
    elif brazo == "C2-budget-soporte":
        return PS.correr_regla_coarse_presupuesto_soporte(p, modo="costo")        # PS sin tocar
    elif brazo == "C2-presupuesto-variable":
        return correr_regla_coarse_presupuesto_variable(p)
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
            print(f"  [{etiqueta}] {p['rule_id']:<16} {brazo:<24} clase={r['clase']:<24} "
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
    print("\n" + "=" * 116)
    print(titulo)
    print("=" * 116)
    for brazo in BRAZOS:
        r = resumen[brazo]
        print(f"  {brazo:<24} n={r['n']:<4} clases={r['cnt']}  frac_III={r['frac_III']*100:5.1f}%  "
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
        ("C2-hard", "C2-hibrido"),
        ("C2-hard", "C2-budget-soporte"),
        ("C2-hard", "C2-presupuesto-variable"),
        ("C2-hibrido", "C2-presupuesto-variable"),
        ("C2-budget-soporte", "C2-presupuesto-variable"),
        ("C2-hibrido", "C2-budget-soporte"),
        ("C2-presupuesto-variable", "C0"),
        ("C2-hibrido", "C0"),
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
    print("\n" + "=" * 116)
    print("COMPARACIONES PAREADAS (pendiente continua, misma regla en ambos brazos)")
    print("=" * 116)
    for (a, b), r in pareadas.items():
        print(f"  {a:<24} vs {b:<24} n={r['n']:<3} {a} gana={r['gana_A']:<3} {b} gana={r['gana_B']:<3} "
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
        guardar_csv(filas_raw, f"{_HERE}/cs090_fase5_presupuesto_variable_piloto_raw.csv")
        guardar_csv(filas_resumen, f"{_HERE}/cs090_fase5_presupuesto_variable_piloto_resumen.csv")
        print(f"\nPILOTO terminado en {(time.time()-t0)/60:.1f} min")
    else:
        t0 = time.time()
        filas_raw, filas_resumen, admitidas, descartadas = correr_lote(
            n_seeds=20, seed_base=SEED_BASE + 1, presupuesto_seg=50 * 60, etiqueta="COMPLETO")
        resumen = resumen_por_brazo(filas_resumen)
        imprimir_resumen("COMPLETO -- resumen por brazo", resumen)
        pareadas = comparaciones_pareadas(filas_resumen)
        imprimir_comparaciones(pareadas)
        guardar_csv(filas_raw, f"{_HERE}/cs090_fase5_presupuesto_variable_resultados.csv")
        guardar_csv(filas_resumen, f"{_HERE}/cs090_fase5_presupuesto_variable_resumen.csv")
        print(f"\nCOMPLETO terminado en {(time.time()-t0)/60:.1f} min")
        print("Fin. No se declara cierre ni veredicto -- números arriba, lectura final de Alexis.")
