"""
CS090 — FASE V-C2-C5: EL CONTROL DE AZAR QUE FALTABA EN LA RAMA ELÁSTICA+VARIABLE
=================================================================================================
QUIÉN SOY: archivo NUEVO (ningún script congelado ni los 4 archivos de las 4 tareas anteriores de esta
línea se tocan: `cs090_fase5_presupuesto_emergente.py`, `cs090_fase5_presupuesto_soporte.py`,
`cs090_fase5_mecanismo_aislado.py`, `cs090_fase5_presupuesto_variable.py`). Sigue de
`FASE5_matriz_2x2_completa_CS.md` (F5-C2-C4), que cerró la matriz 2x2 MECANISMO(estricto/elástico) x
UNIFORMIDAD(uniforme/variable) para C2-hard/C2-hibrido/C2-budget-soporte/C2-presupuesto-variable, y dejó
documentado en su §7 (último punto) que faltaba un control cruzado: el mismo tipo de prueba que
`FASE5_mecanismo_aislado_CS.md` (F5-C2-C3) ya hizo en la rama ESTRICTA -- `C2-random` ahí mostró que,
dentro de estricto+variable, soltar aristas AL AZAR en vez de por soporte local hunde 35.0% -> 5.0%
(30pp de caída) -- pero nunca se hizo el equivalente en la rama ELÁSTICA+variable
(`C2-presupuesto-variable`, 10.0%/15.0% con Clase IV, F5-C2-C4).

Esta tarea agrega esa celda que falta: **C2-presupuesto-variable-azar** -- MISMO mecanismo de knapsack
elástico con B_i variable por nodo (idéntico `MA._cupo_variable`, idéntica "masa total" de presupuesto que
C2-presupuesto-variable), pero en vez de ordenar las aristas vivas de cada nodo por costo ascendente
(conservando las más baratas primero), se toman en un ORDEN ALEATORIO y se aplica el mismo criterio
"sumar hasta agotar B_i" -- la magnitud de poda (el presupuesto de entrada B_i, denominado en las mismas
unidades de costo c_ij) queda idéntica; lo que cambia es que la selección de CUÁLES aristas sobreviven ya
no depende de que sean las más baratas, sólo de en qué orden aleatorio le tocó aparecer a cada una antes de
que se agote el presupuesto del nodo. Nota importante de honestidad: como el criterio de parada sigue
siendo "costo acumulado > B_i", el conteo final de aristas por nodo SÍ puede diferir entre el brazo con
criterio y el de azar (una arista cara temprano en el orden aleatorio consume más presupuesto que una
barata) -- eso es intencional y análogo a como funciona un presupuesto real: la CANTIDAD de dinero es igual,
lo que varía es cuánto rinde según en qué orden se gasta. Esto NO es lo mismo que fijar un conteo exacto de
aristas por nodo (eso sería el control de la rama estricta, ya hecho en F5-C2-C3) -- es el control MÁS FIEL
al mecanismo elástico: mismo B_i, mismo tipo de criterio de parada, sólo se quita la ordenación por costo.

—— CÓMO SE CONSTRUYÓ EL CONTROL DE AZAR ——
Se reusan, sin modificar ninguna línea:
  1. `cs090_fase5_mecanismo_aislado._cupo_variable(grado_inicial, kcap_base)` -- MISMA fórmula, MISMO
     `budget_por_nodo` que usó `C2-presupuesto-variable` (F5-C2-C4).
  2. `cs090_fase5_presupuesto_soporte._costos_relacionales_soporte(...)` -- el `c_ij` de 4 componentes
     (historia + holonomía + compatibilidad + soporte local) tal cual, sin tocar una línea. Se sigue
     necesitando el costo real de cada arista para saber cuándo se agota el presupuesto -- lo que se quita
     es sólo el ORDEN por el que se recorren las aristas antes de sumar, no el costo en sí.
  3. `cs090_fase5_presupuesto_variable.correr_regla_coarse_presupuesto_variable` como PLANTILLA de
     construcción de sustrato/coarse-graining/nulls -- se reimplementa acá (no se importa directo) porque
     la única pieza que cambia es el enforcement interno de la dinámica; todo el resto (constructor,
     medición, coarse-graining, nulls topológicos) es exactamente igual.

Único código nuevo: `_enforce_relacional_variable_azar`, copia de
`cs090_fase5_presupuesto_variable._enforce_relacional_variable` donde el ÚNICO cambio real es que la lista
de vecinos de cada nodo se recorre en un orden ALEATORIO (`rng.permutation`) en vez de ordenada por costo
ascendente -- el costo de cada arista se sigue calculando con la MISMA función de 4 componentes y se sigue
usando para decidir cuándo el presupuesto se agotó, sólo no decide el ORDEN.

—— LOS 4-5 BRAZOS ——
  1. C2-hard                      — `MOT._enforce_kcap`, sin cambios (control, ESTRICTO+UNIFORME).
  2. C2-hibrido                    — reusa `MA.correr_regla_coarse_hibrido(p, modo="soporte")` tal cual,
                                      recalculado fresco (ESTRICTO+VARIABLE+criterio-soporte).
  3. C2-presupuesto-variable       — reusa `PV.correr_regla_coarse_presupuesto_variable(p)` tal cual,
                                      recalculado fresco (ELÁSTICO+VARIABLE+criterio-costo, F5-C2-C4).
  4. C2-presupuesto-variable-azar  — NUEVO, la celda que falta (ELÁSTICO+VARIABLE+SIN criterio).
  5. C0                            — sin límite de escala, sin cambios.

Diseño de control: MISMO lote de reglas admitidas A2-B0-C2 (filtro P1-P5 real) que las 4 tareas anteriores
de esta línea -- mismo `seed_base` (`PE.SEED_BASE=90210` para piloto, `SEED_BASE+1` para completo) para que
las reglas (K,J,noise,meandeg,kcap,seed) sean IDÉNTICAS a las de esas 4 corridas -- comparabilidad directa
entre las 5 tareas F5-C2-C/C2/C3/C4/C5. Ningún número de CSVs anteriores se reusa para comparación
cuantitativa dentro de este script -- los 5 brazos se recalculan frescos acá, en el mismo momento.

No se corre Phantom. No se declara cierre ni veredicto -- se reportan números, la lectura final es de
Alexis. No se modifica ningún script congelado ni `cs090_fase5_presupuesto_emergente.py`,
`cs090_fase5_presupuesto_soporte.py`, `cs090_fase5_mecanismo_aislado.py`, ni
`cs090_fase5_presupuesto_variable.py` (verificable con git diff / mtime). No se hacen commits de git.
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
import cs090_fase5_presupuesto_soporte as PS                 # c_ij de 4 componentes -- NO SE TOCA
import cs090_fase5_mecanismo_aislado as MA                   # _cupo_variable, brazo C2-hibrido -- NO SE TOCA
import cs090_fase5_presupuesto_variable as PV                 # brazo C2-presupuesto-variable -- NO SE TOCA
from cs090_fase5_clasificador import clasificar_regla

EJE_A, EJE_B = "A2", "B0"
N_GRANDE = PE.N_GRANDE                 # 2000
ESCALAS_B = PE.ESCALAS_B               # (1,2,4,8,16)
N_SWEEPS = PE.N_SWEEPS                 # 14
N_SEEDS_NULL_TOPO = PE.N_SEEDS_NULL_TOPO  # 3
SEED_BASE = PE.SEED_BASE               # 90210 -- MISMA base que las 4 tareas anteriores (comparabilidad)


# ============================================================================================
# 1) ENFORCEMENT NUEVO -- copia de PV._enforce_relacional_variable, único cambio real: la lista de
#    vecinos de cada nodo se recorre en ORDEN ALEATORIO (rng.permutation) en vez de ordenada por costo
#    ascendente. El costo c_ij se sigue calculando con la MISMA función de 4 componentes y se sigue usando
#    para decidir cuándo el presupuesto B_i se agotó (criterio de parada idéntico: "sumar hasta agotar
#    B_i") -- lo único que se quita es que el costo decida el ORDEN de recorrido.
# ============================================================================================
def _enforce_relacional_variable_azar(adj, N, S, K, flip_count, rng, budget_por_nodo):
    """C2-presupuesto-variable-azar -- MISMO knapsack elástico por nodo con presupuesto `budget_por_nodo[i]`
    (idéntica magnitud de B_i que `PV._enforce_relacional_variable`), pero las aristas vivas de cada nodo
    se recorren en un orden ALEATORIO en vez de ordenadas por costo ascendente antes de acumular. Se sigue
    sumando el costo REAL de cada arista (no se ignora el costo, sólo deja de decidir el orden) hasta que
    el acumulado excede B_i -- ahí se corta, igual que en el brazo con criterio. Esto aísla si lo que
    separa a C2-presupuesto-variable de C2-hard/C0 es el CRITERIO (costo/soporte) o sólo la combinación
    mecanismo-elástico+variabilidad-de-entrada, sin ningún criterio real de selección."""
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
        orden_azar = rng.permutation(len(nb))
        nb_azar = [nb[k] for k in orden_azar]
        acumulado, mantener = 0.0, []
        for j in nb_azar:
            c = c_por_dirigida[(i, j)]
            if acumulado + c > budget_i:
                continue  # esta arista concreta no entra, pero se sigue probando el resto del orden aleatorio
            acumulado += c
            mantener.append(j)
        mantener = set(mantener)
        if len(mantener) >= len(nb):
            continue
        soltar = set(nb) - mantener
        for j in soltar:
            adj[i].discard(j); adj[j].discard(i)


# ============================================================================================
# 2) DINÁMICA A2-B0 con presupuesto variable azar -- copia adaptada de
#    PV.dinamica_B0_presupuesto_variable: único cambio real es llamar a `_enforce_relacional_variable_azar`
#    en vez de `PV._enforce_relacional_variable`.
# ============================================================================================
def dinamica_B0_presupuesto_variable_azar(sustrato, p, rng, n_sweeps, budget_por_nodo):
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
            _enforce_relacional_variable_azar(adj, N, S, K, flip_count, rng, budget_por_nodo)
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
# 3) CORRER-REGLA-COARSE con presupuesto variable azar -- copia adaptada de
#    PV.correr_regla_coarse_presupuesto_variable: MISMA construcción de sustrato (`MOT.construir_A2`),
#    MISMO coarse-graining, MISMOS nulls emparejados, MISMO cómputo de `budget_por_nodo` con
#    `MA._cupo_variable` sobre el grado inicial recién nacido. El único cambio real es que la dinámica llama
#    a `dinamica_B0_presupuesto_variable_azar` en vez de `PV.dinamica_B0_presupuesto_variable`.
# ============================================================================================
def correr_regla_coarse_presupuesto_variable_azar(p, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                                    n_seeds_null_topo=N_SEEDS_NULL_TOPO):
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)                          # MISMO constructor que usan todos los brazos
    grado_inicial = np.array([len(sustrato["adj"][i]) for i in range(N)], dtype=float)
    budget_por_nodo = MA._cupo_variable(grado_inicial, p["kcap"])   # MA sin tocar -- MISMO B_i que C2-presupuesto-variable
    sustrato = dinamica_B0_presupuesto_variable_azar(sustrato, p, rng, n_sweeps, budget_por_nodo)
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
# 4) DRIVER -- 5 brazos sobre el mismo lote de reglas
# ============================================================================================
BRAZOS = ("C2-hard", "C2-hibrido", "C2-presupuesto-variable", "C2-presupuesto-variable-azar", "C0")


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
        return MA.correr_regla_coarse_hibrido(p, modo="soporte")                            # MA sin tocar
    elif brazo == "C2-presupuesto-variable":
        return PV.correr_regla_coarse_presupuesto_variable(p)                                # PV sin tocar
    elif brazo == "C2-presupuesto-variable-azar":
        return correr_regla_coarse_presupuesto_variable_azar(p)
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
            print(f"  [{etiqueta}] {p['rule_id']:<16} {brazo:<28} clase={r['clase']:<24} "
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
    print("\n" + "=" * 120)
    print(titulo)
    print("=" * 120)
    for brazo in BRAZOS:
        r = resumen[brazo]
        print(f"  {brazo:<28} n={r['n']:<4} clases={r['cnt']}  frac_III={r['frac_III']*100:5.1f}%  "
              f"frac_III+IV={r['frac_IIImasIV']*100:5.1f}%  grado_medio={r['grado_medio']:.2f}  "
              f"n_aristas_medio={r['n_aristas_medio']:.1f}  diam_medio={r['diam_medio']:.2f}  "
              f"pendiente_media={r['pendiente_media']:.3f}  pendiente_mediana={r['pendiente_mediana']:.3f}")


def comparaciones_pareadas(filas_resumen):
    """Compara, regla por regla (mismo rule_id), la pendiente continua entre pares de brazos."""
    por_regla = {}
    for f in filas_resumen:
        por_regla.setdefault(f["rule_id"], {})[f["brazo"]] = f["pendiente"]
    pares = [
        ("C2-hard", "C2-hibrido"),
        ("C2-hard", "C2-presupuesto-variable"),
        ("C2-hard", "C2-presupuesto-variable-azar"),
        ("C2-presupuesto-variable", "C2-presupuesto-variable-azar"),
        ("C2-hibrido", "C2-presupuesto-variable-azar"),
        ("C2-presupuesto-variable-azar", "C0"),
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
    print("\n" + "=" * 120)
    print("COMPARACIONES PAREADAS (pendiente continua, misma regla en ambos brazos)")
    print("=" * 120)
    for (a, b), r in pareadas.items():
        print(f"  {a:<28} vs {b:<28} n={r['n']:<3} {a} gana={r['gana_A']:<3} {b} gana={r['gana_B']:<3} "
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
        guardar_csv(filas_raw, f"{_HERE}/cs090_fase5_control_azar_elastico_piloto_raw.csv")
        guardar_csv(filas_resumen, f"{_HERE}/cs090_fase5_control_azar_elastico_piloto_resumen.csv")
        print(f"\nPILOTO terminado en {(time.time()-t0)/60:.1f} min")
    else:
        t0 = time.time()
        filas_raw, filas_resumen, admitidas, descartadas = correr_lote(
            n_seeds=20, seed_base=SEED_BASE + 1, presupuesto_seg=50 * 60, etiqueta="COMPLETO")
        resumen = resumen_por_brazo(filas_resumen)
        imprimir_resumen("COMPLETO -- resumen por brazo", resumen)
        pareadas = comparaciones_pareadas(filas_resumen)
        imprimir_comparaciones(pareadas)
        guardar_csv(filas_raw, f"{_HERE}/cs090_fase5_control_azar_elastico_resultados.csv")
        guardar_csv(filas_resumen, f"{_HERE}/cs090_fase5_control_azar_elastico_resumen.csv")
        print(f"\nCOMPLETO terminado en {(time.time()-t0)/60:.1f} min")
        print("Fin. No se declara cierre ni veredicto -- números arriba, lectura final de Alexis.")
