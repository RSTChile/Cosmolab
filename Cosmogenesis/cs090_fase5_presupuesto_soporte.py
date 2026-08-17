"""
CS090 — FASE V-C2-C, continuación (opción a): ¿EL SOPORTE LOCAL ES EL INGREDIENTE QUE FALTABA?
=================================================================================================
QUIÉN SOY: archivo NUEVO (ningún script congelado ni `cs090_fase5_presupuesto_emergente.py` se tocan)
que sigue de `FASE5_presupuesto_emergente_CS.md` (F5-C2-C). Ese experimento comparó C2-hard (cupo fijo
de aristas) contra C2-budget (presupuesto de costo con 3 señales: historia/holonomía/compatibilidad) y
encontró que C2-budget NO se parecía a C2-hard (15.0% Clase III) — se parecía a C2-random (15.0%
también), mientras C2-hard daba 45.0%. La lectura alternativa #2 de ese informe (§6) planteaba: quizás
lo que realmente empuja a Clase III en C2-hard es el criterio de **SOPORTE LOCAL** (vecinos compartidos)
que usa `_enforce_kcap` para decidir qué arista quitar — y ese criterio está TOTALMENTE AUSENTE de
`c_ij` (que sólo usa historia/holonomía/compatibilidad, nada de topología/vecindad). Alexis pidió
explícitamente seguir esa pista (opción a).

—— QUÉ SE AGREGA: SOPORTE LOCAL COMO 4º COMPONENTE DE c_ij ——
`MOT._enforce_kcap` (cs090_fase5_motor.py, línea 136-148) calcula, para cada arista i-j que un nodo i
debe evaluar, `soporte(i,j) = len(adj[i] & adj[j])` — cuántos vecinos comparten i y j — y CONSERVA las
aristas de MAYOR soporte (más vecinos en común = arista más "anclada" en la vecindad local, más barata
de conservar). Ese mismo cálculo, literal, se reusa acá: para cada arista viva (i,j),
`soporte_ij = len(adj[i] & adj[j])` (exactamente `len(adj[i] & adj[j])`, sin aproximar).

Conversión de soporte (más alto = mejor) a COSTO (más alto = peor, para que sea consistente con las
otras 3 señales, donde costo alto = arista cara/a podar primero):

    costo_soporte_ij = 1.0 / (1.0 + soporte_ij)

Elegido por 3 razones documentadas: (1) es monótona decreciente en soporte (soporte=0 -> costo=1.0,
máximo; soporte grande -> costo -> 0), igual que se necesita; (2) acotada en (0,1], sin necesidad de
truncar valores negativos ni tocar la escala de las otras señales antes de normalizar; (3) es la
transformación más simple y estándar para "más X es mejor" -> "costo", sin introducir un parámetro
nuevo elegido a mano (no hay escala ni umbral libre).

NORMALIZACIÓN — igual criterio que las 3 señales originales de `cs090_fase5_presupuesto_emergente.py`
(cada componente se divide por su propia media sobre las aristas VIVAS, "1.0 = costo promedio"):

    c_ij = ( norm(hist) + norm(hol) + norm(compat) + norm(costo_soporte) ) / 4.0

DECISIÓN DOCUMENTADA — agregar, no reemplazar: la tarea permitía "agregar un 4º componente" o
"reemplazar algún componente débil". Se optó por AGREGAR (4 componentes, pesos iguales 1/4 cada uno) en
vez de reemplazar ninguno de los 3 originales, por dos razones: (a) no hay evidencia de que alguno de
los 3 originales sea "débil" — el informe anterior no aisló el aporte individual de cada uno, así que
quitar uno sería una apuesta no informada; (b) mantener los 3 originales intactos permite que
"C2-budget-soporte" sea un superconjunto estricto de "C2-budget-original" (misma fórmula + 1 término),
lo que hace la comparación entre ambos brazos más limpia: cualquier diferencia observada es atribuible
al soporte agregado, no a haber quitado o repesado algo más.

—— LOS 5 BRAZOS ——
  1. C2-hard            — `MOT._enforce_kcap`, sin cambios (control/baseline).
  2. C2-budget-original — el `c_ij` de 3 componentes de `cs090_fase5_presupuesto_emergente.py`,
                           reusado TAL CUAL (`PE.correr_regla_coarse_presupuesto`, sin tocar ese
                           archivo) — para tener el mismo baseline recalculado EN ESTA MISMA CORRIDA
                           (mismas semillas, mismo momento), no contra los números archivados del
                           informe anterior.
  3. C2-budget-soporte  — nuevo: `c_ij` de 4 componentes (historia+holonomía+compatibilidad+soporte),
                           conserva las aristas más baratas hasta agotar el presupuesto (mismo
                           mecanismo knapsack greedy que budget-original).
  4. C2-random          — mismo cálculo de "cuántas aristas debe soltar cada nodo" que
                           C2-budget-soporte (RECALCULADO para esta variante, la magnitud de poda de
                           4 componentes no es la misma que la de 3), pero elige cuáles soltar al azar.
  5. C0                 — sin límite de escala, sin cambios.

Diseño de control: MISMO lote de 20 reglas admitidas A2-B0-C2 (filtro P1-P5 real) que
`cs090_fase5_presupuesto_emergente.py` usó en su corrida completa — mismo `seed_base` (`SEED_BASE+1`,
ver docstring de ese archivo) para que las reglas (K,J,noise,meandeg,kcap,seed) sean IDÉNTICAS a las de
esa corrida, y los 5 brazos de ESTA corrida (incluidos C2-hard y C0, recalculados frescos acá) sean
comparables entre sí en el mismo momento. No se reusan números archivados del CSV anterior para ninguna
comparación cuantitativa — sólo como referencia narrativa en el informe.

No se corre Phantom. No se declara cierre ni veredicto — se reportan números, la lectura final es de
Alexis. No se modifica ningún script congelado ni `cs090_fase5_presupuesto_emergente.py` (verificable
con git diff / mtime). No se hacen commits de git.
"""
from __future__ import annotations
import csv, sys, time
import numpy as np
from collections import defaultdict, Counter

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)

import cs090_fase5_generador as GEN
import cs090_fase5_motor as MOT
import cs082_fase4_4sustratos as C82                       # _circ_mean_update, _holonomia_triangulos
import cs090_fase5_presupuesto_emergente as PE              # reuso directo del c_ij de 3 componentes
                                                              # (brazo C2-budget-original) -- NO SE TOCA
from cs090_fase5_clasificador import clasificar_regla

EJE_A, EJE_B = "A2", "B0"
N_GRANDE = PE.N_GRANDE                 # 2000
ESCALAS_B = PE.ESCALAS_B               # (1,2,4,8,16)
N_SWEEPS = PE.N_SWEEPS                 # 14
N_SEEDS_NULL_TOPO = PE.N_SEEDS_NULL_TOPO  # 3
SEED_BASE = PE.SEED_BASE               # 90210 -- MISMA base que la tarea anterior (comparabilidad)


# ============================================================================================
# 1) COSTO RELACIONAL c_ij CON SOPORTE -- 4 señales: las 3 de PE._costos_relacionales (historia,
#    holonomía, compatibilidad -- se replican acá, mismo motivo que PE: el motor congelado no las
#    expone como funciones aparte) MÁS soporte local (EXACTAMENTE el cálculo de MOT._enforce_kcap).
# ============================================================================================
def _costos_relacionales_soporte(adj, N, S, K, flip_count, rng, max_tri=1500, eps=1e-9):
    """Devuelve (edges, c_ij) para el grafo VIVO actual, con 4 componentes normalizados a media 1.0
    cada uno (mismo criterio de normalización que PE._costos_relacionales) y promediados con pesos
    iguales 1/4 (ver docstring del módulo, sección de la decisión de AGREGAR en vez de reemplazar)."""
    edges = sorted(set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i]))
    if not edges:
        return edges, {}
    hist = np.array([flip_count.get(e, 0) for e in edges], dtype=float)

    E_estado = {(i, j): abs(S[i] - S[j]) % K for (i, j) in edges}
    compat = np.array([min(v, K - v) for v in E_estado.values()], dtype=float)

    # --- replicado verbatim de la agregación de holonomía por arista (idéntico a PE._costos_relacionales
    #     y a la lógica interna de MOT._costo_y_podar) -- no expuesto aparte en el motor congelado.
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

    # --- 4º componente NUEVO: soporte local, EXACTAMENTE `len(adj[i] & adj[j])` como MOT._enforce_kcap
    #     (motor, línea 144), convertido a costo (ver fórmula y justificación en el docstring del módulo).
    soporte = np.array([float(len(adj[i] & adj[j])) for (i, j) in edges], dtype=float)
    costo_soporte = 1.0 / (1.0 + soporte)

    def _norm(a):
        m = a.mean()
        return a / (m + eps) if m > eps else np.ones_like(a)

    c = (_norm(hist) + _norm(hol) + _norm(compat) + _norm(costo_soporte)) / 4.0
    return edges, {e: float(v) for e, v in zip(edges, c)}


# ============================================================================================
# 2) ENFORCEMENT -- idéntico patrón secuencial nodo-por-nodo que PE._enforce_relacional, sólo cambia
#    la función de costos que llama (4 componentes en vez de 3).
# ============================================================================================
def _enforce_relacional_soporte(adj, N, S, K, flip_count, rng, budget, modo):
    """modo='costo' -> C2-budget-soporte: conserva las aristas MÁS BARATAS (con soporte incluido) hasta
                        agotar `budget`.
       modo='azar'  -> C2-random: MISMA cantidad de aristas a soltar por nodo bajo el presupuesto de
                        4 componentes (RECALCULADA acá, no reusa la magnitud de PE porque la distribución
                        de costos con soporte es distinta), pero elige cuáles al azar."""
    edges, costos = _costos_relacionales_soporte(adj, N, S, K, flip_count, rng)
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
        costos_nb = sorted(((c_por_dirigida[(i, j)], j) for j in nb))
        acumulado, n_keep = 0.0, 0
        for c, _j in costos_nb:
            if acumulado + c > budget:
                break
            acumulado += c
            n_keep += 1
        if n_keep >= len(nb):
            continue
        n_soltar = len(nb) - n_keep
        if modo == "costo":
            soltar = set(j for _c, j in costos_nb[n_keep:])
        elif modo == "azar":
            soltar = set(rng.choice(nb, size=n_soltar, replace=False).tolist())
        else:
            raise ValueError(f"modo desconocido: {modo}")
        for j in soltar:
            adj[i].discard(j); adj[j].discard(i)


# ============================================================================================
# 3) DINÁMICA A2-B0 con presupuesto+soporte -- copia adaptada de PE.dinamica_B0_presupuesto: único
#    cambio real es llamar a `_enforce_relacional_soporte` en vez de `PE._enforce_relacional`.
# ============================================================================================
def dinamica_B0_presupuesto_soporte(sustrato, p, rng, n_sweeps, modo):
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
            _enforce_relacional_soporte(adj, N, S, K, flip_count, rng, budget=p["kcap"], modo=modo)
        cur = set((i, j) if i < j else (j, i) for i in range(N) for j in adj[i])
        for e in prev_edges ^ cur:
            flip_count[e] += 1
        prev_edges = cur
    sustrato["S"] = S; sustrato["adj"] = adj; sustrato["flip_count"] = flip_count
    edges = sorted(prev_edges)
    E_estado = {(i, j): abs(S[i] - S[j]) % K for (i, j) in edges}
    triangles = MOT._muestrear_triangulos(adj, N, rng)
    conservar = MOT._costo_y_podar(edges, flip_count, E_estado, K, triangles)
    for (i, j) in edges:
        if (i, j) not in conservar:
            adj[i].discard(j); adj[j].discard(i)
    return sustrato


def correr_regla_coarse_presupuesto_soporte(p, modo, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                             n_seeds_null_topo=N_SEEDS_NULL_TOPO):
    rng = np.random.default_rng(p["seed"] * 5000 + N)
    sustrato = MOT.construir_A2(N, rng, p)
    sustrato = dinamica_B0_presupuesto_soporte(sustrato, p, rng, n_sweeps, modo)
    m = MOT.medir(sustrato, p, rng)
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
BRAZOS = ("C2-hard", "C2-budget-original", "C2-budget-soporte", "C2-random", "C0")


def _correr_brazo(brazo, p):
    if brazo == "C2-hard":
        p2 = dict(p); p2["eje_C"] = "C2"
        return MOT.correr_regla_coarse(p2, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                        n_seeds_null_topo=N_SEEDS_NULL_TOPO)
    elif brazo == "C0":
        p2 = dict(p); p2["eje_C"] = "C0"
        return MOT.correr_regla_coarse(p2, N=N_GRANDE, n_sweeps=N_SWEEPS, escalas_b=ESCALAS_B,
                                        n_seeds_null_topo=N_SEEDS_NULL_TOPO)
    elif brazo == "C2-budget-original":
        return PE.correr_regla_coarse_presupuesto(p, modo="costo")            # PE sin tocar
    elif brazo == "C2-budget-soporte":
        return correr_regla_coarse_presupuesto_soporte(p, modo="costo")
    elif brazo == "C2-random":
        return correr_regla_coarse_presupuesto_soporte(p, modo="azar")
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
                           grado_medio=float(np.mean([f["grado_medio_b1"] for f in fb])) if fb else float("nan"),
                           n_aristas_medio=float(np.mean([f["n_aristas_b1"] for f in fb])) if fb else float("nan"),
                           diam_medio=float(np.mean([f["diam_b1"] for f in fb])) if fb else float("nan"),
                           pendiente_media=float(np.mean([f["pendiente"] for f in fb])) if fb else float("nan"))
    return out


def imprimir_resumen(titulo, resumen):
    print("\n" + "=" * 110)
    print(titulo)
    print("=" * 110)
    for brazo in BRAZOS:
        r = resumen[brazo]
        print(f"  {brazo:<20} n={r['n']:<4} clases={r['cnt']}  frac_III={r['frac_III']*100:5.1f}%  "
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
        guardar_csv(filas_raw, f"{_HERE}/cs090_fase5_presupuesto_soporte_piloto_raw.csv")
        guardar_csv(filas_resumen, f"{_HERE}/cs090_fase5_presupuesto_soporte_piloto_resumen.csv")
        print(f"\nPILOTO terminado en {(time.time()-t0)/60:.1f} min")
    else:
        t0 = time.time()
        filas_raw, filas_resumen, admitidas, descartadas = correr_lote(
            n_seeds=20, seed_base=SEED_BASE + 1, presupuesto_seg=50 * 60, etiqueta="COMPLETO")
        resumen = resumen_por_brazo(filas_resumen)
        imprimir_resumen("COMPLETO -- resumen por brazo", resumen)
        guardar_csv(filas_raw, f"{_HERE}/cs090_fase5_presupuesto_soporte_resultados.csv")
        guardar_csv(filas_resumen, f"{_HERE}/cs090_fase5_presupuesto_soporte_resumen.csv")
        print(f"\nCOMPLETO terminado en {(time.time()-t0)/60:.1f} min")
        print("Fin. No se declara cierre ni veredicto -- números arriba, lectura final de Alexis.")
