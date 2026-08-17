"""
cs090_fase6_o3e_analizar.py — FASE VI, tarea O3-E: estadística pareada memoria vs sin-memoria
=========================================================================================================

Lee `cs090_fase6_o3e_memoria_crudo.csv` (una fila por corrida, dos filas por par: brazo `mem` y brazo
`nomem` de la MISMA regla base) y produce, con la misma disciplina que el resto de la línea:

  1. Test de SIGNOS (binomial de dos colas, p=0.5) — ¿en cuántos pares gana el brazo con memoria?
  2. Wilcoxon de rangos con signo pareado — usa además el tamaño de cada diferencia.
  3. Lo mismo para los observables secundarios: κ_V, número de sumideros, pendiente corregida.
  4. CONTROL DE CONFOUND DE DENSIDAD: cuánto difieren los dos brazos en número de aristas y grado medio,
     y si la diferencia de masa se explica por la diferencia de aristas o por la de geometría
     (correlación de Spearman de Δmasa contra Δaristas y contra Δpendiente, más una regresión lineal de
     Δmasa sobre esos dos regresores para ver qué queda cuando se descuenta la geometría).

No declara cierre ni veredicto: imprime números y los deja también en un CSV de diferencias por par.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
RUTA_CRUDO = f"{_HERE}/cs090_fase6_o3e_memoria_crudo.csv"
RUTA_DIFS = f"{_HERE}/cs090_fase6_o3e_diferencias_por_par.csv"

OBSERVABLES = [
    ("fraccion_masa_en_sumideros", "fracción de masa acretada (PRINCIPAL)"),
    ("masa_sumideros_final", "masa absoluta en sumideros"),
    ("kappa_v_agregado", "κ_V agregado"),
    ("n_sumideros", "nº de sumideros"),
    ("pendiente_corregida", "pendiente corregida del grafo (geometría)"),
    ("n_aristas_grafo_final", "nº de aristas del grafo (CONFOUND a vigilar)"),
    ("grado_medio_grafo_final", "grado medio del grafo (CONFOUND a vigilar)"),
]


def _f(x):
    if x in (None, "", "None", "nan"):
        return float("nan")
    return float(x)


def cargar():
    with open(RUTA_CRUDO) as fh:
        filas = list(csv.DictReader(fh))
    por_par = defaultdict(dict)
    for r in filas:
        por_par[int(r["par_seed"])][r["brazo"]] = r
    pares = [(s, d["mem"], d["nomem"]) for s, d in sorted(por_par.items()) if "mem" in d and "nomem" in d]
    return pares


def test_pareado(dif, etiqueta):
    """Signos (binomial de dos colas) + Wilcoxon de rangos con signo, sobre las diferencias mem-nomem."""
    dif = np.asarray([d for d in dif if np.isfinite(d)], dtype=float)
    n = len(dif)
    n_pos = int((dif > 0).sum()); n_neg = int((dif < 0).sum()); n_cero = int((dif == 0).sum())
    n_efectivo = n_pos + n_neg
    p_signos = stats.binomtest(n_pos, n_efectivo, 0.5).pvalue if n_efectivo > 0 else float("nan")
    if n_efectivo >= 5:
        try:
            _, p_wil = stats.wilcoxon(dif[dif != 0])
        except ValueError:
            p_wil = float("nan")
    else:
        p_wil = float("nan")
    print(f"\n  {etiqueta}")
    print(f"    n={n} pares · mem>nomem en {n_pos}, mem<nomem en {n_neg}, empates {n_cero}")
    print(f"    mediana Δ(mem−nomem) = {np.median(dif):+.6g} · media = {np.mean(dif):+.6g} "
          f"(sd {np.std(dif, ddof=1) if n > 1 else float('nan'):.6g})")
    print(f"    signos p={p_signos:.4g} · Wilcoxon p={p_wil:.4g}")
    return dict(observable=etiqueta, n=n, n_pos=n_pos, n_neg=n_neg, n_cero=n_cero,
                mediana=float(np.median(dif)), media=float(np.mean(dif)),
                p_signos=float(p_signos), p_wilcoxon=float(p_wil))


def solapamiento_de_grafos(seeds, N=2000, n_sweeps=14):
    """Reconstruye los dos grafos de cada par y mide CUÁNTO se parecen (Jaccard del conjunto de
    aristas, y cuántas aristas están en uno y no en el otro). Es la magnitud real de la manipulación:
    si los dos grafos comparten el 98% de sus aristas, cualquier diferencia de masa que aparezca
    proviene de un empujón muy chico, y conviene decirlo antes de interpretarla."""
    sys.path.insert(0, _HERE)
    import cs090_fase6_o3e_memoria as O3E
    out = []
    for s in seeds:
        _, mm, _ = O3E.reconstruir(s, ventana_memoria=None, N=N, n_sweeps=n_sweeps)
        _, mn, _ = O3E.reconstruir(s, ventana_memoria=1, N=N, n_sweeps=n_sweeps)
        em = set((i, j) for i in range(N) for j in mm["adj_final"][i] if j > i)
        en = set((i, j) for i in range(N) for j in mn["adj_final"][i] if j > i)
        inter, union = len(em & en), len(em | en)
        out.append(dict(par_seed=s, aristas_mem=len(em), aristas_nomem=len(en),
                        solo_mem=len(em - en), solo_nomem=len(en - em),
                        jaccard=inter / union if union else float("nan")))
    return out


def main():
    pares = cargar()
    print(f"[O3-E] {len(pares)} pares emparejados (misma regla base, brazo con memoria vs sin memoria)")

    difs = {clave: [] for clave, _ in OBSERVABLES}
    filas_dif = []
    for seed, rm, rn in pares:
        fila = dict(par_seed=seed, rule_id_original=rm.get("rule_id_original"),
                    clase_corregida_historica=rm.get("clase_corregida_historica"),
                    K=rm.get("K"), kcap=rm.get("kcap"))
        for clave, _ in OBSERVABLES:
            a, b = _f(rm.get(clave)), _f(rn.get(clave))
            difs[clave].append(a - b)
            fila[f"{clave}_mem"] = a
            fila[f"{clave}_nomem"] = b
            fila[f"d_{clave}"] = a - b
        # cuánto difieren realmente los dos grafos, en porcentaje de aristas
        na, nb = _f(rm.get("n_aristas_grafo_final")), _f(rn.get("n_aristas_grafo_final"))
        fila["pct_dif_aristas"] = 100.0 * (na - nb) / nb if nb else float("nan")
        fila["hist_usado_suma_mem"] = _f(rm.get("hist_usado_suma"))
        fila["hist_usado_suma_nomem"] = _f(rn.get("hist_usado_suma"))
        filas_dif.append(fila)

    print("\n" + "=" * 100)
    print("TESTS PAREADOS (diferencia = brazo CON memoria − brazo SIN memoria)")
    print("=" * 100)
    resumen = [test_pareado(difs[c], etq) for c, etq in OBSERVABLES]

    # ---------------- confound de densidad ----------------
    pct = np.array([f["pct_dif_aristas"] for f in filas_dif], dtype=float)
    print("\n" + "=" * 100)
    print("CONFOUND DE DENSIDAD — cuánto difieren los dos brazos en número de aristas")
    print("=" * 100)
    print(f"  Δ%aristas (mem vs nomem): mediana {np.median(pct):+.2f}% · rango "
          f"[{np.min(pct):+.2f}%, {np.max(pct):+.2f}%] · |Δ%| medio {np.mean(np.abs(pct)):.2f}%")

    # ---------------- ¿pasa por la geometría? ----------------
    dm = np.array(difs["fraccion_masa_en_sumideros"], dtype=float)
    dp = np.array(difs["pendiente_corregida"], dtype=float)
    da = np.array(difs["n_aristas_grafo_final"], dtype=float)
    ok = np.isfinite(dm) & np.isfinite(dp) & np.isfinite(da)
    print("\n" + "=" * 100)
    print("¿EL EFECTO EN MASA PASA POR LA GEOMETRÍA O POR OTRA VÍA?")
    print("=" * 100)
    if ok.sum() >= 4:
        rho_p, p_p = stats.spearmanr(dm[ok], dp[ok])
        rho_a, p_a = stats.spearmanr(dm[ok], da[ok])
        print(f"  Spearman Δmasa vs Δpendiente_corregida : ρ={rho_p:+.3f} (p={p_p:.3g})")
        print(f"  Spearman Δmasa vs Δn_aristas           : ρ={rho_a:+.3f} (p={p_a:.3g})")
        X = np.column_stack([np.ones(ok.sum()), dp[ok], da[ok]])
        coef, *_ = np.linalg.lstsq(X, dm[ok], rcond=None)
        resid = dm[ok] - X @ coef
        print(f"  regresión Δmasa ~ 1 + Δpendiente + Δaristas: intercepto={coef[0]:+.6g} "
              f"(= Δmasa esperado con geometría y densidad iguales), "
              f"β_pendiente={coef[1]:+.6g}, β_aristas={coef[2]:+.6g}")
        print(f"  residuo sd={np.std(resid, ddof=1):.6g} · R²="
              f"{1 - np.var(resid)/np.var(dm[ok]) if np.var(dm[ok])>0 else float('nan'):.3f}")
    else:
        print("  (menos de 4 pares con todos los valores finitos — no se ajusta nada)")

    campos = list(filas_dif[0].keys())
    with open(RUTA_DIFS, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas_dif)
    print(f"\n[diferencias por par] -> {RUTA_DIFS}")
    return resumen, filas_dif


if __name__ == "__main__":
    main()
