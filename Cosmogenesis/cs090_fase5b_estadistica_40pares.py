"""
cs090_fase5b_estadistica_40pares.py -- FASE V-B, escalado a ~40 pares: prueba estadistica FORMAL simple
sobre el CSV consolidado final (cs090_fase5b_TOTAL_40pares.csv, producido por
cs090_fase5b_consolidar_40pares.py). Dos pruebas, ambas sobre las diferencias PAREADAS
(Clase III - Clase I) por par:

  1. TEST DE SIGNOS (binomial exacto, dos colas): bajo H0 "el signo de la diferencia por par es un
     resultado de una moneda justa (p=0.5), es decir Clase III y Clase I son igualmente probables de
     tener mayor fraccion de masa / kappa_V en un par dado" -- se calcula la probabilidad de observar
     tantos-o-mas signos iguales al observado por puro azar. Usa scipy.stats.binomtest.
  2. WILCOXON SIGNED-RANK (dos colas): no asume normalidad de las diferencias, apropiado para n
     pequeno-mediano y datos pareados; prueba H0 "la distribucion de las diferencias pareadas es
     simetrica alrededor de 0" (mas fuerte que el test de signos: usa la MAGNITUD de la diferencia, no
     solo el signo). Usa scipy.stats.wilcoxon.

IMPORTANTE (pedido explicito de la tarea, no se viola): estos p-valores prueban una hipotesis narrow y
especifica sobre el SIGNO/SIMETRIA de la diferencia pareada -- NO prueban "Clase III causa mas fraccion
de masa que Clase I" en un sentido causal, ni "A2-B0-C2 esta confirmado". Un p-valor bajo es evidencia
en contra de la hipotesis nula de azar puro en el signo/magnitud del efecto -- no es lo mismo que
"confirmado". No se declara cierre ni veredicto: se reportan los numeros y su significado tecnico exacto,
la lectura final es de Alexis.

No se modifica ningun script anterior. No corre Phantom.
"""
from __future__ import annotations
import csv
import sys

from scipy import stats

RUTA_TOTAL = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5b_TOTAL_40pares.csv"


def cargar_comparaciones_por_par(ruta=RUTA_TOTAL):
    from collections import defaultdict
    filas = list(csv.DictReader(open(ruta)))
    por_par = defaultdict(dict)
    for f in filas:
        por_par[f["par"]][f["rol"]] = f

    cmp_pares = []
    for par, roles in por_par.items():
        if "I" not in roles or "III" not in roles:
            continue
        fI, fIII = roles["I"], roles["III"]
        d_frac = float(fIII["fraccion_masa_en_sumideros"]) - float(fI["fraccion_masa_en_sumideros"])
        kv_I, kv_III = fI["kappa_v_agregado"], fIII["kappa_v_agregado"]
        d_kv = (float(kv_III) - float(kv_I)) if kv_I not in (None, "", "None") and kv_III not in (None, "", "None") else None
        cmp_pares.append(dict(par=par, match_exacto=fI["match_exacto_K_kcap"],
                               d_frac=d_frac, d_kv=d_kv))
    return cmp_pares


def test_signos(diffs, etiqueta):
    """Test de signos binomial exacto (dos colas), H0: P(signo=+)=0.5. Ignora empates (diff==0, no
    hay ninguno esperado con datos continuos pero se documenta el criterio de todas formas)."""
    no_empate = [d for d in diffs if d != 0]
    n = len(no_empate)
    n_pos = sum(1 for d in no_empate if d > 0)
    n_neg = n - n_pos
    k_mayor = max(n_pos, n_neg)
    res = stats.binomtest(k_mayor, n, 0.5, alternative="two-sided")
    print(f"  [{etiqueta}] test de signos: n={n} (empates excluidos), III>I en {n_pos}/{n} "
          f"({100*n_pos/n:.1f}%), I>III en {n_neg}/{n} -- p (binomial exacta, dos colas, H0 p=0.5) "
          f"= {res.pvalue:.5f}")
    return dict(n=n, n_pos=n_pos, n_neg=n_neg, p_valor=res.pvalue)


def test_wilcoxon(diffs, etiqueta):
    """Wilcoxon signed-rank, dos colas, H0: distribucion de diferencias pareadas simetrica alrededor de
    0. zero_method='wilcox' descarta diferencias exactamente 0 (no se esperan con datos continuos)."""
    no_cero = [d for d in diffs if d != 0]
    try:
        res = stats.wilcoxon(no_cero, alternative="two-sided", zero_method="wilcox")
        print(f"  [{etiqueta}] Wilcoxon signed-rank: n={len(no_cero)}  estadistico W={res.statistic:.2f}  "
              f"p (dos colas, H0 simetria alrededor de 0) = {res.pvalue:.5f}")
        return dict(n=len(no_cero), estadistico=res.statistic, p_valor=res.pvalue)
    except Exception as e:
        print(f"  [{etiqueta}] Wilcoxon FALLO: {type(e).__name__}: {e}")
        return dict(n=len(no_cero), estadistico=None, p_valor=None, error=str(e))


def main():
    cmp_pares = cargar_comparaciones_por_par()
    exactos = [c for c in cmp_pares if str(c["match_exacto"]).lower() == "true"]

    print(f"n pares totales con I+III completos: {len(cmp_pares)}")
    print(f"n pares SOLO match exacto K=kcap: {len(exactos)}")

    resultados = {}
    print("\n=== FRACCION DE MASA EN SUMIDEROS (III - I) ===")
    d_frac_todos = [c["d_frac"] for c in cmp_pares]
    d_frac_exactos = [c["d_frac"] for c in exactos]
    resultados["signos_frac_todos"] = test_signos(d_frac_todos, "fraccion, TODOS")
    resultados["wilcoxon_frac_todos"] = test_wilcoxon(d_frac_todos, "fraccion, TODOS")
    resultados["signos_frac_exactos"] = test_signos(d_frac_exactos, "fraccion, SOLO exactos")
    resultados["wilcoxon_frac_exactos"] = test_wilcoxon(d_frac_exactos, "fraccion, SOLO exactos")

    print("\n=== KAPPA_V AGREGADO (III - I) ===")
    d_kv_todos = [c["d_kv"] for c in cmp_pares if c["d_kv"] is not None]
    d_kv_exactos = [c["d_kv"] for c in exactos if c["d_kv"] is not None]
    resultados["signos_kv_todos"] = test_signos(d_kv_todos, "kappa_V, TODOS")
    resultados["wilcoxon_kv_todos"] = test_wilcoxon(d_kv_todos, "kappa_V, TODOS")
    resultados["signos_kv_exactos"] = test_signos(d_kv_exactos, "kappa_V, SOLO exactos")
    resultados["wilcoxon_kv_exactos"] = test_wilcoxon(d_kv_exactos, "kappa_V, SOLO exactos")

    print("\n=== RECORDATORIO DE INTERPRETACION (no se declara cierre) ===")
    print("  H0 test de signos: el signo de la diferencia por par es una moneda justa (p=0.5) -- NO es")
    print("  lo mismo que 'no hay ninguna relacion entre clase y fisica'. H0 Wilcoxon: la distribucion")
    print("  de diferencias pareadas es simetrica alrededor de 0 (mas exigente, usa magnitud). Un p bajo")
    print("  es evidencia contra esas H0 puntuales, no una confirmacion de A2-B0-C2. Lectura final: Alexis.")

    return resultados


if __name__ == "__main__":
    main()
