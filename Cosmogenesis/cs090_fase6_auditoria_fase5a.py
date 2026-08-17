"""
cs090_fase6_auditoria_fase5a.py — ¿el bug del diámetro toca al BARRIDO GLOBAL de Fase V-A?
============================================================================================

EL PROBLEMA: NO SE PUEDE RE-MEDIR
----------------------------------
El barrido de Fase V-A (`FASE5A_completo_resultado_CS.md`, 150 reglas corridas de 180 planeadas,
distribución global {I:99=66%, II:43=29%, III:8=5%, IV:0}) es el único de esta línea que **no se puede
volver a medir**: su CSV no guarda `seed` por regla, y las semillas se derivaban dentro del script con

    seed_base = abs(hash((eje_A, eje_B, eje_C, "paso1"))) % 100000        # cs090_fase5_completo.py

`hash()` de strings en Python está aleatorizado por proceso salvo que se fije `PYTHONHASHSEED`, que no
quedó registrado. O sea: ni siquiera conociendo el código se puede regenerar el mismo lote de reglas.
Esto se documenta como **limitación**, no se adivina.

LO QUE SÍ SE PUEDE HACER: BUSCAR LA FIRMA DEL BUG EN LOS DATOS GUARDADOS
--------------------------------------------------------------------------
`cs090_fase5_completo_resultados.csv` sí guarda, por regla y por escala de coarse-graining, el diámetro
REAL, el diámetro NULL y la fracción de componente gigante. Con eso se aplican tres pruebas
independientes, todas basadas en propiedades que el bug viola necesariamente:

  PRUEBA 1 — IMPOSIBILIDAD GEOMÉTRICA. Agrupar nodos en cajas conexas nunca puede ALARGAR un camino;
  por lo tanto diám(b=1) < diám(b=2) es imposible si las dos mediciones cayeron en la misma componente.
  En los 15 casos confirmados de la línea A2-B0-C2 esta desigualdad se cumple siempre (diám b=1 = 1-3
  contra 8-19 en b=2). Es una prueba de detección, no de descarte.

  PRUEBA 2 — CALIBRACIÓN CONTRA VERDAD DE TERRENO. Sobre las 430 reglas re-medidas de verdad
  (`cs090_fase6_remedicion_430.csv`) se conoce, para cada una, si descarriló y cuánto valía diám(b=1).
  Eso da un umbral empírico: el valor máximo de diám(b=1) observado entre las descarriladas, y el
  cociente diám(b=1)/diám(b=2) mínimo observado entre las NO descarriladas. Cualquier regla de V-A que
  caiga del lado "sospechoso" de esos dos números se marca.

  PRUEBA 3 — CONDICIÓN NECESARIA DE FRAGMENTACIÓN. El bug requiere que el nodo no aislado de índice más
  bajo caiga fuera de la componente gigante, o sea que el grafo tenga fragmentos sueltos. Se reporta la
  fracción de gigante mínima por combinación.

No corre Phantom. No toca ningún script. No declara veredicto.
"""
from __future__ import annotations
import csv
import sys
import collections
import statistics

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"


def cargar_va():
    """El CSV de V-A repite el mismo rule_id en las dos pasadas del barrido (pasada 1 = r0..r4 con un
    seed_base, pasada 2 = r0..r4 con otro), así que se desambigua por orden de aparición."""
    rows = list(csv.DictReader(open(f"{HERE}/cs090_fase5_completo_resultados.csv")))
    vistos = collections.Counter()
    g, meta = collections.defaultdict(dict), {}
    for r in rows:
        k0 = (r["rule_id"], r["escala_b"])
        vistos[k0] += 1
        k = (r["rule_id"], vistos[k0])
        g[k][int(r["escala_b"])] = r
        meta[k] = dict(combo=f"{r['eje_A']}-{r['eje_B']}-{r['eje_C']}", clase=r["clase_final"],
                       pendiente=float(r["pendiente_real"]))
    return g, meta


def calibrar():
    """Umbrales empíricos desde las 430 reglas re-medidas de verdad."""
    try:
        rows = list(csv.DictReader(open(f"{HERE}/cs090_fase6_remedicion_430.csv")))
    except FileNotFoundError:
        return None
    desc, sano = [], []
    for r in rows:
        d = eval(r["diam_viejo"])
        (desc if r["descarrila_b1"] == "True" else sano).append(d)
    return dict(
        n_desc=len(desc), n_sano=len(sano),
        max_diam_b1_descarrilada=max((d[0] for d in desc), default=None),
        min_diam_b1_sana=min((d[0] for d in sano), default=None),
        min_ratio_sana=min((d[0] / d[1] for d in sano if d[1] > 0), default=None),
        max_ratio_descarrilada=max((d[0] / d[1] for d in desc if d[1] > 0), default=None),
    )


def main():
    g, meta = cargar_va()
    cal = calibrar()
    print(f"[Fase V-A] reglas con curva completa: {len(g)}")
    if cal:
        print("\n[calibración desde las 430 re-medidas de verdad]")
        print(f"   descarriladas: {cal['n_desc']}   sanas: {cal['n_sano']}")
        print(f"   diám(b=1) máximo entre DESCARRILADAS : {cal['max_diam_b1_descarrilada']}")
        print(f"   diám(b=1) mínimo entre SANAS          : {cal['min_diam_b1_sana']}")
        print(f"   cociente diám(b1)/diám(b2) máximo entre DESCARRILADAS: {cal['max_ratio_descarrilada']:.3f}")
        print(f"   cociente diám(b1)/diám(b2) mínimo entre SANAS        : {cal['min_ratio_sana']:.3f}")

    n_p1 = n_p2 = 0
    ratios = []
    por_combo = collections.defaultdict(list)
    for k, d in g.items():
        if 1 not in d or 2 not in d:
            continue
        d1, d2 = float(d[1]["diam_real"]), float(d[2]["diam_real"])
        gi = float(d[1]["giant_real"])
        ratios.append((d1 / d2 if d2 else float("nan"), d1, k, meta[k]))
        if d1 < d2:
            n_p1 += 1
        if cal and cal["max_diam_b1_descarrilada"] is not None and d1 <= cal["max_diam_b1_descarrilada"]:
            n_p2 += 1
        por_combo[meta[k]["combo"]].append((gi, d1))

    print(f"\n[PRUEBA 1 — imposibilidad geométrica] reglas con diám(b=1) < diám(b=2): {n_p1}/{len(g)}")
    if cal:
        print(f"[PRUEBA 2 — calibración] reglas con diám(b=1) <= {cal['max_diam_b1_descarrilada']:.0f} "
              f"(el techo de las descarriladas conocidas): {n_p2}/{len(g)}")
    ratios.sort()
    print("\n   los 8 cocientes diám(b1)/diám(b2) más bajos del barrido (los más sospechosos):")
    for r, d1, k, m in ratios[:8]:
        print(f"      {r:.3f}   diám(b1)={d1:.0f}   {k[0]} (pasada {k[1]})  {m['combo']}  clase={m['clase']}")

    print("\n[PRUEBA 3 — fragmentación] fracción de componente gigante a b=1, por combinación:")
    for c, v in sorted(por_combo.items()):
        gs = [x[0] for x in v]
        ds = [x[1] for x in v]
        print(f"   {c:<12} n={len(v):<3} giant min={min(gs):.4f} mediana={statistics.median(gs):.4f}   "
              f"diám(b=1) min={min(ds):.0f} mediana={statistics.median(ds):.0f}")

    print("\n[distribución de clases publicada, recontada desde el CSV]")
    print("  ", dict(collections.Counter(m["clase"] for m in meta.values())))


if __name__ == "__main__":
    main()
