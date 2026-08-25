"""
DE DÓNDE SACA CADA ACTIVO SU CLIMA
====================================

INSTRUCCIÓN (Alexis, 22-ago-2026): «Usar las estaciones directamente… el sistema
debe ser lo más granular que los datos nos permitan.»

EL PRINCIPIO, Y POR QUÉ NO ES OBVIO
-------------------------------------
El proyecto tiene dos fuentes de precipitación y **no son equivalentes**:

**Pluviómetro** — medición real. 55 estaciones de la Región Metropolitana
(Solange Varela), diarias, 2019-2026. Es el dato.

**Reanálisis ERA5** — un modelo. Cobertura nacional continua desde 1990. Es una
estimación, y está medido lo que le pasa en la zona central: **llueve un 85 % de
más, hasta +237 % en los años secos, y se equivoca de mes**.

Así que donde hay pluviómetro, manda el pluviómetro. No es preferencia: es que
uno mide y el otro estima, y sabemos cuánto se desvía el que estima.

★ LA REGLA, DECLARADA
-----------------------
    Si hay una estación a **10 km o menos**, el activo usa la ESTACIÓN.
    Si no, usa la celda del reanálisis que le corresponde.

**Por qué 10 km y no otro número:** la malla del reanálisis mide unos 9 km, así
que una estación a 10 km está a una distancia comparable a la del propio centro
de celda — pero midiendo en vez de modelando. Más lejos, la lluvia decorrelaciona
lo suficiente como para que un modelo continuo sea mejor que una medición ajena.

★ LO QUE NO SE HACE, Y ES IMPORTANTE
--------------------------------------
**No se rellenan los huecos de la estación con reanálisis.** Mezclar dos fuentes
dentro de una misma serie fabrica saltos artificiales justo en los días sin
observador — y esos días suelen ser los de temporal, cuando el observador no
pudo salir. Un día sin medición queda como hueco declarado.

**Y no se corrige el reanálisis con el factor 0,6022** que se calibró el 22-ago.
Está medido que ese factor **baja el error un 34 % pero no cambia el orden** —
ninguna corrección monótona puede—, y se calibró con estaciones de la Región
Metropolitana, o sea justo donde ya no hace falta. Aplicarlo al resto del país
sería extrapolar sin validar.

USO
---
    ../.venv-esa/bin/python fuente_climatica.py
"""

import csv
import math
import sys
from collections import Counter
from pathlib import Path

import openpyxl

AQUI = Path(__file__).resolve().parent
SUB = AQUI / "submatrices_excel"
DATOS = AQUI / "datos"
ESTACIONES = AQUI.parent / "datos" / "estaciones_rm_puntos.csv"
SALIDA = DATOS / "fuente_climatica_por_activo.csv"

UMBRAL_KM = 10.0
MALLA = 0.10


def km(la1, lo1, la2, lo2):
    return 111.32 * math.hypot(la1 - la2,
                               (lo1 - lo2) * math.cos(math.radians(la1)))


def celda(lat, lon):
    return f"{round(lat/MALLA)}_{round(lon/MALLA)}"


def main():
    est = [(float(e["lat"]), float(e["lon"]), e["nombre"])
           for e in csv.DictReader(ESTACIONES.open(encoding="utf-8")) if e["lat"]]
    print("=" * 80)
    print("FUENTE CLIMÁTICA DE CADA ACTIVO · pluviómetro donde lo haya")
    print("=" * 80)
    print(f"\n  estaciones disponibles : {len(est)} (Región Metropolitana)")
    print(f"  regla declarada        : estación si está a ≤ {UMBRAL_KM:.0f} km; "
          f"si no, celda del reanálisis")

    filas, cuenta, dist = [], Counter(), []
    for p in sorted(SUB.glob("*.xlsx")):
        ws = openpyxl.load_workbook(p, read_only=True).active
        f = list(ws.iter_rows(values_only=True))
        cab = {c: i for i, c in enumerate(f[0])}
        if "Latitud decimal" not in cab:
            continue
        for r in f[1:]:
            la, lo = r[cab["Latitud decimal"]], r[cab["Longitud decimal"]]
            if la in (None, ""):
                continue
            la, lo = float(la), float(lo)
            d, nom = min(((km(la, lo, a, b), n) for a, b, n in est))
            usa_est = d <= UMBRAL_KM
            cuenta["estación" if usa_est else "reanálisis"] += 1
            if usa_est:
                dist.append(d)
            filas.append(dict(submatriz=p.stem, item=r[cab["Ítem"]],
                              activo=str(r[0])[:120],
                              lat=la, lon=lo,
                              fuente="estacion" if usa_est else "reanalisis",
                              referencia=nom if usa_est else celda(la, lo),
                              km_a_la_estacion=round(d, 2)))

    tot = sum(cuenta.values())
    print(f"\n  activos con coordenada : {tot:,}")
    for k in ("estación", "reanálisis"):
        print(f"     {k:12s} {cuenta[k]:7,d}  ({100*cuenta[k]/tot:5.1f} %)")
    if dist:
        dist.sort()
        print(f"\n  de los que usan estación: distancia mediana "
              f"{dist[len(dist)//2]:.1f} km · máxima {dist[-1]:.1f} km")

    # qué sub-matrices quedan mejor servidas
    por_sub = Counter()
    tot_sub = Counter()
    for x in filas:
        tot_sub[x["submatriz"]] += 1
        if x["fuente"] == "estacion":
            por_sub[x["submatriz"]] += 1
    print("\n  sub-matrices con más cobertura de medición real:")
    for s, n in sorted(por_sub.items(), key=lambda t: -t[1] / tot_sub[t[0]])[:8]:
        print(f"      {100*n/tot_sub[s]:5.1f} %  {n:6,d}/{tot_sub[s]:<6,d}  {s}")

    with SALIDA.open("w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\n  escrito: {SALIDA.name} ({len(filas):,} filas)")

    print("\n" + "=" * 80)
    print("LO QUE ESTO CAMBIA Y LO QUE NO")
    print("=" * 80)
    print(f"""
  CAMBIA · {cuenta['estación']:,} activos pasan de estimación a MEDICIÓN. En esos,
    desaparece el sesgo de +85 % del reanálisis y el error de identificar mal el
    mes peor.

  NO CAMBIA · los otros {cuenta['reanálisis']:,} siguen con reanálisis sin corregir, y
    arrastran su sesgo. No se les aplica el factor 0,6022 porque se calibró en la
    Región Metropolitana —justo donde ya no hace falta— y extrapolarlo sería
    inventar.

  LO QUE FALTA · estaciones fuera de la Región Metropolitana. Con las de Atacama
    y Coquimbo que el proyecto ya tiene en `datos/Excel Estaciones/` se puede
    ampliar la cobertura al norte, que es donde viven los aluviones.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
