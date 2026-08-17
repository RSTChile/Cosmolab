#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arbitrar_lluvia_con_humedad_suelo.py -- ¿cuál de las dos series de lluvia de
2019 en adelante es la buena? (14-ago-2026)

EL PLEITO. Después de 2018 no hay pluviómetro en el punto-reloj. Hay dos
formas de estimar la lluvia y discrepan sistemáticamente:
  A · estaciones vecinas encadenadas  (la adoptada)
  B · reanálisis ERA5 corregido por sesgo  (la anterior)
En años secos B pone entre 1,4 y 2,2 veces más agua que A. Con un umbral de
germinación de 15 mm/mes, eso decide si el desierto florece.

EL ÁRBITRO. Humedad de suelo satelital (ESA CCI COMBINED v09.2), que no
participó de ninguna de las dos y es continua a través del corte de 2018.

EL MÉTODO, Y POR QUÉ ASÍ
  1. Se agrega a mensual promediando los días con dato bueno (flag=0), y se
     descartan los meses con menos de MIN_DIAS observaciones: un mes con tres
     días medidos no es un promedio mensual.
  2. SE TRABAJA CON ANOMALÍAS, no con valores crudos. Esto es esencial: tanto
     la lluvia como la humedad tienen un ciclo estacional fortísimo --llueve y
     hay humedad en invierno-- y si se corrrelacionan los valores crudos, las
     dos candidatas dan alto SÓLO porque ambas saben en qué mes llueve. Eso no
     distingue nada. Restando la media de cada mes calendario, lo que queda es
     la pregunta real: cuando un invierno fue más húmedo o más seco QUE LO
     NORMAL, ¿cuál de las dos candidatas lo dice bien?
  3. La climatología que se resta se calcula SÓLO con 1979-2018, el tramo con
     pluviómetro medido. Si se calculara con toda la serie, el tramo en disputa
     participaría de su propia evaluación.
  4. Se reporta Spearman (monotonía, robusto a valores extremos) sobre el
     tramo 2019-2024 para cada candidata, y también sobre el tramo de
     calibración como control de que el árbitro sirve para algo.

QUÉ SIGNIFICARÁ EL RESULTADO
  · Si una candidata correlaciona claramente mejor en 2019-2024, decide.
  · Si las dos dan parecido, el árbitro no resuelve: el desacuerdo entre A y B
    está por debajo de lo que un satélite ve en una celda de 25 km. Eso NO es
    un fracaso, es una cota -- y hay que declararla.
  · El control en 1979-2018 es la salvaguarda: si ahí la correlación ya es
    pobre, entonces el árbitro no sirve para este sitio y nada de lo demás vale.
"""
import csv
import os
import statistics as st
from collections import defaultdict

RAIZ = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima"
F = os.path.join(RAIZ, "investigacion/fuentes")
SM = os.path.join(F, "humedad_suelo_esacci_huintil.csv")
MEDIDA = os.path.join(F, "lluvia_mensual_zhcs_1900_2027.csv")          # CR2 Huintil
CAND_A = os.path.join(F, "lluvia_mensual_zhcs_reconstruida_estaciones_cruda.csv")
CAND_B = os.path.join(F, "lluvia_mensual_zhcs_empalmada.csv")           # ERA5 corregido

MIN_DIAS = 8
CAL_INI, CAL_FIN = "1979-01", "2018-12"
TEST_INI, TEST_FIN = "2019-01", "2024-12"


def spearman(xs, ys):
    n = len(xs)
    if n < 6:
        return None
    def rangos(v):
        orden = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[orden[j + 1]] == v[orden[i]]:
                j += 1
            medio = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[orden[k]] = medio
            i = j + 1
        return r
    rx, ry = rangos(xs), rangos(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def sm_mensual():
    dias = defaultdict(list)
    for r in csv.DictReader(open(SM, encoding="utf-8")):
        if r["estado"] == "ok" and r["flag"] == "0" and r["sm_m3m3"]:
            dias[r["fecha"][:7]].append(float(r["sm_m3m3"]))
    return ({m: st.mean(v) for m, v in dias.items() if len(v) >= MIN_DIAS},
            {m: len(v) for m, v in dias.items()})


def lluvia(ruta, filtro_fuente=None):
    s = {}
    for r in csv.DictReader(open(ruta, encoding="utf-8")):
        if not r.get("lluvia_mm"):
            continue
        if filtro_fuente and filtro_fuente not in (r.get("fuente") or ""):
            continue
        s[r["anio_mes"]] = float(r["lluvia_mm"])
    return s


def anomalias(serie, meses_cal):
    """Resta la media de cada mes calendario, calculada SOLO con meses_cal."""
    clim = defaultdict(list)
    for m in meses_cal:
        if m in serie:
            clim[m[5:7]].append(serie[m])
    medias = {k: st.mean(v) for k, v in clim.items() if len(v) >= 5}
    return {m: v - medias[m[5:7]] for m, v in serie.items() if m[5:7] in medias}


def main():
    sm, ndias = sm_mensual()
    med = lluvia(MEDIDA, "CR2 Huintil")
    A = lluvia(CAND_A)
    B = lluvia(CAND_B)

    print(f"Humedad de suelo: {len(sm)} meses utilizables (>= {MIN_DIAS} días con dato bueno)")
    print(f"  de {min(sm)} a {max(sm)}\n")

    cal = sorted(m for m in sm if CAL_INI <= m <= CAL_FIN and m in med)
    test = sorted(m for m in sm if TEST_INI <= m <= TEST_FIN)
    print(f"Calibración (pluviómetro medido en el punto): {len(cal)} meses")
    print(f"Prueba (tramo en disputa):                    {len(test)} meses\n")

    an_sm = anomalias(sm, cal)
    an_med = anomalias(med, cal)
    an_A = anomalias(A, cal) if A else {}
    an_B = anomalias(B, cal) if B else {}

    print("=" * 74)
    print("CONTROL — ¿el árbitro sirve para este sitio?")
    print("  (humedad de suelo vs lluvia MEDIDA, 1979-2018, en anomalías)")
    ms = [m for m in cal if m in an_sm and m in an_med]
    rho = spearman([an_sm[m] for m in ms], [an_med[m] for m in ms])
    print(f"  n={len(ms)} meses · Spearman = {rho:+.3f}")
    if rho is None or rho < 0.25:
        print("\n  ATENCIÓN: la correlación de control es débil. El satélite no está")
        print("  viendo bien la lluvia de este punto, y entonces NO puede arbitrar.")
    else:
        print("  El árbitro ve la lluvia real. Se puede seguir.")

    print("\n" + "=" * 74)
    print("ARBITRAJE — 2019-2024, cada candidata contra la humedad observada")
    print(f"{'candidata':<40}{'n':>5}{'Spearman':>12}")
    res = {}
    for nombre, an in (("A · estaciones encadenadas (adoptada)", an_A),
                       ("B · ERA5 corregido (anterior)", an_B)):
        ms = [m for m in test if m in an_sm and m in an]
        r = spearman([an_sm[m] for m in ms], [an[m] for m in ms])
        res[nombre] = (r, len(ms))
        print(f"{nombre:<40}{len(ms):>5}{(f'{r:+.3f}' if r is not None else 'n/d'):>12}")

    ra, rb = res["A · estaciones encadenadas (adoptada)"][0], res["B · ERA5 corregido (anterior)"][0]
    print()
    if ra is None or rb is None:
        print("No hay meses suficientes para decidir.")
    elif abs(ra - rb) < 0.10:
        print(f"NO DECIDE: las dos quedan a {abs(ra-rb):.3f} de distancia.")
        print("El desacuerdo entre A y B es más fino que lo que el satélite")
        print("resuelve en una celda de 25 km. Queda como cota, no como veredicto.")
    elif ra > rb:
        print(f"DECIDE A FAVOR DE LAS ESTACIONES (A) por {ra-rb:+.3f}.")
        print("Respalda la serie que ya está publicada.")
    else:
        print(f"DECIDE A FAVOR DEL REANÁLISIS (B) por {rb-ra:+.3f}.")
        print("CONTRADICE la serie publicada: hay que revisar la decisión de ayer.")


if __name__ == "__main__":
    main()
