#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparar_metodos_lluvia_contra_huintil.py -- cara a cara entre los dos modos
de estimar la lluvia del punto-reloj cuando no hay estación (13-ago-2026).

POR QUÉ ESTE SCRIPT EXISTE. Al validar la reconstrucción desde estaciones
reales apareció que su error propio (LOOCV: 5-9 mm medianos, 9-15 mm medios
por mes de lluvia) es del MISMO TAMAÑO que el umbral de germinación del
modelo (15 mm/mes). Un método cuyo error iguala el umbral que debe decidir no
se puede adoptar "porque es medición" ni descartar "porque falló un año":
hay que medir cuál de los dos candidatos se equivoca menos, contra el único
juez disponible -- Huintil MEDIDO, 1966-2018.

LOS DOS CANDIDATOS
  A · ERA5 corregido      reanálisis × factor por mes calendario (lo publicado)
  B · Estación encadenada estación vecina × razón por mes calendario (lo nuevo)

REGLA DE JUEGO IDÉNTICA PARA AMBOS: validación dejando-un-año-fuera. El año
que se predice NO participa de la calibración de los factores/razones. Sin
esto, cualquiera de los dos parecería excelente por haberse ajustado a sí
mismo -- que es exactamente el error de plano que venimos corrigiendo.

LA MÉTRICA QUE MANDA NO ES EL ERROR EN MM. Al modelo no le importa acertar
milímetros: le importa si el mes cruzó o no los 15 mm que disparan la
germinación. Por eso además del error absoluto se mide el ACIERDO DE UMBRAL,
y se separan los dos errores que no son simétricos para este experimento:
  · floración FABRICADA  (el método dice >15 mm y la realidad fue <=15)
  · floración PERDIDA    (el método dice <=15 mm y la realidad fue >15)
La fabricada es la peligrosa: inventa un Desierto Florido que no existió.

AVISO SOBRE EL CANDIDATO B. Acá se lo evalúa con las estaciones HISTÓRICAS
(las únicas que solapan con Huintil). El método real en 2019+ agrega encima
un supuesto que no se puede validar en ningún lado: que la estación DMC nueva
mide lo mismo que la histórica co-localizada. O sea, los números de B en este
informe son su PISO de error, no su error real.
"""
import csv
import math
import os
import sqlite3
import statistics as st

RAIZ = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima"
DB = os.path.join(RAIZ, "investigacion/fuentes/pluviosidad_diaria_consolidada.sqlite")
CSV_FUENTES = os.path.join(RAIZ, "investigacion/fuentes/lluvia_mensual_zhcs_1900_2027.csv")
CSV_ERA5 = os.path.join(RAIZ, "investigacion/fuentes/lluvia_mensual_zhcs_openmeteo.csv")
INFORME = os.path.join(RAIZ, "investigacion/fuentes/lluvia_comparacion_metodos.txt")

HL, HO = -31.5669, -70.9817
UMBRAL_GERMINACION = 15.0
RADIO_KM, MIN_MESES, DIAS_MIN_MES = 60.0, 120, 25

BANDA = []
def log(s=""):
    print(s); BANDA.append(s)


def factores(objetivo, fuente, excluir=None):
    """Cociente de sumas por mes calendario -- robusto donde muchos meses son 0."""
    num = {m: 0.0 for m in range(1, 13)}; den = dict(num); n = {m: 0 for m in range(1, 13)}
    for ym, v in fuente.items():
        if ym not in objetivo or (excluir and ym[:4] == excluir):
            continue
        m = int(ym[5:7]); num[m] += objetivo[ym]; den[m] += v; n[m] += 1
    return {m: num[m] / den[m] for m in range(1, 13) if n[m] >= 8 and den[m] >= 20.0}


def loocv(objetivo, fuente, etiqueta):
    anios = sorted({ym[:4] for ym in fuente if ym in objetivo})
    aes, ok, fabricadas, perdidas, n_hum = [], 0, 0, 0, 0
    total = 0
    for a in anios:
        f = factores(objetivo, fuente, excluir=a)
        if not f:
            continue
        for ym, v in fuente.items():
            if ym[:4] != a or ym not in objetivo:
                continue
            m = int(ym[5:7])
            if m not in f:
                continue
            pred, real = v * f[m], objetivo[ym]
            total += 1
            if real + pred >= 5.0:
                aes.append(abs(pred - real))
            pu, ru = pred > UMBRAL_GERMINACION, real > UMBRAL_GERMINACION
            if pu == ru: ok += 1
            elif pu and not ru: fabricadas += 1
            else: perdidas += 1
            if ru: n_hum += 1
    return {
        "etiqueta": etiqueta, "meses": total,
        "mae": (sum(aes) / len(aes)) if aes else float("nan"),
        "mediana": st.median(aes) if aes else float("nan"),
        "acierto": 100.0 * ok / total if total else 0.0,
        "fabricadas": fabricadas, "perdidas": perdidas,
        "n_humedos": n_hum,
    }


def main():
    huintil = {r["anio_mes"]: float(r["lluvia_mm"])
               for r in csv.DictReader(open(CSV_FUENTES, encoding="utf-8"))
               if "CR2 Huintil" in (r["fuente"] or "") and r["lluvia_mm"] not in ("", None)}
    huintil = {k: v for k, v in huintil.items() if k >= "1966-01"}
    era5 = {r["anio_mes"]: float(r["lluvia_mm"])
            for r in csv.DictReader(open(CSV_ERA5, encoding="utf-8"))}

    con = sqlite3.connect(DB)
    filas = con.execute("""SELECT localidad, lat, lon, substr(fecha,1,7) mes,
                           SUM(lluvia_mm), COUNT(*) FROM pluviosidad_diaria
                           WHERE tipo_fuente='estacion_real' GROUP BY localidad,lat,lon,mes""").fetchall()
    est = {}
    for loc, la, lo, mes, mm, n in filas:
        if loc == "Huintil" or n < DIAS_MIN_MES:
            continue
        d = math.hypot((la - HL) * 111.0, (lo - HO) * 111.0 * math.cos(math.radians(HL)))
        if d > RADIO_KM:
            continue
        est.setdefault(loc, {"km": d, "meses": {}})["meses"][mes] = mm
    hist = {k: v for k, v in est.items() if len(set(v["meses"]) & set(huintil)) >= MIN_MESES}

    log(f"Juez: Huintil MEDIDO, {len(huintil)} meses ({min(huintil)} a {max(huintil)})")
    log(f"Meses húmedos (>{UMBRAL_GERMINACION:.0f} mm) en el juez: "
        f"{sum(1 for v in huintil.values() if v > UMBRAL_GERMINACION)}")
    log(f"Estaciones históricas candidatas: {len(hist)}\n")

    res = [loocv(huintil, era5, "A · ERA5 corregido")]
    for loc, v in sorted(hist.items(), key=lambda x: x[1]["km"])[:8]:
        res.append(loocv(huintil, v["meses"], f"B · {loc[:26]} ({v['km']:.0f} km)"))

    log("=" * 92)
    log("VALIDACIÓN DEJANDO-UN-AÑO-FUERA — todos contra Huintil medido, misma regla")
    log(f"{'método':<38}{'meses':>7}{'err.med':>9}{'MAE':>8}{'acierto':>9}{'FABRICA':>9}{'pierde':>8}")
    for r in res:
        log(f"{r['etiqueta']:<38}{r['meses']:>7}{r['mediana']:>9.1f}{r['mae']:>8.1f}"
            f"{r['acierto']:>8.1f}%{r['fabricadas']:>9}{r['perdidas']:>8}")

    log("\n'FABRICA' = meses en que el método dice >15 mm y Huintil midió <=15 mm:")
    log("floraciones inventadas. 'pierde' = el revés. Sobre "
        f"{sum(1 for v in huintil.values() if v > UMBRAL_GERMINACION)} meses húmedos reales.")

    mejor_b = min((r for r in res if r["etiqueta"].startswith("B")), key=lambda r: r["mae"])
    a = res[0]
    log("\n" + "=" * 92)
    log(f"A (ERA5 corregido):        MAE {a['mae']:.1f} mm · acierto de umbral {a['acierto']:.1f}% · "
        f"fabrica {a['fabricadas']} · pierde {a['perdidas']}")
    log(f"B (mejor estación única):  MAE {mejor_b['mae']:.1f} mm · acierto de umbral {mejor_b['acierto']:.1f}% · "
        f"fabrica {mejor_b['fabricadas']} · pierde {mejor_b['perdidas']}")
    log(f"   -> {mejor_b['etiqueta']}")
    log("\nRecordatorio: B acá usa estaciones históricas. El método real en 2019+ suma")
    log("encima el supuesto NO validable de que la estación DMC nueva mide lo mismo que")
    log("la histórica co-localizada -- y en el único año donde eso se pudo comprobar")
    log("(2018, La Canela a 0.2 km) la cadena completa se fue +105%.")

    with open(INFORME, "w", encoding="utf-8") as f:
        f.write("\n".join(BANDA) + "\n")
    print(f"\nInforme: {INFORME}")


if __name__ == "__main__":
    main()
