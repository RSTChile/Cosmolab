"""
EL UMBRAL EN MILÍMETROS · «¿con cuánta lluvia se activa este punto?»
=====================================================================

INSTRUCCIÓN (Alexis, 23-ago-2026): «en caso de lluvia debería mostrar qué calles
específicas entre qué manzanas específicas se inundarán si caen tantos mm en
tanto tiempo… eso sí le sirve [al alcalde].»

★ EL DATO QUE EL ESTADO DISEÑÓ Y NO LLENÓ
-------------------------------------------
El Ministerio de Obras Públicas **previó exactamente esta pregunta**: su capa de
puntos de inundación tiene una columna `PRECI` para la precipitación asociada.
Está en **−99 —vacío— en los 429 registros, sin una sola excepción**.

Ningún organismo chileno publica el umbral. Pero se puede derivar, y la pieza que
lo permite es el catastro de CIGIDEN: **826 eventos con fecha, lugar y la lluvia
que los provocó, con su duración**:

    "167,7mm/120hrs"    "36,1mm/24hrs"    "4,4mm/15minutos"

★ CÓMO SE DERIVA, Y QUÉ SE PUEDE Y NO SE PUEDE CONCLUIR
---------------------------------------------------------
Se normaliza cada evento a **milímetros por hora sostenidos** y a su acumulado, y
se agrupa por el proceso que desencadenó. Eso da, por tipo de falla, la
distribución de lluvia que la produjo.

⚠️ **Lo que esto NO es.** No es la probabilidad de que un punto se inunde: es la
lluvia observada EN LOS CASOS EN QUE SE INUNDÓ. Falta el denominador — cuántas
veces llovió eso mismo y no pasó nada. Por eso además se compara contra la serie
climática del proyecto: para cada evento, en qué percentil de su propia celda
cae esa lluvia. Un evento que ocurre en el percentil 99 dice algo muy distinto de
uno que ocurre en el percentil 60.

⚠️ **Y es la cuenca del Maipo.** Los 826 eventos son de la Región Metropolitana y
alrededores. Sirve para Santiago —que es justo lo que se pidió— y **no se puede
extrapolar al norte**, donde la misma lluvia significa otra cosa.

USO
---
    ../.venv-esa/bin/python derivar_umbral.py
"""

import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI / "datos" / "crudo" / "puntos_criticos"
SERIE = None  # se resuelve en main() — ver elegir_serie()

# ★★ QUÉ SERIE CLIMÁTICA SE USA — declarado, nunca implícito.
#   Hay DOS y no son intercambiables: ERA5-Land mide ~19 % más que Open-Meteo
#   (Coquimbo, julio 2026: 408,7 mm contra 343,6) y su día empieza a otra hora
#   (acumula en UTC; la vieja se pidió en hora de Santiago). Cambiar de fuente
#   cambia TODOS los umbrales, así que el script dice en voz alta cuál usó.
SERIE_OM = AQUI / "datos" / "clima_diario_celdas.csv"
SERIE_E5 = AQUI / "datos" / "clima_diario_celdas_era5land.csv"


def elegir_serie(argv):
    """ERA5-Land por defecto si ya está convertida; --openmeteo fuerza la vieja."""
    if "--openmeteo" in argv or not SERIE_E5.exists():
        return SERIE_OM, "Open-Meteo (malla ~0,07°, día en hora de Santiago)"
    return SERIE_E5, "ERA5-Land 0,1° de Copernicus (día en UTC)"

SALIDA = AQUI / "datos" / "umbral_por_proceso.csv"

MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
         "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
         "noviembre": 11, "diciembre": 12}
MALLA = 0.10


def norm(s):
    return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def parsear_pp(txt):
    """«167,7mm/120hrs» → (167.7, 120.0). Devuelve None si no se puede leer.

    ★ Se aceptan minutos y días además de horas, porque el catastro los mezcla:
    hay eventos anotados como «4,4mm/15minutos» y otros como «180,2mm/72hrs».
    Tratarlos todos como horas daría intensidades absurdas.
    """
    if not txt or str(txt).strip() in ("-", "None", "s/i", "S/I"):
        return None
    t = norm(txt).replace(" ", "")
    # ★ El catastro escribe la MISMA cosa de dos maneras: «4,4mm/24hrs» y
    #   «11.7/24hrs». La segunda forma son 292 de 826 eventos — casi la mitad de
    #   la muestra útil. La unidad «mm» es opcional en el patrón por eso.
    m = re.search(r"([\d]+(?:[.,]\d+)?)\s*(?:mm)?\s*/\s*([\d]+(?:[.,]\d+)?)\s*(min|hr|hor|dia|d)", t)
    if not m:
        m2 = re.search(r"([\d]+(?:[.,]\d+)?)\s*mm", t)
        return (float(m2.group(1).replace(",", ".")), None) if m2 else None
    mm = float(m.group(1).replace(",", "."))
    n = float(m.group(2).replace(",", "."))
    u = m.group(3)
    horas = n / 60 if u == "min" else (n * 24 if u.startswith("d") else n)
    return mm, horas


def fecha_de(p):
    try:
        a = int(p.get("Año"))
        mes = MESES.get(norm(p.get("Mes")))
        d = int(float(p.get("Dia_Inicio")))
        return date(a, mes, d) if (a > 1900 and mes and 1 <= d <= 31) else None
    except (TypeError, ValueError):
        return None


def celda(lat, lon):
    return f"{round(lat/MALLA)}_{round(lon/MALLA)}"


def percentil_en(v, ordenada):
    lo, hi = 0, len(ordenada)
    while lo < hi:
        m = (lo + hi) // 2
        if ordenada[m] < v:
            lo = m + 1
        else:
            hi = m
    return lo / max(len(ordenada) - 1, 1)


def main():
    global SERIE
    SERIE, nombre = elegir_serie(sys.argv)
    print(f"  serie climática: {nombre}")
    ult = sorted(CRUDO.glob("*"))[-1] if CRUDO.exists() else None
    if not ult or not (ult / "cigiden_crm.json").exists():
        print("falta el espejado de CIGIDEN — corre adaptadores/puntos_criticos.py")
        return 1
    ent = json.loads((ult / "cigiden_crm.json").read_text(encoding="utf-8"))["features"]

    print("=" * 76)
    print("EL UMBRAL EN MILÍMETROS · derivado de 826 eventos con lluvia y fecha")
    print("=" * 76)

    ev = []
    for e in ent:
        p = e["properties"]
        pp = parsear_pp(p.get("PP_mm"))
        f = fecha_de(p)
        g = (e.get("geometry") or {}).get("coordinates")
        if not pp or not pp[1]:
            continue
        ev.append({"mm": pp[0], "horas": pp[1], "fecha": f,
                   "proceso": str(p.get("Proceso_P") or "").strip(),
                   "lon": g[0] if g else None, "lat": g[1] if g else None,
                   "ubic": str(p.get("Ubicacion") or "")[:44]})
    print(f"\n  eventos con lluvia legible y duración : {len(ev)} de {len(ent)}")
    print(f"  con fecha completa                    : {sum(1 for x in ev if x['fecha'])}")
    print(f"  con coordenada                        : {sum(1 for x in ev if x['lat'])}")

    # ── el umbral por tipo de proceso ────────────────────────────────────────
    print("\n" + "=" * 76)
    print("LLUVIA QUE ACTIVÓ CADA TIPO DE FALLA")
    print("=" * 76 + "\n")
    print(f"  {'proceso':<30}{'filas':>6}{'tormentas':>10}{'lugares':>8}"
          f"{'mm med':>8}{'horas':>7}{'mm/h':>7}")
    print("  " + "-" * 76)
    filas = []
    por = defaultdict(list)
    for x in ev:
        por[x["proceso"]].append(x)
    for proc, v in sorted(por.items(), key=lambda t: -len(t[1])):
        if len(v) < 5:
            continue
        mms = sorted(x["mm"] for x in v)
        hrs = sorted(x["horas"] for x in v)
        ints = sorted(x["mm"] / x["horas"] for x in v)
        med = lambda s: s[len(s) // 2]
        p25 = lambda s: s[len(s) // 4]
        # ★ Una tormenta deja MUCHAS filas: el mismo temporal se anota una vez por
        #   cada lugar que falló. Las filas son LUGARES, no observaciones
        #   independientes — «Flujo» son 290 lugares pero sólo 38 tormentas.
        torm = len({(x["fecha"], x["mm"], x["horas"]) for x in v})
        lug = len({(round(x["lon"], 4), round(x["lat"], 4)) for x in v if x["lat"]})
        print(f"  {proc[:30]:<30}{len(v):>6}{torm:>10}{lug:>8}"
              f"{med(mms):>8.1f}{med(hrs):>7.0f}{med(ints):>7.2f}")
        filas.append({"proceso": proc, "n": len(v),
                      "tormentas_distintas": torm, "lugares_distintos": lug,
                      "mm_p25": round(p25(mms), 1), "mm_mediano": round(med(mms), 1),
                      "horas_mediana": round(med(hrs), 1),
                      "mm_por_hora_p25": round(p25(ints), 3),
                      "mm_por_hora_mediano": round(med(ints), 3),
                      "mm_minimo_observado": round(mms[0], 1)})

    print("\n  ★ El mínimo observado es el umbral operativo: por debajo de esa")
    print("    lluvia, este catastro no registra ni un solo caso de ese proceso.\n")
    for f in filas:
        print(f"     {f['proceso'][:30]:<30} mínimo {f['mm_minimo_observado']:>6.1f} mm"
              f"   ·   cuartil bajo {f['mm_p25']:>6.1f} mm")

    # ── contra la serie climática del proyecto ───────────────────────────────
    con_todo = [x for x in ev if x["fecha"] and x["lat"] and x["fecha"].year >= 1990]
    print("\n" + "=" * 76)
    print(f"CONTRASTE CON LA SERIE DEL PROYECTO · {len(con_todo)} eventos desde 1990")
    print("=" * 76)
    if con_todo and SERIE.exists():
        quiero = {celda(x["lat"], x["lon"]) for x in con_todo}
        serie = defaultdict(dict)
        with SERIE.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["celda"] in quiero and r["precip_mm"] not in ("", "None"):
                    serie[r["celda"]][r["fecha"]] = float(r["precip_mm"])
        print(f"\n  celdas necesarias : {len(quiero)} · con serie: {len(serie)}")
        _t = [v for c in serie.values() for v in c.values()]
        frac_secos = (sum(1 for v in _t if v < 0.1) / len(_t)) if _t else 0.0
        pares, sin = [], 0
        for x in con_todo:
            c = celda(x["lat"], x["lon"])
            if c not in serie:
                sin += 1
                continue
            f = x["fecha"].isoformat()
            if f not in serie[c]:
                sin += 1
                continue
            dist = sorted(serie[c].values())
            # ★ CONTROL OBLIGATORIO: la mayoría de los días de la serie son
            #   secos, así que cualquier lluvia > 0 ya supera a esa mayoría por
            #   pura construcción. El percentil honesto es contra los días EN QUE
            #   SÍ LLOVIÓ.
            #   ⚠️ La fracción de días secos SE MIDE, no se escribe a mano:
            #   depende de la fuente. Open-Meteo daba 70,0 % y ERA5-Land da
            #   58,3 % sobre las mismas celdas — un número fijo aquí quedaría
            #   mintiendo en cuanto se cambie de serie.
            mojados = sorted(v for v in dist if v >= 0.1)
            v = serie[c][f]
            pares.append((x, v, percentil_en(v, dist),
                          percentil_en(v, mojados) if mojados else None))
        print(f"  eventos cruzados  : {len(pares)} · sin cobertura: {sin}")
        if pares:
            pcts = sorted(p for _, _, p, _ in pares)
            moja = sorted(m for _, _, _, m in pares if m is not None)
            print(f"\n  ★ En qué percentil de su propia celda cae la lluvia del día"
                  f" del evento:\n")
            print(f"       {'':<8}{'vs TODOS':>12}{'vs días CON lluvia':>22}")
            for q, et in ((0.10, "P10"), (0.25, "P25"), (0.50, "mediana"),
                          (0.75, "P75"), (0.90, "P90")):
                a = 100 * pcts[int(q * (len(pcts) - 1))]
                b = 100 * moja[int(q * (len(moja) - 1))] if moja else float("nan")
                print(f"       {et:<8}{a:11.1f} %{b:21.1f} %")
            altos = sum(1 for p in pcts if p >= 0.95)
            alt2 = sum(1 for m in moja if m >= 0.95)
            print(f"\n     vs TODOS los días  : {altos} de {len(pcts)} "
                  f"({100*altos/len(pcts):.0f} %) en el 5 % más lluvioso.")
            print(f"     vs días CON lluvia : {alt2} de {len(moja)} "
                  f"({100*alt2/len(moja):.0f} %) en el 5 % más lluvioso.")
            print(f"\n     ★ La segunda cifra es la que vale: el {100*frac_secos:.1f} %"
                  f" de los días de")
            print("       la serie son secos, así que la primera está inflada de fábrica.")
            print("     ⚠️ Los que NO, son los que el instrumento climático solo")
            print("        NUNCA habría anticipado: fallan con lluvia corriente.")
            tn = len({(x["fecha"], x["mm"]) for x, _, _, _ in pares})
            print(f"\n     ⚠️ n EFECTIVO: los {len(pares)} eventos cruzados vienen de")
            print(f"        sólo {tn} tormentas. Están correlacionados entre sí:")
            print(f"        NO son {len(pares)} pruebas independientes.")

    if filas:
        with SALIDA.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
        print(f"\n  escrito: {SALIDA.name}")
    print("\n  ⚠️ Cuenca del Maipo. Vale para Santiago; NO se extrapola al norte.")
    print("  ⚠️ Es la lluvia de los casos en que SÍ falló. Falta el denominador.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
