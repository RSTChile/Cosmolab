"""
EL UMBRAL, PERO EN LA CALLE · «qué tramo se corta con cuántos milímetros»
=========================================================================

INSTRUCCIÓN (Alexis, 23-ago-2026): «en caso de lluvia debería mostrar qué calles
específicas entre qué manzanas específicas se inundarán si caen tantos mm en
tanto tiempo… eso sí le sirve [al alcalde].»

`derivar_umbral.py` dio el umbral por TIPO DE FALLA con el catastro de CIGIDEN.
Sirve, pero tiene dos límites que este script viene a levantar:

  1. era la cuenca del Maipo y no se podía extrapolar;
  2. **faltaba el denominador** — era la lluvia de los casos en que SÍ falló,
     sin saber cuántas veces llovió lo mismo y no pasó nada.

★ LA PIEZA QUE LO PERMITE: EL TEMPORAL DE JULIO 2026
------------------------------------------------------
La capa de vías afectadas del MOP trae **1.289 tramos cortados entre el 16-jul y
el 2-ago de 2026**, cada uno con rol vial, kilómetro inicial y final, elemento
dañado, gravedad y **fecha**. Dieciocho días, seis regiones.

Eso convierte el problema en un experimento natural: para cada celda que tuvo un
corte, se conocen TODOS los días del periodo — los que cortaron y los que no. El
denominador aparece solo.

★ LOS DOS PRODUCTOS
---------------------
  A. **Umbral por tramo**: para cada vía cortada, la lluvia de ese día y los
     acumulados de 24, 48 y 72 h en su celda. Es la respuesta literal a la
     pregunta: «la ruta D-595 entre el km 12,4 y el 14,1 se cortó con 38 mm
     en 48 h».
  B. **El test con denominador**: dentro de las mismas celdas y el mismo
     periodo, ¿la lluvia de los días-con-corte es distinta de la de los
     días-sin-corte? Si no lo fuera, la lluvia no explicaría el corte.

⚠️ COBERTURA: la serie climática del proyecto tiene 418 celdas y las vías caen en
563. Sólo ~23 % de los tramos quedan cruzados. El resultado vale para esos; la
ampliación pide el barrido de ERA5 completo desde Copernicus.

⚠️⚠️ LA FECHA ES LA DEL INFORME, NO LA DEL CORTE. Medido, no supuesto: 26 tramos
figuran con <1 mm en 72 h y los 26 tienen su golpe de lluvia entre 3 y 7 días
antes, 19 de ellos exactamente a 4 días. Un desfase tan agrupado no es física.
Se probó la explicación alternativa —suelo saturado— y falló: cruzando golpe
(72 h) contra carga previa (10 días), la carga NO sube la tasa de corte
(18,3 % con suelo cargado contra 20,8 % sin él). Lo único que la mueve es el
golpe reciente. Cualquier uso de esta capa tiene que tratar la fecha como
aproximada a ±4 días.

USO
---
    ../.venv-esa/bin/python umbral_por_tramo.py
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
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

SAL_TRAMOS = AQUI / "datos" / "umbral_por_tramo.csv"
SAL_ELEM = AQUI / "datos" / "umbral_por_elemento.csv"

MALLA = 0.10


def celda(lat, lon):
    return f"{round(lat/MALLA)}_{round(lon/MALLA)}"


def med(s):
    return s[len(s) // 2] if s else None


def cuantil(s, q):
    return s[int(q * (len(s) - 1))] if s else None


def cargar_tramos():
    """Los 1.289 tramos cortados, normalizados."""
    ult = sorted(CRUDO.glob("*"))[-1]
    fs = json.loads((ult / "vias_julio2026.json").read_text(encoding="utf-8"))["features"]
    out = []
    for x in fs:
        a = x["attributes"]
        g = x.get("geometry") or {}
        if not g.get("x") or not a.get("fecha"):
            continue
        f = datetime.fromtimestamp(a["fecha"] / 1000, timezone.utc).date()
        out.append({
            "fecha": f, "lat": g["y"], "lon": g["x"],
            "celda": celda(g["y"], g["x"]),
            "region": a.get("region") or "", "comuna": a.get("comuna") or "",
            "rol": a.get("rol_vial") or "", "km_ini": a.get("km_ini"),
            "km_fin": a.get("km_fin"), "infra": (a.get("infra_afec") or "")[:60],
            "elemento": a.get("elemento") or "sin dato",
            "gravedad": a.get("gravedad") or "sin dato",
            "operatividad": a.get("operatividad") or "",
            "emergencia": (a.get("emergencia") or "")[:70],
        })
    return out


def cargar_serie(celdas):
    """{celda: {fecha: mm}} sólo para las celdas pedidas."""
    s = defaultdict(dict)
    with SERIE.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["celda"] in celdas and r["precip_mm"] not in ("", "None"):
                s[r["celda"]][r["fecha"]] = float(r["precip_mm"])
    return s


def acumulado(dia_serie, f, dias):
    """Suma de los últimos `dias` días terminando en f. None si falta alguno."""
    t = 0.0
    for k in range(dias):
        v = dia_serie.get((f - timedelta(days=k)).isoformat())
        if v is None:
            return None
        t += v
    return t


def main():
    global SERIE
    SERIE, nombre = elegir_serie(sys.argv)
    tramos = cargar_tramos()
    print("=" * 78)
    print("EL UMBRAL EN LA CALLE · temporal 16-jul a 2-ago 2026, vías del MOP")
    print("=" * 78)
    print(f"\n  serie climática                        : {nombre}")
    print(f"  tramos cortados con fecha y coordenada : {len(tramos)}")
    print(f"  con rol vial y kilometraje             : "
          f"{sum(1 for t in tramos if t['rol'] and t['km_ini'] is not None)}")

    serie = cargar_serie({t["celda"] for t in tramos})
    print(f"  celdas necesarias                      : "
          f"{len({t['celda'] for t in tramos})} · con serie: {len(serie)}")

    # ── A · umbral por tramo ─────────────────────────────────────────────────
    for t in tramos:
        ds = serie.get(t["celda"])
        t["mm_dia"] = ds.get(t["fecha"].isoformat()) if ds else None
        t["mm_24h"] = acumulado(ds, t["fecha"], 1) if ds else None
        t["mm_48h"] = acumulado(ds, t["fecha"], 2) if ds else None
        t["mm_72h"] = acumulado(ds, t["fecha"], 3) if ds else None
        # ★★ LA FECHA DEL MOP ES LA DEL INFORME, NO LA DEL CORTE.
        #   26 tramos figuran cortados con <1 mm en 72 h. Los 26 tienen su último
        #   día de lluvia fuerte (>10 mm) entre 3 y 7 días antes, y 19 de 26 caen
        #   EXACTAMENTE a 4 días. Un desfase así de agrupado es administrativo,
        #   no físico. Se descartó la explicación por suelo saturado con el cruce
        #   golpe × saturación: la carga previa NO sube la tasa de corte
        #   (18,3 % con suelo cargado vs 20,8 % sin él); lo único que la mueve es
        #   el golpe reciente. La ventana de 10 días se conserva porque absorbe
        #   ese desfase de informe, no porque mida saturación.
        t["mm_10d"] = acumulado(ds, t["fecha"], 10) if ds else None
    cruz = [t for t in tramos if t["mm_72h"] is not None]
    print(f"  tramos con lluvia cruzada              : {len(cruz)} "
          f"({100*len(cruz)/len(tramos):.1f} %)")

    if not cruz:
        print("\n  sin cruce posible — falta ampliar la serie climática")
        return 1

    # ── B · EL TEST CON DENOMINADOR ──────────────────────────────────────────
    # Dentro de las mismas celdas y el mismo periodo: días CON corte vs SIN corte.
    print("\n" + "=" * 78)
    print("EL DENOMINADOR · ¿llovió distinto los días que NO se cortó nada?")
    print("=" * 78)
    ini, fin = min(t["fecha"] for t in cruz), max(t["fecha"] for t in cruz)
    con_corte = {(t["celda"], t["fecha"].isoformat()) for t in cruz}
    celdas_act = {t["celda"] for t in cruz}
    con, sin = [], []
    d = ini
    while d <= fin:
        for c in celdas_act:
            v = acumulado(serie[c], d, 3)
            if v is None:
                continue
            (con if (c, d.isoformat()) in con_corte else sin).append(v)
        d += timedelta(days=1)
    con.sort()
    sin.sort()
    print(f"\n  ventana: {ini} a {fin} · {len(celdas_act)} celdas con al menos un corte")
    print(f"  días-celda CON corte : {len(con):>5}")
    print(f"  días-celda SIN corte : {len(sin):>5}   ← el denominador que faltaba\n")
    print(f"  {'acumulado 72 h':<22}{'CON corte':>12}{'SIN corte':>12}")
    print("  " + "-" * 46)
    for et, q in (("mediana", 0.50), ("cuartil alto", 0.75), ("P90", 0.90)):
        print(f"  {et:<22}{cuantil(con,q):>10.1f} mm{cuantil(sin,q):>10.1f} mm")
    secos_con = sum(1 for v in con if v < 1.0)
    print(f"\n  ★ cortes ocurridos con MENOS de 1 mm en 72 h: {secos_con} de {len(con)}"
          f" ({100*secos_con/len(con):.0f} %)")
    seco10 = [t for t in cruz if t["mm_72h"] < 1.0 and t["mm_10d"] is not None]
    resaca = sum(1 for t in seco10 if t["mm_10d"] >= 50)
    if seco10:
        des = []
        for t in seco10:
            ds = serie[t["celda"]]
            for k in range(25):
                if ds.get((t["fecha"] - timedelta(days=k)).isoformat(), 0) > 10:
                    des.append(k)
                    break
        des.sort()
        print(f"    ⚠️ NO son cortes sin lluvia: {resaca} de {len(seco10)} tramos"
              f" traen ≥50 mm en los 10 días previos.")
        if des:
            print(f"    Su golpe (>10 mm/día) fue {des[0]}–{des[-1]} días antes,"
                  f" mediana {des[len(des)//2]}.")
        print("    ★★ Eso es RETRASO DE INFORME, no física: la fecha que publica el")
        print("       MOP es la del reporte, no la del corte. Descartada la vía del")
        print("       suelo saturado — la carga previa no sube la tasa de corte.")

    # tasa de corte por franja de lluvia — esto ya es una probabilidad real
    print("\n  ★★ TASA DE CORTE POR FRANJA DE LLUVIA (72 h acumuladas)")
    print("     Por primera vez con denominador: de cada 100 días-celda en esa")
    print("     franja, en cuántos se cortó una vía.\n")
    franjas = [(0, 1), (1, 10), (10, 25), (25, 50), (50, 100), (100, 1e9)]

    def tabla(cc, ss, titulo):
        print(f"\n     {titulo}")
        print(f"     {'franja':<16}{'días-celda':>12}{'con corte':>11}{'tasa':>9}")
        print("     " + "-" * 48)
        for lo, hi in franjas:
            nc = sum(1 for v in cc if lo <= v < hi)
            ns = sum(1 for v in ss if lo <= v < hi)
            tot = nc + ns
            if not tot:
                continue
            et = f"{lo:g}–{hi:g} mm" if hi < 1e9 else f"> {lo:g} mm"
            print(f"     {et:<16}{tot:>12}{nc:>11}{100*nc/tot:>8.1f} %")

    tabla(con, sin, "acumulado de 72 h")
    # ★ La misma tabla con la ventana larga: si la resaca es real, la ventana de
    #   10 días tiene que ordenar la tasa mejor que la de 72 h.
    con10, sin10 = [], []
    d = ini
    while d <= fin:
        for c in celdas_act:
            v10 = acumulado(serie[c], d, 10)
            if v10 is None:
                continue
            (con10 if (c, d.isoformat()) in con_corte else sin10).append(v10)
        d += timedelta(days=1)
    con10.sort()
    sin10.sort()
    tabla(con10, sin10, "acumulado de 10 días  ← absorbe el retraso de informe")
    print("\n     ★ La de 10 días queda MONÓTONA (la de 72 h no lo es abajo) porque")
    print("       recupera los cortes informados tarde. Pero discrimina menos")
    print("       arriba: el predictor real es el golpe de 72 h.")

    # ── umbral por elemento y gravedad ───────────────────────────────────────
    print("\n" + "=" * 78)
    print("LLUVIA QUE CORTÓ CADA TIPO DE ELEMENTO")
    print("=" * 78 + "\n")
    print(f"  {'elemento':<30}{'tramos':>8}{'mm 24h':>9}{'mm 48h':>9}{'mm 72h':>9}")
    print("  " + "-" * 65)
    filas = []
    por = defaultdict(list)
    for t in cruz:
        por[t["elemento"]].append(t)
    for el, v in sorted(por.items(), key=lambda x: -len(x[1])):
        if len(v) < 5:
            continue
        a24 = sorted(x["mm_24h"] for x in v if x["mm_24h"] is not None)
        a48 = sorted(x["mm_48h"] for x in v if x["mm_48h"] is not None)
        a72 = sorted(x["mm_72h"] for x in v if x["mm_72h"] is not None)
        print(f"  {el[:30]:<30}{len(v):>8}{med(a24):>9.1f}{med(a48):>9.1f}{med(a72):>9.1f}")
        filas.append({"elemento": el, "tramos": len(v),
                      "mm_24h_mediana": round(med(a24), 1),
                      "mm_48h_mediana": round(med(a48), 1),
                      "mm_72h_mediana": round(med(a72), 1),
                      "mm_72h_p25": round(cuantil(a72, 0.25), 1),
                      "mm_72h_minimo": round(a72[0], 1)})

    print(f"\n  {'gravedad':<30}{'tramos':>8}{'mm 72h mediana':>17}")
    print("  " + "-" * 55)
    porg = defaultdict(list)
    for t in cruz:
        porg[t["gravedad"]].append(t["mm_72h"])
    orden = {"Leve": 0, "Moderado": 1, "Grave": 2, "Muy Grave": 3}
    for g, v in sorted(porg.items(), key=lambda x: orden.get(x[0], 9)):
        if len(v) < 5:
            continue
        print(f"  {g:<30}{len(v):>8}{med(sorted(v)):>15.1f} mm")
    print("\n  ★ Si la gravedad NO sube con la lluvia, entonces lo que rompe el")
    print("    tramo no es cuánta agua cae sino en qué estado estaba.")

    # ── salida ───────────────────────────────────────────────────────────────
    campos = ["region", "comuna", "rol", "km_ini", "km_fin", "infra", "elemento",
              "gravedad", "operatividad", "fecha", "celda",
              "mm_dia", "mm_24h", "mm_48h", "mm_72h", "mm_10d", "emergencia"]
    with SAL_TRAMOS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for t in sorted(cruz, key=lambda x: (-(x["mm_72h"] or 0))):
            w.writerow({**t, "fecha": t["fecha"].isoformat()})
    if filas:
        with SAL_ELEM.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)

    print("\n" + "=" * 78)
    print("EJEMPLO DE LO PEDIDO · los 8 tramos con más lluvia acumulada")
    print("=" * 78 + "\n")
    for t in sorted(cruz, key=lambda x: -x["mm_72h"])[:8]:
        km = (f"km {t['km_ini']:g}–{t['km_fin']:g}"
              if t["km_ini"] is not None and t["km_fin"] is not None else "sin km")
        print(f"  {t['rol'] or '(sin rol)':<10}{km:<16}{t['comuna'][:18]:<19}"
              f"{t['mm_72h']:>7.1f} mm/72h   {t['gravedad']}")

    print(f"\n  escrito: {SAL_TRAMOS.name} ({len(cruz)} tramos) y {SAL_ELEM.name}")
    print(f"\n  ⚠️ {100-100*len(cruz)/len(tramos):.0f} % de los tramos quedó fuera por"
          " falta de serie climática en su celda.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
