"""
INCENDIOS · el clima no los enciende, pero decide cuánto arden
===============================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «los incendios (la mayoría intencionales) se
han vuelto un problema recurrente: hemos tenido víctimas fatales y destrucción de
inmuebles en grandes cantidades verano por medio, y continuará, así que hay que
incluirlos.»

★★ POR QUÉ ESTE ANÁLISIS ES DISTINTO A TODOS LOS ANTERIORES
--------------------------------------------------------------
Con las vías, la pregunta era «cuánta lluvia hace falta para que esto se corte».
Con los incendios esa pregunta no tiene sentido: **el 24 % son intencionales** y
otro tanto son quemas que se escapan. El clima no los enciende.

Lo que el clima sí decide es **cuánto se propagan**. Y eso se puede medir: para
cada incendio, cuánta lluvia hubo antes en su propia celda, y en qué percentil de
su historia cae ese acumulado.

    la ignición   →  humana, no predecible desde aquí
    la superficie →  depende del combustible, y el combustible depende de la
                     lluvia de los meses previos

Si los incendios grandes se concentran en los percentiles bajos de lluvia previa,
entonces el aporte de este proyecto no es predecir incendios —no se puede— sino
**anticipar cuánto arderían si ocurren**, que es una pregunta de planificación
perfectamente útil.

★ EL DOBLE FILO DE LA LLUVIA, Y POR QUÉ SE MIRAN DOS VENTANAS
---------------------------------------------------------------
La lluvia hace dos cosas opuestas según la escala de tiempo:

    30 días   poca lluvia reciente = combustible seco = arde más
    180 días  MUCHA lluvia en la temporada previa = más pasto crecido =
              más combustible que quemar el verano siguiente

Es el mismo mecanismo que en el desierto florido: un invierno lluvioso deja
pastizal, y ese pastizal es lo que arde. Por eso se miden las dos ventanas por
separado en vez de una sola, y por eso la relación puede no ser monótona.

USO
---
    ../.venv-esa/bin/python incendios_sequia.py
"""

import csv
import glob
import gzip
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI / "datos" / "crudo" / "incendios"
SERIE = AQUI / "datos" / "clima_diario_celdas_era5land.csv"
CUENCA = AQUI / "datos" / "clima_diario_cuenca.csv"
SALIDA = AQUI / "datos" / "incendios_sequia.csv"
SAL_JSON = AQUI / "web" / "publico" / "datos" / "incendios.json"

MALLA = 0.10
VENTANAS = (30, 90, 180)


def celda(la, lo):
    return f"{round(la/MALLA)}_{round(lo/MALLA)}"


MESES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
         "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12}


def parsear_fecha(t):
    """★★ LA MISMA FUENTE USA TRES FORMATOS DISTINTOS.

    Medido sobre las 15 temporadas de CONAF:

        2010-2021  `fh_inicio` es None — no hay fecha, sólo la temporada
        2021-2023  texto en español: «3-feb-2022 20:40»
        2023-2024  epoch en milisegundos: 1688688000000
        2024-2025  vuelve al texto en español

    Leer sólo el epoch —que era el formato del primer archivo que abrí— dejaba
    fuera 20.090 incendios de 25.062, y el análisis corría igual sin avisar de
    nada. El total de incendios con fecha pasa de 4.972 a 25.062.
    """
    if t in (None, "", " "):
        return None
    if isinstance(t, (int, float)):
        try:
            return datetime.fromtimestamp(int(t) / 1000, timezone.utc).date()
        except (ValueError, OSError):
            return None
    partes = str(t).strip().split()[0].split("-")
    if len(partes) != 3:
        return None
    try:
        dia = int(partes[0])
        mes = MESES.get(partes[1][:3].lower())
        anio = int(partes[2])
        return date(anio, mes, dia) if mes else None
    except (ValueError, TypeError):
        return None


def cargar_incendios():
    out = []
    for p in sorted(glob.glob(str(CRUDO / "**" / "incendios_*.geojson.gz"),
                              recursive=True)):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            fs = json.load(fh)["features"]
        for f in fs:
            g = f.get("geometry")
            pr = f["properties"]
            if not g or not g.get("coordinates"):
                continue
            c = g["coordinates"]
            # algunas temporadas vienen como MultiPoint
            while isinstance(c[0], list):
                c = c[0]
            lo, la = c[0], c[1]
            if not (-56 < la < -17 and -110 < lo < -66):
                continue
            fecha = parsear_fecha(pr.get("fh_inicio"))
            if fecha is None:
                continue
            try:
                sup = float(pr.get("superficie") or 0)
            except (TypeError, ValueError):
                sup = 0.0
            out.append({
                "fecha": fecha, "lat": la, "lon": lo, "c": celda(la, lo),
                "sup": sup,
                "causa": (pr.get("causa_gene") or "").strip()[:40],
                "combustible": (pr.get("combus_i") or "")[:28],
                "region": pr.get("region") or "",
                "comuna": pr.get("comuna") or "",
                "temporada": pr.get("temporada") or "",
            })
    return out


def main():
    inc = cargar_incendios()
    print(f"  incendios con fecha y coordenada: {len(inc):,}")
    if not inc:
        return 1
    anios = sorted({i["fecha"].year for i in inc})
    print(f"  temporadas: {anios[0]} a {anios[-1]}")
    print(f"  superficie total: {sum(i['sup'] for i in inc):,.0f} ha")

    necesarias = {i["c"] for i in inc}
    print(f"  celdas necesarias: {len(necesarias):,}", flush=True)
    diario = defaultdict(dict)
    for arch in (SERIE, CUENCA):
        if not arch.exists():
            continue
        with arch.open(encoding="utf-8") as fh:
            r = csv.reader(fh)
            next(r)
            for c, fe, v in r:
                if c in necesarias and v not in ("", "None"):
                    diario[c][fe] = float(v)
    print(f"  con serie: {len(diario):,}")

    # acumulados previos por celda y fecha
    def acumulado(c, d, dias):
        dd = diario.get(c)
        if not dd:
            return None
        t, faltan = 0.0, 0
        for k in range(1, dias + 1):
            v = dd.get((d - timedelta(days=k)).isoformat())
            if v is None:
                faltan += 1
            else:
                t += v
        # se tolera hasta un 10 % de días sin dato; más que eso no es comparable
        return None if faltan > dias * 0.1 else t

    # distribución de cada ventana por celda, para el percentil
    print("  construyendo distribuciones…", flush=True)
    dist = {v: defaultdict(list) for v in VENTANAS}
    for c, dd in diario.items():
        fechas = sorted(dd)
        if len(fechas) < 400:
            continue
        # se muestrea cada 10 días: basta para el percentil y evita 45 M de sumas
        for k in range(200, len(fechas), 10):
            d = date.fromisoformat(fechas[k])
            for v in VENTANAS:
                a = acumulado(c, d, v)
                if a is not None:
                    dist[v][c].append(a)
    for v in VENTANAS:
        for c in dist[v]:
            dist[v][c].sort()

    def pct(v, c, x):
        vals = dist[v].get(c)
        if not vals:
            return None
        lo, hi = 0, len(vals)
        while lo < hi:
            m = (lo + hi) // 2
            if vals[m] < x:
                lo = m + 1
            else:
                hi = m
        return lo / max(len(vals) - 1, 1)

    filas = []
    for i in inc:
        fila = {**i}
        ok = False
        for v in VENTANAS:
            a = acumulado(i["c"], i["fecha"], v)
            fila[f"mm_{v}d"] = round(a, 1) if a is not None else ""
            p = pct(v, i["c"], a) if a is not None else None
            fila[f"pct_{v}d"] = round(p, 4) if p is not None else ""
            ok = ok or p is not None
        if ok:
            filas.append(fila)
    print(f"  con clima previo calculable: {len(filas):,}\n")

    # ── la pregunta: ¿los grandes ocurren tras más o menos lluvia? ──────────
    print("=" * 76)
    print("¿EL CLIMA PREVIO EXPLICA EL TAMAÑO DEL INCENDIO?")
    print("=" * 76 + "\n")
    grandes = [f for f in filas if f["sup"] >= 100]
    chicos = [f for f in filas if f["sup"] < 1]
    print(f"  incendios de 100 ha o más : {len(grandes):,}")
    print(f"  incendios de menos de 1 ha: {len(chicos):,}\n")
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    print(f"  {'ventana':<12}{'grandes (≥100 ha)':>22}{'chicos (<1 ha)':>20}")
    print("  " + "-" * 56)
    for v in VENTANAS:
        g = [f[f"pct_{v}d"] for f in grandes if f[f"pct_{v}d"] != ""]
        ch = [f[f"pct_{v}d"] for f in chicos if f[f"pct_{v}d"] != ""]
        if g and ch:
            print(f"  {v:>4} días{'':<4}{100*med(g):>19.1f} %{100*med(ch):>18.1f} %")
    print("\n  (percentil de lluvia previa en la propia celda: más bajo = más seco)")

    # por tramos de sequía a 90 días
    print("\n  SUPERFICIE MEDIANA SEGÚN QUÉ TAN SECOS FUERON LOS 90 DÍAS PREVIOS")
    tramos = [(0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.01)]
    print(f"  {'percentil de lluvia previa':<30}{'incendios':>11}{'sup. mediana':>15}{'≥100 ha':>10}")
    print("  " + "-" * 68)
    for a, b in tramos:
        sel = [f for f in filas if f["pct_90d"] != "" and a <= f["pct_90d"] < b]
        if not sel:
            continue
        sup = sorted(f["sup"] for f in sel)
        gr = sum(1 for f in sel if f["sup"] >= 100)
        et = f"{100*a:.0f}–{100*b:.0f} %" + ("  (lo más seco)" if a == 0 else "")
        print(f"  {et:<30}{len(sel):>11,}{med(sup):>13.2f} ha{gr:>10,}")

    campos = (["fecha", "temporada", "region", "comuna", "lat", "lon", "c", "sup",
               "causa", "combustible"]
              + [f"mm_{v}d" for v in VENTANAS] + [f"pct_{v}d" for v in VENTANAS])
    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for f in filas:
            w.writerow({**f, "fecha": f["fecha"].isoformat()})
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    print("=" * 76)
    print("INCENDIOS · el clima no los enciende, pero decide cuánto arden")
    print("=" * 76)
    sys.exit(main())
