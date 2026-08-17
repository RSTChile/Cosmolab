#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grafico_archivo.py — el archivo climático disponible para un punto, del registro
más antiguo al presente.

AUTÓNOMO: descarga sus propios datos. No depende de archivos previos ni de rutas
fijas. Todo lo que baja queda en una carpeta `datos/` junto al script, así que la
segunda vez corre sin red.

    python3 grafico_archivo.py
    python3 grafico_archivo.py --lat -27.37 --lon -70.33 --salida copiapo.png

Requiere: matplotlib.   pip install matplotlib

LO QUE EL GRÁFICO NO ES
-----------------------
No es una serie de pluviosidad continua desde el año 1012. No existe tal cosa.
Son tres archivos distintos que no se pueden empalmar:

  · anillos de árbol → índice adimensional de ancho. En el ciprés de la cordillera
                        el ancho depende del agua disponible, así que sirve de
                        proxy, pero NO son milímetros.
  · NASA POWER       → milímetros de verdad, desde 1981.
  · crónicas         → eventos fechados, sin magnitud continua.

Van en paneles separados con un solo eje de tiempo. Empalmarlos en una curva
sería inventar una serie que nadie midió.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos")

# cronologías del banco ITRDB de la NOAA, cerca de Llay-Llay
CRONOLOGIAS = [
    ("chil007", "San Gabriel", 123, "#7fd8a8"),   # 1132–1975, a 123 km
    ("chil002", "El Asiento", 97, "#7fd4e8"),     # 1012–1972, a  97 km
]
URL_CRN = "https://www.ncei.noaa.gov/pub/data/paleo/treering/chronologies/southamerica/{}.crn"
POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"

# grandes inundaciones de Santiago documentadas en crónicas
INUNDACIONES = [1544, 1574, 1597, 1609, 1618, 1647, 1748, 1783]

FONDO, PAPEL, LINEA = "#0b0f1a", "#111725", "#2a3348"
TEXTO, TENUE, ORO, ROJO = "#c9d4e6", "#7b8798", "#e8c477", "#e88a8a"


# ── descarga ───────────────────────────────────────────────────────────────
def bajar(url: str, destino: str) -> str:
    os.makedirs(CARPETA, exist_ok=True)
    ruta = os.path.join(CARPETA, destino)
    if os.path.exists(ruta) and os.path.getsize(ruta) > 100:
        return ruta
    print(f"  bajando {destino} …", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        datos = r.read()
    with open(ruta, "wb") as fh:
        fh.write(datos)
    return ruta


def lluvia_anual(lat: float, lon: float, desde=1981, hasta=2025) -> list[tuple[int, float]]:
    """Lluvia anual medida, sumando los días. Los faltantes (-999) se descartan."""
    ruta = os.path.join(CARPETA, f"lluvia_{lat}_{lon}_{desde}_{hasta}.json")
    os.makedirs(CARPETA, exist_ok=True)
    if os.path.exists(ruta):
        d = json.load(open(ruta, encoding="utf-8"))
    else:
        print(f"  consultando NASA POWER para {lat}, {lon} …", flush=True)
        q = urllib.parse.urlencode({
            "parameters": "PRECTOTCORR", "community": "AG",
            "latitude": lat, "longitude": lon,
            "start": f"{desde}0101", "end": f"{hasta}1231", "format": "JSON"})
        with urllib.request.urlopen(f"{POWER}?{q}", timeout=180) as r:
            d = json.load(r)
        json.dump(d, open(ruta, "w", encoding="utf-8"))

    por_anio: dict[int, list[float]] = {}
    for f, v in d["properties"]["parameter"]["PRECTOTCORR"].items():
        if v == -999:
            continue
        por_anio.setdefault(int(f[:4]), []).append(v)
    # solo años con el registro casi completo
    return sorted((a, round(sum(vs), 1)) for a, vs in por_anio.items() if len(vs) > 360)


# ── lectura del formato ITRDB ──────────────────────────────────────────────
def leer_crn(ruta: str) -> dict[int, tuple[float, int]]:
    """Formato ITRDB: 6 caracteres de código, 4 de década, y 10 grupos de
    4 (índice ×1000) + 3 (número de muestras). 9990 marca dato ausente."""
    serie: dict[int, tuple[float, int]] = {}
    with open(ruta, encoding="latin-1") as fh:
        for ln in fh.read().splitlines()[3:]:
            if len(ln) < 14:
                continue
            try:
                a0 = int(ln[6:10])
            except ValueError:
                continue
            for i in range(10):
                c = ln[10 + i * 7: 17 + i * 7]
                if len(c) < 7:
                    break
                try:
                    v, n = int(c[:4]), int(c[4:])
                except ValueError:
                    continue
                if v == 9990:
                    continue
                serie[a0 + i] = (v / 1000.0, n)
    return serie


def media_movil(ys, v=11):
    out = []
    for i in range(len(ys)):
        a, b = max(0, i - v // 2), min(len(ys), i + v // 2 + 1)
        out.append(sum(ys[a:b]) / (b - a))
    return out


# ── gráfico ────────────────────────────────────────────────────────────────
def dibujar(series, lluvia, lat, lon, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(series)
    alturas = [2.6] * n + [0.75, 1.7]
    fig = plt.figure(figsize=(16, 3.2 + 3.1 * n), facecolor=FONDO)
    gs = fig.add_gridspec(n + 2, 1, height_ratios=alturas, hspace=0.34,
                          left=0.075, right=0.975, top=0.885, bottom=0.075)

    fig.text(0.075, 0.955, "Archivo climático disponible", color=TEXTO,
             fontsize=20, fontweight="bold", va="top")
    fig.text(0.075, 0.917,
             f"Del registro más antiguo al presente  ·  {lat}° , {lon}°  ·  "
             "archivos distintos, un solo eje de tiempo",
             color=TENUE, fontsize=11.5, va="top")

    ini, fin = 1000, 2030
    ejes = []

    def estilo(ax, titulo, subt=None):
        ax.set_facecolor(PAPEL)
        for s in ax.spines.values():
            s.set_color(LINEA)
        ax.tick_params(colors=TENUE, labelsize=9.5)
        ax.grid(color=LINEA, linewidth=.5, alpha=.55)
        ax.set_axisbelow(True)
        ax.set_xlim(ini, fin)
        ax.text(0.006, 1.045, titulo, transform=ax.transAxes, color=TEXTO,
                fontsize=12, fontweight="bold", va="bottom")
        if subt:
            ax.text(0.006, 1.012, subt, transform=ax.transAxes, color=TENUE,
                    fontsize=9.5, va="bottom")

    ultimo_anillo = 0
    for i, (nombre, km, color, serie) in enumerate(series):
        ax = fig.add_subplot(gs[i], sharex=ejes[0] if ejes else None)
        ejes.append(ax)
        a = sorted(serie)
        y = [serie[k][0] for k in a]
        ultimo_anillo = max(ultimo_anillo, a[-1])
        estilo(ax, f"Anillos de ciprés de la cordillera · {nombre}, a {km} km",
               "Índice de ancho de anillo. Más ancho = más agua. No son milímetros.")
        ax.axhline(1.0, color=TENUE, lw=.8, ls="--", alpha=.6)
        ax.plot(a, y, color=color, lw=.55, alpha=.5)
        ax.plot(a, media_movil(y), color=color, lw=1.9)
        ax.set_ylabel("índice", color=TENUE, fontsize=10)
        ax.set_ylim(0.1, 2.35)

        axn = ax.twinx()
        axn.fill_between(a, [serie[k][1] for k in a], color=ORO, alpha=.16, lw=0)
        axn.set_ylim(0, 230)
        axn.set_yticks([0, 25, 50])
        axn.tick_params(colors=ORO, labelsize=8)
        axn.set_ylabel("árboles", color=ORO, fontsize=9)
        for s in axn.spines.values():
            s.set_color(LINEA)

    axc = fig.add_subplot(gs[n], sharex=ejes[0])
    ejes.append(axc)
    estilo(axc, "Grandes inundaciones de Santiago documentadas en crónicas")
    axc.set_yticks([])
    axc.set_ylim(0, 1)
    for yr in INUNDACIONES:
        axc.plot([yr, yr], [0.12, 0.88], color=ROJO, lw=2.4, solid_capstyle="round")
        axc.text(yr, 0.95, str(yr), color=ROJO, fontsize=8, ha="center", va="bottom")
    axc.axvspan(ini, 1544, color="#000", alpha=.35, lw=0)
    axc.text((ini + 1544) / 2, 0.5, "sin crónica escrita", color=TENUE,
             fontsize=9.5, ha="center", va="center", style="italic")

    axl = fig.add_subplot(gs[n + 1], sharex=ejes[0])
    ejes.append(axl)
    estilo(axl, "Lluvia medida · NASA POWER, desde 1981",
           f"Milímetros de verdad. {len(lluvia)} años de los mil que muestra este gráfico.")
    if lluvia:
        la = [p[0] for p in lluvia]
        lv = [p[1] for p in lluvia]
        media = sum(lv) / len(lv)
        axl.bar(la, lv, color="#7fd4e8", width=.82, alpha=.85)
        axl.axhline(media, color=ORO, lw=1.2, ls="--")
        axl.text(fin + 2, media, f" media {media:.0f} mm", color=ORO, fontsize=9, va="center")
    axl.set_ylabel("mm por año", color=TENUE, fontsize=10)
    axl.set_xlabel("año", color=TENUE, fontsize=10.5)

    for ax in ejes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in ejes:
        ax.axvspan(1981, 2026, color="#7fd4e8", alpha=.09, lw=0)
    if ultimo_anillo and ultimo_anillo < 1981:
        for ax in ejes[:n]:
            ax.axvspan(ultimo_anillo + 1, 1981, color=ROJO, alpha=.17, lw=0)
        ejes[0].text((ultimo_anillo + 1982) / 2, 2.2, "brecha", color=ROJO,
                     fontsize=8.5, ha="center", rotation=90, va="top")

    fig.text(0.075, 0.028,
             "Los paneles NO se pueden empalmar en una sola curva: el índice de anillo no son "
             "milímetros y las crónicas no tienen magnitud.  ·  Anillos: banco ITRDB de la NOAA  ·  "
             "Lluvia: NASA POWER  ·  Crónicas: recopilación histórica",
             color=TENUE, fontsize=8.6, va="bottom")

    fig.savefig(salida, dpi=125, facecolor=FONDO)
    return salida


# ── principal ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Grafica el archivo climático disponible para un punto.")
    ap.add_argument("--lat", type=float, default=-32.84)
    ap.add_argument("--lon", type=float, default=-70.95)
    ap.add_argument("--salida", default=None, help="archivo PNG de salida")
    a = ap.parse_args()

    salida = a.salida or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "archivo_climatico.png")
    print(f"archivo climático para {a.lat}, {a.lon}")

    series = []
    for codigo, nombre, km, color in CRONOLOGIAS:
        try:
            ruta = bajar(URL_CRN.format(codigo), f"{codigo}.crn")
            s = leer_crn(ruta)
            if s:
                series.append((nombre, km, color, s))
                print(f"  {nombre}: {min(s)}–{max(s)}  ({len(s)} años)")
        except Exception as e:
            print(f"  no pude traer {codigo}: {str(e)[:90]}", file=sys.stderr)

    if not series:
        print("sin cronologías: no hay nada que graficar", file=sys.stderr)
        return 1

    try:
        lluvia = lluvia_anual(a.lat, a.lon)
        print(f"  lluvia medida: {lluvia[0][0]}–{lluvia[-1][0]}  ({len(lluvia)} años)")
    except Exception as e:
        print(f"  no pude traer la lluvia: {str(e)[:90]}", file=sys.stderr)
        lluvia = []

    try:
        dibujar(series, lluvia, a.lat, a.lon, salida)
    except ImportError:
        print("falta matplotlib:  pip install matplotlib", file=sys.stderr)
        return 1

    ultimo = max(max(s[3]) for s in series)
    print()
    print(f"escrito: {salida}")
    if ultimo < 1981:
        print(f"brecha entre el fin de los anillos ({ultimo}) y el inicio de la medición (1981): "
              f"{1981 - ultimo} años sin solape")
    return 0


if __name__ == "__main__":
    sys.exit(main())
