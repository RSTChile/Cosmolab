"""
CRUCE EXACTO: AMENAZA DE INUNDACIÓN × RED VIAL Y PUENTES
=========================================================

Reemplaza al muestreo cada 100 metros de `cruzar_amenaza_inundacion.py` por
geometría EXACTA, usando dos bibliotecas que sí se pudieron instalar:

  · **Shapely** (versión 2.1.2) — operaciones geométricas. Trae adentro la
    biblioteca GEOS ya compilada, así que NO necesita GDAL
    (*Geospatial Data Abstraction Library*), que es la que había fallado dos
    veces al instalarse. Veníamos pidiendo la herramienta equivocada.
  · **pyproj** (versión 3.7.2) — cálculo de distancias sobre el elipsoide.
    `Geod.geometry_length()` da el largo geodésico exacto en metros, sin tener
    que proyectar ni aproximar un grado por un número fijo de kilómetros.

QUÉ MEJORA RESPECTO DEL MUESTREO
---------------------------------
1. **El largo dentro de la zona es exacto**, no una fracción de puntos. El
   muestreo tenía un error del orden del paso (100 m) por cada entrada y salida.
2. **Detecta los cruces angostos.** Una franja de amenaza de 40 m de ancho —el
   cruce de una quebrada— podía caer entre dos puntos de muestreo y no verse.
3. **Es mucho más rápido**: el muestreo tardó ~25 minutos; esto son segundos,
   porque GEOS está escrito en C y usa un índice espacial de árbol.

CÓMO SE ARMAN LOS POLÍGONOS CON HUECOS
---------------------------------------
El servicio del Ministerio de Obras Públicas (MOP) entrega los polígonos en
formato esriJSON, como una lista de «anillos» sin decir cuál es el borde
exterior y cuál es un hueco. Se resuelve aplicando diferencia simétrica
sucesiva entre los anillos: eso implementa la regla par-impar, que es la misma
convención que usa el formato, y deja los huecos vacíos sin tener que adivinar
la orientación de cada anillo.
"""

import csv, gzip, json
from collections import defaultdict
from pathlib import Path

from pyproj import Geod
from shapely.geometry import shape, Point, Polygon
from shapely.strtree import STRtree

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI / "datos" / "crudo" / "mop"
AMENAZA = CRUDO / "2026-08-19" / "amenaza_inundacion.esrijson"
TRAMOS = CRUDO / "2026-08-17" / "tramos.geojson.gz"
PUENTES = CRUDO / "2026-08-17" / "puentes.geojson"

GEOD = Geod(ellps="WGS84")          # elipsoide de referencia mundial


def poligono_de_anillos(anillos):
    """Anillos esriJSON → polígono con sus huecos, por regla par-impar."""
    geo = None
    for anillo in anillos:
        if len(anillo) < 4:
            continue
        p = Polygon([(x, y) for x, y, *_ in anillo])
        if not p.is_valid:
            p = p.buffer(0)          # repara autointersecciones
        geo = p if geo is None else geo.symmetric_difference(p)
    return geo


def largo_m(geometria):
    """Largo geodésico en metros de una línea o colección de líneas."""
    if geometria.is_empty:
        return 0.0
    if geometria.geom_type == "LineString":
        return GEOD.geometry_length(geometria)
    if geometria.geom_type in ("MultiLineString", "GeometryCollection"):
        return sum(largo_m(g) for g in geometria.geoms
                   if g.geom_type in ("LineString", "MultiLineString",
                                      "GeometryCollection"))
    return 0.0


def main():
    print("=" * 76)
    print("CRUCE EXACTO · AMENAZA DE INUNDACIÓN (MOP) × RED VIAL Y PUENTES")
    print("=" * 76)

    # --- amenaza ---
    d = json.loads(AMENAZA.read_text(encoding="utf8"))
    polis, meta = [], []
    for f in d["features"]:
        anillos = f.get("geometry", {}).get("rings")
        if not anillos:
            continue
        g = poligono_de_anillos(anillos)
        if g is None or g.is_empty:
            continue
        a = f["attributes"]
        polis.append(g)
        meta.append(dict(clase=(a.get("INUNDACION") or "").strip(),
                         region=(a.get("REGION") or "").strip(),
                         fuente=(a.get("FUENTE") or "").strip()))
    arbol = STRtree(polis)
    print(f"\n  polígonos de amenaza válidos : {len(polis):,} de {len(d['features']):,}")
    area_km2 = sum(abs(GEOD.geometry_area_perimeter(g)[0]) for g in polis) / 1e6
    print(f"  superficie total zonificada   : {area_km2:,.0f} km²")

    # --- puentes ---
    pts = json.loads(PUENTES.read_text(encoding="utf8"))["features"]
    exp_p = []
    for f in pts:
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        pt = Point(g["coordinates"][:2])
        for k in arbol.query(pt):
            if polis[k].contains(pt):
                a = f["properties"]
                exp_p.append(dict(
                    codigo=a.get("CODIGO_PUENTE", ""), nombre=a.get("NOMBRE_PUENTE", ""),
                    rol=a.get("ROL", ""), cauce=a.get("CAUCE_QUEB", ""),
                    region=a.get("REGION", ""), provincia=a.get("PROVINCIA", ""),
                    lat=pt.y, lon=pt.x,
                    amenaza_clase=meta[k]["clase"], amenaza_fuente=meta[k]["fuente"]))
                break
    print(f"\n  puentes evaluados             : {len(pts):,}")
    print(f"  ★ puentes DENTRO de amenaza   : {len(exp_p):,}  ({100*len(exp_p)/len(pts):.1f} %)")

    # --- tramos ---
    with gzip.open(TRAMOS, "rt", encoding="utf8") as fh:
        tr = json.load(fh)["features"]
    exp_t, km_total, km_exp = [], 0.0, 0.0
    for f in tr:
        try:
            linea = shape(f["geometry"])
        except Exception:
            continue
        largo = largo_m(linea) / 1000.0
        km_total += largo
        cand = arbol.query(linea)
        if len(cand) == 0:
            continue
        m_dentro, clases = 0.0, defaultdict(float)
        for k in cand:
            if not polis[k].intersects(linea):
                continue
            trozo = polis[k].intersection(linea)
            mm = largo_m(trozo)
            if mm > 0:
                m_dentro += mm
                clases[meta[k]["clase"]] += mm
        if m_dentro <= 0:
            continue
        kme = min(m_dentro / 1000.0, largo)      # tope: no más que el propio tramo
        km_exp += kme
        a = f["properties"]
        exp_t.append(dict(
            rol=a.get("ROL_LABEL") or a.get("ROL") or "",
            nombre=a.get("NOMBRE_CAMINO", ""), clasificacion=a.get("CLASIFICACION", ""),
            carpeta=a.get("CARPETA", ""), region=a.get("REGION", ""),
            concesionado=a.get("CONCESIONADO", ""),
            km_tramo=round(largo, 3), km_en_amenaza=round(kme, 3),
            fraccion=round(kme / largo, 4) if largo else 0,
            clase_dominante=max(clases, key=clases.get) if clases else ""))

    print(f"\n  tramos evaluados              : {len(tr):,}")
    print(f"  ★ tramos que TOCAN amenaza    : {len(exp_t):,}  ({100*len(exp_t)/len(tr):.1f} %)")
    print(f"  kilómetros de red             : {km_total:,.0f} km")
    print(f"  ★ kilómetros EN amenaza       : {km_exp:,.1f} km  ({100*km_exp/km_total:.2f} %)")

    d1 = AQUI / "datos" / "vial_en_amenaza_inundacion.csv"
    with open(d1, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(exp_t[0].keys())); w.writeheader(); w.writerows(exp_t)
    d2 = AQUI / "datos" / "puentes_en_amenaza_inundacion.csv"
    with open(d2, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(exp_p[0].keys())); w.writeheader(); w.writerows(exp_p)

    # --- lecturas ---
    print("\n" + "=" * 76); print("QUÉ MECANISMO AMENAZA CADA KILÓMETRO"); print("=" * 76)
    porcl = defaultdict(float); ncl = defaultdict(int)
    for t in exp_t:
        porcl[t["clase_dominante"] or "(sin clase)"] += t["km_en_amenaza"]
        ncl[t["clase_dominante"] or "(sin clase)"] += 1
    for c, v in sorted(porcl.items(), key=lambda x: -x[1]):
        print(f"     {c[:40]:42s} {v:8,.1f} km   ({ncl[c]:4d} tramos)")

    print("\n" + "=" * 76); print("DÓNDE"); print("=" * 76)
    porreg = defaultdict(float); nreg = defaultdict(int)
    for t in exp_t:
        porreg[t["region"] or "(sin región)"] += t["km_en_amenaza"]; nreg[t["region"]] += 1
    print("\n  kilómetros expuestos por región:")
    for r, v in sorted(porreg.items(), key=lambda x: -x[1])[:10]:
        print(f"     {str(r)[:34]:36s} {v:8,.1f} km   ({nreg[r]:4d} tramos)")

    print("\n  los 12 tramos con más kilómetros dentro de la amenaza:")
    for t in sorted(exp_t, key=lambda x: -x["km_en_amenaza"])[:12]:
        print(f"     {t['km_en_amenaza']:7.1f} de {t['km_tramo']:7.1f} km ({t['fraccion']*100:3.0f}%)  "
              f"{t['rol'][:8]:9s} {t['nombre'][:36]:38s} {t['clase_dominante'][:22]}")

    print("\n  tramos ÍNTEGRAMENTE dentro de zona de amenaza (fracción = 100%):")
    ent = [t for t in exp_t if t["fraccion"] >= 0.999]
    print(f"     son {len(ent)}, con {sum(t['km_en_amenaza'] for t in ent):,.1f} km en total")
    for t in sorted(ent, key=lambda x: -x["km_en_amenaza"])[:6]:
        print(f"       {t['km_en_amenaza']:6.1f} km  {t['rol'][:8]:9s} {t['nombre'][:44]}")

    cauces = defaultdict(int)
    MALOS = {"S/I", "SIN NOMBRE", "S/N", "CAUCE SIN IDENTIFICACION",
             "ESTERO SIN NOMBRE", "DEFINIR REGION", ""}
    for p in exp_p:
        c = (p["cauce"] or "").strip().upper()
        if c not in MALOS:
            cauces[(c, p["region"])] += 1
    print("\n  cauces con más puentes DENTRO de zona de amenaza:")
    for (c, r), n in sorted(cauces.items(), key=lambda x: -x[1])[:10]:
        print(f"     {n:3d} puentes   {c[:40]:42s} región {r}")


if __name__ == "__main__":
    main()
