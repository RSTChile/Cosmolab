"""
CRUCE NACIONAL: ZONAS DE REMOCIÓN EN MASA × RED VIAL, Y COMUNA DE CADA TRAMO
=============================================================================

Hace dos cosas en un solo recorrido de la geometría, porque las dos necesitan
lo mismo: intersectar 14.039 líneas contra polígonos.

1 · ZONA MORFOCLIMÁTICA de cada tramo y cada puente.
    Son las 119 zonas en que SERNAGEOMIN (Servicio Nacional de Geología y
    Minería) declara el peligro de remoción en masa. A diferencia de la capa de
    inundación del MOP (Ministerio de Obras Públicas), que sólo cubre 8 de las
    16 regiones, **estas 119 zonas cubren todo el país**, incluido el norte
    árido. Es la segunda familia de amenaza y la que completa el mapa.

2 · COMUNA de cada tramo.
    Hasta hoy no se podía: `territorio.py` resuelve punto-en-polígono y un tramo
    es una línea. Con Shapely se resuelve exacto, y además se puede repartir el
    largo del tramo entre las comunas que atraviesa en vez de asignarlo entero a
    una sola. Sin esto los kilómetros expuestos no se pueden normalizar ni
    cruzar con población, con los eventos de SENAPRED (Servicio Nacional de
    Prevención y Respuesta ante Desastres) ni con los cortes de luz de la SEC
    (Superintendencia de Electricidad y Combustibles).

POR QUÉ IMPORTA LA HISTORIA DE LA MINUTA
-----------------------------------------
El nivel de peligro de cada zona **tiene vigencia y la capa se sobreescribe a sí
misma**: cuando una declaración vence, desaparece de la fuente. Por eso el
proyecto captura la minuta cuatro veces al día desde el 15-ago-2026. Este script
lee todas las capturas guardadas y reporta qué zonas tuvieron peligro declarado
en algún momento, que es información que ya no existe en el origen.
"""

import csv, glob, gzip, json
from collections import defaultdict
from pathlib import Path

from pyproj import Geod
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

AQUI = Path(__file__).resolve().parent
GEOD = Geod(ellps="WGS84")

ZONAS = AQUI / "datos" / "capas" / "zonas_geograficas.geojson"
COMUNAS = AQUI / "datos" / "capas" / "comunas.geojson"
TRAMOS = AQUI / "datos" / "crudo" / "mop" / "2026-08-17" / "tramos.geojson.gz"
PUENTES = AQUI / "datos" / "crudo" / "mop" / "2026-08-17" / "puentes.geojson"
MINUTAS = AQUI / "datos" / "crudo" / "sernageomin" / "minuta_diaria" / "*.json"


def largo_km(g):
    if g.is_empty:
        return 0.0
    if g.geom_type == "LineString":
        return GEOD.geometry_length(g) / 1000.0
    if g.geom_type in ("MultiLineString", "GeometryCollection"):
        return sum(largo_km(x) for x in g.geoms
                   if "Line" in x.geom_type or x.geom_type == "GeometryCollection")
    return 0.0


def cargar_capa(ruta, campos):
    """geojson → (lista de geometrías, lista de diccionarios de atributos)."""
    d = json.loads(Path(ruta).read_text(encoding="utf8"))
    geos, atrs = [], []
    for f in d["features"]:
        g = f.get("geometry")
        if not g:
            continue
        try:
            gg = shape(g)
            if not gg.is_valid:
                gg = gg.buffer(0)
            if gg.is_empty:
                continue
        except Exception:
            continue
        p = f.get("properties", {})
        geos.append(gg)
        atrs.append({k: p.get(k) for k in campos})
    return geos, atrs


def historia_minuta():
    """Qué nivel tuvo cada zona en cada captura guardada. La fuente no guarda
    esto: si no lo hubiéramos capturado, no existiría."""
    hist = defaultdict(dict)
    archivos = sorted(glob.glob(str(MINUTAS)))
    for a in archivos:
        sello = Path(a).stem
        d = json.loads(Path(a).read_text(encoding="utf8"))
        fs = d if isinstance(d, list) else d.get("features", d)
        for f in fs:
            at = f.get("attributes", f.get("properties", f))
            hist[at.get("OBJECTID")][sello] = at.get("POS_OCURRENCIA")
    return hist, archivos


def main():
    print("=" * 78)
    print("CRUCE NACIONAL · ZONAS DE REMOCIÓN EN MASA × RED VIAL · Y COMUNA POR TRAMO")
    print("=" * 78)

    gz, az = cargar_capa(ZONAS, ["OBJECTID", "ZONA", "REGION", "POS_OCURRENCIA"])
    arbol_z = STRtree(gz)
    print(f"\n  zonas morfoclimáticas (SERNAGEOMIN) : {len(gz)}")
    print(f"  regiones cubiertas                   : {len({(a['REGION'] or '?')[:18] for a in az})}")

    gc, ac = cargar_capa(COMUNAS, ["COMUNA", "NOM_COMUNA", "REGION", "NOM_REGION", "CUT", "COD_COMUNA"])
    arbol_c = STRtree(gc)
    nombre_comuna = lambda a: (a.get("NOM_COMUNA") or a.get("COMUNA") or "?")
    print(f"  comunas                              : {len(gc)}")

    hist, archivos = historia_minuta()
    print(f"  capturas de la minuta guardadas      : {len(archivos)}")
    con_peligro = {z: {v for v in d.values() if v} for z, d in hist.items()}
    con_peligro = {z: v for z, v in con_peligro.items() if v}
    print(f"  ★ zonas con peligro declarado en algún momento de las capturas: "
          f"{len(con_peligro)}")

    # ---------- puentes ----------
    pts = json.loads(PUENTES.read_text(encoding="utf8"))["features"]
    fil_p = []
    for f in pts:
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        pt = Point(g["coordinates"][:2])
        zona = region_z = None
        for k in arbol_z.query(pt):
            if gz[k].contains(pt):
                zona, region_z = az[k]["OBJECTID"], az[k]["REGION"]; break
        com = None
        for k in arbol_c.query(pt):
            if gc[k].contains(pt):
                com = nombre_comuna(ac[k]); break
        a = f["properties"]
        fil_p.append(dict(codigo=a.get("CODIGO_PUENTE", ""), nombre=a.get("NOMBRE_PUENTE", ""),
                          rol=a.get("ROL", ""), cauce=a.get("CAUCE_QUEB", ""),
                          region=a.get("REGION", ""), comuna=com or "",
                          zona_remocion=zona or "", zona_region=region_z or "",
                          zona_nombre=next((a["ZONA"] for a in az if a["OBJECTID"]==zona),""),
                          lat=pt.y, lon=pt.x))
    sinz = sum(1 for x in fil_p if not x["zona_remocion"])
    sinc = sum(1 for x in fil_p if not x["comuna"])
    print(f"\n  puentes: {len(fil_p):,} · sin zona asignada {sinz} · sin comuna {sinc}")

    # ---------- tramos ----------
    with gzip.open(TRAMOS, "rt", encoding="utf8") as fh:
        tr = json.load(fh)["features"]
    fil_t, km_comuna, km_zona = [], defaultdict(float), defaultdict(float)
    km_total = 0.0
    for f in tr:
        try:
            linea = shape(f["geometry"])
        except Exception:
            continue
        largo = largo_km(linea)
        km_total += largo
        a = f["properties"]

        zonas = {}
        for k in arbol_z.query(linea):
            if gz[k].intersects(linea):
                kmz = largo_km(gz[k].intersection(linea))
                if kmz > 0:
                    oid = az[k]["OBJECTID"]
                    zonas[oid] = zonas.get(oid, 0) + kmz
                    km_zona[oid] += kmz
        comunas = {}
        for k in arbol_c.query(linea):
            if gc[k].intersects(linea):
                kmc = largo_km(gc[k].intersection(linea))
                if kmc > 0:
                    n = nombre_comuna(ac[k])
                    comunas[n] = comunas.get(n, 0) + kmc
                    km_comuna[n] += kmc

        fil_t.append(dict(
            rol=a.get("ROL_LABEL") or a.get("ROL") or "", nombre=a.get("NOMBRE_CAMINO", ""),
            clasificacion=a.get("CLASIFICACION", ""), region=a.get("REGION", ""),
            km_tramo=round(largo, 3),
            comuna_principal=max(comunas, key=comunas.get) if comunas else "",
            n_comunas=len(comunas),
            comunas="|".join(f"{c}:{v:.2f}" for c, v in
                             sorted(comunas.items(), key=lambda x: -x[1])),
            zona_remocion=max(zonas, key=zonas.get) if zonas else "",
            zona_nombre=next((a["ZONA"] for a in az if a["OBJECTID"]==max(zonas,key=zonas.get)),"") if zonas else "",
            n_zonas=len(zonas),
            km_en_zona=round(sum(zonas.values()), 3)))

    sin_com = sum(1 for t in fil_t if not t["comuna_principal"])
    multi = sum(1 for t in fil_t if t["n_comunas"] > 1)
    print(f"  tramos: {len(fil_t):,} · {km_total:,.0f} km")
    print(f"     con comuna asignada        : {len(fil_t)-sin_com:,} ({100*(len(fil_t)-sin_com)/len(fil_t):.1f} %)")
    print(f"     que cruzan MÁS DE UNA comuna: {multi:,}  ← por eso no bastaba un punto")
    print(f"     con zona de remoción        : {sum(1 for t in fil_t if t['zona_remocion']):,}")

    # ---------- salidas ----------
    d1 = AQUI / "datos" / "tramos_zona_y_comuna.csv"
    with open(d1, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fil_t[0].keys())); w.writeheader(); w.writerows(fil_t)
    d2 = AQUI / "datos" / "puentes_zona_y_comuna.csv"
    with open(d2, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fil_p[0].keys())); w.writeheader(); w.writerows(fil_p)
    print(f"\n  escrito: {d1.name} · {d2.name}")

    # ---------- lecturas ----------
    print("\n" + "=" * 78)
    print("★ LO QUE ESTABA EXPUESTO CUANDO SERNAGEOMIN DECLARÓ PELIGRO")
    print("=" * 78)
    print("\n  (niveles capturados por nosotros; ya NO están en la fuente)\n")
    tot_km = tot_pu = 0
    for z, niveles in sorted(con_peligro.items(), key=lambda x: -km_zona.get(x[0], 0)):
        kmz = km_zona.get(z, 0.0)
        pu = sum(1 for p in fil_p if p["zona_remocion"] == z)
        tot_km += kmz; tot_pu += pu
        a0 = next((a for a in az if a["OBJECTID"] == z), {})
        print(f"     {str(a0.get('ZONA'))[:24]:26s} {str(a0.get('REGION'))[:22]:24s} "
              f"{sorted(niveles)}  →  {kmz:7,.0f} km · {pu:4d} puentes")
    print(f"\n     TOTAL EXPUESTO EN ESE EPISODIO: {tot_km:,.0f} km de red y {tot_pu:,} puentes")

    print("\n" + "=" * 78); print("COMUNAS CON MÁS RED VIAL"); print("=" * 78)
    for c, v in sorted(km_comuna.items(), key=lambda x: -x[1])[:12]:
        print(f"     {str(c)[:30]:32s} {v:9,.1f} km")

    print("\n" + "=" * 78); print("ZONAS MORFOCLIMÁTICAS CON MÁS RED VIAL"); print("=" * 78)
    for z, v in sorted(km_zona.items(), key=lambda x: -x[1])[:12]:
        a0 = next((a for a in az if a["OBJECTID"] == z), {})
        print(f"     {str(a0.get('ZONA'))[:26]:28s} {str(a0.get('REGION'))[:24]:26s} {v:9,.1f} km")


if __name__ == "__main__":
    main()
