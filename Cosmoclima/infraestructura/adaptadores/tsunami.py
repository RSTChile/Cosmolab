"""
ADAPTADOR TSUNAMI · SENAPRED (Servicio Nacional de Prevención y Respuesta ante
Desastres), que republica las cartas del SHOA
==============================================================================

EL PROBLEMA QUE RESUELVE, Y CÓMO
---------------------------------
Las zonas de inundación por tsunami de Chile las levanta el SHOA (Servicio
Hidrográfico y Oceanográfico de la Armada) en las **CITSU** (Carta de Inundación
por Tsunami). El sitio del SHOA **no se puede raspar**: su `robots.txt` dice
`Disallow: /` para todos los agentes. Este adaptador **nunca toca shoa.cl**, ni
siquiera para preguntar.

No hace falta. **SENAPRED republica las mismas cartas** en su propio servidor
geográfico, cuyo `robots.txt` es `Disallow:` vacío, es decir, permite todo. La
capa se llama literalmente «Cartas de Inundación por Tsunami SHOA» y cada
polígono trae el campo `fuente = CITSU-SHOA` y el campo `edicion` con la edición
de la carta de la que salió. O sea: es el dato del SHOA, por el canal que
corresponde, con la trazabilidad puesta por la propia fuente.

La cita correcta es doble:

    Cartas de Inundación por Tsunami (CITSU), Servicio Hidrográfico y
    Oceanográfico de la Armada (SHOA); publicadas por el Servicio Nacional de
    Prevención y Respuesta ante Desastres (SENAPRED).

DOS CAPAS QUE NO SON LO MISMO
------------------------------
El adaptador trae dos capas de SENAPRED y **no las mezcla**, porque miden cosas
distintas:

  1. **Inundación por tsunami (CITSU)** — 5.553 polígonos, cobertura nacional de
     Arica a Magallanes incluida Rapa Nui. Dice *hasta dónde llega el agua y cuán
     hondo*: el campo `inundacion` trae la **profundidad en metros** en clases
     («0 a 1», «1 a 2», «2 a 4», «4 a 6», «6 y más»). Esto es peligro con
     **magnitud absoluta**, no una categoría relativa — que es exactamente el
     hueco que dejó anotado `CORRECCION_RAREZA_PELIGRO.md`.

  2. **Área a evacuar** — 350 polígonos, con código único territorial (`cut`) de
     comuna. Dice *qué se ordena desalojar*. Es una decisión operativa, y por
     diseño es **más amplia** que la zona inundable: lleva margen de seguridad.

Confundirlas sería un error de fondo. La primera es física; la segunda es
protocolo. Se cruzan las dos por separado y se informan por separado.

CÓMO SE CRUZA CON LA INFRAESTRUCTURA
-------------------------------------
Igual que `cruzar_amenaza_exacto.py`, con geometría exacta y no muestreo:

  · **Shapely** (con GEOS adentro, sin necesidad de GDAL) para las operaciones:
    un índice espacial de árbol `STRtree` descarta los pares imposibles, y
    `contains` / `intersection` resuelven el resto sin aproximar.
  · **pyproj** `Geod` para los largos y las áreas: el largo de un tramo dentro de
    la zona inundable se mide **geodésicamente sobre el elipsoide WGS84**, en
    metros de verdad, no en grados convertidos con una regla de tres.

Con puentes y subestaciones la pregunta es binaria (el punto está dentro o no).
Con los tramos viales la pregunta es cuántos kilómetros del tramo caen adentro,
que es una intersección de línea con polígono.

La clase de profundidad que se le asigna a cada elemento es **la peor que lo
toca**, usando el orden que declara la propia fuente en sus etiquetas. No se
inventa un número a partir de la etiqueta: se guarda la etiqueta tal cual y,
aparte, su posición en el orden.

QUÉ ESCRIBE
------------
  datos/crudo/tsunami/<fecha>/       ← el crudo tal como llegó (comprimido) + manifiesto
  datos/tsunami_citsu_cartas.csv     ← inventario de las cartas CITSU y su edición
  datos/tsunami_puentes.csv
  datos/tsunami_vial.csv
  datos/tsunami_subestaciones.csv
  datos/tsunami_areas_a_evacuar.csv  ← las 350 áreas, por comuna

USO
----
    python adaptadores/tsunami.py              # bajar y cruzar
    python adaptadores/tsunami.py --sin-bajar  # reusa el crudo del día
"""

import argparse
import csv
import gzip
import hashlib
import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from pyproj import Geod
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

# --------------------------------------------------------------------------
# Rutas, servicios y constantes
# --------------------------------------------------------------------------

AQUI = Path(__file__).resolve().parent.parent          # …/infraestructura
DATOS = AQUI / "datos"
CRUDO = DATOS / "crudo" / "tsunami"

TRAMOS = DATOS / "crudo" / "mop" / "2026-08-17" / "tramos.geojson.gz"
PUENTES = DATOS / "crudo" / "mop" / "2026-08-17" / "puentes.geojson"
SUBESTACIONES = DATOS / "subestaciones_puntos.csv"

AGENTE = ("CosmoclimaInfraestructura/1.0 (proyecto academico; "
          "consolidador multi-amenaza; rstchile1@gmail.com)")

# Capa 1: las CITSU del SHOA republicadas por SENAPRED.
CITSU = ("https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/"
         "Inundacion_Tsunami_2018/MapServer/0/query")
# Capa 2: áreas a evacuar (SENAPRED en ArcGIS Online, cuenta `geoportalonemi`).
EVACUAR = ("https://services5.arcgis.com/i7S5PSnIJAUcWvSE/ArcGIS/rest/services/"
           "Amenaza_por_Tsunami_2024/FeatureServer/3/query")

POR_PAGINA = 250        # el servidor devuelve error 500 si se le piden miles
ESPERA_S = 2.0          # ritmo prudente entre páginas

# Orden declarado por la propia fuente en la etiqueta del campo `inundacion`.
# No se convierte a número: sólo se ordena para poder decir «la peor que lo toca».
ORDEN_PROFUNDIDAD = ["0 a 1", "1 a 2", "2 a 4", "4 a 6", "6 y más"]

GEOD = Geod(ellps="WGS84")


# --------------------------------------------------------------------------
# Descarga: el crudo se guarda antes de tocarlo
# --------------------------------------------------------------------------

def _pedir(url, params, intentos=3):
    ultimo = None
    for n in range(intentos):
        try:
            req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params),
                                         headers={"User-Agent": AGENTE})
            with urllib.request.urlopen(req, timeout=240) as r:
                return r.read(), r.status
        except Exception as e:                       # noqa: BLE001
            ultimo = e
            time.sleep(4 * (n + 1))
    return None, f"error: {ultimo}"


def _guardar_crudo(carpeta, nombre, contenido, url, extra=None):
    """Guarda comprimido (gzip es sin pérdida: el dato sigue siendo el que llegó)."""
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / (nombre + ".gz")
    with gzip.open(destino, "wb") as fh:
        fh.write(contenido)
    fila = dict(archivo=destino.name, url=url,
                bajado_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                bytes_sin_comprimir=len(contenido),
                bytes_guardados=destino.stat().st_size,
                sha256_sin_comprimir=hashlib.sha256(contenido).hexdigest())
    if extra:
        fila.update(extra)
    with open(carpeta / "_manifiesto.jsonl", "a", encoding="utf8") as fh:
        fh.write(json.dumps(fila, ensure_ascii=False) + "\n")
    return destino


def _contar(url):
    cuerpo, _ = _pedir(url, dict(where="1=1", returnCountOnly="true", f="json"))
    if cuerpo is None:
        return None
    try:
        return json.loads(cuerpo).get("count")
    except Exception:                                # noqa: BLE001
        return None


def bajar_capa(carpeta, url, nombre, campo_orden, total_esperado=None):
    """Descarga paginada en GeoJSON. Cada página es un archivo de crudo."""
    sub = carpeta / nombre
    total = _contar(url)
    if total is None:
        print(f"  · {nombre}: SIN DATO (el servicio no responde al conteo)")
        return sub, None
    print(f"  · {nombre}: {total:,} polígonos declarados por el servicio")
    if total_esperado and total != total_esperado:
        print(f"      OJO: se esperaban {total_esperado:,}. El servicio cambió; se sigue con {total:,}.")
    pagina, bajados = 0, 0
    while bajados < total:
        destino = sub / f"pagina_{pagina:03d}.geojson.gz"
        if destino.exists():                          # reanudable: no se repite
            with gzip.open(destino, "rt", encoding="utf8") as fh:
                bajados += len(json.load(fh).get("features", []))
            pagina += 1
            continue
        cuerpo, estado = _pedir(url, dict(
            where="1=1", outFields="*", returnGeometry="true", outSR=4326,
            orderByFields=campo_orden, resultOffset=bajados,
            resultRecordCount=POR_PAGINA, f="geojson"))
        if cuerpo is None:
            print(f"      página {pagina}: SIN DATO ({estado}) — se corta aquí")
            break
        try:
            n = len(json.loads(cuerpo).get("features", []))
        except Exception:                            # noqa: BLE001
            print(f"      página {pagina}: respuesta ilegible — se corta aquí")
            break
        if n == 0:
            break
        _guardar_crudo(sub, f"pagina_{pagina:03d}.geojson", cuerpo, url,
                       extra=dict(pagina=pagina, offset=bajados, features=n, http=estado))
        bajados += n
        pagina += 1
        print(f"      {bajados:,}/{total:,}", end="\r", flush=True)
        time.sleep(ESPERA_S)
    print(f"      {bajados:,} de {total:,} bajados en {pagina} páginas")
    return sub, total


# --------------------------------------------------------------------------
# Lectura del crudo
# --------------------------------------------------------------------------

def leer_poligonos(carpeta):
    """Todas las páginas → (lista de geometrías shapely, lista de atributos)."""
    geos, atrs = [], []
    for ruta in sorted(carpeta.glob("pagina_*.geojson.gz")):
        with gzip.open(ruta, "rt", encoding="utf8") as fh:
            d = json.load(fh)
        for f in d.get("features", []):
            g = f.get("geometry")
            if not g:
                continue
            try:
                geo = shape(g)
            except Exception:                        # noqa: BLE001
                continue
            if geo.is_empty:
                continue
            if not geo.is_valid:
                geo = geo.buffer(0)                  # repara autointersecciones
            if geo.is_empty:
                continue
            geos.append(geo)
            atrs.append(f.get("properties") or {})
        del d
    return geos, atrs


# --------------------------------------------------------------------------
# Cruce
# --------------------------------------------------------------------------

def _peor(clases):
    """De las clases de profundidad que tocan un elemento, la de más arriba."""
    conocidas = [c for c in clases if c in ORDEN_PROFUNDIDAD]
    if conocidas:
        c = max(conocidas, key=ORDEN_PROFUNDIDAD.index)
        return c, ORDEN_PROFUNDIDAD.index(c) + 1
    return (sorted(clases)[0] if clases else ""), ""


def largo_km(geometria):
    """Largo geodésico en kilómetros de una línea o colección de líneas."""
    if geometria.is_empty:
        return 0.0
    if geometria.geom_type == "LineString":
        return GEOD.geometry_length(geometria) / 1000.0
    if geometria.geom_type in ("MultiLineString", "GeometryCollection"):
        return sum(largo_km(g) for g in geometria.geoms
                   if g.geom_type in ("LineString", "MultiLineString", "GeometryCollection"))
    return 0.0


def cruzar_puntos(arbol, geos, atrs, campos):
    """Puentes y subestaciones: adentro o afuera, y con qué profundidad."""
    def evaluar(pt):
        clases, lugares, ediciones = [], set(), set()
        for k in arbol.query(pt):
            if geos[k].contains(pt):
                a = atrs[k]
                clases.append((a.get(campos["clase"]) or "").strip())
                lugares.add((a.get(campos["lugar"]) or "").strip())
                ediciones.add((a.get(campos["edicion"]) or "").strip())
        if not clases:
            return None
        peor, nivel = _peor(clases)
        return dict(profundidad_m=peor, nivel_profundidad=nivel,
                    carta=" / ".join(sorted(x for x in lugares if x)),
                    edicion_carta=" / ".join(sorted(x for x in ediciones if x)))
    return evaluar


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sin-bajar", action="store_true")
    ap.add_argument("--fecha", default=date.today().isoformat())
    args = ap.parse_args()

    carpeta = CRUDO / args.fecha
    print("=" * 76)
    print("ADAPTADOR TSUNAMI · SENAPRED (cartas CITSU del SHOA republicadas)")
    print("=" * 76)
    print(f"\ncrudo en: {carpeta}")
    print("NUNCA se accede a shoa.cl: su robots.txt lo prohíbe. El dato viene de SENAPRED.")

    dir_citsu = carpeta / "citsu_inundacion"
    dir_evac = carpeta / "areas_a_evacuar"
    if not args.sin_bajar:
        print("\nDESCARGA")
        bajar_capa(carpeta, CITSU, "citsu_inundacion", "objectid_1", total_esperado=5553)
        bajar_capa(carpeta, EVACUAR, "areas_a_evacuar", "OBJECTID", total_esperado=350)

    print("\nLECTURA DEL CRUDO")
    geos, atrs = leer_poligonos(dir_citsu) if dir_citsu.exists() else ([], [])
    if not geos:
        print("  CITSU: SIN DATO — el cruce no se puede hacer.")
        return
    area_km2 = sum(abs(GEOD.geometry_area_perimeter(g)[0]) for g in geos) / 1e6
    print(f"  CITSU (inundación) : {len(geos):,} polígonos válidos · {area_km2:,.0f} km²")
    arbol = STRtree(geos)

    geos_ev, atrs_ev = leer_poligonos(dir_evac) if dir_evac.exists() else ([], [])
    print(f"  áreas a evacuar    : {len(geos_ev):,} polígonos")

    # --- inventario de cartas CITSU ---
    cartas = defaultdict(lambda: defaultdict(int))
    area_carta = defaultdict(float)
    for g, a in zip(geos, atrs):
        clave = ((a.get("lugar") or "").strip(), (a.get("edicion") or "").strip(),
                 (a.get("fuente") or "").strip())
        cartas[clave][(a.get("inundacion") or "").strip()] += 1
        area_carta[clave] += abs(GEOD.geometry_area_perimeter(g)[0]) / 1e6
    filas_cartas = []
    for (lugar, edicion, fuente), clases in sorted(cartas.items()):
        fila = dict(carta=lugar, edicion=edicion, fuente=fuente,
                    poligonos=sum(clases.values()),
                    area_km2=round(area_carta[(lugar, edicion, fuente)], 3))
        for c in ORDEN_PROFUNDIDAD:
            fila[f"poligonos_{c.replace(' ', '_')}"] = clases.get(c, 0)
        fila["poligonos_otra_clase"] = sum(n for c, n in clases.items()
                                           if c not in ORDEN_PROFUNDIDAD)
        filas_cartas.append(fila)
    escribir(filas_cartas, DATOS / "tsunami_citsu_cartas.csv")

    campos = dict(clase="inundacion", lugar="lugar", edicion="edicion")
    evaluar = cruzar_puntos(arbol, geos, atrs, campos)

    # --- subestaciones eléctricas ---
    print("\nCRUCE CON LA INFRAESTRUCTURA")
    subes = []
    with open(SUBESTACIONES, encoding="utf8") as fh:
        todas_subes = list(csv.DictReader(fh))
    for row in todas_subes:
        r = evaluar(Point(float(row["lon"]), float(row["lat"])))
        if r:
            subes.append(dict(subestacion=row["subestacion"], tipo=row["tipo"],
                              region=row["region"], provincia=row["provincia"],
                              responsable=row["responsable"],
                              lat=float(row["lat"]), lon=float(row["lon"]), **r))
    print(f"  subestaciones : {len(subes)} de {len(todas_subes)} dentro de zona inundable")

    # --- puentes ---
    puentes_crudos = json.loads(PUENTES.read_text(encoding="utf8"))["features"]
    puentes = []
    for f in puentes_crudos:
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        pt = Point(g["coordinates"][:2])
        r = evaluar(pt)
        if r:
            a = f["properties"]
            puentes.append(dict(codigo=a.get("CODIGO_PUENTE", ""),
                                nombre=a.get("NOMBRE_PUENTE", ""),
                                rol=a.get("ROL", ""), cauce=a.get("CAUCE_QUEB", ""),
                                region=a.get("REGION", ""), provincia=a.get("PROVINCIA", ""),
                                lat=pt.y, lon=pt.x, **r))
    print(f"  puentes       : {len(puentes):,} de {len(puentes_crudos):,} dentro "
          f"({100 * len(puentes) / len(puentes_crudos):.2f} %)")

    # --- tramos viales ---
    with gzip.open(TRAMOS, "rt", encoding="utf8") as fh:
        tramos_crudos = json.load(fh)["features"]
    tramos, km_red, km_dentro = [], 0.0, 0.0
    for f in tramos_crudos:
        try:
            linea = shape(f["geometry"])
        except Exception:                            # noqa: BLE001
            continue
        largo = largo_km(linea)
        km_red += largo
        cand = arbol.query(linea)
        if len(cand) == 0:
            continue
        km_tramo_dentro, clases = 0.0, defaultdict(float)
        lugares, ediciones = set(), set()
        for k in cand:
            if not geos[k].intersects(linea):
                continue
            trozo = geos[k].intersection(linea)
            kk = largo_km(trozo)
            if kk > 0:
                km_tramo_dentro += kk
                a = atrs[k]
                clases[(a.get("inundacion") or "").strip()] += kk
                lugares.add((a.get("lugar") or "").strip())
                ediciones.add((a.get("edicion") or "").strip())
        if km_tramo_dentro <= 0:
            continue
        kme = min(km_tramo_dentro, largo)             # tope: no más que el propio tramo
        km_dentro += kme
        peor, nivel = _peor(list(clases))
        a = f["properties"]
        tramos.append(dict(
            rol=a.get("ROL_LABEL") or a.get("ROL") or "",
            nombre=a.get("NOMBRE_CAMINO", ""), clasificacion=a.get("CLASIFICACION", ""),
            carpeta=a.get("CARPETA", ""), region=a.get("REGION", ""),
            concesionado=a.get("CONCESIONADO", ""),
            km_tramo=round(largo, 3), km_en_inundacion=round(kme, 3),
            fraccion=round(kme / largo, 4) if largo else 0,
            profundidad_m_peor=peor, nivel_profundidad=nivel,
            profundidad_m_dominante=max(clases, key=clases.get) if clases else "",
            carta=" / ".join(sorted(x for x in lugares if x)),
            edicion_carta=" / ".join(sorted(x for x in ediciones if x))))
    print(f"  tramos viales : {len(tramos):,} de {len(tramos_crudos):,} tocan zona inundable")
    print(f"  kilómetros    : {km_dentro:,.1f} km dentro, de {km_red:,.0f} km de red "
          f"({100 * km_dentro / km_red:.2f} %)")

    escribir(subes, DATOS / "tsunami_subestaciones.csv")
    escribir(puentes, DATOS / "tsunami_puentes.csv")
    escribir(tramos, DATOS / "tsunami_vial.csv")

    # --- áreas a evacuar (capa distinta, informe separado) ---
    if geos_ev:
        filas_ev = []
        for g, a in zip(geos_ev, atrs_ev):
            filas_ev.append(dict(
                # `cut` viene como entero en el servicio: se pasa por str()
                # antes de limpiarlo. No se rellena con ceros ni se reformatea.
                region=str(a.get("region") or "").strip(),
                provincia=str(a.get("provincia") or "").strip(),
                comuna=str(a.get("comuna") or "").strip(),
                cut=str(a.get("cut") or "").strip(),
                sector=str(a.get("sector") or "").strip(),
                area_km2=round(abs(GEOD.geometry_area_perimeter(g)[0]) / 1e6, 3)))
        escribir(filas_ev, DATOS / "tsunami_areas_a_evacuar.csv")
        comunas = {f["cut"] for f in filas_ev if f["cut"]}
        print(f"\n  áreas a evacuar: {len(filas_ev)} polígonos en {len(comunas)} comunas "
              f"(código único territorial), {sum(f['area_km2'] for f in filas_ev):,.0f} km²")

    # --- lecturas ---
    print("\n" + "=" * 76)
    print("CUÁNTA PROFUNDIDAD DE AGUA LE TOCA A CADA KILÓMETRO")
    print("=" * 76)
    por_clase, n_clase = defaultdict(float), defaultdict(int)
    for t in tramos:
        c = t["profundidad_m_dominante"] or "(sin clase)"
        por_clase[c] += t["km_en_inundacion"]
        n_clase[c] += 1
    for c in ORDEN_PROFUNDIDAD + [x for x in por_clase if x not in ORDEN_PROFUNDIDAD]:
        if c in por_clase:
            print(f"     {c + ' m':16s} {por_clase[c]:8,.1f} km   ({n_clase[c]:4d} tramos)")

    print("\n" + "=" * 76)
    print("DÓNDE")
    print("=" * 76)
    porreg, nreg = defaultdict(float), defaultdict(int)
    for t in tramos:
        porreg[t["region"] or "(sin región)"] += t["km_en_inundacion"]
        nreg[t["region"]] += 1
    print("\n  kilómetros de red vial dentro de zona inundable, por región:")
    for r, v in sorted(porreg.items(), key=lambda x: -x[1]):
        print(f"     {str(r)[:36]:38s} {v:8,.1f} km   ({nreg[r]:4d} tramos)")

    pp = defaultdict(int)
    for p in puentes:
        pp[p["region"] or "(sin región)"] += 1
    print("\n  puentes dentro de zona inundable, por región:")
    for r, n in sorted(pp.items(), key=lambda x: -x[1]):
        print(f"     {n:4d}   {r}")

    if subes:
        print("\n  subestaciones eléctricas dentro de zona inundable:")
        for s in subes:
            print(f"     {s['subestacion'][:44]:46s} {s['profundidad_m']} m  "
                  f"(carta {s['carta']}, {s['edicion_carta']})")

    print("\n  los 12 tramos con más kilómetros dentro de zona inundable:")
    for t in sorted(tramos, key=lambda x: -x["km_en_inundacion"])[:12]:
        print(f"     {t['km_en_inundacion']:6.1f} de {t['km_tramo']:7.1f} km "
              f"({t['fraccion'] * 100:3.0f}%)  {t['rol'][:8]:9s} {t['nombre'][:34]:36s} "
              f"{t['profundidad_m_peor']} m")

    print("\nFUENTE: Cartas de Inundación por Tsunami (CITSU), Servicio Hidrográfico y")
    print("Oceanográfico de la Armada (SHOA); publicadas por el Servicio Nacional de")
    print("Prevención y Respuesta ante Desastres (SENAPRED). Ver la columna `edicion_carta`:")
    print("la vigencia del dato la fija la edición de cada carta, no la fecha de descarga.")


def escribir(filas, ruta):
    if not filas:
        print(f"      SIN DATO: no se escribe {ruta.name}")
        return
    with open(ruta, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"      {ruta.name}: {len(filas):,} filas")


if __name__ == "__main__":
    main()
