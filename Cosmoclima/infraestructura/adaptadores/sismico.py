"""
ADAPTADOR SÍSMICO · CSN (Centro Sismológico Nacional, Universidad de Chile)
===========================================================================

QUÉ TRAE ESTE ADAPTADOR
------------------------
La amenaza sísmica del consolidador nacional. Trae **dos canales oficiales del
CSN** (Centro Sismológico Nacional, de la Facultad de Ciencias Físicas y
Matemáticas de la Universidad de Chile), que son complementarios y se auditan
uno al otro:

  1. **Base de eventos** — `evtdb.csn.uchile.cl`. Es la base de datos de
     registros instrumentales del CSN. Tiene un formulario de consulta que
     acepta rango de fechas, caja de latitud/longitud, profundidad y magnitud, y
     **devuelve todo el resultado en una sola página**. Con UNA petición se
     obtiene el catálogo completo desde 2012 sobre cierto umbral de magnitud.
     Ese «desde 2012» no es una elección nuestra: es donde empieza la base.

  2. **Catálogo diario** — `www.sismologia.cl/sismicidad/catalogo/AAAA/MM/AAAAMMDD.html`.
     Es el listado oficial día por día, **sin umbral de magnitud** (incluye los
     sismos chicos que la base de eventos no tiene). Cuesta una petición por día,
     así que se baja sólo una ventana reciente.

Los dos son del CSN. El primero es barato y profundo en el tiempo; el segundo es
caro y completo en magnitud. Se usan juntos: el diario sirve para **medir cuánto
se le escapa** a la base de eventos en la ventana en que se solapan, y esa
medición queda escrita en el informe en vez de suponerse.

LO QUE ESTE ADAPTADOR **NO** ES
--------------------------------
No es zonificación sísmica y no debe leerse como tal. Un catálogo de sismos dice
**dónde ha temblado y cuánto**, no **cuánto puede llegar a temblar**. La
zonificación legal chilena (NCh433, tablas de zona sísmica por comuna, hecha
obligatoria por el artículo 5.5.7 de la Ordenanza General de Urbanismo y
Construcciones) vive dentro de una norma con derechos de autor del INN
(Instituto Nacional de Normalización) y **no se encontró publicada como capa
geográfica ni como tabla abierta**. Ver `FUENTE_SISMICO_TSUNAMI.md`: ahí está el
detalle de lo que se buscó, dónde, y qué trámite corresponde.

Por eso las columnas de salida se llaman como lo que son —«eventos observados a
menos de N kilómetros»— y no «peligro sísmico». Es la misma lección que dejó
`CORRECCION_RAREZA_PELIGRO.md`: la variable mide lo que mide, y se nombra así.

CONDICIONES DE USO
-------------------
El CSN autoriza el uso de sus datos con fines **académicos y de divulgación**;
cualquier otro fin requiere aprobación expresa por escrito. Alexis autorizó
usarlos sobre la base de que este es un proyecto académico, citando siempre al
CSN como fuente. El adaptador respeta `Crawl-delay: 10` de
`evtdb.csn.uchile.cl/robots.txt` y espacia las peticiones al catálogo diario.
La fuente se cita así:

    Centro Sismológico Nacional, Universidad de Chile.
    https://www.sismologia.cl · https://evtdb.csn.uchile.cl

CÓMO SE MIDE LA DISTANCIA (y por qué en dos pasos)
---------------------------------------------------
Los epicentros y la infraestructura vienen en grados (latitud/longitud). Un
grado no son los mismos kilómetros en Arica que en Punta Arenas, así que medir
en grados sería inventar. Se hace en dos pasos:

  · **Paso 1, filtro grueso**: un índice espacial de árbol (`STRtree`) descarta
    de una sola vez casi todos los pares. Se le pide «lo que esté a menos de
    R/60 grados». Se divide por 60 y no por 111 a propósito: 60 km por grado es
    una **cota inferior** del largo de un grado de longitud en todo el
    territorio chileno continental (a 57° de latitud sur un grado de longitud
    mide 60,6 km), de modo que el filtro siempre deja pasar de más y nunca de
    menos. Filtrar de más no cambia el resultado; filtrar de menos sí.

  · **Paso 2, medición exacta**: a cada candidato que sobrevive se le mide la
    distancia geodésica real sobre el elipsoide WGS84 con `pyproj.Geod`. Ése es
    el número que se guarda.

Para los tramos viales (líneas, no puntos) el punto más cercano de la línea se
elige en grados y **después** se mide geodésicamente. La elección del punto
puede desviarse unos pocos metros respecto del óptimo geodésico; a la escala de
estas distancias (decenas de kilómetros) es irrelevante, pero queda dicho.

QUÉ ESCRIBE
------------
  datos/crudo/sismico/<fecha>/          ← el crudo tal como llegó, con manifiesto
  datos/sismico_catalogo_csn.csv        ← catálogo de la base de eventos
  datos/sismico_catalogo_diario_csn.csv ← catálogo diario oficial, ventana reciente
  datos/sismico_eventos_subestaciones.csv
  datos/sismico_eventos_puentes.csv
  datos/sismico_eventos_vial.csv

USO
----
    python adaptadores/sismico.py              # todo: bajar, parsear y cruzar
    python adaptadores/sismico.py --sin-bajar  # reusa el crudo del día, sólo cruza
    python adaptadores/sismico.py --dias 90    # ventana del catálogo diario
"""

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from html import unescape
from pathlib import Path

from pyproj import Geod
from shapely.geometry import Point, shape
from shapely.ops import nearest_points
from shapely.strtree import STRtree

# --------------------------------------------------------------------------
# Rutas y constantes
# --------------------------------------------------------------------------

AQUI = Path(__file__).resolve().parent.parent          # …/infraestructura
DATOS = AQUI / "datos"
CRUDO = DATOS / "crudo" / "sismico"

TRAMOS = DATOS / "crudo" / "mop" / "2026-08-17" / "tramos.geojson.gz"
PUENTES = DATOS / "crudo" / "mop" / "2026-08-17" / "puentes.geojson"
SUBESTACIONES = DATOS / "subestaciones_puntos.csv"

# Identificarse es parte de la cortesía: el operador del servidor puede ver
# quién le pide dato y por qué, y escribirnos si algo le molesta.
AGENTE = ("CosmoclimaInfraestructura/1.0 (proyecto academico; "
          "consolidador multi-amenaza; rstchile1@gmail.com)")

EVTDB = "https://evtdb.csn.uchile.cl/events"
EVTDB_ESPERA_S = 10          # Crawl-delay declarado en evtdb.csn.uchile.cl/robots.txt
DIARIO = "https://www.sismologia.cl/sismicidad/catalogo/{a}/{m:02d}/{a}{m:02d}{d:02d}.html"
DIARIO_ESPERA_S = 1.5        # sismologia.cl no declara robots.txt; ritmo prudente igual

# --- Capas GEOGRÁFICAS de SENAPRED (Servicio Nacional de Prevención y Respuesta
# ante Desastres). Su servidor declara `robots.txt` con `Disallow:` vacío, o sea
# permite todo. Estas capas NO son la zonificación de la norma NCh433 —esa no se
# encontró abierta, ver FUENTE_SISMICO_TSUNAMI.md— pero sí son geografía sísmica
# oficial y de cobertura nacional, que es lo que el consolidador necesita para
# cruzar contra el inventario de infraestructura.
SENAPRED_SISMO = ("https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/"
                  "Sismologia/MapServer/{capa}/query")
CAPAS_SENAPRED = {
    # nombre de archivo        capa   qué es
    "zonas_de_ruptura": (0, "Zonas de Rupturas: el área que rompió cada gran "
                            "terremoto histórico, con año y magnitud"),
    "fallas": (6, "Fallas: trazas de falla con tipo, actividad (Comprobada / "
                  "Probable) y fuente (MINVU o PUC). ESTA es la capa que se usa "
                  "para medir distancias: tiene geometría."),
    "fallas_activas": (8, "Fallas activas: la capa declara 959 trazas y campo "
                          "`nom_falla`, pero el servidor las devuelve SIN "
                          "GEOMETRÍA (geometry: null) aunque se pida "
                          "returnGeometry=true. Se guarda igual, como registro "
                          "de que existe y de que hoy no se puede usar."),
}
SENAPRED_ESPERA_S = 2.0
RADIO_FALLA_KM = 200         # hasta dónde se busca la falla activa más cercana

# Caja que contiene todo el territorio chileno continental, insular cercano y
# el margen de subducción frente a la costa, que es donde nacen los grandes.
CAJA = dict(min_lat=-60, max_lat=-15, min_lon=-115, max_lon=-64)
MAG_MINIMA = 4.0             # umbral de la consulta a la base de eventos
RADIOS_KM = (25, 50, 100)    # anillos en que se cuentan los eventos
UMBRALES_MAG = (5.0, 6.0, 7.0)

GEOD = Geod(ellps="WGS84")
KM_POR_GRADO_COTA_INFERIOR = 60.0     # ver encabezado: cota, no promedio


# --------------------------------------------------------------------------
# Descarga: guardar el crudo ANTES de tocarlo
# --------------------------------------------------------------------------

def _pedir(url, datos_post=None, intentos=3):
    """Una petición HTTP identificada, con reintentos y sin apuro."""
    cuerpo = urllib.parse.urlencode(datos_post).encode() if datos_post else None
    ultimo = None
    for n in range(intentos):
        try:
            req = urllib.request.Request(url, data=cuerpo,
                                         headers={"User-Agent": AGENTE})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read(), r.status
        except Exception as e:                       # noqa: BLE001
            ultimo = e
            time.sleep(3 * (n + 1))
    return None, f"error: {ultimo}"


def _guardar_crudo(carpeta, nombre, contenido, url, extra=None):
    """Escribe el archivo y su manifiesto (url, fecha, tamaño, huella)."""
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / nombre
    destino.write_bytes(contenido)
    manifiesto = carpeta / "_manifiesto.jsonl"
    fila = dict(archivo=nombre, url=url,
                bajado_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                bytes=len(contenido),
                sha256=hashlib.sha256(contenido).hexdigest())
    if extra:
        fila.update(extra)
    with open(manifiesto, "a", encoding="utf8") as fh:
        fh.write(json.dumps(fila, ensure_ascii=False) + "\n")
    return destino


def bajar_base_de_eventos(carpeta):
    """UNA petición al formulario de la base de eventos del CSN."""
    params = dict(min_date="2000-01-01", max_date=date.today().isoformat(),
                  min_lat=CAJA["min_lat"], max_lat=CAJA["max_lat"],
                  min_lon=CAJA["min_lon"], max_lon=CAJA["max_lon"],
                  min_depth=0, max_depth=700,
                  min_mag=MAG_MINIMA, max_mag=10, filter="Buscar")
    print(f"  · base de eventos CSN  (M>={MAG_MINIMA}, una sola petición)…", end="", flush=True)
    cuerpo, estado = _pedir(EVTDB, params)
    if cuerpo is None:
        print(f"  SIN DATO ({estado})")
        return None
    print(f"  {len(cuerpo):,} bytes")
    return _guardar_crudo(carpeta, "csn_evtdb_eventos.html", cuerpo, EVTDB,
                          extra=dict(consulta=params, http=estado))


def bajar_catalogo_diario(carpeta, dias):
    """Una petición por día. Ritmo prudente, y los huecos se anotan."""
    sub = carpeta / "csn_catalogo_diario"
    hoy = date.today()
    faltantes, bajados = [], 0
    print(f"  · catálogo diario CSN  ({dias} días, ~{dias * DIARIO_ESPERA_S / 60:.0f} min)…")
    for i in range(dias, 0, -1):
        d = hoy - timedelta(days=i)
        url = DIARIO.format(a=d.year, m=d.month, d=d.day)
        nombre = f"{d.year}{d.month:02d}{d.day:02d}.html"
        if (sub / nombre).exists():                 # ya bajado antes: no repetir
            bajados += 1
            continue
        cuerpo, estado = _pedir(url, intentos=2)
        if cuerpo is None or b"sismologia detalle" not in cuerpo:
            faltantes.append(d.isoformat())
        else:
            _guardar_crudo(sub, nombre, cuerpo, url, extra=dict(dia=d.isoformat()))
            bajados += 1
        time.sleep(DIARIO_ESPERA_S)
        if bajados and bajados % 60 == 0:
            print(f"      {bajados} días…", flush=True)
    print(f"      {bajados} días con dato · {len(faltantes)} sin dato")
    if faltantes:
        (sub / "_dias_sin_dato.txt").write_text("\n".join(faltantes), encoding="utf8")
    return sub, faltantes


def bajar_capas_senapred(carpeta):
    """Las tres capas geográficas sísmicas de SENAPRED. Son chicas: una petición
    cada una, en GeoJSON, sin paginar."""
    sub = carpeta / "senapred_sismologia"
    for nombre, (capa, que_es) in CAPAS_SENAPRED.items():
        destino = sub / f"{nombre}.geojson"
        if destino.exists():
            continue
        url = SENAPRED_SISMO.format(capa=capa)
        params = dict(where="1=1", outFields="*", returnGeometry="true",
                      outSR=4326, f="geojson")
        cuerpo, estado = _pedir(url + "?" + urllib.parse.urlencode(params))
        if cuerpo is None:
            print(f"  · {nombre}: SIN DATO ({estado})")
            continue
        try:
            n = len(json.loads(cuerpo).get("features", []))
        except Exception:                            # noqa: BLE001
            print(f"  · {nombre}: respuesta ilegible, SIN DATO")
            continue
        _guardar_crudo(sub, f"{nombre}.geojson", cuerpo, url,
                       extra=dict(capa=capa, features=n, que_es=que_es, http=estado))
        print(f"  · {nombre}: {n:,} elementos")
        time.sleep(SENAPRED_ESPERA_S)
    return sub


def leer_capa_geojson(ruta):
    """GeoJSON en disco → (geometrías shapely reparadas, atributos)."""
    if not ruta.exists():
        return [], []
    d = json.loads(ruta.read_text(encoding="utf8"))
    geos, atrs = [], []
    for f in d.get("features", []):
        g = f.get("geometry")
        if not g:
            continue
        try:
            geo = shape(g)
        except Exception:                            # noqa: BLE001
            continue
        if geo.is_empty:
            continue
        if not geo.is_valid:
            geo = geo.buffer(0)
        if geo.is_empty:
            continue
        geos.append(geo)
        atrs.append(f.get("properties") or {})
    return geos, atrs


# --------------------------------------------------------------------------
# Lectura: del HTML del CSN a filas
# --------------------------------------------------------------------------

def _texto(celda):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", celda))).strip()


def leer_base_de_eventos(ruta):
    """Tabla `id="events"` → filas (fecha UTC, lat, lon, profundidad, magnitud)."""
    h = ruta.read_text(encoding="utf8", errors="replace")
    tabla = re.search(r'<table id="events".*?</table>', h, re.S)
    if not tabla:
        return []
    filas = []
    for tr in re.findall(r"<tr.*?</tr>", tabla.group(0), re.S):
        celdas = [_texto(c) for c in re.findall(r"<t[hd].*?</t[hd]>", tr, re.S)]
        if len(celdas) < 5 or not re.match(r"\d{4}-\d{2}-\d{2}", celdas[0]):
            continue
        ident = re.search(r'href="/event/([0-9a-f]+)"', tr)
        try:
            filas.append(dict(
                id_csn=ident.group(1) if ident else "",
                fecha_utc=celdas[0], lat=float(celdas[1]), lon=float(celdas[2]),
                profundidad_km=float(celdas[3]), magnitud=float(celdas[4]),
                fuente="CSN base de eventos (evtdb.csn.uchile.cl)"))
        except ValueError:
            continue
    return filas


def leer_catalogo_diario(carpeta):
    """Cada página diaria → filas (fecha local, lugar, fecha UTC, lat, lon, …)."""
    filas = []
    for ruta in sorted(carpeta.glob("*.html")):
        h = ruta.read_text(encoding="utf8", errors="replace")
        tabla = re.search(r'<table class="sismologia detalle">.*?</table>', h, re.S)
        if not tabla:
            continue
        for tr in re.findall(r"<tr.*?</tr>", tabla.group(0), re.S):
            celdas = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)
            if len(celdas) < 5:
                continue
            primera = celdas[0]
            m_fecha = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", primera)
            if not m_fecha:
                continue
            partes = [p for p in re.split(r"<br\s*/?>", primera)]
            lugar = _texto(partes[1]) if len(partes) > 1 else ""
            coord = re.findall(r"(-?\d+\.\d+)", celdas[2])
            prof = re.search(r"(-?\d+(?:\.\d+)?)", _texto(celdas[3]))
            mag = re.match(r"(-?\d+(?:\.\d+)?)\s*(\S+)?", _texto(celdas[4]))
            if len(coord) < 2 or not mag:
                continue
            ident = re.search(r"/informes/\d{4}/\d{2}/(\d+)\.html", primera)
            filas.append(dict(
                id_csn=ident.group(1) if ident else "",
                fecha_local=m_fecha.group(1), lugar=lugar,
                fecha_utc=_texto(celdas[1]),
                lat=float(coord[0]), lon=float(coord[1]),
                profundidad_km=float(prof.group(1)) if prof else "",
                magnitud=float(mag.group(1)),
                escala_magnitud=mag.group(2) or "",
                fuente="CSN catálogo diario (www.sismologia.cl)"))
    return filas


# --------------------------------------------------------------------------
# Cruce con la infraestructura
# --------------------------------------------------------------------------

def _resumen_vacio():
    fila = {}
    for r in RADIOS_KM:
        fila[f"eventos_M{MAG_MINIMA:g}+_a_{r}km"] = 0
        for u in UMBRALES_MAG:
            fila[f"eventos_M{u:g}+_a_{r}km"] = 0
    fila.update(mag_max_100km="", fecha_mag_max="", dist_mag_max_km="",
                evento_mas_cercano_km="", mag_evento_mas_cercano="")
    return fila


def _resumen(distancias):
    """distancias: lista de (km, magnitud, fecha). Devuelve las columnas."""
    fila = _resumen_vacio()
    if not distancias:
        return fila
    for r in RADIOS_KM:
        dentro = [d for d in distancias if d[0] <= r]
        fila[f"eventos_M{MAG_MINIMA:g}+_a_{r}km"] = len(dentro)
        for u in UMBRALES_MAG:
            fila[f"eventos_M{u:g}+_a_{r}km"] = sum(1 for d in dentro if d[1] >= u)
    dentro100 = [d for d in distancias if d[0] <= max(RADIOS_KM)]
    if dentro100:
        km, mg, fe = max(dentro100, key=lambda x: x[1])
        fila["mag_max_100km"] = mg
        fila["fecha_mag_max"] = fe
        fila["dist_mag_max_km"] = round(km, 1)
    km, mg, _ = min(distancias, key=lambda x: x[0])
    fila["evento_mas_cercano_km"] = round(km, 1)
    fila["mag_evento_mas_cercano"] = mg
    return fila


def _texto_ruptura(a):
    """Etiqueta legible de una zona de ruptura: «1960 M9.5».

    Ojo con el nombre de los campos de la fuente: `zona_ruptu` NO es el nombre
    del lugar, es el **largo de la ruptura en kilómetros** (el terremoto de 1960
    trae «1000 Km.»). Se guarda aparte con su nombre verdadero y no se mete en
    la etiqueta, para que nadie lo lea como si fuera un topónimo.
    """
    anio = str(a.get("año") or "").strip()
    mag = str(a.get("magnitud") or "").strip()
    return f"{anio} M{mag}".strip() if (anio or mag) else ""


def cruzar(eventos, rupturas=None, atr_rupturas=None, fallas=None, atr_fallas=None):
    """Cruza el catálogo y las capas geográficas con la infraestructura."""
    puntos_ev = [Point(e["lon"], e["lat"]) for e in eventos]
    arbol = STRtree(puntos_ev)
    radio_max = max(RADIOS_KM)
    grados = radio_max / KM_POR_GRADO_COTA_INFERIOR      # filtro grueso, generoso

    rupturas = rupturas or []
    fallas = fallas or []
    arbol_rup = STRtree(rupturas) if rupturas else None
    arbol_fal = STRtree(fallas) if fallas else None
    grados_falla = RADIO_FALLA_KM / KM_POR_GRADO_COTA_INFERIOR

    def _dist_geodesica(geom_a, geom_b):
        """Distancia en km entre dos geometrías: el par de puntos más cercano se
        elige en grados y la distancia se mide sobre el elipsoide."""
        pa, pb = nearest_points(geom_a, geom_b)
        return GEOD.inv(pa.x, pa.y, pb.x, pb.y)[2] / 1000.0

    def zonas_ruptura(geom):
        """Zonas de ruptura histórica que contienen o cortan la geometría."""
        if arbol_rup is None:
            return dict(zonas_de_ruptura="", ruptura_mag_max="")
        toca, mags = [], []
        for k in arbol_rup.query(geom):
            if rupturas[k].intersects(geom):
                toca.append(_texto_ruptura(atr_rupturas[k]))
                try:
                    mags.append(float(atr_rupturas[k].get("magnitud")))
                except (TypeError, ValueError):
                    pass
        return dict(zonas_de_ruptura=" | ".join(sorted(set(toca))),
                    ruptura_mag_max=max(mags) if mags else "")

    def falla_mas_cercana(geom):
        """Falla mapeada más cercana, con su distancia geodésica en kilómetros."""
        if arbol_fal is None:
            return dict(falla_mas_cercana="", actividad_falla="", dist_falla_km="")
        mejor, nombre, actividad = None, "", ""
        for k in arbol_fal.query(geom, predicate="dwithin", distance=grados_falla):
            km = _dist_geodesica(geom, fallas[k])
            if mejor is None or km < mejor:
                a = atr_fallas[k]
                mejor = km
                nombre = (a.get("s__sisnp") or a.get("s__estru_1") or "").strip()
                actividad = (a.get("s__sactivi") or "").strip()
        if mejor is None or mejor > RADIO_FALLA_KM:
            # «Sin dato» explícito: no es cero, es que no hay falla mapeada
            # dentro del radio de búsqueda. Un cero diría lo contrario.
            return dict(falla_mas_cercana="", actividad_falla="",
                        dist_falla_km=f">{RADIO_FALLA_KM}")
        return dict(falla_mas_cercana=nombre, actividad_falla=actividad,
                    dist_falla_km=round(mejor, 1))

    def distancias_a_punto(pt):
        salida = []
        for k in arbol.query(pt, predicate="dwithin", distance=grados):
            e = eventos[k]
            _, _, m = GEOD.inv(pt.x, pt.y, e["lon"], e["lat"])
            km = m / 1000.0
            if km <= radio_max:
                salida.append((km, e["magnitud"], e["fecha_utc"]))
        return salida

    def distancias_a_linea(linea):
        salida = []
        for k in arbol.query(linea, predicate="dwithin", distance=grados):
            e = eventos[k]
            pl, _ = nearest_points(linea, puntos_ev[k])
            _, _, m = GEOD.inv(pl.x, pl.y, e["lon"], e["lat"])
            km = m / 1000.0
            if km <= radio_max:
                salida.append((km, e["magnitud"], e["fecha_utc"]))
        return salida

    # --- subestaciones eléctricas ---
    subes = []
    with open(SUBESTACIONES, encoding="utf8") as fh:
        for row in csv.DictReader(fh):
            pt = Point(float(row["lon"]), float(row["lat"]))
            fila = dict(subestacion=row["subestacion"], tipo=row["tipo"],
                        region=row["region"], provincia=row["provincia"],
                        responsable=row["responsable"], lat=pt.y, lon=pt.x)
            fila.update(_resumen(distancias_a_punto(pt)))
            fila.update(zonas_ruptura(pt))
            fila.update(falla_mas_cercana(pt))
            subes.append(fila)

    # --- puentes ---
    puentes = []
    for f in json.loads(PUENTES.read_text(encoding="utf8"))["features"]:
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        pt = Point(g["coordinates"][:2])
        a = f["properties"]
        fila = dict(codigo=a.get("CODIGO_PUENTE", ""), nombre=a.get("NOMBRE_PUENTE", ""),
                    rol=a.get("ROL", ""), cauce=a.get("CAUCE_QUEB", ""),
                    region=a.get("REGION", ""), provincia=a.get("PROVINCIA", ""),
                    lat=pt.y, lon=pt.x)
        fila.update(_resumen(distancias_a_punto(pt)))
        fila.update(zonas_ruptura(pt))
        fila.update(falla_mas_cercana(pt))
        puentes.append(fila)

    # --- tramos viales ---
    with gzip.open(TRAMOS, "rt", encoding="utf8") as fh:
        crudos = json.load(fh)["features"]
    tramos, km_red = [], 0.0
    for f in crudos:
        try:
            linea = shape(f["geometry"])
        except Exception:                            # noqa: BLE001
            continue
        largo = GEOD.geometry_length(linea) / 1000.0 if not linea.is_empty else 0.0
        km_red += largo
        a = f["properties"]
        fila = dict(rol=a.get("ROL_LABEL") or a.get("ROL") or "",
                    nombre=a.get("NOMBRE_CAMINO", ""),
                    clasificacion=a.get("CLASIFICACION", ""),
                    region=a.get("REGION", ""),
                    concesionado=a.get("CONCESIONADO", ""),
                    km_tramo=round(largo, 3))
        fila.update(_resumen(distancias_a_linea(linea)))
        fila.update(zonas_ruptura(linea))
        fila.update(falla_mas_cercana(linea))
        tramos.append(fila)

    return subes, puentes, tramos, km_red


# --------------------------------------------------------------------------
# Escritura y lectura del resultado
# --------------------------------------------------------------------------

def escribir(filas, ruta):
    if not filas:
        print(f"      SIN DATO: no se escribe {ruta.name}")
        return
    with open(ruta, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"      {ruta.name}: {len(filas):,} filas")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sin-bajar", action="store_true",
                    help="reusa el crudo ya guardado del día de hoy")
    ap.add_argument("--dias", type=int, default=365,
                    help="ventana del catálogo diario del CSN (días hacia atrás)")
    ap.add_argument("--fecha", default=date.today().isoformat(),
                    help="carpeta de crudo a usar (AAAA-MM-DD)")
    args = ap.parse_args()

    carpeta = CRUDO / args.fecha
    print("=" * 76)
    print("ADAPTADOR SÍSMICO · CSN (Centro Sismológico Nacional, U. de Chile)")
    print("=" * 76)
    print(f"\ncrudo en: {carpeta}")

    if not args.sin_bajar:
        print("\nDESCARGA")
        bajar_base_de_eventos(carpeta)
        bajar_capas_senapred(carpeta)
        if args.dias > 0:
            bajar_catalogo_diario(carpeta, args.dias)

    print("\nLECTURA")
    ruta_evtdb = carpeta / "csn_evtdb_eventos.html"
    eventos = leer_base_de_eventos(ruta_evtdb) if ruta_evtdb.exists() else []
    if not eventos:
        print("  base de eventos: SIN DATO — el cruce no se puede hacer.")
        sys.exit(1)
    fechas = sorted(e["fecha_utc"] for e in eventos)
    print(f"  base de eventos : {len(eventos):,} sismos M>={MAG_MINIMA:g}"
          f"  ({fechas[0][:10]} → {fechas[-1][:10]})")
    escribir(eventos, DATOS / "sismico_catalogo_csn.csv")

    diarios = []
    sub = carpeta / "csn_catalogo_diario"
    if sub.exists():
        diarios = leer_catalogo_diario(sub)
        if diarios:
            fd = sorted(d["fecha_utc"] for d in diarios)
            print(f"  catálogo diario : {len(diarios):,} sismos (toda magnitud)"
                  f"  ({fd[0][:10]} → {fd[-1][:10]})")
            escribir(diarios, DATOS / "sismico_catalogo_diario_csn.csv")

    # --- auditoría cruzada entre los dos canales del CSN ---
    if diarios:
        desde, hasta = min(d["fecha_utc"] for d in diarios), max(d["fecha_utc"] for d in diarios)
        en_ventana_diario = {(d["fecha_utc"][:16], round(d["magnitud"], 1))
                             for d in diarios if d["magnitud"] >= MAG_MINIMA}
        en_ventana_base = {(e["fecha_utc"][:16], round(e["magnitud"], 1))
                           for e in eventos if desde <= e["fecha_utc"] <= hasta}
        comunes = en_ventana_diario & en_ventana_base
        print(f"\n  AUDITORÍA entre los dos canales (ventana {desde[:10]} → {hasta[:10]}):")
        print(f"      catálogo diario, M>={MAG_MINIMA:g} : {len(en_ventana_diario):,}")
        print(f"      base de eventos, misma ventana     : {len(en_ventana_base):,}")
        print(f"      coinciden (fecha al minuto + mag)  : {len(comunes):,}")
        if en_ventana_diario:
            print(f"      → la base de eventos recoge el "
                  f"{100 * len(comunes) / len(en_ventana_diario):.0f} % de lo que "
                  f"el catálogo diario lista sobre M{MAG_MINIMA:g}")

    # --- capas geográficas de SENAPRED ---
    sen = carpeta / "senapred_sismologia"
    rupturas, atr_rup = leer_capa_geojson(sen / "zonas_de_ruptura.geojson")
    # Se usa la capa 6 («Fallas») y NO la capa 8 («Fallas activas»): la 8 llega
    # sin geometría desde el servidor, así que no se puede medir con ella.
    fallas, atr_fal = leer_capa_geojson(sen / "fallas.geojson")
    sin_geom, _ = leer_capa_geojson(sen / "fallas_activas.geojson")
    if (sen / "fallas_activas.geojson").exists() and not sin_geom:
        print("  capa «Fallas activas» (SENAPRED, capa 8): SIN DATO GEOMÉTRICO — "
              "el servidor la devuelve con geometry: null. No se usa.")
    if rupturas:
        area = sum(abs(GEOD.geometry_area_perimeter(g)[0]) for g in rupturas) / 1e6
        print(f"  zonas de ruptura (SENAPRED): {len(rupturas)} polígonos · {area:,.0f} km²")
        filas_rup = []
        for g, a in zip(rupturas, atr_rup):
            filas_rup.append(dict(
                anio=a.get("año", ""), magnitud=a.get("magnitud", ""),
                largo_ruptura_km=a.get("zona_ruptu", ""),   # sí: el campo se llama así
                profundidad_km=a.get("profun_km", ""),
                efecto_secundario=a.get("efecto_sec", ""),
                area_km2=round(abs(GEOD.geometry_area_perimeter(g)[0]) / 1e6, 1)))
        escribir(sorted(filas_rup, key=lambda x: str(x["anio"])),
                 DATOS / "sismico_zonas_de_ruptura.csv")
    else:
        print("  zonas de ruptura (SENAPRED): SIN DATO")
    if fallas:
        largo = sum(GEOD.geometry_length(g) for g in fallas) / 1000.0
        act = defaultdict(int)
        for a in atr_fal:
            act[(a.get("s__sactivi") or "").strip() or "(sin declarar)"] += 1
        print(f"  fallas          (SENAPRED): {len(fallas)} trazas · {largo:,.0f} km · "
              + ", ".join(f"{k}: {v}" for k, v in sorted(act.items(), key=lambda x: -x[1])))
    else:
        print("  fallas          (SENAPRED): SIN DATO")

    print("\nCRUCE CON LA INFRAESTRUCTURA")
    t0 = time.time()
    subes, puentes, tramos, km_red = cruzar(eventos, rupturas, atr_rup, fallas, atr_fal)
    print(f"  ({time.time() - t0:.0f} s)")
    escribir(subes, DATOS / "sismico_eventos_subestaciones.csv")
    escribir(puentes, DATOS / "sismico_eventos_puentes.csv")
    escribir(tramos, DATOS / "sismico_eventos_vial.csv")

    # --- lecturas ---
    col100 = f"eventos_M{MAG_MINIMA:g}+_a_100km"
    col6 = "eventos_M6+_a_100km"
    col7 = "eventos_M7+_a_100km"
    print("\n" + "=" * 76)
    print(f"CUÁNTA INFRAESTRUCTURA TIENE SISMOS GRANDES CERCA "
          f"(catálogo CSN {fechas[0][:4]}–{fechas[-1][:4]})")
    print("=" * 76)
    for nombre, filas in (("subestaciones", subes), ("puentes", puentes), ("tramos viales", tramos)):
        con = sum(1 for f in filas if f[col100])
        con6 = sum(1 for f in filas if f[col6])
        con7 = sum(1 for f in filas if f[col7])
        print(f"\n  {nombre}: {len(filas):,}")
        print(f"     con ≥1 sismo M{MAG_MINIMA:g}+ a menos de 100 km : "
              f"{con:,}  ({100 * con / len(filas):.1f} %)")
        print(f"     con ≥1 sismo M6+ a menos de 100 km    : "
              f"{con6:,}  ({100 * con6 / len(filas):.1f} %)")
        print(f"     con ≥1 sismo M7+ a menos de 100 km    : "
              f"{con7:,}  ({100 * con7 / len(filas):.1f} %)")

    print(f"\n  kilómetros de red vial evaluados: {km_red:,.0f} km")
    km7 = sum(f["km_tramo"] for f in tramos if f[col7])
    print(f"  kilómetros con un M7+ a menos de 100 km: {km7:,.0f} km "
          f"({100 * km7 / km_red:.1f} %)")

    print("\n  regiones por kilómetros con un M7+ a menos de 100 km:")
    porreg = defaultdict(float)
    for f in tramos:
        if f[col7]:
            porreg[f["region"] or "(sin región)"] += f["km_tramo"]
    for r, v in sorted(porreg.items(), key=lambda x: -x[1])[:12]:
        print(f"     {str(r)[:36]:38s} {v:9,.0f} km")

    print("\n  las 12 subestaciones con más sismos M5+ a menos de 50 km:")
    for f in sorted(subes, key=lambda x: -x["eventos_M5+_a_50km"])[:12]:
        print(f"     {f['eventos_M5+_a_50km']:4d} sismos M5+  ·  M máx {f['mag_max_100km'] or '—'}  "
              f"·  {f['subestacion'][:44]}")

    if rupturas:
        print("\n" + "=" * 76)
        print("INFRAESTRUCTURA DENTRO DE ZONAS DE RUPTURA DE GRANDES TERREMOTOS")
        print("=" * 76)
        for nombre, filas in (("subestaciones", subes), ("puentes", puentes),
                              ("tramos viales", tramos)):
            n = sum(1 for f in filas if f["zonas_de_ruptura"])
            print(f"  {nombre:16s} {n:6,} de {len(filas):6,}  ({100 * n / len(filas):5.1f} %)")
        km_rup = sum(f["km_tramo"] for f in tramos if f["zonas_de_ruptura"])
        print(f"  kilómetros de red vial dentro de alguna zona de ruptura: "
              f"{km_rup:,.0f} km ({100 * km_rup / km_red:.1f} %)")
        print("\n  las zonas de ruptura que más kilómetros de red vial contienen:")
        porzona = defaultdict(float)
        for f in tramos:
            for z in (f["zonas_de_ruptura"] or "").split(" | "):
                if z:
                    porzona[z] += f["km_tramo"]
        for z, v in sorted(porzona.items(), key=lambda x: -x[1])[:10]:
            print(f"     {z[:52]:54s} {v:9,.0f} km")

    if fallas:
        print("\n" + "=" * 76)
        print("DISTANCIA A LA FALLA MAPEADA MÁS CERCANA")
        print("=" * 76)
        print(f"  (sólo hay {len(fallas)} trazas mapeadas en todo el país: la capa cubre "
              "sobre todo\n   zonas urbanas. Un «>200» no significa que no haya falla, "
              "sino que no hay\n   falla MAPEADA en el radio de búsqueda.)")
        for nombre, filas in (("subestaciones", subes), ("puentes", puentes),
                              ("tramos viales", tramos)):
            cerca = [f for f in filas if isinstance(f["dist_falla_km"], float)]
            print(f"\n  {nombre} ({len(filas):,}):")
            for corte in (5, 10, 25, 50):
                n = sum(1 for f in cerca if f["dist_falla_km"] <= corte)
                print(f"     a menos de {corte:3d} km de una falla mapeada: "
                      f"{n:6,}  ({100 * n / len(filas):5.1f} %)")
            n_sin = len(filas) - len(cerca)
            print(f"     sin falla mapeada a menos de {RADIO_FALLA_KM} km: {n_sin:6,}"
                  f"  ({100 * n_sin / len(filas):5.1f} %)")

    print("\nFUENTE: Centro Sismológico Nacional, Universidad de Chile "
          "(www.sismologia.cl · evtdb.csn.uchile.cl).")
    print("Capas geográficas: SENAPRED (Servicio Nacional de Prevención y Respuesta")
    print("ante Desastres), servicio SIIE/Sismología — zonas de ruptura y fallas activas.")
    print("Uso académico y de divulgación, según las condiciones declaradas por el CSN.")
    print("ESTO ES UN CATÁLOGO DE EVENTOS OBSERVADOS, NO UNA ZONIFICACIÓN DE PELIGRO.")


if __name__ == "__main__":
    main()
