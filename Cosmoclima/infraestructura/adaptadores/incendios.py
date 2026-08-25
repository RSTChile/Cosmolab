"""
ADAPTADOR · INCENDIO FORESTAL
==============================

POR QUÉ ESTA AMENAZA ENTRA AHORA
---------------------------------
En el registro de SENAPRED (Servicio Nacional de Prevención y Respuesta ante
Desastres) el incendio forestal es la **tercera emergencia más frecuente de
Chile**: 4.855 eventos en diez años, el 9,6 % de todas las emergencias
registradas (ver `CATALOGO_MODOS_DE_FALLA.md`). La Matriz ya tiene remoción en
masa e inundación; sin incendio le falta un tercio del problema.

★ EL PROBLEMA DE ACCESO, Y CÓMO SE RESOLVIÓ
--------------------------------------------
CONAF (**Corporación Nacional Forestal**) es el organismo dueño de esta amenaza:
es quien combate los incendios y quien lleva el registro. Pero:

  · `www.conaf.cl` responde, y su `robots.txt` **prohíbe descargar PDF**
    (`Disallow: /*.pdf$`) y **prohíbe las URL con parámetros**. Sus estadísticas
    históricas se publican justamente como PDF y como tablas dinámicas.
  · `sit.conaf.cl` (su Sistema de Información Territorial) responde con una
    página, pero **no expone ningún servicio de datos**: `/arcgis`, `/server` y
    `/geoserver` devuelven 404. Verificado el 17-ago-2026 y de nuevo el
    19-ago-2026.
  · `www.ide.cl`, la Infraestructura de Datos Geoespaciales de Chile, **prohíbe
    explícitamente a ClaudeBot** en su `robots.txt` (`User-agent: ClaudeBot /
    Disallow: /`). Se respeta y **no se consulta**. Queda como pendiente para
    que lo consulte una persona.

La vía que sí funciona es **CIREN** (Centro de Información de Recursos
Naturales), que opera el servidor geográfico de la **IDE MINAGRI** —
Infraestructura de Datos Espaciales del Ministerio de Agricultura, la cartera de
la que depende CONAF. CIREN republica ahí el dato de CONAF con servicios ArcGIS
abiertos y sin restricción en `robots.txt`.

Es la misma solución que el proyecto ya usó para el catastro vegetacional. Vale
la pena anotarlo como patrón: **cuando el organismo dueño no expone servicios,
suele exponerlos la infraestructura de datos de su ministerio.**

QUÉ TRAE ESTE ADAPTADOR — LAS TRES PATAS DEL ENCARGO
-----------------------------------------------------

**(a) Ocurrencia histórica, incendio por incendio.**
`esri.ciren.cl/.../INCENDIOS_FORESTALES` publica **quince temporadas
completas, 2010-2011 a 2024-2025**, un registro por incendio, con coordenada,
comuna, fecha de inicio y de extinción, causa general y específica, tipo de
combustible y **superficie afectada desglosada por tipo de vegetación**
(pino por edad, eucalipto, arbolado nativo, matorral, pastizal, agrícola,
desechos). Es dato de CONAF: el campo `n_mbito` dice quién reportó.

**(b) Zonificación de riesgo.**
No se encontró una capa de peligro de incendio publicada por CONAF. Lo que sí
existe, y se baja, es la capa comunal de **ARCLIM** (Atlas de Riesgo Climático,
del Ministerio del Medio Ambiente con el Centro de Ciencia del Clima y la
Resiliencia, CR2), republicada por SERNAGEOMIN (Servicio Nacional de Geología y
Minería) en su nube. Descompone el riesgo en sus tres partes —**amenaza,
exposición y sensibilidad**— para presente y futuro, y por separado para bosque
nativo y para plantaciones.

⚠️ **Hay que decirlo sin adornos: esto NO es la zonificación de CONAF.** Es un
índice académico-ministerial, y **cubre 215 comunas, no las 345 del país**. Se
usa como referencia declarada, no como si fuera el pronunciamiento del organismo
dueño de la amenaza.

**(c) Combustible / uso de suelo.**
El **Catastro de Recursos Vegetacionales y Usos de la Tierra de CONAF** está en
CIREN, una capa por región, con el año del levantamiento en el nombre (van de
2011 en Aysén a 2019 en varias). Son más de un millón de polígonos en total:
bajarlos enteros no tiene sentido para lo que hace falta. Se pide en cambio, al
propio servicio, la **suma de hectáreas por comuna y por uso de la tierra**
(`groupByFieldsForStatistics`), que es la tabla de carga de combustible que la
Matriz necesita, y cabe en un archivo chico.

★ EL DATO VIENE ROTO Y HAY QUE DECIRLO: QUINCE ESQUEMAS DISTINTOS
------------------------------------------------------------------
Las quince temporadas **no comparten esquema**. Cada una se cargó en su año, con
los nombres de campo que tenía a mano, y nadie las homologó después. Lo que se
encontró al revisarlas una por una:

  · La fecha de inicio se llama `inicio_in` en unas temporadas y `fh_inicio` en
    otras; el combustible es `combus_i` o `combustibl`; el nombre del incendio
    es `nom_incen` o `nombre`; el total de plantaciones es `total_plan` o
    `subtotal_p`… y así con casi todos los campos.
  · **El nombre de la comuna viene en blanco en 4 temporadas completas**
    (2013-2014, 2014-2015, 2016-2017 y casi toda 2010-2011): 19.542 incendios
    en total. En esas temporadas lo que sí viene es `codcom`, el código CUT
    (Código Único Territorial) de la comuna.
  · Al revés, en 2024-2025 el `codcom` viene vacío en los 6.262 registros, y lo
    que sí viene es el nombre.
  · El nombre de la **región** viene en blanco en 6 temporadas.
  · Donde viene el nombre, viene con grafías distintas según el año:
    `CURANILAHUE` en unas y `Curanilahue` en otras. Contadas como texto crudo
    son dos comunas distintas, y el conteo por comuna sale partido en dos.
  · Dos registros traen la temporada corrompida (caracteres ilegibles).

Cómo se resuelve, sin inventar nada:

  1. **La temporada la pone la capa, no el campo.** El servicio dice en el
     nombre de la capa de qué temporada es; eso es autoritativo y no se corrompe.
  2. **La comuna se resuelve por su código CUT** contra `datos/capas/comunas.geojson`
     — la capa COMUNAS_2020 que el proyecto ya bajó de SERNAGEOMIN. Así las
     quince temporadas quedan con un solo nombre por comuna, y con su región.
  3. **Si no hay código, se resuelve por nombre**, comparando en mayúsculas y
     sin acentos contra esa misma capa. Es un desempate mecánico, no una
     adivinanza.
  4. **Si no se resuelve por ninguna de las dos vías, queda «sin resolver» y se
     cuenta aparte.** No se asigna a la comuna más parecida.

Cada fila del CSV lleva la columna `comuna_resuelta_por` diciendo cuál de las
tres cosas pasó. Quien lea el archivo puede descartar lo que no le sirva.

LO QUE ESTE ADAPTADOR **NO** HACE
----------------------------------
No estima peligro. Contar dónde hubo incendios mide **dónde los hubo**, que es
historia, no amenaza futura; y contar hectáreas de matorral mide **cuánto hay
para quemar**, que es carga de combustible, no probabilidad de ignición. Son
insumos del peligro, no el peligro. Confundirlos sería repetir el error que el
proyecto ya se corrigió a sí mismo en `CORRECCION_RAREZA_PELIGRO.md`.
"""

import csv
import gzip
import json
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

from shapely.geometry import Point, shape
from shapely.strtree import STRtree

AQUI = Path(__file__).resolve().parent.parent
FECHA = date.today().isoformat()
CRUDO = AQUI / "datos" / "crudo" / "incendios" / FECHA
SALIDA = AQUI / "datos"

CIREN = "https://esri.ciren.cl/server/rest/services"
SGM = "https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/services"

SRV_INCENDIOS = f"{CIREN}/INCENDIOS_FORESTALES/FeatureServer"
SRV_VEGETACION = f"{CIREN}/RECURSOS_VEGETACIONALES_Y_USOS_DE_LA_TIERRA_1/FeatureServer"
SRV_ARCLIM = f"{SGM}/Riesgo_Incendios_Forestales_WFL1/FeatureServer/4"

# capa → temporada. Ojo con el salto: no hay capa 13 ni 15 en el servicio.
CAPAS_TEMPORADA = {
    0: "2010-2011", 1: "2011-2012", 2: "2012-2013", 3: "2013-2014",
    4: "2014-2015", 5: "2015-2016", 6: "2016-2017", 7: "2017-2018",
    8: "2018-2019", 9: "2019-2020", 10: "2020-2021", 11: "2021-2022",
    12: "2022-2023", 14: "2023-2024", 16: "2024-2025",
}

# capa → (región, año del levantamiento del catastro de CONAF)
CAPAS_VEGETACION = {
    0: ("Arica y Parinacota", 2015), 1: ("Tarapacá", 2016),
    2: ("Antofagasta", 2019), 3: ("Atacama", 2018),
    4: ("Coquimbo", 2014), 5: ("Valparaíso", 2019),
    6: ("Metropolitana", 2019), 7: ("O'Higgins", 2013),
    8: ("Maule", 2016), 9: ("Ñuble", 2015), 10: ("Biobío", 2015),
    11: ("Araucanía", 2014), 12: ("Los Ríos", 2014), 13: ("Los Lagos", 2013),
    14: ("Aysén", 2011),
    15: ("Magallanes · prov. Magallanes", 2018),
    16: ("Magallanes · prov. Antártica Chilena", 2017),
    17: ("Magallanes · prov. Tierra del Fuego", 2017),
    18: ("Magallanes · prov. Última Esperanza", 2019),
}

PAGINA = 5000
PAUSA_S = 1.5      # ritmo prudente: son servicios públicos de un servicio del Estado


# --------------------------------------------------------------------------
# acceso
# --------------------------------------------------------------------------

def _pedir(url, timeout=180):
    """Una petición. Devuelve (json, None) o (None, motivo). Nunca lanza."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.load(r)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, OSError) as e:
        return None, f"{type(e).__name__}: {str(e)[:140]}"
    if isinstance(d, dict) and "error" in d:
        return None, f"el servicio devolvió error: {str(d['error'])[:160]}"
    return d, None


def traer_capa(base, capa, timeout=180):
    """Capa completa como GeoJSON, paginando. (geojson, None) o (None, motivo)."""
    rasgos, offset = [], 0
    while True:
        url = (f"{base}/{capa}/query?where=1%3D1&outFields=*&outSR=4326"
               f"&f=geojson&resultOffset={offset}&resultRecordCount={PAGINA}")
        d, motivo = _pedir(url, timeout)
        if d is None:
            return None, motivo
        lote = d.get("features", [])
        rasgos.extend(lote)
        if len(lote) < PAGINA:
            break
        offset += PAGINA
        time.sleep(PAUSA_S)
    return {"type": "FeatureCollection", "features": rasgos}, None


def estadistica_agrupada(base, capa, campos, campo_suma, timeout=240,
                         contar=True):
    """Le pide al servicio la suma agrupada, en vez de bajar los polígonos.

    Para el catastro vegetacional esto cambia el orden de magnitud: en vez de
    traer más de un millón de polígonos para después sumarlos aquí, el servidor
    devuelve unas decenas de filas ya sumadas. El dato es el mismo; lo que se
    evita es mover geometría que nadie va a mirar.
    """
    stats = [{"statisticType": "sum", "onStatisticField": campo_suma,
              "outStatisticFieldName": "ha"}]
    if contar:
        stats.append({"statisticType": "count", "onStatisticField": "objectid",
                      "outStatisticFieldName": "n_poligonos"})
    outstats = json.dumps(stats)
    q = urllib.parse.urlencode({
        "where": "1=1", "groupByFieldsForStatistics": ",".join(campos),
        "outStatistics": outstats, "returnGeometry": "false", "f": "json"})
    d, motivo = _pedir(f"{base}/{capa}/query?{q}", timeout)
    if d is None:
        return None, motivo
    return [f["attributes"] for f in d.get("features", [])], None


def combustible_de_region(capa):
    """Hectáreas por comuna y uso, con un plan B declarado.

    Las tres regiones más forestales del país —Maule, Ñuble y Biobío— son
    justamente las que más polígonos tienen, y la consulta completa hace caer el
    servicio con un error 400 genérico («Unable to complete operation»).

    Se acotó cuál de las dos piezas de la consulta es la que lo tumba, probando
    por separado: **no es agrupar por subuso, es contar polígonos**. Con
    `count(objectid)` el servicio devuelve 400 en esas tres regiones; pidiendo
    sólo la suma de hectáreas, las mismas tres responden sin problema. El plan B
    baja entonces esa pieza —el número de polígonos, que es informativo— y
    conserva la que importa, que es la hectárea.

    Devuelve (filas, detalle, motivo). `detalle` dice qué se logró pedir, y esa
    marca viaja hasta el CSV: nadie va a creer que tiene un dato que no está.
    """
    filas, motivo = estadistica_agrupada(
        SRV_VEGETACION, capa, ["nom_com", "uso", "subuso"], "superf_ha")
    if filas is not None:
        return filas, "comuna x uso x subuso, con recuento de polígonos", None
    time.sleep(PAUSA_S)
    filas2, motivo2 = estadistica_agrupada(
        SRV_VEGETACION, capa, ["nom_com", "uso", "subuso"], "superf_ha",
        contar=False)
    if filas2 is not None:
        return filas2, ("comuna x uso x subuso, SIN recuento de polígonos "
                        "(el servicio devuelve 400 al contar)"), None
    time.sleep(PAUSA_S)
    filas3, motivo3 = estadistica_agrupada(
        SRV_VEGETACION, capa, ["nom_com", "uso"], "superf_ha", contar=False)
    if filas3 is not None:
        return filas3, ("comuna x uso, sin subuso y sin recuento "
                        "(el servicio no aguantó más detalle)"), None
    return None, None, (f"{motivo} | sin recuento: {motivo2} | "
                        f"sin subuso: {motivo3}")


# --------------------------------------------------------------------------
# (a) ocurrencia
# --------------------------------------------------------------------------

def _num(v):
    """Número o None. Un campo vacío NO se convierte en cero."""
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _txt(v):
    s = str(v).strip() if v is not None else ""
    return "" if s.upper() in ("NONE", "NULL") else s


def _clave(s):
    """Texto → clave de comparación: mayúsculas, sin acentos, sin dobles espacios.

    Sirve para que `CURANILAHUE`, `Curanilahue` y `Curanilahué` sean la misma
    comuna. Es una comparación mecánica y reversible, no una corrección del dato:
    el texto original se conserva intacto en su propia columna.
    """
    s = unicodedata.normalize("NFD", _txt(s))
    # Mn = tildes y diéresis sueltas · Cf = caracteres invisibles de formato.
    # El Cf importa de verdad: el dato de CONAF trae un guion suave (U+00AD)
    # metido dentro de «VEHÍ­CULOS», invisible al ojo, que partía esa causa en
    # dos filas distintas en el conteo.
    s = "".join(c for c in s
                if unicodedata.category(c) not in ("Mn", "Cf"))
    s = s.replace(",", " ").replace(";", " ")
    return " ".join(s.upper().split())


def _primero(p, *nombres):
    """El primer campo de la lista que traiga algo. Las quince temporadas usan
    nombres distintos para lo mismo; esta función es el traductor."""
    for n in nombres:
        if n in p:
            v = _txt(p[n])
            if v:
                return v
    return ""


def _primero_num(p, *nombres):
    for n in nombres:
        if n in p and p[n] not in (None, ""):
            v = _num(p[n])
            if v is not None:
                return v
    return None


def cargar_comunas():
    """COMUNAS_2020 (SERNAGEOMIN) → tres índices: por código CUT, por nombre y
    por geometría.

    Es la misma capa de la que ya depende el resto del proyecto, así que las
    comunas de incendios y las de remoción en masa calzan por construcción.

    ★ POR QUÉ HACE FALTA EL ÍNDICE GEOMÉTRICO — LA REGIÓN DE ÑUBLE
    ---------------------------------------------------------------
    1.547 incendios no se resolvieron ni por código ni por nombre. Al mirarlos
    uno por uno resultó que todos traen códigos CUT que empiezan en `084…`
    (`08401`, `08405`, `08413`…) y el nombre en blanco. Ese `084` era la
    **provincia de Ñuble dentro de la antigua región del Biobío**. En 2018 Ñuble
    se separó y pasó a ser la región 16: los códigos `084xx` **dejaron de
    existir**, y por eso no aparecen en COMUNAS_2020.

    Se podría escribir a mano una tabla `08401 → 16101`, pero sería confiar en
    una correspondencia que no está verificada en ningún dato de esta máquina.
    Se hace lo que sí se puede comprobar: **cada incendio trae su coordenada**,
    así que se le pregunta a la geometría en qué comuna cae. Y como para los
    otros 94.078 incendios el código SÍ resuelve, se pueden comparar las dos
    vías y medir cuánto coinciden — que es la comprobación que hace creíble el
    resultado, en vez de tener que creerle a la tabla.
    """
    ruta = AQUI / "datos" / "capas" / "comunas.geojson"
    if not ruta.exists():
        return {}, {}, None
    d = json.loads(ruta.read_text(encoding="utf8"))
    por_cut, por_nombre, poligonos, refs = {}, {}, [], []
    for f in d["features"]:
        p = f["properties"]
        reg = dict(comuna=_txt(p.get("COMUNA")), region=_txt(p.get("REGION")),
                   provincia=_txt(p.get("PROVINCIA")),
                   cut=_txt(p.get("CUT_COM")))
        por_cut[reg["cut"]] = reg
        por_nombre.setdefault(_clave(reg["comuna"]), reg)
        try:
            g = shape(f["geometry"])
            if not g.is_valid:
                g = g.buffer(0)
            poligonos.append(g)
            refs.append(reg)
        except Exception:
            pass
    geo_idx = (STRtree(poligonos), poligonos, refs) if poligonos else None
    return por_cut, por_nombre, geo_idx


def comuna_por_punto(geo_idx, lon, lat):
    """¿En qué comuna cae este punto? None si en ninguna (mar, frontera, error)."""
    if geo_idx is None or lon is None or lat is None:
        return None
    arbol, polis, refs = geo_idx
    pt = Point(lon, lat)
    for k in arbol.query(pt):
        if polis[k].contains(pt):
            return refs[k]
    return None


def normalizar_causa(texto):
    """Quita la numeración del código de causa y homologa mayúsculas y acentos.

    Las quince temporadas escriben la misma causa de seis formas distintas:
    `2.1. Incendios intencionales`, `INCENDIOS INTENCIONALES`, `2.1 - Incendios
    Intencionales`… Sin esto, la causa más frecuente de Chile aparece repartida
    en seis filas y ninguna gana. Se guarda además el texto crudo.
    """
    t = _txt(texto)
    if not t:
        return ""
    # saca un prefijo tipo "2.1." / "2.1 -" / "1.2.2." del inicio
    partes = t.split(None, 1)
    if partes and all(c.isdigit() or c in ".-" for c in partes[0]) and len(partes) > 1:
        t = partes[1]
    t = t.lstrip("-. ")
    return _clave(t)


def normalizar_incendios(geojson, temporada, por_cut, por_nombre, geo_idx):
    """GeoJSON de una temporada → filas planas, con la comuna ya resuelta.

    `temporada` viene del NOMBRE DE LA CAPA, no del campo: el campo trae dos
    registros con la temporada corrompida y la capa no se corrompe.

    Devuelve (filas, acuerdos, desacuerdos): las dos últimas son el control de
    calidad — cuántas veces el código CUT y la geometría dicen lo mismo, y
    cuántas no. Se mide en TODOS los incendios que tienen las dos cosas, no
    sólo en los que hizo falta resolver por geometría.
    """
    filas, acuerdo, desacuerdo = [], 0, 0
    for f in geojson["features"]:
        p = f.get("properties", {})
        g = f.get("geometry") or {}
        lon = lat = None
        c = g.get("coordinates")
        if c:
            # las capas mezclan Point y MultiPoint: se toma el primer vértice
            if g.get("type") == "Point":
                lon, lat = c[0], c[1]
            elif g.get("type") == "MultiPoint" and c:
                lon, lat = c[0][0], c[0][1]

        cut = _primero(p, "codcom").split(".")[0].zfill(5) if _primero(p, "codcom") else ""
        nombre_crudo = _primero(p, "comuna")

        # control de calidad: qué dice la geometría, siempre que se pueda
        ref_geo = comuna_por_punto(geo_idx, lon, lat)

        # 1) por código CUT · 2) por nombre · 3) por geometría · 4) sin resolver
        ref, via = por_cut.get(cut), "codigo_cut"
        if ref is None and nombre_crudo:
            ref = por_nombre.get(_clave(nombre_crudo))
            via = "nombre"
        if ref is None and ref_geo is not None:
            ref, via = ref_geo, "geometria"
        if ref is None:
            via = "sin resolver"

        if ref is not None and ref_geo is not None and via != "geometria":
            if ref["cut"] == ref_geo["cut"]:
                acuerdo += 1
            else:
                desacuerdo += 1

        causa_cruda = _primero(p, "causa_gene")
        filas.append(dict(
            temporada=temporada,
            nombre=_primero(p, "nom_incen", "nombre"),
            comuna=(ref["comuna"] if ref else ""),
            region=(ref["region"] if ref else ""),
            provincia=(ref["provincia"] if ref else ""),
            codigo_cut=(ref["cut"] if ref else cut),
            comuna_resuelta_por=via,
            comuna_texto_original=nombre_crudo,
            region_texto_original=_primero(p, "region"),
            reportado_por=_primero(p, "ambito", "n_mbito"),
            fecha_inicio=_primero(p, "fh_inicio", "inicio_in"),
            fecha_extincion=_primero(p, "fh_extinci", "extincion"),
            combustible=_primero(p, "combustibl", "combus_i"),
            causa_general=causa_cruda,
            causa_general_normalizada=normalizar_causa(causa_cruda),
            causa_especifica=_primero(p, "causa_espe"),
            inicio_certeza=_primero(p, "inicio_cer", "inicio_c"),
            ha_plantaciones=_primero_num(p, "subtotal_p", "total_plan"),
            ha_vegetacion_natural=_primero_num(p, "subtotal_v", "total_veg"),
            ha_arbolado_nativo=_primero_num(p, "arbolado"),
            ha_matorral=_primero_num(p, "matorral"),
            ha_pastizal=_primero_num(p, "pastizal"),
            ha_otros=_primero_num(p, "subtotal_o", "total_otra", "total_o"),
            ha_total=_primero_num(p, "superficie"),
            utm_este=_primero_num(p, "utm_e"),
            utm_norte=_primero_num(p, "utm_n"),
            huso=_primero(p, "huso_op_", "huso"),
            comuna_segun_geometria=(ref_geo["comuna"] if ref_geo else ""),
            lat=lat, lon=lon,
            tiene_coordenada=bool(lat is not None and lon is not None),
        ))
    return filas, acuerdo, desacuerdo


# --------------------------------------------------------------------------
# (b) riesgo ARCLIM
# --------------------------------------------------------------------------

def traer_arclim():
    """Capa comunal de riesgo ARCLIM. Los nombres reales están en los alias.

    El servicio arrastra los nombres que quedaron de una unión de tablas
    (`ARCLIM_incendios_bosques_nat_20`), ilegibles. El alias sí trae el nombre
    verdadero (`bosques_incendios_bnativos_riesgo_fut`). Se lee la metadata de
    la capa y se renombra con el alias, para que el CSV se pueda leer.
    """
    meta, motivo = _pedir(f"{SRV_ARCLIM}?f=json")
    if meta is None:
        return None, None, motivo
    alias = {f["name"]: (f.get("alias") or f["name"]) for f in meta["fields"]}
    campos = [n for n in alias if n not in ("Shape__Area", "Shape__Length")]
    q = urllib.parse.urlencode({
        "where": "1=1", "outFields": ",".join(campos),
        "returnGeometry": "false", "f": "json"})
    d, motivo = _pedir(f"{SRV_ARCLIM}/query?{q}")
    if d is None:
        return None, None, motivo
    filas = []
    for f in d.get("features", []):
        filas.append({alias[k]: v for k, v in f["attributes"].items()
                      if k in alias})
    return filas, meta, None


# --------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("ADAPTADOR · INCENDIO FORESTAL · CONAF vía CIREN / IDE MINAGRI")
    print("=" * 78)
    CRUDO.mkdir(parents=True, exist_ok=True)

    proc = [
        "INCENDIO FORESTAL · crudo tal como llegó",
        f"fecha de descarga: {FECHA}",
        "",
        "★ POR QUÉ NO SE BAJÓ DE CONAF DIRECTAMENTE:",
        "  · sit.conaf.cl responde una página pero NO expone servicios de datos",
        "    (/arcgis, /server, /geoserver → 404). Verificado 17 y 19-ago-2026.",
        "  · www.conaf.cl/robots.txt prohíbe descargar PDF y URL con parámetros,",
        "    que es justamente el formato de sus estadísticas históricas.",
        "  · www.ide.cl/robots.txt PROHÍBE EXPLÍCITAMENTE a ClaudeBot",
        "    (User-agent: ClaudeBot / Disallow: /). NO se consultó. Pendiente",
        "    para consulta humana.",
        "",
        "  Vía usada: CIREN (Centro de Información de Recursos Naturales), que",
        "  opera el servidor geográfico de la IDE MINAGRI (Infraestructura de",
        "  Datos Espaciales del Ministerio de Agricultura), la cartera de la que",
        "  depende CONAF. esri.ciren.cl no tiene robots.txt (404) y sus",
        "  servicios ArcGIS son de acceso anónimo.",
        "",
    ]

    # ---------------- (a) ocurrencia ----------------
    print("\n(a) OCURRENCIA HISTÓRICA — un registro por incendio")
    por_cut, por_nombre, geo_idx = cargar_comunas()
    if not por_cut:
        print("    ⚠️ falta datos/capas/comunas.geojson: la comuna quedará como")
        print("       venga en cada temporada, con las grafías mezcladas.")
    todas, huecos = [], []
    acuerdo_total = desacuerdo_total = 0
    for capa, temp in CAPAS_TEMPORADA.items():
        geo, motivo = traer_capa(SRV_INCENDIOS, capa)
        if geo is None:
            print(f"    ✗ temporada {temp}: SIN DATO — {motivo}")
            huecos.append((f"incendios {temp}", motivo))
            proc.append(f"  · incendios {temp}: SIN DATO — {motivo}")
            continue
        n = len(geo["features"])
        dest = CRUDO / f"incendios_{temp}.geojson.gz"
        with gzip.open(dest, "wt", encoding="utf8") as fh:
            json.dump(geo, fh, ensure_ascii=False)
        filas, ac, des = normalizar_incendios(geo, temp, por_cut, por_nombre,
                                              geo_idx)
        acuerdo_total += ac
        desacuerdo_total += des
        todas.extend(filas)
        sinres = sum(1 for f in filas if f["comuna_resuelta_por"] == "sin resolver")
        marca = f"  ({sinres} sin comuna resuelta)" if sinres else ""
        print(f"    ✓ {temp}: {n:,} incendios{marca}")
        proc.append(f"  · incendios {temp}: {n} registros · "
                    f"{SRV_INCENDIOS}/{capa}")
        time.sleep(PAUSA_S)

    if todas:
        d1 = SALIDA / "incendios_ocurrencia.csv"
        with open(d1, "w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(todas[0].keys()))
            w.writeheader()
            w.writerows(todas)

        # resumen comuna × temporada: la tabla que la Matriz consume
        agg = defaultdict(lambda: dict(n=0, ha=0.0, ha_sin_dato=0))
        for f in todas:
            k = (f["region"], f["comuna"], f["temporada"])
            agg[k]["n"] += 1
            if f["ha_total"] is None:
                agg[k]["ha_sin_dato"] += 1
            else:
                agg[k]["ha"] += f["ha_total"]
        d2 = SALIDA / "incendios_por_comuna_temporada.csv"
        with open(d2, "w", newline="", encoding="utf8") as fh:
            w = csv.writer(fh)
            w.writerow(["region", "comuna", "temporada", "n_incendios",
                        "ha_afectadas", "n_sin_superficie_declarada"])
            for (r, c, t), v in sorted(agg.items()):
                w.writerow([r, c, t, v["n"], round(v["ha"], 2),
                            v["ha_sin_dato"]])

        sin_c = sum(1 for f in todas if not f["tiene_coordenada"])
        print(f"\n    total incendios          : {len(todas):,}")
        print(f"    ★ con coordenada         : {len(todas)-sin_c:,}")
        print(f"    sin coordenada (hueco)   : {sin_c:,}")
        print(f"    comunas distintas        : "
              f"{len({f['comuna'] for f in todas if f['comuna']}):,}")
        ha = sum(f["ha_total"] for f in todas if f["ha_total"] is not None)
        print(f"    hectáreas acumuladas     : {ha:,.0f} ha en 15 temporadas")

        print("\n    cómo se resolvió la comuna de cada incendio:")
        pv = defaultdict(int)
        for f in todas:
            pv[f["comuna_resuelta_por"]] += 1
        for k, v in sorted(pv.items(), key=lambda x: -x[1]):
            print(f"       {v:7,d}  {k}")

        ct = acuerdo_total + desacuerdo_total
        if ct:
            print("\n    control de calidad — el código CUT contra la geometría:")
            print(f"       coinciden      : {acuerdo_total:7,d}  "
                  f"({100*acuerdo_total/ct:5.2f} %)")
            print(f"       NO coinciden   : {desacuerdo_total:7,d}  "
                  f"({100*desacuerdo_total/ct:5.2f} %)")
            print("       (se comparó en los incendios donde funcionan las dos")
            print("        vías. El desacuerdo es esperable en incendios que")
            print("        empiezan cerca del límite comunal: la coordenada es")
            print("        el punto de inicio, no el área quemada.)")

        print("\n    las 12 comunas con más incendios en 15 temporadas:")
        porcom = defaultdict(int)
        porcom_ha = defaultdict(float)
        for f in todas:
            porcom[(f["comuna"], f["region"])] += 1
            if f["ha_total"]:
                porcom_ha[(f["comuna"], f["region"])] += f["ha_total"]
        for (c, r), n in sorted(porcom.items(), key=lambda x: -x[1])[:12]:
            print(f"       {n:6,d} incendios  {porcom_ha[(c,r)]:11,.0f} ha   "
                  f"{c[:24]:26s} {r[:20]}")

        print("\n    las 12 comunas con más hectáreas quemadas:")
        for (c, r), h in sorted(porcom_ha.items(), key=lambda x: -x[1])[:12]:
            print(f"       {h:11,.0f} ha  {porcom[(c,r)]:6,d} incendios   "
                  f"{c[:24]:26s} {r[:20]}")

        print("\n    incendios y hectáreas por temporada:")
        pt_n, pt_ha = defaultdict(int), defaultdict(float)
        for f in todas:
            pt_n[f["temporada"]] += 1
            if f["ha_total"]:
                pt_ha[f["temporada"]] += f["ha_total"]
        for t in sorted(pt_n):
            print(f"       {t}   {pt_n[t]:6,d} incendios   {pt_ha[t]:11,.0f} ha")

        print("\n    causa general (homologada: sin numeración, sin acentos,")
        print("     en mayúsculas — el texto crudo va igual en el CSV):")
        pc = defaultdict(int)
        for f in todas:
            pc[f["causa_general_normalizada"] or "(sin dato)"] += 1
        tot = len(todas)
        for k, v in sorted(pc.items(), key=lambda x: -x[1])[:10]:
            print(f"       {v:6,d}  ({100*v/tot:4.1f} %)  {k[:56]}")

    # ---------------- (b) riesgo ARCLIM ----------------
    print("\n\n(b) ZONIFICACIÓN DE RIESGO")
    filas_r, meta_r, motivo = traer_arclim()
    if filas_r is None:
        print(f"    ✗ SIN DATO — {motivo}")
        huecos.append(("riesgo ARCLIM", motivo))
        proc.append(f"  · riesgo ARCLIM: SIN DATO — {motivo}")
    else:
        (CRUDO / "riesgo_arclim_comunal.json").write_text(
            json.dumps(filas_r, ensure_ascii=False), encoding="utf8")
        d3 = SALIDA / "incendios_riesgo_arclim_comuna.csv"
        with open(d3, "w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas_r[0].keys()))
            w.writeheader()
            w.writerows(filas_r)
        print(f"    ✓ {len(filas_r)} comunas con índice de riesgo ARCLIM")
        print("    ⚠️ NO es zonificación de CONAF. Es ARCLIM (Atlas de Riesgo")
        print("       Climático, Ministerio del Medio Ambiente + CR2),")
        print("       republicado por SERNAGEOMIN. Cubre 215 de 345 comunas:")
        print("       una comuna ausente NO es una comuna sin riesgo.")
        proc.append(f"  · riesgo ARCLIM comunal: {len(filas_r)} comunas · "
                    f"{SRV_ARCLIM}")
        proc.append("      ⚠️ autoría ARCLIM/CR2, NO CONAF. Cobertura parcial.")

        cl = "bosques_incendios_bnativos_riesgo_pres"
        cf = "bosques_incendios_bnativos_riesgo_fut"
        if filas_r and cl in filas_r[0]:
            hoy = [(f.get("COMUNA") or f.get("NOM_COMUNA"), f[cl], f.get(cf))
                   for f in filas_r if f.get(cl) is not None]
            print("\n    las 10 comunas de mayor riesgo PRESENTE (bosque nativo):")
            for c, a, b in sorted(hoy, key=lambda x: -x[1])[:10]:
                dl = f"{b:.3f}" if b is not None else "s/d"
                print(f"       {a:.3f} hoy → {dl} futuro   {str(c)[:34]}")

    # ---------------- (c) combustible ----------------
    print("\n\n(c) COMBUSTIBLE — Catastro de Recursos Vegetacionales de CONAF")
    print("    (suma de hectáreas por comuna y uso de la tierra, pedida al")
    print("     servicio; no se bajan los polígonos)")
    comb = []
    for capa, (reg, anio) in CAPAS_VEGETACION.items():
        filas_v, detalle, motivo = combustible_de_region(capa)
        if filas_v is None:
            print(f"    ✗ {reg}: SIN DATO — {motivo}")
            huecos.append((f"vegetación {reg}", motivo))
            proc.append(f"  · catastro vegetacional {reg}: SIN DATO — {motivo}")
            continue
        for f in filas_v:
            comb.append(dict(
                region=reg, anio_catastro=anio,
                comuna=_txt(f.get("nom_com")),
                uso_tierra=_txt(f.get("uso")),
                subuso=_txt(f.get("subuso")) if "subuso" in f else "(no pedido)",
                hectareas=round(f.get("ha") or 0, 2),
                n_poligonos=f.get("n_poligonos"),
                granularidad=detalle))
        aviso = "" if "sin subuso" not in detalle else "  ⚠️ sin subuso"
        print(f"    ✓ {reg} (catastro {anio}): {len(filas_v)} combinaciones{aviso}")
        proc.append(f"  · catastro vegetacional {reg} (CONAF {anio}): "
                    f"{len(filas_v)} grupos [{detalle}] · {SRV_VEGETACION}/{capa}")
        time.sleep(PAUSA_S)

    if comb:
        d4 = SALIDA / "incendios_combustible_comuna.csv"
        with open(d4, "w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(comb[0].keys()))
            w.writeheader()
            w.writerows(comb)
        print(f"\n    filas de combustible      : {len(comb):,}")
        print(f"    comunas con catastro      : "
              f"{len({c['comuna'] for c in comb if c['comuna']}):,}")
        print("\n    hectáreas nacionales por uso de la tierra:")
        pu = defaultdict(float)
        for c in comb:
            pu[c["uso_tierra"] or "(sin dato)"] += c["hectareas"]
        for k, v in sorted(pu.items(), key=lambda x: -x[1]):
            print(f"       {v:14,.0f} ha   {k[:50]}")
        print("\n    ⚠️ los años de levantamiento van de 2011 a 2019 según la")
        print("       región. NO es una foto de un mismo año: comparar dos")
        print("       regiones entre sí compara dos fechas distintas.")

    (CRUDO / "PROCEDENCIA.txt").write_text("\n".join(proc) + "\n",
                                           encoding="utf8")
    print(f"\n\n  crudo guardado en : {CRUDO}")
    if huecos:
        print("\n  huecos declarados:")
        for c, m in huecos:
            print(f"     {c}: {m}")


if __name__ == "__main__":
    main()
