"""
ADAPTADOR · DEPÓSITOS Y TRANQUES DE RELAVES (SERNAGEOMIN)
=========================================================

QUÉ ES UN RELAVE Y POR QUÉ ESTÁ EN ESTE PROYECTO
-------------------------------------------------
Un *relave* es el desecho que queda cuando una planta minera muele la roca y le
saca el metal: un lodo de arena fina, agua y reactivos. Se acumula detrás de un
muro, muchas veces construido con el propio material del relave. Ese conjunto
—muro más laguna de lodo— es un **depósito de relaves**.

Para esta Matriz un depósito de relaves es **dos cosas a la vez**:

1. **Infraestructura crítica.** Es la instalación que le permite operar a la
   faena minera. Si falla, para la faena.
2. **Amenaza.** Si el muro cede, el lodo baja por la quebrada. No es agua: es un
   fluido denso que arrasa. Precedentes: **El Cobre (Chile, 1965)** — el sismo
   de La Ligua licuó el muro y el lodo sepultó el pueblo de El Cobre, más de 200
   muertos; **Brumadinho (Brasil, 2019)** — 270 muertos.

Esa doble condición es exactamente lo que el consolidador tiene que poder
representar: el mismo objeto entra por la columna de infraestructura y por la
columna de amenaza.

DE DÓNDE SALE EL DATO — Y POR QUÉ NO DEL SERVIDOR PROPIO
---------------------------------------------------------
El servidor propio de SERNAGEOMIN (**Servicio Nacional de Geología y Minería**),
en el rango de red 190.98.205.0/24, incluye `portalgeo.sernageomin.cl` y está
**caído a nivel de red desde el 17-ago-2026** (verificado de nuevo el 19-ago:
la conexión ni siquiera se establece, código 000).

Pero SERNAGEOMIN también publica en la nube de ArcGIS Online, en la misma
organización `OyjvVdFTl5hfSdX3` de donde este proyecto ya baja la Minuta Técnica
de remoción en masa. Ahí sí responde. Se usa esa vía, y queda anotado que es la
vía de respaldo, no el canal primario.

De los ocho servicios con «relave» en el nombre que publica esa organización, se
eligieron dos, y se bajan LOS DOS a propósito:

  · `PLATAFORMA_PUBLICA_RELAVES_WFL1`, capa 24 `C_RELAVES` — **la fuente
    principal**. Es la Plataforma Pública de Relaves del *Departamento de
    Depósitos de Relaves* de SERNAGEOMIN, el que administra el Decreto Supremo
    N° 248 (el reglamento de depósitos de relaves). Trae latitud, longitud,
    estado, volumen autorizado y actual, tonelaje y método constructivo.
  · `Relaves`, capa 0 — **el contraste independiente**. Viene del Atlas Minero.
    Se baja para poder comparar los dos catastros y declarar en qué difieren, en
    vez de elegir uno y creerle.

Complementos que se bajan porque describen la misma amenaza:
  · `ACCESO_RELAVEDUCTOS` — los *relaveductos*, las tuberías que llevan el lodo
    desde la planta hasta el depósito. Una rotura de relaveducto es un modo de
    falla distinto del colapso del muro, y afecta a lo que cruza la tubería.
  · `ACCESO_RELAVES_ACTIVOS` y `ACCESO_RELAVES_ABANDONADOS` — recortes del Atlas
    Minero 2023 usados para el Fondo de Comunas Mineras. Sirven de tercer
    contraste sobre el estado declarado.

LO QUE ESTE ADAPTADOR **NO** HACE
----------------------------------
No calcula el alcance de una rotura. El dato no trae ni la altura del muro ni la
pendiente aguas abajo, que es lo que gobierna hasta dónde llega el lodo. Todo
lo que se puede afirmar con este dato es *dónde está cada depósito y cuánto
material contiene*. El cruce con infraestructura vive en `cruzar_relaves.py` y
declara ahí su propia aproximación.

REGLA DE ESCRITURA DEL PROYECTO
--------------------------------
Ninguna sigla sin su nombre completo la primera vez. Aquí: SERNAGEOMIN =
Servicio Nacional de Geología y Minería; UTM = Universal Transverse Mercator
(el sistema de coordenadas planas en metros que usa la minería chilena);
WGS84 = World Geodetic System 1984 (el sistema de latitud/longitud del GPS).
"""

import csv
import gzip
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

from pyproj import Geod
from shapely.geometry import Point, shape
from shapely.ops import nearest_points
from shapely.strtree import STRtree

AQUI = Path(__file__).resolve().parent.parent
FECHA = date.today().isoformat()
CRUDO = AQUI / "datos" / "crudo" / "relaves" / FECHA
SALIDA = AQUI / "datos"

ORG = "https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/services"

# nombre corto → (servicio, capa, para qué sirve)
CAPAS = {
    "plataforma_publica": (
        "PLATAFORMA_PUBLICA_RELAVES_WFL1", 24,
        "FUENTE PRINCIPAL · Plataforma Pública de Relaves, Departamento de "
        "Depósitos de Relaves de SERNAGEOMIN (Decreto Supremo N° 248)"),
    "atlas_minero": (
        "Relaves", 0,
        "CONTRASTE · catastro del Atlas Minero de SERNAGEOMIN"),
    "relaveductos": (
        "ACCESO_RELAVEDUCTOS", 1,
        "tuberías de transporte de relave (Atlas Minero 2023)"),
    "acceso_activos": (
        "ACCESO_RELAVES_ACTIVOS", 1,
        "recorte de activos usado para el Fondo de Comunas Mineras 2024"),
    "acceso_abandonados": (
        "ACCESO_RELAVES_ABANDONADOS", 1,
        "recorte de abandonados usado para el Fondo de Comunas Mineras 2024"),
}

PAGINA = 1000        # el servicio pagina; se pide de a mil
PAUSA_S = 1.5        # ritmo prudente entre peticiones al servicio público


def traer_capa(servicio, capa, timeout=120):
    """Baja una capa completa como GeoJSON, paginando.

    Devuelve (geojson, None) o (None, motivo). Nunca lanza: un fallo de red se
    convierte en un hueco declarado, no en una corrida caída.
    """
    rasgos, offset = [], 0
    while True:
        url = (f"{ORG}/{servicio}/FeatureServer/{capa}/query"
               f"?where=1%3D1&outFields=*&outSR=4326&f=geojson"
               f"&resultOffset={offset}&resultRecordCount={PAGINA}")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                datos = json.load(r)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError, OSError) as e:
            return None, f"{type(e).__name__}: {str(e)[:140]}"
        if "error" in datos:
            return None, f"el servicio devolvió error: {datos['error']}"
        lote = datos.get("features", [])
        rasgos.extend(lote)
        if len(lote) < PAGINA:
            break
        offset += PAGINA
        time.sleep(PAUSA_S)
    return {"type": "FeatureCollection", "features": rasgos}, None


def _num(v):
    """Número o None. Nunca convierte un dato ausente en cero: son distintos."""
    if v in (None, "", "S/I", "s/i"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _txt(v):
    s = (str(v).strip() if v is not None else "")
    return "" if s.upper() in ("NONE", "NULL") else s


def normalizar_principal(geojson):
    """C_RELAVES → filas planas con nombres estables para el resto del proyecto.

    Dos decisiones que conviene tener a la vista:

    · **La coordenada de la geometría manda sobre los campos LATITUD/LONGITUD.**
      Los campos vienen redondeados a tres decimales (≈110 m); la geometría trae
      la precisión completa. Se guardan los dos y se anota la diferencia.
    · **Un volumen sin dato NO es un volumen cero.** Un depósito abandonado sin
      volumen declarado es un hueco de información, no un depósito vacío.
      Confundirlos sería inventar tranquilidad.
    """
    filas = []
    for f in geojson["features"]:
        p = f.get("properties", {})
        g = f.get("geometry") or {}
        lon = lat = None
        if g.get("type") == "Point" and g.get("coordinates"):
            lon, lat = g["coordinates"][0], g["coordinates"][1]
        filas.append(dict(
            id_sernageomin=_txt(p.get("ID")),
            empresa=_txt(p.get("NOMBRE_EMP")),
            faena=_txt(p.get("NOMBRE_FAE")),
            instalacion=_txt(p.get("NOMBRE_INS")),
            tipo_deposito=_txt(p.get("TIPO_INS")),
            recurso=_txt(p.get("RECURSO")),
            tipo_mineria=_txt(p.get("MINERIA")),
            region=_txt(p.get("REGION")),
            provincia=_txt(p.get("PROVINCIA")),
            comuna=_txt(p.get("COMUNA")),
            estado=_txt(p.get("ESTADO_INS")),
            metodo_constructivo=_txt(p.get("METODO_CON")),
            volumen_autorizado_m3=_num(p.get("VOL_AUT")),
            volumen_actual_m3=_num(p.get("VOL_ACTUAL")),
            tonelaje_autorizado_t=_num(p.get("TO_AUT")),
            tonelaje_actual_t=_num(p.get("TO_ACTUAL")),
            resolucion_aprueba=_txt(p.get("RES_APRUEBA")),
            resolucion_aprueba_fecha=_txt(p.get("RES_APRUEBA_FECHA")),
            resolucion_plan_cierre=_txt(p.get("RES_PDC_APRUEBA")),
            resolucion_plan_cierre_fecha=_txt(p.get("RES_PDC_FECHA")),
            utm_este=_num(p.get("UTM_ESTE")),
            utm_norte=_num(p.get("UTM_NORTE")),
            lat_campo=_num(p.get("LATITUD")),
            lon_campo=_num(p.get("LONGITUD")),
            lat=lat, lon=lon,
            tiene_coordenada=bool(lat is not None and lon is not None),
        ))
    return filas


def main():
    print("=" * 78)
    print("ADAPTADOR · DEPÓSITOS Y TRANQUES DE RELAVES · SERNAGEOMIN")
    print("=" * 78)
    CRUDO.mkdir(parents=True, exist_ok=True)

    procedencia = [
        "DEPÓSITOS Y TRANQUES DE RELAVES · crudo tal como llegó",
        f"fecha de descarga: {FECHA}",
        "",
        "organismo: SERNAGEOMIN (Servicio Nacional de Geología y Minería)",
        "vía: ArcGIS Online, organización OyjvVdFTl5hfSdX3 (acceso anónimo)",
        "",
        "★ POR QUÉ ESTA VÍA Y NO EL SERVIDOR PROPIO:",
        "  portalgeo.sernageomin.cl (rango 190.98.205.0/24) NO responde a nivel",
        "  de red desde el 17-ago-2026. Verificado de nuevo el 19-ago-2026:",
        "  código 000, la conexión ni se establece. La nube es la vía de",
        "  respaldo, no el canal primario. Reintentar el canal propio.",
        "",
        "condiciones de uso: el servicio es de acceso público anónimo y no",
        "  declara copyrightText. No se encontró un texto de licencia explícito",
        "  en el servicio. Uso académico, citando siempre a SERNAGEOMIN.",
        "",
        "capas bajadas:",
    ]

    resultados, huecos = {}, []
    for corto, (servicio, capa, para_que) in CAPAS.items():
        print(f"\n  bajando {corto} ({servicio}/{capa}) …")
        geo, motivo = traer_capa(servicio, capa)
        if geo is None:
            print(f"    ✗ SIN DATO — {motivo}")
            huecos.append((corto, motivo))
            procedencia.append(f"  · {corto}: SIN DATO — {motivo}")
            continue
        n = len(geo["features"])
        destino = CRUDO / f"{corto}.geojson"
        destino.write_text(json.dumps(geo, ensure_ascii=False), encoding="utf8")
        print(f"    ✓ {n:,} rasgos → {destino.name}")
        resultados[corto] = geo
        procedencia.append(
            f"  · {corto}: {n} rasgos · {ORG}/{servicio}/FeatureServer/{capa}")
        procedencia.append(f"      {para_que}")
        time.sleep(PAUSA_S)

    (CRUDO / "PROCEDENCIA.txt").write_text("\n".join(procedencia) + "\n",
                                           encoding="utf8")

    if "plataforma_publica" not in resultados:
        print("\n✗ SIN DATO en la fuente principal. No se escribe CSV: un CSV "
              "vacío se confunde con «no hay relaves». Se declara el hueco.")
        return

    filas = normalizar_principal(resultados["plataforma_publica"])
    csv_dest = SALIDA / "relaves_depositos.csv"
    with open(csv_dest, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    # ---------------- lecturas honestas del catastro ----------------
    print("\n" + "=" * 78)
    print("QUÉ TRAJO EL CATASTRO PRINCIPAL")
    print("=" * 78)
    con_coord = [f for f in filas if f["tiene_coordenada"]]
    print(f"\n  depósitos en el catastro        : {len(filas):,}")
    print(f"  ★ con coordenada utilizable     : {len(con_coord):,}")
    print(f"  sin coordenada (hueco declarado): {len(filas) - len(con_coord):,}")

    def conteo(campo, top=None):
        c = {}
        for f in filas:
            k = f[campo] or "(sin dato)"
            c[k] = c.get(k, 0) + 1
        items = sorted(c.items(), key=lambda x: -x[1])
        return items[:top] if top else items

    print("\n  estado declarado de la instalación:")
    for k, v in conteo("estado"):
        print(f"     {k[:44]:46s} {v:5d}")

    print("\n  tipo de depósito:")
    for k, v in conteo("tipo_deposito", 12):
        print(f"     {k[:44]:46s} {v:5d}")

    print("\n  método constructivo del muro:")
    print("     (importa: el método «aguas arriba» es el que licuó en El Cobre")
    print("      1965 y en Brumadinho 2019 — el muro se apoya sobre el propio")
    print("      relave, que es lo que se licua con el sismo)")
    for k, v in conteo("metodo_constructivo", 12):
        print(f"     {k[:44]:46s} {v:5d}")

    print("\n  depósitos por región (los 10 primeros):")
    for k, v in conteo("region", 10):
        print(f"     {k[:44]:46s} {v:5d}")

    vols = [f["volumen_actual_m3"] for f in filas
            if f["volumen_actual_m3"] is not None]
    sin_vol = sum(1 for f in filas if f["volumen_actual_m3"] is None)
    en_cero = sum(1 for f in vols if f == 0)
    print(f"\n  volumen actual declarado        : {len(vols):,} depósitos")
    print(f"    de ellos, declarados en cero  : {en_cero:,}")
    print(f"  SIN volumen declarado           : {sin_vol:,}  ← hueco, NO es cero")
    if vols:
        util = sorted(v for v in vols if v > 0)
        print(f"  volumen total declarado         : {sum(util):,.0f} m³")
        if util:
            print(f"  mediana de los que declaran >0  : {util[len(util)//2]:,.0f} m³")

    # contraste con el otro catastro
    if "atlas_minero" in resultados:
        otro = resultados["atlas_minero"]["features"]
        print("\n" + "=" * 78)
        print("CONTRASTE ENTRE LOS DOS CATASTROS")
        print("=" * 78)
        print(f"\n  Plataforma Pública de Relaves : {len(filas):,} depósitos")
        print(f"  Atlas Minero                  : {len(otro):,} depósitos")
        print(f"  diferencia                    : {len(filas) - len(otro):+,}")
        print("\n  El Atlas Minero no publica un identificador comparable con el")
        print("  ID de la Plataforma, así que NO se puede afirmar cuáles son los")
        print("  que sobran ni cuáles faltan. Queda declarado como pendiente, no")
        print("  resuelto por parecido de nombres.")

    print(f"\n  crudo guardado en : {CRUDO}")
    print(f"  CSV escrito en    : {csv_dest}")
    if huecos:
        print("\n  huecos declarados:")
        for c, m in huecos:
            print(f"     {c}: {m}")


# ==========================================================================
# CRUCE · QUÉ INFRAESTRUCTURA ESTÁ AGUAS ABAJO DE UN DEPÓSITO
# ==========================================================================
"""
★★ LEER ESTO ANTES DE USAR EL RESULTADO — QUÉ ES Y QUÉ NO ES

**Esto NO es un análisis de inundación por rotura de muro.** Un análisis de
rotura de verdad resuelve la ecuación de un fluido denso bajando por un valle:
necesita la altura del muro, la curva de vaciado, la reología del lodo y el
terreno del cauce metro a metro. Nada de eso está en el catastro público.

Lo que sí se puede hacer con el dato que hay, y es lo que se hace, son **dos
filtros encadenados**, cada uno con su propia debilidad declarada:

**Filtro 1 · distancia.** ¿Qué infraestructura está a menos de 1, 5 y 10 km del
depósito? Los tres cortes no son inventados: son el orden de magnitud del
alcance documentado de las roturas de referencia. El lodo de **Brumadinho**
(Brasil, 2019) recorrió del orden de 10 km valle abajo; el de **El Cobre**
(Chile, 1965) llegó a distancia comparable. Un corte a 10 km no dice «hasta acá
llega»; dice «más allá de acá, ninguna rotura conocida ha llegado».

*Debilidad*: la distancia es en línea recta y el lodo no va en línea recta.

**Filtro 2 · cota.** De lo que está cerca, ¿qué está **más abajo** que el
depósito? El lodo baja; lo que está por encima de la cota del depósito no lo
alcanza, por cerca que esté. La cota se lee del **Copernicus DEM GLO-30**, el
modelo de elevación de 30 m de la Agencia Espacial Europea, con el mismo lector
que ya usa `adaptadores/terreno.py` — no se agrega ninguna fuente nueva.

*Debilidad*: estar más abajo **no** es estar en la misma quebrada. Un puente
100 m más bajo pero en la cuenca de al lado sale marcado y no corresponde. Esto
sobrestima, y lo hace a propósito: en este filtro un falso positivo cuesta una
revisión y un falso negativo cuesta un puente.

**Cómo se debe leer el resultado, entonces:** como una **lista corta de sitios
que merecen que alguien los mire**, no como una cifra de exposición. Es
exactamente lo que le sirve a un COGRID (Comité para la Gestión del Riesgo de
Desastres) para priorizar dónde pedir un estudio de rotura de verdad.

Lo que hace falta para convertir esto en análisis: la red de cauces vectorial y
un trazado de escurrimiento sobre el modelo de elevación. Queda como pendiente
declarado, no como algo que este archivo resolvió.
"""

CRUDO_MOP = AQUI / "datos" / "crudo" / "mop" / "2026-08-17"
BANDAS_KM = (1.0, 5.0, 10.0)      # ver la explicación de arriba
GEOD = Geod(ellps="WGS84")


def _grados_por_km(lat):
    """Cuántos grados de latitud y de longitud son un kilómetro en esa latitud.

    Sirve sólo para armar la caja de búsqueda del índice espacial: el filtro
    fino usa después la distancia geodésica exacta. La caja se agranda a
    propósito; una caja generosa no se equivoca, sólo revisa de más.
    """
    dlat = 1.0 / 110.574
    dlon = 1.0 / max(0.1, 111.320 * math.cos(math.radians(lat)))
    return dlat, dlon


def _dist_m(lon1, lat1, lon2, lat2):
    return GEOD.inv(lon1, lat1, lon2, lat2)[2]


def _cargar_dem():
    """Presta el lector de Copernicus GLO-30 que ya tiene `terreno.py`.

    Si no se puede importar, el cruce sigue sin el filtro de cota y lo dice.
    No se inventa una elevación ni se cae la corrida.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import terreno
        dem = terreno.Mosaico(
            terreno.FUENTE_DEM["url"],
            "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif",
            1, AQUI / "datos" / "crudo" / "terreno" / "dem")
        return dem, terreno
    except Exception as e:
        print(f"    ⚠️ sin filtro de cota: no se pudo cargar el modelo de "
              f"elevación ({type(e).__name__}: {str(e)[:80]})")
        return None, None


def _cota(dem, terreno, lat, lon, memo):
    """Elevación en metros, o None si no hay tesela o el pixel viene vacío."""
    if dem is None:
        return None
    clave = (round(lat, 4), round(lon, 4))
    if clave in memo:
        return memo[clave]
    z = None
    try:
        paso = dem.paso_en(lat, lon)
        if paso is not None:
            plon, plat = paso
            v = dem.ventana(lat, lon, 0, 0, plat, plon)
            import numpy as np
            if np.isfinite(v[0, 0]):
                z = float(v[0, 0])
    except Exception:
        z = None
    memo[clave] = z
    return z


def cruzar():
    print("=" * 78)
    print("CRUCE · DEPÓSITOS DE RELAVES × INFRAESTRUCTURA CRÍTICA")
    print("=" * 78)
    print("\n  ⚠️ NO es análisis de rotura de muro. Son dos filtros: distancia")
    print("     recta y cota más baja. Ver el comentario del archivo.")

    # --- depósitos ---
    filas = list(csv.DictReader(open(SALIDA / "relaves_depositos.csv",
                                     encoding="utf8")))
    dep = [f for f in filas if f["lat"] and f["lon"]]
    for d in dep:
        d["lat"], d["lon"] = float(d["lat"]), float(d["lon"])
    print(f"\n  depósitos con coordenada : {len(dep):,} de {len(filas):,}")

    dem, terreno = _cargar_dem()
    memo = {}

    # --- infraestructura puntual: puentes y subestaciones ---
    puentes = json.loads((CRUDO_MOP / "puentes.geojson").read_text(
        encoding="utf8"))["features"]
    puntos, meta_p = [], []
    for f in puentes:
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        a = f["properties"]
        puntos.append(Point(g["coordinates"][:2]))
        meta_p.append(dict(tipo="puente",
                           nombre=a.get("NOMBRE_PUENTE", ""),
                           codigo=a.get("CODIGO_PUENTE", ""),
                           rol=a.get("ROL", ""),
                           region=a.get("REGION", "")))
    n_puentes = len(puntos)
    for r in csv.DictReader(open(SALIDA / "subestaciones_puntos.csv",
                                 encoding="utf8")):
        if not r["lat"] or not r["lon"]:
            continue
        puntos.append(Point(float(r["lon"]), float(r["lat"])))
        meta_p.append(dict(tipo="subestacion", nombre=r["subestacion"],
                           codigo="", rol="", region=r["region"]))
    print(f"  puentes                  : {n_puentes:,}")
    print(f"  subestaciones            : {len(puntos) - n_puentes:,}")

    arbol_p = STRtree(puntos)

    # --- red vial ---
    with gzip.open(CRUDO_MOP / "tramos.geojson.gz", "rt", encoding="utf8") as fh:
        tramos_raw = json.load(fh)["features"]
    lineas, meta_t = [], []
    for f in tramos_raw:
        try:
            lineas.append(shape(f["geometry"]))
        except Exception:
            continue
        a = f["properties"]
        meta_t.append(dict(rol=a.get("ROL_LABEL") or a.get("ROL") or "",
                           nombre=a.get("NOMBRE_CAMINO", ""),
                           clasificacion=a.get("CLASIFICACION", ""),
                           region=a.get("REGION", "")))
    arbol_t = STRtree(lineas)
    print(f"  tramos viales            : {len(lineas):,}")

    maxb = max(BANDAS_KM)
    expuestos, sin_cota = [], 0
    resumen = defaultdict(lambda: defaultdict(int))

    print(f"\n  buscando en bandas de {', '.join(str(b) for b in BANDAS_KM)} km …")
    for i, d in enumerate(dep, 1):
        if i % 200 == 0:
            print(f"     {i:,} de {len(dep):,} depósitos")
        dlat, dlon = _grados_por_km(d["lat"])
        caja = Point(d["lon"], d["lat"]).buffer(0)  # placeholder, se usa envelope
        from shapely.geometry import box
        caja = box(d["lon"] - maxb * dlon, d["lat"] - maxb * dlat,
                   d["lon"] + maxb * dlon, d["lat"] + maxb * dlat)
        z_dep = _cota(dem, terreno, d["lat"], d["lon"], memo)

        hallados = []
        for k in arbol_p.query(caja):
            p = puntos[k]
            m = _dist_m(d["lon"], d["lat"], p.x, p.y)
            if m > maxb * 1000:
                continue
            hallados.append((meta_p[k], p.y, p.x, m))
        for k in arbol_t.query(caja):
            linea = lineas[k]
            if not linea.intersects(caja):
                continue
            a, b = nearest_points(Point(d["lon"], d["lat"]), linea)
            m = _dist_m(d["lon"], d["lat"], b.x, b.y)
            if m > maxb * 1000:
                continue
            mt = dict(meta_t[k]); mt["tipo"] = "tramo_vial"
            mt["nombre"] = mt["nombre"] or ""
            hallados.append((mt, b.y, b.x, m))

        for m_info, la, lo, m in hallados:
            z_obj = _cota(dem, terreno, la, lo, memo)
            if z_dep is None or z_obj is None:
                aguas_abajo = ""      # sin dato: NO se supone que sí ni que no
                sin_cota += 1
                desnivel = ""
            else:
                desnivel = round(z_dep - z_obj, 1)
                aguas_abajo = "si" if desnivel > 0 else "no"
            banda = next(b for b in BANDAS_KM if m <= b * 1000)
            expuestos.append(dict(
                relave_id=d["id_sernageomin"], relave_nombre=d["instalacion"],
                relave_empresa=d["empresa"], relave_estado=d["estado"],
                relave_metodo=d["metodo_constructivo"],
                relave_tipo=d["tipo_deposito"],
                relave_volumen_m3=d["volumen_actual_m3"],
                relave_region=d["region"], relave_comuna=d["comuna"],
                relave_lat=d["lat"], relave_lon=d["lon"],
                relave_cota_m=z_dep if z_dep is not None else "",
                infra_tipo=m_info["tipo"], infra_nombre=m_info.get("nombre", ""),
                infra_codigo=m_info.get("codigo", "") or m_info.get("rol", ""),
                infra_region=m_info.get("region", ""),
                infra_lat=round(la, 6), infra_lon=round(lo, 6),
                infra_cota_m=z_obj if z_obj is not None else "",
                distancia_m=round(m, 1), banda_km=banda,
                desnivel_m=desnivel, aguas_abajo=aguas_abajo))
            if aguas_abajo == "si":
                resumen[m_info["tipo"]][banda] += 1

    dest = SALIDA / "relaves_infraestructura_aguas_abajo.csv"
    with open(dest, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(expuestos[0].keys()))
        w.writeheader()
        w.writerows(expuestos)

    # ---------------- lecturas ----------------
    print("\n" + "=" * 78)
    print("RESULTADO")
    print("=" * 78)
    abajo = [e for e in expuestos if e["aguas_abajo"] == "si"]
    arriba = [e for e in expuestos if e["aguas_abajo"] == "no"]
    print(f"\n  pares (depósito, infraestructura) a menos de {maxb:.0f} km : "
          f"{len(expuestos):,}")
    print(f"    de ellos, MÁS ABAJO que el depósito  : {len(abajo):,}")
    print(f"    más ARRIBA (el lodo no sube)         : {len(arriba):,}")
    print(f"    sin cota en el modelo de elevación   : {sin_cota:,}  ← sin dato")

    print("\n  infraestructura MÁS ABAJO que un depósito, por banda:")
    print(f"     {'':16s} {'≤1 km':>9s} {'≤5 km':>9s} {'≤10 km':>9s}")
    for t in ("puente", "tramo_vial", "subestacion"):
        fila = [resumen[t].get(b, 0) for b in BANDAS_KM]
        print(f"     {t:16s} {fila[0]:9,d} {fila[1]:9,d} {fila[2]:9,d}")

    unicos = {}
    for e in abajo:
        k = (e["infra_tipo"], e["infra_nombre"], e["infra_lat"], e["infra_lon"])
        if k not in unicos or e["distancia_m"] < unicos[k]["distancia_m"]:
            unicos[k] = e
    print(f"\n  ★ objetos de infraestructura DISTINTOS aguas abajo de al menos")
    print(f"    un depósito, a ≤10 km : {len(unicos):,}")
    pt = defaultdict(int)
    for e in unicos.values():
        pt[e["infra_tipo"]] += 1
    for t, n in sorted(pt.items(), key=lambda x: -x[1]):
        print(f"       {n:6,d}  {t}")

    cerca = [e for e in unicos.values() if e["distancia_m"] <= 1000]
    print(f"\n  ★★ y a ≤1 km (donde una rotura no da tiempo a nada): "
          f"{len(cerca):,}")
    pt1 = defaultdict(int)
    for e in cerca:
        pt1[e["infra_tipo"]] += 1
    for t, n in sorted(pt1.items(), key=lambda x: -x[1]):
        print(f"       {n:6,d}  {t}")

    print("\n  los depósitos con MÁS infraestructura aguas abajo a ≤5 km:")
    pd_ = defaultdict(set)
    vol = {}
    for e in abajo:
        if e["distancia_m"] <= 5000:
            pd_[e["relave_id"]].add((e["infra_tipo"], e["infra_nombre"],
                                     e["infra_lat"]))
            vol[e["relave_id"]] = e
    for rid, s in sorted(pd_.items(), key=lambda x: -len(x[1]))[:12]:
        e = vol[rid]
        v = e["relave_volumen_m3"]
        v = f"{float(v):,.0f} m³" if v not in ("", None) else "sin volumen decl."
        print(f"     {len(s):4d} objetos   {e['relave_nombre'][:26]:28s} "
              f"{e['relave_comuna'][:14]:16s} {e['relave_estado'][:11]:12s} {v}")

    print("\n  ★ el subconjunto que más importa: depósitos de método AGUAS")
    print("    ARRIBA (el que falló en El Cobre y en Brumadinho) con")
    print("    infraestructura aguas abajo a ≤1 km:")
    crit = defaultdict(set)
    infd = {}
    for e in abajo:
        if e["distancia_m"] <= 1000 and e["relave_metodo"] == "AGUAS ARRIBA":
            crit[e["relave_id"]].add((e["infra_tipo"], e["infra_nombre"],
                                      e["infra_lat"]))
            infd[e["relave_id"]] = e
    print(f"      depósitos en esa condición : {len(crit):,}")
    print(f"      objetos aguas abajo de ellos: "
          f"{sum(len(s) for s in crit.values()):,}")
    for rid, s in sorted(crit.items(), key=lambda x: -len(x[1]))[:10]:
        e = infd[rid]
        print(f"        {len(s):3d} obj.  {e['relave_nombre'][:24]:26s} "
              f"{e['relave_comuna'][:14]:16s} {e['relave_estado'][:11]:12s} "
              f"{e['relave_region'][:12]}")

    print("\n  reparto por región de la infraestructura aguas abajo (≤10 km):")
    pr = defaultdict(int)
    for e in unicos.values():
        pr[e["relave_region"] or "(sin región)"] += 1
    for r, n in sorted(pr.items(), key=lambda x: -x[1])[:10]:
        print(f"       {n:6,d}  {r}")

    print(f"\n  CSV escrito en : {dest}")
    print("\n  RECORDATORIO: «aguas abajo» acá quiere decir «a menor cota y a")
    print("  menos de 10 km», NO «en la trayectoria del lodo». Falta la red de")
    print("  cauces para saber lo segundo. Queda declarado como pendiente.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cruzar":
        cruzar()
    else:
        main()
