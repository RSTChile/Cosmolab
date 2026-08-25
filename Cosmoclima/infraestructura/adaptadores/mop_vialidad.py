"""
Adaptador MOP / Dirección de Vialidad — la red vial de Chile, sus puentes y sus
emergencias, bajados del servicio ArcGIS público del Ministerio de Obras Públicas.

POR QUÉ ESTA FUENTE
-------------------
El piloto del proyecto fueron 39 subestaciones eléctricas, elegidas no por
importancia sino porque eran lo único georreferenciado que había a mano. Las
lluvias de julio de 2026 mostraron dónde está de verdad el punto débil de Chile:
la vialidad. Un camino cortado no apaga una ampolleta, deja una comuna aislada —
que es exactamente el modo de falla que el glosario oficial de GRD llama
«aislamiento» y que el proyecto persigue.

El MOP publica su inventario vial completo, abierto, sin credenciales, en
`https://rest-sit.mop.gob.cl/arcgis/rest/services/`. Nadie lo había bajado.

★ EL CAMPO QUE JUSTIFICA TODO ESTO: `CAUCE_QUEB`
------------------------------------------------
La capa de puentes trae, por cada puente, **el nombre del río o quebrada que
cruza**. Ese campo es el puente semántico (valga) entre las dos mitades del
problema:

    «el río Maipo se desbordó»  ──CAUCE_QUEB──▶  «estos 23 puentes estaban encima»

Sin él, una crecida es una noticia. Con él, es una lista de activos en riesgo.
Y permite ver algo que ninguna de las dos capas dice por separado: un cauce con
muchos puentes es un **punto de fallo múltiple** — una sola crecida se cobra
varios cortes a la vez, y los cortes no se suman, se multiplican, porque cada
uno elimina la ruta alternativa del otro.

QUÉ SE BAJA (cuatro capas, tres tablas)
---------------------------------------
1. `VIALIDAD/Puentes` capa 0 ................. 6.742 puentes (punto)
2. `VIALIDAD/Red_Vial_Chile` capa 3 ......... 14.039 tramos viales (línea)
3. `EMERGENCIA/EMERGENCIA_HISTORICA_MOP` 0 ... 1.850 emergencias 2014-2019
4. `EMERGENCIA/EMERGENCIA_MOP` capa 0 ........ 4.291 emergencias 2022-2026
                                               ──────
   3 + 4 = 6.141 emergencias sobre infraestructura MOP

Las capas 3 y 4 son la MISMA serie partida en dos registros del ministerio, y
**no se solapan en el tiempo: hay un hueco entre 2020-01 y 2022-05**. Eso no se
disimula, se declara: cualquier conteo anual tiene que saltarse esos dos años y
medio o mentirá.

CÓMO SE PAGINA (el servicio es viejo)
-------------------------------------
El servidor corre ArcGIS 10.2.1. Eso trae dos consecuencias que hay que
respetar o los datos salen truncados en silencio:

* **`resultOffset` no existe.** Se probó: el servidor lo ACEPTA sin error y lo
  IGNORA, devolviendo siempre desde el principio. Paginar con offset acá no
  falla ruidosamente, falla callado, que es peor. La paginación se hace por
  identificador: primero `returnIdsOnly=true` para pedir la lista completa de
  OBJECTID (el servidor sí la entrega entera, sin tope), y después se piden los
  registros en lotes por `objectIds=...`. Así el conteo cierra por construcción.
* **`f=geojson` no existe** (soporta `JSON` y `AMF`). El servicio entrega
  esriJSON y la conversión a GeoJSON la hace este módulo, explícitamente.

Además el lote va por **POST**: con 250 identificadores la URL de un GET supera
el largo que el servidor acepta y responde 404 — otro fallo mudo.

`maxRecordCount` del servicio es 1.000, pero al pedir por `objectIds` el límite
que manda es el del lote que uno pide. Se usan lotes conservadores.

RITMO
-----
Es un servicio público de un ministerio. Lotes chicos, una pausa entre pedidos,
reintentos con espera creciente, y `Accept-Encoding: gzip` para no pedirle al
MOP que mande cuatro veces más bytes de los necesarios.

CONDICIONES DE USO — ★ LEER ANTES DE PUBLICAR NADA
---------------------------------------------------
La Red Vial del MOP está publicada en el portal de datos del ministerio bajo
licencia **Creative Commons Atribución-NoComercial (CC BY-NC)**. NO es dominio
público. Para este proyecto —investigación, insumo a SENAPRED, sin venta— el uso
encaja, pero:

  · hay que **atribuir** a la Dirección de Vialidad / MOP en todo derivado;
  · **no** se puede incorporar a un producto comercial;
  · el `copyrightText` que declara el propio servicio es «Dirección de Vialidad,
    Ministerio de Obras Públicas»;
  · **conviene confirmarlo por escrito con la UGIT** (Unidad de Gestión de
    Información Territorial del MOP, la que administra el SIT) antes de publicar
    cualquier derivado, aunque sea gratuito. Queda anotado como pendiente.

Se guarda el crudo tal como llega —las respuestas esriJSON del servidor, sin
tocar, comprimidas— antes de procesar nada. Si mañana se descubre que este
módulo interpretó mal un campo, el dato original sigue ahí.

USO
---
    python adaptadores/mop_vialidad.py --explorar    # catálogo del servicio
    python adaptadores/mop_vialidad.py --bajar       # crudo a datos/crudo/mop/
    python adaptadores/mop_vialidad.py --procesar    # los tres CSV + análisis
    python adaptadores/mop_vialidad.py --todo
"""

import argparse
import csv
import gzip
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

AQUI = Path(__file__).parent.parent
sys.path.insert(0, str(AQUI))
import territorio  # noqa: E402

BASE = "https://rest-sit.mop.gob.cl/arcgis/rest/services"
HOY = date.today().isoformat()
CRUDO = AQUI / "datos" / "crudo" / "mop"
DATOS = AQUI / "datos"

# Chile continental. Se usa un desfase FIJO de -4 h para pasar el epoch UTC de
# ArcGIS a fecha local. En horario de verano el país corre a -3 h, así que un
# evento registrado entre las 00:00 y las 01:00 locales puede quedar asignado al
# día anterior. No cambia ningún conteo mensual salvo justo en el borde del mes,
# y se prefiere un desfase declarado a una conversión que finja exactitud.
DESFASE_CHILE = timezone(timedelta(hours=-4))

PAUSA = 1.2          # segundos entre pedidos: es un servicio público
# Seis intentos, no tres. Comprobado el 17-ago-2026: el servicio se queda mudo
# («read operation timed out») cada tantos lotes y vuelve solo al rato. Con tres
# intentos, una descarga de 57 lotes se cae antes de terminar.
REINTENTOS = 6

# ── qué se baja ──────────────────────────────────────────────────────────────
# `capa` es el índice de la capa dentro del MapServer. `lote` es cuántos
# registros se piden por vez: los tramos viales traen la geometría completa de
# la carretera (cientos de vértices cada uno) y pesan ~26 kB por registro, así
# que van en lotes chicos.
CAPAS = {
    "puentes": dict(
        servicio="VIALIDAD/Puentes", capa=0, lote=500,
        esperado=6742, geometria="punto",
        descripcion="Puentes, viaductos y pasos superiores de tuición de la "
                    "Dirección de Vialidad, nivel nacional."),
    "tramos": dict(
        servicio="VIALIDAD/Red_Vial_Chile", capa=3, lote=250,
        esperado=14039, geometria="linea",
        descripcion="Red vial nacional, capa de máximo detalle (1:1.128). Las "
                    "capas 0/1/2 del mismo servicio son la MISMA red "
                    "generalizada para escalas chicas (630 / 1.452 / 10.725 "
                    "registros): no son inventarios distintos."),
    "emergencias_historicas": dict(
        servicio="EMERGENCIA/EMERGENCIA_HISTORICA_MOP", capa=0, lote=500,
        esperado=1850, geometria="punto",
        descripcion="Emergencias sobre infraestructura MOP, registro histórico "
                    "cerrado (2014-02 a 2019-12)."),
    "emergencias_vigentes": dict(
        servicio="EMERGENCIA/EMERGENCIA_MOP", capa=0, lote=500,
        esperado=4291, geometria="punto",
        descripcion="Emergencias sobre infraestructura MOP, registro operativo "
                    "en uso (2022-06 en adelante)."),
}

FUENTE = dict(
    id="mop_sit_vialidad",
    organismo="Ministerio de Obras Públicas — Dirección de Vialidad (SIT/UGIT)",
    producto="Servicios ArcGIS REST del SIT-MOP: Puentes, Red Vial de Chile, "
             "Emergencias MOP",
    url=BASE,
    formato="esriJSON (convertido a GeoJSON por este módulo)",
    familia="ESTADO",
    acceso="anonimo",
    acceso_verificado=1,
    condiciones_uso="Red Vial MOP publicada como Creative Commons "
                    "Atribución-NoComercial (CC BY-NC). Atribución obligatoria "
                    "a la Dirección de Vialidad / MOP. Uso comercial NO "
                    "permitido. PENDIENTE: confirmar por escrito con la UGIT.",
    permite_automatizacion="si (servicio REST público, sin credenciales)",
    granularidad="activo puntual (puente / emergencia) y tramo lineal",
)


# ── acceso al servicio ───────────────────────────────────────────────────────

def _pedir(url, parametros, metodo="GET", intentos=None):
    """Un pedido al servicio, con reintentos y espera creciente.

    Devuelve el JSON ya parseado. Si el servidor contesta un error de ArcGIS
    (que viene con código 200 y un campo `error`), se levanta excepción: un
    error disfrazado de éxito es la peor forma de perder datos.
    """
    intentos = intentos or REINTENTOS
    cuerpo = urllib.parse.urlencode(parametros).encode("utf-8")
    espera = 3
    for intento in range(1, intentos + 1):
        try:
            if metodo == "POST":
                pedido = urllib.request.Request(url, data=cuerpo)
            else:
                pedido = urllib.request.Request(url + "?" + cuerpo.decode())
            pedido.add_header("Accept-Encoding", "gzip")
            pedido.add_header("User-Agent",
                              "matriz-infraestructura-critica-clima/1.0 "
                              "(investigacion; contacto via MOP UGIT)")
            with urllib.request.urlopen(pedido, timeout=300) as resp:
                bruto = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    bruto = gzip.decompress(bruto)
            datos = json.loads(bruto.decode("utf-8"))
            if isinstance(datos, dict) and "error" in datos:
                raise RuntimeError(f"ArcGIS: {datos['error']}")
            return datos
        except Exception as err:                       # noqa: BLE001
            if intento == intentos:
                raise
            print(f"      reintento {intento}/{intentos - 1} tras {espera}s "
                  f"({type(err).__name__}: {str(err)[:120]})")
            time.sleep(espera)
            espera *= 2
    raise RuntimeError("inalcanzable")


def _url_capa(clave):
    c = CAPAS[clave]
    return f"{BASE}/{c['servicio']}/MapServer/{c['capa']}"


def _pedir_registros(clave, ids, sin_dato, crudos):
    """Pide un grupo de registros. Si el servicio se cae con ese grupo, PARTE el
    grupo en dos y lo intenta de nuevo, hasta aislar el registro culpable.

    Por qué hace falta: comprobado el 17-ago-2026, hay tramos viales que el
    servidor no sabe entregar. Pedirlos devuelve `code 500 Error performing
    query operation` de forma reproducible, y da igual cuántas veces se
    reintente. Un solo registro roto tumbaba la descarga de los 14.039.

    Las dos salidas malas de este problema serían: abortar (y no tener el
    inventario) o pedir el lote entero y quedarse callado con los 250 que no
    llegaron. La bisección hace la tercera: baja **todo lo que el servicio sí
    puede dar** y deja el registro imposible anotado por identificador en
    `sin_dato`, que se guarda en el crudo. Un hueco con nombre y apellido.

    `crudos` recoge las respuestas del servidor sin tocar, para que el archivo
    de crudo siga siendo exactamente lo que el MOP contestó.

    Los reintentos se dosifican: mientras el grupo es grande se insiste poco (2
    intentos) porque conviene partirlo rápido; cuando queda un solo registro se
    insiste con todo (`REINTENTOS`), porque ahí sí importa distinguir un tropiezo
    pasajero del servicio de un registro que de verdad está roto.
    """
    try:
        resp = _pedir(f"{_url_capa(clave)}/query",
                      {"objectIds": ",".join(map(str, ids)),
                       "outFields": "*", "outSR": "4326",
                       "returnGeometry": "true", "f": "json"},
                      metodo="POST",
                      intentos=REINTENTOS if len(ids) == 1 else 2)
        crudos.append(resp)
        return resp.get("features", [])
    except Exception as err:                           # noqa: BLE001
        if len(ids) == 1:
            print(f"      ★ el servicio NO puede entregar el registro "
                  f"{ids[0]}: {str(err)[:90]} — queda como «sin dato»")
            sin_dato.append(ids[0])
            return []
        mitad = len(ids) // 2
        print(f"      el grupo de {len(ids)} falla; se parte en "
              f"{mitad} + {len(ids) - mitad} para aislar el culpable")
        time.sleep(PAUSA)
        izq = _pedir_registros(clave, ids[:mitad], sin_dato, crudos)
        time.sleep(PAUSA)
        der = _pedir_registros(clave, ids[mitad:], sin_dato, crudos)
        return izq + der


# ── 1. exploración del catálogo ──────────────────────────────────────────────

def explorar(carpetas=("VIALIDAD", "EMERGENCIA")):
    """Recorre el servicio y describe cada capa: nombre, geometría, cuántos
    registros tiene y qué campos trae.

    Se limita a las carpetas que le interesan al proyecto. Recorrer las 21 del
    servicio entero serían cientos de pedidos a un servidor ajeno, sin uso.
    """
    catalogo = {"servicio": BASE, "consultado": HOY, "carpetas": {}}
    raiz = _pedir(BASE, {"f": "json"})
    catalogo["carpetas_disponibles"] = raiz.get("folders", [])
    print(f"El servicio expone {len(raiz.get('folders', []))} carpetas: "
          f"{', '.join(raiz.get('folders', []))}")

    for carpeta in carpetas:
        time.sleep(PAUSA)
        print(f"\n── {carpeta} ──")
        lista = _pedir(f"{BASE}/{carpeta}", {"f": "json"})
        servicios = []
        for srv in lista.get("services", []):
            if srv.get("type") != "MapServer":
                continue
            time.sleep(PAUSA)
            nombre = srv["name"]
            try:
                meta = _pedir(f"{BASE}/{nombre}/MapServer", {"f": "json"})
            except Exception as err:                   # noqa: BLE001
                servicios.append({"servicio": nombre, "error": str(err)[:160]})
                print(f"  {nombre}: sin dato ({str(err)[:60]})")
                continue
            capas = []
            for capa in meta.get("layers", []):
                time.sleep(PAUSA)
                ruta = f"{BASE}/{nombre}/MapServer/{capa['id']}"
                try:
                    det = _pedir(ruta, {"f": "json"})
                    cnt = _pedir(f"{ruta}/query",
                                 {"where": "1=1", "returnCountOnly": "true",
                                  "f": "json"}).get("count")
                except Exception as err:               # noqa: BLE001
                    capas.append({"id": capa["id"], "nombre": capa["name"],
                                  "error": str(err)[:160]})
                    continue
                capas.append({
                    "id": capa["id"],
                    "nombre": capa["name"],
                    "geometria": det.get("geometryType"),
                    "registros": cnt,
                    "maxRecordCount": det.get("maxRecordCount"),
                    "campos": [{"nombre": f["name"],
                                "tipo": f["type"].replace("esriFieldType", ""),
                                "alias": f.get("alias")}
                               for f in det.get("fields", [])],
                })
                print(f"  {nombre}/{capa['id']:>2} {capa['name'][:38]:38s} "
                      f"{str(det.get('geometryType')).replace('esriGeometry',''):9s} "
                      f"{cnt if cnt is not None else 'sin dato':>7} reg  "
                      f"{len(det.get('fields', []))} campos")
            servicios.append({
                "servicio": nombre,
                "descripcion": (meta.get("serviceDescription") or "").strip(),
                "copyright": meta.get("copyrightText"),
                "capas": capas,
            })
        catalogo["carpetas"][carpeta] = servicios

    destino = CRUDO / HOY
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "catalogo_servicio.json").write_text(
        json.dumps(catalogo, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nCatálogo guardado en {destino / 'catalogo_servicio.json'}")
    return catalogo


# ── 2. descarga con paginación por identificador ─────────────────────────────

def _identificadores(clave):
    """La lista completa de OBJECTID. El servidor la entrega entera aunque
    `maxRecordCount` sea 1.000: el tope aplica a registros, no a identificadores."""
    resp = _pedir(f"{_url_capa(clave)}/query",
                  {"where": "1=1", "returnIdsOnly": "true", "f": "json"})
    ids = resp.get("objectIds") or []
    return resp.get("objectIdFieldName", "OBJECTID"), sorted(ids)


def _a_geojson(rasgos_esri, tipo_geom):
    """esriJSON → GeoJSON. Conversión explícita porque el servidor no la ofrece.

    Punto:  {x, y}          → Point
    Línea:  {paths:[[..]]}  → LineString si hay un solo trazo, MultiLineString
                              si hay varios. No se fusionan trazos: un tramo con
                              dos trazos es un tramo con dos trazos.
    Las coordenadas ya vienen en WGS84 porque se pide `outSR=4326`; el servicio
    almacena en EPSG:5360 (SIRGAS-Chile / UTM 19S) y la reproyección la hace él.
    """
    salida = []
    for r in rasgos_esri:
        g = r.get("geometry")
        geom = None
        if g:
            if tipo_geom == "punto" and g.get("x") is not None:
                geom = {"type": "Point", "coordinates": [g["x"], g["y"]]}
            elif tipo_geom == "linea" and g.get("paths"):
                trazos = [[[p[0], p[1]] for p in t] for t in g["paths"] if t]
                if len(trazos) == 1:
                    geom = {"type": "LineString", "coordinates": trazos[0]}
                elif trazos:
                    geom = {"type": "MultiLineString", "coordinates": trazos}
        salida.append({"type": "Feature", "geometry": geom,
                       "properties": r.get("attributes", {})})
    return {"type": "FeatureCollection", "features": salida}


def bajar(clave):
    """Baja una capa completa y deja el crudo en disco antes de procesar nada.

    Escribe dos cosas en `datos/crudo/mop/<fecha>/`:
      · `<clave>/lote_NNN.json.gz` — la respuesta esriJSON EXACTA del servidor,
        lote por lote, sin tocar. Es el crudo de verdad.
      · `<clave>.geojson[.gz]` — el mismo contenido convertido a GeoJSON, que es
        lo que pidió el encargo y lo que otros programas pueden abrir.
    """
    conf = CAPAS[clave]
    destino = CRUDO / HOY
    lotes_dir = destino / clave
    lotes_dir.mkdir(parents=True, exist_ok=True)

    campo_id, ids = _identificadores(clave)
    print(f"\n{clave}: el servicio declara {len(ids)} registros "
          f"(esperados {conf['esperado']}"
          f"{' — COINCIDE' if len(ids) == conf['esperado'] else ' — ★ DIFIERE'})")

    rasgos = []
    sin_dato = []                 # identificadores que el servicio no entrega
    n_lotes = (len(ids) + conf["lote"] - 1) // conf["lote"]
    for i in range(n_lotes):
        trozo = ids[i * conf["lote"]:(i + 1) * conf["lote"]]
        archivo = lotes_dir / f"lote_{i:03d}.json.gz"
        # Reanudación: si el lote ya está en disco de una corrida anterior, se
        # relee y no se vuelve a pedir. El servicio del MOP se cae de a ratos
        # (`read operation timed out`) y sin esto un corte al final obliga a
        # bajar los 14.000 tramos de nuevo, castigando a un servidor público por
        # un problema nuestro.
        if archivo.exists():
            with gzip.open(archivo, "rt", encoding="utf-8") as fh:
                guardado = json.load(fh)
            venia_de_disco = True
        else:
            faltantes, crudos = [], []
            _pedir_registros(clave, trozo, faltantes, crudos)
            # El archivo de crudo guarda las respuestas del servidor SIN TOCAR
            # (`respuestas`), más el contexto que hace falta para auditarlo: qué
            # identificadores se pidieron y cuáles el servicio no pudo dar.
            guardado = {"objectIds_pedidos": trozo,
                        "sin_dato": faltantes,
                        "respuestas": crudos}
            with gzip.open(archivo, "wt", encoding="utf-8") as fh:
                json.dump(guardado, fh, ensure_ascii=False)
            venia_de_disco = False
        # Se aceptan los dos formatos de crudo: el envoltorio de arriba y las
        # respuestas sueltas que dejaron las corridas anteriores del adaptador.
        if "respuestas" in guardado:
            traidos = [f for r in guardado["respuestas"]
                       for f in r.get("features", [])]
            sin_dato.extend(guardado.get("sin_dato", []))
        else:
            traidos = guardado.get("features", [])
        rasgos.extend(traidos)
        print(f"   lote {i + 1:>3}/{n_lotes}  pedidos {len(trozo):>4}  "
              f"traídos {len(traidos):>4}  acumulado {len(rasgos):>6}"
              f"{'  (de disco)' if venia_de_disco else ''}")
        if len(traidos) != len(trozo):
            print(f"      ★ el lote no devolvió lo pedido: "
                  f"{len(trozo) - len(traidos)} registros sin dato")
        if not venia_de_disco:
            time.sleep(PAUSA)

    if len(rasgos) != len(ids):
        print(f"   ★ ATENCIÓN: se pidieron {len(ids)} y llegaron {len(rasgos)}. "
              f"La diferencia queda declarada, no rellenada.")
    if sin_dato:
        print(f"   ★ el servicio no pudo entregar {len(sin_dato)} registro(s): "
              f"{sorted(sin_dato)}")

    coleccion = _a_geojson(rasgos, conf["geometria"])
    coleccion["metadata"] = {
        "fuente": FUENTE["id"],
        "servicio": f"{BASE}/{conf['servicio']}/MapServer/{conf['capa']}",
        "campo_identificador": campo_id,
        "bajado": HOY,
        "registros_declarados_por_el_servicio": len(ids),
        "registros_recibidos": len(rasgos),
        "objectids_que_el_servicio_no_entrega": sorted(sin_dato),
        "srs_original": "EPSG:5360 (SIRGAS-Chile / UTM 19S)",
        "srs_entregado": "EPSG:4326 (WGS84), reproyectado por el servidor",
        "licencia": FUENTE["condiciones_uso"],
    }
    # Los tramos viales con geometría completa pesan cientos de MB en texto
    # plano; se guardan comprimidos. Los puntos caben sin comprimir.
    if conf["geometria"] == "linea":
        with gzip.open(destino / f"{clave}.geojson.gz", "wt",
                       encoding="utf-8") as fh:
            json.dump(coleccion, fh, ensure_ascii=False)
        print(f"   → {destino / (clave + '.geojson.gz')}")
    else:
        (destino / f"{clave}.geojson").write_text(
            json.dumps(coleccion, ensure_ascii=False), encoding="utf-8")
        print(f"   → {destino / (clave + '.geojson')}")
    return coleccion


def _leer_crudo(clave, fecha=None):
    """Relee del disco lo ya bajado, para no volver a molestar al servicio."""
    fecha = fecha or _ultima_bajada()
    destino = CRUDO / fecha
    plano = destino / f"{clave}.geojson"
    comprimido = destino / f"{clave}.geojson.gz"
    if plano.exists():
        return json.loads(plano.read_text(encoding="utf-8"))
    if comprimido.exists():
        with gzip.open(comprimido, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    raise FileNotFoundError(f"no hay crudo de «{clave}» en {destino}")


def _ultima_bajada():
    fechas = sorted(p.name for p in CRUDO.glob("*") if p.is_dir())
    if not fechas:
        raise FileNotFoundError(f"no hay nada bajado en {CRUDO}")
    return fechas[-1]


# ── utilidades de limpieza ───────────────────────────────────────────────────

def _texto(valor):
    """Limpia un campo de texto del servicio. Devuelve None cuando no hay dato.

    El registro trae `None`, cadena vacía, `'-'`, `' '` y saltos de línea
    pegados (`'Puente\\r\\n'`) como si fueran valores distintos. Son todos lo
    mismo: sin dato o el mismo valor mal escrito.
    """
    if valor is None:
        return None
    t = " ".join(str(valor).split()).strip()
    if t in ("", "-", "--", "S/N", "s/n", "SIN INFORMACION", "SIN INFORMACIÓN"):
        return None
    return t


def _clave_cauce(nombre):
    """Normaliza el nombre de un cauce para poder agrupar sin fundir cosas
    distintas.

    Mayúsculas, sin tildes, sin espacios de más. NO se le quita la palabra
    «RIO», «ESTERO» o «QUEBRADA»: «Estero Las Cruces» y «Río Las Cruces» pueden
    ser dos cursos de agua distintos y fundirlos inventaría un punto de fallo
    múltiple que no existe.
    """
    if nombre is None:
        return None
    t = unicodedata.normalize("NFKD", nombre.upper())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = " ".join(t.replace(".", " ").replace(",", " ").split())
    return t or None


def _fecha_local(epoch_ms):
    """Epoch en milisegundos UTC (como lo entrega ArcGIS) → fecha de Chile."""
    if epoch_ms in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(int(epoch_ms) / 1000, tz=timezone.utc)\
                       .astimezone(DESFASE_CHILE)
    except (ValueError, OSError, OverflowError):
        return None


def _punto_representativo(geometria):
    """Un punto por geometría, para poder ubicarla en una comuna.

    Para un punto, es el punto. Para una línea, es el **vértice del medio** del
    trazo más largo — no un centroide calculado, que en una carretera curva
    puede caer fuera del camino. Se declara como «representativo», nunca como
    «la ubicación del tramo»: un tramo largo cruza varias comunas y este punto
    sólo dice por dónde pasa el medio.
    """
    if not geometria:
        return None, None
    t = geometria["type"]
    if t == "Point":
        lon, lat = geometria["coordinates"][:2]
        return lat, lon
    trazos = ([geometria["coordinates"]] if t == "LineString"
              else geometria["coordinates"])
    trazos = [t_ for t_ in trazos if t_]
    if not trazos:
        return None, None
    mayor = max(trazos, key=len)
    lon, lat = mayor[len(mayor) // 2][:2]
    return lat, lon


# ── 3. tablas ────────────────────────────────────────────────────────────────

# Texto y número se tratan distinto a propósito: `_texto` limpia y convierte a
# None los «-» y las cadenas vacías, pero pasar por ahí un largo en metros lo
# volvería una cadena y se perdería para cualquier cálculo posterior.
CAMPOS_PUENTE_TEXTO = ["CODIGO_PUENTE", "NOMBRE_PUENTE", "ROL", "CODIGO_CAMINO",
                       "NOMBRE_CAMINO", "CAUCE_QUEB", "PROVINCIA", "REGION",
                       "INFRA_TIPO", "TIPO_ESTRUCTURA", "MAT_VIGAS",
                       "MAT_CEPAS", "EST_PUENTE", "TUICION"]
CAMPOS_PUENTE_NUM = ["M_INICIO", "LARGO", "ANCHO_TOTAL", "ANCHO_CALZADA",
                     "AÑO", "AÑO_CONTRUCCION"]


def tabla_puentes(territorio_=None):
    """`datos/mop_puentes.csv`: cada puente con su cauce, su coordenada y su
    comuna resuelta por geometría."""
    col = _leer_crudo("puentes")
    filas = []
    sin_geom = sin_cauce = 0
    for rasgo in col["features"]:
        p = rasgo["properties"]
        lat, lon = _punto_representativo(rasgo["geometry"])
        if lat is None:
            sin_geom += 1
        cauce = _texto(p.get("CAUCE_QUEB"))
        if cauce is None:
            sin_cauce += 1
        fila = {"objectid": p.get("OBJECTID")}
        for campo in CAMPOS_PUENTE_TEXTO:
            fila[campo.lower()] = _texto(p.get(campo))
        for campo in CAMPOS_PUENTE_NUM:
            fila[campo.lower().replace("ñ", "n")] = p.get(campo)
        fila["cauce_normalizado"] = _clave_cauce(cauce)
        fila["lat"] = lat
        fila["lon"] = lon
        fila["comuna"] = None
        fila["codigo_comuna"] = None
        fila["comuna_origen"] = "sin dato"
        if territorio_ is not None and lat is not None:
            u = territorio_.ubicar(lat, lon)
            fila["comuna"] = u["comuna"]
            fila["codigo_comuna"] = u["codigo_comuna"]
            fila["comuna_origen"] = ("punto_en_poligono_comunal" if u["comuna"]
                                     else "punto_fuera_de_toda_comuna")
        filas.append(fila)

    _escribir(DATOS / "mop_puentes.csv", filas)
    print(f"mop_puentes.csv: {len(filas)} puentes · sin geometría {sin_geom} · "
          f"sin CAUCE_QUEB {sin_cauce} "
          f"({100 * sin_cauce / max(1, len(filas)):.1f}%)")
    return filas


CAMPOS_TRAMO = ["ROL", "ROL_ID", "ROL_LABEL", "CODIGO_CAMINO", "NOMBRE_CAMINO",
                "CLASIFICACION", "CARPETA", "CALZADA", "ORIENTACION",
                "ENROLADO", "CONCESIONADO", "REGION", "TUICION"]


def tabla_tramos():
    """`datos/mop_tramos.csv`: el inventario vial con rol y kilometraje.

    `KM_I`, `KM_F` y `KM_TRAMO` vienen en METROS pese al nombre (verificado:
    KM_I=14474 / KM_F=17244 / KM_TRAMO=2770 en un tramo de 2,77 km). Se guardan
    tal cual vienen y además en kilómetros, con el nombre que corresponde.
    """
    col = _leer_crudo("tramos")
    filas = []
    sin_geom = 0
    for rasgo in col["features"]:
        p = rasgo["properties"]
        lat, lon = _punto_representativo(rasgo["geometry"])
        if lat is None:
            sin_geom += 1
        fila = {"objectid": p.get("OBJECTID")}
        for campo in CAMPOS_TRAMO:
            fila[campo.lower()] = _texto(p.get(campo))
        fila["km_i_metros"] = p.get("KM_I")
        fila["km_f_metros"] = p.get("KM_F")
        fila["largo_tramo_metros"] = p.get("KM_TRAMO")
        fila["largo_tramo_km"] = (round(p["KM_TRAMO"] / 1000.0, 3)
                                  if p.get("KM_TRAMO") is not None else None)
        fila["lat_punto_medio"] = lat
        fila["lon_punto_medio"] = lon
        fila["n_vertices"] = (len(rasgo["geometry"]["coordinates"])
                              if rasgo["geometry"] and
                              rasgo["geometry"]["type"] == "LineString"
                              else (sum(len(t) for t in
                                        rasgo["geometry"]["coordinates"])
                                    if rasgo["geometry"] else 0))
        filas.append(fila)
    _escribir(DATOS / "mop_tramos.csv", filas)
    largo = sum(f["largo_tramo_km"] or 0 for f in filas)
    print(f"mop_tramos.csv: {len(filas)} tramos · sin geometría {sin_geom} · "
          f"largo declarado total {largo:,.0f} km")
    return filas


# Raíces que en el texto libre del registro delatan una causa meteorológica o
# hidrológica. Es una HEURÍSTICA sobre texto escrito a mano por funcionarios de
# turno, no una clasificación oficial: el registro del MOP no trae campo de
# causa. Se declara como tal en cada salida y nunca se presenta como dato duro.
#
# Se buscan con frontera de palabra por delante (`\b`) y sin cerrar por detrás,
# para que «lluvia» atrape «lluvias» pero «rio» NO atrape «sanitario». Ese
# detalle no es cosmético: sin la frontera, «Sistema Sanitario» se clasificaba
# como causa hidrológica y el resultado entero se corrompía.
RAICES_METEO = (
    r"lluvia", r"precipitac", r"crecid", r"desbord", r"inundac", r"anega",
    r"aluvi", r"nieve", r"nevad", r"nevaz", r"temporal", r"frontal",
    r"viento", r"marejad", r"socavac", r"socavon", r"caudal", r"quebrada",
    r"escorrent", r"granizo", r"deshielo", r"rio\b", r"rios\b", r"agua",
)
RAICES_REMOCION = (r"derrumbe", r"rodado", r"rodados", r"deslizamiento",
                   r"remocion", r"desprendimiento", r"talud", r"alud")

_RE_METEO = None
_RE_REMOCION = None


def _sin_tildes(texto):
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _causa_declarada(texto):
    """Etiqueta gruesa sobre el texto libre: meteo / remoción / otra / sin dato."""
    global _RE_METEO, _RE_REMOCION
    if _RE_METEO is None:
        _RE_METEO = re.compile("|".join(r"\b" + r_ for r_ in RAICES_METEO))
        _RE_REMOCION = re.compile("|".join(r"\b" + r_ for r_ in RAICES_REMOCION))
    if not texto:
        return "sin dato"
    t = _sin_tildes(texto)
    meteo = bool(_RE_METEO.search(t))
    remo = bool(_RE_REMOCION.search(t))
    if meteo and remo:
        return "meteo_y_remocion"
    if meteo:
        return "meteo"
    if remo:
        return "remocion_en_masa"
    return "otra_o_no_dice"


def tabla_emergencias(territorio_=None):
    """`datos/mop_emergencias_viales.csv`: las dos capas de emergencias, unidas.

    Se conservan TODAS las emergencias MOP (6.141), no sólo las viales, y se
    marca cuáles lo son con `es_vial`. Descartar las demás perdería la línea
    base contra la cual se mide si la vialidad es o no el punto débil.
    """
    filas = []
    for clave, etiqueta in (("emergencias_historicas", "historica_2014_2019"),
                            ("emergencias_vigentes", "vigente_2022_2026")):
        col = _leer_crudo(clave)
        for rasgo in col["features"]:
            p = rasgo["properties"]
            lat, lon = _punto_representativo(rasgo["geometry"])
            fecha = _fecha_local(p.get("FECHA"))
            servicio = _texto(p.get("SERV_MOP"))
            direccion = _texto(p.get("DIRECCION"))
            descripcion = _texto(p.get("EMERGENCIA"))
            # Vialidad se declara de dos maneras según la capa: la histórica usa
            # DIRECCION='VIALIDAD', la vigente usa SERV_MOP='Dirección de Vialidad'.
            es_vial = ((direccion or "").upper() == "VIALIDAD" or
                       (servicio or "").lower().startswith("dirección de vialidad"))
            fila = {
                "capa_origen": etiqueta,
                "objectid": p.get("OBJECTID"),
                "id_emergencia": p.get("ID_EMER"),
                "fecha": fecha.date().isoformat() if fecha else None,
                "anio": fecha.year if fecha else None,
                "mes": fecha.month if fecha else None,
                "descripcion": descripcion,
                "causa_heuristica": _causa_declarada(descripcion),
                "infraestructura_afectada": _texto(p.get("INFRA_AFEC")),
                "elemento_afectado": _texto(p.get("ELEMENTO")),
                "rol_vial": _texto(p.get("ROL_VIAL")),
                "km_inicio": p.get("KM_INI"),
                "gravedad": _texto(p.get("GRAVEDAD")),
                "servicio_mop": servicio,
                "direccion_mop": direccion,
                "es_vial": int(es_vial),
                "codigo_region_declarado": _texto(p.get("CODREG")),
                "codigo_comuna_declarado": _texto(p.get("CODCOM")),
                "estado": _texto(p.get("ESTADO_EMER")),
                "lat": lat, "lon": lon,
                "comuna": None, "codigo_comuna": None,
                "comuna_origen": "sin dato",
            }
            if territorio_ is not None and lat is not None:
                u = territorio_.ubicar(lat, lon)
                fila["comuna"] = u["comuna"]
                fila["codigo_comuna"] = u["codigo_comuna"]
                fila["comuna_origen"] = ("punto_en_poligono_comunal" if u["comuna"]
                                         else "punto_fuera_de_toda_comuna")
            filas.append(fila)
    _escribir(DATOS / "mop_emergencias_viales.csv", filas)
    sin_fecha = sum(1 for f in filas if f["fecha"] is None)
    print(f"mop_emergencias_viales.csv: {len(filas)} emergencias · "
          f"viales {sum(f['es_vial'] for f in filas)} · sin fecha {sin_fecha}")
    return filas


def _escribir(ruta, filas):
    if not filas:
        print(f"   {ruta.name}: sin filas, no se escribe")
        return
    columnas = list(filas[0].keys())
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)


# ── 4. análisis ──────────────────────────────────────────────────────────────

# Valores que ocupan el campo CAUCE_QUEB pero NO nombran ningún cauce: son
# marcadores de «no sé» que el catastro escribió de siete maneras distintas.
# Contarlos como cauces produce el disparate de que el «cauce» con más puentes de
# Chile sea «S/I» con 121. Se excluyen del ranking y se reportan aparte, que es
# un dato en sí: 47 % de los puentes no dice qué cruza.
NO_SON_CAUCES = {
    "S/I", "S/N", "SIN NOMBRE", "SIN INFORMACION", "SIN IDENTIFICACION",
    "CAUCE SIN IDENTIFICACION", "CAUCE SIN IDENTIFICACIÓN", "DEFINIR REGION",
    "DEFINIR REGIÓN", "NO APLICA", "NO TIENE", "NINGUNO", "0", "X",
    "ESTERO S/N", "ESTERO SIN NOMBRE", "ESTERO SIN IDENTIFICACION",
    "QUEBRADA", "QUEBRADA S/N", "QUEBRADA SIN NOMBRE", "RIO", "RIO S/N",
    "RIO SIN NOMBRE", "CANAL", "CANAL S/N", "CANAL SIN NOMBRE",
    "CANAL DE REGADIO", "CANAL DE REGADÍO", "ESTERO", "VERTIENTE",
    "DESCONOCIDO", "PENDIENTE", "SIN DATO", "SIN CAUCE",
}


def cauces_con_mas_puentes(puentes, tope=25):
    """★ El análisis que justifica la fuente: cuántos puentes cuelga cada cauce.

    Un cauce con 19 puentes no es «19 riesgos independientes»: es UN evento
    hidrológico que puede cobrarse 19 cortes el mismo día. La lista ordenada es,
    literalmente, el ranking de puntos de fallo múltiple de la red vial chilena
    ante crecidas.

    Dos trampas que hay que esquivar o el ranking miente:

    1. **Los marcadores de «no sé»** (`S/I`, `DEFINIR REGION`, `ESTERO S/N`…)
       no son cauces. Se apartan (ver `NO_SON_CAUCES`) y se cuentan por separado.
    2. **Los homónimos.** «Río Claro» aparece 28 veces repartidas en SEIS
       regiones: son varios ríos distintos que se llaman igual, no un río con 28
       puentes. Una crecida del Río Claro del Maule no toca el Río Claro de
       Aysén. Por eso el ranking que vale para riesgo es el de **cauce dentro de
       una región** — y se muestran los dos, con el nombre a secas marcado
       cuando abarca más de una región.
    """
    por_cauce, por_cauce_region, marcadores = (defaultdict(list),
                                               defaultdict(list),
                                               defaultdict(int))
    for p in puentes:
        clave = p["cauce_normalizado"]
        if not clave:
            continue
        if clave in NO_SON_CAUCES:
            marcadores[clave] += 1
            continue
        por_cauce[clave].append(p)
        por_cauce_region[(clave, p["region"] or "sin región declarada")].append(p)

    total_marcadores = sum(marcadores.values())
    sin_campo = sum(1 for p in puentes if not p["cauce_normalizado"])
    print(f"\n★ CAUCES CON MÁS PUENTES ENCIMA")
    print(f"   {len(puentes)} puentes · {sin_campo} con CAUCE_QUEB vacío · "
          f"{total_marcadores} con un marcador de «no sé» "
          f"({sorted(marcadores.items(), key=lambda kv: -kv[1])[:5]}) · "
          f"{len(puentes) - sin_campo - total_marcadores} con cauce nombrado "
          f"en {len(por_cauce)} cauces distintos")

    print("\n   ── por cauce y REGIÓN (el que vale para riesgo: una crecida es "
          "de un río, en un lugar) ──")
    print(f"   {'cauce':32s} {'región':28s} {'puentes':>7}  comunas")
    orden_reg = sorted(por_cauce_region.items(), key=lambda kv: -len(kv[1]))
    for (clave, region), grupo in orden_reg[:tope]:
        comunas = sorted({g["comuna"] for g in grupo if g["comuna"]})
        print(f"   {clave[:32]:32s} {str(region)[:28]:28s} {len(grupo):>7}  "
              f"{', '.join(comunas[:3])}{'…' if len(comunas) > 3 else ''}")

    print("\n   ── por nombre de cauce, ignorando la región "
          "(★ = el nombre se repite en más de una región: son ríos distintos) ──")
    orden = sorted(por_cauce.items(), key=lambda kv: -len(kv[1]))
    for clave, grupo in orden[:tope]:
        regiones = sorted({g["region"] for g in grupo if g["region"]})
        marca = "★" if len(regiones) > 1 else " "
        print(f"   {marca} {clave[:34]:34s} {len(grupo):>4} puentes  "
              f"{len(regiones)} región(es)")
    return orden, orden_reg, marcadores


def estacionalidad(emergencias, solo_viales=False, etiqueta="todas",
                   causas=None):
    """¿Son de invierno las emergencias del MOP?

    Se mide la razón invierno/verano: cuántas emergencias caen en junio-julio-
    agosto contra cuántas caen en diciembre-enero-febrero. Es la misma medida
    con que el proyecto obtuvo 8,41 para fallas de causa meteorológica contra
    0,87 en el control, así que los números son comparables directamente.

    Una razón cerca de 1 significa «pasa todo el año»; muy por encima de 1,
    «esto lo trae el invierno».
    """
    filas = [e for e in emergencias
             if e["fecha"] and (not solo_viales or e["es_vial"])
             and (causas is None or e["causa_heuristica"] in causas)]
    meses = Counter(e["mes"] for e in filas)
    invierno = sum(meses[m] for m in (6, 7, 8))
    verano = sum(meses[m] for m in (12, 1, 2))
    razon = invierno / verano if verano else None
    print(f"\nESTACIONALIDAD — {etiqueta} (n={len(filas)})")
    print("   mes:  " + " ".join(f"{m:>5}" for m in range(1, 13)))
    print("   n:    " + " ".join(f"{meses[m]:>5}" for m in range(1, 13)))
    print(f"   invierno (JJA) {invierno} · verano (DEF) {verano} · "
          f"razón {'sin dato (verano=0)' if razon is None else f'{razon:.2f}'}")
    return {"etiqueta": etiqueta, "n": len(filas), "por_mes": dict(meses),
            "invierno_JJA": invierno, "verano_DEF": verano, "razon": razon}


def estacionalidad_por_anio(emergencias, solo_viales=True):
    """★ La prueba que desarma el número bonito: la misma razón, año por año.

    Agrupando los once años, la vialidad del MOP da una razón invierno/verano de
    8,8. Parece una confirmación redonda del 6,84 que el proyecto midió en
    SENAPRED. **No lo es**, y este desglose es la razón: la razón anual va de
    0,40 (2017) a 148,5 (2023). Un indicador que oscila 370 veces entre años
    contiguos no está midiendo el clima, está midiendo **cuándo se usa el
    registro**.

    El registro de emergencias del MOP es *reactivo*: se llena a mano cuando hay
    una emergencia declarada, y se queda casi vacío el resto del año. En un año
    de sistemas frontales grandes (2023, 2026) casi todo lo anotado es de
    invierno; en un año tranquilo (2017, 2019, 2025) la razón cae por debajo de 1.

    Consecuencia práctica, y hay que decirla: **esta fuente sirve para saber QUÉ
    se rompió y DÓNDE, no para medir estacionalidad**. El promedio agrupado
    debe citarse sólo junto a esta tabla.
    """
    filas = [e for e in emergencias
             if e["anio"] and (not solo_viales or e["es_vial"])]
    por_anio = defaultdict(list)
    for e in filas:
        por_anio[e["anio"]].append(e)
    print("\n★ LA MISMA RAZÓN, AÑO POR AÑO "
          f"({'vialidad' if solo_viales else 'todo el MOP'})")
    print(f"   {'año':>5} {'capa':>20} {'n':>6} {'JJA':>6} {'DEF':>6} "
          f"{'razón':>8}")
    salida = {}
    for anio in sorted(por_anio):
        grupo = por_anio[anio]
        meses = Counter(e["mes"] for e in grupo)
        inv = sum(meses[m] for m in (6, 7, 8))
        ver = sum(meses[m] for m in (12, 1, 2))
        razon = inv / ver if ver else None
        capa = Counter(e["capa_origen"] for e in grupo).most_common(1)[0][0]
        print(f"   {anio:>5} {capa:>20} {len(grupo):>6} {inv:>6} {ver:>6} "
              f"{'sin dato' if razon is None else f'{razon:>8.2f}'}")
        salida[anio] = {"n": len(grupo), "JJA": inv, "DEF": ver, "razon": razon}
    return salida


def concentracion(emergencias, tope=20):
    """Dónde se concentran. Si un puñado de comunas acumula la mayoría, el
    instrumento no necesita cubrir 346 comunas por igual."""
    filas = [e for e in emergencias if e["comuna"]]
    cuenta = Counter(e["comuna"] for e in filas)
    total = sum(cuenta.values())
    print(f"\nCONCENTRACIÓN TERRITORIAL "
          f"({len(cuenta)} comunas con al menos una emergencia ubicada; "
          f"{total} de {len(emergencias)} emergencias tienen comuna resuelta)")
    acum = 0
    for i, (comuna, n) in enumerate(cuenta.most_common(tope), 1):
        acum += n
        print(f"  {i:>2}. {comuna[:32]:32s} {n:>5}  acumulado "
              f"{100 * acum / total:5.1f}%")
    return cuenta


# ── orquestación ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--explorar", action="store_true")
    ap.add_argument("--bajar", action="store_true")
    ap.add_argument("--procesar", action="store_true")
    ap.add_argument("--todo", action="store_true")
    ap.add_argument("--capa", default=None, help="bajar sólo una capa")
    args = ap.parse_args()
    if not any((args.explorar, args.bajar, args.procesar, args.todo)):
        ap.print_help()
        return

    if args.explorar or args.todo:
        explorar()

    if args.bajar or args.todo:
        for clave in ([args.capa] if args.capa else list(CAPAS)):
            bajar(clave)

    if args.procesar or args.todo:
        t = territorio.Territorio()
        estado = t.estado()
        print(f"\nCapas territoriales: {estado}")
        if not t.comunas.disponible:
            print("   ★ sin capa comunal: la columna `comuna` saldrá vacía y "
                  "marcada, no inventada")
            t = None
        puentes = tabla_puentes(t)
        tabla_tramos()
        emerg = tabla_emergencias(t)
        cauces_con_mas_puentes(puentes)

        estacionalidad(emerg, False, "todas las emergencias MOP")
        estacionalidad(emerg, True, "sólo emergencias de Vialidad")
        estacionalidad_por_anio(emerg, solo_viales=True)
        # Los cuatro años calendario completos del registro HISTÓRICO son la
        # única ventana de esta fuente donde el conteo no lo domina un evento
        # puntual. Ahí sí se puede comparar causa contra control.
        hist = [e for e in emerg
                if e["anio"] in (2015, 2016, 2017, 2018) and e["es_vial"]]
        print("\n── ventana comparable: registro histórico 2015-2018 ──")
        estacionalidad(hist, True, "2015-2018 · texto meteo/hídrico",
                       causas={"meteo", "meteo_y_remocion"})
        estacionalidad(hist, True, "2015-2018 · texto de remoción sola",
                       causas={"remocion_en_masa"})
        estacionalidad(hist, True, "2015-2018 · CONTROL (no menciona ninguna)",
                       causas={"otra_o_no_dice", "sin dato"})
        # El contraste que importa: si la razón invernal fuera un artefacto del
        # registro (más funcionarios anotando en invierno), subiría también en el
        # grupo SIN causa climática. Ese grupo es el control.
        estacionalidad(emerg, True, "Vialidad · texto con causa meteo/hídrica",
                       causas={"meteo", "meteo_y_remocion"})
        estacionalidad(emerg, True, "Vialidad · texto de remoción en masa sola",
                       causas={"remocion_en_masa"})
        estacionalidad(emerg, True,
                       "Vialidad · CONTROL, el texto no menciona clima ni "
                       "remoción", causas={"otra_o_no_dice", "sin dato"})

        print("\nDesglose de la etiqueta heurística de causa (texto libre):")
        for causa, n in Counter(e["causa_heuristica"] for e in emerg
                                if e["es_vial"]).most_common():
            print(f"   {causa:20s} {n:>5}")
        concentracion([e for e in emerg if e["es_vial"]])


if __name__ == "__main__":
    main()
