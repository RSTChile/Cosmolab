"""
Adaptador de INVENTARIO — telecomunicaciones.
Fuente: Subsecretaría de Telecomunicaciones (SUBTEL), servidor
`licancabur.subtel.gob.cl`.

POR QUÉ IMPORTA, Y POR QUÉ ES EL ÍTEM MÁS TRAICIONERO DEL INVENTARIO
---------------------------------------------------------------------
Las telecomunicaciones eran el ítem de la Matriz que estaba completamente sin
poblar. Y no es un ítem más: es el que convierte una emergencia local en una
emergencia ciega.

Cuando se corta un camino, la comuna queda aislada físicamente. Cuando además se
cae la antena, la comuna queda aislada **e incomunicada**, y entonces el COGRID
no sabe qué está pasando ahí. El glosario oficial de Gestión del Riesgo de
Desastres llama a eso «aislamiento»; en la práctica es peor, porque un sistema
de alerta que no recibe información de una zona no la trata como zona en
problemas: la trata como zona sin novedad.

Y hay una dependencia que hace todo esto más frágil de lo que parece: **una
antena sin electricidad deja de transmitir cuando se le acaba la batería**, en
horas. O sea que el mismo corte eléctrico que deja sin agua a los sistemas de
agua potable rural sin grupo electrógeno, deja además sin comunicación a la zona.
Con este inventario, esa cadena —energía → agua → comunicación— se puede por fin
medir sobre activos reales y ubicados, no suponerla.

QUÉ SE BAJA
-----------
    clave                servicio SUBTEL                        registros   sistema
    antenas_autorizadas  LDT_ELEMENTOS_AUT_PT_1_2026               52.412   EPSG:4326
    antenas_en_servicio  LDT_ELEMENTOS_SERV_PT_1                   29.875   EPSG:32719
    estaciones_base      Estaciones_CRM_Julio2025_dpa              58.131   EPSG:32719
    red_conectividad     RCD_TRAMOS (líneas)                          430   EPSG:4326

★ AUTORIZADAS ≠ EN SERVICIO, Y LA DIFERENCIA NO ES UN ERROR.
«Autorizadas» son los elementos con acto administrativo de la Ley de Torres
(52.412). «En servicio» son los que SUBTEL registra efectivamente operando
(29.875). Un inventario de infraestructura crítica quiere las que EXISTEN, no
las que se permitieron. **Se bajan las dos y se declara cuál es cuál**; sumarlas
daría 82.287 antenas donde no las hay.

`red_conectividad` son los 430 tramos del programa de conectividad digital, con
geometría de línea. Es lo más cercano a un catastro de fibra óptica troncal que
publica el Estado. **No es la red de fibra completa del país** —esa está
fragmentada por operador y por oficio, sin capa consolidada— y decirlo importa,
porque un mapa de fibra incompleto usado como si fuera completo declararía
«sin cobertura» a zonas que sí la tienen.

CAMPOS QUE VALEN PARA EL MODELO
--------------------------------
· `alias`               la empresa operadora (Entel, Movistar, Claro, WOM…)
· `tiso_descripcion`    tipo de soporte: torre, muro, mástil, poste
· `sopo_altura`         altura del soporte en metros — expuesta a viento
· `tecnologia` / `desc_com_tecnologia`  2G, 3G, 4G, 5G
· `banda`, `frecuencia`, `anchobanda`
· `stdo_descripcion`    estado del acto administrativo

CÓMO SE PAGINA
--------------
Este servidor corre ArcGIS 11.5 y **sí soporta `resultOffset` de verdad** — lo
que no es obvio, porque el servidor del Ministerio de Obras Públicas (ArcGIS
10.2.1) lo acepta y lo IGNORA en silencio. Acá se comprobó explícitamente antes
de confiar en él: pidiendo `resultOffset=0` y `resultOffset=5` con orden por
identificador, las respuestas empiezan en registros distintos (1 y 6). La
comprobación quedó dentro del módulo, en `verificar_paginacion()`, para que
vuelva a hacerse si algún día el servidor cambia.

Aun así el conteo se cierra por construcción: se pide primero cuántos hay, se
pagina, y al final se compara. Si no cuadra, se dice.

CONDICIONES DE USO
------------------
· `https://licancabur.subtel.gob.cl/robots.txt` → **404**: el servidor de datos
  no declara restricciones. Verificado el 20-ago-2026.
· `https://www.subtel.gob.cl/robots.txt` → 200, y sólo bloquea `/wp-admin/`.
  No hay restricción para agentes automáticos ni señales de contenido.
· Estas capas son las que alimentan el mapa público de antenas de la Ley de
  Torres (`antenas.subtel.gob.cl/leydetorres/`), que la propia SUBTEL publica
  para consulta ciudadana. Son datos pensados para ser vistos.
· `copyrightText` viene **vacío**: SUBTEL no declara licencia. Se atribuye a
  SUBTEL en todo derivado y queda **PENDIENTE pedir confirmación por escrito**.
· Ritmo prudente: lotes de 2.000 (el máximo que declara el servicio), pausa
  entre lotes, reintentos con espera creciente.

NO SE RECOLECTAN DATOS DE PERSONAS. Los campos describen instalaciones y
empresas concesionarias. `dte_direccion` es la dirección del emplazamiento de la
antena, no el domicilio de nadie; se conserva porque es la ubicación del activo.

USO
---
    python3 adaptadores/inventario_telecom_subtel.py explorar
    python3 adaptadores/inventario_telecom_subtel.py bajar
    python3 adaptadores/inventario_telecom_subtel.py tablas
"""

import csv
import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

AQUI = Path(__file__).parent.parent
sys.path.insert(0, str(AQUI))
import territorio  # noqa: E402

BASE = "https://licancabur.subtel.gob.cl/server/rest/services"
HOY = date.today().isoformat()
CRUDO = AQUI / "datos" / "crudo" / "subtel"
DATOS = AQUI / "datos"

PAUSA = 1.5
REINTENTOS = 5
LOTE = 2000          # el maxRecordCount que declara el propio servicio


CAPAS = {
    "antenas_autorizadas": dict(
        servicio="LDT_ELEMENTOS_AUT_PT_1_2026", capa=0, esperado=52412,
        item_micr="antenas_de_telecomunicaciones", geometria="punto",
        descripcion="Elementos de antena AUTORIZADOS bajo la Ley de Torres. "
                    "Es el permiso, no necesariamente el fierro instalado."),
    "antenas_en_servicio": dict(
        servicio="LDT_ELEMENTOS_SERV_PT_1", capa=0, esperado=29875,
        item_micr="antenas_de_telecomunicaciones", geometria="punto",
        descripcion="★ Elementos de antena EN SERVICIO. Éste es el inventario "
                    "de lo que efectivamente está operando."),
    "estaciones_base": dict(
        servicio="Estaciones_CRM_Julio2025_dpa", capa=0, esperado=58131,
        item_micr="estaciones_base_moviles", geometria="punto",
        descripcion="Estaciones base del servicio público de radiocomunicación "
                    "móvil, corte de julio de 2025. Registro distinto del de la "
                    "Ley de Torres: no se suman."),
    "red_conectividad": dict(
        servicio="RCD_TRAMOS", capa=0, esperado=430,
        item_micr="red_de_conectividad_digital", geometria="linea",
        descripcion="Tramos del programa de conectividad digital, con traza. "
                    "NO es la red nacional de fibra óptica completa."),
}

FUENTE = dict(
    id="subtel_licancabur",
    organismo="Subsecretaría de Telecomunicaciones (SUBTEL)",
    producto="Servicios ArcGIS REST de SUBTEL: elementos de antena de la Ley de "
             "Torres, estaciones base móviles y tramos de la red de "
             "conectividad digital",
    url=BASE,
    formato="esriJSON (convertido a GeoJSON por este módulo)",
    familia="ESTADO",
    acceso="anonimo",
    acceso_verificado=1,
    robots_txt="licancabur.subtel.gob.cl/robots.txt → 404 (sin restricciones). "
               "www.subtel.gob.cl/robots.txt → 200, sólo bloquea /wp-admin/. "
               "Verificado 20-ago-2026.",
    condiciones_uso="Servicio público sin credenciales; alimenta el mapa "
                    "ciudadano de antenas de la Ley de Torres. "
                    "`copyrightText` VACÍO: sin licencia declarada. "
                    "PENDIENTE confirmación escrita de SUBTEL.",
    permite_automatizacion="si (sin robots.txt que lo restrinja)",
    granularidad="elemento de antena / estación base / tramo de red",
)


# ── acceso ───────────────────────────────────────────────────────────────────

def _pedir(url, parametros, metodo="GET", intentos=REINTENTOS):
    cuerpo = urllib.parse.urlencode(parametros).encode("utf-8")
    espera = 4
    for intento in range(1, intentos + 1):
        try:
            if metodo == "POST":
                pedido = urllib.request.Request(url, data=cuerpo)
            else:
                pedido = urllib.request.Request(url + "?" + cuerpo.decode())
            pedido.add_header("Accept-Encoding", "gzip")
            pedido.add_header("User-Agent",
                              "matriz-infraestructura-critica-clima/1.0 "
                              "(investigacion; inventario de infraestructura "
                              "critica)")
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


# ── 0. la comprobación que no hay que saltarse ───────────────────────────────

def verificar_paginacion(clave="antenas_autorizadas"):
    """¿Este servidor RESPETA `resultOffset`, o lo acepta y lo ignora?

    Vale la pena tener esto escrito y ejecutable: el servidor del Ministerio de
    Obras Públicas (ArcGIS 10.2.1) acepta `resultOffset` sin protestar y
    devuelve siempre desde el principio. Paginar ahí con offset no falla a
    gritos, falla callado, y el inventario sale con los primeros mil registros
    repetidos veinte veces. Antes de confiar en la paginación de CUALQUIER
    servidor nuevo, se comprueba.
    """
    url = f"{_url_capa(clave)}/query"
    base = {"where": "1=1", "outFields": "objectid", "returnGeometry": "false",
            "orderByFields": "objectid", "resultRecordCount": "3", "f": "json"}
    primero = _pedir(url, dict(base, resultOffset="0"))
    time.sleep(PAUSA)
    quinto = _pedir(url, dict(base, resultOffset="5"))
    a = [f["attributes"]["objectid"] for f in primero.get("features", [])]
    b = [f["attributes"]["objectid"] for f in quinto.get("features", [])]
    respeta = bool(a) and bool(b) and a != b
    print(f"   paginación: offset=0 → {a} · offset=5 → {b}")
    print(f"   ⇒ el servidor {'SÍ respeta' if respeta else '★ IGNORA'} "
          f"`resultOffset`")
    return respeta


# ── 1. exploración ───────────────────────────────────────────────────────────

def explorar():
    print(f"SUBTEL · {BASE}\n")
    for clave, c in CAPAS.items():
        try:
            meta = _pedir(_url_capa(clave), {"f": "json"}, intentos=2)
            n = _pedir(f"{_url_capa(clave)}/query",
                       {"where": "1=1", "returnCountOnly": "true", "f": "json"},
                       intentos=2).get("count")
        except Exception as err:                       # noqa: BLE001
            print(f"  {clave:22s} NO RESPONDE ({str(err)[:60]})")
            continue
        sr = (meta.get("extent") or {}).get("spatialReference", {})
        sr = sr.get("latestWkid") or sr.get("wkid") or "sin dato"
        marca = "" if n == c["esperado"] else f"  ★ el módulo esperaba {c['esperado']}"
        print(f"  {clave:22s} {n:>6} reg · {meta.get('geometryType')} · "
              f"EPSG:{sr}{marca}")
        time.sleep(PAUSA)
    print()
    verificar_paginacion()


# ── 2. bajada del crudo ──────────────────────────────────────────────────────

def _a_geojson(rasgos, tipo_geom):
    salida = []
    for r in rasgos:
        props = dict(r.get("attributes") or {})
        g = r.get("geometry") or {}
        geom = None
        if tipo_geom == "linea":
            caminos = g.get("paths") or []
            if caminos:
                geom = ({"type": "LineString", "coordinates": caminos[0]}
                        if len(caminos) == 1
                        else {"type": "MultiLineString", "coordinates": caminos})
        elif g.get("x") is not None and g.get("y") is not None:
            geom = {"type": "Point", "coordinates": [g["x"], g["y"]]}
        salida.append({"type": "Feature", "geometry": geom, "properties": props})
    return salida


def bajar(clave):
    """Baja una capa paginando por `resultOffset`, con el conteo cerrado.

    Las capas vienen en distintos sistemas de referencia (EPSG:4326 unas,
    EPSG:32719 —UTM huso 19 Sur— otras). Se pide siempre `outSR=4326` para que
    la reproyección la haga el servidor, que sabe qué datum usó, en vez de
    hacerla nosotros suponiéndolo.
    """
    c = CAPAS[clave]
    destino = CRUDO / HOY
    destino.mkdir(parents=True, exist_ok=True)
    print(f"\n[{clave}] {c['servicio']} capa {c['capa']}")

    total = _pedir(f"{_url_capa(clave)}/query",
                   {"where": "1=1", "returnCountOnly": "true", "f": "json"}
                   ).get("count")
    print(f"   el servicio declara {total} registros "
          f"(el módulo esperaba {c['esperado']})")
    time.sleep(PAUSA)

    rasgos, offset, lotes = [], 0, 0
    while offset < total:
        resp = _pedir(f"{_url_capa(clave)}/query",
                      {"where": "1=1", "outFields": "*",
                       "returnGeometry": "true", "outSR": "4326",
                       "orderByFields": "objectid",
                       "resultOffset": str(offset),
                       "resultRecordCount": str(LOTE), "f": "json"},
                      metodo="POST")
        trozo = resp.get("features", [])
        if not trozo:
            print(f"   ★ el servidor devolvió 0 registros en offset={offset} "
                  f"con {total - offset} pendientes: se corta acá y se declara")
            break
        rasgos += trozo
        offset += len(trozo)
        lotes += 1
        print(f"   lote {lotes}: {len(trozo)} registros · acumulado {offset}/{total}")
        time.sleep(PAUSA)

    cierra = "✓ cierra" if len(rasgos) == total else "★ NO CIERRA"
    print(f"   {len(rasgos)} de {total} — {cierra}")

    geojson = {
        "type": "FeatureCollection",
        "features": _a_geojson(rasgos, c["geometria"]),
        "procedencia": {
            "url": f"{_url_capa(clave)}/query",
            "servicio": c["servicio"], "capa": c["capa"],
            "descripcion": c["descripcion"],
            "fecha_descarga": HOY,
            "registros_declarados_por_el_servicio": total,
            "registros_recibidos": len(rasgos),
            "sistema_referencia_pedido": "EPSG:4326",
            "fuente": FUENTE,
        },
    }
    with gzip.open(destino / f"{clave}.geojson.gz", "wt", encoding="utf-8") as fh:
        json.dump(geojson, fh, ensure_ascii=False)
    print(f"   → datos/crudo/subtel/{HOY}/{clave}.geojson.gz")
    return geojson


def bajar_todo(claves=None):
    if not verificar_paginacion():
        print("★ El servidor NO respeta `resultOffset`. No se baja nada: una "
              "descarga paginada así saldría con registros repetidos y NADIE "
              "se daría cuenta. Hay que cambiar de estrategia de paginación.")
        return
    for clave in (claves or list(CAPAS)):
        try:
            bajar(clave)
        except Exception as err:                       # noqa: BLE001
            print(f"   ★ {clave} NO SE PUDO BAJAR: {str(err)[:140]}")
            print("     queda como «sin dato» — no se inventa nada")
        time.sleep(PAUSA)


# ── 3. tablas ────────────────────────────────────────────────────────────────

def _ultima_bajada():
    fechas = sorted(p.name for p in CRUDO.glob("*") if p.is_dir())
    if not fechas:
        raise FileNotFoundError(f"no hay nada bajado en {CRUDO}")
    return fechas[-1]


def _leer_crudo(clave, fecha=None):
    fecha = fecha or _ultima_bajada()
    ruta = CRUDO / fecha / f"{clave}.geojson.gz"
    if not ruta.exists():
        raise FileNotFoundError(f"falta {ruta} — correr `bajar {clave}` primero")
    with gzip.open(ruta, "rt", encoding="utf-8") as fh:
        return json.load(fh), fecha


def _texto(valor):
    return "" if valor is None else str(valor).strip()


def _numero(valor):
    if valor is None or valor == "":
        return ""
    try:
        n = float(valor)
    except (TypeError, ValueError):
        return ""
    return int(n) if n == int(n) else round(n, 3)


def _flotante(valor):
    try:
        n = float(valor)
    except (TypeError, ValueError):
        return None
    return n if n == n else None


def _plausible(lat, lon):
    if lat is None or lon is None:
        return False
    return -90.0 <= lat <= -17.0 and -110.0 <= lon <= -66.0


def _punto(geometria):
    """Coordenada representativa. Para una línea, el vértice del medio del
    trazado — que NO es la línea, y el nombre del campo lo dice."""
    if not geometria:
        return None, None, "sin_coordenada", 0
    if geometria.get("type") == "Point":
        lon, lat = (_flotante(v) for v in geometria["coordinates"][:2])
        return (lat, lon, "punto_del_servicio_epsg4326", 1) if None not in (lat, lon) \
            else (None, None, "coordenada_ilegible", 0)
    caminos = ([geometria["coordinates"]]
               if geometria.get("type") == "LineString"
               else geometria.get("coordinates") or [])
    vertices = [v for camino in caminos for v in camino]
    if not vertices:
        return None, None, "sin_coordenada", 0
    lon, lat = (_flotante(v) for v in vertices[len(vertices) // 2][:2])
    if None in (lat, lon):
        return None, None, "coordenada_ilegible", len(vertices)
    return lat, lon, "vertice_medio_de_la_linea_NO_es_la_linea", len(vertices)


CAMPOS_COMUNES = ["id_activo", "item_micr", "nombre", "lat", "lon",
                  "comuna", "provincia", "region", "cut", "zona_geografica",
                  "operador", "fuente", "fecha_captura", "confianza_ubicacion"]


def tabla(clave, fecha=None, terr=None):
    c = CAPAS[clave]
    crudo, fecha = _leer_crudo(clave, fecha)
    terr = terr if terr is not None else territorio.Territorio()
    filas, sin_coord, fuera = [], 0, 0

    for f in crudo["features"]:
        p = f["properties"]
        lat, lon, origen, n_vert = _punto(f.get("geometry"))
        if lat is not None and not _plausible(lat, lon):
            fuera += 1
            lat = lon = None
            origen = "fuera_de_chile_no_usada"
        if lat is None:
            sin_coord += 1
            u = {"comuna": "", "provincia": "", "region": "",
                 "codigo_comuna": "", "zona_geografica": "",
                 "faltan": "sin_coordenada"}
        else:
            x = terr.ubicar(lat, lon)
            u = {"comuna": x["comuna"] or "", "provincia": x["provincia"] or "",
                 "region": x["region"] or "",
                 "codigo_comuna": x["codigo_comuna"] or "",
                 "zona_geografica": x["zona_geografica"] or "",
                 "faltan": "|".join(x["faltan"])}

        operador = _texto(p.get("alias")) or _texto(p.get("empresa"))
        nombre = (_texto(p.get("elm_nombre")) or _texto(p.get("codigo"))
                  or _texto(p.get("caes_estac")) or "")
        filas.append({
            "id_activo": f"TEL-{clave[:3].upper()}-{p.get('objectid')}",
            "item_micr": c["item_micr"],
            "nombre": nombre,
            "lat": lat if lat is not None else "",
            "lon": lon if lon is not None else "",
            "comuna": u["comuna"], "provincia": u["provincia"],
            "region": u["region"], "cut": u["codigo_comuna"],
            "zona_geografica": u["zona_geografica"],
            "operador": operador,
            "fuente": f"SUBTEL · {c['servicio']}",
            "fecha_captura": fecha,
            "confianza_ubicacion": origen,
            # ── propios ──
            "registro": ("Ley de Torres · AUTORIZADO"
                         if clave == "antenas_autorizadas"
                         else "Ley de Torres · EN SERVICIO"
                         if clave == "antenas_en_servicio"
                         else "Concesión de radiocomunicación móvil"
                         if clave == "estaciones_base"
                         else "Programa de conectividad digital"),
            "tipo_soporte": _texto(p.get("tiso_descripcion")),
            "altura_soporte_m": _numero(p.get("sopo_altura")),
            "tecnologia": _texto(p.get("desc_com_tecnologia")) or _texto(p.get("tecnologia")),
            "banda": _texto(p.get("banda")),
            "frecuencia_mhz": _numero(p.get("frecuencia")),
            "estado_acto": _texto(p.get("stdo_descripcion")) or _texto(p.get("estado")),
            "anio_documento": _texto(p.get("anio_doc")),
            "direccion_emplazamiento": _texto(p.get("dte_direccion")) or _texto(p.get("direccion")),
            "comuna_declarada": _texto(p.get("comuna")) or _texto(p.get("nom_com")),
            "region_declarada": _texto(p.get("region")) or _texto(p.get("nom_reg")),
            "vertices_del_trazado": n_vert if c["geometria"] == "linea" else "",
            "territorio_faltan": u["faltan"],
        })

    campos = CAMPOS_COMUNES + [
        "registro", "tipo_soporte", "altura_soporte_m", "tecnologia", "banda",
        "frecuencia_mhz", "estado_acto", "anio_documento",
        "direccion_emplazamiento", "comuna_declarada", "region_declarada",
        "vertices_del_trazado", "territorio_faltan"]
    ruta = DATOS / f"inventario_telecom_{clave}.csv"
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    aviso = f" · {sin_coord} sin coordenada" if sin_coord else ""
    aviso += f" · {fuera} fuera de Chile" if fuera else ""
    print(f"   → {ruta.name}  ({len(filas)} filas{aviso})")
    return filas


def cobertura_por_operador(filas):
    """Quién tiene qué. No es un dato de mercado: es un dato de riesgo.

    Si una comuna depende de UN solo operador, el modo de falla de esa comuna no
    es «se cae una antena», es «se cae una empresa». Y ese es un riesgo de
    concentración que no se ve mirando el mapa de antenas."""
    por_operador = {}
    comunas = {}
    for f in filas:
        if f["lat"] == "":
            continue
        por_operador[f["operador"] or "(sin operador declarado)"] = \
            por_operador.get(f["operador"] or "(sin operador declarado)", 0) + 1
        if f["comuna"]:
            comunas.setdefault(f["comuna"], set()).add(f["operador"])
    print("     antenas en servicio por operador:")
    for op, n in sorted(por_operador.items(), key=lambda x: -x[1])[:10]:
        print(f"        {op[:38]:38s} {n:6d}")
    solo_uno = [c for c, ops in comunas.items() if len(ops) == 1]
    print(f"     ★ comunas con UN SOLO operador: {len(solo_uno)} de "
          f"{len(comunas)} con antena")
    return por_operador


def tablas(fecha=None):
    terr = territorio.Territorio()
    print("Capas de territorio:", terr.estado(), "\n")
    total = 0
    servicio = None
    for clave in CAPAS:
        try:
            filas = tabla(clave, fecha, terr)
            if clave == "antenas_en_servicio":
                servicio = filas
            total += sum(1 for f in filas if f["lat"] != "")
        except FileNotFoundError as err:
            print(f"   ★ {clave}: {err}")
    if servicio:
        cobertura_por_operador(servicio)
    print(f"\nTotal georreferenciado aportado por SUBTEL: {total:,}")
    print("   (recordar: `antenas_autorizadas` y `antenas_en_servicio` NO se "
          "suman entre sí — son el permiso y el fierro)")
    return total


def main():
    orden = sys.argv[1] if len(sys.argv) > 1 else "explorar"
    if orden == "explorar":
        explorar()
    elif orden == "bajar":
        bajar_todo(sys.argv[2:] or None)
    elif orden == "tablas":
        tablas()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
