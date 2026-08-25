"""
Adaptador de INVENTARIO — Coordinador Eléctrico Nacional, plataforma Infotécnica.

★ ESTE MÓDULO DESBLOQUEA EL CUELLO DE BOTELLA DEL PROYECTO
-----------------------------------------------------------
`SUBPROYECTO_SUBMATRICES.md` sección 5 dice, textual: «El caso testigo es el
Coordinador: **tiene las 1.269 subestaciones y ninguna trae coordenada**. Sin
coordenada, un activo no se puede cruzar con ninguna amenaza — y el cruce es
todo el proyecto.»

**Eso era cierto del endpoint que se había probado, y falso del servicio.**

El endpoint `/v1/subestaciones/` efectivamente NO trae coordenadas: devuelve
nombre, código, propietario y banderas de equipamiento, nada más. Pero la misma
interfaz publica `/v1/subestaciones/extended/`, y ahí cada subestación viene con
`latitud` y `longitud`. Verificado el 19-ago-2026: **1.226 de 1.273
subestaciones tienen coordenada (96,3 %)**.

La lección vale más que el dato, y conviene dejarla escrita: *la conclusión «esta
fuente no tiene coordenadas» se había sacado de UN endpoint, no del servicio.*
Antes de declarar que una fuente no tiene un dato, hay que leerle el catálogo
completo. Éste lo publica: `https://api-infotecnica.coordinador.cl/?format=openapi`
es la especificación OpenAPI entera, 176 rutas, sin credenciales.

QUÉ ES EL COORDINADOR Y POR QUÉ PUBLICA ESTO
--------------------------------------------
El Coordinador Eléctrico Nacional es el organismo técnico e independiente que
opera el Sistema Eléctrico Nacional de Chile. La Ley 20.936 le encarga mantener
y publicar la información técnica de todas las instalaciones del sistema: esa es
la plataforma «Infotécnica». No es un portal de cortesía — es una obligación
legal de transparencia, y por eso el dato es completo y está al día (el campo
`fecha_modificacion` de los registros bajados hoy trae la fecha de HOY).

QUÉ SE BAJA
-----------
    endpoint                          registros   con coordenada
    /v1/subestaciones/                    1.269         0   (el que se había probado)
    /v1/subestaciones/extended/           1.273     1.226   ★ el que sirve
    /v1/centrales/extended/               1.218     1.208   centrales de generación
    /v1/taps/extended/                      277       ~277   derivaciones de línea
    /v1/lineas/                           1.241         0   líneas de transmisión
    /v1/secciones-tramos/extended/        2.926         0   tramos, 40.643 km

★ LAS LÍNEAS SIGUEN SIN TRAZA, Y ESO SE DECLARA.
Los tramos traen un campo `coordenadas`, pero viene VACÍO en los 2.926 registros
— comprobado uno por uno, no supuesto. O sea: el Coordinador publica **40.643 km
de línea de transmisión sin geometría**. Lo que sí trae cada tramo son sus dos
extremos por nombre de subestación (`tramo_extremo1` / `tramo_extremo2`), lo que
permitiría dibujar una recta entre dos puntos conocidos. **Este módulo no lo
hace**: una recta entre dos subestaciones no es el trazado de una línea que
cruza cordillera, y un cruce con amenaza hecho sobre una recta inventada daría
un resultado falso con apariencia de exactitud. Se deja la tabla de líneas sin
coordenada, con sus extremos nombrados, y el hueco declarado.

POR QUÉ IMPORTAN LAS CENTRALES Y NO SÓLO LAS SUBESTACIONES
----------------------------------------------------------
La Matriz tiene 14 ítems de Energía con prioridad `Pen = Muy Alta`. Una
subestación caída deja sin luz a una zona; una central caída quita potencia al
sistema entero. Las centrales traen `potencia_maxima` en megavatios y
`tipo_central_nombre`, así que el inventario puede decir no sólo «cuántas» sino
«cuánta potencia» está expuesta a una amenaza dada — que es la pregunta que
importa.

CONDICIONES DE USO — ★ HAY UN MATIZ QUE HAY QUE LEER
-----------------------------------------------------
· `https://api-infotecnica.coordinador.cl/robots.txt` devuelve **404**: el host
  de la interfaz no declara restricciones de rastreo. Verificado 19-ago-2026.
· El servicio es público, anónimo, documentado con OpenAPI/Swagger y pensado
  para acceso programático. No pide credenciales.
· ★ PERO el sitio institucional `https://www.coordinador.cl/robots.txt` **sí**
  declara restricciones, y no son menores: `Content-Signal: search=yes,
  ai-train=no, use=reference`, y `Disallow: /` para una lista de agentes de
  inteligencia artificial (ClaudeBot, GPTBot, CCBot, Google-Extended, Bytespider
  y otros).
  Es otro host y otro servicio —el sitio web, no la interfaz de datos—, pero la
  intención del organismo queda expresada y este módulo la respeta:
    · se identifica con el agente del proyecto, no con un agente de rastreo de
      inteligencia artificial;
    · los datos se usan como **referencia** en un inventario de investigación,
      que es exactamente el `use: reference` que el sitio permite;
    · **no se usan para entrenar ningún modelo**, que es lo que el sitio
      prohíbe;
    · ritmo prudente y descarga completa una sola vez, no rastreo continuo.
· PENDIENTE, y es trámite de Alexis, no del proyecto: **pedir por escrito al
  Coordinador la confirmación de uso**. Es barato ahora y evita rehacer trabajo
  el día que el instrumento pase de investigación a apoyo operativo de SENAPRED.

NO SE RECOLECTAN DATOS DE PERSONAS. Todos los campos bajados describen
instalaciones y empresas propietarias; ninguno identifica a un individuo.

USO
---
    python3 adaptadores/inventario_coordinador_electrico.py explorar
    python3 adaptadores/inventario_coordinador_electrico.py bajar
    python3 adaptadores/inventario_coordinador_electrico.py tablas
"""

import csv
import gzip
import json
import math
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

BASE = "https://api-infotecnica.coordinador.cl"
HOY = date.today().isoformat()
CRUDO = AQUI / "datos" / "crudo" / "coordinador_electrico"
DATOS = AQUI / "datos"

PAUSA = 2.5          # segundos entre pedidos. Son respuestas de ~1 MB cada una
REINTENTOS = 5
TIEMPO_LIMITE = 240  # las respuestas grandes tardan


# ── qué se baja ──────────────────────────────────────────────────────────────

ENDPOINTS = {
    "subestaciones_base": dict(
        ruta="/v1/subestaciones/", esperado=1269, trae_coordenada=False,
        descripcion="El endpoint que el proyecto ya había probado. Se baja para "
                    "dejar constancia de que efectivamente NO trae coordenadas, "
                    "y para poder contrastar el universo con el extendido."),
    "subestaciones": dict(
        ruta="/v1/subestaciones/extended/", esperado=1273, trae_coordenada=True,
        descripcion="★ El que resuelve el problema: mismas subestaciones CON "
                    "latitud, longitud, capacidad en MVA, configuración de "
                    "barra y bandera de SCADA."),
    "centrales": dict(
        ruta="/v1/centrales/extended/", esperado=1218, trae_coordenada=True,
        descripcion="Centrales de generación con potencia máxima y tipo."),
    "taps": dict(
        ruta="/v1/taps/extended/", esperado=277, trae_coordenada=True,
        descripcion="Derivaciones de línea (taps), con coordenada."),
    "unidades_generadoras": dict(
        ruta="/v1/unidades-generadoras/", esperado=None, trae_coordenada=False,
        descripcion="★★ Lo que desambigua las centrales. `centrales` sólo "
                    "distingue cinco tipos —Solares, Termoeléctricas, "
                    "Hidroeléctricas, Eólicas, Geotérmica— y la Matriz separa "
                    "carbón, gas y diésel por un lado, y embalse, pasada y "
                    "bombeo por otro. Cada unidad generadora trae su "
                    "`tipo_tecnologia_nombre` y su `id_combustible`, y con eso "
                    "las 214 termoeléctricas y las 182 hidroeléctricas dejan de "
                    "ser una bolsa."),
    "combustibles": dict(
        ruta="/v1/combustibles/", esperado=None, trae_coordenada=False,
        descripcion="Catálogo de combustibles al que apunta `id_combustible`."),
    "lineas": dict(
        ruta="/v1/lineas/", esperado=1241, trae_coordenada=False,
        descripcion="Líneas de transmisión. SIN geometría."),
    "secciones_tramos": dict(
        ruta="/v1/secciones-tramos/extended/", esperado=2926,
        trae_coordenada=False,
        descripcion="Tramos de línea con tensión, capacidad, km de conductor y "
                    "sus dos extremos por nombre. El campo `coordenadas` viene "
                    "vacío en todos: 40.643 km sin traza."),
}

FUENTE = dict(
    id="coordinador_electrico_infotecnica",
    organismo="Coordinador Eléctrico Nacional — plataforma Infotécnica",
    producto="Interfaz pública Infotécnica del Sistema Eléctrico Nacional",
    url=BASE,
    especificacion=f"{BASE}/?format=openapi",
    formato="JSON",
    familia="ESTADO (organismo técnico independiente, mandato de la Ley 20.936)",
    acceso="anonimo",
    acceso_verificado=1,
    robots_txt="api-infotecnica.coordinador.cl/robots.txt → 404 (sin "
               "restricciones declaradas). www.coordinador.cl/robots.txt → sí "
               "declara: Content-Signal search=yes, ai-train=no, "
               "use=reference; Disallow:/ para agentes de IA. Verificado "
               "19-ago-2026.",
    condiciones_uso="Servicio público sin credenciales, documentado con "
                    "OpenAPI. Uso como REFERENCIA en investigación, sin "
                    "entrenamiento de modelos. PENDIENTE: confirmación escrita "
                    "del Coordinador.",
    permite_automatizacion="si (interfaz pública documentada), con la salvedad "
                           "de robots.txt del sitio institucional anotada arriba",
    granularidad="instalación (subestación / central / tap / línea)",
)


# ── acceso ───────────────────────────────────────────────────────────────────

def _pedir(ruta, intentos=REINTENTOS):
    """Un pedido, con reintentos y espera creciente. Devuelve (json, bytes_crudos)."""
    url = BASE + ruta
    espera = 4
    for intento in range(1, intentos + 1):
        try:
            pedido = urllib.request.Request(url)
            pedido.add_header("Accept", "application/json")
            pedido.add_header("Accept-Encoding", "gzip")
            pedido.add_header(
                "User-Agent",
                "matriz-infraestructura-critica-clima/1.0 "
                "(investigacion academica; inventario de infraestructura "
                "critica; sin entrenamiento de modelos)")
            with urllib.request.urlopen(pedido, timeout=TIEMPO_LIMITE) as resp:
                bruto = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    bruto = gzip.decompress(bruto)
            return json.loads(bruto.decode("utf-8")), bruto
        except Exception as err:                       # noqa: BLE001
            if intento == intentos:
                raise
            print(f"      reintento {intento}/{intentos - 1} tras {espera}s "
                  f"({type(err).__name__}: {str(err)[:120]})")
            time.sleep(espera)
            espera *= 2
    raise RuntimeError("inalcanzable")


# ── 1. exploración ───────────────────────────────────────────────────────────

def explorar():
    """Cuenta, por endpoint, cuántos registros hay y cuántos traen coordenada.

    Es la comprobación que faltaba: pone lado a lado `/subestaciones/` y
    `/subestaciones/extended/` para que se vea que el problema no era la fuente
    sino el endpoint elegido."""
    print(f"Coordinador Eléctrico Nacional · Infotécnica · {BASE}")
    print(f"Especificación completa: {FUENTE['especificacion']}\n")
    for clave, e in ENDPOINTS.items():
        try:
            datos, _ = _pedir(e["ruta"], intentos=2)
        except Exception as err:                       # noqa: BLE001
            print(f"  {clave:20s} NO RESPONDE ({str(err)[:70]})")
            continue
        n = len(datos) if isinstance(datos, list) else "?"
        con = sum(1 for x in datos
                  if isinstance(x, dict)
                  and x.get("latitud") is not None
                  and x.get("longitud") is not None) if isinstance(datos, list) else 0
        marca = "" if n == e["esperado"] else f"  ★ el módulo esperaba {e['esperado']}"
        print(f"  {clave:20s} {n:>6} reg · {con:>6} con coordenada{marca}")
        time.sleep(PAUSA)


# ── 2. bajada del crudo ──────────────────────────────────────────────────────

def bajar(clave):
    """Baja un endpoint y guarda el crudo tal como llegó, comprimido, con su
    procedencia al lado. El crudo se guarda ANTES de procesar nada."""
    e = ENDPOINTS[clave]
    destino = CRUDO / HOY
    destino.mkdir(parents=True, exist_ok=True)
    print(f"\n[{clave}] {e['ruta']}")
    datos, bruto = _pedir(e["ruta"])
    n = len(datos) if isinstance(datos, list) else 0
    con = sum(1 for x in datos if isinstance(x, dict)
              and x.get("latitud") is not None) if isinstance(datos, list) else 0
    print(f"   {n} registros (el módulo esperaba {e['esperado']}) · "
          f"{con} con coordenada")

    with gzip.open(destino / f"{clave}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(datos, fh, ensure_ascii=False)
    (destino / f"{clave}_procedencia.json").write_text(json.dumps({
        "url": BASE + e["ruta"],
        "descripcion": e["descripcion"],
        "fecha_descarga": HOY,
        "registros": n,
        "registros_con_coordenada": con,
        "bytes_recibidos": len(bruto),
        "fuente": FUENTE,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   → datos/crudo/coordinador_electrico/{HOY}/{clave}.json.gz")
    return datos


def bajar_todo(claves=None):
    for clave in (claves or list(ENDPOINTS)):
        bajar(clave)
        time.sleep(PAUSA)


# ── 3. lectura ───────────────────────────────────────────────────────────────

def _ultima_bajada():
    fechas = sorted(p.name for p in CRUDO.glob("*") if p.is_dir())
    if not fechas:
        raise FileNotFoundError(f"no hay nada bajado en {CRUDO}")
    return fechas[-1]


def _leer_crudo(clave, fecha=None):
    fecha = fecha or _ultima_bajada()
    ruta = CRUDO / fecha / f"{clave}.json.gz"
    if not ruta.exists():
        raise FileNotFoundError(f"falta {ruta} — correr `bajar {clave}` primero")
    with gzip.open(ruta, "rt", encoding="utf-8") as fh:
        return json.load(fh), fecha


def _texto(valor):
    if valor is None:
        return ""
    return str(valor).strip()


def _numero(valor):
    if valor is None or valor == "":
        return ""
    try:
        n = float(valor)
    except (TypeError, ValueError):
        return ""
    if n != n:
        return ""
    return int(n) if n == int(n) else round(n, 4)


def _plausible(lat, lon):
    """Chile continental, insular y antártico, con margen.

    Una coordenada fuera de la caja NO se corrige ni se borra en silencio: se
    marca `fuera_de_chile` y se cuenta aparte. Si un día aparecen muchas, es
    señal de que la fuente cambió de sistema de referencia, y eso hay que verlo,
    no taparlo."""
    if lat is None or lon is None:
        return False
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return -90.0 <= lat <= -17.0 and -110.0 <= lon <= -66.0


CAMPOS_COMUNES = ["id_activo", "item_micr", "nombre", "lat", "lon",
                  "comuna", "provincia", "region", "cut", "zona_geografica",
                  "operador", "fuente", "fecha_captura", "confianza_ubicacion"]


def _ubicar(terr, lat, lon):
    """Comuna, provincia, región y zona DERIVADAS del polígono, no copiadas del
    texto que trae la fuente. Regla H-16 del proyecto."""
    vacio = {"comuna": "", "provincia": "", "region": "", "codigo_comuna": "",
             "zona_geografica": "", "faltan": "sin_coordenada"}
    if lat is None or terr is None:
        return vacio
    r = terr.ubicar(lat, lon)
    return {"comuna": r["comuna"] or "", "provincia": r["provincia"] or "",
            "region": r["region"] or "", "codigo_comuna": r["codigo_comuna"] or "",
            "zona_geografica": r["zona_geografica"] or "",
            "faltan": "|".join(r["faltan"])}


def _escribir(ruta, filas, campos):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    print(f"   → {ruta.relative_to(AQUI)}  ({len(filas)} filas)")


# ── 4. las tablas ────────────────────────────────────────────────────────────

def tabla_subestaciones(fecha=None, terr=None):
    """Las 1.273 subestaciones, con coordenada donde el Coordinador la publica.

    Se cuenta y se declara cuántas quedan sin ella: son las que habrá que pedir
    formalmente al operador, y son POCAS — es una lista corta y accionable, no
    un problema abierto.
    """
    datos, fecha = _leer_crudo("subestaciones", fecha)
    terr = terr if terr is not None else territorio.Territorio()
    filas, sin_coord, fuera = [], [], 0

    for s in datos:
        lat, lon = s.get("latitud"), s.get("longitud")
        if lat is not None and lon is not None and not _plausible(lat, lon):
            fuera += 1
            confianza = "fuera_de_chile_no_usada"
            lat = lon = None
        elif lat is None or lon is None:
            confianza = "sin_coordenada"
            lat = lon = None
        else:
            confianza = "coordenada_publicada_por_el_operador"
        if lat is None:
            sin_coord.append(_texto(s.get("nombre")))
        u = _ubicar(terr, lat, lon)
        filas.append({
            "id_activo": f"SE-{s.get('codigo') or s.get('id')}",
            "item_micr": "subestaciones_electricas",
            "nombre": _texto(s.get("nombre")),
            "lat": lat if lat is not None else "",
            "lon": lon if lon is not None else "",
            "comuna": u["comuna"], "provincia": u["provincia"],
            "region": u["region"], "cut": u["codigo_comuna"],
            "zona_geografica": u["zona_geografica"],
            "operador": _texto(s.get("propietario_nombre")),
            "fuente": "Coordinador Eléctrico Nacional · Infotécnica · "
                      "/v1/subestaciones/extended/",
            "fecha_captura": fecha,
            "confianza_ubicacion": confianza,
            # ── propios ──
            "codigo": _texto(s.get("codigo")),
            "nemotecnico": _texto(s.get("nemotecnico")),
            "grupo_propietario": _texto(s.get("grupo_nombre")),
            "capacidad_mva": _numero(s.get("capacidad_mva")),
            "tension_kv": _numero(s.get("tension_kv")),
            "tipo_configuracion": _texto(s.get("tipo_configuracion")),
            "tiene_scada": "si" if s.get("scada") else "no",
            "tiene_pararrayos": "si" if s.get("pararrayos") else "no",
            "equipos_compensacion": "si" if s.get("equipos_compensacion") else "no",
            "fecha_inicio_operacion": _texto(s.get("fecha_inicio_operacion")),
            "estado": _texto(s.get("estado")),
            "comuna_declarada": _texto(s.get("comuna_nombre")),
            "region_declarada": _texto(s.get("region_nombre")),
            "territorio_faltan": u["faltan"],
        })

    campos = CAMPOS_COMUNES + [
        "codigo", "nemotecnico", "grupo_propietario", "capacidad_mva",
        "tension_kv", "tipo_configuracion", "tiene_scada", "tiene_pararrayos",
        "equipos_compensacion", "fecha_inicio_operacion", "estado",
        "comuna_declarada", "region_declarada", "territorio_faltan"]
    _escribir(DATOS / "inventario_subestaciones_electricas.csv", filas, campos)
    print(f"     con coordenada: {len(filas) - len(sin_coord)} de {len(filas)} "
          f"({100 * (len(filas) - len(sin_coord)) / max(len(filas), 1):.1f} %)")
    print(f"     sin coordenada: {len(sin_coord)} — lista corta para pedir al "
          f"operador")
    if fuera:
        print(f"     ★ {fuera} con coordenada fuera de Chile: NO se usaron")
    return filas


def tabla_centrales(fecha=None, terr=None):
    """Las centrales de generación, con su potencia máxima en megavatios."""
    datos, fecha = _leer_crudo("centrales", fecha)
    terr = terr if terr is not None else territorio.Territorio()
    filas, sin_coord = [], 0
    for c in datos:
        lat, lon = c.get("latitud"), c.get("longitud")
        if not _plausible(lat, lon):
            lat = lon = None
            sin_coord += 1
            confianza = "sin_coordenada"
        else:
            confianza = "coordenada_publicada_por_el_operador"
        u = _ubicar(terr, lat, lon)
        filas.append({
            "id_activo": f"CEN-{c.get('codigo') or c.get('id')}",
            "item_micr": "centrales_de_generacion_electrica",
            "nombre": _texto(c.get("nombre")),
            "lat": lat if lat is not None else "",
            "lon": lon if lon is not None else "",
            "comuna": u["comuna"], "provincia": u["provincia"],
            "region": u["region"], "cut": u["codigo_comuna"],
            "zona_geografica": u["zona_geografica"],
            "operador": _texto(c.get("propietario_nombre")),
            "fuente": "Coordinador Eléctrico Nacional · Infotécnica · "
                      "/v1/centrales/extended/",
            "fecha_captura": fecha,
            "confianza_ubicacion": confianza,
            "codigo": _texto(c.get("codigo")),
            "nemotecnico": _texto(c.get("nemotecnico")),
            "grupo_propietario": _texto(c.get("grupo_nombre")),
            "tipo_central": _texto(c.get("tipo_central_nombre")),
            "clasificacion": _texto(c.get("clasificacion")),
            "potencia_maxima_mw": _numero(c.get("potencia_maxima")),
            "potencia_minima_mw": _numero(c.get("potencia_minima")),
            "numero_unidades": _numero(c.get("numero_unidades")),
            "fecha_inicio_operacion": _texto(c.get("fecha_inicio_operacion")),
            "estado": _texto(c.get("estado")),
            "comuna_declarada": _texto(c.get("comuna_nombre")),
            "region_declarada": _texto(c.get("region_nombre")),
            "territorio_faltan": u["faltan"],
        })
    campos = CAMPOS_COMUNES + [
        "codigo", "nemotecnico", "grupo_propietario", "tipo_central",
        "clasificacion", "potencia_maxima_mw", "potencia_minima_mw",
        "numero_unidades", "fecha_inicio_operacion", "estado",
        "comuna_declarada", "region_declarada", "territorio_faltan"]
    _escribir(DATOS / "inventario_centrales_electricas.csv", filas, campos)
    print(f"     sin coordenada usable: {sin_coord} de {len(filas)}")
    mw = sum(f["potencia_maxima_mw"] for f in filas
             if isinstance(f["potencia_maxima_mw"], (int, float)))
    print(f"     potencia máxima catastrada: {mw:,.0f} MW")
    return filas


def tabla_taps(fecha=None, terr=None):
    """Las derivaciones de línea. Son puntos de conexión: si uno cae, cae el
    cliente colgado de él, no el sistema."""
    datos, fecha = _leer_crudo("taps", fecha)
    terr = terr if terr is not None else territorio.Territorio()
    filas, sin_coord = [], 0
    for t in datos:
        lat, lon = t.get("latitud"), t.get("longitud")
        if not _plausible(lat, lon):
            lat = lon = None
            sin_coord += 1
            confianza = "sin_coordenada"
        else:
            confianza = "coordenada_publicada_por_el_operador"
        u = _ubicar(terr, lat, lon)
        filas.append({
            "id_activo": f"TAP-{t.get('codigo') or t.get('id')}",
            "item_micr": "derivaciones_de_linea_electrica",
            "nombre": _texto(t.get("nombre")),
            "lat": lat if lat is not None else "",
            "lon": lon if lon is not None else "",
            "comuna": u["comuna"], "provincia": u["provincia"],
            "region": u["region"], "cut": u["codigo_comuna"],
            "zona_geografica": u["zona_geografica"],
            "operador": _texto(t.get("propietario_nombre")),
            "fuente": "Coordinador Eléctrico Nacional · Infotécnica · "
                      "/v1/taps/extended/",
            "fecha_captura": fecha,
            "confianza_ubicacion": confianza,
            "codigo": _texto(t.get("codigo")),
            "nemotecnico": _texto(t.get("nemotecnico")),
            "circuito": _texto(t.get("circuito")),
            "tiene_scada": "si" if t.get("scada") else "no",
            "estado": _texto(t.get("estado")),
            "territorio_faltan": u["faltan"],
        })
    campos = CAMPOS_COMUNES + ["codigo", "nemotecnico", "circuito",
                               "tiene_scada", "estado", "territorio_faltan"]
    _escribir(DATOS / "inventario_taps_electricos.csv", filas, campos)
    print(f"     sin coordenada usable: {sin_coord} de {len(filas)}")
    return filas


def tabla_lineas(fecha=None):
    """Los tramos de línea de transmisión. SIN COORDENADA, y se dice.

    Se escribe igual porque el km de conductor y la tensión son datos reales que
    el proyecto necesita, y porque los dos extremos con nombre permiten unir
    cada tramo a subestaciones que SÍ están ubicadas. La geometría de la línea
    no se inventa.
    """
    datos, fecha = _leer_crudo("secciones_tramos", fecha)
    con_coord = sum(1 for x in datos if x.get("coordenadas"))
    filas = []
    for t in datos:
        filas.append({
            "id_activo": f"LT-{t.get('codigo') or t.get('id')}",
            "item_micr": "lineas_de_transmision_electrica",
            "nombre": _texto(t.get("nombre")),
            "lat": "", "lon": "",
            "comuna": "", "provincia": "", "region": "", "cut": "",
            "zona_geografica": "",
            "operador": _texto(t.get("propietario_nombre")),
            "fuente": "Coordinador Eléctrico Nacional · Infotécnica · "
                      "/v1/secciones-tramos/extended/",
            "fecha_captura": fecha,
            "confianza_ubicacion": "SIN GEOMETRIA — el campo `coordenadas` "
                                   "viene vacío en los 2.926 registros",
            "linea": _texto(t.get("linea_nombre")),
            "tramo": _texto(t.get("tramo_nombre")),
            "extremo_1": _texto(t.get("tramo_extremo1")),
            "extremo_2": _texto(t.get("tramo_extremo2")),
            "circuito": _texto(t.get("circuito")),
            "tension_kv": _numero(t.get("tension_nominal")),
            "capacidad_mva": _numero(t.get("capacidad_mva")),
            "longitud_conductor_km": _numero(t.get("longitud_conductor")),
            "tipo_tramo": _texto(t.get("tipo_tramo_nombre")),
            "estado": _texto(t.get("estado")),
        })
    campos = CAMPOS_COMUNES + [
        "linea", "tramo", "extremo_1", "extremo_2", "circuito", "tension_kv",
        "capacidad_mva", "longitud_conductor_km", "tipo_tramo", "estado"]
    _escribir(DATOS / "inventario_lineas_transmision_SIN_GEOMETRIA.csv",
              filas, campos)
    km = sum(f["longitud_conductor_km"] for f in filas
             if isinstance(f["longitud_conductor_km"], (int, float)))
    print(f"     ★ {km:,.0f} km de línea de transmisión SIN traza "
          f"({con_coord} de {len(filas)} tramos traen `coordenadas`)")
    return filas


# ── 5. contraste con lo que el proyecto ya tenía ─────────────────────────────

def contrastar_con_las_39(filas_nuevas):
    """Las 39 subestaciones capturadas a mano fueron el piloto del proyecto.
    Ahora hay 1.226 con coordenada del propio operador. Vale la pena medir
    cuánto se desviaban las de antes: es la única validación independiente
    disponible de una captura manual.
    """
    viejo = DATOS / "subestaciones_puntos.csv"
    if not viejo.exists():
        print(f"\n   no está {viejo.name}: no se puede contrastar")
        return None
    antiguas = list(csv.DictReader(viejo.open(encoding="utf-8")))
    por_nombre = {}
    for f in filas_nuevas:
        if f["lat"] != "":
            por_nombre[_normalizar(f["nombre"])] = f

    print(f"\nCONTRASTE con las {len(antiguas)} subestaciones del piloto")
    calzan, distancias, no_calzan = 0, [], []
    for a in antiguas:
        clave = _normalizar(a["subestacion"])
        nuevo = por_nombre.get(clave)
        if nuevo is None:
            no_calzan.append(a["subestacion"])
            continue
        calzan += 1
        d = _metros(float(a["lat"]), float(a["lon"]),
                    float(nuevo["lat"]), float(nuevo["lon"]))
        distancias.append((d, a["subestacion"]))
    print(f"   calzan por nombre: {calzan} de {len(antiguas)}")
    if distancias:
        distancias.sort(reverse=True)
        mediana = sorted(d for d, _ in distancias)[len(distancias) // 2]
        print(f"   distancia mediana entre la coordenada del piloto y la del "
              f"operador: {mediana / 1000:.1f} km")
        print("   las cinco que más se desvían:")
        for d, nombre in distancias[:5]:
            print(f"      {nombre[:45]:45s} {d / 1000:8.1f} km")
    if no_calzan:
        print(f"   no calzan por nombre ({len(no_calzan)}): "
              f"{', '.join(no_calzan[:6])}…")
    return {"calzan": calzan, "total_piloto": len(antiguas)}


def _normalizar(nombre):
    """Nombres para comparar: sin tildes, sin prefijos, sin paréntesis.

    Es una ayuda para el CONTRASTE, no un método de georreferenciación. Cruzar
    por nombre normalizado sirve para ver cuántos coinciden; NUNCA para asignarle
    a un activo la coordenada de otro que se llama parecido.
    """
    t = (nombre or "").lower()
    for a, b in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"),
                 ("ü", "u"), ("ñ", "n")):
        t = t.replace(a, b)
    if "(" in t:
        t = t[:t.index("(")]
    for prefijo in ("subestacion ", "s/e ", "se ", "s.e. "):
        if t.startswith(prefijo):
            t = t[len(prefijo):]
    return " ".join(t.split())


def _metros(lat1, lon1, lat2, lon2):
    """Distancia aproximada, equirrectangular. A escala de kilómetros sobra y no
    arrastra dependencias."""
    dlat = (lat1 - lat2) * 111320.0
    dlon = (lon1 - lon2) * 111320.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dlat, dlon)


# ── main ─────────────────────────────────────────────────────────────────────

def tablas(fecha=None):
    terr = territorio.Territorio()
    print("Capas de territorio:", terr.estado(), "\n")
    se = tabla_subestaciones(fecha, terr)
    tabla_centrales(fecha, terr)
    tabla_taps(fecha, terr)
    tabla_lineas(fecha)
    contrastar_con_las_39(se)


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
