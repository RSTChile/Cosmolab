"""
Adaptador de INVENTARIO — Sistema de Información Territorial del Ministerio de
Obras Públicas (SIT-MOP), las capas de activos que NO son vialidad.

QUÉ APORTA Y POR QUÉ
--------------------
El adaptador hermano `mop_vialidad.py` ya bajó de este mismo servidor los 14.039
tramos viales y los 6.742 puentes de la Dirección de Vialidad. Pero el SIT-MOP
publica mucho más que caminos, y nadie lo había mirado: en el mismo servicio
están el agua potable rural, la red aeroportuaria y las obras portuarias del
país, **todas con coordenada**.

Este módulo baja esas capas. Su valor no es el volumen sino la NATURALEZA de lo
que agrega:

★ AGUA POTABLE RURAL (APR) — el caso emblemático del proyecto.
  Son 2.475 sistemas que dan agua a localidades chicas y aisladas. La ley
  chilena de infraestructura crítica mira grandes operadores; estos sistemas
  quedan fuera por tamaño, y son justamente los que se cortan primero y los que
  nadie repone rápido. Es la convergencia con el paper de Castillo Jofré que el
  README del proyecto ya anota.
  Y traen tres campos que valen oro para el modelo de fallo en cascada:
    · `GRUP_ELEC`   ¿tiene grupo electrógeno? — si no, un corte eléctrico lo
                     deja sin agua: la cascada energía → agua, medida.
    · `SUF_GRUP`    ¿ese grupo alcanza para todo el sistema?
    · `CAMION_ALJ`  ¿depende de camión aljibe? — o sea, ¿ya está fallando?
  Más `BENEF_EST`: cuántas personas quedan sin agua si ese punto cae. Es la
  única capa del inventario que trae población servida por activo.

★ RED AEROPORTUARIA NACIONAL — 316 aeropuertos y aeródromos con coordenada.
  Y su capa 1 («Información de Territorio») trae, por aeródromo, los campos
  `ZONA_AISLADA` y `CATEGORIA_AISLAMIENTO`: el Estado ya clasificó qué
  aeródromos sirven a zonas aisladas. «Aislamiento» es exactamente el modo de
  falla que persigue este proyecto (el glosario oficial de Gestión del Riesgo de
  Desastres lo define así), y acá viene ya etiquetado por el propio ministerio.
  En una zona aislada el aeródromo NO es transporte: es el único acceso cuando
  el camino se corta. Es infraestructura crítica de reemplazo.

★ OBRAS PORTUARIAS — 644 caletas, muelles, varaderos y rampas.
  Ojo: es infraestructura portuaria menor de tuición del MOP, NO los puertos
  comerciales grandes (esos son de las empresas portuarias estatales y de
  DIRECTEMAR). Se declara así para que nadie confunda la cobertura.

★ EMBALSES — 1.370 embalses catastrados. Se bajan porque el ítem «Represas» es
  de prioridad `Pen = Muy Alta` en la Matriz. **Advertencia declarada por el
  propio servicio: el dato está actualizado a diciembre de 2015.** Un embalse no
  se mueve, pero once años sin actualizar significan que faltan los nuevos y que
  el estado operativo puede estar vencido. Se usa como ubicación, no como estado.

QUÉ SE BAJA
-----------
    clave                 servicio SIT-MOP                        registros
    apr                   DOH/APR capa 0                              2.475
    ssr_ley20998          DOH/SSR_Clasificados_Ley_20998 capa 0       2.475
    ssr_contratos         DOH/SSR_Contratos capa 0                      797
    embalses              DOH/Embalses capa 0                         1.370
    portuaria             DOP/CATASTRO_DOP capa 0                       644
    aeropuertos           DAP/Red_Aeroportuaria_Nacional capa 0         316
    aeropuertos_territorio  ídem capa 1 (tabla, sin geometría)          328
    aeropuertos_pistas      ídem capa 2 (tabla, sin geometría)          338

`apr` y `ssr_ley20998` traen el MISMO número de registros y el MISMO esquema de
campos: son, casi con seguridad, la misma base publicada dos veces (una con el
nombre viejo «Agua Potable Rural», otra con el nombre que le da la Ley 20.998 de
Servicios Sanitarios Rurales). El módulo baja las dos y las COMPARA registro a
registro, en vez de suponerlo. El resultado de esa comparación se imprime y se
guarda; sólo una de las dos entra al inventario, para no contar dos veces.

DIFERENCIA CON `ssr_contratos`
------------------------------
`ssr_contratos` NO es un inventario de sistemas existentes: son iniciativas de
obra (proyectos en distintas etapas). Un contrato no es un activo en operación.
Se baja porque anticipa dónde habrá infraestructura, pero se marca aparte y NO
se suma al inventario de activos operativos.

CÓMO SE PAGINA
--------------
El servidor corre ArcGIS 10.2.1 y, como ya se comprobó para vialidad,
**`resultOffset` no existe: el servidor lo acepta sin protestar y lo ignora**,
devolviendo siempre desde el principio. Paginar con offset acá no falla a gritos,
falla callado. La paginación va por identificador: primero `returnIdsOnly=true`
para pedir la lista completa de OBJECTID, después los registros en lotes por
`objectIds=...`, por POST (con muchos identificadores la URL de un GET supera lo
que el servidor acepta y contesta 404, otro fallo mudo). Así el conteo cierra por
construcción: se sabe cuántos se pidieron y cuántos llegaron.

Si un lote falla, se parte en dos hasta aislar el registro que el servicio no
sabe entregar, y ese registro queda anotado por identificador como «sin dato».
Un hueco con nombre y apellido, nunca un silencio.

LA COORDENADA SE VERIFICA CONTRA SÍ MISMA
-----------------------------------------
Las capas del agua potable rural y de aeropuertos traen la coordenada DOS veces:
en la geometría del rasgo y en los campos de atributo `LATITUD` / `LONGITUD`.
Este módulo pide la geometría reproyectada a EPSG:4326 (el servidor la publica en
SIRGAS-Chile, EPSG:5360 o 4674) y la compara con los atributos declarados. Si
discrepan más de ~100 m se anota. No es paranoia: una capa mal reproyectada
desplaza todo el inventario y el error es invisible mirando la tabla.

CONDICIONES DE USO — LEER ANTES DE PUBLICAR NADA
-------------------------------------------------
· `https://rest-sit.mop.gob.cl/robots.txt` devuelve **404**: el servidor no
  declara restricciones de rastreo. Verificado el 19-ago-2026.
· El servicio es público y anónimo, sin credenciales ni registro.
· El `copyrightText` que declara cada servicio nombra a su dirección: Dirección
  de Obras Hidráulicas, Dirección de Obras Portuarias, Dirección de Aeropuertos.
  Hay que atribuir a cada una en todo derivado.
· La Red Vial del mismo ministerio está publicada como Creative Commons
  Atribución-NoComercial (CC BY-NC). Para estas capas el portal no declara
  licencia explícita en el servicio; **se asume el mismo régimen y queda
  PENDIENTE confirmarlo por escrito con la Unidad de Gestión de Información
  Territorial (UGIT) del MOP**. No se ha verificado, y se dice.
· Ritmo prudente: pausa entre pedidos, lotes chicos, reintentos con espera
  creciente, `Accept-Encoding: gzip` para no pedirle al ministerio cuatro veces
  más bytes de los necesarios.

NO SE RECOLECTAN DATOS DE PERSONAS. La capa del agua potable rural trae `RUT` del
comité u organización que administra el sistema: es el rol de una persona
JURÍDICA, no de un individuo, pero por precaución **no se copia al inventario**.
Queda sólo en el crudo tal como lo entregó el ministerio.

USO
---
    python3 adaptadores/inventario_mop_sit.py explorar
    python3 adaptadores/inventario_mop_sit.py bajar          # todo
    python3 adaptadores/inventario_mop_sit.py bajar apr
    python3 adaptadores/inventario_mop_sit.py tablas         # crudo → CSV
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

BASE = "https://rest-sit.mop.gob.cl/arcgis/rest/services"
HOY = date.today().isoformat()
CRUDO = AQUI / "datos" / "crudo" / "mop_sit"
DATOS = AQUI / "datos"

PAUSA = 1.2          # segundos entre pedidos: es un servicio público
REINTENTOS = 6       # el servicio se queda mudo cada tantos lotes y vuelve solo


# ── qué se baja ──────────────────────────────────────────────────────────────

CAPAS = {
    "apr": dict(
        servicio="DOH/APR", capa=0, lote=400, esperado=2475,
        geometria="punto", direccion="Dirección de Obras Hidráulicas (DOH)",
        descripcion="Sistemas de Agua Potable Rural / Servicios Sanitarios "
                    "Rurales, cobertura nacional, con población beneficiaria, "
                    "grupo electrógeno y dependencia de camión aljibe."),
    "ssr_ley20998": dict(
        servicio="DOH/SSR_Clasificados_Ley_20998", capa=0, lote=400,
        esperado=2475, geometria="punto",
        direccion="Dirección de Obras Hidráulicas (DOH)",
        descripcion="Los mismos servicios sanitarios rurales, clasificados "
                    "según el artículo de la Ley 20.998. Se baja para "
                    "COMPARAR con `apr`, no para sumar."),
    "ssr_contratos": dict(
        servicio="DOH/SSR_Contratos", capa=0, lote=400, esperado=797,
        geometria="punto", direccion="Dirección de Obras Hidráulicas (DOH)",
        descripcion="Iniciativas de contrato de servicios sanitarios rurales. "
                    "PROYECTOS, no activos en operación."),
    "embalses": dict(
        servicio="DOH/Embalses", capa=0, lote=400, esperado=1370,
        geometria="punto", direccion="Dirección de Obras Hidráulicas (DOH)",
        descripcion="Catastro de embalses. ★ El propio servicio declara el "
                    "dato actualizado a diciembre de 2015."),
    "derechos_agua": dict(
        servicio="SNIA/SNIA_DerechoAprovechamiento", capa=0, lote=800,
        esperado=76488, geometria="punto",
        direccion="Dirección General de Aguas (DGA) · SNIA",
        descripcion="★★ Derechos de aprovechamiento de aguas del SNIA. Cada "
                    "registro declara el PUNTO donde se capta y si el recurso "
                    "es superficial o subterráneo. ⚠️ Es un registro de "
                    "DERECHOS, no un catastro de obras construidas: el punto "
                    "dice dónde está autorizada la captación."),
    "obras_mayores": dict(
        servicio="SNIA/SNIA_ObrasMayores", capa=0, lote=800, esperado=5405, geometria="punto",
        direccion="Dirección General de Aguas (DGA) · SNIA",
        descripcion="Obras hidráulicas mayores registradas en el SNIA."),
    "canales_cnr": dict(
        servicio="DOH/Canales_CNR", capa=0, lote=400, esperado=13166, geometria="linea",
        direccion="Comisión Nacional de Riego (CNR)",
        descripcion="Catastro de canales de la Comisión Nacional de Riego. "
                    "Geometría de línea: 13.166 canales con su fuente hídrica."),
    "portuaria": dict(
        servicio="DOP/CATASTRO_DOP", capa=0, lote=400, esperado=644,
        geometria="punto", direccion="Dirección de Obras Portuarias (DOP)",
        descripcion="Infraestructura portuaria menor de tuición MOP: caletas, "
                    "muelles, varaderos, rampas. NO los puertos comerciales."),
    "aeropuertos": dict(
        servicio="DAP/Red_Aeroportuaria_Nacional", capa=0, lote=300,
        esperado=316, geometria="punto",
        direccion="Dirección de Aeropuertos (DAP)",
        descripcion="Red aeroportuaria nacional: aeropuertos, aeródromos y "
                    "pequeños aeródromos, con código OACI e IATA."),
    "aeropuertos_territorio": dict(
        servicio="DAP/Red_Aeroportuaria_Nacional", capa=1, lote=400,
        esperado=328, geometria="tabla",
        direccion="Dirección de Aeropuertos (DAP)",
        descripcion="★ Tabla sin geometría, se une por COD_OACI. Trae "
                    "ZONA_AISLADA y CATEGORIA_AISLAMIENTO: el Estado ya "
                    "clasificó qué aeródromos sirven zonas aisladas."),
    "aeropuertos_pistas": dict(
        servicio="DAP/Red_Aeroportuaria_Nacional", capa=2, lote=400,
        esperado=338, geometria="tabla",
        direccion="Dirección de Aeropuertos (DAP)",
        descripcion="Tabla sin geometría, se une por OACI_COD. Largo, ancho, "
                    "carpeta y resistencia de cada pista."),
}

FUENTE = dict(
    id="mop_sit_inventario",
    organismo="Ministerio de Obras Públicas — Sistema de Información "
              "Territorial (SIT-MOP): Dirección de Obras Hidráulicas, "
              "Dirección de Obras Portuarias, Dirección de Aeropuertos",
    producto="Servicios ArcGIS REST del SIT-MOP: Agua Potable Rural, Servicios "
             "Sanitarios Rurales, Embalses, Catastro de Obras Portuarias, Red "
             "Aeroportuaria Nacional",
    url=BASE,
    formato="esriJSON (el servicio no soporta f=geojson; la conversión la hace "
            "este módulo explícitamente)",
    familia="ESTADO",
    acceso="anonimo",
    acceso_verificado=1,
    robots_txt="404 — el servidor no declara robots.txt (verificado 19-ago-2026)",
    condiciones_uso="Servicio público sin credenciales. Atribución obligatoria "
                    "a la dirección del MOP que publica cada capa. PENDIENTE "
                    "confirmar licencia por escrito con la UGIT del MOP; se "
                    "asume el régimen CC BY-NC de la Red Vial del mismo "
                    "ministerio, pero NO está verificado.",
    permite_automatizacion="si (servicio REST público, sin credenciales, sin "
                           "robots.txt que lo restrinja)",
    granularidad="activo puntual",
)


# ── acceso al servicio ───────────────────────────────────────────────────────

def _pedir(url, parametros, metodo="GET", intentos=None):
    """Un pedido al servicio, con reintentos y espera creciente.

    Si el servidor contesta un error de ArcGIS (que viene con código HTTP 200 y
    un campo `error` dentro del cuerpo) se levanta excepción: un error
    disfrazado de éxito es la peor forma de perder datos sin enterarse.
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


def _identificadores(clave):
    """La lista completa de OBJECTID. El servidor la entrega entera, sin tope:
    es lo que permite que el conteo cierre por construcción."""
    resp = _pedir(f"{_url_capa(clave)}/query",
                  {"where": "1=1", "returnIdsOnly": "true", "f": "json"})
    ids = resp.get("objectIds") or []
    return sorted(ids)


def _pedir_registros(clave, ids, sin_dato, crudos, con_geometria):
    """Pide un grupo de registros. Si el servicio se cae con ese grupo, lo PARTE
    en dos y reintenta, hasta aislar el registro que no sabe entregar.

    Las dos salidas malas serían abortar (y no tener el inventario) o pedir el
    lote y quedarse callado con los que no llegaron. La bisección hace la
    tercera: baja todo lo que el servicio sí puede dar y deja el imposible
    anotado por identificador.
    """
    parametros = {"objectIds": ",".join(map(str, ids)),
                  "outFields": "*", "f": "json"}
    if con_geometria:
        parametros["returnGeometry"] = "true"
        parametros["outSR"] = "4326"
    else:
        parametros["returnGeometry"] = "false"
    try:
        resp = _pedir(f"{_url_capa(clave)}/query", parametros, metodo="POST",
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
        izq = _pedir_registros(clave, ids[:mitad], sin_dato, crudos, con_geometria)
        time.sleep(PAUSA)
        der = _pedir_registros(clave, ids[mitad:], sin_dato, crudos, con_geometria)
        return izq + der


# ── 1. exploración ───────────────────────────────────────────────────────────

def explorar():
    """Describe cada capa antes de bajarla: cuántos registros dice tener el
    servicio, con qué geometría y en qué sistema de referencia los publica.

    Se corre primero para que el conteo esperado del módulo no sea una creencia
    heredada sino algo que el servicio acaba de confirmar."""
    print(f"SIT-MOP · {BASE}\n")
    filas = []
    for clave, c in CAPAS.items():
        url = _url_capa(clave)
        try:
            meta = _pedir(url, {"f": "json"}, intentos=2)
        except Exception as err:                       # noqa: BLE001
            print(f"  {clave:24s} NO RESPONDE ({str(err)[:70]})")
            filas.append((clave, "sin dato", "sin dato", "sin dato"))
            continue
        time.sleep(PAUSA)
        try:
            cnt = _pedir(f"{url}/query",
                         {"where": "1=1", "returnCountOnly": "true", "f": "json"},
                         intentos=2).get("count")
        except Exception:                              # noqa: BLE001
            cnt = None
        sr = (meta.get("extent") or {}).get("spatialReference", {})
        sr = sr.get("latestWkid") or sr.get("wkid") or "sin dato"
        geom = meta.get("geometryType") or "(tabla sin geometría)"
        marca = "" if cnt == c["esperado"] else f"  ★ el módulo esperaba {c['esperado']}"
        print(f"  {clave:24s} {str(cnt):>6} reg · {geom:20s} · EPSG:{sr}{marca}")
        filas.append((clave, cnt, geom, sr))
        time.sleep(PAUSA)
    return filas


# ── 2. bajada del crudo ──────────────────────────────────────────────────────

def _a_geojson(rasgos_esri, con_geometria):
    """esriJSON → GeoJSON. La conversión es explícita porque el servicio no
    ofrece `f=geojson`; hacerla a mano deja ver exactamente qué se transformó."""
    salida = []
    for r in rasgos_esri:
        props = dict(r.get("attributes") or {})
        geom = None
        if con_geometria:
            g = r.get("geometry") or {}
            if g.get("x") is not None and g.get("y") is not None:
                geom = {"type": "Point", "coordinates": [g["x"], g["y"]]}
            # ★ Las capas de línea (canales de riego) traen `paths`, no `x`/`y`.
            #   Sin esto los 13.166 canales bajaban con geometry en null y el
            #   archivo parecía correcto: 13.166 registros, cero geometrías.
            elif g.get("paths"):
                caminos = g["paths"]
                geom = ({"type": "LineString", "coordinates": caminos[0]}
                        if len(caminos) == 1
                        else {"type": "MultiLineString", "coordinates": caminos})
            elif g.get("rings"):
                geom = {"type": "Polygon", "coordinates": g["rings"]}
        salida.append({"type": "Feature", "geometry": geom, "properties": props})
    return salida


def bajar(clave):
    """Baja una capa completa y guarda el CRUDO tal como llegó, antes de tocar
    nada. Si después hay que auditar una decisión, tiene que poder
    reconstruirse exactamente lo que el ministerio contestó ese día."""
    c = CAPAS[clave]
    con_geometria = c["geometria"] != "tabla"
    destino = CRUDO / HOY
    destino.mkdir(parents=True, exist_ok=True)

    print(f"\n[{clave}] {c['servicio']} capa {c['capa']}")
    ids = _identificadores(clave)
    print(f"   {len(ids)} identificadores (el módulo esperaba {c['esperado']})")
    time.sleep(PAUSA)

    rasgos, sin_dato, crudos = [], [], []
    lote = c["lote"]
    total_lotes = math.ceil(len(ids) / lote)
    for i in range(0, len(ids), lote):
        trozo = ids[i:i + lote]
        print(f"   lote {i // lote + 1}/{total_lotes} ({len(trozo)} registros)")
        rasgos += _pedir_registros(clave, trozo, sin_dato, crudos, con_geometria)
        time.sleep(PAUSA)

    geojson = {
        "type": "FeatureCollection",
        "features": _a_geojson(rasgos, con_geometria),
        "procedencia": {
            "url": f"{_url_capa(clave)}/query",
            "servicio": c["servicio"], "capa": c["capa"],
            "direccion_mop": c["direccion"],
            "descripcion": c["descripcion"],
            "fecha_descarga": HOY,
            "identificadores_pedidos": len(ids),
            "registros_recibidos": len(rasgos),
            "identificadores_sin_dato": sin_dato,
            "sistema_referencia_pedido": "EPSG:4326" if con_geometria else "(tabla)",
            "fuente": FUENTE,
        },
    }
    ruta = destino / f"{clave}.geojson"
    ruta.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")

    ruta_esri = destino / f"{clave}_respuesta_cruda_esrijson.json.gz"
    with gzip.open(ruta_esri, "wt", encoding="utf-8") as fh:
        json.dump(crudos, fh, ensure_ascii=False)

    cierra = "✓ cierra" if len(rasgos) + len(sin_dato) == len(ids) else "★ NO CIERRA"
    print(f"   {len(rasgos)} recibidos + {len(sin_dato)} sin dato = "
          f"{len(rasgos) + len(sin_dato)} de {len(ids)} pedidos — {cierra}")
    print(f"   → {ruta.relative_to(AQUI)}")
    return geojson


def bajar_todo(claves=None):
    for clave in (claves or list(CAPAS)):
        bajar(clave)


# ── 3. lectura del crudo ─────────────────────────────────────────────────────

def _ultima_bajada():
    fechas = sorted(p.name for p in CRUDO.glob("*") if p.is_dir())
    if not fechas:
        raise FileNotFoundError(f"no hay nada bajado en {CRUDO}")
    return fechas[-1]


def _leer_crudo(clave, fecha=None):
    fecha = fecha or _ultima_bajada()
    ruta = CRUDO / fecha / f"{clave}.geojson"
    if not ruta.exists():
        raise FileNotFoundError(f"falta {ruta} — correr `bajar {clave}` primero")
    return json.loads(ruta.read_text(encoding="utf-8"))


def _texto(valor):
    """Normaliza a texto limpio. Nunca convierte un vacío en un cero ni al
    revés: `None` sale como cadena vacía, y eso significa «sin dato»."""
    if valor is None:
        return ""
    if isinstance(valor, float) and valor != valor:      # NaN
        return ""
    return str(valor).strip()


def _numero(valor):
    """Devuelve el número o cadena vacía. Un campo que no se puede leer NO
    vale cero: vale «sin dato», y así se escribe."""
    if valor is None or valor == "":
        return ""
    try:
        n = float(valor)
    except (TypeError, ValueError):
        return ""
    if n != n:
        return ""
    return int(n) if n == int(n) else n


def _si_no(valor):
    """Los campos «tiene grupo electrógeno / camión aljibe» vienen como texto
    libre. Se normaliza a si/no/sin dato SIN adivinar: lo que no se reconoce
    queda como el texto original, no como «no»."""
    t = _texto(valor).lower()
    if t in ("si", "sí", "s", "1", "true", "x"):
        return "si"
    if t in ("no", "n", "0", "false"):
        return "no"
    if t in ("", "s/i", "s/d", "sin informacion", "sin información", "null",
             "n/a", "no aplica", "-"):
        return ""
    return _texto(valor)


# ── 4. la coordenada, verificada contra sí misma ─────────────────────────────

def _coordenada(props, geometria):
    """Devuelve (lat, lon, origen, discrepancia_m).

    La geometría reproyectada por el servidor a EPSG:4326 es la que manda. Los
    atributos LATITUD/LONGITUD sirven de contraste: si difieren, se anota la
    distancia en metros. Una capa mal reproyectada desplaza TODO el inventario
    y el error no se ve mirando la tabla.
    """
    lat_geo = lon_geo = None
    if geometria and geometria.get("coordinates"):
        lon_geo, lat_geo = geometria["coordinates"][0], geometria["coordinates"][1]

    lat_att = props.get("LATITUD")
    lon_att = props.get("LONGITUD")
    try:
        lat_att = float(lat_att) if lat_att not in (None, "") else None
        lon_att = float(lon_att) if lon_att not in (None, "") else None
    except (TypeError, ValueError):
        lat_att = lon_att = None

    discrepancia = ""
    if None not in (lat_geo, lon_geo, lat_att, lon_att):
        # Equirrectangular: a esta escala (metros) sobra, y no arrastra
        # dependencias. 111.320 m por grado de latitud.
        dlat = (lat_geo - lat_att) * 111320.0
        dlon = (lon_geo - lon_att) * 111320.0 * math.cos(math.radians(lat_geo))
        discrepancia = round(math.hypot(dlat, dlon), 1)

    if lat_geo is not None and _plausible(lat_geo, lon_geo):
        return lat_geo, lon_geo, "geometria_servicio_epsg4326", discrepancia
    if lat_att is not None and _plausible(lat_att, lon_att):
        return lat_att, lon_att, "atributo_LATITUD_LONGITUD", discrepancia
    return None, None, "sin_coordenada", discrepancia


def _plausible(lat, lon):
    """Chile continental, insular y antártico, con margen. Un punto fuera de la
    caja NO se corrige ni se descarta en silencio: se marca como sin coordenada
    y se cuenta aparte, para que el número salga en el informe."""
    if lat is None or lon is None:
        return False
    return -90.0 <= lat <= -17.0 and -110.0 <= lon <= -66.0


# ── 5. las tablas del inventario ─────────────────────────────────────────────

CAMPOS_COMUNES = ["id_activo", "item_micr", "nombre", "lat", "lon",
                  "comuna", "provincia", "region", "cut", "zona_geografica",
                  "operador", "fuente", "fecha_captura", "confianza_ubicacion"]


def _ubicar(terr, lat, lon):
    """Comuna, provincia, región y zona geográfica DERIVADAS de la coordenada,
    no copiadas del atributo. La regla H-16 del proyecto: el texto que trae la
    fuente puede estar viejo o mal escrito; el polígono no opina.

    Se devuelven las dos cosas —lo derivado y lo declarado— para poder medir
    cuánto discrepan, que es un indicador de calidad de la capa."""
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


def tabla_apr(fecha=None, terr=None):
    """Los sistemas de agua potable rural, normalizados al esquema común.

    Ítem de la Matriz: agua potable rural (familia Hídrico / Sistemas de Agua
    Potable y Alcantarillado).
    """
    crudo = _leer_crudo("apr", fecha)
    terr = terr if terr is not None else territorio.Territorio()
    filas, sin_coord, discrepantes = [], 0, 0

    for f in crudo["features"]:
        p = f["properties"]
        lat, lon, origen, disc = _coordenada(p, f.get("geometry"))
        if lat is None:
            sin_coord += 1
        if isinstance(disc, (int, float)) and disc > 100:
            discrepantes += 1
        u = _ubicar(terr, lat, lon)
        filas.append({
            "id_activo": f"APR-{_texto(p.get('ID_IDE')) or p.get('OBJECTID')}",
            "item_micr": "agua_potable_rural",
            "nombre": _texto(p.get("NOMBRE_SSR")),
            "lat": lat if lat is not None else "",
            "lon": lon if lon is not None else "",
            "comuna": u["comuna"], "provincia": u["provincia"],
            "region": u["region"], "cut": u["codigo_comuna"],
            "zona_geografica": u["zona_geografica"],
            "operador": _texto(p.get("TIPO_ORG")),
            "fuente": "MOP · Dirección de Obras Hidráulicas · DOH/APR",
            "fecha_captura": crudo["procedencia"]["fecha_descarga"],
            "confianza_ubicacion": origen,
            # ── propios del activo ──
            "comuna_declarada": _texto(p.get("COMUNA")),
            "region_declarada": _texto(p.get("REGION")),
            "urbano_rural": _texto(p.get("URBAN_RURAL")),
            "ano_puesta_marcha": _texto(p.get("ANO_PTA_MCH")),
            "beneficiarios_estimados": _numero(p.get("BENEF_EST")),
            "arranques": _numero(p.get("CANT_ARR")),
            "unidades_domiciliarias": _numero(p.get("CANT_UD")),
            "estanques": _texto(p.get("CANT_EST")),
            "captaciones": _texto(p.get("CANT_CAPT")),
            "clasificacion_ley_20998": _texto(p.get("CLAS_OP")),
            "tiene_grupo_electrogeno": _si_no(p.get("GRUP_ELEC")),
            "grupo_electrogeno_suficiente": _si_no(p.get("SUF_GRUP")),
            "depende_camion_aljibe": _si_no(p.get("CAMION_ALJ")),
            "tiene_telemetria": _si_no(p.get("TELEMETRIA")),
            "derecho_aprovechamiento_aguas": _texto(p.get("DAA")),
            "punto_referencia_coordenada": _texto(p.get("PTO_REF")),
            "levantamiento": _texto(p.get("LEVANTAMIENTO")),
            "fuente_declarada": _texto(p.get("FUENTE")),
            "discrepancia_geom_vs_atributo_m": disc,
            "territorio_faltan": u["faltan"],
        })

    campos = CAMPOS_COMUNES + [
        "comuna_declarada", "region_declarada", "urbano_rural",
        "ano_puesta_marcha", "beneficiarios_estimados", "arranques",
        "unidades_domiciliarias", "estanques", "captaciones",
        "clasificacion_ley_20998", "tiene_grupo_electrogeno",
        "grupo_electrogeno_suficiente", "depende_camion_aljibe",
        "tiene_telemetria", "derecho_aprovechamiento_aguas",
        "punto_referencia_coordenada", "levantamiento", "fuente_declarada",
        "discrepancia_geom_vs_atributo_m", "territorio_faltan"]
    _escribir(DATOS / "inventario_agua_potable_rural.csv", filas, campos)
    print(f"     sin coordenada usable: {sin_coord} de {len(filas)}")
    print(f"     coordenada geométrica vs atributo, discrepancia >100 m: "
          f"{discrepantes}")
    return filas


def tabla_aeropuertos(fecha=None, terr=None):
    """La red aeroportuaria, unida a sus dos tablas de atributos.

    La unión con `aeropuertos_territorio` es la que importa para este proyecto:
    trae `ZONA_AISLADA` y `CATEGORIA_AISLAMIENTO`. Se une por código OACI, y los
    que no calzan se declaran, no se rellenan.
    """
    crudo = _leer_crudo("aeropuertos", fecha)
    terr = terr if terr is not None else territorio.Territorio()

    territorio_por_oaci, pistas_por_oaci = {}, {}
    for clave, destino, campo in (("aeropuertos_territorio", territorio_por_oaci,
                                   "OACI_COD"),
                                  ("aeropuertos_pistas", pistas_por_oaci,
                                   "OACI_COD")):
        try:
            for f in _leer_crudo(clave, fecha)["features"]:
                cod = _texto(f["properties"].get(campo)).upper()
                if cod:
                    destino.setdefault(cod, f["properties"])
        except FileNotFoundError:
            print(f"   ★ falta el crudo de {clave}: esas columnas saldrán vacías")

    filas, sin_coord, sin_union = [], 0, 0
    for f in crudo["features"]:
        p = f["properties"]
        lat, lon, origen, disc = _coordenada(p, f.get("geometry"))
        if lat is None:
            sin_coord += 1
        u = _ubicar(terr, lat, lon)
        oaci = _texto(p.get("COD_OACI")).upper()
        t = territorio_por_oaci.get(oaci)
        pista = pistas_por_oaci.get(oaci)
        if t is None:
            sin_union += 1
        filas.append({
            "id_activo": f"AER-{oaci or p.get('OBJECTID')}",
            "item_micr": "aeropuertos_y_aerodromos",
            "nombre": _texto(p.get("NOMBRE")),
            "lat": lat if lat is not None else "",
            "lon": lon if lon is not None else "",
            "comuna": u["comuna"], "provincia": u["provincia"],
            "region": u["region"], "cut": u["codigo_comuna"],
            "zona_geografica": u["zona_geografica"],
            "operador": _texto(p.get("ADMINISTRACION")),
            "fuente": "MOP · Dirección de Aeropuertos · DAP/"
                      "Red_Aeroportuaria_Nacional",
            "fecha_captura": crudo["procedencia"]["fecha_descarga"],
            "confianza_ubicacion": origen,
            # ── propios ──
            "codigo_oaci": oaci,
            "codigo_iata": _texto(p.get("CODIGO_IATA")),
            "red": _texto(p.get("RED")),
            "tipo": _texto(p.get("TIPO")),
            "uso": _texto(p.get("USO")),
            "propiedad": _texto(p.get("PROPIEDAD")),
            "horas_operacion": _texto(p.get("HORAS_OPERA")),
            "localidad": _texto(p.get("LOCALIDAD")),
            "comuna_declarada": _texto(p.get("COMUNA")),
            "region_declarada": _texto(p.get("REGION")),
            # ★ lo que hace crítico a un aeródromo en este proyecto
            "zona_aislada": _si_no((t or {}).get("ZONA_AISLADA")),
            "categoria_aislamiento": _texto((t or {}).get("CATEGORIA_AISLAMIENTO")),
            "zona_extrema": _si_no((t or {}).get("ZONA_EXTREMA")),
            "comuna_fronteriza": _si_no((t or {}).get("COMUNA_FRONTERIZA")),
            "zona_rezagada": _si_no((t or {}).get("ZONA_REZAGADA")),
            "largo_pista_m": _numero((pista or {}).get("LONGTITUD_PISTA")),
            "ancho_pista_m": _numero((pista or {}).get("ANCHO_PISTA")),
            "carpeta_pista": _texto((pista or {}).get("CARPETA")),
            "tipo_pavimento": _texto((pista or {}).get("TIPO_PAVIMENTO")),
            "msnm": _numero((pista or {}).get("MSNM")),
            "union_territorio": "si" if t else "sin dato",
            "union_pista": "si" if pista else "sin dato",
            "discrepancia_geom_vs_atributo_m": disc,
            "territorio_faltan": u["faltan"],
        })

    campos = CAMPOS_COMUNES + [
        "codigo_oaci", "codigo_iata", "red", "tipo", "uso", "propiedad",
        "horas_operacion", "localidad", "comuna_declarada", "region_declarada",
        "zona_aislada", "categoria_aislamiento", "zona_extrema",
        "comuna_fronteriza", "zona_rezagada", "largo_pista_m", "ancho_pista_m",
        "carpeta_pista", "tipo_pavimento", "msnm", "union_territorio",
        "union_pista", "discrepancia_geom_vs_atributo_m", "territorio_faltan"]
    _escribir(DATOS / "inventario_aeropuertos.csv", filas, campos)
    print(f"     sin coordenada usable: {sin_coord} de {len(filas)}")
    print(f"     sin unión con la tabla de territorio (código OACI que no "
          f"calza): {sin_union}")
    return filas


def tabla_portuaria(fecha=None, terr=None):
    """Las obras portuarias menores de tuición del MOP."""
    crudo = _leer_crudo("portuaria", fecha)
    terr = terr if terr is not None else territorio.Territorio()
    filas, sin_coord = [], 0
    for f in crudo["features"]:
        p = f["properties"]
        lat, lon, origen, disc = _coordenada(p, f.get("geometry"))
        if lat is None:
            sin_coord += 1
        u = _ubicar(terr, lat, lon)
        filas.append({
            "id_activo": f"POR-{p.get('OBJECTID')}",
            "item_micr": "infraestructura_portuaria_menor",
            "nombre": _texto(p.get("NOMBRE")),
            "lat": lat if lat is not None else "",
            "lon": lon if lon is not None else "",
            "comuna": u["comuna"], "provincia": u["provincia"],
            "region": u["region"], "cut": u["codigo_comuna"],
            "zona_geografica": u["zona_geografica"],
            "operador": "Dirección de Obras Portuarias (MOP)",
            "fuente": "MOP · Dirección de Obras Portuarias · DOP/CATASTRO_DOP",
            "fecha_captura": crudo["procedencia"]["fecha_descarga"],
            "confianza_ubicacion": origen,
            "localizacion": _texto(p.get("LOCATION")),
            "operativa": _texto(p.get("OPERATIVA")),
            "comuna_declarada": _texto(p.get("COMUNA")),
            "provincia_declarada": _texto(p.get("PROVINCIA")),
            "codigo_region_declarado": _texto(p.get("COD_REG")),
            "discrepancia_geom_vs_atributo_m": disc,
            "territorio_faltan": u["faltan"],
        })
    campos = CAMPOS_COMUNES + [
        "localizacion", "operativa", "comuna_declarada", "provincia_declarada",
        "codigo_region_declarado", "discrepancia_geom_vs_atributo_m",
        "territorio_faltan"]
    _escribir(DATOS / "inventario_obras_portuarias.csv", filas, campos)
    print(f"     sin coordenada usable: {sin_coord} de {len(filas)}")
    return filas


def tabla_embalses(fecha=None, terr=None):
    """Los embalses catastrados. ★ Dato del servicio: actualizado a dic-2015."""
    crudo = _leer_crudo("embalses", fecha)
    terr = terr if terr is not None else territorio.Territorio()
    filas, sin_coord = [], 0
    for f in crudo["features"]:
        p = f["properties"]
        lat, lon, origen, disc = _coordenada(p, f.get("geometry"))
        if lat is None:
            sin_coord += 1
        u = _ubicar(terr, lat, lon)
        filas.append({
            "id_activo": f"EMB-{_texto(p.get('CODEMBAL')) or p.get('OBJECTID')}",
            "item_micr": "embalses_y_represas",
            "nombre": _texto(p.get("NOMBRE")),
            "lat": lat if lat is not None else "",
            "lon": lon if lon is not None else "",
            "comuna": u["comuna"], "provincia": u["provincia"],
            "region": u["region"], "cut": u["codigo_comuna"],
            "zona_geografica": u["zona_geografica"],
            "operador": "sin dato",
            "fuente": "MOP · Dirección de Obras Hidráulicas · DOH/Embalses "
                      "(★ el servicio declara el dato actualizado a dic-2015)",
            "fecha_captura": crudo["procedencia"]["fecha_descarga"],
            "confianza_ubicacion": origen,
            "codigo_embalse": _texto(p.get("CODEMBAL")),
            "ano_construccion": _texto(p.get("ANOCONTR")),
            "altura_muro_m": _numero(p.get("ALT_MURO")),
            "cota": _numero(p.get("COTA")),
            "uso_embalse": _texto(p.get("USO_EMBAL")),
            "tipo_embalse": _texto(p.get("TIPO_EMBAL")),
            "tamano": _texto(p.get("TAMANO")),
            "fuente_natural": _texto(p.get("FUENTE_NAT")),
            "monitoreado": _texto(p.get("MONITOR")),
            "comuna_declarada": _texto(p.get("NOMCOM")),
            "region_declarada": _texto(p.get("NOMREG")),
            "discrepancia_geom_vs_atributo_m": disc,
            "territorio_faltan": u["faltan"],
        })
    campos = CAMPOS_COMUNES + [
        "codigo_embalse", "ano_construccion", "altura_muro_m", "cota",
        "uso_embalse", "tipo_embalse", "tamano", "fuente_natural",
        "monitoreado", "comuna_declarada", "region_declarada",
        "discrepancia_geom_vs_atributo_m", "territorio_faltan"]
    _escribir(DATOS / "inventario_embalses.csv", filas, campos)
    print(f"     sin coordenada usable: {sin_coord} de {len(filas)}")
    return filas


# ── 6. la comparación que evita contar dos veces ─────────────────────────────

def comparar_apr_con_ssr(fecha=None):
    """¿Son `DOH/APR` y `DOH/SSR_Clasificados_Ley_20998` la misma base?

    Traen el mismo número de registros y el mismo esquema, pero eso no prueba
    nada. Se compara por `ID_IDE`, que es el identificador estable del sistema,
    y se informa cuántos coinciden y cuántos no. Sólo una de las dos entra al
    inventario: contar 4.950 sistemas de agua potable rural donde hay 2.475
    sería el peor error posible en un inventario.
    """
    try:
        a = _leer_crudo("apr", fecha)
        b = _leer_crudo("ssr_ley20998", fecha)
    except FileNotFoundError as err:
        print(f"   no se puede comparar: {err}")
        return None
    ids_a = {_texto(f["properties"].get("ID_IDE")) for f in a["features"]}
    ids_b = {_texto(f["properties"].get("ID_IDE")) for f in b["features"]}
    ids_a.discard("")
    ids_b.discard("")
    comunes = ids_a & ids_b
    print(f"\nAPR vs SSR Ley 20.998, comparados por ID_IDE:")
    print(f"   APR: {len(a['features'])} rasgos, {len(ids_a)} ID_IDE distintos")
    print(f"   SSR: {len(b['features'])} rasgos, {len(ids_b)} ID_IDE distintos")
    print(f"   en común: {len(comunes)}")
    print(f"   sólo en APR: {len(ids_a - ids_b)} · sólo en SSR: {len(ids_b - ids_a)}")
    if ids_a == ids_b:
        print("   ⇒ SON LA MISMA BASE. Sólo `apr` entra al inventario.")
    else:
        print("   ⇒ NO son idénticas: revisar antes de descartar ninguna.")
    return {"apr": len(ids_a), "ssr": len(ids_b), "comunes": len(comunes)}


# ── 7. lo que el inventario aprende de estos datos ───────────────────────────

def cascada_energia_agua(filas_apr):
    """El cruce que justifica haber bajado el agua potable rural.

    Un corte eléctrico prolongado no deja a esas localidades a oscuras: las deja
    SIN AGUA, porque el sistema bombea. El que tiene grupo electrógeno aguanta;
    el que no, no. Este conteo dice a cuánta gente le pasa eso.
    """
    sin_grupo = [f for f in filas_apr if f["tiene_grupo_electrogeno"] == "no"]
    con_grupo = [f for f in filas_apr if f["tiene_grupo_electrogeno"] == "si"]
    sin_dato = [f for f in filas_apr if f["tiene_grupo_electrogeno"] == ""]
    aljibe = [f for f in filas_apr if f["depende_camion_aljibe"] == "si"]

    def personas(fs):
        return sum(f["beneficiarios_estimados"] for f in fs
                   if isinstance(f["beneficiarios_estimados"], (int, float)))

    print("\nCASCADA ENERGÍA → AGUA (sistemas de agua potable rural)")
    print(f"   con grupo electrógeno : {len(con_grupo):5d} sistemas · "
          f"{personas(con_grupo):>9,.0f} personas")
    print(f"   SIN grupo electrógeno : {len(sin_grupo):5d} sistemas · "
          f"{personas(sin_grupo):>9,.0f} personas  ← un corte eléctrico los seca")
    print(f"   sin dato del campo    : {len(sin_dato):5d} sistemas · "
          f"{personas(sin_dato):>9,.0f} personas  ← hueco declarado, no un «no»")
    print(f"   ya dependen de camión aljibe: {len(aljibe)} sistemas · "
          f"{personas(aljibe):,.0f} personas")
    return {"con_grupo": len(con_grupo), "sin_grupo": len(sin_grupo),
            "sin_dato": len(sin_dato), "aljibe": len(aljibe)}


def aerodromos_de_zona_aislada(filas_aer):
    """Cuántos aeródromos sirven a zonas que el propio Estado declaró aisladas.

    En esas zonas el aeródromo no es «transporte»: es el único acceso cuando el
    camino se corta. Es el activo de reemplazo del modo de falla «aislamiento».
    """
    aislados = [f for f in filas_aer if f["zona_aislada"] == "si"]
    extremos = [f for f in filas_aer if f["zona_extrema"] == "si"]
    print("\nAERÓDROMOS Y AISLAMIENTO (clasificación del propio MOP)")
    print(f"   en zona declarada AISLADA : {len(aislados)} de {len(filas_aer)}")
    print(f"   en zona declarada EXTREMA : {len(extremos)}")
    categorias = {}
    for f in aislados:
        categorias[f["categoria_aislamiento"] or "(sin categoría)"] = \
            categorias.get(f["categoria_aislamiento"] or "(sin categoría)", 0) + 1
    for cat, n in sorted(categorias.items(), key=lambda x: -x[1]):
        print(f"      {cat:40s} {n}")
    return {"aislados": len(aislados), "extremos": len(extremos)}


# ── main ─────────────────────────────────────────────────────────────────────

def tablas(fecha=None):
    terr = territorio.Territorio()
    print("Capas de territorio:", terr.estado())
    comparar_apr_con_ssr(fecha)
    print("\nTablas del inventario:")
    apr = tabla_apr(fecha, terr)
    aer = tabla_aeropuertos(fecha, terr)
    tabla_portuaria(fecha, terr)
    tabla_embalses(fecha, terr)
    cascada_energia_agua(apr)
    aerodromos_de_zona_aislada(aer)


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
