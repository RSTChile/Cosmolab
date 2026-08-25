"""
Adaptador de INVENTARIO — SENAPRED, Sistema Integrado de Información para
Emergencias (SIIE), servidor `visor-grd.senapred.gob.cl`.

★ EL HALLAZGO: EL DESTINATARIO DEL PROYECTO YA PUBLICA SU PROPIO INVENTARIO
---------------------------------------------------------------------------
`INTEGRACION_SENAPRED.md` fija que SENAPRED —el Servicio Nacional de Prevención
y Respuesta ante Desastres— es el **destinatario** de este proyecto, no una
fuente. Eso sigue siendo cierto para las ALERTAS. Pero resulta que SENAPRED
mantiene además un servidor ArcGIS REST público, abierto y sin credenciales, con
**veintinueve capas de infraestructura crítica georreferenciada de todo el país**,
agrupadas bajo el nombre «Infraestructura Vulnerable».

O sea: la institución que decide en emergencia ya tiene un catastro
multi-sectorial ubicado. No estaba en ningún catálogo de datos abiertos; está en
el servidor de su visor de Gestión del Riesgo de Desastres, a la vista.

Esto no vuelve inútil al proyecto — al contrario, precisa qué le falta. SENAPRED
tiene el inventario y tiene las alertas; lo que no publica es el **cruce
sistemático** entre uno y otras, ni la métrica de qué pasa cuando un activo cae.
Pero conviene ser honesto y escribirlo: la frase «nadie cruza amenaza con
infraestructura» hay que matizarla. SENAPRED tiene las dos mitades.

QUÉ SE BAJA — dos servicios, veintinueve capas
-----------------------------------------------
`SIIE/InfraestructuraVulnerable` (23 capas de punto):

    capa  qué                                       registros
     0    Energía (puntual)                             4.384
     1    Comunicación Aérea                              340
     2    Establecimientos de Salud                     5.159
     3    Educación (incluye jardines infantiles)      15.243
     4    Bomberos                                      1.412
     5    Carabineros                                     885
     6    Centros Penitenciarios                           82
     7    Edificios Públicos                              522
     8    Centros Públicos                              1.194
     9    Puentes de Comunicación Vial                  6.628
    10    Superintendencia de Servicios Sanitarios      8.463
    11    Supermercados                                   993
    12    Intendencias Regionales                          16
    13    Direcciones Regionales de la ex-ONEMI             16
    14    Mejor Niñez                                   1.533
    15    Policía de Investigaciones (PDI)                175
    16    Sedes universitarias                            400
    17    Servicio Médico Legal                            47
    18    SENAMA (Servicio Nacional del Adulto Mayor)     881
    19    Suministro Alternativo de Agua                5.743
    20    Gobernaciones                                    55
    21    Municipios                                      345
    22    Recintos Deportivos                             538

`SIIE/Infraestructuras` (6 capas):

     0    Energía lineal (LÍNEAS, con traza)            3.632
     1    Aguas                                         8.463
     2    Telefonía                                    16.669
     3    Puertos                                         441
     4    Pasos fronterizos                                30
     5    Energía puntual                               4.384

★ DOS CAPAS QUE TAPAN HUECOS QUE EL PROYECTO TENÍA ABIERTOS:

· **Telefonía, 16.669 puntos.** Es el ítem de telecomunicaciones de la Matriz,
  que estaba sin poblar. Trae empresa, tipo y un campo `criticidad` que el
  propio SENAPRED asigna.

· **Energía lineal, 3.632 líneas CON GEOMETRÍA.** El Coordinador Eléctrico
  publica 40.643 km de línea de transmisión **sin traza** (comprobado: el campo
  `coordenadas` viene vacío en sus 2.926 tramos). SENAPRED sí publica trazado.
  No son el mismo universo y no se deben mezclar sin comparar primero; pero
  significa que la traza EXISTE en manos del Estado.

· **Recintos Deportivos (538) y Educación (15.243)** son, en la práctica, el
  catastro de albergues potenciales: en emergencia se alberga en gimnasios y
  escuelas. SENAPRED no publica un catastro nacional de albergues como tal
  (verificado: `geoportal.senapred.cl` no resuelve, y el mapa de albergues de su
  cuenta ArcGIS Online está en privado). Estas dos capas son lo más cerca que se
  puede llegar hoy con dato público.

LO QUE NO SE COPIA: DATOS DE PERSONAS
--------------------------------------
Algunas capas traen campos que identifican o pueden identificar personas
naturales: `rut`, `telefono`, `responsabl`, `nombre_rev`, `mail`, `fono`. **Este
módulo NO los copia a ninguna tabla del inventario.** Quedan en el archivo crudo
tal como los entregó el servicio, porque el crudo debe ser exactamente lo que se
recibió, y se excluyen de todo lo que se procesa o se publica.

CONDICIONES DE USO
------------------
· `https://visor-grd.senapred.gob.cl/robots.txt` responde **200** con
  `User-agent: *` / `Disallow:` (vacío): **todo permitido**, sin restricciones.
  Verificado el 19-ago-2026.
· Servicio público, anónimo, sin credenciales ni registro.
· `copyrightText` viene **vacío** en los dos servicios: SENAPRED no declara
  licencia. Eso no es permiso implícito ni prohibición implícita — es un hueco.
  Se atribuye a SENAPRED en todo derivado y queda **PENDIENTE pedir la
  confirmación por escrito**, que además es trámite natural porque SENAPRED es
  el destinatario declarado del proyecto.
· Ritmo prudente: una petición por capa (el servicio declara
  `maxRecordCount = 1.000.000`, así que cada capa cabe en un viaje), con pausa
  entre capas y reintentos con espera creciente.

SOBRE LA ACTUALIZACIÓN — LEER ANTES DE USAR ESTO COMO VERDAD
-------------------------------------------------------------
Ninguna capa trae campo de fecha. Y hay señal de que varias están atrasadas: la
capa de Educación tiene 15.243 registros pero le **faltan más de mil
establecimientos que el Ministerio de Educación sí lista como funcionando en
2025**, y algunas capas conservan nombres de instituciones que ya no existen
(«Intendencias Regionales», «Direcciones_Regionales_Onemi», «Gobernaciones»
fueron reemplazadas por Delegaciones Presidenciales en 2021). Se usa como capa
de COBERTURA amplia y de geometría, no como registro vigente. Donde exista una
fuente sectorial más fresca —Ministerio de Educación para colegios, Departamento
de Estadísticas e Información de Salud para salud, Coordinador Eléctrico para
subestaciones— **manda la sectorial**, y esta queda de respaldo.

USO
---
    python3 adaptadores/inventario_senapred_siie.py explorar
    python3 adaptadores/inventario_senapred_siie.py bajar
    python3 adaptadores/inventario_senapred_siie.py bajar telefonia energia_lineal
    python3 adaptadores/inventario_senapred_siie.py tablas
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

BASE = "https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE"
HOY = date.today().isoformat()
CRUDO = AQUI / "datos" / "crudo" / "senapred_siie"
DATOS = AQUI / "datos"

PAUSA = 2.0
REINTENTOS = 5
TIEMPO_LIMITE = 300

# ★ Campos que NO se copian a ninguna tabla: identifican o pueden identificar a
# personas naturales. Quedan sólo en el crudo, tal como llegaron.
CAMPOS_DE_PERSONAS = {"rut", "telefono", "fono", "mail", "email", "responsabl",
                      "responsable", "nombre_rev", "contacto", "celular"}


# ── qué se baja ──────────────────────────────────────────────────────────────
# `item_micr` es la etiqueta con que este activo entra al inventario. `tabla`
# dice si la capa merece su propio CSV (las que el proyecto va a usar) o si se
# baja sólo para dejar el crudo disponible.

CAPAS = {
    # ── SIIE/InfraestructuraVulnerable ──
    "energia_puntual": dict(
        servicio="InfraestructuraVulnerable", capa=0, esperado=4384,
        item_micr="instalaciones_electricas", tabla=True,
        descripcion="Instalaciones eléctricas puntuales con propietario, tipo, "
                    "sistema eléctrico y `criticidad` asignada por SENAPRED."),
    "comunicacion_aerea": dict(
        servicio="InfraestructuraVulnerable", capa=1, esperado=340,
        item_micr="comunicacion_aerea", tabla=True,
        descripcion="Instalaciones de comunicación aérea."),
    "salud": dict(
        servicio="InfraestructuraVulnerable", capa=2, esperado=5159,
        item_micr="establecimientos_de_salud", tabla=False,
        descripcion="Establecimientos de salud. NO entra al inventario: manda "
                    "el maestro del Departamento de Estadísticas e Información "
                    "de Salud (DEIS), que se actualiza a diario."),
    "educacion": dict(
        servicio="InfraestructuraVulnerable", capa=3, esperado=15243,
        item_micr="establecimientos_educacionales", tabla=True,
        descripcion="★ Incluye los jardines infantiles de JUNJI, INTEGRA y "
                    "VTF, que el Directorio del Ministerio de Educación NO "
                    "tiene. Por eso entra, aunque para colegios mande MINEDUC."),
    "bomberos": dict(
        servicio="InfraestructuraVulnerable", capa=4, esperado=1412,
        item_micr="cuarteles_de_bomberos", tabla=True,
        descripcion="Cuarteles de bomberos. Es infraestructura de RESPUESTA: "
                    "si se aísla, no es que se pierda un edificio, es que la "
                    "comuna se queda sin quien responda."),
    "carabineros": dict(
        servicio="InfraestructuraVulnerable", capa=5, esperado=885,
        item_micr="unidades_de_carabineros", tabla=True,
        descripcion="Unidades de Carabineros de Chile. Infraestructura de "
                    "respuesta."),
    "penitenciarios": dict(
        servicio="InfraestructuraVulnerable", capa=6, esperado=82,
        item_micr="centros_penitenciarios", tabla=True,
        descripcion="Centros penitenciarios: población que no puede evacuarse "
                    "por sus propios medios."),
    "edificios_publicos": dict(
        servicio="InfraestructuraVulnerable", capa=7, esperado=522,
        item_micr="edificios_publicos", tabla=True, descripcion="Edificios públicos."),
    "centros_publicos": dict(
        servicio="InfraestructuraVulnerable", capa=8, esperado=1194,
        item_micr="centros_publicos", tabla=True, descripcion="Centros públicos."),
    "puentes_vial": dict(
        servicio="InfraestructuraVulnerable", capa=9, esperado=6628,
        item_micr="puentes", tabla=False,
        descripcion="Puentes. NO entra al inventario: ya están los 6.742 del "
                    "MOP, que además traen el cauce que cruza cada uno. Se "
                    "baja para poder COMPARAR las dos coberturas."),
    "siss": dict(
        servicio="InfraestructuraVulnerable", capa=10, esperado=8463,
        item_micr="infraestructura_sanitaria", tabla=True,
        descripcion="Infraestructura sanitaria fiscalizada por la "
                    "Superintendencia de Servicios Sanitarios (SISS)."),
    "supermercados": dict(
        servicio="InfraestructuraVulnerable", capa=11, esperado=993,
        item_micr="supermercados", tabla=True,
        descripcion="Supermercados: abastecimiento de alimento en emergencia."),
    "intendencias": dict(
        servicio="InfraestructuraVulnerable", capa=12, esperado=16,
        item_micr="sedes_gobierno_regional", tabla=True,
        descripcion="★ Nombre institucional VENCIDO: las Intendencias fueron "
                    "reemplazadas por Delegaciones Presidenciales Regionales "
                    "en 2021. La ubicación sirve; el nombre no."),
    "direcciones_regionales": dict(
        servicio="InfraestructuraVulnerable", capa=13, esperado=16,
        item_micr="direcciones_regionales_senapred", tabla=True,
        descripcion="★ Las 16 direcciones regionales del propio SENAPRED "
                    "(nombradas todavía como ONEMI). Son el nodo de mando "
                    "regional del sistema al que este proyecto sirve."),
    "mejor_ninez": dict(
        servicio="InfraestructuraVulnerable", capa=14, esperado=1533,
        item_micr="residencias_proteccion_infancia", tabla=True,
        descripcion="Red del Servicio Mejor Niñez: población dependiente."),
    "pdi": dict(
        servicio="InfraestructuraVulnerable", capa=15, esperado=175,
        item_micr="unidades_pdi", tabla=True,
        descripcion="Unidades de la Policía de Investigaciones."),
    "sedes_universitarias": dict(
        servicio="InfraestructuraVulnerable", capa=16, esperado=400,
        item_micr="sedes_universitarias", tabla=True,
        descripcion="Sedes universitarias."),
    "servicio_medico_legal": dict(
        servicio="InfraestructuraVulnerable", capa=17, esperado=47,
        item_micr="servicio_medico_legal", tabla=True,
        descripcion="Servicio Médico Legal."),
    "senama": dict(
        servicio="InfraestructuraVulnerable", capa=18, esperado=881,
        item_micr="establecimientos_adulto_mayor", tabla=True,
        descripcion="Red del Servicio Nacional del Adulto Mayor (SENAMA): "
                    "población que no evacúa por sus propios medios."),
    "suministro_alternativo_agua": dict(
        servicio="InfraestructuraVulnerable", capa=19, esperado=5743,
        item_micr="suministro_alternativo_de_agua", tabla=True,
        descripcion="★ Puntos de abastecimiento alternativo de agua: estanques "
                    "y puntos de carga de camión aljibe. Es el PLAN B del agua "
                    "potable rural, y por eso importa tanto como el plan A."),
    "gobernaciones": dict(
        servicio="InfraestructuraVulnerable", capa=20, esperado=55,
        item_micr="sedes_gobierno_provincial", tabla=True,
        descripcion="★ Nombre institucional VENCIDO (Delegaciones "
                    "Presidenciales Provinciales desde 2021)."),
    "municipios": dict(
        servicio="InfraestructuraVulnerable", capa=21, esperado=345,
        item_micr="municipios", tabla=True,
        descripcion="Las 345 municipalidades. Son el nivel Comunal del COGRID: "
                    "quien manda cuando la emergencia es local."),
    "recintos_deportivos": dict(
        servicio="InfraestructuraVulnerable", capa=22, esperado=538,
        item_micr="recintos_deportivos", tabla=True,
        descripcion="Gimnasios y recintos deportivos: los albergues de hecho."),
    # ── SIIE/Infraestructuras ──
    "energia_lineal": dict(
        servicio="Infraestructuras", capa=0, esperado=3632,
        item_micr="lineas_electricas", tabla=True, geometria="linea",
        descripcion="★ Líneas eléctricas CON TRAZA. El Coordinador publica sus "
                    "40.643 km sin geometría; acá sí hay trazado."),
    "aguas": dict(
        servicio="Infraestructuras", capa=1, esperado=8463,
        item_micr="infraestructura_de_agua", tabla=True,
        descripcion="Infraestructura de agua."),
    "telefonia": dict(
        servicio="Infraestructuras", capa=2, esperado=16669,
        item_micr="telecomunicaciones", tabla=True,
        descripcion="★ 16.669 puntos de telefonía con empresa, tipo y nivel de "
                    "criticidad. Es el ítem de telecomunicaciones de la Matriz, "
                    "que estaba sin poblar."),
    "puertos": dict(
        servicio="Infraestructuras", capa=3, esperado=441,
        item_micr="puertos", tabla=True,
        descripcion="Puertos y obras portuarias."),
    "pasos_fronterizos": dict(
        servicio="Infraestructuras", capa=4, esperado=30,
        item_micr="pasos_fronterizos", tabla=True,
        descripcion="Los 30 pasos fronterizos habilitados."),
}

FUENTE = dict(
    id="senapred_siie",
    organismo="SENAPRED — Servicio Nacional de Prevención y Respuesta ante "
              "Desastres · Sistema Integrado de Información para Emergencias",
    producto="Servicios ArcGIS REST SIIE/InfraestructuraVulnerable y "
             "SIIE/Infraestructuras del visor de Gestión del Riesgo de Desastres",
    url=BASE,
    formato="esriJSON (este servidor, ArcGIS 10.71, tampoco ofrece f=geojson; "
            "la conversión la hace este módulo)",
    familia="ESTADO",
    acceso="anonimo",
    acceso_verificado=1,
    robots_txt="200 · «User-agent: * / Disallow:» → todo permitido "
               "(verificado 19-ago-2026)",
    condiciones_uso="Servicio público sin credenciales. `copyrightText` VACÍO: "
                    "SENAPRED no declara licencia. Atribución a SENAPRED y "
                    "PENDIENTE pedir confirmación escrita.",
    permite_automatizacion="si (robots.txt lo permite explícitamente)",
    granularidad="activo puntual, salvo `energia_lineal` que es lineal",
    advertencia="Ninguna capa trae fecha de actualización, y hay señales de "
                "atraso (nombres institucionales de 2018-2021, cobertura "
                "educacional incompleta frente al Ministerio de Educación "
                "2025). Usar como cobertura y geometría, no como registro "
                "vigente.",
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
                              "(investigacion; insumo a SENAPRED)")
            with urllib.request.urlopen(pedido, timeout=TIEMPO_LIMITE) as resp:
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


# ── 1. exploración ───────────────────────────────────────────────────────────

def explorar():
    print(f"SENAPRED · SIIE · {BASE}\n")
    for clave, c in CAPAS.items():
        try:
            n = _pedir(f"{_url_capa(clave)}/query",
                       {"where": "1=1", "returnCountOnly": "true", "f": "json"},
                       intentos=2).get("count")
        except Exception as err:                       # noqa: BLE001
            print(f"  {clave:28s} NO RESPONDE ({str(err)[:60]})")
            continue
        marca = "" if n == c["esperado"] else f"  ★ el módulo esperaba {c['esperado']}"
        entra = "→ inventario" if c["tabla"] else "  (sólo crudo)"
        print(f"  {clave:28s} {n:>6} reg  {entra}{marca}")
        time.sleep(0.8)


# ── 2. bajada del crudo ──────────────────────────────────────────────────────

def _a_geojson(rasgos, tipo_geom):
    """esriJSON → GeoJSON, explícito. Este servidor tampoco ofrece f=geojson."""
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
        else:
            if g.get("x") is not None and g.get("y") is not None:
                geom = {"type": "Point", "coordinates": [g["x"], g["y"]]}
        salida.append({"type": "Feature", "geometry": geom, "properties": props})
    return salida


def bajar(clave):
    """Baja una capa entera. El servicio declara `maxRecordCount = 1.000.000`,
    así que cada capa cabe en una sola petición — se comprueba igual que el
    número recibido coincida con el que el servicio dice tener, y si el servidor
    marca `exceededTransferLimit` se dice a gritos en vez de guardar un
    inventario truncado."""
    c = CAPAS[clave]
    tipo_geom = c.get("geometria", "punto")
    destino = CRUDO / HOY
    destino.mkdir(parents=True, exist_ok=True)

    print(f"\n[{clave}] {c['servicio']} capa {c['capa']}")
    esperado = _pedir(f"{_url_capa(clave)}/query",
                      {"where": "1=1", "returnCountOnly": "true", "f": "json"}
                      ).get("count")
    time.sleep(PAUSA)
    resp = _pedir(f"{_url_capa(clave)}/query",
                  {"where": "1=1", "outFields": "*", "returnGeometry": "true",
                   "outSR": "4326", "f": "json"}, metodo="POST")
    rasgos = resp.get("features", [])
    truncado = bool(resp.get("exceededTransferLimit"))
    print(f"   {len(rasgos)} recibidos de {esperado} que declara el servicio"
          + ("  ★ TRUNCADO por el servidor" if truncado else ""))
    if truncado or len(rasgos) != esperado:
        print("   ★ ATENCIÓN: el conteo NO cierra. Se guarda igual, con la "
              "discrepancia anotada en la procedencia.")

    geojson = {
        "type": "FeatureCollection",
        "features": _a_geojson(rasgos, tipo_geom),
        "procedencia": {
            "url": f"{_url_capa(clave)}/query",
            "servicio": c["servicio"], "capa": c["capa"],
            "descripcion": c["descripcion"],
            "fecha_descarga": HOY,
            "registros_declarados_por_el_servicio": esperado,
            "registros_recibidos": len(rasgos),
            "truncado_por_el_servidor": truncado,
            "sistema_referencia_pedido": "EPSG:4326",
            "fuente": FUENTE,
        },
    }
    with gzip.open(destino / f"{clave}.geojson.gz", "wt", encoding="utf-8") as fh:
        json.dump(geojson, fh, ensure_ascii=False)
    print(f"   → datos/crudo/senapred_siie/{HOY}/{clave}.geojson.gz")
    return geojson


def bajar_todo(claves=None):
    for clave in (claves or list(CAPAS)):
        try:
            bajar(clave)
        except Exception as err:                       # noqa: BLE001
            print(f"   ★ {clave} NO SE PUDO BAJAR: {str(err)[:140]}")
            print("     queda como «sin dato» — no se inventa nada")
        time.sleep(PAUSA)


# ── 3. lectura y normalización ───────────────────────────────────────────────

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
    if valor is None:
        return ""
    return str(valor).strip()


# Cada capa nombra sus campos como quiere. En vez de escribir veintinueve
# diccionarios de mapeo a mano, se busca el primer campo que exista de una lista
# de candidatos. Si NINGUNO existe, el campo sale vacío: no se rellena con otra
# cosa que se le parezca.
CANDIDATOS_NOMBRE = ("nombre", "nombre_ee", "nombre_del", "name", "compa__ia",
                     "razon_soci", "establecimiento", "glosa")
CANDIDATOS_OPERADOR = ("propiet", "empresa", "dependencia", "operador",
                       "propietario", "tipo_de_ob", "nombre_slep")


def _primero(props, candidatos):
    for c in candidatos:
        v = _texto(props.get(c))
        if v and v.lower() not in ("null", "none", "s/i", "sin informacion"):
            return v
    return ""


def _plausible(lat, lon):
    """Chile continental, insular y antártico, con margen. Fuera de la caja se
    marca y se cuenta; nunca se corrige en silencio."""
    if lat is None or lon is None:
        return False
    return -90.0 <= lat <= -17.0 and -110.0 <= lon <= -66.0


def _flotante(valor):
    """Coordenada a número, o None.

    Hace falta de verdad: comprobado el 19-ago-2026, hay capas de este servidor
    (la de la Superintendencia de Servicios Sanitarios, entre otras) que
    entregan la geometría con las coordenadas como TEXTO en vez de número. Una
    comparación entre texto y número revienta; y si se tapara la excepción, se
    perderían los puntos sin que nadie se entere. Se convierte explícitamente y
    lo que no se puede convertir queda como «sin coordenada»."""
    if valor is None:
        return None
    try:
        n = float(valor)
    except (TypeError, ValueError):
        return None
    return n if n == n else None


def _punto(geometria):
    """Coordenada representativa del rasgo, y de qué tipo es.

    Para un punto, es el punto. Para una línea, el vértice del MEDIO de su
    trazado — que no es el centroide ni el punto medio por distancia, y se dice
    así, porque una línea eléctrica de 200 km «ubicada» en un punto es una
    simplificación y hay que declararla como tal. El trazado completo queda
    intacto en el crudo para quien quiera cruzarlo bien.
    """
    if not geometria:
        return None, None, "sin_coordenada", 0
    tipo = geometria.get("type")
    if tipo == "Point":
        lon, lat = (_flotante(v) for v in geometria["coordinates"][:2])
        if lat is None or lon is None:
            return None, None, "coordenada_ilegible", 0
        return lat, lon, "punto_del_servicio_epsg4326", 1
    caminos = ([geometria["coordinates"]] if tipo == "LineString"
               else geometria.get("coordinates") or [])
    vertices = [v for camino in caminos for v in camino]
    if not vertices:
        return None, None, "sin_coordenada", 0
    lon, lat = (_flotante(v) for v in vertices[len(vertices) // 2][:2])
    if lat is None or lon is None:
        return None, None, "coordenada_ilegible", len(vertices)
    return lat, lon, "vertice_medio_de_la_linea_NO_es_la_linea", len(vertices)


CAMPOS_COMUNES = ["id_activo", "item_micr", "nombre", "lat", "lon",
                  "comuna", "provincia", "region", "cut", "zona_geografica",
                  "operador", "fuente", "fecha_captura", "confianza_ubicacion"]


def tabla(clave, fecha=None, terr=None):
    """Normaliza una capa al esquema común y le agrega TODOS sus campos propios
    con prefijo `src_`, menos los que identifican personas.

    Se conservan todos los campos originales a propósito: recortar ahora, sin
    saber cuáles va a necesitar el modelo, obliga a volver a bajar todo después.
    El prefijo deja claro cuál es dato de la fuente y cuál es derivado nuestro.
    """
    c = CAPAS[clave]
    crudo, fecha = _leer_crudo(clave, fecha)
    terr = terr if terr is not None else territorio.Territorio()
    tipo_geom = c.get("geometria", "punto")

    propios = []
    for f in crudo["features"]:
        for k in f["properties"]:
            kl = k.lower()
            if kl in CAMPOS_DE_PERSONAS or kl in ("shape", "globalid",
                                                  "globalid_1"):
                continue
            if k not in propios:
                propios.append(k)

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
            r = terr.ubicar(lat, lon)
            u = {"comuna": r["comuna"] or "", "provincia": r["provincia"] or "",
                 "region": r["region"] or "",
                 "codigo_comuna": r["codigo_comuna"] or "",
                 "zona_geografica": r["zona_geografica"] or "",
                 "faltan": "|".join(r["faltan"])}

        fila = {
            "id_activo": f"SNP-{clave}-{p.get('objectid') or p.get('objectid_1')}",
            "item_micr": c["item_micr"],
            "nombre": _primero(p, CANDIDATOS_NOMBRE),
            "lat": lat if lat is not None else "",
            "lon": lon if lon is not None else "",
            "comuna": u["comuna"], "provincia": u["provincia"],
            "region": u["region"], "cut": u["codigo_comuna"],
            "zona_geografica": u["zona_geografica"],
            "operador": _primero(p, CANDIDATOS_OPERADOR),
            "fuente": f"SENAPRED · SIIE · {c['servicio']} capa {c['capa']}",
            "fecha_captura": fecha,
            "confianza_ubicacion": origen,
            "vertices_del_trazado": n_vert if tipo_geom == "linea" else "",
            "territorio_faltan": u["faltan"],
        }
        for k in propios:
            fila[f"src_{k}"] = _texto(p.get(k))
        filas.append(fila)

    campos = (CAMPOS_COMUNES + ["vertices_del_trazado", "territorio_faltan"]
              + [f"src_{k}" for k in propios])
    ruta = DATOS / f"inventario_senapred_{clave}.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    aviso = f" · {sin_coord} sin coordenada" if sin_coord else ""
    aviso += f" · {fuera} fuera de Chile" if fuera else ""
    print(f"   → {ruta.name}  ({len(filas)} filas{aviso})")
    return filas


def tablas(fecha=None):
    terr = territorio.Territorio()
    print("Capas de territorio:", terr.estado(), "\n")
    total = 0
    for clave, c in CAPAS.items():
        if not c["tabla"]:
            continue
        try:
            filas = tabla(clave, fecha, terr)
            total += sum(1 for f in filas if f["lat"] != "")
        except FileNotFoundError as err:
            print(f"   ★ {clave}: {err}")
    print(f"\nTotal georreferenciado aportado por SENAPRED SIIE: {total:,}")
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
