"""
Adaptador de INVENTARIO — establecimientos de salud de Chile.
Fuente: Departamento de Estadísticas e Información de Salud (DEIS) del
Ministerio de Salud (MINSAL), publicado en el portal nacional de datos abiertos
`datos.gob.cl`.

POR QUÉ ESTA FUENTE Y NO OTRA
------------------------------
Hay cuatro sitios que publican establecimientos de salud de Chile. Se probaron
los cuatro y este es el que gana, por razones que conviene dejar escritas:

1. **`datos.gob.cl`, recurso del DEIS** ← el elegido.
   · Se actualiza **a diario** (el campo `last_modified` del recurso traía hoy
     la fecha de hoy).
   · Licencia **CC-Zero declarada explícitamente**: dominio público, sin
     restricción de uso. De todas las fuentes de este inventario, es la única
     con licencia limpia y sin pendientes.
   · `robots.txt` de `datos.gob.cl` permite la ruta de descarga; sólo restringe
     `/api/` y pide `Crawl-Delay: 10`. **Este módulo baja el archivo directo y
     NO toca `/api/`** — se respeta la restricción, no se rodea.
2. `deis.minsal.cl` — el sitio del propio departamento. **NO se usa**: su
   `robots.txt` declara `Disallow: /` para agentes automáticos de inteligencia
   artificial y `ai-train=no`. El mismo dato está en `datos.gob.cl` sin esa
   restricción, así que se usa el espejo autorizado y punto.
3. `services3.arcgis.com/CNzkI2T3GmfwkaAR` — la organización ArcGIS Online de
   MINSAL, pública, con 838 servicios. Existe y funciona. **No se usa como
   maestro** porque no declara licencia alguna. Ver la nota del final: tiene un
   contenido que este proyecto debería conocer aunque no lo ingiera.
4. `geoportal.cl` — su recurso de salud está **muerto (404)**.

QUÉ TRAE
--------
5.717 establecimientos, de los cuales **5.272 con coordenada (92,2 %)**, en
grados decimales EPSG:4326. Es infraestructura crítica de primera prioridad y
entra completa: hospitales (234), centros de salud familiar CESFAM (607), postas
de salud rural (1.184), servicios de atención primaria de urgencia SAPU (258),
centros comunitarios CECOSF (304), y el resto del árbol asistencial.

Los campos que hacen la diferencia para este proyecto:
· `TieneServicioUrgencia` — 796 con urgencia. Un hospital sin urgencia que queda
  aislado es un problema; una urgencia aislada es otra cosa.
· `NivelComplejidadEstabGlosa` — Alta (131) / Mediana (1.253) / Baja (3.321).
  Los 131 de alta complejidad son los que no se pueden reemplazar: si uno queda
  incomunicado, el paciente tiene que viajar a otra región.
· `TipoUrgencia` y `ClasificacionTipoSapu` — el tipo de respuesta que se pierde.

TRAMPAS DEL ARCHIVO — comprobadas, no supuestas
------------------------------------------------
★ **`EstadoFuncionamiento` tiene el mismo valor escrito de dos maneras**:
  `'Vigente en Operación Habitual'` (5.093 filas) y
  `'Vigente en operación habitual'` (210 filas), con minúsculas distintas. Un
  filtro por igualdad exacta se come 210 establecimientos sin avisar. Se
  normaliza a minúsculas antes de comparar.

★ **Los códigos son texto, no números.** `ComunaCodigo` es el código único
  territorial (CUT) de 5 caracteres y `RegionCodigo` de 2: ambos conservan el
  cero a la izquierda («09110» para Melipeuco). Leerlos como número los
  destruye. Este módulo lee TODO como texto.

★ **Un registro con la latitud mal puesta.** El establecimiento
  `EstablecimientoCodigo = 120205` (Clínica Adventista, Los Ángeles) trae
  `Latitud = -72.35…`, que es una longitud copiada en el campo equivocado. No se
  corrige a mano —no sabemos cuál era la latitud verdadera— : se marca como sin
  coordenada usable y se cuenta. El filtro de caja de Chile lo atrapa solo.

★ **No trae provincia.** El nivel Provincial es uno de los cuatro niveles
  administrativos que el proyecto está obligado a modelar. Se resuelve como en
  todo el resto del inventario: la provincia se DERIVA del polígono en que cae
  la coordenada, no se copia de un texto.

NO SE RECOLECTAN DATOS DE PERSONAS. El archivo trae una columna
`TelefonoMovil_TelefonoFijo`. Es el teléfono del establecimiento, no de una
persona, pero un teléfono móvil puede ser el de un funcionario: **no se copia**.
Queda sólo en el archivo crudo tal como lo publicó el DEIS.

USO
---
    python3 adaptadores/inventario_salud_deis.py bajar
    python3 adaptadores/inventario_salud_deis.py tabla
"""

import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

AQUI = Path(__file__).parent.parent
sys.path.insert(0, str(AQUI))
import territorio  # noqa: E402

# La ruta de descarga directa del recurso. NO se usa /api/ de datos.gob.cl:
# su robots.txt lo desautoriza, y todo lo que hace falta está en este archivo.
URL = ("https://datos.gob.cl/dataset/3bf4cf7c-f638-4735-9a01-f65faae4beca/"
       "resource/2c44d782-3365-44e3-aefb-2c8b8363a1bc/download")

HOY = date.today().isoformat()
CRUDO = AQUI / "datos" / "crudo" / "minsal_deis"
DATOS = AQUI / "datos"

# ★ No se copia a ninguna tabla.
CAMPOS_DE_PERSONAS = {"TelefonoMovil_TelefonoFijo"}

FUENTE = dict(
    id="minsal_deis_establecimientos",
    organismo="Ministerio de Salud (MINSAL) — Departamento de Estadísticas e "
              "Información de Salud (DEIS)",
    producto="Establecimientos de Salud vigentes (maestro nacional)",
    url=URL,
    pagina="https://datos.gob.cl/dataset/establecimientos-de-salud-vigentes",
    formato="CSV, separador «;», codificación UTF-8, decimal con punto",
    familia="ESTADO",
    acceso="anonimo",
    acceso_verificado=1,
    licencia="CC-Zero (Creative Commons CCZero) — dominio público, declarada "
             "por el portal. Es la licencia MÁS LIMPIA de todo el inventario.",
    robots_txt="200 · datos.gob.cl restringe /dataset/rate/, /revision/, "
               "/dataset/*/history y /api/, con Crawl-Delay: 10. La ruta de "
               "descarga usada NO está restringida y no se toca /api/. "
               "Verificado 19-ago-2026.",
    condiciones_uso="Sin restricción (CC-Zero). Se cita al DEIS igual, por "
                    "trazabilidad.",
    permite_automatizacion="si",
    actualizacion="diaria",
    granularidad="establecimiento",
)


# ── 1. bajada del crudo ──────────────────────────────────────────────────────

def bajar():
    """Baja el archivo tal como está y lo guarda ANTES de mirarlo.

    Se guarda además la cabecera HTTP completa: el `last-modified` del servidor
    es la única fecha real del corte de datos, y sin ella no se puede saber
    después si dos descargas distintas vieron lo mismo.
    """
    destino = CRUDO / HOY
    destino.mkdir(parents=True, exist_ok=True)
    print(f"[salud DEIS] {URL}")

    pedido = urllib.request.Request(URL)
    pedido.add_header("User-Agent",
                      "matriz-infraestructura-critica-clima/1.0 "
                      "(investigacion; inventario de infraestructura critica)")
    espera = 4
    for intento in range(1, 5):
        try:
            with urllib.request.urlopen(pedido, timeout=180) as resp:
                bruto = resp.read()
                cabeceras = dict(resp.headers)
            break
        except Exception as err:                       # noqa: BLE001
            if intento == 4:
                raise
            print(f"   reintento {intento}/3 tras {espera}s "
                  f"({type(err).__name__}: {str(err)[:100]})")
            time.sleep(espera)
            espera *= 2

    ruta = destino / "establecimientos_salud_deis.csv"
    ruta.write_bytes(bruto)
    (destino / "procedencia.json").write_text(json.dumps({
        "url": URL, "fecha_descarga": HOY,
        "bytes": len(bruto),
        "cabeceras_http": cabeceras,
        "last_modified_declarado": cabeceras.get("Last-Modified", "sin dato"),
        "fuente": FUENTE,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   {len(bruto):,} bytes · last-modified: "
          f"{cabeceras.get('Last-Modified', 'sin dato')}")
    print(f"   → datos/crudo/minsal_deis/{HOY}/establecimientos_salud_deis.csv")
    return ruta


# ── 2. lectura ───────────────────────────────────────────────────────────────

def _ultima_bajada():
    fechas = sorted(p.name for p in CRUDO.glob("*") if p.is_dir())
    if not fechas:
        raise FileNotFoundError(f"no hay nada bajado en {CRUDO}")
    return fechas[-1]


def _leer_crudo(fecha=None):
    fecha = fecha or _ultima_bajada()
    ruta = CRUDO / fecha / "establecimientos_salud_deis.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"falta {ruta} — correr `bajar` primero")
    texto = ruta.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(io.StringIO(texto), delimiter=";")), fecha


def _plausible(lat, lon):
    """Chile continental, insular y antártico, con margen.

    Es lo que atrapa el registro 120205, que trae una longitud metida en el
    campo de latitud. No se corrige: se declara sin coordenada usable."""
    if lat is None or lon is None:
        return False
    return -90.0 <= lat <= -17.0 and -110.0 <= lon <= -66.0


def _coordenada(fila):
    lat, lon = fila.get("Latitud", ""), fila.get("Longitud", "")
    if not lat or not lon:
        return None, None, "sin_coordenada_en_la_fuente"
    try:
        lat, lon = float(lat.replace(",", ".")), float(lon.replace(",", "."))
    except (TypeError, ValueError):
        return None, None, "coordenada_ilegible"
    if not _plausible(lat, lon):
        return None, None, "coordenada_fuera_de_chile_NO_usada"
    return lat, lon, "coordenada_publicada_por_el_DEIS"


def _vigente(fila):
    """★ El mismo estado viene escrito con distinta capitalización. Comparar en
    minúsculas o se pierden 210 establecimientos en silencio."""
    return (fila.get("EstadoFuncionamiento") or "").strip().lower()\
        .startswith("vigente")


CAMPOS_COMUNES = ["id_activo", "item_micr", "nombre", "lat", "lon",
                  "comuna", "provincia", "region", "cut", "zona_geografica",
                  "operador", "fuente", "fecha_captura", "confianza_ubicacion"]


def tabla(fecha=None, terr=None):
    """El maestro del DEIS normalizado al esquema común del inventario."""
    crudas, fecha = _leer_crudo(fecha)
    terr = terr if terr is not None else territorio.Territorio()
    filas = []
    sin_coord = vigentes = con_urgencia = alta_complejidad = 0

    for r in crudas:
        lat, lon, origen = _coordenada(r)
        if lat is None:
            sin_coord += 1
        vig = _vigente(r)
        vigentes += 1 if vig else 0
        if lat is None:
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
        urg = (r.get("TieneServicioUrgencia") or "").strip().upper() == "SI"
        con_urgencia += 1 if (urg and vig) else 0
        if (r.get("NivelComplejidadEstabGlosa") or "").strip().lower()\
                .startswith("alta") and vig:
            alta_complejidad += 1

        filas.append({
            "id_activo": f"SAL-{r.get('EstablecimientoCodigo', '')}",
            "item_micr": "establecimientos_de_salud",
            "nombre": (r.get("EstablecimientoGlosa") or "").strip(),
            "lat": lat if lat is not None else "",
            "lon": lon if lon is not None else "",
            "comuna": u["comuna"], "provincia": u["provincia"],
            "region": u["region"], "cut": u["codigo_comuna"],
            "zona_geografica": u["zona_geografica"],
            "operador": (r.get("DependenciaAdministrativa") or "").strip(),
            "fuente": "MINSAL · DEIS · Establecimientos de Salud vigentes "
                      "(datos.gob.cl, CC-Zero)",
            "fecha_captura": fecha,
            "confianza_ubicacion": origen,
            # ── propios ──
            "codigo_establecimiento": (r.get("EstablecimientoCodigo") or "").strip(),
            "tipo_establecimiento": (r.get("TipoEstablecimientoGlosa") or "").strip(),
            "vigente": "si" if vig else "no",
            "estado_funcionamiento": (r.get("EstadoFuncionamiento") or "").strip(),
            "sistema_salud": (r.get("TipoSistemaSaludGlosa") or "").strip(),
            "nivel_atencion": (r.get("NivelAtencionEstabglosa") or "").strip(),
            "nivel_complejidad": (r.get("NivelComplejidadEstabGlosa") or "").strip(),
            "tipo_atencion": (r.get("TipoAtencionEstabGlosa") or "").strip(),
            "tiene_urgencia": "si" if urg else "no",
            "tipo_urgencia": (r.get("TipoUrgencia") or "").strip(),
            "clasificacion_sapu": (r.get("ClasificacionTipoSapu") or "").strip(),
            "servicio_de_salud": (
                r.get("SeremiSaludGlosa_ServicioDeSaludGlosa") or "").strip(),
            "cut_declarado": (r.get("ComunaCodigo") or "").strip(),
            "comuna_declarada": (r.get("ComunaGlosa") or "").strip(),
            "region_declarada": (r.get("RegionGlosa") or "").strip(),
            "direccion": " ".join(x for x in (
                (r.get("TipoViaGlosa") or "").strip(),
                (r.get("NombreVia") or "").strip(),
                (r.get("Numero") or "").strip()) if x),
            "fecha_inicio_funcionamiento": (
                r.get("FechaInicioFuncionamientoEstab") or "").strip(),
            "fecha_cierre": (r.get("FechaCierre") or "").strip(),
            "territorio_faltan": u["faltan"],
        })

    campos = CAMPOS_COMUNES + [
        "codigo_establecimiento", "tipo_establecimiento", "vigente",
        "estado_funcionamiento", "sistema_salud", "nivel_atencion",
        "nivel_complejidad", "tipo_atencion", "tiene_urgencia", "tipo_urgencia",
        "clasificacion_sapu", "servicio_de_salud", "cut_declarado",
        "comuna_declarada", "region_declarada", "direccion",
        "fecha_inicio_funcionamiento", "fecha_cierre", "territorio_faltan"]
    ruta = DATOS / "inventario_salud.csv"
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)

    con_coord = sum(1 for f in filas if f["lat"] != "")
    vig_coord = sum(1 for f in filas if f["lat"] != "" and f["vigente"] == "si")
    print(f"   → datos/inventario_salud.csv  ({len(filas)} filas)")
    print(f"     vigentes: {vigentes} · con coordenada: {con_coord} · "
          f"vigentes CON coordenada: {vig_coord}")
    print(f"     sin coordenada usable: {sin_coord}")
    print(f"     vigentes con servicio de urgencia: {con_urgencia}")
    print(f"     vigentes de ALTA complejidad: {alta_complejidad} "
          f"← los que no se pueden reemplazar")
    _cuenta_por_tipo(filas)
    return filas


def _cuenta_por_tipo(filas):
    """Qué hay, por tipo de establecimiento. Se imprime porque un inventario que
    sólo dice «5.717» no dice nada: 5.717 postas rurales y 5.717 hospitales son
    problemas distintos."""
    cuenta = {}
    for f in filas:
        if f["vigente"] != "si" or f["lat"] == "":
            continue
        cuenta[f["tipo_establecimiento"]] = cuenta.get(f["tipo_establecimiento"], 0) + 1
    print("     los diez tipos más numerosos (vigentes y ubicados):")
    for tipo, n in sorted(cuenta.items(), key=lambda x: -x[1])[:10]:
        print(f"        {tipo[:44]:44s} {n:5d}")


def main():
    orden = sys.argv[1] if len(sys.argv) > 1 else "bajar"
    if orden == "bajar":
        bajar()
    elif orden == "tabla":
        tabla()
    elif orden == "todo":
        bajar()
        tabla()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
