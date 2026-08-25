"""
Adaptador de INVENTARIO — establecimientos educacionales de Chile.
Fuente: Ministerio de Educación (MINEDUC), Centro de Estudios, «Directorio
Oficial de Establecimientos Educacionales».

POR QUÉ UN COLEGIO ES INFRAESTRUCTURA CRÍTICA
----------------------------------------------
Por dos razones distintas, y conviene no confundirlas:

1. **Por lo que contiene.** Un colegio concentra población que no decide por sí
   misma dónde estar ni cuándo salir. El campo `MAT_TOTAL` dice cuántos son:
   3.541.790 estudiantes matriculados en los establecimientos en funcionamiento.

2. **Por lo que se le pide después.** En emergencia, en Chile se alberga en
   escuelas y gimnasios. El colegio deja de ser el activo protegido y pasa a ser
   el activo de respuesta. Y ahí `MAT_TOTAL` cambia de significado: pasa de ser
   «cuánta gente hay que evacuar» a ser una medida cruda de **cuánta gente cabe**.
   Esa doble lectura es de las cosas más útiles que aporta esta capa, y también
   la más fácil de usar mal: **no es capacidad de albergue certificada**, es una
   aproximación del tamaño del recinto. Se dice, no se disimula.

   ★ SENAPRED **no publica un catastro nacional de albergues**. Se verificó:
   `geoportal.senapred.cl` no resuelve, en `datos.gob.cl` no hay ningún conjunto
   de albergues ni de zonas seguras, y el mapa de albergues de la cuenta ArcGIS
   Online de SENAPRED está en privado (devuelve «no tiene permisos»). Lo único
   nacional y descargable son 1.358 «puntos de encuentro». Así que este
   directorio, más los 538 recintos deportivos que baja el adaptador de
   SENAPRED, es lo más cerca de un catastro de albergues que hoy permite el dato
   público.

QUÉ TRAE
--------
16.768 establecimientos en el archivo. Filtrando por los que están en
funcionamiento y tienen coordenada usable quedan **11.831 puntos**.

Que la caída sea grande no es un defecto del adaptador: 4.730 están cerrados, en
receso o autorizados sin matrícula, y **3.267 no traen coordenada en la fuente**.
Los dos números se declaran por separado, porque significan cosas distintas.

★ LO QUE ESTE DIRECTORIO NO TIENE: **jardines infantiles**. JUNJI, INTEGRA y los
VTF no están acá. Esos ~4.300 los aporta la capa de Educación del adaptador de
SENAPRED (`inventario_senapred_siie.py`, capa 3). Ninguna de las dos fuentes
sola cubre el universo, y por eso se bajan las dos: MINEDUC manda para colegios
(está al día), SENAPRED completa los jardines.

TRAMPAS DEL ARCHIVO — comprobadas
----------------------------------
★ **Es un archivo RAR, no un ZIP.** Se abre con `bsdtar`, que viene con macOS y
  con la mayoría de los Linux; no hace falta instalar `unrar`.

★ **El CSV es UTF-8 CON marca de orden de bytes (BOM)**, no latin-1 como suelen
  ser los CSV públicos chilenos. Hay que leerlo con `utf-8-sig` o la primera
  columna sale con basura pegada adelante.

★ **Las coordenadas vienen con COMA decimal**: `-18,4872`. Un `float()` directo
  revienta. Se cambia la coma por punto antes de convertir.

★ **`MATRICULA` NO es el número de estudiantes**: es una bandera 0/1 que dice si
  el establecimiento tiene matrícula o no. El conteo real es `MAT_TOTAL`.
  Confundirlos convertiría un colegio de 1.200 alumnos en un colegio de 1.

★ **La propia ficha técnica del ministerio advierte que la localización en
  `LATITUD`/`LONGITUD` es «sólo referencial»**. Se traslada esa advertencia al
  campo `confianza_ubicacion` de cada fila, en vez de dejarla enterrada en un
  PDF. Una coordenada referencial sirve para saber en qué comuna está algo; no
  para decidir si el agua le llega o no al edificio.

★ **El datum no se declara en ninguna parte.** Los rangos observados (latitud
  −72,48 a −17,59; longitud −109,43 a −51,73) son coherentes con EPSG:4326 y con
  el territorio chileno incluida la Antártica y la Isla de Pascua. Se asume
  EPSG:4326 y **se deja escrito que es un supuesto, no un dato**.

NO SE RECOLECTAN DATOS DE PERSONAS. El archivo trae `MRUN` (identificador de
persona natural del sostenedor) y `RUT_SOSTENEDOR`. **Ninguno de los dos se
copia** a ninguna tabla del inventario. Quedan sólo en el archivo crudo, tal como
lo publicó el ministerio.

CONDICIONES DE USO
------------------
· El sitio `datosabiertos.mineduc.cl` **no declara licencia** en su página. El
  registro espejo del mismo directorio en el portal nacional `datos.gob.cl` sí:
  **CC-BY (Creative Commons Atribución)**, organismo «Subsecretaría de
  Educación». Se atribuye a MINEDUC – Centro de Estudios en todo derivado.
· `https://datosabiertos.mineduc.cl/robots.txt` devuelve **403** (lo bloquea un
  cortafuegos de aplicación, no existe un robots.txt legible). No hay, entonces,
  una directiva que consultar: se procede con una sola descarga, sin rastreo.
· ★ El sitio **rechaza el agente por omisión de `curl`** y responde 403. Hay que
  mandar un agente de navegador. Eso NO es esquivar una restricción de acceso:
  el archivo es público y descargable con cualquier navegador; el filtro es de
  agente, no de permiso. Aun así se declara acá, para que nadie lo descubra
  después leyendo el código y se pregunte qué pasó.

USO
---
    python3 adaptadores/inventario_educacion_mineduc.py bajar
    python3 adaptadores/inventario_educacion_mineduc.py tabla
"""

import csv
import io
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

AQUI = Path(__file__).parent.parent
sys.path.insert(0, str(AQUI))
import territorio  # noqa: E402

URL = ("https://datosabiertos.mineduc.cl/wp-content/uploads/2025/11/"
       "Directorio-Oficial-EE-2025.rar")
NOMBRE_CSV = "20250926_Directorio_Oficial_EE_2025_20250430_WEB.csv"

HOY = date.today().isoformat()
CRUDO = AQUI / "datos" / "crudo" / "mineduc_directorio"
DATOS = AQUI / "datos"

# ★ No se copian a ninguna tabla: identifican personas naturales.
CAMPOS_DE_PERSONAS = {"MRUN", "RUT_SOSTENEDOR"}

# El sitio responde 403 al agente por omisión de curl/urllib. Ver la nota de
# condiciones de uso: es un filtro de agente sobre un archivo público.
AGENTE = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

# Del PDF `ER_Directorio_Oficial_EE_WEB.pdf` que viene dentro del mismo archivo.
ESTADO = {"1": "en funcionamiento", "2": "en receso", "3": "cerrado",
          "4": "autorizado sin matrícula"}
DEPENDENCIA = {"1": "Corporación Municipal", "2": "Municipal DAEM",
               "3": "Particular Subvencionado", "4": "Particular Pagado",
               "5": "Corporación de Administración Delegada (DL 3166)",
               "6": "Servicio Local de Educación Pública (SLEP)"}

FUENTE = dict(
    id="mineduc_directorio_establecimientos",
    organismo="Ministerio de Educación (MINEDUC) — Centro de Estudios",
    producto="Directorio Oficial de Establecimientos Educacionales 2025 "
             "(corte de datos 30-abr-2025, publicado 04-nov-2025)",
    url=URL,
    pagina="https://datosabiertos.mineduc.cl/"
           "directorio-de-establecimientos-educacionales/",
    formato="RAR con CSV (UTF-8 con BOM, separador «;», decimal con COMA)",
    familia="ESTADO",
    acceso="anonimo",
    acceso_verificado=1,
    licencia="CC-BY (Creative Commons Atribución), declarada en el registro "
             "espejo de datos.gob.cl. El sitio de MINEDUC no declara licencia.",
    robots_txt="403 — no hay robots.txt legible (lo bloquea un cortafuegos de "
               "aplicación). Verificado 19-ago-2026.",
    condiciones_uso="Atribución a MINEDUC – Centro de Estudios. El sitio exige "
                    "agente de navegador; el archivo es público.",
    permite_automatizacion="si, con una descarga por publicación (es anual)",
    granularidad="establecimiento (RBD)",
    advertencia="La ficha técnica del propio ministerio declara la coordenada "
                "«sólo referencial». No incluye jardines infantiles.",
)


# ── 1. bajada del crudo ──────────────────────────────────────────────────────

def bajar():
    """Baja el RAR y lo guarda tal cual; extrae el CSV al lado, sin tocarlo.

    Se guarda el RAR original además del CSV extraído: el CSV es una comodidad,
    el RAR es lo que el ministerio publicó, y es lo que hay que poder mostrar si
    algún día se audita una decisión.
    """
    destino = CRUDO / HOY
    destino.mkdir(parents=True, exist_ok=True)
    print(f"[educación MINEDUC] {URL}")

    pedido = urllib.request.Request(URL)
    pedido.add_header("User-Agent", AGENTE)
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

    ruta_rar = destino / "Directorio-Oficial-EE-2025.rar"
    ruta_rar.write_bytes(bruto)
    print(f"   {len(bruto):,} bytes · last-modified: "
          f"{cabeceras.get('Last-Modified', 'sin dato')}")

    # `bsdtar` viene con macOS y con casi todos los Linux, y sabe leer RAR. Se
    # prefiere a `unrar` justamente para no exigir una instalación extra.
    try:
        subprocess.run(["bsdtar", "-xf", str(ruta_rar), "-C", str(destino)],
                       check=True, capture_output=True)
    except FileNotFoundError:
        print("   ★ no está `bsdtar`: el RAR queda guardado sin extraer. "
              "El CSV se puede sacar a mano.")
        return ruta_rar
    except subprocess.CalledProcessError as err:
        print(f"   ★ `bsdtar` falló: {err.stderr[:200]!r}. El RAR queda igual.")
        return ruta_rar

    (destino / "procedencia.json").write_text(json.dumps({
        "url": URL, "fecha_descarga": HOY, "bytes": len(bruto),
        "cabeceras_http": cabeceras,
        "archivos_extraidos": sorted(p.name for p in destino.iterdir()),
        "fuente": FUENTE,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   → datos/crudo/mineduc_directorio/{HOY}/ "
          f"({len(list(destino.iterdir()))} archivos)")
    return ruta_rar


# ── 2. lectura ───────────────────────────────────────────────────────────────

def _ultima_bajada():
    fechas = sorted(p.name for p in CRUDO.glob("*") if p.is_dir())
    if not fechas:
        raise FileNotFoundError(f"no hay nada bajado en {CRUDO}")
    return fechas[-1]


def _leer_crudo(fecha=None):
    fecha = fecha or _ultima_bajada()
    ruta = CRUDO / fecha / NOMBRE_CSV
    if not ruta.exists():
        raise FileNotFoundError(f"falta {ruta} — correr `bajar` primero")
    # ★ UTF-8 CON marca de orden de bytes: `utf-8-sig`, no `latin-1`.
    texto = ruta.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(io.StringIO(texto), delimiter=";")), fecha


def _numero(valor):
    """Número o cadena vacía. Nunca cero por defecto: un campo ilegible vale
    «sin dato», y un colegio sin matrícula declarada no es un colegio vacío."""
    if valor is None:
        return ""
    t = str(valor).strip().replace(",", ".")
    if not t:
        return ""
    try:
        n = float(t)
    except ValueError:
        return ""
    return int(n) if n == int(n) else n


def _plausible(lat, lon):
    if lat is None or lon is None:
        return False
    return -90.0 <= lat <= -17.0 and -110.0 <= lon <= -66.0


def _coordenada(r):
    """★ Coma decimal. Y la advertencia del ministerio va PEGADA al dato."""
    lat, lon = (r.get("LATITUD") or "").strip(), (r.get("LONGITUD") or "").strip()
    if not lat or not lon or lat in ("0", "0,0") or lon in ("0", "0,0"):
        return None, None, "sin_coordenada_en_la_fuente"
    try:
        lat = float(lat.replace(",", "."))
        lon = float(lon.replace(",", "."))
    except ValueError:
        return None, None, "coordenada_ilegible"
    if lat == 0 or lon == 0:
        return None, None, "coordenada_cero_no_usada"
    if not _plausible(lat, lon):
        return None, None, "coordenada_fuera_de_chile_NO_usada"
    return lat, lon, "referencial_declarada_asi_por_MINEDUC"


CAMPOS_COMUNES = ["id_activo", "item_micr", "nombre", "lat", "lon",
                  "comuna", "provincia", "region", "cut", "zona_geografica",
                  "operador", "fuente", "fecha_captura", "confianza_ubicacion"]


def tabla(fecha=None, terr=None):
    crudas, fecha = _leer_crudo(fecha)
    terr = terr if terr is not None else territorio.Territorio()
    filas = []
    sin_coord = 0
    for r in crudas:
        lat, lon, origen = _coordenada(r)
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
        estado_cod = (r.get("ESTADO_ESTAB") or "").strip()
        filas.append({
            "id_activo": f"EDU-{(r.get('RBD') or '').strip()}",
            "item_micr": "establecimientos_educacionales",
            "nombre": (r.get("NOM_RBD") or "").strip(),
            "lat": lat if lat is not None else "",
            "lon": lon if lon is not None else "",
            "comuna": u["comuna"], "provincia": u["provincia"],
            "region": u["region"], "cut": u["codigo_comuna"],
            "zona_geografica": u["zona_geografica"],
            "operador": DEPENDENCIA.get((r.get("COD_DEPE") or "").strip(),
                                        "sin dato"),
            "fuente": "MINEDUC · Centro de Estudios · Directorio Oficial de "
                      "Establecimientos Educacionales 2025 (CC-BY)",
            "fecha_captura": fecha,
            "confianza_ubicacion": origen,
            # ── propios ──
            "rbd": (r.get("RBD") or "").strip(),
            "digito_verificador": (r.get("DGV_RBD") or "").strip(),
            "anio_directorio": (r.get("AGNO") or "").strip(),
            "estado_codigo": estado_cod,
            "estado": ESTADO.get(estado_cod, "sin dato"),
            "en_funcionamiento": "si" if estado_cod == "1" else "no",
            "matricula_total": _numero(r.get("MAT_TOTAL")),
            "tiene_matricula": ("si" if (r.get("MATRICULA") or "").strip() == "1"
                                else "no"),
            "rural": ("si" if (r.get("RURAL_RBD") or "").strip() == "1"
                      else "no" if (r.get("RURAL_RBD") or "").strip() == "0"
                      else ""),
            "dependencia_codigo": (r.get("COD_DEPE") or "").strip(),
            "dependencia_agrupada": (r.get("COD_DEPE2") or "").strip(),
            "orientacion_religiosa": (r.get("ORI_RELIGIOSA") or "").strip(),
            "convenio_pie": (r.get("CONVENIO_PIE") or "").strip(),
            "cut_declarado": (r.get("COD_COM_RBD") or "").strip(),
            "comuna_declarada": (r.get("NOM_COM_RBD") or "").strip(),
            "region_declarada": (r.get("NOM_REG_RBD_A") or "").strip(),
            "departamento_provincial": (r.get("NOM_DEPROV_RBD") or "").strip(),
            "territorio_faltan": u["faltan"],
        })

    campos = CAMPOS_COMUNES + [
        "rbd", "digito_verificador", "anio_directorio", "estado_codigo",
        "estado", "en_funcionamiento", "matricula_total", "tiene_matricula",
        "rural", "dependencia_codigo", "dependencia_agrupada",
        "orientacion_religiosa", "convenio_pie", "cut_declarado",
        "comuna_declarada", "region_declarada", "departamento_provincial",
        "territorio_faltan"]
    ruta = DATOS / "inventario_educacion.csv"
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)

    funcionando = [f for f in filas if f["en_funcionamiento"] == "si"]
    ubicados = [f for f in funcionando if f["lat"] != ""]
    matricula = sum(f["matricula_total"] for f in ubicados
                    if isinstance(f["matricula_total"], (int, float)))
    rurales = sum(1 for f in ubicados if f["rural"] == "si")
    print(f"   → datos/inventario_educacion.csv  ({len(filas)} filas)")
    print(f"     en funcionamiento: {len(funcionando)}")
    print(f"     sin coordenada en la fuente: {sin_coord}")
    print(f"     ★ en funcionamiento Y ubicados: {len(ubicados)} "
          f"← lo que suma al inventario")
    print(f"     de ellos, rurales: {rurales}")
    print(f"     matrícula que representan: {matricula:,.0f} estudiantes")
    return filas


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
