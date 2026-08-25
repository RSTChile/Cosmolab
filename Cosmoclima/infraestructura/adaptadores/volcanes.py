"""
ADAPTADOR · AMENAZA VOLCÁNICA (SERNAGEOMIN — RNVV / OVDAS)
==========================================================

Qué organismo es el dueño de este dato
---------------------------------------
**SERNAGEOMIN** (Servicio Nacional de Geología y Minería) vigila los volcanes de
Chile a través de dos unidades:

  · **RNVV** — Red Nacional de Vigilancia Volcánica. Es el programa que mantiene
    la vigilancia en tiempo real y define los **niveles de alerta técnica
    volcánica** (verde, amarilla, naranja, roja).
  · **OVDAS** — Observatorio Volcanológico de los Andes del Sur, con sede en
    Temuco. Es la sala 24/7 que procesa la señal instrumental. Monitorea
    **43 volcanes priorizados** de los 87 activos del catálogo.
  · **UGPSV** — unidad de Geología y Peligros de Sistemas Volcánicos. Es la que
    dibuja los **mapas de peligro zonificados**, que son los polígonos que a
    este proyecto le sirven para cruzar con infraestructura.

Este adaptador NO monitorea volcanes ni emite alertas. Toma lo que SERNAGEOMIN
ya publica y lo deja en un formato que el consolidador nacional pueda cruzar con
el inventario de infraestructura crítica.

QUÉ TRAE ESTE ADAPTADOR (y qué NO — leer FUENTE_VOLCANES.md)
-------------------------------------------------------------
1. `--catalogo` · **Ranking de Riesgo Específico de Volcanes Activos de Chile,
   2023**: 87 sistemas volcánicos con coordenadas, región, tipo, factor de
   peligro, factor de exposición, riesgo específico y si OVDAS lo monitorea.
   Es el catálogo pedido en el punto (c) de la tarea. Se trae también el ranking
   **2019** (91 volcanes) porque trae campos que el de 2023 no tiene.

2. `--peligro` · **Mapa de peligro volcánico zonificado** que SERNAGEOMIN publica
   como capa vectorial en la nube: 5 polígonos con grado de peligro. Cubre UN
   volcán (Nevados de Chillán). No hay capa nacional de polígonos de peligro
   volcánico accesible — el porqué está documentado en `FUENTE_VOLCANES.md`.

3. `--alerta` · **Foto fechada del estado de alerta**. ★ Ver la advertencia de
   abajo: hoy este canal está degradado por decisión de la propia fuente.

4. `--cruce` · Cruce geométrico exacto contra la infraestructura ya inventariada
   (14.039 tramos viales, 6.742 puentes, 39 subestaciones eléctricas).

★★ ADVERTENCIA SOBRE EL ESTADO DE ALERTA — LEER ANTES DE PROGRAMAR NADA
------------------------------------------------------------------------
El estado de alerta por volcán es un dato **dinámico y sin historia pública**,
igual que la Minuta Técnica de remoción en masa. Pero hoy es PEOR que la minuta,
por dos razones distintas que conviene no confundir:

  (a) **La fuente misma declara el canal caído.** La página oficial de alertas
      volcánicas dice, textualmente: «La vigilancia no se ha visto interrumpida;
      no obstante, la visualización en tiempo real de esta información no se
      encuentra disponible en el sitio web». Es decir: SERNAGEOMIN sigue
      vigilando, pero **dejó de publicar el tablero de alertas**. Lo único que
      queda visible es el volcán que está en alerta en ese momento, en prosa.

  (b) **El sitio propio de la RNVV está detrás de un muro de autenticación.**
      `rnvv.sernageomin.cl` resuelve a Cloudflare Access y pide correo y código
      de acceso. No es una caída: es una puerta cerrada a propósito.

Por eso este adaptador, en modo `--alerta`, **no inventa un tablero**. Guarda una
copia fechada de la página tal como está y extrae SÓLO lo que efectivamente dice,
declarando «sin dato» para los 86 volcanes restantes. Un «sin dato» no es un
«verde»: son dos cosas distintas y confundirlas sería inventar tranquilidad.

Verificado el 19-ago-2026.
"""

import csv
import json
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from html import unescape
from pathlib import Path

AQUI = Path(__file__).resolve().parent.parent          # …/infraestructura
CRUDO = AQUI / "datos" / "crudo" / "volcanes"
DATOS = AQUI / "datos"

# --- fuentes -----------------------------------------------------------------

# Servicio ArcGIS institucional de SERNAGEOMIN en la nube de Esri. Es el mismo
# que ya usa `traer_capas_sernageomin.py` para la minuta de remoción en masa,
# así que las dos amenazas comparten servidor y no hay que aprender otro.
ARCGIS = ("https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/services")

CAPAS = {
    # nombre corto            (servicio, capa, para qué sirve)
    "ranking_2023": (
        "Ranking_Volcanes_2023_v2", 0,
        "Ranking de Riesgo Específico 2023: 87 sistemas volcánicos, con "
        "coordenadas, factor de peligro, factor de exposición y riesgo"),
    "ranking_2019": (
        "PUBLICACIONES_RNVV", 3,
        "Ranking 2019: 91 volcanes, con tipo y categoría; conserva campos que "
        "el de 2023 no publica"),
    "informes_pet": (
        "PUBLICACIONES_RNVV", 1,
        "Informes técnicos PET (Peligros y Evaluación Territorial) por volcán, "
        "con su estado de avance: dice QUÉ volcán tiene mapa de peligro hecho"),
    "peligro_zonificado": (
        "Peligro_Geológico_Shape", 0,
        "Mapa de peligro volcánico zonificado (polígonos con grado de peligro)"),
}

# Página institucional donde SERNAGEOMIN publica hoy el estado de alerta.
PAGINA_ALERTAS = "https://www.sernageomin.cl/alertas-volcanicas/"
PAGINA_RNVV = "https://www.sernageomin.cl/rnvv/"

# El listado completo del ranking, en PDF. Se archiva como respaldo de que la
# capa vectorial y el documento oficial dicen lo mismo.
PDF_RANKING = ("https://www.sernageomin.cl/wp-content/uploads/2026/04/"
               "Ranking-2023_tabloide_20231012.pdf")

AGENTE = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# Los cuatro niveles de alerta técnica volcánica que define la RNVV, en orden.
# Se escriben aquí para poder RECONOCERLOS en el texto, nunca para suponerlos.
NIVELES_ALERTA = ["verde", "amarilla", "naranja", "roja"]


# =============================================================================
# 1 · UTILIDADES DE RED
#     Ninguna de estas funciones lanza excepción hacia arriba. Un fallo de red
#     tiene que quedar anotado como HUECO, no tumbar la corrida ni convertirse
#     en un cero silencioso. Es la regla 1 del consolidador.
# =============================================================================

def _abrir(url, timeout=90):
    """Petición HTTP simple. Devuelve (bytes, None) o (None, motivo).

    ★ POR QUÉ HAY UN SEGUNDO INTENTO CON `curl`
    --------------------------------------------
    `www.sernageomin.cl` entrega su certificado **incompleto**: manda el suyo
    pero se olvida del certificado intermedio que lo respalda. El navegador y
    `curl` disimulan el error porque salen a buscar la pieza que falta por su
    cuenta; Python no hace eso y corta la conexión.

    Es un defecto de configuración del servidor, no una sospecha de seguridad.
    Por eso el remedio es **delegar en `curl`, que SÍ verifica el certificado**,
    y no apagar la verificación. Apagarla sería aceptar cualquier servidor que
    se haga pasar por SERNAGEOMIN, y el dato de este proyecto va a decisión
    pública.
    """
    try:
        pedido = urllib.request.Request(url, headers={"User-Agent": AGENTE})
        with urllib.request.urlopen(pedido, timeout=timeout) as r:
            return r.read(), None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            ssl.SSLError, OSError) as e:
        motivo = f"{type(e).__name__}: {str(e)[:150]}"
        if "CERTIFICATE_VERIFY_FAILED" not in str(e):
            return None, motivo
    try:
        import subprocess
        r = subprocess.run(
            ["curl", "-sS", "-L", "--max-time", str(timeout),
             "-A", AGENTE, url],
            capture_output=True, timeout=timeout + 15)
        if r.returncode != 0 or not r.stdout:
            return None, (f"{motivo} | reintento con curl también falló: "
                          f"{r.stderr.decode('utf8', 'replace')[:120]}")
        return r.stdout, None
    except Exception as e:                                   # noqa: BLE001
        return None, f"{motivo} | reintento con curl: {str(e)[:120]}"


def traer_capa_arcgis(servicio, capa, timeout=120, pagina=1000):
    """Descarga una capa completa del servicio ArcGIS como GeoJSON.

    Pagina de a mil rasgos porque el servicio corta en 2000 por respuesta.
    Devuelve (geojson, None) o (None, motivo).
    """
    rasgos, offset = [], 0
    while True:
        # El nombre del servicio puede llevar tildes («Peligro_Geológico»).
        # `quote` las convierte a la forma con % que exige HTTP; sin esto la
        # petición falla antes de salir del computador.
        ruta = urllib.parse.quote(f"{servicio}/FeatureServer/{capa}/query")
        url = (f"{ARCGIS}/{ruta}"
               f"?where=1%3D1&outFields=*&outSR=4326&f=geojson"
               f"&resultOffset={offset}&resultRecordCount={pagina}")
        crudo, motivo = _abrir(url, timeout)
        if crudo is None:
            return None, motivo
        try:
            datos = json.loads(crudo)
        except json.JSONDecodeError as e:
            return None, f"respuesta no es JSON: {str(e)[:100]}"
        if "error" in datos:
            return None, f"el servicio devolvió error: {datos['error']}"
        lote = datos.get("features", [])
        rasgos.extend(lote)
        if len(lote) < pagina:
            break
        offset += pagina
    return {"type": "FeatureCollection", "features": rasgos}, None


# =============================================================================
# 2 · CATÁLOGO DE VOLCANES  (--catalogo)
# =============================================================================

def _carpeta_hoy():
    c = CRUDO / date.today().isoformat()
    c.mkdir(parents=True, exist_ok=True)
    return c


def _escribir_csv(ruta, filas):
    """Escribe un CSV con la unión de todas las claves. Si no hay filas, no
    escribe nada y lo dice: un archivo vacío se confunde con «no hay peligro»."""
    if not filas:
        print(f"    (sin filas: NO se escribe {ruta.name})")
        return False
    columnas = []
    for f in filas:
        for k in f:
            if k not in columnas:
                columnas.append(k)
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)
    print(f"    → {ruta.relative_to(AQUI)}  ({len(filas)} filas)")
    return True


def modo_catalogo():
    """Baja el catálogo de volcanes y lo deja en CSV, con el crudo archivado."""
    carpeta = _carpeta_hoy()
    print("=" * 76)
    print("CATÁLOGO DE VOLCANES · SERNAGEOMIN / RNVV")
    print("=" * 76)
    problemas = []

    for nombre in ("ranking_2023", "ranking_2019", "informes_pet"):
        servicio, capa, para_que = CAPAS[nombre]
        geo, motivo = traer_capa_arcgis(servicio, capa)
        if geo is None:
            print(f"  ✗ {nombre}: {motivo}")
            problemas.append((nombre, motivo))
            continue
        # El crudo, tal como llegó, con la fecha en la ruta.
        (carpeta / f"{nombre}.geojson").write_text(
            json.dumps(geo, ensure_ascii=False), encoding="utf-8")
        print(f"  ✓ {nombre:18s} {len(geo['features']):4d} rasgos — {para_que}")

    # --- ranking 2023 → CSV de trabajo ---
    ruta = carpeta / "ranking_2023.geojson"
    if ruta.exists():
        geo = json.loads(ruta.read_text(encoding="utf-8"))
        filas = []
        for f in geo["features"]:
            p = f["properties"]
            g = f.get("geometry") or {}
            coords = g.get("coordinates") or [None, None]
            filas.append({
                "posicion_2023": p.get("Posición__2023"),
                "sistema_volcanico": (p.get("Sistema_volcánico") or "").strip(),
                "clasificacion": (p.get("Clasificación") or "").strip(),
                "region": (p.get("Región") or "").strip(),
                # «Categoría» va de 1 a 5 y agrupa el ranking en quintiles de
                # riesgo: 1 = muy alto, 5 = muy bajo.
                "categoria_riesgo": p.get("Categoría"),
                # 1 = OVDAS lo monitorea con instrumentos; 0 = no.
                "monitoreo_ovdas": p.get("Monitoreo_OVDAS"),
                "factor_peligro": p.get("Total_Factores_de_Peligro"),
                "factor_exposicion": p.get("Total_Factores_de_Exposición"),
                "riesgo_especifico": p.get("Total_Riesgo_Específico"),
                "observaciones": (p.get("Observaciones") or "").strip(),
                "lon": coords[0], "lat": coords[1],
            })
        filas.sort(key=lambda x: x["posicion_2023"] or 999)
        _escribir_csv(DATOS / "volcanes_ranking_2023.csv", filas)

    # --- ranking 2019 → CSV (aporta tipo de volcán y riesgo relativo) ---
    ruta = carpeta / "ranking_2019.geojson"
    if ruta.exists():
        geo = json.loads(ruta.read_text(encoding="utf-8"))
        filas = []
        for f in geo["features"]:
            p = f["properties"]
            g = f.get("geometry") or {}
            coords = g.get("coordinates") or [None, None]
            filas.append({
                "posicion_2019": p.get("RIESGO_ABSOLUTO"),
                "sistema_volcanico": (p.get("SISTEMA_VOLCANICO") or "").strip(),
                "nombre_volcan": (p.get("NOMBRE_VOLCAN") or "").strip(),
                "subtipo": (p.get("SUBTIPO_DESC") or "").strip(),
                "categoria_ranking": (p.get("CATEGORIA_RANKING") or "").strip(),
                "factor_peligro": p.get("FACTOR_PELIGRO"),
                "factor_exposicion": p.get("FACTOR_EXPOSICION"),
                "riesgo_relativo": p.get("RIESGO_RELATIVO"),
                "ano_ranking": p.get("ANO_RANKING"),
                "observacion": (p.get("OBSERVACION") or "").strip(),
                "lon": coords[0], "lat": coords[1],
            })
        filas.sort(key=lambda x: x["posicion_2019"] or 999)
        _escribir_csv(DATOS / "volcanes_ranking_2019.csv", filas)

    # --- informes PET: qué volcán tiene mapa de peligro elaborado ---
    ruta = carpeta / "informes_pet.geojson"
    if ruta.exists():
        geo = json.loads(ruta.read_text(encoding="utf-8"))
        filas = []
        for f in geo["features"]:
            p = f["properties"]
            g = f.get("geometry") or {}
            coords = g.get("coordinates") or [None, None]
            filas.append({
                "volcan": (p.get("NOMBRE_VOLCÁN") or "").strip(),
                "estado_informe_pet": (p.get("ESTADO") or "").strip(),
                "lon": coords[0], "lat": coords[1],
            })
        filas.sort(key=lambda x: x["volcan"])
        _escribir_csv(DATOS / "volcanes_informes_pet.csv", filas)

    # --- respaldo documental: el PDF oficial del ranking ---
    pdf, motivo = _abrir(PDF_RANKING, timeout=120)
    if pdf is None:
        print(f"  ✗ PDF del ranking: {motivo}")
        problemas.append(("pdf_ranking", motivo))
    else:
        (carpeta / "Ranking_2023_tabloide.pdf").write_bytes(pdf)
        print(f"  ✓ PDF oficial del ranking archivado ({len(pdf)/1024:.0f} KB)")

    if problemas:
        print("\n  ATENCIÓN: quedaron piezas sin traer. Se anotan como hueco.")
        for n, m in problemas:
            print(f"    · {n}: {m}")
    print(f"\n  crudo fechado en: {carpeta.relative_to(AQUI)}")
    return 1 if problemas else 0


# =============================================================================
# 3 · MAPA DE PELIGRO ZONIFICADO  (--peligro)
# =============================================================================

# Diccionario de códigos del mapa de peligro. SERNAGEOMIN publica los polígonos
# con códigos, no con palabras. Estas equivalencias vienen de la simbología de
# los propios mapas de peligro volcánico de SERNAGEOMIN.
#
# ★ Se declara explícitamente: los códigos PV-DEP-xx y GRADO-xx NO vienen
#   acompañados de su tabla de dominio en el servicio. Lo que sigue es la
#   lectura razonada de los códigos, y se marca como TAL. Para cualquier uso
#   que dependa del significado exacto hay que pedirle la tabla a SERNAGEOMIN.
CODIGOS_PELIGRO_SIN_CONFIRMAR = {
    "PV-DEP-02": "depósito volcánico tipo 2 (sin confirmar)",
    "PV-DEP-03": "depósito volcánico tipo 3 (sin confirmar)",
    "PV-DEP-07": "depósito volcánico tipo 7 (sin confirmar)",
    "PV-DEP-09": "depósito volcánico tipo 9 (sin confirmar)",
}


def modo_peligro():
    """Baja el mapa de peligro volcánico zonificado que sí está en la nube."""
    carpeta = _carpeta_hoy()
    print("=" * 76)
    print("MAPA DE PELIGRO VOLCÁNICO ZONIFICADO")
    print("=" * 76)

    servicio, capa, para_que = CAPAS["peligro_zonificado"]
    geo, motivo = traer_capa_arcgis(servicio, capa)
    if geo is None:
        print(f"  ✗ no se pudo traer: {motivo}")
        return 1

    (carpeta / "peligro_zonificado.geojson").write_text(
        json.dumps(geo, ensure_ascii=False), encoding="utf-8")

    from pyproj import Geod
    from shapely.geometry import shape
    geod = Geod(ellps="WGS84")

    filas = []
    for f in geo["features"]:
        p = f["properties"]
        s = shape(f["geometry"])
        area = abs(geod.geometry_area_perimeter(s)[0]) / 1e6
        b = s.bounds
        filas.append({
            "objectid": p.get("OBJECTID"),
            "nombre_zona": (p.get("NOMBRE") or "").strip(),
            "codigo_zona": (p.get("CODIGO") or "").strip(),
            "grado_peligro": (p.get("GRADO_RIES") or "").strip(),
            "peligro_1": (p.get("PELIGRO_VO") or "").strip(),
            "peligro_2": (p.get("PELIGRO__1") or "").strip(),
            "peligro_3": (p.get("PELIGRO__2") or "").strip(),
            "magnitud_esperada": (p.get("MAGNITUD_E") or "").strip(),
            "escala_mapa": p.get("ESCALA"),
            "area_km2": round(area, 2),
            "lon_min": round(b[0], 4), "lat_min": round(b[1], 4),
            "lon_max": round(b[2], 4), "lat_max": round(b[3], 4),
        })
        print(f"  · {filas[-1]['grado_peligro']:9s} "
              f"{filas[-1]['peligro_1']:10s} {area:9,.1f} km²")

    total = sum(f["area_km2"] for f in filas)
    print(f"\n  {len(filas)} polígonos · {total:,.0f} km² zonificados")
    print("  ★ cobertura: UN complejo volcánico (Nevados de Chillán, Ñuble). "
          "NO es una capa nacional.")
    _escribir_csv(DATOS / "volcanes_peligro_zonificado.csv", filas)
    print(f"\n  crudo fechado en: {carpeta.relative_to(AQUI)}")
    return 0


# =============================================================================
# 4 · ESTADO DE ALERTA  (--alerta)
#     Foto fechada. Liviana a propósito: unos KB, para poder correrla varias
#     veces al día sin costo, igual que el snapshot de la minuta.
# =============================================================================

def _texto_plano(html):
    """HTML → texto legible. No usa ninguna biblioteca externa a propósito:
    este adaptador tiene que poder correr aunque no haya nada instalado."""
    t = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    t = re.sub(r"<style.*?</style>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", "\n", t)
    t = unescape(t)
    return "\n".join(l.strip() for l in t.split("\n") if l.strip())


def _leer_reav(ruta_pdf):
    """Lee un Reporte Especial de Actividad Volcánica (REAV) en PDF.

    ★★ HALLAZGO QUE HAY QUE DECLARAR, NO ESCONDER
    -----------------------------------------------
    El REAV dice en su texto «la alerta técnica volcánica se cambia a:» **y a
    continuación pone una IMAGEN**, no una palabra. El nivel —amarilla, naranja,
    roja— está dibujado, no escrito. Verificado en el REAV del complejo
    volcánico Nevados de Chillán del 15-jun-2026.

    Consecuencia práctica: del PDF se puede sacar **que hubo un cambio de alerta,
    de qué volcán y cuándo**, pero NO a qué nivel cambió. Este lector devuelve
    `nivel_alerta = "SIN DATO (viene como imagen en el PDF)"` y no adivina.
    Deducir un nivel de alerta oficial del color de unos píxeles sería
    exactamente lo que este proyecto no puede hacer.
    """
    salida = {"archivo": ruta_pdf.name, "volcan": "", "emitido": "",
              "hubo_cambio_de_alerta": False,
              "nivel_alerta": "SIN DATO", "motivo_sin_nivel": ""}
    try:
        from pypdf import PdfReader
    except ImportError:
        salida["motivo_sin_nivel"] = "no hay lector de PDF instalado"
        return salida
    try:
        texto = "\n".join(p.extract_text() or ""
                          for p in PdfReader(str(ruta_pdf)).pages)
    except Exception as e:                                   # noqa: BLE001
        salida["motivo_sin_nivel"] = f"no se pudo leer el PDF: {str(e)[:80]}"
        return salida

    m = re.search(r"Regi[óo]n de ([^,\n]+),\s*([^\n]+)", texto)
    if m:
        salida["volcan"] = m.group(2).strip()
        salida["region"] = m.group(1).strip()
    m = re.search(r"Emitido el ([^\n]+)", texto)
    if m:
        salida["emitido"] = m.group(1).strip()

    if re.search(r"alerta t[ée]cnica volc[áa]nica se (cambia|mantiene)", texto,
                 re.I):
        salida["hubo_cambio_de_alerta"] = True
        # ¿el nivel aparece escrito en alguna parte del texto? casi nunca.
        cerca = re.search(
            r"alerta t[ée]cnica volc[áa]nica se (?:cambia|mantiene) a:?\s*"
            r"(verde|amarilla|naranja|roja)", texto, re.I)
        if cerca:
            salida["nivel_alerta"] = cerca.group(1).lower()
        else:
            salida["motivo_sin_nivel"] = (
                "el REAV entrega el nivel como IMAGEN, no como texto; "
                "un humano lo lee de un vistazo, un programa no")
    return salida


def modo_alerta():
    """Guarda una foto fechada de lo que la fuente dice HOY sobre alertas.

    No arma un tablero de 87 volcanes: arma el tablero que la fuente permita, y
    deja el resto en «sin dato», que es distinto de «verde».
    """
    sello = datetime.now().strftime("%Y-%m-%dT%H%M")
    carpeta = CRUDO / "alerta_diaria"
    carpeta.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f"ESTADO DE ALERTA VOLCÁNICA · foto {sello}")
    print("=" * 76)

    crudo, motivo = _abrir(PAGINA_ALERTAS, timeout=60)
    if crudo is None:
        (carpeta / f"{sello}.FALLO.txt").write_text(motivo, encoding="utf-8")
        print(f"  ✗ {motivo}")
        print("    (anotado como hueco: hoy NO hubo dato, que no es lo mismo "
              "que «hoy no había alerta»)")
        return 1

    (carpeta / f"{sello}.html").write_bytes(crudo)
    texto = _texto_plano(crudo.decode("utf-8", "replace"))

    # ¿La fuente sigue declarando caído su tablero en tiempo real?
    canal_caido = "no se encuentra disponible en el sitio web" in texto

    # Qué volcanes NOMBRA la página en su sección de alertas, y con qué nivel si
    # es que lo dice. Se busca el nivel sólo en la vecindad del nombre; si no
    # aparece, se deja en blanco. Nunca se supone «verde».
    volcanes_citados = []
    trozo = texto
    i = trozo.find("Información de volcanes en alertas")
    if i >= 0:
        trozo = trozo[i:]
    for m in re.finditer(
            r"(?:Complejo\s+[Vv]olc[áa]nico|Volc[áa]n|Grupo\s+[Vv]olc[áa]nico|"
            r"Campo\s+[Vv]olc[áa]nico)\s+([A-ZÁÉÍÓÚÑ][^\n]{2,45})", trozo):
        nombre = m.group(1).strip(" .,-")
        ventana = trozo[max(0, m.start() - 200): m.end() + 600].lower()
        nivel = next((n for n in NIVELES_ALERTA
                      if re.search(rf"alerta\s+t[ée]cnica\s+{n}|"
                                   rf"alerta\s+{n}\b", ventana)), "")
        if nombre not in [v["volcan"] for v in volcanes_citados]:
            volcanes_citados.append({"volcan": nombre,
                                     "nivel_alerta": nivel or "SIN DATO"})

    # Los PDF de Reporte Especial de Actividad Volcánica (REAV) enlazados hoy.
    reav = sorted(set(re.findall(
        r'https://www\.sernageomin\.cl/wp-content/uploads/[^"\']*REAV[^"\']*\.pdf',
        crudo.decode("utf-8", "replace"), flags=re.I)))

    resumen = {
        "sello": sello,
        "url": PAGINA_ALERTAS,
        "canal_tiempo_real_declarado_caido": canal_caido,
        "volcanes_citados": volcanes_citados,
        "reav_enlazados": reav,
        # ★ Se guarda explícito para que nadie lo confunda con un tablero.
        "cobertura": (f"{len(volcanes_citados)} volcanes citados en la página; "
                      "los demás del catálogo quedan SIN DATO, que no es verde"),
    }
    (carpeta / f"{sello}.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  canal en tiempo real declarado caído por la fuente: "
          f"{'SÍ' if canal_caido else 'no'}")
    if volcanes_citados:
        print("  volcanes citados hoy:")
        for v in volcanes_citados:
            print(f"    · {v['volcan']:40s} alerta: {v['nivel_alerta']}")
    else:
        print("  ningún volcán citado en la página")
    for r in reav:
        nombre_pdf = r.rsplit("/", 1)[-1]
        print(f"  REAV enlazado: {nombre_pdf}")
        pdf, _ = _abrir(r, timeout=90)
        if pdf:
            (carpeta / nombre_pdf).write_bytes(pdf)
            resumen.setdefault("reav_leidos", []).append(
                _leer_reav(carpeta / nombre_pdf))
    (carpeta / f"{sello}.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    # ¿cambió respecto de la foto anterior? es lo único que vale la pena mirar
    previas = sorted(p for p in carpeta.glob("*.json")
                     if p.name != f"{sello}.json")
    if previas:
        ant = json.loads(previas[-1].read_text(encoding="utf-8"))
        igual = (ant.get("volcanes_citados") == volcanes_citados
                 and ant.get("canal_tiempo_real_declarado_caido") == canal_caido)
        print(f"  respecto de {previas[-1].name}: "
              f"{'sin cambios' if igual else 'CAMBIÓ ★'}")
    else:
        print("  primera foto")
    return 0


# =============================================================================
# 5 · CRUCE CON INFRAESTRUCTURA  (--cruce)
#
#     Se reutiliza tal cual la maquinaria de `cruzar_amenaza_exacto.py`:
#       · shapely 2.1.2 (trae GEOS adentro, no hace falta GDAL)
#       · pyproj 3.7.2 (largos y áreas geodésicas exactas sobre el elipsoide)
#       · STRtree (índice espacial de árbol: convierte un cruce de horas en
#         segundos, porque descarta de golpe todo lo que no puede tocar)
# =============================================================================

import gzip                                              # noqa: E402
from collections import defaultdict                      # noqa: E402

CRUDO_MOP = AQUI / "datos" / "crudo" / "mop" / "2026-08-17"
TRAMOS = CRUDO_MOP / "tramos.geojson.gz"
PUENTES = CRUDO_MOP / "puentes.geojson"
SUBESTACIONES = DATOS / "subestaciones_puntos.csv"

# Radios de cercanía, en kilómetros.
#
# ★★ DECLARACIÓN OBLIGATORIA: estos anillos NO SON UN MAPA DE PELIGRO.
#    Son círculos alrededor del cráter. El peligro volcánico real no es
#    circular: los lahares y los flujos piroclásticos bajan encauzados por los
#    valles y pueden llegar mucho más lejos por un lado que por el otro. Un
#    anillo sirve para UNA sola cosa: filtrar rápido qué infraestructura ni
#    siquiera hace falta mirar. Todo lo que caiga adentro hay que evaluarlo con
#    el mapa de peligro del volcán, no con este número.
RADIOS_KM = [5, 10, 20, 30]


def _circulo_geodesico(lon, lat, radio_km, n=90):
    """Círculo de radio constante sobre el elipsoide, como polígono de n lados.

    Se hace con `pyproj.Geod.fwd`, que avanza una distancia exacta en metros
    desde un punto en un azimut dado. Así el círculo mide lo mismo en Arica que
    en Magallanes, cosa que no pasaría si se sumara un número fijo de grados.
    """
    from pyproj import Geod
    from shapely.geometry import Polygon
    g = Geod(ellps="WGS84")
    pts = []
    for k in range(n):
        az = 360.0 * k / n
        x, y, _ = g.fwd(lon, lat, az, radio_km * 1000.0)
        pts.append((x, y))
    return Polygon(pts)


def _largo_m(geometria, geod):
    """Largo geodésico en metros de una línea o colección de líneas."""
    if geometria.is_empty:
        return 0.0
    if geometria.geom_type == "LineString":
        return geod.geometry_length(geometria)
    if geometria.geom_type in ("MultiLineString", "GeometryCollection"):
        return sum(_largo_m(g, geod) for g in geometria.geoms
                   if g.geom_type in ("LineString", "MultiLineString",
                                      "GeometryCollection"))
    return 0.0


def _cargar_infraestructura():
    """Devuelve (tramos, puentes, subestaciones) ya en geometría de shapely."""
    from shapely.geometry import Point, shape

    with gzip.open(TRAMOS, "rt", encoding="utf8") as fh:
        tr = json.load(fh)["features"]
    tramos = []
    for f in tr:
        try:
            tramos.append((shape(f["geometry"]), f["properties"]))
        except Exception:
            continue

    pts = json.loads(PUENTES.read_text(encoding="utf8"))["features"]
    puentes = []
    for f in pts:
        g = f.get("geometry") or {}
        if g.get("type") == "Point":
            puentes.append((Point(g["coordinates"][:2]), f["properties"]))

    subes = []
    if SUBESTACIONES.exists():
        with open(SUBESTACIONES, encoding="utf-8") as fh:
            for fila in csv.DictReader(fh):
                try:
                    lat = float(fila.get("lat") or fila.get("latitud"))
                    lon = float(fila.get("lon") or fila.get("longitud"))
                except (TypeError, ValueError):
                    continue
                subes.append((Point(lon, lat), fila))
    return tramos, puentes, subes


def _cruzar_contra(zonas, meta, tramos, puentes, subes, geod, etiqueta):
    """Cruza una lista de polígonos contra las tres capas de infraestructura.

    `zonas` son polígonos shapely, `meta` la lista paralela de sus atributos.
    Devuelve tres listas de filas listas para CSV.
    """
    from shapely.strtree import STRtree
    arbol = STRtree(zonas)

    # --- puentes: punto dentro de polígono ---
    f_puentes = []
    for pt, a in puentes:
        for k in arbol.query(pt):
            if zonas[k].contains(pt):
                f_puentes.append(dict(
                    codigo=a.get("CODIGO_PUENTE", ""),
                    nombre=a.get("NOMBRE_PUENTE", ""),
                    rol=a.get("ROL", ""), cauce=a.get("CAUCE_QUEB", ""),
                    region=a.get("REGION", ""), provincia=a.get("PROVINCIA", ""),
                    lat=pt.y, lon=pt.x, **meta[k]))
                break

    # --- subestaciones eléctricas: punto dentro de polígono ---
    f_subes = []
    for pt, a in subes:
        for k in arbol.query(pt):
            if zonas[k].contains(pt):
                f_subes.append(dict(
                    subestacion=a.get("subestacion") or a.get("nombre") or "",
                    lat=pt.y, lon=pt.x, **meta[k]))
                break

    # --- tramos viales: largo EXACTO de la parte que cae dentro ---
    f_tramos = []
    for linea, a in tramos:
        largo_km = _largo_m(linea, geod) / 1000.0
        if largo_km <= 0:
            continue
        cand = arbol.query(linea)
        if len(cand) == 0:
            continue
        m_dentro, por_zona = 0.0, {}
        for k in cand:
            if not zonas[k].intersects(linea):
                continue
            mm = _largo_m(zonas[k].intersection(linea), geod)
            if mm > 0:
                m_dentro += mm
                # se queda la zona que aporta más metros: la dominante
                clave = tuple(sorted(meta[k].items()))
                por_zona[clave] = por_zona.get(clave, 0) + mm
        if m_dentro <= 0:
            continue
        # tope: la parte dentro no puede ser mayor que el tramo entero. Pasa
        # cuando dos zonas se solapan y el mismo metro se cuenta dos veces.
        km_exp = min(m_dentro / 1000.0, largo_km)
        dominante = dict(max(por_zona, key=por_zona.get))
        f_tramos.append(dict(
            rol=a.get("ROL_LABEL") or a.get("ROL") or "",
            nombre=a.get("NOMBRE_CAMINO", ""),
            clasificacion=a.get("CLASIFICACION", ""),
            carpeta=a.get("CARPETA", ""), region=a.get("REGION", ""),
            concesionado=a.get("CONCESIONADO", ""),
            km_tramo=round(largo_km, 3), km_expuestos=round(km_exp, 3),
            fraccion=round(km_exp / largo_km, 4), **dominante))

    print(f"\n  [{etiqueta}]")
    print(f"    puentes dentro       : {len(f_puentes):,} de {len(puentes):,}")
    print(f"    subestaciones dentro : {len(f_subes):,} de {len(subes):,}")
    km = sum(t["km_expuestos"] for t in f_tramos)
    print(f"    tramos que tocan     : {len(f_tramos):,} de {len(tramos):,}")
    print(f"    kilómetros dentro    : {km:,.1f} km")
    return f_tramos, f_puentes, f_subes


def modo_cruce():
    """Cruza la amenaza volcánica con la infraestructura ya inventariada."""
    from pyproj import Geod
    from shapely.geometry import shape
    geod = Geod(ellps="WGS84")

    print("=" * 76)
    print("CRUCE · AMENAZA VOLCÁNICA × INFRAESTRUCTURA CRÍTICA")
    print("=" * 76)

    faltan = [p for p in (TRAMOS, PUENTES) if not p.exists()]
    if faltan:
        print("  ✗ falta infraestructura de base: " +
              ", ".join(str(p) for p in faltan))
        return 1

    tramos, puentes, subes = _cargar_infraestructura()
    km_red = sum(_largo_m(l, geod) for l, _ in tramos) / 1000.0
    print(f"\n  inventario: {len(tramos):,} tramos ({km_red:,.0f} km) · "
          f"{len(puentes):,} puentes · {len(subes):,} subestaciones")

    # ------------------------------------------------------------------
    # A · CONTRA EL MAPA DE PELIGRO REAL (polígonos de SERNAGEOMIN)
    #     Es el único cruce que se apoya en una zonificación oficial.
    # ------------------------------------------------------------------
    ruta_peligro = None
    for c in sorted(CRUDO.glob("*/peligro_zonificado.geojson"), reverse=True):
        ruta_peligro = c
        break
    if ruta_peligro is None:
        print("\n  (sin mapa de peligro descargado: correr antes --peligro)")
    else:
        geo = json.loads(ruta_peligro.read_text(encoding="utf-8"))
        zonas, meta = [], []
        for f in geo["features"]:
            s = shape(f["geometry"])
            if not s.is_valid:
                s = s.buffer(0)
            if s.is_empty:
                continue
            p = f["properties"]
            zonas.append(s)
            meta.append(dict(
                grado_peligro=(p.get("GRADO_RIES") or "").strip(),
                tipo_peligro=(p.get("PELIGRO_VO") or "").strip(),
                magnitud_esperada=(p.get("MAGNITUD_E") or "").strip(),
                fuente_zonificacion="SERNAGEOMIN mapa de peligro 1:75.000"))
        t, pu, su = _cruzar_contra(zonas, meta, tramos, puentes, subes, geod,
                                   "MAPA DE PELIGRO OFICIAL · Nevados de Chillán")
        _escribir_csv(DATOS / "volcanes_vial_en_peligro_zonificado.csv", t)
        _escribir_csv(DATOS / "volcanes_puentes_en_peligro_zonificado.csv", pu)
        _escribir_csv(DATOS / "volcanes_subestaciones_en_peligro_zonificado.csv",
                      su)

    # ------------------------------------------------------------------
    # B · ANILLOS DE CERCANÍA A LOS 87 VOLCANES
    #     ★ NO es un mapa de peligro. Es un filtro de tamizado. Ver el
    #       comentario de RADIOS_KM más arriba.
    # ------------------------------------------------------------------
    ruta_rank = None
    for c in sorted(CRUDO.glob("*/ranking_2023.geojson"), reverse=True):
        ruta_rank = c
        break
    if ruta_rank is None:
        print("\n  (sin catálogo descargado: correr antes --catalogo)")
        return 0

    geo = json.loads(ruta_rank.read_text(encoding="utf-8"))
    volcanes = []
    for f in geo["features"]:
        g = f.get("geometry") or {}
        c = g.get("coordinates")
        if not c:
            continue
        p = f["properties"]
        volcanes.append((c[0], c[1], p))
    print(f"\n  volcanes con coordenada: {len(volcanes)} de "
          f"{len(geo['features'])}")

    resumen_radios = []
    for radio in RADIOS_KM:
        zonas, meta = [], []
        for lon, lat, p in volcanes:
            zonas.append(_circulo_geodesico(lon, lat, radio))
            meta.append(dict(
                volcan=(p.get("Sistema_volcánico") or "").strip(),
                posicion_ranking_2023=p.get("Posición__2023"),
                categoria_riesgo=p.get("Categoría"),
                monitoreo_ovdas=p.get("Monitoreo_OVDAS"),
                region_volcan=(p.get("Región") or "").strip(),
                radio_km=radio,
                advertencia="anillo de cercania, NO es mapa de peligro"))
        t, pu, su = _cruzar_contra(zonas, meta, tramos, puentes, subes, geod,
                                   f"ANILLO DE CERCANÍA · {radio} km "
                                   f"(NO es mapa de peligro)")
        km = sum(x["km_expuestos"] for x in t)
        resumen_radios.append(dict(radio_km=radio, tramos=len(t),
                                   km_viales=round(km, 1), puentes=len(pu),
                                   subestaciones=len(su),
                                   pct_km_red=round(100 * km / km_red, 3)))
        if radio == 30:                        # el más amplio, para el detalle
            _escribir_csv(DATOS / "volcanes_vial_cercania_30km.csv", t)
            _escribir_csv(DATOS / "volcanes_puentes_cercania_30km.csv", pu)
            _escribir_csv(DATOS / "volcanes_subestaciones_cercania_30km.csv", su)

    _escribir_csv(DATOS / "volcanes_resumen_cercania.csv", resumen_radios)

    print("\n" + "=" * 76)
    print("RESUMEN POR RADIO (recordar: anillo, no mapa de peligro)")
    print("=" * 76)
    print(f"  {'radio':>7s} {'tramos':>8s} {'km viales':>11s} {'% red':>7s} "
          f"{'puentes':>8s} {'subest.':>8s}")
    for r in resumen_radios:
        print(f"  {r['radio_km']:5d}km {r['tramos']:8,d} {r['km_viales']:11,.1f} "
              f"{r['pct_km_red']:6.2f}% {r['puentes']:8,d} "
              f"{r['subestaciones']:8,d}")

    # --- qué volcanes concentran la exposición, al radio más amplio ---
    ruta = DATOS / "volcanes_vial_cercania_30km.csv"
    if ruta.exists():
        por_volcan = defaultdict(float)
        pos = {}
        with open(ruta, encoding="utf-8") as fh:
            for fila in csv.DictReader(fh):
                por_volcan[fila["volcan"]] += float(fila["km_expuestos"])
                pos[fila["volcan"]] = fila["posicion_ranking_2023"]
        print("\n  los 15 volcanes con más kilómetros de camino a menos de 30 km:")
        for v, km in sorted(por_volcan.items(), key=lambda x: -x[1])[:15]:
            print(f"     {km:8,.1f} km   #{str(pos[v]):>3s} del ranking   {v}")
    return 0


# =============================================================================

def main():
    modos = {"--catalogo": modo_catalogo, "--peligro": modo_peligro,
             "--alerta": modo_alerta, "--cruce": modo_cruce}
    pedidos = [a for a in sys.argv[1:] if a in modos]
    if not pedidos:
        print(__doc__)
        print("modos: " + "  ".join(modos))
        print("\nsin argumentos hace la bajada completa: "
              "--catalogo --peligro --alerta")
        pedidos = ["--catalogo", "--peligro", "--alerta"]
    salida = 0
    for p in pedidos:
        salida |= modos[p]()
        print()
    return salida


if __name__ == "__main__":
    raise SystemExit(main())
