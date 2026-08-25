"""
Adaptador TERRENO — las condiciones del lugar que ENCAMINAN la amenaza.

POR QUÉ EXISTE ESTE MÓDULO
--------------------------
El estudio de vectores (§3) sostiene que la misma meteorología produce amenazas
distintas según el terreno:

    misma lluvia + suelo seco  + pendiente alta + regolito  →  REMOCIÓN EN MASA
    misma lluvia + suelo saturado + cauce estrechado        →  DESBORDE FLUVIAL

Hasta hoy el proyecto tenía el clima (ERA5, 1990-2026, 39+91 puntos) y NO tenía
NADA del terreno. Este módulo llena ese hueco con las condiciones del vocabulario
de §4.1 que se pueden conseguir con dato público verificado.

QUÉ CONDICIONES DEL VOCABULARIO CUBRE, Y CUÁLES NO
--------------------------------------------------
    `Pend`    ✅ pendiente     — Copernicus DEM GLO-30, calculada por Horn
    `Veg`     ✅ cobertura     — ESA WorldCover 10 m, 2021 y 2020
    `Defor`   🟡 aproximada    — diferencia de arbolado 2020→2021 (ver AVISO)
    `Imperm`  ✅ built-up      — clase 50 de WorldCover en 1 km
    `Cuenca`  ❌ SIN DATO      — superficie aguas arriba exige acumulación de
                                flujo sobre la cuenca entera, no cabe en una
                                ventana local. Ver FUENTE_TERRENO.md.
    `Reg`     ❌ SIN DATO      — geología SERNAGEOMIN, ver FUENTE_TERRENO.md
    `Ribera`  ❌ SIN DATO      — exige red de cauces vectorial + edificación

En lugar de `Cuenca` se entregan tres medidas de relieve que SÍ son locales y
sí son verificables: relieve en 1 km, pendiente máxima en 1 km y orientación.
No son `Cuenca`; no se presentan como si lo fueran.

POR QUÉ SIN GDAL, OTRA VEZ
--------------------------
Misma razón que `territorio.py`: instalar la pila geoespacial falló dos veces en
este proyecto. Acá el problema era leer GeoTIFF. Se resolvió a mano:

  · Copernicus GLO-30 es un COG float32, Deflate (zlib de la stdlib) con
    PREDICTOR=3 (predictor de coma flotante: diferencias byte a byte y planos de
    bytes intercalados). Se decodifica en ~15 líneas de numpy — `_decodificar`.
  · ESA WorldCover es un COG uint8, Deflate, sin predictor. Trivial.

VALIDACIÓN DEL LECTOR — no borrar
---------------------------------
Subestación Arica (-18,478 / -70,297): el lector devuelve 37,3 m; la API de
elevación de Open-Meteo (que sirve Copernicus GLO-90) devuelve 36,0 m. Coinciden
dentro de lo esperable entre GLO-30 y GLO-90. Si alguien toca `_decodificar`,
esta comparación es la prueba de que sigue leyendo bien.

Y la PENDIENTE también, no sólo la elevación: pidiendo a Open-Meteo una rejilla
de 3×3 a ±450 m y aplicándole el mismo Horn, `pendiente_grados_450m` da
0,81° vs 0,82° en Arica y 30,97° vs 32,78° en Chungará. Con el predictor de coma
flotante mal decodificado estos números no podrían caer uno al lado del otro.

POR QUÉ SÓLO SE BAJAN BLOQUES, NO TESELAS ENTERAS
-------------------------------------------------
Las 54 teselas de 1° que cubren los 130 puntos pesan ~1 GB. El COG está partido
en bloques de 1024×1024 y el servidor acepta `Range`, así que se baja SÓLO el
bloque que contiene cada punto y su vecindario. Eso es el «recorte de los puntos»
que autoriza el encargo. El crudo que se guarda es el bloque comprimido tal cual
vino, más su cabecera IFD, más un manifiesto con la URL y el rango de bytes: con
eso cualquiera reconstruye la descarga sin volver a pedirla.

LÍMITES DECLARADOS
------------------
1. El DEM es un DSM (superficie: incluye edificios y copas de árboles), no un
   DTM. En bosque denso o ciudad la pendiente puede venir del dosel, no del
   suelo. Confianza 0,85 y no más.
2. La pendiente depende de la escala a la que se mide. Por eso se entregan DOS:
   a 30 m (la ladera inmediata) y a ~450 m (la forma del cerro). No hay una
   «pendiente verdadera»; hay una pendiente a una escala declarada.
3. WorldCover v100 (2020) y v200 (2021) usan algoritmos distintos. La propia ESA
   advierte que NO deben compararse directamente para detectar cambio. El campo
   `delta_arbolado_1km_pp` se entrega como INDICIO, con confianza 0,40, y no
   sirve como medida de deforestación. La deforestación real pide los dos
   catastros de CONAF.
"""

import csv
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from datetime import date, timezone, datetime
from pathlib import Path

import numpy as np

AQUI = Path(__file__).parent.parent
CRUDO = AQUI / "datos" / "crudo" / "terreno"
CSV_SUBESTACIONES = AQUI / "datos" / "subestaciones_puntos.csv"
CSV_RETERM = AQUI / "datos" / "reterm_puntos.csv"
CSV_SALIDA = AQUI / "datos" / "terreno_puntos.csv"

FECHA_CAPTURA = date.today().isoformat()
AGENTE = "infraestructura-critica-clima/terreno.py (proyecto de investigacion)"
# Ritmo prudente: una petición por segundo por servidor. Son buckets públicos de
# AWS Open Data, pero el encargo pide ritmo prudente y se cumple.
PAUSA_S = 1.0


# =============================================================================
# 1 · LAS FUENTES, DECLARADAS ANTES DE TOCARLAS
# =============================================================================

FUENTE_DEM = dict(
    id="copernicus_glo30",
    organismo="ESA / Copernicus (Airbus DS, DLR) — espejo AWS Open Data",
    producto="Copernicus DEM GLO-30 Public, modelo digital de SUPERFICIE, 30 m",
    url="https://copernicus-dem-30m.s3.amazonaws.com/"
        "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
        "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif",
    formato="GeoTIFF COG float32, Deflate + predictor 3, teselas de 1°",
    familia="TERRENO",
    acceso="anonimo",
    acceso_verificado=1,
    robots_txt="no existe en el bucket (404) — no hay exclusion declarada",
    condiciones_uso="GLO-30 Public es libre para el publico general bajo la "
                    "licencia COP-DEM (dataspace.copernicus.eu, coleccion "
                    "COP-DEM). Exige atribucion. Registro AWS Open Data: "
                    "registry.opendata.aws/copernicus-dem",
    atribucion="© DLR e.V. 2010-2014 y © Airbus Defence and Space GmbH "
               "2014-2018, provisto bajo COPERNICUS por la Union Europea y la "
               "ESA; todos los derechos reservados.",
    permite_automatizacion="si",
    granularidad="pixel 1/3600° (~30 m en latitud)",
    epoca_del_dato="2011-2015 (adquisicion TanDEM-X), publicacion 2021",
    confianza_base=0.85,
    notas="Es un DSM: incluye edificios y vegetacion. En bosque denso o ciudad "
          "la pendiente puede venir del dosel y no del suelo. El oceano no "
          "tiene tesela; su ausencia se declara como sin dato, no como cero.",
)

FUENTE_COBERTURA = dict(
    id="esa_worldcover",
    organismo="ESA WorldCover (VITO y consorcio) — espejo AWS Open Data",
    producto="ESA WorldCover 10 m, mapa global de cobertura de suelo",
    url="https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
        "{version}/{anio}/map/ESA_WorldCover_10m_{anio}_{version}_"
        "{ns}{lat:02d}{ew}{lon:03d}_Map.tif",
    formato="GeoTIFF COG uint8, Deflate sin predictor, teselas de 3°",
    familia="TERRENO",
    acceso="anonimo",
    acceso_verificado=1,
    robots_txt="no existe en el bucket (404) — no hay exclusion declarada",
    condiciones_uso="CC-BY 4.0. Registro AWS Open Data: "
                    "registry.opendata.aws/esa-worldcover-vito",
    atribucion="ESA WorldCover project 2021 / 2020, led by VITO, "
               "doi:10.5281/zenodo.5571936 y doi:10.5281/zenodo.5571935",
    permite_automatizacion="si",
    granularidad="pixel 1/12000° (~10 m)",
    epoca_del_dato="2021 (v200) y 2020 (v100)",
    confianza_base=0.85,
    notas="v100 y v200 NO son comparables directamente: la propia ESA lo "
          "advierte. Cualquier diferencia entre anios es indicio, no medida "
          "de cambio.",
)

# Las 11 clases de WorldCover, con su nombre oficial y el grupo funcional que
# le importa a este proyecto. El grupo se declara acá, a la vista, y no se
# esconde dentro de una fórmula.
CLASES_COBERTURA = {
    10:  ("Bosque / cobertura arborea", "arbolado"),
    20:  ("Matorral", "vegetacion_baja"),
    30:  ("Pradera / herbaceo", "vegetacion_baja"),
    40:  ("Cultivo", "cultivo"),
    50:  ("Construido (urbano/industrial)", "construido"),
    60:  ("Suelo desnudo / vegetacion escasa", "desnudo"),
    70:  ("Nieve y hielo", "nieve_hielo"),
    80:  ("Cuerpo de agua permanente", "agua"),
    90:  ("Humedal herbaceo", "humedal"),
    95:  ("Manglar", "humedal"),
    100: ("Musgo y liquen", "vegetacion_baja"),
}
# Lo que retiene el suelo, en el sentido de la receta de remoción en masa:
# árboles, matorral, pradera, musgo, humedal y cultivo. Nieve, roca desnuda,
# agua y ciudad no retienen.
GRUPOS_QUE_RETIENEN = {"arbolado", "vegetacion_baja", "cultivo", "humedal"}


# =============================================================================
# 2 · LECTOR DE GeoTIFF COG A MANO (sin GDAL)
# =============================================================================

# Tamaño en bytes de cada tipo de campo TIFF, por número de tipo.
_TAM_TIPO = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4,
             10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
_FMT_TIPO = {1: "B", 3: "H", 4: "I", 8: "h", 9: "i", 11: "f", 12: "d", 16: "Q"}

# Etiquetas TIFF que necesitamos. El resto se ignora a propósito.
T_ANCHO, T_ALTO = 256, 257
T_BITS, T_COMPRESION = 258, 259
T_PREDICTOR = 317
T_TES_ANCHO, T_TES_ALTO = 322, 323
T_TES_OFFSETS, T_TES_BYTES = 324, 325
T_FORMATO_MUESTRA = 339
T_ESCALA_PIXEL, T_PUNTO_ATADO = 33550, 33922


class TeselaAusente(Exception):
    """La tesela no existe en el servidor (404). No es un error: es un hueco."""


class TeselaCOG:
    """Una tesela GeoTIFF remota, leída por bloques con peticiones Range.

    No baja el archivo entero. Baja la cabecera (donde están los offsets de cada
    bloque) y después, sólo los bloques que se pidan. Cada cosa que baja queda
    guardada como crudo antes de tocarse.
    """

    def __init__(self, url, nombre, carpeta_crudo):
        self.url = url
        self.nombre = nombre
        self.carpeta = Path(carpeta_crudo)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self._bloques = {}      # cache en memoria de bloques ya decodificados
        self._cab = None

    # -- red -----------------------------------------------------------------

    def _pedir(self, ini, fin, intentos=5):
        """Petición Range con reintentos.

        En una corrida de 130 puntos son cientos de peticiones y S3 corta alguna
        conexión («connection reset by peer»). Un corte de red no puede tirar
        abajo la corrida entera: se reintenta con espera creciente. Un 404 SÍ es
        respuesta definitiva y se convierte en «tesela ausente».
        """
        pet = urllib.request.Request(
            self.url, headers={"Range": f"bytes={ini}-{fin}", "User-Agent": AGENTE})
        ultimo = None
        for intento in range(intentos):
            try:
                with urllib.request.urlopen(pet, timeout=120) as r:
                    datos = r.read()
                time.sleep(PAUSA_S)
                return datos
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    raise TeselaAusente(self.nombre) from e
                ultimo = e
            except (urllib.error.URLError, OSError) as e:
                ultimo = e
            espera = PAUSA_S * (2 ** intento)
            print(f"    · reintento {intento+1}/{intentos} en {espera:.0f}s "
                  f"({type(ultimo).__name__}: {ultimo})", flush=True)
            time.sleep(espera)
        raise ultimo

    # -- cabecera ------------------------------------------------------------

    def cabecera(self):
        """Lee (o recupera del crudo) el IFD: geometría y offsets de bloques."""
        if self._cab is not None:
            return self._cab
        ruta = self.carpeta / f"{self.nombre}.cabecera.bin"
        if ruta.exists():
            bruto = ruta.read_bytes()
        else:
            bruto = self._pedir(0, 65535)
            ruta.write_bytes(bruto)          # ← crudo antes de procesar
        self._cab = self._parsear_ifd(bruto)
        return self._cab

    @staticmethod
    def _parsear_ifd(bruto):
        orden = "<" if bruto[:2] == b"II" else ">"
        pos = struct.unpack(orden + "I", bruto[4:8])[0]
        n = struct.unpack(orden + "H", bruto[pos:pos + 2])[0]
        etiquetas = {}
        for i in range(n):
            p = pos + 2 + i * 12
            etiq, tipo, cuenta = struct.unpack(orden + "HHI", bruto[p:p + 8])
            tam = _TAM_TIPO.get(tipo, 0) * cuenta
            if tam <= 4:
                bytes_valor = bruto[p + 8:p + 8 + tam]
            else:
                off = struct.unpack(orden + "I", bruto[p + 8:p + 12])[0]
                bytes_valor = bruto[off:off + tam]
            fmt = _FMT_TIPO.get(tipo)
            if fmt and len(bytes_valor) == tam:
                etiquetas[etiq] = struct.unpack(orden + fmt * cuenta, bytes_valor)
        return etiquetas

    # -- geometría -----------------------------------------------------------

    @property
    def geo(self):
        """(x0, y0, paso_lon, paso_lat, ancho, alto) — esquina NO y paso."""
        c = self.cabecera()
        esc = c[T_ESCALA_PIXEL]
        ata = c[T_PUNTO_ATADO]
        return (ata[3], ata[4], esc[0], esc[1], c[T_ANCHO][0], c[T_ALTO][0])

    # -- bloques -------------------------------------------------------------

    def _decodificar(self, bruto, c):
        """Deflate + predictor → matriz numpy del bloque.

        El predictor 3 (coma flotante) de TIFF hace dos cosas y hay que
        deshacerlas EN ESTE ORDEN, fila por fila del bloque:
          1. diferencias byte a byte: cada byte guarda su resta con el anterior
             → se deshace con una suma acumulada módulo 256;
          2. planos de bytes: la fila guarda primero TODOS los bytes más
             significativos, después todos los segundos, etc.
             → se deshace reordenando (4, ancho) → (ancho, 4) y leyendo el
             resultado como float32 big-endian.
        """
        crudo = zlib.decompress(bruto)
        tw, th = c[T_TES_ANCHO][0], c[T_TES_ALTO][0]
        bytes_muestra = c[T_BITS][0] // 8
        predictor = c.get(T_PREDICTOR, (1,))[0]

        if predictor == 3:
            a = np.frombuffer(crudo, dtype=np.uint8).reshape(th, tw * bytes_muestra).copy()
            a = np.cumsum(a, axis=1, dtype=np.uint8)
            a = a.reshape(th, bytes_muestra, tw).transpose(0, 2, 1)
            return np.ascontiguousarray(a).view(">f4").reshape(th, tw).astype(np.float32)
        if predictor != 1:
            raise NotImplementedError(f"predictor TIFF {predictor} no implementado")
        if bytes_muestra == 1:
            return np.frombuffer(crudo, dtype=np.uint8).reshape(th, tw)
        raise NotImplementedError(f"{bytes_muestra*8} bits sin predictor no implementado")

    def bloque(self, indice):
        if indice in self._bloques:
            return self._bloques[indice]
        c = self.cabecera()
        ruta = self.carpeta / f"{self.nombre}.bloque{indice:04d}.deflate"
        if ruta.exists():
            bruto = ruta.read_bytes()
        else:
            ini = c[T_TES_OFFSETS][indice]
            largo = c[T_TES_BYTES][indice]
            bruto = self._pedir(ini, ini + largo - 1)
            ruta.write_bytes(bruto)          # ← crudo antes de procesar
            self._anotar(ruta.name, ini, largo)
        mat = self._decodificar(bruto, c)
        # Tope del cache en memoria: un bloque del DEM son 4 MB y los 130 puntos
        # tocan del orden de 100 bloques. Sin tope, el proceso se come la RAM.
        # El crudo en disco sigue estando, así que reabrir un bloque es gratis.
        if len(self._bloques) >= 8:
            self._bloques.pop(next(iter(self._bloques)))
        self._bloques[indice] = mat
        return mat

    def _anotar(self, archivo, ini, largo):
        """Manifiesto: con esto se reconstruye la descarga sin volver a pedirla."""
        manif = self.carpeta / "MANIFIESTO.jsonl"
        with manif.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "archivo": archivo, "url": self.url,
                "rango_bytes": [ini, ini + largo - 1],
                "bajado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, ensure_ascii=False) + "\n")

    # -- lectura por coordenada ---------------------------------------------

    def valores(self, filas, columnas):
        """Devuelve los valores en (fila, columna) enteras. Fuera de rango → nan."""
        x0, y0, plon, plat, ancho, alto = self.geo
        c = self.cabecera()
        tw, th = c[T_TES_ANCHO][0], c[T_TES_ALTO][0]
        n_tx = (ancho + tw - 1) // tw

        salida = np.full(filas.shape, np.nan, dtype=np.float64)
        dentro = (filas >= 0) & (filas < alto) & (columnas >= 0) & (columnas < ancho)
        if not dentro.any():
            return salida
        f, co = filas[dentro], columnas[dentro]
        indices = (f // th) * n_tx + (co // tw)
        parcial = np.full(f.shape, np.nan, dtype=np.float64)
        for bi in np.unique(indices):
            m = indices == bi
            b = self.bloque(int(bi))
            parcial[m] = b[f[m] % th, co[m] % tw]
        salida[dentro] = parcial
        return salida


class Mosaico:
    """Un conjunto de teselas que se comporta como una capa continua.

    Recibe coordenadas geográficas y decide sola qué tesela y qué bloque tocar.
    Si la tesela no existe (océano, o país no liberado en GLO-30 Public), el
    valor sale como «sin dato» — nunca como cero.
    """

    def __init__(self, plantilla_url, plantilla_nombre, grado_tesela, carpeta_crudo):
        self.plantilla_url = plantilla_url
        self.plantilla_nombre = plantilla_nombre
        self.grado = grado_tesela
        self.carpeta = Path(carpeta_crudo)
        self._teselas = {}
        self.ausentes = set()

    def _clave(self, lat, lon):
        la = int(math.floor(lat / self.grado) * self.grado)
        lo = int(math.floor(lon / self.grado) * self.grado)
        return la, lo

    def tesela(self, lat, lon):
        clave = self._clave(lat, lon)
        if clave in self._teselas:
            return self._teselas[clave]
        if clave in self.ausentes:
            return None
        la, lo = clave
        campos = dict(ns="S" if la < 0 else "N", lat=abs(la),
                      ew="W" if lo < 0 else "E", lon=abs(lo))
        url = self.plantilla_url.format(**campos)
        nombre = self.plantilla_nombre.format(**campos)
        t = TeselaCOG(url, nombre, self.carpeta)
        try:
            t.cabecera()
        except TeselaAusente:
            self.ausentes.add(clave)
            return None
        self._teselas[clave] = t
        return t

    def paso_en(self, lat, lon):
        """Paso del pixel (grados lon, grados lat) en la tesela de ese punto."""
        t = self.tesela(lat, lon)
        if t is None:
            return None
        _, _, plon, plat, _, _ = t.geo
        return plon, plat

    def ventana(self, lat, lon, n_lat, n_lon, paso_lat, paso_lon):
        """Rejilla geográfica centrada en el punto, fila 0 = la más al NORTE.

        Se construye en coordenadas y no en pixeles a propósito: así una ventana
        que cruza el borde de una tesela se arma sola de las dos teselas, sin
        casos especiales.
        """
        dlat = (np.arange(2 * n_lat + 1) - n_lat) * paso_lat
        dlon = (np.arange(2 * n_lon + 1) - n_lon) * paso_lon
        lats = lat - dlat[:, None]        # fila 0 = norte → se resta al revés
        lons = lon + dlon[None, :]
        lats = np.broadcast_to(lats, (2 * n_lat + 1, 2 * n_lon + 1))
        lons = np.broadcast_to(lons, (2 * n_lat + 1, 2 * n_lon + 1))

        salida = np.full(lats.shape, np.nan)
        claves = np.stack([np.floor(lats / self.grado).astype(int) * self.grado,
                           np.floor(lons / self.grado).astype(int) * self.grado])
        planas = claves.reshape(2, -1).T
        for clave in {tuple(p) for p in planas}:
            m = (planas[:, 0] == clave[0]) & (planas[:, 1] == clave[1])
            m = m.reshape(lats.shape)
            t = self.tesela(float(lats[m][0]), float(lons[m][0]))
            if t is None:
                continue
            x0, y0, plon, plat, _, _ = t.geo
            col = np.floor((lons[m] - x0) / plon).astype(int)
            fil = np.floor((y0 - lats[m]) / plat).astype(int)
            salida[m] = t.valores(fil, col)
        return salida


# =============================================================================
# 3 · MÉTRICAS DE TERRENO
# =============================================================================

RADIO_TIERRA_M = 6371008.8


def metros_por_grado(lat):
    """Cuánto mide un grado, en metros, a esa latitud. Sin proyectar nada."""
    m_lat = math.pi * RADIO_TIERRA_M / 180.0
    m_lon = m_lat * math.cos(math.radians(lat))
    return m_lon, m_lat


def pendiente_horn(z, dx_m, dy_m):
    """Pendiente y orientación por el método de Horn sobre una matriz 3×3.

    Horn (1981) es el que usan ArcGIS y GRASS: pesa los vecinos en cruz el
    doble que los diagonales, lo que lo hace mucho menos sensible al ruido del
    DEM que una diferencia simple. `z` viene con la fila 0 al NORTE.

    Devuelve (grados de pendiente, azimut de máxima bajada en grados desde el N).
    """
    if np.isnan(z).any():
        return None, None
    dz_este = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) -
               (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (8.0 * dx_m)
    dz_norte = ((z[0, 0] + 2 * z[0, 1] + z[0, 2]) -
                (z[2, 0] + 2 * z[2, 1] + z[2, 2])) / (8.0 * dy_m)
    grados = math.degrees(math.atan(math.hypot(dz_este, dz_norte)))
    # dirección hacia donde BAJA el terreno
    azimut = (math.degrees(math.atan2(-dz_este, -dz_norte)) + 360.0) % 360.0
    return grados, azimut


def mapa_pendiente(z, dx_m, dy_m):
    """Pendiente en grados en cada pixel interior de la ventana (Horn vectorial)."""
    dz_este = ((z[:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:]) -
               (z[:-2, :-2] + 2 * z[1:-1, :-2] + z[2:, :-2])) / (8.0 * dx_m)
    dz_norte = ((z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:]) -
                (z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:])) / (8.0 * dy_m)
    return np.degrees(np.arctan(np.hypot(dz_este, dz_norte)))


# --- parámetros de las ventanas, declarados y no escondidos -------------------
#
# El DEM se lee a su paso nativo (1/3600° ≈ 30 m en latitud). 34 pixeles a cada
# lado son ~1 km: alcanza para el relieve local y deja margen para el estencil
# de 450 m sin pedir más bloques.
DEM_RADIO_PX = 34
DEM_ESTENCIL_LARGO_PX = 15      # ~450 m: la forma del cerro, no la del talud
# WorldCover se lee a 10 m. 300 pixeles son 3 km: es el radio dentro del cual se
# busca agua permanente. Más allá, la respuesta honesta es «no la vi», no una
# distancia inventada.
WC_RADIO_PX = 300
WC_RADIO_FRACCION_PX = 100      # 1 km, para las fracciones de cobertura


def condiciones_dem(mosaico, lat, lon):
    """Elevación, pendiente a dos escalas, relieve y orientación."""
    r = {k: "" for k in ("elevacion_m", "pendiente_grados_30m",
                         "pendiente_grados_450m", "pendiente_max_1km_grados",
                         "relieve_1km_m", "orientacion_grados",
                         "dem_paso_lat_arcsec", "dem_paso_lon_arcsec",
                         "dem_cobertura_ventana_pct")}
    paso = mosaico.paso_en(lat, lon)
    if paso is None:
        return r, "sin tesela GLO-30 en la coordenada"
    plon, plat = paso
    r["dem_paso_lat_arcsec"] = round(plat * 3600, 3)
    r["dem_paso_lon_arcsec"] = round(plon * 3600, 3)

    z = mosaico.ventana(lat, lon, DEM_RADIO_PX, DEM_RADIO_PX, plat, plon)
    validos = np.isfinite(z)
    r["dem_cobertura_ventana_pct"] = round(100.0 * validos.mean(), 1)
    c = DEM_RADIO_PX
    if not validos[c, c]:
        return r, "pixel central sin dato en GLO-30"

    m_lon, m_lat = metros_por_grado(lat)
    dx_m, dy_m = plon * m_lon, plat * m_lat
    r["elevacion_m"] = round(float(z[c, c]), 1)

    p30, _ = pendiente_horn(z[c - 1:c + 2, c - 1:c + 2], dx_m, dy_m)
    if p30 is not None:
        r["pendiente_grados_30m"] = round(p30, 2)

    k = DEM_ESTENCIL_LARGO_PX
    z3 = z[c - k:c + k + 1:k, c - k:c + k + 1:k]
    if z3.shape == (3, 3):
        p450, azi = pendiente_horn(z3, dx_m * k, dy_m * k)
        if p450 is not None:
            r["pendiente_grados_450m"] = round(p450, 2)
            r["orientacion_grados"] = round(azi, 1)

    if validos.all():
        r["relieve_1km_m"] = round(float(np.nanmax(z) - np.nanmin(z)), 1)
        r["pendiente_max_1km_grados"] = round(float(np.nanmax(mapa_pendiente(z, dx_m, dy_m))), 2)
    return r, ""


def condiciones_cobertura(mosaico, lat, lon, sufijo):
    """Clase en el punto, fracciones en 1 km y distancia al agua permanente."""
    campos = ["clase", "clase_nombre", "frac_arbolado_1km", "frac_retiene_1km",
              "frac_desnudo_1km", "frac_construido_1km", "frac_agua_1km",
              "frac_nieve_1km", "dist_agua_permanente_m", "cobertura_ventana_pct"]
    r = {f"{k}_{sufijo}": "" for k in campos}
    paso = mosaico.paso_en(lat, lon)
    if paso is None:
        return r, f"sin tesela WorldCover {sufijo}"
    plon, plat = paso
    v = mosaico.ventana(lat, lon, WC_RADIO_PX, WC_RADIO_PX, plat, plon)
    validos = np.isfinite(v) & (v > 0)
    r[f"cobertura_ventana_pct_{sufijo}"] = round(100.0 * validos.mean(), 1)
    c = WC_RADIO_PX
    if not validos[c, c]:
        return r, f"pixel central sin dato en WorldCover {sufijo}"

    cod = int(v[c, c])
    r[f"clase_{sufijo}"] = cod
    r[f"clase_nombre_{sufijo}"] = CLASES_COBERTURA.get(cod, ("desconocida", ""))[0]

    k = WC_RADIO_FRACCION_PX
    sub = v[c - k:c + k + 1, c - k:c + k + 1]
    val = np.isfinite(sub) & (sub > 0)
    n = int(val.sum())
    if n:
        def frac(codigos):
            return round(float(np.isin(sub[val], codigos).mean()), 4)
        retienen = [c_ for c_, (_, g) in CLASES_COBERTURA.items() if g in GRUPOS_QUE_RETIENEN]
        r[f"frac_arbolado_1km_{sufijo}"] = frac([10])
        r[f"frac_retiene_1km_{sufijo}"] = frac(retienen)
        r[f"frac_desnudo_1km_{sufijo}"] = frac([60])
        r[f"frac_construido_1km_{sufijo}"] = frac([50])
        r[f"frac_agua_1km_{sufijo}"] = frac([80])
        r[f"frac_nieve_1km_{sufijo}"] = frac([70])

    # distancia al agua permanente más cercana dentro del radio buscado
    agua = np.argwhere(v == 80)
    m_lon, m_lat = metros_por_grado(lat)
    if len(agua):
        dfil = (agua[:, 0] - c) * plat * m_lat
        dcol = (agua[:, 1] - c) * plon * m_lon
        r[f"dist_agua_permanente_m_{sufijo}"] = round(float(np.min(np.hypot(dfil, dcol))), 1)
    else:
        radio_m = WC_RADIO_PX * plat * m_lat
        r[f"dist_agua_permanente_m_{sufijo}"] = f">{radio_m:.0f}"
    return r, ""


# =============================================================================
# 4 · LOS PUNTOS
# =============================================================================

def leer_puntos():
    """Los 39 de subestaciones y los 91 de ReTeRM, con su origen declarado."""
    puntos = []
    with CSV_SUBESTACIONES.open(encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            puntos.append(dict(punto=f["subestacion"], conjunto="subestaciones",
                               region=f.get("region", ""), comuna="",
                               lat=float(f["lat"]), lon=float(f["lon"])))
    with CSV_RETERM.open(encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            puntos.append(dict(punto=f["subestacion"], conjunto="reterm",
                               region=f.get("region", ""), comuna=f.get("comuna", ""),
                               lat=float(f["lat"]), lon=float(f["lon"])))
    return puntos


COLUMNAS = [
    "punto", "conjunto", "region", "comuna", "lat", "lon",
    # --- relieve (Copernicus GLO-30) ---
    "elevacion_m", "pendiente_grados_30m", "pendiente_grados_450m",
    "pendiente_max_1km_grados", "relieve_1km_m", "orientacion_grados",
    "dem_paso_lat_arcsec", "dem_paso_lon_arcsec", "dem_cobertura_ventana_pct",
    "fuente_relieve", "epoca_relieve", "confianza_relieve", "nota_relieve",
    # --- cobertura 2021 (ESA WorldCover v200) ---
    "clase_2021", "clase_nombre_2021", "frac_arbolado_1km_2021",
    "frac_retiene_1km_2021", "frac_desnudo_1km_2021", "frac_construido_1km_2021",
    "frac_agua_1km_2021", "frac_nieve_1km_2021", "dist_agua_permanente_m_2021",
    "cobertura_ventana_pct_2021",
    # --- cobertura 2020 (ESA WorldCover v100) ---
    "clase_2020", "clase_nombre_2020", "frac_arbolado_1km_2020",
    "frac_retiene_1km_2020", "frac_desnudo_1km_2020", "frac_construido_1km_2020",
    "frac_agua_1km_2020", "frac_nieve_1km_2020", "dist_agua_permanente_m_2020",
    "cobertura_ventana_pct_2020",
    "fuente_cobertura", "epoca_cobertura", "confianza_cobertura", "nota_cobertura",
    # --- indicio de cambio, con su advertencia pegada ---
    "delta_arbolado_1km_pp", "confianza_delta_arbolado", "nota_delta_arbolado",
    # --- lo que NO se consiguió, dicho en la propia fila ---
    "cuenca_km2", "nota_cuenca",
    "regolito", "nota_regolito",
    "dist_cauce_m", "nota_dist_cauce",
    "fecha_captura",
]

NOTA_CUENCA = ("SIN DATO: la superficie aguas arriba exige acumulacion de flujo "
               "sobre la cuenca completa; no se deriva de una ventana local. "
               "Ver FUENTE_TERRENO.md.")
NOTA_REGOLITO = ("SIN DATO: geologia SERNAGEOMIN no incorporada. "
                 "Ver FUENTE_TERRENO.md.")
NOTA_DELTA = ("INDICIO, NO MEDIDA. WorldCover v100 (2020) y v200 (2021) usan "
              "algoritmos distintos y la ESA advierte que no deben compararse "
              "para detectar cambio. La deforestacion real pide los dos "
              "catastros de CONAF.")


def construir(limite=None, verboso=True):
    carpeta_dem = CRUDO / "copernicus_glo30" / FECHA_CAPTURA
    carpeta_wc = CRUDO / "esa_worldcover" / FECHA_CAPTURA

    dem = Mosaico(FUENTE_DEM["url"],
                  "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM",
                  1, carpeta_dem)

    def mosaico_cobertura(version, anio):
        nombre = ("ESA_WorldCover_10m_{anio}_{version}_"
                  "{{ns}}{{lat:02d}}{{ew}}{{lon:03d}}_Map").format(anio=anio, version=version)
        url = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
               "{version}/{anio}/map/{nombre}.tif").format(
                   version=version, anio=anio, nombre=nombre)
        return Mosaico(url, nombre, 3, carpeta_wc)

    wc21 = mosaico_cobertura("v200", 2021)
    wc20 = mosaico_cobertura("v100", 2020)

    puntos = leer_puntos()
    if limite:
        puntos = puntos[:limite]
    filas = []
    for i, p in enumerate(puntos, 1):
        fila = {c: "" for c in COLUMNAS}
        fila.update(punto=p["punto"], conjunto=p["conjunto"], region=p["region"],
                    comuna=p["comuna"], lat=p["lat"], lon=p["lon"],
                    fecha_captura=FECHA_CAPTURA)

        r, nota = condiciones_dem(dem, p["lat"], p["lon"])
        fila.update(r)
        fila["fuente_relieve"] = FUENTE_DEM["id"]
        fila["epoca_relieve"] = FUENTE_DEM["epoca_del_dato"]
        fila["confianza_relieve"] = FUENTE_DEM["confianza_base"] if not nota else ""
        fila["nota_relieve"] = nota or "DSM (incluye dosel y edificios), no DTM"

        r21, n21 = condiciones_cobertura(wc21, p["lat"], p["lon"], "2021")
        r20, n20 = condiciones_cobertura(wc20, p["lat"], p["lon"], "2020")
        fila.update(r21)
        fila.update(r20)
        fila["fuente_cobertura"] = FUENTE_COBERTURA["id"]
        fila["epoca_cobertura"] = "2021 (v200) y 2020 (v100)"
        fila["confianza_cobertura"] = FUENTE_COBERTURA["confianza_base"] if not n21 else ""
        fila["nota_cobertura"] = "; ".join(x for x in (n21, n20) if x)

        a21, a20 = r21["frac_arbolado_1km_2021"], r20["frac_arbolado_1km_2020"]
        if a21 != "" and a20 != "":
            fila["delta_arbolado_1km_pp"] = round((a21 - a20) * 100, 2)
            fila["confianza_delta_arbolado"] = 0.40
        fila["nota_delta_arbolado"] = NOTA_DELTA

        fila["nota_cuenca"] = NOTA_CUENCA
        fila["nota_regolito"] = NOTA_REGOLITO
        fila["nota_dist_cauce"] = NOTA_DIST_CAUCE
        filas.append(fila)
        if verboso:
            print(f"[{i:3d}/{len(puntos)}] {p['punto'][:44]:44s} "
                  f"z={fila['elevacion_m']!s:>7s} m  "
                  f"pend30={fila['pendiente_grados_30m']!s:>6s}°  "
                  f"cob={fila['clase_nombre_2021'][:22]}", flush=True)

    CSV_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with CSV_SALIDA.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas)
    if verboso:
        print(f"\nEscrito {CSV_SALIDA} con {len(filas)} filas.")
    return filas


NOTA_DIST_CAUCE = ("SIN DATO de red de cauces vectorial. Lo mas cercano que hay "
                   "es `dist_agua_permanente_m_2021`, que sale de WorldCover y "
                   "SOLO ve agua permanente de mas de ~10 m de ancho: no ve "
                   "quebradas ni esteros. Ver FUENTE_TERRENO.md.")


# =============================================================================
# 5 · LA RED DE CAUCES — y, con ella, `Cuenca`
# =============================================================================
#
# POR QUÉ HydroRIVERS Y NO LA CAPA CHILENA COMO BASE
# --------------------------------------------------
# La capa oficial chilena (WFS de geoportal.cl, workspace `Hidrografia`, escala
# 1:25.000, con orden de Strahler y nombre de cuenca) es MEJOR donde existe —
# pero verificado el 17-ago-2026 su cobertura publicada llega hasta ~lat −35:
# en Chillán, Valdivia y Puerto Montt devuelve CERO features. Los puntos del
# proyecto van de −18 a −53. Una capa que se apaga a la mitad del territorio no
# puede ser la base; se usa como REFINAMIENTO donde alcanza.
#
# HydroRIVERS v1.0 Sudamérica cubre el rango completo y, sobre todo, trae
# `UPLAND_SKM`: la superficie de cuenca aguas arriba de cada tramo, ya calculada
# por acumulación de flujo sobre el DEM de HydroSHEDS. Eso es exactamente
# `Cuenca` del vocabulario §4.1 — la condición que este módulo no podía derivar
# de una ventana local.
#
# LÍMITE DECLARADO: HydroRIVERS sólo incluye tramos con cuenca ≥ 10 km² o caudal
# ≥ 0,1 m³/s. Las quebradas secas del norte —justo las que canalizan los
# aluviones— NO están. Para el norte árido, la capa chilena (que sí trae
# «Quebrada» como tipo) es la que hay que mirar.

FUENTE_CAUCES = dict(
    id="hydrorivers_v10_sa",
    organismo="HydroSHEDS / World Wildlife Fund (WWF)",
    producto="HydroRIVERS v1.0, red de drenaje de Sudamerica",
    url="https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_sa_shp.zip",
    formato="Shapefile (polilineas) dentro de ZIP, 90,8 MB, 1.620.963 tramos",
    familia="TERRENO",
    acceso="anonimo",
    acceso_verificado=1,
    robots_txt="data.hydrosheds.org: sin directivas Disallow",
    condiciones_uso="Libre para uso cientifico, educativo y comercial; exige "
                    "atribucion. Ver hydrosheds.org/page/license",
    atribucion="Lehner, B., Grill G. (2013): Global river hydrography and "
               "network routing. HydroSHEDS / WWF, hydrosheds.org",
    permite_automatizacion="si",
    granularidad="tramo de rio (~1 km medio)",
    epoca_del_dato="v1.0, publicado 2019 (base SRTM 2000)",
    confianza_base=0.80,
    notas="Umbral: solo tramos con cuenca >= 10 km2 o caudal >= 0,1 m3/s. NO "
          "incluye quebradas secas ni esteros menores — precisamente los que "
          "canalizan el aluvion en el norte arido.",
)

FUENTE_CAUCES_IDE = dict(
    id="ide_chile_hidrografia",
    organismo="IDE Chile / Ministerio de Bienes Nacionales (GeoServer geoportal.cl)",
    producto="Hidrografia de Chile 1:25.000, capa Hidrografia:hidrografa",
    url="https://geoportal.cl/geoserver/Hidrografia/wfs",
    formato="WFS 2.0.0, outputFormat=application/json (GeoJSON)",
    familia="TERRENO",
    acceso="anonimo",
    acceso_verificado=1,
    robots_txt="geoportal.cl: 'User-agent: * / Disallow:' — todo permitido",
    condiciones_uso="GetCapabilities declara Fees: NONE y AccessConstraints: "
                    "NONE. El catalogo de datos.gob.cl la lista como CC-BY.",
    atribucion="IDE Chile — Ministerio de Bienes Nacionales",
    permite_automatizacion="si",
    granularidad="1:25.000",
    epoca_del_dato="version publicada en WFS al 17-ago-2026",
    confianza_base=0.85,
    notas="COBERTURA PARCIAL: el bounding box declarado llega a lat -34,99 y "
          "se comprobo 0 features en Chillan (-36,6), Valdivia (-39,8) y "
          "Puerto Montt (-41,5). Solo sirve del Maule al norte.",
)

RUTA_ZIP_CAUCES = (CRUDO / "hydrorivers" / "2026-08-17" /
                   "HydroRIVERS_v10_sa_shp.zip")
INTERNO_SHP = "HydroRIVERS_v10_sa_shp/HydroRIVERS_v10_sa.shp"
INTERNO_DBF = "HydroRIVERS_v10_sa_shp/HydroRIVERS_v10_sa.dbf"
# 0,3° son ~33 km en latitud. Si no hay cauce dentro de ese radio, la respuesta
# honesta es «>33 km», no un número sacado de más lejos.
RADIO_CAUCE_GRADOS = 0.3
WFS_URL_CAUCES_IDE = FUENTE_CAUCES_IDE["url"]


def distancia_a_polilinea(lat, lon, xs, ys, partes):
    """Distancia en metros del punto al tramo, sin proyectar nada.

    Se pasa a un plano local en metros centrado en el propio punto (equivale a
    una equirectangular local, exacta a estas distancias) y se resuelve la
    distancia punto-segmento clásica. `partes` marca dónde empieza cada trozo de
    una polilínea múltiple, para no unir trozos que no están unidos.
    """
    m_lon, m_lat = metros_por_grado(lat)
    x = (xs - lon) * m_lon
    y = (ys - lat) * m_lat
    mejor = math.inf
    cortes = list(partes) + [len(xs)]
    for ini, fin in zip(cortes[:-1], cortes[1:]):
        if fin - ini < 1:
            continue
        if fin - ini == 1:
            mejor = min(mejor, float(math.hypot(x[ini], y[ini])))
            continue
        ax, ay = x[ini:fin - 1], y[ini:fin - 1]
        bx, by = x[ini + 1:fin], y[ini + 1:fin]
        dx, dy = bx - ax, by - ay
        largo2 = dx * dx + dy * dy
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(largo2 > 0, -(ax * dx + ay * dy) / np.where(largo2 > 0, largo2, 1), 0.0)
        t = np.clip(t, 0.0, 1.0)
        d = np.hypot(ax + t * dx, ay + t * dy)
        mejor = min(mejor, float(d.min()))
    return mejor


def _medir_cauces_una_pasada(zf, puntos, cajas):
    """Una sola pasada sobre los 1,6 millones de tramos; nada se guarda entero.

    212 MB de geometría no caben cómodos en memoria, y comparar cada tramo con
    cada punto sería 130 × 1,6 M de pruebas. Se evitan las dos cosas con una
    rejilla de 1°: cada punto se apunta en las celdas que toca su caja de
    búsqueda, y de cada tramo se mira SÓLO la celda de su bounding box (que el
    shapefile ya trae escrito). El 99 % se descarta sin desempaquetar un vértice.
    """
    rejilla = {}
    for i, (x0, y0, x1, y1) in enumerate(cajas):
        for cx in range(math.floor(x0), math.floor(x1) + 1):
            for cy in range(math.floor(y0), math.floor(y1) + 1):
                rejilla.setdefault((cx, cy), []).append(i)
    mejor_d = [math.inf] * len(puntos)
    mejor_i = [None] * len(puntos)

    with zf.open(INTERNO_SHP) as fh:
        fh.read(100)                       # cabecera del shapefile
        indice = 0
        while True:
            cab = fh.read(8)
            if len(cab) < 8:
                break
            _, palabras = struct.unpack(">ii", cab)
            cuerpo = fh.read(palabras * 2)
            indice += 1
            if len(cuerpo) < 44 or struct.unpack("<i", cuerpo[:4])[0] != 3:
                continue
            bx0, by0, bx1, by1 = struct.unpack("<4d", cuerpo[4:36])
            candidatos = set()
            for cx in range(math.floor(bx0), math.floor(bx1) + 1):
                for cy in range(math.floor(by0), math.floor(by1) + 1):
                    candidatos.update(rejilla.get((cx, cy), ()))
            if not candidatos:
                continue
            candidatos = [j for j in candidatos
                          if not (bx1 < cajas[j][0] or bx0 > cajas[j][2]
                                  or by1 < cajas[j][1] or by0 > cajas[j][3])]
            if not candidatos:
                continue
            n_partes, n_puntos = struct.unpack("<2i", cuerpo[36:44])
            p0 = 44 + n_partes * 4
            partes = struct.unpack(f"<{n_partes}i", cuerpo[44:p0])
            coord = np.frombuffer(cuerpo[p0:p0 + n_puntos * 16], dtype="<f8")
            xs, ys = coord[0::2], coord[1::2]
            for j in candidatos:
                d = distancia_a_polilinea(puntos[j]["lat"], puntos[j]["lon"],
                                          xs, ys, partes)
                if d < mejor_d[j]:
                    mejor_d[j], mejor_i[j] = d, indice
    return mejor_d, mejor_i


def _leer_dbf_cauces(zf, indices):
    """Lee del .dbf sólo los registros pedidos, en una pasada secuencial."""
    if not indices:
        return {}
    pendientes = sorted(indices)
    with zf.open(INTERNO_DBF) as fh:
        cab = fh.read(32)
        _, _, _, _, _, largo_cab, largo_reg = struct.unpack("<BBBBIHH", cab[:12])
        resto = fh.read(largo_cab - 32)
        campos, i = [], 0
        while i < len(resto) and resto[i] != 0x0D:
            c = resto[i:i + 32]
            campos.append((c[:11].split(b"\x00")[0].decode(), c[16]))
            i += 32
        salida, leidos = {}, 0
        for idx in pendientes:
            saltar = (idx - 1 - leidos) * largo_reg
            while saltar > 0:                      # el zip no permite seek barato
                saltar -= len(fh.read(min(saltar, 1 << 20)))
            reg = fh.read(largo_reg)
            leidos = idx
            pos, vals = 1, {}
            for nombre, largo in campos:
                vals[nombre] = reg[pos:pos + largo].decode("latin-1").strip()
                pos += largo
            salida[idx] = vals
    return salida


def condiciones_cauces(puntos, verboso=True):
    """Para cada punto: distancia al cauce y atributos del tramo más cercano."""
    import zipfile
    if not RUTA_ZIP_CAUCES.exists():
        raise FileNotFoundError(f"falta el crudo de HydroRIVERS: {RUTA_ZIP_CAUCES}")
    cajas = [(p["lon"] - RADIO_CAUCE_GRADOS, p["lat"] - RADIO_CAUCE_GRADOS,
              p["lon"] + RADIO_CAUCE_GRADOS, p["lat"] + RADIO_CAUCE_GRADOS)
             for p in puntos]
    with zipfile.ZipFile(RUTA_ZIP_CAUCES) as zf:
        if verboso:
            print("Recorriendo 1.620.963 tramos de HydroRIVERS…", flush=True)
        dists, indices = _medir_cauces_una_pasada(zf, puntos, cajas)
        medidas = {p["punto"]: (d, i) for p, d, i in zip(puntos, dists, indices)}
        atributos = _leer_dbf_cauces(zf, {i for i in indices if i is not None})

    salida = {}
    radio_m = RADIO_CAUCE_GRADOS * math.pi * RADIO_TIERRA_M / 180.0
    for p in puntos:
        d, idx = medidas[p["punto"]]
        r = dict(dist_cauce_m="", cauce_orden_strahler="", cuenca_km2="",
                 caudal_medio_m3s="", cauce_id="")
        if idx is None:
            r["dist_cauce_m"] = f">{radio_m:.0f}"
            r["nota_cauce"] = (f"ningun tramo de HydroRIVERS dentro de "
                               f"{radio_m/1000:.0f} km; recordar que la capa "
                               f"omite cuencas menores a 10 km2")
        else:
            a = atributos.get(idx, {})
            r["dist_cauce_m"] = round(d, 1)
            r["cauce_id"] = a.get("HYRIV_ID", "")
            r["cauce_orden_strahler"] = a.get("ORD_STRA", "")
            r["cuenca_km2"] = a.get("UPLAND_SKM", "")
            r["caudal_medio_m3s"] = a.get("DIS_AV_CMS", "")
            r["nota_cauce"] = ""
        salida[p["punto"]] = r
    return salida


def cauces_ide(puntos, verboso=True):
    """Refinamiento con la capa chilena 1:25.000, donde exista cobertura."""
    salida = {}
    for i, p in enumerate(puntos, 1):
        r = dict(dist_cauce_ide_m="", cauce_ide_tipo="", cauce_ide_nombre="",
                 cauce_ide_cuenca="", cauce_ide_strahler="", nota_cauce_ide="")
        consulta = (
            f"{WFS_URL_CAUCES_IDE}?service=WFS&version=2.0.0&request=GetFeature"
            f"&typeNames=Hidrografia:hidrografa&outputFormat=application/json"
            f"&count=4000&bbox={p['lat']-RADIO_CAUCE_GRADOS},"
            f"{p['lon']-RADIO_CAUCE_GRADOS},{p['lat']+RADIO_CAUCE_GRADOS},"
            f"{p['lon']+RADIO_CAUCE_GRADOS},urn:ogc:def:crs:EPSG::4326")
        try:
            pet = urllib.request.Request(consulta, headers={"User-Agent": AGENTE})
            with urllib.request.urlopen(pet, timeout=180) as resp:
                gj = json.loads(resp.read().decode("utf-8"))
        except Exception as e:                       # noqa: BLE001
            r["nota_cauce_ide"] = f"consulta fallida: {type(e).__name__}: {e}"
            salida[p["punto"]] = r
            time.sleep(PAUSA_S)
            continue
        time.sleep(PAUSA_S)

        rasgos = gj.get("features", [])
        if not rasgos:
            r["nota_cauce_ide"] = ("0 features: fuera de la cobertura publicada "
                                   "de la capa IDE (llega a ~lat -35)")
            salida[p["punto"]] = r
            continue
        mejor_d, mejor_f = math.inf, None
        for f in rasgos:
            g = f.get("geometry") or {}
            lineas = (g.get("coordinates", []) if g.get("type") == "MultiLineString"
                      else [g.get("coordinates", [])])
            for linea in lineas:
                if len(linea) < 1:
                    continue
                arr = np.asarray(linea, dtype=float)
                d = distancia_a_polilinea(p["lat"], p["lon"], arr[:, 0], arr[:, 1], (0,))
                if d < mejor_d:
                    mejor_d, mejor_f = d, f
        if mejor_f is not None:
            pr = mejor_f.get("properties", {})
            r["dist_cauce_ide_m"] = round(mejor_d, 1)
            r["cauce_ide_tipo"] = pr.get("tipo") or ""
            r["cauce_ide_nombre"] = pr.get("nombre") or ""
            r["cauce_ide_cuenca"] = pr.get("nom_cuen") or ""
            r["cauce_ide_strahler"] = pr.get("strahler_n") or ""
        salida[p["punto"]] = r
        if verboso:
            print(f"  [IDE {i:3d}/{len(puntos)}] {p['punto'][:38]:38s} "
                  f"{r['dist_cauce_ide_m']!s:>9s} m  {r['cauce_ide_tipo']}", flush=True)
    return salida


COLUMNAS_CAUCE = [
    "dist_cauce_m", "cauce_orden_strahler", "cuenca_km2", "caudal_medio_m3s",
    "cauce_id", "fuente_cauce", "epoca_cauce", "confianza_cauce", "nota_cauce",
    "dist_cauce_ide_m", "cauce_ide_tipo", "cauce_ide_nombre", "cauce_ide_cuenca",
    "cauce_ide_strahler", "fuente_cauce_ide", "nota_cauce_ide",
]


def agregar_cauces(con_ide=True, verboso=True):
    """Segunda pasada: mete cauces y `Cuenca` en el CSV ya construido.

    Va aparte del relieve y la cobertura a propósito: son fuentes distintas, con
    licencias y fallos distintos, y si el WFS chileno se cae no puede llevarse
    por delante lo que ya está medido.
    """
    with CSV_SALIDA.open(encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    puntos = [dict(punto=f["punto"], lat=float(f["lat"]), lon=float(f["lon"]))
              for f in filas]

    hr = condiciones_cauces(puntos, verboso=verboso)
    ide = cauces_ide(puntos, verboso=verboso) if con_ide else {}

    for f in filas:
        r = hr.get(f["punto"], {})
        f.update({k: v for k, v in r.items() if k != "nota_cauce"})
        f["fuente_cauce"] = FUENTE_CAUCES["id"]
        f["epoca_cauce"] = FUENTE_CAUCES["epoca_del_dato"]
        f["confianza_cauce"] = FUENTE_CAUCES["confianza_base"] if r.get("dist_cauce_m") != "" else ""
        f["nota_cauce"] = r.get("nota_cauce") or FUENTE_CAUCES["notas"]
        if f.get("cuenca_km2"):
            f["nota_cuenca"] = ("UPLAND_SKM del tramo HydroRIVERS mas cercano: "
                                "es la cuenca de ESE tramo, no la del punto. "
                                "Vale como orden de magnitud del area aguas "
                                "arriba, no como cifra exacta del punto.")
        f.update(ide.get(f["punto"], {}))
        f["fuente_cauce_ide"] = FUENTE_CAUCES_IDE["id"] if con_ide else ""

    columnas = list(COLUMNAS)
    for c in COLUMNAS_CAUCE:
        if c not in columnas:
            columnas.insert(columnas.index("fecha_captura"), c)
    with CSV_SALIDA.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas, extrasaction="ignore")
        w.writeheader()
        for f in filas:
            w.writerow({c: f.get(c, "") for c in columnas})
    if verboso:
        print(f"\nActualizado {CSV_SALIDA} con cauces y cuenca.")
    return filas


# =============================================================================
# 6 · `Reg` GEOLOGÍA y el CATASTRO CHILENO DE USO DE SUELO
# =============================================================================
#
# EL SERVIDOR OFICIAL DE GEOLOGÍA DE SERNAGEOMIN ESTÁ CAÍDO
# ---------------------------------------------------------
# Verificado el 17-ago-2026, en red normal y sin sandbox:
#   portalgeo.sernageomin.cl        → «connect to 190.98.205.167 port 443
#                                      failed: Network is unreachable»
#   geoarcgis.sernageomin.cl/...    → sin conexión (190.98.205.187)
#   geoportal.sernageomin.cl        → ídem, mismo /24
#   portalgeomin.sernageomin.cl     → NXDOMAIN
# Todo el rango legacy 190.98.205.0/24 de SERNAGEOMIN está inalcanzable. El
# `services1.arcgis.com/OyjvVdFTl5hfSdX3` que este proyecto ya usa para las
# minutas tiene 206 FeatureServers, pero ninguno es el mapa geológico nacional
# (el único poligonal, «Geología_10k», cubre un sector de la RM y nada más).
#
# ⚠ PROCEDENCIA FRÁGIL, DECLARADA: el servicio que SÍ responde y SÍ trae el
# esquema del mapa 1:1.000.000 (CD_GEOL, ROCA1..4, AMBIENTE, EDAD…) es una
# RE-SUBIDA a ArcGIS Online hecha por particulares, sin licencia declarada y sin
# garantía de permanencia. NO es la fuente oficial. Se usa porque es lo único
# vivo, con confianza 0,55, se guarda copia local del crudo mientras exista, y
# queda anotado que hay que reemplazarla por la oficial cuando vuelva.
# Las capas de SERNAGEOMIN registradas en datos.gob.cl declaran Creative Commons
# NO COMERCIAL: compatible con este proyecto de investigación, incompatible con
# cualquier uso comercial derivado.
#
# EL CATASTRO DE CONAF NO ESTÁ EN sit.conaf.cl
# --------------------------------------------
# sit.conaf.cl no expone servicio alguno (ArcGIS REST, /server y /geoserver dan
# los tres HTTP 404) y es el único portal del lote con robots.txt restrictivo.
# El catastro se consigue republicado por dos organismos, y los dos sirven:
#   · SMA  — «Catastro de Uso de suelo y vegetación», 16 capas regionales, trae
#            el registro catastral completo (uso, subuso, tipo forestal, especie,
#            cobertura, altura). Es la foto más reciente por región.
#   · CIREN— «Cambio del uso de la tierra CONAF», que trae SEIS fechas por
#            polígono (2001, 2013, 2016, 2017, 2019, 2021) Y las transiciones ya
#            calculadas. Es `Defor` de verdad, no un indicio.
#
# ⚠ TRES CAPAS DE LA SMA TIENEN EL ÍNDICE ESPACIAL ROTO. Comprobado punto por
# punto: Valparaíso (capa 10), Biobío (13) y Araucanía (14) devuelven 0 features
# a cualquier escala, aunque declaren 196.130 registros y un extent que contiene
# el punto. Santiago (18), Aysén (16) y Arica (20) responden bien. No es
# proyección ni tolerancia: la capa está mal indexada del lado del servidor. Para
# esas tres regiones el dato sale de CIREN, y se declara de dónde salió en la
# columna `fuente_uso_suelo`.
#
# ⚠ LOS NOMBRES DE CAMPO CAMBIAN POR REGIÓN: `USO_TIERRA` vs `USO_TIERR`,
# `NOM_COM` vs `COMUNA`, y Aysén no tiene `USO` ni `SUBUSO`. Por eso se pide
# siempre `outFields=*` y se normaliza después: pedir un campo que esa región no
# tiene devuelve `features: []` EN SILENCIO, no un error.

FUENTE_GEOLOGIA = dict(
    id="geologia_chile_1m_agol",
    organismo="esquema de SERNAGEOMIN, servido por re-subida de terceros en ArcGIS Online",
    producto="Geologia de Chile 1:1.000.000 (esquema CD_GEOL/ROCA/AMBIENTE/EDAD)",
    url="https://services.arcgis.com/vQbRu5CNjGqIxk8R/arcgis/rest/services/"
        "geologia_chile_f/FeatureServer/0/query",
    formato="ArcGIS REST, f=json y f=geojson, poligonos",
    familia="TERRENO",
    acceso="anonimo",
    acceso_verificado=1,
    robots_txt="services.arcgis.com: API REST, no aplica exclusion de crawler",
    condiciones_uso="EL SERVICIO NO DECLARA LICENCIA (copyrightText vacio). Las "
                    "capas de SERNAGEOMIN en datos.gob.cl declaran Creative "
                    "Commons NO COMERCIAL. Uso de investigacion, no comercial.",
    atribucion="Mapa Geologico de Chile, SERNAGEOMIN (escala 1:1.000.000)",
    permite_automatizacion="si, con reserva de procedencia",
    granularidad="1:1.000.000 — un poligono puede medir decenas de km2",
    epoca_del_dato="mapa 1:1.000.000; re-subida creada 2025-2026",
    confianza_base=0.55,
    notas="PROCEDENCIA FRAGIL: re-subida por particulares, sin licencia "
          "declarada, sin garantia de permanencia. El servidor oficial de "
          "SERNAGEOMIN (190.98.205.0/24) esta inalcanzable al 17-ago-2026. "
          "Escala 1:1.000.000: sirve para distinguir roca de material suelto, "
          "NO para caracterizar el sitio de una subestacion.",
)

FUENTE_CATASTRO_SMA = dict(
    id="catastro_conaf_via_sma",
    organismo="CONAF, republicado por la Superintendencia del Medio Ambiente",
    producto="Catastro de Uso de suelo y vegetacion, 16 capas regionales",
    url="https://ideserver.sma.gob.cl/arcgis/rest/services/IDE/Biodiversidad/MapServer",
    formato="ArcGIS REST, f=json (NO soporta f=geojson), poligonos",
    familia="TERRENO",
    acceso="anonimo",
    acceso_verificado=1,
    robots_txt="ideserver.sma.gob.cl: 404, sin restricciones",
    condiciones_uso="copyrightText de las capas: 'Corporacion Nacional "
                    "Forestal'. CONAF no publica terminos de uso; la categoria "
                    "equivalente en el catalogo IDE Chile figura como CC-BY.",
    atribucion="Catastro de Recursos Vegetacionales, CONAF; servido por la "
               "Superintendencia del Medio Ambiente (2024)",
    permite_automatizacion="si",
    granularidad="1:50.000",
    epoca_del_dato="catastro base 1993-1997, actualizaciones regionales 2011-2019",
    confianza_base=0.80,
    notas="Capas 10 (Valparaiso), 13 (Biobio) y 14 (Araucania) tienen el indice "
          "espacial roto: 0 features a cualquier escala. La fecha de la foto "
          "cambia por region, de 2011 (Aysen) a 2019 (RM): ocho anios de "
          "desfase entre puntos del mismo CSV.",
)

FUENTE_CAMBIO_CIREN = dict(
    id="cambio_uso_conaf_via_ciren",
    organismo="CONAF, republicado por CIREN (IDE Minagri)",
    producto="Cambio del uso de la tierra CONAF: 6 fechas por poligono con "
             "las transiciones ya calculadas",
    url="https://esri.ciren.cl/server/rest/services/IDEMINAGRI/"
        "CAMBIO_DEL_USO_DE_LA_TIERRA_CONAF/MapServer",
    formato="ArcGIS REST, f=json, poligonos",
    familia="TERRENO",
    acceso="anonimo",
    acceso_verificado=1,
    robots_txt="esri.ciren.cl: 404, sin restricciones",
    condiciones_uso="Catastro de CONAF republicado en la IDE del Minagri; "
                    "sin licencia explicita en el servicio.",
    atribucion="Catastro de Recursos Vegetacionales y su actualizacion, CONAF; "
               "publicado por CIREN / IDE Minagri",
    permite_automatizacion="si, CON REINTENTOS",
    granularidad="1:50.000",
    epoca_del_dato="series por region: 2001/2012/2013/2015-2016/2017/2019/2021",
    confianza_base=0.80,
    notas="Servicio INTERMITENTE: alterna 500 y 200 sin patron; con reintentos "
          "y 2-3 s de pausa entra antes del tercer intento. NO tiene capa de "
          "Antofagasta ni de Aysen.",
)

# --- La regla de `Reg`, escrita a la vista y no dentro de una fórmula ---------
#
# `Reg` del vocabulario §4.1 es «regolito / material suelto disponible». Lo que
# el mapa geológico da son litologías. La traducción se hace por palabra clave
# sobre ROCA1..ROCA4, que es donde el mapa nombra el material, y el resultado
# tiene cuatro valores posibles — uno de ellos es «indeterminado», y se usa.
ROCAS_SUELTAS = (
    "grava", "arena", "limo", "arcilla", "bloque", "till", "morrena", "morena",
    "aluvio", "coluvio", "ceniza", "lapilli", "escoria", "pomacea", "pómez",
    "pumic", "diatomita", "salar", "evaporita", "duna", "loess", "toba",
    "sedimento", "detrito", "caliche", "costra",
)
ROCAS_CONSOLIDADAS = (
    "granito", "granodiorita", "diorita", "gabro", "tonalita", "monzonita",
    "sienita", "pórfido", "porfido", "andesita", "basalto", "dacita", "riolita",
    "traquita", "ignimbrita", "lava", "esquisto", "gneis", "anfibolita",
    "milonita", "serpentinita", "mármol", "marmol", "cuarcita", "pizarra",
    "filita", "caliza", "arenisca", "lutita", "conglomerado", "brecha",
    "grauvaca", "chert", "yeso", "peridotita", "metabasita",
)


NOTA_REGOLITO_CON_DATO = (
    "Escala 1:1.000.000 (un poligono puede medir decenas de km2): sirve para "
    "distinguir roca de material suelto, NO para caracterizar el sitio. "
    "PROCEDENCIA FRAGIL: re-subida a ArcGIS Online por particulares, sin "
    "licencia declarada; el servidor oficial de SERNAGEOMIN esta inalcanzable "
    "al 17-ago-2026. Ver FUENTE_TERRENO.md §7-bis.1.")


def _consulta_arcgis(url_query, lat, lon, carpeta_crudo, etiqueta, intentos=4,
                     campos="*"):
    """Consulta puntual a un ArcGIS REST, guardando la respuesta cruda.

    Se pide siempre `outFields=*`: pedir un campo que esa capa no tiene devuelve
    una lista vacía EN SILENCIO, y eso ya costó pruebas perdidas.
    """
    from urllib.parse import urlencode
    parametros = urlencode({
        "geometry": json.dumps({"x": lon, "y": lat,
                                "spatialReference": {"wkid": 4326}}),
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": campos,
        "returnGeometry": "false",
        "f": "json",
    })
    carpeta_crudo = Path(carpeta_crudo)
    carpeta_crudo.mkdir(parents=True, exist_ok=True)
    ruta = carpeta_crudo / f"{etiqueta}.json"
    if ruta.exists():
        try:
            return json.loads(ruta.read_text(encoding="utf-8")).get("features", []), ""
        except (ValueError, OSError):
            pass
    ultimo = ""
    for intento in range(intentos):
        try:
            pet = urllib.request.Request(f"{url_query}?{parametros}",
                                         headers={"User-Agent": AGENTE})
            with urllib.request.urlopen(pet, timeout=180) as r:
                bruto = r.read()
            datos = json.loads(bruto.decode("utf-8"))
            if isinstance(datos.get("error"), dict):
                ultimo = f"error del servicio: {datos['error'].get('message')}"
            else:
                ruta.write_bytes(bruto)          # ← crudo antes de procesar
                time.sleep(PAUSA_S)
                return datos.get("features", []), ""
        except Exception as e:                   # noqa: BLE001
            ultimo = f"{type(e).__name__}: {e}"
        time.sleep(PAUSA_S * (2 ** intento))
    return [], ultimo


def _capas_por_nombre(url_mapserver):
    """Lee el MapServer y devuelve [(id, nombre)] de sus capas.

    Se consulta en vez de escribirse a mano porque los IDs de capa cambian
    cuando el organismo reordena el servicio, y un ID equivocado devuelve cero
    features sin avisar.
    """
    pet = urllib.request.Request(f"{url_mapserver}?f=json",
                                 headers={"User-Agent": AGENTE})
    with urllib.request.urlopen(pet, timeout=120) as r:
        d = json.loads(r.read().decode("utf-8"))
    time.sleep(PAUSA_S)
    return [(c["id"], c["name"]) for c in d.get("layers", [])]


def _sin_tildes(texto):
    tabla = str.maketrans("áéíóúÁÉÍÓÚñÑüÜ", "aeiouAEIOUnNuU")
    return (texto or "").translate(tabla).lower()


# Palabra que identifica sin ambigüedad a cada región dentro del nombre de una
# capa regional. Escrita a mano porque los organismos escriben el nombre de la
# región de maneras distintas y no queremos que un parecido casual acierte.
CLAVE_REGION = {
    "arica": "arica", "tarapaca": "tarapaca", "antofagasta": "antofagasta",
    "atacama": "atacama", "coquimbo": "coquimbo", "valparaiso": "valparaiso",
    "metropolitana": "metropolitana", "higgins": "higgins", "maule": "maule",
    "nuble": "nuble", "biobio": "biobio", "araucania": "araucania",
    "rios": "rios", "lagos": "lagos", "aysen": "aysen", "aisen": "aysen",
    "magallanes": "magallanes",
}


def _clave_de_region(nombre_region):
    n = _sin_tildes(nombre_region)
    for clave, valor in CLAVE_REGION.items():
        if clave in n:
            return valor
    return None


def clasificar_regolito(atributos):
    """Litología → `Reg`. Devuelve (clase, rocas nombradas).

    Cuatro valores, y «indeterminado» es uno de ellos y se usa: si el mapa no
    nombra una roca que sepamos clasificar, se dice, no se adivina.
    """
    rocas = [str(atributos.get(f"ROCA{i}") or "").strip() for i in (1, 2, 3, 4)]
    rocas = [r for r in rocas if r and r.lower() not in ("none", "null")]
    texto = _sin_tildes(" ".join(rocas))
    suelto = any(p in texto for p in (_sin_tildes(x) for x in ROCAS_SUELTAS))
    firme = any(p in texto for p in (_sin_tildes(x) for x in ROCAS_CONSOLIDADAS))
    if suelto and firme:
        clase = "mixto"
    elif suelto:
        clase = "suelto"
    elif firme:
        clase = "roca"
    else:
        clase = "indeterminado"
    return clase, "; ".join(rocas)


def condiciones_geologia(puntos, verboso=True):
    """`Reg`: litología y clase de material en cada punto."""
    carpeta = CRUDO / "geologia_1m_agol" / FECHA_CAPTURA
    salida = {}
    for i, p in enumerate(puntos, 1):
        etiqueta = _etiqueta(p["punto"])
        rasgos, err = _consulta_arcgis(FUENTE_GEOLOGIA["url"], p["lat"], p["lon"],
                                       carpeta, etiqueta)
        r = dict(regolito="", regolito_rocas="", geologia_codigo="",
                 geologia_unidad="", geologia_ambiente="", geologia_edad_desde="",
                 geologia_edad_hasta="", nota_regolito="")
        if not rasgos:
            r["nota_regolito"] = (f"SIN DATO: {err}" if err else
                                  "SIN DATO: ningun poligono geologico contiene "
                                  "el punto (posible hueco de la capa 1:1.000.000)")
        else:
            a = rasgos[0].get("attributes", {})
            clase, rocas = clasificar_regolito(a)
            r["regolito"] = clase
            r["regolito_rocas"] = rocas
            r["geologia_codigo"] = a.get("CD_GEOL") or ""
            r["geologia_unidad"] = a.get("NOMBRE_DE_") or ""
            r["geologia_ambiente"] = a.get("AMBIENTE") or ""
            r["geologia_edad_desde"] = a.get("EDAD_DESDE") or ""
            r["geologia_edad_hasta"] = a.get("EDAD_HASTA") or ""
            r["nota_regolito"] = NOTA_REGOLITO_CON_DATO
        salida[p["punto"]] = r
        if verboso:
            print(f"  [GEO {i:3d}/{len(puntos)}] {p['punto'][:38]:38s} "
                  f"{r['regolito']:14s} {r['geologia_unidad'][:34]}", flush=True)
    return salida


def _etiqueta(nombre):
    """Nombre de archivo estable y legible para el crudo de cada punto."""
    limpio = _sin_tildes(nombre).replace("·", "-")
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in limpio)[:70]


def condiciones_uso_suelo(puntos, verboso=True):
    """Catastro CONAF: la foto de la SMA y, cuando falla, la serie de CIREN."""
    carpeta_sma = CRUDO / "catastro_conaf_sma" / FECHA_CAPTURA
    carpeta_ciren = CRUDO / "cambio_uso_conaf_ciren" / FECHA_CAPTURA
    capas_sma = _capas_por_nombre(FUENTE_CATASTRO_SMA["url"])
    capas_ciren = _capas_por_nombre(FUENTE_CAMBIO_CIREN["url"])
    # región → ids de capa candidatos (Los Lagos y Magallanes van por provincia,
    # así que una región puede tener varias capas y se prueban todas)
    def indice(capas):
        idx = {}
        for cid, nombre in capas:
            clave = _clave_de_region(nombre)
            if clave:
                idx.setdefault(clave, []).append(cid)
        return idx
    idx_sma, idx_ciren = indice(capas_sma), indice(capas_ciren)

    salida = {}
    for i, p in enumerate(puntos, 1):
        etiqueta = _etiqueta(p["punto"])
        clave = _clave_de_region(p.get("region", ""))
        r = dict(uso_suelo="", uso_suelo_subuso="", uso_suelo_detalle="",
                 uso_suelo_especie="", fuente_uso_suelo="", epoca_uso_suelo="",
                 confianza_uso_suelo="", nota_uso_suelo="",
                 uso_2001="", uso_2021_o_ultimo="", transicion_ultima="",
                 fuente_cambio_uso="", nota_cambio_uso="")
        problemas = []
        if clave is None:
            problemas.append(f"region '{p.get('region','')}' no reconocida")

        # -- 1) la foto de la SMA
        for cid in idx_sma.get(clave, []):
            rasgos, err = _consulta_arcgis(
                f"{FUENTE_CATASTRO_SMA['url']}/{cid}/query", p["lat"], p["lon"],
                carpeta_sma, f"{etiqueta}__capa{cid}")
            if err:
                problemas.append(f"SMA capa {cid}: {err}")
                continue
            if not rasgos:
                problemas.append(f"SMA capa {cid}: 0 features "
                                 f"(indice espacial roto o punto fuera)")
                continue
            a = rasgos[0].get("attributes", {})
            r["uso_suelo"] = a.get("USO") or a.get("USO_TIERRA") or a.get("USO_TIERR") or ""
            r["uso_suelo_subuso"] = a.get("SUBUSO") or ""
            r["uso_suelo_detalle"] = a.get("USO_TIERRA") or a.get("USO_TIERR") or ""
            r["uso_suelo_especie"] = (a.get("ESPECI1_CI") or "").strip()
            r["fuente_uso_suelo"] = FUENTE_CATASTRO_SMA["id"]
            r["epoca_uso_suelo"] = FUENTE_CATASTRO_SMA["epoca_del_dato"]
            r["confianza_uso_suelo"] = FUENTE_CATASTRO_SMA["confianza_base"]
            break

        # -- 2) la serie de CIREN: `Defor` de verdad, y respaldo si la SMA falló
        for cid in idx_ciren.get(clave, []):
            rasgos, err = _consulta_arcgis(
                f"{FUENTE_CAMBIO_CIREN['url']}/{cid}/query", p["lat"], p["lon"],
                carpeta_ciren, f"{etiqueta}__capa{cid}")
            if err:
                problemas.append(f"CIREN capa {cid}: {err}")
                continue
            if not rasgos:
                continue
            a = rasgos[0].get("attributes", {})
            anios = sorted({k[-2:] for k in a if k.startswith("des_uso_")})
            if anios:
                r["uso_2001"] = a.get(f"des_uso_{anios[0]}") or ""
                r["uso_2021_o_ultimo"] = a.get(f"des_uso_{anios[-1]}") or ""
            transiciones = sorted(k for k in a if k.startswith("d_tc_"))
            if transiciones:
                r["transicion_ultima"] = f"{transiciones[-1]}={a[transiciones[-1]]}"
            r["fuente_cambio_uso"] = FUENTE_CAMBIO_CIREN["id"]
            r["nota_cambio_uso"] = next(
                (n for i_, n in capas_ciren if i_ == cid), "")
            if not r["uso_suelo"]:
                r["uso_suelo"] = r["uso_2021_o_ultimo"]
                r["uso_suelo_detalle"] = r["uso_2021_o_ultimo"]
                r["fuente_uso_suelo"] = FUENTE_CAMBIO_CIREN["id"]
                r["epoca_uso_suelo"] = FUENTE_CAMBIO_CIREN["epoca_del_dato"]
                r["confianza_uso_suelo"] = FUENTE_CAMBIO_CIREN["confianza_base"]
            break

        if not r["uso_suelo"]:
            r["nota_uso_suelo"] = "SIN DATO. " + " | ".join(problemas)
        else:
            r["nota_uso_suelo"] = " | ".join(problemas)
        if not r["uso_2021_o_ultimo"]:
            r["nota_cambio_uso"] = ("SIN DATO: CIREN no tiene capa de cambio de "
                                    "uso para esta region (falta Antofagasta y "
                                    "Aysen) o no devolvio poligono")
        salida[p["punto"]] = r
        if verboso:
            print(f"  [USO {i:3d}/{len(puntos)}] {p['punto'][:34]:34s} "
                  f"{(r['uso_suelo'] or 'SIN DATO')[:30]:30s} "
                  f"{r['fuente_uso_suelo'][:24]}", flush=True)
    return salida


COLUMNAS_SUELO = [
    "regolito", "regolito_rocas", "geologia_codigo", "geologia_unidad",
    "geologia_ambiente", "geologia_edad_desde", "geologia_edad_hasta",
    "fuente_regolito", "confianza_regolito", "nota_regolito",
    "uso_suelo", "uso_suelo_subuso", "uso_suelo_detalle", "uso_suelo_especie",
    "fuente_uso_suelo", "epoca_uso_suelo", "confianza_uso_suelo", "nota_uso_suelo",
    "uso_2001", "uso_2021_o_ultimo", "transicion_ultima", "fuente_cambio_uso",
    "nota_cambio_uso",
]


def agregar_suelo(verboso=True):
    """Tercera pasada: `Reg` y el catastro chileno de uso de suelo."""
    with CSV_SALIDA.open(encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    puntos = [dict(punto=f["punto"], lat=float(f["lat"]), lon=float(f["lon"]),
                   region=f.get("region", "")) for f in filas]

    geo = condiciones_geologia(puntos, verboso=verboso)
    uso = condiciones_uso_suelo(puntos, verboso=verboso)

    for f in filas:
        g = geo.get(f["punto"], {})
        f.update(g)
        f["fuente_regolito"] = FUENTE_GEOLOGIA["id"] if g.get("regolito") else ""
        f["confianza_regolito"] = (FUENTE_GEOLOGIA["confianza_base"]
                                   if g.get("regolito") else "")
        f.update(uso.get(f["punto"], {}))

    columnas = list(dict.fromkeys(list(filas[0].keys()) if filas else COLUMNAS))
    for c in COLUMNAS_SUELO:
        if c not in columnas:
            columnas.insert(columnas.index("fecha_captura"), c)
    with CSV_SALIDA.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas, extrasaction="ignore")
        w.writeheader()
        for f in filas:
            w.writerow({c: f.get(c, "") for c in columnas})
    if verboso:
        print(f"\nActualizado {CSV_SALIDA} con regolito y uso de suelo.")
    return filas


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "cauces":
        agregar_cauces()
    elif arg == "cauces-sin-ide":
        agregar_cauces(con_ide=False)
    elif arg == "suelo":
        agregar_suelo()
    else:
        construir(limite=int(arg) if arg else None)
