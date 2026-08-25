"""
El traductor territorial: dada una coordenada, decir dónde está — en los dos
idiomas que habla Chile.

EL PROBLEMA DE LAS DOS GEOGRAFÍAS
---------------------------------
La AMENAZA se declara por franja geográfica: «peligro de aluvión Alto en
Precordillera Alto Loa» (SERNAGEOMIN), «viento 90-110 km/h en Cordillera de
Atacama» (DMC).
La ALERTA se declara por unidad administrativa: «Alerta Temprana Preventiva para
la provincia de El Loa y las comunas de Antofagasta y Taltal» (SENAPRED).

Son dos particiones distintas del mismo territorio y no coinciden. El
instrumento tiene que hablar las dos y traducir entre ellas, porque recibe en
una y entrega en la otra.

POR QUÉ SIN GEOPANDAS
---------------------
Se resuelve con geometría a mano en Python puro. La razón no es purismo: en este
mismo proyecto, instalar la pila geoespacial (gdal/fiona) falló dos veces y
costó horas. Para lo que hace falta acá —meter unos cientos de puntos dentro de
unos cientos de polígonos— el algoritmo de rayo alcanza y sobra, y no arrastra
dependencias que se rompen.

ESTADO
------
La lógica está completa y probada con polígonos sintéticos. Falta enchufarle las
capas oficiales: la de límites comunales y la de zonas geográficas. Mientras no
estén, `ubicar()` devuelve lo que puede y marca lo que no — nunca inventa una
comuna. Es la misma regla del proyecto: un hueco declarado vale más que un dato
adivinado.
"""

import json
from pathlib import Path

AQUI = Path(__file__).parent
CAPAS = AQUI / "datos" / "capas"

# Nombres esperados de las capas. Se completan cuando el catastro entregue las
# URLs oficiales; hasta entonces el módulo funciona en modo degradado.
CAPA_COMUNAS = CAPAS / "comunas.geojson"
CAPA_ZONAS = CAPAS / "zonas_geograficas.geojson"

# Cómo se llaman los campos dentro de cada capa. Se declaran acá porque cada
# organismo bautiza distinto y no queremos adivinar dentro del código.
# Verificados el 15-ago-2026 contra las capas reales de SERNAGEOMIN: la de
# comunas usa COMUNA/PROVINCIA/REGION/CUT_COM y la de zonas usa ZONA.
CAMPOS_COMUNA = {
    "comuna": ("COMUNA", "Comuna", "comuna", "NOM_COMUNA", "nom_comuna", "NOMBRE"),
    "provincia": ("PROVINCIA", "Provincia", "provincia", "NOM_PROVIN"),
    "region": ("REGION", "Region", "region", "NOM_REGION", "Región"),
    # ★ El CUT se trata SIEMPRE como texto de 5 caracteres. El INE lo publica
    # como entero (2101) y SENAPRED como texto con cero adelante ('05303'): si
    # se guarda como número, las comunas de la I a la IX región pierden el cero
    # y dejan de cruzar en silencio.
    "codigo": ("CUT_COM", "cod_comuna", "COD_COMUNA", "CUT", "cut"),
}
CAMPOS_ZONA = {"zona": ("ZONA", "zona", "nombre", "NOMBRE", "Zona")}


# ── geometría ────────────────────────────────────────────────────────────────

def _en_anillo(lon, lat, anillo):
    """Algoritmo del rayo: se lanza una semirrecta horizontal desde el punto y
    se cuentan los cruces con el borde. Impar = adentro, par = afuera.

    Es el mismo truco de dibujar una línea en un mapa y contar cuántas veces
    cruza la frontera: si cruzás un número impar de veces, quedaste adentro.
    """
    adentro = False
    n = len(anillo)
    j = n - 1
    for i in range(n):
        xi, yi = anillo[i][0], anillo[i][1]
        xj, yj = anillo[j][0], anillo[j][1]
        # ¿el borde i-j cruza la altura del punto?
        if (yi > lat) != (yj > lat):
            # ¿lo cruza a la derecha del punto?
            corte_x = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < corte_x:
                adentro = not adentro
        j = i
    return adentro


def _en_poligono(lon, lat, coordenadas):
    """Un polígono GeoJSON: anillo exterior primero, huecos después.
    Estar dentro del exterior pero dentro de un hueco es estar afuera."""
    if not coordenadas:
        return False
    if not _en_anillo(lon, lat, coordenadas[0]):
        return False
    return not any(_en_anillo(lon, lat, hueco) for hueco in coordenadas[1:])


def _en_geometria(lon, lat, geometria):
    """Soporta Polygon y MultiPolygon, que es lo que traen estas capas."""
    if geometria is None:
        return False
    tipo = geometria.get("type")
    coords = geometria.get("coordinates")
    if tipo == "Polygon":
        return _en_poligono(lon, lat, coords)
    if tipo == "MultiPolygon":
        return any(_en_poligono(lon, lat, p) for p in coords)
    return False


def _caja(geometria):
    """Rectángulo que envuelve la geometría. Descartar por rectángulo antes de
    hacer el cálculo fino es lo que vuelve rápida la búsqueda: casi todos los
    polígonos se eliminan con cuatro comparaciones."""
    lons, lats = [], []

    def recorrer(c):
        if isinstance(c[0], (int, float)):
            lons.append(c[0]); lats.append(c[1])
        else:
            for sub in c:
                recorrer(sub)

    coords = geometria.get("coordinates")
    if not coords:
        return None
    recorrer(coords)
    return (min(lons), min(lats), max(lons), max(lats))


def cut_a_texto(valor):
    """El Código Único Territorial, SIEMPRE como texto de 5 caracteres.

    Es una trampa real y silenciosa: el INE publica el CUT como entero (2101) y
    SENAPRED como texto con cero a la izquierda ('05303'). Si se guarda como
    número, todas las comunas de la I a la IX región pierden el cero delante y
    dejan de cruzar — sin error, sin aviso, simplemente no encuentran pareja.
    """
    if valor is None or str(valor).strip() == "":
        return None
    return str(valor).strip().zfill(5)


def _primer_campo(propiedades, candidatos):
    """Devuelve el primer campo que exista, probando varios nombres posibles."""
    for nombre in candidatos:
        if nombre in propiedades and propiedades[nombre] not in (None, ""):
            return propiedades[nombre]
    return None


# ── carga de capas ───────────────────────────────────────────────────────────

class Capa:
    """Una capa de polígonos, con su caja envolvente precalculada."""

    def __init__(self, ruta, campos, nombre_legible):
        self.ruta = Path(ruta)
        self.campos = campos
        self.nombre = nombre_legible
        self.rasgos = []
        self.disponible = False
        if self.ruta.exists():
            self._cargar()

    def _cargar(self):
        with self.ruta.open(encoding="utf-8") as fh:
            datos = json.load(fh)
        for rasgo in datos.get("features", []):
            geometria = rasgo.get("geometry")
            caja = _caja(geometria) if geometria else None
            if caja is None:
                continue
            self.rasgos.append({
                "geometria": geometria,
                "caja": caja,
                "propiedades": rasgo.get("properties", {}),
            })
        self.disponible = bool(self.rasgos)

    def buscar(self, lat, lon):
        """Devuelve las propiedades del polígono que contiene el punto, o None."""
        for rasgo in self.rasgos:
            xmin, ymin, xmax, ymax = rasgo["caja"]
            if not (xmin <= lon <= xmax and ymin <= lat <= ymax):
                continue          # descarte barato por rectángulo
            if _en_geometria(lon, lat, rasgo["geometria"]):
                return rasgo["propiedades"]
        return None


class Territorio:
    """Las dos capas juntas, con la operación que importa: ubicar un punto."""

    def __init__(self, capa_comunas=CAPA_COMUNAS, capa_zonas=CAPA_ZONAS):
        self.comunas = Capa(capa_comunas, CAMPOS_COMUNA, "límites comunales")
        self.zonas = Capa(capa_zonas, CAMPOS_ZONA, "zonas geográficas")

    def estado(self):
        """Qué capas hay cargadas. Se reporta para que nunca se confunda
        «no hay capa» con «el punto no cayó en ninguna»."""
        return {
            "comunas": ("cargada" if self.comunas.disponible else "AUSENTE",
                        len(self.comunas.rasgos)),
            "zonas": ("cargada" if self.zonas.disponible else "AUSENTE",
                      len(self.zonas.rasgos)),
        }

    def ubicar(self, lat, lon):
        """Coordenada → dónde está, en los dos idiomas.

        Nunca inventa. Si la capa no está, el campo sale None y `faltan` dice
        cuál faltó: hay que poder distinguir «no lo sé porque no tengo la capa»
        de «no lo sé porque el punto cayó en el mar».
        """
        salida = {"lat": lat, "lon": lon, "comuna": None, "provincia": None,
                  "region": None, "codigo_comuna": None, "zona_geografica": None,
                  "faltan": [], "resuelto": False}

        if not self.comunas.disponible:
            salida["faltan"].append("capa_comunas_ausente")
        else:
            props = self.comunas.buscar(lat, lon)
            if props is None:
                salida["faltan"].append("punto_fuera_de_toda_comuna")
            else:
                salida["comuna"] = _primer_campo(props, CAMPOS_COMUNA["comuna"])
                salida["provincia"] = _primer_campo(props, CAMPOS_COMUNA["provincia"])
                salida["region"] = _primer_campo(props, CAMPOS_COMUNA["region"])
                salida["codigo_comuna"] = cut_a_texto(
                    _primer_campo(props, CAMPOS_COMUNA["codigo"]))

        if not self.zonas.disponible:
            salida["faltan"].append("capa_zonas_ausente")
        else:
            props = self.zonas.buscar(lat, lon)
            if props is None:
                salida["faltan"].append("punto_fuera_de_toda_zona")
            else:
                salida["zona_geografica"] = _primer_campo(props, CAMPOS_ZONA["zona"])

        salida["resuelto"] = not salida["faltan"]
        return salida


if __name__ == "__main__":
    CAPAS.mkdir(parents=True, exist_ok=True)
    t = Territorio()
    print("Estado de las capas:")
    for nombre, (estado, n) in t.estado().items():
        print(f"   {nombre:10s} {estado:8s} ({n} polígonos)")
    if not (t.comunas.disponible and t.zonas.disponible):
        print("\nFaltan capas. Descargarlas a datos/capas/ según el catastro.")
        print("Mientras tanto ubicar() marca lo que no puede resolver, y no "
              "inventa nada.")
    print("\nPrueba con las subestaciones (Copiapó y Punta Arenas):")
    for nombre, lat, lon in (("SE Copiapó", -27.3793, -70.3363),
                             ("SE Punta Arenas", -53.16, -70.91)):
        r = t.ubicar(lat, lon)
        print(f"   {nombre:18s} → comuna={r['comuna']} zona={r['zona_geografica']} "
              f"faltan={r['faltan']}")
