"""
Resuelve el hallazgo H-13: darle comuna a las 39 subestaciones.

EL PROBLEMA
-----------
La sub-matriz de subestaciones trae región y provincia, pero NO comuna. Y la
comuna es justo el nivel donde opera el COGRID comunal, el primero que responde
cuando algo falla.

CÓMO SE RESUELVE — y cómo NO
-----------------------------
Se deriva de la COORDENADA, preguntándole al servicio oficial en qué comuna cae
cada punto. **Nunca del nombre de la subestación**: «Chungará», «Collahuasi»,
«Crucero», «Maitencillo», «Guindo» y «Chacabuco» son localidades, faenas o
sectores, no comunas. Deducirlas por nombre sería inventar, y este proyecto no
inventa datos territoriales.

POR QUÉ CONSULTA AL SERVICIO EN VEZ DE USAR LA CAPA LOCAL
---------------------------------------------------------
La capa local de comunas bajó con 103 MB aun pidiéndola generalizada: cargarla
entera en memoria para ubicar 39 puntos es desproporcionado. La consulta
espacial del propio servicio es exacta (la hace el servidor con la geometría de
máximo detalle), pesa nada y son 39 peticiones. Para el pipeline general habrá
que resolver la capa local; para esto, no hace falta.

CONTROL INDEPENDIENTE
---------------------
La sub-matriz ya trae región y provincia escritas a mano. Al resolver por
coordenada obtenemos las mismas dos cosas por otra vía. Si coinciden, tenemos
dos fuentes independientes diciendo lo mismo: eso valida a la vez las
coordenadas, la capa y el método. Si no coinciden, hay un error en alguna de las
tres y hay que mirarlo antes de seguir.
"""

import csv
import json
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))
from territorio import cut_a_texto  # noqa: E402

PUNTOS = AQUI / "datos" / "subestaciones_puntos.csv"
SALIDA = AQUI / "datos" / "subestaciones_con_comuna.csv"

CAPA_COMUNAS = ("https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/"
                "services/minutasATG_Flash/FeatureServer/3")
CAPA_ZONAS = ("https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/"
              "services/minutasATG_Flash/FeatureServer/2")


def consultar_punto(capa, lat, lon, campos, timeout=60):
    """Pregunta al servicio qué polígono contiene el punto. None si ninguno."""
    geometria = json.dumps({"x": lon, "y": lat,
                            "spatialReference": {"wkid": 4326}})
    params = {
        "geometry": geometria, "geometryType": "esriGeometryPoint",
        "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
        "outFields": campos, "returnGeometry": "false", "f": "json",
    }
    url = f"{capa}/query?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            datos = json.load(r)
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:80]}"
    if "error" in datos:
        return None, str(datos["error"])[:120]
    rasgos = datos.get("features", [])
    if not rasgos:
        return None, "el punto no cayó en ningún polígono"
    return rasgos[0].get("attributes", {}), None


def limpiar(t):
    """Sin tildes, sin mayúsculas, sin apóstrofos, sin «región de», sin espacios
    de más. Todo lo que sea decoración del nombre y no el nombre."""
    t = unicodedata.normalize("NFD", str(t or "").strip().lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = t.replace("'", "").replace("’", "")
    for ruido in ("region de la ", "region de ", "region del ", "region ",
                  "provincia de ", "provincia "):
        if t.startswith(ruido):
            t = t[len(ruido):]
    return " ".join(t.split())


def igual(a, b):
    """¿Son el mismo lugar, aunque estén escritos distinto?

    El país escribe sus regiones de dos maneras: la corta de todos los días
    («Araucanía», «O'Higgins», «Magallanes») y la formal completa («La
    Araucanía», «Libertador General Bernardo O'Higgins», «Magallanes y de la
    Antártica Chilena»). Comparar carácter a carácter marcaría catorce
    discrepancias que no existen y escondería las dos que sí.

    Por eso: si una está contenida en la otra, es el mismo lugar. «Aysén» dentro
    de «Aysén del General Carlos Ibáñez del Campo» es la misma región; «Iquique»
    y «Tamarugal» no se contienen, y ahí sí hay algo que mirar.
    """
    x, y = limpiar(a), limpiar(b)
    if not x or not y:
        return False
    return x == y or x in y or y in x


def main():
    with PUNTOS.open(encoding="utf-8") as fh:
        puntos = list(csv.DictReader(fh))

    filas, discrepancias, sin_resolver = [], [], []
    print(f"Resolviendo {len(puntos)} subestaciones contra el servicio oficial\n")
    print(f"{'Subestación':34s} {'Comuna':18s} {'CUT':6s} {'Zona morfoclimática':32s} ok")

    for p in puntos:
        lat, lon = float(p["lat"]), float(p["lon"])
        com, err_c = consultar_punto(CAPA_COMUNAS, lat, lon,
                                     "CUT_COM,COMUNA,PROVINCIA,REGION")
        zon, err_z = consultar_punto(CAPA_ZONAS, lat, lon, "ZONA,REGION")
        time.sleep(0.3)          # cortesía con un servicio público

        comuna = (com or {}).get("COMUNA")
        cut = cut_a_texto((com or {}).get("CUT_COM"))
        provincia = (com or {}).get("PROVINCIA")
        region = (com or {}).get("REGION")
        zona = (zon or {}).get("ZONA")

        # control independiente contra lo que ya traía la sub-matriz
        coincide_region = igual(region, p["region"])
        coincide_prov = igual(provincia, p["provincia"])
        marca = "✓" if (coincide_region and coincide_prov) else "✗"
        if com is None:
            marca = "—"
            sin_resolver.append((p["subestacion"], err_c))
        elif not (coincide_region and coincide_prov):
            discrepancias.append((p["subestacion"], p["region"], region,
                                  p["provincia"], provincia))

        print(f"{p['subestacion'][:34]:34s} {str(comuna)[:18]:18s} "
              f"{str(cut):6s} {str(zona)[:32]:32s} {marca}")

        filas.append({**p, "comuna": comuna, "cut_comuna": cut,
                      "provincia_verificada": provincia,
                      "region_verificada": region,
                      "zona_morfoclimatica": zona,
                      "coincide_con_submatriz": int(coincide_region and coincide_prov),
                      "nota": err_c or err_z or ""})

    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    resueltas = sum(1 for f in filas if f["comuna"])
    con_zona = sum(1 for f in filas if f["zona_morfoclimatica"])
    print(f"\nComuna resuelta:        {resueltas}/{len(filas)}")
    print(f"Zona morfoclimática:    {con_zona}/{len(filas)}")
    print(f"Coincide con submatriz: {sum(f['coincide_con_submatriz'] for f in filas)}"
          f"/{len(filas)}")

    if discrepancias:
        print("\n★ DISCREPANCIAS — la coordenada dice una cosa y la sub-matriz otra.")
        print("  No se corrige nada: se reporta para que Alexis decida cuál manda.")
        for se, r_sub, r_geo, p_sub, p_geo in discrepancias:
            print(f"    {se}")
            print(f"       sub-matriz: {r_sub} / {p_sub}")
            print(f"       coordenada: {r_geo} / {p_geo}")
    if sin_resolver:
        print("\n  Sin resolver (se deja en blanco, no se inventa):")
        for se, motivo in sin_resolver:
            print(f"    {se}: {motivo}")

    print(f"\nEscrito: {SALIDA.relative_to(AQUI)}")


if __name__ == "__main__":
    main()
