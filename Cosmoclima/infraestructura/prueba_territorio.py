"""
Prueba la geometría de `territorio.py` con polígonos inventados a propósito.

Se prueba ahora, antes de tener las capas oficiales, para que cuando lleguen ya
sepamos que el algoritmo no es el problema. Los casos difíciles están puestos
adrede: bordes, huecos (una comuna con un enclave adentro), y multipolígonos
(una comuna con islas, que en Chile es lo normal — Castro, Punta Arenas).

Corre sobre archivos temporales; no toca nada del proyecto.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import territorio  # noqa: E402

OK, FALLO = "  ✓", "  ✗ FALLÓ"


def geojson(rasgos):
    return {"type": "FeatureCollection", "features": rasgos}


def rasgo(geom, props):
    return {"type": "Feature", "geometry": geom, "properties": props}


# Un cuadrado simple, con un hueco cuadrado en el medio (tipo enclave).
CUADRADO_CON_HUECO = {
    "type": "Polygon",
    "coordinates": [
        [[-71.0, -34.0], [-70.0, -34.0], [-70.0, -33.0], [-71.0, -33.0],
         [-71.0, -34.0]],                                   # exterior
        [[-70.7, -33.7], [-70.3, -33.7], [-70.3, -33.3], [-70.7, -33.3],
         [-70.7, -33.7]],                                   # hueco
    ],
}

# Dos trozos separados: continente + isla.
CONTINENTE_E_ISLA = {
    "type": "MultiPolygon",
    "coordinates": [
        [[[-74.0, -43.0], [-73.5, -43.0], [-73.5, -42.5], [-74.0, -42.5],
          [-74.0, -43.0]]],
        [[[-73.0, -43.0], [-72.8, -43.0], [-72.8, -42.8], [-73.0, -42.8],
          [-73.0, -43.0]]],
    ],
}


def main():
    resultados = []
    with tempfile.TemporaryDirectory() as tmp:
        ruta_com = Path(tmp) / "comunas.geojson"
        ruta_zon = Path(tmp) / "zonas.geojson"

        ruta_com.write_text(json.dumps(geojson([
            rasgo(CUADRADO_CON_HUECO,
                  {"Comuna": "Cuadrada", "Provincia": "P1", "Region": "R1",
                   "cod_comuna": "13101"}),
            rasgo(CONTINENTE_E_ISLA,
                  {"Comuna": "Archipiélago", "Provincia": "P2", "Region": "R2",
                   "cod_comuna": "10201"}),
        ])), encoding="utf-8")

        ruta_zon.write_text(json.dumps(geojson([
            rasgo({"type": "Polygon",
                   "coordinates": [[[-72.0, -35.0], [-69.0, -35.0],
                                    [-69.0, -32.0], [-72.0, -32.0],
                                    [-72.0, -35.0]]]},
                  {"zona": "Valle Central"}),
        ])), encoding="utf-8")

        t = territorio.Territorio(ruta_com, ruta_zon)

        estado = t.estado()
        ok = estado["comunas"][0] == "cargada" and estado["comunas"][1] == 2
        print(f"{OK if ok else FALLO}  carga 2 comunas y 1 zona  ({estado})")
        resultados.append(ok)

        casos = [
            # (nombre, lat, lon, comuna esperada, zona esperada)
            ("dentro del cuadrado", -33.1, -70.9, "Cuadrada", "Valle Central"),
            ("dentro del HUECO → afuera", -33.5, -70.5, None, "Valle Central"),
            ("fuera de todo", -40.0, -73.0, None, None),
            ("en el continente", -42.7, -73.8, "Archipiélago", None),
            ("en la ISLA", -42.9, -72.9, "Archipiélago", None),
            ("entre continente e isla → agua", -42.9, -73.2, None, None),
        ]
        for nombre, lat, lon, com_esp, zon_esp in casos:
            r = t.ubicar(lat, lon)
            ok = (r["comuna"] == com_esp and r["zona_geografica"] == zon_esp)
            print(f"{OK if ok else FALLO}  {nombre:32s} → comuna={r['comuna']} "
                  f"zona={r['zona_geografica']}")
            resultados.append(ok)

        # los campos derivados tienen que venir completos cuando sí ubica
        r = t.ubicar(-33.1, -70.9)
        ok = (r["provincia"] == "P1" and r["region"] == "R1"
              and r["codigo_comuna"] == "13101" and r["resuelto"] is True)
        print(f"{OK if ok else FALLO}  arrastra provincia, región y código  "
              f"({r['provincia']}, {r['region']}, {r['codigo_comuna']}, "
              f"resuelto={r['resuelto']})")
        resultados.append(ok)

        # distinguir «no tengo capa» de «cayó afuera»
        t_sin = territorio.Territorio(Path(tmp) / "no_existe.geojson", ruta_zon)
        r = t_sin.ubicar(-33.1, -70.9)
        ok = "capa_comunas_ausente" in r["faltan"] and not r["resuelto"]
        print(f"{OK if ok else FALLO}  sin capa avisa «ausente», no «afuera»  "
              f"({r['faltan']})")
        resultados.append(ok)

    total, bien = len(resultados), sum(resultados)
    print(f"\n{'TODO EN ORDEN' if bien == total else '¡HAY FALLAS!'}: "
          f"{bien}/{total} casos")
    return 0 if bien == total else 1


if __name__ == "__main__":
    sys.exit(main())
