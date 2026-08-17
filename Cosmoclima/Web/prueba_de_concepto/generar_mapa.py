#!/usr/bin/env python3
"""
Regenera la capa de especies del mapa (prueba_de_concepto_mapa_capas.html)
a partir de TODOS los CSV que haya en datos_fuentes/.

Como agregar datos nuevos (de Guerrero, otro entomologo, etc.):
  1. Crear un CSV nuevo en datos_fuentes/ con exactamente estas columnas:
       especie,voucher,sample_id,localidad,region,lat,lon,alt,anio,fuente
     (lat/lon en grados decimales; "especie" solo el epiteto, ej. "kingi",
     no "Gyriosomus kingi"; "fuente" = de donde salio el dato, para poder
     rastrearlo despues).
  2. Correr: python3 generar_mapa.py
  3. Recargar el HTML en el navegador. Listo -- no hace falta tocar nada
     mas a mano.

Que hace con los datos:
  - Junta todos los CSV de datos_fuentes/ (uno por especimen).
  - Corrige errores de tipeo conocidos en nombres de especie (ver FIX_NOMBRES).
  - Agrupa por especie: capa de puntos + poligono de area (envolvente
    convexa) SOLO si la especie tiene 3 o mas especimenes.
  - Suma la foto real (si existe) de datos_fuentes/fotos_especies.csv --
    generada por buscar_fotos_especies.py (iNaturalist, licencia CC). Correr
    ese script de nuevo solo si se quiere refrescar/ampliar las fotos.
  - Reemplaza el bloque "var especiesData = {...};" dentro del HTML -- no
    toca nada mas del archivo.
"""
import csv
import glob
import json
import os
import re

CARPETA = os.path.dirname(os.path.abspath(__file__))
DATOS_FUENTES = os.path.join(CARPETA, 'datos_fuentes')
# Dos HTML tienen su propio bloque "var especiesData = {...};" embebido -- el
# prototipo de mapa solo, y el instrumento principal (que incrusta su propio
# mapa Leaflet). Se actualizan los dos para que no queden desincronizados.
HTML_PATHS = [
    os.path.join(CARPETA, 'prueba_de_concepto_mapa_capas.html'),
    os.path.join(CARPETA, 'sim-cosmoclima.html'),
]

# Errores de tipeo conocidos en las fuentes ya procesadas -- agregar aca
# si aparecen nuevos al sumar mas datos.
FIX_NOMBRES = {
    'acurtisi': 'curtisi',
    'plantatus': 'planatus',
    'withei': 'whitei',
    'luczoti': 'luczotii',
    # 'hopei' (con una sola p) era el nombre usado en tabla_s1_anguita_salinas_2026.csv,
    # pero el nombre original valido es "hoppei" (Gray, 1832) -- confirmado contra el
    # checklist de 44 especies de Marcelo Guerrero (Listado Gyriosomus Alexis_041512.xlsx,
    # Hoja1). Se corrige aca para no partir la especie en dos entradas del mapa.
    'hopei': 'hoppei',
    # mismo caso: tabla_s1 trae "peniciliger" (una l), el nombre valido es
    # "penicilliger" Gebien, 1944 (dos l), confirmado contra el mismo checklist.
    'peniciliger': 'penicilliger',
}


def leer_registros():
    registros = []
    for csv_path in sorted(glob.glob(os.path.join(DATOS_FUENTES, '*.csv'))):
        with open(csv_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    lat = float(row['lat'])
                    lon = float(row['lon'])
                except (KeyError, ValueError):
                    continue
                especie = FIX_NOMBRES.get(row['especie'].strip().lower(), row['especie'].strip().lower())
                if especie == 'sp':
                    # "Gyriosomus sp." = especimen del genero sin identificar a nivel
                    # de especie (GBIF). No es una especie real -- mantenerla en el
                    # listado solo genera ruido (a pedido de Alexis, 05-ago-2026).
                    continue
                registros.append({
                    'especie': especie,
                    'voucher': row.get('voucher', ''),
                    'localidad': row.get('localidad', ''),
                    'region': row.get('region', ''),
                    'lat': lat,
                    'lon': lon,
                    'alt': row.get('alt', ''),
                    'anio': row.get('anio', ''),
                    'fuente': row.get('fuente', ''),
                })
    return registros


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 2:
        return None
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return None
    hull.append(hull[0])
    return hull


def leer_fotos():
    """Fotos reales con licencia CC por especie -- ver buscar_fotos_especies.py.
    Si el CSV no existe todavia (primera vez), simplemente no hay fotos."""
    fotos = {}
    ruta = os.path.join(DATOS_FUENTES, 'fotos_especies.csv')
    if not os.path.exists(ruta):
        return fotos
    with open(ruta, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            especie = FIX_NOMBRES.get(row['especie'].strip().lower(), row['especie'].strip().lower())
            fotos[especie] = {
                'url': row['url_foto'],
                'licencia': row['licencia'],
                'atribucion': row['atribucion'],
                'pagina_origen': row['pagina_origen'],
            }
    return fotos


def construir_especies_data(registros, fotos):
    por_especie = {}
    for r in registros:
        por_especie.setdefault(r['especie'], []).append(r)

    especies_out = {}
    for especie, regs in sorted(por_especie.items()):
        puntos_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [r['lon'], r['lat']]},
                    "properties": {
                        "voucher": r['voucher'], "localidad": r['localidad'],
                        "region": r['region'], "alt": r['alt'], "anio": r['anio'],
                        "fuente": r['fuente'],
                    }
                } for r in regs
            ]
        }
        coords = [(r['lon'], r['lat']) for r in regs]
        hull = convex_hull(coords)
        area_geojson = None
        if hull:
            area_geojson = {
                "type": "Feature",
                "properties": {"especie": especie},
                "geometry": {"type": "Polygon", "coordinates": [hull]}
            }
        especies_out[especie] = {
            "n": len(regs), "puntos": puntos_geojson, "area": area_geojson,
            "foto": fotos.get(especie),
        }
    return especies_out


def actualizar_html(especies_data):
    patron = re.compile(r'var especiesData = \{.*?\};\n', re.S)
    nuevo_bloque = 'var especiesData = ' + json.dumps(especies_data, ensure_ascii=False) + ';\n'

    for html_path in HTML_PATHS:
        with open(html_path, encoding='utf-8') as f:
            html = f.read()
        if not patron.search(html):
            raise SystemExit(f'No se encontro "var especiesData = {{...}};" en {html_path} -- revisar a mano.')
        html_nuevo = patron.sub(nuevo_bloque, html, count=1)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_nuevo)


def main():
    registros = leer_registros()
    if not registros:
        raise SystemExit('No se encontraron registros validos en datos_fuentes/*.csv')

    fotos = leer_fotos()
    especies_data = construir_especies_data(registros, fotos)
    actualizar_html(especies_data)

    con_area = sum(1 for v in especies_data.values() if v['area'])
    con_foto = sum(1 for v in especies_data.values() if v['foto'])
    print(f'Especimenes totales: {len(registros)}')
    print(f'Especies: {len(especies_data)} ({con_area} con area, {len(especies_data) - con_area} solo puntos, {con_foto} con foto real)')
    for p in HTML_PATHS:
        print('HTML actualizado:', p)


if __name__ == '__main__':
    main()
