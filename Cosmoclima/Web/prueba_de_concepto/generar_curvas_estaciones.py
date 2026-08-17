#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generar_curvas_estaciones.py -- arma las 2 constantes JS que necesita el
instrumento ET3-Térmico para mostrar, especie por especie (a pedido), la
curva de lluvia REAL de su estación más cercana -- sin tocar la curva única
del reloj (PLUVIOSIDAD_MENSUAL, Huintil/CR2+NASA POWER, que sigue intacta):

  LLUVIA_ESTACIONES   -- { "Nombre Estación": {lat, lon, serie:{"YYYY-MM":mm}} }
                         solo las estaciones que resultan "más cercana" de
                         alguna especie (no las ~97 completas, para no inflar
                         el HTML con datos que ninguna curva va a usar).
  ESTACION_MAS_CERCANA -- { "epiteto": {estacion, dist_km} } para las 42
                         especies reales (se excluye "sp", especímenes sin
                         identificar -- ver generar_mapa.py).

Fuente de la lluvia: investigacion/fuentes/precipitacion_mensual_dmc_anuario2025.csv
(mismo archivo que ya alimenta el mapa de especies, columnas
estacion,lat,lon,mes,anio,lluvia_mm). Fuente de las especies: el propio
bloque "var especiesData = {...}" ya embebido en el HTML (no se recalcula
geometría, solo se lee).

Correr despues de generar_mapa.py si cambiaron los datos de especies, y
cada vez que se sume una campaña nueva a precipitacion_mensual_dmc_anuario2025.csv.
"""
import csv
import json
import math
import os
import re

CARPETA = os.path.dirname(os.path.abspath(__file__))
CSV_LLUVIA = os.path.join(CARPETA, '..', '..', 'investigacion', 'fuentes', 'precipitacion_mensual_dmc_anuario2025.csv')
HTML_PATH = os.path.join(CARPETA, 'sim-cosmoclima.html')
# TODAS_LAS_ESTACIONES (capa "Estaciones meteorológicas" del mapa) va en LOS
# DOS html -- LLUVIA_ESTACIONES/ESTACION_MAS_CERCANA (curva por especie) solo
# en el ET3, que es el unico que tiene el grafico popChart/agregarLineaEspecie.
HTML_PATH_SIMPLE = os.path.join(CARPETA, 'prueba_de_concepto_mapa_capas.html')


def leer_estaciones():
    estaciones = {}  # nombre -> {lat, lon, serie: {"YYYY-MM": mm}}
    with open(CSV_LLUVIA, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not row['lat'] or not row['lon']:
                continue
            nombre = row['estacion']
            if nombre not in estaciones:
                estaciones[nombre] = {
                    'lat': round(float(row['lat']), 4),
                    'lon': round(float(row['lon']), 4),
                    'serie': {},
                }
            clave = f"{int(row['anio']):04d}-{int(row['mes']):02d}"
            if row['lluvia_mm']:
                estaciones[nombre]['serie'][clave] = float(row['lluvia_mm'])
    return estaciones


def leer_especies_data():
    with open(HTML_PATH, encoding='utf-8') as f:
        html = f.read()
    m = re.search(r'var especiesData = (\{.*?\});\n', html, re.S)
    return html, json.loads(m.group(1))


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def calcular_estacion_mas_cercana(especies_data, estaciones):
    resultado = {}
    usadas = set()
    for epiteto, d in especies_data.items():
        if epiteto == 'sp':
            continue
        feats = d['puntos']['features']
        lat_c = sum(f['geometry']['coordinates'][1] for f in feats) / len(feats)
        lon_c = sum(f['geometry']['coordinates'][0] for f in feats) / len(feats)
        mejor_nombre, mejor_dist = None, None
        for nombre, info in estaciones.items():
            dist = haversine(lat_c, lon_c, info['lat'], info['lon'])
            if mejor_dist is None or dist < mejor_dist:
                mejor_nombre, mejor_dist = nombre, dist
        resultado[epiteto] = {'estacion': mejor_nombre, 'dist_km': round(mejor_dist, 1)}
        usadas.add(mejor_nombre)
    return resultado, usadas


def resumir_todas_estaciones(estaciones):
    salida = {}
    for nombre, info in estaciones.items():
        anios = sorted(set(k[:4] for k in info['serie'].keys()))
        n_meses = len(info['serie'])
        if not anios:
            continue
        salida[nombre] = {
            'lat': info['lat'], 'lon': info['lon'],
            'anio_min': anios[0], 'anio_max': anios[-1], 'n_meses': n_meses,
        }
    return salida


def inyectar(html, lluvia_estaciones, estacion_mas_cercana, todas_estaciones):
    bloque = (
        '// LLUVIA_ESTACIONES / ESTACION_MAS_CERCANA / TODAS_LAS_ESTACIONES --\n'
        '// generado por generar_curvas_estaciones.py (06-ago-2026, a pedido de Alexis):\n'
        '// NO reemplaza PLUVIOSIDAD_MENSUAL (el reloj sigue corriendo con Huintil/CR2+NASA\n'
        '// POWER, una sola curva). LLUVIA_ESTACIONES/ESTACION_MAS_CERCANA son la capa\n'
        '// adicional bajo demanda (curva de lluvia de la estacion mas cercana al marcar una\n'
        '// especie). TODAS_LAS_ESTACIONES (110) es para la capa "Estaciones meteorológicas"\n'
        '// del mapa -- verificacion visual pedida por Alexis tras encontrar el problema de\n'
        '// Tal Tal (excluida por error) y preguntar por Chañaral: mejor ver TODAS las\n'
        '// estaciones que tenemos en un mapa que confiar en listas de texto.\n'
        'const LLUVIA_ESTACIONES = ' + json.dumps(lluvia_estaciones, ensure_ascii=False) + ';\n'
        'const ESTACION_MAS_CERCANA = ' + json.dumps(estacion_mas_cercana, ensure_ascii=False) + ';\n'
        'const TODAS_LAS_ESTACIONES = ' + json.dumps(todas_estaciones, ensure_ascii=False) + ';\n'
    )
    marcador_ini = '// === INICIO LLUVIA_ESTACIONES (generado) ===\n'
    marcador_fin = '// === FIN LLUVIA_ESTACIONES (generado) ===\n'
    patron = re.compile(re.escape(marcador_ini) + '.*?' + re.escape(marcador_fin), re.S)
    bloque_completo = marcador_ini + bloque + marcador_fin
    if patron.search(html):
        return patron.sub(bloque_completo, html)
    # primera vez: insertar justo despues de la linea de PLUVIOSIDAD_MENSUAL
    ancla = re.search(r'const PLUVIOSIDAD_MENSUAL = \{.*?\};\n', html, re.S)
    if not ancla:
        raise SystemExit('No se encontro PLUVIOSIDAD_MENSUAL en el HTML -- revisar a mano.')
    pos = ancla.end()
    return html[:pos] + bloque_completo + html[pos:]


def inyectar_solo_estaciones(html, todas_estaciones):
    """Version reducida para prueba_de_concepto_mapa_capas.html -- ese archivo
    no tiene PLUVIOSIDAD_MENSUAL ni el grafico popChart. TODAS_LAS_ESTACIONES
    tiene que quedar declarada ANTES del codigo Leaflet que la usa (la capa
    "Estaciones meteorologicas", justo despues del setup de NDVI) -- ese
    codigo esta MUY arriba en este archivo, antes que especiesData. Anclar
    despues de especiesData (como se hizo la primera vez) rompe la pagina:
    "ReferenceError: TODAS_LAS_ESTACIONES is not defined" por temporal dead
    zone de `const` -- se detecto asi, con Alexis reportando que la capa no
    aparecia. Ahora se ancla justo al abrir el <script> principal (primera
    linea util del archivo), garantizado ANTES de cualquier uso posible."""
    bloque = (
        '// TODAS_LAS_ESTACIONES -- generado por generar_curvas_estaciones.py\n'
        '// (06-ago-2026, a pedido de Alexis): capa "Estaciones meteorológicas" del\n'
        '// mapa, para verificacion visual (asi se encontro el hueco real de Chañaral).\n'
        'const TODAS_LAS_ESTACIONES = ' + json.dumps(todas_estaciones, ensure_ascii=False) + ';\n'
    )
    marcador_ini = '// === INICIO TODAS_LAS_ESTACIONES (generado) ===\n'
    marcador_fin = '// === FIN TODAS_LAS_ESTACIONES (generado) ===\n'
    patron = re.compile(re.escape(marcador_ini) + '.*?' + re.escape(marcador_fin), re.S)
    bloque_completo = marcador_ini + bloque + marcador_fin
    if patron.search(html):
        # sacar el bloque de donde haya quedado (puede estar mal ubicado de
        # una corrida anterior) y reinsertarlo en el lugar correcto
        html = patron.sub('', html)
    # anclar en el <script> principal (no el <script src="...leaflet...">)
    anclas = list(re.finditer(r'<script>\n', html))
    if not anclas:
        raise SystemExit('No se encontro <script> principal en el HTML -- revisar a mano.')
    pos = anclas[0].end()
    return html[:pos] + bloque_completo + html[pos:]


def main():
    estaciones = leer_estaciones()
    html, especies_data = leer_especies_data()
    estacion_mas_cercana, usadas = calcular_estacion_mas_cercana(especies_data, estaciones)
    lluvia_estaciones = {k: v for k, v in estaciones.items() if k in usadas}
    todas_estaciones = resumir_todas_estaciones(estaciones)

    html_nuevo = inyectar(html, lluvia_estaciones, estacion_mas_cercana, todas_estaciones)
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html_nuevo)

    with open(HTML_PATH_SIMPLE, encoding='utf-8') as f:
        html_simple = f.read()
    html_simple_nuevo = inyectar_solo_estaciones(html_simple, todas_estaciones)
    with open(HTML_PATH_SIMPLE, 'w', encoding='utf-8') as f:
        f.write(html_simple_nuevo)
    print(f'HTML actualizado (con curvas + estaciones): {HTML_PATH}')
    print(f'HTML actualizado (solo estaciones): {HTML_PATH_SIMPLE}')

    print(f'Estaciones con serie embebida: {len(lluvia_estaciones)} (de {len(estaciones)} disponibles)')
    print(f'Estaciones en la capa del mapa: {len(todas_estaciones)}')
    print(f'Especies con estación asignada: {len(estacion_mas_cercana)}')
    lejos = [(e, v['estacion'], v['dist_km']) for e, v in estacion_mas_cercana.items() if v['dist_km'] > 80]
    if lejos:
        print('Especies con estación "más cercana" a más de 80 km (referencial, no local):')
        for e, est, dk in sorted(lejos, key=lambda x: -x[2]):
            print(f'  - {e}: {est} ({dk} km)')


if __name__ == '__main__':
    main()
