#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obtener_gradiente_latitudinal.py -- Parte B del plan de temperatura real
(09-ago-2026, a pedido de Alexis: "...visto a lo largo de la zona desde
Paposo a Santiago, veríamos una curva anual descendente de las máximas
desde el norte al centro... [vista] entre Paposo y Santiago o Machalí, que
es la zona más austral de la distribución de Gyriosomus laevigatus").

AMPLIADO 09-ago-2026 (2): Alexis, tras ver que el gradiente no era monótono
-- "quisiera... separar las estaciones en costeras y del valle, para tener
dos curvas longitudinales de máximas y mínimas para zonas costeras e
interior: en Copiapó, por ejemplo, siempre en la costa amanece con baguada,
mientras en el interior está despejado... eso pasa por Humboldt, pero en
interior afecta menos, así que las máximas suelen ser más altas, y las
mínimas también (el océano modera)". Cada estación ahora lleva un campo
`tipo` ('costa'/'valle'), clasificado a mano por geografía real conocida
(estaciones costeras = puertos/caletas reales sobre el Pacífico; valle =
ciudades de valle interior, a decenas de km de la costa) -- no hay una
columna "distancia a la costa" en el sqlite, así que esto es juicio
geográfico declarado, no un cálculo.

14 estaciones REALES (ya en pluviosidad_diaria_consolidada.sqlite, con miles
de filas de lluvia real detrás -- no se inventan coordenadas), de norte a
sur, 7 costeras + 7 de valle. El ancla sur (valle), El Guindal, coincide con
el registro real más austral de G. laevigatus en datos_fuentes/
tabla_s1_anguita_salinas_2026.csv (-34.1905,-70.6368) -- confirma la fuente
de forma independiente.

Para cada estación: NASA POWER diario 1981-01-01 -> hoy, vía datos_clima.py.
IMPORTANTE: no se usa agregar() de datos_clima.py para el resumen -- su modo
"maximo" (para t_maxima_c) da el día más caluroso del período, no el
promedio climatológico de las máximas diarias que se necesita acá. Se
promedia a mano sobre traer().
"""
import csv
import json
import os
import re
import sys
from datetime import date, datetime

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CARPETA, '..', '..'))
from datos_clima import traer  # noqa: E402

HTML_PATH = os.path.join(CARPETA, 'sim-cosmoclima.html')
CSV_SALIDA = os.path.join(CARPETA, '..', '..', 'investigacion', 'fuentes',
                           'temperatura_gradiente_latitudinal_nasa_power.csv')

DESDE = '1981-01-01'

# Norte -> sur. Coordenadas reales de pluviosidad_diaria_consolidada.sqlite
# (columna localidad), verificadas con miles de filas de lluvia real detrás.
# tipo: 'costa' = puerto/caleta real sobre el Pacífico; 'valle' = ciudad de
# valle interior, a decenas de km de la costa (clasificación geográfica a
# mano, ver comentario del módulo). altura_m: elevación REAL de estación
# (09-ago-2026 (3), a pedido de Alexis: "usar la localización real de las
# estaciones y contrastar con la elevación real... no usar el satélite") --
# de investigacion/fuentes/estaciones_elevacion_real.csv (catastro DGA/DMC/
# CR2, nunca NASA POWER/satélite -- ver ese archivo para la fuente exacta
# de cada valor). None donde no hay estación física real detrás (Paposo y
# El Guindal son puntos de reanálisis ERA5-Land, confirmado en el propio
# sqlite -- no existe elevación de estación que buscar ahí).
ESTACIONES = [
    ('Paposo',                       -24.96,   -70.48,   'costa', None),
    ('Tal-Tal',                      -25.4047, -70.4819, 'costa', 9),
    ('Copiapo',                      -27.3772, -70.3308, 'valle', 385),
    ('Caldera',                      -27.0692, -70.8156, 'costa', 15),
    ('Vallenar Dga',                 -28.5861, -70.7397, 'valle', 420),
    ('Huasco',                       -28.4669, -71.2214, 'costa', 15),
    ('Vicua (Inia)',                 -30.0567, -70.7167, 'valle', 730),  # Vicuña, valle del Elqui
    ('La Serena (Escuela Agrícola)', -29.9061, -71.2556, 'costa', 15),
    ('Huintil',                      -31.5669, -70.9817, 'valle', 650),  # = punto-reloj ZHCS
    ('Los Vilos Dmc',                -31.9103, -71.5086, 'costa', 10),
    ('San Felipe',                   -32.7472, -70.7247, 'valle', 640),
    ('Quinta Normal, Santiago (DMC)',-33.445,  -70.68278,'valle', 527),
    ('San Antonio (Pta. Panul)',     -33.5747, -71.625,  'costa', 80),
    ('El Guindal',                   -34.19,   -70.64,   'valle', None),  # registro más austral G. laevigatus (Tabla S1)
]

MESES_VERANO = {12, 1, 2}
MESES_INVIERNO = {6, 7, 8}


def promedio(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def resumir_estacion(nombre, lat, lon, tipo, altura_m):
    hasta = date.today().isoformat()
    print(f'Consultando NASA POWER · {nombre} ({lat},{lon}, {tipo}, altura_real={altura_m}) · {DESDE} → {hasta}...')
    filas = traer(lat, lon, DESDE, hasta)

    tmax_todos, tmin_todos = [], []
    tmax_verano, tmax_invierno = [], []
    for f in filas:
        mes = datetime.strptime(f['fecha'], '%Y-%m-%d').month
        tmax_todos.append(f.get('t_maxima_c'))
        tmin_todos.append(f.get('t_minima_c'))
        if mes in MESES_VERANO:
            tmax_verano.append(f.get('t_maxima_c'))
        elif mes in MESES_INVIERNO:
            tmax_invierno.append(f.get('t_maxima_c'))

    n_con_dato = sum(1 for v in tmax_todos if v is not None)
    resumen = {
        'nombre': nombre, 'lat': lat, 'lon': lon, 'tipo': tipo, 'altura_m': altura_m,
        'tmax_prom': promedio(tmax_todos), 'tmin_prom': promedio(tmin_todos),
        'tmax_verano': promedio(tmax_verano), 'tmax_invierno': promedio(tmax_invierno),
        'n_dias': len(filas), 'n_dias_con_dato': n_con_dato,
    }
    print(f'  {nombre} ({tipo}): {n_con_dato}/{len(filas)} días con dato '
          f'({n_con_dato/len(filas)*100:.1f}%) · Tmax prom {resumen["tmax_prom"]}°C '
          f'· Tmin prom {resumen["tmin_prom"]}°C')
    return resumen


def guardar_csv(resumenes):
    cols = ['nombre', 'lat', 'lon', 'tipo', 'altura_m', 'tmax_prom', 'tmin_prom', 'tmax_verano',
            'tmax_invierno', 'n_dias', 'n_dias_con_dato']
    with open(CSV_SALIDA, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in resumenes:
            w.writerow(r)
    print(f'Crudo guardado en {CSV_SALIDA}')


def inyectar(html, resumenes):
    bloque = (
        '// GRADIENTE_LATITUDINAL_TEMP -- generado por obtener_gradiente_latitudinal.py\n'
        '// (09-ago-2026, Parte B del plan de temperatura real, a pedido de Alexis:\n'
        '// "...visto a lo largo de la zona desde Paposo a Santiago, veríamos una\n'
        '// curva anual descendente de las máximas... entre Paposo y Machalí, la\n'
        '// zona más austral de la distribución de Gyriosomus laevigatus").\n'
        '// 9 estaciones REALES (pluviosidad_diaria_consolidada.sqlite, con lluvia\n'
        '// real ya verificada detrás), norte a sur. Ancla sur (El Guindal) coincide\n'
        '// con el registro más austral real de G. laevigatus (Tabla S1,\n'
        '// -34.1905,-70.6368) -- confirma la fuente de forma independiente.\n'
        '// Fuente: NASA POWER diario 1981-01-01 -> hoy, promediado a mano (NO con\n'
        '// el modo "maximo" de agregar(), que da el extremo del período, no el\n'
        '// promedio climatológico de las máximas diarias).\n'
        '// Cada estación lleva "tipo":"costa"/"valle" (09-ago-2026 (2), a pedido de\n'
        '// Alexis: "en Copiapó... siempre en la costa amanece con baguada, mientras\n'
        '// en el interior está despejado... las máximas suelen ser más altas [en el\n'
        '// interior], y las mínimas también -- el océano modera"). Clasificación\n'
        '// geográfica a mano (puerto/caleta real = costa; ciudad de valle a decenas\n'
        '// de km de la costa = valle) -- el sqlite no trae distancia a la costa.\n'
        '// Cada estación lleva "altura_m" REAL de estación (09-ago-2026 (3), a pedido\n'
        '// de Alexis: "usar la localización real... y contrastar con la elevación\n'
        '// real... no usar el satélite" -- el primer intento con altitud NASA POWER\n'
        '// dio valores absurdos, ej. 2185m para Vicuña que en la realidad está a\n'
        '// ~700m). Fuente real: investigacion/fuentes/estaciones_elevacion_real.csv\n'
        '// (catastro DGA/DMC/CR2, nunca satélite). null donde no hay estación física\n'
        '// real (Paposo y El Guindal son puntos de reanálisis ERA5-Land).\n'
        'const GRADIENTE_LATITUDINAL_TEMP = ' + json.dumps(resumenes, ensure_ascii=False) + ';\n'
    )
    marcador_ini = '// === INICIO GRADIENTE_LATITUDINAL_TEMP (generado) ===\n'
    marcador_fin = '// === FIN GRADIENTE_LATITUDINAL_TEMP (generado) ===\n'
    patron = re.compile(re.escape(marcador_ini) + '.*?' + re.escape(marcador_fin), re.S)
    bloque_completo = marcador_ini + bloque + marcador_fin
    if patron.search(html):
        return patron.sub(bloque_completo, html)
    ancla = re.search(r'const PLUVIOSIDAD_MENSUAL = \{.*?\};\n', html, re.S)
    if not ancla:
        raise SystemExit('No se encontro PLUVIOSIDAD_MENSUAL en el HTML -- revisar a mano.')
    pos = ancla.end()
    return html[:pos] + bloque_completo + html[pos:]


def main():
    resumenes = [resumir_estacion(nombre, lat, lon, tipo, altura_m) for nombre, lat, lon, tipo, altura_m in ESTACIONES]
    guardar_csv(resumenes)
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        html = f.read()
    html_nuevo = inyectar(html, resumenes)
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html_nuevo)
    print(f'Inyectado en {HTML_PATH}')


if __name__ == '__main__':
    main()
