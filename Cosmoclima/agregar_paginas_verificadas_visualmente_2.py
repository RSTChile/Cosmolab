#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agregar_paginas_verificadas_visualmente_2.py -- segunda ronda de revision
visual (06-ago-2026), disparada por dos pedidos de Alexis en la misma sesion:
"añade Tal Tal" + "revisa si hay estacion en Chañaral".

Al ir a buscar Tal Tal se encontro un bug real en pagina_es_riesgosa() de
extraer_anuarios_historicos.py: una pagina con nombres pegados SOLO en las
ultimas filas se daba por segura completa, perdiendo en silencio las
primeras filas (sin nombre pegado) -- ni se extraian NI quedaban en la lista
de pendientes. Se corrigio el detector (ahora cualquier fila sin nombre
pegado marca la pagina entera como riesgosa) y se re-escaneo: de 7 paginas
riesgosas pasaron a 13. Las 6 ya revisadas antes (ver
agregar_paginas_verificadas_visualmente.py) se re-confirmaron completas al
releerlas -- no faltaba nada ahi. Las 4 nuevas (2014 pag149, 2015 pag84,
2016 pag84, 2016 pag88) se leyeron a mano aqui; 2016 pag88 no tenia ninguna
estacion de la zona (todo Los Ríos/Los Lagos/Chiloé), no aporta filas.

De paso se rescato Tal Tal de las paginas ya revisadas antes (2014 pag148,
2015 pag83 [antes "pag82" mal indexado], 2016 pag83) -- estaba en la imagen
pero se habia descartado por error (excluida de la zona sin razon real).
"""
import csv
import os

CARPETA = os.path.dirname(os.path.abspath(__file__))
CSV_OUT = os.path.join(CARPETA, 'investigacion', 'fuentes', 'precipitacion_mensual_dmc_anuario2025.csv')

# Coordenadas: reusa las ya confirmadas en rondas anteriores donde aplica;
# Tal Tal y Los Libertadores/Quilpué -- ver notas en cada una.
COORDS = {
    'Tal Tal': (-25.4, -70.483333, 37),  # catastro anuario 2014/2015/2016, DMS 25°24' 70°29'
    'El Tangue Hacienda': (-30.348333, -71.558888, 21),
    'Hurtado': (-30.266667, -70.683333, 1200),
    'Coirón Retén': (-31.903055, -70.773888, 819),
    'Combarbalá Essco': (-31.185277, -71.001388, 906),
    'Puerto Oscuro': (-31.413333, -71.574999, 142),
    'Chuchiñí Hacienda': (-31.755555, -71.058888, 391),
    'Salamanca': (-31.8, -70.916667, 570),
    'La Vega': (-32.420554, -71.039444, 241),
    'La Vega Fundo': (-32.420554, -71.039444, 241),
    'El Cobre Santa María': (-32.436666, -71.353611, 57),
    'El Cobre Fundo Sta. María': (-32.436666, -71.353611, 57),
    'Catapilco': (-32.569999, -71.305555, 85),
    'Catapilco Hacienda': (-32.569999, -71.305555, 85),
    'La Canela': (-32.706944, -71.329166, 419),
    'La Canela Fundo': (-32.706944, -71.329166, 419),
    'La Ligua': (-32.448055, -71.229999, 57),
    'Curimón': (-32.784444, -70.687777, 716),
    'Los Nichos': (-30.156111, -70.496944, 1337),
    # Catastro confirmado por busqueda directa (no en las paginas ya vistas):
    'Quilpué Esval': (-33.066667, -71.466667, 101),
    # OJO: 2955 msnm -- estacion de alta cordillera (paso Los Libertadores,
    # frontera Argentina), NO analoga al habitat costero/valle de Gyriosomus
    # pese a caer en la franja de latitud. Se guarda igual, con la altura
    # real, para que quede honesto en la base -- no se usa como "cercana" de
    # ninguna especie.
    'Los Libertadores': (-32.833333, -70.116667, 2955),
}

MESES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']


def fila(estacion, anio, pagina, **meses_valores):
    if estacion not in COORDS:
        lat = lon = alt = ''
    else:
        lat, lon, alt = COORDS[estacion]
    filas = []
    for mes_txt, val in meses_valores.items():
        mes_num = MESES.index(mes_txt) + 1
        filas.append({
            'estacion': estacion, 'codigo': '', 'lat': lat, 'lon': lon, 'alt_m': alt,
            'mes': mes_num, 'anio': anio, 'lluvia_mm': val, 'dato_real': 'si',
            'fuente': f'DMC Anuario histórico, revisión visual, ronda 2 ({pagina})',
        })
    return filas


DATOS = []

# --- Tal Tal rescatada de paginas ya vistas antes (ronda 1) ---
DATOS += fila('Tal Tal', 2014, 'anuario-2014.pdf, pagina 148', may=2.7, ago=0.6, sep=4.2, oct=2.7)
DATOS += fila('Tal Tal', 2015, 'anuario-2015.pdf, pagina 83', oct=3.4)
DATOS += fila('Tal Tal', 2016, 'anuario-2016.pdf, pagina 83', abr=0.1, may=0.2, jun=10.5, sep=0.6, oct=0.7, nov=0.8)

# --- 2016, pagina 83 (bloque completo, no solo Tal Tal -- se habia perdido) ---
DATOS += fila('El Tangue Hacienda', 2016, 'anuario-2016.pdf, pagina 83', abr=0.0, may=4.8, jun=26.7, jul=33.5, oct=1.4)
DATOS += fila('Hurtado', 2016, 'anuario-2016.pdf, pagina 83', may=18.9, jun=28.9, jul=9.6)
DATOS += fila('Coirón Retén', 2016, 'anuario-2016.pdf, pagina 83', abr=35.5, may=44.0, jun=132.0, jul=42.0, oct=16.0)
DATOS += fila('Combarbalá Essco', 2016, 'anuario-2016.pdf, pagina 83', abr=23.2, may=12.5, jun=45.7, jul=49.7, oct=0.0, nov=3.0)
DATOS += fila('Puerto Oscuro', 2016, 'anuario-2016.pdf, pagina 83', ene=1.0, abr=9.0, may=21.2, jun=57.0, jul=54.4, oct=13.0)
DATOS += fila('Chuchiñí Hacienda', 2016, 'anuario-2016.pdf, pagina 83', ene=0.7, abr=24.4, may=30.8, jun=106.7, jul=90.4, oct=31.3, dic=0.1)

# --- 2015, pagina 83 (bloque completo, no solo Tal Tal) ---
DATOS += fila('Los Nichos', 2015, 'anuario-2015.pdf, pagina 83', mar=53.5, jul=59.4, ago=41.5, oct=32.8)
DATOS += fila('Hurtado', 2015, 'anuario-2015.pdf, pagina 83', mar=43.9, jul=50.0, ago=68.6, oct=58.8)
DATOS += fila('Coirón Retén', 2015, 'anuario-2015.pdf, pagina 83', mar=14.1, may=3.7, jul=45.2, ago=180.0, sep=28.0, oct=59.0)
DATOS += fila('Combarbalá Essco', 2015, 'anuario-2015.pdf, pagina 83', mar=41.0, jul=47.5)
DATOS += fila('Puerto Oscuro', 2015, 'anuario-2015.pdf, pagina 83', ene=1.0, feb=0.2, mar=11.3, abr=0.1, jul=37.0, ago=95.3, sep=7.1, oct=36.9)
DATOS += fila('Chuchiñí Hacienda', 2015, 'anuario-2015.pdf, pagina 83', mar=16.2, jul=53.6, ago=148.6, sep=17.4, oct=54.0)

# --- 2014, pagina 149 (nueva, encontrada tras corregir el detector) ---
DATOS += fila('Salamanca', 2014, 'anuario-2014.pdf, pagina 149', may=7.0, jun=83.6, jul=1.2, ago=7.8)
DATOS += fila('La Vega', 2014, 'anuario-2014.pdf, pagina 149', may=10.5, jul=9.5, ago=48.4, sep=18.4)
DATOS += fila('El Cobre Santa María', 2014, 'anuario-2014.pdf, pagina 149', may=8.5, jun=105.5, jul=18.5, ago=21.2, sep=24.5)
DATOS += fila('Catapilco', 2014, 'anuario-2014.pdf, pagina 149', may=5.0, jun=111.2, jul=28.1, ago=33.2, sep=32.1)
DATOS += fila('La Canela', 2014, 'anuario-2014.pdf, pagina 149', may=4.1, jun=142.2, jul=54.0, ago=64.0, sep=55.0)
DATOS += fila('Los Libertadores', 2014, 'anuario-2014.pdf, pagina 149', ene=2.0, feb=0.8, mar=0.8, abr=0.2, may=2.1, jun=0.9, ago=0.9, sep=1.4, oct=0.5, nov=0.4, dic=0.2)

# --- 2015, pagina 84 (nueva) ---
DATOS += fila('La Vega', 2015, 'anuario-2015.pdf, pagina 84', jul=44.3, ago=107.0, sep=24.2, oct=25.6)
DATOS += fila('El Cobre Santa María', 2015, 'anuario-2015.pdf, pagina 84', mar=3.5, jul=66.2, ago=68.2, sep=42.4, oct=50.7)
DATOS += fila('Catapilco', 2015, 'anuario-2015.pdf, pagina 84', mar=3.2, abr=3.2, may=0.0, jul=73.3, ago=177.7, sep=37.2, oct=57.3)
DATOS += fila('La Canela', 2015, 'anuario-2015.pdf, pagina 84', mar=0.2, jul=62.2, ago=132.0, sep=41.0, oct=79.0)
DATOS += fila('La Ligua', 2015, 'anuario-2015.pdf, pagina 84', mar=5.2, jul=64.0, ago=148.4, sep=39.8, oct=45.9)
DATOS += fila('Curimón', 2015, 'anuario-2015.pdf, pagina 84', jul=40.8, ago=95.4, oct=19.5)
DATOS += fila('Quilpué Esval', 2015, 'anuario-2015.pdf, pagina 84', mar=3.0, may=1.0, jul=64.5, ago=162.0, sep=78.0, oct=76.5)

# --- 2016, pagina 84 (nueva) ---
DATOS += fila('La Vega Fundo', 2016, 'anuario-2016.pdf, pagina 84', abr=39.6, may=8.5, jun=87.3, jul=60.7, oct=48.8)
DATOS += fila('El Cobre Fundo Sta. María', 2016, 'anuario-2016.pdf, pagina 84', ene=6.0, abr=106.5, may=25.0, jun=76.0, jul=136.5, oct=12.5)
DATOS += fila('La Canela Fundo', 2016, 'anuario-2016.pdf, pagina 84', ene=4.3, abr=67.1, may=23.2, jun=75.0, jul=123.2, oct=28.0, dic=7.3)
DATOS += fila('Los Libertadores', 2016, 'anuario-2016.pdf, pagina 84', ene=0.6, feb=0.9, mar=1.0, abr=5.2, may=0.6, jun=0.9, jul=0.8, ago=0.7, sep=0.7, oct=0.7, nov=0.8, dic=1.6)
DATOS += fila('Catapilco Hacienda', 2016, 'anuario-2016.pdf, pagina 84', ene=8.9, abr=95.6, may=22.8, jun=81.6, jul=130.1, oct=12.8)
DATOS += fila('Quilpué Esval', 2016, 'anuario-2016.pdf, pagina 84', ene=3.0, abr=33.2, may=40.0, jun=69.0, jul=76.5, oct=16.0, dic=18.5)


def main():
    faltan = sorted(set(d['estacion'] for d in DATOS if d['lat'] == ''))
    if faltan:
        print('AVISO: sin coordenada todavia para:', faltan)
    with open(CSV_OUT, encoding='utf-8') as f:
        existentes = list(csv.DictReader(f))
    campos = ['estacion', 'codigo', 'lat', 'lon', 'alt_m', 'mes', 'anio', 'lluvia_mm', 'dato_real', 'fuente']
    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(existentes)
        w.writerows(DATOS)
    print(f'Filas nuevas agregadas: {len(DATOS)}')
    print(f'Total filas en CSV: {len(existentes) + len(DATOS)}')


if __name__ == '__main__':
    main()
