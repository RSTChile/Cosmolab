#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agregar_paginas_verificadas_visualmente.py -- las 7 paginas que
extraer_anuarios_historicos.py marco como "riesgosas" (nombres de estacion en
bloque separado de los datos, probable OCR de tabla escaneada) se leyeron a
mano, con imagen renderizada, verificando visualmente que cada fila de datos
calce con su nombre real -- confirmado en 2024 que el ORDEN de texto crudo NO
coincidia con el orden visual real (justo el riesgo que se queria evitar: el
texto crudo traia "La Canela Fundo, Curimon..." primero, pero la tabla real
empezaba por "Visviri, Los Nichos, Combarbala...").

Ya se corrio una vez (05/06-ago-2026) -- se deja como registro de trazabilidad
de exactamente que 298 valores mensuales vinieron de lectura visual y no de
extraccion automatica de texto. No es idempotente si se corre de nuevo sin
antes borrar esas filas del CSV (quedarian duplicadas).
"""
import csv
import os

CARPETA = os.path.dirname(os.path.abspath(__file__))
CSV_OUT = os.path.join(CARPETA, 'investigacion', 'fuentes', 'precipitacion_mensual_dmc_anuario2025.csv')

COORDS = {
    'Los Nichos Fundo': (-30.156111, -70.496944, 1337),
    'El Tangue Hacienda': (-30.348333, -71.558888, 21),
    'Combarbalá Essco': (-31.185277, -71.001388, 906),
    'Puerto Oscuro': (-31.413333, -71.574999, 142),
    'Chuchiñí': (-31.755555, -71.058888, 391),
    'Huaquén Hacienda': (-32.284166, -71.459722, 93),
    'El Trapiche Longotoma': (-32.321943, -71.290277, 60),
    'Casas de Alicahue': (-32.351111, -70.783610, 642),
    'La Vega Fundo': (-32.420554, -71.039444, 241),
    'La Ligua Esval': (-32.448055, -71.229999, 57),
    'Catapilco Hacienda': (-32.569999, -71.305555, 85),
    'La Canela Fundo': (-32.706944, -71.329166, 419),
    'Curimón Escuela Agrícola': (-32.784444, -70.687777, 716),
    'Hurtado': (-30.266667, -70.683333, 1200),  # DMS del catastro 1988, menos preciso
}

MESES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']


def fila(estacion, anio, pagina, **meses_valores):
    lat, lon, alt = COORDS[estacion]
    filas = []
    for mes_txt, val in meses_valores.items():
        mes_num = MESES.index(mes_txt) + 1
        filas.append({
            'estacion': estacion, 'codigo': '', 'lat': lat, 'lon': lon, 'alt_m': alt,
            'mes': mes_num, 'anio': anio, 'lluvia_mm': val, 'dato_real': 'si',
            'fuente': f'DMC Anuario histórico, revisión visual ({pagina})',
        })
    return filas


DATOS = []

# --- 2014, pagina 148 ---
DATOS += fila('Los Nichos Fundo', 2014, 'anuario-2014.pdf, pagina 148', jun=53.0)
DATOS += fila('Hurtado', 2014, 'anuario-2014.pdf, pagina 148', jun=37.8, jul=4.5, ago=2.5)
DATOS += fila('El Tangue Hacienda', 2014, 'anuario-2014.pdf, pagina 148', ene=0.0, mar=0.0, may=0.0, jun=39.3, ago=2.5, sep=5.2, oct=0.0)
DATOS += fila('Combarbalá Essco', 2014, 'anuario-2014.pdf, pagina 148', may=1.0, jun=67.5, ago=3.5, sep=12.2)
DATOS += fila('Puerto Oscuro', 2014, 'anuario-2014.pdf, pagina 148', may=1.0, jun=66.2, jul=2.6, ago=3.1, sep=12.5, oct=1.5)
DATOS += fila('Chuchiñí', 2014, 'anuario-2014.pdf, pagina 148', may=5.4, jun=79.7, jul=2.3, ago=7.9, sep=14.1)

# --- 2018, pagina 79 ---
DATOS += fila('Los Nichos Fundo', 2018, 'anuario-2018.pdf, pagina 79', jun=26.8, jul=0.0, oct=2.3)
DATOS += fila('El Tangue Hacienda', 2018, 'anuario-2018.pdf, pagina 79', ene=0.0, feb=0.0, may=40.3, jun=34.7)
DATOS += fila('Combarbalá Essco', 2018, 'anuario-2018.pdf, pagina 79', jun=57.0, jul=21.4, oct=0.0)
DATOS += fila('Puerto Oscuro', 2018, 'anuario-2018.pdf, pagina 79', may=21.6, jun=37.3, jul=45.4, ago=4.0)
DATOS += fila('Chuchiñí', 2018, 'anuario-2018.pdf, pagina 79', may=5.7, jun=65.4, jul=31.5, ago=4.2, sep=8.3)
DATOS += fila('Huaquén Hacienda', 2018, 'anuario-2018.pdf, pagina 79', may=22.6, jun=75.3, jul=49.2, sep=7.8)
DATOS += fila('El Trapiche Longotoma', 2018, 'anuario-2018.pdf, pagina 79', may=16.6, jun=86.0, jul=63.7, ago=13.5, sep=12.5)
DATOS += fila('Casas de Alicahue', 2018, 'anuario-2018.pdf, pagina 79', may=13.5, jun=38.7, jul=27.8, ago=12.0, sep=18.0)
DATOS += fila('La Vega Fundo', 2018, 'anuario-2018.pdf, pagina 79', may=5.0, jun=34.7, jul=3.5, ago=8.5, sep=14.5)
DATOS += fila('La Ligua Esval', 2018, 'anuario-2018.pdf, pagina 79', jun=73.3, jul=53.9, ago=13.6, sep=16.4, oct=1.0)
DATOS += fila('Catapilco Hacienda', 2018, 'anuario-2018.pdf, pagina 79', may=19.4, jun=64.4, jul=43.6, ago=11.5, sep=14.7, oct=3.5)
DATOS += fila('La Canela Fundo', 2018, 'anuario-2018.pdf, pagina 79', may=23.0, jun=105.6, jul=71.9, ago=4.5, sep=15.6, oct=1.2)
# --- 2018, pagina 80 ---
DATOS += fila('Curimón Escuela Agrícola', 2018, 'anuario-2018.pdf, pagina 80', may=3.8, jun=22.0, jul=53.0, ago=6.0, sep=18.3)

# --- 2019, pagina 106 ---
DATOS += fila('Los Nichos Fundo', 2019, 'anuario-2019.pdf, pagina 106', abr=0.0, may=19.6, sep=2.5)
DATOS += fila('El Tangue Hacienda', 2019, 'anuario-2019.pdf, pagina 106', mar=0.0, may=3.2, jun=0.0, dic=0.0)
DATOS += fila('Combarbalá Essco', 2019, 'anuario-2019.pdf, pagina 106', abr=0.0, may=4.0, jun=10.0)
DATOS += fila('Puerto Oscuro', 2019, 'anuario-2019.pdf, pagina 106', mar=1.4, abr=3.0, may=3.5, jun=22.2)
DATOS += fila('Chuchiñí', 2019, 'anuario-2019.pdf, pagina 106', may=3.0, jun=14.7)
DATOS += fila('Huaquén Hacienda', 2019, 'anuario-2019.pdf, pagina 106', mar=0.7, abr=2.3)
DATOS += fila('El Trapiche Longotoma', 2019, 'anuario-2019.pdf, pagina 106', may=0.8, jun=35.1, jul=9.2)
DATOS += fila('Casas de Alicahue', 2019, 'anuario-2019.pdf, pagina 106', may=3.6, jun=15.8, jul=2.9, sep=6.0)
DATOS += fila('La Ligua Esval', 2019, 'anuario-2019.pdf, pagina 106', abr=0.6, may=1.7, jun=29.2, jul=7.8)
DATOS += fila('Catapilco Hacienda', 2019, 'anuario-2019.pdf, pagina 106', abr=2.4, may=2.5, jun=38.6, jul=8.3)
DATOS += fila('La Canela Fundo', 2019, 'anuario-2019.pdf, pagina 106', mar=0.2, abr=1.5, may=12.9, jun=54.7, jul=10.3)
DATOS += fila('Curimón Escuela Agrícola', 2019, 'anuario-2019.pdf, pagina 106', may=2.0, jun=8.6, jul=8.0, sep=2.1)

# --- 2020, pagina 106 ---
DATOS += fila('Los Nichos Fundo', 2020, 'anuario-2020.pdf, pagina 106', may=3.1, jun=60.2, jul=0.2, ago=0.0)
DATOS += fila('El Tangue Hacienda', 2020, 'anuario-2020.pdf, pagina 106', jun=75.7, jul=6.0, ago=7.1, sep=0.0)
DATOS += fila('Combarbalá Essco', 2020, 'anuario-2020.pdf, pagina 106', jun=80.0, jul=14.2, ago=1.0)
DATOS += fila('Puerto Oscuro', 2020, 'anuario-2020.pdf, pagina 106', jun=113.6, jul=16.0, ago=8.1, sep=1.0, oct=1.6, nov=2.3)
DATOS += fila('Chuchiñí', 2020, 'anuario-2020.pdf, pagina 106', jun=136.2, jul=28.6, ago=2.2)
DATOS += fila('Huaquén Hacienda', 2020, 'anuario-2020.pdf, pagina 106', abr=3.0, jun=158.6, jul=64.3, ago=17.8)
DATOS += fila('El Trapiche Longotoma', 2020, 'anuario-2020.pdf, pagina 106', jun=79.4, jul=55.2, ago=10.5)
DATOS += fila('Casas de Alicahue', 2020, 'anuario-2020.pdf, pagina 106', abr=2.0, jun=108.3, jul=29.3, ago=3.7)
DATOS += fila('La Ligua Esval', 2020, 'anuario-2020.pdf, pagina 106', jun=145.9, jul=70.8)
DATOS += fila('Catapilco Hacienda', 2020, 'anuario-2020.pdf, pagina 106', may=0.1, jun=132.3, jul=95.1, ago=7.2, sep=0.4)
DATOS += fila('La Canela Fundo', 2020, 'anuario-2020.pdf, pagina 106', jun=156.0, jul=74.6, ago=18.1, sep=1.5)
DATOS += fila('Curimón Escuela Agrícola', 2020, 'anuario-2020.pdf, pagina 106', abr=1.0, jun=91.0, jul=38.0)

# --- 2023, pagina 106 ---
DATOS += fila('Los Nichos Fundo', 2023, 'anuario-2023.pdf, pagina 106', feb=13.6, mar=13.6, may=0.1, jul=14.0, ago=0.7)
DATOS += fila('El Tangue Hacienda', 2023, 'anuario-2023.pdf, pagina 106', jul=28.0)
DATOS += fila('Combarbalá Essco', 2023, 'anuario-2023.pdf, pagina 106', jul=24.5, sep=2.4, nov=5.0)
DATOS += fila('Puerto Oscuro', 2023, 'anuario-2023.pdf, pagina 106', ene=0.1, abr=0.7, jun=2.7, jul=32.7, ago=7.5, sep=7.8, oct=0.1, nov=6.2)
DATOS += fila('Chuchiñí', 2023, 'anuario-2023.pdf, pagina 106', jun=0.7, jul=20.6, ago=9.3, sep=7.7, nov=12.0)
DATOS += fila('Huaquén Hacienda', 2023, 'anuario-2023.pdf, pagina 106', abr=0.8, jun=11.1, jul=49.4, ago=73.2, sep=26.2)
DATOS += fila('El Trapiche Longotoma', 2023, 'anuario-2023.pdf, pagina 106', abr=3.5, may=2.1, jun=6.9, jul=48.1, ago=83.4, sep=32.4, oct=11.7)
DATOS += fila('Casas de Alicahue', 2023, 'anuario-2023.pdf, pagina 106', jun=6.6, jul=15.4, ago=31.9, sep=23.7, nov=13.5)
DATOS += fila('La Ligua Esval', 2023, 'anuario-2023.pdf, pagina 106', abr=9.0, may=1.5, jun=9.1, jul=48.3, ago=87.9, sep=26.8, nov=11.7)
DATOS += fila('Catapilco Hacienda', 2023, 'anuario-2023.pdf, pagina 106', abr=7.4, may=4.6, jun=17.0, jul=52.9, ago=96.9, sep=40.8, oct=0.9, nov=12.6)
DATOS += fila('La Canela Fundo', 2023, 'anuario-2023.pdf, pagina 106', ene=142.5, jun=156.0, jul=74.6, ago=18.1, sep=1.5)
DATOS += fila('Curimón Escuela Agrícola', 2023, 'anuario-2023.pdf, pagina 106', jun=23.5, jul=38.0, ago=81.0, sep=24.6, nov=37.0)

# --- 2024, pagina 254 ---
DATOS += fila('Los Nichos Fundo', 2024, 'anuario-2024.pdf, pagina 254', abr=19.9, may=52.7, jun=30.0, ago=61.0)
DATOS += fila('Combarbalá Essco', 2024, 'anuario-2024.pdf, pagina 254', mar=1.5, abr=32.5, may=154.4, jul=55.6, sep=1.4)
DATOS += fila('Puerto Oscuro', 2024, 'anuario-2024.pdf, pagina 254', ene=0.8, feb=0.9, mar=0.6, may=53.5, jun=103.3, jul=0.1, ago=45.8, sep=2.0, oct=4.0)
DATOS += fila('Chuchiñí', 2024, 'anuario-2024.pdf, pagina 254', abr=0.4, may=72.1, jun=187.6, ago=78.9, sep=1.1)
DATOS += fila('Huaquén Hacienda', 2024, 'anuario-2024.pdf, pagina 254', may=75.6, jun=174.1, ago=35.9, sep=42.2)
DATOS += fila('El Trapiche Longotoma', 2024, 'anuario-2024.pdf, pagina 254', may=89.2, jun=229.4, ago=85.3)
DATOS += fila('Casas de Alicahue', 2024, 'anuario-2024.pdf, pagina 254', feb=2.3, abr=15.2, may=93.1, jun=164.6, ago=89.0, oct=5.0)
DATOS += fila('La Ligua Esval', 2024, 'anuario-2024.pdf, pagina 254', feb=2.0, may=76.1, jun=234.6, jul=1.2, ago=131.5)
DATOS += fila('Catapilco Hacienda', 2024, 'anuario-2024.pdf, pagina 254', feb=3.2, abr=0.3, may=36.5, jun=216.6, jul=1.3, ago=155.7, sep=1.4, oct=1.2, nov=0.7)
DATOS += fila('La Canela Fundo', 2024, 'anuario-2024.pdf, pagina 254', feb=2.1, may=74.8, jun=365.3, jul=1.0, ago=168.0, sep=3.2, oct=7.5)
DATOS += fila('Curimón Escuela Agrícola', 2024, 'anuario-2024.pdf, pagina 254', may=128.0, jun=144.7, ago=68.0, oct=12.0)


def main():
    with open(CSV_OUT, encoding='utf-8') as f:
        existentes = list(csv.DictReader(f))
    campos = ['estacion', 'codigo', 'lat', 'lon', 'alt_m', 'mes', 'anio', 'lluvia_mm', 'dato_real', 'fuente']
    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(existentes)
        w.writerows(DATOS)
    print(f'Filas verificadas visualmente agregadas: {len(DATOS)}')
    print(f'Total filas en CSV: {len(existentes) + len(DATOS)}')


if __name__ == '__main__':
    main()
