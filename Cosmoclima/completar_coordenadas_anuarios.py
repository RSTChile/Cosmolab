#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
completar_coordenadas_anuarios.py -- recorre TODOS los anuarios de nuevo, esta
vez solo para juntar el "CATASTRO DE ESTACIONES" (nombre + lat + lon + altura)
de cada uno -- viene en 3 formatos distintos segun la decada:
  (a) DMS con simbolo de grado, ej. "29°54' 71°12'" (anuarios ~1980-2010)
  (b) decimal con puntos de mas insertados cada 3 digitos por un artefacto de
      generacion del PDF, ej. "-30.156.111" que en realidad es -30.156111
      (anuarios ~2018)
  (c) decimal limpio, ej. "-30.15611" (anuarios ~2023 en adelante)
Con ese catalogo completo, rellena lat/lon/alt_m en las filas del CSV
consolidado que hayan quedado vacias (las que vinieron del extractor
automatico de datos, que no traia coordenadas).
"""
import csv
import glob
import os
import re
import unicodedata

import pypdf

CARPETA = os.path.dirname(os.path.abspath(__file__))
CARPETA_PDFS = os.path.join(CARPETA, 'datos', 'anuarios meteorológicos')
CSV_OUT = os.path.join(CARPETA, 'investigacion', 'fuentes', 'precipitacion_mensual_dmc_anuario2025.csv')


def normalizar(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', s).strip().lower()


def dms_a_decimal(grados, minutos):
    return -(float(grados) + float(minutos) / 60.0)


def dotted_a_decimal(token, es_lat):
    """'-30.156.111' -> -30.156111 (2 digitos de parte entera, siempre)."""
    signo = -1 if token.strip().startswith('-') else 1
    digitos = re.sub(r'[^\d]', '', token)
    if len(digitos) < 3:
        return None
    entero = digitos[:2]
    decimales = digitos[2:]
    try:
        return signo * float(f'{entero}.{decimales}')
    except ValueError:
        return None


PATRON_DMS = re.compile(
    r'^\s*(\d+)\s+([A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ .\-\'/]+?)\s+'
    r'([IVX]+|METROPOLITANA|RM)\s+'
    r'(\d{1,2})[°º](\d{1,2})\'\s+(\d{1,3})[°º](\d{1,2})\'\s+(-?\d+)\s*$'
)
PATRON_DOTTED = re.compile(
    r'^\s*(\d+)\s+([A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ .\-\'/]+?)\s+'
    r'([IVX]+|METROPOLITANA|RM|REGI[ÓO]N.*?)\s+'
    r'(-[\d.]{5,})\s+(-[\d.]{5,})\s+(-?\d+)\s*$'
)


def parsear_catalogo(texto):
    filas = {}
    for linea in texto.split('\n'):
        linea = linea.strip()
        if not linea or not re.match(r'^\d', linea):
            continue
        m = PATRON_DMS.match(linea)
        if m:
            nombre = ' '.join(m.group(2).split())
            lat = dms_a_decimal(m.group(4), m.group(5))
            lon = dms_a_decimal(m.group(6), m.group(7))
            alt = m.group(8)
            filas[normalizar(nombre)] = (lat, lon, alt, 'DMS')
            continue
        m = PATRON_DOTTED.match(linea)
        if m:
            nombre = ' '.join(m.group(2).split())
            lat_txt, lon_txt = m.group(4), m.group(5)
            if lat_txt.count('.') >= 2:
                lat = dotted_a_decimal(lat_txt, True)
                lon = dotted_a_decimal(lon_txt, False)
            else:
                try:
                    lat, lon = float(lat_txt), float(lon_txt)
                except ValueError:
                    continue
            if lat is None or lon is None:
                continue
            alt = m.group(6)
            filas[normalizar(nombre)] = (lat, lon, alt, 'decimal')
    return filas


def main():
    catalogo = {}
    pdfs = sorted(glob.glob(os.path.join(CARPETA_PDFS, 'anuario-*.pdf')))
    for path in pdfs:
        try:
            r = pypdf.PdfReader(path)
        except Exception:
            continue
        for pagina in r.pages:
            t = pagina.extract_text() or ''
            if 'CATASTRO' not in t.upper() and 'NOMBRE' not in t.upper():
                continue
            nuevas = parsear_catalogo(t)
            for k, v in nuevas.items():
                if k not in catalogo:
                    catalogo[k] = v

    print(f'Catalogo construido: {len(catalogo)} estaciones distintas con coordenadas')

    with open(CSV_OUT, encoding='utf-8') as f:
        filas = list(csv.DictReader(f))

    rellenadas = 0
    sin_match = set()
    for row in filas:
        if row['lat'] and row['lon']:
            continue
        nombre_norm = normalizar(row['estacion'])
        coords = catalogo.get(nombre_norm)
        if not coords:
            # probar sin la ultima palabra (ej. "COPIAPO CHAMONATE" vs variantes)
            partes = nombre_norm.split()
            for k in range(len(partes) - 1, 0, -1):
                sub = ' '.join(partes[:k])
                candidatos = [v for key, v in catalogo.items() if key.startswith(sub)]
                if len(candidatos) == 1:
                    coords = candidatos[0]
                    break
        if coords:
            row['lat'], row['lon'], row['alt_m'] = coords[0], coords[1], coords[2]
            rellenadas += 1
        else:
            sin_match.add(row['estacion'])

    campos = ['estacion', 'codigo', 'lat', 'lon', 'alt_m', 'mes', 'anio', 'lluvia_mm', 'dato_real', 'fuente']
    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)

    print(f'Filas con coordenada rellenada: {rellenadas}')
    print(f'Estaciones sin match de coordenada ({len(sin_match)}):')
    for n in sorted(sin_match):
        print('  -', n)


if __name__ == '__main__':
    main()
