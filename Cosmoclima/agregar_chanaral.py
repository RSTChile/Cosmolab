#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agregar_chanaral.py -- Alexis sospechaba que debia haber una estacion en
Chañaral (al lado de Pan de Azucar) y pidio poner TODAS las estaciones en el
mapa para chequear visualmente. Al hacerlo se confirmo que faltaba: Chañaral
SI tiene dato real en el catastro (formato seguro, "NOMBRE ESTACION:") pero
se habia extraido y solo impreso para inspeccion, nunca guardado en el CSV.
Este script toma esas filas (via extraer_anuarios_historicos.procesar_pdf,
ahora con "chanaral" en ESTACIONES_ZONA) y las agrega, con coordenadas del
catastro (26°20'S 70°37'W, confirmado en anuario-1979/1980/1982/1983/1986/
1991/2002).
"""
import csv
import importlib
import os
import unicodedata
import re
import glob

import extraer_anuarios_historicos as m
importlib.reload(m)

CARPETA = os.path.dirname(os.path.abspath(__file__))
CSV_OUT = os.path.join(CARPETA, 'investigacion', 'fuentes', 'precipitacion_mensual_dmc_anuario2025.csv')
COORD_CHANARAL = (-26.333333, -70.616667, 9)


def normalizar(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', s).strip().lower()


def main():
    pdfs = sorted(glob.glob(os.path.join(CARPETA, 'datos', 'anuarios meteorológicos', 'anuario-*.pdf')))
    filas_chanaral = []
    for path in pdfs:
        filas, _ = m.procesar_pdf(path)
        for f in filas:
            if 'chanaral' in normalizar(f[0]):
                filas_chanaral.append(f)

    with open(CSV_OUT, encoding='utf-8') as f:
        existentes = list(csv.DictReader(f))
    ya = {(r['estacion'], r['anio'], r['mes']) for r in existentes}

    nuevas = []
    for nombre, mes, anio, val, fuente_pag in filas_chanaral:
        mes_num = m.MESES.index(mes) + 1
        clave = (nombre, str(anio), str(mes_num))
        if clave in ya:
            continue
        lat, lon, alt = COORD_CHANARAL
        nuevas.append({
            'estacion': nombre, 'codigo': '', 'lat': lat, 'lon': lon, 'alt_m': alt,
            'mes': mes_num, 'anio': anio, 'lluvia_mm': val, 'dato_real': 'si',
            'fuente': f'DMC Anuario histórico ({fuente_pag})',
        })

    campos = ['estacion', 'codigo', 'lat', 'lon', 'alt_m', 'mes', 'anio', 'lluvia_mm', 'dato_real', 'fuente']
    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(existentes)
        w.writerows(nuevas)

    print(f'Filas de Chañaral encontradas: {len(filas_chanaral)}, nuevas (no duplicadas): {len(nuevas)}')
    print(f'Total filas en CSV: {len(existentes) + len(nuevas)}')


if __name__ == '__main__':
    main()
