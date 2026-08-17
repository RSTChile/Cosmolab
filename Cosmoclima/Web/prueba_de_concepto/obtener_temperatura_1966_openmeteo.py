#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obtener_temperatura_1966_openmeteo.py — completa el hueco de temperatura real
1966-1980 del instrumento EIT-3 (12-ago-2026).

POR QUÉ: la corrida completa mostró que 1966-1980 SOLO puede producir 2 de las
4 zonas del Plano Cierre (Selva Hostil y Colapso quedan en 0.00% los 15 años).
La causa es que TEMPERATURA_DIARIA_ZHCS (NASA POWER) empieza en 1981-01-01, y
antes de eso la física corre sobre un vaivén sintético suave que la temperatura
del planeta sigue sin esfuerzo: el acoplamiento nunca baja del umbral y el
sistema nunca puede ser inviable. Alexis: "mejor buscamos los datos reales y
completamos la tabla... eso es lo correcto".

FUENTE: Open-Meteo Historical Weather API (archive-api.open-meteo.com), que
corre sobre ERA5 / ERA5-Land y cubre desde 1940 -- gratis, sin llave, sin
registro. Es la MISMA fuente que este proyecto ya usa para la pluviosidad
diaria (obtener_pluviosidad_diaria_openmeteo.py), así que no se introduce una
dependencia nueva. Es reanálisis, no estación en tierra -- igual naturaleza
que NASA POWER, que es lo que ya alimenta 1981+. Se declara, no se disfraza.

QUÉ HACE: pide 1966-2026 COMPLETO (no solo el hueco) para poder comparar el
tramo 1981-2026 contra NASA POWER y medir cuánto difieren las dos fuentes
ANTES de mezclarlas. Mezclar dos reanálisis sin medir el salto sería
introducir un escalón artificial justo en 1981 -- y el instrumento clasifica
por anomalía semanal, así que un escalón de fuente se leería como señal.

Salida: temperatura_diaria_zhcs_openmeteo.csv + un informe de comparación.
NO toca el HTML: eso se decide después de ver los números.
"""

import csv
import json
import urllib.request
import urllib.parse
from datetime import date

LAT, LON = -31.5669, -70.9817   # Huintil, el mismo punto-reloj de PLUVIOSIDAD_MENSUAL
DESDE = '1966-01-01'
HASTA = date.today().isoformat()

BASE = '/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima'
SALIDA = f'{BASE}/investigacion/fuentes/temperatura_diaria_zhcs_openmeteo.csv'
NASA = f'{BASE}/investigacion/fuentes/temperatura_diaria_zhcs_nasa_power.csv'
INFORME = f'{BASE}/investigacion/fuentes/temperatura_comparacion_fuentes.txt'


def traer():
    params = {
        'latitude': LAT, 'longitude': LON,
        'start_date': DESDE, 'end_date': HASTA,
        'daily': 'temperature_2m_max,temperature_2m_min',
        'timezone': 'America/Santiago',
    }
    url = 'https://archive-api.open-meteo.com/v1/archive?' + urllib.parse.urlencode(params)
    print(f'Consultando Open-Meteo/ERA5 · Huintil ({LAT},{LON}) · {DESDE} -> {HASTA}')
    with urllib.request.urlopen(url, timeout=180) as resp:
        d = json.load(resp)
    return d['daily']['time'], d['daily']['temperature_2m_max'], d['daily']['temperature_2m_min']


def leer_nasa():
    """El CSV de NASA POWER ya en el repo trae MUCHAS columnas y los nombres
    NO son tmax/tmin sino t_maxima_c / t_minima_c -- se leen por nombre exacto
    verificado contra el encabezado real, no por posicion."""
    out = {}
    try:
        with open(NASA, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                k = r.get('fecha')
                if not k:
                    continue
                try:
                    out[k] = (float(r['t_maxima_c']), float(r['t_minima_c']))
                except (KeyError, ValueError, TypeError):
                    pass
    except FileNotFoundError:
        print(f'AVISO: no se encontro {NASA} -- se omite la comparacion')
    return out


def main():
    fechas, tmax, tmin = traer()
    n = len(fechas)
    print(f'  recibidos {n:,} dias')

    with open(SALIDA, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['fecha', 'tmax', 'tmin'])
        for i in range(n):
            if tmax[i] is None or tmin[i] is None:
                continue
            w.writerow([fechas[i], tmax[i], tmin[i]])
    print(f'  escrito {SALIDA}')

    # --- comparacion contra NASA POWER en el tramo que se solapa ---
    nasa = leer_nasa()
    om = {fechas[i]: (tmax[i], tmin[i]) for i in range(n)
          if tmax[i] is not None and tmin[i] is not None}
    comunes = sorted(set(nasa) & set(om))
    lineas = []
    lineas.append(f'Comparacion de fuentes de temperatura -- Huintil ({LAT},{LON})')
    lineas.append(f'Open-Meteo/ERA5: {min(om)} a {max(om)}  ({len(om):,} dias)')
    if nasa:
        lineas.append(f'NASA POWER:      {min(nasa)} a {max(nasa)}  ({len(nasa):,} dias)')
        lineas.append(f'Dias en comun:   {len(comunes):,}')
    if comunes:
        dmax = [om[k][0] - nasa[k][0] for k in comunes]
        dmin = [om[k][1] - nasa[k][1] for k in comunes]

        def stats(v):
            v2 = sorted(v)
            media = sum(v)/len(v)
            desv = (sum((x-media)**2 for x in v)/len(v))**0.5
            return media, desv, v2[len(v2)//2], v2[0], v2[-1]

        for etiqueta, v in (('Tmax', dmax), ('Tmin', dmin)):
            m, s, med, lo, hi = stats(v)
            lineas.append('')
            lineas.append(f'{etiqueta} (Open-Meteo menos NASA POWER), en °C:')
            lineas.append(f'  sesgo medio: {m:+.3f}   mediana: {med:+.3f}   desviacion: {s:.3f}')
            lineas.append(f'  rango: {lo:+.2f} a {hi:+.2f}')
            dentro1 = sum(1 for x in v if abs(x) <= 1.0)/len(v)*100
            dentro2 = sum(1 for x in v if abs(x) <= 2.0)/len(v)*100
            lineas.append(f'  |dif| <= 1 °C en {dentro1:.1f}% de los dias; <= 2 °C en {dentro2:.1f}%')
    texto = '\n'.join(lineas)
    open(INFORME, 'w', encoding='utf-8').write(texto + '\n')
    print()
    print(texto)


if __name__ == '__main__':
    main()
