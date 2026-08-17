#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obtener_lluvia_1966_openmeteo.py — serie ÚNICA de lluvia para el punto-reloj
del instrumento EIT-3 (12-ago-2026).

POR QUÉ: auditando la corrida completa se encontró que la lluvia del
punto-reloj CAMBIA DE FUENTE en 2019, justo en el tramo que se usa para
validar:

    1966-2018  ->  CR2 Huintil, ESTACIÓN REAL
    2019-2025  ->  NASA POWER, SATELITAL

Agravantes verificados: después de marzo-2018 NO queda ninguna estación real
en 60 km del punto-reloj (la serie CR2 termina ahí), y el propio proyecto ya
tenía documentado que NASA POWER subestima hasta -274 mm/año contra estación
real. Efecto medido: 2018 marca 67,4 % de Jardín Fértil con un pico mensual de
65 mm medido por ESTACIÓN, mientras 2024 marca 0,0 % con 13,7 mm medidos por
SATÉLITE. No se estaban comparando años: se estaban comparando instrumentos.

Es el mismo error de plano ya corregido dos veces esta sesión (los κ tratados
como medianas en vez de condiciones de posibilidad; y la temperatura, con un
escalón NASA POWER/ERA5 en 1981) — ahora en el forzante PRINCIPAL del modelo.

FUENTE: Open-Meteo Historical Weather API (archive-api.open-meteo.com), sobre
ERA5/ERA5-Land del ECMWF. Gratis, sin llave, sin registro, desde 1940. Es la
MISMA fuente que ya alimenta la temperatura diaria del instrumento y la
pluviosidad diaria de las 78 localidades del proyecto. Es REANÁLISIS, no
estación en tierra — se declara, no se disfraza.

QUÉ HACE: pide 1966 -> hoy completo y COMPARA contra la serie actual en el
tramo 1966-2018, donde la actual sí es estación real. La comparación se
escribe a un informe y se imprime: si ERA5 se aparta mucho de la estación real
ahí, hay que decirlo y consultar ANTES de adoptar — homogeneizar la fuente no
sirve de nada si el resultado se aleja de lo que de verdad se midió en tierra.

NO toca el HTML. Eso se decide leyendo la comparación.
"""

import csv
import json
import re
import urllib.request
import urllib.parse
from datetime import date

LAT, LON = -31.5669, -70.9817   # Huintil, el punto-reloj del instrumento
DESDE = '1966-01-01'
HASTA = date.today().isoformat()

BASE = '/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima'
HTML = f'{BASE}/Web/prueba_de_concepto/sim-cosmoclima.html'
SALIDA_DIARIA = f'{BASE}/investigacion/fuentes/lluvia_diaria_zhcs_openmeteo.csv'
SALIDA_MENSUAL = f'{BASE}/investigacion/fuentes/lluvia_mensual_zhcs_openmeteo.csv'
INFORME = f'{BASE}/investigacion/fuentes/lluvia_comparacion_fuentes.txt'


def traer():
    params = {
        'latitude': LAT, 'longitude': LON,
        'start_date': DESDE, 'end_date': HASTA,
        'daily': 'precipitation_sum',
        'timezone': 'America/Santiago',
    }
    url = 'https://archive-api.open-meteo.com/v1/archive?' + urllib.parse.urlencode(params)
    print(f'Consultando Open-Meteo/ERA5 · Huintil ({LAT},{LON}) · {DESDE} -> {HASTA}')
    with urllib.request.urlopen(url, timeout=300) as resp:
        d = json.load(resp)
    return d['daily']['time'], d['daily']['precipitation_sum']


def serie_actual_del_html():
    """PLUVIOSIDAD_MENSUAL tal como la usa hoy el instrumento (mm por mes)."""
    s = open(HTML, encoding='utf-8').read()
    m = re.search(r'const PLUVIOSIDAD_MENSUAL = (\{.*?\});', s, re.S)
    return json.loads(m.group(1))


def fuente_por_mes():
    """De dónde sale cada mes de la serie actual (para separar el tramo de
    estación real del satelital)."""
    out = {}
    ruta = f'{BASE}/investigacion/fuentes/lluvia_mensual_zhcs_1900_2027.csv'
    try:
        for r in csv.DictReader(open(ruta, encoding='utf-8')):
            out[r['anio_mes']] = r.get('fuente', '?')
    except FileNotFoundError:
        pass
    return out


def main():
    fechas, pr = traer()
    print(f'  recibidos {len(fechas):,} días')

    with open(SALIDA_DIARIA, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['fecha', 'lluvia_mm'])
        n = 0
        mensual = {}
        for i, fe in enumerate(fechas):
            v = pr[i]
            if v is None:
                continue
            w.writerow([fe, v])
            n += 1
            mensual[fe[:7]] = mensual.get(fe[:7], 0.0) + v
    print(f'  escrito {SALIDA_DIARIA} ({n:,} días con dato)')

    with open(SALIDA_MENSUAL, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['anio_mes', 'lluvia_mm'])
        for k in sorted(mensual):
            w.writerow([k, round(mensual[k], 2)])
    print(f'  escrito {SALIDA_MENSUAL} ({len(mensual):,} meses)')

    # ---- comparación contra la serie actual, separando por fuente ----
    actual = serie_actual_del_html()
    fuentes = fuente_por_mes()
    L = []
    L.append(f'Comparación de fuentes de LLUVIA — punto-reloj Huintil ({LAT},{LON})')
    L.append('')
    L.append(f'Open-Meteo/ERA5 (nueva, serie única): {min(mensual)} a {max(mensual)}, {len(mensual):,} meses')
    act = {k: v for k, v in actual.items() if v is not None}
    L.append(f'Serie actual del instrumento:         {min(act)} a {max(act)}, {len(act):,} meses con valor')

    grupos = {'estacion_real_CR2': [], 'satelital_NASA_POWER': [], 'otros': []}
    for k, v in sorted(act.items()):
        if k not in mensual:
            continue
        fte = fuentes.get(k, '')
        par = (k, v, mensual[k])
        if 'CR2' in fte:
            grupos['estacion_real_CR2'].append(par)
        elif 'POWER' in fte:
            grupos['satelital_NASA_POWER'].append(par)
        else:
            grupos['otros'].append(par)

    def stats(pares, etiqueta):
        if not pares:
            L.append(f'\n{etiqueta}: sin meses en común')
            return
        difs = [b - a for _, a, b in pares]
        n = len(difs)
        media = sum(difs) / n
        orden = sorted(difs)
        med = orden[n // 2]
        sa = sum(a for _, a, _ in pares) / n
        sb = sum(b for _, _, b in pares) / n
        # correlación de Pearson entre ambas series
        ma = sum(a for _, a, _ in pares) / n
        mb = sum(b for _, _, b in pares) / n
        num = sum((a - ma) * (b - mb) for _, a, b in pares)
        da = sum((a - ma) ** 2 for _, a, _ in pares) ** 0.5
        db = sum((b - mb) ** 2 for _, _, b in pares) ** 0.5
        r = num / (da * db) if da and db else float('nan')
        L.append(f'\n{etiqueta}  (n={n:,} meses)')
        L.append(f'  media mensual  actual={sa:7.2f} mm   ERA5={sb:7.2f} mm')
        L.append(f'  ERA5 menos actual: media {media:+.2f} mm/mes, mediana {med:+.2f}')
        L.append(f'  correlación entre ambas series: r = {r:.3f}')
        L.append(f'  total del período: actual={sa*n:,.0f} mm   ERA5={sb*n:,.0f} mm   '
                 f'({(sb/sa-1)*100 if sa else float("nan"):+.1f} %)')

    stats(grupos['estacion_real_CR2'], 'TRAMO CON ESTACIÓN REAL (CR2 Huintil, 1966-2018) — el que importa')
    stats(grupos['satelital_NASA_POWER'], 'TRAMO SATELITAL (NASA POWER, 2019-2025)')
    stats(grupos['otros'], 'Otros meses')

    L.append('')
    L.append('CÓMO LEER ESTO: si en el tramo de ESTACIÓN REAL la correlación es alta y el')
    L.append('sesgo chico, ERA5 reproduce razonablemente lo que se midió en tierra y sirve')
    L.append('como serie única. Si se aparta mucho, homogeneizar la fuente no alcanza: habría')
    L.append('que decidir con Alexis, porque estaríamos cambiando dato real por modelo.')

    texto = '\n'.join(L)
    open(INFORME, 'w', encoding='utf-8').write(texto + '\n')
    print()
    print(texto)


if __name__ == '__main__':
    main()
