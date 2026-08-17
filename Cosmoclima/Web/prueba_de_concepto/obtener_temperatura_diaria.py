#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obtener_temperatura_diaria.py -- Parte A del plan de temperatura real
(09-ago-2026, a pedido de Alexis: "¿Sabes qué nos falta...? La temperatura
diaria con mínima y máxima... ¿Tenemos los datos de temperatura en la BD?").

Confirmado que NO: pluviosidad_diaria_consolidada.sqlite es solo lluvia. Acá
se trae temperatura REAL (no simulada) de NASA POWER para el mismo punto-
reloj que ya usa PLUVIOSIDAD_MENSUAL/LLUVIA_DIARIA_1966_2017 (Huintil,
-31.5669,-70.9817), vía datos_clima.py (ya probado en el proyecto). NASA
POWER solo cubre 1981 en adelante (antes "out of range", verificado en
archivo_clima.py) -- antes de esa fecha la curva queda simplemente sin dato,
no se inventa.

Inyecta TEMPERATURA_DIARIA_ZHCS en el HTML con el mismo patrón de marcador
de bloque que ya usan generar_lluvia_diaria.py / generar_bandas_oni.py.
"""
import json
import os
import re
import sys
from datetime import date

CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CARPETA, '..', '..'))
from datos_clima import traer, escribir_csv  # noqa: E402

HTML_PATH = os.path.join(CARPETA, 'sim-cosmoclima.html')
CSV_SALIDA = os.path.join(CARPETA, '..', '..', 'investigacion', 'fuentes',
                           'temperatura_diaria_zhcs_nasa_power.csv')

LAT, LON = -31.5669, -70.9817  # Huintil, mismo punto-reloj de PLUVIOSIDAD_MENSUAL
DESDE = '1981-01-01'  # NASA POWER no tiene dato antes de esto


def traer_temperatura():
    hasta = date.today().isoformat()
    print(f'Consultando NASA POWER · Huintil ({LAT},{LON}) · {DESDE} → {hasta}...')
    filas = traer(LAT, LON, DESDE, hasta)
    escribir_csv(filas, CSV_SALIDA)
    print(f'Crudo guardado en {CSV_SALIDA}')

    temp_diaria = {}
    con_dato = 0
    for f in filas:
        tmax = f.get('t_maxima_c')
        tmin = f.get('t_minima_c')
        if tmax is not None or tmin is not None:
            con_dato += 1
        temp_diaria[f['fecha']] = {'tmax': tmax, 'tmin': tmin}

    print(f'{con_dato}/{len(filas)} días con dato real '
          f'({con_dato/len(filas)*100:.1f}% cobertura), '
          f'{len(filas)-con_dato} días null (antes del rezago de POWER o sin dato).')
    return temp_diaria, filas[-1]['fecha'] if filas else None


def inyectar(html, temp_diaria, fin_real):
    bloque = (
        '// TEMPERATURA_DIARIA_ZHCS -- generado por obtener_temperatura_diaria.py\n'
        '// (09-ago-2026, Parte A del plan de temperatura real, a pedido de Alexis:\n'
        '// "La temperatura diaria con mínima y máxima... ¿Tenemos los datos en la\n'
        '// BD?" -- confirmado que NO, el sqlite de lluvia no tiene temperatura).\n'
        '// Fuente: NASA POWER (power.larc.nasa.gov), mismo punto-reloj que\n'
        '// PLUVIOSIDAD_MENSUAL/LLUVIA_DIARIA_1966_2017 (Huintil). POWER solo cubre\n'
        '// 1981-01-01 en adelante (antes es "out of range", verificado en\n'
        '// archivo_clima.py) -- antes de esa fecha no hay curva, no se inventa.\n'
        '// POWER tiene rezago de unos días: las fechas mas recientes vienen null,\n'
        '// se declara asi (no es un hueco real del dato historico).\n'
        f'// {sum(1 for v in temp_diaria.values() if v["tmax"] is None)} de '
        f'{len(temp_diaria)} dias son null.\n'
        'const TEMPERATURA_DIARIA_ZHCS = ' + json.dumps(temp_diaria, ensure_ascii=False) + ';\n'
        "const TEMPERATURA_DIARIA_INICIO = '" + DESDE + "';\n"
        "const TEMPERATURA_DIARIA_FIN = '" + (fin_real or '') + "'; // ultimo dia consultado (puede venir null por rezago de POWER)\n"
    )
    marcador_ini = '// === INICIO TEMPERATURA_DIARIA_ZHCS (generado) ===\n'
    marcador_fin = '// === FIN TEMPERATURA_DIARIA_ZHCS (generado) ===\n'
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
    temp_diaria, fin_real = traer_temperatura()
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        html = f.read()
    html_nuevo = inyectar(html, temp_diaria, fin_real)
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html_nuevo)
    print(f'Inyectado en {HTML_PATH}')


if __name__ == '__main__':
    main()
