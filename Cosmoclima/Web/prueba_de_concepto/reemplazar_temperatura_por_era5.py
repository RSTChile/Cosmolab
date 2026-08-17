#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reemplazar_temperatura_por_era5.py — cambia TEMPERATURA_DIARIA_ZHCS de NASA
POWER (1981-2026) a Open-Meteo/ERA5 (1966-2026), fuente UNICA para los 60 anios
(12-ago-2026, autorizado por Alexis: "si tenemos datos coherentes de
Open-Meteo/ERA5 para los 60 anios, usemos eso y lo declaramos").

POR QUE: la corrida completa con C3 mostro que 1966-1980 solo puede producir 2
de las 4 zonas del Plano Cierre (Selva Hostil y Colapso quedan en 0.00% los 15
anios) porque NASA POWER no tiene dato antes de 1981 y esos anios corrian sobre
un vaiven sintetico suave: el acoplamiento nunca bajaba del umbral y el sistema
nunca podia ser inviable.

POR QUE FUENTE UNICA Y NO SOLO RELLENAR EL HUECO: al comparar las dos fuentes
en los 16.654 dias que se solapan (temperatura_comparacion_fuentes.txt),
Open-Meteo/ERA5 da Tmin +1.62 °C de media sobre NASA POWER (mediana +1.92) y
solo el 21% de los dias caen dentro de 1 °C. Rellenar 1966-1980 con una fuente
y dejar 1981+ con la otra meteria un ESCALON de casi 2 °C justo en 1981 -- y el
instrumento clasifica por anomalia semanal contra la distribucion historica de
esa misma semana, asi que ese escalon se leeria como senial climatica real.

Ademas, en 45 anios NASA POWER nunca reporta una minima bajo -0.6 °C, mientras
ERA5 llega a -13.2 °C. Para un punto sobre 1.000 m en el valle, minimas bajo
cero en invierno son reales: MERRA-2 (el modelo detras de POWER) las aplana por
resolucion gruesa en terreno de montania. Aplanar los extremos es exactamente lo
que le quita variacion a la viabilidad, que es lo que este instrumento mide.

ERA5 (ECMWF) es ademas el reanalisis de referencia mundial, y es la MISMA fuente
que este proyecto ya usa para la pluviosidad diaria (Open-Meteo corre sobre
ERA5/ERA5-Land) -- una dependencia menos, no una mas.

Sigue siendo REANALISIS, no estacion en tierra. Se declara asi en el comentario
del bloque, igual que se declaraba NASA POWER.
"""

import csv
import json
import os

BASE = '/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima'
CSV_ERA5 = f'{BASE}/investigacion/fuentes/temperatura_diaria_zhcs_openmeteo.csv'
HTML = f'{BASE}/Web/prueba_de_concepto/sim-cosmoclima.html'

MARCA_INI = '// === INICIO TEMPERATURA_DIARIA_ZHCS (generado) ==='
MARCA_FIN = '// === FIN TEMPERATURA_DIARIA_ZHCS (generado) ==='


def main():
    filas = list(csv.DictReader(open(CSV_ERA5, encoding='utf-8')))
    datos = {}
    for r in filas:
        try:
            datos[r['fecha']] = {'tmax': float(r['tmax']), 'tmin': float(r['tmin'])}
        except (ValueError, TypeError):
            pass
    fechas = sorted(datos)
    print(f'dias con dato: {len(datos):,}   de {fechas[0]} a {fechas[-1]}')

    cuerpo = (
        MARCA_INI + '\n'
        '// TEMPERATURA_DIARIA_ZHCS -- generado por reemplazar_temperatura_por_era5.py\n'
        '// (12-ago-2026). FUENTE UNICA PARA LOS 60 ANIOS: Open-Meteo Historical\n'
        '// Weather API (archive-api.open-meteo.com), que corre sobre ERA5/ERA5-Land\n'
        '// del ECMWF -- gratis, sin llave, sin registro. Mismo punto-reloj que\n'
        '// PLUVIOSIDAD_MENSUAL/LLUVIA_DIARIA_1966_2017 (Huintil, -31.5669/-70.9817)\n'
        '// y la MISMA fuente que ya alimenta la pluviosidad diaria del proyecto.\n'
        '// Es REANALISIS (modelo + asimilacion de observaciones), no estacion en\n'
        '// tierra -- se declara, no se disfraza.\n'
        '//\n'
        '// POR QUE SE CAMBIO (antes era NASA POWER 1981-2026): POWER no tiene dato\n'
        '// antes de 1981, y esos 15 anios corrian sobre un vaiven sintetico suave\n'
        '// que la temperatura del planeta seguia sin esfuerzo -- resultado medido:\n'
        '// 1966-1980 solo producia 2 de las 4 zonas del Plano Cierre (Selva Hostil\n'
        '// y Colapso en 0.00% los 15 anios), porque el sistema nunca podia ser\n'
        '// inviable. No se relleno solo el hueco: al comparar las dos fuentes en\n'
        '// los 16.654 dias que se solapan, ERA5 da Tmin +1.62 °C de media sobre\n'
        '// POWER (solo 21% de los dias dentro de 1 °C), asi que mezclarlas habria\n'
        '// metido un escalon de casi 2 °C justo en 1981 -- y como el instrumento\n'
        '// clasifica por ANOMALIA semanal contra la historia de esa misma semana,\n'
        '// ese escalon se leeria como senial climatica real. Una sola regla para\n'
        '// los 60 anios. Detalle completo en\n'
        '// investigacion/fuentes/temperatura_comparacion_fuentes.txt\n'
        f'// Cobertura: {fechas[0]} a {fechas[-1]}, {len(datos):,} dias, SIN huecos.\n'
        'const TEMPERATURA_DIARIA_ZHCS = ' + json.dumps(datos, separators=(', ', ': '), ensure_ascii=False) + ';\n'
        f"const TEMPERATURA_DIARIA_INICIO = '{fechas[0]}';\n"
        f"const TEMPERATURA_DIARIA_FIN = '{fechas[-1]}';\n"
        + MARCA_FIN
    )

    s = open(HTML, encoding='utf-8').read()
    i = s.find(MARCA_INI)
    j = s.find(MARCA_FIN)
    if i < 0 or j < 0:
        print('ERROR: no se encontraron las marcas del bloque'); raise SystemExit(1)
    j += len(MARCA_FIN)
    antes = len(s)
    s = s[:i] + cuerpo + s[j:]
    open(HTML, 'w', encoding='utf-8').write(s)
    print(f'HTML actualizado: {antes:,} -> {len(s):,} caracteres ({len(s)-antes:+,})')


if __name__ == '__main__':
    main()
