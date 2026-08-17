#!/usr/bin/env python3
# Genera ONI_BANDAS (Fase "etiquetas El Niño/La Niña", 09-ago-2026, a pedido
# de Alexis tras ver los años del Niño marcados en la lluvia diaria/mensual
# ya corregida) e inyecta en sim-cosmoclima.html,
# mismo patrón de marcador que ya usan LLUVIA_DIARIA_1966_2017/etc.
#
# Fuente: investigacion/fuentes/oni_historico_completo_1966_2026.csv --
# tabla ONI (Oceanic Niño Index) real, NOAA CPC v5, bajada el 09-ago-2026
# (misma versión "v5" que ya usaba investigacion/fuentes/
# oni_enso_2026_vs_historico.csv, cruzada y verificada consistente: DJF-1998
# 2.24 vs 2.2, DJF-2015 0.69 vs 0.7, DJF-2024 1.92 vs 1.9 -- solo redondeo).
# Cubre 1966-2026 (2027 no tiene ONI todavía, es futuro, no se inventa).
#
# Clasificación: criterio ESTÁNDAR de NOAA -- episodio El Niño/La Niña =
# al menos 5 temporadas trimestrales SEGUIDAS con ONI >= +0.5 (Niño) o
# <= -0.5 (Niña). No es "a ojo", es el mismo criterio que usa NOAA para
# declarar un evento oficialmente.
#
# Cada temporada (DJF, JFM, ... NDJ) se ubica en el mes CALENDARIO central
# (DJF->enero, JFM->febrero, ..., NDJ->diciembre del año de la fila) --
# mismo calendario simplificado que ya usa todo el instrumento (365
# días/año, Febrero fijo en 28, sin bisiestos) para que las bandas calcen
# exacto con el eje x del gráfico.

import csv
import json
import re
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'investigacion', 'fuentes', 'oni_historico_completo_1966_2026.csv')
HTML_PATH = os.path.join(os.path.dirname(__file__), 'sim-cosmoclima.html')

ANIO_CERO = 1966
DIAS_POR_ANIO_CAL = 365
DIAS_POR_MES_CAL = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
TEMPORADAS = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']


def dia_desde_anio_mes(anio, mes):
    offset = sum(DIAS_POR_MES_CAL[:mes])
    return (anio - ANIO_CERO) * DIAS_POR_ANIO_CAL + offset


def leer_serie():
    serie = []  # [(anio, mes0based, oni_o_None)]
    with open(CSV_PATH, newline='') as f:
        for fila in csv.DictReader(f):
            anio = int(fila['anio'])
            for mes, temporada in enumerate(TEMPORADAS):
                v = fila[temporada].strip()
                serie.append((anio, mes, float(v) if v else None))
    return serie


def clasificar(serie):
    """Devuelve una lista paralela de 'nino'/'nina'/None por posición,
    marcando SOLO las temporadas que pertenecen a una racha de >=5
    seguidas con el mismo signo de umbral (criterio oficial NOAA)."""
    n = len(serie)
    etiqueta = [None] * n
    for signo, umbral, nombre in [(1, 0.5, 'nino'), (-1, -0.5, 'nina')]:
        i = 0
        while i < n:
            if serie[i][2] is None:
                i += 1
                continue
            cumple = (signo == 1 and serie[i][2] >= umbral) or (signo == -1 and serie[i][2] <= umbral)
            if not cumple:
                i += 1
                continue
            j = i
            while j < n and serie[j][2] is not None and (
                (signo == 1 and serie[j][2] >= umbral) or (signo == -1 and serie[j][2] <= umbral)
            ):
                j += 1
            if j - i >= 5:
                for k in range(i, j):
                    etiqueta[k] = nombre
            i = j
    return etiqueta


def armar_bandas(serie, etiqueta):
    bandas = []
    actual = None
    for (anio, mes, _), tipo in zip(serie, etiqueta):
        if tipo != actual:
            if actual is not None:
                bandas[-1]['fin'] = dia_desde_anio_mes(anio, mes) - 1
            if tipo is not None:
                bandas.append({'inicio': dia_desde_anio_mes(anio, mes), 'tipo': tipo})
            actual = tipo
    if actual is not None:
        ultimo_anio, ultimo_mes, _ = serie[-1]
        bandas[-1]['fin'] = dia_desde_anio_mes(ultimo_anio, ultimo_mes) + DIAS_POR_MES_CAL[ultimo_mes] - 1
    return bandas


def inyectar(html, bandas):
    bloque = (
        '// ONI_BANDAS -- generado por generar_bandas_oni.py (09-ago-2026, a\n'
        '// pedido de Alexis: "sería bueno poner una etiqueta indicando esos años").\n'
        '// Fuente: NOAA CPC ONI v5 real (investigacion/fuentes/\n'
        '// oni_historico_completo_1966_2026.csv), clasificado con el criterio\n'
        '// OFICIAL de NOAA (>=5 temporadas trimestrales seguidas con ONI>=+0.5\n'
        '// para Niño, <=-0.5 para Niña) -- no es una lista a ojo de años famosos.\n'
        '// 2027 no tiene ONI todavía (es futuro), no se inventa.\n'
        'const ONI_BANDAS = ' + json.dumps(bandas, ensure_ascii=False) + ';\n'
    )
    marcador_ini = '// === INICIO ONI_BANDAS (generado) ===\n'
    marcador_fin = '// === FIN ONI_BANDAS (generado) ===\n'
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
    serie = leer_serie()
    etiqueta = clasificar(serie)
    bandas = armar_bandas(serie, etiqueta)
    n_nino = sum(1 for b in bandas if b['tipo'] == 'nino')
    n_nina = sum(1 for b in bandas if b['tipo'] == 'nina')
    print(f'{len(bandas)} bandas ({n_nino} Niño, {n_nina} Niña), 1966-2026, criterio oficial NOAA (>=5 temporadas seguidas).')
    for b in bandas:
        print(' ', b)
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        html = f.read()
    html_nuevo = inyectar(html, bandas)
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html_nuevo)
    print(f'Inyectado en {HTML_PATH}')


if __name__ == '__main__':
    main()
