#!/usr/bin/env python3
# Genera LLUVIA_DIARIA_1966_2017 (Fase B.1 del plan de granularidad,
# 08-ago-2026) leyendo pluviosidad_diaria_consolidada.sqlite y lo inyecta en
# sim-cosmoclima.html, mismo patron de
# generar_curvas_estaciones.py (marcador de bloque, idempotente).
#
# Una sola estacion-reloj (Huintil, la misma que ya alimenta PLUVIOSIDAD_MENSUAL
# -- no promediar entre estaciones, mezclaria el pulso local que H2/H4
# necesitan ver). Rango 1966-01-01 a 2017-05-31: confirmado en B.0 que es
# donde el diario real de Huintil existe (se corta en seco el 2017-05-31;
# despues solo hay mensual, ver PLUVIOSIDAD_MENSUAL para ese tramo).
#
# Regla de huecos: dia sin fila real en el sqlite queda `null` explicito --
# NUNCA interpolado ni repetido, misma norma que ya usa el badge "SIN DATO
# REAL" del instrumento para los meses sin dato.

import json
import re
import sqlite3
from datetime import date, timedelta
import os

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
SQLITE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'investigacion', 'fuentes', 'pluviosidad_diaria_consolidada.sqlite')
HTML_PATH = os.path.join(os.path.dirname(__file__), 'sim-cosmoclima.html')

LOCALIDAD = 'Huintil'
DIA_INICIO = date(1966, 1, 1)
DIA_FIN = date(2017, 5, 31)


def leer_lluvia_diaria():
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.execute(
        "SELECT fecha, lluvia_mm FROM pluviosidad_diaria "
        "WHERE localidad=? AND tipo_fuente='estacion_real' AND fecha BETWEEN ? AND ?",
        (LOCALIDAD, DIA_INICIO.isoformat(), DIA_FIN.isoformat()),
    )
    filas = {fecha: mm for fecha, mm in cur.fetchall()}
    con.close()

    salida = {}
    dias_con_dato = 0
    d = DIA_INICIO
    while d <= DIA_FIN:
        clave = d.isoformat()
        if clave in filas:
            salida[clave] = filas[clave]
            dias_con_dato += 1
        else:
            salida[clave] = None
        d += timedelta(days=1)

    dias_totales = (DIA_FIN - DIA_INICIO).days + 1
    print(f'{LOCALIDAD}: {dias_con_dato}/{dias_totales} días reales ({dias_con_dato/dias_totales*100:.1f}% cobertura), {dias_totales - dias_con_dato} días null.')
    return salida


def inyectar(html, lluvia_diaria):
    bloque = (
        '// LLUVIA_DIARIA_1966_2017 -- generado por generar_lluvia_diaria.py\n'
        '// (08-ago-2026, Fase B del plan de granularidad): serie diaria REAL de\n'
        '// Huintil (misma estacion que PLUVIOSIDAD_MENSUAL), 1966-01-01 a\n'
        '// 2017-05-31 -- el diario real se corta ahi (confirmado en B.0: el tramo\n'
        '// 2017-06 a 2018-12 solo existe en mensual, coincide exacto con\n'
        '// PLUVIOSIDAD_MENSUAL, no hay diario detras). Dias sin fila real en el\n'
        '// sqlite quedan `null` explicito -- NUNCA interpolados ni repetidos,\n'
        f'// {sum(1 for v in lluvia_diaria.values() if v is None)} de {len(lluvia_diaria)} dias son null.\n'
        'const LLUVIA_DIARIA_1966_2017 = ' + json.dumps(lluvia_diaria, ensure_ascii=False) + ';\n'
        "const LLUVIA_DIARIA_FIN = '" + DIA_FIN.isoformat() + "'; // ultimo dia con diario real disponible\n"
    )
    marcador_ini = '// === INICIO LLUVIA_DIARIA_1966_2017 (generado) ===\n'
    marcador_fin = '// === FIN LLUVIA_DIARIA_1966_2017 (generado) ===\n'
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
    lluvia_diaria = leer_lluvia_diaria()
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        html = f.read()
    html_nuevo = inyectar(html, lluvia_diaria)
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html_nuevo)
    print(f'Inyectado en {HTML_PATH}')


if __name__ == '__main__':
    main()
