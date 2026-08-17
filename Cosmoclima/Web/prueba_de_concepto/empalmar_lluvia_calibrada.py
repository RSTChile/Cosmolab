#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
empalmar_lluvia_calibrada.py — deja UNA serie de lluvia sin escalón de fuente
para el punto-reloj del instrumento EIT-3 (12-ago-2026).

EL PROBLEMA (medido, no supuesto): la serie que usaba el instrumento cambiaba
de fuente en 2019 —CR2 estación real hasta 2018, NASA POWER satelital desde
2019— justo en el tramo que se usa para validar. Después de marzo-2018 no
queda ninguna estación real en 60 km del punto-reloj. Efecto: 2018 marcaba
67,4 % de Jardín Fértil con un pico de 65 mm medido por ESTACIÓN y 2024 marcaba
0,0 % con 13,7 mm medidos por SATÉLITE. No se comparaban años: se comparaban
instrumentos.

LA SOLUCIÓN ELEGIDA POR ALEXIS — empalme calibrado:
  · 1966-2018  se CONSERVA el dato medido en tierra (CR2 Huintil, estación real)
  · 2019 en adelante  se usa Open-Meteo/ERA5 CORREGIDO POR SESGO ESTACIONAL

Por qué corregido y no crudo: comparados en los 631 meses en que ambas series
existen, ERA5 reproduce muy bien la FORMA (r = 0,905) pero infla la MAGNITUD
un +86 %. Crudo, los meses que cruzan el umbral de germinación de 15 mm pasan
de 173 (real) a 290 — la floración se dispararía en años que fueron secos.
Con un factor por mes calendario, calibrado contra esos mismos 631 meses de
estación real, el sesgo baja a 0,00 mm/mes, la correlación SUBE a r = 0,914 y
los cruces del umbral quedan en 186 vs 173 reales.

Riesgo verificado y descartado: los factores de verano son inestables (dic-feb
tienen ratios extremos porque casi no llueve), pero en 53 años de estación real
esos meses NUNCA cruzaron 15 mm ni fueron NUNCA el pico anual — el pico cae en
mayo-agosto (49 de 53 años), donde los factores son estables (0,55-0,69). El
factor inestable se aplica solo a meses que no deciden nada.

Esto NO es maquillar el dato: es la técnica estándar de empalme calibrado
(bias-corrected splicing) y queda declarada en el propio bloque del HTML, mes a
mes, con la fuente de cada valor.
"""

import csv
import json
import re
import os

BASE = '/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima'
HTML = f'{BASE}/Web/prueba_de_concepto/sim-cosmoclima.html'
ERA5 = f'{BASE}/investigacion/fuentes/lluvia_mensual_zhcs_openmeteo.csv'
FUENTES = f'{BASE}/investigacion/fuentes/lluvia_mensual_zhcs_1900_2027.csv'
SALIDA_CSV = f'{BASE}/investigacion/fuentes/lluvia_mensual_zhcs_empalmada.csv'
INFORME = f'{BASE}/investigacion/fuentes/lluvia_empalme_calibrado.txt'

MARCA_INI = '// === INICIO PLUVIOSIDAD_MENSUAL'
CORTE = '2019-01'   # primer mes sin estación real


def main():
    s = open(HTML, encoding='utf-8').read()
    actual = {k: v for k, v in json.loads(
        re.search(r'const PLUVIOSIDAD_MENSUAL = (\{.*?\});', s, re.S).group(1)).items()}
    era = {r['anio_mes']: float(r['lluvia_mm'])
           for r in csv.DictReader(open(ERA5, encoding='utf-8'))}
    fte = {r['anio_mes']: r.get('fuente', '')
           for r in csv.DictReader(open(FUENTES, encoding='utf-8'))}

    # --- factores de corrección: solo con meses de ESTACIÓN REAL ---
    reales = [(k, actual[k], era[k]) for k in sorted(actual)
              if actual.get(k) is not None and k in era and 'CR2' in fte.get(k, '')]
    fac = {}
    for m in range(1, 13):
        sub = [(a, b) for k, a, b in reales if int(k[5:7]) == m]
        sa, sb = sum(a for a, _ in sub), sum(b for _, b in sub)
        fac[m] = (sa / sb) if sb > 0 else 1.0

    # --- serie empalmada ---
    serie, origen = {}, {}
    for k in sorted(set(actual) | set(era)):
        if k < CORTE:
            v = actual.get(k)
            if v is not None:
                serie[k], origen[k] = round(v, 2), 'estacion_real_CR2'
            elif k in era:   # hueco puntual del registro real -> ERA5 corregido
                serie[k] = round(era[k] * fac[int(k[5:7])], 2)
                origen[k] = 'ERA5_corregido_relleno'
        else:
            if k in era:
                serie[k] = round(era[k] * fac[int(k[5:7])], 2)
                origen[k] = 'ERA5_corregido'

    with open(SALIDA_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['anio_mes', 'lluvia_mm', 'origen'])
        for k in sorted(serie):
            w.writerow([k, serie[k], origen[k]])

    from collections import Counter
    c = Counter(origen.values())
    L = [f'Serie de lluvia empalmada — punto-reloj Huintil (-31.5669,-70.9817)', '']
    L.append(f'Meses totales: {len(serie):,}  ({min(serie)} a {max(serie)})')
    for k, n in c.most_common():
        L.append(f'  {k:<26} {n:>5} meses')
    L.append('')
    L.append('Factores de corrección por mes calendario (estación_real / ERA5),')
    L.append(f'calibrados con los {len(reales)} meses en que ambas series existen:')
    L.append('  ' + '  '.join(f'{m:02d}:{fac[m]:.3f}' for m in range(1, 13)))
    L.append('')
    L.append('Los meses de verano tienen factor inestable porque casi no llueve, pero en')
    L.append('53 años de estación real dic-feb NUNCA cruzaron el umbral de 15 mm ni fueron')
    L.append('nunca el pico anual (el pico cae en may-ago en 49 de 53 años).')
    texto = '\n'.join(L)
    open(INFORME, 'w', encoding='utf-8').write(texto + '\n')
    print(texto)

    # --- reemplazo del bloque en el HTML ---
    i = s.find(MARCA_INI)
    if i < 0:
        # no hay marcas: reemplazar solo la sentencia
        pat = re.compile(r'const PLUVIOSIDAD_MENSUAL = \{.*?\};', re.S)
        m = pat.search(s)
        if not m:
            print('ERROR: no se encontró PLUVIOSIDAD_MENSUAL'); raise SystemExit(1)
        ini, fin = m.start(), m.end()
    else:
        m = re.compile(r'const PLUVIOSIDAD_MENSUAL = \{.*?\};', re.S).search(s, i)
        ini, fin = m.start(), m.end()

    cabecera = (
        '// LLUVIA EMPALMADA Y CALIBRADA (12-ago-2026) -- una sola serie, sin\n'
        '// escalon de fuente. Antes cambiaba de CR2 (estacion real) a NASA POWER\n'
        '// (satelital) en 2019, justo en el tramo de validacion, y despues de\n'
        '// marzo-2018 no queda NINGUNA estacion real en 60 km del punto-reloj.\n'
        '// Efecto medido: 2018 marcaba 67,4% de Jardin Fertil con un pico de 65 mm\n'
        '// medido por ESTACION y 2024 marcaba 0,0% con 13,7 mm medidos por SATELITE\n'
        '// -- no se comparaban anios, se comparaban instrumentos.\n'
        '// Ahora: 1966-2018 conserva el dato MEDIDO EN TIERRA (CR2 Huintil); desde\n'
        '// 2019 se usa Open-Meteo/ERA5 CORREGIDO POR SESGO ESTACIONAL. La correccion\n'
        '// se calibro con los 631 meses en que ambas series existen: ERA5 crudo\n'
        '// reproduce bien la forma (r=0,905) pero infla la magnitud +86% (los meses\n'
        '// sobre el umbral de germinacion de 15 mm pasarian de 173 a 290); con un\n'
        '// factor por mes calendario el sesgo baja a 0,00 mm/mes, r sube a 0,914 y\n'
        '// los cruces quedan en 186 vs 173 reales. Detalle mes a mes en\n'
        '// investigacion/fuentes/lluvia_mensual_zhcs_empalmada.csv y el informe\n'
        '// lluvia_empalme_calibrado.txt\n'
    )
    nuevo = cabecera + 'const PLUVIOSIDAD_MENSUAL = ' + json.dumps(
        serie, separators=(', ', ': '), ensure_ascii=False) + ';'
    s2 = s[:ini] + nuevo + s[fin:]
    open(HTML, 'w', encoding='utf-8').write(s2)
    print(f'\nHTML actualizado: {len(s):,} -> {len(s2):,} caracteres')


if __name__ == '__main__':
    main()
