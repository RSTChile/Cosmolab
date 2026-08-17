#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparar_floracion_vs_ndvi_historico.py -- 09-ago-2026 (2), a pedido directo
de Alexis: "si tenemos datos reales de pluviosidad y temperatura, que no
dependen de una fotografía satelital, podemos contrastar lo que dice la
fotografía contra datos duros de terreno". Y después: "agrega esta
comparación como gráfico nuevo... luego, verifica el contraste valle/costa,
y agregamos esas nuevas curvas en el mismo gráfico".

NO mete NDVI como ingrediente de la curva de floración -- la curva sigue
siendo exactamente la vigente hoy (EMP_B0=-1.2123, EMP_B1=0.0185, sin
cambios, ver v3 en curva_empirica_gyriosomus.md: agregar temperatura no
pasó la validación y no se adoptó). Este script SOLO simula, día por día,
la floración que esa misma curva predeciría con lluvia real -- en DOS
puntos reales distintos, no uno:

- VALLE: Huintil (el punto-reloj de siempre), misma lógica exacta que
  corre hoy en vivo en el HTML (30 días móviles de diario real hasta
  2017-05-31, promedio de 2 meses reales después).
- COSTA: Los Vilos Dmc (-31.9103,-71.5086, misma banda de latitud que
  Huintil -31.5669 -- para que la comparación sea costa-vs-valle, no
  norte-vs-sur mezclado con costa-vs-valle). Estación real (CR2/DGA,
  código 4820001), diario real 1982-01-01 a 2017-05-31 -- MISMA curva,
  SIN recalibrar para esta estación (es exploración, no un segundo modelo
  validado, declarado así en el propio HTML). No se extiende más allá de
  2017-05-31 porque no hay continuación mensual real para Los Vilos como
  sí hay para Huintil (PLUVIOSIDAD_MENSUAL es específica del punto-reloj)
  -- no se inventa una extensión.

Después compara ambas trayectorias contra el NDVI real (satélite, MODIS,
Huintil) en la MISMA ventana de tiempo que las dos cubren en común
(2000-02-18 a 2017-05-31), para que la comparación entre costa y valle
sea pareja -- no una con más años de ventaja que la otra.
"""
import csv
import json
import os
import re
import sqlite3
from datetime import date, timedelta

import numpy as np

CARPETA = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(CARPETA, '..', 'Web', 'prueba_de_concepto',
                          'prueba_de_concepto_ET3-Termico_con_mapa.html')
NDVI_CSV = os.path.join(CARPETA, 'fuentes', 'ndvi_historico_huintil_2000_2026.csv')
SQLITE_PATH = os.path.join(CARPETA, 'fuentes', 'pluviosidad_diaria_consolidada.sqlite')
SALIDA_CSV = os.path.join(CARPETA, 'comparacion_floracion_modelo_vs_ndvi_real.csv')

EMP_B0, EMP_B1 = -1.2123, 0.0185
UMBRAL_GERMINACION = 15
DIAS_RISE, DIAS_DECLINE = 90, 166
TICKS_POR_DIA = 60

SIM_DESDE = date(1996, 1, 1)  # arranca antes de la ventana de comparación para que el estado ya esté asentado
LLUVIA_DIARIA_FIN_HUINTIL = date(2017, 5, 31)

COSTA_LOCALIDAD = 'Los Vilos Dmc'
COSTA_DESDE_REAL = date(1982, 1, 1)  # primer día real de Los Vilos Dmc en el sqlite
COSTA_HASTA_REAL = date(2017, 5, 31)  # último día real -- no se extiende más

VENTANA_COMUN_DESDE = date(2000, 2, 18)  # primer NDVI real disponible
VENTANA_COMUN_HASTA = date(2017, 5, 31)  # hasta donde Los Vilos (costa) tiene dato real


def leer_const_json(html, nombre):
    m = re.search(r'const ' + nombre + r' = (\{.*?\});\n', html, re.S)
    if not m:
        raise SystemExit(f'No se encontró {nombre} en el HTML.')
    return json.loads(m.group(1))


def leer_lluvia_diaria_sqlite(localidad, desde, hasta):
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.execute(
        'SELECT fecha, lluvia_mm FROM pluviosidad_diaria WHERE localidad=? AND fecha BETWEEN ? AND ?',
        (localidad, desde.isoformat(), hasta.isoformat()))
    datos = {fecha: mm for fecha, mm in cur.fetchall()}
    con.close()
    return datos


def objetivo_floracion(lluvia):
    if lluvia < UMBRAL_GERMINACION:
        return 0.0
    logit = EMP_B0 + EMP_B1 * lluvia
    return (1 / (1 + np.exp(-logit))) * 0.9


def simular_valle_huintil(pluviosidad_mensual, lluvia_diaria_huintil, sim_desde, sim_hasta):
    mult_subida = (1 - 1 / (DIAS_RISE * TICKS_POR_DIA)) ** TICKS_POR_DIA
    mult_baja = (1 - 1 / (DIAS_DECLINE * TICKS_POR_DIA)) ** TICKS_POR_DIA
    floracion, trayectoria = 0.0, {}
    d = sim_desde
    while d <= sim_hasta:
        if d <= LLUVIA_DIARIA_FIN_HUINTIL:
            ventana = [d - timedelta(days=i) for i in range(30)]
            valores = [lluvia_diaria_huintil.get(x.isoformat()) for x in ventana]
            valores = [v for v in valores if v is not None]
            lluvia_acumulada = sum(valores) if len(valores) >= 15 else 0.0
        else:
            claves = [f'{d.year:04d}-{d.month:02d}']
            mes_prev, anio_prev = d.month - 1, d.year
            if mes_prev < 1:
                mes_prev, anio_prev = 12, d.year - 1
            claves.append(f'{anio_prev:04d}-{mes_prev:02d}')
            valores = [pluviosidad_mensual.get(c) for c in claves]
            valores = [v for v in valores if v is not None]
            lluvia_acumulada = (sum(valores) / len(valores)) if valores else 0.0

        objetivo = objetivo_floracion(lluvia_acumulada)
        mult = mult_subida if objetivo >= floracion else mult_baja
        floracion = objetivo - (objetivo - floracion) * mult
        floracion = min(max(floracion, 0.0), 0.9)
        trayectoria[d.isoformat()] = round(floracion, 4)
        d += timedelta(days=1)
    return trayectoria


def simular_costa_los_vilos(lluvia_diaria_costa, sim_desde, sim_hasta):
    """MISMA curva (EMP_B0/EMP_B1, 90d/166d), SOLO con lluvia real de Los
    Vilos en vez de Huintil -- exploración declarada, no un segundo modelo
    calibrado. Se detiene en COSTA_HASTA_REAL (sin dato real después)."""
    mult_subida = (1 - 1 / (DIAS_RISE * TICKS_POR_DIA)) ** TICKS_POR_DIA
    mult_baja = (1 - 1 / (DIAS_DECLINE * TICKS_POR_DIA)) ** TICKS_POR_DIA
    floracion, trayectoria = 0.0, {}
    d = sim_desde
    while d <= sim_hasta:
        if d <= COSTA_HASTA_REAL:
            ventana = [d - timedelta(days=i) for i in range(30)]
            valores = [lluvia_diaria_costa.get(x.isoformat()) for x in ventana]
            valores = [v for v in valores if v is not None]
            lluvia_acumulada = sum(valores) if len(valores) >= 15 else 0.0
            objetivo = objetivo_floracion(lluvia_acumulada)
            mult = mult_subida if objetivo >= floracion else mult_baja
            floracion = objetivo - (objetivo - floracion) * mult
            floracion = min(max(floracion, 0.0), 0.9)
            trayectoria[d.isoformat()] = round(floracion, 4)
        else:
            trayectoria[d.isoformat()] = None  # sin dato real de Los Vilos despues de esta fecha -- no se inventa
        d += timedelta(days=1)
    return trayectoria


def correlacion_en_ventana(trayectoria, ndvi_por_fecha, desde, hasta):
    pares = []
    for fecha, ndvi_anom in ndvi_por_fecha.items():
        f = date.fromisoformat(fecha)
        if desde <= f <= hasta and trayectoria.get(fecha) is not None:
            pares.append((ndvi_anom, trayectoria[fecha]))
    if len(pares) < 3:
        return None, len(pares)
    ndvi_v = np.array([p[0] for p in pares])
    modelo_v = np.array([p[1] for p in pares])
    return float(np.corrcoef(ndvi_v, modelo_v)[0, 1]), len(pares)


def main():
    with open(HTML_PATH, encoding='utf-8') as f:
        html = f.read()
    pluviosidad_mensual = leer_const_json(html, 'PLUVIOSIDAD_MENSUAL')
    lluvia_diaria_huintil = leer_const_json(html, 'LLUVIA_DIARIA_1966_2017')
    lluvia_diaria_costa = leer_lluvia_diaria_sqlite(COSTA_LOCALIDAD, COSTA_DESDE_REAL, COSTA_HASTA_REAL)
    print(f'{COSTA_LOCALIDAD}: {len(lluvia_diaria_costa)} días reales leídos del sqlite '
          f'({COSTA_DESDE_REAL.isoformat()} a {COSTA_HASTA_REAL.isoformat()}).')

    sim_hasta = date(2026, 8, 9)
    trayectoria_valle = simular_valle_huintil(pluviosidad_mensual, lluvia_diaria_huintil, SIM_DESDE, sim_hasta)
    trayectoria_costa = simular_costa_los_vilos(lluvia_diaria_costa, SIM_DESDE, sim_hasta)
    print(f'Simulación completa: valle (Huintil) y costa (Los Vilos), {SIM_DESDE.isoformat()} a {sim_hasta.isoformat()}.')

    # NDVI real + anomalia estacional (mismo metodo que antes)
    ndvi_filas = []
    with open(NDVI_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['ndvi'] == '':
                continue
            ndvi_filas.append({'fecha': row['fecha'], 'ndvi_real': float(row['ndvi']),
                                'pixel_reliability': row['pixel_reliability']})
    por_dia_anio = {}
    for f in ndvi_filas:
        doy = date.fromisoformat(f['fecha']).timetuple().tm_yday
        por_dia_anio.setdefault(doy, []).append(f['ndvi_real'])
    promedio_doy = {doy: sum(v) / len(v) for doy, v in por_dia_anio.items()}
    ndvi_anomalia_por_fecha = {}
    for f in ndvi_filas:
        doy = date.fromisoformat(f['fecha']).timetuple().tm_yday
        f['ndvi_anomalia'] = round(f['ndvi_real'] - promedio_doy[doy], 4)
        ndvi_anomalia_por_fecha[f['fecha']] = f['ndvi_anomalia']

    # correlaciones, EN LA MISMA VENTANA COMUN para comparar costa vs valle parejo
    corr_valle, n_valle = correlacion_en_ventana(trayectoria_valle, ndvi_anomalia_por_fecha,
                                                   VENTANA_COMUN_DESDE, VENTANA_COMUN_HASTA)
    corr_costa, n_costa = correlacion_en_ventana(trayectoria_costa, ndvi_anomalia_por_fecha,
                                                   VENTANA_COMUN_DESDE, VENTANA_COMUN_HASTA)
    print(f'\nVentana común de comparación: {VENTANA_COMUN_DESDE.isoformat()} a {VENTANA_COMUN_HASTA.isoformat()}')
    print(f'Correlación anomalía NDVI real vs. floración VALLE (Huintil):  r={corr_valle:.3f}  (n={n_valle} fechas)')
    print(f'Correlación anomalía NDVI real vs. floración COSTA (Los Vilos): r={corr_costa:.3f}  (n={n_costa} fechas)')

    # CSV de trazabilidad completo (diario, ambas trayectorias + NDVI donde exista)
    with open(SALIDA_CSV, 'w', encoding='utf-8', newline='') as f_out:
        w = csv.writer(f_out)
        w.writerow(['fecha', 'floracion_valle_huintil', 'floracion_costa_los_vilos', 'ndvi_real', 'ndvi_anomalia'])
        fechas_todas = sorted(trayectoria_valle.keys())
        ndvi_dict = {f['fecha']: f for f in ndvi_filas}
        for fecha in fechas_todas:
            nd = ndvi_dict.get(fecha)
            w.writerow([fecha, trayectoria_valle[fecha], trayectoria_costa.get(fecha),
                        nd['ndvi_real'] if nd else '', nd['ndvi_anomalia'] if nd else ''])
    print(f'\nCSV completo guardado: {SALIDA_CSV}')

    # inyeccion en el HTML: serie diaria de ambas trayectorias completas (compactas,
    # 4 decimales) + NDVI real (solo donde hay dato, ~605 puntos)
    bloque = (
        '// COMPARACION_FLORACION_NDVI -- generado por comparar_floracion_vs_ndvi_historico.py\n'
        '// (09-ago-2026 (2), a pedido de Alexis: "si tenemos datos reales de pluviosidad y\n'
        '// temperatura, que no dependen de una fotografia satelital, podemos contrastar lo\n'
        '// que dice la fotografia contra datos duros de terreno"). NO es un input de la\n'
        '// curva de floracion -- es una comparacion posterior, con la MISMA curva vigente\n'
        '// (EMP_B0/EMP_B1 sin cambios, ver curva_empirica_gyriosomus.md v3: agregar\n'
        '// temperatura no paso la validacion cruzada y no se adopto).\n'
        '// floracion_valle: Huintil (punto-reloj de siempre). floracion_costa: Los Vilos\n'
        '// Dmc (misma latitud que Huintil, real, CR2/DGA codigo 4820001) -- MISMA curva,\n'
        '// sin recalibrar para esa estacion, exploracion declarada, no un segundo modelo\n'
        '// validado. null despues de 2017-05-31 en costa (no hay dato real de Los Vilos\n'
        '// mas alla de ahi, no se inventa una extension).\n'
        f'// Correlacion NDVI real (anomalia estacional) vs. modelo, ventana comun\n'
        f'// {VENTANA_COMUN_DESDE.isoformat()} a {VENTANA_COMUN_HASTA.isoformat()}: '
        f'valle(Huintil) r={corr_valle:.3f} (n={n_valle}), costa(Los Vilos) r={corr_costa:.3f} (n={n_costa}).\n'
        'const COMPARACION_FLORACION_NDVI = {\n'
        '  floracion_valle: ' + json.dumps(trayectoria_valle, ensure_ascii=False) + ',\n'
        '  floracion_costa: ' + json.dumps(trayectoria_costa, ensure_ascii=False) + ',\n'
        '  ndvi_real: ' + json.dumps({f['fecha']: f['ndvi_real'] for f in ndvi_filas}, ensure_ascii=False) + ',\n'
        '  ndvi_anomalia: ' + json.dumps(ndvi_anomalia_por_fecha, ensure_ascii=False) + ',\n'
        '  correlaciones: ' + json.dumps({
            'ventana_desde': VENTANA_COMUN_DESDE.isoformat(), 'ventana_hasta': VENTANA_COMUN_HASTA.isoformat(),
            'valle_r': round(corr_valle, 3), 'valle_n': n_valle,
            'costa_r': round(corr_costa, 3), 'costa_n': n_costa,
        }) + ',\n'
        '};\n'
    )
    marcador_ini = '// === INICIO COMPARACION_FLORACION_NDVI (generado) ===\n'
    marcador_fin = '// === FIN COMPARACION_FLORACION_NDVI (generado) ===\n'
    patron = re.compile(re.escape(marcador_ini) + '.*?' + re.escape(marcador_fin), re.S)
    bloque_completo = marcador_ini + bloque + marcador_fin
    if patron.search(html):
        html_nuevo = patron.sub(bloque_completo, html)
    else:
        ancla = re.search(r'const PLUVIOSIDAD_MENSUAL = \{.*?\};\n', html, re.S)
        if not ancla:
            raise SystemExit('No se encontro PLUVIOSIDAD_MENSUAL en el HTML -- revisar a mano.')
        pos = ancla.end()
        html_nuevo = html[:pos] + bloque_completo + html[pos:]

    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html_nuevo)
    print(f'Inyectado en {HTML_PATH}')


if __name__ == '__main__':
    main()
