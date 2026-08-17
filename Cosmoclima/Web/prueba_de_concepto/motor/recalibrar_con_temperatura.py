#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recalibrar_con_temperatura.py -- Parte A del plan "recalcular floración"
(09-ago-2026, a pedido de Alexis, alcance acotado tras investigar qué datos
reales existen -- ver /Users/alexis/.claude/plans/majestic-whistling-canyon.md
y investigacion/curva_empirica_gyriosomus.md, ronda v3).

Pregunta: ¿agregar temperatura real (Tmax de TEMPERATURA_DIARIA_ZHCS) como
SEGUNDO predictor de floración, junto al pico mensual de lluvia que ya usa
objetivoFloracionEmpirico(), mejora la exactitud contra los 23 años
documentados? Y -- a diferencia de v1/v2 -- ¿mejora también en validación
cruzada leave-one-out (LOOCV), no solo in-sample? v1/v2 nunca se validaron
así (confirmado, no hay ninguna mención de hold-out en curva_empirica_
gyriosomus.md ni en el código) -- cierra la Fase D.3 que quedó pendiente
del plan de granularidad anterior.

Ventana de temperatura elegida: Tmax promedio de los 90 días QUE EMPIEZAN
el 1° del mes de pico de lluvia (la ventana de CRECIMIENTO de la floración,
no antes de la lluvia) -- coherente con el mismo horizonte de 90 días que
ya usa el ascenso de computeFloracion() en el HTML (rise time real, no un
número nuevo inventado para esto).

No se usa NASA POWER `agregar()` ni sklearn/scipy (no instalados) -- se
implementa regresión logística a mano (Newton-Raphson/IRLS) para tener
control total y poder correr LOOCV (23 refits) sin depender de una
biblioteca externa nueva.

Criterio de adopción declarado ANTES de ver el resultado (para no elegir
el número que se ve mejor después): el modelo con temperatura se adopta
SOLO si su exactitud LOOCV iguala o supera la del modelo solo-lluvia.
"""
import json
import os
import re
import sys
from datetime import date, timedelta

import numpy as np

CARPETA = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(CARPETA, '..', 'sim-cosmoclima.html')
CSV_GROUND_TRUTH = os.path.join(CARPETA, '..', '..', '..', 'investigacion', 'fuentes',
                                  'curva_empirica_lluvia_floracion_gyriosomus.csv')
JSON_SALIDA = os.path.join(CARPETA, 'recalibracion_temperatura_resumen.json')
CSV_SALIDA = os.path.join(CARPETA, 'recalibracion_temperatura_por_anio.csv')

MESES_DIAS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # calendario simplificado del instrumento (Feb=28 fijo)

EMP_B0_ACTUAL, EMP_B1_ACTUAL = -1.2123, 0.0185  # coeficientes vigentes hoy en el HTML (solo lluvia)


def leer_const_json(html, nombre):
    m = re.search(r'const ' + nombre + r' = (\{.*?\});\n', html, re.S)
    if not m:
        raise SystemExit(f'No se encontró {nombre} en el HTML.')
    return json.loads(m.group(1))


def pico_mensual_y_mes(pluviosidad_mensual, anio):
    """Replica picoMensualAnio() del HTML, pero además devuelve el mes (1-12)
    del pico -- el HTML solo necesita el valor, acá necesitamos también el
    mes para definir la ventana de temperatura."""
    mejor_valor, mejor_mes = None, None
    for mes in range(1, 13):
        clave = f'{anio:04d}-{mes:02d}'
        v = pluviosidad_mensual.get(clave)
        if v is not None and (mejor_valor is None or v > mejor_valor):
            mejor_valor, mejor_mes = v, mes
    return mejor_valor, mejor_mes


def ventana_tmax_promedio(temp_diaria, anio, mes_pico, dias=90):
    """Tmax promedio de los `dias` días calendario reales que empiezan el
    1 del mes de pico (calendario real, con años bisiestos -- TEMPERATURA_
    DIARIA_ZHCS viene de NASA POWER con fechas reales, a diferencia del
    calendario interno simplificado del instrumento). Devuelve (promedio,
    cobertura_frac)."""
    inicio = date(anio, mes_pico, 1)
    valores = []
    for i in range(dias):
        fecha = (inicio + timedelta(days=i)).isoformat()
        info = temp_diaria.get(fecha)
        if info and info.get('tmax') is not None:
            valores.append(info['tmax'])
    cobertura = len(valores) / dias
    promedio = round(sum(valores) / len(valores), 3) if valores else None
    return promedio, cobertura


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_logistic_irls(X, y, iters=100, ridge=1e-6):
    """Newton-Raphson (IRLS) para regresión logística. `ridge` es una
    regularización mínima (no cero) solo para evitar que LOOCV con
    separación cuasi-perfecta en submuestras de 22 puntos haga diverger
    la matriz -- declarado, no se usa para "mejorar" el ajuste principal."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(iters):
        z = X @ beta
        prob = sigmoid(z)
        W = prob * (1 - prob)
        W = np.clip(W, 1e-6, None)
        XtW = X.T * W
        hess = XtW @ X + ridge * np.eye(p)
        grad = X.T @ (y - prob)
        delta = np.linalg.solve(hess, grad)
        beta = beta + delta
        if np.max(np.abs(delta)) < 1e-8:
            break
    return beta


def exactitud(X, y, beta):
    pred = (sigmoid(X @ beta) > 0.5).astype(int)
    return float(np.mean(pred == y))


def loocv(X, y):
    n = len(y)
    aciertos = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta = fit_logistic_irls(X[mask], y[mask])
        pred = int(sigmoid(X[i] @ beta) > 0.5)
        aciertos += int(pred == y[i])
    return aciertos / n


def main():
    with open(HTML_PATH, encoding='utf-8') as f:
        html = f.read()
    pluviosidad_mensual = leer_const_json(html, 'PLUVIOSIDAD_MENSUAL')
    temp_diaria = leer_const_json(html, 'TEMPERATURA_DIARIA_ZHCS')

    filas = []
    with open(CSV_GROUND_TRUTH, encoding='utf-8') as f:
        import csv
        for row in csv.DictReader(f):
            if row['floracion_documentada'] not in ('0', '1'):
                continue
            anio = int(row['anio'])
            pico, mes_pico = pico_mensual_y_mes(pluviosidad_mensual, anio)
            if pico is None:
                print(f'AVISO: {anio} sin dato real de lluvia -- se excluye.')
                continue
            tmax_prom, cobertura = ventana_tmax_promedio(temp_diaria, anio, mes_pico)
            filas.append({
                'anio': anio, 'floracion_documentada': int(row['floracion_documentada']),
                'pico_mensual_mm': pico, 'mes_pico': mes_pico,
                'tmax_ventana90d': tmax_prom, 'cobertura_temp': round(cobertura, 3),
            })

    con_temp = [f for f in filas if f['tmax_ventana90d'] is not None]
    sin_temp = [f for f in filas if f['tmax_ventana90d'] is None]
    if sin_temp:
        print(f'AVISO: {len(sin_temp)} años sin cobertura de temperatura (excluidos del modelo con temp): '
              f'{[f["anio"] for f in sin_temp]}')

    print(f'{len(filas)} años documentados totales, {len(con_temp)} con temperatura real disponible.\n')

    y = np.array([f['floracion_documentada'] for f in con_temp], dtype=float)
    X_lluvia = np.column_stack([np.ones(len(con_temp)), [f['pico_mensual_mm'] for f in con_temp]])
    X_ambas = np.column_stack([np.ones(len(con_temp)), [f['pico_mensual_mm'] for f in con_temp],
                                 [f['tmax_ventana90d'] for f in con_temp]])

    beta_lluvia = fit_logistic_irls(X_lluvia, y)
    beta_ambas = fit_logistic_irls(X_ambas, y)

    acc_in_lluvia = exactitud(X_lluvia, y, beta_lluvia)
    acc_in_ambas = exactitud(X_ambas, y, beta_ambas)
    acc_loocv_lluvia = loocv(X_lluvia, y)
    acc_loocv_ambas = loocv(X_ambas, y)

    print('=== Modelo 1: solo lluvia (mismo predictor que hoy) ===')
    print(f'Coeficientes reajustados acá: B0={beta_lluvia[0]:.4f}, B1={beta_lluvia[1]:.4f}')
    print(f'  (vigentes en el HTML hoy:    B0={EMP_B0_ACTUAL}, B1={EMP_B1_ACTUAL} '
          f'-- {"cercano, buena señal de que el método coincide" if abs(beta_lluvia[0]-EMP_B0_ACTUAL)<0.3 else "DIFERENTE, revisar por qué"})')
    print(f'Exactitud in-sample: {acc_in_lluvia*100:.1f}%')
    print(f'Exactitud LOOCV:     {acc_loocv_lluvia*100:.1f}%')

    print('\n=== Modelo 2: lluvia + Tmax (ventana de 90 días desde el mes pico) ===')
    print(f'Coeficientes: B0={beta_ambas[0]:.4f}, B_lluvia={beta_ambas[1]:.4f}, B_tmax={beta_ambas[2]:.4f}')
    print(f'Exactitud in-sample: {acc_in_ambas*100:.1f}%')
    print(f'Exactitud LOOCV:     {acc_loocv_ambas*100:.1f}%')

    adoptar = acc_loocv_ambas >= acc_loocv_lluvia
    print(f'\n=== Criterio de adopción (declarado antes de ver el resultado): '
          f'LOOCV_temp >= LOOCV_lluvia ===')
    print(f'{acc_loocv_ambas*100:.1f}% >= {acc_loocv_lluvia*100:.1f}% -> '
          f'{"SE ADOPTA el modelo con temperatura" if adoptar else "NO se adopta -- se mantiene el modelo actual (solo lluvia), resultado negativo documentado"}')

    # guardar tabla por año con ambas predicciones, para auditoría completa
    for f_ in con_temp:
        Xi_l = np.array([1, f_['pico_mensual_mm']])
        Xi_a = np.array([1, f_['pico_mensual_mm'], f_['tmax_ventana90d']])
        f_['prob_solo_lluvia'] = round(float(sigmoid(Xi_l @ beta_lluvia)), 4)
        f_['prob_lluvia_temp'] = round(float(sigmoid(Xi_a @ beta_ambas)), 4)

    with open(CSV_SALIDA, 'w', encoding='utf-8', newline='') as f_out:
        import csv
        cols = ['anio', 'floracion_documentada', 'pico_mensual_mm', 'mes_pico',
                'tmax_ventana90d', 'cobertura_temp', 'prob_solo_lluvia', 'prob_lluvia_temp']
        w = csv.DictWriter(f_out, fieldnames=cols)
        w.writeheader()
        w.writerows(con_temp)
    print(f'\nTabla por año guardada en {CSV_SALIDA}')

    resumen = {
        'n_anios_total_documentados': len(filas),
        'n_anios_con_temperatura': len(con_temp),
        'anios_excluidos_sin_temp': [f['anio'] for f in sin_temp],
        'modelo_solo_lluvia': {
            'B0': round(float(beta_lluvia[0]), 4), 'B1_lluvia': round(float(beta_lluvia[1]), 4),
            'exactitud_in_sample': round(acc_in_lluvia, 4), 'exactitud_loocv': round(acc_loocv_lluvia, 4),
            'coeficientes_vigentes_html': {'B0': EMP_B0_ACTUAL, 'B1': EMP_B1_ACTUAL},
        },
        'modelo_lluvia_mas_temperatura': {
            'B0': round(float(beta_ambas[0]), 4), 'B1_lluvia': round(float(beta_ambas[1]), 4),
            'B2_tmax90d': round(float(beta_ambas[2]), 4),
            'exactitud_in_sample': round(acc_in_ambas, 4), 'exactitud_loocv': round(acc_loocv_ambas, 4),
        },
        'criterio_adopcion': 'LOOCV_temp >= LOOCV_lluvia',
        'se_adopta_modelo_con_temperatura': bool(adoptar),
        'ventana_temperatura': '90 dias reales desde el dia 1 del mes de pico de lluvia (Tmax promedio, NASA POWER)',
    }
    with open(JSON_SALIDA, 'w', encoding='utf-8') as f_out:
        json.dump(resumen, f_out, ensure_ascii=False, indent=2)
    print(f'Resumen guardado en {JSON_SALIDA}')


if __name__ == '__main__':
    main()
