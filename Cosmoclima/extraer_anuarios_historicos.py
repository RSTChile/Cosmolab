#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extraer_anuarios_historicos.py -- extrae precipitacion MENSUAL real de los 30
anuarios DMC 1979-2024 (datos/anuarios meteorológicos/) para la franja
Atacama-Coquimbo-N.Valparaíso (misma zona de Gyriosomus ya usada en el proyecto),
y la suma a investigacion/fuentes/precipitacion_mensual_dmc_anuario2025.csv (el
mismo archivo que ya trae 2025).

Principio de seguridad (a proposito, no relajar): el nombre de estacion tiene
que estar INMEDIATAMENTE junto a su propio par de filas "Total mensual"/"Máx en
24 hrs" en el texto extraido del PDF, en dos formatos ya verificados a mano:
  (a) "NOMBRE  ESTACION  : X" seguido de DATO/TOTAL/MAX (anuarios ~1988-2010)
  (b) "X Total mensual ..." / "X Máx en 24 hrs. ..." con el nombre pegado a
      cada fila (anuarios 1979, 2010)
Si una pagina trae los numeros en un bloque y los nombres de estacion en OTRO
bloque separado (ej. 2014, 2019, 2024 -- texto casi seguro de OCR sobre una
tabla escaneada, orden de nombres NO geografico, se verificado a mano que
queda desordenado) NO se intenta emparejar por posicion -- se marca esa
pagina como pendiente de revision visual y no se inventa la atribucion.

1972-1978 (sin capa de texto, escaneados) quedan fuera de este script --
requieren OCR aparte, documentado como pendiente.
"""
import csv
import glob
import os
import re
import sys
import unicodedata

import pypdf

CARPETA = os.path.dirname(os.path.abspath(__file__))
CARPETA_PDFS = os.path.join(CARPETA, 'datos', 'anuarios meteorológicos')
CSV_OUT = os.path.join(CARPETA, 'investigacion', 'fuentes', 'precipitacion_mensual_dmc_anuario2025.csv')
LOG_PENDIENTES = os.path.join(CARPETA, 'investigacion', 'fuentes', 'anuarios_historicos_paginas_pendientes.csv')

MESES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']

# Estaciones de la franja Gyriosomus (24.8S-34.2S) que ya reconocimos en el
# catastro de varios anuarios (1979, 1988, 2018, 2023, 2024) -- substrings
# normalizados (sin tilde, minuscula). Se usa contains(), no igualdad exacta,
# para tolerar variantes de nombre entre decadas ("Essco"/"Sendos"/"Fundo").
ESTACIONES_ZONA = [
    "caldera", "copiapo", "chamonate", "inca de oro", "los loros",
    "san felix", "transito escuela", "vallenar", "freirina", "huasco",
    "la serena", "la florida", "ovalle", "illapel", "vivero conaf",
    "andacollo", "hurtado", "el tangue", "pachingo", "cerrillos de tamaya",
    "los penones", "coiron", "chuchini", "combarbala", "puerto oscuro",
    "los nichos", "huaquen", "salamanca", "las vacas", "los vilos",
    "caimanes", "totoralillo", "trapiche longotoma", "casas de alicahue",
    "alicahue", "la ligua", "catapilco", "la canela fundo", "puchuncavi",
    "san felipe", "el tambo", "zapallar", "curimon", "vicuna", "vega fundo",
    "combarbal", "tal tal", "taltal", "chanaral", "paposo",
]

# Estaciones cuyo nombre solo, sin mas contexto, es ambiguo con lugares fuera
# de la zona -- se excluyen explicitamente aunque compartan substring.
# "tal tal" (Taltal, -25.4S, region de Antofagasta) SALIO de esta lista el
# 06-ago-2026: se habia excluido por error -- SI esta dentro de la franja
# Gyriosomus (24.8-34.2S), a solo ~49km de Paposo (Alexis lo noto al pedir
# revisar el problema de las especies del extremo norte).
EXCLUIR = ["mamina", "toconao", "visviri", "quillagua", "pisagua",
           "huatacondo"]


def normalizar(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    return s.lower().strip()


def es_zona(nombre_norm):
    if any(x in nombre_norm for x in EXCLUIR):
        return False
    return any(x in nombre_norm for x in ESTACIONES_ZONA)


def parsear_num(v):
    v = v.strip().replace(',', '.')
    if v in ('-', '.', '', 's/p', 'S/P', '...'):
        return None
    try:
        return float(v)
    except ValueError:
        return None


TOKEN_VALOR = r'(-|\.|s/p|S/P|[\d]+[.,]?[\d]*)'
FILA_12 = r'\s+'.join([TOKEN_VALOR] * 12) + r'(?:\s+' + TOKEN_VALOR + r')?'


def extraer_inline_bloques(texto, anio, fuente_pag):
    """Formato (a): 'NOMBRE ESTACION : X' + DATO/TOTAL/MAX."""
    filas = []
    patron = re.compile(
        r'NOMBRE\s+ESTACION\s*:\s*([A-ZÁÉÍÓÚÑÜ][A-ZÁÉÍÓÚÑÜ .\-]+?)\s*\n'
        r'DATO\s+ENE.*?\n'
        r'TOTAL\s+' + FILA_12 + r'.*?\n'
        r'MAX\s+' + FILA_12,
        re.S,
    )
    for m in patron.finditer(texto):
        nombre = ' '.join(m.group(1).split())
        nombre_norm = normalizar(nombre)
        if not es_zona(nombre_norm):
            continue
        valores = [parsear_num(x) for x in m.groups()[1:13]]
        for mes, val in zip(MESES, valores):
            if val is not None:
                filas.append((nombre, mes, anio, val, fuente_pag))
    return filas


def extraer_prefijo_filas(texto, anio, fuente_pag):
    """Formato (b): '<NOMBRE...> Total mensual <12 valores>' / '<...> Máx en 24 hrs. <12 valores>',
    con el nombre a veces partido en 2 lineas (se reconstruye)."""
    filas = []
    lineas = texto.split('\n')
    i = 0
    nombre_pendiente = ''
    while i < len(lineas) - 1:
        l1 = lineas[i]
        m1 = re.match(r'^(.*?)\s*Total\s+mensual\s+(.+)$', l1, re.I)
        if m1:
            nombre1 = m1.group(1).strip()
            resto1 = m1.group(2).strip()
            l2 = lineas[i + 1] if i + 1 < len(lineas) else ''
            m2 = re.match(r'^(.*?)\s*M[aá]x\.?\s+en\s+24\s*hrs\.?\s+(.+)$', l2, re.I)
            if m2:
                nombre2 = m2.group(1).strip()
                nombre_completo = ' '.join((nombre_pendiente + ' ' + nombre1 + ' ' + nombre2).split())
                nombre_norm = normalizar(nombre_completo)
                if es_zona(nombre_norm):
                    valores = re.findall(TOKEN_VALOR, resto1)[:12]
                    valores = [parsear_num(v) for v in valores]
                    if len(valores) == 12:
                        for mes, val in zip(MESES, valores):
                            if val is not None:
                                filas.append((nombre_completo, mes, anio, val, fuente_pag))
                nombre_pendiente = ''
                i += 2
                continue
        nombre_pendiente = ''
        i += 1
    return filas


def pagina_es_riesgosa(texto):
    """Detecta el formato (c): numeros en bloque, nombres en OTRO bloque
    separado (tipico de OCR sobre tabla escaneada, orden no confiable).
    Formato (b) seguro = el nombre viene pegado ANTES de 'Total mensual' en
    la MISMA linea (ej. 'COMBARBALÁ Total mensual ...').

    ★ FIX 06-ago-2026: antes esta funcion consideraba la PAGINA ENTERA segura
    con que UNA sola fila tuviera nombre pegado -- pero se encontro un caso
    real (anuario-2015.pdf, pagina 83) con una pagina HIBRIDA: las primeras
    10 estaciones (Visviri, Mamiña, Toconao, Tal Tal, Los Nichos, Hurtado,
    Coirón Retén, Combarbalá, Puerto Oscuro, Chuchiñi) sin nombre pegado
    (bloque de nombres suelto al final, orden NO confiable, verificado a
    mano), y las 3 siguientes (Huaquén Hacienda, El Trapiche Longotoma,
    Casas de Alicahue) SI con nombre pegado -- la version vieja de esta
    funcion veia esas 3 ultimas y daba la pagina entera por segura, así que
    las primeras 10 nunca se extrajeron NI quedaron en la lista de
    pendientes: se perdieron en silencio. Ahora: si CUALQUIER fila "Total
    mensual" de la pagina no tiene nombre pegado, la pagina completa queda
    marcada riesgosa (aunque partes ya se hayan podido extraer solas)."""
    if 'Total mensual' not in texto and 'Total Mensual' not in texto:
        return False
    if re.search(r'NOMBRE\s+ESTACION\s*:', texto):
        return False
    lineas_total = [l for l in texto.split('\n') if re.match(r'^(.*?)\s*Total\s+[Mm]ensual\s', l)]
    if not lineas_total:
        return False
    for l in lineas_total:
        m = re.match(r'^(.*?)\s*Total\s+[Mm]ensual\s', l)
        prefijo = m.group(1).strip()
        if not re.search(r'[A-Za-zÁÉÍÓÚÑ]{3,}', prefijo):
            return True  # al menos una fila sin nombre pegado -> pagina riesgosa
    return False


def procesar_pdf(path):
    anio_m = re.search(r'(\d{4})', os.path.basename(path))
    anio = int(anio_m.group(1))
    filas_zona = []
    paginas_riesgosas = []
    try:
        r = pypdf.PdfReader(path)
    except Exception as e:
        print(f'  ERROR abriendo {path}: {e}')
        return filas_zona, paginas_riesgosas
    n = len(r.pages)
    for i in range(n):
        try:
            t = r.pages[i].extract_text() or ''
        except Exception:
            continue
        if not t.strip():
            continue
        fuente_pag = f'{os.path.basename(path)}, pagina {i + 1}'
        filas_zona += extraer_inline_bloques(t, anio, fuente_pag)
        filas_zona += extraer_prefijo_filas(t, anio, fuente_pag)
        if pagina_es_riesgosa(t) and any(es_zona(normalizar(x)) for x in re.findall(r'[A-ZÁÉÍÓÚÑ][a-zA-ZÁÉÍÓÚÑáéíóúñ ]{3,}', t)):
            paginas_riesgosas.append((anio, fuente_pag))
    return filas_zona, paginas_riesgosas


def main():
    pdfs = sorted(glob.glob(os.path.join(CARPETA_PDFS, 'anuario-*.pdf')))
    todas_filas = []
    todas_riesgosas = []
    for path in pdfs:
        anio = re.search(r'(\d{4})', os.path.basename(path)).group(1)
        filas, riesgosas = procesar_pdf(path)
        # dedup exacto (misma estacion/mes/anio/valor/pagina -- un mismo bloque
        # puede calzar con los dos extractores si el formato es ambiguo)
        vistos = set()
        filas_dedup = []
        for f in filas:
            if f not in vistos:
                vistos.add(f)
                filas_dedup.append(f)
        print(f'{os.path.basename(path)}: {len(filas_dedup)} valores mensuales de zona, '
              f'{len(riesgosas)} paginas riesgosas (sin atribuir)')
        todas_filas += filas_dedup
        todas_riesgosas += riesgosas

    # cargar filas existentes (2025, ya cargado en sesion anterior) para no perderlas
    existentes = []
    if os.path.exists(CSV_OUT):
        with open(CSV_OUT, encoding='utf-8') as f:
            existentes = list(csv.DictReader(f))

    nuevas = []
    for nombre, mes, anio, val, fuente_pag in todas_filas:
        nuevas.append({
            'estacion': nombre, 'codigo': '', 'lat': '', 'lon': '', 'alt_m': '',
            'mes': MESES.index(mes) + 1, 'anio': anio, 'lluvia_mm': val,
            'dato_real': 'si', 'fuente': f'DMC Anuario histórico ({fuente_pag})',
        })

    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
        campos = ['estacion', 'codigo', 'lat', 'lon', 'alt_m', 'mes', 'anio', 'lluvia_mm', 'dato_real', 'fuente']
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(existentes)
        w.writerows(nuevas)

    with open(LOG_PENDIENTES, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['anio', 'pagina', 'motivo'])
        for anio, pag in todas_riesgosas:
            w.writerow([anio, pag, 'nombres de estacion en bloque separado de los datos (probable OCR de tabla escaneada) -- requiere revision visual, no se atribuyo por posicion'])

    print(f'\nTotal filas nuevas: {len(nuevas)}')
    print(f'Total filas en CSV final (con 2025 ya existente): {len(existentes) + len(nuevas)}')
    print(f'Paginas pendientes de revision visual: {len(todas_riesgosas)} -> {LOG_PENDIENTES}')


if __name__ == '__main__':
    main()
