#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agregar_estaciones_julio2026_a_consolidada.py — suma a la tabla consolidada los datos
diarios reales de julio 2026 de 4 estaciones (3 nuevas + 1 ya presente que se extiende)
que llegaron en 'datos/LAS ESTACIONES-PRECIPITACION.xlsx' (reporte DMC/INIA, columna
"Suma Diaria" de precipitación cada 6 horas).

Regla de lectura del Excel: "s/p" en la columna Diaria = sin precipitación, se guarda
como 0.0 (dato real confirmado, no ausencia de dato). "." en la columna Diaria = sin
observación ese día, se descarta (no se inventa un cero).

Caldera (270008) ya existía en la tabla consolidada vía API DMC pero solo hasta
2025-12-31 -- este script la EXTIENDE con julio 2026. Los Acacios (300059), Copiapó
Universidad de Atacama (270009) y Copiapó Fundación Frutícola (270016) son estaciones
nuevas en la tabla.
"""

import csv
import sqlite3
import openpyxl

DB = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/fuentes/pluviosidad_diaria_consolidada.sqlite"
CSV_OUT = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/fuentes/pluviosidad_diaria_estaciones_julio2026.csv"
XLSX = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/datos/LAS ESTACIONES-PRECIPITACION.xlsx"

MESES = {"Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
         "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12}


def parse_hoja(ws):
    filas = list(ws.iter_rows(values_only=True))
    etiquetas = filas[0]
    valores = filas[1]

    def valor_de(etiqueta):
        idx = etiquetas.index(etiqueta)
        return valores[idx]

    codigo = valor_de("Código Nacional")
    nombre = valor_de("Nombre")
    lat_raw = valor_de("Latitud")
    lon_raw = valor_de("Longitud")
    lat = float(lat_raw) / 100000
    lon = float(lon_raw) / 100000

    mes_txt = None
    for row in filas:
        for cell in row:
            if isinstance(cell, str):
                for nombre_mes in MESES:
                    if cell.strip().startswith(nombre_mes + " de "):
                        mes_txt = cell.strip()
                        break
        if mes_txt:
            break
    partes = mes_txt.split(" de ")
    mes = MESES[partes[0]]
    anio_texto = partes[1]
    anio = int(anio_texto)

    dias = []
    for row in filas:
        dia = row[0]
        if not isinstance(dia, int):
            continue
        suma = row[5]
        if suma == "s/p":
            valor = 0.0
        elif suma in (None, ".", "") or not isinstance(suma, (int, float, str)):
            continue
        else:
            try:
                valor = float(str(suma).replace(",", "."))
            except ValueError:
                continue
            if suma == ".":
                continue
        dias.append((dia, valor))

    return codigo, nombre, lat, lon, mes, anio, anio_texto, dias


def main():
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    hojas = [parse_hoja(ws) for ws in wb.worksheets if ws.title != "Hoja1"]

    # "Julio de 202" en la hoja de Caldera -- el propio archivo de origen viene con el
    # año truncado (falta el ultimo digito). Se corrige por consenso con las otras 3
    # hojas del mismo reporte (mismo dia, mismo mes, todas dicen 2026), no se inventa.
    anios_validos = [anio for (*_, anio, anio_texto, _) in hojas if len(anio_texto) == 4]
    anio_correcto = max(set(anios_validos), key=anios_validos.count) if anios_validos else None
    for i, (codigo, nombre, lat, lon, mes, anio, anio_texto, dias) in enumerate(hojas):
        if len(anio_texto) != 4 and anio_correcto is not None:
            print(f"AVISO: año truncado '{anio_texto}' en hoja de {nombre} (codigo {codigo}) -- "
                  f"corregido a {anio_correcto} por consenso con las otras hojas del mismo reporte")
            hojas[i] = (codigo, nombre, lat, lon, mes, anio_correcto, anio_texto, dias)

    filas_csv = []
    for codigo, nombre, lat, lon, mes, anio, anio_texto, dias in hojas:
        etiqueta = f"{nombre} (DMC/INIA {codigo})"
        for dia, valor in dias:
            fecha = f"{anio:04d}-{mes:02d}-{dia:02d}"
            filas_csv.append((fecha, etiqueta, lat, lon, valor, f"Anuario/reporte estacion real, codigo {codigo}"))
        print(f"{etiqueta}: {len(dias)} dias reales de {partes_debug(mes, anio)}")

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fecha", "localidad", "lat", "lon", "lluvia_mm", "fuente"])
        w.writerows(filas_csv)
    print(f"Guardado CSV: {CSV_OUT} ({len(filas_csv)} filas)")

    con = sqlite3.connect(DB)
    # evitar duplicados si el script se corre mas de una vez: borrar antes de re-insertar
    fechas_min = min(f for (f, *_ ) in filas_csv)
    fechas_max = max(f for (f, *_ ) in filas_csv)
    localidades = sorted(set(l for (_, l, *_ ) in filas_csv))
    for loc in localidades:
        con.execute(
            "DELETE FROM pluviosidad_diaria WHERE localidad = ? AND fecha BETWEEN ? AND ?",
            (loc, fechas_min, fechas_max),
        )
    con.executemany(
        "INSERT INTO pluviosidad_diaria (fecha, localidad, lat, lon, lluvia_mm, fuente, tipo_fuente) VALUES (?,?,?,?,?,?,?)",
        [(f, l, la, lo, mm, fu, "estacion_real") for (f, l, la, lo, mm, fu) in filas_csv],
    )
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM pluviosidad_diaria").fetchone()[0]
    print(f"Total filas en la tabla consolidada tras sumar julio 2026: {total}")
    con.close()


def partes_debug(mes, anio):
    return f"{mes:02d}/{anio}"


if __name__ == "__main__":
    main()
