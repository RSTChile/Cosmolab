#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extraer_cr2_diario_zona_gyriosomus.py — extrae del dataset real cr2_prDaily_2018
(descargado por Alexis, 874 estaciones, precipitación diaria real 1900-marzo 2018)
solo las estaciones dentro de la franja de distribución de Gyriosomus (25-34°S), y
las agrega a la tabla consolidada de pluviosidad diaria (mismo esquema que la parte
ya armada con Open-Meteo: fecha, localidad, lat, lon, lluvia_mm, fuente).
"""

import csv

DIR_CR2 = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/cr2_prDaily_2018"
ARCHIVO_DATOS = f"{DIR_CR2}/cr2_prDaily_2018.txt"
ARCHIVO_ESTACIONES = f"{DIR_CR2}/cr2_prDaily_2018_stations.txt"

SALIDA_CR2 = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/fuentes/pluviosidad_diaria_cr2_estaciones_reales.csv"
SALIDA_RESUMEN = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/fuentes/pluviosidad_diaria_cr2_estaciones_reales_resumen.csv"

LAT_MIN, LAT_MAX = -34.0, -25.0


def parse_val(v):
    v = v.strip()
    if v in ("-9999", "", "-", "NA", "NaN"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main():
    estaciones = {}
    with open(ARCHIVO_ESTACIONES, encoding="latin-1") as f:
        for r in csv.DictReader(f):
            try:
                lat = float(r["latitud"])
            except (KeyError, ValueError):
                continue
            if LAT_MIN <= lat <= LAT_MAX:
                # el archivo de datos rellena el codigo a 8 digitos con ceros a la
                # izquierda (ej "01000005"); el archivo de estaciones NO (ej
                # "1000005") -- sin este ajuste solo calzan por azar los codigos que
                # ya tenian 8 digitos (9 de 271, el bug real de la primera pasada).
                codigo = r["codigo_estacion"].zfill(8)
                estaciones[codigo] = {
                    "nombre": r["nombre"],
                    "lat": lat,
                    "lon": float(r["longitud"]),
                }
    print(f"{len(estaciones)} estaciones CR2 reales dentro de {LAT_MIN} a {LAT_MAX} lat")

    with open(ARCHIVO_DATOS, encoding="latin-1", newline="") as fdatos:
        reader = csv.reader(fdatos)
        header = next(reader)
        codigos = header[1:]

        # indices (dentro de la fila) de las columnas que nos interesan
        indices_interes = [i for i, c in enumerate(codigos) if c in estaciones]
        print(f"{len(indices_interes)} columnas coinciden en el archivo de datos")

        for _ in range(14):
            next(reader)  # saltar las 14 filas de metadata restantes

        contador_por_estacion = {c: 0 for c in estaciones}
        total_filas = 0
        with open(SALIDA_CR2, "w", newline="", encoding="utf-8") as fsal:
            w = csv.writer(fsal)
            w.writerow(["fecha", "localidad", "lat", "lon", "lluvia_mm", "fuente"])
            for fila in reader:
                if not fila or not fila[0]:
                    continue
                fecha = fila[0]
                for idx in indices_interes:
                    val = parse_val(fila[1 + idx])
                    if val is None:
                        continue
                    codigo = codigos[idx]
                    info = estaciones[codigo]
                    w.writerow([fecha, info["nombre"], info["lat"], info["lon"], val,
                                f"CR2 estacion real (diario, codigo {codigo})"])
                    contador_por_estacion[codigo] += 1
                    total_filas += 1

    with open(SALIDA_RESUMEN, "w", newline="", encoding="utf-8") as fres:
        w = csv.writer(fres)
        w.writerow(["codigo_estacion", "nombre", "lat", "lon", "dias_con_dato_real"])
        for codigo, info in sorted(estaciones.items(), key=lambda kv: -kv[1]["lat"]):
            w.writerow([codigo, info["nombre"], info["lat"], info["lon"], contador_por_estacion[codigo]])

    print(f"\nTotal filas escritas: {total_filas}")
    print(f"Guardado en: {SALIDA_CR2}")
    print(f"Resumen por estacion: {SALIDA_RESUMEN}")


if __name__ == "__main__":
    main()
