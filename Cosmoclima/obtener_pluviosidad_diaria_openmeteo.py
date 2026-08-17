#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obtener_pluviosidad_diaria_openmeteo.py — arma la tabla consolidada de
pluviosidad DIARIA real para toda la zona de distribución de Gyriosomus.

Fuente: Open-Meteo Historical Weather API (open-meteo.com/en/docs/historical-weather-api),
que corre sobre ERA5-Land (reanálisis, ~0.1° / 9km) — gratis, sin llave, sin registro,
cuota generosa (10.000 llamadas/día). NO es estación real en tierra (es reanálisis
satelital/modelo, igual naturaleza que NASA POWER, ya usado en este proyecto) — se
declara así, no se disfraza de estación.

Localidades: las 78 localidades REALES distintas de la Tabla S1 (Anguita-Salinas et al.
2026, especímenes de Gyriosomus con coordenadas), cubriendo el rango completo del
género (~24.8°S a 34.2°S).

Se investigaron también CR2 diario (cr2_prDaily_2018) y la API de la DMC
(getAguaCaidaDiaria) como fuentes de ESTACIÓN real — ambas quedaron fuera de este
script porque requieren que una persona se registre (CR2: cuenta en el sitio; DMC:
usuario + token) — no es algo que se pueda automatizar sin que Alexis lo haga él mismo.
Quedan documentadas como pendiente en investigacion/pluviosidad_diaria_consolidada.md.
"""

import csv
import json
import time
import urllib.request
import urllib.parse
from datetime import date

TABLA_S1 = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/Web/prueba_de_concepto/datos_fuentes/tabla_s1_anguita_salinas_2026.csv"
SALIDA = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/fuentes/pluviosidad_diaria_gyriosomus_openmeteo.csv"
RESUMEN = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/fuentes/pluviosidad_diaria_gyriosomus_openmeteo_resumen.csv"

DESDE = "1966-01-01"
# Open-Meteo RECHAZA (400) una end_date en el futuro -- a diferencia de NASA POWER,
# que silenciosamente devuelve vacio. Hay que pedir hasta hoy de verdad, no 2027.
HASTA = date.today().isoformat()


def localidades_distintas():
    filas = list(csv.DictReader(open(TABLA_S1, encoding="utf-8")))
    vistas = {}
    for r in filas:
        try:
            lat = round(float(r["lat"]), 2)
            lon = round(float(r["lon"]), 2)
        except (KeyError, ValueError):
            continue
        clave = (lat, lon)
        if clave not in vistas:
            vistas[clave] = r["localidad"]
    # ordenar de norte a sur para que el progreso se lea de corrido
    return sorted(
        [(nombre, lat, lon) for (lat, lon), nombre in vistas.items()],
        key=lambda x: -x[1],
    )


def traer_dia_a_dia(lat, lon, desde, hasta, reintentos=3):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": desde,
        "end_date": hasta,
        "daily": "precipitation_sum",
        "timezone": "America/Santiago",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    ultimo_error = None
    for intento in range(reintentos):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["daily"]["time"], data["daily"]["precipitation_sum"]
        except Exception as e:  # noqa: BLE001
            ultimo_error = e
            time.sleep(2 * (intento + 1))
    raise RuntimeError(f"fallo tras {reintentos} intentos: {ultimo_error}")


def main():
    locs = localidades_distintas()
    print(f"{len(locs)} localidades reales distintas (Tabla S1), {DESDE} a {HASTA}")

    filas_totales = 0
    resumen_filas = []
    with open(SALIDA, "w", newline="", encoding="utf-8") as fsal:
        w = csv.writer(fsal)
        w.writerow(["fecha", "localidad", "lat", "lon", "lluvia_mm", "fuente"])
        for i, (nombre, lat, lon) in enumerate(locs, 1):
            t0 = time.time()
            try:
                fechas, valores = traer_dia_a_dia(lat, lon, DESDE, HASTA)
            except RuntimeError as e:
                print(f"[{i}/{len(locs)}] {nombre} ({lat},{lon}) -- ERROR: {e}")
                resumen_filas.append([nombre, lat, lon, 0, 0, "ERROR"])
                continue
            n_con_dato = 0
            for f, v in zip(fechas, valores):
                if v is None:
                    continue
                w.writerow([f, nombre, lat, lon, v, "Open-Meteo (ERA5-Land, reanalisis)"])
                n_con_dato += 1
            filas_totales += n_con_dato
            dt = time.time() - t0
            print(f"[{i}/{len(locs)}] {nombre} ({lat},{lon}) -- {n_con_dato} dias reales, {dt:.1f}s")
            resumen_filas.append([nombre, lat, lon, len(fechas), n_con_dato, "OK"])
            time.sleep(0.3)  # cortesia con la API gratuita, no tiene llave que la limite

    with open(RESUMEN, "w", newline="", encoding="utf-8") as fres:
        w = csv.writer(fres)
        w.writerow(["localidad", "lat", "lon", "dias_pedidos", "dias_con_dato", "estado"])
        w.writerows(resumen_filas)

    print(f"\nTotal filas escritas: {filas_totales}")
    print(f"Guardado en: {SALIDA}")
    print(f"Resumen por localidad: {RESUMEN}")


if __name__ == "__main__":
    main()
