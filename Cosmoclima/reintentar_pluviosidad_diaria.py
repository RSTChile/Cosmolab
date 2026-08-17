#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reintenta SOLO las localidades que quedaron en ERROR (429 Too Many Requests)
en la primera pasada de obtener_pluviosidad_diaria_openmeteo.py -- con un ritmo
mucho mas conservador (Open-Meteo sin llave tiene limite de requests/minuto, no
solo de requests/dia)."""

import csv
import json
import time
import urllib.request
import urllib.parse
from datetime import date

RESUMEN = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/fuentes/pluviosidad_diaria_gyriosomus_openmeteo_resumen.csv"
SALIDA = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/fuentes/pluviosidad_diaria_gyriosomus_openmeteo.csv"

DESDE = "1966-01-01"
HASTA = date.today().isoformat()


def pendientes():
    filas = list(csv.DictReader(open(RESUMEN, encoding="utf-8")))
    return [(r["localidad"], float(r["lat"]), float(r["lon"])) for r in filas if r["estado"] == "ERROR"]


def traer_dia_a_dia(lat, lon, desde, hasta, reintentos=5):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": desde, "end_date": hasta,
        "daily": "precipitation_sum", "timezone": "America/Santiago",
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
            espera = 15 * (intento + 1)
            print(f"    reintento {intento+1}/{reintentos} tras {espera}s ({e})")
            time.sleep(espera)
    raise RuntimeError(f"fallo tras {reintentos} intentos: {ultimo_error}")


def main():
    locs = pendientes()
    print(f"{len(locs)} localidades pendientes de reintentar")
    resueltas = {}
    with open(SALIDA, "a", newline="", encoding="utf-8") as fsal:
        w = csv.writer(fsal)
        for i, (nombre, lat, lon) in enumerate(locs, 1):
            try:
                fechas, valores = traer_dia_a_dia(lat, lon, DESDE, HASTA)
            except RuntimeError as e:
                print(f"[{i}/{len(locs)}] {nombre} -- SIGUE FALLANDO: {e}")
                continue
            n = 0
            for f, v in zip(fechas, valores):
                if v is None:
                    continue
                w.writerow([f, nombre, lat, lon, v, "Open-Meteo (ERA5-Land, reanalisis)"])
                n += 1
            print(f"[{i}/{len(locs)}] {nombre} -- OK, {n} dias")
            resueltas[(nombre, lat, lon)] = n
            fsal.flush()
            time.sleep(8)

    # actualizar el resumen
    filas = list(csv.DictReader(open(RESUMEN, encoding="utf-8")))
    for r in filas:
        key = (r["localidad"], float(r["lat"]), float(r["lon"]))
        if key in resueltas:
            r["dias_con_dato"] = str(resueltas[key])
            r["estado"] = "OK (reintento)"
    with open(RESUMEN, "w", newline="", encoding="utf-8") as fres:
        w = csv.DictWriter(fres, fieldnames=["localidad", "lat", "lon", "dias_pedidos", "dias_con_dato", "estado"])
        w.writeheader()
        w.writerows(filas)
    print(f"\nResueltas en este reintento: {len(resueltas)}/{len(locs)}")


if __name__ == "__main__":
    main()
