#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agregar_dmc_a_consolidada.py — suma a la tabla consolidada las 2 estaciones reales
de la DMC que caen dentro de la zona de Gyriosomus Y que en la copia de CR2 quedaron
vacías (0 días con dato real): Caldera (270008) y La Serena/La Florida (290004).
Usa el campo "total" del JSON de getAguaCaidaDiaria (exige TODAS las observaciones
del día disponibles -- más estricto y más honesto que "parcial", que cuenta días
con observaciones incompletas).

Requiere las variables de entorno DMC_USUARIO y DMC_TOKEN (no se hardcodea ninguna
credencial en este archivo).
"""

import csv
import json
import os
import sqlite3
import sys
import urllib.request
import urllib.parse

DB = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/fuentes/pluviosidad_diaria_consolidada.sqlite"
CSV_DMC = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/fuentes/pluviosidad_diaria_dmc_estaciones_reales.csv"

ESTACIONES = [
    (270008, "Desierto de Atacama, Caldera (DMC)"),
    (290004, "La Florida, La Serena (DMC)"),
]


def traer(codigo, usuario, token):
    params = {"usuario": usuario, "token": token}
    url = f"https://climatologia.meteochile.gob.cl/application/serviciosb/getAguaCaidaDiaria/{codigo}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    usuario = os.environ.get("DMC_USUARIO")
    token = os.environ.get("DMC_TOKEN")
    if not usuario or not token:
        print("Falta DMC_USUARIO/DMC_TOKEN en el entorno", file=sys.stderr)
        sys.exit(1)

    filas = []
    for codigo, etiqueta in ESTACIONES:
        d = traer(codigo, usuario, token)
        info = d["datosEstacion"]
        lat, lon = float(info["latitud"]), float(info["longitud"])
        n = 0
        for anio, meses in d["datosHistoricos"]["diaria"].items():
            for mes, dias in meses.items():
                for dia, valores in dias.items():
                    total = valores.get("total")
                    if total is None:
                        continue
                    fecha = f"{int(anio):04d}-{int(mes):02d}-{int(dia):02d}"
                    filas.append((fecha, etiqueta, lat, lon, float(total), f"DMC estacion real (diario, codigo {codigo})"))
                    n += 1
        print(f"{etiqueta} (codigo {codigo}): {n} dias reales")

    with open(CSV_DMC, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fecha", "localidad", "lat", "lon", "lluvia_mm", "fuente"])
        w.writerows(filas)
    print(f"Guardado CSV: {CSV_DMC} ({len(filas)} filas)")

    con = sqlite3.connect(DB)
    con.executemany(
        "INSERT INTO pluviosidad_diaria (fecha, localidad, lat, lon, lluvia_mm, fuente, tipo_fuente) VALUES (?,?,?,?,?,?,?)",
        [(f, l, la, lo, mm, fu, "estacion_real") for (f, l, la, lo, mm, fu) in filas],
    )
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM pluviosidad_diaria").fetchone()[0]
    print(f"Total filas en la tabla consolidada tras sumar DMC: {total}")
    con.close()


if __name__ == "__main__":
    main()
