#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consolidar_pluviosidad_diaria.py — junta las dos fuentes de pluviosidad diaria real
(CR2 estaciones reales + Open-Meteo/ERA5-Land) en UNA sola tabla consultable.

Se eligió SQLite (un solo archivo .db, una tabla) en vez de pegar los dos CSV en uno
solo: juntos pesan ~368MB en texto plano, y una tabla real con índice por fecha y
localidad se puede consultar sin cargar todo el archivo en memoria -- justo lo que
pidió Alexis ("una tabla única que podamos consultar para diversas localidades").
Los CSV originales quedan intactos como respaldo/trazabilidad de cada fuente.
"""

import csv
import sqlite3

CR2_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/fuentes/pluviosidad_diaria_cr2_estaciones_reales.csv"
OPENMETEO_CSV = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/fuentes/pluviosidad_diaria_gyriosomus_openmeteo.csv"
DB = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/investigacion/fuentes/pluviosidad_diaria_consolidada.sqlite"


def cargar(con, ruta, tipo_fuente):
    cur = con.cursor()
    with open(ruta, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # encabezado
        lote = []
        total = 0
        for fila in reader:
            fecha, localidad, lat, lon, lluvia_mm, fuente = fila
            lote.append((fecha, localidad, float(lat), float(lon), float(lluvia_mm), fuente, tipo_fuente))
            if len(lote) >= 50000:
                cur.executemany(
                    "INSERT INTO pluviosidad_diaria (fecha, localidad, lat, lon, lluvia_mm, fuente, tipo_fuente) VALUES (?,?,?,?,?,?,?)",
                    lote,
                )
                total += len(lote)
                lote = []
        if lote:
            cur.executemany(
                "INSERT INTO pluviosidad_diaria (fecha, localidad, lat, lon, lluvia_mm, fuente, tipo_fuente) VALUES (?,?,?,?,?,?,?)",
                lote,
            )
            total += len(lote)
    con.commit()
    print(f"{ruta.split('/')[-1]}: {total} filas cargadas ({tipo_fuente})")


def main():
    con = sqlite3.connect(DB)
    con.execute("DROP TABLE IF EXISTS pluviosidad_diaria")
    con.execute("""
        CREATE TABLE pluviosidad_diaria (
            fecha TEXT NOT NULL,
            localidad TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            lluvia_mm REAL NOT NULL,
            fuente TEXT NOT NULL,
            tipo_fuente TEXT NOT NULL
        )
    """)

    cargar(con, CR2_CSV, "estacion_real")
    cargar(con, OPENMETEO_CSV, "reanalisis_erA5land")

    print("Creando índices...")
    con.execute("CREATE INDEX idx_fecha ON pluviosidad_diaria(fecha)")
    con.execute("CREATE INDEX idx_localidad ON pluviosidad_diaria(localidad)")
    con.execute("CREATE INDEX idx_lat_lon ON pluviosidad_diaria(lat, lon)")
    con.commit()

    total = con.execute("SELECT COUNT(*) FROM pluviosidad_diaria").fetchone()[0]
    localidades = con.execute("SELECT COUNT(DISTINCT localidad) FROM pluviosidad_diaria").fetchone()[0]
    rango = con.execute("SELECT MIN(fecha), MAX(fecha) FROM pluviosidad_diaria").fetchone()
    por_tipo = con.execute("SELECT tipo_fuente, COUNT(*) FROM pluviosidad_diaria GROUP BY tipo_fuente").fetchall()

    print(f"\nTotal filas en la tabla consolidada: {total}")
    print(f"Localidades/estaciones distintas: {localidades}")
    print(f"Rango de fechas: {rango[0]} a {rango[1]}")
    for tipo, n in por_tipo:
        print(f"  {tipo}: {n}")

    con.close()
    print(f"\nGuardado en: {DB}")


if __name__ == "__main__":
    main()
