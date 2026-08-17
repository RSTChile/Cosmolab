#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datos_clima.py — trae datos meteorológicos reales por coordenadas.

Fuente: NASA POWER (power.larc.nasa.gov). Gratis, sin clave, cobertura global,
resolución diaria desde 1981. Es reanálisis, no medición local: los valores
provienen de un modelo alimentado con satélite y estaciones, interpolado a una
grilla. Sirve para línea base y para forzar simulaciones; NO reemplaza a una
estación en terreno cuando lo que se necesita es el microclima de un sitio.

USO DESDE LA LÍNEA DE COMANDOS
------------------------------
    python3 datos_clima.py --lat -32.84 --lon -70.95 \
        --desde 2024-01-01 --hasta 2026-07-22

    python3 datos_clima.py --lat -32.84 --lon -70.95 \
        --desde 2020-01-01 --hasta 2026-07-22 --periodo mes

    Períodos disponibles: dia (por omisión), semana, mes, trimestre, anio

USO COMO BIBLIOTECA
-------------------
    from datos_clima import traer, agregar, escribir_csv
    filas = traer(-32.84, -70.95, "2024-01-01", "2026-07-22")
    meses = agregar(filas, "mes")

DECISIONES QUE VALE LA PENA CONOCER
-----------------------------------
· Los faltantes vienen como -999 y se convierten en vacío, no en cero. Un cero
  de lluvia y un dato ausente no son lo mismo y confundirlos arruina cualquier
  promedio.
· POWER tiene rezago de unos días: las últimas fechas suelen venir vacías.
· Al agregar por período, la lluvia se SUMA y las temperaturas se PROMEDIAN.
  Sumar temperaturas o promediar lluvia son errores frecuentes y silenciosos.
· Los nombres de columna van en castellano. La equivalencia con los códigos
  originales de POWER queda en la tabla VARIABLES, para poder rastrear todo.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

API = "https://power.larc.nasa.gov/api/temporal/daily/point"
FALTANTE = -999.0

# código POWER → (nombre en castellano, unidad, cómo se agrega)
#   suma     → se acumula en el período (lluvia)
#   media    → se promedia
#   maximo   → se toma el mayor del período
#   minimo   → se toma el menor del período
VARIABLES = [
    ("PRECTOTCORR",       "lluvia_mm",            "mm",        "suma"),
    ("T2M",               "t_media_c",            "°C",        "media"),
    ("T2M_MAX",           "t_maxima_c",           "°C",        "maximo"),
    ("T2M_MIN",           "t_minima_c",           "°C",        "minimo"),
    ("T2MDEW",            "t_rocio_c",            "°C",        "media"),
    ("RH2M",              "humedad_relativa_pct", "%",         "media"),
    ("WS2M",              "viento_ms",            "m/s",       "media"),
    ("WS2M_MAX",          "viento_maximo_ms",     "m/s",       "maximo"),
    ("PS",                "presion_kpa",          "kPa",       "media"),
    ("ALLSKY_SFC_SW_DWN", "radiacion_mj",         "MJ/m²/día", "media"),
    ("TS",                "t_suelo_c",            "°C",        "media"),
    ("GWETTOP",           "humedad_suelo_sup",    "0–1",       "media"),
    ("GWETROOT",          "humedad_suelo_raiz",   "0–1",       "media"),
]

CODIGOS = [v[0] for v in VARIABLES]
NOMBRES = {v[0]: v[1] for v in VARIABLES}
AGREGA = {v[1]: v[3] for v in VARIABLES}
UNIDADES = {v[1]: v[2] for v in VARIABLES}

PERIODOS = ("dia", "semana", "mes", "trimestre", "anio")


# ───────────────────────────────────────────────────────────────────────────
def traer(lat: float, lon: float, desde: str, hasta: str,
          variables=None, tiempo_espera=120) -> list[dict]:
    """Descarga la serie diaria. Devuelve una lista de dicts con 'fecha' y las
    variables en castellano. Los faltantes quedan como None."""
    cods = variables or CODIGOS
    q = urllib.parse.urlencode({
        "parameters": ",".join(cods),
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": desde.replace("-", ""),
        "end": hasta.replace("-", ""),
        "format": "JSON",
    })
    with urllib.request.urlopen(f"{API}?{q}", timeout=tiempo_espera) as r:
        d = json.load(r)

    if "properties" not in d:
        raise RuntimeError(f"respuesta inesperada de POWER: {str(d)[:300]}")

    p = d["properties"]["parameter"]
    coords = d.get("geometry", {}).get("coordinates", [None, None, None])
    fechas = sorted(next(iter(p.values())).keys())

    filas = []
    for f in fechas:
        fila = {"fecha": f"{f[:4]}-{f[4:6]}-{f[6:]}"}
        for c in cods:
            v = p.get(c, {}).get(f, FALTANTE)
            fila[NOMBRES.get(c, c)] = None if v == FALTANTE else v
        filas.append(fila)

    filas_meta = {
        "latitud": coords[1], "longitud": coords[0],
        "altitud_m": coords[2] if len(coords) > 2 else None,
    }
    for fila in filas:
        fila["_meta"] = filas_meta
    return filas


# ───────────────────────────────────────────────────────────────────────────
def _clave_periodo(f: str, periodo: str) -> str:
    d = datetime.strptime(f, "%Y-%m-%d").date()
    if periodo == "dia":
        return f
    if periodo == "semana":
        iso = d.isocalendar()
        return f"{iso[0]}-S{iso[1]:02d}"
    if periodo == "mes":
        return f"{d.year}-{d.month:02d}"
    if periodo == "trimestre":
        return f"{d.year}-T{(d.month - 1) // 3 + 1}"
    if periodo == "anio":
        return str(d.year)
    raise ValueError(f"período no reconocido: {periodo}")


def agregar(filas: list[dict], periodo: str) -> list[dict]:
    """Agrupa la serie diaria en el período pedido, respetando cómo se agrega
    cada variable: la lluvia se suma, las temperaturas se promedian."""
    if periodo == "dia":
        return [{k: v for k, v in f.items() if k != "_meta"} for f in filas]

    grupos: dict[str, list[dict]] = {}
    for f in filas:
        grupos.setdefault(_clave_periodo(f["fecha"], periodo), []).append(f)

    salida = []
    for clave in sorted(grupos):
        g = grupos[clave]
        fila = {"periodo": clave, "dias": len(g)}
        for _, nombre, _, modo in VARIABLES:
            vals = [f[nombre] for f in g if f.get(nombre) is not None]
            fila[f"dias_con_dato_{nombre}"] = len(vals) if len(vals) != len(g) else ""
            if not vals:
                fila[nombre] = None
            elif modo == "suma":
                fila[nombre] = round(sum(vals), 3)
            elif modo == "media":
                fila[nombre] = round(sum(vals) / len(vals), 3)
            elif modo == "maximo":
                fila[nombre] = round(max(vals), 3)
            elif modo == "minimo":
                fila[nombre] = round(min(vals), 3)
        salida.append(fila)
    return salida


# ───────────────────────────────────────────────────────────────────────────
def escribir_csv(filas: list[dict], ruta: str) -> None:
    if not filas:
        raise ValueError("no hay filas que escribir")
    cols = [c for c in filas[0].keys() if c != "_meta"]
    cols = [c for c in cols if not (c.startswith("dias_con_dato_")
                                    and all(not f.get(c) for f in filas))]
    with open(ruta, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for f in filas:
            w.writerow({c: ("" if f.get(c) is None else f.get(c)) for c in cols})


def informe(filas: list[dict], periodo: str) -> str:
    """Resumen legible de qué llegó y qué tan completo viene."""
    if not filas:
        return "sin datos"
    meta = filas[0].get("_meta", {})
    campo = "fecha" if periodo == "dia" else "periodo"
    L = []
    if meta:
        L.append(f"punto devuelto: {meta.get('latitud')}, {meta.get('longitud')}"
                 f"  ·  altitud {meta.get('altitud_m')} m")
    L.append(f"filas: {len(filas)}  ·  de {filas[0][campo]} a {filas[-1][campo]}"
             f"  ·  período: {periodo}")
    L.append("")
    L.append(f"{'variable':<24}{'unidad':<12}{'con dato':>10}{'mínimo':>12}{'máximo':>12}")
    for _, nombre, unidad, _ in VARIABLES:
        vals = [f[nombre] for f in filas if f.get(nombre) is not None]
        if not vals:
            L.append(f"{nombre:<24}{unidad:<12}{'0':>10}{'—':>12}{'—':>12}")
            continue
        pct = 100 * len(vals) / len(filas)
        L.append(f"{nombre:<24}{unidad:<12}{f'{pct:.0f} %':>10}"
                 f"{min(vals):>12.3f}{max(vals):>12.3f}")
    vacias = [f[campo] for f in filas
              if all(f.get(n) is None for _, n, _, _ in VARIABLES)]
    if vacias:
        L.append("")
        L.append(f"⚠  {len(vacias)} filas sin ningún dato: {vacias[0]} … {vacias[-1]}")
        L.append("   POWER tiene rezago de algunos días; las fechas recientes suelen venir vacías.")
    return "\n".join(L)


# ───────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Trae datos meteorológicos reales por coordenadas desde NASA POWER.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="ejemplo:\n"
               "  python3 datos_clima.py --lat -32.84 --lon -70.95 "
               "--desde 2024-01-01 --hasta 2026-07-22 --periodo mes")
    ap.add_argument("--lat", type=float, required=True, help="latitud, negativa al sur")
    ap.add_argument("--lon", type=float, required=True, help="longitud, negativa al oeste")
    ap.add_argument("--desde", required=True, help="fecha inicial, AAAA-MM-DD")
    ap.add_argument("--hasta", required=True, help="fecha final, AAAA-MM-DD")
    ap.add_argument("--periodo", default="dia", choices=PERIODOS,
                    help="agrupación de salida (por omisión: dia)")
    ap.add_argument("--salida", default=None, help="archivo CSV de salida")
    ap.add_argument("--silencioso", action="store_true", help="no imprimir el resumen")
    a = ap.parse_args()

    for f in (a.desde, a.hasta):
        try:
            date.fromisoformat(f)
        except ValueError:
            print(f"fecha inválida: {f} — se espera AAAA-MM-DD", file=sys.stderr)
            return 2
    if a.desde > a.hasta:
        print("la fecha inicial es posterior a la final", file=sys.stderr)
        return 2

    if not a.silencioso:
        print(f"consultando NASA POWER · {a.lat}, {a.lon} · {a.desde} → {a.hasta}", flush=True)

    try:
        filas = traer(a.lat, a.lon, a.desde, a.hasta)
    except Exception as e:
        print(f"falló la consulta: {e}", file=sys.stderr)
        return 1

    salida = agregar(filas, a.periodo)
    ruta = a.salida or (f"clima_{a.lat}_{a.lon}_{a.desde}_{a.hasta}_{a.periodo}.csv"
                        .replace(" ", ""))
    escribir_csv(salida, ruta)

    if not a.silencioso:
        print()
        print(informe(salida if a.periodo != "dia" else filas, a.periodo))
        print()
        print(f"escrito: {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
