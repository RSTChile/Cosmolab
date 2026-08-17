#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
archivo_clima.py — cuánta información acumulada hay para una coordenada.

Responde a una pregunta concreta: para este punto del mapa, ¿hasta dónde llega
hacia atrás el registro, y a qué distancia está la fuente más cercana que lo
sostiene?

Consulta dos archivos:

  1. NASA POWER — reanálisis diario global desde el 1 de enero de 1981. Cubre
     cualquier coordenada, sin excepción, pero solo cuatro décadas y media.

  2. NOAA Paleoclimatology — anillos de árboles, testigos de hielo, sedimentos
     marinos y lacustres, corales. Cobertura irregular: donde hay, hay milenios;
     donde no hay, no hay nada. Por eso el módulo informa la DISTANCIA a cada
     archivo, que es el dato que decide si sirve como proxy del punto.

DECISIONES QUE VALE LA PENA CONOCER
-----------------------------------
· La NOAA mezcla dos escalas de tiempo. Unos estudios fechan en años calendario
  (CE) y otros en «antes del presente» (BP), donde el presente es 1950 por
  convención. El módulo lo normaliza todo a años calendario; sin eso las
  duraciones salen negativas.
· Se descartan los estudios cuyo «sitio» es un recuadro continental o global.
  Un estudio con coordenadas de −90 a 90 no está cerca de nada: cubre el planeta
  y no dice nada sobre este punto en particular.
· La distancia se calcula con la fórmula del semiverseno sobre un radio terrestre
  de 6.371 km. Es aproximada y suficiente: lo que importa es si el proxy está a
  20 km o a 900.

USO
---
    python3 archivo_clima.py --lat -32.84 --lon -70.95
    python3 archivo_clima.py --lat -27.37 --lon -70.33 --radio 400 --salida copiapo.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.parse
import urllib.request

POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"
NOAA = "https://www.ncei.noaa.gov/access/paleo-search/study/search.json"
POWER_INICIO = 1981          # verificado empíricamente: antes responde "out of range"
PRESENTE_BP = 1950           # convención: "antes del presente" cuenta desde 1950
RADIO_TIERRA = 6371.0


# ───────────────────────────────────────────────────────────────────────────
def distancia_km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * RADIO_TIERRA * math.asin(math.sqrt(a))


def a_calendario(anio, unidad) -> int | None:
    """Normaliza a año calendario. Negativo = antes de Cristo."""
    if anio is None:
        return None
    try:
        a = int(float(anio))
    except (TypeError, ValueError):
        return None
    if (unidad or "").upper().strip() == "BP":
        return PRESENTE_BP - a
    return a


def _grados_por_km(lat):
    """Cuántos grados de latitud y longitud equivalen a un kilómetro aquí."""
    dlat = 1 / 111.0
    dlon = 1 / max(1e-6, 111.0 * math.cos(lat * math.pi / 180))
    return dlat, dlon


# ───────────────────────────────────────────────────────────────────────────
def profundidad_instrumental(lat, lon, tiempo_espera=90) -> dict:
    """Confirma desde cuándo POWER entrega datos en este punto."""
    q = urllib.parse.urlencode({
        "parameters": "T2M", "community": "AG", "latitude": lat, "longitude": lon,
        "start": f"{POWER_INICIO}0101", "end": f"{POWER_INICIO}0107", "format": "JSON"})
    try:
        with urllib.request.urlopen(f"{POWER}?{q}", timeout=tiempo_espera) as r:
            d = json.load(r)
        p = d["properties"]["parameter"]["T2M"]
        con_dato = sum(1 for v in p.values() if v != -999)
        coords = d.get("geometry", {}).get("coordinates", [None, None, None])
        return {"fuente": "NASA POWER (reanálisis diario)",
                "desde": POWER_INICIO, "hasta": 2026,
                "anios": 2026 - POWER_INICIO,
                "distancia_km": 0.0,
                "confirmado": con_dato > 0,
                "altitud_m": coords[2] if len(coords) > 2 else None}
    except Exception as e:
        return {"fuente": "NASA POWER", "error": str(e)[:120]}


def archivos_paleo(lat, lon, radio_km=300, limite=400, tiempo_espera=120) -> list[dict]:
    """Busca archivos paleoclimáticos puntuales dentro del radio pedido."""
    dlat, dlon = _grados_por_km(lat)
    q = urllib.parse.urlencode({
        "minLat": lat - radio_km * dlat, "maxLat": lat + radio_km * dlat,
        "minLon": lon - radio_km * dlon, "maxLon": lon + radio_km * dlon,
        "limit": limite})
    with urllib.request.urlopen(f"{NOAA}?{q}", timeout=tiempo_espera) as r:
        d = json.load(r)

    salida = []
    for est in d.get("study", []):
        for sitio in est.get("site", []):
            c = sitio.get("geo", {}).get("geometry", {}).get("coordinates")
            # 2 coordenadas = punto; 4 = recuadro continental o global, se descarta
            if not c or len(c) != 2:
                continue
            try:
                slat, slon = float(c[0]), float(c[1])
            except (TypeError, ValueError):
                continue
            dist = distancia_km(lat, lon, slat, slon)
            if dist > radio_km:
                continue
            for pd in sitio.get("paleoData", []):
                unidad = pd.get("timeUnit") or est.get("timeUnit")
                desde = a_calendario(pd.get("earliestYear"), unidad)
                hasta = a_calendario(pd.get("mostRecentYear"), unidad)
                if desde is None or hasta is None:
                    continue
                if desde > hasta:
                    desde, hasta = hasta, desde
                esp = (pd.get("species") or [{}])
                salida.append({
                    "tipo": est.get("dataType", "?"),
                    "sitio": sitio.get("siteName", "?"),
                    "especie": esp[0].get("speciesCode", "") if esp else "",
                    "latitud": round(slat, 4), "longitud": round(slon, 4),
                    "distancia_km": round(dist, 1),
                    "desde_anio": desde, "hasta_anio": hasta,
                    "anios_cubiertos": hasta - desde,
                    "enlace": est.get("onlineResourceLink", ""),
                    "investigadores": (est.get("investigators") or "")[:70],
                })
    # sin duplicados de sitio+tabla, el más profundo primero
    vistos, limpio = set(), []
    for f in sorted(salida, key=lambda x: -x["anios_cubiertos"]):
        k = (f["sitio"], f["desde_anio"], f["hasta_anio"])
        if k in vistos:
            continue
        vistos.add(k)
        limpio.append(f)
    return limpio


# ───────────────────────────────────────────────────────────────────────────
def informe(lat, lon, inst, paleo, radio_km) -> str:
    L = [f"ARCHIVO DISPONIBLE PARA {lat}, {lon}   ·   radio de búsqueda {radio_km} km", ""]

    if inst.get("confirmado"):
        L.append(f"  instrumental   {inst['fuente']}")
        L.append(f"                 desde {inst['desde']}  ·  {inst['anios']} años"
                 f"  ·  en el punto exacto  ·  altitud {inst['altitud_m']} m")
    else:
        L.append(f"  instrumental   sin confirmar: {inst.get('error','?')}")
    L.append("")

    if not paleo:
        L.append(f"  paleoclimático  ningún archivo puntual dentro de {radio_km} km.")
        L.append("                  Ampliar el radio, o aceptar que este punto no tiene proxy cercano.")
        return "\n".join(L)

    prof = max(p["hasta_anio"] - p["desde_anio"] for p in paleo)
    mas_antiguo = min(p["desde_anio"] for p in paleo)
    L.append(f"  paleoclimático  {len(paleo)} archivos puntuales")
    L.append(f"                  el más profundo cubre {prof:,} años"
             .replace(",", "."))
    L.append(f"                  el registro llega hasta el año {mas_antiguo}"
             + (" (antes de Cristo)" if mas_antiguo < 0 else ""))
    L.append("")
    L.append(f"  {'tipo':<24}{'sitio':<32}{'esp':<6}{'desde':>8}{'hasta':>7}{'años':>8}{'km':>8}")
    for p in paleo[:20]:
        L.append(f"  {p['tipo'][:23]:<24}{p['sitio'][:31]:<32}{p['especie'][:5]:<6}"
                 f"{p['desde_anio']:>8}{p['hasta_anio']:>7}{p['anios_cubiertos']:>8}"
                 f"{p['distancia_km']:>8.0f}")
    if len(paleo) > 20:
        L.append(f"  … y {len(paleo)-20} más en el CSV")

    L.append("")
    L.append("  por cercanía, los cinco más próximos:")
    for p in sorted(paleo, key=lambda x: x["distancia_km"])[:5]:
        L.append(f"    {p['distancia_km']:>6.0f} km  {p['sitio'][:38]:<40}"
                 f"{p['desde_anio']}–{p['hasta_anio']}  ({p['anios_cubiertos']} años)")
    return "\n".join(L)


# ───────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Informa cuánta información acumulada existe para una coordenada.")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--radio", type=float, default=300, help="radio en km (por omisión 300)")
    ap.add_argument("--salida", default=None, help="archivo CSV con el detalle")
    a = ap.parse_args()

    print(f"consultando archivos para {a.lat}, {a.lon} …", flush=True)
    inst = profundidad_instrumental(a.lat, a.lon)
    try:
        paleo = archivos_paleo(a.lat, a.lon, a.radio)
    except Exception as e:
        print(f"falló la consulta paleoclimática: {e}", file=sys.stderr)
        paleo = []

    print()
    print(informe(a.lat, a.lon, inst, paleo, a.radio))

    if a.salida and paleo:
        cols = list(paleo[0].keys())
        with open(a.salida, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(paleo)
        print()
        print(f"escrito: {a.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
