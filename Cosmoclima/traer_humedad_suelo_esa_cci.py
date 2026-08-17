#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traer_humedad_suelo_esa_cci.py -- baja la humedad de suelo satelital del
punto-reloj (Huintil) desde el archivo de la ESA (14-ago-2026).

PARA QUÉ. La lluvia de 2019 en adelante es el punto flojo del instrumento: no
hay pluviómetro en el punto-reloj después de 2018, y las dos formas de
estimarla -- estaciones vecinas encadenadas vs reanálisis ERA5 corregido --
discrepan de forma SISTEMÁTICA, hasta el doble en años secos. Con un umbral de
germinación de 15 mm/mes, esa diferencia decide si el desierto florece.

No había forma de decidir entre ambas porque las dos se apoyan, de un modo u
otro, en el mismo mundo de estaciones que se cortó en 2018. La humedad de
suelo satelital es un TERCER TESTIGO: viene de microondas, es continua desde
1978 y no sabe nada de ese corte. Si sigue a una candidata y no a la otra,
decide. Y si no distingue ninguna, eso también es un resultado -- acota
cuánto se puede saber con lo disponible.

Además está más cerca del fenómeno que la lluvia misma: lo que hace germinar
una semilla es el agua QUE QUEDA en el suelo, no la que cayó del cielo.

QUÉ SE BAJA
  ESA CCI Soil Moisture, producto COMBINED v09.2 (activo + pasivo fusionados),
  malla 0.25°, diario, 1978-2024. Se pide UNA sola celda por día vía OPeNDAP
  --respuesta de milisegundos-- en vez de bajar los ficheros globales, que
  suman unos 25 GB.

  Celda: lat -31.625, lon -70.875. El punto-reloj está en -31.5669/-70.9817,
  o sea a ~13 km del centro de la celda y holgadamente dentro de ella.

CAUTELAS QUE HAY QUE ARRASTRAR AL INFORME
  · La celda mide ~25 km de lado; el punto-reloj es un punto.
  · Entre 1978 y 1991 hay pocos satélites solapados y el dato es pobre.
  · La serie termina en 2024: de los ocho años en disputa cubre seis.
  · 'flag' distinto de 0 marca nieve/temperatura bajo cero, vegetación densa o
    falta de convergencia del algoritmo. Se graba tal cual y se filtra después,
    no acá: el filtro es una decisión de análisis, no de descarga.

Uso:  python3 traer_humedad_suelo_esa_cci.py [--desde 1979] [--hasta 2024]
Reanuda solo: si el CSV ya tiene días, no los vuelve a pedir.
"""
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import requests

RAIZ = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima"
SALIDA = os.path.join(RAIZ, "investigacion/fuentes/humedad_suelo_esacci_huintil.csv")
BASE = ("https://dap.ceda.ac.uk/thredds/dodsC/neodc/esacci/soil_moisture/data/"
        "daily_files/COMBINED/v09.2")
ILAT, ILON = 486, 436          # celda -31.625 / -70.875, verificada contra el eje real
CONSULTA = f"sm%5B0%5D%5B{ILAT}%5D%5B{ILON}%5D,flag%5B0%5D%5B{ILAT}%5D%5B{ILON}%5D"
HILOS = 12                     # cortesía con el servidor público del CEDA
REINTENTOS = 3

_re_sm = re.compile(r"sm\.sm\[1\]\[1\]\[1\]\s*\n\[0\]\[0\],\s*(-?[\d.eE+]+)")
_re_flag = re.compile(r"flag\.flag\[1\]\[1\]\[1\]\s*\n\[0\]\[0\],\s*(-?\d+)")


def arg(nombre, defecto):
    return int(sys.argv[sys.argv.index(nombre) + 1]) if nombre in sys.argv else defecto


def url_del_dia(d):
    f = (f"ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-"
         f"{d:%Y%m%d}000000-fv09.2.nc")
    return f"{BASE}/{d:%Y}/{f}.ascii?{CONSULTA}"


def pedir(d, ses):
    """Devuelve (fecha, sm, flag, estado). sm=None si no hay dato."""
    for intento in range(REINTENTOS):
        try:
            r = ses.get(url_del_dia(d), timeout=45)
            if r.status_code == 404:
                return (d, None, None, "sin_archivo")
            if r.status_code != 200:
                time.sleep(1 + intento * 2)
                continue
            msm, mfl = _re_sm.search(r.text), _re_flag.search(r.text)
            if not msm:
                return (d, None, None, "respuesta_ilegible")
            v = float(msm.group(1))
            fl = int(mfl.group(1)) if mfl else None
            if v <= -9990:
                return (d, None, fl, "sin_dato")
            return (d, v, fl, "ok")
        except Exception:
            time.sleep(1 + intento * 2)
    return (d, None, None, "falla_red")


def main():
    desde, hasta = arg("--desde", 1979), arg("--hasta", 2024)
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)

    ya = set()
    if os.path.exists(SALIDA):
        with open(SALIDA, newline="", encoding="utf-8") as f:
            ya = {r["fecha"] for r in csv.DictReader(f)}
        print(f"Reanudando: {len(ya):,} días ya bajados")

    dias = []
    d, fin = date(desde, 1, 1), date(hasta, 12, 31)
    while d <= fin:
        if d.isoformat() not in ya:
            dias.append(d)
        d += timedelta(days=1)
    if not dias:
        print("Nada que bajar: el CSV ya está completo.")
        return
    print(f"A pedir: {len(dias):,} días ({desde}-{hasta}) · {HILOS} en paralelo")
    print(f"Celda lat[{ILAT}]=-31.625 lon[{ILON}]=-70.875 · ESA CCI SM COMBINED v09.2\n")

    nuevo = not os.path.exists(SALIDA)
    t0 = time.time()
    hechos = 0
    with open(SALIDA, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha", "sm_m3m3", "flag", "estado"])
        ses = requests.Session()
        ses.headers.update({"User-Agent": "cosmoclima-soilmoisture/1.0"})
        with ThreadPoolExecutor(max_workers=HILOS) as ex:
            for (dd, v, fl, estado) in ex.map(lambda x: pedir(x, ses), dias):
                w.writerow([dd.isoformat(), "" if v is None else f"{v:.5f}",
                            "" if fl is None else fl, estado])
                hechos += 1
                if hechos % 500 == 0:
                    f.flush()
                    seg = time.time() - t0
                    falta = (len(dias) - hechos) / (hechos / seg) / 60
                    print(f"  {hechos:,}/{len(dias):,}  ({100*hechos/len(dias):.0f}%) "
                          f"· faltan ~{falta:.0f} min")
    print(f"\nListo en {(time.time()-t0)/60:.1f} min · {SALIDA}")


if __name__ == "__main__":
    main()
