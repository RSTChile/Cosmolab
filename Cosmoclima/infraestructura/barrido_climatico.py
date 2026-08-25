"""
BARRIDO CLIMÁTICO DE LAS 31 SUB-MATRICES
==========================================

INSTRUCCIÓN (Alexis, 22-ago-2026): «Barrido climático.»

Baja 36 años de serie diaria para **todos** los activos georreferenciados del
proyecto y deja el mapa de exposición climática que hoy no existe.

★ POR QUÉ NO SE BAJA ACTIVO POR ACTIVO
----------------------------------------
Son 92.481 activos, y bajar uno por uno sería absurdo: el reanálisis ERA5 tiene
una malla de unos 9 km, así que **todos los activos que caen en la misma celda
comparten exactamente la misma serie**. Medido sobre las 31 sub-matrices:

    92.481 activos  →  3.515 celdas únicas de 0,1°  (≈ 26 activos por celda)

Se baja una vez por celda y después cada activo lee la suya. Es 26 veces menos
trabajo y **el resultado es idéntico**, no una aproximación.

★ ES REANUDABLE, Y HACE FALTA QUE LO SEA
------------------------------------------
Open-Meteo es gratuito y limita la tasa con facilidad: en la prueba de 16 puntos
del 21-ago devolvió «429 demasiadas peticiones» en 5 de los 16 entre las dos
primeras pasadas. Con dos hilos y pausa de 1,5 s entra, pero hay que poder
retomar. Cada celda completa queda anotada y no se vuelve a pedir.

USO
---
    ../.venv-esa/bin/python barrido_climatico.py --celdas    # cuántas son
    ../.venv-esa/bin/python barrido_climatico.py --bajar     # el barrido
    ../.venv-esa/bin/python barrido_climatico.py --estado
"""

import csv
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openpyxl

AQUI = Path(__file__).resolve().parent
SUB = AQUI / "submatrices_excel"
DATOS = AQUI / "datos"
SALIDA = DATOS / "clima_diario_celdas.csv"
HECHAS = DATOS / "clima_celdas_hechas.txt"

MALLA = 0.10                       # ≈ 9 km, la resolución de ERA5-Land
DESDE, HASTA = "1990-01-01", "2026-08-21"
# ★ Corregido el 22-ago: con 2 hilos y 1,5 s Open-Meteo cayó al 11 % de éxito
# (263 celdas buenas de 2.300 intentadas). Un hilo y pausa de 4 s es el ritmo que
# el servicio gratuito tolera de forma sostenida.
HILOS, PAUSA_S = 1, 4.0
URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}"
       "&longitude={lon}&start_date=" + DESDE + "&end_date=" + HASTA +
       "&daily=precipitation_sum&timezone=America%2FSantiago")


def celda(lat, lon):
    """La celda de la malla a la que pertenece un punto, y su centro."""
    i, j = round(lat / MALLA), round(lon / MALLA)
    return f"{i}_{j}", round(i * MALLA, 4), round(j * MALLA, 4)


def celdas_del_proyecto():
    """Todas las celdas únicas que ocupan los activos de las 31 sub-matrices."""
    vistas, activos = {}, 0
    for p in sorted(SUB.glob("*.xlsx")):
        ws = openpyxl.load_workbook(p, read_only=True).active
        filas = list(ws.iter_rows(values_only=True))
        cab = {c: i for i, c in enumerate(filas[0])}
        if "Latitud decimal" not in cab:
            continue
        for f in filas[1:]:
            la, lo = f[cab["Latitud decimal"]], f[cab["Longitud decimal"]]
            if la in (None, "") or lo in (None, ""):
                continue
            activos += 1
            k, clat, clon = celda(float(la), float(lo))
            vistas.setdefault(k, (clat, clon))
    return vistas, activos


def celdas_de_emergencias():
    """★ Las celdas de las vías cortadas y los puntos críticos de SENAPRED.

    Se agregaron el 23-ago-2026: el cruce de `umbral_por_tramo.py` sólo pudo
    usar 293 de 1.289 tramos (22,7 %) porque la serie no cubría su celda. Estas
    van PRIMERO en la cola — son las que desbloquean el análisis en curso,
    mientras que las de activos alimentan el mapa de exposición general.
    """
    crudo = DATOS / "crudo" / "puntos_criticos"
    if not crudo.exists():
        return {}
    ult = sorted(crudo.glob("*"))[-1]
    vistas = {}

    def anotar(lat, lon):
        if lat is None or lon is None:
            return
        # ⚠️ Sólo Chile continental: Pascua, Juan Fernández y la Antártica quedan
        #    fuera del barrido (9 celdas) porque estiran el área sin aportar.
        if not (-56.5 <= lat <= -17.0 and -76.0 <= lon <= -66.0):
            return
        k, cla, clo = celda(lat, lon)
        vistas.setdefault(k, (cla, clo))

    f = ult / "vias_julio2026.json"
    if f.exists():
        for x in json.loads(f.read_text(encoding="utf-8"))["features"]:
            g = x.get("geometry") or {}
            anotar(g.get("y"), g.get("x"))
    f = ult / "pc_2026.json"
    if f.exists():
        for x in json.loads(f.read_text(encoding="utf-8"))["features"]:
            g = (x.get("geometry") or {}).get("coordinates")
            if g:
                anotar(g[1], g[0])
    return vistas


def hechas():
    return set(HECHAS.read_text().split()) if HECHAS.exists() else set()


def bajar_celda(item):
    """Una celda. Devuelve su clave si entró, None si no."""
    k, (lat, lon) = item
    for intento in range(6):
        try:
            with urllib.request.urlopen(URL.format(lat=lat, lon=lon),
                                        timeout=120) as r:
                d = json.loads(r.read())["daily"]
            with SALIDA.open("a", newline="", encoding="utf8") as fh:
                w = csv.writer(fh)
                for fecha, p in zip(d["time"], d["precipitation_sum"]):
                    w.writerow([k, fecha, p if p is not None else ""])
            with HECHAS.open("a", encoding="utf8") as fh:
                fh.write(k + "\n")
            return k
        except Exception:
            time.sleep(12 * (intento + 1))
    return None


def main():
    if "--celdas" in sys.argv or "--estado" in sys.argv:
        v, a = celdas_del_proyecto()
        em = celdas_de_emergencias()
        h = hechas()
        print(f"  vías y puntos críticos    : {len(em):,} celdas"
              f"  · bajadas {sum(1 for k in em if k in h):,}"
              f" ({100*sum(1 for k in em if k in h)/max(len(em),1):.1f} %)")
        print(f"  activos georreferenciados : {a:,}")
        print(f"  celdas únicas de {MALLA}°     : {len(v):,}"
              f"  ({a/max(len(v),1):.0f} activos por celda)")
        print(f"  ya bajadas                : {len(h):,}"
              f"  ({100*len(h)/max(len(v),1):.1f} %)")
        if SALIDA.exists():
            print(f"  archivo                   : "
                  f"{SALIDA.stat().st_size/1e6:.0f} MB")
        return 0

    if "--bajar" not in sys.argv:
        print(__doc__)
        return 1

    vistas, activos = celdas_del_proyecto()
    emerg = celdas_de_emergencias()
    ya = hechas()
    # ★ Orden de la cola: primero vías y puntos críticos, después el resto.
    faltan = [(k, v) for k, v in emerg.items() if k not in ya]
    faltan += [(k, v) for k, v in vistas.items()
               if k not in ya and k not in emerg]
    print(f"celdas de vías y puntos críticos: {len(emerg):,} "
          f"(por bajar {sum(1 for k in emerg if k not in ya):,}) — van primero",
          flush=True)
    if not SALIDA.exists():
        with SALIDA.open("w", newline="", encoding="utf8") as fh:
            csv.writer(fh).writerow(["celda", "fecha", "precip_mm"])
    print(f"activos {activos:,} · celdas {len(vistas):,} · "
          f"ya bajadas {len(ya):,} · por bajar {len(faltan):,}", flush=True)

    t0, ok = time.time(), 0
    with ThreadPoolExecutor(max_workers=HILOS) as ex:
        for i, res in enumerate(
                ex.map(lambda c: (time.sleep(PAUSA_S), bajar_celda(c))[1],
                       faltan), start=1):
            ok += 1 if res else 0
            if i % 50 == 0 or i == len(faltan):
                t = time.time() - t0
                print(f"   {i:,}/{len(faltan):,} · {ok:,} bien · "
                      f"{i/max(t,1)*60:.0f} celdas/min · "
                      f"faltan ~{(len(faltan)-i)/max(i/max(t,1),0.01)/60:.0f} min",
                      flush=True)
    print(f"BARRIDO TERMINADO · {ok:,} de {len(faltan):,} celdas", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
