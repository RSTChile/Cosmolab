"""
DE LOS NETCDF A LA SERIE POR CELDA · ERA5-Land → clima diario
===============================================================

`bajar_era5land_cds.py` deja un netCDF por mes con toda la grilla de Chile
continental (396 × 101 puntos a 0,1°). Este script los convierte en la serie
diaria por celda que consumen `derivar_umbral.py` y `umbral_por_tramo.py`.

★★ LAS DOS CONVERSIONES QUE HAY QUE HACER SÍ O SÍ
---------------------------------------------------
1. **Restar un día.** En ERA5-Land la precipitación acumula desde las 00 UTC, así
   que el paso de las 00:00 del día D trae la lluvia del día **D−1**. Sin este
   corrimiento la serie entera queda corrida y NADIE LO NOTA, porque los números
   siguen pareciendo razonables. Verificado contra la serie vieja: con el
   corrimiento los picos del temporal de julio caen en los mismos días.
2. **Multiplicar por 1.000.** ERA5 acumula en METROS. El máximo medido en julio
   2026 fue 0,1359 m, que son 135,9 mm/día.

⚠️ NO SOBRESCRIBE LA SERIE VIEJA
----------------------------------
La salida va a un archivo NUEVO. La serie de Open-Meteo se conserva a propósito:
fue la que permitió detectar que ERA5-Land mide **19 % más** (Coquimbo, julio
2026: 343,6 mm contra 408,7 mm), y sirve como control independiente. Mezclarlas
en un mismo archivo es justo lo que este trabajo decidió no hacer.

⚠️ EL DÍA NO EMPIEZA A LA MISMA HORA en una y otra serie: la vieja se pidió en
hora de Santiago y ésta acumula en UTC (las 00 UTC son las 20:00 del día anterior
en Chile). Son series distintas, no dos versiones de la misma.

USO
---
    ../.venv-esa/bin/python convertir_era5land_a_serie.py --celdas
    ../.venv-esa/bin/python convertir_era5land_a_serie.py --convertir
"""

import csv
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"
CRUDO = DATOS / "crudo" / "era5land"
SALIDA = DATOS / "clima_diario_celdas_era5land.csv"
# ★★ LAS CELDAS DE CUENCA, en archivo aparte.
#   En el norte los caminos se cortan por lluvia que cae CORDILLERA ARRIBA: el
#   valle de Copiapó puede marcar 0 mm mientras a 80 km al este caen 50 y baja
#   la quebrada. Evaluando sólo la celda del activo, el modelo mira el lugar
#   equivocado y no ve nada. Verificado: la precordillera de Copiapó registra
#   83,4 mm en el aluvión de marzo 2015 y esa celda NO estaba convertida.
#   Van en su propio archivo porque son 4.085 celdas más que triplicarían el
#   peso del que leen todos los demás scripts.
SALIDA_CUENCA = DATOS / "clima_diario_cuenca.csv"
CELDAS_AL_ESTE = 8       # ~0,8° ≈ 80 km, el alcance de una quebrada cordillerana
MALLA = 0.10


def celda(lat, lon):
    return f"{round(lat/MALLA)}_{round(lon/MALLA)}"


def celdas_de_interes():
    """Todas las celdas que el proyecto necesita: vías cortadas, puntos críticos
    de SENAPRED y activos de las sub-matrices. Sólo Chile continental."""
    vistas = set()

    def anotar(lat, lon):
        if lat is None or lon is None:
            return
        if -56.5 <= lat <= -17.0 and -76.0 <= lon <= -66.0:
            vistas.add(celda(lat, lon))

    pc = DATOS / "crudo" / "puntos_criticos"
    if pc.exists():
        ult = sorted(pc.glob("*"))[-1]
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

    # ★ Los activos se leen de `fuente_climatica_por_activo.csv`, que los tiene
    #   TODOS (92.481 filas), y NO de la serie vieja: ésa sólo contiene las
    #   celdas que Open-Meteo alcanzó a bajar, así que usarla como fuente dejaría
    #   fuera las ~1.500 celdas de activos que nunca llegaron a bajarse — que son
    #   justamente las que este trabajo viene a cubrir.
    act = DATOS / "fuente_climatica_por_activo.csv"
    if act.exists():
        with act.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    anotar(float(r["lat"]), float(r["lon"]))
                except (TypeError, ValueError):
                    continue

    # y las de la serie vieja, para poder comparar las dos fuentes celda a celda
    vieja = DATOS / "clima_diario_celdas.csv"
    if vieja.exists():
        with vieja.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                vistas.add(r["celda"])
    return vistas


def celdas_de_cuenca(interes):
    """Las celdas AGUAS ARRIBA de cada celda con activos.

    Aproximación deliberadamente simple: hasta 80 km al este, que en Chile es
    hacia la cordillera. No es la cuenca hidrográfica real —para eso está el
    modelo de elevación que el proyecto ya tiene bajado— pero captura el
    mecanismo: el agua cae arriba y baja.
    """
    out = set()
    for c in interes:
        try:
            i, j = (int(x) for x in c.split("_"))
        except ValueError:
            continue
        for dj in range(1, CELDAS_AL_ESTE + 1):
            k = f"{i}_{j + dj}"
            if k not in interes:
                out.add(k)
    return out


def main():
    quiero = celdas_de_interes()
    ncs = sorted(CRUDO.glob("*.nc"))

    if "--celdas" in sys.argv or not ncs:
        print(f"  celdas de interés : {len(quiero):,}")
        print(f"  netCDF bajados    : {len(ncs)}")
        if not ncs:
            print("  (nada que convertir todavía — corre bajar_era5land_cds.py)")
        return 0

    if "--convertir" not in sys.argv:
        print(__doc__)
        return 1

    print(f"celdas de interés {len(quiero):,} · meses {len(ncs)}", flush=True)

    # ── el mapa celda → índice de grilla, una sola vez ───────────────────────
    d0 = xr.open_dataset(ncs[0])
    lats = d0["latitude"].values
    lons = d0["longitude"].values
    d0.close()
    # ★ Se resuelve por índice y no con .sel(method="nearest") por celda: son
    #   miles de celdas por mes y el nearest punto a punto tarda minutos.
    idx = {}
    for c in quiero:
        i, j = (int(x) for x in c.split("_"))
        la, lo = i * MALLA, j * MALLA
        ii = int(np.abs(lats - la).argmin())
        jj = int(np.abs(lons - lo).argmin())
        # descarta celdas fuera del área descargada
        if abs(lats[ii] - la) < 0.06 and abs(lons[jj] - lo) < 0.06:
            idx[c] = (ii, jj)
    print(f"  celdas dentro del área bajada: {len(idx):,} "
          f"(fuera: {len(quiero)-len(idx):,})", flush=True)

    claves = sorted(idx)
    ii = np.array([idx[c][0] for c in claves])
    jj = np.array([idx[c][1] for c in claves])

    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["celda", "fecha", "precip_mm"])
        filas = 0
        for n, p in enumerate(ncs, 1):
            d = xr.open_dataset(p)
            tc = "valid_time" if "valid_time" in d.coords else "time"
            tp = d["tp"].values                      # (tiempo, lat, lon)
            tiempos = d[tc].values
            for k in range(len(tiempos)):
                # ★ EL CORRIMIENTO: el paso de las 00:00 del día D es el día D−1
                dia = (np.datetime64(tiempos[k], "D").astype("datetime64[D]")
                       .astype(object) - timedelta(days=1)).isoformat()
                capa = tp[k]
                vals = capa[ii, jj] * 1000.0         # ★ metros → milímetros
                for c, v in zip(claves, vals):
                    if not np.isnan(v):
                        w.writerow([c, dia, round(float(v), 2)])
                        filas += 1
            d.close()
            if n % 12 == 0 or n == len(ncs):
                print(f"  [{n}/{len(ncs)}] {p.stem} · {filas:,} filas",
                      flush=True)

    print(f"\n  escrito: {SALIDA.name} · {filas:,} filas · "
          f"{SALIDA.stat().st_size/1e6:.0f} MB")

    # ── las celdas de cuenca ────────────────────────────────────────────────
    cuenca = celdas_de_cuenca(quiero)
    idxc = {}
    for c in cuenca:
        i, j = (int(x) for x in c.split("_"))
        la, lo = i * MALLA, j * MALLA
        ii = int(np.abs(lats - la).argmin())
        jj = int(np.abs(lons - lo).argmin())
        if abs(lats[ii] - la) < 0.06 and abs(lons[jj] - lo) < 0.06:
            idxc[c] = (ii, jj)
    print(f"\n  celdas de cuenca (hasta {CELDAS_AL_ESTE*10} km al este): "
          f"{len(cuenca):,} candidatas · {len(idxc):,} dentro del área", flush=True)
    if idxc:
        cl = sorted(idxc)
        ic = np.array([idxc[c][0] for c in cl])
        jc = np.array([idxc[c][1] for c in cl])
        fc = 0
        with SALIDA_CUENCA.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["celda", "fecha", "precip_mm"])
            for n, p in enumerate(ncs, 1):
                d = xr.open_dataset(p)
                tc = "valid_time" if "valid_time" in d.coords else "time"
                tp = d["tp"].values
                tiempos = d[tc].values
                for k in range(len(tiempos)):
                    dia = (np.datetime64(tiempos[k], "D").astype("datetime64[D]")
                           .astype(object) - timedelta(days=1)).isoformat()
                    vals = tp[k][ic, jc] * 1000.0
                    for c, v in zip(cl, vals):
                        if not np.isnan(v):
                            w.writerow([c, dia, round(float(v), 2)])
                            fc += 1
                d.close()
                if n % 120 == 0:
                    print(f"    [{n}/{len(ncs)}] {fc:,} filas", flush=True)
        print(f"  escrito: {SALIDA_CUENCA.name} · {fc:,} filas · "
              f"{SALIDA_CUENCA.stat().st_size/1e6:.0f} MB")
    print("  ⚠️ La serie vieja de Open-Meteo NO se tocó: son series distintas")
    print("     (ERA5-Land mide ~19 % más y su día empieza a otra hora).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
