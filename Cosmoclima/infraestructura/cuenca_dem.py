"""
LA CUENCA REAL · qué celdas drenan de verdad hacia cada lugar
==============================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «calcula el DEM».

La primera versión de la señal de cuenca usaba «hasta 80 km al este», que es una
aproximación burda: **incluye celdas que no drenan hacia el activo y excluye las
que sí**, si están al norte o al sur. En un valle transversal como el del Copiapó
o el del Elqui, el agua que importa baja por el eje del valle, no desde el este
geométrico.

Aquí se calcula con el relieve real, en tres pasos:

  1. **elevación de cada celda** de 0,1°, muestreada del modelo digital
     Copernicus GLO-30 que el proyecto ya tenía bajado;
  2. **hacia dónde drena cada celda** — el vecino de los ocho con mayor pendiente
     descendente, que es el método D8 clásico;
  3. **quién drena hacia quién**, recorriendo ese grafo hacia atrás.

★ POR QUÉ D8 SOBRE LA GRILLA GRUESA Y NO SOBRE LOS 30 m
---------------------------------------------------------
La pregunta no es por dónde corre el agua, sino **qué celdas de lluvia alimentan
este lugar**. La lluvia viene en celdas de 9 km; calcular el escurrimiento a 30 m
para después promediarlo a 9 km sería gastar mil veces más para responder lo
mismo. La resolución de la respuesta no puede ser mejor que la del dato de
entrada.

⚠️ ALCANCE DELIBERADO: sólo al norte de la latitud −33. Ahí el valle puede estar
seco mientras la cordillera recibe el temporal, y la celda propia no ve nada. En
el sur llueve en toda la cuenca a la vez y la celda propia ya captura el evento,
así que la cuenca aportaría poco y costaría 2,5 GB más de descarga.

⚠️ D8 sobre celdas de 9 km es grueso: no reproduce quebradas individuales. Dice
«esta parte de la cordillera alimenta este valle», que es la escala a la que
sirve la respuesta.

USO
---
    ../.venv-esa/bin/python cuenca_dem.py --elevacion   # muestrea el DEM
    ../.venv-esa/bin/python cuenca_dem.py --cuencas     # D8 y acumulación
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "adaptadores"))
DEM = AQUI / "datos" / "crudo" / "terreno" / "dem"
SAL_ELEV = AQUI / "datos" / "elevacion_celdas.json"
SALIDA = AQUI / "web" / "publico" / "datos" / "cuenca.json"

MALLA = 0.10
LAT_MIN, LAT_MAX = -33.0, -17.0      # la franja donde el mecanismo domina
MAX_AGUAS_ARRIBA = 40                # tope por celda, para que el JSON no explote


def celdas_del_proyecto():
    cel = set()
    for f in ("clima_diario_celdas_era5land.csv", "clima_diario_cuenca.csv"):
        p = AQUI / "datos" / f
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as fh:
            r = csv.reader(fh)
            next(r)
            for c, _, _ in r:
                cel.add(c)
    return cel


def en_franja(c):
    try:
        i, _ = (int(x) for x in c.split("_"))
    except ValueError:
        return False
    return LAT_MIN <= i * MALLA <= LAT_MAX


def elevacion():
    import terreno as T

    m = T.Mosaico(T.FUENTE_DEM["url"],
                  "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM",
                  1, str(DEM))
    cel = sorted(c for c in celdas_del_proyecto() if en_franja(c))
    print(f"  celdas en la franja (lat {LAT_MAX} a {LAT_MIN}): {len(cel):,}")

    ya = {}
    if SAL_ELEV.exists():
        ya = json.loads(SAL_ELEV.read_text(encoding="utf-8"))
        print(f"  ya muestreadas: {len(ya):,}")

    pendientes = [c for c in cel if c not in ya]
    print(f"  por muestrear: {len(pendientes):,}", flush=True)
    for k, c in enumerate(pendientes, 1):
        i, j = (int(x) for x in c.split("_"))
        la, lo = i * MALLA, j * MALLA
        try:
            # ★ 3×3 y mediana, no un solo píxel: un píxel puede caer en el fondo
            #   de una quebrada o en la cima de un cerro y no representar la celda.
            v = np.asarray(m.ventana(la, lo, 3, 3, 0.02, 0.02), dtype=float)
            z = float(np.nanmedian(v)) if np.isfinite(v).any() else None
        except Exception:
            z = None
        ya[c] = z
        if k % 100 == 0:
            SAL_ELEV.write_text(json.dumps(ya), encoding="utf-8")
            con = sum(1 for x in ya.values() if x is not None)
            print(f"    {k}/{len(pendientes)} · con elevación {con:,}", flush=True)
    SAL_ELEV.write_text(json.dumps(ya), encoding="utf-8")
    con = sum(1 for x in ya.values() if x is not None)
    print(f"\n  muestreadas {len(ya):,} · con elevación {con:,} "
          f"({100*con/max(len(ya),1):.0f} %)")
    return 0


def cuencas():
    if not SAL_ELEV.exists():
        print("  falta elevacion_celdas.json — corre --elevacion")
        return 1
    z = {k: v for k, v in json.loads(SAL_ELEV.read_text(encoding="utf-8")).items()
         if v is not None}
    print(f"  celdas con elevación: {len(z):,}")

    def coords(c):
        i, j = (int(x) for x in c.split("_"))
        return i, j

    # ── D8: hacia dónde drena cada celda ────────────────────────────────────
    hacia = {}
    for c, zc in z.items():
        i, j = coords(c)
        mejor, mejor_pend = None, 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                v = f"{i+di}_{j+dj}"
                zv = z.get(v)
                if zv is None or zv >= zc:
                    continue
                # distancia real: la diagonal es más larga, y en latitud alta el
                # grado de longitud se acorta.
                lat = i * MALLA
                dy = di * MALLA * 111.32
                dx = dj * MALLA * 111.32 * math.cos(math.radians(lat))
                d = math.hypot(dx, dy) or 1e-9
                pend = (zc - zv) / d
                if pend > mejor_pend:
                    mejor, mejor_pend = v, pend
        hacia[c] = mejor

    sumideros = sum(1 for v in hacia.values() if v is None)
    print(f"  celdas que drenan a un vecino: {len(hacia)-sumideros:,}")
    print(f"  sumideros (sin vecino más bajo): {sumideros:,}")

    # ── el grafo inverso, y la cuenca de cada celda ─────────────────────────
    desde = defaultdict(list)
    for c, v in hacia.items():
        if v:
            desde[v].append(c)

    cuenca = {}
    for c in z:
        vistos = set()
        cola = deque(desde.get(c, []))
        while cola and len(vistos) < MAX_AGUAS_ARRIBA:
            x = cola.popleft()
            if x in vistos:
                continue
            vistos.add(x)
            cola.extend(desde.get(x, []))
        if vistos:
            # ★ Se ordenan por altura descendente: las de arriba primero, que son
            #   las que reciben el temporal cuando el valle está seco.
            cuenca[c] = sorted(vistos, key=lambda k: -z[k])[:MAX_AGUAS_ARRIBA]

    tam = sorted(len(v) for v in cuenca.values())
    print(f"\n  celdas con cuenca aguas arriba: {len(cuenca):,}")
    if tam:
        print(f"  tamaño de cuenca: mediana {tam[len(tam)//2]} · "
              f"máximo {tam[-1]} celdas")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps({
        "metodo": ("D8 sobre la grilla de 0,1°: cada celda drena al vecino de los "
                   "ocho con mayor pendiente descendente, y la cuenca es el "
                   "conjunto que llega hasta ella siguiendo ese grafo."),
        "elevacion": "Copernicus GLO-30, mediana de una ventana de 3×3 muestras",
        "alcance": f"latitud {LAT_MAX} a {LAT_MIN} — donde el valle puede estar "
                   "seco mientras la cordillera recibe el temporal",
        "limite": ("D8 sobre celdas de 9 km no reproduce quebradas individuales: "
                   "dice qué parte de la cordillera alimenta este valle."),
        "max_por_celda": MAX_AGUAS_ARRIBA,
        "por_celda": cuenca,
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    print("=" * 74)
    print("LA CUENCA REAL · relieve en vez de «80 km al este»")
    print("=" * 74)
    if "--elevacion" in sys.argv:
        sys.exit(elevacion())
    if "--cuencas" in sys.argv:
        sys.exit(cuencas())
    print(__doc__)
    sys.exit(1)
