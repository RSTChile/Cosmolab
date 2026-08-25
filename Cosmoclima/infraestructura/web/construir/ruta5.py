"""
LA RUTA 5, SEGMENTADA Y ATADA AL CLIMA
========================================

Convierte el crudo de OpenStreetMap en dos cosas:

  · **la traza para dibujar**, simplificada — 170.000 vértices no se pueden
    mandar al navegador, y a escala de país no se distinguen de 3.000;
  · **los segmentos con su celda climática y su comuna**, para poder preguntar
    «¿qué tramos de la Ruta 5 superan el umbral hoy?».

★ SIMPLIFICACIÓN: DOUGLAS-PEUCKER, Y POR QUÉ NO BASTA «TOMAR UNO DE CADA N»
-----------------------------------------------------------------------------
Quedarse con uno de cada diez vértices es simple y arruina justo lo que
importa: en una recta de 40 km sobran todos, y en una curva cerrada de la cuesta
faltan. Douglas-Peucker conserva los vértices que cambian la forma y descarta
los que están sobre la línea, así que las rectas se comen su propio peso y las
curvas se mantienen.

★ SEGMENTOS DE ~5 km, Y NO LA RUTA ENTERA COMO UNA COSA
---------------------------------------------------------
La Ruta 5 cruza 3.400 km y decenas de regímenes de lluvia: preguntar si «la
Ruta 5» supera un umbral no significa nada. Partida en segmentos, cada uno cae
en una celda climática y en una comuna, y la pregunta pasa a tener respuesta:
**qué tramos**, con su kilómetro y su comuna.

USO
---
    ../../.venv-esa/bin/python construir/ruta5.py
"""

import json
import sys
from math import cos, hypot, radians
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
RAIZ = AQUI.parent.parent
DATOS = AQUI.parent / "publico" / "datos"
CRUDO = RAIZ / "datos" / "crudo" / "ruta5"
SALIDA = DATOS / "ruta5.json"

MALLA = 0.10
TOLERANCIA = 0.0016      # ~150 m: por debajo de eso no se ve a ninguna escala útil
SEGMENTO_KM = 5.0

import activos as A  # noqa: E402


def celda(la, lo):
    return f"{round(la/MALLA)}_{round(lo/MALLA)}"


def km(a, b):
    """Distancia aproximada en km entre dos (lat, lon)."""
    dy = (b[0] - a[0]) * 111.32
    dx = (b[1] - a[1]) * 111.32 * cos(radians((a[0] + b[0]) / 2))
    return hypot(dx, dy)


def douglas_peucker(pts, tol):
    if len(pts) < 3:
        return pts
    # el vértice más alejado de la recta que une los extremos
    x0, y0 = pts[0][1], pts[0][0]
    x1, y1 = pts[-1][1], pts[-1][0]
    dx, dy = x1 - x0, y1 - y0
    norma = hypot(dx, dy) or 1e-12
    peor, dpeor = 0, 0.0
    for i in range(1, len(pts) - 1):
        px, py = pts[i][1], pts[i][0]
        d = abs(dy * (px - x0) - dx * (py - y0)) / norma
        if d > dpeor:
            peor, dpeor = i, d
    if dpeor <= tol:
        return [pts[0], pts[-1]]
    return (douglas_peucker(pts[:peor + 1], tol)[:-1]
            + douglas_peucker(pts[peor:], tol))


def main():
    dias = sorted(CRUDO.glob("*")) if CRUDO.exists() else []
    if not dias:
        print("  falta el crudo — corre adaptadores/ruta5.py")
        return 1
    d = json.loads((dias[-1] / "overpass_ruta5.json").read_text(encoding="utf-8"))
    rels = [x for x in d.get("elements", []) if x.get("type") == "relation"]
    print(f"  relaciones: {len(rels)}")

    # ── unir la geometría de todos los tramos ───────────────────────────────
    # ★ Deduplicar por id de tramo: 40 de los 11.538 aparecen en más de una
    #   relación, porque los sectores se solapan en los límites regionales.
    lineas = []
    vistos = set()
    for r in rels:
        for m in r.get("members", []):
            g = m.get("geometry")
            if not g or len(g) < 2 or m.get("ref") in vistos:
                continue
            vistos.add(m.get("ref"))
            lineas.append([(p["lat"], p["lon"]) for p in g])
    # ★ Y los tramos SUELTOS, que en el extremo norte son la única forma en que
    #   la Ruta 5 existe en OSM: entre Quillagua y la frontera con Perú no hay
    #   relación que los agrupe. El `vistos` evita contarlos dos veces cuando ya
    #   venían dentro de una relación.
    sueltos = 0
    for w in d.get("elements", []):
        if w.get("type") != "way" or w.get("id") in vistos:
            continue
        g = w.get("geometry")
        if not g or len(g) < 2:
            continue
        vistos.add(w["id"])
        lineas.append([(p["lat"], p["lon"]) for p in g])
        sueltos += 1
    print(f"  tramos sueltos añadidos: {sueltos:,}")
    brutos = sum(len(l) for l in lineas)
    print(f"  tramos {len(lineas):,} · vértices en bruto {brutos:,}")

    # ── simplificar cada tramo ──────────────────────────────────────────────
    simples = []
    sys.setrecursionlimit(20000)
    for l in lineas:
        simples.append(douglas_peucker(l, TOLERANCIA))
    finos = sum(len(l) for l in simples)
    print(f"  vértices tras simplificar: {finos:,} "
          f"({100*finos/max(brutos,1):.1f} % del original)")

    # ── segmentar por distancia acumulada ───────────────────────────────────
    print("  resolviendo comuna de cada segmento…", flush=True)
    indice = A.cargar_geometria()
    cache = {}
    segmentos = []
    acumulado = 0.0
    inicio = None
    ultimo = None
    largo = 0.0
    for l in simples:
        for p in l:
            if inicio is None:
                inicio, ultimo = p, p
                continue
            paso = km(ultimo, p)
            # saltos grandes = tramo nuevo, no continuidad
            if paso > 20:
                inicio, ultimo, largo = p, p, 0.0
                continue
            largo += paso
            acumulado += paso
            ultimo = p
            if largo >= SEGMENTO_KM:
                medio = ((inicio[0] + p[0]) / 2, (inicio[1] + p[1]) / 2)
                k = (round(medio[0], 2), round(medio[1], 2))
                if k not in cache:
                    cache[k] = A.resolver(medio[1], medio[0], indice)
                # ⚠️ NO se guarda «kilómetro de la ruta». El kilometraje oficial
                #    se cuenta desde Santiago y no se puede reconstruir sumando
                #    tramos de OSM: daría un número con pinta de oficial que no
                #    coincidiría con ningún hito de la carretera. Cada segmento
                #    se identifica por su comuna y su latitud, que es lo que la
                #    pregunta necesita y sí es cierto.
                segmentos.append({
                    "y": round(medio[0], 4), "x": round(medio[1], 4),
                    "c": celda(medio[0], medio[1]),
                    "cut": cache[k] or "",
                })
                inicio, largo = p, 0.0

    todas = [p for l in simples for p in l]
    if todas:
        print(f"  cobertura: lat {max(p[0] for p in todas):.2f} "
              f"a {min(p[0] for p in todas):.2f}")
    con_comuna = sum(1 for s in segmentos if s["cut"])
    print(f"  segmentos de ~{SEGMENTO_KM:g} km: {len(segmentos):,} "
          f"· con comuna resuelta: {con_comuna:,}")
    # ⚠️ Este largo NO es el de la Ruta 5: suma las DOS calzadas de cada
    #    autopista, más ramales y variantes. La ruta mide ~3.400 km.
    print(f"  traza total (ambas calzadas): {acumulado:,.0f} km")

    SALIDA.write_text(json.dumps({
        "fuente": "OpenStreetMap · relaciones route=road ref=5, vía Overpass",
        "bajado": dias[-1].name,
        "advertencia": ("La Ruta 5 NO está en el catastro de la Matriz: de sus "
                        "14.036 activos viales, 1.522 la nombran como referencia "
                        "(«Cruce Ruta 5 …») y sólo 7 empiezan por ella. Esto es "
                        "dato público traído al lado, no catastro de la MICR."),
        "simplificacion": f"Douglas-Peucker con tolerancia {TOLERANCIA}° (~150 m)",
        "km_por_segmento": SEGMENTO_KM,
        "traza_km": round(acumulado),
        "nota_largo": ("La traza suma las dos calzadas de cada autopista más "
                       "ramales y variantes, así que NO es el largo de la Ruta 5 "
                       "(unos 3.400 km). Y no se publica kilometraje: el oficial "
                       "se cuenta desde Santiago y no se puede reconstruir aquí."),
        "trazas": [[[round(la, 4), round(lo, 4)] for la, lo in l] for l in simples],
        "segmentos": segmentos,
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    print("=" * 74)
    print("RUTA 5 · simplificada, segmentada y atada a comuna y celda")
    print("=" * 74)
    sys.exit(main())
