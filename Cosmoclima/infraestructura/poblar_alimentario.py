"""
ALIMENTARIO · el sector que estaba en CERO
============================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «completa Hídrico (38), Energía (36) y
Alimentario (26)».

★ EL PUNTO DE PARTIDA
-----------------------
De los tres, el Alimentario era el único **sin ninguna fuente en disco**: 25
ítems físicos, cero activos, y nada que integrar. Hubo que ir a buscar dos
catastros públicos que el proyecto no tenía:

    4.690  concesiones de acuicultura     SUBPESCA · geoportal IDE_PUBLICO
   13.166  canales de riego               Comisión Nacional de Riego · SIT-MOP

★★ EL GRUPO DE ESPECIE ES LO QUE HACE HONESTO EL NÚMERO
---------------------------------------------------------
«Acuicultura» no es «granja piscícola». De las 4.690 concesiones, buena parte son
de moluscos, algas y equinodermos, y la Matriz sólo tiene el ítem **401 · Granjas
Piscícolas (Peces)**. SUBPESCA declara el grupo de especie de cada concesión, así
que los peces van al 401 y el resto queda contado y declarado, sin ítem.

★ POR QUÉ LOS CANALES DE RIEGO VAN AL 403 Y NO AL 14
------------------------------------------------------
La Matriz tiene los dos, y los distingue justamente por esto:

    403  Sistemas de Riego Agrícola                    ← los canales de la CNR
     14  Canales de Transporte de Agua (No Agrícola)   ← explícitamente, no riego

El catastro de la Comisión Nacional de Riego es de canales **de riego**. Ponerlos
en el ítem 14 contradiría el nombre del propio ítem.

⚠️ UN CANAL NO ES UN PUNTO. Los 13.166 canales son líneas de hasta decenas de
kilómetros y el resto de la aplicación trabaja con puntos. Se toma el **vértice
central de la traza** —no el centroide, que en un canal curvo puede caer fuera
del canal— y se guarda además su longitud, para que se vea que el punto
representa algo extenso.

USO
---
    ../.venv-esa/bin/python poblar_alimentario.py
"""

import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "datos" / "alimentario_por_item.csv"

# grupo de especie declarado por SUBPESCA → ítem de la Matriz
PECES = {"SALMONES", "PECES"}
ITEM_PECES = ("401", "Granjas Piscícolas (Peces)")
ITEM_RIEGO = ("403", "Sistemas de Riego Agrícola")


def leer(patron):
    arch = sorted(glob.glob(str(AQUI / "datos" / "crudo" / patron)))
    if not arch:
        return None
    return json.loads(Path(arch[-1]).read_text(encoding="utf-8"))["features"]


def dentro(la, lo):
    return -56 < la < -17 and -110 < lo < -66


def vertice_central(geom):
    """★ El punto que representa una línea.

    Se elige el vértice del medio de la traza y no el centroide del conjunto de
    vértices: en un canal que rodea un cerro, el centroide cae en el cerro.
    """
    t = geom.get("type")
    if t == "LineString":
        pts = geom["coordinates"]
    elif t == "MultiLineString":
        pts = [p for camino in geom["coordinates"] for p in camino]
    else:
        return None, 0.0
    if not pts:
        return None, 0.0
    lo, la = pts[len(pts) // 2][:2]
    # largo aproximado en km, sólo para dar escala al punto
    largo = 0.0
    for a, b in zip(pts, pts[1:]):
        largo += (((b[0] - a[0]) * 92.0) ** 2 + ((b[1] - a[1]) * 111.32) ** 2) ** 0.5
    return (la, lo), largo


def main():
    filas = []

    # ── acuicultura ─────────────────────────────────────────────────────────
    ac = leer("subpesca/*/acuicultura.geojson")
    if ac:
        print(f"  concesiones de acuicultura: {len(ac):,}")
        grupos = Counter()
        n = fuera = 0
        for f in ac:
            p = f["properties"]
            grupo = str(p.get("T_GRUPOESPECIE") or "").strip().upper()
            grupos[grupo or "(sin grupo)"] += 1
            if grupo not in PECES:
                continue
            g = f.get("geometry")
            if not g:
                fuera += 1
                continue
            lo, la = g["coordinates"][0], g["coordinates"][1]
            if not dentro(la, lo):
                fuera += 1
                continue
            filas.append({
                "item": ITEM_PECES[0], "elemento": ITEM_PECES[1],
                "detalle": grupo,
                "nombre": str(p.get("TITULAR") or "")[:70],
                "extra": str(p.get("ESPECIES") or "")[:50],
                "region": str(p.get("REGION") or ""), "comuna": str(p.get("COMUNA") or ""),
                "lat": round(la, 6), "lon": round(lo, 6),
            })
            n += 1
        print(f"    peces → ítem {ITEM_PECES[0]}: {n:,}")
        print(f"    ⚠️ SIN ítem en la Matriz (acuicultura que no es de peces):")
        for g, c in grupos.most_common():
            if g not in PECES:
                print(f"         {g[:34]:<36}{c:>6,}")
        if fuera:
            print(f"    fuera del territorio o sin punto: {fuera}")

    # ── canales de riego ────────────────────────────────────────────────────
    can = leer("mop_sit/*/canales_cnr.geojson")
    if can:
        print(f"\n  canales de riego de la CNR: {len(can):,}")
        n = fuera = 0
        km = 0.0
        for f in can:
            g = f.get("geometry")
            if not g:
                fuera += 1
                continue
            pt, largo = vertice_central(g)
            if pt is None or not dentro(*pt):
                fuera += 1
                continue
            p = f["properties"]
            filas.append({
                "item": ITEM_RIEGO[0], "elemento": ITEM_RIEGO[1],
                "detalle": f"{largo:.1f} km",
                "nombre": str(p.get("NOMCAN") or "").strip()[:70],
                "extra": str(p.get("NOMFUENHID") or "").strip()[:50],
                "region": str(p.get("NOMREG") or "").strip(),
                "comuna": str(p.get("NOMCOM") or "").strip(),
                "lat": round(pt[0], 6), "lon": round(pt[1], 6),
            })
            n += 1
            km += largo
        print(f"    → ítem {ITEM_RIEGO[0]}: {n:,} canales · {km:,.0f} km de traza")
        if fuera:
            print(f"    sin geometría utilizable: {fuera}")

    por = Counter((f["item"], f["elemento"]) for f in filas)
    print(f"\n  {'ítem':<6}{'elemento':<44}{'activos':>9}")
    print("  " + "-" * 60)
    for (n, el), c in sorted(por.items(), key=lambda t: -t[1]):
        print(f"  {n:<6}{el[:43]:<44}{c:>9,}")

    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\n  escrito: {SALIDA.name}")
    return 0


if __name__ == "__main__":
    print("=" * 76)
    print("ALIMENTARIO · acuicultura y canales de riego")
    print("=" * 76)
    sys.exit(main())
