"""
LAS ESTACIONES BASE QUE YA ESTABAN BAJADAS
============================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «creo que tenemos muchos datos sin explotar,
pero los tenemos y sólo tenemos que explorar».

★ EL HUECO
------------
El sector Telecomunicaciones tiene 44 ítems y **43 estaban vacíos**: sólo el 183,
«Torres de Telecomunicaciones», tenía activos. Mientras tanto, espejado desde el
20 de agosto y sin usar:

    estaciones_base.geojson.gz     58.131 elementos
    antenas_autorizadas            52.412
    antenas_en_servicio            29.875

Las estaciones base traen **su generación declarada**, que es exactamente cómo la
Matriz separa sus ítems:

    4G  32.287  →  ítem 185 · Estaciones Base 4G (BTS)
    5G   6.931  →  ítem 186 · Estaciones Base 5G (gNodeB)
    3G  16.805  →  sin ítem propio en la Matriz
    2G   2.108  →  sin ítem propio en la Matriz

★★ QUÉ SE HACE CON 3G Y 2G, Y POR QUÉ NO SE INVENTA
------------------------------------------------------
La Matriz no tiene ítem para ellas. **No se meten en el de 4G**: son 18.913
activos que quedarían contados como algo que no son, y el número de «estaciones
4G» dejaría de ser cierto. Se cuentan aparte y se declara que la Matriz no las
contempla — que es un hallazgo sobre la Matriz, no un problema del dato.

⚠️ Esto es CONSOLIDACIÓN, no catastro propio: el dato viene de Subtel y sólo se
está conectando con el ítem que le corresponde.

USO
---
    ../.venv-esa/bin/python poblar_telecom.py
"""

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI / "datos" / "crudo" / "subtel"
SALIDA = AQUI / "datos" / "telecom_por_item.csv"

# generación declarada por Subtel → ítem de la Matriz
MAPEO = {"4G": "185", "5G": "186"}


def main():
    dias = sorted(CRUDO.glob("*"))
    if not dias:
        print("  falta el espejado de Subtel")
        return 1
    arch = dias[-1] / "estaciones_base.geojson.gz"
    if not arch.exists():
        print(f"  falta {arch.name}")
        return 1

    with gzip.open(arch, "rt", encoding="utf-8") as fh:
        fs = json.load(fh)["features"]
    print(f"  estaciones base espejadas: {len(fs):,}")

    filas = []
    sin_item = Counter()
    sin_coord = 0
    for f in fs:
        g = f.get("geometry")
        p = f["properties"]
        if not g or not g.get("coordinates"):
            sin_coord += 1
            continue
        lo, la = g["coordinates"][0], g["coordinates"][1]
        gen = (p.get("tite_cod") or "").strip()
        item = MAPEO.get(gen)
        if not item:
            sin_item[gen or "(sin generación)"] += 1
            continue
        filas.append({
            "item": item,
            "generacion": gen,
            "empresa": p.get("empresa") or "",
            "soporte": p.get("tsu_codigo") or "",
            "zona": "urbana" if p.get("tiem_cod") == "U" else "rural",
            "lat": round(la, 6),
            "lon": round(lo, 6),
        })

    print(f"  con coordenada y generación mapeable: {len(filas):,}")
    if sin_coord:
        print(f"  sin coordenada: {sin_coord:,}")
    print(f"\n  ⚠️ generaciones que la Matriz NO contempla:")
    for g, n in sin_item.most_common():
        print(f"       {g:<20}{n:>8,}   (no se asignan a ningún ítem)")

    por = Counter(f["item"] for f in filas)
    print(f"\n  {'ítem':<6}{'elemento':<34}{'activos':>10}")
    print("  " + "-" * 52)
    nombres = {"185": "Estaciones Base 4G (BTS)", "186": "Estaciones Base 5G (gNodeB)"}
    for it, n in sorted(por.items()):
        print(f"  {it:<6}{nombres.get(it, ''):<34}{n:>10,}")

    emp = Counter(f["empresa"] for f in filas)
    print(f"\n  por operador: " + " · ".join(f"{k} {v:,}" for k, v in emp.most_common(5)))
    zon = Counter(f["zona"] for f in filas)
    print(f"  por zona: " + " · ".join(f"{k} {v:,}" for k, v in zon.most_common()))

    import csv
    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e6:.1f} MB)")
    print("\n  ⚠️ Falta el paso siguiente: incorporarlo al índice de activos por")
    print("     comuna, que es lo que consume la aplicación.")
    return 0


if __name__ == "__main__":
    print("=" * 74)
    print("POBLAR TELECOMUNICACIONES · con lo que ya estaba en disco")
    print("=" * 74)
    sys.exit(main())
