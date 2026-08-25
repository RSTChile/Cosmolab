"""
HÍDRICO · los puntos donde el país saca agua
==============================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «completa Hídrico (38), Energía (36) y
Alimentario (26)».

★ QUÉ SE ENCONTRÓ, Y QUÉ YA ESTABA
------------------------------------
Antes de bajar nada hubo que mirar qué había: **los relaves, los embalses y los
sistemas de agua potable rural YA estaban cargados** —en los ítems 42, 46 y 17—.
Volver a cargarlos habría inflado el índice contando dos veces lo mismo. Lo que
faltaba de verdad era el registro del **SNIA de la Dirección General de Aguas**:

    76.488  derechos de aprovechamiento, cada uno con su punto de captación
     5.405  obras hidráulicas mayores

★★ POR QUÉ SÓLO SE ASIGNAN LOS SUPERFICIALES AL ÍTEM 6
--------------------------------------------------------
El ítem 6 es «Puntos de Captación de Agua (**Water Intake**)», que en el lenguaje
técnico es la toma de agua **superficial**. El agua subterránea se saca por pozo,
y la Matriz tiene dos ítems para eso:

    7  Pozos Artesianos            el agua sube sola por presión del acuífero
    8  Pozos Superficiales (No Artesianos)   requieren bombeo

El registro de la DGA dice si el recurso es superficial o subterráneo, y **no
dice si un pozo es artesiano**. Es una distinción hidrogeológica que los
catastros administrativos no llevan. Por eso los **47.528 derechos subterráneos
quedan contados y sin asignar**: repartirlos entre los ítems 7 y 8 sería
inventar la mitad del resultado.

Es el mismo patrón que apareció en Energía con carbón/gas/diésel y en el RETC con
los rubros: **la Matriz separa más fino de lo que cualquier catastro nacional
registra**. Conviene que quede anotado cada vez que ocurre.

⚠️ Y UNA ADVERTENCIA SOBRE QUÉ ES ESTE DATO: es un registro de **derechos**, no un
catastro de obras construidas. El punto dice dónde está autorizada la captación.
Que exista el derecho no prueba que la obra esté hecha ni operando.

USO
---
    ../.venv-esa/bin/python poblar_hidrico.py
"""

import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI / "datos" / "crudo" / "mop_sit"
SALIDA = AQUI / "datos" / "hidrico_por_item.csv"

# naturaleza declarada en el SNIA → ítem de la Matriz
SUPERFICIAL = ("superficial", "superficial y corriente", "superficial y detenida",
               "superf. y detenida", "superficial corrientes/detenidas")
ITEM_CAPTACION = ("6", "Puntos de Captación de Agua (Water Intake)")
# obras mayores: sólo las que la propia DGA nombra como obra de conducción
CONDUCCION = {
    "canal": ("14", "Canales de Transporte de Agua (No Agrícola)"),
    "sifones": ("14", "Canales de Transporte de Agua (No Agrícola)"),
    "acueductos (> 1/2m3)": ("14", "Canales de Transporte de Agua (No Agrícola)"),
    "acueductos (> 2m3)": ("14", "Canales de Transporte de Agua (No Agrícola)"),
    "canoas": ("14", "Canales de Transporte de Agua (No Agrícola)"),
}


def ultimo(nombre):
    arch = sorted(glob.glob(str(CRUDO / "*" / f"{nombre}.geojson")))
    if not arch:
        return None
    return json.loads(Path(arch[-1]).read_text(encoding="utf-8"))["features"]


def punto(f):
    g = f.get("geometry")
    if not g or g.get("type") != "Point":
        return None
    lo, la = g["coordinates"][0], g["coordinates"][1]
    if not (-56 < la < -17 and -110 < lo < -66):
        return None
    return la, lo


def main():
    der = ultimo("derechos_agua")
    obras = ultimo("obras_mayores")
    if not der:
        print("  falta el espejado del SNIA")
        return 1
    print(f"  derechos de aprovechamiento: {len(der):,}")

    filas = []
    subterraneos = fuera = 0
    for f in der:
        nat = str(f["properties"].get("NatDesc") or "").strip().lower()
        p = punto(f)
        if p is None:
            fuera += 1
            continue
        if nat not in SUPERFICIAL:
            subterraneos += 1
            continue
        filas.append({
            "item": ITEM_CAPTACION[0], "elemento": ITEM_CAPTACION[1],
            "naturaleza": nat,
            "nombre": str(f["properties"].get("NombreSolicitante") or "")[:70],
            "cuenca": str(f["properties"].get("CueNombre") or "").strip()[:50],
            "region": str(f["properties"].get("RegNombre") or "").strip(),
            "comuna": str(f["properties"].get("ComNombre") or "").strip(),
            "lat": round(p[0], 6), "lon": round(p[1], 6),
        })
    print(f"    superficiales → ítem {ITEM_CAPTACION[0]}: {len(filas):,}")
    print(f"    ⚠️ subterráneos SIN asignar: {subterraneos:,} "
          f"(la DGA no declara si el pozo es artesiano; los ítems 7 y 8 lo exigen)")
    if fuera:
        print(f"    fuera del territorio o sin punto: {fuera:,}")

    if obras:
        print(f"\n  obras hidráulicas mayores: {len(obras):,}")
        n, sin_tipo = 0, Counter()
        for f in obras:
            t = str(f["properties"].get("TobDesc") or "").strip().lower()
            clave = CONDUCCION.get(t)
            p = punto(f)
            if not clave or p is None:
                sin_tipo[t or "(sin tipo declarado)"] += 1
                continue
            filas.append({
                "item": clave[0], "elemento": clave[1], "naturaleza": t,
                "nombre": str(f["properties"].get("NombProyecto")
                              or f["properties"].get("NombreSolicitante") or "")[:70],
                "cuenca": str(f["properties"].get("CueNombre") or "").strip()[:50],
                "region": str(f["properties"].get("RegNombre") or "").strip(),
                "comuna": str(f["properties"].get("ComNombre") or "").strip(),
                "lat": round(p[0], 6), "lon": round(p[1], 6),
            })
            n += 1
        print(f"    obras de conducción → ítem 14: {n:,}")
        print(f"    ⚠️ sin tipo de obra utilizable: "
              f"{sum(sin_tipo.values()):,} → {[t for t, _ in sin_tipo.most_common(3)]}")

    por = Counter((f["item"], f["elemento"]) for f in filas)
    print(f"\n  {'ítem':<6}{'elemento':<48}{'activos':>9}")
    print("  " + "-" * 64)
    for (n, el), c in sorted(por.items(), key=lambda t: -t[1]):
        print(f"  {n:<6}{el[:47]:<48}{c:>9,}")

    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e6:.1f} MB)")
    print("\n  ⚠️ Es un registro de DERECHOS, no de obras construidas: el punto")
    print("     dice dónde está autorizada la captación, no que esté hecha.")
    return 0


if __name__ == "__main__":
    print("=" * 78)
    print("HÍDRICO · los puntos de captación del SNIA")
    print("=" * 78)
    sys.exit(main())
