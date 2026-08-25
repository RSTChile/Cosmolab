"""
LAS 283 MUNICIPALIDADES QUE EL RETC TENÍA Y NADIE HABÍA MIRADO
================================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «revisa todos los datos de los que
disponemos, no creo que nos falten datos, falta integrarlos».

★ EL HUECO
------------
El sector Gobierno tiene 44 ítems y sólo 2 con activos. Mientras tanto, en el
mismo archivo del RETC del que salió el sector Químico, había un rubro sin tocar:

    Municipio    283 establecimientos, todos «ILUSTRE MUNICIPALIDAD DE …»

★★ POR QUÉ ESTE RUBRO SÍ SE ASIGNA Y LOS OTROS DIECIOCHO NO
--------------------------------------------------------------
El RETC trae 19 rubros. La mayoría **no identifica un ítem de la Matriz**:

    «Producción de alimentos» (645)   ¿plantas de carnes, de lácteos, de
                                      envasado? La Matriz los separa; el rubro no
    «Industria manufacturera» (880)   ¿maquinaria, vehículos, electrónica?
    «Comercio» (3.258)                ¿centro comercial o tienda minorista?

Repartirlos sería inventar a qué se dedica cada uno, y el número de «activos del
ítem X» dejaría de ser cierto. **«Municipio» es la excepción**: se verificó
establecimiento por establecimiento y los 283 son sedes municipales, que es
exactamente el ítem 575 · Oficinas Municipales.

⚠️ Y NO SE MAPEAN «Generación de energía» (278) ni «Transmisión y distribución»
(174), aunque serían inequívocos: esos activos **ya están** en el proyecto desde
el catastro del Coordinador Eléctrico, que es mejor fuente. Meterlos otra vez
sería doble conteo.

USO
---
    ../.venv-esa/bin/python poblar_municipios.py
"""

import csv
import glob
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI / "datos" / "crudo" / "retc"
SALIDA = AQUI / "datos" / "municipios_por_item.csv"

ITEM, ELEMENTO = "575", "Oficinas Municipales"


def main():
    arch = sorted(glob.glob(str(CRUDO / "*" / "establecimientos_ckan.xlsx")))
    if not arch:
        print("  falta el espejado del RETC")
        return 1
    import openpyxl

    ws = openpyxl.load_workbook(arch[-1], read_only=True)["Hoja1"]
    it = ws.iter_rows(values_only=True)
    cab = list(next(it))
    ic = {str(c): i for i, c in enumerate(cab) if c}

    filas, sin_coord, no_municipal = [], 0, []
    for f in it:
        if str(f[ic["Rubro RETC"]] or "").strip() != "Municipio":
            continue
        nombre = str(f[ic["Nombre de Establecimiento"]] or "").strip()
        # ★ la verificación: si algún día entra algo que no es una sede
        #   municipal, este chequeo lo deja fuera y lo dice, en vez de
        #   asignarlo al ítem 575 en silencio.
        if "MUNICIPALIDAD" not in nombre.upper():
            no_municipal.append(nombre[:50])
            continue
        try:
            la, lo = float(f[ic["Latitude"]]), float(f[ic["Longitude"]])
        except (TypeError, ValueError):
            sin_coord += 1
            continue
        if not (-56 < la < -17 and -110 < lo < -66):
            sin_coord += 1
            continue
        filas.append({
            "item": ITEM, "elemento": ELEMENTO, "rubro": "Municipio",
            "nombre": nombre[:70],
            "razon_social": str(f[ic["Razón social"]] or "")[:70],
            "region": f[ic["Región"]], "comuna": f[ic["Comuna"]],
            "lat": round(la, 6), "lon": round(lo, 6),
        })

    print(f"  rubro «Municipio» en el RETC: "
          f"{len(filas) + sin_coord + len(no_municipal):,}")
    print(f"  asignados al ítem {ITEM} · {ELEMENTO}: {len(filas):,}")
    if sin_coord:
        print(f"  ⚠️ sin coordenada utilizable: {sin_coord}")
    if no_municipal:
        print(f"  ⚠️ del rubro pero sin «municipalidad» en el nombre: "
              f"{len(no_municipal)} → {no_municipal[:3]}")

    com = {f["comuna"] for f in filas}
    print(f"  comunas cubiertas: {len(com)} de 345 "
          f"({100*len(com)/345:.0f} %)")

    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\n  escrito: {SALIDA.name}")
    return 0


if __name__ == "__main__":
    print("=" * 72)
    print("MUNICIPALIDADES · desde el RETC, al ítem 575")
    print("=" * 72)
    sys.exit(main())
