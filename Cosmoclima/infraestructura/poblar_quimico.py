"""
EL SECTOR QUÍMICO · y lo que el caso de Quilicura deja al descubierto
======================================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «otro sector no incluido es el Químico, y sí
tenemos ese sector desarrollado particularmente en Santiago. De hecho, hace dos
semanas hubo un incendio grande en una planta química de Quilicura, y se debió
decretar cierre de colegios por la emanación de tóxicos al aire.»

★ LA FUENTE
-------------
El **RETC** —Registro de Emisiones y Transferencias de Contaminantes del
Ministerio del Medio Ambiente— publica los establecimientos registrados en
ventanilla única: **26.649 con coordenada, el 100 %**. Es el catastro que faltaba
para un sector que en la Matriz tenía 44 ítems y cero activos.

Se reparten por rubro declarado:

    Químicos                →  ítem 660 · Plantas Químicas
    Combustibles            →  ítem 670 · Tanques de Almacenamiento
    Gestor de residuos      →  ítem 661 · Almacenes de Químicos Peligrosos

⚠️ El rubro «Otras actividades» agrupa 8.701 establecimientos —un tercio del
total— y no se reparte: meterlos en un ítem concreto sería inventar a qué se
dedican. Quedan contados y sin asignar, que es lo que el dato permite decir.

★★ LO QUE ESTE CASO DEJA AL DESCUBIERTO, Y NO SE ARREGLA CON DATOS
--------------------------------------------------------------------
Todo lo que este proyecto modela responde «qué le pasa a este activo». Quilicura
es lo contrario: **el activo daña a otros**. La planta arde y hay que cerrar
colegios a kilómetros, que están intactos.

Es efecto en cadena, y la Matriz no lo tiene. Tampoco lo tiene para el caso
equivalente que ya apareció con la Ruta 5: un corte de camino aísla hospitales
que no sufrieron ningún daño.

No se resuelve agregando fuentes: pide una capa de **relaciones entre activos**
que hoy no existe. Este archivo deja anotado el hueco y, mientras tanto, calcula
lo único que sí se puede calcular sin inventar nada: **qué hay alrededor de cada
establecimiento químico**, sin afirmar que se vería afectado.

USO
---
    ../.venv-esa/bin/python poblar_quimico.py
"""

import csv
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI / "datos" / "crudo" / "retc"
FUENTE_ACTIVOS = AQUI / "datos" / "fuente_climatica_por_activo.csv"
SALIDA = AQUI / "datos" / "quimico_por_item.csv"
SAL_ENTORNO = AQUI / "datos" / "quimico_entorno.csv"

# rubro declarado en el RETC → ítem de la Matriz
# ⚠️ Los nombres son los EXACTOS del RETC, verificados uno por uno. El primer
#    intento usó «químicos» y no capturó ninguna planta: el rubro se llama
#    «Producción química». Se poblaron los otros ítems y el 660 —justo el que
#    se pedía— quedó vacío sin que nada avisara.
MAPEO = {
    "producción química": ("660", "Plantas Químicas"),
    "produccion quimica": ("660", "Plantas Químicas"),
    "combustibles": ("670", "Tanques de Almacenamiento"),
    "gestor de residuos": ("661", "Almacenes de Químicos Peligrosos"),
    # Estos dos manejan volúmenes de sustancias peligrosas equivalentes y la
    # Matriz no les da ítem propio; entran como almacenes de químicos.
    "industria del papel y celulosa": ("661", "Almacenes de Químicos Peligrosos"),
    "producción de metal": ("661", "Almacenes de Químicos Peligrosos"),
}
# radios de referencia para mirar el entorno, en km
RADIOS = (0.5, 1.0, 2.0)


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

    filas, sin_item = [], Counter()
    for f in it:
        try:
            la, lo = float(f[ic["Latitude"]]), float(f[ic["Longitude"]])
        except (TypeError, ValueError):
            continue
        if not (-56 < la < -17 and -110 < lo < -66):
            continue
        rubro = str(f[ic["Rubro RETC"]] or "").strip()
        clave = MAPEO.get(rubro.lower())
        if not clave:
            sin_item[rubro or "(sin rubro)"] += 1
            continue
        filas.append({
            "item": clave[0], "elemento": clave[1], "rubro": rubro,
            "nombre": str(f[ic["Nombre de Establecimiento"]] or "")[:70],
            "razon_social": str(f[ic["Razón social"]] or "")[:70],
            "region": f[ic["Región"]], "comuna": f[ic["Comuna"]],
            "lat": round(la, 6), "lon": round(lo, 6),
        })

    print(f"  establecimientos del RETC con coordenada: "
          f"{len(filas) + sum(sin_item.values()):,}")
    print(f"  asignados a un ítem de la Matriz: {len(filas):,}\n")
    por = Counter((f["item"], f["elemento"]) for f in filas)
    print(f"  {'ítem':<6}{'elemento':<38}{'activos':>9}")
    print("  " + "-" * 54)
    for (n, el), c in sorted(por.items()):
        print(f"  {n:<6}{el:<38}{c:>9,}")

    print(f"\n  ⚠️ rubros que NO se reparten (meterlos en un ítem sería inventar")
    print(f"     a qué se dedican):")
    for r, n in sin_item.most_common(6):
        print(f"       {r[:44]:<46}{n:>7,}")

    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\n  escrito: {SALIDA.name}")

    # ── qué hay alrededor: lo único que se puede decir sin inventar ─────────
    if not FUENTE_ACTIVOS.exists():
        return 0
    print("\n  midiendo qué hay alrededor de cada establecimiento…", flush=True)
    rej = defaultdict(list)
    with FUENTE_ACTIVOS.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                la, lo = float(r["lat"]), float(r["lon"])
            except (TypeError, ValueError):
                continue
            rej[(round(la, 1), round(lo, 1))].append((la, lo, r["item"], r["activo"]))

    def cerca(la, lo, km):
        out = []
        for dla in (-0.1, 0, 0.1):
            for dlo in (-0.1, 0, 0.1):
                for a in rej.get((round(la + dla, 1), round(lo + dlo, 1)), ()):
                    dy = (a[0] - la) * 111.32
                    dx = (a[1] - lo) * 111.32 * math.cos(math.radians(la))
                    if math.hypot(dx, dy) <= km:
                        out.append(a)
        return out

    ent = []
    for f in filas:
        fila = {k: f[k] for k in ("item", "nombre", "comuna", "lat", "lon")}
        for km in RADIOS:
            v = cerca(f["lat"], f["lon"], km)
            fila[f"activos_{km:g}km"] = len(v)
            # los sensibles: educación (441), salud (265), adultos mayores (836),
            # protección de la infancia (837)
            fila[f"sensibles_{km:g}km"] = sum(
                1 for a in v if a[2] in ("441", "265", "836", "837"))
        ent.append(fila)

    with SAL_ENTORNO.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ent[0].keys()))
        w.writeheader()
        w.writerows(ent)

    med = lambda v: sorted(v)[len(v) // 2] if v else 0
    print(f"\n  {'radio':<10}{'activos alrededor (mediana)':>30}{'sensibles':>12}")
    print("  " + "-" * 54)
    for km in RADIOS:
        a = [e[f"activos_{km:g}km"] for e in ent]
        s = [e[f"sensibles_{km:g}km"] for e in ent]
        print(f"  {km:g} km{'':<5}{med(a):>28}{med(s):>12}")

    peor = sorted(ent, key=lambda e: -e["sensibles_1km"])[:6]
    print("\n  los 6 con más establecimientos sensibles a 1 km:")
    for e in peor:
        print(f"     {e['nombre'][:40]:<42}{str(e['comuna'])[:16]:<18}"
              f"{e['sensibles_1km']:>4} sensibles de {e['activos_1km']}")
    print(f"\n  escrito: {SAL_ENTORNO.name}")
    print("\n  ⚠️ «Qué hay alrededor» NO es «qué se vería afectado»: la dispersión")
    print("     de una nube tóxica depende del viento, del compuesto y del tipo de")
    print("     evento, y nada de eso está medido aquí.")
    return 0


if __name__ == "__main__":
    print("=" * 76)
    print("EL SECTOR QUÍMICO · desde el RETC")
    print("=" * 76)
    sys.exit(main())
