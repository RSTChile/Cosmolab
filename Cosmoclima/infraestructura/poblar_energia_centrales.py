"""
LAS 1.208 CENTRALES QUE ESTABAN BAJADAS Y NO LLEGABAN A NINGÚN ÍTEM
=====================================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «completa Hídrico (38), Energía (36) y
Alimentario (26)» — los tres sectores con más hueco de dato del diagnóstico de
cobertura.

★ EL HUECO
------------
El sector Energía tenía 38 ítems físicos y sólo 2 con activos: transmisión (117)
y subestaciones (120). Las **1.208 centrales de generación** del Coordinador
Eléctrico llevaban bajadas desde el 19 de agosto sin conectarse con la Matriz.

★★ POR QUÉ NO BASTABA EL ARCHIVO DE CENTRALES
-----------------------------------------------
`centrales` distingue **cinco** tipos: Solares, Termoeléctricas, Hidroeléctricas,
Eólicas y Geotérmica. La Matriz separa mucho más fino:

    89 carbón · 90 gas natural · 91 diésel     ⊂ «Termoeléctricas»
    95 embalse · 96 bombeo · 97 pasada          ⊂ «Hidroeléctricas»

Meter las 214 termoeléctricas en cualquiera de sus tres ítems sería inventar con
qué se queman. La respuesta estaba en otro endpoint de la misma interfaz:
**`/v1/unidades-generadoras/`**, 1.640 unidades que traen su
`tipo_tecnologia_nombre` — y ahí «Hidroeléctrica de Embalse» y «de Pasada» ya
vienen separadas.

⚠️ LO QUE SIGUE SIN PODERSE: el endpoint declara un campo `id_combustible` y
**viene nulo en las 1.640 unidades**. El catálogo `/v1/combustibles/` existe (52
combustibles, con Carbón, Diésel y Gas Natural entre ellos) y nada apunta a él.
Por eso los ítems 89, 90 y 91 **quedan vacíos a propósito**: el dato existe en el
diseño de la interfaz y no está publicado. Eso es una carencia de la fuente, no
un descuido nuestro, y conviene dejarla anotada.

★★ UNA CENTRAL, UN ACTIVO — NO UNA UNIDAD, UN ACTIVO
------------------------------------------------------
Las 1.640 unidades pertenecen a 1.197 centrales. Lo que la Matriz llama «Parque
Eólico» o «Granja Solar» es la central, no cada turbina o cada inversor. Se
agrupa por central y se le asigna el ítem de la tecnología de sus unidades. Si
una central mezcla tecnologías de ítems distintos —las híbridas— se cuenta aparte
y no se asigna, por la misma razón de siempre.

USO
---
    ../.venv-esa/bin/python poblar_energia_centrales.py
"""

import csv
import glob
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI / "datos" / "crudo" / "coordinador_electrico"
SALIDA = AQUI / "datos" / "energia_por_item.csv"

# tecnología declarada por el Coordinador → ítem de la Matriz
MAPEO = {
    "Fotovoltaica": ("99", "Granjas Solares"),
    "Fotovoltaica + SAE": ("99", "Granjas Solares"),
    "Termosolar": ("101", "Plantas de Concentración Solar de Potencia (CSP)"),
    "Eólica": ("102", "Parques Eólicos"),
    "Geotérmica": ("104", "Plantas Geotérmicas"),
    "Hidroeléctrica de Embalse": ("95", "Represas Hidroeléctricas"),
    "Hidroeléctrica de Pasada": ("97", "Centrales Hidráulicas de Pasada"),
    "Minihidro de pasada": ("97", "Centrales Hidráulicas de Pasada"),
    "Cogeneración - TV": ("116", "Plantas de Cogeneración"),
    "Cogeneración - TG": ("116", "Plantas de Cogeneración"),
}
# ⚠️ Estas quedan fuera a propósito: sin combustible no se sabe si van al ítem
#    89 (carbón), 90 (gas natural) o 91 (diésel).
SIN_COMBUSTIBLE = {
    "Termoeléctrica TG", "Termoeléctrica TV", "Termoeléctrica CC",
    "Motor de Combustión Interna",
}


def ultimo(nombre):
    arch = sorted(glob.glob(str(CRUDO / "*" / f"{nombre}.json.gz")))
    if not arch:
        return None
    with gzip.open(arch[-1], "rt", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    centrales = ultimo("centrales")
    unidades = ultimo("unidades_generadoras")
    if not centrales or not unidades:
        print("  falta el espejado del Coordinador Eléctrico")
        return 1
    print(f"  centrales espejadas: {len(centrales):,}")
    print(f"  unidades generadoras: {len(unidades):,}")

    porcentral = defaultdict(set)
    for u in unidades:
        porcentral[u["id_central"]].add(str(u.get("tipo_tecnologia_nombre")))

    geo = {c["id"]: c for c in centrales}
    filas = []
    sin_tec = sin_comb = hibridas = sin_coord = 0
    tec_no_mapeada = Counter()

    for cid, tecs in porcentral.items():
        c = geo.get(cid)
        if not c:
            sin_tec += 1
            continue
        items = {MAPEO[t][0] for t in tecs if t in MAPEO}
        if not items:
            if tecs & SIN_COMBUSTIBLE:
                sin_comb += 1
            else:
                for t in tecs:
                    tec_no_mapeada[t] += 1
            continue
        if len(items) > 1:
            # central híbrida: sus unidades caen en ítems distintos
            hibridas += 1
            continue
        try:
            la, lo = float(c["latitud"]), float(c["longitud"])
        except (TypeError, ValueError):
            sin_coord += 1
            continue
        if not (-56 < la < -17 and -110 < lo < -66):
            sin_coord += 1
            continue
        item = items.pop()
        elemento = next(v[1] for v in MAPEO.values() if v[0] == item)
        filas.append({
            "item": item, "elemento": elemento,
            "tecnologia": " + ".join(sorted(tecs)),
            "nombre": (c.get("nombre") or "")[:70],
            "propietario": (c.get("propietario_nombre") or "")[:70],
            "potencia_mw": c.get("potencia_maxima"),
            "region": c.get("region_nombre"), "comuna": c.get("comuna_nombre"),
            "lat": round(la, 6), "lon": round(lo, 6),
        })

    print(f"\n  centrales asignadas a un ítem: {len(filas):,}")
    por = Counter((f["item"], f["elemento"]) for f in filas)
    mw = defaultdict(float)
    for f in filas:
        if isinstance(f["potencia_mw"], (int, float)):
            mw[f["item"]] += f["potencia_mw"]
    print(f"\n  {'ítem':<6}{'elemento':<44}{'centrales':>10}{'MW':>10}")
    print("  " + "-" * 72)
    for (n, el), c in sorted(por.items(), key=lambda t: -t[1]):
        print(f"  {n:<6}{el[:43]:<44}{c:>10,}{mw[n]:>10,.0f}")

    print(f"\n  ⚠️ NO asignadas:")
    print(f"       {sin_comb:>5}  térmicas sin combustible publicado "
          f"(ítems 89/90/91 quedan vacíos)")
    print(f"       {hibridas:>5}  híbridas: sus unidades caen en ítems distintos")
    if sin_coord:
        print(f"       {sin_coord:>5}  sin coordenada utilizable")
    if sin_tec:
        print(f"       {sin_tec:>5}  con unidades pero sin ficha de central")
    for t, n in tec_no_mapeada.most_common(5):
        print(f"       {n:>5}  tecnología sin ítem en la Matriz: {t}")

    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\n  escrito: {SALIDA.name}")
    return 0


if __name__ == "__main__":
    print("=" * 78)
    print("ENERGÍA · las centrales del Coordinador, a su ítem de la Matriz")
    print("=" * 78)
    sys.exit(main())
