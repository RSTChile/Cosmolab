"""
CORREGIR LA FILA DE CERRO DOMINADOR EN LA SUB-MATRIZ `centrales`
==================================================================

INSTRUCCIÓN (Alexis, 23-ago-2026): «Corrige las coordenadas con las que te pasé.
Las tomé yo mismo con Google Earth.»

★ TRES ERRORES EN LA MISMA FILA
---------------------------------

**1 · La coordenada, equivocada por 136 km.**

    hoy    -23° 56' 24,0" S / -69° 03' 36,0" W   =  -23,940 / -69,060
    real   -22° 46' 17,6" S / -69° 28' 44,3" W   =  -22,771546 / -69,478968

La posición real la tomó el director con Google Earth. Se contrastó antes de
usarla contra dos fuentes independientes del Coordinador Eléctrico Nacional:
está a **550 m** de la coordenada publicada para la central CSP CERRO DOMINADOR
(código CE01) y a **100 m** de la de su subestación. Tres fuentes, un sitio.

★ Y lo que enseña este error: **la conversión de grados a decimales estaba
bien** —se había validado carácter por carácter en 38 de 39 filas— pero la
coordenada de ORIGEN estaba mal. Validar la conversión no valida el dato.

**2 · El vínculo apuntaba al ítem equivocado.**

La columna de búsqueda `MICR` apuntaba al identificador interno 836, que es la
fila que hoy es el **ítem 846 · Sistemas de Almacenamiento Térmico de Sales
Fundidas**. Pero esta fila es la PLANTA, no su bloque de sales: le corresponde
el **ítem 101 · Plantas de Concentración Solar de Potencia (CSP)**.

★ Esto además explica **de dónde salió la fila huérfana** que estuvo dos días sin
número ni valores: al catastrar Cerro Dominador aquí, la columna de búsqueda
obligaba a elegir un ítem de la Matriz; el nombre no calzaba con el 101 por el
sufijo «(CSP)»; y se creó una fila nueva para poder guardar. El huérfano nació de
este registro, no de un descuido.

**3 · «María Elena» está en el campo Provincia, y es una comuna.**

La provincia es **Tocopilla**. Confirmado contra el Coordinador Eléctrico
(`comuna: María Elena · provincia: Tocopilla · región: Antofagasta`).

USO
---
    ../.venv-esa/bin/python corregir_cerro_dominador.py            # muestra
    ../.venv-esa/bin/python corregir_cerro_dominador.py --escribir
    ../.venv-esa/bin/python corregir_cerro_dominador.py --auditar  # las 147 filas
"""

import csv
import importlib.util
import math
import re
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"

_spec = importlib.util.spec_from_file_location(
    "sp", AQUI / "subir_submatrices_sharepoint.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

LISTA = "Centrales Eléctricas"
LAT, LON = -22.771546, -69.478968
ITEM_CORRECTO = 101


def a_dms(v, pos, neg):
    """Decimal a grados-minutos-segundos, en el formato exacto de la lista."""
    h = pos if v >= 0 else neg
    a = abs(v)
    g = int(a)
    m = int((a - g) * 60)
    s = round(((a - g) * 60 - m) * 60, 1)
    if s >= 60:
        s -= 60
        m += 1
    if m >= 60:
        m -= 60
        g += 1
    return f"{'-' if v < 0 else ''}{g}° {m:02d}' {s:.1f}\" {h}"


def de_dms(txt):
    """DMS a decimal. Devuelve None si no se puede leer."""
    if not txt:
        return None
    m = re.search(r"(-?)\s*(\d+)°\s*(\d+)'\s*([\d.]+)\"?\s*([NSEW]?)", str(txt))
    if not m:
        return None
    signo, g, mi, s, hemi = m.groups()
    v = int(g) + int(mi) / 60 + float(s) / 3600
    return -v if signo == "-" or hemi in ("S", "W") else v


def km(a, b, c, d):
    return 111.32 * math.hypot(a - c, (b - d) * math.cos(math.radians(a)))


def fila_dominador(lid):
    r = sp.llamar("GET", f"/sites/{sp.SITIO}/lists/{lid}/items"
                         "?$expand=fields&$top=200").json()
    for x in r.get("value", []):
        if "dominador" in str(x["fields"].get("Title", "")).lower():
            return x["id"], x["fields"]
    return None, None


def principal(escribir):
    lid = sp.listas_existentes()[LISTA]
    ident, f = fila_dominador(lid)
    if not ident:
        print("no se encontró la fila de Cerro Dominador")
        return 1

    lat_dms, lon_dms = a_dms(LAT, "N", "S"), a_dms(LON, "E", "W")
    viejo = (de_dms(f.get("Latitud")), de_dms(f.get("Longitud")))
    print("=" * 78)
    print("CORRECCIÓN · Cerro Dominador en la sub-matriz «Centrales Eléctricas»")
    print("=" * 78)
    print(f"\n  1 · COORDENADA")
    print(f"      antes   {f.get('Latitud')} / {f.get('Longitud')}")
    print(f"      ahora   {lat_dms} / {lon_dms}")
    if None not in viejo:
        print(f"      el error era de {km(viejo[0], viejo[1], LAT, LON):.0f} km")

    mapa = sp.mapa_micr()
    print(f"\n  2 · VÍNCULO CON LA MATRIZ")
    print(f"      antes   identificador interno {f.get('MICRLookupId')} "
          f"(hoy es el ítem 846, el bloque de sales)")
    print(f"      ahora   identificador interno {mapa[ITEM_CORRECTO]} "
          f"(ítem {ITEM_CORRECTO}, la planta)")

    print(f"\n  3 · PROVINCIA")
    print(f"      antes   «{f.get('Provincia')}»  ← es una comuna, no una provincia")
    print(f"      ahora   «Tocopilla»")

    if not escribir:
        print("\n  (nada escrito · corre con --escribir)")
        return 0

    r = sp.llamar("PATCH", f"/sites/{sp.SITIO}/lists/{lid}/items/{ident}/fields",
                  {"Latitud": lat_dms, "Longitud": lon_dms,
                   "Provincia": "Tocopilla",
                   "MICRLookupId": int(mapa[ITEM_CORRECTO])})
    print(f"\n  {'✓ corregida' if r.status_code < 300 else f'✗ HTTP {r.status_code}'}")
    if r.status_code >= 300:
        print(f"    {r.text[:300]}")
        return 1

    _, g = fila_dominador(lid)
    print(f"\n  verificado en SharePoint:")
    for k in ("Latitud", "Longitud", "Provincia", "MICRLookupId"):
        print(f"     {k:<14} {g.get(k)}")
    return 0


def auditar():
    """¿Hay más filas mal ubicadas? Se contrastan las 147 contra el Coordinador."""
    lid = sp.listas_existentes()[LISTA]
    filas, url = [], (f"/sites/{sp.SITIO}/lists/{lid}/items"
                      "?$expand=fields&$top=200")
    while url:
        d = sp.llamar("GET", url).json()
        filas += [x["fields"] for x in d.get("value", [])]
        url = d.get("@odata.nextLink")

    ref = {}
    p = DATOS / "inventario_centrales_electricas.csv"
    if p.exists():
        for x in csv.DictReader(p.open(encoding="utf-8")):
            if x.get("lat"):
                ref[x["nombre"].upper()] = (float(x["lat"]), float(x["lon"]))

    print("=" * 78)
    print("AUDITORÍA · las 147 centrales contra el Coordinador Eléctrico")
    print("=" * 78)
    print(f"\n  filas en la sub-matriz : {len(filas)}")
    print(f"  centrales de referencia con coordenada : {len(ref)}\n")

    ilegibles, sin_ref, lejos, ok = [], [], [], 0
    for f in filas:
        t = str(f.get("Title", "")).upper()
        la, lo = de_dms(f.get("Latitud")), de_dms(f.get("Longitud"))
        if la is None or lo is None:
            ilegibles.append(f.get("Title"))
            continue
        cand = [v for k, v in ref.items() if t and (t in k or k in t)]
        if not cand:
            sin_ref.append(f.get("Title"))
            continue
        d = min(km(la, lo, a, b) for a, b in cand)
        if d > 5:
            lejos.append((f.get("Title"), d))
        else:
            ok += 1

    print(f"  ✓ dentro de 5 km del Coordinador : {ok}")
    print(f"  ✗ a más de 5 km                  : {len(lejos)}")
    print(f"  · sin referencia para comparar    : {len(sin_ref)}")
    print(f"  · coordenada ilegible             : {len(ilegibles)}")
    if lejos:
        print("\n  las que no cuadran, de mayor a menor distancia:")
        for t, d in sorted(lejos, key=lambda x: -x[1])[:25]:
            print(f"      {d:8.1f} km   {t}")
    print("\n  ★ 'sin referencia' NO significa que estén bien: significa que el")
    print("    Coordinador no publica esa central con ese nombre. Quedan sin")
    print("    comprobar, que no es lo mismo que comprobadas.")
    return 0


if __name__ == "__main__":
    if "--auditar" in sys.argv:
        raise SystemExit(auditar())
    raise SystemExit(principal("--escribir" in sys.argv))
