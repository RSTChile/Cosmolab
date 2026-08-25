"""
ACTIVOS POR COMUNA × ÍTEM · la tabla que alimenta el mapa de calor
===================================================================

Convierte las 31 sub-matrices (96.423 activos) en un índice disperso
`{comuna: {ítem: cuántos}}`, que es lo único que la aplicación necesita para
pintar territorios y para saber cuántos activos quedan expuestos.

★ POR QUÉ ESTO Y NO LOS ACTIVOS UNO POR UNO
---------------------------------------------
96.423 activos son ~40 MB en el navegador. El índice agregado son ~400 KB:
345 comunas × los 42 ítems que tienen sub-matriz, y disperso, porque la mayoría
de las casillas está vacía. **Cien veces menos, y responde exactamente las mismas
preguntas** que el mapa de calor necesita hacer.

Los activos individuales se cargan aparte, y sólo los de la ventana visible.

★ CÓMO SE ASIGNA LA COMUNA · dos caminos, medidos
---------------------------------------------------
**1 · Por nombre** (81 % de las filas). Se normaliza quitando tildes y
mayúsculas, y cruza contra las 345 comunas de la capa oficial. Medido: **cruzan
78.166 de 78.172**, y las 6 que fallan son tres variantes de nombre conocidas que
van en la tabla de alias.

**2 · Por coordenada** (el 19 % restante). ⚠️ **18.251 filas no traen comuna en
la sub-matriz.** Para ésas se resuelve por punto-en-polígono contra la geometría
COMPLETA —no la simplificada— porque la simplificada mueve los bordes y un activo
junto al límite quedaría en la comuna equivocada. La geometría simplificada es
para dibujar; la completa, para decidir.

★ Y LO QUE NO SE PUEDE ASIGNAR SE CUENTA, NO SE ESCONDE
---------------------------------------------------------
Un activo sin comuna y sin coordenada no desaparece del recuento: va a
`sin_territorio`, y la aplicación lo muestra como cifra aparte. Un mapa que
omite en silencio lo que no supo ubicar miente por omisión.

USO
---
    ../../.venv-esa/bin/python construir/activos.py
"""

import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
SUB = RAIZ / "submatrices_excel"
CAPA = RAIZ / "datos" / "capas" / "comunas.geojson"
DATOS = AQUI.parent / "publico" / "datos"
SALIDA = DATOS / "activos_por_comuna.json"

# Variantes de nombre que la capa oficial escribe de otra forma. Medido: son
# exactamente estas tres, y afectan a 6 filas de 96.423.
ALIAS = {"COYHAIQUE": "COIHAIQUE", "LLAYLLAY": "LLAY LLAY", "LA CALERA": "CALERA"}


def norm(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.upper().split())


# ── punto en polígono, sin dependencias ──────────────────────────────────────

def _en_anillo(lon, lat, anillo):
    dentro = False
    n = len(anillo)
    j = n - 1
    for i in range(n):
        xi, yi = anillo[i][0], anillo[i][1]
        xj, yj = anillo[j][0], anillo[j][1]
        if (yi > lat) != (yj > lat):
            if lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi:
                dentro = not dentro
        j = i
    return dentro


def _en_poligono(lon, lat, poli):
    """Primer anillo es el contorno; los siguientes son huecos."""
    if not poli or not _en_anillo(lon, lat, poli[0]):
        return False
    return not any(_en_anillo(lon, lat, h) for h in poli[1:])


def _en_geometria(lon, lat, g):
    if g["type"] == "Polygon":
        return _en_poligono(lon, lat, g["coordinates"])
    if g["type"] == "MultiPolygon":
        return any(_en_poligono(lon, lat, p) for p in g["coordinates"])
    return False


def _caja(g):
    xs, ys = [], []
    def rec(c):
        if isinstance(c[0], (int, float)):
            xs.append(c[0]); ys.append(c[1])
        else:
            for x in c:
                rec(x)
    rec(g["coordinates"])
    return min(xs), min(ys), max(xs), max(ys)


def cargar_geometria():
    """La capa COMPLETA, con índice de cajas para no probar 345 polígonos por punto."""
    print(f"  cargando geometría completa ({CAPA.stat().st_size/1e6:.0f} MB)…",
          flush=True)
    d = json.loads(CAPA.read_text(encoding="utf-8"))
    out = []
    for f in d["features"]:
        p = f["properties"]
        c = str(p.get("CUT_COM") or "").zfill(5)
        out.append((c, _caja(f["geometry"]), f["geometry"]))
    return out


def resolver(lon, lat, indice):
    for c, (x0, y0, x1, y1), g in indice:
        if x0 <= lon <= x1 and y0 <= lat <= y1 and _en_geometria(lon, lat, g):
            return c
    return None


def construir():
    terr = json.loads((DATOS / "territorios.json").read_text(encoding="utf-8"))
    por_nombre = {norm(c["comuna"]): c["cut"] for c in terr["comunas"]}

    filas = []
    for p in sorted(SUB.glob("*.xlsx")):
        ws = openpyxl.load_workbook(p, read_only=True).active
        it = ws.iter_rows(values_only=True)
        cab = list(next(it))
        i = {k: cab.index(k) for k in
             ("Ítem", "Comuna", "Latitud decimal", "Longitud decimal") if k in cab}
        for r in it:
            filas.append({
                "sub": p.stem,
                "item": (int(float(r[i["Ítem"]]))
                         if "Ítem" in i and r[i["Ítem"]] not in (None, "") else None),
                "comuna": r[i["Comuna"]] if "Comuna" in i else None,
                "lat": r[i["Latitud decimal"]] if "Latitud decimal" in i else None,
                "lon": r[i["Longitud decimal"]] if "Longitud decimal" in i else None,
            })
    print(f"  activos leídos : {len(filas):,}")

    # ── 1 · por nombre ──────────────────────────────────────────────────────
    pendientes = []
    via = Counter()
    for f in filas:
        n = norm(f["comuna"])
        n = ALIAS.get(n, n)
        cut = por_nombre.get(n)
        if cut:
            f["cut"] = cut
            via["nombre"] += 1
        else:
            pendientes.append(f)
    print(f"  por nombre     : {via['nombre']:,}")
    print(f"  pendientes     : {len(pendientes):,}")

    # ── 2 · por coordenada, sólo para los pendientes ────────────────────────
    con_coord = [f for f in pendientes if f["lat"] not in (None, "")]
    print(f"     de ésos, con coordenada: {len(con_coord):,}")
    if con_coord:
        indice = cargar_geometria()
        # Se resuelve por coordenada ÚNICA: muchos activos comparten posición
        # aproximada y repetir el cálculo sería tiempo tirado.
        cache = {}
        for k, f in enumerate(con_coord, 1):
            lat, lon = round(float(f["lat"]), 4), round(float(f["lon"]), 4)
            if (lat, lon) not in cache:
                cache[(lat, lon)] = resolver(lon, lat, indice)
            f["cut"] = cache[(lat, lon)]
            via["coordenada" if f["cut"] else "fuera de toda comuna"] += 1
            if k % 4000 == 0:
                print(f"     {k:,}/{len(con_coord):,}", flush=True)
        print(f"  por coordenada : {via['coordenada']:,}")
        if via["fuera de toda comuna"]:
            print(f"  ⚠️  fuera de toda comuna: {via['fuera de toda comuna']:,} "
                  f"(mar adentro, o coordenada errónea)")

    sin = [f for f in filas if not f.get("cut")]
    print(f"  sin territorio : {len(sin):,}")

    # ── el índice disperso ──────────────────────────────────────────────────
    indice = defaultdict(lambda: defaultdict(int))
    sin_coord = defaultdict(int)
    for f in filas:
        if f.get("cut") and f["item"]:
            indice[f["cut"]][str(f["item"])] += 1
        if f["lat"] in (None, "") and f["item"]:
            sin_coord[str(f["item"])] += 1

    total_indexado = sum(sum(v.values()) for v in indice.values())
    DATOS.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps({
        "por_comuna": {k: dict(v) for k, v in indice.items()},
        "sin_territorio": len(sin),
        "sin_coordenada_por_item": dict(sin_coord),
        "total": len(filas),
        "total_indexado": total_indexado,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"\n  comunas con activos : {len(indice)}")
    print(f"  ítems distintos     : {len({i for v in indice.values() for i in v})}")
    print(f"  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e3:.0f} KB)")
    return {"activos": len(filas), "indexados": total_indexado,
            "sin_territorio": len(sin), "comunas_con_activos": len(indice)}


if __name__ == "__main__":
    print("=" * 70)
    print("ACTIVOS POR COMUNA × ÍTEM")
    print("=" * 70)
    construir()
    sys.exit(0)
