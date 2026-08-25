"""
ACTUALIZAR LA MATRIZ EN SHAREPOINT Y VINCULARLA CON SUS SUB-MATRICES
======================================================================

INSTRUCCIÓN (Alexis, 23-ago-2026): «Actualiza todo en SharePoint, y conecta la
MICR con las Sub Matrices con un campo de búsqueda en el mismo sitio que vincule
el ítem de la Matriz con la Sub Matriz respectiva. Las que no hemos desarrollado
aún quedan en blanco por ahora.»

★ POR QUÉ EL VÍNCULO NO PUEDE SER UNA COLUMNA DE BÚSQUEDA DIRECTA
-------------------------------------------------------------------
Una columna de búsqueda de SharePoint apunta a **una** lista. Pero hay 31
sub-matrices en 31 listas distintas, así que no hay una sola lista a la que
apuntar. Y además la relación es **de muchos a muchos**, medido:

    el ítem 572 (Edificios Públicos) tiene TRES sub-matrices
        572 Edificios Publicos · 572 Sedes de Gobierno Provincial ·
        572 Sedes de Gobierno Regional
    el archivo «443 Educacion Superior» cubre DOS ítems: el 443 y el 446

La solución es un **catálogo**: una lista `submatrices` con una fila por
sub-matriz, y en la Matriz una columna de búsqueda **de valores múltiples** que
apunta a ese catálogo. Así el 572 puede tener tres y el 443 puede aparecer en
dos ítems, sin inventar nada.

★ LO QUE SE ESCRIBE EN LA MATRIZ
----------------------------------
Las columnas recalculadas el 23-ago (ver `recalibrar_micr.py`):

    FANC   re-medido en 4 grados contra el Protocolo I de Ginebra
    FVT    ahora CALCULADO, ya no asignado a criterio
    PF     = IB · FVT
    IRMD   ahora derivado de los umbrales de PF
    Pev · Peh · Pen   bandas nuevas, con los cortes congelados

★ Y TRES COLUMNAS NUEVAS: `Pev_num`, `Peh_num`, `Pen_num`.
Hasta hoy SharePoint guardaba sólo la ETIQUETA de banda, no el número. Por eso
el recalibrado del 22-ago se perdió: la etiqueta no permite reconstruir el
valor. Guardar el número cierra ese agujero.

★ CÓMO SE EVITA PERDER FILAS EN SILENCIO
------------------------------------------
El 21-ago se perdieron 208 filas porque `$batch` devuelve HTTP 200 aunque las
peticiones de adentro devuelvan 429. Aquí se revisa **cada sub-respuesta**, se
reintenta lo que falló, y al final `--verificar` vuelve a bajar la lista y la
compara fila por fila contra lo que se quiso escribir.

USO
---
    ../.venv-esa/bin/python actualizar_micr_sharepoint.py --actualizar
    ../.venv-esa/bin/python actualizar_micr_sharepoint.py --catalogo
    ../.venv-esa/bin/python actualizar_micr_sharepoint.py --vincular
    ../.venv-esa/bin/python actualizar_micr_sharepoint.py --verificar
"""

import csv
import importlib.util
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import openpyxl

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"
SUB = AQUI / "submatrices_excel"
RECALIBRADA = DATOS / "micr_recalibrada.csv"

_spec = importlib.util.spec_from_file_location(
    "sp", AQUI / "subir_submatrices_sharepoint.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

CATALOGO = "Sub-Matrices"          # nombre visible de la lista catálogo
LOTE = 20                          # tope del punto de entrada por lotes de Graph

# Sub-matrices que viven SÓLO en SharePoint, sin archivo Excel de respaldo.
# Las dos primeras son anteriores al catastro por archivos; la 846 nació
# directamente en la plataforma porque tiene un único elemento en Chile.
EXTRA = {"Centrales Eléctricas": (147, [101, 102, 103, 104, 105, 106, 107,
                                        108, 109, 110, 111, 112, 113, 114]),
         "Subestaciones": (39, [120]),
         "846 Almacenamiento Termico de Sales Fundidas": (1, [846])}


# ── utilidades ───────────────────────────────────────────────────────────────

def lote(seq, n=LOTE):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def enviar_lote(peticiones, etiqueta=""):
    """Un `$batch`, revisando el estado de CADA sub-petición. Devuelve los ids
    que fallaron después de agotar los reintentos."""
    pendientes = {str(p["id"]): p for p in peticiones}
    for vuelta in range(8):
        if not pendientes:
            return []
        cuerpo = {"requests": list(pendientes.values())}
        r = sp.llamar("POST", "/$batch", cuerpo)
        if r.status_code >= 300:
            time.sleep(5 * (vuelta + 1))
            continue
        espera, fallidas = 0, {}
        for resp in r.json().get("responses", []):
            if resp.get("status", 500) >= 300:
                fallidas[resp["id"]] = pendientes[resp["id"]]
                ra = (resp.get("headers") or {}).get("Retry-After")
                espera = max(espera, int(ra) if ra else 0)
        pendientes = fallidas
        if pendientes:
            time.sleep(max(espera, 2 * (vuelta + 1)))
    if pendientes:
        print(f"    ⚠️  {etiqueta}: {len(pendientes)} sin escribir tras 8 vueltas")
    return list(pendientes)


def submatrices_del_proyecto():
    """{nombre de sub-matriz: (filas, [ítems que cubre])} desde los xlsx."""
    out = {}
    for p in sorted(SUB.glob("*.xlsx")):
        m = re.match(r"^(\d+)\s+(.+)\.xlsx$", p.name)
        if not m:
            continue
        ws = openpyxl.load_workbook(p, read_only=True).active
        it = ws.iter_rows(values_only=True)
        cab = next(it)
        ci = cab.index("Ítem") if "Ítem" in cab else None
        filas, items = 0, set()
        for r in it:
            filas += 1
            if ci is not None and r[ci] not in (None, ""):
                items.add(int(float(r[ci])))
        out[p.stem] = (filas, sorted(items) or [int(m.group(1))])
    out.update(EXTRA)
    return out


# ── 1 · actualizar los valores de la Matriz ──────────────────────────────────

CAMPOS = ["FANC", "FVT", "PF", "IRMD", "Pev", "Peh", "Pen",
          "Pev_num", "Peh_num", "Pen_num"]
NUEVAS = [("Pev_num", "Pev (valor)"), ("Peh_num", "Peh (valor)"),
          ("Pen_num", "Pen (valor)")]


def asegurar_columnas(lid):
    """Crea `Pev_num`/`Peh_num`/`Pen_num` si no existen. Idempotente."""
    r = sp.llamar("GET", f"/sites/{sp.SITIO}/lists/{lid}/columns?$select=id,name")
    hay = {c["name"] for c in r.json().get("value", [])}
    for interno, visible in NUEVAS:
        if interno in hay:
            print(f"    · {interno} ya existe")
            continue
        r = sp.llamar("POST", f"/sites/{sp.SITIO}/lists/{lid}/columns",
                      {"name": interno, "displayName": visible,
                       "number": {"decimalPlaces": "four"}})
        print(f"    {'✓' if r.status_code < 300 else '✗'} creada {interno}"
              + ("" if r.status_code < 300 else f"  HTTP {r.status_code}"))


def actualizar():
    if not RECALIBRADA.exists():
        print("falta micr_recalibrada.csv — corre antes recalibrar_micr.py --escribir")
        return 1
    filas = list(csv.DictReader(RECALIBRADA.open(encoding="utf-8")))
    print("=" * 78)
    print("ACTUALIZANDO LA MATRIZ EN SHAREPOINT")
    print("=" * 78)
    print(f"\n  origen : {RECALIBRADA.name} · {len(filas)} filas")
    print("\n  columnas nuevas para el valor numérico:")
    asegurar_columnas(sp.LISTA_MIC)

    print("\n  resolviendo identificadores internos…")
    mapa = sp.mapa_micr()
    print(f"    {len(mapa)} ítems mapeados")
    sin_id = [f["n"] for f in filas if int(f["n"]) not in mapa]
    if sin_id:
        print(f"    ⚠️  {len(sin_id)} sin identificador: {sin_id[:8]}")

    peticiones, i = [], 0
    for f in filas:
        n = int(f["n"])
        if n not in mapa:
            continue
        i += 1
        peticiones.append({
            "id": str(i), "method": "PATCH",
            "url": f"/sites/{sp.SITIO}/lists/{sp.LISTA_MIC}/items/{mapa[n]}/fields",
            "headers": {"Content-Type": "application/json", "If-Match": "*"},
            "body": {"FANC": f["FANC"],
                     "FVT": round(float(f["FVT"]), 4),
                     "PF": round(float(f["PF"]), 4),
                     "IRMD": f["IRMD"],
                     "Pev": f["Pev_banda"], "Peh": f["Peh_banda"],
                     "Pen": f["Pen_banda"],
                     "Pev_num": round(float(f["Pev"]), 4),
                     "Peh_num": round(float(f["Peh"]), 4),
                     "Pen_num": round(float(f["Pen"]), 4)}})

    print(f"\n  escribiendo {len(peticiones)} filas en lotes de {LOTE}…")
    t0, fallidas = time.time(), []
    for k, grupo in enumerate(lote(peticiones), 1):
        fallidas += enviar_lote(grupo, f"lote {k}")
        if k % 5 == 0 or k * LOTE >= len(peticiones):
            hechas = min(k * LOTE, len(peticiones))
            print(f"    {hechas:4d}/{len(peticiones)}  "
                  f"{hechas/max(time.time()-t0,1):.1f} filas/s", flush=True)
    print(f"\n  {'✓ todas escritas' if not fallidas else f'⚠️ {len(fallidas)} fallaron'}")
    print("\n  ★ corre --verificar para comprobar contra el origen.")
    return 0


# ── 2 · el catálogo de sub-matrices ──────────────────────────────────────────

def catalogo():
    subs = submatrices_del_proyecto()
    print("=" * 78)
    print(f"CATÁLOGO DE SUB-MATRICES · lista «{CATALOGO}»")
    print("=" * 78)
    print(f"\n  sub-matrices detectadas: {len(subs)}")
    todos = sorted({i for _, its in subs.values() for i in its})
    print(f"  ítems de la Matriz cubiertos: {len(todos)}")

    existentes = sp.listas_existentes()
    lid = existentes.get(CATALOGO)
    if lid:
        print(f"\n  la lista ya existe ({lid[:8]}…), se completará lo que falte")
    else:
        r = sp.llamar("POST", f"/sites/{sp.SITIO}/lists", {
            "displayName": CATALOGO,
            "list": {"template": "genericList"},
            "columns": [
                {"name": "Items", "displayName": "Ítems de la Matriz",
                 "text": {}},
                {"name": "Filas", "displayName": "Elementos",
                 "number": {"decimalPlaces": "none"}},
                {"name": "ListaSP", "displayName": "Lista en SharePoint",
                 "text": {}}]})
        if r.status_code >= 300:
            print(f"  ✗ no se pudo crear: HTTP {r.status_code}\n    {r.text[:200]}")
            return 1
        lid = r.json()["id"]
        print(f"\n  ✓ lista creada")

    r = sp.llamar("GET", f"/sites/{sp.SITIO}/lists/{lid}/items"
                         "?$expand=fields($select=Title)&$top=500")
    ya = {x["fields"].get("Title") for x in r.json().get("value", [])}
    faltan = [(k, v) for k, v in sorted(subs.items()) if k not in ya]
    print(f"  ya en la lista: {len(ya)} · por agregar: {len(faltan)}")

    peticiones = []
    for i, (nombre, (filas, items)) in enumerate(faltan, 1):
        peticiones.append({
            "id": str(i), "method": "POST",
            "url": f"/sites/{sp.SITIO}/lists/{lid}/items",
            "headers": {"Content-Type": "application/json"},
            "body": {"fields": {"Title": nombre,
                                "Items": ", ".join(str(x) for x in items),
                                "Filas": filas,
                                "ListaSP": nombre}}})
    for k, grupo in enumerate(lote(peticiones), 1):
        enviar_lote(grupo, f"catálogo lote {k}")
    print(f"\n  ✓ catálogo con {len(ya) + len(faltan)} sub-matrices")
    print(f"  total de elementos catastrados: "
          f"{sum(v[0] for v in subs.values()):,}")
    return 0


# ── 3 · la columna de búsqueda en la Matriz ──────────────────────────────────

def vincular():
    existentes = sp.listas_existentes()
    lid_cat = existentes.get(CATALOGO)
    if not lid_cat:
        print(f"falta la lista «{CATALOGO}» — corre antes --catalogo")
        return 1

    print("=" * 78)
    print("VINCULANDO LA MATRIZ CON SUS SUB-MATRICES")
    print("=" * 78)

    r = sp.llamar("GET", f"/sites/{sp.SITIO}/lists/{sp.LISTA_MIC}/columns"
                         "?$select=id,name")
    if "SubMatriz" in {c["name"] for c in r.json().get("value", [])}:
        print("\n  · la columna «SubMatriz» ya existe")
    else:
        r = sp.llamar("POST", f"/sites/{sp.SITIO}/lists/{sp.LISTA_MIC}/columns",
                      {"name": "SubMatriz", "displayName": "Sub-Matriz",
                       "lookup": {"listId": lid_cat, "columnName": "Title",
                                  "allowMultipleValues": True}})
        if r.status_code >= 300:
            print(f"  ✗ HTTP {r.status_code}\n    {r.text[:300]}")
            return 1
        print("\n  ✓ columna de búsqueda «Sub-Matriz» creada "
              "(admite varios valores)")

    # catálogo: nombre → id interno
    cat, url = {}, (f"/sites/{sp.SITIO}/lists/{lid_cat}/items"
                    "?$expand=fields($select=Title)&$top=500")
    while url:
        d = sp.llamar("GET", url).json()
        for x in d.get("value", []):
            cat[x["fields"].get("Title")] = x["id"]
        url = d.get("@odata.nextLink")

    subs = submatrices_del_proyecto()
    por_item = defaultdict(list)
    for nombre, (_, items) in subs.items():
        for it in items:
            if nombre in cat:
                por_item[it].append(cat[nombre])

    mapa = sp.mapa_micr()
    print(f"\n  ítems con sub-matriz : {len(por_item)}")
    print(f"  ítems que quedan en blanco : {len(mapa) - len(por_item)}")

    peticiones = []
    for i, (item, ids) in enumerate(sorted(por_item.items()), 1):
        if item not in mapa:
            print(f"    ⚠️  el ítem {item} no está en la Matriz")
            continue
        peticiones.append({
            "id": str(i), "method": "PATCH",
            "url": f"/sites/{sp.SITIO}/lists/{sp.LISTA_MIC}/items/{mapa[item]}/fields",
            "headers": {"Content-Type": "application/json", "If-Match": "*"},
            "body": {"SubMatrizLookupId@odata.type": "Collection(Edm.Int32)",
                     "SubMatrizLookupId": [int(x) for x in ids]}})
    fallidas = []
    for k, grupo in enumerate(lote(peticiones), 1):
        fallidas += enviar_lote(grupo, f"vínculo lote {k}")
    print(f"\n  {'✓ vinculados' if not fallidas else f'⚠️ {len(fallidas)} fallaron'}"
          f" {len(peticiones)-len(fallidas)} ítems")
    for it in sorted(por_item)[:6]:
        print(f"      ítem {it:>4} → {len(por_item[it])} sub-matriz(ces)")
    return 0


# ── 4 · verificación contra el origen ────────────────────────────────────────

def verificar():
    filas = {int(f["n"]): f for f in
             csv.DictReader(RECALIBRADA.open(encoding="utf-8"))}
    print("=" * 78)
    print("VERIFICACIÓN · SharePoint contra el origen, fila por fila")
    print("=" * 78)
    sel = "N_x00b0_,FANC,FVT,PF,IRMD,Pev,Peh,Pen,Pev_num,Peh_num,Pen_num"
    leidas, url = {}, (f"/sites/{sp.SITIO}/lists/{sp.LISTA_MIC}/items"
                       f"?$expand=fields($select={sel})&$top=500")
    while url:
        d = sp.llamar("GET", url).json()
        for x in d.get("value", []):
            n = x["fields"].get("N_x00b0_")
            if n is not None:
                leidas[int(float(n))] = x["fields"]
        url = d.get("@odata.nextLink")
    print(f"\n  filas leídas de SharePoint: {len(leidas)}")

    mal = defaultdict(list)
    for n, f in filas.items():
        g = leidas.get(n)
        if g is None:
            mal["ausente"].append(n)
            continue
        for campo, esperado in (("FANC", f["FANC"]), ("IRMD", f["IRMD"]),
                                ("Pev", f["Pev_banda"]), ("Peh", f["Peh_banda"]),
                                ("Pen", f["Pen_banda"])):
            if str(g.get(campo)) != str(esperado):
                mal[campo].append(n)
        for campo, esperado in (("FVT", f["FVT"]), ("PF", f["PF"]),
                                ("Pev_num", f["Pev"]), ("Peh_num", f["Peh"]),
                                ("Pen_num", f["Pen"])):
            try:
                if abs(float(g.get(campo)) - float(esperado)) > 0.0005:
                    mal[campo].append(n)
            except (TypeError, ValueError):
                mal[campo].append(n)

    if not mal:
        print(f"\n  ✓ las {len(filas)} filas coinciden en las diez columnas.")
    else:
        for campo, ns in sorted(mal.items()):
            print(f"  ✗ {campo:9s} {len(ns):4d} discrepancias · ej. {ns[:6]}")
    return 0 if not mal else 1


if __name__ == "__main__":
    acciones = {"--actualizar": actualizar, "--catalogo": catalogo,
                "--vincular": vincular, "--verificar": verificar}
    elegida = next((a for a in sys.argv[1:] if a in acciones), None)
    if not elegida:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(acciones[elegida]())
