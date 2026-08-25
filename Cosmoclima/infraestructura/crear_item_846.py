"""
ÍTEM 846 · SISTEMAS DE ALMACENAMIENTO TÉRMICO DE SALES FUNDIDAS
=================================================================

INSTRUCCIÓN (Alexis, 23-ago-2026): completar la fila huérfana como ítem nuevo y
distinto del 101, y «sólo tiene 1 elemento como submatriz para Chile:
Cerro Dominador».

★ POR QUÉ ES UN ÍTEM APARTE Y NO UN DUPLICADO DEL 101
-------------------------------------------------------
El ítem 101 es la PLANTA de concentración solar. El 846 es el SISTEMA DE
ALMACENAMIENTO que la planta lleva dentro. Misma ubicación física, componentes
distintos — y es exactamente el principio que el proyecto ya midió en el agua
potable rural:

    captación (dentro del río)  17,2 fallas por mil
    estanque  (en tierra)        0,0 fallas por mil

sobre los MISMOS sistemas expuestos a la MISMA lluvia. «Frágil» no es una
propiedad del sistema: es una propiedad de una pieza del sistema. El campo de
heliostatos y el bloque de sales no se rompen por lo mismo.

★ LA COORDENADA, Y UN ERROR QUE APARECIÓ AL BUSCARLA
------------------------------------------------------
Regla del proyecto: no inventar coordenadas. Se usa la del Coordinador Eléctrico
Nacional, declarada como `coordenada_publicada_por_el_operador`:

    -22,771546 / -69,478968   ← entregada por el director el 23-ago-2026

Se contrastó antes de usarla, y concuerda: está a **550 m** de la coordenada que
el Coordinador Eléctrico publica para CSP CERRO DOMINADOR (código CE01) y a
**100 m** de la de su subestación. Dos fuentes independientes, mismo sitio.

⚠️ La sub-matriz `centrales` que ya está en SharePoint tiene Cerro Dominador en
`-23° 56' 24" / -69° 03' 36"`, o sea **-23,94 / -69,06** — a **136 km** de la
posición real. La conversión de grados a decimales estaba bien; lo que estaba
mal era la coordenada de origen. Queda anotado para corregir esa sub-matriz.

★ LOS VALORES, Y DE DÓNDE SALE CADA UNO
-----------------------------------------
Sólo dos se apartan del ítem 101, y cada uno con su razón escrita. Los demás se
heredan porque no hay motivo medido para distinguirlos.

    FEN  Alta   ← SE APARTA (el 101 tiene Media). Razón: el bloque de sales
                  tiene un modo de falla sin retorno que la planta no tiene. Si
                  se pierde el calor, la sal solidifica bajo ~220 °C y el
                  circuito se destruye. Cualquier evento que interrumpa la
                  operación el tiempo suficiente es catastrófico, no reparable.
    FANC Media  ← heredado. Bajo el Protocolo I es un bien civil corriente
                  (art. 52): no es obra que contenga fuerzas peligrosas ni bien
                  indispensable para la supervivencia.
    IB   0,70   ← heredado del 101. Sin almacenamiento, una planta de
                  concentración solar es sólo una planta fotovoltaica cara: el
                  bloque de sales es lo que la vuelve despachable, así que
                  carga con el valor estratégico de la planta entera.
    VT   0,85   ← SE APARTA (el 101 tiene 0,80). Razón: el control térmico del
                  circuito de sales es continuo y crítico; es lo único que
                  impide la solidificación.

⚠️ **Son valores ASIGNADOS POR RAZONAMIENTO, no medidos.** Con un solo activo en
Chile no hay estadística de fallas posible. Quedan declarados como tales, y el
FEN debería revisarse cuando haya registro operacional.

FVT, PF, IRMD, Pev, Peh y Pen NO se asignan: los calcula `micr.py`.

USO
---
    ../.venv-esa/bin/python crear_item_846.py --matriz     # completa la fila
    ../.venv-esa/bin/python crear_item_846.py --submatriz  # crea y puebla la lista
    ../.venv-esa/bin/python crear_item_846.py --verificar
"""

import importlib.util
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import micr                                              # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sp", AQUI / "subir_submatrices_sharepoint.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

_a = importlib.util.spec_from_file_location(
    "act", AQUI / "actualizar_micr_sharepoint.py")
act = importlib.util.module_from_spec(_a)
_a.loader.exec_module(act)

NUMERO = 846
NOMBRE = "Sistemas de Almacenamiento Térmico de Sales Fundidas"
NOMBRE_VIEJO = "Plantas de Concentración Solar de Potencia"
SECTOR = "Energía"
FEN, FANC, IB, VT = "Alta", "Media", 0.70, 0.85

LISTA_SUB = "846 Almacenamiento Termico de Sales Fundidas"
ACTIVO = {
    "Nombre": "Sistema de Almacenamiento Térmico de Sales Fundidas · "
              "CSP Cerro Dominador",
    "Item": NUMERO,
    "Elemento": NOMBRE,
    "Region": "Antofagasta",
    "Provincia": "Tocopilla",
    "Comuna": "María Elena",
    "Lat": -22.771546,
    "Lon": -69.478968,
    "Operador": "Cerro Dominador CSP S.A.",
    "Estado": "EN OPERACION",
    "Fuente": "Coordenada entregada por el director (23-ago-2026). Concuerda "
              "con el Coordinador Eléctrico Nacional (CSP CERRO DOMINADOR, "
              "código CE01) dentro de 550 m, y con su subestación dentro de 100 m.",
    "Confianza": "coordenada_entregada_por_el_director_y_contrastada",
}
COLUMNAS_SUB = [
    ("Item", "Ítem", "number"), ("Elemento", "Elemento MICR", "text"),
    ("Region", "Región", "text"), ("Provincia", "Provincia", "text"),
    ("Comuna", "Comuna", "text"), ("Lat", "Latitud decimal", "number"),
    ("Lon", "Longitud decimal", "number"), ("Operador", "Operador", "text"),
    ("Estado", "Estado", "text"), ("Fuente", "Fuente", "text"),
    ("Confianza", "Confianza de ubicación", "text"),
]


def calculados():
    """Las seis columnas derivadas, calculadas — nunca asignadas."""
    fen, fanc = micr.n01(FEN), micr.n01(FANC)
    fvt = micr.fvt_n(fen, fanc, VT)
    pf = IB * fvt
    v = {"FVT": fvt, "PF": pf, "IRMD": micr.irmd_n(pf),
         "Pev": micr.pev_n(IB, fanc, fvt),
         "Peh": micr.peh_n(fanc, IB, VT, fvt),
         "Pen": micr.pen_n(fen, IB, fvt)}
    v["Pev_banda"] = micr.banda(v["Pev"], micr.CORTES_PEV_2026)
    v["Peh_banda"] = micr.banda(v["Peh"], micr.CORTES_PEH_2026)
    v["Pen_banda"] = micr.banda(v["Pen"], micr.CORTES_PEN_2026)
    return v


def hallar_huerfana():
    """La fila sin número. Devuelve (id, campos) o (None, None)."""
    url = (f"/sites/{sp.SITIO}/lists/{sp.LISTA_MIC}/items"
           "?$expand=fields($select=N_x00b0_,Title)&$top=500")
    while url:
        d = sp.llamar("GET", url).json()
        for x in d.get("value", []):
            f = x["fields"]
            if f.get("N_x00b0_") in (None, "") and NOMBRE_VIEJO in str(f.get("Title")):
                return x["id"], f
        url = d.get("@odata.nextLink")
    return None, None


def matriz():
    v = calculados()
    print("=" * 78)
    print(f"ÍTEM {NUMERO} · {NOMBRE}")
    print("=" * 78)
    print(f"\n  asignados por razonamiento (ver cabecera):")
    print(f"     FEN {FEN}   ← se aparta del 101 (Media): la sal solidifica y no vuelve")
    print(f"     FANC {FANC}  ← heredado · art. 52, bien civil corriente")
    print(f"     IB {IB}    ← heredado del 101")
    print(f"     VT {VT}   ← se aparta del 101 (0,80): control térmico continuo")
    print(f"\n  calculados por micr.py, no asignados:")
    for k in ("FVT", "PF"):
        print(f"     {k:<5} {v[k]:.4f}")
    print(f"     IRMD  {v['IRMD']}")
    for k in ("Pev", "Peh", "Pen"):
        print(f"     {k:<5} {v[k]:.4f}  → {v[k+'_banda']}")

    ident, campos = hallar_huerfana()
    if not ident:
        print("\n  ⚠️  no se encontró la fila huérfana. ¿Ya se completó?")
        return 1
    print(f"\n  fila huérfana encontrada: «{campos.get('Title')}»")

    r = sp.llamar("PATCH",
                  f"/sites/{sp.SITIO}/lists/{sp.LISTA_MIC}/items/{ident}/fields",
                  {"Title": NOMBRE, "N_x00b0_": NUMERO, "Sector": SECTOR,
                   "FEN": FEN, "FANC": FANC, "IB": IB, "VT": VT,
                   "FVT": round(v["FVT"], 4), "PF": round(v["PF"], 4),
                   "IRMD": v["IRMD"],
                   "Pev": v["Pev_banda"], "Peh": v["Peh_banda"],
                   "Pen": v["Pen_banda"],
                   "Pev_num": round(v["Pev"], 4),
                   "Peh_num": round(v["Peh"], 4),
                   "Pen_num": round(v["Pen"], 4)})
    print(f"  {'✓ completada' if r.status_code < 300 else f'✗ HTTP {r.status_code}'}")
    if r.status_code >= 300:
        print(f"    {r.text[:300]}")
        return 1
    return 0


def submatriz():
    print("=" * 78)
    print(f"SUB-MATRIZ «{LISTA_SUB}» · un solo elemento en Chile")
    print("=" * 78)
    existentes = sp.listas_existentes()
    lid = existentes.get(LISTA_SUB)
    if lid:
        print(f"\n  la lista ya existe")
    else:
        cols = []
        for nombre, visible, tipo in COLUMNAS_SUB:
            d = {"name": nombre, "displayName": visible}
            d["number"] = {"decimalPlaces": "eight"} if tipo == "number" else None
            if tipo == "text":
                d.pop("number")
                d["text"] = {}
            cols.append(d)
        cols.append({"name": "MICR", "displayName": "MICR",
                     "lookup": {"listId": sp.LISTA_MIC, "columnName": "Title"}})
        r = sp.llamar("POST", f"/sites/{sp.SITIO}/lists",
                      {"displayName": LISTA_SUB,
                       "list": {"template": "genericList"}, "columns": cols})
        if r.status_code >= 300:
            print(f"  ✗ HTTP {r.status_code}\n    {r.text[:300]}")
            return 1
        lid = r.json()["id"]
        print(f"\n  ✓ lista creada con {len(cols)} columnas")

    r = sp.llamar("GET", f"/sites/{sp.SITIO}/lists/{lid}/items"
                         "?$expand=fields($select=Title)&$top=50")
    if any(x["fields"].get("Title") == ACTIVO["Nombre"]
           for x in r.json().get("value", [])):
        print("  · el activo ya estaba cargado")
        return 0

    mapa = sp.mapa_micr()
    campos = {k: v for k, v in ACTIVO.items() if k != "Nombre"}
    campos["Title"] = ACTIVO["Nombre"]
    if NUMERO in mapa:
        campos["MICRLookupId"] = int(mapa[NUMERO])
    r = sp.llamar("POST", f"/sites/{sp.SITIO}/lists/{lid}/items",
                  {"fields": campos})
    print(f"  {'✓ activo cargado' if r.status_code < 300 else f'✗ HTTP {r.status_code}'}")
    if r.status_code >= 300:
        print(f"    {r.text[:400]}")
        return 1
    print(f"\n     {ACTIVO['Nombre']}")
    print(f"     {ACTIVO['Lat']:.6f} , {ACTIVO['Lon']:.6f}  ·  {ACTIVO['Comuna']}")
    print(f"     {ACTIVO['Confianza']}")
    return 0


def verificar():
    print("=" * 78)
    print("VERIFICACIÓN")
    print("=" * 78)
    url = (f"/sites/{sp.SITIO}/lists/{sp.LISTA_MIC}/items"
           "?$expand=fields&$top=500")
    fila = None
    while url and not fila:
        d = sp.llamar("GET", url).json()
        for x in d.get("value", []):
            n = x["fields"].get("N_x00b0_")
            if n is not None and int(float(n)) == NUMERO:
                fila = x["fields"]
        url = d.get("@odata.nextLink")
    if not fila:
        print(f"\n  ✗ el ítem {NUMERO} no está en la Matriz")
        return 1
    print(f"\n  ✓ ítem {NUMERO} en la Matriz")
    for k in ("Title", "Sector", "FEN", "FANC", "IB", "VT", "FVT", "PF",
              "IRMD", "Pev", "Peh", "Pen", "Pev_num", "Peh_num", "Pen_num"):
        print(f"     {k:<10} {fila.get(k)}")
    sub = fila.get("SubMatriz")
    if sub:
        sub = sub if isinstance(sub, list) else [sub]
        print(f"     {'SubMatriz':<10} " + " · ".join(s.get("LookupValue") for s in sub))
    else:
        print(f"     {'SubMatriz':<10} (sin vincular — corre el vínculo)")

    lid = sp.listas_existentes().get(LISTA_SUB)
    if lid:
        r = sp.llamar("GET", f"/sites/{sp.SITIO}/lists/{lid}/items"
                             "?$expand=fields&$top=50").json()
        print(f"\n  ✓ sub-matriz con {len(r.get('value', []))} elemento(s)")
        for x in r.get("value", []):
            f = x["fields"]
            print(f"     {f.get('Title')}")
            print(f"       {f.get('Lat')} , {f.get('Lon')} · {f.get('Comuna')}")
    return 0


if __name__ == "__main__":
    acciones = {"--matriz": matriz, "--submatriz": submatriz,
                "--verificar": verificar}
    elegida = next((a for a in sys.argv[1:] if a in acciones), None)
    if not elegida:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(acciones[elegida]())
