"""
BAJAR LA MATRIZ (lista `mic`) DESDE SHAREPOINT
================================================

INSTRUCCIÓN (Alexis, 21-ago-2026): «lo que manda al final es cómo queda todo en
SharePoint. Los excel se hacen después bajados desde allí.»

★ POR QUÉ ESTE SCRIPT TENÍA QUE EXISTIR
-----------------------------------------
El 22-ago se descubrió que el CSV local `micr_sharepoint_845.csv` estaba
**caducado**: traía `FANC = Alta` en 810 de 845 filas, cuando la lista publicada
ya tenía 15 tras la re-medición con el Protocolo I de Ginebra y la Ley 21.663.

La causa no fue un descuido de nadie: **la re-medición del FANC y el recalibrado
de la Pev se hicieron interactivamente contra SharePoint y no dejaron script**.
El trabajo existía sólo dentro de la lista. Si esa lista se dañaba, se perdía.

Este script cierra ese agujero. Bajar la Matriz deja de ser una operación
manual irrepetible y pasa a ser un comando.

LO QUE HACE
-----------
Pagina la lista `mic` completa por Microsoft Graph y escribe un CSV con las
mismas trece columnas que ya consumen el resto de los scripts, de modo que sea
un reemplazo directo:

    n, elemento, Sector, FEN, FANC, IB, VT, FVT, PF, IRMD, Pev, Peh, Pen

★ NO SOBRESCRIBE EL ANTERIOR. Cada bajada lleva su fecha en el nombre y además
se deja/actualiza el enlace `micr_sharepoint_ultimo.csv`. Los exports viejos son
el testigo con el que se mide qué cambió, y borrarlos sería borrar la evidencia.

USO
---
    ../.venv-esa/bin/python bajar_micr_sharepoint.py
    ../.venv-esa/bin/python bajar_micr_sharepoint.py --comparar <csv_viejo>
"""

import csv
import importlib.util
import sys
import time
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"

# Se reutiliza la maquinaria ya probada de subida: `token()` renueva en silencio
# con el token de refresco, y `llamar()` respeta el `Retry-After` del servidor.
_spec = importlib.util.spec_from_file_location(
    "sp", AQUI / "subir_submatrices_sharepoint.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)

# nombre de columna en el CSV  →  nombre interno del campo en SharePoint
COLUMNAS = [
    ("n",        "N_x00b0_"),
    ("elemento", "Title"),
    ("Sector",   "Sector"),
    ("FEN",      "FEN"),
    ("FANC",     "FANC"),
    ("IB",       "IB"),
    ("VT",       "VT"),
    ("FVT",      "FVT"),
    ("PF",       "PF"),
    ("IRMD",     "IRMD"),
    ("Pev",      "Pev"),
    ("Peh",      "Peh"),
    ("Pen",      "Pen"),
]
SELECT = ",".join(sorted({c for _, c in COLUMNAS}))


def num(v):
    """★ El número de ítem, como entero.

    SharePoint devuelve el campo numérico como `1.0`; el resto del proyecto lo
    escribe como `1`. Sin normalizar, dos exports de la MISMA lista no cruzan
    entre sí y toda comparación sale «846 filas nuevas, 0 cambios» — que es
    justo el tipo de fallo silencioso que este proyecto persigue.
    """
    if v in (None, ""):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return v


def bajar():
    """Todas las filas de `mic`, en el orden en que las devuelve el servicio."""
    filas, url = [], (f"/sites/{sp.SITIO}/lists/{sp.LISTA_MIC}/items"
                      f"?$expand=fields($select={SELECT})&$top=500")
    paginas = 0
    while url:
        r = sp.llamar("GET", url)
        if r.status_code >= 300:
            print(f"  ✗ Graph devolvió HTTP {r.status_code}")
            print(f"    {r.text[:300]}")
            raise SystemExit(1)
        d = r.json()
        for it in d.get("value", []):
            f = it.get("fields", {})
            fila = {nom: f.get(campo) for nom, campo in COLUMNAS}
            fila["n"] = num(fila["n"])
            filas.append(fila)
        paginas += 1
        url = d.get("@odata.nextLink")
        print(f"    página {paginas}: {len(filas):,} filas acumuladas", flush=True)
    return filas


def resumir(filas, etiqueta):
    print(f"\n  {etiqueta}: {len(filas):,} filas")
    sin_n = [f for f in filas if f["n"] in (None, "")]
    if sin_n:
        print(f"  ⚠️  {len(sin_n)} fila(s) SIN NÚMERO:")
        for f in sin_n:
            vacios = sum(1 for k, v in f.items() if v in (None, ""))
            print(f"        «{f['elemento']}» — {vacios} de 13 campos vacíos")
    for col in ("FEN", "FANC", "IRMD", "Pev", "Peh", "Pen"):
        c = Counter(str(f[col]) for f in filas)
        print(f"     {col:5s} " + " · ".join(f"{k}:{v}" for k, v in c.most_common()))


def comparar(nuevas, ruta_vieja):
    """Qué cambió respecto de un export anterior, fila por fila y por número."""
    viejas = {num(x["n"]): x for x in
              csv.DictReader(Path(ruta_vieja).open(encoding="utf-8"))}
    print("\n" + "=" * 78)
    print(f"CAMBIOS CONTRA {Path(ruta_vieja).name}")
    print("=" * 78)
    campos = [c for c, _ in COLUMNAS if c not in ("n", "elemento")]
    movidas, nuevas_filas = Counter(), []
    for f in nuevas:
        k = f["n"]
        if k not in viejas:
            nuevas_filas.append(f)
            continue
        for c in campos:
            a, b = str(viejas[k].get(c, "")), str(f.get(c, ""))
            try:
                if abs(float(a) - float(b)) < 1e-9:
                    continue
            except (TypeError, ValueError):
                if a == b:
                    continue
            movidas[c] += 1
    print(f"\n  filas nuevas respecto del export viejo : {len(nuevas_filas)}")
    for f in nuevas_filas[:12]:
        print(f"        {f['n']}  {f['elemento']}")
    print("\n  columnas que cambiaron de valor:")
    for c in campos:
        marca = "  ★" if movidas[c] > len(nuevas) * 0.5 else ""
        print(f"     {c:5s} {movidas[c]:5d} de {len(nuevas):,} filas{marca}")


def main():
    print("=" * 78)
    print("BAJANDO LA MATRIZ DESDE SHAREPOINT · lista `mic`")
    print("=" * 78)
    print("\n  sitio : rstchilecom.sharepoint.com/RMD")
    print("  ★ la sesión se renueva sola con el token de refresco;")
    print("    sólo hace falta --login si el canje falla.\n")

    filas = bajar()
    filas.sort(key=lambda f: (f["n"] is None, f["n"] if f["n"] is not None else 0))
    resumir(filas, "BAJADO")

    DATOS.mkdir(exist_ok=True)
    salida = DATOS / f"micr_sharepoint_{time.strftime('%Y%m%d')}.csv"
    with salida.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[c for c, _ in COLUMNAS])
        w.writeheader()
        w.writerows(filas)
    ultimo = DATOS / "micr_sharepoint_ultimo.csv"
    if ultimo.exists() or ultimo.is_symlink():
        ultimo.unlink()
    ultimo.symlink_to(salida.name)
    print(f"\n  escrito : {salida.name}")
    print(f"  enlace  : {ultimo.name} → {salida.name}")

    viejo = next((a for a in sys.argv[1:] if a.endswith(".csv")), None)
    if "--comparar" in sys.argv and viejo:
        comparar(filas, viejo)
    elif (DATOS / "micr_sharepoint_845.csv").exists():
        comparar(filas, DATOS / "micr_sharepoint_845.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
