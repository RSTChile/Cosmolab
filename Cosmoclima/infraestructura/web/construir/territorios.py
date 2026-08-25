"""
TERRITORIOS · la tabla que permite agregar de comuna a país
=============================================================

Extrae de `datos/capas/comunas.geojson` la jerarquía administrativa completa y la
deja como tabla plana. Es la pieza de la que dependen todas las agregaciones de
la aplicación: sin ella no se puede sumar comunas para obtener una provincia.

★ POR QUÉ HAY QUE EXTRAERLA
-----------------------------
La jerarquía existe, pero **enterrada dentro de 98,6 MB de geometría**. Cada
comuna trae `CUT_COM`, `CUT_PROV` y `CUT_REG` en sus propiedades, y los códigos
CUT están anidados por construcción: `CUT_COM` empieza por `CUT_PROV`, que empieza
por `CUT_REG`. O sea que la jerarquía se puede reconstruir de los códigos mismos,
sin tabla externa.

Lo que NO existe es una capa de provincias ni de regiones. Se disuelven desde las
comunas, y este archivo deja listo el índice para hacerlo.

★ LAS 345 vs 346 · RESUELTO EL 23-ago-2026
--------------------------------------------
La capa trae 345 comunas y Chile tiene 346. La que falta es **Antártica
(CUT 12202)**, provincia Antártica Chilena, región de Magallanes.

**No es un defecto de la capa.** Las capas cartográficas de Chile continental
omiten sistemáticamente el territorio antártico, y para esta Matriz no cambia
nada: no hay infraestructura crítica catastrada allí. Queda comprobado por
código —el reparto por región se contrasta con el oficial en cada corrida— para
que la duda no se reabra ni haya que volver a investigarla.

USO
---
    ../../.venv-esa/bin/python construir/territorios.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent                       # …/infraestructura
CAPA = RAIZ / "datos" / "capas" / "comunas.geojson"
SALIDA = AQUI.parent / "publico" / "datos" / "territorios.json"


def cut(v):
    """CUT como texto, con los ceros a la izquierda que el JSON pierde.

    ★ Un CUT es un identificador, no un número: la comuna 01101 (Iquique) no es
    la 1101. Si se lee como entero se pierde el cero y deja de cruzar con
    cualquier otra fuente. Se normaliza siempre a texto.
    """
    if v in (None, ""):
        return None
    s = str(v).strip()
    return s.zfill(5) if len(s) > 3 else s.zfill(len(s))


def leer_propiedades():
    """Sólo las propiedades. La geometría se descarta apenas se lee."""
    print(f"  leyendo {CAPA.name} ({CAPA.stat().st_size/1e6:.0f} MB)…", flush=True)
    with CAPA.open(encoding="utf-8") as fh:
        d = json.load(fh)
    out = []
    for f in d.get("features", []):
        p = f.get("properties", {})
        out.append({
            "cut": cut(p.get("CUT_COM")),
            "comuna": p.get("COMUNA"),
            "cut_prov": cut(p.get("CUT_PROV")),
            "provincia": p.get("PROVINCIA"),
            "cut_reg": cut(p.get("CUT_REG")),
            "region": p.get("REGION"),
            "superficie": p.get("SUPERFICIE"),
        })
    return out


def construir():
    filas = leer_propiedades()
    print(f"  comunas en la capa : {len(filas)}")

    # ── coherencia de la jerarquía, comprobada y no supuesta ────────────────
    # El CUT de comuna DEBE empezar por el de su provincia, y el de provincia
    # por el de su región. Si no, la capa está mal y agregar daría números
    # falsos sin avisar.
    incoherentes = [f for f in filas
                    if not (f["cut"] or "").startswith(f["cut_prov"] or "\0")
                    or not (f["cut_prov"] or "").startswith(f["cut_reg"] or "\0")]
    if incoherentes:
        print(f"  ⚠️  {len(incoherentes)} comunas con jerarquía incoherente:")
        for f in incoherentes[:5]:
            print(f"       {f['comuna']}: {f['cut']} / {f['cut_prov']} / {f['cut_reg']}")
    else:
        print("  ✓ la jerarquía de códigos es coherente en las 345")

    dup = [c for c, n in
           __import__("collections").Counter(f["cut"] for f in filas).items() if n > 1]
    if dup:
        print(f"  ⚠️  CUT repetidos: {dup}")

    # ── provincias y regiones, disueltas desde las comunas ──────────────────
    prov, reg = {}, {}
    for f in filas:
        prov.setdefault(f["cut_prov"], {"cut": f["cut_prov"],
                                        "nombre": f["provincia"],
                                        "cut_reg": f["cut_reg"],
                                        "comunas": []})["comunas"].append(f["cut"])
        reg.setdefault(f["cut_reg"], {"cut": f["cut_reg"],
                                      "nombre": f["region"],
                                      "provincias": set()})
        reg[f["cut_reg"]]["provincias"].add(f["cut_prov"])
    for r in reg.values():
        r["provincias"] = sorted(r["provincias"])

    print(f"  provincias : {len(prov)}")
    print(f"  regiones   : {len(reg)}")

    # ── ¿cuál de las 346 falta? ─────────────────────────────────────────────
    # Se comprueba por región contra el reparto oficial, en vez de dejar la
    # duda anotada. Si una región trae una comuna de menos, ahí está.
    OFICIAL = {"15": 4, "01": 7, "02": 9, "03": 9, "04": 15, "05": 38, "13": 52,
               "06": 33, "07": 30, "16": 21, "08": 33, "09": 32, "14": 12,
               "10": 30, "11": 10, "12": 11}
    real = defaultdict(int)
    for f in filas:
        real[f["cut_reg"]] += 1
    faltan = [(k, OFICIAL[k], real.get(k, 0)) for k in OFICIAL
              if real.get(k, 0) != OFICIAL[k]]
    if faltan:
        print("\n  ★ diferencia contra el reparto oficial de comunas:")
        for k, esperado, hay in faltan:
            nombre = next((f["region"] for f in filas if f["cut_reg"] == k), k)
            print(f"       región {k} «{nombre}»: oficial {esperado}, en la capa {hay}")
    else:
        print("  ✓ el reparto por región coincide con el oficial")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps({
        "comunas": sorted(filas, key=lambda f: f["cut"] or ""),
        "provincias": sorted(prov.values(), key=lambda p: p["cut"] or ""),
        "regiones": sorted(reg.values(), key=lambda r: r["cut"] or ""),
    }, ensure_ascii=False), encoding="utf-8")
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e3:.0f} KB)")
    return {"comunas": len(filas), "provincias": len(prov), "regiones": len(reg)}


if __name__ == "__main__":
    print("=" * 70)
    print("TERRITORIOS · comuna → provincia → región")
    print("=" * 70)
    construir()
    sys.exit(0)
