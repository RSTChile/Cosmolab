"""
FUSIONAR · que todo lo incorporado llegue al índice que la aplicación lee
==========================================================================

Durante la sesión del 25-ago se incorporaron tres conjuntos de activos que
estaban bajados o disponibles y no llegaban a la aplicación:

    39.218  estaciones base 4G y 5G          (Subtel, ítems 185 y 186)
     2.162  establecimientos del RETC        (ítems 660, 661 y 670)
         2  centrales nucleares              (catastro propio, ítem 133)

`activos.py` construye el índice desde las sub-matrices y no los ve. Este paso
los añade **después**, sin tocar el original: si mañana la sub-matriz los
incorpora, se quita este paso y nada más cambia.

★ POR QUÉ SE MARCA EL ORIGEN DE CADA UNO
------------------------------------------
No todos entran con el mismo respaldo:

    consolidado    viene de un registro público georreferenciado
                   (Subtel, RETC) — es lo mismo que hace el resto del proyecto
    catastro propio  lo agregamos nosotros con coordenadas públicas
                   verificables (las dos centrales nucleares)

Mezclarlos borraría la diferencia entre auditar la Matriz y escribirla. Cada
activo lleva su origen y la aplicación puede mostrarlo.

USO
---
    ../../.venv-esa/bin/python construir/fusionar_activos.py
"""

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
RAIZ = AQUI.parent.parent
DATOS = AQUI.parent / "publico" / "datos"
DETALLE = DATOS / "activos"

import activos as A  # noqa: E402

FUENTES = [
    (RAIZ / "datos" / "telecom_por_item.csv", "consolidado", "Subtel",
     lambda r: (r["item"], r.get("empresa", ""), float(r["lat"]), float(r["lon"]),
                f"{r.get('generacion','')} {r.get('empresa','')}".strip())),
    (RAIZ / "datos" / "quimico_por_item.csv", "consolidado", "RETC · MMA",
     lambda r: (r["item"], r.get("nombre", ""), float(r["lat"]), float(r["lon"]),
                r.get("nombre") or r.get("razon_social", ""))),
    (RAIZ / "datos" / "municipios_por_item.csv", "consolidado", "RETC · MMA",
     lambda r: (r["item"], r.get("nombre", ""), float(r["lat"]), float(r["lon"]),
                r.get("nombre") or r.get("razon_social", ""))),
    # ── los tres sectores que Alexis pidió completar el 25-ago ──────────────
    (RAIZ / "datos" / "energia_por_item.csv", "consolidado",
     "Coordinador Eléctrico Nacional",
     lambda r: (r["item"], r.get("nombre", ""), float(r["lat"]), float(r["lon"]),
                r.get("nombre", ""))),
    (RAIZ / "datos" / "hidrico_por_item.csv", "consolidado", "DGA · SNIA",
     lambda r: (r["item"], r.get("nombre", ""), float(r["lat"]), float(r["lon"]),
                r.get("nombre", ""))),
    (RAIZ / "datos" / "alimentario_por_item.csv", "consolidado",
     "SUBPESCA · Comisión Nacional de Riego",
     lambda r: (r["item"], r.get("nombre", ""), float(r["lat"]), float(r["lon"]),
                r.get("nombre", ""))),
    # ── el barrido de los 7 agentes (25-ago) ────────────────────────────────
    # Trae su propia columna `fuente` por fila, así que el rótulo de aquí es
    # genérico y el detalle real viaja con cada activo.
    (RAIZ / "datos" / "barrido_por_item.csv", "consolidado", "barrido 25-ago",
     lambda r: (r["item"], r.get("nombre", ""), float(r["lat"]), float(r["lon"]),
                r.get("nombre", ""))),
]


# ★★ Dos catastros del mismo edificio no son dos edificios.
#    La sede de la Municipalidad de Santiago aparece en el RETC en
#    (-33,43783 · -70,65044) y ya estaba en el índice en (-33,43693 · -70,65027):
#    a unos 100 m, el mismo edificio medido por dos fuentes. Sumarlas duplicaría
#    activos y el conteo del ítem dejaría de ser cierto.
#    150 m es el radio: por debajo, dos registros del MISMO ítem son el mismo
#    objeto; por encima, en una manzana urbana ya pueden ser dos cosas distintas.
DEDUP_M = 150.0


def a_menos_de(la, lo, otros, metros=DEDUP_M):
    for oy, ox in otros:
        dy = (oy - la) * 111320.0
        dx = (ox - lo) * 111320.0 * math.cos(math.radians(la))
        if math.hypot(dx, dy) <= metros:
            return True
    return False


def limpiar_pasada_anterior(idx):
    """★ QUE CORRERLO DOS VECES NO DUPLIQUE NADA.

    La primera versión sumaba al índice y hacía `extend` en el detalle: volver a
    ejecutarla contaba cada activo otra vez, y el total crecía sin que nada
    avisara. Se deshace lo de la corrida anterior antes de rehacerlo.

    Distinguirlos es posible porque los activos agregados aquí llevan `o`
    —su origen— y los que vienen de las sub-matrices no.
    """
    previo = idx.pop("agregados_por_fusion", None)
    if not previo:
        return
    n = sum(previo.values())
    print(f"  deshaciendo la corrida anterior: {n:,} activos en {len(previo)} ítems")
    for cut, d in list(idx["por_comuna"].items()):
        for it in list(d):
            if it in previo:
                del d[it]
        if not d:
            del idx["por_comuna"][cut]
    idx["total_indexado"] = idx.get("total_indexado", 0) - n
    idx["total"] = idx.get("total", 0) - n
    for f in DETALLE.glob("*.json"):
        act = json.loads(f.read_text(encoding="utf-8"))
        quedan = [a for a in act if not a.get("o")]
        if len(quedan) != len(act):
            f.write_text(json.dumps(quedan, ensure_ascii=False,
                                    separators=(",", ":")), encoding="utf-8")


def main():
    idx = json.loads((DATOS / "activos_por_comuna.json").read_text(encoding="utf-8"))
    print(f"  índice actual: {idx['total_indexado']:,} activos en "
          f"{len(idx['por_comuna'])} comunas")
    limpiar_pasada_anterior(idx)

    print("  cargando geometría…", flush=True)
    geo = A.cargar_geometria()
    cache = {}

    # ★★ EL MAR NO ES DE NADIE, Y LOS CENTROS DE CULTIVO ESTÁN EN EL MAR.
    #    Las 1.612 concesiones de acuicultura de SUBPESCA caen en el agua, fuera
    #    de todo polígono comunal: resolviendo sólo por coordenada se ubicaban
    #    88 y se perdían 1.524 sin que nada avisara. La fuente declara a qué
    #    comuna pertenece cada una —el borde costero sí tiene comuna
    #    administrativa— y ese nombre es el respaldo cuando el punto no cae en
    #    tierra.
    terr = json.loads((DATOS / "territorios.json").read_text(encoding="utf-8"))
    por_nombre = {A.norm(c["comuna"]): c["cut"] for c in terr["comunas"]}

    def comuna_de(la, lo):
        k = (round(la, 2), round(lo, 2))
        if k not in cache:
            cache[k] = A.resolver(lo, la, geo)
        return cache[k]

    nuevos = defaultdict(list)
    resumen = defaultdict(int)

    # lo que ya estaba, por comuna e ítem, para no volver a contarlo
    por_nombre_n = Counter()
    sin_ubicar = Counter()
    previos = defaultdict(list)
    for f in DETALLE.glob("*.json"):
        for x in json.loads(f.read_text(encoding="utf-8")):
            previos[(f.stem, x["n"])].append((x["y"], x["x"]))
    dup = defaultdict(int)

    for ruta, origen, fuente, leer in FUENTES:
        if not ruta.exists():
            print(f"  ⚠️ falta {ruta.name}")
            continue
        n = 0
        with ruta.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    item, _, la, lo, nombre = leer(r)
                except (TypeError, ValueError, KeyError):
                    continue
                cut = comuna_de(la, lo)
                if not cut:
                    cut = por_nombre.get(A.norm(r.get("comuna") or ""))
                    if cut:
                        por_nombre_n[ruta.name] += 1
                if not cut:
                    sin_ubicar[ruta.name] += 1
                    continue
                # ⚠️ Sólo contra lo que YA estaba, nunca entre los de la misma
                #    fuente: dos operadoras comparten torre a diez metros y son
                #    dos estaciones base distintas, no una repetida. Aplicarlo
                #    dentro de la fuente descartaba 21.703 de las 32.285 de 4G.
                if a_menos_de(la, lo, previos.get((cut, item), ())):
                    dup[item] += 1
                    continue
                nuevos[cut].append({
                    "n": item, "a": (nombre or "")[:70],
                    "y": round(la, 5), "x": round(lo, 5),
                    "o": origen, "f": fuente,
                })
                resumen[item] += 1
                n += 1
        extra = ""
        if por_nombre_n[ruta.name]:
            extra = f" · {por_nombre_n[ruta.name]:,} por comuna declarada"
        if sin_ubicar[ruta.name]:
            extra += f" · ⚠️ {sin_ubicar[ruta.name]:,} sin ubicar"
        print(f"  {ruta.name:<28}{n:>8,} ubicados · {origen}{extra}")

    # el catastro propio, que va marcado aparte
    prop = DATOS / "catastro_propio.json"
    if prop.exists():
        for a in json.loads(prop.read_text(encoding="utf-8"))["activos"]:
            cut = comuna_de(a["lat"], a["lon"])
            if not cut:
                continue
            nuevos[cut].append({
                "n": a["item"], "a": a["nombre"][:70],
                "y": a["lat"], "x": a["lon"],
                "o": "catastro propio", "f": a.get("operador", ""),
            })
            resumen[a["item"]] += 1
        print(f"  catastro_propio.json{'':<8}{len(json.loads(prop.read_text(encoding='utf-8'))['activos']):>8,} ubicados · catastro propio")

    # ── al índice agregado ──────────────────────────────────────────────────
    for cut, lista in nuevos.items():
        d = idx["por_comuna"].setdefault(cut, {})
        for a in lista:
            d[a["n"]] = d.get(a["n"], 0) + 1
    total_nuevo = sum(len(v) for v in nuevos.values())
    idx["total_indexado"] = idx.get("total_indexado", 0) + total_nuevo
    idx["total"] = idx.get("total", 0) + total_nuevo
    idx.setdefault("agregados_por_fusion", {})
    for it, n in resumen.items():
        idx["agregados_por_fusion"][it] = n
    (DATOS / "activos_por_comuna.json").write_text(
        json.dumps(idx, ensure_ascii=False), encoding="utf-8")

    # ── y al detalle por comuna, que es lo que dibuja los puntos ────────────
    for cut, lista in nuevos.items():
        f = DETALLE / f"{cut}.json"
        actual = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        actual.extend(lista)
        actual.sort(key=lambda d: (d["n"], d.get("a") or ""))
        f.write_text(json.dumps(actual, ensure_ascii=False, separators=(",", ":")),
                     encoding="utf-8")

    matriz = json.loads((DATOS / "matriz.json").read_text(encoding="utf-8"))
    nom = {str(i["n"]): i["elemento"] for i in matriz["items"]}
    print(f"\n  {'ítem':<6}{'elemento':<44}{'nuevos':>9}")
    print("  " + "-" * 62)
    for it, n in sorted(resumen.items(), key=lambda t: -t[1]):
        print(f"  {it:<6}{nom.get(it, '?')[:43]:<44}{n:>9,}")
    if dup:
        print(f"\n  ⚠️ no agregados por ser el mismo objeto ya catastrado "
              f"(a menos de {DEDUP_M:.0f} m del mismo ítem):")
        for it, n in sorted(dup.items(), key=lambda t: -t[1]):
            print(f"       ítem {it:<6}{nom.get(it, '?')[:40]:<42}{n:>7,}")

    print(f"\n  total agregado: {total_nuevo:,}")
    print(f"  índice ahora  : {idx['total_indexado']:,} activos")

    con = set()
    for d in idx["por_comuna"].values():
        con.update(d)
    print(f"  ítems con activos: {len(con)} de {len(matriz['items'])} "
          f"({100*len(con)/len(matriz['items']):.1f} %)")
    return 0


if __name__ == "__main__":
    print("=" * 74)
    print("FUSIONAR · lo incorporado llega al índice de la aplicación")
    print("=" * 74)
    sys.exit(main())
