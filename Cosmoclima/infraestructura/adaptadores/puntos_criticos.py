"""
PUNTOS CRÍTICOS DE INUNDACIÓN · el catastro a resolución de calle
===================================================================

INSTRUCCIÓN (Alexis, 23-ago-2026): «al alcalde de Santiago no le sirve un mapa de
calor rojo de toda la comuna; necesita qué calles entre qué manzanas se inundan
si caen tantos mm en tanto tiempo… alguien debe registrar eso en alguna parte.»

Lo registra SENAPRED, y estaba a la vista: **el visor que pide login tiene la
interfaz de programación completamente abierta.**

★ LO QUE TRAE, Y POR QUÉ CAMBIA EL PROYECTO
---------------------------------------------
15.799 puntos con coordenada y **nombre de calle o cruce**. Ejemplos reales de la
comuna de Santiago, donde el registro del Ministerio de Obras Públicas daba cero:

    Paso bajo nivel Matta      anegamiento · riesgo Alto · -33,46040 / -70,65690
    Marcoleta / Santa Rosa     anegamiento
    San Alfonso / Pilcomayo    colapso de colectores

Es la primera fuente del proyecto que responde **dónde exactamente**, y no
sólo en qué comuna.

★ Y ESTÁ COMPROBADO QUE PREDICE
---------------------------------
Cruzado contra los 1.289 daños viales reales del temporal de julio de 2026, con
un control justo —cada evento desplazado 3 a 8 km al azar—:

    a 500 m   6,2 × sobre el control
    a  1 km   3,3 × sobre el control
    distancia mediana  1,38 km real  contra  3,29 km del control

No es una lista administrativa: tiene señal.

★★ POR QUÉ HAY QUE BAJARLO TODOS LOS DÍAS
-------------------------------------------
**Estas capas se sobrescriben, no se versionan.** La capa hermana
`compilado_invierno` ya está en cero registros, y `compilado_invierno_2` tenía
15.552 donde la actual tiene 15.799. Es el mismo riesgo ya documentado con
SERNAGEOMIN: si no se baja y se fecha, la historia desaparece sin aviso.

Por eso cada corrida escribe en `datos/crudo/puntos_criticos/AAAA-MM-DD/` y
NUNCA pisa lo anterior.

★★★ DOS ADVERTENCIAS QUE HAY QUE ARRASTRAR AL USAR ESTE DATO
--------------------------------------------------------------
**1 · El nivel de riesgo es PERCEPCIÓN, no medición.** Lo dice la nota del propio
tablero de SENAPRED: «determinado de acuerdo con la percepción del riesgo
comunal». Es el mismo defecto que ya encontramos en el FEN — mide el criterio de
quien llenó el formulario, no una magnitud física. Se conserva, pero declarado.

**2 · La etiqueta «se activó el período anterior» tiene sesgo severo de reporte.**
Talcahuano marca 11 de 12 activados y Rancagua 58 de 96, mientras **La Serena,
Copiapó y Antofagasta marcan 0 de 256**. Eso no es que no se inundaran: es que
llenaron el formulario distinto. Usarla cruda diría que en Copiapó no pasa nada.

USO
---
    ../.venv-esa/bin/python adaptadores/puntos_criticos.py           # todo
    ../.venv-esa/bin/python adaptadores/puntos_criticos.py --estado
"""

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
CRUDO = RAIZ / "datos" / "crudo" / "puntos_criticos" / date.today().isoformat()
SALIDA = RAIZ / "datos" / "puntos_criticos.csv"

AGOL = "https://services5.arcgis.com/i7S5PSnIJAUcWvSE/arcgis/rest/services"
VISOR = "https://visor-grd.senapred.gob.cl/arcgis/rest/services"

# ★ Las cuatro temporadas dan la serie con que se puede fijar una referencia
# histórica en vez de recalibrar contra el máximo del año — que es exactamente
# la no estacionariedad que la Matriz venía arrastrando.
CAPAS = [
    ("pc_2026", f"{AGOL}/compilado_INV_2026/FeatureServer/0", 2026, True),
    ("pc_2022", f"{AGOL}/Puntos_Cr%C3%ADticos_Programa_Invierno_2022/FeatureServer/0", 2022, True),
    ("pc_2021", f"{AGOL}/Puntos_Cr%C3%ADticos_Programa_Invierno_2021/FeatureServer/0", 2021, True),
    ("pc_2025", f"{VISOR}/PCRIT_ANTE/PERIODOS_ANTER/MapServer/0", 2025, False),
    ("vias_julio2026",
     f"{VISOR}/EMERGENCIAS/R%C3%ADo_Atmosf%C3%A9rico_Julio_2026/MapServer/4", 2026, False),
    ("cigiden_crm",
     "https://services7.arcgis.com/UeyripQFTg6pfUe5/arcgis/rest/services"
     "/Catastro_CRM_actualizado_18_9/FeatureServer/0", None, True),
]
LOTE = 2000


def pedir(url, extra=""):
    for intento in range(4):
        try:
            with urllib.request.urlopen(url + extra, timeout=120) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(4 * (intento + 1))
    return None


def bajar(nombre, base, geojson):
    """Todas las entidades de una capa, paginando cuando el servidor lo admite."""
    fmt = "geojson" if geojson else "json"
    q = f"/query?where=1%3D1&outFields=*&outSR=4326&f={fmt}"
    todo, off, ultima = [], 0, None
    while True:
        d = pedir(base + q, f"&resultOffset={off}&resultRecordCount={LOTE}")
        if not d:
            break
        ent = d.get("features", [])
        if not ent:
            break
        # ★ ArcGIS 10.2 IGNORA `resultOffset` en silencio y devuelve siempre la
        # misma página. Si eso pasa, girar para siempre o —peor— duplicar filas.
        # Se detecta comparando la primera entidad de la página con la anterior.
        huella = json.dumps(ent[0], sort_keys=True)[:400]
        if huella == ultima:
            break
        ultima = huella
        todo += ent
        # ⚠️ NO usar `exceededTransferLimit`: en salida GeoJSON no viene, y
        # mirarlo cortaba el barrido en la primera página de 2.000. Se pagina
        # mientras la página venga llena, que es la señal fiable.
        if len(ent) < LOTE:
            break
        off += LOTE
        if off > 60000:
            break
    return todo


def propiedades(e):
    return e.get("properties") or e.get("attributes") or {}


def coordenada(e):
    g = e.get("geometry") or {}
    if "coordinates" in g:
        c = g["coordinates"]
        return (c[0], c[1]) if isinstance(c[0], (int, float)) else (None, None)
    if "x" in g:
        return g.get("x"), g.get("y")
    p = propiedades(e)
    return p.get("Longitud"), p.get("Latitud")


def main():
    if "--estado" in sys.argv:
        base = RAIZ / "datos" / "crudo" / "puntos_criticos"
        for d in sorted(base.glob("*")) if base.exists() else []:
            n = sum(1 for _ in d.glob("*.json"))
            print(f"  {d.name}  {n} capas  "
                  f"{sum(f.stat().st_size for f in d.glob('*'))/1e6:.1f} MB")
        return 0

    CRUDO.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("PUNTOS CRÍTICOS DE INUNDACIÓN · espejado con fecha")
    print("=" * 74)
    print(f"\n  destino: {CRUDO.relative_to(RAIZ)}\n")

    filas, resumen = [], {}
    for nombre, url, temporada, gj in CAPAS:
        t0 = time.time()
        ent = bajar(nombre, url, gj)
        resumen[nombre] = len(ent)
        print(f"  {nombre:<16} {len(ent):>6,} entidades  ({time.time()-t0:.1f}s)")
        if not ent:
            print(f"     ⚠️  vacía o inaccesible — se anota y se sigue")
            continue
        (CRUDO / f"{nombre}.json").write_text(
            json.dumps({"features": ent}, ensure_ascii=False), encoding="utf-8")
        if nombre.startswith("pc_"):
            for e in ent:
                p = propiedades(e)
                lon, lat = coordenada(e)
                filas.append({
                    "temporada": temporada,
                    "region": p.get("Región") or p.get("REGION") or p.get("Region"),
                    "comuna": p.get("Comuna") or p.get("COMUNA"),
                    "sector": p.get("Sector") or p.get("SECTOR"),
                    "causa": p.get("Causa_del_") or p.get("CAUSA"),
                    # ⚠️ percepción del riesgo comunal, no medición
                    "nivel_riesgo_declarado": p.get("Nivel_de_R") or p.get("NIVEL"),
                    "responsable": p.get("Si_la_resp"),
                    # ⚠️ con sesgo severo de reporte entre comunas
                    "activado_periodo_anterior": p.get("Punto_Crí") or p.get("afec_per_a"),
                    "lon": lon, "lat": lat,
                })

    if filas:
        SALIDA.parent.mkdir(exist_ok=True)
        with SALIDA.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
        con = sum(1 for f in filas if f["lat"] not in (None, ""))
        print(f"\n  normalizado: {SALIDA.name} · {len(filas):,} filas "
              f"({con:,} con coordenada)")
        print("\n  por temporada: " + " · ".join(
            f"{k}:{v:,}" for k, v in sorted(Counter(f["temporada"] for f in filas).items())))
        print("\n  causas principales:")
        for k, v in Counter(f["causa"] for f in filas if f["causa"]).most_common(6):
            print(f"     {v:>6,}  {str(k)[:58]}")

    print("\n  ★ Las capas se SOBRESCRIBEN en el origen. Este espejado con fecha")
    print("    es lo único que conserva la historia. Correr a diario.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
