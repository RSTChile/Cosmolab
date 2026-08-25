"""
LA RUTA 5 · la longitudinal que el catastro no tiene
======================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «la ruta 5 (panamericana), principal ruta
longitudinal del país, debería poder consultarse específicamente, y mostrarse
completa en el mapa como ruta».

★★ POR QUÉ HAY QUE TRAERLA DE FUERA
-------------------------------------
Medido sobre los 14.036 activos de Red Vial de la Matriz:

    tramos cuyo nombre EMPIEZA con «Ruta 5»  ..........      7
    tramos que dicen «Cruce Ruta 5 (X) – Y»  ..........  1.522

Los 1.522 son caminos que **nacen de** la Ruta 5, no la Ruta 5. Y los 7 son
accesos y pasos inferiores. **La principal ruta longitudinal del país no está en
el catastro: sólo existe como referencia para nombrar otros caminos.**

Es el tipo de hueco que la vista por comuna nunca deja ver, porque uno mira
Rancagua, cuenta doscientos tramos viales y no nota que falta el que los cruza a
todos.

★ DE DÓNDE SE TRAE
--------------------
De OpenStreetMap, vía Overpass. La ruta está partida en relaciones por sector
—«Longitudinal Norte», «Longitudinal Sur» y sus tramos—, todas con `ref=5`. Se
piden con geometría y se guarda el crudo tal como llega, sin tocar, según la
regla del proyecto: el dato original se conserva y las transformaciones van
aparte.

⚠️ Esto es CONSOLIDACIÓN de dato público abierto, no catastro propio. La
distinción importa: la Matriz sigue sin tener la Ruta 5, y este archivo no la
agrega a la Matriz — la trae al lado, declarando su origen.

USO
---
    ../.venv-esa/bin/python adaptadores/ruta5.py
"""

import json
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI.parent / "datos" / "crudo" / "ruta5"
URL = "https://overpass-api.de/api/interpreter"

# ★★ RELACIONES **Y** WAYS SUELTOS
#   Pedir sólo las relaciones perdía el extremo norte: en OSM el tramo entre
#   Quillagua y la frontera con Perú no está agrupado en ninguna relación, sino
#   como ways individuales con `ref=5`. Con la primera consulta la traza moría
#   en la latitud −21,63 y Arica quedaba fuera — 557 km de Panamericana
#   invisibles, justo el trecho que llega al paso internacional.
#   Los ways que ya vienen dentro de una relación se descartan después por id.
CONSULTA = """
[out:json][timeout:900];
area["ISO3166-1"="CL"][admin_level=2]->.cl;
(
  relation(area.cl)["type"="route"]["route"="road"]["ref"="5"];
  way(area.cl)["ref"="5"]["highway"~"^(motorway|trunk|primary)$"];
);
out geom;
"""


def main():
    destino = CRUDO / date.today().isoformat()
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / "overpass_ruta5.json"
    if archivo.exists():
        print(f"  ya estaba: {archivo} ({archivo.stat().st_size/1e6:.1f} MB)")
        return 0

    print("  pidiendo a Overpass la Ruta 5 completa con geometría…", flush=True)
    datos = urllib.parse.urlencode({"data": CONSULTA}).encode()
    peticion = urllib.request.Request(
        URL, data=datos,
        headers={"User-Agent": "MICR/1.0 (proyecto Infraestructura Crítica y Clima)"},
    )
    with urllib.request.urlopen(peticion, timeout=900) as r:
        cuerpo = r.read()
    archivo.write_bytes(cuerpo)

    d = json.loads(cuerpo)
    rel = [x for x in d.get("elements", []) if x.get("type") == "relation"]
    ways = [x for x in d.get("elements", []) if x.get("type") == "way"]
    tramos = sum(len(x.get("members", [])) for x in rel)
    nodos = sum(len(m.get("geometry", [])) for x in rel for m in x.get("members", []))
    nodos += sum(len(x.get("geometry", [])) for x in ways)
    lats = [p["lat"] for x in rel for m in x.get("members", [])
            for p in (m.get("geometry") or [])]
    lats += [p["lat"] for x in ways for p in (x.get("geometry") or [])]
    print(f"  relaciones : {len(rel)}")
    print(f"  tramos     : {tramos:,} en relaciones + {len(ways):,} sueltos")
    print(f"  vértices   : {nodos:,}")
    if lats:
        print(f"  cobertura  : lat {max(lats):.2f} a {min(lats):.2f}")
    print(f"  escrito    : {archivo} ({archivo.stat().st_size/1e6:.1f} MB)")
    for x in rel[:10]:
        t = x.get("tags", {})
        print(f"     · {t.get('name', '(sin nombre)')[:66]}")
    return 0


if __name__ == "__main__":
    print("=" * 74)
    print("RUTA 5 · trayendo de OpenStreetMap lo que el catastro no tiene")
    print("=" * 74)
    sys.exit(main())
