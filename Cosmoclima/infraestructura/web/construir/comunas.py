"""
GEOMETRÍA · las 345 comunas, de 98,6 MB a 1,4 MB
==================================================

★ EL PROBLEMA
---------------
`datos/capas/comunas.geojson` pesa **98,6 MB**. Ningún navegador lo abre. Hay que
simplificarlo sin que desaparezca ninguna comuna ni se rompa la jerarquía.

★ LA DECISIÓN QUE AHORRA 2,7 MB Y VARIAS PIEZAS
-------------------------------------------------
La primera versión generaba TRES archivos: comunas, provincias y regiones,
disolviendo cada nivel. Medido, no supuesto:

    comunas     1.426 KB
    provincias  1.221 KB
    regiones    1.472 KB   ← ¡más pesada que las comunas!

Disolver **no reduce el peso**, porque lo que pesa es el detalle de la costa
—fiordos, canales, islas— y ése sobrevive a la disolución. La región de
Magallanes tiene el mismo litoral tenga una comuna o diez.

**Por eso se genera un solo archivo, el de comunas.** Los otros dos niveles no
necesitan geometría propia: para pintar un mapa de calor por región basta darles
a todas las comunas de esa región el mismo color, y el resultado en pantalla es
idéntico. La agregación se hace por prefijo del código CUT, que está anidado por
construcción: `05101` pertenece a la provincia `051` y a la región `05`.

Una pieza en vez de tres, 1,4 MB en vez de 4,1 MB, y sin geometrías que puedan
desincronizarse entre sí.

★ SOBRE EL AVISO DE `mapshaper`
---------------------------------
Al simplificar avisa de intersecciones que no pudo reparar. Es esperable en la
costa sur de Chile, donde el litoral real es casi fractal: al quitar vértices,
polígonos muy estrechos se cruzan a sí mismos. No afecta al dibujo ni a la
agregación, que se hace por código y no por geometría. `keep-shapes` garantiza
lo único que importa aquí: **que ninguna comuna se pierda**, y eso se comprueba.

USO
---
    ../../.venv-esa/bin/python construir/comunas.py
"""

import json
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
CAPA = RAIZ / "datos" / "capas" / "comunas.geojson"
SALIDA = AQUI.parent / "publico" / "datos" / "comunas.topo.json"
ESPERADAS = 345


def construir():
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    print(f"  origen : {CAPA.name} ({CAPA.stat().st_size/1e6:.0f} MB)")
    orden = [
        "npx", "--yes", "mapshaper", str(CAPA),
        "-filter-fields", "CUT_COM,CUT_PROV,CUT_REG,COMUNA",
        # El CUT es un identificador y el JSON lo entrega como número: 01101
        # llega como 1101. Se restituyen los ceros aquí, no en el navegador.
        "-each", ('CUT_COM=String(CUT_COM).padStart(5,"0"), '
                  'CUT_PROV=String(CUT_PROV).padStart(3,"0"), '
                  'CUT_REG=String(CUT_REG).padStart(2,"0")'),
        "-simplify", "2%", "keep-shapes",
        "-o", str(SALIDA), "format=topojson",
    ]
    r = subprocess.run(orden, capture_output=True, text=True)
    if r.returncode != 0:
        print("  ✗ mapshaper falló:\n" + r.stderr[:600])
        return None

    d = json.loads(SALIDA.read_text(encoding="utf-8"))
    g = d["objects"]["comunas"]["geometries"]
    cuts = {x["properties"]["CUT_COM"] for x in g}
    vacias = [x for x in g if not x.get("arcs")]

    print(f"  salida : {SALIDA.name} ({SALIDA.stat().st_size/1e6:.2f} MB) "
          f"· reducción {CAPA.stat().st_size/SALIDA.stat().st_size:.0f}×")
    print(f"  comunas: {len(g)} · CUT distintos: {len(cuts)} · vacías: {len(vacias)}")

    # ★ El control que importa: simplificar NO puede perder comunas.
    if len(g) != ESPERADAS or len(cuts) != ESPERADAS or vacias:
        print(f"  ✗ se esperaban {ESPERADAS} comunas con geometría")
        return None
    print(f"  ✓ las {ESPERADAS} sobreviven a la simplificación")
    return {"comunas_geometria": len(g), "bytes": SALIDA.stat().st_size}


if __name__ == "__main__":
    print("=" * 70)
    print("GEOMETRÍA · simplificación de las comunas")
    print("=" * 70)
    sys.exit(0 if construir() else 1)
