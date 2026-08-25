"""
ACUICULTURA · las concesiones que pueblan el sector Alimentario
================================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): completar los tres sectores con más hueco de
dato. El Alimentario tenía **25 ítems físicos y cero activos**: era el único de
los tres sin ninguna fuente en disco.

★ LA FUENTE
-------------
El geoportal de la **Subsecretaría de Pesca y Acuicultura** publica en su carpeta
IDE_PUBLICO las **4.690 concesiones de acuicultura** como puntos, cada una con su
titular, su código de centro y —lo que importa aquí— su **grupo de especie**:

    SALMONES · PECES · MOLUSCOS · ALGAS · ABALONES o EQUINODERMOS

★★ POR QUÉ EL GRUPO DE ESPECIE DECIDE EL ÍTEM
-----------------------------------------------
La Matriz tiene **401 · Granjas Piscícolas (Peces)** y ningún ítem para moluscos,
algas ni equinodermos. Sin el grupo de especie habría que meter las 4.690 en el
ítem de peces —y dos tercios no lo son— o dejarlas todas fuera. Con él, cada una
va donde corresponde y las demás quedan contadas y declaradas.

★ ACCESO VERIFICADO (25-ago-2026)
-----------------------------------
· `https://geoportal.subpesca.cl/robots.txt` → **404**: el servidor no declara
  restricciones.
· `https://www.subpesca.cl/robots.txt` → **404** también.
· Servicio REST público, sin credenciales ni registro.

⚠️ La capa publica en **UTM 18S (EPSG:32718)**. En vez de reproyectar aquí —que
es donde se cuelan los errores— se le pide al propio servicio que entregue en
EPSG:4326 con `outSR`. La reproyección la hace quien es dueño del dato.

USO
---
    ../.venv-esa/bin/python adaptadores/inventario_acuicultura_subpesca.py explorar
    ../.venv-esa/bin/python adaptadores/inventario_acuicultura_subpesca.py bajar
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

AQUI = Path(__file__).resolve().parent.parent
BASE = ("https://geoportal.subpesca.cl/server/rest/services/IDE_PUBLICO/"
        "SRMPUB_ACUICULTURA_PT/MapServer/0")
HOY = date.today().isoformat()
CRUDO = AQUI / "datos" / "crudo" / "subpesca"

LOTE = 800
PAUSA = 1.2
REINTENTOS = 5
TIEMPO_LIMITE = 180

FUENTE = dict(
    id="subpesca_acuicultura",
    organismo="Subsecretaría de Pesca y Acuicultura (SUBPESCA)",
    producto="Concesiones de Acuicultura · geoportal IDE_PUBLICO",
    url=BASE,
    formato="esriJSON (se pide outSR=4326; el servicio reproyecta desde UTM 18S)",
    familia="ESTADO (servicio dependiente del Ministerio de Economía)",
    acceso="anonimo",
    acceso_verificado=1,
    robots_txt="geoportal.subpesca.cl/robots.txt → 404 (sin restricciones). "
               "www.subpesca.cl/robots.txt → 404. Verificado 25-ago-2026.",
    condiciones_uso="Servicio REST público sin credenciales. Uso como "
                    "REFERENCIA en investigación.",
    granularidad="concesión de acuicultura (punto)",
)


def pedir(params):
    url = BASE + "/query?" + urllib.parse.urlencode(params)
    for intento in range(REINTENTOS):
        try:
            pedido = urllib.request.Request(url, headers={"User-Agent": "MICR/1.0"})
            with urllib.request.urlopen(pedido, timeout=TIEMPO_LIMITE) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            if intento == REINTENTOS - 1:
                raise
            espera = PAUSA * (2 ** intento)
            print(f"      reintento {intento+1}/{REINTENTOS} en {espera:.0f}s ({e})")
            time.sleep(espera)
    return None


def contar():
    d = pedir({"where": "1=1", "returnCountOnly": "true", "f": "json"})
    return d.get("count", 0)


def explorar():
    print(f"SUBPESCA · {BASE}")
    n = contar()
    print(f"  concesiones de acuicultura: {n:,}")
    d = pedir({"where": "1=1", "outFields": "T_GRUPOESPECIE",
               "returnDistinctValues": "true", "returnGeometry": "false",
               "f": "json"})
    grupos = [list(f["attributes"].values())[0] for f in d.get("features", [])]
    print(f"  grupos de especie: {grupos}")
    return 0


def bajar():
    total = contar()
    print(f"  concesiones a bajar: {total:,}")
    salida = CRUDO / HOY
    salida.mkdir(parents=True, exist_ok=True)

    feats, offset, lote_n = [], 0, 0
    while offset < total:
        lote_n += 1
        print(f"   lote {lote_n} (offset {offset})", flush=True)
        d = pedir({
            "where": "1=1", "outFields": "*", "returnGeometry": "true",
            "outSR": "4326", "resultOffset": offset, "resultRecordCount": LOTE,
            "f": "json",
        })
        fs = d.get("features", [])
        if not fs:
            break
        for f in fs:
            g = f.get("geometry") or {}
            x, y = g.get("x"), g.get("y")
            feats.append({
                "type": "Feature",
                "geometry": ({"type": "Point", "coordinates": [x, y]}
                             if x is not None and y is not None else None),
                "properties": f.get("attributes", {}),
            })
        offset += LOTE
        time.sleep(PAUSA)

    con = sum(1 for f in feats if f["geometry"])
    print(f"   {len(feats):,} recibidos de {total:,} · {con:,} con geometría")
    if len(feats) != total:
        print(f"   ⚠️ NO cierra: faltan {total - len(feats):,}")

    (salida / "acuicultura.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats},
                   ensure_ascii=False), encoding="utf-8")
    (salida / "PROCEDENCIA.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in FUENTE.items())
        + f"\nbajado: {HOY}\nregistros: {len(feats)}\ncon_geometria: {con}\n",
        encoding="utf-8")
    print(f"   → datos/crudo/subpesca/{HOY}/acuicultura.geojson")
    return 0


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "explorar"
    sys.exit(explorar() if modo == "explorar" else bajar())
