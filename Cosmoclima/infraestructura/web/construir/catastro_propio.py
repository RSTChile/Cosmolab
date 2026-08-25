"""
CATASTRO PROPIO · lo que agregamos nosotros, y por qué va aparte
=================================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «en Chile hay 2 centrales nucleares de
investigación: La Reina y Lo Aguirre. La Reina está situada literalmente sobre
una de las fallas transversales, la de San Ramón, y en caso de un terremoto sobre
grado 8 se verá afectada. Es un escenario hipotético, pero yo lo incluiría.»

★★ POR QUÉ ESTO NO ES COMO EL RESTO DEL PROYECTO
--------------------------------------------------
Todo lo demás que esta aplicación muestra viene de un registro público: la
Matriz, el MOP, SENAPRED, CIGIDEN, Copernicus. Este archivo es la excepción, y
por eso vive separado y se muestra marcado.

**Es catastro creado, no consolidado.** Las coordenadas son públicas y
verificables, pero somos nosotros quienes decidimos que estos dos puntos entran.
Confundir eso con el resto sería empezar a fabricar la Matriz en vez de
auditarla.

★ EL HUECO QUE VIENE A CERRAR
-------------------------------
El sector Nuclear de la Matriz tiene **44 ítems** —Reactores Nucleares,
Almacenes de Materiales Nucleares, Sitios de Residuos, todos con riesgo Alto— y
**cero activos georreferenciados**. Es uno de los ocho sectores a ciegas. El país
tiene exactamente dos reactores de investigación y ninguno estaba ubicado.

★★ Y TRAE UN TERCER NIVEL DE EVIDENCIA
----------------------------------------
Hasta ahora había dos: **medido** (hay registro de fallas reales de este tipo de
elemento) y **expuesto** (se sabe cuánta lluvia le cae, no qué le pasa). El
escenario sísmico no es ninguno de los dos:

    escenario declarado — una hipótesis de planificación, sin umbral, sin
    frecuencia y sin validación. Se nombra como tal.

No se suma a ningún conteo de afectados. Un lector tiene que poder distinguir
«esta carretera cedió con 108 mm en julio, aquí está el registro» de «si hubiera
un sismo sobre grado 8 en esta falla, este reactor se vería afectado».

⚠️ La falla de San Ramón figura en la capa de SENAPRED como **«San Ramen»**
—error tipográfico de la fuente oficial— con 19 trazas inferidas. Pero esa capa
se espejó **sin geometría**: las 959 fallas traen nombre, tipo y largo, y ninguna
trae coordenadas. Por eso aquí NO se afirma ninguna distancia a la falla: hasta
rebajar la capa con su traza, la relación espacial queda declarada y sin medir.

USO
---
    ../../.venv-esa/bin/python construir/catastro_propio.py
"""

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DATOS = AQUI.parent / "publico" / "datos"
SALIDA = DATOS / "catastro_propio.json"

MALLA = 0.10

# Aportados por Alexis el 25-ago-2026, en grados-minutos-segundos.
ACTIVOS = [
    {
        "item": "133",                       # Reactores Nucleares
        "nombre": "Centro de Estudios Nucleares La Reina",
        "operador": "Comisión Chilena de Energía Nuclear (CCHEN)",
        "lat": -(33 + 25 / 60 + 42.95 / 3600),
        "lon": -(70 + 31 / 60 + 28.39 / 3600),
        "dms": "33°25'42.95\"S 70°31'28.39\"O",
        "escenarios": [{
            "tipo": "sismo",
            "descripcion": ("Situada sobre la falla de San Ramón. Un sismo de "
                            "magnitud superior a 8 la afectaría."),
            "estado": "escenario declarado — hipótesis de planificación, sin "
                      "umbral medido ni frecuencia estimada",
            "fuente": "aportado por Alexis López Tapia, 25-ago-2026",
            "verificable": ("La capa de fallas activas de SENAPRED registra la "
                            "falla como «San Ramen» (19 trazas inferidas) pero "
                            "se espejó SIN geometría, así que la distancia no "
                            "está medida."),
        }],
    },
    {
        "item": "133",
        "nombre": "Centro de Estudios Nucleares Lo Aguirre",
        "operador": "Comisión Chilena de Energía Nuclear (CCHEN)",
        "lat": -(33 + 27 / 60 + 3.11 / 3600),
        "lon": -(70 + 55 / 60 + 58.74 / 3600),
        "dms": "33°27'3.11\"S 70°55'58.74\"O",
        "escenarios": [],
    },
]


def celda(la, lo):
    return f"{round(la/MALLA)}_{round(lo/MALLA)}"


def main():
    matriz = json.loads((DATOS / "matriz.json").read_text(encoding="utf-8"))
    porN = {str(i["n"]): i for i in matriz["items"]}

    salida = []
    for a in ACTIVOS:
        it = porN.get(a["item"])
        salida.append({
            **a,
            "lat": round(a["lat"], 6),
            "lon": round(a["lon"], 6),
            "celda": celda(a["lat"], a["lon"]),
            "elemento": it["elemento"] if it else "",
            "sector": it["sector"] if it else "",
            "irmd": it["IRMD"] if it else "",
        })

    doc = {
        "que_es": ("Activos que este proyecto AGREGÓ, con coordenadas públicas "
                   "verificables. No vienen de la Matriz ni de ningún registro "
                   "oficial georreferenciado: se muestran marcados como tales."),
        "por_que": ("El sector Nuclear tiene 44 ítems y cero activos ubicados. "
                    "El país tiene dos reactores de investigación y ninguno "
                    "estaba en el catastro."),
        "niveles_de_evidencia": {
            "medido": "hay registro de fallas reales de este tipo de elemento",
            "expuesto": "se sabe cuánta lluvia le cae, no qué le pasa con ella",
            "escenario declarado": ("hipótesis de planificación, sin umbral, sin "
                                    "frecuencia y sin validación — NO se suma a "
                                    "ningún conteo de afectados"),
        },
        "activos": salida,
    }
    SALIDA.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"  activos agregados: {len(salida)}")
    for a in salida:
        print(f"\n  {a['nombre']}")
        print(f"     {a['dms']}  →  {a['lat']}, {a['lon']}")
        print(f"     celda {a['celda']} · ítem {a['item']} {a['elemento']} "
              f"({a['sector']}, riesgo {a['irmd']})")
        for e in a["escenarios"]:
            print(f"     ⚠️ escenario {e['tipo']}: {e['descripcion'][:60]}…")
    print(f"\n  escrito: {SALIDA.name}")
    return 0


if __name__ == "__main__":
    print("=" * 74)
    print("CATASTRO PROPIO · marcado como tal, separado de lo consolidado")
    print("=" * 74)
    sys.exit(main())
