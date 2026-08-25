"""
AFECTACIÓN POR ÍTEM · qué le pasa a cada cosa, y qué no sabemos que le pasa
============================================================================

INSTRUCCIÓN (Alexis, 24-ago-2026): «afectado debe considerar el umbral de la
pestaña 1, pero eso aplica a precipitaciones, por lo cual hay que pensar cómo
mostramos otras condiciones de afectación dependiendo del umbral propio de cada
ítem.»

★★ LA DISTINCIÓN QUE ORGANIZA TODO ESTE ARCHIVO
-------------------------------------------------
Hay dos cosas que se parecen y no son lo mismo:

  **AFECTACIÓN MEDIDA** — «esta carretera cedió con 108,7 mm en 72 h, medido
  sobre 570 tramos reales del temporal de julio». Se puede afirmar.

  **EXPOSICIÓN** — «esta torre de telecomunicaciones está en una celda donde va
  a llover más que el 99 % de los días de su historia». Es verdad, y no dice
  nada sobre si la torre se cae. Nadie ha registrado nunca con cuánta lluvia
  falla una torre.

Mezclarlas produciría el error más caro posible en una herramienta como ésta:
que alguien lea «hospital afectado» donde el dato sólo dice «va a llover fuerte
donde está el hospital». **De los 27 ítems con activos ubicados, sólo 6 tienen
umbral medido.** Los otros 21 quedan declarados como exposición, no afectación.

★ DE DÓNDE SALE CADA UMBRAL
-----------------------------
De lo único que se midió con denominador: los 1.241 tramos de vía del temporal
del 16-jul al 2-ago de 2026 (elementos del MOP) y los 612 eventos de CIGIDEN
(procesos). Sólo se transfiere cuando el ítem es **el mismo tipo de cosa** que lo
medido — una carretera a una carretera, un puente a un puente. No se estira a
«infraestructura en general», que es como se fabrican los números que parecen
rigurosos y no lo son.

★ LAS OTRAS AMENAZAS
----------------------
La Matriz declara para cada ítem su Factor de Exposición Natural (FEN), que
cubre sismo, viento, nieve y marejada además de lluvia. Aquí se conserva esa
declaración **sin umbral**, porque este proyecto todavía no ha medido ninguna de
ellas. Aparecen como amenazas reconocidas y no cuantificadas, que es su estado
real, y dejan el lugar preparado para cuando se midan.

USO
---
    ../../.venv-esa/bin/python construir/afectacion.py
"""

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DATOS = AQUI.parent / "publico" / "datos"
SALIDA = DATOS / "afectacion.json"

# ── ítem de la MICR → umbral medido, sólo cuando es el MISMO tipo de cosa ─────
# (umbral mm/72 h en escala Open-Meteo, origen, confianza)
MEDIDOS = {
    "616": (108.7, "MOP · carpeta de rodadura (570 tramos del temporal)", "alta"),
    "618": (135.1, "MOP · puente (37 tramos)", "media"),
    "33":  (145.9, "MOP · elementos de saneamiento (18 tramos)", "baja"),
    "16":  (159.9, "MOP · red matriz (16 tramos)", "baja"),
    "17":  (114.3, "MOP · captación (22 tramos)", "baja"),
    "46":  (94.7,  "CIGIDEN · desborde de río (9 casos, 7 tormentas)", "baja"),
}

# Por qué NO se le transfiere umbral al resto. El texto se muestra en pantalla:
# si alguien pregunta «¿y esto por qué no tiene número?», la respuesta está ahí.
SIN_UMBRAL = {
    "183": "Nadie ha registrado con cuánta lluvia falla una torre de telecomunicaciones.",
    "441": "No hay registro de fallas de escuelas por lluvia con fecha y lugar.",
    "265": "No hay registro de fallas de hospitales por lluvia con fecha y lugar.",
    "120": "Las subestaciones fallan por lluvia, pero nadie publica el umbral.",
    "117": "Las líneas de transmisión caen por viento y nieve más que por lluvia.",
    "42":  "Un tranque de relave tiene umbral propio de diseño, no transferible.",
    "624": "Una pista de aeropuerto drena distinto a una carretera.",
    "622": "Un puerto se afecta por marejada, no por lluvia acumulada.",
}
GENERICO = ("No hay ningún registro de fallas de este tipo de elemento por "
            "lluvia con fecha y lugar, así que no se le puede asignar umbral.")


def construir():
    DATOS.mkdir(parents=True, exist_ok=True)
    matriz = json.loads((DATOS / "matriz.json").read_text(encoding="utf-8"))
    act = json.loads((DATOS / "activos_por_comuna.json").read_text(encoding="utf-8"))

    con_activos = set()
    for idx in act["por_comuna"].values():
        con_activos.update(idx)

    por_item, medidos, expuestos = {}, 0, 0
    for i in matriz["items"]:
        n = str(i["n"])
        if n not in con_activos:
            continue
        if n in MEDIDOS:
            u, origen, conf = MEDIDOS[n]
            por_item[n] = {
                "tipo": "medido", "umbral_mm_72h": u,
                "origen": origen, "confianza": conf,
            }
            medidos += 1
        else:
            por_item[n] = {
                "tipo": "expuesto", "umbral_mm_72h": None,
                "porque": SIN_UMBRAL.get(n, GENERICO),
            }
            expuestos += 1
        # La amenaza que la Matriz reconoce para el ítem, sin cuantificar.
        por_item[n]["fen"] = i.get("FEN")
        por_item[n]["irmd"] = i.get("IRMD")
        por_item[n]["sector"] = i.get("sector")
        por_item[n]["elemento"] = i.get("elemento")

    doc = {
        "explicacion": {
            "medido": ("Hay registro de fallas reales de este mismo tipo de "
                       "elemento, con fecha y lugar. Se puede decir que cede."),
            "expuesto": ("Se sabe cuánta lluvia le va a caer encima, pero nadie "
                         "ha medido nunca qué le pasa a este tipo de elemento "
                         "con esa lluvia. NO es lo mismo que estar afectado."),
        },
        "ventana_horas": 72,
        "otras_amenazas": ("La Matriz reconoce sismo, viento, nieve y marejada "
                           "en su Factor de Exposición Natural, pero este "
                           "proyecto sólo ha calibrado lluvia. Las demás quedan "
                           "declaradas y sin umbral."),
        "por_item": por_item,
    }
    SALIDA.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    print(f"  ítems con activos ubicados : {len(por_item)}")
    print(f"    con umbral MEDIDO        : {medidos}")
    print(f"    sólo EXPOSICIÓN          : {expuestos}")
    print("\n  los que sí se pueden afirmar:")
    for n, v in sorted(por_item.items(), key=lambda t: -(t[1]["umbral_mm_72h"] or 0)):
        if v["tipo"] == "medido":
            print(f"     {v['elemento'][:44]:<46}{v['umbral_mm_72h']:>7.1f} mm"
                  f"   ({v['confianza']})")
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e3:.0f} KB)")
    return {"items_afectacion": len(por_item), "medidos": medidos}


if __name__ == "__main__":
    print("=" * 70)
    print("AFECTACIÓN POR ÍTEM · lo medido y lo apenas expuesto")
    print("=" * 70)
    sys.exit(0 if construir() else 1)
