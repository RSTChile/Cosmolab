"""
POR QUÉ 33 DE 846 · qué fracción de la Matriz es siquiera ubicable en un mapa
==============================================================================

Tras integrar todo lo que había en disco, el índice tiene 134.024 activos pero
sólo **33 ítems de 846** tienen alguno. La lectura fácil sería «faltan datos».
Este archivo comprueba si eso es cierto, y la respuesta es que en gran parte no:
**una porción grande de la Matriz no enumera cosas que puedan estar en un lugar**.

★ LO QUE SE HACE AQUÍ, Y LO QUE NO
------------------------------------
Se clasifica cada uno de los 846 elementos según **qué clase de cosa nombra**, no
según su importancia. Es una clasificación **por regla declarada sobre el nombre
del elemento**, no una medición: las reglas están abajo, a la vista, y cualquiera
puede discutirlas. Se declara así en la salida para que nadie la lea como un dato
observado.

★★ POR QUÉ IMPORTA LA DISTINCIÓN
----------------------------------
Un ítem como «Personal de TI (Gobierno)» o «Infraestructura Vulnerable a
Ransomware» no tiene coordenada ni la va a tener: no es un activo que se pueda
catastrar, es un rol o una categoría de riesgo. Contarlo como «ítem sin datos»
mezcla dos cosas muy distintas:

    hueco de dato        el ítem nombra algo físico que existe y está ubicado
                         en algún registro que aún no hemos incorporado
                         → se cierra buscando la fuente

    hueco de diseño      el ítem no nombra nada ubicable
                         → no se cierra con datos, y decir «faltan datos» ahí
                           es prometer algo que no va a llegar

Separar los dos convierte «3,9 % de cobertura» —que suena a fracaso— en dos
números que sí orientan el trabajo.

USO
---
    ../../.venv-esa/bin/python construir/cobertura_matriz.py
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DATOS = AQUI.parent / "publico" / "datos"
SALIDA = DATOS / "cobertura.json"

# ── Las reglas, en orden: la primera que calza manda ─────────────────────────
# Cada una es un par (clase, expresión). Están ordenadas de más específica a más
# general, porque «Sistemas de Energía de Respaldo (UPS)» calza con dos.
REGLAS = [
    # ★ Excepción primera: nombres que empiezan por «Sistemas de…» pero designan
    #   una instalación con dirección. Salieron de revisar los ítems que la regla
    #   general clasificaba mal: «Sistema de Tratamiento de Aguas Residuales» es
    #   una planta, no un programa.
    (FISICO_EXC := "activo físico ubicable",
     r"tratamiento de aguas|potabilización|bombeo|\(radar\)|riego agrícola|"
     r"navegación aérea|estaciones de bomberos"),
    ("vulnerabilidad",
     r"vulnerable a|vulnerables a"),
    ("persona o rol",
     r"^personal|personal \(|funcionarios|periodistas|visitantes|productores de "
     r"contenido|equipos de periodistas|estudiantes|docentes|profesionales|"
     r"trabajadores|bomberos|voluntarios"),
    ("contenido o registro",
     r"artículos de prensa|publicaciones en|\(contenido\)|archivos digitales|"
     r"bases de datos|registros |archivos históricos|archivos de transmisiones|"
     r"material genético|semillas"),
    ("programa o servicio digital",
     r"^sistemas? de|^plataformas|^portales|^servidores|^redes sociales|"
     r"^servicios digitales|wi-?fi|^software|\(cms\)|\(scada\)|\(blockchain\)|"
     r"^cadenas de suministro|^redes de comunicación|^protocolos"),
    ("móvil o equipo",
     r"^vehículos|^camiones|^ambulancias|^maquinaria|^tractores|^robots|"
     r"^equipos de|^kits de|^drones|^helicópteros|^buques|^aeronaves|"
     r"^generadores de respaldo|^fertilizantes"),
]
COMPILADAS = [(c, re.compile(x, re.I)) for c, x in REGLAS]
FISICO = FISICO_EXC


def clasificar(nombre, tiene_activos=False):
    """★★ EL DATO MANDA SOBRE LA REGLA.

    Si un ítem tiene activos con coordenada catastrados, entonces nombra algo
    ubicable —está demostrado por el catastro— y ninguna regla léxica puede
    decir lo contrario. Es la misma instrucción que gobierna todo el proyecto:
    el dato real manda sobre la clasificación abstracta.
    """
    if tiene_activos:
        return FISICO
    for clase, rx in COMPILADAS:
        if rx.search(nombre):
            return clase
    return FISICO


def main():
    matriz = json.loads((DATOS / "matriz.json").read_text(encoding="utf-8"))
    act = json.loads((DATOS / "activos_por_comuna.json").read_text(encoding="utf-8"))

    con = set()
    for d in act["por_comuna"].values():
        con.update(d)

    items = matriz["items"]
    for i in items:
        i["_con"] = str(i["n"]) in con
        i["_clase"] = clasificar(i["elemento"], i["_con"])

    porclase = defaultdict(lambda: [0, 0])
    for i in items:
        c = porclase[i["_clase"]]
        c[0] += 1
        if i["_con"]:
            c[1] += 1

    print(f"  ítems de la Matriz: {len(items)}")
    print(f"  con activos ubicados: {len(con)} ({100*len(con)/len(items):.1f} %)\n")
    print(f"  {'qué clase de cosa nombra el ítem':<34}{'ítems':>7}{'con activos':>13}")
    print("  " + "-" * 56)
    for c, (t, n) in sorted(porclase.items(), key=lambda x: -x[1][0]):
        print(f"  {c:<34}{t:>7}{n:>13}")

    fis = porclase[FISICO][0]
    print(f"\n  ★ De los {len(items)} ítems, sólo {fis} nombran algo que pueda")
    print(f"    tener coordenada. La cobertura real es {len(con)}/{fis} = "
          f"{100*len(con)/fis:.1f} %, no {100*len(con)/len(items):.1f} %.")

    # ── el hueco de dato: físicos, sin activos, por sector ───────────────────
    faltan = defaultdict(list)
    for i in items:
        if i["_clase"] == FISICO and not i["_con"]:
            faltan[i["sector"]].append({"n": str(i["n"]), "elemento": i["elemento"]})

    print(f"\n  ★★ HUECO DE DATO POR SECTOR (ítems físicos aún sin catastro)")
    print(f"  {'sector':<34}{'físicos':>9}{'con activos':>13}{'faltan':>8}")
    print("  " + "-" * 64)
    porsector = defaultdict(lambda: [0, 0])
    for i in items:
        if i["_clase"] != FISICO:
            continue
        s = porsector[i["sector"]]
        s[0] += 1
        if i["_con"]:
            s[1] += 1
    for s, (t, n) in sorted(porsector.items(), key=lambda x: -(x[1][0] - x[1][1])):
        print(f"  {s[:33]:<34}{t:>9}{n:>13}{t-n:>8}")

    doc = {
        "advertencia": (
            "Esta es una clasificación POR REGLA sobre el nombre del elemento, "
            "no una medición. Las reglas están en construir/cobertura_matriz.py, "
            "a la vista, y son discutibles."),
        "por_que": (
            "Decir «3,9 % de cobertura» mezcla dos cosas distintas: ítems que "
            "nombran algo físico y aún no tienen catastro —hueco de dato, se "
            "cierra buscando la fuente— e ítems que no nombran nada ubicable "
            "—hueco de diseño, no se cierra con datos—."),
        "total_items": len(items),
        "con_activos": len(con),
        "clases": {c: {"items": t, "con_activos": n}
                   for c, (t, n) in porclase.items()},
        "fisicos": fis,
        "cobertura_sobre_fisicos": round(100 * len(con) / fis, 1),
        "por_sector": {s: {"fisicos": t, "con_activos": n, "faltan": t - n}
                       for s, (t, n) in porsector.items()},
        "faltan_por_sector": {s: v for s, v in faltan.items()},
    }
    SALIDA.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    print("=" * 74)
    print("COBERTURA · qué fracción de la Matriz es siquiera ubicable")
    print("=" * 74)
    sys.exit(main())
