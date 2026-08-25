"""
MCSGS · del activo dañado al sistema que deja de moverse
=========================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): el MICR no modela el efecto en cadena, pero el
RMD 2.0 sí — Módulo de Colapso Sistémico Global Sincronizado (MCSGS 1.1) y
Módulo Coeficiente de Indeterminación Estructural (MCIE 1.1).

★ LA TESIS QUE ESTE MÓDULO TRAE
---------------------------------
    «Los sistemas no colapsan cuando son destruidos.
     Colapsan cuando dejan de poder moverse.»

Todo lo que esta aplicación calculaba hasta ahora responde «qué le pasa a este
activo». El MCSGS responde otra cosa: **qué le pasa al sistema cuando varios
activos fallan a la vez**. Y su premisa central es que la relación entre daño y
pérdida funcional NO es lineal: destruir el 50 % de los nodos críticos no reduce
la funcionalidad en 50 %, puede anularla.

★★ NODO ESTRUCTURAL Y NODO DE FLUJO
-------------------------------------
La distinción que la MICR no hace y que cambia cómo se lee todo:

    nodo estructural  presencia física cuya pérdida afecta capacidad instalada
                      — un hospital, una escuela, una planta química
    nodo de flujo     punto de paso obligado donde converge y se redistribuye
                      el flujo — un puente, un tramo sin alternativa, un paso
                      fronterizo, una subestación

**Un nodo estructural se reconstruye; un nodo de flujo, en el corto plazo, no.**
Y su fallo no se queda en él: interrumpe todo lo que pasaba por ahí.

★★ QUÉ SE PUEDE CALCULAR HOY, Y QUÉ NO
----------------------------------------
El ICSGS canónico es:

    ICSGS = min(100, √(FCN × FSS × FAS × FPI) × (1/FRC) × 100)

De sus cinco factores, este proyecto tiene dato medido para **cuatro**; el
25-ago se cerraron FAS y FPI con el registro de emergencias de SENAPRED:

    FCN  criticidad nodal      ✔ derivado de PF e IRMD, que la Matriz trae
    FSS  sincronización        ✔ cuántos nodos cruzan umbral EL MISMO DÍA,
                                 que es literalmente simultaneidad medida
    FAS  acoplamiento          ✔ medido el 25-ago sobre las 50.457 emergencias
                                 de SENAPRED (construir/acoplamiento.py)
    FPI  propagación           ✔ medido: retardo mediano de 2 días
    FRC  resiliencia           ✘ pide capacidad de absorción por sistema

⚠️ SIGUE SIN PUBLICARSE UN ICSGS, y ahora por un solo factor. La fórmula lleva
**1/FRC** como multiplicador: la resiliencia no es un sumando que se pueda
aproximar, DIVIDE el resultado. Con FRC a ojo, el índice completo cambiaría de
orden de magnitud según lo que uno suponga, que es exactamente el número con
apariencia de medición que este proyecto viene evitando. Se publican los cuatro
factores medidos y se declara el que falta.

⚠️ El MCIE (indeterminación estructural) NO se integra: mide el momento correcto
para decidir en un sistema geopolítico en transición, y sus tres ejes son juicios
sobre compromiso de actores y sensibilidad a eventos discretos. Aplicarlo a
infraestructura sería desnaturalizarlo.

USO
---
    ../../.venv-esa/bin/python construir/mcsgs.py
"""

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DATOS = AQUI.parent / "publico" / "datos"
SALIDA = DATOS / "mcsgs.json"

# ── Clasificación nodo de flujo ──────────────────────────────────────────────
# Criterio del MCSGS: «punto de paso obligado donde converge y se redistribuye
# el flujo», con alternativa nula o de alto costo en el corto plazo.
# Se clasifican por ítem y no por sector: dentro de Transporte, una carretera es
# flujo y un galpón de mantención no.
FLUJO = {
    "616": ("logístico", "Carreteras: el corte de un tramo sin alternativa "
                         "interrumpe todo lo que pasaba por ahí"),
    "618": ("logístico", "Puentes: paso obligado sobre un cauce; no hay rodeo "
                         "corto"),
    "622": ("logístico", "Puertos: punto de transferencia de mercancías"),
    "624": ("logístico", "Aeropuertos: pista única en la mayoría de los casos"),
    "639": ("logístico", "Navegación aérea: sin radar no hay operación"),
    "355": ("logístico", "Pasos fronterizos: paso obligado internacional"),
    "117": ("energético", "Transmisión eléctrica: el flujo va por la línea"),
    "120": ("energético", "Subestaciones: nodo donde converge y se redistribuye "
                          "la energía"),
    "16": ("hídrico", "Tuberías matrices: el agua pasa por ahí o no llega"),
    "17": ("hídrico", "Plantas de agua potable: punto único de tratamiento"),
    "46": ("hídrico", "Embalses: regulan el caudal aguas abajo"),
    "33": ("hídrico", "Tratamiento de aguas servidas: sin él, el sistema "
                      "sanitario se detiene"),
    "183": ("informacional", "Torres celulares: concentran el tráfico de su zona"),
}


def banda_irmd(v):
    return {"Alto": 1.0, "Medio": 0.6, "Bajo": 0.3}.get(v, 0.3)


def main():
    matriz = json.loads((DATOS / "matriz.json").read_text(encoding="utf-8"))
    act = json.loads((DATOS / "activos_por_comuna.json").read_text(encoding="utf-8"))
    items = matriz["items"]

    con_activos = set()
    for idx in act["por_comuna"].values():
        con_activos.update(idx)

    # ── FCN: criticidad nodal ───────────────────────────────────────────────
    # El MCSGS la deriva de PF + IRMD. PF ya viene normalizado 0-1 en la Matriz;
    # el IRMD aporta la banda de riesgo. Se combinan como media geométrica para
    # que un valor bajo en cualquiera de los dos arrastre el resultado — un nodo
    # sólo es crítico si lo es por las dos vías.
    pfs = [i["PF"] for i in items if isinstance(i.get("PF"), (int, float))]
    pf_max = max(pfs) if pfs else 1.0

    salida = {}
    for i in items:
        n = str(i["n"])
        pf = i.get("PF")
        if not isinstance(pf, (int, float)):
            continue
        fcn = ((pf / pf_max) * banda_irmd(i.get("IRMD"))) ** 0.5
        tipo, motivo = FLUJO.get(n, (None, None))
        salida[n] = {
            "elemento": i["elemento"],
            "sector": i["sector"],
            "irmd": i.get("IRMD"),
            "pf": round(pf, 4),
            "fcn": round(fcn, 4),
            "nodo": "flujo" if tipo else "estructural",
            "flujo_tipo": tipo,
            "por_que": motivo,
            "con_activos": n in con_activos,
        }

    flujo = [v for v in salida.values() if v["nodo"] == "flujo"]
    print(f"  ítems clasificados: {len(salida)}")
    print(f"    nodos de FLUJO      : {len(flujo)}  "
          f"({sum(1 for v in flujo if v['con_activos'])} con activos ubicados)")
    print(f"    nodos ESTRUCTURALES : {len(salida)-len(flujo)}")

    print(f"\n  {'ítem':<6}{'elemento':<42}{'tipo':<14}{'FCN':>7}")
    print("  " + "-" * 72)
    for n, v in sorted(flujo and salida.items() or [],
                       key=lambda t: -t[1]["fcn"]):
        if v["nodo"] != "flujo":
            continue
        print(f"  {n:<6}{v['elemento'][:41]:<42}{v['flujo_tipo']:<14}{v['fcn']:>7.3f}")

    fcns = sorted(v["fcn"] for v in salida.values())
    print(f"\n  FCN en los 846 ítems: mínimo {fcns[0]:.3f} · "
          f"mediana {fcns[len(fcns)//2]:.3f} · máximo {fcns[-1]:.3f}")

    doc = {
        "tesis": ("Los sistemas no colapsan cuando son destruidos. Colapsan "
                  "cuando dejan de poder moverse. — MCSGS 1.1, RMD 2.0"),
        "no_lineal": ("Destruir el 50 % de los nodos críticos no reduce la "
                      "funcionalidad en 50 %: bajo acoplamiento y sincronización "
                      "la pérdida funcional puede ser total con daño muy menor."),
        "nodos": {
            "estructural": ("presencia física cuya pérdida afecta capacidad "
                            "instalada — se puede reconstruir"),
            "flujo": ("punto de paso obligado donde converge y se redistribuye "
                      "el flujo — en el corto plazo, no"),
        },
        "icsgs": {
            "formula": "min(100, √(FCN × FSS × FAS × FPI) × (1/FRC) × 100)",
            "medidos": {
                "FCN": "criticidad nodal, derivada de PF e IRMD de la Matriz",
                "FSS": ("sincronización: cuántos nodos cruzan su umbral el mismo "
                        "día, calculado en la aplicación con el pronóstico"),
                "FAS": ("acoplamiento: medido sobre las 50.457 emergencias de "
                        "SENAPRED 2015-2024 con lift = P(A y B)/(P(A)×P(B)) en "
                        "el mismo día y la misma comuna. 58 pares lo superan. "
                        "★ El más acoplado llega a 2,0× — es DÉBIL, y eso es un "
                        "resultado: no sostiene una cadena de colapso automática"),
                "FPI": ("propagación: cuando dos tipos ocurren en secuencia "
                        "dentro de 3 días, el retardo mediano es de 2 días "
                        "(100 secuencias medidas)"),
            },
            "faltan": {
                "FRC": ("resiliencia — el 25-ago se midió una PROXY sobre el "
                        "daño en viviendas de las 50.457 emergencias de "
                        "SENAPRED (ver absorcion.json): de lo dañado, qué "
                        "fracción no se destruyó. NO se adopta como FRC: el "
                        "MCSGS lo define sobre el sistema, no sobre la "
                        "vivienda, y aceptar la equivalencia es una decisión, "
                        "no una medición"),
            },
            "por_que_no_se_publica": (
                "Falta un solo factor, pero es FRC y entra como 1/FRC: divide "
                "el resultado en vez de sumarse. Suponerlo cambiaría el índice "
                "de orden de magnitud. Se publican los cuatro medidos y se "
                "declara el que falta."),
        },
        "mcie": ("No se integra: mide el momento correcto para decidir en un "
                 "sistema geopolítico en transición, y sus ejes son juicios "
                 "sobre actores. Aplicarlo a infraestructura lo desnaturaliza."),
        "por_item": salida,
    }
    SALIDA.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    print("=" * 76)
    print("MCSGS · nodos de flujo y criticidad nodal")
    print("=" * 76)
    sys.exit(main())
