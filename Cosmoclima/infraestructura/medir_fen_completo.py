"""
MEDICIÓN COMPLETA DEL FEN CONTRA EL REGISTRO ENTERO DE FALLAS
==============================================================

INSTRUCCIÓN (Alexis, 21-ago-2026)
-----------------------------------
«Hay que calcular cada cosa dependiendo del tipo de infraestructura… no se puede
usar kilómetros para un puente.» · «Completar medición del FEN.»

QUÉ CAMBIA RESPECTO DE LA MEDICIÓN DEL INVIERNO
--------------------------------------------------
1. **Todo el registro, no un episodio**: 2014-2026 completo, las dos capas del
   Ministerio de Obras Públicas. 3.238 emergencias de causa natural en vez de 908.
2. **Cada tipo con su unidad**, por decisión del director.
3. **La escala de cuatro niveles**, también por decisión del director:
   Baja 0,119 · Moderada 0,339 · Alta 0,661 · Muy Alta 0,881.

★ CÓMO SE RESUELVE «CADA COSA CON SU UNIDAD» SIN PERDER LA COMPARACIÓN
------------------------------------------------------------------------
La instrucción crea una tensión real: si la calzada se mide por kilómetro y el
puente por puente, **las dos tasas no se pueden comparar** — y el `FEN` es
justamente una comparación entre tipos.

Se resuelve separando dos preguntas que se estaban mezclando:

**(a) ¿Qué tan frágil es un elemento de este tipo?** → **por activo, siempre.**
Es la pregunta que responde el `FEN`, y es la que le sirve al Comité para la
Gestión del Riesgo de Desastres: «de los puentes de esta comuna, ¿cuántos van a
fallar?». Un activo es un activo, sea puente o tramo — el tramo es la unidad que
el propio ministerio mantiene, inspecciona y numera.

**(b) ¿Cuánta infraestructura lineal se daña?** → **por cada 100 km**, y sólo
para los elementos lineales. Es una medida de gestión, no de fragilidad, y se
reporta al lado sin mezclarla con la anterior.

Así se cumple la instrucción —a un puente no se le aplican kilómetros— sin perder
la comparabilidad que el `FEN` necesita.

LAS REGLAS, FIJADAS ANTES DE CALCULAR
---------------------------------------
R1 · VENTANA. Registro completo. **Huecos declarados: 2020 y 2021 no existen,
     2014 trae 2 casos y 2022 trae 3.** No se calculan tasas «por año»; se compara
     entre tipos dentro de la misma ventana, que es la misma para todos.

R2 · NUMERADOR. Sólo causa natural declarada (`meteo`, `remocion_en_masa`,
     `meteo_y_remocion`). ★ El registro junta «otra causa» con «no se anotó» en un
     solo valor, así que no existe cota superior informativa. El hueco se declara,
     no se rellena.

R3 · DENOMINADOR. Activos del inventario propio en **las comunas que tuvieron al
     menos una emergencia de causa natural en la ventana**. Un puente de
     Magallanes no sobrevivió al temporal: nunca estuvo expuesto.

R4 · SÓLO SE COMPARA DENTRO DEL MISMO SERVICIO. Vialidad reporta 4.544
     emergencias y Agua Potable Rural 948: eso mide cuánto reporta cada servicio,
     no cuánto falla cada cosa.

R5 · DE LA TASA A LA ETIQUETA. `intensidad = log₂(tasa / mediana del grupo)` y
     después la curva común. Con la escala de cuatro niveles eso significa:

         tasa = la mediana        → 0,500 · entre Moderada y Alta
         tasa = 4× la mediana     → 0,881 · Muy Alta
         tasa = ¼ de la mediana   → 0,119 · Baja

R5-bis · SÓLO SE ETIQUETA UN GRUPO CON AL MENOS CINCO ELEMENTOS MEDIBLES. Con
     menos, la mediana es uno de ellos y saldría «Media» por construcción.

USO
---
    ../.venv-esa/bin/python medir_fen_completo.py
"""

import csv
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import openpyxl

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import normalizar                                     # noqa: E402
import cclimp                                         # noqa: E402

DATOS = AQUI / "datos"
SUB = AQUI / "submatrices_excel"
NATURALES = {"meteo", "remocion_en_masa", "meteo_y_remocion"}
MINIMO_SOLIDO = 5          # menos eventos que esto: la tasa es un rumor

# (elemento del registro, ítem, fuente del denominador, nombre legible, lineal)
TIPOS = [
    ("Carpeta de Rodadura", 616, "@tramos", "Carpeta de rodadura (la calzada)", True),
    ("Puente", 618, "618 Puentes de Carreteras", "Puentes de carreteras", False),
    ("Captación", 17, "17 Agua Potable Rural", "Captación", False),
    ("Red Matriz", 16, "17 Agua Potable Rural", "Red matriz", False),
    ("Red Distribución", 16, "17 Agua Potable Rural", "Red de distribución", False),
    ("Planta Tratamiento", 17, "17 Agua Potable Rural", "Planta de tratamiento", False),
    ("Conducción", 16, "17 Agua Potable Rural", "Conducción", False),
    ("Estanque Regulación", 17, "17 Agua Potable Rural", "Estanque de regulación", False),
    ("Arranque", 16, "17 Agua Potable Rural", "Arranque", False),
    ("Bocatoma", 3, "17 Agua Potable Rural", "Bocatoma", False),
]
GRUPO = {"Carpeta de Rodadura": "VIALIDAD", "Puente": "VIALIDAD"}


def norm(s):
    s = unicodedata.normalize("NFD", str(s or "").upper().strip())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def etiqueta(v):
    """El valor 0-1 vuelto etiqueta, en la escala de CUATRO niveles adoptada."""
    tabla = cclimp.FEN_BASE["cuatro"]
    return min(tabla, key=lambda e: abs(tabla[e] - v)).title()


def main():
    em = [x for x in csv.DictReader(
        (DATOS / "mop_emergencias_viales.csv").open(encoding="utf-8"))
        if x["causa_heuristica"] in NATURALES]
    micr = {int(x["n"]): x for x in csv.DictReader(
        (DATOS / "micr_sharepoint.csv").open(encoding="utf-8"))}
    golpeadas = {norm(x["comuna"]) for x in em if x["comuna"]}
    conteo = Counter(x["elemento_afectado"] for x in em)

    print("=" * 92)
    print("FEN MEDIDO · registro completo del Ministerio de Obras Públicas")
    print("=" * 92)
    anios = Counter(x["anio"] for x in em if x["anio"])
    print(f"\n  emergencias de causa natural : {len(em):,}")
    print(f"  años presentes               : " +
          " ".join(f"{a}:{n}" for a, n in sorted(anios.items())))
    print(f"  comunas con al menos una     : {len(golpeadas)}")
    print("  ★ huecos declarados: 2020 y 2021 no existen; 2014 trae 2 y 2022 trae 3")

    # ── exposición y tasas ──────────────────────────────────────────────────
    tramos = list(csv.DictReader(
        (DATOS / "tramos_zona_y_comuna.csv").open(encoding="utf-8")))
    filas = []
    for elemento, item, fuente, legible, lineal in TIPOS:
        if fuente == "@tramos":
            expuestos = [f for f in tramos if norm(f["comuna_principal"]) in golpeadas]
            n_act = len(expuestos)
            km = sum(float(f["km_tramo"]) for f in expuestos if f["km_tramo"])
        else:
            ws = openpyxl.load_workbook(SUB / f"{fuente}.xlsx", read_only=True).active
            d = list(ws.iter_rows(values_only=True))
            cab = {c: i for i, c in enumerate(d[0])}
            n_act = sum(1 for f in d[1:] if norm(f[cab["Comuna"]]) in golpeadas)
            km = None
        n = conteo.get(elemento, 0)
        if not n_act:
            continue
        filas.append(dict(elemento=legible, item=item,
                          grupo=GRUPO.get(elemento, "APR"), naturales=n,
                          expuestos=n_act, km=km,
                          tasa=1000 * n / n_act,
                          tasa_km=(1000 * n / (km / 100)) if km else None,
                          FEN_matriz=micr[item]["FEN"]))

    for g in ("VIALIDAD", "APR"):
        gg = [f for f in filas if f["grupo"] == g]
        if not gg:
            continue
        print("\n" + "=" * 92)
        print(f"GRUPO {g} · fallas de causa natural por cada mil activos expuestos")
        print("=" * 92)
        tasas = sorted(f["tasa"] for f in gg)
        med = (tasas[len(tasas)//2] if len(tasas) % 2
               else (tasas[len(tasas)//2-1] + tasas[len(tasas)//2]) / 2)
        etiquetable = len(gg) >= 5
        if etiquetable:
            print(f"\n  mediana del grupo: {med:.4f} ‰ sobre {len(gg)} elementos\n")
        else:
            print(f"\n  ⚠ R5-bis · sólo {len(gg)} elementos: NO se etiqueta "
                  f"(la mediana sería uno de ellos).\n    Se reportan tasa y razón.\n")
        print(f"    {'elemento':30s} {'fallas':>7s} {'expuestos':>10s} {'‰':>9s} "
              f"{'FEN medido':>11s}  Matriz")
        print("    " + "-" * 84)
        for f in sorted(gg, key=lambda x: -x["tasa"]):
            fina = f["naturales"] < MINIMO_SOLIDO
            if etiquetable and f["tasa"] > 0 and med > 0:
                v = normalizar.f(math.log2(f["tasa"] / med))
                f["FEN_medido"] = round(v, 4)
                f["etiqueta"] = etiqueta(v)
                txt = f"{v:.4f}"
                et = f["etiqueta"]
            elif etiquetable:
                f["FEN_medido"], f["etiqueta"] = 0.0, "Baja"
                txt, et = "0.0000", "Baja"
            else:
                f["FEN_medido"], f["etiqueta"] = "", ""
                txt, et = "—", "(R5-bis)"
            print(f"    {f['elemento'][:30]:30s} {f['naturales']:7d} "
                  f"{f['expuestos']:10,d} {f['tasa']:9.3f} {txt:>11s}  "
                  f"{et:<10s} {f['FEN_matriz']}{'  ⚠ fina' if fina else ''}")
        if g == "VIALIDAD" and len(gg) == 2:
            a, b = sorted(gg, key=lambda x: -x["tasa"])
            print(f"\n    razón medida: {a['elemento'][:22]} falla "
                  f"{a['tasa']/b['tasa']:.1f} veces más que {b['elemento'][:22]}"
                  f"  (por activo)")

    # ── la medida de gestión, aparte y sin mezclar ──────────────────────────
    lin = [f for f in filas if f["km"]]
    if lin:
        print("\n" + "=" * 92)
        print("MEDIDA DE GESTIÓN, APARTE · sólo para lo lineal, por cada 100 km")
        print("=" * 92)
        for f in lin:
            print(f"\n    {f['elemento']}: {f['naturales']:,} fallas sobre "
                  f"{f['km']:,.0f} km expuestos → {f['tasa_km']:.1f} por mil·100 km")
        print("\n    ★ NO se compara con los puntuales. Es cuánta infraestructura")
        print("      lineal se daña, no qué tan frágil es un elemento.")

    # ── qué gana la Matriz con esto ─────────────────────────────────────────
    med_ok = [f for f in filas if f.get("etiqueta")]
    print("\n" + "=" * 92)
    print("LO QUE ESTO LE DA A LA MATRIZ")
    print("=" * 92)
    print(f"\n  tipos con FEN medido : {len(med_ok)} de los 837 de la Matriz")
    print(f"  la Matriz les dice   : {dict(Counter(f['FEN_matriz'] for f in med_ok))}")
    print(f"  la medición dice     : {dict(Counter(f['etiqueta'] for f in med_ok))}")
    vals = [f["FEN_medido"] for f in med_ok]
    if len(vals) > 1:
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        print(f"\n  ★ VARIANZA del FEN medido : {var:.5f}")
        print(f"    varianza del FEN heredado: 0.00000  (todos «Alta»)")
        print("\n    ⇒ es exactamente lo que faltaba: una entrada que VARÍA.")
        print("      Sin variación, el peso de FEN en Pen no se puede estimar.")

    destino = DATOS / "fen_medido_completo.csv"
    with destino.open("w", newline="", encoding="utf8") as fh:
        campos = ["elemento", "item", "grupo", "naturales", "expuestos", "km",
                  "tasa", "tasa_km", "FEN_matriz", "FEN_medido", "etiqueta"]
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    print(f"\n  escrito: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
