"""
EL INVIERNO DE 2026 COMO EXPERIMENTO NATURAL
==============================================

INSTRUCCIÓN (Alexis, 21-ago-2026)
-----------------------------------
«Calibra Pen primero… hay que calcular el FEN contra los ejemplos reales que ya
tenemos, por ejemplo, este mismo mes con los cortes de carretera documentados
desde la lluvia del mes pasado.»

QUÉ ES ESTE EPISODIO Y POR QUÉ SIRVE
--------------------------------------
El registro de emergencias del Ministerio de Obras Públicas trae **1.503
emergencias sólo en julio de 2026**, contra 29 en enero. En el invierno completo
—junio, julio y agosto— hay **908 de causa natural, todas con coordenada**.

Eso es un experimento natural: una causa conocida y fechada, y un efecto
documentado activo por activo. Es la primera vez que el proyecto tiene un
resultado observable contra el cual ajustar algo.

★ Y hay una coincidencia que conviene mirar: las comunas más golpeadas son
Ovalle, Combarbalá, Canela, Monte Patria e Illapel — la región de Coquimbo, con
402 de las 908. El instrumento climático, corrido ayer sobre las Direcciones
Regionales de SENAPRED sin saber nada de esto, había puesto a **Coquimbo primero
del norte en julio de 2026, con 177 mm en 48 horas**.

LO QUE SE PUEDE Y NO SE PUEDE CALIBRAR CON ESTO
-------------------------------------------------
`Pen` (Prioridad Estratégica para Desastres Naturales) combina `FEN` (Fragilidad
ante Eventos Naturales), `IB` (Importancia Base) y `FVT` (Factor de
Vulnerabilidad Total). **Las tres son propiedades del TIPO de elemento, no del
activo.** Los 14.039 tramos de la Red Vial comparten el ítem 616, así que
comparten las tres entradas y tienen el mismo `Pen`.

⇒ **Dentro de un tipo, `Pen` es una constante y no puede ordenar nada.** No se
puede calibrar «qué tramo se cortó» con `Pen`; eso lo predice el clima y el
terreno, que es otra parte del proyecto.

⇒ **Entre tipos sí.** El invierno golpeó carpeta de rodadura, puentes, obras de
saneamiento, captaciones de agua y obras fluviales, y cada uno de ésos es un tipo
distinto con su propio `FEN`, `IB` y `FVT`. Ése es el nivel al que `Pen` trabaja
y el nivel al que se puede calibrar.

LA REGLA DEL DENOMINADOR, QUE ES LO QUE HACE HONESTA LA MEDIDA
----------------------------------------------------------------
Un puente de Magallanes no «sobrevivió» al temporal: **nunca estuvo expuesto**.
Contar todos los activos del país como denominador diluiría la tasa y premiaría
a los tipos que están lejos.

Por eso la exposición se acota a **las comunas que efectivamente tuvieron al
menos una emergencia de causa natural en el episodio**. Dentro de esas comunas,
la pregunta es limpia: de los activos que sí estuvieron bajo la lluvia, ¿qué
fracción falló?

USO
---
    ../.venv-esa/bin/python calibrar_pen_invierno2026.py
"""

import csv
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"
SUB = AQUI / "submatrices_excel"

MESES = ("6", "7", "8")
ANIO = "2026"
NATURALES = {"meteo", "remocion_en_masa", "meteo_y_remocion"}

# Qué elemento del registro de emergencias corresponde a qué sub-matriz.
# Sólo se mapea lo inequívoco; lo demás se reporta sin tasa.
TIPOS = [
    # (elemento en el registro, ítem, archivo de la sub-matriz, nombre legible)
    # ★ La Red Vial no trae comuna en la sub-matriz —la fuente del Ministerio no
    # la entrega—, pero sí se calculó por intersección el 19-ago y quedó en
    # `datos/tramos_zona_y_comuna.csv`, campo `comuna_principal`, 14.039 de
    # 14.039. Se usa ése.
    ("Carpeta de Rodadura", 616, "@tramos", "Carpeta de rodadura (la calzada)"),
    ("Puente", 618, "618 Puentes de Carreteras", "Puentes de carreteras"),
    ("Captación", 17, "17 Agua Potable Rural",
     "Captación de agua potable rural"),
    ("Red Matriz", 17, "17 Agua Potable Rural",
     "Red matriz de agua potable rural"),
    ("Planta Tratamiento", 17, "17 Agua Potable Rural",
     "Planta de tratamiento de agua potable rural"),
]


def norm(s):
    s = unicodedata.normalize("NFD", str(s or "").upper().strip())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def main():
    emergencias = [x for x in csv.DictReader(
        (DATOS / "mop_emergencias_viales.csv").open(encoding="utf-8"))
        if x["anio"] == ANIO and x["mes"] in MESES
        and x["causa_heuristica"] in NATURALES]
    micr = {int(x["n"]): x for x in csv.DictReader(
        (DATOS / "micr_sharepoint.csv").open(encoding="utf-8"))}

    golpeadas = {norm(x["comuna"]) for x in emergencias if x["comuna"]}
    print("=" * 88)
    print("EL INVIERNO DE 2026 COMO EXPERIMENTO NATURAL")
    print("=" * 88)
    print(f"\n  emergencias de causa natural (jun-jul-ago 2026) : {len(emergencias):,}")
    print(f"  comunas con al menos una                        : {len(golpeadas)}")
    print(f"  gravedad: " + " · ".join(
        f"{k} {v}" for k, v in Counter(x["gravedad"] for x in emergencias).most_common()))

    # ── la tasa por tipo, con exposición acotada a las comunas golpeadas ────
    print("\n" + "=" * 88)
    print("FRAGILIDAD MEDIDA EN ESTE EPISODIO · fallas por cada mil activos expuestos")
    print("=" * 88)
    print(f"\n    {'tipo':44s} {'fallas':>7s} {'expuestos':>10s} {'‰':>8s}  Matriz")
    print("    " + "-" * 82)

    conteo = Counter(x["elemento_afectado"] for x in emergencias)
    filas = []
    for elemento, item, archivo, legible in TIPOS:
        if archivo == "@tramos":
            tr = list(csv.DictReader(
                (DATOS / "tramos_zona_y_comuna.csv").open(encoding="utf-8")))
            expuestos = sum(1 for f in tr
                            if norm(f["comuna_principal"]) in golpeadas)
        else:
            ws = openpyxl.load_workbook(SUB / f"{archivo}.xlsx",
                                        read_only=True).active
            datos = list(ws.iter_rows(values_only=True))
            cab = {c: i for i, c in enumerate(datos[0])}
            expuestos = sum(1 for f in datos[1:]
                            if norm(f[cab["Comuna"]]) in golpeadas)
        fallas = conteo.get(elemento, 0)
        if expuestos == 0:
            print(f"    {legible[:44]:44s} {fallas:7d} {'sin dato':>10s}")
            continue
        tasa = 1000 * fallas / expuestos
        x = micr[item]
        filas.append(dict(elemento=legible, item=item, fallas=fallas,
                          expuestos=expuestos, tasa=tasa,
                          FEN=x["FEN"], IB=float(x["IB"]), FVT=float(x["FVT"]),
                          Pen=x["Pen"]))
        print(f"    {legible[:44]:44s} {fallas:7d} {expuestos:10,d} {tasa:8.2f}  "
              f"FEN {x['FEN']:5s} Pen {x['Pen']}")

    # ── lo que la Matriz dice contra lo que pasó ────────────────────────────
    print("\n" + "=" * 88)
    print("LO QUE LA MATRIZ PREDICE CONTRA LO QUE OCURRIÓ")
    print("=" * 88)
    orden_real = sorted(filas, key=lambda f: -f["tasa"])
    print("\n  Orden REAL de fragilidad en este episodio:")
    for i, f in enumerate(orden_real, 1):
        print(f"      {i}. {f['tasa']:7.2f} ‰  {f['elemento'][:46]:46s} "
              f"[la Matriz dice FEN={f['FEN']}, Pen={f['Pen']}]")

    distintos_fen = {f["FEN"] for f in filas}
    distintos_pen = {f["Pen"] for f in filas}
    print(f"\n  ★ La Matriz asigna {len(distintos_fen)} valor(es) distinto(s) de FEN "
          f"a estos {len(filas)} tipos: {distintos_fen}")
    print(f"    y {len(distintos_pen)} valor(es) distinto(s) de Pen: {distintos_pen}")
    if len(distintos_fen) == 1:
        print("\n    ⇒ NO PUEDE ORDENARLOS. Les asigna la misma fragilidad a todos,")
        print("      mientras el episodio los separa por un factor grande.")

    if len(orden_real) >= 2:
        peor, mejor = orden_real[0], orden_real[-1]
        if mejor["tasa"] > 0:
            print(f"\n    razón entre el más frágil y el menos frágil medido: "
                  f"{peor['tasa']/mejor['tasa']:.1f} a 1")

    # ── qué haría falta para calibrar los pesos ─────────────────────────────
    print("\n" + "=" * 88)
    print("QUÉ SE PUEDE CALIBRAR CON ESTO, Y QUÉ NO")
    print("=" * 88)
    print(f"""
  SE PUEDE · el FEN (Fragilidad ante Eventos Naturales) de estos {len(filas)} tipos,
    porque el episodio los separa y la Matriz no.

  NO SE PUEDE TODAVÍA · los pesos 0,5 / 0,3 / 0,2 de Pen. Ajustar tres pesos
    necesita más tipos que pesos, y con {len(filas)} tipos —de los cuales varios
    comparten ítem— el ajuste quedaría determinado por el ruido. Hacen falta
    más tipos con falla documentada: el registro de la Superintendencia de
    Electricidad y Combustibles (304.419 cortes) y el de deslizamientos de
    SERNAGEOMIN aportan tipos que este registro no tiene.

  NO SE PUEDE NUNCA CON ESTE EPISODIO · distinguir qué tramo concreto se cortó.
    Eso no lo decide Pen, que es constante dentro del tipo, sino el clima y el
    terreno del lugar.
""")
    destino = DATOS / "fragilidad_invierno2026.csv"
    if filas:
        with destino.open("w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
        print(f"  escrito: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
