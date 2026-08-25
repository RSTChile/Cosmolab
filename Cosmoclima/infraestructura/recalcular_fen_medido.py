"""
RECÁLCULO DEL FEN CON DATO REAL, ELEMENTO POR ELEMENTO
========================================================

INSTRUCCIÓN QUE LO ORIGINA (Alexis, 20-ago-2026)
--------------------------------------------------
«Simplemente hay que recalcular el FEN para cada elemento de la submatriz, eso es
lo que manda… recuerda que el modelo se hizo en abstracto, sin datos, sin
localidades, sin infraestructura real.»

POR QUÉ HACÍA FALTA
-------------------
`SUBMATRICES_Y_EL_FEN_CONSTANTE.md` midió que **los 112 ítems de `Pen = Muy Alta`
tienen todos `FEN = Alta`**, sin una sola excepción, y que el FEN determina la
prioridad final casi por completo. Con una columna constante, `FENef` da el mismo
número para un puente, un hospital y un tranque de relaves en el mismo lugar.
Poblar más sub-matrices no arreglaba eso. Medir el FEN, sí.

QUÉ ES UN FEN MEDIDO
--------------------
El FEN es la Fragilidad ante Eventos Naturales del TIPO de elemento. Medida:

    tasa = eventos de causa natural que le pasaron a ese tipo
           ─────────────────────────────────────────────────
                    cuántos activos de ese tipo existen

En simple: de cada cien puentes, ¿cuántos falló el clima? ¿y de cada cien
kilómetros de calzada? El que falla más es más frágil. No hay que opinarlo.

═══════════════════════════════════════════════════════════════════════════
LAS SEIS REGLAS, FIJADAS ANTES DE CALCULAR
═══════════════════════════════════════════════════════════════════════════

R1 · VENTANA. Se usa el registro completo de emergencias del MOP, las dos capas:
     `historica_2014_2019` y `vigente_2022_2026`. **El registro tiene huecos
     declarados: 2020 y 2021 no existen, 2014 trae 2 casos y 2022 trae 3.** No se
     calcula ninguna tasa «por año» por eso. Lo que se compara son tipos entre
     sí DENTRO de la misma ventana, que es la misma para todos: el hueco afecta
     a todos por igual y por lo tanto no distorsiona la comparación.

R2 · NUMERADOR. Cuenta como evento natural el que el propio registro clasifica
     como `meteo`, `remocion_en_masa` o `meteo_y_remocion`.

     ★ HALLAZGO SOBRE LA FUENTE, encontrado al construir esta regla: la
     clasificación de causa del registro tiene sólo cuatro valores, y el cuarto
     —`otra_o_no_dice`, 2.903 de 6.141, el 47 %— **junta «la causa fue otra» con
     «no se anotó la causa»**. Son dos cosas distintas y quedaron en la misma
     bolsa. La consecuencia es que **NO existe un techo informativo**: como no
     hay ni un solo caso de «causa declarada NO natural», el techo aritmético es
     «todas las fallas de ese elemento fueron naturales», que no aporta nada.

     Por eso se reporta:
        · PISO  — sólo causa natural declarada (lo que se usa para el FEN)
        · TOTAL — todas las fallas del elemento, como cota superior TRIVIAL,
                  marcada como tal para que nadie la lea como una estimación
     El hueco del 47 % queda acotado, no resuelto.

R3 · DENOMINADOR. El conteo de activos del inventario propio del proyecto
     (`INVENTARIO_GEORREFERENCIADO.md`). Cada elemento declara su archivo y su
     unidad. Sin denominador no hay tasa: ese elemento se reporta con su conteo
     de fallas y se marca «sin denominador», nunca se le inventa uno.

R4 · SÓLO SE COMPARA DENTRO DEL MISMO SERVICIO. Vialidad reporta 4.544
     emergencias y Agua Potable Rural 948: eso mide cuánto reporta cada
     servicio, no cuánto falla cada cosa. Comparar un puente (Vialidad) con una
     red matriz (APR) mezcla fragilidad con intensidad de reporte. Los grupos
     comparables se declaran abajo y el cruce entre grupos se muestra pero se
     marca como NO COMPARABLE.

     ★ El grupo APR es el mejor experimento natural del registro: captación,
     red matriz y planta de tratamiento son componentes de los MISMOS 2.475
     sistemas, reportados por el MISMO servicio en la MISMA ventana. El
     denominador se cancela y la razón entre ellos es fragilidad pura.

R5 · DE LA TASA A LA ETIQUETA. Se usa la curva común del proyecto
     (`normalizar.f`), con la intensidad definida como el logaritmo en base 2 de
     la razón contra la mediana del grupo:

         intensidad = log₂( tasa del elemento / tasa mediana del grupo )

     No es una elección caprichosa: hace que la escala ordinal que la Matriz ya
     usa signifique algo medible.

         tasa = la mediana        → intensidad  0 → FEN 0,500 = «Media»
         tasa = 4× la mediana     → intensidad +2 → FEN 0,881 = «Alta»
         tasa = ¼ de la mediana   → intensidad −2 → FEN 0,119 = «Baja»

     O sea: **«Alta» pasa a querer decir «falla cuatro veces más que el elemento
     típico de su grupo»**, y deja de querer decir «nos pareció que era alta».

R5-bis · SÓLO SE ETIQUETA UN GRUPO CON AL MENOS CINCO ELEMENTOS MEDIBLES.
     Encontrado al revisar la primera corrida, y es un límite serio de R5: si el
     grupo tiene pocos elementos, **la mediana es uno de ellos**, y ese elemento
     recibe FEN 0,500 = «Media» por construcción, no por medición. Con un solo
     elemento el resultado es puro artefacto; con dos, los dos salen simétricos
     alrededor de 0,5; con tres, el del medio siempre sale «Media».
     En los grupos chicos se reporta la tasa y la razón entre elementos —que sí
     son medidas— y NO se emite etiqueta. Se prefiere no responder a responder
     con un artefacto.

R6 · LA UNIDAD DE ACTIVO SE DECLARA Y SE REPORTAN LAS DOS. Un puente es un punto
     y una carretera es una línea. Decir «un activo» no significa lo mismo para
     los dos, y la razón entre ellos CAMBIA según lo que se elija. Para las
     carreteras se calcula por tramo y por cada 100 km, y se muestran las dos.
     Elegir una es decisión del director, no del script.

USO
---
    ../.venv-esa/bin/python recalcular_fen_medido.py
    ../.venv-esa/bin/python recalcular_fen_medido.py --csv
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import normalizar                                  # noqa: E402

EMERGENCIAS = AQUI / "datos" / "mop_emergencias_viales.csv"
CAUSAS_NATURALES = {"meteo", "remocion_en_masa", "meteo_y_remocion"}
# ★ Por debajo de esto la tasa se apoya en tan pocos eventos que un caso más o
# menos la cambia de etiqueta. No se oculta: se marca «muestra fina».
MINIMO_EVENTOS_SOLIDOS = 5

# Etiquetas de la escala de tres de la MICR, con su valor en la curva común.
# Se calculan, no se copian: son los mismos anclajes que usa `normalizar`.
ANCLAS = {e: round(normalizar.f(normalizar.intensidad_ordinal(e, "fen_3")), 4)
          for e in ("baja", "media", "alta")}


def etiqueta_desde_valor(v):
    """El valor 0-1 vuelto etiqueta, por cercanía a los anclajes de la MICR."""
    return min(ANCLAS, key=lambda e: abs(ANCLAS[e] - v)).capitalize()


# ── R3 · denominadores: archivo, unidad y de dónde sale ─────────────────────
DENOMINADORES = {
    "tramos_viales": ("mop_tramos.csv", "tramo", None),
    "km_viales": ("mop_tramos.csv", "100 km", "largo_tramo_km"),
    "puentes": ("mop_puentes.csv", "puente", None),
    "sistemas_apr": ("inventario_agua_potable_rural.csv", "sistema APR", None),
    "estaciones_dga": ("dga_estaciones.csv", "estación", None),
}

# ── R4 · qué elemento pertenece a qué grupo comparable, y su ítem de la MICR ─
# `item` se pone SÓLO cuando la correspondencia es inequívoca. Donde el elemento
# del registro es un componente que la Matriz no tiene como fila propia, va None
# y se reporta igual: el dato existe aunque la Matriz no tenga dónde ponerlo.
MAPA = {
    # grupo VIALIDAD — Dirección de Vialidad del MOP
    "Carpeta de Rodadura":      ("VIALIDAD", 616, "tramos_viales"),
    "Puente":                   ("VIALIDAD", 618, "puentes"),
    # ★ Sin denominador a propósito: los caminos de acceso NO están en
    # `mop_tramos.csv` (que es la red vial enrolada). Usar ese denominador
    # —como hice en la primera corrida— infla el divisor y subestima la
    # fragilidad de un elemento que no está ahí adentro.
    "Pavimento Camino de Acceso": ("VIALIDAD", None, None),
    "Elementos de Saneamiento": ("VIALIDAD", None, None),
    "Pasarela":                 ("VIALIDAD", None, None),
    "Tunel":                    ("VIALIDAD", None, None),
    "Pontón":                   ("VIALIDAD", None, None),
    # grupo APR — Agua Potable Rural. Mismo denominador para todos: son
    # componentes de los mismos 2.475 sistemas. Éste es el experimento limpio.
    "Captación":                ("APR", None, "sistemas_apr"),
    "Red Matriz":               ("APR", 16, "sistemas_apr"),
    "Red Distribución":         ("APR", None, "sistemas_apr"),
    "Planta Tratamiento":       ("APR", 17, "sistemas_apr"),
    "Conducción":               ("APR", None, "sistemas_apr"),
    "Estanque Regulación":      ("APR", None, "sistemas_apr"),
    "Arranque":                 ("APR", None, "sistemas_apr"),
    "Bocatoma":                 ("APR", None, "sistemas_apr"),
    # grupo DGA
    "Estación Control Superficial": ("DGA", None, "estaciones_dga"),
    # grupo CAUCES — obras fluviales de la Dirección de Obras Hidráulicas.
    # No existe inventario nacional de defensas fluviales: van sin denominador.
    "Enrocado":                 ("CAUCES", None, None),
    "Gavión":                   ("CAUCES", None, None),
    "Muro de Defensa Costero":  ("CAUCES", None, None),
    "Control Aluvional":        ("CAUCES", None, None),
    "Cauce Receptor":           ("CAUCES", None, None),
    "Colector":                 ("CAUCES", None, None),
}


def contar_denominador(clave):
    """Cuántos activos hay, y en qué unidad. Se cuenta el archivo, no se cita."""
    archivo, unidad, campo_largo = DENOMINADORES[clave]
    ruta = AQUI / "datos" / archivo
    with ruta.open(encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    if campo_largo:                      # denominador por cada 100 km
        km = sum(float(f[campo_largo]) for f in filas if f.get(campo_largo))
        return km / 100.0, unidad, f"{archivo} · {km:,.0f} km"
    return float(len(filas)), unidad, f"{archivo} · {len(filas):,} filas"


def main():
    print("=" * 78)
    print("FEN MEDIDO · recálculo con el registro real de fallas del MOP")
    print("=" * 78)

    with EMERGENCIAS.open(encoding="utf-8") as fh:
        emergencias = list(csv.DictReader(fh))

    # R1 · la ventana y sus huecos, a la vista antes de cualquier número
    anios = defaultdict(int)
    for e in emergencias:
        if e["anio"]:
            anios[e["anio"]] += 1
    print(f"\n  emergencias en el registro : {len(emergencias):,}")
    print("  reparto por año            : " +
          " ".join(f"{a}:{n}" for a, n in sorted(anios.items())))
    print("  ★ huecos declarados        : 2020 y 2021 no existen; 2014 trae 2 y "
          "2022 trae 3.\n    Por eso no se calcula ninguna tasa «por año».")

    # Conteos por elemento: naturales, con causa declarada, total
    natural = defaultdict(int)
    con_causa = defaultdict(int)
    total = defaultdict(int)
    for e in emergencias:
        elem = e["elemento_afectado"]
        if not elem or elem == "Otro Elemento":
            continue
        total[elem] += 1
        if e["causa_heuristica"] in CAUSAS_NATURALES:
            natural[elem] += 1
            con_causa[elem] += 1
        elif e["causa_heuristica"] != "otra_o_no_dice":
            con_causa[elem] += 1

    sin_elemento = sum(1 for e in emergencias
                       if not e["elemento_afectado"]
                       or e["elemento_afectado"] == "Otro Elemento")
    print(f"  sin elemento identificado  : {sin_elemento:,} "
          f"({100*sin_elemento/len(emergencias):.0f} %) — quedan fuera, no se "
          f"reparten")

    # ── tasas por grupo ─────────────────────────────────────────────────────
    filas = []
    for grupo in ("VIALIDAD", "APR", "DGA", "CAUCES"):
        elementos = [e for e, (g, _, _) in MAPA.items() if g == grupo]
        print("\n" + "=" * 78)
        print(f"GRUPO {grupo}")
        print("=" * 78)

        medibles = []
        for elem in elementos:
            _, item, clave_den = MAPA[elem]
            n_nat, n_tot = natural.get(elem, 0), total.get(elem, 0)
            if n_tot == 0:
                continue
            # R2 · piso = natural declarada; techo TRIVIAL = todas las fallas,
            # porque el registro no distingue «otra causa» de «no se anotó».
            n_techo = n_tot

            if clave_den is None:
                filas.append(dict(grupo=grupo, elemento=elem, item_micr=item,
                                  fallas=n_tot, naturales=n_nat,
                                  denominador="", unidad="", tasa_piso="",
                                  tasa_techo="", FEN_medido="",
                                  etiqueta_medida="", estado="sin denominador"))
                continue
            den, unidad, procedencia = contar_denominador(clave_den)
            medibles.append((elem, item, n_tot, n_nat, n_techo, den, unidad,
                             procedencia))

        if not medibles:
            for elem in sorted(elementos, key=lambda e: -total.get(e, 0)):
                if total.get(elem):
                    print(f"    {total[elem]:5d} fallas · {natural.get(elem,0):5d} "
                          f"naturales   {elem}   → SIN DENOMINADOR, no hay tasa")
            continue

        # R5 · la mediana del grupo es la referencia
        tasas = sorted(n_nat / den for _, _, _, n_nat, _, den, _, _ in medibles)
        mediana = tasas[len(tasas) // 2] if len(tasas) % 2 else \
            (tasas[len(tasas) // 2 - 1] + tasas[len(tasas) // 2]) / 2

        # R5-bis · con pocos elementos la etiqueta sería un artefacto
        etiquetable = len(medibles) >= 5
        if etiquetable:
            print(f"\n  mediana del grupo: {mediana:.6f} eventos naturales por "
                  f"activo (la referencia de R5), sobre {len(medibles)} elementos\n")
        else:
            print(f"\n  ⚠ R5-bis · sólo {len(medibles)} elementos medibles: NO se "
                  f"emite etiqueta.\n    Con tan pocos, la mediana es uno de ellos "
                  f"y saldría «Media» por construcción.\n    Se reportan la tasa y "
                  f"la razón, que sí son medidas.\n")
        print(f"    {'elemento':28s} {'fallas':>7s} {'nat.':>6s} "
              f"{'denominador':>13s} {'tasa×1000':>10s} {'FEN':>7s}  etiqueta")
        for elem, item, n_tot, n_nat, n_techo, den, unidad, proc in sorted(
                medibles, key=lambda m: -(m[3] / m[5])):
            tasa = n_nat / den
            tasa_t = n_techo / den
            item_txt = f"#{item}" if item else "—"
            if not etiquetable:
                print(f"    {elem[:28]:28s} {n_tot:7d} {n_nat:6d} "
                      f"{den:13,.0f} {1000*tasa:10.3f} {'—':>7s}  "
                      f"(sin etiqueta por R5-bis)   {item_txt}")
                filas.append(dict(grupo=grupo, elemento=elem, item_micr=item or "",
                                  fallas=n_tot, naturales=n_nat,
                                  denominador=round(den, 1), unidad=unidad,
                                  tasa_piso=round(tasa, 8),
                                  tasa_techo=round(tasa_t, 8),
                                  FEN_medido="", etiqueta_medida="",
                                  estado="tasa sin etiqueta (R5-bis)"))
                continue
            fen = normalizar.f(math.log2(tasa / mediana)) if tasa > 0 else 0.0
            etiqueta = etiqueta_desde_valor(fen)
            # ★ Una tasa construida sobre 2 eventos no es una tasa: es un rumor.
            # Se marca y se registra, en vez de presentarla igual que una de 40.
            fina = n_nat < MINIMO_EVENTOS_SOLIDOS
            print(f"    {elem[:28]:28s} {n_tot:7d} {n_nat:6d} "
                  f"{den:13,.0f} {1000*tasa:10.3f} {fen:7.4f}  "
                  f"{etiqueta}{'  ⚠ muestra fina' if fina else '':17s} {item_txt}")
            filas.append(dict(grupo=grupo, elemento=elem, item_micr=item or "",
                              fallas=n_tot, naturales=n_nat,
                              denominador=round(den, 1), unidad=unidad,
                              tasa_piso=round(tasa, 8), tasa_techo=round(tasa_t, 8),
                              FEN_medido=round(fen, 4), etiqueta_medida=etiqueta,
                              estado="medido" if not fina
                                     else "medido · muestra fina"))

        # los del grupo que no tienen denominador, para que no desaparezcan
        huerfanos = [e for e in elementos
                     if total.get(e) and MAPA[e][2] is None]
        if huerfanos:
            print("\n    sin denominador (se reporta el conteo, no la tasa):")
            for elem in sorted(huerfanos, key=lambda e: -total[elem := e]):
                print(f"      {total[elem]:5d} fallas · {natural.get(elem,0):5d} "
                      f"naturales   {elem}")

    # ── R6 · la unidad cambia la respuesta ──────────────────────────────────
    print("\n" + "=" * 78)
    print("R6 · LA UNIDAD DE ACTIVO CAMBIA EL RESULTADO — las dos, a la vista")
    print("=" * 78)
    n_carp = natural.get("Carpeta de Rodadura", 0)
    n_pue = natural.get("Puente", 0)
    den_tr, _, _ = contar_denominador("tramos_viales")
    den_km, _, _ = contar_denominador("km_viales")
    den_pu, _, _ = contar_denominador("puentes")
    print(f"\n  calzada, por tramo     : {1000*n_carp/den_tr:8.3f} por mil tramos")
    print(f"  calzada, por 100 km    : {1000*n_carp/den_km:8.3f} por mil ·100 km")
    print(f"  puente, por puente     : {1000*n_pue/den_pu:8.3f} por mil puentes")
    print(f"\n  razón calzada/puente por TRAMO   : "
          f"{(n_carp/den_tr)/(n_pue/den_pu):.2f} a 1")
    print(f"  razón calzada/puente por 100 KM  : "
          f"{(n_carp/den_km)/(n_pue/den_pu):.2f} a 1")
    print("\n  ★ Un puente es un punto y una carretera es una línea. «Un activo»\n"
          "    no quiere decir lo mismo para los dos, y la razón entre ellos\n"
          "    cambia según lo que se elija. El script no elige.")

    # ── lo que esto le hace a la Matriz ─────────────────────────────────────
    print("\n" + "=" * 78)
    print("LO QUE CAMBIA RESPECTO DE LA MATRIZ")
    print("=" * 78)
    medidos = [f for f in filas if f["estado"] == "medido"]
    print(f"\n  La Matriz dice «Alta» ({ANCLAS['alta']}) para los "
          f"{len(medidos)} elementos medidos, sin excepción.")
    print("  El registro de fallas dice:\n")
    rep = defaultdict(list)
    for f in medidos:
        rep[f["etiqueta_medida"]].append(f["elemento"])
    for etiqueta in ("Alta", "Media", "Baja"):
        if rep[etiqueta]:
            print(f"      {etiqueta:6s} ({ANCLAS[etiqueta.lower()]:.3f}) : "
                  f"{', '.join(rep[etiqueta])}")
    iguales = len(rep["Alta"])
    print(f"\n  Coinciden con la Matriz: {iguales} de {len(medidos)}. "
          f"Cambian: {len(medidos)-iguales}.")

    if "--csv" in sys.argv:
        destino = AQUI / "datos" / "fen_medido_por_elemento.csv"
        with destino.open("w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
        print(f"\n  escrito: {destino}  ({len(filas)} filas)")

    print("\n" + "=" * 78)
    print("LO QUE ESTE RECÁLCULO NO PUEDE HACER TODAVÍA")
    print("=" * 78)
    print("""
  · Sólo cubre infraestructura del MOP. Subestaciones, hospitales, escuelas,
    relaves y telecomunicaciones NO están en este registro y siguen con el FEN
    heredado. Para la energía existe `sec_cortes.csv` (304.419 cortes con comuna
    y hora), que es otro registro y necesita su propio tratamiento.
  · La comparación entre grupos no es válida (R4): mide cuánto reporta cada
    servicio, no cuánto falla cada cosa.
  · El 47 % de causa no declarada se acota con piso y techo, no se resuelve.
  · No se cierra nada sin el director.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
