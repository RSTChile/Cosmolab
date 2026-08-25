"""
VALIDACIÓN DE CClimP — los tres criterios, tal como quedaron fijados
======================================================================

Los tres criterios de aprobación están escritos en
`VARIABLES_Y_METRICAS_PROYECTO.md`, sección 2, ANTES de calcular nada con ellos.
Este script los corre y los reporta. Se copian acá literalmente para que nadie
tenga que confiar en que se respetaron:

  1. En un mes cualquiera (mediana del país), `CClimP` debe valer 1,0 en MÁS DEL
     70 % de los pares activo-mes. Una perilla que se mueve siempre no informa.
  2. En los meses de los eventos ReTeRM, `CClimP` debe ser > 1,0 en MÁS DE LA
     MITAD de los casos.
  3. El ordenamiento de activos que produce `FENef` debe cambiar respecto del
     orden fijo de la matriz en al menos un mes de temporal documentado, y NO
     cambiar en un mes tranquilo.

  «Se aprueba sólo si se cumplen las tres. Si falla alguna, se reporta el fallo
  y no se ajustan los cortes para que pase.»

LO QUE HAY QUE FIJAR ANTES DE CORRER, Y SE FIJA ACÁ
-----------------------------------------------------
El criterio 3 habla de «un mes de temporal documentado» y «un mes tranquilo»
sin nombrarlos. Nombrarlos después de ver el resultado sería elegir el mes que
conviene. Se fijan ahora, y son los que el proyecto ya venía usando en otros
documentos, no meses nuevos elegidos para esta prueba:

    TEMPORAL   2015-03  el aluvión de Copiapó — el ancla del proyecto
    TEMPORAL   2026-07  el temporal que motivó el proyecto
    TRANQUILO  2015-08  el mes tranquilo del ancla, el mismo par que se usó
                        para probar que la corrección rareza→peligro funcionaba

CÓMO SE MIDE «QUE EL ORDEN CAMBIE» (criterio 3)
------------------------------------------------
Las 39 subestaciones del piloto comparten HOY una sola fila de la Matriz, el
ítem 120. O sea que el «orden fijo de la matriz» no es un orden: es un empate de
39. Con esa base, la pregunta se vuelve exacta y sin ambigüedad:

    ¿cuántos de los 741 pares de subestaciones (39×38/2) dejan de estar
    empatados cuando se calcula FENef?

  · en un mes de temporal debe ser mayor que cero  → el instrumento distingue
  · en un mes tranquilo debe ser exactamente cero  → el instrumento no inventa

★ LIMITACIÓN DECLARADA DEL CRITERIO 3. Las 39 subestaciones son todas del MISMO
tipo de elemento, así que comparten el mismo FEN base. Esta prueba mide, por lo
tanto, que FENef distingue LUGARES; NO puede medir que distinga TIPOS, porque
no hay más de un tipo en la muestra. Para probar la dimensión de tipo hace falta
un conjunto multi-sector con clima bajado, que hoy no existe. Queda pendiente
declarado, no resuelto.

LAS DOS DECISIONES ABIERTAS SE REPORTAN CON LAS DOS ALTERNATIVAS
------------------------------------------------------------------
La escala del FEN (tres o cuatro niveles) y la forma de FENef (producto o
potencia) son decisión del director. El script no elige: calcula las cuatro
combinaciones y muestra en qué se diferencian.

USO
---
    ../.venv-esa/bin/python validar_cclimp.py
    ../.venv-esa/bin/python validar_cclimp.py --csv    # + datos/fenef_39.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import cclimp                                    # noqa: E402
from adaptadores import era5                     # noqa: E402

# --- lo fijado antes de calcular -------------------------------------------
MESES_TEMPORAL = [(2015, 3), (2026, 7)]
MES_TRANQUILO = (2015, 8)
UMBRAL_C1 = 0.70          # más del 70 % de meses neutros
UMBRAL_C2 = 0.50          # más de la mitad de los eventos con perilla movida

# La fila 120 de la Matriz: lo que hoy comparten las 39 subestaciones.
FEN_ITEM_120 = "Alta"

# Fuentes de la prueba 2 (eventos independientes, no participaron del diseño).
CLIMA_RETERM = AQUI / "datos" / "clima_diario_reterm_era5_corregido.csv"
PUNTOS_RETERM = AQUI / "datos" / "reterm_puntos.csv"
EVENTOS_RETERM = AQUI / "datos" / "reterm_eventos.csv"


def serie_peligro(csv_diario=None, csv_puntos=None):
    """PelPre y confianza por (punto, año, mes), tal como los produce el adaptador.

    El adaptador está escrito para las subestaciones; apuntarlo a otro par de
    archivos es la forma en que el proyecto ya venía reutilizándolo (lo hace
    `validar_normalizacion_corregido.py`). Se restauran los valores originales
    al salir para no dejar el módulo contaminado para quien lo importe después.
    """
    original = (era5.CSV_DIARIO, era5.CSV_PUNTOS)
    if csv_diario:
        era5.CSV_DIARIO, era5.CSV_PUNTOS = csv_diario, csv_puntos
    try:
        obs, problema = era5.traer()
    finally:
        era5.CSV_DIARIO, era5.CSV_PUNTOS = original
    if problema:
        return None, None, problema

    peligro, confianza = {}, {}
    for o in obs:
        if o["variable"] != "peligro_precipitacion":
            continue
        anio = int(o["vigencia_inicio"][:4])
        mes = int(o["vigencia_inicio"][5:7])
        clave = (o["territorio_id"], anio, mes)
        peligro[clave] = float(o["valor_normalizado"])
        confianza[clave] = float(o["confianza"])
    return peligro, confianza, None


def es_lluvia(detonante):
    t = str(detonante or "").lower()
    return "luvia" in t or "recipitac" in t


def pares_desempatados(valores):
    """De los pares posibles, cuántos dejan de estar empatados.

    La base de comparación es el empate total: las 39 comparten la misma fila de
    la Matriz. Así que un par «cambia de orden» exactamente cuando sus dos
    valores dejan de ser iguales.
    """
    v = list(valores)
    total = distintos = 0
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            total += 1
            if v[i] != v[j]:
                distintos += 1
    return distintos, total


def main():
    print("=" * 78)
    print("VALIDACIÓN DE CClimP · los tres criterios fijados antes de calcular")
    print("=" * 78)

    peligro, confianza, problema = serie_peligro()
    if problema:
        print("SIN DATO:", problema)
        return 1

    activos = sorted({k[0] for k in peligro})
    print(f"\n  subestaciones con serie      : {len(activos)}")
    print(f"  pares (activo, mes)          : {len(peligro):,}")
    print(f"  período                      : "
          f"{min(k[1] for k in peligro)}–{max(k[1] for k in peligro)}")

    # La perilla para cada par, con el bloqueo por confianza activo.
    coef = {k: cclimp.coeficiente(v, confianza[k])[0] for k, v in peligro.items()}
    bloqueados = sum(1 for k, v in peligro.items()
                     if confianza[k] < cclimp.CONFIANZA_MINIMA
                     and cclimp.coeficiente(v)[0] > 1.0)
    print(f"  bloqueados por confianza baja: {bloqueados:,}  "
          f"(habrían movido la perilla y no se les creyó)")

    # ── CRITERIO 1 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("CRITERIO 1 · la perilla tiene que quedarse quieta la mayor parte del tiempo")
    print("=" * 78)
    neutros = sum(1 for v in coef.values() if v == 1.0)
    frac_global = neutros / len(coef)

    por_mes = defaultdict(list)
    for (act, anio, mes), v in coef.items():
        por_mes[(anio, mes)].append(v)
    frac_por_mes = sorted(sum(1 for v in vs if v == 1.0) / len(vs)
                          for vs in por_mes.values())
    mediana_mes = frac_por_mes[len(frac_por_mes) // 2]

    print(f"\n  CClimP = 1,0 en el {100*frac_global:.1f} % de los "
          f"{len(coef):,} pares activo-mes")
    print(f"  mes mediano del país: {100*mediana_mes:.1f} % de las "
          f"subestaciones con la perilla quieta")
    print(f"  reparto de la perilla:")
    reparto = defaultdict(int)
    for v in coef.values():
        reparto[v] += 1
    for v in (1.0, 1.2, 1.4, 1.6):
        print(f"      {v:.1f}  {reparto[v]:7,d}  ({100*reparto[v]/len(coef):5.1f} %)")

    c1 = frac_global > UMBRAL_C1 and mediana_mes > UMBRAL_C1
    print(f"\n  {'✓ PASA' if c1 else '✗ NO PASA'}  "
          f"(se exigía > {100*UMBRAL_C1:.0f} % en las dos lecturas)")
    print("\n  ★ HONESTIDAD SOBRE ESTE CRITERIO: los cortes SON el percentil 75\n"
          "    nacional, así que que el 75 % quede neutro es casi aritmética, no\n"
          "    un hallazgo. Lo que este criterio realmente comprueba es que la\n"
          "    distribución mes a mes no se desarma — que no haya meses en que\n"
          "    TODO el país se mueve. Eso es lo que mide la línea del mes mediano.")

    # ── CRITERIO 2 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("CRITERIO 2 · en los meses de deslizamiento la perilla tiene que moverse")
    print("=" * 78)
    pel_r, conf_r, problema = serie_peligro(CLIMA_RETERM, PUNTOS_RETERM)
    if problema:
        print("  SIN DATO:", problema)
        c2 = None
    else:
        with EVENTOS_RETERM.open(encoding="utf-8") as fh:
            eventos = [e for e in csv.DictReader(fh) if es_lluvia(e["detonante"])]
        con_evento = set()
        for e in eventos:
            clave = (f"ReTeRM · {e['comuna']}", int(e["anio"]), int(e["mes"]))
            if clave in pel_r:
                con_evento.add(clave)

        coef_r = {k: cclimp.coeficiente(pel_r[k], conf_r[k])[0] for k in con_evento}
        movidos = sum(1 for v in coef_r.values() if v > 1.0)
        frac = movidos / len(coef_r) if coef_r else 0.0

        print(f"\n  puntos ReTeRM con serie      : {len({k[0] for k in pel_r})}")
        print(f"  meses-con-deslizamiento      : {len(con_evento)}")
        print(f"  con la perilla movida (> 1,0): {movidos}  "
              f"({100*frac:.1f} %)")
        rep_r = defaultdict(int)
        for v in coef_r.values():
            rep_r[v] += 1
        for v in (1.0, 1.2, 1.4, 1.6):
            print(f"      {v:.1f}  {rep_r[v]:4d}  ({100*rep_r[v]/len(coef_r):5.1f} %)")

        # El contraste que le da sentido: ¿y en los meses SIN deslizamiento?
        sin_evento = [k for k in pel_r if k not in con_evento]
        mov_sin = sum(1 for k in sin_evento
                      if cclimp.coeficiente(pel_r[k], conf_r[k])[0] > 1.0)
        print(f"\n  contraste — meses SIN deslizamiento: "
              f"{100*mov_sin/len(sin_evento):.1f} % con la perilla movida "
              f"({mov_sin:,} de {len(sin_evento):,})")

        c2 = frac > UMBRAL_C2
        print(f"\n  {'✓ PASA' if c2 else '✗ NO PASA'}  "
              f"(se exigía > {100*UMBRAL_C2:.0f} %)")
        print("\n  ★ Los cortes se midieron sobre la distribución de las 39\n"
              "    subestaciones. Acá se aplican a 91 puntos distintos, que no\n"
              "    participaron en fijarlos. Ese traslado es justo lo que hace\n"
              "    que este criterio sea una prueba y no una comprobación.")

    # ── CRITERIO 3 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("CRITERIO 3 · FENef tiene que reordenar en temporal y NO reordenar en calma")
    print("=" * 78)
    print(f"\n  base de comparación: las {len(activos)} comparten el ítem 120 de la\n"
          f"  Matriz (FEN={FEN_ITEM_120}), o sea un empate de {len(activos)}.\n"
          f"  Pares posibles: {len(activos)*(len(activos)-1)//2}")

    combinaciones = [(esc, forma) for esc in ("tres", "cuatro")
                     for forma in ("producto", "potencia")]
    resultados = {}
    for anio, mes in MESES_TEMPORAL + [MES_TRANQUILO]:
        etiqueta = ("TEMPORAL" if (anio, mes) in MESES_TEMPORAL else "TRANQUILO")
        print(f"\n  ── {anio}-{mes:02d}  ({etiqueta}) " + "─" * 40)
        presentes = [a for a in activos if (a, anio, mes) in coef]
        if not presentes:
            print("      sin dato para ese mes")
            continue
        cc = {a: coef[(a, anio, mes)] for a in presentes}
        rep = defaultdict(int)
        for v in cc.values():
            rep[v] += 1
        print("      perilla: " + " · ".join(
            f"{v:.1f}×{rep[v]}" for v in (1.0, 1.2, 1.4, 1.6) if rep[v]))

        for escala, forma in combinaciones:
            base = cclimp.fen_base(FEN_ITEM_120, escala)
            valores = [round(cclimp.fenef(base, cc[a], forma), 6) for a in presentes]
            distintos, total = pares_desempatados(valores)
            niveles = len(set(valores))
            print(f"      FEN {base:.4f} ({escala:6s}) · {forma:9s} → "
                  f"{niveles} niveles distintos, "
                  f"{distintos:3d}/{total} pares desempatados   "
                  f"[{min(valores):.4f} … {max(valores):.4f}]")
            resultados[(anio, mes, escala, forma)] = distintos

    print("\n  Dictamen del criterio 3, combinación por combinación:")
    c3 = {}
    for escala, forma in combinaciones:
        en_temporal = any(resultados.get((a, m, escala, forma), 0) > 0
                          for a, m in MESES_TEMPORAL)
        en_calma = resultados.get((*MES_TRANQUILO, escala, forma), 0)
        ok = en_temporal and en_calma == 0
        c3[(escala, forma)] = ok
        print(f"      escala de {escala:6s} · {forma:9s} : "
              f"reordena en temporal {'sí' if en_temporal else 'NO'} · "
              f"reordena en calma {'NO' if en_calma == 0 else 'SÍ'}  "
              f"{'✓ PASA' if ok else '✗ NO PASA'}")

    # ── la disyuntiva del techo, con números ────────────────────────────────
    print("\n" + "=" * 78)
    print("LA DISYUNTIVA DEL TECHO · qué pierde cada forma (decisión del director)")
    print("=" * 78)
    print("\n  Cuántos pares activo-mes topan en FENef = 1,000 y quedan")
    print("  indistinguibles entre sí:\n")
    print(f"      {'escala':8s} {'forma':10s} {'saturados':>12s} "
          f"{'1,4 vs 1,6':>14s}")
    for escala, forma in combinaciones:
        base = cclimp.fen_base(FEN_ITEM_120, escala)
        vals = [cclimp.fenef(base, v, forma) for v in coef.values()]
        sat = sum(1 for v in vals if v >= 0.999999)
        v14 = cclimp.fenef(base, 1.4, forma)
        v16 = cclimp.fenef(base, 1.6, forma)
        distingue = "indistinguibles" if abs(v16 - v14) < 1e-9 else f"{v16-v14:+.4f}"
        print(f"      {escala:8s} {forma:10s} {sat:8,d} "
              f"({100*sat/len(vals):4.1f} %) {distingue:>14s}")
    print("\n  ★ La ficha decía que el techo sólo mordía con FEN = 0,881. Medido,\n"
          "    con la escala de cuatro (FEN = 0,661) también satura: 0,661 × 1,6 =\n"
          "    1,058, que se corta a 1,000. La saturación no depende de la escala\n"
          "    elegida, sólo de en qué escalón empieza.")

    if "--csv" in sys.argv:
        destino = AQUI / "datos" / "fenef_39.csv"
        campos = ["subestacion", "anio", "mes", "PelPre", "confianza", "CClimP"]
        for escala, forma in combinaciones:
            campos.append(f"FENef_{escala}_{forma}")
        with destino.open("w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            for (act, anio, mes), pel in sorted(peligro.items()):
                cc = coef[(act, anio, mes)]
                fila = dict(subestacion=act, anio=anio, mes=mes,
                            PelPre=round(pel, 4),
                            confianza=confianza[(act, anio, mes)], CClimP=cc)
                for escala, forma in combinaciones:
                    base = cclimp.fen_base(FEN_ITEM_120, escala)
                    fila[f"FENef_{escala}_{forma}"] = round(
                        cclimp.fenef(base, cc, forma), 4)
                w.writerow(fila)
        print(f"\n  escrito: {destino}  ({len(peligro):,} filas)")

    # ── veredicto ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("VEREDICTO")
    print("=" * 78)
    print(f"\n  criterio 1 (perilla quieta)      : {'PASA' if c1 else 'NO PASA'}")
    print(f"  criterio 2 (se mueve en eventos) : "
          f"{'PASA' if c2 else 'NO PASA' if c2 is not None else 'SIN DATO'}")
    todas = all(c3.values())
    alguna = any(c3.values())
    dictamen_c3 = ("PASA en las 4 combinaciones" if todas
                   else "PASA sólo en algunas" if alguna else "NO PASA")
    print(f"  criterio 3 (reordena)            : {dictamen_c3}")
    print("\n  ★ Ningún corte se movió para que esto pasara. Si alguno hubiera")
    print("    fallado, quedaría reportado como fallo.")
    print("\n  ★ NO SE CIERRA NADA sin el director: este script reporta, no aprueba.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
