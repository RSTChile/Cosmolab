"""
Prueba la corrección rareza→peligro contra el criterio fijado de antemano.

El criterio está escrito en CORRECCION_RAREZA_PELIGRO.md, sección 5, ANTES de
calcular nada. Este script lo ejecuta y reporta. No lo ajusta.

Las cuatro exigencias, sobre todos los pares (subestación, mes) de 1990-2026:
  1. Copiapó mar-2015 en el 1% superior del país
  2. Punta Arenas mar-2015 fuera del 10% superior
  3. Copiapó ago-2015 fuera del 10% superior
  4. Copiapó mar-2015 por encima de Copiapó ago-2015

Se aprueba SÓLO si se cumplen las cuatro.
"""

import sys
from pathlib import Path

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI / "adaptadores"))
import era5  # noqa: E402

COPIAPO = "Subestación Copiapó (Urbana)"
PUNTA = "Subestación Punta Arenas (Urbana)"


def main():
    obs, problema = era5.traer()
    if problema:
        print("SIN DATO:", problema)
        return 1

    peligro = {}
    for o in obs:
        if o["variable"] != "peligro_precipitacion":
            continue
        anio, mes = int(o["vigencia_inicio"][:4]), int(o["vigencia_inicio"][5:7])
        peligro[(o["territorio_id"], anio, mes)] = (o["valor_normalizado"],
                                                    o["valor_original"],
                                                    o["notas"])

    todos = sorted(v[0] for v in peligro.values())
    n = len(todos)
    corte_1 = todos[int(n * 0.99)]
    corte_10 = todos[int(n * 0.90)]

    print(f"Universo: {n:,} pares (subestación, mes) de 1990 a 2026")
    print(f"  umbral del 1% superior : {corte_1:.4f}")
    print(f"  umbral del 10% superior: {corte_10:.4f}\n")

    def mirar(se, anio, mes):
        v, mm, nota = peligro[(se, anio, mes)]
        pos = sum(1 for x in todos if x < v) / n
        return v, float(mm), pos, nota

    casos = [
        ("Copiapó, marzo 2015 (el aluvión)", COPIAPO, 2015, 3),
        ("Punta Arenas, marzo 2015", PUNTA, 2015, 3),
        ("Copiapó, agosto 2015 (mes tranquilo)", COPIAPO, 2015, 8),
    ]
    print(f"{'caso':40s} {'mm/48h':>8s} {'peligro':>8s} {'percentil':>10s}")
    datos = {}
    for etiqueta, se, anio, mes in casos:
        v, mm, pos, nota = mirar(se, anio, mes)
        datos[(se, anio, mes)] = v
        print(f"{etiqueta:40s} {mm:8.1f} {v:8.4f} {100*pos:9.2f}%")
        print(f"{'':40s} {nota}")

    cop_mar = datos[(COPIAPO, 2015, 3)]
    pta_mar = datos[(PUNTA, 2015, 3)]
    cop_ago = datos[(COPIAPO, 2015, 8)]

    print("\n" + "=" * 74)
    print("CRITERIO FIJADO DE ANTEMANO")
    print("=" * 74)
    pruebas = [
        ("1. Copiapó mar-2015 en el 1% superior", cop_mar >= corte_1,
         f"{cop_mar:.4f} vs umbral {corte_1:.4f}"),
        ("2. Punta Arenas mar-2015 FUERA del 10% superior", pta_mar < corte_10,
         f"{pta_mar:.4f} vs umbral {corte_10:.4f}"),
        ("3. Copiapó ago-2015 FUERA del 10% superior", cop_ago < corte_10,
         f"{cop_ago:.4f} vs umbral {corte_10:.4f}"),
        ("4. Copiapó mar-2015 > Copiapó ago-2015", cop_mar > cop_ago,
         f"{cop_mar:.4f} vs {cop_ago:.4f}"),
    ]
    for texto, paso, detalle in pruebas:
        print(f"  {'✓' if paso else '✗'}  {texto:48s} {detalle}")

    aprueba = all(p for _, p, _ in pruebas)
    print(f"\n  → {'APRUEBA' if aprueba else 'NO APRUEBA'}: la corrección "
          f"{'distingue' if aprueba else 'NO distingue'} peligro de rareza")

    # Comparación con lo que daba la versión anterior, para ver qué cambió
    print("\n" + "=" * 74)
    print("ANTES Y DESPUÉS")
    print("=" * 74)
    anomalia = {}
    for o in obs:
        if o["variable"] == "anomalia_precipitacion":
            a, m = int(o["vigencia_inicio"][:4]), int(o["vigencia_inicio"][5:7])
            anomalia[(o["territorio_id"], a, m)] = o["valor_normalizado"]
    print(f"{'caso':40s} {'antes':>8s} {'ahora':>8s}")
    for etiqueta, se, anio, mes in casos:
        print(f"{etiqueta:40s} {anomalia[(se, anio, mes)]:8.4f} "
              f"{peligro[(se, anio, mes)][0]:8.4f}")

    print("\nLos 8 meses más peligrosos de los 36 años, en todo el país:")
    top = sorted(peligro.items(), key=lambda x: -x[1][0])[:8]
    for (se, anio, mes), (v, mm, _) in top:
        print(f"   {v:.4f}  {anio}-{mes:02d}  {float(mm):6.1f} mm/48h  {se}")

    return 0 if aprueba else 1


if __name__ == "__main__":
    sys.exit(main())
