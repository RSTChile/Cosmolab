"""
EVIDENCIA · los activos que YA fallaron, y con cuánta lluvia
=============================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «¿cuántos puentes pueden resultar afectados
con el pronóstico en X región? → Puente X, Puente Y, Puente Z. ¿Por qué? Porque
históricamente estos puentes con igual o menor pluviosidad han tenido estos
problemas.»

Decir «los puentes ceden con 135 mm» es una inferencia por tipo. Decir «este
puente se cortó el 20 de julio tras un episodio de 280 mm» es evidencia sobre
ese puente. La segunda vale mucho más, y este archivo la deja calculada.

★★ EL GOLPE, NO LA FECHA DEL INFORME
--------------------------------------
Calcular la lluvia del día que figura en el registro da resultados absurdos:
el Puente Cogotí aparecería cortado con **4,6 mm** y La Cantera con 3,0. Es el
desfase ya medido en este proyecto — el Ministerio publica la fecha del REPORTE,
no la del corte, con una mediana de 4 días de retraso.

Por eso se guarda el **golpe**: el mayor acumulado de 72 h en los diez días
hasta la fecha del registro. Con esa corrección Cogotí pasa de 4,6 a 282,4 mm y
la mediana de los 80 puentes con corte fechado sube de 109,2 a **139,4 mm**.

★ Y ESO VALIDA EL UMBRAL POR OTRA VÍA
---------------------------------------
El umbral del tipo «puente» se midió en 135,1 mm sobre 37 tramos del MOP. El
golpe mediano de 80 puentes concretos, cruzados contra ERA5-Land, da 139,4 mm.
Son dos caminos independientes que convergen: buena señal de que el umbral está
bien puesto.

⚠️ Cercanía no es identidad: el antecedente está a ≤250 m del activo, no
necesariamente sobre él. Se conserva la distancia para que se pueda juzgar.

USO
---
    ../../.venv-esa/bin/python construir/evidencia.py
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
DATOS = AQUI.parent / "publico" / "datos"
SERIE = RAIZ / "datos" / "clima_diario_celdas_era5land.csv"
SALIDA = DATOS / "evidencia.json"

MALLA = 0.10
DIAS_ATRAS = 10          # la ventana que absorbe el desfase de informe
MESES_ATRAS = 2          # para los eventos de CIGIDEN, que sólo traen año y mes


def celda(la, lo):
    return f"{round(la/MALLA)}_{round(lo/MALLA)}"


def main():
    if not SERIE.exists():
        print("  falta la serie ERA5-Land")
        return 1

    # ── los activos que tienen algún antecedente FECHADO ────────────────────
    registros = []
    for f in sorted((DATOS / "activos").glob("*.json")):
        cut = f.stem
        for a in json.loads(f.read_text(encoding="utf-8")):
            for h in a.get("h", []):
                if h["t"] == "via" and h.get("f"):
                    registros.append((cut, a, h, date.fromisoformat(h["f"])))
                    break
                if h["t"] == "ev" and h.get("a", "").isdigit():
                    try:
                        anio, mes = int(h["a"]), int(h.get("m") or 0)
                    except ValueError:
                        continue
                    if 1990 <= anio <= 2026 and 1 <= mes <= 12:
                        # sin día: se toma el fin de mes como referencia
                        d = (date(anio + (mes == 12), (mes % 12) + 1, 1)
                             - timedelta(days=1))
                        registros.append((cut, a, h, d))
                        break
    print(f"  activos con antecedente fechado: {len(registros):,}")

    # ── la serie, sólo para las celdas necesarias ───────────────────────────
    necesarias = {celda(a["y"], a["x"]) for _, a, _, _ in registros}
    print(f"  celdas a leer: {len(necesarias):,}", flush=True)
    diario = defaultdict(dict)
    with SERIE.open(encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for c, fe, v in r:
            if c in necesarias and v not in ("", "None"):
                diario[c][fe] = float(v)

    def ac72(c, d):
        t = 0.0
        for k in range(3):
            v = diario.get(c, {}).get((d - timedelta(days=k)).isoformat())
            if v is None:
                return None
            t += v
        return t

    def golpe(c, d, atras):
        picos = [ac72(c, d - timedelta(days=k)) for k in range(atras + 1)]
        picos = [p for p in picos if p is not None]
        return max(picos) if picos else None

    salida = defaultdict(list)
    sin_serie = 0
    for cut, a, h, d in registros:
        c = celda(a["y"], a["x"])
        atras = DIAS_ATRAS if h["t"] == "via" else MESES_ATRAS * 31
        g = golpe(c, d, atras)
        if g is None:
            sin_serie += 1
            continue
        salida[cut].append({
            "n": a["n"],
            "a": a["a"],
            "y": a["y"], "x": a["x"],
            "f": d.isoformat(),
            "mm": round(g, 1),
            "dia": round(ac72(c, d) or 0, 1),
            "t": h["t"],
            "g": h.get("g", "") if h["t"] == "via" else h.get("p", ""),
            "d": h["d"],
        })

    total = sum(len(v) for v in salida.values())
    print(f"  con lluvia calculable          : {total:,} "
          f"(sin serie: {sin_serie:,})")

    SALIDA.write_text(json.dumps({
        "criterio": (f"golpe = mayor acumulado de 72 h en los {DIAS_ATRAS} días "
                     "hasta la fecha del registro (para vías cortadas) o en los "
                     f"{MESES_ATRAS} meses previos (para eventos de CIGIDEN, que "
                     "sólo traen año y mes)"),
        "por_que": ("La fecha publicada es la del INFORME, no la del corte: sin "
                    "esta corrección el Puente Cogotí figuraría cortado con "
                    "4,6 mm en vez de 282,4."),
        "advertencia": ("El antecedente está a ≤250 m del activo, no "
                        "necesariamente sobre él. La distancia se conserva."),
        "por_comuna": salida,
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # ── resumen por tipo de elemento ────────────────────────────────────────
    matriz = json.loads((DATOS / "matriz.json").read_text(encoding="utf-8"))
    nom = {str(i["n"]): i["elemento"] for i in matriz["items"]}
    por_item = defaultdict(list)
    for v in salida.values():
        for e in v:
            por_item[e["n"]].append(e["mm"])
    print(f"\n  {'elemento':<44}{'con evidencia':>14}{'golpe mediano':>15}")
    print("  " + "-" * 73)
    for n, mms in sorted(por_item.items(), key=lambda t: -len(t[1]))[:12]:
        mms.sort()
        print(f"  {nom.get(n, n)[:43]:<44}{len(mms):>14}{mms[len(mms)//2]:>13.1f} mm")
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    print("=" * 76)
    print("EVIDENCIA · los que ya fallaron, y con cuánta lluvia")
    print("=" * 76)
    sys.exit(main())
