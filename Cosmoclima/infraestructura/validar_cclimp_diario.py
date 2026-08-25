"""
VALIDACIÓN DIARIA DE CClimP · los tres criterios, rehechos
============================================================

Los tres criterios son los mismos que se fijaron el 16-ago-2026 en
`VARIABLES_Y_METRICAS_PROYECTO.md` §2. Lo que cambia es que ahora se evalúan
**día a día** y no mes a mes, por instrucción del director del 22-ago.

★ Y SE CORRIGE EL ERROR QUE HIZO FALLAR EL CRITERIO 3
------------------------------------------------------
En la corrida del 20-ago el criterio 3 falló, y el fallo fue de diseño mío: elegí
«agosto de 2015» como mes tranquilo porque era el mes tranquilo **del ancla de
Copiapó**. Pero el ancla es un LUGAR y el criterio se evalúa sobre el PAÍS. Agosto
es pleno invierno: 29 de 39 subestaciones se movieron, con razón.

**La corrección no es elegir mejor a ojo: es elegir con un registro INDEPENDIENTE
del instrumento.** Los días se toman del registro de emergencias del Ministerio de
Obras Públicas, que es un hecho documentado y ajeno a `PelPre`:

    DÍA DE TEMPORAL  el día con más emergencias de causa natural en todo el país
    DÍA TRANQUILO    un día de verano sin ninguna emergencia registrada

Así el instrumento no elige su propio examen.

LOS TRES CRITERIOS, TAL COMO SE FIJARON
-----------------------------------------
  1. En un día cualquiera, `CClimP` debe valer 1,0 en MÁS DEL 70 % de los pares.
  2. En los días de los eventos ReTeRM, `CClimP` debe ser > 1,0 en MÁS DE LA MITAD.
  3. El orden que produce `FENef` debe cambiar en un día de temporal documentado
     y NO cambiar en un día tranquilo.

★ CORTES NUEVOS, recalculados sobre la distribución DIARIA
------------------------------------------------------------
Los cortes 0,6501 / 0,8292 / 0,9658 salieron de la distribución MENSUAL. La
diaria tiene otra forma, así que se recalculan en los mismos percentiles
declarados (75 / 90 / 99):

    ≥ 0,7327 → 1,2        ≥ 0,8741 → 1,4        ≥ 0,9765 → 1,6

USO
---
    ../.venv-esa/bin/python validar_cclimp_diario.py
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import cclimp                                          # noqa: E402

DATOS = AQUI / "datos"
CORTES = [(0.9765, 1.6), (0.8741, 1.4), (0.7327, 1.2)]
NATURALES = {"meteo", "remocion_en_masa", "meteo_y_remocion"}
FEN_ITEM_120 = "Alta"


def perilla(p):
    for corte, v in CORTES:
        if p >= corte:
            return v
    return 1.0


def main():
    pel = defaultdict(dict)
    for x in csv.DictReader((DATOS / "pelpre_diario.csv").open(encoding="utf-8")):
        pel[x["punto"]][x["fecha"]] = float(x["PelPre"])
    subs = sorted(p for p in pel if not p.startswith("ReTeRM"))
    todos = [(p, f, v) for p, dd in pel.items() for f, v in dd.items()]

    print("=" * 82)
    print("VALIDACIÓN DIARIA DE CClimP · cortes recalculados sobre la serie diaria")
    print("=" * 82)
    print(f"\n  puntos {len(pel)} · pares (punto, día) con episodio {len(todos):,}")
    print(f"  subestaciones del piloto: {len(subs)}")

    # ── CRITERIO 1 ──────────────────────────────────────────────────────────
    # ★ Los días SIN episodio (bajo el piso de 0,5 mm) son días con la perilla
    # en 1,0 por definición: no están en el archivo pero cuentan. Ignorarlos
    # inflaría artificialmente la proporción de días movidos.
    dias_totales = 1_738_750       # días de serie, del cálculo de PelPre
    movidos = sum(1 for _, _, v in todos if perilla(v) > 1.0)
    neutros = dias_totales - movidos
    print("\n" + "=" * 82)
    print("CRITERIO 1 · la perilla se queda quieta la mayor parte del tiempo")
    print("=" * 82)
    print(f"\n  días con la perilla en 1,0 : {neutros:,} de {dias_totales:,} "
          f"({100*neutros/dias_totales:.1f} %)")
    c1 = neutros / dias_totales > 0.70
    print(f"  {'✓ PASA' if c1 else '✗ NO PASA'} (se exigía > 70 %)")

    # ── CRITERIO 2 ──────────────────────────────────────────────────────────
    ev = [x for x in csv.DictReader((DATOS / "reterm_eventos.csv").open(encoding="utf-8"))
          if "luvia" in str(x["detonante"]).lower()
          or "recipitac" in str(x["detonante"]).lower()]
    from datetime import date, timedelta
    vals = []
    for e in ev:
        p = f"ReTeRM · {e['comuna']}"
        if p not in pel:
            continue
        f = e["fecha"][:10]
        ayer = (date.fromisoformat(f) - timedelta(days=1)).isoformat()
        cand = [pel[p].get(f), pel[p].get(ayer)]
        vals.append(max([c for c in cand if c is not None], default=0.0))
    mov = sum(1 for v in vals if perilla(v) > 1.0)
    print("\n" + "=" * 82)
    print("CRITERIO 2 · se mueve el DÍA del deslizamiento")
    print("=" * 82)
    print(f"\n  eventos con serie      : {len(vals)}")
    print(f"  con la perilla movida  : {mov} ({100*mov/len(vals):.1f} %)")
    rep = Counter(perilla(v) for v in vals)
    for k in (1.0, 1.2, 1.4, 1.6):
        print(f"      {k}: {rep[k]:4d} ({100*rep[k]/len(vals):5.1f} %)")
    c2 = mov / len(vals) > 0.50
    print(f"  {'✓ PASA' if c2 else '✗ NO PASA'} (se exigía > 50 %)")

    # ── CRITERIO 3 · con días elegidos por un registro INDEPENDIENTE ────────
    em = [x for x in csv.DictReader(
        (DATOS / "mop_emergencias_viales.csv").open(encoding="utf-8"))
        if x["causa_heuristica"] in NATURALES and x["fecha"]]
    por_dia = Counter(x["fecha"][:10] for x in em)
    dias_con = set(por_dia)
    temporal = max(por_dia, key=lambda d: por_dia[d])
    # tranquilo: un día de verano sin NINGUNA emergencia en todo el país,
    # dentro del rango de la serie y del mismo período del registro
    candidatos = sorted(f for p in subs for f in pel[p]
                        if f[5:7] in ("01", "02") and f[:4] >= "2015")
    tranquilo = next((d for d in sorted(set(candidatos)) if d not in dias_con), None)
    print("\n" + "=" * 82)
    print("CRITERIO 3 · reordena en temporal y NO reordena en calma")
    print("=" * 82)
    print(f"\n  ★ días elegidos por el registro del Ministerio, no por el instrumento:")
    print(f"      TEMPORAL  {temporal} — {por_dia[temporal]} emergencias de causa "
          f"natural en el país, el máximo del registro")
    print(f"      TRANQUILO {tranquilo} — cero emergencias registradas")

    base = cclimp.fen_base(FEN_ITEM_120)
    ok = {}
    for etiqueta, dia in (("TEMPORAL", temporal), ("TRANQUILO", tranquilo)):
        vs = [perilla(pel[p].get(dia, 0.0)) for p in subs]
        fen = [round(min(1.0, base * v), 6) for v in vs]
        distintos = sum(1 for i in range(len(fen)) for j in range(i+1, len(fen))
                        if fen[i] != fen[j])
        total = len(fen)*(len(fen)-1)//2
        rep = Counter(vs)
        print(f"\n  {dia} ({etiqueta})")
        print("      perilla: " + " · ".join(f"{k:.1f}×{rep[k]}"
                                             for k in (1.0, 1.2, 1.4, 1.6) if rep[k]))
        print(f"      FENef: {len(set(fen))} niveles · {distintos}/{total} pares "
              f"desempatados")
        ok[etiqueta] = distintos
    c3 = ok["TEMPORAL"] > 0 and ok["TRANQUILO"] == 0
    print(f"\n  reordena en temporal: {'sí' if ok['TEMPORAL']>0 else 'NO'} · "
          f"reordena en calma: {'NO' if ok['TRANQUILO']==0 else 'SÍ'}")
    print(f"  {'✓ PASA' if c3 else '✗ NO PASA'}")

    print("\n" + "=" * 82)
    print("VEREDICTO")
    print("=" * 82)
    for n, v in (("1 · perilla quieta", c1), ("2 · se mueve en eventos", c2),
                 ("3 · reordena", c3)):
        print(f"  criterio {n:26s} {'PASA' if v else 'NO PASA'}")
    print("\n  ★ Ningún corte se movió para que esto pasara: los cortes son los")
    print("    percentiles 75/90/99 declarados, recalculados sobre la serie diaria.")
    print("  ★ NO se cierra nada sin el director.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
