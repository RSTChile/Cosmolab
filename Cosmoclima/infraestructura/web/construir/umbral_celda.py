"""
EL UMBRAL, TRADUCIDO A CADA LUGAR
===================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «la zona norte del país se ve afectada con
mucha menos lluvia que la zona sur, y el modelo debe reflejar eso necesaria e
imperiosamente. En Arica llueve 1 mm al año promedio: si en un día caen 10 es un
desastre.»

★★ LA IDEA, EN UNA LÍNEA
--------------------------
Un umbral medido a escala nacional —«la carpeta de rodadura cede con 108,7 mm en
72 h»— no se puede aplicar tal cual en Arica ni en Valdivia. Pero **el percentil
que ese umbral ocupa sí viaja**: medido sobre los 1.241 tramos cortados de julio,
los cortes ocurrieron en las cuatro zonas del país por encima del percentil 99,5
de su propia celda, dentro de una banda de 0,46 puntos, mientras los milímetros
variaban 4,1 veces.

Así que aquí se hacen dos cosas:

  1. cada umbral de elemento se traduce al **percentil que ocupa** en las celdas
     donde fue medido;
  2. ese percentil se convierte de vuelta a **milímetros en cada celda del país**.

Resultado: la carretera sigue cediendo con 108,7 mm en la zona donde se midió, y
con muchos menos en el desierto — sin inventar nada, porque el percentil es el
mismo y lo que cambia es la distribución de cada lugar, que está medida.

⚠️ NO ES UN AJUSTE COSMÉTICO. Cambia qué se considera afectado: en el norte
árido baja el listón y en el sur lo sube. Por eso se publican los dos números
—el nacional y el local— y la aplicación muestra ambos.

⚠️ Una celda sin días de lluvia suficientes no admite percentiles altos: si el
percentil 99,5 cae sobre valores de 0 mm, se declara sin umbral local en vez de
devolver cero, que dejaría todo «afectado» siempre.

USO
---
    ../../.venv-esa/bin/python construir/umbral_celda.py
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
SALIDA = DATOS / "umbral_celda.json"

# Percentiles que se guardan por celda. Son los que usan los umbrales medidos
# más los de referencia para leer el gráfico.
CORTES = [0.980, 0.990, 0.995, 0.998, 0.999, 1.0]
PISO_MM = 5.0     # por debajo de esto no hay «episodio» que valga la pena


def main():
    if not SERIE.exists():
        print("  falta la serie ERA5-Land")
        return 1

    print("  leyendo la serie…", flush=True)
    diario = defaultdict(dict)
    with SERIE.open(encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for c, f, v in r:
            if v not in ("", "None"):
                diario[c][f] = float(v)
    print(f"  celdas: {len(diario):,}")

    print("  acumulando 72 h y ordenando cada celda…", flush=True)
    dist = {}
    for c, dias in diario.items():
        vals = []
        for f in sorted(dias):
            d = date.fromisoformat(f)
            t, ok = 0.0, True
            for k in range(3):
                v = dias.get((d - timedelta(days=k)).isoformat())
                if v is None:
                    ok = False
                    break
                t += v
            if ok:
                vals.append(t)
        vals.sort()
        dist[c] = vals

    # ── el percentil que ocupa cada umbral medido ───────────────────────────
    af = json.loads((DATOS / "afectacion.json").read_text(encoding="utf-8"))
    medidos = {n: v for n, v in af["por_item"].items() if v["tipo"] == "medido"}

    # Se calcula sobre las celdas donde el umbral fue medido: las del temporal
    # de julio, que están en el centro-sur. Promediar sobre TODO Chile daría un
    # percentil inflado por el desierto, donde cualquier valor es extremo.
    ref = json.loads((DATOS / "celdas_por_comuna.json").read_text(encoding="utf-8"))
    terr = json.loads((DATOS / "territorios.json").read_text(encoding="utf-8"))
    lat_de = {}
    for cl, vals in dist.items():
        try:
            i, _ = cl.split("_")
            lat_de[cl] = int(i) * 0.1
        except ValueError:
            continue
    celdas_ref = [c for c, la in lat_de.items() if -40 <= la <= -32 and dist[c]]
    print(f"  celdas de referencia (donde se midió, lat −32 a −40): {len(celdas_ref):,}")

    def percentil_de(mm, celdas):
        """Qué percentil ocupa `mm` de media en esas celdas."""
        ps = []
        for c in celdas:
            v = dist[c]
            if not v:
                continue
            lo, hi = 0, len(v)
            while lo < hi:
                m = (lo + hi) // 2
                if v[m] < mm:
                    lo = m + 1
                else:
                    hi = m
            ps.append(lo / max(len(v) - 1, 1))
        ps.sort()
        return ps[len(ps) // 2] if ps else None

    print("\n  UMBRAL NACIONAL → PERCENTIL EQUIVALENTE")
    print(f"  {'elemento':<44}{'mm':>8}{'percentil':>12}")
    print("  " + "-" * 66)
    equivalencia = {}
    for n, v in sorted(medidos.items(), key=lambda t: t[1]["umbral_mm_72h"]):
        p = percentil_de(v["umbral_mm_72h"], celdas_ref)
        equivalencia[n] = round(p, 5)
        print(f"  {v['elemento'][:43]:<44}{v['umbral_mm_72h']:>7.1f}{100*p:>11.3f} %")

    # ── de vuelta a milímetros, celda por celda ─────────────────────────────
    def valor_en(vals, p):
        if not vals:
            return None
        return vals[min(len(vals) - 1, int(p * (len(vals) - 1)))]

    por_celda = {}
    sin_umbral = 0
    for c, vals in dist.items():
        fila = {}
        for corte in CORTES:
            v = valor_en(vals, corte)
            if v is not None:
                fila[f"{corte:.3f}"] = round(v, 1)
        # ⚠️ Si el percentil 99,5 de la celda no llega al piso, esa celda no
        #    tiene episodios: declararla sin umbral es más honesto que poner 0.
        if (fila.get("0.995") or 0) < PISO_MM:
            sin_umbral += 1
            por_celda[c] = None
            continue
        por_celda[c] = fila

    con = sum(1 for v in por_celda.values() if v)
    print(f"\n  celdas con umbral local: {con:,} · sin episodios: {sin_umbral:,}")

    # ── qué tan distinto sale, por zona ─────────────────────────────────────
    print("\n  LO QUE CAMBIA · umbral de carpeta de rodadura (nacional 108,7 mm)")
    p616 = equivalencia.get("616")
    zonas = [("norte árido", -18, -27), ("norte chico", -27, -32),
             ("centro", -32, -37), ("sur", -37, -44)]
    print(f"  {'zona':<20}{'celdas':>8}{'umbral local mediano':>24}")
    print("  " + "-" * 54)
    for nom, a, b in zonas:
        vs = []
        for c, fila in por_celda.items():
            la = lat_de.get(c)
            if la is None or not (b <= la <= a) or not fila:
                continue
            v = valor_en(dist[c], p616)
            if v is not None:
                vs.append(v)
        if vs:
            vs.sort()
            print(f"  {nom:<20}{len(vs):>8}{vs[len(vs)//2]:>21.1f} mm")

    SALIDA.write_text(json.dumps({
        "explicacion": ("Cada umbral medido se traduce al percentil que ocupa en "
                        "las celdas donde fue medido (lat −32 a −40), y ese "
                        "percentil se convierte de vuelta a milímetros en cada "
                        "celda del país."),
        "por_que": ("Medido sobre 1.241 tramos cortados en julio 2026: los cortes "
                    "ocurren en las cuatro zonas por encima del percentil 99,5 de "
                    "su propia celda (banda de 0,46 puntos) mientras los "
                    "milímetros varían 4,1 veces. La rareza local viaja; el "
                    "milímetro no."),
        "piso_mm": PISO_MM,
        "percentil_por_item": equivalencia,
        "por_celda": por_celda,
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    print("=" * 76)
    print("EL UMBRAL, TRADUCIDO A CADA LUGAR")
    print("=" * 76)
    sys.exit(main())
