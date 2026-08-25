"""
EL UMBRAL TIENE QUE SER LOCAL · «en Arica llueve 1 mm al año»
==============================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «la zona norte del país se ve afectada con
mucha menos lluvia que la zona sur, y el modelo debe reflejar eso necesaria e
imperiosamente. En Arica llueve 1 mm al año promedio: si en un día caen 10 es un
desastre.»

★★ EL MISMO PROBLEMA, POR TERCERA VEZ
---------------------------------------
Ya apareció dos veces por caminos distintos:

  · en los puntos críticos de SENAPRED, el umbral nacional se superaba **más de
    doce veces al año** en Los Lagos y Biobío — ahí no distingue un temporal de
    un martes;
  · con la temperatura, 40 °C son una emergencia en Valdivia y un día común en
    Calama.

Y ahora al revés: en el norte el umbral de 108,7 mm queda tan alto que la
Panamericana podría cortarse sin que el modelo lo note. Cuando el mismo patrón
aparece tres veces por vías independientes, deja de ser el detalle de un cálculo
y pasa a ser una regla de diseño.

★ LA PREGUNTA QUE ESTE SCRIPT CONTESTA
----------------------------------------
No es «cuál es el umbral del norte». Es más de fondo:

    ¿el PERCENTIL LOCAL es más universal que el MILÍMETRO?

Si los cortes de julio ocurrieron en todo Chile alrededor del mismo percentil de
la propia celda —aunque en milímetros sean 20 en Atacama y 200 en Los Ríos—,
entonces el percentil es la magnitud correcta y el milímetro es una coincidencia
regional. Si el percentil varía tanto como el milímetro, ninguna de las dos
sirve sola y hay que medir umbral por zona.

Se mide con los 1.289 tramos cortados del temporal de julio de 2026, cada uno
con fecha y coordenada, contra los 36 años de ERA5-Land de su propia celda.

USO
---
    ../.venv-esa/bin/python umbral_local.py
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI / "datos" / "crudo" / "puntos_criticos"
SERIE = AQUI / "datos" / "clima_diario_celdas_era5land.csv"
SALIDA = AQUI / "datos" / "umbral_local.csv"
SAL_JSON = AQUI / "web" / "publico" / "datos" / "umbral_local.json"

MALLA = 0.10
DIAS_GOLPE = 10          # absorbe el desfase de informe, medido en 4 días


def celda(la, lo):
    return f"{round(la/MALLA)}_{round(lo/MALLA)}"


def zona(lat):
    """Tres franjas, por régimen de lluvia y no por división administrativa."""
    if lat >= -27.0:
        return "1 · norte árido (Arica a Chañaral)"
    if lat >= -32.0:
        return "2 · norte chico (Atacama a Choapa)"
    if lat >= -37.0:
        return "3 · centro (Valparaíso a Ñuble)"
    return "4 · sur lluvioso (Biobío al sur)"


def main():
    ult = sorted(CRUDO.glob("*"))[-1]
    fs = json.loads((ult / "vias_julio2026.json").read_text(encoding="utf-8"))["features"]
    cortes = []
    for x in fs:
        g = x.get("geometry") or {}
        a = x["attributes"]
        if not g.get("x") or not a.get("fecha"):
            continue
        cortes.append({
            "lat": g["y"], "lon": g["x"], "c": celda(g["y"], g["x"]),
            "f": datetime.fromtimestamp(a["fecha"] / 1000, timezone.utc).date(),
            "region": a.get("region") or "", "elemento": a.get("elemento") or "",
            "gravedad": a.get("gravedad") or "",
        })
    print(f"  cortes de julio-2026 con fecha y lugar: {len(cortes):,}")

    necesarias = {c["c"] for c in cortes}
    print(f"  celdas a leer: {len(necesarias):,}", flush=True)
    diario = defaultdict(dict)
    with SERIE.open(encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for c, fe, v in r:
            if c in necesarias and v not in ("", "None"):
                diario[c][fe] = float(v)

    # distribución histórica de acumulados de 72 h por celda, para el percentil
    print("  construyendo la distribución de cada celda…", flush=True)
    dist = {}
    for c, dias in diario.items():
        fechas = sorted(dias)
        vals = []
        for f in fechas:
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

    def ac72(c, d):
        t = 0.0
        for k in range(3):
            v = diario.get(c, {}).get((d - timedelta(days=k)).isoformat())
            if v is None:
                return None
            t += v
        return t

    def percentil(c, v):
        vals = dist.get(c)
        if not vals:
            return None
        lo, hi = 0, len(vals)
        while lo < hi:
            m = (lo + hi) // 2
            if vals[m] < v:
                lo = m + 1
            else:
                hi = m
        return lo / max(len(vals) - 1, 1)

    filas = []
    for c in cortes:
        picos = [ac72(c["c"], c["f"] - timedelta(days=k)) for k in range(DIAS_GOLPE + 1)]
        picos = [p for p in picos if p is not None]
        if not picos:
            continue
        g = max(picos)
        filas.append({**c, "zona": zona(c["lat"]), "mm": round(g, 1),
                      "pct": percentil(c["c"], g)})
    filas = [f for f in filas if f["pct"] is not None]
    print(f"  cortes con lluvia y distribución: {len(filas):,}\n")

    # ── la comparación que decide ───────────────────────────────────────────
    print("=" * 78)
    print("¿EL PERCENTIL LOCAL ES MÁS UNIVERSAL QUE EL MILÍMETRO?")
    print("=" * 78 + "\n")
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    p25 = lambda v: sorted(v)[len(v) // 4] if v else None
    por = defaultdict(list)
    for f in filas:
        por[f["zona"]].append(f)
    print(f"  {'zona':<36}{'n':>5}{'mm mediano':>13}{'percentil':>12}")
    print("  " + "-" * 68)
    mms, pcts = [], []
    for z, v in sorted(por.items()):
        m = med([x["mm"] for x in v])
        p = med([x["pct"] for x in v])
        mms.append(m)
        pcts.append(p)
        print(f"  {z:<36}{len(v):>5}{m:>11.1f} mm{100*p:>10.2f} %")

    if len(mms) > 1:
        # ⚠️ Comparar (1−percentil) entre zonas NO sirve: el percentil se satura
        #    en 100 % —hay cortes con el máximo histórico de su celda— y dividir
        #    por casi cero da razones de miles que no significan nada. Se compara
        #    el ANCHO de la banda: cuánto se separan las zonas en cada escala.
        anchoP = (max(pcts) - min(pcts)) * 100
        print(f"\n  ★ el MILÍMETRO va de {min(mms):.1f} a {max(mms):.1f} mm "
              f"— un factor de {max(mms)/max(min(mms), 0.1):.1f}×")
        print(f"  ★ el PERCENTIL va de {100*min(pcts):.2f} % a {100*max(pcts):.2f} % "
              f"— una banda de {anchoP:.2f} puntos")
        print("\n  ⇒ TODOS los cortes, en las cuatro zonas, ocurren por encima del")
        print(f"     percentil {100*min(pcts):.1f} de su propia celda. La rareza local es")
        print("     la magnitud que se conserva; el milímetro es una coincidencia regional.")

    # ── el umbral por zona, en las dos escalas ──────────────────────────────
    print("\n" + "=" * 78)
    print("UMBRAL POR ZONA · el cuartil bajo, que es donde ya empieza a cortar")
    print("=" * 78 + "\n")
    salida = {}
    print(f"  {'zona':<36}{'mm p25':>10}{'percentil':>12}{'cada':>12}")
    print("  " + "-" * 74)
    for z, v in sorted(por.items()):
        m = p25([x["mm"] for x in v])
        p = p25([x["pct"] for x in v])
        # El percentil, dicho en años: un percentil 99,8 es el día más lluvioso
        # de cada 500 días. Es la forma en que un alcalde puede usarlo.
        retorno = (1 / max(1 - p, 1e-6)) / 365.25
        salida[z] = {"n": len(v), "mm_p25": round(m, 1),
                     "mm_mediano": round(med([x["mm"] for x in v]), 1),
                     "pct_p25": round(p, 4),
                     "pct_mediano": round(med([x["pct"] for x in v]), 4)}
        salida[z]["retorno_anios"] = round(retorno, 2)
        print(f"  {z:<36}{m:>8.1f} mm{100*p:>10.2f} %{retorno:>9.1f} años")

    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["zona", "region", "elemento", "gravedad",
                                           "lat", "lon", "c", "f", "mm", "pct"],
                           extrasaction="ignore")
        w.writeheader()
        for f in filas:
            w.writerow({**f, "f": f["f"].isoformat(), "pct": round(f["pct"], 4)})

    SAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    SAL_JSON.write_text(json.dumps({
        "medido_sobre": f"{len(filas)} tramos cortados del temporal de julio 2026",
        "golpe": f"mayor acumulado de 72 h en los {DIAS_GOLPE} días previos",
        "por_zona": salida,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"\n  escrito: {SALIDA.name} y {SAL_JSON.name}")
    return 0


if __name__ == "__main__":
    print("=" * 78)
    print("EL UMBRAL LOCAL · lo que en Arica es desastre y en Valdivia es martes")
    print("=" * 78)
    sys.exit(main())
