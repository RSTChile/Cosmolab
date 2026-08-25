"""
EL UMBRAL DE LA REMOCIÓN EN MASA · quebradas, deslizamientos y flujos
======================================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «esta semana lloverá y te aseguro que
tendremos cortes de caminos, activación de quebradas, etc.»

Hasta ahora la aplicación heredaba para todo el umbral de la carpeta de rodadura,
porque los cortes de vía del MOP eran lo único medido. Pero una quebrada que se
activa no es una calzada que se socava: son procesos distintos y ceden con
lluvias distintas.

★ LA FUENTE QUE YA ESTABA BAJADA Y NO SE USABA
------------------------------------------------
`sernageomin/reterm_eventos.geojson` — el Registro de Remociones en Masa:
**380 eventos con fecha, tipo y detonante declarado**. De ellos **305 detonados
por lluvia** («Precipitaciones», «Lluvias intensas», «Precipitaciones intensas»)
y el resto por deshielo, sismo o intervención humana.

Los tipos son exactamente el fenómeno del que se habla: deslizamiento de suelo,
flujo de detritos, caída de rocas, deslizamiento rotacional.

★ SE MIDE IGUAL QUE LOS CORTES DE VÍA
---------------------------------------
Cada evento se cruza contra los 36 años de ERA5-Land de su propia celda: se toma
el golpe —el mayor acumulado de 72 h en los diez días previos, que absorbe el
desfase de registro— y su percentil local. Así el umbral queda en las dos
escalas y se puede comparar con el de carretera.

⚠️ SÓLO 29 DE LOS 380 EVENTOS ESTÁN AL NORTE DE LA LATITUD −32 (8 %). Para el
norte árido la muestra es escasa y el umbral que salga de ahí hay que tomarlo
como indicativo, no como medido. Se declara en la salida.

USO
---
    ../.venv-esa/bin/python umbral_remocion.py
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI / "datos" / "crudo" / "sernageomin"
SERIE = AQUI / "datos" / "clima_diario_celdas_era5land.csv"
CUENCA = AQUI / "datos" / "clima_diario_cuenca.csv"
SALIDA = AQUI / "datos" / "umbral_remocion.csv"
SAL_JSON = AQUI / "web" / "publico" / "datos" / "umbral_remocion.json"

MALLA = 0.10
DIAS_GOLPE = 10
LLUVIA = ("precipitac", "lluvia")


def celda(la, lo):
    return f"{round(la/MALLA)}_{round(lo/MALLA)}"


def zona(lat):
    if lat >= -27.0:
        return "norte árido"
    if lat >= -32.0:
        return "norte chico"
    if lat >= -37.0:
        return "centro"
    return "sur"


def main():
    dias = sorted(CRUDO.glob("*"))
    arch = None
    for d in reversed(dias):
        if (d / "reterm_eventos.geojson").exists():
            arch = d / "reterm_eventos.geojson"
            break
    if not arch:
        print("  falta reterm_eventos.geojson")
        return 1

    fs = json.loads(arch.read_text(encoding="utf-8"))["features"]
    ev = []
    for f in fs:
        p = f["properties"]
        g = (f.get("geometry") or {}).get("coordinates")
        det = (p.get("Detonante") or "").lower()
        if not g or not any(k in det for k in LLUVIA):
            continue
        t = p.get("Fecha_evento")
        if not t:
            continue
        try:
            fecha = datetime.fromtimestamp(int(t) / 1000, timezone.utc).date()
        except (TypeError, ValueError, OSError):
            continue
        if not (1990 <= fecha.year <= 2026):
            continue
        ev.append({
            "lat": g[1], "lon": g[0], "c": celda(g[1], g[0]), "f": fecha,
            "tipo": (p.get("Tipo") or "")[:44],
            "detonante": (p.get("Detonante") or "")[:30],
            "region": p.get("REGION") or "", "comuna": p.get("COMUNA") or "",
            "zona": zona(g[1]),
        })
    print(f"  eventos de ReTeRM: {len(fs)}")
    print(f"  detonados por lluvia, con fecha en 1990-2026: {len(ev)}")

    # ── la lluvia de cada uno ───────────────────────────────────────────────
    necesarias = {e["c"] for e in ev}
    diario = defaultdict(dict)
    for archivo in (SERIE, CUENCA):
        if not archivo.exists():
            continue
        with archivo.open(encoding="utf-8") as fh:
            r = csv.reader(fh)
            next(r)
            for c, fe, v in r:
                if c in necesarias and v not in ("", "None"):
                    diario[c][fe] = float(v)
    print(f"  celdas necesarias {len(necesarias)} · con serie {len(diario)}")

    dist = {}
    for c, dd in diario.items():
        vals = []
        for f in sorted(dd):
            d = date.fromisoformat(f)
            t, ok = 0.0, True
            for k in range(3):
                v = dd.get((d - timedelta(days=k)).isoformat())
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

    def pct(c, v):
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
    for e in ev:
        picos = [ac72(e["c"], e["f"] - timedelta(days=k)) for k in range(DIAS_GOLPE + 1)]
        picos = [p for p in picos if p is not None]
        if not picos:
            continue
        g = max(picos)
        p = pct(e["c"], g)
        if p is None:
            continue
        filas.append({**e, "mm": round(g, 1), "pct": round(p, 4)})
    print(f"  con lluvia calculable: {len(filas)}\n")

    if not filas:
        print("  sin casos suficientes")
        return 1

    med = lambda v: sorted(v)[len(v) // 2]
    p25 = lambda v: sorted(v)[len(v) // 4]

    print("=" * 74)
    print("CON CUÁNTA LLUVIA SE ACTIVA UNA REMOCIÓN EN MASA")
    print("=" * 74 + "\n")
    print(f"  {'zona':<16}{'n':>5}{'mm p25':>10}{'mm mediano':>13}{'percentil':>12}")
    print("  " + "-" * 58)
    por = defaultdict(list)
    for f in filas:
        por[f["zona"]].append(f)
    salida = {}
    orden = ["norte árido", "norte chico", "centro", "sur"]
    for z in orden:
        v = por.get(z)
        if not v:
            continue
        mm = [x["mm"] for x in v]
        pc = [x["pct"] for x in v]
        salida[z] = {"n": len(v), "mm_p25": round(p25(mm), 1),
                     "mm_mediano": round(med(mm), 1),
                     "pct_mediano": round(med(pc), 4),
                     "confianza": "baja" if len(v) < 20 else "media"}
        print(f"  {z:<16}{len(v):>5}{p25(mm):>8.1f} mm{med(mm):>11.1f} mm"
              f"{100*med(pc):>11.2f} %")

    todos_mm = [f["mm"] for f in filas]
    todos_pc = [f["pct"] for f in filas]
    print(f"\n  {'TODO EL PAÍS':<16}{len(filas):>5}{p25(todos_mm):>8.1f} mm"
          f"{med(todos_mm):>11.1f} mm{100*med(todos_pc):>11.2f} %")
    print(f"\n  ★ para comparar: una carpeta de rodadura cede con 108,7 mm "
          f"(percentil 99,62)")

    # por tipo de proceso
    print("\n  POR TIPO DE PROCESO (los que tienen 8 casos o más)")
    portipo = defaultdict(list)
    for f in filas:
        clave = f["tipo"].split("/")[0].strip().lower()
        clave = "flujo de detritos" if "flujo" in clave else clave
        clave = "deslizamiento" if "desliz" in clave else clave
        clave = "caída de rocas" if "caída" in clave or "caida" in clave else clave
        portipo[clave].append(f["mm"])
    for t, v in sorted(portipo.items(), key=lambda x: -len(x[1])):
        if len(v) >= 8:
            print(f"     {t[:34]:<36}{len(v):>4} casos · mediana {med(v):>6.1f} mm")

    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["zona", "region", "comuna", "tipo",
                                           "detonante", "lat", "lon", "c", "f",
                                           "mm", "pct"], extrasaction="ignore")
        w.writeheader()
        for f in filas:
            w.writerow({**f, "f": f["f"].isoformat()})
    SAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    SAL_JSON.write_text(json.dumps({
        "fuente": "ReTeRM · SERNAGEOMIN, eventos con detonante de lluvia",
        "medido_sobre": len(filas),
        "advertencia": ("Sólo 29 de los 380 eventos del registro están al norte "
                        "de la latitud −32. Para el norte árido el umbral es "
                        "indicativo, no medido."),
        "nacional_mm": round(med(todos_mm), 1),
        "nacional_pct": round(med(todos_pc), 4),
        "por_zona": salida,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"\n  escrito: {SALIDA.name} y {SAL_JSON.name}")
    return 0


if __name__ == "__main__":
    print("=" * 74)
    print("EL UMBRAL DE LA REMOCIÓN EN MASA")
    print("=" * 74)
    sys.exit(main())
