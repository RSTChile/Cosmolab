"""
LOS 15.799 PUNTOS CRÍTICOS · con cuánta lluvia se activa cada uno
==================================================================

SENAPRED publica 15.799 puntos críticos a resolución de calle, con su causa, su
comuna y su nivel de riesgo. Lo que NO publica —ni nadie— es con cuánta lluvia se
activa cada uno. Este script se lo asigna, y mide con qué frecuencia ocurre.

★★ DOS COSAS MUY DISTINTAS, Y CONVIENE NO CONFUNDIRLAS
--------------------------------------------------------
  1. **El umbral es POR ANALOGÍA, no medido.** El catastro de SENAPRED no trae
     fecha de activación: es un inventario de puntos que *pueden* fallar, no un
     registro de fallas. Sin fechas no hay nada que cruzar contra el clima, así
     que a cada causa se le asigna el umbral del proceso equivalente que SÍ se
     midió (CIGIDEN y las vías del MOP). Es una transferencia, y como toda
     transferencia puede estar mal.

  2. **La frecuencia SÍ es medida.** Una vez asignado el umbral, contar cuántos
     días de los 36 años de ERA5-Land lo superaron en la celda de ese punto es
     medición directa sobre 45 millones de filas. Eso convierte «se activa con
     34 mm» en «eso pasa cada tantos años aquí», que es la pregunta útil.

★ POR QUÉ HAY CAUSAS SIN UMBRAL, Y POR QUÉ ESO ES LO CORRECTO
---------------------------------------------------------------
Tres causas no tienen análogo medido y se dejan **explícitamente sin umbral** en
vez de inventarles uno:

  · «Acumulación de nieve» y «Congelamiento de caminos» no dependen de milímetros
    de lluvia líquida sino de temperatura. Aplicarles un umbral de precipitación
    sería un número con apariencia de rigor y sin ningún contenido.
  · «Subsidencia/Licuefacción/Socavamiento/Erosión» y «Daño en infraestructura»
    son categorías demasiado heterogéneas: mezclan procesos con umbrales que
    difieren en un orden de magnitud.

Son 1.705 de 15.799 puntos (10,8 %). Declararlos vacíos deja ver el hueco; darles
un número lo taparía.

⚠️ LA ESCALA ES LA DE OPEN-METEO, no la de ERA5-Land. Los umbrales heredados de
CIGIDEN y del MOP se midieron con la serie de Open-Meteo, que mide ~19 % menos
que ERA5-Land. La frecuencia se cuenta con ERA5-Land porque es la serie completa,
así que el umbral se **reescala** antes de contar. Sin ese ajuste la frecuencia
saldría inflada de forma sistemática y silenciosa.

USO
---
    ../.venv-esa/bin/python umbral_puntos_criticos.py
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI / "datos" / "crudo" / "puntos_criticos"
SERIE = AQUI / "datos" / "clima_diario_celdas_era5land.csv"
SALIDA = AQUI / "datos" / "umbral_puntos_criticos.csv"
RESUMEN = AQUI / "datos" / "umbral_puntos_criticos_resumen.csv"

MALLA = 0.10
# ★ ERA5-Land midió 408,7 mm donde Open-Meteo midió 343,6 en la misma celda y mes.
#   Los umbrales vienen en escala Open-Meteo; para contarlos contra ERA5-Land hay
#   que subirlos en la misma proporción.
FACTOR_E5 = 408.7 / 343.6

# causa de SENAPRED → (umbral mm/72h en escala Open-Meteo, de dónde sale, confianza)
MAPEO = {
    "Inundación por desborde de cauce": (
        94.7, "CIGIDEN · desborde de río (9 casos, 7 tormentas)", "baja"),
    "Anegamiento de caminos/pasos a desnivel": (
        34.2, "CIGIDEN · anegamiento (266 casos, 113 tormentas)", "alta"),
    "Colapso colectores de aguas lluvia/alcantarillados": (
        145.9, "MOP · elementos de saneamiento (18 tramos)", "baja"),
    "Deslizamiento/Derrumbe/Rodado/Caída": (
        31.8, "CIGIDEN · flujo (290 lugares, 38 tormentas)", "media"),
    "Interrupción de caminos": (
        108.7, "MOP · carpeta de rodadura (570 tramos)", "alta"),
    "Activación de quebradas": (
        31.8, "CIGIDEN · flujo (290 lugares, 38 tormentas)", "media"),
    "Flujos de barro/detritos (Aluvión)": (
        31.8, "CIGIDEN · flujo (290 lugares, 38 tormentas)", "media"),
    # ── sin umbral, a propósito ──────────────────────────────────────────────
    "Acumulación de nieve": (None, "no depende de lluvia líquida", None),
    "Congelamiento de caminos": (None, "no depende de lluvia líquida", None),
    "Subsidencia/Licuefacción/Socavamiento/Erosión": (
        None, "categoría heterogénea, sin análogo medido", None),
    "Daño/pérdida en infraestructura (Ej.: Vial, Portuario/Costero, Agrícola)": (
        None, "categoría heterogénea, sin análogo medido", None),
}


def celda(lat, lon):
    return f"{round(lat/MALLA)}_{round(lon/MALLA)}"


def cargar_puntos():
    ult = sorted(CRUDO.glob("*"))[-1]
    fs = json.loads((ult / "pc_2026.json").read_text(encoding="utf-8"))["features"]
    out = []
    for x in fs:
        p = x["properties"]
        g = (x.get("geometry") or {}).get("coordinates")
        if not g:
            continue
        out.append({
            "region": p.get("Región") or "", "comuna": p.get("Comuna") or "",
            "sector": (p.get("Sector") or "")[:70],
            "causa": (p.get("Causa_del_") or "").strip(),
            "nivel_senapred": p.get("Nivel_de_R") or "",
            "responsable": (p.get("Si_la_resp") or "")[:60],
            "lat": g[1], "lon": g[0], "celda": celda(g[1], g[0]),
        })
    return out


def acumulados_por_celda(celdas_pedidas):
    """{celda: {fecha: mm en 72 h}} — se calcula una vez por celda, no por punto."""
    diario = defaultdict(dict)
    with SERIE.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["celda"] in celdas_pedidas and r["precip_mm"] not in ("", "None"):
                diario[r["celda"]][r["fecha"]] = float(r["precip_mm"])
    out = {}
    for c, dias in diario.items():
        fechas = sorted(dias)
        acum = {}
        for f in fechas:
            d = date.fromisoformat(f)
            v = 0.0
            completo = True
            for k in range(3):
                x = dias.get((d - timedelta(days=k)).isoformat())
                if x is None:
                    completo = False
                    break
                v += x
            if completo:
                acum[f] = v
        out[c] = acum
    return out


def main():
    if not SERIE.exists():
        print("falta la serie ERA5-Land — corre convertir_era5land_a_serie.py")
        return 1

    puntos = cargar_puntos()
    print("=" * 78)
    print("LOS PUNTOS CRÍTICOS DE SENAPRED · con cuánta lluvia se activa cada uno")
    print("=" * 78)
    print(f"\n  puntos del catastro : {len(puntos):,}")

    sin_mapa = {p["causa"] for p in puntos} - set(MAPEO)
    if sin_mapa:
        print(f"  ⚠️ causas sin entrada en el mapeo: {sin_mapa}")

    con_umbral = [p for p in puntos if MAPEO.get(p["causa"], (None,))[0]]
    print(f"  con umbral asignado : {len(con_umbral):,} "
          f"({100*len(con_umbral)/len(puntos):.1f} %)")
    print(f"  sin umbral (a propósito): {len(puntos)-len(con_umbral):,}")

    print("\n  cargando la serie ERA5-Land (45 M de filas)…", flush=True)
    necesarias = {p["celda"] for p in con_umbral}
    acum = acumulados_por_celda(necesarias)
    print(f"  celdas pedidas {len(necesarias):,} · con serie {len(acum):,}")

    # ── el conteo: cuántas veces se superó el umbral en 36 años ──────────────
    cache = {}
    for p in puntos:
        u, origen, conf = MAPEO.get(p["causa"], (None, "causa desconocida", None))
        p["umbral_mm_72h"] = u
        p["origen_umbral"] = origen
        p["confianza"] = conf or ""
        p["excedencias"] = p["anios_serie"] = p["por_anio"] = p["retorno_anios"] = None
        p["percentil_local"] = None
        p["umbral_util"] = ""
        if u is None or p["celda"] not in acum:
            continue
        clave = (p["celda"], u)
        if clave not in cache:
            serie = acum[p["celda"]]
            corte = u * FACTOR_E5
            vals = sorted(serie.values())
            n = sum(1 for v in vals if v >= corte)
            anios = len(vals) / 365.25
            # ★★ EL PERCENTIL LOCAL — sin esto el umbral miente en el sur.
            #    Un umbral absoluto no significa lo mismo en todas partes: en
            #    Puyuhuapi 34 mm en 72 h es un martes cualquiera y en Copiapó es
            #    un aluvión. El percentil dice si ese umbral es RARO en ESTE
            #    lugar, que es lo que decide si sirve para avisar de algo.
            pct = 1.0 - (n / len(vals)) if vals else None
            cache[clave] = (n, anios, pct)
        n, anios, pct = cache[clave]
        p["percentil_local"] = round(pct, 4) if pct is not None else None
        # Si el umbral se supera más de 12 veces al año, no distingue nada aquí.
        p["umbral_util"] = "no" if (anios and n / anios > 12) else "si"
        p["excedencias"] = n
        p["anios_serie"] = round(anios, 1)
        p["por_anio"] = round(n / anios, 2) if anios else None
        # ★ Periodo de retorno de EPISODIOS, no de días: días consecutivos sobre
        #   el umbral son el mismo temporal. Se divide por 3 (la ventana) para no
        #   contar tres veces la misma tormenta.
        eps = n / 3.0
        p["retorno_anios"] = round(anios / eps, 2) if eps >= 1 else None

    medidos = [p for p in puntos if p["excedencias"] is not None]
    print(f"  puntos con frecuencia medida: {len(medidos):,}\n")

    # ── resumen por causa ────────────────────────────────────────────────────
    print("=" * 78)
    print("UMBRAL POR CAUSA · y cada cuánto se supera")
    print("=" * 78 + "\n")
    print(f"  {'causa':<46}{'n':>6}{'mm/72h':>8}{'retorno':>10}")
    print("  " + "-" * 72)
    filas = []
    porc = defaultdict(list)
    for p in puntos:
        porc[p["causa"]].append(p)
    for causa, v in sorted(porc.items(), key=lambda t: -len(t[1])):
        u = MAPEO.get(causa, (None,))[0]
        rets = sorted(x["retorno_anios"] for x in v if x["retorno_anios"])
        ret = f"{rets[len(rets)//2]:.1f} años" if rets else "—"
        print(f"  {causa[:46]:<46}{len(v):>6}"
              f"{(f'{u:.1f}' if u else '—'):>8}{ret:>10}")
        filas.append({
            "causa": causa, "puntos": len(v),
            "umbral_mm_72h": u if u else "",
            "origen": MAPEO.get(causa, (None, "", None))[1],
            "confianza": MAPEO.get(causa, (None, None, ""))[2] or "",
            "retorno_mediano_anios": rets[len(rets) // 2] if rets else "",
            "con_frecuencia_medida": sum(1 for x in v if x["excedencias"] is not None),
        })

    # ── los que se activan más seguido ───────────────────────────────────────
    print("\n" + "=" * 78)
    print("LOS 12 PUNTOS QUE SE ACTIVAN MÁS SEGUIDO")
    print("=" * 78 + "\n")
    ranking = sorted((p for p in medidos if p["retorno_anios"]),
                     key=lambda p: p["retorno_anios"])[:12]
    print(f"  {'comuna':<18}{'sector':<34}{'mm':>7}{'cada':>9}")
    print("  " + "-" * 70)
    for p in ranking:
        print(f"  {p['comuna'][:17]:<18}{p['sector'][:33]:<34}"
              f"{p['umbral_mm_72h']:>7.0f}{p['retorno_anios']:>7.1f} a")

    campos = ["region", "comuna", "sector", "causa", "nivel_senapred", "responsable",
              "lat", "lon", "celda", "umbral_mm_72h", "origen_umbral", "confianza",
              "excedencias", "anios_serie", "por_anio", "retorno_anios",
              "percentil_local", "umbral_util"]
    with SALIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(puntos)
    with RESUMEN.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    # ── dónde el umbral nacional NO sirve ────────────────────────────────────
    inutil = [x for x in medidos if x["umbral_util"] == "no"]
    print("\n" + "=" * 78)
    print("DÓNDE ESTE UMBRAL NO SIRVE · y por qué había que mirarlo")
    print("=" * 78)
    print(f"\n  puntos donde el umbral se supera MÁS DE 12 VECES AL AÑO: "
          f"{len(inutil):,} de {len(medidos):,} ({100*len(inutil)/len(medidos):.1f} %)")
    print("  En esos lugares el umbral no distingue un temporal de un día normal.")
    if inutil:
        porreg = defaultdict(int)
        for x in inutil:
            porreg[x["region"]] += 1
        print("\n  concentrados en:")
        for r, k in sorted(porreg.items(), key=lambda t: -t[1])[:6]:
            tot = sum(1 for x in medidos if x["region"] == r)
            print(f"     {r[:30]:<32}{k:>5} de {tot:<6} ({100*k/tot:.0f} %)")
    print("\n  ★ La lectura correcta NO es el milímetro sino el PERCENTIL LOCAL:")
    print("    la columna `percentil_local` dice qué tan raro es ese umbral en esa")
    print("    celda. Un umbral en el percentil 0,99 avisa de algo; uno en 0,50 no.")

    print(f"\n  escrito: {SALIDA.name} ({len(puntos):,} puntos) y {RESUMEN.name}")
    print("\n  ⚠️ El umbral es TRANSFERIDO por analogía de causa; la frecuencia sí")
    print("     está medida sobre 36 años. No confundir una cosa con la otra.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
