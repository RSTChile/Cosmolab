"""
UMBRALES · convertir «van a caer 60 mm» en «esto es lo que se rompe»
======================================================================

★ LA PREGUNTA QUE ESTE ARTEFACTO CONTESTA
-------------------------------------------
Textual de Alexis (23-ago-2026): «en caso de lluvia debería mostrar qué calles
específicas entre qué manzanas específicas se inundarán si caen tantos mm en
tanto tiempo… eso sí le sirve [al alcalde].»

El resto de la aplicación dice DÓNDE hay infraestructura y CUÁNTA lluvia viene.
Esta pieza es la que une las dos: con cuánta agua cede cada cosa, y con qué
frecuencia cedió cuando llovió así.

★★ TRAE ALGO QUE NINGÚN ORGANISMO PUBLICA: EL DENOMINADOR
-----------------------------------------------------------
El Ministerio de Obras Públicas previó exactamente esta pregunta —su capa de
puntos de inundación tiene una columna `PRECI` para la precipitación asociada— y
la dejó **en −99 en los 429 registros, sin una sola excepción**.

Lo que sí existe es el temporal del 16-jul al 2-ago de 2026: 1.289 tramos de vía
cortados, cada uno con rol vial, kilómetro y fecha. Como se conocen TODOS los días
del periodo en esas celdas, se puede contar no sólo los días en que hubo corte
sino los días en que llovió lo mismo y NO pasó nada. Eso convierte «con esta
lluvia falló» en una tasa de verdad:

    más de 100 mm en 72 h  →  se cortó una vía en ~1 de cada 4 días-celda
    menos de 50 mm         →  ~1 de cada 20

⚠️⚠️ EL PUNTO MÁS DELICADO DE TODO ESTE ARCHIVO
------------------------------------------------
**Los umbrales se miden con una serie y se aplican a otra.** El pronóstico viene
de Open-Meteo; los umbrales se midieron con ERA5-Land, que sobre las mismas
celdas mide **~19 % más lluvia** (Coquimbo, julio 2026: 408,7 mm contra 343,6).

Aplicar un umbral de ERA5-Land a un pronóstico de Open-Meteo **subestimaría el
riesgo de forma sistemática y silenciosa**: el pronóstico diría 84 mm donde el
umbral espera 100 y la app mostraría una franja de peligro más baja de la que
corresponde. Nadie lo notaría, porque los dos números son plausibles.

Por eso aquí se calculan **las dos versiones**, cada una con su serie, y la
aplicación usa la de Open-Meteo — la que habla el mismo idioma que el pronóstico.
La de ERA5-Land queda publicada al lado para poder comparar.

USO
---
    ../../.venv-esa/bin/python construir/umbrales.py
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
DATOS = AQUI.parent / "publico" / "datos"
SALIDA = DATOS / "umbrales.json"

sys.path.insert(0, str(RAIZ))

# Las franjas en que se corta la lluvia acumulada de 72 h. No son redondas por
# gusto: el salto medido está en los 50 mm y se dispara sobre los 100.
FRANJAS = [(0, 1), (1, 10), (10, 25), (25, 50), (50, 100), (100, 10**9)]


def tasa_de_corte(serie_path):
    """La tabla con denominador: días-celda con corte contra días-celda sin él.

    Devuelve también las medianas de una y otra, que son el contraste que el
    test de permutación confirmó con p = 0,0001.
    """
    import umbral_por_tramo as U

    U.SERIE = serie_path
    tramos = U.cargar_tramos()
    serie = U.cargar_serie({t["celda"] for t in tramos})
    cruz = [t for t in tramos
            if t["celda"] in serie
            and U.acumulado(serie[t["celda"]], t["fecha"], 3) is not None]
    if not cruz:
        return None

    ini = min(t["fecha"] for t in cruz)
    fin = max(t["fecha"] for t in cruz)
    con_corte = {(t["celda"], t["fecha"].isoformat()) for t in cruz}
    celdas = {t["celda"] for t in cruz}
    con, sin = [], []
    d = ini
    while d <= fin:
        for c in celdas:
            v = U.acumulado(serie[c], d, 3)
            if v is None:
                continue
            (con if (c, d.isoformat()) in con_corte else sin).append(v)
        d += timedelta(days=1)
    con.sort()
    sin.sort()

    filas = []
    for lo, hi in FRANJAS:
        nc = sum(1 for v in con if lo <= v < hi)
        ns = sum(1 for v in sin if lo <= v < hi)
        if nc + ns == 0:
            continue
        filas.append({"desde": lo, "hasta": None if hi > 10**8 else hi,
                      "dias_celda": nc + ns, "con_corte": nc,
                      "tasa": round(nc / (nc + ns), 4)})
    med = lambda s: s[len(s) // 2]
    return {
        "tramos": len(cruz), "tramos_totales": len(tramos),
        "dias_celda_con_corte": len(con), "dias_celda_sin_corte": len(sin),
        "mediana_con_corte": round(med(con), 1),
        "mediana_sin_corte": round(med(sin), 1),
        "ventana": [ini.isoformat(), fin.isoformat()],
        "franjas": filas,
    }


def leer_csv(nombre, clave):
    p = RAIZ / "datos" / nombre
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    for f in filas:
        for k, v in list(f.items()):
            if k != clave:
                try:
                    f[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return filas


def construir():
    DATOS.mkdir(parents=True, exist_ok=True)
    om = RAIZ / "datos" / "clima_diario_celdas.csv"
    e5 = RAIZ / "datos" / "clima_diario_celdas_era5land.csv"

    print("  midiendo la tasa de corte en cada serie…", flush=True)
    tasas = {}
    if om.exists():
        tasas["openmeteo"] = tasa_de_corte(om)
        print(f"    Open-Meteo : {tasas['openmeteo']['tramos']} tramos", flush=True)
    if e5.exists():
        tasas["era5land"] = tasa_de_corte(e5)
        print(f"    ERA5-Land  : {tasas['era5land']['tramos']} tramos", flush=True)

    doc = {
        # ★ Cuál manda para la aplicación, y por qué. Declarado en el propio dato
        #   para que nadie tenga que adivinarlo leyendo el código.
        "serie_para_pronostico": "openmeteo",
        "por_que": ("El pronóstico viene de Open-Meteo. ERA5-Land mide ~19 % más "
                    "sobre las mismas celdas, así que aplicar sus umbrales a un "
                    "pronóstico de Open-Meteo subestimaría el riesgo."),
        "ventana_acumulacion_horas": 72,
        "tasa_de_corte": tasas,
        "por_elemento": leer_csv("umbral_por_elemento.csv", "elemento"),
        "por_proceso": leer_csv("umbral_por_proceso.csv", "proceso"),
        "advertencias": [
            "Las filas del catastro son LUGARES, no observaciones independientes: "
            "612 filas de CIGIDEN provienen de 173 tormentas distintas.",
            "La fecha que publica el MOP es la del INFORME, no la del corte: los "
            "tramos que figuran sin lluvia tienen su golpe 3-9 días antes, con "
            "mediana de 4 días.",
            "La gravedad NO sube con la lluvia (Grave 86,8 mm pide menos que Leve "
            "99,9 mm): lo que decide el daño es el estado del activo, no cuánta "
            "agua cae.",
            "El umbral por proceso viene de la cuenca del Maipo y no se "
            "extrapola al norte árido.",
        ],
    }
    SALIDA.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    t = tasas.get("openmeteo") or tasas.get("era5land")
    if t:
        print(f"\n  tasa de corte ({doc['serie_para_pronostico']}):")
        for f in t["franjas"]:
            et = f"{f['desde']:g}–{f['hasta']:g} mm" if f["hasta"] else f"> {f['desde']:g} mm"
            print(f"     {et:<14}{f['dias_celda']:>7} días-celda"
                  f"{100*f['tasa']:>8.1f} %")
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e3:.0f} KB)")
    return {"franjas": len(t["franjas"]) if t else 0,
            "elementos": len(doc["por_elemento"]),
            "procesos": len(doc["por_proceso"])}


if __name__ == "__main__":
    print("=" * 70)
    print("UMBRALES · de milímetros a consecuencias")
    print("=" * 70)
    sys.exit(0 if construir() else 1)
