"""
Validación independiente: 376 remociones en masa que ocurrieron de verdad.

POR QUÉ ESTA PRUEBA Y NO OTRA VEZ COPIAPÓ
------------------------------------------
Copiapó sirvió para diagnosticar el problema, y por eso mismo dejó de servir
para validar la solución: probar una corrección contra el caso que la motivó es
examinarse con la prueba que uno escribió. Da igual cuán buena sea la idea.

ReTeRM es la salida limpia. Son **376 eventos reales** de remoción en masa que
SERNAGEOMIN registró en terreno entre 1996 y 2026 — deslizamientos, flujos de
detritos, caídas de roca— con fecha, comuna y detonante. **329 fueron detonados
por lluvia.** Ninguno de ellos participó en el diseño de la medida.

LA PREGUNTA
-----------
Si la medida corregida sirve, los meses en que ocurrió un deslizamiento tienen
que puntuar más alto que los meses en que no pasó nada **en el mismo lugar**.

LA MEDIDA PRINCIPAL
-------------------
Qué fracción de los eventos cae en el decil superior de peligro **de su propio
punto**. Si la medida no sirviera de nada, sería ~10% por puro azar. Se compara
contra eso, no contra una intuición.

Comparar dentro del mismo punto es deliberado: evita que el resultado se explique
por «los deslizamientos ocurren en el sur, que es donde llueve». Cada lugar se
compara consigo mismo.
"""

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI / "adaptadores"))
import era5  # noqa: E402

EVENTOS = AQUI / "datos" / "reterm_eventos.csv"
CLIMA = AQUI / "datos" / "clima_diario_reterm_era5.csv"
PUNTOS = AQUI / "datos" / "reterm_puntos.csv"
SEMILLA = 20260816
PERMUTACIONES = 1000


def es_lluvia(detonante):
    t = str(detonante or "").lower()
    return "luvia" in t or "recipitac" in t


def main():
    if not CLIMA.exists():
        print(f"Falta {CLIMA.name}: la descarga no terminó. No se corre la "
              f"prueba con datos parciales.")
        return 1

    # el adaptador apunta a las subestaciones; se lo redirige a los puntos ReTeRM
    era5.CSV_DIARIO = CLIMA
    era5.CSV_PUNTOS = PUNTOS
    obs, problema = era5.traer()
    if problema:
        print("SIN DATO:", problema)
        return 1

    # peligro por (punto, año, mes)
    peligro = {}
    for o in obs:
        if o["variable"] != "peligro_precipitacion":
            continue
        a, m = int(o["vigencia_inicio"][:4]), int(o["vigencia_inicio"][5:7])
        peligro[(o["territorio_id"], a, m)] = o["valor_normalizado"]

    # distribución de peligro de cada punto, para poder hablar de deciles propios
    por_punto = defaultdict(list)
    for (se, a, m), v in peligro.items():
        por_punto[se].append(v)
    for se in por_punto:
        por_punto[se].sort()

    with EVENTOS.open(encoding="utf-8") as fh:
        eventos = [e for e in csv.DictReader(fh)]
    lluvia = [e for e in eventos if es_lluvia(e["detonante"])]

    def decil_de(se, valor):
        """En qué decil de SU PROPIO punto cae ese valor (1 = más bajo)."""
        muestra = por_punto.get(se)
        if not muestra:
            return None
        debajo = sum(1 for x in muestra if x < valor)
        return debajo / len(muestra)

    emparejados, sin_dato = [], 0
    for e in lluvia:
        se = f"ReTeRM · {e['comuna']}"
        clave = (se, int(e["anio"]), int(e["mes"]))
        if clave not in peligro:
            sin_dato += 1
            continue
        v = peligro[clave]
        emparejados.append((se, int(e["anio"]), int(e["mes"]), v,
                            decil_de(se, v)))

    if len(emparejados) < 50:
        print(f"Sólo {len(emparejados)} eventos emparejados con clima. "
              f"Muestra insuficiente; no se afloja el criterio.")
        return 1

    print("=" * 74)
    print("VALIDACIÓN INDEPENDIENTE · 376 remociones en masa reales (ReTeRM)")
    print("=" * 74)
    print(f"\n  eventos totales:            {len(eventos)}")
    print(f"  detonados por lluvia:       {len(lluvia)}")
    print(f"  emparejados con clima:      {len(emparejados)}"
          f"   (sin dato: {sin_dato})")

    posiciones = [p for *_, p in emparejados if p is not None]
    en_decil_alto = sum(1 for p in posiciones if p >= 0.90)
    en_cuartil_alto = sum(1 for p in posiciones if p >= 0.75)
    mediana = sorted(posiciones)[len(posiciones) // 2]

    print(f"\n  ¿Dónde caen los meses con deslizamiento, dentro de la historia")
    print(f"  de su propio punto?\n")
    print(f"    posición mediana:              {100*mediana:.1f}%   "
          f"(azar puro daría 50%)")
    print(f"    en el decil superior:          {100*en_decil_alto/len(posiciones):.1f}%"
          f"   (azar puro daría 10%)")
    print(f"    en el cuartil superior:        {100*en_cuartil_alto/len(posiciones):.1f}%"
          f"   (azar puro daría 25%)")

    # ── brazos nulos ────────────────────────────────────────────────────────
    rng = random.Random(SEMILLA)
    real = en_decil_alto / len(posiciones)

    # NULL-1 · misma cantidad de eventos por punto, meses al azar DEL MISMO PUNTO
    conteo = defaultdict(int)
    for se, *_ in emparejados:
        conteo[se] += 1
    nulos1 = []
    for _ in range(PERMUTACIONES):
        alto = total = 0
        for se, cuantos in conteo.items():
            muestra = por_punto[se]
            for v in rng.sample(muestra, min(cuantos, len(muestra))):
                total += 1
                if decil_de(se, v) >= 0.90:
                    alto += 1
        nulos1.append(alto / total if total else 0)

    # NULL-2 · mismos meses, pero atribuidos a puntos al azar
    puntos = list(por_punto)
    nulos2 = []
    for _ in range(PERMUTACIONES):
        alto = total = 0
        for _se, a, m, _v, _p in emparejados:
            otro = rng.choice(puntos)
            if (otro, a, m) in peligro:
                total += 1
                if decil_de(otro, peligro[(otro, a, m)]) >= 0.90:
                    alto += 1
        nulos2.append(alto / total if total else 0)

    print(f"\n  Brazo REAL: {100*real:.1f}% de los eventos en el decil superior\n")
    for nombre, nulos in (("NULL-1 · meses al azar del mismo punto", nulos1),
                          ("NULL-2 · eventos atribuidos a otros puntos", nulos2)):
        media = sum(nulos) / len(nulos)
        peores = sum(1 for x in nulos if x >= real)
        p = (peores + 1) / (PERMUTACIONES + 1)
        print(f"    {nombre}")
        print(f"       media nula {100*media:.1f}%  ·  p = {p:.4f}")

    print("\n" + "=" * 74)
    # Umbral declarado antes de correr: el triple del azar. Si de cada diez
    # deslizamientos reales al menos tres caen en el 10% más peligroso de su
    # propio lugar, el número está apuntando a algo.
    veredicto = real >= 0.30
    dictamen = ("el peligro calculado se concentra en los meses en que hubo "
                "deslizamientos reales") if veredicto else (
                "no hay concentración suficiente")
    print(f"  → {'PASA' if veredicto else 'NO PASA'}: {dictamen}")
    print("\n  Lo que esta prueba NO dice: que la medida sirva para operar, ni")
    print("  que el peligro CAUSE el deslizamiento. Dice que el número apunta")
    print("  donde ocurrieron los hechos, sobre 329 casos que no participaron")
    print("  en su diseño.")
    return 0 if veredicto else 1


if __name__ == "__main__":
    sys.exit(main())
