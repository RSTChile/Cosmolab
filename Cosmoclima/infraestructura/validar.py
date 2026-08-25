"""
Validación: ¿el método sirve o no?

Corre las pruebas del PROTOCOLO_VALIDACION.md que ya se pueden correr con lo
que hay descargado. El protocolo se escribió ANTES de calcular nada, y este
script no lo modifica: sólo lo ejecuta.

Se corre tal cual y se reporta lo que dé. Si el brazo real no se separa de los
nulos, eso es el resultado — no se tocan pesos ni umbrales buscando que dé.
"""

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI / "adaptadores"))
import era5              # noqa: E402
import senapred_eventos  # noqa: E402

SUBESTACIONES = AQUI / "datos" / "subestaciones_con_comuna.csv"
SEMILLA = 20260815      # fija, para que la corrida sea reproducible
PERMUTACIONES = 1000


def cargar_subestaciones():
    with SUBESTACIONES.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def indexar_era5(obs):
    """(subestacion, anio, mes, variable) → valor normalizado."""
    idx = {}
    for o in obs:
        anio, mes = int(o["vigencia_inicio"][:4]), int(o["vigencia_inicio"][5:7])
        idx[(o["territorio_id"], anio, mes, o["variable"])] = o["valor_normalizado"]
    return idx


# ────────────────────────────────────────────────────────────────────────────
def prueba_1_ancla(idx):
    """El aluvión de Copiapó, 24-25 de marzo de 2015.

    Dos exigencias, y las dos tienen que cumplirse:
      · encendido    — Copiapó marzo 2015 tiene que marcar peligro alto
      · especificidad— Punta Arenas ese MISMO mes no tiene que marcarlo
    Un instrumento que marca todo no marca nada.
    """
    print("\n" + "=" * 74)
    print("PRUEBA 1 · ANCLA DE VERDAD TERRENO — Copiapó, marzo de 2015")
    print("=" * 74)

    casos = [
        ("Subestación Copiapó (Urbana)", 2015, 3, "debe encenderse"),
        ("Subestación Punta Arenas (Urbana)", 2015, 3, "NO debe encenderse"),
        ("Subestación Copiapó (Urbana)", 2015, 8, "control: mes tranquilo"),
    ]
    print(f"\n{'caso':38s} {'anomalía':>9s} {'evento':>9s}  {'esperado'}")
    resultados = {}
    for se, anio, mes, esperado in casos:
        anom = idx.get((se, anio, mes, "anomalia_precipitacion"))
        even = idx.get((se, anio, mes, "evento_precipitacion_intensa"))
        resultados[(se, anio, mes)] = (anom, even)
        a = f"{anom:.3f}" if anom is not None else "—"
        e = f"{even:.3f}" if even is not None else "—"
        print(f"{se[:36]:38s} {a:>9s} {e:>9s}  {esperado}")

    cop = resultados[("Subestación Copiapó (Urbana)", 2015, 3)]
    pta = resultados[("Subestación Punta Arenas (Urbana)", 2015, 3)]
    enciende = cop[0] is not None and cop[0] > 0.9
    especifico = pta[0] is not None and pta[0] < 0.7

    print(f"\n  encendido en Copiapó (>0,90):        "
          f"{'SÍ' if enciende else 'NO'}  ({cop[0]:.4f})")
    print(f"  silencio en Punta Arenas (<0,70):    "
          f"{'SÍ' if especifico else 'NO'}  ({pta[0]:.4f})")
    veredicto = enciende and especifico
    print(f"\n  → {'PASA' if veredicto else 'NO PASA'}: el método "
          f"{'distingue' if veredicto else 'NO distingue'} el evento real "
          f"del ruido de fondo")
    return veredicto


def prueba_2_separacion(idx, subestaciones):
    """¿Separa activos que hoy son idénticos?

    Las 39 comparten FEN=Alta, PF=0,75 y Pen=Muy Alta desde Arica hasta Punta
    Arenas. Si el consolidado les da el mismo número, no aporta nada.
    """
    print("\n" + "=" * 74)
    print("PRUEBA 2 · SEPARACIÓN TERRITORIAL — un mes cualquiera de invierno")
    print("=" * 74)

    anio, mes = 2024, 7
    valores = []
    for s in subestaciones:
        v = idx.get((s["subestacion"], anio, mes, "anomalia_precipitacion"))
        if v is not None:
            valores.append((v, s["subestacion"], s["comuna"],
                            s["zona_morfoclimatica"]))
    valores.sort(reverse=True)

    print(f"\njulio de {anio} — todas parten de FEN=Alta, PF=0,75, Pen=Muy Alta\n")
    print(f"{'#':>3} {'anomalía':>9s} {'subestación':34s} {'comuna':16s} zona")
    for i, (v, se, com, zona) in enumerate(valores[:6], 1):
        print(f"{i:3d} {v:9.3f} {se[:34]:34s} {str(com)[:16]:16s} {str(zona)[:26]}")
    print("   ...")
    for i, (v, se, com, zona) in enumerate(valores[-4:], len(valores) - 3):
        print(f"{i:3d} {v:9.3f} {se[:34]:34s} {str(com)[:16]:16s} {str(zona)[:26]}")

    solos = [v for v, *_ in valores]
    rango = max(solos) - min(solos)
    print(f"\n  n={len(solos)}   mínimo {min(solos):.3f}   máximo {max(solos):.3f}"
          f"   rango {rango:.3f}")
    veredicto = rango > 0.3
    print(f"  → {'PASA' if veredicto else 'NO PASA'}: "
          f"{'separa' if veredicto else 'no separa'} activos que la matriz "
          f"trata como idénticos")
    return veredicto


def prueba_5_contra_fallas(idx, subestaciones, obs_senapred):
    """Contra fallas eléctricas que ocurrieron de verdad (SENAPRED 2015-2024).

    La pregunta: en los meses-comuna donde hubo falla eléctrica registrada,
    ¿la anomalía de lluvia era más alta que en los meses sin falla?

    Con dos brazos nulos que rompen vínculos distintos:
      NULL-1 baraja las FECHAS  → descarta acertar por estación del año
      NULL-2 baraja los ACTIVOS → descarta acertar por «el norte es seco»
    """
    print("\n" + "=" * 74)
    print("PRUEBA 5 · CONTRA FALLAS ELÉCTRICAS REALES (SENAPRED 2015-2024)")
    print("=" * 74)

    # fallas eléctricas por (comuna normalizada, anio, mes)
    fallas = set()
    for o in obs_senapred:
        if o["variable"] != "falla_electricidad":
            continue
        anio, mes = int(o["vigencia_inicio"][:4]), int(o["vigencia_inicio"][5:7])
        fallas.add((o["comuna"], anio, mes))

    # pares (anomalía, hubo_falla) para las subestaciones con comuna resuelta
    pares = []
    for s in subestaciones:
        comuna = senapred_eventos.normalizar_nombre(s["comuna"])
        if not comuna:
            continue
        for anio in range(2015, 2025):
            for mes in range(1, 13):
                v = idx.get((s["subestacion"], anio, mes,
                             "anomalia_precipitacion"))
                if v is None:
                    continue
                pares.append((v, (comuna, anio, mes) in fallas,
                              s["subestacion"], anio, mes))

    con = [p[0] for p in pares if p[1]]
    sin = [p[0] for p in pares if not p[1]]
    if len(con) < 30:
        print(f"\n  Muestra insuficiente: sólo {len(con)} meses-comuna con falla.")
        print("  No se corre la prueba. NO se afloja el criterio para que dé.")
        return None

    media = lambda x: sum(x) / len(x)
    real = media(con) - media(sin)
    print(f"\n  meses con falla eléctrica:  n={len(con):5,d}  "
          f"anomalía media {media(con):.4f}")
    print(f"  meses sin falla:            n={len(sin):5,d}  "
          f"anomalía media {media(sin):.4f}")
    print(f"  diferencia REAL:            {real:+.4f}")

    rng = random.Random(SEMILLA)
    etiquetas = [p[1] for p in pares]
    valores = [p[0] for p in pares]

    # NULL-1 · baraja las fechas dentro de cada subestación
    por_se = defaultdict(list)
    for v, falla, se, anio, mes in pares:
        por_se[se].append((v, falla))
    nulos1 = []
    for _ in range(PERMUTACIONES):
        c, s = [], []
        for se, lista in por_se.items():
            vs = [x[0] for x in lista]
            fs = [x[1] for x in lista]
            rng.shuffle(fs)
            for v, f in zip(vs, fs):
                (c if f else s).append(v)
        nulos1.append(media(c) - media(s) if c and s else 0.0)

    # NULL-2 · baraja las etiquetas entre todos los activos
    nulos2 = []
    for _ in range(PERMUTACIONES):
        mezcla = etiquetas[:]
        rng.shuffle(mezcla)
        c = [v for v, f in zip(valores, mezcla) if f]
        s = [v for v, f in zip(valores, mezcla) if not f]
        nulos2.append(media(c) - media(s) if c and s else 0.0)

    for nombre, nulos in (("NULL-1 (fechas barajadas)", nulos1),
                          ("NULL-2 (activos barajados)", nulos2)):
        mas_extremos = sum(1 for x in nulos if x >= real)
        p = (mas_extremos + 1) / (PERMUTACIONES + 1)
        print(f"\n  {nombre}")
        print(f"     media nula {media(nulos):+.4f}   "
             f"real por encima del {100*(1-p):.1f}% de las permutaciones")
        print(f"     p = {p:.4f}")

    print("\n  Lectura honesta: esto mide si los meses con falla eléctrica")
    print("  registrada tuvieron más lluvia anómala. NO prueba causalidad, y la")
    print("  falla eléctrica puede tener mil causas ajenas al clima.")
    return real


def main():
    print("VALIDACIÓN · corrida del", "2026-08-15")
    print("Protocolo escrito antes de calcular: PROTOCOLO_VALIDACION.md")

    obs_era5, problema = era5.traer()
    if problema:
        print("SIN DATO era5:", problema)
        return 1
    idx = indexar_era5(obs_era5)
    subestaciones = cargar_subestaciones()
    print(f"\n{len(obs_era5):,} observaciones ERA5 · "
          f"{len(subestaciones)} subestaciones")

    r1 = prueba_1_ancla(idx)
    r2 = prueba_2_separacion(idx, subestaciones)

    obs_sen, problema = senapred_eventos.traer()
    if problema:
        print("\nSIN DATO senapred:", problema)
        r5 = None
    else:
        r5 = prueba_5_contra_fallas(idx, subestaciones, obs_sen)

    print("\n" + "=" * 74)
    print("RESUMEN")
    print("=" * 74)
    print(f"  Prueba 1 · ancla Copiapó 2015      {'PASA' if r1 else 'NO PASA'}")
    print(f"  Prueba 2 · separación territorial  {'PASA' if r2 else 'NO PASA'}")
    print(f"  Prueba 5 · contra fallas reales    "
          f"{'corrida' if r5 is not None else 'no corrida'}")
    print("\n  Pruebas 3 y 4 (nulos completos del consolidado, y contraste")
    print("  contra SERNAGEOMIN) quedan pendientes: falta el consolidado con")
    print("  la minuta, que hoy no tiene historia.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
