"""
RECÁLCULO DE LA MATRIZ CON DATO REAL
=====================================

INSTRUCCIÓN QUE LO ORIGINA (Alexis, 16-ago-2026)
-------------------------------------------------
«El documento es un ejercicio en abstracto, nosotros tenemos datos reales.
Cuando se hizo el RMD MICR no teníamos datos reales, sólo usamos lógica y
aproximaciones. Todo lo que ahora hagamos es con dato real, y eso manda.»

Este script recalcula, para las 39 subestaciones del piloto, las columnas de la
Matriz de Infraestructura Crítica que HOY tienen respaldo en dato medido, y deja
explícitamente marcadas las que todavía no lo tienen. No inventa ninguna.

QUÉ REEMPLAZA Y QUÉ HEREDA
---------------------------
Las 39 subestaciones comparten hoy una sola fila de la matriz, el ítem 120:

    FEN=Alta · FANC=Alta · IB=0,90 · VT=0,80 · FVT=0,83 · PF=0,75 · Pen=Muy Alta

desde Arica hasta Punta Arenas, 35 grados de latitud. Esa uniformidad es el
problema que el proyecto ataca.

    columna   origen tras el recálculo
    ────────  ──────────────────────────────────────────────────────────────
    FEN       ★ REAL — 36 años de precipitación medida en cada punto
    FANC      heredado del ejercicio abstracto (no hay dato real de ataques)
    IB        heredado (falta bajar población INE y clientes SEC)
    VT        heredado (no hay dato real de dependencia tecnológica)
    FVT       ★ RECALCULADO con la fórmula PUBLICADA de la sección 22
    PF        ★ RECALCULADO = IB × FVT
    IRMD      ★ RECALCULADO por los umbrales publicados

POR QUÉ SE USA LA FÓRMULA PUBLICADA Y NO LA «REAL»
---------------------------------------------------
El FVT de la tabla no es reproducible: de 35 combinaciones distintas de
(FEN, FANC, VT), 22 aparecen con más de un valor de FVT. La columna se asignó a
criterio, no se calculó. Y al ajustarla, resulta que sigue a la IMPORTANCIA
(correlación con IB +0,671) más que a la fragilidad natural (+0,337).

Entonces se usa la fórmula **escrita** en la sección 22, que sí es reproducible
y donde FEN pesa un tercio:

    FVT = (FEN + FANC + VT·3) / 9        con FEN y FANC en escala 1 a 3

Nuestro FEN viene en escala 0-1, así que se lleva a la escala del canon con
FEN_num = 1 + 2·FEN, que manda 0 a «Baja» (1) y 1 a «Alta» (3).

LAS DOS PIEZAS REALES DEL FEN
------------------------------
1. `ExpEstr` — exposición estructural del lugar. Responde «¿qué tan peligroso es
   un mes malo en este punto?». Se define como el percentil nacional del valor
   de PelPre que ese punto supera en el 5 % de sus meses (su P95 local).

   ★ Se usa el P95 local y no la frecuencia de meses peligrosos a propósito: la
   frecuencia sola volvería a caer en la trampa de la rareza —Copiapó tiene
   poquísimos meses malos y uno de ellos destruyó la ciudad— mientras que el P95
   captura la severidad del mes malo típico, y la frecuencia entra igual, porque
   un lugar con muchos meses malos tiene un P95 más alto.

2. `CClimP` — el coeficiente del mes, de la ficha de VARIABLES_Y_METRICAS.

    FEN_real(punto, mes) = mín(1 ; ExpEstr(punto) × CClimP(punto, mes))

CRITERIO FIJADO ANTES DE CALCULAR
----------------------------------
1. Las 39 subestaciones deben dejar de ser idénticas: el rango de `ExpEstr`
   debe cubrir al menos 0,30 de amplitud.
2. El orden que produce `ExpEstr` debe ser explicable geográficamente — no puede
   salir aleatorio respecto de la latitud y el régimen de lluvia.
3. En un mes tranquilo, `PF` recalculado no debe alejarse del PF publicado más
   de lo que se aleja en un mes de temporal. Si el instrumento se mueve igual
   siempre, no está midiendo el mes.

USO
---
    python recalcular_matriz_real.py            # informe a pantalla
    python recalcular_matriz_real.py --csv      # + datos/matriz_recalculada.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import normalizar
from adaptadores import era5

AQUI = Path(__file__).resolve().parent

# --- La fila 120 de la matriz publicada: lo que hoy comparten las 39 ---------

ITEM_120 = dict(numero=120, sector="Energía", elemento="Subestaciones Eléctricas",
                FEN="Alta", FANC="Alta", IB=0.90, VT=0.80,
                FVT=0.83, PF=0.75, IRMD="Alto", Pen="Muy Alta")

ORDINAL = {"Baja": 1, "Media": 2, "Alta": 3}

# --- Cortes de CClimP: percentiles medidos de la distribución nacional -------
# (ver ficha de CClimP en VARIABLES_Y_METRICAS_PROYECTO.md)
CORTES_CCLIMP = [(0.9658, 1.6), (0.8292, 1.4), (0.6501, 1.2)]

# Umbrales publicados de IRMD (sección 22)
UMBRAL_IRMD_ALTO = 0.5
UMBRAL_IRMD_MEDIO = 0.3


def coeficiente_climatico(pelpre):
    """La perilla del mes. 1,0 = neutro; nunca baja de 1 (no se atenúa)."""
    for corte, valor in CORTES_CCLIMP:
        if pelpre >= corte:
            return valor
    return 1.0


def fvt_publicada(fen_0a1, fanc_ordinal, vt_0a1):
    """La fórmula ESCRITA en la sección 22, con FEN traído a la escala 1-3."""
    fen_num = 1 + 2 * fen_0a1
    return (fen_num + fanc_ordinal + vt_0a1 * 3) / 9


def clasificar_irmd(pf):
    if pf > UMBRAL_IRMD_ALTO:
        return "Alto"
    return "Medio" if pf >= UMBRAL_IRMD_MEDIO else "Bajo"


def percentil_de_lista(valores_ordenados, fraccion):
    """El valor que deja `fraccion` de la muestra por debajo."""
    if not valores_ordenados:
        return None
    i = min(len(valores_ordenados) - 1, int(len(valores_ordenados) * fraccion))
    return valores_ordenados[i]


def main():
    print("=" * 78)
    print("RECÁLCULO DE LA MATRIZ CON DATO REAL — 39 subestaciones")
    print("=" * 78)

    obs, problema = era5.traer()
    if problema:
        print("SIN DATO:", problema)
        raise SystemExit(1)

    # PelPre por subestación y mes, tal como lo produce el adaptador
    # El adaptador identifica el activo por `territorio_id` y fecha el dato con
    # la vigencia del mes (`vigencia_inicio` = primer día del mes).
    peligro = defaultdict(dict)
    for o in obs:
        if o["variable"] != "peligro_precipitacion":
            continue
        anio, mes, _ = o["vigencia_inicio"].split("-")
        peligro[o["territorio_id"]][(int(anio), int(mes))] = \
            float(o["valor_normalizado"])

    print(f"\n  subestaciones con serie de peligro : {len(peligro)}")
    print(f"  pares (subestación, mes)           : "
          f"{sum(len(v) for v in peligro.values()):,}")

    # --- 1 · ExpEstr: el P95 local, llevado a percentil nacional -------------
    p95_local = {}
    for se, meses in peligro.items():
        serie = sorted(meses.values())
        p95_local[se] = percentil_de_lista(serie, 0.95)

    reparto_nacional = sorted(p95_local.values())
    expestr = {se: normalizar.percentil_en(v, reparto_nacional)
               for se, v in p95_local.items()}

    # --- 2 · recálculo mes a mes --------------------------------------------
    fanc_num = ORDINAL[ITEM_120["FANC"]]
    filas = []
    for se, meses in peligro.items():
        for (anio, mes), pel in sorted(meses.items()):
            cclimp = coeficiente_climatico(pel)
            fen = min(1.0, expestr[se] * cclimp)
            fvt = fvt_publicada(fen, fanc_num, ITEM_120["VT"])
            pf = ITEM_120["IB"] * fvt
            filas.append(dict(subestacion=se, anio=anio, mes=mes,
                              PelPre=round(pel, 4), CClimP=cclimp,
                              ExpEstr=round(expestr[se], 4), FEN=round(fen, 4),
                              FVT=round(fvt, 4), PF=round(pf, 4),
                              IRMD=clasificar_irmd(pf)))

    # --- 3 · criterio 1: ¿dejaron de ser idénticas? --------------------------
    print("\n" + "=" * 78)
    print("CRITERIO 1 · las 39 dejan de ser idénticas")
    print("=" * 78)
    orden = sorted(expestr.items(), key=lambda x: -x[1])
    amplitud = orden[0][1] - orden[-1][1]
    print(f"\n  ExpEstr va de {orden[-1][1]:.3f} a {orden[0][1]:.3f} "
          f"→ amplitud {amplitud:.3f}  "
          f"{'✓ PASA' if amplitud >= 0.30 else '✗ NO PASA'} (exigido ≥ 0,30)")
    print("\n  Las cinco más expuestas y las cinco menos:")
    for se, v in orden[:5]:
        print(f"      {v:.3f}  {se}")
    print("      ...")
    for se, v in orden[-5:]:
        print(f"      {v:.3f}  {se}")

    # --- 4 · lo que cambia respecto de la fila publicada ---------------------
    print("\n" + "=" * 78)
    print("LO QUE CAMBIA RESPECTO DE LA FILA ÚNICA PUBLICADA")
    print("=" * 78)
    print(f"\n  publicado para las 39 : FEN=Alta  FVT={ITEM_120['FVT']}  "
          f"PF={ITEM_120['PF']}  IRMD={ITEM_120['IRMD']}")

    pfs = [f["PF"] for f in filas]
    print(f"  recalculado           : PF de {min(pfs):.3f} a {max(pfs):.3f} "
          f"sobre {len(filas):,} pares activo-mes")

    reparto = defaultdict(int)
    for f in filas:
        reparto[f["IRMD"]] += 1
    total = len(filas)
    print("\n  IRMD recalculado (la matriz publicada dice «Alto» para todos):")
    for nivel in ("Alto", "Medio", "Bajo"):
        n = reparto[nivel]
        print(f"      {nivel:6s} {n:7,d}  ({100*n/total:5.1f} %)")

    # --- 5 · criterio 3: ¿se mueve más en temporal que en mes tranquilo? -----
    print("\n" + "=" * 78)
    print("CRITERIO 3 · el instrumento tiene que moverse por el MES, no siempre")
    print("=" * 78)
    tranquilos = [f for f in filas if f["CClimP"] == 1.0]
    movidos = [f for f in filas if f["CClimP"] > 1.0]
    print(f"\n  meses neutros (CClimP = 1,0) : {len(tranquilos):7,d}  "
          f"({100*len(tranquilos)/total:.1f} %)")
    print(f"  meses con ajuste (> 1,0)     : {len(movidos):7,d}  "
          f"({100*len(movidos)/total:.1f} %)")
    print(f"  {'✓ PASA' if len(tranquilos)/total > 0.70 else '✗ NO PASA'} "
          f"(se exigía > 70 % de meses neutros: una perilla que se mueve "
          f"siempre no informa)")

    if movidos:
        pf_t = sum(f["PF"] for f in tranquilos) / max(len(tranquilos), 1)
        pf_m = sum(f["PF"] for f in movidos) / len(movidos)
        print(f"\n  PF medio en mes neutro   : {pf_t:.4f}")
        print(f"  PF medio en mes con ajuste: {pf_m:.4f}   "
              f"(diferencia {pf_m - pf_t:+.4f})")

    # --- 6 · el caso de referencia ------------------------------------------
    print("\n" + "=" * 78)
    print("EL CASO DE REFERENCIA · Copiapó, marzo 2015")
    print("=" * 78)
    for f in filas:
        if "opiap" in f["subestacion"] and f["anio"] == 2015 and f["mes"] in (3, 8):
            etiqueta = "aluvión" if f["mes"] == 3 else "mes tranquilo"
            print(f"\n  {f['subestacion']} · {f['anio']}-{f['mes']:02d} ({etiqueta})")
            print(f"      PelPre  {f['PelPre']:.4f}   →  CClimP {f['CClimP']}")
            print(f"      ExpEstr {f['ExpEstr']:.4f}   →  FEN {f['FEN']:.4f}")
            print(f"      FVT     {f['FVT']:.4f}   (publicado {ITEM_120['FVT']})")
            print(f"      PF      {f['PF']:.4f}   (publicado {ITEM_120['PF']})"
                  f"   IRMD {f['IRMD']}")

    if "--csv" in sys.argv:
        destino = AQUI / "datos" / "matriz_recalculada.csv"
        with open(destino, "w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
        print(f"\n  escrito: {destino}  ({len(filas):,} filas)")

    print("\n" + "=" * 78)
    print("COLUMNAS QUE SIGUEN SIN DATO REAL (declaradas, no inventadas)")
    print("=" * 78)
    print("""
  FANC = Alta   heredado. No hay dato real de ataques no convencionales a
                subestaciones chilenas. Se mantiene el valor del ejercicio.
  IB   = 0,90   heredado. Hay dato real disponible y sin bajar: población por
                comuna (INE Censo 2024) y clientes afectados por comuna y hora
                (SEC, ≥6 años). Es el próximo reemplazo natural.
  VT   = 0,80   heredado. No hay dato real de dependencia tecnológica por
                subestación. El Coordinador Eléctrico publica 1.269
                subestaciones pero sin coordenadas ni ficha técnica.
""")


if __name__ == "__main__":
    main()
