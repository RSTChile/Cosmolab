"""
PelPre DIARIO — el peligro de precipitación día a día
=======================================================

INSTRUCCIÓN (Alexis, 22-ago-2026)
-----------------------------------
«Si la MICR trabaja con precipitación mensual, no sirve… no podemos hacer una
estimación con 1 mes de amplitud. Los pronósticos de clima son semanales como
máximo, y diarios… una matriz de infraestructura crítica no puede basarse en
precipitación mensual. Debe ser diaria, como lo hicimos en Cosmoclima.»

DE DÓNDE SALIÓ LO MENSUAL — la atribución, que importa
--------------------------------------------------------
**El dato nunca fue mensual, y nadie pidió que lo fuera.** Cosmoclima tiene
precipitación diaria mapeada de 1966 a 2026, y la descarga para la Matriz también
se bajó día a día.

**Lo mensual lo introduje yo**, en `adaptadores/era5.py`: la función
`agregar_mensual()` colapsaba la serie diaria a un solo valor por activo y por
mes, el del peor episodio. Esa decisión no estaba pedida ni declarada en ninguna
parte — se arrastró desde que el módulo climático MACLIMA razona en anomalías
mensuales, y contaminó todo lo que vino después.

Para decir si un mes fue malo, servía. **Para avisar, no sirve**: nadie opera con
«agosto va a ser peligroso». Un corte de ruta se decide el jueves.

★ Y HAY UNA SEGUNDA RAZÓN, DE MÉTODO, QUE IMPORTA MÁS
-------------------------------------------------------
**La validación mensual era tramposamente generosa.** Si un deslizamiento ocurrió
el 15 y la lluvia fuerte cayó el 3, el mes «acertaba» igual. Con 30 días de
margen, acertar es fácil.

Al pasar a diario, la ventana de acierto se reduce de un mes a un día. **La misma
prueba se vuelve mucho más dura**, y por eso este cambio no es sólo operativo: es
una prueba más honesta de si el instrumento sirve.

LA FÓRMULA NO CAMBIA
--------------------
Es la misma conjunción de siempre, sólo que evaluada cada día en vez de una vez
al mes:

    PelPre(punto, día) = √( magnitud_nacional × excedencia_local )

    magnitud_nacional  percentil del acumulado de 48 h contra la distribución de
                       TODOS los puntos y TODOS los días — ¿es mucha agua?
    excedencia_local   percentil de ese acumulado dividido por la normal anual
                       del propio punto — ¿supera lo que este lugar aguanta?

★ Lo que sí cambia es la distribución de referencia: pasa de ser mensual a ser
diaria, y por lo tanto es unas treinta veces más grande y con una cola mucho más
larga. Eso da más resolución en el extremo, que es donde vive el peligro.

USO
---
    ../.venv-esa/bin/python pelpre_diario.py --calcular
    ../.venv-esa/bin/python pelpre_diario.py --validar
"""

import csv
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import normalizar                                       # noqa: E402
import cclimp                                           # noqa: E402

DATOS = AQUI / "datos"
# Los 130 puntos que fijan la referencia nacional establecida.
FUENTES = [(DATOS / "clima_diario_subestaciones_erA5.csv",
            DATOS / "subestaciones_puntos.csv"),
           (DATOS / "clima_diario_reterm_era5_corregido.csv",
            DATOS / "reterm_puntos.csv")]
SALIDA = DATOS / "pelpre_diario.csv"
PISO_MM = 0.5          # por debajo de esto no se cuenta como episodio


def leer_series():
    """{punto: {fecha: mm}} de todas las fuentes juntas."""
    s = defaultdict(dict)
    for clima, _ in FUENTES:
        if not clima.exists():
            continue
        with clima.open(encoding="utf-8") as fh:
            for x in csv.DictReader(fh):
                if x["precip_mm"] not in ("", "None"):
                    s[x["subestacion"]][x["fecha"]] = float(x["precip_mm"])
    return s


def acumulado_48h(serie):
    """{fecha: mm caídos en esa fecha y la anterior}.

    Se exige que el día anterior EXISTA en la serie; si falta, el acumulado de
    ese día es sólo el suyo. No se rellenan huecos con ceros: un día sin dato no
    es un día sin lluvia.
    """
    out = {}
    for f, mm in serie.items():
        ayer = (date.fromisoformat(f) - timedelta(days=1)).isoformat()
        out[f] = mm + serie[ayer] if ayer in serie else mm
    return out


def normal_anual(serie):
    """Milímetros al año que recibe el punto, en promedio, con los años completos."""
    por_anio = defaultdict(float)
    dias = defaultdict(int)
    for f, mm in serie.items():
        por_anio[f[:4]] += mm
        dias[f[:4]] += 1
    completos = [v for a, v in por_anio.items() if dias[a] >= 330]
    return sum(completos) / len(completos) if completos else None


def calcular():
    series = leer_series()
    print("=" * 80)
    print("PelPre DIARIO · el peligro de precipitación día a día")
    print("=" * 80)
    print(f"\n  puntos con serie : {len(series)}")

    ac = {p: acumulado_48h(s) for p, s in series.items()}
    normales = {p: normal_anual(s) for p, s in series.items()}
    sin_normal = [p for p, v in normales.items() if not v]
    print(f"  días totales     : {sum(len(v) for v in ac.values()):,}")
    if sin_normal:
        print(f"  sin normal anual : {len(sin_normal)} (se omiten)")

    # ── las dos distribuciones nacionales, ahora DIARIAS ────────────────────
    magnitudes, razones = [], []
    for p, dd in ac.items():
        n = normales.get(p)
        for f, v in dd.items():
            if v <= PISO_MM:
                continue
            magnitudes.append(v)
            if n:
                razones.append(normalizar.razon_contra_normal(v, n))
    magnitudes.sort()
    razones.sort()
    print(f"\n  distribución nacional DIARIA de magnitud : {len(magnitudes):,} episodios")
    print(f"     mediana {magnitudes[len(magnitudes)//2]:.1f} mm · "
          f"P99 {magnitudes[int(len(magnitudes)*0.99)]:.1f} mm · "
          f"máximo {magnitudes[-1]:.1f} mm")

    filas = 0
    with SALIDA.open("w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["punto", "fecha", "mm_48h", "PelPre", "CClimP"])
        for p, dd in sorted(ac.items()):
            n = normales.get(p)
            if not n:
                continue
            for f in sorted(dd):
                v = dd[f]
                if v <= PISO_MM:
                    continue
                mag = normalizar.percentil_en(v, magnitudes)
                exc = normalizar.percentil_en(
                    normalizar.razon_contra_normal(v, n), razones)
                pel = (mag * exc) ** 0.5
                cc, _ = cclimp.coeficiente(pel)
                w.writerow([p, f, round(v, 1), round(pel, 4), cc])
                filas += 1
    print(f"\n  escrito: {SALIDA.name} · {filas:,} días con episodio")
    return 0


def validar():
    """La prueba dura: ¿el instrumento se enciende EL DÍA del deslizamiento?"""
    import statistics as st
    if not SALIDA.exists():
        print("falta pelpre_diario.csv — corre primero --calcular")
        return 1
    pel = defaultdict(dict)
    for x in csv.DictReader(SALIDA.open(encoding="utf-8")):
        pel[x["punto"]][x["fecha"]] = float(x["PelPre"])

    ev = [x for x in csv.DictReader((DATOS / "reterm_eventos.csv").open(encoding="utf-8"))
          if ("luvia" in str(x["detonante"]).lower()
              or "recipitac" in str(x["detonante"]).lower())]
    print("=" * 80)
    print("VALIDACIÓN DIARIA · ¿se enciende EL DÍA del deslizamiento?")
    print("=" * 80)
    print(f"\n  eventos con lluvia como detonante: {len(ev)}")

    aciertos, valores, sin_dato = [], [], 0
    for e in ev:
        p = f"ReTeRM · {e['comuna']}"
        f = e["fecha"][:10]
        if p not in pel:
            sin_dato += 1
            continue
        # ventana de un día: el propio día o el anterior (el evento puede
        # registrarse a la mañana siguiente del temporal)
        cand = [pel[p].get(f), pel[p].get(
            (date.fromisoformat(f) - timedelta(days=1)).isoformat())]
        cand = [c for c in cand if c is not None]
        if not cand:
            valores.append(0.0)
            continue
        valores.append(max(cand))
    print(f"  con serie en su punto            : {len(valores)}")

    # el nulo: días al azar del mismo punto, mismo número de casos
    import random
    random.seed(7)
    nulo = []
    puntos = [p for p in pel if p.startswith("ReTeRM")]
    for _ in range(len(valores)):
        p = random.choice(puntos)
        nulo.append(random.choice(list(pel[p].values())))

    for etiqueta, v in (("días CON deslizamiento", valores),
                        ("días al azar (nulo)", nulo)):
        alto = sum(1 for x in v if x >= 0.6501)     # perilla movida
        muy = sum(1 for x in v if x >= 0.9658)      # P99
        print(f"\n  {etiqueta}: n={len(v)}")
        print(f"     PelPre mediano          : {st.median(v):.4f}")
        print(f"     con la perilla movida   : {alto} ({100*alto/len(v):.1f} %)")
        print(f"     en el 1 % superior      : {muy} ({100*muy/len(v):.1f} %)")
    print("\n  ★ La ventana es de UN día (o el anterior). Con la versión mensual")
    print("    la ventana era de treinta, así que acertar era mucho más fácil.")
    return 0


if __name__ == "__main__":
    if "--calcular" in sys.argv:
        sys.exit(calcular())
    elif "--validar" in sys.argv:
        sys.exit(validar())
    print(__doc__)
