"""
LAS FÓRMULAS DE LA MATRIZ, COMO CÓDIGO
========================================

★ POR QUÉ ESTE MÓDULO TENÍA QUE EXISTIR
-----------------------------------------
Hasta hoy (23-ago-2026) **ningún archivo del proyecto calculaba la Matriz**. Las
diez fórmulas vivían como CADENAS DE TEXTO dentro de
`formalizar_micr_en_protocolo.py`, que es un escritor de fichas para el Excel del
Protocolo, no un motor. Todo cálculo se hizo a mano, una vez, sin dejar rastro.

Eso tuvo una consecuencia medible: **el recalibrado de los cortes de la Pev del
22-ago se perdió**. Se había hecho interactivamente contra SharePoint y no quedó
en ninguna parte. (Se recuperó por inferencia; ver `CORTES_PEV_RECUPERADOS`.)

Mientras las fórmulas no sean código, cada medición es irrepetible.

LA REGLA QUE GOBIERNA ESTE ARCHIVO
------------------------------------
INSTRUCCIÓN (Alexis, 21-ago-2026): «la MICR intentó normalizar y sistematizar la
infraestructura crítica pero sin comprobar si funcionaba para todo… ahora estamos
haciendo eso, no en el Word, no en el Excel, AQUÍ.»

Por eso este módulo separa dos cosas que suelen confundirse:

  · **`legado`** — las fórmulas TAL COMO ESTÁN PUBLICADAS. No se tocan. Sirven
    de CONTROL: si dejan de reproducir la Matriz publicada, algo se rompió.
  · lo que venga después — las fórmulas corregidas, en `recalibrar_micr.py`.

Nunca se ajusta el legado para que calce. Si no reproduce, se reporta.

SIGLAS, EXPANDIDAS
------------------
  FEN   Fragilidad ante Eventos Naturales          (etiqueta)
  FANC  Fragilidad ante Ataques No Convencionales  (etiqueta)
  IB    Importancia Base                           (0 a 1)
  VT    Vulnerabilidad Tecnológica                 (0 a 1)
  FVT   Factor de Vulnerabilidad Total             (0 a 1)
  PF    Ponderación Final                          (0 a 1)
  IRMD  índice del Riesgo Multi-Dimensional        (etiqueta)
  Pev   Prioridad Estratégica · vertical-regular   (guerra convencional)
  Peh   Prioridad Estratégica · horizontal-irregular (ataque irregular/cibernético)
  Pen   Prioridad Estratégica · desastres naturales

USO
---
    ../.venv-esa/bin/python micr.py            # corre el control de reproducción
    from micr import fvt, pf, irmd, pev, peh, pen, banda
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"
FUENTE = DATOS / "micr_sharepoint_ultimo.csv"

# ── escalas ──────────────────────────────────────────────────────────────────
# ★ CUATRO NIVELES, por decisión de Alexis del 21-ago-2026 («Sí, 4 niveles»),
# para equiparar la escala del FEN a la oficial de SERNAGEOMIN.
#
# ⚠️ El mapeo canónico del Word NO contempla «Muy Alta»: dice
# `SI(FEN="Alta";3;SI(FEN="Media";2;1))`, que mandaría «Muy Alta» a 1 — o sea que
# trataría lo más frágil como lo menos frágil. Hay 3 filas con esa etiqueta.
# Aquí se declara explícitamente para que el hueco no vuelva a pasar inadvertido.
NIVEL = {"Muy Alta": 4, "Alta": 3, "Media": 2, "Moderada": 2, "Baja": 1}
NIVEL_MAX = 4


def fen_num(etiqueta):
    """FEN de etiqueta a número. `Moderada` se acepta como sinónimo de `Media`."""
    return NIVEL[etiqueta]


def fanc_num(etiqueta):
    return NIVEL[etiqueta]


# ── las fórmulas publicadas ──────────────────────────────────────────────────
# Transcritas de `formalizar_micr_en_protocolo.py`, líneas 220, 260, 286, 323,
# 356 y 383. Se conservan literales, con sus defectos, porque son el control.

def fvt(fen, fanc, vt):
    """Factor de Vulnerabilidad Total = (FEN + FANC + VT·3) / 9.

    ⚠️ MEDIDO: esta fórmula reproduce sólo 10 de 845 filas de la columna
    publicada. De 35 combinaciones distintas de (FEN, FANC, VT), 22 aparecen con
    más de un valor de FVT. **La columna publicada no es función de sus
    entradas**: se asignó a criterio experto. Para reproducir la Matriz tal como
    está hay que LEER la columna, no calcularla.
    """
    return (fen + fanc + vt * 3) / 9


def pf(ib, fvt_):
    """Ponderación Final = IB · FVT. Reproduce 816 de 845; el resto es redondeo."""
    return ib * fvt_


def irmd(pf_):
    """⚠️ MEDIDO: reproduce 616 de 845. Tampoco es función de su entrada."""
    return "Alto" if pf_ > 0.5 else "Medio" if pf_ >= 0.3 else "Bajo"


# Divisores. ★ Son el MÁXIMO OBSERVADO, no una constante: por eso la Matriz
# tiene no estacionariedad — al agregar una fila más alta, TODA la columna se
# recalcula. Es el hallazgo H-A12 y la razón de la renormalización de la Fase 2.
# ⚠️ Los del Word oficial (1,61 / 1,94 / 1,87) están viejos: con 1,87 la Pen
# reproduce 377/835 en vez de 845/845.
DIV_PEV, DIV_PEH, DIV_PEN = 1.549, 1.944, 1.959


def pev(ib, fanc, fvt_, divisor=DIV_PEV):
    """Prioridad ante guerra convencional. Pesos: IB 0,5 · FANC 0,3 · FVT 0,2."""
    return (0.5 * ib + 0.3 * fanc + 0.2 * fvt_) / divisor


def peh(fanc, vt, fvt_, divisor=DIV_PEH):
    """Prioridad ante ataque irregular. Pesos: FANC 0,5 · VT 0,3 · FVT 0,2.

    ⚠️ EL DEFECTO CENTRAL: la IB **no aparece, y el FVT tampoco la contiene**, así
    que el peso efectivo de la importancia es CERO EXACTO. Medido: los
    Dispositivos IoT (IB 0,50) quedan por encima de los Cuarteles Policiales
    (IB 0,95). No es un defecto de dato — es de fórmula.
    """
    return (0.5 * fanc + 0.3 * vt + 0.2 * fvt_) / divisor


def pen(fen, ib, fvt_, divisor=DIV_PEN):
    """Prioridad ante desastres naturales. Pesos: FEN 0,5 · IB 0,3 · FVT 0,2."""
    return (0.5 * fen + 0.3 * ib + 0.2 * fvt_) / divisor


# ── bandas ───────────────────────────────────────────────────────────────────
# Los cortes canónicos del Word, iguales para las tres.
CORTES_CANONICOS = [(0.96, "Muy Alta"), (0.85, "Alta"),
                    (0.70, "Media"), (0.50, "Baja")]

# ★ CORTES RECUPERADOS POR INFERENCIA el 23-ago-2026.
#
# El recalibrado del 22-ago no dejó script: se hizo interactivamente contra
# SharePoint. Se reconstruyó midiendo, sobre el export fresco, el rango de
# valores que ocupa cada banda PUBLICADA. Los rangos **no se solapan** — entre
# banda y banda hay un hueco limpio — así que las fronteras quedan determinadas.
#
# ★ Cada corte es el PUNTO MEDIO DEL HUECO, no el mínimo de la banda de arriba.
# Poner el corte en el mínimo lo deja pegado al dato y una diezmilésima de
# redondeo lo cruza: probado, la fila 18 «Sistemas de Desalinización para Agua
# Potable» caía de «Media» a «Baja» por 0,00001. El punto medio es el único
# lugar donde el corte no depende del redondeo.
#
# ⚠️ Son valores RECUPERADOS, no documentados: nadie los escribió en su momento.
# Se conservan para poder reproducir lo publicado, no como valor canónico.
CORTES_PEV_RECUPERADOS = [(0.778567, "Muy Alta"), (0.737896, "Alta"),
                          (0.708522, "Media"), (0.632667, "Baja")]

# ⚠️ En la Peh el corte de «Alta» es IGUAL al de «Muy Alta», y no por descuido:
# entre 0,728395 y 0,733539 la distribución NO TIENE NINGÚN VALOR. Los dos
# cortes cayeron dentro del mismo hueco, así que la banda «Alta» **no puede
# llenarse nunca**, con ningún dato. Es un defecto de forma, no de calibración:
# el FANC pesa 0,5 y sólo toma cuatro valores, así que la distribución se
# apelmaza en escalones y cinco bandas no caben. Lo arregla la Fase 2.
CORTES_PEH_RECUPERADOS = [(0.730967, "Muy Alta"), (0.730967, "Alta"),
                          (0.707562, "Media"), (0.611111, "Baja")]

# La Pen NO fue recalibrada: reproduce 845/845 con los cortes canónicos del
# Word. Sus huecos son enormes (0,156 entre «Alta» y «Media»), así que la
# distribución tolera cortes redondos. Se deja explícito para que se note que
# es un hecho medido y no un olvido.
CORTES_PEN = CORTES_CANONICOS


def cortes_desde_bandas(filas, columna, calcular):
    """★ Deriva los cortes de una columna desde sus bandas ya publicadas.

    Es lo que permitió recuperar el recalibrado perdido, y sirve para volver a
    verificarlo cuando el export cambie. Devuelve la lista de cortes, o levanta
    ValueError si dos bandas se solapan — en cuyo caso las bandas publicadas NO
    provienen de cortes sobre esta fórmula y no hay nada que recuperar.
    """
    ORDEN = ("Muy Alta", "Alta", "Media", "Baja", "Muy Baja")
    por = defaultdict(list)
    for x in filas:
        por[x[columna]].append(calcular(x))
    cortes, anterior_min = [], None
    for etiqueta in ORDEN[:-1]:
        vals = por.get(etiqueta)
        if not vals:
            # banda vacía: hereda el corte de la de arriba, que es justo lo que
            # la vuelve inalcanzable. Se conserva para reproducir fielmente.
            if cortes:
                cortes.append((cortes[-1][0], etiqueta))
            continue
        lo, hi = min(vals), max(vals)
        if anterior_min is not None and hi >= anterior_min:
            raise ValueError(f"{columna}: «{etiqueta}» se solapa con la banda "
                             f"superior ({hi:.6f} >= {anterior_min:.6f})")
        siguiente = max((max(v) for e, v in por.items()
                         if v and max(v) < lo), default=lo)
        cortes.append((round((lo + siguiente) / 2, 6), etiqueta))
        anterior_min = lo
    return cortes


def banda(valor, cortes=CORTES_CANONICOS):
    """Etiqueta de banda. El primer corte que el valor alcanza, de mayor a menor."""
    for umbral, nombre in cortes:
        if valor >= umbral:
            return nombre
    return "Muy Baja"


# ── control de reproducción ──────────────────────────────────────────────────

def leer(ruta=FUENTE):
    """Las filas con número. La fila huérfana sin `n` se excluye y se avisa."""
    filas, huerfanas = [], []
    for x in csv.DictReader(Path(ruta).open(encoding="utf-8")):
        (filas if x["n"] not in (None, "") else huerfanas).append(x)
    return filas, huerfanas


def reproducir(filas):
    """¿Las fórmulas publicadas reproducen la Matriz publicada? Sin ajustar nada."""
    r = defaultdict(int)
    for x in filas:
        fen, fanc = fen_num(x["FEN"]), fanc_num(x["FANC"])
        ib, vt, fvt_pub = float(x["IB"]), float(x["VT"]), float(x["FVT"])
        r["n"] += 1
        r["FVT"] += abs(fvt(fen, fanc, vt) - fvt_pub) < 0.005
        r["PF"] += abs(pf(ib, fvt_pub) - float(x["PF"])) < 0.005
        r["IRMD"] += irmd(float(x["PF"])) == x["IRMD"]
        r["Pev"] += banda(pev(ib, fanc, fvt_pub),
                          CORTES_PEV_RECUPERADOS) == x["Pev"]
        r["Peh"] += banda(peh(fanc, vt, fvt_pub),
                          CORTES_PEH_RECUPERADOS) == x["Peh"]
        r["Pen"] += banda(pen(fen, ib, fvt_pub), CORTES_PEN) == x["Pen"]
    return r


def main():
    filas, huerfanas = leer()
    print("=" * 78)
    print("CONTROL DE REPRODUCCIÓN · ¿las fórmulas publicadas dan la Matriz publicada?")
    print("=" * 78)
    print(f"\n  fuente : {FUENTE.name}  ·  {len(filas)} filas con número")
    for h in huerfanas:
        print(f"  ⚠️  excluida por no tener número: «{h['elemento']}»")

    r = reproducir(filas)
    n = r["n"]
    print()
    for col in ("FVT", "PF", "IRMD", "Pev", "Peh", "Pen"):
        pctj = 100 * r[col] / n
        marca = "✓" if r[col] == n else ("~" if pctj >= 95 else "✗")
        print(f"  {marca} {col:5s} {r[col]:4d} / {n}  ({pctj:5.1f} %)")

    print("\n" + "=" * 78)
    print("CÓMO LEER ESTO")
    print("=" * 78)
    print("""
  Pev · Peh · Pen deben dar 845/845. Son el CONTROL: si alguna baja, o el
  export cambió o la transcripción está mal, y no se sigue adelante.

  FVT e IRMD ✗ NO es un fallo de este módulo: es el hallazgo. Esas dos
  columnas se asignaron a criterio y no se pueden reproducir desde sus
  entradas. La Fase 2 las vuelve calculadas.
""")

    # el reparto de bandas, que es donde se ve el defecto de la Peh
    print("=" * 78)
    print("REPARTO DE BANDAS PUBLICADO")
    print("=" * 78 + "\n")
    for col in ("Pev", "Peh", "Pen"):
        c = Counter(x[col] for x in filas)
        linea = " · ".join(f"{b}:{c.get(b, 0)}" for b in
                           ("Muy Alta", "Alta", "Media", "Baja", "Muy Baja"))
        vacias = [b for b in ("Muy Alta", "Alta", "Media", "Baja", "Muy Baja")
                  if not c.get(b)]
        print(f"  {col}  {linea}")
        if vacias:
            print(f"       ⚠️  banda(s) VACÍA(S): {', '.join(vacias)}")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# LA MATRIZ RENORMALIZADA · 23-ago-2026
# ═══════════════════════════════════════════════════════════════════════════
# Todo lo de arriba es el LEGADO y se conserva como control. Lo de aquí abajo
# es la Matriz corregida, y es lo que rige de ahora en adelante.
#
# Qué cambió y por qué, en una línea cada uno:
#   · entradas normalizadas a 0-1 ⇒ **el divisor desaparece**, y con él la no
#     estacionariedad: agregar filas ya no recalcula la columna entera.
#   · la IB entra en la Peh con peso 0,30, el mismo que ya tenía en la Pen.
#   · el FVT y el IRMD pasan de asignados a criterio a CALCULADOS.
#   · el FANC se midió en cuatro grados contra el Protocolo I (ver medir_fanc.py).

PESOS = {
    # Pev y Pen conservan sus pesos publicados: ya sumaban 1, el problema
    # estaba en las entradas, no en ellos.
    "Pev": {"ib": 0.50, "fanc": 0.30, "fvt": 0.20},
    "Pen": {"fen": 0.50, "ib": 0.30, "fvt": 0.20},
    # ★ Peh: elegida por malla con cinco criterios declarados. Ver
    # `recalibrar_micr.py`. Spearman 0,9558 contra la fórmula anterior, o sea
    # que corrige el defecto sin rehacer el orden.
    "Peh": {"fanc": 0.45, "ib": 0.30, "vt": 0.20, "fvt": 0.05},
}

# ★ CORTES CONGELADOS.
# Calculados UNA VEZ, el 23-ago-2026, por los percentiles 95/80/50/20 sobre las
# 845 filas de entonces. **No se recalculan.** Recalcularlos en cada corrida
# reintroduciría la no estacionariedad que la renormalización vino a eliminar:
# agregar cien ítems movería de banda a filas que nadie tocó.
#
# Es el mismo principio que ya se aplicó al PelPre: la referencia se congela.
CORTES_PEV_2026 = [(0.7333333333, "Muy Alta"), (0.6677777778, "Alta"),
                   (0.5708333333, "Media"), (0.5011111111, "Baja")]
CORTES_PEH_2026 = [(0.7477777778, "Muy Alta"), (0.6744444444, "Alta"),
                   (0.5633333333, "Media"), (0.4766666667, "Baja")]
CORTES_PEN_2026 = [(0.7141666667, "Muy Alta"), (0.6255555556, "Alta"),
                   (0.4788888889, "Media"), (0.4516666667, "Baja")]
CORTES_IRMD_2026 = [(0.4272222222, "Alto"), (0.3411111111, "Medio")]


def n01(etiqueta):
    """Etiqueta de fragilidad a 0-1. Baja→0 · Media→⅓ · Alta→⅔ · Muy Alta→1."""
    return (NIVEL[etiqueta] - 1) / (NIVEL_MAX - 1)


def fvt_n(fen, fanc, vt):
    """FVT calculado: promedio de las tres entradas, todas ya en 0-1."""
    return (fen + fanc + vt) / 3


def pev_n(ib, fanc, fvt_):
    w = PESOS["Pev"]
    return w["ib"] * ib + w["fanc"] * fanc + w["fvt"] * fvt_


def peh_n(fanc, ib, vt, fvt_):
    w = PESOS["Peh"]
    return w["fanc"] * fanc + w["ib"] * ib + w["vt"] * vt + w["fvt"] * fvt_


def pen_n(fen, ib, fvt_):
    w = PESOS["Pen"]
    return w["fen"] * fen + w["ib"] * ib + w["fvt"] * fvt_


def irmd_n(pf_):
    """Tres niveles. Respaldo propio: el IRMD no tiene «Muy Baja»."""
    return next((nom for c, nom in CORTES_IRMD_2026 if pf_ >= c), "Bajo")


if __name__ == "__main__":
    sys.exit(main())
