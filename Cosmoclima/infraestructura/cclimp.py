"""
CClimP y FENef — la perilla del clima y la fragilidad efectiva
===============================================================

QUÉ HACE ESTE MÓDULO
--------------------
Implementa las dos variables declaradas en `VARIABLES_Y_METRICAS_PROYECTO.md`,
secciones 2 y 3, y NADA MÁS. No valida: validar es trabajo de
`validar_cclimp.py`, que corre los tres criterios que la ficha fijó antes de
calcular. Separarlos es a propósito — quien implementa no puede ser quien decide
si su implementación pasó.

    PelPre  →  CClimP  →  FENef
    (mide)     (perilla)  (fragilidad del activo, este mes, en este lugar)

  · `PelPre` (Peligro de Precipitación) ya existe: lo produce `adaptadores/era5.py`
    y vale entre 0 y 1.
  · `CClimP` (Coeficiente Climático por Precipitación) traduce ese número a un
    multiplicador de cuatro escalones que el canon del RMD ya tenía definido.
  · `FENef` (Fragilidad ante Eventos Naturales, Efectiva) aplica esa perilla a
    la fragilidad de fábrica del tipo de elemento.

EN SIMPLE
---------
La Matriz sabe que una subestación eléctrica es frágil. No sabe que ESTA
subestación, ESTE mes, está bajo un temporal. `CClimP` es el volumen: queda en 1
cuando el mes es normal y no cambia nada, y sube hasta 1,6 cuando el mes está
entre el 1 % peor del país en 36 años. `FENef` es el resultado de subirle el
volumen a la fragilidad.

★ DOS DECISIONES QUE ESTE MÓDULO **NO** TOMA
---------------------------------------------
Las dos quedan como parámetro explícito, con las dos alternativas
implementadas, porque las dos son decisión del director y no del programador:

1. ★ **RESUELTO el 21-ago-2026 por Alexis: la escala de CUATRO niveles**, la
   oficial de SERNAGEOMIN. Lo que sigue se conserva porque explica de dónde
   venía la duda y por qué los números cambian.

   **Qué escala usar para el FEN de la Matriz** (`escala_fen`). La ficha de
   FENef habla de una escala de CUATRO niveles (H-15, adoptada el 16-ago-2026
   porque SERNAGEOMIN publica cuatro). Pero la Matriz de Infraestructura Crítica
   (MICR) trae TRES: medido sobre las 835 filas, `Media` 592 · `Alta` 167 ·
   `Baja` 76, y ni una sola dice «Muy Alta». El ítem 120 —Subestaciones
   Eléctricas, las 39 del piloto— dice `Alta`.

   Eso deja dos lecturas y dan números distintos:

       "cuatro"  por nombre    Baja 0,119 · Media 0,339 · Alta 0,661
       "tres"    por posición  Baja 0,119 · Media 0,500 · Alta 0,881

   El ejemplo de la ficha calcula con 0,881 y lo llama «Muy Alta», que es la
   lectura "tres" con el nombre de la escala de cuatro. La discrepancia queda
   declarada acá y se reporta en la validación con las dos.

2. ★ **RESUELTO el 21-ago-2026 por Alexis: forma PRODUCTO**, `mín(1; FEN × CClimP)`.
   ★ Y las dos decisiones juntas salieron bien: con la escala de cuatro el FEN de
   «Alta» vale 0,661, así que la saturación en 1,000 cae del 24,9 % al **1,0 %**
   de los pares. La disyuntiva del techo queda prácticamente disuelta.

   **La disyuntiva del techo** (`forma`). La ficha la deja abierta textualmente:
   la multiplicación simple es lo que MACC dice literalmente, pero satura —un
   activo en 0,881 topa en 1,000 ya con CClimP = 1,4, y entonces 1,4 y 1,6 son
   indistinguibles justo para los activos más frágiles, que son los que
   interesan. La forma de potencia no satura y con CClimP = 1 devuelve
   exactamente el FEN base.

FUENTES DE CADA NÚMERO
----------------------
  · Cortes 0,6501 / 0,8292 / 0,9658 : percentiles P75/P90/P99 MEDIDOS sobre la
    distribución nacional de PelPre (17.160 pares activo-mes, 39 subestaciones,
    1990-2026). No son elegidos. ★ Limitación H-10 vigente: salen de 39
    subestaciones, que son muestra y no inventario. Al ampliar el universo hay
    que recalcularlos, y eso se declara como RECÁLCULO, no como corrección.
  · Valores 1,0 / 1,2 / 1,4 / 1,6 : son los del canon MACC (0,8-1,2 leve,
    1,2-1,4 moderado, 1,4-1,6 alto, 1,0 neutro). No inventados.
  · Bloqueo por confianza baja : también del canon MACC.
  · Fragilidades 0,119 / 0,339 / 0,500 / 0,661 / 0,881 : salen de la curva
    logística común de `normalizar.f()` aplicada a las escalas ordinales
    oficiales. No se escriben a mano acá; se calculan al importar.
"""

import normalizar

# ── CClimP ──────────────────────────────────────────────────────────────────

# Percentiles medidos de la distribución nacional de PelPre. Ordenados de mayor
# a menor porque se recorren buscando el primer corte que el valor supera.
CORTES = [(0.9658, 1.6),    # P99 y superior  → ajuste alto
          (0.8292, 1.4),    # P90 – P99       → ajuste alto
          (0.6501, 1.2)]    # P75 – P90       → ajuste moderado
NEUTRO = 1.0                # bajo el P75 nacional: la perilla no se mueve

# ★ DECLARADO ACÁ, no estaba fijado antes. El canon MACC exige bloquear el
# coeficiente «si la confianza del dato es baja» pero no dice dónde está el
# corte. Se fija en 0,60 por esta razón: ERA5 tiene confianza base 0,70 y la
# cobertura del mes la multiplica, así que 0,60 equivale a exigir al menos un
# 86 % de los días del mes con dato. Un mes al que le faltan cuatro días o más
# puede haberse perdido justo el temporal, y entonces el número miente hacia
# abajo sin avisar.
CONFIANZA_MINIMA = 0.60

# ★ NO SE ATENÚA POR DEBAJO DE 1,0. MACC permite bajar hasta 0,8 y acá no se
# usa: un mes seco no vuelve más resistente a un puente, sólo no lo exige.
# Atenuar afirmaría que el activo es MENOS frágil, y eso no lo sostiene ningún
# dato. Si alguna vez se quiere atenuación, se justifica y se valida aparte.


def coeficiente(pelpre, confianza=None):
    """La perilla del mes: PelPre en [0,1] → multiplicador en {1,0 · 1,2 · 1,4 · 1,6}.

    `confianza` es opcional. Si viene y está bajo el mínimo declarado, el
    coeficiente se fuerza a neutro: ante un dato del que no nos podemos fiar, la
    respuesta correcta es no mover nada, no adivinar.

    Devuelve (valor, motivo) — el motivo dice qué regla se aplicó, para que
    ninguna celda del resultado quede sin poder explicarse.
    """
    if confianza is not None and confianza < CONFIANZA_MINIMA:
        return NEUTRO, f"bloqueado por confianza {confianza:.2f} < {CONFIANZA_MINIMA}"
    for corte, valor in CORTES:
        if pelpre >= corte:
            return valor, f"PelPre {pelpre:.4f} ≥ {corte}"
    return NEUTRO, f"PelPre {pelpre:.4f} < {CORTES[-1][0]} (bajo el P75 nacional)"


# ── FEN base por tipo de elemento ───────────────────────────────────────────

def _fragilidad(etiqueta, escala):
    """Lleva una etiqueta ordinal a la escala 0-1 por la curva común del proyecto."""
    return round(normalizar.f(normalizar.intensidad_ordinal(etiqueta, escala)), 4)


# Las dos lecturas de la columna FEN de la MICR. Se construyen, no se copian.
# "cuatro": el nombre de la MICR se traduce al nombre equivalente de la escala
#           de SERNAGEOMIN (Media → Moderada) y se lee en la escala de 4.
# "tres"  : se lee en la escala propia de la MICR, que es la que ella declara.
FEN_BASE = {
    "cuatro": {"baja": _fragilidad("baja", "peligro_4"),
               "media": _fragilidad("moderada", "peligro_4"),
               "alta": _fragilidad("alta", "peligro_4"),
               "muy alta": _fragilidad("muy alta", "peligro_4")},
    "tres": {"baja": _fragilidad("baja", "fen_3"),
             "media": _fragilidad("media", "fen_3"),
             "alta": _fragilidad("alta", "fen_3"),
             # La MICR nunca dice «Muy Alta» en FEN. Se admite el nombre para
             # que no reviente si alguna vez aparece, y toma el tope de la
             # escala de 3, que es lo único que esa escala puede ofrecer.
             "muy alta": _fragilidad("alta", "fen_3")},
}


# ★ DECIDIDO POR ALEXIS el 21-ago-2026: «4 niveles», la escala oficial de
# SERNAGEOMIN (Baja · Moderada · Alta · Muy Alta). Queda como predeterminada.
# La de tres se conserva sólo para poder reproducir cálculos anteriores.
ESCALA_ADOPTADA = "cuatro"


def fen_base(etiqueta_micr, escala_fen=ESCALA_ADOPTADA):
    """La fragilidad de fábrica del TIPO de elemento, según la MICR.

    `escala_fen` es la decisión declarada arriba: "tres" (la escala que la MICR
    realmente usa) o "cuatro" (la escala H-15 adoptada por el proyecto).
    """
    if escala_fen not in FEN_BASE:
        raise ValueError(f"escala_fen debe ser 'tres' o 'cuatro', no {escala_fen!r}")
    clave = str(etiqueta_micr).strip().lower()
    tabla = FEN_BASE[escala_fen]
    if clave not in tabla:
        raise ValueError(f"FEN {etiqueta_micr!r} no está en la escala "
                         f"(admite: {sorted(tabla)})")
    return tabla[clave]


# ── FENef ───────────────────────────────────────────────────────────────────

def fenef(fen, cclimp, forma="producto"):
    """La fragilidad efectiva del activo, en ESTE lugar y ESTE mes.

    Dos formas, la disyuntiva que la ficha dejó abierta:

    "producto"  FENef = mín(1 ; FEN × CClimP)
        Es lo que MACC dice literalmente. Satura: con FEN = 0,881, un CClimP de
        1,4 ya llega a 1,000, así que 1,4 y 1,6 dan lo mismo — y eso pasa justo
        en los activos más frágiles, que son los que interesan priorizar.

    "potencia"  FENef = 1 − (1 − FEN) ^ CClimP
        No satura nunca, jamás pasa de 1, y con CClimP = 1 devuelve exactamente
        el FEN base. Lo que hace es comerse una fracción de lo que le FALTA al
        activo para ser totalmente frágil, en vez de multiplicar lo que ya es.

    En simple: la primera forma es «súbele el volumen y córtalo si se pasa»; la
    segunda es «acércalo al máximo sin llegar nunca». Las dos coinciden cuando
    el mes es normal; se separan cuando hay temporal.
    """
    if forma == "producto":
        return min(1.0, fen * cclimp)
    if forma == "potencia":
        return 1.0 - (1.0 - fen) ** cclimp
    raise ValueError(f"forma debe ser 'producto' o 'potencia', no {forma!r}")


if __name__ == "__main__":
    print("CClimP — la perilla\n")
    for p in (0.30, 0.6501, 0.75, 0.8292, 0.90, 0.9658, 0.9888):
        v, motivo = coeficiente(p)
        print(f"   PelPre {p:.4f}  →  CClimP {v}    ({motivo})")
    v, motivo = coeficiente(0.9888, confianza=0.40)
    print(f"   PelPre 0.9888 con confianza 0,40  →  CClimP {v}   ({motivo})")

    print("\nFEN base de la MICR, en las dos lecturas\n")
    print(f"   {'etiqueta':10s} {'escala de 3':>12s} {'escala de 4':>12s}")
    for etiqueta in ("Baja", "Media", "Alta"):
        print(f"   {etiqueta:10s} {fen_base(etiqueta,'tres'):12.4f} "
              f"{fen_base(etiqueta,'cuatro'):12.4f}")

    print("\nFENef — el caso de referencia, Copiapó marzo 2015 (PelPre 0,9888)\n")
    cc, _ = coeficiente(0.9888)
    print(f"   {'FEN base':>10s} {'CClimP':>7s} {'producto':>10s} {'potencia':>10s}")
    for escala in ("tres", "cuatro"):
        base = fen_base("Alta", escala)
        print(f"   {base:10.4f} {cc:7.1f} "
              f"{fenef(base, cc, 'producto'):10.4f} "
              f"{fenef(base, cc, 'potencia'):10.4f}   (escala de {escala})")
    print("\n   y el mismo activo en un mes tranquilo (CClimP = 1,0):")
    for escala in ("tres", "cuatro"):
        base = fen_base("Alta", escala)
        print(f"   {base:10.4f} {1.0:7.1f} "
              f"{fenef(base, 1.0, 'producto'):10.4f} "
              f"{fenef(base, 1.0, 'potencia'):10.4f}   (escala de {escala})")
