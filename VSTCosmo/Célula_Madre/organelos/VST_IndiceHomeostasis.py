#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ÍNDICE DE HOMEOSTASIS — ¿están las variables INTERNAS en su rango adecuado?
================================================================================
6 de agosto de 2026.

QUÉ ES LA HOMEOSTASIS. La capacidad del cuerpo de mantener un equilibrio interno
estable y constante A PESAR de los cambios del entorno. Por eso el índice **marca
siempre un valor**: vale 1 en un organismo con todos sus sistemas estables y baja
sólo cuando una variable interna se sale de su rango — como una fiebre, donde la
temperatura está por encima de su media.

POR QUÉ HIZO FALTA ESCRIBIRLO. El organismo tenía dos señales llamadas
`H_homeostasis` y `H_homeostasis_real`, y NINGUNA de las dos medía esto: las dos
eran funciones de `A_sys_env`, o sea del ACOPLAMIENTO CON EL ENTORNO. Eso es
adaptación —«¿sigo agarrado a mi nicho?»— y ahora se llama `acople_sostenido`
(canónica) y `acople_sostenido_daisy` (la anterior). Dos generaciones del mismo
instrumento peleándose un nombre que no les correspondía.

POR QUÉ UN PROMEDIO Y NO UN PRODUCTO. Las dos viejas eran productos de 4-5
factores, así que valían 0 por defecto y 1 sólo si todo se alineaba a la vez.
MEDIDO sobre 107.027 pasos: `H_homeostasis` daba mediana 0,0000 con p95 0,9731 —
un interruptor, no un índice, y encima al revés de lo que pide el concepto. El
promedio de las saludes parciales da mediana 0,4121 con sólo el 2,05 % en cero:
marca siempre. (Se ensayó también el MÍNIMO, «la peor variable»: vuelve a ser un
interruptor, 58,17 % en cero, porque basta una variable en el suelo. Se publica
igual como `H_peor` para que un problema grave en una sola variable no quede
diluido en el promedio — el índice resume, la componente denuncia.)

LO QUE ESTE ÍNDICE **NO** ES. No es el gasto metabólico. Sostener el equilibrio
CUESTA energía, y cuesta más cuanto más empuja el entorno, pero ese costo es el
COSTO de mantener los niveles en rango, no el equilibrio mismo. Si H dependiera
del esfuerzo, un organismo que se esfuerza mucho se leería sano por esforzarse, y
uno que no tiene con qué esforzarse se leería sano por no gastar. Son dos
magnitudes distintas y se miden por separado.

POR QUÉ LOS RANGOS SON ABSOLUTOS Y NO MEDIAS MÓVILES. Es la advertencia 2 de
`escala.py`, literal: «un consigo-mismo aplicado a un SETPOINT HOMEOSTÁTICO borra
el estado sostenido: un organismo crónicamente hambriento leería "hambre normal"
y dejaría de registrar que se muere». Un rango de viabilidad es una CONDICIÓN DE
VIDA, no una percepción. El mismo criterio que el código ya respeta con `A_min`
(«debe seguir siendo absoluta aunque el organismo se acostumbre a estar mal»).

Y AUN ASÍ, NINGÚN NÚMERO NUEVO. Cada salud parcial se mide contra el rango que la
propia variable YA declara: `x_interna` vive declarada en [x_min, x_max] y la
reserva ya viene normalizada en [0,1], donde 0 es la muerte y 1 está llena. No
hay ningún umbral que elegir. (El primer ensayo dividía la reserva por 0,5 para
decidir «cuánta reserva es suficiente»: exactamente el tipo de constante que este
proyecto persigue. Se quitó.)
"""
from __future__ import annotations

# Cada entrada: (columna que se lee, forma del rango, parámetros, columna publicada).
# La FORMA importa: hay variables con dos bordes (un desorden puede ser demasiado alto
# o demasiado bajo, como la temperatura) y variables de una sola cara (la reserva sólo
# puede faltar). Confundirlas es lo que hacía que un cuerpo lleno de energía puntuara
# igual que uno vacío.
VARIABLES: list[tuple] = [
    ("x_interna", "centrada", (0.0, 1.0), "H_var_desorden"),
    ("met_energia", "piso", (), "H_var_reserva"),
]

# Las dos margaritas y el trabajo de regular viajan aquí porque hasta hoy NO llegaban al CSV
# —lo decía el propio organelo: «se puede ver que H se apaga, no POR QUÉ se apaga»— y sin
# verlas no se puede juzgar ni la regulación ni lo que cuesta. `costo_activo` se publica y
# todavía NO se cobra: primero medir su escala frente a met_gasto, después ponerle precio.
# `estres`, `perturb_habitual` y `perturb_exceso` se agregaron el 6-ago-2026: el estrés es
# la ENTRADA que mueve toda la regulación y tampoco llegaba al CSV. Se veía a las margaritas
# reaccionar y no a qué reaccionaban, y por eso el defecto del cero —que el organismo vivía
# con estrés permanente por el solo hecho de tener mundo— hubo que descubrirlo reconstruyendo
# la fórmula a mano desde e_R.
COLS_TRABAJO_HOMEO = ["x_interna_orden", "x_interna_entropia",
                      "x_interna_esfuerzo", "x_interna_costo_activo",
                      "x_interna_estres", "x_interna_perturb_habitual", "x_interna_perturb_exceso"]
COLS_INDICE_HOMEO = (["H_homeostasis", "H_peor", "H_n_variables"]
                     + [v[3] for v in VARIABLES] + COLS_TRABAJO_HOMEO)


def _num(v, defecto=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return defecto
    return defecto if x != x else x            # NaN → sin dato


def salud_centrada(x: float, lo: float, hi: float) -> float:
    """1 en el centro del rango declarado, 0 en los bordes y fuera.

    Sin parámetro libre: la escala es la geometría del rango que la variable ya declara.
    Ojo con una tensión real y deliberada: la REGULACIÓN de `x_interna` es Daisyworld y
    no tiene setpoint (el equilibrio emerge de la competencia). Este índice no le impone
    uno — no empuja a `x` a ningún lado, sólo la OBSERVA contra el rango viable que su
    propio organelo declara. Regular sin meta y medir contra un rango son cosas distintas:
    la fiebre existe aunque nadie le haya dicho al cuerpo cuál es su temperatura.
    """
    centro = (lo + hi) / 2.0
    semi = (hi - lo) / 2.0
    if semi <= 0.0:
        return 1.0
    return max(0.0, 1.0 - abs(x - centro) / semi)


def salud_piso(x: float) -> float:
    """Para condiciones de UNA sola cara: la reserva ya está normalizada en [0,1], donde
    0 es la muerte y 1 está llena. Su salud es ella misma; no hay umbral que elegir."""
    return max(0.0, min(1.0, x))


def indice_homeostasis(fila: dict) -> dict:
    """H = promedio de la salud de cada variable interna respecto de su rango viable.

    Devuelve también cada componente, porque el defecto que dio origen a este módulo fue
    precisamente que se podía ver que H se apagaba y no POR QUÉ se apagaba. Un índice sin
    sus componentes es un diagnóstico que no se puede discutir.

    Si una variable no llega (columna ausente o NaN) NO se cuenta: promediar contra un
    cero por defecto convertiría la falta de dato en enfermedad, que es la clase de
    silencio que este proyecto ya pagó caro.
    """
    partes: dict[str, float] = {}
    for col, forma, args, nombre in VARIABLES:
        x = _num(fila.get(col))
        if x is None:
            continue
        partes[nombre] = salud_centrada(x, *args) if forma == "centrada" else salud_piso(x)
    if not partes:
        return {}                              # sin ninguna variable no se inventa un valor
    vals = list(partes.values())
    salida = {nombre: round(v, 4) for nombre, v in partes.items()}
    salida["H_homeostasis"] = round(sum(vals) / len(vals), 4)
    salida["H_peor"] = round(min(vals), 4)
    salida["H_n_variables"] = len(vals)
    return salida
