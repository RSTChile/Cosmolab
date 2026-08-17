#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ESCALA — comparar cada magnitud con lo que es habitual PARA ESTE ORGANISMO
================================================================================
Por qué existe. Una auditoría de siete agentes sobre 237 constantes numéricas
(4-ago-2026) encontró que 168 violan la norma del proyecto: son números que
deciden qué le pasa al organismo comparados contra una escala que nadie midió.
Y su primera recomendación de trabajo fue ésta, literal:

    «Un módulo compartido, no 168 parches. Si el patrón se escribe 168 veces,
     en tres meses hay 168 variantes.»

Tenían razón: el 3 y 4 de agosto escribí este mismo cálculo cuatro veces a mano
—en el paladar del metabolismo, en el reparto de RC_A, en el recurso `es_norm` y
en los detectores de la bitácora— y ya empezaban a divergir en el arranque y en
la constante de tiempo. Aquí vive una sola vez.

LA REGLA. `rel(x, escala)` devuelve dónde cae `x` respecto de lo habitual del
organismo, con el mapeo r/(1+r): exactamente 0,5 cuando el valor es el de
siempre, sube hacia 1 cuando es mucho mayor, baja hacia 0 cuando es mucho menor.
No hay constante que elegir: la escala la pone la historia.

DOS ADVERTENCIAS QUE LA AUDITORÍA DEJÓ POR ESCRITO Y QUE HAY QUE RESPETAR:

1. `r/(1+r)` NO es universal. Cuando existe otra magnitud DEL PROPIO ORGANISMO
   con las mismas unidades, se compara contra ella y no contra la propia
   historia: la reserva contra el precio de acuñar, el gasto contra la ingesta,
   el efecto contra su línea base. La media móvil es el recurso cuando no hay
   nada mejor, no la respuesta por defecto.

2. Un consigo-mismo aplicado a un SETPOINT HOMEOSTÁTICO borra el estado
   sostenido: un organismo crónicamente hambriento leería «hambre normal» y
   dejaría de registrar que se muere. Hay variables que DEBEN poder quedarse
   mal. Antes de relativizar una, preguntar si es una percepción (relativa, sí)
   o una condición de viabilidad (absoluta, no).

ARRANQUE. Mientras no haya observaciones suficientes, `rel` devuelve `neutro`
(0,5) y `madura` es False: quien clasifique debe abstenerse en vez de inventar.
Si cada módulo improvisa su propio calentamiento, acabamos de crear una familia
nueva de constantes escondidas — que es justo lo que este módulo evita.

CORRECCIÓN 8-ago-2026 — ERA UN PROMEDIO DE TODA LA VIDA, NO UNA MEDIA MÓVIL.
El peso de cada observación se calculaba `min(tasa, 1/n)`. Con `min`, en cuanto
`n > 20` el peso deja de ser `tasa` y pasa a ser `1/n`, encogiendo para siempre:
lo que salía no era una media móvil sino el promedio acumulado de toda la vida,
cada vez más rígido, con `tasa` inerte. El docstring de `observar` afirmaba «los
primeros pasos pesan más» y hacía exactamente lo contrario. Medido en producción:
la escala del metabolismo (`_escala_RC`, la que gobierna el letargo) llevaba
77.176 observaciones creyendo que lo normal era 0,0560 cuando el valor real de
dos días era 0,1126 — aprendía 3.859 veces más lento de lo que su propia `tasa`
declaraba, y desde ahí necesitaba 233 horas de vida continua, sin un apagón, para
alcanzar el 90 % del régimen en curso. El precio: `_letargo = 2·rel(RC, escala)`
daba 1,265 en vez de 0,935, o sea un 35 % de gasto de más, permanente, en actuar
y cantar. El arreglo es un carácter: `max`.

EL CONTRATO NUEVO — `tasa` Y `minimo` SON LA MISMA IDEA, Y VAN LIGADAS.
Con `a = max(tasa, 1/n)` la escala tiene exactamente dos tramos, y la frontera
cae donde debe si y sólo si `tasa = 1/minimo`:

    n = 1            → a = 1,0        la media ES la primera observación
    n ≤ minimo       → a = 1/n        promedio aritmético del calentamiento
    n > minimo       → a = tasa       EMA de ventana `minimo`

Por eso el constructor deriva una de la otra en vez de aceptar dos números que
puedan divergir: `minimo` (cuántas observaciones) es el único mando, y `tasa` es
su recíproco. Si se declaran ambos, manda `minimo`. Y `madura` deja de ser un
umbral aparte: es justo el paso en que termina el calentamiento.

Esto arregla de paso el arranque en frío. `media` nacía en 0,0, así que una
escala nueva creía que lo habitual era CERO y al organismo recién nacido todo le
parecía exceso; con peso 1,0 en `n=1` la primera observación fija la escala.

OBSERVAR CERO NO ES OBSERVAR. Un régimen largo de ceros (`lateralidad` en
silencio, `RC_total` sin campo) multiplicaba la media por (1−tasa) en cada paso
hasta ~1e-121: al volver la señal, o saltaba el guarda de `rel` y todo era
NEUTRO, o —peor— la media quedaba en el casi-cero donde el guarda no salta y
`rel` devolvía 1,0, «todo es exceso», durante una ventana entera. La respuesta no
es poner un suelo (sería otra constante inventada): es que un cero no enseña
nada. `lo habitual` de una magnitud es cuánto vale CUANDO ESTÁ, no cuántas veces
faltó. La ausencia ya está contestada aguas abajo sin necesidad de aprenderla:
`rel(0, escala)` vale 0 por construcción. Así que `observar(0)` no cuenta, no
mueve la media y no acerca la madurez.

CUIDADO — LO QUE ESTE ARREGLO DESTAPA (advertencia 2, ahora en serio). Con `min`
un organismo en hambre crónica tardaba ~5.000 pasos en leer «hambre normal»; con
`max` tarda ~50. La protección de antes era accidental y temporal —también acababa
cediendo—, así que la respuesta correcta NO es volver a `min`: es no meter un
setpoint homeostático dentro de una `Escala`. Hay exactamente dos, y hay que
sacarlos de aquí antes de desplegar esto:

  · organelos/VST_OrganoComunicacion.py:1002-1008 — `met_energia`, LA RESERVA, es
    una de las tres patas del arousal y pasa por `rel(x, Escala)` (la escala se
    alimenta en la línea 325). La reserva es lo que separa vivir de morir:
    relativizada, el hambre crónica se lee «normal» y el organismo canta
    activación normal mientras se muere. Se arregla con `rel_contra(met_energia,
    precio)`, que es lo que el propio archivo ya hace bien en las líneas 1189 y
    1345 para decidir si puede acuñar o emular una voz.

  · genoma/VST_Bloque08_DinamicaEvolutiva.py:140-145 — `e_R` relativizado contra
    su propia media para decidir SI MUTAR. Un organismo crónicamente mal acoplado
    acabaría leyendo «error normal» y dejaría de mutar justo cuando más lo
    necesita. Se arregla comparando `e_R` contra una magnitud real con sus mismas
    unidades (`rel_contra`), no contra su historia.

`analisis/esc_regresion.py` los reimprime con el porqué y mide, sobre los CSV ya
grabados, cuánto se le mueve la decisión a CADA consumidor de este módulo.
"""
from __future__ import annotations

NEUTRO = 0.5          # ni alto ni bajo: lo que devuelve una escala sin historia


class Escala:
    """Lo habitual de UNA magnitud, aprendido de la propia experiencia del organismo.

    UN SOLO MANDO, DOS NOMBRES. `minimo` es cuántas observaciones dura el
    calentamiento; `tasa` es la constante de tiempo del aprendizaje. Son la misma
    idea escrita al derecho y al revés — `tasa = 1/minimo` — y el constructor las
    liga para que no puedan divergir, porque el tramo de promedio aritmético y el
    de EMA sólo se empalman sin escalón cuando coinciden.

    Si se declaran las dos y no concuerdan, manda LA VENTANA MÁS LARGA. No es un
    capricho: `madura` afirma que la escala ya significa algo, y no puede decirlo
    antes de haber visto una constante de tiempo entera. Así `Escala(tasa=1/100)`
    con el `minimo` por omisión mide de verdad sobre 100 pasos en vez de darse por
    madura a los 20 con la media todavía a medio hacer.
    """

    __slots__ = ("tasa", "minimo", "n", "media", "dispersion")

    def __init__(self, tasa: float | None = None, minimo: int | None = None) -> None:
        # 20 observaciones = 1/0,05: el valor de siempre, ahora escrito una sola vez.
        ventana = 20
        cand = []
        if tasa is not None:
            try:
                t = float(tasa)
            except (TypeError, ValueError):
                t = 0.0
            if t > 0.0:
                cand.append(int(round(1.0 / t)))
        if minimo is not None:
            try:
                cand.append(int(minimo))
            except (TypeError, ValueError):
                pass
        if cand:
            ventana = max(cand)                  # la más larga de las dos declaradas
        self.minimo = max(1, ventana)
        self.tasa = 1.0 / self.minimo            # derivada, nunca declarada aparte
        self.n = 0
        self.media = 0.0
        self.dispersion = 0.0

    def observar(self, x: float) -> "Escala":
        """Incorpora una observación. Devuelve self para poder encadenar.

        El peso es `max(tasa, 1/n)`: 1,0 en la primera observación —la media NACE
        siendo el primer valor visto, no cero—, luego `1/n` mientras dura el
        calentamiento (promedio aritmético de las primeras `minimo`) y `tasa` a
        partir de ahí (EMA de ventana `minimo`). Con `min` en vez de `max` esto era
        el promedio de toda la vida: a los 100.000 pasos cada observación nueva
        pesaba 1/100.000 y la escala ya no aprendía nada. Ver la cabecera.

        UN CERO NO ENSEÑA. Cuando la señal falta —silencio, campo apagado— no hay
        magnitud que promediar, y aprender esos ceros hunde la media hasta el
        casi-cero donde `rel` responde 1,0 a cualquier cosa. La ausencia no cuenta,
        no mueve la media y no acerca la madurez; `rel(0, escala)` ya vale 0 solo.
        """
        try:
            v = float(x)
        except (TypeError, ValueError):
            return self
        if v != v:                      # NaN
            return self
        if v == 0.0:                    # ausencia de la magnitud, no un valor suyo
            return self
        self.n += 1
        a = max(self.tasa, 1.0 / max(1, self.n))     # los primeros pasos pesan más
        self.media += a * (v - self.media)
        self.dispersion += a * (abs(v - self.media) - self.dispersion)
        return self

    @property
    def madura(self) -> bool:
        """¿Tiene historia suficiente para que 'lo habitual' signifique algo?

        Ya no es un umbral aparte: es exactamente el paso en que el peso deja de ser
        `1/n` y pasa a ser `tasa`, o sea donde acaba el calentamiento.
        """
        return self.n >= self.minimo

    def snapshot(self) -> dict:
        return {"n": self.n, "media": self.media, "dispersion": self.dispersion}

    def restore(self, d: dict) -> "Escala":
        """Sin esto el organismo reaprende en cada arranque qué es 'mucho' y qué 'poco',
        y sus primeras decisiones del día quedan tomadas contra una escala vacía.

        No hace falta migrar los estados grabados bajo el defecto anterior. Una media
        heredada con `n` enorme sólo era rígida porque el peso valía `1/n`; con el peso
        en `tasa` la escala se corrige sola en una ventana. El caso peor medido es
        `_escala_RC`, guardada con n=103.058 y media 0,0692 cuando lo real ronda 0,11:
        alcanza el 90 % del régimen en unos 45 pasos en vez de en 233 horas."""
        if isinstance(d, dict):
            self.n = int(d.get("n", 0) or 0)
            self.media = float(d.get("media", 0.0) or 0.0)
            self.dispersion = float(d.get("dispersion", 0.0) or 0.0)
        return self


def rel(x: float, escala: "Escala | float | None") -> float:
    """Dónde cae `x` respecto de lo habitual. 0,5 = lo de siempre. Sin parámetro libre.

    Acepta una `Escala` o directamente el valor habitual (un float), para los sitios
    que ya llevan su propia media móvil y todavía no migran.
    """
    hab = escala.media if isinstance(escala, Escala) else escala
    try:
        v = max(0.0, float(x))
    except (TypeError, ValueError):
        return NEUTRO
    if not hab or hab <= 1e-12:
        return NEUTRO
    r = v / float(hab)
    return r / (1.0 + r)


def rel_contra(x: float, referencia: float) -> float:
    """Comparar contra OTRA MAGNITUD DEL ORGANISMO con las mismas unidades.

    Ésta es la forma preferida cuando existe: la reserva contra el precio de acuñar,
    el gasto contra la ingesta, el efecto contra su línea base. Es más fuerte que la
    media móvil porque no depende de la historia, sino de una relación real entre dos
    cosas que el organismo tiene ahora mismo.
    """
    try:
        v = max(0.0, float(x)); r = float(referencia)
    except (TypeError, ValueError):
        return NEUTRO
    if r <= 1e-12:
        return NEUTRO
    q = v / r
    return q / (1.0 + q)


def clasificar(x: float, escala: "Escala") -> str:
    """'alto' / 'normal' / 'bajo' / 'indeterminado' mientras no haya historia.

    La banda no se elige: es la dispersión propia de la magnitud. Y devolver
    'indeterminado' en vez de un valor por defecto es lo que evita que el organismo
    tome decisiones recién nacido contra una escala que todavía no tiene.
    """
    if not isinstance(escala, Escala) or not escala.madura:
        return "indeterminado"
    d = escala.dispersion
    if d <= 1e-12:
        return "normal"
    z = (float(x) - escala.media) / d
    return "alto" if z > 1.0 else ("bajo" if z < -1.0 else "normal")
