"""
Normalización: cómo «Alta», «90 km/h» y «magnitud 2,8» pasan a ser comparables.

EL PROBLEMA
-----------
SERNAGEOMIN publica peligro de aluvión en tres palabras. La DMC publica viento
en kilómetros por hora. El CSN publica magnitud. Son tres reglas distintas y no
hay forma de decidir cuál «pesa» más — es como preguntar si tres manzanas son
más que dos kilos.

LO QUE MANDA EL CANON
---------------------
MACLIMA define una función de estandarización `f(·)`, «la misma para todas las
variables», que lleva cada una a escala 0-1 «evitando que una unidad física
domine por magnitud». Es exactamente el problema de arriba. Pero el canon NO
especifica la forma de esa función — dice que exista y que sea única.

CÓMO SE RESUELVE ACÁ (y qué queda pendiente de decisión)
--------------------------------------------------------
Se parte en dos pasos, y el segundo es el que cumple la exigencia canónica:

  paso 1  ·  a_intensidad(valor, tipo)  — específico de cada variable, DECLARADO.
             Lleva el dato a una «intensidad» adimensional comparable: cuántas
             desviaciones típicas, o en qué escalón de una escala ordinal.
  paso 2  ·  f(intensidad)              — LA MISMA para todas las variables.
             Aplasta la intensidad al rango 0-1 con una curva logística.

Así ninguna unidad física domina (que es lo que el canon quiere evitar) y existe
una única f (que es lo que el canon exige), sin tener que fingir que «Alta» y
«90 km/h» se pueden meter en la misma fórmula sin traducción previa.

PENDIENTE PARA ALEXIS — no lo decido yo:
  · la pendiente `k` de la logística (hoy 1,0). Controla qué tan rápido una
    anomalía se vuelve «peligro alto». Es una decisión del modelo, no técnica.
  · si el punto medio debe ser 0 (la normalidad) o algún otro valor.
Ambos están como constantes con nombre, arriba y a la vista, para que cambiarlos
sea un gesto y no una excavación.

REGLA QUE NO SE NEGOCIA
-----------------------
Toda salida viene acompañada del NOMBRE DEL MÉTODO que la produjo. Un número
normalizado sin método declarado es un número sin auditoría, y el esquema lo
rechaza. Es la misma exigencia de trazabilidad que el propio MACC impone
(dato → regla → justificación).
"""

import math

# ── parámetros de la curva común (decisión de modelo, no técnica) ───────────
PENDIENTE_K = 1.0      # qué tan brusco es el paso de «normal» a «peligroso»
CENTRO = 0.0           # intensidad que se mapea a 0,5

# Escalas ordinales oficiales que usa el país. El valor es la posición en la
# escala, no una medida: «Alta» no es «tres veces Baja».
ESCALAS_ORDINALES = {
    # ★ SERNAGEOMIN, peligro de remoción en masa — SON CUATRO NIVELES.
    # El catastro del 15-ago-2026 verificó en el diccionario de la propia capa:
    # MT-POSOC-01=Baja · 02=Moderada · 03=Alta · 04=MUY ALTA.
    # Tanto el módulo MACLIMA como la Matriz de Infraestructura Crítica hablan
    # de tres niveles (Alta/Media/Baja). Calibrar el FEN contra tres cuando la
    # fuente publica cuatro haría que «Muy Alta» y «Alta» quedaran pegadas —
    # justo la distinción que más importa para priorizar. Queda anotado como
    # hallazgo H-15; la escala de 3 se conserva para lo que sí usa tres.
    "peligro_4": {"baja": 0, "moderada": 1, "alta": 2, "muy alta": 3},
    "peligro_3": {"baja": 0, "moderada": 1, "alta": 2},
    # MICR: FEN y FANC
    "fen_3": {"baja": 0, "media": 1, "alta": 2},
    # SENAPRED: escala de alerta
    "alerta_3": {"verde": 0, "amarilla": 1, "roja": 2},
    # MICR: prioridad estratégica
    "prioridad_5": {"muy baja": 0, "baja": 1, "media": 2, "alta": 3,
                    "muy alta": 4},
    # DMC: dos escalones de alertamiento
    "dmc_2": {"aviso": 0, "alerta": 1},
}


def f(intensidad, k=PENDIENTE_K, centro=CENTRO):
    """LA función de estandarización: intensidad adimensional → [0, 1].

    Es una logística. Se eligió porque cumple tres cosas que hacen falta:
    nunca se sale de 0-1 por más extremo que sea el dato, es monótona (más
    intensidad nunca baja el resultado), y comprime las colas — la diferencia
    entre una anomalía de 5 y una de 6 sigmas importa mucho menos que la que hay
    entre 0 y 1.
    """
    return 1.0 / (1.0 + math.exp(-k * (intensidad - centro)))


# ── paso 1: llevar cada tipo de dato a intensidad ───────────────────────────

def intensidad_ordinal(valor, escala):
    """Escala ordinal (Alta/Moderada/Baja y parientes) → intensidad.

    Se reparte la escala simétricamente alrededor del centro, de modo que el
    escalón del medio caiga en la normalidad. Con 3 niveles: -2, 0, +2. Con 5:
    -2, -1, 0, +1, +2. Así «Alta» de un peligro y «Alta» de otro pesan igual,
    que es justamente lo que permite compararlos.
    """
    if escala not in ESCALAS_ORDINALES:
        raise ValueError(f"escala ordinal desconocida: {escala!r}")
    tabla = ESCALAS_ORDINALES[escala]
    clave = str(valor).strip().lower()
    if clave not in tabla:
        raise ValueError(f"valor {valor!r} no está en la escala {escala!r} "
                         f"(admite: {sorted(tabla)})")
    posicion = tabla[clave]
    n = len(tabla)
    medio = (n - 1) / 2
    # ancho 2 por escalón en escalas de 3; se achica al crecer la escala para
    # que el extremo siempre quede en ±2
    paso = 2.0 / medio if medio else 0.0
    return (posicion - medio) * paso


def intensidad_anomalia(valor, referencia, desviacion):
    """Medida continua → intensidad, como anomalía tipificada.

    Es la misma operación que ANTermic y ANPrecip de MACLIMA: cuántas
    desviaciones típicas se aparta de lo normal PARA ESE LUGAR. Por eso 20 mm
    de lluvia pueden ser nada en Valdivia y una catástrofe en Copiapó.

    ★ ACÁ SE RESPETA EL HALLAZGO H-07: se devuelve CON SIGNO. El canon aplica
    valor absoluto y con eso mete sequía e inundación en el mismo número; para
    infraestructura eso es inservible, porque una sequía no corta un camino y un
    temporal sí. Quien quiera el comportamiento canónico que use abs() sobre el
    resultado, pero que lo haga explícito.
    """
    if desviacion is None or desviacion <= 0:
        raise ValueError("la desviación debe ser > 0 para tipificar")
    return (valor - referencia) / desviacion


def razon_contra_normal(evento_mm, normal_anual_mm, piso_mm=5.0):
    """Cuántas veces la lluvia normal de un año cayó de golpe.

    ★ Corrección propuesta por Alexis el 16-ago-2026: el peligro tiene que
    calcularse contra la precipitación promedio del lugar.

    POR QUÉ ERA NECESARIA
    ---------------------
    La versión anterior medía la excedencia como PERCENTIL contra la historia
    del propio punto, y un percentil está topado en 1. Por más sin precedentes
    que fuera el evento, no podía pasar de ahí. Resultado: en el rango alto la
    conjunción quedaba gobernada por la magnitud absoluta, y un temporal de
    130 mm en Concepción superaba a los 104 mm que destruyeron Copiapó.

    Una razón no tiene techo, y por eso sí puede expresar la diferencia:

        Copiapó, marzo 2015 : 104 mm / ~12 mm al año  =  8,7 años de golpe
        Curicó, agosto 2023 : 202 mm / ~1000 mm al año =  0,2 años de golpe

    El piso en el denominador evita que un lugar con normal casi nula produzca
    razones infinitas por un milímetro. No es una corrección cosmética: sin él,
    cualquier gota en el desierto daría el número más alto del país.

    Ojo — esta razón NO se usa sola. Por sí misma tendría el defecto simétrico:
    1 mm en Copiapó también da razón alta. Lo que la vuelve útil es ir
    multiplicada por la magnitud absoluta en `peligro()`.
    """
    return evento_mm / max(normal_anual_mm, piso_mm)


def peligro(magnitud_nacional, excedencia_local):
    """Conjunción de las dos condiciones que hacen peligrosa a la lluvia.

    Corregido el 16-ago-2026, tras fallar el ancla de Copiapó. Ver
    CORRECCION_RAREZA_PELIGRO.md para el razonamiento completo.

    EL ERROR QUE CORRIGE
    --------------------
    Antes se usaba sólo la anomalía local y se la llamaba peligro. Eso mide
    RAREZA, no peligro: en Copiapó llueven 12 mm al año, así que 8 mm en agosto
    son un evento de +3 sigmas —rarísimo— y no le hacen nada a nadie. El
    instrumento los ponía casi al nivel del aluvión que destruyó la ciudad.

    LAS DOS CONDICIONES
    -------------------
    · `magnitud_nacional` [0-1]: ¿es mucha agua en términos absolutos? Se mide
      contra la historia de TODO el país, no la del propio lugar.
    · `excedencia_local`  [0-1]: ¿supera lo que este lugar aguanta? Contra la
      historia del propio punto. Acá la anomalía sí corresponde — pero como
      proxy de «más de aquello para lo que está construido», no como «clima
      raro».

    POR QUÉ MEDIA GEOMÉTRICA Y NO PROMEDIO
    --------------------------------------
    Porque es una conjunción: si falta cualquiera de las dos, no hay peligro.
    Un promedio dejaría que 8 mm rarísimos dieran «medio peligro», y no es medio
    peligro, es ninguno. La raíz del producto se va a cero si cualquier factor
    se va a cero.

    Y sigue la forma que el propio canon usa para conjunciones: el ICSGS combina
    sus cuatro factores como √(FCN × FSS × FAS × FPI).
    """
    a = max(0.0, min(1.0, magnitud_nacional))
    b = max(0.0, min(1.0, excedencia_local))
    return round((a * b) ** 0.5, 6)


def percentil_en(valor, muestra_ordenada):
    """Qué fracción de la muestra queda por debajo del valor. Devuelve 0-1.

    Recibe la muestra YA ORDENADA porque en el uso real se pregunta miles de
    veces contra la misma muestra nacional: reordenarla cada vez sería tirar
    tiempo a la basura.
    """
    n = len(muestra_ordenada)
    if n == 0:
        return None
    lo, hi = 0, n
    while lo < hi:                      # búsqueda binaria
        medio = (lo + hi) // 2
        if muestra_ordenada[medio] < valor:
            lo = medio + 1
        else:
            hi = medio
    return lo / n


def intensidad_percentil(valor, muestra):
    """Medida continua sin climatología → intensidad, por su percentil.

    Sirve cuando no hay una referencia y una desviación confiables pero sí una
    serie histórica del mismo punto: se pregunta «de todos los valores que vi
    acá, ¿qué tan arriba está éste?». Se mapea el percentil al rango ±2 para
    que quede en la misma vara que los otros métodos.
    """
    limpia = sorted(v for v in muestra if v is not None)
    if len(limpia) < 10:
        raise ValueError(f"muestra insuficiente para percentiles: {len(limpia)}")
    debajo = sum(1 for v in limpia if v < valor)
    percentil = debajo / len(limpia)
    return (percentil - 0.5) * 4.0


# ── envoltorio: lo que usan los adaptadores ─────────────────────────────────

def normalizar(valor, metodo, **parametros):
    """Devuelve (valor_normalizado, nombre_del_metodo).

    El nombre viaja con el número a propósito: el esquema rechaza un normalizado
    sin método declarado, y así queda registrado en la base cómo se produjo cada
    cifra sin depender de que alguien se acuerde de anotarlo.
    """
    if metodo == "ordinal":
        escala = parametros["escala"]
        intensidad = intensidad_ordinal(valor, escala)
        nombre = f"ordinal[{escala}]+logistica(k={PENDIENTE_K})"
    elif metodo == "anomalia":
        intensidad = intensidad_anomalia(valor, parametros["referencia"],
                                         parametros["desviacion"])
        nombre = f"anomalia_con_signo+logistica(k={PENDIENTE_K})"
    elif metodo == "percentil":
        intensidad = intensidad_percentil(valor, parametros["muestra"])
        nombre = f"percentil(n={len(parametros['muestra'])})+logistica(k={PENDIENTE_K})"
    else:
        raise ValueError(f"método de normalización desconocido: {metodo!r}")

    return round(f(intensidad), 6), nombre


# ── confianza ───────────────────────────────────────────────────────────────

def confianza(base_fuente, cobertura=1.0, antiguedad_dias=0, vida_util_dias=None):
    """Cuánto se le puede creer a un dato, entre 0 y 1.

    Tres cosas la bajan, y las tres son razones distintas:
      · el techo de la fuente (`base_fuente`): un reanálisis no merece lo mismo
        que una estación medida;
      · la cobertura: si el mes trae 4 días de dato de 30, el número es frágil
        aunque la fuente sea buena;
      · la antigüedad: un peligro vencido hace una semana ya no describe hoy.

    Se multiplican en vez de promediarse a propósito: si cualquiera de las tres
    es mala, el resultado tiene que ser malo. Un promedio dejaría que una fuente
    excelente disimulara una cobertura pésima.
    """
    factor_tiempo = 1.0
    if vida_util_dias:
        factor_tiempo = max(0.0, 1.0 - antiguedad_dias / vida_util_dias)
    valor = base_fuente * max(0.0, min(1.0, cobertura)) * factor_tiempo
    return round(max(0.0, min(1.0, valor)), 4)


if __name__ == "__main__":
    print("f(·) — la curva común\n")
    for i in (-3, -2, -1, 0, 1, 2, 3):
        print(f"   intensidad {i:+d}  →  {f(i):.4f}")

    print("\nEscalas oficiales del país, puestas en la misma vara\n")
    for etiqueta in ("Baja", "Moderada", "Alta"):
        v, m = normalizar(etiqueta, "ordinal", escala="peligro_3")
        print(f"   SERNAGEOMIN «{etiqueta:9s}» → {v:.4f}   [{m}]")
    for etiqueta in ("Verde", "Amarilla", "Roja"):
        v, _ = normalizar(etiqueta, "ordinal", escala="alerta_3")
        print(f"   SENAPRED    «{etiqueta:9s}» → {v:.4f}")

    print("\n★ H-07: la anomalía conserva el signo — sequía ≠ inundación\n")
    for mm, etiqueta in ((0.0, "mes seco"), (12.0, "mes normal"),
                         (109.0, "Copiapó marzo 2015")):
        v, m = normalizar(mm, "anomalia", referencia=12.0, desviacion=18.0)
        print(f"   {etiqueta:22s} {mm:6.1f} mm → {v:.4f}")
    print("   (con valor absoluto, el mes seco daría lo mismo que uno lluvioso)")

    print("\nConfianza\n")
    print(f"   estación medida, cobertura completa, dato de hoy : "
          f"{confianza(0.95, 1.0, 0, 7)}")
    print(f"   reanálisis, cobertura 40%, dato de hoy           : "
          f"{confianza(0.70, 0.4, 0, 7)}")
    print(f"   peligro vencido hace 5 días (vida útil 7)        : "
          f"{confianza(0.95, 1.0, 5, 7)}")
