#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GLOSARIO — el nombre en castellano de cada sigla del organismo
================================================================================
Por qué existe (3-ago-2026, ampliado el 8-ago-2026): la fisiología publica 369
columnas con nombres como `Lambda_Cos`, `e_R`, `OI`, `XE`, `C_b_norm` o `A_sys_env`.
Auditar un umbral exige saber QUÉ mide la variable que compara, y con siglas no se
puede: llevamos meses arrastrando números puestos a mano precisamente porque nadie
podía leer contra qué se comparaban. Y el dueño del proyecto lo dijo más corto:
«el organismo tiene tantas variables y siglas que simplemente me confundo y no
entiendo qué estás midiendo o qué significa».

LAS SIGLAS NO SE RENOMBRAN. Están en inglés, están en el CSV, están en la UI y en
media docena de organelos: cambiarlas rompería todo. Lo que se hace es ponerles al
lado su nombre descriptivo en castellano, su definición, su unidad y su rango.

QUÉ HAY AQUÍ (el contrato, estable):

    NOMBRE[sigla]     -> nombre descriptivo en castellano
    DEFINICION[sigla] -> una frase: qué mide y por qué importa
    UNIDAD[sigla]     -> "fraccion" "porcentaje" "grados" "conteo" "rms"
                         "acumulador" "adimensional" "booleano" "texto"
                         "segundos" "hz"
    RANGO[sigla]      -> (min, max) DECLARADO en el código; si el código no
                         promete nada, el MEDIDO sobre la historia real
    NODO[sigla]       -> el nodo de la Teoría que lo sustenta ("O-N9.14"…) o ""

    describir(col)        -> dict con los cinco campos; nunca lanza excepción
    es_fraccion(col)      -> True si la variable ES una fracción de algo
    formatear(col, valor) -> la representación legible en castellano

POR QUÉ `formatear` ES EL CORAZÓN DE ESTO. Convertir a porcentaje todo lo que
parece pequeño es la manera más rápida de volver ilegible un organismo. Medido en
éste: `e_R` vale 8,7 GRADOS sobre un dominio de ±90° y como porcentaje daría
«870 % de error»; `act_fatiga` es un acumulador sin techo que llegó a 347 y daría
«34.700 %»; `voz_creadas` es un CONTEO de voces y con 21 voces daría «2.100 %».
Por eso cada sigla declara su unidad, y sólo lo que el código acota a [0,1] se
imprime como porcentaje. Lo demás se imprime con su unidad y su rango.

DE DÓNDE SALE CADA CAMPO, sin inventar nada:
  · la LISTA de columnas, de los CSV de `~/.anima/history/*/fisiologia/`;
  · el NOMBRE, la DEFINICIÓN, la UNIDAD y el RANGO DECLARADO, de LEER el código
    que produce la columna (organelos/, genoma/, campo/, web/) y sus comentarios,
    que en este proyecto son extensos y suelen decir exactamente qué es cada cosa;
  · el RANGO MEDIDO, de los datos: lo calcula `analisis/glo_rangos.py --py` y se
    pega aquí; no se escribe a mano ningún mínimo ni ningún máximo;
  · el NODO, de la cabecera del organelo o del comentario que lo cita. Cuando el
    código NO cita ningún nodo, el campo queda vacío. No se infiere.

Y LO QUE NO SE SABE SE DECLARA COMO NO SABIDO: una columna cuya definición no se
pudo leer en el código queda con `definicion=""`, y `analisis/glo_cobertura.py`
la lista. Un glosario que rellena huecos con adivinanzas es peor que no tenerlo.

Uso:
    from glosario import describir, formatear, es_fraccion, nombre
    nombre("e_R")               -> 'Error de representación'
    formatear("e_R", 8.7)       -> '8,7° (de ±90°)'
    formatear("ICR_ratio", .23456) -> '23,456 %'
    formatear("voz_creadas", 21)   -> '21 voces'
    formatear("act_fatiga", 330)   -> '330 de 347 (máx. visto)'

Medida del trabajo:
    python analisis/glo_cobertura.py     # qué cubre y qué falta
    python analisis/glo_rangos.py        # el rango real, y las fugas de lo declarado
"""
from __future__ import annotations

# ==============================================================================
# LA TABLA
# ==============================================================================
# Una sola fuente de verdad, y de ella se derivan los cinco diccionarios del
# contrato. Cinco tablas paralelas mantenidas a mano divergen: es exactamente lo
# que este proyecto ya aprendió con `escala.py` («si el patrón se escribe 168
# veces, en tres meses hay 168 variantes»).
#
#   sigla: (nombre, definición, unidad, rango_declarado_o_None, nodo)
#
# `rango_declarado` es None cuando el código NO promete ninguna cota. En ese caso
# `RANGO` toma el medido (abajo), y `rango_es_medido(sigla)` lo dice.
_TABLA: dict = {

    # ── EL RELOJ Y EL RÉGIMEN ────────────────────────────────────────────────
    "ts_real": (
        "Marca de tiempo real (reloj de pared)",
        "Instante Unix en que se escribió la fila; es la llave para cruzar la fisiología con "
        "grabaciones, transcripciones y las etiquetas de experimento, y NO es una magnitud del "
        "organismo: metida en una normalización satura cualquier escala.",
        "segundos", None, ""),
    "modo_vida": (
        "Modo de vida en que se registró la fila",
        "Régimen vital declarado al escribir la fila (continuous, basal, experimento, "
        "intervención, reposo, comunicación): separa la vida continua de la intervención.",
        "texto", None, ""),
    "t": (
        "Tiempo de vida acumulado",
        "Reloj interno del organismo, que suma dt en cada ciclo metabólico: cuánto lleva "
        "viviendo esta sesión, con independencia del reloj de pared.",
        "segundos", None, ""),

    # ── ESTADO REPRESENTACIONAL Y CAMPO ──────────────────────────────────────
    "Omega": (
        "Estado representacional (organización del campo)",
        "Nivel de organización del campo Φ, reescalado a [0,1] desde ω_A: es el estado "
        "representacional global sobre el que operan la consciencia y la libertad.",
        "fraccion", (0.0, 1.0), ""),
    "omega_A": (
        "Orientación del campo: lo que percibe",
        "Media de los dos hemisferios de entrada (ω_L+ω_R)/2, o sea lo que el organismo PERCIBE "
        "del mundo; enfrentada a la expectativa produce el gradiente.",
        "adimensional", None, ""),
    "omega_B": (
        "Referencia interna: lo que espera",
        "Estado del sistema B, que corre en silencio: la expectativa interna. Sin él no habría "
        "contra qué contrastar lo percibido y no existiría la sorpresa.",
        "adimensional", None, ""),
    "gradiente": (
        "Sorpresa: percibido menos esperado",
        "ω_A − ω_B, el desajuste entre lo que llega y lo que se esperaba; es la sorpresa que "
        "fuerza al sistema, y de ella salen el error de representación y el ruido no resuelto.",
        "adimensional", None, ""),
    "e_R": (
        "Error de representación",
        "Distancia entre la orientación que el organismo sostiene y la que su representación "
        "pide, medida sobre el mismo dominio angular de ±90° de la cabeza; sube con la novedad y "
        "con el disgusto, y sólo es viable mientras se mantenga por debajo de κ_O.",
        "grados", (-90.0, 90.0), "O-N4.1"),
    "A_sys_env": (
        "Acoplamiento con el entorno",
        "Cuán ajustado está el organismo a su nicho ahora mismo; es la variable unificadora del "
        "modelo, y el invariante κ_V exige mantenerla por encima de su piso para seguir viable.",
        "fraccion", (0.05, 1.0), "O-N2.1"),
    "presion_desacople": (
        "Presión de desacople (tensión acumulada)",
        "Integrador con fuga del error por el desacoplamiento: la tensión de NO estar acoplado, "
        "que se acumula mientras el desajuste dura y gatea el juego y el ritual.",
        "acumulador", (0.0, 500.0), "O-N2.1 · O-N4.1"),
    "C_b": (
        "Consciencia básica (distinciones registradas)",
        "Magnitud de la representación de primer orden: suma de cuánto se aparta hoy cada una de "
        "las cinco variables representables respecto de su propia costumbre.",
        "adimensional", (0.0, 5.0), "O-N5.1"),
    "C_b_norm": (
        "Consciencia básica normalizada",
        "La consciencia básica dividida por el número de representables, para poder usarla sin "
        "arrastrar su escala; es el objetivo que persigue R2 y por tanto el origen de la libertad.",
        "fraccion", (0.0, 1.0), "O-N5.1"),
    "R2": (
        "Modelo de la propia representación (2º orden)",
        "Paso bajo que persigue a la consciencia básica normalizada: el modelo que el organismo "
        "hace de su propia representación. Si R2 es mayor que cero, hay libertad estructural.",
        "fraccion", (0.0, 1.0), "O-N5.2"),
    "self_coherencia": (
        "Coherencia del sí-mismo",
        "Cuán quieto sostiene su automodelo COMPARADO con lo que suele sostenerlo: su neutro es "
        "0,5 («como de costumbre»), no 1.",
        "fraccion", (0.0, 1.0), "O-N5.3"),
    "LF_struct": (
        "Libertad funcional estructural (latente)",
        "Capacidad latente de libertad, que es R2 acotado: el cimiento del que la libertad "
        "ejercible descuenta el ruido que el organismo no consigue resolver.",
        "fraccion", (0.0, 1.0), "O-N13.8"),
    "INR": (
        "Ruido no resuelto",
        "Parte de la sorpresa que el organismo NO logra resolver; recorta directamente la "
        "libertad ejercible, porque LF_op = LF_struct·(1−INR).",
        "fraccion", (0.0, 1.0), "O-N13.8.1"),
    "LF_op": (
        "Libertad funcional ejercible",
        "La libertad que queda tras descontar el ruido no resuelto de la libertad estructural; es "
        "la que consumen la razón cosmosemiótica, la organismicidad y el invariante κ_LF.",
        "fraccion", (0.0, 1.0), "O-N13.8.1"),
    "LF_rel": (
        "Libertad ejercible contra su propio neutro",
        "La libertad ejercible leída contra su neutro estructural (0,25 = todo como siempre); es "
        "el índice que de verdad clasifica el nivel de libertad, no LF_op en crudo.",
        "fraccion", (0.0, 1.0), "O-N13.8.1"),
    "lf_nivel": (
        "Nivel en la escala de libertad (0 a 3)",
        "Escala canónica: 0 salida forzada · 1 «No sé» · 2 «Disiento» · 3 «¿Y si…?». Clasifica "
        "LF_rel, no LF_op; el suelo (nivel 0) sí es absoluto y lo pone κ_LF.",
        "conteo", (0.0, 3.0), "O-N7.1 · O-N13.8.1"),
    "juego": (
        "Juego activo (1er escalón de la libertad)",
        "Exploración libre: desacople enactuado, actuar con el significado en suspenso. Primer "
        "estadio de la genealogía de la libertad funcional.",
        "booleano", (0.0, 1.0), "O-N7.2 · O-N10.7"),
    "ritual": (
        "Ritual activo (2º escalón de la libertad)",
        "Desacople ya fijado en una estructura reproducible que no se puede negar desde dentro: "
        "el patrón repetido estabilizado.",
        "booleano", (0.0, 1.0), "O-N7.2 · O-N7.3"),
    "negacion": (
        "Negación operativa: decir «No»",
        "Declarar que la representación NO determina la acción y suspenderla; es operar sobre la "
        "propia representación, cosa distinta de una simple inhibición.",
        "booleano", (0.0, 1.0), "O-N10.1 · O-N10.2 · O-N10.13"),
    "demanda_entorno": (
        "Demanda del entorno (exigencia física)",
        "Cuánto le exige el mundo ahora mismo, derivado de la energía que entra por los dos "
        "oídos; comparada con el dominio operativo decide adaptación, exaptación y activación.",
        "adimensional", None, ""),
    "Omega_op": (
        "Dominio operativo (anchura de dominio)",
        "Anchura del dominio en el que el organismo puede operar: nace en 1 y sólo crece cuando "
        "la exaptación abre dominio nuevo consumiendo reserva.",
        "acumulador", None, "O-N8.3"),
    "XE": (
        "Exaptación (dominio nuevo abierto)",
        "Cuánto dominio nuevo ha abierto medido contra el dominio con el que nació: 0,5 significa "
        "que lo ha duplicado. Uso NUEVO de una capacidad existente; es el motor de la evolución "
        "cosmosemiótica.",
        "fraccion", (0.0, 1.0), "O-N8.3"),
    "C_m": (
        "Consciencia metacognitiva (sube en crisis)",
        "Emerge cuando el registro básico falla de forma sostenida y hay libertad para "
        "reorganizarse: convoca la reorganización, y por eso sube en crisis y baja al resolverse.",
        "fraccion", (0.0, 1.0), "O-N8.4"),
    "OI": (
        "Organismicidad integrada (cuánto ES organismo)",
        "Media de homeostasis, memoria, exaptación y libertad, penalizada por la desviación "
        "ética: 0,7 o más organismo pleno · entre 0,4 y 0,7 protoorganismo · menos, no organismal.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "Lambda_Cos": (
        "Razón cosmosemiótica (salud del cierre)",
        "Diferencia estructural por libertad, dividida por el error y ponderada por el "
        "acoplamiento: alta cuando hay diferencia, libertad y ajuste; baja cuando el error manda.",
        "adimensional", None, "C-N2.8.12"),
    "invariantes_ok": (
        "Invariantes de viabilidad cumplidos",
        "Cuántas de las seis condiciones de viabilidad se cumplen (κ_P persistencia, κ_Δ "
        "diferencia, κ_O error acotado, κ_V acoplamiento, κ_LF libertad, κ_H analizabilidad).",
        "conteo", (0.0, 6.0), ""),
    "mutacion": (
        "Variación ciega sobre el error no filtrado",
        "Perturbación aleatoria con signo emitida sobre la parte del error que escapa a lo "
        "habitual: la fuente de novedad ciega, anterior a cualquier selección.",
        "adimensional", None, "O-N8.1"),
    "adaptacion_activa": (
        "Adaptación activa (mejorar sin ganar libertad)",
        "El organismo optimiza su acoplamiento DENTRO del dominio que ya tiene, sin abrir dominio "
        "nuevo: se arregla mejor con lo mismo. No es exaptación.",
        "booleano", (0.0, 1.0), "O-N8.2"),
    "exaptacion_activa": (
        "Exaptación ocurriendo en este paso",
        "En este paso se cumplió la condición de exaptación: se amplió el dominio operativo Y se "
        "ganó libertad, habiendo reserva para pagarlo.",
        "booleano", (0.0, 1.0), "O-N8.3 · O-N8.5 · O-N8.19"),
    "activacion_latente": (
        "Demanda de activar una capacidad latente",
        "Hay déficit de capacidad porque la demanda del entorno excede el dominio operativo: es "
        "el nexo operativo de la pluripotencia.",
        "booleano", (0.0, 1.0), "O-N8.12"),
    "estructura": (
        "Orden del sonido que entra",
        "Media del orden de los dos oídos, medido como localización de la vibración en la "
        "membrana (1 menos la participación inversa). No usa FFT ni lee el espectro.",
        "fraccion", (0.0, 1.0), ""),
    "estructura_L": (
        "Orden que entra por el oído izquierdo",
        "Orden del sonido en el tímpano izquierdo: 1 si la vibración se concentra en pocos modos "
        "(sonido con forma), 0 si se reparte por toda la membrana (ruido).",
        "fraccion", (0.0, 1.0), ""),
    "estructura_R": (
        "Orden que entra por el oído derecho",
        "Orden del sonido en el tímpano derecho, medido igual que el izquierdo; con él forma "
        "`estructura`, el orden reconocido que sostiene la conversión en sentido.",
        "fraccion", (0.0, 1.0), ""),

    # ── HOMEOSTASIS DAISYWORLD (las margaritas) ──────────────────────────────
    "x_interna": (
        "Desorden interno regulado",
        "La variable interna que la homeostasis debe mantener en rango: la sube el estrés y la "
        "rama ENTROPÍA, la baja la rama ORDEN. Puede salirse un 10 % por cada lado antes de topar.",
        "adimensional", (-0.1, 1.1), "C-N5.1"),
    "en_rango": (
        "La variable interna está dentro de su rango viable",
        "1 si el desorden interno se mantiene entre su mínimo y su máximo viables; es la lectura "
        "binaria de si la homeostasis está cumpliendo su función.",
        "booleano", (0.0, 1.0), "C-N5.1"),
    "x_interna_orden": (
        "Margarita ORDEN (la rama que ordena)",
        "Población de la rama que baja el desorden interno; prospera cuando hay demasiado "
        "desorden o cuando el acoplamiento se pierde y hay que recobrar el nicho.",
        "fraccion", (0.0, 1.0), "C-N5.1 · O-N2.1"),
    "x_interna_entropia": (
        "Margarita ENTROPÍA (la rama que desordena)",
        "Población de la rama que sube el desorden interno; prospera ante la rigidez, pero sólo "
        "si el acoplamiento está sano: en crisis no se puede permitir desordenarse.",
        "fraccion", (0.0, 1.0), "C-N5.1 · O-N2.1"),
    "x_interna_esfuerzo": (
        "Esfuerzo de regular (contra-empuje neto)",
        "Cuánto tira una rama más que la otra: cero cuando no hay nada que corregir, máximo "
        "cuando una sola sostiene el equilibrio. Es el trabajo real de mantenerse en rango.",
        "adimensional", None, "C-N5.1"),
    "x_interna_costo_activo": (
        "Costo activo de regular",
        "Lo que cuesta EXTRA defenderse de la perturbación, aparte del gasto basal; el "
        "metabolismo lo cobra de verdad, porque regular contra el entorno se paga.",
        "adimensional", (0.0, 1.0), "C-N5.1"),
    "x_interna_estres": (
        "Estrés real (la entrada de toda la regulación)",
        "Sólo el error que EXCEDE lo que este organismo sostiene habitualmente, más su "
        "desacoplamiento: es a esto a lo que reaccionan las margaritas.",
        "adimensional", None, "C-N5.1 · O-N2.1"),
    "x_interna_perturb_habitual": (
        "Error habitual aprendido",
        "El error que este organismo suele sostener, aprendido de su propia historia; es el cero "
        "móvil contra el que se mide el exceso, y por tanto el estrés.",
        "adimensional", None, "C-N5.1"),
    "x_interna_perturb_exceso": (
        "Exceso de error no cerrado",
        "Cuánto error hay por encima del habitual: el halcón que corrige su rumbo da cero, el que "
        "pierde la presa sube. Es lo que el organismo NO está logrando cerrar.",
        "adimensional", None, "C-N5.1"),
    "acople_sostenido_daisy": (
        "Acople sostenido, versión Daisyworld (antigua)",
        "Si la competencia orden↔entropía sostiene el acoplamiento estable en su banda. Pese al "
        "nombre viejo (`H_homeostasis` hasta el 6-ago-2026) NO mide homeostasis interna.",
        "fraccion", (0.0, 1.0), "C-N5.1 · O-N6.1 · O-N9.14"),

    # ── CAMPO BINAURAL Y MEMBRANA ────────────────────────────────────────────
    "omega_L": (
        "Estado del campo del hemisferio izquierdo",
        "Media del campo Φ del hemisferio izquierdo tras oír el canal L: el estado "
        "representacional de ese oído, y una de las dos mitades de lo percibido.",
        "adimensional", (-1.0, 1.0), "C-N7 · O-N3.1"),
    "omega_R": (
        "Estado del campo del hemisferio derecho",
        "Media del campo Φ del hemisferio derecho tras oír el canal R; junto con el izquierdo da "
        "el gradiente lateral, que es distinto del gradiente percibido-vs-esperado.",
        "adimensional", (-1.0, 1.0), "C-N7 · O-N3.1"),
    "omega_A_L": (
        "Percibido menos esperado, oído izquierdo",
        "Desajuste entre lo que percibe el oído izquierdo y lo que su referencia interna esperaba: "
        "la sorpresa desglosada por oído, para ver cuál de los dos está desajustado.",
        "adimensional", (-2.0, 2.0), "C-N7 · O-N3.1"),
    "omega_A_R": (
        "Percibido menos esperado, oído derecho",
        "Desajuste entre lo percibido por el oído derecho y su expectativa interna; con el "
        "izquierdo permite atribuir la sorpresa a un lado concreto.",
        "adimensional", (-2.0, 2.0), "C-N7 · O-N3.1"),
    "energia_L": (
        "Energía de vibración del oído izquierdo",
        "Energía que entrega el tímpano izquierdo (media del cuadrado del desplazamiento de la "
        "membrana): cuánto se mueve ese oído. Si queda en cero, el oído no está recibiendo mundo.",
        "rms", None, "C-N7 · O-N3.1"),
    "energia_R": (
        "Energía de vibración del oído derecho",
        "Energía que entrega el tímpano derecho; es la entrada que cruza al hemisferio funcional "
        "rápido y, con la izquierda, define el balance biaural.",
        "rms", None, "C-N7 · O-N3.1"),
    "balance_LR": (
        "Balance de energía entre los dos oídos",
        "+1 si toda la entrada llega por la izquierda, −1 si por la derecha, 0 si es pareja: "
        "delata un oído muerto o un cable desconectado antes que ninguna otra columna.",
        "adimensional", (-1.0, 1.0), "C-N7 · O-N3.1"),
    "lateralidad": (
        "Separación entre los dos hemisferios",
        "Cuánto difieren los campos de los dos hemisferios. OJO: incluye el sesgo estructural de "
        "los hemisferios, no sólo la entrada; para juzgar la entrada hay que mirar el balance.",
        "adimensional", (0.0, 2.0), "C-N7 · O-N3.1"),
    "coherencia_biaural": (
        "Coherencia entre los campos de ambos oídos",
        "1 si los dos campos son idénticos, 0 si no guardan relación, −1 si son opuestos: dice si "
        "los dos oídos están viviendo la misma escena.",
        "adimensional", (-1.0, 1.0), "C-N7 · O-N3.1"),
    "campo_env": (
        "Nivel de sonido que llega al campo",
        "Nivel eficaz del audio en la ventana del campo (el mayor de los dos oídos): el nivel "
        "bruto contra el que el organismo compara lo que para él es habitual.",
        "rms", None, ""),
    "campo_F": (
        "Fuerza del sonido sobre el campo",
        "Empuje con signo que el sonido ejerce sobre el campo Φ, como desviación respecto del "
        "nivel habitual de ese oído; pasado su tope el campo pierde el pozo negativo y se clava. "
        "OJO: la cota se puede desactivar por entorno, y medido llega a 0,79, el doble de lo "
        "declarado.",
        "adimensional", (-0.3849, 0.3849), ""),
    "campo_apertura": (
        "Cuánto abre la membrana",
        "Cuánto del exterior deja entrar el organismo (su propia permeabilidad, con un paso de "
        "retraso): 1 es abrirse del todo, bajar es cerrarse cuando el mundo lo desborda.",
        "fraccion", (0.0, 1.0), ""),
    "tim_ds_L": (
        "Agitación del tímpano izquierdo",
        "Cuánto se mueve en total la membrana izquierda, antes de distinguir si ese movimiento es "
        "coherente o caótico; es la diferencia estructural física del oído.",
        "rms", None, ""),
    "tim_ds_R": (
        "Agitación del tímpano derecho",
        "Cuánto se mueve en total la membrana derecha, antes de distinguir si ese movimiento es "
        "coherente o caótico.",
        "rms", None, ""),
    "tim_estructura_L": (
        "Orden medido por el tímpano izquierdo",
        "1 si la vibración se concentra en pocos modos (sonido con forma), 0 si se reparte por "
        "toda la membrana (ruido). Es una medida de LOCALIZACIÓN, no de espectro.",
        "fraccion", (0.0, 1.0), ""),
    "tim_estructura_R": (
        "Orden medido por el tímpano derecho",
        "Misma medida de localización para la membrana derecha; con la izquierda forma el orden "
        "reconocido que sostiene la conversión de ruido en sentido.",
        "fraccion", (0.0, 1.0), ""),
    "tim_energia_L": (
        "Energía de vibración de la membrana izquierda",
        "Energía media por nodo de la membrana izquierda: el análogo físico de la sonoridad. Si "
        "queda clavada en cero, ese oído está sordo.",
        "adimensional", None, ""),
    "tim_energia_R": (
        "Energía de vibración de la membrana derecha",
        "Energía media por nodo de la membrana derecha; comparada con la izquierda es como se "
        "descubrieron oídos muertos que llevaban miles de pasos en cero.",
        "adimensional", None, ""),
    "tim_centro_L": (
        "Dónde vibra la membrana izquierda",
        "Posición a lo largo de la membrana izquierda donde se concentra la vibración: es "
        "POSICIÓN (código de lugar), no frecuencia.",
        "adimensional", None, ""),
    "tim_centro_R": (
        "Dónde vibra la membrana derecha",
        "Posición del centro de energía en la membrana derecha; con la izquierda forma el "
        "percepto que la memoria perceptual guarda y compara.",
        "adimensional", None, ""),
    "tim_lateralidad": (
        "Asimetría física entre los dos tímpanos",
        "Diferencia de energía entre las membranas izquierda y derecha: el balance medido sobre "
        "la vibración real de las membranas, no sobre el campo.",
        "adimensional", None, ""),
    "tim_flujo": (
        "Rapidez con que cambia lo transmitido",
        "Cuán rápido cambia el percepto de un paso a otro: es la estructura temporal, el alimento "
        "del hemisferio rápido (transitorios).",
        "adimensional", None, ""),
    "tim_transmitido_L": (
        "Empuje que llega al martillo izquierdo",
        "Lo que la membrana izquierda TRANSMITE de verdad hacia dentro, no lo que se agita: el "
        "empuje neto coherente sobre el martillo.",
        "rms", None, ""),
    "tim_transmitido_R": (
        "Empuje que llega al martillo derecho",
        "Lo que la membrana derecha transmite de verdad hacia dentro; con la coherencia y el "
        "reflejo forma la armonía física que se convierte en placer sensorial.",
        "rms", None, ""),
    "tim_coherencia": (
        "Coherencia de la vibración del tímpano",
        "Cerca de 1 la membrana vibra en fase como un cono rígido (graves y medios), cerca de 0 "
        "vibra fragmentada y en antifase (agudos y caos) y se cancela a sí misma.",
        "fraccion", (0.0, 1.0), ""),
    "tim_reflejo": (
        "Reflejo del estribo (protección ante lo fuerte)",
        "Tensión del músculo que rigidiza la cadena de huesecillos ante un sonido fuerte: es la "
        "aversión orgánica, el «esto es demasiado», y resta bienestar.",
        "fraccion", (0.0, 1.0), ""),
    "hemi_rapido_omega": (
        "Estado del hemisferio rápido",
        "Campo del hemisferio de constante corta, alimentado por el oído DERECHO cruzado y "
        "filtrado hacia lo transitorio: el sustrato de la novedad y de R2.",
        "adimensional", (-1.0, 1.0), ""),
    "hemi_lento_omega": (
        "Estado del hemisferio lento",
        "Campo del hemisferio de constante larga, alimentado por el oído IZQUIERDO cruzado y "
        "filtrado hacia lo sostenido: el sustrato de la lateralidad funcional.",
        "adimensional", (-1.0, 1.0), ""),
    "hemi_R2": (
        "Error de autopredicción del hemisferio rápido",
        "Cuánto falló el hemisferio rápido al predecir su propio estado tras un retardo: cero es "
        "autopredicción perfecta.",
        "adimensional", None, ""),
    "hemi_lateralidad_func": (
        "Lateralidad funcional (rápido menos lento)",
        "Separación entre la vía temporal y la sostenida: es lateralidad de PROCESO, distinta de "
        "la lateralidad espacial de los oídos.",
        "adimensional", (-2.0, 2.0), ""),
    "hemi_divergencia": (
        "Divergencia entre los hemisferios funcionales",
        "Cuánto se han separado los dos hemisferios; pasado su umbral dispara el cuerpo calloso, "
        "que vuelve a reunirlos.",
        "adimensional", (0.0, 2.0), ""),
    "hemi_calloso_activo": (
        "Cuerpo calloso activo en este paso",
        "1 si la divergencia superó el umbral y hubo acoplamiento calloso (reunión de los "
        "hemisferios), 0 si trabajaron en paralelo.",
        "booleano", (0.0, 1.0), ""),
    "hemi_integracion": (
        "Integración hemisférica",
        "Estado hemisférico ya reunido tras pasar por el calloso: la media de la vía rápida y la "
        "lenta.",
        "adimensional", (-1.0, 1.0), ""),

    # ── PROPIOCEPCIÓN: EL ORGANISMO SE SIENTE ────────────────────────────────
    "placer_sensorial": (
        "Placer sensorial (armonía física del oído)",
        "Coherencia de lo transmitido menos el reflejo de protección: el organismo DISFRUTA lo "
        "armónico y sufre lo que lo abruma. Es el canal del gusto que no pasa por la valoración.",
        "adimensional", (-1.0, 1.0), ""),
    "prop_bienestar": (
        "Bienestar sentido",
        "Suma sentida de lo que sostiene la vida (energía, acople, homeostasis, libertad, "
        "organismicidad) menos lo que la mina, más el placer sensorial: la base de toda valoración.",
        "fraccion", (0.0, 1.0), ""),
    "prop_vigor": (
        "Vigor sentido (energía más libertad)",
        "Cuánta capacidad de actuar siente el organismo en este paso, promediando su reserva "
        "energética y su libertad ejercible.",
        "fraccion", (0.0, 1.0), ""),
    "prop_acople": (
        "Acople sentido (ajuste más regulación)",
        "Cuán bien acoplado al entorno y regulado por dentro se siente, promediando el "
        "acoplamiento con el entorno y el índice de homeostasis.",
        "fraccion", (0.0, 1.0), ""),
    "prop_malestar": (
        "Malestar sentido (suma de costos)",
        "Lo que duele, agota o abruma: necesidad, error que no se cierra, ruido desviado, fatiga "
        "y presión de desacople, cada uno medido contra lo habitual en él y promediados.",
        "fraccion", (0.0, 1.0), ""),
    "prop_dW": (
        "Variación del bienestar respecto del ánimo basal",
        "Bienestar menos la línea lenta del propio ánimo: si sube, lo que está pasando ahora me "
        "MEJORA. Es la base causal del gusto por un sonido.",
        "adimensional", (-1.0, 1.0), ""),
    "prop_dW_rel": (
        "Cuán grande es ese vaivén del ánimo para él",
        "Dónde cae la magnitud del cambio de ánimo respecto de cuánto suele moverse este "
        "organismo: 0,5 es un vaivén de los de siempre, más alto es un movimiento grande para él. "
        "Es la vara con que la cara decide si hay algo que expresar.",
        "fraccion", (0.0, 1.0), ""),

    # ── LA CARA ──────────────────────────────────────────────────────────────
    "cara_valoracion": (
        "Expresión de la boca (contenta, neutra, enojada)",
        "+1 sonríe, 0 recta, −1 enojada. En el organismo web la decide el CAMBIO de bienestar "
        "respecto de su propio ánimo basal, y sólo cuando ese cambio es grande para él: la cara "
        "expresa que algo va mejor o peor de lo suyo, no un nivel absoluto. No influye en nada.",
        "adimensional", (-1.0, 1.0), ""),
    "cara_modo": (
        "Modo de la expresión de la boca",
        "De dónde sale la expresión: «neutra» en reposo, «anticipada» mientras predice cómo "
        "acabará la experiencia y «final» al cerrarla; «sentida» si el bucle la pisó con el "
        "bienestar, que es lo que hace la versión web.",
        "texto", None, ""),
    "cara_confianza": (
        "Confianza de la cara anticipada",
        "Cuántos vecinos históricos respaldan la valoración que la cara anticipa; por debajo de su "
        "piso la boca no se mueve por anticipación.",
        "fraccion", (0.0, 1.0), ""),
    "cara_t_restante": (
        "Segundos que le quedan a la expresión final",
        "Tiempo que la cara mantiene la expresión FINAL de una experiencia ya cerrada antes de "
        "volver a neutra.",
        "segundos", (0.0, 4.0), ""),
    "cara_error_prediccion": (
        "Error de predicción de la cara",
        "Diferencia entre la cara que se anticipó y la que resultó al cerrar la experiencia: la "
        "corrección de la expectativa evaluativa.",
        "adimensional", (-2.0, 2.0), ""),

    # ── RUIDO CONTEXTUAL: DE QUÉ SE ALIMENTA EL ORGANISMO ────────────────────
    "RC_total": (
        "Energía semiótica disponible (ruido contextual)",
        "Todo el ruido contextual del paso, reuniendo el que llega de la relación, el que llega "
        "del mundo y la novedad: es el alimento semiótico que se repartirá en sentido o desecho.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_externo": (
        "Ruido que entra por el oído derecho (el mundo)",
        "Ruido contextual del canal derecho: música, vídeo, mundo. Cuánta materia semiótica llega "
        "de algo que no es el par.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_relacional": (
        "Ruido que entra por el oído izquierdo (el par)",
        "Ruido contextual del canal izquierdo: la voz y el estado del otro organismo. Cuánta "
        "materia semiótica llega de la relación.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "ICR": (
        "Lo convertido en sentido",
        "Parte del ruido contextual integrada como sentido y acoplamiento: lo que efectivamente "
        "nutre al organismo en este paso.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "IRDE": (
        "Lo disipado sin convertir",
        "Parte del ruido contextual que se desvía como riesgo o desacople: lo que se pierde o "
        "daña en vez de nutrir.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "ICR_ratio": (
        "Fracción convertida en sentido",
        "Cuota del ruido metabolizada como sentido, ganada por competencia contra la rama que "
        "disipa: dice si el paso fue de nutrición o de pérdida.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "IRDE_ratio": (
        "Fracción disipada",
        "Cuota metabolizada como riesgo o desacople; complemento exacto de la anterior (suman 1) "
        "y medida directa del riesgo del paso.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_delta_salud": (
        "Cambio neto de salud semio-organísmica",
        "Suma de mejoras menos empeoramientos de organismicidad, homeostasis, razón, "
        "acoplamiento, libertad, metacognición, exaptación y error entre pasos: es lo que reparte "
        "el ruido entre sentido y desecho.",
        "adimensional", None, "O-N1"),
    "destino_RC": (
        "Destino del ruido de este paso",
        "Veredicto del reparto en una palabra: silencio si no llegó ruido, ICR si alimentó, IRDE "
        "si dañó, mixto si ambas cosas.",
        "texto", None, "O-N1"),
    "RC_atencion_L": (
        "Atención que reclama el oído izquierdo",
        "Cuánto reclama orientación el canal izquierdo, sumando saliencia, ruido relacional, "
        "energía, novedad y memoria del lazo; va antes que la comprensión y el riesgo.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_atencion_R": (
        "Atención que reclama el oído derecho",
        "Cuánto reclama orientación el canal derecho, con los mismos ingredientes que el "
        "izquierdo pero sobre el mundo externo.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_comprension_L": (
        "Comprensión del canal izquierdo",
        "Atención izquierda efectivamente convertida en sentido: es el permiso para girar hacia "
        "la izquierda.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_comprension_R": (
        "Comprensión del canal derecho",
        "Atención derecha efectivamente convertida en sentido: es el permiso para girar hacia la "
        "derecha.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_ema_comp_L": (
        "Memoria del lazo de atención izquierdo",
        "Media móvil de la comprensión izquierda: lo que nutrió antes vuelve a atraer la mirada. "
        "Es un sesgo atencional que emerge de la historia, no un parámetro puesto a mano.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_ema_comp_R": (
        "Memoria del lazo de atención derecho",
        "Media móvil de la comprensión derecha: el mismo sesgo atencional emergente por el lado "
        "del mundo.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_riesgo_L": (
        "Riesgo del canal izquierdo",
        "Atención izquierda que cae del lado de la desviación: frena el giro hacia ese lado en "
        "vez de habilitarlo.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_riesgo_R": (
        "Riesgo del canal derecho",
        "Atención derecha que cae del lado de la desviación: frena el giro hacia ese lado.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_consenso_orientacion": (
        "Consenso para orientarse (el signo dice el lado)",
        "Diferencia de comprensión entre los dos canales: negativo pide izquierda, positivo "
        "derecha, cero no hay consenso. Es la propuesta de giro que hace el ruido contextual.",
        "adimensional", (-1.0, 1.0), "O-N1"),
    "RC_confianza_comprension": (
        "Permiso de giro por comprensión",
        "Confianza global de que lo comprendido justifica orientar la cabeza; es lo que habilita "
        "la conducta.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_freno_riesgo": (
        "Freno corporal por riesgo",
        "Freno global que impone el riesgo contextual: el contrapeso exacto del permiso por "
        "comprensión.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_base_relacional": (
        "Evidencia bruta del canal izquierdo",
        "Energía, saliencia, balance y novedad del oído izquierdo antes de reunirlos: la materia "
        "prima del ruido relacional.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_base_externo": (
        "Evidencia bruta del canal derecho",
        "Energía, saliencia, incoherencia y novedad del oído derecho antes de reunirlos: la "
        "materia prima del ruido externo.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_soporte_conversion": (
        "Soporte interno para convertir en sentido",
        "Cuánto estado propio (organismicidad, homeostasis, acople, libertad, R2, metacognición, "
        "exaptación) respalda convertir el ruido en sentido: una de las dos fuerzas del reparto.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_vulnerabilidad_desviacion": (
        "Vulnerabilidad interna a disipar",
        "Cuánto empujan las carencias del organismo hacia la disipación: la otra fuerza del "
        "reparto, la que hace que el ruido se pierda en vez de alimentar.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_base_ICR": (
        "Potencia de conversión antes de normalizar",
        "Ruido disponible por el peso de conversión y por el orden del sonido: sin orden que "
        "convertir no hay sentido posible.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_base_IRDE": (
        "Potencia de disipación antes de normalizar",
        "Ruido disponible por el peso de desviación más la parte no estructurada: el ruido sin "
        "orden se disipa aunque haya disposición a convertirlo.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_peso_ICR": (
        "Peso de la competencia hacia el sentido",
        "Cuota que gana la conversión al competir las mejoras contra los empeoramientos del "
        "estado; es lo que decide el reparto.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_peso_IRDE": (
        "Peso de la competencia hacia la desviación",
        "Cuota que gana la desviación en esa misma competencia; complemento del peso de "
        "conversión.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "RC_apertura_desacople": (
        "La tensión con el entorno abrió conversión",
        "1 si la tensión entre error y desacoplamiento amplió el techo de conversión en este "
        "paso; hace falsable el mecanismo de libertad funcional. Vale 0 mientras está apagado.",
        "booleano", (0.0, 1.0), "O-N1"),

    # ── HOMEOSTASIS EMERGENTE: EL ACOPLE QUE SE SOSTIENE ─────────────────────
    "acople_sostenido": (
        "Salud homeostática real (acople sostenido)",
        "Con qué calidad la competencia sentido↔desecho sostiene el acoplamiento estable, viable "
        "y sin fuga dentro de su banda propia. Es la H canónica: no mide distancia a un objetivo.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "acople_A_estabilidad": (
        "Estabilidad del acople en su banda",
        "Cuánto fluctúa el acoplamiento respecto de la anchura que este organismo tolera: mide si "
        "se mueve dentro de lo suyo o se está yendo.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "acople_RC_vivo": (
        "Razón viva (no está anestesiado)",
        "Compuerta que impide contar como salud un acoplamiento logrado apagando la percepción: "
        "si no llega ruido, no hay mérito en estar acoplado.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "acople_competencia_ICR_IRDE": (
        "Competencia sentido↔desecho viva",
        "Vale 1 si las dos ramas están presentes y cae si una colapsa: sin las dos no hay "
        "metabolismo semiótico, sólo una máquina que repite.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "acople_recuperacion_A": (
        "Recuperación del acoplamiento",
        "Tendencia del acoplamiento a subir, con la compuerta de razón viva puesta: mide si se "
        "está reacoplando de verdad y no por dejar de registrar el entorno.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "acople_autoencierro": (
        "Patología: autoencierro",
        "La rama del orden domina mientras el acoplamiento cae: el organismo se ordena hacia "
        "adentro perdiendo el mundo. Se resta de la salud.",
        "fraccion", (0.0, 1.0), "O-N2.1"),
    "acople_anestesia": (
        "Patología: anestesia",
        "La razón cae a casi silencio mientras el acoplamiento es bajo: deja de registrar el "
        "entorno estando desacoplado. Se resta de la salud.",
        "fraccion", (0.0, 1.0), "O-N2.1"),
    "acople_banda_centro_A": (
        "Centro de la banda viable del acople",
        "Media móvil del acoplamiento que define el centro de su banda viable: la banda sale de "
        "la historia del organismo, no de un objetivo puesto a mano.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "acople_banda_var_A": (
        "Anchura de la banda del acople",
        "Cuánto fluctúa habitualmente el acoplamiento alrededor de su centro; es la vara con la "
        "que se juzga su estabilidad.",
        "adimensional", None, "O-N9.14"),
    "acople_dA_sys_env": (
        "Tendencia del acoplamiento (con signo)",
        "Variación suavizada del acoplamiento por paso: positiva si mejora, negativa si se "
        "degrada. Alimenta la recuperación, el autoencierro y el riesgo de desacople.",
        "adimensional", None, "O-N9.14"),
    "A_soporte_LF": (
        "Soporte del acople por libertad",
        "Libertad que sostiene el acoplamiento, penalizada si sube el riesgo o si el acople cae: "
        "libertad sin acoplamiento es deriva, no salud.",
        "fraccion", (0.0, 1.0), "O-N2.1"),
    "A_soporte_comprension": (
        "Soporte del acople por comprensión",
        "Comprensión de los dos canales contabilizada como soporte, y sólo si el acoplamiento no "
        "está cayendo.",
        "fraccion", (0.0, 1.0), "O-N2.1"),
    "A_soporte_confianza": (
        "Soporte del acople por confianza",
        "Confianza corporal del actuador tomada como soporte relacional directo del acoplamiento "
        "con el entorno.",
        "fraccion", (0.0, 1.0), "O-N2.1"),
    "A_soporte_S_shared": (
        "Soporte del acople por sentido compartido",
        "Memoria externalizable o coordinación con otros: lo compartido sostiene el "
        "acoplamiento.",
        "fraccion", (0.0, 1.0), "O-N2.1"),
    "A_soporte_altruismo": (
        "Soporte del acople por cooperación",
        "Disposición a cooperar, penalizada si sube la cuota de riesgo: sólo cuenta si es "
        "recíproca y no desacopla.",
        "fraccion", (0.0, 1.0), "O-N2.1"),
    "A_soporte_fatiga": (
        "Soporte del acople por energía disponible",
        "Lo contrario de la fatiga del actuador: la fatiga alta resta capacidad de sostener el "
        "acoplamiento.",
        "fraccion", (0.0, 1.0), "O-N2.1"),
    "A_soporte_RC": (
        "Soporte del acople por conversión de ruido",
        "Cuota de conversión contabilizada como soporte si el acople no cae: convertir ruido en "
        "sentido sostiene el acoplamiento sólo cuando el acoplamiento se mantiene.",
        "fraccion", (0.0, 1.0), "O-N2.1"),
    "A_soporte_total": (
        "Soporte total del acoplamiento",
        "Promedio de los siete soportes: diagnóstico global de cuánto el estado interno sostiene "
        "el acople. Mide, no manda.",
        "fraccion", (0.0, 1.0), "O-N2.1"),
    "A_riesgo_desacople": (
        "Riesgo de desacople",
        "Cuota de desecho combinada con un acoplamiento que está cayendo: señala desacoplamiento "
        "en curso, no un riesgo meramente potencial.",
        "fraccion", (0.0, 1.0), "O-N2.1"),

    # ── PERMEABILIDAD ACTIVA DE LA MEMBRANA ──────────────────────────────────
    "act_perm": (
        "Permeabilidad activa de la membrana",
        "Cuánto se abre para reacoplarse: demanda por desacople, por disposición a convertir, por "
        "energía. Abre si hay conversión viva y se repliega si manda el riesgo.",
        "fraccion", (0.0, 1.0), ""),
    "act_perm_demanda": (
        "Demanda de reacople (presión por actuar)",
        "Lo que falta para tener el acople sostenido: la presión que dispara la permeabilidad, "
        "sin objetivo ni valor deseado puestos a mano.",
        "fraccion", (0.0, 1.0), ""),
    "act_perm_modo": (
        "Disposición a abrir y convertir",
        "Cuota de conversión usada como modo de la membrana: abrir cuando hay disposición a "
        "convertir, protegerse cuando domina la desviación.",
        "fraccion", (0.0, 1.0), ""),
    "act_perm_energia": (
        "Energía disponible para abrir",
        "Lo contrario de la fatiga: sin energía no hay apertura posible aunque haya demanda y "
        "disposición.",
        "fraccion", (0.0, 1.0), ""),
    "act_perm_alpha_sugerido": (
        "Rigidez de acople sugerida (latente)",
        "Factor que el soma usaría si se cerrara este lazo conductual; por defecto NO se aplica: "
        "se mide y se publica, nada más.",
        "adimensional", (1.0, 2.0), ""),

    # ── ACTUADOR: LA CABEZA 3D ───────────────────────────────────────────────
    "act_orientacion_deg": (
        "Orientación real de la cabeza",
        "Ángulo horizontal que la cabeza ocupa de verdad tras la inercia, el temblor y la fatiga; "
        "es el que sigue la cámara.",
        "grados", (-90.0, 90.0), ""),
    "act_objetivo_deg": (
        "Orientación deseada",
        "Ángulo hacia el que el organismo quiere mirar: la brújula del sentido reconocido, vetada "
        "por el bloqueo de riesgo y sesgada por el barrido que provoca el hambre.",
        "grados", (-90.0, 90.0), ""),
    "act_delta_deg": (
        "Giro efectivo del paso",
        "Cuánto se movió la cabeza de verdad, después de la inercia y el temblor: no cuánto "
        "quería moverse.",
        "grados", None, ""),
    "act_pitch_deg": (
        "Inclinación real de la cabeza",
        "Ángulo vertical actual: sube o baja por activación organísmica, y no hay ninguna otra "
        "inclinación oculta.",
        "grados", (-22.0, 22.0), ""),
    "act_pitch_objetivo_deg": (
        "Inclinación deseada",
        "Inclinación a la que tiende el eje vertical antes de la inercia, derivada del impulso "
        "vertical.",
        "grados", (-22.0, 22.0), ""),
    "act_pitch_delta_deg": (
        "Cambio de inclinación del paso",
        "Variación vertical realmente aplicada tras la zona muerta y la inercia.",
        "grados", None, ""),
    "act_vertical_drive": (
        "Impulso vertical organísmico",
        "Cuánto empuja el cuerpo la cabeza hacia arriba o hacia abajo: presión de desacople, "
        "hambre y falta de cierre. Es la causa interna de la inclinación.",
        "fraccion", (0.0, 1.0), ""),
    "act_confianza": (
        "Confianza corporal para moverse",
        "Confianza que sale de R2, libertad, homeostasis, acoplamiento y exaptación, modulada por "
        "el permiso y el bloqueo: gatea la velocidad de giro efectiva.",
        "fraccion", (0.0, 1.0), ""),
    "act_fatiga": (
        "Fatiga motora acumulada",
        "Acumulador de esfuerzo con fuga: suma cada giro amplificado por el bloqueo y el "
        "conflicto. NO tiene techo declarado; ensancha la zona muerta y frena el movimiento.",
        "acumulador", None, ""),
    "act_zona_muerta": (
        "Zona muerta angular (umbral para moverse)",
        "Error angular por debajo del cual el organismo NO se mueve; crece con la fatiga, el "
        "bloqueo y el conflicto hasta su tope duro.",
        "grados", (2.0, 18.0), ""),
    "act_temblor_rms": (
        "Temblor motor",
        "Nivel eficaz del ruido motor de las últimas iteraciones: crece con la fatiga, el bloqueo "
        "y el conflicto. Mide inquietud corporal, no giro útil.",
        "rms", None, ""),
    "act_lateralidad_dw": (
        "Saliencia lateral neta (derecha menos izquierda)",
        "Diferencia de saliencia entre los dos oídos, ponderada por la presencia de cada canal: "
        "entra como evidencia de lado, nunca como causa directa del giro.",
        "adimensional", None, ""),
    "act_atencion_L": (
        "Atención del oído izquierdo (con presencia)",
        "Lo que reclama el canal izquierdo una vez multiplicado por la presencia del canal: un "
        "oído apagado no puede capturar atención.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "act_atencion_R": (
        "Atención del oído derecho (con presencia)",
        "Lo que reclama el canal derecho multiplicado por su presencia; sin presencia no hay "
        "demanda atencional.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "act_comprension_L": (
        "Comprensión del canal izquierdo (cruda)",
        "Comprensión atribuida al oído izquierdo tal como la publica el organelo de ruido, sin la "
        "ganancia adaptativa que el actuador sí usa por dentro.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "act_comprension_R": (
        "Comprensión del canal derecho (cruda)",
        "Comprensión atribuida al oído derecho en crudo: la base de la brújula hacia lo "
        "comprensible.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "act_riesgo_L": (
        "Riesgo del canal izquierdo (en el actuador)",
        "Riesgo atribuido al oído izquierdo; alimenta la amenaza que resta a la razón de girar "
        "hacia ese lado.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "act_riesgo_R": (
        "Riesgo del canal derecho (en el actuador)",
        "Riesgo atribuido al oído derecho; alimenta la amenaza que resta a la razón de girar "
        "hacia ese lado.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "act_consenso_RC": (
        "Consenso de orientación (alias histórico)",
        "Copia literal de la decisión organísmica, conservada por compatibilidad: positivo pide "
        "derecha, negativo izquierda.",
        "adimensional", (-1.0, 1.0), ""),
    "act_conflicto_RC": (
        "Conflicto entre atención y razón",
        "Cuánto se contradicen lo que propone la escucha lateral y lo que decide la razón: "
        "ensancha la zona muerta, el temblor y la fatiga.",
        "fraccion", (0.0, 1.0), ""),
    "act_freno_RC": (
        "Freno corporal por riesgo (en el actuador)",
        "Freno que el ruido desviado impone al cuerpo; entra en la vulnerabilidad, en la base de "
        "riesgo y en la amenaza de cada oído.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "act_rc_mix": (
        "Mezcla de ruido (alias histórico del permiso)",
        "Valor idéntico al permiso decisional; se conserva para no romper el esquema del CSV.",
        "fraccion", (0.0, 1.0), ""),
    "act_presencia_L": (
        "Presencia sensorial del oído izquierdo",
        "Compuerta suave de existencia del canal izquierdo: habilita el oído sin que el volumen "
        "dirija el giro.",
        "fraccion", (0.0, 1.0), ""),
    "act_presencia_R": (
        "Presencia sensorial del oído derecho",
        "Compuerta suave del canal derecho; si las dos presencias caen, toda la decisión se pone "
        "a cero y la cabeza se centra.",
        "fraccion", (0.0, 1.0), ""),
    "act_propuesta_atencional": (
        "Propuesta de la escucha lateral",
        "Diferencia de evidencia entre oídos: lo que la escucha propone ANTES de que la razón "
        "decida. Su choque con la decisión define el conflicto.",
        "adimensional", (-1.0, 1.0), ""),
    "act_decision_RC": (
        "Decisión motora final",
        "Decisión organísmica ya filtrada por el permiso y por el bloqueo de riesgo: lo que el "
        "cuerpo puede convertir de verdad en giro.",
        "adimensional", (-1.0, 1.0), ""),
    "act_bloqueo_IRDE": (
        "Bloqueo por riesgo y desacople",
        "Veto que el ruido desviado ejerce sobre la decisión y el objetivo: degrada el giro, no "
        "lo invierte.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "act_permiso_decisional": (
        "Permiso endógeno para orientarse",
        "Media geométrica de confianza, comprensión e integración interna: sin permiso, la "
        "comprensión no mueve la cabeza.",
        "fraccion", (0.0, 1.0), ""),
    "act_evidencia_L": (
        "Evidencia sensorial izquierda",
        "Atención y saliencia del oído izquierdo promediadas: la lateralidad entra como dato que "
        "habilita, nunca como causa del giro.",
        "fraccion", (0.0, 1.0), ""),
    "act_evidencia_R": (
        "Evidencia sensorial derecha",
        "Atención y saliencia del oído derecho; con la izquierda define la claridad del estímulo "
        "y la propuesta atencional.",
        "fraccion", (0.0, 1.0), ""),
    "act_razon_L": (
        "Razón para mirar a la izquierda",
        "Evidencia izquierda multiplicada por lo que atrae menos lo que amenaza: un motivo "
        "corporal para orientarse a ese lado, no un reflejo.",
        "adimensional", (-1.0, 1.0), ""),
    "act_razon_R": (
        "Razón para mirar a la derecha",
        "Evidencia derecha por lo que atrae menos lo que amenaza; su resta con la izquierda "
        "produce la decisión organísmica.",
        "adimensional", (-1.0, 1.0), ""),
    "act_necesidad_cierre": (
        "Necesidad de recuperar el cierre",
        "Cuánta urgencia hay de volver a acoplarse: falta de organismicidad, de homeostasis y de "
        "acople, más error, metacognición de crisis y falta de libertad.",
        "fraccion", (0.0, 1.0), ""),
    "act_decision_organismica": (
        "Decisión antes del permiso y del freno",
        "Lo que el organismo decidiría si el cuerpo no pusiera permiso ni freno; comparada con la "
        "decisión final muestra el coste corporal de decidir.",
        "adimensional", (-1.0, 1.0), ""),
    "act_soporte_sentido": (
        "Soporte para ir hacia el sentido",
        "Cuánto sostiene el cuerpo el ir hacia lo comprensible: organismicidad, homeostasis, "
        "metacognición, libertad, integración y confianza de comprensión.",
        "fraccion", (0.0, 1.0), ""),
    "act_vulnerabilidad_riesgo": (
        "Vulnerabilidad que habilita el freno",
        "Cuánto derecho tiene el riesgo a frenar ahora: desorganización, poca homeostasis, poca "
        "libertad, error, fatiga y freno acumulado.",
        "fraccion", (0.0, 1.0), ""),
    "act_base_sentido": (
        "Fuerza bruta del lado del sentido",
        "Conversión por soporte por comprensión y evidencia disponibles, antes de normalizar: la "
        "potencia cruda del polo que atrae.",
        "fraccion", (0.0, 1.0), ""),
    "act_base_riesgo": (
        "Fuerza bruta del lado del riesgo",
        "Desviación por vulnerabilidad por riesgo, freno y desintegración, antes de normalizar: "
        "la potencia cruda del polo que retiene.",
        "fraccion", (0.0, 1.0), ""),
    "act_peso_sentido": (
        "Peso emergente del sentido",
        "Cuota que se lleva el polo que atrae al competir contra el que retiene: la ponderación "
        "nace del estado del organismo y no de un coeficiente fijo.",
        "fraccion", (0.0, 1.0), ""),
    "act_peso_riesgo": (
        "Peso emergente del riesgo",
        "Cuota que se lleva el polo que retiene; complementaria de la anterior salvo cuando las "
        "dos bases son nulas y ambas valen cero.",
        "fraccion", (0.0, 1.0), ""),
    "act_comp_gain_eff": (
        "Ganancia adaptativa de la comprensión",
        "Cuánto amplifica la comprensión de cada oído; no está fijada a mano: sube si la atención "
        "sostenida no daña el acoplamiento y no dispara el desecho.",
        "adimensional", (1.0, 8.0), ""),
    "act_k_motor_eff": (
        "Velocidad de giro efectiva",
        "Velocidad máxima fisiológica ya filtrada por confianza, permiso, ausencia de bloqueo, "
        "ausencia de fatiga y persistencia, y energizada por el hambre.",
        "adimensional", None, ""),
    "act_persistencia_decision": (
        "Persistencia del signo de la decisión",
        "1 si el organismo insiste en un lado, 0 si oscila y se cancela a sí mismo: mide "
        "constancia, no intensidad.",
        "fraccion", (0.0, 1.0), ""),
    "act_claridad_estimulo": (
        "Claridad del estímulo lateral",
        "Asimetría normalizada de la evidencia entre oídos: alta con un blanco lateral nítido, "
        "cerca de cero con una escena simétrica o ambigua.",
        "fraccion", (0.0, 1.0), ""),
    "act_error_motor": (
        "Error angular pendiente",
        "Distancia angular que aún falta por girar entre el objetivo y la orientación actual.",
        "grados", None, ""),
    "act_mejora_motor": (
        "Mejora angular respecto del paso anterior",
        "Positiva si el movimiento acercó la cabeza a su objetivo, negativa si lo empeoró; entra "
        "como castigo en la adaptación motora.",
        "grados", None, ""),
    "act_adaptacion_motor": (
        "Cuánto recortó el cuerpo su capacidad de girar",
        "Diferencia entre la velocidad efectiva y el máximo fisiológico: negativa si el cuerpo se "
        "frenó, positiva si el hambre lo empujó por encima.",
        "adimensional", None, ""),
    "act_adaptacion_comprension": (
        "Cambio de la ganancia de comprensión",
        "Cuánto subió o bajó la ganancia en este paso: mide si el cierre del organismo premió o "
        "castigó amplificar la comprensión.",
        "adimensional", None, ""),

    # ── METABOLISMO: COMER, GASTAR, DIGERIR ──────────────────────────────────
    "met_costo_extra": (
        "Costo extra de hablar o acuñar una palabra",
        "Gasto del aparato fonador acumulado desde la última lectura, que el bucle inyecta al "
        "metabolismo: crear una voz propia cuesta de verdad, y aquí se ve cuánto.",
        "acumulador", None, ""),
    "met_energia": (
        "Reserva de energía",
        "La reserva viva tras integrar lo comido menos lo gastado: 0 es la muerte y 1 está llena. "
        "Es una condición de viabilidad ABSOLUTA y nunca se relativiza contra la propia historia.",
        "fraccion", (0.0, 1.0), ""),
    "met_IM": (
        "Índice metabólico sostenido (nutre o intoxica)",
        "Saldo sostenido entre lo convertido y lo disipado, suavizado como una digestión: "
        "positivo la experiencia alimenta, negativo se disipa.",
        "adimensional", (-1.0, 1.0), "O-N1"),
    "met_clase": (
        "Veredicto del bocado (1 nutritiva · 0 neutra · −1 tóxica)",
        "Código del veredicto que emite la digestión sobre lo comido, decidido por el balance "
        "sostenido contra la banda del costo basal. En el CSV es un número, no una palabra.",
        "adimensional", (-1.0, 1.0), ""),
    "met_ingesta": (
        "Lo que comió",
        "Lo que entra en el paso: nutrición semiótica, aporte fotosintético y negentropía captada "
        "por radio. Es el lado de entrada del balance.",
        "adimensional", None, "O-N1"),
    "met_gasto": (
        "Lo que gastó en vivir",
        "Lo que cuesta el paso: el basal irreducible más el trabajo, la voz, el costo de regularse "
        "y la desincronía, modulado por el letargo y por la toxicidad de lo comido.",
        "adimensional", None, ""),
    "met_balance": (
        "Balance: comió menos gastó",
        "Con signo: es el juez que decide si el bocado alimentó o intoxicó, y lo que mueve la "
        "reserva hacia arriba o hacia abajo.",
        "adimensional", None, ""),
    "met_costo_homeostasis": (
        "Costo de sostener el equilibrio interno",
        "Lo que cuesta defenderse de una perturbación: cero en calma y sube al regular. Regular "
        "contra el entorno se paga, y más cuanto más empuja.",
        "adimensional", (0.0, 0.006), "C-N5.1"),
    "met_hambre": (
        "Hambre",
        "Cuánto le falta al organismo para estar lleno: es exactamente lo contrario de la reserva.",
        "fraccion", (0.0, 1.0), ""),
    "met_saciedad": (
        "Saciedad general",
        "Estar saciado no es un contador aparte: es TENER RESERVA. Publica el mismo número que la "
        "reserva de energía, y estando llena baja el rendimiento de lo que se come.",
        "fraccion", (0.0, 1.0), ""),
    "met_estructura": (
        "Orden del alimento",
        "Orden reconocido en lo que entra por la membrana (1 sonido con forma, 0 ruido). Ya no "
        "entra en el cálculo: se publica para poder detectar si el orden se cobra dos veces.",
        "fraccion", (0.0, 1.0), ""),
    "met_preferencia": (
        "Preferencia por el alimento de este paso",
        "Gusto aprendido para el tipo de alimento actual, entrenado por el impacto real del "
        "bocado; con signo, para poder apartarse de lo que daña.",
        "adimensional", (-1.0, 1.0), ""),
    "met_nutricion": (
        "Calidad nutritiva del bocado",
        "Cuánto de lo disponible se convirtió en sentido, descontando el hambre general y la "
        "saciedad de ese tipo concreto; es lo que la memoria lee para SACIAR la necesidad.",
        "fraccion", (0.0, 1.0), "O-N1"),
    "met_alimento_modo": (
        "Regla vigente de qué cuenta como alimento",
        "«conversion» (la canónica: alimenta lo ya convertido en sentido) o «duelo» (la antigua, "
        "que exigía ganar la competencia).",
        "texto", None, "O-N1"),
    "met_pref_top": (
        "Alimento favorito del paladar",
        "Clave del tipo de alimento con mayor preferencia entre todas las modalidades: lo que "
        "este organismo más gusta de comer.",
        "texto", None, ""),
    "met_pref_top_val": (
        "Cuánto le gusta su alimento favorito",
        "Valor del favorito del paladar, en la misma escala con signo de la preferencia.",
        "adimensional", (-1.0, 1.0), ""),
    "met_impacto": (
        "Impacto real del bocado (con signo)",
        "Qué parte de lo que costó vivir el paso alcanzó a pagar el bocado: +1 se pagó con "
        "creces, 0 salió a mano, −1 hubo que poner reserva.",
        "adimensional", (-1.0, 1.0), ""),
    "met_clave": (
        "Tipo de alimento que está comiendo",
        "Firma del bocado: de qué modalidad viene (mundo o voz del otro), por qué lado entra y "
        "cuánto orden trae, todo medido contra la propia historia del organismo.",
        "texto", None, ""),
    "met_saciedad_tipo": (
        "Saciedad de este tipo de alimento",
        "Qué parte de una ración lleva comida DE ESE TIPO: empuja a variar la dieta sin que nadie "
        "se lo ordene (saciedad sensorial específica).",
        "fraccion", (0.0, 1.0), ""),
    "met_tipos_n": (
        "Tipos de alimento aprendidos",
        "Cuántas claves distintas de alimento tiene en su paladar: la amplitud de su repertorio "
        "dietético.",
        "conteo", None, ""),
    "met_paladar": (
        "Paladar: preferencia por tipo de alimento",
        "Los ocho tipos de alimento mejor valorados con su preferencia, de lo que más gusta a lo "
        "que menos.",
        "texto", None, ""),
    "met_reloj_fase": (
        "Fase dentro del segundo del reloj externo",
        "Posición dentro del segundo que marca el latido GPS: el reloj del cielo entra como "
        "propiocepción, no como mandato.",
        "fraccion", (0.0, 1.0), ""),
    "met_reloj_deriva": (
        "Deriva del tempo interno frente al cielo",
        "Segundos internos por segundo externo, menos uno: 0 es sincronía, positivo va adelantado "
        "y negativo atrasado.",
        "adimensional", None, ""),
    "met_reloj_confianza": (
        "Hay reloj externo confiable",
        "1 si llegó cuenta de pulsos del reloj externo y 0 si no; sin ella la desincronía no se "
        "cobra.",
        "booleano", (0.0, 1.0), ""),
    "met_tempo_estres": (
        "Estrés de desincronía temporal",
        "Magnitud de la deriva acotada, y puesta a cero si no hay reloj confiable: es lo que el "
        "metabolismo cobra como costo suave de ir a destiempo.",
        "fraccion", (0.0, 1.0), ""),

    # ── MEMORIA E HISTORIA INTERNA ───────────────────────────────────────────
    "mem_familiaridad": (
        "Familiaridad de lo que oye",
        "Cuánto se parece el campo actual a su propia historia, medido contra la divergencia que "
        "este organismo suele tener.",
        "fraccion", (0.0, 1.0), ""),
    "mem_novedad": (
        "Novedad de lo que oye",
        "Lo contrario de la familiaridad: cuán nuevo le resulta el momento que está viviendo.",
        "fraccion", (0.0, 1.0), ""),
    "mem_carga_estructural": (
        "Carga de la memoria implícita del cuerpo",
        "Cuánto ha grabado el cuerpo en sus conexiones, que no es lo mismo que lo que recuerda "
        "explícitamente.",
        "adimensional", None, ""),
    "mem_valencia_estado": (
        "Si esta situación le sentó bien o mal",
        "Valencia afectiva del estado, con doble escala (el hábito de segundos y la identidad de "
        "minutos) y con el impacto metabólico de maestro. Sin signo no habría aversión.",
        "adimensional", (-1.0, 1.0), ""),
    "mem_persistencia": (
        "Permanencia del objeto en su ausencia",
        "Confianza de que lo que dejó de percibirse sigue ahí, con una constante que crece con la "
        "vida vivida: es lo que permite echar algo de menos.",
        "fraccion", (0.0, 1.0), ""),
    "mem_recall": (
        "Hubo evocación explícita",
        "1 si la situación actual coincide con algún episodio guardado y se evocó; 0 si no hay "
        "nada que recordar de este estado.",
        "booleano", (0.0, 1.0), ""),
    "mem_recall_tipo": (
        "Tipo de recuerdo evocado (0 ninguno · 1 amenaza · 2 logro · 3 novedad · 4 neutro)",
        "Código del episodio recuperado: da el color biográfico del recuerdo. En el CSV es un "
        "número, no una palabra.",
        "adimensional", (0.0, 4.0), ""),
    "mem_recall_costo": (
        "Costo de evocar",
        "Qué parte de la propia memoria episódica hay que revisar para evocar: lo que cuesta "
        "buscar en el archivo. OJO: se declara fracción pero NO está acotada, y medido llega a "
        "1,19; o sea que a veces se revisa más de una vez el archivo entero.",
        "fraccion", (0.0, 1.0), ""),
    "mem_episodios_n": (
        "Episodios guardados",
        "Cuántos recuerdos episódicos tiene archivados; el archivo está acotado y al llenarse "
        "olvida el más débil.",
        "conteo", (0.0, 64.0), ""),
    "mem_saciedad": (
        "Refractariedad tras saciarse",
        "Cuánto de la necesidad ya quedó satisfecha por reacoplarse o por comer bien; decae sola "
        "y es lo que cierra el lazo necesidad→comer→saciedad.",
        "fraccion", (0.0, 1.0), ""),
    "mem_relacional_confianza": (
        "Confianza acumulada hacia el otro",
        "Confianza hacia el par construida por reciprocidad (sentido compartido por disposición a "
        "cooperar); sólo llega a plena en díada.",
        "fraccion", (0.0, 1.0), ""),
    "Cb_integrado": (
        "Presión de desacople acumulada",
        "La presión de desacople leída en crudo, sin normalizar: es la historia que convierte una "
        "disposición del momento en una necesidad con peso.",
        "acumulador", None, ""),
    "necesidad": (
        "Necesidad con historia",
        "La disposición del momento amplificada por la presión acumulada respecto de la habitual: "
        "la necesidad con historia, no el impulso instantáneo.",
        "fraccion", (0.0, 1.0), ""),
    "necesidad_efectiva": (
        "Necesidad tras descontar la saciedad",
        "La necesidad ya descontada la refractariedad: lo que de verdad empuja al organismo a "
        "actuar ahora mismo.",
        "fraccion", (0.0, 1.0), ""),

    # ── ÍNDICE DE HOMEOSTASIS (variables internas en rango) ──────────────────
    "H_var_desorden": (
        "Salud del desorden interno",
        "Salud de la variable interna respecto de su rango viable: 1 en el centro, 0 en los "
        "bordes y fuera. Es una variable de dos caras, como la temperatura.",
        "fraccion", (0.0, 1.0), "C-N5.1"),
    "H_var_reserva": (
        "Salud de la reserva",
        "Salud de una sola cara: la reserva ya viene normalizada, así que su salud es ella misma. "
        "La reserva sólo puede faltar, nunca sobrar.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "H_homeostasis": (
        "Índice de homeostasis (variables internas en rango)",
        "Promedio de la salud de cada variable interna respecto de su rango viable: 1 con todo "
        "estable, y baja sólo cuando alguna se sale. NO mide el acoplamiento con el entorno.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "H_peor": (
        "Peor variable interna",
        "La menor de las saludes parciales, publicada para que un problema grave en una sola "
        "variable no quede diluido en el promedio.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "H_n_variables": (
        "Variables internas contadas",
        "Cuántas variables internas llegaron con dato y entraron en el promedio; si falta una NO "
        "se cuenta, para no convertir el silencio en enfermedad.",
        "conteo", (0.0, 2.0), ""),

    # ── ATENCIÓN SOCIAL: A QUIÉN ESCUCHA Y A QUIÉN LE HABLA ──────────────────
    "as_esc_objetivo": (
        "Identificador del organismo al que ESCUCHA",
        "Identificador del vecino elegido como fuente de audio; queda vacío si en este paso no "
        "atiende a nadie.",
        "texto", None, ""),
    "as_esc_nombre": (
        "A quién escucha (nombre)",
        "Nombre legible del vecino atendido: permite reconstruir la red social dirigida sin "
        "resolver identificadores.",
        "texto", None, ""),
    "as_esc_score": (
        "Puntaje de atención del escuchado",
        "Cuánto prefiere al elegido sobre el resto de candidatos, promediando sus sesgos con los "
        "pesos del lado de la escucha.",
        "adimensional", None, ""),
    "as_esc_sesgo": (
        "Por qué escucha a ése",
        "Sesgo que más aportó al puntaje del elegido: éxito, similitud, dominancia, prestigio, "
        "conformidad o parentesco.",
        "texto", None, ""),
    "as_esc_explorando": (
        "La escucha salió de explorar, no de elegir",
        "1 si el objetivo salió del sorteo de exploración y no del mejor puntaje: separa explorar "
        "de explotar lo ya conocido.",
        "booleano", (0.0, 1.0), ""),
    "as_esc_atender": (
        "Está atendiendo a alguien",
        "1 si hay objetivo de escucha; 0 cuando ejerce su libertad de NO atender a nadie, cosa "
        "que hace más cuando la necesidad es alta.",
        "booleano", (0.0, 1.0), ""),
    "as_esc_dwell_s": (
        "Cuánto lleva escuchando al mismo",
        "Segundos desde que fijó el objetivo de escucha actual: mide compromiso, porque no salta "
        "de uno a otro antes de su permanencia mínima.",
        "segundos", None, ""),
    "as_habla_objetivo": (
        "Identificador del organismo al que HABLA",
        "Identificador del destinatario al que dirige su voz: la dirección intencional de la "
        "emisión.",
        "texto", None, ""),
    "as_habla_nombre": (
        "A quién le habla (nombre)",
        "Nombre legible del destinatario de la voz: el lado emisor de la red social dirigida.",
        "texto", None, ""),
    "as_habla_score": (
        "Puntaje de atención del destinatario",
        "Cuánto prefiere a ese destinatario, promediando sus sesgos con los pesos del lado del "
        "habla, que favorecen la similitud y la conformidad.",
        "adimensional", None, ""),
    "as_habla_sesgo": (
        "Por qué le habla a ése",
        "Sesgo que más aportó al puntaje del destinatario elegido.",
        "texto", None, ""),
    "as_habla_dirige": (
        "Está dirigiendo la voz a alguien",
        "1 si hay destinatario elegido; 0 si habla sin dirigirse a nadie, que también es una "
        "libertad.",
        "booleano", (0.0, 1.0), ""),
    "as_habla_dwell_s": (
        "Cuánto lleva hablándole al mismo",
        "Segundos desde que fijó el destinatario actual de su voz: compromiso del lado emisor.",
        "segundos", None, ""),
    "as_b_exito": (
        "Sesgo: al escuchado le va bien",
        "Cuánto prospera el escuchado (energía, poca necesidad, organismicidad): seguir a quien "
        "le va bien.",
        "fraccion", (0.0, 1.0), ""),
    "as_b_similitud": (
        "Sesgo: el escuchado se parece a mí",
        "Cuánto se parece el escuchado (mismo género, mismo tono, edad cercana): atender a los "
        "pares.",
        "fraccion", (0.0, 1.0), ""),
    "as_b_dominancia": (
        "Sesgo: el escuchado es el más activo",
        "Fuerza y actividad del escuchado (activación, energía, cuánto emite): observar al que "
        "manda la escena.",
        "fraccion", (0.0, 1.0), ""),
    "as_n_candidatos": (
        "Candidatos de atención en el campo",
        "Cuántos vecinos elegibles había cuando decidió: el tamaño real de la población social "
        "sobre la que eligió.",
        "conteo", None, ""),
    "as_modo": (
        "Modo de la atención social",
        "Si el organelo decide solo (auto), lo decide el usuario (manual) o está apagado (off): "
        "define si las demás columnas son decisiones propias.",
        "texto", None, ""),
    "as_aplicado": (
        "A quién escuchó DE VERDAD",
        "Identificador que el hilo aplicador llevó de veras a la fuente de audio: la decisión "
        "efectiva, no la instantánea, que parpadea muchas veces por segundo.",
        "texto", None, ""),

    # ── ALTRUISMO Y DÍADA ────────────────────────────────────────────────────
    "disposicion_cooperar": (
        "Ganas de cooperar con el par",
        "Disposición actual a cooperar con el otro, que se relaja hacia el objetivo que fija la "
        "gobernanza del altruismo.",
        "fraccion", (0.0, 1.0), "O-N22"),
    "altruismo_coopera": (
        "Coopera con el par",
        "1 sólo si se cumplen a la vez el umbral crítico, la regla de Hamilton, leer al otro como "
        "sujeto, el mutualismo sostenido y que separarse cueste: habilita el salto a díada.",
        "booleano", (0.0, 1.0), "O-N22 · C-N8"),
    "altruismo_beta_crit": (
        "Umbral crítico para cooperar",
        "Cuánta disposición hace falta para que cooperar sea viable; baja con más libertad "
        "funcional y con menos error: cooperar es más fácil cuando se es más libre y se yerra menos.",
        "fraccion", (0.0, 1.0), "O-N22.2"),
    "altruismo_psi_alma": (
        "Lee al otro como sujeto",
        "Grado en que trata al par como SUJETO y no como cosa; sin esto no hay altruismo "
        "voluntario, y sin altruismo voluntario no se impone la multicelularidad.",
        "fraccion", (0.0, 1.0), "O-N3.4b"),
    "altruismo_tau": (
        "Reloj de mutualismo sostenido",
        "Segundos acumulados de mutualismo continuo, que se reinician a cero si el otro deja de "
        "querer; hay que pasar su mínimo antes de poder fusionarse.",
        "segundos", None, "O-N9.9"),
    "altruismo_costo_desacople": (
        "Cuánto costaría separarse del par",
        "Cuánto peor estaría el acoplamiento SIN el otro: es el criterio de que la díada ya es "
        "una unidad de orden superior y no dos individuos juntos.",
        "fraccion", (0.0, 1.0), "O-N9.9"),
    "altruismo_S_shared": (
        "Sentido compartido con el par",
        "Capacidad representacional común entre los dos, ponderada por lo alineados que están; es "
        "el parentesco de la regla de Hamilton.",
        "fraccion", (0.0, 1.0), "O-N22"),
    "altruismo_atractor": (
        "Régimen de la díada",
        "«comunicando» si coopera, «emergiendo» si la disposición ya manda sobre el piso de "
        "exploración, «mudo» si no hay nada.",
        "texto", None, "O-N22"),

    # ── LA VOZ: QUÉ CANTA Y CON QUÉ VOCABULARIO ──────────────────────────────
    "voz_emitida": (
        "Voz que emite ahora",
        "Etiqueta de la voz más cercana a su afecto en este paso, o «—» si calla: deja registrado "
        "qué sonido usa en cada contexto.",
        "texto", None, ""),
    "voz_titulo": (
        "Título legible de la voz emitida",
        "Nombre en castellano de la voz que está emitiendo, para poder leer la conversación sin "
        "descifrar etiquetas internas.",
        "texto", None, ""),
    "voz_origen": (
        "De dónde sale esa voz",
        "«banco» si es una voz curada, «creado» si la acuñó él, «aprendida» si la emuló del otro: "
        "separa el léxico heredado del léxico propio.",
        "texto", None, ""),
    "voz_id": (
        "Identificador global de la palabra emitida",
        "Identificador con la letra del organismo que la emitió, para poder TRAZAR rutas léxicas "
        "entre organismos; coincide con la voz emitida.",
        "texto", None, ""),
    "voz_emulada_de": (
        "De qué palabra ajena la copió",
        "Identificador de la palabra del OTRO que esta voz emula, vacío si no emula ninguna: es la "
        "arista con la que se reconstruye quién le copió qué a quién.",
        "texto", None, ""),
    "voz_propias": (
        "Vocabulario propio activo",
        "Cuántas voces propias tiene vivas en el banco (acuñadas y aprendidas, provisionales y "
        "estables); es el vocabulario que posee, y fija el precio de acuñar una más.",
        "conteo", (0.0, 64.0), ""),
    "voz_estables": (
        "Voces propias que cuajaron",
        "Cuántas voces propias pasaron a estables por reuso y se guardaron en disco: el patrimonio "
        "léxico que sobrevive a un reinicio.",
        "conteo", None, ""),
    "voz_aprendidas": (
        "Voces aprendidas del otro",
        "Cuántas voces incorporó emulando palabras que inventó el otro: mide el léxico compartido "
        "frente a la divergencia de dos linajes.",
        "conteo", None, ""),
    "voz_aprendidas_forma": (
        "Voces aprendidas copiando la forma",
        "Cuántas de las aprendidas entraron copiando la FORMA oída en vez de volver a sintetizarla; "
        "sin este desglose el experimento de imitación corre a ciegas.",
        "conteo", None, ""),
    "voz_creadas": (
        "Palabras propias acuñadas (histórico)",
        "Cuántas voces propias ha inventado en toda su vida, contando las recuperadas de disco: "
        "sólo sube, nunca baja.",
        "conteo", None, ""),
    "voz_arousal": (
        "Activación afectiva de la voz",
        "Cuán activado está, promediando ruido disponible, energía y lateralidad medidos contra su "
        "propia historia; con la valencia elige qué voz emitir.",
        "fraccion", (0.0, 1.0), ""),
    "voz_valence": (
        "Valencia afectiva de la voz",
        "Cuán bien le va: lo que lo sostiene (organismicidad, homeostasis, orden) menos lo que le "
        "falta; con la activación fija su punto en el plano afectivo.",
        "adimensional", (-1.0, 1.0), ""),
    "voz_paso_repertorio": (
        "Resolución de su repertorio",
        "Distancia típica entre una voz y su vecina en el plano afectivo: la vara PROPIA con la que "
        "juzga si su estado actual cae en un hueco del vocabulario.",
        "adimensional", None, ""),
    "voz_gap_banco": (
        "Hueco entre su estado y su voz más cercana",
        "Distancia afectiva de lo que siente ahora a la voz más próxima de su banco; comparada con "
        "su resolución, decide si hay hueco y toca acuñar palabra nueva.",
        "adimensional", None, ""),
    "voz_gap_peer": (
        "Hueco entre la palabra del otro y las suyas",
        "Distancia afectiva de la palabra que el otro emite a la voz propia más cercana; comparada "
        "con su resolución, decide si merece la pena emularla.",
        "adimensional", None, ""),
    "voz_bloqueo_motivo": (
        "Por qué no acuñó palabra nueva",
        "Motivo del último intento de acuñar (sin hueco, hueco fugaz, sin energía, libertad, "
        "vocabulario lleno…): hace auditable la vía de la invención léxica.",
        "texto", None, ""),
    "voz_emular_bloqueo": (
        "Por qué no emuló al otro",
        "Motivo del último intento de emular (sin par, ya cubierto, sin energía, libertad…). Es "
        "diagnóstico y va un paso por detrás, porque se consulta antes de intentarlo.",
        "texto", None, ""),

    # ── APRENDIZAJE POR IMITACIÓN (memoria ecoica) ───────────────────────────
    "oao_oido": (
        "Energía que oyó en este paso",
        "La mayor de las energías de los dos oídos: por encima de su umbral cuenta como «oír "
        "algo», y sólo entonces puede memorizar lo escuchado.",
        "adimensional", None, ""),
    "oao_aprendio": (
        "Incorporó lo oído en este paso",
        "1 si guardó en su memoria ecoica lo que acaba de oír. Incorporar NO es automático: a "
        "veces oye y no aprende, y eso también es libertad funcional.",
        "booleano", (0.0, 1.0), ""),
    "oao_echoica_n": (
        "Trazas vivas en la memoria ecoica",
        "Cuántas trazas de gestos oídos siguen presentes en la memoria ecoica antes de decaer: es "
        "el material del que puede salir una imitación.",
        "conteo", (0.0, 512.0), ""),
    "oao_imitacion_mag": (
        "Fuerza del sesgo de imitación",
        "Cuánto tira ahora la voz del otro sobre su voz futura; decae a cero cuando deja de oír, "
        "así que mide una influencia viva, no una memoria.",
        "adimensional", None, ""),
    "oao_eco_freq": (
        "Eco oído: hacia qué frecuencia empuja",
        "Componente de frecuencia del atractor de imitación, promediando lo oído con más peso para "
        "lo reciente.",
        "adimensional", None, ""),
    "oao_eco_intensidad": (
        "Eco oído: hacia qué intensidad empuja",
        "Componente de intensidad del atractor de imitación: hacia qué fuerza de emisión lo empuja "
        "lo que ha escuchado.",
        "adimensional", None, ""),
    "oao_eco_pausa": (
        "Eco oído: hacia cuánta pausa empuja",
        "Componente de pausa del atractor de imitación: hacia qué grado de pausado lo empuja lo "
        "escuchado.",
        "adimensional", None, ""),
    "oao_eco_repeticion": (
        "Eco oído: hacia cuánta repetición empuja",
        "Componente de repetición del atractor de imitación: hacia cuánto repetir lo empuja lo "
        "escuchado.",
        "adimensional", None, ""),

    # ── EL GESTO VOCAL (la forma del sonido, no la palabra) ──────────────────
    "g_freq": (
        "Frecuencia del gesto vocal",
        "Coordenada de frecuencia del gesto acústico que está explorando (NO son hercios); con las "
        "otras tres define la FORMA de la vocalización, no su etiqueta.",
        "adimensional", (-1.0, 1.0), ""),
    "g_intensidad": (
        "Intensidad del gesto vocal",
        "Coordenada de intensidad del gesto acústico actual: parte de la exploración libre con la "
        "que puede descubrir qué forma de vocalizar mueve al otro.",
        "adimensional", (-1.0, 1.0), ""),
    "g_pausa": (
        "Pausa del gesto vocal",
        "Coordenada de pausa del gesto acústico actual: cuán entrecortada es la emisión.",
        "fraccion", (0.0, 1.0), ""),
    "g_repeticion": (
        "Repetición del gesto vocal",
        "Coordenada de repetición del gesto acústico actual: cuánto insiste sobre el mismo "
        "elemento dentro de la emisión.",
        "fraccion", (0.0, 1.0), ""),
    "g_bucket": (
        "Celda del gesto (etiqueta discreta)",
        "El gesto reducido a una celda del espacio acústico, o «—» en silencio: es la clave con la "
        "que la memoria guarda qué gesto usó en cada situación.",
        "texto", None, ""),

    # ── EXPRESIÓN: DECIDIR SI HABLAR O CALLAR ────────────────────────────────
    "expr_vocalizando": (
        "Vocaliza en este paso",
        "1 si emite un gesto vocal, 0 si calla: es el resultado del primer acto, decidir SI "
        "hablar, donde el silencio y la voz compiten de verdad.",
        "booleano", (0.0, 1.0), ""),
    "expr_en_conducta": (
        "La conducta vocal sigue abierta",
        "1 si la secuencia de gestos continúa tras este paso: distingue una emisión suelta de una "
        "conducta sostenida.",
        "booleano", (0.0, 1.0), ""),
    "expr_long_conducta": (
        "Longitud de la conducta vocal en curso",
        "Cuántos gestos lleva emitidos en esta conducta; su longitud no está prefijada: emerge del "
        "recurso que le queda paso a paso.",
        "conteo", None, ""),
    "expr_recurso": (
        "Recurso que le queda para seguir hablando",
        "Arranca de su reserva de energía y baja con cada gesto; cuanto menor, más probable es que "
        "la conducta termine.",
        "acumulador", None, ""),
    "expr_novedad": (
        "Novedad de la situación en que está",
        "Alta cuando casi no tiene historia en esta región de su estado, y entonces explora; baja "
        "cuando ya la tiene, y entonces reutiliza lo que le funcionó.",
        "fraccion", None, ""),
    "expr_p_voz": (
        "Probabilidad de vocalizar",
        "Con qué probabilidad decide hablar en vez de callar, por competencia entre los pesos "
        "históricos de VOZ y de SILENCIO más un empuje por lo saliente de la situación.",
        "fraccion", None, ""),
    "expr_familiaridad": (
        "Familiaridad vocal de la situación",
        "Cuánta historia vocal tiene acumulada en esta región de su estado: la suma de los pesos "
        "de todos los gestos que memorizó aquí.",
        "acumulador", None, ""),
    "expr_consecuencia": (
        "Consecuencia de la conducta que acaba de cerrar",
        "Valor ecológico de la voz más contingencia social al cerrar la conducta: sesga cuánto se "
        "refuerza esa conducta, sin llegar a imponerla.",
        "adimensional", None, ""),
    "expr_estado_key": (
        "Identificador de la región de estado",
        "Código de la región en que cae su estado global. No tiene significado propio: sirve para "
        "ver si el organismo vuelve a la misma situación.",
        "adimensional", (0.0, 99999.0), ""),
    "expr_silencio": (
        "Está sosteniendo una conducta de silencio",
        "1 si calla como CONDUCTA, no por ausencia: el silencio se almacena, se refuerza y se "
        "olvida igual que la voz.",
        "booleano", (0.0, 1.0), ""),
    "expr_long_silencio": (
        "Longitud del silencio en curso",
        "Cuántos pasos consecutivos lleva la conducta de silencio actual; su longitud también "
        "emerge, y se registra al terminar.",
        "conteo", None, ""),
    "expr_peso_silencio": (
        "Peso histórico del silencio en esta situación",
        "Cuánto pesa la conducta de callar en esta región del estado: es la mitad de la balanza "
        "que compite contra el peso de la voz al decidir si habla.",
        "acumulador", None, ""),

    # ── ALTERIDAD: DESCUBRIR QUE HAY OTRO Y QUE PUEDO AFECTARLO ──────────────
    "alt_otro_presente": (
        "Hay otro presente y dando señal",
        "1 si el par está vivo y emitiendo algo: es la compuerta de la que depende todo el órgano "
        "de la alteridad.",
        "booleano", (0.0, 1.0), ""),
    "alt_efecto_basal": (
        "Cuánto cambia el otro por su cuenta",
        "Cuánto cambia el otro en la ventana JUSTO ANTES de que yo emita: la línea base con la que "
        "se descuenta el ambiente compartido.",
        "fraccion", (0.0, 1.0), "O-N3.4"),
    "alt_contingencia_social": (
        "Cuánto MÁS cambia el otro tras mi emisión",
        "Exceso del cambio del otro después de que yo emita sobre lo que cambia solo: es lo que "
        "separa que me esté respondiendo de que compartamos ambiente.",
        "fraccion", (0.0, 1.0), "O-N3.4"),
    "alt_agencia_otro": (
        "Agencia: qué parte del otro depende de mí",
        "Fracción del cambio del otro que es contingente a mi emisión. Es una señal falsable: DEBE "
        "colapsar cuando se le desordena el audio del par, si no hay agencia real.",
        "fraccion", (0.0, 1.0), "O-N3.4"),
    "alt_modelo_otro": (
        "Modelo del otro: qué suele provocar este patrón",
        "Efecto que este patrón suele producir en el otro, aprendido sólo por consecuencias: el "
        "modelo del otro que el organismo se hace sin que nadie se lo cuente.",
        "fraccion", (0.0, 1.0), ""),
    "alt_prediccion_respuesta": (
        "Respuesta que predice del otro",
        "Cuánto espera que cambie el otro tras el patrón que acaba de emitir; publica el mismo "
        "valor que el modelo del otro, con otro nombre.",
        "fraccion", (0.0, 1.0), ""),
    "alt_error_prediccion": (
        "Cuánto falla su modelo del otro",
        "Diferencia media entre el efecto observado en el otro y el que predijo: sirve además como "
        "ruido propio de referencia para no declarar hitos falsos.",
        "fraccion", (0.0, 1.0), ""),
    "alt_efecto_sobre_otro": (
        "Efecto medido de mi emisión sobre el otro",
        "Cuánto cambia el otro (su organismicidad, su necesidad, su orientación, su voz) en la "
        "ventana posterior a que yo emita.",
        "fraccion", (0.0, 1.0), ""),
    "alt_efecto_sobre_mi": (
        "Qué gano yo emitiendo (con signo)",
        "Cómo quedo tras emitir: mejora de organismicidad y acople menos necesidad. Negativo "
        "significa que emitir me deja peor, y es el criterio de si conviene repetir.",
        "adimensional", None, ""),
    "alt_valor_emision": (
        "Valor aprendido de este patrón en este contexto",
        "Beneficio propio acumulado para ese patrón, contado SÓLO cuando el otro respondió: es "
        "aprender qué decir y cuándo, por consecuencias.",
        "adimensional", None, ""),
    "alt_intencion_comunicativa": (
        "Intención comunicativa emergente",
        "Cuánto ha descubierto que el otro le responde Y que eso le beneficia: la medida central "
        "de este órgano, el paso de hacer ruido a comunicar.",
        "fraccion", (0.0, 1.0), ""),
    "alt_patron_emitido": (
        "Patrón vocal emitido",
        "El gesto acústico emitido, o la voz si no hay libertad expresiva, o «—» en silencio. "
        "Nunca es una palabra: es la forma del sonido.",
        "texto", None, ""),
    "alt_patron_repetido": (
        "Ya había emitido ese patrón en este contexto",
        "1 si ese patrón en ese contexto ya se emitió antes: dice si está reincidiendo en un gesto "
        "o estrenándolo.",
        "booleano", (0.0, 1.0), ""),
    "alt_confianza_relacional": (
        "Confianza relacional heredada de la memoria",
        "Copia de la confianza que la memoria acumuló hacia el otro por reciprocidad; hoy llega "
        "casi siempre en cero porque la fila en vivo no trae sus ingredientes.",
        "fraccion", (0.0, 1.0), ""),
    "alt_contacto_presencia": (
        "Llamada de contacto («¿sigues ahí?»)",
        "1 en el paso en que emite estando el otro ausente o recién perdido: es una llamada, no "
        "una respuesta.",
        "booleano", (0.0, 1.0), ""),
    "alt_contacto_recuperado": (
        "Contacto recuperado tras una llamada",
        "Pulso de 1 cuando el otro reaparece después de una llamada: cierra el episodio "
        "llamada→retorno.",
        "booleano", (0.0, 1.0), ""),
    "alt_turno_detectado": (
        "Turno emisor detectado",
        "1 cuando CAMBIA el patrón emitido, no en cada paso: marca el acto emisor que abre una "
        "ventana de medición del efecto.",
        "booleano", (0.0, 1.0), ""),

    # ── QUÉ VALE LA VOZ DEL OTRO (valor ecológico) ───────────────────────────
    "voz_otro_valor_ecologico": (
        "Cuánto vale que el otro suene",
        "Cuánto mejora su situación después de esa estructura de voz respecto de antes: el valor "
        "biológico de que el otro emita, sin suponerle ningún significado.",
        "adimensional", None, ""),
    "voz_otro_relevancia_metabolica": (
        "La voz del otro, ¿me alimenta?",
        "Cuánto cambian su energía y su necesidad tras oír al otro: qué parte del beneficio fue "
        "metabólica.",
        "adimensional", None, ""),
    "voz_otro_relevancia_acople": (
        "La voz del otro, ¿me ajusta al entorno?",
        "Cuánto cambia su acoplamiento con el entorno tras oír al otro: si escucharlo lo ajusta "
        "mejor al mundo.",
        "adimensional", None, ""),
    "voz_otro_relevancia_permeabilidad": (
        "La voz del otro, ¿me abre o me cierra?",
        "Cuánto cambia su permeabilidad tras oír al otro: si escucharlo lo abre al mundo o lo hace "
        "replegarse.",
        "adimensional", None, ""),
    "voz_otro_predice_mejora": (
        "Mejora que espera tras esta voz del otro",
        "Mejora media aprendida para esa firma acústica: el modelo de para qué le sirve que el "
        "otro suene así. Tiene signo.",
        "adimensional", None, ""),
    "voz_otro_error_prediccion": (
        "Cuánto falla esa predicción",
        "Diferencia media entre la mejora observada y la que esperaba: es el umbral de ruido "
        "propio con el que decide si una mejora fue real.",
        "adimensional", None, ""),
    "voz_otro_historia_beneficio": (
        "Beneficio histórico acumulado por la voz del otro",
        "Suma de todas las mejoras atribuidas a voces del otro que valieron: la biografía de "
        "cuánto le ha servido, en total, que el otro suene.",
        "acumulador", None, ""),
    "voz_otro_confianza_ecologica": (
        "Confianza en que la voz del otro sirve",
        "Con qué frecuencia esa voz resultó valer por encima de su propia escala de valores; se "
        "abstiene mientras no tenga historia suficiente.",
        "fraccion", (0.0, 1.0), ""),
    "voz_otro_modulacion_aplicada": (
        "Modulación aplicada al escuchar al otro",
        "Cuánto ajusta la absorción de la voz del par: 1 es neutro. Nunca controla la conducta, "
        "sólo facilita o entorpece levemente el paso de esa voz.",
        "adimensional", (0.75, 1.25), ""),
    "voz_otro_efecto_real": (
        "Efecto real medido tras la última voz del otro",
        "La última mejora medida al cerrar la ventana, con signo: el dato crudo del que sale el "
        "valor ecológico.",
        "adimensional", None, ""),

    # ── EXPECTATIVA: ¿VALE LA PENA SEGUIR MIRANDO? ───────────────────────────
    "expectativa": (
        "Expectativa tras esta firma de voz",
        "Cuánto mejora la situación después de esa firma respecto de antes: si históricamente vale "
        "la pena seguir observando cuando la oye.",
        "adimensional", None, ""),
    "expectativa_confianza": (
        "Confianza en la expectativa",
        "Con qué frecuencia esa firma resultó valer, contra la escala de sus propias expectativas; "
        "sin historia madura se abstiene en vez de regalar confianza.",
        "fraccion", (0.0, 1.0), ""),
    "expectativa_error": (
        "Error de la expectativa",
        "Diferencia media entre la mejora observada y la esperada: el ruido propio contra el que "
        "decide si una mejora es real o casualidad.",
        "adimensional", None, ""),
    "expectativa_historia": (
        "Beneficio histórico de explorar tras la voz",
        "Suma de las mejoras obtenidas cuando explorar tras oír esa firma sirvió de algo: la "
        "biografía del aprendizaje expectativo.",
        "acumulador", None, ""),
    "expectativa_utilidad": (
        "Utilidad esperada de la firma actual",
        "Mejora media aprendida para esta firma sin descontar la línea base; con signo, o sea que "
        "puede ser negativa.",
        "adimensional", None, ""),
    "expectativa_exploracion": (
        "Disposición a seguir observando",
        "Único efecto de este órgano sobre el organismo: un empujón LEVE a hacer una segunda "
        "observación. Nunca orienta ni decide.",
        "fraccion", (0.0, 0.2), ""),
    "expectativa_confirmaciones": (
        "Veces que explorar tras la voz sí sirvió",
        "Cuántas veces explorar después de esa voz mejoró por encima de su propio error de "
        "predicción: el numerador de su tasa de acierto.",
        "conteo", None, ""),
    "expectativa_falsaciones": (
        "Veces que explorar tras la voz NO sirvió",
        "Cuántas veces explorar tras la voz no mejoró, contadas sólo cuando ya había ruido medido: "
        "sin esto la expectativa no sería falsable.",
        "conteo", None, ""),

    # ── VALORACIÓN EXPERIENCIAL: LA BIOGRAFÍA ────────────────────────────────
    "ove_experiencias": (
        "Experiencias cerradas",
        "Cuántas experiencias ha segmentado y cerrado por silencio en toda su vida: el tamaño real "
        "de su biografía.",
        "conteo", None, ""),
    "ove_valoracion_actual": (
        "Valoración de la última experiencia",
        "Hipótesis histórica de cuánto contribuyó esa experiencia a seguir existiendo: "
        "reorganización más estabilidad más libertad, menos lo que costó.",
        "adimensional", (-1.0, 1.0), ""),
    "ove_confianza": (
        "Confianza en esa valoración",
        "Cuántas experiencias parecidas respaldan la valoración: poca confianza significa que la "
        "hipótesis acaba de nacer.",
        "fraccion", (0.0, 1.0), ""),
    "ove_novedad": (
        "Novedad de la experiencia",
        "Distancia a la región conocida más cercana de su paisaje experiencial: alto significa una "
        "experiencia sin precedente en su historia.",
        "adimensional", None, ""),
    "ove_reorganizacion": (
        "Reorganización lograda en la experiencia",
        "Cuánto se transformó el organismo mientras duró la experiencia, medido como la magnitud "
        "del cambio de su estado global.",
        "adimensional", None, ""),
    "ove_coste": (
        "Costo energético de la experiencia",
        "El precio metabólico que la experiencia se llevó, normalizado; es lo que se descuenta de "
        "su valoración.",
        "adimensional", (0.0, 1.0), ""),
    "ove_persistencia": (
        "Estabilidad alcanzada tras la experiencia",
        "1 si quedó muy estable después: es el componente de continuidad de la valoración, lo que "
        "distingue una experiencia que asentó de otra que sólo agitó.",
        "fraccion", (0.0, 1.0), ""),
    "ove_radio": (
        "Radio del vecindario de comparación",
        "Distancia típica entre regiones de su paisaje: la escala PROPIA con la que decide si algo "
        "es una región nueva o se asimila a una que ya tiene.",
        "adimensional", None, ""),
    "ove_region": (
        "Región del paisaje experiencial",
        "Identificador NUMÉRICO del grupo de experiencias semejantes al que se asimiló la última; "
        "las regiones emergen por distancia, nadie las etiquetó antes.",
        "adimensional", None, ""),
    "ove_memoria": (
        "Registros de experiencia guardados",
        "Cuántos registros conserva de verdad en su memoria experiencial, que tiene tope; es "
        "distinto del total histórico de experiencias vividas.",
        "conteo", (0.0, 1200.0), ""),
    "ove_comparaciones": (
        "Comparaciones contra el paisaje",
        "Cuántas veces una experiencia nueva se comparó con el paisaje que ya tenía: mide el "
        "trabajo de memoria que lleva hecho.",
        "conteo", None, ""),
    "ove_preferencia": (
        "Preferencia histórica del vecindario",
        "Valoración media de las experiencias vecinas de esa región. Es SÓLO observable: por "
        "cláusula del órgano, jamás se convierte en conducta.",
        "adimensional", (-1.0, 1.0), ""),

    # ── PRESENCIA: LA SOCIEDAD ALREDEDOR ─────────────────────────────────────
    "presencia_vivo": (
        "El órgano de presencia está activo",
        "1 si el descubrimiento de vecinos está funcionando; si vale 0, todas las demás columnas "
        "de presencia son ciegas y no significan «estoy solo».",
        "booleano", (0.0, 1.0), ""),
    "presencia_vecinos_n": (
        "Vecinos vivos detectados",
        "Cuántos organismos ANIMA ve ahora mismo, contando sólo los que se anunciaron hace poco: "
        "el tamaño real de su sociedad en este instante.",
        "conteo", None, ""),
    "presencia_local_n": (
        "Vecinos en la red local",
        "Cuántos vecinos detecta en la red local; hoy publica exactamente el mismo número que el "
        "total de vecinos, porque el código no filtra por origen.",
        "conteo", None, ""),
    "presencia_global_n": (
        "Vecinos fuera de la red local",
        "Cuántos vecinos remotos ve por la plaza pública. Hoy está escrito literalmente como cero "
        "y nunca se calcula: es una columna constante.",
        "conteo", (0.0, 0.0), ""),
    "presencia_confiable_n": (
        "Vecinos con los que puede contar",
        "Cuántos vecinos alcanzan el nivel de confianza necesario para tenerlos por reales: con "
        "cuántos otros cuenta de verdad.",
        "conteo", None, ""),
    "presencia_confianza": (
        "Confianza media en los vecinos",
        "Confianza media ponderada por lo reciente de su última señal: cae sola cuando dejan de "
        "anunciarse, sin necesidad de declarar muerto a nadie.",
        "fraccion", (0.0, 0.95), ""),
    "presencia_aislamiento": (
        "Aislamiento acumulado",
        "Sube con el tiempo que lleva sin vecinos y baja lentamente cuando alguno reaparece: es el "
        "hambre de otro, en bruto.",
        "fraccion", (0.0, 1.0), ""),
    "presencia_retorno": (
        "Volvió un vecino conocido",
        "Pulso que dura unos segundos cuando un vecino ausente reaparece: marca el reencuentro, no "
        "la presencia sostenida.",
        "booleano", (0.0, 1.0), ""),
    "presencia_novedad": (
        "Apareció un vecino nunca visto",
        "Pulso que dura unos segundos al descubrir una instalación desconocida: distingue conocer "
        "a alguien nuevo de reencontrarse con un conocido.",
        "booleano", (0.0, 1.0), ""),
    "presencia_densidad": (
        "Densidad social del entorno",
        "Cuán poblado está su entorno social, como n/(n+1) sobre los vecinos vivos: cada vecino "
        "nuevo suma menos que el anterior, pero suma, y nunca llega a 1 porque no existe un número "
        "máximo de otros. Estar solo es 0; el primer otro vale la mitad de la escala.",
        "fraccion", (0.0, 1.0), ""),
    "presencia_proximidad": (
        "Cuán recientemente se hicieron notar los vecinos",
        "Frescura media de las señales de los vecinos vivos: es lo que sustituye a una distancia "
        "física, que este organismo no puede medir.",
        "fraccion", (0.0, 1.0), ""),
    "hambre_social": (
        "Hambre social",
        "La necesidad de otro combinada con su necesidad fisiológica propia: por qué busca "
        "compañía además de por qué busca alimento.",
        "fraccion", (0.0, 1.0), ""),
    "comunicacion_foco": (
        "Foco comunicativo",
        "Cuánta razón tiene ahora mismo para dirigirse a alguien: se dispara cuando aparece un "
        "vecino nuevo o cuando vuelve uno conocido.",
        "fraccion", (0.0, 1.0), ""),

    # ── TRAZABILIDAD DEL EXPERIMENTO (etiquetas puestas desde fuera) ─────────
    "exp_topologia": (
        "Etiqueta: topología del experimento",
        "Marca puesta desde fuera (no la produce el organismo) para poder agrupar filas por la "
        "topología de la red en que corría; vacía si nadie la puso.",
        "texto", None, ""),
    "exp_ciclo": (
        "Etiqueta: ciclo del experimento",
        "Marca externa del ciclo o repetición al que pertenece la fila; vacía si nadie la puso.",
        "texto", None, ""),
    "exp_mundo_audio": (
        "Etiqueta: mundo sonoro del experimento",
        "Marca externa del material sonoro con que se corrió la fase; vacía si nadie la puso.",
        "texto", None, ""),
    "exp_control": (
        "Etiqueta: condición de control",
        "Marca externa de la condición de control acústico (real, sin par, o par desincronizado); "
        "vacía si nadie la puso.",
        "texto", None, ""),
    "exp_fuente_relacion": (
        "Etiqueta: de dónde venía la relación",
        "Marca externa de qué fuente ocupaba el lugar del par en esa fase; vacía si nadie la puso.",
        "texto", None, ""),

    # ── ESQUEMA VIEJO: las diez `H_*` de antes del 6-ago-2026 ────────────────
    # Se llamaban así cuando la homeostasis emergente publicaba con prefijo `H_`.
    # Hoy son las `acople_*`. Están aquí para que los CSV viejos también se lean.
    "H_homeostasis_real": (
        "Salud homeostática real (nombre viejo de acople_sostenido)",
        "Lo mismo que `acople_sostenido`: con qué calidad la competencia sentido↔desecho sostiene "
        "el acoplamiento. Columna de CSV anteriores al 6-ago-2026.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "H_A_estabilidad": (
        "Estabilidad del acople (nombre viejo)",
        "Lo mismo que `acople_A_estabilidad`, en el esquema anterior al 6-ago-2026.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "H_RC_vivo": (
        "Razón viva (nombre viejo)",
        "Lo mismo que `acople_RC_vivo`, en el esquema anterior al 6-ago-2026.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "H_competencia_ICR_IRDE": (
        "Competencia sentido↔desecho (nombre viejo)",
        "Lo mismo que `acople_competencia_ICR_IRDE`, en el esquema anterior al 6-ago-2026.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "H_recuperacion_A": (
        "Recuperación del acople (nombre viejo)",
        "Lo mismo que `acople_recuperacion_A`, en el esquema anterior al 6-ago-2026.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "H_autoencierro": (
        "Patología de autoencierro (nombre viejo)",
        "Lo mismo que `acople_autoencierro`, en el esquema anterior al 6-ago-2026.",
        "fraccion", (0.0, 1.0), "O-N2.1"),
    "H_anestesia": (
        "Patología de anestesia (nombre viejo)",
        "Lo mismo que `acople_anestesia`, en el esquema anterior al 6-ago-2026.",
        "fraccion", (0.0, 1.0), "O-N2.1"),
    "H_banda_centro_A": (
        "Centro de la banda del acople (nombre viejo)",
        "Lo mismo que `acople_banda_centro_A`, en el esquema anterior al 6-ago-2026.",
        "fraccion", (0.0, 1.0), "O-N9.14"),
    "H_banda_var_A": (
        "Anchura de la banda del acople (nombre viejo)",
        "Lo mismo que `acople_banda_var_A`, en el esquema anterior al 6-ago-2026.",
        "adimensional", None, "O-N9.14"),
    "H_dA_sys_env": (
        "Tendencia del acople (nombre viejo)",
        "Lo mismo que `acople_dA_sys_env`, en el esquema anterior al 6-ago-2026.",
        "adimensional", None, "O-N9.14"),
}


# ==============================================================================
# EL PLURAL DE CADA CONTEO
# ==============================================================================
# Un conteo sin sustantivo no se lee: «21» no dice nada, «21 voces» sí. Aquí vive
# de qué es cada conteo, en singular y en plural.
COSA: dict = {
    "lf_nivel":                  ("nivel de libertad", "niveles de libertad"),
    "invariantes_ok":            ("invariante cumplido", "invariantes cumplidos"),
    "met_tipos_n":               ("tipo de alimento", "tipos de alimento"),
    "mem_episodios_n":           ("episodio", "episodios"),
    "H_n_variables":             ("variable interna", "variables internas"),
    "as_n_candidatos":           ("candidato", "candidatos"),
    "voz_propias":               ("voz propia", "voces propias"),
    "voz_estables":              ("voz estable", "voces estables"),
    "voz_aprendidas":            ("voz aprendida", "voces aprendidas"),
    "voz_aprendidas_forma":      ("voz copiada de forma", "voces copiadas de forma"),
    "voz_creadas":               ("voz", "voces"),
    "oao_echoica_n":             ("traza ecoica", "trazas ecoicas"),
    "expr_long_conducta":        ("gesto", "gestos"),
    "expr_long_silencio":        ("paso de silencio", "pasos de silencio"),
    "expectativa_confirmaciones": ("confirmación", "confirmaciones"),
    "expectativa_falsaciones":   ("falsación", "falsaciones"),
    "ove_experiencias":          ("experiencia", "experiencias"),
    "ove_memoria":               ("registro guardado", "registros guardados"),
    "ove_comparaciones":         ("comparación", "comparaciones"),
    "presencia_vecinos_n":       ("vecino", "vecinos"),
    "presencia_local_n":         ("vecino local", "vecinos locales"),
    "presencia_global_n":        ("vecino remoto", "vecinos remotos"),
    "presencia_confiable_n":     ("vecino confiable", "vecinos confiables"),
}


# ==============================================================================
# EL RANGO MEDIDO — generado, no escrito a mano
# ==============================================================================
# Volcado de `python analisis/glo_rangos.py --py` sobre la historia real del
# organismo ANIMA_5Z934MWHNNRH (los CSV recientes más los del esquema viejo).
# Sólo se usa para las columnas cuyo código NO declara ninguna cota: para todas
# las demás manda lo que el código promete, y `glo_rangos.py --fugas` avisa
# cuando lo prometido y lo medido no coinciden.
_MEDIDO: dict = {
    "Cb_integrado": (1.3628, 212.591),
    "H_banda_var_A": (0, 0.119106),
    "H_dA_sys_env": (-0.01711, 0.03289),
    "Lambda_Cos": (0, 0.08592),
    "Omega_op": (2.4627, 3),
    "RC_delta_salud": (-1.20294, 1.20579),
    "acople_banda_var_A": (7e-06, 0.025327),
    "acople_dA_sys_env": (-0.012155, 0.021206),
    "act_adaptacion_comprension": (-0.077976, 0.007588),
    "act_adaptacion_motor": (-0.036688, 0.004943),
    "act_delta_deg": (-1.10294, 1.77086),
    "act_error_motor": (0.0119, 110.478),
    "act_fatiga": (0.0546, 347.219),
    "act_k_motor_eff": (0.023312, 0.064943),
    "act_lateralidad_dw": (-0.1259, 0.3201),
    "act_mejora_motor": (-100.248, 72.2728),
    "act_pitch_delta_deg": (-0.20789, 0.52248),
    "act_temblor_rms": (0.0537, 0.2392),
    "alt_efecto_sobre_mi": (-0.1184, 0.064),
    "alt_valor_emision": (-0.0894, 0.1918),
    "altruismo_tau": (0, 4),
    "as_esc_dwell_s": (0, 42.8),
    "as_esc_score": (0, 0.6355),
    "as_habla_dwell_s": (0, 47.4),
    "as_habla_score": (0, 0.5473),
    "as_n_candidatos": (2, 6),
    "campo_env": (0, 0.448129),
    "demanda_entorno": (1, 1262.91),
    "energia_L": (0, 471.489),
    "energia_R": (0, 468.576),
    "expectativa": (0, 0.0312),
    "expectativa_confirmaciones": (0, 234),
    "expectativa_error": (0, 0.1557),
    "expectativa_falsaciones": (0, 1011),
    "expectativa_utilidad": (-0.14, 0.0238),
    "expr_familiaridad": (0, 707.662),
    "expr_long_conducta": (0, 6),
    "expr_long_silencio": (0, 20),
    "expr_novedad": (0, 1),
    "expr_p_voz": (0.0075, 1),
    "expr_peso_silencio": (0, 44.529),
    "expr_recurso": (-0.6, 0.5889),
    "gradiente": (-0.0122, 0.5343),
    "hemi_R2": (0, 0.02353),
    "mem_carga_estructural": (0, 0.11997),
    "met_balance": (-0.07588, 0.12891),
    "met_costo_extra": (0, 0.0461221),
    "met_gasto": (0.003, 0.07727),
    "met_ingesta": (0, 0.15419),
    "met_tipos_n": (1, 6),
    "mutacion": (-1.95361, 2.0098),
    "oao_eco_freq": (-0.108, 0.099),
    "oao_eco_intensidad": (-0.129, 0.138),
    "oao_eco_pausa": (0.002, 0.146),
    "oao_eco_repeticion": (0.003, 0.118),
    "oao_imitacion_mag": (0.0093, 0.1875),
    "oao_oido": (0, 471.489),
    "omega_A": (-0.0055, 1),
    "omega_B": (0.0066, 0.5955),
    "ove_comparaciones": (0, 3460),
    "ove_experiencias": (0, 3461),
    "ove_novedad": (0, 6.7175),
    "ove_radio": (0, 6.7175),
    "ove_reorganizacion": (0, 18.0473),
    "presencia_confiable_n": (2, 6),
    "presencia_local_n": (2, 6),
    "presencia_vecinos_n": (2, 6),
    "t": (0.1, 1623.4),
    "tim_centro_L": (0, 0.5638),
    "tim_centro_R": (0, 0.5645),
    "tim_ds_L": (0, 21.7138),
    "tim_ds_R": (0, 21.6466),
    "tim_energia_L": (0, 471.489),
    "tim_energia_R": (0, 468.576),
    "tim_flujo": (0, 7.97759),
    "tim_lateralidad": (-282.879, 296.528),
    "tim_transmitido_L": (0, 10.0565),
    "tim_transmitido_R": (0, 10.3777),
    "ts_real": (1.78578e+09, 1.78621e+09),
    "voz_creadas": (0, 21),
    "voz_estables": (0, 11),
    "voz_gap_banco": (0.0017, 0.2422),
    "voz_otro_efecto_real": (-0.9711, 0.7857),
    "voz_otro_error_prediccion": (0, 0.1572),
    "voz_otro_historia_beneficio": (0, 1.8203),
    "voz_otro_predice_mejora": (-0.138, 0.0227),
    "voz_otro_relevancia_acople": (-0.0165, 0.0201),
    "voz_otro_relevancia_metabolica": (-0.1218, 0.0028),
    "voz_otro_relevancia_permeabilidad": (-0.0031, 0.0826),
    "voz_otro_valor_ecologico": (0, 0.0294),
    "x_interna_esfuerzo": (0, 0.1407),
    "x_interna_estres": (0.002, 0.14198),
    "x_interna_perturb_exceso": (0, 12.6516),
    "x_interna_perturb_habitual": (8.2487, 11.9116),
}


# ==============================================================================
# LOS CINCO DICCIONARIOS DEL CONTRATO — derivados de la tabla
# ==============================================================================
NOMBRE: dict = {c: v[0] for c, v in _TABLA.items()}
DEFINICION: dict = {c: v[1] for c, v in _TABLA.items() if v[1]}
UNIDAD: dict = {c: v[2] for c, v in _TABLA.items() if v[2]}
NODO: dict = {c: v[4] for c, v in _TABLA.items() if v[4]}

# El rango: lo DECLARADO manda; donde el código no promete nada, entra lo medido.
RANGO: dict = {}
_RANGO_MEDIDO: set = set()
for _c, _v in _TABLA.items():
    if _v[3] is not None:
        RANGO[_c] = (float(_v[3][0]), float(_v[3][1]))
    elif _c in _MEDIDO:
        RANGO[_c] = (float(_MEDIDO[_c][0]), float(_MEDIDO[_c][1]))
        _RANGO_MEDIDO.add(_c)
del _c, _v

SIN_RANGO: tuple = (None, None)

# Columnas a las que NO se les imprime la escala aunque se les haya medido un mínimo y
# un máximo: un reloj absoluto no tiene rango, tiene lecturas. Poner «1.786.204.801 s
# (de 1.785.780.000 a 1.786.210.000 s)» no informa de nada, sólo ensucia la línea.
SIN_ESCALA: frozenset = frozenset(("ts_real",))

# Las unidades legales. Si aparece una que no está aquí, es una errata.
UNIDADES: frozenset = frozenset((
    "fraccion", "porcentaje", "grados", "conteo", "rms", "acumulador",
    "adimensional", "booleano", "texto", "segundos", "hz",
))

# El sufijo con que se imprime cada unidad cuando no tiene formato propio.
_SUFIJO: dict = {
    "grados": "°", "segundos": " s", "hz": " Hz", "rms": " RMS",
    "porcentaje": " %", "acumulador": "", "adimensional": "",
    "conteo": "", "fraccion": "", "booleano": "", "texto": "",
}

# Cuántos decimales pide cada unidad para leerse sin ruido.
_DECIMALES: dict = {
    "grados": 1, "segundos": 1, "hz": 1, "rms": 4, "porcentaje": 1,
    "adimensional": 4, "fraccion": 3, "conteo": 0, "acumulador": 0,
}


# ==============================================================================
# NÚMEROS EN CASTELLANO
# ==============================================================================
def _numero(v: float, decimales: int = 3) -> str:
    """Un número como se escribe en castellano: miles con punto, decimales con coma."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x or x in (float("inf"), float("-inf")):
        return str(v)
    s = f"{abs(x):,.{decimales}f}"
    # de «1,234.567» a «1.234,567» sin pasar dos veces por el mismo carácter
    s = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return ("-" + s) if x < 0 else s


def _como_numero(valor):
    """El valor como float, o None si no lo es. Un CSV mezcla números y texto."""
    if isinstance(valor, bool):
        return 1.0 if valor else 0.0
    if isinstance(valor, (int, float)):
        x = float(valor)
        return x if x == x else None
    if isinstance(valor, str):
        s = valor.strip().replace(",", ".")
        if not s or s in ("-", "None", "nan", "NaN", "?"):
            return None
        try:
            x = float(s)
        except ValueError:
            return None
        return x if x == x else None
    return None


def _rango_texto(col: str, unidad: str, dec: int = -1) -> str:
    """« (de ±90°)» o « (de 0 a 1)»: la escala en la que hay que leer el número."""
    if col in SIN_ESCALA:
        return ""
    lo, hi = RANGO.get(col, SIN_RANGO)
    if lo is None or hi is None:
        return ""
    if dec < 0:
        dec = _DECIMALES.get(unidad, 4)
    suf = _SUFIJO.get(unidad, "")
    if lo == -hi and hi > 0:
        return f" (de ±{_numero(hi, dec)}{suf})"
    return f" (de {_numero(lo, dec)} a {_numero(hi, dec)}{suf})"


def _entero(x) -> bool:
    """¿Este número es un entero disfrazado de decimal?"""
    try:
        return float(x) == int(float(x))
    except (TypeError, ValueError, OverflowError):
        return False


def _decimales_de(col: str, unidad: str, x: float) -> int:
    """Cuántos decimales pide ESTE valor. Un código (met_clase −1/0/1, mem_recall_tipo
    0..4, lf_nivel) escrito como «2,0000» se lee peor, no mejor: si tanto el valor como
    su escala son enteros, se imprime entero."""
    dec = _DECIMALES.get(unidad, 4)
    if unidad != "adimensional" or not _entero(x):
        return dec
    lo, hi = RANGO.get(col, SIN_RANGO)
    if lo is None or hi is None:
        return dec
    return 0 if (_entero(lo) and _entero(hi)) else dec


# ==============================================================================
# LA INTERFAZ
# ==============================================================================
def nombre(col: str) -> str:
    """Nombre descriptivo en castellano, o la sigla si todavía no está en el glosario."""
    return NOMBRE.get(col, col)


def definicion(col: str) -> str:
    """La frase que dice qué mide, o cadena vacía si nadie la ha podido leer del código."""
    return DEFINICION.get(col, "")


def unidad(col: str) -> str:
    """La unidad declarada, o cadena vacía si la columna no está en el glosario."""
    return UNIDAD.get(col, "")


def nodo(col: str) -> str:
    """El nodo de la Teoría que sustenta la columna, o cadena vacía si el código no cita ninguno."""
    return NODO.get(col, "")


def rango(col: str) -> tuple:
    """(min, max) declarado o medido; (None, None) si no hay ninguno."""
    return RANGO.get(col, SIN_RANGO)


def rango_es_medido(col: str) -> bool:
    """True si el rango sale de los datos porque el código no declara ninguna cota.

    Importa para no confundir una promesa con una observación: un techo declarado
    es una ley del organismo, un máximo visto es sólo lo que ha pasado hasta hoy.
    """
    return col in _RANGO_MEDIDO


def describir(col: str) -> dict:
    """Todo lo que se sabe de una columna. NUNCA falla: una sigla desconocida
    devuelve su propio nombre y los campos vacíos, que es la respuesta honesta."""
    col = str(col)
    lo, hi = RANGO.get(col, SIN_RANGO)
    return {
        "sigla": col,
        "nombre": NOMBRE.get(col, col),
        "definicion": DEFINICION.get(col, ""),
        "unidad": UNIDAD.get(col, ""),
        "rango": (lo, hi),
        "nodo": NODO.get(col, ""),
    }


def es_fraccion(col: str) -> bool:
    """True si la variable ES una fracción de algo, o sea si vive en [0,1] porque
    el CÓDIGO la acota ahí (un clamp, un `_c01`, un rango [0,1] declarado).

    No se decide a ojo columna por columna, y por eso no basta con que los datos
    caigan entre 0 y 1: `met_energia` nunca pasó de 0,59 y sí es una fracción;
    `acople_banda_var_A` tampoco pasa de 0,03 y NO lo es, porque nadie la acota.
    Lo contrario se comprueba con `analisis/glo_rangos.py --fugas`, que lista las
    columnas que declaran [0,1] y se salen.
    """
    return UNIDAD.get(col, "") == "fraccion"


def formatear(col: str, valor) -> str:
    """La columna, legible. Cada unidad se imprime como lo que es.

    La regla que motivó todo esto: NO convertir a porcentaje lo que no es una
    fracción. Medido en este organismo, hacerlo daba «870 % de error» para 8,7
    grados, «34.700 %» para una fatiga de 347 y «2.100 %» para 21 voces. Eso es
    MENOS legible, no más.
    """
    u = UNIDAD.get(col, "")
    x = _como_numero(valor)

    if u == "texto" or (x is None and u != ""):
        s = "" if valor is None else str(valor).strip()
        return s if s else "—"
    if x is None:
        s = "" if valor is None else str(valor).strip()
        return s if s else "—"

    if u == "booleano":
        return "no" if x == 0 else "sí"

    if u == "fraccion":
        return f"{_numero(x * 100.0, 3)} %"

    if u == "porcentaje":
        return f"{_numero(x, 1)} %"

    if u == "grados":
        return f"{_numero(x, 1)}°{_rango_texto(col, u)}"

    if u == "conteo":
        n = int(round(x))
        sing, plur = COSA.get(col, ("", ""))
        etiqueta = sing if (n == 1 and sing) else plur
        return f"{_numero(n, 0)} {etiqueta}".strip()

    if u == "acumulador":
        dec = 0 if abs(x) >= 10 else 4
        lo, hi = RANGO.get(col, SIN_RANGO)
        if hi is None:
            return _numero(x, dec)
        techo = _numero(hi, 0 if abs(hi) >= 10 else 4)
        visto = " (máx. visto)" if rango_es_medido(col) else ""
        return f"{_numero(x, dec)} de {techo}{visto}"

    if u == "segundos":
        return f"{_numero(x, 1)} s{_rango_texto(col, u)}"

    if u == "hz":
        return f"{_numero(x, 1)} Hz{_rango_texto(col, u)}"

    if u == "rms":
        return f"{_numero(x, 4)} RMS{_rango_texto(col, u)}"

    # adimensional y todo lo que no declare unidad: el número con su escala
    dec = _decimales_de(col, u, x)
    return f"{_numero(x, dec)}{_rango_texto(col, u, dec)}"


def linea(col: str, valor=None) -> str:
    """Una línea de informe: «Reserva de energía (met_energia): 59,490 %»."""
    d = describir(col)
    cab = f"{d['nombre']} ({col})" if d["nombre"] != col else col
    return cab if valor is None else f"{cab}: {formatear(col, valor)}"


def sin_traducir(columnas) -> list:
    """Las columnas que aún no tienen nombre en castellano. Lo que no se puede
    nombrar no se puede auditar, así que esta lista es trabajo pendiente."""
    return sorted(c for c in columnas if c not in NOMBRE)


def sin_definicion(columnas=None) -> list:
    """Las columnas nombradas cuya definición NO se pudo leer del código.
    Van marcadas a propósito: rellenarlas de memoria sería peor que dejarlas."""
    cols = list(columnas) if columnas is not None else list(NOMBRE)
    return sorted(c for c in cols if not DEFINICION.get(c))


def sin_nodo(columnas=None) -> list:
    """Las columnas que no citan ningún nodo de la Teoría en el código que las produce."""
    cols = list(columnas) if columnas is not None else list(NOMBRE)
    return sorted(c for c in cols if not NODO.get(c))


def comprobar() -> list:
    """Errores internos del propio glosario (unidades ilegales, rangos al revés,
    conteos sin sustantivo). Devuelve la lista de problemas; vacía es que está bien."""
    problemas = []
    for c, v in _TABLA.items():
        if v[2] not in UNIDADES:
            problemas.append(f"{c}: unidad ilegal «{v[2]}»")
        if v[2] == "conteo" and c not in COSA:
            problemas.append(f"{c}: es un conteo y no dice de QUÉ (falta en COSA)")
        lo, hi = RANGO.get(c, SIN_RANGO)
        if lo is not None and hi is not None and lo > hi:
            problemas.append(f"{c}: rango al revés {(lo, hi)}")
    return problemas


if __name__ == "__main__":
    import sys as _sys
    try:                                   # la consola de Windows nace en cp1252
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    _p = comprobar()
    print(f"glosario: {len(NOMBRE)} columnas · {len(DEFINICION)} definiciones · "
          f"{len(UNIDAD)} unidades · {len(RANGO)} rangos · {len(NODO)} nodos")
    print(f"problemas internos: {len(_p)}")
    for _x in _p:
        print("  " + _x)
