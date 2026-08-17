# Fase V — universalidad de S>0: qué necesito que definas antes de poder codear

**Fecha:** 8-ago-2026 · **Para:** Alexis López Tapia · **Preparado por:** CC, a partir del roadmap del 5-ago y
de todo lo aprendido en las Fases I-IV de esta semana.

## Por qué esto no se puede codear sin tu decisión primero

Fase V pregunta si la fecundidad que vimos toda la semana (S>0 ⇒ persistencia ⇒ historia ⇒ estructura)
pertenece al **principio relacional en sí**, o sólo al conjunto particular de ecuaciones que usa Cosmogénesis
para implementarlo. Para responder eso hace falta generar una POBLACIÓN de reglas distintas y ver si la
fecundidad aparece en muchas de ellas o sólo en la nuestra. Pero "una población de reglas relacionales" no es
algo que se pueda armar sin decidir primero **qué puede variar y qué tiene que quedar fijo** — si yo lo decido
solo, corro el riesgo de fabricar el resultado (armar reglas que sé que van a funcionar, o al revés) en vez de
medirlo. Por eso este documento no propone código: propone las decisiones concretas que necesito de vos.

## Lo que esta semana ya nos enseñó, y que debería informar el diseño

1. **En la jerarquía NULL de CS073, lo que separó "cero estructura" de "estructura como REAL" no fue ningún
   detalle fino — fue tener ALGÚN grafo/proceso relacional de fondo, sea cual sea (NULL-1/2 sin grafo = cero;
   NULL-3/4 con grafo, aunque alterado = REAL).** Esto sugiere que el eje más importante a barrer en Fase V
   podría no ser "qué ecuación exacta", sino algo más binario: ¿la regla tiene un mecanismo genuinamente
   relacional de fondo o no?
2. **En Fase IV, la aridad relacional sola (hipergrafo sin retroalimentación) no bastó — hizo falta que una
   relación pudiera actuar sobre otras relaciones** (el 2-complejo activo). Esto sugiere un segundo eje: ¿la
   regla permite retroalimentación entre relaciones, o sólo entre entidades?
3. **En Fase III, ni renormalizar ni podar por costo relacional resolvieron el mundo-pequeño del todo** — el
   espacio de reglas que Fase V explore podría necesitar, como tercer eje, alguna noción de escala/localidad
   más fuerte que la que ya se probó, o aceptar que "mundo-pequeño persistente" sea una de las clases de
   universalidad válidas a reportar, no un fracaso del barrido.

## Las decisiones concretas que necesito

**1. La forma funcional del "principio único".** El roadmap dice "localidad relacional, ausencia de
coordenadas, ausencia de objetivos, ausencia de valores físicos horneados, persistencia mínima posible,
interacción recíproca, complejidad algorítmica acotada" — son restricciones, no una regla. Necesito que definas
(aunque sea en prosa, yo lo traduzco a código): **¿qué es lo mínimo que una regla necesita tener para contar
como "instancia de S>0 ⇒ relación", y qué la descalificaría de entrada?** Por ejemplo: ¿alcanza con cualquier
regla de actualización local sobre un grafo con memoria, o hace falta algo más específico (una noción de
diferencia/persistencia explícita, no sólo cualquier dinámica)?

**2. Qué ejes barrer.** Dado el punto anterior (grafo de fondo sí/no; retroalimentación entre relaciones sí/no)
— ¿esos son los ejes correctos para Fase V, o tenés en mente otros que la Teoría considera más centrales
(por ejemplo, algo sobre κ_P/κ_Δ/κ_V, o sobre la Libertad Funcional de O-N7.7 una vez que se resuelva su
observable)?

**3. Tamaño del barrido y presupuesto.** ¿Cuántas reglas distintas es razonable generar y correr (decenas,
cientos)? Esto determina si se puede hacer con el motor liviano tipo CS064-068 (grafos puros, minutos por
regla) o si hace falta pensar en un subconjunto que después se valide en Phantom (mucho más caro).

**4. Qué contaría como resultado fuerte vs. débil.** El roadmap ya anticipa que "no sería necesario que TODAS
las reglas produzcan universos complejos — lo relevante sería descubrir CLASES de universalidad" (disolución,
mundo-pequeño congelado, geometría extensa). ¿Estás de acuerdo con que el objetivo sea un mapa de clases, no
un sí/no? Si es así, ¿qué proporción de reglas cayendo en la clase "geometría extensa como la nuestra" contaría
como apoyo real al principio (¿10%? ¿mayoría?) — o el criterio es otro (que exista AL MENOS una clase amplia
de reglas fecundas, no necesariamente que la nuestra sea típica)?

**5. Relación con O-N7.7.** Dado que O-N7.7 quedó abierto (el observable de "masa en sumideros" resultó
inadecuado), ¿querés que Fase V espere a que eso se resuelva, o son preguntas independientes que se pueden
avanzar en paralelo?

## Lo que NO necesito de vos todavía

El código, los controles NULL, la implementación del barrido — eso lo armo yo una vez que tenga tus respuestas
a lo de arriba. Este documento es sólo para la conversación de mañana.
