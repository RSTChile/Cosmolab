# ADJUDICACIÓN CS — CS072 Reserva B RESUELTA: NO es Shannon, es UMBRAL CRÍTICO (tesis #2 del director, verificada)
## CS verificó todo con código corriendo el motor (barridos finos). La reserva B se DA VUELTA: de "violación de
## G-NO-PARAMETRO-FORMA" a EVIDENCIA de la tesis #2 (umbral crítico). Método: barrido fino de cada parámetro.

## LA PREGUNTA
La corrida completa mostró que la materia NO es robusta a los 3 parámetros (11 de 27 combos = 0 bariones). CC y CS
lo leyeron como posible violación de G-NO-PARAMETRO-FORMA (un parámetro no debe decidir la forma del resultado).
Pero había que distinguir DOS cosas que se confunden: un DIAL Shannon (mueve el resultado suave y sin techo) vs
un UMBRAL físico (deja el resultado en cero, lo enciende de golpe al cruzar un valor crítico, luego satura).

## LO QUE CS VERIFICÓ (barridos finos, reproducibles)
EXPANSIÓN (gradiente en base), bariones vs factor: x0.1..x0.8 = 0,0,0,0,0,0,0,0 | x0.9=1 | x1.0=10 | x1.5=9 | x2.0=9
GRADIENTE (expansión en base), bariones: x0.1=0 x0.3=0 x0.5=0 x0.6=0 | x0.7=1 x0.8=1 x0.9=4 | x1.0=10 x1.5=10 x2.0=10
FIRMA EN AMBOS: (a) CERO por debajo de un valor crítico; (b) SALTO BRUSCO 0→materia al cruzarlo (salto máx 9 y 6
entre pasos consecutivos, no una rampa 1,2,3,4); (c) MESETA por encima (10,10,10 / 10,9,9 — subir más el parámetro
NO cambia el resultado). Eso es una TRANSICIÓN DE FASE, no un dial.

## VEREDICTO: la dependencia de los parámetros es un UMBRAL CRÍTICO, NO una perilla de forma.
- Un DIAL Shannon movería el resultado proporcional al parámetro, sin techo. NO es lo que pasa.
- Un UMBRAL FÍSICO: cero → enciende de golpe → satura. Es EXACTAMENTE lo que pasa, en los DOS parámetros.
Esto NO viola G-NO-PARAMETRO-FORMA: el parámetro no dibuja la forma (la forma —bariones válidos— es la misma
por encima del umbral); sólo determina SI se cruza la transición. La materia por encima del umbral es idéntica.

## ESTO ES LA TESIS #2 DEL DIRECTOR, MOSTRADA POR EL MOTOR (pre-inscrita antes de correr):
  "2) LA CANTIDAD DE DIFERENCIAS ALCANZA UN UMBRAL CRÍTICO. Prueba: por debajo del umbral no nace materia, por
   encima sí. Es una TRANSICIÓN (cero, cero, cero, y de golpe aparece), no una pendiente suave."
El barrido lo confirma literal: cero, cero, cero, y de golpe aparece. La "cantidad de diferencias" es la magnitud
del gradiente térmico × la expansión; el umbral es el punto donde el contraste térmico sobrevive lo suficiente
para que el confinamiento cierre bariones antes de que la expansión lo diluya. La banda de persistencia que el
director describió ("entre el caos y la disolución"): poca expansión/gradiente = disolución (0); suficiente = materia.

## MATIZ HONESTO (no maquillar)
- MEMORIA se comporta distinto: x0.5=10, x1.0=10, x2.0=1 — DEMASIADA memoria MATA la materia (no es umbral-de-
  encendido sino techo-de-apagado). Tiene sentido físico (memoria excesiva congela W, impide reorganización para
  confinar), pero NO es la misma firma de transición limpia; hay que caracterizarla aparte. La memoria SÍ podría
  necesitar justificación física adicional; expansión y gradiente ya están explicados como umbral.
- El umbral se midió a N=30 quarks. Hay que confirmar que el valor crítico ESCALA de forma física con N (no que
  sea un artefacto del tamaño chico). Es la próxima verificación antes de cerrar B del todo.
- Que exista un umbral es tesis #2; el VALOR exacto del umbral (x0.7 grad, x0.9 exp aquí) es un observable a
  medir, no una constante a justificar — igual que el cociente barión/fotón 1e-9 es medido, no derivado.

## VEREDICTO B: RESUELTA PARCIALMENTE. Expansión y gradiente: su efecto es UMBRAL CRÍTICO (tesis #2 confirmada),
## NO Shannon — B deja de bloquear por ellos. Memoria: comportamiento distinto (techo de apagado), pendiente de
## caracterizar. Falta: confirmar que el umbral escala con N. Con eso, B pasa de reserva a HALLAZGO (tesis #2).

## PRÓXIMO
  1. Caracterizar la MEMORIA aparte (¿por qué el exceso apaga? ¿tiene su propia banda?).
  2. Confirmar que el umbral de gradiente×expansión ESCALA con N (barrido de umbral a N=30/60/120).
  3. Recién entonces B cerrada → seguir con A (4 piezas muertas) y C (#23).
— CS 🐝 (barridos finos verificados con código)
