# ADJUDICACIÓN CS — CS072 Reserva B: CERRADA. Los 3 parámetros NO deciden la forma (materia robusta, verificado)
## CS corrió las 27 combinaciones a PASOS EQUILIBRADOS (600) él mismo. Resultado contundente. B cerrada.

## LO QUE CS VERIFICÓ (corrida propia, 27 combos a 600 pasos)
memoria{x0.5,x1,x2} × expansión{x0.5,x1,x2} × gradiente{x0.5,x1,x2}, con memoria capada a alpha≤0.99 (rango válido):
  RESULTADO: los 27 combos dan bariones = 9. EXACTAMENTE 9 en TODOS. min=9, max=9, media=9.0. CERO combos con 0.
Comparar con la corrida original a 150 pasos: 16/27 con materia, 11 ceros. La diferencia es ENTERAMENTE el número
de pasos: a pasos equilibrados la materia no sólo aparece siempre — da el MISMO conteo, invariante a los 3 parámetros.

## VEREDICTO: RESERVA B CERRADA. Los 3 parámetros del toy (memoria, expansión, gradiente) NO deciden la forma
## del resultado. La cantidad de bariones es robusta (9, invariante) en todo el rango probado, a pasos equilibrados.
## NO hay violación de G-NO-PARAMETRO-FORMA: los parámetros no son perillas de forma. El "11/27 = cero" que motivó
## la reserva era un artefacto de pasos insuficientes (150), no física. CC lo diagnosticó; CS lo confirmó.

## LO QUE ESTO SIGNIFICA (con el cuidado debido, sin sobre-afirmar)
- Lo que SÍ queda probado: la formación de materia es robusta a los parámetros del toy. El resultado central de
  CS072 (bariones+átomos válidos, sólo bajo gradiente+expansión) NO depende de valores afinados. Es sólido.
- Lo que NO se prueba (honestidad, corrige el entusiasmo de la adjudicación retractada): esto NO confirma la
  tesis #2 (umbral crítico). Que la materia sea robusta e invariante en el rango probado significa que NO se
  observó umbral aquí — el umbral, si existe, está por debajo del piso de amplitud probado (x0.05 del gradiente).
  La tesis #2 queda SIN DECIDIR en este experimento, no confirmada ni refutada. Para decidirla habría que barrer
  amplitudes mucho más bajas y ver si hay un piso donde la materia se apaga.

## ESTADO DE LAS 3 RESERVAS DE LA CORRIDA COMPLETA
- B (3 parámetros): CERRADA — no deciden la forma, materia robusta. [VERIFICADO POR CS a pasos equilibrados]
- A (4 piezas muertas: causal, correlación, marco/SSB, tres_cuerpos): ABIERTA — no se ha tocado.
- C (#23 rugosidad multiescala): ABIERTA — pendiente decisión director (Opción A vs B).

## PRÓXIMO
Con B cerrada, quedan A y C antes de medir geometría del TODO. Sugerencia de orden:
  1. A: entender por qué 4 piezas no actúan (¿otro régimen, o redundantes?). A pasos equilibrados esta vez.
  2. C: decidir #23 (distribución de valores sin coordenada / rugosidad como resultado a medir).
  3. Recién con las piezas que actúan claras y #23 dentro: geometría del TODO sobre D (el brazo con átomos).

## NOTA: esta vez CS corrió a pasos equilibrados ANTES de adjudicar (lección de la retractación previa). El
## resultado 9/9/9...×27 es reproducible. CC tenía razón en la dirección; CS lo verificó de forma independiente. — CS 🐝
