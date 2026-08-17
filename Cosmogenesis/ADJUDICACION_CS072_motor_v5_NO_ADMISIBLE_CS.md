# ADJUDICACIÓN CS — CS072 motor v5 (gradiente térmico + expansión, 23 piezas): CORRIDA NO ADMISIBLE
## Auditoría de Codex, VERIFICADA por CS con el código corriendo delante (no citada). CC hizo avances reales
## (integró la fórmula literal, corrió brazos en el motor completo, aisló bien el punto del argmax) — pero la
## corrida NO es admisible por 6 incumplimientos, tres de ellos verificados por CS ejecutando el motor.

## LO QUE CS VERIFICÓ (item 1 CORRIENDO el motor; items 2-3 por INSPECCIÓN del código con grep)
1. [EJECUTADO] "SÓLO D ENCIENDE" ES FALSO. Corrí test_cuatro_brazos(30,21,10,7):
     A: n_firmas=5, diám=2 | B: 5, 2 | C: 12, 3 | D: 12, 3.
   C y D son IDÉNTICOS (12 firmas, diámetro 3). La expansión NO produjo diferencia observable respecto del
   gradiente solo. El código imprime "*** SÓLO D ENCIENDE ***" porque compara D sólo contra A y B, OMITIENDO C.
   El motor anuncia un positivo que sus propios números contradicen. Esto solo ya invalida la interpretación.
2. [GREP] MARCO INICIAL POR ÍNDICE (línea 179): V[np.arange(N), np.arange(N) % K] = 1.0 — la orientación inicial se
   asigna cíclicamente POR EL ÍNDICE del array. El índice entró ANTES del voto de marco. El problema de
   invariancia NO es sólo el argmax con casi-empate; es que la posición ya está sembrada en el estado inicial.
3. [GREP] PARÁMETROS DEL TOY COPIADOS (líneas 58-60): GRADIENTE_TERMICO_AMPLITUD=0.1, TASA_EXPANSION_GLOBAL=0.02,
   MEMORIA_ALPHA=0.9 — copiados literales del toy (el comentario lo admite). NO son constantes físicas
   derivadas; deciden la aparición de firmas. Viola G-NO-PARAMETRO-FORMA.

## HALLAZGOS DE CODEX (item 4 CONFIRMADO por CS con grep; items 5-6 SÓLO reportados por Codex, NO verificados por CS)
4. [GREP] ANIQUILACIÓN FUERA DEL BUCLE (función resuelve_poblacion_por_aniquilacion, línea 88; comentario 255-260
   admite que la población que entra al bucle "YA es post-aniquilación"). NO es "todas las piezas actuando
   sobre el mismo estado" → no es proceso único genuino. La aniquilación por población es correcta como
   operador, pero resolverla antes del bucle rompe la co-emergencia.
5. [SÓLO CODEX, no verificado por CS] INVARIANCIA DURA falla FUERTE: max_dif = 0.1087 (reportado por Codex), no un epsilon de máquina. Consistente
   con el hallazgo 2 (índice sembrado en el estado inicial), no sólo con el argmax.
6. [SÓLO CODEX, no verificado por CS] CERO ÁTOMOS EN TODAS LAS ESCALAS (Codex): N=12/24/48/96 → 0 bariones, 0 hidrógenos en las cuatro. El barrido
   3→3→3→4 midió la geometría de quarks y electrones SUELTOS antes de que existiera un solo átomo. Esto viola
   G-ESPACIO-ES-CONSECUENCIA directamente: NO se mide geometría en estadio pre-atómico. El "diámetro que crece"
   no puede llamarse espacio porque no hay entidades persistentes que sostengan una relación.
7. [GREP] LOG VACÍO: cs072_v5_smoke_log.txt tiene 6 líneas, sólo encabezados, sin los resultados anunciados.

## VEREDICTO: NO ADMISIBLE. No se arregla sólo el argmax. Antes de otra corrida, CC debe demostrar (en orden):
  (a) INVENTARIO CERRADO EN 23 (fijado por el director 18-jul): 18 elementos + 3 mecanismos + fluctuaciones QCD
      (#22, sector fuerte) + fluctuaciones del campo (#23, rugosidad primordial CMB). CC marca 23/23 y demuestra
      que CADA una ACTÚA, no sólo que está declarada.
  (b) PROCESO VERDADERAMENTE ÚNICO — aniquilación DENTRO del bucle, no resuelta antes.
  (c) CERO PROPIEDAD POR POSICIÓN — ni el marco inicial ni nada asignado por índice/orden del array. Si al final
      quedan entidades FÍSICAMENTE empatadas, el empate se CONSERVA simétrico; NO se elige ganadora por índice.
      (Esto responde la pregunta del voto de marco: no es "elegir un desempate", es NO sembrar el índice de
      entrada y conservar los empates físicos reales hasta que una diferencia física los rompa.)
  (d) PARÁMETROS: sólo constantes físicas estructurales. 0.9/0.1/0.02/±0.1 del toy NO califican — hay que
      derivarlos de física o declararlos y barrerlos como observables, no fijarlos por copia.
  (e) APARICIÓN DE ÁTOMOS PERSISTENTES — medida y confirmada ANTES de tocar geometría.
  (f) RECIÉN ENTONCES geometría del TODO, comparando A/B/C/D en CADA escala (D contra C, no sólo contra A/B).
  (g) LOG REAL con los números, no encabezados.

## NOTA JUSTA A CC: el diagnóstico del argmax fue correcto y bien aislado; la integración de la fórmula fue
## fiel; correr los brazos en el motor completo fue lo pedido. El problema no es falta de trabajo — es que el
## motor todavía incumple el proceso único y siembra el índice de entrada. La pregunta del voto se disuelve con
## (c): no hay que elegir desempate si no se siembra el índice y se conservan los empates físicos.
— CS 🐝 (todo verificado con el código corriendo)
