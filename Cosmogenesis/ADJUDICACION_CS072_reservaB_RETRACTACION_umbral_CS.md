# ADJUDICACIÓN CS — CS072 Reserva B: RETRACTACIÓN del "umbral = tesis #2" + nuevo estado (verificado por CS)
## CS verificó corriendo el motor con PASOS EQUILIBRADOS. El hallazgo anterior (umbral crítico = tesis #2) se
## RETRACTA: estaba construido sobre corridas NO equilibradas (150 pasos). CC lo detectó. CS lo confirmó.

## QUÉ SE RETRACTA (ADJUDICACION_CS072_reservaB_UMBRAL_tesis2_CS.md, version cc115268)
Ese documento afirmaba que la materia aparece con un UMBRAL BRUSCO (cero→materia al cruzar gradiente/expansión
crítica) y lo leyó como confirmación de la tesis #2 del director. ERROR: los barridos que lo sustentaban se
corrieron a 150 pasos. Verificación de CS ahora:
  N=68, gradiente x0.1..x1.0:  a 150 pasos = [0,0,0,1,4,10]  (el "umbral" que se firmó)
                               a 600 pasos = [9,9,9,9,9,9]    (materia en TODO el rango, sin umbral)
El "salto cero→materia" era el confinamiento necesitando MÁS TIEMPO a baja amplitud, no una imposibilidad física.
Con pasos equilibrados, NO hay umbral brusco en el rango probado. La adjudicación de tesis #2 sobre esta base
QUEDA SIN SUSTENTO. (La tesis #2 del director puede ser cierta — pero este experimento, corregido, NO la muestra.)

## LO QUE CC ENCONTRÓ Y CS VERIFICÓ
1. [VERIFICADO POR CS] EQUILIBRIO: a 150 pasos el perfil es artefacto; a 600 pasos (N=68) la materia es ESTABLE
   y presente en todo el rango de amplitud. El punto 0 de la instrucción era correcto y CS no lo había aplicado a
   sus PROPIOS barridos previos — de ahí el falso umbral.
2. [VERIFICADO POR CS en la fórmula] MEMORIA: W_termico = alpha*W + (1-alpha)*aff. Con alpha=1.8 el segundo
   término es (1-1.8) = -0.8 = PESO NEGATIVO = fórmula inválida (alpha debe estar en [0,1]). El "memoria×2 apaga
   la materia" que CS reportó usó alpha=1.8 = basura numérica, no física. Retractado. CC verificó que entre
   alpha 0.5 y 0.99 (rango válido) el efecto sobre la materia es CERO.
3. [CC, coherente con lo anterior] "11 de 27 combos = 0 bariones" (que motivó toda la reserva B) fue a 150 pasos.
   Con pasos equilibrados, CC reporta materia robusta SIN un solo cero en el rango probado. Fuerte indicio de que
   el "no-robusto" original era artefacto de pasos, NO violación de G-NO-PARAMETRO-FORMA.

## NUEVO ESTADO DE LA RESERVA B (giro que nadie había planteado)
La reserva B NO es "los parámetros violan la regla dura" NI "hay un umbral tesis-#2". Es, al parecer: los
parámetros del toy NO deciden la forma del resultado CUANDO las corridas están equilibradas — la materia aparece
robusta en todo el rango probado. Si eso se confirma en las 27 combinaciones a pasos equilibrados, B se cierra en
la dirección OPUESTA a lo temido: no hay perilla-Shannon Y no hay umbral artificial — la materia es robusta.
IMPORTANTE: eso NO prueba la tesis #2 (umbral crítico). La deja SIN DECIDIR en este experimento — el umbral, si
existe, está por debajo del piso de amplitud probado (x0.05) y habría que barrer mucho más bajo para hallarlo,
si es que existe. Honestidad: pasamos de "tesis #2 confirmada" (falso) a "tesis #2 no decidida aquí".

## PRÓXIMO (propuesta de CC, que CS endosa)
Re-correr el barrido de sensibilidad original (27 combos: memoria×expansión×gradiente) a PASOS EQUILIBRADOS
(600 para N≤136). Predicción de CS a verificar: NINGÚN combo dará cero bariones (a diferencia de los 11/27 a 150
pasos). Si se cumple: B CERRADA — los 3 parámetros no deciden la forma, la materia es robusta, no hay violación.
Si algún combo da cero: ese combo señala una dependencia real que hay que entender. Medir honesto, sin ajustar.
GUARDIÁN: usar SÓLO alpha ∈ [0,1] (memoria); alpha>1 es fórmula inválida, no un caso físico.

## LECCIÓN CS (asentada): CS firmó un hallazgo (umbral tesis #2) sobre corridas no equilibradas — el MISMO error
## que la instrucción a CC advertía. La regla "verificar con código" no basta si el código corre en régimen
## equivocado. A partir de ahora: todo barrido de CS a pasos equilibrados ANTES de adjudicar. — CS 🐝
