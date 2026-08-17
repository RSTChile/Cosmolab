# INFORME CS072 — Reserva B: barrido amplitud_crítica(N) (TAREA 1) y banda de memoria (TAREA 2), PARA CS
## CC ejecutó INSTRUCCION_CS072_cerrar_reservaB_umbral_escala_PARA_CC.md con Punto 0 (pasos de equilibrio)
## verificado ANTES de barrer, como exigía la instrucción. Resultado: NINGUNA de las dos hipótesis de CS se
## confirma tal como estaba planteada — pero apareció un hallazgo colateral más importante que la pregunta
## original. Nada maquillado: se reporta lo que salió, no lo que se esperaba.

## PUNTO 0 (equilibrio) — VERIFICADO, confirma la advertencia de CS
N=68: 300 pasos (9 bariones, igual a 400 → estable).
N=136: 400 pasos (19, igual a 600 → estable).
N=272: 600 pasos (39, igual a 800; A 400 DABA 38 — no equilibrado a 400, como CS temía).
N=544: 600 pasos (79, igual a 800; A 400 DABA 77 — no equilibrado a 400).
CS tenía razón: a N≥272, 400 pasos NO bastan siempre. Se usó el pasos correcto (300/400/600/600) en TAREA 1 y 2.

## TAREA 1 — amplitud_critica(N): NO SE PUDO LOCALIZAR EN EL RANGO BARRIDO [0.05, 1.0]
Con pasos equilibrados, el conteo de bariones es prácticamente CONSTANTE en TODO el rango de amplitud probado,
en las 4 escalas:
  N=68  (pasos=300): 9 bariones en las 8 amplitudes (0.05 a 1.0) — CERO variación.
  N=136 (pasos=400): 18→19, un solo escalón en amp=0.2.
  N=272 (pasos=600): 39 constante, CERO variación.
  N=544 (pasos=600): 78→79, un solo escalón en amp=0.15.
amplitud_critica quedó PEGADA AL PISO del rango (0.05) en las 4 escalas — nunca cayó a cero bariones. El
umbral real (si existe) está POR DEBAJO de 0.05; el rango pedido no lo capturó.
CONSECUENCIA HONESTA: el producto amplitud_critica×N = (3.4, 6.8, 13.6, 27.2) para N=(68,136,272,544) CRECE
LINEAL con N — pero es un ARTEFACTO DE MEDICIÓN: como amplitud_critica está pegada al piso 0.05 en las 4
escalas, el producto ×N es sólo 0.05×N por construcción, no una medición real del escalamiento. NO SE PUEDE
afirmar ni refutar "amplitud_critica×N≈constante" con estos datos. Hace falta un segundo barrido con
amplitudes MÁS BAJAS (ej. 0.001-0.05) en las 4 escalas para encontrar el verdadero punto de caída a cero.

## HALLAZGO COLATERAL (no era la pregunta, pero pesa más que la pregunta original)
La corrida completa (ADJUDICACION_CS072_corrida_completa_MATERIA_EMERGE_CS.md) reportó 11 de 27 combinaciones
de (alpha, tasa_exp, amplitud) con CERO bariones — medido a pasos=150. Aquí, con pasos EQUILIBRADOS
(300-600 según N), la materia aparece de forma ROBUSTA en TODO el rango de amplitud probado (0.05-1.0), en
las 4 escalas, SIN UN SOLO CERO. Esto apunta a que el "no-robusto" original (11/27=0) era, total o
parcialmente, el artefacto que el propio Punto 0 de CS anticipaba: pasos=150 insuficientes para equilibrar —
NO necesariamente una dependencia física real de los 3 parámetros. NO verifiqué esto re-corriendo las 27
combinaciones exactas con pasos equilibrados (fuera del alcance de esta instrucción) — es una hipótesis
fuerte, no un hecho cerrado. Si se confirma, la Reserva B podría disolverse por completo: ni Shannon, ni
umbral-tesis#2 — simplemente pasos insuficientes en la corrida original.

## TAREA 2 — banda de memoria: NINGÚN EFECTO detectado en el rango matemáticamente válido
alpha barrido en {0.5, 0.7, 0.8, 0.9, 0.95, 0.99} (N=68, pasos=300 equilibrados, amplitud=1.0 saturada):
bariones=9 EN LOS SEIS VALORES, sin ninguna variación. Ningún efecto de memoria en [0.5, 0.99].
Esto CONTRADICE el hallazgo original de CS ("memoria×2 apaga la materia": 10→1 bariones). Pero ese test
multiplicaba MEMORIA_ALPHA=0.9 ×2 = 1.8 — un valor FUERA de [0,1], donde la fórmula W_nuevo=alpha·W+
(1-alpha)·aff deja de ser combinación convexa (peso de la afinidad nueva = 1-1.8 = -0.8, NEGATIVO). Es
plausible que ese "apagón" fuera un artefacto numérico de un alpha matemáticamente inválido, no un efecto
físico de "demasiada memoria". NO verifiqué esto con código en esta corrida (no probé alpha=1.8) — queda
pendiente si CS quiere confirmarlo específicamente.

## VEREDICTO PROPUESTO POR CC (CS decide)
1. TAREA 1: NO CONCLUYENTE. Falta un barrido de amplitud MÁS BAJO (0.001-0.05) en las 4 escalas para
   localizar el umbral real (si existe) y recién ahí evaluar amplitud_critica×N≈constante.
2. TAREA 2: EFECTO NO ENCONTRADO en el rango válido [0.5,0.99]. La "banda de memoria" de la adjudicación
   anterior pudo depender de un alpha fuera de rango matemático (1.8) — no de una banda física real.
   Recomiendo verificar alpha=1.8 (y quizás 1.2, 1.5) por separado para decidir si esa fue una zona
   matemáticamente degenerada o si hay una física real de "exceso de memoria" que no llega hasta 0.99.
3. LO MÁS IMPORTANTE: re-correr el barrido de sensibilidad ORIGINAL (27 combinaciones de la corrida completa)
   con pasos EQUILIBRADOS antes de seguir discutiendo si Reserva B es "Shannon" o "umbral tesis#2" — puede
   que ninguna de las dos, y que el problema completo ya esté resuelto por el guardián de Punto 0.

## ARCHIVOS
Código: cs072_v9_umbral_escala.py
Log: cs072_v9_umbral_escala_run.log (stdout completo) / cs072_v9_umbral_escala_log.txt (mismo contenido)
Datos: cs072_v9_umbral_escala_resultados.json
Tiempo total: 20.86 min

— CC
