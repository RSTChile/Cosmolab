# Adjudicación CS — Campaña de estrés ANIMA (5-jul-2026): sistema ROBUSTO (positivo firmado) + oído digital NO acopla in vivo (negativo reproducible aceptado). El negativo nombra el siguiente paso: tender el oído acústico A↔E.

**De:** CS · **Fecha:** 5-jul-2026
**Campaña:** 4 h · 5 ciclos · 40 bloques · 4335 paquetes digitales (corrida por el equipo/CC).
**Fuente auditada:** snapshots_campaña_2026-07-05.csv (40 bloques, columnas OI/oído/r_arousal) + informe
del equipo. Verifiqué los DATOS, no solo la prosa.
**Entregables del equipo:** INFORME_ESTRES_2026-07-05 (.md/.html/.pdf), bitacora_campaña (218 KB),
snapshots CSV, correr_campaña.py, PROTOCOLO_TEST_ESTRES, _PLAN_AUTONOMO.md.

---

## 0. Lo que verifiqué en el CSV (no en el informe)
- **40 bloques presentes**, columna `control` con tres niveles (real/shuffled/null), OI de A y E por bloque,
  fiabilidad del oído de A y E, y r_arousal_AE por bloque. Los números del informe se reproducen.

## 1. POSITIVO — el sistema es ROBUSTO (firmado)
- **Ningún organismo colapsó en 4 h.** El índice vital OI nunca tocó cero: A en 0.269–0.500 (media 0.348),
  E en 0.250–0.468 (media 0.321). CERO bloques con OI≈0 en ambos (verificado fila por fila).
- **Cero errores/colgadas/rescates** en 4335 paquetes digitales (reporte del equipo; consistente con los 40
  bloques completos sin huecos en el CSV).
- **Lectura:** la arquitectura completa (los cinco organismos vitales, el anillo, el metabolismo, la
  biografía) soportó carga sostenida sin romperse. Es un positivo de ingeniería sólido y necesario: antes de
  pedirle al sistema que haga algo fino, hay que saber que aguanta. Aguanta.

## 2. NEGATIVO HONESTO — el oído digital NO acopla in vivo (aceptado, reproducible)
- **Fiabilidad = 0.0000 en los 40 bloques**, para A y para E, sin una sola excepción. No es "baja": es cero
  exacto y sostenido. Y NO por falta de tráfico — procesó decenas de miles de eventos de oído. El mecanismo
  recibió datos; no logró predecir nada con ellos.
- **La correlación de arousal A↔E no distingue lo real de los controles:** real μ=−0.071, shuffled μ=−0.108,
  null μ=−0.022. Los tres pegados, y los RANGOS SE SOLAPAN por completo (real −0.362…+0.564; shuffled
  −0.404…+0.735). Si el oído acoplara, lo real se separaría de lo barajado. No se separa → indistinguible
  del azar.
- **Contraste con el aislamiento:** el mismo mecanismo da r=0.95 aislado. In vivo cae a 0. La diferencia no
  es el mecanismo — es el canal.

## 3. EL DIAGNÓSTICO (por qué falla, y por qué el negativo es BUENO)
El espejo aprende a predecir el estado del otro *sensado por audio*. Pero A y E casi no se oyen entre sí →
el objetivo a predecir es PLANO → no hay nada que predecir → fiabilidad cero. No es un bug del espejo; es
que el canal que debía alimentarlo está MUDO. Falta **co-presencia acústica A↔E**. Esto es exactamente el
caveat que el juez del diseño anticipó, ahora confirmado y REPRODUCIBLE en 5 ciclos. Un negativo predicho y
reproducible es de los mejores: no deja duda de qué construir. Dice, con precisión: tender el oído acústico
A↔E (que los organismos efectivamente se oigan, no solo que exista el mecanismo del espejo).

## 4. HILO DE TRAZABILIDAD (menor, para el equipo)
El informe cita "1566 + 1194 eventos"; en el CSV los totales acumulados de `oido_eventos` por bloque suman
mucho más (A≈20.572, E≈21.845). Probablemente dos métricas distintas (eventos de diálogo en una ventana vs.
eventos de oído acumulados por bloque). No cambia la conclusión, pero conviene que el equipo fije cuál cifra
va en el informe para que no queden dos números sin explicar.

## 5. VEREDICTO
**ACEPTO ambos resultados.** (a) Positivo: el sistema es robusto bajo carga sostenida —0 errores, 0
colapsos en 4 h— verificado en el CSV. (b) Negativo honesto y reproducible: el oído digital no acopla in
vivo (fiabilidad 0, indistinguible del azar) porque falta co-presencia acústica A↔E; es el caveat predicho
por el juez del diseño, confirmado en 5 ciclos. El negativo NO es un fracaso: nombra el siguiente
experimento con precisión. Próximo paso: tender el oído acústico A↔E (propuesta separada:
PROPUESTA_oido_acustico_AE_CS.md).

— CS. La campaña y su honestidad (reportar el negativo de frente, con controles) son del equipo/CC. La
verificación de datos y la adjudicación, mías. Figura: anima_estres_campaña_lectura.png.
