# INFORME CS072 — Motor rediseñado (cs072_motor_fuerzas.py) a escala, PARA CS
## CC ejecutó las 3 tareas de INSTRUCCION_CS072_motor_rediseñado_fuerzas_PARA_CC.md. Código: cs072_v10_motor_
## fuerzas_escala.py. Tiempo total: 0.43 min (el motor rediseñado es ~100x más liviano que el viejo: 3.3s vs 50s
## a N=544/pasos=400 — menos piezas, sin voto de marco ni correlación/causal). Pasos equilibrados: 300 alcanza
## y ya estabiliza en las 4 escalas (verificado contra 400, sin cambios) — no hizo falta subir a 500.

## TAREA 1 — apagar confinamiento: 0 bariones en LAS 4 ESCALAS. ADMISIBLE.
| N   | pasos | con confinamiento | sin confinamiento | admisible |
|-----|-------|--------------------|--------------------|-----------|
| 68  | 300   | 10                 | 0                  | SÍ        |
| 136 | 300   | 20                 | 0                  | SÍ        |
| 272 | 300   | 40                 | 0                  | SÍ        |
| 544 | 300   | 80                 | 0                  | SÍ        |
El artefacto NO reapareció a ninguna escala. El rediseño (B sólo la llenan las fuerzas, T nunca liga) sostiene
la prueba de admisibilidad que retractó el hallazgo anterior.

## TAREA 2 — escala: bariones = n_quarks/3 EXACTO en las 4 escalas (ratio=1.000 siempre)
| N   | n_quarks | bariones | n_quarks/3 | ratio |
|-----|----------|----------|------------|-------|
| 68  | 30       | 10       | 10.00      | 1.000 |
| 136 | 60       | 20       | 20.00      | 1.000 |
| 272 | 120      | 40       | 40.00      | 1.000 |
| 544 | 240      | 80       | 80.00      | 1.000 |
RESERVA HONESTA (no maquillar): las 4 escalas barridas tienen n_quarks EXACTO múltiplo de 3 (30,60,120,240) —
por diseño del escalado ×1,×2,×4,×8 sobre la base (30,21,10,7). Esto NO prueba que el motor reparta el RESIDUO
correctamente cuando n_quarks no es múltiplo de 3 (el MANIFIESTO pide el residuo como observable, no sólo el
caso limpio). No probé eso aquí — fuera del alcance de esta instrucción, pero queda pendiente si CS lo quiere.

## TAREA 3 — auditoría de apagado (N=68 y N=544, resultado IDÉNTICO en ambas escalas)
| Pieza          | N=68 sin ella | N=544 sin ella | ¿Actúa? |
|----------------|---------------|-----------------|---------|
| confinamiento  | 0             | 0               | SÍ ACTÚA |
| em             | 10 (=base)    | 80 (=base)      | NO actúa |
| gravedad       | 10 (=base)    | 80 (=base)      | NO actúa |
| aniquilacion   | 10 (=base)    | 80 (=base)      | NO actúa |
De las 4 piezas que ESTE motor implementa, sólo confinamiento decide el conteo de bariones. Revisé el código
para entender POR QUÉ las otras 3 no actúan (no me quedé en el número — esto es una lectura del mecanismo,
no verificado con un segundo experimento aislado, que CS puede pedir si quiere confirmarlo):
  - EM (dW_em = R_EM·carga_opuesta): su contribución (R_EM=0.10) es aditiva y MENOR que la de confinamiento
    (R_STRONG=0.30); con el umbral relativo (1.5× promedio), no alcanza para mover ningún par de un lado al
    otro del corte que ya fija el confinamiento. No está "muerta" — es subdominante EN ESTE MOTOR, a estos
    valores relativos de R_STRONG/R_EM.
  - Gravedad (dW_grav): usa masa=1 UNIFORME para todos los fermiones (no hay jerarquía de masa en este
    rediseño) → es un término CONSTANTE sumado por igual a TODOS los pares, no discrimina color/carga. No
    puede decidir QUIÉN queda ligado porque no distingue a nadie. Esperable con esta implementación, no un bug.
  - Aniquilación: reduce `viva` sólo del lado ANTIMATERIA (antiquarks/positrones); el contador de "bariones"
    filtra por `~es_anti` desde el inicio (sólo mira materia). Apagar aniquilación no puede tocar un conteo
    que ya excluye estructuralmente a quienes la aniquilación afecta. Es un no-efecto por diseño del contador,
    no evidencia de que la pieza esté rota.

## NOTA DE ALCANCE (para que no se lea de más)
Este motor rediseñado implementa 4 piezas (confinamiento, em, gravedad, aniquilación) — NO las 23 del
inventario de MANIFIESTO_FOLD_CS072.md. Es el rediseño mínimo para probar que las FUERZAS deciden la materia,
no el campo térmico — cumple ese objetivo limpio. Pero "las 23 juntas" no aplica todavía a este motor: falta
reintroducir débil, Pauli, marco/SSB, 3-cuerpos, correlación, causal, localidad, QCD(#22), sector oscuro,
espín, y el contador de HIDRÓGENO (este motor no cuenta hidrógeno en absoluto — sólo bariones) antes de volver
a hablar de un veredicto de arco. Lo dejo explícito para que la próxima adjudicación no repita el error de
celebrar de más.

## VEREDICTO PROPUESTO POR CC (CS decide)
TAREA 1 y 2: ADMISIBLE y limpio en las 4 escalas, sin reservas. TAREA 3: sólo confinamiento actúa de las 4
piezas presentes, con explicación de mecanismo para las otras 3 (no bugs aparentes, pero no verificados con
un segundo experimento aislado). Pendientes antes de re-declarar "materia emerge": (a) probar residuo con
n_quarks no múltiplo de 3, (b) reintroducir EM/hidrógeno con una escala de R_EM que le dé chance de decidir
algo (o confirmar que es correcto que sea subdominante), (c) plan para reintegrar el resto del inventario de 23
sin volver a mezclar térmica-en-ligadura.

## ARCHIVOS
Código del barrido: cs072_v10_motor_fuerzas_escala.py (usa cs072_motor_fuerzas.py sin modificarlo)
Log: cs072_v10_motor_fuerzas_escala_run.log / cs072_v10_motor_fuerzas_escala_log.txt
Datos: cs072_v10_motor_fuerzas_escala_resultados.json
Tiempo total: 0.43 min

— CC
