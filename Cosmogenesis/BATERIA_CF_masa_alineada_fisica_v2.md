# BATERÍA CF (v2) — Experimentos de emergencia de la Masa, alineados con la Física

**Director:** Alexis López Tapia · **Diseño:** Claude Science (CS) · **Fecha:** 2026-07-23/24
**Serie:** CF = Cosmo-Física (distinta de CS = topología, que el director da por cerrada).

Cada experimento lleva nombre propio, no solo número. Detalle completo de reglas, trampas T1-T7/T0
y disciplina de pre-registro: ver la instrucción original de la batería y cada
`PROTOCOLO_CF-n_PREREGISTRO.md` / `ADJUDICACION_CF-n_*.md` individual. Este documento es el índice.

| # | Nombre | Estatuto (2026-07-23/24) |
|---|--------|---------------------------|
| **CF-1** | Persistencia de la diferencia bajo expansión | **PASS cualificado — sello propuesto**, ver `ADJUDICACION_CF-1_sello_CS.md`. Pendiente de tu firma. |
| **CF-2** | Enfriar es expandir: estiramiento y caída de densidad | **PASS** (NULL muerde, T4 OK) — con reserva: la variabilidad entre semillas es casi nula (PDE cuasi-determinista), el "10/10" no son 10 confirmaciones independientes. Ver resultados en `Cosmogenesis-Web/results/CF2_estiramiento/`. |
| **CF-3** | 1ª emergencia — masa elemental como cambio de fase del vacío (tipo Higgs) | **En pausa — a discutir con CS antes de codificar** (terreno resbaladizo, riesgo T0). Antecedente relevante ya en disco: `Higgs_TEST_REAL` v1-v4 + `suite_crono_higgs` (22-jul), con germen de señal (v3: rate_signal=0.90 multi-seed) pero suspendido por violar orden cronológico (masa antes de la ruptura en 40-57% del barrido). No re-implementar desde cero sin revisar eso primero. |
| **CF-4** | 2ª emergencia — masa como energía de ligadura (tipo CDC) | **FAIL con causa identificada** (no un negativo ciego): ratio_lig (m2/m1) nunca cerca del umbral 5.0; diagnóstico = coeficientes D_PHI/R0/U heredados de v6 sin recalibrar, nunca barridos — violó T1 sin querer. Ver `Cosmogenesis-Web/results/CF4_ligadura/`. |
| **CF-4b** | ¿Existe un régimen donde la masa-ligadura domina? | **En curso** — ver `INSTRUCCION_CF-4b_masa_ligadura_barrido_acoplamiento_PARA_CC_y_Grok.md`. |
| **CF-5** | Cronología de la masa: nace en el confinamiento, no tras el átomo | Bloqueado por CF-4b (reutiliza su motor). |
| **CF-6** | Contingencia: ¿es nuestra configuración única o una entre muchas? | Bloqueado por CF-4b/CF-4 (reutiliza histograma de k). |

## Nota sobre la ronda anterior (etapa 7 / Tracks A-E)

Los 5 tracks de la ronda "etapa 7" (segregación de masa, ancla inercial, diagnóstico de E_mutual,
Kepler dinámico, auditoría anti-Shannon) midieron todos en la época equivocada (post-átomo/gravedad
— fila 8 de la línea de tiempo, no la fila 5 de confinamiento). Sus resultados no se descartan:
sirven como **control negativo para CF-5** (si la masa nace en confinamiento, no debería requerir
gravedad post-átomo para aparecer — y en efecto, ninguno de esos tracks encontró una señal limpia
ahí). El track de auditoría (E) sigue siendo válido en general: encontró deuda real en TEST_RHO
(reparada por CF-2) y dos bugs de orquestación en `motor_1a7/pipeline.py` (aún sin corregir, a
la espera de que CS los arregle tras cerrar esta batería).
