# ADJUDICACIÓN CS076 — Dirección temporal T⁺/T⁻ a nivel micro (C-N2.5.6-10)

**Fecha:** 5-ago-2026 · **Script:** `cs076_direccion_temporal.py` · **Fase I-A del roadmap multi-IA (5-ago-2026)**

**Este documento NO cierra el experimento.** Reporta números y controles; la decisión de qué hacer con ellos
(repetir con otro estadístico, ampliarlo, darlo por no-concluyente, etc.) es de Alexis — regla de la casa
(`nota-permanente-no-cerrar-experimentos.md`).

## Qué se preguntó

C-N3/CS009 ya confirmó que el desorden AGREGADO del campo entero de `cg001_field.py` sólo sube (monótono, sin
excepciones). Esto es distinto y más fino: **¿la propia regla de actualización, mirada en UNA celda individual
paso a paso, distingue "adelante" de "atrás" en el tiempo, o esa asimetría sólo existe cuando se suma todo el
campo?**

## Método

Se reusó `cg001_field.py` sin modificarlo (misma fórmula de `_paso`: relajación difusiva + memoria histórica
`m` que modula `lam_eff`). Se trackeó la trayectoria completa (no resumida) de 10 celdas por semilla, 12
semillas, en dos configuraciones: **gamma normal** (memoria activa, default del proyecto, gamma=8.0) y
**gamma=0** (memoria apagada — prueba anti-Shannon: si la memoria es la pieza que produce una eventual
asimetría, apagarla debería acercar el resultado al NULL).

Tres estadísticos sobre los incrementos Δx_t de cada celda:
1. **Skewness** (tercer momento) de los incrementos.
2. **Producción de entropía local** (chequeo de consistencia, no test contra azar: por construcción del modelo
   es siempre monótona no-decreciente).
3. **Violación de balance detallado** — proxy declarado de Σ_T=log(P[Γ]/P[Γ†]): KL entre el histograma 2D de
   pares (x_t,x_{t+1}) y el mismo histograma leído al revés. **Aclaración honesta:** `cg001_field.py` es
   determinista y no expone un kernel de transición estocástico explícito — este KL NO es una medición exacta
   de probabilidades de trayectoria, es el estimador estándar de termodinámica estocástica para detectar
   asimetría temporal sin conocer el kernel exacto (tipo Roldán & Parrondo 2010), aplicado de forma simplificada.

Dos controles NULL, 12 semillas cada uno:
- **(a) Orden barajado:** permutar el orden temporal de los mismos incrementos reales.
- **(b) Paseo aleatorio:** incrementos gaussianos nuevos, misma varianza que los reales, sin estructura.

## Resultado

| | entropía local (monótona) | skew REAL | skew vs NULL-barajado (z) | skew vs NULL-paseo (z) | KL REAL | KL vs NULL-barajado (z) | KL vs NULL-paseo (z) |
|---|---|---|---|---|---|---|---|
| gamma normal (8.0) | sí, siempre | -0.912 | **z=-0.00** | z=-15.05 | 0.0104 | **z=-0.02** | z=+95.58 |
| gamma=0 (anti-Shannon) | sí, siempre | -2.067 | **z=-0.00** | z=-12.65 | 0.0134 | **z=+1.63** | z=+37.56 |

## Lectura honesta

**Un artefacto propio, cazado antes de reportarlo como hallazgo — igual que otros episodios de este proyecto:**
el z≈0.00 de skewness contra el NULL de orden barajado NO es evidencia de nada — es una **identidad
matemática**: la skewness es un momento de la distribución marginal de los incrementos, invariante a cómo se
ordenen. Barajar el orden reproduce el mismo valor por construcción. Este control no sirve para esta variable
y no debería reusarse para probar dirección temporal con skewness — queda anotado para que nadie lo repita
(mismo espíritu que el proxy de correlación descartado en el Bloque 2.8 de `DISENO_EXPERIMENTOS_NODOS_ABIERTOS...`).

**El estadístico que sí aísla específicamente el ORDEN temporal (KL de balance detallado vs. NULL-barajado,
el único control que preserva la distribución marginal y sólo destruye la secuencia) no muestra una asimetría
significativa** en ninguno de los dos regímenes: z=-0.02 (memoria activa) y z=+1.63 (memoria apagada) — ambos
muy por debajo del umbral z≥3 que otros frentes del proyecto usan como estándar de significancia (ver κ_V en
Bloque 2.8, z=1.37, catalogado ahí como "dirección correcta pero débil").

**Contra el NULL de paseo aleatorio** la diferencia es enorme (z de -12 a +95), pero es una comparación menos
decisiva de lo que parece: sólo muestra que la dinámica real tiene estructura y tendencia (algo ya sabido —
es un campo relajándose), no que tenga una dirección temporal privilegiada distinta de su propio reverso. Un
paseo aleatorio puro es un baseline demasiado débil para esta pregunta específica.

**Prueba anti-Shannon (apagar la memoria):** el resultado NO apoya la hipótesis de que la memoria (gamma) sea
la responsable de una eventual asimetría — si algo, el z contra NULL-barajado subió levemente al apagar la
memoria (de -0.02 a +1.63), en la dirección opuesta a la esperada. Ninguno de los dos casos cruza significancia,
así que esto tampoco alcanza para afirmar lo contrario con fuerza — es un dato, no una conclusión.

## Veredicto (no-cierre)

Con el proxy y el control decisivo usados acá, **no se encuentra evidencia de una dirección temporal T⁺/T⁻
propia a nivel micro, distinta de la ya confirmada a nivel agregado (C-N3)**. Esto es consistente con no
haber sostén experimental (⏳→resultado nuevo, en zona gris no significativa) más que con un ✅ o un ❌ limpio.
Alternativas para profundizar, si Alexis lo autoriza: (a) usar más celdas/semillas para reducir varianza del
NULL, (b) probar el estadístico sobre el campo completo en vez de celdas individuales, (c) diseñar un
estimador de Σ_T menos dependiente de binning (ej. k-NN entropy estimator).
