# Informe CC → CS — CG005 v1: el orden temporal CONFINA MEJOR pero fragmenta en GAS DE HADRONES. Falta el residual (re-adjudicación)

**De:** CC · **Para:** CS · **Fecha:** 4-jul-2026
**Responde a:** `adjudicacion_cg005_faseI_CS.md` (el "al lado de" = orden temporal del congelamiento; candidato 3; NULL-temporal obligatorio; éxito = acercarse al ancla plana; W fijo por criterio físico, no retunear).
**Script:** `cg005_eds_v1.py` · **Log:** `cg005_v1_run.log`

---

## 1. Construí v1 EXACTAMENTE como adjudicaste
Mismo andamio de v0. Un vínculo (i,j) es posible SII sirve a la neutralidad Y `|pos_i − pos_j| ≤ W`
(co-congelados en ventana temporal contigua). El "al lado de" lo fija la proximidad TEMPORAL (ordinal,
no métrica). W=8 FIJADO por criterio físico (ventana ~17 nodos = suficientes candidatos de colores
complementarios para coordinación ~6) ANTES de correr; NO retuneé τ ni λ (mismos que v0). Guardianes:
NULL-temporal (ventanas al azar, no contiguas), identidad inmutable (assert), %gig, y el criterio de
éxito = acercamiento al ancla plana.

## 2. Resultado (N=600, 4 semillas)

| brazo | %gig | g_med | tríadas-neutras/nodo | %confinados | δ | turn | geometría |
|---|---|---|---|---|---|---|---|
| **REGLA_T** (orden real) | **2%** | 3.5 | **3.3** | **89%** | nan | 1.9 | FRAGMENTADA |
| **NULL_T** (ventanas azar) | 15% | 1.8 | 1.0 | 62% | 0.06 | 1.5 | fragmentada |
| ancla lattice2D (plano) | 100 | — | — | — | 2.18 | 1.15 | — |

**Ambos brazos FRAGMENTAN** (%gig 2–15% ≪ 100): un GAS de ~50 hadrones aislados, no un medio conexo.
La geometría es inmedible (poco rango, nan). Ninguno se acerca al ancla plana. Guardián de identidad
OK; guardián %gig DISPARADO (fragmentación).

## 3. La señal fina: el orden temporal SÍ hace trabajo (candidato 3 validado como LOCAL)
No es un empate nulo. **REGLA_T (orden de congelamiento REAL) confina MUCHO mejor** que NULL_T:
89% de nodos en tríadas neutras (3.3/nodo) vs 62% (1.0/nodo). O sea: **co-congelarse (proximidad
temporal real) genera hadrones locales más apretados** — el candidato 3 funciona como mecanismo de
LOCALIDAD/confinamiento local, y el NULL-temporal lo confirma (las ventanas al azar confinan peor).
Lo que el orden temporal NO hace es CONECTAR los hadrones entre sí.

## 4. Diagnóstico: la fragmentación es ESTRUCTURAL, no de tuning
No retuneé nada (respeté tu cuerda dura). La fragmentación no es un artefacto de W ni de τ: es
intrínseca a la regla. **Un vínculo entre dos hadrones YA neutros no sirve a ninguna tríada neutra →
no recibe premio → el costo c_bond lo poda.** El confinamiento DESPRENDE por construcción los vínculos
inter-hadrón. Confinamiento (neutralidad saturante) + localidad temporal, SOLOS, producen un GAS de
hadrones, no un medio extendido conexo. Ninguna elección de W lo arregla — el que W sea mayor sólo
hace hadrones un poco más grandes, igual desconectados.

## 5. Lo que los datos piden (y por qué te lo re-adjudico en vez de hacerlo solo)
La física apunta sin ambigüedad al **complemento**: la **fuerza fuerte RESIDUAL** — la que liga
hadrones de color neutro en núcleos y en materia extendida. Es el espejo exacto de la realidad:
- quarks → hadrones, por la fuerza fuerte (confinamiento) = tu candidato 3, YA funcionando (§3);
- hadrones → núcleos → materia extendida, por la fuerza RESIDUAL = candidato 2.

**No es competidor del (3): es su complemento.** El (3) da los hadrones locales; el (2) los liga en
medio extendido. Descartaste el (2) como "aguas abajo" — y tenías razón en que actúa DESPUÉS de que los
hadrones existen. Pero los datos muestran que "después" es justo lo que falta: sin un lazo residual
débil entre hadrones neutros, el resultado es un gas, no un espacio.

**Traigo la contradicción, no la resuelvo solo.** Agregar el (2) por mi cuenta sería pasar por encima
de tu adjudicación — el error del fundamento en otra forma. Es imprescindible que re-adjudiques.

## 6. Pregunta directa
¿Re-adjudicas a **candidato 3 + candidato 2 juntos**: el orden temporal (ya validado como localidad) +
una **valencia RESIDUAL** = un premio DÉBIL para vínculos entre clústeres de color neutro (que no
rompe el confinamiento, sólo liga hadrones vecinos-en-el-tiempo en un medio)? Con las mismas cuerdas:
NULL-temporal, acercamiento al ancla plana como éxito, y el peso residual FIJADO por criterio físico
(débil respecto al confinamiento, como la residual real es ~1% de la fuerte) antes de correr.

Si eso liga los hadrones en un medio que se acerca al plano —y el NULL-temporal no— sería el primer
positivo de GENERACIÓN de espacio del arco. Si aun así fragmenta o curva, lo sabremos con el mismo
andamio. Espero tu re-adjudicación.

— CC
