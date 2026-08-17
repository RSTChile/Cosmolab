# ENFOQUE 5 — Informe completo, los 30 experimentos

**Fecha:** 2026-07-25 · **Completo, no parcial** — los 30, con su estado real. Ningún veredicto de arco está adjudicado; esto es entrega cruda organizada, para que decidas tú qué cerrar y qué re-correr.

---

## 0. Los 3 arreglos — qué se hizo y qué se encontró (en simple)

**Arreglo 1 (E5.5-3 no debía inventar energía):** se cambió "sumarle un patrón nuevo al campo muerto" por "reordenar lo que ya hay" (como remover las brasas de una fogata en vez de echarle leña nueva). Verificado: conserva energía exacto. Resultado: reordenar **nunca** revive energía útil por encima de la línea base, en ninguna de las 13 fracciones probadas — solo puede empeorar.

**Arreglo 2 (el ruido se desmadraba en sistemas grandes):** el ruido por paso ahora se reparte según cuántos pasos va a durar la corrida (como repartir la misma cantidad de sal entre más o menos cucharadas, no echar la misma sal en cada cucharada sin importar cuántas haya). Verificado con números: a N=2048, antes z=0.04 (ciego) y 44% de fuga de energía; después z=98.5 y 0.03% de fuga. Se aplicó también a **E5.6-3** (su corrida original, la que detectó el problema) y a **E5.1-4** y **E5.1-2** (que tenían el mismo bug, con impacto real medido: hasta +123% de inflación en algunos números).

**Arreglo 3 (una sola regla para "energía útil"):** se escribió la regla común (la del experimento 3, E5.2-2) y se aplicó a los 5 experimentos que usaban una regla incompatible. Acá salió el hallazgo más importante del día — no es un resultado de física, es una propiedad de la regla de medir:

> **La regla común es ciega cuando el control es "barajar el campo".** Es como preguntar "¿esta baraja está en el mismo orden de antes?" usando solo la cuenta de cartas rojas y negras — barajar nunca cambia esa cuenta, así que la pregunta nunca puede notar el barajado, sin importar qué tan mezclada quede la baraja. La regla común mide "cuánto se desvió cada carta de su valor de referencia", una suma que da lo mismo sin importar el orden espacial. Como el control de la mayoría de los experimentos de esta batería es justamente "barajar el campo final", la regla común da el mismo número para lo real y para el control — **siempre, exacto, matemáticamente** — en esos experimentos. No es un error de nadie ni un bug: es que la regla común y ese tipo de control miden cosas incompatibles.

Confirmado en **4 experimentos independientes** (E5.5-3, E5.6-3, E5.6-4, y la propia regla vieja de E5.1-5 tiene el mismo problema por la misma razón): el `z` de la regla común da **0.0 exacto**, siempre, en ese tipo de control. En **E5.1-4**, donde el control NO es barajar (es una corrida física distinta, sin ruido inicial), la regla común **sí logra distinguir** — confirma que el problema es específicamente sobre el tipo de control, no sobre la regla en sí.

**Consecuencia práctica:** en los experimentos donde esto aplica, la regla vieja (que sí nota el orden espacial, vía autocorrelación) sigue siendo la que puede contestar "¿es distinto de barajado?" — la regla común sirve para comparar magnitudes entre experimentos, pero no reemplaza la vieja para esa pregunta específica.

---

## 1. TEMA 1 — Persistencia de exergía

**E5.1-1 · Supervivencia a r extremo — señal real, pero con la vieja falla del ruido activa (no re-corrido).** X sube de 0.03 (sin expansión) a ~0.9-1.0 (r≥100), separado del control (z=2.8-4.9). Corrido ANTES del arreglo del ruido — la señal cualitativa probablemente aguanta, pero hay celdas con hasta 36.6% de fuga de energía, así que los números finos no son de fiar todavía.

**E5.1-2 · Vida media sin expansión — PASS, re-corrido hoy con la regla común.** τ≈0.628/D (R²=0.9999). La regla vieja y la común dan el **mismo número exacto** aquí — no coincidencia, es que en este experimento (nunca expande) las dos reglas terminan siendo matemáticamente la misma fórmula. Extra: el arreglo del ruido sí importó — a N=524 el bug viejo inflaba el resultado 123% de más.

**E5.1-3 · Exergía en 2D — PASS, replica el 1D.** A r=100: X=0.973 (2D) vs 0.991 (1D), ambos separados del control (z≈7.5-7.8). No es un artefacto de estar en un anillo 1D.

**E5.1-4 · Umbral de ruido — corrido por primera vez hoy, con el ruido ya arreglado. Sin explosión numérica, pero sin umbral único.** Antes del arreglo llegaba a números de 27 millones (explosión); ahora queda acotado (máximo φ=5.2). Pero las 3 formas de medir NO coinciden en dónde está el "umbral": la vieja (ref. fija) ni siquiera baja, sube sin techo (artefacto de su propia referencia); una variante con otra referencia sí colapsa limpio entre amplitud 1e-3 y 3e-2; la común distingue fuerte a ruido bajo (por el aislamiento de la expansión) y débil-pero-no-ciega a ruido alto.

**E5.1-5 · Expansión no monótona — PASS, re-corrido hoy.** El orden entre perfiles (ráfaga temprana > frenante > constante > ...) se mantiene bajo la regla común. Nota técnica: el chequeo automático de "robustez" al principio pareció decir que el hallazgo desaparecía, pero era un piso numérico del código mal calibrado a la escala nueva — recalculado a mano, el patrón viejo reaparece casi idéntico.

---

## 2. TEMA 2 — Conservación del presupuesto

**E5.2-1 · Balance paso a paso — mayoría conserva, cola no.** Mediana de fuga ≈1e-9 (excelente), pero 37% de las celdas (283/768) supera el umbral fijado, con un máximo de 0.0063 en los casos más extremos.

**E5.2-2 · Anticorrelación X↔S_ent — PASS casi perfecto (es la fuente de la regla común).** r≈-0.9999 a -1.0000 en 44/44 celdas centrales.

**E5.2-3 · ¿El ruido rompe el balance? — SÍ, y es literalmente el síntoma del bug (no re-corrido).** A amplitud=1.0, la fuga de energía llega a 198 (debería quedarse cerca de 1). Este experimento en efecto MIDE el bug de Arreglo 2 en su forma más pura — no es física nueva, es el problema que hoy se arregló, documentado en vivo.

**E5.2-4 · Presupuesto por componentes — PASS muy limpio.** Peor fuga = 1.9e-13, muy por debajo de la tolerancia.

**E5.2-5 · Robustez a la resolución — PASS.** La fuga máxima cae de forma prolija con N: 1.4e-5 (N=128) → 2.2e-9 (N=2048).

---

## 3. TEMA 3 — Eficiencia de conversión (el ancla contra 4.9%/31.5%)

**E5.3-1 · Eficiencia 12 décadas — señal real pero débil.** Diferencia real-control chica pero genuina (+0.4%). Algunas celdas caen cerca de 4.9%/31.5% sin ajuste, pero justo donde hay más ruido — queda como observación cruda, no confirmación.

**E5.3-2 · Eficiencia vs ligadura — señal parcial, con definición propia declarada.** 36% de celdas (32/90) con separación real. El propio motor advierte que usó su propia definición de "eficiencia" porque el experimento que debía fijarla (E5.3-1) todavía no existía cuando se escribió.

**E5.3-3 · Estabilidad temporal — PASS fuerte.** A r alto, eficiencia≈0.96-0.996, separación hasta z=41, y el 100% de las semillas se congela y se queda así por hasta 100.000 pasos.

**E5.3-4 · Sensibilidad a axiomas — PASS de consistencia interna.** La predicción ("sin el canal de redistribución no hay nada que recortar") se cumple exacto.

**E5.3-5 · Falsación externa — negativo limpio, tras encontrar 2 defectos de instrumento.** Controlando por ellos, ninguna celda con separación real cae cerca de 4.9% ni 31.5%. El punto más estable da eficiencia≈0.2725 (z≈2.4), a 4.25 puntos de 31.5% — fuera de tolerancia.

---

## 4. TEMA 4 — Exergía y enfriamiento adiabático

**E5.4-1 · Exergía vs enfriamiento — FAIL, bien diagnosticado.** El control (sin expansión) también correlaciona con la temperatura — comparten el mismo reloj de relajación, la correlación no distingue real de control.

**E5.4-2 · Exponente de enfriamiento — FAIL, corrido por primera vez hoy.** 0/96 combinaciones dan una sola ley de potencia limpia. Pero el diagnóstico es rico: al principio el gas se enfría casi como se esperaba (n≈0.95), pero luego se aplana (n≈0.03) — dos regímenes distintos, no uno solo. El chequeo de conservación de energía es casi perfecto (2×10⁻¹⁴), así que la física del motor es confiable — el "no PASS" es real, no un artefacto.

**E5.4-3 · Reversibilidad / tiempo de no-retorno — FAIL según su propio criterio (ya estaba corrido, no hacía falta repetirlo).** Solo 12/432 combinaciones pasan (se necesitaba 55%). Hallazgo interesante: si hay ruido dinámico, el ruido mismo "revive" más energía útil de la que la difusión logra quitar — impide medir la reversibilidad limpiamente.

**E5.4-4 · Exergía espectral — PASS.** "Las escalas grandes se congelan primero" confirmado (16/16 semillas), verificado con precisión de máquina.

**E5.4-5 · Control con baño externo — PASS limpio.** Con baño térmico, la conservación se rompe activamente (como se esperaba); sin baño, se respeta a precisión de máquina.

---

## 5. TEMA 5 — Muerte térmica vs Nada

**E5.5-1 · Barrido fino ε→0 — PASS en las tres curvas (es la fuente de la "E" de la regla común).** La fuga de energía cae exactamente como ε² (medido, no impuesto).

**E5.5-2 · Tiempo de muerte térmica — la divergencia esperada NO apareció (no re-corrido).** El motor mismo lo marca: `diverge_como_se_predijo: False`.

**E5.5-3 · Reversibilidad de la muerte — re-corrido HOY con dos arreglos (1 y 3).** Con la redistribución que conserva energía exacto: nunca revive por encima de la línea base. Con la regla común: hallazgo extra — la regla común es EXACTAMENTE ciega a cualquier reordenamiento (da el mismo número, bit a bit, en las 13 fracciones), porque es una suma que no le importa el orden — solo la regla vieja (que sí mira el orden) puede contestar esta pregunta específica.

**E5.5-4 · Muerte térmica ≠ Nada — PASS muy limpio.** 240/360 celdas alcanzan "muerte", y en TODAS esas la energía total se queda en ~1.0 exacto (nunca 0) — la muerte térmica retiene el presupuesto completo; "Nada" tendría E=0.

**E5.5-5 · Universalidad del límite — PASS en entropía/energía, matiz en exergía.** Entre familias de perturbación distintas, entropía y energía convergen casi a precisión de máquina; la exergía muestra una dispersión pequeña pero real entre familias (std≈0.0056) — vale la pena mirarlo con más cuidado en algún momento, no es un PASS absoluto.

---

## 6. TEMA 6 — Definición y verificación cruzada de la exergía

**E5.6-1 · Doble medida (termo vs informacional) — FAIL contra el umbral fijado (0.9).** Hay separación real del control (0.452 vs 0.191), pero muy por debajo de lo necesario para decir que ambas miden "lo mismo".

**E5.6-2 · Energía libre F=E−TS — coherencia parcial (forma sí, escala no).** La FORMA coincide casi perfecto (correlación 0.9996), pero la escala numérica solo calza en 6.7% de las celdas.

**E5.6-3 · Invariancia a N — re-corrido HOY con dos arreglos (2 y 3), el más caro (2.35h).** Con el ruido ya arreglado, NINGUNA definición resulta "intensiva" (ni la vieja ni la común, pendiente≈-1 en ambas, muy lejos del criterio). El veredicto se movió, pero por el RUIDO, no por la definición — antes (con el bug) la pendiente era ruidosa e inconsistente (-0.57±0.12, control ciego a N grande); ahora es limpia y clara (-0.99, control fuerte en todo N). La exergía cruda NO crece con el tamaño del sistema — se satura en un techo fijo que se reparte entre más sitios.

**E5.6-4 · Sensibilidad a la referencia — re-corrido HOY.** El resultado viejo se reprodujo EXACTO (32/98 celdas invariantes, igual que antes — buena señal de reproducibilidad). La regla común, al agregarse como cuarta referencia, resultó ser ciega al control por barajado en el 100% de las celdas (mismo hallazgo transversal de la sección 0) — su "32% de acuerdo" con la referencia vieja es enteramente trivial (coincide solo donde ambas dicen "no", nunca donde la vieja dice "sí").

**E5.6-5 · Exergía informacional — hallazgo negativo notable, sin tocar (no re-corrido).** El control correlaciona MÁS que lo real (0.939 vs 0.784) — resultado invertido respecto a lo esperado. Vale la pena que alguien lo revise con calma antes de darlo por definitivo.

---

## 7. Resumen numérico

- **30/30 experimentos tienen resultado en disco.**
- **PASS claro:** E5.1-2, E5.1-3, E5.2-2, E5.2-4, E5.2-5, E5.3-3, E5.3-4, E5.4-4, E5.4-5, E5.5-1, E5.5-4 (11)
- **FAIL claro (con diagnóstico honesto, no roto):** E5.4-1, E5.4-2, E5.4-3, E5.6-1 (4)
- **Señal parcial / débil / con matices:** E5.1-1, E5.1-5, E5.2-1, E5.3-1, E5.3-2, E5.3-5, E5.5-2, E5.5-5, E5.6-2, E5.6-3, E5.6-4 (11)
- **Hallazgos que son el síntoma del bug de ruido, no física (quedan documentados así, no re-corridos):** E5.2-3
- **Ciego por construcción bajo la regla común (no es FAIL, es el instrumento equivocado para esa pregunta):** E5.5-3, E5.6-4 (y E5.6-3 en su costado NULL)
- **Pendiente de revisión manual (resultado invertido, raro):** E5.6-5

## 8. Lo que sigue siendo tuyo decidir

- De los 14 corridos con reglas viejas (E5.1-1, E5.1-3, E5.2-1, E5.2-3, E5.2-4, E5.2-5, E5.3-2, E5.3-3, E5.3-4, E5.5-2, E5.5-4, E5.5-5, E5.6-1, E5.6-5): ¿cuáles vale la pena re-correr con los 3 arreglos, y cuáles quedan como están? Candidatos más urgentes por estar directamente contaminados por el bug de ruido: **E5.1-1, E5.2-3**.
- E5.6-5 (resultado invertido) y E5.5-5 (matiz en X) piden una segunda mirada antes de reportarlos como definitivos.
- Ningún veredicto de arco (Tema completo o batería completa) está adjudicado — sigue firme la regla de no cerrar sin tu autorización explícita.
