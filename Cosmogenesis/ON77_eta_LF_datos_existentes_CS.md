# O-N7.7: operacionalizar η_LF con datos YA existentes (jerarquía CS073)

**Encargo:** operacionalizar Ω_proc, Ω_op y LF para los 5 sistemas de la jerarquía de controles CS073
(REAL, NULL-1, NULL-2, NULL-3, grafo-random+layout), y calcular η_LF = LF/|Ω_proc| con datos que ya
estaban en disco — sin correr Phantom, sin cómputo nuevo. **No se declara cierre ni veredicto sobre
O-N7.7 ni sobre CS073 — sólo se reportan números y definiciones. La lectura final es de Alexis.**

Fuentes leídas (sin tocar): `NULL1_bateria_completa_CS.md`, `NULL2_bateria_completa_CS.md`,
`NULL3_resultado_CS.md`, `NULL3_robustecido_motivos_dosis_CS.md`, `TEST_layout_vs_identidad_grafo_CS.md`,
`REAL_semillas_adicionales_CS.md`.

---

## 1. Ω_proc — espacio de procesamiento

**Definición elegida: conteo directo de triángulos del grafo generador** (proxy de "grados de libertad
efectivos / mecanismo estructurado", ya medido en `null3_motivos_directos.py` y
`grafo_random_motivos.py`, no recalculado aquí).

| sistema | Ω_proc (triángulos) | de dónde |
|---|---|---|
| REAL | **2780** | `null3_motivos_directos.py`, malla causal real, N=2000 |
| NULL-3 (tol=0.2) | **2005** (−27.9%) | mismo script, seed=501 |
| grafo-random+layout | **21** (−99.2%) | `grafo_random_motivos.py`, seed=701 |
| NULL-1 | **0** | no aplica — no hay grafo (ángulo isótropo aleatorio, sin malla causal ni `layout_resortes`) |
| NULL-2 | **0** | no aplica — no hay grafo (Zel'dovich sobre grilla, sin malla causal ni `layout_resortes`) |

**Por qué esta elección y no las otras candidatas:**
- Se descartó "espacio completo de grafos Erdős-Rényi posibles" (C(C(n,2), m) ≈ 10^varios-miles) o "nº
  de double-edge-swaps válidos bajo el filtro de longitud" como Ω_proc porque **sólo son definibles para
  el mecanismo de grafo-random / NULL-3 respectivamente** — no hay forma no forzada de aplicarlos a
  NULL-1/NULL-2 (que no pasan por ningún grafo), así que no sirven para una tabla comparativa de los 5
  sistemas a la vez.
- Triángulos/motivos sí tiene una lectura que se extiende, de forma honesta, a los 5: para NULL-1/NULL-2
  el valor es **0 por construcción**, no por elección arbitraria — estos dos controles literalmente no
  tienen ningún proceso relacional/histórico (así los define el propio encargo), así que el proxy de
  "mecanismo estructurado" que produjeron es cero.

**Limitación honesta:** el conteo de triángulos de REAL/NULL-3/random se midió en **una sola semilla
representativa** de cada batería (seed=501 para REAL/NULL-3, seed=701 para random), no en las 8 (ó 6)
semillas de cada batería completa. Para REAL esto es exacto (la malla causal es idéntica en las 6
semillas REAL — sólo varía `seed_layout`, nunca la topología del grafo, ver `REAL_semillas_adicionales_CS.md`).
Para NULL-3 y random, cada semilla de la batería usa un swap/grafo distinto, así que el número real varía
semilla a semilla (la tasa de aceptación del swap NULL-3 sí varió entre 0.6% y 0.7% en las 8 semillas
de `NULL3_resultado_CS.md`) — se usa el valor de una semilla como representativo, no medido en las 8.

---

## 2. Ω_op — dominio operativo

**Definición elegida: masa total media en sumideros al final de la corrida**, el mismo observable ya
usado en toda la jerarquía CS073 (se reporta también nº de sumideros como referencia secundaria).

| sistema | Ω_op (masa media en sumideros) | DE | n semillas | nº sumideros |
|---|---|---|---|---|
| REAL | **2196.47** | 95.98 | 6 | 8/8 en las 6 |
| NULL-3 (tol=0.2) | **2186.68** | 53.16 | 8 | 8/8 en las 8 |
| grafo-random+layout | **1143.28** | 32.54 | 8 | 8/8 en las 8 |
| NULL-1 | **0.0** | 0.0 | 8 | 0/8 |
| NULL-2 | **0.0** | 0.0 | 8 | 0/8 |

Esto no requiere justificación adicional — es el observable "de siempre" de la jerarquía, ya validado
como la métrica que separa REAL de NULL en todos los escalones anteriores.

---

## 3. LF — Libertad Funcional (la parte delicada)

Un sumidero no tiene conducta ni repertorio de respuesta, así que LF no puede medirse en el sentido
conductual habitual. Se evaluaron **dos candidatas no-conductuales**, ninguna plenamente satisfactoria,
y se documentan ambas con sus problemas — siguiendo la instrucción del encargo de que "no se pudo
operacionalizar LF de forma limpia, y por qué" es un resultado válido.

### Candidata A — LF ≈ Ω_op (simplificación fuerte)

Si no hay repertorio conductual que medir, la opción más simple —y la más honesta sobre sus límites— es
**asumir que la "capacidad funcional lograda" en este sustrato ES el dominio operativo mismo** (no hay
un eje independiente de LF que se pueda separar de Ω_op con los datos que hay). Bajo esta simplificación,
η_LF colapsa a **η = Ω_op / Ω_proc**, que sí es calculable para los 5 sistemas y es la lectura más
literal de "cuánta estructura final por unidad de mecanismo".

**Problema declarado:** esto NO es una medición independiente de LF — es renombrar Ω_op. Se reporta
igual porque es la única opción que no requiere inventar un número, y porque de todas formas ilustra la
pregunta central de O-N7.7 (¿menos mecanismo puede sostener el mismo o más dominio operativo?).

### Candidata B — LF ≈ diversidad entre semillas (coeficiente de variación, CV = DE/media)

Interpretando LF como "variabilidad/diversidad de los resultados finales viables que el sistema puede
producir desde distintas semillas" (la sugerencia explícita del encargo). Se usa CV en vez de DE crudo
para que sea comparable entre sistemas de escala de masa muy distinta.

| sistema | CV (%) entre semillas |
|---|---|
| REAL | 4.370% |
| NULL-3 | 2.431% |
| grafo-random+layout | 2.846% |
| NULL-1 | 0/0 — INDEFINIDO |
| NULL-2 | 0/0 — INDEFINIDO |

**Problema declarado, importante:** el signo de esta candidata es ambiguo y no se resuelve con los
datos disponibles. Un CV más ALTO puede leerse de dos formas opuestas:
- **(i) más "libertad" = más diversidad de resultados** (lectura literal del encargo) → CV alto = LF alto.
- **(ii) más "libertad funcional" = más confiabilidad/canalización** (el sistema alcanza consistentemente
  un resultado viable pese a partir de semillas distintas — lectura más cercana al uso conductual
  habitual de LF, donde "repertorio" no es lo mismo que "ruido") → CV alto = LF BAJO, y 1/CV sería el
  proxy correcto.

No hay forma de decidir entre (i) y (ii) sólo con estos números — es una elección teórica previa que
este informe no puede zanjar. Se reportan ambas orientaciones en la tabla de η_LF (sección 4) para no
esconder la ambigüedad.

**Conclusión honesta sobre LF:** no se logró una operacionalización no-conductual de LF que sea a la vez
(a) independiente de Ω_op y (b) de signo no-ambiguo. La Candidata A es independiente-de-signo pero
redundante con Ω_op. La Candidata B tiene contenido propio (mide algo que Ω_op no mide — variabilidad
entre corridas) pero su signo depende de una decisión teórica no resuelta aquí.

---

## 4. Tabla comparativa de η_LF — los 5 sistemas, las 3 variantes

| sistema | Ω_proc | Ω_op | η = Ω_op/Ω_proc (Cand. A) | CV (%) (Cand. B) | η_LF = CV/Ω_proc (Cand. B, signo i) | η_LF = (1/CV)/Ω_proc (Cand. B, signo ii) |
|---|---|---|---|---|---|---|
| **REAL** | 2780 | 2196.47 | **0.7901** | 4.370 | 0.001572 | 0.00823 |
| **NULL-3** | 2005 | 2186.68 | **1.0906** | 2.431 | 0.001213 | 0.02052 |
| **grafo-random+layout** | 21 | 1143.28 | **54.4419** | 2.846 | 0.135533 | 1.67308 |
| **NULL-1** | 0 | 0.0 | **INDEFINIDO (0/0)** | 0/0 | INDEFINIDO | INDEFINIDO |
| **NULL-2** | 0 | 0.0 | **INDEFINIDO (0/0)** | 0/0 | INDEFINIDO | INDEFINIDO |

---

## 5. Lectura de los números (sin cerrar nada)

**Lo que SÍ confirma la intuición de O-N7.7:**

- **NULL-1 y NULL-2 dan η_LF INDEFINIDO en las 3 variantes**, no un número bajo cualquiera — 0 dividido
  entre 0. Esto es, de hecho, la lectura MÁS fuerte posible de "sin proceso histórico/relacional no hay
  ni mecanismo (Ω_proc) ni dominio operativo (Ω_op) que relacionar" — el patrón anticipado en el encargo
  ("NULL-1/2 con η_LF≈0 o indefinido") se cumple literalmente, no por aproximación.
- **NULL-3 tiene η (Candidata A) LIGERAMENTE MAYOR que REAL** (1.09 vs 0.79): con ~28% menos triángulos
  que REAL, NULL-3 sostiene una masa en sumideros prácticamente igual (2186.68 vs 2196.47, estadísticamente
  indistinguibles según `NULL3_resultado_CS.md`, p=0.42). Leído en el lenguaje de O-N7.7, esto es
  exactamente "menos mecanismo, capacidad igual o mayor" — la lectura de condensación exaptativa, no de
  acumulación adaptativa.

**Lo que COMPLICA la lectura simple — y de forma seria:**

- **El grafo-random+layout, no NULL-3, tiene el η_LF más alto de los 5 sistemas por un margen enorme**
  (54.4 en la Candidata A — 50 veces el de NULL-3, 69 veces el de REAL). Esto pasa porque su Ω_proc
  (21 triángulos) es casi cero, mientras su Ω_op (1143, ~52% de REAL) sigue siendo sustancial. Si η_LF
  alto fuera evidencia de "condensación exaptativa exitosa", el sistema que la teoría predeciría como
  el CAMPEÓN de esta tabla es un grafo Erdős-Rényi genuinamente independiente de REAL — que no comparte
  ni una arista, ni el grado, ni ninguna propiedad estructural con la malla causal real. Eso no encaja
  con la idea de O-N7.7 de que la historia "filtra configuraciones incompatibles y densifica los
  acoplamientos que quedan" — el grafo random nunca pasó por ningún proceso de filtrado histórico; su
  Ω_proc bajo es SIMPLE AUSENCIA de estructura, no el residuo denso de una selección.
- El problema es estructural del cociente: cuando el denominador (Ω_proc) se acerca a cero por CUALQUIER
  motivo —sea filtrado histórico genuino (NULL-3) o simple arbitrariedad sin historia (random)— el
  cociente se dispara igual. La razón simple Ω_op/Ω_proc no distingue "poco mecanismo porque la historia
  lo depuró" de "poco mecanismo porque nunca hubo relación con la malla real". Esa distinción es
  justamente el corazón conceptual de O-N7.7, y esta operacionalización no la captura.
- Con la Candidata B (diversidad entre semillas), el patrón es más plano y menos dramático (CV de REAL,
  NULL-3 y random están todos en el rango 2.4%-4.4%, sin separación clara), pero sigue heredando el mismo
  problema del denominador chico para random en cuanto se divide por Ω_proc.

**Analogía simple:** imaginá tres artesanos construyendo la misma mesa. El primero (REAL) usa una caja de
herramientas completa. El segundo (NULL-3) usa una caja bastante más chica —le sacaron un cuarto de las
herramientas, pero justo las que le sacaron eran las que menos usaba— y termina una mesa casi idéntica:
eso SÍ se parece a "logró más con menos", la idea central de O-N7.7. El tercero (grafo-random+layout) no
tiene caja de herramientas real: le dieron un montón de piezas sueltas que no tienen nada que ver con el
plano de la mesa, prácticamente al azar. Aun así, a fuerza de sacudir las piezas con el mismo proceso
físico (el "layout de resortes"), termina construyendo como la mitad de una mesa reconocible. Si uno mide
"mesa construida ÷ herramientas usadas", este tercer artesano da el número más alto de los tres —no
porque tuviera una caja de herramientas densa y bien curada, sino porque casi no tenía caja de
herramientas para empezar. El cociente no sabe distinguir entre "pocas herramientas, muy bien elegidas
por experiencia" y "casi ninguna herramienta, y aun así algo de suerte estructural". Esa es exactamente
la distinción que O-N7.7 necesita para significar algo, y con esta primera operacionalización el número
no la hace por sí solo.

**Resumen para Alexis:** el patrón NO es un "sí" limpio ni un "no" limpio. Confirma con fuerza el extremo
NULL-1/NULL-2 (indefinido = sin proceso relacional, tal como predice la teoría) y confirma de forma leve
pero real el contraste REAL vs NULL-3 (NULL-3 con η ligeramente mayor pese a menos motivos). Pero el
grafo-random+layout — que la jerarquía ya había mostrado como un punto intermedio en masa (52% de REAL,
ver `TEST_layout_vs_identidad_grafo_CS.md`) — se convierte, bajo esta métrica de η_LF, en el sistema con
la MAYOR densidad de capacidad por mecanismo de los 5, lo cual va en contra de la lectura simple de que
η_LF alto = condensación exaptativa exitosa. Esto no refuta la teoría (el random+layout nunca reclamó ser
un caso de "condensación exaptativa" — es un control de identidad-de-grafo, no de historia), pero sí
muestra que **esta primera operacionalización de η_LF, tal como está definida (razón simple Ω_op/Ω_proc),
no alcanza por sí sola para distinguir "mecanismo reducido por historia" de "mecanismo reducido por
ausencia de relación"** — haría falta un término adicional en la definición (algo que capture si el
Ω_proc remanente está relacionado con REAL o no) para que η_LF discrimine lo que O-N7.7 quiere que
discrimine.

---

## 6. Qué no se hizo (fuera de alcance de esta tarea, presupuesto de tiempo)

- No se recalcularon triángulos por semilla individual (sólo 1 semilla representativa por sistema) —
  limitación ya declarada en la sección 1.
- No se probó una tercera definición de Ω_proc que combine motivos CON un término de "relación con REAL"
  (ej. fracción de aristas compartidas con REAL, ya medida para el grafo-random en
  `TEST_layout_vs_identidad_grafo_CS.md`: 0.243% de solapamiento) — sería el candidato más directo para
  resolver la complicación de la sección 5, pero implica una definición nueva no pedida explícitamente en
  el encargo y no se construyó aquí para no forzar el resultado hacia lo que "se esperaba".
- No se tocó ningún archivo/script congelado de la jerarquía — sólo lectura de los `.md` ya existentes y
  cálculo de razones sobre los números ya reportados ahí.
