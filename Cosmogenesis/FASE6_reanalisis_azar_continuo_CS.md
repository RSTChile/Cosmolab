# Fase VI — reanálisis sobre los 80 datos ya existentes de Fase V-B: control de parejas al azar + observable continuo

**Fecha:** 11-ago-2026 · Ejecuta: CC (Claude) · Reanálisis puro sobre `cs090_fase5b_TOTAL_40pares.csv`
(el consolidado de 80 filas de `FASE5B_escala_40pares_CS.md`). **No se corrió Phantom.** No se modificó
ningún script ni CSV existente — sólo lectura + 2 scripts nuevos. No se hicieron commits de git. No se
declara cierre ni veredicto sobre A2-B0-C2: se reportan los números y dos lecturas honestas (a favor / en
contra) para cada análisis — la lectura final es de Alexis.

## 0. En simple, con analogía

Fase V-B encontró que, en 40 parejas de "maquetas de alambre" (misma malla, distinto cableado — Clase I
compacta vs Clase III extendida), la extendida acumula más arena bajo gravedad casi siempre (31 de 40
veces). Pero alguien podría objetar dos cosas razonables:

1. *"¿Y si separo las 80 maquetas al azar, ignorando cuál es cuál — sigue ganando 'la de la derecha' así
   de seguido?"* Si sí, el resultado no depende de ser Clase I o III, depende de otra cosa del conjunto de
   80. Esto es el **Análisis 1**.
2. *"¿Por qué usar una etiqueta binaria (I o III) en vez del número real de 'qué tan extendida' es cada
   maqueta?"* Si el número real predice la arena acumulada de forma pareja y sin saltos, la etiqueta I/III
   deja de ser necesaria — la geometría por sí sola alcanza. Esto es el **Análisis 2**.

Ambos análisis usan **los mismos 80 datos ya generados**, sin fabricar ni correr nada nuevo — es
reordenar y recombinar lo que ya existe, con mucho cuidado en no mezclar mal las columnas (ver más abajo,
el "bug de colisión de nombres" documentado en tareas anteriores).

## 1. Verificación previa de los datos base

`cs090_fase5b_TOTAL_40pares.csv` tiene exactamente **80 filas: 40 con `clase=I` y 40 con `clase=III`**,
agrupadas en 40 pares únicos (columna `par`), con la columna `clase` coincidiendo en las 80 filas con la
columna `rol` (redundante mismo dato, verificado igual). No hay mezcla rara. Verificado por assert en
`cs090_fase6_control_azar.py` antes de cualquier cálculo.

Nota aparte (no afecta la validez, sólo documentada): 4 `rule_id` están repetidos entre pares distintos
(`A2-B0-C2-r9`, `A2-B0-C2-r6`, `A2-B0-C2-r19`, más la reutilización de `r39` en dos pares) — son reglas
físicamente reutilizadas del piloto original en más de un par, tal como documenta `FASE5B_escala_40pares_CS.md`
§1. No son duplicados accidentales.

---

## 2. ANÁLISIS 1 — Control de parejas al azar

### 2.1 Metodología

Script: `cs090_fase6_control_azar.py`. Semilla `np.random.default_rng(2026)`, `N_PERM=10000`.

- **Referencia (real):** las 40 diferencias pareadas (fila `III` − fila `I`, usando el emparejamiento real
  de la columna `par`) en `fraccion_masa_en_sumideros` y `kappa_v_agregado` — esto reproduce el resultado
  ya reportado en Fase V-B (test de signos + Wilcoxon signed-rank, `scipy.stats`).
- **Null al azar:** en cada una de las 10 000 permutaciones se baraja el orden de las **80 filas
  completas** (ignorando la columna `clase`/`rol`), se forman 40 pares consecutivos, y la diferencia de
  cada par es (segunda fila del par) − (primera fila del par). Como el barajado es sobre las 80 filas
  completas, el orden dentro de cada par es arbitrario por construcción — un par al azar puede terminar
  siendo I-I, I-III, III-I o III-III, sin ningún sesgo hacia "la que era III menos la que era I". Se corre
  el mismo test de signos + Wilcoxon sobre esas 40 diferencias al azar, 10 000 veces.
- Se compara la referencia real contra la distribución de las 10 000 permutaciones: qué fracción de las
  permutaciones da un resultado tan o más extremo (en conteo de "victorias" ±20 de simetría, en p-valor de
  signos, y en p-valor de Wilcoxon).

### 2.2 Resultados

| métrica | real: victorias/40 | real p_signos | real W (Wilcoxon) | real p_Wilcoxon | azar: media victorias (±std) | percentil del real en la distr. azar |
|---|---|---|---|---|---|---|
| fracción de masa | 31/40 | 0.000680 | 81.50 | 0.000010 | 19.77 (±3.19) | **100.00** (el real supera o iguala el máximo de las 10 000 permutaciones) |
| κ_V agregado | 28/40 | 0.016589 | 195.00 | 0.003201 | 19.97 (±3.19) | **99.67** |

*(Nota: mi W de Wilcoxon en fracción de masa, 81.50, difiere levemente del 80.00 reportado en
`FASE5B_escala_40pares_CS.md` — diferencia de manejo de empates/versión de `scipy` entre corridas, el
p-valor resultante es esencialmente idéntico, 0.00001 en ambos casos. No cambia ninguna conclusión.)*

**Qué tan raro es el resultado real bajo la hipótesis "el emparejamiento no importa"** (fracción de las
10 000 permutaciones al azar que da un resultado tan o más extremo que el real):

| métrica | fracción de permutaciones tan extrema en victorias | fracción con p_signos tan bajo | fracción con p_Wilcoxon tan bajo |
|---|---|---|---|
| fracción de masa | **0.05%** (5 de 10 000) | 0.03% (3 de 10 000) | **0.00%** (0 de 10 000) |
| κ_V agregado | **1.83%** (183 de 10 000) | 1.79% (179 de 10 000) | 0.33% (33 de 10 000) |

CSV con los resúmenes: `cs090_fase6_control_azar_resumen.csv`. Distribuciones completas (10 000 filas
cada una): `cs090_fase6_azar_distribucion_fraccion_masa_en_sumideros.csv`,
`cs090_fase6_azar_distribucion_kappa_v_agregado.csv`.

### 2.3 Lectura honesta (sin forzar)

**A favor de que el efecto depende de la clase real, no sólo del pool de 80 grafos:** en **fracción de
masa**, el resultado real cae fuera del rango que produjeron 10 000 re-emparejamientos al azar del mismo
pool (ninguna o casi ninguna permutación alcanza 31/40 victorias tan desbalanceadas) — es un outlier claro,
mucho más extremo que el 1% típico que se pediría para hablar de "atípico". En **κ_V** el resultado real
también cae en la cola (1.8%-3.3% de las permutaciones son tan extremas), pero de forma menos contundente
que en fracción de masa — sigue siendo poco común bajo azar puro, pero no tan aplastantemente raro.

**En contra / matiz:** el control de azar no descarta que el pool de 80 grafos comparta alguna propiedad
común no relacionada con la teoría (ej. algo correlacionado con el orden de generación) que el
emparejamiento real aproveche mejor que el azar — sólo dice que **la distinción I/III específica, tal como
está codificada en el emparejamiento real, importa más que un emparejamiento arbitrario**. Tampoco dice
nada sobre causalidad. Y el efecto en κ_V, aunque también atípico, es visiblemente más débil que en
fracción de masa bajo esta prueba — consistente con lo que ya reportaba Fase V-B ("κ_V similar pero menos
fuerte").

---

## 3. ANÁLISIS 2 — Observable continuo, sin clases

### 3.1 El join — verificación cuidadosa (la parte crítica)

Script: `cs090_fase6_observable_continuo.py`. Las 80 filas de `cs090_fase5b_TOTAL_40pares.csv` no traen la
pendiente continua — hay que traerla desde los 4 CSV de origen donde cada regla fue clasificada:

- `cs090_fase5_profundizar_a2b0c2_resumen.csv` (sólo la sección `origen=='nueva_profundizar'`, que sí trae
  `seed`; la sección `original_barrido180` no tiene `seed` y se excluyó del todo del join, no se usó)
- `cs090_fase5b_candidatas_v2.csv`
- `cs090_fase5b_candidatas_v3.csv`
- `cs090_fase5b_candidatas_v4.csv`

**Por qué el join se hizo por `seed` y no por `rule_id`:** esta línea de trabajo tuvo un bug real,
documentado en `FASE5B_investigacion_8sumideros_y_escala_CS.md` §2.1 — dos reglas físicamente distintas
llegaron a compartir el mismo `rule_id` (`A2-B0-C2-r2`, `r9`, `r12`) en un momento del proceso, resuelto
después con los sufijos `v1fix`/`v2fix`. El `rule_id` es justo lo que colisionó; el `seed` nunca colisionó.
Se combinaron los 4 CSV de origen (430 filas en total) y se verificó **primero, antes de confiar en nada**,
que los 430 `seed` son únicos — cero colisiones, confirmado por assert en el script.

**Verificación por fila:** para cada una de las 80 filas de `TOTAL_40pares.csv` se buscó su `seed` en la
tabla combinada de 430, y se exigió que **K, kcap y clase coincidieran exactamente** entre lo que ya decía
`TOTAL_40pares.csv` y lo que decía el CSV de origen encontrado por `seed`. Sólo si las tres coincidían se
aceptó la `pendiente` de esa fila; si algo no coincidía, la fila se iba a "inválida" con la razón exacta
documentada, sin adivinar nada.

**Resultado del join: 80/80 filas válidas (100%), cero exclusiones.** Las 80 pendientes se pudieron atar
de forma verificada. CSV completo con el join (incluye columna `fuente` = de qué CSV vino cada pendiente,
y `razon` = "OK" en las 80): `cs090_fase6_pendientes_unidas.csv`.

### 3.2 Cobertura del rango de pendientes (nota metodológica honesta)

- Rango completo: pendiente entre **−0.9424** y **+1.0889**.
- **25 de 80 puntos (31%) caen dentro de ±0.05 del umbral de clasificación (0.7)** — a diferencia de lo
  que se anticipaba como posible ("puede haber pocos o ningún grafo con pendiente intermedia"), **la zona
  cercana al umbral está densamente cubierta**: Clase I llega hasta 0.698 y Clase III empieza en 0.706,
  hay muchísimos puntos apretados justo alrededor del corte.
- Sin embargo, hay **3 puntos aislados con pendiente muy negativa** (−0.94, −0.81, −0.65, los tres
  Clase I) separados del resto del conjunto por el hueco más grande de todo el rango (1.13 de ancho, entre
  −0.65 y +0.48 — el siguiente punto más cercano). Estos 3 son reglas reales, verificadas contra su CSV de
  origen sin ninguna duda (`A2-B0-C2-batch3-r100`, `A2-B0-C2-batch4-r51`, `A2-B0-C2-batch3-r143`), no son
  errores de join.

### 3.3 Correlación de Spearman y regresión

Sobre las **80 filas individuales** (no diferencias de pares):

| métrica | n | Spearman rho | Spearman p | R² lineal | R² cuadrático | mejora cuadrático vs lineal |
|---|---|---|---|---|---|---|
| pendiente vs fracción de masa | 80 | **0.4550** | 0.000022 | 0.0107 | **0.6567** | +0.6460 |
| pendiente vs κ_V agregado | 80 | **0.3687** | 0.000766 | 0.0293 | **0.6222** | +0.5929 |

El salto tan grande de R² lineal a cuadrático **no es casualidad de sobreajuste**: viene específicamente
de los 3 puntos con pendiente muy negativa, que tienen **fracción de masa alta** (0.10–0.15, similar a las
Clase III más extremas) en vez de baja — rompen la monotonía si se los incluye linealmente, generando una
forma en "U" (visible en el gráfico).

**Repitiendo el mismo cálculo excluyendo sólo esos 3 puntos** (n=77, el resto del conjunto):

| métrica | n | Spearman rho | Spearman p | R² lineal |
|---|---|---|---|---|
| pendiente vs fracción de masa | 77 | **0.5825** | 2.7×10⁻⁸ | 0.5627 |
| pendiente vs κ_V agregado | 77 | **0.4553** | 3.2×10⁻⁵ | — |

Sin esos 3 puntos, la relación es notablemente más monótona y más lineal (R² pasa de 0.01 a 0.56 sólo con
un ajuste lineal simple). CSV con el resumen de ambos ajustes (con los 80): `cs090_fase6_observable_continuo_resumen.csv`.

### 3.4 Gráfico

`cs090_fase6_observable_continuo.png` — pendiente (eje X) vs fracción de masa en Phantom (eje Y),
coloreado por clase original (azul=I, naranja=III), con la curva LOWESS (implementación manual liviana,
`statsmodels` no está instalado en este entorno; span=0.6) y una línea vertical en el umbral 0.7.

**Lo que se ve:** en el grueso del rango (pendiente entre ~0.48 y ~1.09, donde vive el 96% de los datos),
la nube sube de forma razonablemente pareja y **sin ningún salto discreto justo en el umbral 0.7** — la
transición de Clase I a Clase III alrededor de 0.7 se ve como una continuación de la misma tendencia, no
como un escalón. Los 3 puntos de pendiente muy negativa aparecen aislados a la izquierda, con valores de
fracción de masa altos que "traicionan" la tendencia general y jalan la curva LOWESS hacia una forma de
U poco intuitiva en ese extremo.

### 3.5 Lectura honesta (sin forzar)

**A favor de que la geometría continua predice la respuesta gravitacional:** hay una correlación de
Spearman positiva, estadísticamente significativa y nada débil (rho≈0.37–0.46 con las 80, p<0.001 en
ambas métricas; rho≈0.46–0.58 excluyendo los 3 outliers, p<10⁻⁴). La zona alrededor del umbral de
clasificación está bien cubierta por datos reales (no es un vacío), y en esa zona **no hay salto visible**
— la transición I→III se ve como parte de una tendencia continua, no como una frontera dura. Esto es
compatible con la idea de que el número continuo (no la etiqueta binaria) es lo que realmente importa.

**En contra / matiz:** la relación **no es monótona en todo el rango cubierto** — los 3 puntos de
pendiente muy negativa (todos Clase I) tienen fracción de masa alta, casi al nivel de las Clase III más
extremas, lo que rompe una historia simple de "a mayor pendiente, siempre mayor fracción". Esto podría
significar (a) que esas 3 reglas son un régimen físico distinto donde el mecanismo cambia de signo, (b)
que hay un confound no controlado que covaría con pendientes muy negativas, o (c) simplemente ruido con
n=3 tan chico en esa zona — con sólo 3 puntos ahí, no se puede distinguir entre estas lecturas con este
diseño. El Spearman con las 80 filas completas (0.455 y 0.369) es *moderado*, no *fuerte* — y baja
justamente por incluir esos 3 puntos. La correlación no implica causalidad, y ambas correlaciones (fracción
y κ_V) están sobre las mismas 80 corridas de Phantom que ya se usaron para Fase V-B — no son una
validación independiente con datos nuevos.

---

## 4. Síntesis final (ambos análisis, sin cerrar nada)

| pregunta | resultado | lectura |
|---|---|---|
| ¿El efecto Clase III>I sobrevive control de azar? | Sí en fracción de masa (outlier extremo, ≤0.05% de permutaciones tan extremas); sí mediano en κ_V (~1.8-3.3%) | El efecto depende de la clase real, más fuerte en fracción de masa que en κ_V — coherente con lo ya visto en Fase V-B |
| ¿La pendiente continua predice sin necesidad de clases? | Spearman moderado y significativo (rho 0.37-0.46, p<0.001) sobre las 80; más fuerte (rho 0.46-0.58) excluyendo 3 outliers; sin salto visible en el umbral 0.7 | Compatible con predicción continua en el grueso del rango, pero no monótona en el extremo de pendientes muy negativas — la etiqueta binaria I/III no queda invalidada, pero tampoco es estrictamente necesaria en la zona bien cubierta |

Ninguno de los dos análisis "cierra" nada sobre A2-B0-C2: ambos son reanálisis de los mismos 80 datos ya
generados en Fase V-B, con las limitaciones explícitas de diseño ya conocidas (mismo N=2000, mismo
protocolo de Phantom, mismo pool de reglas). Generar grafos deliberadamente "de pendiente intermedia" o
"de pendiente muy negativa" (para entender esos 3 outliers) sería un experimento nuevo, fuera del alcance
de esta tarea de puro reanálisis. La interpretación final de qué tan lejos llega esta evidencia es de
Alexis.

## 5. Archivos de esta tarea

- `cs090_fase6_control_azar.py` — Análisis 1: reproduce el resultado real (test de signos + Wilcoxon sobre
  las 40 diferencias pareadas reales) y genera 10 000 re-emparejamientos al azar del pool de 80 filas
  (semilla `np.random.default_rng(2026)`), comparando el resultado real contra esa distribución nula.
- `cs090_fase6_control_azar_resumen.csv` — resumen de ambas métricas (real vs distribución de azar).
- `cs090_fase6_azar_distribucion_fraccion_masa_en_sumideros.csv`,
  `cs090_fase6_azar_distribucion_kappa_v_agregado.csv` — las 10 000 permutaciones completas de cada
  métrica (n_pos, p_signos, W, p_wilcoxon por permutación).
- `cs090_fase6_observable_continuo.py` — Análisis 2: construye la tabla maestra de pendientes desde los 4
  CSV de origen (verificando cero colisiones de `seed`), hace el join verificado por `seed`+K+kcap+clase
  contra las 80 filas de `TOTAL_40pares.csv`, calcula Spearman + regresión lineal/cuadrática + LOWESS
  manual, y genera el gráfico.
- `cs090_fase6_pendientes_unidas.csv` — las 80 filas con su `pendiente` unida y verificada (columnas
  `valido`, `razon`, `fuente` documentando el resultado del join fila por fila).
- `cs090_fase6_observable_continuo_resumen.csv` — Spearman + coeficientes de regresión lineal/cuadrática
  para ambas métricas.
- `cs090_fase6_observable_continuo.png` — gráfico pendiente vs fracción de masa, coloreado por clase, con
  LOWESS y línea de umbral.
- Este informe.

No se modificó ningún script ni CSV existente. No se corrió Phantom. No se declaró cierre ni veredicto
sobre A2-B0-C2. No se hicieron commits de git.
