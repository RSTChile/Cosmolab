# O2-B — Genealogías independientes escaladas: de 4 a 20 familias de A2-B0-C2

**Fecha:** 11-ago-2026 · Ejecuta: CC (Claude) · Tarea **O2-B** del `FASE6_PLAN_EJECUCION_COMPLETA_CS.md`
(pedida por los tres analistas: GPT-5.6 Sol F6-05 pedía 10-12 genealogías, el segundo analista 20).
Antecedente directo: `FASE5_genealogias_independientes_CS.md` (4 genealogías, declarado allí mismo como
baja potencia). Medición de diámetro: **la corregida** (`cs090_diam_corregido.diam_gigante`), según
`FASE6_adopcion_diam_corregido_CS.md`.

**No se modificó ningún script existente ni congelado.** **No se corrió Phantom.** **No se hicieron commits
de git.** **No se declara cierre ni veredicto** — se reportan números; la lectura final es de Alexis.

---

## 0. En simple, con analogía

Una **genealogía** es una *red social entera y distinta*: un punto de partida propio del generador
(`seed_base`) del que salen 20 familias de parámetros (K, J, ruido, grado medio, tope de amigos `kcap`,
semilla) creadas desde cero. Una **realización** es un *día distinto de la misma red social*: una de esas
20 semillas.

La tarea anterior miró **4 redes sociales** y vio el patrón bimodal en las 4 (45-75 % de "días extendidos",
Clase III). Pero con 4 puntos no se podía distinguir dos historias muy diferentes:

- **(a)** *"el mecanismo produce ~50 % de días extendidos en CUALQUIER red social"*, o
- **(b)** *"hay 2-3 redes sociales especialmente fértiles que sostienen el promedio, y muchas que no dan
  nada"* — la pregunta textual de GPT-5.6 Sol.

Esta tarea corrió **20 redes sociales nuevas** (400 corridas de motor por brazo) para poder separarlas.

**El resumen en tres frases.** Ninguna de las 20 familias es estéril: la peor da 20 % y la mejor 65 %, y el
promedio (44 %) queda muy cerca del 45 % histórico — o sea, la historia (b) en su forma fuerte queda
descartada. Al mismo tiempo, con 20 familias sí aparece una pizca de "efecto de familia" que con 4 no se
veía (permutación p = 0,040), pero es **pequeña**: equivale a decir que 400 reglas valen unas 245
independientes. Y cuando se mira por qué unas familias rinden más, la respuesta no es que la familia tenga
algo especial: es **cuántas reglas de `kcap` bajo le tocaron en el sorteo** — `kcap` explica el 62 % de la
variación, la genealogía apenas el 8 %.

---

## 1. Las 20 genealogías y por qué se consideran independientes

| # | etiqueta | seed_base | | # | etiqueta | seed_base |
|---|---|---|---|---|---|---|
| 0 | H00_113477 | 113 477 | | 10 | H10_2043761 | 2 043 761 |
| 1 | H01_218903 | 218 903 | | 11 | H11_2296589 | 2 296 589 |
| 2 | H02_344251 | 344 251 | | 12 | H12_2571043 | 2 571 043 |
| 3 | H03_662819 | 662 819 | | 13 | H13_2814697 | 2 814 697 |
| 4 | H04_741037 | 741 037 | | 14 | H14_3102859 | 3 102 859 |
| 5 | H05_905683 | 905 683 | | 15 | H15_3389417 | 3 389 417 |
| 6 | H06_1128409 | 1 128 409 | | 16 | H16_3670213 | 3 670 213 |
| 7 | H07_1357061 | 1 357 061 | | 17 | H17_3948071 | 3 948 071 |
| 8 | H08_1604923 | 1 604 923 | | 18 | H18_4213589 | 4 213 589 |
| 9 | H09_1889347 | 1 889 347 | | 19 | H19_4507921 | 4 507 921 |

**Ninguna repite** las 4 ya usadas (90 210 · 471 829 · 823 001 · 156 644) ni los `seed_base` de otras tareas
del proyecto (271 828 · 371 828 · 471 828 · 571 828). Verificado por `grep` sobre `cs090*.py`.

**La independencia no se afirma, se garantiza aritméticamente y se comprueba en tiempo de ejecución.**
`generar_reglas_clase` deriva cada regla con `seed = seed_base + intento*97 + 1`; con `max_intentos = 80`,
la cadena de semillas individuales de una genealogía ocupa como mucho `[seed_base+1, seed_base+7761]`. Las
20 semillas base están separadas **entre sí por ≥ 78 218** y de **cada** semilla ya usada en el proyecto
por **≥ 23 267** — más de 3 y de 2,9 veces el ancho de cadena respectivamente. Conclusión: **ninguna
genealogía comparte ni una sola semilla individual con otra ni con las históricas.** La función
`_verificar_separacion()` del script lo comprueba y aborta si falla; la corrida real imprimió:

```
[verificación] 20 semillas nuevas; ancho de cadena por genealogía = 7761;
               separación mínima nueva-nueva = 78218; separación mínima nueva-vs-ya_usada = 23267
[verificación] saltos consecutivos entre semillas nuevas, todos distintos = True -> no hay progresión aritmética
[verificación] OK: ninguna genealogía puede compartir una semilla individual con otra.
```

Además: los saltos entre semillas consecutivas de la lista son **todos distintos** (no es una progresión
aritmética ni hay múltiplos unos de otros), y las magnitudes abarcan de 10⁵ a 4,5×10⁶.

**Salvedad honesta:** PCG64 (`np.random.default_rng`) no tiene una estructura conocida por la que semillas
numéricamente cercanas den secuencias correlacionadas, así que la separación numérica es una precaución
*adicional*, no el argumento principal. Lo que sostiene la independencia es la no-superposición de cadenas.

---

## 2. Qué se corrió, y con qué vara se midió

| ítem | valor |
|---|---|
| genealogías | **20** (todas las planificadas; ninguna recortada por tiempo) |
| reglas por genealogía | 20, admitidas por el **filtro P1-P5 real** del generador congelado |
| filtro P1-P5 | **20/20 admitidas en las 20 genealogías, 0 descartes** — el filtro no es ni más permisivo ni más estricto con ninguna semilla base |
| brazos | **C2-hard** (`MOT.correr_regla_coarse`) **y C2-hibrido** (`MA.correr_regla_coarse_hibrido(modo="soporte")`) — alcanzó el presupuesto para los dos |
| corridas de motor | **800** (20 × 20 × 2) |
| tamaño / escalas | N = 2000, 14 barridos, coarse-graining b = 1,2,4,8,16, 3 semillas de NULL_topo |
| clasificación | `cs090_fase5_clasificador.clasificar_regla`, **umbrales sin tocar** |
| costo | **8,7 min** de reloj (5 shards en paralelo) · 42,8 min de CPU acumulados |

### 2.1 La medición de diámetro es la corregida — y acá se nota

Regla vigente (`FASE6_adopcion_diam_corregido_CS.md`): todo cálculo nuevo usa `diam_gigante` (doble-BFS
arrancando en la componente conexa **más grande**), no el `_diam` de cs055, que arranca en el nodo no
aislado de índice más bajo del grafo entero y, si ése cayó en un fragmento suelto, **mide el fragmento** —
*el metro apoyado en el buzón de la vereda en vez del edificio*.

Cómo se aplicó sin tocar archivos: los motores llaman siempre a `cs090_fase5_motor._diam(...)`, que es una
búsqueda de atributo resuelta en el momento de la llamada; se **sustituyó ese atributo en memoria**
(`MOT._diam = DC.diam_gigante`) al arrancar cada proceso, el mismo mecanismo ya usado y verificado por
`cs090_fase6_remedir_mecanismo.py`. **Ningún archivo cambió en disco.**

Tres marcas de que la vara corregida funcionó, en las 800 corridas:

| control | resultado |
|---|---|
| reglas con pendiente **negativa** (geométricamente imposible: agrupar cajas conexas no puede alargar un camino) | **0 de 800** |
| reglas en la casilla **"intermedio (sin clase clara)"** | **0 de 800** (con la vara vieja, la tarea de 4 genealogías tuvo 4 de 160) |
| diámetro mínimo a b=1 | **6** (las 15 descarriladas del lote de 430 tenían ≤ 3; las sanas ≥ 8) |

---

## 3. Resultado — %Clase III por genealogía (tabla completa)

### 3.1 Brazo C2-hard (el mejor caracterizado)

| genealogía | seed_base | n | I | II | **III** | IV | otro | **%Clase III** | SE binom. (n=20) | pend. media | pend. mediana | diám. medio b=1 | grado medio b=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H05_905683 | 905 683 | 20 | 6 | 1 | **13** | 0 | 0 | **65,0 %** | 10,67 | 0,775 | 0,769 | 15,10 | 3,34 |
| H07_1357061 | 1 357 061 | 20 | 6 | 1 | **13** | 0 | 0 | **65,0 %** | 10,67 | 0,790 | 0,768 | 14,75 | 3,36 |
| H15_3389417 | 3 389 417 | 20 | 5 | 1 | **13** | 1 | 0 | **65,0 %** | 10,67 | 0,767 | 0,724 | 14,20 | 3,26 |
| H03_662819 | 662 819 | 20 | 7 | 1 | **12** | 0 | 0 | **60,0 %** | 10,95 | 0,756 | 0,735 | 14,35 | 3,40 |
| H06_1128409 | 1 128 409 | 20 | 7 | 1 | **12** | 0 | 0 | **60,0 %** | 10,95 | 0,765 | 0,771 | 14,40 | 3,49 |
| H01_218903 | 218 903 | 20 | 9 | 0 | **10** | 1 | 0 | **50,0 %** | 11,18 | 0,765 | 0,707 | 13,90 | 3,42 |
| H00_113477 | 113 477 | 20 | 10 | 1 | **9** | 0 | 0 | **45,0 %** | 11,12 | 0,719 | 0,640 | 13,55 | 3,54 |
| H08_1604923 | 1 604 923 | 20 | 9 | 2 | **9** | 0 | 0 | **45,0 %** | 11,12 | 0,704 | 0,620 | 13,50 | 3,58 |
| H12_2571043 | 2 571 043 | 20 | 10 | 1 | **9** | 0 | 0 | **45,0 %** | 11,12 | 0,743 | 0,662 | 14,00 | 3,48 |
| H13_2814697 | 2 814 697 | 20 | 9 | 2 | **9** | 0 | 0 | **45,0 %** | 11,12 | 0,751 | 0,695 | 15,00 | 3,40 |
| H14_3102859 | 3 102 859 | 20 | 9 | 2 | **9** | 0 | 0 | **45,0 %** | 11,12 | 0,702 | 0,656 | 13,80 | 3,61 |
| H19_4507921 | 4 507 921 | 20 | 9 | 1 | **9** | 1 | 0 | **45,0 %** | 11,12 | 0,755 | 0,719 | 14,80 | 3,41 |
| H02_344251 | 344 251 | 20 | 12 | 0 | **8** | 0 | 0 | **40,0 %** | 10,95 | 0,652 | 0,631 | 12,75 | 3,73 |
| H09_1889347 | 1 889 347 | 20 | 11 | 1 | **8** | 0 | 0 | **40,0 %** | 10,95 | 0,690 | 0,671 | 13,75 | 3,67 |
| H10_2043761 | 2 043 761 | 20 | 10 | 1 | **8** | 1 | 0 | **40,0 %** | 10,95 | 0,703 | 0,691 | 13,70 | 3,62 |
| H16_3670213 | 3 670 213 | 20 | 12 | 2 | **6** | 0 | 0 | **30,0 %** | 10,25 | 0,646 | 0,626 | 12,35 | 3,89 |
| H17_3948071 | 3 948 071 | 20 | 13 | 1 | **6** | 0 | 0 | **30,0 %** | 10,25 | 0,655 | 0,630 | 12,70 | 3,73 |
| H04_741037 | 741 037 | 20 | 13 | 2 | **5** | 0 | 0 | **25,0 %** | 9,68 | 0,648 | 0,601 | 13,10 | 3,77 |
| H11_2296589 | 2 296 589 | 20 | 13 | 1 | **4** | 2 | 0 | **20,0 %** | 8,94 | 0,635 | 0,617 | 12,70 | 3,69 |
| H18_4213589 | 4 213 589 | 20 | 14 | 1 | **4** | 1 | 0 | **20,0 %** | 8,94 | 0,659 | 0,658 | 13,05 | 3,63 |

**Totales C2-hard (400 reglas):** I = 194 · II = 23 · III = 176 · IV = 7 · intermedio = 0.

### 3.2 Brazo C2-hibrido

| genealogía | n | I | II | **III** | IV | **%Clase III** | pend. media | diám. medio b=1 | grado medio b=1 |
|---|---|---|---|---|---|---|---|---|---|
| H07_1357061 | 20 | 6 | 3 | **11** | 0 | **55,0 %** | 0,775 | 11,80 | 3,87 |
| H03_662819 | 20 | 8 | 2 | **10** | 0 | **50,0 %** | 0,737 | 12,05 | 4,07 |
| H05_905683 | 20 | 8 | 3 | **9** | 0 | **45,0 %** | 0,713 | 11,60 | 4,04 |
| H01_218903 | 20 | 10 | 2 | **8** | 0 | **40,0 %** | 0,736 | 11,75 | 3,93 |
| H12_2571043 | 20 | 11 | 2 | **7** | 0 | **35,0 %** | 0,687 | 11,30 | 4,23 |
| H14_3102859 | 20 | 9 | 4 | **7** | 0 | **35,0 %** | 0,605 | 10,60 | 4,48 |
| H15_3389417 | 20 | 12 | 1 | **7** | 0 | **35,0 %** | 0,731 | 11,70 | 3,96 |
| H06_1128409 | 20 | 9 | 5 | **6** | 0 | **30,0 %** | 0,642 | 10,75 | 4,31 |
| H08_1604923 | 20 | 9 | 5 | **6** | 0 | **30,0 %** | 0,668 | 11,25 | 4,40 |
| H09_1889347 | 20 | 10 | 4 | **6** | 0 | **30,0 %** | 0,583 | 10,45 | 4,49 |
| H19_4507921 | 20 | 12 | 2 | **6** | 0 | **30,0 %** | 0,680 | 11,40 | 4,12 |
| H00_113477 | 20 | 12 | 3 | **5** | 0 | **25,0 %** | 0,639 | 10,75 | 4,32 |
| H10_2043761 | 20 | 13 | 1 | **5** | 1 | **25,0 %** | 0,677 | 11,40 | 4,34 |
| H13_2814697 | 20 | 10 | 4 | **5** | 1 | **25,0 %** | 0,688 | 11,30 | 4,21 |
| H16_3670213 | 20 | 12 | 3 | **5** | 0 | **25,0 %** | 0,629 | 10,85 | 4,59 |
| H17_3948071 | 20 | 10 | 5 | **5** | 0 | **25,0 %** | 0,598 | 10,45 | 4,64 |
| H04_741037 | 20 | 11 | 5 | **4** | 0 | **20,0 %** | 0,590 | 10,60 | 4,55 |
| H18_4213589 | 20 | 11 | 5 | **4** | 0 | **20,0 %** | 0,604 | 10,70 | 4,47 |
| H02_344251 | 20 | 14 | 3 | **3** | 0 | **15,0 %** | 0,565 | 10,60 | 4,62 |
| H11_2296589 | 20 | 12 | 4 | **2** | 2 | **10,0 %** | 0,582 | 10,20 | 4,62 |

**Totales C2-hibrido (400 reglas):** I = 209 · II = 66 · III = 121 · IV = 4 · intermedio = 0.

**C2-hard queda por encima de C2-hibrido en 19 de las 20 genealogías**, y en la restante empatan
(H18_4213589, 20 % vs 20 %); **hibrido no supera a hard en ninguna**. Medias 44,0 % vs 30,3 %. El orden
relativo entre brazos que sostiene toda la línea F5-C2-C **se replica en las 20 familias nuevas**, ahora
con 20 réplicas independientes en vez de 4.

---

## 4. Varianza ENTRE genealogías vs. varianza DENTRO de cada genealogía

### 4.1 Descriptivo, y comparación contra el ruido de muestreo (mismo cálculo que la tarea anterior)

| brazo | n gen. | media %III | **std entre gen.** | **CV** | min | max | rango | SE binomial esperado dentro de UNA genealogía (n=20) | razón std/SE |
|---|---|---|---|---|---|---|---|---|---|
| **C2-hard** | 20 | **44,00 %** | **14,20 pp** | **32,3 %** | 20,0 % | 65,0 % | 45,0 pp | 10,64 pp (promedio; 11,10 pp con el p global) | **1,335** |
| **C2-hibrido** | 20 | **30,25 %** | **11,18 pp** | **37,0 %** | 10,0 % | 55,0 % | 45,0 pp | 9,92 pp (10,27 pp con el p global) | **1,127** |

*Cómo leer la última columna:* si todas las familias tuvieran exactamente la misma tasa verdadera, y la
única variación fuera el azar de sacar 20 reglas, la razón valdría **1,0**. C2-hard da **1,335** (un 34 %
más de dispersión que el puro muestreo); C2-hibrido da **1,127** (un 13 % más, prácticamente nada).

Para comparar con la tarea de 4: allí las razones eran **1,12** (hard: 11,92 / 10,66) y **1,02** (hibrido:
10,61 / 10,37). Con 20 familias la razón de hard **sube**, y con 5 veces más grados de libertad la
diferencia deja de ser indistinguible del ruido (§4.3).

### 4.2 ANOVA de una vía — descomposición formal

| brazo | variable | MSB (gl) | MSW (gl) | F | p (F) | comp. varianza ENTRE | comp. DENTRO | **ICC** |
|---|---|---|---|---|---|---|---|---|
| C2-hard | indicador 0/1 de Clase III | 0,40316 (19) | 0,23921 (380) | **1,685** | **≈ 0,036** | 0,00820 | 0,23921 | **0,0331** |
| C2-hard | pendiente (continua) | 0,05233 (19) | 0,04299 (380) | 1,217 | ≈ 0,240 | 0,00047 | 0,04299 | 0,0107 |
| C2-hibrido | indicador 0/1 de Clase III | 0,24987 (19) | 0,20961 (380) | 1,192 | ≈ 0,261 | 0,00201 | 0,20961 | 0,0095 |
| C2-hibrido | pendiente (continua) | 0,07638 (19) | 0,06833 (380) | 1,118 | ≈ 0,330 | 0,00040 | 0,06833 | 0,0059 |

ICC = (MSB − MSW) / (MSB + (m−1)·MSW) con m = 20. En simple: **la ICC es "cuánto se parecen entre sí dos
reglas de la misma familia, por el solo hecho de ser de la misma familia"**. 0 = nada; 1 = todas las reglas
de una familia son clones. Los valores acá van de **0,006 a 0,033**: hay algo, y es poco.

### 4.3 Test de permutación (sin supuestos de normalidad — con datos 0/1 y n=20 el F es aproximado)

Se barajan las 400 reglas entre las 20 familias conservando el tamaño de cada una, 20 000 veces, y se mira
si el desvío estándar de los promedios por familia observado es más grande que el barajado.

| brazo | variable | std observado | std esperado barajando (p95) | **p** | percentil del observado |
|---|---|---|---|---|---|
| **C2-hard** | indicador Clase III | 0,1384 | 0,1069 (0,1356) | **0,0401** | 96,0 |
| C2-hard | pendiente | 0,0499 | 0,0449 (0,0568) | 0,241 | 75,9 |
| C2-hibrido | indicador Clase III | 0,1089 | 0,0991 (0,1260) | 0,271 | 72,9 |
| C2-hibrido | pendiente | 0,0602 | 0,0565 (0,0716) | 0,336 | 66,4 |

**Lectura:** en C2-hard, la dispersión entre familias **sí excede** el ruido de muestreo — apenas, y sólo en
la variable categórica (p = 0,040, percentil 96). En la **pendiente continua**, que es la misma información
sin el corte binario, **no hay señal** (p = 0,24). Y en C2-hibrido no hay señal en ninguna de las dos. Es
decir: el "efecto de familia" que aparece es **frágil y sólo se ve al binarizar** — hay que tomarlo como una
pista, no como un hecho establecido.

### 4.4 N efectivo (lo que pidió GPT-5.6 Sol)

Efecto de diseño de un muestreo por conglomerados: **deff = 1 + (m−1)·ICC**, con m = 20 reglas por familia;
**N_eff = N_total / deff**.

| brazo | variable | ICC | deff | N_total | **N EFECTIVO** | equivale a |
|---|---|---|---|---|---|---|
| **C2-hard** | indicador Clase III | 0,0331 | 1,630 | 400 | **245,5** | 12,3 genealogías completas |
| C2-hard | pendiente | 0,0107 | 1,204 | 400 | 332,2 | 16,6 genealogías |
| C2-hibrido | indicador Clase III | 0,0095 | 1,181 | 400 | 338,8 | 16,9 genealogías |
| C2-hibrido | pendiente | 0,0059 | 1,111 | 400 | 360,0 | 18,0 genealogías |

**Incertidumbre del número (bootstrap sobre genealogías, 4000 remuestreos, C2-hard / indicador):**
ICC puntual 0,0331, **IC 95 % = [0,0000 · 0,0726]** → **N_eff IC 95 % = [168 · 400]**. O sea: el N efectivo
está entre el 42 % y el 100 % del nominal, con mejor estimación **~61 %**.

**Regla práctica que se desprende, para Fase V-B:** el caso conservador es **multiplicar el N nominal por
≈ 0,6** (o inflar los errores estándar por √1,63 ≈ 1,28) cuando las reglas provienen de pocas familias.
Aplicado a los 40 pares: si esos 80 reglas vinieran de un puñado de genealogías, el N efectivo sería del
orden de 24-25 pares en vez de 40. El p de Wilcoxon publicado (1,03×10⁻⁶ sobre los 37 pares válidos) tiene
holgura de sobra para absorber esa corrección — **pero el cálculo exacto exige saber de cuántas genealogías
salieron esos 80, que no es dato de esta tarea** (los batches 3/4 vienen de `seed_base` 471828 y 571828, o
sea de **dos** familias; con m grande y esta ICC el deff sería mucho peor que 1,63 — ver §7, caveats).

---

## 5. La pregunta central: ¿efecto repartido, o 2-3 familias fértiles?

| indicador | C2-hard | C2-hibrido |
|---|---|---|
| familias **estériles** (0 % Clase III) | **0 de 20** | **0 de 20** |
| familias con ≥ 30 % | **17 de 20** | 11 de 20 |
| familias con ≥ 45 % (el piso de las 4 anteriores) | **12 de 20** | 3 de 20 |
| las **3 familias más fértiles** aportan… | 39 de 176 Clase III = **22,2 %** (reparto uniforme daría 15,0 %) | 30 de 121 = 24,8 % |
| mediana / cuartiles | 45,0 % / 37,5 %-52,5 % | 30,0 % / 25 %-35 % |

**Respuesta:** el efecto está **repartido**, no sostenido por unas pocas familias. En C2-hard **ninguna de
las 20 familias es estéril**, 17 de 20 pasan el 30 % y la mediana cae exactamente en el 45 % histórico. Las
3 mejores concentran el 22 % de las Clase III cuando un reparto perfectamente uniforme daría 15 % — un
exceso de 7 puntos, muy lejos del escenario "dos o tres familias sostienen todo".

La analogía: no es que haya tres barrios que producen todos los edificios altos y el resto sea llano. Es
que **todos los barrios producen edificios altos**, unos un poco más que otros.

---

## 6. Por qué unas familias rinden más que otras: **es el sorteo de `kcap`, no la familia**

Al mirar qué distingue a las familias fértiles de las flacas aparece una respuesta muy limpia — y le quita
casi todo el misterio al "efecto de genealogía" de §4.3.

### 6.1 A nivel de REGLA, `kcap` decide casi todo (400 reglas, C2-hard)

| kcap (tope de vecinos) | n | **% Clase III** | pendiente media |
|---|---|---|---|
| **4** | 62 | **98,4 %** | +1,063 |
| **5** | 152 | **71,7 %** | +0,773 |
| **6** | 122 | **4,9 %** | +0,591 |
| **7** | 64 | **0,0 %** | +0,471 |

Es un escalón, no una pendiente suave: entre `kcap`=5 y `kcap`=6 la tasa se desploma de 71,7 % a 4,9 %.
(Este escalón ya estaba documentado como *"borde probable artefacto de umbral"* en el mapa kcap×K del
informe consolidado del equipo; acá se replica en 20 familias independientes.)

### 6.2 A nivel de GENEALOGÍA, la "fertilidad" sigue al `kcap` sorteado

Correlaciones de Spearman entre los 20 puntos (una por familia):

| relación | C2-hard | C2-hibrido |
|---|---|---|
| %Clase III vs **`kcap` medio del lote** | **−0,842** | −0,675 |
| %Clase III vs % de reglas con `kcap`=4 | +0,620 | +0,659 |
| %Clase III vs **grado medio realizado** (b=1) | **−0,893** | −0,818 |
| %Clase III vs diámetro medio (b=1) | +0,792 | +0,786 |
| %Clase III vs `meandeg` (parámetro) | +0,358 | +0,389 |
| %Clase III vs K medio | −0,114 | +0,092 |

El `kcap` medio de las 20 familias va de **5,10 a 5,90** — pura suerte del sorteo, ya que el generador
elige `kcap` por regla. Las tres familias más fértiles sacaron 6, 4 y 3 reglas con `kcap`=4; las dos más
flacas sacaron 0 y 3, y ambas cargaron 8 reglas con `kcap`=6.

### 6.3 Descomposición: `kcap` explica 62 %, la genealogía 8 %

| brazo | η² de `kcap` sobre el indicador Clase III | η² de la genealogía |
|---|---|---|
| C2-hard | **0,619** | 0,078 |
| C2-hibrido | 0,443 | 0,056 |

Y si se **descuenta el efecto de `kcap`** (se resta a cada regla la media de su propio `kcap`) y se repite
todo el análisis sobre los residuos:

| brazo | ICC residual | deff | **N_eff residual** | permutación (p) |
|---|---|---|---|---|
| C2-hard | 0,0226 (era 0,0331) | 1,430 | **279,7** de 400 (era 245,5) | **0,089** (era 0,040) |
| C2-hibrido | — | — | — | 0,245 (era 0,271) |

**El único p que estaba bajo 0,05 deja de estarlo** al descontar `kcap`. Dicho en simple: *lo que parecía
"esta familia es más fértil" es en buena parte "a esta familia le tocaron más reglas con el tope de amigos
bajo".* Queda un resto (ICC 0,023, N_eff 280) que no se explica por `kcap` — pequeño, y sin evidencia de
que exceda el azar.

**Consecuencia práctica, y no es menor:** si el objetivo es evitar pseudorreplicación, **la unidad de
agrupamiento que de verdad importa no es la genealogía sino `kcap`**. Dos reglas de familias distintas con
`kcap`=4 se parecen muchísimo más entre sí que dos reglas de la misma familia con `kcap` 4 y 7.

---

## 7. Comparación con las 4 genealogías anteriores, y caveats honestos

### 7.1 Las 4 anteriores estaban en el lado fértil

Re-medidas con el diámetro corregido (de `cs090_fase6_remedicion_mecanismo.csv`, columna `clase_corregida`):

| conjunto | n gen. | %Clase III (C2-hard) | media | std |
|---|---|---|---|---|
| **4 anteriores** (G0/G1/G2/G3) | 4 | 55 · 45 · 70 · 75 | **61,25 %** | 13,77 pp |
| **20 nuevas** (esta tarea) | 20 | 20 … 65 (ver §3.1) | **44,00 %** | 14,20 pp |
| las 24 juntas | 24 | 20 … 75 | 46,88 % | 15,31 pp |

Las 4 originales promedian **17 pp por encima** de las 20 nuevas. Con un desvío entre familias de ~14 pp, el
error estándar de una media de 4 es ~7 pp, así que 17 pp son ≈ 2,4 errores estándar: **es una diferencia
grande pero no imposible por azar de haber sorteado 4 familias**. Lo honesto es decir que **el lote original
de 4 cayó del lado alto de la distribución**, y que el 58,75 %/61,25 % que reportó esa tarea es, con estos
20 puntos nuevos, un **valor optimista** respecto del centro real (~44-47 %). Nótese que la mediana de las
20 nuevas es exactamente **45,0 %** — el número histórico de la línea F5-C2-C.

### 7.2 Caveats

- **El único p significativo es frágil.** C2-hard / indicador da p = 0,040 en permutación; el mismo dato en
  su forma continua (pendiente) da p = 0,24, y al descontar `kcap` sube a 0,089. Con 4 tests de permutación
  reportados, un 0,040 no sobrevive a ninguna corrección por comparaciones múltiples. **No se afirma que
  exista un efecto de genealogía**; se reporta que aparece una señal débil donde con n=4 no había ninguna.
- **La ICC tiene un IC 95 % que incluye el 0** ([0,000 · 0,073]). El N_eff de 245 es la mejor estimación
  puntual, no un número firme: el rango honesto es 168-400.
- **`m` = 20 amplifica cualquier ICC.** El deff crece con el tamaño del conglomerado: con ICC = 0,033 y
  m = 20 da 1,63, pero con m = 40 daría 2,3. Aplicar el N_eff de esta tarea a otro diseño exige recalcular
  con el `m` de ese diseño.
- **Sobre los 40 pares de Fase V-B:** esta tarea da la herramienta (ICC y deff), no la respuesta. Las 80
  reglas de los 40 pares salen de `seed_base` 471828 y 571828 — es decir, de **2 familias** con ~40 reglas
  cada una. Con ICC = 0,033 y m ≈ 40, deff ≈ 2,3 y el N efectivo sería ≈ 17 pares de 40. **Ese cálculo no se
  hizo acá sobre los datos reales de los 40 pares** (haría falta recuperar la genealogía de cada regla y
  recalcular la ICC en ese conjunto); queda planteado como el siguiente paso natural, y es exactamente el
  cálculo que permitiría cerrar la amenaza de pseudorreplicación en vez de acotarla.
- **Sólo se probó A2-B0-C2.** Nada de esto dice si el patrón entre-genealogías es igual en otros combos.
- **El corte binario Clase III es lo que crea buena parte de la señal.** El clasificador convierte una
  pendiente continua en un sí/no con umbral 0,7; muchas reglas caen cerca del umbral y un empujón mínimo las
  cambia de casilla. Que la señal aparezca en la versión binaria y no en la continua es coherente con eso, y
  es un motivo más para no sobreinterpretar el p = 0,040.
- **No se corrieron los brazos de control** (C0, budget, random, presupuesto-variable) en estas 20 familias:
  fuera del alcance de O2-B, que apuntaba a grados de libertad entre-grupo en el brazo mejor caracterizado.

---

## 8. Archivos de esta tarea

**Script nuevo** (no modifica nada existente):

- `cs090_fase6_o2b_genealogias_escaladas.py` — genera y corre las 20 genealogías (con sharding para
  paralelizar sin cambiar un solo número, porque el motor es determinista por `seed`), aplica la medición
  de diámetro corregida sustituyendo `MOT._diam` en memoria, clasifica con los umbrales congelados, y hace
  todo el análisis de varianza (ANOVA, ICC, N efectivo, permutación) y las figuras.
  Uso: `--modo correr --shard k --nshards M --brazos C2-hard,C2-hibrido` · `--modo analisis`.

**Datos:**

- `cs090_fase6_o2b_genealogias_reglas_TODAS.csv` — **800 filas**, una por genealogía × regla × brazo: clase,
  pendiente, z, holonomía, diámetros por escala, grado medio, y los parámetros completos (K, J, noise,
  meandeg, kcap, seed).
- `cs090_fase6_o2b_genealogias_reglas_shard0..4.csv` — los 5 shards crudos (160 filas cada uno).
- `cs090_fase6_o2b_genealogias_por_genealogia_C2-hard.csv` y `..._C2-hibrido.csv` — 20 filas cada uno, el
  agregado por familia (tablas de §3).

**Figuras:**

- `cs090_fase6_o2b_genealogias_distribucion.png` — %Clase III por familia (barras ordenadas) + la nube de
  pendientes individuales por familia, para los dos brazos.
- `cs090_fase6_o2b_genealogias_kcap.png` — el mecanismo de §6: %Clase III vs `kcap` a nivel regla, y
  %Clase III de cada familia vs el `kcap` medio que le tocó.

**Logs:** `cs090_fase6_o2b_shard0..4.log` (corrida), `cs090_fase6_o2b_analisis.log` (análisis completo).

No se modificó ningún script existente ni congelado. No se corrió Phantom. No se hicieron commits de git.
No se declara cierre ni veredicto — los números de §3-§6 y los caveats de §7 están arriba; la lectura final
es de Alexis.
