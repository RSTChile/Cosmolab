# CS087 — Laplaciano de Hodge L₁ (aristas): ¿el componente armónico explica la Pared de Fase IV?

**Script:** `cs087_hodge_fase4.py` (nuevo; importa `cs082_fase4_4sustratos.py` sin tocarlo) · **Fecha:** ago-2026
**Antecedentes:** `FASE4_orden_superior_resultado_CS.md` (cs082), `FASE4_robustecido_CS.md` (cs083),
`CS084_espectro_laplaciano_resultado_CS.md` (L₀ de nodos).
**Estado:** exploratorio. Reporta números. **Ningún resultado de este documento cierra ni confirma nada
— el veredicto final es de Alexis.**

---

## 1. La pregunta y por qué L₀ no alcanzaba

CS084 ya usó el laplaciano de grafo simple L₀=D-A (autovalores sobre NODOS) y encontró que el tejido
real "suena" más ancho que el ruido — pero L₀ no puede, por construcción, ver nada de lo que pasa en
los TRIÁNGULOS: la holonomía que Fase IV midió (`cs082`/`cs083`) vive en las ARISTAS y en los CICLOS de
3, no en los nodos. El objeto correcto es el laplaciano de Hodge de aristas, **L₁**, construido con los
operadores de borde de un complejo simplicial orientado:

- `∂₁` : aristas → nodos (matriz de incidencia arista-nodo, con signo — el borde de siempre del grafo)
- `∂₂` : caras → aristas (matriz de incidencia cara-arista con signo: para una cara ordenada (i,j,k)
  con i<j<k, su borde orientado es **+arista(j,k) − arista(i,k) + arista(i,j)** — la fórmula estándar de
  un 2-símplice, la única elección de signos que garantiza `∂₁·∂₂ = 0`, condición necesaria para que
  la teoría de Hodge tenga sentido)
- **L₁ = ∂₁ᵀ∂₁ + ∂₂∂₂ᵀ** (down-Laplacian + up-Laplacian, ambos actuando sobre el espacio de aristas)

Por el teorema de descomposición de Hodge discreta, toda arista-señal se parte en 3 partes ortogonales:
**gradiente** (explicada por un potencial de nodo), **rotor/curl** (explicada por alguna cara — el
"borde de un triángulo") y **armónico** (autovalores ~0 de L₁ — ciclos que NO son borde de NINGUNA cara
presente). El componente armónico es exactamente el candidato que pidió la tarea: la parte de la
holonomía que ninguna cara, por más que empuje, puede jamás cerrar.

**Se aplicó a los sustratos 3 (simplicial pasivo) y 4 (2-complejo con feedback) de `cs082`** — los dos
únicos con caras/triángulos definidos como objeto-relación propio. Sustratos 1 (diádico) y 2
(hipergrafo) se documentan aparte en §2, sin forzar un L₁ equivalente para ellos.

**Método:** `construir_operadores_borde()` construye ∂₁,∂₂ directamente de `edges`/`triangles` que
devuelve `cs082.construir_base(seed)` (sin tocar ese archivo). 8 semillas (1-8; las 5 primeras
comparables 1:1 con `cs082`/`cs083`). `∂₁ᵀ∂₂` se verificó exactamente cero (`max|∂1·∂2|=0.00e+00` en
las 8 semillas) — chequeo de sanidad matemática pasado. Diagonalización densa completa (`np.linalg.eigh`,
numpy solo) de matrices |E|×|E| (495-606 aristas) — 32.6s en total para las 8 semillas × 2 sustratos ×
3 brazos (REAL/NULL/SHUFFLED, reusando `null_de`/`shuffled_de` de `cs082` sin cambios).

---

## 2. Sustratos 1 y 2: por qué no se les forzó un L₁ equivalente

- **Sustrato 1 (grafo diádico):** tiene las MISMAS aristas que 3/4, pero en su propia definición NO
  hay ninguna cara activa (el objeto-relación es sólo la arista — las caras del grafo base existen
  geométricamente, pero la dinámica del sustrato 1 nunca las usa). Se reporta un caso **degenerado
  explícito**: `L₁_deg = ∂₁ᵀ∂₁` solamente (sin el término de caras) — NO es equivalente al L₁ completo
  de 3/4, sólo sirve para mostrar cuánto CRECE el espacio "armónico" cuando ninguna cara explica los
  ciclos (ver tabla §3).
- **Sustrato 2 (hipergrafo):** su objeto-relación nativo es el triángulo-como-hiperarista, no la
  arista — el campo de aristas que expone (`E`) es una PROYECCIÓN derivada (promedio circular de las
  hiperaristas que tocan cada par), y ni siquiera cubre necesariamente todas las aristas base. Construir
  un ∂₁/∂₂ para un hipergrafo 3-uniforme exigiría inventar una orientación que no es parte del modelo —
  se documenta la razón y se lo deja fuera de esta batería en vez de forzarlo.

---

## 3. Hallazgo metodológico previo (hay que leerlo antes que la tabla de resultados)

Los sustratos 3 y 4 de `cs082` corren sobre **exactamente la misma base combinatoria** — mismo
`construir_base(seed)`, mismas aristas, mismos triángulos; sólo cambia la DINÁMICA (3 = sin feedback, 4
= con feedback cara→arista). Como ∂₁ y ∂₂ son matrices de incidencia 0/±1 que dependen SÓLO de qué
aristas y qué caras existen — no de los valores Z₆ que trae cada corrida —, **el espectro completo de
L₁, y en particular la dimensión del subespacio armónico (número de Betti b₁), es IDÉNTICO entre
sustrato 3 y sustrato 4 en la misma semilla.** Esto no es un error del script: es la primera pieza de
evidencia de esta tarea — la topología ("el tablero") no cambia entre pasivo y activo, sólo cambia dónde
cae el campo (los valores dinámicos) sobre ese tablero fijo.

| seed | \|E\| | \|T\| | b₁ (L₁ completo, 3=4) | b₁_deg (sólo ∂₁ᵀ∂₁, sustrato 1) | b₁/b₁_deg |
|---:|---:|---:|---:|---:|---:|
| 1 | 545 | 161 | 276 | 436 | 0.633 |
| 2 | 513 | 161 | 248 | 404 | 0.614 |
| 3 | 606 | 221 | 282 | 497 | 0.567 |
| 4 | 562 | 188 | 275 | 453 | 0.607 |
| 5 | 583 | 193 | 283 | 474 | 0.597 |
| 6 | 530 | 145 | 276 | 421 | 0.656 |
| 7 | 558 | 167 | 284 | 449 | 0.633 |
| 8 | 495 | 131 | 255 | 386 | 0.661 |

**Lectura:** agregar las caras SÍ reduce el espacio armónico (de b₁_deg a b₁, en promedio una caída del
~38%) — las caras "cierran" cerca de 4 de cada 10 ciclos que antes quedaban sueltos. Pero incluso con
todas las caras presentes, **~50-66% del espacio de aristas sigue siendo armónico** (b₁/|E| ≈ 0.45-0.51
en las 8 semillas) — es decir, más de la mitad de los "lazos" posibles en este grafo NO son borde de
ningún triángulo del grafo, sea cual sea el sustrato. Esto pone un **techo topológico** fijo: ningún
mecanismo que sólo empuje a través de caras (como el feedback del sustrato 4) puede, en principio,
tocar más de la mitad del espacio de aristas — la otra mitad es, por construcción del grafo, invisible
a cualquier corrección basada en triángulos.

---

## 4. Resultado principal — proyección del campo real sobre L₁ (promedio ± DE, 8 semillas)

| sustrato | brazo | fracción armónica | curl-energy/cara (up-Laplacian) | grad-energy/arista (down) | holonomía mod-K (métrica de cs082, ya medida) |
|---|---|---:|---:|---:|---:|
| 3_simplicial | REAL | 0.464±0.026 | 0.237±0.028 | 0.210±0.026 | 1.167±0.438 |
| 3_simplicial | NULL | 0.500±0.047 | 8.763±0.597 | 5.594±0.754 | 1.556±0.086 |
| 3_simplicial | SHUFFLED | 0.523±0.036 | 0.245±0.018 | 0.149±0.023 | 1.150±0.453 |
| 4_2complejo | REAL | 0.485±0.065 | 0.253±0.191 | 0.364±0.628 | **0.261±0.120** |
| 4_2complejo | NULL | 0.500±0.047 | 8.763±0.597 | 5.594±0.754 | 1.556±0.086 |
| 4_2complejo | SHUFFLED | 0.509±0.032 | 0.309±0.303 | 0.226±0.278 | 0.485±0.364 |

(La columna "holonomía mod-K" es la métrica original de `cs082`/`cs083` —reimportada tal cual, sin
tocarla— y sirve de chequeo de reproducibilidad: 1.167/1.556/0.261 acá contra 1.20/1.54/0.30 del informe
original — coincide, confirmando que esta batería replica bien la anterior antes de agregar L₁ encima.)

**Lo que muestra esta tabla, leído con cuidado:**

1. **La "fracción armónica" (columna 2) NO distingue nada** — REAL, NULL y SHUFFLED caen todos entre
   0.46 y 0.52 en ambos sustratos, prácticamente el mismo valor que la proporción topológica b₁/|E|
   (~0.45-0.51 de §3). Para un campo genérico proyectado sobre un subespacio de dimensión relativa ~0.5,
   la energía se reparte ~50/50 casi sin importar si el campo tiene estructura o es ruido puro — el
   componente armónico, medido así (proyección lineal directa), **no separa nada entre brazos ni entre
   sustratos**.

2. **La "curl-energy" (columna 3, la forma cuadrática del up-Laplacian, ∂₂∂₂ᵀ) SÍ separa fuertemente
   NULL (≈8.8) de REAL/SHUFFLED (≈0.24-0.31)** — pero separa IGUAL en sustrato 3 que en sustrato 4, y
   **no distingue REAL de SHUFFLED** en ninguno de los dos (0.237 vs 0.245 en el 3; 0.253 vs 0.309 en el
   4, con desviaciones tan grandes que la diferencia no es sólida). Esto reproduce sólo el contraste
   "hubo dinámica" vs "no hubo dinámica" — exactamente el mismo tipo de separación amplia-pero-poco-fina
   que CS084 ya encontró con L₀ (λ_max/std_eig separan REAL de NULL con facilidad, pero no resuelven
   estructura más fina).

3. La columna de holonomía mod-K (la métrica ORIGINAL de Fase IV) es la ÚNICA de las cuatro que
   reproduce el patrón fino que cs082/cs083 encontraron: sustrato 3 casi no se mueve de NULL (1.167 vs
   1.556), sustrato 4 sí (0.261 vs 1.556), y dentro del sustrato 4, SHUFFLED (0.485) queda entre REAL y
   NULL — la misma jerarquía REAL < SHUFFLED < NULL que motivó todo `cs083`.

---

## 5. ¿Por qué L₁ "de libro" no reproduce lo que la holonomía de cs082 sí ve?

Se hizo el chequeo directo: por cada semilla y brazo, se calculó el **curl lineal con signo estándar**
(`∂₂ᵀ · campo`, SIN el mod-K) triángulo por triángulo, y se lo correlacionó (Pearson) contra la
holonomía mod-K de `cs082` (import directo, sin cambios) para los MISMOS triángulos:

| sustrato | brazo | r promedio (8 semillas) |
|---|---|---:|
| 3_simplicial | REAL | +0.075 |
| 3_simplicial | NULL | −0.016 |
| 3_simplicial | SHUFFLED | −0.020 |
| 4_2complejo | REAL | +0.040 |
| 4_2complejo | NULL | −0.016 |
| 4_2complejo | SHUFFLED | +0.142 |

**La correlación es esencialmente nula en todos los casos** (|r|<0.15). Dos razones concretas, ambas
verificables por inspección del código:

- **Convención de signos distinta.** `cs082._holonomia_triangulos` suma los TRES valores de arista SIN
  alternar signo (`eij+ejk+eik`, los tres con +), mod K, centrada — una medida de "cuánto desacuerdan
  los tres bordes", no una holonomía de conexión propiamente dicha. El ∂₂ estándar (el que exige la
  teoría de Hodge para que `∂₁·∂₂=0`) **alterna signo** (`+arista(j,k) − arista(i,k) + arista(i,j)`) —
  es la única elección consistente con una orientación de borde bien definida. Son dos objetos
  matemáticos DISTINTOS por construcción, no la misma cantidad con nombres distintos.
- **El wrap-around de Z₆ rompe la linealidad.** L₁ es un operador lineal sobre números reales; el campo
  real vive en un círculo de circunferencia 6. Se aplicó una mitigación (recentrado circular por media
  circular antes de restar) pero no elimina el problema — dos valores angularmente cercanos (ej. 0.1 y
  5.9) pueden seguir viéndose como muy lejanos en la resta lineal si caen en semillas/tripletas distintas
  del cúmulo. La consecuencia práctica más visible: la forma cuadrática v^T·L₁_up·v (curl-energy, §4)
  queda dominada por la **varianza marginal** de la distribución de valores (qué tan disperso está el
  campo en general), no por si CADA triángulo específico cierra — por eso SHUFFLED (mismos valores,
  aristas permutadas) da un curl-energy casi idéntico a REAL: permutar valores dentro de un mismo cúmulo
  de baja varianza no cambia mucho la varianza marginal, aunque destruya por completo el cableado
  correcto cara↔sus-propias-3-aristas. Es justo la variable que `cs083` tuvo que aislar con un control
  quirúrgico (rewire fino) y un test pareado por semilla (z=−4.14) para poder ver — la forma cuadrática
  "de libro" de L₁, sin ese cuidado, no la ve.

---

## 6. Respuesta a la pregunta central de la tarea

**¿El espectro de L₁ (en particular su componente armónico) explica por qué sólo el sustrato 4 separaba
de NULL en holonomía?** Con la evidencia de arriba, la respuesta honesta es **no, no de forma directa —
y el porqué es en sí mismo informativo:**

- **El espectro puro (b₁, la dimensión armónica) es idéntico entre sustrato 3 y 4** — no puede ser la
  fuente de la diferencia observada en holonomía, porque no cambia entre ellos. Lo que Fase IV encontró
  no es un cambio de TOPOLOGÍA (qué ciclos son o no borde de una cara) sino un cambio en **dónde cae el
  campo dinámico** sobre una topología fija — algo que el espectro por sí solo no puede capturar, sólo la
  proyección del campo específico.
- **La proyección directa del campo sobre el subespacio armónico (fracción armónica) tampoco separa
  nada** — se queda pegada a ~0.5 (el valor "genérico" esperado por la sola dimensión relativa del
  subespacio) en los 6 combos sustrato×brazo. No hay evidencia, en esta medición, de que el feedback del
  sustrato 4 empuje el campo preferentemente HACIA o LEJOS del subespacio armónico.
- **La energía de curl (up-Laplacian) sí distingue "hubo dinámica" de "no la hubo"** (REAL/SHUFFLED
  ≪ NULL, en ambos sustratos) — consistente con, y del mismo tipo que, lo que CS084 ya vio con L₀. Pero
  **no reproduce la distinción más fina que SÍ importa** (sustrato 3 vs 4, o REAL vs SHUFFLED dentro del
  4) — la misma distinción que `cs083` necesitó un control quirúrgico y un test pareado para aislar
  (~8% del efecto, z=−4.14). Esto sugiere que el ~8% local que `cs083` encontró es una señal DEMASIADO
  FINA para el aparato "de libro" de Hodge tal como se aplicó acá — no refuta esa señal, simplemente
  este instrumento en particular, sin el mismo cuidado metodológico de cs083, no la resuelve.
- **El techo topológico de §3 (≥50% del espacio de aristas es armónico incluso con todas las caras
  presentes) es compatible, como lectura posible no probada, con por qué el componente LOCAL que cs083
  aisló es chico (~8%) y no grande:** cualquier mecanismo de corrección basado en caras (como el
  feedback del sustrato 4) sólo puede, en principio, actuar sobre la mitad "curl" del espacio de aristas
  — la otra mitad (armónica) es estructuralmente inalcanzable para ese mecanismo, sea cual sea su
  fuerza. Esto es una hipótesis interpretativa, NO una prueba causal — no se corrió ningún experimento
  que varíe la densidad de caras para confirmarla.

**En síntesis: L₁ no "resuelve" el misterio de Fase IV de un plumazo — lo que sí aporta es (a) confirmar
que la diferencia sustrato-3-vs-4 vive en el campo, no en la topología compartida, (b) mostrar que las
cantidades espectrales "de libro" (armónico puro, curl-energy cruda) son demasiado gruesas para ver el
efecto fino de 8% que `cs083` encontró con métodos más quirúrgicos, y (c) señalar un techo topológico
(mitad del espacio de aristas es armónico) que es coherente con, aunque no demuestra, por qué ese efecto
local resultó chico.**

---

## 7. En simple, con analogía

Pensá en una sala con 110 personas y unos 500-600 "hilos de acuerdo" tendidos entre pares que se
conocen (las aristas), y unos 150-220 "jueces de trío" (los triángulos) que pueden revisar si tres
personas conectadas entre sí quedaron de acuerdo LOS TRES a la vez.

**Primer hallazgo (topológico):** en esta sala hay tan pocos jueces de trío comparados con la cantidad
de hilos, que **más de la mitad de los posibles lazos de la sala NUNCA tienen un juez que los revise** —
sin importar si los jueces existen sólo para anotar (sustrato 3) o si además corrigen activamente
(sustrato 4). Es un límite del PLANO DE LA SALA, no de qué tan bien trabajan los jueces.

**Segundo hallazgo (con el campo real puesto encima):** cuando medís, con la regla estándar de "cuánta
energía cae en la parte que ningún juez puede ver" (el componente armónico), la respuesta es "más o
menos la mitad, siempre" — no importa si la sala tiene jueces activos, jueces pasivos, o si todos hablan
al azar. Esa regla, tal como se la aplicó acá, es demasiado gruesa para notar la diferencia.

**Tercer hallazgo:** una regla relacionada, "cuánta tensión total hay en los tríos que SÍ tienen juez"
(la energía de curl), sí nota la diferencia entre "hubo alguna conversación" (REAL o incluso una versión
con los mismos comentarios pero mezclados al azar entre hilos, SHUFFLED) contra "puro ruido sin ninguna
conversación" (NULL). Pero **no nota la diferencia entre que el juez corrija a LOS TRES CORRECTOS o a
tres cualquiera** — que es justo lo único que `cs083` había encontrado que SÍ importaba (un 8% chico
pero real). Es como usar una balanza de baño para medir un cambio de un gramo: la herramienta funciona,
pero no tiene la sensibilidad para esa pregunta en particular — hace falta la balanza de precisión
(el control quirúrgico y el test pareado) que `cs083` ya construyó.

---

## 8. Qué NO se reclama

- No se afirma que la holonomía de Fase IV esté "explicada" ni "reducida a" L₁ — sólo se muestra, con
  números, qué tan bien (o mal) el aparato estándar de Hodge la reproduce.
- La correlación nula entre curl lineal y holonomía mod-K (§5) NO es evidencia de que cs082/cs083 estén
  mal — es evidencia de que son construcciones matemáticas distintas (signo alternado vs sin alternar,
  lineal vs mod-K), documentado explícitamente para que quede claro que no es la misma cantidad con dos
  nombres.
- La hipótesis del "techo topológico" (§6, último punto) es una lectura posible, no un resultado
  probado — no se corrió ningún control que varíe la densidad de caras para testearla.
- Sustratos 1 y 2 no se corrieron con la misma profundidad (ver §2) — por presupuesto de tiempo y
  porque el enunciado de la tarea pedía priorizar 3/4.
- Escala igual a `cs082`/`cs083` (N=110, K=6, 8 semillas, un solo punto de J/J_FACE/ruido) — no se barrió
  espacio de parámetros.
- Ningún resultado de este documento se declara cerrado, confirmado o refutado. Los números están
  arriba. El veredicto es de Alexis.

**Reproducibilidad:** `cs087_hodge_fase4.py`, numpy-only, importa `cs082_fase4_4sustratos.py` sin
tocarlo, corre en ~33s (`./venv/bin/python3 cs087_hodge_fase4.py`). Salidas: `cs087_hodge_fase4.csv`
(48 filas: 8 semillas × 2 sustratos × 3 brazos) y `cs087_espectros_L1.npz` (espectros completos de L₁ y
del degenerado ∂₁ᵀ∂₁, por semilla).
