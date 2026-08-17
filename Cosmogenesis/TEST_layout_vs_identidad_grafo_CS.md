# Control grafo-random (Erdős-Rényi, INDEPENDIENTE de REAL) — Fase II CS073

**Encargo:** `NULL3_robustecido_motivos_dosis_CS.md` señaló, como hallazgo especulativo NO verificado
directamente, que los NULL1-8 ORIGINALES de CS073 (swap Maslov-Sneppen SIN filtro de longitud, que
destruye ~99.9% de los triángulos de la malla causal REAL) **todavía formaron sumideros PARCIALMENTE**
(masa 680-770), mientras que NULL-1 (radio exacto/ángulo aleatorio) y NULL-2 (Zel'dovich) — que NO pasan
por ningún grafo de vecindad ni por `layout_resortes` — dieron **CERO sumideros en 16 corridas
combinadas**. La lectura especulativa: quizás no es la identidad de la malla causal real lo que importa,
sino el mero hecho de pasar por un grafo de vecindad + `layout_resortes` (relajación de resortes, un
proceso físico), sea cual sea la identidad de ese grafo.

Esta tarea testea la versión más limpia posible de esa hipótesis: un grafo **Erdős-Rényi G(n,m)
genuinamente independiente** de la malla causal REAL (no comparte ni una sola arista con probabilidad
significativa, no comparte grado, no comparte longitud, no comparte ninguna propiedad salvo el orden de
magnitud de n y de nº de aristas), puesto a pasar por el **mismo** `layout_resortes` + misma dilatación +
mismo campo de velocidad + misma escritura ASCII que toda la jerarquía (REAL/NULL-1/NULL-2/NULL-3).

**No se declara cierre ni veredicto sobre CS073 ni sobre la jerarquía — sólo se reportan números. La
lectura final es de Alexis.**

---

## Paso 1 — Construcción del grafo de control

**Método (`grafo_random_layout_generar_ic.py`, función `generar_grafo_erdos_renyi`):** Erdős-Rényi
G(n,m) por *rejection sampling* de pares únicos — se eligen `n_aristas` pares (i,j), i≠j, uniformes
sobre TODOS los C(n,2) pares posibles de n nodos, rechazando pares repetidos o auto-loops, hasta juntar
exactamente `n_aristas` aristas distintas. Con la densidad de esta malla (~0.25% de C(n,2)) la tasa de
colisión es despreciable — a N=2000 se necesitaron 4948 intentos para 4945 aristas (99.94% de
aceptación en el primer intento).

**n y m objetivo:** contados directamente sobre la malla causal REAL ya existente
(`malla_causal_atomos`, D=3, k=4, `seed_ejes=2000` — los MISMOS parámetros que usa `traducir_pool`/
`generar_null3` en toda la jerarquía), función nueva `contar_aristas_malla_real` — **sólo para contar,
nunca se usa ninguna arista de REAL para construir el grafo random**:

| escala | n | n_aristas (REAL) | grado medio |
|---|---|---|---|
| N=500 (piloto) | 500 | 1245 | 4.980 |
| N=2000 (batería completa) | 2000 | 4945 | 4.945 |

**Verificación explícita de independencia** (`grafo_random_motivos.py`): del grafo random generado a
N=2000 (seed=701), sólo **12 de 4945 aristas (0.243%) coinciden con REAL por puro azar** — exactamente
lo esperado por la probabilidad de arista de un grafo aleatorio de esta densidad (p=m/C(n,2)=0.00247).
El grafo NO preserva grado (a diferencia de NULL-3), NO preserva longitud, NO tiene ninguna relación
estructural con REAL más allá del tamaño.

---

## Motivos / triángulos del grafo random (Paso 4, N=2000, seed=701)

Mismo método exacto que `null3_motivos_directos.py` (funciones reusadas, no reescritas):

| grafo | triángulos | clustering promedio | clustering global | ciclos de 4 |
|---|---|---|---|---|
| REAL | 2780 | 0.60588 | 0.40548 | 4402 |
| NULL-3 (tol_relativa=0.2) | 2005 | 0.42405 | 0.29244 | 2844 |
| **Erdős-Rényi (este control)** | **21** | **0.00211** | **0.00257** | **72** |

**El grafo Erdős-Rényi tiene 21 triángulos de 2780 de REAL (−99.2%), prácticamente ninguno** — consistente
con la teoría estándar de grafos aleatorios: el nº esperado de triángulos en G(n,m) es
C(n,3)·p³ = 20.15 (p=m/C(n,2)=0.002474), y el valor medido (21) coincide casi exacto con esa predicción
teórica. Esto confirma que el grafo random construido es, en efecto, un Erdős-Rényi genuino sin
estructura de orden superior — un punto de referencia mucho más extremo que el swap sin restricción de
longitud de los NULL1-8 originales (4 triángulos de 2780, pero ese swap SÍ preserva la secuencia de
grados exacta de REAL, cosa que este control no hace).

---

## Paso 3 — Piloto (N=500, 3 semillas nuevas 901-903)

Pipeline: `grafo_random_piloto_generar.py` (extrae el mismo pool N=500 determinista que usa
`null1_piloto_generar.py`/`null3_piloto_generar.py`, genera el grafo random, `layout_resortes`, misma
dilatación/velocidad/escritura) + `grafo_random_piloto_correr.py` (Phantom, mismo patrón que el resto
de la jerarquía). Comparado contra la REAL de referencia ya en disco (`piloto_null1/real/`).

| corrida | seed | r_mean | r_std | exit setup/run | masa en sumideros | nº sumideros |
|---|---|---|---|---|---|---|
| REAL (referencia) | — | 45.498 | 5.703 | — | 282.0 | 4 |
| random_s1 | 901 | 38.902 | 9.882 | 0/0 | **0.0** | **0** |
| random_s2 | 902 | 39.307 | 9.709 | 0/0 | **0.0** | **0** |
| random_s3 | 903 | 39.160 | 9.830 | 0/0 | **0.0** | **0** |

**Las 3 corridas terminaron limpias (exit 0/0 en `phantomsetup`/`phantom`, sin abortos de conservación,
sin NaN en los logs), pero NINGUNA formó sumideros** — 0 de 3, igual que NULL-1/NULL-2, muy distinto de
NULL-3 (3-5 sumideros a esta misma escala). El piloto salió limpio en todos los criterios de la
salvaguarda → se escaló directo a la batería completa.

---

## Paso 3 (cont.) — Batería completa (N=2000, 8 semillas nuevas 701-708)

Pipeline: `grafo_random_bateria_generar.py` (lee el pool N=2000 ya en disco, genera 8 grafos random
independientes — uno por semilla, `seed_layout=seed` variable como se pidió — `layout_resortes`, misma
dilatación/velocidad/escritura) + `grafo_random_bateria_correr.py` (Phantom, mismos parámetros físicos
de toda la jerarquía). **8/8 exit code 0 en setup y en run, sin abortos de conservación, sin NaN en
ningún log.**

| corrida | seed | r_mean | r_std | masa en sumideros | nº sumideros |
|---|---|---|---|---|---|
| ic_random_s701 | 701 | 62.069 | 15.569 | 1099.8 | 8 |
| ic_random_s702 | 702 | 62.469 | 15.597 | 1165.6 | 8 |
| ic_random_s703 | 703 | 62.080 | 15.909 | 1184.4 | 8 |
| ic_random_s704 | 704 | 62.312 | 15.602 | 1099.8 | 8 |
| ic_random_s705 | 705 | 62.265 | 15.552 | 1156.2 | 8 |
| ic_random_s706 | 706 | 62.574 | 15.339 | 1165.6 | 8 |
| ic_random_s707 | 707 | 62.450 | 15.658 | 1118.6 | 8 |
| ic_random_s708 | 708 | 62.258 | 15.480 | 1156.2 | 8 |
| **media / DE** | — | 62.310 / 0.172 | 15.588 / 0.161 | **1143.28 / 32.54** | **8 (las 8)** |

**Resultado inesperado respecto del piloto: a N=2000, las 8 corridas del grafo Erdős-Rényi (independiente
de REAL) SÍ formaron sumideros — 8 sumideros en cada una de las 8 corridas (igual cantidad que REAL y
NULL-3), con una masa total (1099.8–1184.4, media 1143.3) muy consistente entre semillas (DE=32.5, ~2.8%
de la media — la variabilidad más baja de toda la jerarquía) pero **muy por debajo de REAL/NULL-3
(~52% de su masa) y muy por encima de NULL-1/NULL-2 (masa 0).**

**Nota explícita sobre la discrepancia piloto (N=500) vs batería (N=2000):** a N=500 el mismo mecanismo
(mismo tipo de grafo, mismo pipeline) dio CERO sumideros en las 3 semillas; a N=2000 dio sumideros en
las 8 de 8, de forma consistente. Esto es paralelo a lo que ya se documentó para los NULL1-8 originales
(masa parcial 680-770 a N=2000, nunca corridos a N=500 con este pipeline para comparar). No se investiga
aquí la causa de esta dependencia de escala (¿umbral de masa/densidad para colapso gravitacional que
sólo se cruza con más partículas? ¿mayor nº de vecinos por resolución? — ambas especulativas, no
verificadas) — se señala como dato relevante para que Alexis lo pondere, siguiendo la misma convención
que el hallazgo especulativo original.

---

## Estadísticos de separación (test de permutación exacto, N=2000)

### (a) REAL (n=6) vs CONTROL grafo-random (n=8)
- estadístico observado (media_REAL − media_RANDOM) = **1053.19**
- C(14,6) = 3003 asignaciones, rank = 1 de 3003
- **p (una cola, REAL>RANDOM) = 0.000333** (piso teórico exacto de este diseño)
- z-score = (media_REAL − media_RANDOM) / DE_RANDOM = **32.37**

### (b) NULL-3 (n=8) vs CONTROL grafo-random (n=8) — ¿importa la identidad del grafo?
- estadístico observado (media_NULL3 − media_RANDOM) = **1043.40**
- C(16,8) = 12870 asignaciones, rank = 1 de 12870
- **p (una cola, NULL-3>RANDOM) = 0.0000777** (piso teórico exacto)

### (c) CONTROL grafo-random (n=8) vs NULL-1 (n=8) — ¿el mero hecho de pasar por `layout_resortes` ya produce algo?
- estadístico observado (media_RANDOM − media_NULL1) = **1143.28**
- C(16,8) = 12870 asignaciones, rank = 1 de 12870
- **p (una cola, RANDOM>NULL-1) = 0.0000777** (piso teórico exacto)

### (d) CONTROL grafo-random (n=8) vs NULL-2 (n=8)
- estadístico observado (media_RANDOM − media_NULL2) = **1143.28** (idéntico a (c), NULL-1/NULL-2 dieron ambos masa 0 exacta)
- **p (una cola, RANDOM>NULL-2) = 0.0000777** (piso teórico exacto)

Las 4 comparaciones son estadísticamente nítidas con este diseño y este observable: **el grafo random
se separa con claridad tanto de REAL/NULL-3 (por debajo) como de NULL-1/NULL-2 (por encima)** — cae
exactamente en un punto intermedio, sin solapamiento de rango en ninguna de las 4 comparaciones
(rank=1 en las 4 pruebas, el piso teórico exacto de cada diseño).

---

## Tabla completa del panorama (REAL, NULL-1, NULL-2, NULL-3, control grafo-random)

| control | qué preserva de REAL | pasa por `layout_resortes` | triángulos (N=2000) | masa en sumideros (N=2000, n corridas) | nº sumideros |
|---|---|---|---|---|---|
| **REAL** | — (es la malla real) | sí | 2780 | 2196.47 ± 95.98 (n=6) | 8/8 |
| **NULL-3** (tol=0.2) | grado exacto + longitud de arista | sí | 2005 (−27.9%) | 2186.68 ± 53.16 (n=8) | 8/8 |
| **CONTROL grafo-random** (este test) | nada (Erdős-Rényi independiente, sólo n/m aprox.) | sí | 21 (−99.2%) | **1143.28 ± 32.54 (n=8)** | **8/8** |
| **NULL1-8 originales** (swap sin filtro long., pipeline antiguo `traducir_pool`) | grado exacto (secuencia completa) | sí | 4 (−99.9%) | ≈680-770 (referencia cualitativa, pipeline distinto) | parcial (dato de contexto, no medido directo aquí) |
| **NULL-1** (radio exacto/ángulo aleatorio) | radio/perfil de densidad | **no** | — (no aplica, no hay grafo) | 0.0 ± 0.0 (n=8) | 0/8 |
| **NULL-2** (Zel'dovich, P(k)) | espectro de potencia | **no** | — (no aplica, no hay grafo) | 0.0 ± 0.0 (n=8) | 0/8 |

**Lectura de los números (sin cerrar nada):** con este observable y este diseño, la masa en sumideros
ordena así: REAL ≈ NULL-3 (2186-2196) > **CONTROL grafo-random (1143, ~52% de REAL)** > NULL-1 = NULL-2
(0). El control grafo-random — que NO comparte ninguna estructura con REAL más allá del orden de
magnitud de n/m, y que en su conteo directo de motivos (21 triángulos) es incluso MÁS aleatorio que el
swap sin restricción de los NULL1-8 originales (4 triángulos, pero ese preserva grado exacto) — de
todas formas formó sumideros en las 8 de 8 corridas a N=2000, con una masa intermedia clara y
significativamente distinta tanto de REAL/NULL-3 como de NULL-1/NULL-2. Esto es consistente con la
lectura especulativa de `NULL3_robustecido_motivos_dosis_CS.md` (el mero hecho de pasar por un grafo de
vecindad + `layout_resortes` siembra ALGO, independientemente de la identidad de ese grafo) — pero
también muestra que la identidad/estructura específica de la malla causal real SÍ aporta algo adicional
y estadísticamente nítido (la diferencia RANDOM vs NULL-3/REAL es tan clara como la diferencia RANDOM vs
NULL-1/NULL-2, ambas al piso teórico del test exacto). Ninguna de las dos lecturas se cierra aquí — los
números de arriba son el entregable; la interpretación final es de Alexis.

---

## Salvaguarda de tiempo

Presupuesto pedido: ~40-50 min reales.

| paso | tiempo |
|---|---|
| Motivos directos (grafo random N=2000, seed=701) | 1.1 s |
| Piloto: generación 3 IC (N=500, incluye extracción del pool ~164s) | 177.4 s |
| Piloto: Phantom 3 corridas | 17.3 s |
| Batería: generación 8 IC (N=2000) | 304.6 s |
| Batería: Phantom 8 corridas | 69.8 s |
| **total cómputo** | **≈570 s ≈ 9.5 min** |

Muy por debajo de la salvaguarda de 40-50 min — se completó la batería completa (no sólo el piloto).

---

## Entregables de esta tarea

- `grafo_random_layout_generar_ic.py` — módulo central: `generar_grafo_erdos_renyi` (construcción del
  grafo de control, rejection sampling), `contar_aristas_malla_real` (cuenta n/m de la malla REAL, sólo
  para dimensionar), `generar_control_random` (pipeline: grafo random → `layout_resortes` → dilatación
  → velocidad → escritura ASCII, mismo patrón que `null3_generar_ic.py`). Reusa (importado, no
  modificado) `layout_resortes`/`malla_causal_atomos` de `p_semilla_causal.py`, `Expansion` de
  `p_expansion.py`, `T0`/`_T_reloj` de `cs073_cierre_holistico.py`, `HFACT`/`POLYK` de
  `fase1_traducir_a_phantom.py`, `aristas_de` de `null3_investigacion_preliminar.py`.
- `grafo_random_motivos.py` — Paso 4: triángulos/clustering/ciclos-4 del grafo random vs REAL (reusa
  `contar_triangulos_y_clustering`/`contar_ciclos_4`/`reportar` de `null3_motivos_directos.py`, no
  reescritas), más verificación de independencia (aristas solapadas con REAL) y referencia teórica
  C(n,3)·p³.
- `grafo_random_piloto_generar.py` / `grafo_random_piloto_correr.py` — piloto N=500, semillas 901-903,
  en `/Users/alexis/phantom_cs073/piloto_grafo_random/`.
- `grafo_random_bateria_generar.py` / `grafo_random_bateria_correr.py` / `grafo_random_bateria_comparar.py`
  — batería completa N=2000, semillas 701-708, en
  `/Users/alexis/phantom_cs073/bateria_grafo_random_n2000/`.
- `/Users/alexis/phantom_cs073/piloto_grafo_random/` y `/Users/alexis/phantom_cs073/bateria_grafo_random_n2000/`
  — carpetas nuevas con las corridas de Phantom (IC, `cosmog.in`, `setup.log`, `run.log`, `.sink`,
  dumps). No se tocó ninguna carpeta de batería/piloto anterior ni ningún script congelado — sólo
  lectura/importación.
- Este informe.
