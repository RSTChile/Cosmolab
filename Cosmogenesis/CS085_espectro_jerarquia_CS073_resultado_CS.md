# CS085 — el espectro del laplaciano (CS084) aplicado a la jerarquía de 6 controles de CS073

**Fecha:** 9-ago-2026 · **Origen:** pregunta directa de Alexis — CS084 encontró que el espectro completo
del laplaciano de grafo separa el tejido real de sus controles NULL con una limpieza que el diámetro
nunca logró, pero eso se midió sobre el tejido puro de CS066 (sin gravedad, sin Phantom). CS073 es la
línea DISTINTA donde sí se sabe con precisión, escalón por escalón, qué preserva/destruye cada control,
y donde sí se corrió Phantom de verdad para medir sumideros. **No se declara cierre ni veredicto — sólo
se reportan números. La lectura final es de Alexis.**

## Qué se hizo

Se reconstruyeron, sin tocar ningún script congelado (sólo import/lectura), los grafos puros (sin
Phantom) de tres escalones de la jerarquía CS073:

- **REAL**: la malla causal (`malla_causal_atomos`, D=3, k=4) con `seed_ejes=2000` — la CANÓNICA, la que
  corrió Phantom en el resto de la jerarquía — más **4 variantes genuinas** (mismo método,
  `seed_ejes=2001..2004`, mallas REALES distintas de verdad, no repeticiones de la misma).
- **NULL-3**: double-edge-swap con filtro de longitud (`tol_relativa=0.2`) sobre la malla canónica, 5
  semillas de barajado (501-505) — ~11-13% de aristas distintas de REAL en esta corrida (consistente con
  el ~12.5% ya documentado en `NULL3_investigacion_preliminar`).
- **RANDOM**: Erdős-Rényi G(n,m) independiente (mismo n=2000, mismo nº de aristas=4945 que REAL, sin
  ninguna otra relación), 5 semillas (9001-9005).
- **NULL-4** y **NULL-5**: no se reconstruyó una batería — se confirmó por qué su espectro TIENE que ser
  idéntico al de REAL (ver más abajo).
- **NULL-1 / NULL-2**: no tienen grafo de fondo — documentado, no forzado.

Sobre cada grafo se corrieron los mismos 3 diagnósticos de `cs084_espectro_laplaciano.py`, reusando sus
funciones de cálculo tal cual (`dimension_espectral`, `unfolding_local`, `estadisticas_espaciado`):
forma del espectro (λ_max, media, dispersión), dimensión espectral por núcleo de calor, y estadística de
espaciado de niveles (Poisson vs GOE). Tiempo total de cómputo: **25.4 s** (muy por debajo del
presupuesto de 45-55 min — a N=2000 la diagonalización tarda ~0.9s/matriz, no ~65s como a N=8000 en
CS084).

**Nota honesta sobre semillas de REAL:** la malla canónica (`seed_ejes=2000`) es la única que corrió
Phantom en toda la jerarquía CS073 — a nivel de TOPOLOGÍA, es n=1. Los 4 `seed_ejes` adicionales dan
variabilidad genuina (mismo mecanismo, mallas realmente distintas) pero **no** son la malla que produjo
los números de sumideros reportados en `NULL5_resultado_CS.md` — se marcan aparte en la tabla.

## Tabla comparativa — lo que el espectro mide

| grupo | n | λ_max | λ2 (algebraica) | std(eig) | d_s(t=1.0) | n_componentes | grado: min–max (std) |
|---|---|---|---|---|---|---|---|
| **REAL canónica** (seed_ejes=2000) | 1 | 11.581 | 0.0199 | 2.4505 | 1.963 | 1 (conexo) | 4–10 (σ=1.03) |
| **REAL variantes** (seed_ejes 2001-04) | 4 | 11.31 – 11.43 | 0.0196 – 0.0222 | 2.457 – 2.489 | 1.96 – 2.05 | 1 (conexo) | ~4–10 |
| **NULL-3** (swap+filtro long., 5 semillas) | 5 | 11.53 – 11.79 | 0.077 – 0.158 | **2.4505 (idéntico, ver nota)** | 2.45 – 2.54 | 1 (conexo) | 4–10 (exacto, mismo grado que REAL) |
| **RANDOM** (Erdős-Rényi, 5 semillas) | 5 | **15.55 – 17.59** | **0.234 – 0.314** | 3.14 – 3.20 | 2.73 – 2.83 | **10–17 fragmentos** (giant 99.2-99.5%) | 0–13 (σ=2.22) |

**Nota matemática, no hallazgo empírico:** el double-edge-swap preserva la secuencia de grados de cada
nodo EXACTAMENTE, y `std(eig)` (segundo momento del espectro) está determinado por completo por esa
secuencia (`trace(L²) = Σdeg_i² + 2·nº_aristas`, una identidad algebraica) — por eso NULL-3 da
`std(eig)=2.4505` IDÉNTICO a REAL canónica en las 5 semillas, sin ninguna variación: no es que el swap
"no cambie nada", es que ESE estadístico en particular no puede detectar lo que el swap sí cambia. Los
estadísticos que sí son sensibles al reordenamiento de aristas (no sólo al grado) son λ_max, λ2, y las
curvas de núcleo de calor/espaciado — por eso la tabla se apoya en esos, no en `std(eig)`, para comparar
NULL-3 contra REAL.

## Chequeos de sanidad: NULL-4 y NULL-5

- **NULL-4** (topología 100% idéntica a REAL, orden de inserción de aristas rebarajado): se reconstruyó
  con un reorden nuevo (semilla 4001), se verificó por `assert` que el CONJUNTO de aristas es idéntico a
  REAL, y se comparó el espectro completo: **max|Δeigenvalue| = 0.000e+00** — exactamente cero, no "casi
  cero". Esto es lo esperado matemáticamente (L=D-A depende sólo del conjunto de aristas, nunca del orden
  en que se insertaron) — se confirma numéricamente en vez de asumirse, y se reporta como chequeo de
  sanidad, no como resultado nuevo.
- **NULL-5** (topología Y conjunto de posiciones 100% idénticos a REAL, sólo permuta qué nodo ocupa cuál
  posición en el archivo de Phantom): ni siquiera hace falta recomputar nada — según el Paso 0 de
  `NULL5_resultado_CS.md`, NULL-5 nunca vuelve a tocar `adj`; la adyacencia por índice de nodo es
  literalmente la misma matriz que REAL. diff=0.0 exacto por definición, no por cómputo.

## NULL-1 / NULL-2: sin grafo de fondo

NULL-1 (radio exacto de REAL, ángulo aleatorio isótropo) y NULL-2 (Zel'dovich, espectro de potencia)
actúan directo sobre posiciones — nunca pasan por `malla_causal_atomos` ni por `layout_resortes`. No
hay ningún `adj` que darle a este diagnóstico. Se documenta como N/A explícito, no se fuerza un grafo
artificial para poder llenar una celda de tabla.

## La pregunta central — ¿escala con la estructura preservada?

**Sí, y de forma limpia y monótona en λ2 y en la dimensión espectral por núcleo de calor — pero en la
dirección OPUESTA a la que encontró CS084.**

En CS084 (tejido de CS066), el REAL tenía λ_max mucho MÁS ALTO que sus controles (más "picos" agudos,
motivos/hubs locales concentrados). Acá pasa lo contrario: **RANDOM tiene λ2 y λ_max MÁS ALTOS que
REAL/NULL-3, sin solapamiento entre RANDOM y el par REAL/NULL-3** (λ2: REAL≤0.022, NULL-3 en
0.077-0.158, RANDOM en 0.234-0.314 — orden monótono limpio REAL < NULL-3 < RANDOM, sin solape entre
ningún par de grupos). Lo mismo con la dimensión espectral a t=1.0 (REAL~2.0, NULL-3~2.5, RANDOM~2.8).

La explicación no es misteriosa, y se verificó directamente en el grado de cada grafo: la malla causal
REAL de CS073 es un grafo casi-regular (grado 4-10, σ=1.03 — viene de un método tipo k-vecinos-más-
cercanos, k=4) — muy homogéneo, sin hubs, con una estructura "en cadena/local" que la hace difícil de
cortar en dos mitades bien separadas (λ2 bajo = "cuesta desconectar"). El grafo Erdős-Rényi con el
MISMO número de aristas tiene grado 0-13 (σ=2.22, distribución de Poisson) — bastante más irregular, con
11 nodos aislados (grado 0) de 2000 y 73 con un solo enlace, lo que de hecho lo deja FRAGMENTADO en
10-17 piezas separadas (REAL/NULL-3 siempre dan 1 sola pieza conexa). Un grafo con nodos casi aislados
es, contra-intuitivamente, MÁS fácil de "cortar" globalmente (λ2 alto) — el cuello de botella ya está
puesto por los propios nodos débilmente conectados.

Es decir: el espectro SÍ escala con cuánta estructura de REAL se preserva, pero lo que mide acá no es
"cuántos armónicos agudos tiene" (como en CS084) sino **cuán fácil es partir el grafo en dos** — y esa
propiedad decrece (se hace más difícil partir, λ2 baja) cuanto más se preserva de la malla causal real.

## ¿Correlaciona con la formación de sumideros ya medida?

| escalón | λ2 (algebraica) | masa en sumideros (Phantom, ya medida) | nº sumideros |
|---|---|---|---|
| REAL | 0.0199 – 0.0222 | 2196.47 ± 95.98 (n=6) | 8/8 |
| NULL-3 | 0.077 – 0.158 | 2186.68 ± 53.16 (n=8) | 8/8 |
| RANDOM (grafo+layout) | **0.234 – 0.314** | **1143.28 ± 32.54 (n=8), ~52% de REAL** | 8/8 |
| NULL-1 / NULL-2 | N/A (sin grafo) | 0.0 ± 0.0 (n=8) | 0/8 |
| NULL-4 (idéntico a REAL) | idéntico a REAL (0.0 exacto de diferencia) | 2136.93 ± 33.01 (n=3) | 8-9/9 |
| NULL-5 (idéntico a REAL) | idéntico a REAL (0.0 exacto de diferencia) | 2124.4 exacto (n=2) | 8/8 |

**Correlación parcial, en la dirección esperada pero NO como predictor fino:** el grupo con λ2 más alto
por lejos (RANDOM, 0.234-0.314) es también el único de los tres con grafo que forma MENOS masa en
sumideros (52% de REAL) — la dirección coincide: "más fácil de partir en dos" (λ2 alto) ↔ "menos masa
retenida en sumideros". Pero dentro del par que SÍ forma sumideros completos (REAL y NULL-3), λ2 difiere
entre ellos por un factor de 4 a 8× (0.02 vs 0.08-0.16) sin que la masa en sumideros se mueva casi nada
(2196 vs 2187, diferencia <1%, ya reportada como REAL≈NULL-3 con p=0.42 en `NULL5_resultado_CS.md`). Es
decir: el espectro es sensible a un cambio de estructura (el 12-13% de aristas que NULL-3 reemplaza) que
Phantom, al nivel de masa total en sumideros, no llega a notar — el instrumento espectral es más fino
que el observable físico en este punto concreto, no al revés.

## Síntesis, en simple

Pensá en la malla causal REAL como una **retícula de vecinos** (cada partícula conectada a 4-10 vecinas
cercanas, más o menos parejo para todos) — es como una red de pesca bien tejida, sin nudos sueltos ni
hilos colgando. El grafo Erdős-Rényi al azar, con el mismo número de hilos totales, es más como tirar
hilos sueltos entre puntos elegidos a ciega — la mayoría caen bien, pero unos 11 puntos de 2000 quedan
sin ningún hilo (aislados) y otros con uno solo, así que la red completa queda floja, con puntos débiles
por donde se rasgaría fácil (eso es lo que mide λ2: qué tan fácil es "cortar" la red en dos partes). Esa
diferencia de textura (retícula pareja vs red floja con nudos sueltos) es justo lo que el espectro
detecta — y la red pareja (REAL, y también NULL-3, que reordena hilos pero mantiene el mismo tejido
parejo) es la que, en la simulación física real, terminó reteniendo el doble de masa en sus sumideros que
la red floja al azar. El espectro no predice la masa exacta (REAL y NULL-3 "suenan" bastante distinto
entre sí, pero terminan con casi la misma masa), pero sí distingue con toda claridad la red pareja de la
red floja, y esa distinción va en la misma dirección que el resultado físico ya medido.

## Archivos

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs085_espectro_jerarquia_cs073.py` — código nuevo, no
  toca `null3_generar_ic.py`, `null4_generar_ic.py`, `null4_verificar_invarianza_orden.py`,
  `grafo_random_layout_generar_ic.py`, `cs084_espectro_laplaciano.py` (sólo import/lectura).
- `cs085_espectro_jerarquia_cs073.csv` — datos crudos completos (15 grafos: 5 REAL + 5 NULL-3 + 5 RANDOM,
  todas las columnas del diagnóstico).
- Este informe.

No se declara cierre ni veredicto sobre CS073, CS084, ni la comparación entre ambos — los números de
arriba son el entregable; la síntesis final es de Alexis.
