# INFRAESTRUCTURA · Layout de resortes O(N log N) (Barnes-Hut) — validación, curva θ, escalamiento y demo

**Fecha:** 12-ago-2026 · **Ejecuta:** CC (Claude) · **Encargo:** tarea de camino crítico — desbloquear
resolución para F7-07/F7-08 (el grano del instrumento medido en `FASE7_F704_cortar_bien_vs_azar_CS.md`:
1 partícula = 0,0005 de fracción de masa contra residuales de ~0,0016) y para la rama de
regularidad/escalamiento, hoy clavada en N=4000 por el costo del layout.

**No se declara cierre ni veredicto: se reportan números.** Ningún archivo congelado fue modificado —
`cs072_modulos/piezas/p_semilla_causal.py` (donde vive `layout_resortes`),
`cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_correr.py`, `cs090_fase5b_analizar.py`,
`cs090_fase6_o4a_observable_comun.py` sólo se **importan**. No se hicieron commits de git.

**Archivos nuevos de esta tarea:**

| archivo | qué es |
|---|---|
| `cs090_layout_barnes_hut.py` | el layout nuevo, intercambiable con `layout_resortes` |
| `cs090_layout_bh_validar.py` | la validación (fuerza, layout completo, escalamiento, figuras) |
| `cs090_layout_bh_demo_n8000.py` | demo end-to-end grafo → layout → IC → Phantom a N=8000 |
| `cs090_layout_bh_fuerza.csv` | test de fuerza: error y tiempo por θ, N = 500…4000 |
| `cs090_layout_bh_validacion.csv` / `_resumen.csv` | layout completo sobre 12 grafos reales, N=2000 |
| `cs090_layout_bh_escala.csv` / `_theta03.csv` | cronometraje N = 1000…16000, con θ=0,5 y con el θ operativo 0,3 |
| `cs090_layout_bh_error_vs_velocidad.png`, `cs090_layout_bh_escalamiento.png` | las dos figuras |

---

## 0. En simple, con analogía

El layout de resortes decide **dónde nace cada partícula** antes de que Phantom la deje caer. Para eso,
en cada una de sus 100 iteraciones, hace que **cada partícula se empuje con todas las demás**: N×N
empujones. Al doblar N el trabajo se cuadruplica (y en la práctica peor: se midió exponente 2,74). Por eso
la serie está clavada en N=4000, y por eso los residuales de ~0,0016 que perseguimos están a **tres
partículas** de la resolución del instrumento.

La idea de Barnes-Hut (1986) es la de cualquiera parado en una plaza: para saber cuánto tira de mí la
multitud del fondo **no necesito preguntarle uno por uno** — me alcanza con tratar a ese grupo lejano como
**un bulto en su centro**. Sólo a los que tengo al lado los cuento de a uno. Se arma una caja que se
subdivide en ocho, y otra vez en ocho (un "octree"); para cada partícula se baja por el árbol y, si una
caja **se ve chica desde donde estoy** — su ancho dividido por la distancia es menor que un umbral θ —, se
usa su centro de masa y no se abre. El costo pasa de N² a ~N·log N.

El riesgo obvio: es una **aproximación**. La pregunta no es "¿da lo mismo?" (no da lo mismo, ni puede),
sino **"¿se equivoca menos de lo que el método viejo se equivoca consigo mismo?"**. Y esa vara existe y
está medida en el proyecto: el layout Fruchterman-Reingold es **caóticamente sensible**. En
`FASE6_O3B_control_rewiring_CS.md` se re-corrió el layout original sobre el **mismo grafo con la misma
semilla** y en 4 de 12 casos las coordenadas terminaron a **15-23 unidades** de distancia, porque una
diferencia de nivel 1e-16 se amplifica en 100 iteraciones. Ése es el temblor propio de la balanza. La
balanza nueva tiene que temblar menos que eso.

---

## 1. Qué se mantiene idéntico (fidelidad de protocolo antes que elegancia)

`layout_barnes_hut()` tiene la **misma firma** que `layout_resortes()` más `theta`. Cambia **una sola
cosa**: cómo se suma el término de repulsión. Todo lo demás está copiado línea por línea:

| pieza del protocolo | estado |
|---|---|
| semilla y consumo del RNG (`default_rng(seed).uniform(0, lado, (n, dims))`) | **idéntico** |
| k_FR = (lado³/n)^(1/3) | **idéntico** |
| construcción de la lista de aristas (misma expresión, mismo orden de recorrido) | **idéntico** |
| término atractivo sobre aristas + `np.add.at` (ya era O(M)) | **idéntico, sin tocar** |
| enfriamiento: `paso = lado*0.1`, `paso_it = paso*(1 - it/iters)`, 100 iteraciones | **idéntico** |
| cap del desplazamiento `(desplaz/norm)*min(norm, paso_it)`, `norm==0 → 1.0` | **idéntico** |
| frontera reflectante `_reflejar` | **importada** del módulo congelado |
| post-proceso `sep_min` (`_imponer_separacion_minima`, default `None`) | **importado**, default sin cambio |
| dilatación posterior `Expansion` 60 pasos, turbulencia Mach=3 seed=42, masa fija 18800, lado fijo 2000^(1/3) | fuera del layout, **sin cambio** |

**La verificación dura de esa fidelidad, hecha primero:** el módulo nuevo trae un modo
`metodo="exacta"` que corre el bucle nuevo con la suma N² vieja. Sobre el mismo grafo y la misma semilla,
`layout_barnes_hut(..., metodo="exacta")` devuelve **`np.array_equal(...) == True`, diferencia máxima
exactamente 0.0** contra `layout_resortes`. O sea: el bucle es el mismo bucle. Lo único que se está
evaluando de acá en adelante es el árbol.

**La fórmula que hay que reproducir**, exactamente como está en `p_semilla_causal.py` (líneas 116-120):

```
delta_ij = pos_i − pos_j ;  dist_ij = |delta_ij| + 1e-6 ;  f_ij = k_FR² · delta_ij / dist_ij²
```

El `+1e-6` es un regularizador **aditivo** y va antes de la diagonal-infinito: la fuerza **no** es un
1/r limpio. La aproximación de monopolo lo respeta: una celda con `cnt` partículas y centro de masa `c`
aporta `cnt · k_FR² · (pos_i − c)/(|pos_i − c| + 1e-6)²`.

**Criterio de apertura, escrito como producto y no como cociente:** `aceptar ⟺ ancho < θ · distancia`.
Así queda bien definido cuando la partícula cae justo sobre el centro de masa de una celda (distancia 0):
la celda **se abre**, que es lo correcto, y no hay división por cero. Con **θ=0 la condición es imposible**
→ ninguna celda se acepta nunca → el árbol se abre hasta las hojas → la suma es la **misma suma
partícula-a-partícula del original**.

**Implementación:** octree **uniforme** de profundidad D ≈ log₈(N/4) sobre la caja [0, lado]³ (que es
exactamente el dominio, porque `_reflejar` devuelve ahí las posiciones en cada iteración). Sin recursión
en Python: conteos y centros de masa por nivel con `np.bincount`, y el recorrido se hace **nivel por
nivel** sobre una lista vectorizada de pares (partícula, celda) — los pares aceptados descargan su
monopolo y salen, los abiertos se expanden a las hasta 8 hijas no vacías. En la hoja, los pares que
siguen abiertos se resuelven partícula a partícula, excluyendo la propia partícula.

---

## 2. TEST 1 — ¿θ=0 reproduce la suma exacta? (el test de bugs)

Script: `cs090_layout_bh_validar.py fuerza` → `cs090_layout_bh_fuerza.csv`. Se compara **una sola
evaluación de fuerza** (sin dinámica, sobre las mismas posiciones) contra la suma N² del original.

La comparación se hace contra un **control de redondeo**: la misma suma N², con los **mismos sumandos**,
sumada en **orden inverso** (`repulsion_exacta_reordenada`). Matemáticamente es idéntica; en punto
flotante difiere en el último bit. Ese control da la escala de lo que "exacto" puede significar en
float64, porque la suma en coma flotante **no es asociativa**.

| N | error relativo medio, **BH θ=0** | error relativo medio, **control de redondeo** (misma cuenta, otro orden) |
|---|---|---|
| 500 | **7,28e-16** | 7,46e-16 |
| 1000 | **9,95e-16** | 1,03e-15 |
| 2000 | **1,44e-15** | 1,48e-15 |
| 4000 | **2,03e-15** | 2,09e-15 |

**Los cuatro puntos: BH con θ=0 está por DEBAJO del control de redondeo.** No hay ningún residuo por
encima del épsilon de máquina. La respuesta a "¿θ=0 reproduce el O(N²) exactamente?" es **sí, en el único
sentido en que la pregunta tiene respuesta en float64: reproduce el mismo conjunto de sumandos, y la
diferencia que queda es más chica que la que la propia suma exacta tiene consigo misma al cambiarle el
orden.** No hay bug en el árbol.

*(Dicho al revés, y sin adornos: bit a bit no puede ser, y no lo es. Pedir igualdad bit a bit sería pedir
que el árbol sume en el mismo orden que `numpy.sum(axis=1)`, que es una propiedad del recorrido de
memoria, no del algoritmo.)*

**La confirmación dinámica** (el layout entero, 100 iteraciones, sobre dos grafos reales de la serie —
donde una diferencia de 1e-15 tiene 100 pasos para crecer):

| grafo | BH θ=0 vs original: RMS / Δ FoF b=0,20 | control de redondeo vs original: RMS / Δ FoF b=0,20 |
|---|---|---|
| `batch4-r43` | 0,333 / −0,0010 | 0,375 / +0,0110 |
| `batch4-r57` | 0,292 / +0,0180 | 0,333 / −0,0040 |

En los dos casos el árbol con θ=0 aterriza **dentro** de la dispersión que la propia suma exacta produce
al cambiarle el orden de los sumandos (RMS incluso algo menor). No hay nada que distinga a BH θ=0 de "el
original corrido dos veces".

---

## 3. TEST 2 — la curva error-vs-velocidad de θ (a nivel de fuerza)

Misma tabla, ahora mirando los θ > 0. `interacciones` es la medida **algorítmica** del costo (cuántos
pares partícula-celda o partícula-partícula se evalúan por iteración), independiente de la máquina.

| N=4000 | err. rel. medio | err. rel. máx | interacciones/iter | interacc. por partícula | tiempo por evaluación |
|---|---|---|---|---|---|
| exacta (N²) | — | — | 15 996 000 | 3 999 | 10,28 s |
| θ = 0,0 | 2,0e-15 | 7,5e-15 | 15 996 000 | 3 999 | 24,51 s |
| θ = 0,3 | 3,39e-03 | 4,81e-03 | 3 282 836 | 821 | 5,50 s |
| θ = 0,5 | 9,91e-03 | 1,56e-02 | 1 207 370 | 302 | 2,20 s |
| θ = 0,7 | 2,33e-02 | 3,85e-02 | 580 050 | 145 | 1,29 s |
| θ = 1,0 | 4,59e-02 | 7,28e-02 | 256 743 | 64 | 0,62 s |

Dos lecturas que conviene tener presentes:

1. **El error relativo por θ es casi independiente de N** (θ=0,5 da 8,0e-3 / 9,1e-3 / 9,7e-3 / 9,9e-3 a
   N = 500/1000/2000/4000). O sea: θ fija la precisión, N fija el costo. Eso es lo que uno quiere de un
   parámetro de calidad.
2. **θ=0 es más caro que la suma exacta** (24,5 s contra 10,3 s a N=4000): hace las mismas N²
   interacciones más el costo del árbol. θ=0 es un instrumento de auditoría, no un modo de producción.

---

## 4. TEST 3 — el layout COMPLETO sobre grafos reales de la serie, a N=2000

Script: `cs090_layout_bh_validar.py worker <i>` → `cs090_layout_bh_validacion.csv` +
`cs090_layout_bh_resumen.csv`; figura `cs090_layout_bh_error_vs_velocidad.png`.

**Los grafos**: los **12** lanzados, tomados de los pares seleccionados en O3-A
(`cs090_fase6_o3a_pares_seleccionados.json`) y reconstruidos **con sus semillas** por
`reconstruir_regla_a2b0c2(seed, N=2000, n_sweeps=14)` — los mismos grafos que ya corrieron en la serie,
no grafos nuevos. Cubren 6 pares repartidos a lo largo del ranking de Δmasa (los dos brazos de cada par).

**Las varas**: exactamente las de `cs090_fase6_o3a_geometria_ic.py` — `fof_masa` con b = 0,20 / 0,30 /
0,50 y umbral de masa fija 47,0, más `dens_k8_cv`, sobre las posiciones ya dilatadas por el mismo
`Expansion` de 60 pasos que aplica la IC real.

### 4.1 Primero, el piso de ruido — y es grande

| variante | n | RMS de posición vs original | máximo de posición | tiempo | aceleración |
|---|---|---|---|---|---|
| **control de redondeo** (misma suma N², orden inverso) | 12 | **0,321** | **3,11** | 190,4 s | ×0,98 |
| BH θ=0,0 (testigo) | 2 | **0,312** | **2,62** | 507,8 s | ×0,37 |
| BH θ=0,3 | 12 | 1,231 | 11,70 | 119,1 s | ×1,59 |
| BH θ=0,5 | 12 | 1,356 | 12,06 | 68,0 s | ×2,75 |
| BH θ=0,7 | 12 | 1,479 | 12,09 | 36,5 s | ×5,09 |
| BH θ=1,0 | 12 | 1,662 | 12,46 | 22,7 s | ×8,31 |

(layout original: 185,9 s de media por grafo, bajo la misma contención.)

Lo primero que hay que ver son las **dos primeras filas**. Cambiar **nada más que el orden de la suma** —
una perturbación de 1e-16 — mueve las posiciones finales un RMS de **0,321** y hasta **3,11** unidades en
una caja de lado 12,6. Y el árbol con **θ=0**, que hace la misma cuenta exacta, se mueve **0,312 / 2,62**:
*menos* que el propio original consigo mismo. Es la sensibilidad caótica del FR que `FASE6_O3B` había
visto por accidente, ahora reproducida a pedido y medida en 12 grafos. **A nivel de coordenadas,
"reproducir el layout" no es una propiedad que este método tenga**, tampoco con el algoritmo viejo. Por
eso la comparación de posiciones se informa pero no decide: la que decide es la de los observables.

### 4.2 Los observables — diferencia media CON SIGNO (lo que importa es el sesgo, no el desorden)

Diferencia media (variante − original) sobre los 12 grafos, ± error estándar, con Wilcoxon pareado:

| variante | Δ FoF b=0,20 | Δ FoF b=0,30 | Δ FoF b=0,50 | Δ dens_k8_cv |
|---|---|---|---|---|
| **control de redondeo** | +0,0029 ± 0,0042 (p=0,79) | +0,0015 ± 0,0012 (p=0,23) | +0,0007 ± 0,0007 (p=0,23) | −0,017 ± 0,038 (p=0,68) |
| BH θ=0,0 *(n=2, testigo)* | +0,0085 | −0,0008 | −0,0025 | −0,097 |
| **BH θ=0,3** | +0,0071 ± 0,0039 (p=0,06) | +0,0038 ± 0,0018 (p=0,06) | +0,0024 ± 0,0012 (p=0,11) | −0,045 ± 0,021 (**p=0,02**) |
| BH θ=0,5 | +0,0038 ± 0,0057 (p=0,34) | +0,0061 ± 0,0015 (**p=0,00**) | +0,0059 ± 0,0017 (**p=0,00**) | −0,013 ± 0,041 (p=0,85) |
| BH θ=0,7 | +0,0109 ± 0,0029 (**p=0,00**) | +0,0125 ± 0,0018 (**p=0,00**) | +0,0118 ± 0,0017 (**p=0,00**) | −0,005 ± 0,040 (p=0,85) |
| BH θ=1,0 | +0,0211 ± 0,0043 (**p=0,00**) | +0,0160 ± 0,0021 (**p=0,00**) | +0,0161 ± 0,0015 (**p=0,00**) | −0,013 ± 0,034 (p=0,79) |

Y las mismas cifras en valor absoluto (la magnitud típica del corrimiento, con el piso de ruido arriba de
todo para comparar; entre paréntesis, cuántas veces el piso):

| variante | \|Δ\| FoF b=0,20 | \|Δ\| FoF b=0,30 | \|Δ\| FoF b=0,50 | \|Δ\| dens_k8_cv |
|---|---|---|---|---|
| **piso de ruido (control de redondeo)** | **0,0112** | **0,0036** | **0,0020** | **0,109** |
| BH θ=0,0 *(n=2, testigo)* | 0,0095 *(0,85×)* | 0,0078 *(2,1×)* | 0,0035 *(1,8×)* | 0,097 *(0,89×)* |
| **BH θ=0,3** | 0,0095 *(0,85×)* | 0,0059 *(1,6×)* | 0,0039 *(1,9×)* | 0,053 *(0,49×)* |
| BH θ=0,5 | 0,0161 *(1,4×)* | 0,0070 *(1,9×)* | 0,0062 *(3,1×)* | 0,104 *(0,96×)* |
| BH θ=0,7 | 0,0115 *(1,0×)* | 0,0125 *(3,4×)* | 0,0118 *(5,9×)* | 0,100 *(0,91×)* |
| BH θ=1,0 | 0,0219 *(2,0×)* | 0,0160 *(4,4×)* | 0,0161 *(8,1×)* | 0,093 *(0,86×)* |

### 4.3 Qué dice esto, sin adornos

1. **El error de θ NO es desorden: es un SESGO con dirección.** Todos los θ dan Δ FoF **positivo**: el
   layout aproximado nace **más apelotonado** que el exacto. Tiene sentido físico — reemplazar un grupo
   lejano por su centro de masa **subestima** la repulsión de sus partículas más cercanas, así que la nube
   se expande un poco menos. Y el sesgo **crece monótonamente con θ** en las dos varas más finas (b=0,30 y
   b=0,50): +0,0038 → +0,0061 → +0,0125 → +0,0160.
2. **θ=0,7 y θ=1,0 quedan fuera:** sesgo sólido (p=0,00 en las tres varas de FoF) y de 3 a 8 veces el piso
   de ruido en b=0,50. No son admisibles.
3. **θ=0,5 tampoco pasa:** p=0,00 en b=0,30 y b=0,50, con sesgo de +0,006 (3,1× el piso en b=0,50). El
   θ=0,5 clásico de la literatura **no** sirve en este pipeline.
4. **θ=0,3 es el único que no cruza el umbral en ninguna vara de FoF** (p = 0,06 / 0,06 / 0,11) — y hay
   que decirlo con todas las letras: **p=0,06 es "al borde", no "cero"**. Con n=12 el sesgo de θ=0,3
   (+0,0024 a +0,0071) es *chico pero probablemente real*, y de la misma dirección que el de los demás
   θ. Lo que sí queda establecido es su tamaño: **0,85× el piso de ruido en b=0,20, 1,6× en b=0,30, 1,9×
   en b=0,50 y 0,49× en `dens_k8_cv`** — o sea, del mismo orden que el temblor propio del método.
5. **El testigo θ=0** (n=2 grafos, layout completo de 100 iteraciones) da 0,85× / 2,1× / 1,8× / 0,89× el
   piso: indistinguible del control de redondeo, como tiene que ser. Sirve como calibración de cuánto de
   lo que se ve en la fila de θ=0,3 es aproximación y cuánto es simplemente caos con n=12.
6. **La vara b=0,20 no ordena** (0,3 < 0,7 < 0,5 < 1,0 en valor absoluto). Es la de piso más alto
   (0,0112): con n=12 no resuelve diferencias de este tamaño. Se la reporta, no se la usa para decidir.
   Por eso la figura `cs090_layout_bh_error_vs_velocidad.png` grafica la curva error-vs-velocidad con
   b=0,50, que es la de piso más bajo y la que sí ordena.

### 4.4 El θ operativo elegido, y su letra chica

> **θ = 0,3.** Criterio: es el θ más grande cuyo sesgo inducido sobre los observables de la serie **no
> cruza el umbral de detección en ninguna de las tres varas de FoF** (p = 0,06 / 0,06 / 0,11, contra
> p=0,00 de θ ≥ 0,5) y cuya magnitud está **dentro del factor 2 del ruido propio del método** (0,85× el
> piso en b=0,20; 1,6× en b=0,30; 1,9× en b=0,50; 0,49× en `dens_k8_cv`). Cuesta ×1,59 menos que el
> original a N=2000, y la ventaja crece con N (§5).

**Tres advertencias que van junto con la decisión, no después:**

- **No mezclar layouts dentro de una comparación ni dentro de una serie.** El sesgo de θ=0,3 (+0,0024 a
  +0,0071 en FoF) es **mayor que el residual que persigue F7-04** (~0,0016 en fracción de masa). Como es
  un sesgo del LAYOUT y no de la clase de regla, en una comparación **pareada** III − I se cancela casi
  entero — pero sólo si los dos brazos usan el mismo layout y el mismo θ. Un punto viejo (N²) contra un
  punto nuevo (BH) **no** es una comparación limpia.
- **El sesgo de θ=0,3 no está probado cero, está acotado.** p=0,06 con n=12 significa que si hiciera falta
  cerrar esa rendija, el camino es más grafos, no más argumentos. Y si en algún contraste el sesgo
  importara, θ se puede bajar: el error escala suavemente y θ=0,2 o θ=0,15 siguen siendo mucho más
  rápidos que N² a N grande.
- **Esto no valida retroactivamente la serie vieja.** Lo medido es que el layout nuevo con θ=0,3 no es más
  ruidoso que el viejo consigo mismo. Lo que NO se midió — porque no se puede — es que el layout viejo
  fuera reproducible: no lo es, y el piso de 0,0112 en b=0,20 (0,321 de RMS en posición) es una propiedad
  del método, no de este cambio.

---

## 5. Escalamiento medido del layout nuevo

Script: `cs090_layout_bh_validar.py escala_rapida` → `cs090_layout_bh_escala.csv`, figura
`cs090_layout_bh_escalamiento.png`.

**Cómo se midió, y por qué así.** La máquina estuvo compartida durante toda la tarea con ~20 agentes en
paralelo (load average 73 con 16 núcleos; ver §6.1). Cronometrar 100 iteraciones completas en esas
condiciones habría medido la contención, no el algoritmo. Se hizo entonces lo siguiente:

- se cronometra **una evaluación de repulsión** (el 100 % de lo que cambia; el término atractivo es O(M)
  y ya era barato), y se multiplica por 100 para dar el costo del layout;
- se toman **3 repeticiones y se reporta el mínimo**, que es el estimador robusto bajo contención;
- y sobre todo se reporta el **número de interacciones por iteración**, que es la medida **algorítmica**
  del costo y no depende de la máquina en absoluto.

| N | interacciones/iter | por partícula | prof. octree | t por evaluación (min de 3) | **t layout, 100 iters** | original O(N²), medido en O3-A §8.1 |
|---|---|---|---|---|---|---|
| 1 000 | 198 301 | 198 | 3 | 0,166 s | **16,6 s** | 23,3 s |
| 2 000 | 523 496 | 262 | 3 | 0,737 s | **73,7 s** | 98,4 s |
| 4 000 | 1 207 370 | 302 | 4 | 2,028 s | **202,8 s** | 656 s |
| 8 000 | 2 970 951 | 371 | 4 | 4,611 s | **461,1 s** (7,7 min) | ~4 380 s (≈73 min, extrapolado) |
| 16 000 | 7 160 479 | 448 | 4 | 10,63 s | **1 062,6 s** (17,7 min) | ~28 600 s (≈8 h, extrapolado con 2,74) |

**Exponentes ajustados (log-log):**

| tramo | exponente del **tiempo** | exponente de las **interacciones** |
|---|---|---|
| 1 000 → 2 000 | 2,15 *(inflado: el punto de N=1000 pescó una ventana sin contención — mediana 0,36 s contra mínimo 0,166 s)* | 1,40 |
| 2 000 → 4 000 | 1,46 | 1,21 |
| 4 000 → 8 000 | **1,19** | 1,30 |
| 8 000 → 16 000 | **1,20** | 1,27 |
| **global 1 000-16 000** | **1,465** | **1,285** |

**El número que importa:** en el tramo que interesa (4 000 → 16 000) el exponente medido es **1,19-1,20**,
contra el **2,74** medido para el layout original entre 2 000 y 4 000. El costo del layout a **N=8000 pasa
de ~73 min a 7,7 min (×9,5)** y **N=16000 deja de ser inalcanzable**: 17,7 min por grafo, contra las ~8 h
que costaría con el algoritmo viejo.

### 5.1 Y con el θ operativo (0,3), que es el que se va a usar

La tabla de arriba está medida con θ=0,5, que fue el θ con el que se cronometró y con el que corrió la
demo. Como §4.4 elige **θ=0,3**, hay que decir cuánto cuesta ése — y el número honesto es que **cuesta
~2,5 veces más**. Lo machine-independiente, medido:

| N | interacciones/partícula, θ=0,5 | interacciones/partícula, **θ=0,3** | cociente | t layout 100 iters, θ=0,3 (medido) |
|---|---|---|---|---|
| 2 000 | 262 | **667** | ×2,55 | 241 s *(y 119 s en el layout completo real, §4.1 — la diferencia es contención)* |
| 4 000 | 302 | **821** | ×2,72 | 735 s |
| 8 000 | 371 | **1 087** | ×2,93 | **1 682 s (28 min)** |
| 16 000 | 448 | **1 395** | ×3,11 | **3 983 s (66 min)** |

Exponente del tiempo con θ=0,3: **1,33 global**; por tramos, 1,61 (2 000→4 000), **1,19** (4 000→8 000),
**1,24** (8 000→16 000). Es el mismo comportamiento que con θ=0,5 — θ mueve la constante, no el exponente.

O sea: con **θ=0,3**, N=8000 cuesta **28 min por grafo** contra los ~73 min del layout viejo (**×2,6**), y
N=16000 cuesta **66 min** contra las ~8 h extrapoladas del viejo (**×7,2**). Menos espectacular que el
×9,5 de θ=0,5, pero es el número que corresponde al θ que pasa la validación — y la ventaja **sigue
creciendo con N**, porque el exponente es 1,2 contra 2,74. *(Los tiempos de esta tabla se midieron con la máquina saturada: el
mismo layout θ=0,3 a N=2000, cronometrado dentro de una corrida real en §4.1, dio 119 s y no 241 s. Como
cotas superiores sirven; como medida de la máquina libre, dividir por ~2.)*

*(Por qué el exponente no es el 1,0-1,1 del N·log N de manual: el octree es UNIFORME y su profundidad se
elige por escalones — D pasa de 3 a 4 entre N=2000 y N=4000 y ahí se queda hasta N=16000. Con la
profundidad congelada, la ocupación de hoja crece con N y con ella el trabajo directo por partícula (302 →
371 → 448 interacciones). Es un margen de mejora conocido y disponible — un octree adaptativo, o subir D a
5 por encima de N≈20000 —, no un límite del método.)*

---

## 6. Demostración end-to-end a N=8000

Script: `cs090_layout_bh_demo_n8000.py`. Se generó el **par extremo** de O3-A (`batch4-r23`, Clase I,
seed 574060, y `batch4-r10`, Clase III, seed 572799 — el par más favorable a N=2000, y el mismo par que
O3-A intentó y tuvo que abortar a N=8000). Protocolo idéntico al de la serie: masa fija 18800, lado fijo
2000^(1/3), `Expansion` 60 pasos, turbulencia Mach=3 seed 42, 100 iteraciones de layout, seed_layout
12345, 14 sweeps del motor, y en Phantom `icreate_sinks=1`, `rho_crit_cgs=1000`, `r_crit=0.600`,
`h_acc=0.300`, `f_acc=0.800`, `tmax=0.500`, `dtmax=0.001`.

**Aviso de honestidad sobre el θ de la demo:** la demo se corrió con **θ=0,5**, no con el θ=0,3 que §4.4
termina eligiendo. Se lanzó antes de que la validación estuviera cerrada, y con la máquina saturada no
alcanzaba el presupuesto para rehacerla. Eso **no afecta** lo que la demo demuestra (que la cadena
completa corre a esa resolución, y cuánto cuesta cada eslabón), pero sí quiere decir que **los números
físicos de estas dos corridas no son puntos de la serie**: para eso hay que regenerarlas con θ=0,3, lo
que multiplica el costo del layout por ×2,9 (de 461 s a **1 682 s medidos**, §5.1) y deja igual todo lo
demás.

### 6.1 Las condiciones en que se midió

La máquina estuvo compartida durante toda la tarea con el resto de los agentes de la sesión:
**load average 73 y en un momento 207, sobre 16 núcleos**, con procesos ajenos al 170 % y 110 % de CPU.
Los procesos de esta tarea recibieron entre el 8 % y el 45 % de un núcleo. **Todos los tiempos de abajo
son de pared bajo esa contención**, o sea cotas superiores: en una máquina libre serían menores.

### 6.2 Lo que costó cada eslabón

| eslabón | N=8000, medido | referencia N=4000 (O3-A) | referencia N=8000 con el layout viejo |
|---|---|---|---|
| motor relacional (grafo, 14 sweeps) | incluido abajo | 3,7 s | — |
| **grafo + `layout_barnes_hut` θ=0,5 + escribir la IC** | **339 s (Clase I) / 345 s (Clase III)**, de pared y con la máquina saturada | 656 s sólo el layout | **~4 380 s ≈ 73 min** (extrapolado, O3-A §8.1) |
| el mismo layout, cronometrado en limpio (§5) | 461 s | 202,8 s | — |
| `phantomsetup` + Phantom | ver §6.3 | 19-62 s | nunca se llegó a medir |

O sea: **el eslabón que bloqueaba la serie dejó de bloquearla.** Las dos condiciones iniciales completas
de N=8000 —8 000 partículas, 13 569 y 12 084 aristas, masa por partícula 2,35— quedaron escritas en
`/Users/alexis/phantom_cs073/infra_layout_bh_demo_n8000/N8000/`, en **menos de 6 minutos cada una y en
paralelo**, contra los 17 min 51 s en que O3-A abortó sin terminar **ninguno** de 4 layouts.

### 6.3 Phantom a N=8000: el cuello de botella se mudó

Phantom **arrancó, corrió y produjo métricas**: `phantomsetup` aceptó las IC sin quejas, el bloque de
sumideros se reescribió igual que siempre, la integración avanzó, y `cs090_fase5b_analizar.analizar_carpeta`
—el mismo lector congelado de toda la serie— leyó el resultado sin tocar nada. La cadena completa
**grafo → layout → IC → phantomsetup → Phantom → métricas funciona a N=8000.**

| | Clase I (`batch4-r23`) | Clase III (`batch4-r10`) |
|---|---|---|
| grafo + layout BH + IC | 339 s | 345 s |
| Phantom, corrido hasta el tope | 1 500 s (25 min), `timeout=True` | 1 500 s (25 min), `timeout=True` |
| último dump alcanzado | `cosmog_00327` de 501 (t ≈ 0,327 de tmax=0,500) | `cosmog_00259` (t ≈ 0,259) |
| sumideros formados | **121** | **127** |
| fracción de masa en sumideros (dump parcial) | 0,1978 | 0,2007 |
| **costo total por corrida, medido** | **1 504 s ≈ 25 min** (y sin llegar a tmax) | 1 504 s ≈ 25 min |

**Ninguna de las dos llegó a tmax.** Los 121-127 sumideros **coinciden con el antecedente**:
`ON77_sistemaA_cierre` había medido ~122 sumideros a N=8000, y ésa fue justamente la razón por la que
aquella corrida tuvo que abortarse. Al momento del corte el paso de tiempo ya se estaba partiendo (bin 1
con dt = 5·10⁻⁴, 44 930 fallos acumulados). **Es el mismo fenómeno de siempre, y ahora se lo puede ver de
cerca porque ya no está tapado por el costo del layout.**

**El número que hay que llevarse:** a N=8000, **el layout dejó de ser el cuello de botella y pasó a serlo
Phantom.** El reparto medido es ~6 min de grafo+layout (θ=0,5; ~28 min con θ=0,3) contra **más de 25 min
de integración sin terminar** — extrapolando el ritmo observado y sabiendo que se degrada al aparecer más
sumideros, una corrida completa a N=8000 está en el orden de **1-2 h de Phantom**. Cualquier trabajo
futuro para hacer viable N=8000 en tanda tiene que atacar **el lado de Phantom** — paso de tiempo
individual, `h_acc`, política de fusión de sumideros —, no el layout. Es un cambio de diagnóstico
respecto de O3-A §8.1, que dejó escrito "no es más RAM, es cambiar el algoritmo del layout": el algoritmo
del layout ya se cambió, y el problema que quedó es otro.

*(Las fracciones de masa de la tabla son de dumps PARCIALES y de corridas hechas con θ=0,5. **No son
puntos de la serie ni se comparan entre sí**: son la prueba de que el lector devuelve números sobre esta
cadena, nada más. Ver el aviso de θ en el encabezado de §6.)*

---

## 7. Reproducir

```bash
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis
./venv/bin/python cs090_layout_barnes_hut.py                 # autocomprobación rápida
./venv/bin/python cs090_layout_bh_validar.py fuerza          # TEST 1 y 3 (tabla de fuerza)
for i in 0 2 4 6 8 10 13 15 17 19 21 23; do \
  ./venv/bin/python cs090_layout_bh_validar.py worker $i & done; wait   # TEST 2
./venv/bin/python cs090_layout_bh_validar.py escala_rapida 1000,2000,4000,8000,16000
./venv/bin/python cs090_layout_bh_validar.py tabla           # CSV resumen + las dos figuras
./venv/bin/python cs090_layout_bh_demo_n8000.py A2-B0-C2-batch4-r23 574060 I  8000 0.5 1500
./venv/bin/python cs090_layout_bh_demo_n8000.py A2-B0-C2-batch4-r10 572799 III 8000 0.5 1500
```

*(La demo se corrió con θ=0,5 y tope de 1500 s — ver el aviso de §6. Para regenerarla con el θ operativo,
cambiar `0.5` por `0.3` y subir el tope: el layout pasa de ~460 s a ~1 680 s por grafo.)*

---

## 8. Lo que queda pendiente, dicho explícito

1. **Phantom a N=8000 no llegó a tmax** dentro del tope de 25 min de esta demo (llegó a t≈0,33 y t≈0,26
   de 0,500, con 121 y 127 sumideros). Cuánto cuesta realmente una corrida COMPLETA a esa resolución
   sigue **sin medir** — es el número que hay que ir a buscar antes de comprometer una batería, y ahora
   es el único que falta, porque el layout ya no estorba.
2. **El sesgo de θ=0,3 está acotado, no descartado** (p=0,06 con n=12). Si hiciera falta cerrarlo: más
   grafos, o bajar a θ=0,2.
3. **El octree es uniforme.** Un octree adaptativo bajaría el exponente de 1,2 hacia el 1,05-1,10 teórico
   y ayudaría sobre todo de N=16000 para arriba.
4. **Nada de esto reprocesa la serie existente.** El layout viejo sigue intacto y sigue siendo el que
   generó todos los puntos publicados. Adoptar el nuevo es una decisión del director, no de esta tarea.
