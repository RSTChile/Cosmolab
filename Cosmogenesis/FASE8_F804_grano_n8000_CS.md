# F8-04 · ¿Cuál es el GRANO del instrumento a N=8000? — y ¿cuánto cuesta una corrida completa?

**12 de agosto de 2026** · **Ejecuta:** CC (Claude) · **Encargo:** la tarea que decide si la alta
resolución es viable, ANTES de que nadie gaste en una batería a N=8000.

**No se declara cierre ni veredicto sobre la Teoría: se reportan números** y, al final, un veredicto
**operativo** sobre viabilidad de instrumento. Ningún archivo existente fue modificado; todo lo nuevo está
listado abajo. No se hicieron commits.

| archivo nuevo | qué es |
|---|---|
| `cs090_f804_grano_n8000.py` | el experimento: perturbación de redondeo, layout, IC, Phantom, medición al dump común |
| `cs090_f804_lanzador.sh` | encadena Phantom detrás de cada layout apenas su IC está escrita |
| `cs090_f804_grano_n8000.csv` | CSV crudo: una fila por réplica |
| `cs090_f804_grano_n8000.resumen.json` | σ por grafo |
| `cs090_f804_sigma_vs_tiempo.csv` | **σ dump por dump** — la tabla central de §2.1 |
| `cs090_f804_grafos/` | **los grafos guardados** (adyacencia + metadatos), compartidos por las réplicas |
| `cs090_f804_costo_por_dump.py` / `.csv` | costo de pared por dump y por resolución (la parte 2) |

Las 12 corridas quedaron en `/Users/alexis/phantom_cs073/f804_grano_n8000/N8000/`.

### Los cuatro números, arriba de todo

| pregunta | respuesta medida |
|---|---|
| **grano (σ) a N=8000**, 6 réplicas, t=0,246 (54-78 sumideros) | **0,00952** (= 76 partículas) |
| ¿es menor que **0,0143** (F7-03)? | **Sí a t=0,246** (1,5σ). **No** extrapolado a tmax (σ≈0,0160) |
| ¿es menor que **0,0016** (F7-04)? | **NO**, por un factor de 6 |
| **costo de una corrida completa** | **no medible: 0 de 14 corridas a N=8000 llegan a tmax=0,500** |

---

## 0. En simple, con analogía

Una balanza de cocina que pesa "al gramo" no sirve para pesar una pestaña. Antes de comprar una balanza
más grande hay que preguntarle dos cosas: **cuánto tiembla la aguja cuando no pasa nada** (el grano), y
**cuánto tarda en estabilizarse** (el costo). Este trabajo le hace esas dos preguntas a la balanza de
N=8000.

Para medir el temblor se hace lo mismo que se le haría a una balanza real: se pesa **el mismo objeto**
muchas veces, cambiando **nada** — y se mira cuánto baila el número. Acá "cambiar nada" es literal: se
mueve el último bit de un número (una parte en 10 000 000 000 000 000), lo suficiente para que la
aritmética de la máquina redondee distinto y nada más. Si la aguja baila más que el efecto que uno
quiere pesar, la balanza no sirve para eso — por más grande que sea.

---

## 1. Qué se midió y cómo

### 1.1 Los grafos, y que son los mismos de siempre

Los **2 grafos** son el par extremo de O3-A, el mismo par que corrió la demo de infraestructura a N=8000:

| clave | rule_id | seed | clase | aristas a N=8000 | grado medio | diámetro (`cs090_diam_corregido`) |
|---|---|---|---|---|---|---|
| `r23_I` | `A2-B0-C2-batch4-r23` | 574060 | I | **13 569** | 3,392 | 16 |
| `r10_III` | `A2-B0-C2-batch4-r10` | 572799 | III | **12 084** | 3,021 | 17 |

Los conteos de aristas **coinciden exactamente** con los de la demo (`INFRA_layout_barnes_hut_CS.md`
§6.2: 13 569 y 12 084). Los grafos quedaron **guardados** en `cs090_f804_grafos/` — se generan una sola
vez y las réplicas los comparten, porque el motor relacional es idéntico en todas: **lo único que cambia
entre réplicas es el redondeo del layout.**

### 1.2 La perturbación: dónde se inyecta el 1e-16, y por qué ahí

El control de redondeo que usó la validación a N=2000 (`metodo="exacta_reordenada"` de
`cs090_layout_barnes_hut.py`: la misma suma N² en orden inverso) da **una sola** variante alternativa, y a
N=8000 exige la suma N² completa (~73 min de layout por réplica). Para 6 réplicas hace falta una
**familia** de perturbaciones del mismo orden y del mismo carácter.

Se usa **`lado`, el lado de la caja del layout, movido k ULPs** (`np.nextafter`, k = 0…5). El ULP de
`lado = 2000^(1/3) = 12,599210498948731` es **1,78e-15 absoluto = 1,4e-16 relativo**: exactamente el orden
del control de redondeo. Es la inyección más limpia disponible porque el layout Fruchterman-Reingold es
**covariante de escala** — `pos ~ lado`, `k_FR ~ lado`, la repulsión `k_FR²·δ/d² ~ lado`, la atracción
`d²/k_FR ~ lado`, el paso `lado·0,1 ~ lado` —, así que multiplicar `lado` por (1+ε) da, **en aritmética
exacta, el mismo layout multiplicado por (1+ε)**: la misma configuración salvo un reescalado de 1e-16, que
en un observable de fracción de masa es rigurosamente nada. **Todo lo que aparezca por encima de 1e-16 en
las posiciones finales es redondeo amplificado por las 100 iteraciones** — que es justo lo que se quiere
medir. Y no obliga a tocar una línea de código congelado: `lado` es un argumento.

### 1.3 La verificación del instrumento, hecha PRIMERO (a N=2000, donde hay vara)

Si esta perturbación temblara **menos** que el control de redondeo documentado, estaría midiendo un piso
falso. Se comprobó a N=2000, con el mismo grafo y el mismo layout, comparando posiciones finales:

| perturbación | RMS de posición vs k=0 | máximo | tiempo de layout |
|---|---|---|---|
| **k = 1 ULP de `lado`** | **0,2831** | **3,2338** | 98,3 s |
| **k = 2 ULP de `lado`** | **0,2873** | **3,1166** | 73,1 s |
| *vara: control de redondeo (INFRA §4.1, n=12)* | *0,321* | *3,11* | *190,4 s* |
| *vara: BH θ=0 vs original (INFRA §4.1, n=2)* | *0,312* | *2,62* | *507,8 s* |

**Cae exactamente encima de la vara**, en RMS y en máximo. La perturbación de ULP es el mismo temblor que
el reordenamiento de la suma: es un control de redondeo legítimo, y encima cuesta una fracción.
*(Y de paso reproduce el hecho de fondo: en una caja de lado 12,6, mover el último bit corre las
posiciones finales 0,28 de RMS y hasta 3,2. El layout FR no es reproducible a nivel de coordenadas, ni
con el algoritmo viejo ni con el nuevo.)*

### 1.4 Todo lo demás, idéntico al protocolo de la serie

Masa total fija 18 800, lado nominal 2000^(1/3), `Expansion` de 60 pasos, turbulencia Mach=3 semilla 42,
100 iteraciones de layout, `seed_layout=12345`, 14 sweeps del motor relacional; en Phantom
`icreate_sinks=1`, `rho_crit_cgs=1000`, `r_crit=0.600`, `h_acc=0.300`, `f_acc=0.800`, `tmax=0.500`,
`dtmax=0.001`. `editar_cosmog_in` se **importa** de `cs090_fase5b_correr.py` (congelado); el lector es
`leer_volcado_phantom.leer_dump` (congelado).

**θ = 0,5 en TODAS las réplicas** — no es el θ operativo (0,3) que eligió la validación de
infraestructura. Se usa 0,5 porque cuesta ×2,9 menos y porque lo que se mide acá es una **dispersión entre
réplicas que comparten θ**: el sesgo de θ es una constante común a las 6 y **se cancela exacto en la σ**.
Queda dicho como limitación: **no se midió si la σ misma depende de θ.**

### 1.4b Un hallazgo que salió del chequeo cruzado, y que conviene leer antes de todo lo demás

Se comparó la réplica **k=0** de `r23_I` (perturbación CERO) contra la condición inicial que la demo de
infraestructura escribió **con el mismo θ=0,5, la misma `seed_layout=12345`, el mismo grafo y el mismo
protocolo**, sólo que desde otro script y en otro momento. La expectativa razonable era "idénticas".

| comparación | resultado |
|---|---|
| ¿idénticas bit a bit? | **NO** |
| partículas exactamente iguales | **0 de 8 000** |
| RMS de posición (caja dilatada de lado ≈ 97,6) | **1,804** |
| diferencia máxima | **22,90** |
| RMS relativo al lado de la caja | **1,85 %** *(a N=2000 el control de redondeo da 0,321/12,6 = 2,5 %)* |

O sea: **a N=8000 el layout tampoco es reproducible ni consigo mismo**, y por el mismo motivo de siempre
(las 100 iteraciones de Fruchterman-Reingold amplifican caóticamente cualquier diferencia de último bit;
acá basta con que el entorno fije distinto el número de hilos de numpy y cambie el orden de una suma).
No es un defecto de este trabajo: es **la propiedad del método** que `FASE6_O3B` vio por accidente y que
`INFRA §4.1` midió a pedido. Para este experimento es una buena noticia metodológica: significa que la
réplica k=0 **no es un "original" privilegiado**, es una tirada más — las 6 réplicas son 6 tiradas
equivalentes de la misma distribución, que es exactamente lo que una σ necesita.

Y da, de paso, la primera cifra del grano a N=8000 **en posiciones**: en términos relativos al tamaño de
la caja, el temblor del layout a N=8000 (1,85 %) es **algo MENOR** que a N=2000 (2,5 %).

### 1.5 Cómo se lee el observable: al DUMP COMÚN

Comparar réplicas en dumps de tiempos distintos mezclaría ruido con evolución. Todas las réplicas de un
grafo se leen en el **mismo índice de dump** (mismo tiempo simulado), el máximo que alcanzaron todas.

---

## 2. PARTE 1 — EL GRANO A N=8000

**6 réplicas por grafo** (k = 0…5 ULPs), 2 grafos, 12 corridas. CSV crudo: `cs090_f804_grano_n8000.csv`
(una fila por réplica) y `cs090_f804_sigma_vs_tiempo.csv` (σ dump por dump).

### 2.1 El número, y la sorpresa: el grano NO es una constante — CRECE con el nº de sumideros

`r23_I` (Clase I), 6 réplicas, σ de la fracción de masa en sumideros medida al mismo dump:

| dump (t) | nº de sumideros | media | **σ entre réplicas** | rango | σ/media | σ en partículas |
|---|---|---|---|---|---|---|
| 20 (0,020) | 8 – 8 | 0,02529 | **0,00039** | 0,00088 | 1,5 % | 3,1 |
| 40 (0,040) | 8 – 8 | 0,02983 | **0,00034** | 0,00100 | 1,1 % | 2,7 |
| 60 (0,060) | 8 – 10 | 0,03596 | 0,00110 | 0,00325 | 3,1 % | 8,8 |
| 100 (0,100) | 14 – 17 | 0,05975 | 0,00081 | 0,00238 | 1,4 % | 6,5 |
| 140 (0,140) | 18 – 26 | 0,07454 | 0,00180 | 0,00538 | 2,4 % | 14,4 |
| 180 (0,180) | 26 – 36 | 0,09056 | 0,00329 | 0,00900 | 3,6 % | 26,3 |
| 200 (0,200) | 34 – 46 | 0,10246 | 0,00444 | 0,01213 | 4,3 % | 35,6 |
| 220 (0,220) | 48 – 60 | 0,11996 | 0,00369 | 0,01025 | 3,1 % | 29,5 |
| **246 (0,246)** | **54 – 78** | **0,14106** | **0,00952** | **0,02575** | **6,7 %** | **76,1** |

`r10_III` (Clase III), 6 réplicas (una murió — §2.3), al dump común 72; y las **5 supervivientes** más
adelante:

| corte | n | nº de sumideros | media | **σ** | rango |
|---|---|---|---|---|---|
| dump 72 (t=0,072), **las 6** | 6 | 9 – 13 | 0,04623 | **0,00194** | 0,00525 |
| dump 101 (t=0,101), **las 5 que siguen vivas** | 5 | 19 – 29 | 0,07330 | **0,00343** | 0,00950 |

**El grano no es un número: es una función del número de sumideros.** Ajuste log-log sobre las 13 filas
de `r23_I` con más de 8 sumideros:

> **σ ∝ (nº de sumideros)^1,24**, coeficiente de correlación **R = 0,92**

| nº de sumideros | régimen donde ocurre | σ (medida o predicha por el ajuste) | en partículas |
|---|---|---|---|
| **8** | **es TODA la corrida a N=2000** | 0,00054 *(medida: 0,00034 – 0,00039)* | 4 |
| 29 | N=4000 a tmax | 0,00265 | 21 |
| 66 | N=8000 a t≈0,246 — **hasta donde se llegó** | 0,00733 *(medida: 0,00952)* | 59 |
| **124** | **N=8000 a tmax=0,500** (la demo midió 121-127) | **0,01602** *(extrapolada)* | **128** |

### 2.2 Contra las varas que importan

| vara | valor | ¿cómo queda contra el grano de N=8000? |
|---|---|---|
| **1 partícula a N=8000** (cuantización) | 0,000125 | El grano medido (0,0095) es **76 partículas**. La cuantización **no es el límite**: el límite es el caos. |
| **1 partícula a N=2000** | 0,0005 | — |
| **σ empírica a N=2000** (derivada de O3-B: 12 re-corridas del mismo grafo, sd(Δ)=0,00122 ⇒ σ=sd(Δ)/√2) | **0,00086** *(1,7 partículas)* | El grano a N=8000, **al mismo número de sumideros (8)**, es 0,00034-0,00054: **igual o mejor**. Pero a t=0,246 ya es **0,0095 = 11× peor**. |
| **F7-03, apiñamiento** (`solap`−`disj`) | **+0,01433** | σ=0,0095 ⇒ el efecto es **1,5 σ**. A t=0,246: **replicable** (§2.4). Extrapolado a tmax (σ≈0,0160): **el grano SUPERA al efecto**. |
| **F7-04, residual** (`soporte`−`antisoporte`) | **+0,00163** | σ=0,0095 ⇒ el efecto es **0,17 σ**. **No es medible a N=8000** por ningún camino razonable (§2.4). |

### 2.3 Dos réplicas de doce terminaron en otro estado — y sólo se diferencian en el último bit

| réplica | qué pasó |
|---|---|
| `r10_III_rep03` (k=3 ULP) | **`FATAL ERROR! evolve: Conservation errors too large to continue simulation`**, en t=0,072. Murió. |
| `r10_III_rep05` (k=5 ULP) | Viva pero **congelada**: el paso de tiempo se partió en **4096 sub-pasos** (`dt = 2,44E-07` contra `dtmax = 1E-03`). En t=0,101 avanzaba ~1 200 cpu-s por dump. |

**2 de 12 corridas que difieren SÓLO en una parte en 10¹⁶ terminaron en un estado cualitativamente
distinto** — una abortada por violación de conservación, otra con colapso total del paso de tiempo. Eso
no es dispersión del observable: es **dispersión del desenlace de la corrida**. A N=2000 y N=4000 esto no
aparece en ningún informe del proyecto (las 6 y 8 corridas de referencia llegan todas a `cosmog_00500`).

### 2.4 Qué tamaño de efecto se podría detectar, en números

Diseño pareado con n pares (dos brazos por par, cada uno una corrida ⇒ σ de la diferencia = σ·√2),
α=0,05 a dos colas, potencia 80 %:

| σ usada | n = 6 pares | n = 12 | n = 20 | n = 40 |
|---|---|---|---|---|
| **0,00499** (t≈0,23, ~56 sumideros) | 0,00815 | 0,00576 | 0,00447 | 0,00316 |
| **0,00952** (t=0,246, ~66 sumideros) | 0,01555 | 0,01100 | 0,00852 | 0,00602 |

Y al revés — cuántos pares harían falta:

| efecto a detectar | con σ=0,00499 | con σ=0,00952 | con σ≈0,0160 (extrapolada a tmax) |
|---|---|---|---|
| **F7-03 (+0,01433)** | **2 pares** | **7 pares** | ~20 pares |
| **F7-04 (+0,00163)** | 156 pares | **567 pares** | ~1 600 pares |

Con **>780 s por corrida y sin llegar ni a la mitad del tiempo** (§3.4), 567 pares son 1 134 corridas.

---

## 3. PARTE 2 — el costo real de una corrida a N=8000

### 3.1 El cronómetro que ya estaba grabado

La demo de infraestructura dejó un número agregado (1504 s **sin llegar a tmax**) y de ahí extrapoló
"1-2 h por corrida". Un número agregado no distingue entre *"es caro, linealmente"* y *"el costo por
unidad de tiempo simulado está DIVERGIENDO"* — y esa diferencia decide si la extrapolación significa algo.

Phantom escribe **un volcado por cada 0,001 de tiempo simulado** (`cosmog_00000` … `cosmog_00500`). La
**fecha de modificación de cada volcado** es entonces un cronómetro gratuito, ya grabado en disco, que da
el **tiempo de pared por unidad de tiempo simulado**, dump por dump. `cs090_f804_costo_por_dump.py` lo
extrae. *(Mide pared, no CPU: incluye la contención. Sirve para comparar la FORMA de la curva entre
resoluciones — que es machine-independiente — y como cota superior de costo.)*

### 3.2 Corridas COMPLETAS: la referencia de N=2000 y N=4000

| resolución | corridas medidas | ¿llegan a tmax=0,500? | pared total de la corrida completa |
|---|---|---|---|
| **N=2000** (Fase V-B) | 6 | **6/6** | **13 – 16 s** |
| **N=4000** (O3-A) | 8 | **8/8** | **18 – 57 s** |
| **N=8000** (demo INFRA) | 2 | **0/2** | 351 s a t=0,327 · 1230 s a t=0,259, y después el tope |

*(A N=2000 y N=4000 los volcados intermedios fueron borrados para liberar disco; quedan `cosmog_00000` y
`cosmog_00500`, que es justo lo que hace falta para el total.)*

### 3.3 La curva, que es lo que importa: el costo por dump EXPLOTA

Segundos de pared por dump, por tramos, en las dos corridas de N=8000 de la demo:

| tramo de dumps (t simulado) | `r23_I` (Clase I) | `r10_III` (Clase III) |
|---|---|---|
| 0 – 50 | 0,11 | 0,12 |
| 50 – 100 | 0,13 | 0,15 |
| 100 – 150 | 0,35 | 0,39 |
| 150 – 200 | 0,44 | 1,02 |
| 200 – 250 | 0,65 | 2,26 |
| 250 – 300 | 0,68 | **113,2** |
| 300 – 327 / 250-259 | **8,68** | **240** (último dump) |
| **dump siguiente, nunca terminado** | **> 1 149 s** | **> 270 s** |

Los últimos 25 dumps de `r23_I`, uno por uno: `…2, 2, 5, 1, 0, 0, 0, 0, 1, 1, 5, 1, 2, 5, 2, 5, 6, 2, 15,
10, 22, 32, 34, 36, 42`. Y el `run.log` termina así:

```
> step 1 / 4 t = 0.3272500 dt = 2.50E-04 moved  5 in  185 cpu-s <
> step 3 / 8 t = 0.3273750 dt = 1.25E-04 moved  2 in 1367 cpu-s <
> step 4 / 8 t = 0.3275000 dt = 1.25E-04 moved 42 in 1730 cpu-s <
```

El paso de tiempo ya se partió a 1/8 de `dtmax` y mover **42 partículas** cuesta 1730 cpu-s. Esto **no
es "caro"**: es **colapso del paso de tiempo**. La causa es la misma que ya estaba anotada: a N=8000 se
forman **121-127 sumideros** (contra 8 a N=2000 y ~29 a N=4000), y con `h_acc=0,300` fijo, muchos
sumideros cerca imponen tiempos dinámicos cada vez más cortos.

**El costo por dump se duplica cada ~6 dumps** en el tramo final de `r23_I` (de ~2 s/dump en el dump 300
a ~42 s/dump en el 327). Extrapolar esa pendiente de 327 a 500 da **~29 duplicaciones**: un número sin
sentido físico, pero cuyo mensaje sí lo tiene — **la corrida no se termina "esperando un poco más".**

### 3.4 Las 12 corridas nuevas de esta tarea: ninguna llegó a tmax

Se lanzaron con tope de pared de 4 200 s y se detuvieron al agotarse el presupuesto de la tarea. Estado
al momento del corte (12 corridas en paralelo, máquina compartida con el resto de los agentes):

| corrida | último dump (t de 0,500) | pared | s/dump en los últimos 10 |
|---|---|---|---|
| `r23_I_rep00` … `rep05` | **252 · 265 · 246 · 275 · 256 · 261** (t = 0,25 – 0,28) | 768 – 789 s | 7,9 · 11,6 · 8,5 · 9,8 · 7,9 · 16,5 |
| `r10_III_rep00,01,02,04` | 157 · 138 · 160 · 144 | 459 – 474 s | 13,6 · 20,5 · 3,9 · 5,9 |
| `r10_III_rep03` | **72** — murió (conservación) | 113 s | 3,4 |
| `r10_III_rep05` | **101** — congelada (`dt=2,44E-07`) | 161 s | 3,4 |

**Ninguna de las 14 corridas de N=8000 que existen en el proyecto (12 de esta tarea + 2 de la demo) llegó
a `tmax=0,500`.** La más lejana llegó a **t=0,275**, o sea **el 55 %**, en 780 s de pared, con el costo
por dump ya en 10 s y subiendo. Y la mitad que falta es **la cara**: es donde el nº de sumideros pasa de
~66 a ~124.

**Estimación honesta del costo de una corrida completa a N=8000, y por qué no se da un número:** con el
costo por dump duplicándose cada ~6 dumps en el tramo final, cualquier extrapolación a los 225 dumps que
faltan da cifras sin sentido (10⁴ – 10¹⁰ s). Lo que sí se puede afirmar con lo medido es que **el costo
no es "1-2 h": es una divergencia**, y que **el protocolo actual (`dtmax=0.001`, `h_acc=0.300`,
`f_acc=0.800`, sin fusión de sumideros) no termina una corrida a N=8000.** La comparación limpia:

| resolución | sumideros a tmax | corrida COMPLETA hasta tmax=0,500 |
|---|---|---|
| N=2000 | 8 | **13 – 16 s** (6/6 completas) |
| N=4000 | ~29 | **18 – 57 s** (8/8 completas) |
| **N=8000** | **121 – 127** | **0 de 14 completas** — la mejor, t=0,275 en 780 s |

---

## 4. VEREDICTO OPERATIVO (de instrumento, no de la Teoría)

La pregunta era: ¿el grano a N=8000 es menor que 0,0143? ¿Y menor que 0,0016?

> **(c) con matiz de (b): con el protocolo actual, una batería a N=8000 es INVIABLE tal como está
> planteada — y no principalmente por el grano, sino porque el endpoint de la serie (fracción de masa a
> `tmax=0.500`) NO EXISTE a esa resolución: 0 de 14 corridas llegan.** Si se acepta cambiar el endpoint a
> un tiempo intermedio alcanzable (t≈0,25), entonces sí es viable, **pero sólo para efectos grandes.**

Punto por punto:

1. **¿El grano es menor que 0,0143 (F7-03)?** **En t=0,246, sí: σ = 0,00952 < 0,0143**, y F7-03 se
   replicaría con **7 pares**. Pero **extrapolado a tmax=0,500** (σ ≈ 0,0160 por el ajuste
   σ ∝ n_sumideros^1,24) **el grano SUPERA al efecto**. La respuesta depende de dónde se pare el reloj.
2. **¿El grano es menor que 0,0016 (F7-04)?** **NO, por un factor de 6.** El residual chico **no es
   medible a N=8000** — harían falta ~567 pares (1 134 corridas de >13 min cada una). Y el sentido común
   del asunto es incómodo: **subir la resolución EMPEORA la medición de ese residual**, no la mejora.
   A N=2000, con σ=0,00086 y 12 pares, el MDE es 0,00098 — **más chico que el efecto**. La resolución que
   ve el residual de F7-04 es la que ya se está usando.
3. **El grano se rompió en dos sentidos, no en uno.** Además de σ, hay **dispersión del desenlace**: 1 de
   12 corridas murió por error de conservación y otra quedó congelada con `dt=2,44E-07`, diferenciándose
   sólo en el último bit. Una batería con ~8 % de corridas que mueren o se congelan de manera dependiente
   del redondeo tiene un problema de **sesgo de supervivencia**, no sólo de ruido: las que sobreviven no
   son una muestra al azar de las que se lanzaron.
4. **Lo que NO cambió a peor.** El **layout** ya no estorba (297 s por grafo a N=8000 con θ=0,5, seis en
   paralelo), y el temblor de posiciones del layout a N=8000 es **relativamente menor** que a N=2000
   (1,85 % del lado de la caja contra 2,5 %). El problema **no está en el layout ni en el grafo: está en
   Phantom**, exactamente como anticipó `INFRA §6.3`.
5. **Y el diagnóstico de fondo, en una línea:** *el ruido no lo pone N, lo ponen los sumideros.* A igual
   número de sumideros (8), N=8000 mide **igual o mejor** que N=2000 (σ = 0,00034-0,00054 contra
   0,00086). Lo que arruina la medición es que a N=8000 nacen **15 veces más sumideros**, y σ crece como
   la potencia 1,24 de ese número.

### 4.1 Qué habría que cambiar para que N=8000 sea viable (lo que se puede decir con lo medido)

Todo lo que sigue es **hipótesis operativa a testear, no recomendación cerrada** — ninguna se probó acá:

- **Atacar el número de sumideros, no el costo por sumidero.** Es la única palanca que el ajuste
  σ ∝ n^1,24 señala directamente. Candidatos del protocolo: **fusión de sumideros** (hoy no hay),
  `h_acc` **escalado con la resolución** en vez de fijo en 0,300, o `rho_crit_cgs` escalado.
  Ojo: cambiar cualquiera de ellos **rompe la comparabilidad con toda la serie de N=2000**.
- **Mover el endpoint a un tiempo alcanzable** (t≈0,25) y re-medir la serie de N=2000 y N=4000 en el
  MISMO t. Es lo más barato y lo único que hace comparable las tres resoluciones — pero **cambia el
  observable de todo el proyecto**, así que es decisión del director, no de esta tarea.
- **Paso de tiempo individual / `tolv`, `tolh`**: el colapso observado (`dt` partido en 4096 sub-pasos)
  es el síntoma clásico. No se auditó si el bloque actual ya usa timestepping individual bien afinado.
- **Lo que NO hay que hacer es "más réplicas".** Con σ=0,0095 el residual de F7-04 pide 567 pares; no es
  un problema de estadística, es un problema de instrumento.

### 4.2 Consecuencias directas para el plan de Fase VIII

- **OLA 3 / F8-06 (F7-03 a N=4000 y N=8000):** a **N=4000 no hay obstáculo** (8/8 corridas completas,
  18-57 s, ~29 sumideros ⇒ σ≈0,0027 ⇒ F7-03 = 5,3σ). A **N=8000, con el protocolo actual, no se puede
  hacer con el endpoint de la serie.**
- **Rama "Regularidad Cosmogénesis" (R1-R3, ≥4 resoluciones):** el tercer punto de escala **no puede ser
  N=8000 a tmax** hoy. Si hace falta un tercer punto ya mismo, el camino barato es N=3000 o N=6000 con
  el mismo endpoint, no N=8000.

---

## 5. Limitaciones, dichas explícitas

1. **n = 6 réplicas por grafo.** El error estándar de una σ con n=6 es ~32 %. Los σ de la tabla 2.1 son
   ruidosos entre sí (0,00081 en el dump 100 y 0,00323 en el 120): **la tendencia es sólida (R=0,92 en
   13 puntos), cada punto suelto no lo es.**
2. **θ = 0,5, no el θ operativo 0,3.** Todas las réplicas lo comparten, así que el sesgo de θ se cancela
   exacto en la σ; pero **no se midió si la σ misma depende de θ**.
3. **σ ∝ n_sumideros^1,24 es un ajuste sobre UN grafo** (`r23_I`, 13 puntos). `r10_III` sólo llegó a 29
   sumideros y es consistente (σ=0,00343 a 19-29 sumideros contra 0,00265 predicho), pero es un chequeo,
   no una segunda medición independiente del exponente.
4. **La σ extrapolada a tmax (0,0160) es una extrapolación**, con todo lo que eso implica. Lo medido
   llega a 66-78 sumideros; los 121-127 de tmax no se alcanzaron.
5. **Los tiempos de pared son bajo contención** (la máquina estuvo compartida con el resto de la sesión;
   en un momento la memoria entró en swap y hubo que bajar la concurrencia de 16 a 6 layouts). Son cotas
   superiores. La FORMA de las curvas (costo por dump, σ vs sumideros) no depende de eso.
6. **No se testeó ninguna de las palancas de §4.1.** Este trabajo mide el instrumento; no lo arregla.

---

## 6. Reproducir

```bash
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis
./venv/bin/python cs090_f804_grano_n8000.py sanity          # valida la perturbación a N=2000
for r in 0 1 2 3 4 5; do echo "r23_I $r"; echo "r10_III $r"; done > /tmp/tareas.txt
xargs -P 6 -L 1 ./venv/bin/python cs090_f804_grano_n8000.py ic < /tmp/tareas.txt   # 6 en paralelo: más
                                                                 # concurrencia entra en swap (~700 MB c/u)
./cs090_f804_lanzador.sh 4200        # encadena Phantom detrás de cada layout
./venv/bin/python cs090_f804_grano_n8000.py curva     # σ vs tiempo simulado
./venv/bin/python cs090_f804_grano_n8000.py medir     # CSV crudo al dump común
./venv/bin/python cs090_f804_costo_por_dump.py        # la curva de costo (parte 2)
```

Los grafos quedaron guardados en `cs090_f804_grafos/`; las 12 corridas, en
`/Users/alexis/phantom_cs073/f804_grano_n8000/N8000/`.
