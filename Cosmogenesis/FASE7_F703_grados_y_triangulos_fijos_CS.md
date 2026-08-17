# FASE VII · F7-03 — Grados **y** número de triángulos fijos: ¿importa *cómo están organizados*?

**Fecha:** 12-ago-2026 · **Ejecuta:** CC (Claude) · **Tarea:** F7-03 de la Fase VII, propuesta por GPT-5.6 Sol
**Antecedente directo:** `FASE7_F702_escalera_clustering_CS.md` (el clustering quedó establecido como palanca causal)
**Vara de resolución:** `FASE7_F704_cortar_bien_vs_azar_CS.md` (1 partícula = 0.0005 de fracción de masa)
**Phantom:** autorizado por Alexis para esta línea · **Diámetro:** medición oficial vigente (`cs090_diam_corregido.diam_gigante`)

> No se declara cierre ni veredicto. Ningún script congelado fue modificado (todos sólo se importan).
> No se hicieron commits de git.

---

## 0. En simple, con analogía

F7-02 dejó probado que los "triangulitos" son una palanca: con la misma cantidad de nudos y **exactamente
el mismo número de alambres en cada nudo**, agregar triangulitos hace que la maqueta junte más arena
(12/12 maquetas, +37%).

Queda una pregunta: ¿lo que importa es **cuántos** triangulitos hay, o **dónde están puestos**?

Esta tarea contesta eso. Se fabrican cinco versiones de la misma maqueta con:
- el mismo número de nudos,
- el mismo número total de alambre,
- **el mismo número de alambres en cada nudo, uno por uno**, y
- **exactamente el mismo número de triangulitos**,

y lo único distinto es **cómo se repartieron esos triangulitos**: todos amontonados en unos pocos
barrios; repartidos parejo por todo el pueblo; pegados unos a otros compartiendo varilla; o separados,
sin tocarse entre sí.

Si la arena responde igual en las cinco, entonces "clustering" era el nombre correcto: lo que importa es
**cuántos** triángulos hay. Si responde distinto, entonces el clustering era el nombre grueso de algo más
fino, y hay que buscar qué.

---

## 1. Qué se hizo, con qué archivos

| Archivo nuevo | Qué hace |
|---|---|
| `cs090_fase7_f703_organizacion.py` | Construye el piso común, corre los cinco brazos de organización, iguala el nº de triángulos, verifica grados nodo por nodo, mide estructura + organización + pendiente corregida, escribe las condiciones iniciales |
| `cs090_fase7_f703_correr.py` | Corre Phantom (mismo protocolo exacto de toda la línea) |
| `cs090_fase7_f703_analizar.py` | Verificación cruzada contra `meta_regla.json`, extrae métricas, estadística pareada, correlaciones y parciales, PNG |

| CSV / PNG de salida | Contenido |
|---|---|
| `cs090_fase7_f703_estructura_shard{0..3}.csv` | estructura y organización medidas de cada brazo (crudo) |
| `cs090_fase7_f703_phantom_crudo.csv` | **una fila por corrida de Phantom** (CSV crudo pedido) |
| `cs090_fase7_f703_por_grafo.csv` | una fila por grafo base con sus cinco brazos |
| `cs090_fase7_f703_estadistica.csv` | todas las pruebas |
| `cs090_fase7_f703_correlaciones.csv` | masa contra cada medida de organización |
| `cs090_fase7_f703_parciales.csv` | la mejor medida descontando cada covariable arrastrada |
| `cs090_fase7_f703_organizacion.png` | el resultado dibujado |

Batería de Phantom en `/Users/alexis/phantom_cs073/bateria_fase7_f703_organizacion/`, carpetas
`<rule_id>_s<seed>_f703_<brazo>` — **prefijo `f703` jamás usado antes**, con el **seed dentro del
nombre** (hay reglas distintas con el mismo `rule_id` en lotes distintos: bug documentado en
`FASE6_O3B` §2.1).

Scripts sólo importados, nunca tocados: `cs090_fase7_f702_escalera.py` (motor de swaps de F7-02),
`cs090_fase6_o3b_rewiring.py`, `cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_analizar.py`,
`cs090_fase7_f702_analizar.py`, `cs090_fase5_generador/motor/clasificador.py`, `cs090_diam_corregido.py`,
`cs080_renormalizacion.py`.

---

## 2. El diseño

**12 grafos base** — exactamente los mismos de F7-02 y O3-B (se llama a
`cs090_fase7_f702_escalera.seleccionar_grafos`, la misma función, no una copia): las 3 reglas de mayor
pendiente corregida dentro de cada uno de los 4 lotes de `seed_base`.

Para cada grafo base:

1. **Piso común.** Se le destruyen los triángulos con el `bajar_clustering` de F7-02 (swaps de
   Maslov-Sneppen que sólo aceptan movimientos que no aumentan triángulos). Los cinco brazos parten
   **del mismo piso**, así que comparten ancestro y no sólo secuencia de grados.
2. **Cinco brazos.** Desde ese piso se vuelven a construir triángulos con el mismo swap dirigido de
   F7-02 —`(x-p),(y-q) → (x-y),(p-q)`, que **conserva el grado de los cuatro nodos**— cambiando sólo el
   criterio de aceptación:

   | brazo | criterio de aceptación (además de ganar triángulos) | qué organización produce |
   |---|---|---|
   | `libre` | ninguno — es literalmente el criterio de F7-02 | la organización "natural" del proceso (referencia) |
   | `conc` | el ápice se sortea de una bolsa donde cada nodo pesa por triángulos ya cerrados, y el trío nuevo debe tocar un nodo que ya está en un triángulo | **concentrada**: los triángulos crecen por barrios |
   | `disp` | ninguno de los tres vértices puede superar un cupo de triángulos por nodo (arranca en 1 y sube de a uno sólo al atascarse) | **dispersa**: repartidos parejo, llenado por capas |
   | `solap` | el triángulo nuevo debe **compartir una arista** con un triángulo ya existente | **solapada**: "libros" de triángulos sobre la misma arista |
   | `disj` | cada arista tocada debe quedar en **exactamente un** triángulo | **no solapada**: empaquetamiento disjunto en aristas |

3. **Igualación del número de triángulos.** De cada brazo se guarda la **lista de swaps aceptados** (no
   fotos del grafo), así que el grafo en cualquier punto del recorrido se reconstruye reaplicando los
   primeros *k* swaps sobre el piso. Cada brazo se corre hasta su techo, con la **misma restricción de
   conectividad de F7-02** (no se admite un estado que rompa más del 3% de la componente gigante del
   original). Después se toma **T\* = el mínimo de los cinco techos** y se rebobina cada brazo al swap
   cuyo número de triángulos queda más cerca de T\*.

4. **Todo lo demás, idéntico.** Mismo N=2000, misma masa total fija 18800 (9.4 por partícula), mismo
   lado de caja `2000^(1/3)`, mismo `layout_resortes` con `seed_layout=12345`, misma dilatación
   `Expansion` de 60 pasos, misma turbulencia Mach=3 seed=42, `icreate_sinks=1`, `rho_crit_cgs=1000`,
   `r_crit=0.6`, `h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`.

**Un hecho matemático que el diseño regala gratis:** con la secuencia de grados fija, el número de
tríadas conectadas (`Σ d(d-1)/2`) es el mismo en todos los brazos; y como la **transitividad global** es
`3·triángulos / tríadas`, fijar grados *y* triángulos **fija la transitividad global exactamente**. Se
verificó numéricamente: la transitividad coincide dígito a dígito entre los cinco brazos de cada grafo.
Lo que sí puede variar (y varía) es el **clustering local medio** de Watts-Strogatz, que promedia por
nodo y por lo tanto sí depende de en cuántos nodos distintos están repartidos los triángulos.

---

## 3. Verificaciones hechas antes de mirar ningún resultado

- **Grados idénticos, nodo por nodo:** `np.array_equal(grados_original, grados_brazo)` sobre los 2000
  nodos, con `assert` que aborta el grafo si falla — nunca se asume. Los 60 brazos pasaron.
- **Mismo nº de aristas** y **sin bucles i-i** en cada brazo (`assert`).
- **Dos conteos independientes de triángulos** por brazo (el de `O3B.clustering` y el de la enumeración
  explícita que usa el módulo de organización) deben coincidir (`assert`).
- **Verificación cruzada contra `meta_regla.json`** de cada carpeta, antes de que ninguna corrida entre
  en la estadística: tarea declarada, que el brazo y el `(rule_id, seed)` del meta coincidan con el
  nombre de la carpeta, que la carpeta declarada dentro del meta sea la carpeta donde está el meta, que
  el meta declare `grados_identicos_al_original = true`, que los cinco brazos de un grafo tengan el
  mismo nº de aristas, la misma `seed_layout` **y el mismo nº de triángulos**, y que la corrida haya
  arrancado con las 2000 partículas de gas (chequeo anti-IC-truncado).
- **La unión con la estructura es por `(rule_id, seed, brazo)`**, nunca por `rule_id` solo.
- Se verificó a mano que los cinco archivos de condición inicial de un mismo grafo tienen **la misma
  masa por partícula (9.4) y el mismo N (2000)** pero **md5 distinto** — es decir, el layout realmente
  cambió y no se corrió cinco veces lo mismo.

---

## 4. Qué tan iguales quedaron los grafos (el control del diseño)

**60 corridas de Phantom** (12 grafos × 5 brazos), todas con `exit_run=0`, dump final `cosmog_00500` y
2000 partículas de gas iniciales.

| control | resultado |
|---|---|
| secuencia de grados, nodo por nodo (`np.array_equal`) | **idéntica en los 60 brazos** |
| nº de aristas | **idéntico** dentro de cada grafo |
| **nº de triángulos** | **idéntico en 11 de 12 grafos**; en `batch4-r36` el brazo `libre` quedó en 89 y los otros cuatro en 90 → **diferencia máxima conseguida = 1 triángulo** (media 0.083) |
| transitividad global | **idéntica dígito a dígito** entre los cinco brazos (consecuencia matemática de fijar grados + triángulos) |
| masa por partícula / N | 9.4 / 2000 en los 60; `md5` de la condición inicial distinto en los cinco brazos de cada grafo |

El T\* logrado por grafo va de **57 a 1029 triángulos** (mediana 275). Lo limita casi siempre el brazo
`solap`: apilar triángulos sobre la misma arista se agota rápido, porque con grados ≤8 una arista sólo
puede tener tantos triángulos como vecinos comunes admitan sus dos extremos. **Ese techo estructural es
un hallazgo en sí:** con la secuencia de grados clavada, las organizaciones "apretadas" no pueden
alojar tantos triángulos como las "sueltas".

---

## 5. Qué tan distintas quedaron las cinco organizaciones

Medias sobre los 12 grafos (mismo N, mismas aristas, mismos grados, **mismos 391.3 triángulos de
media**):

| medida de organización | `disj` | `disp` | `libre` | `conc` | `solap` |
|---|---|---|---|---|---|
| **aristas en >1 triángulo** (solapamiento) | **0.000** | 0.027 | 0.061 | 0.293 | **0.853** |
| triángulos por arista con triángulo | 1.000 | 1.027 | 1.063 | 1.350 | **2.558** |
| aristas que están en algún triángulo | 0.331 | 0.314 | 0.300 | 0.239 | **0.124** |
| nodos que tocan algún triángulo | 0.451 | 0.466 | 0.418 | 0.290 | **0.104** |
| **Gini de triángulos por nodo** (concentración) | 0.607 | 0.571 | 0.653 | 0.805 | **0.926** |
| máximo de triángulos en un solo nodo | 2.3 | 2.1 | 3.6 | 7.0 | **10.2** |
| nº de cúmulos de triángulos | 148 | 217 | 163 | 93 | **28** |
| modularidad de la partición inducida | 0.205 | 0.283 | 0.291 | 0.234 | 0.123 |
| distancia media entre triángulos (saltos) | 7.35 | 7.42 | 7.46 | 8.09 | 7.92 |
| — | | | | | |
| clustering local medio | 0.1375 | 0.1424 | 0.1335 | 0.1109 | **0.0767** |
| **transitividad global** | **0.1020** | **0.1020** | **0.1020** | **0.1020** | **0.1020** |
| componente gigante | 1880 | 1880 | 1881 | 1882 | 1843 |
| pendiente corregida | 0.752 | 0.761 | 0.752 | 0.750 | 0.841 |

**El contraste logrado es grande**: el solapamiento de aristas recorre **0.000 → 0.853** (rango medio
dentro de cada grafo: **0.853**), el Gini de concentración 0.57 → 0.93, el nº de cúmulos 217 → 28, y el
clustering local medio recorre un rango medio de **0.066** dentro de cada grafo (de 0.013 en el grafo
más chico a 0.163 en el más grande). Para calibrar: el rango de clustering de F7-04 fue 0.0089 —
**este experimento mueve una palanca 7 veces más larga que aquél**, aunque 6 veces más corta que la de
F7-02 (0.418).

---

## 6. El resultado: **la masa SÍ cambia según cómo estén organizados los triángulos**

Fracción de masa en sumideros, un grafo por fila. Dentro de cada fila, **todo es idéntico salvo dónde
están puestos los triángulos**:

| regla | lote | T\* | `libre` | `conc` | `disp` | `solap` | `disj` | rango (partículas) |
|---|---|---|---|---|---|---|---|---|
| batch3-r0 | 471828 | 57 | 0.1485 | 0.1490 | 0.1500 | **0.1540** | 0.1455 | 17 |
| batch3-r111 | 471828 | 158 | 0.1170 | 0.1210 | 0.1195 | **0.1280** | 0.1150 | 26 |
| batch3-r60 | 471828 | 187 | 0.1140 | 0.1185 | 0.1145 | **0.1285** | 0.1190 | 29 |
| batch4-r10 | 571828 | 280 | 0.1285 | 0.1330 | 0.1250 | **0.1340** | 0.1310 | 18 |
| batch4-r36 | 571828 | 89 | 0.1500 | **0.1510** | 0.1450 | 0.1505 | 0.1455 | 12 |
| batch4-r62 | 571828 | 264 | 0.1060 | 0.1100 | 0.1070 | **0.1195** | 0.1060 | 27 |
| r14 | 271828 | 237 | 0.1035 | 0.1065 | 0.1030 | **0.1125** | 0.1010 | 23 |
| r17 | 271828 | 286 | 0.1190 | 0.1235 | 0.1175 | **0.1285** | 0.1140 | 29 |
| r19 | 271828 | 788 | 0.1045 | 0.1070 | 0.1115 | **0.1290** | 0.1085 | 49 |
| r20 | 371828 | 468 | 0.0940 | 0.0960 | 0.0985 | **0.1075** | 0.0910 | 33 |
| r28 | 371828 | 1029 | 0.1000 | 0.1065 | 0.0950 | **0.1260** | 0.1020 | 62 |
| r39 | 371828 | 852 | 0.0855 | 0.0925 | 0.0880 | **0.1190** | 0.0865 | 67 |
| **media** | | 391 | 0.1142 | 0.1179 | 0.1145 | **0.1281** | 0.1138 | **32.7** |

### 6.1 — Las pruebas

| prueba (n=12 grafos, diseño pareado) | Δ medio | en partículas | signos | Wilcoxon |
|---|---|---|---|---|
| **Friedman** (¿algún brazo difiere?) | χ²=31.5 | — | — | **p = 2.4e-06** |
| **`solap` − `disj`** (eje SOLAPAMIENTO) | **+0.01433 (+13.8%)** | **+28.7** | **12/12** | **p = 4.9e-04** |
| **`solap` − `libre`** | +0.01387 (+13.6%) | +27.7 | **12/12** | **p = 4.9e-04** |
| **`conc` − `libre`** (eje CONCENTRACIÓN) | +0.00367 (+3.5%) | +7.3 | **12/12** | **p = 4.9e-04** |
| `conc` − `disp` | +0.00333 | +6.7 | 9/12 | p = 0.028 |
| `disp` − `libre` | +0.00033 | +0.7 | 7/12 | p = 0.83 |
| `disj` − `libre` | −0.00046 | −0.9 | 5/12 | p = 0.60 |

**El rango dentro de cada grafo es de 32.7 partículas en promedio** (mediana 28, mínimo 12, máximo 67).

### 6.2 — Contra el grano del instrumento

`FASE7_F704` midió que 1 partícula = 0.0005 de fracción de masa, y que los efectos de ~3 partículas que
perseguía allí estaban en el filo de la resolución. Acá:

| | Δ observado | en partículas | ¿supera el grano? |
|---|---|---|---|
| F7-04 (`soporte`−`antisoporte`) | +0.00163 | +3.3 | apenas, en el filo |
| **F7-03 (`solap`−`disj`)** | **+0.01433** | **+28.7** | **sí, 29 veces el grano** |
| F7-02 (e4−e0, referencia) | +0.03937 | +78.7 | sí |

**El contraste supera el grano por un factor ~29.** Esto no es un efecto marginal: es del mismo orden que
la escalera completa de F7-02 (36% de aquél), pero **conseguido sin agregar un solo triángulo**.

### 6.3 — El efecto crece con el número de triángulos disponibles

La diferencia `solap`−`disj` correlaciona fuertemente con T\*: **Spearman ρ = +0.818, p = 0.0011** (de
+12 partículas en el grafo con 89 triángulos a +67 en el de 1029). Consistente con que lo que se está
moviendo es *cómo se reparte una cantidad dada de triángulos*: cuanto más material hay para repartir,
más importa dónde se pone.

### 6.4 — Observables secundarios: el mismo patrón de siempre

| observable | `solap` | `disj` | pareado |
|---|---|---|---|
| **κ_V agregado** | 0.954 | 0.747 | 11/12, **p = 9.8e-04** |
| **nº de sumideros** | 8.08 | 8.08 | 1/12, p = 1.0 (**no cambia**) |
| **t del primer sumidero** | 0.0314 | 0.0361 | baja en 9/12, **p = 0.0073** |

**No se forman más grumos: cada grumo come más y empieza antes.** Es exactamente el patrón de O3-B y de
F7-02, ahora con el número de triángulos clavado.

---

## 7. Entonces, ¿qué propiedad es la que manda?

Spearman de la masa contra cada medida, **centrando cada valor en la media de su propio grafo** (es
decir, mirando sólo la variación *entre brazos*, que es lo que la intervención movió). n=60.

| medida | ρ | p |
|---|---|---|
| **aristas que están en algún triángulo** (soporte en aristas) | **−0.781** | 1.7e-13 |
| **triángulos por arista con triángulo** (densidad local) | **+0.776** | 3.2e-13 |
| **clustering local medio** | **−0.770** | 6.7e-13 |
| nodos que tocan algún triángulo (soporte en nodos) | −0.765 | 1.1e-12 |
| **Gini de triángulos por nodo** (concentración) | +0.736 | 2.1e-11 |
| máximo de triángulos en un nodo | +0.712 | 1.8e-10 |
| **aristas en >1 triángulo** (solapamiento) | +0.696 | 6.9e-10 |
| nº de cúmulos de triángulos | −0.589 | 7.2e-07 |
| modularidad de la partición inducida | −0.576 | 1.5e-06 |
| distancia media entre triángulos | +0.511 | 3.0e-05 |
| componente gigante | −0.494 | 6.0e-05 |
| asortatividad de grados | −0.401 | 0.0015 |
| pendiente corregida | +0.326 | 0.011 |
| fracción de triángulos en el cúmulo mayor | +0.185 | 0.16 |
| aristas compartidas con el original | +0.176 | 0.18 |
| transitividad global | — | fija por diseño, sin variación |

Las siete primeras son **la misma cosa medida de siete maneras**: *cuán apretado está el soporte de los
triángulos*. Entre `frac_aristas_en_triangulo` y `clustering_local` la correlación es ρ=0.981 — no se
pueden separar con estos datos, y por eso la parcial entre ellas cae a −0.214 (p=0.10). Con las demás
covariables el efecto sobrevive casi intacto:

| parcial de `frac_aristas_en_triangulo` con la masa, descontando… | ρ parcial | p |
|---|---|---|
| componente gigante | −0.706 | 3.0e-10 |
| nº de componentes | −0.742 | 1.1e-11 |
| asortatividad | −0.732 | 3.0e-11 |
| pendiente corregida | −0.755 | 3.2e-12 |
| aristas compartidas con el original | −0.778 | 2.6e-13 |
| solapamiento de aristas (`frac_aristas_multi_tri`) | −0.508 | 3.4e-05 |
| Gini de concentración | −0.400 | 0.0016 |

### 7.1 — El punto más incómodo, y por eso el más importante

**El clustering local medio correlaciona con la masa en el signo CONTRARIO al de F7-02.** Allá, subir C
subía la masa (ρ=+0.960). Acá, a triángulos fijos, el brazo de MENOR C (`solap`, C=0.0767) es el de MÁS
masa, y el de MAYOR C (`disp`, C=0.1424) está entre los de menos (ρ=−0.770).

La predicción cuantitativa de F7-02 falla en el signo: con su pendiente de +0.0941 de fracción de masa
por unidad de clustering, el rango de C de este experimento (0.066, con `solap` abajo) predice que
`solap` debería tener **−0.0062** de masa respecto de `disj`. Se observó **+0.0143**. No es que el efecto
sea más chico de lo predicho: **va para el otro lado**.

Traducción: **el coeficiente de clustering no es el mediador.** Era un buen indicador mientras el número
de triángulos se movía con él (F7-02), pero cuando se fija el número de triángulos, C mide otra cosa —
mide *sobre cuántos nodos distintos están repartidos* — y esa otra cosa va al revés.

### 7.2 — La lectura que unifica los dos experimentos

Las dos observaciones se juntan en una sola variable: **la densidad de triángulos por arista que los
sostiene** (`tri_por_arista_media`, ρ=+0.776 acá).

- En F7-02, subir el nº de triángulos sobre el mismo grafo **también** subía esa densidad → más masa.
- En F7-03, a nº de triángulos fijo, apilarlos sobre menos aristas sube esa densidad → más masa; y
  repartirlos sobre más aristas la baja → menos masa.

Dicho en simple: **no gana el que tiene más triangulitos repartidos por todo el pueblo, sino el que los
tiene apretados unos contra otros compartiendo varilla.** Lo que junta arena parece ser el *nudo denso*
—unas pocas zonas donde muchos triángulos se apoyan en las mismas aristas— y no el recuento total.

### 7.3 — Un control interno contra el confound de fragmentación

`solap` rompe un poco más el grafo (componente gigante 1843 contra ~1880, componentes 8.5 contra 3.7), y
la masa correlaciona negativamente con el tamaño de la gigante (ρ=−0.494). ¿Y si todo fuera
fragmentación disfrazada? Dos cosas dicen que no:

1. la parcial descontando la gigante sigue en **−0.706** (p=3e-10) y descontando el nº de componentes en
   **−0.742**;
2. **`conc` no fragmenta nada** (gigante 1881.8, incluso un pelo más grande que `libre`: 1880.8;
   componentes 3.2 contra 3.7) y **aun así gana +7.3 partículas sobre `libre` en 12/12 grafos**
   (p=4.9e-04). El eje de concentración produce efecto **sin** tocar la conectividad.

---

## 8. Contestando literalmente la pregunta de la tarea

> *Si NO cambia: los triángulos (su cantidad) son el mecanismo — el clustering captura lo que importa.*
> *Si SÍ cambia: el clustering era un correlato de algo más fino.*

**Cambia, y por mucho: 12/12 grafos, +28.7 partículas entre la organización más apretada y la más
suelta, sobre un instrumento cuyo grano es 1 partícula.** Con la cantidad de triángulos clavada al
número exacto y la transitividad global fija por construcción, el observable se mueve un 13.8%.

La medida que mejor lo sigue, de las que se midieron, es **el tamaño del soporte de los triángulos**:
sobre cuántas aristas distintas se apoyan (ρ=−0.781) o, equivalentemente, cuántos triángulos hay por
arista que los sostiene (ρ=+0.776). Detrás vienen, indistinguibles entre sí por colinealidad, la
concentración por nodo (Gini, ρ=+0.736) y el solapamiento de aristas (ρ=+0.696). Las medidas
mesoscópicas "de comunidad" —modularidad (ρ=−0.576) y distancia entre triángulos (ρ=+0.511)— siguen la
masa, pero peor: no parece que lo que importe sea *el barrio*, sino **el apretamiento local**.

---

## 9. Lo que este experimento NO puede decidir

- **`frac_aristas_en_triangulo` contra `clustering_local` no se pueden separar** (ρ=0.981 entre ellas).
  Hace falta un diseño que las mueva en direcciones distintas.
- **T\* está limitado por el brazo `solap`** (57-1029 triángulos según el grafo, contra techos de 800-1400
  de los brazos sueltos). Los contrastes se hicieron en el T\* más chico posible de cada grafo; con la
  restricción relajada el efecto podría ser mayor todavía (ya crece con T\*: ρ=+0.818).
- **El brazo `conc` estricto (exigir vecindarios ya densos, ≥3 triángulos) satura en 38 triángulos** en
  el grafo piloto (`r14`, contra un techo de 1337 del brazo libre), por el techo de
  grados: un nodo de grado d entra como mucho en d(d−1)/2 triángulos. Se relajó a "pegate a un triángulo
  que ya existe". La concentración *fuerte* de verdad no es alcanzable con estos grados, y eso limita
  cuánto se pudo empujar ese eje.
- **12 grafos, todos Clase III del mismo linaje A2-B0-C2.** No se sabe si el patrón vale fuera de ahí.
- No se probó ninguna hipótesis sobre *por qué* el apretamiento local junta más masa. Que κ_V suba y el
  primer sumidero llegue antes sugiere que el layout de resortes hace nudos más compactos donde los
  triángulos se apilan, pero eso es una conjetura, no una medición.

---

## 10. Costos

Generación de las 60 condiciones iniciales en 4 turnos paralelos: 3300-3600 s por turno (la máquina
estaba a carga media 500-700 por otros trabajos, así que cada condición inicial tardó ~180 s contra los
~71 s típicos de F7-02). Phantom: ~21 s por corrida, 60 corridas, solapadas con la generación.
Batería en disco: 1.6 GB.

---

## Archivos

**Nuevos (esta tarea):** `cs090_fase7_f703_organizacion.py`, `cs090_fase7_f703_correr.py`,
`cs090_fase7_f703_analizar.py`, `cs090_fase7_f703_estructura_shard{0..3}.csv`,
`cs090_fase7_f703_estructura_piloto{,2}.csv`, `cs090_fase7_f703_phantom_crudo.csv`,
`cs090_fase7_f703_por_grafo.csv`, `cs090_fase7_f703_estadistica.csv`,
`cs090_fase7_f703_correlaciones.csv`, `cs090_fase7_f703_parciales.csv`,
`cs090_fase7_f703_organizacion.png`, `cs090_fase7_f703_shard{0..3}.log`,
`cs090_fase7_f703_phantom_driver.log`, y la batería
`/Users/alexis/phantom_cs073/bateria_fase7_f703_organizacion/`.

**Sólo importados, nunca modificados:** `cs090_fase7_f702_escalera.py`, `cs090_fase7_f702_analizar.py`,
`cs090_fase6_o3b_rewiring.py`, `cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_analizar.py`,
`cs090_fase5_generador.py`, `cs090_fase5_motor.py`, `cs090_fase5_clasificador.py`,
`cs090_diam_corregido.py`, `cs080_renormalizacion.py`.

> Sin cierre, sin veredicto, sin commits.
