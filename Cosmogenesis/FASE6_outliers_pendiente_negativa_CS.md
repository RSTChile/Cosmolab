# Fase VI — investigación de los "casos raros": las reglas de pendiente muy negativa

**Fecha:** 11-ago-2026 · Ejecuta: CC (Claude) · Encargo de Alexis: *"investiga los casos raros"*
(los 3 puntos aislados de pendiente muy negativa documentados en
`FASE6_reanalisis_azar_continuo_CS.md` §3.2 y §3.5).

No se modificó ningún script ni CSV existente — sólo lectura + 4 scripts nuevos. Phantom SÍ se corrió
(autorizado explícitamente para esta línea). No se hicieron commits de git. **No se declara cierre ni
veredicto**: se reportan los números y la lectura final es de Alexis.

---

## 0. En simple, con analogía

Cada "maqueta de alambre" (grafo) de esta línea se resume en un número, la **pendiente**: qué tan
rápido se encoge su *diámetro* (la distancia entre los dos nodos más lejanos) cuando uno va agrupando
nodos en cajas cada vez más grandes. Pendiente alta = geometría extensa (Clase III); pendiente baja =
la cosa se disuelve (Clase I).

Tres maquetas dieron pendiente **muy negativa** (−0.94, −0.81, −0.65), que literalmente querría decir
"el diámetro CRECE al agrupar" — algo raro. Y encima dos de esas tres acumulaban mucha masa en Phantom,
rompiendo la tendencia general. Eso generó la famosa "forma de U" del reanálisis anterior.

**Lo que encontré es que no había ninguna U.** La rutina que mide el diámetro
(`_diam`, congelada desde cs055) arranca su recorrido en **el primer nodo de la lista que todavía tenga
alguna conexión** — el de índice más bajo. En estas 3 maquetas, ese nodo había quedado en un
**pedacito suelto de 2 nodos**, un par colgando aparte del cuerpo principal. Así que la rutina midió
"el diámetro" de ese par: **1**. Mientras tanto, el cuerpo principal de la maqueta tenía 1.700-1.900
nodos y un diámetro real de 22-25.

La analogía: es como medir la altura de un edificio, pero el metro se apoyó en el buzón de la vereda en
vez de en el edificio. Da 30 cm. Después, al "agrupar" (cambiar de escala), la numeración de las cajas
cambia y el metro sí cae sobre el edificio: de golpe da 17 metros. Un ajuste lineal que junta el "30 cm"
con los "17 metros" te dice que el edificio se hace más alto cuanto más lejos te parás — que es
exactamente la pendiente negativa. El edificio nunca hizo nada raro; el primer punto de la medición
estaba tomado sobre otro objeto.

Corrigiendo dónde se apoya el metro (medir siempre sobre el pedazo más grande del grafo), las 14
pendientes negativas se dan vuelta y quedan **todas positivas, entre +0.59 y +1.44** — o sea, en el
territorio de la geometría extensa, no de la disolución.

---

## 1. PASO 1 — Qué son esas 3 reglas

Script: `cs090_fase6_outliers_paso1_curvas.py`. Reconstruye el grafo final con la misma cadena exacta de
`cs090_fase5_motor.correr_regla_coarse()` (mismo `p` desde el `seed`, mismo rng `seed*5000+N`, mismo
`construir_A2`+`dinamica_B0`+`medir`, mismas cajas `cs080.cajas_bfs` con rng `seed*7000+b*31`) y vuelve
a medir las 5 escalas. **Verificación dura:** la pendiente recalculada coincide con la del CSV de origen
hasta `1e-9` en las 7 reglas medidas — la reproducción es fiel, no hay ambigüedad sobre qué se está
mirando.

### 1.1 Parámetros — ¿tienen algo obviamente distinto?

| regla | clase (CSV) | pendiente | seed | K | J | noise | meandeg | kcap | z_agg |
|---|---|---|---|---|---|---|---|---|---|
| A2-B0-C2-batch3-r100 | I | −0.9423 | 481529 | 5 | 0.569 | 0.195 | 6.61 | **4** | 2.86 |
| A2-B0-C2-batch4-r51 | I | −0.8118 | 576776 | 8 | 0.449 | 0.196 | 5.29 | **4** | 0.86 |
| A2-B0-C2-batch3-r143 | I | −0.6471 | 485700 | 4 | 0.585 | 0.132 | 6.04 | 5 | 2.03 |

Nada extremo en K, J, noise ni sim_thr_frac. Lo único que se despega es la combinación
**meandeg alto + kcap bajo** (ver §2.3): grafo inicial denso al que después se le aplica un tope duro de
grado muy apretado. Eso es exactamente la receta para dejar el grafo **fragmentado**, con muchos nodos
sueltos y pedacitos colgando — y ahí está el problema.

### 1.2 La curva de coarse-graining CRUDA (lo importante)

Los 5 puntos crudos de cada una, y de 4 reglas de referencia "normales" que sí pasaron por Phantom
(datos completos en `cs090_fase6_outliers_curvas.csv`, gráfico en `cs090_fase6_outliers_curvas.png`):

| regla | pend. | diám b=1 | b=2 | b=4 | b=8 | b=16 | forma |
|---|---|---|---|---|---|---|---|
| batch3-r100 (I) | −0.94 | **1** | 17 | 10 | 7 | 5 | **NO MONÓTONA** |
| batch4-r51 (I) | −0.81 | **1** | 15 | 9 | 6 | 5 | **NO MONÓTONA** |
| batch3-r143 (I) | −0.65 | **1** | 9 | 7 | 5 | 4 | **NO MONÓTONA** |
| r12 (I, referencia) | +0.58 | 12 | 7 | 6 | 5 | 4 | decreciente |
| batch3-r86 (I, referencia) | +0.68 | 13 | 9 | 7 | 5 | 4 | decreciente |
| batch3-r71 (III, referencia) | +0.90 | 14 | 11 | 8 | 5 | 4 | decreciente |
| batch3-r111 (III, referencia) | +0.91 | 15 | 12 | 9 | 5 | 4 | decreciente |

La respuesta a la pregunta del encargo es tajante: **la curva NO es genuinamente decreciente y monótona,
es no-monótona**, y el quiebre está **en un solo punto, el b=1**. De b=2 en adelante las 3 outliers se
comportan igual que las reglas normales (bajan de forma ordenada). Es el mismo tipo de caso que la
regla `r16` ya documentada en `FASE5_matriz_2x2_completa_CS.md`.

### 1.3 El estado del grafo — de dónde sale ese "1"

`_diam(adj, N)` (cs055, congelado) elige su nodo de arranque así:

```python
src = next((i for i in range(N) if adj[i]), 0)   # PRIMER nodo, por índice, que conserve alguna arista
```

Es un doble-BFS: mide el diámetro **de la componente conexa donde cayó ese nodo**, no del grafo entero.
Medí, escala por escala, en qué tamaño de componente cae esa medición:

| regla | b | n_cajas | diám medido | giant | nº componentes | **tamaño de la componente medida** | aislados |
|---|---|---|---|---|---|---|---|
| batch3-r100 | 1 | 2000 | **1** | 0.8445 | 29 | **2** | 242 |
| | 2 | 1298 | 17 | 0.7835 | 11 | 1017 | 260 |
| | 4 | 878 | 10 | 0.6913 | 2 | 607 | 269 |
| | 8 | 666 | 7 | 0.5946 | 1 | 396 | 270 |
| | 16 | 559 | 5 | 0.5170 | 1 | 289 | 270 |
| batch4-r51 | 1 | 2000 | **1** | 0.8905 | 23 | **2** | 159 |
| | 2 | 1244 | 15 | 0.8449 | 13 | 1051 | 169 |
| | 4 | 821 | 9 | 0.7795 | 1 | 640 | 181 |
| | 8 | 588 | 6 | 0.6922 | 1 | 407 | 181 |
| | 16 | 488 | 5 | 0.6291 | 1 | 307 | 181 |
| batch3-r143 | 1 | 2000 | **1** | 0.9540 | 8 | **2** | 77 |
| | 2 | 1196 | 9 | 0.9289 | 2 | 1111 | 83 |
| | 4 | 769 | 7 | 0.8908 | 1 | 685 | 84 |
| | 8 | 518 | 5 | 0.8378 | 1 | 434 | 84 |
| | 16 | 404 | 4 | 0.7921 | 1 | 320 | 84 |
| **r12 (referencia)** | 1 | 2000 | 12 | 0.9780 | 3 | **1956** | 40 |
| **batch3-r86 (ref.)** | 1 | 2000 | 13 | 0.9735 | 2 | **1947** | 51 |
| **batch3-r71 (ref.)** | 1 | 2000 | 14 | 0.9080 | 14 | **1816** | 155 |
| **batch3-r111 (ref.)** | 1 | 2000 | 15 | 0.9245 | 10 | **1849** | 128 |

**El patrón es limpio:** en las 3 outliers, la medición de b=1 cae sobre una componente de **2 nodos**
(un par suelto) mientras la componente gigante tiene 1.689-1.908 nodos. En las 4 referencias, cae sobre
la gigante (1.816-1.956 nodos). No es que el grafo de las outliers esté "roto": su gigante tiene entre
84% y 95% de los nodos, comparable a las referencias (91%-98%). Lo que pasa es que tienen **más
fragmentos chicos** (8-29 componentes, 77-242 nodos aislados) y basta con que **uno** de esos fragmentos
contenga al nodo de índice más bajo que aún tiene aristas para que la medición se descarrile.

### 1.4 Comprobación directa: medir el diámetro sobre la componente gigante

Script: `cs090_fase6_outliers_diagnostico_diam_gigante.py`. Vuelve a medir las 5 escalas eligiendo el
nodo de arranque **en la componente más grande** (elección determinista, independiente de cómo estén
numerados los nodos), y recalcula la pendiente. Se hizo sobre **87 reglas**: las 76 reglas distintas que
corrieron Phantom en Fase V-B + las 14 de pendiente muy negativa (3 en ambos grupos).

| regla | pend. ORIGINAL | pend. CORREGIDA | diám orig (b=1..16) | diám gigante (b=1..16) | comp. medida b=1 | gigante b=1 |
|---|---|---|---|---|---|---|
| batch4-r53 | −1.2016 | **+1.2490** | 1, 16, 10, 7, 5 | 19, 16, 10, 7, 5 | 2 | 1524 |
| batch4-r49 | −1.1006 | **+1.2519** | 1, 15, 9, 7, 5 | 21, 15, 9, 7, 5 | 2 | 1580 |
| batch4-r110 | −1.0965 | **+1.4432** | 1, 19, 12, 8, 5 | 28, 19, 12, 8, 5 | 3 | 1570 |
| batch3-r40 | −1.0008 | **+1.0608** | 1, 13, 9, 6, 5 | 18, 13, 9, 6, 5 | 2 | 1689 |
| **batch3-r100** | −0.9423 | **+1.2550** | 1, 17, 10, 7, 5 | 25, 17, 10, 7, 5 | 2 | 1689 |
| r0 | −0.8544 | **+1.0282** | 1, 13, 9, 6, 5 | 21, 13, 9, 6, 5 | 2 | 1738 |
| **batch4-r51** | −0.8118 | **+1.0807** | 1, 15, 9, 6, 5 | 22, 15, 9, 6, 5 | 2 | 1781 |
| batch4-r156 | −0.7737 | **+0.9659** | 1, 12, 8, 7, 4 | 18, 12, 8, 7, 4 | 2 | 1827 |
| batch4-r118 | −0.7340 | **+0.7733** | 1, 11, 8, 5, 5 | 15, 11, 8, 5, 5 | 2 | 1859 |
| **batch3-r143** | −0.6471 | **+0.7640** | 1, 9, 7, 5, 4 | 14, 9, 7, 5, 4 | 2 | 1908 |
| batch4-r56 | −0.6170 | **+0.8409** | 1, 11, 8, 5, 4 | 15, 11, 8, 5, 4 | 2 | 1900 |
| batch3-r15 | −0.5919 | **+0.5937** | 1, 8, 6, 5, 4 | 12, 8, 6, 5, 4 | 2 | 1970 |
| batch4-r178 | −0.5601 | **+1.0613** | 2, 13, 9, 7, 5 | 20, 13, 9, 7, 5 | 3 | 1688 |
| batch3-r45 | −0.4891 | **+1.2658** | 2, 16, 11, 6, 5 | 22, 16, 11, 6, 5 | 4 | 1635 |
| *(extra)* batch3-r95 | −0.1859 | **+1.1474** | 3, 15, 10, 6, 5 | 19, 15, 10, 6, 5 | 5 | 1609 |

**Tres números que resumen todo:**

1. **Descarrilamiento en b=1: 14 de 14** entre las de pendiente < −0.3, y **0 de 73** entre todas las
   demás (criterio: la componente donde cae `_diam` tiene menos del 10% del tamaño de la gigante). La
   regla extra de pendiente −0.186 también descarrila; la siguiente hacia arriba (+0.314) ya no. La
   frontera cae exactamente en **pendiente = 0**.
2. **Todas las pendientes negativas se dan vuelta a positivas** al corregir: de +0.59 a +1.44. Trece de
   las catorce quedan por encima del umbral 0.7 con el que se define Clase III.
3. **Los diámetros de b=2 a b=16 no cambian ni un entero** al corregir (ver la tabla: las columnas 2-5
   de "orig" y "gigante" son idénticas en las 14). Es decir: **el único punto afectado es el b=1**. Eso
   descarta que sea un problema del coarse-graining o de la dinámica; es dónde arranca la medición
   nativa.

### 1.5 Efecto sobre la correlación pendiente ↔ masa (la "forma de U")

Rehaciendo el Análisis 2 de `FASE6_reanalisis_azar_continuo_CS.md` §3.3, ahora sobre las **76 reglas
distintas** que corrieron Phantom (las 80 filas incluyen 4 reglas reutilizadas en dos pares):

| observable usado como predictor | n | Spearman rho | p | R² lineal |
|---|---|---|---|---|
| pendiente ORIGINAL (todas) | 76 | +0.4634 | 2.5×10⁻⁵ | 0.0105 |
| pendiente ORIGINAL, excluyendo los 3 outliers | 73 | +0.5976 | 2.4×10⁻⁸ | 0.5879 |
| **pendiente CORREGIDA (todas, sin excluir nada)** | 76 | **+0.6283** | **1.2×10⁻⁹** | **0.6631** |

La pendiente corregida, **sin tirar ningún dato**, es mejor predictor que la original con los 3 outliers
borrados a mano. La forma de U desaparece: los 3 puntos rojos pasan de estar aislados a la izquierda a
sentarse arriba a la derecha, sobre la misma tendencia que el resto (panel derecho de
`cs090_fase6_outliers_diam_gigante.png`).

Y hay un detalle que encaja fino: de los 3 outliers, los dos con **fracción de masa alta**
(batch3-r100 = 0.1500, percentil 96 de las 80; batch4-r51 = 0.1450, percentil 95) son justamente los que
tienen **pendiente corregida alta** (+1.255 y +1.081). El tercero (batch3-r143) tiene fracción de masa
0.1000 — que es la **mediana** exacta de las 80, no un valor alto — y su pendiente corregida también es
la más baja de las tres (+0.764). O sea: la descripción "los 3 tienen masa alta" del reanálisis anterior
era 2 de 3; el tercero siempre fue ordinario, y la pendiente corregida lo ordena bien.

---

## 2. PASO 2 — ¿Cuántas reglas así hay entre las 430?

Script: `cs090_fase6_outliers_paso2_distribucion.py` (sólo lectura). Combina los 4 CSV de origen con el
mismo criterio de selección que usó `cs090_fase6_observable_continuo.py` (del CSV "profundizar", sólo la
sección `origen=='nueva_profundizar'`, la única que trae `seed`) y verifica que los 430 `seed` sean
únicos antes de nada.

### 2.1 La distribución completa

`cs090_fase6_outliers_histograma.png`. Percentiles de las 430:

| p0 | p1 | p2.5 | p5 | p25 | p50 | p75 | p95 | p99 | p100 |
|---|---|---|---|---|---|---|---|---|---|
| −1.2016 | −0.9168 | −0.5988 | +0.3748 | +0.5437 | +0.6474 | +0.7797 | +1.0829 | +1.3167 | +1.5031 |

media +0.6419, sd 0.3368.

**No es una cola continua: es un grupo separado.** Ordenando las 430 pendientes y midiendo la separación
entre valores consecutivos, los dos huecos más grandes de todo el rango son:

- **0.5000** entre −0.1859 y +0.3142 ← el hueco más grande del rango completo
- **0.3032** entre −0.4891 y −0.1859

Para comparar, todos los demás huecos del rango son ≤ 0.101. O sea: hay un vacío de medio punto entre el
grupo negativo y el cuerpo principal de la distribución.

### 2.2 El corte elegido y por qué

**Corte: pendiente < 0.** Justificación, en este orden:

1. El hueco más grande de la distribución (0.50) está justo ahí, entre −0.186 y +0.314.
2. Es el corte que coincide exactamente con el diagnóstico mecánico: de las reglas re-medidas, las
   **15 con pendiente < 0 descarrilan la medición de diámetro en b=1** y **ninguna de las de pendiente
   ≥ 0 lo hace** (74 verificadas: las 73 no-negativas de las 87 + la de +0.314). No es un corte
   estadístico arbitrario, es la frontera de un mecanismo identificado.
3. Físicamente, pendiente < 0 significaría "el diámetro crece al agrupar nodos", que no tiene sentido
   para el coarse-graining tal como está definido — agrupar nunca puede alargar caminos.

*(El corte alternativo −0.3, que usé para elegir las candidatas de Phantom antes de tener el diagnóstico
de §1.4, da 14 en vez de 15: deja fuera sólo a `batch3-r95`, pendiente −0.1859. Se documenta la
diferencia; ninguna conclusión cambia.)*

### 2.3 Cuántas son y qué comparten

**15 de 430 (3.5%).** De ellas, sólo 3 habían pasado por Phantom — o sea, la respuesta a la pregunta
del encargo es: **son genuinamente pocas (3.5%), no es que hubiera decenas y sólo 3 cayeran en la
muestra.** Pero tampoco son "3 entre 430": son 15, y 12 nunca se habían corrido.

Un dato que sale gratis y vale la pena: **las 11 reglas de las 430 clasificadas como
"intermedio (sin clase clara)" están TODAS en este grupo de 15.** La categoría "intermedio" del
clasificador, en este lote de 430, es en su totalidad producto de este descarrilamiento de la medición.
Las otras 4 del grupo quedaron etiquetadas Clase I.

Distribución de clases en las 430: I=224, III=164, II=25, intermedio=11, IV=6.

**Parámetros compartidos.** Comparando el grupo (n=15) contra el resto (n=415):

| parámetro | grupo pend<0 | resto | ¿separa? |
|---|---|---|---|
| K | 6.20 ± 1.42 | 5.95 ± 1.23 | no |
| J | 0.577 ± 0.123 | 0.542 ± 0.140 | no |
| noise | 0.223 ± 0.068 | 0.223 ± 0.072 | no |
| sim_thr_frac | 0.250 ± 0.052 | 0.252 ± 0.056 | no |
| **meandeg** | **6.877 ± 0.745** (mediana 6.97) | **5.933 ± 1.171** (mediana 5.85) | **sí** |
| **kcap** | **4.400 ± 0.632** (mediana 4) | **5.583 ± 0.954** (mediana 6) | **sí, el más fuerte** |
| z_agg | 4.210 ± 1.773 | 0.761 ± 0.565 | sí (consecuencia, no causa — ver abajo) |

La tasa de descarrilamiento por `kcap` es un gradiente muy limpio:

| kcap | reglas con pendiente<0 / total | tasa |
|---|---|---|
| 4 | 10 / 70 | **14.3%** |
| 5 | 4 / 135 | 3.0% |
| 6 | 1 / 147 | 0.7% |
| 7 | 0 / 78 | 0.0% |

Tiene sentido mecánico: `kcap` es el tope duro de vecinos por nodo (`_enforce_kcap`). Un `kcap` bajo
sobre un grafo inicial denso (`meandeg` alto) poda muchísimas aristas, deja más nodos aislados y más
fragmentos chicos, y por lo tanto sube la probabilidad de que el nodo de índice más bajo con aristas
caiga en un fragmento en vez de en el cuerpo principal. El `z_agg` alto del grupo (4.13 vs 0.77) es
**consecuencia** del mismo artefacto: ese z compara el diámetro REAL contra el de un NULL Erdős-Rényi, y
si el REAL vale 1 cuando debería valer ~20, la separación contra el null se dispara.

---

## 3. PASO 3 — ¿La masa alta replica en Phantom?

Scripts: `cs090_fase6_outliers_paso3_phantom.py` (genera IC + corre Phantom) y
`cs090_fase6_outliers_paso3_analizar.py` (sólo lectura, cruza resultados).

### 3.1 Qué se corrió y con qué protocolo

Las **11 candidatas** con pendiente < −0.3 que **no** habían pasado por Phantom (el corte −0.3 se fijó
antes de tener el diagnóstico de §1.4; queda fuera sólo `batch3-r95`, pendiente −0.186 — ver §2.2).
Protocolo idéntico al de las 80 corridas de Fase V-B, sin ninguna perilla nueva: N=2000, masa total fija
18800, lado de caja fijo 2000^(1/3), `seed_layout=12345`, turbulencia Mach=3 seed=42,
`icreate_sinks=1`, `rho_crit_cgs=1000`, `r_crit=0.6`, `h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`. Se
reusaron `reconstruir_regla_a2b0c2` + `generar_ic_masa_fija_desde_grafo` del adaptador congelado y
`correr_una` de `cs090_fase5b_correr.py` (mismo binario, misma edición de `cosmog.in`). Carpeta de
salida nueva: `/Users/alexis/phantom_cs073/bateria_fase6_outliers_negativos`.

**Doble verificación cruzada** (por el bug de colisión de nombres documentado en
`FASE5B_investigacion_8sumideros_y_escala_CS.md` §2.1), ambas por `assert` en el script:
1. **Antes** de generar la IC: el `p` regenerado desde el `seed` debe coincidir con el CSV de origen en
   K, J, noise, meandeg, kcap y sim_thr_frac; y el seed no puede estar entre los 76 que ya corrieron
   Phantom. Las 11 pasaron sin discrepancias.
2. **Después** de correr: se relee `meta_regla.json` de la carpeta y se exige que rule_id, seed, K y
   kcap coincidan con el CSV. Las 11 pasaron.

**Desviación forzada por el entorno, declarada:** `sarracen` (la librería que lee los volcados binarios
de Phantom) ya no está instalada en ningún intérprete de esta máquina — verificado en
python3.9/3.10/3.11/3.13 y por búsqueda en todo el disco. Por eso la fracción de masa se calcula desde
el log `.sink` como `masa_acretada_total / 18800` en vez de leerse del dump binario. **No es una métrica
distinta:** sobre las 80 corridas ya existentes (que tienen ambos números en
`cs090_fase5b_TOTAL_40pares.csv`) las dos formas coinciden con una diferencia máxima de **5.6×10⁻¹⁷** en
las 80 filas — el mismo número hasta el último bit del punto flotante. κ_V, nº de sumideros y
t_primer_sumidero salen de `analizar_sink`, que nunca usó sarracen.

Costo real: 458 s de generación de IC (~42 s cada una, el `layout_resortes` es el grueso) + ~11 s de
Phantom cada una. Total 581 s para las 11.

### 3.2 Resultados

Distribución de referencia (las 80 corridas de Fase V-B): media 0.0992, sd 0.0178, min 0.0620,
mediana 0.0985, p75 0.1040, **máximo 0.1535**.

| regla | pend. ORIGINAL | pend. CORREGIDA | fracción de masa | percentil vs las 80 | κ_V | nº sumideros |
|---|---|---|---|---|---|---|
| batch4-r53 | −1.2016 | +1.2490 | **0.1560** | **100** | 1.351 | 8 |
| batch4-r49 | −1.1006 | +1.2519 | **0.1570** | **100** | 1.514 | 8 |
| batch4-r110 | −1.0965 | +1.4432 | **0.1555** | **100** | 1.356 | 8 |
| batch3-r40 | −1.0008 | +1.0608 | **0.1560** | **100** | 1.495 | 8 |
| r0 | −0.8544 | +1.0282 | 0.1415 | 95.0 | 1.167 | 8 |
| batch4-r156 | −0.7737 | +0.9659 | 0.1325 | 95.0 | 1.027 | 8 |
| batch4-r118 | −0.7333 | +0.7733 | 0.1235 | 91.2 | 0.886 | 8 |
| batch4-r56 | −0.6170 | +0.8409 | 0.1065 | 76.2 | 0.568 | 8 |
| **batch3-r15** | −0.5919 | **+0.5937** | **0.0770** | **7.5** | 0.365 | 8 |
| batch4-r178 | −0.5601 | +1.0613 | 0.1520 | 98.8 | 1.276 | 8 |
| batch3-r45 | −0.4891 | +1.2658 | 0.1530 | 98.8 | 1.295 | 8 |

- media de las 11 = **0.1373** (sd 0.0259) contra 0.0992 de las 80.
- **10 de 11** por encima de la mediana de las 80; **10 de 11** por encima del p75;
  **4 de 11 por encima del MÁXIMO de las 80 corridas previas.**
- Mann-Whitney (11 nuevas vs 80 previas, dos colas): U=770.0, **p=6.0×10⁻⁵**.

**La masa alta replicó, y con margen.** No se dispersaron por todo el rango: se apilaron en el extremo
alto, cuatro de ellas fuera de todo lo visto antes.

### 3.3 La única que no replicó no es una excepción: es la que la pendiente corregida predice baja

`batch3-r15` cayó en 0.0770 (percentil 7.5, bien abajo). Es exactamente la que tiene la **pendiente
corregida más baja de las 11** (+0.5937, la única por debajo de 0.7). Su pendiente original (−0.5919) no
la distingue de las demás; su pendiente corregida sí.

Correlaciones:

| población | predictor | n | Spearman rho | p | R² lineal |
|---|---|---|---|---|---|
| sólo las 11 nuevas | pendiente ORIGINAL | 11 | **−0.6424** | 0.033 | 0.275 |
| sólo las 11 nuevas | pendiente CORREGIDA | 11 | **+0.8200** | 0.0020 | 0.777 |
| 87 reglas distintas (76 Fase V-B + 11 nuevas) | pendiente ORIGINAL | 87 | **+0.0644** | 0.554 | 0.286 |
| 87 reglas distintas (76 Fase V-B + 11 nuevas) | **pendiente CORREGIDA** | 87 | **+0.7397** | **2.7×10⁻¹⁶** | **0.7781** |

Lo que hay que mirar de esa tabla: al agregar las 11 nuevas, **la pendiente original deja de correlacionar
del todo** (rho +0.064, p=0.55 — indistinguible de nada) porque las 11 forman una segunda rama arriba a la
izquierda, convirtiendo la "U" de tres puntos en una U completa y simétrica. Con la **pendiente corregida**,
las mismas 87 reglas forman **una sola nube monótona** con rho=+0.74 y R²=0.78 — el ajuste más fuerte
visto en toda esta línea. Gráfico: `cs090_fase6_outliers_paso3.png` (paneles central y derecho).

---

## 4. Síntesis de números (sin cerrar nada)

| pregunta del encargo | número |
|---|---|
| ¿Qué son las 3 reglas? ¿Curva no-monótona? | **Sí, no-monótona**, y el quiebre está en **un solo punto**: diám=1 en b=1, luego 17/10/7/5. De b=2 en adelante son indistinguibles de reglas normales |
| ¿Grafo fragmentado? | El grafo **no está roto** (gigante 84%-95%, comparable a las referencias 91%-98%), pero tiene **más fragmentos chicos** (8-29 componentes, 77-242 aislados). `_diam` de cs055 arranca en el nodo de índice más bajo con aristas, y ése cayó en un **par de 2 nodos** en las tres |
| ¿Parámetros extremos? | Ni K, ni J, ni noise, ni sim_thr_frac. Sí **kcap bajo + meandeg alto**: tasa de descarrilamiento 14.3% con kcap=4, 3.0% con kcap=5, 0.7% con kcap=6, 0% con kcap=7 |
| ¿Cuántas hay como ellas en las 430? | **15 (3.5%)**, separadas del cuerpo principal por el hueco más grande de la distribución (0.50, entre −0.186 y +0.314). 12 nunca habían corrido Phantom. Las **11 reglas "intermedio (sin clase clara)" de las 430 están todas en ese grupo** |
| Corrigiendo el punto de arranque de la medición | Las 15 pendientes negativas se dan vuelta a **positivas (+0.59 a +1.44)**; 13 de 14 quedan por encima del umbral 0.7. Los diámetros de b=2..16 **no cambian ni un entero** |
| ¿La masa alta replica en Phantom? | **Sí.** 11 corridas nuevas: media 0.1373 vs 0.0992 de las 80 previas; 10/11 sobre el p75; **4/11 sobre el máximo previo**; Mann-Whitney p=6.0×10⁻⁵ |
| ¿Régimen aparte o misma tendencia? | Con la pendiente corregida, las 87 reglas distintas caen en **una sola nube monótona** (rho=+0.74, R²=0.78). Con la original, la correlación se cae a rho=+0.064 (p=0.55) |

**Lo que estos números NO dicen** (y por eso no hay veredicto acá): (a) que la pendiente corregida sea *la*
métrica correcta del proyecto — es un diagnóstico que calculé en un script nuevo, los umbrales de Clase
I-IV se calibraron con la medición original y reetiquetar 15 reglas es una decisión de método que no me
corresponde; (b) que el problema afecte sólo a A2-B0-C2 — no medí ningún otro combo de ejes, y `_diam`
es la misma función en toda la línea; (c) que las 80 corridas de Fase V-B estén mal — 73 de las 76
reglas distintas **no** descarrilan, así que el resultado principal de Fase V-B (Clase III acumula más
masa que Clase I, 31/40) no depende de esto, pero 3 de sus 76 reglas sí traen la etiqueta afectada;
(d) nada sobre causalidad. La lectura final es de Alexis.

---

## 5. Archivos de esta tarea

**Scripts nuevos** (ninguno modifica nada existente; `cs055_proceso_acoplado.py`,
`cs090_fase5_motor.py`, `cs090_fase5_clasificador.py`, `cs090_fase5b_phantom_adaptador.py`,
`cs090_fase5b_correr.py` y `cs090_fase5b_analizar.py` sólo se importan):

- `cs090_fase6_outliers_paso2_distribucion.py` — Paso 2: junta los 4 CSV de origen (430 reglas, verifica
  cero colisiones de `seed`), describe la distribución de pendientes, busca huecos, cruza contra las que
  ya corrieron Phantom y compara parámetros del grupo negativo vs el resto.
- `cs090_fase6_outliers_paso1_curvas.py` — Paso 1: reconstruye el grafo con la cadena exacta de
  `correr_regla_coarse`, re-mide las 5 escalas con diagnóstico extra de componentes, y verifica que la
  pendiente recalculada reproduzca la del CSV hasta 1e-9.
- `cs090_fase6_outliers_diagnostico_diam_gigante.py` — re-mide el diámetro sobre la componente gigante
  (elección determinista) en 87 reglas y compara pendiente original vs corregida como predictor.
- `cs090_fase6_outliers_paso3_phantom.py` — Paso 3: genera IC + corre Phantom sobre las 11 candidatas,
  con la doble verificación cruzada antes y después.
- `cs090_fase6_outliers_paso3_analizar.py` — cierre numérico del Paso 3.

**Datos crudos:**

- `cs090_fase6_outliers_430_todas.csv` — las 430 reglas con pendiente, parámetros, fuente y si pasaron
  por Phantom, ordenadas por pendiente.
- `cs090_fase6_outliers_candidatas.csv` — las 11 candidatas del Paso 3.
- `cs090_fase6_outliers_curvas.csv` — 35 filas (7 reglas × 5 escalas) con los puntos crudos de la curva
  de coarse-graining, componentes y tamaño de la componente medida.
- `cs090_fase6_outliers_diam_gigante.csv` — 87 reglas con pendiente original, pendiente corregida, las
  dos series de diámetros y el diagnóstico de descarrilamiento.
- `cs090_fase6_outliers_phantom_metricas.csv` — las 11 corridas nuevas de Phantom (métricas crudas).
- `cs090_fase6_outliers_paso3_resumen.csv` — las 11 con percentil vs las 80 y pendiente corregida.

**Gráficos:**

- `cs090_fase6_outliers_histograma.png` — distribución de las 430 + quién llegó a Phantom.
- `cs090_fase6_outliers_curvas.png` — curvas crudas de los 3 outliers vs 4 referencias (diámetro,
  gigante, nº de componentes).
- `cs090_fase6_outliers_diam_gigante.png` — original vs corregida, y el efecto sobre la correlación.
- `cs090_fase6_outliers_paso3.png` — dónde caen las 11 nuevas y las dos nubes (original vs corregida).

Corridas de Phantom: `/Users/alexis/phantom_cs073/bateria_fase6_outliers_negativos` (11 carpetas, cada
una con `cosmogenesis_ic.txt`, `meta_regla.json`, `cosmog.in`, dumps y `cosmog01.sink`).

No se modificó ningún script ni CSV existente. No se hicieron commits de git. No se declaró cierre ni
veredicto.
