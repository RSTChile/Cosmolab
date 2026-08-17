# Fase VI — adopción del diámetro corregido: qué se mueve y qué no

**Fecha:** 11-ago-2026 · Ejecuta: CC (Claude) · Encargo de Alexis: *"medición corregida reemplaza a la
oficial si estaba mala… aunque no sé si hay que correr todos los experimentos de nuevo por eso"*.

Antecedente: `FASE6_outliers_pendiente_negativa_CS.md` (el bug de `_diam`).
**No se editó ningún script congelado.** **No se corrió Phantom** (la física no cambió: se re-analizan
datos que ya estaban en disco). **No se hicieron commits.** **No se declara cierre ni veredicto** — se
reportan los números; qué re-correr lo decide Alexis.

---

## 0. En simple, con analogía

Medir el "diámetro" de una maqueta de alambre es medir la distancia entre sus dos puntos más lejanos.
La rutina que lo hacía (`_diam`, congelada desde cs055) empezaba a medir desde **el primer nodo de la
lista que tuviera algún cable** — el de número más bajo. Si ese nodo había quedado en un pedacito suelto
de dos nodos colgando al costado, la rutina medía **el pedacito**: daba 1, cuando la maqueta principal
medía 20.

La analogía sigue siendo la misma del informe anterior: **el metro se apoyó en el buzón de la vereda en
vez de en el edificio.** Da 30 cm.

Lo que se hizo acá tiene tres partes:

1. **Se arregló el metro** — se escribió una rutina nueva que siempre apoya la medición en el pedazo más
   grande de la maqueta. Se guardó en un **archivo aparte**, sin tocar la vieja. ¿Por qué? Porque decenas
   de experimentos ya publicados usan la vieja: si la cambiáramos en su lugar, ninguno de esos informes
   se podría volver a reproducir, y perderíamos justamente la forma de comprobar si un cambio de método
   mueve algo. Es como corregir una regla de medir mal calibrada: uno se compra una nueva, pero **guarda
   la vieja en el cajón** para poder verificar qué se midió con ella.
2. **Se volvió a medir todo lo que se podía volver a medir**: 430 reglas del barrido A2-B0-C2, 640
   corridas de la línea del mecanismo, 9 sustratos de Fase III. Con las dos varas a la vez, lado a lado.
3. **Se miró, uno por uno, si algún resultado ya publicado cambia.**

**El resumen en una frase:** el buzón sólo estaba en 15 de 430 maquetas (3,5%), y ninguna de las
conclusiones grandes se cae; lo que cambia es que **desaparece por completo la categoría "intermedio, sin
clase clara"** (las 11 reglas que estaban ahí eran todas casos de metro mal apoyado) y que **el observable
continuo pasa de no predecir nada a predecir bastante bien.**

---

## 1. PASO 1 — Cómo se implementó la corrección

### 1.1 Módulo nuevo: `cs090_diam_corregido.py`

Contiene:

| función | qué es |
|---|---|
| `diam_original(adj, N)` | la versión **histórica** de cs055, traída TAL CUAL por extracción AST (mismo truco que ya usa `cs090_fase5_motor`), para poder comparar lado a lado |
| `diam_gigante(adj, N)` | **la medición oficial desde el 11-ago-2026**: el MISMO doble-BFS, pero arrancando en la componente conexa más grande |
| `componentes(adj, N)` | componentes conexas en orden determinista de descubrimiento |
| `diagnostico(adj, N)` | las dos mediciones + tamaño de la componente que midió la vieja vs. la gigante + flag `descarrila` |
| `verificar_equivalencia(grafos)` | comprueba numéricamente que las dos coinciden donde deben coincidir |
| `correr_regla_coarse_doble(p)` | copia exacta de la cadena de `correr_regla_coarse` que devuelve las filas medidas de las **dos** maneras (REAL y los 3 NULL_topo), para poder clasificar dos veces sin correr dos veces |

**Elección determinista de la gigante:** `max(componentes, key=len)`; ante empate gana la primera
descubierta (la de índice más bajo). El nodo de arranque es el de índice más bajo *de esa componente*. No
depende de cómo estén numerados los nodos.

### 1.2 Por qué en un módulo nuevo y no parcheando `cs055`

`cs055_proceso_acoplado.py` está congelado y lo importan, directa o indirectamente, cs057, cs066, cs080
(Fase III), cs082 (Fase IV), `cs090_fase5_motor` (Fase V-A/V-B) y toda la línea del mecanismo. Si se
edita `_diam` en el lugar:

- cualquiera de esos scripts, al re-correrse, daría números distintos de los de su informe;
- se perdería la capacidad de **reproducir** los resultados históricos — que es exactamente lo que
  permitió, en esta tarea, separar "lo que movió la corrección" de "lo que se movió por otra cosa"
  (ver §2.3, la deriva de entorno);
- rompe la regla de la casa de no editar experimentos cerrados.

Entonces: **la vieja se conserva en `cs055` sólo para reproducir historia; la nueva es la oficial para
cálculo nuevo.** Ambas conviven y se pueden imprimir juntas.

### 1.3 Verificación de equivalencia (no se asumió: se comprobó)

Primero el argumento formal: si el nodo de arranque de la vieja (`src` = el no-aislado de índice más bajo
del grafo entero) ya está **dentro** de la gigante, entonces `src` es también el nodo de índice más bajo
*de* la gigante, o sea es exactamente el nodo desde el que arranca la corregida — y de ahí en adelante el
algoritmo es idéntico. Las dos tienen que dar el **mismo entero**.

Ahora la comprobación numérica (`python3.9 cs090_diam_corregido.py`), sobre grafos reales del proyecto:

```
[equivalencia] grafos probados: 600
   sin descarrilamiento: 594  ->  idénticos orig==corr: 594 (100.00%)
   con descarrilamiento: 6     (acá SÍ deben diferir: es el bug)
```

600 grafos = 30 reglas A2-B0-C2 (5 del grupo negativo + 25 muestreadas a lo largo de toda la
distribución) × 5 escalas de coarse-graining × (1 real + 3 NULL_topo). **594 de 594 idénticas.** La
corrección no re-escribe la historia donde la historia estaba bien.

Un dato adicional del re-medido masivo de §2 que refuerza lo mismo: de las **430 reglas**, la pendiente
corregida difiere de la vieja en **16**; en las otras **414 el número es idéntico bit a bit**.

---

## 2. PASO 2 — Re-medición de las 430 reglas

Script: `cs090_fase6_remedir_430.py` · Salida: `cs090_fase6_remedicion_430.csv` (430 filas) ·
Costo: **9,4 minutos** (1,3 s por regla).

Roster: las 430 reglas consolidadas en `cs090_fase6_outliers_430_todas.csv`, que viene de los 4 CSV de
origen (`cs090_fase5_profundizar_a2b0c2_resumen.csv` sección `nueva_profundizar` + `candidatas_v2/v3/v4`).
Cada regla se reconstruye con la cadena determinista exacta de `correr_regla_coarse` y se clasifica con
`cs090_fase5_clasificador.clasificar_regla` **sin cambiar ni un umbral**. Se corrigen también los
NULL_topo (el z-score compara REAL contra NULL; corregir un solo lado inventaría una asimetría).

### 2.1 El número central: **15 de 430 cambian de clase (3,5%)**

| | I | II | III | IV | intermedio | total |
|---|---|---|---|---|---|---|
| clase **vieja** | 225 | 24 | 164 | 6 | 11 | 430 |
| clase **corregida** | 222 | 24 | **176** | **8** | **0** | 430 |

Transiciones:

| de | a | n |
|---|---|---|
| intermedio (sin clase clara) | III | 9 |
| I | III | 3 |
| I | IV | 1 |
| intermedio | IV | 1 |
| intermedio | I | 1 |
| **total** | | **15** |

**Las tres respuestas pedidas:**

1. **¿Cuántas cambian?** 15 de 430 = 3,5 %.
2. **¿Son sólo las 15 de pendiente negativa?** **Sí, exactamente esas.** De las 15 con pendiente vieja
   < 0, **cambian las 15**. De las 415 con pendiente ≥ 0, **cambia 0**. La frontera cae exactamente en
   pendiente = 0, tal como anticipaba el informe anterior. (Hay una regla número 16 cuya *pendiente*
   cambia sin cambiar de clase — ver §2.2.)
3. **¿En qué dirección?** Siempre hacia arriba salvo una: 11 de las 15 terminan en III o IV. La categoría
   **"intermedio (sin clase clara)" desaparece del lote: 11 → 0.** Las 11 reglas que estaban en esa
   casilla eran, todas, el metro apoyado en el buzón.

Tabla completa de las 15 (columnas: pendiente y z antes/después, y las dos series de diámetros b=1..16):

| regla | seed | kcap | pend vieja | pend corr. | z vieja | z corr. | clase vieja | clase corr. | diám viejo | diám corregido |
|---|---|---|---|---|---|---|---|---|---|---|
| batch4-r53 | 576970 | 4 | −1.2016 | **+1.2490** | 4.84 | 1.02 | intermedio | III | [1,16,10,7,5] | [19,16,10,7,5] |
| batch4-r49 | 576582 | 4 | −1.1006 | **+1.2519** | 3.95 | 0.74 | intermedio | III | [1,15,9,7,5] | [21,15,9,7,5] |
| batch4-r110 | 582499 | 4 | −1.0965 | **+1.4432** | 5.40 | 3.43 | intermedio | III | [1,19,12,8,5] | [28,19,12,8,5] |
| batch3-r40 | 475709 | 4 | −1.0008 | **+1.0608** | 8.06 | 0.85 | intermedio | III | [1,13,9,6,5] | [18,13,9,6,5] |
| batch3-r100 | 481529 | 4 | −0.9423 | **+1.2550** | 2.86 | 1.17 | I | III | [1,17,10,7,5] | [25,17,10,7,5] |
| r0 | 271829 | 4 | −0.8544 | **+1.0282** | 4.75 | 0.83 | intermedio | III | [1,13,9,6,5] | [21,13,9,6,5] |
| batch4-r51 | 576776 | 4 | −0.8118 | **+1.0807** | 0.86 | 0.85 | I | III | [1,15,9,6,5] | [22,15,9,6,5] |
| batch4-r156 | 586961 | 5 | −0.7737 | **+0.9659** | 5.02 | 2.83 | intermedio | III | [1,12,8,7,4] | [18,12,8,7,4] |
| batch4-r118 | 583275 | 5 | −0.7340 | **+0.7733** | 3.47 | 0.53 | intermedio | III | [1,11,8,5,5] | [15,11,8,5,5] |
| batch3-r143 | 485700 | 5 | −0.6471 | **+0.7640** | 2.03 | 0.05 | I | **IV** | [1,9,7,5,4] | [14,9,7,5,4] |
| batch4-r56 | 577261 | 5 | −0.6170 | **+0.8409** | 2.39 | 0.82 | I | III | [1,11,8,5,4] | [15,11,8,5,4] |
| batch3-r15 | 473284 | 6 | −0.5919 | **+0.5937** | 4.38 | 0.85 | intermedio | **I** | [1,8,6,5,4] | [12,8,6,5,4] |
| batch4-r178 | 589095 | 4 | −0.5601 | **+1.0613** | 5.36 | 0.95 | intermedio | **IV** | [2,13,9,7,5] | [20,13,9,7,5] |
| batch3-r45 | 476194 | 4 | −0.4891 | **+1.2658** | 5.99 | 1.09 | intermedio | III | [2,16,11,6,5] | [22,16,11,6,5] |
| batch3-r95 | 481044 | 4 | −0.1859 | **+1.1474** | 3.79 | 1.23 | intermedio | III | [3,15,10,6,5] | [19,15,10,6,5] |

Nótese que **los diámetros de b=2 a b=16 no cambian ni un entero en ninguna**: el único punto tocado es
el b=1. Y el `z_agg` **baja** en 14 de 15 (de 3-8 a 0-3): el z alto de esas reglas era el artefacto del
REAL valiendo 1 contra un NULL de ~20.

La tasa de descarrilamiento sigue el gradiente de `kcap` ya documentado: 14,3 % con kcap=4 · 3,0 % con
kcap=5 · 0,7 % con kcap=6 · 0,0 % con kcap=7.

### 2.2 Un hallazgo nuevo: el descarrilamiento también puede pegar en b=2

`A2-B0-C2-batch4-r152` (seed 586573) tiene diámetros viejos **[23, 2, 11, 6, 5]** — el 2 está en **b=2**,
no en b=1. Corregido: [23, **18**, 11, 6, 5]. Su pendiente sube de +0.795 a +1.505, pero **no cambia de
clase** (III → III). Es la única de las 430 en esa situación (16 reglas con descarrilamiento en alguna
escala = 15 en b=1 + 1 en b=2). Vale anotarlo porque el criterio de detección "pendiente < 0" **no la
habría encontrado**: el criterio robusto es el que se usa acá — *la medición cayó en una componente con
menos del 10 % del tamaño de la gigante*, en cualquier escala.

### 2.3 Control de fidelidad, y una deriva de entorno que hay que declarar

La reconstrucción se verificó exigiendo que la pendiente y el z_agg re-medidos **con la vara vieja**
coincidan con el CSV histórico hasta 1e-9:

- **386 de 430 coinciden exactamente** (mediana de la diferencia: 4×10⁻¹⁶, o sea cero de punto flotante).
- **44 de 430 tienen una diferencia pequeña**, máximo |Δpendiente| = **0.097**, típicamente 1e-4.
- De esas 44, **una sola cambia de clase respecto al CSV** (`A2-B0-C2-r15`: II en el CSV, I al
  reconstruir) — y ese cambio **no tiene nada que ver con la corrección** (esa regla no descarrila).

La causa es deriva de entorno (versión de numpy / orden de iteración), no del método: el diámetro y el
z son idénticos y lo que se mueve en esos casos es el número de cajas del coarse-graining en ±1. Se
declara acá porque afecta a cualquier re-corrida futura de esta línea, no sólo a esta tarea.

### 2.4 Efecto sobre el observable continuo (el reanálisis de Fase VI)

Sobre las **76 reglas distintas** que pasaron por Phantom en Fase V-B, la pendiente como predictor de la
fracción de masa en sumideros:

| predictor | n | Spearman rho | p | R² lineal |
|---|---|---|---|---|
| pendiente **vieja** | 76 | +0.4634 | 2.5×10⁻⁵ | **0.011** |
| pendiente **corregida** | 76 | **+0.6283** | **1.2×10⁻⁹** | **0.663** |

Este es el resultado publicado que **más se mueve**: el R² pasa de 0.011 (o sea, la recta no explicaba
nada) a 0.663. Con las 11 corridas nuevas del informe anterior incluidas (87 reglas) el ajuste llega a
rho=+0.74 / R²=0.78.

---

## 3. PASO 3 — Impacto sobre cada resultado publicado

### 3.1 Fase V-A, barrido global de 150/180 reglas — **no se puede re-medir; tres pruebas dicen que no está tocado**

Script: `cs090_fase6_auditoria_fase5a.py`.

**Limitación primero, sin adornos.** El barrido de V-A es el único de esta línea que **no se puede volver
a medir**: su CSV no guarda `seed`, y el script derivaba las semillas con

```python
seed_base = abs(hash((eje_A, eje_B, eje_C, "paso1"))) % 100000   # cs090_fase5_completo.py
```

`hash()` de strings en Python está aleatorizado por proceso salvo que se fije `PYTHONHASHSEED`, que no
quedó registrado. Ni conociendo el código se regenera el mismo lote. **No se adivina nada**: se buscan las
firmas del bug en los datos que sí quedaron guardados (`cs090_fase5_completo_resultados.csv` tiene
diámetro REAL, NULL y gigante por regla y por escala).

**Prueba 1 — imposibilidad geométrica.** Agrupar nodos en cajas conexas nunca puede alargar un camino, así
que diám(b=1) < diám(b=2) es imposible salvo que la medición haya saltado de componente. En las 15
confirmadas se cumple siempre. En Fase V-A: **0 de 150.**

**Prueba 2 — calibración contra verdad de terreno.** De las 430 re-medidas de verdad se conoce el rango:

| | descarriladas (n=15) | sanas (n=415) |
|---|---|---|
| diám(b=1) | **≤ 3** (1, 2 o 3) | **≥ 8** |
| cociente diám(b1)/diám(b2) | ≤ 0.20 | ≥ 1.10 |

Separación total, sin zona gris. En Fase V-A el **diám(b=1) mínimo de las 150 reglas es 4** (y ese 4 es de
un combo A0, donde el grafo de medición es conexo al 100 %); las 140 reglas no-A0 tienen diám(b=1) ≥ 6.
**0 de 150 caen en el rango de las descarriladas.**

**Prueba 3 — fragmentación.** El bug exige fragmentos sueltos. La fracción de componente gigante a b=1 en
Fase V-A: los tres combos A0 dan **exactamente 1.0000** (grafo de medición conexo, imposible que muerda);
los otros doce tienen mediana 0.974-0.999 y mínimo 0.776. El combo A2-B0-C2 de V-A, que es el "peor caso"
esperable, tiene gigante mínima 0.923 y diám(b=1) mínimo 10 — mucho más sano que los batches del barrido
de 430.

**Conclusión sobre V-A (con la limitación puesta arriba de la mesa):**

| afirmación publicada | ¿cambia? |
|---|---|
| distribución global {I: 99 (66 %), II: 43 (29 %), III: 8 (5 %), IV: 0} | **no hay ninguna evidencia de que cambie**; ninguna de las 3 pruebas marca una sola regla |
| "muy fuerte CONTRADICHO": las únicas dos combinaciones con Clase III son **A1-B0-C2** (3/10) y **A2-B0-C2** (5/10), las dos con **B0**, y B1+C1/C2 dio 0 | **no cambia.** El bug, cuando actúa, **añade** Clase III; nunca la quita (las 15 relabeladas van 11 hacia III/IV, 3 se quedan y 1 baja a I). Si actuara sobre B1+C1/C2 sólo podría *reforzar* la refutación de la hipótesis "muy fuerte", no revertirla |
| falsación de "A0 nunca Clase II+" (8/30 en Clase II) | **no cambia, y acá es imposible que cambie**: los tres combos A0 tienen **gigante = 1.0000** en las 30 reglas, o sea grafo de medición conexo, o sea `_diam` no tiene dónde descarrilar |
| "0 reglas en Clase IV en todo V-A" | **no verificable con certeza**, pero las 3 pruebas no marcan nada; y las relabeladas del barrido de 430 producen IV sólo en 2 de 430 casos |

**Lo que haría falta para cerrarlo del todo:** re-correr el barrido V-A completo con `seed` guardado. Coste
estimado en §4.

### 3.2 La línea del mecanismo F5-C2-C → C5 — **el orden de los brazos se sostiene; se estrecha una sola comparación**

Script: `cs090_fase6_remedir_mecanismo.py` · Salida: `cs090_fase6_remedicion_mecanismo.csv` (640 filas) ·
Costo: **13,5 minutos**.

**Método, porque importa:** los seis scripts calculan el diámetro llamando siempre a
`cs090_fase5_motor._diam(...)`, que es una búsqueda de atributo de módulo resuelta en el momento de la
llamada. Entonces basta con **sustituir ese atributo en memoria** por `diam_gigante` antes de invocar la
función de brazo de cada script y restaurarlo después. Ningún archivo se modifica en disco y cada brazo
corre con su cadena exacta. El lado viejo no se re-corre: se reconstruye desde los CSV históricos
`cs090_fase5_*_resultados.csv`, que ya guardan diám/NULL/N por escala — y se verificó que la clase
recalculada desde el CSV reproduce la `clase_final` guardada en **640 de 640** filas.

**Resultado: 18 de 640 (2,8 %) cambian de clase.** Pero hay que separar dos cosas, y se separan:

- **12 son descarrilamientos genuinos** (diám b=1 entre 1 y 4, contra 9-24 corregido — dos de ellos en
  b=2, no b=1).
- **6 son deriva de entorno** (la de §2.3): saltos de ±0.05 en la pendiente justo encima de un umbral
  (II↔I, IV↔III). No tienen nada que ver con la corrección.

Para aislar el efecto **de la corrección**, la tabla siguiente adopta la clase corregida **sólo** donde
hubo descarrilamiento y conserva la histórica en el resto. Ese es el número honesto de "qué pasa si
adoptamos la medición corregida":

| tarea | brazo | n | %III publicado | %III adoptando la corrección |
|---|---|---|---|---|
| presupuesto_emergente | C2-hard | 20 | 45.0 | **45.0** |
| | C2-budget | 20 | 15.0 | 15.0 |
| | C2-random | 20 | 15.0 | 15.0 |
| | C0 | 20 | 0.0 | 0.0 |
| presupuesto_soporte | C2-hard | 20 | 45.0 | **45.0** |
| | C2-budget-original | 20 | 15.0 | 15.0 |
| | C2-budget-soporte | 20 | 10.0 | 10.0 |
| | C2-random | 20 | 20.0 | 20.0 |
| | C0 | 20 | 0.0 | 0.0 |
| mecanismo_aislado | C2-hard | 20 | 45.0 | **45.0** |
| | **C2-hibrido** | 20 | 35.0 | **40.0** |
| | C2-budget-soporte | 20 | 10.0 | 10.0 |
| | C2-random | 20 | 5.0 | 5.0 |
| | C0 | 20 | 0.0 | 0.0 |
| presupuesto_variable (matriz 2×2) | C2-hard | 20 | 45.0 | **45.0** |
| | **C2-hibrido** | 20 | 35.0 | **40.0** |
| | C2-presupuesto-variable | 20 | 10.0 | 10.0 |
| | C2-budget-soporte | 20 | 10.0 | 10.0 |
| | C0 | 20 | 0.0 | 0.0 |
| control_azar_elastico | C2-hard | 20 | 45.0 | **45.0** |
| | **C2-hibrido** | 20 | 35.0 | **40.0** |
| | C2-presupuesto-variable | 20 | 10.0 | 10.0 |
| | C2-presupuesto-variable-azar | 20 | 10.0 | 10.0 |
| | C0 | 20 | 0.0 | 0.0 |

Genealogías independientes:

| genealogía | brazo | %III publicado | %III adoptado |
|---|---|---|---|
| G0_original_90210 | C2-hard | 50.0 | **55.0** |
| | C2-hibrido | 40.0 | 40.0 |
| G1_471829 | C2-hard | 45.0 | 45.0 |
| | C2-hibrido | 25.0 | **40.0** |
| G2_823001 | C2-hard | 65.0 | **70.0** |
| | C2-hibrido | 25.0 | **35.0** |
| G3_156644 | C2-hard | 75.0 | 75.0 |
| | C2-hibrido | 50.0 | **55.0** |

**Qué se sostiene y qué se mueve:**

| conclusión publicada | ¿se sostiene? |
|---|---|
| **C2-hard ≫ todo lo demás** (45 % vs 0-20 %) — el hallazgo central de la línea | **Se sostiene tal cual, sin mover un decimal.** C2-hard queda en 45.0 % en las 5 tareas; C0 en 0.0 %; budget/budget-soporte/presupuesto-variable/random entre 5 y 20 %, todos idénticos a lo publicado |
| "ni la señal de costo ni la uniformidad del cupo reproducen kcap fijo" (C2-budget ≈ C2-random ≈ 15 %; presupuesto-variable = presupuesto-variable-azar = 10 %) | **Se sostiene sin cambios.** Ninguno de esos brazos mueve un punto |
| "lo que hace falta es **rigidez del corte + criterio de soporte juntos**" | **Se sostiene**, pero con un matiz cuantitativo: la brecha C2-hard vs C2-hibrido (que es el brazo "rigidez sin uniformidad") se **estrecha de 10 pp a 5 pp** (45 % vs 40 %) en las 3 tareas donde aparece |
| genealogías: "el patrón se sostiene entre redes distintas, n=4 baja potencia" | **Se sostiene, y con menos dispersión.** C2-hard sigue por encima de C2-hibrido en las 4 genealogías (55/45/70/75 vs 40/40/35/55). Medias: hard 58.8 → **61.3 %**, hibrido 35.0 → **42.5 %**; la brecha media baja de 23.8 pp a 18.8 pp |

**El único texto que hay que corregir:** donde los informes dicen que C2-hibrido queda en 35 % (o 25 % en
G1/G2), el número adoptando la corrección es 40 % (y 40 %/35 %). La conclusión cualitativa —C2-hard
sigue arriba en todas las comparaciones— no se toca.

### 3.3 Fase V-B, los 40 pares (el más importante) — **sobrevive**

Script: `cs090_fase6_reanalisis_40pares_corregido.py` · Salida:
`cs090_fase6_reanalisis_40pares_corregido.csv`. **No se corrió Phantom**: los sumideros son los mismos,
lo único que cambia es qué está comparando cada par.

De las 80 reglas de los 40 pares, **3 cambian de etiqueta** — las 3 en el brazo "Clase I", las 3 hacia
arriba:

| par | brazo | regla | vieja → corregida | efecto sobre el par |
|---|---|---|---|---|
| batch3-r100 vs batch3-r0 | I | A2-B0-C2-batch3-r100 | I → **III** | **contraste roto** (III vs III) |
| batch4-r51 vs batch4-r36 | I | A2-B0-C2-batch4-r51 | I → **III** | **contraste roto** (III vs III) |
| batch3-r143 vs batch3-r70 | I | A2-B0-C2-batch3-r143 | I → **IV** | **contraste invertido** (IV vs III) |

Estado de los 40 pares tras re-etiquetar: **37 válidos, 2 rotos, 1 invertido.**

| análisis | n | III > I | signos (p) | Wilcoxon (W, p) |
|---|---|---|---|---|
| **A) como se publicó**, 40 pares, etiquetas viejas | 40 | 31/40 (77.5 %) | 0.00068 | W=80, **1.41×10⁻⁶** |
| **B) sólo los 37 pares que siguen siendo contraste válido** | 37 | **29/37 (78.4 %)** | **0.00075** | W=57, **1.03×10⁻⁶** |
| **C) 37 válidos + el invertido re-orientado** (sensibilidad) | 38 | 30/38 (78.9 %) | 0.00047 | W=58, 6.5×10⁻⁷ |

Lo mismo con κ_V: 28/40 p=0.017 (publicado) → **26/37 p=0.020** (válidos).

**El resultado sobrevive prácticamente intacto**, y hay una razón mecánica que conviene ver: los 3 pares
afectados tenían diferencias de masa **casi nulas** — |Δfracción| = 0.0005, 0.0050 y 0.0085, contra una
mediana de 0.0075 y un máximo de 0.034 en los 37 válidos. Eran empates técnicos: sacarlos no le quita
señal al conjunto, se la quita al ruido. La proporción de aciertos incluso **sube** levemente (77.5 % →
78.4 %).

### 3.4 Fase III, renormalización — **el bug no la toca, y está verificado, no supuesto**

Script: `cs090_fase6_remedir_fase3.py` · Salida: `cs090_fase6_remedicion_fase3.csv`.

`cs080_renormalizacion.py` mide con `C7._diam`, que es la misma función con el bug, así que la pregunta
era legítima. Se reconstruyeron los 9 sustratos (3 semillas × 3 brazos: local / local_barajado / er_null,
N=8000, con `construir_sustrato` sin tocarlo) y se midió en las 6 escalas con las dos varas:

```
[resultado] escalas donde la medición vieja DESCARRILA: 0/54
[resultado] escalas donde viejo != corregido:           0/54
```

Pendientes, vieja contra corregida, **idénticas hasta el cuarto decimal en las 9**:

| brazo | seed 80100 | seed 80200 | seed 80300 |
|---|---|---|---|
| local | +0.4110 / +0.4110 | +0.3600 / +0.3600 | +0.3904 / +0.3904 |
| local_barajado | +0.3671 / +0.3671 | +0.4311 / +0.4311 | +0.3690 / +0.3690 |
| er_null | +0.3965 / +0.3965 | +0.3885 / +0.3885 | +0.3907 / +0.3907 |

Las dos conclusiones de `FASE3_renormalizacion_resultado_CS.md` — mundo-pequeño persiste a todas las
escalas, y la pendiente 0.35-0.45 es indistinguible entre REAL / barajado / ER — **quedan intactas**.

**Un matiz honesto que sí conviene anotar:** los grafos de Fase III **sí se fragmentan** de forma
comparable a los de A2-B0-C2 (a b=1 tienen 13-28 componentes y 220-245 nodos aislados; los que
descarrilaron en el barrido de 430 tenían 8-29 componentes y 77-242 aislados). O sea: **no es que Fase III
sea inmune por construcción**, es que en estas 9 realizaciones concretas el nodo de índice más bajo cayó
siempre dentro de la gigante. Si alguna vez se re-corre Fase III con otras semillas, conviene usar la
medición corregida por precaución.

**Limitación declarada:** `cs080_renormalizacion.py` deriva varias semillas de rng con `hash(arm)`, que en
Python está aleatorizado por proceso; la corrida histórica no es reproducible bit a bit y este script fija
`PYTHONHASHSEED=0`. Los grafos son del mismo tipo pero no la misma realización, así que **no se comparan
los diámetros número a número contra el CSV histórico** — y por eso la afirmación de arriba es "el bug no
muerde en este tipo de sustrato", no "el CSV histórico está verificado línea por línea".

---

## 4. PASO 4 — Qué habría que re-correr DE VERDAD

### A) Se sostiene tal cual, no hay que tocar nada

| resultado | por qué |
|---|---|
| **Fase V-B, 40 pares** (el resultado principal de la línea) | re-analizado con etiquetas corregidas: 29/37, p=0.00075, Wilcoxon 1.03×10⁻⁶. Los 3 pares afectados eran empates técnicos |
| **Fase III renormalización** | re-medido: 0/54 descarrilamientos, pendientes idénticas al cuarto decimal |
| **La línea del mecanismo, hallazgo central** (C2-hard 45 % ≫ budget/soporte/random/variable 0-20 %) | re-medido: C2-hard, C0, budget, budget-soporte, random y presupuesto-variable **no mueven ni un punto** |
| **Fase V-A: falsación de "A0 nunca Clase II+"** | los 3 combos A0 tienen gigante = 1.0000 (grafo de medición conexo): el bug no tiene dónde morder |
| **Fase V-A: "muy fuerte CONTRADICHO"** | el bug sólo puede añadir Clase III, nunca quitarla; sólo podría reforzar la refutación |

### B) Hay que **corregir el texto**, sin re-computar nada

1. **Las 11 reglas "intermedio (sin clase clara)" del barrido de 430 dejan de existir** (11 → 0). Toda
   frase que las trate como una categoría real hay que reescribirla: eran artefacto de medición.
   Distribución oficial nueva del lote de 430: **I=222, II=24, III=176, IV=8**.
2. **`FASE6_reanalisis_azar_continuo_CS.md`, §3.3 (la "forma de U")**: hay que sustituirlo por rho=+0.628,
   R²=0.663 sobre las 76 con Phantom (o rho=+0.74 / R²=0.78 sobre las 87 con las 11 nuevas). La U no
   existe.
3. **Informes de la línea del mecanismo**: donde dicen C2-hibrido 35 %, poner 40 %; en genealogías G1 y G2,
   25 % → 40 % y 35 %. La brecha C2-hard vs C2-hibrido pasa de 10 pp a 5 pp (y de 23.8 a 18.8 pp de
   media en genealogías). El orden no cambia.
4. **`FASE5B_escala_40pares_CS.md`**: agregar que 3 de los 40 pares dejaron de ser contrastes I-vs-III y
   que el resultado con los 37 válidos es 29/37, p=0.00075.
5. **Anotar el criterio de detección correcto** para el futuro: no es "pendiente < 0" (eso se le escapa el
   caso de descarrilamiento en b=2), es *"la medición cayó en una componente < 10 % de la gigante, en
   cualquier escala"*.

### C) Re-corridas de cómputo real que se podrían proponer — con su costo

Ninguna es imprescindible para sostener lo ya publicado. Van en orden de valor/costo:

| # | qué | costo real medido/estimado | qué compra |
|---|---|---|---|
| **1** | **Nada.** Adoptar `cs090_diam_corregido` como oficial de acá en adelante y corregir el texto según §4B | 0 | ya está todo lo necesario en disco |
| **2** | **Re-correr el barrido Fase V-A completo con `seed` guardado** (150-180 reglas × 15 combos, motor liviano) | por analogía con las 430 (1,3 s/regla en A2-B0-C2; los combos B1 y A0 son más caros): **~15-40 min de una máquina, sin Phantom** | cierra el único punto donde hoy hay *inferencia indirecta* en vez de medición. También arregla, de paso, dos problemas de reproducibilidad que **no** son del diámetro: que las semillas salgan de `hash()` y que los `rule_id` se repitan entre las dos pasadas del barrido |
| **3** | **Phantom sobre las 3 reglas re-etiquetadas de los 40 pares**, para reemplazar los 2 pares rotos + 1 invertido por contrastes válidos nuevos | ~45 s de generación de IC + ~11 s de Phantom por regla ⇒ **~3 min por par nuevo**, más elegir las parejas | devolvería los 40 pares a 40 contrastes válidos. **Valor bajo**: el resultado ya sobrevive con 37 y la proporción de aciertos incluso sube |
| **4** | **Re-correr Fase III con la medición corregida y `PYTHONHASHSEED` fijado** | **~2,5 min** (el histórico tardó 2,4 min) | valor casi nulo para el resultado (0/54 descarrilamientos), pero deja Fase III reproducible bit a bit hacia adelante, que hoy no lo es |
| **5** | **Auditar con el detector barato el resto de la línea (Fase IV / cs082, CS073, CS084-089)** — buscar `diám(b=1) < diám(b=2)` y `diám(b=1) ≤ 3` en todos los CSV que guarden diámetro por escala | **segundos** (es sólo lectura de CSV) | esta tarea cubrió Fase III y toda la Fase V; **Fase IV y la línea CS07x-CS08x no se auditaron**, y usan el mismo `_diam`. Es lo más barato que queda por hacer |

**Recomendación, para que quede en una línea:** hacer **#1 + #5** (coste ~0), y considerar **#2** si Alexis
quiere que el mapa global de V-A deje de depender de una inferencia indirecta. **#3 y #4 no cambian
ninguna conclusión.** Nada de esto es un cierre: la lectura final es de Alexis.

---

## 5. Archivos de esta tarea

**Scripts nuevos** (ninguno modifica nada existente):

- `cs090_diam_corregido.py` — **el módulo de la medición oficial corregida** + la vieja para comparar +
  verificación de equivalencia + `correr_regla_coarse_doble`. Tiene auto-test: `python3.9 cs090_diam_corregido.py`.
- `cs090_fase6_remedir_430.py` — re-mide las 430 reglas con las dos varas y re-clasifica con los umbrales
  oficiales sin cambiarlos.
- `cs090_fase6_remedir_mecanismo.py` — re-mide las 640 corridas de la línea del mecanismo + genealogías,
  sustituyendo `MOT._diam` en memoria (ningún archivo tocado en disco).
- `cs090_fase6_reanalisis_40pares_corregido.py` — re-etiqueta los 40 pares de Fase V-B y rehace signos +
  Wilcoxon reusando las funciones de `cs090_fase5b_estadistica_40pares.py`.
- `cs090_fase6_remedir_fase3.py` — reconstruye los 9 sustratos de Fase III y compara las dos mediciones.
- `cs090_fase6_auditoria_fase5a.py` — las 3 pruebas indirectas sobre el barrido V-A que no se puede re-medir.

**Datos:**

- `cs090_fase6_remedicion_430.csv` — **la tabla vieja-vs-corregida de las 430** (pendiente, z, clase,
  cambia_clase, diámetros por escala, tamaño de componente medida vs gigante).
- `cs090_fase6_remedicion_mecanismo.csv` — 640 filas (tarea, regla, brazo, clase vieja/corregida, diámetros).
- `cs090_fase6_reanalisis_40pares_corregido.csv` — 40 pares con estado del contraste.
- `cs090_fase6_remedicion_fase3.csv` — 54 filas (3 semillas × 3 brazos × 6 escalas).

**Logs:** `cs090_fase6_remedir_430.log`, `cs090_fase6_remedir_mecanismo.log`,
`cs090_fase6_remedir_fase3.log`, `cs090_fase6_remedir_fase3_N8000.log`.

No se editó ningún script congelado. No se corrió Phantom. No se hicieron commits de git. No se declaró
cierre ni veredicto.
