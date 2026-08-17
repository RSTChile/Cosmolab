# Fase VI · O2-E — Barrido de Fase V-A re-corrido de forma REPRODUCIBLE, con las dos varas de diámetro

**Fecha:** 11-ago-2026 · Ejecuta: CC (Claude) · Script: `cs090_fase6_o2e_barrido_va_reproducible.py`
(único archivo nuevo de código) · Log: `cs090_fase6_o2e_barrido.log`

No se declara cierre ni veredicto sobre S>0 ni sobre ninguna Clase. Se reportan números; la lectura final
es de Alexis. No se corrió Phantom. No se tocó ningún script congelado ni ningún CSV existente. No se
hicieron commits.

---

## 0. En simple, con analogía (leer esto primero)

Teníamos una duda de instrumento, no de física. Habíamos descubierto que la regla de medir "cuán largo es
el camino más largo del tejido" (el diámetro) a veces se equivocaba: si el tejido tenía trocitos sueltos,
la regla podía terminar midiendo un trocito suelto en vez de la tela grande, y devolvía un número
ridículamente chico. Ya habíamos re-medido casi todo lo de esta línea con la regla arreglada. **Faltaba un
solo experimento: el barrido grande de Fase V-A**, y no se podía re-medir porque no habíamos anotado las
semillas — es como tener la foto de una torta pero haber perdido la receta: no se puede volver a hornear
*esa misma* torta.

Lo que hicimos ahora: **volver a hornear las 150 tortas desde cero, anotando esta vez la receta exacta de
cada una** (semilla fija, escrita en el CSV), y **pesar cada torta con las dos balanzas al mismo tiempo**
— la vieja (rota) y la nueva (arreglada) — en la misma pasada, sin hornear dos veces.

Resultado: **las dos balanzas dan exactamente el mismo peso en las 150 tortas.** Ninguna cambia de
categoría. La balanza rota no llegó a torcer nada acá, porque estas tortas casi no tienen migas sueltas
(la tela grande abarca entre el 83 % y el 100 % de cada tejido; el bug necesita migas para morder).

Ahora bien — el lote nuevo no es *idéntico* al publicado en agosto, y ahí está el punto fino del informe:
sí hay diferencias contra los números publicados, pero **ninguna viene de la balanza; todas vienen de que
son otras tortas.** Es la diferencia entre "el termómetro estaba mal calibrado" y "hoy hizo un grado
menos". Lo primero invalida la medición; lo segundo es simplemente el clima. Acá es lo segundo.

---

## 1. Qué se corrió, y por qué ahora sí es reproducible

Mismo barrido de Fase V-A, pieza por pieza: **18 combinaciones** (eje A {A0,A1,A2} × eje B {B0,B1} × eje C
{C0,C1,C2}) **× 10 reglas**, mismo generador (`cs090_fase5_generador`, filtro P1-P5), mismo motor
(`cs090_fase5_motor`), mismo clasificador y **los mismos umbrales sin tocar** (`cs090_fase5_clasificador`),
mismos parámetros: N=2000, n_sweeps=14, escalas de coarse-graining b=1/2/4/8/16, 3 semillas de NULL_topo.

Tres diferencias, **todas de método de registro, ninguna de física**:

| | barrido original (10-ago) | O2-E (hoy) |
|---|---|---|
| semilla de cada combo | `abs(hash((A,B,C,"paso1"))) % 100000` — `hash()` de strings está aleatorizado por proceso salvo `PYTHONHASHSEED`, que no quedó registrado | **`seed_base = SEED_RAIZ + PASO_COMBO × índice_de_combo`**, con **`SEED_RAIZ = 620000`** y **`PASO_COMBO = 5000`**. Función pura del índice. Sin `hash()`, sin reloj, sin entorno |
| columna `seed` en el CSV | **no se guardaba** | **se guarda** en los dos CSV (crudo y resumen). Cualquier regla se reconstruye con `GEN.generar_regla(A,B,C,idx,seed)` |
| medición del diámetro | sólo la vara vieja (`_diam` de cs055) | **las dos varas a la vez** sobre el mismo campo (`correr_regla_coarse_doble` de `cs090_diam_corregido.py`), sin correr la dinámica dos veces |

Detalle menor: el original corría en 2 pasadas de 5+5 con `seed_base` distinto, lo que hacía que los
`rule_id` se repitieran entre pasadas (había dos `...-r0` por combo). Acá van r0..r9 en una sola pasada y
son únicos. Verificado sobre el CSV: **150 filas, 150 `rule_id` únicos, 150 `seed` únicas.**

`PASO_COMBO=5000` es mayor que el desplazamiento máximo que puede usar el generador dentro de un combo
(20 intentos × 97 = 1940), así que dos combos nunca comparten semilla. `SEED_RAIZ=620000` cae fuera del
rango de semillas de todos los barridos previos de esta línea (~270k a ~590k): **este lote es una muestra
genuinamente nueva, no un re-etiquetado de una vieja.**

**Salidas:** `cs090_fase6_o2e_barrido_va_raw.csv` (750 filas = 150 reglas × 5 escalas),
`cs090_fase6_o2e_barrido_va_resumen.csv` (150 filas, una por regla, **con `seed`**),
`cs090_fase6_o2e_verificacion_a0_ejeC.csv` (15 filas). Tiempo total del barrido: **30,8 min**.

**Se reproduce el hallazgo metodológico del original:** los 3 combos **A0-B1-C0/C1/C2 siguen sin ser
ejecutables** con el motor congelado (`KeyError: 'adj'` — `dinamica_B1` lee `sustrato["adj"]`, que el
sustrato A0 por diseño no tiene). Quedan en 0/10, igual que en agosto. **Filtro P1-P5: 180 admitidas, 0
descartadas** (mismo comportamiento que el original: el hueco es del motor, no del generador).

---

## 2. Impacto DIRECTO del bug de diámetro sobre las 150 — ya no es inferencia

Éste era el punto de la tarea. `FASE6_adopcion_diam_corregido_CS.md` §3.1 sólo pudo inferir el impacto
sobre V-A con tres pruebas indirectas sobre los datos guardados (las tres dieron 0/150). Ahora está
**medido**, regla por regla, con las dos varas sobre el mismo campo:

| medición sobre las 150 reglas del lote nuevo | resultado |
|---|---|
| reglas que **CAMBIAN DE CLASE** (vieja → corregida) | **0 / 150 (0,0 %)** |
| reglas con descarrilamiento (componente medida < 10 % de la gigante, en alguna escala) | **0 / 150** |
| reglas donde el diámetro **REAL** difiere entre las dos varas (en alguna escala) | **0 / 150** |
| reglas donde el diámetro de **algún NULL** difiere | **2 / 150** |
| reglas donde la **pendiente** cambia | **0 / 150** |
| reglas con pendiente vieja **negativa** (la firma clásica del bug) | **0 / 150** |

Los dos únicos casos donde el bug movió *algún* número — ambos en **A2-B0-C2**, el combo más frágil, y
ambos en la escala b=1, la única con fragmentación apreciable:

| regla | seed | fracción gigante b=1 | diám NULL viejo → corregido | z_agg viejo → corregido | clase |
|---|---|---|---|---|---|
| A2-B0-C2-r2 | 690195 | 0,827 (24 componentes) | 14,33 → **20,67** | 0,648 → 0,811 | III → III |
| A2-B0-C2-r9 | 690874 | 0,949 (8 componentes) | 9,67 → **13,67** | 0,7778174583 → 0,7778174580 | III → III |

Nótese la dirección: la vara vieja **sub-medía el NULL** (medía un fragmento suelto del grafo aleatorio).
En un caso eso movió el z_agg en 0,16 — muy lejos del umbral z>3, así que no tocó la clase. En el otro el
efecto es del noveno decimal.

**Por qué el bug tenía tan poco dónde morder acá** (los dos indicadores de calibración de
`FASE6_adopcion_diam_corregido_CS.md` §3.1, ahora medidos sobre el lote nuevo):

- **diám(b=1) mínimo del lote (vara vieja) = 4.** Las 15 reglas confirmadas como descarriladas en el
  barrido de 430 tenían diám(b=1) ≤ 3; las 415 sanas tenían ≥ 8. Ninguna de las 150 cae en el rango de las
  descarriladas.
- **fracción de componente gigante a b=1, mínima del lote = 0,827.** El bug exige fragmentos sueltos
  grandes; con la gigante arriba del 82 % no hay dónde saltar. (Para las 30 reglas A0 la gigante va de
  0,9995 a 1,0000 — el grafo de medición derivado es prácticamente siempre conexo.)

**Esto confirma DIRECTAMENTE lo que las tres pruebas indirectas de §3.1 habían inferido.** Con una
limitación que hay que dejar escrita, sin adornos: se midió sobre **un lote nuevo de 150 reglas del mismo
diseño**, no sobre las 150 originales — ésas siguen siendo irrecuperables (semillas perdidas). Lo que
queda probado es *"en esta región del espacio de parámetros y con este pipeline, el bug no muerde"*, no
*"aquellas 150 reglas concretas estaban intactas"*. Sumado a las tres pruebas indirectas hechas sobre los
datos guardados del lote original, la evidencia va en la misma dirección desde dos lados independientes.

---

## 3. El mapa global recalculado

Columna "corregida" = vara nueva. Columna "vieja" = vara de cs055, **sobre las mismas reglas del lote
nuevo**. Columna "publicado" = `FASE5A_completo_resultado_CS.md` §2.

| combo | n | I / II / III / IV (corregida) | I / II / III / IV (vieja) | publicado |
|---|---|---|---|---|
| A0-B0-C0 | 10 | 9 / 1 / 0 / 0 | 9 / 1 / 0 / 0 | 5 / 5 / 0 / 0 |
| A0-B0-C1 | 10 | 8 / 2 / 0 / 0 | 8 / 2 / 0 / 0 | 9 / 1 / 0 / 0 |
| A0-B0-C2 | 10 | 9 / 1 / 0 / 0 | 9 / 1 / 0 / 0 | 8 / 2 / 0 / 0 |
| A0-B1-C0/C1/C2 | 0 | — | — | no ejecutable |
| A1-B0-C0 | 10 | 7 / 3 / 0 / 0 | 7 / 3 / 0 / 0 | 6 / 4 / 0 / 0 |
| A1-B0-C1 | 10 | 6 / 4 / 0 / 0 | 6 / 4 / 0 / 0 | 6 / 4 / 0 / 0 |
| A1-B0-C2 | 10 | 5 / 3 / **2** / 0 | 5 / 3 / 2 / 0 | 5 / 2 / **3** / 0 |
| A1-B1-C0 | 10 | 6 / 4 / 0 / 0 | 6 / 4 / 0 / 0 | 8 / 2 / 0 / 0 |
| A1-B1-C1 | 10 | 6 / 4 / 0 / 0 | 6 / 4 / 0 / 0 | 7 / 3 / 0 / 0 |
| A1-B1-C2 | 10 | 8 / 2 / 0 / 0 | 8 / 2 / 0 / 0 | 7 / 3 / 0 / 0 |
| A2-B0-C0 | 10 | 8 / 2 / 0 / 0 | 8 / 2 / 0 / 0 | 6 / 4 / 0 / 0 |
| A2-B0-C1 | 10 | 8 / 2 / 0 / 0 | 8 / 2 / 0 / 0 | 8 / 2 / 0 / 0 |
| A2-B0-C2 | 10 | 4 / 1 / **4** / **1** | 4 / 1 / 4 / 1 | 5 / 0 / **5** / 0 |
| A2-B1-C0 | 10 | 5 / 5 / 0 / 0 | 5 / 5 / 0 / 0 | 7 / 3 / 0 / 0 |
| A2-B1-C1 | 10 | 5 / 5 / 0 / 0 | 5 / 5 / 0 / 0 | 5 / 5 / 0 / 0 |
| A2-B1-C2 | 10 | 6 / 4 / 0 / 0 | 6 / 4 / 0 / 0 | 7 / 3 / 0 / 0 |

| distribución global | I | II | III | IV | intermedio |
|---|---|---|---|---|---|
| **O2-E, vara corregida** | **100 (67 %)** | **43 (29 %)** | **6 (4 %)** | **1 (1 %)** | 0 |
| O2-E, vara vieja | 100 (67 %) | 43 (29 %) | 6 (4 %) | 1 (1 %) | 0 |
| publicado (10-ago) | 99 (66 %) | 43 (29 %) | 8 (5 %) | 0 | 0 |

Las dos primeras filas son **idénticas celda por celda en el mapa entero**. La tercera difiere en
{I: +1, III: −2, IV: +1}.

---

## 4. Las tres conclusiones publicadas, recalculadas

### (a) "A0 nunca alcanza Clase II o superior" — ¿sigue falsificada?

| | Clase II+ sobre 30 reglas A0 |
|---|---|
| O2-E, vara corregida | **4 / 30 (13 %)** — {I: 26, II: 4, III: 0, IV: 0} |
| O2-E, vara vieja | 4 / 30 (13 %) — idéntico |
| publicado | 8 / 30 (27 %) |

**Sigue falsificada** en el sentido literal de la afirmación (4 > 0: hay reglas A0 en Clase II), pero **con
la mitad de fuerza que la cifra publicada**. La diferencia 4/30 vs 8/30 no es distinguible del muestreo
(Fisher exacto dos colas, **p = 0,334**). **Cambio por el bug: 0/30** — y era esperable, la gigante de las
A0 va de 0,9995 a 1,0000: el bug no tiene dónde morder.

Tres advertencias sobre cómo leer esa fila, todas de `FASE6_O1C_cierre_A0_CS.md`:

1. **La etiqueta Clase II en A0 no es reproducible.** Re-medir el **mismo campo** (mismo array `S` bit a
   bit) cambiando sólo las semillas de medición devuelve Clase II sólo el **45,6 %** de las veces (1 de 15
   reglas es 100 % estable). Con esa tasa, que un lote dé 8/30 y otro 4/30 es exactamente lo que se
   espera. Ninguno de los dos números es "el" número de A0.
2. **El Eje C no existe dentro de A0** (verificado acá, §5). Las tres filas A0-B0-* del mapa **no son tres
   celdas independientes**: son la misma condición sorteada tres veces con semillas distintas. Leídas
   correctamente son **30 réplicas de UNA sola configuración**, no 3 celdas de 10.
3. Con esa lectura, la mayor discrepancia del mapa entero se disuelve: A0-B0-C0 pasó de 5/10 a 1/10
   (Fisher p = 0,141 tomada aislada), pero **A0-B0-C0/C1/C2 es una sola celda**, y agregada da 4/30 hoy
   contra 8/30 en agosto — la misma comparación de arriba, p = 0,334.

### (b) "Muy fuerte" CONTRADICHO — ¿B0+C2 sigue superando a B1+C1/C2?

| | Clase III/IV |
|---|---|
| O2-E, vara corregida | **B0+C2: 7/30** · **B1+C1/C2: 0/40** |
| O2-E, vara vieja | B0+C2: 7/30 · B1+C1/C2: 0/40 |
| publicado | B0+C2: 8/30 (A1-B0-C2 3/10 · A2-B0-C2 5/10) · B1+C1/C2: 0/40 |

**Se sostiene, y es la conclusión que se replica más limpio de las tres.** El patrón es casi calcado:
las Clases III/IV siguen apareciendo **sólo** en los dos mismos combos (A1-B0-C2 con 2/10 y A2-B0-C2 con
5/10 = 4·III + 1·IV), los dos con **B0**; y las cuatro celdas que cruzan B1 con C1/C2 vuelven a dar **0 de
40**. La hipótesis del pre-registro ("III/IV mayoritaria cuando B1 y C1/C2 están juntos") sigue apuntando
al revés en esta muestra. **Cambio por el bug: 0.**

### (c) "0 reglas en Clase IV en todo V-A" / "B0 nunca alcanza Clase IV"

| | Clase IV |
|---|---|
| O2-E, vara corregida | **1 / 150** (y es una regla **B0**) |
| O2-E, vara vieja | 1 / 150 (la misma) |
| publicado | 0 / 150 |

**Es la única de las tres que NO se replica** — y no por el bug (la vara vieja también la marca IV), sino
por muestreo. La regla es **A2-B0-C2-r3, seed 690292**, y hay que decir que entra a Clase IV **rozando los
dos umbrales a la vez**: pendiente **0,7016** (umbral > 0,7) y holonomía NULL/REAL **5,21** (umbral ≥ 5,0).
Es un caso de borde en las dos coordenadas simultáneamente, no un ejemplar nítido.

Dato de contexto que conviene tener a mano al leer esa Clase IV: **el holon_ratio máximo entre las reglas
de Clase I/II del lote es 7,26** — más alto que el 5,21 del único ejemplar de Clase IV. O sea: el criterio
de holonomía no está ordenando reglas por sí solo; funciona como una compuerta que se aplica *después* de
pasar el filtro de Clase III, y hay reglas con holonomía más "cerrada" que nunca llegan a que se les
aplique. Eso es diseño del clasificador pre-registrado, no un hallazgo nuevo, pero cambia cuánto peso
darle a "apareció una Clase IV".

### (d) Criterio "débil" (>15 % de II+III por combo)

| | combos que superan el 15 % |
|---|---|
| O2-E, corregida | 13 / 15 |
| O2-E, vieja | 13 / 15 |
| publicado | 14 / 15 |

Se sigue cumpliendo ampliamente. Los dos que no llegan son A0-B0-C0 y A0-B0-C2, con 1/10 — otra vez la
celda A0, la de etiqueta menos reproducible.

---

## 5. La verificación del Eje C dentro de A0

O1-C había encontrado leyendo el código que en `cs090_fase5_motor.py` (líneas 190-199) `dinamica_B0`
retorna en la rama del campo en anillo **antes** de cualquier bloque de costo (`if kind == "A0": ...
return sustrato`), así que C0/C1/C2 no pueden hacer nada. Acá se verificó **numéricamente** en vez de
darlo por bueno: se corrieron A0-B0-C0, A0-B0-C1 y A0-B0-C2 con la **misma semilla** y se compararon las
filas resultantes campo por campo.

**Resultado: 5/5 semillas idénticas bit a bit** (`cs090_fase6_o2e_verificacion_a0_ejeC.csv`, columna
`identico_a_C0` = True en las 15 filas). Mismo diámetro por escala, mismo nº de aristas, misma pendiente,
mismo z_agg, misma clase.

| seed | clase | pendiente | diám por escala | aristas b=1 | C1 ≡ C0 | C2 ≡ C0 |
|---|---|---|---|---|---|---|
| 810000 | II | +0,3630 | 5\|4\|3\|3\|2 | 12746 | True | True |
| 810137 | I | +0,2854 | 6\|5\|4\|4\|3 | 9767 | True | True |
| 810274 | I | +0,3024 | 4\|3\|3\|2\|2 | 16755 | True | True |
| 810411 | I | +0,2335 | 4\|3\|3\|3\|2 | 14488 | True | True |
| 810548 | I | +0,2881 | 4\|3\|3\|2\|2 | 20620 | True | True |

**Consecuencia para leer el mapa:** las 3 filas A0-B0-* no aportan 3 grados de libertad. En el diseño
nominal de 18 celdas, 2 son duplicados exactos de una tercera. El barrido "18×10" es en realidad, en
términos de condiciones experimentales distintas, **16 condiciones** (13 celdas medibles reales + 3 no
ejecutables), con la condición A0-B0 sobre-representada 3 a 1.

---

## 6. Bug vs muestreo — la comparación que importa

Ésta es la separación que el barrido original no podía hacer y que era el objetivo de O2-E.

**"Cambió por el BUG"** = clase vieja vs clase corregida, **sobre las mismas 150 reglas nuevas**, mismo
campo, misma dinámica, misma semilla. Es una comparación pareada perfecta: lo único que varía es la vara.

> **0 de 150 reglas (0,0 %) cambian de clase.** Global vieja [100, 43, 6, 1] → corregida [100, 43, 6, 1].
> Idéntico en las 15 celdas del mapa. La pendiente no se mueve en ninguna regla. Sólo 2 reglas tienen
> *algún* número distinto, y es en el NULL, y no las mueve de clase.

**"Cambió por la MUESTRA"** = este lote medido con la **vara vieja** (o sea, exactamente el mismo
instrumento del barrido de agosto) contra los números publicados. Todo lo que difiera acá es muestreo puro,
porque el instrumento es literalmente el mismo.

> Este lote con vara vieja: [100, 43, 6, 1]. Publicado: [99, 43, 8, 0]. Diferencia {I: +1, II: 0, III: −2,
> IV: +1}.
> **χ² (I / II / III+IV) = 0,14, gl = 2, p = 0,9347** — observado [100, 43, 7] contra esperado [99, 43, 8].

**Lectura de las dos juntas:** el 100 % de la diferencia entre este barrido y el publicado es atribuible a
que es otra muestra; el 0 % al bug de diámetro. Y esa diferencia de muestra, a nivel global, es de las más
pequeñas que se podían obtener (p = 0,93 es prácticamente coincidencia perfecta). A nivel de celda
individual, con n=10, sí se mueven cosas: la mayor es A0-B0-C0 (5/10 → 1/10 en Clase II, Fisher p = 0,141),
que se disuelve al agregar las tres A0-B0-* como corresponde (§5).

**Cuánto puede moverse una celda con n=10 — dos referencias para calibrar la lectura:**

- El test-retest de O1-C mide una desviación estándar de la pendiente, **entre re-mediciones del mismo
  campo**, de sd ≈ 0,033 (máx 0,066), contra un ancho de banda de Clase II de 0,10.
- Aplicando ese ±0,033 a las 150 pendientes de este lote, **55 de 150 reglas (37 %) están lo bastante cerca
  de un umbral como para cambiar de clase con sólo re-medir**: 5/30 en A0, 22/60 en A1, 28/60 en A2.
  (Advertencia: ese sd fue medido sobre A0; extrapolarlo a A1/A2 no está verificado — es una calibración
  aproximada, no una medición.)

Es decir: la razón por la que las celdas se mueven entre lotes no es misteriosa. **Es que la clasificación
en 4 cajas por umbrales duros de pendiente convierte una variable continua y ruidosa en una etiqueta
discreta**, y una porción grande de las reglas vive cerca de las paredes.

---

## 7. Dos observaciones adicionales que salieron al analizar (no estaban pedidas)

1. **`z_sostenido` es True en 0 de 150 reglas.** La Clase III se puede alcanzar por dos caminos según el
   clasificador pre-registrado: pendiente > 0,7 **o** separación z > 3 sostenida contra el NULL en todas
   las escalas. En este lote (y, por las pendientes publicadas, presumiblemente también en el de agosto)
   **el segundo camino nunca se activa**: las 7 reglas de Clase III/IV llegaron todas por la pendiente. La
   rama del z sostenido está, de hecho, inerte en Fase V-A.
2. **Las Clases III se apilan contra su umbral.** Las 7 pendientes de III/IV son 0,702 · 0,705 · 0,711 ·
   0,737 · 0,770 · 0,808 · 1,133. **Tres de las siete están a menos de 0,012 del umbral 0,7.** Sólo la de
   1,133 está holgadamente adentro. Con el sd de re-medición de arriba, el conteo "6 III + 1 IV" es
   bastante sensible al acto de medir.

---

## 8. Archivos de esta tarea

- `cs090_fase6_o2e_barrido_va_reproducible.py` — driver nuevo (único archivo de código; importa sin tocar
  `cs090_fase5_generador`, `cs090_fase5_motor`, `cs090_fase5_clasificador`, `cs090_diam_corregido`).
- `cs090_fase6_o2e_barrido_va_raw.csv` — 750 filas (150 reglas × 5 escalas), con `seed`, con las dos
  mediciones de diámetro REAL y NULL en paralelo.
- `cs090_fase6_o2e_barrido_va_resumen.csv` — 150 filas, una por regla, **con `seed`**, con `clase_vieja`,
  `clase_corregida` y `cambia_clase`.
- `cs090_fase6_o2e_verificacion_a0_ejeC.csv` — 15 filas (5 semillas × C0/C1/C2).
- `cs090_fase6_o2e_barrido.log` — log completo de la corrida.
- Este informe.

De aquí en adelante, cualquier regla de este lote se reconstruye sola: el barrido de Fase V-A **deja de
ser el único resultado no re-medible de la línea**. Ningún script congelado fue modificado, ningún CSV
existente fue tocado, no se corrió Phantom, no se hicieron commits. No se declara cierre.
