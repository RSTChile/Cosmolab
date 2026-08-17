# FASE VI — O1-C: la pregunta de A0 con muestra suficiente (400 reglas)

**Fecha:** 11-ago-2026 · **Ejecuta:** CC (Claude) · **Plan:** `FASE6_PLAN_EJECUCION_COMPLETA_CS.md`, tarea O1-C
**Antecedentes directos:** `FASE5A_completo_resultado_CS.md` §3 (el 27% de A0 en Clase II) y
`FASE5_A0_metricas_nativas_CS.md` (métricas nativas, pero con sólo 2 Clase II — sin poder estadístico).

No se declara cierre ni veredicto. Se reportan números; la lectura final es de Alexis. No se corrió Phantom.
No se editó ningún script congelado ni `cs090_fase5_a0_nativo.py` (se importan y reusan).

---

## 0. Qué se pedía y qué se hizo

Se pedía: juntar **≥12-15 reglas A0 que el método viejo clasifique Clase II** más un control de Clase I,
medir cada una con los dos métodos (viejo y nativo), y correr un test formal (KS, más Mann-Whitney).

Se hizo eso y tres cosas más que la evidencia fue pidiendo sobre la marcha:

| # | Qué | Resultado en una línea |
|---|---|---|
| 1 | Barrido de **400 reglas** A0-B0-C0 (filtro P1-P5 real) | **70 Clase II** y 330 Clase I — 4-5x la muestra pedida |
| 2 | Test formal KS + Mann-Whitney + permutación + Cliff's δ + Holm | 2 de 5 métricas nativas separan con efecto **chico** (δ≈0.20-0.25); las otras 3, nada |
| 3 | **Chequeo de confound** (no estaba pedido) | los dos grupos difieren en el **umbral `sim_thr_frac`** (p<1e-4), parámetro que **los dos métodos comparten** |
| 4 | **Test-retest** (no estaba pedido) | re-medir el **mismo campo** con otras semillas devuelve Clase II sólo el **45.6%** de las veces |

**Tiempo de cómputo:** 12.2 min (barrido) + 7.6 min (test-retest) + ~2 min (análisis) ≈ 22 min.

---

## 1. Cuántas reglas hubo que generar — y la tasa base bien medida

- **400 reglas A0-B0-C0 generadas, 400 admitidas, 0 descartadas** por el filtro P1-P5 (el filtro corre de
  verdad: `GEN.generar_reglas_clase` con `max_intentos=1200`; simplemente ninguna regla A0 falla P1-P5).
- **Clase II: 70/400 = 17.5%** (intervalo binomial 95% aproximado: **13.8% – 21.2%**).
- Clase I: 330/400. **Ninguna** regla llegó a Clase III, Clase IV ni a "intermedio".

Esto pone en contexto las dos estimaciones previas, que parecían contradecirse:

| corrida | n | Clase II | tasa |
|---|---|---|---|
| Barrido de 180 (`FASE5A_completo`) | 30 (A0-B0-C0/C1/C2) | 8 | 27% |
| Métricas nativas (`FASE5_A0_metricas_nativas`) | 35 | 2 | 5.7% |
| **O1-C (esta corrida)** | **400** | **70** | **17.5%** |

Las dos anteriores son compatibles con 17.5% por puro azar de muestreo (con n=30 y p=0.175, ver 8 o más
Clase II no es raro; con n=35, ver 2 o menos tampoco lo es). **No hacía falta explicar la diferencia: era
ruido de muestra chica.**

### 1.1 Un dato que corrige la lectura del informe original (verificado, no supuesto)

`FASE5A_completo_resultado_CS.md` reporta A0-B0-**C0** 5/10 = 50%, A0-B0-**C1** 1/10 = 10%, A0-B0-**C2**
2/10 = 20%, y comenta que A0-B0-C0 es una de las combinaciones "con más señal". **Esa variación no puede
ser un efecto del Eje C**: en `cs090_fase5_motor.dinamica_B0`, la rama `kind == "A0"` hace su bucle de
difusión y **hace `return` antes de tocar cualquier bloque de costo/poda**, así que el eje C no entra
nunca en la dinámica de A0. Verificado empíricamente en la corrida (`verificar_combos_A0`, imprime en el
log): con la misma semilla, el campo de A0-B0-C1 y el de A0-B0-C2 son **idénticos bit a bit** al de
A0-B0-C0 (`np.array_equal` → True en los dos casos). El 50% vs 10% vs 20% del informe original es
dispersión de muestreo entre tres lotes de 10, no una diferencia entre combinaciones.

Corolario operativo, y por eso el barrido de O1-C es todo A0-B0-C0: **el Eje A0 tiene una sola combinación
con dinámica propia.** A0-B1-* no existe (`KeyError: 'adj'`, ya documentado en `FASE5A_completo` §1, y
re-verificado acá), y A0-B0-C1/C2 son la misma dinámica que C0. Juntar la muestra combinando ejes habría
sido duplicar filas con etiquetas distintas.

---

## 2. El bug de diámetro NO afecta a esta línea (comprobado en las 400)

La tarea pedía verificar si `cs090_diam_corregido.py` cambia algo acá. Se midió **cada regla con las dos
versiones a la vez** (`correr_regla_coarse_doble`), en las 5 escalas, en REAL y en los tres NULL_topo:

- reglas con alguna escala **descarrilada** (la versión histórica midiendo una componente <10% de la
  gigante): **0 de 400**;
- reglas cuya **clase cambia** con el diámetro corregido: **0 de 400**;
- **|Δ pendiente| entre las dos versiones: 0.00000 exacto**, media y máximo.

Motivo, no casualidad: el grafo de medición derivado de A0 es **denso y de una sola pieza** — ~10.000 a
19.000 aristas sobre 2.000 nodos, `n_componentes = 1`, `n_aislados = 0` en todas las escalas. El bug sólo
aparece cuando el grafo se fragmenta y el nodo de índice más bajo cae en un pedacito suelto; acá no hay
pedacitos sueltos. **Todo lo que sigue vale igual para las dos versiones del método viejo** (por eso el
análisis no las reporta por separado: sería la misma tabla dos veces).

*Verificación adicional de que el envoltorio no altera el método viejo:* se comprobó que
`correr_regla_coarse_doble(...)[0]` devuelve exactamente los mismos valores que
`cs090_fase5_motor.correr_regla_coarse(...)` para las mismas reglas — todos los campos idénticos salvo
`dt`, que es tiempo de reloj.

---

## 3. Control NULL — las métricas nativas siguen viendo estructura (punto 4 de la tarea)

Campo REAL contra el **mismo campo con las posiciones barajadas** (misma distribución de valores, orden
espacial destruido), sobre las 400 reglas nuevas:

| métrica nativa | REAL (media, rango) | NULL barajado | REAL > NULL en | Wilcoxon pareado |
|---|---|---|---|---|
| pendiente de dominio local | 1.471 [0.494, 2.314] | 0.135 [-0.231, 0.390] | **400/400** | p = 2.7e-67 |
| pendiente de ξ(r) | 0.508 [0.300, 0.720] | **0.000** exacto | **400/400** | p = 6.8e-71 |
| fracción en dominio mayor (b=1) | 0.368 [0.011, 1.000] | 0.006 [0.003, 0.013] | **400/400** | p = 2.7e-67 |
| nº de dominios (b=1) | 50.1 [1, 572] | 1012.5 [574, 1414] | 0/400 (menos, como debe) | p = 2.7e-67 |

Separación total, en las 400, en las cuatro columnas. **Las métricas nativas no son ciegas**: cualquier
"no hay diferencia entre Clase I y Clase II" que aparezca abajo no se puede atribuir a un instrumento
que no mide nada.

---

## 4. La comparación central — Clase II vs Clase I en métricas nativas

n = 330 (Clase I) vs 70 (Clase II). "perm p" = permutación de etiquetas, 20.000 remuestreos (importa
porque ξ(r) tiene empates masivos y ahí el KS asintótico se vuelve conservador). "Holm" corrige por las
5 métricas testeadas.

| métrica nativa | media I | media II | KS D | **KS p** | MW p | perm p | Cliff δ | KS p (Holm) |
|---|---|---|---|---|---|---|---|---|
| pendiente de ξ(r) | 0.5091 | 0.5032 | 0.084 | **0.771** | 0.462 | 0.511 | +0.050 | 1.000 |
| long. de correlación ξ (b=1) | 4.003 | 3.914 | 0.084 | **0.771** | 0.405 | 0.338 | +0.054 | 1.000 |
| pendiente de dominio local | 1.494 | 1.360 | 0.195 | **0.021** | 0.008 | 0.010 | +0.201 | 0.064 |
| fracción en dominio mayor (b=1) | 0.395 | 0.245 | 0.231 | **0.003** | 0.001 | 0.001 | +0.248 | **0.014** |
| nº de dominios (b=1) | 48.1 | 59.4 | 0.264 | **0.0005** | 0.0006 | 0.308 | −0.260 | **0.003** |

**Lectura directa, sin adornos: la respuesta no es un "no" limpio.** Tres cosas al mismo tiempo:

1. **Las dos métricas de ξ(r) no ven absolutamente nada** (D = 0.084 cuando el D detectable a α=0.05 con
   estas n es 0.179 — el efecto observado es menos de la mitad del mínimo detectable; δ = 0.05, "insignificante").
2. **Las tres métricas de dominio sí separan**, con p pequeños pero **tamaño de efecto chico**
   (δ = 0.20-0.26; la convención de Romano llama "chico" a 0.147-0.33). El KS mide que las distribuciones
   son distinguibles, no que estén separadas: en el panel (b) del gráfico las dos curvas van pegadas casi
   todo el recorrido.
3. **La dirección es la contraria a la que "mundo-pequeño congelado" sugeriría**: las reglas Clase II
   tienen dominios **más chicos** (0.245 vs 0.395), pendiente de dominio **más baja** y **más** dominios
   (59.4 vs 48.1). O sea, si las métricas nativas dijeran algo sobre la clase, dirían que Clase II es
   **menos** coherente espacialmente, no más.

**Potencia (para que "p>0.05" signifique algo):** por simulación bootstrap con estas n, un desplazamiento
de 0.5σ en la pendiente de dominio se detecta el 92-95% de las veces, y uno de 0.25σ en la fracción de
dominio mayor, ~100%. El estudio tenía potencia de sobra; los "no" de ξ(r) no son falta de n.

### 4.1 Y acá aparece el confound: el umbral que los dos métodos comparten

Antes de leer el punto 2 de arriba como "el campo continuo SÍ hace algo", hay que mirar los **parámetros**
de las reglas de cada grupo:

| parámetro | Clase I | Clase II | Mann-Whitney p |
|---|---|---|---|
| K (alfabeto de fase) | 6.139 | 6.257 | 0.487 |
| J (acople) | 0.549 | 0.524 | 0.195 |
| ruido | 0.2224 | 0.2227 | 0.925 |
| **sim_thr_frac (umbral de similitud)** | **0.2538** | **0.2117** | **< 1e-4** |
| **thr absoluto = sim_thr_frac · K** | **1.558** | **1.317** | **0.0002** |
| aristas del grafo derivado (b=1) | 15.155 | 12.652 | < 1e-4 |

`sim_thr_frac` **no es una propiedad del campo: es un parámetro de la regla**, y lo usan **los dos
métodos**: el grafo derivado del método viejo conecta si la diferencia de fase < `thr`, y
`dominios_locales` (nativo) corta el dominio si la diferencia de fase ≥ el **mismo** `thr`. Con lo cual
las tres métricas de dominio están *obligadas* a moverse con `thr`, y de hecho lo hacen:

| relación | Spearman ρ | p |
|---|---|---|
| thr → nº de aristas del grafo derivado | **+0.779** | 1e-82 |
| nº de aristas → diámetro a b=1 | **−0.891** | 2e-138 |
| thr → fracción en dominio mayor (nativa) | **+0.650** | 2e-49 |
| thr → pendiente de dominio (nativa) | **+0.558** | 4e-34 |
| thr → nº de dominios (nativa) | **−0.672** | 8e-54 |
| **thr → pendiente de ξ(r) (nativa)** | **−0.001** | **0.98** |

La última fila es la clave: **ξ(r) es la única métrica nativa que NO usa el umbral** (`correlacion_circular`
no recibe `thr`), y es exactamente la única que no correlaciona con el umbral **y** la única que no
distingue Clase I de Clase II. Las que sí comparten el parámetro, separan — en la dirección que el
parámetro predice.

**Prueba directa del confound — comparación emparejada.** Para cada regla Clase II se buscó la regla
Clase I con el `thr` más parecido (70 pares, |Δthr| medio = 0.0028, máximo 0.016). Con el umbral igualado:

| métrica nativa | Clase I (emparejada) | Clase II | Wilcoxon pareado p |
|---|---|---|---|
| pendiente de ξ(r) | 0.5196 | 0.5032 | 0.264 |
| pendiente de dominio local | 1.4658 | 1.3596 | **0.083** |
| fracción en dominio mayor | 0.3124 | 0.2451 | **0.093** |
| long. de correlación ξ | 4.114 | 3.914 | 0.111 |
| nº de dominios (b=1) | 39.46 | 59.44 | **0.028** |

Al igualar el umbral, las dos diferencias más fuertes de la tabla anterior (p = 0.003 y 0.021) **se caen
por encima de 0.05**. La única que queda es el nº de dominios con p = 0.028 sin corregir — que sobre una
familia de 5 tests da Holm ≈ 0.14. Es decir: **lo que separaba a los grupos en las métricas de dominio se
explica, dentro del poder de esta muestra, por el parámetro que los dos métodos comparten.**

### 4.2 La versión continua, sin dicotomía (el test más limpio)

El corte 0.35 es arbitrario. Si "Clase II" midiera algo del campo, la **pendiente vieja como variable
continua** debería correlacionar con las métricas nativas. Sobre las 400 reglas:

| métrica nativa | Spearman ρ | p | ρ parcial (descontando thr) | p |
|---|---|---|---|---|
| pendiente de ξ(r) | −0.047 | 0.347 | −0.047 | 0.347 |
| pendiente de dominio | −0.072 | 0.149 | −0.067 | 0.184 |
| fracción dominio mayor | −0.077 | 0.126 | −0.075 | 0.136 |
| long. de correlación ξ | −0.045 | 0.367 | −0.045 | 0.370 |
| nº de dominios | +0.073 | 0.144 | +0.071 | 0.157 |

**Todas indistinguibles de cero** (|ρ| ≤ 0.077, ninguna p < 0.12), con n=400. La pendiente del método
viejo tampoco correlaciona con los parámetros de la regla de forma monótona (|ρ| < 0.09 con
sim_thr_frac, K, J, ruido y nº de aristas). La razón está en la sección siguiente: la pendiente vieja no
es una función monótona de nada, es una función **escalonada** de un puñado de enteros.

---

## 5. De dónde sale realmente la "pendiente vieja": una tupla de enteros chicos

El diámetro de un grafo es un **entero chico**. Acá, en las 5 escalas de coarse-graining (b=1,2,4,8,16),
los diámetros valen entre 2 y 7. La pendiente que decide la clase es la recta ajustada a **cinco enteros
chicos** — hay poquísimas combinaciones posibles. Tabulando qué tupla tiene cada regla:

| diámetros (b=1,2,4,8,16) | pendiente | n Clase I | n Clase II | total |
|---|---|---|---|---|
| (4, 3, 3, 2, 2) | 0.295 | 147 | 0 | 147 |
| **(5, 4, 3, 3, 2)** | **0.364** | **0** | **52** | **52** |
| (4, 3, 3, 3, 2) | 0.229 | 50 | 0 | 50 |
| (5, 4, 3, 3, 3) | 0.234 | 26 | 0 | 26 |
| (4, 4, 3, 3, 2) | 0.281 | 16 | 0 | 16 |
| (5, 4, 4, 3, 3) | 0.230 | 13 | 0 | 13 |
| (6, 4, 4, 3, 3) | 0.307 | 12 | 0 | 12 |
| **(5, 4, 4, 3, 2)** | **0.361** | **0** | **8** | **8** |
| … 28 combinaciones más | | | | |

**36 combinaciones distintas en 400 reglas; 33 de ellas son "puras"** (todas sus reglas caen en la misma
clase). **Una sola tupla, (5,4,3,3,2), concentra 52 de las 70 Clase II (74%)**; con (5,4,4,3,2) se llega a
86%. Y su vecina más poblada, (4,3,3,2,2) con 147 reglas, cae en Clase I con pendiente 0.295.

Dicho de otro modo: **el "Clase II" de A0 es, en la práctica, "al grafo derivado le salió diámetro 5 en
vez de 4 en la escala fina"**:

| diámetro a b=1 | n reglas | Clase II | tasa |
|---|---|---|---|
| 4 | 217 | 1 | **0.5%** |
| 5 | 115 | 61 | **53%** |
| 6 | 60 | 4 | 7% |
| 7 | 8 | 4 | 50% |

Y ese diámetro está fijado por la densidad del grafo derivado (ρ = −0.891), que está fijada por el umbral
de la regla (ρ = +0.779). La cadena completa es: **parámetro `thr` → densidad del grafo que el método
fabrica → diámetro entero → tupla → clase.** El campo no aparece en ningún eslabón.

**Consecuencia práctica**, visible en el panel (f) del gráfico: por encima de thr = 2.1 (59 reglas) el
grafo derivado es tan denso que el diámetro es 4 siempre y **nunca sale Clase II — 0%**. La tasa de Clase
II de una regla A0 se puede predecir mirando su umbral, sin correr la dinámica.

---

## 6. Test-retest: la misma bolsa de harina, pesada seis veces

Esta prueba no estaba pedida, pero es la más directa. Se **congela el campo** (mismo sustrato, misma
dinámica, mismo array `S` bit a bit) y se repite **sólo la medición** 6 veces, cambiando únicamente las
semillas de: qué 15 candidatos al azar mira cada sitio, qué cajas arma el BFS del coarse-graining, y qué
grafos ER salen de NULL_topo. 15 reglas etiquetadas Clase II y 15 etiquetadas Clase I.

| grupo original | fracción de réplicas que dan Clase II | reglas 100% estables (6/6 o 0/6) |
|---|---|---|
| etiquetadas **Clase II** | **0.456** (rango por regla 0.17 – 1.00) | **1 de 15** |
| etiquetadas **Clase I** | 0.044 (rango 0.00 – 0.33) | 12 de 15 |

- De las 90 re-mediciones de reglas "Clase II", **41 dieron Clase II y 49 dieron Clase I** — el mismo
  campo, medido de nuevo.
- Sólo **4 de 15** reglas Clase II repiten la etiqueta en 4 o más de 6 réplicas; **8 de 15** la repiten
  en 2 o menos.
- La pendiente de una MISMA regla varía entre réplicas con **sd media = 0.033 (máx 0.066)**. La banda
  entera que define Clase II mide 0.45 − 0.35 = **0.10**. O sea: **el ruido de medición es un tercio del
  ancho de la categoría** (y hasta dos tercios en el peor caso).
- Pero **no es puro azar**: la propensión difiere entre grupos (0.456 vs 0.044, Mann-Whitney p = 1.1e-05).
  Hay algo sistemático que hace que ciertas reglas caigan más seguido del lado II — y por la §5, ese algo
  es el umbral/densidad, no el campo.

**Traducción:** "Clase II" en A0 no es una propiedad discreta de la regla que la medición revela; es un
**resultado probabilístico de cada acto de medición**, con una propensión que depende de un parámetro de
la regla. Etiquetar una regla A0 como "Clase II" a partir de una sola medición es como decidir si una
moneda "es cara" tirándola una vez.

---

## 7. Lectura honesta, en orden de solidez (sin cerrar nada)

1. **Las métricas nativas funcionan** (400/400 separan campo real de campo barajado, p ~ 1e-67). Nada de
   lo que sigue es "el instrumento no mide".
2. **El bug de diámetro no toca esta línea**: 0/400 descarrilamientos, 0/400 cambios de clase, Δpendiente
   exactamente 0. Los resultados valen igual con la medición histórica y con la corregida.
3. **La única métrica nativa que no comparte parámetros con el método viejo — ξ(r) — no distingue Clase I
   de Clase II** (D = 0.084 contra 0.179 detectable; δ = 0.05), y su versión continua tampoco (ρ = −0.047).
4. **Las tres métricas de dominio sí distinguen, con efecto chico (δ 0.20-0.26) y en dirección contraria
   a "más estructura"** — y esa diferencia **se cae al emparejar por el umbral compartido** (p pasa de
   0.003/0.021 a 0.093/0.083). Es consistente con mediación por el parámetro, no con estructura del campo.
5. **La pendiente vieja, como variable continua, no correlaciona con NINGUNA métrica nativa** (|ρ| ≤ 0.077,
   n = 400) ni con ningún parámetro de forma monótona — porque es una función escalonada de una tupla de
   5 enteros chicos, de la que una sola combinación concentra el 74% de las Clase II.
6. **La etiqueta no es reproducible**: re-medir el mismo campo la devuelve el 45.6% de las veces.

**Qué NO se puede afirmar con esto.** (a) No se probó que el campo A0 no pueda producir mundo-pequeño en
ningún sentido; se probó que **esta** medición, aplicada a **este** sustrato, no está midiendo eso.
(b) El residuo del nº de dominios emparejado (p = 0.028 sin corregir) queda ahí: es chico y no sobrevive
Holm, pero no es cero. (c) Todo esto es a N = 2000, 14 sweeps, un campo por regla. (d) Nada de esto se
transfiere automáticamente a A1/A2, donde el grafo **es** el sustrato y no hay grafo derivado: ahí el
método viejo mide un objeto que existe. **La pregunta de si el 27% original era artefacto era específica
de A0, y sólo sobre A0 hablan estos números.**

---

## 8. En lenguaje simple

Imaginate 2.000 personas en una ronda, cada una susurrándole sólo a los dos vecinos de al lado. Esa es la
receta A0: no hay teléfonos, no hay atajos, el chisme sólo puede caminar de vecino en vecino.

Para decir "acá el mundo es chico", el método viejo hace esto: agarra a cada persona, la conecta por
teléfono con 15 desconocidos elegidos **al azar** de cualquier parte de la ronda si por casualidad dicen
algo parecido, y después mide cuántos saltos hacen falta para ir de cualquiera a cualquiera **por esa red
de teléfonos que acaba de instalar él mismo**. Si le da 5 saltos en vez de 4, dice "mundo pequeño". En
esta corrida eso pasó 70 veces de 400.

Tres hallazgos, cada uno más incómodo que el anterior:

- **El número de saltos depende de cuántos teléfonos instaló, no de qué se dice en la ronda.** Y cuántos
  teléfonos instala está fijado por una perilla de la regla (qué tan parecido tiene que sonar el susurro
  para conectar). Con la perilla alta instala tantos teléfonos que siempre da 4 saltos y **nunca**
  aparece un "mundo pequeño"; con la perilla baja aparece una de cada tres veces.
- **Caminando la ronda a pie** (sin instalar nada: ¿hasta dónde se sigue pareciendo el susurro?), las 70
  rondas "especiales" se ven **igual** que las otras 330. La única medida que no usa la misma perilla ni
  siquiera roza una diferencia. Las que sí usan la perilla muestran una diferencia chiquita, y cuando se
  comparan rondas con la **misma** posición de perilla, esa diferencia se esfuma.
- **Y lo más simple de todo:** si pesás la misma bolsa seis veces, la balanza tiene que dar lo mismo. Se
  agarró el mismo campo — el mismo, sin cambiarle un solo valor — y se lo volvió a medir seis veces
  cambiando sólo a qué desconocidos llama cada persona. De las 90 re-mediciones de las rondas "mundo
  pequeño", **41 dijeron mundo pequeño y 49 dijeron que no**. La bolsa no cambió. Cambió la balanza.

---

## 9. Archivos generados (todos nuevos; ningún congelado tocado)

**Scripts**
- `cs090_fase6_o1c_cierre_a0.py` — barrido de 400 reglas: método viejo (histórico + corregido, vía
  `correr_regla_coarse_doble`) y métricas nativas (importadas de `cs090_fase5_a0_nativo`) sobre el mismo
  campo. Incluye `verificar_combos_A0()`.
- `cs090_fase6_o1c_analisis.py` — KS, Mann-Whitney, permutación, Cliff's δ, Holm, Spearman parcial,
  emparejamiento por umbral, potencia por simulación, tabla de tuplas de diámetro, gráfico.
- `cs090_fase6_o1c_reproducibilidad.py` — test-retest: mismo campo, 6 mediciones independientes.

**Datos**
- `cs090_fase6_o1c_a0_resumen.csv` — 400 filas, una por regla (parámetros, clase histórica y corregida,
  pendientes, métricas nativas REAL y NULL, diagnóstico de fragmentación).
- `cs090_fase6_o1c_a0_viejo_raw.csv` — 2.000 filas (400 reglas × 5 escalas): diámetros orig y corr,
  tamaño de componente medida vs. gigante, nº de componentes/aislados.
- `cs090_fase6_o1c_a0_nativo_raw.csv` — 2.000 filas: métricas nativas REAL y NULL por escala.
- `cs090_fase6_o1c_tests.csv` — 46 filas, todos los tests con sus estadísticos y p.
- `cs090_fase6_o1c_reproducibilidad.csv` — 30 filas, una por regla re-medida.

**Gráfico**
- `cs090_fase6_o1c_distribuciones.png` — 6 paneles: (a) la pendiente vieja y su corte; (b) y (c) ECDF de
  las dos métricas nativas por clase; (d) control NULL; (e) el confound del umbral compartido; (f) tasa
  de Clase II por banda de umbral.

**Logs**
- `cs090_fase6_o1c_cierre_a0.log`, `cs090_fase6_o1c_analisis.log`, `cs090_fase6_o1c_reproducibilidad.log`.
