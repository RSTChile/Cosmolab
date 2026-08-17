# O3-C — El factorial mecanístico completo pasado por Phantom: mecanismo → geometría → gravedad

**Fecha:** 11-ago-2026 · Ejecuta: CC (Claude) · Tarea **O3-C** del plan `FASE6_PLAN_EJECUCION_COMPLETA_CS.md`
(propuesta **F6-03** de GPT-5.6 Sol) · Phantom autorizado por Alexis · **Se ejecuta ahora, se analiza al
final de la tanda; este informe deja los números, no un veredicto.**

---

## 0. Qué pregunta responde, en simple

Hasta hoy teníamos **dos historias separadas, contadas por dos grupos de experimentos que nunca se
cruzaron**:

- **Historia A (Fase V, línea del mecanismo F5-C2-C … C5):** si el sistema tiene un *corte rígido*
  (un límite duro, sin excepciones) **y** decide a quién soltar *mirando el soporte local* (con quién
  comparte vecinos), entonces la red que queda es **extendida** (geometría "estirada", pendiente alta).
  Si le sacás la rigidez, o le sacás el criterio, la geometría se aplana.
- **Historia B (Fase V-B, 40 pares en Phantom):** las redes con **geometría extendida** acretan **más
  masa** cuando se las traduce a una nube de gas y se deja actuar la gravedad.

La analogía: la historia A dice *"esta receta de cocina produce un pan más aireado"*; la historia B dice
*"los panes más aireados pesan distinto al hornearlos"*. **Nadie había puesto la receta directamente en el
horno.** Esta tarea hace exactamente eso: toma las 4 recetas del factorial, las cocina sobre las
**mismas** genealogías, y mete **las cuatro** en Phantom, anotando para cada una tanto lo "aireado" del
pan (la pendiente) como lo que pesa después de hornear (la fracción de masa acretada).

Y pregunta lo que ninguna de las dos historias podía preguntar sola: **¿el mecanismo llega a la gravedad
a través de la geometría, o hay un atajo?** Es decir: cuando la geometría desaparece, ¿desaparece también
el efecto gravitacional (la cadena es real), o el mecanismo mueve la masa por otra vía que no pasa por la
forma de la red (hay un efecto directo)?

---

## 1. El factorial: 4 condiciones, mismas genealogías

Las dos perillas del factorial son las dos que la línea F5-C2-C…C5 aisló como determinantes:

|                         | **criterio = soporte/costo local** | **criterio = azar** |
|---|---|---|
| **corte RÍGIDO** (conteo exacto)   | **(1)** `C2-hibrido` | **(2)** `C2-random` |
| **corte ELÁSTICO** (presupuesto)   | **(3)** `C2-presupuesto-variable` | **(4)** `C2-presupuesto-variable-azar` |

La tercera dimensión de esa línea (uniformidad del cupo) se **fija en VARIABLE** en las 4 celdas. Dos
razones, ninguna de conveniencia: (a) es la única fila donde las 4 celdas ya estaban implementadas y
validadas por tareas anteriores, y (b) `FASE5_matriz_2x2_completa_CS.md` mostró que es la dimensión que
**menos** mueve la aguja (0-10 pp) — fijarla cuesta poco y deja el factorial limpio.

**Mismas genealogías:** las 12 reglas son las 12 primeras del lote A2-B0-C2 admitido (filtro P1-P5 real)
con `seed_base=90211`, **exactamente el mismo lote** que usaron las 5 tareas de la línea del mecanismo en
su corrida "completo". Cada regla pasa por las 4 condiciones con el mismo `seed`, `K`, `J`, `noise`,
`meandeg`, `kcap` — es un diseño **pareado dentro de regla**: las 4 condiciones son 4 versiones del mismo
universo, no 4 universos distintos.

---

## 2. Qué se reusó sin tocar (y qué es código nuevo)

Todo el motor y todo el pipeline ya existían. Lo único nuevo es el pegamento que los une:

| pieza | de dónde sale | qué hace |
|---|---|---|
| `GEN.generar_reglas_clase` / `generar_regla` | `cs090_fase5_generador.py` | el mismo lote de 12 reglas |
| `MOT.construir_A2` / `medir` / `correr_regla_coarse` | `cs090_fase5_motor.py` | sustrato y medición |
| `MA._cupo_variable`, `MA.dinamica_B0_hibrido`, `MA.correr_regla_coarse_hibrido` | `cs090_fase5_mecanismo_aislado.py` | condiciones **1** y **2** |
| `PV.dinamica_B0_presupuesto_variable`, `PV.correr_regla_coarse_presupuesto_variable` | `cs090_fase5_presupuesto_variable.py` | condición **3** |
| `CAE.dinamica_B0_presupuesto_variable_azar`, `CAE.correr_regla_coarse_presupuesto_variable_azar` | `cs090_fase5_control_azar_elastico.py` | condición **4** |
| `DC.diam_gigante` | `cs090_diam_corregido.py` | el metro oficial desde `FASE6_adopcion_diam_corregido_CS.md` |
| `clasificar_regla` | `cs090_fase5_clasificador.py` | pendiente + clase |
| `generar_ic_masa_fija_desde_grafo` | `cs090_fase5b_phantom_adaptador.py` | grafo → condición inicial de Phantom, masa total fija |
| `correr_una` | `cs090_fase5b_correr.py` | `phantomsetup` + `phantom`, mismos parámetros CS073 |
| `analizar_carpeta` | `cs090_fase5b_analizar.py` | masa en sumideros, κ_V, primer sumidero |

**Archivos nuevos (los únicos):** `cs090_fase6_o3c_factorial_mecanistico.py` (genera geometría + grafo +
IC, corre Phantom, tabula) y `cs090_fase6_o3c_mediacion.py` (el análisis de la cadena).

**Detalle técnico que importa:** las 4 funciones `correr_regla_coarse_*` originales construyen el grafo
final por dentro pero **sólo devuelven las filas de coarse-graining**, no el grafo. Para poder mandarlo a
Phantom, `_grafo_final_de_condicion` repite el **mismo prólogo determinista** de las 4 (mismo
`rng = default_rng(seed*5000+N)`, mismo `construir_A2`, mismo `_cupo_variable` sobre el grado recién
nacido) y llama a **la misma función de dinámica importada de su módulo original**. Que sea el mismo grafo
no se asume: se verifica numéricamente (chequeo #3 abajo).

El diámetro corregido se aplica con la misma técnica de `cs090_fase6_remedir_mecanismo.py`: se sustituye
`MOT._diam` **en memoria** por `DC.diam_gigante` durante la llamada y se restaura después. Ningún archivo
en disco se modifica.

---

## 3. Los 5 chequeos cruzados obligatorios (lección del bug de colisión de nombres)

Ninguno es decorativo. Cuatro de ellos (**#1, #3, #4, #5**) abortan la corrida con `AssertionError` si
fallan, y **los cuatro pasaron en las 48 celdas, sin excepción**. El **#2** es el único que quedó como
diagnóstico registrado en vez de aborto — y no pasó en 8 de 48 celdas; el porqué de esa decisión y los
números están en §5.2, no escondidos acá.

1. **#1 — parámetros contra el archivo.** El `seed` regenerado por el generador para cada regla debe ser
   idéntico al archivado en `cs090_fase6_remedicion_mecanismo.csv` para la regla histórica equivalente
   (`A2-B0-C2-r{idx}`), en las 4 condiciones. Se corre **antes** de generar nada.
2. **#2 — reproducibilidad de la geometría** (*diagnóstico, no aborto*). La pendiente corregida
   recalculada fresca se compara con la archivada en ese mismo CSV, tolerancia `1e-6`. Es el determinismo
   del motor **entre sesiones** puesto a prueba — y efectivamente detectó algo (§5.2). Cada celda guarda
   `dif_vs_archivada` y `reproduce_archivada`.
3. **#3 — identidad del grafo.** `n_aristas` del grafo reconstruido debe ser exactamente igual a
   `n_aristas` de la fila `b=1` del brazo oficial. Garantiza que **el grafo que va a Phantom es el mismo
   que produjo la pendiente**, no uno parecido.
4. **#4 — `meta_regla.json` releído del disco.** Tras escribirlo, se vuelve a leer del disco y se exige
   que `rule_id`, `cond_id`, `seed`, `K` y `kcap` sean los esperados. Es el chequeo que faltaba cuando
   ocurrió el bug de colisión de nombres de Fase V-B.
5. **#5 — al analizar.** Antes de aceptar la fila de métricas de una carpeta, su `meta_regla.json` se
   vuelve a cruzar contra la tabla de tareas (`rule_id`, `cond_id`, `seed`).

**Prefijo sin colisión:** `rule_id = "A2-B0-C2-mec-r{idx}"` (prefijo `mec`, nunca usado: los ocupados eran
`r0-r19`, `r0-r39`, `batch3-*`, `batch4-*`, `*v1fix`, `*v2fix`, `*pendNEG`). Además la carpeta es
`{rule_id}__{cond_id}`, así que **las 4 condiciones de una misma regla no pueden pisarse entre sí ni por
accidente**.

---

## 4. Costo real medido (no estimado)

Disciplina piloto-primero: se corrió **1 regla × 4 condiciones** completa antes de comprometer el resto.

| paso | costo por celda | notas |
|---|---|---|
| geometría (brazo original con diámetro corregido) | ~4 s | |
| reconstrucción del grafo final | ~3 s | |
| condición inicial de Phantom (`layout_resortes` + expansión + turbulencia) | ~160 s | domina; máquina compartida con otras tareas de la tanda |
| Phantom (`phantomsetup` + `phantom`, N=2000, tmax=0.5) | ~35 s | |

Con 8 procesos en paralelo para la parte de generación y 3 corridas simultáneas de Phantom, las 48 celdas
entraron holgadas en el presupuesto de la tarea.

---

## 5. Lo que efectivamente se corrió, y las dos cosas que salieron mal (declaradas)

**48 celdas** (12 reglas × 4 condiciones) generadas y con condición inicial de Phantom escrita. **47 de 48
corridas de Phantom llegaron a `tmax=0.5`.** Dos incidentes, ninguno escondido:

1. **Una corrida de Phantom abortó por física, no por el pipeline.** `A2-B0-C2-mec-r10__c1-rigido-soporte`
   se detuvo en el volcado 100 (de 500) con `FATAL ERROR! evolve: Conservation errors too large to
   continue simulation`. No se la forzó a seguir (Phantom ofrece `I_WILL_NOT_PUBLISH_CRAP=yes`; usarlo
   habría dado un número no comparable con los otros 47). Como el diseño es pareado dentro de regla, **la
   regla `r10` se excluye entera** — quedan **11 reglas × 4 condiciones = 44 celdas** en el análisis.
   *Advertencia honesta:* la corrida que reventó es de la condición 1, justamente la que más colapsa;
   excluir esa regla podría, en principio, sacar del promedio un caso de acreción alta y jugar **en
   contra** de la condición 1. Se deja anotado: no se sabe cuánto habría acretado.
2. **8 de 48 celdas no reproducen bit a bit la pendiente archivada** en
   `cs090_fase6_remedicion_mecanismo.csv` (las otras 40 la reproducen con error ≤ 2.2e-15, es decir,
   exactas). Las 8 son todas de brazos con un paso de **selección aleatoria** (6 de `c2`/`c4`, 2 de `c3`);
   las diferencias van de 1.0e-4 a **3.4e-2** (la mayor, `r11/c2`). El chequeo #2 se **degradó de aborto a
   diagnóstico registrado** por una razón concreta: el análisis usa la pendiente **fresca**, y el chequeo
   #3 garantiza que esa pendiente y el grafo que fue a Phantom salieron de la misma corrida — comparar
   contra el archivo es un control de reproducibilidad *entre sesiones*, no un requisito de correctitud de
   esta tarea. La causa no se investigó a fondo (queda como pista abierta: lo más probable es que una
   diferencia de último bit, amplificada por la dinámica, cambie una decisión de recableo en esos brazos;
   el archivo se calculó en otra sesión, posiblemente con otra versión de numpy). Cada celda lleva
   `dif_vs_archivada` y `reproduce_archivada` en el CSV crudo.

Los chequeos **#1, #3, #4 y #5 pasaron en las 48 celdas, sin una sola excepción.**

Dato que conviene tener presente antes de leer los números: **las 44 celdas formaron exactamente 8
sumideros cada una, sin ninguna excepción.** El conteo de sumideros está saturado; toda la señal
gravitacional está en **cuánta masa** termina en ellos, no en cuántos se forman.

---

## 6. Resultado 1 — las 4 condiciones, geometría y gravedad lado a lado

| condición | n | pendiente media | pendiente mediana | **fracción de masa acretada** (media) | mediana | sumideros | %Clase III |
|---|---|---|---|---|---|---|---|
| **(1) rígido + soporte** | 11 | **0.6440** | 0.6091 | **0.08841** | 0.07300 | 8.00 | **36.4%** |
| (4) elástico + azar | 11 | 0.5494 | 0.5710 | 0.07473 | 0.07750 | 8.00 | 0.0% |
| (3) elástico + soporte | 11 | 0.5181 | 0.5249 | 0.06868 | 0.06900 | 8.00 | 0.0% |
| (2) rígido + azar | 11 | 0.4869 | 0.5152 | 0.06573 | 0.06400 | 8.00 | 0.0% |

**El orden de las 4 condiciones es EL MISMO en geometría y en gravedad: 1 > 4 > 3 > 2.** La condición 1 es
la única que produce Clase III (4 de 11), la única con pendiente media por encima de 0.6, y la que más
masa acreta. La condición 2 (rígido pero eligiendo al azar) es la última en las dos columnas.

### El factorial 2×2: lo que domina es la INTERACCIÓN, no los efectos principales

|  | criterio = soporte | criterio = azar |
|---|---|---|
| **corte RÍGIDO** — pendiente / masa | 0.6440 / 0.08841 | 0.4869 / 0.06573 |
| **corte ELÁSTICO** — pendiente / masa | 0.5181 / 0.06868 | 0.5494 / 0.07473 |

| contraste | en PENDIENTE | en MASA |
|---|---|---|
| efecto principal de RIGIDEZ (rígido − elástico) | +0.032 | +0.0054 |
| efecto principal de CRITERIO (soporte − azar) | +0.063 | +0.0083 |
| **INTERACCIÓN** (el criterio sólo rinde si el corte ya es rígido) | **+0.188** | **+0.0287** |

La interacción es **3 a 6 veces** más grande que cualquiera de los dos efectos principales, **y lo es en
las dos variables a la vez, con la misma forma**. Esto es exactamente lo que la línea de Fase V había
encontrado en geometría (`FASE5_control_azar_elastico_CS.md` §6: *"el criterio sólo importa MUCHO cuando
ese límite duro ya está"*) — y ahora aparece **también en la respuesta gravitacional**, medida en un
integrador que no sabe nada de grafos.

En simple: **ninguna de las dos perillas sirve sola.** El corte rígido sin criterio (condición 2) es la
PEOR de las cuatro, peor que cualquier elástica. El criterio sin corte rígido (condición 3) tampoco rinde.
Sólo las dos juntas levantan la geometría y la masa. Es como una cerradura de dos llaves: cada llave sola
no abre nada, y girar sólo la primera deja peor que no tocar nada.

### Comparaciones pareadas (misma regla en las dos condiciones, n=11)

| comparación | variable | gana A | gana B | mediana dif. | p signos | p Wilcoxon |
|---|---|---|---|---|---|---|
| (1) vs (2) | pendiente | 8 | 3 | +0.0988 | 0.227 | **0.042** |
| (1) vs (2) | **masa** | 8 | 2 | +0.0020 | 0.109 | **0.037** |
| (1) vs (3) | pendiente | 6 | 5 | +0.0254 | 1.000 | 0.240 |
| (1) vs (3) | masa | 6 | 5 | 0.0000 | 1.000 | 0.278 |
| (1) vs (4) | pendiente | 6 | 5 | +0.0017 | 1.000 | 0.320 |
| (1) vs (4) | masa | 6 | 5 | +0.0030 | 1.000 | 0.278 |
| (2) vs (4) | masa | 2 | 9 | −0.0095 | 0.065 | **0.005** |
| (3) vs (4) | pendiente | 2 | 9 | −0.0440 | 0.065 | **0.014** |

El contraste que sale claro y en la misma dirección en las dos variables es **(1) contra (2)** — las dos
condiciones con corte rígido, que se diferencian sólo en el criterio: la condición 1 gana en pendiente
(8-3) y en masa (8-2), con Wilcoxon p≈0.04 en ambas. Contra las dos elásticas, la condición 1 gana pero
por poco (6-5 en las cuatro comparaciones): **con n=11 la ventaja de la condición 1 es nítida frente al
corte rígido sin criterio, y sólo direccional frente a las elásticas.**

---

## 7. Resultado 2 — la cadena de mediación: ¿el mecanismo llega a la masa a través de la forma?

Con X = "es la condición 1" (0/1), M = pendiente corregida, Y = fracción de masa acretada, n=44 celdas:

| tramo | qué pregunta | valor |
|---|---|---|
| **a** : condición → pendiente | ¿el mecanismo mueve la geometría? | a = **+0.1258** · r=+0.360 (p=0.016) |
| **b** : pendiente → masa \| condición | ¿la geometría mueve la gravedad, a igual condición? | b = **+0.1116** · r parcial=**+0.881** (p=3e-15) |
| **c** : condición → masa (TOTAL) | efecto total del mecanismo sobre la gravedad | c = **+0.0187** · r=+0.412 (p=0.0055) |
| **c'** : condición → masa \| pendiente | lo que queda del mecanismo al controlar la forma | c' = **+0.0046** · r parcial=+0.216 (**p=0.159**) |
| **a·b** | efecto indirecto (el que pasa por la geometría) | **+0.0140**, IC95% bootstrap por regla **[+0.0007, +0.0282]** |

**Proporción mediada = 75.1%.** El coeficiente de la condición **cae un 75% en cuanto se mete la pendiente
en la ecuación**, y lo que sobra deja de distinguirse de cero (p=0.16). El efecto indirecto tiene un
intervalo de confianza que **no incluye el cero** (remuestreando reglas enteras, para respetar el
apareamiento).

**La misma cuenta dentro de cada regla** (restándole a cada variable la media de su propia genealogía, lo
que elimina cualquier diferencia entre universos y deja sólo lo que movió el cambio de mecanismo): a=+0.126
(r=+0.477, p=0.001), b=+0.107 (r parcial=+0.853), c=+0.0187 (r=+0.540, p=0.00016), **c'=+0.0052 (r
parcial=+0.287, p=0.059)**, indirecto **+0.0135, IC95% [+0.0006, +0.0281]**, **72.2% mediado**. La imagen
no cambia.

Con las otras dos codificaciones (rigidez sola, criterio solo) la proporción mediada es parecida (68.4% y
87.4%) pero el intervalo del efecto indirecto **sí incluye el cero** — coherente con §6: por separado,
ninguna de las dos perillas mueve mucho; lo que mueve es la combinación.

### La prueba directa de "si la geometría desaparece, ¿desaparece el efecto?"

Se parten las 44 celdas en dos mitades **por la mediana de la pendiente, ignorando de qué condición viene
cada una**, y se comparan las masas:

- pendiente ALTA (>0.524, n=22): masa media **0.08507** · mediana 0.07775
- pendiente BAJA (≤0.524, n=22): masa media **0.06370** · mediana 0.06275
- **Mann-Whitney p = 2.4e-05**

Y el mismo corte hecho **por condición** (condición 1 contra las otras tres): masa 0.08841 vs 0.06971,
**Mann-Whitney p = 0.416**.

**Partir por la forma separa la masa; partir por el mecanismo, no.** Es el resultado más directo de esta
tarea: si conocés la pendiente de una celda, sabés bastante de cuánta masa va a acretar; si sólo conocés
de qué condición vino, sabés mucho menos.

Y dentro de las condiciones 2/3/4 — las tres que **perdieron** la geometría (0% Clase III en las tres) —
la pendiente **sigue** prediciendo la masa: **Spearman ρ=+0.693 (p=8e-06, n=33)**. Es decir: no es que la
condición 1 tenga un canal propio; es que **allí donde queda algo de geometría, queda algo de gravedad, y
allí donde no queda, no queda** — en cualquiera de las cuatro condiciones.

En simple: el mecanismo no le habla a la gravedad directamente. Le habla a la **forma** de la red, y la
forma es la que le habla a la gravedad. Como el pan: la receta no hace el peso del pan, hace la miga; y la
miga hace el peso. Si dos recetas distintas producen la misma miga, producen el mismo pan.

---

## 8. El confound que hay que declarar: la pendiente no está sola

Antes de leer nada de arriba como "la geometría es el canal", este control es obligatorio. La otra cosa que
el mecanismo cambia es **cuántas aristas sobreviven**, y eso también correlaciona con la masa:

| correlato de la masa acretada | Pearson r | Spearman ρ |
|---|---|---|
| **pendiente corregida** | **+0.897** (p=2e-16) | +0.777 (p=6e-10) |
| grado medio / n_aristas del grafo final | −0.835 (p=2e-12) | **−0.951** (p=4e-23) |
| componente gigante | −0.936 (p=1e-20) | −0.796 |
| diámetro b=1 corregido | +0.781 | +0.771 |
| holonomía | +0.125 (p=0.42) | +0.105 |

El grado medio es, **en rango, el correlato más fuerte de todos** (ρ=−0.951): cuantas menos aristas
sobreviven, más masa acreta. Y la pendiente correlaciona con el grado medio (r=−0.755), así que las dos no
son independientes. Los controles:

- masa ~ pendiente **controlando grado medio**: r parcial = **+0.739** (p=1e-08) → la pendiente **sobrevive**.
- masa ~ grado medio **controlando pendiente**: r parcial = **−0.546** (p=1e-04) → el grado **también** sobrevive.
- masa ~ pendiente controlando grado **y** diámetro: r parcial = **+0.808** (p=3e-11).
- Varianza explicada de la masa: sólo pendiente **R²=0.805** · sólo grado medio **R²=0.698** · las dos
  juntas **R²=0.863** · las dos + condición **R²=0.887** · sólo condición **R²=0.170**.
- **Mediación usando el grado medio como mediador en vez de la pendiente**: proporción mediada **25.5%**
  (contra 75.1% con la pendiente).

**Lectura honesta:** los dos observables llevan información y están enredados, pero **la pendiente es el
mejor mediador de los dos por un margen amplio** (75% vs 25% de proporción mediada; R² 0.805 vs 0.698), y
sobrevive a controlar por el grado, mientras que la condición no sobrevive a controlar por la pendiente.
Aun así, **esta tarea no puede separar limpiamente "geometría extendida" de "red con menos aristas"** —
son dos caras de lo que el mecanismo hace, y haría falta un diseño que fije el número de aristas y varíe
sólo la forma (el control de rewiring de O3-B apunta justamente ahí) para desenredarlas del todo.

---

## 9. Lecturas alternativas honestas (no se fuerza ninguna)

- **n=11 reglas es poco.** Los intervalos del efecto indirecto son anchos y apenas excluyen el cero
  (`[+0.0007, +0.0282]`). Las comparaciones pareadas condición 1 vs las elásticas son 6-5 — o sea, empates.
  Lo que sí es fuerte y no depende de n es el tramo **b** (pendiente→masa, r parcial 0.88 con p=3e-15).
- **La condición 1 gana claro sólo contra la condición 2.** Contra las dos elásticas gana por poco. Si se
  mira sólo el ranking de medias, la condición 1 domina las dos columnas; si se mira regla por regla, la
  ventaja frente a las elásticas es direccional, no decisiva.
- **La condición 4 (elástico + azar) quedó SEGUNDA en las dos columnas**, por encima de la condición 3
  (elástico + soporte) y muy por encima de la 2. Esto repite, en gravedad, la rareza que
  `FASE5_control_azar_elastico_CS.md` ya había visto en geometría (el azar dentro del presupuesto elástico
  no empeora, incluso mejora un poco). No se explica acá; queda documentado como el mismo caso raro,
  ahora replicado en un observable físico independiente.
- **El conteo de sumideros está saturado en 8** en las 44 celdas. Toda la variación gravitacional pasa por
  la masa acretada. Si se quisiera un observable con más rango, habría que mover los parámetros de
  creación de sumideros — pero eso rompería la comparabilidad con toda la jerarquía CS073.
- **La mediación es estadística, no causal.** Se manipuló la condición (eso sí es experimental), pero la
  pendiente **no** se manipuló: es un observable medido, no una perilla. Que el coeficiente caiga 75% al
  controlar por ella es consistente con "la geometría es el canal", y también con "la pendiente y la masa
  comparten una causa común río arriba que la condición sólo empuja de a poco". Los números están; la
  distinción entre esas dos lecturas necesita un experimento que **fije la geometría a mano** y vea si la
  masa la sigue.
- **La regla `r10` excluida** podría haber cambiado los promedios (ver §5.1), y las **8 celdas que no
  reproducen el archivo** (§5.2) muestran que algunos brazos de esta línea no son perfectamente
  reproducibles entre sesiones — un dato metodológico que vale para toda la línea F5-C2-C…C5, no sólo para
  esta tarea.

---

## 10. Archivos de esta tarea

- `cs090_fase6_o3c_factorial_mecanistico.py` — genera geometría + grafo + IC, corre Phantom, tabula (nuevo).
- `cs090_fase6_o3c_mediacion.py` — el análisis de la cadena (nuevo).
- `cs090_fase6_o3c_crudo.csv` — **47 filas**, una por celda corrida: geometría + gravedad + verificaciones.
- `cs090_fase6_o3c_mediacion.csv` — 21 filas, la tabla de la cadena y las comparaciones pareadas.
- `cs090_fase6_o3c_ic.log` / `cs090_fase6_o3c_ic2.log` / `cs090_fase6_o3c_phantom.log` — corridas.
- `/Users/alexis/phantom_cs073/bateria_fase6_o3c_mecanistico/` — 48 carpetas con IC, `meta_regla.json`,
  volcados y `.sink` de Phantom.
- Este informe.

Ningún archivo existente fue modificado. No se hicieron commits de git. **No se declara cierre ni
veredicto** sobre si la cadena mecanismo→geometría→gravedad queda establecida: los números de §6, §7 y el
confound de §8 están arriba; la lectura final es de Alexis.
