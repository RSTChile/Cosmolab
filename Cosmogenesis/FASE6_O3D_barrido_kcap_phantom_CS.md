# O3-D — Barrido de `kcap` directamente en Phantom

**Fecha:** 11-ago-2026 · Ejecuta: CC (Claude) · Tarea **O3-D** del `FASE6_PLAN_EJECUCION_COMPLETA_CS.md`
(propuesta original: 2do analista, punto 1.1). **Phantom autorizado por Alexis para esta tanda.**
Antecedente directo y motivo de que esta tarea subiera de prioridad:
`FASE6_O2B_genealogias_escaladas_CS.md` (800 corridas de motor) encontró que **`kcap` domina la
geometría por lejos** — kcap=4 → 98,4 % Clase III · 5 → 71,7 % · 6 → 4,9 % · 7 → **0 %**
(η² kcap = 0,619 vs η² genealogía = 0,078) — pero eso se midió **sólo en el grafo, nunca en Phantom**.

Medición de diámetro: **la corregida** (`cs090_diam_corregido.diam_gigante`), según
`FASE6_adopcion_diam_corregido_CS.md`. Variable de geometría: **la pendiente continua**, no las clases
(consigna vigente: el escalón pierde contra la rampa, R² = 0,663 vs 0,182).

**No se modificó ningún script existente ni congelado. No se hicieron commits. No se declara cierre ni
veredicto** — se reportan números; la lectura final es de Alexis.

---

## 0. En simple, con analogía

`kcap` es **el tope de amigos que puede tener cada nodo**: con kcap=4 nadie puede sostener más de cuatro
relaciones; con kcap=7, hasta siete. O2-B ya había mostrado que ese tope decide casi por completo la
*forma* de la red resultante: con el tope apretado la red se estira y se vuelve un tejido extenso
(Clase III); con el tope flojo se apelmaza en una bola compacta (Clase I) — como la diferencia entre un
pueblo donde cada casa toca a cuatro vecinas y se forma un barrio largo, y una plaza donde todos se
amontonan alrededor de unos pocos.

Hasta ahora eso era **geometría de alambre**: una maqueta sin peso. Esta tarea es la primera vez que se
le pone **arena y gravedad de verdad** a las maquetas de los cuatro topes: se toman 8 reglas
A2-B0-C2 por cada valor de kcap ∈ {4, 5, 6, 7}, se les construye la misma cantidad exacta de masa
(18 800 unidades repartidas en 2 000 partículas, en una caja del mismo tamaño, con el mismo viento
turbulento), y se las suelta en Phantom a ver **cuánta arena termina cayendo dentro de los grumos**.

La pregunta de fondo tiene dos capas. La primera es fácil: *¿la masa también sigue al tope de amigos?*
La segunda es la interesante: *si la sigue, ¿es porque el tope cambia la forma y la forma junta la
masa (kcap → geometría → masa), o el tope hace algo a la masa por su cuenta que la forma no explica?*

---

## 1. Diseño — y por qué está hecho así

| ítem | valor |
|---|---|
| clase de regla | **A2-B0-C2** (grafo dinámico co-emergente · sin retroalimentación de 2º orden · tope duro + poda por costo) |
| valores de kcap | **4, 5, 6, 7** — los cuatro del rango calibrado `RANGO_KCAP=(4,7)` del generador congelado |
| reglas por kcap | **8** (32 en total) |
| eje secundario K | **4 con K ≤ 5 y 4 con K ≥ 7 en cada kcap** → diseño balanceado 4 × 2 con 4 reglas por celda |
| filtro de admisión | **P1-P5 real** del generador congelado (140 candidatas generadas, 140 admitidas, 0 descartes) |
| `seed_base` | **9 314 159**, prefijo de rule_id `A2-B0-C2-kbar-r` (ambos nuevos) |
| grafo | N = 2 000, 14 barridos, `seed_layout` = 12 345 (misma realización espacial que toda la línea Fase V-B) |
| Phantom | `rho_crit_cgs=1000`, `icreate_sinks=1`, `r_crit=0.6`, `h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`, masa total fija 18 800 — **idénticos a toda la jerarquía CS073** |
| controles sin estructura | 6 nuevos Erdős-Rényi emparejados en aristas (3 a 2 321, 3 a 4 608) + 2 históricos del proyecto (4 945 aristas) |

**La selección de reglas no mira ningún resultado.** `kcap` es un parámetro del sorteo del generador,
conocido *antes* de correr nada: se generaron 140 candidatas, se agruparon por su `kcap` y se tomaron
las primeras 8 de cada grupo, balanceando K. En ningún momento se eligió por pendiente, por clase ni
por masa — eso habría sido sesgo de selección y habría contaminado justo la pregunta que se hace.

**`seed_base` nuevo, verificado antes de gastar cómputo.** Con `seed = seed_base + intento*97 + 1` y
`max_intentos = 560`, la cadena de semillas de esta tarea ocupa como mucho 54 321 números. La
verificación automática imprimió: *separación mínima contra las 30 semillas base ya usadas en el
proyecto = 4 806 238* → holgura de 88,5×. Ninguna regla de O3-D puede compartir semilla con ninguna
tarea previa. (Es la lección del bug de colisión de nombres de Fase V-B, ahora comprobada por script.)

**Verificación cruzada doble, obligatoria.** (i) Antes de generar nada, `generar_regla(seed)` tiene que
devolver el `kcap` que el selector pidió, si no aborta. (ii) Después de escribir la condición inicial,
el `meta_regla.json` real en disco se relee y se compara contra lo pedido, si no aborta. Las 32 reglas
pasaron las dos.

**Los controles van emparejados en aristas, y en los dos extremos.** El número de aristas del grafo
final está atado a `kcap` casi por aritmética (medido sobre las 16 reglas de los extremos *antes* de
correr nada: kcap=4 → mediana 2 321 aristas; kcap=7 → mediana 4 608). Un único control habría parecido
"más denso" que kcap=4 y "menos denso" que kcap=7, mezclando densidad con estructura. Por eso cada
extremo tiene su propio espejo sin estructura: mismo pipeline completo (layout de resortes → dilatación
estática → turbulencia Mach 3 semilla 42 → masa fija), misma cantidad de aristas, pero el grafo es
Erdős-Rényi puro: sin dinámica, sin recableo, sin `kcap`.

---

## 2. Costo medido (disciplina piloto-primero)

Se corrió **una regla completa antes de comprometer la batería**, como pide la consigna:

| etapa | costo medido (1 regla) |
|---|---|
| motor con coarse-graining (pendiente corregida) | 3-7 s |
| reconstrucción del grafo final | 2-5 s |
| **generación de la condición inicial (`layout_resortes`)** | **130-262 s** ← el 95 % del costo |
| Phantom (setup + corrida + análisis + poda) | **33-42 s** |

El paso caro no es Phantom: es el layout de resortes en Python (Fruchterman-Reingold O(N²), 100
iteraciones). Por eso **la generación de IC va en paralelo** (monohilo, cada worker en su propia carpeta,
resultado bit a bit idéntico porque cada regla deriva su rng de su propia semilla) y **Phantom va
serial** (compilado con OpenMP sobre los 16 hilos; la misma razón que documenta
`cs090_fase6_o3a_correr_phantom.sh`).

**Salvedad honesta sobre los tiempos:** esta tanda se corrió con la Mac compartida por varias tareas de
la sesión en paralelo (carga media del sistema entre 500 y 700, con otras instancias de Phantom activas
de otras tareas). Los tiempos de arriba son *bajo esa carga* y no son comparables con los ~15 s por
corrida de Phantom que reportó el piloto de Fase V-B en una máquina libre. La carga afecta el reloj, no
los números: el conteo de hilos de OpenMP es el mismo de siempre.

**Poda de dumps:** cada corrida escribe 500 dumps (~27 MB) de los que el análisis sólo lee el primero,
el último y el `.sink`. Los intermedios se borran **después** de que el resultado está escrito, nunca
antes (el disco de esta Mac tiene ~16 GiB libres).

---

## 3. Pregunta 1 — ¿la masa acretada varía monótonamente con kcap?

**Sí, y sin un solo cruce.** Las 32 corridas alcanzaron `tmax=0.500` sin abortar; masa total 18 800 en
las 32; 8 sumideros en 31 de 32 corridas y 9 en una.

| kcap | n | fracción de masa (media ± ee) | mediana | rango | pendiente media | κ_V medio | grado medio del grafo | t 1er sumidero |
|---|---|---|---|---|---|---|---|---|
| **4** | 8 | **0,1488 ± 0,0022** | 0,1520 | 0,1390-0,1545 | 1,047 | 1,297 | 2,35 | 0,028 |
| **5** | 8 | **0,1097 ± 0,0040** | 0,1083 | 0,0965-0,1285 | 0,807 | 0,650 | 3,20 | 0,036 |
| **6** | 8 | **0,0820 ± 0,0025** | 0,0798 | 0,0750-0,0940 | 0,587 | 0,409 | 3,95 | 0,041 |
| **7** | 8 | **0,0649 ± 0,0021** | 0,0655 | 0,0545-0,0720 | 0,480 | 0,390 | 4,61 | 0,050 |

| observable | Spearman(kcap, ·) | Kruskal-Wallis | η²(kcap) |
|---|---|---|---|
| fracción de masa en sumideros | **ρ = −0,969** (p = 1×10⁻⁵ o menor) | H = 29,11 · p < 10⁻⁵ | **0,949** |
| pendiente corregida (geometría) | ρ = −0,938 | H = 27,34 · p = 1×10⁻⁵ | 0,858 |
| κ_V agregado | ρ = −0,893 | H = 26,12 · p = 1×10⁻⁵ | 0,924 |
| grado medio del grafo final | ρ = +0,969 | H = 29,09 · p < 10⁻⁵ | 0,971 |

Los cuatro grupos están **completamente separados**: el peor kcap=4 (0,1390) queda por encima del mejor
kcap=5 (0,1285), el peor kcap=5 (0,0965) por encima del mejor kcap=6 (0,0940), y el peor kcap=6 (0,0750)
por encima del mejor kcap=7 (0,0720). **No hay un solo solape entre grupos vecinos en 32 reglas.** La
masa acretada más que duplica de kcap=7 a kcap=4 (0,065 → 0,149, ×2,3). El tiempo al primer sumidero
también se ordena (0,028 → 0,050): con el tope apretado los grumos nacen antes.

El η² de kcap sobre la **masa** (0,949) es mayor que el η² de kcap sobre la **geometría** en el propio
grafo (0,858) y del mismo orden que el que O2-B había medido sobre la clase (0,619). En cuanto a las
clases: kcap=4 dio 7 Clase III + 1 Clase IV, kcap=5 dio 6 Clase III + 2 Clase I, kcap=6 dio 8 Clase I,
kcap=7 dio 5 Clase I + 3 Clase II — la misma tendencia que O2-B, con la salvedad de que aquí kcap=5
salió más "fértil" (75 % III) que el 71,7 % de O2-B y kcap=6 menos (0 % III vs 4,9 %), diferencias
compatibles con n = 8 por grupo.

**El eje secundario K no hace nada.** ANOVA de dos vías sobre la masa, diseño balanceado 4 × 2 con 4
reglas por celda: kcap F(3,24) = 153,30 · p < 10⁻⁴ · **η² = 0,949**; grupo_K F(1,24) = 0,08 · p = 0,778 ·
**η² = 0,000**; interacción F(3,24) = 0,28 · p = 0,840 · η² = 0,002. K alto (media 0,1018) y K bajo
(0,1009) son indistinguibles. Todo el efecto está en kcap.

---

## 4. Pregunta 2 — ¿pasa por la geometría, o es un efecto directo de kcap?

Usando la **pendiente continua** como variable (no las clases), las tres regresiones anidadas:

| modelo | R² | coeficientes |
|---|---|---|
| masa ~ kcap | 0,920 | kcap β = −0,0279 (t = −18,6) |
| masa ~ pendiente | 0,928 | pendiente β = +0,1335 (t = +19,7) |
| **masa ~ kcap + pendiente** | **0,965** | kcap β = −0,0139 (t = −5,59 · p < 10⁻⁴) · pendiente β = +0,0731 (t = +6,19 · p < 10⁻⁴) |
| masa ~ kcap + pendiente + grado medio | 0,975 | kcap β = +0,0009 (t = +0,18 · **p = 0,856**) · pendiente β = +0,0482 (t = +3,81 · p = 0,0007) · grado medio β = −0,0261 (t = −3,33 · p = 0,0025) |

Correlaciones parciales: r(kcap, masa | pendiente) = **−0,720** (p < 10⁻⁴) · r(pendiente, masa | kcap) =
**+0,754** (p < 10⁻⁴). Mediación con bootstrap de 10 000 remuestreos: efecto indirecto
a·b = −0,0141 (IC95 % [−0,0188, −0,0103]) · efecto directo c′ = −0,0139 (IC95 % [−0,0176, −0,0092]) ·
**proporción mediada por la geometría ≈ 50 %**, con ambas mitades distintas de cero.

**Leído así, la respuesta sería "mitad y mitad". Pero eso es un espejismo, y el propio análisis lo
delata.** Los tres candidatos a causa son casi la misma variable:

| par | r |
|---|---|
| kcap ↔ grado medio del grafo | **+0,984** |
| kcap ↔ pendiente | −0,915 |
| pendiente ↔ grado medio | −0,943 |

VIF(kcap) = 32,5 · VIF(pendiente) = 9,4 · VIF(grado medio) = 47,8. Con esa colinealidad, "repartir" el
efecto entre kcap y pendiente es pedirle a los datos una distinción que no contienen: `kcap` **limita
aritméticamente** las aristas (2,35 de grado medio a kcap=4 · 4,61 a kcap=7) y el número de aristas es
lo que fija tanto la pendiente como la compacidad del layout que entra a Phantom. Prueba de ello: al
poner el grado medio en la ecuación, el coeficiente de kcap **se va a cero** (β = +0,0009, p = 0,856).

### 4.1 La prueba que decide: el control sin estructura cae SOBRE la recta

El experimento crítico no es estadístico sino físico. Se midió la pendiente de los 6 grafos de control
**con la misma vara exacta** (mismo coarse-graining b = 1…16, mismo `diam_gigante`,
`cs090_fase6_o3d_pendiente_controles.py`) y se los puso en el mismo gráfico masa-vs-pendiente:

| control (Erdős-Rényi puro) | aristas | pendiente | fracción de masa | κ_V |
|---|---|---|---|---|
| a2321-s3000001 | 2 321 | 1,053 | 0,1510 | 1,360 |
| a2321-s3000002 | 2 321 | 1,181 | 0,1505 | 1,176 |
| a2321-s3000003 | 2 321 | 1,089 | 0,1490 | 1,449 |
| a4608-s3000011 | 4 608 | 0,639 | 0,0695 | 0,395 |
| a4608-s3000012 | 4 608 | 0,505 | 0,0685 | 0,425 |
| a4608-s3000013 | 4 608 | 0,458 | 0,0730 | 0,564 |

**Los seis caen sobre la recta ajustada con las 32 reglas**: residuo medio −0,0032, rango
[−0,0197, +0,0079], dentro de la dispersión propia de las reglas (desvío estándar de sus residuos
0,0089, rango [−0,0162, +0,0206]).

Y hay una corroboración interna, medida **dentro de cada regla** por el propio motor, que apunta a lo
mismo: el NULL_topo de cada regla (un grafo ER fresco con su misma densidad) tiene **la misma pendiente
que el REAL** en los cuatro kcap:

| kcap | pendiente REAL | pendiente NULL_topo (ER de la misma densidad) | z_agg medio |
|---|---|---|---|
| 4 | 1,047 | 1,075 | 0,81 |
| 5 | 0,807 | 0,768 | 0,76 |
| 6 | 0,587 | 0,571 | 0,45 |
| 7 | 0,480 | 0,471 | 0,72 |

Ningún z_agg llega a 1 (el umbral de "separación sostenida" del clasificador es 3). Es decir: en esta
batería, la etiqueta Clase III de kcap=4 **no viene de separarse de un grafo al azar**, viene sólo de
cruzar el umbral de pendiente 0,7 — umbral que un grafo al azar de la misma densidad también cruza.

**En simple, con analogía.** Creíamos estar comparando *pueblos con distinto urbanismo*. Lo que el
control muestra es que lo único que cambiaba era **cuántas calles había por casa**: un pueblo trazado
completamente al azar, con la misma cantidad de calles, se estira igual y junta la misma arena. El
"tope de amigos" no está esculpiendo una forma especial: está regulando cuántas relaciones hay, y de
ahí sale todo lo demás.

---

## 5. Pregunta 3 — el extremo kcap=7

Con kcap=7 (el valor donde O2-B midió 0 % de geometría extendida en el grafo), la fracción de masa cae
a **0,0649 ± 0,0021**, contra 0,1488 en kcap=4: diferencia +0,0839 · Mann-Whitney U = 64,0 · **p = 0,0009**.

¿Cae al nivel de un control sin estructura? **Depende de contra cuál control se compare, y ahí está el
punto.**

| referencia | aristas | fracción de masa |
|---|---|---|
| **kcap = 7** (8 reglas) | 4 204-5 036 (mediana 4 608) | **0,0649** (0,0545-0,0720) |
| control ER emparejado a kcap=7 (3) | 4 608 | **0,0703** (0,0685-0,0730) |
| controles históricos del proyecto (2) | 4 945 | **0,0605** (0,0590-0,0620) |
| control ER emparejado a kcap=4 (3) | 2 321 | 0,1502 (0,1490-0,1510) |
| **kcap = 4** (8 reglas) | 2 224-2 556 | 0,1488 (0,1390-0,1545) |

- Contra su **espejo de la misma densidad**, kcap=7 está prácticamente empatado, con la estructura
  ligeramente **por debajo** del azar (0,0649 vs 0,0703). El control histórico de 4 945 aristas (0,0605)
  queda apenas más abajo, coherente con tener aún más aristas.
- Contra el promedio de los 6 controles juntos (0,1103) daría una diferencia grande y "significativa"
  (U = 4,0 · p = 0,012), pero **ese número no significa nada**: ese promedio mezcla dos densidades muy
  distintas. Se reporta porque estaba en el plan de análisis, no porque sea informativo.
- La simetría es lo importante: **kcap=4 tampoco supera a su espejo** (0,1488 vs 0,1502). En los dos
  extremos del barrido, un grafo sin ninguna estructura con la misma cantidad de aristas acreta lo
  mismo o un pelo más que la regla A2-B0-C2.

κ_V sigue exactamente el mismo patrón: kcap=4 → 1,297 vs su control 1,328; kcap=7 → 0,390 vs su
control 0,461. La forma temporal de la acreción también queda explicada por la densidad.

---

## 6. Qué quedó medido, y qué preguntas abre (sin cerrar nada)

1. **La masa acretada sí sigue a kcap, monótona y sin solapes** — es el efecto más limpio medido en
   Phantom en toda esta línea (η² = 0,949, ρ = −0,969, cuatro grupos separados en 32 reglas).
2. **La cadena kcap → geometría → masa no se puede separar de kcap → densidad de aristas → masa** con
   estos datos: las tres variables tienen r ≥ 0,92 entre sí, y al controlar por grado medio el
   coeficiente de kcap se anula.
3. **El control sin estructura reproduce todo**: mismo número de aristas ⇒ misma pendiente, misma masa,
   mismo κ_V, aunque el grafo sea Erdős-Rényi puro sin dinámica, sin recableo y sin `kcap`.
4. Corolario incómodo para la línea Fase V: en esta batería el **z contra NULL_topo nunca pasa de 1**.
   La etiqueta Clase III, tal como la asigna el clasificador vigente, aquí es equivalente a "el grafo
   tiene pocas aristas por nodo".

**Preguntas que esto abre** (para que Alexis decida, no propuestas de cierre):
- ¿Hay algún observable donde la regla A2-B0-C2 *sí* se separe de su espejo ER de la misma densidad?
  El candidato natural sería medir REAL vs NULL a **densidad igualada y layout igualado** en algo que no
  sea diámetro (holonomía, espectro del laplaciano à la CS084-CS089, distribución radial de sumideros).
- ¿Conviene reexaminar el η² = 0,619 de O2-B a la luz de esto? Ese resultado sigue siendo cierto como
  descripción (kcap decide la clase), pero esta batería sugiere que "clase" ahí puede estar leyendo
  densidad.
- ¿Vale la pena un barrido de **aristas a kcap fijo**? Sería el experimento que rompe la colinealidad:
  reglas con el mismo `kcap` pero grafos de densidad deliberadamente distinta (o al revés, densidad
  igualada por podado con distinto kcap). Es lo único que separaría de verdad las dos historias.

---

## 7. Archivos de esta tarea

| archivo | qué es |
|---|---|
| `cs090_fase6_o3d_barrido_kcap.py` | generación de candidatas + selección + worker (motor con pendiente corregida, grafo, IC) + controles + corrida de Phantom + consolidación |
| `cs090_fase6_o3d_generar_ic.sh` | generación de IC de las 32 reglas en paralelo |
| `cs090_fase6_o3d_generar_controles.sh` | generación de IC de los 6 controles Erdős-Rényi emparejados en aristas |
| `cs090_fase6_o3d_pendiente_controles.py` | mide la pendiente de los grafos de control con la misma vara que las reglas |
| `cs090_fase6_o3d_analizar.py` | estadística (monotonía, mediación, colinealidad, ANOVA 2 vías, kcap=7 vs control) + figura |
| `cs090_fase6_o3d_candidatas.csv` | las 140 candidatas generadas y admitidas por P1-P5 |
| `cs090_fase6_o3d_seleccion.json` · `cs090_fase6_o3d_trabajos.txt` | las 32 reglas elegidas (8 por kcap, 4+4 en K) |
| **`cs090_fase6_o3d_crudo.csv`** | **datos crudos: 38 filas (32 reglas + 6 controles), 31 columnas** |
| `cs090_fase6_o3d_resumen_por_kcap.csv` | resumen por kcap |
| `cs090_fase6_o3d_pendiente_controles.csv` | pendiente + masa de los 6 controles |
| `cs090_fase6_o3d_control_historico.csv` | los 2 controles ER históricos del proyecto (sólo lectura de sus carpetas) |
| **`cs090_fase6_o3d_barrido.png`** | **figura de 4 paneles del barrido** |
| `cs090_fase6_o3d_analisis.log` | salida completa del análisis |
| `/Users/alexis/phantom_cs073/bateria_fase6_o3d_kcap/<rule_id>/` (×38) | IC, `cosmog.in`, logs, primer y último dump, `.sink`, `meta_regla.json`, `resultado_o3d.json` |

Ningún script congelado fue modificado. No se declaró cierre ni veredicto. No se hicieron commits de git.

---

## Apéndice — las 32 reglas, una por una

| regla | kcap | K | clase | pendiente | aristas | fracción de masa | κ_V |
|---|---|---|---|---|---|---|---|
| r7 | 4 | 8 | III | 1,219 | 2270 | 0,1525 | 1,391 |
| r74 | 4 | 8 | III | 1,191 | 2224 | 0,1545 | 1,293 |
| r0 | 4 | 4 | IV | 1,115 | 2294 | 0,1530 | 1,429 |
| r2 | 4 | 5 | III | 1,050 | 2303 | 0,1520 | 1,274 |
| r32 | 4 | 5 | III | 1,021 | 2340 | 0,1520 | 1,312 |
| r65 | 4 | 7 | III | 1,012 | 2372 | 0,1475 | 1,319 |
| r19 | 4 | 5 | III | 0,912 | 2479 | 0,1400 | 1,236 |
| r6 | 4 | 7 | III | 0,858 | 2556 | 0,1390 | 1,119 |
| r11 | 5 | 7 | III | 0,990 | 3086 | 0,1285 | 1,000 |
| r15 | 5 | 5 | III | 0,866 | 3123 | 0,1245 | 0,898 |
| r10 | 5 | 7 | III | 0,838 | 3193 | 0,1085 | 0,592 |
| r26 | 5 | 5 | III | 0,827 | 3159 | 0,1035 | 0,536 |
| r25 | 5 | 7 | III | 0,823 | 3244 | 0,1080 | 0,622 |
| r20 | 5 | 5 | III | 0,809 | 3188 | 0,1090 | 0,563 |
| r21 | 5 | 4 | I | 0,693 | 3330 | 0,0990 | 0,470 |
| r36 | 5 | 7 | I | 0,611 | 3316 | 0,0965 | 0,516 |
| r18 | 6 | 8 | I | 0,645 | 3835 | 0,0940 | 0,452 |
| r4 | 6 | 7 | I | 0,633 | 3923 | 0,0905 | 0,403 |
| r9 | 6 | 5 | I | 0,609 | 3849 | 0,0810 | 0,364 |
| r35 | 6 | 7 | I | 0,603 | 4086 | 0,0780 | 0,325 |
| r3 | 6 | 5 | I | 0,586 | 3880 | 0,0835 | 0,446 |
| r14 | 6 | 4 | I | 0,574 | 3865 | 0,0785 | 0,491 |
| r54 | 6 | 7 | I | 0,569 | 4093 | 0,0755 | 0,389 |
| r28 | 6 | 5 | I | 0,475 | 4089 | 0,0750 | 0,405 |
| r132 | 7 | 7 | I | 0,594 | 4204 | 0,0720 | 0,411 |
| r33 | 7 | 5 | I | 0,568 | 4678 | 0,0635 | 0,400 |
| r27 | 7 | 8 | I | 0,472 | 4524 | 0,0675 | 0,390 |
| r128 | 7 | 8 | I | 0,471 | 4732 | 0,0615 | 0,351 |
| r12 | 7 | 7 | I | 0,450 | 5036 | 0,0545 | 0,387 |
| r22 | 7 | 5 | II | 0,436 | 4767 | 0,0615 | 0,300 |
| r31 | 7 | 5 | II | 0,429 | 4403 | 0,0715 | 0,435 |
| r8 | 7 | 5 | II | 0,420 | 4539 | 0,0675 | 0,444 |

(rule_id completo = `A2-B0-C2-kbar-<r…>`. La pendiente es la corregida, con `diam_gigante`.)
