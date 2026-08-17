# F7-01 — Factorial ortogonal `kcap` × número de aristas (M)

**Fecha:** 12-ago-2026 · Ejecuta: CC (Claude) · Tarea **F7-01** de Fase VII. **Phantom autorizado.**

Antecedente directo: `FASE6_O3D_barrido_kcap_phantom_CS.md` y la Parte 3.bis de
`INFORME_EQUIPO_FASE6_11ago2026_CS.md`. Tres diseños independientes convergieron en que **la densidad
explica la mayor parte del efecto**, pero el número no se podía interpretar porque `kcap` y la densidad
estaban casi perfectamente pegados: **r = +0,984 · VIF hasta 47,8**. Esta tarea rompe esa colinealidad
por diseño y mide las dos preguntas por separado.

Medición de diámetro: **la corregida** (`cs090_diam_corregido.diam_gigante`). Desenlaces geométricos:
**pendiente continua** y **clustering** — en ningún lugar de este informe se usa "% Clase III"
(de hecho, las 36 reglas de esta batería cayeron todas en Clase I, así que la etiqueta no distingue
nada acá; toda la señal está en las variables continuas).

**No se modificó ningún script existente. No se hicieron commits. No se declara cierre ni veredicto** —
se reportan números; la lectura final es de Alexis.

---

## 0. En simple, con analogía

`kcap` es **el tope de amigos que puede tener cada nodo**. O3-D encontró que bajar ese tope hace que
caiga mucha más masa en los grumos (más del doble de kcap=7 a kcap=4). Pero encontró también el
problema: **apretar el tope de amigos también reduce el total de amistades de la red**. Las dos cosas
se movían juntas, y no había forma de saber cuál de las dos hacía el trabajo.

Es como querer saber si un barrio es más transitable porque *hay pocas calles* o porque *cada casa
tiene un límite de calles que puede tocar*. Mientras las dos cosas cambien a la vez, la pregunta no
tiene respuesta.

Lo que hace esta tarea es armar **barrios con exactamente la misma cantidad de calles pero distinto
límite por casa**, y a la vez **barrios con el mismo límite por casa pero distinta cantidad de
calles**. Con las dos perillas separadas, cada pregunta se contesta sola.

Y hay una decisión importante en el "cómo": para igualar la cantidad de calles **no se demolió
ninguna**. Se fabricaron mil doscientos barrios y se conservaron sólo los que ya habían nacido con la
cantidad de calles buscada. Demoler al azar era tentador y barato, pero **"podar al azar" es
justamente uno de los brazos del experimento F7-04 que corre en paralelo**: usarlo acá habría mezclado
el control de una tarea con el tratamiento de la otra.

---

## 1. Paso previo obligatorio — el mapa de M alcanzable por `kcap`

Antes de comprometer un solo minuto de Phantom se mapeó **qué rango de M produce naturalmente cada
`kcap`**. Se generaron **1 200 reglas A2-B0-C2** con el filtro P1-P5 real del generador congelado
(1 200 admitidas, 0 descartes) y se midió el M del grafo final de cada una.

Ese mapeo es viable porque existe un atajo barato: para contar aristas no hace falta el
coarse-graining, ni los NULL, ni el diámetro, ni la holonomía — sólo `construir_A2` + `dinamica_B0`.
**Costo medido: 0,18 s por regla con 8 procesos** (1 200 reglas en 221 s). El atajo no es una promesa:
el worker vuelve a reconstruir el grafo completo y **aborta si el M no coincide exactamente** con el
del mapa. Las 36 reglas pasaron esa verificación.

| kcap | n | min | p5 | p25 | mediana | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| 4 | 225 | 2 049 | 2 138 | 2 252 | 2 426 | 2 545 | 2 615 | **2 665** |
| 5 | 400 | 2 887 | 3 020 | 3 173 | 3 261 | 3 324 | 3 400 | **3 454** |
| 6 | 381 | **3 491** | 3 616 | 3 932 | 4 025 | 4 096 | 4 173 | **4 287** |
| 7 | 194 | **3 722** | 3 904 | 4 422 | 4 732 | 4 876 | 4 989 | 5 093 |

**Resultado del mapeo — y es en sí mismo un dato:** los rangos son casi disjuntos. La única
intersección real es **kcap 6 × kcap 7: M ∈ [3 722, 4 287]** (565 aristas de ancho, ~15 %). kcap=4 no
llega ni cerca (tope 2 665) y **kcap 5 y 6 se cruzan por 37 aristas de nada** (3 454 vs 3 491), sin
solape. Con 1 200 candidatas no hay un solo valor de M donde convivan tres `kcap`.

Por eso, siguiendo la consigna de *priorizar cubrir bien la intersección antes que abarcar los cinco
valores de kcap*, **todo el presupuesto se puso en la banda 6 × 7**. `kcap=8` no existe: el generador
congelado tiene `RANGO_KCAP=(4,7)` y ampliarlo habría sido modificar un archivo congelado.

**Qué mueve M dentro de un mismo `kcap`** (correlación con M, sobre las 1 200): es casi todo
`meandeg`, el grado medio del grafo Erdős-Rényi de partida — y **su signo se da vuelta con el tope**:
kcap=4 → **r = −0,96** (empezar más denso termina en *menos* aristas: el tope y la poda por costo
destruyen más de lo que se aportó), kcap=5 → −0,59, kcap=6 → **+0,59**, kcap=7 → **+0,90**. `K`, `J` y
`noise` casi no participan (|r| ≤ 0,37).

---

## 2. Diseño de la grilla — congelado antes de correr Phantom

| ítem | valor |
|---|---|
| clase de regla | **A2-B0-C2** (grafo dinámico co-emergente · sin retroalimentación de 2º orden · tope duro + poda por costo) |
| factor 1 | `kcap` ∈ **{6, 7}** — el único par con intersección real de M |
| factor 2 | M objetivo ∈ **{3 800, 4 000, 4 200}** — tres niveles dentro de [3 722, 4 287] |
| reglas por celda | **6** → **36 corridas de Phantom**, diseño 2 × 3 perfectamente balanceado |
| tolerancia de M | **±1,5 %** (±57 / ±60 / ±63 aristas) |
| **igualación de M** | **100 % por SELECCIÓN · 0 celdas por poda** |
| criterio de elección | las más cercanas al M objetivo, desempate por `seed` ascendente. **Ciego al resultado**: no mira pendiente, ni clase, ni holonomía, ni masa |
| `seed_base` | **70 701 000**, prefijo `A2-B0-C2-f701-r` (ambos nuevos) |
| grafo | N = 2 000, 14 barridos, `seed_layout` = 12 345 (misma realización espacial que toda la línea) |
| Phantom | `rho_crit_cgs=1000`, `icreate_sinks=1`, `r_crit=0.6`, `h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`, masa total fija 18 800 — **idénticos a toda la jerarquía CS073** |

**Separación de semillas verificada antes de generar nada.** El ancho de cadena de esta tarea es
965 601 (incluye el offset de +500 000 que usa el filtro P1-P5, que la verificación de O3-D no
contaba); la separación mínima contra las 37 semillas base ya usadas en el proyecto es 50 440 189 →
holgura **52×**. Además se comprobó que las tareas F7-02 y F7-04 que corren en paralelo reusan las
bases históricas 271828/371828/471828/571828: no hay riesgo de colisión con ninguna.

**Verificación cruzada triple, obligatoria, sobre las 36:** (i) `generar_regla(seed)` tiene que
devolver el `kcap` pedido; (ii) el grafo reconstruido tiene que dar **exactamente** el M del mapa;
(iii) el `meta_regla.json` releído de disco tiene que coincidir con lo pedido en seed, kcap y M. Las
36 pasaron las tres.

**Ampliación declarada.** El plan se congeló con **4 reglas por celda** y se amplió a **6** cuando
iban 6 corridas de 24. La ampliación fue **uniforme sobre las 6 celdas** (no se agregó a una y no a
otra), con **el mismo criterio ciego** ya declarado, y se hizo por potencia estadística sobre el
número que decide — no por lo que estaban dando los resultados. Queda escrito acá y en
`cs090_fase7_f701_plan.json` para que esté en el registro. Con tolerancia ±1,5 % la celda
kcap=7 / M=3 800 tiene **exactamente 6 candidatas disponibles**: 6 es el máximo que el diseño por
selección admite sin aflojar la tolerancia.

### 2.1 Costo medido antes de comprometer la grilla

| etapa | costo medido |
|---|---|
| mapeo de M (1 200 reglas, 8 procesos) | 221 s en total · **0,18 s/regla** |
| motor con coarse-graining (pendiente corregida) | 1,8-3,5 s · media **2,6 s** |
| reconstrucción del grafo + clustering exacto | 1,2-2,1 s · media **1,6 s** |
| **generación de la IC (`layout_resortes`)** | 94-129 s · media **108 s** ← el 96 % del costo |
| Phantom (setup + corrida) | 0,8-2,0 s + 8,8-14,5 s · media **13,8 s** |

Igual que en O3-D, el cuello no es Phantom sino el layout de resortes en Python. Por eso **la
generación de IC va en paralelo** (monohilo, cada worker en su carpeta, resultado bit a bit idéntico
porque cada regla deriva su azar de su propia semilla) y **Phantom va serial** (compilado con OpenMP
sobre los 16 hilos). Tiempo de reloj total de las 36: ~19 minutos, con la Mac compartida con otras
tareas de la sesión (carga media 20-30). **Poda de dumps** después de escribir el resultado, nunca
antes.

---

## 3. Lo primero: ¿se rompió la colinealidad?

**Sí, completamente.**

| | O3-D (barrido de kcap) | **F7-01 (este factorial)** |
|---|---|---|
| r(kcap, M) | **+0,984** | **+0,008** |
| VIF(kcap) | 32,5 | **1,00** |
| VIF(grado medio) | 47,8 | **1,00** |
| M medio de kcap=6 | 3 953 (rango 3 835-4 093) | **3 999,7** (3 786-4 206) |
| M medio de kcap=7 | 4 610 (rango 4 204-5 036) | **4 002,3** (3 749-4 232) |

Los dos grupos de `kcap` tienen ahora **la misma densidad media hasta la tercera cifra** (3 999,7 vs
4 002,3, diferencia de 2,6 aristas sobre 4 000) y se solapan en todo el rango. Es exactamente el
experimento que O3-D pedía en su sección 6 y no podía hacer.

---

## 4. Tabla por celda — las seis celdas del factorial

| kcap | M objetivo | n | M real | grado medio | meandeg ER | clustering | pendiente | **fracción de masa** | ee | κ_V | sumideros | t 1er sumidero |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 3 800 | 6 | 3 801,0 | 3,801 | 4,62 | 0,0016 | 0,583 | **0,0852** | 0,0009 | 0,440 | 8 | 0,041 |
| 6 | 4 000 | 6 | 3 999,3 | 3,999 | 5,86 | 0,0038 | 0,551 | **0,0783** | 0,0007 | 0,399 | 8 | 0,042 |
| 6 | 4 200 | 6 | 4 198,8 | 4,199 | 6,37 | 0,0048 | 0,544 | **0,0744** | 0,0010 | 0,404 | 8 | 0,046 |
| 7 | 3 800 | 6 | 3 812,0 | 3,812 | 4,14 | 0,0013 | 0,640 | **0,0868** | 0,0013 | 0,427 | 8 | 0,043 |
| 7 | 4 000 | 6 | 4 013,2 | 4,013 | 4,38 | 0,0010 | 0,574 | **0,0781** | 0,0008 | 0,411 | 8 | 0,041 |
| 7 | 4 200 | 6 | 4 181,8 | 4,182 | 4,65 | 0,0016 | 0,579 | **0,0745** | 0,0009 | 0,400 | 8 | 0,046 |

Las 36 corridas alcanzaron `tmax=0,500` sin abortar; masa total 18 800 en las 36; **8 sumideros en las
36**. Ninguna corrida se descartó.

Léase la tabla por columnas: **bajando por M la masa cae siempre; cruzando de kcap 6 a kcap 7 a la
misma altura de M, la masa no se mueve** (0,0852 vs 0,0868 · 0,0783 vs 0,0781 · 0,0744 vs 0,0745).

---

## 5. Pregunta 1 — a `kcap` fijo, ¿qué hace M?

**Mueve la masa, y fuerte.**

| kcap | n | R²(M) | r | pendiente | t | p |
|---|---|---|---|---|---|---|
| 6 | 18 | **0,822** | −0,907 | **−0,0270 por 1 000 aristas** | −8,59 | 2×10⁻⁷ |
| 7 | 18 | **0,850** | −0,922 | **−0,0340 por 1 000 aristas** | −9,53 | 5×10⁻⁸ |

> **R² MEDIO de M sobre la masa, a `kcap` fijo = 0,836**

Y la pendiente medida **coincide con la que predecía la recta de densidad de O3-D**: aquella recta
(ajustada sobre 32 reglas que cubrían M de 2 224 a 5 036, mezclando los cuatro kcap) daba −0,0374 por
1 000 aristas globalmente y −0,0259 restringida a kcap 6-7. Acá, con `kcap` **quieto**, sale −0,0270 y
−0,0340. La perilla de densidad reproduce sola, dentro de un solo valor de `kcap`, el efecto que O3-D
atribuía al barrido entero.

---

## 6. Pregunta 2 — a M fijo, ¿qué queda de `kcap`?

**Sobre la masa: prácticamente nada, y lo poco que hay va en la dirección contraria.**

| M objetivo | n | R²(kcap) | media kcap=6 | media kcap=7 | Δ (7 − 6) | t | p | MWU p |
|---|---|---|---|---|---|---|---|---|
| 3 800 | 12 | 0,100 | 0,0852 | 0,0868 | **+0,0017** | −1,05 | 0,321 | 0,335 |
| 4 000 | 12 | 0,006 | 0,0783 | 0,0781 | **−0,0003** | +0,24 | 0,817 | 0,747 |
| 4 200 | 12 | 0,000 | 0,0744 | 0,0745 | **+0,0001** | −0,06 | 0,951 | 1,000 |

> **R² MEDIO de `kcap` sobre la masa, a M fijo = 0,035**

Modelos anidados sobre las 36 corridas:

| modelo | R² | coeficientes |
|---|---|---|
| masa ~ M | **0,825** | M β = −3,028×10⁻⁵ (t = −12,64 · p = 2×10⁻¹⁴) |
| masa ~ kcap | **0,002** | kcap β = +0,0005 (t = +0,28 · **p = 0,784**) |
| masa ~ M + kcap | 0,828 | M β = −3,030×10⁻⁵ (t = −12,56) · kcap β = +0,00058 (t = +0,76 · **p = 0,453**) |
| masa ~ M + kcap + meandeg | 0,829 | kcap β = +0,0011 (**p = 0,441**) · meandeg β = +0,00043 (**p = 0,660**) |

Correlación parcial **r(kcap, masa | M) = +0,131** (p = 0,45) — no sólo no es significativa: **el signo
es el opuesto** al que tendría un efecto directo de `kcap` en el sentido de O3-D.

**La cota, que es el número más informativo de todo el informe.** El coeficiente ingenuo de `kcap`
medido en O3-D era **−0,0279 de masa por unidad de kcap**. Acá, con la densidad igualada, ese
coeficiente vale **+0,00058 con IC95 % = [−0,00097, +0,00213]**. El extremo negativo del intervalo es
**el 3,5 % del coeficiente ingenuo**. Es decir: estos datos son compatibles con que `kcap` no haga
absolutamente nada a la masa por su cuenta, y **descartan al 95 % que le quede siquiera un 4 % del
efecto** que se le atribuía.

### 6.1 Sobre la geometría, en cambio, sí queda algo — con una advertencia grande

Los dos endpoints geométricos **no** se comportan como la masa:

| endpoint | R²(M) a kcap fijo | R²(kcap) a M fijo | r parcial de kcap dada M | qué pasa al agregar `meandeg` |
|---|---|---|---|---|
| **clustering** | 0,291 | **0,496** | **−0,701** (p = 3×10⁻⁶) | kcap → p = 0,417 · **meandeg t = +6,42** (p = 3×10⁻⁷) |
| **pendiente corregida** | 0,162 | 0,262 | **+0,391** (p = 0,020) | kcap → p = 0,520 · meandeg p = 0,432 |
| κ_V agregado | 0,046 | 0,014 | −0,015 (p = 0,930) | — |
| fracción de masa | **0,836** | 0,035 | +0,131 (p = 0,453) | — |

A M igualado, **kcap=6 cierra 2,8 veces más triángulos que kcap=7** (clustering 0,0038-0,0048 vs
0,0010-0,0016 en M = 4 000 y 4 200; t = 4,04 y 5,86 · MWU p = 0,004 y 0,002), y la pendiente de
kcap=7 queda ~0,04 por encima de la de kcap=6. Son residuales reales y reproducibles **en la
geometría**.

**Pero el confound que queda declarado, y hay que declararlo fuerte:** *no se puede fijar M, `meandeg`
y `kcap` a la vez con este generador*. A M igualado, `r(kcap, meandeg | M) = −0,840`: las reglas
kcap=6 de esta grilla arrancaron de grafos Erdős-Rényi más densos (meandeg 4,6-6,4) que las kcap=7
(4,1-4,6), porque ésa es la única forma de que un tope más apretado termine en las mismas 4 000
aristas. Y `meandeg` predice el clustering casi perfectamente a M igualado (**r parcial = +0,879**):
al meterlo en la ecuación, el coeficiente de `kcap` sobre el clustering **se va a cero**
(t = +0,82 · p = 0,417) mientras `meandeg` se queda con todo (t = +6,42).

Traducido: el residual de clustering es atribuible a **de cuán denso partió la red**, no al tope de
vecinos en sí. Es el mismo tipo de espejismo que O3-D detectó con `kcap` y el grado medio, un nivel
más abajo. **La masa no tiene este problema**: ni `kcap` ni `meandeg` la explican una vez que está M
(p = 0,441 y p = 0,660).

### 6.2 El detalle que sí es informativo aunque esté confundido

Aun con el confound, el hecho crudo se sostiene y merece quedar anotado: **hay dos grafos con la misma
cantidad exacta de aristas, uno con casi tres veces más triángulos cerrados que el otro, y los dos
acretan la misma masa** (0,0783 vs 0,0781 a M = 4 000; 0,0744 vs 0,0745 a M = 4 200). El clustering
del grafo relacional, a densidad igualada, **no se traduce en masa** en este pipeline.

---

## 7. Contra la recta de densidad de O3-D

| M | recta de densidad de O3-D predice | F7-01 kcap=6 | F7-01 kcap=7 |
|---|---|---|---|
| 3 800 | 0,0913 | 0,0852 | 0,0868 |
| 4 000 | 0,0838 | 0,0783 | 0,0781 |
| 4 200 | 0,0763 | 0,0744 | 0,0745 |

La pendiente coincide; el nivel de F7-01 queda sistemáticamente ~0,005 por debajo de la recta de
O3-D, lo cual es esperable: aquella recta se ajustó sobre M de 2 224 a 5 036 y una relación levemente
convexa deja la interpolación del medio un poco alta. Lo que importa para esta tarea es que **las dos
columnas de F7-01 son la misma columna**.

---

## 8. Qué quedó medido (sin cerrar nada)

1. **La colinealidad se rompió de verdad**: r(kcap, M) pasó de +0,984 a +0,008, VIF de 47,8 a 1,00,
   con los dos `kcap` a la misma densidad media (3 999,7 vs 4 002,3).
2. **A `kcap` fijo, M manda**: R² = 0,836 (0,822 en kcap=6, 0,850 en kcap=7), con la pendiente que
   predecía la recta de densidad de O3-D.
3. **A M fijo, `kcap` no mueve la masa**: R² = 0,035, Δ ≤ 0,0017 en los tres niveles, p ≥ 0,32 en los
   tres, y el coeficiente con IC95 % que **descarta más del 3,5 % del efecto ingenuo**.
4. **`kcap` sí mueve la geometría a M igualado** (clustering ×2,8 · p = 0,002; pendiente +0,04 ·
   p = 0,020) — pero ese residual se disuelve al controlar por `meandeg`, que a M igualado está
   inevitablemente atado a `kcap` (r parcial = −0,840). Queda como confound declarado, no resuelto.
5. **Un grafo con casi el triple de triángulos cerrados y la misma cantidad de aristas acreta la misma
   masa.** El clustering, a densidad igualada, no se traduce en masa en este pipeline.

**Preguntas que esto abre** (para que Alexis decida, no propuestas de cierre):

- El confound `meandeg` se podría atacar con un factorial de tres factores (`kcap` × M × `meandeg`),
  pero el mapeo dice que las tres perillas no son independientes en el generador congelado: fijar dos
  determina casi la tercera. ¿Vale la pena, o el aparato de este generador ya dio lo que podía dar?
- El punto de kcap 5 × 6 quedó a **37 aristas** de existir (3 454 vs 3 491 con 1 200 candidatas).
  Con ~3 000-4 000 candidatas más probablemente aparezca una franja angosta de solape, y sería una
  **réplica independiente** de la pregunta 2 en otro punto del eje `kcap`. Costo estimado: ~12 min de
  mapeo + 8-12 corridas.
- kcap=4 y kcap=5 **no pueden** entrar a un factorial de M con el generador congelado (topes de M
  2 665 y 3 454, muy por debajo de la banda de 6-7). Si esa parte del eje importa, la única vía sería
  igualar M en el otro sentido: subir el M de kcap=4 en lugar de bajar el de kcap=7 — lo que requiere
  tocar `RANGO_MEANDEG` o `RANGO_KCAP`, es decir, modificar un archivo congelado.
- Este resultado toca directamente al `η² = 0,619` de O2-B y al `η² = 0,949` de O3-D: siguen siendo
  ciertos como descripción del barrido de `kcap`, pero acá queda medido que el vehículo es M.

---

## 9. Archivos de esta tarea

| archivo | qué es |
|---|---|
| `cs090_fase7_f701_factorial.py` | mapeo barato de M + selección de la grilla + worker (motor, grafo, clustering, IC) + Phantom serial + consolidación |
| `cs090_fase7_f701_analizar.py` | los dos R² del criterio congelado sobre los 4 endpoints + colinealidad + confound `meandeg` + figura |
| `cs090_fase7_f701_generar_ic.sh` | generación de las 36 IC en paralelo |
| `cs090_fase7_f701_plan.json` | **el plan congelado** (celdas, tolerancia, n por celda, nota de la ampliación) |
| `cs090_fase7_f701_mapa_M.csv` | **el mapa: 1 200 reglas con su kcap y su M** (lo que hizo posible igualar por selección) |
| `cs090_fase7_f701_trabajos.txt` · `cs090_fase7_f701_seleccion.json` | las 36 reglas elegidas |
| **`cs090_fase7_f701_crudo.csv`** | **datos crudos: 36 filas, 33 columnas** |
| `cs090_fase7_f701_resumen_celdas.csv` | la tabla por celda de la §4 |
| **`cs090_fase7_f701_superficie.png`** | **figura de 4 paneles: masa, clustering, pendiente y κ_V vs M, una nube por kcap** |
| `cs090_fase7_f701_analisis.log` · `..._ic.log` · `..._phantom*.log` | salidas completas |
| `/Users/alexis/phantom_cs073/bateria_fase7_f701_kcapM/<rule_id>/` (×36) | IC, `cosmog.in`, logs, primer y último dump, `.sink`, `meta_regla.json`, `resultado_f701.json` |

Ningún script congelado fue modificado. No se declaró cierre ni veredicto. No se hicieron commits.

---

## Apéndice — las 36 reglas, una por una

| regla | kcap | celda M | M real | K | meandeg | clase | pendiente | clustering | fracción de masa | κ_V |
|---|---|---|---|---|---|---|---|---|---|---|
| r1161 | 6 | 3800 | 3786 | 5 | 4,35 | I | 0,603 | 0,0015 | 0,0840 | 0,400 |
| r203 | 6 | 3800 | 3797 | 6 | 4,56 | I | 0,609 | 0,0011 | 0,0875 | 0,529 |
| r1132 | 6 | 3800 | 3801 | 7 | 4,70 | I | 0,521 | 0,0022 | 0,0860 | 0,467 |
| r598 | 6 | 3800 | 3801 | 7 | 4,67 | I | 0,555 | 0,0021 | 0,0825 | 0,398 |
| r1050 | 6 | 3800 | 3808 | 4 | 4,47 | I | 0,628 | 0,0017 | 0,0875 | 0,392 |
| r949 | 6 | 3800 | 3813 | 6 | 4,94 | I | 0,582 | 0,0009 | 0,0835 | 0,456 |
| r732 | 6 | 4000 | 3997 | 5 | 6,01 | I | 0,478 | 0,0060 | 0,0790 | 0,377 |
| r470 | 6 | 4000 | 3999 | 5 | 5,23 | I | 0,680 | 0,0020 | 0,0805 | 0,414 |
| r48 | 6 | 4000 | 3999 | 5 | 4,92 | I | 0,553 | 0,0027 | 0,0780 | 0,529 |
| r954 | 6 | 4000 | 3999 | 7 | 6,40 | I | 0,554 | 0,0034 | 0,0765 | 0,404 |
| r12 | 6 | 4000 | 4001 | 7 | 6,77 | I | 0,566 | 0,0055 | 0,0795 | 0,304 |
| r73 | 6 | 4000 | 4001 | 8 | 5,81 | I | 0,474 | 0,0034 | 0,0765 | 0,367 |
| r1176 | 6 | 4200 | 4188 | 4 | 6,17 | I | 0,560 | 0,0035 | 0,0730 | 0,383 |
| r7 | 6 | 4200 | 4189 | 5 | 6,67 | I | 0,494 | 0,0056 | 0,0760 | 0,383 |
| r52 | 6 | 4200 | 4202 | 5 | 6,60 | I | 0,516 | 0,0057 | 0,0740 | 0,340 |
| r822 | 6 | 4200 | 4203 | 5 | 6,59 | I | 0,579 | 0,0065 | 0,0745 | 0,412 |
| r190 | 6 | 4200 | 4205 | 5 | 6,11 | I | 0,558 | 0,0037 | 0,0710 | 0,492 |
| r68 | 6 | 4200 | 4206 | 4 | 6,09 | I | 0,555 | 0,0038 | 0,0780 | 0,412 |
| r976 | 7 | 3800 | 3749 | 7 | 4,04 | I | 0,624 | 0,0019 | 0,0885 | 0,414 |
| r75 | 7 | 3800 | 3798 | 4 | 4,11 | I | 0,595 | 0,0011 | 0,0900 | 0,443 |
| r1119 | 7 | 3800 | 3807 | 5 | 4,29 | I | 0,670 | 0,0010 | 0,0905 | 0,417 |
| r964 | 7 | 3800 | 3822 | 6 | 4,15 | I | 0,638 | 0,0021 | 0,0840 | 0,442 |
| r1181 | 7 | 3800 | 3845 | 6 | 4,15 | I | 0,681 | 0,0004 | 0,0850 | 0,463 |
| r343 | 7 | 3800 | 3851 | 7 | 4,09 | I | 0,628 | 0,0010 | 0,0830 | 0,382 |
| r352 | 7 | 4000 | 3985 | 5 | 4,29 | I | 0,588 | 0,0007 | 0,0810 | 0,407 |
| r1179 | 7 | 4000 | 3993 | 8 | 4,42 | I | 0,581 | 0,0014 | 0,0775 | 0,429 |
| r157 | 7 | 4000 | 4003 | 4 | 4,34 | I | 0,454 | 0,0021 | 0,0760 | 0,452 |
| r463 | 7 | 4000 | 4020 | 8 | 4,40 | I | 0,603 | 0,0006 | 0,0785 | 0,396 |
| r875 | 7 | 4000 | 4027 | 8 | 4,36 | I | 0,591 | 0,0004 | 0,0795 | 0,404 |
| r368 | 7 | 4000 | 4051 | 7 | 4,49 | I | 0,627 | 0,0009 | 0,0760 | 0,377 |
| r141 | 7 | 4200 | 4162 | 6 | 4,72 | I | 0,548 | 0,0020 | 0,0765 | 0,426 |
| r308 | 7 | 4200 | 4167 | 7 | 4,60 | I | 0,618 | 0,0019 | 0,0740 | 0,314 |
| r336 | 7 | 4200 | 4168 | 5 | 4,60 | I | 0,592 | 0,0019 | 0,0775 | 0,458 |
| r991 | 7 | 4200 | 4178 | 5 | 4,75 | I | 0,577 | 0,0011 | 0,0715 | 0,362 |
| r454 | 7 | 4200 | 4184 | 4 | 4,51 | I | 0,559 | 0,0015 | 0,0735 | 0,302 |
| r894 | 7 | 4200 | 4232 | 5 | 4,71 | I | 0,582 | 0,0012 | 0,0740 | 0,539 |

(rule_id completo = `A2-B0-C2-f701-<r…>`. La pendiente es la corregida, con `diam_gigante`. Las 36
reglas tienen 8 sumideros y masa total 18 800.)
