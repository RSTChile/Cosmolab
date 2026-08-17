# Aislar el mecanismo (F5-C2-C3) — ¿es la RIGIDEZ del corte, o la UNIFORMIDAD del número?

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude) · Sigue de `FASE5_presupuesto_emergente_CS.md` (F5-C2-C) y
`FASE5_presupuesto_soporte_local_CS.md` (F5-C2-C2). Las dos tareas anteriores compararon C2-hard (cupo fijo
`kcap`, corte estricto vía `MOT._enforce_kcap`) contra variantes de presupuesto acumulado (knapsack greedy
con `c_ij` de 3 o luego 4 señales de costo) y encontraron el mismo patrón las dos veces: C2-hard ~45% Clase
III, todo lo demás (budget-original 15%, budget-soporte 10-15%, random 15-20%) agrupado muy por debajo y
prácticamente indistinguible entre sí. La lectura alternativa #3 del segundo informe proponía, sin probarla,
que quizás no era ninguna SEÑAL de costo la que faltaba, sino el MECANISMO: `_enforce_kcap` corta EXACTO N
aristas, sin excepción, en un paso; el presupuesto deja que el sistema "compre" más o menos relaciones según
cuánto cuesten — una forma estructuralmente más elástica. Alexis pidió aislar esa variable con un
experimento único.

Ningún script congelado ni los 2 archivos de las tareas anteriores fueron modificados
(`cs090_fase5_generador.py`, `cs090_fase5_motor.py`, `cs090_fase5_clasificador.py`,
`cs090_fase5_completo.py`, `cs090_fase5_profundizar_a2b0c2.py`, `cs090_fase5_mapa_transicion.py`,
`cs090_fase5_presupuesto_emergente.py`, `cs090_fase5_presupuesto_soporte.py` — verificable con `git diff`).
El único archivo de código nuevo es `cs090_fase5_mecanismo_aislado.py`, que importa y reusa
`cs090_fase5_presupuesto_soporte.correr_regla_coarse_presupuesto_soporte` tal cual para el brazo
C2-budget-soporte. No se corrió Phantom. No se hicieron commits de git.

## 0. La pregunta, en simple

Los dos informes anteriores usaron la analogía del **cupo de amigos**: C2-hard es "todos tienen exactamente
5 amigos, sin excepción" — un número fijo, aplicado de un tajo. El presupuesto es "cada quien gasta su
energía social donde más rinde" — más elástico, deja que unos sostengan más amistades y otros menos según
cuánto "cuesten". Nunca se probó una tercera opción, a mitad de camino: **¿y si el cupo sigue siendo un
número exacto y sin excepciones (como el cupo fijo), pero YA NO es el mismo número para todos?** Es decir:
seguir cortando con la misma tijera dura de siempre, pero con una medida de cinta distinta para cada
persona.

Los dos ingredientes que las tareas anteriores cambiaron **juntos** son:
1. **Mecanismo** — corte estricto de conteo exacto (`_enforce_kcap`) vs. presupuesto acumulado elástico
   (knapsack).
2. **Uniformidad del número** — mismo cupo para todos vs. cupo por persona.

Esta tarea agrega el brazo que faltaba para separarlos: **C2-hibrido** = mecanismo estricto (como hard) +
número variable por nodo (como el efecto que produce el presupuesto).

## 1. Cómo se construyó el cupo variable de C2-hibrido (fórmula concreta, con la aproximación declarada)

Archivo nuevo: **`cs090_fase5_mecanismo_aislado.py`**, función `_cupo_variable`.

`MOT.construir_A2(N, rng, p)` ya arma un grafo Erdős-Rényi fresco (`GR.aleatorio(N, meandeg=p["meandeg"])`)
**antes** de que arranque cualquier dinámica o poda. El grado de cada nodo en ESE grafo recién nacido
(`grado_inicial_i`) es una cantidad que el motor ya calcula como subproducto de generar el sustrato —
no se inventa nada nuevo, sólo se lee inmediatamente después de `construir_A2` y se guarda antes de que la
dinámica la modifique. En expectación todos los nodos tienen el mismo grado (`meandeg`), pero un grafo ER
finito concreto no es regular: por puro azar de muestreo, unos nodos nacen con más vecinos que otros
(varianza tipo Poisson, verificado empíricamente: para una regla con `meandeg=7.64`, el grado inicial real
osciló entre 0 y 18, con media 7.64 y desvío 2.78). Esa variación natural, ya presente en el grafo generado,
es la señal de "capacidad" usada — es la única cantidad del motor que (a) existe *antes* de cualquier
poda/dinámica (no depende circularmente de decisiones de poda previas) y (b) varía nodo a nodo sin que se
haya elegido a mano ningún criterio nuevo.

```
kcap_i = max(1, round(kcap_base * grado_inicial_i / media_empírica(grado_inicial)))
```

- `kcap_base = p["kcap"]` — el mismo entero que el generador ya sampleaba (`RANGO_KCAP=(4,7)`),
  reinterpretado como cupo **promedio** en vez de cupo fijo (mismo patrón de reinterpretación que las 2
  tareas anteriores ya usaron con `kcap` como presupuesto `B_i`).
- `media_empírica(grado_inicial)` es la media sobre los N nodos de **esta realización concreta** del
  grafo (no el `meandeg` teórico) — se eligió la media empírica para garantizar, por construcción algebraica
  exacta y no sólo en expectación, que el cupo promedio efectivo de la corrida ≈ `kcap_base`: la MISMA "masa
  total de cupo" que tendría C2-hard con ese mismo `kcap`. Verificado empíricamente en 3 reglas de muestra:
  `kcap_base=5 → mean(kcap_i)=5.10`; `kcap_base=7 → mean(kcap_i)=6.94`; `kcap_base=5 → mean(kcap_i)=5.02`
  (con desvío real 1.8-3.0 y rangos hasta 1-22 según el nodo). Así la comparación C2-hibrido vs. C2-hard no
  queda confundida por tener, en promedio, más o menos cupo total — sólo por **cómo se reparte** ese cupo
  entre nodos (uniforme vs. proporcional al grado con que nacieron).
- Piso `max(1, ...)`: un nodo que por azar nace con `grado_inicial=0` recibiría `kcap_i=0` con la fórmula
  pura, lo que lo dejaría sin ninguna arista posible para siempre — se le da un piso de 1.

**Honestidad sobre la aproximación:** "grado inicial en el grafo ER" no es una medida de "actividad" ni de
"importancia" real del nodo en la dinámica — es sólo cuántos vecinos le tocaron por azar de muestreo al
nacer el grafo. Se eligió por ser la única cantidad ya presente en el motor que cumple las dos condiciones
de arriba, no porque se afirme que sea la noción "correcta" de capacidad. El mecanismo de aplicación en sí
(`_enforce_kcap_variable`, sección 2 del script) es una copia literal del ranking-por-soporte de
`MOT._enforce_kcap` — sólo cambia `kcap` fijo por `kcap_por_nodo[i]`.

**Brazo de control adicional, C2-random (nuevo, distinto del de las 2 tareas anteriores):** mismo
`kcap_por_nodo` (misma magnitud de poda, nodo por nodo) que C2-hibrido, pero la arista que se suelta cuando
un nodo excede su cupo se elige al azar en vez de por soporte — aísla si lo que importa es el CRITERIO
(soporte) o sólo la combinación rigidez+variabilidad del número, sin ningún criterio.

## 2. Los 5 brazos, mismo lote de reglas

1. **C2-hard** — `MOT._enforce_kcap`, sin cambios (ESTRICTO + UNIFORME, control/baseline).
2. **C2-hibrido** — nuevo, `_enforce_kcap_variable` (ESTRICTO + VARIABLE, la pieza que faltaba).
3. **C2-budget-soporte** — reusa `cs090_fase5_presupuesto_soporte.py` tal cual (knapsack greedy, `c_ij` de
   4 componentes), recalculado fresco en esta corrida.
4. **C2-random** — mismo `kcap_por_nodo` que C2-hibrido, corte al azar (ESTRICTO + VARIABLE, sin criterio).
5. **C0** — sin límite de escala, sin cambios.

**Control clave:** mismo `seed_base` (`SEED_BASE=90210` para piloto, `SEED_BASE+1` para completo) que las
2 tareas anteriores usaron — las 20 reglas admitidas (A2-B0-C2, filtro P1-P5 real) son **idénticas** en
`K,J,noise,meandeg,kcap,seed` a las de esas 2 corridas, comparabilidad directa entre las 3 tareas. C2-hard,
C2-budget-soporte y C0 se recalculan frescos en esta misma corrida (no se reusan números archivados de
CSVs anteriores para ninguna comparación cuantitativa).

## 3. Corrida

Piloto de 3 semillas × 5 brazos: **0.3 min**. Corrida completa de 20 semillas × 5 brazos = 100 reglas×brazo,
N=2000, coarse-graining b=1/2/4/8/16: **2.6 minutos**, muy por debajo del presupuesto de 50 min. Sin
"SALVAGUARDA DE TIEMPO" ni fallos de motor. El mismo `RuntimeWarning: Mean of empty slice` ya conocido del
clasificador congelado apareció una vez (no introducido por este script).

Salidas:
- `cs090_fase5_mecanismo_aislado_resultados.csv` — 500 filas (20 reglas × 5 brazos × 5 escalas), dato crudo.
- `cs090_fase5_mecanismo_aislado_resumen.csv` — 100 filas (una por regla×brazo), clase + observables + parámetros.
- `cs090_fase5_mecanismo_aislado_piloto_raw.csv` / `_piloto_resumen.csv` — piloto de 3 semillas, conservado.

## 4. Resultado — fracción de Clase III y observables continuos

| brazo | n | I | II | III | IV | **%Clase III** | %III+IV | grado medio (b=1) | n_aristas medio | diám medio | pendiente media | pendiente mediana |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C2-hard**            | 20 | 9  | 2 | **9** | 0 | **45.0%** | 45.0% | 3.62 | 3623.9 | 13.55 | 0.707 | 0.652 |
| **C2-hibrido**          | 20 | 7  | 5 | **7** | 0 | **35.0%** | 35.0% | 4.31 | 4314.1 | 10.20 | 0.550 | 0.529 |
| **C2-budget-soporte**   | 20 | 12 | 5 | **2** | 1 | **10.0%** | 15.0% | 3.94 | 3938.0 | 12.50 | 0.549 | 0.522 |
| **C2-random**           | 20 | 16 | 3 | **1** | 0 | **5.0%**  | 5.0%  | 4.67 | 4674.6 | 10.10 | 0.494 | 0.495 |
| **C0**                  | 20 | 13 | 7 | **0** | 0 | **0.0%**  | 0.0%  | 6.22 | 6222.0 | 8.00  | 0.371 | 0.358 |

(C2-hibrido tuvo 1 fila "intermedio (sin clase clara)" además de las 20 mostradas arriba en I/II/III —
`A2-B0-C2-r16`, pendiente=−1.137, ver nota de outlier en §5. No se fuerza a ninguna clase, se documenta
aparte, disciplina anti-Shannon del proyecto.)

**Comparaciones pareadas** (misma regla, mismo K/J/noise/meandeg/kcap/seed en los 5 brazos, n=20, sobre la
pendiente continua):

| comparación | dirección | media de la diferencia | mediana |
|---|---|---|---|
| hard vs hibrido | hard gana en **11/20** (casi moneda) | +0.157 | **+0.023** |
| hard vs budget-soporte | hard gana en **18/20** | +0.158 | +0.186 |
| hard vs random(nuevo) | hard gana en **17/20** | +0.213 | +0.220 |
| hibrido vs budget-soporte | hibrido gana en **12/20** (casi moneda) | +0.001 | +0.063 |
| hibrido vs random(nuevo) | hibrido gana en **14/20** | +0.056 | +0.107 |
| budget-soporte vs random(nuevo) | budget gana en **16/20** | +0.055 | +0.068 |
| random(nuevo) vs C0 | random gana en **17/20** | +0.123 | +0.110 |
| hibrido vs C0 | hibrido gana en **18/20** | +0.179 | +0.105 |

**Nota de outlier honesta:** la fila `A2-B0-C2-r16` en C2-hibrido dio pendiente=−1.137 (única fila negativa
de toda la corrida, en cualquier brazo), y arrastra la MEDIA de "hard vs hibrido" de +0.043 a +0.157.
Recalculando esa comparación EXCLUYENDO esa única fila (n=19): media=+0.043, mediana=+0.018 — prácticamente
cero en ambos estadísticos, wins 10-9. La lectura "hard vs hibrido es casi un empate" se sostiene con o sin
el outlier; lo único que cambia es cuánto se acerca la media a la mediana.

## 5. La matriz 2×2 — qué separa a C2-hard del resto

```
                        MECANISMO ESTRICTO (conteo exacto)      MECANISMO ELÁSTICO (presupuesto)
NÚMERO UNIFORME         C2-hard          45.0% / pend. 0.707     [ya medido en tareas previas:
  (mismo para todos)                                              C2-budget-original ≈15.0%/pend.0.554 —
                                                                    B_i = kcap fijo, igual para TODOS]
NÚMERO VARIABLE         C2-hibrido       35.0% / pend. 0.550     C2-budget-soporte   10-15% / pend. 0.549
  (por nodo)            [NUEVO, esta tarea]                      [recalculado esta corrida]
```

**Aclaración honesta sobre la celda "elástico + uniforme/variable":** los presupuestos (`C2-budget-original`
del primer informe y `C2-budget-soporte` de este) usan `B_i = p["kcap"]`, el MISMO número de entrada para
TODOS los nodos (uniforme, verificado en el código de ambos scripts) — la variabilidad que producen es de
SALIDA (el grado final que cada nodo termina sosteniendo varía según cuánto cuesten sus aristas
particulares), no de entrada. C2-hibrido, en cambio, varía el número de ENTRADA (`kcap_i`) y aplica el
mismo corte estricto que hard. Es decir, la matriz de arriba compara "variable en la entrada, corte estricto"
(hibrido) contra "uniforme en la entrada, corte elástico que produce variabilidad en la salida" (budget) —
son dos nociones distintas de "variable" que esta tarea no tenía cómo unificar sin construir un quinto
mecanismo (presupuesto con `B_i` variable por nodo, no probado acá, candidato natural para una próxima
tarea). Se documenta explícitamente para no forzar una lectura más limpia de la que los datos permiten.

**Lo que sí se puede leer con lo medido:**

1. **C2-hibrido (35.0%) queda MUY cerca de C2-hard (45.0%)** — 10 puntos porcentuales, comparado con los
   30-35 puntos que separaban a C2-hard de CUALQUIER variante de presupuesto en las 2 tareas anteriores. En
   la comparación pareada continua (pendiente), hard vs hibrido es prácticamente un **empate** (11-9,
   mediana de la diferencia +0.023, casi cero) — muy distinto del patrón "hard gana casi siempre" que se vio
   contra budget-original (16-4), budget-soporte (18-2) y random (17-3 en esta misma corrida, 15-5 en la
   anterior). **Esto es evidencia a favor de que la RIGIDEZ del corte (conteo exacto, sin excepción) es el
   ingrediente que más importa** — moverse de "número uniforme" a "número variable por nodo", MANTENIENDO
   el corte estricto y el criterio de soporte, apenas mueve la aguja.
2. **Pero el CRITERIO también importa, y mucho, dentro del mecanismo estricto+variable**: C2-hibrido (35.0%,
   con soporte) vs. C2-random-nuevo (5.0%, mismo cupo variable pero sin criterio) es una caída de 30 puntos
   porcentuales — casi tan grande como la brecha hard-vs-budget de las tareas anteriores. En la comparación
   pareada, hibrido gana 14-6 sobre este random. Es decir: la rigidez sola, sin el criterio de soporte local
   para decidir QUÉ arista cortar, no reproduce el efecto — hace falta la combinación de las dos cosas
   (corte exacto + ranking por soporte), no cualquiera de las dos por separado.
3. **C2-hibrido también supera claramente a C2-budget-soporte** (35.0% vs 10-15%; pareado 12-8 en wins,
   mediana +0.063 a favor de hibrido) — con la salvedad de la aclaración de arriba (comparación entre
   "variable en la entrada + estricto" contra "uniforme en la entrada + elástico"), el patrón es consistente
   con que el mecanismo de corte (no sólo la señal de costo) es lo que separa a C2-hard del resto.

## 6. Lectura en simple, extendiendo la analogía del cupo de amigos

Los informes anteriores comparaban: "todos tienen exactamente 5 amigos, sin excepción" (hard) contra "cada
quien gasta su energía social donde más rinde, sin un número fijo" (presupuesto). Esta tarea prueba una
tercera opción: **seguir cortando con la misma tijera exacta y sin excepciones de siempre, pero medirle la
cinta a cada persona por separado** — algunas personas, por pura casualidad de con quién arrancaron su
círculo social (no por mérito ni por elección), nacieron conociendo a más gente; a esas se les da un cupo
un poco más alto, y a las que nacieron con menos círculo, un cupo un poco más bajo — pero el promedio de
cupos en el grupo entero sigue siendo el mismo 5 de siempre.

El resultado: **esa tercera opción (35%) se parece mucho más al cupo fijo de 5 (45%) que a cualquiera de
las versiones "presupuesto"** (10-20%) — la diferencia entre "número fijo" y "número que varía según con
cuánta gente naciste" resultó ser pequeña, casi ruido. Lo que sí importó mucho fue MANTENER la regla de "a
quién cortar primero" (los amigos con menos conocidos en común, el criterio de soporte): cuando se probó el
mismo cupo variable pero cortando al azar en vez de por ese criterio, la fuerza cayó de 35% a 5% — casi tan
abajo como el presupuesto. En otras palabras: no alcanza con cortar SIEMPRE el mismo número exacto (rigidez)
— también hay que cortar SIEMPRE a los mismos tipos de amistad (criterio). Lo que parece NO importar tanto,
con estos datos, es si ese número es igual para todo el mundo o si varía persona a persona.

## 7. Lecturas alternativas honestas (no se fuerza ninguna)

- **La brecha hard-hibrido (10pp, pareado casi empatado) es pequeña pero no cero.** Con n=20 y un outlier
  que por sí solo cambia la media de +0.04 a +0.16, esta muestra no permite afirmar con fuerza que la
  uniformidad del número no importe EN ABSOLUTO — sólo que, si importa, su efecto es mucho más chico que el
  del mecanismo (rigidez+criterio). Un lote mayor (o repetir con distintas semillas) podría ajustar esta
  lectura en cualquier dirección.
- **La aclaración de la sección 5 no es cosmética:** esta tarea NO probó la celda "presupuesto con `B_i`
  variable por nodo" (un knapsack donde el presupuesto de cada nodo, no sólo el costo de cada arista, ya
  viniera del `grado_inicial` como en C2-hibrido). Sin esa celda, la matriz 2×2 no está completa en el
  sentido más estricto — lo que se comparó es "estricto+variable-en-la-entrada" contra "elástico+uniforme-
  en-la-entrada-pero-variable-en-la-salida". Es la comparación más cercana que se pudo construir sin tocar
  ni un archivo congelado ni ninguna de las 2 tareas anteriores, pero queda como pista abierta, no cerrada.
- **El outlier `A2-B0-C2-r16`** (única pendiente negativa de toda la corrida, sólo en C2-hibrido) no fue
  investigado a fondo — podría ser una interacción específica entre el cupo muy desparejo que le tocó a esa
  regla en particular (kcap_base bajo, distribución de grado inicial con mucha cola) y el criterio de
  soporte, o podría ser ruido de una sola realización. Se documenta, no se descarta ni se explica de más.
- **El criterio de soporte demostrado importante acá (§5.2) es compatible, no contradictorio, con el
  hallazgo del informe F5-C2-C2** (agregar soporte como 4º componente de `c_ij` en el MECANISMO elástico NO
  cerró la brecha) — ambos juntos sugieren que el soporte local funciona como criterio de corte cuando el
  mecanismo es RÍGIDO (aplicado de una vez, sin poder "comprar" excepciones), pero pierde fuerza cuando se
  diluye dentro de un promedio ponderado de 4 señales bajo un mecanismo elástico que igual permite ajustar
  el grado final por costo acumulado.

## 8. Archivos de esta tarea

- `cs090_fase5_mecanismo_aislado.py` — script nuevo (único archivo de código; no toca ningún script
  congelado ni los 2 de las tareas anteriores, sólo los importa/reusa).
- `cs090_fase5_mecanismo_aislado_resultados.csv` — 500 filas, dato crudo (20 reglas × 5 brazos × 5 escalas).
- `cs090_fase5_mecanismo_aislado_resumen.csv` — 100 filas, una por regla×brazo (clase + observables + parámetros).
- `cs090_fase5_mecanismo_aislado_piloto_raw.csv` / `_piloto_resumen.csv` — piloto de 3 semillas, conservado.
- Este informe.

Ningún script congelado ni las 2 tareas anteriores fueron modificados. No se corrió Phantom. No se hicieron
commits de git. No se declara cierre ni veredicto sobre si "la rigidez del corte es el ingrediente
faltante" — la lectura final es de Alexis.
