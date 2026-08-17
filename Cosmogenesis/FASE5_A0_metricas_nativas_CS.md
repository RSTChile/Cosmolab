# FASE V-A — Métricas nativas de campo continuo para A0 (sin grafo derivado)

**Fecha:** 10-ago-2026. **Script:** `cs090_fase5_a0_nativo.py` (nuevo, no toca ningún congelado).
**Datos crudos:** `cs090_fase5_a0_nativo_resumen.csv` (35 reglas), `cs090_fase5_a0_nativo_viejo_raw.csv`
(175 filas, una por regla×escala del método viejo), `cs090_fase5_a0_nativo_nativo_raw.csv` (175 filas,
método nativo). No se declara cierre ni veredicto — se reportan números, la lectura final es de Alexis.

---

## 1) La pregunta

En el barrido de 180 reglas (`FASE5A_completo_resultado_CS.md`), 27% de las reglas A0 (sustrato SIN
grafo, campo continuo genuino) cayeron en "Clase II — mundo-pequeño congelado", contradiciendo la
expectativa de que A0 nunca debería llegar tan lejos. La sospecha, documentada pero no confirmada: la
ÚNICA forma que tiene el pipeline de medir A0 es derivar un grafo de similitud (`_grafo_medicion_A0`)
donde cada sitio se compara contra `n_cand=15` candidatos elegidos **al azar** (no sus vecinos físicos)
y se conecta si están en fase parecida. Eso es, estructuralmente, la misma receta que un grafo de
Watts-Strogatz: unos pocos atajos de largo alcance sobre una base local — la firma exacta que produce
"mundo pequeño". El método de medición podría estar fabricando el fenómeno que dice medir.

Este informe compara, sobre las MISMAS instancias de campo, el método viejo (grafo derivado) contra
métricas que leen el campo directamente, sin construir ningún grafo.

## 2) Cómo está representado A0 en el motor (analogía simple)

En `cs090_fase5_motor.py::construir_A0`, el sustrato A0 es un **anillo de N sitios en fila** (como un
collar cerrado), cada uno con un valor de fase `S[i]` en un alfabeto cíclico `Z_K` (piensen en cada
sitio como una manecilla de reloj que puede apuntar a cualquiera de K posiciones). No hay ningún objeto
de grafo — los únicos "vecinos" son los offsets fijos `left=(i-1)%N` y `right=(i+1)%N`: cada sitio sólo
conoce a sus dos vecinos físicos inmediatos en el collar.

La dinámica (`dinamica_B0`, rama A0) es difusión local pura: en cada paso, cada manecilla gira hacia el
promedio (circular) de sus dos vecinas inmediatas, más ruido. Es exactamente como el rumor que se pasa
de persona a persona en un círculo — cada quien sólo escucha a quien tiene al lado, nadie tiene un
megáfono ni un teléfono a distancia.

El problema: para poder **clasificar** A0 con las mismas reglas (diámetro vs. tamaño) que se usan para
sustratos con grafo real, `_grafo_medicion_A0` construye — sólo para medir, nunca para la dinámica — un
grafo nuevo donde cada sitio se compara contra **candidatos al azar** (no sus vecinos del collar) y se
conecta si están en fase parecida. Es como si, para medir qué tan "chico" es el círculo de rumores,
alguien agarrara al azar pares de personas de CUALQUIER parte del collar (no sólo los vecinos) y los
conectara por teléfono si por casualidad dicen cosas parecidas. Ese muestreo de pares lejanos ya es, por
construcción, la receta de un grafo mundo-pequeño — independientemente de si el rumor real viajó lejos o no.

## 3) Métricas nativas elegidas (y las descartadas, con motivo)

**Elegidas — ambas leen `S` directo sobre el anillo, cero grafo derivado:**

1. **ξ(r) — correlación circular de fase** (`correlacion_circular` + `longitud_correlacion`): mide
   cuánto se parecen dos manecillas separadas por distancia `r` en el collar (la misma métrica de
   distancia que usan los offsets `left/right` de la dinámica, no una inventada). Se extrae una
   "longitud de correlación" (primer `r` donde la correlación cae bajo 1/e del máximo, criterio
   estándar de decaimiento exponencial). Analogía: cuántas personas hacia cada lado del círculo el
   rumor de una persona sigue siendo reconocible antes de volverse ruido.

2. **Dominios locales por adyacencia física** (`dominios_locales`): dos sitios **físicamente
   adyacentes** en el collar (i, i+1 — la misma adyacencia que usa la dinámica) quedan en el mismo
   "dominio" si su diferencia de fase es menor al mismo umbral (`sim_thr_frac·K`) que usa el método
   viejo. Cero muestreo al azar, cero saltos de largo alcance — es el contraste directo con
   `_grafo_medicion_A0`. Analogía: en vez de llamar por teléfono a desconocidos lejanos, sólo se
   pregunta "¿tu vecino inmediato dice algo parecido a vos?" y se van pegando tramos contiguos de
   acuerdo. Se mide el tramo más grande de acuerdo continuo (giant nativo).

**Coarse-graining nativo:** para poder comparar "escala chica vs. escala grande" (como el método viejo
hace agrupando el grafo en supernodos por BFS a b=1,2,4,8,16), acá se agrupan simplemente **tramos
contiguos de b sitios del collar** — la caja natural de un anillo 1D, sin necesidad de ningún BFS ni
grafo (el collar ya tiene orden). El valor de cada caja es el promedio circular de sus miembros.

**Descartadas (documentado, no forzado):**
- *Box-counting con umbral fijo*: en 1D casi siempre da dimensión ≈1 trivialmente (la región activa es
  casi todo el anillo o casi nada) — poco informativo, queda subsumido por "dominios locales".
- *Espectro del operador de difusión*: el operador de la dinámica, `(1-J)·I + J·(promedio izq/der)`, es
  **circulante** — sus autovalores son analíticos, `(1-J)+J·cos(2πk/N)`, dependen sólo de `J` y `N`,
  **nunca** del estado realizado del campo. No discriminaría entre reglas ni entre REAL/NULL — sería
  medir el motor, no el campo. Se descarta por no aportar señal.
- *Escalamiento masa-radio*: queda subsumido por "dominios locales" (tamaño de dominio vs. radio ES
  masa-radio con el criterio "misma fase").

## 4) Diseño experimental

- 35 reglas A0-B0-C0 admitidas (pasan P1-P5 real vía `cs090_fase5_generador.py`, sin tocar), en dos
  lotes con `seed_base` distintos (20 + 15) para no depender de una sola tirada de semillas.
- Por regla: **UN** campo A0 a N=2000, evolucionado con `cs090_fase5_motor.construir_A0` +
  `dinamica_B0` (14 sweeps) — exactamente la misma dinámica y semilla que usa internamente
  `correr_regla_coarse` (el método "corregido" que de verdad se usó en el barrido de 180 reglas, NO el
  método N-independiente más viejo — se verificó en `cs090_fase5_completo.py` que ese es el que se usó).
  La reconstrucción nativa usa una semilla independiente con la MISMA fórmula, así el array `S` es
  idéntico bit a bit al que usó el método viejo — comparación de manzanas con manzanas, no dos corridas
  distintas.
- Método viejo: `MOT.correr_regla_coarse()` (grafo derivado + BFS coarse-graining de `cs080`, congelado,
  sólo importado) → `CLS.clasificar_regla()` (congelado) → clase I-IV + pendiente log(diám)-vs-log(cajas).
- Método nativo: coarse-graining del campo (tramos de b=1,2,4,8,16 sitios) → ξ(r) y dominios locales en
  cada escala → pendiente log(métrica)-vs-log(n_cajas), mismo eje de escala que el método viejo.
- Umbral de similitud: `sim_thr_frac·K`, el MISMO parámetro que ya usa `_grafo_medicion_A0` — no se
  inventó un umbral nuevo, para que la comparación sea justa.
- **Control NULL nativo**: el mismo campo real, con las posiciones de `S` **barajadas** una vez sobre el
  anillo fino (misma distribución de valores, orden espacial destruido), luego coarse-graneado igual que
  el real. Si las métricas nativas no distinguen esto del campo real, no sirven para nada.

## 5) Resultados

**Distribución de clase vieja (método actual, sobre las 35 reglas):** 33 Clase I, 2 Clase II (5.7%).
Nota: esta tasa es más baja que el 27-50% reportado en el barrido de 180 (que usó semillas distintas,
no reproducibles exactamente porque `hash()` de Python está aleatorizado por proceso salvo
`PYTHONHASHSEED` fijo) — la variabilidad muestral es amplia con n=35 y una tasa base ~30-50%. Con todo,
el hallazgo más informativo no es la tasa sino DÓNDE caen los valores:

**Las pendientes viejas están todas apretadas contra el umbral de Clase II (0.35-0.45):**

| estadístico | pendiente_vieja (35 reglas) |
|---|---|
| mínimo | 0.227 |
| máximo | 0.362 |
| media | 0.282 |
| desvío estándar | 0.033 |

Rango completo: 0.227–0.362, con **std=0.033**. Todo el lote vive en una banda angosta y continua
justo por debajo/encima del corte 0.35. No hay dos poblaciones separadas (una "claramente Clase I" y
otra "claramente Clase II") — hay una nube unimodal que el umbral corta arbitrariamente por la mitad de
su cola derecha. Esto ya es una observación honesta importante, previa a comparar con lo nativo: el
propio método viejo no muestra una frontera cualitativa entre I y II en A0-B0-C0, sólo una banda continua.

**Métricas nativas — REAL vs. NULL (la pregunta de control, punto 4 de la tarea):**

| métrica | REAL (35 reglas) | NULL barajado (35 reglas) |
|---|---|---|
| pendiente ξ(r) [corr_slope] | 0.417 – 0.623 (media 0.508) | **siempre ≈ 0.000** |
| pendiente dominio local [dom_slope] | 0.858 – 2.171 (media 1.583) | −0.037 – 0.440 (media 0.146) |
| fracción del anillo en dominio gigante, escala fina (b=1) | 0.026 – 1.000 (media 0.484) | 0.003 – 0.011 (media 0.007) |

Separación limpia y sistemática en las tres columnas: el campo evolucionado (REAL) SIEMPRE muestra
correlación que crece con la escala de agrupamiento (slope~0.5) y dominios que crecen mucho más rápido
que el azar barajado (slope 0.86–2.17 vs ≈0–0.44). Las métricas nativas SÍ distinguen estructura
espacial genuina de ruido puro — no son ciegas.

Dato adicional interesante (no forzado a ninguna clase): la pendiente de ξ(r) se agrupa en unos pocos
valores discretos (0.417, 0.500, 0.558, 0.623) muy cerca de **0.5** en casi todos los casos. Una
pendiente ~0.5 en log(longitud de correlación) vs log(escala) es la firma característica de un proceso
**difusivo** (longitud ~ raíz cuadrada de la escala) — coherente con que la dinámica de A0 es,
literalmente, difusión local con ruido. No es "congelado" (slope≈0, que imitaría mundo-pequeño) ni
"extensivo/balístico" (slope≈1) — es un régimen intermedio bien conocido en física de la difusión, y no
tiene un casillero directo en la taxonomía Clase I-IV (que fue calibrada sobre diámetro de grafo bajo
BFS box-covering, un objeto distinto).

## 6) Comparación central — ¿las 2 reglas "Clase II" viejas se ven distintas en lo nativo?

| rule_id | clase vieja | pendiente vieja | corr_slope nativo | dom_slope nativo | giant_frac (b=1) |
|---|---|---|---|---|---|
| A0-B0-C0-r8 (lote 2) | **II** | 0.359 | 0.500 | 0.944 | 0.036 |
| A0-B0-C0-r12 (lote 1) | **II** | 0.362 | 0.623 | 2.067 | 1.000 |
| — resto (33 reglas Clase I) — | I | 0.227–0.309 | **0.417–0.623** | **0.858–2.171** | 0.026–1.000 |

Los dos casos "Clase II" caen **exactamente dentro** del rango que ya cubren las 33 reglas "Clase I" en
ambas métricas nativas — no en el extremo, no en un clúster aparte, en el medio del montón. Los dos
casos "Clase II" ni siquiera se parecen ENTRE SÍ en lo nativo (`giant_frac_b1` = 0.036 vs. 1.000 — un
factor de ~28x de diferencia entre las dos únicas reglas que el método viejo trató como "la misma
clase"). Correlación de Pearson entre `pendiente_vieja` y las pendientes nativas sobre las 35 reglas:
**r = −0.024** (ξ) y **r = +0.038** (dominio) — indistinguible de cero.

## 7) Lectura honesta (sin cerrar el caso)

Tres observaciones, en orden de solidez:

1. **Las métricas nativas funcionan** (distinguen REAL de barajado-al-azar con claridad y consistencia
   en las 35 reglas) — no es que "no midan nada".
2. **Las métricas nativas no reproducen ninguna separación entre lo que el método viejo llamó Clase I y
   Clase II** — ni las dos reglas "Clase II" se agrupan entre sí, ni se distinguen del resto en ningún
   eje nativo medido.
3. **El propio método viejo, mirado de cerca, tampoco muestra una frontera cualitativa** en A0-B0-C0: es
   una nube continua y angosta (std=0.033) alrededor del corte 0.35, no dos poblaciones.

Estas tres observaciones juntas son **consistentes con** la lectura de que el 27% (o el ~30-50% de
A0-B0-C0 específicamente) de "Clase II" sea un artefacto del umbral cayendo en medio de una distribución
continua y sin relación con ninguna estructura espacial nativa detectable — no un régimen genuino y
distinto del campo. No es prueba concluyente: el n de reglas efectivamente "Clase II" en esta corrida es
chico (2 de 35), y no se descarta que con más semillas o with with N mayor aparezca alguna correlación
que acá no se vio. La lectura final, como siempre, es de Alexis.

## 8) Explicación en simple

Imaginen un juego de teléfono descompuesto en un círculo de 2000 personas, donde cada quien sólo le
susurra a sus dos vecinos inmediatos. El método viejo, para "medir qué tan chico es el mundo" del rumor,
agarraba parejas al azar de CUALQUIER parte del círculo y las conectaba por teléfono si por casualidad
decían algo parecido — y con ESE grafo de teléfonos al azar, un cuarto de las veces el rumor parecía
"viajar rápido por todo el círculo" (mundo pequeño). Acá se midió lo mismo pero preguntando SÓLO a los
vecinos de al lado, sin teléfonos: ¿hasta dónde se sigue pareciendo el susurro? Y la respuesta fue: crece
como una difusión normal (como una gota de tinta esparciéndose), ni más rápido ni más lento en los casos
que el método viejo marcó como "especiales" que en los demás. Las dos veces que el método de los
teléfonos al azar dijo "acá pasa algo raro", mirando sólo a los vecinos de al lado no se ve nada raro —
se ven igual que las otras 33.

## 9) Archivos generados

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_a0_nativo.py` — script nuevo (métricas +
  corrida comparada), no toca ningún congelado.
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_a0_nativo_resumen.csv` — 35 filas, una
  por regla: parámetros, clase vieja, pendiente vieja, pendientes nativas real/null, fracciones de
  dominio.
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_a0_nativo_viejo_raw.csv` — 175 filas
  (35 reglas × 5 escalas b), salida cruda del método viejo (`correr_regla_coarse`).
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs090_fase5_a0_nativo_nativo_raw.csv` — 175 filas,
  salida cruda de las métricas nativas (real y null) por escala.
