# CS089 — O-N7.7 con observable ESPECTRAL (sin Phantom), en vez de masa en sumideros

**Fecha:** 9-ago-2026 · **Método:** código nuevo (`cs089_on77_espectral.py`), no toca `ON77_sistemaA_cierre.py`,
`ON77_sistemaB_cierre.py` ni `cs084_espectro_laplaciano.py` — sólo importa/reusa sus funciones. Corrida
completa en **22.0 minutos** (presupuesto disponible: 45-55 min). **No se declara cierre ni veredicto sobre
O-N7.7 — sólo se reportan números. La lectura final es de Alexis.**

---

## 0 · Por qué este observable (y qué cambia respecto de `ON77_sistemaAB_cierre_CS.md`)

Ese intento anterior testeó O-N7.7 con "masa en sumideros" (vía Phantom real) y dio resultados **en contra**
de la predicción en ambos sistemas. La crítica ya documentada de Alexis: un sumidero es un **horizonte** — la
materia que cae ahí ya no tiene más historia posible, y no emite nada de vuelta. "Masa en sumideros" mide el
**fin** de la historia de esa materia, casi lo opuesto de "dominio operativo creciente" que pide O-N7.7.

Este script mide el **espectro del grafo** (λ del laplaciano L=D-A) **antes** de que nada colapse — sobre la
malla de relaciones tal cual la construye la regla generativa, sin traducirla a Phantom. Esto evita el
horizonte por completo (no depende de qué cae a un sumidero) y, como bonus, es mucho más barato: no hace
falta correr Phantom **ni** el layout de resortes (las propiedades espectrales dependen sólo de la
adyacencia, no de las posiciones) — así que se pudieron correr **varias semillas por punto** en vez de una
sola.

**Cuatro cantidades espectrales**, reportadas todas, sin elegir de antemano cuál es "la buena":
1. **λ_max** (autovalor más grande — el armónico más agudo)
2. **λ2** (conectividad algebraica — qué tan difícil es cortar el grafo en dos pedazos)
3. **dispersión** (desviación estándar de todo el espectro — qué tan ancho "suena")
4. **dimensión espectral d_s(t)** por núcleo de calor (Tr(e^{-tL}) ~ t^{-d_s/2}), en 4 tiempos de difusión

Sistema A: N ∈ {2000, 4000, 8000}, 3 semillas de embedding por N (más N=16000 con 1 semilla, fuera del
presupuesto principal). Sistema B: N=2000 fijo, H ∈ {1,2,4,8,16}, 5 semillas de orden de revelación por H.
Todos los grafos, en ambos sistemas y en todas las corridas, quedaron en **una sola componente conexa**
(`n_componentes=1`, `giant_frac=1.000`) — así que λ2 es una medida genuina de conectividad, no un artefacto
de fragmentación.

---

## 1 · Sistema A — N variable, regla fija (malla_causal_atomos, sin barajar), SIN Phantom

| N | λ2 (media±std) | λ_max (media±std) | dispersión std_eig | d_s(t=1.0) | grado medio (control) |
|---|---|---|---|---|---|
| 2000 | 0.01989 ± 0.00081 | 11.310 ± 0.643 | 2.477 ± 0.017 | 1.967 ± 0.022 | 4.979 |
| 4000 | 0.01264 ± 0.00026 | 11.318 ± 0.051 | 2.462 ± 0.007 | 2.009 ± 0.009 | 4.952 |
| 8000 | 0.00813 ± 0.00021 | 11.414 ± 0.043 | 2.451 ± 0.003 | 2.010 ± 0.006 | 4.933 |
| 16000* | 0.00499 | 12.271 | 2.452 | 2.052 | 4.977 |

*N=16000: 1 sola semilla (tardó 14.0 min sólo esa diagonalización), no comparable 1:1 con el resto (3
semillas) — indicativo, no un punto de sweep validado.

**El grado medio queda prácticamente CONSTANTE (~4.93–4.98) en todo el sweep** — confirma que la regla
generativa de Sistema A (k=4 fijo) no está inflando la densidad del grafo al subir N, así que cualquier
patrón espectral de abajo no es un artefacto de "más N = más conexiones por nodo".

**Ganancia marginal (Δcantidad/ΔN) entre puntos consecutivos:**

| Tramo | Δλ_max/ΔN | Δstd_eig/ΔN | Δd_s(t=1.0)/ΔN — Δ total |
|---|---|---|---|
| 2000→4000 | +4.1e-6 (ruido, std≫Δ) | −7.3e-6 (ruido) | +2.1e-5 → **Δtotal +0.042** |
| 4000→8000 | +2.4e-5 (ruido) | −2.9e-6 (ruido) | +0.4e-6 → **Δtotal +0.002** |

**d_s(t) a las 4 escalas de difusión, por N (nadie se elige de antemano):**

| N | d_s(t=0.05) | d_s(t=0.2) | d_s(t=1.0) | d_s(t=5.0) |
|---|---|---|---|---|
| 2000 | 0.490 | 1.556 | 1.967 | 2.218 |
| 4000 | 0.488 | 1.552 | 2.009 | 2.298 |
| 8000 | 0.486 | 1.548 | 2.010 | 2.357 |
| 16000* | 0.487 | 1.552 | 2.052 | 2.421 |

**Lectura de Sistema A:** λ_max y la dispersión quedan **esencialmente PLANAS** desde N=2000 (las diferencias
son del tamaño del ruido entre semillas) — no crecen apreciablemente con N. λ2 **decrece** de forma monótona
pero con magnitud de caída cada vez más chica (−3.6e-6 → −1.1e-6 por N), compatible con la caída ~1/N típica
de grafos de grado acotado (una posible explicación puramente estructural, no necesariamente "saturación de
Ω_op" — ver salvedad abajo). **d_s(t=1.0) sí muestra un patrón de saturación limpio**: creció +0.042 entre
2000→4000 y prácticamente se DETUVO (+0.002) entre 4000→8000 — ganancia marginal decreciente, en la
DIRECCIÓN que predice O-N7.7(a). d_s(t=5.0) también crece con tasa decreciente (+0.080 luego +0.059) —
saturación parcial, más suave. El punto de N=16000 (1 semilla) sigue creciendo un poco en d_s(t) y en λ_max,
pero no es comparable 1:1 con el resto.

---

## 2 · Sistema B — N=2000 fijo, H variable, mecanismo de reorganización acotada YA VALIDADO, SIN Phantom

| H | λ2 (media±std) | λ_max (media±std) | dispersión std_eig | d_s(t=1.0) | grado medio | n_reorganizaciones |
|---|---|---|---|---|---|---|
| 1 | 0.01992 ± 0.0000 | 11.581 ± 0.000 | 2.451 ± 0.000 | 1.963 ± 0.000 | 4.945 | 0 |
| 2 | 0.03399 ± 0.00128 | 14.241 ± 0.348 | 2.972 ± 0.020 | 2.357 ± 0.024 | 5.715 | 6.0 |
| 4 | 0.04982 ± 0.00104 | 17.349 ± 0.590 | 3.477 ± 0.017 | 2.663 ± 0.019 | 6.412 | 29.2 |
| 8 | 0.06546 ± 0.00137 | 20.373 ± 0.621 | 3.905 ± 0.037 | 2.893 ± 0.012 | 6.960 | 118.4 |
| 16 | 0.07608 ± 0.00220 | 22.698 ± 1.179 | 4.209 ± 0.033 | 3.037 ± 0.006 | 7.359 | 329.8 |

**d_s(t) a las 4 escalas de difusión, por H:**

| H | d_s(t=0.05) | d_s(t=0.2) | d_s(t=1.0) | d_s(t=5.0) |
|---|---|---|---|---|
| 1 | 0.487 | 1.551 | 1.963 | 2.203 |
| 2 | 0.554 | 1.700 | 2.357 | 2.493 |
| 4 | 0.611 | 1.817 | 2.663 | 2.601 |
| 8 | 0.654 | 1.896 | 2.893 | 2.621 |
| 16 | 0.684 | 1.950 | 3.037 | 2.582 |

**Lectura de Sistema B — las 4 cantidades principales CRECEN de forma monótona (o casi) con H**, sin
excepción: λ2 (0.0199→0.0761), λ_max (11.58→22.70), dispersión (2.45→4.21), d_s(t=1.0) (1.963→3.037, con
tasa de crecimiento decreciente: +0.394, +0.306, +0.230, +0.144 — saturando). d_s(t=0.05) y d_s(t=0.2)
también crecen monótono. Sólo d_s(t=5.0) no es limpiamente monótono: sube hasta H=8 (2.621) y baja un poco
en H=16 (2.582). **Esta es la DIRECCIÓN que predice O-N7.7(b)** ("más historia → más capacidad"), y es
opuesta a la que dio la masa en sumideros del intento anterior.

**Salvedad importante, honesta, antes de leer esto como confirmación:** el **grado medio del grafo también
crece monótono con H** (4.945 → 7.359, +49%). El mecanismo de "reorganización acotada" (tal como está
implementado, reusado tal cual de `ON77_sistemaB_cierre.py`) no mantiene el número de aristas fijo entre
distintos H: cada vez que un nodo nuevo se revela en un batch posterior, elige sus k=4 vecinos más cercanos
entre TODOS los ya revelados (incluidos los muy antiguos), agregando aristas nuevas hacia nodos viejos; la
reconsideración acotada (que sí poda) sólo AFECTA una arista por vez y no compensa ese ingreso neto. Con más
batches (H más alto), hay más "rondas" de nodos nuevos añadiendo aristas hacia nodos viejos, así que el
grafo termina más denso. **Esto es estructuralmente análogo al confound de "más masa = más estructura" que
la propia versión de masa fija de Sistema A corrigió explícitamente** — acá, en el grafo de Sistema B, ese
control de densidad NO está puesto. No se puede afirmar limpiamente que el crecimiento espectral con H
confirme O-N7.7(b) hasta separar "más historia" de "más aristas totales" — sería necesario, en un paso
futuro, fijar el número de aristas (o el grado medio) igual en todos los H y repetir la medición.

---

## 3 · Comparación explícita contra el resultado con masa en sumideros

| | Masa en sumideros (`ON77_sistemaAB_cierre_CS.md`) | Espectro (este informe) |
|---|---|---|
| **Sistema A** (N↑) | Ganancia marginal POSITIVA y grande (+0.35 masa/N, 2000→4000) — **SIN saturación**, en contra de O-N7.7(a) | **MIXTO**: λ_max y dispersión ya PLANOS desde N=2000 (sin crecimiento que saturar); λ2 decae hacia 0 (posible artefacto ~1/N); **d_s(t=1.0) SÍ satura limpio** (+0.042 → +0.002) — señal a favor de la DIRECCIÓN de O-N7.7(a) que la masa no mostró |
| **Sistema B** (H↑) | Masa DECRECE monótona (1118.6→705.0) — dirección OPUESTA a O-N7.7(b) | **λ2, λ_max, dispersión y d_s(t≤1.0) CRECEN monótonos** con H — dirección **IGUAL** a la que predice O-N7.7(b), es decir **OPUESTA** al resultado de masa. Pero con el confound de densidad de aristas creciente sin controlar (ver §2) |

**Resumen de la comparación, sin declarar cierre:** el observable espectral **NO reproduce la misma
dirección** que la masa en sumideros en ninguno de los dos sistemas. En Sistema A, donde la masa no mostró
ningún indicio de saturación, el espectro (en particular d_s por núcleo de calor) sí muestra una saturación
limpia. En Sistema B, donde la masa cayó con H, el espectro sube con H de forma consistente en las 4
cantidades — un giro de 180° respecto de la lectura anterior — aunque con la salvedad de densidad de aristas
no controlada que hay que resolver antes de leerlo como confirmación. **No es una repetición del mismo
resultado con otro disfraz, ni una confirmación limpia de O-N7.7: es una lectura distinta, en ambos casos
más cercana (aunque no idéntica) a la dirección que predice el nodo, con una salvedad metodológica concreta
pendiente de resolver en Sistema B.**

---

## 4 · Explicación en simple, con analogía

La vez pasada medimos "cuánto metal fundido cayó al fondo del horno" (la masa en los sumideros) — y eso mide
el FIN de la historia de esa materia, no su capacidad de seguir procesando. Esta vez, en cambio, **golpeamos
la campana ANTES de fundir nada** y escuchamos cómo suena — el espectro del grafo es literalmente eso: las
frecuencias de resonancia de la red de relaciones, tal como está, sin que nada haya colapsado todavía.

- **Sistema A** es como comparar la misma campana hecha con más bronce del mismo tipo (más partículas, mismo
  "molde" de conexión: cada átomo sigue conectándose sólo con sus 4 vecinos más cercanos, ni más ni menos).
  El tono más agudo de la campana (λ_max) y su timbre general (dispersión) casi no cambiaron al agregar más
  bronce — suena prácticamente igual desde el principio, como si la forma sonora ya estuviera fijada con
  relativamente poco material. Sólo una nota particular (la profundidad de la difusión de calor a t=1.0) sí
  siguió subiendo un poco entre 2000 y 4000 partículas, y luego se quedó casi quieta entre 4000 y 8000 — como
  si esa nota específica sí se hubiera "asentado" en su tono final, justo el patrón de "cada vez rinde
  menos" que predice la teoría.
- **Sistema B** es como preguntar otra vez "si el bibliotecario reorganiza la MISMA biblioteca más veces
  (más pasadas), ¿la biblioteca queda más rica o más pobre?" — pero esta vez, en vez de pesar los libros que
  cayeron al fondo de un pozo, escuchamos cómo suena toda la estantería. Y esta vez SUENA más rico con más
  reorganizaciones: más agudo, más ancho, más profundo — justo lo contrario de lo que había mostrado la masa.
  Pero hay que decirlo con la misma honestidad con la que se reportó lo anterior: cada vez que el
  bibliotecario reorganiza más veces, TAMBIÉN termina agregando más estanterías nuevas conectadas entre sí en
  el camino (el grafo terminó con 49% más conexiones promedio por nodo entre H=1 y H=16) — así que no se sabe
  todavía si suena más rico porque la HISTORIA genuinamente construye más capacidad, o simplemente porque,
  con este mecanismo tal como está armado, reorganizar más veces también deja, sin querer, más hilos
  físicos conectados en total.

---

## 5 · Qué falta / limitaciones honestas

- N=16000 (Sistema A) corrió con **1 sola semilla** (14.0 min sólo esa diagonalización densa) — no
  comparable 1:1 con el resto del sweep (3 semillas); es indicativo, no un punto validado.
- La caída de λ2 con N en Sistema A es compatible con una explicación puramente estructural (escalamiento
  ~1/N genérico de grafos de grado acotado), no necesariamente con "saturación de Ω_op" en el sentido de la
  teoría — no se investigó cuál de las dos lecturas es correcta.
- **El hallazgo de Sistema B (crecimiento espectral con H) tiene un confound sin resolver: el grado medio
  del grafo crece un 49% entre H=1 y H=16** — sería necesario repetir la medición fijando el número de
  aristas (o el grado medio) igual en todos los H para saber cuánto del crecimiento espectral es "historia"
  genuina y cuánto es simplemente "más conexiones totales".
- Sólo se probó UN mecanismo de reorganización (el ya validado de `ON77_sistemaB_cierre.py`, sin re-tunear
  `TAM_MUESTRA_RECONSIDERACION` ni las semillas base) y UNA regla generativa para Sistema A
  (`malla_causal_atomos`, k=4 fijo) — no se exploró si otras reglas darían espectros distintos.
- No se corrió el diagnóstico de espaciado de niveles (Poisson vs. Wigner-Dyson/GOE) que sí usó CS084 — el
  encargo pidió específicamente λ_max, λ2, dispersión y dimensión, no ese tercer diagnóstico.

**No se declara cierre ni veredicto sobre O-N7.7.** Se reportan los números tal como salieron, con las
salvedades metodológicas que se encontraron en el camino — la lectura final es de Alexis.

---

## Archivos

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs089_on77_espectral.py` — código nuevo, no toca
  `ON77_sistemaA_cierre.py`, `ON77_sistemaB_cierre.py` ni `cs084_espectro_laplaciano.py` (sólo importa).
- `cs089_on77_espectral_resultado.json` — datos completos (crudo por semilla + resúmenes por N/H).
- `cs089_sistemaA_espectral.csv`, `cs089_sistemaB_espectral.csv` — datos crudos planos (una fila por
  semilla).
- `logs/cs089_on77_espectral_run.log` — bitácora de ejecución completa.
