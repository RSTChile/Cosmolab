# CS073 factorial q_E × q_T — separar identidad de aristas de orden de inserción, con el espectro del laplaciano (sin Phantom)

**Fecha:** 10-ago-2026 · **Origen:** NULL-3 (double-edge-swap + filtro de longitud) y NULL-4 (mismo
conjunto de aristas, orden de inserción rebarajado) resultaron ambos indistinguibles de REAL en masa de
sumideros (`cs073-fase2-jerarquia-completa-cierre-8ago2026`), pero cambian dos cosas a la vez de forma no
independiente: NULL-3 cambia identidad de aristas (¿cuáles existen?) y de hecho también revuelve el orden
al reconstruir el dict de adyacencia desde un `set`; NULL-4 cambia sólo el orden. Este experimento
construye un dial CONTINUO en cada eje por separado — q_E (identidad de aristas) y q_T (orden de
inserción) — para cruzarlos en una grilla factorial y ver cuál mueve la distancia espectral a REAL.
**No se corrió Phantom bajo ninguna circunstancia — todo lo de abajo se midió sobre el grafo/malla ANTES
del colapso gravitacional, con el mismo instrumento espectral que ya usó CS085.** No se declara cierre ni
veredicto — sólo se reportan números. La lectura final es de Alexis.

## Cómo se parametrizó cada eje

**q_E — identidad de aristas conservadas (adapta NULL-3).** `barajar_aristas_preservando_longitud`
(`null3_investigacion_preliminar.py`, congelado, sólo importado) ya expone el parámetro que controla
"cuánto" se baraja: `factor_swaps`, que fija el número de INTENTOS de double-edge-swap como
`n_intentos = factor_swaps * n_aristas` (cada intento se acepta o rechaza según el filtro de longitud,
`tol_relativa=0.2`, sin tocarlo). Se define:

```
factor_swaps(q_E) = round( 10 * (1 - q_E) )
```

`q_E = 1.0` → `factor_swaps = 0` → cero intentos → grafo original intacto (0 aristas distintas de REAL,
por construcción). `q_E = 0.0` → `factor_swaps = 10`, EXACTAMENTE el mismo valor que usó CS085 para
"NULL-3 puro" → debería reproducir su ~12-13% de aristas distintas. Los 3 valores intermedios (0.75,
0.5, 0.25) redondean a `factor_swaps = 2, 5, 8`. Como la aceptación depende del filtro geométrico (no
es determinista), la relación entre `q_E` nominal y el % de aristas realmente distintas se MIDE, no se
asume lineal (columna `pct_aristas_distintas` del CSV).

**q_T — orden de inserción conservado (adapta NULL-4).** NULL-4 congelado sólo conoce el barajado
COMPLETO del orden (`rng.permutation` entero, vía `null4_verificar_invarianza_orden.construir_adj_en_orden`,
también sólo importado). Para tener un dial continuo se escribió (código nuevo, no toca lo congelado):

```
orden = identidad (0..n_aristas-1)
n_tocadas = round( (1 - q_T) * n_aristas )
se eligen n_tocadas posiciones al azar (sin reemplazo) y se revuelven SOLO esas posiciones entre sí
  (el resto queda exactamente en su lugar original)
```

`q_T = 1.0` → 0 posiciones tocadas → orden idéntico al de construcción natural de `malla_causal_atomos`.
`q_T = 0.0` → el 100% de las posiciones entra en el sorteo → equivalente a un `rng.permutation` completo,
el mismo mecanismo que usa NULL-4 puro. El conjunto FINAL de aristas nunca cambia por este barajado (se
verifica con un `assert` en cada celda, no se asume) — sólo cambia la secuencia de inserción en cada
`set()` de Python.

**Aislamiento genuino de los ejes:** para cada `(q_E, semilla_swap)` se genera el grafo con aristas
barajadas UNA sola vez, y sobre ESE MISMO conjunto de aristas resultante se aplican los distintos `q_T` —
así el eje `q_T` queda comparando exactamente el mismo conjunto de aristas, sólo con distinto orden, sin
mezclar con una nueva realización del swap de `q_E`.

## Grilla usada

| eje | valores | por qué |
|---|---|---|
| q_E | {1.0, 0.75, 0.5, 0.25, 0.0} — 5 niveles | resolución fina, es el eje con expectativa de variación real |
| q_T | {1.0, 0.5, 0.0} — 3 niveles | expectativa matemática a priori de efecto NULO (ver más abajo) — se
usan 3 puntos para confirmar la planitud con un rango representativo, no 5, para no gastar presupuesto
en un eje sin variación esperada |
| semillas de swap | 5 (`seed_swap = 6000..6004`) | mismo nº que usó CS085 para NULL-3 |

Total: 5 × 3 × 5 = **75 celdas** + 1 fila de referencia REAL pura = 76 grafos, cada uno con
diagonalización densa completa (N=1599, el mismo `n` que usó CS085 — `dens_bar.npy` del pool tiene 1599
partículas, no 2000, pese al nombre de la carpeta). **Tiempo total: 255.7 s** (~4.3 min), muy por debajo
del presupuesto de 55-65 min.

**Por qué la grilla es asimétrica (documentado antes de correr, no post-hoc):** L=D-A (el laplaciano) es
una función del CONJUNTO de aristas únicamente, nunca del orden en que ese conjunto se insertó en las
estructuras de adyacencia — CS085 ya lo había confirmado para el caso extremo (NULL-4 puro:
`max|Δeigenvalue|` vs REAL = 0.000e+00 exacto). Por lo tanto se esperaba, ANTES de correr, que el eje
q_T diera el espectro IDÉNTICO para cualquier valor, sea cual sea q_E — no una tendencia a descubrir,
sino una identidad algebraica. Se corrió la grilla igual, con más resolución en q_E que en q_T, para
confirmarlo numéricamente en el caso general (no sólo en el extremo) en vez de asumirlo.

## Resultado — confirmación de la invarianza de q_T

**Para las 75 celdas, sin una sola excepción:** dentro de cada grupo `(q_E, seed_swap)`, los 3 valores de
`q_T` (1.0, 0.5, 0.0) dieron `lambda2`, `lambda_max`, `std_eig`, y toda la curva `d_s(t)` **exactamente
idénticos** (diferencia = 0.0, no "casi cero" — el chequeo automático que compara el spread dentro de
cada grupo no imprimió ninguna alerta). Ejemplo, primera fila de cada bloque:

| q_E | seed_swap | q_T=1.0 → λ2 | q_T=0.5 → λ2 | q_T=0.0 → λ2 |
|---|---|---|---|---|
| 0.75 | 6000 | 0.053034 | 0.053034 | 0.053034 |
| 0.50 | 6001 | 0.105926 | 0.105926 | 0.105926 |
| 0.00 | 6004 | 0.079763 | 0.079763 | 0.079763 |

Descomposición simple de varianza sobre `lambda2` en toda la grilla (suma de cuadrados entre grupos,
no un ANOVA formal con F-test, sólo una forma de cuantificar cuánto "mueve la aguja" cada eje):

```
SS_total = 0.1288
SS_q_E   = 0.0990  (76.9% de la varianza total)
SS_q_T   = 0.0000  (0.0% exacto)
```

No hay término de interacción que reportar: para que exista interacción, el efecto de q_T tendría que
depender del nivel de q_E — pero q_T no tiene efecto en NINGÚN nivel de q_E, así que no hay nada con qué
interactuar.

## Resultado — cómo escala q_E (colapsando sobre q_T, que no aporta variación)

| q_E | factor_swaps | λ2 media (rango) | λ2 std (entre 5 semillas) | λ_max media | % aristas distintas | d_s(t=1.0) media |
|---|---|---|---|---|---|---|
| 1.00 | 0 | 0.0199 (exacto, = REAL) | 0.0000 | 11.581 | 0.0% | 1.963 |
| 0.75 | 2 | 0.0498 (0.041–0.057) | 0.0068 | 11.592 | 2.6% | 2.083 |
| 0.50 | 5 | 0.0837 (0.064–0.106) | 0.0159 | 11.646 | 6.8% | 2.265 |
| 0.25 | 8 | 0.1072 (0.076–0.133) | 0.0252 | 11.636 | 10.7% | 2.424 |
| 0.00 | 10 | 0.1177 (0.076–0.152) | 0.0346 | 11.635 | 12.9% | 2.513 |

Lectura de esta tabla: `lambda2` y `d_s(t=1.0)` crecen de forma monótona en la MEDIA a medida que q_E baja
de 1.0 a 0.0 (misma dirección que encontró CS085: más identidad de arista destruida → más fácil de
"partir" el grafo → λ2 más alto). Pero la dispersión ENTRE semillas también crece con la cantidad de
swaps (std pasa de 0.0000 en q_E=1.0 a 0.0346 en q_E=0.0) — a partir de q_E=0.5 los rangos de niveles
adyacentes se solapan bastante (ej. el máximo de q_E=0.50 es 0.106, dentro del rango de q_E=0.25 que
llega a 0.133; el mínimo de q_E=0.25 y de q_E=0.00 casi coinciden, 0.076 en ambos). `lambda_max`, en
cambio, apenas se mueve con q_E (11.58 a 11.65, dentro del ruido de semilla) — igual que ya había
encontrado CS085, es `lambda2` (no `lambda_max`) el estadístico sensible a este tipo de cambio.

## Chequeo contra los extremos ya documentados en CS085

| escalón | documentado en CS085 (NULL-3/NULL-4 "puros") | este experimento (extremo equivalente) |
|---|---|---|
| NULL-3 puro (factor_swaps=10) | λ2 ∈ [0.077, 0.158], λ_max ∈ [11.53, 11.79] | q_E=0.0: λ2 ∈ [0.076, 0.152], λ_max ∈ [11.54, 11.89] — mismo orden de magnitud, rangos casi idénticos |
| % aristas distintas NULL-3 puro | ~12-13% (documentado en `NULL3_investigacion_preliminar`) | q_E=0.0: 12.9% de media — coincide |
| NULL-4 puro (orden 100% rebarajado) | `max|Δeigenvalue|` vs REAL = 0.000e+00 exacto | q_T=0.0 con q_E=1.0: λ2 == 0.019923 (idéntico a REAL) en el 100% de los casos |

Los dos extremos de la grilla reproducen los números ya documentados — la parametrización continua no
introdujo ningún artefacto nuevo en los bordes.

## Lectura central — ¿qué eje domina?

**q_E domina el 100% de lo que este instrumento puede detectar. q_T aporta 0%, no como hallazgo empírico
sino porque el instrumento es matemáticamente ciego a ese eje.** El laplaciano L=D-A se calcula a partir
de la matriz de adyacencia final — una tabla de "quién está conectado con quién ahora". Esa tabla no
tiene ninguna columna de "quién se conectó primero": lo mismo puede haberse formado en un orden que en
otro, la matriz final (y por lo tanto el espectro completo, no sólo λ2) sale idéntica bit a bit. No hace
falta un experimento para saber esto en el caso puro (CS085 ya lo había confirmado), pero este experimento
confirma que **sigue siendo cierto en TODA la grilla**, incluso cuando se combina con distintos grados de
alteración de identidad de arista (q_E) — no es una propiedad exclusiva del caso "topología 100% intacta".

**Esto NO significa que "el orden de formación no importa" para el sistema real** — sólo que este
instrumento en particular (el espectro puramente topológico, calculado ANTES de correr `layout_resortes`)
no puede verlo. `null4_verificar_invarianza_orden.py` (ya en disco, no tocado) había encontrado
exactamente lo contrario a nivel de POSICIONES: el mismo conjunto de aristas, insertado en distinto
orden, produce layouts de resortes con hasta 38% de diferencia en la escala típica de coordenada — el
proceso de relajación de `layout_resortes` SÍ es sensible al orden de inserción (una función iterativa
que arranca de un estado y lo va actualizando arista por arista puede terminar en un mínimo local distinto
según el camino). Es decir: el orden probablemente sí deja una huella física real, pero en las POSICIONES
finales tras la relajación, no en la topología pura que mide este diagnóstico espectral. Si se quisiera
aislar genuinamente el eje q_T con un instrumento que pueda verlo, haría falta correr `layout_resortes`
(barato, sin Phantom) para cada celda y medir algo sobre las POSICIONES resultantes (ej. un espectro
laplaciano PONDERADO por distancia geométrica, o alguna estadística directa de la nube de puntos) — no se
hizo en este experimento por estar fuera del alcance pedido (que especificaba el espectro del laplaciano
topológico, el mismo de CS085), y se documenta acá como el camino no tomado, no como una limitación oculta.

## Síntesis, en simple

Pensá la malla causal como una red de amistades: q_E es "cuántas amistades sos vos mismo (cuántos pares
de personas siguen siendo amigas, cuáles cambiaron)"; q_T es "el orden en el que esas amistades se
fueron formando con el tiempo". El instrumento que usamos acá (el espectro del laplaciano) es como una
FOTO de la red tomada HOY: muestra quién es amigo de quién en este momento, pero la foto no tiene ninguna
marca de "estos se conocieron primero, estos después". Por eso, sin importar cuánto revolvamos el ORDEN
en que se formaron las amistades (q_T), mientras el conjunto final de "quién es amigo de quién" no
cambie, la foto sale exactamente igual — no un poquito parecida, IDÉNTICA hasta el último decimal. En
cambio, si cambiamos CUÁLES son las amistades (q_E) — aunque sea la misma cantidad de amigos por
persona —, la foto sí cambia, y cambia más cuanto más mezclamos. Esto no quiere decir que el orden de
formación no haya dejado ninguna marca en la vida real de esa red (quizás sí, en cómo terminó "acomodada"
físicamente en el espacio, según ya se había visto en otro chequeo) — sólo que esta FOTO en particular
no puede verlo, hace falta un tipo distinto de instrumento (algo más parecido a un VIDEO que a una foto)
para detectar esa huella.

## Archivos

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs073_factorial_qe_qt.py` — código nuevo. No toca
  `null3_generar_ic.py`, `null4_generar_ic.py`, `null3_investigacion_preliminar.py`,
  `null4_verificar_invarianza_orden.py`, `cs085_espectro_jerarquia_cs073.py`, ni ningún otro archivo
  congelado — sólo import/lectura de sus funciones (`barajar_aristas_preservando_longitud`,
  `adj_a_lista_aristas`, `construir_adj_en_orden`, `malla_causal_atomos`, `laplaciano_denso_desde_adj`,
  `dimension_espectral`, `unfolding_local`, `estadisticas_espaciado`, `leer_ic_txt`).
- `cs073_factorial_qe_qt.csv` — datos crudos completos (76 filas: 1 referencia REAL + 75 celdas de la
  grilla 5×3×5, todas las columnas del diagnóstico espectral + metadatos de q_E/q_T/semillas).
- Este informe.

No se declara cierre ni veredicto sobre CS073, CS085, ni esta grilla — los números de arriba son el
entregable; la síntesis final es de Alexis.
