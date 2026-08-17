# CS079 — Delimitación FoF como hipótesis propia (nodo C-N4)

**Fecha:** 5-ago-2026 · **Tipo:** exploratorio, Fase I-C · **No es cierre de arco.**

## La pregunta (C-N4)

CS073 traza sumideros/grumos de gas con un criterio "amigos-de-amigos" (friends-of-friends,
FoF) de **longitud de enlace fija**. Ese criterio nunca se puso a prueba como hipótesis
propia. La pregunta que ataca este script: ¿esa frontera cae donde el gas de verdad tiene
una discontinuidad de densidad, o cualquier longitud de enlace razonable dibuja "algo
parecido a un grupo" igual, incluso sobre un campo sin estructura genuina (NULL)?

## Qué se corrió (de verdad, sobre datos ya existentes)

Script: `cs079_delimitacion_cn4.py` — ejecutado con
`venv/bin/python3 cs079_delimitacion_cn4.py` (salida completa reproducida abajo).
No corrió ninguna simulación Phantom nueva; leyó el volcado más evolucionado
(`cosmog_00500`, el último paso disponible en las 9 corridas) de:

- `/Users/alexis/phantom_cs073/bateria_n2000/ic_real/`
- `/Users/alexis/phantom_cs073/bateria_n2000/ic_null1/` .. `ic_null8/` (las 8 corridas
  NULL disponibles, no sólo 2-3 — el costo computacional era bajo así que se usaron todas)

usando `leer_dump()` de `leer_volcado_phantom.py` (importado de sólo lectura, sin tocar el
archivo — ver `INFRAESTRUCTURA_lector_phantom_CS.md` para su verificación previa).

### Método

1. Para cada corrida: k-d tree (`scipy.spatial.cKDTree`) sobre (x,y,z) de las partículas
   de gas → distancia al **8º vecino más cercano** de cada partícula → histograma de esas
   distancias.
2. Cuantificación de bimodalidad con dos métricas independientes (no se instaló
   `diptest`/`unidip` — no estaban en el venv del proyecto; se usó lo que ya trae
   scipy/numpy):
   - **Coeficiente de bimodalidad de Sarle/Pfister (BC)**, a partir de skewness y
     kurtosis muestral. BC > 5/9 ≈ 0.555 = umbral clásico de "sospecha de bimodalidad"
     (heurística, **no** es una prueba formal tipo dip test de Hartigan).
   - **Profundidad de valle relativa**: `scipy.signal.find_peaks` sobre el histograma
     suavizado ubica los dos picos más prominentes y el mínimo entre ellos; se reporta
     `(altura_pico_menor − altura_valle) / altura_pico_menor`.
3. Segunda métrica, independiente: **perfil de densidad radial** alrededor de la
   partícula de gas más densa de cada corrida (columna `rho`, calculada por
   `leer_dump()`/`calc_density` de sarracen) — cascarones log-espaciados, rho promedio
   por cascarón, pendiente log-log ajustada por mínimos cuadrados.

## Resultado (números crudos, las 9 corridas)

| corrida | n_gas | n_sinks | mediana dist-8NN | BC (Sarle) | prof. valle | dist. entre modos | pendiente log-log ρ(r) | R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ic_real** | 1774 | 8 | **3.454** | **0.803** | 0.509 | **12.42** | −0.265 | 0.481 |
| ic_null1 | 1922 | 8 | 8.819 | 0.508 | 0.461 | 5.75 | −0.516 | 0.529 |
| ic_null2 | 1927 | 8 | 9.016 | 0.526 | 0.561 | 5.25 | −0.046 | 0.035 |
| ic_null3 | 1923 | 8 | 9.121 | 0.512 | 0.516 | 5.14 | −0.531 | 0.553 |
| ic_null4 | 1928 | 7 | 9.117 | 0.518 | 0.460 | 5.60 | −1.100 | 0.913 |
| ic_null5 | 1923 | 8 | 9.192 | 0.482 | 0.497 | 4.39 | 0.031 | 0.009 |
| ic_null6 | 1923 | 8 | 9.113 | 0.480 | 0.425 | 5.27 | −0.266 | 0.324 |
| ic_null7 | 1918 | 8 | 9.095 | 0.505 | 0.464 | 5.42 | −0.188 | 0.168 |
| ic_null8 | 1923 | 8 | 8.878 | 0.533 | 0.567 | 5.93 | −0.390 | 0.464 |

Rango NULL (n=8) para referencia: mediana dist-8NN ∈ [8.82, 9.19]; BC ∈ [0.48, 0.53];
dist. entre modos ∈ [4.39, 5.93]. **REAL cae fuera de ese rango en las tres métricas, sin
solapamiento con ninguna de las 8 corridas NULL.**

JSON completo con todos los números (incluye n de partículas por cascarón, conteos de
histograma, etc.): `cs079_resultados/resultados_cs079.json`. Plots: `cs079_resultados/hist_knn.png`
y `cs079_resultados/perfil_radial.png`.

## Lectura de los números (sin cerrar nada)

**Lo que separa limpio, REAL vs los 8 NULL, sin solapamiento:**
- La **distancia mediana al 8º vecino** en REAL (3.45) es ~2.5-2.6× menor que en
  cualquiera de los 8 NULL (8.8–9.2, casi idénticos entre sí). El gas de REAL está mucho
  más compacto/concentrado en general — coherente con el hallazgo z=48.69 de CS073 (más
  masa acretada en sumideros en REAL: aquí también REAL termina con menos partículas de
  gas libres, 1774 vs ~1920-1928 en los NULL, exactamente el mismo patrón de acreción).
- El **coeficiente de bimodalidad (BC)** de REAL (0.803) supera el umbral 5/9 y a los 8
  NULL (0.48–0.53, todos por debajo del umbral).
- La **distancia entre los dos modos** detectados en el histograma es más del doble en
  REAL (12.4) que en cualquier NULL (4.4–5.9).

**Lo que NO separa — y es importante decirlo con la misma claridad:**
- La **profundidad de valle** —la métrica diseñada específicamente para capturar la
  noción intuitiva de "dos poblaciones separadas por un hueco real"— da REAL=0.509,
  **dentro** del rango que dan los 8 NULL (0.425–0.567). Los 8 histogramas NULL también
  tienen 2 picos detectados con un valle de profundidad comparable a REAL.
- Revisando el plot (`hist_knn.png`): los 8 NULL producen un patrón bimodal **muy
  reproducible** entre sí (pico en ~5, pico en ~11, casi superpuestos entre los 8) — no
  es un histograma "suave/sin estructura" como predeciría ingenuamente la hipótesis nula
  de C-N4. REAL, en cambio, tiene una forma cualitativamente distinta: un pico angosto y
  alto a distancia muy corta (~2-3) con una cola larga hacia distancias mayores, no dos
  poblaciones equivalentes con un hueco limpio entre ellas. El algoritmo de detección de
  picos igual encuentra "2 picos" en ambos casos, pero por razones distintas: en NULL
  porque hay dos modos reales del patrón numérico; en REAL porque la cola larga alcanza a
  producir un segundo máximo local menor. Esto es relevante para C-N4: **la mera
  presencia de un valle en el histograma NO es evidencia suficiente de "frontera real"**,
  porque los NULL también la tienen, de forma consistente entre las 8 corridas — sugiere
  que ese patrón de doble pico podría ser un artefacto del setup numérico compartido
  (condición inicial tipo grid/glass común a las 8 corridas NULL) y no la ausencia de
  estructura que se esperaría de un campo "sin nada".
- El **perfil de densidad radial** (pendiente log-log alrededor del pico de mayor
  densidad) no dio una señal limpia: la pendiente de REAL (−0.265) cae en medio del rango
  ruidoso de los NULL (−1.10 a +0.03), y el R² del ajuste varía mucho y sin patrón entre
  REAL y NULL (0.01 a 0.91). Con un solo dump por corrida y un único "pico" definido como
  la partícula más densa, esta métrica no tiene la resolución para discriminar aquí — se
  reporta igual, como exige el protocolo, pero no se le puede sacar una lectura firme.

## En síntesis (números, no veredicto)

- **3 de 4 métricas** (mediana dist-8NN, BC, distancia entre modos) separan a REAL de las
  8 corridas NULL sin ningún solapamiento — separación robusta y consistente.
- **1 de 4 métricas** (profundidad de valle, la más directamente ligada a la pregunta de
  C-N4 sobre "discontinuidad real") **no discrimina** — REAL cae dentro del rango NULL.
- El perfil de densidad radial no aportó señal confiable con esta muestra (n=1 dump por
  corrida, pico único).
- Caveat metodológico honesto: los histogramas NULL son sorprendentemente reproducibles
  entre las 8 corridas (mismo patrón bimodal casi calcado) — antes de leer esto como
  "ausencia de estructura confirmada", vale la pena que alguien revise si las 8 corridas
  NULL comparten alguna condición inicial/semilla que explique ese parecido, porque si es
  un artefacto compartido, cambia cómo se debe interpretar la comparación.

**No se declara cierre ni refutación del nodo C-N4.** Estos son números de un análisis
exploratorio de solo lectura sobre volcados ya existentes. La decisión sobre qué significan
para la validez del criterio FoF de CS073 —y si ameritan una batería más rigurosa (más
dumps por corrida, dip test de Hartigan formal, más semillas NULL independientes)— es de
**Alexis López Tapia**, director del proyecto.

## Archivos

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs079_delimitacion_cn4.py` — script,
  ejecutado, código autodescriptivo.
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs079_resultados/resultados_cs079.json`
  — todos los números crudos de las 9 corridas.
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs079_resultados/hist_knn.png` —
  histograma de distancias k-NN, REAL vs null1/2/3.
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs079_resultados/perfil_radial.png` —
  perfil de densidad radial, REAL vs null1/2/3.
