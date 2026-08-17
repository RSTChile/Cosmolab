# Resumen descriptivo — batería EIT-3 Térmico κ_H

Solo estadística descriptiva. Ninguna conclusión sobre si hay o no "hallazgo":
eso es del investigador principal.

## Cobertura

| experimento | barridos | filas | tiempo real |
|---|---|---|---|
| A · repetición (30 semillas) | 30 | 1.800 | 168 s (14 procesos en paralelo) |
| B · multivariable (48 combinaciones × 10 semillas) | 480 | 28.800 | 1.983 s ≈ 33 min (14 procesos en paralelo) |
| C · barajado sobre A | 30 series × 1.000 barajes | 30 filas de resumen | segundos |
| C · barajado sobre B (extra, no pedido explícitamente para B pero corrido) | 480 series × 1.000 barajes | 480 filas de resumen | segundos |

0 filas con `NaN`/`Infinity` en A y en B. `saturacion_sensor=1` aparece
únicamente en `luminosidad=1.950` (el último punto del eje) en el 100% de los
barridos de A y B — ver `defectos_encontrados.md` punto 3.

## Pregunta 1 — ¿se sostiene la correlación −0,756 (huella vs entropía de la conducta) y con qué dispersión?

Correlación de Pearson entre `huella` y `entropia_abs_local`, calculada sobre
los 60 puntos de cada barrido (eje completo 0.25→1.95):

- **Experimento A (30 semillas, parámetros fijos):** r = **−0,236 ± 0,073**
  (media ± desviación entre semillas), rango **[−0,356, −0,130]**.
- **Experimento B (480 barridos, grilla completa):** r = **−0,243 ± 0,065**
  (media ± desviación sobre las 480 combinaciones semilla×parámetros).

En ningún caso, ni en A ni en B, el promedio se acerca a −0,756. La dispersión
entre semillas es considerable en términos relativos (la desviación es ~30%
de la media). Ver punto 5 de `defectos_encontrados.md`: el rango de eje usado
aquí (0.25→1.95, el que pide el encargo) es más ancho que el preset por
defecto del simulador (0.60→1.40), y la huella no es monótona en luminosidad
— esto puede explicar buena parte de la distancia con el −0,756 de referencia,
pero no se investigó corriendo el rango angosto (no estaba pedido).

## Pregunta 1b — resultado del barajado (Experimento C)

Percentil del r real dentro de la distribución nula (1.000 barajes de la
correspondencia eje↔huella, por barrido):

- **A:** percentil medio **5,39** (mediana 4,6), rango **[0,1 — 17,6]**.
  De 30 semillas, **15 caen fuera del percentil 95%** (bilateral, ≤2,5 o
  ≥97,5) de su propia distribución nula, y **7 caen fuera del percentil 99%**.
- **B:** percentil medio **4,52 ± 5,77** (media ± desviación sobre las 480
  combinaciones).

Todos los percentiles observados son bajos (ningún caso por encima de 50): la
correlación real, siendo negativa, cae siempre en la cola izquierda de la
distribución nula, nunca en la derecha. La media de la distribución nula está
centrada muy cerca de 0 (~−0,003) en todos los casos revisados, como es
esperable al barajar una correspondencia arbitraria.

## Pregunta 2 — ¿se mueve la frontera de la huella (~0,40 en el encargo) entre semillas?

Criterio usado (documentado en `defectos_encontrados.md` punto 4): luminosidad
donde `huella` toca su **mínimo global**, excluyendo puntos con
`saturacion_sensor=1`. La huella no es monótona: cae en una V pronunciada
alrededor de `luminosidad≈1,0` y se recupera después, subiendo hasta ~9 en el
extremo superior del eje.

- **Experimento A (30 semillas):** la frontera cae en **`luminosidad =
  0,99915...`** en las 30 semillas, sin excepción — desviación entre semillas
  del orden de 10⁻¹⁵ (cero a precisión de punto flotante, es decir: no se
  mueve nada dentro de esta muestra). Lo que sí varía entre semillas es la
  **profundidad** del mínimo (`huella_minima`): rango aproximado
  **[1,189 — 1,278]** según la semilla (ver `experimento_C_barajado.csv` no
  — ver `analisis_A_detalle.json` → `fronteras`).
- **Experimento B (grilla persistencia×difusión×potencia_base):** la
  frontera cae exactamente en el mismo punto (`luminosidad = 0,99915...`,
  desviación ~10⁻¹⁵) para **cada uno** de los 4 niveles de beta, los 4 de
  sigma y los 3 de potencia_base (promediando sobre semillas y los otros dos
  factores) — no se desplaza al variar ninguno de los tres parámetros
  barridos en B, dentro de esta grilla y con este criterio.

Nota: la ubicación coincide con la posición exacta de un punto de la grilla
de 60 paradas del eje (no es una coincidencia de redondeo entre semillas
distintas — es literalmente el mismo índice de parada en las 510 corridas).

## Pregunta 3 — meseta sobre luminosidad > 1,40

Comparación descriptiva del rango (máximo − mínimo) de las variables en el
tramo `luminosidad > 1,40` (20 de 60 puntos) contra el resto del eje (40 de
60 puntos):

| variable | rango en meseta (A) | rango en resto (A) | rango en meseta (B) | rango en resto (B) |
|---|---|---|---|---|
| huella | 3,985 | 5,605 | 5,162 | 6,561 |
| multiplicidad | 0,0496 | 0,2375 | 0,4038 | 0,5106 |
| acoplamiento | 0,2019 | 0,6530 | 0,3449 | 0,7663 |

En ambos experimentos el rango de variación en el tramo `>1,40` es menor que
en el resto del eje para las tres variables, pero no está cerca de cero —
sigue habiendo variación apreciable ahí (no es un tramo plano en sentido
estricto). No se investigó si el origen es un punto de equilibrio real o un
techo del instrumento (PTC): eso es parte de lo que pide el encargo dejar en
manos del investigador principal. El único techo duro detectado en esa zona
es la saturación del sensor, pero solo en el último punto del eje
(`luminosidad=1,95`), no en todo el tramo `>1,40`.

## Archivos de esta batería

- `motor.mjs` — motor físico extraído (validado bit a bit, ver `validacion.md`)
- `experimento_A_repeticion.csv` (1.800 filas), `experimento_B_multivariable.csv` (28.800 filas)
- `experimento_C_barajado.csv` (30 filas, sobre A), `experimento_C_barajado_B.csv` (480 filas, sobre B — extra)
- `analisis_A_detalle.json`, `analisis_B_detalle.json` — detalle fila por fila de fronteras y correlaciones, para auditoría
- `validacion.md`, `defectos_encontrados.md` — verificación bit-a-bit y defectos/decisiones
