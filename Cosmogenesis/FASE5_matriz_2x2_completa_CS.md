# La matriz 2×2 completa (F5-C2-C4) — la celda que faltaba: presupuesto elástico con cupo variable

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude) · Cierra el arco de 4 tareas: `FASE5_presupuesto_emergente_CS.md`
(F5-C2-C), `FASE5_presupuesto_soporte_local_CS.md` (F5-C2-C2), `FASE5_mecanismo_aislado_CS.md` (F5-C2-C3) y
esta (F5-C2-C4). Las 3 tareas anteriores separaron dos ingredientes que estaban mezclados —
**MECANISMO** (corte estricto de conteo exacto vs. presupuesto elástico/knapsack) y **UNIFORMIDAD** del
número de cupo (mismo para todos los nodos vs. variable por nodo) — y dejaron 3 de las 4 celdas de una
matriz 2×2 medidas, con la cuarta documentada honestamente como pendiente en §5 y §7 de F5-C2-C3: los
presupuestos usados hasta ahora (`C2-budget-original`, `C2-budget-soporte`) usan `B_i = p["kcap"]`, EL
MISMO número de entrada para todos los nodos — nunca se había probado un presupuesto con `B_i` variable
por nodo **desde la entrada** (no sólo variable en la salida, que ya ocurre naturalmente en cualquier
presupuesto). Alexis dijo "sí" a cerrar esa celda. Esta tarea agrega **C2-presupuesto-variable** y arma la
matriz completa con las 4 celdas recalculadas.

Ningún script congelado ni los 3 archivos de las tareas anteriores fueron modificados
(`cs090_fase5_generador.py`, `cs090_fase5_motor.py`, `cs090_fase5_clasificador.py`,
`cs090_fase5_presupuesto_emergente.py`, `cs090_fase5_presupuesto_soporte.py`,
`cs090_fase5_mecanismo_aislado.py` — verificable con `git diff`). El único archivo de código nuevo es
`cs090_fase5_presupuesto_variable.py`, que importa y reusa `MA._cupo_variable` (F5-C2-C3) y
`PS._costos_relacionales_soporte` (F5-C2-C2) tal cual, sin tocar una línea de ninguno de los dos. No se
corrió Phantom. No se hicieron commits de git.

## 0. La pregunta, en simple, cerrando la analogía del cupo de amigos

Las 3 tareas anteriores probaron: "todos tienen exactamente 5 amigos, sin excepción" (C2-hard, 45.0%),
"cada quien gasta su energía social donde más rinde, mismo presupuesto para todos" (C2-budget-soporte,
10-15%), y "la misma tijera exacta de siempre, pero con una medida de cinta distinta para cada persona"
(C2-hibrido, 35.0%). Faltaba la cuarta combinación: **"cada quien gasta su energía social donde más
rinde, pero la CANTIDAD de energía que cada quien recibe para gastar ya varía desde el principio, según
con cuánta gente nació conociendo"** — el presupuesto mismo, no sólo lo que compra, varía persona a
persona desde la entrada.

## 1. Cómo se integró `B_i` variable en el mecanismo elástico

Archivo nuevo: **`cs090_fase5_presupuesto_variable.py`**. Se reusan, sin modificar una línea:

1. **`cs090_fase5_mecanismo_aislado._cupo_variable(grado_inicial, kcap_base)`** — la misma fórmula que ya
   validó C2-hibrido:
   ```
   kcap_i = max(1, round(kcap_base * grado_inicial_i / media_empírica(grado_inicial)))
   ```
   donde `grado_inicial_i` es el grado de cada nodo en el grafo Erdős-Rényi recién construido por
   `MOT.construir_A2`, leído inmediatamente después, antes de cualquier poda o dinámica — exactamente el
   mismo criterio de "capacidad" usado en F5-C2-C3, para que la comparación entre las 4 celdas no se
   confunda con un criterio de variabilidad distinto. Acá el mismo número `kcap_i` se reinterpreta como
   **presupuesto** `B_i` en vez de tope de conteo — el mismo patrón de reinterpretación que las 3 tareas
   anteriores ya usaron repetidamente con `p["kcap"]`.
2. **`cs090_fase5_presupuesto_soporte._costos_relacionales_soporte(...)`** — el `c_ij` de 4 componentes
   (historia + holonomía + compatibilidad + soporte local) tal cual, sin cambios.

El único código genuinamente nuevo es `_enforce_relacional_variable`: una copia de
`PS._enforce_relacional_soporte` (modo `'costo'`: knapsack greedy por nodo, conserva las aristas más
baratas hasta agotar el presupuesto) donde el único cambio real es `budget=budget_por_nodo[i]` en vez de
`budget` fijo para todos los nodos. Todo lo demás — cálculo de `c_ij`, orden de recorrido, criterio de
corte dentro del presupuesto de cada nodo — es exactamente el mecanismo ya usado en F5-C2-C2.

**Honestidad sobre la integración:** el grado inicial en el ER sigue siendo, como en F5-C2-C3, sólo una
cantidad de muestreo (no una medida real de "importancia"), y se usa acá con el mismo piso `max(1, ...)` y
la misma normalización por media empírica para que la "masa total" de presupuesto sea comparable a la de
C2-budget-soporte (mismo `kcap_base` de entrada). Esto significa que `C2-presupuesto-variable` es, por
diseño, un superconjunto de "mismo mecanismo que C2-budget-soporte + variabilidad de C2-hibrido" — no se
introduce ningún ingrediente adicional no probado en las 3 tareas anteriores.

## 2. Los 5 brazos, mismo lote de reglas

1. **C2-hard** — `MOT._enforce_kcap`, sin cambios (ESTRICTO + UNIFORME, control).
2. **C2-hibrido** — reusa `MA.correr_regla_coarse_hibrido(p, modo="soporte")` tal cual, recalculado
   fresco (ESTRICTO + VARIABLE).
3. **C2-budget-soporte** — reusa `PS.correr_regla_coarse_presupuesto_soporte(p, modo="costo")` tal cual,
   recalculado fresco (ELÁSTICO + UNIFORME-en-la-entrada).
4. **C2-presupuesto-variable** — NUEVO, la celda que faltaba (ELÁSTICO + VARIABLE-en-la-entrada).
5. **C0** — sin límite de escala, sin cambios.

**Control clave:** mismo `seed_base` (`SEED_BASE=90210` para piloto, `SEED_BASE+1` para completo) que las
3 tareas anteriores — las 20 reglas admitidas (A2-B0-C2, filtro P1-P5 real) son **idénticas** en
`K,J,noise,meandeg,kcap,seed` a las de esas 3 corridas, comparabilidad directa entre las 4 tareas de esta
línea. Los 5 brazos se recalculan frescos en esta misma corrida (no se reusan números archivados de CSVs
anteriores para ninguna comparación cuantitativa).

## 3. Corrida

Piloto de 3 semillas × 5 brazos: **0.4 min**, sin fallos. Corrida completa de 20 semillas × 5 brazos = 100
reglas×brazo, N=2000, coarse-graining b=1/2/4/8/16: **2.5 minutos**, muy por debajo del presupuesto de 50
min. Sin "SALVAGUARDA DE TIEMPO" ni fallos de motor. Nota de entorno: el intérprete `python3` del sistema
no tiene `scipy` instalado; se usó `./venv/bin/python3` (venv del proyecto, ya presente en el repo) para
correr piloto y corrida completa.

Salidas:
- `cs090_fase5_presupuesto_variable_resultados.csv` — 500 filas (20 reglas × 5 brazos × 5 escalas), dato crudo.
- `cs090_fase5_presupuesto_variable_resumen.csv` — 100 filas (una por regla×brazo), clase + observables + parámetros.
- `cs090_fase5_presupuesto_variable_piloto_raw.csv` / `_piloto_resumen.csv` — piloto de 3 semillas, conservado.

## 4. Resultado — fracción de Clase III y observables continuos

| brazo | n | I | II | III | IV | otro | **%Clase III** | %III+IV | grado medio (b=1) | n_aristas medio | diám medio | pendiente media | pendiente mediana |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C2-hard**                | 20 | 9  | 2 | **9** | 0 | 0 | **45.0%** | 45.0% | 3.62 | 3623.9 | 13.55 | 0.707 | 0.652 |
| **C2-hibrido**              | 20 | 7  | 5 | **7** | 0 | 1 | **35.0%** | 35.0% | 4.31 | 4314.1 | 10.20 | 0.550 | 0.529 |
| **C2-budget-soporte**       | 20 | 12 | 5 | **2** | 1 | 0 | **10.0%** | 15.0% | 3.94 | 3938.0 | 12.50 | 0.549 | 0.522 |
| **C2-presupuesto-variable** | 20 | 12 | 5 | **2** | 1 | 0 | **10.0%** | 15.0% | 4.51 | 4513.3 | 11.10 | 0.555 | 0.526 |
| **C0**                      | 20 | 13 | 7 | **0** | 0 | 0 | **0.0%**  | 0.0%  | 6.22 | 6222.0 | 8.00  | 0.371 | 0.358 |

(La única fila "otro" es `A2-B0-C2-r16` en C2-hibrido, "intermedio — sin clase clara", pendiente=−1.137 —
la misma regla que ya era outlier en F5-C2-C3. En C2-budget-soporte y C2-presupuesto-variable, esa misma
regla `r16` cae en Clase IV — único caso Clase IV de cada brazo, ver nota §6.)

**Comparaciones pareadas** (misma regla, mismo K/J/noise/meandeg/kcap/seed en los 5 brazos, n=20, sobre la
pendiente continua):

| comparación | dirección | media de la diferencia | mediana |
|---|---|---|---|
| hard vs hibrido (fila ESTRICTO: uniforme→variable) | hard gana en **11/20** (casi moneda) | +0.157 | +0.023 |
| hard vs budget-soporte (columna UNIFORME: estricto→elástico) | hard gana en **18/20** | +0.158 | +0.186 |
| hard vs presupuesto-variable (diagonal: estricto+uniforme→elástico+variable) | hard gana en **18/20** | +0.153 | +0.156 |
| hibrido vs presupuesto-variable (fila VARIABLE: estricto→elástico) | hibrido gana en **11/20** (casi moneda con outlier) | −0.004 | +0.021 |
| budget-soporte vs presupuesto-variable (fila ELÁSTICO: uniforme→variable) | variable gana en **12/20** | −0.005 | −0.015 |
| hibrido vs budget-soporte | hibrido gana en **12/20** (casi moneda) | +0.001 | +0.063 |
| presupuesto-variable vs C0 | variable gana en **19/20** | +0.184 | +0.128 |
| hibrido vs C0 | hibrido gana en **18/20** | +0.179 | +0.105 |

**Nota de outlier (misma regla `A2-B0-C2-r16` que en F5-C2-C3):** excluyendo esa fila, "hibrido vs
presupuesto-variable" pasa de media=−0.004/mediana=+0.021 (n=20) a **media=+0.101/mediana=+0.025 (n=19,
hibrido gana 11/8)** — la única comparación de esta tarea con outlier tan influyente. "budget-soporte vs
presupuesto-variable" (la otra comparación de fila) NO tiene ese problema: prácticamente idéntica con o
sin la fila (media/mediana ya cerca de cero en ambos casos).

## 5. La matriz 2×2 completa — las 4 celdas juntas

```
                        MECANISMO ESTRICTO (conteo exacto)      MECANISMO ELÁSTICO (presupuesto/knapsack)
NÚMERO UNIFORME         C2-hard              45.0%              C2-budget-soporte      10.0% (15.0% c/IV)
  (mismo para todos)    pendiente 0.707                          pendiente 0.549
NÚMERO VARIABLE         C2-hibrido           35.0%              C2-presupuesto-variable 10.0% (15.0% c/IV)
  (por nodo)            pendiente 0.550      [F5-C2-C3]          pendiente 0.555   [NUEVO, F5-C2-C4]
```

- **Diferencia dentro de la fila ESTRICTO** (uniforme→variable, C2-hard→C2-hibrido): **10 puntos
  porcentuales** (45.0%→35.0%).
- **Diferencia dentro de la fila ELÁSTICO** (uniforme→variable, C2-budget-soporte→C2-presupuesto-variable):
  **0 puntos porcentuales** (10.0%→10.0%, o 15.0%→15.0% contando Clase IV) — la variabilidad de entrada no
  movió la aguja NADA en esta fila.
- **Diferencia dentro de la columna UNIFORME** (estricto→elástico, C2-hard→C2-budget-soporte): **35 puntos
  porcentuales** (45.0%→10.0%).
- **Diferencia dentro de la columna VARIABLE** (estricto→elástico, C2-hibrido→C2-presupuesto-variable):
  **25 puntos porcentuales** (35.0%→10.0%).

**Lectura que sostienen estos 4 números juntos:** en las dos columnas (uniforme y variable), pasar de
mecanismo estricto a elástico produce una caída grande (35pp y 25pp respectivamente). En las dos filas
(estricto y elástico), pasar de número uniforme a variable produce una caída chica o nula (10pp y 0pp
respectivamente). Es decir: **el efecto del MECANISMO es 2.5-∞ veces más grande que el efecto de la
UNIFORMIDAD**, y esto se sostiene en ambas columnas/filas, no sólo en una. La celda nueva
(C2-presupuesto-variable) no se acercó a C2-hibrido ni a C2-hard — se quedó pegada a C2-budget-soporte,
prácticamente en el mismo número (10.0%/15.0% en ambos, pendiente media 0.549 vs 0.555, diferencia
pareada indistinguible de cero).

La comparación pareada continua es consistente con esta lectura, con una salvedad: la fila ESTRICTO
(hard vs hibrido) ya era casi un empate en F5-C2-C3 (11-9, mediana +0.023) — no es que la fila ESTRICTO
tenga un efecto grande de uniformidad que la fila ELÁSTICO no tenga; ambas filas muestran un efecto chico
de uniformidad en la pendiente pareada. Lo que separa fuertemente en la pendiente pareada es cruzar de
mecanismo *desde* C2-hard (18-2 contra budget-soporte, 18-2 contra presupuesto-variable) — el efecto de
mecanismo es más visible cuando se compara *contra el extremo estricto+uniforme* que cuando se compara
dentro de la fila variable (hibrido vs presupuesto-variable, 11-9 con el outlier, 11-8 sin él, un efecto
presente pero más modesto que el de la fila uniforme en fracción de Clase III).

## 6. Lectura en simple, cerrando la analogía del cupo de amigos

Las 4 tareas de esta línea, juntas, cuentan esta historia: empezar con "todos tienen exactamente 5 amigos,
sin excepción" (cupo fijo) da la red más "extendida" (45%). Cambiar SÓLO la rigidez del corte —dejar que
cada quien gaste su presupuesto social donde más rinda, sin un número fijo— hunde ese número a 10-15%,
sin importar si el presupuesto de partida es igual para todos o si ya varía según con cuánta gente nació
cada quien conociendo (esta tarea: 10.0% en ambos casos, número idéntico). Cambiar SÓLO la uniformidad del
número —mantener la tijera exacta y sin excepciones, pero medirle la cinta a cada persona por separado—
apenas mueve la aguja (45%→35% con la tijera dura; 10%→10% con el presupuesto elástico, es decir, nada).

En otras palabras: lo que más importa para que emerja la geometría extensa no es "cuánto cupo social tiene
cada quien" (eso varía poco el resultado), sino **si existe, en algún punto del proceso, un límite duro y
sin excepciones que corte de un tajo** — dejar que el sistema "compre" relaciones según cuánto le convenga,
aunque sea con un presupuesto repartido de forma desigual y "justa" según el punto de partida de cada
nodo, no reproduce ese efecto.

## 7. Lecturas alternativas honestas (no se fuerza ninguna)

- **La celda nueva confirma la lectura #2 de F5-C2-C3, no la #1.** F5-C2-C3 dejó dos lecturas abiertas: (a)
  "la rigidez del corte es lo que más importa" (apoyada por hard≈hibrido, 45%≈35%) y (b) que faltaba
  probar si la uniformidad importaba también DENTRO del mecanismo elástico. Con la celda nueva medida,
  budget-soporte≈presupuesto-variable (10%≈10%, diferencia pareada esencialmente cero) — el patrón se
  sostiene en la fila ELÁSTICO igual que en la ESTRICTO, reforzando que el mecanismo domina en las dos
  filas, no sólo en una donde podría haber sido casualidad de esa fila específica.
- **La brecha de fracción de Clase III entre hibrido (35%) y presupuesto-variable (10-15%) es más grande
  que lo que sugiere la pendiente pareada continua** (11-9 con el outlier, 11-8 sin él, efecto modesto) —
  el umbral de clasificación (pendiente>0.7) es sensible a la forma de la distribución, no sólo a su media:
  ambos brazos tienen pendiente media casi idéntica (0.550 vs 0.555) pero hibrido tiene más filas
  individuales por encima de 0.7 (7/20) que presupuesto-variable (2/20). Esto no contradice la lectura de
  la sección 5, pero muestra que la fracción de Clase III y la pendiente pareada continua no siempre se
  mueven en la misma magnitud — vale la pena mirar ambas, como se hizo acá.
- **La misma regla `A2-B0-C2-r16` vuelve a ser un caso especial** en las 4 tareas de esta línea (única
  pendiente negativa en C2-hibrido; único caso Clase IV en C2-budget-soporte y en C2-presupuesto-variable
  de esta corrida). No se investigó su causa a fondo — sigue documentado como caso aislado, no una
  tendencia.
- **`grado_medio` de C2-presupuesto-variable (4.51) es más alto que el de C2-budget-soporte (3.94)** —
  más cerca del grado de C2-hibrido (4.31) que del de su comparación directa de fila. Esto sugiere que
  variar el presupuesto de entrada sí cambia ALGO estructural (cuántas aristas sobreviven en total), aunque
  ese cambio no se traduce en más geometría extensa (Clase III) — el grado más alto y el diámetro más bajo
  (11.10 vs 12.50) son consistentes con una red algo más densa/compacta, pero eso no basta para cruzar el
  umbral de pendiente>0.7 con más frecuencia.
- **No se corrió un control random para esta celda** (equivalente al `C2-random` de F5-C2-C3, mismo
  `budget_por_nodo` pero soltando aristas al azar en vez de por costo) — el criterio ya se validó como
  importante en F5-C2-C3 dentro del mecanismo estricto; esta tarea priorizó completar la matriz 2×2 con las
  4 celdas principales dentro del presupuesto de tiempo, dejando ese control cruzado (elástico+variable+
  sin criterio) como pista abierta, no cerrada, si se quisiera profundizar.
- **El criterio de "capacidad" (grado inicial en el ER) sigue siendo una aproximación declarada, no una
  medida de importancia real del nodo** — la misma honestidad de F5-C2-C3 aplica acá sin cambios, y la
  consistencia del resultado (mismo criterio, misma conclusión de que domina el mecanismo) no depende de
  que ese criterio sea "el correcto", sólo de que sea el mismo en las dos celdas VARIABLE de la matriz.

## 8. Archivos de esta tarea

- `cs090_fase5_presupuesto_variable.py` — script nuevo (único archivo de código; no toca ningún script
  congelado ni los 3 archivos de las tareas anteriores, sólo los importa/reusa).
- `cs090_fase5_presupuesto_variable_resultados.csv` — 500 filas, dato crudo (20 reglas × 5 brazos × 5 escalas).
- `cs090_fase5_presupuesto_variable_resumen.csv` — 100 filas, una por regla×brazo (clase + observables + parámetros).
- `cs090_fase5_presupuesto_variable_piloto_raw.csv` / `_piloto_resumen.csv` — piloto de 3 semillas, conservado.
- Este informe.

Ningún script congelado ni las 3 tareas anteriores de esta línea fueron modificados. No se corrió Phantom.
No se hicieron commits de git. No se declara cierre ni veredicto sobre si "el mecanismo es la variable
dominante y la uniformidad no importa" — los 4 números de la matriz y las comparaciones pareadas están
arriba; la lectura final es de Alexis.
