# Presupuesto relacional emergente (F5-C2-C) — ¿el sistema descubre solo su límite, o se lo imponemos?

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude) · Sigue de `FASE5A_profundizar_A2B0C2_resultado_CS.md`
(bimodal I/III de A2-B0-C2 sostenido, n=30), `FASE5_auditoria_C2_resultado_CS.md` (kcap es topológico
puro, PASS), y `FASE5_mapa_transicion_C2_resultado_CS.md` (el borde nítido I↔III probablemente es
artefacto del corte de clasificación, no una transición de fase genuina).

Ningún script congelado fue modificado (`cs090_fase5_generador.py`, `cs090_fase5_motor.py`,
`cs090_fase5_clasificador.py`, `cs090_fase5_completo.py`, `cs090_fase5_profundizar_a2b0c2.py`,
`cs090_fase5_mapa_transicion.py`, y los reusados de fases anteriores `cs080_renormalizacion.py`,
`cs082_fase4_4sustratos.py`) — verificado por fecha de modificación de archivo (todos con timestamp
anterior a esta tarea) y porque el único archivo escrito fue el nuevo. No se corrió Phantom. No se
hicieron commits de git. No se declara cierre ni veredicto — se reportan números, la lectura final es de
Alexis.

## 0. La pregunta

¿Puede el sistema **descubrir** por sí mismo que no puede relacionarse con todo, en vez de que se lo
impongamos con un número fijo (`kcap`, idéntico para todo nodo, todo el tiempo)? Analogía: ¿es lo mismo
darle a cada persona un **cupo fijo de 5 amigos** (kcap) que darle un **presupuesto de tiempo/energía** y
dejar que el número de amistades que sostiene sea lo que le alcanza el presupuesto — pudiendo tener más
de 5 amigos "baratos" (relaciones fáciles) o menos de 5 si las suyas son "caras" (conflictivas)?

## 1. Cómo se construyó `c_ij` y `B_i` — honestidad sobre qué es real y qué es aproximación

Archivo nuevo: **`cs090_fase5_presupuesto_emergente.py`**. Tres señales, las tres YA EXISTENTES en
`cs090_fase5_motor.py`, ninguna inventada:

| componente de `c_ij` | de dónde sale, literalmente | ¿ya se usaba en el motor? |
|---|---|---|
| **historia** | `flip_count[e]` — cuántas veces esa arista exacta se prendió/apagó durante la corrida | Sí — es el componente "inconsistencia histórica" de `_costo_y_podar` (motor, línea 154), reusado tal cual |
| **conflicto de holonomía** | holonomía media de los triángulos que tocan la arista, vía `C82._holonomia_triangulos` | Sí — el otro componente de `_costo_y_podar`. La lógica de agregación por-arista está embebida dentro de esa función y no expuesta aparte, así que se **replicó verbatim** (mismas ~10 líneas) en el archivo nuevo — el original no se tocó |
| **reciprocidad/compatibilidad** (aproximación declarada) | diferencia circular de estado `\|S_i-S_j\| mod K` entre los dos extremos | Sí, pero para OTRO propósito: es EXACTAMENTE la métrica que `_recablear_A2` (motor, línea 118-120) ya usa para decidir si una arista es "cara" y debe caerse en el recableo co-emergente. El motor **no** trackea una cuenta explícita de interacción mutua i↔j (no hay "reciprocidad" literal en el código) — se usa esta compatibilidad de estado como proxy honesto, documentado como aproximación, no como un contador nuevo inventado |

`c_ij` = promedio simple de los 3 componentes, cada uno normalizado dividiendo por su propia media sobre
las aristas vivas (así "1.0" = arista de costo promedio — misma idea de pesos iguales que ya usa
`_costo_y_podar` para sus 2 componentes).

**`B_i`** = `p["kcap"]` — el mismo entero que YA sampleaba el generador congelado por regla
(`RANGO_KCAP=(4,7)`), **no se inventó ningún valor nuevo**. Lo único que cambia es la unidad: en C2-hard
kcap cuenta *aristas*; en C2-budget cuenta *unidades de costo*. Un nodo cuyas relaciones son todas de
costo promedio conserva ~kcap aristas (igual que antes); uno con relaciones baratas puede sostener MÁS,
uno con relaciones caras debe sostener MENOS — **el grado máximo efectivo queda determinado por cuántas
aristas caben bajo el presupuesto dado su costo real, no por un conteo fijo de entrada.**

Enforcement (`_enforce_relacional`, análogo a `_enforce_kcap` original: mismo recorrido secuencial
nodo-por-nodo 0..N-1, un nodo puede podar una arista que otro querría conservar, igual que el original):
para cada nodo, ordena sus aristas vivas por costo ascendente y se queda con las más baratas hasta que
sumar la siguiente excedería `B_i` (knapsack greedy, mismo estilo "aligerado" que el resto del motor).

## 2. Los 4 brazos

1. **C2-hard** — `MOT._enforce_kcap` sin cambios, vía `correr_regla_coarse(p, eje_C="C2")`.
2. **C2-budget** — `_enforce_relacional(modo="costo")`: conserva las aristas más baratas hasta agotar `B_i`.
3. **C2-random** — usa el **mismo cálculo** de cuántas aristas debe soltar cada nodo bajo el presupuesto
   que C2-budget (misma magnitud de poda, nodo por nodo), pero elige **cuáles** soltar al azar en vez de
   por costo.
4. **C0** — sin límite de escala, vía `correr_regla_coarse(p, eje_C="C0")`, sin cambios.

**Control clave:** se generó **un solo lote** de 20 reglas admitidas (filtro P1-P5 real, 20/20 admitidas,
0 descartadas) con `cs090_fase5_generador.generar_reglas_clase("A2","B0","C2", ...)`, y **los mismos
parámetros (K, J, noise, meandeg, kcap, seed)** se reutilizaron en los 4 brazos — cambia sólo el mecanismo
de límite de escala. Los 4 brazos parten del mismo grafo ER inicial (mismo seed). Nota honesta: las
trayectorias no quedan bit-a-bit idénticas después del primer paso de poda porque `_enforce_relacional`
consume números aleatorios adicionales que `_enforce_kcap` no consume — el control es "misma regla, mismo
punto de partida", no "mismo trazo aleatorio completo".

## 3. Piloto (3 semillas × 4 brazos = 12 corridas, 22s) y escalado

El piloto corrió limpio sin errores (0 fallas de motor); se escaló a **20 semillas × 4 brazos = 80
reglas** (dentro de la banda 60-80 pedida), N=2000, coarse-graining b=1/2/4/8/16, mismo método que el
resto de Fase V. **Tiempo total real: 2.4 minutos** (muy por debajo del presupuesto de 55-65 min) —
sobró tiempo de sobra, no hubo que recortar ningún brazo.

Salidas:
- `cs090_fase5_presupuesto_emergente_resultados.csv` — 400 filas (20 reglas × 4 brazos × 5 escalas), dato crudo.
- `cs090_fase5_presupuesto_emergente_resumen.csv` — 80 filas (una por regla×brazo), clase + observables
  continuos (pendiente, z_agg, holon_ratio, n_aristas, grado_medio, diám, giant) + parámetros.
- (piloto: `..._piloto_raw.csv` / `..._piloto_resumen.csv`, conservados para trazabilidad.)

## 4. Resultado — fracción de Clase III y observables continuos

| brazo | n | Clase I | Clase II | Clase III | Clase IV | **%Clase III** | grado medio (b=1) | pendiente media | pendiente mediana |
|---|---|---|---|---|---|---|---|---|---|
| **C2-hard**   | 20 | 9 | 2 | **9** | 0 | **45.0%** | 3.62 | 0.707 | 0.652 |
| **C2-budget** | 20 | 11 | 6 | **3** | 0 | **15.0%** | 3.98 | 0.554 | 0.522 |
| **C2-random** | 20 | 13 | 2 | **3** | 2 | **15.0%** | 3.68 | 0.591 | 0.590 |
| **C0**        | 20 | 13 | 7 | **0** | 0 | **0.0%**  | 6.22 | 0.371 | 0.358 |

**Comparaciones pareadas** (misma regla, mismo K/J/noise/meandeg/kcap/seed en los 4 brazos, n=20):

| comparación | dirección | media de la diferencia (pendiente) |
|---|---|---|
| hard vs budget | hard > budget en **16/20** reglas | +0.153 |
| budget vs random | budget > random sólo en **6/20** (empatados en la práctica) | −0.037 |
| random vs C0 | random > C0 en **20/20** reglas | +0.220 |
| hard vs C0 | hard > C0 en **19/20** reglas | +0.336 |

## 5. ¿Se sostiene la predicción del equipo?

**Predicción a poner a prueba:** C2-budget≈C2-hard > C2-random > C0.

**Lo que se observó, en los dos observables (fracción de Clase III y pendiente continua), de forma
consistente entre sí:**

```
C2-hard (45.0%, pendiente media 0.707)
    >>
C2-budget (15.0%, 0.554) ≈ C2-random (15.0%, 0.591)
    >>
C0 (0.0%, 0.371)
```

**La predicción NO se sostuvo tal como quedó operacionalizada acá.** C2-budget no se parece a C2-hard —
se parece a C2-random (misma fracción de Clase III, 15.0% vs 15.0%; pendiente pareada budget>random sólo
en 6 de 20 reglas, prácticamente una moneda al aire, diferencia media casi nula). Lo que SÍ separa
limpiamente a los tres brazos con límite (hard/budget/random) del brazo sin límite (C0) es la presencia de
*cualquier* mecanismo de poda: los tres reducen el grado medio de ~6.2 a ~3.6-4.0 y elevan la pendiente de
~0.37 a ~0.55-0.71, y esa brecha es grande y consistente (random>C0 en 20/20 reglas). Pero **dentro** del
grupo con límite, el criterio de selección (costo real vs. azar) no marcó una diferencia detectable con
esta muestra — lo único que sí importó fue si la restricción era la ESTRICTA de conteo fijo (hard) o
cualquiera de las dos versiones "más permisivas" de presupuesto (budget/random), que resultaron
prácticamente intercambiables entre sí.

**Lectura en simple, con la analogía del principio:** un cupo fijo de 5 amigos (kcap/hard) fuerza a la red
a recablear constantemente buscando compatibilidad bajo una restricción dura de CANTIDAD, y eso es lo que
más empuja hacia la geometría extendida (Clase III). Darle a cada nodo un presupuesto de energía en vez de
un cupo (budget) sí cambia el resultado — pero, en esta implementación concreta, termina pareciéndose
más a "botar amistades al azar hasta gastar el mismo presupuesto" (random) que a "el cupo fijo de
siempre". El GRADO EFECTIVO sí terminó siendo una salida del sistema (grado medio 3.98 en vez de un
kcap fijo, con dispersión propia por regla) — eso funcionó como estaba diseñado — pero esa emergencia por
sí sola, con el criterio de costo tal como quedó definido acá, no bastó para reproducir la fuerza
geométrica del cupo estrictamente fijo, ni para distinguirse de simplemente podar al azar la misma
cantidad.

## 6. Lecturas alternativas honestas (no se fuerza ninguna)

- Puede que el problema esté en la **selección específica del criterio de costo** (los 3 componentes
  usados, o sus pesos iguales), no en la idea de presupuesto emergente en sí — un `c_ij` distinto podría
  dar otro resultado. No se probaron variantes de pesos por presupuesto de tiempo.
- Puede que lo que realmente empuja a Clase III en C2-hard sea específicamente el criterio de **soporte
  local** (vecinos compartidos) que usa `_enforce_kcap`, no la dureza del límite de conteo — ese criterio
  no está presente en absoluto en `c_ij` (que usa historia/holonomía/compatibilidad, no soporte). Esto es
  compatible con, y no contradice, el hallazgo de `FASE5A_profundizar_A2B0C2_resultado_CS.md` de que
  `kcap` bajo correlaciona con más Clase III — pero ahí no se aisló si era el "kcap bajo" o el "criterio
  de soporte" el que hacía el trabajo; este experimento sugiere que podría ser más lo segundo.
- El hecho de que C2-random también se aleje tan claramente de C0 (20/20 pareado) confirma que "cualquier
  poda de esa magnitud" ya genera bastante estructura por sí sola — el candidato original del roadmap
  (`equipo-analisis-fase5-10ago2026`) planteaba esto como posibilidad a descartar, y con estos números NO
  se descarta: gran parte del efecto de C2-hard sobre C0 podría deberse simplemente a que existe una poda
  de esa escala, no a que sea por costo ni por soporte específicamente — aunque hard SÍ se distingue de
  random/budget (45% vs 15%), así que "cualquier poda" no explica el 100% de la brecha, sólo una parte.

## 7. Archivos de esta tarea

- `cs090_fase5_presupuesto_emergente.py` — script nuevo (único archivo de código de esta tarea; no toca
  ningún script congelado, sólo importa/reusa).
- `cs090_fase5_presupuesto_emergente_resultados.csv` — 400 filas, dato crudo (20 reglas × 4 brazos × 5 escalas).
- `cs090_fase5_presupuesto_emergente_resumen.csv` — 80 filas, una por regla×brazo (clase + observables + parámetros).
- `cs090_fase5_presupuesto_emergente_piloto_raw.csv` / `_piloto_resumen.csv` — piloto de 3 semillas, conservado.
- Este informe.

Ningún script congelado fue modificado. No se corrió Phantom. No se hicieron commits de git. No se
declara cierre ni veredicto sobre si "el sistema descubre por sí mismo su límite" — la lectura final es
de Alexis.
