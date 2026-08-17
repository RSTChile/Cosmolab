# Presupuesto relacional + soporte local (F5-C2-C2) — ¿faltaba "cuántos amigos en común tienen"?

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude) · Sigue de `FASE5_presupuesto_emergente_CS.md` (F5-C2-C):
ahí, un presupuesto relacional `c_ij` con 3 señales (historia + holonomía + compatibilidad de estado) NO
reprodujo la fuerza geométrica del cupo fijo `kcap` — C2-budget quedó indistinguible de podar al azar
(15.0% Clase III en ambos, contra 45.0% de C2-hard y 0.0% de C0). La lectura alternativa #2 de ese informe
(§6) proponía: quizás falta el criterio de **soporte local** (vecinos compartidos) que sí usa
`MOT._enforce_kcap` para decidir qué arista quitar, y que estaba totalmente ausente de `c_ij`. Alexis pidió
seguir esa pista.

Ningún script congelado fue modificado (`cs090_fase5_generador.py`, `cs090_fase5_motor.py`,
`cs090_fase5_clasificador.py`, `cs090_fase5_presupuesto_emergente.py` — este último se **reusa tal cual**
para recalcular el brazo C2-budget-original dentro de esta misma corrida, no se toca ni una línea). El
único archivo de código nuevo es `cs090_fase5_presupuesto_soporte.py`. No se corrió Phantom. No se hicieron
commits de git. No se declara cierre ni veredicto — se reportan números, la lectura final es de Alexis.

## 0. La pregunta

En la analogía del informe anterior: darle a cada nodo un **presupuesto de energía** en vez de un **cupo
fijo de amigos** no bastó para que el sistema "descubriera" el mismo límite que el cupo duro imponía a la
fuerza. Esta tarea agrega un ingrediente al costo de cada amistad: **cuántos amigos tienen en común los
dos** (soporte local). La idea original de `_enforce_kcap` es que una amistad sostenida por muchos
conocidos compartidos es más "barata" de mantener (más anclada, cuesta menos esfuerzo social sostenerla)
que una amistad sin nadie en común (que depende sólo del vínculo directo, más frágil y más cara). La
pregunta: ¿ese ingrediente extra hace que el presupuesto emergente empiece a comportarse como el cupo fijo?

## 1. La fórmula concreta del nuevo `c_ij`

Archivo nuevo: **`cs090_fase5_presupuesto_soporte.py`**, función `_costos_relacionales_soporte`. Reusa
verbatim los 3 componentes de `cs090_fase5_presupuesto_emergente.py` (historia, holonomía, compatibilidad
de estado — ver fórmulas en el informe anterior) y agrega un 4º:

- **Soporte local**: para cada arista viva `(i,j)`, `soporte_ij = len(adj[i] & adj[j])` — literalmente el
  mismo cálculo, sin aproximar, que usa `MOT._enforce_kcap` (motor congelado, línea 136-148) para decidir
  qué arista conservar bajo el cupo fijo.
- **Conversión a costo** (soporte alto = barato; las otras 3 señales ya son "alto = caro", hay que
  mantener la misma dirección):

  ```
  costo_soporte_ij = 1.0 / (1.0 + soporte_ij)
  ```

  Monótona decreciente (soporte=0 → costo=1.0, máximo; soporte grande → costo→0), acotada en (0,1] sin
  truncar nada, sin parámetro libre nuevo elegido a mano.

- **Normalización** — igual criterio que las otras 3 señales: cada componente se divide por su propia
  media sobre las aristas vivas (1.0 = costo promedio de esa señal). Combinación final, **pesos iguales,
  4 componentes**:

  ```
  c_ij = ( norm(historia) + norm(holonomía) + norm(compatibilidad) + norm(costo_soporte) ) / 4.0
  ```

**Decisión documentada:** agregar el 4º componente en vez de reemplazar alguno de los 3 originales — no
había evidencia de que alguno fuera "débil" (el informe anterior no aisló su aporte individual), y así
C2-budget-soporte queda como superconjunto estricto de C2-budget-original: cualquier diferencia observada
entre ambos es atribuible sólo al soporte agregado.

## 2. Los 5 brazos (mismo lote de reglas, misma corrida)

1. **C2-hard** — `MOT._enforce_kcap` sin cambios (control/baseline duro).
2. **C2-budget-original** — el `c_ij` de 3 componentes del informe anterior, **recalculado en esta misma
   corrida** (mismas 20 reglas, mismo momento) vía `PE.correr_regla_coarse_presupuesto`, sin tocar ese
   archivo — no se reusan los números archivados del CSV anterior para ninguna comparación cuantitativa.
3. **C2-budget-soporte** — el nuevo `c_ij` de 4 componentes, mismo mecanismo knapsack greedy (conserva las
   aristas más baratas hasta agotar el presupuesto).
4. **C2-random** — misma magnitud de poda que C2-budget-soporte (recalculada para esta variante, porque la
   distribución de costos con soporte no es la misma que la de 3 componentes), pero elige qué arista soltar
   al azar en vez de por costo.
5. **C0** — sin límite de escala, sin cambios.

**Control clave:** las 20 reglas admitidas (filtro P1-P5 real, A2-B0-C2) son las **mismas** que usó
`cs090_fase5_presupuesto_emergente.py` en su corrida completa (mismo `seed_base`), así que los 5 brazos de
esta corrida — incluidos C2-hard y C0, recalculados frescos acá — son comparables entre sí en el mismo
momento y bajo las mismas reglas de parámetro.

## 3. Corrida

20 semillas × 5 brazos = 100 reglas×brazo, N=2000, coarse-graining b=1/2/4/8/16 (mismo método del resto de
Fase V). Corrida completa terminada en **3.2 minutos**, muy por debajo del presupuesto de tiempo. El log
(`completo_run.log`) no muestra fallos de motor ni "SALVAGUARDA DE TIEMPO" — sólo un `RuntimeWarning: Mean
of empty slice` recurrente en `cs090_fase5_clasificador.py:60` (ya existente en el clasificador congelado,
no algo introducido por este script), que produce un `holon_ratio` en `nan` para 1 de las 100 filas
(`A2-B0-C2-r12`, brazo C2-random) — no afecta la clase asignada de esa regla (quedó Clase III por el
criterio de pendiente/z, no por holonomía).

Salidas:
- `cs090_fase5_presupuesto_soporte_resultados.csv` — 500 filas (20 reglas × 5 brazos × 5 escalas), dato crudo.
- `cs090_fase5_presupuesto_soporte_resumen.csv` — 100 filas (una por regla×brazo), clase + observables + parámetros.
- (piloto: `..._piloto_raw.csv` / `..._piloto_resumen.csv`, conservado.)

## 4. Resultado — fracción de Clase III y observables continuos

| brazo | n | I | II | III | IV | **%Clase III** | %III+IV | grado medio (b=1) | n_aristas medio | diám medio | pendiente media | pendiente mediana |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C2-hard**            | 20 | 9  | 2 | **9** | 0 | **45.0%** | 45.0% | 3.62 | 3623.9 | 13.55 | 0.707 | 0.652 |
| **C2-budget-original**  | 20 | 11 | 6 | **3** | 0 | **15.0%** | 15.0% | 3.98 | 3982.2 | 12.95 | 0.554 | 0.522 |
| **C2-budget-soporte**   | 20 | 12 | 5 | **2** | 1 | **10.0%** | 15.0% | 3.94 | 3938.0 | 12.50 | 0.549 | 0.522 |
| **C2-random**           | 20 | 13 | 3 | **4** | 0 | **20.0%** | 20.0% | 3.65 | 3654.3 | 12.70 | 0.587 | 0.560 |
| **C0**                  | 20 | 13 | 7 | **0** | 0 | **0.0%**  | 0.0%  | 6.22 | 6222.0 | 8.00  | 0.371 | 0.358 |

(La única fila Clase IV de toda la corrida es `A2-B0-C2-r16` en C2-budget-soporte: pendiente=0.845>0.7
**y** holonomía NULL/REAL=9.37≥5 — cumple el criterio de "retroalimentación cerrada" además de geometría
extensa. Con n=1 no se puede leer nada estructural en eso; se cuenta aparte por transparencia y se suma a
"%III+IV" como la lectura más generosa posible para C2-budget-soporte.)

**Comparaciones pareadas** (misma regla, mismo K/J/noise/meandeg/kcap/seed en los 5 brazos, n=20,
sobre la pendiente continua):

| comparación | dirección | media de la diferencia | mediana |
|---|---|---|---|
| hard vs budget-soporte | hard > soporte en **18/20** | +0.158 | +0.186 |
| hard vs budget-original | hard > original en **16/20** | +0.153 | +0.158 |
| hard vs random | hard > random en **15/20** | +0.120 | +0.127 |
| budget-soporte vs budget-original | soporte > original en **11/20** (casi moneda) | −0.005 | +0.021 |
| budget-soporte vs random | **random > soporte en 16/20** | −0.038 | −0.035 |
| random vs C0 | random > C0 en **20/20** | +0.216 | +0.193 |
| budget-soporte vs C0 | soporte > C0 en **19/20** | +0.179 | +0.161 |

## 5. ¿Se cerró la brecha entre budget y hard al agregar soporte local?

**No, en ninguno de los dos observables (fracción de Clase III, pendiente continua pareada).**

```
C2-hard              (45.0%, pendiente media 0.707)
    >>>
C2-random            (20.0%, 0.587)
C2-budget-original    (15.0%, 0.554)
C2-budget-soporte     (10.0% / 15.0% con IV, 0.549)
    >>>
C0                   (0.0%, 0.371)
```

C2-budget-soporte quedó prácticamente pegado a C2-budget-original (diferencia de pendiente pareada casi
nula: 11 gana/9 pierde, media −0.005 — indistinguible de ruido) y, si acaso, **por debajo** de C2-random en
la comparación pareada directa (random gana en 16 de 20 reglas). La fracción de Clase III estricta incluso
bajó (10.0% vs 15.0% del original), aunque sumando la única fila Clase IV (que es, si algo, "más" geometría
extensa, no menos) empata en 15.0% — mismo lugar que antes de agregar soporte.

Ninguna de las dos lecturas (10.0% ni 15.0%) se acerca al 45.0% de C2-hard. La brecha hard vs. budget que
dejó abierta el informe anterior (30 puntos porcentuales, pendiente pareada +0.153) sigue prácticamente
intacta después de agregar soporte local (30-35 puntos, pendiente pareada +0.158) — **agregar el 4º
componente no movió la aguja de forma detectable con esta muestra.**

**Lectura en simple, extendiendo la analogía:** si el cupo fijo de 5 amigos (hard) es lo que más empuja
hacia la red "extendida" (Clase III), y darle a cada persona un presupuesto de energía en vez de un cupo
(budget) ya se parecía más a "botar amistades al azar" que al cupo estricto, agregar "cuántos amigos en
común tienen" como parte del costo — hacer más barata la amistad bien anclada en la vecindad, más cara la
amistad sin nadie en común — **no cambió esa conclusión.** El presupuesto con soporte sigue pareciéndose
más a los otros dos mecanismos "permisivos" (budget-original, random) que al cupo estrictamente duro. El
ingrediente que faltaba, si es que falta alguno, no parece ser (sólo) el soporte local tal como se
incorporó acá.

## 6. Lecturas alternativas honestas (no se fuerza ninguna)

- **El soporte local sí se calculó y sí entró en el costo** (verificable en el código, sección 1) — esto no
  es un experimento nulo por implementación incompleta. Lo que se descarta, con esta muestra, es que ese
  ingrediente por sí solo, con peso 1/4 igual a los otros 3, sea suficiente para reproducir la fuerza del
  cupo duro.
- Puede que el **peso** importe: 1/4 puede ser demasiado poco frente a las otras 3 señales para que el
  soporte domine la decisión de poda. No se probó una versión donde el soporte pesara más (o reemplazara a
  alguna de las otras 3) — eso sigue siendo una variante no explorada.
- Puede que lo que realmente distingue a C2-hard no sea ningún ingrediente particular del costo, sino la
  **dureza del límite en sí** (conteo fijo de aristas, sin posibilidad de "comprar" más relaciones baratas)
  — algo que ningún mecanismo de presupuesto, por diseño, puede replicar, porque el punto de un presupuesto
  es precisamente permitir grado variable según costo. Esta lectura es consistente con el patrón repetido en
  ambos informes: los tres mecanismos "no estrictamente duros" (budget-original, budget-soporte, random)
  quedan agrupados entre sí (10-20%) y lejos de C2-hard (45%), sin importar qué señal específica define el
  costo.
- El hecho de que C2-random (20.0%) termine incluso un poco **por encima** de C2-budget-soporte (10-15%) en
  esta corrida (no estadísticamente fuerte con n=20, pero tampoco a favor de "el costo por soporte ayuda")
  reabre, sin resolverla, la pregunta honesta del informe anterior: cuánto del efecto de C2-hard sobre C0 es
  simplemente "existe una poda de esa magnitud" y cuánto es específico del *criterio* usado para podar.
  Con los datos de esta tarea, el criterio de soporte local no aportó una señal distinguible de azar.

## 7. Archivos de esta tarea

- `cs090_fase5_presupuesto_soporte.py` — script nuevo (único archivo de código de esta tarea; no toca
  ningún script congelado ni `cs090_fase5_presupuesto_emergente.py`, sólo lo importa/reusa).
- `cs090_fase5_presupuesto_soporte_resultados.csv` — 500 filas, dato crudo (20 reglas × 5 brazos × 5 escalas).
- `cs090_fase5_presupuesto_soporte_resumen.csv` — 100 filas, una por regla×brazo (clase + observables + parámetros).
- `cs090_fase5_presupuesto_soporte_piloto_raw.csv` / `_piloto_resumen.csv` — piloto de 3 semillas, conservado.
- Este informe.

Ningún script congelado fue modificado. No se corrió Phantom. No se hicieron commits de git. No se declara
cierre ni veredicto sobre si "el soporte local era el ingrediente faltante" — la lectura final es de Alexis.
