# Mapa de transición kcap × K para A2-B0-C2 — ¿la bimodalidad I↔III es una transición de fase genuina?

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude), tarea encargada por `equipo-analisis-fase5-10ago2026` como
frente F5-C2-B, siguiendo a `FASE5A_profundizar_A2B0C2_resultado_CS.md` (que dejó abierto que `kcap` y
`K` correlacionaban moderadamente con caer en Clase III, r=-0.43 y r=+0.45 sobre n=18, pero con solape
total de rangos y sin umbral limpio). Script nuevo: `cs090_fase5_mapa_transicion.py`. No toca ningún
script congelado (`cs090_fase5_generador.py`, `cs090_fase5_motor.py`, `cs090_fase5_clasificador.py` se
usan tal cual, sólo `import`). No se corrió Phantom. No se declara cierre ni veredicto — se reportan
números, la lectura final es de Alexis.

## Resumen de una línea

Con la grilla completa (4×5, 20 semillas/celda) la **superficie de clasificación P(Clase III) tiene un
borde marcado entre kcap=5 y kcap=6** (salto de hasta 0.75 en una sola casilla de la grilla), pero **los
observables continuos que alimentan esa clasificación (grado de mundo pequeño, tamaño de la componente
gigante, clustering, diámetro) se mueven todos de forma suave y monótona con kcap, sin ningún salto ni
divergencia propios** — el borde nítido en P(III) es lo que se ve cuando una cantidad que cambia
gradualmente cruza una línea de corte fija, no evidencia de una transición de fase genuina en el sistema
subyacente. La histéresis dio resultados mixtos y con una limitación honesta: la Parte A (réplica fiel
del motor, sin memoria) no mostró histéresis, como se esperaba; la Parte B (con la topología encadenada
entre puntos, agregada sólo para esta prueba) sí mostró diferencias grandes según la dirección del
barrido — pero es continuidad SÓLO de la topología, no del estado completo de los nodos, así que no es
una prueba cerrada de histéresis en el sentido fuerte.

---

## Parte 1 — Tamaño de grilla: 4×5 (20 celdas), y por qué no es más grande

`cs090_fase5_generador.py` ya tenía, congelados de antes, los rangos de muestreo de estos dos parámetros:
`RANGO_KCAP=(4,7)` y `RANGO_K=(4,8)`, **ambos enteros**. Eso significa que en el espacio que la auditoría
de C2 (`FASE5_auditoria_C2_resultado_CS.md`) puso a prueba **sólo existen 4 valores posibles de kcap
(4,5,6,7) y 5 de K (4,5,6,7,8)** — 20 combinaciones en total. Un pedido de "8-10 valores por eje" excede
ese espacio ya calibrado.

**Decisión:** cubrir la grilla completa de enteros (20 celdas, el máximo posible dentro del rango ya
auditado) en vez de extender kcap/K más allá de esos rangos hacia un régimen que la auditoría nunca puso
a prueba. Es la opción más honesta — territorio conocido completo, no territorio desconocido más ancho.
La razón de que la grilla final tenga 20 celdas y no 64-100 es **estructural** (el generador sólo conoce
esos 4×5 enteros), no de presupuesto de cómputo: el motor resultó rápido (Paso 1, medido con el reloj
real sobre 4 celdas de esquina × 5 semillas: 3.06-4.56s por regla, media 3.80s), lo cual dejó margen para
subir las semillas por celda de lo mínimo a **20 semillas/celda** (400 reglas en total, presupuesto usado
22.1 min de un tope de 62 min).

**Analogía simple:** es como querer fotografiar un paisaje con 10 tomas por lado, pero descubrir que el
terreno mapeado sólo tiene 4 colinas de un lado y 5 del otro — sacar más fotos no agrega terreno nuevo,
sólo repite el mismo terreno con más semillas (más veces tirando la moneda en cada colina). Se optó por
eso: más semillas en el terreno ya conocido, en vez de inventar colinas fuera del mapa calibrado.

---

## Parte 2 — La superficie P(Clase III | kcap, K)

Superficie medida (filas = kcap, columnas = K, 20 semillas por celda):

| kcap\K | K=4 | K=5 | K=6 | K=7 | K=8 |
|---|---|---|---|---|---|
| **kcap=4** | 0.85 | 0.80 | 0.90 | 0.80 | 0.90 |
| **kcap=5** | 0.60 | 0.65 | 0.80 | 0.75 | 0.70 |
| **kcap=6** | 0.05 | 0.05 | 0.05 | 0.05 | 0.10 |
| **kcap=7** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

- Salto máximo entre celdas vecinas: **0.75**, entre kcap=5 y kcap=6 a K=6 (media de todos los saltos
  vecinos en la grilla: 0.16 — este salto es ~4.7x el típico).
- Los tres saltos más grandes de toda la grilla están los tres en el mismo eje (kcap 5→6), a distintos K
  — **K casi no mueve la aguja**, el borde es esencialmente un fenómeno de kcap solo.
- Distribución global de clases sobre las 400 reglas: I=183 (46%), III=161 (40%), II=37 (9%),
  intermedio=12 (3%), IV=7 (2%).

### Los observables continuos que están DETRÁS de esa clasificación

Antes de clasificar, el script guardó los observables crudos por regla. Promediando por kcap (marginal
sobre K, n=100 por valor de kcap):

| kcap | pendiente_real (log-diám vs log-N) | giant_nativo (fracción en comp. gigante) | clustering_nativo | diám_nativo | n_aristas_nativo |
|---|---|---|---|---|---|
| 4 | 0.806 ± 0.612 | 0.871 ± 0.050 | 0.020 ± 0.015 | 17.8 ± 6.4 | 2424 ± 160 |
| 5 | 0.724 ± 0.242 | 0.954 ± 0.020 | 0.009 ± 0.008 | 14.2 ± 2.3 | 3267 ± 108 |
| 6 | 0.561 ± 0.215 | 0.980 ± 0.007 | 0.005 ± 0.004 | 11.3 ± 2.1 | 3987 ± 154 |
| 7 | 0.481 ± 0.078 | 0.991 ± 0.005 | 0.003 ± 0.002 | 10.0 ± 1.1 | 4592 ± 382 |

**Ninguno de estos cinco números da un salto entre kcap=5 y kcap=6.** Todos bajan o suben de forma
gradual y monótona en las cuatro columnas — la componente gigante crece de a poco (0.871→0.954→0.980→
0.991), el diámetro baja de a poco (17.8→14.2→11.3→10.0), el número de aristas sube de a poco
(2424→3267→3987→4592). No hay ningún indicio de divergencia, de meseta seguida de quiebre, ni de "algo
que colapsa de golpe" en estas cantidades físicas.

Lo único que SÍ tiene un umbral duro es la **regla de clasificación** (`cs090_fase5_clasificador.py`,
heredada tal cual, no tocada en esta tarea): Clase III requiere `pendiente_real > 0.7`. Mirando la
columna de pendiente arriba: la media pasa de 0.806 (kcap=4, sobre el corte) a 0.724 (kcap=5, todavía
sobre el corte pero con desvío 0.242 — mitad de las semillas ya caen debajo) a 0.561 (kcap=6, claramente
debajo). Es decir: **la media de la pendiente cruza el valor de corte 0.7 justo entre kcap=5 y kcap=6**,
y como la pendiente tiene ruido entre semillas, ese cruce de medias produce que la *fracción* de semillas
que caen del lado "Clase III" del corte se derrumbe en ese mismo tramo — sin que la pendiente misma dé
ningún salto, sólo se desliza.

**Analogía simple:** imaginate que subís gradualmente la dificultad de un examen y mirás qué porcentaje
de alumnos saca nota "aprobado" (un corte fijo, digamos 6/10). Si el puntaje promedio de la clase baja
suavemente con la dificultad, el % de aprobados puede desplomarse en un tramo angosto de dificultad —
justo cuando el promedio cruza el 6 — aunque ningún alumno individual haya tenido una caída brusca de
nota. Eso es lo que se ve acá: la "nota" (pendiente) baja suave; el "% aprobados" (P(Clase III)) tiene un
acantilado, porque el corte de aprobación es fijo y rígido, no porque el sistema se haya roto de golpe.

---

## Parte 3 — ¿Hay señales de transición de fase genuina cerca del borde?

Se separaron las celdas en "de borde" (0.2 ≤ P(III) ≤ 0.8, n=7 celdas: todo kcap=4-5 salvo dos esquinas)
y "lejos del borde" (n=13 celdas: kcap=4 y 6-7 en los extremos). Si hubiera una transición de fase real,
uno esperaría fluctuaciones AMPLIFICADAS cerca del borde (como la varianza que diverge cerca de un punto
crítico en física estadística).

- Desvío estándar de la pendiente entre semillas, **cerca del borde**: media = 0.327 (n=7 celdas)
- Desvío estándar de la pendiente entre semillas, **lejos del borde**: media = 0.234 (n=13 celdas)

Hay más varianza cerca del borde (+40%), lo cual **no contradice** una lectura de transición de fase —
pero es un efecto modesto, no una divergencia. Con sólo 20 semillas por celda y sin haber medido
correlación espacial entre celdas vecinas (lo que en física estadística sería el "largo de correlación"),
esto no alcanza para distinguir entre (a) fluctuaciones genuinamente amplificadas cerca de un punto
crítico, o (b) el efecto trivial de que, cerca de un corte de clasificación, CUALQUIER variable con ruido
normal va a producir más cambios de categoría por semilla que lejos del corte — que es exactamente lo que
predice la lectura de la Parte 2 (una distribución suave cruzando una línea fija).

**En síntesis de esta parte:** no hay evidencia, en esta grilla, de que la bimodalidad I↔III sea una
transición de fase en el sentido físico fuerte (divergencia de un observable continuo, fluctuaciones que
crecen sin límite cerca del borde). Lo que hay es un umbral de clasificación fijo que una cantidad
continua y suave (la pendiente log-diám-vs-log-N) cruza en un tramo angosto de kcap. Esto no cierra la
pregunta — sólo dice que, con los observables medidos acá, el candidato más simple ("hay un punto crítico
genuino") no tiene apoyo, y el candidato más simple alternativo ("es un artefacto del corte de
clasificación sobre una variable que se mueve suave") si lo tiene.

---

## Parte 4 — Test de histéresis a K=6 fijo (elegido por tener el rango más amplio de P(III) sobre kcap:
0.90, de 0.90 en kcap=4 a 0.00 en kcap=7), recorriendo kcap=[4,5,6,7] en las dos direcciones

Antes de correr esto se revisó el motor congelado (`cs090_fase5_motor.py`) para ver si tenía memoria de
estado entre puntos de parámetro: **no la tiene**. En `dinamica_B0`, rama A1/A2, cada llamada resamplea
el estado de cada nodo desde cero (`S = rng.uniform(0, K, N)`), sin leer nunca el estado previo de un
sustrato. Esto se documenta como limitación honesta del motor tal cual está, no se fabrica una
continuidad que no existe. Lo que SÍ es técnicamente posible sin tocar el motor es encadenar la
**topología** (`sustrato['adj']`) entre llamadas sucesivas, porque `_enforce_kcap` es una función
expuesta y componible. De ahí las dos partes:

### Parte A — independiente (réplica fiel de cómo corre el motor tal cual, SIN continuidad)

| kcap | alto→bajo | bajo→alto |
|---|---|---|
| 4 | III, III, III | III, III, III |
| 5 | III, I, III | III, I, III |
| 6 | I, I, I | I, I, I |
| 7 | I, I, I | I, I, I |

**Resultado: cero diferencia entre direcciones** (las clases son idénticas semilla por semilla en ambos
sentidos). Es el resultado esperado y sirve de control: confirma que, corriendo el motor tal cual está
hoy (sin ningún agregado de memoria), no hay ninguna dependencia de la historia — cada punto de kcap se
resuelve solo, sin importar de dónde "venías". Nótese también que a kcap=5 aparece la misma mezcla
III/I/III que ya se veía en la Parte 2 (zona de borde con ruido entre semillas).

### Parte B — con la topología encadenada (un solo sustrato de grafo que se reutiliza y se re-poda al
cambiar kcap, en vez de generar uno nuevo en cada punto)

| dirección | kcap | diám_nativo | giant_nativo | n_aristas |
|---|---|---|---|---|
| alto→bajo | 7 | 8.0 | 0.994 | 4812 |
| alto→bajo | 6 | 11.0 | 0.986 | 4184 |
| alto→bajo | 5 | 14.0 | 0.974 | 3560 |
| alto→bajo | 4 | 16.0 | 0.945 | 2859 |
| bajo→alto | 4 | 16.0 | 0.918 | 2583 |
| bajo→alto | 5 | 16.0 | 0.918 | 2548 |
| bajo→alto | 6 | 18.0 | 0.914 | 2499 |
| bajo→alto | 7 | 20.0 | 0.912 | 2474 |

Acá **sí aparecen diferencias grandes según la dirección**, en el mismo kcap: en kcap=7, `n_aristas` va
de 4812 (llegando desde arriba, kcap ya venía siendo grande) a 2474 (llegando desde abajo, kcap venía
siendo chico) — casi la mitad. El diámetro y la fracción en la componente gigante también difieren de
forma consistente: la rama alto→bajo llega a cada kcap con MÁS aristas y MENOS diámetro que la rama
bajo→alto.

**Por qué esto NO es una prueba cerrada de histéresis (tal como quedó anotado en el log de la corrida):**
esta Parte B encadena sólo la ADYACENCIA del grafo (qué nodo está conectado con cuál), no el estado
interno de fase de cada nodo — eso es "continuidad topológica", no "continuidad de estado completa". Y el
propio mecanismo de `_enforce_kcap` es asimétrico por construcción: cuando kcap baja, corta aristas de
forma inmediata y esas aristas desaparecen del sustrato encadenado; cuando kcap sube después, no hay
ningún paso que las reconstruya automáticamente — sólo se pueden agregar aristas nuevas si la dinámica de
construcción vuelve a proponerlas. Entonces parte de la diferencia que se ve en la tabla puede ser
simplemente eso: un candado de un solo sentido en la poda (fácil sacar aristas, no automático
recuperarlas), no necesariamente una firma profunda de "memoria física" del sistema en el sentido que
tendría una transición de fase con histéresis genuina (donde el estado completo, no sólo la topología,
carga la historia).

**Analogía simple:** es como podar un arbusto. Si vas cortando ramas de a poco (alto→bajo), terminás con
un arbusto más chico y compacto. Si en cambio empezás con un arbusto ya podado y le das tiempo a crecer
de nuevo antes de decidir si podarlo más (bajo→alto), las ramas nuevas no salen exactamente en los mismos
lugares que las que cortaste antes — el arbusto "recuerda" el corte (no vuelve a las ramas viejas
mágicamente), pero eso es memoria de la FORMA del arbusto (topología), no memoria de cuánta savia tenía
cada rama (estado). La Parte B mide lo primero; la pregunta de histéresis "fuerte" pediría lo segundo, y
el motor congelado hoy no tiene ese segundo tipo de memoria.

---

## Qué queda para Alexis (no es cierre, son preguntas abiertas)

- La lectura más simple de la Parte 2/3 es que el borde I↔III es un efecto de umbral de clasificación
  sobre una variable continua y suave, no una transición de fase con divergencia propia. Si se quisiera
  poner a prueba más duro, habría que medir un análogo de "largo de correlación" (¿la clase de una celda
  predice la de sus vecinas más allá de lo que predice la tendencia suave de kcap?) — no se hizo acá.
- La Parte B de histéresis sugiere memoria de la topología, pero está confundida con la asimetría propia
  del mecanismo de poda (fácil cortar, no automático reconectar). Separar esas dos cosas pediría un
  diseño donde la poda sea reversible por construcción, o agregar de verdad memoria de estado al motor —
  ninguna de las dos se tocó en esta tarea (el motor congelado no se editó).

## Archivos de esta tarea

- `cs090_fase5_mapa_transicion.py` — script nuevo (único archivo de código de esta tarea; no modifica
  ningún script congelado).
- `cs090_fase5_mapa_transicion_costo.csv` — Paso 1, medición de costo (20 filas).
- `cs090_fase5_mapa_transicion_grid.csv` — Paso 2, grilla completa 4×5×20 semillas (400 filas).
- `cs090_fase5_mapa_transicion_histeresis.csv` — Paso 4, test de histéresis Partes A y B (33 filas).
- Este informe.

No se corrió Phantom. No se declara cierre ni veredicto sobre si hay "una transición de fase real" ni
sobre A2-B0-C2 como candidato. No se hicieron commits de git.
