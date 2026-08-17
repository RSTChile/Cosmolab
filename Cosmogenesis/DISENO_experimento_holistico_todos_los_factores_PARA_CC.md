# El experimento holístico — todos los factores juntos, en un solo proceso
### La energía (S=I·E) integrada como capa transversal al motor · diseño para CC · leer entero

**Nota de numeración (corregida):** las piezas canónicas del motor van del **1 al 23** (el
23 son las fluctuaciones cuánticas; el inventario está cerrado). El motor además tiene el
**tiempo como pieza #24** (`p24_tiempo.py`) y varias piezas auxiliares (expansión,
enfriamiento, materia oscura). **La energía NO es un factor numerado nuevo** — es una capa
de contabilidad que atraviesa TODAS las piezas (la mitad "E" de la ley S=I·E que hasta
ahora no estaba en el sustrato). No ocupa un casillero; envuelve a todos.

**Director:** Alexis López Tapia · **Diseño:** Claude Science (CS) · **Fecha:** 25-jul-2026
**Base real (verificada en disco):** motor `cs072_modulos/` (Estado + 23 piezas + proceso).
**Lo que aporta el Enfoque 5:** las reglas con las que la energía se porta bien (abajo).

---

## 0. QUÉ ES ESTO Y QUÉ NO ES (para no repetir el error)

**El error que NO se repite:** correr la energía sola y pedirle el 5%. La energía sola no
puede dar el reparto de materia — es una capa, no el todo. **Esto es un PROCESO
HOLÍSTICO:** todos los factores del motor (las 23 piezas canónicas + el tiempo + las
auxiliares) MÁS la energía como capa transversal (la mitad "E" que faltaba de la ley
central S = I·E), **todos actuando juntos, en una sola corrida.**

**Lo que este experimento pregunta, de verdad:** cuando la energía-exergía-entropía corre
JUNTO con las 23 piezas —no antes, no después, no aparte— ¿qué emerge del conjunto que no
emergía de las piezas sin energía? El 5%/31,5% de materia, si aparece, aparecería AQUÍ —
del todo, no de la energía.

---

## 1. LO QUE YA ESTÁ (no se reconstruye)

El motor `cs072_modulos/` ya tiene la arquitectura correcta:
- **Un `Estado` compartido** — todas las piezas leen y escriben el mismo estado (holístico
  por diseño: nadie actúa aislado).
- **23 piezas** que actúan por época según la temperatura (`T_umbral`): gravedad, fuerza
  fuerte, EM, aniquilación, fluctuaciones, expansión, enfriamiento, materia oscura, etc.
- **Regla de admisibilidad anti-Shannon ya incorporada:** apagar una pieza DEBE cambiar su
  observable; si no, la pieza está declarada pero no actúa. El núcleo lo verifica.
- **El enfriamiento y la expansión ya están** como piezas — que es justo donde la energía
  engancha.

**No hay que rehacer el motor. Hay que agregarle la contabilidad de energía que lo
atraviesa.**

## 2. CÓMO ENTRA LA ENERGÍA — las reglas que aprendimos en el Enfoque 5

Esto es para lo que sirvieron los 30 experimentos: **saber cómo la energía se porta bien.**
Se agrega al `Estado` una capa de energía con estas cuatro reglas, todas ya probadas:

- **REGLA 1 — Presupuesto cerrado (conservación exacta):** el Estado arranca con un
  presupuesto total de energía E_total, y **se conserva exacto en cada paso** — ninguna
  pieza puede crear ni destruir energía, solo transformarla o moverla. (Del arreglo 1 y
  E5.2-2: la contabilidad cierra si se hace así.)
- **REGLA 2 — Exergía = energía útil, ligada a la diferencia:** cada configuración del
  Estado tiene una exergía X (la parte de E capaz de hacer trabajo), que **depende de las
  diferencias presentes** — sin diferencias, X=0 aunque E sea máxima. (De E5.5-4: muerte
  térmica = E máx, X=0.)
- **REGLA 3 — La expansión convierte E latente en X (no la crea):** la pieza de expansión,
  al aislar regiones antes de que se mezclen, **rescata exergía** — es el único mecanismo
  que lo hace (ni reordenar ni re-inyectar; de E5.5-3). El enfriamiento adiabático ES esa
  conversión medida.
- **REGLA 4 — Cada pieza que forma estructura PAGA su costo del presupuesto:** cuando la
  fuerza fuerte liga un trío, cuando la gravedad junta átomos, cuando la EM captura un
  electrón — **esa ligadura tiene un costo energético que sale de E_total.** Aquí está lo
  nuevo y lo potente: la energía ligada en estructura es medible, y **la masa es esa
  energía de ligadura** (lo que la literatura del Modelo Estándar confirma: el 99% de la
  masa del protón es energía de la fuerza fuerte).

## 3. EL OBSERVABLE HOLÍSTICO (lo que se mide del conjunto)

No se mide "la energía" ni "las fuerzas" por separado. Se mide **del proceso completo:**

- **Balance de energía a lo largo del proceso:** cómo E_total se reparte, paso a paso,
  entre {exergía libre, energía degradada/desorden, energía ligada en estructura}.
- **La fracción que termina ligada en estructura estable** = candidata a "materia". Esta
  es SALIDA emergente del conjunto — la eficiencia de conversión del proceso entero.
- **Comparación con el reparto real (4,9% / 31,5%)** — SOLO como test de salida, JAMÁS
  como entrada. Si el proceso holístico escupe algo cercano SIN que lo ajustemos → es el
  hallazgo que buscábamos. Si escupe otra cosa → dato honesto del modelo.

## 4. LAS TRAMPAS (las de siempre + la lección de hoy)

- **T-holística (la nueva, la de hoy):** NO pedirle a una pieza lo que solo da el todo. El
  observable de materia se lee del PROCESO COMPLETO con todas las piezas + la energía
  actuando, nunca de la energía sola ni de las fuerzas solas.
- **T1** ningún número a mano — solo ε, las palancas físicas, y E_total como presupuesto.
- **T-conservación** — la energía se conserva exacto cada paso; si el balance no cuadra, el
  experimento FALLA (es el chequeo duro, imposible de trampear).
- **T-target** — el 4,9%/31,5% es test de salida, nunca entrada. Ajustar para acercarse =
  el 20.0 = anulado.
- **T-admisibilidad** (ya en el motor) — apagar cualquier factor, incluida la energía, DEBE
  cambiar su observable. Si apagar la energía no cambia nada, la energía no está actuando.

## 5. LA PRUEBA DE QUE LA ENERGÍA DE VERDAD ACTÚA (admisibilidad de la capa de energía)

Igual que el motor verifica cada pieza apagándola, hay que verificar la energía:
**correr el proceso completo CON energía y SIN energía (presupuesto infinito, sin costo de
ligadura) y mostrar que el resultado cambia.** Si la materia emergente es la misma con y
sin la contabilidad de energía, entonces la energía está de adorno (Shannon). Si cambia
—si el presupuesto cerrado limita qué estructura puede formarse— entonces la energía es un
factor real del proceso. **Esta es la prueba de que integramos la energía de verdad, no de
mentira.**

## 6. EL BARRIDO (holístico, sobredimensionado)

Un solo proceso, barriendo:
- **ε** (la asimetría fundacional) ∈ rango amplio.
- **E_total** (el presupuesto de energía) ∈ rango amplio — desde escaso hasta abundante.
- Las palancas físicas que ya barre el motor (tasas relativas de las fuerzas, expansión).
- **Todo junto, en una sola corrida por punto** — no las piezas por separado.

Y se lee, para cada punto: qué fracción del presupuesto terminó como materia, como exergía
libre, como desorden — la fotografía energética completa del universo-modelo.

## 7. QUÉ ENTREGAR A CS

- Pre-registro fechado (observable holístico, el test de admisibilidad de la energía, el
  criterio de comparación con 4,9%/31,5% como salida, rangos, semillas).
- El motor con la capa de energía integrada al `Estado` (no un motor aparte).
- La corrida CON y SIN energía (la prueba de admisibilidad de la capa de energía).
- El balance energético completo del proceso, crudo, sin adjudicar.
- **NO adjudicar** — CS lee qué emergió del conjunto.

---

**En una frase:** ya no le pedimos el 5% a la energía sola — integramos la energía como
capa de contabilidad que atraviesa todas las piezas del motor, con las reglas que
aprendimos a los golpes en el Enfoque 5, y dejamos correr el proceso completo, holístico,
para ver qué reparto de materia emerge del
TODO. Si aparece algo cercano al universo real sin ponerlo, ese sí sería el hallazgo. Y si
no, habremos probado honestamente hasta dónde llega el modelo con todas sus piezas juntas —
que es lo único que nunca habíamos hecho.
