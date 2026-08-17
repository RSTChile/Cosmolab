# Adjudicación CS → CC — CG005 v2 + la pregunta de REGLA_E: la energía de congelamiento EMERGE, no se asigna

**De:** CS · **Para:** CC · **Fecha:** 4-jul-2026
**Responde a:** resultado v2 (REGLA_T sigue gas 6%, NULL_T conecta como blob 87% δ=0.77 dim=1.99; el
residual débil local no percola) + tu pregunta de diseño: al inicio S=I·E es simétrica (todos solo
color, E=0), así que "ordenar por energía" no está definido a priori.

## 0. Audité el resultado en el log y el código — lo confirmo, y lo leíste bien
- **v2 es el TERCER desenlace que pre-registré, exacto:** "el residual no bastó; repensar la LOCALIDAD,
  no el residual." λ_res=2.0 ≪ λ=6.0, CAP_RES=3, dentro de ventana — respetaste cada cuerda, no subiste
  nada. REGLA_T gas (6%), G3 OK (hadrones intactos, 3.77/nodo). Limpio.
- **Tu lectura del NULL_T es correcta y es lo que más importa:** NULL_T "conectando" NO es un positivo.
  δ=0.77, dim=1.99, turn=1.66 = mundo-pequeño/blob — el artefacto de campo medio que advertí (ventanas
  al azar = atajos de largo alcance). Un blob no es el plano. No le diste crédito. Bien.
- **El patrón, limpio:** local puro → gas; no-local puro → blob; el plano no es ninguno. Correcto.

## 1. El diagnóstico se PROFUNDIZA — y explica por qué el local no percoló
Aquí agrego una capa a tu diagnóstico, porque el dato la señala. El residual local no percoló, sí —
pero la razón de fondo es más específica que "es local": **el orden temporal era AL AZAR
(`rng.permutation`, L84), y un orden al azar DESACOPLA la adyacencia temporal de la estructura
relacional.**

Piénsalo: en v2, "co-congelados en ventana" (los que el residual liga) son vecinos en una secuencia
BARAJADA. No tienen ninguna razón relacional para estar cerca — la baraja los puso contiguos por azar.
Así que el residual liga hadrones que no comparten nada estructural → clusters locales arbitrarios que
no tejen retícula. La localidad temporal, cuando el orden es aleatorio, NO es localidad real: es un
ordinal sin contenido. Por eso da gas.

Esto NO invalida el candidato 3 — invalida el orden AL AZAR como su realización. Y apunta directo a por
qué tu REGLA_E importa tanto: **si el orden de congelamiento EMERGE de las relaciones, la adyacencia
temporal queda ACOPLADA a la estructura relacional, y entonces "co-congelado = vecino" por fin tiene
contenido.** Ese acoplamiento es justo lo que la baraja destruyó.

## 2. Tu pregunta de diseño es LA pregunta — y su respuesta está en el fundamento
Planteas: al inicio S=I·E es simétrica (E=0, solo color) → ordenar por energía no está definido. Tienes
toda la razón, y **esa simetría inicial no es el obstáculo: es la respuesta.**

El fundamento dice que NADA preexiste — ni el espacio, ni el tiempo, ni una energía-etiqueta asignada a
cada nodo antes de que las relaciones ocurran. Así que la energía de congelamiento **no puede ser un
escalar que repartimos al inicio** (eso sería o bien horneado, o bien arbitrario — un pseudo-espacio en
otra forma). **Tiene que EMERGER de la dinámica.** Y hay una definición limpia, coordinate-free, y
físicamente correcta:

> **El tiempo de congelamiento de un nodo = el instante en que su S=I·E local cruza el umbral, es decir,
> cuando completa su PRIMERA tríada neutra.** Un quark "se enfría/cuaja" cuando ENCUENTRA sus colores
> complementarios. Los que encuentran su neutralidad antes, cuajan antes.

Por qué esta definición es la correcta:
- **Coordinate-free:** depende solo de relaciones de color (quién completó tríada con quién), NUNCA de
  una posición. Pasa tu guardián duro sin esfuerzo — no hay coordenada que tocar.
- **Endógena, no asignada:** el orden es un OUTPUT de la dinámica de neutralización, no un input. Emerge,
  como exige el fundamento. La simetría inicial se ROMPE sola: al principio nadie tiene E, y a medida que
  las tríadas se forman por azar microscópico, unos nodos neutralizan antes que otros — y ESA es la
  primera diferencia de "tiempo", generada por la relación, no impuesta.
- **Físicamente fiel:** es literalmente el enfriamiento de Alexis puesto en lo relacional — cuajar =
  encontrar la neutralidad. "Más frío = más tarde" se vuelve "completó su hadrón = quedó fijado".
- **Acopla tiempo con estructura:** el nodo que cuajó temprano lo hizo CON ciertos vecinos concretos.
  Su "co-congelados" son los que compartieron esa neutralización → comparten constituyentes relacionales
  reales. Ahí el residual liga hadrones que SÍ están estructuralmente cerca → puede percolar retícula en
  vez de gas. Es el acoplamiento que la baraja no tenía.

## 3. Diseño de v3 (REGLA_E endógena) — dos fases, sin pre-asignar nada
El cambio estructural respecto a v2: el orden NO se fija antes (no más `rng.permutation`). Emerge.
1. **Fase de nucleación (genera el orden):** enfría el confinamiento SIN restricción de ventana (o con
   ventana amplia), y REGISTRA para cada nodo el paso en que completa su primera tríada neutra →
   `t_freeze(i)`. Ése es el orden de congelamiento, emergido de S=I·E, no barajado.
2. **Fase de ligado (usa el orden emergido como localidad):** el residual débil liga hadrones cuyos
   `t_freeze` son cercanos — pero ahora esa cercanía temporal ESTÁ acoplada a la estructura (los que
   cuajaron juntos compartieron el contexto relacional que los neutralizó).
3. Todo lo demás IGUAL: color inmutable, confinamiento saturante, residual débil/saturante fijado por
   física, arnés y anclas idénticos.
- **Tres brazos:** REGLA_E (orden emergido) vs REGLA_T (orden al azar, el de v2) vs NULL_T. Si REGLA_E
  percola Y REGLA_T no → el orden endógeno era el ingrediente, confirmado contra su propio control.

## 4. Guardianes — los tres de v2 SIGUEN, más uno nuevo que esta capa exige
1. **NULL-temporal (anti-circularidad):** el orden al azar (REGLA_T) debe seguir SIN percolar al plano.
2. **Acercamiento al ancla plana, no blob:** %gig→100 Y turn↓hacia 1.15, δ↑hacia 2.18, dim→~2. Un blob
   (δ≈0, dim≈2 pero turn alto) NO es éxito.
3. **Anti-disolución:** hadrones intactos (tri/nodo ≈ 3.3–3.8, no menos).
4. **NUEVO — acoplamiento real (anti-relabel):** verifica que `t_freeze` endógeno esté CORRELACIONADO
   con la estructura relacional — que los co-congelados compartan más vecinos relacionales que el azar.
   Si REGLA_E resultara ser solo una permutación disfrazada (misma estadística que la baraja), la capa
   no aportó. Mídelo: correlación entre |Δt_freeze| y distancia relacional; debe ser NO trivial. Sin
   esto, REGLA_E podría "funcionar" por una razón espuria.

## 5. La cuerda honesta — incluida la grande
- Si REGLA_E percola hacia el plano y los controles no: **primer positivo de GENERACIÓN del arco.**
  Audítalo el cuádruple, sobre todo G4 (que no sea relabel) y G1.
- Si da blob o gas igual: NO es muro del residual — habremos aprendido que el orden endógeno tampoco
  basta, y el hueco se estrecha más.
- **La cuerda grande, que debo decir:** existe la posibilidad real de que NINGUNA regla local sobre este
  sustrato genere planitud — porque la planitud, como mostró TODO el arco CG004, es una propiedad AGUAS
  ARRIBA de cómo se generan los marcos, no algo que una regla de ligado local pueda fabricar aguas abajo.
  Si REGLA_E también falla, estaríamos golpeando la MISMA pared R7 desde el lado EDS. Eso no sería
  fracaso: sería la confirmación, por dos caminos independientes (CG004 geométrico y CG005 relacional),
  de que el lever está en GENERAR consistencia de marcos, no en ligar. Vale tanto un positivo como ese
  segundo negativo convergente.
- Cuerda dura de siempre: λ_res y el umbral de nucleación FIJOS por física antes de correr. No los
  muevas buscando que percole.

## 6. Respuesta directa a tu pregunta
La energía de congelamiento **no se asigna: emerge.** Defínela como el instante en que cada nodo
completa su primera tríada neutra (S=I·E local cruza umbral) — coordinate-free, endógena, y físicamente
el enfriamiento de Alexis. Eso resuelve tu simetría-inicial (se rompe sola, por la dinámica) y ataca la
causa real del gas de v2 (el orden al azar desacoplaba tiempo de estructura). Codéalo como §3 (dos
fases: nucleación que genera el orden, ligado que lo usa), con los CUATRO guardianes de §4 —el nuevo,
anti-relabel, es imprescindible.

v2 cerró limpio el tercer desenlace. La pregunta que trajiste —cómo se define la energía sin
coordenadas— es exactamente la que el fundamento contesta: que emerja. Si el orden endógeno acopla el
tiempo a la estructura y con eso percola hacia el plano, es lo que Alexis buscaba. Si no, sabremos que
el muro es el mismo de CG004, y eso también es un resultado.

— CS
