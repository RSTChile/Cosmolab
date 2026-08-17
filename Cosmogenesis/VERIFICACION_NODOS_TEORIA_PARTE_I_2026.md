# Verificación experimental de la Parte I de la Teoría Cosmosemiótica (Cosmogénesis y Cosmología)

**Para:** Alexis López Tapia · **Base:** Teoria_Cosmosemiotica_Canónica_17-07-2026.pdf (Bloques 1 a 5, Parte I) contrastados contra la cronología completa de experimentos de Cosmogénesis · **Fecha:** 3-ago-2026

---

## Cómo se hizo esta verificación

Se tomó cada nodo de la Parte I de la Teoría (Bloques 1, 2, 2.5, 2.6, 2.7, 2.8, 3, 4 y 5 — desde "Persistencia Pre-Estructural" hasta "Dinámica", tal como pediste) y se lo contrastó, uno por uno, contra el registro completo de experimentos de Cosmogénesis (el documento de auditoría anterior). Para cada nodo se pregunta: **¿hay un experimento concreto que lo haya puesto a prueba? ¿Con qué resultado? ¿O sigue siendo, por ahora, un enunciado teórico sin experimento propio?**

**Una aclaración importante antes de empezar, sobre qué significa "verificar" acá:** algunos nodos de la Teoría —empezando por el primero, S>0— están explícitamente declarados en el propio documento canónico como **no empíricos sino trascendentales**: no describen un estado del mundo que se pueda medir, sino la condición de posibilidad para que haya mundo alguno. Esos nodos no "se prueban" en el sentido de un experimento de laboratorio — se los pone a trabajar, y se mira si lo que se construye sobre ellos sostiene o se cae. Esto se marca claramente cada vez que aparece.

**Los símbolos que vas a ver:**
- ✅ **Confirmado** — hay un experimento concreto de Cosmogénesis, con su control de azar, que sostiene lo que el nodo afirma.
- ⊘ **Parcial / indirecto** — el nodo está operacionalizado en el motor y es consistente con lo encontrado, pero no hay un experimento *dedicado* que lo haya aislado y puesto a prueba por sí solo.
- ❌ **Puesto a prueba y no sostenido** — se probó específicamente la hipótesis del nodo y el resultado fue negativo.
- ⚠️ **Confirmado, pero con una salvedad grave** — el experimento que se citaba como evidencia resultó, en auditoría posterior, parcialmente fabricado (aplica sobre todo a resultados que dependían del motor CS072 caído).
- ⏳ **Sin experimento propio en Cosmogénesis** — el nodo es válido dentro de la Teoría, pero nada en el código de Cosmogénesis lo ha puesto a prueba todavía (puede estar operacionalizado en otro proyecto de la Teoría, como la Célula Madre, pero eso queda fuera de esta revisión).
- 🔷 **Es una aclaración de alcance, no una afirmación empírica** — el nodo mismo dice "esto no pretende ser una ley física", así que no corresponde pedirle un experimento.

Un hallazgo que vas a ver repetirse: **varios nodos del Bloque 2.7 no son hipótesis a la espera de comprobación — son, literalmente, el resumen que la propia Teoría ya hizo de los resultados de Cosmogénesis.** El documento canónico, fechado 17-jul-2026, ya incorporó como texto los hallazgos de los experimentos CS066 a CS069. Eso se señala explícitamente donde corresponde.

---

## BLOQUE 1 — Persistencia Pre-Estructural

### C-N1 — S > 0 (condición de existencia)
*"Algo persiste lo suficiente para no anularse completamente."*

🔷 **No es un nodo para probar — es el punto de partida de todo el proyecto.** El propio documento lo dice: "no es empírica sino trascendental". Cosmogénesis no puede "confirmar" que algo deba persistir para que haya universo — lo toma como semilla y mira qué se puede construir encima. Dicho eso, hay algo indirecto que sí vale la pena anotar: **cada experimento que produjo estructura real (bariogénesis en CS008, hidrógeno y helio dependientes de fuerzas reales en CS072, la coherencia relacional del "puente" en CS073) es una demostración práctica de que arrancar de una diferencia mínima y dejarla evolucionar SÍ puede producir algo — lo cual es consistente con la apuesta del nodo, aunque no la "pruebe" en sentido estricto.**

### C-N1.1 — S ≤ 0 ⇒ ∅
*"Lo que no deja huella no se puede diferenciar ni estructurar."*

⏳ Es el caso complementario de C-N1, una definición, no algo que se someta a prueba por separado.

### C-N1.2 — S(t+Δt) > 0 ⇒ persistencia
*"La persistencia se confirma cuando algo sigue existiendo en el instante siguiente."*

✅ **Confirmado, y de forma muy repetida.** Este es, literalmente, lo que mide CS007 ("¿las diferencias persisten en el tiempo?" — sí) y lo que verifica, paso a paso, cada experimento del proyecto que compara un estado en t contra el mismo estado en t+1 (desde el campo básico de CG001 hasta el hidrógeno de CS072 que no se disuelve, hasta los sumideros de CS073 que siguen creciendo). No es un experimento único — es la condición mínima que TODOS los experimentos exitosos del proyecto satisfacen para poder decir que encontraron algo.

*Experimento clave: CS007 (persistencia). Script: cg001_ipad_persistencia.py*

### C-N1.3 — Ω_persistente ⊆ Ω_posible
*"No todo lo posible persiste — la persistencia filtra el espacio ontológico."*

⊘ **Es la lógica de fondo de todo el "arco de eliminación" (CS052–CS071), pero un experimento que lo probó de forma aislada y literal dio un resultado débil.** CS053 construyó exactamente esto — un filtro que deja pasar sólo las geometrías que "aguantan" con el tiempo — y el resultado fue que **el filtro es demasiado permisivo**: sobrevive cualquier retículo de dimensión 2 o más, por igual. El principio ("la persistencia filtra") es verdadero y se usa en todo el proyecto, pero medido de forma aislada no alcanza para elegir nada específico — hace falta combinarlo con otras condiciones (que es justo lo que el resto del arco de eliminación hizo).

*Experimento clave: CS053 (filtro de persistencia, ❌ demasiado permisivo por sí solo). Script: cs053_persistencia_geometria.py*

---

## BLOQUE 2 — Acoplamiento Originario

### C-N2 — S = I ⟺ E
*"La persistencia se sostiene mediante interacción entre interior y exterior."*

✅ **Confirmado extensamente — es la variable central de la mitad del proyecto.** Esta ecuación es, en la práctica, el llamado `A_sys-env` (acoplamiento sistema-entorno) que atraviesa CG002 completo: la criticidad (CS012, transición de fase real), los gradientes de estabilidad, y decenas de experimentos posteriores. No es un hallazgo de un solo experimento — es el marco de trabajo que decenas de experimentos usaron y que, en conjunto, sostuvieron (con las excepciones puntuales ya documentadas, como CS016, paredes de dominio, que no salió).

*Experimentos clave: CS008–CS025 (arco fundacional de CG002), CS012 (criticidad). Script: cg002_experimentos_arco.py*

### C-N2.1 — S > 0 ⟺ I > 0 ∧ E > 0
*"Sin interior y exterior simultáneamente activos, nada persiste relacionalmente."*

⊘ Condición lógica incorporada en el diseño de todos los experimentos de acoplamiento (nunca se construyó un experimento con "interior sin exterior"), pero no hay un experimento dedicado a poner a prueba específicamente esta condición como hipótesis aislada.

### C-N2.2 — E = 0 ∨ I = 0 ⇒ S = 0
*"Si uno de los dos desaparece, la relación —y la persistencia— colapsan."*

✅ **Confirmado, de la forma más directa posible: apagando la pieza y viendo que el resultado desaparece.** Esto es exactamente la regla anti-Shannon de todo el proyecto, y se probó una y otra vez: apagar el electromagnetismo hace que el hidrógeno colapse a cero (CS072, dependencia genuina, sobrevivió la auditoría); apagar la fuerza fuerte hace que el helio colapse a cero (CS072, ídem); apagar la gravedad en el control positivo de CS073 hace que la nube no se ligue en absoluto (E=+0.00, no liga). Es uno de los nodos con más apoyo experimental directo de toda la Parte I.

*Experimentos clave: CS072 (apagar EM→H=0, apagar fuerte→He=0, ambos verificados en auditoría), CS073 (control positivo sin gravedad, no liga). Scripts: cs072_modulos/p04_em.py, p03_fuerte.py; cs073_cierre_holistico.py*

---

## BLOQUE 2.5 — Emergencia del Tiempo

### C-N2.5 — t ≡ orden inducido por Δ_struct
*"El tiempo no preexiste: emerge cuando hay diferencias estructurales acumuladas."*

⊘ **Implementado tal cual en CS072, pero sin una auditoría independiente dedicada como la que sí recibieron bariones y dimensión.** El motor de CS072 cuenta el tiempo contando transiciones irreversibles (cada átomo neutro formado es un "tic" del reloj), no pasos de simulación — es una implementación literal de este nodo. A diferencia del conteo de bariones y la dimensión, este mecanismo específico no aparece señalado como fabricado en la auditoría de parámetros a mano — pero tampoco fue sometido a la misma prueba de apagado/control que sí recibieron otras piezas del motor. Queda como "implementado y consistente", no como "probado a fondo".

*Experimento: CS072 (tiempo emergente = 75, H+He). Script: cs072_modulos/piezas/p24_tiempo.py*

### C-N2.5.1 a C-N2.5.4 — relación de orden, secuencia de constricciones, Δ_struct=0⇒t=∅, condición de posibilidad de Ω(t)
⊘ Consecuencias lógicas del nodo anterior, consistentes con el diseño de CS072 y con la flecha del tiempo de CS009, pero no aisladas como hipótesis propias en ningún experimento.

### C-N2.5.5 — Asimetría primordial como evidencia empírica de S > 0 pre-temporal
*"Las violaciones CP observadas en física de partículas constituyen evidencia potencial de una asimetría ontológica mínima."*

⊘ **Resultado mixto, y hay que separar dos preguntas que el nodo mezcla.** Cosmogénesis probó dos versiones relacionadas de esta idea, con resultados opuestos:
- **CF-1** (¿una diferencia mínima sembrada persiste si el universo se expande más rápido de lo que se difumina?) — ✅ **sí**, con mecanismo claro y z entre 4,9 y 7,8. Esto apoya la parte del nodo que dice que una asimetría mínima, dadas las condiciones correctas, puede persistir.
- **CS070** (¿esa asimetría mínima tipo CP se amplifica hasta producir una dirección estable?) — ❌ **no**, la dirección midió 0.000 en las 96 corridas — la semilla "se lava" igual que la sopa simétrica.

En criollo: **que una asimetría mínima persista (si se dan las condiciones) sí se sostiene; que esa asimetría, por sí sola, alcance para producir una dirección estable, no.** El nodo, tal como está escrito, sólo afirma la primera parte (que la violación CP es evidencia de una asimetría mínima que existió) — en ese sentido más acotado, sí tiene apoyo.

*Experimentos: CF-1 (✅, persistencia bajo expansión), CS070 (❌, la semilla no produce dirección). Scripts: cs074_rcruz.py, cs070_*.py*

### C-N2.5.6 a C-N2.5.10 — dirección de la flecha temporal (T⁺, T⁻), inercia histórica, cancelación de orientaciones incompatibles
*Aclaración importante para no confundir dos cosas parecidas:* estos nodos hablan de la **dirección del tiempo** (si el tiempo tiene un "hacia adelante" privilegiado, T⁺ vs T⁻) — es un concepto distinto de la **dirección espacial** (los ejes/orientaciones del espacio) que sí se probó extensamente en CS064–CS070. No hay que confundirlos aunque usen palabras parecidas.

⏳ **Sin experimento propio en Cosmogénesis que aísle específicamente la dirección temporal (T⁺/T⁻) de la dirección espacial.** Lo más cercano es la nota interpretativa del propio nodo C-N2.5.10, que sugiere que la aniquilación materia-antimateria podría leerse como "cancelación de orientaciones históricas incompatibles" — pero el mecanismo de aniquilación de CS072, al ser auditado, mostró que apagarlo **no tiene efecto** sobre el resultado final ("la afirmación 'apagar aniquilación → 10 en vez de 3' también es falsa en este motor") — así que, si algo, la auditoría le resta apoyo a esta interpretación específica, más que dárselo. Queda como un nodo teórico sin sostén experimental directo todavía.

*Nota sobre CS072: ver auditoría en la Parte 2 del informe anterior.*

---

## BLOQUE 2.6 — Constricción Estructural

### C-N2.6 — ∂S ≠ 0 ⇒ curvatura del espacio de estados
*"Donde hay diferencia estructural, el espacio de estados se curva: hay zonas más probables."*

✅ **Confirmado — es literalmente lo que midió el experimento más grande de todo el proyecto.** CS057 (el paisaje completo, 69.648 universos) construyó exactamente un mapa de "qué tan probable/viable es cada combinación de fuerzas", y encontró que el punto correspondiente a nuestras fuerzas físicas reales cae en una zona con 4 veces más probabilidad que el fondo al azar — es decir, el espacio de posibilidades SÍ está curvado (no es plano/uniforme), con zonas de mayor y menor viabilidad. CS062 repitió esto con una gravedad más realista y confirmó el mismo patrón general.

*Experimentos: CS057 (paisaje completo), CS062 (paisaje corregido). Scripts: cs057_paisaje_completo.py, cs062 (paisaje con peso intrínseco)*

### C-N2.6.1 a C-N2.6.4 — gradientes de estabilidad, trayectorias siguen el gradiente, mínimos locales como atractores, acumulación de constricción ⇒ organización a gran escala
⊘ **Consistentes con múltiples hallazgos, sin un experimento que aísle cada uno por separado.** La idea de que los sistemas "caen" hacia configuraciones estables y que eso acumulado produce organización a gran escala es coherente con: CS055 (dos fuerzas en pulseada buscando un equilibrio), el hallazgo de cs074-A (la asimetría inicial tiene una "meseta" estable en un rango, y colapsa fuera de él — un mínimo local real, medido), y la observación general de Enfoque 5 de que las estructuras grandes "se congelan primero" y retienen la mayoría de la energía útil disponible. Ninguno de estos experimentos se diseñó para probar *este nodo específico*, pero todos son compatibles con él.

*Experimentos relacionados: cs074A_asimetria_techo.py (meseta estable), BATERIA_ENFOQUE5 (congelamiento jerárquico)*

---

## BLOQUE 2.7 — Regímenes Fundamentales de Acoplamiento

*Este es el bloque con más apoyo experimental directo de toda la Parte I — y también donde hay que ser más cuidadoso, porque varios de sus nodos citan a los propios experimentos de Cosmogénesis por nombre dentro del texto de la Teoría.*

### C-N2.7 — I ⟺ E se manifiesta en regímenes discretos
*"Las cuatro fuerzas fundamentales son regímenes del mismo principio de acoplamiento."*

⊘ **Es la arquitectura completa de CG002 (serie r7) y CS056 (las cuatro fuerzas reales), pero la unificación profunda entre regímenes nunca se puso a prueba contra una alternativa — se construyó así desde el diseño.** Que color, carga, fuerza débil y gravedad puedan implementarse como "variantes de la misma regla de acoplamiento, con distinto alcance e intensidad" SÍ se construyó y corrió (con resultados mixtos por régimen, ver abajo) — pero no hay un experimento que haya comparado "cuatro regímenes de un mismo principio" contra "cuatro mecanismos genuinamente distintos" para decidir entre las dos.

### C-N2.7.1 a C-N2.7.4 — los cuatro regímenes específicos (fuerte, electromagnético, débil, gravitacional)
⊘ **Resultado mixto, ya documentado en detalle en el informe de cronología.** El régimen fuerte "de a pares" (color) funcionó (CS026 ✅), pero su versión completa no-abeliana (el gluón, r7b) quedó bloqueada (❌, la Pared R7). El electromagnético (carga, r7d) funcionó (✅). El débil, en su forma de freeze-out protón:neutrón, es justamente la pieza de CS072 que **no sobrevivió la auditoría** (el 7:1 estaba puesto a mano). El gravitacional se construyó de verdad recién en CS073 (gravedad de N-cuerpos real) y mostró, en la batería robusta, que **si se apaga, la estructura no se liga** (✅ confirmado).

*Experimentos: CS026 (⊘pareado sí), CS027/CS031 (❌ no-abeliano/Higgs bloqueados), CS072 (⚠️ freeze-out fabricado), CS073 (✅ gravedad real confirmada por apagado).*

### C-N2.7.5 — Ley de regímenes de acoplamiento (el régimen depende de la escala, no de principios distintos)
⊘ Principio de diseño usado en todo el proyecto (la misma lógica de acoplamiento se aplica a escala de quarks, átomos y galaxias), consistente con los resultados pero no aislado como hipótesis propia verificable.

### C-N2.7.6 — Correspondencia estructural ≠ reducción física
🔷 **No es una afirmación empírica — es una aclaración de alcance que la propia Teoría hace sobre sí misma.** El nodo mismo dice que Cosmosemiótica no deriva las cuatro fuerzas ni pretende sustituir a la física que las describe formalmente — sólo afirma que, leídas desde las condiciones de posibilidad, las formas de interacción conocidas satisfacen la estructura I⟺E. No corresponde pedirle un experimento; es una cláusula de honestidad metodológica, y el proyecto la respeta (nunca declaró haber "derivado" las cuatro fuerzas desde cero, sólo haber implementado versiones de juguete inspiradas en ellas).

### C-N2.7.7 — Δ encadenada ⇒ distancia
*"La distancia no preexiste — emerge cuando varias distinciones se encadenan y forman un recorrido."*

✅ **Confirmado, y el propio texto de la Teoría cita a Cosmogénesis como la fuente de esa confirmación:** *"Los experimentos Cosmogénesis comprobaron operacionalmente que, al imponer localidad relacional fuerte, aparece un tejido con distancias efectivas que no es reproducido por los controles barajados o sin dinámica local."* Esto es, literalmente, el hallazgo de CS066 (localidad primero): con localidad fuerte sí aparece un tejido con especificidad real (agrupamiento 4× el control, diámetro triplicado).

*Experimento: CS066 (✅ tejido con distancia real). Script: cs066_localidad_geometrogenesis.py*

### C-N2.7.8 — Escalera de la geometría: Dir>0⇒Dim>0⇒Dist>0⇒S>0
*"Cada peldaño no garantiza por sí solo la aparición del siguiente."*

✅ **Confirmado — y de nuevo, el propio texto cita a Cosmogénesis:** *"Los experimentos Cosmogénesis midieron estos cuatro niveles separadamente y pueden presentar comportamientos distintos dentro de un mismo modelo."* Esto describe con precisión el arco CS066–CS069: la distancia (Dist) apareció, pero la dimensión (Dim) y la dirección (Dir) no siempre la siguieron — de hecho, casi nunca lo hicieron, hasta el punto de motivar el nodo siguiente.

*Experimentos: CS066–CS069 (la escalera medida en la práctica, con peldaños que no se siguen automáticamente).*

### C-N2.7.9 — Distancia y dirección son separables (Dist>0 ⇏ Dir>0)
✅ **Confirmado — es, palabra por palabra, el resultado de CS066.** El texto de la Teoría dice: *"los experimentos Cosmogénesis comprobaron esta separación sobre mismo sustrato: el tejido conservó distancias locales efectivas, mientras sus orientaciones no lograron establecerse y colapsaron frente a los controles correspondientes."* Coincide exactamente con lo reportado en la cronología: Nivel 1 (tejido/distancia) ✅, Nivel 2 (direcciones) ❌ — y de hecho las direcciones se agravaron, no sólo fallaron.

*Experimento: CS066. Script: cs066_localidad_geometrogenesis.py*

### C-N2.7.10 — La dirección no emerge de la relación pura, clásica ni cuántica (familia de modelos FCG)
✅ **Confirmado por dos rutas independientes, tal como el propio nodo lo describe:** *"Las rutas clásica y cuántica fueron ensayadas independientemente y, frente a sus respectivos controles nulos, convergieron independientemente al mismo resultado."* Esto es CS064–CS068 (ruta clásica) y CS069 (ruta cuántica, el frente cuántico) — dos caminos completamente distintos llegando a la misma pared. El nodo incluso usa la sigla "FCG" (familia de modelos de Cosmogénesis) para acotar el alcance de la conclusión — una honestidad metodológica que vale la pena destacar: el nodo no dice "la dirección es imposible en general", dice "no surgió en esta familia de modelos probada".

*Experimentos: CS064–CS068 (clásica), CS069 (cuántica). Ambas ❌, convergentes.*

### C-N2.7.11 — π y las constantes geométricas son contingentes, no predeterminadas
✅ **Confirmado — es, otra vez, un resultado medido de Cosmogénesis citado dentro de la propia Teoría:** *"Los experimentos Cosmogénesis comprobaron que esta razón permanece estable en redes con geometría definida y se vuelve inestable o indefinida en tejidos que poseen distancia y carecen de dimensión y dirección consolidadas."* Coincide exactamente con CS068: π sale constante (2.0, 2.99, 1.5) en redes con geometría real, y "explota" (de 2.5 a 48, sin asentarse) en las redes tipo mundo-pequeño.

*Experimento: CS068 (medición directa de π en distintas redes). Este es, probablemente, el nodo con el vínculo más directo y literal entre teoría y experimento de toda la Parte I.*

### C-N2.7.12 — La geometría macroscópica es un estado condensado, no a priori
✅ **Confirmado, en el mismo sentido que los tres anteriores** — es una lectura de más alto nivel del mismo arco CS066–CS069: la geometría no es el escenario donde ocurren las relaciones, sino algo que puede "condensarse" cuando las relaciones alcanzan cierta organización — y en la familia de modelos probada, esa condensación NO llegó a completarse (hay distancia, no hay geometría plena).

---

## BLOQUE 2.8 — Invariantes Cosmológicos del Cierre

*Este bloque completo (κ_P, κ_Δ, κ_O, κ_V, κ_LF, κ_H y sus consecuencias, C-N2.8 a C-N2.8.16) usa un vocabulario —"Libertad Funcional" (LF), error acotado, acoplamiento sistema-entorno como variable de cierre— que en la práctica del proyecto se operacionaliza sobre todo en el trabajo de la Célula Madre (VSTCosmo), no en Cosmogénesis. Ahí sí existen intentos concretos de medir estas cantidades (por ejemplo, "act_perm" como candidato a órgano que gestiona la Libertad Funcional). Pero eso es un proyecto distinto, fuera del alcance de esta revisión.*

**⏳ Sin experimento propio en Cosmogénesis, bloque completo.** Ninguno de los experimentos CS0xx, CF, Enfoque5, cs074 o cs075 mide directamente κ_P (mínimo de persistencia), κ_Δ (mínimo de diferencia operable), κ_O (cota de error), κ_V (piso de acoplamiento), κ_LF (mínimo de libertad funcional) ni κ_H (umbral de analizabilidad) como cantidades explícitas. Esto no significa que sean ideas descartadas — significa que, dentro de este código específico, siguen siendo teóricas, a la espera de que alguien diseñe el experimento que las mida.

*Nota aparte: si en algún momento se quiere una revisión de estos mismos nodos contra el trabajo de la Célula Madre, es un documento distinto — ese proyecto sí usa activamente el vocabulario de Libertad Funcional y acoplamiento sistema-entorno.*

---

## BLOQUE 3 — Historia

### C-N3, C-N3.1, C-N3.2 — Ω(t) acumula constricciones; |Ω(t+1)| < |Ω(t)|; Σ_t ΔΩ < 0
*"La historia acumula constricciones según la orientación temporal dominante; el número de estados posibles decrece en cada paso; la acumulación de constricciones es la firma estructural del tiempo."*

✅ **Confirmado — es, casi palabra por palabra, CS009.** "¿El desorden solo puede subir, nunca bajar, como en un vaso de agua que se puede desparramar pero no juntar solo?" — sí, monótono, sin excepciones. También es consistente con el hallazgo repetido en Enfoque 5 de que la entropía sólo crece y la energía útil sólo puede caer o mantenerse, nunca subir espontáneamente (verificado en 240 de 240 casos de "muerte térmica"). Y con CS014 (constantes vs. historia): algunos números son leyes (no cambian entre universos), otros son pura historia acumulada — la distinción misma que este bloque describe.

*Experimentos: CS009 (flecha del tiempo), CS014 (constantes vs. historia), BATERIA_ENFOQUE5 (240/240 casos de muerte térmica con entropía monótona).*

---

## BLOQUE 4 — Delimitación

### C-N4, C-N4.1, C-N4.2 — S=f(I,∂S); ∂S≠0 ⇒ sistema definido; interior∩exterior=∅
*"Un sistema se define cuando su persistencia presenta variación interna distinguible del entorno; sin frontera no hay sistema."*

⊘ **Operacionalizado una y otra vez, en cada nivel de escala del proyecto, pero nunca como una hipótesis propia y aislada — siempre como una herramienta que otro experimento necesita para funcionar.** Cada vez que el motor de CS072 decide "estos tres quarks son un barión" o "este electrón y este protón son un átomo", está aplicando este nodo: trazando una frontera que separa un sistema del resto. Lo mismo en CS073: el método de "amigos-de-amigos" que agrupa átomos en cúmulos, y la decisión de cuándo una región de gas "ya es" un sumidero (una estrella), son delimitaciones concretas, hechas una y otra vez. Nunca se corrió un experimento que preguntara "¿la delimitación en sí misma funciona?" como pregunta aislada — se la usa como herramienta dada por sentada en todos los demás.

*Uso extenso pero no aislado: cs072_modulos/nucleo.py (_detecta_trios), cs073 (friends-of-friends, criterio de sumidero).*

---

## BLOQUE 5 — Dinámica

### C-N5, C-N5.1, C-N5.2 — x(t) ∈ [x_min, x_max]; estabilidad dinámica en rango; colapso fuera de rango
*"Los sistemas estables operan dentro de rangos: ni demasiado ni demasiado poco."*

✅ **Confirmado, y con un ajuste muy preciso: el experimento cs074-A midió literalmente esta forma de "rango viable".** No es una coincidencia aproximada — es el resultado central de ese experimento: con asimetría inicial baja (hasta ≈0,5) el sistema se mantiene en una **meseta estable** (~77% de masa ligada, plano); pasado ese rango, primero fragmenta (0,9–2,3) y más allá (≳3,8) **colapsa por completo**. Es exactamente la forma "estable dentro de un rango, colapso fuera de él" que describe el nodo, medida con números concretos. También aparece en el propio arco de CS073: el control positivo con Phantom mostró núcleos que se estabilizan temporalmente en una "meseta" de densidad antes de (intentar) seguir colapsando — otra instancia del mismo patrón.

*Experimento clave: cs074A_asimetria_techo.py (✅ meseta/fragmentación/colapso, tres regímenes medidos). También: CS073, mesetas de densidad en el control positivo.*

---

## Tabla resumen

| Bloque | Nodo(s) | Estado | Experimento(s) clave |
|---|---|---|---|
| 1 | C-N1 (S>0) | 🔷 axioma, no se prueba | — |
| 1 | C-N1.2 (persistencia confirmada en t+1) | ✅ | CS007 y prácticamente todo el proyecto |
| 1 | C-N1.3 (la persistencia filtra) | ⊘ principio verdadero, filtro aislado débil | CS053 |
| 2 | C-N2 (S=I⟺E) | ✅ | CS008–CS025 |
| 2 | C-N2.2 (sin E o I, S=0) | ✅ | CS072 (apagados), CS073 (sin gravedad) |
| 2.5 | C-N2.5 (tiempo emergente) | ⊘ implementado, no auditado a fondo | CS072 |
| 2.5 | C-N2.5.5 (asimetría primordial) | ⊘ mixto: persiste sí, produce dirección no | CF-1 (✅) / CS070 (❌) |
| 2.5 | C-N2.5.6–10 (dirección temporal T⁺/T⁻) | ⏳ sin experimento propio | — |
| 2.6 | C-N2.6 (curvatura del espacio de estados) | ✅ | CS057, CS062 |
| 2.6 | C-N2.6.1–4 (gradientes, atractores) | ⊘ consistente, no aislado | cs074-A, Enfoque 5 |
| 2.7 | C-N2.7.1–4 (los cuatro regímenes) | ⊘ mixto por régimen | CS026 ✅ / CS027,CS031 ❌ / CS072 ⚠️ / CS073 ✅ |
| 2.7 | C-N2.7.7 (Δ encadenada ⇒ distancia) | ✅ citado en la propia Teoría | CS066 |
| 2.7 | C-N2.7.8 (escalera de la geometría) | ✅ citado en la propia Teoría | CS066–CS069 |
| 2.7 | C-N2.7.9 (distancia y dirección separables) | ✅ citado en la propia Teoría | CS066 |
| 2.7 | C-N2.7.10 (dirección no emerge, clásica ni cuántica) | ✅ citado en la propia Teoría | CS064–CS069 |
| 2.7 | C-N2.7.11 (π contingente) | ✅ citado en la propia Teoría | CS068 |
| 2.7 | C-N2.7.12 (geometría = estado condensado) | ✅ lectura del mismo arco | CS066–CS069 |
| 2.8 | Todo el bloque (κ_P, κ_Δ, κ_O, κ_V, κ_LF, κ_H) | ⏳ sin experimento en Cosmogénesis | — (existe trabajo relacionado en Célula Madre, fuera de alcance) |
| 3 | C-N3, C-N3.1, C-N3.2 (historia irreversible) | ✅ | CS009, CS014, Enfoque 5 |
| 4 | C-N4, C-N4.1, C-N4.2 (delimitación) | ⊘ usado siempre, nunca aislado | CS072, CS073 |
| 5 | C-N5, C-N5.1, C-N5.2 (estabilidad en rango) | ✅ | cs074-A |

---

## Síntesis: qué tan sólida está la Parte I de la Teoría, según Cosmogénesis

**Lo mejor sostenido, con evidencia directa y específica:** el Bloque 2.7 (regímenes de acoplamiento, y en particular la cadena distancia→dimensión→dirección, la separabilidad de distancia y dirección, y π como huella contingente) es, con diferencia, la parte de la Teoría con más apoyo experimental concreto — tanto que el propio documento canónico ya incorpora los resultados de Cosmogénesis como parte de su propio texto. También están firmes: la persistencia en el tiempo (Bloque 1), el acoplamiento como variable central (Bloque 2), la historia irreversible (Bloque 3), y la estabilidad dentro de un rango con colapso fuera de él (Bloque 5).

**Lo que hay que sostener con una salvedad:** todo lo que dependía del motor de partículas de CS072 (el régimen débil/freeze-out, la interpretación de la aniquilación como "cancelación de orientaciones") hereda la misma advertencia que ya se documentó en el informe de cronología — el experimento que se citaba como evidencia resultó, en su propia auditoría interna, parcialmente fabricado.

**Lo que queda sin experimento propio dentro de Cosmogénesis:** la dirección específicamente *temporal* de la flecha del tiempo (distinta de la dirección espacial, que sí se probó), y el bloque completo de los invariantes cosmológicos del cierre (κ_P, κ_Δ, κ_O, κ_V, κ_LF, κ_H) — este último con trabajo relacionado en otro proyecto de la Teoría (la Célula Madre), pero no en Cosmogénesis específicamente.

**Un patrón que vale la pena nombrar:** los nodos con mejor sostén son, casi todos, los que se pusieron a prueba con un control de azar explícito y sobrevivieron un intento honesto de refutarlos (CS066, CS068, CS069, cs074-A). Los que quedan más débiles son los que dependían de un solo motor sin ese tipo de control cruzado (el motor de partículas de CS072) o los que, hasta ahora, nadie diseñó cómo medir (el Bloque 2.8 entero). Es la misma lección del informe anterior, aplicada ahora nodo por nodo: la Teoría avanza donde se la sometió a la prueba más dura que se pudo diseñar, no donde simplemente se la dio por sentada.
