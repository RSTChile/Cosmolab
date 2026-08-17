# Experimentos específicos para los nodos a medio probar o sin probar (desde C-N2.5.5)

**Para:** Alexis López Tapia · **Base:** `VERIFICACION_NODOS_TEORIA_PARTE_I_2026.md` (los nodos marcados ⊘ o ⏳ desde C-N2.5.5 en adelante) · **Fecha:** 3-ago-2026

---

## Qué se hizo y cómo leer esto

Se tomó cada nodo que quedó "a medio probar" (⊘) o "sin experimento propio" (⏳) desde C-N2.5.5 en adelante, y para cada uno se preguntó: **¿existe hoy, con el código y los datos que ya tenemos, un experimento concreto que lo ponga a prueba con su control de azar? ¿Se puede correr ahora mismo, o hay algo que lo bloquea?**

De siete frentes abiertos, en dos ya se corrió el experimento nuevo (con resultado, abajo), en uno se descubrió que el experimento ya estaba pre-inscrito por el propio proyecto pero **bloqueado por un resultado anterior** (no hace falta inventar nada, hay que reportar honestamente que la puerta sigue cerrada), y en cuatro se diseñó el experimento (hipótesis, mecanismo, control NULL, criterio de falsación) pero no se corrió todavía — quedan listos para ejecutar cuando decidas.

Símbolos de estado:
- 🟢 **EJECUTADO** — se corrió, con resultado y veredicto.
- 🔒 **BLOQUEADO** — hay un experimento pre-inscrito, pero un resultado anterior del propio proyecto le cierra el paso; se explica por qué.
- 🟡 **DISEÑADO, no corrido** — protocolo completo, listo para ejecutar, pendiente de tu autorización o de tiempo de máquina.
- ⛔ **NO APLICABLE en Cosmogénesis** — el nodo pide algo que este código, por lo que es (no por descuido), no puede medir.

---

## 1 · C-N2.5.5 — Asimetría primordial: la puerta "semilla + métrica" 🔒 BLOQUEADO

**Lo que quedó pendiente:** la adjudicación de CS070 (17-jul-2026) encontró que la semilla primordial se "lava" en tres rutas independientes (clásica, cuántica, clásica+semilla), pero dejó anotado, sin cerrar, un cuarto camino: *"si algún día un experimento hace emerger un sustrato métrico genuino, la pregunta '¿la semilla prende AHÍ?' queda pre-inscrita desde hoy."* Es decir, el experimento YA está diseñado — lo único que falta es el ingrediente que necesita.

**Por qué está bloqueado, no pendiente de diseño:** ese ingrediente (un tejido con distancia real que crezca como √N, "métrico" en el sentido estricto) es exactamente lo que CS068 buscó y **no encontró** en su cierre de arco. El tejido residual, incluso después de podar todos los atajos de mundo-pequeño, midió un diámetro de 6-7,5 a N=900-2500 — **13 veces menor** que lo que un tejido 2D real daría a esa escala (~58-96). CS068 lo llamó, con toda razón, "Mundo B... compacto: mundo-pequeño hasta el fondo."

**Veredicto de este chequeo:** no hay nada nuevo que correr hoy sin violar la propia regla del proyecto (no fabricar la métrica que se busca). El experimento sigue pre-inscrito, tal como quedó anotado el 17-jul, a la espera de que **algún otro arco** (no éste) produzca un sustrato genuinamente métrico. Si eso pasa, la prueba (inyectar la misma semilla de CS070 sobre ese sustrato, comparar coherente vs. barajada) se puede correr en una tarde — es la infraestructura de `cs070_semilla.py` reutilizada tal cual.

*Nada ejecutado esta vez — verificación honesta de que la puerta sigue cerrada, y por qué.*

---

## 2 · BLOQUE 2.8 — Invariantes del cierre (κ_P, κ_Δ, κ_O, κ_V, κ_LF, κ_H) 🟢 EJECUTADO (parcial) + ⛔ (parcial)

**Por qué quedó "sin experimento propio":** nadie había definido qué, concretamente, contar como κ_P/κ_Δ/κ_V, etc., **en el sustrato físico real de Cosmogénesis** (Phantom, N-cuerpos). Existía sí un marco de trabajo — la auditoría de Codex del 17-jul (`AUDITORIA_CODEX_CS072_invariantes_C-N2.8_PARA_CS.md`) — pero estaba escrito para el motor de partículas de CS072, que la auditoría posterior retractó. Ese marco (llamado ahí "Puerta U", U0 a U6) es reutilizable igual: sólo había que aplicarlo a un candidato que SÍ sea real. **El mejor candidato del proyecto es el sumidero de CS073** (el resultado de mayor solidez estadística de todo Cosmogénesis, z=48,69): un sumidero es, literalmente, "una región que se distingue del entorno, persiste en el tiempo y crece por intercambio con su medio" — exactamente lo que U0-U3 piden.

### Lo que se pudo medir (U0-U3), sobre los datos YA existentes de la batería N=2000 (sin correr ninguna simulación nueva — se leyeron directamente los archivos `.sink` de Phantom, REAL vs NULL1-8)

| Invariante | Qué se midió | REAL | NULL (media±DE) | z | Veredicto |
|---|---|---|---|---|---|
| U0 (¿existe candidato?) | nº de sumideros que nacieron | 8 | 7,88 ± 0,35 | 0,35 | **No distingue.** Nacen sumideros en ambos casos — nacer no basta, hace falta ver qué hacen después. |
| U1 · κ_P (persistencia) | duración de vida / duración total de la corrida | 0,981 | 0,764 ± 0,033 | **6,53** | ✅ **Confirmado.** Los sumideros reales, una vez nacidos, viven casi toda la corrida; los de NULL se apagan antes. |
| U2 · κ_Δ (diferencia operable) | crecimiento de masa (masa final / masa inicial) | 3,74× | 1,53× ± 0,065 | **33,79** | ✅ **Confirmado, con la señal más fuerte de las cuatro.** |
| U2 · κ_Δ (alternativa) | masa total acretada por sumidero | 193,9 | 31,5 ± 3,7 | **43,33** | ✅ **Confirmado.** |
| U3 · κ_V (acoplamiento sostenido) | masa acretada en el último tercio de vida / en el primer tercio | 0,832 | 0,511 ± 0,235 | 1,37 | ⊘ **Dirección correcta, pero débil.** No alcanza el umbral que el proyecto exige (z≥3) — con 8 semillas NULL la varianza es demasiado grande para decidir. |

*(Nota técnica sobre U3: el primer proxy que se probó —correlación entre el tiempo y la masa acretada acumulada— dio r=1,000 en REAL Y en NULL por igual: es un artefacto, porque la masa acretada acumulada nunca puede bajar en este código. Se descartó y se reemplazó por el cociente "último tercio / primer tercio" de la tabla, que si mide algo real: si un sumidero está genuinamente acoplado a un flujo de escala mayor, su acreción se sostiene o crece con el tiempo; si sólo devoró una burbuja local aislada, se apaga. Vale la pena dejar anotado el proxy fallido para que nadie lo reuse.)*

**Lectura honesta:** de los tres invariantes de viabilidad que SÍ se pueden medir en un sumidero (κ_P, κ_Δ, κ_V), dos quedan confirmados con solidez estadística real (κ_P y κ_Δ) y uno queda en zona gris (κ_V, correcto en dirección, sin fuerza estadística con 8 semillas — correríamos más NULL si se quiere cerrar esto).

### Lo que NO se pudo medir — y por qué, honestamente, no es un vacío del código sino un límite del nivel de descripción

⛔ **κ_O (error operativo), κ_LF (libertad funcional) y κ_H (analizabilidad conductual) no son medibles sobre un sumidero — y no porque falte instrumentación, sino porque un sumidero no tiene qué medirle.** Lo dice con precisión la propia auditoría de Codex: κ_O necesita que el candidato tenga una "regularidad propia desde la cual distinguir respuesta esperada de respuesta realizada" (un sumidero no espera nada, sólo cae); κ_LF necesita "un repertorio propio de respuestas" (un sumidero no elige, sólo acreta); κ_H necesita variación de una *respuesta*, no de un estado físico. Estos tres invariantes describen algo con **conducta** — un organismo, no una bola de gas colapsando. Ese nivel de descripción existe en el otro proyecto de la Teoría (la Célula Madre, con `act_perm` como candidato a órgano conativo), no en Cosmogénesis. Forzarlos aquí sería precisamente el error que Codex advirtió: "confundir estos pares produciría un positivo por sustitución semántica, no por medición."

**Veredicto de Bloque 2.8, actualizado:** deja de ser "⏳ sin experimento propio, bloque completo" y pasa a un veredicto de dos niveles, tal como Codex propuso (Geometrogénesis / Viabilidad / Analizabilidad): **Geometrogénesis: sí (el sumidero emerge, CS073). Viabilidad (U_Cos_viab): parcialmente sí — κ_P y κ_Δ confirmados, κ_V sugerido no confirmado. Analizabilidad (U_Cos_anal, κ_H): no aplica a este nivel de descripción.**

*Script nuevo: `analisis_kappa_bloque28.py` (lee `cosmog01.sink` de `ic_real` e `ic_null1..8` en `/Users/alexis/phantom_cs073/bateria_n2000/`, sin simulación nueva).*

---

## 3 · C-N2.7.1-4 — Régimen débil (freeze-out): ¿se puede redimir la pieza retractada? 🟢 EJECUTADO — se confirma que NO se puede, y se mide por qué

**Contexto:** la auditoría de cronología ya había encontrado que `freeze_out.py` fabrica el 7:1 con una constante puesta a mano (`h = tasa_expansion*20.0`, línea 20). La pregunta nueva, específica: **¿ese 20.0 es una fabricación burda (cualquier valor cercano rompe todo) o es una estructura genuina que sólo necesitaba mejor justificación (el resultado es robusto en una banda ancha alrededor de 20)?** Si fuera lo segundo, valdría la pena buscarle una derivación real. Se puso a prueba barriendo la "escala" entre 1 y 1000, usando `tasa_expansion=0,02`, el valor que el proyecto usa de verdad en producción (`cs073_cierre_holistico.py`), no uno inventado para la prueba.

**Resultado del barrido:**

| escala | ratio p:n |
|---|---|
| 1 | 143,7 |
| 10 | 11,2 |
| **20** | **7,10** (el valor usado) |
| 30 | 5,68 |
| 100 | 3,42 |
| 1000 | 1,98 |

La función es suave y monótona (no es una fabricación "de precipicio" — no hay un valor mágico rodeado de caos). Pero acertar el número real (7, no 3 ni 11) exige una escala entre **16,4 y 27,0** — una ventana de factor 1,64×. Fuera de urgencia por justificar nada: **eso sigue siendo un ajuste fino, sólo que moderado, no extremo.**

**Veredicto:** el nodo **no se puede redimir con este mismo modelo de juguete.** Encontrar el 7:1 en una banda de factor 1,64× no es evidencia de estructura genuina, porque nada en el resto del motor de Cosmogénesis fija ese factor de forma independiente — habría que importar una escala física externa (un acoplamiento de Fermi real, una masa de Planck) que el resto del proyecto explícitamente se prohíbe usar (C-N2.8.15: "no son constantes físicas... ausencia de escalas físicas horneadas"). Es decir: **arreglarlo violaría la misma regla anti-Shannon que el proyecto usa para todo lo demás.** El régimen débil queda honestamente donde estaba: ⚠️ retractado, y ahora sabemos con precisión por qué no hay arreglo barato.

*Script nuevo: `redo_freezeout_sensibilidad.py`.*

---

## 4 · C-N2.5.6-10 — Dirección temporal (T⁺/T⁻), distinta de la flecha entrópica 🟡 DISEÑADO, no corrido

**Por qué es un nodo distinto de C-N3 (ya ✅):** C-N3 mide que el desorden sólo sube (una cantidad agregada, monótona). C-N2.5.6-10 pregunta algo más fino: **¿la propia regla de actualización, a nivel micro, distingue "adelante" de "atrás" en el tiempo, o sólo lo hace la estadística agregada?** Un ejemplo para que se entienda con una analogía: que un vaso de agua derramada nunca se junte solo es la flecha entrópica (agregada). Pero si grabás en video la caída de una sola gota y la reproducís al revés, un físico puede notar que "eso no puede pasar así" mirando sólo la gota — sin necesitar contar el desorden de todo el vaso. Eso segundo es lo que este nodo pide, y nadie lo midió todavía.

**Experimento propuesto (protocolo, sin correr):**
1. Tomar una trayectoria del campo básico (`cg001_field.py`), guardando el estado completo en cada paso (no un resumen agregado).
2. Elegir una cantidad microscópica que fluctúe en ambos sentidos (sube y baja), no una que sólo pueda subir — por ejemplo, la posición o velocidad de un nodo individual, no la entropía total.
3. Calcular un estadístico de irreversibilidad clásico: la asimetría (tercer momento) de los incrementos Δx_t = x_{t+1} − x_t. En un proceso reversible (como un paseo al azar simétrico) ese estadístico es cero en expectación; en un proceso genuinamente disipativo/direccional, no.
4. **Control NULL:** generar una serie sintética con la misma distribución marginal de incrementos pero sin la regla de actualización real (barajar el orden temporal de los mismos incrementos, o generar un paseo al azar simétrico con igual varianza). Comparar el estadístico REAL contra la distribución NULL con varias semillas.
5. **Criterio de falsación:** si el estadístico real cae dentro de la distribución NULL, el nodo queda sin sostén (la regla no tiene una dirección temporal propia, sólo agregada). Si cae afuera (z grande), confirma que hay una asimetría T⁺/T⁻ genuina a nivel micro.

**Factibilidad:** alta — es Python puro, reutiliza el campo ya construido, no necesita Phantom ni horas de cómputo (minutos). No se corrió esta vez por acotar el alcance de esta sesión a los dos experimentos ya ejecutados arriba; queda listo para la próxima si querés que lo corra.

---

## 5 · C-N2.6.1-4 — Gradientes de estabilidad, mínimos locales como atractores 🟡 DISEÑADO, no corrido

**Lo que falta aislar:** cs074-A ya midió que hay una "meseta" estable en un rango de asimetría inicial (evidencia indirecta de un atractor), pero nunca se comparó eso contra la alternativa obvia: **¿el sistema converge a esa meseta porque de verdad "cae" siguiendo un gradiente, o llegaría a un lugar parecido igual si el paso, en cada instante, apuntara en una dirección al azar (con la misma magnitud)?** Sin ese control, "atractor" y "cualquier proceso que se estabiliza solo" son indistinguibles.

**Experimento propuesto:**
1. Reusar el motor de `cs074A_asimetria_techo.py` (barrido de asimetría inicial → estado final).
2. Construir un brazo NULL: en cada paso de la dinámica, conservar la magnitud del cambio que el modelo real produce, pero **reasignar su dirección al azar** (rompe "seguir el gradiente", conserva "cuánto se mueve").
3. Comparar, REAL vs NULL, en varias semillas: (a) el ancho de la meseta estable, (b) la fracción de corridas que terminan en la misma cuenca/atractor, (c) la varianza del estado final.
4. **Falsación:** si el NULL (dirección al azar) produce una meseta de ancho y varianza estadísticamente indistinguible de la REAL, el nodo cae — la estabilidad sería un efecto de la magnitud del paso, no de "caer hacia" nada. Si la meseta REAL es mucho más angosta/estable que el NULL, confirma que hay un gradiente genuino guiando la trayectoria.

**Factibilidad:** alta — reutiliza código existente casi sin cambios (sólo hay que interceptar el paso de actualización y reemplazar su dirección). Estimado: una tarde de cómputo dado que cs074A ya corre rápido.

---

## 6 · C-N2.7 / C-N2.7.5 — ¿Los cuatro regímenes son UN principio a distinta escala, o cuatro mecanismos distintos? 🟡 DISEÑADO (más costoso, prioridad menor)

**Por qué es el más difícil de los siete:** no es una pregunta de "¿pasa o no pasa?" sino de **comparación de modelos** — hay que demostrar que una familia de reglas con un solo parámetro de escala explica el patrón de éxito/fracaso observado en los cuatro regímenes (fuerte-pareado ✅, no-abeliano ❌, EM ✅, débil ⚠️, gravedad ✅) tan bien como cuatro reglas independientes, sin ajustar ese único parámetro por separado para cada caso.

**Experimento propuesto (esquema, requiere más diseño antes de codear):**
1. Tomar los cuatro (cinco) resultados ya medidos como patrón objetivo: qué combinaciones de alcance/intensidad tuvieron éxito y cuáles no.
2. Formular la "ley de régimen único" del nodo C-N2.7.5 como una función de un solo parámetro de escala (alcance/intensidad) que prediga el patrón observado.
3. Ajustar esa función a los cuatro regímenes SIMULTÁNEAMENTE (un solo conjunto de parámetros para los cuatro) y comparar el ajuste contra permitir un parámetro libre por régimen (equivalente a "cuatro mecanismos distintos").
4. **Falsación:** si el modelo de un solo parámetro no puede reproducir el patrón sin un parámetro adicional por régimen, el nodo (unificación) queda sin sostén — apoyaría más bien "cuatro mecanismos" con una interfaz común, no un principio único.

**Factibilidad:** media-baja. A diferencia de los otros cinco, éste no es una corrida nueva sencilla — exige primero decidir la forma funcional de "la ley de régimen único", lo cual es una decisión teórica, no sólo de ingeniería. **Recomiendo no correrlo todavía** sin que primero se adjudique esa forma funcional (riesgo de terminar fabricando la unificación en vez de probarla, exactamente el vicio que este proyecto evita). Queda anotado como pendiente de diseño teórico previo, no de código.

---

## 7 · C-N4 — Delimitación como hipótesis propia (¿la frontera es real, o cualquier frontera sirve?) 🟡 DISEÑADO, no corrido — bloqueo técnico identificado

**Lo que falta aislar:** en CS073, el criterio "amigos-de-amigos" (friends-of-friends) traza fronteras entre sumideros y el resto del gas usando una longitud de enlace fija. Nunca se preguntó, como hipótesis propia: **¿esa frontera cae donde el gas de verdad tiene una discontinuidad de densidad (∂S≠0 real), o cualquier longitud de enlace razonable dibuja "algo parecido a un grupo" igual, incluso sobre un campo sin estructura?**

**Experimento propuesto:**
1. Tomar las posiciones de partículas de una corrida ya hecha (REAL y NULL de la batería N=2000).
2. Barrer la longitud de enlace del friends-of-friends en un rango amplio y medir el histograma de distancias entre partículas vecinas: si hay una frontera real, debería aparecer un "vacío" natural en ese histograma (un salto claro entre "vecinos del mismo grupo" y "vecinos de grupos distintos") que NO aparece en el campo NULL barajado.
3. **Falsación:** si el histograma de distancias de REAL y NULL son indistinguibles (ambos suavemente decrecientes, sin salto), la "frontera" es un artefacto de la longitud de enlace elegida, no algo que el sistema mismo produzca. Si REAL muestra un vacío claro que NULL no muestra, confirma que la delimitación captura algo genuino.

**Por qué no se corrió esta vez — bloqueo técnico, no conceptual:** esto requiere leer las posiciones crudas de los volcados binarios de Phantom (`cosmog_00030`, etc.), que están en el formato propio de Phantom (sphNG/Fortran binario), no en un formato que Python pueda leer directamente. El proyecto no tiene todavía un lector de volcados en Python (herramientas como `plonk` o `sarracen`, o compilar las utilidades propias de Phantom, `phantom2gadget`/`phantomsinks`). Instalar/probar eso es una tarea de infraestructura de un rato, no una decisión de diseño — queda identificada como el único paso que falta para correr este experimento.

*Nota: el análisis del Bloque 2.8 (sección 2) evitó este problema porque los archivos `.sink` sí son texto plano — pero las posiciones de TODAS las partículas de gas (necesarias para el histograma de distancias) sólo están en los volcados binarios.*

---

## Resumen

| Nodo | Estado | Resultado / razón |
|---|---|---|
| C-N2.5.5 (semilla+métrica) | 🔒 bloqueado | pre-inscrito por CS070, pero CS068 nunca produjo el sustrato métrico que necesita |
| Bloque 2.8, κ_P/κ_Δ/κ_V | 🟢 ejecutado | κ_P ✅ (z=6,53), κ_Δ ✅ (z=33,8/43,3), κ_V ⊘ (z=1,37, dirección correcta, débil) |
| Bloque 2.8, κ_O/κ_LF/κ_H | ⛔ no aplicable | requieren conducta/repertorio de respuesta — nivel de descripción de la Célula Madre, no de Cosmogénesis |
| C-N2.7.1-4 (débil, redo) | 🟢 ejecutado | confirma que NO se puede redimir sin violar la regla de no hornear escalas físicas |
| C-N2.5.6-10 (dirección temporal) | 🟡 diseñado | protocolo listo, Python puro, no corrido esta sesión |
| C-N2.6.1-4 (gradientes/atractores) | 🟡 diseñado | protocolo listo, reutiliza cs074A, no corrido esta sesión |
| C-N2.7/2.7.5 (unificación de regímenes) | 🟡 diseñado, prioridad baja | necesita adjudicación teórica previa (forma funcional) antes de poder codearse sin fabricar el resultado |
| C-N4 (delimitación aislada) | 🟡 diseñado, bloqueo técnico | necesita un lector de volcados Phantom en Python (no existe todavía en el proyecto) |

**En una línea:** de los siete frentes, dos ya tienen resultado nuevo y verificado (uno mixto-positivo en Bloque 2.8, uno negativo honesto en el régimen débil), uno queda formalmente cerrado-por-ahora con la razón exacta documentada, y cuatro quedan con protocolo completo y listo para correr — tres de ellos en cuanto decidas, y uno (unificación de regímenes) necesita primero una decisión teórica tuya, no de código.
