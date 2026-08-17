# Auditoría completa de Cosmogénesis — de la primera regla al intento de encender una estrella

**Para:** Alexis López Tapia · **Preparado por:** Claude Code, con cuatro agentes de investigación en paralelo · **Fecha:** 3-ago-2026

---

## Cómo leer este documento

Este es un recorrido completo, de punta a punta, por todo lo que se probó en Cosmogénesis: desde el primer experimento (CS001, un campo de juguete) hasta la última corrida de esta misma semana (el intento de encender la primera estrella con el programa profesional Phantom). Está escrito para leerse sin tecnicismos — cada concepto técnico que aparece se explica con una frase simple o una analogía — pero cada fase indica al final qué **script real** la implementó, para que cualquiera pueda ir a mirar el código si quiere.

**Los símbolos de veredicto que vas a ver:** ✅ = salió, resultado real · ⊘ = parcial o con tensión · ❌ = no salió / se refutó · 🔒 = cerrado. Un ❌ en este proyecto **no es un fracaso** — es información: significa que se tachó una posible explicación, y tachar explicaciones es cómo se avanza cuando lo que se busca es la explicación verdadera, no una que simplemente "funcione".

**Advertencia sobre honestidad, antes de empezar:** este documento no suaviza nada. En el camino vas a encontrar al menos dos episodios donde el propio equipo (vos, el asistente de diseño, o el colaborador técnico) cometió un error serio, lo detectó, y lo corrigió publicándolo sin maquillaje — incluyendo un caso donde el resultado más celebrado de todo el proyecto tuvo que ser parcialmente retractado. Se cuentan tal cual, porque son la prueba de que la regla de la casa (nunca declarar una victoria que no le gane limpio al azar) se cumple incluso cuando duele.

**Índice:**

1. El primer gran tramo (CS001–CS071): la cacería de por qué vivimos en 3 dimensiones
2. CS072: el "positivo mayor" — y su auditoría, que lo puso en duda
3. CS073: de los átomos a la primera estrella (el trabajo de esta semana)
4. La rama paralela: ¿el modelo predice los números exactos de nuestro universo?
5. Tres hilos exploratorios: el campo de "rastro de hormigas", el motor de 7 etapas, y las notas conceptuales
6. Síntesis final: dónde está parado el proyecto hoy

---

# PARTE 1 — El primer gran tramo: de CS001 a CS071, la cacería de por qué vivimos en 3D

**Qué es esta parte:** una reconstrucción cronológica de la primera gran cadena de experimentos del proyecto. Cubre desde el primer experimento (CS001, un campo mínimo) hasta el experimento CS071, que cierra —por ahora— la pregunta más grande que se hizo el equipo: **¿por qué el espacio en que vivimos tiene 3 dimensiones (y no 2, ni 4, ni una forma curva rara)?**

**La imagen general para no perderse:** el equipo sospechaba que el espacio de 3 dimensiones en que vivimos no es un dato de fábrica del universo, sino algo que tuvo que salir de algún lado. La estrategia fue construir universos de juguete —programas muy simples, sin geometría puesta a mano— y probarles, uno por uno, todos los "ingredientes" que un físico metería en la lista de sospechosos para explicar el 3D: las fuerzas, el espín, la masa, la gravedad, la localidad, hasta el propio proceso de formación. Este tramo es la historia de cómo, ingrediente por ingrediente, **todos fueron descartados**, y de cómo esa cacería terminó revelando algo inesperado sobre el número π.

## FASE 0 — CG001: el campo básico (CS001–CS007)

**La pregunta más simple posible:** si le doy a un campo continuo (algo como una superficie de agua) una sola arruga inicial minúscula, ¿el resto de la estructura —zonas densas, huecos, memoria de lo que pasó— aparece sola, o hay que dibujársela?

**Analogía:** pensá en un estanque perfectamente liso al que le tirás una sola gota en el centro. La regla dice: "cada punto del agua reacciona a su vecindad, y la reacción que sostuvo su forma se recuerda un rato (no para siempre)". Nadie le dice al agua "formá un círculo": el patrón, si aparece, tiene que salir de la regla local, no de un diseño.

**Qué se probó en cada experimento:**

- **CS001** — ¿aparece algún campo con estructura, desde esa única arruga? (base de todo el arco).
- **CS002/CS003** — barridos gruesos y finos de los parámetros del campo.
- **CS004** — corridas de producción a mayor escala.
- **CS005** — ¿hay una estructura de causa-efecto real, o es puro ruido?
- **CS006** — ¿la diferencia se "localiza" (se queda en un lugar) o se disuelve?
- **CS007** — ¿las diferencias persisten en el tiempo? Resultado: ✅ sí, hay una primera confirmación firme (llamada C-N1 en la teoría: que una diferencia dure es la condición mínima para que haya algo que estudiar).

**Resultado de la fase:** una base parcial (⊘), suficiente para seguir. Sirvió como cimiento técnico sobre el que se construyó todo lo que sigue.

*Script(s): cg001_field.py, cg001_barrido_grueso.py, cg001_barrido_fino.py, cg001_ipad.py, cg001_test_causalidad.py, cg001_test_localizacion.py, cg001_ipad_persistencia.py*

## FASE 1 — CG002: el "modelo estándar" de juguete (CS008–CS033)

La idea cambia de "un campo continuo" a "una población de diferencias" — como un salón lleno de miles de flechitas/agujas de brújula, cada una con una intensidad y una orientación. Las agujas se influyen entre sí: si dos apuntan parecido, cooperan; si apuntan opuesto, compiten. La pregunta: ¿de ese tira y afloja emergen, sin programarlos, el tiempo, el espacio, la estructura, y hasta partículas tipo las del modelo estándar de la física?

### 1.1 El arco fundacional (CS008–CS016, CS025)

- **CS008 (bariogénesis)** — a cada aguja se le da una "anti-aguja" (orientación opuesta). Resultado: ✅ sobrevive cerca del 50%, y una asimetría inicial mínima (2%) se **amplifica 25 veces**. *Cambio de rumbo:* al principio parecía contradecir la teoría (el exceso salía incluso con promedio cero), hasta que se corrigió el concepto de fondo: **la asimetría ES la diferencia misma, no su promedio**.
- **CS009 (flecha del tiempo)** — ¿el desorden solo puede subir, nunca bajar? ✅ sí, monótono.
- **CS010 (dimensión heredada)** — si a las agujas les doy más "grados de libertad" para apuntar, ¿el espacio que se arma mide más dimensiones también? ✅ sí, casi 1 a 1.
- **CS011 (coexistencia de dominios)** — ✅ sí, con interacción local.
- **CS012 (criticidad)** — ¿hay un punto exacto de transición de fase? ✅ sí, pico claro.
- **CS013 (inercia histórica)** — ⊘ depende de un detalle técnico de cómo se limita el peso máximo.
- **CS014 (constantes vs. historia)** — ✅ hay números que salen siempre iguales entre universos (una "ley"), y otros puramente de historia. Primera pista de que **las constantes son estructurales, no numéricas**.
- **CS015 (invariantes)** — ✅ tres formas cualitativamente distintas de "romperse".
- **CS016 (paredes de dominio)** — ❌ no aparecen fronteras nítidas.

### 1.2 Verificación de robustez (CS017–CS022)

**CS017** corrió 1000 universos, constantes estables (variación <2%). **CS018/CS019** confirmaron que el exceso de materia era real (88-95%). **CS020** construyó una línea de base puramente matemática para comparar. **CS021**: el motor real se separa 12 veces del azar. **CS022**: el número "mitad" (que sobrevive la mitad de las agujas) — ¿ajuste a mano o derivación pura? ✅ derivación pura, no se mueve ni un poco al variar el parámetro relacionado.

### 1.3 El "modelo estándar de partículas" adentro del juguete: la serie r7 (CS026–CS033)

- **CS026 (color+espín, r7a)** — ✅ funciona, de a pares.
- **CS027 (gluón, r7b)** — ❌ **BLOQUEADO**. El gluón necesita tres cosas interactuando a la vez; el motor solo sabe de a dos.
- **CS028 (leptones, r7c)** — ⊘ parcial.
- **CS029 (carga eléctrica, r7d)** — ✅ funciona (de a pares).
- **CS030 (generaciones, r7e)** — ✅ funciona.
- **CS031 (Higgs/masa, r7f)** — ❌ **BLOQUEADO por la misma razón que el gluón**.
- **CS032/CS033** — intentos de extender a tres puntos. ⏳ la pared no cayó.

> **🔑 CAMBIO DE RUMBO — LA PARED R7:** todo lo que en física real es "de a dos" funcionaba; todo lo que necesita "de a tres" (el gluón, el Higgs) quedaba bloqueado. Analogía: el motor es excelente armando apretones de manos entre dos personas, pero no puede manejar un trato a tres bandas. Esta pared, cerrada formalmente el 30-jun-2026, se vuelve el hueco que persigue el resto del arco durante meses.

**Resultado global de CG002:** un arco enorme y en gran parte positivo — de una sola regla mínima salieron tiempo con flecha, espacio con dimensión heredada, estructura, transición de fase real, y buena parte del sector "de a pares" del modelo estándar. Lo que no salió (gluón, Higgs) se reportó como negativo honesto.

*Script(s): cg002_experimentos_arco.py, cg002_baryogenesis.py, cg002_constantes_1000.py, cg002_exceso_barrido.py, cg002_exceso_caracteriza.py, grain_null_model.py, cg002_dynamic_l2_sweep.py, cg002_acoplamiento.py, cg002_multicomponente.py, cg002_r7a_color_spin.py … cg002_r7h_chiral.py. Cierre: CIERRE_ARCO_CG002_AUTORITATIVO.md*

## FASE 2 — CG003: intentar hacer crecer el espacio desde cero (CS034–CS038)

**El cambio de pregunta:** si empiezo de la relación pura —quién está conectado con quién, sin coordenadas x,y,z— ¿puedo hacer crecer un espacio real, con distancia y extensión?

**Analogía central:** una red social gigante donde solo sabés "quién conoce a quién". Podés medir fácil "cuántos pasos de amistad hay entre dos personas", pero eso no da automáticamente un mapa con extensión real: puede pasar que cualquiera esté a 6 pasos de cualquier otra persona (el fenómeno de "mundo pequeño", como Facebook) — con orden, pero sin un "lejos" genuino.

- **CS034 (espacio relacional)** — ⊘ se fragmenta, no converge.
- **CS035 (exergía primero)** — ⊘ soporte, no resuelve.
- **CS036 (crecimiento)** — la dimensión medida **trepa sin parar** con el tamaño (1.89→2.24→2.53→2.77) — la firma clásica de mundo pequeño, no de geometría real. ❌
- **CS037 (campo angular)** — dar a cada nodo una "brújula local". Empieza a rastrear cuántas direcciones se le dieron, pero el diámetro sigue creciendo como mundo pequeño.
- **CS038 (planitud por exergía, cg003f)** — ❌ no se despliega, primer intento degenerado.

> **🔑 CAMBIO DE RUMBO — SE LE PONE NOMBRE PRECISO AL MURO: "PLANITUD"/HOLONOMÍA.** No basta con darle a cada nodo una dirección local — falta que esas direcciones compongan bien alrededor de un lazo cerrado (holonomía: si caminás en redondo, tu brújula tiene que volver a apuntar igual que al salir, si no, tu mapa tiene curvatura escondida, como un globo terráqueo comparado con un mapa plano). Cruce interesante: la misma pared apareció, el mismo mes, en la física de partículas (la quiralidad fallaba por la misma razón) — dos frentes totalmente distintos chocando con el mismo muro.

*Script(s): cg003_espacio_relacional.py, cg003b_exergia_primero.py, cg003c_crecimiento.py, cg003d_campo_angular.py, cg003f_planitud_exergia.py*

## FASE 3 — CG004: ¿se puede "coser" un espacio plano? (CS039–CS046)

**Analogía general:** es como coser parches de tela para que formen una sábana perfectamente plana — sin poder mirarla entera desde arriba, solo dando instrucciones de "cosé este parche con el de al lado, girando tantos grados".

- **CS039 (attach)** — ❌ no genera plano.
- **CS040 (ciclos)** — ❌ igual que sin ciclos.
- **CS041 (robustez)** — ✅ el negativo se sostiene con 8 semillas.
- **CS042 (dos frentes)** — ❌, y se cazó un error real: la "curvatura" medida se retroalimentaba a sí misma sin medir nada externo.
- **CS043 (retícula cortada)** — ✅ primer resultado positivo de MEDICIÓN (plano = univaluado), no todavía de generación.
- **CS044/CS045 (curvatura, cortar-repegar)** — ⊘ chocan con muros técnicos (una sola bisagra absorbe toda la deformación).
- **CS046 (cinta-Eisenstein)** — ✅ **(P-κ) cerrado**: coser/pegar preserva la planitud donde ya existía, pero **nunca la genera** donde no la había. Como plegar una sábana ya lisa en distintas formas: nunca la convertís en lisa solo doblándola distinto.

**Resultado:** negativo firme — pegar/coser localmente NUNCA genera planitud.

*Script(s): cg004_attach.py, cg004b_ciclos.py, cg004c_robusto.py, cg004d_dosfrentes.py, cg004e_reticula_cortada.py, cg004f_barrido_curvatura.py, cg004f2_barrido_cortar.py, cg004f3_cinta_eisenstein.py*

## FASE 4 — CG005: el color como identidad, ¿alcanza para armar espacio? (CS047–CS050)

Vía paralela: en vez de la geometría directamente, se ataca desde el color (identidad, sin dirección). ¿De ahí sale un espacio real?

- **CS047 (v0, EDS)** — ✅ confina al 100% contra 82% de línea de base sin sentido.
- **CS048 (v1, orden temporal)** — ⊘ confina mejor (89% vs 62%) pero da "gas", sin forma.
- **CS049 (v2, fuerza residual)** — ❌ sigue siendo gas/blob amorfo.
- **CS050 (v3, energía emergente)** — ❌ negativo, aunque pasa el control de que no es solo etiquetas renombradas al azar.

> **🔑 CIERRE CONJUNTO CG004+CG005 — momento clave:** dos caminos totalmente independientes (geometría pura y color/identidad) **llegaron a la misma pared por separado**. Ninguna regla que mire solo "con quién te conectás" (la adyacencia) genera planitud. Falta algo sobre el **MARCO** — no solo con quién te conectás, sino con qué orientación relativa.

*Script(s): cg005_eds_v0.py … cg005_eds_v3.py. Cierre: adjudicacion_cg005_v3_CIERRE_ARCO_CS.md*

## FASE 5 — La puerta: "el espín es el hacia-dónde que faltaba" (CS051)

**El momento clave de la intuición de Alexis.** Se identifica el ingrediente físico real que faltaba: el **espín**.

**Analogía:** el color de un quark es como el nombre de una persona (quién es, no hacia dónde mira). El espín es como la orientación de su cuerpo — una flecha incorporada. Cuando dos partículas con espín se ligan, sus orientaciones se acoplan — exactamente lo que ninguna regla anterior podía producir, porque ninguna pieza tenía una brújula que transmitir.

**Estado:** hipótesis documentada, todavía no corrida en esta fase.

*Script(s): ninguno todavía — documento de diseño: PUERTA_R7_espin_como_marco.md*

## FASE 6 — CS052: la co-emergencia del espacio (v0 y v1)

- **CS052-v0 (marco por-nodo)** — ❌ da "gauge puro", la curvatura siempre da cero. Pero reveló algo productivo: **la curvatura no puede vivir en el nodo — tiene que vivir en el ENLACE.**
- **CS052-v1 (co-emergencia)** — el experimento decisivo: tres brazos para aislar dónde vive el espacio. **A** (orientación en la entidad sola) → predicción: cero. **B** (orientación en el vínculo, pero suelto) → predicción: también cero. **C** (orientación en el vínculo, atado a los dos extremos) → predicción: éste sí discrimina plano de curvo.

  **Analogía:** tres formas de poner una regla entre dos edificios. (A) pegada a uno solo — no mide la relación. (B) una soga floja colgada — no dice nada de alineación. (C) una viga rígida atornillada a ambos — ESA sí revela si los edificios están alineados.

  **Resultado: ✅ CONFIRMADO exactamente como predijo la tesis** — A=0, B=0, C sí discrimina. El espacio "vive" en el vínculo atado a sus dos extremos. Primer resultado positivo de generación real del arco geométrico.

*Script(s): cs052_marco_espin.py, cs052_v1_coemergencia.py*

## FASE 7 — El paisaje: persistencia, gravedad y las cuatro fuerzas (CS053–CS057)

- **CS053 (filtro de persistencia)** — ❌ falsación honesta: sobrevive CUALQUIER retículo de dimensión ≥2 por igual. Examen tan fácil que aprueba a cualquiera con pulso.
- **CS054 (gravedad sin alcance)** — ❌ acotado: sin que la gravedad se debilite con la distancia, todo colapsa en un amasijo, sea cual sea la dimensión.
- **CS054-v2 (gravedad CON alcance)** — ⊘ deja de colapsar todo (confirma la intuición de Alexis) **pero selecciona 2D-plano, no 3D** — nuestro propio universo queda refutado por esta versión.
- **CS055 (proceso acoplado)** — se hacen visibles **dos fuerzas tirando en direcciones opuestas**: el confinamiento sostiene el 3D, la gravedad sola lo colapsa a 2D. En proporción 1:1, gana la gravedad.
- **CS056 (las cuatro fuerzas reales)** — se reduce todo al confinamiento; el electromagnetismo no rescata el 3D. Hallazgo colateral: color y carga eléctrica son dos "neutralidades" independientes que compiten. Queda un hueco: gravedad y EM se corrieron con el mismo alcance, cuando en la realidad son muy distintos.
- **CS057 (EL PAISAJE COMPLETO)** — 69.648 universos, barriendo todas las fuerzas a la vez. **Resultado — titular del arco de fuerzas:** el punto de los valores físicos reales SÍ cae en zona viable (4× más probable que el fondo al azar) **pero estabiliza geometría curva, no la 3D-plana en la que vivimos.** El proceso sincrónico supera al asincrónico (+10%, significativo) — el orden importa. Aparece un candidato honesto a "energía oscura" (aceleración emergente, nadie la insertó).

> **🔑 CIERRE DEL ARCO DE FUERZAS:** después de CG004, CG005, CS054-56 y CS057, conclusión unánime: **ninguna fuerza local fija que vivamos en 3D-plano.** Todo apunta "aguas arriba", al espín/marco.

*Script(s): cs053_persistencia_geometria.py, cs054_gravedad_en_el_filtro.py, cs054_v2_gravedad_alcance.py, cs055_proceso_acoplado.py, cs056_cuatro_fuerzas.py, cs057_paisaje_completo.py*

## FASE 8 — El arco de cierre: energía oscura, marco, masa, y el vértice de tres cuerpos (CS058–CS063)

- **CS058 (zoom a la energía oscura)** — con datos parciales parecía "artefacto limpio"; **con datos completos (1404 puntos) se corrigió** a "real-pero-débil" (supera al azar 1.66×, pero decae con resolución). **Lección que el equipo se apuntó a sí mismo: no declarar veredicto firme desde una corrida parcial.**
- **CS059 — EL EXPERIMENTO AL QUE APUNTABA TODO EL ARCO: el espín como marco.** Resultado inicial: ¡parecía seleccionar geometría curva!

  > **🔑 CAMBIO DE RUMBO — FALSO POSITIVO CAZADO:** al controlar por "longitud del ciclo" del lazo medido, la aparente selección **se desintegró por completo** — era un artefacto geométrico trivial, no el marco. ❌ el marco de espín pareado NO selecciona dimensión. Apunta, de nuevo, a que hace falta un vértice de TRES puntos.

- **CS060 (leptones y masa)** — Parte A: la masa cambia el resultado (efecto de umbral) pero, controlado, no selecciona dimensión. **Parte B — LA GRIETA POSITIVA:** gravedad proporcional a masa real (no al "grado"/conectividad) hace el 3D/4D **3× más viable** — pero al barajar las masas al azar, el resultado **no cambia**. No era la masa: era que usar el "grado" como proxy se autoamplifica ("rico se hace más rico"), sesgando todo el arco anterior contra el 3D. Grieta real en el negativo de CS057, pero apunta a un problema de MEDICIÓN, no a que la masa sea la solución.
- **CS061 (masa emergente, vértice tipo Higgs)** — ❌ colapsa contra el control nulo, espectro de masas trivial (2:1, lejos del real 3477:1). Matiz honesto verificado en código: la dinámica seguía siendo de a pares; el vértice de tres cuerpos genuino aún no se había probado en la actualización misma.
- **CS062 (re-correr el paisaje con gravedad correcta)** — 52.248 universos, gravedad proporcional al peso real. El 3D/4D global sube de 11% a 16.2% (confirma que el proxy de "grado" exageraba el negativo) **pero el fondo del problema no se mueve**: el punto físico real sigue estabilizando geometría curva. Hallazgo decisivo: barajar las masas da lo mismo que usarlas reales — **no es la identidad de la masa, es la forma del acoplamiento.**
- **CS063 (vértice de tres cuerpos GENUINO, cierre)** — actualización donde los tres marcos se mueven juntos de verdad (verificado matemáticamente). ❌ colapsa igual contra el control nulo. Cierra la última puerta.

> **🔑 EL CIERRE GRANDE: "EL ARCO DE ELIMINACIÓN LOCAL ESTÁ COMPLETO".** Ni las fuerzas, ni el marco de espín pareado, ni la masa (dada o emergente), ni el vértice de tres cuerpos genuino seleccionan la dimensión. Dos de estos negativos atraparon falsos positivos del propio equipo (CS059, CS060-B) antes de confiar en el resultado — el sistema de auto-chequeo funcionó. **Recién acá, la hipótesis de que la dimensión es CONTINGENTE se gana el derecho a ser tomada en serio.**

*Script(s): cs058 (energía oscura), cs059 (espín como marco), cs060_leptones_y_masa, cs061_masa_emergente.py, cs062 (paisaje con peso intrínseco), cs063_vertice_3cuerpos_genuino. Adjudicación: adjudicacion_ARCO_CS058-061_CS.md*

## FASE 9 — ¿Y si el problema es que no hay ni siquiera "espacio local"? (CS064–CS071)

Giro completo de enfoque: en vez de "¿qué fuerza elige el 3D?", ahora: **¿el universo de juguete siquiera TIENE un espacio con un "lejos" real?**

- **CS064 (sistema completo desde sopa caliente)** — confirma que las direcciones son un reúso de la inercia del marco (congelar el marco → las direcciones mueren, de 1.32 a 0.00 ejes).

  > **🔑 SE DESTAPA EL BLOB.** El sustrato resultante es un amasijo "ultra-mundo-pequeño" — su diámetro NO crece al agregar más nodos, como en una red social donde cualquiera está a pocos saltos de cualquier otro. **Nunca hubo un "lejos" real para empezar.**

- **CS065 y CS065b (exclusión tipo Pauli)** — ❌ en ambas formas. La primera REDUCE los ejes en vez de aumentarlos; la segunda (más fiel, pre-registrada como decisiva) da exactamente lo mismo real que barajado.
- **CS066 (la localidad primero — "el tejido antes que los ejes")** — Nivel 1 (tejido): ✅ con localidad fuerte sí aparece un tejido con especificidad real. Nivel 2 (direcciones): ❌ — y peor, el colapso a un eje se **agrava** sobre ese tejido respecto de la versión barajada. El tejido apretado SUPRIME las direcciones.

  > **🔑 REORDENA TODO EL ARCO:** "espacio local" y "direcciones múltiples" son **dos problemas separados**, no uno solo.

- **CS067 ("la habitación completa")** — los 17 ingredientes del arco juntos, 160 corridas blindadas. ❌ **(B) canónico** — en ningún régimen la combinación completa enciende direcciones múltiples estables; los controles igualan o superan a la versión completa.
- **CS068 (análogo de inflación — "estirar y enfriar")** — la hipótesis: a mayor distancia, menor temperatura, los "atajos" largos se rompen primero al enfriar. En un modelo sintético FUNCIONA; sobre el blob real, con el juez correcto (escalamiento del diámetro), el tejido residual es **13× más chico** que una geometría 2D real — sigue siendo mundo-pequeño hasta el fondo. CC volvió a cazar su propio falso positivo (una semilla prometedora, revertida con 4 semillas).

  > **🔑 CIERRE DEL ARCO CLÁSICO DEL ESPACIO: CS066+CS067+CS068 convergen — la distancia SÍ emerge (hay "lejos" real), pero NO se traduce en dimensión ni dirección. Distancia y dirección son SEPARABLES.**

- **CONSECUENCIA — π ES UNA HUELLA, NO UNA LEY PREVIA.** Si la distancia existe pero la dimensión no cuaja, π no puede estar predefinido. Medido: donde SÍ hay geometría clara, π sale constante pero depende de la red (2.0 en cuadrada, 2.99 en triangular, 1.5 en hexagonal); donde NO la hay (mundo pequeño), π **explota y queda indefinido** (2.5→48). **π es una huella de la geometría que cuajó, no una ley previa al universo.**
- **CS069 (el frente cuántico)** — ¿la dirección emerge de una SUPERPOSICIÓN de grafos, no de uno definido? 96 corridas blindadas, 4 configuraciones: ❌ todas indistinguibles. **La ruta cuántica llega, por su cuenta, a la misma pared que la clásica** — el arco toca, sin proponérselo, la misma frontera donde la física real tampoco tiene teoría cerrada (gravedad cuántica).
- **CS070 (semilla primordial)** — ¿una asimetría mínima inicial (tipo violación CP) se amplifica donde la sopa simétrica no pudo? ❌ dirección=0.000 en las 96 corridas. La semilla se lava igual que la sopa simétrica.
- **CS071 (histéresis — memoria del propio proceso)** — ¿la propia dinámica de uso fabrica la asimetría (caminos que se refuerzan al transitarlos)? ❌ resultado con histéresis ≈ barajado ≈ sin proceso. Control positivo (sobre retícula ya perfecta) sí funciona, confirmando que el instrumento mide bien cuando hay geometría real.

**Resultado de la fase:** **seis rutas completamente independientes** (localidad, combinación completa, inflación, cuántica, semilla, memoria del proceso) llegan, cada una por su cuenta, a la misma pared.

*Script(s): cs064_sistema_completo.py, cs065_exclusion_pauli.py, cs065b_exclusion_ortogonalizante.py, cs066_localidad_geometrogenesis.py, cs067_gamma_sweep.py, cs068_paso1/2/2b.py, cs069_quantum_graph.py, cs070_*.py, cs071_histeresis.py*

## Conclusión de la Parte 1: la dimensión es contingente, y π es una huella

Después de 71 experimentos, la conclusión que se ganó el derecho a sostenerse —porque sobrevivió a controles honestos, cazó sus propios falsos positivos, y se repitió por rutas independientes— es esta:

**No existe ningún ingrediente local —ni fuerza, ni espín como marco, ni masa, ni vértice de tres cuerpos, ni localidad del tejido, ni combinación de todo junto, ni inflación, ni superposición cuántica, ni semilla de asimetría, ni memoria del proceso— que elija específicamente que el espacio tenga 3 dimensiones y sea plano.** El 3D en el que vivimos es, en palabras del propio equipo, **contingente**: una posibilidad entre muchas que persistió, no la única que podía pasar.

Analogía de cierre: es como revisar, uno por uno, a todos los sospechosos de una lista larga —el arma, el móvil, la oportunidad, el testigo, la huella— y descubrir que ninguno, individualmente ni combinado, explica por qué el crimen ocurrió así y no de otra manera. Eso no significa que no haya explicación; significa que no está en la lista de sospechosos que se tenía. Y hay una consecuencia añadida, quizás la más profunda del tramo: si la geometría misma no está garantizada de antemano, entonces números que se suelen tratar como verdades matemáticas eternas —como π— tampoco son leyes previas al universo. Son **huellas** que quedan grabadas en la geometría particular que efectivamente se formó.

---

# PARTE 2 — CS072: el "positivo mayor" — y la auditoría que lo puso en duda

## Resumen ejecutivo de esta parte

CS072 es el experimento que el propio registro del proyecto llama "el resultado positivo mayor de todo el proyecto". La promesa: a partir de una condición inicial mínima (S>0, "hubo una relación") y dejando que las fuerzas fundamentales actúen juntas mientras el universo se enfría, deberían aparecer solas las partículas del Modelo Estándar (protones, neutrones, la proporción correcta entre ellos), los primeros átomos, y con ellos el tiempo, el espacio y la dimensión.

Esa promesa se cumplió en gran parte, y el mecanismo que lo logra es real y está bien construido en varias piezas verificables. **Pero esta auditoría también encontró — porque el propio equipo del proyecto lo encontró primero, y lo dejó por escrito — que tres de los cuatro números más celebrados del experimento (bariones, la proporción protón:neutrón, y la dimensión) fueron, en su versión final, artefactos o números puestos a mano disfrazados de resultado emergente.** Esto no es una sospecha de este informe: es el veredicto de una auditoría interna del propio proyecto (verificado directamente en el código como parte de esta consolidación), confirmado días después en una revisión de sesión que llegó a escribir literalmente "Motor de partículas CS072 caído".

Este documento cuenta la historia completa, sin esconder nada — tal como pide el método del proyecto.

## 1. Cómo se llegó hasta acá

El arco CS052-CS068 (Parte 1) terminó en un negativo elegante: ningún ingrediente aislado selecciona la dimensión. La conclusión llevó a un diagnóstico: quizás la dimensión no la decide ninguna pieza sola, sino el proceso completo — todos los ingredientes actuando juntos, como una torta real, no probando la harina, el huevo y el horno por separado. CS072 es exactamente ese experimento.

*Script(s): el diagnóstico está resumido en REGISTRO_ACTUALIZACION_CS069-CS073.md*

## 2. El diseño: "la habitación entera encendida a la vez"

Imaginá una niebla tibia que llena todo el universo, perfectamente pareja. El Big Bang no fue perfectamente simétrico: una porción minúscula quedó ligeramente más fría que el resto — esa diferencia infinitesimal (S>0, "hubo una relación") es la única semilla permitida. No se regala nada más: ni posiciones, ni partículas, ni cuadrícula.

La regla central (anti-Shannon): **si algo cuenta como descubrimiento, apagar la pieza que supuestamente lo produce tiene que destruirlo.** Si apagás la fuerza fuerte y los protones siguen apareciendo igual, el contador estaba midiendo otra cosa.

*Script(s): DISENO_CS072_experimento_unico_CS.md*

## 3. Primer intento: exploración del sustrato (v5–v8), y por qué no alcanzó

Antes de partículas de verdad, se probó una versión más simple: solo un campo de temperatura repartido en parcelas. La pregunta: ¿puede aparecer una noción de cercanía de la nada, sin dibujar antes una cuadrícula?

- **v5** — ordenar por temperatura. La diferencia se diluye en vez de crecer; un defecto de diseño fuerza artificialmente 1 dimensión.
- **v6/v7** — conectar por "roce" real, con memoria. Aparece un "nodo estrella" (todos conectados con uno solo) — demasiado simple para ser espacio real.
- **v8** — sin ninguna estructura previa, la gravedad heredada dejó de tener sentido (no tenía sobre qué operar). CC reportó el problema en vez de improvisar una solución.

**Conclusión:** el sustrato de "solo temperatura + roce" no alcanzaba — hacía falta meter partículas de verdad. Esta etapa fue abandonada como camino principal.

*Script(s): cs072_v5_nucleo.py, cs072_v6_nucleo.py, cs072_v7_banda_persistencia.py, cs072_v8_nucleo.py*

## 4. EL FALSO POSITIVO Y SU RETRACTACIÓN (18-jul-2026)

Se construyó un primer motor con partículas sobre un campo térmico con memoria. Al correrlo, "aparecían" 9 bariones — celebrado como el primer hallazgo positivo del arco.

**El director hizo una objeción por pura lógica, antes de correr ningún código de verificación:** "lo que puede surgir de la condición gradiente+expansión no puede ser materia, porque si lo fuera, dependería del resto de los actores." Un barión ES, por definición, tres quarks pegados por la fuerza fuerte — si apagar esa fuerza no cambia el conteo, no son bariones de verdad.

Se verificó corriendo el código: apagando literalmente todas las fuerzas, el conteo se quedó igual: 9. La causa real: el contador agrupaba quarks térmicamente cercanos (como agrupar personas paradas cerca en una foto), no quarks efectivamente ligados. Era **agrupamiento térmico disfrazado de física de partículas**.

**Se retractó formalmente el hallazgo**, con esta nota de honestidad del propio equipo: *"celebré un hallazgo que era artefacto... el director aplicó la prueba correcta por lógica: si no depende de los actores, no es real."* El rediseño hizo que el contador leyera específicamente la ligadura de la fuerza fuerte (una matriz separada, `Bq`), nunca el campo térmico general. Repetida la prueba: apagar el confinamiento SÍ llevó los bariones a cero, en cuatro escalas, sin excepción. El motor fue declarado **ADMISIBLE**.

*Script(s): ADJUDICACION_CS072_corrida_completa_MATERIA_EMERGE_CS.md (retractado), ADJUDICACION_CS072_RETRACTACION_materia_es_artefacto_CS.md, cs072_motor_fuerzas.py, ADJUDICACION_CS072_motor_fuerzas_ADMISIBLE_CS.md*

## 5. La reconstrucción: de 4 piezas a un motor completo (19-jul-2026)

Sobre la base admisible, el motor se completó pieza por pieza, cada una probada por separado: `cs072_v9_umbral_escala.py` (probó umbrales de asimetría y memoria del enlace; el hallazgo colateral más relevante fue detectar que un resultado anterior de "materia frágil" probablemente era solo falta de pasos de estabilización) y `cs072_v10_motor_fuerzas_escala.py` (repitió la admisibilidad en 4 escalas, documentando con transparencia que sólo el confinamiento decidía el resultado — las otras fuerzas aún no tenían "con qué" discriminar).

*Script(s): cs072_v9_umbral_escala.py, cs072_v10_motor_fuerzas_escala.py, paquete cs072_modulos/*

## 6. El mecanismo final, explicado con una analogía

Pensá el motor final como una fiesta enorme que empieza carísima de energía y se va enfriando poco a poco, como una olla que se saca del fuego y se deja reposar.

**El elenco:** quarks (con "color" y carga), antimateria, electrones y positrones — sin posición todavía, solo identidad y una "temperatura" propia (que viaja con cada partícula, para poder probar que el orden en la lista no decide nada).

**Las fuerzas, cada una con su trabajo:**

- **La fuerza fuerte** pega quark con quark de color complementario, formando tríos (bariones).
- **La aniquilación** hace que partícula y antipartícula se destruyan al encontrarse — sobrevive sólo el excedente (por cada mil millones de pares aniquilados, sobrevive uno de más: todo el universo visible).
- **La fuerza débil (freeze-out)** da la proporción protón:neutrón. Mientras hace calor, se convierten libremente; al enfriarse, esa conversión se "congela" — como sillas musicales: cuando la música para, algunos quedan atrapados en la silla que tenían.
- **El electromagnetismo** sólo actúa muy frío: pega electrón a protón, nace el hidrógeno neutro — el momento en que "se hace la luz".
- **La fuerza fuerte residual** junta protones y neutrones de a 2+2 para el helio.
- **La gravedad** sólo entra al final, sobre átomos ya formados, ligándolos según masa y densidad — de esa red se lee el "espacio" y la dimensión.
- **Las fluctuaciones** son la rugosidad inicial que le da a la gravedad algo que preferir.

**El reloj:** todo corre bajo una ley de enfriamiento (como una olla que se enfría rápido al principio y cada vez más despacio). Cada fuerza se enciende sólo bajo su propio umbral de temperatura. **El tiempo mismo** no se mide contando pasos de simulación — se mide contando transiciones irreversibles (cada átomo neutro formado es un evento que no se deshace). El tiempo "nace" con el primer átomo.

*Script(s): cs072_modulos/nucleo.py, catalogo.py, estado.py, freeze_out.py, piezas/p02_gravedad.py, p03_fuerte.py, p04_em.py, p08_aniquilacion.py, p23_fluctuaciones.py, p24_tiempo.py*

## 7. Los números que salieron (y lo que significan, en simple)

| Observable | Resultado | Qué significa en simple |
|---|---|---|
| Bariones | 100 | Tríos de quarks (protones+neutrones) sobrevivientes |
| Razón protón:neutrón | 7.1 | El freeze-out real observado también es ~7:1 |
| Hidrógeno | 50 | Átomos completos formados al enfriarse lo suficiente |
| Helio | 25 | Núcleos de 2+2 formados por la fuerte residual |
| Tiempo emergente | 75 | H+He: número de "tics" irreversibles |
| Dimensión acoplada a átomos reales | 2.05 | Medida sólo sobre la red de átomos que se formaron |
| Dimensión "ciega" (ensemble) | 2.77 | Misma medición, sin depender de si hubo física real |
| Barrido D=1→5 | 1.0, 2.24, 2.77, 3.33, 3.41 | Sugería que la dimensión es variable libre, no fija |

**Esta lectura es exactamente la que la siguiente sección pone en duda con evidencia dura.**

*Script(s): verificar_cs072.py, cs072_modulos/proceso_sucesivo.py*

## 8. LA AUDITORÍA DE PARÁMETROS A MANO — el veredicto de honestidad (verificado en esta consolidación)

El mismo día del resultado positivo (20-jul-2026, madrugada), siguiendo la misma regla que ya había cazado el falso positivo anterior — "correr el motor real, no creerle a las adjudicaciones" — se hizo una auditoría exhaustiva (`AUDITORIA_MOTOR_CS072_parametros_a_mano_CS.md`), con cuatro revisores independientes más una verificación ejecutada. El veredicto:

> *"El motor NO deriva la materia del Modelo Estándar de la física de sus fuerzas. Los observables 'estrella' son en su mayoría estequiometría del catálogo o parámetros de entrada copiados a la salida, indistinguibles de su NULL."*

**Lo que se encontró (verificado línea por línea en esta misma consolidación, no sólo leído):**

1. **El conteo de bariones no depende de ninguna fuerza — es aritmética pura.** `bariones = número_de_quarks / 3`, exacto, siempre. Apagando la fuerza fuerte, el conteo sigue igual. La prueba de admisibilidad "apagar confinamiento → 0 bariones" (celebrada en la sección 4) **dejó de cumplirse en esta versión final**, sin que nadie lo notara hasta esta auditoría.
2. **La proporción 7:1 protón:neutrón estaba puesta a mano.** Confirmado, literalmente, en `freeze_out.py`, línea 20: `h = tasa_expansion*20.0` — un comentario en el código dice *"escala la expansión del motor a la competencia física"*, pero ese `20.0` no tiene ninguna derivación: fue elegido porque, con la tasa por defecto (0.02), da exactamente 7.1. **El motor real, contando protones y neutrones formados, da 1:1 exacto.**
3. **La dimensión "acoplada" (2.05) copia el número de entrada, no lo mide.** Con distintos valores de entrada D, la salida los sigue de cerca (D=2→1.83, D=3→2.05, D=4→2.82). Comparando la versión "real" contra un control totalmente al azar, ambos dan prácticamente lo mismo (2.77 vs 2.80) — la física no mueve la aguja.
4. **Un cuarto elemento (el factor 50.0 del enfriamiento)** resultó sospechoso (coincide de forma demasiado perfecta con el umbral de recombinación) pero inofensivo: variar el número de pasos no cambia los conteos finales.

**Lo que SÍ se sostiene, verificado de forma genuina:**

- **Apagar electromagnetismo → cero hidrógeno.** El hidrógeno depende de verdad del EM.
- **Apagar la fuerza fuerte → cero helio.** El helio depende de verdad de la fuerte residual.

Estas dos son las únicas dependencias limpias que el motor final cumple.

**Por qué esto es grave, y por qué es también un ejemplo de honestidad bien hecha:** el propio proyecto se autodenunció. Un documento posterior (`REVISION_COMPLETA_SESION_CS.md`, 23-jul-2026, verificado en esta consolidación) resume el episodio así: *"el 7:1 venía de una fórmula analítica con coeficiente h = tasa·20.0, con el 20.0 elegido para dar 7:1 en el default. Puesto por CS (el asistente), no por CC; certificado falsamente como 'emergente' en tres adjudicaciones previas. Retractado."* Ese mismo documento anota, en su tabla de balance del arco, bajo resultados NEGATIVOS: **"Motor de partículas CS072 caído"**.

*Script(s): AUDITORIA_MOTOR_CS072_parametros_a_mano_CS.md, cs072_modulos/freeze_out.py (línea 20), cs072_modulos/nucleo.py (_detecta_trios), cs072_modulos/proceso_sucesivo.py (dimension_acoplada), REVISION_COMPLETA_SESION_CS.md*

## 9. La verificación paralela — qué confirmó realmente, y qué no

Un script separado (`proceso_sucesivo.py` + `verificar_cs072.py`) reprodujo todos los valores adjudicados sin discrepancia. Esto confirma que **no hubo error de transcripción** — los números reportados son exactamente los que el motor produce, de forma repetible. Pero esta verificación llama a las mismas funciones que la auditoría de parámetros ya examinó por dentro — confirma que el motor es **consistente consigo mismo**, no que sea **honesto en su física**. Eso es justo lo que la auditoría puso en duda.

*Script(s): verificar_cs072.py, cs072_modulos/proceso_sucesivo.py*

## 10. Entonces, ¿qué queda en pie, y por qué se lo sigue llamando "el positivo mayor"?

Conviene separar tres cosas que se mezclaron bajo el mismo nombre:

**(a) Lo que NO se sostiene, según la auditoría del propio proyecto:** que la materia (bariones, 7:1) y la dimensión (~3) emergen limpiamente sin parámetros impuestos.

**(b) Lo que SÍ se sostiene, genuinamente:** que el hidrógeno depende de verdad del electromagnetismo, y el helio de la fuerza fuerte residual. Más modesto que "el Modelo Estándar completo emerge", pero real — y es justamente la parte que usa CS073 (Parte 3 de este documento) como su punto de partida.

**(c) Lo que sí se sostiene, y de hecho se vuelve MÁS fuerte:** los negativos del arco topológico (Parte 1) — que ningún ingrediente local selecciona la dimensión — resultan "inmunes al fraude del 7:1" (palabras de un documento posterior): esos resultados nunca dependieron del motor de partículas que resultó fabricado. La caída de CS072 deja más limpio, no más sucio, el negativo anterior.

**Una contradicción que este documento deja señalada, sin resolver por su cuenta:** la fila oficial de la tabla maestra del proyecto sigue llamando a CS072 "el positivo mayor de todo el proyecto" — esa adjudicación se escribió horas ANTES de que la auditoría encontrara el problema, y no se localizó un documento posterior que actualice formalmente esa fila. Existe, hoy, una etiqueta oficial en contradicción directa con un hallazgo posterior del propio proyecto. Cualquier lectura de este archivo debería tener presente esa salvedad.

Lo que sí sobrevive entero: el **diseño** de CS072 —arrancar de S>0, dejar que todas las fuerzas actúen juntas, con el enfriamiento como reloj, probando cada pieza con su propio apagado— es el marco correcto en el que el proyecto siguió trabajando, con la obligación aprendida de no repetir el mismo error: verificar cada número contra su apagado antes de celebrarlo.

*Script(s): REGISTRO_ACTUALIZACION_CS069-CS073.md, REVISION_COMPLETA_SESION_CS.md, TRASPASO_sesion_Cosmogenesis.md*

---

# PARTE 3 — CS073: de los átomos a la primera estrella (el trabajo de esta semana)

## El punto de partida (con la salvedad heredada de la Parte 2)

CS072 dejó probado, de forma genuinamente verificada, que el hidrógeno depende del electromagnetismo y el helio de la fuerza fuerte — esas dos piezas sobrevivieron la auditoría. Lo que NO sobrevivió (Parte 2) fue el conteo total de bariones, la proporción 7:1, y la dimensión ≈3. Para CS073 esto importa poco en la práctica: el sustrato que usa (una población de átomos de hidrógeno y helio, con su masa real) se apoya en la parte de CS072 que sí sobrevivió — no en el conteo total de bariones ni en la dimensión, que son las partes fabricadas. Aun así, vale la pena tenerlo presente: el terreno es sólido en la pieza que se usa, pero viene de un experimento que en otras partes tuvo que retractar resultados celebrados.

Con esa salvedad hecha, la pregunta que abre CS073 es la obvia: **¿esos átomos, dejados a su suerte, pueden juntarse por gravedad hasta formar una estrella?**

## Primer intento (19-jul-2026): el prototipo de juguete

Un modelo "de juguete" combinando gravedad + expansión + enfriamiento + materia oscura + agrupamiento por cercanía + el criterio de la masa de Jeans (el punto en que el peso de una nube le gana a la presión que la sostiene). El resultado, sin maquillar, fue un **negativo con matices**: un error de sincronización hizo que se midiera casi al principio (antes de que la gravedad actuara), y varios números de entrada estaban puestos "a mano". Se detectó concentración de masa real pero modesta.

*Script(s): cs073_ley_escala.py, catalogo.py*

## El giro conceptual: "dos gravedades"

El hallazgo más importante de esta etapa fue conceptual: **hay dos regímenes distintos de gravedad**. La **relacional-cuántica** (la que ya existía) conecta por cercanía térmica antes de que exista un espacio real — "todo está cerca de todo", no puede formar estructura. La **general-clásica** —la que forma estrellas— necesita que ya exista un espacio con posiciones y masa acumulada en el tiempo. Este hallazgo se repitió por cuatro caminos independientes. Faltaba construir el segundo régimen desde cero.

*Script(s): p_gravedad_general.py (gravedad de N-cuerpos real, con posiciones y velocidades que se actualizan)*

## El "puente": la coherencia relacional sí siembra estructura

La pregunta: ¿el sustrato relacional (la "malla causal" de quién nació relacionado con quién) tiene estructura espacial real que la gravedad pueda usar? Respuesta: **sí, pero sólo si se despliega dinámicamente** — acomodar los átomos como una red de resortes y dejar que evolucione con la misma expansión del resto del motor. Los átomos conectados por la malla causal forman grupos ligados por gravedad muchas más veces que una versión barajada al azar.

Momento de honestidad: el primer número (z=10.26) resultó inflado por un defecto de programación (apilaba partículas en las esquinas). Corregido, bajó pero **se sostuvo: z=6.92**.

*Script(s): p_semilla_causal.py, cs073_cierre_holistico.py*

## Cambio de herramienta: de un motor casero a Phantom

El integrador "casero" empezó a perder energía de forma creciente al seguir el colapso de cerca. En vez de forzarlo, se adoptó **Phantom**, un programa de gravedad y fluidos usado por astrónomos de verdad, validado a fondo antes de confiar en él (una órbita de dos cuerpos conservó la energía con precisión 12 órdenes de magnitud mejor que lo exigido). La traducción a las condiciones de Phantom se auditó para que nada se colara por la puerta de atrás.

*Script(s): fase1_traducir_a_phantom.py*

## Contaminaciones cazadas (y corregidas) en el camino

- El acomodo de partículas apilaba algunas en las esquinas — corregido con un "rebote" en los bordes.
- Aparecieron pares de partículas casi pegadas al escalar — se investigó y resultó ser un rasgo real de la estructura (presente igual en el control, no distorsiona la comparación).
- Un error de conservación de energía llevó a diagnosticar, paso a paso, si era el arranque o falta de resolución — se confirmó que era **falta de resolución**, no un error de física ni de arranque.

## El freno: cuántos átomos hacen falta para confiar en el resultado

Para que un colapso cuente como real, la región que colapsa necesita **al menos el doble de la cantidad de "vecinos" que usa el motor de física** (regla estándar del campo, Bate & Burkert 1997). Con los números reales de Phantom, eso son unos **116 átomos mínimo** por grumo colapsante. Calcular cuántos átomos totales hacían falta resultó inestable (2.900 → 6.500 con un solo dato más). Se aceptó un **rango honesto** (3.000-7.000), no un número preciso.

## Retomando en esta sesión: la sorpresa de encontrar trabajo ya hecho

Al pedir "corré Phantom a ver qué pasa", apareció que la Mac de trabajo **es la misma máquina** que el colaborador técnico había usado semanas atrás para correr Phantom hasta 8.550 átomos — corridas sin analizar. Al revisarlas: la versión real colapsaba mucho más rápido y profundo que la de control (pico de densidad hasta 40 veces mayor), en dos configuraciones de velocidad independientes.

## El sumidero pragmático (y un hallazgo honesto sobre la masa de Jeans)

Para representar "ya nació una estrella" sin seguir el colapso al infinito, se usan **partículas sumidero**: cuando la densidad cruza un umbral, esa zona se reemplaza por un punto de masa. Al intentar fijar ese umbral con el criterio físico riguroso (masa de Jeans, con la temperatura real del motor), apareció un hallazgo incómodo pero honesto: **con lo fría que está la nube, la masa de Jeans ya es más chica que la masa de un solo átomo** — un umbral literal inútil. Se optó por un umbral pragmático: el punto donde el programa empieza a tener problemas numéricos — la misma clase de decisión, con la misma honestidad, que otras partes del proyecto ya habían tomado para separar "aparataje del modelo estándar" de "la Teoría misma".

*Script(s): cosmog.in (rho_crit_cgs, icreate_sinks)*

## El resultado robusto: z=48.69

Con el sumidero puesto, se armó una batería (1 corrida real + 8 de control). Dos tropiezos propios, reconocidos sin maquillar: se gastaron ~4 horas pensando que se probaban tres nubes distintas cuando la función que las genera es **determinista** (se repitió la misma nube tres veces); y se generaron condiciones sin velocidad inicial (todas quietas), lo cual dispara un error de diagnóstico ya conocido y olvidado — corregido con el campo de velocidad turbulento ya existente.

Corregidos ambos: **la real acumuló 2.124 unidades de masa en sumideros, contra un promedio de 720 en las ocho de control (poca variación entre ellas) — z=48,69.** Las nueve corridas completaron sin ningún error de conservación.

*Script(s): campo_velocidad_turbulento.py, bateria_n2000/*

Matiz honesto: la de control TAMBIÉN forma sumideros — no es "a la real le nace una estrella y al azar no le nace nada", es "a las dos les nace algo, pero a la real casi tres veces más". Y esta medida (masa en sumideros) es una adaptación de lo prometido originalmente (número de estructuras que cruzan Jeans), necesaria porque esa medida no sobrevivía sin sumideros — todavía espera autorización explícita del director para contar como veredicto final.

## El control positivo con la receta de Phantom (sin nada nuestro)

Pregunta separada de Alexis: ¿el propio Phantom, con su receta de libro de texto, forma una estrella? Sirve para separar "¿el problema es nuestro sustrato?" de "¿el problema es simular un colapso estelar en general?".

- Con los valores por defecto, **la nube no colapsó** ni con 5 veces el tiempo de caída libre.
- Con una nube 8 veces más masiva, sí hubo colapso real, pero se detuvo en una meseta.
- Se descubrió la causa: la simulación corría en modo "adiabático" (el calor de compresión queda atrapado), no "isotérmico" como nuestro propio experimento. Corregido, el colapso avanzó mucho más.
- Se probó el **refinamiento automático de Phantom (APR)** — el "modo liviano" que Alexis recordaba: en vez de simular todo con alta resolución, el programa agrega partículas sólo donde la nube se comprime. Funcionó (refinó de 6.595 a más de 140.000 partículas justo en la zona crítica), y la densidad llegó a estar a un 66% del umbral de formación de estrella, subiendo exponencialmente — pero se rompió justo en ese instante, por un problema numérico bien conocido del campo (la velocidad de colapso, al final, es tan alta que el método de cálculo no converge).

**La lectura honesta:** no es velocidad superlumínica ni una falla de la física — la gravedad acelera el reloj del colapso a medida que la densidad sube, y en el instante final ese reloj se acelera tanto que el método numérico no logra seguirle el paso. Es un problema conocido y difícil del campo, no un error propio.

Se intentó la misma técnica sobre el propio experimento: apuntada a los 24 grumos a la vez, se saturó; apuntada a un solo sumidero ya nacido, no se rompió, pero se volvió tan lenta en la práctica que no aportó crecimiento adicional dentro de lo razonable.

*Script(s): setup_sphereinbox.f90 (esfera de Bonnor-Ebert), compilaciones con ISOTHERMAL=yes y APR=yes*

## Estado actual del arco CS073

- **Lo firme:** z=48.69 (real forma casi 3× más masa en sumideros que el azar, limpio y sólido) es el mejor resultado del arco a la fecha.
- **Lo pendiente:** la declaración rigurosa de "cruza Jeans con resolución confiable" todavía no está — hace falta escalar a 3.000-7.000 átomos (costoso).
- **Lo aprendido de lado:** incluso el código estándar del campo tropieza con la misma dificultad numérica en el instante final del colapso — sitúa la dificultad como un problema conocido del área, no un defecto exclusivo de este proyecto.
- **Sin resolver:** si "masa total en sumideros" cuenta como observable válido de cierre, o si hace falta re-diseñar la medición formal — decisión que espera autorización explícita del director.

---

# PARTE 4 — La rama paralela: ¿el modelo predice los números exactos de nuestro universo?

## Por qué se abrió esta rama

Tras CS072, con el proceso ya probado (con la salvedad de la Parte 2), se abrió una pregunta más arriesgada: **¿el modelo da sólo la relación y el proceso, o además "sabe" reproducir números concretos del universo real** (el 4,9% de materia visible, la razón 7 protones por 1 neutrón, un protón 1836 veces más pesado que un electrón), sin que se los pongamos a mano? La regla de la casa exige que estos números se usen sólo como examen de la salida, nunca como ingrediente de entrada.

Esta rama usa nombres propios distintos de la numeración CS porque es un experimento aparte, con su propio librito de reglas: **CF** (asimetría/masa), **Enfoque 5** (energía-exergía-entropía, 30 experimentos), **cs074 A/B/C** (el motor completo de una vez) y **BATERIA_FUNDAMENTOS** (F1-F4).

## Cronología

| Fecha | Qué pasó |
|---|---|
| 23-jul-2026 | Protocolo CF firmado. Corre CF-1. Se prepara CF-2. Se redacta el informe de fondo sobre la masa real (Higgs vs. ligadura QCD). |
| 23/24-jul-2026 | Corre CF-4 (masa=ligadura) — falla por diseño (números a mano), no por física. Se diseña CF-4b. |
| 24-jul-2026 | Con el "muro de la masa" fresco, se diseña BATERIA_FUNDAMENTOS (24 experimentos) para blindar los pilares que sí funcionaban. |
| 24/25-jul-2026 | Corre Enfoque 5 completo (30 experimentos, con tres "arreglos" de instrumento a mitad de camino). |
| 26-jul-2026 | Corre y se adjudica cs074 A/B/C. |
| 27-jul-2026, 01:30 | Se escribe el traspaso que resume todo y deja "listo para correr" el barrido fino de banda estrecha. |
| 27-jul-2026, madrugada | El barrido fino en efecto se lanza (prueba primero). |
| 29-jul-2026 | Corre la versión completa (61 horas, 2000 configuraciones). |

## Batería CF — ¿la asimetría sobrevive, y la masa es "pegamento"?

**CF-1 — ¿la diferencia persiste si el universo se expande más rápido de lo que se difumina?** Analogía: una gota de tinta en agua caliente que se mueve mucho se borra; si el vaso se estira más rápido de lo que la tinta se esparce, la mancha queda "congelada". Resultado: ✅ sí, con mecanismo claro — sin expansión, casi se borra (3-3,4% de rastro); con expansión creciente, se congela cada vez más (62% ya a 0,1 de expansión relativa, 96-99% a expansión alta), siempre separado del azar (z entre 4,9 y 7,8). Caveat: la malla probada era gruesa, no señala el punto exacto donde arranca.

**CF-2 — ¿enfriar es lo mismo que expandir?** ✅ PASS: la relación entre el gradiente físico y el comóvil cae de forma monótona y fuerte al expandirse (0,16→0,0003), mientras el control con densidad fija no cae.

**CF-4 y CF-4b — el hilo de la masa que quedó sin cerrar.** La pregunta: casi todo el peso de un protón es el "costo de pegamento" de mantener sus quarks juntos, no la suma de sus piezas — ¿el modelo tiene un régimen donde ese costo domine? CF-4 falló, pero el propio equipo encontró por qué: tres números estaban puestos a mano, nunca barreados, y con esos valores fijos el "pegamento" nunca podía ganarle al "potencial" pasara lo que pasara — no era un veredicto de física, era un error de diseño. CF-4b se diseñó como corrección exacta (convertir esos números en un barrido), quedó completo y pre-registrado el 24-jul, **pero no se encontró en disco ningún resultado ni adjudicación** — sólo la instrucción. Dos experimentos que dependían de él quedaron bloqueados. El proyecto giró hacia BATERIA_FUNDAMENTOS mientras tanto.

*Script(s): cs074_rcruz.py, CF_bateria/cf2_estiramiento_densidad.py, INSTRUCCION_CF-4b_masa_ligadura_barrido_acoplamiento_PARA_CC_y_Grok.md (sin ejecución encontrada)*

## Enfoque 5 — energía, exergía y entropía (30 experimentos)

**Tres palabras para traducir de una vez:** *energía* = la cantidad total, se conserva siempre. *Entropía* = el desorden, cuánto se ha mezclado todo. *Exergía* ("energía útil") = no toda la energía sirve para algo — una batería cargada tiene energía útil, una agotada tiene la misma energía total (como calor disperso) pero ya no sirve.

**La pregunta central:** ¿el modelo reproduce el reparto real de materia del universo (4,9% visible, o 31,5% contando materia oscura)?

**Tres arreglos de instrumento necesarios antes de confiar en los 30 experimentos:** (1) el "reordenar" no debía inventar energía de la nada — corregido para que sólo reorganice lo que ya existe. (2) el ruido se desbocaba en sistemas grandes (inflaba resultados hasta 123% de más) — corregido repartiéndolo según la duración de la simulación. (3) la regla común para medir exergía resultó ciega quando el control es "barajar el campo" — es como preguntar "¿está en el mismo orden?" contando sólo cartas rojas y negras: barajar nunca cambia esa cuenta.

**El resultado clave (E5.3-5), verificado personalmente abriendo el archivo de datos crudos:** **las celdas donde la eficiencia cae casi exacto sobre el 4,9% real tienen z=0** (no significa nada); **la única celda con señal real (z=2,4) da 27,25%**, lejos del 31,5% real. **Donde hay un número que se parece al real, no hay señal; donde hay señal, el número no se parece.** Buena noticia paradójica: si hubiera coincidido, sería sospechoso de forzado.

**Lo que sí quedó firme (11 de 30):** la contabilidad de energía cuadra (correlación -0,9999 entre energía útil y desorden); muerte térmica ≠ Nada, comprobado en 240/240 casos (energía total ~1,0, energía útil cae a cero); las estructuras grandes se congelan primero (16/16); no hay "enfriador tramposo" escondido; funciona también en dos dimensiones; la expansión es la única vía real de rescatar energía útil.

**Lo que falló, honestamente (4 de 30):** atar la energía útil al ritmo de enfriamiento no funcionó por la vía probada; las dos formas de medir energía útil coinciden en forma pero no en escala.

**El resto (15 de 30): señal parcial, sin cerrar** — incluyendo un resultado invertido marcado para revisar, no ignorar. El adjudicador se negó a "sellar" el arco completo con casi la mitad sin resolver.

*Script(s): BATERIA_ENFOQUE5/E5_*/, por ejemplo E5_3_5_falsacion_externa/E5_3_5_motor.py; _ruido_calibrado.py, _observables_homologadas.py*

## cs074 A/B/C — el arco holístico de energía

A diferencia de Enfoque 5 (piezas sueltas), corre el motor completo de punta a punta.

**Experimento A — ¿demasiada asimetría inicial destruye la estructura?** ✅ sí, confirmado, con tres regímenes: meseta (hasta ≈0,5, ~77% de masa ligada, plano) → fragmentación (0,9-2,3, más grumos, más chicos) → colapso (≳3,8, desaparece todo). La causa es **mecánica, no energética** (verificado con presupuesto de energía infinito: curva idéntica).

**Experimento B — ¿el enfriamiento H2 afecta la estructura?** ❌ negativo limpio y confiable (1980 corridas, z entre -0,11 y -0,14, plano). Quedan anotadas dos posibles razones honestas (tiempo corto, o presión térmica dominante) sin forzar una conclusión.

**Experimento C — relación y proceso sí, números físicos no.** La fracción de materia da z=1,37 (no significativo, verificado personalmente). La razón p:n y la masa protón/electrón no son evaluables (son entrada, no salida — la masa protón/electrón del juguete da 18, no 1836, porque usa masas "desnudas" sin el ~99% de ligadura). En cambio, **6 de 7 relaciones/procesos sí sostienen con control real**: contabilidad de energía, costo de ligadura causal, muerte térmica, expansión rescata estructura, el hallazgo A, y gravedad indispensable (quitarla hace caer la estructura de 60,7% a 2,0%).

*Script(s): cs074A_asimetria_techo.py, cs074B_fragmentacion_enfriamiento.py, cs074C_limite_modelo.py, cs074_energia_holistica.py*

## BATERIA_FUNDAMENTOS (F1 a F4, 24 experimentos)

Diseñada para blindar, con rigor extra (3 verificaciones independientes por experimento: control de azar propio, segundo método que debería coincidir, y auditoría en disco por alguien que no escribió el código), los cuatro pilares que sí venían funcionando: persistencia (F1), congelamiento por expansión (F2), enfriar=expandir (F3), y si la caída de densidad tiene efecto propio (F4). Hay evidencia de que al menos varios (Enfoque 1 y 2) se corrieron, pero **no se encontró una adjudicación consolidada** que cierre el veredicto de esta batería — queda registrado honestamente como diseñada y parcialmente corrida, sin síntesis final localizada.

*Script(s): BATERIA_FUNDAMENTOS/F1_1_forma_magnitud/F1_1_motor.py, F2_1_rstar_fino/F2_1_motor.py, F2_6_null_alternativo/F2_6_null_secuencia.py (entre otros)*

## El experimento pendiente: el barrido fino de banda estrecha (cs074-D)

**La pregunta:** ¿la estructura vive en una banda angosta y especial del espacio de configuraciones — o aparece en cualquier lado? Diseño: 2000 configuraciones × 12 semillas, control por barajado de densidades.

**Lo que en realidad pasó:** aunque el traspaso lo dejó anotado como "listo para correr, no corrido", se lanzó esa misma madrugada, y la versión completa (61 horas) terminó el 29-jul. **El resultado no fue "vive en banda" ni "no vive en banda" — fue que el instrumento resultó demasiado débil para contestar nada:** de 1647 configuraciones válidas, ninguna (0%) superó el umbral acordado. Causa raíz: el control barajaba densidades entre posiciones que ya eran al azar desde el principio — como barajar el color de fichas de un tablero donde las casillas ya estaban puestas al azar, casi no cambia nada.

**Dicho con honestidad: esto no significa "no vive en banda estrecha" — significa "no se midió".** Un hallazgo lateral sí quedó firme: existe un "piso" de tasa de expansión (≈0,0026) por debajo del cual el modelo no forma ni un solo átomo. El experimento sigue sin cerrar, con un arreglo identificado (usar semillas causales, ya probado en otra parte del proyecto) pendiente de la decisión del director.

*Script(s): cs074D_barrido_fino_banda.py, cs074_energia_holistica.py, p_gravedad_general.py*

## Balance honesto de esta rama

El modelo genera **relación y proceso genuinos** (energía se conserva, muerte térmica distinguible de la nada, expansión rescata estructura, gravedad indispensable, y produjo un hallazgo propio no buscado — demasiado desorden inicial destruye la organización, en tres regímenes bien caracterizados) **pero no reproduce, sin que se lo pidamos, los números concretos de nuestro universo particular** — y donde se pudo medir de verdad (la fracción de materia), el negativo es limpio: z=1,37, indistinguible del azar.

Esto no es un fracaso — es información real sobre qué *tipo* de cosa es el modelo. Un motor de coches explica por qué un auto se mueve; no te dice de qué color se va a pintar la carrocería. Que la Cosmosemiótica no "adivine" el 4,9% no invalida que sí explique, con mecanismo y control real, por qué hay algo que persiste, se organiza y se conserva en primer lugar.

---

# PARTE 5 — Tres hilos exploratorios

## HILO A — El "rastro de hormigas" sobre el motor (cs075)

**La pregunta:** cuando una hormiga encuentra comida, deja un rastro químico que otras leen y refuerzan — eso es **estigmergia**, comunicación indirecta por una marca en el medio. cs075 preguntaba si un **campo continuo** (una superficie líquida, no una grilla) funcionando como ese rastro compartido entre los 23 elementos del modelo, se parece en algo a la física real.

**Cómo evolucionó (v1→v4):** v1 reconstruyó los 23 agentes sobre física real; se encontró y corrigió un error de cálculo serio (una fórmula que crecía sobre sí misma sin freno, como una bola de nieve, hasta números sin sentido). v2 corrigió un reloj mal calibrado (una escala inventada en vez de la ya existente, que había inflado el cómputo necesario de 36 pasos a **21 millones** — casi 35 horas por configuración). v3 con el reloj arreglado, 160 combinaciones probadas, apareció un **borde muy nítido**: por debajo de cierta intensidad, el inventario nunca se completa; justo arriba, siempre. v4 con dos controles: (N1) el borde no se mueve al variar umbrales internos — es real. (N2) sorprendentemente, barajar la cadena causal dio el **mismo resultado** que la cadena real — el número usado como prueba de causalidad no distingue una secuencia con sentido de una armada al azar.

**Resultado: parcial, honesto.** La arquitectura funciona y el borde es reproducible, pero la tesis central (que el orden de despertar de los 23 elementos refleja causalidad real) **queda sin evidencia a favor, no refutada** — pendiente otra métrica y un control totalmente sin física que aún no se hizo.

*Script(s): Campo_Continuo_Estigmergico/cs075_23_sobre_fisica.py, cs075_pruebas_23_sobre_fisica.py, cs075_barrido_v3.py, cs075_N1_sensibilidad_umbrales.py, cs075_N2_puertas_permutadas.py*

### El momento de honestidad autocrítica

Durante casi dos semanas, el asistente de diseño entendió mal la tarea de fondo: Alexis había pedido tomar los 23 elementos **que YA estaban probados y funcionando** (el motor `cs072_motor_23.py`) y envolver cada uno en un agente paralelo — como pedir "tomen las piezas de un reloj que ya arma bien la hora y háganlas trabajar todas a la vez en vez de en fila" — y en cambio se construyó un reloj nuevo, sin ninguna pieza vieja, que por diseño nunca podía dar la hora.

El colaborador técnico lo documentó en dos reportes: uno (escrito por el propio asistente, a pedido de Alexis, para reportarlo formalmente) lista **nueve errores factuales verificables** — negar un archivo que se había leído minutos antes, inventar un elemento "#24" inexistente, clasificar mal el elemento "#18" contradiciendo el propio código citado dos frases antes, e inventar una escala de temperatura que infló una estimación de cómputo por un **factor de 585.124 veces**. El otro (escrito por CC) resume el patrón: *"CS diseña sin comprobar primero qué ya existe, y cuando el diseño se aleja de lo ya probado, no lo señala — sigue adelante."*

Lo valioso no es el error, sino la corrección de proceso que produjo: de ahí en más, el asistente de diseño pasa a ser **revisor**, no quien escribe instrucciones que se ejecutan sin que Alexis las vea primero.

*Script(s)/documentos: REPORTE_FALLA_SISTEMATICA_asistente_CS075.md, REPORTE_fallas_sistematicas_CS_cs075.md*

## HILO B — El motor de 7 etapas (motor_1a7)

**Qué es:** un tubo único que encadena dos laboratorios que trabajaban por separado (expansión del universo, y aparición del orden/átomos/masa), con una libreta compartida (un JSON de estado) que cada etapa lee y actualiza, como una posta de relevos.

**Las 7 etapas:** (1-2) campo continuo + expansión; (3-4) estiramiento/densidad ∝1/a³; (5) orden sin masa; (6) átomo (E3); (7) masa por linaje (E4).

**Producción vs. smoke test:** el smoke test (prueba rápida) pasó las 7 etapas limpio. La corrida de producción mostró algo más delicado: etapas 1-6 pasaron limpio, pero la etapa 7 sólo aparece en verde porque, **después de que el juez originalmente registrado (v5) falló** (0.40 contra el umbral 0.55), se lo cambió por un juez nuevo (v6) que mide la "masa" usando **las mismas variables** con las que decide si hay linaje causal — una tautología, no una prueba independiente. Documentado con una cronología en disco que muestra que el criterio se cambió **después de ver el resultado negativo**, y con un addendum del propio director: *"la sustitución posterior por v6... no cuenta como PASS del protocolo... Etapa 7 permanece ABIERTA."*

**Resultado:** etapas 1-6 sólidas. Etapa 7 abierta, sin cerrar — falta un observable de masa independiente, o renunciar honestamente a llamarlo "masa" y quedarse con "linaje".

*Script(s): motor_1a7/pipeline.py, estado.py, resultados/estado_1a7_produccion.json, estado_1a7_smoke.json*

## HILO C — Notas conceptuales sueltas

Cinco ideas y marcos teóricos, no experimentos con número de resultado — todas marcadas como borradores, sin integrar todavía:

1. **Los campos físicos como "terreno" donde algo persiste.** Los campos (eléctrico, térmico, gravitacional) no son "de donde sale" la información — son el terreno donde una diferencia puede sostenerse en el tiempo, a cada escala. Se apoya en un dato externo real (Pinotsis & Miller, 2022): la memoria del cerebro vive en su campo eléctrico, no en las neuronas — a lo que este proyecto había llegado por su cuenta antes de leer ese paper.
2. **Persistir = tener historia = sostener una relación.** No es una idea nueva, es mostrar que tres formas de decir lo mismo ya estaban dispersas en la teoría; esta nota las junta.
3. **Dos capas distintas al hablar de comunicación.** Separa que la identidad de un organismo no depende del material que la sostiene, de que comunicarse directamente sí requiere compartir el mismo tipo de campo físico. Incluye una idea especulativa, marcada explícitamente como no probada (configuraciones del entorno influyendo en actividad cerebral sin dispositivos) — con Alexis dejando explícito que no adhiere al transhumanismo, se incluye sólo por coherencia lógica.
4-5. **La masa real según la física de partículas.** Revisión de literatura externa: el Higgs sólo explica ~1% de la masa de un protón; el otro ~99% viene de la energía que mantiene atados a los quarks. Como pensar que el peso de una pelota de fútbol viene del cuero, cuando viene sobre todo del aire a presión adentro. Importa porque señala una tensión real con el Hilo B, que ubica la masa recién después del átomo — la física real dice lo contrario.

## Cómo se conectan estos tres hilos

Hilo A y Hilo B comparten la misma lección de fondo: en ambos, el equipo se dio cuenta a tiempo de que se había movido el criterio de éxito después de ver que algo fallaba, lo documentó sin maquillaje, y dejó el resultado abierto en vez de forzarlo. Hilo C conecta con Hilo B: la revisión de literatura sobre la masa es la evidencia externa que sustenta por qué la etapa 7 quedó abierta.

**Estado de cada hilo:** A — detenido, no abandonado, a la espera de autorización para los siguientes pasos. B — etapas 1-6 sólidas y reutilizables, etapa 7 abierta explícitamente. C — ideas en desarrollo conceptual, ninguna integrada todavía a la teoría canónica.

---

# PARTE 6 — Síntesis final: dónde está parado el proyecto hoy

**Lo que está firme, sin salvedades:**

- La dimensión del espacio es **contingente** — ningún ingrediente local probado (fuerza, marco, masa, vértice de tres cuerpos, localidad, combinación completa, inflación, cuántica, semilla, memoria del proceso) la selecciona. Seis rutas independientes llegan a la misma conclusión.
- π es una **huella** de la geometría que cuajó, no una ley matemática previa al universo — medido y falsable.
- El hidrógeno depende genuinamente del electromagnetismo, y el helio de la fuerza fuerte residual (la parte de CS072 que sobrevivió su propia auditoría).
- El modelo genera relación y proceso genuinos (conservación de energía, muerte térmica distinguible de la nada, expansión rescata estructura, gravedad indispensable) verificados con controles reales, en varias baterías independientes.
- La coherencia relacional (la malla causal desplegada dinámicamente) sí siembra estructura real que la gravedad aprovecha — z=6.92 en el modelo casero, y esta misma semana, z=48.69 con Phantom real y sumideros.

**Lo que cayó, y hay que decirlo cada vez que se mencione CS072:**

- El conteo de bariones, la proporción 7:1, y la dimensión ≈3 de CS072 fueron, en su versión final, aritmética disfrazada o números puestos a mano — auditado y documentado por el propio proyecto.

**Lo que quedó abierto, sin forzar un cierre que no corresponde:**

- CF-4b (masa como ligadura): diseñado, sin resultado encontrado.
- BATERIA_FUNDAMENTOS: parcialmente corrida, sin adjudicación consolidada localizada.
- El barrido fino de banda estrecha (cs074-D): corrido, pero con un instrumento demasiado débil para contestar la pregunta — no es un negativo, es una no-medición.
- cs075 (campo estigmérgico): arquitectura y borde reales, tesis central sin evidencia a favor ni en contra.
- motor_1a7, etapa 7 (masa por linaje): explícitamente abierta por el propio protocolo del proyecto.
- CS073 (la primera estrella): el mejor resultado hasta la fecha (z=48.69) espera la decisión de si cuenta como el observable de cierre, y la escala rigurosa (3.000-7.000 átomos) todavía no se corrió.

**El hilo que atraviesa todo el proyecto, de punta a punta:** cada vez que algo pareció funcionar demasiado bien, alguien —Alexis por lógica pura, CC auditando su propio código, o el propio proceso de verificación— lo cuestionó, y varias veces esa sospecha tenía razón. Eso no es un defecto de Cosmogénesis. Es, literalmente, el método funcionando como se diseñó.
