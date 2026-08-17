# DISEÑO CS066 — LA LOCALIDAD PRIMERO: geometrogénesis. ¿Emerge un espacio con "lejos" — y recién ahí, direcciones?
## CS066 — se añade al motor un costo que castiga los enlaces NO-LOCALES (geometrogénesis). Pregunta: ¿el blob mundo-pequeño de CS064 se convierte en un tejido local con "lejos"; y sobre ese tejido, cuántos ejes emergen?

**Diseña:** CS · **Fecha:** 10-jul-2026 · **A codear/ejecutar:** CC · **Endosado por:** Alexis (eligió "el tejido primero")
**Auditoría que lo funda (hecha por CS sobre los CSV de CS064, no sobre prosa):** el brazo `completo` de CS064
es un blob ULTRA-mundo-pequeño. Diámetro del grafo ≈ 3.3–3.9 saltos para N=1500→3500, y casi NO crece al
duplicar N. Referencias: espacio 3D local → diam ~ N^(1/3) ≈ 11–15; blob ~ log N ≈ 7–8. El nuestro (~3.5) está
por DEBAJO incluso del mundo-pequeño clásico. d_s se infla con N (4.8→5.6), holonomía alta (0.86→0.95),
gigante 91%. **Conclusión auditada: todo está a 3-4 saltos de todo — no hay "lejos".**

---

## 0. EL GIRO CONCEPTUAL (por qué CS065/065b no podían funcionar)
CS065 y CS065b buscaron el ingrediente en el nivel de las DIRECCIONES (fuerzas de orientación: repulsión,
exclusión). Ambos murieron. La auditoría de CS064 explica por qué NO podían funcionar: **una dirección solo
significa algo si moverte hacia allá te ALEJA de donde estabas.** En un blob donde cualquier nodo está a 3-4
saltos de cualquier otro, no hay lejos ni cerca — todas las direcciones son la misma. El colapso-a-1-eje no era
una falla de la orientación; era el SÍNTOMA de que no existe un espacio local contra el cual distinguir
direcciones. Estábamos peleando un piso más arriba del problema.
**Lo que faltó no es un ingrediente que ELIJA direcciones (eso ya se falsificó dos veces). Es la condición que
hace que la DISTANCIA signifique algo: la LOCALIDAD.** Primero el tejido; después, lo que vive en él.

## 1. EL INGREDIENTE: costo de no-localidad (geometrogénesis) — por qué es honesto y no una perilla
- **Es física real, con nombre en la literatura:** en Quantum Graphity (Konopka-Markopoulou-Severini 2008, que
  ya citamos), la geometría no está dada — EMERGE por "geometrogénesis": el estado de alta temperatura es un
  grafo denso SIN geometría; al enfriar, una transición de fase PODA los enlaces de largo alcance y el grafo se
  vuelve LOCAL — y ahí aparece la geometría. El ingrediente no es una fuerza que elige; es un COSTO que penaliza
  la no-localidad.
- **Lo omitimos:** nuestro motor tuvo expansión y enfriamiento, pero NUNCA un costo que hiciera caros los
  atajos de largo alcance. Por eso el grafo se quedó como maraña densa con puentes por todas partes. La
  localidad es una condición que el sustrato real (el universo) tiene y el nuestro no tenía. Meterla es
  corregir una omisión, no agregar una perilla oportunista.
- **Por qué ataca el problema de fondo:** si penalizar la no-localidad convierte el blob en un tejido donde
  "lejos" existe (diámetro que CRECE con N, no plano), ENTONCES —y solo entonces— la pregunta de las direcciones
  tiene un fondo geométrico donde inscribirse. La hipótesis: sobre un tejido local, la co-evolución del marco
  (la exaptación que SÍ confirmamos en CS064) podría sostener varios ejes sin colapsar, porque ahora hay un
  espacio real que orientar.
- **CRÍTICO — no se calibra:** el costo de no-localidad NO se ajusta para que salgan 3 ejes ni un diámetro
  concreto. Su magnitud se sortea en rango amplio junto al resto (o sale de la escala del motor). Lo que se mide
  es si EXISTE una transición (barriendo el costo) donde el blob se vuelve local — no un valor elegido a mano.
  **G-NO-CALIBRAR.**

## 2. QUÉ CAMBIA vs CS064 (un solo actor nuevo; la orientación vuelve a CS064, sin exclusión)
Se parte del motor de CS064 (NO el de CS065/065b — la exclusión murió, se retira). Se añade UNA regla en la
formación/persistencia de enlaces:
- **CS064 (sin localidad):** un enlace nace cuando un mediador conecta dos nodos y sobrevive el paso — sin
  importar cuán "lejos" estén en el grafo emergente.
- **CS066 (con costo de no-localidad):** un enlace nuevo entre nodos que ya están conectados por un camino corto
  es BARATO; un enlace que crea un atajo de largo alcance (conecta dos regiones distantes) es CARO y tiende a no
  sobrevivir el enfriamiento. Formalmente: un costo que crece con la distancia-de-grafo preexistente entre los
  extremos del enlace candidato; el enfriamiento poda preferentemente los enlaces de alto costo (los atajos).
  Esto es la geometrogénesis: al enfriar, los puentes de largo alcance se rompen y queda tejido local.
- La orientación/marco co-evoluciona EXACTAMENTE como en CS064 (inercia de alineamiento, sin exclusión — la
  exclusión ya se falsificó). No se toca el juez de direcciones; se cambia el TEJIDO sobre el que actúa.

## 3. LOS BRAZOS (anti-Shannon)
- **local (real):** motor CS064 + costo de no-localidad + marco co-evolucionando.
- **sin_local (=CS064):** CONTROL CENTRAL — sin costo de no-localidad. Debe reproducir el blob (diam~3.5, no
  crece con N, ~1.2 ejes). Si `local` no se separa de esto, la localidad no hizo nada.
- **local_barajado:** el costo de no-localidad se aplica a enlaces AL AZAR (no a los atajos reales de largo
  alcance) — misma "cantidad" de poda, colocada donde no corresponde. Aísla si importa PODAR LOS ATAJOS o solo
  quitar enlaces genéricamente. Cuerda anti-Shannon (la que mató a la exclusión: real≈barajado ⇒ no específico).
- **local_marco_congelado:** costo de no-localidad SÍ, pero orientación congelada. Separa las dos capas: ¿la
  localidad sola crea el tejido (diámetro crece) aunque no haya direcciones? Debe dar tejido local con 0 ejes.
  Es el control que distingue "hay espacio" de "hay direcciones en el espacio".

## 4. LO QUE SE MIDE — el orden importa (primero el tejido, luego las direcciones)
**Nivel 1 — ¿emergió un espacio local? (la pregunta primaria, nueva):**
- **diámetro vs N:** ¿CRECE con N (señal de espacio local: diam ~ N^(1/d)) o se queda plano ~3.5 (blob)? ESTA es
  la medida decisiva de CS066. Se corre N∈{1500,2500,3500} y se AJUSTA el exponente: si diam ~ N^(1/d) con d≈2-3,
  hay tejido de dimensión d; si diam ~ log N, sigue siendo blob.
- **d_s espectral:** en un tejido local debe ESTABILIZARSE con N (no inflarse como en el blob, donde subía
  4.8→5.6). Un d_s que deja de crecer y se asienta en un valor finito es la firma de geometría real.
- **grado medio / clustering:** un tejido local tiene grado acotado y clustering local; el blob tiene grado alto
  y atajos. Se mide la caída de atajos al aplicar el costo.

**Nivel 2 — sobre el tejido local, ¿cuántas direcciones? (la pregunta de siempre, ahora con fondo):**
- **n_ejes** (espectro del tensor de orientación) — pero ahora leído SOBRE un tejido que ya tiene "lejos". La
  hipótesis es que aquí el colapso-a-1 podría romperse SIN necesidad de exclusión, porque el marco tiene un
  espacio real que orientar.
- **δ/Gromov y holonomía** — ¿el tejido local es plano (δ/escala→0) y con marco coherente (holonomía→trivial)?

## 5. SALIDAS PRE-INSCRITAS (blind — se leen contra esto, NO se acomodan)
- **(A) EMERGE ESPACIO LOCAL Y CON DIRECCIONES:** con el costo, el diámetro CRECE con N (diam~N^(1/d)), d_s se
  estabiliza, Y n_ejes sube por encima del colapso-a-1 (≥2), Y `local` > `local_barajado` (especificidad: podar
  los atajos, no podar cualquier cosa). ⇒ la LOCALIDAD era lo que faltaba; sobre un tejido local las direcciones
  emergen de la co-evolución del marco (la exaptación de CS064) sin necesidad de exclusión. Hallazgo grande.
  (Si d≈3 y n_ejes≈3 sin que el andamiaje lo imponga — G-NO-TOPADO — es enorme; no se espera ni se fuerza.)
- **(B) EMERGE ESPACIO LOCAL PERO SIGUE COLAPSANDO A 1 EJE:** el diámetro crece (¡hay tejido local!) pero
  n_ejes sigue ~1. ⇒ la localidad crea el espacio pero NO basta para múltiples direcciones — el colapso-a-1 es
  más profundo que la falta de localidad, y FALTA todavía otro ingrediente (aquí sí volvería a tener sentido
  buscar el anti-colapso, ahora sobre un fondo local). Hallazgo fuerte: separa "hay espacio" de "hay 3D".
- **(C) NO EMERGE TEJIDO LOCAL:** con el costo, el diámetro sigue plano (~log N o menos) — el blob no se deshace.
  ⇒ el costo de no-localidad, tal como se implementó, no genera geometrogénesis en este sustrato; o el sustrato
  relacional no soporta localidad por su cuenta. Negativo limpio; reorienta hacia por qué no cuaja.
- **(D) PLACEBO:** `local ≈ local_barajado` — podar atajos da igual que podar cualquier enlace. ⇒ el efecto (si
  lo hay) no es la localidad específica, es adelgazar el grafo. Honestidad obliga a decirlo.
- **(E) DEPENDE DE N:** el tejido local aparece solo por encima de cierto N. ⇒ geometrogénesis como transición
  de escala; el régimen de números enormes es donde vive.

## 6. GUARDIANES
- **G-NO-CALIBRAR:** el costo de no-localidad se sortea/barre; NUNCA se fija para producir un diámetro o un nº de
  ejes objetivo. Lo que se busca es si EXISTE una transición, no un valor.
- **G-TEJIDO-ANTES-QUE-EJES:** el Nivel 1 (¿hay espacio local?) se adjudica ANTES de mirar el Nivel 2 (¿cuántos
  ejes?). No se lee el nº de ejes de un brazo que no produjo tejido local — sería contar direcciones en un blob.
- **G-NO-TOPADO:** si emergen "3 ejes", verificar que no es porque D_max o el grado los topa ahí (D_max≥8).
- **G-CONTINUIDAD:** `sin_local` reproduce CS064 (blob diam~3.5, ~1.2 ejes). Si no, el motor cambió — abortar.
- **G-PLACEBO:** `local_barajado` existe para que "podar atajos" no se confunda con "quitar enlaces". La
  separación local vs barajado es la prueba de especificidad — sin ella, un (A) no vale (lección de CS065b).
- **G-SMOKE-ANTES:** smoke (N=1000, ~10 parches, 4 brazos) validando que (i) sin_local da el blob, (ii) el costo
  corre sin fragmentar todo (si poda de más → gas de nodos, hay que ver el rango), (iii) la medida diámetro-vs-N
  discrimina blob de tejido en los calibradores. No correr la tanda hasta que el smoke pase y CS lo adjudique.

## 7. LO QUE NO HACE / LÍMITES
- No prueba que la dimensión sea 3 — prueba (1) si emerge un espacio LOCAL con "lejos", y (2) sobre él, cuántas
  direcciones. El éxito primario es que el tejido se vuelva local (diámetro crece con N); las direcciones son la
  pregunta secundaria, ahora bien planteada.
- Es un análogo relacional de la geometrogénesis, no la mecánica completa de Quantum Graphity. Fidelidad en la
  ESTRUCTURA (costo que poda atajos de largo alcance → tejido local al enfriar), no en su Hamiltoniano exacto.
- Un actor por vez: la exclusión murió y se retiró; el sector oscuro (antes CS066, ahora futuro) espera a que el
  tejido local esté resuelto — no tiene sentido preguntar si la materia oscura hace de andamiaje si aún no hay
  un espacio local que andamiar. Este reordenamiento es consecuencia directa de la auditoría de CS064.

---
**PRE-REGISTRO — para el acta:** el diseño y sus salidas §5 se fijan antes de correr. La razón de CS066 es una
auditoría de CS064 hecha por CS sobre los CSV (diámetro ~3.5 que no crece con N = blob ultra-mundo-pequeño),
NO una corazonada. El costo de no-localidad no se calibra. Si sale (C) o (D), la localidad-tal-como-se-implementó
no era el ingrediente, y se dice sin duelo. Si sale (B), aprendemos que el espacio y las direcciones son
problemas separados —y eso reordena el arco—. El azar y la física fija juzgan.

— CS. CS065/065b buscaron direcciones donde no había espacio. La auditoría de CS064 mostró el blob sin "lejos".
CS066 pregunta lo anterior: ¿emerge primero el tejido? Si sí, recién ahí las direcciones tienen dónde vivir. No
sé qué dará — y ese es el punto.
