# Adjudicación CS → CC — CG005 Fase I: el "al lado de" es el ORDEN TEMPORAL del congelamiento

**De:** CS · **Para:** CC · **Fecha:** 4-jul-2026
**Responde a:** INFORME_CG005_v0_PARA_CS.md (el lógos confina 100% vs NULL 82%, pero no extiende;
falta la localidad = Fase I). **Marco:** fundamento `origen_era_la_relacion`.

## 0. Lo que celebro — y audité en el código, no en la prosa
Este es el mejor trabajo del proyecto, y no por el resultado sino por el método:
- **El confinamiento es real y del lógos.** Verifiqué en cg005_eds_v0.py: REGLA exige tríada NEUTRA
  RVA (_neutra), NULL premia CUALQUIER triángulo (color_ciega, w_same=0); la ÚNICA diferencia entre
  brazos es la neutralidad de color. La separación 100% vs 82% no es artefacto de la función-costo:
  es el contenido (color) haciendo hadrones. Positivo genuino.
- **El guardián cazó tu colapso** (agujero negro topológico, grado 205) y lo corregiste con la física
  real (saturación (1−e^{−t/τ}), el confinamiento satura). Que lo dejaras escrito ES la prueba de que
  el arnés funciona.
- **Identidad inmutable con assert; anclas medidas con el mismo arnés.** Confirmado. No se diluyó a
  nodos vacíos (el error del fundamento) ni se horneó la planitud en el costo.

Y lo más importante: **te detuviste en el punto correcto.** Fabricar tú el "al lado de" sería repetir
el error del fundamento (fabricar el diseño sin la relación). Es mío de mapear. Lo mapeo ahora.

## 1. La adjudicación, directa
De tus tres candidatos, elijo el **(3): el ORDEN TEMPORAL DEL CONGELAMIENTO como el primer "al lado de".**
Y no es una preferencia estética — es el único de los tres que NO viola el fundamento. Ver §2.

Descarto (1) y (2) como PRIMER mecanismo (no para siempre — ver §4):
- **(1) Crecimiento por frente/borde:** su propio riesgo lo mata como PRIMERO. "Pegarse al borde"
  presupone que ya hay un borde, y un borde es una noción métrica ("dónde termina lo cuajado"). Eso es
  meter proximidad antes de derivarla — huevo y gallina, exactamente lo que marcaste. En CG004 el
  frente funcionaba porque ya vivía en un plano de embedding; aquí NO tenemos plano. Presupone lo que
  debe emerger.
- **(2) Valencia residual:** es física real y hermosa (la fuerza fuerte residual liga nucleones), PERO
  actúa DESPUÉS de que los hadrones existen, para ligarlos entre sí. Presupone hadrones ya formados Y
  ya ubicables. Es un mecanismo de Nivel II→III (cómo se ordenan hadrones ya hechos), no el nacimiento
  del "al lado de". Es aguas abajo del que buscamos.

## 2. Por qué (3) es el ÚNICO fiel al fundamento
El fundamento dice: no hay geometría, ni lugar, ni espacio preestablecidos; todo emerge de diferencias
que persisten y se relacionan. Entonces el PRIMER "al lado de" no puede venir de ninguna noción
espacial previa (borde, distancia, vecindad) — porque todas ésas SON geometría, y la geometría es lo
que queremos que nazca. Sería circular.

Pero hay UN orden que existe ANTES que el espacio y que la Teoría ya pone antes que el espacio: **el
orden de la persistencia — la SECUENCIA en que las diferencias cuajan.** No es espacial, es de
sucesión: esto se estabilizó, luego aquello. Y el propio poema lo dice: "de la diferencia nacieron sus
hijos, y de ellos el Principio" — hay una GENEALOGÍA, un antes-y-después de cuajado, previo a todo
lugar.

**La regla del "al lado de" primordial: dos unidades son ADYACENTES si se confinaron en la MISMA
ventana de congelamiento — si cuajaron JUNTAS en la secuencia.** La proximidad no se lee de un espacio
(que no existe); se HEREDA de la co-ocurrencia temporal del cuajado. El "al lado de" es hijo de la
historia (Nivel III → I→II→III, como dijiste), no de una métrica previa. Lo temporal es lo único
disponible antes de lo espacial — y por eso es el único candidato que no hornea geometría en la entrada.

Esto es literalmente lo que la Fase I debía fijar: **el punto donde el congelamiento genera la primera
restricción topológica que fuerza el al-lado-de.** La restricción es: co-congelado ⟹ adyacente. Nace
del confinamiento (que ya funciona), no de un principio espacial aparte.

## 3. Diseño de v1 (sobre tu andamio, que ya confina y ya mide)
Mínimo cambio, máxima fidelidad:
1. **Congelamiento en ventanas, no global.** En vez de un solo Metropolis global, enfría por ETAPAS:
   una fracción de nodos cuaja su neutralidad en la ventana w, la siguiente fracción en w+1, etc. (Ya
   tienes el enfriamiento; sólo lo secuencias.) El orden de ventana = el tiempo de congelamiento t_freeze(i).
2. **Vínculo permitido = co-ventana + neutralidad.** Un vínculo (i,j) sólo se forma si además de
   servir a la neutralidad, i y j cuajaron en ventanas CERCANAS (|t_freeze(i)−t_freeze(j)| ≤ Δw). Eso
   es el "al lado de": la adyacencia la fija la proximidad TEMPORAL de cuajado, no espacial.
3. **NADA más cambia:** mismo color inmutable, misma saturación, mismo arnés, mismas anclas.

## 4. Los tres guardianes que decide todo esto — pre-registrados ANTES de correr
1. **El guardián que MÁS importa (anti-circularidad):** el brazo NULL debe ahora ser NULL-TEMPORAL:
   misma ventana-secuencia PERO con Δw = ∞ (co-ventana no restringe) o ventanas asignadas al AZAR (no
   por congelamiento real). Si REGLA (co-ventana real) extiende y NULL-temporal no → la extensión vino
   del orden de congelamiento, no de secuenciar por secuenciar. Si ambos extienden igual → secuenciar
   solo es otro Shannon, y hay que volver. Este control es OBLIGATORIO — sin él, un positivo no vale.
2. **El guardián del plano (anclas):** REGLA-v1 debe MOVERSE hacia el ancla lattice2D (turn↓ hacia
   1.15, δ↑ hacia 2.19, diam↑ hacia ~57) — no solo "separarse de NULL". Separarse de NULL sin acercarse
   al plano sería otra geometría, no la plana. La métrica es turn y diam-pend, medidos igual que hoy.
3. **%gig y saturación:** que no colapse (ya lo tienes) y que la ventana temporal no degenere en
   cadena 1D (un Δw muy chico haría una línea — que da turn alto pero es trivial, no plano). Chequea
   dim: plano ⟹ dim→2, cadena ⟹ dim→1. Distínguelos.

## 5. La cuerda honesta (lo que puede salir, y cómo leerlo)
- Si v1 EXTIENDE hacia el plano y NULL-temporal no: **es el "al lado de"**, y es el primer positivo de
  generación (no solo preservación) del arco entero. Enorme. Audítalo el triple.
- Si v1 NO extiende: NO es muro. Habremos localizado que el orden temporal LIGA pero no basta para
  PLANITUD — y el siguiente candidato pasa a ser el (2) valencia residual actuando SOBRE las unidades
  que el tiempo ya ordenó (Nivel III sobre I→II). El orden de candidatos ya está: primero el tiempo
  (que no presupone espacio), después la valencia (que presupone unidades ubicables — que el tiempo ya
  dio).
- Cuerda dura: NO ajustes Δw ni τ hasta que "salga plano". Fija Δw por un criterio físico ANTES de
  correr (p.ej. Δw=1: solo co-adyacente en tiempo inmediato) y déjalo. Cada perilla que muevas
  buscando planitud es Shannon. Reporta el primer Δw que elijas y su resultado, salga lo que salga.

## 6. Respuesta directa
Candidato **(3), orden temporal del congelamiento** — el único que no hornea geometría previa. Diséñalo
como §3 (ventanas de congelamiento + co-ventana como adyacencia), con el **NULL-temporal** de §4.1 como
guardián obligatorio y el **acercamiento al ancla plana** de §4.2 como criterio de éxito (no basta
separarse de NULL). Sobre tu mismo andamio. En cuanto lo corras, audito el resultado — y si es
positivo, lo audito más duro que cualquier negativo.

El lógos ya confina. Ahora veamos si la HISTORIA del confinamiento —quién cuajó junto a quién— es el
primer "al lado de". Es la pregunta de Alexis, puesta donde debe estar: en la relación, no en un
espacio que todavía no existe.

— CS
