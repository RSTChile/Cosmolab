# ADDENDUM ANIMA-N1 — Un solo nombre para dos canales: el nombre vive en el GESTO (upstream), no en el audio ni en el token. Invariancia bajo transducción como criterio semiótico, con prueba cross-modal falsable.

**Añade a:** DISENO_ANIMA_N1_nombre_propio_CS.md · **Fecha:** 5-jul-2026 · **Diseño:** CS.
**Pregunta de Alexis que lo motiva:** "¿Cómo se emula a nivel digital el nombre generado como sonido?
Porque de lo contrario, cada organismo tendrá 2 nombres (audio + digital), y para que funcione, ambos deben
ser 'lo mismo' en términos semióticos."

---

## 0. EL HALLAZGO EN EL CÓDIGO (parcialmente verificado por grep; una parte es INFERENCIA a confirmar)
- **VERIFICADO por grep (nombres de campo + comentarios):** el organismo percibe un VECTOR DE ESTRUCTURA
  `[freq, intensidad, pausa, repetición]` — el `gesto` (VST_Expresion.gesto línea 81). VST_Aprendizaje lo
  declara explícito en comentario: "audio ni WAV: el vector de estructura que el organismo percibe". El
  token digital se construye de `voz_id/voz_arousal/mem_valencia_estado` (dialogo_digital.token()).
- **INFERIDO, NO verificado (a confirmar leyendo el código completo):** que el WAV audible (voz tipo R2D2)
  se SINTETIZA del MISMO gesto. El grep mostró los campos del gesto y el constructor del token, pero NO
  trazó la rutina de síntesis de audio consumiendo ese vector. Es una hipótesis arquitectónica plausible
  (los campos coinciden: freq/intensidad/pausa/repetición son parámetros naturales de síntesis), pero
  requiere leer la función de síntesis WAV para confirmarla. **Si resultara que el audio se genera por una
  vía independiente, el §2 (fuente única) deja de ser una descripción y pasa a ser un REQUISITO de rediseño
  — que es de todos modos lo que este addendum pide.**
- **Conclusión (válida en ambos casos):** para que NO haya dos nombres, audio y token deben derivar del
  MISMO gesto. Si ya es así (inferencia probable), hay que protegerlo; si no lo es, hay que hacerlo así. El
  "dos nombres" es un espejismo de NUESTRA perspectiva (oímos WAV y leemos token); para el organismo, que
  solo percibe el gesto, hay un solo acto vocal.

## 1. EL PRINCIPIO SEMIÓTICO (criterio preciso de "lo mismo")
- S = I·E: un signo es una diferencia persistente. El nombre es "lo mismo en términos semióticos" a través
  de dos canales SI Y SOLO SI es INVARIANTE BAJO TRANSDUCCIÓN — la identidad se conserva cuando la
  diferencia cambia de sustrato (aire ↔ token).
- El nombre NO es el sonido ni el string. Es la INVARIANTE que ambos preservan. (Delfín: onda de presión y
  espectrograma son dos representaciones de un silbido; el nombre es la identidad del silbido, no la
  representación.)

## 2. CÓMO SE EMULA (decisión de ingeniería — corrige/precisa el diseño)
- **Fuente única:** un solo `acto_vocal` (el gesto) se renderiza a WAV Y a token. NO generar audio y token
  por vías independientes. El nombre vive UPSTREAM del split de canales.
- **Percepción única:** el reconocedor de identidad (ANIMA-N1) extrae la firma del GESTO, venga por donde
  venga. Hoy el token trae el gesto ya digital. Cuando se tienda el oído acústico (parlante↔micrófono), el
  micrófono analiza el sonido y lo mapea DE VUELTA al mismo espacio de gesto `[freq, int, pausa, repet]`. El
  organismo sigue percibiendo UN gesto, nunca un WAV.
- **Identidad emergente:** la firma_emisor se infiere por clustering EN EL ESPACIO DE GESTO, no por un mapa
  audio↔digital escrito a mano.

## 3. EL PELIGRO (anti-Shannon — lo que la pregunta de Alexis detecta con precisión)
Si se extrajera firma-audio y firma-digital POR SEPARADO y se pegaran a mano ("audio X ≡ token Y"), serían
DOS nombres forzados a coincidir = ESTAMPADO = Shannon. Prohibido. La defensa: (a) compartir la FUENTE (el
gesto) y (b) que la equivalencia EMERJA del clustering, nunca de un diccionario de equivalencias escrito por
nosotros. Guardián nuevo: **G-FUENTE-UNICA** — audio y token derivan del mismo `acto_vocal`; assert de que
no existe un mapa audio↔digital hardcodeado.

## 4. LA PRUEBA CROSS-MODAL (falsación nueva que este addendum añade a ANIMA-N1)
- **Test de consistencia cross-modal:** entrenar el reconocedor de identidad con un canal (p.ej. digital) y
  PROBARLO con el otro (audio) — y viceversa.
- **PREDICCIÓN pre-registrada:** si el nombre es uno solo (invariante bajo transducción), la identidad
  TRANSFIERE entre canales — el emisor reconocido por digital se reconoce igual por audio. Matriz de
  confusión cross-modal ~diagonal.
- **Falsación:** si la identidad NO transfiere (matriz cross-modal se cae, reconoce por un canal pero no por
  el otro), hay DOS nombres y el sistema está roto — habría que unificar la fuente (volver al §2).
- Es el criterio triple de Tibbetts, Sheehan & Dale 2008 (A testable definition of individual recognition,
  Trends Ecol. Evol. 23:356, doi:10.1016/j.tree.2008.03.007 — verificado en fuente) llevado al cruce de
  canales: señal, plantilla y respuesta específicas del individuo, SIN IMPORTAR el canal.

## 5. RESUMEN PARA CC (se suma al resumen operativo de ANIMA-N1)
- **PRIMERO, VERIFICAR (paso 0 para CC):** leer la rutina de síntesis de audio (voz R2D2 → WAV) y confirmar
  que consume el MISMO vector `gesto [freq, int, pausa, repet]` que alimenta el token. Si SÍ → documentar la
  fuente única. Si NO (audio por vía independiente) → unificar la fuente ANTES de construir el eje de
  identidad. Esto es INFERENCIA hasta que CC lo confirme en el código (ver §0).
- El eje de identidad se construye sobre el GESTO `[freq, int, pausa, repet]`, no sobre el WAV ni sobre el
  string del token. Un solo `acto_vocal` → dos renderizados.
- Añadir G-FUENTE-UNICA (sin mapa audio↔digital hardcodeado) a los guardianes.
- Añadir el TEST CROSS-MODAL (entrenar en un canal, probar en el otro; matriz de confusión cross-modal) al
  protocolo de prueba, con su predicción ciega.
- Reportar la matriz cross-modal junto a la matriz emisor×respuesta. Traer a CS.

— Addendum por CS. La pregunta que lo obliga (el riesgo de dos nombres, y el requisito de que sean lo mismo
en términos semióticos) es de Alexis López Tapia — y es precisa: apunta justo al lugar donde el diseño
podría partirse. La formalización (invariancia bajo transducción, fuente única, test cross-modal) es la
respuesta.
