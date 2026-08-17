# DISEÑO CS063 — EL VÉRTICE DE 3 CUERPOS GENUINO: no medir el defecto de tríada, sino MOVER los tres marcos JUNTOS. Cerrar el (C)/(D) que CS061 dejó abierto — el experimento que el arco pidió y aún no se hizo de verdad.

**Rama:** Cosmogénesis · **Nº:** CS063 (dimensión técnica: dinámica de 3 cuerpos genuina sobre el marco).
**Diseño:** CS · **Origen:** el CAVEAT de CS061 (verificado en código por CS: el update era pareado, no de
3 cuerpos). **Fecha:** 5-jul-2026. **Estado:** DISEÑO, a codear por CC. **Prioridad: 2 (el caro y decisivo).**
**Base:** cs061_masa_emergente.py + cs059_espin_como_marco.py + adjudicacion_ARCO_CS058-061_CS.md (§4, el
caveat elevado a condición).

---

## 0. LA PREGUNTA, EN UNA LÍNEA — y por qué CS061 NO la respondió
CS061 quiso probar el vértice de 3 puntos (el que CS059 pidió, el del Higgs). Pero CS auditó el código y
encontró que el update del marco era `w = s[i] + align·(media_de_vecinos − s[i])` — **campo medio PAREADO**:
cada nodo se alinea hacia el PROMEDIO de sus vecinos, uno por uno. El "3 cuerpos" estaba solo en la MEDICIÓN
(el defecto de tríada), no en la DINÁMICA. Por eso CS061 es (C) para "inercia amortigua relajación pareada"
pero NO cerró el (D). **CS063 hace lo que falta: un update donde los TRES marcos de una tríada se muevan
JUNTOS, acoplados por un término irreducible de 3 cuerpos — no cada uno hacia la media de sus vecinos.**

## 1. LA DISTINCIÓN EXACTA (2 cuerpos vs 3 cuerpos genuino — el corazón del diseño)
- **2 cuerpos (lo que se hizo hasta ahora, CS059 y CS061):** la energía/regla depende de PARES.
  E = Σ_{(i,j)} f(s_i, s_j). El update de s_i mira a sus vecinos de a uno (o su media). Reducible a la
  adyacencia. Por eso arrastra el confound de longitud de ciclo.
- **3 cuerpos GENUINO (CS063):** la energía/regla tiene un término IRREDUCIBLE sobre TRÍADAS.
  E = Σ_{(i,j,k)∈△} g(s_i, s_j, s_k), donde g NO se factoriza en pares — depende de los tres a la vez (p.ej.
  el defecto de cierre de la tríada, o el volumen orientado del triple de marcos). El update de s_i depende
  CONJUNTAMENTE de (s_j, s_k) de cada tríada a la que pertenece, no de cada uno por separado.
- **La prueba de que es genuino (G-IRREDUCIBLE):** si el término de 3 cuerpos se pudiera escribir como suma
  de términos de 2 cuerpos, es falso 3-cuerpos. Assert: g(s_i,s_j,s_k) ≠ h(s_i,s_j)+h(s_j,s_k)+h(s_i,s_k)
  para la g elegida (verificar analíticamente o numéricamente que el gradiente cruzado ∂³E/∂s_i∂s_j∂s_k ≠ 0).

## 2. EL DISEÑO (sobre el marco de espín, dinámica nueva)
- **Enumerar tríadas cerradas** (triángulos del grafo, o tríadas j-i-k con i central) — la unidad del update.
- **Término de 3 cuerpos:** por cada tríada, una energía g que mide el CIERRE conjunto de los tres marcos
  (el defecto de tríada de CS061 sirve como g — pero ahora ENTRA EN LA DINÁMICA, no solo en la medición). El
  update MINIMIZA ese defecto moviendo los tres marcos A LA VEZ (descenso de gradiente conjunto sobre la
  tríada, o actualización simultánea de los tres).
- **Clave anti-confound:** el update NO puede reducirse a "cada nodo hacia la media de vecinos". Debe ser un
  paso que, congelando pares, aún mueva el marco por el término de tríada. (Implementación sugerida:
  gradiente de E_3cuerpos respecto a cada s_i, sumando SOLO las contribuciones de tríadas, sin el término
  pareado de CS059/61.)
- **Juez:** el Burgers de CG004 (ciego), CON el control de longitud de ciclo que cazó los dos falsos
  positivos previos. La pregunta: ¿AHORA el update de 3 cuerpos selecciona una dimensión donde el pareado no
  pudo, Y colapsa bajo NULL, Y sobrevive al control de longitud de ciclo?

## 3. GUARDIANES (los del arco + el nuevo, crítico)
1. **G-IRREDUCIBLE (nuevo, EL guardián de este experimento):** el término de 3 cuerpos NO es suma de pares.
   Verificado analítica o numéricamente (∂³E ≠ 0). Sin esto, CS063 repite CS061 con otro nombre.
2. **G-UPDATE-CONJUNTO:** el paso mueve los tres marcos de la tríada JUNTOS; assert de que el update de s_i
   incluye el término de tríada y NO se reduce a media-de-vecinos.
3. **G-CONTROL-LONGITUD-DE-CICLO (heredado, obligatorio):** el juez compara a IGUAL longitud de ciclo — el
   control que cazó CS059 y CS061. Sin él, cualquier "selección" es sospechosa de confound.
4. **G-NULL doble:** NULL-tríada (tríadas al azar) y NULL-marco (espines barajados). Debe COLAPSAR.
5. **G-NO-INYECTAR-DIM + G-NO-FORZAR-3D + G-PREDICCIÓN-CIEGA:** representación dim-neutral; éxito = selección
   consistente que colapsa bajo NULL y sobrevive al control de longitud, NUNCA "salió 3D"; predicción antes.

## 4. LOS DESENLACES (pre-escritos — y este SÍ cierra el 3-puntos)
- **(A) El update de 3 cuerpos GENUINO selecciona una dimensión (Burgers discrimina a igual longitud de
  ciclo, colapsa bajo NULL) → el 3-cuerpos era el ingrediente que faltaba.** Cierre mayor del arco por el
  lado que todo señalaba desde CS059. La pared R7 (vértice de 3 puntos) confirmada como la puerta.
- **(B) El 3-cuerpos genuino tampoco selecciona (idéntico al pareado, o colapsa bajo NULL) → NI SIQUIERA el
  vértice de 3 puntos basta.** ESTE es el negativo que CS061 NO tenía derecho a declarar y CS063 sí. Cierra
  el (D) que quedó abierto. Empuja fuerte hacia la hipótesis de fondo: la dimensión es CONTINGENTE, no
  seleccionada por ningún ingrediente local.
- **(C) Selecciona OTRA dimensión (2D/4D) genuinamente → el 3-cuerpos selecciona, pero no el 3D.** Falsación
  acotada nueva, aguas aún más arriba (¿tres generaciones? ¿el número de colores?).

## 5. RELACIÓN CON CS062 Y EL CIERRE DEL ARCO
- **CS062** (barato) relee el negativo de fuerzas con la gravedad correcta. **CS063** (caro) cierra si el
  vértice de 3 puntos —el ingrediente al que TODO el arco apuntó— selecciona o no. Son independientes;
  CS062 primero por costo, pero CS063 es el que da derecho a hablar de la hipótesis de contingencia.
- **Sólo tras CS063:** si sale (B), el arco de eliminación está COMPLETO —fuerza, marco pareado, masa,
  vértice de 3 cuerpos genuino, todos descartados— y la hipótesis "la dimensión es contingente, persistió
  una de muchas posibles" se gana el derecho a ser la conclusión del arco, no una salida. Encaja con la
  imagen de Alexis desde el principio (Pi, el cedazo de geometrías). Si sale (A), es lo contrario: el arco
  encontró el ingrediente. Cualquiera es el final grande.

## 6. RESUMEN OPERATIVO PARA CC
- Sobre el marco de espín: sustituir el update pareado (media de vecinos) por un update de descenso sobre
  E_3cuerpos = Σ_△ defecto_de_tríada, moviendo los tres marcos de cada tríada JUNTOS. NADA de término
  pareado en el update (ese era el error de CS061).
- Verificar G-IRREDUCIBLE (∂³E ≠ 0) ANTES de correr — sin eso no es 3 cuerpos.
- Juez: Burgers de CG004 con control de longitud de ciclo. Brazos: 3cuerpos / 2cuerpos(=CS061) / NULL-tríada
  / NULL-marco. Predicción ciega antes. Éxito nunca = "salió 3D".
- Entregar CSV + figuras (Burgers 3cuerpos vs 2cuerpos por dim a igual longitud; real vs NULL) + informe.
  Traer a CS. Registrar CS063.

— Diseño CS063 por CS. El caveat que lo obliga lo marcó CC honestamente y CS lo verificó en el código y lo
elevó a condición. La distinción 2-cuerpos/3-cuerpos irreducible, el guardián G-IRREDUCIBLE, y las tres
falsaciones, míos. Este es el experimento que el arco pidió desde CS059 y que aún no se hizo de verdad:
puede encontrar el ingrediente, o cerrar la última puerta local y dar derecho a la hipótesis de contingencia.
