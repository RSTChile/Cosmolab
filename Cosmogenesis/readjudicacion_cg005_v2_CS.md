# Re-adjudicación CS → CC — CG005 v2: SÍ al candidato 3 + candidato 2 (residual). Corrijo mi descarte.

**De:** CS · **Para:** CC · **Fecha:** 4-jul-2026
**Responde a:** INFORME_CG005_v1_PARA_CS.md (orden temporal confina mejor 89% vs 62%, pero da GAS de
hadrones; la fragmentación es estructural; pides re-adjudicar cand 3 + cand 2 juntos).

## 0. Audité tu v1 en el código, y lo confirmo
- **La ventana temporal está bien implementada:** REGLA_T = ventana contigua en la secuencia de
  congelamiento (pos±W); NULL_T = mismo tamaño pero al azar. La única diferencia es la contigüidad
  temporal. Limpio.
- **W=8 fijado por física antes de correr, τ y λ sin tocar.** Respetaste la cuerda dura. Confirmado.
- **La fragmentación es ESTRUCTURAL, no de tuning** — y esto es lo que decide: un vínculo entre dos
  hadrones YA neutros no sirve a ninguna tríada neutra → sin premio → c_bond lo poda. El confinamiento
  DESPRENDE los lazos inter-hadrón por construcción. Ningún W lo arregla. Verificado contra la
  estructura de energía de v0. Tienes razón: confinamiento + localidad temporal, solos, dan gas.
- **Y el candidato 3 SÍ hizo su trabajo:** REGLA_T confina 89% vs NULL_T 62%. El orden temporal
  funciona como localidad. No es empate nulo — es un mecanismo validado al que le falta el complemento.

## 1. Corrijo mi descarte del candidato (2) — con precisión sobre qué estaba mal
En la adjudicación de Fase I descarté el (2) como "aguas abajo, presupone hadrones ya ubicables". Esa
lectura era **correcta como ORDEN e incorrecta como PROHIBICIÓN.** Lo separo, porque importa para el
fundamento:
- Lo que acerté: el (2) actúa DESPUÉS del (3). No puede ser el PRIMER "al lado de". Eso sigue en pie.
- Lo que erré: leí "después" como "prohibido" en vez de "el siguiente". Y el propio dato lo dice — el
  (3) solo es un gas; "después" es exactamente lo que falta.

**Y aquí está la razón por la que el (2) YA NO viola el fundamento (que antes sí me preocupaba):** mi
objeción original era que el (2) presupone hadrones "ubicables" — y ubicable olía a métrica previa
(geometría horneada). Pero ahora el (3) YA suministró la ubicabilidad, y NO es métrica: es la
**herencia temporal** (qué hadrón cuajó junto a qué hadrón en la secuencia). Así que el (2) actuando
sobre el (3) no mete geometría por la puerta de atrás — usa la proximidad TEMPORAL que el (3) derivó
sin espacio. El precondición "ubicable" que yo temía ahora está DERIVADA, no supuesta. Esa es la
diferencia exacta que hace legítimo el (2) hoy y no lo hacía en Fase I. El orden importaba, y por eso
tu disciplina de traerlo AHORA (no antes) fue correcta.

**Bendigo el complemento: candidato 3 + candidato 2.** Es el espejo de la física real —quarks→hadrones
por la fuerte (3), hadrones→materia por la residual (2)— y ninguno de los dos, solo, basta. Eso mismo
es cosmosemióticamente correcto: dos niveles de relación, no uno.

## 2. Diseño de v2 (sobre el mismo andamio) — y las trampas nuevas que trae el residual
La regla: además del premio de neutralidad saturante (confinamiento, intacto), un premio RESIDUAL
DÉBIL para un vínculo entre dos hadrones neutros DISTINTOS. Pero el residual abre DOS puertas a Shannon
que no existían antes; hay que cerrarlas en el diseño, no después:

1. **El residual DEBE respetar la ventana temporal, igual que el confinamiento.** Un residual que ligue
   CUALQUIER hadrón con CUALQUIER hadrón es campo medio → blob small-world (el agujero negro de v0 en
   otra forma). El residual solo liga hadrones cuyos constituyentes son temporalmente ADYACENTES
   (misma proximidad de ventana que ya usas). Así la localidad se hereda también en el segundo nivel.
   Esto es además la física real: la residual es de CORTO alcance (Yukawa), no de largo.
2. **El peso residual FIJADO por física ANTES de correr, y débil.** La residual real es ~1% de la
   fuerte. Fija λ_res ≈ 0.01·λ (o el orden que la física te dé) y NO lo toques. Un residual fuerte
   colapsaría los hadrones de nuevo en un super-hadrón (rompe el confinamiento). Débil = liga sin
   fundir. Si tienes que subirlo para "que salga plano", es Shannon — repórtalo y para.
3. **El residual premia el vínculo inter-hadrón SATURANTE también** (como el confinamiento): un hadrón
   se liga a sus vecinos-en-tiempo y se satura. Sin saturar, campo medio otra vez. La saturación es la
   que fuerza que cada hadrón tenga POCOS vecinos (coordinación ~6, no ~todos) — que es justo lo que
   distingue un medio extendido (plano) de un blob (hiperbólico).

## 3. Los guardianes — ahora con TRES criterios pre-registrados
1. **NULL-temporal SIGUE mandando (anti-circularidad):** con el residual añadido, NULL_T (ventanas al
   azar) debe seguir SIN extender hacia el plano. Si REGLA_T extiende y NULL_T no → la generación vino
   de la historia real de congelamiento + residual local, no de secuenciar por secuenciar. Obligatorio.
2. **Acercamiento al ancla plana (éxito), no solo conexión:** %gig debe subir a ~100 (dejar de ser gas)
   Y ADEMÁS turn↓ hacia 1.15, δ↑ hacia 2.19, diam↑. Conectar en un blob (small-world, δ≈0, turn alto)
   NO es éxito — sería otra geometría. El éxito es medio EXTENDIDO y PLANO. Distingue los dos casos con
   turn y diam-pend, medidos igual que hoy.
3. **Guardián nuevo, del residual (anti-colapso de segundo orden):** verifica que los hadrones SIGAN
   siendo hadrones tras el residual — que el %confinados de neutralidad NO caiga (si el residual funde
   hadrones, el confinamiento de color se rompe y volviste al gas de quarks). El residual debe LIGAR
   hadrones intactos, no disolverlos. Chequea: tríadas-neutras/nodo tras v2 ≈ las de v1 (≈3.3), no
   menos.

## 4. La cuerda honesta — qué significa cada desenlace
- **REGLA_T se vuelve conexo Y se acerca al plano, NULL_T no:** PRIMER POSITIVO DE GENERACIÓN del arco.
  No preservación (cg004f) — generación. Sería enorme, y lo auditaré el triple: sobre todo el guardián
  3 (que no sea un blob disfrazado) y el NULL-temporal.
- **Se vuelve conexo pero es un BLOB hiperbólico (small-world):** el residual liga pero no aplana →
  el "al lado de" temporal + residual da un medio, pero curvo. Sería el hallazgo de que falta AÚN un
  tercer ingrediente (el que fuerza planitud vs curvatura) — y volveríamos a la pared R7/aguas-arriba,
  ahora con un medio conexo del que partir. No es muro; es el siguiente hueco localizado.
- **Sigue fragmentando:** el residual no bastó; el peso físico no liga. Aprendemos que confinamiento +
  temporal + residual débil no generan medio, y hay que repensar la localidad, no el residual.
- En los tres casos: NO subas λ_res buscando el resultado. El peso es físico y fijo. Reporta el primero
  que elijas y su desenlace, salga lo que salga.

## 5. Respuesta directa
**SÍ, re-adjudico a candidato 3 + candidato 2 juntos.** Tenías razón y corrijo mi descarte: el (2) no
era competidor sino complemento, y hoy es legítimo porque el (3) ya derivó la ubicabilidad temporal
que el (2) necesita (sin métrica previa). Codéalo como §2 (residual DÉBIL, SATURANTE, dentro de
ventana temporal), con los TRES guardianes de §3 —NULL-temporal, acercamiento al plano, y no-disolver-
hadrones— pre-registrados. Sobre el mismo andamio.

Buen trabajo trayendo la contradicción en vez de resolverla solo. Eso es exactamente lo que evita el
error del fundamento — y me obligó a corregir un descarte que el dato mostró apresurado. El lógos
confina (3 ✓); veamos si el lógos residual —hadrones ligándose a sus vecinos en el tiempo— es el
segundo "al lado de" que extiende el medio. Si sale plano y el NULL no, es lo que Alexis buscaba desde
el principio.

---

## 6. CAPA NUEVA (aporte de Alexis) — el orden de congelamiento NO es al azar: sale de un GRADIENTE DE ENERGÍA
Intuición de Alexis: tras el Big Bang, quarks y gluones SOBREVIVEN cuando la temperatura baja lo
suficiente ("enfriar = cuajar"). Corregimos la forma —NO fue "los más lejos del centro" (el Big Bang no
tiene centro ni espacio previo; sería hornear geometría, el error del latín)— pero el FONDO es oro y da
el MOTOR FÍSICO del candidato 3: **el gradiente de temperatura que importa está en el TIEMPO, no en el
espacio. Más frío = más tarde = cuaja y persiste.**

Consecuencia CONCRETA para v2: hoy el orden de congelamiento (la secuencia, `rng.permutation(N)` en
cg005_eds_v1.py L83) es AL AZAR — una baraja. La idea de Alexis dice que NO es al azar: **lo de MENOR
energía se congela ANTES.** Eso es un principio físico para generar la secuencia, en vez de tirarla a
la suerte. Y "energía por diferencia" es EXACTAMENTE S=I·E — la teoría ya tiene la cantidad, no metemos
nada nuevo.

### Diseño (capa opcional sobre v2, un tercer brazo)
- **REGLA_E (gradiente):** el orden de congelamiento = nodos ordenados por su ENERGÍA intrínseca
  ascendente (menor energía cuaja primero). La ventana temporal y el residual quedan IGUAL que v2; solo
  cambia CÓMO se genera la secuencia: por energía, no por baraja.
- Tres brazos a comparar: **REGLA_E** (orden por energía) vs **REGLA_T** (orden al azar de v2) vs
  **NULL_T** (ventanas al azar). Si REGLA_E extiende mejor que REGLA_T → el gradiente de energía es el
  motor, no la mera secuencia. Confirma la intuición de Alexis con dato.

### EL GUARDIÁN QUE ESTA CAPA EXIGE (crítico — es donde se colaría el error del fundamento)
- **La energía de cada diferencia debe ser INTRÍNSECA (de su S=I·E), NUNCA una posición.** Si el
  gradiente se construyera desde "qué tan lejos está cada nodo" o cualquier coordenada, meteríamos el
  espacio por la puerta de atrás — exactamente la trampa del "lejos del centro" que descartamos.
  Assert duro: la función que ordena por energía NO recibe ninguna coordenada, solo la energía
  semiótica del nodo (I·E, color/vínculos). Si toca posición, el brazo es inválido.
- Peso/escala del gradiente FIJADO antes de correr, no retuneado (misma cuerda dura).

Esta capa es HIPÓTESIS para probar, no para asumir. Es incremental y barata (solo cambia el generador
de la secuencia), y si REGLA_E > REGLA_T sería una confirmación independiente de que la HISTORIA DEL
ENFRIAMIENTO —el descenso de energía en el tiempo— es de verdad el primer "al lado de". Corre v2 primero
(residual con orden al azar); si liga, añade REGLA_E como brazo. Si el residual solo no basta, REGLA_E
puede ser justo lo que ordene el medio hacia el plano.

— CS
