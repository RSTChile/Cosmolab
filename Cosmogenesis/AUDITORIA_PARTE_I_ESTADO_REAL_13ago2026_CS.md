# AUDITORÍA DE LA PARTE I DE LA TEORÍA — qué está probado, qué falta probar, qué no se estudió

**13-ago-2026. Encargo de Alexis López Tapia.** Cobertura: los 255 informes del directorio, sin privilegiar
fechas (156 de ellos son anteriores al 3-ago). Fuente primaria: los ~50 documentos `ADJUDICACION_*_CS.md`,
que son la capa de veredictos propia del proyecto, más `CIERRE_ARCO_CG002_AUTORITATIVO.md`,
`INFORME_CG002_VEREDICTOS_TABLA.md` y las auditorías de motor.

**No se declara ningún nodo cerrado.** Esa decisión es del director. Esto es un inventario del estado de la
evidencia.

---

## 0. LO PRIMERO, PORQUE CAMBIA CÓMO SE LEE TODO LO DEMÁS

Esta auditoría empezó con la idea de actualizar `VERIFICACION_NODOS_TEORIA_PARTE_I_2026.md` (3-ago), que era
el inventario vigente. **Ese documento no se puede usar como base.** No por estar desactualizado diez días,
sino por cómo se escribió.

**Existe una adjudicación dedicada a los Bloques 1, 2, 2.5 y 2.6 —
`ADJUDICACION_nodos_Cosmogenesis_B1-2.6_CS.md`, del 17 de julio— que se presenta a sí misma como
"qué nodo tiene evidencia con NULL, cuál es parcial, cuál no se tocó. Sin inflar". El documento del 3 de
agosto no la cita ni una vez.** No usa la palabra "adjudicación" en ningún lugar de su texto: se escribió sin
mirar las actas del proyecto. Y difiere de ellas en cinco nodos, **en las dos direcciones**.

Verificado directamente para esta auditoría:

| Nodo | Acta del 17-jul | Documento del 3-ago |
|---|---|---|
| **C-N2 / C-N2.1 / C-N2.2** | **NO PROBADO en Cosmogénesis** — *"su evidencia viva está en ANIMA, no en la raíz. Vale anotarlo como **hueco de la RAÍZ**"* | C-N2 ✅ *"confirmado extensamente"*; C-N2.2 *"uno de los nodos con más apoyo experimental directo de toda la Parte I"* |
| C-N1.1 | **PROBADO como método** (es la cuerda anti-Shannon misma) | ⏳ degradado a "es una definición" |
| C-N1.3 | **PROBADO, de lo más firme** | ⊘ "filtro aislado débil" |
| C-N2.6.3 | **PROBADO** (CS057: viabilidad 0,47 en el punto físico vs 0,17 global) | agrupado en un ⊘ genérico |
| C-N2.6.4 | **TENSIONADO, con instrucción de partirlo en dos**: local ✅ / global-direccional ❌ | agrupado como ⊘ "consistente" |

En simple: **veníamos usando un resumen escrito sin leer las actas.** Infla algunos nodos, pierde otros, y en
un caso borra un negativo que ya estaba medido.

Además, el mismo documento contiene un caso rastreable de teléfono descompuesto. El "240 de 240" que le
atribuye al Bloque 3:

| Dónde | Qué dice |
|---|---|
| Fuente cruda (`RESUMEN_COMPLETO_ENFOQUE5_30de30_PARA_CS.md:87`) | **240/360** celdas alcanzan muerte térmica, y en todas ellas **la energía total** se queda en ~1,0 |
| Adjudicación ENFOQUE5 | "240 de 240 casos" |
| Auditoría completa | "240/240 casos" |
| **Documento del 3-ago** | "240 de 240 casos" — ahora atribuido a **"entropía monótona"** |

Se perdió el denominador (360 → 240) **y** cambió el observable: el experimento medía que la muerte térmica
conserva la energía —para distinguirla de la Nada—, no que la entropía siempre crezca. En la última versión
sostiene una afirmación que nunca midió.

**Consecuencia para el encargo:** las tres columnas que pediste se construyen abajo desde las adjudicaciones,
no desde el documento del 3-ago.

---

## 1. CÓMO SE CLASIFICÓ, Y POR QUÉ HAY UNA CUARTA COLUMNA

Pediste tres categorías. La evidencia obligó a una cuarta, porque hay nodos que **estuvieron probados y
después dejaron de estarlo** — y meterlos en "por probarse" borraría justo la información más valiosa.

- **A. Probado** — hay experimento dedicado, con control NULL o ablación explícita, que sobrevivió un intento
  honesto de tumbarlo, y **lo medido corresponde a lo que el nodo afirma**.
- **B. Probado en una versión más chica** — el experimento es sólido, pero mide un caso particular, no la
  afirmación del nodo. Es la categoría más poblada del proyecto.
- **C. Por probarse** — sin experimento, o con experimento diseñado y no corrido, o puesto a prueba y sin
  señal.
- **D. Retractado o anulado** — se dio por probado y una auditoría posterior lo tumbó.

Un criterio que aplico y conviene explicitar: **un nodo convertido en regla del juez no es un nodo probado.**
Si el enunciado se usa como criterio para decidir qué cuenta como evidencia, ya no puede fallar. Eso le pasa a
varios nodos del Bloque 1.

---

## 2. RESULTADO POR BLOQUE

### BLOQUE 1 — Persistencia Pre-Estructural

| Nodo | Estado | Base |
|---|---|---|
| C-N1 (S>0) | **Marco, no resultado** | Es el axioma del arco; el acta del 17-jul lo dice así. Se operacionalizó en Fase V como filtro de admisión P1 y **rechazó 0 de 165 reglas** — un filtro que no rechaza a nadie no discrimina |
| C-N1.1 (S≤0 ⇒ ∅) | **A, pero como método** | Es la cuerda anti-Shannon misma. Instancias duras: NULL-1 y NULL-2 dan **0 de 8 sumideros**, densidad cuatro órdenes de magnitud debajo del umbral. Salvedad de estatuto: es el criterio con el que se juzga, no una hipótesis en riesgo |
| C-N1.2 (persistencia en t+1) | **B** | El ✅ del 3-ago descansa en "todo el proyecto lo satisface", que no es un test. La mejor evidencia real está archivada bajo otros nodos: κ_P **z=6,53**, y Enfoque 5 con **z=41** y 100% de semillas congeladas por 100.000 pasos |
| C-N1.3 (la persistencia filtra) | **A** | El nodo con más historia del tramo. Jerarquía NULL completa: REAL vs NULL original **p=0,000333**. Poda por costo 0,786 vs azar 0,655 vs sin poda 0,421. **Acotación importante:** REAL y NULL-3 **no se separan** (p=0,42), y un Erdős-Rényi ajeno produce el 52% de la masa de REAL — el filtro existe, pero es **poco selectivo** |

### BLOQUE 2 — Acoplamiento Originario — **el hueco de la raíz**

Este es el hallazgo más serio de la auditoría, y no es nuevo: está escrito desde el 17 de julio.

| Nodo | Estado | Base |
|---|---|---|
| C-N2 (S = I ⟺ E) | **C — por probarse** | `PROTOCOLO_CG002_ACOPLAMIENTO_ORIGINARIO.md` pre-registró el veredicto **V1** justamente como el test que distingue C-N2 de "C-N1 prolongado". `INFORME_AVANCE_Cosmogenesis_29jun2026.md` lo reporta como **"Producción V1–V3, V6: PENDIENTE"** y no aparece corrido en ningún documento posterior. Lo único que corrió fue un **smoke de N=2 con 3 semillas** |
| C-N2.1 | **C** | Mismo veredicto V1 pendiente. **Cero menciones en todo el corpus posterior.** El propio 3-ago admite la circularidad: *"nunca se construyó un experimento con interior sin exterior"* |
| C-N2.2 (sin E o I ⇒ S=0) | **B, con instrumento averiado** | Sobreviven exactamente **dos** ablaciones auditadas: apagar electromagnetismo → cero hidrógeno; apagar fuerza fuerte → cero helio. Todo lo demás de ese motor se cayó (ver §3). Y **la identificación "electromagnetismo ≈ exterior, fuerza fuerte ≈ interior" no está justificada en ningún documento del registro** |

**El nodo más grande del bloque tiene el test dedicado más chico del proyecto: N=2, tres semillas.**

### BLOQUE 2.5 — Emergencia del Tiempo

El documento del 3-ago declara el grupo C-N2.5.6–2.5.10 como *"⏳ sin experimento propio"*. **Es incorrecto
para cuatro de los cinco.**

| Nodo | Estado | Base |
|---|---|---|
| C-N2.5 (tiempo emergente) | **B** | En CG002 el tiempo τ está construido literalmente así, **con su ablación**: α=0 → τ=0. Pero el acta lo marca PARCIAL por la razón correcta: *"lo usamos como ingrediente, no lo probamos como resultado"* |
| C-N2.5.1, 2.5.2, 2.5.4 | **C** | Sin experimento. C-N2.5.1 y C-N2.5.4: cero menciones en el corpus |
| C-N2.5.3 (Δ=0 ⇒ t=∅) | **B** | Sí tiene medición literal — Control G: *"α=0 → τ=0, Δ_struct=0, verificado"* — a escala smoke |
| C-N2.5.5 (asimetría primordial) | **C, y con alerta de fuente** | CF-1 da PASS cualificado (P≈0,033); CS070 da `direccion_real = 0.000` en **las 96 corridas sin excepción**. Pero **ninguno toca violación CP ni datos de partículas**, que es lo que el nodo afirma. El acta del 17-jul avisa: la única cadena de apoyo venía de *"una afirmación de Grok que NO está auditada contra el registro. **No apoyarse en ellas hasta verificar**"* |
| C-N2.5.6 (flecha temporal) | **C — probado y sin señal** | CS076 (5-ago): el único estadístico que aísla el orden da **z=−0,02 y z=+1,63**, muy debajo del umbral z≥3 del proyecto. Apagar la memoria **subió** el z, al revés de lo esperado |
| C-N2.5.7 (orientación emergente) | **B — omitido por el 3-ago** | ✅ cualificado en CG002 fila C1 |
| C-N2.5.8 (inercia histórica) | **B — omitido** | ⊘ con **tensión medida contra C-N5.1**: *"banda dura: umbral ×2 siempre; blanda: >×8 no voltea"* |
| C-N2.5.9 (coexistencia) | **B — omitido** | ✅ coexistencia en CG002 fila A5 |
| C-N2.5.10 (cancelación de orientaciones) | **B — y es el caso más grave de omisión** | El cierre autoritativo de CG002 dice: *"**el nodo más alto que aterrizó fue C-N2.5.10**… prefactor ½ que no se mueve al barrer la banda 6× (**cambio 0,0000 sobre 1080 corridas**)"*, ✅ DERIVACIÓN. **Y en el mismo arco tiene un ❌** (paredes de dominio: el déficit resultó independiente del ángulo → efecto de borde genérico) más un TENSIONADO del 17-jul: *"corrió como aniquilación de partículas, no como el nodo la interpreta"* |

**El nodo con el mejor resultado del arco fundacional está catalogado como "sin experimento".**

### BLOQUE 2.6 — Constricción Estructural

| Nodo | Estado | Base |
|---|---|---|
| C-N2.6 (curvatura del espacio de estados) | **B**, con un caveat que el 3-ago borró | El acta: *"'curvatura del espacio de estados' ≠ 'curvatura espacial' — **el nodo mezcla las dos con el ejemplo de los planetas**. Probamos lo primero, no lo segundo"*. CS057 midió un espacio de **parámetros**, no de estados |
| C-N2.6.1 | **C** | NO PROBADO: pertenece al acoplamiento sistema-entorno, que Cosmogénesis nunca aisló |
| C-N2.6.2 (trayectorias siguen el gradiente) | **B — sustitución declarada** | CS077 es sólido: REAL forma **7–18× más masa ligada** que dirección al azar en los 13 puntos, con validación bit a bit del motor. Pero el acta ya había advertido: *"el enfriamiento sigue un gradiente de energía, **pero no es el ∇A_sys-env del nodo**"*. Se midió gravedad contra patadas al azar |
| C-N2.6.3 (mínimos locales = atractores) | **A — omitido por el 3-ago** | CS057: viabilidad **0,47 en el punto físico vs 0,17 global**. Reforzado por cs074-A, con un control decisivo: el control sin energía da la curva **idéntica a la real, diferencia +0,0 en los 20 valores** → el techo no es energético |
| C-N2.6.4 (acumulación ⇒ organización) | **Partido en dos, por instrucción del propio proyecto** | **Local: A.** CS066, tejido con clustering 0,41. **Global-direccional: refutado**, por tres rutas independientes (CS067B, CS068, CS069B), y **Fase III lo reconfirmó el 8-ago** |

### BLOQUE 2.7 — Regímenes de Acoplamiento — **donde está la circularidad**

El documento del 3-ago marca seis nodos como probados con la justificación *"✅ citado en la propia Teoría"*.
Es decir: la Teoría cita al experimento, y el experimento se da por bueno porque la Teoría lo cita. **Cinco de
los seis dependen de esa cita.** Y tres de ellos —2.7.7, 2.7.9 y 2.7.12— descansan sobre **un solo
experimento**, CS066.

| Nodo | Estado | Base |
|---|---|---|
| C-N2.7 (regímenes discretos) | **B** | Sólido para el régimen gravitacional: 6 controles NULL, p=0,000333 |
| C-N2.7.1–2.7.4 | **D en buena parte** | El bloque con dos retractaciones, dos motores NO ADMISIBLE, una firma suspendida y una auditoría que tumba el motor entero (§3) |
| **C-N2.7.7 (Δ encadenada ⇒ distancia)** | **B, y la evidencia original está erosionada** | Ver abajo — es el nodo central |
| C-N2.7.8 (escalera de la geometría) | **D en su peldaño clave** | Su único positivo sobre dimensión fue anulado: **REAL 2,77 = NULL 2,80** (§3). Además se apoya en pendientes de diámetro medidas con un `_diam` que tenía un bug — y **el diámetro de CS066 nunca se re-midió** |
| C-N2.7.9 (distancia y dirección separables) | **B** | El mismo CS066 que 2.7.7: dos nodos consecutivos sobre un solo experimento. El juez de ejes necesitó dos correcciones formales, y una encontró que **el candado obligatorio nunca se había implementado** |
| C-N2.7.10 (la dirección no emerge) | **A — el mejor del bloque** | **El único con convergencia genuina**: tres rutas independientes (clásica, cuántica, memoria de enlace) con matemáticas y jueces distintos, 96+160+96 corridas, más un segundo estimador (δ-Gromov) que confirma. No es la misma medición repetida |
| C-N2.7.11 (π contingente) | **C, y es el caso más frágil** | Cronología: hipótesis del director → medición 16-jul → incorporado al canon **al día siguiente** → adjudicación que ya usa el nodo **como norma para vetar evidencia** → doc del 3-ago que cita el canon. **Sin NULL, sin estadística, 3 de 4 sustratos hechos a mano**, y atribuido a CS068, **que no lo midió** |
| C-N2.7.12 (geometría como estado condensado) | **C** | Sin experimento propio. Y en CS072 funcionó como dispositivo que convierte un resultado nulo en confirmación: *"no es muro ni fracaso: es exactamente lo que la Teoría predice"* |
| C-N2.7.5, C-N2.7.6 | **C** | 2.7.5 fue diseñado y **no corrido, por riesgo declarado de fabricar el resultado** — una decisión correcta. 2.7.6 es aclaratorio |

#### C-N2.7.7 en detalle — el nodo al que apunta todo el trabajo de esta semana

El nodo afirma: *"la distancia no preexiste — emerge cuando varias distinciones se encadenan"*.

**A favor:** CS066 (clustering 0,41 vs 0,10 barajado, diámetro 12 vs 3,9), y en CS072 la materia oscura como
condición de la distancia (sin ella el diámetro es **1 fijo** en todo N; con ella, 19–26).

**En contra, y esto ya estaba antes del 3-ago:**
- **El confirmatorio de CS066 falló su propio test pre-registrado**: el exponente no confirma d≈3 en **0 de 6
  celdas**. Veredicto textual: *"esponja 3D-local con mundo-pequeño residual, **NO 3-manifold métrico**"*.
- **CS068**: el residual medido es **13× del lado equivocado** — 6,0 a N=900 cuando una métrica 2D predice
  ~58. *"El sustrato jamás tuvo métrica bajo los atajos: mundo-pequeño hasta el fondo."*
- **Fase III (8-ago)**: agrupando a distintas escalas, la pendiente es indistinguible entre real (0,376),
  barajado (0,420) y azar puro (0,406). Mundo-pequeño **a cualquier escala**.

**Y lo que se midió esta semana mide otra cosa.** Fases V-B a VIII establecieron una cadena real y bien
controlada: **apiñamiento del soporte → pico local de densidad → masa acretada**, con el pico llevándose el
75–82%. Eso es un resultado sólido. Pero es *"cómo la estructura inicial de un grafo cambia cuánta masa
acreta un simulador de gravedad"*, no *"la distancia emerge del encadenamiento de diferencias"*.

**Y ni siquiera esa versión chica sobrevive el cambio de resolución.** F8-06, terminado hoy: a N=4000 el
efecto cae a **+1,07 σ, signos 6/11, p=0,354**, contra +13,8% con 12/12 y p=4,9·10⁻⁴ a N=2000. La regla no
dejó de funcionar — la balanza dejó de poder verla, porque el ruido lo ponen los sumideros y a esa resolución
nacen 30 en vez de 8.

### BLOQUE 2.8 — Invariantes del Cierre — **nunca estuvo vacío**

El documento del 3-ago (12:11) declara el bloque completo sin experimento. **Diecinueve minutos después**,
`DISENO_EXPERIMENTOS_NODOS_ABIERTOS_desde_2.5.5_CS.md` reporta mediciones con control.

| Invariante | Estado |
|---|---|
| κ_P (mínimo de persistencia) | **A** — z=6,53 contra 8 NULL |
| κ_Δ (mínimo de diferencia operable) | **A** — z=33,8 y 43,3 |
| κ_V (piso de acoplamiento) | **B débil** — p=0,111 es el piso de n=9; descartado como puente; **se invierte el signo** a N=4000 |
| κ_O, κ_LF, κ_H | **No aplicables a este nivel de descripción** — declarado con razón escrita el 17-jul y reafirmado el 3-ago. **No es un vacío de instrumentación: es una decisión argumentada** |

**Contradicción matemática del canon, abierta desde el 17 de julio y sin resolver:** la fórmula
`Λ_Cos = (Δ_struct · LF · A_sys-env) / |e_R|` implica que `Λ_Cos → +∞` cuando `|e_R| → 0`. Es decir, **la
fórmula califica como salud máxima justo el límite que el propio teorema declara no viable**. Codex recomendó
no usarla como juez hasta adjudicarla. El proyecto hermano Cosmoclima cazó el mismo defecto de forma
independiente y empírica el 12-ago.

### BLOQUE 3 — Historia

| Nodo | Estado |
|---|---|
| C-N3, C-N3.1, C-N3.2 | **B, sobre base más delgada de lo que parecía** — la evidencia principal, CS009 y CS014, **existe sólo como dos líneas de tabla de registro: sin informe, sin números y sin control NULL**. Y el "240 de 240" es 240/360 y mide otra cosa (§0). El nodo está marcado ✅ sin salvedades |

### BLOQUE 4 — Delimitación

| Nodo | Estado |
|---|---|
| C-N4, C-N4.1, C-N4.2 | **B, y con la consecuencia práctica más grande sin resolver** |

Hay dos experimentos del 5-ago con 3 de 4 métricas separando sin solape. Pero **F8-05 contestó la pregunta de
C-N4 por accidente**: sobre el mismísimo volcado, el mismo contraste da **+0,01433 con 12/12** medido como
masa en sumideros y **−0,02275 con 2/12** medido con otra frontera. **La elección de frontera invierte el
signo de un resultado.** Sumado a que la métrica más pertinente de CS079 no discrimina (REAL 0,509 dentro del
rango NULL 0,425–0,567) y a que los 8 histogramas NULL son sospechosamente idénticos entre sí **sin revisar**.

Si la delimitación no es neutral, ningún resultado del corpus es independiente de ella.

### BLOQUE 5 — Dinámica

| Nodo | Estado |
|---|---|
| C-N5, C-N5.1, C-N5.2 | **A** — cs074-A, 1920 corridas, con pre-registro y control energético verificado en disco. Es de lo más limpio del proyecto |

---

## 3. LO QUE SE PROBÓ Y DESPUÉS SE CAYÓ

Esta columna no estaba en el encargo, pero es donde vive la lección más cara del proyecto.

| Qué se afirmó | Qué prevaleció |
|---|---|
| **"Materia emerge — el resultado más sólido del arco"** (18-jul) | **RETRACTADO el mismo día.** Apagando confinamiento + electromagnetismo + gravedad + QCD + Pauli, el conteo se queda en **9 bariones, sin moverse uno**. Era agrupamiento térmico renombrado |
| **"HALLAZGO MAYOR: la dimensión emerge"** (18-jul), dim 1,00…3,85, ley N^(1/D), **verificado en paralelo línea por línea** | **CAE dos días después.** La dimensión **copia el input**; REAL 2,77 = NULL 2,80. Y el docstring del módulo traía **el número objetivo escrito** |
| "Umbral crítico cero→materia = tesis #2" | **RETRACTADO.** Artefacto de correr sólo 150 pasos; a 600 pasos todo da 9. Más una fórmula con peso negativo (−0,8) |
| Motor v5, motor v6 | **NO ADMISIBLES.** En v6 la temperatura nacía del índice del catálogo |
| Firma del fold de 21 piezas | **SUSPENDIDA.** Los dos brazos usaron **semillas distintas** — eran universos distintos, no la misma partida con un cambio |
| "Apagar confinamiento → 0 bariones: ADMISIBLE" | **CAE** con la auditoría del 20-jul: en el motor vivo `bariones = quarks/3` exacto e invariante; la máscara de ligadura era **código muerto** |

**La lección de método, y es la más importante del corpus:** el "hallazgo mayor" de la dimensión fue
**verificado en paralelo por una corrida independiente, línea por línea, sin una sola discrepancia** — y era
falso. Lo detectó **un NULL** corrido dos días después.

> **Reproducibilidad no es validez.** Dos ejecuciones que coinciden perfectamente no detectan nada, porque el
> problema no era el azar de la máquina sino la ausencia de un control.

---

## 4. RESPUESTA DIRECTA AL ENCARGO

**Qué está comprobado completamente.** En sentido estricto —experimento dedicado, control que podía tumbarlo,
y correspondencia entre lo medido y lo afirmado— la lista es corta y honesta:

- **C-N2.7.10** (la dirección no emerge de la relación pura). El mejor del proyecto: tres rutas
  independientes con jueces distintos convergiendo. Es un resultado **negativo**, y por eso es robusto: un
  parámetro puesto a mano no puede fabricar una ausencia.
- **C-N5 / C-N5.1 / C-N5.2** (estabilidad en rango, colapso fuera). 1920 corridas, pre-registro, control
  energético.
- **C-N2.6.3** (mínimos locales como atractores). Con el control decisivo de "sin energía da lo mismo".
- **C-N1.3** (la persistencia filtra), acotado: el filtro **existe** pero es **poco selectivo**.
- **κ_P y κ_Δ** del Bloque 2.8.
- **C-N1.1**, con la salvedad de que es el método del proyecto más que una hipótesis en riesgo.

**Qué está por probarse.** La mayoría — y sobre todo lo que está más arriba en la Teoría:

- **Todo el Bloque 2 (C-N2, C-N2.1, C-N2.2)**, que es la raíz. El test que podía falsarlo se diseñó, se
  pre-registró y **nunca se corrió en producción**.
- **C-N2.5.5** (asimetría primordial) y **C-N2.5.6** (flecha temporal): el segundo se puso a prueba y no dio
  señal.
- **C-N2.7.7** en lo que el nodo afirma. Lo medido esta semana es una instancia concreta y bien controlada, y
  ni siquiera esa sobrevive al cambio de resolución.
- **C-N2.7.11** (π contingente) y **C-N2.7.12**.
- **C-N4** como pregunta abierta, con la agravante de que su respuesta condiciona todo lo demás.

**Qué no se estudió.** Menos de lo que el inventario anterior decía:

- **C-N2.5.1, C-N2.5.2, C-N2.5.4** — sin experimento, y dos de ellos sin una sola mención en 255 informes.
- **C-N2.6.1** — pertenece al acoplamiento sistema-entorno, que este proyecto nunca aisló.
- **C-N2.7.5** — diseñado y no corrido, por riesgo declarado de fabricar el resultado.
- **κ_O, κ_LF, κ_H** — con la precisión importante de que **no es un vacío, es una decisión argumentada**: se
  declararon no aplicables a este nivel de descripción.

**Y una cuarta respuesta que no pediste pero que el registro impone:** hay nodos que estaban en la columna
"comprobado" del inventario anterior y hoy están en "retractado" — la materia que emergía, la dimensión que
emergía, el umbral crítico. Ninguno de esos tres aparece en el documento del 3-ago, ni en su versión positiva
ni en su anulación.

---

## 5. ADVERTENCIAS TRANSVERSALES

1. **El observable "masa en sumideros" sostiene casi todo lo reciente** — la jerarquía NULL entera y las
   Fases V-B a VIII. El propio director lo declaró conceptualmente inadecuado para preguntas de historia y
   futuros: *"un sumidero es un horizonte — la materia que cruza no tiene más historia posible"*. La objeción
   se levantó contra O-N7.7; **el instrumento es el mismo**.
2. **El piso del instrumento no es 1 partícula, es ~5.** Todo efecto reportado entre 1 y 5 debería releerse.
3. **Subir la resolución empeora la medición** de efectos chicos (σ ∝ sumideros^1,24). Confirmado hoy por
   medición directa: la extrapolación estaba 2,4× optimista.
4. **El bug `_diam`** afectó 15 de 430 reglas y la categoría "intermedio" era artefacto completo. **El
   diámetro de CS066 —evidencia central de C-N2.7.7— nunca se re-midió.**
5. **Discrepancia numérica sin resolver:** el 3-ago atribuye a CF-1 *"z entre 4,9 y 7,8"*; la adjudicación de
   sello reporta **P≈0,032–0,034** y no menciona esos z.
6. **Pendiente de verificar:** si el motor de viabilidad de CS057/CS062 —evidencia única de C-N2.6 y
   C-N2.6.3— comparte código con las piezas fabricadas de CS072. El registro no lo dice en ninguna dirección.
7. **Dos scripts de referencia no están en disco:** el que produjo los números originales de κ_V, y el barrido
   de Fase V-A no es reproducible porque sus semillas salían de `hash()` sin fijar `PYTHONHASHSEED`.

---

## 6. LO QUE ESTA AUDITORÍA NO HACE

No propone experimentos. No declara nodos cerrados. No resuelve las discrepancias entre el documento del
3-ago y las actas del 17-jul: las expone para que el director decida cuál vale.

Una observación final sobre el patrón, porque es la que más se repite: **los nodos mejor sostenidos son casi
todos negativos** — la dirección no emerge, el sustrato no es métrico, el techo no es energético, apagar esto
apaga aquello. Los más frágiles son casi todos positivos y llegaron rápido al canon. Eso no es casualidad ni
mala suerte: un resultado negativo es difícil de fabricar sin querer, y uno positivo es fácil. El proyecto ya
lo aprendió tres veces por las malas, y las tres veces lo escribió.
