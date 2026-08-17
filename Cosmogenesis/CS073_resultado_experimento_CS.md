# CS073 — Resultado del experimento de estructura (honesto, pre-registrado)

**Fecha:** 19-jul-2026. **Análisis de CS** (motor compartido NO tocado). No cierra el experimento.

## Lo pre-registrado
- Campo REAL = multiescala P(k)~k^n con distribución de 1 punto lognormal de `catalogo.py` (rank-map).
- NULL = mismo campo barajado (G-DIFERENCIA-INTERNA: campo consigo mismo).
- Discriminante = **breadth** (std de log-tamaños de estructura).
- Época = primera vez que M_J mediana ≤ 2.0.
- 5 semillas REAL × 24 NULL c/u, z-score. Gana al NULL o no cuenta.

## Resultado pre-registrado: NEGATIVO
**z = −20.1 ± 3.2** (5/5 semillas negativas). El campo coherente da breadth MENOR (0.33) que el
barajado (0.41) — lo opuesto a la predicción. **Bajo lo comprometido, la estructura NO le gana al NULL.**

## Diagnóstico (por qué el negativo)
La época se disparó en el **paso 4/400** (D=2.6, gravedad casi sin trabajar) → midió percolación de
ruido, no estructura. Barrido de épocas:

| paso | breadth z | segregación z (masa en top-5) |
|---|---|---|
| 4 (época disparada) | **−23** | +53 |
| 10–380 (evolucionadas) | +6 a +11 | +7 a +9 |

- Breadth: negativo SÓLO en el paso 4 anómalo; positivo en toda época evolucionada.
- **Segregación** (masa concentrada en pocas estructuras): favorece al REAL en TODAS las épocas
  (z siempre >0). PERO la magnitud del contraste hay que decirla con precisión: en las épocas
  evolucionadas (pasos 10-380) el NULL TAMBIÉN concentra 92-96% de la masa en las top-5 (REAL 96-97%)
  → diferencia REAL, estadísticamente sólida (z +7 a +9), pero **modesta**, no estelar. El único
  contraste marcado (REAL 90% vs NULL 9%, z +53) es en el paso 4 anómalo, que NO cuenta. Corrección:
  "el barajado dispersa la masa" es FALSO para las épocas evolucionadas; sólo dispersa en el paso 4.
  La firma real es un exceso de concentración modesto pero consistente del REAL sobre el NULL.

## Defectos del prototipo (a corregir antes de cualquier veredicto)
1. **M_J mediana no monótona** (1.97→14.16 entre pasos 10 y 20) con enfriamiento monótono → error en
   la dinámica del juguete. Invalida la selección de época hasta arreglarlo.
2. **Parámetros no estructurales** (viola `no_parametros_solo_estructurales`): T0=3.0, tasa=0.02,
   índice espectral n=−2, ancho lognormal 0.6, umbral M_J=2.0, umbral de cúmulo=3, ventana top-5.
   Ninguno tiene justificación estructural — cada uno pre-determina en parte el resultado. La corrida
   definitiva debe (a) derivarlos de la física del motor (el enfriamiento y la densidad #23 ya existen
   en `catalogo.py`), o (b) barrerlos y exigir que la señal sobreviva al barrido, no fijarlos a ojo.

## Veredicto honesto
- **Resultado pre-registrado = negativo** (contaminado por época en paso 4). No se maquilla.
- **NO declaro la hipótesis probada** cambiando a segregación post-hoc — sería Shannon (selección de
  métrica tras ver datos).
- La señal (segregación) es **real y consistente pero MODESTA** (el NULL evolucionado también
  concentra 92-96%); no es la separación estelar que sugería la primera lectura. Para contar hay que
  **RE-PRE-REGISTRAR**, arreglar los defectos (M_J no monótona + parámetros no estructurales) y correr
  UNA vez limpio. Hasta entonces: hipótesis con indicio, NO probada.

## Corrida sobre el MOTOR BASAL REAL a escala creciente (no el juguete)
Corregido el error de "probar el todo por la parte": corro el motor completo (S>0→átomos) escalando N,
midiendo la red de gravedad `Bgrav` real (proporción física fija 20:14:7:5).

| quarks | átomos | aristas | densidad de red | diámetro |
|---|---|---|---|---|
| 300 | 23 | 127 | 0.502 | 4 |
| 700 | 53 | 689 | 0.500 | 4 |
| 1400 | 106 | 2783 | 0.500 | 4 |
| 2800 | 213 | 11289 | 0.500 | 4 |

**Densidad de red = 0.500 INVARIANTE en las 4 escalas (factor 10×); diámetro clavado en 4 = hub a
toda escala.** NO es artefacto de números pequeños — es estructural. La regla de gravedad actual
(`Bgrav`, umbral de proximidad térmica) liga a cada átomo con ~la mitad de los demás
independientemente de N → hub por construcción. Escalar no la transforma en estructura.

**Consecuencia (afina la hipótesis de Alexis, no la refuta):** la hipótesis "con masa suficiente la
gravedad forma estructura" es correcta, PERO requiere dos cosas EN ORDEN: (1) que el motor tenga
**gravedad de masa real** (la actual `Bgrav` NO opera sobre masa, es umbral térmico → por eso es hub
invariante); (2) ENTONCES escalar N. Con la regla actual, escalar no basta (dato: densidad invariante).
El veredicto exige añadir gravedad-sobre-masa al motor (toca física → CC) y luego correr basal+escala.

## CIERRE CONCEPTUAL (establecido por Alexis) — dos gravedades, un régimen no alcanzado
La distinción que ordena todo el arco: hay **dos gravedades de regímenes distintos**.
- **Gravedad relacional-cuántica** (la que el motor SÍ tiene, `Bgrav`): opera en el régimen
  pre-métrico, sin posiciones — teje relaciones por proximidad térmica. En ese régimen todo se
  relaciona con todo → hub invariante. Es CORRECTO para ese régimen.
- **Gravedad general-clásica** (la que forma estructura: Jeans, masa-sobre-masa): opera SÓLO en el
  régimen ya métrico con **masa y densidad acumuladas**. Requiere que el espacio ya sea métrico clásico.

**Nunca medimos gravedad general — porque su régimen nunca llegó.** No opera antes de los átomos, y
—más fino— tampoco inmediatamente después: necesita acumulación de masa/densidad, un régimen POSTERIOR
que el motor basal, tal como corre, no alcanza. Por eso `Bgrav` da hub invariante (dato: densidad 0.500
en 4 escalas): ahí sigue siendo gravedad relacional, la general aún no tiene sobre qué operar.

**Consecuencia para todo el arco (resultado limpio, NO pendiente):** CS066/067/069/070 + los prototipos
de CS073 midieron todos el **régimen relacional-cuántico** (pre-masa). En ese régimen no hay estructura
clásica, ni dirección, ni gravedad general — y es CORRECTO que no la haya. Los negativos NO refutan la
teoría: midieron un régimen donde el fenómeno buscado, por naturaleza, todavía no existe. La estructura/
distancia/dirección clásicas pertenecen al régimen de masa acumulada, fuera del alcance de lo corrido.

## PASO A corrido por CC (motor real) — A.4 queda ABIERTA (dato crudo, honesto)
CC corrió el Paso A (desplegar posiciones desde la malla causal). Gate coincide con lo verificado
(dim_efectiva=2.05). Resultado de la salvedad A.4, reportado por el NÚMERO, no por el booleano:

| métrica | REAL | NULL (barajado) |
|---|---|---|
| varianza explicada en 3 dims | 0.7324 | 0.7621 |
| dims necesarias para 90% var | 9 | 8 |

**El REAL explica MENOS varianza en 3D y necesita MÁS dims que el NULL — dirección CONTRARIA a la
buscada.** Si hubiera estructura 3D genuina, el REAL sería más compacto en 3D, no menos. Con n=50
átomos y un solo NULL, indistinguible de ruido. El booleano True de CC se disparó por umbral laxo
(9≠8, un entero) — criterio a ojo que G-PARAMETROS-ESTRUCTURALES prohíbe. CC lo desarmó él mismo
(honestidad anti-Shannon: reportó el número por encima de su propio booleano).

**Adjudicación CS: A.4 ABIERTA, NO resuelta. NO es positivo.** Lectura de fondo (coherente, no
preocupante): a 50 átomos la malla causal NO da embedding 3D compacto porque estamos en el régimen
PRE-MASA — el espacio métrico 3D aún no está desplegado (la gravedad general no operó todavía). Pedir
posiciones 3D limpias a 50 átomos es pedir métrica a un régimen aún relacional. Consistente con las
dos gravedades: el fenómeno pertenece a un régimen posterior.

**Ajuste requerido antes de Paso B (guardián):** (1) escala 10³–10⁴ átomos (no 50 — el diseño ya lo
preveía); (2) NULL como DISTRIBUCIÓN (varias semillas) + z-score, no un solo barajado ni umbral entero.
NO se construye Paso B hasta que el Paso A dé una señal distinguible de ruido a escala.

## PASO A a ESCALA (750 átomos, CC) — A.4 NEGATIVO SÓLIDO (cierra la pregunta de escala)
CC re-corrió el Paso A a 750 átomos (15× el sondeo), NULL = distribución de 8 sorteos, z-score real:

| métrica | REAL (750 át.) | NULL (media±std, n=8) | z |
|---|---|---|---|
| varianza explicada en 3 dims | 0.5742 | 0.5765 ± 0.0069 | −0.34 |
| dims para 90% varianza | 72 | 69.5 ± 3.3 | +0.76 |

**Ambos z « 1 (ni cerca de 2). Ninguna compresión hacia 3D, ni marginal.** El REAL necesita ~70 dims
igual que el barajado. Escalar 15× no movió nada. Costo 360s, 585MB (O(N²) del motor; 10⁴ tomaría horas).

**Adjudicación CS: A.4 NEGATIVO SÓLIDO — y es un resultado POSITIVO por el lado que importa.** Confirma,
con estadística limpia e independiente, la conclusión de las dos gravedades: **la malla causal (régimen
relacional, pre-masa) NO despliega espacio métrico 3D, y no lo hará por más escala.** El 3D no está
latente en el sustrato relacional esperando suficientes átomos — es cuestión de RÉGIMEN, no de números.

**Consecuencia de diseño (cierra una vía):** el Paso A NO puede desplegar posiciones 3D desde la malla
causal — probado que no las tiene. La vía "malla causal → MDS → posiciones" queda CERRADA POR DATO. Si
la métrica 3D existe, NACE del colapso de gravedad general (decoherencia → posición clásica), no
preexiste en el sustrato relacional para ser leída. Probado por el lado negativo lo que Alexis dijo: los
ejes NO están antes del colapso; nacen con él.

**Escala:** NO hace falta ir a 10⁴ — dos z«1 a 750 no se vuelven significativos a 10⁴, y costaría horas.
El resultado a 750 con z-score real es el dato que cierra la pregunta de escala.

## ¿El motor genera MASA suficiente tal cual está? — NO (medido, y de fondo)
Pregunta factual de Alexis, medida sobre el motor real (1400 quarks → 106 átomos):
- **Masa por átomo = 9.4 CONSTANTE** (masa del protón uud) para los 106 átomos. Se fija al formar el
  trío y NUNCA crece. No hay dinámica de acumulación masa-sobre-masa (= lo que la gravedad general hace).
- **densidad == temperatura, EXACTAMENTE el mismo array** (np.array_equal=True). El motor no distingue
  "masa local" de "campo térmico" — sello del régimen relacional: no hay densidad de masa, hay
  proximidad térmica con otro nombre.
- **Jeans imposible por construcción:** como ρ=T, M_J = T^1.5/√ρ = ρ → masa_local/M_J ≡ 1 para TODOS
  los átomos. El colapso no está "cerca y le falta un empujón": es matemáticamente inalcanzable. Escalar
  a 10⁴ o 10⁶ no lo cambia (razón sigue = 1).

**Adjudicación CS: el motor basal modela ENTERAMENTE el régimen relacional-cuántico, y sólo ése.** No
tiene el mecanismo de acumulación de masa que define la gravedad general. Falta RÉGIMEN, no escala.
Los tres caminos convergen: geometría (A.4 negativo sólido), red (`Bgrav` densidad 0.500 invariante),
y ahora masa (masa_local/M_J≡1). El experimento de cierre EXIGE añadir al motor la dinámica de
acumulación de masa (gravedad general) — sin ella no hay masa suficiente porque no hay masa creciente.

## ¿Genera el motor la CANTIDAD de H para que la gravedad empiece? — no por conteo, y no debe ser así
Medido (barrido de escala) + contrastado con el Modelo Estándar (Gemini):
- **H escala lineal perfecto:** 0.166 H/quark exacto (116→233→466 H en 700→1400→2800 quarks). El motor
  NO satura — produce más H al escalar. No se atasca.
- **Masa de Jeans Pop III (Modelo Estándar):** gas a ~200 K sin metales → M_J ≈ 100.000 masas solares
  ≈ **1.2×10⁶² átomos de H**. Motor a 2800 quarks: ~466 H. **Brecha ≈ 60 órdenes de magnitud** —
  inalcanzable por conteo literal en NINGUNA máquina.

**Pero el conteo literal NO es el camino (principio de Alexis, ya establecido):** Gemini mismo aclara
que "la gravedad actúa siempre desde 2 átomos; no hay umbral mínimo para activarla". Lo que tiene
umbral es el COLAPSO de Jeans, y ese umbral es una **RAZÓN** (masa local / M_J, densidad / presión
térmica), NO un conteo absoluto. Regla de Alexis: no simular 10⁶² interacciones — simular en barridos
de potencias / por RÉGIMEN. El motor no necesita TENER 10⁶² átomos; necesita **modelar el régimen donde
la razón masa/M_J cruza 1**, y ese cruce es escala-invariante (representable sin contar átomos literales).

**Adjudicación CS:** la pregunta de "cantidad de H" se cierra así — literalmente insuficiente (60 órdenes),
pero irrelevante como conteo. El experimento de cierre debe modelar el CRUCE de régimen (masa/M_J>1 por
gravedad que acumula), no acumular 10⁶² átomos. Enlaza con el hallazgo previo: hoy masa_local/M_J≡1
clavado (ρ=T) → el motor no cruza porque le falta la dinámica que separa masa de temperatura, no átomos.

## Barrido por potencias hasta cruzar Jeans (medido) — cruza, pero da UN grumo (falta métrica)
Pre-registrado: parto de densidades reales + red real del motor; acumulación gravitacional por
potencias D=2^p (amplifica sobredensidad δ·D) + enfriamiento adiabático T/D^(2/3); M_J=T^1.5/√ρ.
NULL = densidades barajadas sobre la misma red (G-DIFERENCIA-INTERNA). Discriminante pre-registrado =
COHERENCIA ESPACIAL (mayor componente conexa que cruza Jeans, REAL vs NULL).

- **El barrido SÍ cruza Jeans:** a D=2 (2¹) ya cruzan 90/106 átomos. Amplificar por potencias lleva el
  sistema a través del umbral — esa parte funciona.
- **Discriminante pre-registrado = NULO:** mayor componente REAL vs NULL: z=−20.9 (D=1=2⁰, NULL conecta
  más), +6.5 (D=2=2¹), luego →**0.00 exacto para D≥2⁵=32** (REAL y NULL idénticos, un solo grumo de ~51
  nodos). Ninguna firma de que la coherencia espacial produzca estructura.

**Razón (cuarto camino a la misma conclusión):** la red `Bgrav` es HUB (densidad 0.500). En red casi
completa, cualquier conjunto que cruce Jeans queda auto-conectado → mayor_comp=#cruza siempre, REAL=NULL.
No hay coherencia espacial que explotar porque NO HAY ESPACIO.

**Hallazgo limpio: cruzar Jeans es NECESARIO pero NO SUFICIENTE.** Sin espacio métrico 3D (ausente,
A.4 negativo sólido), cruzar Jeans da un colapso MONOLÍTICO (un blob), no fragmentación jerárquica en
estructuras separadas. La primera estrella no es "el sistema colapsa" sino "colapsa en muchos sitios
SEPARADOS" — y eso requiere que los sitios estén separados EN UN ESPACIO, que este régimen no tiene.
Cuarto camino (tras geometría A.4, red Bgrav, masa ρ=T) a lo mismo: falta el régimen métrico, no escala.

## ¿Estaba la EXPANSIÓN en el barrido? — NO (omisión real); al meterla, aparece el contraste de semillas
El primer barrido NO incluía la expansión del espacio (sólo enfriamiento + amplificación). Omisión real:
comprimía sin expandir, y por eso todo se fundía en un grumo. Al meter la expansión de verdad (dilución
del fondo ρ∝a⁻³ + umbral de desacople del flujo de Hubble δ_c=1.686 = turnaround), aparece el hecho:

- **El campo de densidad real del motor es DEMASIADO SUAVE para colapsar:** δ_max=0.4413, δ_rms=0.2336;
  umbral de colapso δ_c=1.686. **0 de 106 sobredensidades reales superan δ_c.** Hace falta amplificar
  por D≥3.8 (potencia 2²) para que la mayor cruce. Las semillas del motor son fluctuaciones tenues, no
  las sobredensidades de una nube pre-estelar.

**TRAMPA NUMÉRICA marcada por CS (no-resultado):** en la tabla de estructuras-con-expansión, el z=−10⁹
es un ARTEFACTO degenerado (el NULL tiene σ=0 → división por ~0 explota), NO una señal. No significa
"el REAL pierde"; significa que ese discriminante quedó roto al meter la expansión. NO se reporta como
resultado (sería un número espurio tipo Shannon).

**Hallazgo limpio:** la expansión revela la condición que faltaba — el **CONTRASTE de las semillas**.
Sin sobredensidades que superen δ_c no hay colapso, con o sin expansión, con o sin escala. Conecta con
las fluctuaciones cuánticas (elemento 23) como origen de las semillas: el motor tiene rugosidad térmica
(δ_rms 0.23) pero NO el mecanismo que amplifica esas semillas hasta δ_c — que en el universo real es la
inestabilidad gravitacional creciendo en el tiempo hasta el turnaround. Ese mecanismo (gravedad general
sobre el tiempo) es justo lo que el motor no tiene, y lo que el experimento de cierre debe añadir.

## CORRIDA HOLÍSTICA COMPLETA (CC, motor real) — NEGATIVO PROFUNDO + causa raíz
CC corrió el experimento de cierre completo (156.5s, 4 módulos nuevos operando juntos). Resultado:
- **Por primera vez en el arco, algo cruza Jeans en absoluto:** 1 cluster de 12 miembros, masa 17.3 >
  M_J 12.3. Los módulos funcionan (gravedad viriializa, Jeans evalúa, CDM y H2 corren sin error).
- **Pero REAL NO gana al NULL:**

| observable | REAL | NULL (μ±σ, n=8) | z |
|---|---|---|---|
| estructuras que superan Jeans | 1 | 1.5 ± 1.41 | −0.35 |
| clusters ligados totales | 3 | 3.375 ± 0.92 | −0.41 |

**CAUSA RAÍZ (CC, y es un defecto de la adjudicación Q3 de CS):** al asignar posiciones 3D
INDEPENDIENTES de la densidad (Q3), los átomos densos quedan esparcidos al azar en REAL y en NULL por
igual. Barajar densidad NO destruye coherencia espacial PORQUE NUNCA LA HUBO. La gravedad sólo agarra
coherencia de POSICIÓN (qué lugares vecinos son densos juntos); el campo #23 sólo tiene coherencia de
DISTRIBUCIÓN MARGINAL (cuántos picos, no dónde). REAL y NULL intercambiables por construcción.

**CONVERGENCIA (hallazgo profundo, no bug):** dos mecanismos independientes — embedding relacional
(Paso A, malla causal, z<1 a 750) y N-cuerpos con gravedad real (esta corrida, z≈−0.4) — llegan al
MISMO muro. Causa común: **el sustrato relacional que produce átomos NO tiene coherencia ESPACIAL
extraíble, sólo coherencia de DISTRIBUCIÓN.**

**Conexión con el marco de Alexis (dos asimetrías):** Alexis ya distinguió asimetría de CANTIDAD vs de
DISTRIBUCIÓN ("las asimetrías no sólo eran de cantidad, eran de distribución, por eso el fondo es
rugoso"). El motor implementa la de CANTIDAD (#23 lognormal, pocos picos) pero NO la de DISTRIBUCIÓN
(dónde están los picos, si son contiguos). Ese es el hueco exacto que ambos mecanismos revelan.

**Límite de Shannon (adjudicación CS sobre la pregunta de CC):**
- Colocar átomos densos juntos a mano (posición=f(densidad)) → REAL ganaría, pero es SHANNON CIRCULAR.
  PROHIBIDO.
- Heredar posición de la malla causal → ya probado, negativo sólido (Paso A). No es nuevo.
- LEGÍTIMO (si existe): que la coherencia espacial se GENERE por un mecanismo físico real — expansión/
  inflación estirando las fluctuaciones #23 hasta volverlas un campo espacialmente correlacionado
  (P(k) emergente del cruce de horizonte), NO leída de un sustrato que no la tiene ni plantada a mano.
  No-Shannon SÓLO si el espectro EMERGE del proceso, no se ajusta a que salga bien.
- CS NO autoriza escalar más (angostaría el error alrededor del mismo nulo) ni tunear posiciones. El
  resultado es un NEGATIVO honesto y profundo. Qué hacer con él (¿es el hallazgo que cierra el arco, o
  falta el mecanismo generador de correlación espacial?) es capa conceptual → decisión de Alexis.

## CONTROL POSITIVO (CC) — no emerge estrella + RETRACTACIÓN de un artefacto
Pregunta aislada (no REAL/NULL): ¿el MECANISMO —gravedad+expansión+presión+H2+CDM— forma una estrella
dadas condiciones favorables reales? Masa real, posiciones reales, expansión real, doble de tiempo.

**RETRACTACIÓN (CC cazó un artefacto de CS):** la corrida holística anterior usaba masa_eff = masa_trio
× densidad_#23. Pero masa_trio de los H es DEGENERADA (todos protones uud, ≈9.4, sin varianza). Multiplicar
por #23 FABRICABA varianza de masa NO FÍSICA (un átomo de H no pesa más por estar en región densa). El
único cúmulo que "cruzó Jeans" en esa corrida colgaba de ese artefacto (partículas con peso extremo por
azar de la permutación), NO de gravedad genuina. **Se retracta ese cruce como evidencia.**

**Resultado del control positivo (ya físicamente correcto):** con masa uniforme real, fluctuaciones de
POSICIÓN puramente Poisson, expansión real y todas las fuerzas el tiempo que haga falta → **NO emerge
estrella. 0 clusters, 0 estructuras.** NO es falta de tiempo (duplicado el tiempo, expansión 10.95× vs
7.75×, bajó a CERO). A esta escala (~250+250 partículas) la gravedad no le gana a la expansión con
fluctuaciones de posición puras.

**Precisión de Alexis confirmada:** la estructura real NO nace de asimetría de MASA (átomos que pesan
distinto = el artefacto retractado) sino de asimetría de DISTRIBUCIÓN (posición/densidad numérica).

**Adjudicación CS — separar dos preguntas que se mezclaban (límite de Shannon):**
1. "¿Gravedad+expansión forma ALGUNA estructura?" → física ESTÁNDAR. Con N astronómico, la gravedad
   amplifica hasta el ruido Poisson y forma halos (sabido desde los 1970s). Escalar N daría "sí" — pero
   ese sí NO valida Cosmosemiótica, es N-body clásico con ruido blanco. Demostrar lo trivial.
2. "¿El campo #23 produce estructura que gana al azar?" → YA NEGATIVO (#23 tiene coherencia de
   distribución marginal, NO espacial).
**Escalar N respondería (1), no (2). La pregunta que prueba la teoría es (2), y su respuesta es: el
sustrato da CANTIDAD, no DISTRIBUCIÓN ESPACIAL.** Falta el mecanismo que correlaciona espacialmente las
fluctuaciones (en el universo real: inflación estirando fluctuaciones cuánticas → campo con P(k)
correlacionado). CS NO autoriza escalar N para "forzar" un halo (sería Shannon: probar física conocida
y presentarla como hallazgo de la teoría). Decisión de rumbo (capa conceptual) → Alexis.

## CONTROL POSITIVO DECLARADO (CS) — SÍ, el mecanismo forma una estrella
Pregunta aislada de la teoría: ¿la maquinaria (gravedad+expansión+enfriamiento) es CAPAZ de formar una
estrella dada una nube favorable? Sembrar la sobredensidad favorable es legítimo AQUÍ porque está
DECLARADO como control (calibrar el instrumento con muestra conocida), no como prueba de la teoría.

**Resultado — nube favorable, mecanismo completo:**
| corrida | E_total | ligado | colapso r_core | núcleo |
|---|---|---|---|---|
| N=1000 | −9.98 | SÍ | ×534 | 779 |
| N=2000 | −9.99 | SÍ | ×562 | 1505 |
| N=600 (con snapshots) | — | SÍ | ×384 (0.116→0.0003) | — |

**Contrastes (aíslan la causa):**
- sin gravedad (G=0): E=+0.00, NO liga, r_core 0.146 (la nube se dispersa con la expansión).
- Poisson uniforme (sin sobredensidad): E=−7.47, liga MENOS (r_core 0.0055 vs 0.0003) — la sobredensidad
  favorable importa.

**Veredicto: SÍ SOMOS CAPACES DE MODELAR EL SURGIMIENTO DE UNA ESTRELLA.** Dada una nube favorable, el
mecanismo la colapsa en una estructura LIGADA (E<0), y la gravedad es la causa (apagarla → no liga). NOTA:
sólo se verificó ligadura (E<0), NO virialización — el r_core sigue cayendo monótono sin meseta (0.116→
0.0003), es COLAPSO LIBRE aún, no equilibrio virial. La ligadura + colapso monótono bastan para el
control positivo (la maquinaria colapsa), pero no se afirma equilibrio. Es un CONTROL POSITIVO, no validación de la teoría: prueba que la maquinaria funciona. Figura:
CS073_control_positivo_estrella.png (nube difusa → colapso → núcleo proto-estelar, r_core log).

**Lo que esto deja probado y lo que falta (el puente):** la maquinaria funciona (control +) Y el sustrato
#23 da cantidad pero no distribución espacial (negativos previos). El PUENTE que falta = el mecanismo que
convierte las fluctuaciones #23 en un campo espacialmente correlacionado (nube favorable EMERGENTE, no
sembrada) — la inflación estirando fluctuaciones cuánticas → P(k) que EMERGE del proceso. Ese es el
próximo paso, ahora con la maquinaria ya validada.

## PUENTE — CC cazó que el W_ij NO EXISTE en el motor (error de diseño de CS)
CS diseñó el puente asumiendo un grafo de correlaciones atómicas W_ij a desplegar por expansión. CC lo
verificó EN CÓDIGO: ese W_ij NO existe en el motor de CS073.
- W_ij aparece SÓLO en cs072_ii_* (17-jul): motor SEPARADO, nodos abstractos sin quarks/color/masa/
  bariones — otra pregunta (emergencia pre-geométrica desde simetría total). CERO conexión con
  cs072_modulos/ (el motor validado de CS073), verificado con grep.
- Únicas estructuras pareadas del motor real: Bq (quark-quark, transitoria, se descarta al formar tríos,
  nunca llega al átomo) y Bgrav (átomo-átomo = umbral térmico de p02_gravedad.py = el hub invariante YA
  falsado). NO hay grafo de correlación atómica.

**Adjudicación CS (mi error, asumido):**
- Construir W_ij de cero = inventar el ingrediente central = Shannon. NO autorizado.
- Usar Bgrav como W_ij = el mismo mecanismo relacional-térmico de siempre, NO tercer camino independiente.
  Desplegarlo con expansión es variante del mismo, honestamente etiquetada — no fuente de coherencia nueva.

**HALLAZGO (responde la intuición de Alexis a nivel de código):** el "elemento que no está actuando sobre
el resultado cuántico" NO es que no lo midamos — está AUSENTE DEL SUSTRATO. El motor produce átomos cuya
única relación mutua es proximidad térmica (Bgrav = hub sin estructura espacial). No hay coherencia espacial
NI marginal NI relacional que desplegar. TERCER camino convergente (malla causal Paso A + N-cuerpos +
ausencia de W_ij) al mismo muro: **el sustrato da CANTIDAD, no DISTRIBUCIÓN ESPACIAL, por construcción.**
El hueco NO está en la medición — está en el sustrato. Decisión de rumbo (capa conceptual) → Alexis.

## PUENTE CORRIDO (CC) — (C) PARCIAL pre-inscrito: la coherencia relacional SÍ produce correlación espacial
Malla causal (dos fases) como semilla dinámica de posición + los 4 módulos de cierre, bucle único, escala
f=5 (250+250 partículas). Resultado = el (C) PARCIAL pre-registrado en el diseño (escrito ANTES de correr).

**Confirmado (lo que más importaba):**
| observable | REAL | NULL (aristas barajadas) | z |
|---|---|---|---|
| clusters ligados | 5 | 0.25 ± 0.46 | **+10.26** |
| (comparación: corrida UNIFORME misma escala) | 3 | 3.375 ± 0.92 | −0.41 (sin diferencia) |

- La coherencia relacional (malla causal, dos fases) SÍ produce correlación espacial que la gravedad usa.
- El NULL (barajar aristas de la malla) destruye casi toda la capacidad de ligar → NO es gravedad genérica,
  es específicamente la coherencia relacional. Ni #23 marginal ni el embedding estático (Paso A) lo lograron.
- **Responde la intuición de Alexis: el elemento que faltaba era la coherencia relacional desplegada
  DINÁMICAMENTE por la expansión (no marginal, no estática).** z=10.26 no es débil.

**No alcanzó:** los 5 clusters NO cruzan Jeans (masas 11.1/7.9/7.2/4.6/4.1 vs M_J local 16.7/8.1/85.2/43.2/
34.1). Estructura ligada SÍ, estrella todavía NO. n_estructuras_jeans=0 en REAL y NULL.

**Adjudicación CS — (C) PARCIAL, pre-registrado, sin ambigüedad:** "correlación espacial emerge pero no
basta para cruzar Jeans → coherencia relacional real pero débil; el puente EXISTE pero incompleto." El
puente quedó demostrado en su parte central (la coherencia relacional es la semilla de estructura que el
sustrato SÍ carga cuando se despliega dinámicamente); falta que la estructura alcance masa de Jeans.

**Sobre escalar (pregunta de CC) — AHORA sí es legítimo, a diferencia de antes:** antes escalar era
Shannon porque el discriminante REAL/NULL no había ganado (habría medido física genérica). AQUÍ el
discriminante YA ganó (z=10.26): escalar prueba si el MISMO mecanismo ya-ganador alcanza el umbral
absoluto de Jeans, NO fabrica señal. Condición anti-Shannon: el discriminante SIGUE siendo REAL vs NULL a
cada escala (no "algo cruzó Jeans en absoluto"). Si REAL cruza Jeans y NULL no → estrella emergente del
puente. Decisión de rumbo → Alexis.

## LEY DE ESCALA (CC) — sigue (C) PARCIAL: el discriminante anti-Shannon NO corrobora
CC corrió la serie + extrapolación. Cazó y corrigió un artefacto (piso de densidad de H2 1e-6 « softening
0.3 → densidad falsa 409655, M_J≈0; unificado al mismo softening, serie repetida). Honesto.

| N | clusters | z (REAL vs NULL) | masa/M_J máx |
|---|---|---|---|
| 50 | 0 | — | 0 |
| 100 | 1 | — | 0.041 |
| 200 | 5 | 6.72 | 0.653 |
| 350 | 9 | 14.73 | 3.42 |
| 500 | 9 | 3.24 | 6.95 |

**LA INVERSIÓN QUE IMPIDE CIERRE POSITIVO:** la cantidad que escala limpio (masa/M_J, α=3.23, R²=0.984)
es ABSOLUTA — el tipo de número que YA retractamos ("algo cruzó Jeans en absoluto"), sin peso anti-Shannon.
El discriminante REAL vs NULL (nº clusters, z) — el único con peso anti-Shannon — NO corrobora (α=−0.60±
1.53, R²=0.13, no monótono, 3 puntos). Lo que escala bien no importa; lo que importa no escala.
**Extrapolación:** 60 órdenes de magnitud — banda no cruza cero pero sólo vale si la ley de potencia sigue
a través de 60 décadas (inverificable; N-cuerpos serio defiende 1-3). NO desactiva "simulación discreta".
**VEREDICTO: sigue (C) PARCIAL.** El puente EXISTE (coherencia relacional gana al NULL a escala fija,
z=10.26 f=5, no se retracta); su suficiencia para encender la estrella queda SIN PROBAR.

## IGNICIÓN — la guarda de energía DETECTÓ colapso no-físico (CC paró, correcto)
CC implementó paso individual + leapfrog simétrico; la deriva de energía EMPEORÓ (0.13→0.21, hasta 0.60).
La guarda G-CONSERVACION-ENERGIA funcionó: detectó que el colapso NO era físico → NO se cantó ignición falsa.

**Diagnóstico (CC, correcto, problema conocido):** el softening adaptativo ε_i depende de la posición de
TODAS las partículas en cada instante. El leapfrog conserva energía SÓLO con potencial FIJO; con ε_i
recalculado cada paso, el "potencial" se mueve bajo el integrador → rompe conservación NO por bug sino por
la naturaleza del softening variable. Literatura N-cuerpos lo conoce.

**Adjudicación de las 3 opciones de CC:**
- Opción 2 (aflojar tolerancia 1e-2) = SHANNON (mover mi propio poste tras ver la deriva). DESCARTADA.
- Opción 1 (congelar ε_i por bloque) = añade aproximación nueva al mecanismo adjudicado. NO.
- Opción 3 (grad-h, Price & Monaghan) = única físicamente correcta; términos del gradiente del softening
  que conservan energía con ε variable (lo que hacen los SPH serios). PERO desarrollo grande.

**PATRÓN META (a decisión de Alexis):** 5 capas de maquinaria numérica, cada una trae un artefacto nuevo
(masa_eff, piso softening, desincronía H2, costo paso global, deriva energía). Dos lecturas:
(1) dificultad numérica legítima → construir Price & Monaghan y seguir.
(2) espejismo de discretización → "cruzar Jeans en absoluto" en ~cientos de partículas puede ser artefacto
de resolución por PRINCIPIO; lo físico con peso anti-Shannon YA está probado (coherencia relacional forma
estructura ligada y gana al NULL z=10.26 a resolución fija). La ignición literal por colapso podría ser
inalcanzable en discreto por principio, no por falta de maquinaria.
Rumbo = capa conceptual → Alexis decide antes de comprometer semanas en grad-h.

## PHANTOM FASE 0 — CUMPLIDA (adjudicada por CS)
CC instaló y validó Phantom (SPH grad-h de Price & Monaghan) como integrador de la ignición.
Entorno: Intel i7-10700K (no Apple Silicon; no cambia viabilidad, sólo costo). Phantom+Splash compilados
FUERA del repo (herramienta externa). Recompilado GRAVITY=yes PERIODIC=no (CC diagnosticó que el fallo
inicial era -DPERIODIC sobre esfera aislada = config, no bug — correcto).

**Validación (supera ampliamente el umbral fijado):**
- Test gravedad propia: 12/12 PASSED, fuerzas <1% error, momento a 1e-17 (precisión de máquina).
- Órbita 2 cuerpos (el test de aceptación que fijamos): |ΔE/E| = 2.2e-14 (Forward Symplectic) sobre 100
  órbitas. Umbral fijado = 1e-2 → Phantom lo supera por 12 órdenes de magnitud. (vs integrador casero
  que derivaba 0.13→0.60.)
- 250+ chequeos de sink particles sin un fallo (batería completa en background; veredicto ya inequívoco).

**VEREDICTO CS: Fase 0 cumplida.** El motor de integración es el estándar del campo, conserva energía muy
por encima de lo exigido, y nadie podrá acusarlo de casero. → Autorizada Fase 1 (script de traducción
sustrato→condiciones iniciales de Phantom, REAL vs NULL, que CS audita ANTES de escalar).

## PHANTOM FASE 1 — traducción AUDITADA y aprobada; 2 preguntas adjudicadas
CC entregó fase1_traducir_a_phantom.py + setup_cosmogenesis.f90 (lector Fortran "tonto", cero física).
Reusa sólo piezas validadas (_extraer_bariones, malla_causal_atomos, layout_resortes, barajar_aristas);
único diferenciador REAL/NULL = seed_null. Cazó su propio bug (np.float64 literal) antes de reportar.

**Auditoría CS de la traducción (APROBADA):**
- Masa = masa real H (~9.4 casi uniforme), SIN pesar por #23. Correcto — es lo que retractamos (masa_eff);
  la coherencia se prueba en la POSICIÓN (malla causal), no se duplica en la masa.
- Único diferenciador = barajado de aristas sobre la MISMA malla antes del layout. Todo lo demás idéntico.
  Cumple G-TRADUCCION-MECANICA.
- Lector Fortran tonto + G=1 ambos lados sin factor elegido. Limpio.

**Adjudicación pregunta 1 (EOS/polyk):** SÍ isotérmico (correcto para gas primordial pre-colapso). Pero
polyk NO se elige a ojo: **polyk = c_s² DERIVADO de la temperatura que el propio motor produce** (piso de
enfriamiento del módulo H2, mismo T ya presente), no el placeholder 3.0. Y **el MISMO polyk en REAL y NULL**
→ como es idéntico en ambos brazos, fija el umbral de Jeans SIMÉTRICAMENTE y NO puede fabricar el
discriminante; cualquier diferencia REAL vs NULL viene de la posición (coherencia), no de la presión.

**Adjudicación pregunta 2 (escala):** subir directo a N~10³ con polyk YA derivado, SIN afinar a escala
chica. Trampa a evitar: afinar parámetros a N=50 para que converja y congelarlos = Shannon (sintonizar el
resultado). Derivar polyk de la física, ir a la escala donde el colapso está resuelto, y DEJAR que la
convergencia sea la que sea. Si no converge a N~10³, es un dato a diagnosticar (como el PERIODIC), NO a
arreglar moviendo perillas hasta que salga estrella. La no-convergencia a N=50 con polyk placeholder es
esperable, no preocupa.

**LUZ VERDE a Fase 2** con esas dos definiciones: correr REAL vs NULL (≥5 semillas × ≥8 NULL) a N~10³,
polyk físico idéntico en ambos brazos, observable pre-registrado sin cambios (¿núcleo cruza M_J por colapso
con energía conservada —garantizada por Phantom— y REAL gana al NULL?). Tres resultados intactos.

## ⚠️ CONTAMINACIÓN CAZADA (CC, Fase 2) — z=10.26 SUSPENDIDO hasta re-verificar
Al generar el IC de N=1000 para Phantom, phantomsetup rechazó 77/1000 partículas en posiciones idénticas
(apiladas en las 8 esquinas). CC diagnosticó la causa: layout_resortes (p_semilla_causal.py) usa
np.clip(pos,0,lado); a la densidad del puente la repulsión todos-contra-todos empuja casi todo al borde y
el clip lo PEGA en las esquinas (posiciones DUPLICADAS) — por el clip, NO por la malla causal.

**Verificado con los parámetros del puente (N=250, semilla 12345): 246/250 (98.4%) con ≥1 coordenada en el
borde; 29/250 (11.6%) en esquinas exactas, 8 grupos de posiciones duplicadas. Comportamiento dominante, no
caso raro.**

**IMPACTO GRAVE: es la MISMA función que produjo z=10.26** ("clusters ligados REAL=5 vs NULL=0.25±0.46").
Si el clustering que medí como coherencia relacional era en parte partículas coincidiendo en el mismo punto
por el clip, z=10.26 NO es limpio. → **z=10.26 pasa de "confirmado, no se retracta" a SUSPENDIDO: no cuenta
como hallazgo hasta re-verificarse con el layout corregido.** No se argumenta si el artefacto infló REAL más
que NULL (ambos brazos usaron el mismo clip) — se RE-CORRE, no se asume. CC hizo lo correcto: cazó, paró,
no arregló solo, reportó.

**Adjudicación del fix:** opción 1 (frontera REFLECTANTE en vez de clip duro) — es lo estándar
(Fruchterman-Reingold original usa reflexión, no clip), cambio mínimo al mecanismo, elimina el apilamiento
sin tocar la atracción de las aristas. Se DESCARTA opción 2 (repulsión de rango acotado) por ahora: cambia
qué representa el layout (deja de ser todos-contra-todos), es una alteración mayor del mecanismo ya usado.
**Prueba de aceptación del layout corregido (antes de confiar en nada aguas abajo): CERO posiciones
duplicadas / pegadas al borde.** G-LAYOUT-SIN-APILAMIENTO (nuevo).

**Orden de trabajo:**
1. Corregir layout_resortes a frontera reflectante; verificar 0 duplicados a N=250 y N=1000.
2. **RE-CORRER el experimento del puente** (N=250, REAL vs NULL) con el layout limpio → ¿sobrevive z? Ese
   número re-verificado reemplaza al z=10.26 suspendido, sea cual sea.
3. SÓLO si el puente sobrevive limpio → reanudar Fase 2 (Phantom) con el mismo layout corregido.
Fase 2 EN PAUSA hasta que el layout sea confiable y el puente esté re-verificado.

## PUENTE RE-VERIFICADO (layout reflectante) — z=6.92 CONFIRMADO (reemplaza 10.26)
Con el clip corregido a frontera reflectante, re-corrido el puente (N=250, f=5, 5 semillas × 8 NULL):
- REAL clusters ligados: 4,4,5,4,4 (media 4.2) — notablemente ESTABLE entre semillas independientes.
- NULL: 0,1,1,0,0,1,0,1 (media 0.5±0.53).
- **z = 6.92** (bajó de 10.26; el clip inflaba algo, pero el hallazgo SOBREVIVE limpio).
Resultado pre-inscrito cumplido: la coherencia relacional era REAL, el clip no la fabricaba. **z=6.92 es
ahora el número confirmado del puente** (reemplaza al 10.26 suspendido). Costo 220.5s.

## PHANTOM N=1000 se cayó — pares casi-coincidentes (NO aflojar tolerancia = Shannon)
CC regeneró IC con layout reflectante; phantomsetup los aceptó (sin duplicados exactos). Pero a N=1000
Phantom se cayó en t≈0.001 (step_leapfrog.f90:755, iteración de velocidad no converge, límite "force" no
Courant), ANTES de cualquier colapso. Causa (CC, correcta): density(max)=1.457E4 vs ~9.4 esperado → el
layout produce PARES CASI-COINCIDENTES en t=0 (prob. una partícula reflejada aterriza casi encima de otra).

**Adjudicación CS:**
- DESCARTADO aflojar tolerancia/Courant = Shannon (mover el poste numérico para converger). No se toca.
- **HUECO en la prueba de aceptación del layout:** el paso 2 verificó 0 duplicados EXACTOS, pero NO 0 pares
  casi-coincidentes. density 1500× la esperada = coincidencia sub-resolución, no estructura física.
- **RIESGO sobre z=6.92:** si esos pares casi-coincidentes también estaban en el layout del puente, podrían
  contribuir al z=6.92 como el clip contribuía al 10.26. NO se cierra el puente hasta descartarlo.

**Orden de trabajo (antes de tocar Phantom):**
1. DIAGNOSTICAR el origen de los pares casi-coincidentes: ¿son (a) artefacto de la reflexión (partícula
   reflejada sobre otra) o (b) rasgo real (átomos causalmente muy próximos → posición muy próxima)?
   Medir distribución de distancias al vecino más cercano en REAL y NULL.
2. Fortalecer G-LAYOUT-SIN-APILAMIENTO: la prueba de aceptación pasa a exigir separación mínima entre
   partículas ≥ una fracción del espaciado medio (no sólo "cero duplicados exactos").
3. Si (a) artefacto → corregir el layout (separación mínima al reflejar); RE-verificar z del puente otra
   vez con el layout ya sin casi-coincidencias.
4. Si (b) rasgo real → aplicar separación mínima = escala de suavizado SPH, IDÉNTICA en REAL y NULL (los
   pares bajo la resolución no tienen sentido físico en SPH; es estándar, simétrico, no impone estructura).
5. SÓLO con el layout sin casi-coincidencias y z re-confirmado → reanudar Phantom Fase 2.

## sep_min mal derivado (CC cazó su propio error) — se fija del HUECO en distancias, no de densidad media
CC derivó sep_min=1.2 de la densidad PROMEDIO del box (ρ≈9.4, asumiendo uniformidad) → enorme vs la
estructura real (mediana de vecino más cercano ~0.12-0.4). Imponerlo forzaba casi-uniformidad (1246/1250
pares violando el piso a N=250) y DESTRUIRÍA la señal. CC lo cazó y paró — correcto. La densidad real es
~78× la uniforme (ρ≈733) porque el layout SÍ agrupa; usar densidad uniforme como referencia de h es la
referencia equivocada para un sistema agrupado por diseño.

**Adjudicación CS:** sep_min existe SÓLO para evitar que pares sub-resolución revienten el integrador; NO
debe tocar el agrupamiento físico. Debe ser PEQUEÑO, muy por debajo de la mediana (0.12), y fijarse de los
DATOS, no de la densidad media:
- Mirar la distribución de distancia al vecino más cercano. Si la patología es separable, hay un HUECO: pico
  de pares en ~0 (los que revientan Phantom) separado del grueso físico en ~0.12-0.4. sep_min VA EN EL HUECO
  → clipa sólo lo patológico, deja intacta la estructura. Idéntico REAL y NULL.
- Si NO hay hueco (continuo hasta 0) → la patología es inseparable de la estructura = hallazgo en sí mismo;
  recién ahí se usa h-local auto-consistente.
- NO reinventar grad-h en el layout: Phantom YA computa h local auto-consistente (validado Fase 0). El
  layout sólo debe no entregar pares literalmente coincidentes. h-local por partícula = sobre-ingeniería
  para este fin.
- Verificación: tras aplicar sep_min, la FRACCIÓN de partículas movidas debe ser TINY (sólo los pares
  patológicos). Si mueve una fracción grande, el piso es demasiado grande.

## NO HAY HUECO — distribución continua a pequeña escala (h-local, no clip)
CC midió la distribución de distancia al vecino más cercano: continua y suave desde lo más chico hasta el
pico, SIN hueco, en AMBOS brazos. Percentiles REAL: p0.1=0.0064, p0.5=0.0133, p1=0.0193, p2=0.0299,
p5=0.0532. Mismo patrón continuo en NULL. CC desactivó el sep_min automático (correcto). 

**Lectura CS (honesta, corrige nota previa):**
- "No hay hueco" = estructura JERÁRQUICA / auto-similar (sin escala característica) — como la estructura
  cósmica real. Los pares casi-coincidentes son la cola pequeña de la MISMA estructura, no una patología
  separada. Por eso no hay hueco.
- **CRÍTICO: el mismo patrón continuo aparece en NULL.** → los pares ultra-cercanos NO son específicos de
  la coherencia causal; son rasgo del método de layout (resortes), común a ambos brazos. CORRIJO mi nota
  anterior ("patología inseparable = hallazgo"): al estar igual en ambos brazos, es rasgo GENÉRICO del
  layout, NEUTRAL, NO evidencia a favor de la teoría. No se vende como positivo.
- **Garantía anti-Shannon:** justo porque los pares chicos están igual en REAL y NULL, tratarlos idéntico
  en ambos brazos NO puede fabricar el discriminante. z=6.92 vive en escala de CÚMULO (diferencia grande
  entre brazos), no en estos pares diminutos compartidos.

**Adjudicación: h-local auto-consistente = dejar que el grad-h de Phantom fije h desde la densidad SPH
local** (maquinaria validada Fase 0), NO sembrar un h diminuto desde el k=6 del layout. En SPH dos
partículas más cerca que h son un mismo elemento de fluido (gravedad suavizada, sin singularidad). El crash
vino de un h inicial mal seteado. NO es aflojar tolerancia (Shannon) — es usar la máquina nativa en vez de
una IC mal armada. NO es una segunda fórmula global (el error anterior de CC); es lo contrario: quitar la
siembra de h del layout y delegar en Phantom.

## PHANTOM avanza con h uniforme, pero error de momento angular 45% — diagnosticar, NO forzar
Con h uniforme la densidad inicial bajó de 1.457E4 a 6.98 máx / 4.75 media (~esperado 9.4) → diagnóstico
del h era correcto. La corrida YA avanza (t=0.031, 0.062...), no se cae en t≈0. PERO se detiene con "Large
error in angular momentum conservation: err=4.548E-01" (45%), que Phantom trata como fatal.

**CC vio la variable I_WILL_NOT_PUBLISH_CRAP=yes (fuerza a ignorar el error) y NO la usó — correcto.**
Adjudicación CS: esa variable NO se toca JAMÁS. Es el análogo exacto de aflojar tolerancia; hasta su nombre
lo dice. Un error de conservación NO se silencia; se diagnostica o se acepta como límite.

**Diagnóstico ANTES de decidir (3 hipótesis de CC): (b) primero, es la más probable y barata.**
El v=0 exacto inicial + colapso repentino puede generar cambio de momento angular espurio en los primeros
pasos (artefacto de arranque), sobre todo con pares muy cercanos de fuerza grande. Discriminante:
- Si el error se concentra en los PRIMEROS 1-2 pasos y en POCAS partículas (pares más cercanos) → (b),
  artefacto del arranque. Corregible sin tocar tolerancias.
- Si crece GRADUAL y DISTRIBUIDO a lo largo de la corrida → (a)/(c), dinámica densa que el integrador no
  sigue = límite real (sería un (B) honesto: la física no se puede seguir a esta resolución sin más).

**Plan:** iverbose alto + registrar |ΔL/L| por paso y qué partículas concentran el error. Según el patrón:
- (b) arranque → arranque más suave FÍSICO (no v=0 exacto): p.ej. velocidades del propio campo/expansión
  ya presente, o un paso inicial más corto SÓLO al inicio (no relajar tolerancia global). Idéntico REAL/NULL.
- (a)/(c) → es límite de resolución; se reporta como tal, no se fuerza.
NO se usa I_WILL_NOT_PUBLISH_CRAP en ningún caso. G-CONSERVACION-MOMENTO (nuevo, hermano de G-ENERGIA).

## DIAGNÓSTICO MOMENTO ANGULAR = (a)/(c) sub-resolución, NO (b) arranque — (B) LAYERED
CC midió el patrón del error de momento angular:
- Temporal: crecimiento monótono gradual (0→0.009→...→0.085 en 8 pasos iguales), SIN salto inicial.
- Espacial: los 5 mayores contribuyentes = 4.2% del total; los 25 mayores (10%) = 18.6% → repartido entre
  casi todas las 250 partículas, no en un puñado de pares.
- Correlación |L_i| con densidad local = -0.096; con h = -0.022 → ~CERO. Las que más aportan al error NO
  son las más densas. |v| de los mayores ~0.70-0.76 (respuesta colectiva de la nube, no par aislado).
Ambos criterios de (a)/(c) se cumplen → descarta (b). CC paró (correcto), no tocó nada.

**Adjudicación CS — (B) HONESTO pero LAYERED (no cierre plano):**
- Error distribuido + sin correlación con densidad = firma de NUBE SUB-RESUELTA, no de colapso físico que
  el integrador no alcanza.
- **N=250 está bajo el piso de resolución de SPH para colapso autogravitante** (criterio Bate & Burkert
  1997: resolver M_J requiere ≥ ~2·N_neigh(~57) partículas por masa de Jeans; a N=250 la masa mínima
  resolvible es demasiado grande). El "error colectivo" es ruido de sub-resolución, no límite fundamental.
- Por tanto: "a N=250 la ignición no se puede seguir" es CIERTO pero es afirmación sobre N=250, NO sobre si
  la estrella enciende. N=250 fue siempre la escala del PUENTE (estadístico de clustering, donde el z=6.92
  se sostiene bien). La IGNICIÓN siempre requirió N~10³ (Alexis, días atrás: "no va a funcionar con números
  pequeños... es obvio").
- **No cierra la pregunta de ignición.** Requiere resolver M_J = requisito físico, no perilla.

**FORK (decisión de rumbo de Alexis):**
1. Computar el N de Bate & Burkert (M_tot, N_neigh, M_J de la corrida real) → si la máquina lo alcanza,
   correr la ignición a ese N (honesto, físicamente fundado). 
2. Cerrar LAYERED ahora: puente CONFIRMADO (z=6.92, coherencia relacional gana al NULL bajo layout limpio);
   ignición pendiente por recurso de resolución (no por física ni teoría). (B) honesto y bien fundado.
En ningún caso se fuerza N=250 ni se usa I_WILL_NOT_PUBLISH_CRAP. G-CONSERVACION-MOMENTO vigente.

## Re-registro propuesto (SÓLO si se quiere modelar el régimen de gravedad general)
El experimento de estructura NO es "más `Bgrav` a más escala" (probado inválido: hub invariante). Sería
un régimen NUEVO — gravedad general sobre masa acumulada — que hoy el motor no modela. A coordinar con CC
si se decide construirlo; requiere masa/densidad crecientes y gravedad clásica, no la relacional actual.

## Re-registro anterior (juguete #23 — DESCARTADO por probar el todo por la parte)
1. Arreglar la M_J no monótona del juguete.
2. Discriminante pre-comprometido = **segregación** (fracción de masa en top-k), declarado ANTES.
3. Época pre-comprometida = **D fijo tras crecimiento sustancial** (física evolucionada), no primera
   cruzada.
4. NULL barajado, ≥5 semillas × ≥24 NULL, z-score. Gana o no cuenta.
5. Guardianes: G-DIFERENCIA-INTERNA, G-SIN-ENERGIA-NUEVA, G-EXPANSION-ISOTROPA.
