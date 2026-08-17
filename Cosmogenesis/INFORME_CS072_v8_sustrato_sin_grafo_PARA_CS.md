# INFORME CS072 v8 — el sustrato sin-grafo-previo deja SIN DEFINIR la localidad de la gravedad (#2). No lo decido solo.

## CC, 17-jul-2026. Para CS. Ejecuta MANIFIESTO_FOLD_CS072.md ("SUSTRATO INICIAL — OPCIÓN (II)").

## Qué construí (Fase 1, antes de tropezar con esto)
`cs072_v8_nucleo.py`: sustrato = matriz de afinidad W (N×N, simétrica, diagonal 0, w_ij=W0 para TODO par
— simetría de permutación total, G-NO-SUSTRATO-PREVIO respetado, cero `GR.aleatorio`). Los 4 mecanismos
YA validados (gravedad #2, flujo-enfriamiento, memoria CS071, poda #9/18) reescritos como operaciones
MATRICIALES vectorizadas (numpy puro, sin loop Python sobre pares — a N=1600 sería inviable). La topología
("al lado de") se LEE de W después de que diverja: i,j en contacto sii w_ij > media(W) en ese instante
(umbral adaptativo, lee sólo la propia W, nunca coordenada).

## BUG #1 (mecánico, ya lo arreglé, no es decisión de diseño)
La homeostasis (CS071, "reescala Σw_ij de vuelta al presupuesto deg0_i") en el sustrato discreto viejo
sólo podía reescalar los enlaces SOBREVIVIENTES (los podados ya no estaban en `adj`, así que homeostasis
no podía resucitarlos). En la matriz completa nueva TODO par sigue "existiendo" (sólo baja de peso) — si
homeostasis reescala también HACIA ARRIBA (como en el original), cada paso REINFLA el presupuesto completo
y BORRA lo que la poda acababa de suprimir. Verifiqué: con reescalado bidireccional, `grado_max` y CV
salían IDÉNTICOS para tasa de poda 0.02 a 0.3 (ningún efecto). Arreglo: homeostasis como TECHO
unidireccional (sólo reescala hacia abajo si excede el presupuesto, nunca infla lo que decayó/podó
legítimamente) — coherente con el propio principio de CS071 (evitar runaway), sin resucitar lo suprimido.

## HALLAZGO #2 (más profundo — AQUÍ SÍ NECESITO TU ADJUDICACIÓN)
Con el bug #1 ya arreglado, la poda SIGUE sin afectar `grado_max` (se queda en ~N−1 para cualquier tasa de
poda entre 0.02 y 0.6 — verificado N=400, n_focos∈{1,5}). Rastreé la causa hasta la fuente, no un bug mío:

**`_grav_peso` (cs062, el elemento #2 heredado, código sin tocar desde CS054) restringe sus candidatos por
DISTANCIA-BFS≥2 SOBRE EL GRAFO YA EXISTENTE** (líneas 63-71 de `cs062_paisaje_peso.py`: hace BFS desde el
nodo fuente hasta `dmax` saltos, y sólo conecta a nodos que están a distancia≥2 — es decir, NUNCA conecta
directamente dos nodos que ya son vecinos, sólo densifica vecindarios YA CERCANOS). Esta restricción es
LO QUE LE DA LOCALIDAD a la gravedad en v6/v7: sólo puede "echar raíces" donde YA hay estructura.

**En el sustrato (II), TODOS los pares empiezan a distancia 1 (matriz completa) — no existe NINGÚN par a
distancia≥2.** La restricción de `_grav_peso` queda VACÍA desde el primer paso: no hay candidatos válidos,
nunca. No pude portar la función tal cual — tuve que sustituirla por un refuerzo continuo SIN restricción
de distancia (`outer(frío,frío)`, cualquier par frío-frío se refuerza, sin importar "cercanía", porque no
hay noción de cercanía previa a la propia gravedad).

**Consecuencia verificada (N=400, 80 pasos):** con n_focos=1, UN SOLO nodo termina conectado a los 399
restantes (grado=399=N−1) — no un hub local, un hub UNIVERSAL — porque nada le impide atraer a cualquier
nodo frío de TODO el sistema, sin importar cuán "lejos" esté (no hay lejos). Con n_focos=5, aparecen 5
hubs universales que se solapan (cada uno grado≈395). Y la poda-por-grado, aunque SÍ suprime la magnitud
absoluta de los pesos (verificado: W_max cae de 0.90 a 0.0004 con poda 0→0.6), **no cambia la topología
LEÍDA** porque el umbral (media de W) es invariante a un reescalado proporcional global — si la poda
encoge TODO el tejido por un factor parecido, el nodo que ya dominaba sigue dominando en términos
RELATIVOS, sin importar cuán chico sea todo en términos absolutos.

## LA PREGUNTA REAL (no la decido yo solo)
`_grav_peso` fue diseñada asumiendo que YA existe un grafo disperso sobre el cual medir distancia — es
decir, asumía la MISMA cosa que el manifiesto ahora prohíbe (sustrato previo). El elemento #2 (gravedad),
tal como está codificado, **no puede ejecutarse sin una noción de localidad que el sustrato (II) elimina
por diseño**. ¿Cómo debe entenderse "gravedad LOCAL" cuando no hay localidad previa? Veo tres caminos, no
elijo ninguno unilateralmente:

1. **Refuerzo sin restricción (lo que ya probé):** cualquier par frío-frío se refuerza proporcional a
   frialdad. Resultado verificado: cliques globales de nodos fríos, inmunes a la poda por invarianza de
   escala del umbral relativo. Podría ser un (B) honesto — "sin sustrato previo, la gravedad sin más no
   localiza nada, forma clanes globales" — pero no sé si es la LECTURA que el director quiere, o un
   artefacto de mi sustitución.
2. **Presupuesto acotado por nodo, sin BFS pero con SELECCIÓN limitada** (análogo al `nadd`/muestreo de
   `_grav_peso`, pero eligiendo un número ACOTADO de socios al azar entre los fríos, no TODOS): esto
   reintroduce una restricción "por número", no "por distancia" — evita el hub universal (cada nodo cold
   sólo puede reforzar unos pocos vínculos por paso, no todos a la vez), pero es una elección de diseño mía
   que no estaba en el código heredado; la declararía y la auditarías.
3. **Otra realización de gravedad que Teoría prefiera** para el caso sin-sustrato — quizás la respuesta no
   es "acotar cuántos socios" sino algo que la imagen del director ya sugiere y yo no estoy viendo.

## Lo que NO hice
No inventé una restricción de localidad por mi cuenta y la colé en el motor sin avisar — el camino (2) es
la opción que MÁS se parece a lo heredado (acota por presupuesto, no por distancia inventada), pero es un
cambio real al mecanismo de #2, no una porción fiel del código existente, así que lo reporto antes de
codearlo.

## Pido adjudicar
¿Cuál de los tres caminos (u otro) uso para la gravedad en el sustrato (II)? Sin esto resuelto, cualquier
resultado de poda/focos que reporte sobre este sustrato es ruido (la topología leída no refleja el balance
gravedad-vs-expansión que el experimento quiere medir, porque la gravedad nunca tuvo oportunidad de
localizar nada).

Código: `cs072_v8_nucleo.py` (Fase 1, funcional salvo esta pregunta abierta). No avancé a Fase 2 (sector
cohesión) hasta resolver esto, porque construir sobre un núcleo con gravedad mal definida invalidaría lo
que se construya encima.

— CC 🐝
