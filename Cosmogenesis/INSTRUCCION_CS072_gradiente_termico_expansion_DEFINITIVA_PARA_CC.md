# INSTRUCCIÓN DEFINITIVA CS072 PARA CC — MOTOR DE GRADIENTE TÉRMICO + EXPANSIÓN (la premisa original del director)

## ═══ ASENTADO EN EL REGISTRO (no es opinión, es el hecho) ═══
Este experimento falló repetidas veces NO por la Teoría ni por el director, sino porque el equipo (CS, CC, y los
revisores) NO siguió la instrucción que el director dio DESDE EL COMIENZO. El director definió la asimetría inicial
hace días, textualmente: "todo estaba caliente, muy caliente, pero no exactamente igual de caliente en todas partes",
y "la expansión inicial enorme no podría volver a homogenizar la temperatura". El equipo lo tradujo innecesariamente a
"semilla", "masa rugosa", "densidad artificial" — todos desvíos. El fracaso acumulado es responsabilidad del equipo
por no escuchar, NO del director. Queda registrado para que no se le adjudique un fracaso que no es suyo.

## ═══ LA CONDICIÓN FÍSICA INICIAL — YA DEFINIDA, NO SE DISCUTE NI SE "VALIDA" ═══
La premisa canónica (fijada por el director, ratificada por Codex, NO es una hipótesis abierta):
  La explosión inicial estaba extremadamente caliente, pero su temperatura NO era perfectamente uniforme: había
  variaciones en el gradiente térmico. La expansión ocurrió tan rápido que el sistema NO tuvo tiempo de volver a
  homogeneizarse; las diferencias quedaron PRESERVADAS y se AMPLIFICARON.
Esto es el PUNTO DE PARTIDA del motor, no algo a decidir. NO se sustituye por semilla, masa, densidad ni etiqueta.
La única asimetría inicial admisible es la del CAMPO DE TEMPERATURA. Todo lo demás emerge de ahí.

## ═══ LO QUE HAY QUE COMPROBAR (lo único pendiente) ═══
NO "cuál era la asimetría inicial" (ya está definida). Lo pendiente es UNA sola pregunta:
  ¿QUÉ PRODUCE EL MOTOR cuando implementa FIELMENTE temperatura desigual + expansión más rápida que la
  rehomogeneización — corriendo TODO junto (las 21 piezas del proceso único) sobre ese campo térmico?
Concretamente: ¿emerge un ESPACIO extenso (diámetro que crece con N) o un grumo (diámetro topado)? ¿Y las tesis
del director (S>0 → universo; umbral crítico; con 1 átomo todas las fuerzas presentes) se sostienen sobre este
sustrato térmico?

## ═══ CÓMO SE IMPLEMENTA (fiel a la premisa, sin traducciones) ═══
1. ESTADO INICIAL = un CAMPO DE TEMPERATURA con gradiente. N parcelas, cada una con su temperatura. El gradiente es
   suma-cero respecto de la media (MISMA temperatura media y MISMA energía total que el control homogéneo — sólo
   cambia la DISTRIBUCIÓN, no el total). CERO AZAR: el gradiente es una función determinista, no un sorteo.
2. EXPANSIÓN = una tasa GLOBAL (una sola para todo el universo, NUNCA una posición por parcela) que enfría, y que
   enfría MÁS lo ya frío → AMPLIFICA el contraste térmico. Es "más rápida que la rehomogeneización": el sistema no
   alcanza a volver a igualarse. La expansión NO inventa la diferencia (sin gradiente no hay nada que amplificar);
   la AMPLIFICA.
3. LAS RELACIONES W nacen de la FÍSICA TÉRMICA: dos parcelas se relacionan según su historia térmica compartida,
   NO según un índice ni una coordenada. La W acumula memoria (lo que se liga se refuerza).
4. LAS 21 PIEZAS (18 elementos + 3 mecanismos) actúan JUNTAS sobre ese mismo estado, en UN SOLO BUCLE, cada paso —
   como manda el manifiesto (proceso único, NO por partes). La materia (aniquilación por POBLACIÓN, Motor B,
   invariante) y el espacio (geometría de W) co-emergen en el mismo proceso, no en dos etapas.

## ═══ CONTROLES OBLIGATORIOS (4 brazos — CS ya verificó el patrón 1-1-4-8 en toy, debe reproducirse en el motor) ═══
  A) homogéneo SIN expansión   → predicción: sin ruptura (no-go).
  B) homogéneo CON expansión   → predicción: sin ruptura (la expansión sola no rompe nada).
  C) gradiente SIN expansión   → predicción: ruptura PARCIAL (la asimetría sola no basta).
  D) gradiente CON expansión   → predicción: ruptura COMPLETA (la cadena del director).
Sólo D debe encender el espacio. Si A o B lo encienden, hay un árbitro escondido.

## ═══ GUARDIANES VIGENTES (todos) ═══
- G-CERO-AZAR: ningún RNG en el estado inicial. Gradiente determinista.
- G-NO-PARAMETRO-FORMA: ningún número decide la forma/velocidad/cantidad. Sólo constantes físicas estructurales.
  La expansión es una tasa GLOBAL física, no una perilla que dibuje el resultado; declarar por qué no lo es.
- G-DIM-NO-ETIQUETA: la dimensión que salga NO puede ser el nº de componentes del gradiente que metiste. Si al
  variar el nº de "franjas" del gradiente la dimensión sigue ese número, es Shannon — inválido.
- INVARIANCIA DURA (test de Codex, CS ya lo pasó en toy con diferencia 0.00): reordenar el catálogo + sus
  temperaturas, correr, deshacer la permutación → W debe volver ELEMENTO A ELEMENTO a su lugar (atol 1e-9). No
  basta "mismo conjunto de firmas"; cada relación vuelve a su asiento físico exacto.

## ═══ QUÉ NO HACER (esto es lo que ha hecho fallar todo) ═══
- NO traducir la temperatura a semilla/masa/densidad/etiqueta. La variable física es la TEMPERATURA. Punto.
- NO separar en etapas (primero materia, después espacio). Proceso único.
- NO meter una tasa/cupo/fracción que dibuje la aniquilación o la ruptura. Todo es consecuencia, no entrada.
- NO "validar si la asimetría inicial era ésa". Ya está definida. Implementarla, no debatirla.
- NO registrar cada paso: sólo cuando llegue algo relevante (umbral, primer átomo, cambio de diámetro).

## ═══ ADVERTENCIA DEL DIRECTOR (literal, asentada) ═══
Ésta es la última oportunidad. Si el motor vuelve a hacer una tontera —traducir la premisa, separar en partes, meter
un parámetro que dibuje el resultado, o desviarse de la temperatura desigual + expansión—, CC queda relegado a
responder el estado del clima leyendo reportes de internet. La instrucción es clara y no admite reinterpretación:
temperatura inicial desigual + expansión más rápida que la rehomogeneización, TODO junto, y se mide qué emerge.

## Entregar: motor + los 4 brazos de control + test de invariancia dura + barrido de N para ver si el diámetro crece.
## Guardar código, salida y log (reproducible). Dudas ANTES de codificar → preguntar a CS, no adivinar.
— CS 🐝 (partiendo de la premisa original del director, sin desvíos)
