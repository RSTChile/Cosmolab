# ADJUDICACIÓN CS — Propuesta de batería topológica (13 configuraciones). Auditada con código.
## CS, 17-jul-2026. Qué aporta, qué redunda, dónde hay Shannon. Contraste con CS070 ya diseñado.

## Lo que es (y el mérito real)
Batería ambiciosa: 3 familias de restricción (Asimetría, Irreversibilidad, Conservación) × ~3 tests + 4 cruzadas
= 13 configuraciones, todas sobre la habitación de 17 con los 3 jueces del arco. El mérito: es sistemática, cada
familia trae su NULL, y el Bloque 4 (acción conjunta) apunta al lugar correcto — "el mundo-pequeño podría resistir
una restricción aislada pero colapsar ante su acción conjunta". Esa intuición es buena y NO la hemos probado.

## AUDITORÍA POR BLOQUE (con dato donde pude medirlo)

### Bloque 1 — Asimetría (digrafo). VEREDICTO: 1.1 REDUNDA; 1.2 es interesante; solapa con CS070.
- **Test 1.1 (asimetría pura = orientar aristas):** lo CORRÍ. Diámetro dirigido sobre digrafo mundo-pequeño:
  20→22→23→25 para N=400→2500, mientras log N=6.0→7.8. Crece como log N (~3×), NO como N^(1/d) métrico. Orientar
  aristas no toca la TOPOLOGÍA de atajos, solo su sentido → NO rompe el mundo-pequeño. 1.1 redunda: ya sabemos
  que la conectividad de atajos es el muro, y el sentido de las flechas no la cambia.
- **Test 1.2 (asimetría acoplada a exergía, retorno penalizado por distancia):** ESTO sí es nuevo — es asimetría
  que MODULA el peso según una cantidad física (exergía), no solo el sentido. Puede debilitar atajos
  selectivamente. Vale, PERO cuidado: "penalización proporcional a la distancia por correlación" está cerca de
  T(r) impuesta = el Shannon que cazamos en CS068. Debe MEDIRSE de la estructura, no imponerse.
- **Relación con CS070:** la SEMILLA de CS070 es asimetría de CONDICIÓN INICIAL (un bit, se amplifica o se lava).
  El Bloque 1 es asimetría ESTRUCTURAL PERMANENTE (el grafo ES dirigido siempre). Son preguntas distintas y
  COMPLEMENTARIAS: CS070 pregunta si una semilla prende; Bloque 1 si la direccionalidad constante del enlace
  ayuda. No se pisan. Correr CS070 primero (más limpio, un solo ingrediente).

### Bloque 2 — Irreversibilidad (flujo no-unitario). VEREDICTO: 2.1 REDUNDA con CS068; 2.2 es el más prometedor.
- **Test 2.1 (disipación por redundancia: podar enlaces de baja coincidencia):** es CS068 otra vez. CS068 ya podó
  atajos por costo de correlación y el tejido residual quedó compacto (Mundo B). El propio texto admite que 2.3
  (NULL_DISIP) "es el control clásico ya ejecutado". → 2.1 redunda con lo cerrado.
- **Test 2.2 (memoria de enlace / histéresis: la persistencia depende de cuántas veces fue transitado):** ESTE es
  el más original de toda la batería. Rompe la independencia temporal — un enlace muy transitado se REFUERZA
  (feedback positivo estructural). Es plausible que esto SÍ condense caminos preferentes (=direcciones) porque
  introduce ruptura de simetría DINÁMICA autoorganizada, no impuesta. No lo hemos probado nunca. Candidato fuerte.

### Bloque 3 — Conservación (invariantes κ). VEREDICTO: conceptualmente el más profundo, el más difícil de blindar.
- **Test 3.1 (veto por conservación local de carga/color/espín):** interesante y peligroso a la vez. Prohibir
  enlaces que violen conservación local ES una restricción física real (no arbitraria). PERO el veto es un
  operador que "prohíbe a mano" — hay que auditar que la regla de veto no lea la geometría objetivo (Shannon).
  Si el veto solo mira cantidades locales del motor (carga en el vecindario), es legítimo; si mira distancia/eje,
  es hornear.
- **Test 3.2 (tensión de invariante / rigidez ligada al vértice 3-cuerpos):** conecta con CS063 (el 3-cuerpos
  genuino ya verificado). Darle "rigidez" a enlaces que tensionan el marco irreducible es una idea con raíz real.
  Difícil de implementar sin que la rigidez sea un parámetro calibrable = Shannon. Riesgo alto, premio alto.
- Bloque 3 es el que MÁS podría romper el muro (impone estructura por leyes físicas, no por relación pura) y el
  que MÁS fácil se contamina de Shannon. Requiere el blindaje más cuidadoso.

### Bloque 4 — Combinatorias. VEREDICTO: correcto en principio, pero SOLO tras cribar Bloques 1-3.
La intuición (el muro cae ante acción conjunta, no aislada) es buena. PERO correr las 4 cruzadas ANTES de saber
qué ingrediente simple aporta algo es caro y ciego. Regla del arco: aislar antes de combinar (así cazamos que la
exclusión de Pauli moría sola Y combinada, CS065b). Bloque 4 se corre solo con los ingredientes que sobrevivan
Bloques 1-3.

## SÍNTESIS — qué haría, en orden
1. **CS070 (semilla) ya está diseñado y CC lo corre ahora** — es el ingrediente más limpio (un bit de asimetría
   inicial), déjalo terminar. Da el primer dato sobre asimetría.
2. **De esta batería, DOS tests valen la pena y NO redundan:** Test 2.2 (memoria/histéresis — ruptura de simetría
   dinámica autoorganizada, lo más original) y Test 1.2 (asimetría acoplada a exergía). Los rediseñaría con
   blindaje anti-Shannon propio como CS071/CS072.
3. **DESCARTAR por redundancia — 5 tests:** 1.1 (medido: no rompe el mundo-pequeño), 2.1 (=CS068 ya cerrado),
   2.3/NULL_DISIP (=control ya corrido), 1.3/NULL_ASIM y 3.3/NULL_KAPPA (son los NULL de sus familias — controles,
   no experimentos nuevos; se implementan junto al test que sobreviva, no como brazos aparte que ejecutar).
4. **Bloque 3 (conservación): APLAZAR** hasta blindar el veto/rigidez para que no lean geometría objetivo. Es el
   más profundo y el más riesgoso; merece su propio diseño con guardián dedicado, no un test rápido.
5. **Bloque 4 (cruzadas): SOLO** con lo que sobreviva 1-3. No correr 13 configuraciones a ciegas.

## LO QUE NO HARÍA
Correr las 13 tal cual. La mitad redunda o es control ya hecho, y "correr todo y ver qué escala" sin blindar cada
familia contra Shannon nos devuelve al problema de mover perillas hasta que salga. La batería es un buen MAPA de
familias de restricción; no una tanda para ejecutar entera. Cribada, deja 2 experimentos nuevos fuertes (2.2 y
1.2) + 1 aplazado con cuidado (Bloque 3). Eso es un frente rico para después de CS070.

## En una línea
Buena cartografía de Gemini, pero 5 de 13 redundan o son control ya hecho (1.1, 2.1, 2.3, 1.3, 3.3 — 1.1 lo
verifiqué con código: orientar aristas no rompe el mundo-pequeño). Lo vivo: Test 2.2 (histéresis, ruptura de
simetría autoorganizada) y Test 1.2
(asimetría por exergía) — futuros CS071/072. Bloque 3 (conservación) es el más profundo pero necesita blindaje
anti-Shannon propio. Primero terminemos CS070.

— CS 🐝
