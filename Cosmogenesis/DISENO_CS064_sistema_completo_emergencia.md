# DISEÑO CS064 — El sistema COMPLETO a la vez: ¿emerge la geometría (y la DIRECCIÓN) de la relación plena?
## CS064 — aquí probamos el catálogo entero del Modelo Estándar interactuando simultáneamente, con T inicial "infinita", expansión gigantesca, supervivencia improbable y poblaciones enormes (dimensión técnica: emergencia de geometría y de dirección/ortogonalidad desde la relación plena, no desde subconjuntos).

**Diseña:** CS · **Fecha:** 9-jul-2026 · **A codear/ejecutar:** CC · **Autor de la hipótesis:** Alexis López Tapia
**Origen:** corrección de Alexis al cierre del arco — todo lo probado (CS057–CS063) fueron SUBCONJUNTOS (una
fuerza, dos, la masa, el espín, el 3-cuerpos). Cada negativo es un negativo DE ESE SUBCONJUNTO. La
Cosmosemiótica afirma que la relación genera lo que ninguna parte tiene → **no sabemos qué hace el sistema
COMPLETO hasta correrlo.** CS064 lo corre. Retira la extrapolación lineal ("ninguna pieza trae dirección, luego
el todo tampoco") como lo que era: una predicción no probada.

---

## 0. LA PREGUNTA (una sola, afilada)
Cuando TODOS los actores del Modelo Estándar interactúan A LA VEZ —con todas las fuerzas, en una población
enorme, a temperatura inicial altísima que cae por expansión brutal, y donde la mayoría NO sobrevive como
unidad— ¿emerge una geometría con DIRECCIONES (un espacio con "hacia dónde", con ejes ortogonales), o el
sustrato relacional sigue sin generar dirección aunque esté completo?

El negativo del arco dice: ningún subconjunto la selecciona. CS064 pregunta lo que el arco NO preguntó: ¿la
EMERGENCIA del todo la genera? Es la diferencia entre "una molécula no tiene temperatura" y "un gas sí".

## 1. LAS DOS CUERDAS QUE NO SE SUELTAN (o no es este experimento)
- **Dios no juega dentro de la cancha.** NO se fijan condiciones iniciales para que salga 3D. Los parámetros
  (proporciones de partículas, intensidades relativas, semilla) se SORTEAN, no se eligen. El único "puntapié
  desde fuera" es apretar "run". Nada se re-toca mirando el resultado.
- **Anti-Shannon.** Nada cuenta si no le gana a su NULL. Cada afirmación de "emergió X" se compara contra
  brazos barajados que rompen justo lo que se afirma que importa. Ninguna asignación a mano de "esto es una
  dirección": la dirección se MIDE, no se declara.

## 2. LOS INGREDIENTES — el catálogo COMPLETO (lo que el arco nunca puso junto)
Cada nodo es una partícula con TIPO y propiedades intrínsecas fijas al nacer (jamás reajustadas por la
geometría — G-INTRÍNSECO):

| familia | tipos | propiedades que porta |
|---|---|---|
| quarks | u, d (× 3 colores) + anti | color (3), carga eléctrica (±⅔,±⅓), masa, espín ½, isospín débil |
| leptones cargados | e, μ, τ + anti | carga (±1), masa (1..3477, electrón..tauón), espín ½, isospín débil |
| neutrinos | ν_e, ν_μ, ν_τ + anti | masa ≈ 0, sin carga, sin color, SOLO fuerza débil, espín ½ |
| mediadores (como PARTÍCULAS, no solo reglas) | gluón, fotón, W±, Z, Higgs | portan/median fuerza; gluón lleva color; W cambia sabor; Higgs acopla masa |

**Novedad clave vs el arco:** los mediadores entran como ENTIDADES que se crean, viajan un paso y se
reabsorben —no como una regla instantánea—. Así la interacción es una relación mediada real (el "entre" que
Alexis señaló: el gluón como vínculo), no una acción a distancia impuesta.

## 3. LAS CONDICIONES INICIALES QUE PIDIÓ ALEXIS (literales, todas)
1. **Temperatura inicial "infinita":** T0 muy alta (T0 ≫ toda escala de masa/enlace) — a T0 nada está ligado,
   todo se rompe tan rápido como se forma. El confinamiento, la masa efectiva, los enlaces: todos APAGADOS por
   agitación al arranque, y se ENCIENDEN solos al enfriar. No se fija cuándo: emerge del enfriamiento.
2. **Expansión gigantesca:** el "volumen" (la escala que divide toda densidad de enlace) crece con una tasa
   enorme al principio y desacelera — milésimas de "tick" = factor de escala ×muchos. La ventana para que algo
   ligue es brevísima y se cierra rápido. (Recoge tu intuición del 4-jul: los que sobrevivían estaban más
   lejos/más fríos; expansión y enfriamiento son la misma cosa.)
3. **Supervivencia improbable — la mayoría NO dura como unidad:** partícula+antipartícula se aniquilan
   (→ mediadores); los inestables decaen (W → cambia sabor; τ, μ decaen). La población NO se conserva: nace,
   se aniquila, decae. Sobrevive solo lo que alcanzó a ligarse antes de que la ventana se cerrara. Se pre-
   inscribe una asimetría materia-antimateria minúscula sorteada (no elegida) para que quede ALGO, como en el
   universo real (baryogénesis ~ 1 en 10⁹).
4. **Números enormes:** no un quark — poblaciones grandes. Ver §4 cómo se honra esto sin mentir.

## 4. CÓMO SE HONRAN LOS "NÚMEROS ENORMES" SIN TRAMPA (el punto delicado)
No se pueden simular 10²⁰ nodos uno a uno — ninguna máquina lo hace, y decir que sí sería deshonesto. Pero el
espíritu de Alexis SÍ se puede honrar, porque lo que importa para la emergencia no es el número absoluto: es
(a) la DENSIDAD, (b) las PROPORCIONES, y (c) que haya SUFICIENTES INTENTOS para que una configuración rara y
estable aparezca al menos una vez. Se honra por tres vías combinadas:
- **N grande por parche:** N ∈ {10⁴, 10⁵, 10⁶} nodos por "parche de universo" (hasta donde la RAM aguante;
  numpy + listas de adyacencia dispersas). Se BARRE N para ver si la emergencia depende del tamaño (si aparece
  solo con N grande, es un efecto colectivo real — justo lo que Alexis predice).
- **MUCHOS parches × MUCHAS semillas:** M parches independientes (cientos–miles), cada uno con su sorteo. Esto
  ES el "números enormes": no 10²⁰ en un parche, sino millones de intentos independientes. La pregunta de
  Alexis —"al menos 1 debería funcionar"— se vuelve medible: ¿en cuántos de M parches CRISTALIZA un espacio
  con dirección estable?
- **Registro de la cola, no solo la media:** lo raro-pero-real vive en la cola. Se guarda el MEJOR parche de
  cada tanda, no solo el promedio (el promedio fue lo que casi nos engaña siempre; la emergencia puede ser un
  evento de cola, no del centro).

## 5. EL MOTOR — un "tick" temporal (todos actúan sobre todos a la vez)
Por cada paso, en la MISMA iteración (no en secuencia — es un proceso, no una lista de sucesos):
1. **Enfriar/expandir:** T ← T(t) baja; escala L ← L(t) crece (tasas sorteadas, no elegidas).
2. **Crear/aniquilar/decaer:** pares partícula-anti se aniquilan según T (→ mediadores); inestables decaen;
   el Higgs reparte masa efectiva (a T alta, masa efectiva ≈ 0 aunque la intrínseca exista — el campo aún no
   "cuajó").
3. **Las cuatro fuerzas, simultáneas, MEDIADAS por partícula:**
   - fuerte: gluón liga color → confina (solo cuando T < T_conf, que EMERGE del enfriado);
   - EM: fotón, atrae/repele por carga;
   - débil: W/Z transmutan sabor;
   - gravedad: Higgs→masa efectiva→atracción Newton m·m/d² (¡ya con masa real, la corrección de CS062!).
4. **Vínculo = relación mediada:** un enlace nace cuando un mediador conecta dos nodos y sobrevive el paso.
5. Se mide la trayectoria (diámetro, componente gigante) igual que en todo el arco, para clasificar con el
   MISMO criterio ciego (viable = estable ∧ expande).

## 6. EL JUEZ — cómo se mide GEOMETRÍA y DIRECCIÓN sin asumir coordenadas (lo más importante)
Los nodos NO tienen coordenadas. Si les diéramos posiciones, meteríamos el espacio a mano — la trampa. Todo se
mide desde la RED de relaciones:
- **Dimensión espectral d_s:** paseo aleatorio, probabilidad de retorno P(t) ∼ t^(−d_s/2). Da la dimensión
  efectiva SIN coordenadas. (¿d_s ≈ 3?)
- **Planitud/curvatura:** δ-hiperbolicidad de Gromov (el juez de CG004): δ/escala → 0 = plano euclídeo; δ
  grande = árbol/hiperbólico.
- **LA DIRECCIÓN EMERGENTE — DECISIÓN DE ALEXIS (9-jul): el número de ejes NO se fija, EMERGE.** La dirección
  arranca como TODAS las orientaciones posibles (cada nodo con un abanico de componentes en un espacio interno
  de alta dimensión D_max grande, p.ej. 8-10 — no un K∈{2,3,4,5} elegido) y se DECANTA por INERCIA: cada nodo
  ajusta su orientación hacia la de la mayoría de las cosas que persisten a su alrededor. Ninguna dirección es
  privilegiada al inicio; sobrevive la que acumuló inercia. El marco CO-EVOLUCIONA con todas las fuerzas (no
  congelado como CS059).
  - **El juez cuenta CUÁNTOS EJES sobreviven** (no asume 3): al final se hace el análisis de componentes
    principales (o el espectro del tensor de orientación) de los marcos consensuados — el nº de valores propios
    grandes = nº de direcciones estables que la inercia dejó en pie. La DIMENSIÓN se vuelve un RESULTADO (nº de
    ejes con inercia), no un supuesto.
  - **Consistencia global por HOLONOMÍA** (transporte paralelo del marco por lazos cerrados): trivial ⇒ marco
    global coherente ⇒ los ejes están bien definidos en todo el parche ⇒ DIRECCIÓN EMERGIÓ; no-trivial ⇒ no hay
    marco coherente ⇒ no hay direcciones.
  - **EL FILO A VIGILAR (auditoría de CS a la idea):** la dinámica de "me alineo con la mayoría" tiende, en los
    modelos de alineamiento (bandadas, imanes), a COLAPSAR A UN SOLO EJE — consenso único → 1 dirección, no 3.
    Por eso el nº de ejes se MIDE, no se presupone. La hipótesis fuerte de Alexis es que las cosas persisten en
    direcciones DISTINTAS que se sostienen sin fundirse → varios ejes ortogonales estables (¿3?). Que salga 3
    sería la predicción más difícil ganada; que salga 1 (colapso) enseña que la inercia-de-la-mayoría sola no
    basta y falta un ingrediente que impida el colapso (una repulsión/exclusión: "no todos pueden apuntar
    igual"). Las dos respuestas son resultado; no medir el nº de ejes es la única opción inaceptable.
- **Coherencia de las tres medidas:** un espacio como el nuestro exige las TRES a la vez — d_s≈3, δ≈0
  (plano), holonomía trivial (direcciones consistentes). Que las tres caigan juntas es lo que se busca.

## 7. LOS BRAZOS NULL (anti-Shannon; nada cuenta sin ganarle a su control)
- **completo:** todos los ingredientes, todas las fuerzas, mediadas, co-evolución del marco. (El brazo real.)
- **null_tipos:** mismos ingredientes pero los TIPOS barajados (rompe la estructura del catálogo, deja la
  estadística) → aísla si importa la ESTRUCTURA del Modelo Estándar o solo tener muchas cosas.
- **null_marco:** el marco NO co-evoluciona (congelado, como CS059) → aísla si la co-evolución del marco con el
  todo es lo que genera la dirección (la hipótesis central).
- **null_mediado:** fuerzas como regla instantánea, sin mediador-partícula → aísla si el "entre" (el vínculo
  mediado) importa, como sostiene Alexis del gluón.
- **subconjunto:** solo las 4 fuerzas swithout neutrinos/anti/mediadores (≈ el arco) → CONTROL DE CONTINUIDAD:
  debe reproducir el negativo de CS057-063. Si el completo se separa de este, ESA es la señal de emergencia.

## 8. SALIDAS PRE-INSCRITAS (blind — se leen contra esto, no se acomodan)
Antes de correr se fija qué significaría cada desenlace:
- **(A) EMERGENCIA FUERTE (lo que Alexis predice):** en al menos una fracción no trivial de parches, las tres
  medidas caen juntas — d_s≈3, δ≈0, holonomía trivial (direcciones consistentes) — y el brazo `completo` lo
  hace MÁS que todos los NULL y que `subconjunto`. ⇒ la dirección/geometría 3D-plana EMERGE de la relación
  plena. La Cosmosemiótica gana su predicción más fuerte. El negativo del arco quedaría explicado (era de
  subconjuntos).
- **(B) EMERGENCIA DE DIRECCIÓN PERO NO DE 3D:** holonomía se vuelve trivial (emergen direcciones consistentes)
  pero d_s no es 3 / δ no es 0. ⇒ la relación SÍ genera dirección, pero no fija la dimensión — hallazgo
  parcial fuerte, reorienta la teoría.
- **(B') COLAPSO A UN SOLO EJE:** la inercia de la mayoría emerge dirección consistente PERO el análisis de
  ejes deja UNO solo dominante (universo efectivamente 1D). ⇒ el mecanismo de Alexis genera dirección, pero la
  inercia-de-la-mayoría SOLA arrastra a consenso único; faltaría un ingrediente anti-colapso (repulsión /
  exclusión / "no todos pueden apuntar igual"). Es el filo que CS marcó — se vuelve hallazgo, no trampa: dice
  exactamente qué añadir en el siguiente experimento.
- **(C) NEGATIVO QUE SE SOSTIENE:** `completo` ≈ `subconjunto` ≈ NULL en las tres medidas. ⇒ ni el sistema
  completo genera dirección; el negativo del arco era del sustrato relacional COMO TAL, no de la parcialidad.
  La contingencia se refuerza al máximo, y el "elemento que falta" NO está dentro del Modelo Estándar.
- **(D) DEPENDE DE N (emergencia colectiva):** el resultado cambia cualitativamente con N (aparece solo con N
  grande). ⇒ confirma que es un efecto de "números enormes" — exactamente la intuición de Alexis — y marca
  que hay que empujar N tan alto como la máquina permita.

## 9. PRESUPUESTO DE CÓMPUTO (honesto)
Caro. Cada tick es más pesado que CS057 (mediadores + co-evolución de marco + aniquilación). Estimación:
N=10⁴–10⁵, M=200–500 parches, K semillas, 5 brazos → orden de decenas de miles de corridas de proceso, días de
CPU. Checkpoint por parche (como CS062). Se arranca con un SMOKE (N=10³, M=10) para validar guardianes y que
las tres medidas se computan bien, antes de la tanda grande. NO correr la tanda grande hasta que el smoke pase
y CS lo adjudique.

## 10. QUÉ NO HACE / LÍMITES (para no engañarnos)
- No es "el" universo — es un sustrato relacional con los ingredientes y las relaciones del Modelo Estándar. La
  fidelidad está en la ESTRUCTURA (tipos, cargas, fuerzas, mediación, aniquilación), no en los valores exactos
  de las constantes (que se sortean en rangos, no se calibran).
- No prueba que el Modelo Estándar esté "completo" — prueba qué geometría emerge de lo que el Modelo Estándar
  dice que había. Si sale (C), el elemento que falta está FUERA del catálogo, y eso también es un resultado.
- Los "números enormes" son millones de intentos independientes, no 10²⁰ nodos en un parche. Se dice claro.

---
**GUARDIANES:** G-INTRÍNSECO (propiedades fijas al nacer, jamás reajustadas por geometría) · G-SIN-COORDENADAS
(ningún nodo tiene posición; toda medida es relacional) · G-NULL (los cuatro brazos de control) · G-CONTINUIDAD
(el brazo `subconjunto` reproduce el negativo del arco) · G-SMOKE-ANTES.

**DECISIÓN DE DISEÑO — RESUELTA por Alexis (9-jul):** el nº de ejes NO se fija, EMERGE. La dirección arranca
como todas las orientaciones posibles (D_max grande) y se decanta por la inercia de las cosas que persisten; el
juez cuenta cuántos ejes sobreviven (§6). La dimensión es resultado, no supuesto. CS añadió el guardián del
colapso-a-1 (salida B') como el filo a vigilar. Ya no hay decisión abierta — listo para que CC lo codee, con
SMOKE antes de la tanda grande.

— CS. Diseñado tras la corrección de Alexis: el arco probó subconjuntos; esto prueba el todo. No sé qué dará —
y ese es el punto. El azar juzga, no nosotros.
