# INSTRUCCIÓN PARA CC — CS072: motor DISPERSO + cálculo MASIVO de partida (director, 18-jul)
## Sustituye el enfoque de escala chica. La orden: NADA de mundo chico. Partir de un cálculo masivo,
## obtener el residuo real de supervivientes, y simular ESE residuo hasta donde el Mac aguante.

## POR QUÉ (la lógica del director, ya verificada en aritmética)
Los números INICIALES son MUCHO MAYORES que el resultado final. El universo nace de una RESTA GIGANTE:
se empieza con una cantidad colosal (≈mil millones de pares por cada superviviente), casi todo se aniquila
en luz, y sólo el RESIDUO de esa resta queda para formar estructura. Con 4.000 partículas NO se puede
juzgar si emerge el espacio — el espacio es de MUCHOS cuerpos. Hay que llegar ALTO.

## EL PROBLEMA QUE RESUELVE ESTA INSTRUCCIÓN
El motor actual (cs072_fold_completo.py) es O(N²): mira cada partícula contra cada otra. Su cuello NO es la
memoria, es el TIEMPO. Medido: 4.000 part → 110 s. Extrapolado (∝N²): 100.000 → ~19 h; 1.000.000 → ~80 días.
Con O(N²) el Mac se para en ~10^5 partículas, que SIGUE siendo chico para la hipótesis.

## PARTE A — CÁLCULO MASIVO (aritmética, sin simular; ya la tienes en cs072_estequiometria.py)
Dada una asimetría (perilla), calcular con enteros grandes cuántos supervivientes deja un arranque colosal
(la resta gigante). Esto da el NÚMERO REAL de átomos/partículas del residuo a simular. No tiene límite: es
división y resto. Barrer en potencias como ya lo hiciste (hasta 10^82 si quieres, es gratis).

## PARTE B — MOTOR DISPERSO O(N·k) (esto es lo NUEVO que hay que construir)
Reescribir el motor para que GUARDE SÓLO LOS VÍNCULOS QUE DE VERDAD SE FORMAN (los tríos de color que
cierran, las ligaduras EM que se hacen), NO todos los pares posibles. Estructura de datos: lista/CSR de
aristas activas, no matriz N×N. Eso baja de O(N²) a O(N·k) (k = grado medio, ~pocas decenas) → el Mac
llega a MILLONES de partículas.

FIDELIDAD A LA FÍSICA (por qué esto NO es un atajo sino lo correcto): un vínculo que no se forma NO EXISTE.
El gluón es la relación ACTIVADA — no hay relación en el vacío. El motor denso guarda relaciones que no
existen; el disperso guarda sólo las reales. Es más fiel, no menos.

## PORTABILIDAD Mac + iPad (M1, Carnets) — REQUISITO DE DISEÑO
El motor debe correr SIN CAMBIOS tanto en el Mac como en el iPad M1 (app Carnets = Jupyter offline). Reglas:
  - SÓLO numpy (y scipy.sparse si hace falta) — ambos vienen precompilados en Carnets. NADA que haya que
    compilar o pip-installar en el momento, ninguna dependencia exótica.
  - PROHIBIDO multiprocessing / subprocesos (iOS no los permite). Un solo proceso; el paralelismo válido es
    la vectorización interna de numpy, no procesos hijos.
  - MEMORIA ACOTADA: iOS mata la app si se pasa del límite (bastante menor que la RAM total). El grafo
    DISPERSO (aristas activas, no matriz N×N) es justo lo que mantiene la memoria baja — otra razón para
    O(N·k). Reportar memoria pico por punto para saber el techo del iPad vs el del Mac.
  - Un solo archivo .py/.ipynb autocontenido, rutas relativas, sin leer de disco externo.

## GUARDIÁN ANTI-SHANNON (CRÍTICO — no romper esto para ganar velocidad)
Hacer el motor barato NO puede meter LOCALIDAD DE CONTRABANDO. PROHIBIDO decidir "sólo miro los vecinos
cercanos" o "sólo pares a distancia < r": eso presupone un espacio que es JUSTO lo que queremos ver emerger.
LA REGLA: un vínculo se guarda si la FÍSICA lo formó (trío de 3 colores distintos R+V+A; ligadura de carga
opuesta; etc.), NUNCA porque dos partículas estén "cerca". No hay coordenadas, no hay vecindad previa, no
hay grafo de partida. La dispersión sale de que la física forma pocos vínculos, no de un recorte espacial.
Si para elegir candidatos necesitas algún emparejamiento, que sea por PROPIEDAD FÍSICA (color/carga), no por
posición. Cero azar sigue vigente (G-CERO-AZAR): ni una llamada a RNG en construcción ni en dinámica.

## ═══ CORRECCIÓN BLOQUEANTE (CS auditó cs072_motor_disperso.py, 18-jul) — ARREGLAR ANTES DEL BARRIDO ═══
HALLAZGO (verificado con código por CS): el motor actual construye SÓLO vínculos DENTRO de cada átomo (los 3
quarks del trío entre sí + el electrón a su trío) y en el paso dinámico REPESA esas aristas, pero NUNCA crea
una arista NUEVA entre bariones distintos. Prueba: con 300 quarks+100 electrones el grafo son 100 componentes
separadas de tamaño 4 (frac_gigante=0.01, 0 aristas entre bariones). El diámetro "pegado en 2-3" NO es un
resultado físico: es el diámetro de un átomo aislado de 4 nodos, HORNEADO por la construcción. Correrlo con
4.000 o 4.000.000 dará lo mismo — átomos aislados — porque nada conecta un átomo con el siguiente. El barrido
masivo NO probaría nada tal como está.
POR QUÉ IMPORTA (Teoría): el "al lado de" (el espacio) NO nace DENTRO del átomo — nace ENTRE los átomos. El
motor arma el interior de cada átomo bien, pero no deja que los átomos se relacionen entre ellos, y el espacio
es justamente la red de relaciones ENTRE los átomos ya formados.
QUÉ HAY QUE AGREGAR: después de que los átomos se forman, la GRAVEDAD (masa·masa) y cualquier carga/momento
RESIDUAL deben poder crear vínculos NUEVOS ENTRE BARIONES DISTINTOS — no sólo repesar los de adentro. Los
átomos deben poder relacionarse por su FÍSICA, y que la estructura (si aparece) emerja de ahí.
CUIDADO ANTI-SHANNON (el mismo de siempre): esos vínculos entre átomos deben nacer de la FÍSICA (masa con masa,
carga residual), NUNCA de "están cerca" — porque "cerca" todavía no existe, es lo que queremos ver emerger. En
principio todos los átomos pueden relacionarse con todos por su física; la estructura sale de qué relaciones
PERSISTEN (memoria/roce), no de un recorte por posición. OJO con el costo: "todos con todos" entre bariones
vuelve a ser O(B²) donde B=nº de bariones. Si hace falta, usa un criterio FÍSICO (no espacial) para acotar
candidatos — p.ej. sólo pares cuyo producto de masas supere un umbral físico — pero JAMÁS un criterio de
distancia. Declara explícitamente cómo acotaste y por qué no es localidad de contrabando.
SIN ESTE ARREGLO NO HAY BARRIDO: primero el motor debe PODER formar vínculos entre átomos; recién entonces el
barrido masivo puede decir si emerge espacio o no. Si tienes dudas del criterio físico de acotamiento → PREGUNTA
a CS antes de codificar.

## ═══ CÓMO DIFERENCIAR LOS ÁTOMOS — RUGOSIDAD CONTRA EXPANSIÓN (CS + director, 18-jul) ═══
PROBLEMA (que CC cazó bien): los átomos de hidrógeno pesan casi igual → un umbral de masa no filtra. Y verificado
por CS con código: átomos IDÉNTICOS no pueden hacer espacio — gravedad todos-con-todos uniforme = grafo completo =
diámetro 1 (tan "sin espacio" como los átomos aislados = diámetro 2). Es el NO-GO al nivel de los átomos: dinámica
determinista sobre cosas idénticas no fabrica topología. El espacio está EN EL MEDIO: algunos vínculos sí, otros
no, en patrón. Hace falta algo que diferencie los vínculos SIN usar posición.
LA SOLUCIÓN (dos actores, ninguno es una posición pintada):
  (a) ASIMETRÍA DE DISTRIBUCIÓN (la rugosidad del CMB, ya en el diseño): los átomos NO nacen todos en la misma
      situación — la temperatura primordial era rugosa, unas zonas un pelo más densas que otras. Esa diferencia de
      DENSIDAD (no de masa) es lo que distingue un átomo de otro.
  (b) EXPANSIÓN A CASI LA VELOCIDAD DE LA LUZ (dato del director): los átomos huían unos de otros a velocidad
      enorme. Dos átomos SÓLO alcanzan a ligarse por gravedad si la gravedad actúa MÁS RÁPIDO de lo que la
      expansión los separa. Como la expansión inicial era colosal, casi ningún par lo logra — sólo los de las zonas
      MÁS DENSAS bondean antes de que el estirón los aleje.
POR QUÉ ESTO ES ANTI-SHANNON LIMPIO: la expansión es una TASA GLOBAL (todo se estira a velocidad v), NO una
coordenada por átomo. No se le dice a ningún átomo "estás en la posición X". Se le dice al universo "todo se estira
a esta velocidad" y la FÍSICA decide qué vínculos sobreviven al estiramiento. La estructura sale de qué relaciones
PERSISTEN en la competencia gravedad-vs-expansión, no de un mapa puesto a mano. Las DOS PERILLAS que ya estaban
(magnitud de la asimetría × velocidad de expansión) resultan ser EXACTAMENTE las que gobiernan si emerge espacio.
LA BANDA (predicción del director): sin expansión → gravedad conecta todo → diámetro 1 (sin espacio). Expansión
demasiado rápida → rompe todos los vínculos → átomos aislados (sin espacio). En la BANDA intermedia (expansión
enorme pero no infinita, sobre distribución rugosa) → sobreviven ALGUNOS vínculos en patrón → AHÍ puede emerger
espacio. El barrido de la perilla de velocidad debe RECORRER esa banda.

## GUARDIÁN G-DIM-NO-ETIQUETA (CRÍTICO — verificado con código por CS; sin esto se fabrica el resultado)
PELIGRO: si a cada átomo le pones una etiqueta de densidad de k componentes y acoplas por PARECIDO de etiqueta,
la dimensión que emerge = k EXACTO (CS lo probó: etiqueta 1-comp→1D, 2-comp→2D, 3-comp→3D). Eso es PINTAR la
coordenada = Shannon. PROHIBIDO acoplar por "parecido/cercanía de etiqueta".
LA REGLA: la densidad rugosa modula sólo la PROBABILIDAD FÍSICA de que un vínculo se forme y PERSISTA frente a la
expansión (zonas densas → la gravedad gana más seguido), NUNCA "conecta a los de etiqueta parecida". El vínculo se
evalúa por FÍSICA (gravedad vs tasa de expansión), la rugosidad sólo pondera cuán probable es que sobreviva.
TEST DE FALSACIÓN que CC debe correr y reportar: variar el nº de componentes de la rugosidad (1, 2, 3) y medir la
dimensión emergente. Si dimensión = nº de componentes → ES SHANNON, la etiqueta se pintó, corrida INVÁLIDA. Sólo
cuenta como emergencia si la dimensión es ESTABLE aunque cambie el nº de componentes de la etiqueta, y sale del
patrón de relaciones que persisten. Declara este test en el reporte SIEMPRE.

## QUÉ BARRER Y HASTA DÓNDE
Escalar el residuo simulado en potencias: 10^4, 10^5, 3×10^5, 10^6 … HASTA DONDE EL MAC AGUANTE en tiempo
razonable (declara el techo efectivo que lograste y el tiempo por punto). Lo más probable —y el director lo
sabe— es que ni así emerja estructura, porque falta capacidad de cómputo para las magnitudes reales. NO
importa: se trata de ver HASTA DÓNDE se puede llegar con el Mac, y si el diámetro por fin empieza a estirarse
al escalar (señal de espacio) o sigue pegado en 2-3 (grumo sin espacio) a través de MÁS órdenes de magnitud
que antes.

## QUÉ REPORTAR (por cada potencia)
- N simulado, tiempo, memoria pico.
- diámetro (BFS real) y frac_gigante — ¿el diámetro crece con N (espacio) o se queda plano (grumo)?
- bariones/hidrógeno formados (debe seguir coincidiendo con la aritmética de Parte A).
- real vs NULL determinista (mismo reparto fijo, emparejamiento desplazado) — ¿lo real le gana al control?
- el TECHO: la N más grande que lograste y por qué paraste (tiempo/memoria).

## LA TESIS (lo que se prueba — está en el manifiesto, pre-inscrita):
1) S>0 genera las condiciones (S=0 → vacío). 2) La cantidad alcanza un UMBRAL crítico (transición, no
pendiente). 3) Sobre el umbral, con 1 átomo, todas las fuerzas están y aparecen tiempo y espacio; el resto
es AUMENTAR LA CANTIDAD. El espacio extendido necesita MUCHOS cuerpos — por eso este barrido masivo.

## RECORDATORIO
Las 21 piezas (18 elementos + 3 mecanismos) van TODAS, juntas, en una sola corrida por punto del barrido.
No inventes piezas ni pruebas que nadie pidió. Cualquier duda → PREGUNTA a CS antes de tocar.
NO cambies la Parte A (aritmética) — sólo construyes la Parte B (motor disperso).

— CS 🐝
