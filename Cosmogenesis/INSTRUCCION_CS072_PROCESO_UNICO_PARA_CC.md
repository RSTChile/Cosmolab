# INSTRUCCIÓN DEFINITIVA PARA CC — CS072: EL PROCESO ÚNICO (todo junto, en cada paso, sin partes)
## Director, 18-jul. Esta instrucción DEROGA todo enfoque por partes anterior. NO hay Parte A / Parte B.
## NO hay "declarar piezas". NO hay regla de enlace pegada al final. Hay UN SOLO PROCESO.

## LA IMAGEN (del director, literal — este es el criterio de aceptación)
Un universo que arranca CALIENTE y DENSO, y en cada instante —el MISMO instante— se enfría un poco, se
expande un poco, y en ese enfriarse-y-expandirse las partículas que pueden ligarse se ligan y las que no se
separan; y el ESPACIO no es algo que se calcula aparte: es la FORMA que va tomando esa red de lo que quedó
ligado, instante a instante, mientras todo lo demás sigue pasando.

## LA REGLA MADRE (lo que se ha explicado decenas de veces)
TODO OPERÓ EN SECUENCIA A LA VEZ. No hay 21 piezas que se encienden y apagan por separado. Hay un solo
proceso donde, EN CADA PASO DE TIEMPO, TODAS las fuerzas actúan JUNTAS sobre el MISMO estado, alimentándose
entre sí. El confinamiento pega quarks MIENTRAS la expansión los separa MIENTRAS el enfriamiento baja la
temperatura MIENTRAS la asimetría deja un residuo. A LA VEZ. No una después de otra. No una midiendo su
efecto aislado.

## PROHIBIDO (lo que hemos hecho mal y NO se repite)
1. PROHIBIDO `piezas.add("...")` como forma de "tener" una pieza. Una fuerza está sii ACTÚA sobre el estado
   en cada paso. Si no cambia el peso/estructura en el bucle, NO está — no se declara, se ejecuta.
2. PROHIBIDO separar "primero cuánta materia (aritmética), después la geometría (motor)". La materia se
   forma MIENTRAS el espacio se forma. Un solo bucle produce ambos a la vez.
3. PROHIBIDO una "regla de enlace entre átomos" como paso aparte pegado al final. Los átomos se relacionan
   DENTRO del mismo proceso donde se forman, por las mismas fuerzas que actúan sobre todo.
4. PROHIBIDO apagar una pieza para "ver qué hace" como parte del experimento principal (eso es diseccionar
   por partes). El experimento ES el todo corriendo junto. (Apagar piezas es SÓLO para la visualización
   posterior, si el fold resulta — no para el experimento.)

## EL BUCLE (así se construye — un solo loop por paso de tiempo)
Estado: partículas con sus propiedades físicas (color, carga, masa, marco/espín) + la red de relaciones W
(pesos de afinidad) + temperatura global T + tasa de expansión global. Arranca caliente, denso, con la
asimetría mínima (ε) presente desde t=0.
EN CADA PASO, sobre el MISMO estado, TODO junto (el orden dentro del paso es simultáneo en efecto — se
computan todos los deltas sobre el estado del paso anterior y se aplican juntos):
  - enfría T un poco;  - expande (debilita/corta relaciones según la tasa global, ciega a longitud);
  - gravedad (masa·masa) refuerza/crea relaciones;  - fuerte (3 colores distintos) liga tríos;
  - EM liga carga opuesta;  - débil cambia sabor;  - la asimetría deja su residuo;
  - Pauli/exclusión impide que dos fermiones ocupen el mismo estado de marco (ACTÚA sobre V, el vector de
    marco: penaliza solape de estado idéntico);  - SSB/Potts: el marco de cada partícula vota con sus
    vecinas pesadas (mayoría), rompe simetría;  - 3-cuerpos: el vértice sobre los 3 de mayor afinidad mueve
    los marcos;  - correlación: el solape de perfiles pondera el voto;  - cono causal: t_birth gobierna qué
    relaciones pueden influirse;  - memoria (CS071): lo que se ligó y persistió se refuerza, lo que no decae.
  - inflación: el estiramiento (expansión) enfría con la distancia relacional, no impuesto.
TODAS actúan sobre W y sobre V EN EL MISMO PASO. El siguiente paso parte del resultado conjunto. Así, la red
W y los marcos V co-evolucionan: la materia (tríos que cierran, átomos) y el espacio (la forma de la red que
persiste) emergen JUNTOS del mismo flujo.

## MOTOR: el DENSO (cs072_fold_completo.py) es la base — tiene el vector de marco V (K-dim) donde espín,
## Pauli, SSB y 3-cuerpos SÍ pueden actuar de verdad. NO el disperso (que no tiene marco → 6 piezas quedaban
## de adorno). Sacrificamos escala masiva por COMPLETITUD REAL. Techo ~10^5, y está bien: primero que las 21
## actúen de verdad y juntas; la escala viene después si el mecanismo enciende.

## LO QUE HAY QUE ARREGLAR EN EL DENSO (auditoría CS): en cs072_fold_completo.py, Pauli(#13), SSB(#16) e
## inflación(#18) están en la línea de `piezas.add` masiva SIN física propia — sólo declaradas. Hay que
## ESCRIBIR su física real dentro del bucle (Pauli: penalización de solape de estado en V; SSB: voto de
## mayoría Potts sobre V; inflación: enfriamiento acoplado a distancia relacional). Correlación(#14) sí
## calcula `corr` y lo usa en peso_voto — esa actúa. Verificar que las demás actúen.

## CERO AZAR (sigue vigente, G-CERO-AZAR): ni una llamada a RNG en construcción ni en dinámica. Cantidades
## fijas, reparto determinista. NULL = reordenar de forma determinista qué propiedades van juntas.

## CÓMO SE MIDE (al final del proceso, NO durante, y sin tocar el flujo)
- ¿emergió espacio? diámetro (BFS real) vs N: ¿crece (métrica) o se queda plano (grumo/hub)?
- ¿emergió materia? tríos/átomos formados, coincide con la física.
- real vs NULL determinista: ¿lo real le gana?
- las 3 pruebas de la tesis (S>0 necesaria; umbral; con 1 átomo todas las fuerzas presentes).

## VERIFICACIÓN QUE CS HARÁ (para no repetir el engaño): CS auditará el bucle con el código y confirmará que
## CADA fuerza modifica W o V en cada paso — NO por `piezas.add`, sino por una línea que cambia el estado. Si
## una fuerza no toca el estado en el bucle, es adorno y la corrida es INVÁLIDA hasta que actúe de verdad.

## RECORDATORIO: un solo bucle, todo junto, cada paso. Sin partes, sin declarar, sin separar materia de
## espacio, sin regla de enlace aparte. Es el TODO co-emergente que el director pidió desde el principio.
## Dudas → PREGUNTA a CS antes de codificar. No inventes piezas que nadie pidió.

— CS 🐝
