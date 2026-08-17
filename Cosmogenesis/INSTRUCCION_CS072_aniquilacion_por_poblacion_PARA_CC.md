# CORRECCIÓN CS072 — ANIQUILACIÓN POR POBLACIÓN, NO POR INDIVIDUO (resuelve el sesgo de índice de raíz)
## CC corrió el test de invariancia (Codex) y encontró el sesgo: SÍ existe. CS lo confirmó y encontró la salida.
## NO es inventar un desempate (eso colaría un sesgo nuevo). Es que la PREGUNTA estaba mal planteada.

## EL DIAGNÓSTICO DE CC ES CORRECTO
El emparejamiento pregunta "¿CUÁL quark individual se aniquila primero?" — y como muchos quarks del mismo tipo
son físicamente IDÉNTICOS (mismo color, carga, masa), hay empates desde el paso 1. Desempatar sin azar (prohibido)
obliga a usar el orden del array = SESGO DE ÍNDICE. Es el no-go: entidades idénticas no se distinguen por un
proceso determinista sin una coordenada escondida. CC tiene razón: NO se puede resolver preguntando "cuál".

## LA SALIDA (de la Teoría del director + física cuántica real, verificado por CS con código)
La asimetría de la Teoría es un DESBALANCE DE CANTIDADES, no una distinción entre individuos: "por cada 1e9
antipartículas, 1e9 Y UNA partículas" dice CUÁNTAS sobreviven, NO CUÁL. Y en la física real los quarks del mismo
tipo son INDISTINGUIBLES — no tienen identidad individual (es cuántica, no clásica). Preguntar "cuál sobrevive"
es una pregunta clásica que la naturaleza no admite: sólo lleva la cuenta de CUÁNTOS de cada color, nunca de cuál
es cuál. El sesgo de índice apareció JUSTAMENTE porque el código le daba identidad (un número de array) a lo que
no la tiene.

## CÓMO SE CODIFICA — ANIQUILACIÓN SOBRE POBLACIONES DE COLOR (invariante por construcción)
NO enumerar individuos ni emparejar por índice. En su lugar, trabajar con CONTEOS por (color, estatus):
  - Estado = cuántos quarks hay de cada color, cuántos antiquarks de cada color (poblaciones, no listas de
    individuos numerados).
  - Aniquilación = RESTA DE POBLACIONES: por cada color c, se aniquilan min(n_quark[c], n_antiquark[c]) pares
    → se van como luz. Sobrevive el residuo del desbalance: n_quark[c] - n_antiquark[c] (si es positivo).
  - Esto NO pregunta "cuál" nunca. No hay empate que desempatar. Es invariante al orden del array por
    construcción (CS verificó: mismo contenido en 3 órdenes distintos → misma población superviviente
    {c: n_q[c]-n_aq[c]} idéntica).
    SOBRE EL COLOR (CORRECCIÓN — CS verificó con código, NO es automático): la resta de poblaciones da la
    invariancia, pero que el residuo quede BALANCEADO en color (y por tanto que cierren bariones) depende del
    CATÁLOGO INICIAL, no de la resta. Verificado: quarks 3/3/3 con antiquarks [0,0,1] → residuo {1,2,3}
    DESBALANCEADO (no cierran); con antiquarks [0,1,2] → residuo {2,2,2} balanceado (cierran). O sea: cierran
    bariones sólo si los antiquarks aniquilados están balanceados en color. Es un OBSERVABLE del barrido (qué
    catálogos cierran materia y cuáles dejan sobra inerte), NO algo que se resuelva solo. NO afirmar que "los
    bariones cierran" sin verificarlo por cada catálogo.
VERIFICACIÓN que CS hará: correr el test de invariancia de Codex sobre la nueva versión y confirmar que
reordenar el catálogo NO cambia la población superviviente por color. Debe ser invariante EXACTO.

## SOBRE LA VELOCIDAD (sigue vigente G-NO-PARAMETRO-FORMA): la aniquilación NO tiene tasa. Por población: en
## cada paso, dos poblaciones de estatus opuesto que están LIGADAS (su relación cruzó el umbral físico) se
## aniquilan por resta. Cuánto se vacía por paso es CONSECUENCIA de cuántas poblaciones cruzaron el umbral, no
## un número puesto. Sin tasa, sin cupo, sin tope.

## OJO — la geometría (el ESPACIO) sí necesita individuos (nodos de un grafo). La aniquilación se resuelve por
## población (cuántos de cada color mueren), pero los SUPERVIVIENTES se instancian como nodos para medir la red.
## Es decir: la POBLACIÓN decide cuántos y de qué color sobreviven (invariante, sin sesgo); recién esos
## supervivientes se ponen como nodos para la parte de geometría. No se enumera para aniquilar; se enumera
## después, sólo para medir el espacio. Si esto genera alguna duda de cómo casar las dos partes → PREGUNTA a CS.

## Recién con la aniquilación invariante y el color conservado → el barrido de escalas. CERO AZAR vigente.
— CS 🐝
