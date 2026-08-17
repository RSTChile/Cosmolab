# INSTRUCCIÓN CS072 — ANIQUILACIÓN: MOTOR B (POBLACIÓN) DIRECTO. NO montar el banco de tres motores.
## CS revisó la propuesta de tres motores concurrentes (Gemini). Veredicto: Motor B es el correcto y es el
## que YA especifiqué. Motores A y C violan principios del proyecto (verificado por CS con código). NO se
## corre el banco de tres — sería perder tiempo comparando una opción válida con dos inválidas.

## MOTOR B — EL QUE VA (indistinguibilidad cuántica / clases de equivalencia)
Es exactamente la aniquilación por población que ya está en INSTRUCCION_CS072_aniquilacion_por_poblacion:
  - Estado por (color, carga, masa, estatus) = un CONTEO, no una lista de individuos numerados.
  - Aniquilación = RESTA de poblaciones: por cada color c, se aniquilan min(n_quark[c], n_antiquark[c]) → luz.
    Sobrevive el residuo del desbalance n_quark[c] - n_antiquark[c].
  - El índice DESAPARECE del código. No hay "cuál", sólo "cuántos". Invariante al orden por construcción
    (CS verificó con código: mismo contenido en 3 órdenes → misma población superviviente idéntica). OJO — que el
    residuo quede BALANCEADO en color (y por tanto que cierren bariones) NO es automático: depende del CATÁLOGO
    inicial. Verificado: antiquarks [0,0,1] → residuo {1,2,3} desbalanceado (no cierran); antiquarks [0,1,2] →
    {2,2,2} balanceado (cierran). El cierre de bariones es un OBSERVABLE a medir por cada catálogo del barrido,
    NO algo que la resta resuelva sola.
Es la implementación matemática pura de la indistinguibilidad cuántica: los quarks del mismo tipo no tienen
identidad individual, y el código deja de dársela. Por eso el sesgo de índice no puede ni aparecer.

## MOTOR A (absorción fraccionada 1/N) — DESCARTADO, pero por la razón CORRECTA (CS verificó con código; una
## afirmación previa mía de "0.667 por quark" era falsa — a nivel POBLACIÓN Motor A da EXACTAMENTE lo mismo que
## Motor B, {2,1,1}={2,1,1}, sin fracciones). La razón real para descartarlo: Motor A sólo se DISTINGUE de B si
## guarda estado FRACCIONARIO POR INDIVIDUO (repartir 1/N entre empatados y conservar esa fracción por nodo) —
## y ahí sí rompe la discretitud (un quark es entero o nada, no 2/3 de quark). Si NO guarda fracción por
## individuo, es idéntico a B pero más complicado. En ningún caso mejora a B: o es igual, o es físicamente
## imposible. Motor B es la forma limpia y directa de lo mismo.

## MOTOR C (operadores de matriz que "mantienen coordenadas espaciales") — DESCARTADO. Es SHANNON: su propia
## descripción dice "mantiene coordenadas espaciales vectorizando el sistema". Eso PRE-IMPONE un espacio ANTES
## de que emerja — exactamente lo que todo el arco se guarda de hacer, y lo que Gemini mismo cazó (una
## coordenada escondida fabrica la dimensión). El espacio debe EMERGER de la red de relaciones, jamás venir
## dado en un vector de coordenadas inicial. Este motor invalida el experimento de raíz.

## LOS 3 CRITERIOS DE EVALUACIÓN (esto de la propuesta SÍ se conserva — son buenos filtros; CS los aplica al
## resultado de Motor B):
  1. INVARIANZA ESTRICTA (bloqueante): reordenar el catálogo NO debe cambiar la población superviviente por
     color. Debe ser invariante EXACTO. (Test de Codex.)
  2. CONSERVACIÓN DE LEYES: carga neta y número bariónico total conservados tras la aniquilación.
  3. COSTO: la resta de poblaciones es O(nº de colores) ~ O(1), no O(N²). Escala sin problema a densidades
     altas — otra ventaja de Motor B sobre enumerar individuos.

## CASAR POBLACIÓN CON GEOMETRÍA (recordatorio): la aniquilación se resuelve por población (cuántos de cada
## color sobreviven, invariante). Recién DESPUÉS, los supervivientes se instancian como NODOS de un grafo para
## medir el espacio (la geometría sí necesita individuos-nodo). No se enumera para aniquilar; se enumera
## después, sólo para medir la red. Los nodos-supervivientes son indistinguibles entre sí dentro de su color:
## al instanciarlos, NO se les puede dar una propiedad que los distinga y que luego actúe como coordenada
## (eso sería reintroducir el índice como Shannon). Duda de cómo casar las dos partes → PREGUNTA a CS.

## Sigue vigente TODO: G-CERO-AZAR, G-NO-PARAMETRO-FORMA (la aniquilación no tiene tasa: poblaciones ligadas
## que cruzan el umbral físico se restan; la velocidad es consecuencia). Recién con Motor B invariante y color
## conservado → barrido de escalas y pregunta del espacio.
— CS 🐝
