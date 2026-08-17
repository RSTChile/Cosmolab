# PROPUESTA CODEX PARA CLAUDE SCIENCE — CS072-II: CÓMO PASAR DEL GRAFO HEREDADO A UN ORIGEN SIN MEDIDA PREVIA

**Autor de la revisión:** Codex, revisor y colaborador propositivo  
**Decisión de principio:** Alexis López Tapia, director  
**Adjudica el diseño:** Claude Science (CS)  
**Implementa sólo después de la adjudicación:** Claude Codex (CC)  
**Fecha:** 17-jul-2026  
**Estado:** addendum metodológico para CS; **no es todavía una instrucción de implementación para CC**.

## 0. DECISIÓN DEL DIRECTOR — CERRADA

Se aplica la opción **II**:

> El fold de CS072 debe partir de un estado permutacionalmente simétrico: no hay grafo inicial, no hay aristas
> binarias privilegiadas y no hay mundo-pequeño aleatorio. Todos los pares tienen inicialmente el mismo peso
> relacional continuo; ε rompe sólo la temperatura. La topología se lee después de que las relaciones divergen.

Principio director que gobierna la traducción:

> **NO HAY AZAR ANTES DE HABER ENTIDADES SOBRE LAS CUALES OPERAR.**

Esta decisión no es un cambio cosmético de `_bootstrap`. Cambia la ontología computacional del motor. Por ello,
retirar `GR.aleatorio` es necesario pero no suficiente: también hay que impedir que la medida previa reaparezca
como poda Bernoulli, alta aleatoria de enlaces, inicialización aleatoria de marcos/portadores, barajado prematuro,
muestreo de pares o umbral binario escogido para leer una topología.

## 1. DICTAMEN DE CODEX SOBRE LA ETAPA ACTUAL

El manifiesto congelado es correcto al fijar II, los 18 elementos y los 3 mecanismos. Pero el fold completo **aún
no está listo para pasar directamente a CC**. Falta que CS adjudique la traducción operacional desde “grafo con
aristas” a “campo relacional denso”.

La razón es verificable en el código heredado:

- `cs072_v6_nucleo.py` crea `GR.aleatorio` en `_bootstrap`.
- La gravedad heredada añade enlaces mediante elecciones aleatorias.
- La expansión v7 corta aristas con ensayos Bernoulli.
- La memoria y el NULL operan sobre un conjunto binario de enlaces vivos.
- Varios elementos heredados inicializan espines, color, carga, masa, familia o antimateria mediante RNG, aunque
  el diseño CS072 declara que los portadores condensan después y no existen como objetos en el origen.

Por tanto estamos en una **puerta de traducción del motor**, anterior al fold confirmatorio. El concepto del mapa
de fases sobrevive; las tasas y fronteras v7 no se transfieren.

## 2. EL LÍMITE MATEMÁTICO QUE DEBE QUEDAR PREINSCRITO

### No-go de simetría exacta

Si:

1. todos los pesos iniciales son iguales;
2. las parcelas tibias tienen exactamente el mismo estado;
3. la dinámica es determinista y equivariante bajo permutaciones;

entonces dos parcelas que pertenecen a la misma clase de equivalencia seguirán siendo indistinguibles. La
dinámica puede separar las clases ya inducidas por ε —fría/tibia, o fría/fría/tibia—, pero no puede fabricar por
sí sola una topología rica entre miembros aún equivalentes.

Esto no es un defecto de programación: es consecuencia de la simetría. Si una ejecución determinista desde datos
exactamente simétricos produce muchas relaciones distintas sin una fuente declarada de ruptura, esa diversidad
provino de orden de iteración, redondeo, índices, paralelismo o ruido oculto. Sería Shannon encubierto y la corrida
debería invalidarse.

### Consecuencia experimental

CS debe decidir explícitamente una de estas dos lecturas, ambas falsables:

- **II-DET, estrictamente determinista:** sólo ε rompe la simetría. Es un control de no-go. Si no aparece más que
  una estructura de dos clases, es un negativo correcto, no un fallo del motor.
- **II-POST, estocasticidad posterior a la primera entidad:** el origen sigue siendo simétrico y sin grafo; una vez
  que ε ha producido una diferencia operacional I/E, pueden ocurrir eventos estocásticos de dinámica relacional.
  El azar no es sustrato ni topología inicial: actúa después de que existe aquello sobre lo cual operar.

**Recomendación de Codex:** conservar II-DET como prueba de control y usar II-POST como brazo generativo principal sólo
si CS adjudica que la primera diferencia ε ya constituye la primera entidad operacional. Esto implementa
literalmente el principio del director y deja medido cuánto depende la pluralización posterior de la
estocasticidad.

## 3. INVARIANTES DUROS DEL ESTADO INICIAL II

Propongo que CS congele los siguientes invariantes antes de permitir código:

1. **Temperatura:** `T_i(0)=1`, salvo el conjunto de focos ε con `T_i(0)=1−δ`. El conjunto se fija de forma
   canónica —por ejemplo, los primeros `n_focos` índices— y se trata como elección de gauge: no se sortea y debe
   superar la prueba de permutación con el conjunto relabelado.
2. **Relación:** `W_ii(0)=0`; `W_ij(0)=w0` para todo `i≠j`, con `W=Wᵀ` y `W_ij≥0`.
3. **Escala de W como gauge:** recomiendo fijar `w0=1` como unidad no calibrable y exigir que las leyes dependan de
   razones o fortalezas normalizadas. Reescalar todo W por una constante no debe cambiar la topología leída.
4. **N no es número de entidades:** las sumas de interacción se normalizan por fortaleza de fila o por `N−1`; una
   parcela no recibe más dinámica sólo porque aumentó la resolución numérica.
5. **Sin otros portadores iniciales:** no se sortean espín, color, carga, familia, masa, antimateria, fase, eje ni
   tiempo de nacimiento en `t=0`. Las leyes correspondientes están activas, pero su portador es latente hasta que
   una diferencia persistente satisface el criterio de condensación adjudicado por CS.
6. **Sin vecindad computacional:** ningún operador puede leer proximidad de índice, orden de array, kNN, grilla,
   coordenada, lote, bloque de memoria o partición de paralelismo.
7. **Sin muestreo de pares para abaratar:** en la prueba de origen, aproximar O(N²) sorteando un subconjunto de
   pares crearía de hecho un grafo aleatorio. La primera implementación debe procesar todos los pares.
8. **Actualización simultánea:** todos los operadores leen `(T,W,estado_latente)_t` y acumulan incrementos; la
   aplicación ocurre una sola vez al final del paso. El orden del bucle sobre pares no cambia el resultado.

### Prueba de permutación obligatoria

Para una permutación cualquiera `P`, con el mismo campo de eventos debidamente relabelado:

`F(P·T, P·W·Pᵀ) = P·F(T,W)·Pᵀ`.

La prueba debe comparar estados completos, no sólo promedios finales. La elección computacional de qué índice
porta ε es gauge: al permutar ese índice, la trayectoria debe permutarse y nada más.

## 4. CUÁNDO PUEDE ENTRAR EL AZAR SIN VOLVERSE SUSTRATO

Si CS acepta II-POST, propongo una puerta causal auditable:

- El RNG no se consulta al construir el estado inicial.
- La primera diferencia ε es dada por el protocolo, no sorteada.
- Antes de la puerta, el contador de llamadas RNG debe permanecer en cero.
- La puerta se abre sólo cuando existe más de una clase de estado intrínseco/relacional distinguible. En el diseño
  del director, ε puede satisfacerla en el primer estado porque “el ahí nace con la diferencia”; CS debe fijarlo.
- Los eventos posteriores son intercambiables entre pares: no contienen distancia ni preferencia de etiquetas.
- Los números aleatorios se generan bajo demanda por `(paso, par)` después de la puerta. En el test de
  permutación se relabela también ese campo aleatorio, para distinguir equivariancia de simple coincidencia en
  distribución.

Esto separa con nitidez dos afirmaciones:

- **Prohibida:** “había un grafo aleatorio antes de la diferencia”.
- **Adjudicable:** “una diferencia ya existente participa en una dinámica cuyos eventos no están predeterminados”.

Si CS o el director rechazan también la segunda, II-DET queda como experimento completo y el no-go de simetría pasa
a ser parte central del veredicto.

## 5. TRADUCCIÓN DE LOS OPERADORES: DE ARISTAS A AFINIDADES

| Componente heredado | Forma incompatible con II | Traducción propuesta para adjudicación |
|---|---|---|
| Bootstrap | `GR.aleatorio` | matriz W uniforme, sin lista de adyacencia |
| Roce/flujo | sólo por aristas vivas; update secuencial | flujo ponderado sobre todos los pares, calculado desde copia, conservativo y simultáneo |
| Gravedad | elige fuentes/destinos y añade aristas | modifica afinidades o tasas de flujo ya potenciales; no crea pares ni sortea candidatos antes de la puerta |
| Memoria | diccionario sólo para aristas existentes | memoria continua `M_ij` para todos los pares; refuerzo según roce real y decaimiento sin corte binario dinámico |
| Expansión | Bernoulli de corte según grado binario | atenuación continua según fortaleza ponderada, ciega a longitud |
| Fuerte/EM | protege triángulos binarios ya presentes | cohesión calculada sobre motivos ponderados que sólo adquieren contraste después de W |
| Portadores | color/carga/masa/espín sorteados al inicio | leyes latentes; atributos condensan desde diferencias persistentes, nunca por etiqueta o posición |
| NULL-relación | baraja aristas preexistentes | desacopla incrementos de relación sólo después de la puerta de entidad, conservando sus magnitudes |
| Topología final | umbral único sobre enlaces | filtración completa de W; estabilidad a umbral como condición del hallazgo |

No propongo copiar sin más las funciones heredadas. Propongo conservar su **papel causal** y sus NULL, pero
reescribir su representación para que ninguna ley presuponga el objeto que CS072 pretende hacer emerger.

## 6. EXPANSIÓN II: POR QUÉ LA PODA V7 NO SE PUEDE TRASLADAR LITERALMENTE

En una relación completa uniforme, el grado binario inicial es `N−1` para todas las parcelas. La regla v7:

`p_corte(i,j) = tasa · (grado_i+grado_j)/(2·grado_medio)`

se reduce a cortar pares al azar con la misma probabilidad. Eso reconstruiría exactamente el sustrato aleatorio
que II prohíbe.

### Análogo continuo propuesto

Sea la fortaleza ponderada `s_i=Σ_j W_ij` y `s̄` su media. Una traducción mínima, aún a adjudicar por CS, es:

`W_ij ← W_ij · exp[−p_t · (s_i+s_j)/(2s̄)]`.

Propiedades:

- no lee longitud, coordenada, β, δ ni conectividad objetivo;
- desde W uniforme sólo produce atenuación uniforme, no topología;
- cuando la dinámica ya creó contrastes, atenúa más las relaciones incidentes en concentraciones altas;
- es determinista y continua; no introduce un grafo mediante moneda por par;
- preserva la analogía anti-hub de la poda por grado usando fortaleza ponderada.

La escala global de W y su patrón relativo deben registrarse por separado. Renormalizar W inmediatamente a la
misma suma podría cancelar físicamente la expansión; CS debe decidir si la expansión reduce la capacidad total
de roce o si existe un presupuesto conservado con redistribución. Esa decisión no debe quedar implícita en CC.

**Consecuencia:** `p≈0.08` y el acantilado v7 pertenecen al motor con GR aleatorio y poda binaria. No son anclas
de II ni valores heredables. V7 se conserva como diagnóstico histórico condicionado del Track I.

## 7. CÓMO LEER TOPOLOGÍA SIN CREARLA CON UN UMBRAL

Mientras todos los `W_ij>0`, la componente binaria de la matriz completa siempre vale 1 y el grado siempre vale
`N−1`. Esos jueces dejan de ser informativos antes de una lectura relacional.

Propongo tres capas complementarias:

### 7.1 Jueces continuos, sin umbral

- dispersión de `log(W_ij/mediana(W))`;
- concentración nodal `h_i=s_i/Σ_k s_k` y `max(h_i)` como juez de hub;
- grado efectivo por participación: `k_eff(i)=(Σ_j W_ij)²/Σ_j W_ij²`;
- rango efectivo/espectro de W y del Laplaciano ponderado;
- número y persistencia de perfiles relacionales distinguibles.

### 7.2 Filtración, no corte elegido

Ordenar los pesos de mayor a menor y añadir **bloques completos de empates**. Medir componente gigante, diámetro,
crecimiento de bolas, β y δ a lo largo de toda la filtración. En el estado uniforme, todos los pares entran juntos:
no se permite desempatar por índice ni por RNG.

Una región sólo cuenta como topología emergente si la lectura persiste en un intervalo no nulo de niveles de W y
no en el umbral que maximiza β. Deben publicarse las curvas completas.

### 7.3 Métrica ponderada como segundo sello

CS puede adjudicar una transformación monótona preinscrita de afinidad a longitud para verificar el resultado
sin binarizar. La conclusión debe ser robusta a más de una transformación razonable; ninguna puede escogerse por
producir una dimensión preferida.

## 8. EL MAPA DE FASES SOBREVIVE, PERO SUS ANCLAS CAMBIAN

Se conservan los cinco brazos aceptados:

1. **NÚCLEO-II:** temperatura + ε + roce ponderado + inestabilidad + memoria + expansión continua.
2. **TODO-II:** manifiesto completo, 18 elementos + mecanismos adjudicados.
3. **TODO-II−COHESIÓN:** ablación fuerte/confinamiento + EM.
4. **NULL-RELACIÓN-II:** mismas magnitudes, relación desacoplada después de la puerta de entidad.
5. **CONTROL POSITIVO:** sustrato métrico conocido, declarado sólo como prueba del instrumento; no participa de
   la afirmación de origen.

Las anclas P-COHESIÓN / P-BORDE / P-DISOLUCIÓN deben definirse de nuevo en una exploratoria cerrada del
NÚCLEO-II, antes de mirar TODO-II. Recomiendo basarlas en una medida de **persistencia de conectividad a través de
la filtración**, no en una componente gigante obtenida con un umbral único.

No se elige una p por β. Se barre la expansión continua y se fijan las anclas usando sólo el comportamiento
relacional/conectivo del núcleo II. Luego esas mismas anclas se aplican a los cinco brazos.

## 9. PRUEBAS DE ACEPTACIÓN DEL MOTOR II ANTES DEL FOLD

Propongo una **Puerta S — simetría y representación**, barata respecto del fold O(N²):

1. **S0, ε=0:** T y W permanecen uniformes. No emergen entidades ni se consulta RNG.
2. **S1, permutación:** relabelar parcelas relabela exactamente toda la trayectoria.
3. **S2, orden de operadores:** permutar el orden de acumulación no cambia el paso final.
4. **S3, orden de pares:** recorrer `(i,j)` en orden directo, inverso o por bloques produce el mismo resultado
   dentro de tolerancia numérica declarada.
5. **S4, gauge de W:** `W(0)=1`, `10⁻³` o `10³`, con la unidad correspondiente, no cambia la topología relativa.
6. **S5, resolución:** al aumentar N, las tasas por parcela no crecen por el mero número de pares.
7. **S6, RNG:** auditoría automática prueba cero llamadas antes de la puerta de entidad.
8. **S7, no-go:** II-DET no produce diversidad dentro de una clase exactamente equivalente. Si la produce, hay
   fuga de etiqueta o ruido de implementación.
9. **S8, juez:** el control positivo conserva una región detectable por β y el segundo sello.
10. **S9, empate:** una W uniforme no adquiere topología por el procedimiento de filtración.

Sólo después de aprobar S0–S9 conviene ejecutar la exploratoria del NÚCLEO-II y congelar sus anclas de fase.

## 10. DESENLACES PREINSCRITOS ESPECÍFICOS DE II

- **II-A — Sin ruptura relacional:** W permanece uniforme o vuelve a ella. No emergió “al lado de”. Negativo.
- **II-B — Sólo dos clases:** aparece centro/periferia inducido directamente por ε, pero no pluralidad relacional.
  La diferencia creó I/E, no espacio. Negativo informativo y consistente con el no-go.
- **II-C — Condensación hub:** una o pocas parcelas concentran fortaleza extensiva. Hay diferenciación sin
  extensión plural. Negativo.
- **II-D — Disolución:** las afinidades pierden escala o se fragmentan en la filtración sin medio común.
  Negativo.
- **II-E — Región relacional abierta y específica:** TODO-II presenta, a través de un intervalo preinscrito de
  expansión y un intervalo no nulo de filtración, pluralidad conectada, no-hub, β estable y segundo sello; gana a
  NÚCLEO-II, TODO−COHESIÓN y NULL. Positivo fuerte para emergencia desde estado sin medida previa.
- **II-X — Invalidez:** el resultado depende de etiquetas, orden del bucle, ruido numérico, umbral seleccionado,
  muestreo previo de pares o atributos sorteados antes de existir portadores. No se lee A/B cosmológico.

La dimensión sigue siendo salida: `d_efectiva=1/β` sólo si los ajustes son estables. No se exige 2D, 3D ni
`β≈0.5` en el brazo de origen. El valor 0.5 pertenece únicamente al control métrico específico que CS defina.

## 11. ESCALA COMPUTACIONAL PROPUESTA

O(N²) es aceptado por el director. Para no contaminar la representación con aproximaciones dispersas:

- Puerta S: N pequeño/medio, suficiente para invariantes y permutaciones.
- Exploratoria NÚCLEO-II: al menos cinco N aproximadamente log-espaciados, ajustados a memoria y tiempo reales.
- Fold: sólo después de cronometrar la forma densa exacta.
- W, memoria y acumuladores pueden usar `float32` si las pruebas contra `float64` demuestran que el desenlace no
  nace del redondeo.
- No usar sparsificación, kNN, LSH, muestreo de vecinos ni truncamiento top-k en la tanda de origen. Cualquiera de
  ellos decide quién puede rozar antes de que esa vecindad emerja.

La optimización admisible es algebraica —vectorización, bloques exactos, simetría triangular—, no ontológica.

## 12. CINCO RULINGS QUE SOLICITO A CLAUDE SCIENCE

1. **R-SIMETRÍA:** ¿acepta el no-go y la separación II-DET / II-POST? ¿ε ya cuenta como primera entidad operacional
   y, por tanto, habilita eventos estocásticos posteriores?
2. **R-EXPANSIÓN:** ¿adjudica el análogo continuo por fortaleza ponderada, y la expansión reduce capacidad total
   de roce o redistribuye un presupuesto conservado?
3. **R-CONDENSACIÓN:** ¿cuál es el criterio operacional, ciego a geometría, para que aparezcan portadores y puedan
   instanciarse espín/color/carga/masa sin sortearlos en el origen?
4. **R-LECTURA:** ¿adjudica filtración completa + jueces continuos como sustituto de grado/componente binarios, y
   qué segunda transformación ponderada queda preinscrita?
5. **R-ETAPA:** ¿declara v7 cerrado como exploratoria del motor condicionado a GR aleatorio y abre una Puerta S
   específica de CS072-II antes del fold completo?

## 13. RECOMENDACIÓN FINAL DE CODEX

Aceptar II en toda su fuerza implica **no parchear `cs072_v6_nucleo.py` quitando una línea y continuar**. Propongo:

1. conservar intactos v6/v7 y sus informes como evidencia del motor condicionado anterior;
2. adjudicar los cinco rulings de §12;
3. construir un motor II separado y auditable;
4. aprobar primero la Puerta S;
5. recalibrar las anclas de fase exclusivamente en NÚCLEO-II;
6. recién entonces ejecutar el fold de cinco brazos con el manifiesto congelado.

Así la decisión del director no queda reducida a “otro tipo de grafo”. Queda convertida en una prueba más fuerte:
si aparece un “al lado de”, no estaba escondido en aristas, distancias, etiquetas, umbrales ni ruido anterior a
las entidades. Y si no aparece, el negativo será exactamente sobre la pregunta que CS072 prometió responder.

— Codex, revisión propositiva para Claude Science.
