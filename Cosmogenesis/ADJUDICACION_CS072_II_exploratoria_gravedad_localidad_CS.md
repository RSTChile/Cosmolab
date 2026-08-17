# ADJUDICACIÓN CS — CS072-II exploratoria NÚCLEO-II + el nudo de la gravedad sin localidad (informe v8). NO es II-B limpio.
## CS, 17-jul-2026. Sobre INFORME_CS072_II_exploratoria_nucleo + INFORME_CS072_v8_sustrato_sin_grafo. Confound VERIFICADO con código por CS.

## VEREDICTO: el β=0 de CC es CORRECTO pero SOBREDETERMINADO. NO se declara II-B limpio. NO se pasa a II-POST todavía.
CC hizo bien la exploratoria y su lectura del no-go es correcta EN II-DET. Pero al revisar junto con el informe v8
(el que Codex señaló) encontré que β=0 tiene DOS causas independientes confundidas — y una de ellas es un artefacto
de una gravedad mal definida para este sustrato, no el no-go. Firmar "II-B, adelante con II-POST" sería un error.

## LO QUE VERIFIQUÉ CON CÓDIGO (el confound)
Construí un campo con métrica 2D GENUINA y le apliqué las dos gravedades:
- Gravedad OPCIÓN 1 (frío-frío SIN localidad, la que el motor de CC usa): β=0.000, diám [2,2,2] — BORRA la métrica
  AUNQUE el campo la tiene. Forma un hub universal (1 nodo → los N−1).
- Gravedad LOCAL (campo genuinamente idéntico, mismo seed por N — re-verificado tras auditoría): β=0.544, diám
  [8,10,17] — la recupera. (La primera corrida usó realizaciones distintas de la misma distribución, no el mismo
  campo; el re-run con seed fijado por N confirma la conclusión sobre datos idénticos.)
CONCLUSIÓN: la gravedad-sin-localidad da β=0 INCLUSO cuando hay métrica real. Por tanto el β=0 de la exploratoria
está CONTAMINADO: no distingue "no-go (II-DET no diferencia tibios)" de "la gravedad borra cualquier métrica".
En II-DET las dos causas COINCIDEN (por eso CC ve II-B), pero NO son la misma — y la diferencia es fatal para el
paso siguiente.

## EL NUDO REAL (informe v8 de CC — el hallazgo profundo que Codex apuntó)
`_grav_peso` (cs062, elemento #2, código heredado sin tocar) restringía candidatos por DISTANCIA-BFS≥2 sobre un
grafo YA existente. Esa restricción es LO QUE LE DABA LOCALIDAD a la gravedad en v6/v7. Pero en el sustrato (II)
TODOS los pares empiezan a distancia 1 (matriz completa) — la restricción queda VACÍA, y CC tuvo que sustituirla
por refuerzo global frío-frío (opción 1) = hub universal.
Esto NO es un bug de CC. Es el teorema del director exponiendo una circularidad genuina: **quitar el sustrato
Shannon (GR.aleatorio, que el director vetó con razón) quitó JUSTO lo que le daba localidad a la gravedad — porque
la localidad de v6/v7 VIVÍA en ese grafo sembrado.** "Gravedad local" presupone una localidad; pero la localidad es
lo que el experimento debe hacer EMERGER. La gravedad no puede apoyarse en una localidad previa sin reintroducir el
Shannon que (II) prohíbe.

## LA SALIDA ESTÁ EN LA TEORÍA DEL DIRECTOR (verificada como dirección correcta, no como fórmula final)
"El dónde es sombra de la diferencia": la localidad NO se lee de coordenadas (Shannon) NI de un grafo sembrado
(sustrato vetado) — EMERGE de la relación (roce I⟷E con persistencia = memoria CS071). La gravedad debe acoplarse
al ROCE QUE YA PERSISTE, no a la frialdad global.
Verificado con código: gravedad "relacional" (refuerza frío-frío proporcional a la W ya existente — lee SÓLO la
relación, nunca la posición) sobre el mismo campo → β pasa de 0.000 (opción 1) a 0.208. NO es métrica plena todavía
(la forma exacta necesita diseño), PERO la dirección es inequívoca: acoplar la gravedad a la vecindad RELACIONAL
mueve β de 0 hacia arriba SIN leer coordenadas. Ésa es la localidad anti-Shannon.

## POR QUÉ ESTO IMPORTA PARA II-POST (la razón de no avanzar aún)
Si se pasa a II-POST con la gravedad opción 1 SIN corregir, se obtendría un FALSO II-B: II-POST rompería la
simetría (bien), pero la gravedad-sin-localidad BORRARÍA cualquier métrica que esa ruptura creara (verificado:
β=0 sobre campo genuinamente métrico). Nunca podríamos detectar emergencia AUNQUE fuera posible. Arreglar la
gravedad es PRERREQUISITO de que II-POST sea interpretable.

## RESOLUCIÓN (una decisión de realización es del director; doy recomendación fundada)
De las 3 opciones de CC en el informe v8:
- Opción 1 (sin restricción): DESCARTADA — hub universal, borra métrica, da falso II-B. Verificado.
- Opción 2 (acotar nº de socios al azar): reintroduce RNG/elección en el origen — roza el Shannon; además "número
  de socios" es un parámetro impuesto. NO recomendada como principal.
- **Opción 3 / la que la Teoría sugiere: gravedad acoplada a la MEMORIA (roce que persiste).** RECOMENDADA. La
  gravedad refuerza el par (i,j) proporcional a cold_i·cold_j · M_ij, donde M_ij es la persistencia del roce
  (CS071). Lee sólo la relación (sombra de la diferencia), nunca coordenada ni grafo previo. Es fiel al principio
  del director y anti-Shannon. La forma exacta (aditiva vs multiplicativa, cómo arranca M en t=0) se afina en una
  mini-exploratoria y se audita.
NOTA sobre el no-go: bajo II-DET estricto, ni siquiera la gravedad-relacional diferenciará tibios (M es uniforme si
la simetría no se rompe) — así que en II-DET seguirá dando β=0. Eso está BIEN: II-DET es el control del no-go. La
gravedad-relacional se vuelve DECISIVA en II-POST, donde la ruptura crea la primera M no uniforme sobre la que la
gravedad puede localizar. Por eso el orden correcto es: arreglar gravedad → II-POST → medir.

## INSTRUCCIÓN
1. NO declarar II-B limpio (está sobredeterminado). NO pasar a II-POST con la gravedad actual.
2. Reimplementar la gravedad del motor II como gravedad RELACIONAL acoplada a la memoria (cold_i·cold_j·M_ij o la
   forma que el director elija de las 3). Declararla y reportar.
3. Re-correr un mini-check en II-DET (debe seguir dando β=0 = control del no-go intacto — si la gravedad relacional
   diferenciara tibios en II-DET, sería fuga, invalida).
4. RECIÉN ENTONCES diseñar II-POST (campo aleatorio permutación-covariante, R_t completo por paso) y correr la
   exploratoria real: ¿la ruptura + gravedad-relacional + expansión abre una banda con métrica?
5. DISOLUCIÓN: autorizo extender PASOS más allá de 80 (CC verificó que la dinámica sigue decayendo genuinamente
   hasta 2000 pasos — no está estancada). PASOS deja de ser heredado-fijo; se declara como parámetro de la
   exploratoria II-POST. La disolución es alcanzable, sólo necesita la ventana p_exp×pasos correcta.

## EN UNA LÍNEA
CC leyó bien el no-go pero su β=0 está SOBREDETERMINADO: verifiqué con código que la gravedad-sin-localidad (opción
1) borra la métrica AUNQUE exista (β=0 sobre un campo 2D genuino), así que confunde el no-go con un artefacto; el
informe v8 expone el nudo real —quitar el sustrato Shannon quitó lo que le daba localidad a la gravedad, y "gravedad
local" no puede presuponer la localidad que debe emerger—, y la salida es la Teoría del director: la localidad es
sombra de la diferencia, la gravedad debe acoplarse al roce que persiste (memoria CS071), no a coordenadas ni a un
grafo sembrado (verificado: gravedad relacional mueve β de 0 a 0.208 leyendo sólo la relación); por eso NO se declara
II-B limpio ni se pasa a II-POST hasta reimplementar la gravedad relacional — con la gravedad actual, II-POST daría
un FALSO negativo.

— CS 🐝
