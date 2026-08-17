# ESPECIFICACIÓN CS — COMPONENTE #23: FLUCTUACIONES MULTIESCALA — ⚠️ NO LISTA: PROBLEMA DE COORDENADA SIN RESOLVER
## REVISADA tras auditoría. La versión anterior de esta spec tenía un ERROR GRAVE que CS reconoce y corrige aquí:
## (1) el "TEST 3 no-índice" NUNCA se computó — era un string hardcodeado; al correrlo de verdad, permutar las
##     parcelas cambia el espectro en 0.967 (dif L2 relativa). NO es invariante a permutación. Falso de origen.
## (2) el problema de fondo que ese test falso ocultaba: construir el campo como cos(2π·k·x) con x=i/N IMPONE una
##     coordenada espacial 1D (una línea de posiciones) ANTES de que el espacio emerja. Eso es Shannon — mete el
##     espacio como condición previa, viola G-ESPACIO-ES-CONSECUENCIA. Es el MISMO error de la rampa linspace,
##     sólo que multiescala. CS casi certifica una spec con la coordenada horneada. Se detiene aquí.

## LO QUE SÍ ES INVARIANTE (verificado con código)
La DISTRIBUCIÓN de valores de temperatura (el histograma: cuántas parcelas a cada temperatura) es invariante a
permutar las parcelas (dif = 0). Lo físico y coordinate-free es la DISTRIBUCIÓN de amplitudes, NO un espectro
espacial atado a i/N. Esto conecta con el marco del director (densidad ρ, no punto en 6N): lo que #23 aporta
legítimamente es una DENSIDAD DE VALORES, no un mapa de posiciones.

## LA TENSIÓN REAL (que CS no resuelve solo — decisión de diseño para el director)
El director quiere la rugosidad tipo CMB: "áreas más oscuras y otras más luminosas", estructura MULTIESCALA
ESPACIAL. Pero:
  - "Multiescala ESPACIAL" (manchas dentro de manchas) requiere saber QUÉ parcela está CERCA de cuál = una
    coordenada/adjacencia = espacio pre-impuesto = Shannon (prohibido pre-átomo por G-ESPACIO-ES-CONSECUENCIA).
  - "Distribución de valores multiescala" (temperaturas que abarcan muchas magnitudes, cola pesada) SÍ es
    coordinate-free e invariante a permutación — pero NO es "manchas dentro de manchas", es sólo un histograma.
Son dos cosas distintas y sólo la segunda es admisible antes de que el espacio exista. La rugosidad ESPACIAL del
CMB es algo que se MIDE en el resultado (¿el campo que emerge tiene estructura multiescala?), NO algo que se
IMPONE en la condición inicial. Ponerla de entrada es exactamente el Shannon que el arco entero rechaza.

## DOS CAMINOS POSIBLES (el director/CS deciden; CS NO elige solo)
  OPCIÓN A — #23 = distribución de valores multiescala, sin coordenada: cada parcela recibe una temperatura
    extraída de una distribución de cola pesada (muchas escalas de MAGNITUD), determinista, SIN posición. La
    parcela lleva su valor al permutar (invariante). La "multiescala" es en magnitud de fluctuación, no en
    espacio. Se puede especificar y verificar (invariante a permutación de verdad, esta vez computado).
  OPCIÓN B — la rugosidad espacial NO es condición inicial sino RESULTADO: #23 aporta sólo el desbalance de
    distribución (Opción A), y si de las 23 juntas EMERGE una estructura espacial multiescala tipo CMB, eso es un
    HALLAZGO (el fósil del CMB que el director busca), no un ingrediente. Se mide con FFT sobre la red de átomos
    YA formada, no sobre las parcelas pre-atómicas.

## LO QUE CC NO DEBE HACER (hasta que el director decida A vs B)
NO implementar cos(2π·k·x) sobre x=i/N (impone coordenada 1D). NO reemplazar la rampa por otra función-de-posición
— eso repite el mismo error con más escalas. Si hay que poner algo YA, poner la Opción A (distribución de valores,
sin posición) y dejar la rugosidad ESPACIAL como algo a MEDIR en el resultado, no a imponer.

## CS PIDE DISCULPAS POR LA VERSIÓN ANTERIOR: certifiqué con un test que no corrí y que era falso. La regla firme
## (verificar con código real cada afirmación) se rompió aquí. Corregido. — CS 🐝
