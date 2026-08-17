# ON-77 — Sistema A/B corregido: qué se arregló, qué mostró, y por qué quedó sin resolver

**Fecha:** 7-ago-2026 · **Escribe:** orquestador (el agente que corrió las dos correcciones dejó el cómputo
completo en disco pero no llegó a escribir este informe — se redacta a partir de los archivos `.json`/`.log`
que sí produjo, verificados en disco, sin volver a correr nada). **No se declara cierre ni veredicto — sólo
números. La lectura final es de Alexis.**

## Qué se corrigió respecto del piloto anterior

- **Sistema A:** la métrica pasó de `clustering_global/N` (intensiva, sospechosa) a **masa en sumideros vía
  Phantom** (extensiva, la misma que usa toda la jerarquía NULL de esta sesión).
- **Sistema B:** el mecanismo pasó de "recalcular el grafo kNN desde cero" (degenerado, sin memoria) a
  **reorganización de presupuesto acotado** — un número limitado de operaciones de reconexión por tanda, para
  que el resultado en la tanda H dependa genuinamente del camino recorrido, no sólo del estado final.

## Resultado 1 — Sistema B, verificación de grafo (barata, corrida ANTES de Phantom): el arreglo funcionó

| H | Jaccard vs H=1 | triángulos REAL | clustering REAL | reorganizaciones acumuladas |
|---|---|---|---|---|
| 1 | 1.0000 (referencia) | 342 | 0.444 | 0 |
| 2 | 0.6937 | 428 | 0.394 | 7 |
| 4 | 0.6052 | 520 | 0.364 | 42 |
| 8 | 0.5725 | 612 | 0.367 | 96 |
| 16 | 0.5604 | 716 | 0.385 | 196 |

**El mecanismo YA NO es degenerado** — el Jaccard se aleja de forma monótona de H=1 a medida que H crece
(0.69→0.61→0.57→0.56), confirmando que cada tanda deja una huella genuina del camino recorrido, no sólo del
estado final. Esto era justamente lo que el piloto anterior no lograba probar.

## Resultado 2 — ambos sistemas, nivel Phantom: cero sumideros en TODOS los puntos, de los dos sistemas

**Sistema A** (N=50/100/200/400, regla generativa fija):

| N | masa en sumideros | masa/N | ganancia marginal |
|---|---|---|---|
| 50 | 0.00 | 0.0000 | — |
| 100 | 0.00 | 0.0000 | +0.0000 |
| 200 | 0.00 | 0.0000 | +0.0000 |
| 400 | 0.00 | 0.0000 | +0.0000 |

**Sistema B** (N=200 fijo, H=1/2/4/8/16):

| H | masa en sumideros |
|---|---|
| 1 | 0.00 |
| 2 | 0.00 |
| 4 | 0.00 |
| 8 | 0.00 |
| 16 | 0.00 |

**Ninguna de las 9 corridas Phantom (4 de Sistema A + 5 de Sistema B) formó un solo sumidero.** No hay
ganancia marginal que medir, ni efecto de H que medir, en el observable extensivo — ambos criterios de
falsación quedan **sin poder ponerse a prueba** con este diseño, no confirmados ni refutados.

## Por qué pasó esto — no es un fracaso nuevo, es el MISMO problema ya diagnosticado

Los N usados acá (50 a 400) están todos **por debajo** de N=500, que [[cosmogenesis-confound-caja-escala-n-6ago2026]]
ya había encontrado que da cero sumideros de forma sistemática — no por la estructura que se prueba, sino
porque el generador de condiciones iniciales hace crecer la caja física con `n^(1/3)`, así que a menor N hay
menos masa total absoluta (no sólo menos resolución), y el sistema queda muy por debajo del umbral de colapso
sin importar qué tan "real" o "condensada" sea su historia relacional. Este experimento heredó exactamente ese
confound, sin haberlo controlado — un vacío del diseño, no del análisis.

## Lectura, en simple

Es como querer probar si un horno bien precalentado cocina más rápido que uno frío, pero poner tan poca masa
de pan en ambos que ninguno de los dos llega nunca a la temperatura de cocción — el experimento no distingue
"horno bueno" de "horno malo" porque ninguno de los dos cruza el umbral. Arreglamos el termómetro (la métrica
de A) y arreglamos que el horno realmente recuerde su historia de precalentado (el mecanismo de B) — pero
seguimos sin poner suficiente masa como para que la comparación diga algo.

## Qué haría falta para que esto sí pruebe algo

Escalar Sistema A y Sistema B a un N donde YA SABEMOS que se forman sumideros (N≈2000, como el resto de la
jerarquía NULL) — o, más barato, rehacer el generador con **masa total fija** en vez de dejarla escalar con N
(la misma corrección que ya está pendiente de autorización desde el hallazgo del confound de caja/masa). Sin
eso, cualquier sweep de N chico en este proyecto va a dar cero, sea cual sea la hipótesis que se esté probando.

## Archivos

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/ON77_sistemaA_corregido.py`,
  `ON77_sistemaB_corregido.py` — código corregido, no toca nada congelado.
- `/Users/alexis/phantom_cs073/ON77_sistemaA_corregido/ON77_sistemaA_corregido_resultado.json`,
  `/Users/alexis/phantom_cs073/ON77_sistemaB_corregido/ON77_sistemaB_corregido_resultado.json` — datos crudos.
- `logs/ON77_sistemaA_corregido_run.log`, `logs/ON77_sistemaB_corregido_run.log` — bitácora de ejecución.
