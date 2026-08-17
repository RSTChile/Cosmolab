# ADJUDICACIÓN CS — CS072 v6/v7: la expansión como operador de PODA. Gemini acierta, con una trampa de Shannon marcada.
## CS, 17-jul-2026. Sobre la propuesta del director (vía Gemini): expansión = operador que DESTRUYE enlaces. Verificado con código por CS.

## LO CORRECTO (confirmado)
El diagnóstico previo dejó el balance a medias: gravedad tejía enlaces sin oposición, expansión sólo enfriaba
(nunca tocaba topología) → hub. La propuesta lo completa: **la expansión es el operador que PODA enlaces** — la
tasa fundamental a la que las relaciones se debilitan y cortan por el flujo irreversible de entropía del todo. Es
una competencia puramente topológica: la gravedad teje LOCAL, la expansión poda; si la gravedad teje más rápido,
súper-hub (B); si hay balance, la poda obliga a las conexiones a ordenarse en redes locales → condensa geometría.
Esto es CORRECTO y es la mitad que faltaba. Verificado: con poda, el hub desaparece (grado_max plano ~10 vs N−1).

## LA TRAMPA DE SHANNON (marcada — el director la evita si se enuncia bien)
La propuesta dice "podar enlaces LARGOS". **"Largo" = distancia = la métrica que el experimento debe DERIVAR.**
Si el código lee la longitud/distancia de un enlace para decidir si lo corta, ya supuso la geometría que quiere
obtener. PROHIBIDO. La poda debe ser CIEGA a la longitud.

## VERIFICADO CON CÓDIGO (N∈{400,900,1600}, gravedad local + flujo-enfriamiento + poda)
| poda | β | grado_max | hub |
| por longitud | PROHIBIDA (lee la métrica = Shannon) | — | — |
| uniforme (cada enlace, misma prob, ciego a longitud) | 0.11 | ~20 | eliminado |
| por grado (nodos muy conectados = más frágiles) | 0.28 | ~10 plano | eliminado |
Ambas versiones anti-Shannon ELIMINAN el hub (antes grado_max=N−1). La poda-por-grado (anti-hub) es la más
efectiva (β 0→0.28) y físicamente defendible SIN leer distancia: un nodo con mucha conectividad es más expuesto a
que la expansión le corte enlaces. NINGUNA llega a métrica pura sola (β=0.28, no 0.5) — falta que más ingredientes
del arco actúen juntos (que es el punto de CS072: el TODO). Pero el bloqueo (hub) está resuelto sin Shannon.

## POR QUÉ NO ES SHANNON (registro para el auditor)
- La poda NO lee longitud ni distancia (esas no existen aún). Lee sólo grado (nº de enlaces de un nodo) o es
  uniforme — ambas son propiedades del grafo, no de una métrica supuesta.
- La tasa de poda sale de la expansión/entropía del todo (uniforme), no de un objetivo geométrico.
- El grado NO se topa a mano (no hay cap); la poda es probabilística por paso, la escala del grado emerge del
  balance tejer-vs-podar, no de un número escrito.

## VEREDICTO OPERATIVO
1. **Incorporar la poda como la ley de expansión sobre la topología** (además del enfriamiento sobre T). Es la
   mitad que faltaba del balance que el director diseñó — NO es un ingrediente nuevo, es la expansión (ya en el
   arco) actuando donde debía: en el tejido, no sólo en la temperatura.
2. **La poda es CIEGA a la longitud (dura).** Realización anti-Shannon: por grado (anti-hub) o uniforme. NUNCA
   "enlaces largos". CC elige entre grado/uniforme, DECLARA cuál y AUDITA que no lee distancia; CS audita el código.
3. **Correr junto al barrido de nº-de-focos (§6)** — los dos contrapesos al hub se prueban juntos: focos (reparte
   la atracción) + poda (corta la sobre-conexión). Reportar β/δ/grado_max/CV por paso.
4. **Sólo tras esa exploratoria se pliegan las 10 leyes y se corre la tanda.** No se lee (A)/(B) hasta que el
   balance completo (gravedad-teje + expansión-poda + focos) corra con todo el repertorio.
5. Parámetros heredados; cambiar uno = otro número CS. La tasa de poda se BARRE como parámetro de realidad (igual
   que la tasa de expansión): se mide a qué tasa condensa estructura, no se elige.

## EN UNA LÍNEA
La expansión-como-poda es la mitad que faltaba del balance —y verificado, elimina el súper-hub que bloqueaba todo—;
pero "podar enlaces largos" es Shannon (largo=la métrica a derivar), así que la poda debe ser CIEGA a la longitud:
por grado (anti-hub) o uniforme, que suben β de 0 a 0.28 y matan el hub sin leer una sola distancia. No alcanza
métrica sola porque faltan los demás ingredientes actuando juntos — que es el punto de CS072. Se corre con el
barrido de focos, y no se lee veredicto hasta que el todo co-emergente corra completo.

— CS 🐝
