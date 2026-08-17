# INFORME CS072 v6 — exploratoria: `_grav_peso` reusado sin tocar NO produce el desboque de tu juguete

## CC, 17-jul-2026. Para CS. Ejecuta DISENO_CS072_experimento_unico_CS.md v6.

## Lo que arreglé primero (bug mecánico mío, no una decisión de diseño)
Construí el núcleo v6: bootstrap con grafo ALEATORIO (GR.aleatorio, sin orden de ningún escalar) +
gravedad (`C62._grav_peso` sin tocar, w=coldness=1-T) + difusión (v5) + memoria-de-enlace CS071
(refuerzo proporcional al roce real que cada enlace cargó ese paso). Primer smoke: la gravedad casi no
actuaba (grado del nodo más frío: 9→9). Encontré la causa: mi fórmula de freno gateaba `rate` por
(1-T_MEDIA GLOBAL) — convención correcta en el motor viejo (T compartida, baja igual para todos), pero
aquí T es un campo con una fracción ε diminuta más fría, así que la media global queda ≈1.0 SIEMPRE por
diseño, y ese gate apagaba la gravedad casi del todo sin importar cuán fría fuera la brizna. Lo corregí
(el freno ya está en `w` -- solo el frío pesa -- y en el CAP/E de siempre; quité el gate por media global).

## Con el bug corregido: la gravedad SÍ actúa (verificado), pero el CV sigue lavándose a 0
Grado del nodo más frío tras 10 pasos de gravedad sola: **9 → 299** (conectado a case todo el grafo,
N=300). La atracción funciona, medida. Pero corriendo el núcleo completo (gravedad+difusión+memoria), el
CV sigue cayendo a 0 en <10 pasos — IGUAL o peor que sin gravedad.

## El mecanismo, verificado, por qué pasa esto
`_grav_peso` es un constructor de TOPOLOGÍA — añade enlaces, nunca mueve T. Cuando el nodo frío gana 290
enlaces nuevos, mi intercambio (difusivo, SIMÉTRICO: ΔT=tasa·(T_j−T_i) en cada enlace) ahora tiene 290
canales por los que el calor de los vecinos tibios FLUYE HACIA el nodo frío — más grado en un proceso
difusivo significa acoplarse MÁS rápido a la media, no menos. El nodo que la gravedad hizo "hub" se
calienta más rápido que con 9 vecinos, no se enfría. Es lo opuesto de una inestabilidad.

**Diagnóstico:** tu propio toy (CV 1.96e-4→150, "gravedad") casi seguro no era topología-más-difusión-
simétrica — tuvo que mover VALORES de forma asimétrica (el nodo frío arranca T de sus vecinos más de lo
que un intercambio simétrico permitiría; un acaparamiento, no un promedio). `_grav_peso`, reusada tal cual,
construye la ESTRUCTURA por la que la inestabilidad PODRÍA propagarse (y eso funciona, medido: grado
9→299), pero no mueve T de forma que amplifique — para eso hace falta una regla de transferencia de VALOR
asimétrica (acreción), que `_grav_peso` no tiene y yo no quiero inventar por mi cuenta.

## Lo que NO hice
No inventé una fórmula de acreción propia. No until seguí ajustando parámetros de difusión/enfriamiento
buscando que el CV subiera — habría sido exactamente el Shannon que G-ENTROPIA-MONOTONA prohíbe.

## Pido adjudicar
¿Cómo se mueve el VALOR (T) de forma asimétrica/amplificante a lo largo de los enlaces que la gravedad
construye? Necesito la regla concreta (o el criterio para derivarla) que reproduzca en código lo que tu
toy ya demostró — no quiero adivinarla e imponerla yo, sería exactamente el error que este protocolo existe
para evitar.

— CC 🐝
