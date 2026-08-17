# Defectos y decisiones de diseño — batería EIT-3 Térmico κ_H

Ninguno de los puntos siguientes fue bloqueante: la verificación bit-a-bit
(tareas 3 y 4, ver `validacion.md`) pasó en todos los casos probados, así que
la batería completa se corrió. Se documentan igual porque cambian cómo hay
que leer los resultados.

## 1. `runSweep()` original nunca llama a `resetSimulation()` (defecto de reproducibilidad)

En el HTML, el botón "▶ Correr barrido κ_H" invoca `runSweep()`, que arranca
directo desde el estado que haya en ese momento — el que dejó la última
interacción manual (mover un slider, correr manualmente, etc.). Ni siquiera
"la única corrida limpia" mencionada en el encargo tiene garantizado un
estado inicial reproducible: si alguien tocó algo antes de apretar el botón,
esa corrida partió de un punto distinto.

**Corrección aplicada en la batería:** `motor.mjs` (vía `correr_barrido.mjs`)
llama a `resetSimulation()` una vez al inicio de cada barrido completo
(semilla × combinación de parámetros), antes de la pasada de calibración y
antes de la pasada de medición. El reset parcial por punto del eje
(`resetField()`, `aBuf=[]`, `noiseEchoBuf=[]`, `_A_prev=0`, `_Awin=[]`) se
preservó tal cual el original: la vegetación (`black`/`white`/`bare`) y `Tf`
NO se reinician entre puntos del mismo barrido, solo entre barridos.

No se tocó el HTML para agregar esta llamada porque cambiaría el
comportamiento interactivo del simulador (el usuario puede querer arrancar
un barrido desde un estado ya asentado a propósito); el fix vive solo en el
motor de la batería.

## 2. `pseudoNoise(x,y,t)` no usa `Math.random()` — decisión sobre cómo seedearla

El encargo la lista como uno de los "tres lugares que usan azar", pero es un
hash determinístico de `(x,y,t)`: `sin(x*12.9898+y*78.233+t*0.021)*43758.5453`,
fraccionado. A igual `(x,y,t)` da siempre el mismo valor — no consume
`Math.random()` en ningún punto.

**Decisión:** en vez de reemplazarla por `rng()` (lo que habría cambiado la
estructura espacial del ruido de correlacionado-por-hash a i.i.d. por celda,
alterando la física), se le inyectó la semilla como offset determinístico del
argumento temporal: `pseudoNoise(x, y, state.tick + state.seed*1013.9)`. Esto
preserva exactamente la fórmula y el rango original, y logra el efecto que sí
pedía el encargo — que **distintas semillas produzcan campos de ruido
distintos** (verificado en `validacion.md`: mismo número de pasos, semillas 42
y 777 dan `field` con checksums y sumas distintas). El término estocástico de
`Tf` y `passiveNoiseSample()` sí son usos reales de `Math.random()` y se
reemplazaron directamente por `rng()` del único `mulberry32(seed)`.

## 3. Saturación del sensor, solo en el extremo superior del eje

En Experimento A (30 semillas), la bandera `saturacion_sensor=1` aparece
**únicamente** en `luminosidad=1.950` (el último punto del barrido, 30/30
semillas), 0 veces en el resto de los 59 puntos. Con `tc_ptc=18` y
`exponente_ptc=4.1` (los valores que pide el encargo, distintos del default de
la UI que es `tc_ptc=25`, `exponente_ptc=8`), el sensor llega a su techo justo
en el borde de arriba del rango pedido. No se tocó el rango del eje (0.25→1.95
es el que especifica el encargo) — se deja constancia para que el punto
`luminosidad=1.95` de cada barrido se pueda descartar sin volver a correr
nada, tal como pide el encargo ("hay que poder descartarla sin volver a
correrla").

## 4. El primer criterio de "frontera" probado enganchaba el artefacto del punto 3, no el colapso real

Para la Tarea 9 se probó primero "el punto del eje con la caída más
pronunciada entre dos pasos consecutivos de huella". Ese criterio eligió,
para las 30 semillas, el último intervalo del eje (1.921→1.950) — exactamente
el intervalo donde el sensor se satura (punto 3), no el colapso biótico real.
Se descartó y se reemplazó por "mínimo global de huella, excluyendo puntos
con `saturacion_sensor=1`", que sí ubica el colapso genuino (una caída en V
alrededor de `luminosidad≈1.0` que se recupera después). Ver metodología en
`resumen_descriptivo.md`.

## 5. La correlación de −0,756 no es directamente comparable sin aclarar el rango del eje

El encargo compara los resultados de Experimento A contra "la única corrida
que tenemos" (−0,756), pero no especifica en qué rango de luminosidad se
corrió esa referencia. El preset por defecto de la UI del simulador es
`0.60→1.40` (ver `sweepAxis.onchange` en el HTML), bastante más angosto que el
`0.25→1.95` que pide Experimento A — y la huella tiene una forma no monótona
(cae en V cerca de `luminosidad≈1.0` y luego sube fuerte hasta ~9 en el
extremo superior). Correlacionar sobre un tramo angosto alrededor del colapso
da un número distinto que correlacionar sobre el eje completo, que incluye la
recuperación pronunciada del lado derecho. Esto no es un defecto del
instrumento — es una diferencia metodológica entre la corrida de referencia
(rango desconocido) y la batería (rango fijado por el encargo). Se reporta
para que el investigador principal decida si quiere una corrida adicional con
el rango 0.60→1.40 para comparar manzanas con manzanas.

## 6. Sin NaN / Infinity / resultados absurdos

Se revisaron los 1.800 registros de Experimento A: cero filas con `NaN` o
`Infinity` en cualquier columna. (Experimento B se revisó igual antes de
darlo por bueno; ver `resumen_descriptivo.md` para el conteo.)
