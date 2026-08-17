# Paso 1 — sensibilidad al settle (v7.4.1, modo `parada`)

Barrido de 20 puntos, luminosidad 0.25→1.95, modo=`parada`, measure=120,
semilla=1, `powerBase=0.47 beta=0.94 sigma=6.8 noise=0.0079 band=1.105 tOpt=25
ptcTc=18 ptcSharp=4.1`, día/noche apagado. `settle` en 150, 300, 600, 1200,
2400 (más un nivel extra 4800, no pedido, agregado para poder distinguir
"converge lento" de "no converge" — ver más abajo). Motor usado: `motor2.mjs`,
validado bit a bit contra el script real (`validacion2.md`, Tarea 5).

## Resultado headline: el eje NO es homogéneo frente al settle

La huella (`footprint`) y la entropía (`H_absLocal`) se comportan de forma muy
distinta según la zona del eje:

- **El resto del eje** (16 de los 20 puntos, todos menos k=6..9) se acerca a un
  valor estable de forma razonablemente rápida.
- **La zona de la caída de la huella** (k=6..9, x∈[0.787, 1.055] — el colapso
  en V que ya se había visto en la primera batería, justo donde vive la
  "frontera" que mide Experimento D) converge mucho más lento, y **todavía se
  está moviendo de forma apreciable en settle=2400**, con solo una mejora
  parcial al duplicar a 4800.

## Tabla — resto del eje (16 puntos, excluye la zona de colapso)

| transición | huella: dif abs media | huella: dif abs máx | huella: dif rel máx % | H: dif abs media | H: dif abs máx | H: dif rel máx % |
|---|---|---|---|---|---|---|
| 150→300   | 0.554 | 2.484 | 39.7  | 0.470 | 0.977 | 85.3  |
| 300→600   | 0.239 | 0.716 | 19.1  | 0.140 | 0.553 | 100.0 |
| 600→1200  | 0.105 | 0.381 | 9.3   | 0.081 | 0.463 | 46.5  |
| 1200→2400 | 0.062 | 0.289 | 8.7   | 0.080 | 0.534 | 36.6  |

La huella (medida en diferencia absoluta media y máxima) decrece de forma
consistente y aproximadamente geométrica al doblar el settle — es una
trayectoria de convergencia normal, solo lenta. La entropía, en cambio, **no
decrece limpiamente**: el `H_rel_max%` se queda alto (36–100%) incluso en la
transición 1200→2400. Esto no parece ser falta de convergencia física: `H` es
una entropía de Shannon con 24 bins, y un desplazamiento minúsculo de la
trayectoria puede mover una muestra de un bin a otro y saltar el valor de H de
forma discreta — el ruido de binning por sí solo puede producir saltos
relativos grandes en un punto aislado aunque el sistema ya esté físicamente
asentado. **La huella es el indicador más confiable acá; la entropía tiene un
piso de ruido propio del método de medición que no baja con más settle.**

## Tabla — zona de colapso (k=6..9, x=0.787→1.055)

| transición | huella: dif abs media | huella: dif abs máx | huella: dif rel máx % | H: dif abs media | H: dif abs máx | H: dif rel máx % |
|---|---|---|---|---|---|---|
| 150→300   | 0.523 | 0.890 | 128.6 | 0.191 | 0.416 | 21.2 |
| 300→600   | 0.492 | 0.737 | 203.2 | 0.288 | 0.617 | 40.0 |
| 600→1200  | 0.458 | 0.642 | 106.4 | 0.043 | 0.087 | 9.6  |
| 1200→2400 | 0.213 | 0.380 | 35.9  | 0.021 | 0.056 | 6.2  |
| 2400→4800 (extra) | 0.082 | 0.160 | 9.7 | 0.121 | 0.331 | 33.1 |

Acá la huella se queda prácticamente PLANA en diferencia absoluta hasta
settle=1200 (0.52 → 0.49 → 0.46, apenas baja), y recién empieza a decaer con
más fuerza entre 1200 y 4800 (0.46 → 0.21 → 0.08). El patrón — diferencia casi
constante durante mucho tiempo y después un decaimiento que se acelera — es
consistente con un **enlentecimiento crítico cerca de una bifurcación**: el
colapso de la huella es, físicamente, el sistema decidiendo entre dos regímenes
(vegetación que sobrevive o no), y cerca de ese punto de decisión el transitorio
tarda mucho más en apagarse que en cualquier otra parte del eje.

## Criterio aplicado y su resultado

Criterio propuesto (documentado antes de correr): cambio relativo máximo (sobre
los 20 puntos) por debajo de 1% tanto en huella como en entropía respecto al
nivel anterior, sostenido también en la transición siguiente.

**Ese criterio nunca se cumple dentro del rango probado (150→4800)**, ni
siquiera en el resto del eje — por el ruido de binning de la entropía descrito
arriba, que no baja con más settle. Si se mira solo la huella (más confiable),
el resto del eje sí converge razonablemente rápido (dif. abs. media <0.1 desde
settle=1200 en adelante), pero **la zona de colapso sigue moviéndose de forma
sustancial hasta 2400, y todavía cambia ~10% entre 2400 y 4800** — la mejora es
real pero lenta, sin señal clara de haber tocado un piso dentro del rango
practicable de settle para una batería de cientos o miles de barridos (un
barrido de 20 puntos a settle=4800 ya tarda 63s; a 60 puntos con el rango
completo de D/A'/B' y settle mucho mayor, cada barrido individual se iría a
varios minutos, y la batería completa (mínimo 1.140 barridos) a muchas horas o
días).

## No se recomienda un settle único para todo el eje

**Siguiendo la regla del encargo ("si a 2400 todavía se mueven, dilo y no
sigas"): la zona de colapso (k=6..9, exactamente donde vive la frontera que
mide Experimento D) todavía se mueve de forma apreciable en settle=2400, y no
alcanza a estabilizar limpiamente ni siquiera duplicando a 4800.** No propongo
un valor de settle para usar en Experimento D / A' / B' — sería medir esa zona
con un instrumento que sabemos que todavía no asentó, exactamente en el punto
que el experimento existe para medir.

Esto no es un defecto de cálculo del instrumento (las fórmulas están
validadas bit a bit, ver `validacion2.md`) — es una propiedad física del
sistema cerca del colapso, y el protocolo de "settle fijo igual para todo el
eje" no la resuelve. Replantear el protocolo es una decisión que le toca al
investigador principal (opciones que se me ocurren pero que NO voy a decidir
por mi cuenta: settle variable por zona del eje, un criterio de parada
adaptativo por convergencia en vez de un conteo fijo de pasos, o aceptar la
zona de colapso como intrínsecamente ruidosa y medirla con otro diseño).

## Velocidad medida (para dimensionar cualquier decisión)

Con `motor2.mjs`, barrido de 20 puntos, modo=parada: ~1400–1600 pasos/s en esta
máquina (ver tabla completa en `validacion2.md`). A 60 puntos (el rango real de
D/A'/B') y con el settle que se termine eligiendo, el tiempo por barrido escala
aproximadamente lineal con `steps × (calSteps+settle+measure)`.
