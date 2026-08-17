# Encargo para Claude Code · segunda batería EIT-3 Térmico

Archivo: **`EIT3_Termico_kappaH_v7.5.html`**

Tu trabajo anterior se conservó íntegro: el generador `mulberry32`, el ruido del
campo dependiente de la semilla y la separación `pasoFisica()` /
`updateSimulation()` / `stepHeadless()`. Nada de eso se tocó.

Lo que se agregó encima, y por qué, está abajo. **La primera batería tiene un
resultado invalidado y este encargo existe para corregirlo.**

---

## Lo que se descubrió analizando tu batería

La frontera de la huella caía en la misma casilla del eje en las 510 corridas,
con desviación de 10⁻¹⁵, sin moverse al variar ningún parámetro.

Reproduje el barrido por fuera y encontré la causa: **el barrido reiniciaba el
campo y los búferes de medición pero no el estado del sistema.** `black`,
`white`, `Tf` y `powerLive` llegaban heredados de la parada anterior — las
cuatro variables que determinan la huella. Reproducido con arrastre, el mínimo
queda clavado para toda combinación de parámetros; reproducido con reinicio en
cada parada, se mueve y sí depende de ellos.

**Esa invariancia era del método de barrido, no del sistema.** Es el resultado
que hay que rehacer.

## Lo que cambió en el archivo

- **Selector de punto de partida** (`sweepReset`), con tres modos:
  `parada` (reinicia en cada parada, por defecto), `inicio` (reinicia solo al
  empezar y después arrastra), `ninguno` (el comportamiento anterior). El modo
  queda registrado en la bitácora.
- **Deslizador de semilla** (`seedInput`) en la interfaz. Tu `setSeed()` estaba
  pero sin forma de llamarla a mano.
- El generador **se siembra desde el arranque**. Antes quedaba en `Math.random`
  hasta que alguien llamara a `setSeed`, así que una corrida hecha sin tocar la
  semilla no era reproducible ni sabiéndolo.
- `resetSimulation()` ahora limpia también `_Awin`, la ventana de doce muestras
  del error operativo, que sobrevivía al reinicio.

---

## Paso 0 · Verificación, antes de correr nada

Ya la hice en navegador y debe darte lo mismo. Si no, para y reporta.

Con `desde=0.6 hasta=1.4 puntos=8 settle=40 measure=30 trazas=0`:

| semilla | modo | resultado esperado |
|---|---|---|
| 7 | parada | archivo X |
| 7 | parada | **idéntico a X, byte a byte** |
| 99 | parada | distinto de X |
| 7 | inicio | distinto de X |

**Si dos corridas con la misma semilla y el mismo modo no dan archivos
idénticos, hay una fuente de azar sin sembrar y todo lo demás no vale.**

---

## Paso 1 · Ya está resuelto, y cambió el instrumento

Hiciste bien en detenerte. Lo que encontraste no es un problema numérico: es
**ralentización crítica**. Medí el mecanismo por fuera y el máximo del tiempo de
asentamiento cae donde las margaritas negras se extinguen. Es una bifurcación, y
la lentitud cerca de ella es el indicador de alerta temprana descrito por
Scheffer y otros (*Nature*, 2009).

**Por eso el `settle` deja de ser un ajuste y pasa a ser la medición.** La v7.5
agrega cinco columnas al barrido:

| columna | qué es |
|---|---|
| `pasos_recuperacion` | cuántos pasos tarda la población de margaritas en volver tras un golpe de 0,03, promediado sobre 5 golpes |
| `recuperaron_todos` | 1 si los cinco golpes se recuperaron dentro del tope, 0 si alguno no |
| `tasa_recuperacion` | su inverso |
| `varianza_pl` | varianza de powerLive, **con la tendencia quitada** |
| `autocorr1_pl` | autocorrelación de retardo 1, también sin tendencia |

Tres cosas que costaron dos intentos y conviene que sepas, porque explican el
diseño:

- **No se mide «cuándo deja de moverse».** Con ruido estocástico el estado nunca
  se detiene: el término aleatorio mueve la temperatura ±0,055 por paso. Se mide
  recuperación ante una perturbación, que sí está definida.
- **Se golpea la población, no la temperatura.** La temperatura se recupera en
  unos quince pasos en todo el eje y no informa nada. La variable lenta es la
  vegetación.
- **Se le quita la tendencia antes de la varianza y la autocorrelación.** Sin
  eso la deriva hacia el equilibrio domina y la autocorrelación sale ~0,99 en
  todo el eje, que fue lo que pasó en el primer intento.

Verificado en navegador, 24 puntos de 0,6 a 1,4, semilla 7:

```
lum 0,600–0,739   ~4.000–5.100 pasos   no se recuperan dentro del tope
lum 0,774          198 pasos
lum 1,017           48 pasos   ← el mínimo
lum 1,400           70 pasos
```

**108 veces de diferencia entre el punto más lento y el más rápido.** Y hay un
tramo, bajo luminosidad 0,77, donde ni siquiera se recupera dentro del tope de
20.000 pasos: la columna `recuperaron_todos` viene en 0. Eso es un dato, no un
fallo — significa que en ese tramo el sistema no vuelve.

**No repitas la prueba de sensibilidad al settle.** Ya no hace falta: el settle
solo controla cuánto se deja asentar antes de golpear, y la medición ya no
depende de él.

## Paso 2 · Experimento D — el modo de reinicio como factor

Es el experimento que demuestra o descarta el artefacto. Va primero.

El mismo barrido, **10 semillas × 2 modos** de reinicio: `parada` e `inicio`.

**El tercer modo, `ninguno`, queda fuera del experimento a propósito.** Existe en
la interfaz para poder reproducir el comportamiento antiguo a mano, pero no sirve
como factor: solo se distingue de `inicio` si el barrido arranca con estado sucio,
y en una tanda automática eso no ocurre — o si ocurre, el resultado depende del
orden en que se corrieron las cosas, que es justamente lo que no queremos medir.
Si lo corres igual, decláralo aparte y no lo promedies con los otros dos.

```
eje       luminosidad, de 0,25 a 1,95
paradas   60
settle    el que salió del paso 1 · measure 120
Tc PTC 18 · exponente PTC 4,1 · día/noche APAGADO
potencia base 0,47 · beta 0,94 · sigma 6,8 · ruido 0,0079 · banda 1,105 · tOpt 25
```

Qué reportar, por modo:

- posición del mínimo de la huella: media y desviación entre semillas
- correlación huella ↔ entropía: media y desviación
- si la posición del mínimo se mueve entre semillas en cada modo

**Predicción a poner a prueba:** en modo `inicio` la posición del mínimo no se
mueve entre semillas (desviación ~0, como pasó en tu primera batería); en modo
`parada` sí se mueve. Si sale al revés, mi diagnóstico está equivocado y quiero
saberlo.

Referencia medida en navegador con 8 paradas, semilla 7: el mínimo cae en k=5
(lum 1,171) con `parada` y en k=4 (lum 1,057) con `inicio`. Los dos modos son
reproducibles: dos corridas seguidas con la misma semilla dan el CSV byte a byte
idéntico.

---

## Paso 3 · Experimento A' — repetición, ahora con reinicio

Igual que tu experimento A: 30 semillas, parámetros fijos, **modo `parada`**.

Qué reportar: correlación huella ↔ entropía (media y desviación) y posición del
mínimo (media y desviación). Comparar contra los valores que te dieron con el
método anterior: r = −0,236 ± 0,073, mínimo invariante.

---

## Paso 4 · Experimento B' — multivariable, rediseñado

**Sigma sale de la grilla.** Verifiqué que no puede influir: solo entra en
`evolveField`, que alimenta `deltaStruct`, y ni la huella ni la entropía de la
conducta tocan el campo bidimensional. Tus propios datos lo confirman — sigma
dio `r = −0,2432 ± 0,0647` idéntico a cuatro decimales en sus cuatro niveles.
Un tercio del experimento midió el efecto de algo sin efecto, y la culpa es del
encargo anterior, no tuyo.

**`band` tampoco sirve, por la misma razón.** No lo uses como reemplazo.

Grilla nueva, con un parámetro por cada variable medida:

- **persistencia (beta)**: 0,80 · 0,88 · 0,94 · 0,98 — llega a la huella por el amortiguamiento
- **temperatura preferida (tOpt)**: 22 · 25 · 28 — llega a la huella por dos vías: la curva de crecimiento de la vegetación y el amortiguamiento térmico
- **brusquedad del sensor (ptcSharp)**: 3,0 · 4,1 · 6,0 — llega a la entropía por la respuesta del sensor
- **potencia base**: 0,30 · 0,47 · 0,65 — llega a ambas

Son 4 × 3 × 3 × 3 = **108 combinaciones**. Con 10 semillas, 1.080 barridos.
Modo `parada`.

**Vigila la saturación:** mover `tOpt` y `ptcSharp` puede sacar al sensor de
rango. La bandera `saturacion_sensor` está corregida desde la v7.2 y detecta
los dos topes. **Reporta qué fracción de puntos queda saturada en cada
combinación, y descarta de los promedios las que superen el 10 %** — un barrido
con el sensor topado no mide nada, y hay que poder decir cuáles fueron.

---

## Paso 5 · Experimento C — barajado

Igual que antes, 1.000 barajes por serie, sobre A' y sobre B'.

En tu batería anterior el resultado fue claro y conviene tenerlo de referencia:
72,1 % de las combinaciones caían fuera del 95 % y 28,5 % fuera del 99 %, contra
el 5 % y 1 % esperables sin efecto. La correlación sobrevivía al barajado.
**Comprueba si sigue sobreviviendo con el método corregido.**

---

## Presupuesto de cómputo, y una decisión que puede hacer falta

Estimado a 2.624 pasos por segundo de física pura, que es lo medido en Node.
El número de barridos no depende del `settle`, pero el tiempo sí — y el paso 1
puede obligar a subirlo:

| settle | por barrido | D + A' + B' = 1.140 barridos |
|---|---|---|
| 300 | 9,6 s | **3 h** |
| 600 | 16,5 s | 5 h |
| 1.200 | 30,2 s | **9,5 h** |

Si el paso 1 exige un `settle` de 1.200 o más, la grilla de B' queda cara.
En ese caso **no bajes las semillas** —la dispersión entre semillas es lo que
distingue un hallazgo de una casualidad y es lo que ya nos falló una vez—:
recorta la grilla. Propuesta, si hay que recortar: deja `tOpt` en dos niveles
(22 y 28) y `ptcSharp` en dos (3,0 y 6,0), lo que baja a 48 combinaciones y
480 barridos. Reporta qué recorte hiciste.

---

## Qué no te toca

**No concluyas.** Estadística descriptiva y nada más; la interpretación es del
investigador principal.

Si encuentras otro defecto del instrumento —van cinco hasta ahora, tres de
cálculo y dos de método— **detente y repórtalo antes de correr las baterías
completas**.

Y si algo de este encargo te parece mal planteado, dilo. El error de sigma
estuvo doce horas dentro de un experimento de 480 corridas porque nadie lo
cuestionó.
