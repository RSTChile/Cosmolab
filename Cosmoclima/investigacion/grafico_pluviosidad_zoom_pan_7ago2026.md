# Gráfico de Pluviosidad en Vivo: rehecho con zoom/pan real (07-ago-2026)

Alexis, tras varias rondas previas (v2/v3/v4, ver comentarios en el propio
HTML) sin quedar conforme: "¿habría posibilidad de mejorarlo para que se
pueda hacer zoom sobre un período específico, o ver todos los datos
completos en una sola vista, o avanzar y retroceder manualmente?"

## Qué se sacó
El sistema viejo (v4, 01-ago-2026): 4 botones Día/Semana/Mes/Año que
cambiaban la AGREGACIÓN de los datos, un `<div overflow-x:auto>` con ancho
en píxeles fijo por modo (hasta ~15.000px), y arrastre/rueda hechos a mano
para mover el scroll del div. Se sacó todo: los 4 botones, `bucketsAnio/
Semana/DiaAnio`, `actualizarAnchoPopChart`, `wireScrollPopChart`,
`popChartScroll`/`popChartInner`.

## Qué se puso
[chartjs-plugin-zoom](https://www.chartjs.org/chartjs-plugin-zoom/) sobre
un eje x numérico real (día-calendario, ya no categorías de texto) — la
resolución de los datos queda SIEMPRE mensual (el dato real, 744 puntos),
y el zoom decide cuánto rango se ve, no una re-agregación:

- **Rueda del mouse / pellizcar en trackpad** = zoom in/out centrado en el cursor.
- **Clic y arrastrar sobre el gráfico** = zoom a la región exacta que arrastraste (rectángulo).
- **Botones ◀ Retroceder / Avanzar ▶** = mover la ventana visible sin cambiar el zoom.
- **Botón ↺ Ver todo** = volver al rango completo 1966–2027 de un clic.

## Un hallazgo real en el camino: el "pan" por arrastre del plugin no respondía
Se probó exhaustivamente (eventos sintéticos Mouse/Pointer, arrastre real
del sistema vía automatización, con y sin hammer.js, dos versiones del
plugin) — `pan.enabled:true` + arrastrar **nunca** movió el gráfico, aunque
la configuración era correcta. En cambio, **la API programática
`chart.pan({x:...})` sí funciona siempre**, y **drag-to-zoom (arrastrar
para seleccionar una región y hacerle zoom) también responde perfecto** con
los mismos eventos que el pan no procesaba. En vez de dejar una función
"debería andar" sin confirmar, se armó la solución con las dos piezas que
SÍ se verificaron funcionando: arrastrar = zoom a región, botones ◀▶ = pan
programático. Los 3 pedidos de Alexis quedan cubiertos por mecanismos
probados uno por uno, no por fe en la librería.

## Verificado en vivo (Chrome, no solo el código)
Secuencia completa probada en el navegador: alejar con rueda → zoom por
arrastre a una región de ~4,5 años → Retroceder → Avanzar (vuelve exacto al
mismo rango, confirmado bit a bit) → Ver todo (vuelve a 0–22629 días,
1966–2027 completo). Sin errores de consola.

## Actualización 08-ago-2026: barra de arrastre real (lo que faltaba)
Alexis probó lo de arriba y dijo: "el slider todavía no permite barrer hacia
adelante o hacia atrás, independientemente del zoom que apliques". Aclaramos
que se refería a la VISTA del gráfico (no a la barra vieja "Recorrer
1966-2027 a mano", que es otra cosa: mueve el reloj de la simulación, no el
gráfico). Los botones ◀▶ ya cubrían esto a saltos fijos; lo que faltaba era
una barra de scroll de verdad, arrastrable con el mouse, para barrer
continuo con cualquier zoom puesto.

Se agregó una barra celeste justo debajo del gráfico (`#popPanTrack` +
`#popPanThumb`): el ancho del "pulgar" representa qué fracción del rango
completo (1966–2027) se está viendo (se achica al hacer zoom), y arrastrarlo
mueve la ventana visible sin tocar el nivel de zoom — usando
`chart.zoomScale()` (API pública), no el `pan.enabled` del plugin que nunca
respondió a arrastre real (ver arriba). También se puede tocar directo en la
barra (fuera del pulgar) para saltar ahí. Verificado con eventos de puntero
reales en el navegador: el arrastre del pulgar sí responde y mueve el
gráfico exactamente lo esperado (delta en píxeles → delta en días,
comprobado que coinciden).

**Un bug real que se coló y se corrigió antes de entregar**: al conectar la
barra nueva, quedó una función (`syncPopPanUI`) llamada desde afuera de
`makeCharts()` sin estar declarada ahí — un error de alcance de variables de
JavaScript. Eso rompía la carga de la página a mitad de camino: el
precálculo de Floración/Gyriosomus (3-4s) nunca llegaba a ejecutarse, así
que esa curva habría aparecido vacía. Se detectó probando con una recarga
limpia del archivo (no alcanzaba con mirar el código), no con la consola del
navegador (que no mostró el error — quedó como aviso aparte para no confiar
ciegamente en esa herramienta la próxima vez), sino insertando una marca
antes/después de la llamada para confirmar dónde se cortaba la ejecución.
Se corrigió moviendo la sincronización inicial adentro de `makeCharts()`, y
se volvió a probar con una carga 100% limpia: el precálculo corre bien (744
puntos) y la barra arranca sincronizada.

## Actualización 08-ago-2026 (2): se sacó el reloj manual viejo
Alexis, pensando en la analogía con Daisyworld: ahí el reloj mueve el sol
(un driver hipotético que no existe hasta que se simula). Acá lo que se
recorre es un período real y acotado (1966-2027) con datos concretos, así
que un control aparte para "saltar" a mano por ese período ya no tenía
sentido — quedaba redundante con la barra celeste nueva, que recorre ese
mismo rango real. Se sacó toda la barra "Recorrer 1966-2027 a mano"
(`#popScrub`, `#popScrubStop`, `saltarAFecha()`, `syncPopScrubUI()`,
`diaVivoActual()`, `saltoActive`) del HTML y del JS.

**Lo que NO se tocó**: el reloj de verdad (Iniciar/Pausa/velocidad de
simulación) sigue intacto. Es un caso distinto: el panel "Estado del
experimento" (Tf, régimen, el planeta) es un modelo físico con memoria
día a día — la temperatura de hoy depende de la de ayer, no se puede
"consultar" un día suelto sin haber corrido todos los anteriores — así que
sí necesita algo que avance de verdad, aunque el rango de fechas ya sea
conocido. Verificado en el navegador tras el cambio: la barra vieja ya no
existe en el DOM, el precálculo de Floración/Gyriosomus sigue corriendo
bien, y Iniciar/Pausa/Reiniciar del reloj físico funcionan igual que antes.

## Actualización 08-ago-2026 (3): se sacaron los botones ◀▶/Ver todo
Con la barra celeste ya andando, Alexis notó que los botones "↺ Ver todo",
"◀ Retroceder" y "Avanzar ▶" quedaron redundantes — la barra + la rueda del
mouse ya cubren exactamente lo mismo. Se sacaron los tres del HTML y su
wiring del JS (`panPopChart()` completo). Sin botón "Ver todo" explícito: se
vuelve al rango completo alejando con la rueda hasta el tope (el plugin no
deja pasarse de 1966-2027 por `limits.x`, verificado que efectivamente
clampa en exactamente `{min:0,max:22629}`). Verificado en el navegador:
zoom-in con rueda, arrastre de la barra, y zoom-out hasta el rango completo
funcionan igual de bien sin los botones.

## Lo que NO cambió
- La curva única del reloj (`PLUVIOSIDAD_MENSUAL`, Huintil/CR2+NASA POWER)
  sigue exactamente igual — este cambio es solo de visualización.
- Las líneas de presencia por especie y de lluvia de la estación más
  cercana (agregadas 05/06-ago) siguen funcionando igual, ahora sobre el
  mismo eje x numérico compartido.
- El control "Recorrer 1966-2027 a mano" (que mueve el RELOJ de la
  simulación, no la vista del gráfico) no se tocó — es una función distinta.
