# Barrido automático y Evolución temporal unidos a datos reales — 09-ago-2026 (3)

Alexis: *"hay que unir el barrido de Evolución temporal con los datos
reales, y el resto de los parámetros de la simulación también, para ver
qué sucede"*. Investigación previa (Explore) encontró que esto no era solo
"falta de conexión" — había un bug real y una limitación de diseño de
por medio. Plan completo en `/Users/alexis/.claude/plans/majestic-whistling-canyon.md`.

## Hallazgo 1 — bug real: el eje "Lluvia acumulada" del barrido no hacía nada

Desde que la lluvia se volvió 100% calendario-automática (01-ago-2026),
`els.lluvia` quedó `disabled` y nunca se leyó de vuelta a `state.
lluviaAcumulada`. El eje "Lluvia acumulada (Gyriosomus)" del barrido
(`sweepAxis`) escribía ese campo y no pasaba nada — código muerto.
**Reemplazado** por un eje que sí funciona: **"Año calendario real (barre
la historia)"** — interpola un AÑO real entre Desde/Hasta y siembra el
reloj interno en ESE año en cada parada.

## Hallazgo 2 — por qué CUALQUIER barrido siempre arrancaba en 1966 seco

Cada parada llama `reiniciarSilencioso()`, que reseteaba `state.tick=0` —
como el calendario real se deriva de `tick`, TODA parada arrancaba el
1-ene-1966 (lluvia real de ese día: 0mm), sin importar qué eje se barriera.
Luminosidad sí se movía, pero siempre contra el mismo día sintético seco.

**Arreglado**: `reiniciarSilencioso(tickInicial)` ahora acepta un tick de
arranque opcional. Nuevo control **"Anclar a un año real (0=no anclar)"**
aplica a CUALQUIER eje (Luminosidad, Sobre, Ambos) — con un año real puesto
ahí, cada parada arranca en el 1 de enero de ese año, con lluvia/Día-Noche/
Estaciones reales de ese punto de la historia. El año-ancla usado queda
trazable en el CSV exportado (`anioAncla`, nueva columna).

**Verificado con datos reales** (vía consola, no la UI, para aislar la
lógica): anclar a **julio de 1997** (mega Niño) da 111,8mm de lluvia real
acumulada → floración objetivo 63%. Anclar a **julio de 2019** (sequía) da
3,4mm → floración objetivo 0%. Antes de este arreglo, cualquier barrido
habría dado el mismo día sintético siempre, sin importar el año.

## Hallazgo 3 — por qué "Evolución temporal" nunca mostraba la historia completa

`evoChart` solo grafica los últimos 120 puntos de `state.history`
(máx. 6000, "~100 días"), que solo se llena en vivo, tick a tick. El único
mecanismo que corre el calendario real completo 1966-2027,
`calcularDistribucionRegimen()` ("Experimento Completo"), llama
`pasoFisica(false)` a propósito para SALTARSE ese guardado (sería carísimo
sobre 1,36 millones de ticks) — y solo guardaba 4 números para percentiles,
descartando Tf/envTemp/stress/H(a_t)/Λ_exp cada tick.

**Arreglado**: la misma función ahora ADEMÁS guarda un valor por DÍA
calendario real (no por tick — 22.631 días, liviano, ~900KB) de esas 5
variables, sobreescribiendo en cada tick del día (el último tick "gana",
mismo truco que ya usaba `conteoPorAnio`). Se reusa la corrida cara que ya
se estaba haciendo — no se duplica el cálculo. Resultado queda en
`EVOLUCION_REAL_COMPLETA` (global, `null` hasta la primera corrida
completa).

**Nuevo toggle** en "Evolución temporal": **"Historia real completa
(1966-2027)"** — cambia el gráfico de la sesión en vivo (eje de categorías,
últimos ~100 días) a la corrida real completa (eje de fecha real, mismo
patrón de `popXScale` ya usado en otros gráficos). Si se activa antes de
correr "Experimento Completo", muestra un aviso claro en vez de fallar
silenciosamente.

## Verificación

- **Ancla de año real**: confirmado con datos reales (arriba) que 1997 y
  2019 dan lluvia acumulada y floración-objetivo distintas — antes del
  arreglo habría dado lo mismo siempre.
- **Toggle sin datos**: confirmado que muestra el aviso y NO rompe nada,
  se queda en modo vivo (sin errores de consola).
- **Mecanismo de guardado diario**: verificado con una corrida real
  sincrónica de 45 días (junio-julio 1997, sin pasar por la UI para evitar
  el throttling de pestaña en segundo plano — ver nota abajo) — lluvia real
  sube de 12,3mm a 111,8mm, Tf varía genuinamente día a día (45/45 valores
  distintos, no un valor congelado).
- **Renderizado del gráfico histórico**: verificado con ese mismo resultado
  real inyectado en `EVOLUCION_REAL_COMPLETA` — el toggle cambia la escala
  a "Fecha real (1966-2027)", el conteo de puntos calza, y un punto de
  muestra mapea exacto a su fecha real (1997-06-21). Sin errores de consola.

### Nota honesta: no se completó una corrida real de 1966-2027 vía navegador automatizado

Intenté correr "Experimento Completo" completo para verificar de punta a
punta, pero la pestaña de Chrome controlada por automatización quedó en
segundo plano y Chrome frena agresivamente los `setTimeout` de pestañas no
visibles — el progreso cayó a ~7 ticks/seg en vez de los ~900-2000/seg ya
medidos antes en sesión normal (no es un bug del código nuevo: confirmado
viendo que `state.tick` apenas avanzaba). Cancelé esa corrida y verifiqué
la MISMA lógica con una corrida real corta (45 días, síncrona, sin
esperas) en su lugar — prueba honesta del mecanismo, no una corrida
completa. **Cuando Alexis corra "Experimento Completo" en su propio
navegador (pestaña activa, en primer plano), debería andar a la velocidad
normal de siempre** — vale la pena que él la corra una vez para confirmar
el resultado completo con sus propios ojos.

## Archivos modificados

- `Web/prueba_de_concepto/prueba_de_concepto_ET3-Termico_con_mapa.html`:
  UI del barrido (eje nuevo + ancla), `RANGO_EJE`, `setAxisValue`,
  `reiniciarSilencioso`, `runSweep`, `exportSweepCSV`,
  `calcularDistribucionRegimen` (arrays diarios nuevos), `updateCharts`
  (dos modos), toggle nuevo en la sección Evolución temporal.
