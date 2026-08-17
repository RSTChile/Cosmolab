# Pluviosidad real mensual, ZHCS 1900-2027

Fecha: 01-ago-2026. A pedido de Alexis: la floración salía en flor casi todo el
tiempo (1915-2027) porque la curva empírica se había calibrado con lluvia ANUAL
de una serie más seca (NASA POWER, corredor Huasco-Freirina, típico 7-140mm/año),
pero el calendario real termina consultando la estación Huintil (típico 70-540mm/año)
— dos series con magnitudes muy distintas. Antes de tocar la curva de floración,
Alexis pidió volver al dato más sólido que hay: **pluviosidad real, en la mejor
resolución que exista** — "esa es la primera curva real". Datos en
`fuentes/lluvia_mensual_zhcs_1900_2027.csv`.

## La resolución real disponible (no se sabía hasta revisar las fichas)
- **CR2 (estación Huintil, 1900-2019)**: la ficha oficial del dataset dice
  "Resolución Temporal: Mensual" — no es un dato diario, pero tampoco es solo un
  total anual como se venía usando. Ya estaba descargado (`cr2_prAmon_2019/`), solo
  se había colapsado a suma anual sin necesidad.
- **NASA POWER (2019 en adelante)**: la API que ya se usa (`datos_clima.py`) es de
  resolución DIARIA — se agregó a mensual para que las dos fuentes calcen en la
  misma unidad de tiempo, pero el dato diario existe si hace falta más adelante.

No hay dato diario real para 1900-2019 (la estación en tierra solo reporta por
mes) — se optó por MENSUAL como grano uniforme para todo el rango 1900-2027, en
vez de mezclar resoluciones distintas en una misma curva.

## Fuente y relleno (mismo criterio que la tabla anual ya existente)
Huintil (CR2 4723002) primero; Lautaro Embalse (CR2 3430006) solo rellena los
meses de 1933-1937 donde Huintil no tiene dato (30 de 60 meses de esa ventana);
NASA POWER (mismo punto ZHCS, -30.6,-71.2) desde 2019-01. Verificado: sumar los
12 meses reales de cualquier año ya publicado en `lluvia_anual_zhcs_1900_2027.csv`
da EXACTAMENTE el mismo total anual (diferencia 0.0mm en los 67 años cruzados) —
la tabla mensual no contradice la anual, la refina.

## Resumen
1536 meses totales (1900-01 a 2027-12): **1057 con dato real (69%), 479 sin dato
real (31%)**. Igual que con la tabla anual, los huecos se muestran como huecos
reales en el gráfico (no como cero, no interpolados) — `spanGaps:false` en Chart.js.

## Hallazgo que la vista anual escondía
Julio de 2026 (El Niño "Godzilla" en curso) trae **408mm en un solo mes** — la
vista anual ya mostraba ~415mm para todo 2026, pero la vista mensual muestra que
es prácticamente UN SOLO EVENTO, no lluvia repartida en el año. Es exactamente el
tipo de señal (pulso concentrado, no promedio anual) que la hipótesis H2 de
Cosmoclima ya señalaba como la variable que probablemente importa más para el
Desierto Florido — pendiente de usar esto para reconstruir la curva de floración,
a propósito no se hizo todavía (Alexis pidió mostrar bien la pluviosidad primero).

## Qué cambia en el simulador
`prueba_de_concepto_ET3-Termico_con_mapa.html`: nueva constante
`PLUVIOSIDAD_MENSUAL` (clave "AAAA-MM" → mm o `null`). El gráfico que antes se
llamaba "Floración y Gyriosomus en vivo" pasa a llamarse **"Pluviosidad real en
vivo"** y grafica esta tabla directo — a diferencia del motor simulado, no
depende de que la simulación haya "corrido" hasta esa fecha: se ve el rango
1900-2027 completo desde que se abre la página. El control "Agrupar línea de
tiempo" se simplificó a Mes/Año (ya no Día/Semana, que habrían inventado una
precisión que el dato real no tiene). `LLUVIA_HISTORICA` (anual) y
`computeFloracion()` NO se tocaron — la floración sigue calculándose por debajo
(sigue moviendo Tf, la física real del planeta) pero dejó de mostrarse en este
gráfico mientras se decide cómo reconstruirla sobre la base mensual.
