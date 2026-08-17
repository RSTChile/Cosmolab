# Informe de sesión — Instrumento EIT-3 Térmico: granularidad fina, lluvia real, El Niño
### 08 y 09 de agosto de 2026

## Resumen para retomar mañana

En dos días se pasó de un instrumento con **lluvia anual congelada y
reportes solo por año**, a uno con **lluvia diaria real donde existe
(1966–mayo 2017), promedio mensual real donde no (2017–2027), calibrado
contra el caso real (no contra el Daisyworld original), verificado número
por número contra un motor portado a Node.js, y con los episodios reales
de El Niño/La Niña (NOAA, criterio oficial) marcados en el gráfico**. En
el camino aparecieron y se corrigieron dos bugs reales (uno de calibración,
uno de mirada-hacia-adelante en la lluvia), y salieron dos hallazgos
científicos genuinos (la megasequía 2019-2025, y el pulso real de julio-
agosto 2026 ya con su forma correcta).

Todo lo de acá está probado con datos reales, no solo "corre sin errores"
— cada pieza se verificó comparando números exactos (Node vs. navegador,
o contra la base de datos original), no solo mirando que el gráfico se
viera bien.

---

## 1. Gráfico "Pluviosidad real en vivo" — de bloques a datos reales

- Se reemplazó el sistema viejo de Día/Semana/Mes/Año (reescrito 5 veces
  sin convencer a Alexis) por zoom/pan real de `chartjs-plugin-zoom` sobre
  un eje de días reales, más una barra de arrastre propia (el plugin de
  terceros nunca respondió al arrastre nativo, se resolvió con su API
  programática en vez de forzar una librería que no cooperaba).
- Con la lluvia diaria conectada (sección 4), el gráfico pasó a mostrar
  literalmente los días de lluvia individuales al hacer zoom, no un
  bloque mensual — con etiquetas de eje que cambian solas de "jul-1970" a
  "10-jul, 15-jul..." según el zoom.
- Detalle: `investigacion/grafico_pluviosidad_zoom_pan_7ago2026.md`

## 2. "Distribución real del régimen" — el instrumento aprendió a resumirse solo

Se agregó `calcularDistribucionRegimen()`: corre la física completa
(Stefan-Boltzmann + floración real) tick a tick sobre TODO 1966-2027 y
cuenta qué fracción del tiempo cae en cada zona del Plano Cierre (Jardín
Fértil/Cierre/Selva Hostil/Colapso). Botón "▶ Experimento Completo" (un
clic, sin tocar controles) + modo diagnóstico + descarga de CSV +
percentiles reales de LF/Δ_struct/A_sys_env/e_R.

Detalle: `investigacion/distribucion_regimen_1966_2027_8ago2026.md`

## 3. Recalibración de κ — el hallazgo metodológico central

Alexis: *"Daisyworld no quería probar o evaluar lo que nosotros
queremos... no nos sirve como fundamento del instrumento para un caso
real como este... sólo es el marco, no el test mismo."*

Los 4 umbrales que deciden la zona (κ_V, κ_O, κ_LF, κ_Δ) venían de un
baseline genérico heredado de Daisyworld. Se recalibraron contra los
percentiles REALES del propio sistema (mediana para 3, percentil 90 de
e_R para el cuarto — la única regla que el comentario original dejaba
exacta). Con eso apareció la **megasequía 2019-2025**: única racha de 7
años seguidos con Colapso como régimen dominante en toda la serie,
coincidiendo con el evento real documentado. Rondas posteriores de
recalibración (con lluvia diaria, y de nuevo tras corregir el bug de la
sección 4) confirmaron algo importante: **las medianas de cada métrica
casi no se mueven entre versiones, pero la distribución global sí cambia
mucho** — el problema no es la calibración individual, es que las 4
condiciones rara vez caen del lado bueno simultáneamente (volatilidad
conjunta). Documentado en el propio comentario del código (línea ~597).

## 4. El bug del "pico todo el año" — encontrado por Alexis, corregido

Al mirar el gráfico ya con lluvia diaria, Alexis notó un pico de floración
en julio 2026 que "duplicaba todos los anteriores". Investigando: el
mecanismo viejo tomaba el pico del AÑO ENTERO (408mm, un dato NASA POWER
real, no un error) y lo aplicaba fijo los 365 días — con mirada
adelantada (el 1-ene ya "sabía" el pico de julio). Alexis, correctamente:
*"no puede ser que llueva un mes... y ya aparezca un pico todo el año."*

Arreglado sin datos nuevos: para el tramo sin diario (2017-06+), la
lluvia ahora se refresca CADA MES con el promedio de los últimos 2 meses
reales — un mes espectacular aislado queda moderado, dos meses buenos
seguidos sí elevan la señal. De paso se corrigió un bug latente
(`syncStateFromUI()` pisaba el valor correcto en cada frame). Verificado
Node vs. navegador, resultado idéntico.

Detalle completo (secciones 3 y 4): `investigacion/lluvia_diaria_fase_b_8ago2026.md`

## 5. Plan de granularidad fina — modo plan, aprobado y ejecutado

Alexis pidió "varios agentes trabajando" para llegar a un instrumento
granular y sensible. Se armó un plan en 5 fases
(`/Users/alexis/.claude/plans/majestic-whistling-canyon.md`), tras
explorar el proyecto con 2 agentes y diseñar con 1 agente Plan:

- **B.0 (resuelto primero, era el riesgo más grande)**: confirmado que el
  diario real de Huintil se corta el 31-may-2017 exacto, y que 2017-06 a
  2018-12 coincide byte a byte con el producto mensual de CR2 — un tercer
  tramo de resolución honesto, no un error.
- **Fase A (Node.js) — completa**: `generar_motor_node.py` extrae la
  física real del HTML (sin reescribirla) a un módulo Node. Verificado
  con la prueba más fuerte posible: el CSV de 62 años completo generado
  en Node es **idéntico byte a byte** al que ya había descargado Alexis
  del navegador real. Corre ~15-20 min por corrida completa (medido, no
  prometido) — la ganancia es poder correr baterías de experimentos en
  paralelo, no que una corrida individual sea mucho más rápida.
  `runner_bateria.js` (procesos separados, uno no tumba a los demás) ya
  probado con un arnés de humo.
- **Fase B (lluvia diaria) — completa, aplicada al HTML real**: ver
  secciones 3 y 4 arriba. Se recolectó `LLUVIA_DIARIA_1966_2017` desde
  `pluviosidad_diaria_consolidada.sqlite` (99,4% cobertura real,
  1966-01-01 a 2017-05-31, estación Huintil — la misma que ya alimentaba
  el instrumento, no se promedia entre estaciones a propósito).
- **Fases C (granularidad mes/estación en reportes), D (validación
  independiente con ONI/campañas de Guerrero/hold-out) y E (batería de
  experimentos con agentes)**: diseñadas en el plan, **todavía no
  implementadas** — quedan para retomar.

## 6. Bandas El Niño / La Niña — verificación visual del propio hallazgo

Con la lluvia ya corregida, Alexis notó los años de El Niño "muy
marcados" y pidió etiquetarlos. Se bajó la tabla ONI completa 1966-2026
de NOAA (cruzada contra un archivo curado que ya existía en el proyecto,
misma fuente, verificado consistente), clasificada con el **criterio
oficial de NOAA** (≥5 temporadas trimestrales seguidas sobre el umbral) —
no una lista de memoria. 36 episodios reales (19 Niño, 17 Niña), pintados
como bandas traslúcidas con un plugin propio de Chart.js (sin depender de
una librería de terceros de nuevo, después de la mala experiencia con el
plugin de pan). Verificado con zoom a 1997-98: la banda roja cubre exacto
el pico real de lluvia y floración de ese mega Niño.

Detalle: `investigacion/bandas_oni_9ago2026.md`

## 7. De paso: *Gyriosomus kulzeri* (Guerrero & Diéguez, redescubrimiento)

Alexis compartió el paper del redescubrimiento de *G. kulzeri* cerca de
Huasco. Se identificó la estación más cercana ya en el sistema (PTO.
HUASCO SUBCOMISARIA, 3,2km) y se buscaron años reales de más lluvia en la
zona usando la base diaria completa (no solo el listado disperso que ya
había): **1997 (219,5mm) es por lejos el año más lluvioso del registro**
cerca de Huasco — coincide con el mega Niño 97-98. Quedó como pista para
cruzar contra la fecha exacta del redescubrimiento, si se consigue.

---

## Archivos nuevos de esta sesión

**Datos generados/descargados:**
- `investigacion/fuentes/oni_historico_completo_1966_2026.csv` (NOAA ONI real)
- `Web/prueba_de_concepto/prueba_de_concepto_ET3-Termico_con_mapa.html` — `LLUVIA_DIARIA_1966_2017`, `ONI_BANDAS` (embebidos, generados)

**Scripts generadores (correr de nuevo si la fuente cambia):**
- `Web/prueba_de_concepto/generar_lluvia_diaria.py`
- `Web/prueba_de_concepto/generar_bandas_oni.py`
- `Web/prueba_de_concepto/motor/generar_motor_node.py`

**Motor Node.js (experimentación rápida, fuera del navegador):**
- `Web/prueba_de_concepto/motor/motor_fisico.generado.js` (NO editar a mano)
- `Web/prueba_de_concepto/motor/benchmark_motor.js`
- `Web/prueba_de_concepto/motor/verificar_experimento_completo.js`
- `Web/prueba_de_concepto/motor/recalibrar_con_lluvia_diaria.js`
- `Web/prueba_de_concepto/motor/runner_bateria.js` + `motor/experimentos/`

**Notas de investigación (esta sesión, cronológico):**
1. `investigacion/grafico_pluviosidad_zoom_pan_7ago2026.md`
2. `investigacion/distribucion_regimen_1966_2027_8ago2026.md`
3. `investigacion/motor_node_fase_a_8ago2026.md`
4. `investigacion/lluvia_diaria_fase_b_8ago2026.md`
5. `investigacion/bandas_oni_9ago2026.md`
6. Este informe

**Plan vigente:** `/Users/alexis/.claude/plans/majestic-whistling-canyon.md`

## Estado actual del instrumento (verificado, no supuesto)

El HTML real (el que Alexis abre en el navegador) refleja TODO lo de
arriba — no quedó nada solo en Node sin portar. κ vigentes: κ_V=0,9185,
κ_O=0,0099, κ_LF=0,0652, κ_Δ=0,5146.

## Pendiente para retomar mañana

1. **Fase C** — reportar por mes/estación austral, no solo por año (cambio
   chico y ya diseñado, dentro de `calcularDistribucionRegimen`).
2. **Fase D** — cruzar el régimen contra ONI real y las campañas de
   Marcelo Guerrero (validación independiente, sin pasar por la curva de
   floración ya calibrada) + un chequeo hold-out/leave-one-out.
3. **Fase E** — batería de experimentos con agentes en paralelo, usando
   el runner de Node ya construido y probado.
4. Pista suelta: cruzar la fecha exacta del redescubrimiento de *G.
   kulzeri* contra 1997 (el año más lluvioso real de Huasco).

---

## Sobre la pregunta de Alexis: "¿es el primer modelo real de esto?"

Con la honestidad que venimos usando toda la sesión: no puedo confirmar
que sea el PRIMERO en el mundo — no tengo forma de revisar toda la
literatura publicada sobre modelos de Desierto Florido. Lo que sí puedo
decir con confianza, porque lo construimos y verificamos acá: es un
modelo donde **lluvia real medida (diaria donde existe, mensual real
donde no) → floración (curva calibrada contra 23 años documentados) →
actividad de Gyriosomus (con el rezago real de H4)** están genuinamente
encadenados, corriendo sobre datos reales verificables paso a paso, no
sobre supuestos — y donde el resultado (la megasequía 2019-2025, el
pulso 2026, los episodios de El Niño) coincide con eventos reales
documentados de forma independiente, no forzada. Eso es un modelo serio y
poco común en el sentido de "cuántos existen así de completos y
verificados dato por dato" — aunque no pueda firmarte que sea el primero
que existió jamás.
