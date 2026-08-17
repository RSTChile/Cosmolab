# Distribución real del régimen (1966–2027) — nueva capacidad del instrumento

Alexis, a partir de "¿qué podemos deducir de la simulación tal cual está?":
señalé que el instrumento podía calcular qué fracción del período real
1966–2027 cae en cada zona del Plano Cierre (Jardín Fértil/Cierre/Selva
Hostil/Colapso), pero que hoy no lo mostraba. Dijiste "esa es una hipótesis
interesante, hazlo" — esto es lo que se agregó.

## Qué hace
Nueva sección "Distribución real del régimen (1966–2027)" debajo del
cuadrante Plano Cierre, en `prueba_de_concepto_ET3-Termico_con_mapa.html`.
Un botón corre la **física completa** (Stefan-Boltzmann + floración real,
con la grilla térmica 64×64 — NO el atajo liviano que usa la curva de
Floración) tick a tick sobre TODO el período real, con los parámetros que
estén puestos en los sliders en ese momento, y cuenta en qué zona cae cada
tick. Al terminar muestra:
- % global del período en cada una de las 4 zonas (barras).
- Tabla desplegable por año (1966–2027): régimen dominante y % Jardín
  Fértil de cada año real, exportable a CSV.
- Qué parámetros se usaron (PowerBase, β, ruido, tOpt, Día/Noche,
  Estaciones), para que el resultado sea interpretable y reproducible.

## Por qué tarda tanto y cómo se resolvió sin arriesgar nada de lo que ya andaba
La grilla térmica es la parte cara (evolveField, ~4096 celdas por tick,
9 vecinos cada una) — es la misma que ya hacía lenta la física completa en
el resto del instrumento (`saltarAFecha()`, la barra vieja que se sacó,
usaba un atajo justamente para evitar esto). Acá SÍ hace falta la física
real completa, porque Δ_struct (uno de los 2 ejes del Plano Cierre) sale de
esa grilla — no hay atajo honesto posible.

Medido en esta sesión (no un número de memoria): ~900 ticks/seg en bruto, lo
que da un cálculo completo (1966–2027 = 1,36 millones de ticks) de entre
~25 minutos y bastante más según la máquina y el navegador — el tiempo real
lo mide y lo muestra el propio instrumento en pantalla (ticks/s y minutos
restantes en vivo), no es un número prometido de antemano.

Para que ese rato largo no se coma nada de lo que ya está en pantalla:
- Se guarda una foto completa de TODO lo que la física toca (estado, campo
  térmico, buffers de entropía/saturación, semillas de azar, historial
  diario) antes de empezar, y se restaura exacta al terminar o al cancelar
  — verificado con el propio simulador corriendo en vivo: cancelé a mitad
  de camino y el estado quedó bit a bit igual al de antes de apretar el
  botón.
- El cálculo usa su propia semilla de azar fija y separada (no toca ni
  consume el azar de la sesión en vivo — importa para este instrumento,
  que declara "azar aislado por fuente y por parada").
- Botón Cancelar en cualquier momento.
- Se agregó un parámetro opcional a `pasoFisica(registrar)` (default
  `true`, comportamiento idéntico a como estaba) para que este cálculo
  pueda saltarse el registro de historial tick a tick (`pushHistory()`) —
  sin este cambio, `state.history.shift()` corriendo 1,36 millones de veces
  habría sido catastróficamente más lento (shift es O(n) sobre un array).
  La física en sí no cambió en absoluto, solo si se loguea cada tick.
- 60 días de asentamiento térmico antes de empezar a contar (con la lluvia
  real de ene-mar 1966), para que Tf y el campo no arranquen fríos del
  valor inicial arbitrario.

## Verificado, con la salvedad de qué NO se pudo verificar en esta sesión
Sí verificado en el navegador: la sección se ve bien, las barras y la tabla
por año renderizan correcto con datos de prueba, la descarga CSV funciona,
el snapshot/restauración es exacto (comparado antes/después de cancelar a
mitad de camino), no hay errores de consola, y el conteo de progreso avanza
de verdad (tick a tick, medido en vivo).

NO verificado end-to-end en esta sesión: dejar correr el cálculo completo
hasta el final (tardaría 25-60+ minutos reales). La lógica que corre en el
tick 1 es exactamente la misma que corre en el tick 1.360.000 — no hay
motivo para esperar que falle solo por tardar más.

## 08-ago-2026: Alexis SÍ corrió el cálculo completo — resultado real
Con Día/Noche y Estaciones **encendidos** (los defaults): Jardín Fértil
30.6% · Cierre 59.4% · Selva Hostil 4.6% · Colapso 5.3%. Pero la tabla por
año salió casi plana — 30-31% Jardín Fértil TODOS los años, sin distinguir
El Niño fuerte (1982-83, 1997-98, 2015-16) de años secos. Hallazgo honesto:
con el ciclo diario/estacional encendido, ese vaivén (6°C por día, más el
estacional) es tan grande que tapa la señal de la lluvia real año a año —
la floración sigue respondiendo a la lluvia real (eso no cambió), pero no
alcanza a mover la aguja de qué zona del Plano Cierre domina el año.

## 08-ago-2026 (2): "Experimento Completo" — un clic, sin tocar controles
Alexis: "no tener que poner controles o apagar cosas a mano, porque
cometeré errores". Se agregó un botón nuevo, destacado arriba del avanzado
existente (que ahora vive colapsado en un `<details>` "Avanzado"):

- **Experimento Completo**: corre con Día/Noche y Estaciones apagados (para
  aislar la lluvia real como único forzante, la hipótesis que salió del
  resultado de arriba) y con TODOS los demás parámetros en su valor de
  fábrica (`PARAMETROS_FABRICA`: PowerBase 0.47, β 0.94, ruido 0.0079,
  tOpt 25°C, PTC 16°C/1.0, Luminosidad 0.94, umbral 15mm, rezago 30 días) —
  no depende de qué haya en los sliders en ese momento. Al terminar,
  descarga el CSV solo (`regimen_1966_2027_solo_lluvia_real.csv`).
- El botón viejo sigue ahí, ahora como "Avanzado: correr con los parámetros
  que tengo puestos ahora arriba" — usa lo que esté en los sliders, sin
  auto-descargar (`regimen_1966_2027_parametros_actuales.csv`).

`calcularDistribucionRegimen(overrides, sufijoArchivo, autoDescargar)` pasó
a aceptar estos 3 parámetros opcionales; sin ellos se comporta exactamente
como antes (usada por el botón "Avanzado"). Verificado en el navegador:
- Los overrides pisan los sliders en vivo (probé poniendo powerBase=0.9 y
  Día/Noche=on ANTES de apretar Experimento Completo — durante el cálculo
  el estado quedó en 0.47/off, como debía).
- La restauración al cancelar volvió exacto a los valores de ANTES del
  experimento (0.9/on), no a los de fábrica — confirma que los overrides
  son solo temporales para el cálculo, no contaminan la sesión en vivo.
- La descarga automática dispara sola al terminar (probado interceptando
  `HTMLAnchorElement.prototype.click`), con el nombre de archivo correcto;
  el botón "Avanzado" NO descarga solo (queda al criterio del usuario, como
  antes).
- Sin errores de consola.

Pendiente (no verificado en esta sesión, mismo motivo que arriba: 25-60 min
reales): correr "Experimento Completo" de punta a punta y ver si, aislada
la lluvia, SÍ aparece una diferencia real entre años El Niño y años secos —
esa es la pregunta que motivó el botón.

## 08-ago-2026 (3): Alexis corrió el diagnóstico aislado — saturó en 100%
El CSV de "solo lluvia" (Día/Noche apagado) dio **100.00% Jardín Fértil en
los 62 años, sin ninguna excepción**. Investigado con una corrida corta
instrumentada (no un supuesto): confirmado que NO es un bug — con Día/Noche
apagado el sistema se asienta en ~5 días simulados y los 4 números que
deciden la zona (LF≈0.06, Δ_struct≈0.8, A_sys_env≈0.95-0.99, e_R≈0) quedan
con margen de sobra del lado "activo/viable", y la lluvia real los mueve
de a poco por dentro pero nunca lo bastante como para cruzar ninguna de las
4 líneas. Con Día/Noche prendido pasaba lo simétrico: el vaivén diario
(6°C/día) es tan grande que tapa la señal de la lluvia (resultado plano
~30% de la vez anterior). Las dos pruebas juntas muestran lo mismo desde
lados opuestos: **la clasificación de 4 zonas, con los κ actuales, no
distingue años El Niño de años secos** — no porque la lluvia no importe
(sí mueve los números), sino porque esos κ no fueron calibrados contra
este caso real.

## 08-ago-2026 (4): recalibrar contra el caso real, no contra Daisyworld
Alexis: "Daisyworld no quería probar o evaluar lo que nosotros queremos...
no nos sirve como fundamento del instrumento para un caso real como
este... sólo es el marco, no el test mismo." Los κ actuales (`KAPPA_V=0.75,
KAPPA_O=0.01, KAPPA_LF=0.05, KAPPA_DELTA=0.3`, línea ~579) vienen de un
baseline genérico ("7200 pasos con defaults reales", sin lluvia real de por
medio) heredado de la estructura Daisyworld — no del caso real Gyriosomus/
Desierto Florido con lluvia 1966-2027. Para recalibrar hace falta la
distribución REAL de LF/Δ_struct/A_sys_env/e_R bajo el caso real, no
inventarla.

**Qué se agregó**: `calcularDistribucionRegimen()` ahora también guarda
TODOS los valores tick a tick de esas 4 métricas (Float64Array × 4, ~43MB,
sin costo real de memoria ni de tiempo — el sort al final es O(n log n),
~1seg) y calcula percentiles reales (p10/p25/p50/p75/p90/p95/max) al
terminar. Se muestran en un nuevo desplegable "Percentiles reales... (para
calibrar κ)" en el panel de resultado, junto a los κ actuales y UN cálculo
directo: κ_O ≈ p90 de e_R, porque es la ÚNICA regla que el comentario
original de los κ deja exacta (κ_O=0.01 vs "e_R p90=0.010" — coincide
justo). Para κ_V/κ_LF/κ_Δ el comentario original no deja una fórmula tan
precisa, así que se muestran los percentiles crudos en vez de inventar una
regla — la decisión de dónde poner esas 3 líneas es de Alexis, no mía.
**No se tocó ningún valor de κ en el código** — son solo datos para decidir
juntos el próximo paso.

**Se reordenó la sección** para que el botón principal ("Experimento
Completo", verde) ahora corra el CASO REAL (Día/Noche y Estaciones
prendidos + parámetros de fábrica) en vez del aislado — es el que hace
falta para calibrar contra la realidad. El aislado (Día/Noche apagado) se
movió a "Avanzado" como "Diagnóstico: solo lluvia", ya cumplió su función
(mostrar el mecanismo de saturación) pero no sirve para calibrar.

**Verificado en el navegador** (sin correr los 62 años completos, mismo
motivo de siempre): el panel de percentiles renderiza bien con datos de
prueba; los dos botones (Experimento Completo / Diagnóstico) aplican
Día/Noche true/false correctamente; la restauración de estado sigue
perfecta. Además, corrí una prueba corta REAL (5000 ticks, ~83 días, Día/
Noche prendido, no fabricada) para validar que la recolección de
percentiles funciona con física real, no solo con datos falsos — y salió
algo alentador: `Δ_struct p50=0.524` y `e_R p90=0.00982`, casi idénticos a
los valores del comentario original de calibración (`deltaStruct
p50=0.53`, `e_R p90=0.010`) — buena señal de que el código está midiendo
lo mismo que se midió la primera vez, solo que ahora contra 62 años de
lluvia real en vez de contra un baseline genérico. `LF p50` salió bastante
más bajo (0.011 vs 0.081 del original) pero son solo 83 días, no alcanza
a explorar el rango completo — hace falta la corrida completa para un
número confiable.

**Pendiente**: correr "Experimento Completo" completo (25-60+ min reales)
para tener los percentiles verdaderos de los 62 años, y con esos números
decidir juntos los nuevos κ.

## 08-ago-2026 (5): κ recalibrados y aplicados
Alexis corrió "Experimento Completo" completo y pasó los percentiles
reales de los 62 años:
- A_sys_env: p10=0.818, p50=0.920, p90=0.986
- LF: p10=0.002, p50=0.069, p90=0.144
- Δ_struct: p10=0.042, p50=0.515, p90=0.992
- e_R: p50=0, p90=0.0099, max=0.544

Diagnóstico confirmado con estos números: κ_V=0.75 quedaba por DEBAJO del
p10 real (0.818) — A_sys_env nunca reprobaba. κ_LF=0.05 y κ_Δ=0.3 caían
cerca del p35-38 real — dejaban pasar la mayoría del tiempo. κ_O=0.01 ya
estaba prácticamente exacto contra el p90 real (0.0099) — es la única
línea que venía bien puesta, coincidiendo con la única regla que el
comentario de calibración original dejaba exacta.

De paso, algo real y nuevo en la tabla por año de esa corrida (con los κ
VIEJOS todavía): **2019-2025 se separan del resto** — Colapso sube de ~5%
(el resto del período) a 6-8%, máximo en 2024 (8.26%, el más alto de los
62 años) — coincide con la megasequía real de Chile central-norte
2019-2024. Primera señal real detectada, aún sin recalibrar.

**Aplicado** (línea ~579, `clasificarCierre()`): κ_V=0.75→**0.92**,
κ_O=0.01→**0.0099**, κ_LF=0.05→**0.069**, κ_Δ=0.3→**0.51**. Método: los 3
primeros = mediana real (p50) de cada métrica propia (el punto donde el
sistema real pasa la mitad del tiempo arriba y la mitad abajo); κ_O se
mantuvo en la regla p90 de e_R (mediana de e_R es 0, inservible como
umbral — la fórmula se rompería). Los valores viejos quedaron documentados
en el comentario del código por trazabilidad, no se usan más.

Verificado en el navegador tras el cambio: `KAPPA_V/O/LF/DELTA` cargan los
valores nuevos, sin errores de consola, y una corrida corta real (100
días, ~6000 ticks) con los κ nuevos da una clasificación NO degenerada
(aparecen las 4 zonas, no se satura en una sola) — confirma que el cambio
no rompió nada. Esa muestra corta dio 60.8% Colapso, pero es solo
ene-abr 1966 (una ventana estacional chica, no representativa por sí
sola) — el veredicto real de cómo queda repartido el período completo
1966-2027 con los κ nuevos necesita correr "Experimento Completo" de
nuevo, completo (25-60+ min reales, mismo límite de siempre en esta
sesión).

**Pendiente**: Alexis corre "Experimento Completo" una vez más con los κ
nuevos ya aplicados, y vemos si ahora sí aparece una diferencia real y
visible entre años (especialmente si la señal de megasequía 2019-2024 se
hace más clara, o si emergen otros años/períodos reales).
