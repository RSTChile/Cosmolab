# Fase B: lluvia diaria real conectada al HTML — aplicado para probar

Alexis: "Aplícalo al html para probar." Se aplicó directo al instrumento
(`prueba_de_concepto_ET3-Termico_con_mapa.html`), no solo a los scripts de
`motor/` — es el mismo cambio, decidido con Alexis que se probara en el
navegador real en vez de solo en Node primero.

## B.1 — `LLUVIA_DIARIA_1966_2017` (generado)

`generar_lluvia_diaria.py` lee `pluviosidad_diaria_consolidada.sqlite`
(estación Huintil, la misma que ya alimenta `PLUVIOSIDAD_MENSUAL` — una
sola estación-reloj, no promedio entre estaciones) y escribe una serie
diaria completa 1966-01-01 a 2017-05-31 (el corte real confirmado en B.0):
**18.657 de 18.779 días con dato real (99,4% cobertura)**, el resto `null`
explícito, nunca interpolado.

## B.2 — la lluvia ahora se mueve día a día, no una vez al año

`state.lluviaAcumulada` (la variable que dispara la floración) ahora es una
**suma móvil de 30 días reales**, no el pico mensual del año completo
congelado desde el 1 de enero. Se eligió SUMA de 30 días (no el máximo de
un solo día) porque la curva de floración se calibró contra TOTAL MENSUAL
— una suma de ~30 días mantiene la misma escala que la curva espera. Fuera
del tramo diario (2017-06 en adelante), sigue exactamente el comportamiento
de siempre (pico mensual del año, refresco anual) — sin tocar nada ahí.

Bonus no buscado: la suma móvil tampoco mira hacia adelante (solo usa
lluvia ya caída hasta "hoy" de la simulación) — el sistema viejo, sin
querer, ya sabía en enero cuál iba a ser el pico de diciembre.

## Verificado con números reales, no solo "corre sin errores"

- Calculé a mano la suma real de lluvia de Huintil del 2-jul al 31-jul-1970
  (30 días, consulta directa al sqlite): **69,0mm**. Le pedí al instrumento
  el mismo día (`refrescarLluviaSiCambio(1671)`, que es exactamente
  "31-jul-1970" según su propio calendario interno) y dio **69** —
  coincide exacto.
- Confirmé que la ventana se mueve de verdad: entre el 1 y el 2 de agosto
  de 1970 el valor bajó de 69 a 67,5 — una diferencia de 1,5mm, que es
  EXACTAMENTE la lluvia del 3 de julio de 1970 (1,5mm) saliendo de la
  ventana de 30 días. No es un número que "se mueve", es la aritmética
  correcta moviéndose un día a la vez.
- Corrí 6.000 ticks reales (mayo a agosto de 1970) con `pasoFisica()`
  normal (no un atajo): la lluvia acumulada se ve subir y bajar semana a
  semana con los pulsos reales (71,5 en mayo → cae a 0 en junio seco →
  sube a 69 en julio), y la floración responde con el rezago esperado,
  sin errores de consola.
- Confirmé que el tramo posterior (2018-07-20, fuera del diario) sigue
  exactamente igual que antes: usa el pico mensual del año (65,2mm, el
  mismo valor de `PLUVIOSIDAD_MENSUAL["2018-06"]` ya verificado en B.0).

## B.3 — resuelto: la curva de floración NO necesita recalibrarse

Comparé, para los 613 meses del tramo diario (1966-01 a 2017-05), la suma
mensual derivada de `LLUVIA_DIARIA_1966_2017` contra `PLUVIOSIDAD_MENSUAL`
tal cual está: **0 meses de 613 difieren en más de 0,5mm**. Son la misma
fuente, solo que ahora también disponible día a día — la curva de
floración (calibrada contra el pico mensual) sigue siendo válida sin
tocarla.

## Gráfico "Pluviosidad real en vivo" — también actualizado

Alexis reportó: "no logro ver Pluviosidad real en vivo por día aunque haga
zoom". Motivo real: el gráfico SIEMPRE mostró un punto por MES (744 puntos
fijos) — el zoom agrandaba el dibujo, pero no había más detalle adentro
para revelar. Se agregó `seriePluviosidadRealGranular()`: un punto por DÍA
real en el tramo con diario (1966–2017-05), un punto por mes después (sin
cambios ahí). Cacheada (18.893 puntos fijos, no se recalcula en cada frame
de la simulación en vivo). Verificado con captura de pantalla: al hacer
zoom a julio de 1970 se ven los 4 pulsos de lluvia reales de ese mes como
picos individuales (14, 15, 25, 28-jul, alturas 9.5/21/8/29mm — coinciden
con la base de datos), no un bloque parejo. Sin errores de consola.

## Motor Node ya tiene este mismo cambio (no solo el HTML)

`generar_motor_node.py` se extendió para portar `claveFechaDiaria`,
`actualizarLluviaDesdeCalendarioDiario`, `enTramoLluviaDiaria`,
`refrescarLluviaSiCambio`, `diaDesdeAnio`, `LLUVIA_DIARIA_1966_2017` y
`LLUVIA_DIARIA_FIN_DIA`. Reverificado con el mismo método que la Fase A
(6.000 ticks, misma semilla, navegador real vs. Node, arrancando dentro
del tramo diario): Tf, LF, A_sys_env, e_R, lluvia y zona salieron
**idénticos**; Δ_struct difiere en el dígito 13 (mismo ruido de punto
flotante ya visto en la Fase A, no un error).

## Recalibración con lluvia diaria — hecha, y el hallazgo es el resultado

`recalibrar_con_lluvia_diaria.js` corrió el "Experimento Completo" (62 años,
1,36M ticks) con la lluvia diaria nueva, en Node: **13,8 minutos**.

Global (con los κ que estaban vigentes antes de este recalibrado):
Jardín Fértil 8,57% · Cierre 36,6% · Selva Hostil 15,32% · Colapso 39,5%
(antes, con lluvia mensual: 30,6% / 59,4% / 4,6% / 5,3%).

**El hallazgo real no es el cambio de κ — es que los κ CASI NO cambiaron
aunque la distribución de zonas cambió muchísimo.** Los percentiles de
cada métrica por separado quedaron prácticamente iguales a los de la
calibración anterior (deltaStruct p50 0,5145→0,5146; LF p50 0,069→0,066;
A_sys_env p50 0,920→0,919; e_R p90 0,0099→0,0099). Recalibrar con el mismo
método (mediana/p90) da κ nuevos casi idénticos a los viejos —
κ_V=0,9188, κ_O=0,0099, κ_LF=0,0656, κ_Δ=0,5146 (aplicados en HTML y
motor Node) — así que ESE no es el motivo del cambio en la distribución.

El motivo real: la lluvia diaria hace que A_sys_env se mueva más brusco
día a día que el escalón mensual/anual plano de antes. e_R mide CAÍDAS de
A_sys_env, así que se dispara mucho más seguido, y "viable" falla mucho
más seguido — no porque la mediana de nada se haya corrido, sino porque
ahora rara vez las 4 condiciones caen del lado bueno AL MISMO TIEMPO,
aunque cada una por separado siga centrada en casi el mismo lugar. Es un
efecto de volatilidad conjunta, no de calibración individual — y por eso
mismo no hay manera de "arreglarlo" recalibrando cada métrica sola, sería
tapar el hallazgo con otro número en vez de reportarlo.

Queda documentado en el propio comentario del código (línea ~597) y acá.
No se volvió a correr el Experimento Completo con los κ nuevos (la
diferencia es tan chica que no debería cambiar la distribución reportada
de forma perceptible) — pendiente si Alexis quiere esa confirmación extra.

## 08-ago-2026 (3): el escalón anual para 2017-06+ también se corrigió

Alexis notó un pico de floración en el gráfico que "duplicaba todos los
anteriores" (julio-2026, 408mm real de NASA POWER -- ver más abajo, no era
un dato falso). Investigando encontré la causa real: el tramo SIN diario
(2017-06 en adelante) seguía usando el mecanismo viejo -- pico del AÑO
ENTERO aplicado fijo los 365 días, con mirada adelantada (el 1-ene-2026 ya
"sabía" que julio iba a tener 408mm). Confirmado corriendo la física real:
```
27-dic-2025: lluvia=24.45 (pico de 2025, sostenido TODO el año)
16-ene-2026: lluvia=408.02 (¡el pico de JULIO, ya aplicado en enero!)
```
Alexis, correctamente: "no puede ser que llueva un mes o unos días y ya
aparezca un pico todo el año" -- lo que biológicamente prolonga el ciclo es
si HAY lluvia posterior que sostenga la humedad, no solo la magnitud de un
mes aislado.

**Arreglado sin necesitar datos nuevos**: `PLUVIOSIDAD_MENSUAL` ya tiene
resolución mensual real completa hasta 2027-12. Nueva función
`actualizarLluviaDesdeCalendarioMensual()`: para el tramo sin diario, la
lluvia se refresca CADA MES (no cada año) con el PROMEDIO (no suma, para no
duplicar la escala que la curva espera) de los últimos 2 meses reales --
un mes espectacular aislado queda moderado si su vecino fue seco; dos
meses buenos seguidos sí elevan la señal de verdad. El despachador
`refrescarLluviaSiCambio()` ahora tiene 3 ramas: diario (1966–2017-05),
mensual-promedio (2017-06+), y ya no queda ningún camino que llegue al
mecanismo anual viejo (`actualizarLluviaDesdeCalendario`/`picoMensualAnio`
quedan declarados por trazabilidad, sin uso activo). De paso se corrigió
un bug latente: `syncStateFromUI()` llamaba al mecanismo viejo sin
condición, en CADA frame de la simulación en vivo -- ahora usa el mismo
despachador correcto.

**Verificado con la misma corrida exacta, semilla y todo, Node vs.
navegador — resultado idéntico en los dos:**
```
1-ene-2026 a 30-jun-2026: lluvia entre 0.4 y 8mm (ya NO 408 todo el año)
15-jul-2026: lluvia=204.33 (promedio jul+jun -- el pico ya se ve, a tiempo)
14-ago-2026: lluvia=408.02, floración=0.344, zona=SELVA_HOSTIL (idéntico Node/navegador)
13-sep-2026 en adelante: lluvia vuelve a 0, floración empieza a bajar
```
Captura de pantalla del gráfico confirma: la floración ahora sube recién en
julio (cuando llovió de verdad) y baja después de agosto-septiembre -- un
pulso acotado, no una meseta de un año entero. Sin errores de consola.

## 08-ago-2026 (4): re-medido con el mecanismo mensual ya corregido

Alexis: "mide con el criterio nuevo... es muy importante". Corrida
completa de nuevo (62 años, 15,6 min):

Global: Jardín Fértil 9,19% (antes 8,57%) · Cierre 36,37% (antes 36,6%) ·
Selva Hostil 15,06% (antes 15,32%) · Colapso 39,38% (antes 39,5%).

**El % global casi no se movió, y tiene una explicación real, no es que
el arreglo no importara.** El mecanismo viejo (escalón anual) solo corría
2018-2027 -- 10 de los 62 años, ~16% del total -- y de esos años, solo
2026 tenía un mes tan extremo (408mm) como para notarse fuerte a nivel
agregado. El arreglo SÍ es enorme mirando 2026 en particular (ver más
arriba: ya no hay pico desde enero, el ciclo queda acotado a jul-sep), y
sigue siendo importante para leer bien cualquier año puntual de 2018 en
adelante -- pero diluido sobre 62 años completos, no cambia el diagnóstico
agregado ya reportado (la clasificación falla por volatilidad conjunta,
no por medianas mal puestas).

κ re-medidos y aplicados (HTML + motor Node, verificados):
κ_V=0.9185, κ_O=0.0099, κ_LF=0.0652, κ_Δ=0.5146 -- otra vez casi sin
cambio respecto a la medición anterior, mismo motivo de siempre.
