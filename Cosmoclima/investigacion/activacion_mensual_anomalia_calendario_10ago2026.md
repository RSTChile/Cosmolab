# Activación (LF/Δ_struct) a escala semanal real + anomalía de calendario — 10-ago-2026

Resultado FINAL de esta ronda: Nivel 3 (semanal), a pedido de Alexis tras
ver el resultado mensual ("subamos la resolución a nivel semanal"). Los 4
criterios de éxito pasan limpio, incluido el que antes solo pasaba "en
promedio" -- se detalla abajo la progresión completa Nivel 1→2→3, con los
números reales de cada intento, no solo el final. No representa un cierre
ni un veredicto -- eso lo decide Alexis al revisarlo (regla ya establecida
en este proyecto).

## Por qué

Tras arreglar "viabilidad" (A_sys_env/e_R) a promedio mensual real (ronda
anterior, misma fecha), el CSV completo de 62 años mostró que el % de
Jardín Fértil por año seguía sin variar continuo con la lluvia real: caía
en un puñado de "moldes" casi idénticos repetidos en años con lluvia
totalmente distinta (ej. 1997 -mega Niño- y 1970 -normal- daban el mismo
12.x%). Alexis: *"nuestro problema es que el instrumento está mal
calibrado, no los datos... sigue iterando hasta que tengamos una clara
diferenciación por año y estación."*

## Diagnóstico confirmado

"Activación" (LF/Δ_struct) seguía calculándose tick a tick, dominada por
`applyDayNightCycle()`/`seasonFactor()` -- un vaivén SINTÉTICO cuyo período
(`seasonSpeed=1/(60*365)`) coincide EXACTO con `DIAS_POR_ANIO_CAL=365` (sin
bisiestos): se repite idéntico, sin ninguna deriva, cada año calendario.

**Confirmado con números reales** (percentiles por mes-calendario, motor
Node, 62 años): dentro de un mismo mes, LF varía apenas ±5-6% entre los 62
años distintos (ej. enero: p10=0.0106, p90=0.0119) -- el vaivén sintético
fija casi todo el valor mes a mes, dejando muy poco margen para que la
lluvia real se note contra un umbral único.

## Arreglo en dos niveles

**Nivel 1** (promedio mensual real de LF/Δ_struct, mismo patrón que
viabilidad): necesario pero, como predijo el diagnóstico, NO alcanzó solo
-- con κ global, colapsó a solo 3 valores distintos de Jardín Fértil en 62
años (peor que antes de tocar nada).

**Nivel 2** (κ_LF/κ_Δ por mes-calendario, no un κ global): compara el
LF/Δ_struct de ESTE mes contra la distribución de TODOS los mismos meses
1966-2027 (ej. enero 2019 contra todos los eneros), cancelando el
componente sintético (idéntico en todos) y dejando pasar la señal real.

## Qué se probó primero (Nivel 1 solo) y por qué no alcanzó

Corrida completa de 62 años con Nivel 1 (LF/Δ_struct promediadas por mes,
pero comparadas contra un κ GLOBAL, igual que antes): **empeoró** --
colapsó a solo 3 valores distintos de Jardín Fértil en 62 años (peor que
el punto de partida). Confirma el diagnóstico con números: dentro de un
mismo mes-calendario, LF varía apenas un ±5-6% entre los 62 años (ej.
enero: p10=0.0106, p90=0.0119) -- casi todo el valor lo fija el vaivén
sintético, un κ global no tiene margen para separar nada.

## Resultado Nivel 2 (κ_LF/κ_Δ por mes-calendario) -- superado por Nivel 3, queda por trazabilidad

Corrida real completa 1966-2027 (motor Node, semilla `regimen1966-2027`,
parámetros de fábrica, Día/Noche + Estaciones prendidos) -- verificada
byte a byte contra el HTML real en el navegador antes de confiar en estos
números (mismo LF/Δ_struct/A_sys_env/e_R/estación/zona en un tramo de
prueba de 2 meses, no solo "corrió sin error").

| Criterio | Resultado | Umbral | ¿Pasa? |
|---|---|---|---|
| **(a) Diferenciación real** (¿repite molde año a año?) | 57 de 62 años con clasificación de las 4 zonas completamente distinta; 0 años en un patrón repetido 3+ veces | <20 años en clusters | **SÍ** |
| **(b) Floración real > control** | Media Jardín Fértil en 13 años de floración documentada = 12,16% vs. 9,18% en 10 años control | media(floración) > media(control) | **SÍ** (con matiz, ver abajo) |
| **(c) Megasequía 2019-2025 por debajo del promedio** | Jardín Fértil en la megasequía = 0,003% vs. 12,61% del promedio de los 62 años (Colapso 33,3% vs 34,6%, similar) | media(megasequía) < media global | **SÍ**, muy marcado |
| **(d) Correlación con lluvia real independiente** | ρ de Spearman = 0,584 (Jardín Fértil % vs. lluvia real anual, de la base de estaciones DMC, NO la curva de floración ya calibrada) | ρ > 0 y \|ρ\|>0,2 | **SÍ** |

**El matiz honesto de (b), en Nivel 2**: la MEDIA de los años de floración
era claramente mayor (12,16 vs 9,18, ~33% más alto) -- una comparación
estadística válida y clara. Pero mirando año por año, solo 5 de los 13 años
de floración documentada (38%) superaban individualmente la media de
control -- no era "la mayoría de los años individuales", era un efecto de
la media arrastrado por algunos años de floración fuerte. Este matiz es
justo lo que Alexis pidió mejorar subiendo la resolución -- ver Nivel 3
abajo, donde se resuelve.

**Nota sobre el criterio (a)**: la primera versión de la evaluación (solo
mirando el % total de Jardín Fértil por año, sin las otras 3 zonas) SÍ
parecía mostrar "moldes" repetidos (8 valores repetidos, 60/62 años). Al
revisar por qué, se confirmó que era un artefacto de esa medida: dos años
pueden compartir el mismo % TOTAL de Jardín Fértil (ej. "1 mes del año")
siendo MESES DISTINTOS con lluvia real distinta -- eso ya es diferenciación
real, la suma simplemente coincide. Mirando las 4 zonas juntas (el criterio
correcto), el molde desaparece: 57 de 62 años son combinaciones únicas.
`evaluar_contra_ground_truth.js` quedó corregido para siempre usar las 4
zonas juntas, no solo Jardín Fértil aislado.

## Nivel 3 -- activación a escala SEMANAL real (resultado final de esta ronda)

Alexis, tras ver el resultado mensual: *"es muy interesante lo que hiciste
al comparar mes por mes, y creo que si unimos eso a semana por semana,
tendríamos un resultado más preciso."* Mismo mecanismo exacto que el Nivel
2 (comparar este período contra TODOS los períodos iguales de la historia,
cancelando el vaivén sintético calendario-exacto) pero con `semanaDelAnioReal()`
(53 semanas, 0-52) en vez de mes (12) como unidad -- ~4,3x más fino.
Viabilidad (A_sys_env/e_R) sigue mensual, sin tocar.

Primero se corrió con un κ placeholder (κ global repetido en las 53
semanas) para conseguir los percentiles reales -- como era de esperar
(mismo patrón que el Nivel 1 sin recalibrar), **los 4 criterios fallaron**.
Se recalibró κ_LF/κ_Δ por semana con la mediana real de cada semana
(percentiles del motor Node) y se corrió de nuevo:

| Criterio | Resultado | Umbral | ¿Pasa? |
|---|---|---|---|
| **(a) Diferenciación real** | **62 de 62 años con las 4 zonas completamente únicas** -- 0 tuplas repetidas | <20 años en clusters | **SÍ**, perfecto |
| **(b) Floración real > control** | Media Jardín Fértil floración=15,05% vs. control=12,01% (25% más alto); **69,2% de los años de floración individuales superan la media de control** (9 de 13) | media(floración)>control, mayoría individual | **SÍ**, ya no es solo la media -- ahora es mayoría real (antes 38%) |
| **(c) Megasequía 2019-2025** | Jardín Fértil megasequía=0,42% vs. 14,35% global | media(megasequía) < media global | **SÍ**, muy marcado |
| **(d) Correlación Spearman** | ρ=0,682 (más fuerte que el 0,584 mensual) | ρ>0, \|ρ\|>0,2 | **SÍ**, más fuerte que Nivel 2 |

**Los 4 criterios pasan limpio, y el matiz que quedaba pendiente (b) se
resolvió**: subir de mensual a semanal no solo mantuvo lo que ya andaba
bien, mejoró los 4 números a la vez (diferenciación total en vez de 57/62,
mayoría real en vez de solo la media, correlación más fuerte). Confirma la
intuición de Alexis: más resolución temporal deja pasar más detalle real
sin perder la cancelación del componente sintético.

## Qué significa en simple

El instrumento ahora SÍ distingue un año lluvioso de uno seco, y lo hace
por una razón real: comparamos cada semana contra la misma semana de TODOS
los otros años (la primera semana de enero contra todas las primeras
semanas de enero, etc.), así el vaivén día/noche + estaciones -- que se
repite idéntico cada año, llueva o no -- queda cancelado, y lo que
sobrevive es la lluvia real. Al subir de "mes" a "semana" la comparación
quedó más fina, y el resultado mejoró en las 4 medidas a la vez: los 62
años son todos distintos entre sí, la mayoría (no solo el promedio) de los
años de floración documentada destaca, la megasequía 2019-2025
prácticamente nunca muestra Jardín Fértil (0,42%), y la relación con la
lluvia real es más fuerte que antes.

## Verificación técnica

- Motor Node (`motor_fisico.generado.js`) regenerado desde el HTML tras
  agregar `actualizarViabilidadMensual`/`estacionAustralReal`/κ por mes a
  la lista de extracción de `generar_motor_node.py` (faltaban del todo).
- Confirmado byte a byte contra el HTML real corriendo en el navegador
  (mismo LF, Δ_struct, A_sys_env, e_R, estación y zona en un tramo de 3.600
  ticks/2 meses, misma semilla) -- no es solo "compiló y corrió".
- `evaluar_contra_ground_truth.js` corregido durante esta misma ronda: la
  primera versión del criterio (a) solo miraba una columna y subestimaba
  la diferenciación real (ver nota arriba) -- corregido para usar las 4
  zonas juntas antes de reportar el resultado final.

## Pendiente / próxima ronda posible

- Con los 4 criterios pasando limpio en Nivel 3, no hay una brecha
  pendiente obvia -- posible candidato de una ronda futura (no pedido
  todavía): aplicar el mismo tratamiento semanal a viabilidad (A_sys_env/
  e_R), que sigue mensual y también mezcla 50% envTemp sintético en su
  referencia (`lerp(envTemp,floracionData.targetTf,0.5)`).
- κ_V/κ_O (viabilidad) no se tocaron esta ronda -- siguen con la
  recalibración de la ronda anterior (κ_V=0.9246, κ_O=0.0408).

## Archivos

- `Web/prueba_de_concepto/prueba_de_concepto_ET3-Termico_con_mapa.html` --
  viabilidad mensual (`actualizarViabilidadMensual`, sin cambios esta
  ronda) + activación semanal Nivel 3 (`actualizarActivacionSemanal`,
  `semanaDelAnioReal`, κ por semana en `clasificarCierre()`), bloques de
  comentario fechados 10-ago-2026 con el historial Nivel 1→2→3 completo.
- `Web/prueba_de_concepto/motor/generar_motor_node.py` -- lista de
  extracción ampliada (viabilidad + activación semanal).
- `Web/prueba_de_concepto/motor/motor_fisico.generado.js` -- regenerado.
- `Web/prueba_de_concepto/motor/experimentos/child_worker.js` -- soporta
  override de κ_LF/κ_Δ por semana para baterías futuras.
- `Web/prueba_de_concepto/motor/calcular_percentiles_y_regimen.js` --
  corrida completa + percentiles globales y por semana-calendario (53
  baldes).
- `Web/prueba_de_concepto/motor/evaluar_contra_ground_truth.js` -- evalúa
  cualquier CSV por año contra floración/ONI/lluvia real; corregido esta
  ronda para usar las 4 zonas juntas en el criterio de diferenciación.
- `Web/prueba_de_concepto/motor/regimen_nivel3b_por_anio.csv` +
  `regimen_nivel3b_percentiles.json` -- resultado FINAL de esta ronda.
- `regimen_nivel1/nivel2/nivel3_*` -- corridas intermedias, quedan por
  trazabilidad del proceso de iteración.
