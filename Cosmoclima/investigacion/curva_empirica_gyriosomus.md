# Primera curva empírica: lluvia → floración → Gyriosomus

Fecha: 2026-07-31. A pedido de Alexis: en vez de que el simulador use constantes de
crecimiento "plausibles" (elegidas a mano, al estilo de las tasas originales de
Daisyworld), la relación lluvia→floración se ajustó directo a datos reales acumulados
por año — aunque el conjunto sea chico. Datos en
`fuentes/curva_empirica_lluvia_floracion_gyriosomus.csv`.

## Los datos
- **Lluvia anual real**: NASA POWER (misma fuente ya usada para H5), corredor
  Huasco-Freirina (-28.53°S, -71.15°W) — 19 años con dato: 13 documentados como años de
  floración (Chávez et al. 2019 satelital + catálogo prensa/CONAF 1983-2024) y 6 años
  "quietos" de control, no catalogados como floración.
- **Ocurrencias de Gyriosomus por año**: facet de GBIF ya consultado antes en la sesión
  (género completo, no por especie) — solo hay dato desde 2015 en adelante con volumen
  real, antes de eso GBIF prácticamente no tiene registros.

## La curva (lluvia → probabilidad de floración)
Regresión logística de 1 variable, ajustada por descenso de gradiente a los 19 puntos:

```
logit(P_floración) = -1.995 + 0.0539 × lluvia_mm_anual
```

- **Cruce de 50% de probabilidad: ~37 mm/año.**
- Exactitud sobre los mismos 19 puntos (ajuste, no validación externa): **79%**.
- Con menos de ~20mm/año NUNCA hay floración documentada en esta muestra (1990:
  15,5mm, 1996: 11,2mm, 2019: 7,6mm — los tres "no"). Por sobre ~80mm, casi siempre SÍ
  (con la excepción real de 2008: 81mm sin floración documentada).

## Limitación honesta, no se fuerza
n=19 es chico — esto es un ajuste de exploración, no un modelo validado. Y el
solapamiento real (2008 con 81mm sin floración; 2012 y 2021, años de floración, con
menos de 31mm) confirma lo que Anguita-Salinas et al. 2026 ya decía: el total anual
solo no lo explica todo — la DISTRIBUCIÓN del pulso importa tanto como el acumulado
(H2 de `hipotesis_y_modelo_formal.md`). Esta curva es un primer piso empírico, no la
respuesta final.

## Por qué NO se ajustó floración→Gyriosomus con GBIF
Cruzando años con floración Y dato de Gyriosomus (2015, 2017, 2021, 2022, 2024) no hay
relación limpia: 2015 (bloom documentado, 61mm) solo tiene 12 registros, mientras 2020
(NO es año de floración catalogado, 42mm) tiene 77. La explicación más honesta no es
ecológica — es que el uso de iNaturalist/ciencia ciudadana en Chile creció fuerte
después de 2015-2017, así que el conteo por año mezcla "cuánta gente salió a
fotografiar" con "cuántos escarabajos hay realmente". Forzar una curva ahí sería
inventar precisión que no existe. Por eso el simulador sigue usando el rezago temporal
real de H4 (~1 mes, sí confirmado con fechas exactas) para CUÁNDO aparece Gyriosomus,
pero no una curva de magnitud ajustada a GBIF — la magnitud de Gyriosomus se deriva
proporcional a la floración misma (misma amplitud, con el rezago aplicado), no de un
ajuste independiente que estaría fabricando una precisión falsa.

## Qué cambia en el simulador
`computeFloracion()` en `Web/prueba_de_concepto/prueba_de_concepto_ET3-Termico_con_mapa.html`
usa esta curva logística real para decidir hacia dónde converge la floración según la
Lluvia acumulada, en vez de una tasa de crecimiento inventada. El umbral de 15mm
(Chávez et al. 2019) se mantiene como piso mecánico duro — por debajo de eso, cero,
sin importar lo que diga la curva.

## v2 (01-ago-2026) — recalibrada sobre el PICO MENSUAL, no el total anual
La v1 se calibró contra lluvia ANUAL de NASA POWER (corredor Huasco-Freirina, típico
7-140mm/año). Cuando el calendario real (1900-2027) empezó a alimentar el motor con la
lluvia de la estación **Huintil** (típico 70-540mm/año — mucho más lluviosa), casi
cualquier año con dato real superaba por lejos el umbral de 37mm y la floración
quedaba prácticamente permanente en ~90%. Alexis lo detectó viendo el rango completo
1900-2027 en el gráfico ("estuvo florecido todo el tiempo... está mal").

Se probaron variables candidatas sobre el mismo catálogo de 23 años documentados
(floración sí/no, Chávez et al. 2019 + prensa/CONAF), esta vez con la lluvia tomada
de `PLUVIOSIDAD_MENSUAL` (Huintil/CR2 + NASA POWER, la fuente real que el simulador
ya consulta — ver `linea_tiempo_pluviosidad_mensual_1900_2027.md`):

| variable candidata | exactitud (23 pts) |
|---|---|
| **pico mensual (mes más lluvioso del año)** | **70%** |
| suma de los 2 meses más lluviosos | 61% |
| Nº de meses ≥15mm ese año | 52% |

El pico mensual ganó — coherente con H2 (la distribución del pulso importa más que
el acumulado). Nueva curva:

```
logit(P_floración) = -1.2123 + 0.0185 × lluvia_mm_del_mes_mas_lluvioso
```

Cruce de 50% en **~66mm en el mes pico**. `state.lluviaAcumulada` (mismo nombre de
variable) ahora se llena con `picoMensualAnio(año)`, no con la suma de los 12 meses.

**Limitación honesta que la v2 tampoco resuelve**: 2021 (pico de 12mm) y 2024 (pico de
14mm) tienen floración documentada real con lluvia muy baja — coherente con el hallazgo
ya registrado de que la mayoría de las campañas de Gyriosomus fueron años La Niña,
floraciones en PARCHES localizados que un promedio agregado de toda la ZHCS no puede
ver. No se forzó la curva para "explicar" estos 2 puntos. Alexis, avisado de esto de
antemano: "entiendo que tenemos sólo una fuente de datos para eso, y que no es muy
buena, pero es lo que hay" — se procedió con el mejor ajuste disponible, declarado
como tal.

Verificado con Playwright (`stepFloracionSolo`, 1966-2027 completo): la floración ya
NO queda pegada arriba — sube claramente en 1997 (El Niño fuerte real, llega a 0.86) y
baja en la sequía 2019 (cae a 0.055) antes de volver a subir en 2020. Sigue pasando
buena parte del tiempo en valores intermedios/altos (48% de las muestras >0.5), lo cual
es honesto: Huintil es, en los hechos, una estación bastante lluviosa la mayoría de los
años — no es un artefacto de calibración, es lo que dice el dato real de esa estación.

## v3 (09-ago-2026) — se probó agregar temperatura real como segundo predictor, NO SE ADOPTA

Alexis pidió recalcular floración con 4-5 parámetros (lluvia, albedo, Tmax/Tmin,
elevación) cruzados por costa/valle. Investigación previa (2 agentes Explore, ver
`/Users/alexis/.claude/plans/majestic-whistling-canyon.md` para el detalle completo)
encontró que solo temperatura tiene cobertura real completa para los 23 años
documentados (NASA POWER desde 1981, los 23 años son 1983-2024) — albedo/NDVI real
solo existe para 5 meses de 2026 (no histórico), y no hay ningún ground truth
histórico multi-sitio que permita validar una curva partida por costa/valle. Se
acotó el alcance a esto: probar temperatura, con la validación que v1/v2 nunca
tuvieron.

**Diseño**: `Web/prueba_de_concepto/motor/recalibrar_con_temperatura.py`, regresión
logística implementada a mano (Newton-Raphson/IRLS, sin sklearn) para poder correr
leave-one-out (LOOCV, refit 23 veces dejando un año afuera cada vez) además de la
exactitud in-sample de siempre. Predictor de temperatura: Tmax promedio de los 90
días reales que empiezan el 1° del mes de pico de lluvia (misma ventana temporal que
ya usa el ascenso de `computeFloracion()`, no un número nuevo inventado para esto).
**Criterio de adopción declarado ANTES de correr el ajuste** (para no elegir el
resultado que se ve mejor después de verlo): se adopta el modelo con temperatura
SOLO si su exactitud LOOCV iguala o supera la del modelo actual.

**Resultado**:

| modelo | exactitud in-sample | exactitud LOOCV |
|---|---|---|
| solo lluvia (reajustado acá, B0=-1.2434, B1=0.0188 — coincide con los B0=-1.2123, B1=0.0185 vigentes, valida el método) | 69,6% | **65,2%** |
| lluvia + Tmax (B_tmax=-0.7952) | 69,6% (empatado) | **60,9%** |

In-sample los dos modelos empatan — agregar temperatura ni siquiera mejoraba el
ajuste que ya se venía reportando. En LOOCV, agregar temperatura empeora (65,2%→
60,9%): con n=23 y un segundo parámetro, el modelo memoriza más que generaliza —
exactamente el riesgo de sobreajuste que la v1/v2 nunca habían medido. **No se
adopta, se mantiene `EMP_B0=-1.2123, EMP_B1=0.0185` sin cambios.** Resultado
negativo real, documentado en vez de descartado — cierra la Fase D.3 (hold-out/
leave-one-out) que había quedado pendiente del plan de granularidad anterior: la
curva actual SÍ sobrevive a una validación más honesta que el in-sample de siempre
(65,2% LOOCV vs. 70% in-sample — una caída esperable y no catastrófica).

Tabla completa por año (ambos modelos, ambas predicciones):
`Web/prueba_de_concepto/motor/recalibracion_temperatura_por_anio.csv`. Resumen
JSON: `Web/prueba_de_concepto/motor/recalibracion_temperatura_resumen.json`.

## Contraste contra NDVI real (satélite) — 09-ago-2026 (2), NO como recalibración

Alexis, tras el resultado de temperatura: *"si tenemos datos reales de pluviosidad
y temperatura, que no dependen de una fotografía satelital, podemos contrastar lo
que dice la fotografía contra datos duros de terreno"*. Esto NO es una recalibración
más — la curva sigue siendo la misma de siempre (v2, sin cambios). Es un testigo
INDEPENDIENTE: se simula día a día lo que la curva ya vigente predice con lluvia
real, y se compara DESPUÉS contra NDVI real (verdor visto por satélite MODIS,
descargado con `investigacion/fuentes/descargar_ndvi_historico_huintil.py`, 605
composites reales de 16 días, Huintil, 2000-2026 — límite real descubierto de la
API pública de ORNL DAAC: máximo ~160 días por pedido, se armó en bloques).

**Dos trayectorias simuladas** (`investigacion/comparar_floracion_vs_ndvi_historico.py`):
- **Valle**: Huintil, la misma lluvia real de siempre.
- **Costa** (exploratorio, a pedido de Alexis para "verificar el contraste
  valle/costa"): Los Vilos Dmc (misma latitud que Huintil, real, CR2/DGA código
  4820001, diario real 1982-2017-05-31) — MISMA curva, sin recalibrar para esa
  estación, se corta donde termina el dato real (no se inventa una extensión).

**Resultado, ventana común 2000-02-18 a 2017-05-31 (n=398 fechas en ambas)**:

| serie | correlación con anomalía NDVI real |
|---|---|
| Floración modelo — VALLE (Huintil) | **r=0,335** |
| Floración modelo — COSTA (Los Vilos, misma curva sin recalibrar) | r=0,288 |

El valle correlaciona más con el NDVI real que la costa — coherente y esperable:
la curva se ajustó con lluvia de Huintil, así que sigue mejor el verdor real de
Huintil que cuando se traslada sin recalibrar a otro punto. No se fuerza a que
salga distinto. Chequeo cruzado con el paper real de Campos Nazer et al. (2021,
GeoFocus, acceso abierto): **2017 es el máximo tanto en NDVI real como en la
floración del modelo** — coincide con que ellos también encontraron 2017 como el
año de mayor floración de su serie 2000-2017, dos fuentes independientes de
acuerdo. 2021 es el mínimo en ambas series también.

**Gráfico nuevo en el instrumento**: "Floración del modelo vs. NDVI real", 3
curvas (valle sólida verde, costa punteada naranja, NDVI real en puntos morados,
eje propio), en `prueba_de_concepto_ET3-Termico_con_mapa.html`, sección "Estado
del experimento", justo debajo del gradiente latitudinal. Verificado en navegador:
sin errores de consola, 11.179 puntos diarios por trayectoria, 605 puntos NDVI
reales, corte real de la serie costa en 2017-05-31 confirmado.
