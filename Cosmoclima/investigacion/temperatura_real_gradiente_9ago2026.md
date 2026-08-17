# Temperatura real: curva diaria + gradiente latitudinal — 09-ago-2026

Alexis, mirando el gráfico "Pluviosidad real en vivo": *"¿Sabes qué nos
falta...? La temperatura diaria con mínima y máxima... te aseguro que visto
a lo largo de la zona desde Paposo a Santiago, veríamos una curva anual
descendente de las máximas desde el norte al centro... ¿Tenemos los datos
de temperatura en la BD?"* Confirmado que no: `pluviosidad_diaria_
consolidada.sqlite` es solo lluvia. Pidió las dos cosas: integrar
temperatura real en el gráfico existente, y una vista nueva con el
gradiente latitudinal Paposo→Machalí.

## Parte A — Temperatura real en "Pluviosidad real en vivo"

- Fuente: NASA POWER (`power.larc.nasa.gov`), vía `datos_clima.py`
  (biblioteca ya existente y probada en el proyecto). Mismo punto-reloj que
  `PLUVIOSIDAD_MENSUAL` (Huintil, -31.5669,-70.9817).
- `Web/prueba_de_concepto/obtener_temperatura_diaria.py`: trae diario
  1981-01-01→hoy (POWER no cubre antes), inyecta `TEMPERATURA_DIARIA_ZHCS`
  en el HTML (mismo patrón de marcador de bloque que `LLUVIA_DIARIA_1966_
  2017`/`ONI_BANDAS`). **16.654/16.657 días con dato real (100,0% cobertura,
  solo 3 días null por el rezago normal de POWER de unos días).**
- HTML: 2 datasets nuevos en `popChart` (Tmax rojo `#f87171`, Tmin celeste
  `#60a5fa`), eje `y3` propio (°C, derecha, offset). `serieTemperaturaGranular()`
  (cacheada, igual criterio que `seriePluviosidadRealGranular()`) reusa
  `claveFechaDiaria(dia)` ya existente — es SOLO visualización, no alimenta
  ningún cálculo de la física ni la floración.
- Verificado en navegador: sin errores de consola, curva visible desde 1981
  en adelante (nada antes, como corresponde), oscilación anual clara.

## Parte B — Gradiente latitudinal (Paposo→El Guindal)

- 9 estaciones REALES, ya presentes en `pluviosidad_diaria_consolidada.sqlite`
  (con miles de filas de lluvia real detrás, no coordenadas inventadas),
  norte a sur: Paposo, Tal-Tal, Copiapó, Vallenar, La Serena, Huintil (=
  mismo punto-reloj), San Felipe, Santiago (Quinta Normal), El Guindal.
- El ancla sur, **El Guindal** (-34,19°S), coincide casi exacto con el
  registro real MÁS AUSTRAL de *Gyriosomus laevigatus* en la Tabla S1
  (-34,1905,-70,6368) — confirma la fuente de forma independiente del
  sqlite de lluvia. Es el punto que Alexis identificó como "Machalí".
- `Web/prueba_de_concepto/obtener_gradiente_latitudinal.py`: NASA POWER
  diario 1981→hoy por estación, **promediado a mano** (no con el modo
  `"maximo"` de `agregar()`, que da el día más caluroso del período, no el
  promedio climatológico de las máximas diarias que se necesitaba acá).
  100% cobertura en las 9 estaciones.
- **Resultado real, verificado, NO monótono** (declarado así en el propio
  instrumento, no escondido):

  | Estación | lat | Tmax prom | Tmin prom |
  |---|---|---|---|
  | Paposo | -24,96 | 19,27°C | 14,90°C |
  | Tal-Tal | -25,40 | 19,45°C | 14,27°C |
  | Copiapó | -27,38 | 23,31°C | 11,73°C |
  | Vallenar | -28,59 | **24,65°C** (máximo del gradiente) | 11,85°C |
  | La Serena | -29,91 | 22,06°C | 12,84°C |
  | Huintil | -31,57 | 21,98°C | 11,09°C |
  | San Felipe | -32,75 | 18,66°C (mínimo del tramo sur) | 5,87°C |
  | Santiago | -33,45 | 23,02°C | 9,02°C |
  | El Guindal | -34,19 | 22,11°C | 8,41°C |

  Paposo/Tal-Tal (costeras, Corriente de Humboldt/camanchaca) son MÁS FRÍAS
  que Copiapó/Vallenar pese a estar más al norte — efecto real, no error.
  Confirma parcialmente la intuición de Alexis (Vallenar→sur sí tiende a
  bajar, con el valle interior San Felipe como excepción notable) pero NO
  una curva descendente pareja de punta a punta — el gradiente real mezcla
  estaciones costeras e interiores, declarado así en el texto de ayuda del
  instrumento.
- HTML: sección nueva "Temperatura en gradiente latitudinal", chart Chart.js
  con eje X lineal propio (latitud, `reverse:true` para Norte-izquierda),
  cada punto es una estación real (no interpolación), tooltip con el nombre
  real de la estación.
- Verificado: sin errores de consola; datos del chart confirmados
  programáticamente (9 puntos por dataset, valores exactos, tooltip
  devuelve el nombre correcto de cada estación en orden norte→sur).

## Ampliación — costa vs. valle interior (misma noche, 09-ago)

Alexis, tras ver que el gradiente no era monótono: *"quisiera... separar las
estaciones en costeras y del valle, para tener dos curvas longitudinales de
máximas y mínimas para zonas costeras e interior: en Copiapó, por ejemplo,
siempre en la costa amanece con baguada, mientras en el interior está
despejado... eso pasa por Humboldt, pero en interior afecta menos, así que
las máximas suelen ser más altas, y las mínimas también (el océano
modera)"*. Y pidió mover el chart justo debajo de "Pluviosidad real en
vivo" (estaba al final de la página).

- Ampliado de 9 a **14 estaciones reales** (7 costeras + 7 de valle,
  clasificación geográfica a mano — puerto/caleta real = costa, ciudad de
  valle a decenas de km de la costa = valle; el sqlite no trae distancia a
  la costa). Nuevas: Caldera, Huasco, Los Vilos Dmc, San Antonio (Pta.
  Panul), Vicuña/"Vicua (Inia)" — todas reales, verificadas en
  `pluviosidad_diaria_consolidada.sqlite` antes de usarlas.
- 4 datasets en vez de 2: máxima/mínima × costa/valle. Mismo color por
  variable (rojo=máxima, azul=mínima, consistente con el resto del
  instrumento), línea sólida=costa, punteada=valle.
- **Resultado real, visualmente claro**: el patrón que describió Alexis se
  confirma en la mayoría de los pares costa/valle a latitud comparable —
  Vallenar (valle) 24,65°C máx vs. Huasco (costa) 21,10°C; Santiago (valle)
  9,02°C mín vs. San Antonio (costa) 12,10°C mín (el océano modera de
  noche, tal como dijo). Excepción real declarada, no escondida: Vicuña
  (valle) sale MÁS FRÍA que La Serena (costa) en máxima (19,74°C vs.
  22,06°C) — dato real, no se fuerza a calzar el patrón general.
- Chart movido de su posición original (al final de la página) a
  inmediatamente después del chart-card de "Pluviosidad real en vivo",
  dentro de la misma sección "Estado del experimento" — verificado
  programáticamente (`popCard.nextElementSibling===card`).
- Verificado en navegador: sin errores de consola, 4 datasets con 7 puntos
  cada uno, dash pattern correcto en los datasets de valle, posición en el
  DOM confirmada.

## Archivos nuevos

- `Web/prueba_de_concepto/obtener_temperatura_diaria.py`
- `Web/prueba_de_concepto/obtener_gradiente_latitudinal.py`
- `investigacion/fuentes/temperatura_diaria_zhcs_nasa_power.csv`
- `investigacion/fuentes/temperatura_gradiente_latitudinal_nasa_power.csv`
- HTML: `TEMPERATURA_DIARIA_ZHCS`, `GRADIENTE_LATITUDINAL_TEMP` (generados,
  mismo patrón de marcador de bloque que el resto del instrumento).

## Pendiente

Ninguno de los 4 valores κ ni la física existente se tocó. Fases C/D/E del
plan de granularidad (`investigacion/informe_sesion_granularidad_8_9ago2026.md`)
siguen pendientes, sin relación con este cambio.
