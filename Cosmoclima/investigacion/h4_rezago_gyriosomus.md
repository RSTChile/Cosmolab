# H4 — Rezago ecológico de *Gyriosomus* respecto al pico de floración

Fecha: 2026-07-31. Primera pasada con datos ya disponibles. **Estado: evidencia parcial
a favor, NO es un cierre de H4** — ver limitaciones abajo. Corresponde a la hipótesis
formalizada en `hipotesis_y_modelo_formal.md` (sección H4).

## Qué se hizo
Los `gyriosomus_gbif_facets.csv` originales eran dos distribuciones marginales (mes
agregado sobre todos los años, año agregado sobre todos los meses) — no permiten saber
en qué mes ocurrió el pico DENTRO de un año específico. Se consultó directo la API de
GBIF (`occurrence/search`, `taxonKey=4760162` — género *Gyriosomus*, resuelto vía
`species/match`) pidiendo el facet de mes filtrado año por año, para los años con
registros (1957, 1989, 2015-2025). Esto da la distribución mes×año real (conteo
conjunto), no solo las dos marginales.

## Los dos casos con fecha de floración conocida con precisión
De los papers ya revisados (ver `desierto_florido_albedo_ndvi.md`, sección "rizobacterias
del Desierto Florido"), solo dos años tienen **fecha de pico de floración documentada
con precisión de mes**:

| Año | Pico de floración (fuente) | Ocurrencias *Gyriosomus* GBIF por mes (ese año) | Pico de escarabajos |
|---|---|---|---|
| 2014 | octubre (floración local) | nov=1 (único registro con mes ese año) | noviembre |
| 2015 | septiembre (floración extensa) | oct=4, nov=3, dic=1 (8 de 12 registros con mes) | octubre |

En ambos casos el pico de escarabajos cae **~1 mes después** del pico de floración
reportado — dirección consistente con H4 (rezago positivo, no simultaneidad). n es muy
chico (1 y 8 registros respectivamente), así que esto es sugerente, no concluyente.

## Patrón agregado (contexto, no prueba directa)
Con todos los años que tienen algún dato de mes (1957, 1989, 2015-2025), el pico de
ocurrencias sigue concentrado en **octubre-noviembre**, coherente con lo ya visto en el
facet agregado original. Encaja con la mecánica ya establecida: si la lluvia
germinadora cae en invierno (jun-jul) y el NDVI responde con ~3 meses de rezago
(dato de He et al. 2017), el pico de floración caería hacia septiembre-octubre — y el
pico de escarabajos en octubre-noviembre queda justo encima o levantemente después de
esa ventana. Un caso atípico: **2021 tiene 6 registros en julio** (pleno invierno, fuera
del patrón sept-dic) — no se investigó a qué corresponde; queda como pregunta abierta,
no se fuerza una explicación.

## Limitaciones honestas (por qué esto NO cierra H4)
1. **Solo 2 años con fecha de pico de floración documentada con precisión** (2014,
   2015) — el resto del catálogo de años de floración (1983-2024) da solo el año, no el
   mes exacto del pico. La Tabla 1 de Chávez et al. 2019 (bloqueada) tendría esa fecha
   para los 13 eventos satelitales — sigue siendo el hueco que más ayudaría aquí.
2. **La mayoría de los registros GBIF no tienen mes** (para 2017, por ejemplo, solo 21
   de 90 registros totales tienen mes poblado) — el facet de mes solo cuenta los que sí
   lo tienen, así que el "pico" por año puede estar sesgado por qué registros
   específicos traen fecha completa, no solo por biología real.
3. **2017 no se pudo usar**: el resumen que tenemos del paper de rizobacterias dice
   "evento completo, con etapas de pre-floración y floración plena" pero no da el mes
   exacto del pico — haría falta el PDF completo de ese paper (no obtenido aún) para
   sumar un tercer caso de comparación directa.
4. El conteo de registros GBIF por AÑO (mucho más alto 2017-2025 que antes) responde
   probablemente más al crecimiento del uso de iNaturalist/ciencia ciudadana en Chile
   que a la intensidad real de floración de cada año — por eso este análisis usa solo
   la distribución DENTRO de cada año (mes relativo), donde ese sesgo de esfuerzo de
   muestreo se cancela, y evita comparar magnitudes absolutas entre años.

## Por qué el rezago es FIJO y corto aquí, no variable (experiencia de campo, 01-ago-2026)
Alexis comentó una observación propia de décadas de expediciones de colecta: en el SUR
de Chile (Concepción hacia el sur), cuando los inviernos eran muy lluviosos, la
entomofauna typo demoraba MÁS en emerger (buenos meses de colecta se corrían a
fines de diciembre-febrero, en vez de octubre) — un corrimiento de 1-2 meses ligado a
la abundancia de lluvia invernal. Consultado si esto aplicaría a *Gyriosomus* y H4, fue
explícito en que **NO es extrapolable** a la zona árida del norte: esa observación es de
una zona y fauna distinta (sur, húmeda, lluvia regular), no de Atacama-Coquimbo.

Su explicación de por qué el sistema norte-árido debería comportarse distinto (y por
qué el rezago FIJO ~30 días de H4 tiene sentido biológico, no es solo un ajuste
estadístico): en la zona de estudio la norma es la sequía — las especies **no pueden
esperar**, porque esperar significaría perder la ventana de recurso efímero. Emergen
apenas la humedad alcanza el umbral suficiente para asegurar que las plantas también
van a emerger (germinación de las anuales), sin margen para un retraso variable como el
que se observa en el sur húmedo (donde el recurso no es tan efímero/urgente). Esto es
apoyo ecológico cualitativo para mantener el rezago de H4 como una constante corta y
rápida (no una función variable de la intensidad de lluvia) — no cambia el modelo, pero
documenta por qué NO se debe intentar una versión "rezago proporcional a la lluvia"
copiando el patrón del sur.

## Conclusión provisoria
Los dos casos con fecha precisa (2014, 2015) apuntan en la dirección que predice H4
(rezago positivo, escarabajos después de la floración, no simultáneos) y son
consistentes con la observación de campo ya citada de Anguita-Salinas et al. 2026 (mayor
densidad cuando las anuales ya se estaban marchitando). Pero con n=2 eventos y datos de
mes incompletos en GBIF, esto es **evidencia de apoyo, no una prueba** — H4 queda
"no falsada hasta ahora" con lo que tenemos, a la espera de la Tabla 1 de Chávez et al.
(años/meses-pico de los 13 eventos) para poder correr la comparación completa que pide
la hipótesis original.

## Actualización 01-ago-2026 — refuerzo independiente (agentes de investigación)
Una ronda de búsqueda bibliográfica sistemática (4 agentes en paralelo, ver
`gyriosomus_base_de_datos_consolidada.md` §3) encontró **4 fuentes adicionales
independientes**, de décadas y sitios distintos, todas consistentes con la misma ventana
temporal:

- **Cepeda-Pizarro (1989, Las Cardas, 30°13'S)**: 73% de toda la actividad anual de *G.
  luczoti* concentrada en un único pulso de primavera (1978), ligado explícitamente por
  el autor a la lluvia del invierno previo.
- **Cepeda-Pizarro et al. (2005a, Llanos de Challe, año ENOS 1997)**: octubre = mes de
  mayor contribución numérica; *G. kingi* + *G. planicollis* dominan 96,3% de los
  tenebriónidos del hábitat dunario costero ese año.
- **Cepeda-Pizarro et al. (2005b, transecto 27°-30°S)**: modelo fenológico explícito
  propuesto por los propios autores — lluvia >20mm a fines de invierno dispara
  floración con pico oct-nov y declinación hacia enero; ciclo univoltino con diapausa
  larvaria, pupación fin de invierno, maduración de ovarios en primavera.
- **Pizarro-Araya et al. (2007)**: dato directo de laboratorio — oviposición de *G.
  kingi* inicia a fines de septiembre.
- **Zúñiga-Reinoso, Pinto & Predel (2019)**: *G. camanchaca* colectada 16-X-2017 durante
  floración intensa en Paposo Norte; cita textual explícita conectando emergencia
  masiva de adultos con años lluviosos de El Niño.

**Esto no cierra H4** (sigue faltando el cruce evento-por-evento contra la Tabla 1 de
Chávez), pero cambia el estado de la hipótesis de "n=2, evidencia sugerente" a **"patrón
de ventana sep-dic/pico oct-nov reproducido de forma independiente en al menos 4-5
fuentes distintas"** — un refuerzo real, no solo acumulación de citas: son estudios de
autores, décadas y localidades distintas todos convergiendo en la misma ventana
temporal sin haberse propuesto explícitamente testear H4 tal como está formalizada aquí.
