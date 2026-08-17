# Desierto Florido: albedo, NDVI y monitoreo satelital

## Fuentes
1. Chávez, R.O. et al. (2019). "GIMMS NDVI time series reveal the extent, duration, and
   intensity of 'blooming desert' events in the hyper-arid Atacama Desert, Northern
   Chile". *International Journal of Applied Earth Observation and Geoinformation* 76:
   193-203. DOI: 10.1016/j.jag.2018.11.013.
   Registro (sin PDF completo accesible): https://repositorio.uchile.cl/handle/2250/171758
2. Estudio de validación de albedo MODIS en el Atacama (PMC, acceso abierto):
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8494836/
3. CONAF — plataforma de monitoreo fenológico satelital (SNASPE):
   https://sites.google.com/conaf.cl/monitoreo-snaspe/pagina-principal/monitoreo-satelital/monitoreo-de-fenologia

## Eventos de floración del desierto, 1981-2015 (Chávez et al. 2019)
El estudio usa NDVI GIMMS para reconstruir la fenología de superficie del Atacama
precordillerano (8 km de resolución de píxel) y detecta anomalías positivas de NDVI
como eventos de "desierto florido".

- **13 eventos de floración identificados entre 1981 y 2015.**
- Duración promedio reportada: ~166 días. Evento más largo: 2002-03 (hasta 270 días
  según un extracto; otro extracto ubica el evento mayor en 2011 con 180 días y
  11.136 km²). **Los dos extractos automáticos del abstract no coinciden entre sí en
  los detalles finos** (métricas exactas por evento) — el texto completo del paper
  está paywalled, así que estos números puntuales deben tratarse como PROVISIONALES,
  no como dato verificado, hasta poder leer la Tabla 1 original.
- Años de eventos mencionados en al menos un extracto: 1982-83, 1991, 1997-98, 2000,
  2002-03, 2005, 2011, 2012, 2015-16. (Lista a confirmar contra el PDF completo.)
- Dato sí confiable (aparece en ambos extractos y en el resumen general de búsqueda):
  **los tres eventos "mayores" según todas las métricas fueron 1997-98, 2002-03 y
  2011** — y 1997-98 es, no por casualidad, uno de los El Niño más fuertes jamás
  registrados instrumentalmente.
- **~60% de los eventos comienzan en julio-agosto** (pleno invierno austral) — coherente
  con el mecanismo: lluvia de invierno → germinación → floración de primavera.

## Albedo — validación MODIS en el Atacama (dato de referencia, sin floración)
Producto: **MODIS MCD43A3 v6**, 500 m de resolución, bandas espectrales 1-7 (459-2155 nm),
albedo black-sky y white-sky. Validado en terreno en septiembre 2018 con
espectrorradiómetro ASD FieldSpec 4 en 20 sitios.

- Rango visible (400-700 nm): mínimo ~0.10 (Pampa Sur), máximo ~0.25 (Huara).
- Sitios norteños (18-22°S): promedio ~0.17.
- Sitios sureños (27-30°S, más cerca de la zona de floración de Norte Chico):
  ~10% más bajo en visible, ~50% más bajo en UV que el norte — atribuido en el propio
  paper a la presencia de "vegetación dispersa".
- Precisión de MODIS vs. terreno: R = 0.94-0.98; sesgo dentro de ±5% para <27°S,
  sesgo de -8% para 27-30°S.
- **Este paper NO mide directamente el efecto de un evento de floración sobre el
  albedo** — es la línea base de albedo de suelo desértico (con y sin vegetación
  dispersa de fondo), útil como referencia, no como el experimento en sí.

## CONAF SNASPE — monitoreo fenológico satelital
Plataforma operativa de land-surface phenology (LSP) vía MODIS (16 días), con
PhenoCams de validación en terreno. **Cubre 4 áreas protegidas — La Campana, Río
Clarillo, Nahuelbuta, Pumalín — ninguna de ellas en la zona del Desierto Florido.**
Sirve como referencia metodológica (cómo Chile ya monitorea fenología por satélite),
pero no como fuente directa de datos para Norte Chico.

## Actualización 2026-07-31: Chávez et al. 2019 (resumen público, no el texto completo)
Alexis consiguió el resumen/vista previa pública de ScienceDirect (no pudo comprar el
PDF completo) — guardado en `fuentes/chavez_2019_desierto_florido_preview_sciencedirect.docx`.
Esto SÍ resuelve la contradicción que habíamos anotado antes entre "2011: 11.136 km²,
180 días" y "2002-03: 270 días" — **no son contradictorios, son métricas distintas**:
- **Evento más importante según el criterio combinado (área × duración × intensidad):
  2011 — 11.136 km², 180 días, entre julio y diciembre.**
- **Evento más LARGO en duración: 2002-2003, 270 días.**
- Los tres eventos "importantes" de los 13 detectados: **1997-98, 2002-03, 2011.**
- Duración promedio de los 13 eventos: 166 días. 60% de los eventos comienzan en
  julio-agosto (invierno) y desaparecen para el verano siguiente.
- **Umbral de germinación**: algunas especies necesitan un mínimo de **15 mm de
  precipitación acumulada en un año** para que las semillas dormidas germinen — dato
  concreto y citable (Armesto et al. 1993; Vidiella et al. 1999).
- Ventana de acumulación: la floración se dispara por precipitación acumulada durante
  **2 a 12 meses antes y durante** el evento — no es solo la lluvia del mes.
- Años ENSO "muy fuertes" según NOAA asociados a floración: **1982-83, 1997-98,
  2015-16**. Otros años de floración documentados en la literatura previa (no todos
  necesariamente ENSO fuerte): 1983, 1991, 1998, 2000, 2005, 2012, 2015.
- Zona geográfica: la floración ocurre principalmente en la **transición entre la
  región bioclimática hiperárida y la mediterránea**, en la parte sur del Atacama y la
  costa centro-sur — el mismo ecotono de Norte Chico que ya establecimos con Gyriosomus.
- **Dato citado dentro del propio paper, y muy importante para nosotros**: "estos
  aumentos repentinos en producción primaria... generan cambios en el clima local: un
  aumento de la evapotranspiración y **una disminución del albedo** (He et al., 2017)."
  Es decir, Chávez et al. citan como hecho ya establecido exactamente nuestra hipótesis
  — lo que nos llevó a buscar ese paper también (ver abajo).

## He et al. (2017) — TEXTO COMPLETO obtenido (Alexis lo consiguió, 2026-07-31)
**He, B., Huang, L., Liu, J., Wang, H., Lű, A., Jiang, W. & Chen, Z. (2017). "The
observed cooling effect of desert blooms based on high-resolution Moderate Resolution
Imaging Spectroradiometer products". *Earth and Space Science* 4(5): 247-256.**
DOI: 10.1002/2016EA000238. Acceso abierto (CC BY-NC-ND). Guardado en
`fuentes/he_2017_cooling_effect_desert_blooms.pdf`.

### Zona de estudio — ojo con el desfase geográfico respecto a Gyriosomus
Regiones de Tarapacá, Antofagasta y Atacama, **17.5°S a 29.7°S** (67.0°W-71.5°W).
**Esto se solapa solo parcialmente con la zona de Gyriosomus (25°S-34°S) y NO llega a
cubrir la Zona de Alta Simpatría Cladística (30.5°S-31.5°S)**, que es donde está la
mayor diversidad de escarabajos. El corredor Vallenar-Huasco (27-29°S) sí queda
adentro. Hay que ser honestos con esto: lo que sigue es evidencia sólida del mecanismo
general, no una medición hecha exactamente en el punto caliente de Gyriosomus.

### Datos y método
MODIS Terra LST (MOD11A2, 2000-2015, ~10:30/22:30 hora local) + MODIS Aqua LST
(MYD11A2, 2002-2015, ~13:30/01:30) + MODIS ET (MOD16A2, 2000-2014, Penman-Monteith) +
MODIS albedo (MCD43B3, 2000-2015, 16 días, promedio black-sky/white-sky) + NDVI/EVI
(MOD13A2) + precipitación CRU TS3.23 (0.5°, 2000-2014). Umbral de vegetación: NDVI≤0.15
o EVI≤0.1 = desierto/suelo desnudo; el resto, vegetado.

Dos métodos de comparación, **con resultados de calidad muy distinta**:
- **ΔLST_Space** (píxel vegetado vs. píxeles desnudos ADYACENTES, mismo mes): la
  comparación robusta.
- **ΔLST_Time** (el MISMO píxel, año con floración vs. año sin floración): la
  comparación ruidosa — confundida por clima de fondo año a año (El Niño/La Niña,
  niebla, viento).
- **Lección metodológica para nuestro propio experimento**: comparar espacialmente
  (vegetado vs. desnudo, mismo momento) da una señal limpia; comparar temporalmente
  (mismo lugar, distintos años) es mucho más ruidoso. Si diseñamos algo con series de
  tiempo año-a-año, hay que esperar más ruido que si comparamos zonas vegetadas vs.
  no-vegetadas en el mismo satélite-pasada.

### Los números — con la comparación espacial (robusta)
- Área vegetada 2000-2015: entre 2.400 km² y 25.200 km²; máximos en **2000, 2012 y
  2015** (coincide con años que Chávez et al. también marcan como floración grande).
- Rezago vegetación-lluvia: **3 meses** (NDVI, R=0.36) o **2 meses** (EVI, R=0.30).
- **Enfriamiento diurno: 0.31°C ± 0.05°C. Calentamiento nocturno: 0.02°C ± 0.02°C.
  Enfriamiento neto diario: 0.29°C ± 0.07°C** (jul-2002 a dic-2015, promedio de ambos
  satélites).
- **Cambio de albedo medido directamente: cae 2.1% ± 0.31%** (comparación espacial,
  vegetado vs. desnudo) y **cae 0.9% ± 0.58%** (comparación temporal, año con floración
  vs. sin ella) — **valores relativos pequeños pero reales y medidos**, no supuestos.
- Cambio de evapotranspiración: **+0.21 mm ± 0.11 mm/día** (espacial) y **+1.9 mm ±
  1.46 mm** (temporal, tras la floración).
- **Regresión lineal (Tabla 2 del paper) — reparto de responsabilidad, no solo
  "gana uno u otro":**
  - De día: ET + albedo explican el **57.1%** de la varianza de ΔLST (R²=0.571,
    p<0.001). De esa varianza explicada, **ET aporta 58.8% y albedo aporta 41.2%**
    — el albedo SÍ pesa casi la mitad, no es un jugador menor.
  - De noche: ET + albedo explican el 48.7% (R²=0.487, p<0.001). De esa varianza, el
    **albedo solo explica 92.5%** — de noche el albedo manda y la evapotranspiración
    es casi irrelevante.
  - Números simples para divulgación: **+0.01 de NDVI = 0.062°C de enfriamiento
    de día, y 0.004°C de calentamiento de noche.**
- Con la comparación TEMPORAL (más parecida a "antes/después" en el mismo lugar), el
  efecto neto anual es mucho más débil y ruidoso: **0.13°C ± 0.38°C** (la desviación
  estándar es casi 3 veces el efecto — es decir, con este método el efecto casi no se
  distingue del ruido), y en **septiembre el efecto se invierte a calentamiento neto**
  (por baja precipitación ese mes, que debilita el enfriamiento por evapotranspiración).
- Los propios autores son honestos sobre los límites: esto es correlación, no
  prueba causalidad; hay otros factores no controlados (niebla, viento, ENSO de
  fondo); y el ET no se pudo separar entre evaporación de suelo y transpiración de
  planta (aunque argumentan que en período de floración domina la transpiración).
- Dato adicional útil para divulgación: una floración dura ~19 semanas tras un solo
  pulso de lluvia pequeño (Vidiella et al. 1999) — casi calza con las 166 días (~24
  semanas) promedio que reportó Chávez et al. 2019.

### Síntesis para el experimento: qué significa esto en limpio
1. **La hipótesis de Lovelock queda confirmada con números reales, no supuesta**: la
   floración SÍ baja el albedo (2.1% en comparación espacial), medido con MODIS.
2. **Pero el efecto neto sobre temperatura no es solo del albedo** — de día gana la
   evapotranspiración (58.8% de la varianza explicada), aunque el albedo aporta un
   41.2% nada despreciable; de noche el albedo sí domina casi por completo (92.5%).
   La forma honesta de decirlo: "la floración enfría la superficie en el neto diario,
   y el albedo es uno de los dos mecanismos reales detrás de eso — no el único, pero
   tampoco menor."
3. **Método a copiar si hacemos nuestra propia medición**: comparación espacial
   (vegetado vs. desnudo, mismo momento), no comparación temporal (mismo lugar, años
   distintos) — esta última es demasiado ruidosa por el clima de fondo.
4. **Advertencia geográfica**: este estudio no cubre la Zona de Alta Simpatría
   Cladística de Gyriosomus (30.5-31.5°S) — si se quiere un vínculo directo con el
   punto caliente de diversidad de escarabajos, hay que correr este mismo análisis (u
   otro con Sentinel-2) específicamente ahí, no asumir que el número de He et al. se
   traslada sin más al sur de su zona de estudio.

## Tres papers adicionales (todos de acceso abierto, Frontiers) con años y sitios reales
1. **Flower coloring / eco-evolución del color floral** (Frontiers Ecol. Evol. 2022,
   10.3389/fevo.2022.957318) — muestreó la floración masiva de **Caldera, sept-oct
   2021**, especie *Cistanthe longiscapa* (110 flores analizadas). Menciona periodos
   históricos húmedos (1920-1945, 1976-2002) y secos (~1910, 1945-1975) — es decir,
   evidencia independiente de que el ciclo húmedo/seco de la zona tiene fluctuaciones
   de escala DECADAL, no solo evento-a-evento. Hipótesis del paper: el polimorfismo de
   color de la flor favorece la polinización cruzada bajo lluvia impredecible — otro
   mecanismo de acoplamiento ecológico profundo, en la misma línea que el argumento de
   Gyriosomus.
2. **Rizobacterias del Desierto Florido** (Frontiers Microbiol. 2020, 10.3389/fmicb.2020.00571)
   — muestreó **tres eventos reales con fecha y sitio**: 2014 (floración local, octubre),
   2015 (floración extensa, septiembre), 2017 (evento completo, con etapas de pre-
   floración y floración plena), en tres puntos entre 27.5°S y 28.8°S (Vallenar-Huasco,
   el corazón de la zona de Gyriosomus). No mide albedo directamente, pero confirma
   fechas/sitios reales que sí sirven para cruzar con Gyriosomus y con NDVI.
3. Búsqueda de un catálogo consolidado de años de floración (más allá de lo satelital
   1981-2015): la Wikipedia en español y sitios de turismo confirman 1983, 1987, 1991,
   1995, 1997, 2000, 2002, 2015 (dos floraciones, abril-mayo y sept-oct), 2017, 2021,
   2022, 2024 (temprana). **Esta lista es de calidad mixta** (mezcla papers, prensa y
   turismo) — sirve para tener un panorama, no para citar como dato duro sin volver a
   la fuente original de cada año.

## Bloqueado (para que Alexis intente con acceso institucional)
- Chávez et al. (2019), *Int. J. Applied Earth Observation and Geoinformation* 76:
  193-203 — DOI: https://doi.org/10.1016/j.jag.2018.11.013 — ya tenemos el resumen
  público completo, pero la **Tabla 1** (métricas exactas de los 13 eventos, no solo
  los 3 "importantes") sigue sin verse. Bloqueado en ScienceDirect (403/paywall) y
  ResearchGate ("Request PDF").

## Lo que falta para el experimento real
1. La Tabla 1 completa de Chávez et al. 2019 (ver arriba — bloqueada). He et al. 2017
   ya se obtuvo completo, no es pendiente.
2. Una serie de albedo MODIS (MCD43A3) o NDVI (MOD13Q1/Sentinel-2) recortada
   específicamente a la Zona de Alta Simpatría Cladística de Gyriosomus (30.5-31.5°S)
   — fuera del área que cubrió He et al. 2017 (17.5-29.7°S) — cruzando los años ya
   confirmados con fecha real (2014, 2015, 2017, 2021).
3. Idealmente, imágenes Sentinel-2 (10 m, desde 2015) para el evento actual de 2026,
   que tiene mucha mejor resolución espacial que MODIS/GIMMS para una franja angosta
   de floración costera.
