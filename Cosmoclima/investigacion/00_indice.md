# Índice de investigación — El Niño / Desierto Florido / Gyriosomus

Fase: **Aprehensión** (recopilar todo dato relevante, sin diseñar aún el experimento).
Fecha de esta ronda de investigación: 2026-07-31.

Alcance acotado del experimento futuro (decidido con Alexis): validar la hipótesis
de Lovelock de que la flora cambia el albedo local/regional (NO el efecto planetario
completo de Daisyworld), y mostrar que el ciclo El Niño/Desierto Florido/Gyriosomus es
un fenómeno cíclico y recursivo antiguo — no un producto del calentamiento antropogénico
reciente, en respuesta específica a coberturas mediáticas que atribuyen este Niño
"Godzilla" 2026 al cambio climático.

## Archivos de esta carpeta
- `enso_cronologia_chile_1536_1900.csv` + `enso_cronologia_notas.md` — cronología
  año-por-año de eventos lluviosos/ENSO en Chile central, 1536-1900, extraída de
  Ortlieb (1994). Hallazgo clave: correlación débil pre-1817, fuerte post-1817 — el
  ciclo es real y antiguo pero con acoplamiento variable en el tiempo.
- `gyriosomus_distribucion.md` — distribución, endemismo y ecología de *Gyriosomus*
  (34-44+ especies, 25°S-34°S, Provincia Biogeográfica Coquimbana, rango ancestral
  desde el Eoceno).
- `desierto_florido_albedo_ndvi.md` — 13 eventos de floración satelital 1981-2015
  (Chávez et al. 2019), validación de albedo MODIS en el Atacama, y qué monitoreo
  satelital ya existe en Chile (CONAF SNASPE).
- `fuentes/ortlieb_1994_precipitaciones_enso_chile.pdf` — PDF completo de la fuente
  primaria de la cronología ENSO.

## Actualización 2026-07-31 (segunda ronda)
- `gyriosomus_distribucion.md` ampliado: resumen de la filogenia molecular 2026
  (9 clados, 21 especies validadas, 12 sinónimos, 12 candidatas — texto completo
  bloqueado por Wiley/Cloudflare, link dejado para que Alexis lo intente con acceso
  institucional) + ocurrencias reales con fecha vía API de GBIF (624 registros,
  pico estacional claro en primavera sept-nov, con el sesgo de muestreo de
  iNaturalist ya anotado).
- `gyriosomus_gbif_facets.csv` — nuevo, conteos crudos por mes y por año de GBIF.
- `desierto_florido_albedo_ndvi.md` ampliado: dos papers Frontiers de acceso abierto
  con fechas/sitios reales de floración (2014, 2015, 2017 — rizobacterias; 2021
  Caldera — color floral), y una lista compilada (calidad mixta, no todo peer-review)
  de años de floración 1983-2024. Chávez et al. 2019 (la Tabla 1 con métricas exactas
  de los 13 eventos satelitales) sigue bloqueado — link dejado para Alexis:
  https://doi.org/10.1016/j.jag.2018.11.013

## Actualización 2026-07-31 (tercera ronda — Alexis consiguió el paper de Gyriosomus)
- **Anguita-Salinas et al. 2026, texto completo obtenido** (Alexis, acceso
  institucional). Dato definitivo: **44 especies descritas, 9 clados, 21 validadas /
  12 sinónimos / 12 candidatas nuevas** según la revisión molecular — ya no hay que
  promediar cifras viejas. Hallazgo importante que HAY QUE PRESERVAR: la mayoría de
  sus campañas de muestreo fueron en años **La Niña** (con floraciones en parches), no
  El Niño — el mecanismo real es "pulso de lluvia bien distribuido → floración →
  escarabajos", con El Niño como vía común pero no única. Ver `gyriosomus_distribucion.md`.
- **Chávez et al. 2019, resumen público completo** (Alexis, sin poder comprar el PDF
  completo): resuelve la aparente contradicción anterior — 2011 es el evento más
  IMPORTANTE (11.136 km², 180 días), 2002-03 es el más LARGO (270 días), no son el
  mismo dato. Umbral de germinación: 15mm de lluvia acumulada en el año.
- **Nuevo hallazgo, vía búsqueda**: He et al. (2017), *Earth and Space Science* —
  mide DIRECTAMENTE el efecto de la floración sobre la temperatura superficial:
  enfriamiento neto de 0.29°C±0.07°C, pero se descompone en evapotranspiración
  (enfría, domina) vs. albedo (por sí solo calentaría, pero pierde). **Esto es central
  para el experimento**: confirma la caída de albedo pero obliga a no atribuirle a
  ella sola el efecto neto de temperatura. Bloqueado, solo tenemos el resumen.

## Actualización 2026-07-31 (cuarta ronda — He et al. 2017 completo, EL dato central)
Alexis consiguió también el texto completo de He et al. (2017), *Earth and Space
Science*. Es el hallazgo más importante de toda la investigación hasta ahora, porque
mide DIRECTAMENTE (no por inferencia) la hipótesis de Lovelock con MODIS real:
- **Albedo cae 2.1%±0.31%** en píxeles vegetados vs. desnudos (medido, no supuesto).
- Enfriamiento neto diario 0.29°C±0.07°C, con el albedo aportando el **41.2%** de la
  varianza explicada de día y el **92.5%** de noche (la evapotranspiración domina de
  día, pero el albedo no es menor).
- **Lección metodológica para nuestro propio experimento**: comparar espacialmente
  (vegetado vs. desnudo, mismo momento) da señal limpia; comparar temporalmente
  (mismo lugar, año con floración vs. sin ella) es mucho más ruidoso (0.13°C±0.38°C,
  la desviación estándar casi triplica el efecto).
- **Advertencia geográfica real**: la zona de estudio de He et al. (17.5-29.7°S) NO
  cubre la Zona de Alta Simpatría Cladística de Gyriosomus (30.5-31.5°S) — el vínculo
  con el punto caliente de diversidad de escarabajos sigue sin medirse directamente.
Detalle completo en `desierto_florido_albedo_ndvi.md`.

## Actualización 2026-07-31 (sexta ronda — formalización de hipótesis y modelo)
Se cierra la fase de Aprehensión y se abre diseño: `hipotesis_y_modelo_formal.md`
contiene la pregunta de investigación, 4 hipótesis formales con predicción y criterio
de falsación (H1 albedo/Lovelock, H2 pulso-de-lluvia vs. fase-ENSO, H3 profundidad
histórica sin ruptura antropogénica, H4 rezago ecológico de Gyriosomus), la
arquitectura de 4 módulos del modelo (A forzante, B floración, C albedo/temperatura,
D Gyriosomus) + capa de validación histórica, y los huecos de datos pendientes.

## Actualización 2026-07-31 (séptima ronda — primera pasada de H4)
`h4_rezago_gyriosomus.md` — primer intento de correr H4 con lo que ya hay, mientras se
espera la respuesta de CEAZA y las tablas suplementarias. Se consultó la API de GBIF
directo (no solo los facets guardados) para tener mes×año real, no solo marginales.
Resultado: **evidencia parcial a favor** (los dos únicos años con fecha de pico de
floración precisa — 2014 octubre, 2015 septiembre — muestran el pico de escarabajos
~1 mes después), pero con limitaciones honestas de tamaño de muestra y datos
incompletos — NO es un cierre de H4, queda "no falsada hasta ahora".

## Actualización 2026-07-31 (quinta ronda — contacto con CEAZA)
Se buscó un "piso duro" (medición en terreno) de albedo/radiación reflejada
específicamente en la Zona de Alta Simpatría Cladística de Gyriosomus (30.5-31.5°S,
corredor Vicuña-Ovalle-Combarbalá-Andacollo). Conclusión: **no existe publicado** — el
sitio de validación más austral conocido (Vicuña, del estudio de albedo MODIS) queda
justo en el borde norte de la zona, fuera de ella. CEAZA-Met sí tiene estaciones ahí
mismo, pero su instrumentación (piranómetro + pirgeómetro) mide radiación incidente
para prospección solar, no radiación reflejada — no hay evidencia de que midan albedo.

**Alexis López Tapia envió un correo a CEAZA (info@ceaza.cl, dirigido al Director
Ejecutivo Dr. Carlos Olavarría) el 2026-07-31**, preguntando: 1) si alguna estación mide radiación
reflejada/albedo real, 2) si habría interés en sumar esa instrumentación coincidiendo
con un evento de floración, 3) si existe algún dato histórico de referencia en esa
zona. **Queda pendiente la respuesta de CEAZA** — retomar cuando llegue.

## Qué ya tenía la carpeta Cosmoclima antes de esta ronda (contexto, no repetir)
- Lluvia mensual real 1900-2019, 879 estaciones (CR2).
- Estación Quinta Normal 1950-2026 con comparación lluvia real vs. NASA POWER
  (sesgo del satélite ya cuantificado, hasta -274 mm/año de subestimación).
- NASA POWER: script `datos_clima.py` para cualquier coordenada/fecha.
- Anillos de ciprés (El Asiento 1012-1972, San Gabriel 1132-1975) vía `archivo_clima.py`
  y `grafico_archivo.py` — proxy indirecto (índice, no mm) que ya cubre el rango
  medieval, con una brecha de ~6-9 años sin dato antes de 1981.

## Actualización 2026-07-31 (octava ronda — contacto directo con Marcelo Guerrero)
Marcelo Guerrero (coautor de Anguita-Salinas et al. 2026) es amigo de Alexis. Ya
conversaron y Guerrero va a enviar sus papers, imágenes y material asociado (a
confirmar si incluye los registros crudos de colecta por fecha/especie/localidad, lo
que resolvería de raíz la limitación de tamaño de muestra de
`h4_rezago_gyriosomus.md` y probablemente también sirva para H2 — Tabla S5
campaña-por-campaña Niño/Niña). Queda pendiente el envío, en paralelo a la respuesta
de CEAZA.

## Actualización 2026-07-31 (novena ronda — Tabla S1 conseguida, ya no está bloqueada)
Alexis encontró y descargó `investigacion/syen70011-sup-0001-supinfo.xlsx` — el material
suplementario completo de Anguita-Salinas et al. 2026 (9 hojas: TabS1-TabS8 + FigS1).
**TabS1 (153 especímenes, 41 especies, con Voucher/Localidad/Región/Lat/Long/Alt/Año)
ya no es un hueco** — se parseó completo (sin librerías externas, XML directo del
.xlsx) y se usó para reconstruir `Web/prueba_de_concepto/prueba_de_concepto_mapa_capas.html`
con datos reales por especie (ver esa sección más abajo). Hallazgo colateral valioso:
**38 de esos especímenes (16 especies) caen dentro de la ZHCS (30.5-31.5°S)** —
localidades reales: Ovalle, Socos, desembocadura del río Limarí, entre otras — esto
empieza a llenar el hueco de "nadie ha medido el vínculo directo con el punto caliente
de diversidad" que quedaba abierto desde la ronda de He et al. Pendiente: revisar
TabS5 (intensidad Niño/Niña por campaña, ya identificada antes) y TabS8 (clave
dicotómica) por si aportan algo más al modelo.

## Actualización 2026-07-31 (décima ronda — H5: proyección hacia adelante)
Alexis pidió agregar una hipótesis nueva, distinta a H1-H4: no explicar datos ya
cerrados, sino **proyectar hacia adelante** con el Niño 2026 en curso. `H5` en
`hipotesis_y_modelo_formal.md` — dos zonas candidatas (Huasco-Freirina y Ovalle-ZHCS),
con lluvia real de NASA POWER ya medida para 2026 (186,7 mm y 408,0 mm SOLO en julio,
muy sobre el umbral de 15 mm), especies candidatas por zona sacadas de la Tabla S1 real
(no inventadas), predicción de fechas (floración sept-oct 2026, escarabajos oct-dic
2026 según el rezago de H4), y **cierre temporal fijado al 28-feb-2027** — se revisa esa
fecha y se reporta lo que haya pasado, sin forzar. Datos de lluvia guardados en
`investigacion/fuentes/lluvia_2026_huasco_freirina_nasa_power.csv` y
`lluvia_2026_ovalle_zhcs_nasa_power.csv`.

## Actualización 2026-07-31 (undécima ronda — curva empírica lluvia→floración)
`curva_empirica_gyriosomus.md` — primera curva ajustada a datos reales (no inventada):
regresión logística de lluvia anual (NASA POWER, Huasco-Freirina) vs. floración
documentada, 19 años (13 con floración, 6 sin ella). Cruce de 50% en ~37mm/año, 79% de
exactitud en esos mismos 19 puntos (ajuste exploratorio, n chico, declarado así).
Datos en `fuentes/curva_empirica_lluvia_floracion_gyriosomus.csv`. Se probó cruzar
floración→Gyriosomus con los conteos anuales de GBIF también, pero no dio una relación
limpia — está confundido por el crecimiento del uso de iNaturalist en Chile, no es
señal ecológica real; queda documentado como intento honesto, no forzado. Esta curva
ya se usa en el motor del simulador (`Web/prueba_de_concepto/`).

## Lo que aún falta (próxima ronda de Aprehensión, no diseño)
1. Texto completo de Chávez et al. 2019 (Tabla 1 con métricas exactas de los 13
   eventos de floración) — paywalled, ver si hay acceso institucional.
2. Texto completo de Anguita-Salinas et al. 2026 (filogenia molecular de Gyriosomus,
   conteo definitivo de especies, tiempos de divergencia) — paywalled.
3. Serie de albedo/NDVI satelital (MODIS o Sentinel-2) recortada específicamente a la
   franja Vallenar-Huasco-Copiapó, no solo el Atacama central.
4. Registros de ocurrencia con fecha de Gyriosomus (GBIF/iNaturalist) para poder cruzar
   años de emergencia masiva con años de floración/lluvia — no existe en los papers de
   biogeografía revisados.
5. Catálogo de años de Desierto Florido documentado (más allá de lo satelital
   1981-2015) — prensa/CONAF/informes de turismo para el registro 2016-2026, incluido
   el evento actual.
