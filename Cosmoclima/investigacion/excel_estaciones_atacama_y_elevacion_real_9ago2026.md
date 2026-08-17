# Estaciones Excel Atacama (17 nuevas) + catálogo de elevación real — 09-ago-2026

Alexis mandó 17 archivos Excel (`datos/Excel Estaciones/`), formato "Informe
Anual de Agua Caída" (DMC/Fundación para el Desarrollo Frutícola/INIA),
lluvia diaria real 2019–agosto 2026 para estaciones de la Región de
Atacama. Pidió sumar todo: los datos de lluvia al sqlite consolidado, y las
alturas reales al catálogo de elevación (que quedó pendiente de la sesión
de esta misma noche, tras notar que Vicuña sale más fría que La Serena en
el gradiente latitudinal — hipótesis: elevación real).

## Verificación antes de tocar la base de datos

- **Backup**: `investigacion/fuentes/pluviosidad_diaria_consolidada.sqlite.bak_09ago2026_pre_excel_atacama`
  (copia completa, 812MB, antes de cualquier escritura).
- **Convención `s/p`/`.` verificada, no asumida**: crucé "Total Mensual"
  contra "N° de días con Agua≥0.1" para un mes con huecos reales — confirma
  `s/p` = CERO real (día observado, sin lluvia) y `.` = SIN DATO (falta
  observación). Misma norma que `LLUVIA_DIARIA_1966_2017`.
- **Metadata de estación real** (Código Nacional, Nombre, Propietario,
  Latitud, Longitud, **Altura(Mts.)**) viene en cada archivo — es la misma
  fuente que el catastro DMC/DGA que ya había cruzado por web, pero local y
  completa, sin necesidad de más scraping.
- **Un typo real detectado y corregido** (documentado, no oculto): hoja
  `"20219"` en `ALTO.DEL.CARMEN-ATACAMA.xlsx` = 2019 (un dígito de más).
- **Redundancia real detectada y evitada**: el archivo de Caldera trae
  además hojas `JULIO2026`/`JUNIO2026`/`MAYO2026` con OTRO layout (reporte
  por horas UTC) que duplican lo que ya está en la grilla año-a-la-fecha
  `AGOSTO-2026` — se ignoraron a propósito para no insertar el mismo dato
  dos veces.

## Resultado

`agregar_excel_estaciones_atacama_2026_a_consolidada.py` — inserción
idempotente (DELETE + INSERT por localidad y rango de fechas, mismo patrón
que `agregar_estaciones_julio2026_a_consolidada.py`, el precedente directo).
**42.423 filas reales nuevas**, 18 series (17 archivos, uno con dos
estaciones — Alto del Carmen y Alto del Carmen Los Sauces son archivos
separados). Total tabla consolidada: **4.519.702 filas**.

**Verificado con 4 valores exactos** cruzados a mano contra la celda cruda
del Excel (Copiapó 2024-08-02=1.2mm, 2024-05-07=16.3mm, 2024-04-13=11.6mm,
2024-11-26=8.4mm) — los 4 coinciden exactos en el sqlite.

### Solapes conocidos, no fusionados a propósito

3 de las 17 estaciones coinciden geográficamente (<2km) con una localidad
YA EXISTENTE de registro histórico largo, bajo OTRO nombre de localidad
(fuente distinta, ingestada en otro momento). Se insertaron como series
separadas — fusionar dos series de fuente/linaje distinto sin revisar valor
a valor no fue pedido, y mezclarlas a ciegas podría introducir un problema
peor que dejarlas declaradas aparte:

- `Desierto de Atacama, Caldera (DMC)` [2005–2025, 7375 filas] vs.
  `Desierto de Atacama, Caldera Ad. (DMC/INIA 270008)` [2005–2026, este
  script, 5151 filas — sí, este Excel también traía histórico desde 2005].
- `Copiapo` [1971–2018] vs. `Copiapo (DMC/INIA 270016)` [2019–2026].
- `Copiapó Universidad de Atacama (DMC/INIA 270009)` ya existía con solo
  julio-2026 (31 filas, de una carga anterior) — este script la extendió a
  2019–2026 completo (2644 filas), incluyendo esos mismos 31 días de julio
  (reemplazados, no duplicados — mismo rango, delete+insert).

## Catálogo de elevación real — `estaciones_elevacion_real.csv`

Nuevo archivo persistente, `investigacion/fuentes/estaciones_elevacion_real.csv`
— consolida TODA la elevación real (no satelital) encontrada hoy, de tres
fuentes distintas, con código de estación para trazabilidad:

1. 12 estaciones del gradiente latitudinal original, vía
   `cr2_prDaily_2018_stations.txt`/`cr2_prAmon_2019_stations.txt` (archivos
   ya en el proyecto, nunca antes consultados para esto).
2. 17 estaciones nuevas de Atacama, vía la hoja de metadata de cada Excel.
3. 3 estaciones de respaldo encontradas por web (catastro DMC,
   `climatologia.meteochile.gob.cl`) para casos donde el archivo local no
   alcanzaba.

**Aplicado a `GRADIENTE_LATITUDINAL_TEMP`** (`obtener_gradiente_latitudinal.py`,
tercera ronda): cada una de las 14 estaciones del gradiente ahora lleva
`altura_m` real (`null` para Paposo y El Guindal, que — hallazgo aparte —
resultaron ser puntos de reanálisis ERA5-Land, no estaciones físicas reales,
confirmado en el propio sqlite). Tooltip del gráfico actualizado para
mostrarla. **Confirma la hipótesis de Alexis**: Vicuña real está a 730m
(no los 2185m que daba NASA POWER) — consistente con que salga más fría
que La Serena (15m) en máxima pese a estar catalogada "valle".

## Archivos nuevos/modificados

- `agregar_excel_estaciones_atacama_2026_a_consolidada.py` (nuevo, raíz del proyecto)
- `investigacion/fuentes/pluviosidad_diaria_estaciones_atacama_excel_2019_2026.csv` (nuevo, trazabilidad)
- `investigacion/fuentes/estaciones_elevacion_real.csv` (nuevo, catálogo persistente)
- `investigacion/fuentes/pluviosidad_diaria_consolidada.sqlite.bak_09ago2026_pre_excel_atacama` (backup, se puede borrar cuando Alexis confirme que todo está bien)
- `Web/prueba_de_concepto/obtener_gradiente_latitudinal.py` (altura_m real agregado)
- `Web/prueba_de_concepto/prueba_de_concepto_ET3-Termico_con_mapa.html` (`GRADIENTE_LATITUDINAL_TEMP` con altura_m, tooltip actualizado)

## Pendiente / no hecho todavía

La pregunta original de Alexis (recalcular la emergencia de vegetación con
4-5 parámetros: Pluviosidad, Albedo real (NDVI satelital, no el del
modelo), Temperatura Máx/Mín, Elevación, cruzados por Costa/Valle) sigue
sin implementarse — esta sesión fue investigación/consolidación de datos
(temperatura, gradiente, elevación real), no la recalibración en sí.
