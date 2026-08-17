# Datos nuevos integrados: Anuario DMC 2025 + reporte estaciones julio 2026

Fecha: 05-ago-2026. A pedido de Alexis: revisar `datos/anuario-2025.pdf` (Anuario
Climatológico 2025 completo de la DMC, 275 páginas) y
`datos/LAS ESTACIONES-PRECIPITACION.xlsx` (reporte diario DMC/INIA de julio 2026,
4 estaciones), e integrar lo que fuera dato nuevo a la base que usa la simulación.

## 1. Julio 2026 (Excel) — SÍ entró a la tabla diaria consolidada

`agregar_estaciones_julio2026_a_consolidada.py` parseó las 4 hojas del Excel (columna
"Suma Diaria" de precipitación cada 6 horas) e insertó en
`fuentes/pluviosidad_diaria_consolidada.sqlite` (misma tabla que usa el simulador):

- **Los Acacios** (300059, Coquimbo, INIA) — estación NUEVA en la base, 31 días reales.
- **Desierto de Atacama, Caldera Ad.** (270008) — YA existía vía API DMC pero solo
  hasta 2025-12-31; esto la **extiende** con 31 días reales de julio 2026.
- **Copiapó Universidad de Atacama** (270009) — estación NUEVA, 30 días reales
  (falta el día 31, sin dato en el reporte).
- **Copiapó, Fundación para el Desarrollo Frutícola** (270016) — estación NUEVA,
  31 días reales.

**123 filas nuevas**, tabla consolidada pasó de 4.477.248 a 4.477.371 filas.

Reglas de lectura: `s/p` en la columna Diaria = sin precipitación real (0.0 mm,
confirmado, no ausencia de dato); `.` = sin observación ese día, se descarta.

**Bug real encontrado y corregido en el camino**: la hoja de Caldera trae el mes
truncado en el archivo de origen — dice literalmente `"Julio de 202"` (falta el último
dígito), un defecto del propio reporte DMC, no un error de lectura. Se corrigió por
consenso con las otras 3 hojas del mismo reporte (mismo día de descarga, mismo mes,
todas dicen 2026) — el script imprime un aviso explícito cuando hace esta corrección,
no la aplica en silencio.

CSV de respaldo/trazabilidad:
`fuentes/pluviosidad_diaria_estaciones_julio2026.csv` (123 filas).

## 2. Anuario 2025 (PDF, 275 páginas) — mensual, en archivo SEPARADO (no en la tabla diaria)

El Anuario trae dos tipos de tabla, ambas por MES (no por día) — mezclarlas con la
tabla diaria habría sido deshonesto con la granularidad real del dato, así que quedan
en `fuentes/precipitacion_mensual_dmc_anuario2025.csv` (columnas: estacion, codigo,
lat, lon, alt_m, mes, anio, lluvia_mm, dato_real, fuente).

**Barrido completo del PDF**: se revisaron las 48 estaciones climatológicas completas
(páginas 8-248) y las ~130 estaciones del apéndice de pluviometría (páginas 253-256,
venían como imagen escaneada, no texto — se renderizaron y leyeron visualmente). Solo
quedaron dentro del rango de Gyriosomus (25°S-34°S) las que siguen.

### 2a. Estaciones climatológicas completas — 2 son NUEVAS y caen dentro de la ZHCS
- **Ovalle, Escuela Agrícola** (-30.579999, -71.186666, 310 m) — **dentro de la Zona de
  Alta Simpatría Cladística (30.5-31.5°S)**. Total anual 2025: **95.1 mm**, máximo en
  24h 34.5 mm (julio).
- **Vivero Conaf, Illapel** (-31.632499, -71.151944, 333 m) — también dentro de la
  ZHCS. Sin total anual publicado (noviembre sin dato); suma de meses con dato real:
  ~118.8 mm.
- Copiapó Universidad de Atacama (270009) también viene en esta sección — se agregó
  igual aunque ya había quedado cubierta por el Excel (julio 2026); esto suma los
  otros 11 meses de 2025 que antes no estaban.

### 2b. Apéndice de pluviometría — 9 estaciones con dato real, 2 sin información
Coordenadas verificadas una por una contra la ficha oficial de cada estación en
`climatologia.meteochile.gob.cl` (no se asumieron por nombre parecido — un nombre
candidato inicial, "Canela", resultó ser una estación distinta a "La Canela Fundo"
una vez verificado, en Puchuncaví y no en Choapa; se descartó la asunción):

Los Nichos (300006), Combarbalá Essco (310003), Puerto Oscuro (310004, coincide con
la coordenada que ya usaba el proyecto para esta localidad), Chuchiñi (310019),
El Trapiche Longotoma (320006, OJO: NO es el mismo "El Trapiche" que ya existía en
`pluviosidad_diaria_gyriosomus_openmeteo_resumen.csv` a -29.34/-71.15 — ese es un
lugar homónimo cerca de Vallenar, éste está en La Ligua/Longotoma), Casas de Alicahue
(320007, coordenada DMC oficial algo más precisa que la que ya teníamos), La Ligua
Esval (320012), Catapilco Hacienda (320015), La Canela Fundo (320018, Puchuncaví,
borde sur del rango).

**Sin información en 2025** (quedaron fuera, no se inventó nada): Huaquén Hacienda
(320005), Curimón Escuela Agrícola (320024).

## Por qué importa para el experimento
Puerto Oscuro y Alicahue eran justamente 2 de las 15 localidades que habían quedado
**pendientes** en `pluviosidad_diaria_gyriosomus_openmeteo.csv` por el límite de tasa
de Open-Meteo (ver `pluviosidad_diaria_consolidada.md`) — ahora tienen dato real
DMC de 2025, aunque a nivel mensual, no diario. Ovalle e Illapel son estaciones
climatológicas completas justo dentro de la ZHCS, algo que antes solo se cubría con
NASA POWER (reanálisis) o CR2 (que en varias de estas estaciones ya no reporta).

## Pendiente
- El apéndice de pluviometría del Anuario 2025 no trae lat/lon — se completaron todas
  vía la ficha oficial de cada código DMC, pero si en algún momento se agrega otro año
  de Anuario, conviene guardar el mismo cruce código→coordenada acá para no rehacerlo.
- No se recalculó ningún total anual que el propio Anuario dejó sin publicar (ej.
  Illapel) — si se necesita, sumar solo los meses marcados `dato_real=si`, nunca tratar
  un mes `no` como 0.
