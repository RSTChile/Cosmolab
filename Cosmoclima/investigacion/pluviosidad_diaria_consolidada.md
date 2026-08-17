# Pluviosidad diaria consolidada, toda la zona de distribución de Gyriosomus

Fecha: 01-ago-2026. A pedido de Alexis: reunir pluviosidad DIARIA real, no solo para
la ZHCS, sino para varias localidades a lo largo de todo el rango del género
*Gyriosomus* (~24.8°S a 34.2°S), consolidada en una tabla única.

## Lo que se investigó (4 agentes en paralelo)
- **CR2 diario** (`cr2_prDaily_2018`): ~816 estaciones reales, desde 1900, mismo
  formato que el mensual ya usado. **Requiere crear una cuenta en cr2.cl** (portal
  WP Download Manager con muro de login) — verificado en vivo: la descarga directa
  por URL da 403/redirige a una página que exige login. No se puede automatizar sin
  que una persona se registre.
- **DMC** (`getAguaCaidaDiaria`, climatologia.meteochile.gob.cl): API JSON real y
  documentada, con estaciones confirmadas dentro del rango (La Serena desde 1954,
  Caldera desde 2005). **Requiere registrar un usuario + generar un token personal**
  (`/application/usuario/registroUsuario`).
- **GHCN-Daily (NOAA)**: solo 2 estaciones útiles en todo el corredor
  Atacama-Coquimbo (La Serena 1964-2025 activa; Copiapó/Chamonate 1968-2004,
  dejó de reportar). Cobertura demasiado escasa para "toda la zona".
- **Open-Meteo** (hallazgo del cuarto agente): API REST gratuita, **sin registro ni
  llave**, corre sobre ERA5-Land (reanálisis, ~0.1°/9km), historia desde 1966+
  (probado en vivo), cualquier lat/lon.

## Por qué CR2-diario y DMC quedaron fuera de la tabla automática
Ambas son las fuentes de mejor calidad (estación real, no reanálisis) pero **piden
que una PERSONA se registre** — no es algo que se pueda hacer en nombre de Alexis sin
su participación directa (crear cuentas de terceros no es algo que se automatice sin
que el dueño de los datos lo autorice y lo haga él mismo). Quedan como **pendiente
real, no descartado**:
- Si Alexis quiere sumarlas después, el camino es: él se registra en cr2.cl y/o pide
  el token de la DMC (ambos son gratis, solo piden un formulario), y con eso sí se
  puede automatizar la descarga y sumarla a esta misma tabla.

## Lo que sí se armó: Open-Meteo, todas las localidades reales de la Tabla S1
Script: `obtener_pluviosidad_diaria_openmeteo.py` (en la raíz de `Cosmoclima/`).
Toma las **78 localidades reales distintas** de la Tabla S1 (Anguita-Salinas et al.
2026 — las mismas coordenadas ya usadas en el mapa de especies), consulta
pluviosidad diaria real (Open-Meteo/ERA5-Land) para cada una, desde 1966 hasta hoy
(mismo rango que ya usa el instrumento), y las junta en una sola tabla:

`investigacion/fuentes/pluviosidad_diaria_gyriosomus_openmeteo.csv`
columnas: `fecha, localidad, lat, lon, lluvia_mm, fuente`

`investigacion/fuentes/pluviosidad_diaria_gyriosomus_openmeteo_resumen.csv`
(una fila por localidad: cuántos días se pidieron, cuántos vinieron con dato real)

**Honestidad sobre qué tipo de dato es**: Open-Meteo/ERA5-Land es reanálisis
(satélite + modelo), NO estación en tierra — misma naturaleza que NASA POWER, ya
usado en este proyecto. Es la única forma de tener cobertura DIARIA en las 78
localidades sin depender de que existan estaciones físicas ahí (que no existen para
la enorme mayoría de esos puntos). Para las pocas localidades cercanas a estaciones
reales (Caldera, Freirina/Huasco, La Serena/Ovalle), en el futuro conviene
contrastar contra CR2-diario o DMC cuando esas fuentes se sumen.

## Resultado final (01-ago-2026)
**63 de 78 localidades reales** consiguieron su serie diaria completa (1966 a hoy,
~22.128 días cada una, sin huecos — ERA5-Land no tiene meses/días faltantes como sí
tienen las estaciones reales). Total: **1.394.064 filas** en
`pluviosidad_diaria_gyriosomus_openmeteo.csv`.

**15 localidades quedaron pendientes** — no por falta de dato, sino porque la API
gratuita de Open-Meteo (sin cuenta/llave) tiene un límite de solicitudes que se
activó durante la descarga masiva (`HTTP 429 Too Many Requests`), y con pausas de
hasta 75s entre reintentos igual no cedió para estas 15 puntuales:

Soruco, Cuesta el Espino, Los Pozos, Canela, Puerto Oscuro, Peaje Canela, Ruta 5
Norte Km268, Quendaño, Mincha Sur, Cuesta Cavilolen, Los Vilos, Los Molles, Alicahue,
Pullay, El Guindal — casi todas agrupadas entre 31.1°S y 32.4°S (zona Canela/Los
Vilos), más el punto más austral del rango (El Guindal, 34.2°S).

**Cobertura lograda**: 24.85°S a 33.59°S de las 63 exitosas — cubre la enorme mayoría
del rango real del género, con un hueco real entre ~31°S y ~32.4°S. El script
`reintentar_pluviosidad_diaria.py` puede volver a correrse más tarde (la cuota de
Open-Meteo se resetea) para completar esas 15 sin tener que rehacer nada de lo ya
conseguido — solo reintenta lo que sigue marcado "ERROR" en el resumen.

## ★ CR2 diario + DMC sumados (01-ago-2026, mismo día) — Alexis se registró él mismo
Alexis se registró en cr2.cl y bajó `cr2_prDaily_2018` completo (874 estaciones,
diario real, enero 1900 a marzo 2018, 227MB) a
`Cosmoclima/cr2_prDaily_2018/`. Se extrajeron las **262 estaciones reales** (de 271
candidatas por latitud) dentro de 25-34°S con
`extraer_cr2_diario_zona_gyriosomus.py` — **2.765.499 días reales de estación**
(bug real encontrado y corregido en el camino: el archivo de estaciones usa códigos
SIN ceros a la izquierda, ej. "1000005", pero el archivo de datos SÍ los rellena a 8
dígitos, ej. "01000005" — sin normalizar esto solo calzaban 9 de 271 estaciones por
azar).

Alexis también se registró en la API de la DMC y compartió usuario+token (avisado en
el momento: no hace falta la contraseña para esta API, solo correo+token — no se usó
ni se guardó en ningún archivo). Se detectó que **Caldera y La Serena** — las 2
únicas estaciones DMC dentro del corredor real de Gyriosomus — estaban en la lista
de CR2 pero con **0 días de dato real** (columnas vacías en la copia de CR2, aunque
la estación existe). Se pidieron directo a la DMC (`getAguaCaidaDiaria`, campo
`total`, que exige que estén disponibles TODAS las observaciones del día — más
estricto que `parcial`) y llenaron ese hueco: **18.698 días reales** más
(Caldera 2005-2025, La Serena 1995-2026).

### La tabla consolidada final
En vez de pegar los CSV de cada fuente en un archivo único de texto (habrían sumado
~370MB planos), se armó **una base SQLite** — literalmente "una tabla única que se
puede consultar", con índices por fecha, localidad y lat/lon:

`investigacion/fuentes/pluviosidad_diaria_consolidada.sqlite`
tabla `pluviosidad_diaria` — columnas: `fecha, localidad, lat, lon, lluvia_mm, fuente, tipo_fuente`

**4.178.261 filas totales, 315 localidades/estaciones distintas, 1900-02-01 a
2026-08-01**:
- `estacion_real`: 2.784.197 días (261 localidades) — CR2 (262 estaciones) + DMC
  (Caldera y La Serena, llenando el hueco de CR2).
- `reanalisis_era5land`: 1.394.064 días, 63 localidades reales por coordenada exacta
  (56 nombres distintos de texto — unas pocas localidades de la Tabla S1 repiten
  nombre en puntos de colecta cercanos pero no idénticos, ej. "Carrizalillo" o "La
  Higuera" aparecen 2-3 veces con coordenadas ligeramente distintas; el dato en sí
  está completo y correcto por lat/lon, es solo un detalle de cómo se cuenta "por
  nombre" vs "por coordenada") — Open-Meteo/ERA5-Land.

Los CSV de cada fuente (`pluviosidad_diaria_cr2_estaciones_reales.csv`,
`pluviosidad_diaria_dmc_estaciones_reales.csv`,
`pluviosidad_diaria_gyriosomus_openmeteo.csv`) quedan intactos como respaldo/
trazabilidad de cada fuente por separado.

**Nota de seguridad**: ni la contraseña ni el token de la DMC se escribieron en
ningún script ni archivo del proyecto — los scripts los leen de variables de entorno
(`DMC_USUARIO`, `DMC_TOKEN`) que hay que pasar a mano cada vez que se necesite volver
a consultar la API.
