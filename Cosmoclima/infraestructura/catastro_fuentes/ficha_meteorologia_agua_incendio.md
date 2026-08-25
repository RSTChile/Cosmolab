# Catastro de fuentes — METEOROLOGÍA, AGUA E INCENDIO

Proyecto «Infraestructura Crítica × Clima» (RMD 2.0) · Dominio: DMC, DGA/MOP, CONAF, CR2 y
fuentes globales con cobertura Chile.

**Fecha de verificación: 15–16 de agosto de 2026.** Todo lo marcado **VERIFICADO** se probó
con una petición real ese día y se copió la respuesta. Lo marcado **SUPUESTO** no se probó o
se probó y falló.

---

## RESUMEN EJECUTIVO — las dos preguntas grandes

### 1. ¿La DMC tiene API oficial? **SÍ.** Y no hay que parsear HTML.

`https://climatologia.meteochile.gob.cl/` publica **27 web-services oficiales** (17 JSON + 10
GeoJSON), cada uno con su ficha de documentación, su identificador y ejemplos de consumo con
`wget` y `php`. Es decir: **la DMC no sólo permite el uso automatizado, lo documenta paso a
paso.** La documentación dice que hay que pasar `?usuario=…&token=…` (registro gratuito), pero
**probamos sin credenciales y responde igual con el dato completo**. Recomendación: registrarse
igual, porque es la puerta formal y porque el día que cierren el acceso anónimo no queremos
enterarnos en producción.

*Analogía:* es como un herbario que además de dejarte ver las cajas te entrega la pinza, el
guante y un instructivo de cómo sacar los ejemplares sin romperlos.

### 2. ¿Existe la capa de zonas geográficas (Litoral, Pampa, Precordillera…)? **SÍ, y la tenemos localizada.**

`https://archivos.meteochile.gob.cl/portaldmc/AAA/sistemaAAA_coordenadas_zonas_2026.js`
— **584 KB, 183 polígonos, 11.421 vértices**, cubriendo Chile completo. Contiene literalmente
`02_Precordillera_Occidental`, `02_Precordillera_Alto_Loa`, `02_Precordillera_Salar` — las
mismas zonas exactas que cita la alerta de Antofagasta en `INTEGRACION_SENAPRED.md`. Es el
polígono oficial que la DMC dibuja en su propio mapa de alertas.

**Pero hay una piedra en el camino, y es seria:** ese archivo vive en el host
`archivos.meteochile.gob.cl`, cuyo `robots.txt` dice `User-agent: * / Disallow: /` — o sea,
**prohíbe el rastreo automatizado de todo el host**. Ver ficha 03 para el detalle y la salida
propuesta (pedir autorización escrita / convenio a la DMC).

---

# BLOQUE A — METEOROLOGÍA (DMC)

---

## FICHA 01 · DMC — Servicios Climáticos (API JSON/GeoJSON)

**1. Organismo y producto**
Dirección Meteorológica de Chile (DMC, dependiente de la DGAC) — Portal «Servicios Climáticos»,
sección *Recursos JSON & GeoJSON de Datos*. Versión del portal: 3.14.0.

**2. URL exacta y endpoint**
- Índice de servicios: `https://climatologia.meteochile.gob.cl/application/index/menuTematicoJson`
- Documentación de cada servicio: `https://climatologia.meteochile.gob.cl/application/documentacion/getDocumento/{0..24}`
- Patrón de endpoint JSON:
  `https://climatologia.meteochile.gob.cl/application/servicios/{idServicio}?usuario=correo@correo.cl&token=apiKey_personal`
- Patrón de endpoint GeoJSON:
  `https://climatologia.meteochile.gob.cl/application/geoservicios/{idServicio}?usuario=…&token=…`
- Registro de usuario: `https://climatologia.meteochile.gob.cl/application/usuario/registroUsuario`
- Contacto: `portalserviciosclimaticos@meteochile.gob.cl`

Servicios relevantes para nosotros (identificadores verificados en la documentación oficial):

| Identificador | Qué entrega |
|---|---|
| `getEstacionesRedEma` | Catastro de estaciones EMA (lat/lon/altura/región/**zonaGeografica**) |
| `getEmaResumenDiario` | Resumen diario por estación (Tmáx, Tmín, agua caída), admite `/codigoNacional` |
| `getRiesgoIncendio` | GeoJSON de riesgo de incendio forestal (ficha 05) |
| — | Datos 12 h minutarios, 15-min mensuales, históricos diarios de T/PP/presión (30 años), WRF |

**3. Qué publica (variables, unidades)**
Temperatura del aire a 1,5 / 2 / 10 / 30 m (°C), punto de rocío (°C), máximas y mínimas de 15
min y de 12 h, humedad relativa (%), dirección del viento (grados) e intensidad (kt y km/h),
agua caída (mm), presión a nivel del mar (hPa), radiación UVB (índice), y salidas del modelo
WRF-DMC. Cada estación trae `codigoNacional`, `codigoOMM`, `codigoOACI`, lat/lon, altura,
**provincia**, **comuna**, región y `zonaGeografica`.

**4. Familia:** AMENAZA (variables meteorológicas que alimentan el peligro) + BASE TERRITORIAL
(el catastro de estaciones trae la adscripción comuna/provincia/zona).

**5. Formato:** JSON y GeoJSON (`FeatureCollection`, geometría `Point`, CRS declarado
`urn:ogc:def:crs:OGC:1.3:CRS8`).

**6. Acceso: ANÓNIMO — VERIFICADO.**
`GET https://climatologia.meteochile.gob.cl/application/servicios/getEstacionesRedEma` **sin
token** devolvió HTTP 200 con 148 estaciones y el dato completo (16-ago-2026, 01:30 UTC). La
documentación sí pide `usuario` + `token` (API key personal, registro gratuito), así que el
acceso anónimo puede ser una puerta que hoy está abierta y mañana no. **Registrarse igual.**

**7. Granularidad territorial:** puntual (estación), con etiqueta de comuna, provincia, región
y zona geográfica. 148 estaciones EMA activas al día de la verificación.

**8. Frecuencia:** de minutal a diaria según el servicio. El resumen diario y los boletines,
1×/día; los datos EMA, cada 1–15 minutos.

**9. Historia:** hasta 30 años en los servicios «históricos» de precipitación, temperatura y
presión diaria; mensual y anual completos para las series largas.

**10. Condiciones de uso — VERIFICADO.**
- `robots.txt`: **no existe** en este host (HTTP 404). No hay prohibición de rastreo.
- Nota oficial al pie del portal: *«Todos los datos e información publicados en este Portal son
  de acceso y uso público; no obstante, se solicita citar a la Dirección Meteorológica de Chile
  como la fuente de la información.»*
- La propia documentación entrega ejemplos con `wget` y `curl`/PHP.
→ **Uso automatizado permitido y explícitamente previsto.** Obligación: citar a la DMC.

**11. Para qué sirve en el modelo**
Es la fuente de observación meteorológica de base del consolidador: alimenta las variables
climáticas del `C_clim` y sirve de verificación *a posteriori* de los umbrales que la DMC
anuncia en sus avisos (ficha 02). También aporta la traducción estación→comuna→zona.

**12. Utilidad: ★★** — no cambia por sí sola la prioridad de un activo (una temperatura no es
una amenaza), pero es el sustrato que permite calibrar y auditar lo que sí la cambia.

**13. Riesgos y límites**
- La cobertura es de **puntos**, no de superficie: 148 estaciones para 4.300 km de país. Un
  activo puede estar a 80 km de la estación más cercana. No interpolar en silencio.
- El acceso anónimo no está garantizado por escrito. Riesgo de corte sin aviso.
- La red es de la DGAC: sesgada a aeropuertos y zonas bajas; la cordillera está sub-muestreada,
  y es justo donde caen los aluviones.

---

## FICHA 02 · DMC — Sistema de Alerta Meteorológica (Avisos / Alertas / Alarmas) ★ LA JOYA

**1. Organismo y producto**
DMC — Sistema AAA (Aviso · Alerta · Alarma). Son **tres** escalones, no dos:
- **Aviso**: fenómeno de severidad *moderada*, potencialmente riesgoso.
- **Alerta**: severidad *fuerte*, con probabilidad de generar riesgo a las personas.
- **Alarma**: severidad *intensa*, potencial muy alto de daño material y a la vida.

**2. URL exacta y endpoint**
- **Feed estructurado de todo lo vigente** (el hallazgo importante):
  `https://archivos.meteochile.gob.cl/portaldmc/AAA/datos_AAA.js`
- Mapa oficial (la página que lo consume): `https://archivos.meteochile.gob.cl/portaldmc/AAA/aaa_mapa.php`
- Tabla/listado: `https://archivos.meteochile.gob.cl/portaldmc/AAA/aaa_tabla.php`
- Documento vivo de cada evento: `https://archivos.meteochile.gob.cl/portaldmc/AAA/doc/evento_{CODIGO}_{AÑO}.php`
  (el `/` del código se reemplaza por `_`: `A413/2026` → `evento_A413_2026.php` — **verificado**).

**3. Qué publica**
`datos_AAA.js` es un archivo JavaScript con contadores globales y un arreglo `AAA[]`. Al momento
de la verificación (actualizado 2026-08-15 18:51): **11 Avisos, 3 Alertas, 0 Alarmas**. Cada
entrada trae:

| Campo | Contenido |
|---|---|
| `id` | Identificador interno (ej. 5464) |
| `tipo` | `Aviso` / `Alerta` / `Alarma` |
| `codigoMeteo` | Identificador citable y **versionado**: `A413/2026`, `A419-1/2026`, `AA137/2026` |
| `emision` | Fecha y hora de emisión |
| `fenomeno` | Precipitaciones · Viento · Nevadas · Tormentas Eléctricas |
| `condicionSinoptica` | Sistema frontal, inestabilidad atmosférica, etc. |
| `desde` / `hasta` | **Vigencia temporal explícita** en lenguaje natural |
| `tipoZonaAfecta` | `zonas` / `regiones` / `area` |
| `dataZonaAfecta` | Códigos de zona: `04_Precordillera,04_Cordillera` |
| `textoZonaAfecta` | Versión legible: «Coquimbo (Precordillera y valles precordilleranos, Cordillera)» |
| `tablaHTML` | **Los umbrales cuantitativos por zona y por día** (ej. 20-30 mm; 20-30 CM de nieve) |
| `titulo` | Título del evento |

Fenómenos observados en la muestra: Precipitaciones, Nevadas, «Nevadas en otras áreas»,
Tormentas Eléctricas, Viento.

**4. Familia: AMENAZA.** Es el disparador operativo de toda la cadena de SENAPRED.

**5. Formato:** JavaScript con arreglo de objetos (JS, no JSON puro — se convierte a JSON con
un parseo trivial y estable). Los documentos de evento son HTML.

**6. Acceso: ANÓNIMO — VERIFICADO.** HTTP 200, 21.619 bytes, sin credenciales, 16-ago-2026.
El documento histórico `evento_AA130_2026.php` (alerta del 3-ago, ya no vigente) **sigue
respondiendo** → hay archivo por URL.

**7. Granularidad territorial:** **zona geográfica DMC** (no administrativa). Algunas
declaraciones son por región completa (`04,05,05m,06,07,08a,08b`) y otras por punto/área
(Isla de Pascua con un círculo de radio).

**8. Frecuencia:** continua. Es un documento vivo: se actualiza y se versiona con sufijo
(`A419-1`, `A414-3`, `A388-8` = octava actualización del aviso 388).

**9. Historia:** ⚠️ **No encontramos índice ni archivo histórico navegable.** Los documentos
individuales persisten por URL si uno conoce el código, pero no hay listado de eventos pasados.
Consecuencia práctica: **si queremos historia, hay que empezar a guardarla nosotros desde hoy**,
capturando `datos_AAA.js` cada N horas. Cada día que pasa sin capturar es un día perdido para
siempre.

**10. Condiciones de uso — ⚠️ VERIFICADO Y PROBLEMÁTICO.**
`https://archivos.meteochile.gob.cl/robots.txt` devuelve:
```
User-agent: *
Disallow: /
```
**El host completo prohíbe el rastreo automatizado.** El dato es público (cualquiera lo ve en el
navegador), pero *dato público ≠ uso automatizado permitido*, y acá la señal es explícita.

**Salida propuesta, en este orden:**
1. Pedir a la DMC autorización escrita / convenio para consumir `datos_AAA.js` a baja frecuencia
   (contacto: `portalserviciosclimaticos@meteochile.gob.cl`). Es una petición razonable: el
   destinatario final es SENAPRED, no un negocio.
2. Preguntarles si existe una salida **CAP (Common Alerting Protocol)**. Buscamos y **no
   encontramos evidencia de que la DMC publique CAP** (ver ficha 04). La OMM empuja CAP para
   todos los servicios meteorológicos nacionales: es probable que exista un proyecto interno o
   una salida no publicada. Vale la pregunta directa.
3. Mientras tanto: **no automatizar**. Captura manual o pausada, y anotarlo.

**11. Para qué sirve en el modelo**
Es *la* entrada del `FEN` dinámico meteorológico. Da las tres cosas que
`INTEGRACION_SENAPRED.md` exige de la salida: identificador estable y versionado, vigencia
temporal explícita y adscripción a zona geográfica. Además, `tablaHTML` da el **umbral
cuantitativo por zona y por día**, que es exactamente lo que necesita el coeficiente para no
inventarse magnitudes.

**12. Utilidad: ★★★** — sin discusión. Un aviso de 90-110 km/h de viento en la Cordillera de
Atacama cambia hoy, y por tres días, la prioridad de toda subestación que caiga en ese polígono.
Es la fuente que más mueve la aguja de todo el dominio.

**13. Riesgos y límites**
- **El `robots.txt` es el riesgo número uno.** No es técnico, es de legitimidad.
- Formato JS no versionado como API: la DMC puede cambiar la estructura sin avisar a nadie,
  porque para ellos es un archivo interno de su mapa, no un contrato público. El adaptador debe
  degradar a «sin dato» y gritar, nunca adivinar.
- Fechas en **lenguaje natural** («la mañana del sábado 15 de agosto del 2026»): hay que
  parsearlas, y ese parseo es frágil. Los umbrales vienen dentro de HTML embebido en el JS.
- El texto viene con entidades HTML (`&aacute;`, `&ntilde;`) — decodificar antes de comparar.

---

## FICHA 03 · DMC — Capa de zonas geográficas del Sistema AAA ★ LA PIEZA QUE FALTABA

**1. Organismo y producto**
DMC — polígonos oficiales de las zonas geográficas usadas para declarar avisos y alertas.
Resuelve el «problema de las dos geografías» (sección 4 de `INTEGRACION_SENAPRED.md`).

**2. URL exacta y endpoint**
- **Vigente:** `https://archivos.meteochile.gob.cl/portaldmc/AAA/sistemaAAA_coordenadas_zonas_2026.js`
- Regiones y puntos insulares: `https://archivos.meteochile.gob.cl/portaldmc/AAA/sistemaAAA_coordenadas_regiones_2024.js`
- Definición antigua (región de Aysén): `…/sistemaAAA_coordenadas_zonas_2024_region_11_antigua.js`

**3. Qué publica**
183 variables `var coordenadas_<ZONA> = [{lat: …, lng: …}, …]` con **11.421 vértices** en total,
WGS84 en grados decimales. Nomenclatura: `<bloque><zona>`, donde el bloque va
`01a, 01b, 02, 03, 04, 05, 05m, 06, 07, 08a, 08b, 09, 10a, 10b, 11, 12` (bloques internos de la
DMC, **no** son códigos de región del INE — hay que mapearlos).

Tipos de zona presentes (taxonomía completa verificada):
Litoral · Litoral_Interior · Litoral_Interior_Norte · Cordillera_Costa · Pampa ·
Pampa_Patagonica_Norte/Sur · Valle_Longitudinal · Precordillera · **Precordillera_Occidental ·
Precordillera_Alto_Loa · Precordillera_Salar** · Cordillera · Cordillera_Austral ·
Cordillera_Austral_Norte/Sur · Patagonia_Subandina (+Norte/Sur) · Chiloe · Insular_Norte /
Insular_Central / Insular_Sur.

Algunas zonas son **multi-polígono**: el archivo `aaa_mapa.php` trae una función
`expandocoordenadas()` que traduce `02_Cordillera` → `02_Cordillera1,02_Cordillera2`. Hay que
portar esa tabla de expansión o el polígono queda incompleto.

**4. Familia: BASE TERRITORIAL.** Es la Piedra Rosetta entre la geografía de la amenaza y la
administrativa.

**5. Formato:** JavaScript (arreglos de objetos `{lat, lng}`). Conversión a GeoJSON/Shapefile:
directa, un parseo de una tarde.

**6. Acceso: ANÓNIMO — VERIFICADO.** HTTP 200, 598.723 bytes, sin credenciales.

**7. Granularidad territorial:** el polígono de zona es la unidad. Cruzándolo con la capa de
límites comunales (pendiente H-13, no es de mi dominio) se obtiene la traducción
zona ↔ comuna que el proyecto necesita.

**8. Frecuencia:** estático. Cambia **por año de definición**: existe una versión 2024 y una
2026, y un archivo explícito de «región 11 antigua». Es decir, **las zonas se redefinen**.

**9. Historia:** parcial — sobreviven en el servidor los archivos 2024 y 2026. Bajar y guardar
**ambos**: sin la definición del año correspondiente, una alerta histórica de 2024 no se puede
ubicar en el mapa correcto.

**10. Condiciones de uso — ⚠️ el mismo problema de la ficha 02.** Mismo host
`archivos.meteochile.gob.cl` con `robots.txt: Disallow: /`.
**Atenuante importante:** esto es una **descarga única de un archivo estático**, no un rastreo
recurrente. El espíritu del `robots.txt` es impedir el crawling; bajar tres archivos una vez, a
mano, no es crawling. Aun así, **incluir esta capa en la misma solicitud de autorización a la
DMC** y no redistribuirla sin permiso.

**11. Para qué sirve en el modelo**
Es lo que permite decir «estas tres subestaciones están dentro de la zona de peligro Alto». Sin
esta capa, el cruce entre el aviso de la DMC y el inventario de 835 elementos **no se puede
hacer**: son dos idiomas sin diccionario. Con ella, es una operación geométrica de punto-en-
polígono.

**12. Utilidad: ★★★** — máxima, y es la de tipo distinto: no cambia la prioridad de un activo
*por sí misma*, la **habilita**. Es la que convierte todas las demás fuentes de amenaza en algo
que se puede cruzar con activos. Sin ella, la ficha 02 vale cero.

**13. Riesgos y límites**
- Legitimidad del `robots.txt` (arriba).
- **No es cartografía oficial certificada**: es la geometría que la DMC usa para dibujar su
  mapa web. Puede estar generalizada. No sirve para litigio ni para deslinde; sí para asignar
  amenaza. Declararlo en la trazabilidad.
- Las zonas cambian de definición entre años → guardar la versión usada en cada consolidación.
- El mapeo bloque DMC (`08a`, `08b`, `05m`) → región INE **no está documentado**; hay que
  inferirlo cruzando con `zonaGeografica` del catastro de estaciones (ficha 01), que sí trae
  región y comuna. Ese cruce es una tarea concreta y acotada.

---

## FICHA 04 · DMC — IDE-DMC / GeoNode (infraestructura de datos espaciales)

**1. Organismo y producto:** DMC — IDE-DMC, catálogo geoespacial sobre plataforma GeoNode.

**2. URL exacta y endpoint:** `http://geonode.meteochile.gob.cl/`
(sub-rutas típicas de GeoNode: `/geoserver/wms`, `/geoserver/wfs`, `/geonetwork` — **no
probadas**, ver punto 10).

**3. Qué publica:** catálogo de capas meteorológicas y climatológicas con visor de mapas.
**No verificado el contenido** (ver punto 6).

**4. Familia:** AMENAZA / BASE TERRITORIAL (según capa).

**5. Formato:** SUPUESTO — GeoNode expone habitualmente WMS, WFS, GeoJSON y Shapefile.

**6. Acceso: NO VERIFICABLE HOY.** Al 16-ago-2026 el portal muestra: *«nuestro portal se
encuentra temporalmente en mantenimiento, los servicios de back end se encuentran operando con
normalidad»*. No se pudo listar ninguna capa. **Reintentar en unas semanas.**

**7–9. Granularidad / frecuencia / historia:** desconocidas.

**10. Condiciones de uso — VERIFICADO.** `http://geonode.meteochile.gob.cl/robots.txt`:
```
User-agent: *
Disallow: /geoserver/
Disallow: /geonetwork/
```
→ **Las rutas de servicios OGC (WMS/WFS) están explícitamente prohibidas al rastreo
automatizado.** El catálogo web sí está permitido. Por respeto a esa regla **no probamos**
`/geoserver/wfs?request=GetCapabilities`, aunque técnicamente hubiera sido una sola petición.
Es exactamente el caso «dato público ≠ uso automatizado permitido».

**11. Para qué sirve en el modelo:** sería el canal *correcto y bendecido* para obtener la capa
de zonas de la ficha 03 en formato geoespacial estándar, si es que está ahí. Vale la pena
preguntarlo por escrito junto con la solicitud de autorización.

**12. Utilidad: ★ (hoy)** — potencialmente ★★★ si al volver del mantenimiento publica los
polígonos de zonas como capa WFS con licencia clara. Re-evaluar.

**13. Riesgos y límites:** portal caído; sirve HTTP sin cifrar; robots restringe justo lo útil.
No construir nada que dependa de él hasta verificar.

---

## FICHA 05 · DMC — Condiciones de riesgo de incendio forestal (`getRiesgoIncendio`)

**1. Organismo y producto:** DMC — GeoJSON «Condiciones de riesgo para los incendios
forestales», datos actuales **y pronosticados**.

**2. URL exacta y endpoint**
`https://climatologia.meteochile.gob.cl/application/geoservicios/getRiesgoIncendio?usuario=…&token=…`
Documentación: `https://climatologia.meteochile.gob.cl/application/documentacion/getDocumento/10`

**3. Qué publica (verificado con respuesta real del 16-ago-2026)**
Por estación: temperatura actual (°C), humedad relativa (%), dirección (grados) e intensidad
del viento (kt), y además —esto es lo valioso—:
- `temperatura6h`, `humedad6h`, `intensidadVientoMaximo6h` (extremos de las últimas 6 h)
- `temperaturaMaximaHoy` / `Manana` (°C)
- `humedadMinimaHoy` / `Manana` (%)
- `intensidadVientoMaximoHoy` / `Manana` (km/h)
Cada *feature* trae `provincia`, `comuna`, `region` y geometría de punto.

**4. Familia: AMENAZA.** Es el trío clásico del comportamiento del fuego (calor + sequedad +
viento), con horizonte de 48 h.

**5. Formato:** GeoJSON `FeatureCollection`, geometría `Point`.

**6. Acceso: ANÓNIMO — VERIFICADO.** Respondió sin token con datos de las 148 estaciones
(ej.: Visviri, 2026-08-16 00:40, viento máximo pronosticado mañana 88,3 km/h).

**7. Granularidad territorial:** puntual por estación, **ya etiquetada con comuna y provincia**
— o sea, agregable directo a unidad administrativa sin capa auxiliar.

**8. Frecuencia:** los datos observados vienen con marca de tiempo de minutos atrás; el
pronóstico se refresca al menos diariamente.

**9. Historia:** ninguna. Es una foto del ahora + 48 h. **Si la queremos, la guardamos nosotros.**

**10. Condiciones de uso:** las de la ficha 01 (portal `climatologia`, sin `robots.txt`
restrictivo, uso automatizado documentado, obligación de citar a la DMC). **Uso automatizado
permitido — VERIFICADO.**

**11. Para qué sirve en el modelo:** alimenta `RIncFor` (variable 2.6 del glosario), que hoy
está a medias, con **pronóstico a 48 h** y no sólo con estadística histórica. Es el complemento
predictivo de las fichas 08 y 09 (CONAF), que son observación y pasado.

**12. Utilidad: ★★★** — sube la prioridad de líneas de transmisión, subestaciones y caminos de
acceso en comunas donde mañana se pronostican 88 km/h de viento con 27% de humedad. Cambia la
prioridad, hoy, por activo y por comuna.

**13. Riesgos y límites**
- **Es riesgo meteorológico, no riesgo de incendio.** No incluye combustible, pendiente,
  historial de ignición ni interfaz urbano-forestal. No confundir «condiciones favorables al
  fuego» con «va a haber un incendio».
- Cobertura puntual con red DGAC (aeropuertos): las zonas de interfaz forestal del centro-sur
  están mal representadas.
- Sin historia propia: no se puede validar retrospectivamente salvo que empecemos a acumular.

---

# BLOQUE B — AGUA (DGA / MOP)

---

## FICHA 06 · DGA — Sistema Hidrométrico en Línea (SNIA / DGA-SAT)

**1. Organismo y producto**
Dirección General de Aguas (DGA), Ministerio de Obras Públicas — Servicio Hidrométrico Nacional,
plataforma SNIA (Sistema Nacional de Información del Agua).

**2. URL exacta y endpoint**
- Página institucional: `https://dga.mop.gob.cl/sistema-hidrometrico-en-linea/`
- Mapa público en línea: `https://snia.mop.gob.cl/sat/site/informes/mapas/mapas.xhtml` — **HTTP 200 verificado** (2,1 MB)
- Portal DGA-SAT: `https://snia.mop.gob.cl/dgasat/pages/dgasat_main/dgasat_main.htm` — **HTTP 200, pero es una pantalla de LOGIN**
- Consulta de estaciones por parámetro: `https://snia.mop.gob.cl/dgasat/pages/dgasat_param/dgasat_param.jsp?param=1` — **HTTP 403**
- Oficina virtual: `https://snia.mop.gob.cl/portal-web/`

**3. Qué publica**
**1.330 estaciones** que transmiten en línea por satélite o GPRS:
- Fluviométrico: caudal (m³/s), nivel (m), temperatura del agua (°C)
- Meteorológico: precipitación (mm), temperatura (°C), humedad relativa (%)
- Calidad de agua: pH, oxígeno disuelto, turbiedad, otros
- Nivométrico: altura de nieve (cm) y equivalente en agua (mm)
- **Niveles y volúmenes de embalses y lagos**

**4. Familia: AMENAZA (crecidas, deshielo) + ESTADO-CONSECUENCIA (disponibilidad hídrica,
`EstHidric`, `D_uso`).**

**5. Formato:** HTML/JSF (`.xhtml`, `.jsp`). **No hay API documentada.**

**6. Acceso: MIXTO — VERIFICADO con matices.**
- El mapa público responde **con User-Agent de navegador**; con el User-Agent por defecto de
  `curl` devuelve **403 Forbidden**. Es decir: **el servidor filtra por User-Agent**, que es la
  manera técnica de decir «esto es para personas mirando, no para robots».
- `https://snia.mop.gob.cl/robots.txt` devuelve **403**: no hay reglas publicadas, pero tampoco
  bienvenida.
- El sistema DGA-SAT propiamente tal está **detrás de login** (`dgasat_login.htm`).
→ **No público para uso automatizado.** Hay dato en línea para mirar, no para consumir.

**7. Granularidad territorial:** puntual (estación) y por cuenca. 1.330 puntos, la red más densa
del país en el dominio agua.

**8. Frecuencia:** «en su mayoría, los datos se entregan cada 1 hora».

**9. Historia:** los datos en línea son **provisorios**. La serie validada vive en el Banco
Nacional de Aguas (ficha 07). Advertencia textual de la DGA: *«Todos los datos en línea son
provisorios y están sujetos a revisiones y/o modificaciones.»*

**10. Condiciones de uso — ⚠️ VERIFICADO.** Sin `robots.txt` accesible (403), con bloqueo por
User-Agent y con la parte operativa tras credenciales. **No automatizar.** Camino correcto:
solicitud formal a la DGA (Ley de Transparencia / convenio institucional) pidiendo (a) acceso a
la API interna si existe y (b) credenciales de solo lectura de DGA-SAT. Es de nuevo un caso
donde el destinatario final —SENAPRED— juega a nuestro favor.

**11. Para qué sirve en el modelo:** es la única fuente nacional de caudal y volumen de embalse
en tiempo casi real. Alimenta `EstHidric` y `D_uso`, y da la señal de crecida que ninguna otra
fuente da. Además, los nivométricos son el precursor de aluvión por deshielo.

**12. Utilidad: ★★★ (si conseguimos el acceso) / ★ (hoy)** — un embalse por sobre cota o un
caudal en crecida cambia inmediatamente la prioridad de puentes, captaciones y plantas de
tratamiento aguas abajo. Pero hoy no lo podemos consumir legítimamente. **Convertir el acceso a
la DGA en pendiente de alta prioridad del proyecto.**

**13. Riesgos y límites**
- Acceso: el problema central.
- Dato provisorio: no usar la serie en línea para historia ni para validar.
- Un caudal alto no es una amenaza por sí solo; depende de la cota del activo. Sin el modelo
  digital de elevación, la señal es ambigua.

---

## FICHA 07 · DGA — Banco Nacional de Aguas (BNA) y estadísticas de estaciones

**1. Organismo y producto:** DGA — BNA, reportes oficiales de la Red Hidrométrica Nacional
(serie validada, no provisoria).

**2. URL exacta y endpoint**
- `https://snia.mop.gob.cl/BNAConsultas/reportes` — **HTTP 200 verificado** (45 KB, con
  User-Agent de navegador)
- `https://dga.mop.gob.cl/estadisticas-estaciones-dga/`
- Mapoteca digital (cartografía): `https://dga.mop.gob.cl/mapoteca-digital/`

**3. Qué publica:** datos fluviométricos y meteorológicos validados de estaciones desde Arica a
Tierra del Fuego, más reportes de calidad de agua.

**4. Familia:** AMENAZA + ESTADO-CONSECUENCIA (línea base hídrica).

**5. Formato:** consulta web (JSF); descarga a Excel/CSV desde la interfaz.

**6. Acceso: ANÓNIMO pero con User-Agent de navegador — VERIFICADO parcialmente.** Responde la
página de consulta; **no verificamos** que se pueda armar una URL de descarga directa.
→ El resto, **SUPUESTO**.

**7. Granularidad:** estación / cuenca.
**8. Frecuencia:** validación diferida (meses).
**9. Historia:** larga — es el archivo oficial del país en materia hídrica.

**10. Condiciones de uso:** mismas cautelas de la ficha 06. Mismo host filtrado por User-Agent.
**No automatizar sin autorización.**

**11. Para qué sirve:** es la línea base contra la cual se juzga si un caudal actual es alto o
bajo. Sin normal climatológica, un número de caudal no significa nada.

**12. Utilidad: ★★** — no cambia la prioridad hoy, pero sin ella la fuente de la ficha 06 no se
puede interpretar. Es contexto, no gatillo.

**13. Riesgos y límites:** consulta pensada para humanos; sin API; sin garantía de estabilidad
de URLs.

---

## FICHA 08 · Portal de Datos Abiertos del Estado (datos.gob.cl) — vía CKAN

**1. Organismo y producto:** Gobierno de Chile — portal nacional de datos abiertos, motor CKAN.
Incluye conjuntos publicados por la propia DGA.

**2. URL exacta y endpoint**
`https://datos.gob.cl/api/3/action/package_search?q={consulta}&rows={n}` — **VERIFICADO,
HTTP 200, JSON**.

**3. Qué publica (verificado con dos consultas reales)**
- *Derechos de aprovechamiento de aguas a nivel nacional registrados en DGA 2025* — DGA, XLSX,
  licencia **CC-BY**
- *Mapoteca Dirección General de Aguas* — DGA, HTML, CC-BY
- *Nómina de patentes de aguas* — Tesorerías, CSV, CC-0
- Otros de medio ambiente (emisiones al agua y al aire), XLSX/CSV
- En incendios forestales: la búsqueda devolvió **sólo 2 resultados y ninguno de CONAF**.

**4. Familia: ACTIVO / BASE TERRITORIAL** (derechos de agua = quién puede usar qué, dónde).

**5. Formato:** API CKAN (JSON) que apunta a recursos XLSX/CSV/HTML.

**6. Acceso: ANÓNIMO — VERIFICADO.** Sin registro, sin token.

**7. Granularidad:** varía por conjunto; los derechos de agua traen ubicación.
**8. Frecuencia:** por conjunto, típicamente anual.
**9. Historia:** por conjunto.

**10. Condiciones de uso — VERIFICADO en el propio metadato.** Licencias explícitas: **CC-BY**
(atribución) y **CC-0** (dominio público). Es el único canal del dominio con **licencia formal
declarada máquina-legible**. → **Uso automatizado permitido sin ambigüedad.**

**11. Para qué sirve:** aporta el «quién depende de qué agua», útil para la consecuencia
(`D_uso`) más que para la amenaza. Y sirve como **canal de respaldo legítimo** cuando el portal
del organismo bloquea.

**12. Utilidad: ★** — no cambia la prioridad de un activo por sí sola, pero es la única fuente
con licencia limpia y por eso vale como base de trazabilidad.

**13. Riesgos y límites:** cobertura pobre e irregular (CONAF prácticamente ausente); datos
anuales y desactualizados respecto del portal del organismo.

---

# BLOQUE C — INCENDIO FORESTAL (CONAF)

---

## FICHA 09 · CONAF — Informe Diario Estadístico de Incendios Forestales

**1. Organismo y producto**
CONAF — Central Nacional de Coordinación de Incendios Forestales. «Estadística de ocurrencia y
daño de incendios forestales — INFORME DIARIO ESTADÍSTICO». Fuente de origen: sistema **SIDCO**.

**2. URL exacta y endpoint**
`https://www.conaf.cl/mapas-incendios-forestales/ocurrencia-nacional.htm` — **HTTP 200
verificado**, 29.982 bytes.

**3. Qué publica (verificado con la respuesta real del 15-ago-2026, 21:30)**
Tres secciones:
1. **Reporte del día**, por región: total de incendios, bajo observación, en combate,
   controlado, extinguido en el día, superficie afectada (ha).
2. **Resumen acumulado de la temporada** (2026-2027) por región: número de incendios y
   superficie afectada (ha), **comparado con la temporada anterior y con el promedio del
   quinquenio**, con la variación porcentual ya calculada (ej. nacional: 29 incendios, +7% vs
   temporada anterior, −17% vs quinquenio; 38,28 ha, +5% y −96% respectivamente).
3. **Situación de los incendios de magnitud vigentes**, con columnas:
   `Región | Provincia | Comuna | Nombre | Ámbito | Fecha de Inicio | Superficie afectada (ha) | Estado`.

**4. Familia: AMENAZA (evento activo) + ESTADO-CONSECUENCIA (superficie ya dañada).**

**5. Formato:** HTML estático con tablas bien estructuradas (`<table class="tabla">`). Parseo
simple y estable. **No es JSON**, pero es de los HTML más parseables que hay.

**6. Acceso: ANÓNIMO — VERIFICADO.** Sin credenciales, sin filtro de User-Agent.

**7. Granularidad territorial:** región en las tablas agregadas; **comuna y provincia en la
tabla de incendios de magnitud vigentes** — que es justo la que importa para el cruce con
activos.

**8. Frecuencia:** la propia página declara **«actualización cada 5 minutos»**. Es prácticamente
tiempo real.

**9. Historia:** la página es sólo del día. La historia está en la ficha 10.

**10. Condiciones de uso — VERIFICADO.** `https://www.conaf.cl/robots.txt` bloquea únicamente
`/cgi-bin`, `/wp-admin/`, `/wp-includes/`, plugins, temas y feeds. **La ruta
`/mapas-incendios-forestales/` NO está bloqueada.** `https://sidco.conaf.cl/robots.txt` sólo
bloquea `/images/` y `/template/`.
→ **Uso automatizado permitido.** Aun así: consultar a ritmo razonable (una vez cada 15–30 min
como máximo, no cada 5), y citar a CONAF.

**11. Para qué sirve en el modelo:** es el **evento real**, no el pronóstico. Da el incendio
activo con comuna, fecha de inicio, superficie y estado — el insumo directo para marcar activos
en riesgo inmediato y para validar retrospectivamente el `RIncFor` de la ficha 05.

**12. Utilidad: ★★★** — un incendio de magnitud vigente en una comuna cambia la prioridad de
todos los activos de esa comuna, hoy y con nombre propio. Máxima.

**13. Riesgos y límites**
- HTML sin contrato: CONAF puede rediseñar la página cuando quiera. Adaptador aislado,
  degradación explícita.
- Sin coordenadas en la tabla vigente: llega a **comuna**, no a punto. Para cruzar con activos
  puntuales hay que asumir la comuna entera, lo cual sobre-estima. Declararlo.
- La codificación del HTML es irregular (aparecen caracteres mal formados en algunos títulos):
  forzar decodificación robusta.

---

## FICHA 10 · CONAF — Estadísticas históricas de incendios forestales

**1. Organismo y producto:** CONAF — Centro documental, serie histórica de ocurrencia y daño
(origen SIDCO).

**2. URL exacta y endpoint**
Índice: `https://www.conaf.cl/centro-documentals/estadisticas-historicas/`
Documentos (verificados en el índice, contenido no descargado):
- `…/centro-documental/resumen-de-ocurrencia-y-dano-por-comuna-1985-2023/` → **por comuna, 1985–2024**
- `…/centro-documental/hectareas-por-incendio-nacional-de-incendios-forestales-por-region-1977-2023/` → por región, **1977–2024**
- `…/distribucion-nacional-de-la-ocurrencia-n-de-incendios-forestales-segun-causalidad-2003-2023/` → causalidad, 2003–2024
- `…/ocurrencia-y-dano-de-incendios-de-magnitud-1985-2023/`
- `…/ocurrencia-nacional-de-incendios-forestales-segun-mes-1985-2023/` y otros de estacionalidad
(⚠️ Las URL conservan el rango antiguo `1985-2023` en la ruta aunque el título ya diga 2024.)

También existe *Registro histórico de incendios forestales* en
`https://www.plataformadedatos.cl/datasets/es/94233F2661964B0` (2002–2020) — **no verificado**.

**3. Qué publica:** número de incendios y hectáreas afectadas, por comuna / región / mes / causa
/ rango horario / magnitud.

**4. Familia: AMENAZA (frecuencia base) — es el «cuán a menudo pasa acá».**

**5. Formato:** SUPUESTO — XLSX o PDF según documento (no descargamos los archivos, por la regla
de no descargar masivamente).

**6. Acceso: ANÓNIMO — SUPUESTO** (el índice sí es anónimo y verificado; los archivos no se
probaron).

**7. Granularidad territorial: COMUNA** en el documento principal. Es la granularidad correcta
para el modelo.

**8. Frecuencia:** anual, al cierre de temporada.

**9. Historia: 1985–2024 por comuna; 1977–2024 por región.** Cuarenta años. Es la mejor serie
histórica de todo mi dominio.

**10. Condiciones de uso:** `robots.txt` de CONAF permite `/centro-documental*`. **Uso
automatizado permitido**, citando a CONAF.

**11. Para qué sirve en el modelo:** es la **probabilidad base por comuna**. El pronóstico de la
ficha 05 dice «hoy hay condiciones»; esta serie dice «acá se queman 40 ha al año desde 1985».
La prioridad se construye multiplicando las dos: condición actual × propensión histórica.

**12. Utilidad: ★★** — no cambia la prioridad *hoy*, pero es la que fija la prioridad **de
fondo**, la que determina qué activos hay que reforzar antes de que llegue la temporada. Para
un instrumento de planificación (que es lo que somos, no de alerta), esto pesa mucho.

**13. Riesgos y límites**
- El registro es de **ocurrencia**, no de **impacto sobre infraestructura**. Un incendio grande
  en zona sin activos no debe subir prioridad.
- Cambios de metodología y de límites comunales en 40 años: comparar con cuidado.
- Formato de archivo pensado para lectura humana (tablas Excel con encabezados combinados):
  el parseo es tedioso y frágil.

---

# BLOQUE D — CLIMA DE REFERENCIA Y FUENTES GLOBALES

---

## FICHA 11 · CR2 — CR2MET v2.5 (producto grillado para Chile)

**1. Organismo y producto:** Centro de Ciencia del Clima y la Resiliencia (CR)2 — CR2MET v2.5,
producto grillado de precipitación y temperaturas extremas para Chile continental.

**2. URL exacta y endpoint**
- Página: `https://www.cr2.cl/datos-productos-grillados/`
- Precipitación: `https://www.cr2.cl/download/cr2met-v2-5-pr-day-1960-2021/?wpdmdl=40969`
- Temperaturas extremas: `https://www.cr2.cl/download/cr2met-v2-5-txn-day-1960-2021/?wpdmdl=40966`
- DOI: `10.5281/zenodo.7529682`

**3. Qué publica:** precipitación diaria (`pr`, mm/día) y temperaturas máxima/mínima diarias
(`txn`, °C), grilla de **0,05° × 0,05°** (≈5,5 km), **1960–2021**.

**4. Familia: BASE TERRITORIAL / AMENAZA (climatología de referencia).**

**5. Formato:** netCDF.

**6. Acceso: ANÓNIMO con formulario de registro — VERIFICADO en la página.** El sitio pide
completar un formulario (institución, área, uso previsto) antes de la descarga. No es un
paywall, es un registro de cortesía.

**7. Granularidad territorial:** celda de ~5,5 km. **Es la mejor resolución espacial disponible
para Chile** en mi dominio: cubre la superficie completa, no puntos.

**8. Frecuencia:** estático por versión (v2.5 publicada en 2023).

**9. Historia: 1960–2021.** Sesenta y dos años.

**10. Condiciones de uso — VERIFICADO.** `robots.txt` de `cr2.cl` sólo bloquea `/wp-admin/`.
**Obligación de citar** (cita formal indicada por el propio CR2: Boisier, J.P., 2023, Zenodo).
Además, existe copia en **Zenodo** con DOI, que es el canal correcto y estable para automatizar:
Zenodo permite descarga programática y tiene licencia explícita. **Preferir Zenodo sobre el
WordPress del CR2.**

**11. Para qué sirve en el modelo:** es la climatología de referencia contra la que se juzga si
un evento es excepcional. Y como es **grillado**, resuelve el problema estructural de todas las
fuentes de estación: permite asignar un valor climático a *cada uno de los 835 elementos*, esté
donde esté, sin interpolar a mano.

**12. Utilidad: ★★** — no cambia la prioridad hoy (termina en 2021), pero define el
denominador. Sin él, «120 mm en 24 h» no se sabe si es un martes cualquiera en Valdivia o un
récord en Calama.

**13. Riesgos y límites**
- **Termina en 2021.** No sirve para operación, sólo para línea base.
- Chile **continental**: sin Isla de Pascua, sin Juan Fernández, sin Antártica — y la ficha 02
  muestra que la DMC sí emite alertas para Isla de Pascua. Hueco declarado.
- Es un producto **reanalizado**, no observación: hereda los sesgos de su modelo. Precedente
  propio: en Cosmoclima ya vimos que un empalme con reanálisis puede comprimir el rango de los
  años secos. Usar con la misma desconfianza.

---

## FICHA 12 · CR2 — Explorador Climático, VisMet y series depuradas DMC+DGA

**1. Organismo y producto:** (CR)2 — plataformas de consulta sobre la base de datos consolidada
que el centro arma con datos de DMC, DGA y la Red Agroclimática Nacional.

**2. URL exacta y endpoint**
- Explorador Climático: `https://www.cr2.cl/explorador-climatico/` (visor en `http://explorador.cr2.cl/`)
- Visualizador Meteorológico (VisMet): `https://www.cr2.cl/visualizador-meteorologico/`
- Datos de precipitación: `https://www.cr2.cl/datos-de-precipitacion/`
- Bases de datos: `https://www.cr2.cl/bases-de-datos/`
- Explorador de Cuencas: `https://www.cr2.cl/explorador-de-cuencas/`

**3. Qué publica:** temperatura (máx/mín/media), precipitación y **caudal**, de estaciones de la
DMC y la DGA. Precipitación acumulada diaria y mensual descargable **1930–2020**. VisMet trabaja
con más de **500 estaciones a paso horario**.

**4. Familia: AMENAZA + ESTADO-CONSECUENCIA.**

**5. Formato:** CSV descargable desde la interfaz; visores web.

**6. Acceso: ANÓNIMO — SUPUESTO.** No probamos las descargas (regla de no descargar
masivamente). La navegación del sitio sí es anónima y verificada.

**7. Granularidad territorial:** estación; el Explorador de Cuencas agrega por cuenca.

**8. Frecuencia:** VisMet es horario y actual; las series descargables llegan hasta 2020.

**9. Historia: 1930–2020** en precipitación. Es la serie más larga del dominio.

**10. Condiciones de uso:** `robots.txt` permisivo (sólo `/wp-admin/`). Uso académico esperado,
con cita al (CR)2. **Uso automatizado probablemente permitido — SUPUESTO**, confirmar antes de
programar descargas recurrentes.

**11. Para qué sirve en el modelo:** el (CR)2 hace un trabajo que nosotros no queremos rehacer:
**depurar** las series de DMC y DGA (huecos, saltos, cambios de instrumento). Y —dato no menor—
**es la vía indirecta para tener datos de caudal de la DGA sin depender del acceso bloqueado de
la ficha 06.**

**12. Utilidad: ★★** — indirecta pero real: es el atajo legítimo a la DGA y la fuente de series
limpias para calibrar.

**13. Riesgos y límites**
- Es un **tercero**, no el organismo dueño del dato. Sirve para calibrar y prototipar; para un
  insumo de decisión pública hay que declarar la cadena (DGA → CR2 → nosotros). Mismo criterio
  que aplicamos a `api.gael.cloud` con el CSN.
- Rezago de años en las series descargables.
- Sin contrato de servicio: es un centro de investigación con financiamiento por proyecto.

---

## FICHA 13 · Open-Meteo Archive (ERA5) — reanálisis global

**1. Organismo y producto:** Open-Meteo (Suiza) — API de archivo climático sobre reanálisis
ERA5/ERA5-Land del ECMWF.

**2. URL exacta y endpoint**
`https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=…&end_date=…&daily=precipitation_sum,temperature_2m_max&timezone=UTC`

**3. Qué publica:** precipitación (mm), temperaturas (°C), viento, humedad, radiación — horario
y diario, para **cualquier coordenada del planeta**.

**4. Familia: AMENAZA (serie continua de respaldo).**

**5. Formato:** JSON (también CSV y FlatBuffers).

**6. Acceso: ANÓNIMO — VERIFICADO EN USO PREVIO, hoy con tope.** La petición de prueba del
16-ago-2026 devolvió **HTTP 429: «Hourly API request limit exceeded»** — es decir, el servicio
está vivo y respondiendo, pero la cuota horaria gratuita ya estaba consumida (probablemente por
el propio trabajo de Cosmoclima). El endpoint ya está validado en trabajo anterior del equipo.

**7. Granularidad territorial:** grilla ERA5 (~28 km) / ERA5-Land (~9 km), consultada por punto.
**Cubre Isla de Pascua y Juan Fernández**, que CR2MET no cubre.

**8. Frecuencia:** diaria/horaria, con ~5 días de rezago.

**9. Historia:** desde **1940**.

**10. Condiciones de uso:** uso no comercial gratuito con **límite de peticiones por hora**
(comprobado en carne propia). Licencia de los datos: **CC-BY 4.0** del ECMWF/Copernicus.
→ **Uso automatizado permitido, con cuota.** Para producción nacional habría que planificar
cacheo agresivo o plan pagado.

**11. Para qué sirve en el modelo:** relleno universal. Cuando un activo está lejos de toda
estación, ERA5 da un valor. Es el «no hay dato» convertido en «hay un dato de menor confianza»
— y el MACC exige declarar esa confianza, no esconderla.

**12. Utilidad: ★★** — cambia la prioridad sólo de forma indirecta, evitando que un activo quede
sin evaluar por falta de estación cercana. Su valor es de cobertura, no de precisión.

**13. Riesgos y límites**
- **Precedente propio, y es serio:** ya verificamos en Cosmoclima (13-ago) que ERA5 corregido
  contra 38 estaciones de Coquimbo **comprime el rango**, sobrestimando los años secos hasta
  2,4×. No usar ERA5 crudo como si fuera observación.
- Tope horario que corta sin aviso (lo vimos hoy).
- Es un intermediario privado sobre dato de Copernicus. Para producción pública, considerar el
  CDS de Copernicus directamente.

---

## FICHA 14 · NASA POWER — series meteorológicas satelitales

**1. Organismo y producto:** NASA Langley Research Center — POWER (Prediction Of Worldwide
Energy Resources), API diaria por punto.

**2. URL exacta y endpoint**
`https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start=YYYYMMDD&end=YYYYMMDD&format=JSON`

**3. Qué publica:** precipitación corregida (`PRECTOTCORR`, **mm/día**), temperaturas, humedad,
viento, radiación solar. Fuente subyacente declarada en la respuesta: `GEOSIT`.

**4. Familia: AMENAZA (respaldo).**

**5. Formato:** GeoJSON/JSON (`type: Feature`, con `header`, `parameters` y `fill_value: -999.0`).

**6. Acceso: ANÓNIMO — VERIFICADO.** API v2.9.6 respondió HTTP 200 el 16-ago-2026 para
(-29,9; -71,25) con datos de agosto 2026. Sin token, sin registro.

**7. Granularidad territorial:** grilla ~0,5° (≈50 km). Gruesa.

**8. Frecuencia:** diaria, con algunos días de rezago.

**9. Historia:** desde 1981 (algunos parámetros desde 1984).

**10. Condiciones de uso:** datos de la NASA, **dominio público**, con solicitud de citar la
fuente. Uso automatizado explícitamente soportado (es una API pública documentada).
→ **Permitido — con la salvedad de que no verificamos el `robots.txt` de este host.**

**11. Para qué sirve en el modelo:** segunda opinión independiente frente a ERA5. Cuando dos
reanálisis de linaje distinto coinciden, la confianza sube; cuando discrepan, es señal de que
la zona está mal restringida y el dato debe declararse de baja confianza.

**12. Utilidad: ★** — 50 km de celda es demasiado grueso para decidir sobre un activo puntual en
la geografía chilena, donde en 50 km se pasa del mar a 4.000 m. Sirve como control cruzado, no
como insumo directo.

**13. Riesgos y límites:** resolución inadecuada para terreno complejo; valor de relleno −999
que hay que filtrar o contamina cualquier promedio; es una fuente extranjera para un instrumento
de decisión pública chilena — usar como control, no como base.

---

## FICHA 15 · CHIRPS — precipitación satelital de alta resolución

**1. Organismo y producto:** Climate Hazards Center, UC Santa Barbara — CHIRPS 2.0.

**2. URL exacta y endpoint**
`https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/` — **HTTP 200
verificado** (16-ago-2026, servidor nginx, listado de directorio abierto).

**3. Qué publica:** precipitación diaria estimada (mm/día), combinando satélite e
interpolación con estaciones, en grilla de **0,05°** (≈5,5 km).

**4. Familia: AMENAZA.**

**5. Formato:** netCDF (también GeoTIFF y BIL en otras carpetas del mismo árbol).

**6. Acceso: ANÓNIMO — VERIFICADO.** Servidor de archivos abierto, sin credenciales.

**7. Granularidad territorial:** ~5,5 km, cobertura 50°S–50°N.
**⚠️ Chile llega a 56°S: Magallanes y la Antártica quedan FUERA de CHIRPS.** Hueco estructural.

**8. Frecuencia:** diaria, con producto preliminar a ~2 días y final a ~3 semanas.

**9. Historia:** desde **1981**.

**10. Condiciones de uso:** dato de acceso abierto, con solicitud de citar (Funk et al., 2015).
→ **Uso automatizado permitido — SUPUESTO** (no consultamos `robots.txt` de `data.chc.ucsb.edu`;
es un servidor de datos científicos pensado para descarga programática).

**11. Para qué sirve:** tercera opinión sobre lluvia, con resolución comparable a CR2MET pero
llegando hasta hoy (donde CR2MET termina en 2021). Cubre el hueco temporal del producto chileno.

**12. Utilidad: ★** — mismo argumento que NASA POWER: control cruzado, no insumo primario, y
además con un agujero geográfico grande justo en la zona austral.

**13. Riesgos y límites**
- **Sin Magallanes ni Antártica.** Para un consolidador *nacional*, eso no es un detalle.
- Los archivos globales son enormes: descargar el mundo entero para mirar Chile es un desperdicio
  y una falta de cortesía con un servidor universitario. Buscar el subconjunto sudamericano.
- El producto preliminar y el final difieren: no mezclar.

---

## FICHA 16 · ESA CCI Soil Moisture — humedad de suelo vía OPeNDAP del CEDA

**1. Organismo y producto:** European Space Agency — Climate Change Initiative, producto
COMBINED de humedad de suelo superficial, servido por el CEDA (Reino Unido).

**2. URL exacta y endpoint:** acceso por **OPeNDAP anónimo del CEDA**. El endpoint concreto
está documentado en el trabajo previo de Cosmoclima (nota `cosmoclima-humedad-suelo-esacci-
arbitraje-14ago2026`). **No se re-verificó hoy: se toma como VERIFICADO por el equipo el
14-ago-2026.**

**3. Qué publica:** humedad volumétrica del suelo superficial (m³/m³), diaria, grilla ~25 km.

**4. Familia: AMENAZA (precursor de remoción en masa) + ESTADO-CONSECUENCIA (sequía).**

**5. Formato:** netCDF vía OPeNDAP (permite pedir sólo el trozo espacial/temporal que se
necesita, sin bajar el archivo completo — es la manera cortés de consumirlo).

**6. Acceso: ANÓNIMO — VERIFICADO por el equipo (14-ago-2026)**, sin GDAL ni el paquete de la
ESA.

**7. Granularidad territorial:** ~25 km. Gruesa.
**8. Frecuencia:** diaria, con rezago de meses.
**9. Historia:** desde 1978 (combinando misiones).

**10. Condiciones de uso:** datos ESA CCI de acceso abierto con obligación de citar. El CEDA
exige uso responsable del servicio OPeNDAP. → **Permitido, a baja frecuencia.**

**11. Para qué sirve en el modelo:** es el **eslabón que conecta lluvia con aluvión**. La misma
lluvia sobre suelo saturado y sobre suelo seco produce cosas distintas. Es la variable que puede
explicar por qué la Minuta Técnica de SERNAGEOMIN declara peligro Alto un día y no otro con
lluvias parecidas — o sea, es candidata a **validar nuestro `C_clim` contra la minuta**.

**12. Utilidad: ★** hoy, con potencial de ★★ — lección propia y ya pagada: en Cosmoclima esta
misma fuente **no decidió, pero acotó** el desacuerdo a 5 meses de 91. Es una fuente que afina,
no que resuelve. Y un detalle metodológico que ya nos costó aprender: **usar el MÁXIMO mensual,
no el promedio.**

**13. Riesgos y límites**
- 25 km es demasiado grueso para una quebrada. En la Precordillera de Antofagasta, una celda
  cubre varias zonas de peligro distintas.
- Rezago de meses: **inútil para operación**, sólo para análisis retrospectivo y calibración.
- Mide humedad **superficial** (primeros centímetros), no el perfil que gobierna la estabilidad
  de una ladera.

---

# SÍNTESIS OPERATIVA

## Ranking por el criterio único (¿cambia la prioridad de un activo?)

| ★★★ — sí, hoy, con nombre y apellido | ★★ — habilita o calibra | ★ — control cruzado / contexto |
|---|---|---|
| **02** DMC Sistema AAA (avisos/alertas) | **01** DMC API climatológica | **04** IDE-DMC GeoNode (hoy caído) |
| **03** DMC capa de zonas *(habilitadora absoluta)* | **07** DGA BNA histórico | **08** datos.gob.cl CKAN |
| **05** DMC riesgo de incendio 48 h | **10** CONAF histórico por comuna | **14** NASA POWER |
| **09** CONAF informe diario | **11** CR2MET v2.5 | **15** CHIRPS |
| **06** DGA hidrométrico *(★★★ sólo si se consigue acceso)* | **12** CR2 exploradores | **16** ESA CCI humedad de suelo |
| | **13** Open-Meteo / ERA5 | |

## Semáforo de permiso de uso automatizado

| Verde — permitido y verificado | Ámbar — pedir permiso primero | Rojo — bloqueado |
|---|---|---|
| DMC `climatologia.meteochile.gob.cl` (fichas 01, 05) — sin robots.txt, con ejemplos de `wget` en la propia documentación | DMC `archivos.meteochile.gob.cl` (fichas 02, 03) — **`robots.txt: Disallow: /`** | DGA `snia.mop.gob.cl` DGA-SAT (ficha 06) — **login** |
| CONAF (fichas 09, 10) — robots.txt permite las rutas que usamos | DMC GeoNode `/geoserver/` `/geonetwork/` (ficha 04) — prohibidos por robots | |
| datos.gob.cl (ficha 08) — CC-BY / CC-0 explícitas | DGA `snia.mop.gob.cl` público (fichas 06, 07) — filtra por User-Agent, robots devuelve 403 | |
| CR2, Zenodo, Open-Meteo, NASA POWER, CHIRPS, CEDA (11–16) | | |

## Pendientes concretos que abre este catastro

1. **Carta a la DMC** (`portalserviciosclimaticos@meteochile.gob.cl`) pidiendo tres cosas:
   (a) autorización para consumir `datos_AAA.js` a baja frecuencia pese al `robots.txt`;
   (b) permiso para usar la capa de zonas `sistemaAAA_coordenadas_zonas_2026.js`;
   (c) **preguntar si existe o está planificada una salida CAP** (Common Alerting Protocol).
   Buscamos y no hallamos evidencia de que la DMC publique CAP; la OMM lo empuja para todos los
   servicios meteorológicos nacionales, así que la pregunta es pertinente y puede ahorrarnos
   todo el parseo.
2. **Registrarse en el portal de Servicios Climáticos** de la DMC y obtener token, aunque hoy
   el acceso anónimo funcione.
3. **Solicitud formal a la DGA** por acceso a datos hidrométricos (ficha 06). Es el hueco más
   grande del dominio: el agua es la amenaza con mejor red de medición del país y la que peor
   podemos consumir.
4. **Empezar a capturar historia HOY** de las fuentes que no la guardan: `datos_AAA.js`
   (ficha 02), `getRiesgoIncendio` (ficha 05) y el informe diario de CONAF (ficha 09). Ninguna
   de las tres tiene archivo histórico consultable. Cada día sin capturar es un día perdido para
   siempre — y sin historia no hay validación.
5. **Convertir la capa de zonas a GeoJSON** y cruzarla con el catastro de estaciones (ficha 01,
   que trae `zonaGeografica` + comuna + provincia) para inferir el mapeo bloque DMC → región INE,
   que no está documentado en ninguna parte.
6. **Re-visitar el IDE-DMC (GeoNode)** cuando termine el mantenimiento: si publica los polígonos
   de zona como capa WFS con licencia, el problema del `robots.txt` desaparece de un plumazo.
