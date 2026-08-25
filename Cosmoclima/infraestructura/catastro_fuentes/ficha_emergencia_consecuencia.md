# Familia ESTADO-CONSECUENCIA: el registro de lo que realmente pasó

Catastro de fuentes · Proyecto «Infraestructura Crítica × Clima» (RMD 2.0)
Verificado el **15-16 de agosto de 2026** con peticiones HTTP directas.

---

## Por qué esta familia decide si el proyecto es ciencia o adorno

El proyecto quiere decir «esta subestación va a caer». Una frase así sólo vale
algo si **existe una libreta donde anotar si cayó o no**. Sin libreta, el modelo
es como un entomólogo que describe una especie preciosa que nadie puede ir a
buscar al campo: elegante, coherente, e imposible de contradecir. Esta familia
es la libreta.

### El hallazgo, en una línea

> **Chile SÍ tiene la libreta, y llega hasta la comuna: 50.457 eventos de
> emergencia (2015–2024) más 12.896 fichas históricas (1970–2014). Juntas dan
> 55 años continuos. Y 13.577 de esos eventos son, literalmente, fallas de
> servicio, suministro o conectividad vial.**

La libreta estaba escondida: no está en el portal de datos abiertos ni en el
sitio principal de SENAPRED, sino en su **biblioteca digital**, publicada en
septiembre de 2025, en un Excel de 11 MB que casi nadie mira.

### Mapa rápido de las fichas

| # | Fuente | Desde | ¿Comuna? | ¿Infraestructura? | Utilidad |
|---|---|---|---|---|---|
| 1 | **SENAPRED · Consolidado Eventos 2015–2024** | 2015 | ✅ sí | ✅ sí (13.577 fallas) | ★★★ |
| 2 | SENAPRED · Eventos por Año (10 Excel) | 2015 | ✅ sí | ✅ sí | ★★★ |
| 3 | **DesInventar Chile (UNDRR)** | **1970** | ✅ sí | ⚠️ parcial | ★★★ |
| 4 | BiblioGRD · OAI-PMH (el canal automático) | 2012 | — | — | ★★ |
| 5 | SENAPRED · Informes Estadísticos anuales | 1997 | ⚠️ región | ⚠️ agregado | ★★ |
| 6 | SENAPRED · Alertas vigentes (ArcGIS público) | hoy | ✅ sí | ❌ no | ★★ |
| 7 | SENAPRED · API de alertas (AppSync) | ? | ? | ? | ★ (cerrada) |
| 8 | Informes ALFA y DELTA (originales) | — | ✅ sí | ✅ sí | ★ (no públicos) |
| 9 | datos.gob.cl (CKAN) | — | — | ❌ | ★ (vacío) |
| 10 | EM-DAT (CRED/UCLouvain) | 1900 | ❌ país | ❌ | ★ |
| 11 | CIGIDEN · Repositorio del Desastre | 2023 | narrativo | ❌ | ★ |
| 12 | INE · clasificación de emergencias | — | — | — | ★ (no hallada) |

---

---

# FICHA 1 ★★★ — SENAPRED · Consolidado de Eventos de Emergencia en Chile 2015–2024

**LA FUENTE DE VALIDACIÓN DEL PROYECTO.** Si sólo se pudiera bajar un archivo
de todo este catastro, es éste.

**1. Organismo y producto**
SENAPRED (ex ONEMI) — *Consolidado de Eventos de Emergencia en Chile 2015–2024*.
Publicado en BiblioGRD (biblioteca digital institucional), colección
«Documentación → Estadísticas e Indicadores», con fecha 05-sep-2025.
Los registros los levantan las **Unidades de Alerta Temprana Regionales (UAT,
ex CAT)** a partir de los Informes de Incidentes/Emergencias del sistema
interno **SID**.

**2. URL exacta y endpoint** — **VERIFICADO**
- Ficha del ítem:
  `https://bibliogrd.senapred.gob.cl/handle/1671/7120`
- Archivo directo (HTTP 200, `application/vnd.openxmlformats...sheet`,
  11.227.555 bytes):
  `https://bibliogrd.senapred.gob.cl/bitstream/handle/1671/7120/Eventos_Emergencia_2015_2024.xlsx?sequence=1&isAllowed=y`

**3. Qué publica**
Un Excel con dos hojas:
- `Nota Aclaratoria` — el origen del dato y sus advertencias.
- `Eventos_de_Emergencia_2015_2024` — **50.457 filas de eventos**, 44 columnas.

Columnas verificadas leyendo el archivo:

| Bloque | Columnas |
|---|---|
| Temporal | `Fecha Inicio`, `Hora Evento` |
| **Geográfico** | `Región`, `Provincia`, **`Comuna`** |
| Taxonomía | `Origen Evento` (Natural/Antrópico), `Clase Evento`, `Tipo Evento`, `Sub Evento 1`, `Sub Evento 2` |
| Respuesta | `Organismos de Respuesta` |
| Personas | Afectados, Damnificados, Albergados, N° Albergues, Aislados, Damnificados Laborales, Fallecidos, Heridos, Desaparecidos, Extraviados, Evacuados (cada uno desagregado hombres/mujeres/total) |
| Viviendas | `Viviendas Con Daño Menor`, `Viviendas Con Daño Mayor`, `Viviendas Destruidas` |
| Texto libre | `Antecedentes Observaciones` |

**Lo decisivo para el proyecto** — la columna `Clase Evento` (conteos reales):

| Clase de evento | N° eventos 2015–2024 |
|---|---|
| Incendios (+ variante con espacio) | 15.338 + 2.641 |
| **Falla de Servicios y Suministros** | **11.835** |
| Accidente de Medios de Transporte | 6.769 |
| Precipitaciones | 3.214 |
| Accidente por Materiales Peligrosos | 2.412 |
| **Interrupción Suministro Eléctrico** | **1.105** |
| **Falla de Conectividad Vial** (2 variantes) | **479** |
| Erupciones volcánicas | 239 |
| Remoción en Masa | 185 |

Filtrando por servicio/suministro/conectividad: **13.577 eventos**, que por
`Tipo Evento` se abren así:

| Tipo de falla | N° |
|---|---|
| Interrupción / Alteración Suministro **Eléctrico** (5 variantes) | 10.150 |
| Interrupción / Alteración **Agua Potable** (4 variantes) | 1.269 |
| Interrupción Suministros Básicos (genérico) | 1.242 |
| **Telecomunicaciones** (4 variantes) + Fibra Óptica | 340 |
| **Conectividad vial / accesibilidad** (5 variantes) | 311 |
| **Gas** | 57 |
| **Alcantarillado** | 24 |

**4. Familia** — **ESTADO-CONSECUENCIA**, en estado puro. Es el único dato de
todo el catastro que dice *qué se rompió, dónde y qué día*.

**5. Formato** — XLSX (Excel 2007+). Se abre sin librerías raras: `zipfile` +
`xml` de la librería estándar de Python bastan.

**6. Acceso** — **anónimo, sin registro. VERIFICADO** (descarga completa
exitosa, un solo GET).

**7. Granularidad territorial** — **COMUNA** (más provincia y región).
⚠️ Nombres de comuna en texto libre: aparecen **433 valores distintos** cuando
Chile tiene 346 comunas. Hay variantes de tildes, mayúsculas y localidades
coladas. **No trae código CUT.** Hay que normalizar contra la capa comunal
oficial (ver hallazgo H-13 del proyecto) antes de cruzar nada.

**8. Frecuencia** — publicación única (sep-2025) que cierra en 2024. No es un
flujo: es una foto de diez años. Habrá que pedir la actualización o construir
2025-en-adelante con la captura diaria de la Ficha 6.

**9. Historia disponible — DESDE QUÉ AÑO** ⭐
**Desde el 1 de enero de 2015 hasta el 31 de diciembre de 2024. Diez años
completos, sin agujeros.** Distribución anual verificada:

```
2015: 6.753   2018: 5.272   2021: 4.366   2024: 5.409
2016: 5.485   2019: 5.267   2022: 4.400
2017: 4.719   2020: 4.502   2023: 4.284
```

Lo notable es que **no hay años mudos**. Es la primera serie de todo el
catastro que cumple eso.

**10. Condiciones de uso**
`robots.txt` de `bibliogrd.senapred.gob.cl` (VERIFICADO): es un DSpace estándar.
Sólo prohíbe `/discover` y `/search-filter` (las búsquedas facetadas, que son
caras para el servidor). **No prohíbe `/handle/` ni `/bitstream/` ni `/oai/`.**
→ **Bajar este archivo de forma automatizada está permitido.** Igual: es una
descarga por publicación, no un rastreo. Citar SENAPRED como fuente.

**11. Para qué sirve en el modelo**
Es **la capa de validación completa**. Para cada predicción del tipo «tal
comuna, tal fecha, tal servicio en riesgo», este archivo dice si ocurrió.
Concretamente permite:
- **Curva ROC real** del `C_clim` y del `FEN`: comparar el peligro que declaró
  el modelo contra las fallas efectivamente registradas.
- **Tasa base por comuna**: cuántas veces al año se corta la luz en cada
  comuna. Sin esa tasa base, cualquier acierto del modelo puede ser azar.
- **Calibrar `PF` y `Pen`** con frecuencias observadas en vez de juicio experto.
- **Contrastar el criterio de Alerta Amarilla** («recursos locales superados»)
  contra `FRC`/`ICSGS` del MCSGS, usando albergados y albergues como proxy.

**12. Utilidad ★★★** — ¿sirve para validar una predicción sobre un activo?
**Sí, y es la única que lo hace con este nivel de detalle.** Tiene las tres
patas: fecha, comuna, y tipo de servicio caído.

**13. Riesgos y límites** (leerlos antes de confiar)
- **PRIVACIDAD.** La columna `Antecedentes Observaciones` (24.500 filas con
  texto) es prosa libre de operadores. Se encontraron **2 filas con patrón de
  RUT** y hay textos que describen personas fallecidas y su retiro por el
  Servicio Médico Legal. → **Regla para el proyecto: descartar la columna
  `Antecedentes Observaciones` en la ingesta, o pasarla por un filtro que borre
  RUT y nombres. Usar sólo conteos agregados por comuna.** El propio SENAPRED
  advierte que ese texto va «sin edición ni corrección».
- **Taxonomía sucia.** El mismo concepto aparece escrito de varias formas
  («Interrupción Suministro Eléctrico» / «Interrupción de suministro de
  electricidad» / «Alteración Suministro Eléctrico»). Hay que armar un
  diccionario de equivalencias a mano. Es trabajo de una tarde, no un problema.
- **`Sub Evento 1` está corrupta**: mezcla texto con números sueltos (199,
  37242, 116…) que parecen ser conteos de clientes afectados metidos en la
  columna equivocada. **No usarla** sin auditarla aparte.
- **Registra el servicio, no el activo.** Dice «se cortó la luz en Salamanca»,
  no «cayó la subestación X». El cruce con las 39 subestaciones hay que hacerlo
  por comuna y ventana temporal, no por identificador. Es una validación a
  nivel de comuna-servicio, no de pieza de fierro.
- **Sesgo de registro.** Un evento existe si un operador de la UAT lo escribió.
  Comunas con dirección regional activa reportarán más que comunas remotas. Es
  el mismo sesgo de un muestreo entomológico donde unas quebradas se visitan
  más que otras: hay que mirar el esfuerzo de muestreo antes de comparar.
- **Se congela en 2024.** No hay compromiso público de actualización.

---

# FICHA 2 ★★★ — SENAPRED · Eventos de Emergencia en Chile por Año 2015–2024

**1. Organismo y producto** — SENAPRED, mismo origen (UAT / sistema SID).
Es el mismo cuerpo de datos partido en diez archivos anuales.

**2. URL exacta y endpoint** — **VERIFICADO** (los 10 enlaces existen en la ficha)
- Ficha: `https://bibliogrd.senapred.gob.cl/handle/1671/7121`
- Patrón de archivo:
  `https://bibliogrd.senapred.gob.cl/bitstream/handle/1671/7121/AAAA_Eventos_Emergencia.xlsx?sequence=N&isAllowed=y`
  con `AAAA` = 2015…2024 y `sequence` = 1…10 en ese orden.

**3. Qué publica** — Diez XLSX, uno por año, con la misma estructura de la
Ficha 1 (temporal, geográfica, afectación a personas y viviendas, antecedentes).

**4. Familia** — ESTADO-CONSECUENCIA.
**5. Formato** — XLSX × 10.
**6. Acceso** — anónimo. **VERIFICADO** (enlaces listados; no se descargaron,
para no bajar dos veces lo mismo).
**7. Granularidad territorial** — comuna (idéntica a Ficha 1).
**8. Frecuencia** — publicación única, sep-2025.
**9. Historia disponible: desde 2015**, hasta 2024.
**10. Condiciones de uso** — igual que Ficha 1: permitido.
**11. Para qué sirve** — cargar año por año sin tragarse 11 MB de una, y sobre
todo **como control de integridad**: si la suma de los diez anuales no da los
50.457 del consolidado, alguno de los dos está mal. Vale la pena hacer esa
comprobación antes de construir encima.
**12. Utilidad ★★★** — misma sustancia que la Ficha 1.
**13. Riesgos y límites** — los mismos, más uno: **no está verificado que el
contenido sea idéntico al consolidado**. Podrían diferir en columnas o en
criterios de limpieza. Comprobarlo es barato y evita una sorpresa fea.

---

# FICHA 3 ★★★ — DesInventar Chile (UNDRR) — la mitad que falta: 1970–2014

**1. Organismo y producto**
UNDRR (Oficina de Naciones Unidas para la Reducción del Riesgo de Desastres),
proyecto DesInventar Sendai / LA RED. Base histórica de pérdidas y daños de
Chile, código de país `chl`.

**2. URL exacta y endpoint** — **VERIFICADO**
- Perfil de país (HTTP 200, con tablas de datos ya renderizadas):
  `https://www.desinventar.net/DesInventar/profiletab.jsp?countrycode=chl&lang=ES`
- Módulo de consulta (filtros por Tipo de Desastre, Región, Provincia, **Comuna**, Causa):
  `https://www.desinventar.net/DesInventar/main.jsp?countrycode=chl&lang=ES`
- Descarga de la base completa + mapas:
  `https://www.desinventar.net/DesInventar/download_base.jsp?countrycode=chl`
  (enlace presente en `https://www.desinventar.net/download.html` — **VERIFICADO
  que el enlace existe**; la descarga en sí **no** se ejecutó, y en una prueba
  previa el servidor no respondió dentro de 45 s, así que puede ser lento →
  **SUPUESTO** que la descarga funciona)
- Espejo: `https://db.desinventar.org/`

**3. Qué publica** — Metadatos del propio sitio, leídos textualmente:

> **Chile — DataCards: 12.896 · Period: 1970 – 2014**

Cada «ficha» (DataCard) es un evento con su lugar, fecha, tipo y efectos. Las
columnas de efectos verificadas en la tabla de perfil:

`Fichas · Muertos · Heridos · Desaparecidos · Viviendas Destruidas · Viviendas
Afectadas · Afectados · Damnificados · Reubicados · Evacuados · Pérdidas $USD ·
Pérdidas $Local ·` **`Centros Educativos · Centros Médicos · Daños cultivos Ha ·
Ganado · Daños en vías Mts`**

Las cuatro últimas son **infraestructura contada explícitamente**: colegios
dañados, centros de salud dañados y **metros de vía dañados**.

Totales por tipo de evento (extracto verificado del perfil):

| Evento | Fichas | Muertos | Viviendas destr. | C. Educ. | C. Médicos | Vías (m) |
|---|---|---|---|---|---|---|
| EARTHQUAKE | 461 | 909 | 575.233 | 36 | 42 | 11.218 |
| ALLUVION | 83 | 135 | 1.495 | 4 | 27 | 39.654 |
| DROUGHT | 817 | 10 | — | — | — | — |
| ACCIDENT | 273 | 395 | 232 | 1 | 25 | — |

Taxonomía de amenazas disponible (36 tipos), incluidas `ALLUVION`, `FLOOD`,
`LANDSLIDE`, `FORESTFIRE`, `DROUGHT`, `STRONGWIND`, `HEATWAVE`, `STRUCTURE`,
`ELECTRICSTORM`. Y una lista de **causas** que incluye `SHORTCIRCUIT`,
`FAILURE`, `DETERIORATION`, `DESIGNERROR`, `ENSO_KID` / `ENSO_GIRL` (¡El Niño y
La Niña como causa codificada — engancha directo con Cosmoclima!).

**4. Familia** — ESTADO-CONSECUENCIA.

**5. Formato** — Base MySQL/SQLite exportable + mapas, desde `download_base.jsp`.
Consulta interactiva vía JSP. También exporta Excel desde la interfaz
(«bájelo como Excel» aparece en el perfil).

**6. Acceso** — **anónimo, sin registro. VERIFICADO** para consulta y perfil.
**SUPUESTO** para la descarga masiva de la base.

**7. Granularidad territorial** — **COMUNA. VERIFICADO**: el formulario de
consulta ofrece los cuatro niveles *Region / Provincia / Comuna / Causa* como
filtros independientes. (El perfil de país agrega a región, pero eso es la
vista resumida, no el dato.)

**8. Frecuencia** — **congelada**. La base cierra en 2014 y no se actualiza.

**9. Historia disponible: DESDE 1970** ⭐ — 45 años, hasta 2014.

**10. Condiciones de uso**
`desinventar.net` **no tiene `robots.txt`** (la petición devuelve una página de
error, no reglas) → **no hay prohibición explícita de rastreo**. El software es
Apache-2; las bases son «contribuciones de sus propias organizaciones» que los
gobiernos autorizaron a compartir públicamente. → uso permitido con **cita
obligatoria a UNDRR/DesInventar**. Aun así: bajar la base **una vez** desde
`download_base.jsp`, nunca raspar el JSP consulta por consulta (es lento y
descortés).

**11. Para qué sirve en el modelo**
**Es la otra mitad del reloj.** DesInventar cubre **1970–2014**; SENAPRED cubre
**2015–2024**. Juntas dan **55 años continuos de registro de desastres a nivel
de comuna en Chile**, sin traslape y sin hueco. Eso convierte el proyecto de
«validable en 10 años» a «validable en medio siglo».
Además, es la única fuente con **daño a infraestructura contado en unidades
físicas** (metros de vía, número de escuelas y centros médicos), que es
exactamente el tipo de variable que el modelo pretende predecir.
Y sirve de **serie larga para estimar períodos de retorno** por comuna: cuántas
veces en 45 años esta comuna tuvo aluvión, y con qué daño.

**12. Utilidad ★★★** — ¿sirve para validar una predicción sobre un activo?
**Sí**, y para el pasado profundo es la única. Con la salvedad de que valida
categorías de activo (escuelas, centros médicos, vías), no activos con nombre.

**13. Riesgos y límites**
- **Se detiene en 2014.** Para lo reciente no sirve; ahí manda la Ficha 1.
- **Metodología distinta a SENAPRED.** DesInventar se construyó en buena parte
  con **prensa y reportes institucionales retrospectivos**; SENAPRED con partes
  operativos. **Empalmarlas sin advertirlo produciría un salto artificial en
  2015** — exactamente el mismo error que ya se detectó en Cosmoclima cuando la
  lluvia cambiaba de fuente en 2019 (Ronda 15). **Tratar 1970–2014 y 2015–2024
  como dos series con calibración distinta, no como una sola.**
- **Sesgo hacia lo grande y lo antiguo.** Antes de los años 90 sólo quedó en la
  prensa lo espectacular. La densidad de registro sube con el tiempo por razones
  documentales, no porque haya más desastres.
- **Nomenclatura comunal de 1970–2014**, con comunas que cambiaron de nombre,
  se fusionaron o se crearon (Ñuble es región recién desde 2018). Otro trabajo
  de normalización territorial.
- La descarga masiva no está verificada; el servidor respondió lento.

---

# FICHA 4 ★★ — BiblioGRD · OAI-PMH (el canal automático hacia SENAPRED)

**1. Organismo y producto**
SENAPRED — *BiblioGRD, Biblioteca Digital SENAPRED* (ex Repositorio Digital
ONEMI). Es un **DSpace**, el software estándar de repositorios académicos, lo
que significa que trae de fábrica un protocolo de cosecha automática.

**2. URL exacta y endpoint** — **VERIFICADO** (los tres respondieron HTTP 200)
- Portal: `https://bibliogrd.senapred.gob.cl/`
- **Endpoint OAI-PMH: `https://bibliogrd.senapred.gob.cl/oai/request`**
  - `?verb=Identify` → identidad del repositorio
  - `?verb=ListSets` → **100 colecciones**
  - `?verb=ListRecords&metadataPrefix=oai_dc&set=<setSpec>` → registros
- Colecciones útiles (setSpec verificados):
  - `col_2012_21` — **Estadísticas e Indicadores** (61 registros) ← acá viven las Fichas 1, 2 y 5
  - `col_2012_16` — Informes
  - `col_2012_25` — Emergencias (**devuelve 0 registros: está vacía**)
  - `col_2012_37` — Manejo de desastres
  - `col_2012_1844` — Catastro de Estudios
  - `col_123456789_4201` — History Maps de Desastres

**3. Qué publica** — Metadatos Dublin Core (título, fecha, descripción,
materias, región, identificador/handle) de todo lo que SENAPRED sube: informes,
planes, cartografía, estadísticas. Desde el metadato se arma la URL del archivo:
`/bitstream/handle/<handle>/<nombre>?sequence=N&isAllowed=y`.

**4. Familia** — ESTADO-CONSECUENCIA (es la puerta a los registros de daño),
aunque el repositorio en sí mezcla familias.

**5. Formato** — XML OAI-PMH 2.0 (`oai_dc`). Granularidad de fecha
`YYYY-MM-DDThh:mm:ssZ`, con soporte de `from`/`until` para cosechas
incrementales.

**6. Acceso** — **anónimo, sin registro, sin API key. VERIFICADO.**

**7. Granularidad territorial** — no aplica al protocolo; los ítems traen
metadato `Región` (los 16 nombres regionales oficiales).

**8. Frecuencia** — continua. Se puede cosechar sólo lo nuevo con
`&from=2026-08-16` en vez de bajar todo cada vez.

**9. Historia disponible: desde 2012-05-07** (`earliestDatestamp` declarado por
el propio repositorio). Ojo: eso es la fecha en que se *cargó* el documento más
antiguo, no la fecha del contenido — hay ítems de 1997 cargados en 2012.

**10. Condiciones de uso**
`robots.txt` **VERIFICADO**: prohíbe `/discover` y `/search-filter`. **No
prohíbe `/oai/`, `/handle/` ni `/bitstream/`.** OAI-PMH está *diseñado* para ser
cosechado por máquinas. → **automatización permitida**, con la cortesía de
espaciar las peticiones. Contacto declarado del repositorio:
`repositoriodigitalonemi@gmail.com`.

**11. Para qué sirve en el modelo**
Es **el radar que avisa cuándo SENAPRED publica un consolidado nuevo**. Una
cosecha semanal de `col_2012_21` con `&from=` detecta automáticamente si
aparece el «Consolidado 2015–2025». Sin esto, el proyecto se entera por
casualidad. Analogía: en vez de ir todos los días a mirar si floreció, se pone
una trampa que avisa sola.

**12. Utilidad ★★** — no valida nada por sí misma, pero **es el único canal
programático y estable que tiene SENAPRED**. Sin él, todo lo demás es a mano.

**13. Riesgos y límites**
- El portal declara estar **«en mantenimiento»** en su propia página de inicio.
- Los `dc:identifier` traen URLs históricas rotas (`repositoriodigitalonemi.cl`,
  `onemi.gov.cl`, incluso `http://localhost:80/xmlui/...`). **Hay que
  reconstruir la URL desde el handle, no confiar en el identifier.**
- `deletedRecord: transient` → el repositorio **no garantiza** avisar de
  borrados. Un documento puede desaparecer sin rastro.
- El `repositoryIdentifier` es `localhost` (mal configurado). Señal de que la
  instalación está a medio afinar: puede romperse sin aviso.

---

# FICHA 5 ★★ — SENAPRED/ONEMI · Informes Estadísticos anuales y semestrales (1997→)

**1. Organismo y producto**
ONEMI (hasta 2022) y SENAPRED (desde 2023) — serie de informes estadísticos
institucionales: compendios, boletines mensuales, informes semestrales y
anuales.

**2. URL exacta y endpoint** — **VERIFICADO** vía OAI, colección `col_2012_21`.
Títulos y handles confirmados (muestra):

| Documento | Handle / identificador |
|---|---|
| Compendio de estadísticas de emergencia **1997–2000** | `handle/2012/52` |
| Compendio estadístico año **2004** | `handle/2012/59` |
| Consolidados de informes técnicos y fondos de emergencia **2014** | `handle/2012/1685` |
| Boletín Estadístico de Emergencias N°1 (ene-2015) … N°6 (jun-2015) | col. `col_2012_21` |
| Informes Estadísticos Trimestrales ONEMI N°2 y N°3 (2015) | col. `col_2012_21` |
| 1° Informe Estadístico Semestral **2017** | `handle/2012/1789` |
| 1° Informe Estadístico Semestral **2018** | `handle/2012/1840` |
| Informe Estadístico Semestral ONEMI **2019** | `handle/123456789/3347` |
| Informe Estadístico Anual ONEMI **2019** | `handle/123456789/4106` |
| Informe Estadístico Anual ONEMI **2020** | `handle/123456789/4153` |
| Informe Estadístico Anual ONEMI **2021** | `handle/123456789/5342` |
| **Informe Estadístico 2024** (SENAPRED) | `handle/1671/7051` |
| Inundaciones Región Metropolitana **1982/1987** | `handle/2012/...` |

URL de ficha: `https://bibliogrd.senapred.gob.cl/handle/<handle>`

**3. Qué publica** — Recopilación cuantitativa anual/semestral de emergencias
registradas, más actividad institucional (cursos, simulacros, planes).

**4. Familia** — ESTADO-CONSECUENCIA (mezclada con gestión institucional).

**5. Formato** — **PDF**. Tablas dentro del PDF, no datos sueltos.

**6. Acceso** — anónimo. **VERIFICADO** a nivel de metadato y de ficha.
**SUPUESTO** que todos los PDF están abiertos (se comprobó el patrón
`/bitstream/` en dos ítems, no en los 61).

**7. Granularidad territorial** — típicamente **REGIÓN**, a veces dirección
regional. **No llega a comuna** de forma sistemática. Es su gran debilidad.

**8. Frecuencia** — irregular: hubo boletines mensuales (2015), trimestrales
(2015), semestrales (2017-2019) y anuales (2019-2024). La serie no es pareja.

**9. Historia disponible: desde 1997** (el compendio 1997-2000 es el más
antiguo). Con huecos importantes entre 2005 y 2014.

**10. Condiciones de uso** — igual que Ficha 4: permitido.

**11. Para qué sirve en el modelo**
Dos usos, ninguno de validación fina:
- **Control de totales.** Comparar el total anual de eventos del Consolidado
  (Ficha 1) contra el que reporta el Informe Estadístico del mismo año. Si no
  cuadran, hay que entender por qué **antes** de usar el consolidado.
- **Extender la serie hacia atrás de 2015**, a nivel regional, para el período
  1997–2014 donde DesInventar y estos informes se pueden contrastar.

**12. Utilidad ★★** — ¿valida una predicción sobre un activo? **Sólo
parcialmente**: al ser regional y estar en PDF, no permite decir «esta comuna,
este día». Sirve de auditoría, no de validación.

**13. Riesgos y límites**
- **PDF**: extraer tablas de PDF es frágil y produce errores silenciosos.
  Si se hace, hay que verificar los totales a mano.
- Serie discontinua y con cambios de criterio entre ONEMI y SENAPRED.
- Mezcla emergencias con actividad administrativa: hay que separar el grano.

---

# FICHA 6 ★★ — SENAPRED · Alertas vigentes (servicio ArcGIS público, por comuna)

**1. Organismo y producto**
Dato originado en SENAPRED, **publicado en ArcGIS Online por `minsal_admin`
(Ministerio de Salud)** en el tablero *ALERTAS SENAPRED*. Es decir: **es una
republicación de un tercero del Estado, no el canal oficial de SENAPRED.**

**2. URL exacta y endpoint** — **VERIFICADO**
- Tablero: `https://www.arcgis.com/apps/dashboards/bdc345e01e324af490800634d0f0e3a5`
  (creado 16-jun-2025, modificado **14-ago-2026** → está vivo)
- Mapas web que consume: `WM_METEOROLOGICAS`, `WM_BIOLOGICAS`, `WM_VOLCANICAS`,
  `WM_FORESTALES`
- **Servicios de datos consultables** (organización `CNzkI2T3GmfwkaAR`):
  ```
  https://services3.arcgis.com/CNzkI2T3GmfwkaAR/arcgis/rest/services/METEOROLOGICAS_AMARILLA/FeatureServer/0
  https://services3.arcgis.com/CNzkI2T3GmfwkaAR/arcgis/rest/services/METEOROLOGICAS_ROJA/FeatureServer/0
  https://services3.arcgis.com/CNzkI2T3GmfwkaAR/arcgis/rest/services/METEOROLOGICAS_VERDE/FeatureServer/0
  ```
  (existen equivalentes para biológicas, volcánicas y forestales, no listados
  aquí porque no se abrieron esos mapas — **SUPUESTO**)
- Consulta verificada:
  `.../FeatureServer/0/query?where=1=1&outFields=CUT_COM,COMUNA,REGION,TIPO_ALERT,CAUSALIDAD,FECHA_INI&returnGeometry=false&f=json`

**3. Qué publica** — Un polígono **por comuna bajo alerta**, con estos campos
(verificados): `CUT_REG`, `CUT_PROV`, **`CUT_COM`**, `REGION`, `PROVINCIA`,
`COMUNA`, `SUPERFICIE`, `TIPO_ALERT`, `CAUSALIDAD`, `FECHA_INI`.

Registro real obtenido el 15-ago-2026:
```
CUT_COM: 04204 · COMUNA: Salamanca · REGION: Coquimbo
TIPO_ALERT: Amarilla
CAUSALIDAD: Evento meteorológico (mod. even. 07-08-2026)
FECHA_INI: 04-08-2026
```
En ese momento había **177 comunas con Alerta Amarilla meteorológica**.

**4. Familia** — frontera entre AMENAZA y ESTADO-CONSECUENCIA. La alerta no es
daño, pero **es la respuesta oficial del Estado**, y el criterio de la Amarilla
(«recursos locales superados») es una medida de capacidad, no de amenaza.

**5. Formato** — GeoJSON / Esri JSON vía ArcGIS REST. Polígonos comunales.

**6. Acceso** — **anónimo, sin token. VERIFICADO** (dos consultas exitosas).

**7. Granularidad territorial** — **COMUNA, y con código `CUT_COM`.**
Esto es oro: es la única fuente del catastro que trae el **código oficial**, no
sólo el nombre. Sirve de tabla de traducción para normalizar los nombres sucios
de la Ficha 1.

**8. Frecuencia** — se sobrescribe cuando cambian las alertas (el ítem se
modificó el 14-ago-2026). Fuentes de prensa mencionan refresco cada 6 h
(**SUPUESTO**, no verificado).

**9. Historia disponible: NINGUNA. Es una foto del presente.**
**VERIFICADO**: las 177 filas tienen `FECHA_INI` de agosto de 2026. La capa no
acumula; se reemplaza. **Todo lo anterior está perdido para siempre desde este
canal.**

**10. Condiciones de uso**
El servicio no declara `copyrightText` (campo vacío, verificado). Está publicado
como `access: public` en ArcGIS Online. → consulta permitida técnicamente, pero
**el dueño del dato (MINSAL republicando a SENAPRED) no declaró licencia**.
Antes de usarlo en un producto que alimente decisión pública, hay que pedir
confirmación a SENAPRED. Anotado como pendiente.

**11. Para qué sirve en el modelo** — dos cosas, ambas valiosas:
- **Construir la historia que no existe.** Una captura diaria automática de
  estos servicios crea, a partir de hoy, el archivo histórico de alertas por
  comuna que el país no publica. En un año habrá 365 fotos; en tres, una serie.
  Es la misma lógica del monitoreo entomológico: si nadie tomó la muestra, la
  muestra no existe — así que empezá a tomarla hoy.
- **Diccionario CUT ↔ nombre de comuna**, para limpiar la Ficha 1.

**12. Utilidad ★★** — hoy **no valida nada** (sin historia). Pero es la única
vía para que en el futuro sí valide, y a costo casi cero.

**13. Riesgos y límites**
- **No es la fuente oficial de SENAPRED**: es MINSAL republicando. Si MINSAL
  deja de mantenerlo, se apaga sin aviso. **No construir nada crítico encima.**
- Sin historia y **sin fecha de término** de la alerta (sólo `FECHA_INI`).
- `CAUSALIDAD` mezcla la causa con notas de modificación en texto libre
  («Evento meteorológico (mod. even. 07-08-2026)»).
- Se verificaron sólo las capas meteorológicas. Las demás son supuesto.

---

# FICHA 7 ★ — SENAPRED · API de alertas del sitio oficial (AppSync) — CERRADA

**1. Organismo y producto**
SENAPRED — la aplicación de alertas del sitio institucional
(`senapred.cl/informate/alertas`). Se investigó porque era el objetivo del
encargo: encontrar el índice histórico de alertas declaradas.

**2. URL exacta y endpoint** — **VERIFICADO** (leyendo el JavaScript público)
- Página: `https://senapred.cl/informate/alertas` → devuelve **1.961 bytes** de
  cáscara HTML. Es una aplicación React: sin JavaScript no hay contenido.
- Bundle: `https://senapred.cl/static/js/main.e062748e.js` (3,1 MB)
- Configuración encontrada dentro del bundle, literal:
  ```
  aws_appsync_graphqlEndpoint:
    "https://rz2uv7ifxbgflh2bqmp6kmh4le.appsync-api.us-east-1.amazonaws.com/graphql"
  aws_appsync_authenticationType: "AMAZON_COGNITO_USER_POOLS"
  aws_project_region: "us-east-1"
  ```

**3. Qué publica** — Presumiblemente el listado de alertas declaradas que
alimenta la web. **No se pudo comprobar el contenido.**

**4. Familia** — ESTADO-CONSECUENCIA / AMENAZA.
**5. Formato** — GraphQL sobre AWS AppSync.
**6. Acceso** — **NO PÚBLICO. VERIFICADO por el propio código**: el tipo de
autenticación es `AMAZON_COGNITO_USER_POOLS`, o sea **exige un usuario y clave
de un pool de Cognito**. No es una API abierta.
→ **No se intentó autenticar, y no se debe.** Forzar la entrada a un servicio
que declara requerir credenciales sería acceso no autorizado a un sistema del
Estado. Queda documentado y ahí se detiene.
**7. Granularidad territorial** — desconocida.
**8. Frecuencia** — desconocida.
**9. Historia disponible** — **desconocida.** Es justamente el dato que se
buscaba y no se pudo obtener.
**10. Condiciones de uso** — `robots.txt` de `senapred.cl` **VERIFICADO**:
```
User-agent: *
Disallow: /wp-admin/
Allow: /wp-admin/admin-ajax.php
Sitemap: https://qa2.senapred.cl/sitemap.xml
```
No prohíbe el sitio público. Pero el `robots.txt` **apunta al sitio de pruebas
`qa2.senapred.cl`** — un error de configuración de ellos. **No usar ese
dominio**: es un ambiente de QA, el dato de ahí no es oficial.
La API REST de WordPress (`/wp-json/`) está **desactivada**: devuelve la portada
en HTML, no JSON (**VERIFICADO**).
**11. Para qué sirve** — hoy, nada. Documentado para no repetir la búsqueda.
**12. Utilidad ★** — no accesible.
**13. Riesgos y límites** — **La vía correcta acá no es técnica sino
institucional: pedirle a SENAPRED el histórico de alertas declaradas por
oficio o por Ley de Transparencia.** Es un dato que existe (lo usan ellos
mismos) y que el proyecto necesita. Ese pedido debería incluir: fecha de inicio,
fecha de término, nivel, causa, y comunas incluidas, de 2010 a la fecha.

*(Extra verificado: el sitemap de noticias `https://senapred.cl/post-sitemap.xml`
lista **359 notas de prensa desde 2010** — desde la época ONEMI. Es texto, no
dato, pero permite fechar eventos si algún día hace falta.)*

---

# FICHA 8 ★ — Informes ALFA y DELTA (los originales) — NO PÚBLICOS

**1. Organismo y producto**
SENAPRED / municipios — Informes **ALFA** (evaluación preliminar de daños,
levantado por el municipio) y **DELTA** (profundización, necesidades y
capacidades), instrumentos del **Plan DEDO$** del Sistema de Evaluación de Daños
y Necesidades.

**2. URL exacta y endpoint** — **NO EXISTE endpoint público.**
Lo que sí se verificó:
- Los formularios y su metodología aparecen citados en planes comunales de
  emergencia (ej. Municipalidad de San Nicolás) y en oficios a la Cámara de
  Diputados.
- El sistema que los recibe se llama **SID** — declarado explícitamente en la
  nota aclaratoria del Consolidado (Ficha 1): *«informes de incidentes y
  emergencias registrados en el sistema SID»*.
- **No hay repositorio público de informes ALFA/DELTA individuales.**
  Se buscó en: sitemaps de senapred.cl, las 100 colecciones de BiblioGRD,
  y datos.gob.cl. Nada.

**3. Qué publica** — nada, hacia afuera.
**4. Familia** — ESTADO-CONSECUENCIA (sería la fuente primaria ideal).
**5. Formato** — formulario interno (SID).
**6. Acceso** — **NO PÚBLICO. VERIFICADO** por ausencia en las tres búsquedas.
**7. Granularidad territorial** — comuna (es un instrumento municipal).
**8. Frecuencia** — por evento.
**9. Historia disponible** — el Plan DEDO$ opera desde los años 90; el sistema
SID es más reciente. **Sin acceso, no verificable.**
**10. Condiciones de uso** — no aplica.
**11. Para qué sirve en el modelo**
**Indirectamente, ya está sirviendo**: el Consolidado 2015–2024 (Ficha 1) *es*
la agregación de estos informes. O sea, el proyecto ya tiene el jugo aunque no
tenga la fruta.
**12. Utilidad ★** — no accesible directamente.
**13. Riesgos y límites**
- ⚠️ **PRIVACIDAD, en serio.** Los ALFA/DELTA contienen **nóminas nominadas de
  damnificados y fallecidos** (nombre, RUT, domicilio). Es dato personal
  sensible bajo la ley chilena. **Si alguna vez SENAPRED entregara estos
  informes al proyecto, la regla es agregarlos a conteo por comuna en la
  ingesta y NO persistir ningún registro individual.** Conviene dejar esto
  escrito en la solicitud misma: «se pide dato agregado por comuna, sin
  identificación de personas». Pedir menos de lo que se puede es la forma
  correcta de trabajar con esto.
- Si se pide por Transparencia, pedir el **agregado**, no los formularios.

---

# FICHA 9 ★ — datos.gob.cl (portal de datos abiertos del Estado) — CASI VACÍO

**1. Organismo y producto**
Ministerio Secretaría General de la Presidencia — portal nacional de datos
abiertos, sobre software **CKAN**.

**2. URL exacta y endpoint** — **VERIFICADO**, la API responde
- `https://datos.gob.cl/api/3/action/package_search?q=<término>&rows=N`
- `https://datos.gob.cl/api/3/action/organization_list?all_fields=true`

**3. Qué publica** — Resultados verificados de la búsqueda:

| Consulta | Resultados | Comentario |
|---|---|---|
| `emergencia` | 55 | **casi todo son urgencias hospitalarias** (Hospital Roberto del Río, Fricke, Magallanes…). Ruido. |
| `senapred` | **0** | — |
| `onemi` | 1 | «Plan de Protección Civil Municipal ante Tsunami» (ZIP) |
| `desastre` | 4 | ninguno es serie de eventos |
| `riesgo de desastres` | 23 | mayoría irrelevante (bancos, ISL) |

Y el dato más contundente, verificado en `organization_list`:
> La organización **«Servicio Nacional de Prevención y Respuesta ante
> Desastres» tiene exactamente 1 (un) conjunto de datos publicado.**

**4. Familia** — transversal, pero en la práctica no aporta a ESTADO-CONSECUENCIA.
**5. Formato** — JSON (API CKAN); recursos en CSV/XLSX/ZIP.
**6. Acceso** — **anónimo, sin API key. VERIFICADO.**
**7. Granularidad territorial** — variable, mayormente regional.
**8. Frecuencia** — irregular.
**9. Historia disponible** — no aplica: no hay serie de emergencias.
**10. Condiciones de uso** — ⚠️ **`robots.txt` VERIFICADO**:
```
User-agent: *
Disallow: /api/
Crawl-Delay: 10
```
**El portal prohíbe explícitamente a los robots la ruta `/api/` y pide 10
segundos entre peticiones.** Hay una tensión real: la API de CKAN existe para
uso programático, pero el `robots.txt` la marca como no rastreable.
→ **Criterio para el proyecto: consultas puntuales y espaciadas (≥10 s) sí;
rastreo sistemático no.** Es la lectura conservadora, y para este portal no
cuesta nada porque casi no hay dato que traer.
**11. Para qué sirve en el modelo** — poco. Sirve como **chequeo periódico**
(¿publicó SENAPRED algo nuevo acá?), y para descubrir datasets de otros
organismos.
**12. Utilidad ★** — no valida ninguna predicción hoy.
**13. Riesgos y límites** — **el hallazgo importante es negativo y hay que
decirlo claro: el portal oficial de datos abiertos de Chile NO tiene el registro
de emergencias del país.** El dato bueno está en la biblioteca de SENAPRED. Si
alguien buscara sólo en datos.gob.cl concluiría que Chile no lleva registro, y
se equivocaría.

---

# FICHA 10 ★ — EM-DAT (CRED / UCLouvain)

**1. Organismo y producto**
Centre for Research on the Epidemiology of Disasters (CRED), Université
catholique de Louvain — *EM-DAT, The International Disaster Database*.

**2. URL exacta y endpoint** — **VERIFICADO parcialmente**
- Plataforma: `https://public.emdat.be/` (responde; es una app Next.js con
  menú **Login / Register**)
- Documentación: `https://doc.emdat.be`
- Proyecto: `https://www.emdat.be`
- `https://public.emdat.be/robots.txt` → **HTTP 404** (no hay reglas; el sitio
  devuelve su página «Page not found»)

**3. Qué publica** — Desastres de todo el mundo con umbral de inclusión
(típicamente ≥10 muertos, ≥100 afectados, declaración de emergencia o llamado
de ayuda internacional): fecha, tipo, país, muertos, afectados, pérdidas USD.

**4. Familia** — ESTADO-CONSECUENCIA.
**5. Formato** — extractos descargables (XLSX/CSV) desde la plataforma tras
iniciar sesión.
**6. Acceso** — **REGISTRO OBLIGATORIO. VERIFICADO**: la barra de navegación
ofrece «Login» y «Register» como pasos previos a las herramientas. Es gratuito
para uso académico y no comercial (**SUPUESTO** en cuanto a los términos
exactos: no se leyeron por no crear cuenta).
**7. Granularidad territorial** — **PAÍS**, con localización textual del evento.
**No llega a comuna.**
**8. Frecuencia** — continua.
**9. Historia disponible: desde 1900.** Es la serie más larga del catastro.
**10. Condiciones de uso** — sin `robots.txt`. Los términos se aceptan al
registrarse; hay política de citación obligatoria. **No automatizar sin leer
esos términos con la cuenta creada.**
**11. Para qué sirve en el modelo**
**Contraste externo, no validación.** Sirve para responder «¿los grandes
desastres que registra SENAPRED aparecen también en la base mundial?». Si un
evento chileno grande está en SENAPRED y no en EM-DAT, o al revés, hay algo que
entender. Es el control de que la libreta chilena no esté sesgada.
**12. Utilidad ★** — ¿valida una predicción sobre un activo? **No.** Sin comuna
y con umbral de inclusión alto, se pierde exactamente lo que el proyecto quiere
predecir: la falla localizada.
**13. Riesgos y límites**
- **Umbral de inclusión**: los eventos chicos —que son la mayoría de las fallas
  de infraestructura— no entran. De los 50.457 eventos de SENAPRED,
  probablemente menos de 100 califican para EM-DAT.
- Granularidad de país: inútil para trabajo comunal.
- Requiere cuenta: dependencia externa y término de uso a revisar.

---

# FICHA 11 ★ — CIGIDEN · Repositorio del Desastre

**1. Organismo y producto**
CIGIDEN — Centro de Investigación para la Gestión Integrada del Riesgo de
Desastres (ANID/UC). *Repositorio del Desastre*.

**2. URL exacta y endpoint** — **VERIFICADO**
- Hub: `https://repositorio-del-desastre-cigiden-cigiden.hub.arcgis.com/`
  (HTTP 200)
- Organización ArcGIS: `UeyripQFTg6pfUe5`
- **Prueba clave**:
  `https://services.arcgis.com/UeyripQFTg6pfUe5/arcgis/rest/services?f=json`
  → HTTP 200 pero **`services: []` — cero servicios públicos.**

**3. Qué publica** — «Relatos de datos» (*data storytelling*) sobre desastres
investigados en terreno: inundaciones 2023, incendios 2024, aluvión de Quebrada
de Macul (1993). Según su propia difusión, incorpora datos aportados por
SENAPRED.

**4. Familia** — ESTADO-CONSECUENCIA (narrativa).
**5. Formato** — ArcGIS StoryMaps (páginas web). **No hay capas descargables**
según la prueba anterior.
**6. Acceso** — anónimo para leer. **VERIFICADO.**
**7. Granularidad territorial** — por evento y sitio de estudio, no sistemática.
**8. Frecuencia** — por publicación de investigación.
**9. Historia disponible: desde 2023** en cuanto a eventos cubiertos, con
retrospectivas puntuales (Macul 1993).
**10. Condiciones de uso** — no se halló declaración de licencia. Es material
académico: **citar y pedir permiso antes de reutilizar** (**SUPUESTO**).
**11. Para qué sirve en el modelo**
**Como contraparte académica, no como fuente de datos.** CIGIDEN es el centro
del país que estudia esto; el valor está en el contacto, no en el archivo.
También sirve para **entender casos** en profundidad cuando el modelo falle en
un evento concreto y haya que averiguar por qué.
**12. Utilidad ★** — no valida predicciones: sin serie ni capas descargables.
**13. Riesgos y límites**
- Cero servicios de datos públicos (verificado). Todo el contenido es relato.
- Cobertura por conveniencia (los eventos que investigaron), no sistemática.

---

# FICHA 12 ★ — INE · clasificación estadística de emergencias y desastres — NO HALLADA

**1. Organismo y producto**
Instituto Nacional de Estadísticas (INE) — supuesta clasificación estadística de
emergencias y desastres, citada en el glosario oficial de GRD de 2021.

**2. URL exacta y endpoint** — **NO ENCONTRADA.**
Se revisó `https://www.ine.gob.cl/herramientas/portal-de-mapas/clasificaciones`:
la página existe y trata de clasificaciones estadísticas, pero **sólo de
actividad económica, mercado laboral y producción. Ninguna de emergencias ni
desastres.**
El INE sí aparece en datos.gob.cl con un «Informe Anual de Medioambiente»
(sin recursos descargables asociados).

**3. Qué publica** — respecto de emergencias: nada localizable.
**4. Familia** — sería una norma de clasificación, no un dato.
**5. Formato** — n/a.
**6. Acceso** — **NO VERIFICABLE** (no se halló el producto).
**7-9.** n/a.
**10. Condiciones de uso** — n/a.
**11. Para qué sirve en el modelo**
Si existiera, serviría para **una sola cosa pero importante**: darle al proyecto
un vocabulario oficial de tipos de evento, en vez de inventar uno o heredar la
taxonomía sucia de SENAPRED. Es el equivalente a tener una clave taxonómica
publicada en vez de nombrar las especies uno mismo.
**12. Utilidad ★** — no hallada.
**13. Riesgos y límites**
- **Pendiente concreto**: escribir a `ine@ine.gob.cl` preguntando por la
  clasificación de emergencias y desastres citada en el glosario 2021.
  Mientras tanto, **la taxonomía de trabajo debe ser la del propio SENAPRED**
  (`Origen / Clase / Tipo / Sub-evento` de la Ficha 1), limpiada, porque es la
  que efectivamente usa el país para registrar.

---

---

## Resumen operativo: los tres pasos que valen la pena

**Paso 1 — bajar la libreta.**
Un solo archivo, un solo GET, y el proyecto pasa de infalsable a validable:
`https://bibliogrd.senapred.gob.cl/bitstream/handle/1671/7120/Eventos_Emergencia_2015_2024.xlsx?sequence=1&isAllowed=y`
Al cargarlo: **descartar la columna `Antecedentes Observaciones`** (trae datos
de personas) y **descartar `Sub Evento 1`** (está corrupta).

**Paso 2 — normalizar las comunas.**
El Excel trae 433 nombres distintos para 346 comunas, sin código. La tabla de
traducción está gratis en el servicio ArcGIS de la Ficha 6, que sí trae
`CUT_COM`. Es el mismo problema de dos geografías que ya está descrito en
`INTEGRACION_SENAPRED.md` §4, sólo que acá es entre dos maneras de escribir la
misma comuna.

**Paso 3 — empezar a guardar el presente.**
La alerta de hoy se borra mañana. Una captura diaria del servicio ArcGIS
(Ficha 6) crea, desde cero y sin costo, el archivo histórico de alertas que
Chile no publica. Dentro de un año valdrá oro; hoy vale un cron.

## Lo que el catastro NO consiguió, y cómo conseguirlo

| Falta | Vía |
|---|---|
| Histórico de alertas declaradas 2010-2026 (fecha, nivel, comunas) | **Oficio o Ley de Transparencia a SENAPRED.** La API existe pero está cerrada con Cognito. |
| Actualización del Consolidado más allá de 2024 | Misma solicitud; o cosecha OAI (Ficha 4) para detectar cuándo lo publiquen. |
| Registro con **nombre de activo** («cayó la subestación X») | No existe en fuente pública de emergencia. Habría que buscarlo del lado del **regulador sectorial** (fuera de esta familia: lo cubre el agente de servicios básicos). |
| Clasificación oficial de tipos de emergencia | Consulta al INE. |

## Nota metodológica sobre privacidad

Se aplicó la regla del encargo. Concretamente:
- Se inspeccionó el Excel **contando patrones**, sin extraer ni copiar ningún
  dato personal. Se detectaron 2 filas con patrón de RUT y textos que describen
  personas fallecidas: **ninguno se reproduce en este documento**.
- No se intentó autenticarse contra la API AppSync de SENAPRED, pese a tener
  el identificador del pool de Cognito a la vista en el código público.
- No se pidieron ni buscaron informes ALFA/DELTA individuales, que contienen
  nóminas nominadas.
- Recomendación para la ingesta: **el proyecto trabaja con conteos por comuna.
  Ninguna columna de texto libre entra a la base.**
