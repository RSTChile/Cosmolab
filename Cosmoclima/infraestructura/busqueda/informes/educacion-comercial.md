# Catastros públicos georreferenciados · sectores EDUCACIÓN y COMERCIAL

Fecha de la búsqueda: **2026-08-25**
Lote: los 21 ítems físicos sin catastro del sector **Educación** y los 20 del sector **Comercial**
(según `busqueda/CATALOGO_FALTANTES.txt`).

Todo lo que sigue fue **verificado con petición real** (HTTP + conteo de registros + conteo de
coordenadas). Lo que no se pudo verificar está marcado como **NO VERIFICADO** y no se reporta como
fuente buena.

Siglas usadas: **MICR** = Matriz de Infraestructura Crítica y Riesgo · **IDE Chile** = Infraestructura
de Datos Geoespaciales de Chile · **SNIT** = Sistema Nacional de Información Territorial ·
**WFS** = Web Feature Service (servicio OGC que entrega los datos vectoriales, no sólo la imagen) ·
**CSW** = Catalogue Service for the Web (el buscador de metadatos del Geoportal) ·
**JUNJI** = Junta Nacional de Jardines Infantiles · **SCJ** = Superintendencia de Casinos de Juego ·
**SERNATUR** = Servicio Nacional de Turismo · **MINDEP** = Ministerio del Deporte ·
**UTM** = Universal Transverse Mercator (sistema de coordenadas en metros, no en grados).

---

## Hallazgo transversal: el Geoportal de Chile sí tiene WFS, y eso cambia el método

`geoportal.cl` (IDE Chile / SNIT, Ministerio de Bienes Nacionales) publica **dos cosas que este
proyecto todavía no estaba usando**:

1. Un **catálogo CSW** en `https://geoportal.cl/csw` con **978 registros** de metadatos. Se puede
   barrer entero con `GetRecords` en 10 peticiones y buscar por palabra. Es el índice de qué capas
   existen en el Estado chileno.
2. Un **GeoServer con WFS por capa** en `https://geoportal.cl/geoserver/<Workspace>/wfs`, que
   entrega **GeoJSON en EPSG:4326** con paginación (`count` + `startIndex`) y reproyección al vuelo
   (`srsName=EPSG:4326`). No hay ArcGIS REST aquí: `geoportal.cl/arcgis/rest/services` devuelve
   **HTTP 404** (verificado).
   - Ojo: el WFS **exige workspace**. `https://geoportal.cl/geoserver/wfs` sin workspace devuelve
     `ExceptionText: No workspace specified`.
   - El servidor corta la conexión si se le pide mucho: con `count=1000` en paralelo se cae con
     `RemoteDisconnected` / `Connection reset by peer`. Con `count=500`, 1 segundo de pausa,
     reintentos con espera creciente y caché por página, baja completo.

`robots.txt` de `geoportal.cl`: **HTTP 200**, contenido `User-agent: *` / `Disallow:` → **todo
permitido**, sin `Crawl-delay`.

---

# FUENTES VERIFICADAS

## 1. Jardines Infantiles JUNJI (2024)

- **Ítems MICR que poblaría**: refuerza **445 Escuelas Rurales** (subconjunto) y el nivel parvulario
  en general. *Ver la advertencia del cierre: en la lista de faltantes de Educación no hay un ítem
  llamado "Jardines Infantiles" ni "Educación Parvularia".*
- Organismo · producto: **JUNJI**, vía IDE Chile · capa `jardines_infantiles_junji_2024`
  (unidades educativas en operación, abril 2024)
- URL del servicio:
  `https://geoportal.cl/geoserver/Jardines_Infantiles_JUNJI/wfs?service=WFS&version=2.0.0&request=GetFeature&typeNames=Jardines_Infantiles_JUNJI%3Ajardines_infantiles_junji_2024&outputFormat=application/json&srsName=EPSG:4326&count=1000&startIndex=N`
- Formato · registros · coordenadas: **GeoJSON · 3.015 · 3.015 con coordenada (100 %)**
- Campos: `nombre`, `direccion`, `region`, `comuna`, `cod_junji`, `programa_e`, `modalidad`,
  `capacidad_` (capacidad de atención), `latitud`, `longitud`
- robots.txt: HTTP 200 · `Disallow:` vacío → permitido
- Crudo: `busqueda/crudo/junji/2026-08-25/jardines_junji_2024.geojson`
- **Veredicto: SIRVE** — catastro nacional completo, con nombre, dirección y capacidad, 100 % georreferenciado.

## 2. Jardines Infantiles Fundación Integra

- **Ítems MICR que poblaría**: igual que el anterior (nivel parvulario / 445 en su parte rural)
- Organismo · producto: **Fundación Integra**, vía IDE Chile · capa `jardines_infantiles_fundacin_integra`
- URL del servicio:
  `https://geoportal.cl/geoserver/Jardines_Infantiles_Fundacion_Integra/wfs?service=WFS&version=2.0.0&request=GetFeature&typeNames=Jardines_Infantiles_Fundacion_Integra%3Ajardines_infantiles_fundacin_integra&outputFormat=application/json&srsName=EPSG:4326&count=1000&startIndex=N`
- Formato · registros · coordenadas: **GeoJSON · 1.160 · 1.160 (100 %)**
- Campos: `nombre_jar`, `direcci__n`, `region`, `glosa_comu`, `localidad`, `modalidad_`,
  `zona_ubica`, `matricula_`, `lat`, `long`
- robots.txt: HTTP 200 · permitido
- Crudo: `busqueda/crudo/integra/2026-08-25/jardines_integra.geojson`
- **Veredicto: PARCIAL** — completo y 100 % georreferenciado, pero el metadato del catálogo declara
  **2013-11-11**; es dato viejo. La versión al día la publica Integra en `geobuscador.integra.cl`,
  cuyo mecanismo de descarga **NO fue verificado**.

## 3. Establecimientos de Educación Parvularia (MINEDUC)

- **Ítems MICR que poblaría**: **445 Escuelas Rurales** (los 2.640 con `rural_esta=1`) y el nivel
  parvulario completo
- Organismo · producto: **Ministerio de Educación / Subsecretaría de Educación Parvularia**
  (incluye JUNJI, Integra y particulares), vía IDE Chile · capa `establecimientos_educacin_parvularia`, año 2021
- URL del servicio:
  `https://geoportal.cl/geoserver/Establecimientos_Educacion_Parvularia/wfs?service=WFS&version=2.0.0&request=GetFeature&typeNames=Establecimientos_Educacion_Parvularia%3Aestablecimientos_educacin_parvularia&outputFormat=application/json&srsName=EPSG:4326&count=500&startIndex=N`
- Formato · registros · coordenadas: **GeoJSON · 11.951 · 11.951 (100 %)**
- Campos clave: `nom_estab`, `nom_com_es`, `dependenci`, **`rural_esta`** (0 = 9.311, 1 = **2.640**),
  `estado_est`, matrículas por nivel, `n_total`, `direccion`
- robots.txt: HTTP 200 · permitido
- Crudo: `busqueda/crudo/parvularia/2026-08-25/establecimientos_parvularia.geojson`
- **Veredicto: SIRVE** — es el **superconjunto** de JUNJI + Integra + particulares y es el único de
  los tres que trae la **marca rural**, que es justo lo que necesita el ítem 445.

## 4. Infraestructura Deportiva (Ministerio del Deporte)

- **Ítems MICR que poblaría**: **567 Gimnasios y Centros Deportivos** (los 20.701) y, como sede,
  **542 Eventos Masivos (Conciertos)** (los 180 registros cuyo tipo es *Estadio*, más las
  combinaciones que incluyen estadio)
- Organismo · producto: **Ministerio del Deporte** (declarado en los campos `emisor` y `admin` de la
  propia capa), vía IDE Chile · capa `infraestructura_deportiva`
- URL del servicio:
  `https://geoportal.cl/geoserver/Infraestructura_Deportiva/wfs?service=WFS&version=2.0.0&request=GetFeature&typeNames=Infraestructura_Deportiva%3Ainfraestructura_deportiva&outputFormat=application/json&srsName=EPSG:4326&count=500&startIndex=N`
- Formato · registros · coordenadas: **GeoJSON · 20.701 · 20.701 (100 %)**
- Desglose por tipo (campo `condicio`): Multicancha 8.218 · Cancha de Fútbol 3.555 · Cancha de
  Futbolito 1.417 · Otros 1.291 · **Pabellón o Gimnasio 1.117** · Multicancha+Pabellón/Gimnasio 478 ·
  Sala especializada 442 · Medialuna 439 · **Estadio 180** · Piscina 112 · Cancha de Tenis 110 ·
  Cancha de Baloncesto 102 · resto, combinaciones
- Trae además `sector` (Público / Privado), `acceso_dis` (accesibilidad para discapacidad) y
  `tipo_entra` (Libre / Restringido)
- robots.txt: HTTP 200 · permitido
- Crudo: `busqueda/crudo/infraestructura_deportiva/2026-08-25/infraestructura_deportiva.geojson`
- **Veredicto: SIRVE** — es, con diferencia, **el catastro más grande de este lote**: 20.701 recintos
  deportivos del país, todos con punto. Metadato 2020-03-04.

## 5. Casinos de juego en operación (Superintendencia de Casinos de Juego)

- **Ítems MICR que poblaría**: **533 Casinos**
- Organismo · producto: **SCJ**, publicado por la Secretaría Ejecutiva SNIT en el Geoportal ·
  planilla `opengov_ofertadejuegos-infooperacional-casinoschile.xlsx`
- URL de descarga: `https://geoportal.cl/geoportal/catalog/download/8a0165c8-8892-3fb4-bea2-15f48e63e222`
  (responde `Content-Disposition: attachment`, 121.665 bytes)
- Formato · registros · coordenadas: **XLSX · 26 casinos · 26 con coordenada (100 %)**, columnas
  `latitud` / `longitud` en grados decimales
- Campos: `ID_CASINO`, `Vigente`, `nombre_comercial`, `razón social`, `Tipo de operación`,
  `Grupo empresarial`, `Dirección`, `Comuna`, `Región`, `CUT_COMUNA`, `CUT_REGION`, `RUT`, `URL_SITIO_SCJ`
- Trae además una hoja `Parque_y_Operaciones` con **495 filas mensuales** (nº de máquinas de azar,
  posiciones de bingo, ingresos brutos, **número de visitas**) — sirve para estimar población expuesta.
- robots.txt: HTTP 200 · permitido
- Crudo: `busqueda/crudo/casinos_scj/2026-08-25/opengov_ofertadejuegos-infooperacional-casinoschile.xlsx`
- **Veredicto: SIRVE** — universo cerrado y completo (los casinos de la ley 19.995 son todos los que
  hay), con coordenada y con número de visitas mensuales.

## 6. III Catastro Nacional de Espacios Públicos y Privados de Uso Cultural 2021

- **Ítems MICR que poblaría**: **531 Cines y Teatros** (24 "Cine o sala de cine" + 120 "Teatro o
  sala de teatro" = **144**) y **483 Centros de Educación Comunitaria** (391 Bibliotecas + 239
  Centros culturales / Casas de la cultura + 7 Cecrea = **637**)
- Organismo · producto: **Ministerio de las Culturas, las Artes y el Patrimonio — Observatorio
  Cultural** · `BBDD-Espacios-Culturales_V3.xlsx`
- URL de descarga:
  `https://observatorio.cultura.gob.cl/wp-content/uploads/2022/04/BBDD-Espacios-Culturales_V3.xlsx`
  (página: `https://observatorio.cultura.gob.cl/index.php/2022/04/06/iii-catastro-nacional-de-espacios-publicos-y-privados-de-uso-cultural-2021/`)
- Formato · registros · coordenadas: **XLSX · 1.333 filas · 1.332 con coordenada (99,9 %)**
- **ADVERTENCIA IMPORTANTE**: las columnas `POINT_X` / `POINT_Y` **no están en grados**, están en
  **UTM** (huso 19S en el continente; p. ej. La Cisterna X=346.053 Y=6.289.158). Isla de Pascua trae
  X negativo (−3.700.234) y hay que tratarla aparte. **Hay que reproyectar antes de usar.**
- Desglose de `tipologia`: Biblioteca 391 · Centro cultural / Casa de la cultura 239 · Museos 203 ·
  Teatro o sala de teatro 120 · Librería 104 · Otros espacios con uso cultural 97 · Sala de
  exposición 28 · Espacios artesanías 25 · Cine o sala de cine 24 · Galería de arte 21 · Sala de
  concierto 19 · Sala de Ensayo 17 · Sitio/Espacio de Memoria 13 · Centro de documentación 11 ·
  Cecrea 7 · Estudio de grabación 6 · Circo o carpa 5 · Archivo 3
- El cuestionario incluye preguntas de **amenaza** (P72: maremoto/tsunami, incendios estructurales,
  vandalismo, robos) — dato de fragilidad ya levantado en terreno.
- robots.txt: `https://observatorio.cultura.gob.cl/robots.txt` → HTTP 200, `User-agent: *` /
  `Disallow:` vacío → permitido
- Crudo: `busqueda/crudo/espacios_culturales/2026-08-25/BBDD-Espacios-Culturales_V3.xlsx`
- **Veredicto: PARCIAL** — es el único catastro nacional georreferenciado de cines y teatros que
  encontré, pero es un **censo por encuesta de 2021** y subrepresenta las multisalas comerciales
  (sólo 10 "Multisala de cine" en todo Chile, cuando Cinemark/Cineplanet solos tienen más).

## 6-bis. Espacios Socioculturales (mismo catastro)

- URL: `https://observatorio.cultura.gob.cl/wp-content/uploads/2022/04/BBDD-Espacios-Socioculturales.xlsx`
- **460 registros · 0 con coordenada** (sólo `COD_REG`, `COD_PROV`, `CODOMUNA`)
- Crudo: `busqueda/crudo/espacios_culturales/2026-08-25/BBDD-Espacios-Socioculturales.xlsx`
- **Veredicto: NO SIRVE** para georreferenciar — sólo llega a nivel comuna. Se guardó igual porque
  sirve para conteo comunal.

## 7. Atractivos Turísticos Nacionales 2020 (SERNATUR)

- **Ítems MICR que poblaría**: **566 Zonas de Entretenimiento** (65 de categoría "CENTRO O LUGAR DE
  ESPARCIMIENTO", más el grueso de los 4.881 como zonas de concentración de público) y **542 Eventos
  Masivos (Conciertos)** (los **833** de categoría "ACONTECIMIENTOS PROGRAMADOS", que son eventos
  con fecha y lugar fijo)
- Organismo · producto: **SERNATUR**, vía IDE Chile · capa `atractivos_tursticos_2020`
- URL del servicio:
  `https://geoportal.cl/geoserver/Atractivos_Turisticos_Nacionales/wfs?service=WFS&version=2.0.0&request=GetFeature&typeNames=Atractivos_Turisticos_Nacionales%3Aatractivos_tursticos_2020&outputFormat=application/json&srsName=EPSG:4326&count=500&startIndex=N`
- Formato · registros · coordenadas: **GeoJSON · 4.881 · 4.881 (100 %)**
- Campos: `nombre`, `jerarquia` (INTERNACIONAL 477 / NACIONAL 1.544 / REGIONAL 1.747 / LOCAL 1.113),
  `categoria`, `tipo`, `subtipo`, `region`, `comuna`, `direccion`, `propiedad`, `administra`,
  `uso_tur`, `demanda`, **`servicios`** (agua potable / electricidad / sin servicios), `estado_1`
- robots.txt: HTTP 200 · permitido
- Crudo: `busqueda/crudo/atractivos_turisticos/2026-08-25/atractivos_turisticos_2020.geojson`
- **Veredicto: PARCIAL** — no es un catastro de "zonas de entretenimiento" en el sentido comercial,
  pero es el único inventario nacional georreferenciado de **lugares que concentran público**, y trae
  jerarquía y demanda, que es exactamente lo que el MICR necesita para exposición.

---

# FUENTES REVISADAS QUE **NO** SIRVEN (verificado, no supuesto)

## SERNATUR — Registro Nacional de Prestadores de Servicios Turísticos (hoteles y restaurantes)

- Ítems que habría poblado: **530 Hoteles y Resorts**, **544 Restaurantes y Comedores**
- URL verificada: `https://serviciosturisticos.sernatur.cl/resultado.php?page=1&tipo_servicio=1`
  → **HTTP 200**. El buscador declara **8.721 servicios de alojamiento turístico** registrados.
- **El problema**: el listado devuelve sólo *clase* (Hotel / Camping / Residencial…), *nombre* y
  *región*. **No hay dirección ni coordenada en el resultado.** Y la ficha individual
  (`registro.sernatur.cl/registroempresarios/fichaservicio?sid=…&sign=…`) sí trae dirección de texto
  pero **tampoco trae coordenadas**, y exige un parámetro `sign=` (hash firmado por servicio), o sea
  no se puede recorrer sistemáticamente.
- `descargas.php` existe pero sólo publica PDF de decálogos al turista; **no hay descarga masiva**.
- robots.txt de `serviciosturisticos.sernatur.cl`: **HTTP 404** (no existe). El de `www.sernatur.cl`:
  HTTP 200, sólo `Disallow: /suggest/?*`.
- **Veredicto: NO SIRVE** — hay 8.721 alojamientos formales registrados y ninguno tiene coordenada
  publicada. **Este es el mayor agujero del lote Comercial** y probablemente requiere una gestión
  con SERNATUR (Ley de Transparencia) más que una descarga.

## Portal de Datos Abiertos `datos.gob.cl`

- `https://datos.gob.cl/` → HTTP 200, pero **`/dataset?q=…` devuelve HTTP 405 Method Not Allowed**
  (probado con y sin User-Agent de navegador, y también vía WebFetch). Las páginas
  `/dataset/<id>`, `/group/<x>` y `/organization` **sí responden 200**.
- `robots.txt`: HTTP 200 · `Disallow: /revision/`, `/dataset/rate/`, `/dataset/*/history`,
  **`Disallow: /api/`**, `Crawl-Delay: 10`. → **La API CKAN está explícitamente prohibida a los
  robots**; se respetó y no se usó.
- El grupo `cultura` (13 datasets) y el grupo `educacion` (13 datasets) se revisaron uno a uno: no
  hay nada georreferenciado útil para este lote.
- **Veredicto: NO SIRVE** para este lote (sí sirve el Geoportal, que es donde está el dato).

## MINEDUC — datos abiertos

- `https://centroestudios.mineduc.cl/robots.txt` → **HTTP 403 Forbidden**
- `https://datosabiertos.mineduc.cl/robots.txt` → **HTTP 403 Forbidden**
- No hace falta: el **Directorio Oficial de Establecimientos 2025 ya está en el proyecto**
  (`datos/crudo/mineduc_directorio/2026-08-19/`, 16.768 filas). **Se revisó de sólo lectura, no se
  tocó nada.**
- **Hallazgo aprovechable sin bajar nada**: ese archivo trae la columna **`RURAL_RBD`**.
  `RURAL_RBD=1` → **4.935 establecimientos**, de los cuales **4.204 con coordenada válida**
  (`LATITUD`/`LONGITUD` vienen con **coma decimal**, hay que reemplazarla por punto o se leen como
  cero — de hecho leyéndolas mal salían 1 de 4.935). Eso **puebla el ítem 445 Escuelas Rurales
  sin ninguna descarga nueva**.

## IND / Ministerio del Deporte — Directorio de Recintos Deportivos

- `https://ind.cl/directorio-de-recintos-deportivos/` → **HTTP 403** (bloquea al cliente).
  robots.txt de `ind.cl` responde 200 y no prohíbe esa ruta, pero el servidor igual rechaza.
- **No hace falta**: la capa `Infraestructura_Deportiva` del Geoportal es el mismo catastro del
  MINDEP y sí está accesible.

## Subtel / MINEDUC — "Conectividad para la Educación 2030"

- Ítem que habría poblado: **463 Infraestructura de Conectividad (Internet)**
- El programa declara más de 9.000 escuelas conectadas, pero la plataforma de gestión (SAGEC) es de
  Subtel y **no se encontró ningún listado descargable**.
- **Veredicto: NO VERIFICADO** — no se reporta como fuente.

## JUNAEB — Programa de Alimentación Escolar (PAE)

- Ítem que habría poblado: **469 Comedores Escolares**
- No existe organización JUNAEB en `datos.gob.cl` (`/organization/junaeb` y
  `/organization/junta-nacional-de-auxilio-escolar-y-becas` → **HTTP 404**) y no se encontró listado
  abierto de establecimientos con PAE.
- **Veredicto: NO VERIFICADO**.

## Bibliotecas Públicas (SNBP / Servicio Nacional del Patrimonio Cultural)

- `https://www.bibliotecaspublicas.gob.cl/` → HTTP 200, pero
  `/buscar-biblioteca` → **HTTP 500** y `/servicios/busca-tu-biblioteca-publica` no expone ni
  coordenadas ni endpoint JSON (se inspeccionó el HTML).
- **No es bloqueante**: las **391 bibliotecas** del Catastro de Espacios Culturales cubren el caso
  con coordenada.

## Establecimientos de Educación Superior (probado para bibliotecas universitarias)

- Capa `Establecimientos_Educacion_Superior` del Geoportal: **1.297 inmuebles**, con campo
  `nombre_inm` (nombre del inmueble/sede).
- Se consultó con filtro CQL `nombre_inm LIKE '%BIBLIOTECA%'` → **2 registros**. No sirve para el
  ítem 472.
- La capa en sí ya está cubierta por lo que el proyecto tiene de universidades; **no se bajó**.

## Madrasas (ítem 444)

- Verificado sobre el Directorio MINEDUC 2025: la columna `ORI_RELIGIOSA` sólo tiene los valores
  1 (7.203), 9 (4.266), 2 (3.189), 7 (1.584), 3 (523) y 5 (3 — los tres colegios hebreos). **No hay
  categoría islámica.** Los 7 establecimientos con "árabe" en el nombre (Escuela República Árabe
  Siria, Colegio Árabe, etc.) son colegios laicos de colectividad, **no madrasas**.
- **Veredicto: el ítem 444 no aplica a Chile.** No es que falte la fuente: no existe el objeto.

## Otras capas del Geoportal revisadas y descartadas para este lote

- `Ferias_Libres` (+ RM, Valparaíso, Biobío, de ODEPA): son **ferias libres de barrio** →
  corresponden al ítem **414 Mercados Locales (Ferias)** del sector **Alimentario**, que **no es de
  este lote**. Se deja anotado para quien lleve Alimentario. **No se bajó.**
- `Destinos_Turisticos` 2024, `Circuitos_Turisticos`, `Territorios_con_Potencial_Turistico`: son
  polígonos de planificación turística, no activos físicos.
- `Localización de establecimientos educacionales ingresados al SNCAE`: certificación ambiental de
  escuelas; no corresponde a ningún ítem de este lote.
- `Jardines junji 2026 (MINEDUC)`, `Jardines y salacuna Integra 2026 (MINEDUC)`,
  `Establecimientos de educacion escolar 2025 (MINEDUC)`: en el catálogo figuran como
  "INFORMACIÓN BASE" pero **su enlace es a un buscador web** (`buscatujardin.junji.gob.cl`,
  `geobuscador.integra.cl`, `centroestudios.mineduc.cl/datos-abiertos/`), **no a un servicio de
  datos**. Serían las versiones 2026 de lo que sí bajé en versión 2021/2024. **NO VERIFICADO** el
  acceso a los datos.
- `Centros de Salud, Educación y Cuidados (MDSyF)`: el enlace del metadato apunta a una **carpeta de
  Google Drive**, no a un servicio. **NO VERIFICADO.**

---

# TABLA RESUMEN · ítem MICR → fuente → registros

| Ítem MICR | Fuente | Registros (con coordenada) | Veredicto |
|---|---|---|---|
| **445** Escuelas Rurales | MINEDUC Directorio 2025 **ya en el proyecto**, columna `RURAL_RBD=1` | 4.935 (**4.204** con coord.) | SIRVE — sin descarga nueva |
| **445** Escuelas Rurales (parvulario) | MINEDUC · Establecimientos Educación Parvularia (`rural_esta=1`) | 2.640 (2.640) | SIRVE |
| **483** Centros de Educación Comunitaria | Mincultura · Catastro Espacios Culturales 2021 (Biblioteca + Centro cultural + Cecrea) | 637 (637, **UTM**) | PARCIAL |
| **457** Materiales Educativos (Libros) | Mincultura · Catastro Espacios Culturales (Librería) | 104 (104, **UTM**) | PARCIAL — es el punto de venta, no el material |
| **531** Cines y Teatros | Mincultura · Catastro Espacios Culturales 2021 | 144 (144, **UTM**) | PARCIAL |
| **533** Casinos | Superintendencia de Casinos de Juego (vía Geoportal) | 26 (26) | SIRVE |
| **542** Eventos Masivos (Conciertos) | MINDEP · Infraestructura Deportiva, tipo *Estadio* | 180 (180) | PARCIAL — sede, no evento |
| **542** Eventos Masivos (Conciertos) | SERNATUR · Atractivos, categoría *Acontecimientos Programados* | 833 (833) | PARCIAL |
| **566** Zonas de Entretenimiento | SERNATUR · Atractivos Turísticos 2020 (65 "Centro o lugar de esparcimiento" / 4.881 totales) | 4.881 (4.881) | PARCIAL |
| **567** Gimnasios y Centros Deportivos | MINDEP · Infraestructura Deportiva | **20.701** (20.701) | SIRVE |
| *(nivel parvulario, sin ítem propio en la lista)* | JUNJI 2024 | 3.015 (3.015) | SIRVE |
| *(nivel parvulario, sin ítem propio en la lista)* | Fundación Integra | 1.160 (1.160) | PARCIAL (dato 2013) |
| *(nivel parvulario, sin ítem propio en la lista)* | MINEDUC Parvularia 2021 (total) | 11.951 (11.951) | SIRVE |

**Total de activos georreferenciados descargados en este lote: 43.066**
= JUNJI 3.015 + Integra 1.160 + Parvularia 11.951 + Infraestructura Deportiva 20.701 +
Atractivos Turísticos 4.881 + Casinos 26 + Espacios Culturales 1.332.
(Se bajaron además las 460 filas de Espacios Socioculturales, que **no** tienen coordenada y por eso
no se cuentan.)

---

# ÍTEMS DE MI LOTE SIN NINGUNA FUENTE ENCONTRADA

## Educación (16 de 21)

| Ítem | Por qué |
|---|---|
| **442** Escuelas Secundarias (Infraestructura) | Aparece en `CATALOGO_FALTANTES.txt`, pero el encargo dice que MINEDUC ya lo pobló. **Contradicción a resolver**: o el catálogo está desactualizado, o el ítem 442 quedó sin poblar. |
| **444** Madrasas | **No aplica a Chile** (verificado sobre el Directorio MINEDUC: no hay orientación religiosa islámica). |
| **447** Aulas (Espacios de Enseñanza) | Es un componente interno del establecimiento. Ningún catastro chileno georreferencia aulas una a una. |
| **448** Bibliotecas Escolares | Las bibliotecas CRA existen en casi toda escuela subvencionada, pero **no hay catastro público georreferenciado** de ellas. Heredarían la coordenada del establecimiento. |
| **449** Laboratorios Escolares | Igual que el anterior. |
| **458** Materiales Digitales (Plataformas) | No es un activo físico localizable. |
| **463** Infraestructura de Conectividad (Internet) | CpE2030 / SAGEC (Subtel) no publica listado. NO VERIFICADO. |
| **464** Equipos Tecnológicos (Computadoras) | Componente interno; sin catastro. |
| **465** Proyectores y Pizarras Digitales | Componente interno; sin catastro. |
| **468** Instalaciones Sanitarias (Escuelas) | Componente interno; sin catastro. **Es IRMD Alto y queda vacío.** |
| **469** Comedores Escolares | JUNAEB no publica listado abierto del PAE. NO VERIFICADO. |
| **470** Centros de Investigación Educativa | Sin catastro. |
| **471** Escuelas de Formación Docente | Se podría derivar cruzando la capa de Educación Superior (1.297 inmuebles con coordenada) con la oferta de carreras de Pedagogía del SIES — **cruce NO verificado**, no se reporta como fuente. |
| **472** Bibliotecas Universitarias | Verificado por CQL: sólo 2 inmuebles de educación superior se llaman "biblioteca". Sin catastro. |
| **474** Aulas de Educación a Distancia | No es activo físico localizable. |
| **475** Escuelas de Emergencia (Campamentos) | Se levantan durante la emergencia; no hay catastro previo. |
| **476** Materiales de Emergencia (Educación) | Sin catastro. |
| **482** Programas de Alfabetización (Comunidades) | Es un programa, no un activo. |

## Comercial (13 de 20)

| Ítem | Por qué |
|---|---|
| **530** Hoteles y Resorts | **IRMD Alto.** Hay 8.721 alojamientos formales en el registro SERNATUR, pero **ninguno con coordenada publicada**. El agujero más importante del lote. |
| **532** Centros de Convenciones | Sin catastro nacional. |
| **534** Oficinas Comerciales | **IRMD Alto.** Sin catastro nacional georreferenciado (las patentes municipales no se publican en conjunto ni con punto). |
| **543** Tiendas Minoristas | Sin catastro nacional. Sólo aparecen 104 librerías y 19 puntos de artesanía en el catastro cultural, que no representan el universo. |
| **544** Restaurantes y Comedores | SERNATUR tiene la categoría en su registro, **sin coordenadas**. |
| **546** Infraestructura de Conectividad | Fuera del alcance de este lote (lo cubre Telecomunicaciones). |
| **549** Estacionamientos (Centros Comerciales) | Componente interno; sin catastro. |
| **550** Ascensores y Escaleras Mecánicas | Componente interno; sin catastro. |
| **553** Almacenes (Centros Comerciales) | Componente interno; sin catastro. |
| **554** Centros de Gestión (Eventos) | Sin catastro. |
| **555** Pantallas y Sistemas de Audio | Componente interno; sin catastro. |
| **559** Centros de Capacitación (Personal) | Sin catastro. |
| **568** Centros de Almacenamiento (Mercancías) | Sin catastro en este lote (podría venir del sector Industrial/Transporte). |
| **570** Áreas de Exposición (Ferias) | **IRMD Alto.** El catastro cultural tiene 28 "salas de exposición" y 21 galerías de arte, pero eso es arte, **no ferias de exposición comercial** (FISA, Espacio Riesco). No lo reporto como fuente. |
| **571** Centros de Atención al Cliente | Sin catastro. |

---

# NOTAS PARA QUIEN INTEGRE ESTO

1. **Los ítems que son "componentes internos del edificio"** (aulas, bibliotecas escolares,
   laboratorios, baños, proyectores, ascensores, estacionamientos, pantallas) **no tienen ni van a
   tener catastro georreferenciado propio en Chile**. Son 18 de los 41 ítems de este lote. Si el MICR
   los quiere medir, la única salida honesta es **heredar la coordenada del establecimiento
   contenedor** y declarar la confianza como derivada, no medida.
2. **Coordenadas del catastro cultural en UTM**: reproyectar antes de cruzar con clima. Isla de
   Pascua viene con X negativo y se cae si se procesa junta con el continente.
3. **Coma decimal en el Directorio MINEDUC**: `LATITUD` = `-18,4872`. Si se lee con `float()` directo
   devuelve error y se pierde el 99,98 % de los puntos (lo verifiqué: salían 1 de 4.935 rurales).
4. **El WFS de `geoportal.cl` se cae con concurrencia.** Una descarga a la vez, `count=500`, pausa de
   1 s, reintentos con espera creciente y caché por página. Con eso bajaron las 20.701 filas
   deportivas sin problema.
5. **Contradicción a levantar con el director**: `CATALOGO_FALTANTES.txt` lista el ítem **442
   Escuelas Secundarias** como sin catastro, pero el encargo dice que ya está poblado desde MINEDUC.
