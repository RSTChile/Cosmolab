# Búsqueda de catastros públicos georreferenciados · sectores ENERGÍA, REPRESAS y NUCLEAR

Fecha del barrido: **2026-08-25**
Lote: los 59 ítems físicos sin catastro de `busqueda/CATALOGO_FALTANTES.txt`
(Energía 29, Represas 17, Nuclear 13).
Todo lo que aparece abajo con número de registros fue **descargado y contado**, no estimado.
Lo que no se pudo verificar se dice "NO VERIFICADO" en forma explícita.

---

## 1. CNE · Geo Portal SIGCRA (GeoServer público) — **LA FUENTE MÁS IMPORTANTE DEL LOTE**

- **Ítems MICR que puebla**: **89** (carbón), **90** (gas natural), **91** (diésel),
  **106** (biomasa), **111** (refinerías, parcial), **112** (GNL), **113** (almacenamiento de combustibles).
  De regalo, fuera de mi lote: gasoductos y oleoductos.
- **Organismo**: Comisión Nacional de Energía (CNE) — Ministerio de Energía.
- **Producto**: workspace `cne-sigcra-new` del GeoServer de `energiamaps.cne.cl`.
- **URL exacta** (patrón):
  `https://energiamaps.cne.cl/geoserver/cne-sigcra-new/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=cne-sigcra-new:<capa>&outputFormat=application/json`
  Capabilities: `https://energiamaps.cne.cl/geoserver/cne-sigcra-new/ows?service=WFS&version=1.1.0&request=GetCapabilities` (47 capas)
- **Formato**: GeoJSON (también SHAPE-ZIP y KML). SRS de salida EPSG:4326.
- **robots.txt**: `https://energiamaps.cne.cl/robots.txt` → **HTTP 200**, contenido
  `User-agent: *` **sin ninguna línea Disallow** ⇒ todo permitido.
- **Registros / con coordenada** (todas 100 % con geometría):

| capa | registros | con coord. |
|---|---:|---:|
| `centrales_termoele_20181016234834` | 157 | 157 |
| `centrales_biomasa_20181016234835` | 32 | 32 |
| `almace_de_combustible_20181016234835` | 71 | 71 |
| `terminales_maritimos_20181016234836` | 29 | 29 |
| `gaseoductos_20181016234835` | 56 | 56 |
| `oleoductos_20181016234836` | 33 | 33 |
| `concesion_geotermina_20181016234836` | 12 | 12 |

- **Crudo guardado**: `busqueda/crudo/cne_sigcra/2026-08-25/` (+ `PROCEDENCIA.txt`).
- **Veredicto: SIRVE**, y es el hallazgo central. La capa `centrales_termoele` trae el campo
  **`combustibl` POBLADO** para las 157 centrales térmicas: `PETROLEO DIESEL` 103,
  `GAS NATURAL` 14, `CARBON` 10, `DIESEL` 7, `GAS NATURAL / PETROLEO` 4, `CARBON / PETCOKE` 4,
  `FUEL OIL NRO 6` 3, `PETROLEO` 3, `GNL` 2, y 6 categorías con 1 cada una
  (`PETROLEO / GAS NATURAL`, `PROPANO`, `PETCOKE`, `COGENERACION`,
  `PETROLEO DIESEL / GAS NATURAL`, `CARBON/GAS NATURAL`, `PETROLEO DIESEL/CARBON`).
  Contando las mezclas: **16 centrales tocan carbón, 23 tocan gas natural, 123 tocan diésel/petróleo**.
  `centrales_biomasa` trae `BIOMASA` 16, `BIOGAS` 12, `BIOMASA-PETROLEO N°6` 4.
- **Dos reservas honestas**:
  1. El campo `fech_act` de las capas de centrales dice **2018-06-30**. Es dato de 2018:
     hay centrales nuevas del Coordinador que no están, y centrales de 2018 ya retiradas que sí están.
  2. **El rótulo administrativo viene mal en al menos un registro**: `AGUAS BLANCAS`
     (lon −69,8646 / lat −24,1355, o sea Antofagasta) aparece rotulada
     `ARICA Y PARINACOTA / ARICA / CAMARONES`. **Usar la coordenada, nunca el rótulo `region`.**
- En `almace_de_combustible` están, con coordenada, **ENAP REFINERIAS ACONCAGUA**,
  **ENAP REFINERIAS BIO BIO** y **COMPLEJO GREGORIO** (ENAP Magallanes) ⇒ ítem 111 parcial
  (son las 3 refinerías de ENAP, que es el universo real de refinerías del país).
- En `terminales_maritimos` están **GNL QUINTERO** y **GNL MEJILLONES**, ambos con coordenada
  ⇒ ítem 112 completo (Chile tiene exactamente esos dos terminales de GNL).

---

## 2. Coordinador Eléctrico Nacional · FICHAS TÉCNICAS por unidad generadora

- **Ítems MICR que puebla**: **89**, **90**, **91**, **106** (el combustible de cada central
  térmica) y **121** (transformadores de subestación).
- **Organismo**: Coordinador Eléctrico Nacional.
- **Producto**: endpoint de fichas técnicas de la API Infotécnica, no listado entre los
  recursos "normales" pero sí en el openapi.
- **URL exacta**:
  `https://api-infotecnica.coordinador.cl/v1/{slug_instalacion}/{id}/fichas-tecnicas/{slug_tipo_ficha}/`
  Verificados: `.../unidades-generadoras/82/fichas-tecnicas/general/`,
  `.../unidades-generadoras/311/fichas-tecnicas/turbinas-vapor/`,
  `.../unidades-generadoras/797/fichas-tecnicas/turbinas-gas/`,
  `.../unidades-generadoras/4562/fichas-tecnicas/motor-combustion-interna/`.
  Catálogos: `/v1/tipos-fichas-tecnicas/` (43 tipos) y `/v1/conceptos/` (2.121 conceptos).
- **Formato**: JSON. **Acceso**: público, sin token.
- **robots.txt**: `https://api-infotecnica.coordinador.cl/robots.txt` → **HTTP 404**
  (no existe; ninguna restricción declarada). Se usó pausa de 0,6 s entre peticiones.
- **Registros**: 1.024 peticiones (512 fichas generales + 121 fichas de turbina + 391 de MCI),
  **HTTP 200 en las 1.024, cero errores**. Las fichas no traen coordenada propia: se
  georreferencian por `id_central` contra `/v1/centrales/extended/` (1.217 centrales,
  1.207 con lat/lon).
  Además `/v1/transformadores-2d/` **2.110** y `/v1/transformadores-3d/` **373**
  (2.483 transformadores de poder), con `id_subestacion` ⇒ georreferenciables contra las
  subestaciones que el proyecto ya tiene.
- **Crudo guardado**: `busqueda/crudo/coordinador_combustible/2026-08-25/` (+ `PROCEDENCIA.txt`).
- **Veredicto: SIRVE**. Ver la sección "EL BLOQUEO DEL COMBUSTIBLE".

---

## 3. SERNAGEOMIN · Instalaciones mineras (ArcGIS Online)

- **Ítems MICR que puebla**: **108** (campos petrolíferos), **109** (pozos de gas natural),
  **110** (minas de carbón).
- **Organismo**: Servicio Nacional de Geología y Minería.
- **Producto**: capa `INSTALACIONES_MINERAS_20251126` — catastro nacional de instalaciones
  mineras, **28.999 puntos** en todo Chile.
- **URL exacta**:
  `https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/services/INSTALACIONES_MINERAS_20251126/FeatureServer/0/query?where=<filtro>&outFields=*&returnGeometry=true&outSR=4326&f=json`
  (misma organización de ArcGIS Online que el proyecto ya usa en `traer_capas_sernageomin.py`).
- **Formato**: esriJSON. **Acceso**: público, sin token, HTTP 200 verificado.
- **robots.txt**: `https://services1.arcgis.com/robots.txt` → **HTTP 403 "Invalid URL"**
  (ArcGIS Online no sirve robots.txt en el host de servicios; no hay prohibición declarada).
- **Registros / con coordenada** (100 % con geometría):

| filtro sobre `RecursoMineroInstalacion` | registros | de los cuales |
|---|---:|---|
| `%CARB%` (CARBÓN) | **553** | MINA SUBTERRANEA 194 + MINA RAJO ABIERTO 75 = **269 minas de carbón** |
| `%GAS NATURAL%` | **635** | **POZO DE GAS 437**, CENTRAL-GAS 46, PLATAFORMA COSTA AFUERA 12 |
| `%PETR%` (PETRÓLEO) | **248** | **POZO DE PETROLEO 159**, BATERIA DE RECEPCIÓN 20 |

- **Crudo guardado**: `busqueda/crudo/sernageomin_instalaciones_mineras/2026-08-25/`.
- **Veredicto: SIRVE**, y con creces. El campo `RecursoMineroInstalacion` tiene 105 valores
  distintos, así que el universo completo (28.999) también sirve al sector Industrial —
  aquí sólo se bajaron los tres subconjuntos de este lote.

---

## 4. DOH · SIALL — Sistema de Información de Aguas Lluvias

- **Ítems MICR que puebla**: **58** (infraestructura de drenaje), **69** (áreas de control de
  inundaciones), **84** (zonas de protección, parcial), **60** (tuberías de distribución, por analogía).
- **Organismo**: Dirección de Obras Hidráulicas, MOP.
- **URL exacta**: `https://rest-sit.mop.gob.cl/arcgis/rest/services/DOH/SIALL/MapServer`
  (capas: `.../MapServer/layers?f=json`).
- **Formato**: **esriJSON obligatorio** — ver la nota técnica más abajo.
- **robots.txt**: `https://rest-sit.mop.gob.cl/robots.txt` → **HTTP 200, cuerpo vacío**
  (sin ninguna línea Disallow).
- **Registros / con coordenada** (todo 100 % con geometría, 142 MB en disco):

| archivo | capa | registros |
|---|---|---:|
| `descargas` | 1 DESCARGAS | 1.951 |
| `camaras` | 2 CAMARAS | 27.718 |
| `otras_obras_puntos` | 3 OTRAS OBRAS (PUNTOS) | 203 |
| `sumideros` | 4 SUMIDEROS | 29.808 |
| `cauce_red_primaria` | 5 CAUCE RED PRIMARIA | 5.854 |
| `sistema_colector` | 6 SISTEMA DE COLECTOR | 2.937 |
| `tramo_colector` | 7 TRAMO COLECTOR | 27.084 |
| `otras_obras_poligonos` | 8 OTRAS OBRAS (POLÍGONOS) | 280 |
| `puntos_inundacion` | 10 PUNTOS DE INUNDACION | 429 |
| `puntos_remocion_masa` | 11 PUNTOS REMOCION EN MASA | 11 |
| `areas_tributarias` | 13 AREAS TRIBUTARIAS | 588 |
| `zonas_plan_maestro` | 14 ZONAS PLAN MAESTRO | 148 |
| `subareas_tributarias_MUESTRA` | 12 SUBAREAS TRIBUTARIAS | **2.400 de 20.414 (MUESTRA)** |
| **total descargado** | | **99.411** |

- **Crudo guardado**: `busqueda/crudo/mop_doh_siall/2026-08-25/`.
- **Veredicto: SIRVE** para el 58 y el 69; **PARCIAL** para el 84 y el 60.
  Los polígonos de "OTRAS OBRAS" traen `TIPO` con `ZONA DE EXCLUSION URBANA` 44,
  `ZONA DE RESTRICCION URBANA` 29, `SISTEMA DE REGULACION` 25, `RETENCIÓN` 16+5,
  `CONTROL ALUVIONAL` 7, `PLANICIE INUNDACION` 11 — que es material directo del ítem 69.
- **★ Nota técnica que hay que guardar** (perdí tres intentos con esto):
  este ArcGIS 10.21 **no soporta `f=geojson`** ("Output format not supported") y
  **ignora `resultOffset`** (devuelve una y otra vez el primer lote, sin avisar).
  Hay que paginar con `returnIdsOnly=true` y luego pedir por `objectIds` en lotes de 500,
  **por POST**, porque la URL con 500 ids da 404 por longitud.
  El patrón `resultOffset/resultRecordCount` del encargo **no funciona en este servidor**.

---

## 5. DGA · ESTACION_EMBALSE

- **Ítem MICR que puebla**: **87** (estaciones de monitoreo asociadas a embalses).
- **URL exacta**: `https://rest-sit.mop.gob.cl/arcgis/rest/services/DGA/ESTACION_EMBALSE/MapServer/0/query?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=json`
- **Formato**: esriJSON. **robots.txt**: HTTP 200, cuerpo vacío.
- **Registros**: **11**, todos con coordenada. Trae nivel [m], volumen [Mm³] y fecha de medición:
  Rapel, Ralco, Pangue, Recoleta, Cogotí, Peñuelas, Melado, Colbún, Laguna de la Invernada,
  Laguna del Maule, Laguna de La Laja.
- **Crudo guardado**: `busqueda/crudo/dga_estacion_embalse/2026-08-25/`.
- **Veredicto: SIRVE (chico pero exacto)**. La red hidrométrica general de la DGA
  (`DGA/Red_Hidrometrica`, 5.084 estaciones) **NO se bajó a propósito**: el proyecto ya
  tiene 4.110 estaciones DGA en `datos/dga_estaciones.csv`.

---

## 6. Fuentes revisadas que NO sirvieron (y por qué)

| fuente | qué pasó | veredicto |
|---|---|---|
| `arcgis2.minenergia.cl` (IDE Energía, servicio `IDE_ENERGIA/IDE_2019/MapServer`) | el DNS resuelve a 163.247.59.201 pero **la conexión TCP nunca se establece** ("Network is unreachable", HTTP 000, verificado dentro y fuera del sandbox) | **NO SIRVE hoy** — el host está caído o bloqueado desde aquí. Vale la pena reintentar otro día: es el catálogo oficial de capas del Ministerio de Energía |
| `ide-energia.minenergia.cl` | **HTTP 403** en la raíz | NO SIRVE sin credenciales |
| `energiaabierta.cl` / `www.energiaabierta.cl` | **no resuelve** (HTTP 000) | NO SIRVE |
| `api.cne.cl` (Api Combustibles: bencina, gas, parafina) | **exige registro y token**; robots.txt HTTP 200 `Disallow:` (permitido) | **PIDE TOKEN** — anotado y no se intentó saltar. Cubriría el ítem 642 (estaciones de servicio), que no es de este lote |
| `datos.gob.cl` (CKAN) | su **robots.txt prohíbe `/api/`** (`Disallow: /api/`, `Crawl-Delay: 10`). Se hicieron 7 consultas de sondeo y se paró. Búsquedas: "nuclear" → 9 datasets, **todos administrativos de la CCHEN** (convenios, dosimetría, Fukushima), ninguno georreferenciado. "embalse" → 0. "gas licuado" → 0. "refineria" → 1 (minería mundial) | **NO SIRVE** para este lote, y además no se debe raspar |
| `energia.gob.cl/electromovilidad/ecocarga` (plataforma EcoCarga, ítem 128) | la página existe (HTTP 200) pero **no expone ningún endpoint de datos**; `ecocarga.cl` no resuelve. robots.txt con `Crawl-delay: 10` | **NO VERIFICADO** — hay dato (≈1.097 puntos de carga públicos según prensa oficial) pero no encontré la vía pública de descarga |
| `www.cchen.cl` (ítems nucleares) | robots.txt → **HTTP 404**. La transparencia publica **autorizaciones sueltas en PDF**, una por instalación; **no hay nómina consolidada ni capa georreferenciada** | **NO SIRVE** sin raspado manual de PDFs |
| `arcgis.sernageomin.cl` | HTTP 000, no conecta | NO SIRVE (se usó ArcGIS Online en su lugar, que sí funciona) |
| Coordinador `/v1/torres/` y `/v1/conexiones-tierra/` (ítem 118, torres de transmisión) | el catálogo `/v1/instalaciones-tipos/` **declara el tipo 37 "Torres"**, pero el endpoint devuelve **HTTP 404**: no está expuesto | **NO SIRVE** — el dato existe adentro, la API no lo publica |
| `geoportal.cl` / `www.ide.cl` | responden 200; el catálogo de geoportal.cl redirige y no encontré API de búsqueda estable en el tiempo disponible | **NO VERIFICADO** |

---

# EL BLOQUEO DEL COMBUSTIBLE

**Resuelto. Por dos caminos independientes que además se controlan uno al otro.**

### Lo que estaba mal planteado
`id_combustible` de `https://api-infotecnica.coordinador.cl/v1/unidades-generadoras/`
**sigue viniendo nulo en las 1.640 unidades** (lo volví a verificar hoy: 0 de 1.640 no nulas),
y `/v1/combustibles/` (52 combustibles) sigue huérfano, sin que nada le apunte.
Eso no se arregla: **ese campo simplemente no se llena**. El dato del combustible
**no vive en el campo, vive en las FICHAS TÉCNICAS**, como texto libre.

### Camino A — dentro del propio Coordinador (el dato oficial y actual)

El endpoint `/v1/{slug}/{id}/fichas-tecnicas/{slug_ficha}/` es **público y funciona**.
El catálogo `/v1/conceptos/` tiene **24 conceptos que nombran el combustible**. Los que sirven:

| concepto | nombre | dónde aparece | poblado |
|---|---|---|---|
| **673** | `13.2.1 Tipo de combustible principal` | fichas `turbinas-vapor` / `turbinas-gas` | **45 de 121** unidades TV/TG/CC |
| **676** | `13.2.1 Combustibles secundarios` | idem | 45 |
| **721** | `12.3.6 Tipos de combustibles y consumo específico` | idem | 69 |
| **669** | `11.1.35 Combustible primario utilizado` | ficha `motor-combustion-interna` | **200 de 391** MCI |
| **311** | `11.1.4 Potencia máxima bruta, para cada tipo de combustible que pueda operar` | ficha `general` | 505 con texto; **27 nombran el combustible en el texto** (p. ej. `129,378 (Diesel) / 134,971 (GNL)`) |
| **932 / 2953** | tipo de tecnología de la unidad | ficha `general` | **512 de 512** |

Ejemplos reales bajados hoy: TER GUACOLDA `CARBON` / secundario `FO`; TER ANGAMOS
`Carbón Pulverizado`; TER COCHRANE `Carbón pulverizado`; TER CAMPICHE y TER SANTA MARIA
`Carbón Bituminoso y sub-bituminoso`; TER IEM `Coal (LHV: 23.350 kJ/kg)`; TER SAN ISIDRO
`Gas natural` con secundario `Diesel`; TER PETROPOWER `PETCOKE`; TER CMPC LAJA y
TER CMPC PACIFICO `LICOR NEGRO - BIOMASA`; TER VALDIVIA / ARAUCO / CELCO / LAUTARO `Biomasa`.

Y el concepto **932** parte solo las 512 unidades térmicas en:
`Parque de Motores Diésel` 316, `Turbina Gas` 69, `Térmica Vapor` 55, `Motores Diésel` 40,
`Térmica Ciclo Combinado` 22, cogeneración/termosolar 3, vacío 7.
O sea: **356 unidades quedan clasificadas como diésel por el propio tipo declarado**,
aunque nadie haya llenado el campo de combustible.

### Camino B — la capa de la CNE (control cruzado, con coordenada propia)

`cne-sigcra-new:centrales_termoele` trae **157 centrales térmicas georreferenciadas con el
campo `combustibl` lleno**, más 32 de biomasa. Es dato de 2018, pero es independiente del
Coordinador y sirve de contraste.

### Producto que desbloquea las centrales térmicas

`busqueda/crudo/coordinador_combustible/2026-08-25/centrales_termicas_combustible.csv`

Cruza las **213 centrales termoeléctricas del Coordinador** (con la lat/lon del propio
Coordinador) contra las dos fuentes, y deja en una columna **de qué concepto salió cada dato**:

| | centrales |
|---|---:|
| termoeléctricas del Coordinador | **213** |
| con combustible por **ficha del Coordinador** | **174** |
| con combustible por **capa CNE** (cruce por nombre normalizado) | 89 |
| **con combustible por alguna de las dos** | **192 (90 %)** |
| sin ninguna de las dos | 21 |

Clasificadas contra los ítems de la MICR (una central puede tocar más de un ítem si es dual):

- **ítem 89 · Carbón: 10 centrales** (16 si se cuentan las mezclas de la capa CNE)
- **ítem 90 · Gas Natural: 18 centrales** (23 con mezclas)
- **ítem 91 · Diésel: 153 centrales**
- **ítem 106 · Biomasa/biogás: 25 centrales** (+32 en la capa CNE de biomasa)
- sin clasificar: 25

**Las 206 centrales térmicas que estaban bloqueadas quedan desbloqueadas: 192 de 213 con
combustible identificado y con coordenada.** Las 21 que quedan son PMGD chicos cuya ficha
está vacía; para ésos el tipo de tecnología (concepto 932) igual dice si es motor diésel.

**Advertencia metodológica**: el combustible viene como **texto libre escrito por cada
empresa**, no como código. Hay `CARBON`, `Carbón`, `Carbón Pulverizado`,
`Carbon Bituminiso y sub-bituminoso` (con la errata incluida) y `Coal` conviviendo.
Normalizar es obligatorio y **la normalización es una decisión del proyecto, no del dato**.

---

# Tabla resumen · ítem → fuente → registros

## Energía

| ítem | qué es | fuente | registros con coordenada |
|---:|---|---|---:|
| 89 | Generación a carbón | Coordinador fichas + CNE `centrales_termoele` | **10–16 centrales** |
| 90 | Generación a gas natural | Coordinador fichas + CNE `centrales_termoele` | **18–23 centrales** |
| 91 | Generación a diésel | Coordinador fichas + CNE `centrales_termoele` | **153 centrales** |
| 106 | Plantas de biomasa | CNE `centrales_biomasa` + fichas Coordinador | **32** (capa) / 25 (Coordinador) |
| 108 | Campos petrolíferos | SERNAGEOMIN instalaciones mineras | **248** (159 pozos de petróleo) |
| 109 | Pozos de gas natural | SERNAGEOMIN instalaciones mineras | **635** (437 pozos de gas) |
| 110 | Minas de carbón | SERNAGEOMIN instalaciones mineras | **553** (269 minas) |
| 111 | Refinerías de petróleo | CNE `almace_de_combustible` | **3** (Aconcagua, Bío Bío, Gregorio) |
| 112 | Terminales de GNL | CNE `terminales_maritimos` | **2** (Quintero, Mejillones) = universo completo |
| 113 | Almacenamiento de combustibles | CNE `almace_de_combustible` | **71** |
| 121 | Transformadores de subestación | Coordinador `/v1/transformadores-2d/` + `-3d/` | **2.483** (georreferenciables por `id_subestacion`) |

## Represas

| ítem | qué es | fuente | registros con coordenada |
|---:|---|---|---:|
| 45 | Estructuras de presas | DOH `Catastro de Embalses` (campo `ALT_MURO`) | **ya está en el proyecto** (`datos/crudo/mop_sit/2026-08-19/embalses.geojson`) |
| 47 | Canales de derivación | DOH `Canales_CNR` + SIALL `OBRA DE TOMA` | ya bajado por otro agente hoy + **13** |
| 57 | Estaciones de bombeo | SIALL `PLANTA ELEVADORA` | **1** (muy parcial) |
| 58 | Infraestructura de drenaje | DOH SIALL (7 capas) | **95.555** |
| 60 | Tuberías de distribución | SIALL `TRAMO COLECTOR` (por analogía) | **27.084** |
| 69 | Áreas de control de inundaciones | SIALL zonas plan maestro + áreas tributarias + otras obras polígonos | **1.016** |
| 84 | Zonas de protección | SIALL polígonos `ZONA DE EXCLUSION/RESTRICCION URBANA` | **73** |
| 87 | Estaciones de monitoreo (clima) | DGA `ESTACION_EMBALSE` | **11** (+ 4.110 estaciones DGA ya en el proyecto) |

## Nuclear

| ítem | fuente | registros |
|---:|---|---|
| 134, 135, 136, 145, 146, 150, 159, 160, 161, 165, 172, 173, 175 | **ninguna** | 0 |

---

# Ítems sin fuente (de mi lote)

### Energía — 18 de 29 siguen sin catastro
- **92 Plantas nucleares**, **93 Barras de combustible nuclear**, **94 SMRs**:
  **el activo no existe en Chile**. Verificado en el propio Coordinador: el catálogo
  `/v1/centrales-tipos/` **declara el tipo 5 "Nuclear"**, y las 1.217 centrales se reparten
  Solares 753 / Termoeléctricas 213 / Hidroeléctricas 182 / Eólicas 68 / Geotérmica 1:
  **cero nucleares**. No es que falte el dato: no hay el activo.
- **96 Centrales hidráulicas de bombeo**: no hay ninguna en operación en el Coordinador.
- **100 Inversores solares**, **103 Turbinas eólicas individuales**, **125–127 solar/eólica
  residencial**: existen fichas `generacion-solar` y `turbinas-eolicas` por planta en el
  Coordinador, pero **no hay catastro con coordenada por equipo individual**. NO VERIFICADO
  si las fichas traen el número de inversores (no se revisó, queda como pista).
- **107 Gasificación de biomasa**: sin fuente.
- **114 BESS**: el Coordinador tiene la ficha `9.- Sistema de Almacenamiento de Energía`
  (tipo 108) y un tipo de unidad `Fotovoltaica + SAE` con **1 sola unidad**, pero
  **ninguna central se llama BESS/SAE/almacenamiento** en las 1.217. Sin catastro.
- **118 Torres de transmisión eléctrica**: el tipo de instalación 37 "Torres" está declarado
  en el Coordinador pero **el endpoint devuelve 404**. El dato existe y no se publica.
- **119 Cables submarinos de energía**: sin fuente (`/v1/tramos-tipos/` sólo tiene el valor "-").
- **122 Redes de distribución BT**, **123 Transformadores de distribución**,
  **130 Sensores de red**, **132 UPS**: sin fuente pública.
- **128 Estaciones de carga de VE**: existe EcoCarga (Ministerio de Energía) pero
  **no encontré endpoint público de descarga**. NO VERIFICADO.

### Represas — 9 de 17 siguen sin catastro
**48 Compuertas de control**, **59 Aliviaderos**, **62 Conectividad**,
**70 Infraestructura de refuerzo**, **71 Centros de gestión**, **72 Centros de capacitación**,
**73 Contingencia**, **77 Almacenes de equipos de emergencia**, **85 Muros de refuerzo**.
Ninguno tiene catastro público: son partes internas de la obra, y el catastro de embalses
de la DOH es **un punto por embalse**, no un despiece de la presa.

### Nuclear — 13 de 13 sin catastro
La CCHEN **no publica ningún catastro georreferenciado**. Su portal de transparencia
publica **autorizaciones sueltas en PDF**, una por instalación, sin nómina consolidada
ni coordenadas. Y `datos.gob.cl` sólo tiene 9 datasets de la CCHEN, todos administrativos.
Lo único público y verificable son **las direcciones postales** de los dos centros
(CEN La Reina, Las Condes — reactor RECH-1; CEN Lo Aguirre, Pudahuel — reactor RECH-2,
planta de procesamiento de combustible, planta de tratamiento y almacenamiento de residuos
radiactivos), que cubrirían por geocodificación los ítems 134, 135, 136 y 175.
**No las geocodifiqué: la regla del proyecto es no inventar coordenadas.** Queda como
pendiente documentado, a resolver con geocodificación verificada o consulta a la CCHEN.

---

# Pistas para el próximo barrido

1. **Reintentar `arcgis2.minenergia.cl/public/rest/services/IDE_ENERGIA/IDE_2019/MapServer`**.
   Es el catálogo oficial de capas del Ministerio de Energía y hoy el host no acepta conexión.
   Si vuelve, probablemente cubra BESS, electrolineras y capas más nuevas que las de 2018 de la CNE.
2. **Pedir token en `api.cne.cl`** (gratis, con registro) para el ítem 642 y para cruzar
   almacenamiento de combustibles con estaciones de servicio.
3. **El universo completo de SERNAGEOMIN (28.999 instalaciones mineras)** está a una consulta
   de distancia y sirve a los sectores Industrial y Químico, no sólo a Energía.
4. **`DOH/SIALL` capa 12 (SUBAREAS TRIBUTARIAS)**: quedan 18.014 polígonos sin bajar.
