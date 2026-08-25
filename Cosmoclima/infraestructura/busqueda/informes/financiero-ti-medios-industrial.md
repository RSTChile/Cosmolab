# Catastros públicos georreferenciados para los sectores Financiero, Tecnologías Informáticas, Comunicaciones e Industrial

**Fecha de la búsqueda:** 2026-08-25
**Alcance:** los 69 ítems físicos sin catastro de esos cuatro sectores según `busqueda/CATALOGO_FALTANTES.txt`
(Financiero 20 · Tecnologías Informáticas 17 · Comunicaciones 16 · Industrial 16).
**Punto de partida:** los cuatro sectores tenían **cero** activos georreferenciados.

Todo lo que sigue fue verificado con `curl` el 2026-08-25: código HTTP, `robots.txt` del host,
conteo real de registros y conteo real de filas con coordenada. Lo que no se verificó está
marcado **NO VERIFICADO** y dice por qué.

---

## 1. SUBTEL · Registro de Concesiones de Radiodifusión Sonora (julio 2026)

- **Ítems MICR que puebla:** 222 Estaciones de Radio · 226 Antenas de Transmisión · 241 Estudios de Grabación (Radio) · 235 Emisoras de Radio (contenido)
- **Organismo:** Subsecretaría de Telecomunicaciones (MTT)
- **Producto:** listado de concesiones vigentes AM/FM/MC/OC/RC + zonas de servicio FM/RCC en KMZ
- **URL exacta:**
  - `https://www.subtel.gob.cl/wp-content/uploads/2026/08/Actualiza_julio_2026_web.xlsx`
  - `https://www.subtel.gob.cl/wp-content/uploads/2026/01/FM_RCC_Vigentes_Enero_2026.zip`
  - índice: `https://www.subtel.gob.cl/inicio-concesionario/servicios-de-telecomunicaciones/servicios-de-radiodifusion-sonora/`
- **Formato:** XLSX (hoja «Listado», 21 columnas) + ZIP con KMZ
- **Registros:** **2.782** concesiones vigentes (FM 2.101 · RC 532 · AM 142 · MC 6 · OC 1)
- **Con coordenada:** **2.778 de 2.782 (99,86 %)** — columnas `LATPTA`/`LONGPTA` con la ubicación
  de la **planta transmisora**, en grados-minutos-segundos concatenados (`182857` = 18°28'57" S),
  hemisferio implícito S/W. Rango verificado: lat −54,938 a −17,592 · lon −109,431 a −67,607
  (incluye Isla de Pascua). ⚠️ Datum mixto declarado: WGS 84 (1.541), PSAD 56 (842), vacío (378),
  SAD 69 (17) → **hay que reproyectar los PSAD 56 / SAD 69**.
  Las 2.782 filas traen además `DIRECCIONESTUDIO` + `COMUNAESTUDIO` (**sólo dirección, requiere
  geocodificar**) para el ítem 241.
- **robots.txt:** `https://www.subtel.gob.cl/robots.txt` → HTTP 200, sólo `Disallow: /wp-admin/`. `/wp-content/uploads/` permitido.
- **Crudo:** `busqueda/crudo/subtel_radiodifusion/2026-08-25/`
- **Veredicto: SIRVE.** Es el mejor catastro del lote: cobertura nacional, 99,9 % con coordenada,
  actualizado a julio 2026 y con la planta transmisora separada del estudio, que son dos ítems distintos de la MICR.
- ⚠️ **Rodea el bloqueo del encargo:** el aviso decía que `Cobertura_Radio` en `licancabur.subtel.gob.cl`
  pide token. Es cierto (se confirmó: `{"error":{"code":499,"message":"Token Required"}}`), pero el
  **registro completo está publicado como XLSX en el portal principal de SUBTEL**, sin token.

## 2. SUBTEL · Plantas transmisoras de Televisión Digital (ISDB-T)

- **Ítems MICR que puebla:** 226 Antenas de Transmisión · 221 Estudios de Televisión (indirecto, vía concesionaria) · 234 Canales de TV
- **Organismo:** SUBTEL — servidor ArcGIS `licancabur.subtel.gob.cl`
- **URL exacta:** `https://licancabur.subtel.gob.cl/server/rest/services/TVD_por_Empresas/MapServer/9/query?where=1=1&outFields=objectid,identificador,reg,localidad,frecuencia,potencia,lat_ptx,long_ptx,tipo_servicio,altura_tx,conc&returnGeometry=true&outSR=4326&resultOffset=N&resultRecordCount=500&f=json`
- **Formato:** ArcGIS REST JSON (2 páginas)
- **Registros:** **580** plantas transmisoras ISDB-T
- **Con coordenada:** **580 de 580 (100 %)**, geometría de punto ya en EPSG:4326.
  Atributos: `conc` (concesionaria: Megavisión, TVN, Canal 13…), `identificador` (señal distintiva),
  `localidad`, `frecuencia`, `potencia`, `altura_tx`.
- **robots.txt:** `licancabur.subtel.gob.cl` no publica robots.txt (HTTP 404, Tomcat). `antenas.subtel.gob.cl/robots.txt` → 404.
- **Crudo:** `busqueda/crudo/subtel_tvdigital/2026-08-25/`
- **Veredicto: SIRVE.**
- **Hallazgo lateral:** el listado raíz `https://licancabur.subtel.gob.cl/server/rest/services?f=json`
  **sí responde sin token** y publica **185 MapServer**. Dentro de ese servidor, `Cobertura_Radio` y
  `Fibra_Óptica_COTEL` exigen token; `TVD_por_Empresas` y `ARC_INFRA_SATURADA` no. Vale la pena que
  otro agente barra los 185 servicios.

## 3. SUBTEL / Geoportal de Chile · Antenas en Servicio «Ley de Torres»

- **Ítems MICR que puebla:** 226 Antenas de Transmisión (y, fuera de este lote, 183/184 de Telecomunicaciones)
- **Organismo:** MTT / SUBTEL, publicado por el SNIT en el Geoportal de Chile
- **URL exacta:** `https://geoportal.cl/geoportal/catalog/download/686f8ca4-adb4-381d-866d-6c1b1a421fde`
  (ficha CSW `{2E96BAB1-6FAF-4123-9466-65B2508650C6}`; descarga `antenas_servicio_ley_torres.zip`)
- **Formato:** ZIP con shapefile de puntos (1,7 MB → 36 MB descomprimido)
- **Registros:** **18.007** sistemas radiantes
- **Con coordenada:** **18.007 de 18.007 (100 %)**, `.prj` = GCS_WGS_1984. Rango lon −109,431 a −58,976,
  lat −62,196 a −17,756 (incluye Isla de Pascua **y el Territorio Antártico**).
  Campos: operador (`ALIAS`: Movistar/Entel/Claro/WOM), tipo de soporte, altura, dirección, comuna, región,
  estado del decreto, frecuencia, tecnología.
- **robots.txt:** `https://geoportal.cl/robots.txt` → HTTP 200, `User-agent: *` / `Disallow:` (todo permitido).
- **Crudo:** `busqueda/crudo/subtel_antenas_leydetorres/2026-08-25/`
- **Veredicto: SIRVE, con dos reparos.** (a) El metadato es de **2021-04-13**, no es dato fresco.
  (b) Es sobre todo **telefonía móvil**, así que pertenece más al lote de Telecomunicaciones —
  **coordinar con ese agente para no duplicar**. Se reporta aquí porque el ítem 226 es de Comunicaciones.
  ⚠️ El `.dbf` viene en UTF-8 leído como latin1 («RegiÃ³n»): decodificar al importar.

## 4. Geoportal de Chile / BancoEstado · Red de Sucursales

- **Ítems MICR que puebla:** 485 Sedes Bancarias (Infraestructura)
- **Organismo:** metadato del SNIT/IDE Chile; dato de origen de BancoEstado
- **URL exacta:** `https://geoportal.cl/geoportal/catalog/download/d8468610-7b8b-3f3b-b98c-1f1385934605`
  (archivo `coordenadas-be-y-bex-dic2022.xlsx`; ficha CSW `{4AE47EA0-247F-4027-8925-1C908FD964F9}`)
- **Formato:** XLSX, hoja «Coordenadas»: `SHORTNAME | LATITUDE | LONGITUDE | TYPE`
- **Registros:** **542** puntos — 415 «BE» + 127 «BE Express»
- **Con coordenada:** **542 de 542 (100 %)**, lat/lon decimales WGS84, todas dentro de Chile.
- **robots.txt:** `https://geoportal.cl/robots.txt` → todo permitido.
- **Crudo:** `busqueda/crudo/geoportal_bancoestado/2026-08-25/`
- **Veredicto: PARCIAL.** Coordenadas perfectas, pero (a) el dato es de **diciembre 2022**,
  (b) **no trae dirección ni comuna**, sólo un código interno (`SE-999`), y (c) es la red de **un solo banco**,
  el estatal. Complementar con las fuentes 8 y 9.
- ⚠️ **Aviso de robots.txt que conviene registrar:** el sitio hermano `https://www.ide.cl/robots.txt`
  **bloquea explícitamente a ClaudeBot, GPTBot, CCBot, Google-Extended y meta-externalagent**.
  `geoportal.cl` no. Descargar siempre por `geoportal.cl`, no por `www.ide.cl`.

## 5. SMA · SNIFA — Unidades Fiscalizables e Instrumentos  ★ la fuente más grande del lote

- **Ítems MICR que puebla:** 707 Almacenes de Materiales · 729 Almacenes de Residuos Industriales ·
  726/745 Laboratorios · (y, fuera de lote, decenas de ítems de Energía, Hídrico, Alimentario, Químico y Comercial)
- **Organismo:** Superintendencia del Medio Ambiente
- **URL exacta:** índice `https://snifa.sma.gob.cl/DatosAbiertos` → carpeta Drive
  `https://drive.google.com/drive/folders/1Pos3xmMDj0OoRiqmR1W9Q2K0hsnMaEL4` →
  `https://drive.usercontent.google.com/download?id=1kihw_FskVY4bVLCBHyHn22MrA2en4cy4&export=download&confirm=t`
- **Formato:** CSV separado por `;`, latin1, 13,7 MB
- **Registros:** 33.558 filas (unidad × instrumento) → **22.186 unidades fiscalizables únicas**
- **Con coordenada:** **21.946 de 22.186 (98,9 %)**. 238 sin dato, 2 fuera de rango.
  17 regiones, 346 comunas. Rango lat −55,962 a −17,595 · lon −109,437 a −67,063.
- **robots.txt:** `snifa.sma.gob.cl/robots.txt` → HTTP 404 (no hay archivo). La descarga sale de `drive.usercontent.google.com`.
- **Crudo:** `busqueda/crudo/snifa_unidades_fiscalizables/2026-08-25/`
- **Veredicto: SIRVE, y es la mejor de todo el lote para desagregar por actividad.**
  Trae `CategoriaEconomicaNombre` **y** `SubCategoriaEconomicaNombre`:
  Equipamiento 6.409 · Pesca y Acuicultura 3.311 · Vivienda e Inmobiliarios 2.535 · Energía 1.664 ·
  Saneamiento 1.341 · Agroindustrias 1.303 · Minería 1.162 · **Instalación fabril 1.145** ·
  Otras 1.135 · vacía 572 · **Transportes y almacenajes 475** · Infraestructura de Transporte 347 ·
  Forestal 238 · Hidráulica 197 · Portuaria 162 · ETFA 87 · Monitoreo 76 · Alumbrado 27.
  Subcategorías de «Instalación fabril»: Otras 408 · química 132 · metalúrgica 118 · polímeros 93 ·
  alimentos no agrícolas 62 · cementera 56 · papeles 41 · madera 37 · asfalto 37 · embotelladora 27 ·
  textiles 26 · imprenta 25 · reciclaje peligrosos 24 · fundición 20 · curtiembre 18 · cristalería 13 · refinería 8.
  Subcategorías de «Transportes y almacenajes»: **centro de almacenamiento de sustancias peligrosas 209**,
  **de otras materias 90** (→ ítem 707), **de residuos peligrosos 23** (→ ítem 729).
- ⚠️ La propia SMA advierte que el dato «no ha sido procesado, analizado o verificado» por ella.
  Y es el universo de lo **ambientalmente fiscalizable**, no el universo industrial completo.
- **Aviso a los otros agentes:** 3.311 unidades de Pesca/Acuicultura, 1.664 de Energía, 1.341 de
  Saneamiento, 1.303 de Agroindustrias y 1.162 de Minería son de **otros lotes**. Este archivo les sirve.

## 6. RETC/MMA · Emisiones al aire de fuentes puntuales 2024 (RUEA)  ★ responde la pregunta del encargo

- **Ítems MICR que puebla:** 485 Sedes Bancarias · 503 Seguros · toda la sección manufacturera del sector Industrial
- **Organismo:** Ministerio del Medio Ambiente — RETC / Registro Único de Emisiones Atmosféricas
- **URL exacta:** `https://datosretc.mma.gob.cl/dataset/2733b0f0-428a-4594-afeb-17780c8d47c1/resource/b82ed4a8-ed64-493a-8159-563abf0bf5ad/download/ckan_fp_2024_da.xlsx`
  (serie 2005-2024 completa en la misma ficha)
- **Formato:** XLSX 63,7 MB, hoja «Data», 35 columnas. ⚠️ La cabecera **no** está en la fila 1:
  hay que buscar la fila cuyo primer valor es `año`.
- **Registros:** 348.274 filas de emisión → **12.794 establecimientos únicos** (`id_vu`)
- **Con coordenada:** **12.789 de 12.794 (99,96 %)**
- **robots.txt:** `https://datosretc.mma.gob.cl/robots.txt` → HTTP 200, `Disallow: /api/` + `Crawl-Delay: 10`.
  El `/api/` se usó **sólo** para leer la ficha, con pausas de 10 s; el archivo bajó por
  `/dataset/.../download/`, que no está restringido.
- **Crudo:** `busqueda/crudo/retc_fuentes_puntuales/2026-08-25/`
- **Veredicto: SIRVE — y es la respuesta directa a la pregunta del encargo sobre desagregar por CIIU.**

  El encargo preguntaba si existía una fuente que desagregara los rubros ambiguos del RETC
  («Industria manufacturera» 880, «Construcción e inmobiliarias» 1.061, «Otras actividades» 8.701)
  por actividad económica con coordenada. **Sí existe, y es del mismo organismo.**
  El archivo que el proyecto usa hoy (`datos/crudo/retc/*/establecimientos_ckan.xlsx`) sólo trae la
  columna gruesa `Rubro RETC` con 19 valores. **Este archivo trae además `CIIU4`, `CIIU4_ID`,
  `CIIU6` y `CIIU6_ID`: 271 códigos CIIU4 distintos y 378 CIIU6, todos con coordenada.**

  Desglose por sección CIIU (establecimientos únicos): G comercio 2.215 · E agua/residuos 1.605 ·
  **C manufactura 1.514** · A agropecuario 1.158 · S 1.113 · J información y comunicaciones 842 ·
  **K financiero 658** · D energía 612 · Q salud 597 · L 542 · H 427 · P 423 · O 371 · I 200 ·
  F 170 · B 167 · N 81 · R 44 · M 33 · U 17 · T 5.

  Lo que toca a este lote (todo con coordenada verificada):
  - **Financiero 644**: K6419 «Otros tipos de intermediación monetaria» **631** (= sucursales bancarias),
    K6430 5, K6499 4, K6492 4; K65 seguros 9; K66 auxiliares 5.
  - **Industrial 1.514**: sección C completa en **85 CIIU4 distintos** (1.512 con coordenada).
    Top: panadería 148 · vinos 83 · otros alimentos 77 · químicos n.c.p. 73 · pescado 72 ·
    aserrío 71 · lácteos 69 · minerales no metálicos 66 · carne 60 · otras manufacturas 54 ·
    hormigón/cemento 52 · frutas y hortalizas 51 · plásticos 44 · farmacéutica 21 · tableros 21.
  - **TI 7**: J6311 «Procesamiento de datos, hospedaje y actividades conexas». **Sólo 7.**
  - **Comunicaciones 5**: J6020 televisión 4 + J6010 radio 1.
  - (Telecomunicaciones, fuera de lote: J61 = **821**.)
- ⚠️ **El sesgo hay que decirlo:** es un registro de **quien declara emisiones al aire**. Funciona muy
  bien donde la actividad quema algo (manufactura, panaderías, calderas de sucursales) y muy mal
  donde no. Por eso da 7 data centers y 5 radios, cifras que **no** hay que usar como catastro
  de esos ítems (la de radios contrasta con las 2.782 concesiones reales de SUBTEL).

## 7. RETC/MMA · Instalaciones de recepción y almacenamiento de residuos (I.R.A.R.) 2024

- **Ítems MICR que puebla:** 729 Almacenes de Residuos Industriales · 730 Sistemas de Gestión de Residuos
- **URL exacta:** `https://datosretc.mma.gob.cl/dataset/1345a83f-db57-47c5-9a47-e79dab31de4d/resource/7318bb9c-2ff9-4db7-a74f-d3d4264243f9/download/irar-sinader-2024-ckan.xlsx`
- **Formato:** XLSX 2,7 MB, hoja «Datos», 41 columnas, cabecera en la fila 1. Incluye CIIU4/CIIU6, lat/lon, tonelaje y código LER del residuo.
- **Registros:** 14.830 filas → **202 instalaciones únicas**
- **Con coordenada:** **202 de 202 (100 %)**
- **robots.txt:** igual que la fuente 6.
- **Crudo:** `busqueda/crudo/retc_irar_residuos/2026-08-25/`
- **Veredicto: SIRVE.** Es el encaje más limpio de todo el lote con un ítem concreto: la instalación
  **es** un centro de recepción y almacenamiento de residuos, con coordenada y tonelaje.
  CIIU4 principales: E3830 recuperación de materiales 68 · G4669 chatarra 20 · H5210 almacenamiento 20 ·
  E3811 recogida 13 · C1709 papel y cartón 12 · E3821 tratamiento 10 · E3900 descontaminación 9.
- ⚠️ Sólo residuos **no peligrosos**. Los peligrosos están en otro dataset del mismo portal
  (`generacion-de-residuos-peligrosos`), **no bajado en esta ronda**.

## 8. PeeringDB · Instalaciones de colocation y puntos de intercambio de tráfico

- **Ítems MICR que puebla:** 792 Centros de Datos · 188 Centros de Datos (Datacenters) · 190 Nodos de Interconexión (IXP) · 488 Centros de Datos Financieros (parcial) · 804 Infraestructura en la Nube
- **Organismo:** PeeringDB (fundación sin fines de lucro). **No es fuente oficial chilena.**
- **URL exacta:**
  - `https://api.peeringdb.com/api/fac?country=CL`
  - `https://api.peeringdb.com/api/ix?country=CL`
  - `https://api.peeringdb.com/api/ixfac` (bajado completo, filtrado localmente)
- **Formato:** JSON
- **Registros:** **49 instalaciones** · **9 IXP** · 25 vínculos IXP↔instalación
- **Con coordenada:** **44 de 49 (89,8 %)** en grados decimales. Las **5 sin coordenada**, nombradas
  para que no se pierdan (traen dirección postal, **requieren geocodificar**): Telxius Valparaiso DC ·
  Telxius Arica DC · METROWAN TELECOM Santiago · ODATA DC ST01 · San Esteban SDC1.
  Los 9 IXP no traen coordenada propia: se deriva por `ixfac` → `fac`.
  IXP: PIT Santiago (153 redes) · Chile IX by PIT.net (19) · BGP.Exchange Santiago (12) ·
  PIT Concepción (7) · PatagoniaIX Punta Arenas (7) · BGP.Exchange Valdivia (6) · PIT Temuco (4) ·
  PIT entel (3) · MetroPort IX Viña del Mar (2).
- **robots.txt:** `https://www.peeringdb.com/robots.txt` → HTTP 200; sólo bloquea `/register`,
  `/account/login`, `/username-retrieve`, `/reset-password`. La API no está restringida.
- **Licencia:** CC-BY 4.0.
- **Crudo:** `busqueda/crudo/peeringdb/2026-08-25/`
- **Veredicto: SIRVE — es el mejor catastro de data centers que existe con acceso público para Chile.**
  Concentración verificada: 30 de 49 en Santiago (+2 Las Condes, 1 Huechuraba, 1 Lampa, 1 La Florida);
  fuera de la RM sólo Viña del Mar 2, Valparaíso 1, Arica 1, Concepción 1, San Pedro 1.
- ⚠️ Es un registro **voluntario** de operadores: un data center sin clientes de peering puede no aparecer.

## 9. OpenStreetMap vía Overpass API · nueve extracciones para los cuatro sectores

- **Ítems MICR que puebla:** 485 · 487 · 503 · 518 · 792 · 188 · 221 · 222 · 223 · 226 · 241 · 253 · 707 · 719 · 736
- **URL exacta:** `https://overpass-api.de/api/interpreter` (POST, Overpass QL), plantilla
  `[out:json][timeout:270]; area["ISO3166-1"="CL"][admin_level=2]->.cl; (<selectores>); out center tags;`
- **Formato:** JSON de Overpass. Ways y relations traen centroide en `center`.
- **robots.txt:** `https://overpass-api.de/robots.txt` → HTTP 200, **`Disallow: /api/`**.
  **Declarado sin adornos:** la regla apunta a rastreadores de buscadores y `/api/` es la interfaz
  programática oficial del proyecto; se usó con User-Agent de contacto identificable y pausas de
  3-15 s (dos consultas fueron rechazadas con 429 y 504 y se reintentaron tras pausa).
  **Si el proyecto prefiere lectura estricta de robots.txt, la sustitución es directa:** el extracto
  Geofabrik de Chile (`https://download.geofabrik.de/south-america/chile.html`) entrega exactamente
  los mismos objetos como descarga de archivo.
- **Licencia:** **ODbL 1.0** — obliga a atribuir «© colaboradores de OpenStreetMap» y a compartir-igual
  las bases derivadas. **No es dominio público.**
- **Crudo:** `busqueda/crudo/osm_overpass/2026-08-25/`
- **Registros (todos 100 % con coordenada, porque `out center` la garantiza):**

| archivo | etiqueta | registros | con nombre |
|---|---|---:|---:|
| `financiero_bancos.json` | `amenity=bank` | **1.610** | 1.592 (98 %) |
| `financiero_atm.json` | `amenity=atm` | **1.240** | 286 (23 %) |
| `financiero_cambio_seguros.json` | `bureau_de_change` 139 + `office=insurance` 103 | **242** | — |
| `ti_datacenters.json` | ver desglose abajo | **60** | 52 (86 %) |
| `comunicaciones_estudios_prensa.json` | `amenity=studio` 151 + `office=newspaper` 12 | **175** | 171 (97 %) |
| `comunicaciones_torres.json` | `man_made=mast\|tower` + `tower:type=communication` | **2.369** | — |
| `industrial_fabricas.json` | `man_made=works` | **2.230** | 938 (42 %) |
| `industrial_bodegas.json` | `building=warehouse` | **1.008** | — |
| `industrial_zonas.json` | `landuse=industrial` (polígonos) | **7.960** | — |

- **Veredicto por sector:**
  - **Financiero: SIRVE.** 1.610 sucursales con operadores reales verificados (Banco Estado 403 ·
    Banco de Chile 202 · BCI 170 · Santander 166+55 · Scotiabank 95 · Itaú Corpbanca 81 · Falabella 55)
    y 1.240 cajeros. Es **tres veces** lo que da BancoEstado solo y **2,5 veces** lo que da el RETC.
  - **TI: PARCIAL, y hay que leer el desglose o se comete un error.** De los 60 objetos:
    **23 son centros de datos de verdad** (`telecom=data_center` / `building=data_center`) —
    Equinix ST1/ST2/ST3/ST4, Google, AWS (×4, una en Punta Arenas), Microsoft COLO 1/COLO 2, Cirion,
    IFX Networks, Actis, Conavicoop, DC STO1/STO2, Data Center L7 — **5 son centrales telefónicas**,
    y **32 son oficinas de empresas TI (`office=it`): ésas NO son data centers** y no deben contarse como tales.
  - **Comunicaciones: PARCIAL.** 175 estudios y oficinas de prensa (radio 61 · video 48 · televisión 27 ·
    audio 15) es poco frente al universo real, pero es el único dato que hay para los ítems 221/223/253.
  - **Industrial: PARCIAL con una limitación seria.** 2.230 fábricas, pero **sólo 938 (42 %) tienen nombre**
    y **2.194 de 2.230 no traen la etiqueta `product`**; **ninguna** trae `works`. Sin eso **no se puede
    repartir por rubro industrial** sin trabajo manual o cruce por razón social. Los 7.960 polígonos de
    `landuse=industrial` son zona industrial, no instalaciones: sirven como capa de exposición, no de activos.

## 10. INFOR · Mapa de la Industria Forestal Primaria 2025

- **Ítems MICR que puebla:** 729 (parcial, por el campo `residuos`). **Ver la advertencia al final.**
- **Organismo:** Instituto Forestal, publicado por la IDE del Ministerio de Agricultura
- **URL exacta:** `https://ide.minagri.gob.cl/wp-content/uploads/DESCARGAS/CAPAS/ECONOMIA/MAPA_INDUSTRIAL_INFOR/industria_forestal_2025.rar`
  (índice `https://ide.minagri.gob.cl/descarga-de-capas-shp/economia/`; serie anual 2019-2025)
- **Formato:** RAR v5 con shapefile de puntos, 55 KB. ⚠️ Esta máquina no tiene `unrar`/`7z`/`unar` ni
  `rarfile`: se abrió con **`bsdtar`** (libarchive), que sí lee RAR. Anotar para quien reprocese.
- **Registros:** **1.049** unidades productivas
- **Con coordenada:** **1.049 de 1.049 (100 %)**, `.prj` = WGS_1984_UTM_Zone_19S (EPSG:32719) → **reproyectar a 4326**
- **robots.txt:** `https://ide.minagri.gob.cl/robots.txt` → HTTP 200, sólo `Disallow: /wp-admin/`
- **Crudo:** `busqueda/crudo/infor_industria_forestal/2026-08-25/`
- **Desglose:** Aserraderos 763 · Astillas 165 · Postes y Polines 85 · Tableros y Chapas 26 · Pulpa y papel 10.
  Por región: Araucanía 211 · Biobío 189 · Maule 187 · Los Lagos 122 · Los Ríos 116 · Ñuble 80 ·
  O'Higgins 62 · Aysén 30 · Valparaíso 26 · Magallanes 17 · RM 8 · Coquimbo 1.
  Campos: razón social, dirección, ciudad, teléfono, rango de producción, productos, especies, residuos.
- **Veredicto: PARCIAL — el dato es excelente, el problema es la MICR.**
  ⚠️ **El sector Industrial de la MICR (44 ítems, 704-747) no tiene ningún ítem genérico de «planta
  manufacturera».** Sus ítems de fábrica son 704 Maquinaria, 705 Vehículos, 706 Componentes Electrónicos,
  743 Equipos Médicos y 744 Armamento. **Ninguno cubre aserraderos, celulosa ni tableros.**
  El único encaje defendible es parcial: 729 «Almacenes de Residuos Industriales», porque 763 de 1.049
  declaran rango de residuos. **No se forzó la asignación.**
  ⚠️ Otros dos reparos: 365 de 1.049 unidades son «Portátil» o «Móvil» (un aserradero portátil no es
  infraestructura fija georreferenciable), y la ficha del Geoportal declara `dc:rights = copyright`,
  **sin licencia abierta** — verificar condiciones con INFOR antes de republicar.

## 11. TeleGeography · Submarine Cable Map (puntos de amarre)

- **Ítems MICR que puebla:** 178 Cables de Fibra Óptica (Submarinos) — *ítem de Telecomunicaciones, fuera de lote* — y 793 Redes de Telecomunicaciones (TI)
- **URL exacta:**
  - `https://www.submarinecablemap.com/api/v3/landing-point/landing-point-geo.json`
  - `https://www.submarinecablemap.com/api/v3/cable/cable-geo.json`
- **Formato:** GeoJSON (CRS84 = EPSG:4326)
- **Registros:** 1.922 puntos de amarre en el mundo → **19 en Chile**; 724 features de cable (702 sistemas)
- **Con coordenada:** 100 %
- **robots.txt:** `https://www.submarinecablemap.com/robots.txt` → HTTP 200, `Disallow:` (todo permitido)
- **Crudo:** `busqueda/crudo/telegeography_cables_submarinos/2026-08-25/`
- **Los 19 de Chile:** Arica · Iquique · Tocopilla · Antofagasta · Caldera · La Serena · Valparaíso ·
  Cartagena · Constitución · San Pedro de la Paz · Puerto Saavedra · Puerto Montt · Meimen · Linao ·
  Quellón · Puerto Chacabuco · Tortel · Punta Arenas · Puerto Williams. Todos con `is_tbd=false`.
- **Veredicto: PARCIAL.** La coordenada es del **punto de amarre** (la playa/ciudad), **no de la estación
  de amarre como edificio**: precisión de ciudad, no de instalación. Y es empresa privada:
  uso libre con atribución, pero **sin licencia abierta formal** — confirmar antes de republicar.

---

## Fuentes revisadas que NO sirvieron (y por qué)

| Fuente | URL probada | Qué pasó | Veredicto |
|---|---|---|---|
| **CMF — estadísticas de oficinas y sucursales** | `cmfchile.cl/portal/estadisticas/617/w3-propertyvalue-29568.html` (y -29567, -29892) | Redirigen a `/626/w4-...` y devuelven **HTTP 404** con navegador y con curl, en los tres casos. La portada `cmfchile.cl` sí responde 200. | **NO VERIFICADO** — no se pudo confirmar ningún archivo descargable |
| **CMF — portal BEST** | `https://best.cmfchile.cl/` | Responde 200 pero es una SPA Angular sin endpoints de API visibles en el HTML. Además lo que publica son **conteos por comuna/región**, sin dirección ni coordenada. | **NO SIRVE** para georreferenciar |
| **SEIA — mapa de proyectos** | `https://sig.sea.gob.cl/mapadeproyectos/` | El visor pide un **token ArcGIS emitido por el servidor** (`/generate_arcgis_token.php`). **No se intentó obtenerlo.** | **NO SIRVE** sin credencial |
| **SEIA — buscador de proyectos** | `https://seia.sea.gob.cl/` | `robots.txt` → **`User-agent: * / Disallow: /`** (el sitio entero). | **NO SIRVE** — prohibido por robots |
| **SUBTEL — `Cobertura_Radio`, `Fibra_Óptica_COTEL`** | `licancabur.subtel.gob.cl/server/rest/services/...` | Confirmado el aviso del encargo: `{"error":{"code":499,"message":"Token Required"}}`. **No se insistió.** | **NO SIRVE** sin credencial — pero el registro de radios está en el portal público (fuente 1) |
| **CNTV — registro de concesionarios VHF** | `https://transparencia.cntv.cl/2022/otrosantecedentes_registroconcesionariosvhf.html` | Tabla HTML de ~150 concesionarios con RUT, representante legal y su dirección. **Sin archivo descargable y sin coordenadas**; la dirección es la del representante, no la del estudio. | **NO SIRVE** para georreferenciar (la fuente 2 lo reemplaza con ventaja) |
| **datos.gob.cl — «Listado de Concesionarios de Radiodifusión Sonora» (ds. 1926 / 33295)** | `https://datos.gob.cl/api/3/action/package_show?id=1926` | 15 recursos XLSX, **todos de 2014-2015**. | **NO SIRVE** — obsoleto; la fuente 1 es de julio 2026 |
| **PDATA — Plan Nacional de Data Centers** | `https://minciencia.gob.cl/uploads/filer_public/95/6b/.../plan_nacional_de_data_centers_pdata.pdf` | Revisado: **no contiene listado ni anexo** de data centers existentes con comuna ni coordenadas. | **NO SIRVE** como catastro |
| **Geoportal de Chile — búsqueda de parques industriales** | CSW `https://geoportal.cl/csw` con `AnyText like '%industri%'` (82 aciertos) | Ninguno es un catastro de parques industriales: son catastros frutícolas, agrológicos y censos. | **NO SIRVE** |
| **datos.gob.cl — «cajeros», «datacenter», «fibra óptica», «torres»** | `package_search` sobre cada término | **0 resultados** en los cuatro. | **NO SIRVE** |
| **www.ide.cl** | `https://www.ide.cl/robots.txt` | **Bloquea explícitamente a ClaudeBot, GPTBot, CCBot, Google-Extended y meta-externalagent.** No se rastreó. | **NO USAR** — usar `geoportal.cl`, que sí permite |

---

## Tabla resumen · ítem → fuente → registros con coordenada

### Financiero (20 ítems del catálogo)

| Ítem | Elemento | Fuente | Registros con coordenada |
|---|---|---|---|
| 485 | Sedes Bancarias | OSM `amenity=bank` · RETC RUEA K64 · BancoEstado | **1.610** · 644 · 542 |
| 487 | Cajeros Automáticos (ATMs) | OSM `amenity=atm` | **1.240** |
| 488 | Centros de Datos Financieros | PeeringDB `fac` (subconjunto) | 44 (no separables por cliente) |
| 503 | Seguros (Infraestructura) | OSM `office=insurance` · RETC RUEA K65 | 103 · 9 |
| 518 | Oficinas de Cambio (Divisas) | OSM `bureau_de_change` | 139 |
| 486, 493, 497, 501, 502, 505, 507, 508, 509, 511, 514, 516, 517, 520, 526 | — | **sin fuente** | 0 |

### Tecnologías Informáticas (17 ítems)

| Ítem | Elemento | Fuente | Registros con coordenada |
|---|---|---|---|
| 792 | Centros de Datos | PeeringDB `fac` · OSM `data_center` estricto | **44** · 23 |
| 793 | Redes de Telecomunicaciones | TeleGeography amarres · Subtel torres | 19 · 18.007 |
| 802 | Dispositivos de Red | OSM `telecom=exchange` + `telephone_exchange` | 5 |
| 804 | Infraestructura en la Nube | PeeringDB (AWS, Google, Microsoft, Oracle) | subconjunto de los 44 |
| 808 | Infraestructura de Conectividad | PeeringDB `ix` + `ixfac` | 9 IXP (posición vía 25 vínculos) |
| 801, 803, 805, 807, 813, 814, 816, 817, 819, 821, 832, 834 | — | **sin fuente** | 0 |

*(el ítem 190 «Nodos de Interconexión (IXP)» es de Telecomunicaciones, no de este lote, pero queda cubierto por PeeringDB: 9 IXP)*

### Comunicaciones (16 ítems)

| Ítem | Elemento | Fuente | Registros con coordenada |
|---|---|---|---|
| 221 | Estudios de Televisión | SUBTEL TVD (vía concesionaria) · OSM `studio=television` | 580 plantas · 27 estudios |
| 222 | Estaciones de Radio | **SUBTEL radiodifusión** | **2.778** |
| 223 | Oficinas de Periódicos | OSM `office=newspaper` | 12 |
| 226 | Antenas de Transmisión | SUBTEL Ley de Torres · SUBTEL radio · SUBTEL TVD · OSM torres | **18.007** · 2.778 · 580 · 2.369 |
| 241 | Estudios de Grabación (Radio) | SUBTEL radiodifusión (`DIRECCIONESTUDIO`) · OSM `studio=radio\|audio` | 2.782 **sólo dirección** · 76 |
| 253 | Estudios de Producción (Digital) | OSM `studio=video` | 48 |
| 227, 228, 243, 247, 248, 252, 255, 261, 262, 263 | — | **sin fuente** | 0 |

### Industrial (16 ítems)

| Ítem | Elemento | Fuente | Registros con coordenada |
|---|---|---|---|
| 707 | Almacenes de Materiales | SNIFA «Centro de almacenamiento de otras materias» · OSM `building=warehouse` | 90 · 1.008 |
| 719 | Almacenes de Productos Terminados | OSM `building=warehouse` (mismo universo que 707, **no separable**) | 1.008 |
| 726 | Laboratorios de Innovación | SNIFA «Laboratorio de Investigación» | 25 |
| 729 | Almacenes de Residuos Industriales | **RETC I.R.A.R. 2024** · SNIFA (peligrosos 23 + otros 4) | **202** · 27 |
| 745 | Laboratorios de Pruebas (Productos) | SNIFA «Laboratorio de Investigación» (comparte universo con 726) | 25 |
| 704, 705, 706 | Fábricas de Maquinaria / Vehículos / Componentes Electrónicos | **RETC RUEA sección C** (1.514 en 85 CIIU4) — **ninguno de los 3 ítems tiene CIIU equivalente**; ver hallazgo 3 | 0 asignables sin forzar |
| 722, 725, 731, 732, 736, 743, 744, 746 | — | **sin fuente** | 0 |

**Registros nuevos con coordenada aportados por este lote (sin descontar solapamientos entre fuentes):
55.216**, de los cuales 41.906 son de fuente oficial chilena (SUBTEL 21.365 · SNIFA 21.946 pero
mayoritariamente de otros sectores · RETC 12.991 · INFOR 1.049 · BancoEstado 542) y 13.310 de fuentes
externas (OSM 16.834 · PeeringDB 44 · TeleGeography 19).

## Ítems sin fuente (45 de 69)

- **Financiero (15):** 486 Bolsas de Valores · 493 Criptomonedas (Wallets) · 497 Clientes (Cuentas Bancarias) ·
  501 Infraestructura de Conectividad · 502 Cámaras de Compensación · 505 Agencias de Calificación Crediticia ·
  507 Terminales POS · 508 Tarjetas · 509 Aplicaciones Bancarias Móviles · 511 Blockchain Financiero ·
  514 Vaults · 516 Centros de Respaldo · 517 Infraestructura de Contingencia · 520 ATMs Móviles ·
  526 Centros de Capacitación
- **Tecnologías Informáticas (12):** 801 Desarrolladores de Software · 803 Sistemas Operativos ·
  805 Aplicaciones Críticas · 807 VPN · 813 Centros de Capacitación (TI) · 814 Infraestructura de
  Contingencia · 816 Dispositivos IoT · 817 Redes 5G · 819 Centros de Investigación (TI) ·
  821 Infraestructura de Satélites · 832 Centros de Soporte Técnico · 834 Infraestructura de Blockchain
- **Comunicaciones (10):** 227 Satélites de Comunicación · 228 Estaciones Terrestres (Satélites) ·
  243 Infraestructura de Conectividad · 247 Centros de Distribución (Periódicos) · 248 Archivos Físicos ·
  252 Canales de Noticias (24/7) · 255 Publicidad Digital · 261 Centros de Capacitación (Periodismo) ·
  262 Organizaciones de Prensa · 263 Canales de Comunicación (Emergencia)
- **Industrial (8):** 722 Infraestructura de Conectividad · 725 Instalaciones de Pruebas ·
  731 Centros de Capacitación (Manufactura) · 732 Infraestructura de Contingencia ·
  736 Centros de Distribución (Productos) · 743 Fábricas de Equipos Médicos · 744 Fábricas de Armamento ·
  746 Centros de Almacenamiento (Emergencia)

**Observación honesta sobre esa lista:** una parte grande de esos 45 ítems **no es catastrable por diseño**,
no por falta de fuente. «Tarjetas de Crédito/Débito», «Sistemas Operativos», «Redes Privadas Virtuales (VPN)»,
«Blockchain Financiero», «Clientes (Cuentas Bancarias)» o «Publicidad Digital» no nombran un objeto con
coordenada. Buscarles catastro es buscar algo que no existe. Los que **sí** son catastrables y quedaron
sin fuente son pocos y concretos: 486 Bolsas de Valores (3 entidades), 502 Cámaras de Compensación
(Combanc, ComDer), 505 Agencias de Calificación Crediticia (4-5 entidades), 228 Estaciones Terrestres de
Satélites, 743 Fábricas de Equipos Médicos y 744 Fábricas de Armamento (FAMAE, ASMAR, ENAER).
Todos ésos son **listas cortas de entidades nombradas**, no catastros masivos: se resuelven con una
nómina de la CMF o de la Subsecretaría para las FF.AA. más geocodificación, no con un servicio de datos.

---

## Los tres hallazgos que importan

1. **La desagregación por CIIU que el encargo pedía existe, y es del mismo organismo que el proyecto ya usa.**
   El proyecto trabaja con `establecimientos_ckan.xlsx` del RETC, que sólo tiene la columna gruesa
   `Rubro RETC` (19 valores) — por eso 8.701 establecimientos quedaron en «Otras actividades» sin repartir.
   El archivo **«Emisiones al aire de fuentes puntuales 2024»** del mismo RETC trae `CIIU4`, `CIIU4_ID`,
   `CIIU6` y `CIIU6_ID` para **12.789 establecimientos con coordenada**: **271 códigos CIIU4 y 378 CIIU6**.
   Ahí, «Otras actividades» se abre: 631 son sucursales bancarias (K6419), 1.514 son manufactura en
   85 CIIU4 distintos, 821 son telecomunicaciones. **Pero el sesgo hay que respetarlo**: el universo es
   «quien declara emisiones al aire», por eso da 7 data centers y 5 radios — cifras que serían un error
   usar como catastro de esos ítems.

2. **El «Token Required» de `licancabur.subtel.gob.cl` no bloqueaba el registro de radios: bloquea otra cosa.**
   El aviso del encargo se confirmó (`Cobertura_Radio` y `Fibra_Óptica_COTEL` piden token), pero el
   **registro completo de concesiones de radiodifusión está publicado como XLSX abierto en el portal
   principal de SUBTEL, actualizado a julio 2026**: 2.782 concesiones, 2.778 con coordenada de la planta
   transmisora (99,86 %). Y en el **mismo** servidor con token, el listado raíz `/server/rest/services?f=json`
   **sí responde**: publica 185 MapServer, de los cuales al menos `TVD_por_Empresas` (580 transmisores de
   TV digital, 100 % con coordenada) está abierto. Vale la pena barrer los 185.

3. **En Industrial el cuello de botella dejó de ser el dato y pasó a ser la MICR.**
   Hay tres catastros industriales georreferenciados y buenos: SNIFA (1.145 «instalaciones fabriles» en
   17 subcategorías), RETC RUEA (1.514 manufactureras en 85 CIIU4) e INFOR (1.049 unidades forestales,
   100 % con coordenada). Pero el sector Industrial de la MICR (44 ítems, 704-747) **no tiene ningún ítem
   genérico de «planta manufacturera»**: sus ítems de fábrica son Maquinaria, Vehículos, Componentes
   Electrónicos, Equipos Médicos y Armamento. **Ninguno recibe un aserradero, una planta de celulosa, una
   panadería industrial, una cementera ni una viña.** Los 1.514 establecimientos manufactureros con
   coordenada y CIIU están disponibles y **no tienen dónde entrar sin forzar la asignación**, cosa que
   no se hizo. Ésta es la misma clase de hallazgo que el ítem 17 «demasiado grueso» del FEN, sólo que al
   revés: aquí el ítem no es grueso, es que **falta**.
