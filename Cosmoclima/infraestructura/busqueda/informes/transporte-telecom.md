# Catastros públicos georreferenciados · lote **Transporte** y **Telecomunicaciones**

**Buscado, verificado y bajado el 25-ago-2026.**
Crudo en `busqueda/crudo/<clave>/2026-08-25/`, cada carpeta con su `PROCEDENCIA.txt`.

Regla que se siguió en todo: **nada se dio por supuesto**. Cada servicio se
consultó de verdad (`?f=json`, `returnCountOnly`, y después la bajada real), se
leyó el `robots.txt` de su host, y **se contó la geometría según el tipo de la
capa** — punto por `x`/`y`, línea por `paths`, polígono por `rings` — no el
número de filas. Donde no se pudo verificar, dice **NO VERIFICADO**.

---

## 0 · Resumen en una pantalla

| | |
|---|---|
| fuentes nuevas verificadas y bajadas | **5** (59 servicios ArcGIS REST, 73 capas) |
| registros efectivamente bajados | **90.544** features, **90.530 con geometría** (99,98 %) |
| universo alcanzable declarado por los servicios (redes de acceso, sólo muestreado) | **2.280.118** |
| ítems del lote que quedan poblados | **13 de 43** (7 completos, 6 parciales) |
| ítems del lote sin ninguna fuente | **30** |
| peso en disco | 183 MB |

Los tres hallazgos que importan están en §4.

---

## 1 · TRANSPORTE

### 1.1 · SENAPRED · SIIE `RedComunicaciones` y `PlanesProteccion` ★★★

- **Ítems MICR que puebla:**
  **619 Túneles de Carreteras**, **620 Ferrocarriles (Vías)**,
  **621 Estaciones de Tren**, **650 Rutas de Evacuación**,
  y parcialmente **617 Carreteras Secundarias (Rutas Rurales)**.
- **Organismo · producto:** SENAPRED — Servicio Nacional de Prevención y
  Respuesta ante Desastres · Visor GRD, sistema SIIE.
- **URL exacta:**
  `https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/RedComunicaciones/MapServer`
  y `.../SIIE/PlanesProteccion/MapServer`
- **Formato:** esriJSON (ArcGIS REST). Convertido a GeoJSON por el bajador de
  esta búsqueda. Reproyección a EPSG:4326 la hace el servidor (`outSR=4326`).
- **robots.txt:** `https://visor-grd.senapred.gob.cl/robots.txt` → **HTTP 200**,
  contenido `User-agent: *` + `Disallow:` (línea vacía). Es la forma explícita
  de decir «no restrinjo nada».
- **Acceso:** anónimo, sin credenciales.

| capa | geometría | declarados | recibidos | **con geometría** | ítem |
|---|---|---:|---:|---:|---|
| `RedComunicaciones/3` Túneles | Polyline | 58 | 58 | **58** | 619 |
| `RedComunicaciones/12` Línea Férrea | Polyline | 933 | 933 | **933** | 620 |
| `RedComunicaciones/7` Metro · Red | Polyline | 48 | 48 | **48** | 620 |
| `RedComunicaciones/6` Metro · Estación | Point | 132 | 132 | **132** | 621 |
| `RedComunicaciones/9` Huellas y senderos | Polyline | 5.241 | 5.241 | **5.241** | 617 (parcial) |
| `RedComunicaciones/1` Balsas | Point | 25 | 25 | **25** | — (fuera de lote) |
| `PlanesProteccion/4` Vías de evacuación | Polyline | 2.740 | 2.740 | **2.740** | 650 |
| `PlanesProteccion/1` Límite del área de evacuación | Polyline | 539 | 539 | **539** | — (fuera de lote) |
| | | **9.716** | **9.716** | **9.716** | |

- **Crudo:** `busqueda/crudo/senapred_redcom/2026-08-25/` (28 MB).
- **Veredicto: SIRVE.** Cierre exacto capa por capa y el 100 % trae geometría;
  es la única fuente pública encontrada que georreferencia a la vez vía férrea,
  túneles y vías de evacuación.

**Advertencias que hay que llevarse escritas:**

1. ⚠️ **La Línea Férrea NO cubre Chile entero.** Los 933 tramos están en 8
   regiones: Antofagasta 269, Atacama 190, Coquimbo 131, O'Higgins 94, Biobío
   90, La Araucanía 81, Valparaíso 58, Metropolitana 20. **Faltan Maule, Ñuble,
   Los Ríos y Los Lagos** — o sea, buena parte de la red sur de EFE. Tratarlo
   como inventario nacional sería falso.
2. ⚠️ **«Estaciones de Tren» son 132 estaciones del Metro de Santiago**, no
   estaciones de ferrocarril. El campo `tipo` lo dice: Línea 1/2/3/4/4A/5/6,
   combinaciones, terminales y hasta 2 «Estaciones Fantasma». Las 132 están en
   la Región Metropolitana. **No hay ni una estación de EFE, Merval o Biotrén**
   en esta capa, y no se encontró catastro público que las tenga.
3. La capa de Túneles declara `copyrightText`: *«SIG y Cartografía, Agosto,
   2009.»* — el dato puede tener 17 años.
4. «Huellas y senderos» son huellas y senderos, no carreteras secundarias
   pavimentadas. Sirve como capa rural complementaria, **no** como el ítem 617
   completo.

### 1.2 · MOP · SIT-MOP: Vialidad, Concesiones (CCOP), ILOG y DAP ★★

- **Ítems MICR que puebla:** **619 Túneles de Carreteras**,
  **642 Estaciones de Servicio (Combustible)** (parcial),
  **625 Torres de Control Aéreo** y **626 Terminales de Pasajeros
  (Aeropuertos)** (ambos sólo por *proxy*, ver abajo).
- **Organismo · producto:** MOP — UGIT, Dirección de Vialidad, Dirección General
  de Concesiones (CCOP), Unidad de Infraestructura Logística (ILOG), Dirección
  de Aeropuertos (DAP).
- **URL base:** `https://rest-sit.mop.gob.cl/arcgis/rest/services`
- **Formato:** esriJSON, ArcGIS Server **10.2.1**.
- **robots.txt:** `https://rest-sit.mop.gob.cl/robots.txt` → **HTTP 404**. El
  host no declara restricción de rastreo (coincide con lo que ya documentaba
  `FUENTE_MOP.md`).
- **Acceso:** anónimo.

| capa | geometría | declarados | recibidos | **con geom.** | ítem |
|---|---|---:|---:|---:|---|
| `VIALIDAD/Infraestructura_Vial/3` Túnel | Polyline | 34 | 34 | **34** | 619 |
| `CCOP/INFRA_DGC/2` Túneles (concesionados) | Point | 25 | 25 | **25** | 619 |
| `CCOP/INFRA_DGC/8` Áreas de Servicio | Point | 64 | 64 | **64** | 642 (parcial) |
| `CCOP/INFRA_DGC/7` Áreas de Venta | Point | 40 | 40 | **40** | 642 (parcial) |
| `CCOP/INFRA_DGC/0` Viaductos | Point | 6 | 6 | **6** | — |
| `CCOP/INFRA_DGC/5` Pasarelas | Point | 464 | 464 | **464** | — |
| `ILOG/INFRA_ILOG/0` Infra. Portuaria | Point | 644 | 644 | **644** | 623 (no sirve, ver §3) |
| `ILOG/INFRA_ILOG/1` Infra. Aeroportuaria | Point | 327 | 327 | **327** | 625/626 (proxy) |
| `ILOG/INFRA_ILOG/2` Infra. Vial | Polyline | 450 | 450 | **450** | — |
| `DAP/Red_Aeroportuaria_Nacional/0` | Point | 316 | 316 | **316** | 625/626 (proxy) |
| | | **2.370** | **2.370** | **2.370** | |

- **Crudo:** `busqueda/crudo/mop_transporte/2026-08-25/` (31 MB).
- **Veredicto: PARCIAL.** Los túneles y las áreas de servicio de concesiones son
  dato limpio y cierran exacto; los aeropuertos sólo sirven como *proxy* porque
  la fuente no cataloga ni torres de control ni terminales de pasajeros.

**Lo que hay que saber:**

- **619 se puede triangular con tres fuentes independientes** (34 MOP Vialidad +
  25 MOP Concesiones + 58 SENAPRED = 117 registros con solape desconocido). Son
  tres censos distintos del mismo objeto y **no se suman**: hay que cruzarlos por
  geometría antes de contar.
- **625 y 626 no existen como capa.** `DAP/Red_Aeroportuaria_Nacional` da 316
  sitios con `TIPO` (**7 Aeropuertos**, 309 Aeródromos), `RED` (17 Primaria, 12
  Secundaria, 287 Pequeño Aeródromo), `USO` (127 Público, 189 Privado) y código
  OACI/IATA. Los 7 aeropuertos y los 17 de red primaria **son** los recintos que
  tienen torre de control y terminal de pasajeros, pero eso es una **inferencia
  nuestra, no un dato de la fuente**. Declararlo así o no declararlo.
- ★ `ILOG/INFRA_ILOG/1` trae **327** sitios contra los **316** de DAP: los 11
  extra son la **red militar** (`RED='Militar'`, `PUBLICABLE='No'`), que el
  servicio de la DAP no publica y el de ILOG sí. Es una fuga de la propia fuente,
  y conviene decidir explícitamente si se usa.
- `CCOP/INFRA_DGC/8` «Áreas de Servicio» (64) son las áreas de servicio de las
  rutas concesionadas — incluyen estación de combustible, pero **no son el
  catastro nacional de estaciones de servicio**. Cobertura: 13 regiones.

---

## 2 · TELECOMUNICACIONES

> El hallazgo de fondo: SUBTEL publica **sin credenciales** las declaraciones de
> **infraestructura crítica de telecomunicaciones** que los operadores le
> entregan por oficio (**Of. 213/2024** red troncal, **Of. 322** elementos de
> red de acceso, **Of. 468** trazas de red de acceso). Son **56 servicios**
> ArcGIS que nadie había mirado — el proyecto sólo tenía 4 servicios de Subtel
> espejados (antenas y estaciones base).

### 2.1 · SUBTEL · Of.213/2024 — red **troncal** de fibra óptica ★★★

- **Ítems MICR que puebla:** **177 Cables de Fibra Óptica (Terrestres)** y,
  parcialmente, **178 Cables de Fibra Óptica (Submarinos)**.
- **Organismo · producto:** SUBTEL · declaraciones de infraestructura crítica,
  Oficio 213/2024, red troncal por operador.
- **URL base:** `https://licancabur.subtel.gob.cl/server/rest/services`
- **Formato:** esriJSON, ArcGIS Server **11.5** (sí soporta `resultOffset`,
  a diferencia del MOP).
- **robots.txt:** `https://licancabur.subtel.gob.cl/robots.txt` → **HTTP 404**
  (sin restricciones). `https://www.subtel.gob.cl/robots.txt` → 200, sólo
  bloquea `/wp-admin/`.
- **Acceso:** anónimo. `copyrightText` **vacío en todas las capas**: sin
  licencia declarada.

| servicio | operador | declarados | recibidos | **con geometría** |
|---|---|---:|---:|---:|
| `Red_Ufinet_of213_2024_IC/0` | Ufinet | 37.700 | 37.700 | **37.699** |
| `Red_Troncal_ClaroVTR/1` | Claro / VTR | 2.962 | 2.962 | **2.962** |
| `Red_Sheath_Troncal_Entel_of213_2024_IC/0` | Entel | 1.290 | 1.290 | **1.290** |
| `Red_LD_Telefonica_of213_2024_IC/0` | Telefónica (larga distancia) | 231 | 231 | **231** |
| `Red_Telesat_TS_of213_2024_IC/0` | Telsur | 111 | 111 | **110** |
| `Red_kmz_CMET_of213_2024_IC/0` | CMET | 106 | 106 | **106** |
| `Red_Holdco_MP_of213_2024_IC/0` | Holdco | 20 | 20 | **20** |
| `Silica_Networks_of213_2024/19` | Silica Networks | 10 | 10 | **10** |
| `Red_Cirion_of213_2024_IC/0` | Cirion | 9 | 9 | **9** |
| `Red_kmz_Cirion_of213_2024_IC/0` | Cirion (kmz) | 7 | 7 | **7** |
| **total** | | **42.446** | **42.446** | **42.444** |

- Todas las capas son `esriGeometryPolyline`. **2 registros llegaron sin
  geometría** (1 de Ufinet, 1 de Telsur) y quedan declarados como hueco, no
  rellenados.
- **Atributos utilizables**, y son buenos: `TIPO_TENDI` (Aéreo 1.752 / Mixto 622
  / Canalizado 496 / En interior 27 / En azotea 3 / Desconocida 62 en Claro-VTR),
  `fijacion` (Subterráneo 367 / Aéreo 918 en Entel), `CAPACIDAD` y
  `fiber_coun` (nº de filamentos), `capacidad_gb`, `tecnología` (DWDM),
  `nivel_red`, `decreto`. **El tendido aéreo es exactamente la fibra que se corta
  con viento y con caída de árboles** — es un discriminador climático directo.
- **Crudo:** `busqueda/crudo/subtel_fibra_troncal_of213/2026-08-25/` (95 MB).
- **Veredicto: SIRVE**, y es la fuente más importante de todo el lote.

**Sobre el ítem 178 (submarinos) — PARCIAL y hay que decirlo así:** dentro de
estas mismas capas hay **2 tramos etiquetados explícitamente como submarinos**:
`Red_Cirion` objectid 8 (`nombrered='Cable Submarino SAC'`, `tipotendid='Submarino'`)
y `Red_LD_Telefonica` objectid 222 (`name='Cable Submarino'`). No es el catastro
de cables submarinos de Chile; es lo que dos operadores declararon. *(Otro agente
de esta misma búsqueda trabajó `telegeography_cables_submarinos`; conviene
cruzarlo con estos 2 antes de dar por poblado el 178.)*

### 2.2 · SUBTEL · Of.213 nodos + Of.322 OLT y NodoHFC ★★

- **Ítems MICR que puebla:** **198 Routers de Distribución (Carrier-Grade)**
  [los OLT], **199 Switches de Distribución Local** [los nodos troncales],
  **196 Redes de Distribución (Cables de Cobre)** [los NodoHFC].
- **URL base:** `https://licancabur.subtel.gob.cl/server/rest/services`
- **robots.txt:** 404 (sin restricciones). **Acceso:** anónimo.

| grupo | servicios | declarados | recibidos | **con geometría** |
|---|---:|---:|---:|---:|
| Nodos troncales Of.213 (Ufinet 4.392, Telsur-TS 1.348, Claro/VTR 124, Entel hub 80, Telefónica 7 tramos 134, Telsur-CN 66, CMET 37, Internexa 23, Holdco 10, Cirion 4+6) | 17 | 6.224 | 6.224 | **6.212** |
| Sitios y POI terrestres de Silica Networks | 2 | 26 | 26 | **26** |
| OLT Of.322 (GTD 571, Mundo 567, Entel 149, Wom 142, VTR 124, Claro 34) | 6 | 1.587 | 1.587 | **1.587** |
| NodoHFC Of.322 (VTR 121, Claro 54) | 2 | 175 | 175 | **175** |
| **total** | **27** | **8.012** | **8.012** | **8.000** |

- **12 registros sin geometría** (11 de nodos Ufinet, 1 de Internexa), declarados.
- **Crudo:** `busqueda/crudo/subtel_nodos_of213/2026-08-25/` (6,2 MB).
- **Veredicto: SIRVE.** Bajado completo, cierre exacto.

### 2.3 · SUBTEL · Of.322 CTO/TAP y Of.468 red de acceso — **MUESTRA**

- **Ítems MICR que puebla:** **195 Redes de Distribución (Fibra Óptica Local)**,
  **196 Redes de Distribución (Cables de Cobre)**,
  **197 Gabinetes de Calle (Distribución)**.
- **URL base:** `https://licancabur.subtel.gob.cl/server/rest/services`
- **robots.txt:** 404 (sin restricciones). **Acceso:** anónimo, salvo dos
  servicios que piden token (ver §5).

| grupo | ítem | servicios | **declarados por el servicio** | bajado ahora |
|---|---|---:|---:|---:|
| CTO — Caja Terminal Óptica (OnNet 351.369, Mundo 278.314, Entel GO 43.649, Entel RIG 42.539, VTR 28.055, Entel DC 27.852, Claro 19.786, Wom 11.704) | **197** | 8 | **803.268** | 16.000 |
| TAP — punto de acometida HFC (VTR 550.166, Claro 176.380) | **196** | 2 | **726.546** | 4.000 |
| Of.468 red de acceso **fibra** (Infraco 311.419, GTD-CN 134.921, GTD-TS 108.324, Claro 76.217, Entel distribución 53.654, VTR FTTH 32.337, Entel primarias 9.626) | **195** | 7 | **726.498** | 7.000 |
| Of.468 red de acceso **HFC/coaxial** (Claro) | **196** | 1 | **23.806** | 1.000 |
| **total** | | **18** | **2.280.118** | **28.000** |

- **La muestra bajada trae 28.000 de 28.000 con geometría (100 %).** Puntos con
  `x`/`y`; líneas con `paths`.
- **Por qué muestra y no todo:** 2,28 millones de elementos superan con holgura
  el tope de ~200 MB fijado para esta búsqueda (la sola muestra de 28.000 pesa
  23 MB → el universo completo rondaría **1,8 GB**). Se bajaron 1.000–2.000
  registros por capa **con su conteo declarado exacto**, para dejar probado el
  acceso, el esquema y la calidad geométrica. La bajada completa es factible y
  queda pendiente como decisión del proyecto, no como límite técnico.
- **Crudo:** `busqueda/crudo/subtel_red_acceso_of322_of468/2026-08-25/` (23 MB).
  Los archivos llevan el prefijo `MUESTRA_` y el `metadata.muestra=true` dentro
  del GeoJSON, para que nadie los confunda con un censo.
- **Veredicto: PARCIAL por decisión de tamaño, no por la fuente.** La fuente
  SIRVE y está completa; lo que está incompleto es la copia local.

### 2.4 · Ítem 184 (Antenas de Microondas) — no hace falta bajar nada

Revisando el crudo que el proyecto **ya tenía** desde el 20-ago
(`datos/crudo/subtel/2026-08-20/antenas_en_servicio.geojson.gz`, 29.875
elementos), el campo `banda` separa **SHF = 3.904 elementos** de UHF (25.876) y
VHF (22). SHF es la banda de microondas, y las frecuencias lo confirman (7.135 /
15.089 / 17.765 / 21.812 / 23.012 / 23.070 MHz). **El ítem 184 se puede poblar
filtrando dato que ya está en disco**, sin una sola petición nueva.
**Veredicto: SIRVE (derivado, sin descarga).**

---

## 3 · Fuentes que se probaron y NO sirvieron

| fuente probada | resultado verificado | por qué no sirve |
|---|---|---|
| `ILOG/INFRA_ILOG/0` para el ítem **623 Terminales de Contenedores** | bajada, 644/644 con geometría | El campo `OBRA` es Caleta Pesquera 172, Embarcadero 185, Paseo Costero 99, Defensa Costera 77, Rampa 76, **Terminal de Conectividad 15**, Muelle 9, Playa Artificial 7, Puerto de Conectividad 1. Ningún terminal de contenedores: son obras portuarias menores. Además duplica `DOP/CATASTRO_DOP/0` y el proyecto ya lo tiene en `inventario_obras_portuarias.csv`. **NO SIRVE para 623.** |
| `ILOG/INFRA_ILOG/3` para el ítem **640 Centros de Logística** | consultada, 49 filas | Es una **Table** sin geometría (`geometryType: null`). **NO SIRVE.** |
| `datos.gob.cl` | robots.txt 200 | Declara `Disallow: /api/` y `Crawl-Delay: 10`. La API CKAN, que es la única vía útil de descarga masiva, **está expresamente desautorizada**. No se usó. |
| `www.ide.cl` (IDE Chile) | robots.txt 200 | Bloque gestionado por Cloudflare con `User-agent: ClaudeBot / Disallow: /` (y lo mismo para GPTBot, CCBot, Google-Extended, Bytespider…) más `Content-Signal: ai-train=no`. Se respetó y **no se descargó nada de ese host**. |
| `api.cne.cl/v3/` (CNE — estaciones de servicio, ítems 642/644) | verificada: existe y responde | **Pide token.** La app pública `bencinaenlinea.cl` lleva un token embebido en su `config.json`; **no se usó** — es la credencial de otra aplicación. Lo correcto es pedirle a la CNE un token propio. **NO VERIFICADO como fuente del proyecto.** |
| `apps.sec.cl`, `www.sec.cl` (SEC, combustibles) | robots 404 / 200 (`Allow: /`) | No se encontró ningún endpoint público georreferenciado de instalaciones de combustible. **NO VERIFICADO.** |
| `ecocarga.cl` (electrolineras, ítem 643) | **DNS no resuelve** (código 000) | El portal de MinEnergía enlaza a Ecocarga pero el host no respondió. **NO VERIFICADO.** |
| `energiaabierta.cl` | responde sólo por **http** (https falla) | robots: sólo bloquea `/wp-admin/`. Es un WordPress; no se halló servicio geoespacial propio. **NO VERIFICADO.** |
| `sig.minenergia.cl` | host vivo, robots 404 | No hay ArcGIS en `/arcgis/rest/services` ni en `/server/rest/services` (404 de IIS). **NO SIRVE.** |
| `geoportal.mtt.gob.cl`, `geonode.ide.cl` | **DNS no resuelve** | **NO VERIFICADO.** |

---

## 4 · Los tres hallazgos que hay que llevarse

**★★★ 1. SUBTEL publica en abierto la infraestructura crítica de
telecomunicaciones del país, y estaba sin tocar.** El servidor `licancabur`
expone **185 servicios**, de los cuales **56 son de red** (troncal, nodos, red de
acceso, CTO, OLT, TAP) publicados bajo los oficios 213/2024, 322 y 468. El
proyecto tenía espejados 4. Con esto el sector Telecomunicaciones pasa de 1 ítem
poblado (183, Torres) a **8 ítems**, y el frente ya no es «no hay dato»: es
**2,28 millones de elementos declarados**, sin credenciales y sin robots que lo
impidan. Además vienen con `TIPO_TENDIDO`, o sea **se sabe qué fibra va por el
aire** — que es justo la que se corta con temporal.

**★★ 2. «Estaciones de Tren» en Chile, en fuente pública georreferenciada, son
132 estaciones del Metro de Santiago y nada más.** Y la vía férrea publicada
cubre 8 regiones de 16: **no hay ni un metro de línea al sur de La Araucanía, ni
Maule, ni Ñuble, ni Los Ríos, ni Los Lagos**. Chile tiene EFE, Merval y Biotrén
operando ahí, y ningún organismo los publica georreferenciados. **El silencio del
instrumento coincide otra vez con la zona donde el fenómeno vive** — es el mismo
patrón ya anotado para los caudalímetros y el satélite. Poblar el ítem 620 con
933 tramos y decir «red ferroviaria nacional» sería falso.

**★ 3. Un conversor que sólo lee puntos habría perdido dos tercios de este lote
sin avisar.** De los 90.544 registros bajados, **60.489 son líneas** (el 67 %) (fibra troncal,
vía férrea, túneles, vías de evacuación, red de acceso): su geometría viene en
`paths`, no en `x`/`y`. Se contó por tipo y el resultado es 90.530 con geometría.
Los **14 huecos reales** (2 en fibra troncal, 12 en nodos) están declarados uno a
uno en los `PROCEDENCIA.txt` y no se rellenaron. Aparte, **dos capas de SUBTEL
devuelven `x`/`y` como *string***, no como número: el bajador reventó con
`TypeError` en Ufinet e Internexa hasta que se le puso coerción — si en vez de
reventar hubiera devuelto `None`, habrían quedado 4.415 nodos «sin coordenada»
en silencio.

---

## 5 · Servicios que piden token (anotados, no forzados)

Se toparon y **se dejaron pasar sin intentar saltarlos**:

| servicio | respuesta exacta |
|---|---|
| `Fibra_Óptica_COTEL/MapServer` | `{"code": 499, "message": "Token Required"}` |
| `CTO_COTEL/MapServer` | `{"code": 499, "message": "Token Required"}` |
| `Of468_CTR_RedAcceso/MapServer` | `{"code": 499, "message": "Token Required"}` |
| `Of468_Mundo_RedAcceso/MapServer` | `{"code": 499, "message": "Token Required"}` |
| `api.cne.cl/v3/` (CNE, combustibles) | requiere token propio |

Los dos `Of468_*` con token son **red de acceso de CTR y de Mundo**: es dato del
mismo tipo que sí está abierto para los otros 8 operadores, o sea que el ítem 195
queda con un hueco conocido y acotado. Vale la pena pedirle acceso a SUBTEL.

---

## 6 · Tabla resumen · ítem MICR → fuente → registros

### Telecomunicaciones (25 ítems del lote)

| ítem | nombre | fuente | registros | veredicto |
|---:|---|---|---:|---|
| **177** | Cables de Fibra Óptica (Terrestres) | SUBTEL Of.213/2024, 10 servicios | **42.446** (42.444 c/geom) | SIRVE |
| **178** | Cables de Fibra Óptica (Submarinos) | SUBTEL Of.213 (etiquetados «Submarino») | **2** | PARCIAL |
| 179 | Satélites de Telecomunicaciones (GEO) | — | 0 | sin fuente |
| 180 | Satélites de Telecomunicaciones (LEO) | — | 0 | sin fuente |
| 181 | Satélites de Telecomunicaciones (MEO) | — | 0 | sin fuente |
| 182 | Estaciones Terrestres de Satélites | — | 0 | sin fuente |
| **184** | Antenas de Microondas | SUBTEL Ley de Torres, ya en disco, filtro `banda='SHF'` | **3.904** | SIRVE (derivado) |
| 187 | Infraestructura de Edge Computing | — | 0 | sin fuente |
| 188 | Centros de Datos (Datacenters) | — | 0 | sin fuente |
| 190 | Nodos de Interconexión (IXP) | — | 0 | sin fuente |
| 192 | Hardware de Sistemas de Control (NOC) | — | 0 | sin fuente |
| **195** | Redes de Distribución (Fibra Óptica Local) | SUBTEL Of.468, 7 servicios | **726.498** declarados · 7.000 bajados | PARCIAL (muestra) |
| **196** | Redes de Distribución (Cables de Cobre) | SUBTEL Of.468 HFC + Of.322 TAP + NodoHFC | **750.527** declarados · 5.175 bajados | PARCIAL (muestra) |
| **197** | Gabinetes de Calle (Distribución) | SUBTEL Of.322 CTO, 8 servicios | **803.268** declarados · 16.000 bajados | PARCIAL (muestra) |
| **198** | Routers de Distribución (Carrier-Grade) | SUBTEL Of.322 OLT, 6 servicios | **1.587** | SIRVE |
| **199** | Switches de Distribución Local | SUBTEL Of.213 nodos, 19 servicios | **6.250** (6.238 c/geom) | SIRVE |
| 202–206 | Teléfonos, computadores, IoT | — | 0 | sin fuente |
| 211 | Radios de Comunicación (Emergencia) | — | 0 | sin fuente |
| 212 | Redes Satelitales Portátiles (Emergencia) | — | 0 | sin fuente |
| 214 | IMSI Catchers (Stingrays) | — | 0 | sin fuente |
| 215 | Redes Mesh (Comunicación Descentralizada) | — | 0 | sin fuente |

### Transporte (18 ítems del lote)

| ítem | nombre | fuente | registros | veredicto |
|---:|---|---|---:|---|
| **617** | Carreteras Secundarias (Rutas Rurales) | SENAPRED `RedComunicaciones/9` Huellas y senderos | **5.241** | PARCIAL |
| **619** | Túneles de Carreteras | SENAPRED (58) + MOP Vialidad (34) + MOP Concesiones (25) | **117** (con solape) | SIRVE |
| **620** | Ferrocarriles (Vías) | SENAPRED Línea Férrea (933) + Metro Red (48) | **981** | PARCIAL (8 de 16 regiones) |
| **621** | Estaciones de Tren | SENAPRED `RedComunicaciones/6` | **132** (sólo Metro de Santiago) | PARCIAL |
| 623 | Terminales de Contenedores | — | 0 | sin fuente |
| **625** | Torres de Control Aéreo | MOP DAP (proxy: 7 aeropuertos / 17 red primaria) | **316** sitios | PARCIAL (proxy) |
| **626** | Terminales de Pasajeros (Aeropuertos) | MOP DAP + ILOG (mismo proxy, +11 militares) | **327** sitios | PARCIAL (proxy) |
| 629 | Trenes de Carga | — | 0 | sin fuente |
| 630 | Trenes de Pasajeros | — | 0 | sin fuente |
| 633 | Aviones Comerciales | — | 0 | sin fuente |
| 634 | Aviones de Carga | — | 0 | sin fuente |
| 640 | Centros de Logística y Distribución | — | 0 | sin fuente (ILOG lyr3 sin geometría) |
| **642** | Estaciones de Servicio (Combustible) | MOP CCOP Áreas de Servicio (64) + Áreas de Venta (40) | **104** | PARCIAL |
| 643 | Estaciones de Carga Eléctrica | — | 0 | sin fuente |
| 644 | Depósitos de Combustible (Puertos/Aeropuertos) | — | 0 | sin fuente |
| **650** | Rutas de Evacuación | SENAPRED `PlanesProteccion/4` Vías de evacuación | **2.740** | SIRVE |
| 657 | Puentes Ferroviarios | — | 0 | sin fuente |
| 658 | Almacenes de Vehículos (Depósitos) | — | 0 | sin fuente |

---

## 7 · Ítems del lote **sin ninguna fuente encontrada** (30)

**Transporte (11):** 623 Terminales de Contenedores · 629 Trenes de Carga ·
630 Trenes de Pasajeros · 633 Aviones Comerciales · 634 Aviones de Carga ·
640 Centros de Logística y Distribución · 643 Estaciones de Carga Eléctrica ·
644 Depósitos de Combustible (Puertos/Aeropuertos) · 657 Puentes Ferroviarios ·
658 Almacenes de Vehículos (Depósitos).
*(617 queda contado como parcial, no como vacío.)*

**Telecomunicaciones (19):** 179 Satélites GEO · 180 Satélites LEO ·
181 Satélites MEO · 182 Estaciones Terrestres de Satélites ·
187 Edge Computing · 188 Centros de Datos · 190 Nodos de Interconexión (IXP) ·
192 Hardware de Sistemas de Control (NOC) · 202 Teléfonos y Smartphones ·
203 Computadoras y Laptops · 204 IoT (Cámaras) · 205 IoT (Sensores) ·
206 IoT (Electrodomésticos) · 211 Radios de Emergencia ·
212 Redes Satelitales Portátiles · 214 IMSI Catchers · 215 Redes Mesh.

**Y ahí hay una lectura sobre la Matriz, no sobre la búsqueda.** De esos 30, unos
**16 no son infraestructura fija georreferenciable** (aviones, trenes, teléfonos,
laptops, electrodomésticos IoT, radios portátiles, satélites en órbita). No es
que falte el catastro: **es que el ítem no admite una coordenada**. Buscarles
fuente es buscar algo que por construcción no existe. Los que sí son
georreferenciables y de verdad faltan son pocos y concretos: **623 terminales de
contenedores, 640 centros logísticos, 643 electrolineras, 644 depósitos de
combustible, 657 puentes ferroviarios, 188 datacenters, 190 IXP y 182 estaciones
terrenas** — ocho huecos reales, no diecinueve.

---

## 8 · Ritmo y respeto al servicio

Pausa de **1,1 s** entre peticiones, lotes de **≤1.000** registros (el
`maxRecordCount` que declara cada servicio), `Accept-Encoding: gzip`, `User-Agent`
que identifica el proyecto y un correo de contacto, POST cuando la lista de
identificadores hace la URL larga, y reintentos con espera creciente (4→64 s).
Ningún servicio devolvió 429 ni cortó la conexión. Las respuestas crudas se
guardaron **antes** de convertirlas: `<clave>/lote_NNNN.json.gz` dentro de cada
carpeta de crudo.
