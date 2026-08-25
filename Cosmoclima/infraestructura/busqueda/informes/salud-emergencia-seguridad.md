# Catastros públicos georreferenciados · sectores **Salud**, **Servicios de Emergencia** y **Seguridad**

Fecha de la búsqueda: **2026-08-25**
Lote: 48 ítems de la MICR (21 Salud + 12 Servicios de Emergencia + 15 Seguridad) tomados de
`busqueda/CATALOGO_FALTANTES.txt`.
Todo lo que aparece abajo fue **verificado con petición real** (código HTTP, conteo de registros y conteo de
geometrías). Lo que no se pudo verificar está marcado explícitamente como **NO VERIFICADO**.

**Resultado global: 25 fuentes verificadas · 29.872 registros bajados · 29.427 con coordenada (98,5 %).**
Todas las fuentes traen coordenada en el 100 % de sus registros **salvo el CSV del DEIS**, que tiene 5.272 de 5.717.
Hay solapamiento deliberado entre fuentes (el DEIS aparece en tres versiones, bomberos y carabineros en dos):
el conteo de *activos distintos* es menor y depende de cuál se elija como primaria en cada ítem.

---

## El hallazgo que ordena todo el lote: el visor GRD de SENAPRED

`https://visor-grd.senapred.gob.cl/arcgis/rest/services` es un **ArcGIS Server público, sin token, con
`robots.txt` que no restringe nada**, y contiene el Sistema Integrado de Información para Emergencias (SIIE).
Su servicio `SIIE/InfraestructuraVulnerable` trae, en una sola pila y con el mismo sistema de referencia,
las capas de **salud, bomberos, carabineros, PDI, centros penitenciarios, servicio médico legal y las
direcciones regionales de ONEMI/SENAPRED**. Es decir: los tres sectores de este lote caen casi enteros dentro
de un solo servidor, y encima es el servidor del organismo con el que la Matriz tiene que hablar.

Su carpeta `levcap` (Levantamiento de Capacidades) agrega los **medios materiales** de la respuesta:
transporte, maquinaria, insumos y albergues/acopio. Eso es lo que permite poblar ítems que hasta ahora
parecían imposibles (Centros de Comando Móvil, Barcos de Rescate, Almacenes de Suministros).

Advertencia de método sobre `levcap`: son **capacidades declaradas** por cada institución, no un catastro
exhaustivo. 22 albergues/acopios en todo Chile es obviamente un piso, no el universo. Hay que declararlo
como tal (regla de diseño "confianza declarada").

---

# Fuentes verificadas

## 1 · MINSAL/DEIS — Establecimientos de Salud vigentes (CSV)

- **Ítems MICR que poblaría**
  - 266 Clínicas Rurales — Posta de Salud Rural 1.184 + Consultorio General Rural 13 + Servicio de Urgencia Rural 175
  - 267 Centros de Atención Primaria — CESFAM 607 + CECOSF 304 + CGU 12 + CEAP 6
  - 270 Laboratorios Clínicos — Laboratorio Clínico 441 + Sala Externa de Toma de Muestras 99
  - 273 Almacenes de Vacunas — Vacunatorio 144
  - 293 Unidades Móviles de Salud — Unidad de Procedimientos Móvil 56
  - 268 UCI / 269 Salas de Cirugía / 272 Farmacias Hospitalarias — sólo por *proxy*: Hospital 234, de los cuales 131 de Alta Complejidad
  - 310 Centros de Paramédicos (Serv. Emergencia) — SAPU 258 + SAR 103 + SUR 175
  - 308 Redes de Evacuación Médica — 1 Centro de Regulación Médica de las Urgencias (SAMU)
- Organismo · producto: **Ministerio de Salud, Departamento de Estadísticas e Información de Salud (DEIS)** · Establecimientos de Salud, corte 2026-08-18
- URL: `https://datos.gob.cl/dataset/3bf4cf7c-f638-4735-9a01-f65faae4beca/resource/2c44d782-3365-44e3-aefb-2c8b8363a1bc/download/establecimientos_20260818.csv`
- Formato · registros: CSV `;` UTF-8 · **5.717 filas · 5.272 con Latitud/Longitud** (445 sin coordenada)
- robots.txt: `datos.gob.cl/robots.txt` → **HTTP 200**. Prohíbe `/dataset/rate/`, `/revision/`, `/dataset/*/history` y **`/api/`**; `Crawl-Delay: 10`. La descarga usada cuelga de `/dataset/.../download/`, que **no** está prohibida. La consulta al catálogo sí se hizo por `/api/3/action/package_search`, que **sí** lo está: usar con moderación o navegar el HTML.
- Crudo: `busqueda/crudo/deis_establecimientos_salud/2026-08-25/establecimientos_20260818.csv`
- **Veredicto: SIRVE.** Es la fuente primaria de Salud del país, se actualiza semanalmente y trae la tipología completa (tipo, urgencia, complejidad, SAPU) que permite repartir un solo archivo entre ocho ítems distintos de la MICR.
- ⚠️ Filtrar por `EstadoFuncionamiento`: 414 filas están "Cerrado".

## 2 · SENAPRED SIIE — Establecimientos de Salud

- **Ítems MICR**: los mismos que la fuente 1 (respaldo geométrico).
- Organismo · producto: **SENAPRED · visor GRD / SIIE**, capa `Establecimientos de Salud`
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/InfraestructuraVulnerable/MapServer/2`
- Formato · registros: ArcGIS REST → JSON de puntos, `outSR=4326` · **5.159 declarados · 5.159 recibidos · 5.159 con geometría**
- robots.txt: `visor-grd.senapred.gob.cl/robots.txt` → **HTTP 200**, `User-agent: *` + `Disallow:` vacío = **sin restricciones**
- Crudo: `busqueda/crudo/senapred_salud/2026-08-25/establecimientos_salud.json`
- **Veredicto: PARCIAL.** Es una copia del DEIS con el mismo esquema pero más antigua y con 558 registros menos; sirve como respaldo y como puente hacia el resto de las capas del SIIE, no como fuente primaria.

## 3 · SENAPRED SIIE — Centros Penitenciarios

- **Ítems MICR**: **356 Prisiones de Alta Seguridad**
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/InfraestructuraVulnerable/MapServer/6`
- Formato · registros: ArcGIS REST → JSON de puntos · **82 · 82 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/senapred_centros_penitenciarios/2026-08-25/centros_penitenciarios.json`
- **Veredicto: SIRVE (con reserva).** Es la **única** fuente pública georreferenciada de recintos penales que apareció en toda la búsqueda — Gendarmería no publica capa ni listado con coordenadas —, pero no marca cuáles son "de alta seguridad": hay que derivarlo del prefijo del nombre (CP/CCP/CDP/CPF) o cruzarlo a mano con el listado de Gendarmería.
- Trae además población penal por sexo (`h`, `m`, `total`), que es exposición humana directa.

## 4 · SENAPRED SIIE — PDI

- **Ítems MICR**: complementa 365 Centros de Comando y Control y es el único catastro de cuarteles de la Policía de Investigaciones.
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/InfraestructuraVulnerable/MapServer/15`
- Formato · registros: ArcGIS REST → JSON de puntos · **175 · 175 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/senapred_pdi/2026-08-25/pdi.json`
- **Veredicto: SIRVE.** Cuarteles y complejos con `unidad`, `inmueble`, dirección, comuna y región; incluye unidades especializadas (LACRIM, BRISEXME).

## 5 · SENAPRED SIIE — Direcciones Regionales ONEMI/SENAPRED

- **Ítems MICR**: **312 Centros de Coordinación** · **365 Centros de Comando y Control**
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/InfraestructuraVulnerable/MapServer/13`
- Formato · registros: ArcGIS REST → JSON de puntos · **16 · 16 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/senapred_direcciones_regionales/2026-08-25/direcciones_regionales_onemi.json`
- **Veredicto: SIRVE.** Son las 16 sedes regionales donde se instala el COGRID; es exactamente lo que la MICR llama Centro de Coordinación, con dirección y teléfonos.

## 6 · SENAPRED SIIE — Servicio Médico Legal

- **Ítems MICR**: 270 Laboratorios Clínicos (complemento forense) · 306 Centros de Investigación Médica (parcial)
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/InfraestructuraVulnerable/MapServer/17`
- Formato · registros: ArcGIS REST → JSON de puntos · **47 · 47 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/senapred_servicio_medico_legal/2026-08-25/servicio_medico_legal.json`
- **Veredicto: PARCIAL.** Vale por sus banderas SÍ/NO de capacidad instalada (tanatología, sala de autopsia, laboratorio, genética forense, clínica, salud mental), que es justo el tipo de dato de fragilidad que la MICR asigna hoy a criterio.
- ⚠️ Los campos `latitud`/`longitud` vienen como texto con **coma decimal**; usar la geometría del shape, no esas columnas.

## 7 · SENAPRED SIIE — Farmacias

- **Ítems MICR**: 281 Cadena de Suministro de Medicinas · 282 Almacenes de Medicinas (Distribución) — ambos como el eslabón minorista
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/Farmacias/MapServer/0`
- Formato · registros: ArcGIS REST → JSON de puntos · **4.503 · 4.503 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/senapred_farmacias/2026-08-25/farmacias.json`
- **Veredicto: PARCIAL.** Son farmacias **comunitarias**, no hospitalarias (ítem 272), y no son bodegas de distribución; sirven como capa de acceso a medicamentos, no como cadena de suministro.

### 7-bis · MINSAL Farmanet (fuente viva, NO se guardó)

- URL: `https://midas.minsal.cl/farmacia_v2/WS/getLocales.php` (ficha: `https://datos.gob.cl/dataset/farmacias-en-chile`)
- Verificado el 2026-08-25: **HTTP 200, JSON, 2.012 locales, 1.837 con coordenada**, actualización diaria.
- robots.txt: `midas.minsal.cl/robots.txt` → **HTTP 200**. `User-agent: *` → `Allow: /`, **pero declara `Disallow: /` para ClaudeBot, GPTBot, CCBot, Amazonbot, Google-Extended, Bytespider y meta-externalagent.**
- **Veredicto: SIRVE pero NO SE BAJÓ.** El operador pidió explícitamente que los agentes de IA no recolecten su contenido. El acceso quedó verificado; si se quiere el dato, que lo baje una persona. La capa de SENAPRED (fuente 7) cubre el mismo fenómeno sin ese problema.

## 8 · SENAPRED levcap — Medios de Transporte

- **Ítems MICR**
  - **337 Centros de Comando Móvil** — 35 "Puesto de mando/comunicaciones móvil (COE Móvil)"
  - **346 Barcos de Rescate** — 27 Embarcación Menor + 11 Embarcación Mayor
  - 293 Unidades Móviles de Salud / 308 Redes de Evacuación Médica — 16 Ambulancias + 16 aeronaves
  - complementa 310 (56 Carros de Rescate) y Seguridad (175 vehículos policiales de patrullaje)
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/levcap/medios_de_transporte/MapServer/0`
- Formato · registros: ArcGIS REST → JSON de puntos · **1.067 · 1.067 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/senapred_levcap_transporte/2026-08-25/medios_de_transporte.json`
- **Veredicto: SIRVE.** Es la única fuente encontrada que georreferencia medios móviles de respuesta, y resuelve dos ítems (337 y 346) que no tenían ninguna otra opción; pero es capacidad **declarada**, así que el conteo es un piso.

## 9 · SENAPRED levcap — Elementos e Insumos de Emergencia

- **Ítems MICR**: **330 Almacenes de Suministros** · **349 Almacenes de Medicamentos** (7 puntos de "equipos/insumos para atención médica, incl. fármacos y vacunas") · 296 Botiquines de Emergencia (parcial)
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/levcap/elementos_e_insumos_de_emergencia/MapServer/0`
- Formato · registros: ArcGIS REST → JSON de puntos · **189 · 189 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/senapred_levcap_insumos_emergencia/2026-08-25/elementos_e_insumos_de_emergencia.json`
- **Veredicto: PARCIAL.** Cubre los ítems, pero con 189 puntos para todo Chile es un inventario declarado, no un catastro.

## 10 · SENAPRED levcap — Albergues y Centros de Acopio

- **Ítems MICR**: **324 Centros de Logística** (8 acopio + 7 acopio y distribución + 2 distribución) · 332 Infraestructura de Contingencia (5 albergues)
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/levcap/albergues_y_centros_de_acopios/MapServer/0`
- Formato · registros: ArcGIS REST → JSON de puntos · **22 · 22 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/senapred_levcap_albergues_acopio/2026-08-25/albergues_y_centros_de_acopio.json`
- **Veredicto: PARCIAL.** 22 puntos en todo el país; sirve para abrir el ítem, no para cerrarlo.

## 11 · SENAPRED levcap — Maquinaria y Equipamiento Mayor

- **Ítems MICR**: 324 Centros de Logística · 332 Infraestructura de Contingencia (equipamiento de respuesta)
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/levcap/maquinaria_y_equipamiento_mayor/MapServer/0`
- Formato · registros: ArcGIS REST → JSON de puntos · **619 · 619 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/senapred_levcap_maquinaria/2026-08-25/maquinaria_y_equipamiento_mayor.json`
- **Veredicto: PARCIAL.** Es capacidad de respuesta georreferenciada (136 camiones aljibe, 8 equipos de búsqueda y rescate, 7 puentes mecano), no infraestructura fija; encaja mejor como atributo de resiliencia que como activo de la MICR.

## 12 · IDE Chile — Compañías de Bomberos

- **Ítems MICR**: base física de **310 Centros de Paramédicos** y de toda la respuesta de emergencia; complementa 331 y 337
- Organismo · producto: **IDE Chile / SNIT** (dato de la Junta Nacional de Cuerpos de Bomberos) · Compañías de Bomberos de Chile, capa 20231110
- URL: `https://www.geoportal.cl/geoportal/catalog/download/e476987d-7422-3fa4-b02a-0c67ce3b3ac8`
- Formato · registros: Shapefile ZIP (`.prj` = GEOGCS WGS 84) · **1.095 · 1.095 con geometría**
- robots.txt: `geoportal.cl/robots.txt` → **HTTP 200**, `User-agent: *` + `Disallow:` vacío = **sin restricciones**
- Crudo: `busqueda/crudo/bomberos_companias/2026-08-25/companias_de_bomberos_shp.zip`
- **Veredicto: SIRVE.** Cobertura nacional completa, coordenadas en el 100 % de los registros y con el CUT de compañía y de cuerpo, que permite agregarlo por comuna sin geocodificar nada.

## 13 · IDE Chile — Cuerpos de Bomberos

- **Ítems MICR**: igual que la fuente 12, a nivel de la unidad administrativa (el Cuerpo agrupa Compañías)
- URL: `https://www.geoportal.cl/geoportal/catalog/download/4a41e805-278f-3be3-99bc-17b6d7adf086`
- Formato · registros: Shapefile ZIP, WGS 84 · **309 · 309 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/bomberos_cuerpos/2026-08-25/cuerpos_de_bomberos_shp.zip`
- **Veredicto: SIRVE.** Da el nivel administrativo que falta para modelar el nivel Comunal/Provincial exigido por el marco normativo del proyecto.

## 14 · SENAPRED SIIE — Bomberos (compañías + cuerpos unidos)

- **Ítems MICR**: los mismos que 12 y 13, en una sola capa
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/InfraestructuraVulnerable/MapServer/4`
- Formato · registros: ArcGIS REST → JSON de puntos · **1.412 · 1.412 con geometría** (1.100 Compañía + 310 Cuerpo + 2 sin clasificar)
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/senapred_bomberos/2026-08-25/bomberos.json`
- **Veredicto: SIRVE.** Es la unión de las dos capas de IDE Chile (1.095 + 309 = 1.404 vs 1.412 aquí) con el campo `tipo` para separarlas; conviene como fuente única si se quiere un solo adaptador.

## 15 · IDE Chile — Cuarteles de Carabineros

- **Ítems MICR**: base de Seguridad; el desglose por `TIPO_DE_UN` permite separar comisarías de retenes
- Organismo · producto: **Carabineros de Chile**, publicado vía IDE Chile / Ministerio del Interior · capa 20220309
- URL: `https://www.geoportal.cl/geoportal/catalog/download/24cae8aa-4e9c-36a9-9e92-d1dca97e3a0c`
- Formato · registros: Shapefile ZIP, WGS 84 · **874 · 874 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/carabineros_cuarteles/2026-08-25/cuarteles_carabineros_shp.zip`
- Desglose: RETEN 370 · COMISARIA 193 · TENENCIA 170 · SUBCOMISARIA 93 · TENENCIA CARRETERAS 26 · AVANZADA 17 · SUBCOMISARIA CARRETERAS 3 · PUESTO DE CONTROL 1 · GARITA COMUNITARIA 1
- **Veredicto: SIRVE.** Cobertura nacional, 100 % con coordenada.
- ⚠️ La ficha de datos.gob.cl (dataset `16123`) apunta a `https://www.geoportal.cl/ArcGIS/rest/services/MinisteriodeInterior/chile_minterior_carabineros_cuarteles/MapServer`, que **ya no existe (404)**: geoportal.cl migró de ArcGIS Server a GeoServer y ese endpoint quedó muerto. El ZIP sí funciona.
- ⚠️ **No** incluye escuelas de formación, así que no alcanza para el ítem 380.

## 16 · SENAPRED SIIE — Carabineros

- **Ítems MICR**: igual que 15
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/InfraestructuraVulnerable/MapServer/5`
- Formato · registros: ArcGIS REST → JSON de puntos · **885 · 885 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/senapred_carabineros/2026-08-25/carabineros.json`
- **Veredicto: SIRVE.** Mismo esquema que la capa de IDE Chile y 11 registros más; es la versión que SENAPRED usa operativamente.

## 17 · CONAF (vía SIIE) — Recursos de Detección

- **Ítems MICR**: **364 Torres de Vigilancia**
- Organismo · producto: **CONAF**, publicado en el visor SIIE de SENAPRED · Recursos de Detección temporada 2022-2023
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/conaf/MapServer/2`
- Formato · registros: ArcGIS REST → JSON de puntos · **78 · 78 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/conaf_recursos_deteccion/2026-08-25/recursos_deteccion_conaf.json`
- **Veredicto: PARCIAL.** Son torres y puestos de observación de incendios forestales con material y condición fija/móvil — encajan con "Torres de Vigilancia" si ese ítem se lee como vigilancia territorial, no como torre militar; además la temporada 2022-2023 ya no está vigente.

## 18 · SernamEG (vía IDE Chile) — Centros de la Mujer

- **Ítems MICR**: **348 Centros de Atención a Víctimas**
- URL: `https://www.geoportal.cl/geoportal/catalog/download/16e75941-dfb6-3477-9a21-2d0b6f1a587d`
- Formato · registros: Shapefile ZIP, WGS 84 · **45 · 45 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/sernameg_centros_mujer/2026-08-25/centros_sernam_shp.zip`
- **Veredicto: PARCIAL.** Es el único catastro georreferenciado de atención a víctimas que apareció, pero cubre sólo una parte del país y sólo una clase de víctima (violencia de género); trae `COBERTURA` con las comunas que atiende cada centro, que sirve para el nivel Comunal.

## 19 · SENAPRED SIIE — Recintos Deportivos (IND)

- **Ítems MICR**: 332 Infraestructura de Contingencia (candidatos a albergue)
- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/InfraestructuraVulnerable/MapServer/22`
- Formato · registros: ArcGIS REST → JSON de puntos · **538 · 538 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/senapred_recintos_deportivos/2026-08-25/recintos_deportivos.json`
- **Veredicto: PARCIAL.** Trae m², dirección y tipo de administración, y en Chile el recinto deportivo es el albergue típico, pero **la capa no declara aptitud de albergue**: es una lista de candidatos, no un catastro de albergues. No usarla como si lo fuera.

## 20 · SENAPRED SIIE — "Centros Públicos" *(fuera de este lote, se deja anotado)*

- URL: `https://visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/InfraestructuraVulnerable/MapServer/8`
- Formato · registros: ArcGIS REST → JSON de puntos · **1.194 · 1.194 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/senapred_centros_publicos/2026-08-25/centros_publicos.json`
- **Veredicto: NO SIRVE para este lote.** Pese al nombre, el contenido son **supermercados** (Unimarc y similares). Se deja bajado y anotado porque le sirve al sector **Alimentario** (ítems 413 Supermercados y Tiendas, 414 Mercados Locales) y al agente que lo cubra.

## 21 · IDE Chile — Establecimientos de salud de Chile, Diciembre 2025 (shapefile)

- **Ítems MICR**: los mismos ocho que la fuente 1 (266, 267, 270, 273, 293, 268/269/272 por *proxy*, 310, 308)
- Organismo · producto: **IDE Chile / SNIT**, con dato del DEIS-MINSAL · corte diciembre 2025
- URL: `https://geoportal.cl/geoportal/catalog/download/5b2f29d1-94d4-398f-a207-7f9f3056e5d1`
  (ficha: `https://geoportal.cl/geoportal/catalog/36779/Establecimientos%20de%20salud%20de%20Chile%20Diciembre%202025`)
- Formato · registros: Shapefile ZIP, WGS 84 · **5.181 · 5.181 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/ide_establecimientos_salud_2025/2026-08-25/establecimientos_salud_dic2025_shp.zip`
- **Veredicto: SIRVE — y probablemente es la mejor versión de las tres.** Es el DEIS ya geocodificado y con los
  **CUT de comuna, provincia y región resueltos** (`CUT_COMUNA`, `CUT_PROVIN`, `CUT_REGION`), que es exactamente
  lo que hace falta para amarrar cada activo a los cuatro niveles administrativos que el proyecto está obligado a
  modelar. Y no tiene el agujero del CSV: 5.181 de 5.181 con coordenada, contra 5.272 de 5.717.
- Recomendación práctica: **este** para el amarre territorial, **el CSV del DEIS** para lo más reciente.
- También expuesto como WFS en `https://geoportal.cl/geoserver/EstablecimientosdesaluddeChile2025/wfs` (no probado).

## 22 · MINSAL (vía IDE Chile) — Direcciones de los Servicios de Salud

- **Ítems MICR**: **312 Centros de Coordinación** (rama sanitaria) · 365 Centros de Comando y Control
- URL: `https://geoportal.cl/geoportal/catalog/download/70d5fd11-016d-3934-8587-deb32edb9c6e`
- Formato · registros: Shapefile ZIP, WGS 84 · **29 · 29 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/minsal_direcciones_servicios_salud/2026-08-25/direcciones_servicios_salud_shp.zip`
- **Veredicto: SIRVE.** Son las 29 sedes de los Servicios de Salud, los nodos que coordinan la red asistencial en
  emergencia; complementan las 16 direcciones regionales de SENAPRED con la mitad sanitaria de la coordinación.

## 23 · MINSAL (vía IDE Chile) — SEREMI de Salud

- **Ítems MICR**: 312 Centros de Coordinación · 365 Centros de Comando y Control
- URL: `https://geoportal.cl/geoportal/catalog/download/060e547a-d3f0-38e5-9e60-e57eedb305c5`
- Formato · registros: Shapefile ZIP, WGS 84 · **15 · 15 con geometría**
- robots.txt: **HTTP 200, sin restricciones**
- Crudo: `busqueda/crudo/minsal_seremis_salud/2026-08-25/seremis_salud_shp.zip`
- **Veredicto: PARCIAL.** Autoridad sanitaria regional (nodo de decisión, no de atención), y faltaría una para
  cubrir las 16 regiones.

## 24 · OpenStreetMap — instalaciones militares en Chile *(fuente NO oficial)*

- **Ítems MICR**
  - **354 Bases Militares (FFAA)** — 221 `landuse=military` + 7 `military=base` + 8 `military=airfield`
  - **381 Centros de Entrenamiento Militar** — 14 `military=training_area`
  - **395 Campos de Simulación (Entrenamiento)** — 6 `military=range` + 13 `military=danger_area`
  - 364 Torres de Vigilancia (débil) — 26 `military=checkpoint` + 36 `military=bunker`
- URL: `https://overpass-api.de/api/interpreter` (POST; `landuse=military` y `military=*` sobre el bbox de Chile)
- Formato · registros: JSON de Overpass · **825 devueltos por el bbox → 464 dentro del polígono de Chile, 464 con coordenada**
- robots.txt: `overpass-api.de/robots.txt` → **HTTP 200**, declara `Disallow: /api/` y `Disallow: /munin/`. Ese
  `Disallow` apunta a rastreadores de buscadores (evita indexar resultados generados al vuelo); la Overpass API está
  hecha para consulta programática y su *usage policy* la permite. **No automatizar de forma recurrente sin revisarla.**
- Crudo: `busqueda/crudo/osm_militar_chile/2026-08-25/osm_militar_chile.json`
- **Veredicto: PARCIAL, confianza declarada BAJA.** Es la **única** opción encontrada para los ítems 354, 381 y 395,
  porque ningún organismo chileno publica catastro de instalaciones militares (y es razonable que no lo haga).
  Los 78 `military=trench` son casi seguro trincheras históricas, no infraestructura activa: hay que revisarlos antes de usar.
- El recorte a Chile se hizo con el polígono del país de Nominatim y `shapely` (el bbox metía Argentina: 361 elementos descartados).

## 25 · OpenStreetMap — recintos penitenciarios en Chile *(fuente NO oficial)*

- **Ítems MICR**: **356 Prisiones de Alta Seguridad** (control cruzado de la fuente 3)
- URL: `https://overpass-api.de/api/interpreter` (POST; `amenity=prison` sobre el bbox de Chile)
- Formato · registros: JSON de Overpass · **204 devueltos → 157 dentro del polígono de Chile, 157 con coordenada**
- robots.txt: igual que la fuente 24
- Crudo: `busqueda/crudo/osm_prisiones_chile/2026-08-25/osm_prisiones_chile.json`
- **Veredicto: PARCIAL, confianza declarada BAJA-MEDIA.** Casi **duplica** los 82 puntos de SENAPRED, lo que sugiere
  que la capa oficial trae sólo los recintos cerrados y OSM además incluye CET, CRS, CIP-CRC y recintos "ex".
  Usar como control cruzado del catastro de SENAPRED, nunca como fuente única. La discrepancia 82 vs 157 es en sí
  misma un dato: el catastro oficial de recintos penales está incompleto.

---

# Fuentes descartadas o no verificables

| Fuente | Qué pasó | Veredicto |
|---|---|---|
| `www.geoportal.cl/geoportal/catalog/download/d0381538-...` — "Establecimientos de salud" de IDE Chile | **HTTP 404**. La ficha de datos.gob.cl la lista, pero el archivo ya no está. | NO SIRVE (enlace roto) |
| `www.geoportal.cl/ArcGIS/rest/services` — servidor ArcGIS histórico de IDE Chile | Redirige a `geoportal.cl` y devuelve **404**. Migraron a GeoServer, que exige workspace y sólo expone `Hidrografia` y `Capitales`. | NO SIRVE |
| `miratuterritorio.cl` — Visor de Emergencia Meteorológica de SNIT/IDE Chile (albergues, acopio, hospitales) | **DNS no resuelve** (`HTTP 000` en `https://`, `https://www.` y `http://www.`) el 2026-08-25. Sería la fuente natural de albergues activos. | **NO VERIFICADO** — revisar más adelante |
| SECTRA — `CENTRO_DE_SALUD` (206), `CUARTEL_DE_BOMBERO` (66), `CUARTEL_POLICIAL` (99) en `services6.arcgis.com/feQ9HId8vmgonvvD` | Verificado, HTTP 200, pero cubre sólo áreas de estudio de transporte, no el país. Superado por IDE Chile y SENAPRED. | NO SIRVE (subnacional y redundante) |
| ArcGIS Online, org `services2.arcgis.com/1GsyJOWdIs6zblbt` (`1_ESTABLECIMIENTOS_REFES_2026`, owner `nalzueta_minsal`) | Aparece al buscar "hospitales Chile" pero es el **MINSAL de Argentina** (registro REFES). Falso positivo. | NO SIRVE |
| Gendarmería de Chile — `gendarmeria.gob.cl/establecimientos.html` | HTTP 200 pero la página es sólo un **glosario** de tipos de recinto: sin listado, sin direcciones, sin coordenadas. | NO SIRVE |
| MINSAL — "Lugares y horarios de donación de sangre" (PDF) | Existe un PDF con 4 Centros de Sangre y 51 puntos fijos, pero **sin coordenadas**; habría que geocodificar. No se descargó. | **NO VERIFICADO** como capa geográfica |
| Catálogo completo de IDE Chile | Se recorrieron las 166 páginas del catálogo (pausa 0,8 s, ~15 páginas se perdieron por timeout del servidor): **737 fichas distintas capturadas**. De ahí salieron las fuentes 21, 22 y 23, que no aparecían por ninguna otra vía. Quedan ~900 fichas sin ver. | Barrido **parcial** — vale la pena repetirlo |
| `www.geoportal.cl/geoportal/catalog?action=search&text=...` | El buscador del geoportal es Livewire: el HTML inicial devuelve el listado **por defecto**, ignorando el texto buscado. Filtrar por `text=` desde `curl` **no funciona**; hay que paginar el catálogo completo. | Anotado como trampa |

---

# Tabla resumen · ítem MICR → fuente → registros

| Ítem | Nombre | Fuente | Registros aportados |
|---|---|---|---|
| 266 | Clínicas Rurales | DEIS (PSR + CGR + SUR) | 1.372 |
| 267 | Centros de Atención Primaria | DEIS (CESFAM + CECOSF + CGU + CEAP) | 929 |
| 268 | Unidades de Cuidados Intensivos (UCI) | DEIS — *proxy* hospitales Alta Complejidad | 131 |
| 269 | Salas de Cirugía | DEIS — *proxy* hospitales | 234 |
| 270 | Laboratorios Clínicos | DEIS (Lab. Clínico + SETM) + SML SENAPRED | 540 + 47 |
| 272 | Farmacias Hospitalarias | DEIS — *proxy* hospitales | 234 |
| 273 | Almacenes de Vacunas | DEIS (Vacunatorio) | 144 |
| 281 | Cadena de Suministro de Medicinas | SENAPRED Farmacias — *proxy* minorista | 4.503 |
| 282 | Almacenes de Medicinas (Distribución) | SENAPRED Farmacias — *proxy* minorista | 4.503 |
| 293 | Unidades Móviles de Salud | DEIS (Unidad de Procedimientos Móvil) + levcap Ambulancias | 56 + 16 |
| 296 | Botiquines de Emergencia | levcap insumos (atención médica) | 7 |
| 306 | Centros de Investigación Médica | SENAPRED Servicio Médico Legal — parcial | 47 |
| 308 | Redes de Evacuación Médica | DEIS (SAMU) + levcap (ambulancias + aeronaves) | 1 + 32 |
| 310 | Centros de Paramédicos | DEIS (SAPU + SAR + SUR) + levcap Carros de Rescate | 536 + 56 |
| 312 | Centros de Coordinación | SENAPRED Direcciones Regionales (16) + Direcciones Servicios de Salud (29) + SEREMI Salud (15) | 60 |
| 324 | Centros de Logística | levcap albergues y centros de acopio | 17 |
| 330 | Almacenes de Suministros | levcap elementos e insumos de emergencia | 189 |
| 332 | Infraestructura de Contingencia | levcap albergues (5) + Recintos Deportivos IND (538) | 543 |
| 337 | Centros de Comando Móvil | levcap medios de transporte (puesto de mando móvil) | 35 |
| 346 | Barcos de Rescate | levcap medios de transporte (embarcación mayor + menor) | 38 |
| 348 | Centros de Atención a Víctimas | SernamEG Centros de la Mujer | 45 |
| 349 | Almacenes de Medicamentos | levcap insumos (fármacos y vacunas) | 7 |
| 354 | Bases Militares (FFAA) | OpenStreetMap (landuse=military + base + airfield) — *no oficial* | 236 |
| 356 | Prisiones de Alta Seguridad | SENAPRED Centros Penitenciarios (82) + OSM (157, control cruzado) | 82 |
| 381 | Centros de Entrenamiento Militar | OpenStreetMap (military=training_area) — *no oficial* | 14 |
| 395 | Campos de Simulación (Entrenamiento) | OpenStreetMap (range + danger_area) — *no oficial* | 19 |
| 364 | Torres de Vigilancia | CONAF Recursos de Detección (78) + OSM checkpoint/bunker (62, débil) | 78 |
| 365 | Centros de Comando y Control | SENAPRED Direcciones Regionales + PDI + Direcciones Servicios de Salud | 16 + 175 + 29 |
| — | *base transversal de emergencia* | Bomberos (IDE Chile 1.095 + 309 · SENAPRED 1.412) | 1.412 |
| — | *base transversal de seguridad* | Carabineros (IDE Chile 874 · SENAPRED 885) | 885 |

**29 de los 48 ítems del lote quedan con al menos una fuente georreferenciada verificada**
(13 de 21 en Salud, 8 de 12 en Servicios de Emergencia, 6 de 15 en Seguridad).
De esos 29, **25 se apoyan en fuente oficial chilena** y 4 (354, 381, 395 y el refuerzo de 356) dependen de
OpenStreetMap, con confianza declarada baja.

---

# Ítems del lote SIN ninguna fuente encontrada

## Salud (8 de 21)
- **271 Bancos de Sangre** — sólo un PDF del MINSAL con 4 Centros de Sangre y 51 puntos fijos, sin coordenadas. Derivable geocodificando o marcando los hospitales de Alta Complejidad del DEIS.
- **275 Ventiladores y Respiradores** — equipamiento interno; ningún registro público lo georreferencia.
- **276 Monitores de Signos Vitales** — ídem.
- **283 Suministro de Oxígeno (Tanques)** — ídem; podría venir de la SEC o de permisos de sustancias peligrosas, no verificado.
- **297 Hospitales de Campaña** — no hay catastro; son móviles y se despliegan por evento.
- **299 Incubadoras (Neonatología)** — equipamiento interno.
- **300 Desfibriladores** — la Ley 21.156 obliga a instalar DEA en recintos de alta afluencia, pero **no se encontró registro público consultable**; sería una fuente muy valiosa si existe.
- **307 Cadena de Suministro de Equipos Médicos** — sin fuente (CENABAST no publica bodegas georreferenciadas).

## Servicios de Emergencia (4 de 12)
- **311 Hospitales de Campaña** — igual que 297.
- **326 Infraestructura de Conectividad** — corresponde al sector Telecomunicaciones; fuera de este lote.
- **331 Centros de Capacitación (Emergencia)** — la Academia Nacional de Bomberos y la academia de SENAPRED no publican capa; no verificado.
- **312/324/330/332** quedan cubiertos pero **sólo con capacidad declarada**: tratarlos como piso, no como universo.

## Seguridad (9 de 15) — sigue siendo el sector más pobre del lote
- **361 Aviones Militares** — sin fuente pública.
- **366 Radios de Comunicación** — sin fuente (parcialmente en el registro de antenas de SUBTEL, que es del sector Telecomunicaciones).
- **371 Armamento (Arsenal)** — sin fuente pública, y es razonable que así sea.
- **372 Municiones (Almacenamiento)** — ídem.
- **374 Satélites Militares** — sin fuente pública.
- **375 Estaciones Terrestres (Satélites)** — sin fuente pública.
- **377 Infraestructura de Conectividad** — sector Telecomunicaciones.
- **380 Centros de Entrenamiento Policial** — la capa de Carabineros **no** incluye escuelas de formación; habría que agregarlas a mano (Escuela de Carabineros, Escuela de Suboficiales, Escuela de Investigaciones Policiales). Son pocas y conocidas: es el ítem más barato de cerrar del sector.
- **396 Redes de Informantes** — no es un activo físico georreferenciable; probablemente el ítem debería salir del catálogo de "físicos".

**Conclusión sobre Seguridad:** con OpenStreetMap se rescatan 354, 381 y 395 a confianza baja, pero 9 de sus 15
ítems son de defensa nacional y **no van a tener catastro público oficial, por diseño**. Vale la pena decidir a nivel de la MICR si esos ítems se dejan permanentemente vacíos, si se poblan con dato agregado por región, o si se marcan como "no observables" para que no arrastren el índice hacia abajo por ausencia de dato.
