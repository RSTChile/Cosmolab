# Catastros públicos georreferenciados · sectores GOBIERNO e INDUSTRIA DE DEFENSA

Fecha de barrido: **2026-08-25**
Lote: los 13 ítems del sector **Gobierno** y los 16 ítems del sector **Industria de Defensa**
del archivo `busqueda/CATALOGO_FALTANTES.txt`.

Todo lo que aparece abajo fue verificado con petición HTTP real (código de respuesta, tamaño,
conteo de registros). Lo que no se pudo verificar está marcado explícitamente como
**NO VERIFICADO**. No se geocodificó ninguna dirección: donde dice "0 con coordenada" es
porque la fuente publica sólo el domicilio postal.

**Criterio aplicado en Defensa** (no negociable, ver punto 2 del encargo): sólo se recogió lo que
la propia entidad estatal publica abiertamente en su sitio institucional. No se dedujo, infirió ni
cruzó ninguna fuente para localizar instalaciones militares no publicadas. Los ítems que sólo se
podrían poblar con información no publicada están reportados como
**NO CORRESPONDE — no publicado oficialmente**.

---

## 1. ChileAtiende — sucursales de la red de atención del Estado

- **Ítems MICR que poblaría**: 614 (Oficinas de Atención Ciudadana); apoyo parcial a 586.
- **Organismo**: ChileAtiende (SEGPRES / Instituto de Previsión Social)
- **Producto**: Sucursales de la red de atención presencial
- **URL exacta**: `https://www.chileatiende.gob.cl/ayuda/sucursales`
  (JSON embebido en el atributo `:data` del HTML; la API `…/api/sucursales` devuelve **403 Unauthorized**)
- **Formato**: HTML con JSON embebido → JSON + CSV
- **Registros**: **375** · **con coordenada: 322** (`lat`/`lng` en el propio registro; las otras 53 traen dirección y la coordenada de su comuna)
- **robots.txt**: HTTP 200 — `User-agent: * / Disallow: /buscar? / Disallow: /buscar`. La ruta usada **no** está prohibida.
- **Crudo**: `busqueda/crudo/chileatiende_sucursales/2026-08-25/`
- **Veredicto**: **SIRVE**. Es la mejor fuente del lote: 375 puntos de atención presencial del Estado, 86 % ya georreferenciados, actualizados (campo `updated_at` hasta 2026-08-20).
- **Salvedad honesta**: los 375 registros traen `institution_id = "AL005"` y `type = "personas"`, y varios domicilios coinciden con oficinas del Registro Civil (Teatinos 601, Alameda 1353). El sitio los publica como "sucursales ChileAtiende". **No verifiqué** a qué institución corresponde el código AL005.

## 2. Ministerio de Bienes Nacionales — Propiedad Fiscal Administrada

- **Ítems MICR que poblaría**: se buscó para 574 / 592 / 611. **No sirve para ninguno.**
- **Organismo**: Ministerio de Bienes Nacionales — IDE MBN
- **Producto**: Propiedad Fiscal Administrada, abril 2026
- **URL exacta**: `https://idembn.bienes.cl/geoserver/ppff_admin/ows?service=WFS&version=2.0.0&request=GetFeature&typeNames=ppff_admin:propiedad_fiscal_administrada__abril_2026&outputFormat=application/json&srsName=EPSG:4326&count=1000&startIndex=N`
  (ficha: `https://geoportal.cl/geoportal/catalog/41920/Propiedad%20Fiscal%20Administrada`)
- **Formato**: GeoJSON (polígonos), WFS 2.0 sin token
- **Registros**: **9.074** (`resultType=hits`) · **con coordenada: 9.074 (100 %)**
- **robots.txt**: HTTP 200 — `User-agent: * / Disallow:` (todo permitido)
- **Crudo**: `busqueda/crudo/mbn_propiedad_fiscal/2026-08-25/` (10 archivos, 9,7 MB)
- **Veredicto**: **NO SIRVE para Gobierno** — pero se dejó bajado porque es un catastro nacional válido para otros usos.
  Motivo: los atributos son `nombre, id_institu, rol_sii, sup_m2, sup_ha, urb_rur`. **No hay campo de destino ni de servicio.** `nombre` es en la práctica la dirección del predio ("SITIO 4 COMUNA ALTO HOSPICIO"), y `id_institu` tiene **9.074 valores distintos en 9.074 filas**, o sea no es un código de institución. Filtrando por texto sólo aparecen 10 predios con palabra "gobernación/intendencia/Moneda", 0 con "archivo" y 0 con "embajada". No permite decir qué predio fiscal es un edificio de gobierno.

## 3. SENAPRED — Direcciones Regionales

- **Ítems MICR que poblaría**: 595 (Centros de Comando · Emergencias)
- **Organismo**: SENAPRED · **URL**: `https://www.senapred.gob.cl/regiones/`
- **Formato**: HTML · **Registros**: **16** · **con coordenada: 0** (dirección postal, director, teléfono y correo)
- **robots.txt**: **NO VERIFICADO** (`/direcciones-regionales/` redirige a la home)
- **Crudo**: `busqueda/crudo/senapred_direcciones_regionales/2026-08-25/`
- **Veredicto**: **PARCIAL**. Cubre las 16 direcciones regionales completas pero sin coordenada; requiere geocodificación posterior. No cubre los COGRID comunales ni provinciales.

## 4. MINREL / consulado.gob.cl — Consulados extranjeros en Chile

- **Ítems MICR que poblaría**: 597 (Consulados)
- **Organismo**: MINREL — Dirección General de Asuntos Consulares
- **URL exacta**: índice `https://www.consulado.gob.cl/red/consulados-extranjeros-en-chile`; fichas `https://www.consulado.gob.cl/consulados/consulados-en-chile/<pais>/<ciudad>/<slug>`
- **Formato**: HTML (scrape con pausa de 1 s) → JSON
- **Registros**: **53** · **con coordenada: 51** · con dirección postal: 52
- **robots.txt**: **NO VERIFICADO** en `consulado.gob.cl`
- **Crudo**: `busqueda/crudo/minrel_consulados/2026-08-25/`
- **Veredicto**: **PARCIAL/SIRVE**. La coordenada sale del `iframe` de Google Maps que publica cada ficha (parámetros `!2d`=lon `!3d`=lat); en algunos casos ese mapa está centrado en el barrio y no en el inmueble, así que hay que tratarla como aproximada. La dirección textual sí es exacta.

## 5. MINREL — Embajadas acreditadas en Chile

- **Ítems MICR que poblaría**: 596 (Embajadas)
- **Fuente oficial identificada**: "Guía Diplomática / Lista del Cuerpo Diplomático y Organismos Internacionales acreditados en Chile" (PDF de MINREL).
- **URLs probadas** (las tres devuelven **HTTP 302 → `https://www.minrel.gob.cl/`**, con y sin `Referer`, con User-Agent de navegador):
  - `…/minrel/site/docs/20250624/20250624172948/guia_diplomatica____version_al_25_de_marzo_de_2026.pdf`
  - `…/minrel/site/docs/20250624/20250624172948/guia_diplomatica____version_al_1_de_junio_de_2025__1_.pdf`
  - `…/minrel/site/docs/20201026/20201026163539/guia_diplomatica____diciembre_12_2024__ley_de_modernizacion.pdf`
- `https://www.minrel.gob.cl/embajadas-y-consulados` es un selector JavaScript sin listado servible (guardado como evidencia).
- **Veredicto**: **NO SIRVE (por ahora)** — la fuente existe y es la correcta, pero **no se pudo descargar**. Enlace caído del lado del servidor. Queda pendiente pedirlo por Transparencia o reintentar.

## 6. SERVEL — Juntas Electorales

- **Ítems MICR que poblaría**: 610 (Oficinas de Registro Electoral)
- **URL exacta**: `https://www.servel.cl/contacto/juntas-electorales/` · Excel: `https://www.servel.cl/wp-content/uploads/2026/05/Juntas-Electorales.xlsx`
- **Formato**: XLSX (60 KB) + HTML · **Registros**: **113** · **con coordenada: 0**
- **robots.txt**: `https://www.servel.cl/robots.txt` → **HTTP 404** (no existe robots.txt)
- **Crudo**: `busqueda/crudo/servel_juntas_electorales/2026-08-25/`
- **Veredicto**: **PARCIAL**. 113 juntas con domicilio, pero la dirección viene **embebida dentro del campo "Circunscripción"**, mezclada con la lista de comunas y con basura `_x000d_`; hay que parsearla. Requiere geocodificación.
- **Hallazgo negativo importante**: **SERVEL no publica un catastro georreferenciado de locales de votación.** Usa georreferenciación interna para asignar electores (Ley 21.385), pero no la publica como capa ni como archivo. La página `/direcciones-regionales/` hoy responde 404; la sede central es Esmeralda 611, Santiago.

## 7. SII — oficinas y puntos de atención

- **Ítems MICR que poblaría**: 586 (Sistemas Tributarios); apoyo a 614
- **URLs exactas**:
  (a) `https://www.sii.cl/transparencia/oficinas_atencion.html`
  (b) `https://www.sii.cl/destacados/renta/2026/puntos_atencion_or_2026.html` → CSV `https://www.sii.cl/destacados/renta/2026/naf_nonaf_2026.csv`
- **Formato**: (a) HTML → CSV · (b) CSV con `;`
- **Registros**: (a) **69** oficinas, **0 con coordenada** · (b) **39** puntos, **39 con coordenada**
- **robots.txt**: `https://www.sii.cl/robots.txt` → HTTP 200 pero devuelve la página de error 404 del sitio, **no** un robots real.
- **Crudo**: `busqueda/crudo/sii_oficinas/2026-08-25/`
- **Veredicto**: **PARCIAL**. (a) es el listado permanente de direcciones regionales y unidades del SII, sin coordenada. (b) sí trae lat/lon pero son **puntos temporales** de la Operación Renta 2026 (universidades y bibliotecas municipales, vigentes en abril 2026), no inmuebles del SII. Además las columnas `Latitud`/`Longitud` vienen mal formateadas (`-18.488.283`); la coordenada limpia está en la columna `Link`.

## 8. TGR — Tesorerías

- **Ítems MICR que poblaría**: 586 (Sistemas Tributarios)
- **URL exacta**: `https://www.tgr.gob.cl/oficinas/`
- **Formato**: HTML → CSV · **Registros**: **51** con dirección · **con coordenada: 0**
- **robots.txt**: **NO VERIFICADO**
- **Crudo**: `busqueda/crudo/tgr_oficinas/2026-08-25/`
- **Veredicto**: **PARCIAL**. Estructura limpia (Tesorería / Dirección / Horario / Comunas), muy fácil de geocodificar. `tesoreria.cl` no resuelve; el host vivo es `tgr.gob.cl`.

## 9. Delegaciones Presidenciales Regionales y Provinciales

- **Ítems MICR que poblaría**: 574 (Palacios de Gobierno)
- **Organismo**: Ministerio del Interior — División de Gobierno Interior
- **URLs**: índices `https://www.interior.gob.cl/regionales/` y `https://www.interior.gob.cl/provinciales/`; sitios `https://dpr<region>.dpr.gob.cl/` y `https://dpp<provincia>.dpp.gob.cl/`
- **Formato**: HTML (56 archivos guardados) → CSV + JSON
- **Registros**: **56 sitios** (16 regionales + 40 provinciales), **los 56 responden HTTP 200** · **con coordenada: 0** (ninguno publica mapa embebido) · con dirección extraíble automáticamente: **~18-20 de 56**
- **robots.txt**: **NO VERIFICADO**
- **Crudo**: `busqueda/crudo/delegaciones_presidenciales/2026-08-25/`
- **Veredicto**: **PARCIAL**. El universo está completo y verificado, pero las plantillas de los 56 sitios son heterogéneas y la dirección suele estar en el pie sin etiqueta.
- **Camino limpio identificado (no recorrido)**: la ficha del **Portal de Transparencia** trae "Dirección" y "Contacto y Oficinas de Atención" por organismo:
  `https://www.portaltransparencia.cl/PortalPdT/directorio-de-organismos-regulados/?org=<COD>` — verificada para `AB007` = Delegación Presidencial Regional de Arica y Parinacota → *"Calle San Marcos N° 157"*.
  **No se recorrió el resto porque su `robots.txt` (HTTP 200) declara `Crawl-delay: 60`**, es decir una petición por minuto. Recorrer los ~56 organismos tomaría ~1 hora respetando esa regla.

## 10. Archivo Nacional

- **Ítems MICR que poblaría**: 592 (Archivos Físicos · Gobierno) y 611 (Centros de Almacenamiento · Documentos)
- **Organismo**: Archivo Nacional — Servicio Nacional del Patrimonio Cultural
- **URL**: `https://www.archivonacional.gob.cl/<slug>` (12 sedes: ARNAD, ANH y 10 archivos regionales)
- **Formato**: HTML → CSV · **Registros**: **12**, todas HTTP 200 · **con coordenada: 0**
- **robots.txt**: **NO VERIFICADO**
- **Crudo**: `busqueda/crudo/archivo_nacional/2026-08-25/`
- **Veredicto**: **PARCIAL**. El universo institucional está completo. La extracción automática de la dirección salió bien en 6 de 12 (Atacama, Aysén, La Araucanía, Valparaíso, Maule y la casa central Miraflores 50); en las otras 6 el HTML quedó guardado y hay que leer la dirección a mano. **No se inventó ninguna dirección.**

## 11. Congreso Nacional

- **Ítems MICR que poblaría**: 573 (Parlamentos)
- **URL**: `https://www.senado.cl/` (pie de página)
- **Registros**: **2** sedes publicadas — *Avenida Pedro Montt s/n, Valparaíso* y *Morandé 441, Santiago* · **con coordenada: 0**
- `https://www.camara.cl/` responde **HTTP 403** a peticiones automatizadas (respuesta guardada).
- **robots.txt**: **NO VERIFICADO**
- **Crudo**: `busqueda/crudo/congreso_nacional/2026-08-25/`
- **Veredicto**: **PARCIAL**. El ítem es minúsculo (2-3 edificios) y no existe catastro; se resuelve a mano con la dirección publicada.

## 12. Industria de Defensa — FAMAE, ASMAR, ENAER, IDIC

- **Ítems MICR que poblaría**: 748 (Fábricas de Armamento), 749 (Fábricas de Vehículos Militares), 750 (Laboratorios de I+D Militar), 787 (Fábricas de Misiles), 789 (Laboratorios de Pruebas · Armas)
- **URLs exactas**:
  - `http://www.famae.cl/contacto/`  *(www.famae.cl sólo responde por HTTP; https no conecta)*
  - `https://www.asmar.cl/contacto/`, `/astilleros/talcahuano`, `/astilleros/valparaiso`, `/astilleros/magallanes`
  - `https://www.enaer.cl/contacto/`
  - `https://www.idic.cl/`
- **Formato**: HTML de respaldo + `defensa_instalaciones_publicadas.csv`
- **Registros**: **12 instalaciones** · **con coordenada: 4** (ASMAR Talcahuano `-36,7503 / -73,1121`, ASMAR Valparaíso `-33,2817 / -71,3924`, ASMAR Magallanes `-53,0677 / -70,9258`, IDIC Santiago `-33,4737 / -70,6654`), todas tomadas del `iframe` de Google Maps que las propias entidades publican.
- **robots.txt**: **NO VERIFICADO** en ninguno de los cuatro hosts
- **Crudo**: `busqueda/crudo/defensa_empresas_publicas/2026-08-25/`
- **Veredicto**: **PARCIAL, y es todo lo que corresponde buscar.**
  - FAMAE publica en su propia página de contacto: Plantas Industriales Talagante (Avda. Manuel Rodríguez 02), Oficina Santiago (Avda. Ejército Libertador 353) y cuatro Centros de Mantenimiento Industrial (Pozo Almonte, Arica, Antofagasta, Punta Arenas).
  - **Los dos que FAMAE declara dentro de cuarteles del Ejército (Pozo Almonte y Arica) se transcriben tal como FAMAE los publica y se dejaron deliberadamente SIN coordenada.** No se buscó, dedujo ni cruzó nada para ubicarlos.
  - ASMAR publica 3 astilleros + casa matriz; ENAER publica su planta de El Bosque; el IDIC (Instituto de Investigaciones y Control del Ejército) publica Av. Pedro Montt 2136.

## 13. Fuentes revisadas que NO aportaron nada al lote

| Fuente | Qué se comprobó | Resultado |
|---|---|---|
| **datos.gob.cl** (CKAN, `package_search`) | consultas: registro civil, servel/locales de votación, chileatiende, bienes nacionales, embajadas/consulados, oficinas de atención | Ningún catastro georreferenciado de Gobierno ni Defensa. Casi todo es material de Transparencia (presupuestos, atenciones, encuestas). robots.txt: HTTP 200, **`Disallow: /api/`, `Crawl-Delay: 10`** — se hicieron sólo 6 consultas de descubrimiento, espaciadas |
| **IDE Chile / geoportal.cl** (catálogo) | categorías `intelligenceMilitary` (**0 capas**), `society` (5, todas SAG/INDAP), `structure` (5, todas observatorios); búsquedas: gobierno, defensa, militar, embajada, votación, delegación, archivo, emergencia | **No existe ninguna capa de Gobierno ni de Defensa en el IDE.** robots.txt: HTTP 200, `Disallow:` (todo permitido) |
| **Servicio de Registro Civil e Identificación** (`registrocivil.cl`, `srcei.cl`) | `/oficinas`, `/principal/oficinas`, `robots.txt` | **BLOQUEADO**: el host responde con una página de desafío JavaScript de protección anti-bot (F5/TSPD) en todas las rutas, incluido `robots.txt`. Cero contenido útil. Se anota y se sigue: no se intentó saltar la protección |
| **Portal de Transparencia** (`portaltransparencia.cl`) | ficha `?org=AB007` — trae Dirección por organismo | Fuente válida y transversal, **pero `robots.txt` declara `Crawl-delay: 60`**. Se hizo **1** petición. No se recorrió masivamente |
| **interior.gob.cl** `/regionales/` `/provinciales/` | HTTP 200 pero el HTML no contiene ni un solo `href` real | Renderizado por JavaScript. La lista de 56 dominios se obtuvo con un lector de páginas y **después se verificó una por una con HTTP** |
| **interior.gob.cl/transparenciaactiva/** | índice de 73 organismos | Es el **"SITIO HISTÓRICO 2009-2021"**: habla de Intendencias y Gobernaciones, figuras suprimidas en 2021. Obsoleto |
| **www.camara.cl**, **www.igm.cl** | GET con User-Agent de navegador | **HTTP 403** a peticiones automatizadas |
| **tesoreria.cl** | GET | No resuelve. El host vivo es `tgr.gob.cl` |

---

## Tabla resumen · ítem → fuente → registros

| Ítem | Nombre | Fuente | Registros | Con coordenada | Veredicto |
|---|---|---|---|---|---|
| 573 | Parlamentos | Senado (pie del sitio) | 2 | 0 | PARCIAL |
| 574 | Palacios de Gobierno | Delegaciones Presidenciales (56 sitios, HTTP 200) | 56 | 0 | PARCIAL |
| 586 | Sistemas Tributarios | SII Transparencia (69) + TGR (51) + SII NAF Renta 2026 (39) | 159 | 39 | PARCIAL |
| 592 | Archivos Físicos (Gobierno) | Archivo Nacional | 12 | 0 | PARCIAL |
| 595 | Centros de Comando (Emergencias) | SENAPRED Direcciones Regionales | 16 | 0 | PARCIAL |
| 596 | Embajadas | MINREL — Guía Diplomática (PDF) | — | — | NO SIRVE (enlace 302 a la home) |
| 597 | Consulados | consulado.gob.cl — fichas | 53 | 51 | PARCIAL/SIRVE |
| 610 | Oficinas de Registro Electoral | SERVEL — Juntas Electorales (XLSX) | 113 | 0 | PARCIAL |
| 611 | Centros de Almacenamiento (Documentos) | Archivo Nacional (mismo que 592) | 12 | 0 | PARCIAL |
| 614 | Oficinas de Atención Ciudadana | ChileAtiende — sucursales | 375 | 322 | **SIRVE** |
| 748 | Fábricas de Armamento | FAMAE (contacto) | 2 | 0 | PARCIAL |
| 749 | Fábricas de Vehículos Militares | ASMAR (3 astilleros + matriz) + ENAER | 5 | 3 | PARCIAL |
| 750 | Laboratorios de I+D Militar | IDIC | 1 | 1 | PARCIAL |
| 787 | Fábricas de Misiles | FAMAE Talagante (línea "cohetes y misiles") | 1 | 0 | PARCIAL |
| 789 | Laboratorios de Pruebas (Armas) | IDIC (mismo que 750) | 1 | 1 | PARCIAL |
| — | (colateral, no del lote) | MBN Propiedad Fiscal Administrada | 9.074 | 9.074 | NO SIRVE para Gobierno |

**Totales del lote**: **806 registros nuevos** repartidos en 15 de los 29 ítems, de los cuales
**416 traen coordenada** (322 ChileAtiende + 51 consulados + 39 SII-NAF + 4 Defensa).
Aparte quedaron bajados 9.074 polígonos de propiedad fiscal que no sirven para atribuir función de gobierno.

---

## Ítems SIN fuente

### A. No encontré fuente pública (14 ítems)

**Gobierno (3):**
- **576 Centros de Datos Gubernamentales** — Chile tiene un *Plan Nacional de Data Centers* (MinCiencia) y un modelo multinube estatal, pero **no publica ninguna nómina ni ubicación** de centros de datos del Estado. Tampoco aparece en datos.gob.cl ni en el IDE.
- **589 Infraestructura de Conectividad (Gobierno)** — la Red de Conectividad del Estado no publica nodos ni trazado.
- **613 Redes de Diplomacia Digital** — no es infraestructura física localizable; no hay catastro.

**Industria de Defensa (11):** 751 Almacenes de Armas · 763 Almacenes de Componentes · 766 Infraestructura de Conectividad · 769 Instalaciones de Pruebas · 770 Laboratorios de Innovación · 773 Almacenes de Residuos · 775 Centros de Capacitación (Defensa) · 776 Infraestructura de Contingencia · 780 Centros de Distribución (Armas) · 788 Fábricas de Drones · 790 Centros de Almacenamiento (Emergencia).

### B. Decidí NO buscarlos por el criterio del punto 2 (11 ítems)

Los once ítems de Defensa listados arriba se reportan además como
**NO CORRESPONDE — no publicado oficialmente**: sólo se podrían poblar deduciendo, infiriendo o
cruzando fuentes para localizar instalaciones militares que el Estado no publica. Por instrucción
expresa, **no se intentó**. Esto incluye no haber intentado inferir nada a partir de:

- los 20 predios de la capa de propiedad fiscal del MBN cuyo nombre contiene "Ejército", "Armada",
  "Fuerza Aérea" o "Carabineros" — se dejaron sin tocar;
- los dos Centros de Mantenimiento de FAMAE declarados dentro de cuarteles: se transcribió el texto
  que FAMAE publica y **no** se les asignó coordenada.

### C. Caso aparte

- **596 Embajadas** — la fuente correcta existe y está identificada (Guía Diplomática de MINREL), pero
  el PDF devuelve **302 a la home** desde el propio servidor. Es un enlace caído, no una decisión.

---

## Tres cosas que conviene saber

1. **El Estado chileno no publica catastro georreferenciado de sí mismo.** El IDE Chile tiene
   **cero capas** en la categoría `intelligenceMilitary` y ninguna de Gobierno; datos.gob.cl tampoco.
   Todo lo que se consiguió salió de **raspar sitios institucionales uno por uno**, y sólo dos fuentes
   (ChileAtiende y los `iframe` de mapas) traen coordenada de verdad. El patrón ya visto en este
   proyecto se repite: **el instrumento se degrada justo donde vive el fenómeno.**

2. **SERVEL georreferencia y no publica.** Usa un algoritmo de georreferenciación para asignar
   electores a local de votación (Ley 21.385, distancia media bajó de 2.407 m a 1.213 m), pero
   **no publica ni la capa ni el archivo**. Los locales de votación —que en emergencia son albergues—
   quedan invisibles. Lo mismo con el Registro Civil, cuyo sitio está detrás de protección anti-bot.

3. **El atajo transversal existe pero es lento a propósito.** El Portal de Transparencia tiene la
   dirección postal de **todos** los organismos del Estado en una ficha estándar por organismo, y
   funciona. Su `robots.txt` declara `Crawl-delay: 60`. Es la vía limpia para completar 574, 592, 610
   y 611 de una sola pasada, pero hay que presupuestar **un minuto por organismo**.
