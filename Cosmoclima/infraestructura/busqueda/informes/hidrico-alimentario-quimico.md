# Catastros públicos georreferenciados para los sectores Hídrico, Alimentario y Químico de la MICR

Fecha del barrido: **2026-08-25**
Ítems del lote: Hídrico 36 · Alimentario 24 · Químico 15 = **75 ítems**
Crudo bajado en: `infraestructura/busqueda/crudo/<clave>/2026-08-25/` (cada carpeta con su `PROCEDENCIA.txt`)

Todo lo que aparece abajo con número de registros fue **verificado con petición real** (código HTTP, conteo del
servidor y recuento registro por registro de cuántos traen geometría de verdad). Lo que no se verificó está
marcado **NO VERIFICADO**.

---

## Nota sobre robots.txt

| host | robots.txt | qué dice | qué hice |
|---|---|---|---|
| `lineasdebasepublicas.mma.gob.cl` | HTTP 200 | `User-agent: *` / `Disallow:` (vacío) | todo permitido |
| `www.geoportal.cl` | HTTP 200 | `User-agent: *` / `Disallow:` (vacío) | todo permitido |
| `dga.mop.gob.cl` | HTTP 200 | sólo `Disallow: /wp-admin/` | `/uploads/` permitido |
| `ide.minagri.gob.cl` | HTTP 200 | sólo `Disallow: /wp-admin/` | permitido |
| `esri.ciren.cl` | **HTTP 404** | no existe robots.txt | sin restricción declarada |
| `rest-sit.mop.gob.cl` | **HTTP 404** | no existe robots.txt | sin restricción declarada |
| `geoportal.subpesca.cl` | 404 en la raíz del servicio | sin robots.txt | sin restricción declarada |
| `www.siss.gob.cl` | HTTP 200 | **`Disallow: /586/w3-*` y `/587/w3-*`** | ⚠️ ahí es donde SISS publica sus tablas regionales de PTAS. **NO se scrapeó siss.gob.cl.** El dato se sacó del Geoportal, que sí lo permite |
| `datos.gob.cl` y `datos.odepa.gob.cl` | HTTP 200 | `Disallow: /api/`, `Crawl-Delay: 10` | ⚠️ usé la API CKAN **sólo para descubrir** fichas, a ritmo lento y con pausas; las descargas se hicieron por `/dataset/.../download/`, que sí está permitido. Queda anotado como uso en zona gris |
| `www.ide.cl` | HTTP 200 | bloquea `GPTBot`, `Google-Extended`, `meta-externalagent`; `User-agent: *` → `Allow: /` | no se usó ide.cl para bajar dato |

---

# HÍDRICO

## 1. ★ Inventario Público de Glaciares de Chile 2022 (DGA) — LA PIEZA QUE FALTABA

- **Ítems MICR:** **1 Glaciares** (directo) · **2 Nacientes Cordilleranas y Mantos de Nieve** (parcial, vía "Glaciarete")
- **Organismo:** Dirección General de Aguas (DGA/MOP), publicado por el MMA en el portal *Líneas de Base Públicas* (CKAN)
- **URL:** `https://lineasdebasepublicas.mma.gob.cl/datos_abiertos/dataset/440f1c3f-6e99-45f9-9aba-4f1fe2db4b3b/resource/770c612f-17f8-484c-9acf-f4cace2e3a43/download/inventario-de-glaciares_geojson.zip`
  · ficha: `https://lineasdebasepublicas.mma.gob.cl/datos_abiertos/dataset/glaciares`
  · SHP equivalente: `.../resource/0000f580-c2f1-4b93-9320-192825162558/download/inventario-de-glaciares_shapefile.zip`
- **Formato:** GeoJSON de **polígonos** en ZIP · 141.762.718 B comprimido / 534.969.558 B expandido
- **Registros:** **78.564** · **con geometría: 78.564 (100 %)** — comprobado feature por feature: los 78.564 traen
  `Polygon` con `coordinates` no vacío, y además cada uno trae atributos `latitud`/`longitud` y `cod_epsg 4326`
- **robots.txt:** HTTP 200, `Disallow:` vacío → permitido
- **Crudo:** `busqueda/crudo/dga_glaciares/2026-08-25/` (ZIP original + `glaciares_centroides.csv` derivado)
- **Clases:** Glaciarete 54.633 · Glaciar de montaña 11.532 · Glaciar rocoso 10.794 · Glaciar de valle 930 · Glaciar efluente 675
- **Regiones:** 16 + 111 sin región
- **Campos útiles para el eje climático:** `area_km2`, `vol_km3`, `eq_agua_km3`, `hmax`, `hmin`, `hmedia`, `pendiente`,
  `orientacion`, `mzon_glaciar`, `nombre_sscuenca`
- **VEREDICTO: SIRVE.** Cobertura nacional, geometría real, reserva hídrica cuantificada por glaciar.
  ⚠️ El brief decía "~24.000 glaciares": el inventario 2022 v2 tiene **78.564**. La cifra vieja está desactualizada.

## 2. Red hidrográfica nacional "Hidrografía V2" (IDE Chile)

- **Ítems MICR:** **3 Fuentes Naturales de Agua (Ríos o Lagos)** · 2 (parcial, cabeceras por orden de Strahler) · 40 (muy parcial)
- **Organismo:** IDE Chile / Geoportal de Chile
- **URL:** `https://www.geoportal.cl/geoportal/catalog/download/e109bc13-483e-3452-b918-2456cbc3db74`
  · WFS: `https://www.geoportal.cl/geoserver/Hidrografia/wfs?service=WFS&version=1.1.0&request=GetCapabilities` (`Hidrografia:hidrografa`)
- **Formato:** Shapefile en ZIP (134.451.894 B) · **PolyLine**
- **Registros:** **122.852** · **con geometría: 122.852 (100 %)** — verificado leyendo el `.shx` y el tipo de shape de
  cada registro: 122.852 de tipo 3, **cero nulos**. El WFS confirma `numberMatched="122852"`.
- **robots.txt:** permitido
- **Crudo:** `busqueda/crudo/ide_hidrografia/2026-08-25/`
- **TIPO:** Quebrada 120.533 · Estero 1.377 · Rio 908 · Arroyo 23 · Canal 6 · Zanjon 5
- **CRS:** SIRGAS-Chile 2016 / UTM 19S → hay que reproyectar
- **VEREDICTO: SIRVE.** ★ Las **120.533 quebradas** son lo más valioso: son los cauces secos que se activan en el
  temporal. La memoria del proyecto ya había registrado que HydroRIVERS *no* trae quebradas secas; esta capa sí.

## 3. Acuíferos SHAC (DGA)

- **Ítems MICR:** **5 Acuíferos Subterráneos**
- **URL:** `https://dga.mop.gob.cl/uploads/sites/13/2024/07/INV_ACUIFEROS_SHAC_202302.zip` · ficha `https://geoportal.cl/geoportal/catalog/35248/Acuíferos%20SHAC`
- **Formato:** Shapefile en ZIP (11.410.407 B) · **Polygon**
- **Registros:** **715** · **con geometría: 715 (100 %)** (contados uno a uno en el `.shp`; el `.dbf` declara los mismos 715)
- **robots.txt:** `dga.mop.gob.cl` sólo bloquea `/wp-admin/`
- **Crudo:** `busqueda/crudo/dga_acuiferos_shac/2026-08-25/` (+ `shac_atributos.csv`)
- **TIPO_LIMIT:** Abierto 515 · Zona de Prohibición 102 · Área de Restricción 98
- **CRS:** WGS84 UTM 19S → reproyectar
- **VEREDICTO: SIRVE.** Da el polígono real del acuífero y, de yapa, su estado jurídico (= indicador de estrés hídrico).

## 4. Acuíferos protegidos y prohibiciones (SIT-MOP / DGA-SNIA)

- **Ítems MICR:** 5 Acuíferos · 4 Manantiales (vegas y bofedales) · 7/8/9 Pozos (parcial)
- **URL raíz:** `https://rest-sit.mop.gob.cl/arcgis/rest/services`
- **robots.txt:** HTTP 404 (no existe)
- **Crudo:** `busqueda/crudo/dga_acuiferos/2026-08-25/`

| capa | geom | declarados | bajados | con geometría |
|---|---|---|---|---|
| `DGA/Acuiferos_Protegidos/0` (vegas y bofedales protegidos) | polígono | 216 | 216 | **216 (100 %)** |
| `SNIA/SNIA_ProhibicionesCentroides/0` | punto | 3.206 | 3.206 | **3.206 (100 %)** |
| `DGA/Areas_de_Restriccion_y_Zonas_de_Prohibicion/0` | polígono | 736 | 0 | — timeout |
| `SNIA/SNIA_Prohibiciones/0` | polígono | 3.206 | 0 | — timeout |
| `DGA/Decretos_Escasez_Hidrica/0` | polígono | 627 | 0 | — timeout |
| `DGA/Declaracion_de_Agotamiento/0` | polígono | 15 | 0 | — timeout |
| `SNIA/SNIA_CAUCES/0` (Cauces de Chile) | polilínea | **378.787** | 0 | — inviable |

- **VEREDICTO: PARCIAL.** ⚠️ `rest-sit.mop.gob.cl` responde en ~2 s las capas de **puntos** y se cae por timeout
  (>300 s) con las de **polígonos**, incluso pidiendo 200 registros. Los conteos sí están verificados
  (`returnCountOnly=true`). El contenido de "Áreas de Restricción y Zonas de Prohibición" ya viene, de hecho,
  dentro del SHAC (campo `TIPO_LIMIT`), que sí se bajó completo.

## 5. ★ Plantas de Tratamiento de Aguas Servidas y sus descargas (SISS)

- **Ítems MICR:** **35 Estanques de Lodos Activados** · **38 Puntos de Vertimiento a Cursos de Agua** ·
  19/20 (parcial) · 34 Membrane Bioreactors (**ninguna PTAS declarada como MBR** → el ítem queda sin activo real)
- **Organismo:** Superintendencia de Servicios Sanitarios (SISS), distribuido por IDE Chile
- **URLs:**
  - PTAS: `https://geoportal.cl/geoportal/catalog/download/03d33855-a5ba-381a-8509-45fe76f95b76` (ficha 35180)
  - Puntos de descarga: `https://geoportal.cl/geoportal/catalog/download/7e32586f-6e70-3493-82af-3d71ee2001f0` (ficha 35369)
  - "PTAS + Descargas + Territorio Operacional + Grifos": `https://geoportal.cl/geoportal/catalog/download/3087c4d0-711f-3781-a41c-d849f2b51966` (ficha 35179)
- **Formato:** Shapefile de **puntos** dentro de **.rar** (el servidor manda `.rar` aunque la URL no lo diga).
  Se abrió con `tar -xf` (bsdtar/libarchive de macOS); **no hay unrar ni 7z en esta máquina**.
- **Registros:** **294 PTAS** y **294 puntos de descarga** · **con coordenada: 294/294 (100 %)** en ambos
  (los 294 son shape tipo 1, cero nulos). Aunque los campos se llamen `UTM_NORTE`/`UTM_ESTE`, la geometría del `.shp`
  ya viene **en grados** (lon −73,84 a −68,95 · lat −53,29 a −18,46), `.prj` = GCS_SIRGAS_2000: no hay que reproyectar.
- **robots.txt:** el del Geoportal permite todo; **el de siss.gob.cl bloquea `/586/w3-*`** (ver tabla arriba)
- **Crudo:** `busqueda/crudo/siss_ptas/2026-08-25/` (2 .rar + `ptas_puntos.csv` + `ptodescarga_puntos.csv`)
- **TIP_TRATAM:** LODOS_ACTIVADOS 180 · LAGUNA_AIREADA 60 · EMISARIO_SUBMARINO 33 · TRAT. PRIMARIO C/DESINFECCIÓN 11 ·
  LAGUNA FACULTATIVA 3 · LOMBRIFILTRO 2 · resto 5. **ESTADO_USO:** EN_OPERACION 292 · USO_FUTURO 2
- **TIPO_RECEP (descargas):** RIO_SIN_DILUCION 159 · RIO_CON_DILUCION 61 · MAR_FUERA_ZPL 34 · RIEGO 17 · resto 23
- **VEREDICTO: SIRVE**, con dos límites: el dato es de **2016-2017**, y ⚠️ **la ficha 35179 promete el
  *territorio operacional* y los *grifos* pero entrega exactamente el mismo archivo de puntos de descarga**
  (mismo nombre, mismos 20.606 bytes). El territorio operacional **NO se obtuvo**.

## 6. Bocatomas y singularidades de infraestructura de riego (CNR / CIREN)

- **Ítems MICR:** **844 Bocatoma** y 6 Captación · **20 Estanques de Sedimentación** (desarenadores) ·
  **23 Cámaras de Distribución** · **24 Válvulas de Control** (compuertas + marcos partidores) ·
  **15 Estaciones de Bombeo** (bocatomas de elevación mecánica) · 47/48 (sector Represas)
- **URL:** `https://esri.ciren.cl/server/rest/services/IDEMINAGRI/INFRAESTRUCTURA_RIEGO/MapServer`
  · alternativa SHP: `https://ide.minagri.gob.cl/descarga-de-capas-shp/aguas-continentales/`
  · ficha IDE: `https://www.geoportal.cl/geoportal/catalog/41543/Catastro%20de%20singularidades%20de%20infraestructura%20de%20riego`
- **Crudo:** `busqueda/crudo/cnr_infraestructura_riego/2026-08-25/`

| archivo | declarados | bajados | con geometría |
|---|---|---|---|
| `bocatomas.geojson` | 8.555 | 8.555 | **8.555 (100 %)** |
| `singularidades_infraestructura_riego.geojson` | 2.127 | 2.127 | **2.127 (100 %)** |
| `plan_de_embalses.geojson` | 24 | 24 | **0** ⚠️ es una tabla, no tiene geometría |

- **`singularid`:** compuerta entrega 579 · **bocatoma 444** · compuerta descarga 217 · **desarenador 167** ·
  cruce camino/ffcc 138 · alcantarilla 123 · cruce tubería 83 · marco partidor 52 · puente 50 · sifón 46 ·
  otro puntual 46 · compuerta automática 29 · **cámara 28** · caída 22 · revestimiento 19 · canoa 16 · resto 41
- **`tip_cap` bocatomas:** Gravitacional 7.598 · S/I 658 · **Elevación Mecánica 299**
- **VEREDICTO: SIRVE** para bocatomas y singularidades · **NO SIRVE** `plan_de_embalses` (0 con geometría).
- Los canales de riego CNR **ya los tiene el proyecto**; no se volvieron a bajar.

## 7. Humedales nacionales (MMA)

- **Ítems MICR:** **37 Humidales Artificiales** (sólo la clase "Sistemas antropizados", 164) · **3 Ríos y Lagos**
  (clases "Cuerpos" 15.202 y "Rios" 6.336) · 2 (parcial)
- **URL descarga directa:** `https://geoportal.cl/geoportal/catalog/download/992b5df7-abe4-3fa4-947a-bc76e54a8b77`
  → `Humedales_2015.rar`, **231.732.514 B (231 MB)**
- **URL WFS:** `https://www.geoportal.cl/geoserver/Humedales/wfs` · `typeName=Humedales:humedales`
- **Registros:** **40.378** (verificado con `resultType=hits`: `numberMatched="40378"`) ·
  **con coordenada: 40.378 (100 %)** — MultiPolygon, y además cada fila trae `lat`/`long` propios; las 40.378 filas
  del CSV tienen `lat` y `long` no vacíos
- **Crudo:** `busqueda/crudo/mma_humedales_nacional/2026-08-25/`
  - `humedales_nacional_atributos.csv` → **las 40.378 filas** con atributos + lat/long (9 MB)
  - `muestra_2000_humedales_con_geometria.geojson.gz` → los 2.000 primeros **con** geometría (muestra)
- ⚠️ **No se bajó la geometría completa**: una sola página WFS de 2.000 humedales pesa **101.970.601 B (102 MB)**;
  los 40.378 serían ~2 GB. Supera de lejos el tope de ~200 MB.
- **VEREDICTO: SIRVE** (como inventario con coordenada por humedal); **PARCIAL** si se necesita el polígono completo.

## 8. Humedales urbanos declarados (MMA, Ley 21.202)

- **Ítems MICR:** 37 (parcial) · 3 (parcial)
- **URL:** `https://lineasdebasepublicas.mma.gob.cl/datos_abiertos/dataset/6f95dcf8-51fd-474c-b34e-79521e50a4cf/resource/cd03641d-ef56-4ab2-876e-4c19b1ebbcf1/download/humedales-urbanos_geojson.zip`
- **Registros:** **107** · **con geometría: 107 (100 %)**, MultiPolygon
- **Crudo:** `busqueda/crudo/mma_humedales_urbanos/2026-08-25/`
- **VEREDICTO: PARCIAL.** Son humedales urbanos **naturales declarados**, no artificiales de tratamiento.
  Es un subconjunto de la capa nacional (punto 7).

---

# ALIMENTARIO

## 9. ★★ Rol Único Pecuario — Bovinos 2025 (SAG, vía CIREN/IDE MINAGRI)

- **Ítems MICR:** **399 Granjas Ganaderas (Bovinos)**
- **URL:** `https://esri.ciren.cl/server/rest/services/IDEMINAGRI/ROL_UNICO_PECUARIO/MapServer` (capas 0, 2, 3)
- **Formato:** ArcGIS REST (esriJSON) → GeoJSON de **puntos**. `maxRecordCount=2000`, paginación soportada.
- **Registros:** **77.943** (77.885 continental + 53 Isla de Pascua + 5 Juan Fernández) ·
  **con geometría: 77.943 (100 %)**
- **robots.txt:** `esri.ciren.cl` → 404 (no existe)
- **Crudo:** `busqueda/crudo/ciren_rol_unico_pecuario/2026-08-25/`
- **Campos:** `rup` (rol único pecuario), `sector_sag`, `nom_com`, `x_coor`/`y_coor`/`huso`, `especie`
- **VEREDICTO: SIRVE.** Cobertura nacional completa, un punto por establecimiento pecuario.
  ⚠️ `especie` = **Bovinos** en los 77.943: el RUP de **aves y porcinos NO está publicado** en este servicio
  (el folder `Plantas_aves_2020` del servidor del SAG existe pero está vacío/cerrado al público) → **ítem 400 sin fuente**.

## 10. ★ Catastro Frutícola CIREN-ODEPA (capas espaciales)

- **Ítems MICR:** **398 Campos de Cultivo (Frutas y Verduras)** · **405 Almacenes Refrigerados** ·
  **406 Cámaras de Maduración** (parcial) · **409 Plantas de Envasado** · 407/408/440 (agroindustrias, sin desglose de rubro)
- **URL:** `https://esri.ciren.cl/server/rest/services/IDEMINAGRI/CATASTRO_FRUTICOLA/MapServer` (69 capas: 5 grupos × región)
- **Crudo:** `busqueda/crudo/ciren_catastro_fruticola/2026-08-25/`

| archivo | ítem MICR | declarados | bajados | con geometría |
|---|---|---|---|---|
| `productores_fruticolas.geojson` | 398 | 94.731 | 94.731 | **94.731 (100 %)** |
| `agroindustrias.geojson` | 407/408/440 | 1.425 | 1.425 | **1.425 (100 %)** |
| `plantas_embalaje.geojson` | 409 | 916 | 916 | **916 (100 %)** |
| `camaras_de_frio.geojson` | 405 | 480 | 480 | **480 (100 %)** |
| `camaras_de_fumigacion.geojson` | 406 (parcial) | 66 | 66 | **66 (100 %)** |
| **total** | | **97.618** | **97.618** | **97.618 (100 %)** |

- `agroindustrias.desctiem`: ARTESANAL 588 · INDUSTRIAL 584 · sin dato 253
- Los años varían por región (2016 a 2025) y quedan registrados en el campo `_capa_nombre` de cada feature.
- **VEREDICTO: SIRVE.**
- ⚠️ El CSV tabular del Catastro Frutícola en ODEPA
  (`https://datos.odepa.gob.cl/dataset/ea82304e-.../download/catastro_fruticola_2025.csv`) trae especie, variedad,
  año de plantación, método de riego y superficie **pero sin coordenadas** (sólo comuna). Estas capas de CIREN sí traen el punto.

## 11. ★ Terrenos Agrícolas del Catastro de Usos de la Tierra (CONAF, vía CIREN)

- **Ítems MICR:** **397 Campos de Cultivo (Granos)** y **398 Campos de Cultivo (Frutas y Verduras)**
- **URL:** `https://esri.ciren.cl/server/rest/services/USOS_DE_LA_TIERRA__CONAF/MapServer` (25 capas regionales/provinciales),
  filtrando `where=uso='Terrenos Agrícolas'`
- **Registros:** **74.981** · **con geometría: 74.981 (100 %)** — todos con `rings`, cero nulos
- **Crudo:** `busqueda/crudo/conaf_usos_tierra_agricola/2026-08-25/terrenos_agricolas.geojson.gz`
  (568 MB sin comprimir → se guarda gzip de 160 MB) + `detalle_capas.txt` con el desglose por capa
- **uso_tierra:** Rotación Cultivo-Pradera 46.850 · Terreno(s) de Uso Agrícola 28.131 (tres grafías distintas del mismo valor)
- **VEREDICTO: SIRVE.** El catastro CONAF **no separa grano de fruta**: da el polígono agrícola. Para partir 397 de 398
  hay que cruzarlo con los 94.731 puntos de productores frutícolas del punto 10.
- ⚠️ **TRAMPA:** los nombres de campo **no son iguales en todas las capas** (unas tienen `objectid_1`+`id_uso`, otras
  sólo `objectid`). Pedir una lista fija de `outFields` hace que el servidor devuelva
  `{"error":{"code":400,"message":"Failed to execute query."}}` en **23 de 25 capas sin decir por qué**.
  Hay que leer los `fields` de cada capa y pedir sólo los que existen.

## 12. Caletas Pesqueras Artesanales (SUBPESCA)

- **Ítems MICR:** 414 Mercados Locales (parcial) · 412 Centros de Logística Alimentaria (parcial)
- **URL:** `https://geoportal.subpesca.cl/server/rest/services/IDE_PUBLICO/SRMPUB_CALETAS_PESQUERAS/MapServer/0`
- **Registros:** **481** · **con geometría: 481 (100 %)**, puntos
- **Crudo:** `busqueda/crudo/subpesca_caletas_pesqueras/2026-08-25/`
- **Campos:** NOMBRE, C_REGION, C_COMUNA, NROORG, NROEMBT (embarcaciones), NROPESM/NROPESH (pescadores), ESPECIES, PROPIEDAD
- **VEREDICTO: PARCIAL.** Sirve como capa de abastecimiento alimentario costero, no como "feria libre" urbana.
- El proyecto **ya tiene** las concesiones de acuicultura de SUBPESCA; esta capa es distinta.

## 13. Ferias Libres (IDE Chile / SNIT)

- **Ítems MICR:** **414 Mercados Locales (Ferias)**
- **URL:** `https://geoportal.cl/geoportal/catalog/download/9449ddc0-50c4-3b4e-b192-ad37023ebfa1` (fichas 35097 y 41484)
- **Registros:** **370** · **con geometría: 370 (100 %)**
  ⚠️ **son LÍNEAS (shape type 3), no puntos**: cada feria es el tramo de calle que ocupa
- **Crudo:** `busqueda/crudo/geoportal_ferias_vertederos/2026-08-25/` (+ `ferias_libres.csv` con el centro del bounding box)
- **Campos:** NOMBRE_FER, COMUNA, CALLE, LIMITE_1/2, LUNES..DOMINGO, PUESTOS, ALIMENTOS
- **VEREDICTO: PARCIAL.** Los 370 registros son **todos de la RM**. Las fichas de Valparaíso (41485) y Biobío (41486)
  **no ofrecen archivo**: sólo enlazan a `https://icet.odepa.gob.cl/`. Es el único catastro georreferenciado de ferias que encontré.

---

# QUÍMICO

## 14. ★ Catastro Nacional de Suelos con Potencial Presencia de Contaminantes (SPPC, MMA)

- **Ítems MICR:** **675 Materiales Químicos Peligrosos** (370 "mal manejo de sustancias peligrosas") ·
  **676 / 699 / 700 Producción química, fertilizantes, cloro** (153 "formulación o fabricación de productos químicos") ·
  **684 Instalaciones de Tratamiento de Residuos** (73 reciclaje y valorización) ·
  **692 Almacenes de Residuos Químicos** (1.519 disposición de residuos sólidos + 1.643 residuos mineros masivos) ·
  663 Centros de Distribución (parcial)
- **URL:** `https://lineasdebasepublicas.mma.gob.cl/datos_abiertos/dataset/0749d4f1-5d20-4797-9bec-66d34e319a4a/resource/304d7956-a9ef-4bf8-b9a7-f75633eb8513/download/catastro-nacional-de-suelos-con-potencial-presencia-de-contaminantes-sppc_geojson.zip`
  · también en XLSX, CSV, SHP, GeoPackage y Parquet en la misma ficha
- **Registros:** **10.255** · **con geometría: 10.255 (100 %)** — MultiPolygon + atributos `latitud`/`longitud` en cada fila
- **robots.txt:** permitido
- **Crudo:** `busqueda/crudo/mma_sppc_suelos_contaminantes/2026-08-25/` (+ `sppc_puntos.csv`)
- **`actividad_principal`:** Estaciones de servicio de combustibles 2.027 · Extracción/procesamiento Cu-Ag-Mo-Au 1.730 ·
  Disposición de Residuos Mineros Masivos 1.643 · Disposición de residuos sólidos 1.519 · Minería petróleo y gas 920 ·
  Industria Forestal 610 · **Mal manejo de sustancias peligrosas 370** · Talleres mecánicos 358 · Minería no metálica 302 ·
  **Formulación o fabricación de productos químicos 153** · Fabricación de muebles 85 · Generación eléctrica >3MW 83 ·
  Maestranzas/astilleros 74 · **Reciclaje y valorización de residuos 73** · Cemento/hormigón/asfalto 59 · Puertos 49 ·
  Carbón 49 · Hierro y acero 45 · Tratamiento de metales 16 · Curtiembre 9 · Reciclaje electrónicos 4 · Baterías Pb 2 · resto 4
- **`prioridad`:** Alta 5.743 · Mediana 3.979 · NoPriorizado 364 · Moderada 129 · Baja 40
- **VEREDICTO: SIRVE.** Es **el** catastro químico que le faltaba al proyecto. El campo `actividad_principal` es
  exactamente lo que permite repartir las filas por ítem de la MICR, y la geometría es el **polígono del sitio**
  (no un punto), así que el cruce con amenaza de inundación o remoción en masa se puede hacer sin aproximar.

## 15. Vertederos Ilegales (MMA)

- **Ítems MICR:** 692 (parcial) · 684 (parcial)
- **URL:** `https://geoportal.cl/geoportal/catalog/download/f8dda955-b14b-35de-809b-a7ee51d0ec99` (ficha 35653)
- **Registros:** **78** · **con geometría: 78 (100 %)**, puntos, SIRGAS-Chile UTM → reproyectar
- **Crudo:** `busqueda/crudo/geoportal_ferias_vertederos/2026-08-25/`
- **VEREDICTO: PARCIAL.** Muy pocos registros, cobertura sólo RM/Valparaíso, dato de 2020. Para residuos, el catastro
  fuerte es el SPPC (punto 14).

## 16. ⚑ Hallazgo sobre lo que el proyecto YA tiene: el RETC cubre 3 de mis sectores y no estaba mapeado

`datos/crudo/retc/2026-08-25/establecimientos_ckan.xlsx` — **26.649 establecimientos con `Latitude`/`Longitude`** —
ya está bajado, pero su campo **`Rubro RETC`** no se había usado para poblar ítems de estos tres sectores:

| Rubro RETC | n | ítems MICR que poblaría |
|---|---|---|
| Suministro y tratamiento de aguas | 470 | 12, 15, 19, 20, 34, 35, 36, 38 |
| Producción de alimentos | 645 | 407, 408, 409, 440 |
| Industria agropecuaria y silvicultura | 1.204 | 399, 400, 404 |
| Gestor de residuos | 1.354 | 684, 692 |
| Producción química | 125 | 662, 676, 699, 700 |
| Combustibles | 683 | (Energía) |
| Pesca | 1.298 | 407, 412 |

**No se volvió a bajar** (instrucción explícita), pero conviene explotarlo: son ~5.100 activos georreferenciados
adicionales para Hídrico/Alimentario/Químico que están en disco sin usar.

## 16bis. ★★ SNIFA / SMA — Unidades Fiscalizables: 3.067 activos de MIS TRES SECTORES

⚠️ **Corrección a lo que escribí más arriba.** Empecé descartando el SNIFA porque su buscador web no tiene API.
Un agente en paralelo sí llegó al archivo, por las carpetas públicas de Google Drive que la SMA enlaza desde
`https://snifa.sma.gob.cl/DatosAbiertos`. El crudo ya está en
`busqueda/crudo/snifa_unidades_fiscalizables/2026-08-25/UF_Instrumentos.csv` (13,7 MB, 33.558 filas,
**22.186 unidades fiscalizables únicas, 98,9 % con lat/lon**). **No lo volví a bajar**; lo que sigue es el
mapeo de su campo `SubCategoriaEconomicaNombre` a los ítems de mi lote, contado sobre el archivo real.

| SubCategoríaEconómicaNombre | n | con coord | ítem MICR |
|---|---|---|---|
| **Planta de Tratamiento de Aguas Servidas** | 528 | **524** | **34/35/36** |
| Sistema de agua potable | 293 | 293 | 12, 19, 29 |
| Planta de tratamiento de RILes | 106 | 106 | 684 |
| Vertedero | 98 | 97 | 692 |
| Relleno sanitario | 76 | 76 | 692 |
| Alcantarillado | 52 | 52 | 31 |
| Planta de reciclaje | 41 | 41 | 684 |
| Planta de compostaje | 37 | 37 | 684 |
| Planta de tratamiento de RISes | 36 | 36 | 684 |
| Planta de tratamiento de residuos peligrosos | 27 | 27 | **684, 692** |
| Planta desaladora | 13 | 13 | 6, 19 |
| Emisario submarino | 12 | 12 | 38 |
| Mono relleno | 6 | 6 | 692 |
| Embalse / presa / tranque | 71 | 70 | 10, 11, 45 |
| Canal / acueducto / sifón | 46 | 46 | 40, 47 |
| Drenaje / desecación | 6 | 6 | 40 |
| Planta procesadora de productos agrícolas | 345 | 343 | 409, 440 |
| Vinificación / producción de alcohol | 295 | 295 | 440 |
| Producción agrícola / cultivos | 181 | 180 | 397, 398 |
| **Plantel de cerdos** | 141 | **139** | (sector pecuario) |
| **Matadero / frigorífico** | 95 | **95** | **407 Plantas de Procesamiento (Carnes)** |
| **Plantel de aves** | 81 | **81** | **400 Granjas Avícolas (Pollos)** |
| **Elaboración de productos lácteos** | 80 | **79** | **408 Plantas de Procesamiento (Lácteos)** |
| Plantel de bovinos / equinos / ovinos | 28 | 27 | 399 |
| Planta procesadora de productos pecuarios | 22 | 22 | 407 |
| **Feria agrícola** | 22 | **22** | **414 Mercados Locales (Ferias)** — y a nivel NACIONAL |
| Planta deshidratadora | 6 | 6 | 440 |
| **Planta procesadora de productos químicos** | 132 | **132** | **676, 699, 700** |
| Planta de Polímeros sintéticos | 93 | 92 | 676 |
| Plantas procesadoras de alimentos (no agrícola) | 62 | 61 | 440 |
| Embotelladora | 27 | 27 | 409 |
| Reciclaje de residuos y sustancias peligrosas | 24 | 24 | 692 |
| **TOTAL** | **3.082** | **3.067** | |

**Esto cierra tres ítems que yo había dado por perdidos:**
- **400 Granjas Avícolas (Pollos)** → 81 planteles de aves georreferenciados
- **407 Plantas de Procesamiento (Carnes)** → 95 mataderos/frigoríficos + 22 procesadoras pecuarias
- **408 Plantas de Procesamiento (Lácteos)** → 80 plantas lácteas

Y **duplica** la cobertura de PTAS: la SISS publica 294 plantas (concesionarias urbanas) y el SNIFA fiscaliza **528**
(incluye las no concesionadas: mineras, agroindustriales, sanitarias rurales). Hay que unir las dos, no elegir una.

---

# Fuentes probadas que NO sirvieron (y por qué)

| fuente | qué pasó |
|---|---|
| ~~**SNIFA / SMA**~~ | **DESCARTADO POR ERROR MÍO — ver punto 16bis.** El buscador web no tiene API (formulario POST a `/UnidadFiscalizable/Resultado`, HTML paginado), pero los datos abiertos SÍ se bajan desde las carpetas de Google Drive que enlaza `/DatosAbiertos`. Otro agente lo obtuvo: 22.186 unidades, 98,9 % con coordenada |
| **SIT SISS** (`https://sit.siss.gob.cl`) | HTTP 200, es un portal i-datum. No expone geoserver, WMS/WFS ni MapServer: probé `/geoserver/web/`, `/geoserver/wfs`, `/cgi-bin/mapserv`, `/ows` → todos 404 |
| **Geoportal SAG** (`https://geoportal.sag.gob.cl/server/rest/services`) | ArcGIS Server público con folders `Hosted`, `Plantas_aves_2020`, `SAG_ESPACIAL`, `Survey123EGDB` — **todos vacíos al listarlos anónimamente**. Sólo se ve un GPServer. Ahí estaría el catastro de planteles avícolas (ítem 400) |
| **SIT CONAF** (`sit.conaf.cl`) | HTTP 200 la web, pero `/geoserver/wfs`, `/server/rest/services` y `/arcgis/rest/services` → 404. El dato de CONAF sí está, pero republicado por CIREN (punto 11) |
| **`ide.mma.gob.cl` / `sit.mma.gob.cl` / `sig.sernapesca.cl` / `arcgis.ciren.cl`** | 404 / no resuelven |
| **ODEPA — catastro frutícola CSV** | 26 años de CSV descargables, pero **sin coordenadas**: la unidad espacial es la comuna |
| **`SNIA/SNIA_CAUCES`** (378.787 polilíneas) | conteo verificado, pero a la velocidad de `rest-sit.mop.gob.cl` con polígonos/líneas serían horas. **Reemplazado por la capa Hidrografía V2 del IDE (122.852 tramos, un solo ZIP)** |
| **INE — Encuesta de Mataderos de Ganado / de Aves** | son encuestas estadísticas agregadas, sin establecimientos ni coordenadas |
| **`datos.gob.cl`: "planteles", "avícola", "porcino", "faenadora", "supermercados", "bancos de alimentos"** | **0 resultados** con dato georreferenciado |

---

# Tabla resumen: ítem → fuente → registros

| ítem | nombre | fuente | registros con geometría |
|---|---|---|---|
| **1** | Glaciares | DGA — Inventario Público de Glaciares 2022 | **78.564** |
| 2 | Nacientes Cordilleranas y Mantos de Nieve | DGA glaciares (clase Glaciarete) — parcial | 54.633 |
| **3** | Fuentes Naturales de Agua (Ríos o Lagos) | IDE Hidrografía V2 (Rio+Estero+Arroyo 2.308 / total 122.852) + MMA Humedales (Cuerpos+Rios) | **122.852** + 21.538 |
| 4 | Manantiales Naturales | DGA Acuíferos que alimentan vegas y bofedales — parcial | 216 |
| **5** | Acuíferos Subterráneos | DGA Inventario SHAC | **715** |
| 6 | Puntos de Captación | CNR/CIREN bocatomas | 8.555 |
| 7/8/9 | Pozos | SNIA Prohibiciones (centroides) — parcial | 3.206 |
| 11 | Embalses Pequeños | CNR plan de embalses — **0 con geometría** | 0 |
| 15 | Estaciones de Bombeo de Agua | bocatomas `tip_cap=Elevación Mecánica` | 299 |
| 19/20 | Filtros / Estanques de Sedimentación | CNR desarenadores 167 + SISS trat. primario 11 | 178 |
| 23 | Cámaras de Distribución de Agua | CNR singularidad "camara" | 28 |
| 24 | Válvulas de Control Principales | CNR compuertas 825 + marcos partidores 52 | 877 |
| 34 | Membrane Bioreactors (MBRs) | — **ninguna PTAS declarada como MBR** | 0 |
| **35** | Estanques de Lodos Activados | SISS PTAS `TIP_TRATAM=LODOS_ACTIVADOS` | **180** |
| 36 | Digestores Anaerobios | — sin tipo equivalente en SISS | 0 |
| **37** | Humidales Artificiales | MMA Humedales clase "Sistemas antropizados" | **164** |
| **38** | Puntos de Vertimiento a Cursos de Agua | SISS puntos de descarga PTAS | **294** |
| 40 | Canales de Drenaje | Hidrografía "Canal"+"Zanjon" 11 — muy parcial | 11 |
| **844** | Bocatoma de Agua Potable Rural | CNR bocatomas 8.555 + singularidad "bocatoma" 444 | **8.999** |
| **397** | Campos de Cultivo (Granos) | CONAF Terrenos Agrícolas | **74.981** |
| **398** | Campos de Cultivo (Frutas y Verduras) | CIREN productores frutícolas | **94.731** |
| **399** | Granjas Ganaderas (Bovinos) | SAG/CIREN Rol Único Pecuario 2025 | **77.943** |
| **405** | Almacenes Refrigerados | CIREN cámaras de frío | **480** |
| 406 | Cámaras de Maduración (Frutas) | CIREN cámaras de fumigación — parcial | 66 |
| 407/408/440 | Plantas de Procesamiento y Producción | CIREN agroindustrias | 1.425 |
| **409** | Plantas de Envasado (Alimentos) | CIREN plantas de embalaje | **916** |
| 412 | Centros de Logística Alimentaria | SUBPESCA caletas — parcial | 481 |
| **414** | Mercados Locales (Ferias) | IDE Ferias Libres (sólo RM) | **370** |
| **400** | Granjas Avícolas (Pollos) | SNIFA "Plantel de aves" | **81** |
| **407** | Plantas de Procesamiento (Carnes) | SNIFA mataderos/frigoríficos 95 + procesadoras pecuarias 22 | **117** |
| **408** | Plantas de Procesamiento (Lácteos) | SNIFA "Elaboración de productos lácteos" | **79** |
| **34/35/36** | PTAS (MBR / lodos activados / digestores) | SNIFA "Planta de Tratamiento de Aguas Servidas" (además de las 294 de SISS) | **524** |
| 31 | Tuberías de Conexión Domiciliaria (Aguas Residuales) | SNIFA "Alcantarillado" | 52 |
| 12/19/29 | Tanques, filtros, APR | SNIFA "Sistema de agua potable" | 293 |
| **675** | Materiales Químicos (Peligrosos) | MMA SPPC "mal manejo de sustancias peligrosas" | **370** |
| 676/699/700 | Producción química / fertilizantes / cloro | MMA SPPC "formulación o fabricación de productos químicos" | 153 |
| **684** | Instalaciones de Tratamiento (Residuos) | MMA SPPC reciclaje 73 + vertederos ilegales 78 | **151** |
| **692** | Almacenes de Residuos Químicos | MMA SPPC disposición residuos sólidos 1.519 + mineros masivos 1.643 | **3.162** |
| — | (capa transversal) | MMA SPPC completo | 10.255 |

**Total de activos georreferenciados nuevos bajados por mí en este lote: 391.219**
(+ 3.067 activos de mis sectores identificados dentro del SNIFA que ya bajó otro agente)
(78.564 glaciares + 122.852 tramos hídricos + 74.981 polígonos agrícolas + 77.943 bovinos + 97.618 frutícola +
10.255 SPPC + 10.706 riego/bocatomas/acuíferos/PTAS/caletas/ferias/vertederos/humedales urbanos + 40.378 humedales
como tabla con coordenada — sumando sin descontar solapes entre capas).

---

# Ítems del lote SIN NINGUNA FUENTE encontrada

**Hídrico (13 de 36):**
25 Tuberías de Conexión Domiciliaria (Agua Potable) · 32 Fosas Sépticas (Áreas Rurales) ·
43 Canales de Alimentación de Tranques de Relave · 36 Digestores Anaerobios ·
**838** Captación APR · **839** Planta de Tratamiento APR · **840** Red Matriz APR · **841** Red de Distribución APR ·
**842** Conducción APR · **843** Arranque Domiciliario APR · **845** Estanque de Regulación APR ·
34 Membrane Bioreactors *(hay 524 PTAS del SNIFA + 294 de SISS, pero ninguna declarada como MBR)* ·
11 Embalses Pequeños *(el `plan_de_embalses` de la CNR trae 0 geometrías; el SNIFA sí tiene 70 "embalse/presa/tranque")*
(10 Represa, 41 Tortas de Relaves y 29 Instalaciones APR **ya los tiene el proyecto** y por eso no se buscaron de nuevo;
 12 Tanques y 31 Tuberías de Aguas Residuales quedan cubiertos parcialmente por el SNIFA — ver punto 16bis)

> ⚠️ Los ocho ítems **838-845** son **componentes internos** de cada sistema de Agua Potable Rural. Ninguna fuente
> pública los publica por separado: el DOH publica **un punto por sistema APR**, no su captación, su estanque ni su red.
> Es una brecha estructural, no una búsqueda mal hecha.

**Alimentario (12 de 24):**
402 Invernaderos Agrícolas · 404 Silos de Almacenamiento (Granos) · 410 Almacenes de Distribución ·
413 Supermercados y Tiendas · 415 Almacenes de Alimentos (Hogares) · 416 Restaurantes y Comedores ·
417 Bancos de Alimentos · 425 Depósitos de Agua (Procesamiento) · 429 Almacenes de Emergencia (Alimentos) ·
432 Granjas Verticales (Urbanas) · 438 Almacenes de Granos (Comunidades) · 439 Huertos Urbanos

**Químico (9 de 15):**
671 Tuberías de Transporte · 678 Infraestructura de Conectividad ·
685 Laboratorios de Pruebas (Calidad) · 687 Centros de Capacitación · 688 Infraestructura de Contingencia ·
701 Laboratorios de Bioseguridad · 702 Centros de Almacenamiento (Emergencia) · 663 Centros de Distribución
*(sólo parcialmente, vía SPPC)* · 662 Laboratorios Químicos

---

# Recomendaciones concretas

1. **Correr el cruce del SPPC con la amenaza climática ya**: son 10.255 polígonos de sitio con `prioridad` declarada
   (5.743 "Alta") y actividad tipificada. Es el catastro de riesgo químico que faltaba, y viene con polígono, no con punto.
2. **Explotar el `Rubro RETC` del archivo que ya está en disco** (punto 16): ~5.100 activos más para estos tres sectores
   sin bajar nada.
3. **Unir el SNIFA con lo demás antes que nada** (punto 16bis): 3.067 activos de estos tres sectores ya están en disco,
   bajados por otro agente, y cierran los ítems 400, 407 y 408 que yo había dado por perdidos.
4. **Pedir igual al SAG el Rol Único Pecuario de aves y porcinos.** El de bovinos está publicado (77.943 puntos);
   el SNIFA sólo trae los 81 planteles de aves que fiscaliza, no el universo. El servicio `Plantas_aves_2020` del
   geoportal del SAG existe pero está cerrado al público.
5. **Pedir a la SISS el territorio operacional.** La ficha 35179 del Geoportal lo promete y entrega otro archivo.
   Y el dato de PTAS que sí está publicado es de 2016-2017.
6. **Reproyectar antes de cruzar**: Hidrografía V2 (SIRGAS-Chile 2016 UTM 19S), SHAC (WGS84 UTM 19S) y vertederos
   (SIRGAS-Chile UTM) vienen en metros; glaciares, SPPC, PTAS, ferias y todo lo de CIREN vienen ya en grados.
7. **`rest-sit.mop.gob.cl` no sirve para capas de polígonos.** Si algo de ahí hace falta, buscar primero si el mismo
   dato está republicado como archivo (fue el caso del SHAC, de la hidrografía y de la infraestructura de riego).
