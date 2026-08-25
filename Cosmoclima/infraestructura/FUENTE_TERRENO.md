# Fuente TERRENO — las condiciones del lugar que encaminan la amenaza

**17-ago-2026.** Catastro y captura de las condiciones de terreno del vocabulario
de `ESTUDIO_VECTORES_DE_AMENAZA.md` §4.1, para los 39 puntos de
`datos/subestaciones_puntos.csv` y los 91 de `datos/reterm_puntos.csv`.

Este documento dice **qué existe, qué se bajó, qué NO y por qué**. La regla es la
del proyecto: un hueco declarado vale más que un dato adivinado.

---

## 1 · El hueco que se venía a llenar

El estudio de vectores sostiene que el terreno no modula el peligro: **lo
encamina**. Misma lluvia sobre suelo seco y pendiente → remoción en masa; misma
lluvia sobre suelo saturado y cauce estrechado → desborde fluvial. Hasta hoy el
proyecto tenía el clima (ERA5, 1990-2026, 39+91 puntos) y **nada** del terreno.

Del vocabulario de §4.1, las condiciones de terreno son seis:
`HumSuelo` · `Pend` · `Veg` · `Cuenca` · `Reg`, más las de intervención humana
`Ribera` · `Defor` · `Imperm`.

---

## 2 · Estado final, condición por condición

| condición | estado | fuente | puntos con dato |
|---|---|---|---|
| `Pend` pendiente | ✅ **bajada y calculada** | Copernicus DEM GLO-30 | 130 / 130 |
| `Cuenca` superficie aguas arriba | ✅ **bajada** (con matiz, §6) | HydroRIVERS v1.0 `UPLAND_SKM` | 130 / 130 |
| `Veg` cobertura vegetal | ✅ **bajada y calculada** | ESA WorldCover 10 m, 2021 y 2020 | 130 / 130 |
| red de cauces / distancia | ✅ **bajada y calculada** | HydroRIVERS (todo el país) + IDE Chile 1:25.000 (norte de −35) | 130 / 130 · IDE: **49 / 130** (todos al norte de −35,15) |
| `Imperm` impermeabilización | ✅ derivada | WorldCover clase 50 en 1 km | 130 / 130 |
| `Reg` regolito / geología | ✅ **bajada** (procedencia frágil, §7-bis.1) | Mapa Geológico 1:1.000.000, esquema SERNAGEOMIN | **126 / 130** |
| uso de suelo oficial chileno | ✅ **bajado** | Catastro CONAF vía SMA y vía CIREN, 1:50.000 | **125 / 130** |
| `Defor` deforestación | ✅ **bajada** (§7-bis.2) | CIREN: 6 fechas de uso por polígono, 2001→2021 | **115 / 130**, de los cuales **20 cambiaron de uso** |
| `Defor` (respaldo, degradado) | 🟡 sólo indicio, confianza 0,40 | diferencia WorldCover 2020→2021 | 130 / 130 |
| `Ribera` ocupación de riberas | ❌ **SIN DATO** | ver §7.1 | 0 |
| `HumSuelo` humedad del suelo | ➖ fuera de este encargo | ESA CCI, ya en curso en otro frente | — |

Elevación no está en el vocabulario pero se entrega igual, porque es el insumo de
la alternativa 1 de `Iso0` que el propio estudio declara calculable
(`Tmax` + elevación + gradiente 6,5 °C/km).

`datos/terreno_puntos.csv`: **130 filas × 88 columnas**, 307 KB.

### ★ Hallazgo colateral: CINCO puntos de entrada están EN EL MAR

Los mismos cinco puntos que quedaron sin geología y sin uso de suelo son los
únicos cinco cuya clase de WorldCover en el pixel del punto es **«cuerpo de agua
permanente»**, con `dist_agua_permanente_m = 0,0`:

| punto | archivo de origen | `frac_agua_1km` |
|---|---|---|
| Subestación Valparaíso (Urbana) | `subestaciones_puntos.csv` | 0,55 |
| ReTeRM · Arauco | `reterm_puntos.csv` | **0,99** |
| ReTeRM · Concón | `reterm_puntos.csv` | 0,57 |
| ReTeRM · Puerto Montt | `reterm_puntos.csv` | **1,00** |
| ReTeRM · Talcahuano | `reterm_puntos.csv` | 0,41 |

No es que falten las capas: **las capas terminan en la línea de costa y la
coordenada cae mar adentro.** Con `frac_agua_1km = 1,00`, Puerto Montt está a más
de un kilómetro de tierra en todas las direcciones. Es un defecto de los CSV de
puntos, no de las fuentes de terreno, y afecta a cualquier variable que se calcule
sobre esas coordenadas —incluido el clima que ya está bajado. **Hay que corregir
las cinco coordenadas antes de usarlas en la matriz.** No se tocaron acá: están
fuera de los entregables de este encargo.

---

## 3 · Lo que se bajó, con su URL exacta y su licencia

### 3.1 · Copernicus DEM GLO-30 — `Pend` y elevación

| | |
|---|---|
| **URL** | `https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_S19_00_W071_00_DEM/Copernicus_DSM_COG_10_S19_00_W071_00_DEM.tif` (patrón por tesela de 1°) |
| **formato** | GeoTIFF COG, float32, Deflate + **predictor 3**, bloques 1024×1024 |
| **tamaño** | ~18,9 MB por tesela de 1°; **54 teselas** cubren los 130 puntos ≈ 1 GB |
| **bajado** | **sólo los bloques necesarios** por peticiones HTTP `Range`: **221 MB en 172 archivos** |
| **licencia** | GLO-30 Public, libre para el público general bajo la licencia COP-DEM (`dataspace.copernicus.eu`, colección COP-DEM). Exige atribución. |
| **atribución obligatoria** | © DLR e.V. 2010-2014 y © Airbus Defence and Space GmbH 2014-2018, provisto bajo COPERNICUS por la Unión Europea y la ESA |
| **robots.txt** | no existe en el bucket (404) — sin exclusión declarada |
| **época del dato** | adquisición TanDEM-X 2011-2015, publicación 2021 |
| **confianza** | 0,85 |

**Se descartó SRTM/NASADEM** aunque también está accesible sin registro
(`https://s3.amazonaws.com/elevation-tiles-prod/skadi/S19/S19W071.hgt.gz`,
8,3 MB por tesela, verificado HTTP 200): es de 2000, tiene 20 años más de
antigüedad y su cobertura se corta en 60°S. Copernicus GLO-30 es mejor y estaba
al mismo precio. Queda anotado como respaldo si el bucket de Copernicus cae.

### 3.2 · ESA WorldCover 10 m — `Veg`, `Imperm`, indicio de `Defor`

| | |
|---|---|
| **URL 2021** | `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_S21W072_Map.tif` |
| **URL 2020** | `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_S21W072_Map.tif` |
| **formato** | GeoTIFF COG, uint8, Deflate sin predictor, bloques 1024×1024, teselas de **3°** (esquina SO en múltiplos de 3) |
| **tamaño** | ~19 MB por tesela de 3°; se bajaron **sólo los bloques necesarios**: **37 MB en 665 archivos** (dos años) |
| **licencia** | **CC-BY 4.0** |
| **atribución obligatoria** | ESA WorldCover project 2021 / 2020, led by VITO — doi:10.5281/zenodo.5571936 y doi:10.5281/zenodo.5571935 |
| **robots.txt** | no existe en el bucket (404) |
| **confianza** | 0,85 |

### 3.3 · HydroRIVERS v1.0 Sudamérica — cauces y `Cuenca`

| | |
|---|---|
| **URL** | `https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_sa_shp.zip` |
| **formato** | Shapefile de polilíneas dentro de ZIP; **1.620.963 tramos** |
| **tamaño** | 95.257.204 bytes (90,8 MB) — bajado **entero**, es una sola descarga |
| **licencia** | libre para uso científico, educativo y comercial; exige atribución |
| **atribución obligatoria** | Lehner, B., Grill G. (2013): Global river hydrography and network routing. HydroSHEDS / WWF |
| **robots.txt** | `data.hydrosheds.org`: sólo comentarios de content-signals, **cero directivas `Disallow`** |
| **época del dato** | v1.0 publicada 2019, base hidrológica SRTM 2000 |
| **confianza** | 0,80 |

Trae por tramo: `HYRIV_ID`, `ORD_STRA` (orden de Strahler), **`UPLAND_SKM`
(superficie de cuenca aguas arriba)**, `DIS_AV_CMS` (caudal medio), `CATCH_SKM`.

### 3.4 · IDE Chile — Hidrografía 1:25.000 (refinamiento del norte)

| | |
|---|---|
| **URL** | `https://geoportal.cl/geoserver/Hidrografia/wfs?service=WFS&version=2.0.0&request=GetFeature&typeNames=Hidrografia:hidrografa&outputFormat=application/json&bbox=<lat0>,<lon0>,<lat1>,<lon1>,urn:ogc:def:crs:EPSG::4326` |
| **formato** | WFS 2.0.0, GeoJSON nativo (también csv, SHAPE-ZIP, GML) |
| **tamaño** | 122.852 features en total; se consulta por *bbox* de cada punto, sin descargar la capa |
| **licencia** | `GetCapabilities` declara `Fees: NONE` y `AccessConstraints: NONE`; el catálogo de datos.gob.cl la lista como **CC-BY** |
| **robots.txt** | `geoportal.cl`: `User-agent: *` / `Disallow:` — **todo permitido** |
| **confianza** | 0,85 |

Trae `tipo` (Río / Estero / Quebrada), `nombre`, `nom_cuen`, `strahler_n`.
Es **mejor** que HydroRIVERS donde existe: nombra las quebradas, que HydroRIVERS
no ve. Por eso se usa como refinamiento y no se descarta.

> ⚠️ **Aviso sobre el formato CSV de este WFS:** en `outputFormat=csv` el WKT sale
> con **lat y lon invertidos** (`MULTILINESTRING ((-26.87 -68.54 …`). En
> `application/json` el orden es el correcto (lon, lat). Usar siempre GeoJSON.

**Cobertura medida sobre los 130 puntos:** devolvió cauce en **49**, todos entre
lat −18,17 y **−35,15**; los 81 restantes están al sur de eso y llevan
`nota_cauce_ide` diciendo «0 features: fuera de la cobertura publicada». De los 49,
**32 son `Quebrada`**, 12 `Rio`, 4 `Estero` y 1 `Canal` — es decir, esta capa aporta
justo lo que HydroRIVERS no ve, y lo aporta exactamente en la mitad árida del país,
donde el vector es el aluvión.

---

## 4 · Cómo se leyó GeoTIFF sin GDAL

`gdal` falló dos veces en este proyecto y costó horas. Igual que `territorio.py`
resolvió la geometría a mano, acá se resolvió el GeoTIFF a mano, con `numpy` y el
`zlib` de la biblioteca estándar. Está en `adaptadores/terreno.py`:

1. **Cabecera IFD**: se piden los primeros 64 KB con `Range` y se parsean las
   etiquetas TIFF que hacen falta (ancho, alto, compresión, predictor, tamaño de
   bloque, offsets y largos de cada bloque, escala de pixel y punto atado).
2. **Bloques por `Range`**: el COG guarda cada bloque de 1024×1024 en un rango de
   bytes conocido. Se baja **sólo el bloque que contiene el punto y su
   vecindario**, no la tesela. Eso es el «recorte de los puntos» que el encargo
   autoriza cuando la capa nacional pesa demasiado.
3. **Deflate**: `zlib.decompress`.
4. **Predictor 3 de TIFF** (coma flotante), que es la única parte no obvia. Hace
   dos cosas por fila del bloque y hay que deshacerlas en orden:
   - diferencias byte a byte → suma acumulada módulo 256
     (`np.cumsum(fila, dtype=np.uint8)`);
   - planos de bytes: la fila guarda primero todos los bytes más significativos,
     después todos los segundos, etc. → se reordena `(4, ancho)` → `(ancho, 4)`
     y se lee el resultado como `float32` **big-endian**.

**Prueba de que el lector lee bien** (dejarla, es la única red de seguridad si
alguien toca `_decodificar`): Subestación Arica (−18,478 / −70,297) da **37,3 m**
con este lector, contra **36,0 m** que devuelve la API de elevación de Open-Meteo
(que sirve Copernicus GLO-90). Es la diferencia esperable entre GLO-30 y GLO-90.

**Y la pendiente también se contrastó contra una fuente independiente**, no sólo
la elevación: se pidió a la API de elevación de Open-Meteo una rejilla de 3×3 a
±450 m de cada punto y se le aplicó el mismo Horn. Comparación de
`pendiente_grados_450m`:

| punto | nuestro lector, GLO-30 | Open-Meteo, GLO-90 |
|---|---|---|
| Subestación Arica | 0,81° | 0,82° |
| Subestación Chungará | 30,97° | 32,78° |

Coinciden. Si el predictor de coma flotante estuviera mal decodificado, no habría
manera de que estos números cayeran uno al lado del otro.

El shapefile de HydroRIVERS se parseó igual de a mano: el `.shp` es `struct`
puro y el `.dbf` es dBase III. Se recorren los 1,6 M de tramos en **una sola
pasada de 7 segundos**, descartando cada tramo por su propio *bounding box*
contra una rejilla de 1° construida con las cajas de búsqueda de los 130 puntos.
Nunca se carga la capa entera en memoria ni se extrae del ZIP.

---

## 5 · Qué se calculó para cada punto

Salida: `datos/terreno_puntos.csv`, una fila por punto, cada valor con su fuente,
su época y su confianza en columnas propias.

**Relieve** (Copernicus GLO-30, ventana de ±34 px ≈ ±1 km):

| columna | qué es |
|---|---|
| `elevacion_m` | elevación del pixel del punto |
| `pendiente_grados_30m` | pendiente por el método de **Horn** en 3×3 a paso nativo |
| `pendiente_grados_450m` | Horn sobre un estencil de ±15 px: la forma del cerro, no la del talud |
| `pendiente_max_1km_grados` | la pendiente más fuerte que hay en 1 km a la redonda |
| `relieve_1km_m` | máximo menos mínimo en 1 km |
| `orientacion_grados` | azimut de máxima bajada, 0 = norte |
| `dem_paso_lat_arcsec`, `dem_paso_lon_arcsec` | paso real del pixel en esa tesela |
| `dem_cobertura_ventana_pct` | qué fracción de la ventana traía dato |

**Cobertura** (ESA WorldCover, ventana de ±300 px = ±3 km; fracciones en ±1 km),
por duplicado para 2021 y 2020:

`clase_2021` · `clase_nombre_2021` · `frac_arbolado_1km_2021` ·
`frac_retiene_1km_2021` · `frac_desnudo_1km_2021` · `frac_construido_1km_2021` ·
`frac_agua_1km_2021` · `frac_nieve_1km_2021` · `dist_agua_permanente_m_2021` ·
`cobertura_ventana_pct_2021` — e idénticas con sufijo `_2020`.

`frac_retiene_1km` agrupa las clases que retienen suelo (arbolado, matorral,
pradera, cultivo, humedal, musgo) y excluye las que no (nieve, roca desnuda,
agua, ciudad). La agrupación está escrita a la vista en `CLASES_COBERTURA` y
`GRUPOS_QUE_RETIENEN`, no escondida en una fórmula.

**Cauces y cuenca**:

`dist_cauce_m` · `cauce_orden_strahler` · `cuenca_km2` · `caudal_medio_m3s` ·
`cauce_id` (HydroRIVERS, 130/130) y `dist_cauce_ide_m` · `cauce_ide_tipo` ·
`cauce_ide_nombre` · `cauce_ide_cuenca` · `cauce_ide_strahler` (IDE Chile, sólo
al norte de −35).

**`Reg` regolito** (Mapa Geológico 1:1.000.000):

`regolito` (uno de `suelto` / `roca` / `mixto` / `indeterminado`) ·
`regolito_rocas` (las litologías que nombra el mapa, tal cual) ·
`geologia_codigo` · `geologia_unidad` · `geologia_ambiente` ·
`geologia_edad_desde` · `geologia_edad_hasta`.

**Uso de suelo oficial chileno y `Defor`** (catastro CONAF, 1:50.000):

`uso_suelo` · `uso_suelo_subuso` · `uso_suelo_detalle` · `uso_suelo_especie` ·
`fuente_uso_suelo` (**dice de cuál de los dos servicios salió el valor**) ·
`epoca_uso_suelo` · `uso_2001` · `uso_2021_o_ultimo` · `transicion_ultima`.

**Contraste de sanidad de los dos cauces, con los valores reales del CSV.** No es
una validación formal: es comprobar que no estamos midiendo cualquier cosa, y de
paso el contraste destapa el límite de §6.3 mejor que cualquier explicación.

*Subestación Arica* (−18,478 / −70,297): HydroRIVERS da **637,7 m** a un cauce de
**orden 5** con `cuenca_km2 = 3.276,6`. Es el Lluta, cuya cuenca son ~3.400 km².
El número cae donde debe.

*Subestación Copiapó* (−27,3793 / −70,3363), el ancla que reventó el 15-16 de
agosto: HydroRIVERS da **209,2 m** pero a un cauce de **orden 2** con
`cuenca_km2 = 44,0`; el IDE chileno, en el mismo punto, da **665,6 m** al **«Rio
Copiapo»**. Los dos aciertan y dicen cosas distintas: **el cauce más cercano es un
tributario de 44 km², no el río principal.** Ésa es exactamente la razón por la
que `cuenca_km2` se declara como «la cuenca del tramo más cercano» y no como «la
cuenca del punto». Quien use `cuenca_km2` para la receta de desborde fluvial tiene
que mirar también `cauce_orden_strahler`: un orden 2 no desborda como un orden 5.

---

## 6 · Límites declarados — no borrar

1. **El DEM es un DSM, no un DTM.** Copernicus DEM representa la *superficie*:
   incluye edificios y copas de árboles. En bosque valdiviano denso o en el
   centro de una ciudad, la pendiente de 30 m puede venir del dosel y no del
   suelo. Por eso la confianza es 0,85 y no más, y por eso se entrega también la
   pendiente a 450 m, que promedia el efecto.
2. **La pendiente depende de la escala a la que se mide.** No hay una «pendiente
   verdadera» del punto. Hay una pendiente a 30 m y otra a 450 m, y las dos van
   en el CSV con su escala en el nombre de la columna.
3. **`cuenca_km2` es la cuenca del TRAMO más cercano, no la del punto.** Es el
   `UPLAND_SKM` que HydroRIVERS calculó para ese tramo de río. Sirve como orden
   de magnitud del área que drena hacia el sector del punto; **no** es el área
   que drena exactamente hacia el punto. Calcular eso exige acumulación de flujo
   sobre la cuenca completa y no se hace desde una ventana local.
4. **HydroRIVERS omite las quebradas secas.** Su umbral son tramos con cuenca
   ≥ 10 km² o caudal ≥ 0,1 m³/s. Justo las quebradas del norte árido que
   canalizan los aluviones quedan fuera. Para el norte hay que mirar
   `cauce_ide_tipo`, que sí distingue «Quebrada».
5. **La capa chilena se apaga en el Maule.** El *bounding box* declarado del WFS
   llega a lat −34,99 y se comprobó **0 features** en Chillán (−36,6), Valdivia
   (−39,8) y Puerto Montt (−41,5). Los puntos al sur de −35 llevan
   `nota_cauce_ide` explicando el hueco, no un valor vacío sin explicación.
6. **`dist_agua_permanente_m` incluye el mar.** La clase 80 de WorldCover es
   «cuerpo de agua permanente» y no separa el océano de un río. En los puntos
   costeros esa distancia puede ser la distancia al mar. Para río, usar
   `dist_cauce_m`.
7. **`delta_arbolado_1km_pp` NO mide deforestación.** WorldCover v100 (2020) y
   v200 (2021) usan algoritmos distintos y la propia ESA advierte que no deben
   compararse para detectar cambio. Se entrega con confianza **0,40** y la
   advertencia pegada en la propia fila del CSV. **Para `Defor` usar las columnas
   de CIREN** (`uso_2001`, `uso_2021_o_ultimo`, `transicion_ultima`), que traen
   la transición de uso ya declarada por CONAF con fecha (§7-bis.2).
8. **Un año de cobertura contra 60 de clima.** El clima del proyecto va de 1990 a
   2026; WorldCover es 2020-2021 y el DEM es 2011-2015. Las condiciones de
   terreno se están tratando como estáticas. Para `Pend` y `Cuenca` eso es
   razonable (cambian en siglos). Para `Veg` **no lo es**: la cobertura de 2021
   no describe el bosque que había en 1990, y cualquier receta que cruce lluvia
   de los 90 con vegetación de 2021 está mezclando épocas. Hay que declararlo en
   cada uso.

---

## 7 · Lo que NO se consiguió, y por qué

### 7.1 · `Ribera` — ocupación de riberas

No se intentó, y no por olvido: `Ribera` no es una capa que se baje, es un
**cruce** entre la red de cauces (ya la tenemos) y la edificación. Con lo que hay
ahora se puede construir: `frac_construido_1km` de WorldCover intersectado con
una franja de N metros alrededor de las polilíneas de HydroRIVERS / IDE. Es
trabajo de cálculo, no de catastro, y cae fuera de este encargo. **Queda
declarado como el siguiente paso barato**, porque los dos insumos ya están
bajados.

### 7.2 · `HumSuelo`

Fuera de este encargo. Ya hay un frente propio (ESA CCI vía OPeNDAP del CEDA, y
ERA5-Land) y está registrado en el estudio de vectores §5.

### 7.3 · Lo que se verificó y NO sirve — con el error exacto

| endpoint probado | resultado |
|---|---|
| `sit.conaf.cl/arcgis/rest/services?f=json` | **HTTP 404** (Apache) |
| `sit.conaf.cl/server/rest/services?f=json` | **HTTP 404** |
| `sit.conaf.cl/geoserver/wfs?…GetCapabilities` | **HTTP 404** |
| `portalgeo.sernageomin.cl` | `connect to 190.98.205.167 port 443 failed: Network is unreachable` |
| `geoarcgis.sernageomin.cl/ArcGIS/rest/services/geoportal/GeologiaBase/MapServer` | **sin conexión** (190.98.205.187) — es el endpoint oficial de geología |
| `geoportal.sernageomin.cl` | ídem, mismo /24 |
| `portalgeomin.sernageomin.cl:6080`, `tienda.sernageomin.cl` | **NXDOMAIN** |
| `services1.arcgis.com/OyjvVdFTl5hfSdX3` (AGOL de SERNAGEOMIN, el que este proyecto ya usa) | vivo, **206 FeatureServers**, pero ninguno es el mapa geológico nacional: el único poligonal, `Geología_10k`, cubre un sector de la RM |
| `esri.ciren.cl/…/WMSServer?…GetCapabilities` | **HTTP 500** «ArcGIS Web Adaptor – Application Error» |
| `www.geoportal.cl/ArcGIS/rest/services/…Red_hidrografica/MapServer` | **301 → 404**: las URLs ArcGIS de datos.gob.cl están obsoletas; ya no existe ningún ArcGIS REST en geoportal.cl |
| `ide.mma.gob.cl/geoserver/wfs` | HTTP 200 pero **19 feature types, ninguno hidrográfico** |
| `ide.minagri.gob.cl/geoserver/wfs` | **HTTP 404** — es un WordPress, no hay GeoServer |
| `snia.mop.gob.cl/arcgis/rest/services` | **HTTP 403** |
| `ide.igm.cl/geoserver/wfs` | **HTTP 403** |
| `arcgis.mop.gob.cl`, `geoportal.mop.gob.cl` | **no resuelven DNS** |
| CKAN datos.gob.cl con `q=cauces`, `q=cuencas`, `q=hidrologia` | **count: 0** en los tres |
| `overpass-api.de/api/interpreter` (OSM, alternativa de cauces) | funciona **sólo por POST** (GET da 406); **HTTP 504 en ~2 de cada 3** consultas. Descartado como fuente primaria por inestable |
| `geoportal.cl/geoportal/catalog/download/e109bc13-…` (`Hidrografia_V2.zip`, 134.451.894 bytes) | descarga **entera o nada**: el servidor **ignora la cabecera `Range`**, así que no se puede inspeccionar la cobertura sin bajar los 128 MB. No verificado. |

**Nota sobre robots.txt.** `www.ide.cl` (el sitio CMS) trae `User-agent: ClaudeBot` / `Disallow: /` y `Content-Signal: ai-train=no`; se respetó y no se usó ese host. El host de datos, `geoportal.cl`, permite todo. `sit.conaf.cl` es el único portal del lote con robots.txt restrictivo (`Disallow: all`, malformado pero de intención restrictiva) — y da igual, porque ahí no hay API que consumir. `datos.gob.cl` pide `Crawl-Delay: 10` y prohíbe `/api/`.

---

## 7-bis · `Reg` y el catastro chileno: se consiguieron, por rutas distintas a las esperadas

### 7-bis.1 · Geología — el servidor oficial está caído; se usa una re-subida, y se dice

| | |
|---|---|
| **URL** | `https://services.arcgis.com/vQbRu5CNjGqIxk8R/arcgis/rest/services/geologia_chile_f/FeatureServer/0/query` |
| **formato** | ArcGIS REST, `f=json` y **`f=geojson`**, polígonos |
| **esquema** | el del Mapa Geológico de Chile **1:1.000.000** de SERNAGEOMIN: `CD_GEOL`, `EDAD_DESDE/HASTA`, `ROCA1..4`, `AMBIENTE`, `TIPO`, `NOMBRE_DE_`, `RESUMEN` |
| **licencia** | ⚠ **EL SERVICIO NO DECLARA NINGUNA** (`copyrightText` vacío). Las capas de SERNAGEOMIN registradas en datos.gob.cl declaran **Creative Commons NO COMERCIAL** |
| **confianza** | **0,55** |

> ⚠️ **PROCEDENCIA FRÁGIL, DECLARADA.** Este servicio es una **re-subida a
> ArcGIS Online hecha por particulares** (creada 2025-2026), no la fuente
> oficial. Se usa porque es lo único vivo: **todo el rango legacy
> `190.98.205.0/24` de SERNAGEOMIN está inalcanzable a nivel de red** —
> comprobado con y sin sandbox, con `traceroute`: los paquetes salen bien, el
> host no responde en 80/443. Se guarda copia local del crudo mientras exista.
> **Hay que reemplazarla por la oficial cuando el servidor vuelva**, y el
> instrumento no debe presentar `Reg` como dato oficial mientras venga de acá.

**Cómo se traduce litología a `Reg`.** El vocabulario pide «regolito / material
suelto disponible»; el mapa da litologías. La traducción se hace por palabra
clave sobre `ROCA1..ROCA4` (donde el mapa nombra el material) y da cuatro
valores: `suelto`, `roca`, `mixto` e **`indeterminado`** — que se usa cuando el
mapa no nombra una roca clasificable, en vez de adivinar. Las dos listas de
palabras (`ROCAS_SUELTAS`, `ROCAS_CONSOLIDADAS`) están escritas a la vista en
`terreno.py`, no dentro de una fórmula.

**Y da justo lo que el proyecto necesitaba en el punto que le falló.** Copiapó
—el ancla que reventó el 15-16 de agosto— sale `CD_GEOL=Qf`, «Depósitos
fluviales», `grava; arena; limo`, ambiente sedimentario continental →
`regolito = suelto`. Es exactamente la condición `Reg` alta que pide la receta de
aluvión de §4.2, y es dato, no interpretación. Angol sale `CPg`, granitos y
granodioritas plutónicas de 328-235 Ma → `regolito = roca`.

**Límite de escala, sin adornos:** 1:1.000.000. Un polígono puede medir decenas
de km². Sirve para distinguir roca de material suelto; **no** para caracterizar
el sitio de una subestación. La carta 1:100.000 sigue faltando.

**Reparto medido sobre los 130 puntos:** `roca` 51 · `suelto` 43 · `mixto` 8 ·
`indeterminado` 24 · sin polígono 4. Los 24 `indeterminado` son polígonos donde el
mapa nombra litologías que las dos listas de palabras no clasifican; van así, no
repartidos a ojo. Y hay **12 puntos con clase de regolito pero sin
`geologia_unidad`**: el polígono existe y nombra la roca, pero su campo
`NOMBRE_DE_` viene nulo en la fuente.

### 7-bis.2 · Catastro CONAF — no está en `sit.conaf.cl`, está republicado dos veces

`sit.conaf.cl` no expone servicio alguno (§7.3). El catastro se consigue por dos
organismos que lo republican, y los dos sirven:

| | **SMA** | **CIREN (IDE Minagri)** |
|---|---|---|
| **URL** | `https://ideserver.sma.gob.cl/arcgis/rest/services/IDE/Biodiversidad/MapServer/<capa>/query` | `https://esri.ciren.cl/server/rest/services/IDEMINAGRI/CAMBIO_DEL_USO_DE_LA_TIERRA_CONAF/MapServer/<capa>/query` |
| **qué trae** | el registro catastral completo: uso, subuso, tipo forestal, especie, cobertura, altura | **SEIS fechas por polígono** (2001, 2013, 2016, 2017, 2019, 2021) **y las transiciones ya calculadas** (`d_tc_01_13` … `d_tc_19_21`), más clasificación IPCC |
| **estructura** | 16 capas regionales (IDs 6-21) | 20 capas; Los Lagos y Magallanes van **por provincia** |
| **escala** | 1:50.000 | 1:50.000 |
| **formato** | ArcGIS REST `f=json` (**no** soporta `f=geojson`) | ArcGIS REST `f=json` |
| **robots.txt** | 404, sin restricciones | 404, sin restricciones |
| **licencia** | `copyrightText`: «Corporación Nacional Forestal». CONAF no publica términos de uso; la categoría equivalente en el catálogo IDE Chile figura como **CC-BY** | catastro CONAF republicado por la IDE del Minagri; sin licencia explícita en el servicio |
| **confianza** | 0,80 | 0,80 |

**CIREN resuelve `Defor` de verdad.** La deforestación no hay que derivarla de un
NDVI ni de la diferencia entre dos WorldCover incomparables: viene declarada, por
polígono, como uso con fecha. Eso convierte `delta_arbolado_1km_pp` (§6.7) en lo
que siempre fue: un indicio de respaldo.

**Y ya trae señal medida, no potencial:** de los **115 puntos** con serie CIREN,
**20 cambiaron de uso** entre la primera y la última fecha. Algunos ejemplos, tal
como los declara el catastro:

| punto | primera fecha | última fecha |
|---|---|---|
| ReTeRM · Castro | Praderas y Matorrales, Pradera Perenne | **Áreas Urbanas e Industriales** |
| ReTeRM · Hualqui | Terrenos Agrícolas, Rotación Cultivo-Pradera | **Áreas Urbanas e Industriales** |
| ReTeRM · Corral | Bosques, **Bosque Nativo, Renoval** | Bosques, **Plantación, Adulta** |
| ReTeRM · Mariquina | Praderas y Matorrales, Pradera Perenne | Bosques, Plantación, Adulta |
| ReTeRM · Pelluhue | Bosques, Bosque Mixto Nativo con Exóticas | Praderas y Matorrales, Matorral |

Dos de esos casos son **urbanización de la cuenca** y uno es **sustitución de
bosque nativo por plantación** — las dos intervenciones humanas que el estudio
nombra como causa del desborde, en puntos que además tienen eventos ReTeRM
registrados.

> ⚠️ **No usar `transicion_ultima` como si dijera «cambió / no cambió».** Ese
> campo (`d_tc_*`) sólo toma dos valores en los 130 puntos: «Otros Usos
> Permanentes» (95) y «Bosque Continuo» (20); es una categoría de CONAF, no una
> bandera de cambio. **El cambio se lee comparando `uso_2001` con
> `uso_2021_o_ultimo`.**

> ⚠️ **Trampa 1 — la mitad de las capas de la SMA no responde a consulta
> espacial.** Medido sobre los 130 puntos: devolvieron **0 features** la capa 13
> (Biobío) en 20 puntos, la 14 (Araucanía) en 13, la 10 (Valparaíso) en 12, la 12
> (Maule) en 11, la 9 (Coquimbo) en 9, la 8 (Atacama) en 4, la 6 (Tarapacá) en 2,
> la 11 (O'Higgins) en 2, la 21 (Ñuble) en 2 y la 15 (Los Lagos) en 1 —
> **es decir, en TODOS los puntos que les tocaban, salvo Los Lagos.** Se probó
> punto, envelope y polígono, a varias escalas, con `outFields=*`: siempre 0,
> aunque la capa declare 196.130 registros y un extent que contiene el punto. Las
> capas 7 (Antofagasta), 16 (Aysén), 17 (Magallanes), 18 (RM), 19 (Los Ríos) y 20
> (Arica) responden bien, lo que descarta proyección y tolerancia: **las otras
> están mal indexadas del lado del servidor.** En esos casos el adaptador cae a
> CIREN y **escribe en `fuente_uso_suelo` de qué servicio salió cada valor** más
> el error exacto en `nota_uso_suelo`. Resultado: **54 puntos por la SMA, 71 por
> CIREN, 5 sin dato** (los cinco que están en el mar).

> ⚠️ **Trampa 2 — pedir un campo inexistente devuelve `features: []` EN SILENCIO,
> no un error.** Los nombres de campo cambian por región: `USO_TIERRA` vs
> `USO_TIERR`, `NOM_COM` vs `COMUNA`, `CODCOM` vs `CODCOMUN`, y Aysén no tiene
> `USO` ni `SUBUSO`. Por eso el adaptador pide siempre `outFields=*` y normaliza
> después.

> ⚠️ **Trampa 3 — CIREN es intermitente.** Alterna HTTP 500 y 200 sin patrón. No
> se consulta sin reintentos con espera creciente.

> ⚠️ **Trampa 4 — la foto del catastro tiene fecha distinta en cada región.**
> Aysén 2011 · O'Higgins 2013 · Coquimbo 2014 · Arica 2015 · Biobío 2015 · Ñuble
> 2015 · Tarapacá 2016 · Maule 2016 · Araucanía 2017 · Los Ríos 2017 · Atacama
> 2018 · Los Lagos 2018 · Antofagasta 2019 · Valparaíso 2019 · Magallanes 2019 ·
> RM 2019. **Ocho años de desfase** entre Aysén y las de 2019, dentro del mismo
> CSV. Va declarado en `epoca_uso_suelo`.

> ⚠️ **CIREN no tiene capa de cambio de uso para Antofagasta ni para Aysén.** Los
> puntos de esas dos regiones llevan `nota_cambio_uso` diciéndolo.

---

## 8 · Crudo guardado

```
datos/crudo/terreno/
├── copernicus_glo30/2026-08-17/
│   ├── Copernicus_DSM_COG_10_S<lat>_00_W<lon>_00_DEM.cabecera.bin
│   ├── Copernicus_DSM_COG_10_S<lat>_00_W<lon>_00_DEM.bloque<nnnn>.deflate
│   └── MANIFIESTO.jsonl        ← URL + rango de bytes + hora UTC de cada bloque
├── esa_worldcover/2026-08-17/
│   ├── ESA_WorldCover_10m_2021_v200_S<lat>W<lon>_Map.cabecera.bin
│   ├── ESA_WorldCover_10m_2021_v200_S<lat>W<lon>_Map.bloque<nnnn>.deflate
│   ├── (ídem 2020 / v100)
│   └── MANIFIESTO.jsonl
├── hydrorivers/2026-08-17/
│   └── HydroRIVERS_v10_sa_shp.zip     (95.257.204 bytes, sin tocar)
├── geologia_1m_agol/2026-08-17/
│   └── <punto>.json                   ← respuesta ArcGIS cruda, una por punto
├── catastro_conaf_sma/2026-08-17/
│   └── <punto>__capa<n>.json          ← incluidas las que devolvieron 0 features:
│                                        el hueco también es evidencia
└── cambio_uso_conaf_ciren/2026-08-17/
    └── <punto>__capa<n>.json
```

**352 MB en total**: Copernicus 221 MB (172 archivos) · WorldCover 37 MB (665
archivos, dos años) · HydroRIVERS 91 MB (1 archivo) · CIREN 1,4 MB (145) · SMA
1,0 MB (129) · geología 516 KB (129).

Los bloques se guardan **comprimidos tal como vinieron del servidor**, antes de
decodificarlos. El `MANIFIESTO.jsonl` anota por cada archivo la URL exacta, el
rango de bytes pedido y la hora UTC: con eso cualquiera reconstruye la descarga
sin volver a pedirla, y se puede comprobar que el bloque no fue alterado.

El crudo del WFS de IDE Chile **no se guarda como archivo**: es una consulta por
*bbox* de cada punto y la respuesta entra directa. La consulta exacta queda
escrita en `FUENTE_CAUCES_IDE["url"]` y en `cauces_ide()`, que es lo que permite
repetirla.

---

## 9 · Qué falta para cerrar el vocabulario de condiciones

En orden de lo que más cambiaría el instrumento:

1. **`Ribera`** — la única condición del vocabulario que sigue en cero y que
   **ya es calculable con lo bajado**: cruzar `frac_construido` de WorldCover con
   una franja de N metros alrededor de las polilíneas de HydroRIVERS / IDE. Es el
   insumo que Alexis nombró como causa del desborde y una de las tres condiciones
   reversibles. Es trabajo de cálculo, no de catastro.
2. **Reemplazar la geología por la fuente oficial** — hoy `Reg` viene de una
   re-subida de terceros sin licencia declarada, con confianza 0,55. Hay que
   volver a `geoarcgis.sernageomin.cl` cuando el `/24` de SERNAGEOMIN vuelva a
   responder, y pasar a la carta 1:100.000 donde exista: 1:1.000.000 no
   caracteriza el sitio de una subestación.
3. **`Veg` con historia** — la cobertura de 2021 no describe 1990, y el clima del
   proyecto arranca en 1990. CIREN ya da seis fechas desde 2001, lo que cubre 20
   de los 36 años; para los 90 hace falta NDVI MODIS (2000-hoy, hay experiencia
   previa en Cosmoclima) o el catastro base 1993-1997.
4. **`Cuenca` exacta del punto** — hoy es el `UPLAND_SKM` del tramo más cercano
   (§6.3). La cifra exacta pide acumulación de flujo. Con el lector de COG ya
   escrito el impedimento no es el formato: hay que bajar la cuenca entera aguas
   arriba de cada punto, no una ventana.
5. **Cauces del norte árido** — HydroRIVERS omite las quebradas secas y la capa
   chilena que sí las nombra se apaga en el Maule. Para el norte, donde el vector
   es el aluvión, la red de quebradas hay que conseguirla de otra parte o
   derivarla del DEM.
6. **Un DTM donde importe** — para los puntos en bosque denso, la pendiente del
   DSM está contaminada por el dosel. No hay DTM global gratuito de 30 m; sí hay
   productos derivados (FABDEM, con licencia no comercial) que habría que
   revisar antes de usar.
7. **Homogeneizar la fecha del catastro** — ocho años de desfase entre regiones
   dentro del mismo CSV (§7-bis.2, trampa 4). No es un defecto que se arregle
   bajando más dato: es una propiedad de la fuente que hay que arrastrar
   declarada en cada uso.

---

## 10 · Cómo se repite

```bash
V=/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/.venv-esa/bin/python

# 1) relieve + cobertura (Copernicus GLO-30 y ESA WorldCover).
#    ~1 h 40 min la primera vez; después reusa el crudo y es casi instantáneo.
$V adaptadores/terreno.py

# 2) cauces y `Cuenca` (HydroRIVERS + WFS de IDE Chile). ~10 min.
$V adaptadores/terreno.py cauces
$V adaptadores/terreno.py cauces-sin-ide      # si el WFS chileno se cae

# 3) `Reg` y catastro CONAF (geología AGOL + SMA + CIREN). ~25 min.
$V adaptadores/terreno.py suelo
```

Las tres pasadas van **por separado a propósito**: son fuentes distintas con
licencias, fallos y ritmos distintos, y si una se cae no puede llevarse por
delante lo que las otras ya midieron. Las pasadas 2 y 3 leen y reescriben
`datos/terreno_puntos.csv`, así que hay que correrlas en ese orden después de la 1.

Todo lo ya bajado queda en `datos/crudo/terreno/` y **se reutiliza**: volver a
correr no vuelve a pedir nada al servidor. Para forzar una descarga nueva, borrar
la carpeta de fecha correspondiente — y guardar la anterior, que es la historia.

**Atribuciones obligatorias, juntas, para copiar donde se publique un resultado:**

> Copernicus DEM GLO-30: © DLR e.V. 2010-2014 y © Airbus Defence and Space GmbH
> 2014-2018, provisto bajo COPERNICUS por la Unión Europea y la ESA · ESA
> WorldCover 2021/2020, led by VITO (CC-BY 4.0), doi:10.5281/zenodo.5571936 y
> doi:10.5281/zenodo.5571935 · HydroRIVERS v1.0: Lehner, B., Grill G. (2013),
> HydroSHEDS / WWF · Hidrografía 1:25.000: IDE Chile — Ministerio de Bienes
> Nacionales · Catastro de Recursos Vegetacionales: CONAF, vía Superintendencia
> del Medio Ambiente y vía CIREN / IDE Minagri · Mapa Geológico de Chile
> 1:1.000.000: SERNAGEOMIN (⚠ obtenido de una re-subida de terceros, ver
> §7-bis.1).
