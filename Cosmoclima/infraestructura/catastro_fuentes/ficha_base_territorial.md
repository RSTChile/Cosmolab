# Catastro de fuentes — BASE TERRITORIAL Y ADMINISTRATIVA

Proyecto «Infraestructura Crítica × Clima» (RMD 2.0) · Chile
Levantamiento del 15-ago-2026. Todo lo marcado **VERIFICADO** se probó con petición
HTTP real ese día (se anota código, tipo de contenido y tamaño). Lo demás va como
**SUPUESTO** y no debe usarse hasta comprobarlo.

**Nada se descargó completo.** Se comprobó que las URLs responden y se leyó el
encabezado que declara el peso. La descarga la decide Alexis.

---

## Resumen ejecutivo: las dos respuestas que desbloquean el proyecto

**(a) Capa de límites comunales — RESUELTA.** Hay tres caminos verificados. El
recomendado es el **servicio del INE con el Censo 2024**, que trae en la misma
capa la geometría de las 346 comunas, el código CUT, la provincia, la región **y
la población censada**. Resuelve de un saque los puntos 1, 3 y 4 del encargo.

**(b) Capa de zonas geográficas DMC/SERNAGEOMIN — EXISTE, y pesa 452 KB.**
No hay bloqueo. Está en
`https://services5.arcgis.com/i7S5PSnIJAUcWvSE/arcgis/rest/services/coordenadas_zonas_DMC/FeatureServer/0`
(181 polígonos, GeoJSON de 463 KB en una sola petición). **Pero** no la publica la
DMC: es una republicación de ONEMI/SENAPRED en la nube, **sin licencia declarada**,
y **le falta el corte de 2026** que separó `02_Precordillera` en
`02_Precordillera_Occidental` y `02_Precordillera_Alto_Loa` —justo las dos que
aparecen en el ejemplo de Antofagasta. Ese parche hay que hacerlo a mano. Detalle
completo en la sección 2.

**Dos cosas urgentes que salieron de paso y no admiten espera:**
1. **Empezar hoy a guardar una foto diaria** de las capas vivas (minuta de
   SERNAGEOMIN, avisos DMC, alertas SENAPRED). **Ninguna tiene archivo histórico**:
   lo que vence, desaparece. Cada día sin capturar es evidencia que no se recupera.
2. **Pedir por escrito** a DMC y SENAPRED la capa oficial y sus condiciones de uso.
   Casi todo lo encontrado está publicado sin licencia y sin dueño declarado, lo
   que es un problema para un insumo de decisión pública.

Analogía para ubicarse: las comunas son como las **cajas de la colección** —
etiquetadas, numeradas, estables, y todo el mundo usa las mismas. Las franjas
geográficas (litoral, pampa, precordillera) son como los **pisos altitudinales**
donde vive un bicho: reales, todos los describen, pero el mapa con el contorno
exacto de cada uno estaba traspapelado en el cajón de otra oficina. Apareció.

## Las cinco piezas que hay que bajar (todas verificadas, menos de 30 MB en total)

| # | Qué es | URL | Peso |
|---|---|---|---|
| 1 | **Comunas + CUT + población Censo 2024** (geodatabase) | `https://www.arcgis.com/sharing/rest/content/items/d083bf61913d4c87842c2f4a30e9855d/data` | 6,6 MB |
| 2 | **Zonas geográficas DMC** (181 polígonos, GeoJSON) | `https://services5.arcgis.com/i7S5PSnIJAUcWvSE/arcgis/rest/services/coordenadas_zonas_DMC/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson` | 452 KB |
| 3 | **Planilla oficial del CUT** (SUBDERE) | `https://www.subdere.gov.cl/sites/default/files/documentos/CUT_2018_v04.xls` | 68 KB |
| 4 | **Censo 2024 por comuna en CSV** | `https://hub.arcgis.com/api/download/v1/items/d93e7edbe70f4eecb42c457221c7a73c/csv?layers=0` | 95 KB |
| 5 | **Aislamiento y tiempos a servicios** (ILCA 2021, SUBDERE) | `https://ide.subdere.gov.cl/wp-content/uploads/2023/11/LAC_2021_CENSO2017.csv` | 14 MB |

Las cinco empalman por **CUT**, y **coinciden en 346 comunas** —Censo 2024,
límites DPA censal 2025 y planilla SUBDERE 2018—: tres fuentes independientes que
cuadran, que es confianza medida y no declarada.

Si además se quiere el límite con validez legal (1:50.000, con la autorización de
DIFROL), está la DPA 2023 del Geoportal: 297 MB, ficha 1.2.

---

# 1 · LÍMITES COMUNALES, PROVINCIALES Y REGIONALES

## FICHA 1.1 — ★★★ INE · Censo 2024, indicadores por comuna (RECOMENDADA)

1. **Organismo y producto** — Instituto Nacional de Estadísticas (INE), cuenta
   institucional de geodatos. Servicio «Indicadores Censales por Comuna, Región y
   País» (`CENSO2024_V2_gdb`).
2. **URL exacta de descarga**
   - Archivo completo (File Geodatabase comprimida):
     `https://www.arcgis.com/sharing/rest/content/items/d083bf61913d4c87842c2f4a30e9855d/data`
     → entrega `CENSO2024_V2.gdb.zip`
   - Servicio consultable en vivo (no hace falta descargar nada):
     `https://services5.arcgis.com/hUyD8u3TeZLKPe4T/arcgis/rest/services/CENSO2024_V2_gdb/FeatureServer`
     · capa 0 = COMUNA · capa 1 = REGIÓN · capa 2 = PAÍS
   - Ficha del servicio: `https://www.arcgis.com/home/item.html?id=d93e7edbe70f4eecb42c457221c7a73c`
3. **Qué contiene** — 346 polígonos comunales (el total del país) con 38 campos:
   `CUT`, `REGION`, `PROVINCIA`, `COMUNA`, y 28 indicadores del Censo 2024
   (población total, hombres, mujeres, tramos 0-14 / 15-64 / 65+, índice de
   envejecimiento, viviendas, hogares…).
4. **Familia** — BASE TERRITORIAL (y además ESTADO-CONSECUENCIA, porque trae la
   población que convierte «se cortó la luz» en «40.000 personas sin luz»).
5. **Formato y tamaño** — File Geodatabase ESRI en ZIP, **6.619.849 bytes
   (≈ 6,6 MB)**. Alternativa: GeoJSON servido por consulta; el país entero con
   geometría completa pesa ≈ 10 MB, y generalizado (tolerancia 0,005°) ≈ 1,3 MB.
6. **Acceso** — anónimo, sin registro. **VERIFICADO**:
   - `HTTP 200`, `content-type: application/zip`, `content-length: 6619849`,
     `content-disposition: attachment; filename="CENSO2024_V2.gdb.zip"`.
   - Consulta por punto probada: coordenada `-68,9333 / -22,4667` devuelve
     `CUT 2201 · CALAMA · provincia EL LOA · REGIÓN DE ANTOFAGASTA ·
     población 166.334 · viviendas 60.835`. **El «dada una lat/lon, ¿en qué
     comuna estoy?» funciona hoy.**
   - Consulta por región probada: Antofagasta devuelve `2101 ANTOFAGASTA 401.096`,
     `2102 MEJILLONES 14.084`, `2103 SIERRA GORDA 1.472`, `2104 TALTAL 12.097`.
7. **Granularidad** — comuna (346), región y país. Sin nivel provincia como capa
   propia, pero el campo `PROVINCIA` está en cada comuna, así que las provincias
   se arman uniendo comunas.
8. **Vigencia** — Censo 2024 (levantamiento 2024, difusión 2025). Ítem del
   servicio modificado el 25-mar-2025; la geodatabase, el 04-dic-2025. Es la
   división vigente al censo. ⚠️ Ojo: las comunas cambian por ley (la última
   creada es Alto Hospicio-era; hoy son 346). Hay que reverificar el conteo cada
   vez que se actualice.
9. **Historia disponible** — no. Es una foto del Censo 2024. Para serie histórica
   está el Censo 2017 (ficha 1.5) y las proyecciones INE.
10. **Licencia** — ⚠️ **el ítem NO declara licencia** (`licenseInfo` vacío). Es
    publicación oficial del INE en su cuenta institucional y de acceso público
    sin registro, pero **no hay texto de licencia que citar**. Antes de publicar
    productos derivados hay que pedirle al INE la condición de uso por escrito, o
    apoyarse en la ficha 1.2, que sí trae condiciones explícitas. Atribución
    mínima sugerida: «Instituto Nacional de Estadísticas de Chile, Censo 2024».
11. **Para qué sirve en el modelo** — es el **eje del cruce**. Cada activo (las 39
    subestaciones, por ejemplo) se ubica en su comuna con una sola consulta; la
    comuna trae CUT estable y población, y con eso la salida ya se puede expresar
    en el idioma en que SENAPRED alerta.
12. **Utilidad** — ★★★
13. **Riesgos y límites**
    - **Depende de un servicio en la nube de ESRI.** Si se cae o el INE lo
      despublica, el instrumento se queda ciego. Mitigación obligatoria: bajar la
      geodatabase de 6,6 MB una vez y guardarla con fecha, como manda la regla de
      «el dato crudo se guarda tal como llegó».
    - Licencia no declarada (ver punto 10).
    - Nombres de comuna en MAYÚSCULA y sin tildes normalizados a su manera:
      cruzar **siempre por CUT**, nunca por nombre.
    - Escala de dibujo no declarada; para límites finos (activo justo en el borde
      entre dos comunas) conviene contrastar con la DPA 2023 de la ficha 1.2.

## FICHA 1.2 — ★★★ SUBDERE / IDE Chile · División Política Administrativa 2023 (la OFICIAL de límites)

1. **Organismo y producto** — SUBDERE, con IGM, DIFROL e INE, en el Grupo de
   Trabajo DPA de la IDE Chile. Producto «División Política Administrativa 2023»,
   publicado en el Geoportal de Chile.
2. **URL exacta de descarga**
   `https://geoportal.cl/geoportal/catalog/download/912598ad-ac92-35f6-8045-098f214bd9c2`
   · Ficha: `https://geoportal.cl/geoportal/catalog/36391/Divisi%C3%B3n%20Pol%C3%ADtica%20Administrativa%202023`
   · Entrada en el catálogo nacional: `https://datos.gob.cl/dataset?tags=DPA`
3. **Qué contiene** — tres capas de polígonos separadas: **comunas, provincias y
   regiones** (excluye el Territorio Antártico). Sistema de referencia SIRGAS
   Chile, escala 1:50.000.
4. **Familia** — BASE TERRITORIAL.
5. **Formato y tamaño** — Shapefile ESRI en ZIP (`DPA_2023.zip`),
   **310.992.026 bytes ≈ 297 MB**. Es pesado porque el detalle es de 1:50.000.
6. **Acceso** — anónimo, sin registro. **VERIFICADO**: `HTTP 200`,
   `content-type: application/zip`, `content-length: 310992026`,
   `content-disposition: attachment; filename=DPA_2023.zip`; los primeros bytes
   son la firma `PK` de un ZIP legítimo.
7. **Granularidad** — comuna, provincia, región. Es la más fina de las tres capas
   de límites que hay.
8. **Vigencia** — actualizada al **29-sep-2023**. Es la versión más reciente
   publicada como polígonos. Existe una versión 2026 pero en geometría de
   **líneas** y publicada por un tercero (ficha 1.4).
9. **Historia disponible** — sí, hay versiones previas (DPA 2018, DPA 2020),
   aunque los enlaces antiguos de `ide.cl` están rotos (ver riesgos).
10. **Licencia** — el catálogo nacional `datos.gob.cl` la clasifica como
    **Creative Commons Attribution (CC BY)**. **Condición obligatoria adicional**:
    todo producto derivado debe llevar la leyenda *«Autorizada su circulación por
    Resolución Nº50 del 2019 de la Dirección Nacional de Fronteras y Límites del
    Estado»*, y los derivados deben revisarse y validarse con DIFROL. Esto no es
    trámite decorativo: son límites de Estado.
11. **Para qué sirve en el modelo** — es la **referencia legal** del límite. La
    ficha 1.1 es la herramienta de trabajo diaria; esta es la que se cita cuando
    el producto va a una mesa de decisión pública y alguien pregunta «¿de dónde
    sacaron ese borde?».
12. **Utilidad** — ★★★
13. **Riesgos y límites**
    - **297 MB.** Para consultar «lat/lon → comuna» es un martillo para una
      chinche: conviene simplificar la geometría una vez y guardar la versión
      liviana, dejando el original archivado como respaldo.
    - **Los enlaces antiguos murieron.** `https://www.ide.cl/descargas/capas/subdere/DivisionPoliticoAdministrativa2020.zip`
      y `.../limites_dpa_v0702_sirgaschile_gcs.rar`, que `datos.gob.cl` todavía
      publica como vigentes, dan **HTTP 404 (VERIFICADO)**. También murió
      `https://www.geoportal.cl/arcgis/rest/services/MinisteriodeInterior/chile_minterior_subdere_DPA2020/MapServer`.
      Lección práctica: **el catálogo nacional está desactualizado**; hay que
      probar cada enlace, no confiar en la ficha.
    - No es un servicio consultable: es un archivo. No sirve para preguntar en
      vivo, sólo para descargar y procesar.
    - Sin población ni ningún atributo estadístico. Sólo geometría y nombres.

## FICHA 1.3 — ★★ BCN · Mapas vectoriales (la alternativa liviana y sin trámite)

1. **Organismo y producto** — Biblioteca del Congreso Nacional de Chile, SIIT,
   «Mapas vectoriales».
   Página: `https://www.bcn.cl/siit/mapas_vectoriales/index_html`
2. **URL exacta de descarga**
   - Comunas: `https://www.bcn.cl/obtienearchivo?id=repositorio/10221/10396/5/comunas_final.zip`
   - Provincias: `https://www.bcn.cl/obtienearchivo?id=repositorio/10221/10397/4/Provincias_Final.zip`
   - Regiones: `https://www.bcn.cl/obtienearchivo?id=repositorio/10221/10398/2/Regiones.zip`
3. **Qué contiene** — los tres niveles administrativos como polígonos, en
   archivos separados. La misma página ofrece además capas muy útiles para este
   proyecto: red vial (50,6 MB), red ferroviaria (449 KB), red hidrográfica
   (48,1 MB), áreas urbanas (515 KB), aeropuertos y aeródromos (17 KB), áreas
   silvestres protegidas, masas de agua y nombres geográficos.
4. **Familia** — BASE TERRITORIAL (y las capas de redes son ACTIVO potencial).
5. **Formato y tamaño** — Shapefile ESRI en ZIP. **VERIFICADO**: comunas
   **42.225.382 bytes ≈ 40 MB**, provincias **37.175.852 ≈ 35 MB**, regiones
   **31.571.181 ≈ 30 MB**. (La página anuncia 30,6 / 30,1 / 25,2 MB — el archivo
   real pesa más que lo anunciado.)
6. **Acceso** — anónimo. **VERIFICADO**: redirección `302` → `HTTP 200`,
   `content-type: application/zip`, con `content-disposition` correcto.
   ⚠️ Requiere enviar un `User-Agent` de navegador: con el agente por defecto de
   `curl`, la página del índice devuelve `HTTP 401`. Es un detalle de
   implementación que hay que dejar anotado en el adaptador.
7. **Granularidad** — comuna, provincia, región.
8. **Vigencia** — **diciembre de 2018** (la página menciona ediciones 2014, 2017 y
   dic-2018). Es la más desactualizada de las tres.
9. **Historia disponible** — ediciones 2014 y 2017 mencionadas en la página.
10. **Licencia** — la más limpia de todas: *«puestos a disposición en virtud del
    principio de transparencia de la función pública»*, **uso libre citando como
    fuente a la Biblioteca del Congreso Nacional de Chile**. Sin cláusula
    no-comercial y sin trámite DIFROL. La propia BCN advierte que el material es
    **únicamente referencial y no apto para trabajos que exijan precisión
    geodésica**.
11. **Para qué sirve en el modelo** — plan B liviano y legalmente cómodo. Y sobre
    todo: **sus capas de red vial y ferroviaria son un catastro de activos
    gratuito** para la parte de accesibilidad («una de ellas sin acceso
    alternativo»).
12. **Utilidad** — ★★ para límites (por antigüedad) · ★★★ para las capas de redes.
13. **Riesgos y límites**
    - **2018.** No refleja cambios comunales posteriores. Para límites, preferir
      1.1 o 1.2.
    - Precisión declarada como referencial por la propia fuente: no usarla para
      decidir de qué lado de un borde cae un activo.
    - Necesita `User-Agent` de navegador.

## FICHA 1.4 — ★ OCUC (Observatorio de Ciudades UC) · Comunas y DPA 2026

1. **Organismo y producto** — Centro de Datos OCUC, Observatorio de Ciudades de
   la Universidad Católica. **Es un tercero que republica dato del INE y SUBDERE**,
   no la fuente oficial.
2. **URL exacta**
   - «Comunas de Chile» (polígonos):
     `https://services9.arcgis.com/kKJR3Qt68ohAWuet/arcgis/rest/services/Comunas_de_Chile/FeatureServer`
     · ficha: `https://hub.arcgis.com/datasets/e3a1f8c4aa014429847c2944e3d92406_0`
   - «División Político Administrativa de Chile 2026» (**líneas**, no polígonos):
     `https://services9.arcgis.com/kKJR3Qt68ohAWuet/arcgis/rest/services/Divisi%C3%B3n_Pol%C3%ADtico_Administrativa_de_Chile/FeatureServer`
     · ficha: `https://ideocuc-ocuc.hub.arcgis.com/datasets/6a8418c478cd41b7b21c479a71546f1d_0`
3. **Qué contiene** — «Comunas de Chile» trae campos `cut`, `comuna`, `provincia`,
   `region`, `ine`. La DPA 2026 es la geometría **lineal** de los límites
   actualizada a 2026.
4. **Familia** — BASE TERRITORIAL.
5. **Formato y tamaño** — servicio ArcGIS consultable; descarga en Shapefile, CSV,
   KML o GeoJSON desde el Hub. Tamaño no declarado.
6. **Acceso** — anónimo. **VERIFICADO**: el servicio de comunas responde
   `HTTP 200`; consulta por punto `-70,40 / -23,65` devuelve
   `cut 2101 · ANTOFAGASTA · provincia ANTOFAGASTA · región II`.
7. **Granularidad** — comuna (con provincia y región como atributos).
8. **Vigencia** — «Comunas de Chile» creada en 2019, última modificación
   02-jul-2026. La DPA lineal declara vigencia 2026 (mod. 04-may-2026).
9. **Historia disponible** — no declarada.
10. **Licencia** — **CC BY-NC 4.0** (Atribución - **No Comercial**), *«sin
    perjuicio de las condiciones específicas definidas por la fuente original:
    INE; SUBDERE»*. La cláusula no-comercial es un límite real si el instrumento
    llegara a tener uso comercial.
11. **Para qué sirve en el modelo** — es el único camino verificado con dato
    declarado **2026**, útil para chequear si hubo cambios de límite después de
    2023. Como fuente principal, no.
12. **Utilidad** — ★
13. **Riesgos y límites**
    - ⚠️ **Es un tercero.** Misma advertencia que quedó anotada para
      `api.gael.cloud` en `FUENTES_DE_DATOS_NACIONALES.md`: sirve para prototipar
      y contrastar, **no** para alimentar decisión pública.
    - Restricción **no comercial**.
    - La capa «Comunas de Chile» devuelve **10.785 registros**, no 346: cada isla
      y cada pedazo suelto va como fila aparte. Hay que agrupar por `cut` antes de
      usarla, o los conteos salen absurdos.
    - La DPA 2026 son **líneas**, no polígonos: no sirve directamente para
      preguntar «¿en qué comuna cae este punto?».

## FICHA 1.5 — ★★ INE · Microdato Censo 2024 georreferenciado (jerarquía completa)

1. **Organismo y producto** — INE, servicio `CPV2024_Microdato_gdb` (Censo de
   Población y Vivienda 2024, microdato público georreferenciado).
2. **URL exacta**
   `https://services5.arcgis.com/hUyD8u3TeZLKPe4T/arcgis/rest/services/CPV2024_Microdato_gdb/FeatureServer`
   · ficha: `https://www.arcgis.com/home/item.html?id=f3889397813c44c3ac09bfbf01877c57`
3. **Qué contiene** — **ocho niveles territoriales encajados**, del país a la
   manzana: capa 0 REGIÓN · 1 COMUNA · 2 DISTRITO · 3 LOCALIDAD · 4 ENTIDAD ·
   5 LUC · 6 ZONA · 7 MANZANA. La capa de comuna trae **101 campos**, entre ellos
   `CUT`, `n_per` (personas), `n_hombres`, `n_mujeres`, tramos de edad
   (`edad_0_5` … `edad_60_mas`), `n_inmigrantes`, `n_pueblos_orig`,
   **`n_discapacidad`**, `prom_escolaridad18`, `prom_per_hog`.
4. **Familia** — BASE TERRITORIAL + ESTADO-CONSECUENCIA.
5. **Formato y tamaño** — servicio ArcGIS; salida en JSON o GeoJSON por consulta.
   Descarga masiva de la capa de manzanas sería pesada (227.410 polígonos).
6. **Acceso** — anónimo, público. **VERIFICADO**: capa 1 (comuna) devuelve
   **346** registros; capa 7 (manzana) devuelve **227.410**.
7. **Granularidad** — hasta **manzana censal**. Es, con diferencia, lo más fino
   que publica el Estado con población asociada.
8. **Vigencia** — Censo 2024. Ítem modificado el 10-oct-2025.
9. **Historia disponible** — existen los equivalentes del Censo 2017
   («Microdatos Censo 2017: Manzana», ítem `54e0c40680054efaabeb9d53b09e1e7a`;
   «Entidad», `5725492dd66340c8852e40cfae5e5928`), lo que permite comparar
   2017 vs 2024. **SUPUESTO**: no se verificó el contenido de esas dos capas.
10. **Licencia** — igual que la ficha 1.1: **no declarada en el ítem**. Acceso
    público sin registro. Hay que confirmarlo con el INE antes de publicar
    derivados.
11. **Para qué sirve en el modelo** — es lo que permite pasar de «se cortó la luz
    en la comuna de Antofagasta» (401.096 personas, cifra inútilmente gruesa) a
    «se cortó la luz en estas manzanas, donde viven N personas, de las cuales X
    tienen discapacidad y Z son mayores de 60». Eso es exactamente lo que el
    criterio oficial de Alerta Amarilla —capacidad de respuesta superada— necesita
    para justificarse.
12. **Utilidad** — ★★ hoy, ★★★ cuando el modelo baje de comuna a manzana.
13. **Riesgos y límites**
    - **Riesgo de identificación de personas.** Es microdato: agregado, pero fino.
      Se llama «público» y el INE lo publica así, pero conviene no redistribuir
      cruces a nivel manzana sin pensarlo dos veces.
    - 227.410 polígonos: no se procesa en una notebook a la ligera. Conviene
      recortar por región antes de trabajar.
    - Licencia no declarada.

## FICHA 1.6 — ★★★ HALLAZGO COLATERAL · Alertas SENAPRED vigentes, ya en polígonos comunales con CUT

No lo buscaba, pero apareció y cambia el panorama: **SENAPRED publica sus alertas
vigentes como capas geoespaciales por comuna, con el código CUT adentro, y son
consultables sin registro.** Es decir, el «idioma administrativo» al que hay que
traducir ya está publicado en formato de máquina.

1. **Organismo y producto** — SENAPRED; publicación operada desde la cuenta
   ArcGIS de MINSAL (`minsal_admin`). Tablero público «ALERTAS SENAPRED».
   Tablero: `https://www.arcgis.com/apps/dashboards/bdc345e01e324af490800634d0f0e3a5`
2. **URL exacta de los servicios** (alertas meteorológicas, una capa por color):
   - Verde: `https://services3.arcgis.com/CNzkI2T3GmfwkaAR/arcgis/rest/services/METEOROLOGICAS_VERDE/FeatureServer/0`
   - Amarilla: `https://services3.arcgis.com/CNzkI2T3GmfwkaAR/arcgis/rest/services/METEOROLOGICAS_AMARILLA/FeatureServer/0`
   - Roja: `https://services3.arcgis.com/CNzkI2T3GmfwkaAR/arcgis/rest/services/METEOROLOGICAS_ROJA/FeatureServer/0`
   - Mapas web hermanos, mismo esquema, otras amenazas (**SUPUESTO**, sólo se
     verificó el meteorológico): `WM_VOLCANICAS` (`581bec7448d44e279f302889ebe97b7d`),
     `WM_FORESTALES` (`c8ffaf4f34d9484fa984bdb663462099`),
     `WM_BIOLOGICAS` (`2e5e4b2a068b44e88996bcff44d667ce`).
3. **Qué contiene** — polígonos de las comunas **en alerta ahora mismo**, con
   campos `CUT_REG`, `CUT_PROV`, `CUT_COM`, `REGION`, `PROVINCIA`, `COMUNA`,
   `SUPERFICIE`, `TIPO_ALERT`, `CAUSALIDAD`, `FECHA_INI`.
4. **Familia** — AMENAZA / ESTADO-CONSECUENCIA, apoyada en BASE TERRITORIAL.
5. **Formato y tamaño** — servicio ArcGIS, respuesta JSON/GeoJSON. Liviano: son
   decenas o pocos cientos de polígonos.
6. **Acceso** — anónimo, sin registro. **VERIFICADO** el 15-ago-2026:
   `METEOROLOGICAS_VERDE` = **118** comunas · `METEOROLOGICAS_AMARILLA` = **177** ·
   `METEOROLOGICAS_ROJA` = **0**. Registros de muestra reales:
   `06111 · Olivar · Cachapoal · O'Higgins · Amarilla · Evento meteorológico · 13-08-2026`;
   `05303 · Rinconada · Los Andes · Valparaíso`; `13122 · Peñalolén · Santiago · RM`.
7. **Granularidad** — comuna.
8. **Vigencia** — es un **estado vivo**: refleja las alertas del día. El mapa web
   meteorológico se modificó el 03-ago-2026.
9. **Historia disponible** — **no**. Sólo lo vigente; cuando la alerta se cancela,
   la fila desaparece. ⚠️ Si el proyecto quiere serie histórica para validar el
   `C_clim`, **hay que empezar a guardar una foto diaria desde ya**. Cada día que
   pasa sin capturar es historia perdida para siempre.
10. **Licencia** — **no declarada** en los ítems. Publicación pública sin
    registro. Antes de reutilizar en un producto formal, pedir la condición de uso
    a SENAPRED.
11. **Para qué sirve en el modelo** — cierra el circuito de la sección 4 del
    documento de integración: si nuestra salida usa CUT, calza directo con la
    alerta oficial. Además da **contra qué validar**: podemos preguntar «¿nuestro
    coeficiente marcó riesgo en las mismas comunas que SENAPRED puso en amarilla?».
12. **Utilidad** — ★★★
13. **Riesgos y límites**
    - ⚠️ **El CUT viene como texto con cero adelante** (`'05303'`, `'06111'`),
      mientras el INE lo entrega como número entero (`2101`). Si se cruzan sin
      normalizar, no calza **nada**. Es el clásico error silencioso: no da error,
      simplemente devuelve cero coincidencias.
    - Publicado desde una cuenta de **MINSAL**, no de SENAPRED. Hay que confirmar
      con SENAPRED que es su canal oficial y no un montaje interno que puede
      desaparecer sin aviso.
    - Sin historia (punto 9).
    - No se verificaron las capas volcánicas, forestales ni biológicas.

---

# 2 · ZONAS GEOGRÁFICAS DE LA DMC / SERNAGEOMIN

## Veredicto: **EXISTE — pero por la puerta de atrás, y con un parche pendiente**

Respuesta corta: **sí, los polígonos existen y se pueden bajar hoy, sin registro.**
Pero **no** los publica la DMC ni SERNAGEOMIN en un catálogo oficial. Están
republicados en la nube de ESRI por otros organismos del propio Estado —ONEMI/
SENAPRED y SERNAGEOMIN— sin licencia declarada y sin compromiso de continuidad.

Y el sitio que *debería* tenerlos, el **GeoNode de la DMC**
(`https://geonode.meteochile.gob.cl/`), **está caído**: responde, pero sólo con un
cartel de «Portal en Mantenimiento» por tiempo prolongado. Su servidor de mapas
contesta pero con **cero capas publicadas** (`FeatureTypeList` vacío en WFS 1.1.0
y 2.0.0; WMS sin capas nombradas). Todas sus rutas de API dan 404. **VERIFICADO.**

La analogía: el mapa de los pisos altitudinales existe y alguien lo dibujó bien,
pero está pegado en el corcho de otra oficina, sin firma ni fecha, y la oficina
que lo hizo tiene la puerta cerrada por refacciones.

## FICHA 2.1 — ★★★ ONEMI/SENAPRED · `coordenadas_zonas_DMC` (LA CAPA MAESTRA)

1. **Organismo y producto** — cuenta ArcGIS de ONEMI (`udt_onemi`), capa
   «coordenadas_zonas_DMC». Es la **traducción a polígonos de las franjas con que
   la DMC declara**.
2. **URL exacta**
   - Servicio: `https://services5.arcgis.com/i7S5PSnIJAUcWvSE/arcgis/rest/services/coordenadas_zonas_DMC/FeatureServer/0`
   - Descarga completa en GeoJSON, **una sola petición**:
     `https://services5.arcgis.com/i7S5PSnIJAUcWvSE/arcgis/rest/services/coordenadas_zonas_DMC/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson`
   - Ficha ArcGIS: `https://www.arcgis.com/home/item.html?id=bdb3e9905b844513871bdef9c7ec2e02`
3. **Qué contiene** — **181 polígonos** con tres campos: `region` (código de dos
   caracteres: `01a, 01b, 02, 03, 04, 05, 05m, 06, 07, 08a, 08b, 09, 10a, 10b, 11, 12`),
   `zona` (el nombre de la franja) e `indice` (la clave canónica, `region_zona`).
   Vocabulario de `zona` verificado: **Litoral · Cordillera_Costa · Pampa ·
   Precordillera · Precordillera_Salar · Cordillera · Valle_Longitudinal ·
   Litoral_Interior · Chiloe · Cordillera_Austral(_Norte/_Sur) ·
   Patagonia_Subandina(_Norte/_Sur) · Insular(_Norte/_Central/_Sur) ·
   Pampa_Patagonica(_Norte/_Sur)**.
4. **Familia** — BASE TERRITORIAL (la «segunda geografía» del problema de la
   sección 4 de `INTEGRACION_SENAPRED.md`).
5. **Formato y tamaño** — GeoJSON, **463.227 bytes (≈ 452 KB)**. Ridículamente
   liviano comparado con los 297 MB de la DPA. Cabe entero en el repositorio.
6. **Acceso** — anónimo, sin registro. **VERIFICADO** el 15-ago-2026: `HTTP 200`,
   181 registros, GeoJSON completo de 463.227 bytes en una petición. Muestra real
   de la Región de Antofagasta: `02_Litoral`, `02_Cordillera_Costa`, `02_Pampa1`,
   `02_Pampa2`, `02_Precordillera`, `02_Precordillera_Salar`, `02_Cordillera1`,
   `02_Cordillera2`.
7. **Granularidad** — zona geográfica dentro de región. Algunas franjas vienen
   partidas en trozos numerados (`Pampa1`/`Pampa2`, `Cordillera1`/`Cordillera2`).
8. **Vigencia** — creada el **23-oct-2024**.
9. **Historia disponible** — no. Es la definición de las zonas, no un registro de
   eventos.
10. **Licencia** — ⚠️ **`licenseInfo` en `null`. Sin metadato, sin atribución
    declarada, sin condiciones de uso.** Es un dato público en la nube, pero
    formalmente huérfano.
11. **Para qué sirve en el modelo** — **es la pieza que faltaba.** Con esta capa
    más la comunal (ficha 1.1), cada subestación se puede ubicar en las **dos
    geografías a la vez** con dos consultas: «está en la comuna de Calama (CUT
    2201) **y** en la zona `02_Precordillera`». Ese doble domicilio es literalmente
    el requisito de «doble adscripción territorial» de la sección 6 del documento
    de integración.
12. **Utilidad** — ★★★ (la más importante de todo este catastro después de la
    comunal)
13. **Riesgos y límites**
    - ★ **PARCHE OBLIGATORIO — el desfase de 2026.** La capa trae
      `02_Precordillera`, un solo polígono. Pero los avisos vigentes de la DMC hoy
      hablan de **`02_Precordillera_Occidental` y `02_Precordillera_Alto_Loa`**
      por separado (justamente las dos que aparecen en el ejemplo de Antofagasta
      del documento de integración). **La DMC partió esa zona en dos después de
      octubre de 2024 y la capa no se actualizó.** Sin resolver esto, el cruce con
      los avisos actuales de Antofagasta **no calza** — y falla en silencio, que es
      lo peor. Hay que dividir el polígono a mano o contrastarlo con la capa
      SERNAGEOMIN/DGA de la ficha 2.2.
    - **Sin licencia ni respaldo institucional.** Republicación de un tercero
      dentro del Estado. Cachear copia propia con fecha, sí o sí.
    - Nadie garantiza que siga viva mañana.
    - **Acción recomendada:** pedir la capa oficial a la DMC por Ley de
      Transparencia (`contacto@meteochile.gob.cl`, Oficina Banco de Datos). Eso
      convierte un dato huérfano en un dato citable.

## FICHA 2.2 — ★★★ SERNAGEOMIN · servicio «Minuta Técnica» (zonas + peligro + isoterma)

1. **Organismo y producto** — SERNAGEOMIN, servicio `Minuta_Técnica`. El nombre
   interno de una de sus capas (`DGA_TEST.ADM_DGA.PDMC_ZONAS_PRONOSTICOPP_P`)
   delata el origen: es la geodatabase de la **DGA**, con las «zonas de pronóstico
   de precipitación de la DMC». O sea: la cadena real es DMC → DGA → SERNAGEOMIN.
2. **URL exacta**
   `https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/services/Minuta_T%C3%A9cnica/FeatureServer`
   - capa **0** — `Isoterma 0°C` (65 polilíneas, campos `ALTURA`, `COD_ISOT`)
   - capa **1** — `Minuta técnica`: **127 polígonos**, campos `POS_OCURRENCIA`
     (alta/media/baja), `ZONA`, `REGION`, `FECHA`
   - capa **2** — zonas de pronóstico: **127 polígonos**, campos `REGION`,
     `COD_REGION`, `COD_ZONA`, `COD_SECTOR` (ej. `02-6`), `SECTOR`
   - Visor público: `https://sernageomin.maps.arcgis.com/apps/webappviewer/index.html?id=14bdfecbb9154dbc976f958a3420cf52`
3. **Qué contiene** — la geografía de las zonas **y**, encima, el peligro de
   remoción en masa declarado por zona con su fecha de vigencia. Es exactamente el
   «FEN dinámico ya operativo» que identifica el documento de integración.
4. **Familia** — AMENAZA, sobre BASE TERRITORIAL.
5. **Formato y tamaño** — servicio ArcGIS. ⚠️ La capa 2 sin simplificar pesa
   **≈ 96 MB** en GeoJSON. Con el parámetro `maxAllowableOffset=0.01` baja a
   **1.162.725 bytes (≈ 1,1 MB)** — **VERIFICADO**. Usar siempre la versión
   simplificada, o paginar.
6. **Acceso** — anónimo, sin registro. **VERIFICADO** el 15-ago-2026: capas 1 y 2
   devuelven **127** registros cada una; sistema de referencia EPSG:4326 nativo.
7. **Granularidad** — zona / sector dentro de región. Mezcla zonas atómicas
   («Precordillera-Salar», «Pampa Nortina», «Depresión Central») con **zonas
   agregadas de uso operativo** («T. sur Precord. de Antofagasta y Precord.
   Atacama», «Región de Antofagasta»).
8. **Vigencia** — capa operativa, viva.
9. **Historia disponible** — **no publicada.** Ver riesgos.
10. **Licencia** — no declarada. La cuenta que la publica no es un `owner`
    identificado como SERNAGEOMIN institucional.
11. **Para qué sirve en el modelo** — es la **fuente de validación independiente**
    del `C_clim`: SERNAGEOMIN dice alta/media/baja por zona, nosotros decimos lo
    nuestro, y se comparan. Además es el mejor candidato para arreglar el parche de
    la ficha 2.1, porque su segmentación puede tener la Precordillera ya partida.
12. **Utilidad** — ★★★
13. **Riesgos y límites**
    - ⚠️ **Hoy, 15-ago-2026, `POS_OCURRENCIA` y `FECHA` están en `null` en los 127
      registros. VERIFICADO.** Es una capa que se **rellena cuando hay minuta
      vigente** y se vacía cuando no. Consecuencia dura: **no hay archivo
      histórico.** Si el proyecto quiere validar su coeficiente contra
      SERNAGEOMIN, tiene que **empezar hoy a guardar una foto cada pocas horas**.
      Cada día sin capturar es evidencia perdida para siempre. Esto responde,
      además, el pendiente nº 3 de `INTEGRACION_SENAPRED.md`: es accesible
      programáticamente, **pero sin historia**.
    - Las zonas agregadas mezcladas con las atómicas obligan a un trabajo de
      limpieza antes de cruzar.
    - 96 MB si se pide mal.

## FICHA 2.3 — ★★★ DMC vía MINSAL · avisos, alertas y alarmas VIVOS con geometría

1. **Organismo y producto** — capas `Meteochile_AAA_*` («Avisos, Alertas,
   Alarmas»), publicadas desde la misma cuenta ArcGIS que las alertas SENAPRED de
   la ficha 1.6. Se actualizan solas: hay un campo `fecha_ejecucion_notebook`, o
   sea que un proceso automático las refresca (observado: ≈ cada hora).
2. **URL exacta** (base `https://services3.arcgis.com/CNzkI2T3GmfwkaAR/arcgis/rest/services/`)
   - `Meteochile_AAA_Avisos_Zonas/FeatureServer/0` — avisos vigentes
   - `Meteochile_AAA_Alertas_Area/FeatureServer/0` — alertas vigentes
   - `Meteochile_AAA_Alarmas_Vista/FeatureServer/0` — alarmas vigentes
3. **Qué contiene** — el aviso/alerta con su **polígono** y sus metadatos.
   Campos verificados en `Alertas_Area`: `codigoMeteo`, `tipo`, `emision`,
   `fenomeno`, `condicionSinoptica`, **`desde`** y **`hasta`**, `tipoZonaAfecta`,
   **`dataZonaAfecta`**, `textoZonaAfecta`, `titulo`, `tablaHTML_limpia`
   (los montos estimados), `fecha_actualizacion`.
   En `Avisos_Zonas`: `id_evento`, `tipo`, `codigoMeteo`, `fenomeno`,
   `codigo_zona`, `zona_real`, `titulo`, `fecha_actualizacion`.
4. **Familia** — AMENAZA, sobre BASE TERRITORIAL.
5. **Formato y tamaño** — servicio ArcGIS, muy liviano (unidades a decenas de
   registros).
6. **Acceso** — anónimo. **VERIFICADO** el 15-ago-2026, 18:51 hrs de última
   actualización de la DMC: **11 avisos**, **3 alertas**, **0 alarmas**.
   Registros reales capturados:
   - `A413/2026 · Aviso · Precipitaciones · «Precipitaciones Normales a Moderadas
     en zonas de la región de Coquimbo» · 2 zonas DMC asociadas`
   - `AA137/2026 · Alerta · Precipitaciones · Sistema frontal · desde «la madrugada
     del lunes 17 de agosto del 2026» hasta «la mañana del lunes 17» ·
     dataZonaAfecta = ip · textoZonaAfecta = Isla de Pascua · montos 25-35 mm`
7. **Granularidad** — zona DMC (y a veces región completa, según `tipoZonaAfecta`).
8. **Vigencia** — **es el estado vivo**, refrescado cada hora aprox.
9. **Historia disponible** — **no.** Cuando el aviso vence, desaparece.
10. **Licencia** — no declarada.
11. **Para qué sirve en el modelo** — **tres cosas de golpe.**
    (a) Es la fuente de disparo en tiempo real que pedía la prioridad 2 de
    `FUENTES_DE_DATOS_NACIONALES.md`, y **en formato estructurado**, no HTML que
    haya que parsear: resuelve el pendiente «ver si la DMC tiene servicio de datos
    abiertos».
    (b) Los códigos `A413/2026` y `AA137/2026` son **exactamente** el identificador
    estable y versionado que exige la sección 6 del documento de integración
    (los `AA130`, `A388-5` del ejemplo de Alexis).
    (c) Los campos `desde`/`hasta` dan la **vigencia temporal explícita**.
12. **Utilidad** — ★★★
13. **Riesgos y límites**
    - ★ **Es la llave del join, y trae el vocabulario NUEVO.** El campo
      `dataZonaAfecta` entrega los códigos DMC actuales —ya con
      `02_Precordillera_Occidental` y `02_Precordillera_Alto_Loa` separados— y esos
      códigos **empatan con el campo `indice`** de la capa ONEMI (ficha 2.1). Ese
      es el punto de unión entre las dos geografías. Pero justamente por eso
      **el parche de la Precordillera es obligatorio**: sin él, esas dos zonas no
      encuentran polígono.
    - **Sin historia** (punto 9): capturar desde hoy o perderla.
    - Publicado desde una cuenta de **MINSAL**, no de la DMC. Sin licencia ni SLA.
    - `textoZonaAfecta` sirve además como **diccionario de validación**: entrega en
      texto plano el listado región → zonas, que es lo más parecido a la
      definición oficial de las franjas que se pudo encontrar.

## Callejones sin salida — verificados, para no repetir el esfuerzo

Vale la pena dejarlos anotados: cada uno costó intentos y no hay que volver a
gastarlos.

| Fuente | Qué pasó |
|---|---|
| **GeoNode DMC** `geonode.meteochile.gob.cl` | Responde `HTTP 200` pero es un cartel de «Portal en Mantenimiento» (DGAC/DMC), por tiempo prolongado |
| Su API GeoNode (`/api/v2/datasets/`, `/api/layers/`, `/catalogue/`, …) | Todas **404** |
| Su WFS 1.1.0 y 2.0.0 (`GetCapabilities`) | `HTTP 200` pero **cero FeatureTypes** publicados |
| Su WMS 1.3.0 | `HTTP 200` (164 KB, todo son sistemas de coordenadas) pero **sin capas nombradas** |
| Su GeoServer REST | `HTTP 401`, pide credenciales |
| API climatológica DMC + OGC API WIS2 (`wischile.meteochile.gob.cl`) | Sólo datos **por estación puntual**; ninguna capa de zonas |
| **datos.gob.cl** con `zonas meteorologicas`, `meteorologica`, `DMC`, `sernageomin peligro` | **0 resultados** en todas |
| Búsqueda del catálogo de **geoportal.cl** | `HTTP 500` |
| `portalgeomin.sernageomin.cl` | **No resuelve DNS** |
| `miratuterritorio.cl` (visor de emergencias de IDE Chile) | **No resuelve DNS** — el visor anunciado en 2024 ya no existe |
| GeoServer de `geoportal.cl` (`/geoserver/wfs`) | Existe, pero exige nombre de espacio de trabajo; se probaron `subdere`, `ide`, `snit`, `dpa`, `minterior`, `interior` → todos **404**. Queda pendiente averiguar el nombre correcto |

**Conclusión operativa de esta sección:** el bloqueo que se temía **no existe**,
pero el dato está en una situación frágil —sin licencia, sin dueño declarado, sin
historia y con un desfase de vocabulario. La acción urgente no es técnica sino
administrativa: **pedir la capa oficial a la DMC** y, mientras tanto, **empezar a
guardar fotos diarias** de las capas vivas.

---

# 3 · CÓDIGO ÚNICO TERRITORIAL (CUT)

El CUT es el número de catálogo de cada comuna: `2201` = Calama, `13101` =
Santiago. Es lo que evita el desastre de cruzar por nombre, donde «Ñuñoa»,
«NUNOA» y «ÑUÑOA» son tres comunas distintas para la máquina.

## FICHA 3.1 — ★★★ SUBDERE · Planilla oficial de Códigos Únicos Territoriales

1. **Organismo y producto** — SUBDERE, «Códigos Únicos Territoriales actualizados
   al 06 de septiembre de 2018», planilla `CUT_2018_v04`.
2. **URL exacta**
   - Planilla: `https://www.subdere.gov.cl/sites/default/files/documentos/CUT_2018_v04.xls`
   - Página: `https://www.subdere.gov.cl/documentacion/códigos-únicos-territoriales-actualizados-al-06-de-septiembre-2018`
   - Respaldo legal: `https://www.subdere.gov.cl/sites/default/files/documentos/DTO-1115-SEPT_2018.pdf`
     (Decreto Exento N°1.115 del Ministerio del Interior, publicado en el Diario
     Oficial el 21-sep-2018; es el que crea la Región de Ñuble y sus comunas)
3. **Qué contiene** — la tabla maestra región / provincia / comuna con su código.
4. **Familia** — BASE TERRITORIAL.
5. **Formato y tamaño** — **XLS heredado** (Excel 97-2003, no XLSX),
   **69.120 bytes (68 KB)**. El decreto pesa 1.715.731 bytes.
6. **Acceso** — anónimo. **VERIFICADO**: `HTTP 200`,
   `content-type: application/vnd.ms-excel`, `content-length: 69120`,
   `last-modified: 12-abr-2022`.
7. **Granularidad** — comuna, provincia, región.
8. **Vigencia** — **2018, versión 04, y sigue siendo la vigente.** No es descuido:
   desde la creación de Ñuble no se han creado comunas nuevas. Es la respuesta a
   la pregunta «¿qué año? las comunas cambian»: **el país tiene 346 comunas desde
   2018 y el código no ha cambiado.**
9. **Historia disponible** — el decreto de 2018 y sus antecesores.
10. **Licencia** — **no declarada**, ni en la página ni en el archivo.
11. **Para qué sirve en el modelo** — es la **piedra de toque**. Toda tabla del
    proyecto lleva CUT como clave, y esta planilla es contra qué validarla.
12. **Utilidad** — ★★★
13. **Riesgos y límites**
    - XLS viejo: algunas librerías modernas ya no lo leen sin un motor extra.
    - Sin licencia declarada.
    - Un espejo en `geoportal.cl` (ficha 35342) apunta al **mismo archivo**, no es
      fuente independiente.
    - ⚠️ **Nunca guardar el CUT como número entero.** Los códigos del norte llevan
      cero adelante (`05303`, `06111`, `01101`). Si se guarda como número, el cero
      se pierde y el cruce con las capas de SENAPRED —que lo entregan como texto—
      falla en silencio (ver ficha 1.6).

## FICHA 3.2 — ★★★ INE · «Límites DPA Censal», capa de comunas simplificada (CUT al día, con geometría)

1. **Organismo y producto** — INE, servicio `Limites_DPA_Censal_C17_2025`, capa 14
   `Comunas_Simp16R_Grs`. Es la versión **liviana y actual** de los límites.
2. **URL exacta**
   `https://services5.arcgis.com/hUyD8u3TeZLKPe4T/arcgis/rest/services/Limites_DPA_Censal_C17_2025/FeatureServer/14`
   · ficha ArcGIS del servicio: item `c50219e257bc46db807b7ba919592e31`
3. **Qué contiene** — **346 comunas** con `CUT`, `NOM_REGION`, `NOM_PROVINCIA`,
   `NOM_COMUNA` y geometría simplificada. El servicio completo tiene **23 capas**:
   0-2 límites DPA censal · 3-5 región/provincia/comuna simplificadas · 6-8 sin
   islas · 9-19 variantes en el sistema GRS · 20 espacios marítimos ·
   **21 límite internacional DIFROL v4 2023** · 22 áreas en litigio.
4. **Familia** — BASE TERRITORIAL.
5. **Formato y tamaño** — servicio ArcGIS; geometría simplificada, muy liviana.
6. **Acceso** — anónimo. **VERIFICADO**: `HTTP 200`, **346** registros, campos
   confirmados uno por uno.
7. **Granularidad** — comuna, provincia, región (y espacios marítimos y límite
   internacional en otras capas).
8. **Vigencia** — modificado el **08-oct-2025**. Es lo más al día con geometría.
9. **Historia disponible** — no.
10. **Licencia** — no declarada en el ítem (ver sección de licencias).
11. **Para qué sirve en el modelo** — es la **llave maestra recomendada**: liviana,
    al día, del INE, con CUT, y **cuadra en 346 comunas** con el Censo 2024 y con
    la planilla SUBDERE 2018. Tres fuentes independientes que coinciden: eso es
    confianza real, no declarada.
12. **Utilidad** — ★★★
13. **Riesgos y límites** — geometría **simplificada**: no sirve para decidir de
    qué lado de un borde fino cae un activo. Para eso, la DPA 2023 (ficha 1.2).
    Depende del mismo servicio en la nube que las demás.

## Callejones sin salida del CUT (verificados)

| Fuente | Qué pasó |
|---|---|
| Codificador de Gobierno Digital `codificador.digital.gob.cl` | `HTTP 200` pero devuelve 3.705 bytes de cascarón de aplicación web, sin datos. Se probaron tres rutas de API: ninguna entrega el dato |
| Observatorio Logístico `datos.observatoriologistico.cl` (dataview 262940, «[MAESTRO] Códigos Únicos Territoriales») | **No resuelve DNS**, dos intentos. Descartado |
| `datos.gob.cl` buscando «codigos unicos territoriales» | **0 resultados**. El CUT no está en el catálogo nacional |

---

# 4 · POBLACIÓN POR COMUNA

Esta es la variable que convierte un dato técnico en una decisión: «se cortó la
luz» → «se cortó la luz a 40.000 personas».

**Hallazgo de método que ahorra horas:** el INE **no publica sus descargas como
enlaces normales**. Usa una interfaz interna que hay que consultar con dos
peticiones (`POST .../hijosCarpeta/` y `POST .../getArchivos/` con
`idFolder=<identificador>`) y que devuelve las URLs reales en JSON. Por eso el
catálogo nacional muestra el dataset del INE **con cero archivos** y por eso los
buscadores no encuentran las rutas. Detalle en la sección de notas al final.

## FICHA 4.1 — ★★★ INE · Censo 2024 por comuna, vía servicio geoespacial

Es la **misma fuente de la ficha 1.1**, mirada ahora como dato de población.
Además del servicio consultable, hay una **descarga directa en CSV**:

- CSV completo de las 346 comunas:
  `https://hub.arcgis.com/api/download/v1/items/d93e7edbe70f4eecb42c457221c7a73c/csv?layers=0`
  **VERIFICADO**: `HTTP 200`, `text/csv`, **97.104 bytes (95 KB)**.
  ⚠️ **La primera llamada devuelve `HTTP 202` con `{"status":"Pending"}`**: el
  archivo se genera bajo demanda y hay que reintentar unos 25 segundos después.
  Un adaptador ingenuo toma ese 202 por éxito y guarda 108 bytes de basura en vez
  del CSV. Es un error que no da error.
- Consulta puntual verificada: `where=CUT=13101` devuelve
  `SANTIAGO · población 438.856 · viviendas 239.265 · hogares 206.458`.
- **Utilidad ★★★.** Es el **nivel actual** de población, medido, no proyectado.

## FICHA 4.2 — ★★★ INE · Estimaciones y proyecciones 2002-2035 por comuna (base Censo 2017)

1. **Organismo y producto** — INE, «Estimaciones y Proyecciones de la Población de
   Chile 2002-2035, comunas», base Censo 2017.
2. **URL exacta**
   - CSV: `https://www.ine.gob.cl/docs/default-source/proyecciones-de-poblacion/cuadros-estadisticos/base-2017/ine_estimaciones-y-proyecciones-2002-2035_base-2017_comunas0381d25bc2224f51b9770a705a434b74.csv?sfvrsn=b6e930a7_3&download=true`
   - XLSX: `https://www.ine.gob.cl/docs/default-source/proyecciones-de-poblacion/cuadros-estadisticos/base-2017/estimaciones-y-proyecciones-2002-2035-comunas.xlsx?sfvrsn=8c87fc3f_3`
   - Variante **comuna × urbano/rural**: mismo directorio,
     `estimaciones-y-proyecciones-2002-2035-comuna-y-área-urbana-y-rural.csv?sfvrsn=62ba4af3_4&download=true`
   - Portal: `https://www.ine.gob.cl/estadisticas/sociales/demografia-y-vitales/proyecciones-de-poblacion`
3. **Qué contiene** — cabecera real leída:
   `Region, Nombre Region, Provincia, Nombre Provincia, Comuna, Nombre Comuna,
   Sexo (1=Hombre 2=Mujer), Edad, Poblacion 2002, …, Poblacion 2035`.
   La columna `Comuna` **es el CUT**.
4. **Familia** — ESTADO-CONSECUENCIA sobre BASE TERRITORIAL.
5. **Formato y tamaño** — CSV **9.768.366 bytes (9,3 MB)**; XLSX **11.057.485
   bytes (10,5 MB)**; la variante urbano/rural, ≈ 4,8 MB.
6. **Acceso** — anónimo. **VERIFICADO**: `HTTP 200`,
   `content-type: application/octet-stream`, `last-modified: 29-nov-2019`.
7. **Granularidad** — comuna × sexo × **edad simple (año a año)** × año.
8. **Vigencia** — base Censo 2017, congelada desde noviembre de 2019.
9. **Historia disponible** — **34 años: 2002 a 2035.** 2002-2017 son estimaciones
   retrospectivas, 2018 en adelante son proyección. Es la única serie larga.
10. **Licencia** — ver la sección de licencias: hay una **contradicción** entre lo
    que declara el sitio del INE y lo que declara el catálogo nacional.
11. **Para qué sirve en el modelo** — cuando el modelo mire hacia atrás (validar
    contra eventos pasados) o hacia adelante, la población tiene que ser la del
    año correspondiente, no la de hoy. Esta serie es lo que lo permite.
12. **Utilidad** — ★★★
13. **Riesgos y límites**
    - ⚠️ **Codificación latin-1 / ISO-8859-1**, no UTF-8. Si se lee como UTF-8, los
      nombres de comuna se corrompen.
    - ⚠️ **Ya diverge del censo real.** Para Santiago, el Censo 2024 midió 438.856
      personas y la proyección da otra cifra. **Regla práctica: proyecciones para
      la serie temporal, Censo 2024 para el nivel actual.**
    - ⚠️ La URL de la variante urbano/rural lleva **tildes sin escapar** (`área`).
      Con `curl` hay que usar `-g --url "…"` o codificar la URL, o falla.
    - La cadena `?sfvrsn=<código>` **cambia cuando el INE republica el archivo**.
      No conviene dejarla fija en el código: resolverla en el momento con la
      interfaz de carpetas.

## FICHA 4.3 — ★★ INE · Censo 2024, cuadros estadísticos oficiales (XLSX)

1. **Organismo y producto** — INE, cuadros del Censo 2024. **La página no aparece
   en el menú del sitio**, hay que ir directo.
2. **URL exacta**
   - Índice: `https://censo2024.ine.gob.cl/estadisticas/` — **VERIFICADO**
     `HTTP 200`, 303.552 bytes, **20 archivos XLSX**.
   - Cuadro clave: `https://censo2024.ine.gob.cl/wp-content/uploads/2025/03/D1_Poblacion-censada-por-sexo-y-edad-en-grupos-quinquenales.xlsx`
   - Síntesis en PDF: `https://censo2024.ine.gob.cl/wp-content/uploads/2025/12/sintesis_resultados_censo2024.pdf`
3. **Qué contiene** — el D1 trae la tabla «Población censada por sexo y razón
   hombre-mujer, **según comuna**» y la misma por grupo quinquenal × comuna. Su
   diccionario dice textualmente: *«Código comuna: refiere al **código único
   territorial** de cada comuna del país (CUT)»* — confirmación oficial de que
   el CUT es la clave.
   Los otros 19 cuadros: D2 envejecimiento · D4 inmigración internacional ·
   D5 migración interna · D6 fecundidad · V1 viviendas y hogares · V2-V5 vivienda,
   hacinamiento y servicios básicos · H1 servicios básicos del hogar ·
   **P1 discapacidad** · P2-P3 pueblos y lenguas indígenas · P4 afrodescendencia ·
   P5 género · P6 religión · P7-P8 educación.
4. **Familia** — ESTADO-CONSECUENCIA.
5. **Formato y tamaño** — XLSX. D1 = **505.437 bytes**; V1 = 148.334 bytes;
   la síntesis PDF = **18.266.109 bytes (17,4 MB)**.
6. **Acceso** — anónimo. **VERIFICADO**: D1 responde `HTTP 200`,
   `spreadsheetml.sheet`, 505.437 bytes, `last-modified: 04-dic-2025`.
7. **Granularidad** — comuna.
8. **Vigencia** — Censo 2024, publicado por partes entre marzo y diciembre de 2025.
9. **Historia disponible** — el equivalente de 2017 está en el mismo sitio.
10. **Licencia** — la del INE (ver sección de licencias).
11. **Para qué sirve en el modelo** — **P1 discapacidad** y **D2 envejecimiento**
    son las variables de vulnerabilidad que justifican el criterio de Alerta
    Amarilla: no es cuánta gente hay, sino cuánta gente **no puede resolverlo
    sola**.
12. **Utilidad** — ★★
13. **Riesgos y límites** — son cuadros para leer, no bases de datos: hay que
    extraerlos hoja por hoja. Página no enlazada desde el menú: puede moverse.

## FICHA 4.4 — ★★ INE · Censo 2017 por comuna con urbano/rural, superficie y densidad

- CSV directo: `https://hub.arcgis.com/api/download/v1/items/1c64fcb18f5a41e088b25ef9f42b58d7/csv?layers=0`
  **VERIFICADO**: `HTTP 200`, `text/csv`, **66.537 bytes**.
- Cabecera real: `OBJECTID, REGION, NOM_REGION, PROVINCIA, NOM_PROVIN, COMUNA,
  NOM_COMUNA, T_HOM_R, T_MUJ_R, T_POB_R, T_HOM_U, T_MUJ_U, T_POB_U, T_HOM, T_MUJ,
  T_POB, T_VIV_U, T_VIV_R, T_VIV, …, SUPERFICIE__KM2_, Densidad_`. `COMUNA` = CUT.
- **Lo que aporta y nadie más da junto: población urbana y rural separadas,
  superficie en km² y densidad.** Ahí está el punto 5 del encargo, resuelto para
  el corte 2017.
- **Utilidad ★★.** Límite: es Censo 2017, no 2024.

## Huecos confirmados en población

1. ⚠️ **No existe proyección comunal rebasada al Censo 2024.** El INE publicó las
   proyecciones base 2024 con horizonte **1992-2070**
   (`https://www.ine.gob.cl/docs/default-source/proyecciones-de-poblacion/cuadros-estadisticos/proyección-base-2024/estimaciones-y-proyecciones-de-población-1992-2070_base-2024_base-de-datos.xlsx?sfvrsn=8e7df320_6`
   — **VERIFICADO** `HTTP 200`, **918.408 bytes**, actualizado 28-ene-2026), **pero
   918 KB no alcanzan para 346 comunas × 101 edades × 79 años**: es casi seguro
   país y región nomás. La carpeta «Proyección base 2024» sólo tiene esos dos
   archivos. **Este es el hueco más importante para planificar a futuro.**
2. ⚠️ **Los microdatos del Censo 2024 no son localizables públicamente**, pese a
   que el INE anunció su publicación el 04-dic-2025 (y la de manzana-entidad el
   16-dic-2025). Se recorrió la interfaz de carpetas del INE, la página de
   resultados, ambas notas de prensa y el sitio del censo. **No se inventa la URL.**
   Sí está el equivalente georreferenciado por manzana, en la ficha 1.5.
3. **`datos.gob.cl` es inútil para esto.** El dataset del INE figura con
   `num_resources: 0`: sólo enlaza al sitio. No sirve como fuente.

---

# 5 · RURALIDAD, AISLAMIENTO Y ACCESIBILIDAD TERRITORIAL

**Respuesta directa a la pregunta del encargo: no, no es sólo PDF.** Hay CSV,
XLSX, shapefile y geodatabase, y todos responden.

## FICHA 5.1 — ★★★ SUBDERE · Localidades en Condición de Aislamiento (ILCA 2021)

1. **Organismo y producto** — SUBDERE, «Estudio actualización de base censal.
   Identificación de localidades en condición de aislamiento» (ILCA 2021, sobre
   base Censo 2017).
   Página: `https://ide.subdere.gov.cl/estudio-actualizacion-de-base-censal-identificacion-de-localidades-en-condicion-de-aislamiento/`
2. **URL exacta** — cinco archivos, **los cinco VERIFICADOS**:

   | Archivo | URL | Respuesta |
   |---|---|---|
   | CSV | `https://ide.subdere.gov.cl/wp-content/uploads/2023/11/LAC_2021_CENSO2017.csv` | `200 · text/csv · 14.426.296 B` |
   | XLSX | `https://ide.subdere.gov.cl/wp-content/uploads/2023/11/LAC_2021_CENSO2017.xlsx` | `200 · xlsx · 12.333.191 B` |
   | Shapefile | `https://ide.subdere.gov.cl/wp-content/uploads/2023/11/SHP.zip` | `200 · application/zip · 121.009.459 B` |
   | Geodatabase | `https://ide.subdere.gov.cl/wp-content/uploads/2023/11/GDB.7z` | `200 · x-7z-compressed · 48.874.275 B` |
   | Diccionario | `https://ide.subdere.gov.cl/wp-content/uploads/2023/11/Diccionario-de-variables.xlsx` | `200 · xlsx · 12.217 B` |

   Documentación en el mismo directorio `/wp-content/uploads/2023/11/`:
   `EABC-ILCA-2021-2.pdf`, `Definiciones_tecnicas.pdf`,
   `Tiempos_desplazamiento.pdf`, `Listado_localidades_aisladas.pdf`,
   `Ficha-de-Metadatos-1.pdf`.
3. **Qué contiene** — **54 columnas**, y son justo las que este proyecto necesita:
   `CUT`, `COM_NOM`, `TIPO`, `SUB_TIPO`, `NOMBRE`, `ZONA`, `HABITANTES`,
   `VIVIENDAS`, y sobre todo **horas de desplazamiento a cada servicio**:
   `H_ENS_BAS`, `H_ENS_MED`, `H_ENS_PARV` (educación), `H_SS_PRIM` (atención
   primaria), `H_SS_URG` (**urgencias**), `H_A_SS_HOS` / `H_M_SS_HOS` /
   `H_B_SS_HOS` (hospital de alta, media y baja complejidad), `H_BANCO_*`,
   `H_CIUDAD`, `H_CAP_REG`, `H_CAP_PROV`, `H_CAP_COM`. Más un **factor de
   confinamiento climático** (`CONF_CLIMA`), índices compuestos (`I_EDU`,
   `I_SALUD`, `I_BYS`, `INTEGRACION`, `ESTRUCTURA`) y el veredicto:
   `AISLAMIENTO` (numérico), `CONDICION` (AISLADA / no) y `SUCEPTIBLE`.
4. **Familia** — BASE TERRITORIAL (y ESTADO-CONSECUENCIA: mide capacidad de
   respuesta).
5. **Formato y tamaño** — ver tabla. El CSV de 14 MB es lo más práctico.
6. **Acceso** — anónimo. **VERIFICADO**: `HTTP 200`, `text/csv`,
   `content-length: 14426296`, `last-modified: 21-nov-2023`.
7. **Granularidad** — **localidad**, no comuna. Trae CUT, así que se agrega por
   comuna cuando haga falta. El estudio identifica del orden de 3.896 localidades
   en condición de aislamiento.
8. **Vigencia** — ILCA 2021 sobre base Censo 2017. **No hay versión con Censo 2024.**
9. **Historia disponible** — sí, hay versiones anteriores en el repositorio
   Proactiva de SUBDERE: `LOC_ACTUALIZADAS_2018.xlsx`, `SHP 2018.rar`,
   `LOC_AIS_2018.gdb.rar` (identificador 515), y los estudios de 2012 y 2011 en
   PDF (identificadores 387 y 389). **SUPUESTO**: no se les hizo prueba directa.
10. **Licencia** — **no declarada** en ninguna página ni archivo de SUBDERE.
11. **Para qué sirve en el modelo** — **es lo que el criterio de Alerta Amarilla
    pide y nadie tiene a mano.** El criterio oficial no es «cuánto daño», es
    «si los recursos locales alcanzan». `H_SS_URG` —cuántas horas hay hasta una
    urgencia— es exactamente eso, medido, localidad por localidad. Y `CONF_CLIMA`
    es un factor de confinamiento **climático** ya calculado por el Estado: es
    prácticamente un insumo pre-hecho para el `FRC`.
12. **Utilidad** — ★★★
13. **Riesgos y límites**
    - ⚠️ **Separador `;` y coma decimal**, codificación latin-1. Leerlo con la
      configuración por defecto de las herramientas habituales lo destroza sin
      avisar.
    - Base Censo 2017: desactualizado respecto del Censo 2024.
    - Sin licencia declarada.
    - La geodatabase viene en **7z** y otros archivos de SUBDERE en **RAR**: hacen
      falta `p7zip` y `unar`, no basta con descomprimir ZIP.

## FICHA 5.2 — ★★★ SUBDERE · Zonas de Rezago (clasificación COMUNAL)

- URL: `https://ide.subdere.gov.cl/descargas/SHP/ComunasZonasRezago_15112022.zip`
  **VERIFICADO**: `HTTP 200`, `application/zip`, **37.317.387 bytes (35,6 MB)**,
  `last-modified: 21-feb-2024`.
- Es la única capa de vulnerabilidad territorial de SUBDERE que ya viene
  **agregada por comuna**: no hay que agrupar nada.
- Índice de descargas de SUBDERE (**VERIFICADO**, `HTTP 200`):
  `https://ide.subdere.gov.cl/descargas-con-filtros/`
- Utilidad ★★★. Licencia no declarada.

## FICHA 5.3 — ★★ SUBDERE · Localidades aisladas, capas geoespaciales livianas

| Capa | URL | Estado |
|---|---|---|
| Localidades Aisladas, shapefile | `https://ide.subdere.gov.cl/descargas/SHP/LocalidadesAisladas_15112022.zip` | **VERIFICADO** `200 · zip · 223.109 B` |
| Localidades Aisladas, KMZ | `https://ide.subdere.gov.cl/descargas/KML/LocalidadesAisladas_15112022.kmz` | **VERIFICADO** `200 · kmz · 250.348 B` |
| Localidades **No** Aisladas, shapefile | `https://ide.subdere.gov.cl/descargas/SHP/LocalidadesNoAisladas_15112022.zip` | **SUPUESTO** (aparece en el índice verificado, sin prueba directa) |

Del mismo índice, y todos **SUPUESTO**: `MBHT_15112022.zip`,
`RedDiversa_01032023.zip`, `MUNICIPIOS_SIRGAS-CHILE_GCS_V04.rar`,
`OFICINAS_MUNICIPALES_V01.rar`, `UnidadesRegionales_15112022.zip`,
`DelegacionesMunicipales_15112022.zip`, `layer_osm2025_20250206095315.zip`, y unos
29 XLSX de cartera de proyectos (`layer_ip2024_*`, `layer_ise2024_*`,
`layer_icp2024_*`, `layer_rep24_*`, `layer_ptrac24_*`).
**`OFICINAS_MUNICIPALES_V01`** merece una mirada aparte: son puntos de
infraestructura pública municipal, o sea catastro de activos gratis.

## FICHA 5.4 — ★★ SUBDERE · Ruralidad, clasificación comunal

- Datos: `https://proactiva.subdere.gov.cl/bitstream/handle/123456789/514/SALIDA_R.xlsx?sequence=4&isAllowed=y`
  **VERIFICADO**: `HTTP 200`, XLSX, **58.384 bytes**.
- Informe asociado (**SUPUESTO**): «DPDT Asentamientos Humanos Rurales en Chile —
  Clasificación Comunal», mismo identificador 514, `sequence=1`.
- **Granularidad comunal**, que es lo que se pedía.
- Complemento (**SUPUESTO**, del índice verificado): densidad de espacios
  habitados rurales,
  `https://proactiva.subdere.gov.cl/bitstream/handle/123456789/559/Base_Densidad_Rural(shp).rar?sequence=2`
- Biblioteca digital de SUBDERE (**VERIFICADO**, `HTTP 200`):
  `https://ide.subdere.gov.cl/biblioteca-digital-2/`

## FICHA 5.5 — ★ SUBDERE · Límites DPA oficiales (la versión pesada)

- `https://ide.subdere.gov.cl/descargas/SHP/Limite_DPA_03082023.rar`
  **VERIFICADO**: `HTTP 200`, `application/vnd.rar`, **262.380.302 bytes (250 MB)**,
  versión del 03-ago-2023.
- Es el mismo producto de la ficha 1.2 servido desde SUBDERE en RAR. **No hace
  falta si ya se tiene la DPA 2023 del Geoportal.** Se anota para que no se
  descargue dos veces lo mismo.

---

# 6 · LICENCIAS: el punto débil de todo este catastro

Hay que decirlo derecho, porque para un insumo de decisión pública importa tanto
como el dato: **la mayor parte de lo encontrado no declara licencia.**

| Fuente | Qué declara | Estado |
|---|---|---|
| **INE, sitio oficial** | **CC BY-SA 4.0** (Atribución-CompartirIgual). Exige atribución con enlace, permite uso comercial, y **obliga a licenciar los derivados igual**. Formato pedido: *«Fuente: INE, nombre del producto donde se extrae la información, actualizada AAAA»*. Además exige advertir al usuario final de cualquier análisis o transformación y no dar a entender que lo hizo el INE | **VERIFICADO** en `https://www.ine.gob.cl/terminos-de-uso-y-licencia-de-datos-abiertos` |
| **INE vía catálogo nacional** | **CC BY-NC 2.0** (`license_id: cc-nc`, `isopen: false`) para el dataset de proyecciones | **VERIFICADO** vía API de `datos.gob.cl` |
| **INE en la nube ArcGIS** | `licenseInfo: null` — nada | **VERIFICADO** |
| **SUBDERE / IDE SUBDERE** | Nada, en ninguna página ni archivo | **VERIFICADO** (por ausencia) |
| **DPA 2023 en Geoportal** | **CC BY** según `datos.gob.cl`, **más la leyenda obligatoria de DIFROL** (Resolución Nº50 de 2019) y validación de derivados con DIFROL | **VERIFICADO** |
| **BCN** | Uso libre citando a la Biblioteca del Congreso Nacional, por principio de transparencia. **La más limpia de todas** | **VERIFICADO** |
| **OCUC** | **CC BY-NC 4.0** explícita | **VERIFICADO** |
| **Zonas DMC (ONEMI), Minuta SERNAGEOMIN, avisos DMC, alertas SENAPRED** | Nada. `licenseInfo` en `null` en todas | **VERIFICADO** |

🚨 **Contradicción a resolver:** para el **mismo** dataset de proyecciones del INE,
el sitio oficial dice **BY-SA (comercial permitido)** y el catálogo nacional dice
**NC (no comercial)**. Se contradicen. Si el proyecto llega a tener cualquier
arista comercial, hay que pedir confirmación por escrito al INE antes de publicar
derivados. Y el **CompartirIgual** también pesa: incorporar estos datos puede
obligar a que el producto derivado quede bajo CC BY-SA.

**Recomendación:** una sola carta, con la lista de capas de este documento,
pidiendo a INE, SUBDERE, DMC y SENAPRED la condición de uso por escrito. Es
trámite barato y despeja el único riesgo que no se arregla programando.

---

# 7 · Notas técnicas que van a ahorrar horas de depuración

Trampas reales, encontradas probando. Ninguna de estas da error: todas fallan en
silencio, que es lo peligroso.

1. **El CUT no es un número, es un texto.** El INE lo entrega como entero (`2101`),
   SENAPRED como texto con cero adelante (`'05303'`). Si se guarda como número, el
   cero se pierde y el cruce devuelve cero coincidencias sin quejarse. **Guardarlo
   siempre como texto de 5 caracteres.**
2. **Codificación latin-1, no UTF-8**, en el CSV comunal del INE y en el CSV del
   ILCA de SUBDERE. Leerlos como UTF-8 corrompe los nombres.
3. **Separador `;` y coma decimal** en el CSV de SUBDERE.
4. **El Hub de ArcGIS responde `HTTP 202` en la primera llamada** y entrega
   `{"status":"Pending"}`. Hay que reintentar unos 25 segundos después. Un
   adaptador ingenuo guarda 108 bytes de JSON creyendo que bajó el CSV.
5. **Las URLs del INE llevan `?sfvrsn=<código>` que cambia al republicar.** No
   dejarlas fijas en el código: resolverlas en el momento con la interfaz de
   carpetas del INE:
   ```
   POST https://www.ine.gob.cl/estadisticas-por-tema/demografia-y-poblacion/<PRODUCTO>/hijosCarpeta/
   POST https://www.ine.gob.cl/estadisticas-por-tema/demografia-y-poblacion/<PRODUCTO>/getArchivos/
        cuerpo: idFolder=<identificador>
   ```
   Devuelve `{"folder":[{Titulo,Id,…}]}` y `{"documento":[{Titulo,Tipo,Peso,Url}]}`.
   **VERIFICADO** con varias llamadas. Identificadores raíz de
   `estimaciones-y-proyecciones-de-poblacion`: cuadros estadísticos
   `14b161d4-68d5-48e9-8483-212f348b9a59`; subcarpetas Base 2024
   `59db6608-a588-4766-bc2e-fc608107f471`, Base 2017
   `c2ad609a-4bb8-45c1-99d7-160fc251aa82`. Para `censo-de-poblacion-y-vivienda`:
   bases de datos `b7173e2b-1ff5-4f76-b3ad-479c43a37a2a`.
   ⚠️ El JSON devuelve las URLs en `http://` — hay que reescribirlas a `https://`.
6. **Tildes sin escapar** en varias rutas del INE (`proyección-base-2024`, `área`,
   `población`). Con `curl` hay que usar `-g --url "…"` o codificar la URL.
7. **La BCN exige un `User-Agent` de navegador**: con el de `curl` devuelve `401`.
8. **RAR y 7z** en SUBDERE y en los microdatos del INE: hacen falta `unar` y
   `p7zip`.
9. **La capa de comunas de OCUC devuelve 10.785 filas, no 346**: cada isla va
   aparte. Agrupar por `cut` antes de contar.
10. **Pedir la capa 2 de la Minuta Técnica sin simplificar baja 96 MB.** Con
    `maxAllowableOffset=0.01` baja a 1,1 MB.
11. **El catálogo nacional `datos.gob.cl` está desactualizado.** Publica como
    vigentes tres URLs de la DPA que hoy dan `404`, y muestra el dataset del INE
    con cero archivos. **Probar cada enlace, nunca confiar en la ficha.**

---

# 8 · Lo que hay que hacer ahora

**Urgente y que no admite espera** (cada día que pasa es dato perdido):

1. **Empezar a guardar una foto diaria** de las tres capas vivas: la Minuta
   Técnica de SERNAGEOMIN (`POS_OCURRENCIA`, hoy en blanco), los avisos y alertas
   de la DMC, y las alertas de SENAPRED. **Ninguna tiene archivo histórico.**
   Sin esa captura no habrá nunca con qué validar el `C_clim`.

**Importante, esta semana:**

2. **Bajar y archivar** con fecha las cinco piezas base, que juntas pesan menos de
   30 MB: geodatabase Censo 2024 (6,6 MB) · zonas DMC en GeoJSON (452 KB) ·
   planilla CUT de SUBDERE (68 KB) · CSV del Censo 2024 por comuna (95 KB) ·
   CSV del ILCA de SUBDERE (14 MB).
3. **Resolver el parche de la Precordillera** (`02_Precordillera` →
   `02_Precordillera_Occidental` + `02_Precordillera_Alto_Loa`), contrastando la
   capa ONEMI con la de SERNAGEOMIN/DGA. Sin esto, el ejemplo de Antofagasta que
   motivó todo el proyecto no cruza.
4. **Escribir la carta de licencias** a INE, SUBDERE, DMC y SENAPRED.

**Cuando haya tiempo:**

5. Averiguar el nombre del espacio de trabajo del GeoServer de `geoportal.cl`,
   que daría una vía abierta sin depender de la nube de ESRI.
6. Revisar las capas de red vial y ferroviaria de la BCN como catastro de activos.
7. Verificar las capas volcánicas, forestales y biológicas de SENAPRED, que
   siguen el mismo esquema que la meteorológica ya verificada.
