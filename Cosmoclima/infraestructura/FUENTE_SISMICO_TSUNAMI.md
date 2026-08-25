# Sísmico y tsunami: qué se consiguió, por qué canal, y qué quedó bloqueado

Fecha del levantamiento: **19-20 de agosto de 2026**.
Adaptadores: `adaptadores/sismico.py` · `adaptadores/tsunami.py`.

Estas son las dos amenazas históricamente mayores de Chile y son las dos que
faltaban en el consolidador, que ya tenía remoción en masa e inundación. El
resumen de una línea:

> **Tsunami quedó resuelto y es la mejor amenaza que tiene hoy el proyecto**,
> porque trae profundidad de agua en metros. **Sísmico quedó resuelto a medias**:
> hay catálogo de eventos y hay geografía sísmica oficial, pero **la zonificación
> de la norma NCh433 no está publicada abierta** y ése es el trámite pendiente.

---

## 1. Sísmico

### 1.1 Lo que sí se consiguió

**a) Catálogo de eventos — CSN (Centro Sismológico Nacional, Universidad de Chile).**
Dos canales del mismo organismo, complementarios:

| canal | URL | qué da | costo |
|---|---|---|---|
| Base de eventos | `https://evtdb.csn.uchile.cl/events` (POST al formulario) | catálogo M≥4 desde 2012, **todo en una sola respuesta** | 1 petición |
| Catálogo diario | `https://www.sismologia.cl/sismicidad/catalogo/AAAA/MM/AAAAMMDD.html` | listado oficial día por día, **sin umbral de magnitud** | 1 petición por día |

El hallazgo útil acá fue la **base de eventos**. El pendiente que dejó
`FUENTES_DE_DATOS_NACIONALES.md` era «verificar el canal oficial del CSN (no el
tercero)». Está verificado, y además resulta que hay uno mejor que el archivo
diario: `evtdb.csn.uchile.cl` acepta un formulario con rango de fechas, caja
geográfica, profundidad y magnitud, y **devuelve el resultado completo en una
página, sin paginar**. Con una petición se baja el catálogo entero. Eso hace
innecesario raspar catorce años de páginas diarias.

Los dos canales se usan juntos y **se auditan uno al otro**: el adaptador compara
lo que cada uno lista en la ventana en que se solapan y escribe el porcentaje de
coincidencia en el informe, en vez de suponer que la base de eventos está
completa. Importa porque `evtdb` es la base de **registros instrumentales**: no
está garantizado que tenga todos los eventos localizados, sólo los que tienen
registro. El número medido queda impreso cada vez que corre.

**Condiciones de uso.** El CSN autoriza uso **académico y de divulgación**;
cualquier otro fin requiere aprobación expresa por escrito. Alexis autorizó
usarlo sobre esa base (proyecto académico), citando la fuente. `evtdb` declara
`Crawl-delay: 10` en su `robots.txt` y el adaptador lo respeta; al catálogo
diario se le pide 1 página cada 1,5 segundos. Sigue en pie la salvedad ya
anotada: **el día que el instrumento pase de investigación a apoyo operativo del
COGRID o de SENAPRED, el encuadre académico puede dejar de cubrirlo**. Pedir la
autorización escrita ahora es barato.

**b) Geografía sísmica oficial — SENAPRED (Servicio Nacional de Prevención y
Respuesta ante Desastres).** Su servidor `visor-grd.senapred.gob.cl` tiene
`robots.txt` con `Disallow:` vacío (permite todo) y publica, en el servicio
`SIIE/Sismologia`, tres capas de cobertura nacional que sí son geografía:

| capa | qué es | elementos | ¿utilizable? |
|---|---|---|---|
| 0 · Zonas de Rupturas | el área que rompió cada gran terremoto histórico, con **año y magnitud** | 19 polígonos | **sí** |
| 6 · Fallas | trazas de falla con tipo, actividad (*Comprobada* 21, *Probable* 49, sin declarar 179) y fuente (MINVU 205, PUC 44) | 249 líneas, 248 con geometría válida | **sí** |
| 8 · Fallas activas | trazas con `nom_falla` y `largo_km` | 959 declaradas | **no**: llegan `geometry: null` |

⚠️ **La capa 8 no sirve hoy.** Declara 959 trazas y responde HTTP 200, pero el
servidor devuelve `"geometry": null` en todas, incluso pidiendo
`returnGeometry=true` y probando `f=json` además de `f=geojson`. El adaptador
igual **guarda el crudo** —para dejar registro de que la capa existe y de que en
esta fecha no entregaba geometría— y **mide con la capa 6**, que sí la tiene.
Esto es «sin dato» declarado, no un cero silencioso.

⚠️ **Y la capa 6 es rala**: 249 trazas en todo el país, concentradas en zonas
urbanas (la fuente son MINVU y la Pontificia Universidad Católica). Por eso, en
los CSV de salida, `dist_falla_km` dice `>200` cuando no hay ninguna traza
mapeada dentro del radio de búsqueda. **`>200` no significa «acá no hay
fallas»**: significa «acá no hay falla *mapeada en esta capa*». Leerlo como
ausencia de amenaza sería exactamente el error que el proyecto ya cometió una
vez.

Las zonas de ruptura son lo más valioso: son polígonos con **magnitud absoluta
asociada**. Permiten decir «este puente está dentro del área que rompió el
terremoto de 1960» sin inventar nada. Cubren de **1835 a 2010**, con magnitudes
de 7,5 a 9,5, y traen además el efecto secundario declarado («Tsunami
Destructivo», «Tsunami Moderado», etc.). Eso las hace útiles justo donde el
catálogo del CSN es corto: llegan mucho más atrás que 2012.

⚠️ **Trampa de nombres en esta capa**: el campo se llama `zona_ruptu` pero **no
es un topónimo, es el largo de la ruptura en kilómetros** (el terremoto de 1960
trae «1000 Km.»). El adaptador lo guarda como `largo_ruptura_km` y no lo usa
como etiqueta de lugar. Quien lea el crudo directo, que no se confunda.

### 1.2 Lo que NO se consiguió: la zonificación NCh433

**Resultado: SIN DATO abierto.** No es que no se haya buscado.

La zonificación sísmica legal de Chile es la de la **NCh433** (norma chilena de
diseño sísmico de edificios), que divide el país en **zonas sísmicas 1, 2 y 3**
mediante tablas de comunas y provincias, y define los tipos de suelo. Es
obligatoria por el artículo 5.5.7 de la Ordenanza General de Urbanismo y
Construcciones. Dónde se buscó y qué salió:

| dónde | qué se probó | resultado |
|---|---|---|
| **BCN** — Biblioteca del Congreso Nacional | Se bajó el texto oficial completo del **Decreto Supremo N°61 de 2011 del MINVU** (Ministerio de Vivienda y Urbanismo), que es el reglamento vigente de diseño sísmico, vía `bcn.cl/leychile/consulta/obtxml?opt=7&idNorma=1034101` — 761 KB de XML, HTTP 200 | **El decreto NO contiene la tabla de comunas por zona sísmica.** Se verificó buscando en el texto decodificado: cero apariciones de «zona sísmica», «zonificación», «comuna», «Tabla 4.1». El DS 61 trata clasificación de suelos (categorías A–F, parámetro Vs30) y espectros; la zonificación por comuna se queda en la norma NCh433 |
| **INN** — Instituto Nacional de Normalización | la NCh433 es una norma con derechos de autor, que el INN **vende** | no hay versión abierta ni redistribuible |
| **IDE Chile** — Infraestructura de Datos Geoespaciales de Chile (`ide.cl`, `geoportal.cl`) | catálogo y API de búsqueda | `geoportal.cl/geoportal/rest/find/document` redirige a `/login` (HTTP 302): requiere sesión. Las fichas `/catalog/` son públicas pero no hay ficha de zonificación sísmica |
| **datos.gob.cl** | búsqueda `/dataset?q=sismica` (ruta permitida por su `robots.txt`, con `Crawl-Delay: 10` respetado) | **HTTP 405**, el buscador no responde a peticiones programáticas. Su `robots.txt` prohíbe además `/api/`, así que no se insistió |
| **ArcGIS Online** (búsqueda pública `arcgis.com/sharing/rest/search`) | consultas «zonificacion sismica chile», «NCh433», «peligro sismico chile» | única coincidencia real: una capa «Zonas sísmicas de Chile» de una cuenta universitaria (`centenario.pucv2024`), que **exige token** (`code 499, Token Required`). No es fuente oficial y no es accesible |
| **MOP** — Ministerio de Obras Públicas | el **Manual de Carreteras, Volumen 3**, sección 3.1004, trae su propia zonificación sísmica. Se probaron `mc.mop.gob.cl` y el repositorio `repositoriodirplan.mop.gob.cl` | **ambos exigen sesión**: `mc.mop.gob.cl` tiene reCAPTCHA y el repositorio devuelve la página de login de DSpace. Sin descarga programática |
| **SERNAGEOMIN** | su servidor ArcGIS (el mismo que ya usa `traer_capas_sernageomin.py`) tiene `M643_Fallas_geológicas` y `Peligro_Geológico_Shape` | **son locales, no nacionales**: las fallas cubren una sola hoja de carta (~100 × 166 km) y el peligro geológico son 5 polígonos de un área volcánica puntual |
| **CSN** | la página «Avances en la Zonificación Sísmica de Chile» presenta un mapa de peligro de 2014 | **sólo una imagen**, sin capa ni datos descargables. El propio CSN dice que la actualización de la NCh433 «sigue pendiente» |

**Trámite formal recomendado** (corresponde a Alexis, no al proyecto):

1. **Comprar la NCh433.Of1996 Mod.2009 al INN** y, en la misma gestión, **pedir
   autorización escrita para usar la tabla de zonas por comuna dentro de un
   instrumento**. Comprar la norma da derecho a leerla, no necesariamente a
   redistribuir su tabla en un archivo de datos: eso hay que pedirlo aparte.
2. En paralelo, **solicitar a MINVU (División Técnica de Estudio y Fomento
   Habitacional) o a SENAPRED la capa geográfica de zonificación sísmica**, si la
   tienen digitalizada. Es dato que la administración usa; una solicitud por Ley
   de Transparencia (Ley 20.285) es el canal formal.
3. Si aparece, entra como un adaptador nuevo y **no reemplaza** al catálogo: son
   cosas distintas (ver más abajo).

### 1.3 La advertencia que hay que repetir cada vez

**Un catálogo de sismos NO es una zonificación de peligro.** El catálogo dice
*dónde ha temblado y cuánto en los últimos catorce años*. La zonificación dice
*cuánto puede llegar a temblar*. Son preguntas distintas y la segunda no se
deduce de la primera sin un modelo de peligro sísmico que este proyecto no tiene
ni debe inventar.

Por eso las columnas de `sismico_eventos_*.csv` se llaman
`eventos_M6+_a_100km` y no «peligro sísmico». Es la misma lección de
`CORRECCION_RAREZA_PELIGRO.md`: la variable se nombra por lo que mide.

Hay además un sesgo concreto que hay que tener presente: **catorce años son un
pestañeo en escala sísmica**. Una zona que no ha tenido un M8 desde 2012 puede
ser precisamente la que lo tiene pendiente — el llamado «vacío sísmico» es un
argumento de mayor peligro, no de menor. Leer el conteo de eventos como si fuera
peligro invertiría el signo justo donde más importa. Las **zonas de ruptura de
SENAPRED**, que cubren terremotos históricos mucho más allá de 2012, existen en
parte para tapar ese hueco, y por eso se incorporaron.

---

## 2. Tsunami

### 2.1 El bloqueo del SHOA, y por qué no fue un problema

El SHOA (Servicio Hidrográfico y Oceanográfico de la Armada) es quien levanta las
**CITSU** (Carta de Inundación por Tsunami). Su sitio **no se puede raspar**:
`robots.txt` dice `Disallow: /` para todos los agentes. El adaptador **nunca
accede a shoa.cl**, ni con un `HEAD`.

No hizo falta: **SENAPRED republica las mismas cartas**. La capa de su servidor
se llama literalmente «Cartas de Inundación por Tsunami SHOA», su descripción
dice «Actualización SHOA 082018», y **cada polígono trae `fuente = CITSU-SHOA` y
el campo `edicion` con la edición de la carta de la que salió** (de «1ra Ed 2013»
a «1ra Ed 2024»). Es el dato del SHOA, por el canal que corresponde, con la
trazabilidad puesta por la propia fuente.

Esto **resuelve el pendiente «403 del SHOA»** de `FUENTES_DE_DATOS_NACIONALES.md`:
el 403 no había que vencerlo, había que rodearlo por la puerta legítima.

### 2.2 Las dos capas, que no son lo mismo

| | capa 1 · Inundación (CITSU) | capa 2 · Área a evacuar |
|---|---|---|
| servicio | `visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE/Inundacion_Tsunami_2018/MapServer/0` | `services5.arcgis.com/i7S5PSnIJAUcWvSE/.../Amenaza_por_Tsunami_2024/FeatureServer/3` |
| qué dice | **hasta dónde llega el agua y cuán hondo** | **qué se ordena desalojar** |
| polígonos | 5.553 | 350 |
| campo clave | `inundacion` = profundidad en metros por clase: «0 a 1», «1 a 2», «2 a 4», «4 a 6», «6 y más» | `cut` = código único territorial de comuna |
| cobertura | nacional, Arica a Magallanes, incluida Rapa Nui (−109,4° a −67,6° long) | 15 regiones, 97 comunas |
| naturaleza | física | protocolo, con margen operativo |

**No se mezclan y no se sustituyen.** La segunda es más amplia que la primera por
diseño: una zona de evacuación lleva margen de seguridad deliberado. Usar una en
lugar de la otra sería confundir el agua con la orden de arrancar.

Lo importante para la Matriz: **el campo `inundacion` trae magnitud absoluta en
metros**, no una categoría relativa. Es justo lo que
`CORRECCION_RAREZA_PELIGRO.md` señaló que le faltaba a la variable de remoción en
masa. Tsunami es hoy la amenaza mejor medida del consolidador.

Y trae algo que la regla de «confianza declarada» pedía: **la edición de cada
carta**. Una carta de 2013 y una de 2024 no valen lo mismo, y el propio dato lo
dice. La columna `edicion_carta` viaja hasta los CSV de salida.

### 2.3 Licencia

Ninguna de las dos vías declara licencia formal: `licenseInfo` y `copyrightText`
vienen vacíos. Lo único declarado está en la ficha de IDE Chile: *«si utiliza
información geoespacial publicada en las plataformas de IDE Chile, solicitamos
citar a la institución proveedora»*. En la práctica: **dato público, atribución
solicitada**. La cita correcta es doble:

> Cartas de Inundación por Tsunami (CITSU), Servicio Hidrográfico y Oceanográfico
> de la Armada (SHOA); publicadas por el Servicio Nacional de Prevención y
> Respuesta ante Desastres (SENAPRED). Edición de cada carta según el campo
> `edicion` del propio dato.

### 2.4 Descartadas, con motivo

- **MINVU** (`geoide.minvu.cl/.../Amenaza_Tsunami/FeatureServer`): responde HTTP
  200 pero con `{"error":{"code":499,"message":"Token Required"}}`. No sirve.
- **IDE Chile, descarga como archivo**
  (`geoportal.cl/geoportal/catalog/download/1dbd467c-…`): sí funciona (HTTP 200,
  3,5 MB, geodatabase de archivo), pero **está más atrasada que el servicio**:
  294 polígonos de área a evacuar contra 350, y sólo 3 capas. Se prefirió el
  servicio. *(Nota de manejo: el ZIP trae los nombres en CP437 y `unzip` falla;
  hay que extraerlo con el módulo `zipfile` de Python.)*
- **«Áreas a evacuar por tsunami a nivel nacional»** (`services7.arcgis.com/UeyripQFTg6pfUe5`):
  293 polígonos, pero su propio metadato dice «Fuente: ONEMI, 2019». Copia vieja.
- **OpenStreetMap**: no se necesitó, y no habría servido como fuente oficial.
- Subdominios que no resuelven: `geoportal.senapred.gob.cl`,
  `arcgis.senapred.gob.cl`, `geonode.senapred.gob.cl`, `ide.senapred.gob.cl`. El
  host bueno es `visor-grd.senapred.gob.cl`.

---

## 3. Hallazgo lateral: el servidor de SENAPRED tiene mucho más

Explorando `visor-grd.senapred.gob.cl` (39 carpetas, `robots.txt` permisivo)
aparecieron servicios que tocan directamente el objetivo del proyecto y que
**nadie ha revisado todavía**:

- `SIIE/Infraestructuras`, `SIIE/InfraestructuraVulnerable`, `SIIE/Energiav10f`,
  `SIIE/RedComunicaciones`, `SIIE/SISS`, `SIIE/Subtelv10f` — inventario de
  infraestructura crítica **del propio SENAPRED**. Vale la pena compararlo con el
  inventario que usa la Matriz: puede completarlo o contradecirlo, y las dos
  cosas son informativas.
- `Visor_Shakemap/shakemap1..5` — mapas de intensidad instrumental **por evento**.
  Se verificó: son polígonos de un sismo puntual con fecha (el ejemplo mirado era
  del 5 de julio de 2026, 4 polígonos, campo `intens_ins`). **No es zonificación**,
  es un producto operativo vivo. Sirve como canal de reacción, no de planificación.
- `SIIE/sismos_online` — sismos en línea, tres capas (todos / últimos 30 días
  M≥5 / últimos 30 días M<5).
- `SIIE_AMENAZAS/amenazas`, `AMENAZA_INUNDACION/rio_mataquito`, `Observatorio/*`
  (proyección climática, amenaza de incendios, exposición y vulnerabilidad).

Ninguno se integró acá porque quedaba fuera del encargo. Queda anotado.

---

## 4. Qué escribe cada adaptador

### `adaptadores/sismico.py`

```
datos/crudo/sismico/<fecha>/csn_evtdb_eventos.html        crudo de la base de eventos
datos/crudo/sismico/<fecha>/csn_catalogo_diario/*.html    crudo del catálogo diario
datos/crudo/sismico/<fecha>/senapred_sismologia/*.geojson crudo de las capas geográficas
datos/crudo/sismico/<fecha>/_manifiesto.jsonl             url, fecha, tamaño y huella SHA-256

datos/sismico_catalogo_csn.csv          catálogo M≥4
datos/sismico_catalogo_diario_csn.csv   catálogo diario, toda magnitud
datos/sismico_zonas_de_ruptura.csv      las 19 zonas de ruptura, con año y magnitud
datos/sismico_eventos_subestaciones.csv
datos/sismico_eventos_puentes.csv
datos/sismico_eventos_vial.csv
```

### `adaptadores/tsunami.py`

```
datos/crudo/tsunami/<fecha>/citsu_inundacion/pagina_NNN.geojson.gz
datos/crudo/tsunami/<fecha>/areas_a_evacuar/pagina_NNN.geojson.gz
datos/crudo/tsunami/<fecha>/*/_manifiesto.jsonl

datos/tsunami_citsu_cartas.csv       inventario de cartas CITSU y su edición
datos/tsunami_subestaciones.csv
datos/tsunami_puentes.csv
datos/tsunami_vial.csv
datos/tsunami_areas_a_evacuar.csv
```

El crudo de tsunami se guarda **comprimido con gzip**, que es sin pérdida: el
byte a byte que llegó se puede reconstruir, y el manifiesto lleva el SHA-256 del
contenido **sin comprimir** para poder comprobarlo.

## 5. Cómo se mide, para que nadie tenga que adivinar

- **Distancias**: dos pasos. Primero un índice espacial de árbol (`STRtree`)
  descarta los pares imposibles pidiendo «lo que esté a menos de R/60 grados»
  — se divide por 60 y no por 111 a propósito, porque 60 km por grado es una
  **cota inferior** del largo de un grado de longitud en todo Chile, de modo que
  el filtro deja pasar de más y nunca de menos. Después, a cada sobreviviente se
  le mide la distancia **geodésica real sobre el elipsoide WGS84** con
  `pyproj.Geod`. Ése es el número que se guarda.
- **Largos y áreas**: geodésicos, nunca en grados convertidos con regla de tres.
- **Kilómetros de un tramo dentro de una zona**: intersección exacta de la línea
  con el polígono (Shapely/GEOS), no muestreo cada N metros. Un cruce angosto de
  40 metros —una quebrada— se ve.
- **Clase de profundidad de un elemento**: la peor de las que lo tocan, usando el
  orden que declara la propia fuente en sus etiquetas. **No se convierte la
  etiqueta a un número**: se guarda la etiqueta tal cual y, aparte, su posición
  en el orden.
- **Sin dato es sin dato**: si no hay falla activa dentro del radio de búsqueda,
  la celda dice `>200`, no `0`.
