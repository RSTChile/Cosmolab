# Fuente MOP · Vialidad: la red vial de Chile, sus puentes y sus emergencias

**Explorado y bajado el 17-ago-2026** · adaptador: `adaptadores/mop_vialidad.py`

> El piloto del proyecto fueron 39 subestaciones eléctricas, elegidas no por
> importancia sino porque eran lo único georreferenciado que había a mano. Las
> lluvias de julio de 2026 mostraron dónde está de verdad el punto débil de
> Chile: **la vialidad**. El MOP publica su inventario vial completo, abierto,
> sin credenciales, y nadie lo había bajado.

Servicio: <https://rest-sit.mop.gob.cl/arcgis/rest/services/>
Administra: **UGIT** — Unidad de Gestión de Información Territorial, Dirección de
Vialidad, MOP. Responde sin credenciales. `robots.txt` **no existe** (404), o sea
que el host no declara ninguna restricción de rastreo.

---

## 1 · El resumen en cinco líneas

| lo que se bajó | registros | cierre |
|---|---:|---|
| Puentes (con el campo `CAUCE_QUEB`) | **6.742** | exacto |
| Tramos viales con rol y kilometraje | **14.039** | exacto |
| Emergencias MOP históricas 2014-2019 | **1.850** | exacto |
| Emergencias MOP vigentes 2022-2026 | **4.291** | exacto |
| **total emergencias** | **6.141** | exacto |

Los cuatro conteos coinciden **registro por registro** con lo que el propio
servicio declara. Nada se estimó.

Y los dos resultados que importan:

- ★ **El cauce con más puentes encima de Chile es el río Copiapó: 19 puentes**,
  en tres comunas (Caldera, Copiapó, Tierra Amarilla). Le siguen el Cautín (16,
  Araucanía) y el Maipo (15, Metropolitana). Son **puntos de fallo múltiple**: una
  sola crecida se cobra 19 cortes, no uno.
- ⚠️ **La estacionalidad del registro MOP NO confirma de forma limpia el hallazgo
  del proyecto.** El número agrupado sale espectacular (razón invierno/verano
  **8,84** en vialidad) pero se deshace al mirarlo año por año: va de **0,40** en
  2017 a **148,50** en 2023. Lo que oscila no es el clima, es **cuándo se usa el
  registro**. Hay una ventana donde sí se puede comparar, y ahí el resultado es
  bueno — ver §6.

---

## 2 · Catálogo del servicio

El servicio expone **21 carpetas**: `CCOP, DA, DAP, DGA, DGOP, DIRPLAN, DOH, DOP,
DV, EMERGENCIA, GEOPROCESOS, IDE_EXTERNA, IDE_MOP, ILOG, INTEROP, MAPA_BASE,
Pruebas, SEMAT, SNIA, Utilities, VIALIDAD`.

Se recorrieron a fondo las dos que le sirven al proyecto (**VIALIDAD** y
**EMERGENCIA**): **28 servicios MapServer y 78 capas** (22 + 59 en VIALIDAD,
6 + 19 en EMERGENCIA). El catálogo completo —nombre,
geometría, nº de registros y **la lista de campos con tipo y alias de cada una**—
quedó en `datos/crudo/mop/2026-08-17/catalogo_servicio.json` (218 kB). Acá va el
índice.

### Carpeta VIALIDAD

| servicio / capa | geometría | registros | campos |
|---|---|---:|---:|
| `Catastro_Vial/0` PUENTES | Point | 1.573 | 47 |
| `Catastro_Vial/1` CATASTRO SANEAMIENTO | Point | **54.446** | 24 |
| `Catastro_Vial/2` CATASTRO SEÑALES | Point | **72.961** | 24 |
| `Catastro_Vial/3` CATASTRO PARADEROS | Point | 1.157 | 24 |
| `Catastro_Vial/4` SOLERAS | Polyline | 1.150 | 25 |
| `Catastro_Vial/5` CATASTRO LINEA | Polyline | 17.908 | 25 |
| `Catastro_Vial/6` CARPETA | Polyline | 784 | 21 |
| `Catastro_Vial/7-9` CAD CARPETA / SANEAMIENTO / SEGURIDAD | Polyline | 243 / 66 / 66 | 40 |
| `Contratos_Globales_CC/0-1` | Polyline | 4.390 / 150 | 43 / 15 |
| `EGC_y_Control_Pesaje/0-3` pesaje fijo, móvil, empresas de carga, plazas | Point | 37 / 79 / 887 / 116 | 17-57 |
| `Emergencias_Vialidad/0` (emergencias **vigentes**, puntual) | Point | 715 | 40 |
| `Emergencias_Vialidad/1` (vigentes, subconjunto) | Point | 217 | 40 |
| `Emergencias_Vialidad/2` (vigentes, lineal) | Polyline | 498 | 41 |
| `Estado_Red_Vial_Pavimentada/0` Estado | Polyline | **52.739** | 23 |
| `Infraestructura_Vial/0-3` peaje, pesaje, **balsa**, **túnel** | Point/Polyline | 116 / 212 / 24 / 34 | 11-23 |
| `Iniciativas_De_Inversion_Etapas/0-2` | Multipoint/Polyline/Polygon | 765 / 2.017 / 93 | 38-40 |
| `Ordenes_Trabajo_CAD/0-1` | Point / Polyline | 874 / **23.073** | 25 / 26 |
| `Pasos_Fronterizos/0` | Point | 31 | 33 |
| `Pesaje_Rutas_Tipicas/0` | Polyline | 1 | 6 |
| `Plan_Nacional_de_Censos/0-6` (censos de tránsito 2014-2025) | Point | 5.026 en total | 24 |
| `PNC_Equipos_Edicion/0` | Point | 6 | 18 |
| `Puentes_Ruta5/0` Puentes Ruta 5 en Evaluación | Point | 14 | 6 |
| **`Puentes/0` Puentes** ★ **bajado** | Point | **6.742** | 43 |
| `Puntos_Censales_Equipos/0` contadores de tránsito | Point | 141 | 24 |
| `Red_Vial_Chile/0-2` (misma red generalizada) | Polyline | 630 / 1.452 / 10.725 | 20 |
| **`Red_Vial_Chile/3` (1:1.128)** ★ **bajado** | Polyline | **14.039** | 20 |
| `Red_Vial_Chile_/0-3` (variante) | Polyline | 636 / 1.453 / 10.579 / **13.915** | 19 |
| `Red_Vial_Chile_Base_gris/0-3` (variante) | Polyline | 630 / 1.452 / 10.725 / 14.039 | 20 |
| `Red_Vial_Estructurante/0` | Polyline | 724 | 28 |
| `Restricciones_Vialidad/0` estado de pasos fronterizos | Point | 32 | 19 |
| `Solicitudes_Especiales/0` | Polyline | 2 | 22 |
| `Zonas_de_Descanso/0-1` | Point | 41 / 165 | 16 / 27 |

**Trampa detectada.** Hay **tres** servicios de red vial casi idénticos:
`Red_Vial_Chile`, `Red_Vial_Chile_` y `Red_Vial_Chile_Base_gris`. Los tres tienen
cuatro capas que son la misma red a distinta generalización (1:5.000.000 →
1:1.128). En la capa de máximo detalle, `Red_Vial_Chile` y `Base_gris` dan
**14.039** y `Red_Vial_Chile_` da **13.915** — 124 tramos de diferencia, y 19
campos en vez de 20. **Qué versión es la buena no está documentado en el
servicio.** Se tomó `Red_Vial_Chile/3` porque es el nombre sin sufijo y coincide
con el conteo mayoritario; queda anotado como pregunta para la UGIT. Elegir la
capa 0, 1 o 2 por descuido habría dado un inventario de 630, 1.452 o 10.725
tramos creyendo tener el país entero.

### Carpeta EMERGENCIA

| servicio / capa | geometría | registros | campos |
|---|---|---:|---:|
| `CONTRATOS_GLOBALES_DV/0-8` (contratos por región) | Polyline | 3.048 en total | 9 |
| **`EMERGENCIA_HISTORICA_MOP/0`** ★ **bajado** | Point | **1.850** | 13 |
| **`EMERGENCIA_MOP/0`** ★ **bajado** | Point | **4.291** | 24 |
| `Emergencia_Mayo/0-1` maquinaria y cuadrillas | Point | 8 / 36 | 9 / 7 |
| `Emergencia_Mayo/2` Emergencia SIEMOP Histórica | Multipoint | 500 | 8 |
| `Emergencia_Mayo/3` **Amenaza de Inundación** ★★ | **Polygon** | **6.066** | 8 |
| `EMERNORTE/0-2` emergencia norte puntual/lineal/poligonal | Point/Polyline/Polygon | 1.336 / 135 / 23 | 21-23 |
| `MAPA_ESTACIONES_DGA/0` | Point | 4 | 22 |

---

## 3 · Cómo se pagina un servicio de 2013 (y por qué falla en silencio)

El servidor corre **ArcGIS 10.2.1**. Tres cosas que hay que saber o los datos
salen truncados sin que nadie se dé cuenta:

1. **`resultOffset` no existe — y no da error.** Se probó: el servidor acepta
   `resultOffset=5&resultRecordCount=2` y devuelve el lote completo desde el
   principio, ignorando ambos parámetros. Un paginador escrito para ArcGIS
   moderno bajaría el primer lote 15 veces y reportaría 15.000 registros
   felizmente. **Falla callado, que es la peor forma de fallar.**
   → La paginación se hace **por identificador**: `returnIdsOnly=true` devuelve
   la lista completa de `OBJECTID` (el tope de `maxRecordCount=1000` no aplica a
   identificadores), y después se piden los registros en lotes con
   `objectIds=...`. Así el conteo cierra por construcción y se sabe exactamente
   qué falta.
2. **`f=geojson` no existe.** `supportedQueryFormats` es `JSON, AMF`. El servicio
   entrega esriJSON y la conversión a GeoJSON la hace el adaptador,
   explícitamente (`{x,y}`→`Point`, `{paths}`→`LineString`/`MultiLineString`).
   Se pide `outSR=4326` y la reproyección desde el EPSG:5360 nativo
   (SIRGAS-Chile / UTM 19S) la hace el servidor.
3. **Con 250 identificadores, un GET devuelve 404** por largo de URL. Hay que
   usar **POST**. Otro fallo mudo.

Además, el servicio **se cae de a ratos**: `read operation timed out` y
`code 500 Error performing query operation`, de forma intermitente y a veces
reproducible sobre el mismo lote. El adaptador lo maneja con tres cosas:

- **Reintentos con espera creciente** (6 intentos, 3→96 s).
- **Bisección.** Si un lote falla tras los reintentos, se parte en dos y se
  reintenta cada mitad, hasta aislar el registro culpable. Comprobado: el lote 56
  de los tramos (250 registros) fallaba con `code 500` de forma reproducible, y
  al partirlo en 125+125 **las dos mitades pasaron**. Era un problema de tamaño
  de respuesta, no un registro roto. Si hubiera sido un registro roto, quedaría
  anotado por identificador en el crudo y en la metadata del GeoJSON, como hueco
  declarado — nunca rellenado.
- **Reanudación.** Cada lote se guarda comprimido en cuanto llega. Una corrida
  interrumpida se retoma leyendo de disco los lotes que ya están, y sólo pide los
  que faltan. Sin esto, un corte en el lote 56 de 57 obliga a bajar los 14.039
  tramos otra vez — castigando a un servidor público por un problema nuestro.

**Ritmo:** lotes de 250 (tramos, que pesan ~26 kB por registro) o 500 (puntos),
pausa de 1,2 s entre pedidos, `Accept-Encoding: gzip` (baja el tráfico ~3,2×:
2,0 MB de red por 6,6 MB de JSON) y `User-Agent` identificando el proyecto.

---

## 4 · Lo que se bajó, y dónde está

### Crudo — `datos/crudo/mop/2026-08-17/` (280 MB)

```
catalogo_servicio.json            218 kB   catálogo completo con campos
puentes/lote_000..013.json.gz              respuestas del servidor, sin tocar
puentes.geojson                   7,2 MB
tramos/lote_000..056.json.gz               (57 lotes)
tramos.geojson.gz                 125 MB   comprimido: en texto plano son ~370 MB
emergencias_historicas/…                   (4 lotes)
emergencias_historicas.geojson    813 kB
emergencias_vigentes/…                     (9 lotes)
emergencias_vigentes.geojson      3,4 MB
```

Cada `lote_NNN.json.gz` contiene **la respuesta esriJSON tal como la mandó el
MOP**, más los identificadores que se pidieron y los que el servicio no pudo
entregar. Cada `.geojson` lleva un bloque `metadata` con la URL exacta del
servicio, la fecha, el conteo declarado, el conteo recibido, los dos sistemas de
coordenadas y la licencia. Si mañana se descubre que este adaptador interpretó
mal un campo, el dato original sigue ahí, auditable.

### Tablas — `datos/`

| archivo | filas | columnas | tamaño |
|---|---:|---:|---:|
| `mop_puentes.csv` | 6.742 | 27 | 1,4 MB |
| `mop_tramos.csv` | 14.039 | 21 | 3,1 MB |
| `mop_emergencias_viales.csv` | 6.141 | 24 | 1,8 MB |

**Comuna resuelta por geometría**, no copiada de un campo: cada punto se metió en
los polígonos comunales con `territorio.py` (345 comunas, algoritmo del rayo, sin
geopandas). Resultado: **6.732 de 6.742 puentes** y **6.133 de 6.141
emergencias** caen dentro de una comuna. Los 18 que no, quedan marcados
`punto_fuera_de_toda_comuna` — no se les asignó la comuna más cercana.

**`mop_tramos.csv` — dos avisos de lectura.**
`KM_I`, `KM_F` y `KM_TRAMO` vienen **en metros** pese al nombre (verificado:
KM_I=14474, KM_F=17244, KM_TRAMO=2770 en un tramo de 2,77 km). Se guardan tal
cual y además en kilómetros, con el nombre correcto. Y `lat_punto_medio` /
`lon_punto_medio` es el **vértice del medio del trazo más largo**, no un
centroide: en una carretera curva el centroide cae fuera del camino. Un tramo
largo cruza varias comunas, así que ese punto dice *por dónde pasa el medio*, no
*dónde está el tramo*. No se le asignó comuna al tramo por eso.

Suma de largos declarados: **81.446 km** de red, en 8.150 roles y 8.747 códigos de
camino, las 16 regiones del país (el campo `REGION` trae 17 cadenas distintas
porque Los Lagos aparece escrito de dos maneras, `Región de Los Lagos` y
`Región de Los lagos`; no se corrigió a mano, se anota).
Reparto por carpeta: Ripio 4.861 · Pavimento Básico 4.083 ·
Pavimento 2.412 · Suelo Natural 1.488 · Grava Tratada 585 · Doble Calzada 381 ·
Sin Información 228. Sólo **197 tramos concesionados** de 14.039.

### Cobertura real de los campos de puentes (lo que está vacío también es dato)

| campo | con dato | |
|---|---:|---|
| `CODIGO_PUENTE`, `NOMBRE_PUENTE`, `REGION`, `M_INICIO`, `TIPO_ESTRUCTURA` | 100 % | |
| `PROVINCIA` | 6.739 / 6.742 | |
| `LARGO` | 91 % | suma 183,6 km de puentes; el más largo 2.300 m |
| `ANCHO_TOTAL` | 88 % | |
| `ROL` | 73 % | |
| **`CAUCE_QUEB`** | **53 %** | ver §5 |
| `TUICION` | 41 % | |
| `MAT_VIGAS` | 17 % | |
| `AÑO_CONTRUCCION` | **10 %** (655 de 6.742) | rango 1902-2025 |
| `EST_PUENTE` (estado del puente) | **0 %** | ★ el campo existe y está **vacío en los 6.742** |
| `INFRA_TIPO` | **0 %** | ídem |
| `AÑO` | **0 %** | ídem |

★ **`EST_PUENTE` vacío es el hueco más caro de esta fuente.** El campo que diría
en qué estado está cada puente existe en el esquema y no tiene un solo valor. Sin
él, la fuente dice *dónde hay un puente*, no *qué puente está por caerse*. La
edad tampoco ayuda: sólo 1 de cada 10 declara año de construcción. **La
fragilidad estructural del puente chileno no está en este servicio.**

Puentes por región (código INE): 09 Araucanía **1.536** · 10 Los Lagos 1.098 ·
08 Biobío 787 · 07 Maule 706 · 14 Los Ríos 542 · 11 Aysén 425 · 16 Ñuble 378 ·
06 O'Higgins 376 · 05 Valparaíso 261 · 13 Metropolitana 241 · 04 Coquimbo 153 ·
12 Magallanes 100 · 03 Atacama 85 · 02 Antofagasta 25 · 15 Arica 17 ·
01 Tarapacá 12. (El campo `REGION` de puentes trae **el código numérico**, no el
nombre, al contrario que en tramos.)

---

## 5 · ★ `CAUCE_QUEB`: unir «el río se desbordó» con «qué había encima»

Este es el campo que justifica la fuente entera. Por cada puente, el catastro
declara **el nombre del río o quebrada que cruza**:

```
«el río Copiapó se desbordó»  ──CAUCE_QUEB──▶  «19 puentes estaban encima»
```

Sin él, una crecida es una noticia. Con él, es una lista de activos. Y aparece
algo que ninguna de las dos capas dice por separado: **un cauce con muchos
puentes es un punto de fallo múltiple**. Los cortes no se suman, se multiplican,
porque cada uno elimina la ruta alternativa del otro. Es exactamente el mecanismo
que el catálogo de modos de falla del proyecto describe para `redundancia`: *«lo
único que separa un atraso de un aislamiento»*.

### Primero, dos trampas que hay que esquivar o el ranking miente

**Trampa 1 — los marcadores de «no sé» no son cauces.** 443 puentes tienen en
`CAUCE_QUEB` un texto que no nombra nada: `S/I` (121), `DEFINIR REGION` (111),
`CAUCE SIN IDENTIFICACION` (64), `SIN NOMBRE` (33), `ESTERO S/N` (25), y siete
variantes más. Contarlos produce el disparate de que el cauce con más puentes de
Chile sea «S/I». Se apartan en una lista declarada (`NO_SON_CAUCES` en el
adaptador) y se cuentan como lo que son: huecos.

El balance honesto del campo, entonces:

| | puentes | |
|---|---:|---|
| `CAUCE_QUEB` vacío | **3.167** | 47,0 % |
| con un marcador de «no sé» | 443 | 6,6 % |
| **con un cauce realmente nombrado** | **3.132** | **46,5 %**, en 1.779 cauces distintos |

**Menos de la mitad de los puentes de Chile dice qué cruza.** El cruce
río↔puente que esta fuente hace posible, lo hace posible **para la mitad del
inventario**. Es un resultado utilizable y es una limitación grande; las dos
cosas a la vez.

**Trampa 2 — los homónimos.** «Río Claro» sale 28 veces… repartidas en **seis
regiones**. No es un río con 28 puentes: son varios ríos distintos que se llaman
igual. Una crecida del Río Claro del Maule no toca el Río Claro de Aysén.
Agrupar por nombre a secas **inventa un punto de fallo múltiple que no existe**.
El ranking que sirve para riesgo es el de **cauce dentro de una región**.

*(No se normalizó quitando la palabra «RIO», «ESTERO» o «QUEBRADA»: «Estero Las
Cruces» y «Río Las Cruces» pueden ser dos cursos de agua distintos, y fundirlos
sería el mismo error al revés.)*

### El ranking bueno: cauce × región

| cauce | región | puentes | comunas |
|---|---|---:|---|
| **RIO COPIAPO** | 03 Atacama | **19** | Caldera, Copiapó, Tierra Amarilla |
| **RIO CAUTIN** | 09 Araucanía | **16** | Curacautín, Lautaro, Nueva Imperial… |
| **RIO MAIPO** | 13 Metropolitana | **15** | Buin, Isla de Maipo, Melipilla… |
| **RIO ELQUI** | 04 Coquimbo | **14** | La Serena, Vicuña |
| **RIO ACONCAGUA** | 05 Valparaíso | **13** | Calera, Catemu, Concón… |
| RIO NEGRO | 10 Los Lagos | 12 | Ancud, Chaitén, Frutillar… |
| RIO HUASCO | 03 Atacama | 11 | Alto del Carmen, Freirina, Huasco… |
| RIO GRANDE | 04 Coquimbo | 11 | Monte Patria, Ovalle |
| RIO MAULE | 07 Maule | 11 | Colbún, Constitución, San Clemente |
| RIO ALLIPEN | 09 Araucanía | 11 | Cunco, Freire, Melipeuco |
| RIO LEUFUCADE | 14 Los Ríos | 11 | Lanco, Panguipulli |
| RIO HURTADO | 04 Coquimbo | 10 | Andacollo, Ovalle, Río Hurtado |
| ESTERO SAN FRANCISCO | 05 Valparaíso | 10 | San Esteban, San Felipe, Santa María |
| RIO CLARO | 06 O'Higgins | 10 | Malloa, Rengo, San Fernando… |
| RIO BLANCO | 10 Los Lagos | 10 | Chaitén, Cochamó, Fresia… |
| RIO RAHUE | 10 Los Lagos | 10 | Ancud, Fresia, Osorno… |
| RIO LOA | 02 Antofagasta | 9 | Calama, Iquique, María Elena |
| ESTERO CHIMBARONGO | 06 O'Higgins | 9 | Chimbarongo, Chépica, Nancagua… |
| RIO CLARO | 07 Maule | 9 | Molina, Romeral, San Clemente… |
| RIO QUINO | 09 Araucanía | 9 | Melipeuco, Traiguén, Victoria |
| ESTERO POCURO | 05 Valparaíso | 8 | Calle Larga, Rinconada, San Felipe |
| RIO MAPOCHO | 13 Metropolitana | 8 | El Monte, Lo Barnechea, Maipú… |
| RIO LINGUE | 14 Los Ríos | 8 | Loncoche, Mariquina |
| ESTERO TRAIGUEN | 14 Los Ríos | 8 | La Unión |
| RIO TURBIO | 04 Coquimbo | 7 | Vicuña |

Nombres que se repiten en varias regiones y por eso **no** deben agruparse:
`RIO CLARO` (6 regiones), `RIO BLANCO` (6), `RIO GRANDE` (5), `RIO TURBIO` (3),
`ESTERO LAS PALMAS` (3), `RIO CRUCES` (3), `RIO MAIPO` (2), `RIO NEGRO` (2),
`RIO MAULE` (2), `RIO RAHUE` (2), `RIO BIOBIO` (2).

*(Detalle sucio de origen, para tenerlo presente: el Biobío está escrito de dos
maneras, `RIO BIOBIO` (9 puentes) y `RIO BIO BIO` (8). Sumados son 17. No se
fusionaron a mano porque abriría la puerta a fusionar otras cosas por parecido
ortográfico; queda anotado como el caso donde la normalización conservadora
subcuenta.)*

### Por qué el río Copiapó, encabezando esto, no es una casualidad

El proyecto ya se había topado con Copiapó por otro lado: **el ancla de validación
de Copiapó falló** en la noche del 15-16 de agosto, y ese fallo fue lo que reveló
que la variable medía *rareza* y no *peligro*. Ahora la vialidad del MOP pone al
río Copiapó primero en el ranking de fallo múltiple, y la concentración de
emergencias viales pone **Tierra Amarilla (162) y Copiapó (134) en los dos
primeros lugares del país** (§7). Tres caminos independientes apuntando al mismo
valle. No es prueba de nada por sí solo, pero es el lugar obvio donde probar el
cruce completo cuando se arme.

---

## 6 · Estacionalidad: qué confirma el MOP y qué no

**Lo que el proyecto había medido**, en el registro de SENAPRED
(`CATALOGO_MODOS_DE_FALLA.md`, §4.2-4.3):

| grupo | razón invierno/verano |
|---|---:|
| falla de infraestructura, causa **meteorológica** | **6,84** (n=1.255) |
| falla de infraestructura, causa **NO** meteo (control) | **0,84** (n=4.031) |
| línea base del registro completo | 0,93 (n=50.457) |
| **conectividad vial**, causa meteo | **3,03** (n=386) |
| **conectividad vial**, control | **0,89** (n=3.729) |

*(El encargo citaba 8,41 contra 0,87; los números que el proyecto tiene escritos
son 6,84 contra 0,84. Se usan estos últimos porque son los que están en el
documento y se pueden rastrear. La diferencia no cambia ninguna conclusión: el
patrón es el mismo.)*

### El resultado bruto del MOP, que se ve demasiado bien

Razón invierno (JJA) / verano (DEF), sobre las 6.127 emergencias con fecha:

```
mes:      1     2     3     4     5     6     7     8     9    10    11    12
todo:   149   210   248   158   346  1391  1694  1425   252   118    65    71   → 10,49
vial:   132   176   194    91   324   998  1197  1068   107   117    65    61   →  8,84
```

Y desglosado por lo que dice el texto libre del registro:

| grupo (heurística sobre texto libre) | n | razón |
|---|---:|---:|
| vialidad · el texto menciona lluvia/crecida/nieve/caudal… | 1.879 | **13,01** |
| vialidad · el texto menciona derrumbe/rodado/deslizamiento | 748 | **12,17** |
| vialidad · **control**: el texto no menciona ninguna de las dos | 1.903 | **5,81** |

**Acá se cae.** En SENAPRED el control quedaba en 0,84 — pegado a la línea base —
y eso era la prueba de que el exceso invernal era del clima y no del registro.
Acá el control da **5,81**. Un control que se comporta como el tratamiento no
valida nada.

### La prueba que lo desarma: la misma razón, año por año

| año | capa | n | JJA | DEF | razón |
|---|---|---:|---:|---:|---:|
| 2014 | histórica | 2 | 0 | 1 | 0,00 |
| 2015 | histórica | 529 | 276 | 5 | **55,20** |
| 2016 | histórica | 189 | 58 | 91 | **0,64** |
| 2017 | histórica | 455 | 49 | 121 | **0,40** |
| 2018 | histórica | 155 | 92 | 21 | 4,38 |
| 2019 | histórica | 190 | 32 | 52 | **0,62** |
| 2022 | vigente | 3 | 3 | 0 | sin dato |
| 2023 | vigente | 950 | 891 | 6 | **148,50** |
| 2024 | vigente | 694 | 624 | 24 | 26,00 |
| 2025 | vigente | 111 | 21 | 22 | **0,95** |
| 2026 | vigente | 1.252 | 1.217 | 26 | 46,81 |

**De 0,40 a 148,50.** Un indicador que oscila 370 veces entre años del mismo
registro no está midiendo el clima: está midiendo **cuándo se usa el registro**.

La explicación es simple y es estructural, no un error de nadie: **el registro de
emergencias del MOP es reactivo**. Se llena a mano cuando hay una emergencia
declarada y queda casi vacío el resto del año. En un año de sistemas frontales
grandes (2015, 2023, 2024, 2026) casi todo lo anotado es de invierno; en un año
tranquilo (2016, 2017, 2019, 2025) la razón cae por debajo de 1. Además el
registro *no contiene* una línea base de fallas viales no climáticas —los
choques, el desgaste, las obras— porque esas no son «emergencias MOP». **Sin
denominador no hay tasa**, y es el mismo error que el propio catálogo del proyecto
advierte para `ExpEstr`.

Y hay una segunda razón por la que el control de 5,81 no es un control: mirando
los textos etiquetados `otra_o_no_dice` aparecen *«Camino Cortado»*, *«camino
cortado»*, *«Puente Sin Nombre cortado»*, *«derumbe camino T-450»*. **Son eventos
climáticos que el funcionario no escribió como climáticos.** El grupo de control
está contaminado con el tratamiento. La heurística de texto no puede separarlos
porque el dato de causa **no existe** en esta fuente: no hay campo de causa, sólo
texto libre escrito a mano por el turno de guardia.

### La ventana donde sí se puede comparar — y ahí el MOP sí confirma

Los cuatro años calendario completos del registro **histórico** (2015-2018) son
la única ventana de esta fuente donde el conteo no lo domina un evento puntual:

| grupo, vialidad 2015-2018 | n | JJA | DEF | razón |
|---|---:|---:|---:|---:|
| texto meteo/hídrico | 456 | 190 | 63 | **3,02** |
| texto de remoción en masa | 195 | 100 | 26 | **3,85** |
| **control** (no menciona ninguna) | 677 | 185 | 149 | **1,24** |

Comparado con lo que el proyecto midió en SENAPRED **para el mismo modo de
falla**, en fuentes independientes y con métodos distintos:

| | causa meteo | control |
|---|---:|---:|
| SENAPRED · conectividad vial | **3,03** | 0,89 |
| MOP Vialidad · 2015-2018 | **3,02** | 1,24 |

**3,03 contra 3,02.** El lado del tratamiento coincide con una precisión que no
se pidió. El lado del control queda peor en el MOP (1,24 contra 0,89), y eso es
esperable por la contaminación descrita: parte del «control» del MOP son eventos
climáticos sin etiquetar, que lo empujan hacia arriba.

**Veredicto, y hay que leerlo entero:**

> La vialidad del MOP **confirma la dirección y la magnitud** del hallazgo del
> proyecto en la ventana 2015-2018 (razón meteo ≈ 3,0 para conectividad vial,
> idéntica a SENAPRED). **No confirma** —y no puede— la limpieza del control, ni
> sirve para citar una razón agrupada: el registro vigente 2022-2026 es reactivo
> y su estacionalidad refleja qué años hubo temporal, no cuánto más frágil es el
> invierno. **Esta fuente sirve para saber QUÉ se rompió y DÓNDE. Para CUÁNDO,
> sólo la ventana histórica, y siempre con la tabla año por año al lado.**

*(La etiqueta de causa es una **heurística sobre texto libre** —raíces de palabra
con frontera por delante: `\blluvia` atrapa «lluvias», `\brio\b` no atrapa
«sanitario»— y está declarada como tal en la columna `causa_heuristica` del CSV.
No es una clasificación oficial. El MOP no publica campo de causa.)*

---

## 7 · Concentración territorial

De las 4.544 emergencias viales, **todas** tienen comuna resuelta por geometría,
repartidas en **299 comunas**. Las 20 primeras acumulan el **29,4 %**:

| # | comuna | n | acum. | | # | comuna | n | acum. |
|---:|---|---:|---:|---|---:|---|---:|---:|
| 1 | **Tierra Amarilla** | 162 | 3,6 % | | 11 | Cauquenes | 59 | 19,9 % |
| 2 | **Copiapó** | 134 | 6,5 % | | 12 | Combarbalá | 58 | 21,2 % |
| 3 | San Pedro de Atacama | 81 | 8,3 % | | 13 | Cisnes | 52 | 22,3 % |
| 4 | Linares | 78 | 10,0 % | | 14 | Nacimiento | 52 | 23,5 % |
| 5 | Alto del Carmen | 67 | 11,5 % | | 15 | Punitaqui | 48 | 24,5 % |
| 6 | Canela | 67 | 13,0 % | | 16 | Vallenar | 46 | 25,6 % |
| 7 | Ovalle | 67 | 14,4 % | | 17 | San José de Maipo | 44 | 26,5 % |
| 8 | Diego de Almagro | 66 | 15,9 % | | 18 | Los Ángeles | 44 | 27,5 % |
| 9 | San Clemente | 63 | 17,3 % | | 19 | Illapel | 43 | 28,4 % |
| 10 | Monte Patria | 61 | 18,6 % | | 20 | Petorca | 42 | 29,4 % |

El mapa es **de valle transversal y precordillera semiárida**: Atacama (Tierra
Amarilla, Copiapó, Alto del Carmen, Diego de Almagro, Vallenar) y Coquimbo
(Canela, Ovalle, Monte Patria, Combarbalá, Punitaqui, Illapel) copan 11 de los 20
primeros lugares. Es coherente con lo que el proyecto ya había encontrado en
SENAPRED para la causa meteorológica —«cordillerano, austral y de valle
transversal», con Vicuña alto en la lista— y **no** con el eje urbano central que
domina la causa no climática.

**Advertencia obligatoria, la misma de siempre:** son conteos **absolutos, sin
normalizar por longitud de red, por población ni por número de puentes**. Tierra
Amarilla encabeza en parte porque tiene mucho camino de ripio en quebrada, y en
parte porque su dirección regional anota mucho. **Leer este cuadro como ranking
de fragilidad sería repetir el error de `ExpEstr`.** Para normalizarlo hace falta
cruzar con `mop_tramos.csv` por comuna, que todavía no se hizo porque un tramo
cruza varias comunas (§9).

Qué se rompe, en las emergencias viales: **Carpeta de Rodadura 2.528** ·
**Puente 422** · Otro Elemento 317 · Elementos de Saneamiento 175 · Pavimento de
acceso 145 · Elementos de Seguridad 15 · Pasarela 8. Gravedad declarada:
Moderado 2.473 · Grave 1.778 · Leve 1.483 · Muy Grave 401.

---

## 8 · Condiciones de uso — ★ leer antes de publicar cualquier derivado

**Licencia: Creative Commons Atribución-NoComercial (CC BY-NC). NO es dominio
público.** Verificado el 17-ago-2026 en el Portal de Datos Abiertos del Estado:

- <https://datos.gob.cl/organization/direccion_de_vialidad> — la Dirección de
  Vialidad publica **4 conjuntos de datos** y la faceta de licencia muestra
  **«Creative Commons Non-Commercial (Cualquiera)» para los 4**, sin excepción.
  Los cuatro son: *Puentes* (Arica y Parinacota, Antofagasta, Atacama, Tarapacá)
  y *Red Vial Nacional* (dos entradas).
- <https://datos.gob.cl/dataset/28413/resource/e5fd138d-a640-485e-b9b7-216e71697cba>
  — «Red Vial Nacional», licencia **CC BY-NC 2.0**.
- El propio servicio declara `copyrightText`: **«Dirección de Vialidad, Ministerio
  de Obras Públicas»** (red vial) y **«Dirección de Vialidad, MOP»** (puentes).
  Las emergencias declaran **«Ministerio de Obras Públicas»**.

**Qué implica para este proyecto:**

| | |
|---|---|
| ✅ Uso de investigación, no comercial | encaja en la licencia |
| ✅ Insumo a SENAPRED / COGRID, gratuito | encaja |
| ⚠️ **Atribución obligatoria** | «Dirección de Vialidad, MOP» en TODO derivado, mapa, tabla o informe |
| ❌ Incorporarlo a un producto comercial | **prohibido por la licencia** |
| ❌ Redistribuir el crudo sin la nota de licencia | no |

### ★ Por qué hay que confirmarlo con la UGIT, y no es un trámite formal

**La entrada de datos.gob.cl que declara la licencia apunta a otro servicio.** El
recurso licenciado como CC BY-NC señala
`https://wservice-sit.mop.gov.cl/ArcGIS/rest/services/DV/red_caminera_de_chile/MapServer`
con **última actualización el 23-dic-2013**. Nosotros bajamos de
`https://rest-sit.mop.gob.cl/arcgis/rest/services/VIALIDAD/Red_Vial_Chile/`, que
es un endpoint distinto y vivo. Es razonable suponer que la licencia del dato
sigue al dato y no a la URL —y el `copyrightText` del servicio nuevo nombra al
mismo titular— **pero es una suposición, y este proyecto no publica sobre
suposiciones.**

Y las capas de **emergencias** (`EMERGENCIA/EMERGENCIA_MOP`,
`EMERGENCIA_HISTORICA_MOP`) **no tienen entrada en datos.gob.cl en absoluto**. Su
licencia es, hoy, **sin dato**. Se usan bajo el mismo criterio académico con que
el director autorizó el uso del CSN, y queda declarado.

> **Pendiente para Alexis** (es un trámite del director, no del proyecto):
> pedir por escrito a la **UGIT — Unidad de Gestión de Información Territorial,
> Dirección de Vialidad, MOP** la confirmación de que (a) `rest-sit.mop.gob.cl`
> está cubierto por la misma CC BY-NC, (b) qué licencia tienen las capas de
> emergencias, y (c) si el uso como insumo a SENAPRED cuenta como no comercial.
> Es la misma salvedad anotada para el CSN: **el día que el instrumento pase de
> investigación a apoyo operativo, el encuadre académico podría dejar de
> cubrirlo.** Pedirlo ahora es barato; rehacer el trabajo después, no.

Sobre el acceso automatizado: `rest-sit.mop.gob.cl/robots.txt` **devuelve 404**,
o sea que el host no declara restricción de rastreo. El servicio es un REST
público sin credenciales, pensado para ser consumido por programas. Aun así se
bajó con lotes chicos, pausa de 1,2 s, gzip y `User-Agent` identificado.

---

## 9 · Limitaciones — lo que esta fuente NO puede hacer

1. **47 % de los puentes no dice qué cauce cruza** (más 6,6 % con un marcador de
   «no sé»). El cruce río↔puente funciona para la mitad del inventario.
2. **`EST_PUENTE` está vacío en los 6.742 registros** y sólo el 10 % declara año
   de construcción. La fuente dice dónde hay un puente, **no qué puente está por
   caerse**. La fragilidad estructural hay que buscarla en otra parte.
3. **Hueco de 2 años y medio en las emergencias.** Verificado sobre las fechas
   ya convertidas: la capa histórica va del **13-feb-2014 al 23-dic-2019** y la
   vigente del **13-jun-2022 al 13-ago-2026**. **No hay un solo registro entre
   enero de 2020 y mayo de 2022.** Cualquier conteo anual tiene que saltarse ese
   tramo o mentirá. (Y la serie empieza en 2014, no en 2015 como suele decirse.)
   14 de las 6.141 emergencias no traen fecha y quedan fuera de todo cálculo
   temporal, declaradas.
4. **El registro de emergencias es reactivo, no un muestreo.** Ver §6. No sirve
   para medir tendencia ni para citar una razón invierno/verano agrupada.
5. **No hay campo de causa.** Sólo texto libre escrito a mano. La columna
   `causa_heuristica` es nuestra, es una heurística y está marcada como tal.
6. **Los tramos no tienen comuna.** Un tramo lineal cruza varias; asignarle una
   sería inventar. El CSV trae un punto representativo (vértice medio) y la
   región declarada por el MOP, nada más. Repartir kilómetros de red por comuna
   necesita intersección línea↔polígono, que `territorio.py` todavía no hace
   (hoy sólo resuelve punto en polígono).
7. **Tres servicios de red vial con conteos distintos** (14.039 / 13.915 /
   14.039) y ninguna nota que diga cuál es el vigente.
8. **Suciedad de origen que se dejó tal cual, declarada:** `GRAVEDAD` contiene 5
   registros con un nombre de región en vez de un nivel (`REGIÓN DE ÑUBLE`,
   `REGIÓN DE O'HIGGINS`) y uno con `Operativo`; `ELEMENTO` trae `Puente` y
   `Puente\r\n` como valores separados; el Biobío está escrito `RIO BIOBIO` y
   `RIO BIO BIO`; Los Lagos aparece como `Región de Los Lagos` y
   `Región de Los lagos`. No se corrigió a mano nada de esto: se anota.
9. **`REGION` significa distinto según la capa**: código numérico en puentes
   (`09`), nombre completo en tramos (`Región de La Araucanía`). No cruzarlos sin
   traducir.
10. **Fechas.** El servicio entrega epoch UTC en milisegundos. Se convierte a hora
    de Chile con un desfase **fijo de −4 h**; en horario de verano el país corre a
    −3 h, así que un evento de entre 00:00 y 01:00 puede quedar asignado al día
    anterior. No mueve ningún conteo mensual salvo en el borde exacto del mes.

---

## 10 · Lo que queda pendiente, en orden

**Del cruce que este proyecto existe para hacer:**

1. **Cruzar `CAUCE_QUEB` con la DGA.** El eslabón que falta es el caudal: la DGA
   publica estaciones fluviométricas por cauce. Unir «19 puentes sobre el río
   Copiapó» con «el caudal del Copiapó hoy» cierra el circuito completo
   amenaza→exposición, para el 46,5 % de los puentes que sí declaran cauce.
2. **Cruzar los puentes con el FEN dinámico de SERNAGEOMIN** (remoción en masa,
   3 niveles por zona con vigencia). Es el otro modo de falla del puente de
   montaña, y SERNAGEOMIN ya lo publica.
3. **Reemplazar las 39 subestaciones por la vialidad como piloto**, empezando por
   el valle de Copiapó, donde tres indicadores independientes coinciden (§5).
4. **Repartir la red por comuna.** Necesita intersección línea↔polígono en
   `territorio.py`. Es lo que permitiría normalizar el cuadro de §7 y dejar de
   leer conteos absolutos.

**Capas de este mismo servicio que hay que bajar, y no se bajaron:**

| capa | registros | por qué importa |
|---|---:|---|
| ★★ `EMERGENCIA/Emergencia_Mayo/3` **Amenaza de Inundación** | **6.066 polígonos** | Es una capa de **amenaza**, no de activos: el MOP ya tiene zonificado dónde se inunda. Cruzarla con los 14.039 tramos es el producto del proyecto, servido en bandeja. **Máxima prioridad.** |
| ★ `VIALIDAD/Estado_Red_Vial_Pavimentada/0` | **52.739** | Es el **estado** de la red, que es justo lo que `EST_PUENTE` no da para los puentes. Puede ser el proxy de fragilidad que falta. |
| ★ `VIALIDAD/Catastro_Vial/1` CATASTRO SANEAMIENTO | **54.446** | Alcantarillas, badenes, cunetas: **los elementos que fallan cuando llueve**. El registro de emergencias culpa 175 veces a «Elementos de Saneamiento». |
| `EMERGENCIA/EMERNORTE/0-2` | 1.336 + 135 + 23 | Emergencias del norte que **no** están en las dos capas bajadas. Podrían tapar parte del hueco 2020-2022. |
| `VIALIDAD/Infraestructura_Vial/2-3` Balsa y Túnel | 24 + 34 | Cuellos de botella sin ruta alternativa por definición: una balsa cortada aísla, siempre. |
| `VIALIDAD/Red_Vial_Estructurante/0` | 724 | El MOP ya marcó **qué caminos dan conectividad a centros poblados**. Es una capa de criticidad hecha por quien sabe. |
| `VIALIDAD/Pasos_Fronterizos/0` + `Restricciones_Vialidad/0` | 31 + 32 | El estado de los pasos es un dato **operativo y vivo**, actualizado. |
| `Catastro_Vial/0` PUENTES (1.573) | 1.573 | Un catastro de puentes **distinto** al de 6.742, con 47 campos. Hay que ver qué trae de más. |

**Preguntas para la UGIT, además de la licencia:**

- ¿Cuál de los tres `Red_Vial_Chile*` es el vigente?
- ¿Por qué `EST_PUENTE`, `INFRA_TIPO` y `AÑO` están vacíos en las 6.742 filas?
  ¿Existen en otro servicio?
- ¿Qué pasó con las emergencias entre enero de 2020 y mayo de 2022?

---

## Cómo reproducirlo

```bash
cd infraestructura
PY=../.venv-esa/bin/python
$PY adaptadores/mop_vialidad.py --explorar   # catálogo del servicio (~10 min)
$PY adaptadores/mop_vialidad.py --bajar      # crudo (~35 min, reanudable)
$PY adaptadores/mop_vialidad.py --procesar   # los 3 CSV + todo el análisis de arriba
```

`--bajar` es **reanudable**: si se corta, se vuelve a correr y sólo pide los
lotes que faltan. Se puede bajar una capa sola con `--capa puentes`.

---

*Datos: **Dirección de Vialidad, Ministerio de Obras Públicas de Chile**
(SIT-MOP / UGIT), bajados el 17-ago-2026. Licencia CC BY-NC — uso no comercial,
atribución obligatoria. Confirmación con la UGIT pendiente.*
