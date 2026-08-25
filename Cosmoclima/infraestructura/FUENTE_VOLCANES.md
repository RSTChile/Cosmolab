# Fuente: amenaza volcánica (SERNAGEOMIN — RNVV / OVDAS)

**19-ago-2026.** Adaptador: `adaptadores/volcanes.py`.
Todo lo que dice este documento está verificado con petición real ese día. Lo que
no se pudo verificar aparece como «sin dato» y con el motivo.

**Siglas, la primera vez que aparecen:**

| sigla | nombre completo | qué es |
|---|---|---|
| **SERNAGEOMIN** | Servicio Nacional de Geología y Minería | el organismo dueño de la amenaza volcánica en Chile |
| **RNVV** | Red Nacional de Vigilancia Volcánica | el programa de SERNAGEOMIN que vigila los volcanes y define los niveles de alerta |
| **OVDAS** | Observatorio Volcanológico de los Andes del Sur | la sala 24/7 en Temuco que procesa la señal instrumental |
| **UGPSV** | unidad de Geología y Peligros de Sistemas Volcánicos | la unidad que dibuja los mapas de peligro zonificados |
| **REAV** | Reporte Especial de Actividad Volcánica | el documento con que SERNAGEOMIN comunica un cambio de alerta |
| **PET** | informe de Peligros y Evaluación Territorial | el estudio del que sale el mapa de peligro de un volcán |
| **MOP** | Ministerio de Obras Públicas | dueño del inventario vial y de puentes que usamos para cruzar |
| **ArcGIS** | — | la plataforma de mapas de la empresa Esri; SERNAGEOMIN publica ahí sus capas en la nube |
| **GeoJSON** | — | formato de texto para intercambiar mapas; lo entienden todos los programas de mapas |
| **STRtree** | *Sort-Tile-Recursive tree* | índice espacial: descarta de golpe todo lo que no puede tocar, y convierte un cruce de horas en segundos |

---

## 1 · Resumen en cinco líneas

- ✅ **Catálogo completo conseguido**: los **87 sistemas volcánicos activos** del
  Ranking de Riesgo Específico 2023, con coordenadas, factores de peligro y de
  exposición, y si OVDAS los monitorea. Más el ranking anterior (2019, 91
  volcanes) y el listado de informes PET.
- ⚠️ **Mapas de peligro zonificados: sólo hay UNO accesible**, el del complejo
  volcánico Nevados de Chillán (5 polígonos, 440 km²). **No existe capa nacional
  de polígonos de peligro volcánico accesible por programa** — el porqué está en
  la sección 4, con los cuatro caminos que se probaron y fallaron.
- ❌ **El estado de alerta por volcán NO se puede bajar hoy.** Y no es una caída
  pasajera: la propia fuente declara que dejó de publicarlo, y el sitio de la
  RNVV está detrás de un muro de autenticación. Detalle en la sección 5.
- ✅ **Cruce hecho** contra 14.039 tramos viales, 6.742 puentes y 39
  subestaciones eléctricas. Resultados en la sección 7.
- ★ **Recomendación sobre captura programada: SÍ, pero no como la minuta.** No
  hay tablero que fotografiar; hay que vigilar la *aparición* de reportes.
  Sección 6.

---

## 2 · Qué se probó, uno por uno

| dónde se buscó | qué pasó |
|---|---|
| `services1.arcgis.com/OyjvVdFTl5hfSdX3` (nube de SERNAGEOMIN) | ✅ **responde y es anónimo.** 206 servicios publicados; 11 tienen que ver con volcanes. Es de donde salió todo lo que se consiguió. |
| `rnvv.sernageomin.cl` (sitio propio de la RNVV) | ❌ **puerta cerrada.** El nombre apunta a `rnvv.onr.cl` detrás de Cloudflare Access; pide correo y código de acceso. No es una caída: es autenticación puesta a propósito. |
| `portalgeo.sernageomin.cl` (geoportal) | ❌ **no conecta.** Está en 190.98.205.167, el rango caído desde el 17-ago. Se probó una vez y no se insistió, según lo instruido. |
| `sdngsig.sernageomin.cl` (servidor de mapas propio) | ❌ **no conecta.** 190.98.205.186, mismo rango caído. Es donde viven los mapas de peligro por volcán. |
| `portalgeomin.sernageomin.cl` | ❌ **ni siquiera resuelve** el nombre. El servidor ya no existe con ese nombre. |
| `www.sernageomin.cl` (sitio institucional) | ✅ **responde.** De ahí salieron la página de alertas, el ranking en PDF y el REAV vigente. |
| Catálogo de ArcGIS Online (`arcgis.com/sharing/rest/search`) | ✅ **responde.** Sirvió para encontrar capas que el listado de servicios no muestra. |
| `catalogo.geoportal.cl` (catálogo nacional de datos geoespaciales) | ❌ **tiempo agotado** a los 40 segundos. |
| `geoide.minvu.cl` — servicio «Amenaza volcánica» del Ministerio de Vivienda y Urbanismo | ❌ **exige credencial** (`Token Required`). Existe y republica amenaza volcánica, pero no es público. |

### Condiciones de uso, revisadas ANTES de automatizar

- **`www.sernageomin.cl/robots.txt`**: `User-agent: *` / `Disallow: /wp-admin/`.
  Todo lo demás está permitido. Lo que se descarga (página de alertas, PDF del
  ranking, PDF del REAV) está fuera de la zona prohibida.
- **Capa «Ranking Volcanes 2023 v2»**: marcada `access: public`, sin credencial.
  Su campo de licencia dice literalmente *«2025 | Servicio Nacional de Geología y
  Minería | Red Nacional de Vigilancia Volcánica»*, con enlace a
  `rnvv.sernageomin.cl`. **Es una exigencia de atribución, y se cumple**: toda
  salida de este adaptador cita a SERNAGEOMIN / RNVV como fuente.
- **Capa «Peligro Geológico Shape»**: marcada `access: public`, **sin ninguna
  información de licencia declarada**. Se descarga porque está publicada
  abiertamente, pero queda anotado que la fuente no dijo bajo qué condiciones.
- **Ritmo**: entre una y dos peticiones por segundo, con pausas. Ningún servicio
  recibió carga apreciable. El modo `--alerta` pesa unos pocos KB por foto.
- ⚠️ **Pendiente que le corresponde a Alexis, no al proyecto**: pedirle a
  SERNAGEOMIN autorización por escrito. Es el mismo razonamiento que quedó
  anotado para el CSN (Centro Sismológico Nacional): mientras esto sea académico
  el encuadre alcanza, pero el destino declarado del instrumento es servir de
  insumo al COGRID (Comité para la Gestión del Riesgo de Desastres) y a SENAPRED
  (Servicio Nacional de Prevención y Respuesta ante Desastres), y ese día el
  encuadre académico puede dejar de cubrirlo.

---

## 3 · Lo que SÍ se bajó

Crudo archivado con fecha en `datos/crudo/volcanes/2026-08-19/` — la regla de
guardar el dato tal como llegó, antes de tocarlo, se cumple.

| archivo de trabajo | filas | qué contiene |
|---|---|---|
| `datos/volcanes_ranking_2023.csv` | **87** | Ranking de Riesgo Específico 2023. Coordenadas, región, tipo, categoría de riesgo (1 a 5), si OVDAS monitorea, factor de peligro, factor de exposición, riesgo específico |
| `datos/volcanes_ranking_2019.csv` | 91 | ranking anterior; conserva `NOMBRE_VOLCAN` y `SUBTIPO_DESC`, que el de 2023 no publica |
| `datos/volcanes_informes_pet.csv` | 11 | qué volcanes tienen informe PET ejecutado, en ejecución o proyectado |
| `datos/volcanes_peligro_zonificado.csv` | 5 | el único mapa de peligro volcánico zonificado accesible |
| `datos/crudo/volcanes/2026-08-19/Ranking_2023_tabloide.pdf` | — | el listado oficial en PDF, como respaldo documental |
| `datos/crudo/volcanes/alerta_diaria/…` | — | fotos fechadas de la página de alertas + el REAV vigente |

### 3.1 · El catálogo de 87 volcanes

Metodología, en palabras de la propia fuente: desde 2018 la RNVV actualiza el
ranking cada 3 años. Para 2023 evaluó **13 factores de peligro** y **12 factores
de exposición**; la suma de cada grupo da un puntaje, y el **producto** de ambos
da el «riesgo específico» que ordena el ranking.

Vale la pena notar la forma de la fórmula: **es un producto, no un promedio.**
Si la exposición es cero, el riesgo es cero por muy peligroso que sea el volcán.
Es exactamente el criterio que el proyecto ya adoptó para los vectores naturales
(«si falta una condición, no hay amenaza») y que le falta a la MICR, donde el
peligro y la exposición no se multiplican.

Reparto por categoría de riesgo (1 = muy alto, 5 = muy bajo):

| categoría | volcanes |
|---|---|
| 1 | 14 |
| 2 | 17 |
| 3 | 23 |
| 4 | 17 |
| 5 | 16 |

**43 de los 87 tienen monitoreo instrumental de OVDAS; 44 no.** Es decir: **la
mitad de los volcanes activos de Chile no están instrumentados.** Esto no es un
reproche a SERNAGEOMIN —monitorear un volcán cuesta— sino un dato que el
consolidador tiene que llevar consigo, porque cambia la confianza de cualquier
cosa que se diga sobre esos 44.

Reparto por región (los primeros): Antofagasta 17 · Los Lagos 16 · Magallanes y
Antártica Chilena 10 · Aysén 7 · La Araucanía 6 · Maule 5.

Los diez primeros del ranking 2023: Villarrica, Calbuco, Llaima,
Puyehue-Cordón Caulle, Grupo Descabezados, Carrán-Los Venados, Chaitén, Osorno,
Mocho-Choshuenco, Nevados de Chillán.

### 3.2 · Los informes PET: el mapa de la cobertura que falta

De los 87 volcanes, **la capa de informes PET nombra sólo 11**: 6 ejecutados
(Antuco, Chaitén, Copahue, Llaima, Nevados de Chillán, Villarrica), 3 en
ejecución (Lonquimay, Mocho-Choshuenco, Osorno) y 2 proyectados (Callaqui,
Taapaca).

Esto explica por sí solo por qué no hay mapa nacional de peligro volcánico: **no
existe todavía**. Se está construyendo volcán por volcán, y va por el 7 %.

---

## 4 · Los mapas de peligro: qué hay, qué no, y por qué

### 4.1 · Lo único que se consiguió: Nevados de Chillán

Capa `Peligro_Geológico_Shape`, servicio ArcGIS público de SERNAGEOMIN,
5 polígonos, escala declarada **1:75.000**, en total **440 km²**:

| grado | tipo de peligro (código) | superficie |
|---|---|---|
| GRADO-01 | PV-DEP-03 | 115,0 km² |
| GRADO-03 | PV-DEP-03 | 88,4 km² |
| GRADO-04 | PV-DEP-03 | 91,8 km² |
| GRADO-06 | PV-DEP-07 | 22,2 km² |
| GRADO-24 | PV-DEP-09 | 122,5 km² |

**Dos cosas hay que declarar y no esconder:**

1. **Los códigos no vienen con su diccionario.** El servicio entrega
   `GRADO-04` y `PV-DEP-03` pero **no** la tabla que dice qué significa cada uno.
   Sin esa tabla no se puede afirmar que GRADO-04 sea «más peligroso» que
   GRADO-03, ni que PV-DEP-03 sean lahares. **Se deja sin traducir a propósito**:
   inventar la equivalencia sería exactamente el tipo de error que este proyecto
   no puede permitirse. Pedirle la tabla de dominios a SERNAGEOMIN es un correo.

2. **El atributo `NOMBRE` dice «Zona volcánica central» y eso no calza.** Los
   polígonos están entre 36,5° y 37,0° de latitud sur y entre 71,1° y 71,8° de
   longitud oeste — es decir, **rodeando el complejo volcánico Nevados de
   Chillán** (36,871° S; 71,368° O), que está en la zona volcánica *sur*, no en
   la central. La ubicación se verificó con la geometría, que es dato duro; la
   etiqueta es texto y no calza. **Se usa la geometría y se anota la
   contradicción.**

### 4.2 · Lo que existe pero NO se pudo bajar

**(a) Los mapas de peligro por volcán, vivos pero inalcanzables.**
SERNAGEOMIN publica un visor por volcán con una capa llamada «Peligros
volcánicos» — se encontraron **46 visores**, numerados del 01 al 46, uno por
volcán (Nevados de Chillán, Villarrica, Llaima, Calbuco, Puyehue-Cordón Caulle,
Osorno, Láscar, Copahue, Chaitén, Hudson, Lonquimay, Laguna del Maule,
Planchón-Peteroa, Taapaca, Isluga, Ollagüe, Cay, y así hasta 46). Las capas viven en
`sdngsig.sernageomin.cl`, que está en el rango de red caído. También hay un
camino indirecto por el intermediario de ArcGIS
(`utility.arcgis.com/usrsvcs/…`), que se probó y devuelve
`Error generating token` — falla por la misma razón: el intermediario no puede
hablar con un servidor que no responde.

> **Esto vale más que un dato**: los mapas de peligro por volcán **existen, están
> publicados y son polígonos**. En cuanto la red de SERNAGEOMIN vuelva, hay que
> bajarlos. Los 46 visores quedaron identificados por nombre y se pueden
> recuperar con la misma búsqueda del catálogo de ArcGIS Online que usó este
> adaptador.

**(b) Los mapas de susceptibilidad volcánica: existen, pero son imágenes.**
Se encontraron seis productos nacionales de susceptibilidad volcánica publicados
por SERNAGEOMIN: **caída de cenizas**, **flujo piroclástico**, **lahar**, **flujo
de lava**, **caída de balísticos** y **probabilidad estimada de amenaza
volcánica** (esta última en siete horizontes: 1, 10, 50, 100, 475, 1.000 y 2.500
años).

Y sin embargo no sirven para cruzar, por dos motivos independientes:

1. **Son teselas de imagen, no polígonos.** Están publicados como
   `ArcGISTiledMapServiceLayer`: cuadraditos de dibujo, no geometría. Se pueden
   mirar, no intersectar. Sacar polígonos de ahí sería redibujar a ojo la
   frontera de un peligro, que es inventar.
2. **Cubren sólo la cuenca del Maipo**, no el país. Los nombres de los servicios
   lo dicen: `Susceptibilidad_lahar_28_8`,
   `Susceptibilidad_CaidaCeniza_Maipo_29092021`, y todos comparten la capa de
   apoyo «Comunas cuenca del Maipo».

**(c) El servicio del Ministerio de Vivienda y Urbanismo.** `geoide.minvu.cl`
tiene un servicio llamado «Amenaza volcánica» que republica esta información,
pero responde `Token Required`. No es público.

### 4.3 · La conclusión honesta

**Chile no tiene hoy —o no publica de forma accesible— una capa nacional de
polígonos de peligro volcánico.** Tiene 6 mapas de peligro terminados de 87
volcanes, unos cuantos visores por volcán, y un juego de susceptibilidad
nacional que está en formato de imagen y acotado a una cuenca.

Esto es exactamente el mismo hallazgo que el proyecto ya hizo con la remoción en
masa, pero al revés: allá había un mapa vivo y nadie lo cruzaba con
infraestructura; acá **el mapa todavía no está hecho**.

---

## 5 · ★ El estado de alerta: por qué no se puede bajar hoy

La RNVV define cuatro niveles de alerta técnica volcánica —**verde, amarilla,
naranja, roja**— y los actualiza por evento, no por calendario. Es el equivalente
volcánico de la Minuta Técnica de remoción en masa: un dato dinámico, que se
sobrescribe y no guarda historia.

**Pero hoy está peor que la minuta, por tres razones distintas que conviene no
confundir:**

**(a) La propia fuente declara caído su tablero.** La página oficial
(`www.sernageomin.cl/alertas-volcanicas/`, actualizada el 8-jul-2026) dice
textualmente:

> «El monitoreo volcánico de Sernageomin, realizado por nuestro Observatorio
> Volcanológico de los Andes del Sur (OVDAS), se mantiene activo de forma
> permanente (24/7). La vigilancia no se ha visto interrumpida; **no obstante, la
> visualización en tiempo real de esta información no se encuentra disponible en
> el sitio web**.»

Es decir: siguen vigilando, pero **dejaron de publicar**. Lo único que queda es
prosa sobre el volcán que está en alerta en ese momento.

**(b) El sitio propio de la RNVV está autenticado.** `rnvv.sernageomin.cl` pide
correo y código de acceso. No es una caída — es una puerta cerrada a propósito.

**(c) ★ Y el hallazgo más incómodo: el nivel de alerta viene como DIBUJO.**
El REAV del complejo volcánico Nevados de Chillán del 15-jun-2026 dice, en su
texto:

> «Debido a lo anterior, la alerta técnica volcánica se cambia a:»

…y a continuación **pone una imagen**. El nivel —amarilla, naranja o roja— está
dibujado, no escrito. Un humano lo lee de un vistazo; un programa no lo ve.
Se verificó abriendo el PDF: es una imagen JPEG de 1.560 × 1.020 píxeles.

Se decidió **no adivinarlo**. Se podría instalar una biblioteca de imágenes y
deducir el nivel del color de unos píxeles, pero declarar un nivel de alerta
oficial a partir del color de un dibujo es exactamente lo que un instrumento que
alimenta decisión pública no debe hacer. **El adaptador devuelve «SIN DATO» y
dice por qué.**

**Lo que sí quedó capturado y es real:** que hubo un cambio de alerta, de qué
volcán (Complejo Volcánico Nevados de Chillán), de qué región (Ñuble) y cuándo
(15 de junio de 2026, 17:00 hora local). Y el REAV completo archivado.

Del contenido del REAV, además, sale un dato físico útil: SERNAGEOMIN espera
«explosiones de baja magnitud a nivel de cráter Nicanor que afecten el entorno
inmediato **en un radio de 1 km**», con eyección de piroclastos balísticos,
emisión de ceniza y gases; y no descarta dispersión de ceniza a distancias
mayores.

**Un «sin dato» no es un «verde».** De los 87 volcanes del catálogo, hoy se sabe
el estado de **uno**, y de ese uno se sabe que cambió, no a qué. Los otros 86
quedan en «sin dato», que es un hueco declarado, no tranquilidad.

---

## 6 · ★ Recomendación sobre captura programada

**Sí corresponde, pero no la misma receta que la minuta de remoción en masa.**

La minuta es un **pizarrón**: hay un tablero con 119 zonas y un nivel por zona, y
lo que se pierde si no se fotografía es el estado de ese tablero. Por eso ahí la
receta es fotografiar cuatro veces al día.

Acá **no hay tablero que fotografiar**. Lo que hay es una página que aparece y
desaparece volcanes según haya o no actividad, y unos PDF que se publican y
después se borran del listado. Lo que se pierde para siempre no es un estado: es
**el hecho de que existió un reporte**.

Por eso la recomendación concreta es:

1. **Correr `volcanes.py --alerta` dos veces al día** (unos pocos KB por foto).
   No para reconstruir un tablero —no se puede— sino para dejar constancia
   fechada de qué volcanes estaban citados y qué REAV había publicados. El
   adaptador ya avisa si algo cambió respecto de la foto anterior.
2. **Archivar cada REAV que aparezca.** Ya lo hace. Es la única forma de armar
   una historia de eventos volcánicos que hoy nadie está guardando: la fuente
   deja el reporte vigente y saca el anterior. Se comprobó consultando el
   archivo de medios del sitio: **hay un solo REAV**, el del 15-jun-2026.
3. **Cuando la red de SERNAGEOMIN vuelva, correr `--peligro` y volver a probar
   los 46 visores por volcán.** Los mapas de peligro son geometría estable —no
   cambian de un día para otro— así que ahí no urge la frecuencia, urge no
   olvidarse.
4. **No** correr `--catalogo` a diario: el ranking se actualiza cada 3 años. Con
   una vez al mes basta y sobra.

Y una recomendación que no es técnica: **el canal más barato para arreglar todo
esto es un correo a SERNAGEOMIN** pidiendo (i) la tabla de dominios de los
códigos `GRADO-xx` y `PV-DEP-xx`, (ii) el estado de alerta en formato de datos, y
(iii) el archivo histórico de REAV. Las tres cosas existen adentro; el problema
es de publicación, no de producción.

---

## 7 · El cruce con la infraestructura

Método: el mismo de `cruzar_amenaza_exacto.py`, reutilizado tal cual —
**Shapely 2.1.2** para la geometría (trae GEOS adentro, así que no hace falta
GDAL), **pyproj 3.7.2** para largos y áreas geodésicas exactas sobre el
elipsoide, y **STRtree** como índice espacial.

Inventario cruzado: **14.039 tramos viales (90.971 km) · 6.742 puentes ·
39 subestaciones eléctricas**.

### 7.1 · Contra el mapa de peligro OFICIAL (Nevados de Chillán)

Éste es el único cruce que se apoya en una zonificación de peligro dibujada por
SERNAGEOMIN. Es el resultado con más peso de todo el trabajo.

| | dentro de la zona de peligro |
|---|---|
| tramos viales que la tocan | **20** |
| kilómetros de camino dentro | **75,2 km** |
| puentes dentro | **7** |
| subestaciones eléctricas dentro | **0** |

Por grado de peligro: GRADO-03 → 48,8 km · GRADO-04 → 23,6 km · GRADO-01 → 2,8 km.

Los caminos más comprometidos:

| km dentro / km del tramo | rol | camino |
|---|---|---|
| 23,45 / 74,06 | **N-55** | Cruce N-545 (Chillán) — hacia Termas de Chillán |
| 8,38 / 10,18 | N-313 | Cruce N-31 (Punilla) - Los Mayos - El Roble |
| 8,34 / 8,34 | **N-633** | Cruce N-55 (Los Lleuques) - Tranque Diguillín — **el tramo entero está dentro** |
| 7,48 / 14,13 | N-31 | Cruce Ruta 5 (San Carlos) - San Fabián |

Los 7 puentes: Renegado, Marchant, Aserradero, Torrealba, S/N 76P10067,
San Fabián (sobre el **río Ñuble**) y Las Truchas.

**Lectura, en simple:** la ruta **N-55** es el camino a las Termas de Chillán —
el mismo corredor turístico que la propia página de alertas de SERNAGEOMIN
menciona cuando describe Las Trancas, con sus centros de esquí y termas. Que 23
de sus 74 km caigan dentro de la zona de peligro no es un dato abstracto:
significa que la vía por la que entra y sale la gente de ese valle atraviesa la
zona zonificada. Y los puentes de la lista están sobre esteros y sobre el río
Ñuble, que son justo por donde bajaría un lahar.

### 7.2 · Anillos de cercanía a los 87 volcanes

**★★ ADVERTENCIA QUE HAY QUE LEER ANTES QUE LOS NÚMEROS.**
Estos anillos **no son un mapa de peligro**. Son círculos alrededor del cráter.
El peligro volcánico real **no es circular**: los lahares y los flujos
piroclásticos bajan encauzados por los valles y llegan muchísimo más lejos por
un lado que por el otro; la ceniza va donde la lleve el viento, que en Chile
sopla casi siempre hacia el este. Un anillo sirve para **una sola cosa**: filtrar
rápido qué infraestructura ni siquiera hace falta mirar. Todo lo que caiga
adentro hay que evaluarlo después con el mapa de peligro del volcán.

| radio | tramos | km viales | % de la red | puentes | subestaciones |
|---|---|---|---|---|---|
| 5 km | 72 | 205,2 | 0,23 % | 19 | 1 |
| 10 km | 238 | 1.119,1 | 1,23 % | 134 | 1 |
| 20 km | 581 | 4.962,2 | 5,46 % | 755 | 2 |
| 30 km | 936 | **8.539,3** | **9,39 %** | **1.195** | 2 |

Los volcanes que concentran más camino a menos de 30 km:

| km | puesto en el ranking | volcán |
|---|---|---|
| 566,1 | **#1** | Villarrica |
| 483,7 | #3 | Llaima |
| 451,0 | #11 | Lonquimay |
| **397,1** | **#42** | **Taapaca** |
| 338,8 | #2 | Calbuco |
| 262,8 | #6 | Carrán - Los Venados |
| 252,9 | #41 | Isluga |

Las dos subestaciones eléctricas a menos de 30 km de un volcán activo:
**Chungará** (rural, Parque Nacional Lauca) junto al **Guallatiri** (#25), y
**Collahuasi** (rural) junto al **Olca-Paruma** (#61).

### 7.3 · Tres hallazgos que salen de este cruce

**(1) Los corredores internacionales son lo más expuesto.**
De los 8.539 km a menos de 30 km de un volcán, **1.468 km son «Camino Nacional
con Carácter de Internacional»** — los pasos fronterizos. Los casos concretos:

- **Ruta 11-CH** (Arica – Putre – Tambo Quemado, hacia Bolivia): **148,5 km** a
  menos de 30 km del volcán **Taapaca**.
- **Ruta 199** (Camino Internacional Mons. Francisco Valdés, hacia Argentina por
  Mamuil Malal): 115,3 km cerca del **Villarrica**.
- **Ruta 181** (Victoria – Curacautín – Túnel Las Raíces): 180,7 km cerca del
  **Lonquimay**.
- **Ruta 15-CH** (Huara – Colchane, hacia Bolivia): 25,7 km cerca del **Isluga**.

Esto importa porque un paso fronterizo **no tiene redundancia**: si se corta, no
hay otro al lado. Es justo la propiedad `redundancia` que el estudio de vectores
de amenaza puso como parte de la vulnerabilidad, y acá aparece medida.

**(2) La mitad de la exposición está junto a volcanes que nadie instrumenta.**
**3.278 de los 8.539 km** (el 38 %) están cerca de volcanes que el ranking 2023
marca como **sin monitoreo de OVDAS** (campo `Monitoreo_OVDAS = 0`). El caso más
llamativo es **Taapaca**: cuarto en kilómetros de camino expuesto, con el
corredor a Bolivia encima. Su informe PET figura como «proyectado», o sea que el
mapa de peligro todavía no está hecho.

> ⚠️ **Contradicción dentro de la propia fuente, que se declara y no se
> resuelve:** el ranking 2023 marca Taapaca con `Monitoreo_OVDAS = 0`, pero
> existe un visor institucional llamado `33_Taapaca` dentro de la serie de
> visores de monitoreo. Los dos productos son de SERNAGEOMIN y dicen cosas
> distintas. No se elige uno: **se anota que hay que preguntar**. Mientras tanto
> el número de «38 % junto a volcanes sin monitoreo» hay que leerlo como «según
> el campo del ranking 2023», no como hecho verificado.

Y esto revela una asimetría que el ranking, por diseño, no puede ver: **el
ranking mide riesgo a la población, no a la infraestructura.** Taapaca está en el
puesto 42 de 87 porque cerca vive poca gente. Pero por su falda pasa el camino
por donde entra y sale la carga de una frontera. **Son dos preguntas distintas y
el ranking sólo responde una.** Éste es exactamente el hueco que el proyecto
existe para llenar.

**(3) La escala vuelve a decidir el resultado.** Entre 5 km y 30 km, los
kilómetros expuestos se multiplican por 42 (205 → 8.539). No hay un radio
«correcto»: hay que preguntarle al mapa de peligro de cada volcán, y ése es
justamente el que falta. Por eso los cuatro radios se publican juntos y ninguno
se presenta como «la respuesta».

---

## 8 · Limitaciones declaradas (leer antes de citar cualquier número)

1. **El cruce del 7.1 vale para un volcán de 87.** Es el único con zonificación
   accesible. No se puede extrapolar: cada volcán tiene su forma, su valle y su
   viento.
2. **Los anillos del 7.2 no son peligro.** Ver la advertencia. No usar esos
   números como «infraestructura en riesgo» sin el mapa de peligro correspondiente.
3. **Los códigos `GRADO-xx` y `PV-DEP-xx` están sin traducir**, porque la fuente
   no publicó el diccionario. No se puede ordenar los grados por severidad.
4. **La ceniza no está evaluada en absoluto.** Es el peligro que más lejos llega
   y el que afecta aeropuertos y redes eléctricas — que es parte de por qué esta
   amenaza importa aquí. Requiere el mapa de susceptibilidad de caída de cenizas,
   que existe pero está en formato de imagen y sólo para la cuenca del Maipo.
5. **Los aeropuertos no están en el inventario del proyecto.** El cruce se hizo
   contra caminos, puentes y subestaciones, que es lo que hay. La red aeroportuaria
   (Dirección General de Aeronáutica Civil) es la pieza que le falta a este
   cruce, y es la más sensible a la ceniza.
6. **El estado de alerta no está.** Cualquier producto que lo necesite tiene que
   declararlo como hueco, no rellenarlo.
7. **El inventario vial es del MOP y sólo cubre la red del MOP.** Los caminos
   privados, forestales y mineros no están, y en zona volcánica hay muchos.
8. **Isla de Pascua y la Antártica están en el catálogo** (volcanes Isla de
   Pascua y Orca). Sus coordenadas son correctas, pero el inventario vial del MOP
   ahí es distinto o inexistente; los números de esos dos no son comparables con
   los del continente.

---

## 9 · Cómo se corre

```bash
PY=/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/.venv-esa/bin/python
cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/infraestructura

$PY adaptadores/volcanes.py --catalogo   # catálogo de 87 volcanes (mensual basta)
$PY adaptadores/volcanes.py --peligro    # mapa de peligro zonificado
$PY adaptadores/volcanes.py --alerta     # ★ foto fechada, 2 veces al día
$PY adaptadores/volcanes.py --cruce      # cruce con infraestructura (~2 min)
```

Sin argumentos hace la bajada completa (`--catalogo --peligro --alerta`).

**Nota técnica:** `www.sernageomin.cl` entrega su certificado de seguridad
incompleto —manda el suyo pero se olvida del intermedio que lo respalda—. El
navegador y `curl` disimulan el error saliendo a buscar la pieza que falta;
Python no. Por eso el adaptador, si Python falla por eso, **reintenta con `curl`,
que sí verifica el certificado**. No se apaga la verificación en ningún momento:
apagarla sería aceptar a cualquiera que se haga pasar por SERNAGEOMIN.

---

## 10 · Qué queda pendiente

| pendiente | quién | por qué importa |
|---|---|---|
| Volver a probar `sdngsig.sernageomin.cl` cuando la red vuelva | proyecto | ahí están los mapas de peligro de 33 volcanes, en polígonos |
| Pedir a SERNAGEOMIN la tabla de dominios `GRADO-xx` / `PV-DEP-xx` | Alexis | sin ella, el mapa de peligro que sí tenemos no se puede ordenar por severidad |
| Pedir el estado de alerta en formato de datos y el archivo histórico de REAV | Alexis | es lo único que convierte esto en una serie con historia |
| Autorización por escrito de SERNAGEOMIN | Alexis | mismo razonamiento que quedó anotado para el CSN |
| Incorporar los aeropuertos al inventario | proyecto | es la infraestructura más sensible a la ceniza y hoy no está |
| Susceptibilidad de caída de cenizas en formato vectorial | pedir | el producto existe; está publicado como imagen |
| Resolver el catálogo nacional `catalogo.geoportal.cl` (tiempo agotado) | proyecto | puede tener capas que la nube de ArcGIS no muestra |
