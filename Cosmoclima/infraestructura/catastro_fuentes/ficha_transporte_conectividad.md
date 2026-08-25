# Catastro de fuentes — TRANSPORTE Y CONECTIVIDAD

Dominio: transporte y conectividad. Verificado el **15-16 de agosto de 2026** con
peticiones HTTP reales (una o dos por fuente, sin descargas masivas).

Criterio único de utilidad: **¿cambia la prioridad de un activo de infraestructura?**
★★★ sí, directamente · ★★ sí, indirectamente · ★ contexto

---

## Resumen de una línea, antes del detalle

**Sí existe el inventario vial georreferenciado del MOP, y es mucho mejor de lo
esperado.** Está publicado en un servidor de mapas abierto (`rest-sit.mop.gob.cl`)
y además se puede bajar como archivo. Trae **14.039 tramos de camino** con rol de
ruta, kilómetro inicial y final, clasificación y tipo de carpeta; **6.742 puentes**
con el cauce que cruzan y su año de construcción; y —el hallazgo que no esperaba—
**6.141 emergencias viales georreferenciadas** (4.291 vigentes/recientes + 1.850
históricas desde 2015), con rol de ruta y kilómetro. Es decir: el MOP publica dónde
está cada camino **y** dónde se le ha roto históricamente.

**Transporte Informa NO tiene API** (la tiene apagada a propósito), pero **sí tiene
historia**: unos 62.000 registros de estado de tránsito desde 2013, alcanzables por
su mapa del sitio.

---

## La analogía que ordena todo esto

Pensá la red vial como la **nervadura de una hoja**. La nervadura central (Ruta 5)
está en todos los mapas y todos la cuidan. Pero una porción de tejido de la hoja se
seca cuando se corta **la vena única** que la alimenta — y esa vena es delgada, no
figura en ninguna foto, y nadie la mira.

Eso es exactamente el problema del proyecto: «Carreteras Secundarias (Rutas
Rurales)» queda última en la matriz (PF = 0,41), pero es la vena única. El glosario
oficial define **aislamiento** como «acceso normal interrumpido y **no se cuenta
tampoco con un acceso alternativo**» — o sea, define el problema en términos de la
**red**, no del camino suelto.

Y acá está lo importante: **con la capa del MOP se puede calcular eso**. Si tenés
todos los tramos con su geometría, armás un grafo (un dibujo de qué se conecta con
qué) y buscás los tramos cuya caída parte el dibujo en dos. En teoría de grafos a
esos tramos se los llama, con toda justicia, **puentes**. Hoy el proyecto no puede
hacerlo porque no tiene ningún inventario vial ubicado. Con esta fuente, puede.

---

# FICHAS

---

## F-T01 · Red Vial Nacional georreferenciada — MOP / Dirección de Vialidad ★★★

**1. Organismo y producto**
Ministerio de Obras Públicas · Dirección de Vialidad · Unidad de Gestión de
Información Territorial (UGIT). Producto: **Red Vial Nacional** (todos los caminos
de tuición del MOP, con su trazado real).

**2. URL exacta y endpoint** — todo VERIFICADO

- Servicio de mapas consultable:
  `https://rest-sit.mop.gob.cl/arcgis/rest/services/VIALIDAD/Red_Vial_Chile/MapServer`
- Capa con todos los atributos (la de mayor detalle): `.../MapServer/3`
- Consulta de ejemplo probada:
  `.../MapServer/3/query?where=1=1&returnCountOnly=true&f=json` → `{"count":14039}`
- Descarga directa del archivo (probada, devuelve ZIP de 10,1 MB):
  - Shapefile: `https://mapasvialidad.mop.gob.cl/archivos/sites/10/2026/08/Red_Vial_shp2026_08_05.zip`
  - Geodatabase: `https://mapasvialidad.mop.gob.cl/archivos/sites/10/2026/08/Red_Vial_gdb2026_08_05.zip`
- Página madre de descargas: `https://mapasvialidad.mop.gob.cl/descargas/`

> ⚠️ La URL del ZIP **lleva la fecha adentro** (`2026_08_05`). Cuando el MOP
> actualice, esa URL muere. No se puede dejarla escrita a fuego en el código: hay que
> leer la página de descargas y sacar el enlace vigente. El servicio REST, en cambio,
> tiene URL estable.

**3. Qué publica**
14.039 tramos de camino, cada uno como una línea en el mapa, con estos campos
verificados: `ROL` (el número de la ruta), `ROL_ID`, `ROL_LABEL`, `CODIGO_CAMINO`,
`NOMBRE_CAMINO`, `CLASIFICACION`, `CARPETA` (tipo de superficie), `KM_I` y `KM_F`
(kilómetro inicial y final), `KM_TRAMO` (largo), `CALZADA`, `ORIENTACION`,
`ENROLADO`, `CONCESIONADO`, `REGION`, `TUICION`.

Las 23 clases de `CLASIFICACION` que devuelve el servicio incluyen: *Camino Nacional
Longitudinal*, *Camino Nacional*, *Camino Nacional con Carácter de Internacional*,
*Camino Regional Principal*, *Camino Regional Provincial*, *Camino Regional Comunal*,
*Camino Regional de Acceso*, *Camino por Enrolar*, *Camino Privado*, *Sin
Información*, más variantes «en proceso de enrolamiento» y «conexión vial urbana».

Esto importa mucho: **«Camino Regional Comunal» y «Camino Regional de Acceso» son,
literalmente, las rutas rurales que la matriz castiga con PF = 0,41.** La capa las
identifica una por una y las ubica.

**4. Familia** · **ACTIVO** (es el inventario de activos viales). Secundariamente
BASE TERRITORIAL, porque el trazado sirve de esqueleto para ubicar todo lo demás.

**5. Formato**
ArcGIS REST (JSON), Shapefile, Geodatabase, KMZ. El servicio además declara soporte
`KmlServer` y `WMSServer`. Sistema de referencia **SIRGAS-Chile** (código EPSG 5360),
coordenadas geográficas.

**6. Acceso** · **anónimo, VERIFICADO.** Sin registro, sin clave, sin token.
Un detalle práctico verificado: la descarga del ZIP **exige un `User-Agent` de
navegador**. Con un cliente «pelado» el servidor devuelve una página HTML de 2.519
bytes en vez del archivo. No es un bloqueo, es un filtro tosco.

**7. Granularidad territorial**
Máxima: geometría de línea, escala aproximada 1:10.000 a 1:25.000 según región. Trae
`REGION` como campo. **No trae comuna ni provincia** — hay que cruzarlo espacialmente
con la capa de límites comunales (pendiente H-13 del proyecto).

**8. Frecuencia de actualización**
El archivo publicado dice **05-ago-2026**, o sea diez días antes de esta revisión. El
propio MOP la describe como «base de datos dinámica en constante depuración». No hay
calendario declarado: **SUPUESTO**, parece de orden mensual o trimestral.

**9. Historia disponible**
Prácticamente ninguna. Se publica **una sola foto, la vigente**. El servicio declara
`hasVersionedData: true`, pero no expone versiones anteriores por la interfaz
pública. Si el proyecto quiere serie histórica, tiene que **guardar cada descarga con
su fecha** (que es justamente la regla de diseño 2 del documento de fuentes).

**10. Condiciones de uso**

- `rest-sit.mop.gob.cl` **no tiene robots.txt** (404). Sin prohibición declarada, pero
  tampoco permiso explícito.
- `mapasvialidad.mop.gob.cl/robots.txt` (verificado): sólo bloquea `/wp-admin/`. El
  resto está permitido.
- Restricción textual del propio sitio, copiada literal: *«La información referente a
  calles (vías urbanas) desplegada a través del visualizador es de carácter
  referencial. Estos datos **no pueden ser utilizados para otros efectos, más que la
  sola publicación a través de nuestros sistemas**.»* → **las vías urbanas quedan
  fuera de uso.** Los caminos interurbanos, que es lo que le compete a Vialidad y lo
  que el proyecto necesita, no tienen esa restricción.
- ⚠️ **Hallazgo de licencia:** el catálogo oficial `datos.gob.cl` declara la Red Vial
  Nacional bajo **Creative Commons Non-Commercial (Any)**, es decir **no comercial**.
  Para un instrumento de política pública sin fin de lucro esto no estorba, pero hay
  que dejarlo escrito. La ficha de datos.gob.cl es vieja (2015) y apunta a un servidor
  antiguo, así que conviene **pedir confirmación por escrito a la UGIT** antes de
  publicar cualquier derivado.
- **Veredicto**: automatización razonable, con cortesía (una descarga por ciclo, no
  consultas en ráfaga), y con la licencia no-comercial anotada.

**11. Para qué sirve en el modelo**
Es la pieza que le falta al proyecto entero. Tres usos:

1. **Ubicar los activos viales.** Hoy la matriz dice «carreteras secundarias, PF 0,41»
   en abstracto. Con esto, cada tramo tiene coordenadas y se puede meter dentro (o
   fuera) de la zona de peligro Alto que declara SERNAGEOMIN.
2. **Construir el grafo y detectar aislamiento.** Éste es el uso que vale oro. Con la
   geometría se arma la red y se buscan los tramos cuya caída desconecta un pueblo.
   Eso convierte «no tiene acceso alternativo» de frase de glosario en **número
   calculable**, que es la definición operativa que pide el glosario oficial.
3. **Reponderar el PF.** Un camino rural que es vena única no puede valer lo mismo que
   uno que corre en paralelo a otros dos. El `PF = 0,41` es un promedio de categoría;
   la red permite corregirlo activo por activo.

**12. Utilidad** · **★★★**. Sin discusión: cambia la prioridad de miles de activos, y
habilita el criterio de aislamiento que hoy no se puede calcular.

**13. Riesgos y límites**

- **La antigüedad del levantamiento.** El propio servicio lo dice: levantamiento GNSS
  original **2000-2003**, con los caminos básicos actualizados a **agosto de 2015**, y
  la descripción menciona vigencia «al año 2017». O sea: el trazado es bueno, pero la
  frescura del detalle es despareja. No suponer que todo está al día.
- **Advertencia jurídica del propio MOP**, literal: *«no representa la naturaleza
  jurídica de los caminos graficados»* y son *«documentos de trabajo»*. No sirve para
  discutir propiedad ni tuición; sí para riesgo y conectividad.
- **Sólo caminos de tuición MOP.** Los caminos municipales, privados y forestales no
  están (o están como «Camino Privado» / «por Enrolar», incompletos). En zonas rurales
  esto puede subestimar la conectividad real — o sobreestimar el aislamiento.
- **Tope de 1.000 registros por consulta** (`maxRecordCount: 1000`). Para bajar los
  14.039 tramos por API habría que paginar 15 veces. **Mejor bajar el ZIP una vez** —
  es más cortés con el servidor y llega todo junto.
- La URL de descarga es frágil (lleva fecha). Ver advertencia arriba.

---

## F-T02 · Catastro de Puentes — MOP / Dirección de Vialidad ★★★

**1. Organismo y producto** · MOP · Dirección de Vialidad. **Catastro nacional de
puentes.** (Existe, y es bueno.)

**2. URL exacta y endpoint** — VERIFICADO

- `https://rest-sit.mop.gob.cl/arcgis/rest/services/VIALIDAD/Puentes/MapServer/0`
- Conteo probado: `.../0/query?where=1=1&returnCountOnly=true&f=json` → `{"count":6742}`
- Descargas (mismas condiciones que F-T01, con `User-Agent` de navegador):
  - SHP: `https://mapasvialidad.mop.gob.cl/archivos/sites/10/2026/08/Puentes_shp_2026_08_05.zip`
  - GDB: `https://mapasvialidad.mop.gob.cl/archivos/sites/10/2026/08/Puentes_gdb2026_08_05.zip`
  - KMZ: `https://mapasvialidad.mop.gob.cl/archivos/sites/10/2026/08/Puentes_kmz2026_08_05.kmz`
- Hay una capa hermana para la Ruta 5: `.../VIALIDAD/Puentes_Ruta5/MapServer`
- Y una tercera vía, dentro del catastro vial: `.../VIALIDAD/Catastro_Vial/MapServer/0`
  (capa «PUENTES»)

**3. Qué publica**
6.742 puentes como puntos, con 43 campos verificados. Los que importan para riesgo:

| Campo | Qué es | Por qué importa acá |
|---|---|---|
| `CODIGO_PUENTE`, `NOMBRE_PUENTE` | identificación | clave para citar el activo |
| `ROL`, `CODIGO_CAMINO`, `M_INICIO` | en qué ruta y en qué metro está | **permite pegarlo al tramo vial de F-T01** |
| **`CAUCE_QUEB`** | el río o quebrada que cruza | **el dato de oro: liga el puente a la amenaza hidrometeorológica** |
| `PROVINCIA`, `REGION` | ubicación administrativa | traduce al idioma de SENAPRED |
| `AÑO_CONTRUCCION`, `AÑO` | antigüedad | proxy de vulnerabilidad estructural |
| `TIPO_ESTRUCTURA`, `MAT_VIGAS`, `MAT_CEPAS`, `MAT_ESTRIB`, `INFRA_TIPO` | materiales y tipo | vulnerabilidad |
| `LARGO`, `ANCHO_CALZADA`, `NUM_CEPAS`, `NUM_VIGAS` | dimensiones | exposición |
| `TUICION` | de quién es | filtra responsabilidad |

**4. Familia** · **ACTIVO**.

**5. Formato** · ArcGIS REST (JSON), SHP, GDB, KMZ. Geometría de punto.

**6. Acceso** · **anónimo, VERIFICADO.**

**7. Granularidad** · punto exacto, **con provincia y región ya en el dato**. Mejor
que la red vial en esto (que sólo trae región).

**8. Frecuencia** · misma fecha de corte que la red vial: **05-ago-2026**.

**9. Historia** · sólo la foto vigente. Guardar cada descarga fechada.

**10. Condiciones de uso** · idénticas a F-T01. Sin robots.txt en `rest-sit`;
`mapasvialidad` permite todo salvo `/wp-admin/`. Misma advertencia de licencia
no-comercial declarada en datos.gob.cl.

**11. Para qué sirve en el modelo**
Los puentes son **el punto único de falla por excelencia**, y el catastro lo hace
medible por primera vez. Dos cruces inmediatos y baratos:

- **Puente × cauce × amenaza.** El campo `CAUCE_QUEB` dice qué río cruza cada puente.
  Cuando la DMC anuncia lluvia intensa en una zona o la DGA marca caudal alto en un
  río, se puede listar **exactamente qué puentes están sobre ese cauce**. Eso es el
  cruce que el proyecto dice que nadie hace, hecho con dos campos.
- **Puente × grafo × aislamiento.** Un puente sobre una vena única no es un activo
  más: es el activo cuya caída aísla. Cruzar `ROL` + `M_INICIO` con el tramo vial de
  F-T01 permite marcar cuáles.

**12. Utilidad** · **★★★**.

**13. Riesgos y límites**

- ⚠️ **El campo `EST_PUENTE` (estado del puente) está VACÍO.** Lo probé pidiendo sus
  valores distintos y devuelve un único valor: `null`. Es decir: **el catastro dice
  dónde está cada puente y de qué está hecho, pero NO dice en qué condición está.**
  Ése es el dato que más se querría y no está publicado. Hay que suplirlo con la
  antigüedad (`AÑO_CONTRUCCION`) y el material, que son proxies pobres. Vale la pena
  preguntar formalmente a Vialidad si existe un índice de condición que no se publica.
- Hay varios campos que parecen de estudios internos (`Estudio_1034_puentes`,
  `LARGO_FEMN`, `MAT_INFRA_FEMN`, `HOMOLOGADO`, `UNICO`) sin documentación pública.
  No usarlos sin entenderlos.
- 6.742 puentes es plausible pero no cubre alcantarillas ni badenes, que también
  cortan caminos rurales.

---

## F-T03 · Red Vial Estructurante (RVE) — MOP / Dirección de Vialidad ★★★

**1. Organismo y producto** · MOP · Dirección de Vialidad. **Red Vial Estructurante.**

**2. URL exacta y endpoint** — VERIFICADO
`https://rest-sit.mop.gob.cl/arcgis/rest/services/VIALIDAD/Red_Vial_Estructurante/MapServer/0`
Conteo probado: **724 tramos**.

**3. Qué publica**
Descripción literal del servicio: *«La Red Vial Estructurante (RVE) es una capa que
identifica los caminos que proveen de mayor conectividad a los centros poblados y
enclaves productivos del país y que, por tanto, implica prioridad en s[u
mantenimiento]»*.

Traducido: **el propio MOP ya hizo un ejercicio de "qué caminos importan por
conectividad"** y lo publicó. 724 tramos de 14.039.

**4. Familia** · **ACTIVO**, con carga de priorización.

**5. Formato** · ArcGIS REST (JSON). **SUPUESTO**: no aparece en la página de
descargas, así que probablemente sólo por servicio.

**6. Acceso** · **anónimo, VERIFICADO.**

**7. Granularidad** · geometría de línea, nacional.

**8. Frecuencia** · **no declarada. SUPUESTO**: se actualiza junto con la red vial.

**9. Historia** · ninguna publicada.

**10. Condiciones de uso** · mismas que F-T01.

**11. Para qué sirve en el modelo**
Es un **contraste independiente para nuestra propia priorización**, y por eso vale
tanto. La lógica es la misma que el documento de SENAPRED aplica a SERNAGEOMIN: si
nuestro modelo marca como crítico un camino que el MOP no puso en la RVE, o al revés,
**alguno de los dos está mal y hay que mirarlo**. Un ejercicio concreto y barato:

> Calcular con el grafo qué tramos son «vena única» (su caída desconecta un pueblo) y
> ver **cuántos de ésos NO están en la RVE**. Esa diferencia es, muy probablemente, el
> aporte específico del proyecto: los caminos que aíslan y que el criterio actual de
> priorización no captura.

**12. Utilidad** · **★★★**. Cambia prioridades por dos vías: da una lista oficial de
caminos prioritarios, y sirve de espejo para validar la nuestra.

**13. Riesgos y límites**

- No pude verificar los campos de la capa (no los consulté para no cargar el servicio
  con más peticiones). **Queda pendiente** revisar `.../MapServer/0?f=json`.
- No hay metodología publicada de cómo se eligieron esos 724 tramos. Sin eso, el
  contraste es sugerente pero no concluyente. **Pedirla a la Dirección de Vialidad.**
- 724 tramos es una red gruesa. Por definición **no incluye** las rutas rurales que
  aíslan pueblos chicos — que es justamente el punto ciego que el proyecto persigue.

---

## F-T04 · Emergencias históricas sobre infraestructura MOP ★★★

**1. Organismo y producto** · Ministerio de Obras Públicas. **Emergencias Históricas**
sobre infraestructura MOP.

**2. URL exacta y endpoint** — VERIFICADO
`https://rest-sit.mop.gob.cl/arcgis/rest/services/EMERGENCIA/EMERGENCIA_HISTORICA_MOP/MapServer/0`
Conteo probado: **1.850 eventos**.

**3. Qué publica**
Descripción literal: *«información de emergencias ocurridas sobre la infraestructura
MOP, desde el año 2015 a la fecha. La información se representa como punto y comprende
los eventos que afectaron a la infraestructura de Vialidad, Portuaria, Aeroportuaria,
Hidráulica y de APR»*.

Campos verificados: `ID_EMER`, `EMERGENCIA` (descripción en texto libre),
`INFRA_AFEC` (qué infraestructura), **`ROL_VIAL`** y **`KM_INI`** (dónde, en lenguaje
de ruta), `ELEMENTO`, `CODREG`, `FECHA`, **`GRAVEDAD`**, `DIRECCION` (qué dirección
del MOP), `COD_OPERATI`.

Registro real devuelto por el servicio, para que se vea la forma:
`{'ID_EMER': 3948, 'EMERGENCIA': 'incendio sistema APR Linares de Perales',
'INFRA_AFEC': 'UNION MAULE CLARO (LINARES DE PERALES)', 'ELEMENTO': 'Red Matriz',
'CODREG': '07', 'FECHA': 1485448164000, 'GRAVEDAD': 'Grave', 'DIRECCION': 'APR'}`

`GRAVEDAD` toma valores del tipo *Grave* / *Moderado* — o sea, **una escala ordinal
ya existente**, igual que la de SERNAGEOMIN.

**4. Familia** · **ESTADO-CONSECUENCIA**. Es el registro del daño consumado.

**5. Formato** · ArcGIS REST (JSON), geometría de punto. `FECHA` viene en
milisegundos desde 1970 (formato de máquina, hay que convertirlo).

**6. Acceso** · **anónimo, VERIFICADO.**

**7. Granularidad** · punto georreferenciado + código de región (`CODREG`) + rol de
ruta y kilómetro cuando el evento es vial.

**8. Frecuencia** · «desde 2015 a la fecha», sin calendario declarado. **SUPUESTO**:
se alimenta a medida que se cierran los eventos.

**9. Historia disponible** · **once años, 2015 a hoy.** Es la fuente con más historia
de todo este dominio.

**10. Condiciones de uso** · sin robots.txt en el host (404). Sin términos específicos
localizados. **Automatizar con moderación**; 1.850 registros se bajan en dos consultas
paginadas y no hace falta repetirlo seguido.

**11. Para qué sirve en el modelo**
Éste es **el banco de pruebas del proyecto**, y probablemente el hallazgo más útil de
esta revisión después del inventario vial.

El documento de integración con SENAPRED dice que hace falta «contra qué validar». Acá
está, para el dominio vial: **once años de dónde se rompió realmente la
infraestructura**. Y como los eventos traen `ROL_VIAL` + `KM_INI`, se pueden **pegar
directamente sobre el inventario de F-T01**, que trae `ROL` + `KM_I` + `KM_F`. Las dos
capas hablan el mismo idioma.

Eso permite la pregunta que valida el modelo entero:

> **¿Los tramos que nuestro modelo marca como prioritarios son los mismos donde el país
> ha tenido emergencias durante once años?**

Si la respuesta es no, el modelo está mal y se descubre antes de entregárselo a nadie.
Y si aparecen tramos con muchas emergencias que el modelo no marca, ahí hay una
variable faltante. En términos de entomología: es tener once años de trampas ya
instaladas y revisadas, en vez de empezar a muestrear desde cero.

Además, la frecuencia histórica de emergencias por tramo es, en sí misma, **una
variable de priorización**: un camino que se corta todos los inviernos no vale lo
mismo que uno que nunca se cortó.

**12. Utilidad** · **★★★**. Cambia prioridades directamente (frecuencia histórica de
falla) y valida todo lo demás.

**13. Riesgos y límites**

- **1.850 eventos en 11 años son pocos** para un país que se corta entero cada
  invierno. Casi seguro es un registro **incompleto o filtrado** (quizá sólo
  emergencias con decreto o con contrato asociado). No leerlo como censo de cortes.
- **Sesgo de registro:** sólo aparece lo que afectó a infraestructura **MOP**. Los
  caminos municipales y privados —los que más aíslan— quedan invisibles. El sesgo va
  justo en contra de lo que el proyecto busca.
- `EMERGENCIA` es texto libre, escrito por personas distintas en años distintos. Para
  clasificar la causa (lluvia, remoción, incendio, sismo) hay que leer texto, con todo
  el error que eso trae. **No automatizar la clasificación sin revisión humana de una
  muestra.**
- El servicio corre sobre ArcGIS 10.2, versión antigua. Puede desaparecer o migrar.
  **Bajar una copia y guardarla fechada, ya.**

---

## F-T05 · Emergencias vigentes sobre infraestructura MOP ★★★

**1. Organismo y producto** · MOP. **Emergencias en Infraestructura MOP** (la capa
viva, hermana de F-T04).

**2. URL exacta y endpoint** — VERIFICADO
`https://rest-sit.mop.gob.cl/arcgis/rest/services/EMERGENCIA/EMERGENCIA_MOP/MapServer/0`
Conteo probado: **4.291 eventos**.
Trae además una tabla auxiliar de filtros: `.../MapServer/1`.

Y hay más servicios en la misma carpeta `EMERGENCIA`, verificados por listado:
`CONTRATOS_GLOBALES_DV`, `Emergencia_Mayo`, `EMERNORTE`, `MAPA_ESTACIONES_DGA`.

**3. Qué publica**
Emergencias sobre infraestructura MOP: *Agua Potable Rural, Obras Portuarias, Obras
Aeroportuarias, Riego, Cauce, Aguas Lluvias, Estaciones DGA, Edificación Pública* (y
Vialidad).

Campos verificados, **más ricos que la capa histórica**: `ID_EMER`, `EMERGENCIA`,
`COD_INFRA`, `INFRA_AFEC`, **`ROL_VIAL`, `KM_INI`, `KM_FIN`** (tramo completo, no sólo
el punto de inicio), `ELEMENTO`, `CODREG`, **`CODCOM` (código de comuna)**, `FECHA`,
`GRAVEDAD`, **`OPERATIVIDAD`**, `DIRECCION`, **`ESTADO_EMER`**, `COD_OPERATI`,
`SERV_MOP`, más campos de auditoría (`created_date`, `last_edited_date`).

**4. Familia** · **ESTADO-CONSECUENCIA**.

**5. Formato** · ArcGIS REST (JSON), punto.

**6. Acceso** · **anónimo, VERIFICADO.**

**7. Granularidad** · **la mejor de todo este dominio**: punto + código de comuna
(`CODCOM`) + rol de ruta con kilómetro inicial **y final**. Habla los dos idiomas del
«problema de las dos geografías» sin que haya que traducir.

**8. Frecuencia** · tiene campos de última edición (`last_edited_date`), lo que sugiere
**actualización continua**. **SUPUESTO** — no lo verifiqué en el tiempo.

**9. Historia** · 4.291 registros acumulados. Se solapa con F-T04; habrá que ver cómo
se reparten (probablemente ésta es la operativa y aquélla la depurada).

**10. Condiciones de uso** · sin robots.txt (404), sin términos localizados.
Automatizar con moderación.

**11. Para qué sirve en el modelo**
Dos campos la hacen especial:

- **`OPERATIVIDAD` y `ESTADO_EMER`** dicen si el camino quedó *operativo*, *con
  restricción* o *cortado*. Eso es exactamente la variable de estado que el módulo de
  colapso necesita, y es lo más parecido a un «FEN dinámico vial» que publica el país.
- **`KM_INI` + `KM_FIN`** definen el **tramo cortado**, no un punto. Con eso el corte
  se puede meter en el grafo y **calcular qué localidades quedan sin acceso
  alternativo** — el cálculo de aislamiento, hecho sobre un corte real y vigente.

**12. Utilidad** · **★★★**. Es la fuente de estado en tiempo casi real del dominio.

**13. Riesgos y límites**

- **No verifiqué la frescura.** No sé si «vigentes» significa de hoy o si arrastra
  eventos viejos sin cerrar. **Antes de usarla en serio: pedir el máximo de `FECHA` y
  la distribución de `ESTADO_EMER`.** Es una consulta barata y responde todo.
- Los nombres de las capas del servicio hermano `Emergencias_Vialidad` vienen
  **corruptos** (`'__'`, `'___'`, `'____'`), señal de problemas de codificación de
  caracteres en el servidor. Cuidado con las tildes al leer estos servicios.
- Mismo sesgo que F-T04: sólo infraestructura MOP.

---

## F-T06 · Infraestructura MOP compartida con ONEMI/SENAPRED ★★

**1. Organismo y producto** · MOP · IDE institucional. Servicio **INFRA_MOP_ONEMI**.

**2. URL exacta y endpoint** — VERIFICADO
`https://rest-sit.mop.gob.cl/arcgis/rest/services/IDE_MOP/INFRA_MOP_ONEMI/MapServer`
Capas verificadas: `2` Servicio Sanitario Rural · `3` Obras Portuarias · `4` Catastro
de Embalses · `5` **Puentes** · `7` **Red Vial Nacional**.

**3. Qué publica**
Descripción literal: *«Servicio WFS de ArcGIS 10.2 con infraestructura MOP y
emergencias de Vialidad, **para ser compartida con ONEMI**. La cobertura de la
Información es de carácter nacional.»*

**4. Familia** · **ACTIVO**.

**5. Formato** · ArcGIS REST / WFS.

**6. Acceso** · **anónimo, VERIFICADO.**

**7. Granularidad** · nacional, geometría.

**8. Frecuencia** · **no declarada. SUPUESTO**: espeja las capas madre.

**9. Historia** · ninguna.

**10. Condiciones de uso** · sin robots.txt. Al estar declarado como servicio de
intercambio institucional, es el canal de menor fricción política.

**11. Para qué sirve en el modelo**
El contenido es **redundante** con F-T01, F-T02 y F-T05 (mismas capas, menos campos).
Su valor no es el dato: **es institucional**. Es la prueba documentada de que **el MOP
ya tiene un canal formal para entregarle su infraestructura a SENAPRED**. El proyecto
no tiene que inventar la cañería — tiene que enchufarse a una que ya existe, y que
además ya resolvió el problema de qué capas se comparten.

Recomendación: **usar las capas madre para calcular** (tienen más campos) y **citar
ésta al conversar con SENAPRED**, porque es la que ellos ya conocen.

**12. Utilidad** · **★★**. No cambia prioridades por sí sola; sí allana el camino de
entrega, que es la mitad del proyecto.

**13. Riesgos y límites**
Menos campos que las capas madre. Al ser un espejo, puede quedar desactualizado
respecto del original sin aviso. No usarlo como fuente de cálculo.

---

## F-T07 · Transporte Informa — MTT ★★

**1. Organismo y producto** · Ministerio de Transportes y Telecomunicaciones.
**Transporte Informa**, el canal público de estado de la movilidad.

**2. URL exacta y endpoint** — VERIFICADO

- Sitio: `https://www.transporteinforma.cl/`
- Sección de estado: `https://www.transporteinforma.cl/estado-de-la-movilidad/`
- **Subsitios regionales** (verificados, existen de verdad):
  `https://coquimbo.transporteinforma.cl/` · `https://valparaiso.transporteinforma.cl/`
  · `https://biobio.transporteinforma.cl/` · `https://loslagos.transporteinforma.cl/`
- **Mapa del sitio (la puerta a la historia):**
  `https://www.transporteinforma.cl/sitemap_index.xml`
  → contiene **62 archivos** `estado_de_transito-sitemapN.xml`, de 1.000 URL cada uno.
- ❌ **API REST de WordPress: APAGADA.** Verificado:
  `GET https://www.transporteinforma.cl/wp-json/` →
  `{"code":"rest_not_logged_in","message":"Rest API disabled.","data":{"status":401}}`
  Lo mismo en los subsitios regionales (probado en Coquimbo). **No hay JSON.**
- ❌ **robots.txt: no existe** (404).

**3. Qué publica**
Trabajos e incidencias vigentes, vías reversibles, vías exclusivas, pistas solo buses,
tiempos de viaje. Entradas reales vistas: *«Camino Lonquén – 4 Poniente» (14-ago-2026)*,
*«Froilán Roa – Departamental» (14-ago-2026)*, *«Paso Los Libertadores se mantendrá
CERRADO toda esta semana, debido a nieve acumulada» (14-ago-2026)*.

**4. Familia** · **ESTADO-CONSECUENCIA**.

**5. Formato** · **HTML solamente.** Hay que leer la página. El sitemap es XML.

**6. Acceso** · **anónimo, VERIFICADO** para el HTML. **API: no pública, VERIFICADO**
(deshabilitada explícitamente por el administrador del sitio).

**7. Granularidad territorial**
Muy desigual. El sitio central es **casi todo Región Metropolitana**, a nivel de calle
y esquina. Las regiones tienen subsitios propios pero sólo cuatro: Coquimbo,
Valparaíso, Biobío y Los Lagos. **No hay cobertura nacional.**

**8. Frecuencia** · continua, según ocurren los eventos.

**9. Historia disponible** · **Sí, y bastante: aproximadamente 62.000 registros de
estado de tránsito, desde 2013.** El primer archivo del mapa del sitio trae 1.000 URL
con fechas de mayo y noviembre de 2013 hacia adelante. La API está apagada, pero el
mapa del sitio deja la lista completa a la vista. VERIFICADO por conteo.

**10. Condiciones de uso** · ⚠️ **el punto delicado de esta ficha.**

- No hay robots.txt (404): **no hay prohibición declarada, y tampoco permiso.**
- El administrador **apagó deliberadamente la API REST**. Eso no es un accidente
  técnico: es una señal de que el sitio no quiere consumo programático. Un robots.txt
  ausente no anula esa señal.
- **Recomendación: NO raspar el archivo histórico completo por cuenta propia.** Sesenta
  y dos mil páginas HTML sobre un servicio público es exactamente lo que la regla de
  cortesía prohíbe, y además contradice una decisión explícita del dueño del sitio.
- **El camino correcto es pedirlo.** El archivo existe y está ordenado; el MTT puede
  entregarlo en un archivo. La conversación es mucho más barata que el raspado, y deja
  al proyecto del lado correcto.

**11. Para qué sirve en el modelo**
Si se consigue el archivo por la vía formal, es la **serie histórica de cortes de ruta
declarados por el MTT**: complementa a F-T04/F-T05 (que registran daño a la
infraestructura) con el corte tal como se le comunica al público. La entrada del *Paso
Los Libertadores cerrado por nieve* muestra bien el tipo de evento que sirve.

Sin el archivo, sirve como **fuente de estado del día**, leyendo la página vigente —
que es un puñado de peticiones, no un raspado.

**12. Utilidad** · **★★**. Es la fuente de estado que Alexis señaló, y tiene historia
real. Baja de ★★★ por dos razones concretas: **cobertura territorial parcial** (RM más
cuatro regiones, cuando el problema está en el Chile rural que no cubre) y **acceso
programático explícitamente cerrado**.

**13. Riesgos y límites**

- La cobertura no calza con el problema. Transporte Informa mira la congestión urbana;
  el aislamiento de pueblos ocurre donde el sitio no llega.
- Texto libre y sin identificador estable de ruta: no hay `ROL` ni kilómetro. Ligar
  «Camino Lonquén – 4 Poniente» a un tramo del inventario del MOP es trabajo manual o
  de geocodificación, con error. **Las capas del MOP, en cambio, ya vienen con `ROL` y
  kilómetro** — por eso son mejor cimiento.
- Una de las fechas del sitemap es `1970-01-01`, o sea un registro con fecha inválida.
  Los datos del archivo tienen ruido.

---

## F-T08 · API de la UOCT (semáforos y flujo, Santiago) ★

**1. Organismo y producto** · Unidad Operativa de Control de Tránsito (UOCT), MTT. Es
el sistema que alimenta a Transporte Informa en su parte de tránsito.

**2. URL exacta y endpoint** — VERIFICADO
`https://api.uoct.cl/api/v1/junction/general-info`
Respuesta real obtenida: `{"result":"ok","data":[{"total_junctions":3501,"by_status":
[{"status_id":1,"count":2362}, ... ]}]}`
(La raíz `https://api.uoct.cl/api/v1/` devuelve 404; hay que ir al endpoint concreto.
No hay robots.txt: 404.)

**3. Qué publica** · estado de 3.501 cruces semaforizados, agregado por estado.

**4. Familia** · **ESTADO-CONSECUENCIA** (indirecta: un semáforo caído delata corte de
energía o falla de comunicaciones).

**5. Formato** · JSON limpio.

**6. Acceso** · **anónimo, VERIFICADO.** Sin clave.

**7. Granularidad** · Región Metropolitana. Este endpoint entrega **sólo el total
agregado**, sin ubicación por cruce.

**8. Frecuencia** · **SUPUESTO**: tiempo real o casi.

**9. Historia** · **ninguna** por este endpoint. Sólo la foto de ahora.

**10. Condiciones de uso** · sin robots.txt, sin términos publicados localizados. Es
una API interna que quedó accesible, no un producto de datos abiertos declarado. **No
la trataría como fuente estable ni la consultaría con frecuencia alta.**

**11. Para qué sirve en el modelo**
Poco, hoy. Como curiosidad útil: los semáforos apagados son un **indicador indirecto de
corte eléctrico masivo**, porque dependen de la red. Pero eso es dominio de servicios
básicos, no mío, y sólo cubre Santiago.

**12. Utilidad** · **★**. No cambia la prioridad de ningún activo de infraestructura
crítica en el sentido del proyecto.

**13. Riesgos y límites** · API sin documentación pública, sin garantía de continuidad,
sin términos de uso. Los `status_id` son números sin diccionario publicado. Cobertura
sólo metropolitana. **No construir nada que dependa de ella.**

---

## F-T09 · Observatorio Logístico — MTT (puertos, ferrocarril, aéreo, vial) ★★

**1. Organismo y producto** · Ministerio de Transportes y Telecomunicaciones ·
Programa de Desarrollo Logístico. **Observatorio Logístico**, con **API pública y
documentada**.

**2. URL exacta y endpoint** — VERIFICADO

- Portal: `https://www.observatoriologistico.cl/`
- **API base: `https://www.observatoriologistico.cl/api/v1`**
- Documentación oficial:
  `https://filesprod.observatoriologistico.cl/assets/frontend-graficos/api-datastream-docs/datastreams_api_docs.html`
- Endpoints: `GET /datastreams/lists` · `GET /datastreams/{code}/metadata` ·
  `GET /datastreams/{code}/data`
- Llamada real probada: `.../api/v1/datastreams/lists?limit=100` → **48 conjuntos de
  datos**, respuesta JSON con `status`, `data`, `pagination`.

**3. Qué publica** · 48 conjuntos. Los pertinentes a este dominio, con su código real:

| Código | Título |
|---|---|
| `P065` | Red Vial Nacional Distancia (por región y tipo de carpeta) |
| `P066` | Red Vial Nacional Doble Calzada |
| `P012` | Toneladas transferidas por puertos estatales, mensualizado |
| `P013` | Indicadores operacionales en puertos estatales |
| `P038` | Tamaño máximo de lote (TEU) en puertos estatales |
| `C018` | Carga transferida en puertos por tipo de carga y región |
| `M003` | Maestro de puerto y su respectiva región de Chile |
| `M013` | Maestro de terminales de puertos estatales con PORTCODE |
| `P039` | Carga transportada en ferrocarril, mensualizado |
| `P040` | Toneladas transferidas por puertos transportadas en ferrocarril |
| `C086` | Envíos postal y courier de ingreso por aeropuerto |

**4. Familia** · **ACTIVO** (los maestros de puertos y terminales) y **contexto de
consecuencia** (los flujos de carga).

**5. Formato** · JSON. Paginado (1 a 100 por página). CORS habilitado.

**6. Acceso** · **anónimo, VERIFICADO.** La documentación lo dice literal: *«No
requiere autenticación ni registro previo. Acceso inmediato para desarrollo y
producción.»* Límite declarado: **100 peticiones por minuto por IP**, con HTTP 429 y
cabecera `Retry-After` si se excede. Es la única fuente de este dominio con **reglas
de uso escritas**, lo que la vuelve la más segura de automatizar.

**7. Granularidad territorial** · **regional y por puerto/terminal**. No llega a
comuna. El maestro `M003` da la región de cada puerto; `M013` da los terminales con su
código internacional.

**8. Frecuencia** · **al día**: los conjuntos consultados traen
`date_last_update: 2026-08-15T21:40:49-04:00`, o sea de hoy mismo. Los datos de flujo
son mensuales.

**9. Historia** · **SUPUESTO**: las series mensualizadas (`P012`, `P039`) deberían
traer años de historia, pero no consulté el contenido para no cargar el servicio.
**Pendiente**: pedir `/datastreams/P012/metadata`, que es una sola llamada y aclara el
rango.

**10. Condiciones de uso** · robots.txt verificado (es un Drupal estándar, no prohíbe
la API). Documentación con límite de uso explícito. **Sí permite uso automatizado**,
respetando las 100 peticiones por minuto. La más limpia del lote.

**11. Para qué sirve en el modelo**
Da lo que las capas del MOP no dan: **cuánto pasa por cada nodo**. Un puerto que mueve
el 40% de la carga del país y otro que mueve el 0,3% no pueden tener la misma
prioridad, aunque ambos sean «puerto» en la matriz. Los conjuntos `P012`, `P013` y
`C018` permiten **ponderar los activos portuarios por importancia real**, en vez de
por categoría. `P039` y `P040` hacen lo mismo con el ferrocarril, y de paso resuelven
el encargo de EFE sin tener que ir a EFE (ver F-T12).

**12. Utilidad** · **★★**. Cambia prioridades dentro del sector portuario y
ferroviario, pero no aporta al problema del aislamiento vial, que es el corazón del
asunto.

**13. Riesgos y límites**
Es dato **estadístico**, no georreferenciado: no ubica nada en el mapa por sí solo (hay
que unirlo por nombre de puerto con F-T11, con el riesgo de nombres que no calzan).
Granularidad regional, no comunal. Y son promedios: sirven para priorizar en frío, no
para operar una emergencia.

---

## F-T10 · Red Aeroportuaria Nacional — MOP / Dirección de Aeropuertos ★★

**1. Organismo y producto** · MOP · Dirección de Aeropuertos · División de
Infraestructura Aeroportuaria (UGIT). **Red Aeroportuaria Nacional.**

**2. URL exacta y endpoint** — VERIFICADO
`https://rest-sit.mop.gob.cl/arcgis/rest/services/DAP/Red_Aeroportuaria_Nacional/MapServer/0`
Conteo probado: **316 recintos**.

**3. Qué publica**
Descripción literal: *«Ubicación de la infraestructura aeroportuaria a nivel nacional,
diferenciando entre aeropuertos y aeródromos, así como si estos pertenecen a la red
primaria, secundaria o pequeños aeródromos»*.

Campos verificados: `NOMBRE`, `CODIGO_IATA`, `COD_OACI`, `ADMINISTRACION`, **`REGION`,
`PROVINCIA`, `COMUNA`, `LOCALIDAD`**, **`RED`** (primaria/secundaria/pequeños), `TIPO`,
`USO`, `PROPIEDAD`, `HORAS_OPERA`, `LATITUD`, `LONGITUD`, `FECHA_ACTUALIZACION`,
`FUENTE`, `PUBLICABLE`, `MACRO`.

**4. Familia** · **ACTIVO**.

**5. Formato** · ArcGIS REST (JSON), punto. Además trae latitud y longitud como campos
de texto, lo que facilita usarlo sin herramientas de mapas.

**6. Acceso** · **anónimo, VERIFICADO.**

**7. Granularidad** · **excelente: llega a comuna y localidad**, ya en el dato. Sin
cruces espaciales.

**8. Frecuencia** · trae campo `FECHA_ACTUALIZACION` por registro. **SUPUESTO**:
actualización por evento, no calendarizada.

**9. Historia** · ninguna.

**10. Condiciones de uso** · sin robots.txt en el host. Mismo régimen que el resto de
`rest-sit.mop.gob.cl`.

**11. Para qué sirve en el modelo**
Es el **inventario aeroportuario ubicado**, con dos regalos: el campo `RED`
(primaria / secundaria / pequeños aeródromos) es **una priorización ya hecha por el
propio MOP**, igual que la RVE en lo vial; y llega a comuna sin trabajo extra.

Y hay un uso que se conecta directo con el aislamiento: **en el Chile austral y en las
islas, el aeródromo ES el acceso alternativo.** Cuando el glosario pregunta si existe
«acceso alternativo», en Aysén o Magallanes la respuesta muchas veces es un aeródromo
de 316 de esta lista. Sin esta capa, el cálculo de aislamiento daría falsos positivos
en todo el sur.

**12. Utilidad** · **★★**. No cambia prioridades por sí sola, pero **es indispensable
para no equivocarse en el cálculo de aislamiento** donde no hay caminos.

**13. Riesgos y límites**
Hay un campo `PUBLICABLE` cuyo significado no está documentado: podría haber registros
que el MOP no autoriza a difundir. **Revisar sus valores antes de publicar la capa
completa.** El dato es de infraestructura, no de operación: no dice si el aeródromo
está operativo hoy (eso sería DGAC, ver F-T13). `HORAS_OPERA` no aclara si es horario
o disponibilidad.

---

## F-T11 · Catastro de infraestructura portuaria — MOP / Dirección de Obras Portuarias ★★

**1. Organismo y producto** · MOP · Dirección de Obras Portuarias. **Catastro DOP.**

**2. URL exacta y endpoint** — VERIFICADO
`https://rest-sit.mop.gob.cl/arcgis/rest/services/DOP/CATASTRO_DOP/MapServer/0`
Conteo probado: **644 obras**.

**3. Qué publica**
Descripción literal: *«infraestructura Portuaria a nivel nacional... Dicha
infraestructura considera: **Caletas, Muelles y Varaderos**»*.
Campos verificados: `NOMBRE`, `LOCATION`, **`OPERATIVA`**, `COD_REG`, `PROVINCIA`,
`COMUNA`.

**4. Familia** · **ACTIVO**.

**5. Formato** · ArcGIS REST (JSON) y WMS, punto.

**6. Acceso** · **anónimo, VERIFICADO.**

**7. Granularidad** · **llega a comuna y provincia**, ya en el dato.

**8. Frecuencia** · **no declarada. SUPUESTO.**

**9. Historia** · ninguna.

**10. Condiciones de uso** · mismo régimen que el resto de `rest-sit.mop.gob.cl`.

**11. Para qué sirve en el modelo**
Aquí está el mismo argumento que en los aeródromos, y quizá más fuerte: **en el litoral
de Chiloé, Aysén y Magallanes, la caleta y el muelle son el acceso**, no una comodidad.
Un pueblo costero cuyo único camino se corta **no queda aislado si tiene muelle
operativo** — y el campo `OPERATIVA` es justamente el que lo dice.

Además, 644 obras es un número grande de activos hoy invisibles para la matriz: caletas
pesqueras que son a la vez sustento económico y vía de evacuación.

**12. Utilidad** · **★★**. Igual que F-T10: corrige el cálculo de aislamiento donde el
camino no es la única vía.

**13. Riesgos y límites**
Sólo 6 campos: es un catastro pobre comparado con el de puentes. No hay tamaño, ni
estado estructural, ni capacidad. `OPERATIVA` no está documentada (¿qué valores toma,
con qué fecha de corte?): **verificar antes de confiar**. No incluye los puertos
estatales grandes, que son de las Empresas Portuarias, no de la DOP — para ésos hay que
ir a F-T09.

---

## F-T12 · EFE — Ferrocarriles ⛔ NO AUTOMATIZAR

**1. Organismo y producto** · Empresa de los Ferrocarriles del Estado.

**2. URL** · `https://www.efe.cl/` · robots.txt verificado en `https://www.efe.cl/robots.txt`

**3. Qué publica** · información corporativa y de servicios. No revisada en detalle,
por lo que sigue.

**4. Familia** · ACTIVO (potencial).

**5. Formato** · HTML.

**6. Acceso** · ⛔ **RESTRINGIDO PARA AGENTES AUTOMÁTICOS. VERIFICADO.**

**10. Condiciones de uso** · el robots.txt de EFE contiene, literal:

```
User-agent: ClaudeBot
Disallow: /
User-agent: GPTBot
Disallow: /
User-agent: OAI-SearchBot
Disallow: /
User-agent: ChatGPT-User
Disallow: /
...
```

EFE **prohíbe explícitamente** el acceso automatizado de agentes de inteligencia
artificial, nombrando a ClaudeBot en primer lugar. **No corresponde raspar este sitio,
ni con otro nombre de agente.** Es una decisión declarada del dueño del sitio y se
respeta. No propongo ninguna forma de eludirlo.

**11. Para qué sirve en el modelo**
Igual hay salida, y buena: **los datos ferroviarios que el proyecto necesita están en
el Observatorio Logístico del MTT (F-T09), en una API que sí autoriza el uso
automatizado** — conjuntos `P039` (carga transportada en ferrocarril) y `P040`. Y el
MTT publica además la encuesta anual de ferrocarriles vía datos.gob.cl.
Para la red férrea georreferenciada, la vía correcta es **pedirla formalmente a EFE**,
no bajarla.

**12. Utilidad** · **★**. Por acceso, no por contenido.

**13. Riesgos y límites** · el riesgo principal es de conducta: automatizar acá sería
incumplir una restricción explícita. Anotado para que nadie lo intente por
desconocimiento.

---

## F-T13 · Aeronáutica y marítimo operacional — DGAC, JAC, DIRECTEMAR ⚠️ pendiente

Los agrupo porque a los tres les pasa lo mismo: **el portal responde, pero no localicé
el dato estructurado** y no insistí más, según la regla de no empantanarse.

| Organismo | URL verificada | Resultado |
|---|---|---|
| **DGAC** | `https://www.dgac.gob.cl/` | HTTP 200. robots.txt permite todo salvo `/wp-admin/`. **No localicé** producto de datos de estado operacional de aeródromos. |
| **JAC** (Junta de Aeronáutica Civil) | `https://www.jac.gob.cl/` | HTTP 200. **Sin robots.txt** (404). Publica estadística aérea; no verifiqué formato. |
| **DIRECTEMAR** | `https://www.directemar.cl/` | robots.txt verificado, sin prohibiciones (sólo declara sitemap). Probé una ruta de «estado de puertos» y devolvió **404**: **no existe** con esa dirección. **No invento la URL correcta.** |

**Qué falta hacer, concretamente:**

1. **DIRECTEMAR es la que más importa** de las tres. Cierra puertos por temporal —eso
   es estado-consecuencia puro— y el documento de integración con SENAPRED ya la
   nombra como fuente de la cadena («Aviso de Condiciones Meteorológicas Especiales»).
   Encontrar la ruta correcta de «estado de puertos» dentro de su sitio queda como
   **pendiente prioritario de este dominio**.
2. **DGAC**: buscar si publica NOTAM o estado de aeródromos en formato consultable.
3. **JAC**: estadística de pasajeros y carga por aeropuerto, para ponderar los 316
   recintos de F-T10 igual que F-T09 pondera los puertos.

**Utilidad estimada** · **★★ SUPUESTO** para DIRECTEMAR, ★ para las otras dos. No lo
puedo afirmar sin verificar.

---

## F-T14 · datos.gob.cl y geoportal.cl (catálogos) ★

**1. Organismo y producto** · Portal de Datos Abiertos del Estado (`datos.gob.cl`,
software CKAN) e Infraestructura de Datos Geoespaciales de Chile (`geoportal.cl`).

**2. URL exacta y endpoint** — VERIFICADO
- API CKAN: `https://datos.gob.cl/api/3/action/package_search?q=...`
  Llamada real probada, devolvió 3 resultados para «vialidad / puentes / red vial».
- Catálogo geoespacial: `https://www.geoportal.cl/geoportal/catalog?categories[]=transportation&action=search`
  (devuelve 301, hay que seguir la redirección).

**3. Qué publica** · fichas de catálogo, no los datos. Las tres fichas halladas:
«Red Vial Nacional» (×2, Dirección de Vialidad) y «Categoría Geoespacial: Transporte»
(IDE Chile).

**4. Familia** · ninguna — es un índice.

**5. Formato** · JSON (CKAN) y HTML.

**6. Acceso** · **anónimo, VERIFICADO.**

**8. Frecuencia** · las fichas de Red Vial Nacional están **modificadas por última vez
en 2015** y apuntan a `wservice-sit.mop.gov.cl/ArcGIS/...`, un servidor **antiguo** que
no es el que hoy funciona. El catálogo está desactualizado.

**10. Condiciones de uso** · ⚠️ **importante**: el robots.txt de `datos.gob.cl`
contiene `Disallow: /api/` y `Crawl-Delay: 10`. Es decir, **el propio portal pide que
no se rastree su API**. Se puede consultar puntualmente (como hice, una vez), pero **no
corresponde automatizar consultas sistemáticas contra ella**. `geoportal.cl/robots.txt`
verificado: `Disallow:` vacío, o sea permite todo.

**11. Para qué sirve en el modelo**
Sólo para dos cosas: **descubrir** fuentes nuevas, y **leer la licencia declarada** —
que es como encontré que la Red Vial Nacional figura como **Creative Commons
No-Comercial**. Ese hallazgo, por sí solo, justificó la consulta.

**12. Utilidad** · **★**. No cambia ninguna prioridad.

**13. Riesgos y límites**
**El catálogo miente por omisión.** Si el proyecto se hubiera guiado sólo por
datos.gob.cl, habría concluido que el inventario vial de Chile es una ficha de 2015 con
un enlace roto. La realidad —23 servicios de Vialidad vivos y actualizados hace diez
días— está **fuera del catálogo oficial**. Lección para el resto del proyecto: **ir al
organismo, no al catálogo.**

---

## F-T15 · Publicaciones de Vialidad (contexto y validación) ★

**1. Organismo y producto** · MOP · Dirección de Vialidad, Departamento de Gestión
Vial. **«Red Vial Nacional: Dimensionamiento y Características»**, anuario.

**2. URL exacta** — VERIFICADO (HTTP 200, `application/pdf`, 8,4 MB)
`https://vialidad.mop.gob.cl/uploads/sites/9/2025/01/Red_Vial_Nacional_Dimensionamiento_y_Caracteristicas_2022.pdf`
Índice de informes: `https://vialidad.mop.gob.cl/document/informes-y-estudios/`
Gestión vial: `https://vialidad.mop.gob.cl/gestion-vial/`
Plan Nacional de Censo Vial: `https://vialidad.mop.gob.cl/plan-nacional-de-censo-vial/`
(existe también el servicio `.../VIALIDAD/Plan_Nacional_de_Censos/MapServer`)

**3. Qué publica** · kilómetros de camino por región y tipo de pavimento, más conteos
de puentes, pasarelas, ciclovías y túneles. Edición anual.

**4. Familia** · contexto / validación.

**5. Formato** · **PDF.** No estructurado.

**6. Acceso** · **anónimo, VERIFICADO.**

**7. Granularidad** · **regional**, no comunal.

**8. Frecuencia** · anual. La edición hallada es **2022**, publicada en enero de 2025:
hay unos dos años de rezago.

**9. Historia** · sí, hay ediciones anteriores (verifiqué la existencia de un
`a2016.pdf` en el mismo servidor). Serie larga en PDF.

**10. Condiciones de uso** · robots.txt de `mop.gob.cl` verificado: sólo bloquea
`/wp-admin/`. Descarga individual sin problema.

**11. Para qué sirve en el modelo**
**Como control de calidad, no como fuente.** Antes de usar la capa de F-T01 en serio
conviene sumar sus kilómetros por región y comparar con el total oficial del anuario.
Si no calzan, la capa está incompleta y hay que averiguar por qué **antes** de
construir nada encima. Es la misma disciplina de las 38 estaciones de Coquimbo
auditando ERA5 en Cosmoclima: un número independiente contra el que chocar el nuestro.

**12. Utilidad** · **★**. No cambia prioridades, pero evita construir sobre arena.

**13. Riesgos y límites** · PDF, o sea trabajo manual para sacar las tablas. Rezago de
dos años. Granularidad regional insuficiente para el modelo.

---

# Servicios verificados que no alcancé a fichar

Todos existen y responden en `https://rest-sit.mop.gob.cl/arcgis/rest/services/VIALIDAD/`
(listado completo obtenido del servidor). Los dejo anotados porque varios pueden
resultar valiosos y ya sé que están vivos:

| Servicio | Por qué puede importar |
|---|---|
| `Estado_Red_Vial_Pavimentada` | capa «Estado» — **posiblemente la condición del pavimento**, que es justo lo que falta en puentes. Merece revisión temprana. |
| `Catastro_Vial` | 10 capas: puentes, saneamiento, señales, paraderos, soleras, carpeta. **«Saneamiento» es drenaje** — clave para riesgo de lluvia. |
| `Pasos_Fronterizos` y `Restricciones_Vialidad` (capa «Estado Pasos Fronterizos») | estado de pasos cordilleranos, que es de los cortes más frecuentes (el propio Paso Los Libertadores del ejemplo). |
| `Emergencias_Vialidad` | tres capas con **nombres corruptos** (`'__'`); revisar con cuidado de codificación. |
| `EMERNORTE`, `Emergencia_Mayo` | servicios de episodios concretos; probablemente el detalle fino de eventos pasados. |
| `MAPA_ESTACIONES_DGA` | estaciones hidrométricas — es dominio de otro agente, lo anoto por si sirve. |
| `Zonas_de_Descanso`, `EGC_y_Control_Pesaje`, `Pesaje_Rutas_Tipicas` | activos menores y flujo de camiones. |
| `Iniciativas_De_Inversion_Etapas`, `Ordenes_Trabajo_CAD`, `Contratos_Globales_CC` | obras en curso: un camino en obra tiene disponibilidad reducida. |
| `DIRPLAN/PROYECTOS_CCLIMATICO` | **proyectos MOP asociados a cambio climático** — puede indicar dónde el propio ministerio ya reconoce vulnerabilidad climática. |

⚠️ **Advertencia de encuadre**: la abundancia es una trampa. Hay más de 30 servicios
disponibles y la tentación es integrarlos todos. **Con F-T01 (red vial), F-T02
(puentes) y F-T04/F-T05 (emergencias) el proyecto ya puede hacer algo que hoy nadie
hace.** Lo demás es refinamiento posterior.

---

# Cierre: qué hacer con esto

**Lo que cambió con esta revisión.** El proyecto entró a este dominio diciendo «hoy no
tenemos ningún inventario vial ubicado». Sale con 14.039 tramos de camino y 6.742
puentes georreferenciados, descargables sin registro, actualizados hace diez días —
más once años de emergencias ubicadas en el mismo sistema de rol de ruta y kilómetro.

**Las tres cosas que valen más, en orden:**

1. **El inventario vial permite calcular el aislamiento.** Deja de ser una definición
   de glosario y pasa a ser un número: con la geometría se arma el grafo y se buscan
   los tramos cuya caída desconecta una localidad. Ése es el argumento que da vuelta el
   `PF = 0,41` de las rutas rurales, y ahora es calculable.
2. **`CAUCE_QUEB` en el catastro de puentes.** Un campo de texto que liga cada puente
   al río que cruza. Cuando la DMC anuncia lluvia en una zona, se puede listar qué
   puentes están sobre esos cauces. Es el cruce que el proyecto dice que nadie hace, y
   sale con dos columnas.
3. **Once años de emergencias con rol de ruta y kilómetro.** El proyecto tiene contra
   qué validarse sin esperar a que ocurra el próximo temporal.

**Cuatro advertencias que no se pueden perder:**

- ⚠️ **El estado de los puentes no se publica** (`EST_PUENTE` viene vacío). Sabemos
  dónde están y de qué son, no cómo están. Es lo que más se querría y no está.
- ⚠️ **Licencia no-comercial declarada** para la Red Vial Nacional en datos.gob.cl.
  Conviene confirmarlo por escrito con la UGIT antes de publicar derivados.
- ⚠️ **Transporte Informa apagó su API a propósito y EFE prohíbe expresamente a
  ClaudeBot.** En ambos casos el camino es pedir el dato, no tomarlo.
- ⚠️ **Sesgo estructural en todo el dominio**: todas estas capas cubren sólo
  infraestructura **de tuición MOP**. Los caminos municipales, privados y forestales —
  que son precisamente los que aíslan pueblos chicos— **no están**. El sesgo va
  exactamente en contra de lo que el proyecto busca, y hay que declararlo en cualquier
  resultado, nunca compensarlo inventando.

**Lo que queda pendiente en este dominio:**

1. Encontrar la ruta correcta del estado de puertos de **DIRECTEMAR** (la probada dio
   404; no inventé otra).
2. Revisar los campos de la **Red Vial Estructurante** y pedir su metodología.
3. Consultar `Estado_Red_Vial_Pavimentada` — puede traer la condición que falta.
4. Verificar la frescura real de `EMERGENCIA_MOP` (máximo de `FECHA` y valores de
   `ESTADO_EMER`): una sola consulta lo aclara.
5. Cruzar la suma de kilómetros de la capa contra el anuario de Vialidad, antes de
   construir nada encima.
6. Preguntar formalmente a la **UGIT de Vialidad** por: licencia, calendario de
   actualización, y si existe un índice de condición de puentes que no se publica.

---

*Fichas levantadas y verificadas el 15-16 de agosto de 2026. Todas las URL, conteos y
campos citados provienen de respuestas reales del servidor, no de documentación ni de
memoria. Lo no verificado está marcado SUPUESTO.*
