# Catastro de fuentes — AMENAZAS GEOFÍSICAS

Dominio: sismos (CSN), volcanes (SERNAGEOMIN/OVDAS), **remoción en masa** (SERNAGEOMIN),
tsunami (SHOA), relaves (SERNAGEOMIN).

Levantado el **15-ago-2026**. Todo lo marcado **VERIFICADO** se probó con una o dos
peticiones HTTP desde esta máquina, sin descargas masivas. Lo marcado **SUPUESTO** no se
probó y no debe usarse como si estuviera confirmado.

---

## 0. Respuesta a la pregunta prioritaria: la Minuta Técnica de SERNAGEOMIN

Las cuatro preguntas que había que responder, respondidas con dato en mano:

**¿Es accesible de forma programática?** **SÍ, y muchísimo mejor de lo esperado.**
No es un PDF: es una capa geográfica viva en un servidor ArcGIS público de SERNAGEOMIN,
que entrega **GeoJSON con polígonos y nivel de peligro**, sin registro ni clave.
Verificado hoy con una consulta real que devolvió geometría.

**¿Hay archivo histórico?** **NO. Y ésta es la mala noticia del día.**
La capa vigente se **sobrescribe encima de sí misma** cada vez que se emite una minuta
nueva. Hoy trae 25 zonas con nivel asignado y el campo `FECHA` (que existe y se llama
literalmente «Fecha emisión minuta técnica») viene **vacío en las 25**. Es decir: podemos
leer la foto de hoy, pero **la foto de ayer ya no existe en ninguna parte**.
Analogía: es un pizarrón, no un cuaderno. Cada mañana alguien borra y vuelve a escribir.

Consecuencia dura y directa: **si queremos validar contra la minuta, tenemos que empezar a
guardar la foto nosotros, desde mañana.** Un snapshot diario del GeoJSON, fechado. Cada
día que pasa sin hacerlo es un día de historia que se pierde para siempre. Es barato
(unos pocos cientos de KB por día) y es irrecuperable si no se hace.

Hay **dos** historias parciales que sí existen, y valen mucho:
- **380 eventos reales de remoción en masa, de 1996 a 2026**, georreferenciados, con
  comuna, tipo y detonante (los ReTeRM). Eso no es el pronóstico, es **lo que efectivamente
  pasó** — que para validar es incluso mejor. Ver ficha G-2.
- Un servicio **congelado** que sí guardó la cadena completa DMC→minuta entre
  **may-2021 y jun-2023** (174 eventos meteorológicos con su código `AA`/`A`, más el
  pronóstico de precipitación en mm y la isoterma 0 °C). Dejó de actualizarse en junio de
  2023, pero muestra **exactamente cómo se acopla la minuta al aviso de la DMC**. Ver G-3.

**¿Qué zonas geográficas usa exactamente?** Se llaman **zonas morfoclimáticas** y están
publicadas como polígonos (119 en total, catálogo nacional completo). Los nombres reales,
extraídos de la capa:
- Norte: `Litoral` · `Cordillera Costa` · `Pampa Nortina` · `Precordillera` ·
  `Precordillera-Salar` · `Cordillera`
- Centro-sur: `Litoral` · `Cordillera Costa` · `Valle Longitudinal` ·
  `Valles Precordilleranos` · `Precordillera` · `Cordillera` · `Cordillera-Lonquimay` ·
  `Depresión Central` · `Isla de Pascua`
- Austral: `Insular (Norte/Central/Sur)` · `Litoral - Interior (Norte/Sur)` ·
  `Patagonia subandina (Norte/Sur)` · `Cordillera Austral (Norte/Sur)` ·
  `Pampa Patagónica (Norte/Sur)` · `Chiloé` · y zonas ad-hoc por comuna
  (ej. `Comuna de Cisnes`, `Comuna de O'Higgins`)

Esto **resuelve el pendiente n.º 1 de INTEGRACION_SENAPRED.md** («conseguir la capa de
zonas geográficas — no la tenemos»). Ya la tenemos, y es descargable.

**Corrección importante al documento del proyecto:** los niveles **no son tres, son
cuatro**. El diccionario oficial de la capa dice:
`MT-POSOC-01=Baja` · `MT-POSOC-02=Moderada` · `MT-POSOC-03=Alta` · **`MT-POSOC-04=Muy Alta`**.
Tanto `FUENTES_DE_DATOS_NACIONALES.md` como `INTEGRACION_SENAPRED.md` dicen «tres niveles
(Alta/Moderada/Baja)». Falta «Muy Alta». Si el `FEN` se calibra contra tres, queda mal
calzado en el extremo superior justo donde más importa.

**¿Con qué periodicidad se emite?** No hay calendario fijo: **se emite por evento**, no por
reloj. Se dispara cuando la DMC anuncia un sistema frontal o precipitación relevante. La
capa vigente fue editada por última vez el **13-ago-2026** y hoy tiene nivel asignado sólo
en las regiones de Arica-Parinacota a Coquimbo (4 zonas en `Alta`, 7 en `Moderada`,
14 en `Baja`) — coherente con un evento activo en el norte. Fuera de evento, la capa queda
sin niveles asignados. Es decir: **la ausencia de nivel no es «peligro bajo», es «no hay
minuta vigente»**, y el consolidador tiene que distinguir esas dos cosas o va a mentir.

**Bonus no buscado:** en el mismo servicio viene la capa `COMUNAS_2020` con las **345
comunas y sus códigos CUT** (región/provincia/comuna). Eso resuelve también el pendiente
n.º 2 (hallazgo H-13, «capa de límites comunales oficial»), y viene del mismo organismo que
publica la amenaza — o sea, las dos geografías del problema de la sección 4 de
INTEGRACION_SENAPRED vienen ya empaquetadas juntas y calzando entre sí.

---

## G-1 · Minuta Técnica de peligro de remoción en masa — capa VIGENTE ★★★

1. **Organismo y producto** — SERNAGEOMIN, Unidad de Peligros Geológicos / Unidad de
   Emergencias (ATG). Producto: *Minuta Técnica de posibilidad de ocurrencia de remoción en
   masa*, capa `ZonasMorfoclimaticas_ATG_Flash_V2026`.
2. **URL / endpoint**
   - Servicio: `https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/services/minutasATG_Flash/FeatureServer`
   - Capa de la minuta: `.../FeatureServer/2`
   - Consulta GeoJSON:
     `.../FeatureServer/2/query?where=POS_OCURRENCIA+IS+NOT+NULL&outFields=ZONA,REGION,POS_OCURRENCIA,FECHA&f=geojson`
   - Visor humano: `https://experience.arcgis.com/experience/9afa8fc96a954bd79895c6cb19360949`
     (link corto institucional `https://arcg.is/081Hin1`, publicado por el propio SERNAGEOMIN)
3. **Qué publica exactamente** — Polígono de zona morfoclimática + `ZONA` (texto) +
   `REGION` (texto) + `POS_OCURRENCIA` (categoría codificada, 4 valores:
   Baja/Moderada/Alta/Muy Alta) + `FECHA` (fecha de emisión de la minuta; **hoy viene
   vacía**). Sin unidades físicas: es una escala cualitativa de posibilidad de ocurrencia.
   119 polígonos en el catálogo; 25 con nivel asignado al 15-ago-2026.
4. **Familia** — AMENAZA (y la capa sin nivel asignado funciona además como BASE
   TERRITORIAL: es el catálogo nacional de zonas morfoclimáticas).
5. **Formato** — ArcGIS Feature Service: JSON o **GeoJSON** a elección (`f=geojson`).
   También sirve la geometría completa. Sistema de referencia de salida configurable
   (el nativo es Web Mercator 3857; se puede pedir WGS84 con `outSR=4326`).
6. **Acceso** — **anónimo, sin registro. VERIFICADO** (consulta real ejecutada hoy;
   devolvió polígono con coordenadas y nivel `Alta`). El servicio declara
   `capabilities: Query` para lectura pública.
7. **Granularidad territorial** — **zona geográfica** (zona morfoclimática dentro de una
   región). No es comuna: hay que cruzar. En el mismo servicio viene `COMUNAS_2020` (345
   comunas con CUT) para hacer ese cruce.
8. **Frecuencia** — **por evento, no por calendario.** Se emite cuando hay pronóstico
   meteorológico gatillante. Última edición registrada: 13-ago-2026.
9. **Historia** — **NO HAY.** La capa se sobrescribe. `FECHA` está vacía en todos los
   registros vigentes. **Sin snapshot propio, no hay serie histórica posible.**
10. **Condiciones de uso** — `https://www.sernageomin.cl/robots.txt` sólo bloquea
    `/wp-admin/`; el resto está permitido (**VERIFICADO**). El Feature Service está marcado
    como `access: public` en ArcGIS Online y el propio SERNAGEOMIN difunde el visor en su
    sala de prensa como «de libre acceso». No encontré un texto de licencia explícito en
    los metadatos del servicio (campo `licenseInfo` vacío) → **antes de operar en
    producción hay que pedir confirmación por escrito a SERNAGEOMIN**, sobre todo por el
    campo `copyrightText: "SERNAGEOMIN, OSIG; ATG"` que exige atribución.
11. **Para qué sirve en el modelo** — Es **el `FEN` dinámico de remoción en masa, ya
    hecho**. Entra directo como amenaza territorial y, además, es **la única vara
    independiente contra la cual validar nuestro `C_clim`**. Cruzada con los 835 elementos
    de la MICR da, sin más pasos, la lista de activos dentro de zona `Alta`/`Muy Alta`.
12. **Utilidad — ★★★.** Cambia la prioridad de un activo de forma inmediata y directa: una
    subestación dentro de un polígono `Muy Alta` sube de prioridad hoy y baja mañana.
    Es la fuente más valiosa de todo mi dominio.
13. **Riesgos y límites**
    - **Se borra sola.** Riesgo n.º 1. Mitigación: snapshot diario propio, desde ya.
    - **`FECHA` vacía** → hoy no se puede saber la vigencia temporal desde el dato mismo.
      Hay que registrar la hora de descarga como sustituto (y decir que es un sustituto).
    - **Nivel ausente ≠ peligro bajo.** Confundirlos produce un falso «todo tranquilo».
    - Depende de un ArcGIS Online alojado por Esri: si SERNAGEOMIN cambia de plataforma o
      renombra el servicio, el adaptador se cae. Debe degradar a «sin dato», no interpolar.
    - Los nombres de zona tienen ruido real (`Litoral ` con espacio final, `Cordillera
      Costa` vs `Cordillera de la Costa` en los comunicados). Hay que normalizar.

---

## G-2 · ReTeRM — eventos reales de remoción en masa, 1996-2026 ★★★

1. **Organismo y producto** — SERNAGEOMIN. *Reportes Técnicos de Remociones en Masa
   (ReTeRM)*: la evaluación en terreno que hace el Servicio después de (o durante) un
   evento, y que alimenta al COGRID y al equipo técnico de SENAPRED.
2. **URL / endpoint**
   - `https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/services/minutasATG_Flash/FeatureServer/0`
     (capa `Minutas_ATG_Flash_V2026`, 380 registros)
   - Complementaria, sólo la contingencia en curso (25 registros):
     `https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/services/MinutaATG_Contingente/FeatureServer/0`
3. **Qué publica exactamente** — Punto (coordenada UTM `Este`/`Norte` + geometría) con:
   `Nombre_Minuta` (identificador legible, ej. `2026-08-09_RETERM_Valparaiso_16_Avenida_Espana_MPS`),
   `Fecha_evento`, `Tipo` (taxonomía de remoción: deslizamiento, flujo de detritos, caída
   de rocas, etc.), **`Detonante`** (ej. «Lluvia intensa»), `Identificador`, y
   `REGION`/`PROVINCIA`/`COMUNA`.
4. **Familia** — AMENAZA (registro de ocurrencia observada) → uso principal: **VALIDACIÓN**.
5. **Formato** — ArcGIS Feature Service, JSON / GeoJSON.
6. **Acceso** — anónimo. **VERIFICADO** (380 registros contados, muestra leída).
7. **Granularidad territorial** — **punto**, con comuna/provincia/región ya asignadas en el
   propio registro. Es la granularidad ideal para cruzar con activos.
8. **Frecuencia** — según ocurra el evento. Los últimos registros son de agosto de 2026.
9. **Historia** — **1996-06-11 → 2026-08-08. VERIFICADO** con consulta de mín/máx.
   380 eventos. Ojo: la cobertura de los años 90 y 2000 es rala; el grueso es reciente.
10. **Condiciones de uso** — mismas que G-1 (robots permisivo, servicio público, atribución
    a SERNAGEOMIN). Sin licencia explícita → confirmar por escrito.
11. **Para qué sirve en el modelo** — Es **el conjunto de validación**. Permite preguntar:
    «de los eventos reales de los últimos 30 años, ¿cuántos cayeron en zonas que nuestro
    `C_clim` marcaba como peligrosas?». Sin esto, el modelo no tiene contra qué medirse.
    El campo `Detonante` además separa lo gatillado por lluvia de lo gatillado por sismo o
    por intervención antrópica — clave para no atribuirle al clima lo que no es del clima.
12. **Utilidad — ★★★.** No cambia la prioridad de un activo *hoy*, pero es lo que decide si
    el modelo que asigna prioridades sirve o no. Sin validación, todo lo demás es opinión.
13. **Riesgos y límites**
    - **Sesgo de reporte grave:** sólo existe ReTeRM donde SERNAGEOMIN fue a mirar. Se va a
      mirar donde hay gente y donde hubo daño. Entonces la ausencia de evento en cordillera
      despoblada **no significa que no pasó nada**. Tratarlo como presencia-solo, nunca como
      presencia-ausencia.
    - `Tipo` está sin normalizar: hay más de 100 variantes de texto libre para lo mismo
      («Deslizamiento de suelo », «Deslizamiento suelo», «Deslizamientos de Suelo»…).
      Requiere una tabla de homologación antes de contar nada por tipo.
    - Registros duplicados por modificación (`..._MPS` y `..._MPS_MOD` en la misma fecha y
      coordenada). Hay que deduplicar por `Identificador`.

---

## G-3 · Servicio «Minuta Técnica» con acople DMC (archivo 2021-2023) ★★

1. **Organismo y producto** — SERNAGEOMIN + DMC. Servicio `Minuta Técnica`: la minuta de
   remoción en masa **junto con** el insumo meteorológico que la gatilla.
2. **URL / endpoint**
   - `https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/services/Minuta_T%C3%A9cnica/FeatureServer`
   - Capas: `0` Isoterma 0 °C (línea) · `1` Minuta técnica (polígono, 127) ·
     `2` Zonas de pronóstico de precipitación DMC (polígono) · `3` Tabla de evento
     meteorológico (174) · `4` Tabla pronóstico de precipitaciones (739) ·
     `5` Isoterma 0 °C pronosticada
   - Visor: `https://sernageomin.maps.arcgis.com` ítem `14bdfecbb9154dbc976f958a3420cf52`
3. **Qué publica exactamente**
   - Capa 1: mismos campos que G-1 (`POS_OCURRENCIA` con las 4 categorías, `ZONA`,
     `REGION`, `FECHA`).
   - Tabla 3: `COD_AA` (código del aviso/alerta DMC, ej. `AA36-3/2023`, `A217-7/2023`),
     `FECHA`, `TIPO_AA` (Aviso/Alerta), `CONDICION` (condición sinóptica), `TITULO`, `TEXTO`.
   - Tabla 4: `COD_AA`, `COD_SECTOR`, `SECTOR`, `REGION`, `DIA_PRONOSTICO`,
     **`PP_MIN` y `PP_MAX` en mm** — el umbral cuantitativo por sector y por día.
   - Capa 0: `ALTURA` de la isoterma 0 °C en metros, por región.
4. **Familia** — AMENAZA + ESTADO-CONSECUENCIA (es la cadena causal explícita).
5. **Formato** — ArcGIS Feature Service, JSON / GeoJSON.
6. **Acceso** — anónimo. **VERIFICADO** (esquemas y conteos leídos; rango de fechas medido).
7. **Granularidad territorial** — zona morfoclimática (capa 1) y sector de pronóstico DMC
   (tabla 4).
8. **Frecuencia** — **ninguna: está congelado.** Último evento cargado, 24-jun-2023.
9. **Historia** — **29-may-2021 → 24-jun-2023, VERIFICADO.** 174 eventos meteorológicos y
   739 filas de pronóstico. La capa 1 tiene `FECHA` vacía, así que **no se puede reconstruir
   qué nivel tuvo cada zona en cada fecha pasada** — sólo el catálogo de eventos.
10. **Condiciones de uso** — igual que G-1. Sin licencia declarada.
11. **Para qué sirve en el modelo** — Es **el plano de ingeniería de la cadena
    DMC→SERNAGEOMIN**: muestra exactamente qué variables entran (mm de precipitación
    mín/máx por sector-día, altura de isoterma 0 °C) para producir un nivel de peligro por
    zona. Es el mejor documento disponible para diseñar nuestro `C_clim` de modo que hable
    el mismo idioma. Además da 2 años de códigos `AA`/`A` reales para calzar con la DMC.
12. **Utilidad — ★★.** No cambia la prioridad de ningún activo hoy (está congelado), pero
    define la forma que debe tener nuestro modelo. Vale como especificación, no como dato
    operativo.
13. **Riesgos y límites**
    - Congelado desde 2023: **no usarlo como si estuviera vivo**. Riesgo real de que un
      adaptador lo consulte y devuelva peligro de junio de 2023 como si fuera de hoy.
    - `REGIONES` viene vacío en los registros que revisé; el ámbito hay que leerlo del
      `TITULO`, que es texto libre.
    - Que exista este servicio *y* el `minutasATG_Flash` sugiere que SERNAGEOMIN migró de
      plataforma. Conviene preguntarles cuál es el canal oficial vigente antes de casarse.

---

## G-4 · Zonas morfoclimáticas y límites comunales (base territorial) ★★★

1. **Organismo y producto** — SERNAGEOMIN (OSIG). Catálogo nacional de zonas
   morfoclimáticas + capa comunal.
2. **URL / endpoint**
   - Zonas: `.../minutasATG_Flash/FeatureServer/2` (los 119 polígonos, con y sin nivel)
   - Comunas: `.../minutasATG_Flash/FeatureServer/3` (`COMUNAS_2020`, 345 registros)
3. **Qué publica exactamente** — Zonas: polígono + `ZONA` + `REGION`. Comunas: polígono +
   `CUT_REG`, `CUT_PROV`, `CUT_COM`, `REGION`, `PROVINCIA`, `COMUNA`, `SUPERFICIE`.
4. **Familia** — BASE TERRITORIAL.
5. **Formato** — ArcGIS Feature Service, JSON / GeoJSON.
6. **Acceso** — anónimo. **VERIFICADO** (conteos 119 y 345 obtenidos).
7. **Granularidad territorial** — zona geográfica y comuna, respectivamente.
8. **Frecuencia** — capas estables; cambian sólo si se redefine la zonificación.
9. **Historia** — no aplica (son capas de referencia, no serie temporal).
10. **Condiciones de uso** — igual que G-1.
11. **Para qué sirve en el modelo** — Es **el traductor entre las dos geografías** del
    problema descrito en la sección 4 de `INTEGRACION_SENAPRED.md`: con las 39
    subestaciones georreferenciadas se hace un cruce espacial contra las dos capas y cada
    activo queda con su zona morfoclimática **y** su comuna. Eso permite recibir la amenaza
    en el idioma de SERNAGEOMIN y entregar el insumo en el idioma de SENAPRED.
12. **Utilidad — ★★★.** Sin esto, ningún dato de amenaza se puede cruzar con ningún activo.
    Es la pieza que hace posible el producto entero.
13. **Riesgos y límites**
    - `COMUNAS_2020` trae 345 comunas y Chile tiene 346: falta al menos una (verificar si es
      la Antártica). Antes de usarla como capa oficial, compararla con la del INE/IDE Chile.
    - Es una copia que mantiene SERNAGEOMIN para su propio uso, no la capa oficial del
      Estado. Ventaja: calza perfecto con su minuta. Desventaja: puede quedar desactualizada
      respecto de la oficial.
    - Los polígonos vienen en Web Mercator (EPSG:3857). Para cálculos de área o distancia
      hay que reproyectar a UTM local o el resultado sale mal (Web Mercator deforma con la
      latitud, y Chile es larguísimo en latitud).

---

## G-5 · Mapas de peligro de remoción en masa por localidad (estudios) ★★

1. **Organismo y producto** — SERNAGEOMIN. *Visor Cartografía de Estudios de Peligros
   Geológicos*: catálogo de los mapas de peligro elaborados por el Servicio.
2. **URL / endpoint**
   - `https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/services/Visor_Estudios_RM_WFL1/FeatureServer/0`
   - Visor: `https://experience.arcgis.com/experience/8dbb736ebf0549f8aa98974f9bf33072`
3. **Qué publica exactamente** — 78 registros con `CODIGO`, `TITU_MAP` (título del mapa),
   `TIPO_PELI` (tipo de peligro), `REF_BIBLIO`, `REGION`, `ANO`, `AREA_EST` (área
   estudiada), `Tipo_entregable`, `Escala`.
4. **Familia** — AMENAZA (peligro de base, estático).
5. **Formato** — ArcGIS Feature Service, JSON / GeoJSON.
6. **Acceso** — anónimo. **VERIFICADO** (78 registros contados).
7. **Granularidad territorial** — polígono de área estudiada (localidad / cuenca).
8. **Frecuencia** — se agrega un estudio cuando se publica; ritmo de años.
9. **Historia** — campo `ANO` por estudio; no lo recorrí en detalle.
10. **Condiciones de uso** — igual que G-1. Los mapas en sí (PDF) están en el catálogo de la
    biblioteca institucional y en `tiendadigital.sernageomin.cl` — **algunos son de pago**;
    lo que está abierto es el índice de dónde hay estudio.
11. **Para qué sirve en el modelo** — Es la **susceptibilidad de base**, el terreno: dónde
    el suelo es propenso aunque no llueva. La minuta (G-1) es el gatillo dinámico; esto es
    la predisposición estática. Un activo en zona con estudio de peligro alto **y** minuta
    `Alta` es doblemente prioritario.
12. **Utilidad — ★★.** Sí cambia la prioridad de un activo, pero sólo donde hay estudio
    (78 áreas, no el país entero), y no cambia día a día.
13. **Riesgos y límites**
    - Cobertura parcial y desigual: hay estudio donde hubo desastre o presión de
      planificación. **Falta de estudio ≠ falta de peligro.**
    - Escalas heterogéneas: mezclar un mapa 1:25.000 con uno 1:100.000 sin declararlo
      produce falsa precisión.
    - Lo que se descarga acá es el **índice**, no la cartografía de peligro en sí.

---

## G-6 · RNVV / OVDAS — alertas volcánicas ★★

1. **Organismo y producto** — SERNAGEOMIN, Red Nacional de Vigilancia Volcánica (RNVV),
   Observatorio Volcanológico de los Andes del Sur (OVDAS). Producto: nivel de alerta
   volcánica (Verde/Amarilla/Naranja/Roja) y *Reportes Especiales de Actividad Volcánica*
   (REAV).
2. **URL / endpoint**
   - Página humana: `https://www.sernageomin.cl/alertas-volcanicas/` y
     `https://www.sernageomin.cl/rnvv/` (**VERIFICADO**, HTTP 200)
   - Los REAV son PDF colgados en la biblioteca de medios de WordPress, y **son
     descubribles programáticamente** vía la API REST de WordPress:
     `https://www.sernageomin.cl/wp-json/wp/v2/media?search=REAV` (**VERIFICADO**: devolvió
     JSON con el REAV de Nevados de Chillán del 15-jun-2026 y su URL de PDF)
   - `https://rnvv.sernageomin.cl/` **no resuelve** (**VERIFICADO**, sin conexión).
3. **Qué publica exactamente** — Nivel de alerta por volcán (escala de 4 colores) y
   descripción del estado de actividad. Los REAV traen hora, volcán, tipo de señal
   (sismicidad, columna eruptiva en metros, etc.) en prosa dentro de un PDF.
4. **Familia** — AMENAZA.
5. **Formato** — **HTML** (página) y **PDF** (REAV). El índice de PDFs, vía **API JSON** de
   WordPress. No hay servicio geográfico del nivel de alerta vigente.
6. **Acceso** — anónimo. **VERIFICADO** para la página y para `wp-json`. **No verificado**
   ningún canal estructurado del nivel de alerta, porque **no existe hoy**: la propia página
   dice textualmente *«la visualización en tiempo real de esta información no se encuentra
   disponible en el sitio web»*.
7. **Granularidad territorial** — **punto** (el volcán) más un radio de exclusión declarado
   en el texto del reporte. La página lista además las comunas afectadas en prosa.
8. **Frecuencia** — el monitoreo es 24/7; los REAV se publican por evento.
9. **Historia** — los PDF quedan en la biblioteca de medios de WordPress, fechados y
   consultables vía `wp-json`. No hay serie de niveles de alerta.
10. **Condiciones de uso** — `robots.txt` de SERNAGEOMIN permite todo salvo `/wp-admin/`
    (**VERIFICADO**). La API `wp-json` es pública por defecto en WordPress. Ritmo bajo y
    atribución obligatoria.
11. **Para qué sirve en el modelo** — Amenaza volcánica para el `FEN`. Con el volcán como
    punto y su radio de exclusión se puede seleccionar activos dentro del radio.
12. **Utilidad — ★★.** Cuando hay alerta, cambia radicalmente la prioridad de los activos
    del entorno. Pierde una estrella porque **hoy el nivel de alerta no es legible por
    máquina**: hay que leer un PDF, y eso no es base para un instrumento operativo.
13. **Riesgos y límites**
    - Extraer el nivel de alerta de un PDF es frágil y propenso a error. **No recomiendo
      automatizarlo.** Recomiendo pedir a SERNAGEOMIN el canal estructurado — el
      Observatorio Nacional de Peligros Geológicos y Mineros (proyecto ITP Corfo lanzado
      este año, con convenio en trámite con SENAPRED) está construyendo justamente eso.
    - La página reconoce que la visualización en tiempo real está caída: hay un hueco
      declarado por el propio organismo. Ese hueco debe declararse en el modelo, no
      rellenarse.

---

## G-7 · Ranking de riesgo volcánico 2023 (87 volcanes) ★★

1. **Organismo y producto** — SERNAGEOMIN / OVDAS. *Ranking de Riesgo Específico de
   Volcanes Activos de Chile 2023*.
2. **URL / endpoint**
   - `https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/services/Ranking_Volcanes_Chile_2023/FeatureServer/0`
   - Tabloide PDF: `https://www.sernageomin.cl/wp-content/uploads/2026/04/Ranking-2023_tabloide_20231012.pdf`
3. **Qué publica exactamente** — 87 volcanes, punto (`LAT`/`LONG`), con el desglose completo
   del puntaje: factores de peligro (`FP_TIPO`, `FP_VEI_MAX`, `FP_RECURR`, `FP_LAHARPR`,
   `FP_UNREST`… → `TOTAL_FP`) y factores de exposición (`FE_Log1015`, `FE_LOG1030`,
   `FE_TURISMO`, `FE_EVACUA`, `FE_AVIACIO`, **`FE_INFRAES`**, **`FE_INFENER`**,
   **`FE_INFTRAN`** → `TOTAL_FE`), más `TOTAL_RIES`, `CATEGORIA` y `DESCRIPCION`.
4. **Familia** — AMENAZA + ESTADO-CONSECUENCIA (trae ya la exposición).
5. **Formato** — ArcGIS Feature Service, JSON / GeoJSON.
6. **Acceso** — anónimo. **VERIFICADO** (87 registros contados, esquema leído).
7. **Granularidad territorial** — punto (volcán), con región.
8. **Frecuencia** — se recalcula cada varios años (versiones 2019, 2023).
9. **Historia** — versiones puntuales; no es serie temporal.
10. **Condiciones de uso** — igual que G-1.
11. **Para qué sirve en el modelo** — Doble uso. Primero, prioriza los 87 volcanes entre sí
    (`CATEGORIA`), lo que ordena a qué alerta hacerle más caso. Segundo, y más interesante:
    **`FE_INFRAES`, `FE_INFENER` y `FE_INFTRAN` son exactamente el mismo cruce que nosotros
    queremos hacer** — infraestructura, energía y transporte como factores de exposición.
    Es un precedente metodológico chileno, del mismo organismo, de que el cruce
    amenaza × infraestructura es legítimo y ya se hace. Vale la pena leer su metodología
    antes de inventar la nuestra.
12. **Utilidad — ★★.** Modula la prioridad de activos cerca de volcanes, pero es estático:
    no cambia de un día para otro.
13. **Riesgos y límites**
    - Es un ranking de riesgo **de 2023**; no refleja el estado de actividad de hoy.
    - Sus factores de exposición usan supuestos propios (población en radios de 10/30 km).
      No mezclarlos con nuestros pesos sin entender su escala, o se suman peras con manzanas.

---

## G-8 · Catastro de Depósitos de Relaves (CDR) 2025 ★★★

1. **Organismo y producto** — SERNAGEOMIN. *Catastro de Depósitos de Relaves de Chile*
   (variable de riesgo 2.9 del glosario oficial).
2. **URL / endpoint**
   - Puntual: `https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/services/CDR_CHILE_PUNTUAL_2025/FeatureServer/0`
   - Areal (polígonos del depósito):
     `https://services1.arcgis.com/OyjvVdFTl5hfSdX3/arcgis/rest/services/CDR_CHILE_AREAL_2025/FeatureServer`
   - Planilla oficial: ítem ArcGIS `CATASTRO_RELAVES_CHILE_OCT2025` (Excel)
   - Visor: dashboard «PLATAFORMA PÚBLICA DE RELAVES» del mismo portal
3. **Qué publica exactamente** — **839 depósitos**, con `NOMBRE_EMPRESA_O_PRODUCTOR_MINE`,
   `NOMBRE_FAENA`, `NOMBRE_INSTALACION`, `TIPO_DEPOSITO`, `RECURSO`, `TIPO_MINERIA`,
   `CANTIDAD_MUROS__CONTENCION_`, `METODO_CONSTRUCTIVO_MURO`, **`ESTADO_INSTALACION`**
   (activo/inactivo/abandonado), `VOL_AUTORIZADO` y `VOL_ACTUAL`, `TONELAJE_AUTORIZADO` y
   `TONELAJE_ACTUAL`, coordenadas (`LATITUD`/`LONGITUD` y UTM con `DATUM`/`HUSO`),
   `CUT_REG`/`CUT_PROV`/`CUT_COM` + nombres, y las resoluciones que lo aprueban con fecha.
4. **Familia** — **ACTIVO y AMENAZA a la vez.** Un relave es infraestructura *y* es la cosa
   que puede colapsar encima de otra infraestructura. Es el caso más claro de doble rol de
   todo el catastro.
5. **Formato** — ArcGIS Feature Service (JSON/GeoJSON) + Excel.
6. **Acceso** — anónimo. **VERIFICADO** (839 registros contados, esquema leído).
7. **Granularidad territorial** — **punto** (y polígono en la capa areal), con comuna y CUT.
8. **Frecuencia** — actualización anual del catastro (versiones 2023 y oct-2025 presentes).
9. **Historia** — hay al menos dos cortes (`CATASTRO_RELAVES_2023` y
   `CATASTRO_RELAVES_CHILE_OCT2025`), o sea sí se puede comparar entre años.
10. **Condiciones de uso** — igual que G-1.
11. **Para qué sirve en el modelo** — Entra por los dos lados. Como **amenaza**: un relave
    aguas arriba de una comuna o de una subestación es una fuente de peligro con volumen
    conocido en m³, cruzable con la minuta de remoción en masa (un relave dentro de zona
    `Alta` es prioridad de otro orden). Como **activo**: 839 instalaciones con comuna, que
    amplían el inventario de la MICR en un rubro que hoy no está cubierto.
12. **Utilidad — ★★★.** Cambia la prioridad de activos de forma directa y verificable, y
    aporta el único dato de *magnitud física* (volumen y tonelaje) de todo mi dominio.
13. **Riesgos y límites**
    - `VOL_ACTUAL` y `TONELAJE_ACTUAL` son autodeclarados por el titular; la fecha de
      declaración importa y no siempre está.
    - **No confundir los 839 relaves con los 835 elementos de la MICR.** Números parecidos,
      universos distintos. Coincidencia pura, pero es exactamente el tipo de coincidencia
      que produce un error difícil de encontrar después.
    - Coordenadas en distintos husos UTM (`HUSO` 18 y 19) y distintos datums (`DATUM`).
      Reproyectar con cuidado; un huso mal aplicado desplaza un relave cientos de km.
    - El estado «abandonado» suele ser el más peligroso y el peor documentado.

---

## G-9 · CSN — sismos, canal oficial ★★★

1. **Organismo y producto** — Centro Sismológico Nacional, Universidad de Chile. Catálogo de
   sismicidad e informes de sismo.
2. **URL / endpoint**
   - Portada con los últimos eventos: `https://www.sismologia.cl/` (**VERIFICADO**, HTTP 200)
   - **Informe por evento:** `https://www.sismologia.cl/sismicidad/informes/AAAA/MM/{id}.html`
     (**VERIFICADO**: leí el evento `379393`, 15-ago-2026, 47 km al SO de Mina Collahuasi)
   - **Catálogo diario, con archivo:**
     `https://www.sismologia.cl/sismicidad/catalogo/AAAA/MM/AAAAMMDD.html`
     (**VERIFICADO**: `.../2015/09/20150916.html` devolvió la secuencia completa del
     terremoto de Illapel, con el 8.4 Mw. También responden 2000-01-01, 2010-03-01,
     2012-01-01 y 2014-04-01)
   - `https://www.sismologia.cl/links/eventos.html` → **HTTP 403** (**VERIFICADO**, no usar)
   - `https://api.sismologia.cl/` → **no resuelve** (**VERIFICADO**). No existe API oficial.
3. **Qué publica exactamente** — Por evento: referencia geográfica en texto, hora local y
   hora UTC, latitud, longitud, profundidad en km, magnitud con **tipo declarado**
   (`Ml` local, `Mw` momento-Brüne, `Mww` fase W, `MLv`), y observaciones. El catálogo
   diario trae la misma tabla para todos los eventos del día UTC.
4. **Familia** — AMENAZA.
5. **Formato** — **HTML** estático, tabular y regular. Hay que parsearlo; no hay JSON.
6. **Acceso** — anónimo. **VERIFICADO**. El sitio está servido desde almacenamiento estático
   (S3): no tiene `robots.txt` (devuelve 403 con `AccessDenied`, o sea **el archivo no
   existe**, no que esté prohibido).
7. **Granularidad territorial** — **punto** (hipocentro: lat, lon, profundidad).
8. **Frecuencia** — minutos tras el evento; el catálogo diario se consolida por día UTC.
9. **Historia** — **al menos desde 2000-01-01, VERIFICADO**, por día, con URL predecible.
   Es un archivo excelente. Para catálogo revisado y formas de onda existe además
   `https://evtdb.csn.uchile.cl/` (no probado — **SUPUESTO**).
10. **Condiciones de uso** — **Acá está el problema, y es serio.** La página oficial
    `https://www.sismologia.cl/accesos/uso-de-datos.html` (**VERIFICADO**, leída completa)
    dice: *«Se autoriza el uso, reproducción, divulgación o distribución de estos datos y
    material gráfico **con fines académicos y de divulgación** siempre que se cite la fuente
    […]. El uso de los datos y material gráfico **para cualquier otro fin deberá contar con
    la aprobación expresa, y por escrito, del CSN**, de lo contrario constituye una violación
    de la propiedad intelectual y está estrictamente prohibida»*.
    Un instrumento que produce insumo para decisión pública **no es claramente «académico o
    de divulgación»**. Mi lectura: **hay que pedir autorización escrita al CSN antes de
    operar**. No recomiendo automatizar la descarga sin ese permiso. Mientras tanto se puede
    usar para prototipo interno citando la fuente, y conviene tramitar el permiso ya, porque
    tarda.
11. **Para qué sirve en el modelo** — Amenaza sísmica dinámica: un sismo relevante cerca de
    un activo dispara revisión de prioridad. Y con el archivo desde 2000 se puede construir
    la exposición sísmica histórica por comuna, que es amenaza **de base**.
12. **Utilidad — ★★★.** Sismo es la amenaza que más define la infraestructura crítica en
    Chile, y el dato es de calidad y con historia larga. La estrella se sostiene **a
    condición** de resolver el permiso.
13. **Riesgos y límites**
    - **Restricción de licencia sin resolver.** Es el límite principal, y es legal, no
      técnico.
    - Sin API: parseo de HTML, que se rompe si cambian la plantilla. Adaptador aislado y con
      degradación a «sin dato».
    - Magnitudes preliminares que se revisan después: el mismo evento puede cambiar de
      magnitud horas más tarde. Guardar la versión descargada con su hora, y volver a
      consultar el catálogo del día siguiente para el valor consolidado.
    - Ritmo: consultar como mucho cada pocos minutos, y el archivo histórico una sola vez.

---

## G-10 · CSN vía tercero (api.gael.cloud) — ★ y con advertencia

1. **Organismo y producto** — **No es un organismo.** `gael.cloud` es un tercero privado que
   republica los sismos del CSN.
2. **URL / endpoint** — `https://api.gael.cloud/general/public/sismos`
3. **Qué publica** — JSON con `Fecha`, `Profundidad`, `Magnitud`, `RefGeografica`.
4. **Familia** — AMENAZA.
5. **Formato** — API JSON.
6. **Acceso** — anónimo. **VERIFICADO** (HTTP 200, `application/json`).
7. **Granularidad territorial** — punto.
8. **Frecuencia** — sigue al CSN.
9. **Historia** — sólo los últimos eventos. **Sin archivo.**
10. **Condiciones de uso** — no revisé términos propios del proveedor. Y **republicar el
    dato del CSN no le transfiere a nadie el permiso del CSN**: la restricción de uso de
    G-9 sigue aplicando al dato original.
11. **Para qué sirve** — prototipar rápido. Nada más.
12. **Utilidad — ★.** No cambia ninguna prioridad que no cambiara el canal oficial, y agrega
    un punto de falla y una duda legal.
13. **Riesgos y límites** — Un intermediario privado sin compromiso de servicio, alimentando
    una decisión pública. **Confirmo la advertencia ya escrita en
    `FUENTES_DE_DATOS_NACIONALES.md`: sirve para prototipo, no para producción.**

---

## G-11 · SHOA — tsunami · **NO ACCESIBLE Y NO PERMITIDO**

1. **Organismo y producto** — Servicio Hidrográfico y Oceanográfico de la Armada. Cartas de
   Inundación por Tsunami (CITSU) y el Sistema Nacional de Alarma de Maremotos (SNAM).
2. **URL / endpoint** — `https://www.shoa.cl/` · `https://www.shoa.cl/php/citsu.php` ·
   `https://snam.shoa.cl/`
3. **Qué publica exactamente** — CITSU: cota máxima de inundación esperada para ~75
   localidades costeras de Arica a Puerto Williams, en PDF, KMZ y visor web (según su propia
   difusión — **SUPUESTO**, no pude verificarlo directamente).
4. **Familia** — AMENAZA.
5. **Formato** — PDF · KMZ · visor web. Sin API.
6. **Acceso** — **NO PÚBLICO para acceso programático. VERIFICADO en tres intentos:**
   `https://www.shoa.cl/` devuelve un cuerpo `403 Forbidden`;
   `https://www.shoa.cl/php/citsu.php` devuelve **HTTP 403** incluso con navegador simulado;
   `https://snam.shoa.cl/` no resuelve.
7. **Granularidad territorial** — zona costera / localidad.
8. **Frecuencia** — las CITSU se actualizan por localidad, con años de por medio.
9. **Historia** — la serie de CITSU existe desde 1997, pero no me es accesible.
10. **Condiciones de uso** — **`https://www.shoa.cl/robots.txt` dice `User-agent: *` →
    `Disallow: /` (VERIFICADO).** Es una prohibición explícita de rastreo automatizado del
    sitio completo. **No propongo ninguna forma de sortearla.** El camino correcto es
    solicitar los datos formalmente al SHOA, o usar la vía G-12.
11. **Para qué sirve en el modelo** — Tsunami es amenaza de primer orden para la
    infraestructura costera chilena (puertos, plantas, subestaciones de borde). Es un hueco
    grande.
12. **Utilidad — ★ (hoy).** El dato cambiaría mucho la prioridad de activos costeros, pero
    **hoy no lo tengo**, y una fuente inaccesible no prioriza nada. Sube a ★★★ el día que
    haya convenio o se use la vía G-12.
13. **Riesgos y límites** — El 403 más el `Disallow: /` no son un obstáculo técnico a
    superar: son una **decisión del organismo**. Insistir por vías automatizadas sería
    incorrecto además de inútil. Registro esto como **pendiente institucional**, no técnico:
    hay que pedirlo por oficio.

---

## G-12 · IDE Chile / Geoportal — vía alterna para CITSU ★★ (a verificar)

1. **Organismo y producto** — Infraestructura de Datos Geoespaciales de Chile (IDE Chile,
   Ministerio de Bienes Nacionales). Republica capas de otros organismos, **incluidas las
   cartas de inundación por tsunami del SHOA**.
2. **URL / endpoint** — `https://www.ide.cl/` (**VERIFICADO**, HTTP 200) ·
   `https://www.geoportal.cl/` (**VERIFICADO**, HTTP 200). El endpoint concreto de las capas
   CITSU **no lo verifiqué**: es **SUPUESTO**, basado en las noticias del propio IDE Chile
   que anuncian la incorporación de información de tsunami al Centro de Descargas.
3. **Qué publica exactamente** — capas vectoriales de inundación por tsunami por comuna
   (**SUPUESTO**).
4. **Familia** — AMENAZA.
5. **Formato** — presumiblemente **WMS/WFS y descarga vectorial** (**SUPUESTO**).
6. **Acceso** — anónimo (**SUPUESTO** para las capas; verificado sólo que los portales
   responden).
7. **Granularidad territorial** — comuna / localidad costera.
8. **Frecuencia** — según publique el SHOA.
9. **Historia** — hay comparaciones publicadas entre ediciones 2012 y 2022 de las CITSU, o
   sea existe más de una versión (**SUPUESTO**).
10. **Condiciones de uso** — **`https://geoportal.cl/robots.txt` → `User-agent: *` con
    `Disallow:` vacío = todo permitido (VERIFICADO).**
    **`https://www.ide.cl/robots.txt` → `User-agent: *` → `Allow: /`, pero bloquea
    explícitamente por nombre a los rastreadores de IA, incluido `ClaudeBot`, y también
    `GPTBot`, `CCBot`, `Google-Extended`, `Bytespider`, `Applebot-Extended` y
    `meta-externalagent` (VERIFICADO).** Traducción práctica: el sitio permite un cliente de
    datos que se identifique como tal, y prohíbe el rastreo para entrenamiento de IA.
    Nuestro adaptador debe identificarse con un `User-Agent` propio del proyecto y de
    contacto, ir despacio, y **no** hacerse pasar por navegador ni por rastreador de IA. Si
    hay la menor duda, pedir permiso a IDE Chile: es un organismo de datos abiertos y suele
    concederlo.
11. **Para qué sirve en el modelo** — Es **la única vía practicable hoy para meter tsunami
    al modelo**, dado el bloqueo del SHOA (G-11).
12. **Utilidad — ★★ provisional.** Subirá a ★★★ si se confirma que las capas CITSU están
    ahí y son descargables. Es lo primero que haría en la próxima ronda.
13. **Riesgos y límites**
    - Todo lo específico de CITSU en esta ficha es **SUPUESTO**. No construir nada encima
      hasta verificarlo.
    - Es una **copia**: puede ir por detrás de la versión del SHOA. La autoridad de la carta
      sigue siendo del SHOA, y así hay que citarla.

---

## Resumen operativo del dominio

| Ficha | Fuente | Formato | Acceso | Historia | Utilidad |
|---|---|---|---|---|---|
| G-1 | Minuta Técnica vigente (remoción en masa) | GeoJSON | anónimo **VERIF.** | **ninguna** | ★★★ |
| G-2 | ReTeRM, eventos reales | GeoJSON | anónimo **VERIF.** | 1996-2026 | ★★★ |
| G-3 | Minuta + acople DMC (congelado) | GeoJSON | anónimo **VERIF.** | 2021-2023 | ★★ |
| G-4 | Zonas morfoclimáticas + comunas | GeoJSON | anónimo **VERIF.** | n/a | ★★★ |
| G-5 | Mapas de peligro por localidad | GeoJSON | anónimo **VERIF.** | por año | ★★ |
| G-6 | RNVV/OVDAS alertas volcánicas | PDF + wp-json | anónimo **VERIF.** | PDFs fechados | ★★ |
| G-7 | Ranking volcánico 2023 | GeoJSON | anónimo **VERIF.** | 2019, 2023 | ★★ |
| G-8 | Catastro de relaves 2025 | GeoJSON + Excel | anónimo **VERIF.** | 2023, 2025 | ★★★ |
| G-9 | CSN sismos, canal oficial | HTML | anónimo **VERIF.**, licencia restrictiva | desde 2000 | ★★★ |
| G-10 | gael.cloud (tercero) | JSON | anónimo **VERIF.** | ninguna | ★ |
| G-11 | SHOA tsunami | — | **bloqueado VERIF.** | — | ★ |
| G-12 | IDE Chile / Geoportal (CITSU) | WMS/WFS **SUP.** | permitido **VERIF.** | **SUP.** | ★★ |

### Lo urgente, en orden

1. **Empezar el snapshot diario de la minuta (G-1) mañana mismo.** Es lo único de esta lista
   que se pierde para siempre si no se hace hoy. Todo lo demás se puede recuperar después.
2. **Pedir por escrito al CSN** la autorización de uso para fin distinto al académico (G-9).
   Es trámite y demora; empezarlo ya.
3. **Pedir por oficio al SHOA** las CITSU (G-11), y en paralelo **verificar la vía IDE Chile**
   (G-12). No hay atajo automatizado y no lo va a haber.
4. **Consultar a SERNAGEOMIN cuál es el canal oficial vigente** de la minuta: conviven el
   servicio nuevo (`minutasATG_Flash`, vivo) y el viejo (`Minuta_Técnica`, congelado en
   2023). También vale preguntar por el nivel de alerta volcánica en formato estructurado:
   el Observatorio Nacional de Peligros Geológicos y Mineros lo está construyendo y hay
   convenio con SENAPRED en trámite — es el momento de pedirlo.
5. **Corregir en los documentos del proyecto** que la escala de la minuta tiene **cuatro**
   niveles, no tres. Falta `Muy Alta`.
