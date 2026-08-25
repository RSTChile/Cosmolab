# Ficha de fuentes — ESTADO REAL DE LOS SERVICIOS BÁSICOS
## Electricidad · Agua · Telecomunicaciones

**Proyecto:** Infraestructura Crítica × Clima (RMD 2.0) · **Fecha de verificación:** 15-ago-2026
**Alcance:** la capa de ESTADO-CONSECUENCIA. No dice «puede fallar»: dice «falló, acá, ahora».

---

## Resumen ejecutivo (leer esto primero)

Tres hallazgos, en orden de importancia:

**1. SÍ HAY HISTORIA. Mucha más de la que esperábamos.** La SEC publica, sin registro
y sin clave, la cantidad de **clientes sin suministro eléctrico por COMUNA y por HORA**,
de **todo el país y de todas las empresas juntas**. Y no sólo de hoy: probamos fechas
sueltas y respondió para el **17-jul-2026** (421.263 clientes sin luz a las 12:00, en
312 comunas), para el **01-ago-2024** (446.892) y para el **15-ene-2020** (18.188).
O sea: hay al menos **seis años y medio de historia horaria por comuna**. Esa es,
literalmente, la capa contra la cual el modelo se puede equivocar y darse cuenta.

**2. SÍ, el Coordinador tiene el catastro real de subestaciones.** Son **1.269
subestaciones** de **318 propietarios**, en JSON abierto y sin autenticación. Nosotros
teníamos 39: es una muestra del 3%. Pero hay un pero grande: **ese listado no trae
coordenadas**. Trae nombre, código, propietario y a quién le corresponde el control —
no dónde está. Georreferenciarlas es un trabajo aparte y pendiente.

**3. Los mapas de las empresas se ven, pero no son libres.** El mapa de CGE resultó
tener detrás un archivo de mapa (KMZ) descargable sin clave — pero los **términos de uso
de CGE prohíben expresamente reproducir su contenido sin autorización escrita**. Mismo
patrón, con matices, en las demás distribuidoras. **La salida limpia es no tocarlos:
la SEC publica lo mismo agregado, es organismo público, y con eso alcanza.**

Y una advertencia que cruza todo el dominio: varios de estos mapas permiten buscar
«por número de cliente». **Eso no se toca.** El proyecto trabaja por comuna y por activo.

---

# BLOQUE A — ELECTRICIDAD

---

## FICHA A1 ★★★ — SEC · Interrupciones en Línea (LA CAPA DE VALIDACIÓN)

**1. Organismo/empresa y producto**
Superintendencia de Electricidad y Combustibles (SEC), organismo público fiscalizador.
Producto: «Interrupciones en Línea» — consolidado nacional de clientes sin suministro
eléctrico, con todas las distribuidoras juntas.

**2. URL exacta y endpoint**
- Página humana: `https://www.sec.cl/interrupciones-en-linea/`
- Aplicación: `https://apps.sec.cl/INTONLINEv1/index.aspx`
- **Endpoints de datos** (POST, `Content-Type: application/json`), relativos a
  `https://apps.sec.cl/INTONLINEv1/`:

| Endpoint | Cuerpo | Devuelve |
|---|---|---|
| `ClientesAfectados/Get` | `{}` | Serie **horaria nacional** de los últimos ~7 días: `[{anho,mes,dia,hora,clientes_afectados}]` |
| `ClientesAfectados/GetPorFecha` | `{"anho":2026,"mes":7,"dia":17,"hora":12}` | **Por comuna** en esa hora exacta: `[{NOMBRE_REGION,NOMBRE_COMUNA,CLIENTES_AFECTADOS}]` |
| `ClientesAfectados/GetClientesNacional` | `{}` | Total nacional de clientes (denominador) — no probado |
| `ClientesAfectados/GetClientesRegional` | (fecha) | Total regional — no probado |
| `ClientesAfectados/GetHoraServer` | `{}` | Hora del servidor (crédito de los gráficos) — no probado |

**3. Qué publica**
Cuántos clientes están sin luz, hora por hora, comuna por comuna, sumando **todas** las
empresas distribuidoras del país. Es el termómetro nacional del apagón.

**4. Familia:** ESTADO-CONSECUENCIA (la más valiosa del proyecto).

**5. Formato:** JSON limpio (`application/json; charset=utf-8`). Detrás hay una base
ClickHouse — o sea, está pensado para consultas, no es un PDF disfrazado.

**6. Acceso: anónimo, sin registro — VERIFICADO**
Probado el 15-ago-2026 con dos peticiones POST reales:
- `Get` → 12.777 bytes, serie horaria del 08-ago al 15-ago-2026.
- `GetPorFecha` con `{2026,8,10,6}` → 140 filas comuna/clientes.

**7. Granularidad territorial: COMUNA.** Con nombre de región y de comuna en texto.
Es exactamente la unidad en que SENAPRED declara alerta (ver `INTEGRACION_SENAPRED.md`,
sección 4): esta fuente **no** sufre el problema de las dos geografías.

**8. Frecuencia de actualización:** horaria (la serie viene con campo `hora`). La página
de la SEC declara refresco cada 10 minutos para los mapas embebidos.

**9. HISTORIA DISPONIBLE — VERIFICADO, y es el hallazgo del dominio**
El gráfico visible muestra 7 días, pero **el endpoint acepta cualquier fecha**:

| Fecha probada | Filas (comunas) | Total clientes sin luz |
|---|---|---|
| 17-jul-2026 12:00 (temporal) | 312 | **421.263** |
| 01-ago-2024 12:00 | 243 | **446.892** |
| 15-ene-2020 12:00 | 211 | 18.188 |

Es decir: **al menos desde enero de 2020 hay dato horario por comuna.** Para validar el
modelo esto es oro: podés tomar un temporal del pasado, correr la matriz con los datos
de amenaza de ese día, y **comparar lo que el modelo predijo contra lo que realmente
pasó**, comuna por comuna. Sin esto, el modelo sería elegante e infalsable.

**10. Condiciones de uso**
- `https://www.sec.cl/robots.txt` → `User-agent: * / Allow: /` (sólo bloquea `/wp-admin/`).
  **VERIFICADO.** No hay `robots.txt` que prohíba el acceso automatizado.
- «Normas de uso» (`https://www.sec.cl/area-ciudadana/normas-de-uso/`): **no** menciona
  propiedad intelectual, ni reproducción, ni scraping. Sí advierte que la SEC
  «utiliza programas computacionales que monitorean el tráfico de red e identifican a
  los usuarios no autorizados», y que la información es **informativa**, sin compromiso
  contractual. **VERIFICADO** por lectura del texto.
- Es órgano público: rige Ley 20.285 de Transparencia. La información es pública.
- **Regla propia que propongo:** consultar sólo **ventanas de evento** (los días de
  temporal), no barrer el calendario. Una hora = una petición; un año hora por hora
  serían 8.760 peticiones, y eso ya es descarga masiva. Para reconstruir historia
  completa **lo correcto es pedirle a la SEC un extracto por Ley de Transparencia**,
  no martillarles el servidor. El servidor además es lento (respuestas de 30-100 s).

**11. Para qué sirve en el modelo**
- **Validación (uso principal):** contrastar la prioridad que la matriz asignó contra
  el corte que efectivamente ocurrió. Es el único juez externo que tenemos.
- **Calibración de `C_clim`:** si el modelo marca peligro alto donde nunca se cortó la
  luz en seis años, el que está mal es el modelo.
- **Consecuencia real:** convierte «subestación en zona de peligro» en «tantos miles de
  clientes sin luz», que es el lenguaje en que decide SENAPRED.
- **Detección de fragilidad crónica:** comunas que aparecen en la lista con cada
  temporal, aunque el evento sea chico. Eso ya es una prioridad, sin necesidad de modelo.

**12. Utilidad ★★★** — ¿cambia la prioridad de un activo? **Sí, y de la manera más
fuerte: es la única fuente que puede sacar un activo de la lista de prioridades por
haber demostrado que aguanta, o subirlo porque falla siempre.**

**13. Riesgos y límites**
- **No dice POR QUÉ se cortó.** Un corte por temporal y uno por un choque contra un
  poste se ven igual. Hay que cruzarlo con la amenaza del día (DMC/SERNAGEOMIN) y
  aceptar que ese cruce es una inferencia nuestra, no un dato de la SEC.
- **No dice QUÉ activo falló.** Da la consecuencia (clientes), no la causa (subestación,
  alimentador, línea). El puente activo→consecuencia hay que construirlo, y es el
  eslabón más débil de toda la cadena del proyecto.
- **Cuenta clientes, no personas.** Un «cliente» es un empalme: una casa, un hospital y
  una minera cuentan 1 cada uno. Para criticidad hay que ponderar por tipo de cliente,
  dato que esta fuente no trae.
- Servidor lento e intermitente; hay que programar reintentos y aceptar huecos.
- El endpoint no está documentado como API pública: **puede cambiar sin aviso**. El
  adaptador tiene que degradar a «sin dato» y decirlo, nunca inventar.
- La granularidad temporal es la **hora en punto**: un corte de 20 minutos entre horas
  puede no aparecer nunca.

---

## FICHA A2 ★★ — SEC · Anuario e informes de continuidad (SAIDI / SAIFI)

**1. Organismo y producto**
SEC · «Anuario SEC — Resumen Anual de la Industria Energética» e «Informe SEC — Resumen
Mensual». Contienen los indicadores oficiales de continuidad: **SAIDI** (cuántas horas
al año está sin luz un cliente promedio) y **SAIFI** (cuántas veces al año se le corta).

**2. URL exacta**
- Ejemplo verificado: `https://www.sec.cl/sitio-web/wp-content/uploads/2024/05/Anuario-SEC-2023.pdf`
  (2,6 MB, descargado y leído el 15-ago-2026).
- Serie mensual, mismo patrón de ruta: `.../uploads/AAAA/MM/Informe-SEC-<Mes><Año>.pdf`.
- Cuentas públicas regionales: `.../uploads/AAAA/MM/Cuenta-Publica-AAAA_region-<Región>.pdf`.

**3. Qué publica**
SAIDI informado por las empresas y **SAIDI Recalificado SEC** — la SEC revisa lo que las
empresas declararon como «fuerza mayor» y lo reclasifica. Distingue tres causas:
**Interna** (falla de la distribuidora), **Externa** (falla de generación/transmisión) y
**Fuerza Mayor** (hechos irresistibles e impredecibles, «como terremoto»).

Esa distinción nos importa muchísimo: **«Fuerza Mayor» es, casi textualmente, nuestra
categoría de amenaza climática.** Y el hecho de que la SEC *recalifique* significa que
hay un juicio experto, año a año, sobre cuánto de un apagón es clima y cuánto es
desidia de la empresa. Es una segunda opinión gratis sobre la pregunta central del
proyecto.

**4. Familia:** ESTADO-CONSECUENCIA (agregado anual).

**5. Formato:** PDF **infográfico**. Y acá está el problema: extraje el texto con
`pdftotext` y **los números están dentro de los gráficos, como imagen**. El texto trae
los títulos («SAIDI Anual», «Evolución mensual SAIDI», «SAIDI Comparado 2023»,
«Dato informado Empresas / Dato Recalificado SEC») pero **no las cifras**. VERIFICADO.

**6. Acceso: anónimo, sin registro — VERIFICADO** (descarga directa del PDF).

**7. Granularidad territorial:** por **empresa distribuidora** y por **región** (en las
cuentas públicas regionales). **No llega a comuna.** El área de concesión de una empresa
no coincide con los límites comunales: traducir de una a otra es una aproximación, y hay
que declararla como tal.

**8. Frecuencia:** anual (Anuario) y mensual (Informe SEC). Consolidación al 31-dic.

**9. Historia disponible:** **sí, larga** — hay informes al menos desde 2018 (verificado
por búsqueda: ediciones 2018, 2019, 2020, 2021, 2023). Pero es historia **agregada
anual**, no eventos. Sirve para tendencia, no para validar un temporal concreto.

**10. Condiciones de uso:** mismas que A1 (`robots.txt` permisivo, organismo público,
Ley 20.285). Los PDF están publicados abiertamente para difusión.

**11. Para qué sirve en el modelo**
- **Línea base de fragilidad por empresa/región:** cuánto se corta la luz «normalmente»
  en cada territorio. Un activo en una zona con SAIDI alto ya arranca con desventaja.
- **Contraste conceptual:** la proporción Interna/Externa/Fuerza Mayor da una idea de
  cuánto del riesgo eléctrico de una zona es realmente climático.

**12. Utilidad ★★** — cambia la prioridad, pero de forma gruesa: mueve territorios
enteros, no activos individuales. Y para extraer los números hay que **transcribirlos a
mano de las infografías**, lo que limita mucho la escala.

**13. Riesgos y límites**
- Los números no son legibles por máquina. Automatizar esto es OCR sobre gráficos:
  frágil y propenso a error silencioso. **No recomiendo automatizarlo.**
- La granularidad por empresa obliga a un mapeo empresa→comunas que no tenemos.
- **Pendiente que vale la pena:** preguntar a la SEC si entrega los datos base del
  SAIDI/SAIFI en planilla (por Transparencia). Si los entrega, esta ficha pasa a ★★★.

---

## FICHA A3 ★★★ — Coordinador Eléctrico Nacional · Infotécnica (EL CATASTRO REAL)

**1. Organismo y producto**
Coordinador Eléctrico Nacional (CEN), el operador del sistema. Producto: **Infotécnica**,
el registro de todas las instalaciones del Sistema Eléctrico Nacional.

**2. URL exacta y endpoint**
- Portal humano: `https://infotecnica.coordinador.cl/instalaciones/subestaciones`
- **API abierta:** `https://api-infotecnica.coordinador.cl/v1/subestaciones/`
- Detalle individual: `https://api-infotecnica.coordinador.cl/v1/subestaciones/{id}/`
- Buscador de categorías: `https://api-infotecnica.coordinador.cl/v1/search/categories/`
- Otros tipos de instalación bajo el mismo patrón `/v1/{tipo}/`: `centrales`, `lineas`,
  `empresas`, `transformadores`, `interruptores`, `barras`, `secciones-tramos`,
  `unidades-generadoras`, entre otros (~22 tipos; sólo verifiqué `subestaciones`).

**3. Qué publica — VERIFICADO con dato en mano**
`GET /v1/subestaciones/` devolvió **507 KB de JSON con 1.269 subestaciones** de **318
propietarios distintos**. Campos por registro:

```
id · id_propietario · propietario_nombre · numero · codigo · nemotecnico ·
nombre · descripcion · flag_equipocom · flag_scada · flag_pararrayos ·
id_coordinado · coordinado_nombre · id_centro_control · centro_control_nombre
```

Ejemplo real: `{"id":199,"codigo":"SE001","nombre":"S/E CENTRAL ALFALFAL",
"propietario_nombre":"AES ANDES S.A.","flag_scada":true,...}`

Los mayores propietarios: CGE Transmisión (224), Transelec (76), Sistema de Transmisión
del Sur (70), Sociedad Transmisora Metropolitana (58), Chilquinta Transmisión (43),
Minera Escondida (41), Engie (38).

**4. Familia: ACTIVO** (es inventario, no estado). Lo incluyo en esta ficha porque
responde una pregunta explícita del encargo y porque **sin el activo bien identificado,
la capa de estado no tiene a qué engancharse**.

**5. Formato:** JSON plano, lista de objetos. Limpísimo.

**6. Acceso: anónimo, sin registro, sin API key — VERIFICADO** (dos peticiones GET,
HTTP 200, `application/json`).

**7. Granularidad territorial: NINGUNA — y este es el problema.**
El listado de subestaciones **no trae coordenadas, ni comuna, ni región**. Busqué el
campo: no existe en `/v1/subestaciones/` ni en el detalle individual. El portal web sí
muestra un mapa y agrupa por región y comuna, y en el código del sitio aparece el campo
`comuna_nombre` — pero asociado a otras tablas (centrales), no al listado de
subestaciones que devuelve la API. **Georreferenciar las 1.269 subestaciones queda como
tarea abierta.**

**8. Frecuencia:** el registro se actualiza cuando cambian las instalaciones (altas,
bajas, cambios de propietario). No es un dato «vivo» de operación.

**9. Historia disponible:** **no, para lo que nos importa.** Es una foto del estado
actual del inventario. No vimos versionado ni fechas de corte en el endpoint.
El CEN sí publica aparte estudios de falla y estadísticas de operación
(`https://www.coordinador.cl/reportes-y-estadisticas/`) — **no verificados**, quedan
como pendiente prometedor: ahí podría estar la historia de fallas de transmisión.

**10. Condiciones de uso**
`https://www.coordinador.cl/robots.txt` — **VERIFICADO**, y es particular: además de
`Allow: /` para todos, declara señales de contenido
`Content-Signal: search=yes, ai-train=no, use=reference`, con reserva expresa de
derechos bajo la directiva europea de copyright, y bloquea explícitamente varios bots
(Amazonbot, Applebot-Extended, Bytespider...). Traducido: **permite consultar y citar
como referencia, prohíbe usarlo para entrenar modelos.** Nuestro uso (consolidar un
inventario y citar la fuente) cae del lado permitido; entrenar cualquier cosa con esto,
no. `api-infotecnica.coordinador.cl` no tiene `robots.txt` propio verificado.

**11. Para qué sirve en el modelo**
- **Multiplica por 32 el universo de activos eléctricos:** de 39 subestaciones a 1.269.
- **Identifica al dueño de cada una**, que es exactamente lo que necesita SENAPRED para
  saber a quién llamar.
- `flag_scada` dice si la subestación tiene telemando: una sin SCADA se repone con una
  cuadrilla física yendo hasta allá — eso es **más tiempo de reposición**, o sea peor
  consecuencia. Es una variable de resiliencia gratis que ya está en el dato.
- Permite distinguir subestaciones de **transmisión** (las que al caer dejan sin luz a
  una región) de las de **una minera** (que al caer no afectan a nadie más). Hoy no
  hacemos esa distinción y deberíamos.

**12. Utilidad ★★★** — cambia la prioridad de un activo de la forma más básica posible:
**hoy 1.230 subestaciones no tienen prioridad porque no sabemos que existen.**

**13. Riesgos y límites**
- **Sin coordenadas no se puede cruzar con amenaza.** Hasta resolver eso, esta fuente
  aporta el «qué» y el «de quién», no el «dónde». Es el cuello de botella.
- **No cubre Magallanes ni Aysén.** Verifiqué: no aparecen Edelmag ni Edelaysén entre
  los propietarios. Son sistemas medianos aislados, fuera del SEN, y por lo tanto fuera
  del Coordinador. Justo las zonas más expuestas a clima extremo.
- Es el sistema **coordinado**: las redes de **distribución** (los cables que llegan a
  las casas) no están acá. Un temporal corta distribución mucho más seguido que
  transmisión. Este catastro describe el esqueleto, no los capilares.
- Los ~22 tipos de instalación no fueron verificados uno por uno; sólo `subestaciones`.

---

## FICHA A4 ★★ — CGE · Mapa de afectación (el ejemplo de Alexis)

**1. Empresa y producto**
CGE (Compañía General de Electricidad), distribuidora privada. Producto: «Mapa de
afectación» — dónde hay corte de luz en su zona de concesión, ahora.

**2. URL exacta y endpoint — ESTO ES LO QUE HAY DETRÁS DEL MAPA**
La página que dio Alexis (`https://sucursalvirtual.cge.cl/mapa-de-afectacion`) es sólo
una cáscara: adentro hay un iframe que carga la aplicación real, y esa aplicación
alimenta el mapa con un archivo de mapa descargable.

- Aplicación real: `https://mapa-afectaciones.grupocge.cl/afectaciones/`
- **Capa de datos: `https://mapa-afectaciones.grupocge.cl/afectaciones/mapa_cge.kmz`**
- Refresco de la capa: `https://mapa-afectaciones.grupocge.cl/afectaciones/key.php?empresa=cge`
  (devuelve un hash, `21a694fe...`, que se usa como `?key=` para forzar recarga —
  es un antimemoria, no una credencial).

**3. Qué publica — VERIFICADO abriendo el archivo**
Descargué el KMZ (23.706 bytes, generado el 15-ago-2026 a las 21:27). Adentro hay un
KML con **48 polígonos de corte**. Cada uno trae:

```
Clientes afectados: 124
Restauración estimada: 16/08/2026 00:30 hrs
etiqueta: "124 / 00:51 h"
geometría: un punto + un polígono del sector afectado
```

**4. Familia:** ESTADO-CONSECUENCIA (tiempo real).

**5. Formato:** KMZ (un KML comprimido — el formato de mapas de Google Earth).
Se abre con cualquier herramienta geográfica.

**6. Acceso: anónimo, sin registro — VERIFICADO** (HTTP 200, `application/vnd.google-earth.kmz`).

**7. Granularidad territorial: POLÍGONO DE SECTOR.** Más fina que comuna: es el área
concreta que quedó a oscuras, con su geometría. **No** trae el nombre de la comuna
(hay que cruzarlo con la capa de límites comunales, hallazgo H-13).

**8. Frecuencia:** continua. El archivo se regenera permanentemente (marca de tiempo del
KML: 21:27 del mismo día de la consulta).

**9. HISTORIA DISPONIBLE: NO. Y es el límite grave de esta fuente.**
El KMZ es una **foto del instante**: cada versión pisa a la anterior. No hay archivo, no
hay endpoint de fechas pasadas, no hay identificador de evento persistente. Para tener
historia habría que guardar copias uno mismo cada X minutos — y eso es justamente lo
que los términos de uso no permiten (ver punto 10).

**10. Condiciones de uso — ACÁ ESTÁ EL PROBLEMA**
- `https://sucursalvirtual.cge.cl/robots.txt` → `User-Agent: * / Disallow:` (o sea,
  **no prohíbe nada**). **VERIFICADO.**
- **PERO** los Términos y Condiciones de CGE (`https://www.cge.cl/terminos-y-condiciones/`)
  dicen que los derechos de propiedad intelectual del sitio son de CGE S.A. y que
  **«se prohíbe la reproducción, total o parcial, en cualquier forma o por cualquier
  medio del contenido de este sitio... sin autorización expresa y por escrito»**, y
  prohíben expresamente al usuario «la cesión, reproducción, recirculación,
  sublicenciamiento, retransmisión, comercialización y/o distribución» de los
  contenidos. **VERIFICADO** por lectura del texto.

**Conclusión, sin vueltas: el dato se ve, pero NO es libre.** `robots.txt` permisivo no
es un permiso: es la ausencia de una traba técnica. Los términos mandan. **No propongo
descargar esto de forma automatizada ni almacenarlo.** Si el proyecto lo quiere usar en
serio, el camino es **pedir autorización por escrito a CGE** (o pedirle a la SEC el
consolidado, que ya lo tiene y es público).

**⚠ PRIVACIDAD:** la misma aplicación expone `buscar_cliente.php?id=<n°>` y
`buscar_reclamo.php?id=<ticket>`, que ubican en el mapa a un cliente o un reclamo
concreto. **No se usan, no se consultan, no se guardan.** El proyecto trabaja agregado.
Queda anotado como límite ético de la fuente, no como capacidad.

**11. Para qué sirve en el modelo**
Si hubiera autorización: el polígono de corte es la **evidencia geométrica** de la
consecuencia — permitiría medir directamente cuánto del área afectada cae dentro de una
zona de peligro declarada. Es lo más cerca que se puede estar de «el modelo acertó».

**12. Utilidad ★★** — el dato cambiaría prioridades, pero **la restricción legal lo baja
de ★★★ a ★★**. Sirve como **prueba de concepto de formato** (nos dice que las
distribuidoras publican polígonos + clientes + hora estimada de reposición, que es
exactamente el esquema que deberíamos pedirle al regulador).

**13. Riesgos y límites**
- Términos de uso restrictivos (ver arriba). Es el límite principal.
- Sin historia: hoy sólo sirve para «ahora», no para validar.
- Sólo la zona de concesión de CGE.
- La app cambia (el iframe, el `key.php`, el nombre del KMZ): frágil ante rediseños.
- El «clientes afectados» de la empresa y el de la SEC **pueden no coincidir**; si algún
  día se usan los dos, hay que declarar cuál manda.

---

## FICHA A5 ★★ — Grupo Saesa (Saesa · Frontel · Edelaysén · Luz Osorno)

**1. Empresas y producto**
Grupo Saesa, cuatro distribuidoras del sur: **Saesa**, **Frontel**, **Edelaysén** y
**Luz Osorno**. Producto: «Mapa de desconexiones».

**2. URL exacta y endpoint**
- Mapa: `https://desconexiones.gruposaesa.cl/mapa?empresa=S`
  (`S`=Saesa, `F`=Frontel, `E`=Edelaysén, `L`=Luz Osorno — los cuatro links están
  enlazados desde la propia página de la SEC).
- **Capa de datos: `https://mfallas.saesa.cl/outage.kml`** (cortes actuales)
- **Cortes programados: `https://mfallas.saesa.cl/cortes_futuros.kml`**
- API de detalle por orden: `https://apim.saesa.cl/inspira/desconexion/prd/v2/public/cortes/orden/{id}`

**3. Qué publica — VERIFICADO abriendo el archivo**
Descargué `outage.kml` (26.275 bytes): **39 placemarks**, cada uno con:

```
name: 41342444                      ← número de orden de la falla
Clientes afectados: 6
Restauración estimada: 16/08/2026 14:30:00
Estado: En curso (confirmado)
Comuna: ANGOL, LOS SAUCES           ← ¡la comuna viene explícita!
geometría: punto + polígono
```

**Esta es la mejor estructura de las tres que verifiqué**: trae comuna en texto, número
de orden (identificador estable) y estado del corte. Y además tiene un archivo de
**cortes programados**, que es información de futuro — permite saber que una zona va a
estar sin luz *antes* de que llegue el temporal.

**4. Familia:** ESTADO-CONSECUENCIA (tiempo real) + un poco de planificación.

**5. Formato:** KML (`application/vnd.google-earth.kml+xml`), sin comprimir. La API de
detalle es JSON.

**6. Acceso: anónimo, sin registro — VERIFICADO** (HTTP 200 sobre `outage.kml`).
La API `apim.saesa.cl` tiene además un endpoint de token
(`/inspira/seguridad/prd/v2/public/token`): **no la probé** — no hace falta, y meterse
con tokens de una empresa privada sin permiso no corresponde.

**7. Granularidad territorial: POLÍGONO + COMUNA nombrada.** La mejor del bloque.

**8. Frecuencia:** continua (mismo motor que CGE — ambos usan el mismo software de
gestión de fallas, se nota en la estructura idéntica del KML).

**9. Historia disponible: NO.** Igual que CGE: el KML es el estado actual y se
sobrescribe. El **número de orden** (`41342444`) sí es un identificador persistente, lo
que sugiere que internamente hay historial — pero no está publicado.

**10. Condiciones de uso**
- `https://desconexiones.gruposaesa.cl/robots.txt` → `User-agent: * / Disallow:`
  (no prohíbe nada). **VERIFICADO.**
- Términos y condiciones específicos de Grupo Saesa: **NO VERIFICADOS**. Dado el
  precedente de CGE, **hay que asumir que son restrictivos hasta comprobar lo
  contrario** y no automatizar descargas sin revisarlos.

**⚠ PRIVACIDAD:** el mismo bundle de la aplicación tiene una función
`busquedaNroCliente` (búsqueda por número de cliente). **No se usa.**

**11. Para qué sirve en el modelo**
Cubre el sur profundo — Araucanía, Los Ríos, Los Lagos, Aysén — que es donde más
temporales hay y donde la red es más larga y más frágil. Si algún día hay convenio,
el campo `Comuna` explícito lo hace **integrable sin ningún trabajo de cruce geográfico**.

**12. Utilidad ★★** — misma lógica que CGE: dato excelente, permiso incierto.
Con autorización sería ★★★.

**13. Riesgos y límites**
- Términos de uso sin verificar (asumir restrictivo).
- Sin historia publicada.
- Cobertura sólo del sur.
- Un placemark puede listar **varias comunas a la vez** («ANGOL, LOS SAUCES»): al
  agregar por comuna hay que decidir cómo repartir los clientes, y esa decisión se
  declara, no se esconde.

---

## FICHA A6 ★ — Enel Distribución · Mapa de emergencia

**1. Empresa y producto**
Enel Distribución Chile (Región Metropolitana, la concesión más poblada del país).
Producto: mapa de emergencia con cortes.

**2. URL exacta y endpoint**
- `https://mapaemergencia.enel.com/mapa.html?mode=comuna` ← **el parámetro `mode=comuna`
  es propio: el mapa se puede ver agregado por comuna**, que es nuestra unidad.
- Archivos internos vistos en el código: `js/TileManager.js`, `js/capas/trafos/...`
  (capa de **transformadores**), y una carga de capas por archivos JSON cuyo directorio
  se arma en tiempo de ejecución.

**3. Qué publica:** cortes de suministro en su zona de concesión, con vista por comuna y
—aparentemente— una capa a nivel de transformador.

**4. Familia:** ESTADO-CONSECUENCIA.

**5. Formato:** HTML + capas JSON servidas desde el propio sitio. **Ruta exacta de los
JSON: NO VERIFICADA** — se construye dinámicamente en el JavaScript y reconstruirla
requeriría más peticiones de las que corresponde hacer sin permiso.

**6. Acceso: anónimo para ver — VERIFICADO** (la página carga, HTTP 200, 29.738 bytes).
**Pero el sitio está detrás de Imperva/Incapsula** (se ve el recurso
`/_Incapsula_Resource?...` inyectado): es un cortafuegos **anti-bot**. Traducido: la
empresa está señalizando activamente que no quiere tráfico automatizado.

**7. Granularidad territorial:** comuna (modo declarado en la URL); posiblemente más
fina a nivel de transformador. **No verificado.**

**8. Frecuencia:** continua (es un mapa de emergencia).

**9. Historia disponible: NO detectada.**

**10. Condiciones de uso**
- `https://mapaemergencia.enel.com/robots.txt` → **404, no existe**. **VERIFICADO.**
- `https://www.enel.cl/robots.txt` → `User-agent: * / Allow: /`. **VERIFICADO.**
- **Pero el WAF anti-bot pesa más que el robots.txt permisivo**: es una manifestación
  técnica de voluntad. Términos de uso de Enel: **no verificados**.
- **Recomendación: no automatizar.** Ni siquiera intentarlo: sortear un anti-bot es
  exactamente lo que no debe hacer un proyecto que aspira a ser insumo de decisión
  pública.

**11. Para qué sirve en el modelo**
Cubre la Región Metropolitana — **el mayor número de clientes del país**. Si hubiera
convenio, sería la fuente de mayor impacto poblacional. Y la capa de transformadores
sería un catastro de activos de distribución que hoy no tenemos por ningún lado.

**12. Utilidad ★** — hoy, en la práctica, **no cambia ninguna prioridad**: no es
utilizable de forma legítima y automatizada. Lo mismo agregado está en la SEC (ficha A1).
Sube a ★★★ sólo con convenio formal.

**13. Riesgos y límites**
WAF anti-bot; endpoint no documentado; sin historia; términos no verificados.
**Recomendación operativa: usar la SEC para la RM, no Enel.**

---

## FICHA A7 ★ — Chilquinta (Valparaíso) · Mapa de interrupciones

**1. Empresa y producto:** Chilquinta Energía, distribuidora de la Región de Valparaíso.
Producto: mapa de interrupciones.

**2. URL exacta**
- Página: `https://www.chilquinta.cl/interrupciones` (redirige desde `/interruptions`)
- Mapa real: `https://mapainterrupciones.chilquinta.cl/mapa` (extraído del código de la
  aplicación de Chilquinta)
- La SEC la embebe como `https://www.chilquinta.cl/interruptions`.

**3. Qué publica:** interrupciones de suministro en su zona. **NO VERIFICADO en detalle.**

**4. Familia:** ESTADO-CONSECUENCIA.

**5. Formato:** desconocido. **NO VERIFICADO.**

**6. Acceso: NO SE PUDO VERIFICAR — 3 intentos fallidos.**
El host `mapainterrupciones.chilquinta.cl` **rechaza la conexión segura** desde este
equipo: error `tlsv1 alert protocol version` con `curl`, y
`WRONG_VERSION_NUMBER` con el otro cliente. Es una configuración de TLS incompatible o
mal configurada del servidor, no un bloqueo dirigido. Queda anotado, como manda el
encargo, y seguí.
La página `www.chilquinta.cl/interrupciones` sí responde (HTTP 200) pero es una
aplicación JavaScript que no entrega contenido sin ejecutar el navegador.

**7. Granularidad territorial:** desconocida. **8. Frecuencia:** presumiblemente continua.
**9. Historia:** desconocida, presumiblemente no.

**10. Condiciones de uso**
`https://www.chilquinta.cl/robots.txt` **no existe como tal**: el sitio devuelve la
aplicación web para cualquier ruta. **VERIFICADO** (respondió HTML de la app, no texto).
Es decir: no hay directivas de robots. Términos: no verificados. Asumir restrictivo.

**11-12. Para qué sirve / Utilidad ★** — misma conclusión que Enel: lo que necesitamos de
Valparaíso ya está, agregado por comuna, en la SEC.

**13. Riesgos y límites:** inaccesible desde nuestro entorno; sin verificación de formato
ni de permisos. **No invertir más tiempo acá.**

---

## FICHA A8 ★ — Edelmag (Magallanes) · Mapa de suministro

**1. Empresa y producto:** Edelmag, distribuidora de Magallanes. «Mapa de Suministro»,
habilitado en agosto de 2025.

**2. URL exacta:** `https://www.edelmag.cl/datos-utiles/mapa-de-suministro/`
**VERIFICADO** (HTTP 200, 1,2 MB de HTML). En el código de la página aparecen
referencias a **KML/KMZ**: mismo patrón que CGE y Saesa.

**3. Qué publica:** cortes por emergencia (rojo) y programados (azul), con **número de
clientes afectados** y **hora estimada de reposición**. Además publica listados escritos
de cortes programados por sector y para la semana. (Descripción de la propia empresa;
**no verifiqué la capa de datos**.)

**4. Familia:** ESTADO-CONSECUENCIA. **5. Formato:** KML/KMZ (indicios en el HTML,
ruta exacta no verificada).

**6. Acceso: la página es anónima — VERIFICADO.** El `robots.txt` de `edelmag.cl`
**devuelve 403 (bloqueado por CloudFront)**: hay un cortafuegos delante. **VERIFICADO.**
Como en Enel: señal de que el tráfico automatizado no es bienvenido.

**7. Granularidad:** sector/polígono, con clientes afectados. **8.** Continua.
**9. Historia: no.** **10.** Términos no verificados; `robots.txt` inaccesible por WAF →
**asumir restrictivo, no automatizar.**

**11. Para qué sirve:** Magallanes es **la única ventana** a esa región, porque —como
verifiqué en la ficha A3— **Edelmag no está en el Coordinador** (sistema aislado). Es
también la región con el clima más duro del país. Vale la pena por eso.

**12. Utilidad ★** hoy (no automatizable, sin historia). **★★★ si se consigue convenio**,
por ser la única fuente de una región que ninguna otra capa cubre.

**13. Riesgos:** WAF; formato no verificado; sistema eléctrico aislado que no calza con
ninguna de nuestras otras fuentes eléctricas.

---

## FICHA A9 ★ — Transmisoras (TEN, Transelec y las demás)

**1. Empresas:** Transmisora Eléctrica del Norte (TEN), Transelec, CGE Transmisión,
Sistema de Transmisión del Sur, Sociedad Transmisora Metropolitana, Chilquinta
Transmisión, etc.

**2. URL:** **no tienen mapa público de cortes, y es lógico**: no tienen clientes finales.
Sus instalaciones están en el catastro del Coordinador (ficha A3): CGE Transmisión con
224 subestaciones, Transelec 76, STS 70, Transmisora Metropolitana 58, Chilquinta
Transmisión 43. **TEN específicamente NO aparece** como propietario en
`/v1/subestaciones/` — puede figurar bajo otra razón social o sólo con líneas.
**VERIFICADO** por consulta al listado descargado.

**3-9.** El estado de la transmisión se ve indirectamente: cuando falla la transmisión,
la SEC lo cuenta como **causa «Externa»** en el SAIDI (ficha A2), y el corte aparece
igual en Interrupciones en Línea (ficha A1). **La operación en detalle** (fallas,
mantenimientos, indisponibilidades) la publica el Coordinador en
`https://www.coordinador.cl/reportes-y-estadisticas/` — **NO VERIFICADO**, pendiente.

**10.** Rige el `robots.txt` del Coordinador (ficha A3): permitido citar, prohibido
entrenar modelos.

**11. Para qué sirve:** la transmisión es donde una sola falla deja sin luz a una región
entera. En términos del proyecto: **pocos activos, consecuencia enorme.** Es el caso
donde la matriz más aporta.

**12. Utilidad ★** por ahora (no hay dato de estado propio verificado).
**Pendiente prioritario:** verificar los reportes de operación del Coordinador. Si tienen
historia de fallas de transmisión con fecha e instalación, esa ficha nace en ★★★, porque
sería la única fuente que conecta **activo específico → falla → consecuencia**, que es
justo el eslabón que hoy nos falta.

**13. Riesgos:** todo por verificar.

---

# BLOQUE B — AGUA

---

## FICHA B1 ★★ — SISS · Continuidad del servicio de agua potable

**1. Organismo y producto**
Superintendencia de Servicios Sanitarios (SISS). Producto: los «Cuadros» del **Informe de
Gestión del Sector Sanitario**, publicados también como datos abiertos.

**2. URL exacta y endpoint**
- Ficha de datos abiertos:
  `https://datos.gob.cl/dataset/cuadro-047-continuidad-del-servicio-de-agua-potable`
- **Archivo:** `https://datos.gob.cl/dataset/e8f21bdb-018d-4cfa-a0ab-13fe7ebd7145/resource/ce93628d-7c53-4989-ba68-891920a9ef97/download/cuadro-47-continuidad-del-servicio-de-agua-potable.xlsx`
  **VERIFICADO** (HTTP 200, tipo Excel).
- Relacionados en el mismo catálogo: CUADRO 51 (continuidad de alcantarillado),
  CUADRO 2 (empresas y participación), CUADRO 3 (clientes por tipo de servicio),
  CUADRO 7 (APR con asesoría de concesionarias), CUADRO 8 y 9 (consumo y dotación).
- Informes completos: `https://www.siss.gob.cl/586/w3-propertyvalue-6415.html`
- Consulta a la API del catálogo: `https://datos.gob.cl/api/3/action/package_search?q=servicios+sanitarios`
  → **49 conjuntos de datos** de la SISS. **VERIFICADO.**

**3. Qué publica**
El **indicador y el ranking de continuidad** del servicio de agua potable: cuánto tiempo
del año el servicio estuvo efectivamente disponible, empresa por empresa.

**4. Familia:** ESTADO-CONSECUENCIA (agregado anual).

**5. Formato:** XLSX (planilla). Legible por máquina, a diferencia de los PDF de la SEC.

**6. Acceso: anónimo, sin registro — VERIFICADO.**

**7. Granularidad territorial: EMPRESA SANITARIA.** Cada empresa tiene un área de
concesión que abarca varias comunas. **No es comuna.** Mismo problema de traducción que
el SAIDI de la SEC.

**8. Frecuencia:** anual.

**9. Historia disponible: sí, pero con una advertencia.** El Informe de Gestión del
Sector Sanitario se publica todos los años (verifiqué referencias a 2019, 2020, 2021).
**Pero la ficha del CUADRO 47 en datos.gob.cl tiene última modificación en junio de 2019**
— o sea, **el dato abierto está desactualizado siete años**. La versión al día está
dentro de los PDF del informe anual de la SISS, no en el catálogo. **VERIFICADO** por
metadato.

**10. Condiciones de uso**
- Licencia declarada del CUADRO 47: **Creative Commons Attribution** — o sea,
  **uso libre citando la fuente**. Es la licencia más permisiva que encontré en todo
  este dominio. **VERIFICADO** por metadato del catálogo.
- `https://www.siss.gob.cl/robots.txt` → sólo bloquea rutas internas
  (`/587/w3-*`, `/586/w3-*`). **VERIFICADO.** Ojo: `/586/` es justamente donde viven
  los informes — o sea, **el sitio de la SISS pide no rastrear automáticamente esa
  sección**. Descargar un archivo puntual que ya conocés no es rastrear; barrer la
  sección, sí. Respetarlo.

**11. Para qué sirve en el modelo**
- **Línea base de fragilidad del agua por territorio**, análoga al SAIDI eléctrico.
- El agua es el servicio que **más rápido convierte un corte en emergencia sanitaria**:
  sin luz se aguantan días; sin agua, horas. Para el módulo de consecuencia esto pesa
  distinto que la electricidad y no deberíamos tratarlos igual.

**12. Utilidad ★★** — cambia prioridades a nivel de territorio/empresa, no de activo.
Y el dato abierto está viejo.

**13. Riesgos y límites**
- **No hay equivalente al «Interrupciones en Línea» de la SEC para el agua.** No
  encontré ninguna fuente pública, en tiempo real y consolidada, de cortes de agua por
  comuna. Es **el hueco más grande de este dominio.** Los cortes se informan empresa por
  empresa (Aguas Andinas, Esval, Essbio...), cada una en su web, sin formato común.
- Granularidad por empresa, no por comuna.
- Dato abierto de 2019.
- **Pendiente:** verificar si la SISS publica los cortes no programados con fecha y
  comuna en alguna parte, o si eso sólo existe puertas adentro. Si existe, es ★★★.

---

## FICHA B2 ★★★ — MOP · Registro de Servicios Sanitarios Rurales (la infraestructura invisible)

**1. Organismo y producto**
Ministerio de Obras Públicas, Dirección de Obras Hidráulicas, Subdirección de Servicios
Sanitarios Rurales. Producto: **Registro de Operadores SSR** (lo que antes se llamaba
APR, Agua Potable Rural).

**2. URL exacta y endpoint**
- Página: `https://ssr.mop.gob.cl/registro-operadores/`
- **Archivo:** `https://ssr.mop.gob.cl/uploads/sites/11/2026/03/REGISTRO-DE-OPERADORES_V_MARZO_2026_PagWeb.xlsx`
  **VERIFICADO**: descargado, 522.154 bytes, Excel 2007+ legítimo, versión **marzo de 2026**.
- Existe además una «Base de Datos» más amplia (diciembre 2025) con todos los comités y
  cooperativas operativos, estén o no registrados. **No verificada.**
- Copia antigua en datos.gob.cl (2021):
  `https://datos.gob.cl/dataset/listado-de-sistemas-de-agua-potable-rural-en-chile`
  (licencia CC Non-Commercial). **La del MOP está mucho más al día — usar esa.**

**3. Qué publica — VERIFICADO abriendo la planilla**
**~2.293 sistemas sanitarios rurales**, con estos campos:

```
MXLOCATION · PROVINCIA · COMUNA · NOMBRE_OFICIAL_SISTEMA · RUT ·
AÑO_PUESTA_EN_MARCHA · N°_ARRANQUES · BENEFICIARIOS_ESTIMADOS ·
CLASIFICACIÓN_ART_106_D50 (MENOR/MEDIANO/MAYOR) · APELACIÓN_SEGMENTO ·
REGISTRO_OPERADORES (código RO-1501-SSR001648)
```
Y una hoja resumen con el conteo por región (las 16 regiones).

**4. Familia: ACTIVO** — pero un activo que hoy **no está en ninguna parte de nuestra
matriz**, y ese es exactamente el punto.

**5. Formato:** XLSX, dos hojas, limpio y con códigos estables por sistema.

**6. Acceso: anónimo, sin registro — VERIFICADO** (descarga directa).

**7. Granularidad territorial: PROVINCIA y COMUNA por sistema.** Sin coordenadas, pero
con comuna nombrada, que es nuestra unidad de trabajo. Hay un campo `MXLOCATION` cuyo
contenido no inspeccioné — **podría ser un identificador de localidad**, y valdría la
pena mirarlo.

**8. Frecuencia:** «se actualiza permanentemente según los procesos de recolección de
datos». La versión publicada es de marzo de 2026 — cinco meses de antigüedad.

**9. Historia disponible:** el registro es una foto actual. **Pero trae
`AÑO_PUESTA_EN_MARCHA`**, o sea la edad de cada sistema — y la edad de la
infraestructura es, en sí misma, una variable de vulnerabilidad. No es historia de
eventos, es historia del activo.

**10. Condiciones de uso**
`https://ssr.mop.gob.cl/robots.txt` → sólo bloquea `/wp-admin/`. **VERIFICADO.**
Organismo público, Ley 20.285. La copia en datos.gob.cl está bajo CC Non-Commercial
(o sea: uso no comercial permitido — nuestro caso).
**Nota de datos personales:** la planilla trae **RUT de las organizaciones** (comités y
cooperativas, personas jurídicas), no de personas naturales. Aun así: usar sólo comuna,
tamaño y beneficiarios; el RUT no aporta nada al modelo y no hay razón para propagarlo.

**11. Para qué sirve en el modelo — y esto es importante**
La literatura del propio proyecto señala que **la infraestructura rural pequeña es la más
vulnerable y es la que el estatuto de infraestructura crítica deja fuera**. Este archivo
es la prueba material de esa afirmación: **2.293 sistemas que le dan agua a la población
rural de Chile y que no figuran en ninguna matriz de infraestructura crítica.**

Con este dato el proyecto puede decir algo que hoy nadie dice:
> «En esta comuna, la zona de peligro Alto declarada por SERNAGEOMIN contiene 7 sistemas
> sanitarios rurales, clasificados MENOR, con 1.200 beneficiarios estimados y sin
> respaldo alternativo.»

Eso **es** el producto del proyecto, aplicado al punto ciego del sistema.

**12. Utilidad ★★★** — cambia prioridades de la forma más radical posible: **incorpora
una clase entera de activos que hoy tiene prioridad cero por no existir en el inventario.**

**13. Riesgos y límites**
- **Sin coordenadas.** La adscripción es a comuna, no a punto. Para cruzar con zonas de
  peligro (que son franjas geográficas, no comunas) hay que trabajar a nivel comunal y
  aceptar la pérdida de precisión — o conseguir la georreferenciación aparte.
- **No hay dato de estado para estos sistemas.** No existe un «Interrupciones en Línea»
  del agua rural: si un APR se queda sin agua, no lo publica nadie. **La consecuencia
  es invisible por diseño.** Es el vacío más serio que encontré en todo el dominio, y
  conviene decirlo así en el informe a SENAPRED: no es que el dato sea malo, es que no
  existe.
- La clasificación MENOR/MEDIANO/MAYOR es administrativa (Ley 20.998), no técnica: no
  mide fragilidad física.
- Actualización con rezago de meses.

---

# BLOQUE C — TELECOMUNICACIONES

---

## FICHA C1 ★ — SUBTEL · Estado de redes en emergencia

**1. Organismo y producto**
Subsecretaría de Telecomunicaciones (SUBTEL). Producto: reportes de «Estado de las redes
de telecomunicaciones» publicados **durante** cada emergencia.

**2. URL exacta**
- Ejemplos verificados por búsqueda:
  `https://www.subtel.gob.cl/subtel-estado-de-las-redes-ante-sistema-frontal/`,
  `https://www.subtel.gob.cl/conoce-el-estado-de-redes-de-telecomunicaciones-por-sistema-frontal/`,
  `https://www.subtel.gob.cl/reporte-estado-zona-norte3/` (este último **leído**: es de
  marzo de 2015).
- Informe trimestral estructural:
  `https://www.subtel.gob.cl/wp-content/uploads/2025/01/Informe_Nacional_4T_2024_08012025.pdf`
- Sitemap: `https://www.subtel.gob.cl/wp-sitemap.xml`

**3. Qué publica**
Durante un evento: qué porcentaje de las **estaciones base móviles** sigue operativo, qué
comunas están degradadas o sin conectividad, y qué empresas están afectadas.
Ejemplo real leído (marzo 2015): *«en las comunas de Taltal, Chañaral, Diego de Almagro y
Tierra Amarilla existe degradación de servicios y existen zonas sin conectividad»*, causa
«doble corte en la fibra óptica», empresas Claro, Entel, Movistar, VTR.
Ejemplo de julio 2026: *«la red móvil mantuvo más del 93% de sus estaciones base
operativas a nivel nacional»*.

**4. Familia:** ESTADO-CONSECUENCIA.

**5. Formato: NOTICIA EN HTML.** Prosa, con alguna tabla suelta. **No es dato.**
Es el mismo problema que ya tiene el proyecto con los eventos de la DMC.

**6. Acceso: anónimo — VERIFICADO** (páginas leídas sin registro).

**7. Granularidad territorial: variable e inconsistente.** A veces nacional («93% de
estaciones base»), a veces por región, a veces por comuna. Depende de quién redactó el
comunicado ese día. **Es el punto débil: no hay un esquema fijo.**

**8. Frecuencia: por evento.** No hay serie. Se publica cuando hay emergencia y se deja
de publicar cuando pasa.

**9. Historia disponible: sí, pero como archivo de noticias.** Las páginas viejas siguen
en línea (la de 2015 abre perfectamente). O sea: **hay once años de reportes de
emergencia archivados**, pero en prosa. Para convertirlos en dato habría que leerlos uno
por uno. Es historia, sí, pero cara.

**10. Condiciones de uso**
`https://www.subtel.gob.cl/robots.txt` → sólo bloquea `/wp-admin/`, y **publica su
sitemap** (o sea, invita a que lo indexen). **VERIFICADO.** Organismo público,
Ley 20.285.

**11. Para qué sirve en el modelo**
Las telecomunicaciones son el servicio **más dependiente de los otros dos**: cuando se
corta la luz, las antenas aguantan lo que dure su batería (típicamente horas) y después
caen. Es decir: **es la consecuencia de segundo orden**, y el proyecto justamente trata
de cascadas. Un reporte de SUBTEL que dice «93% operativo» al mismo tiempo que la SEC
dice «421.000 clientes sin luz» **es una medición del efecto dominó**. Nadie está
cruzando eso.

**12. Utilidad ★** — hoy no cambia la prioridad de ningún activo, porque no es dato
estructurado ni sistemático. Sirve para **contexto y para argumentar la cascada** en el
informe.

**13. Riesgos y límites**
- Sin estructura, sin periodicidad, sin esquema fijo. Automatizar esto es leer prosa:
  frágil.
- **Hallazgo comprobado y relevante:** SUBTEL se sumó al «grupo de gestión de crisis que
  el Ministerio de Energía mantiene con las empresas eléctricas para priorizar la
  reposición en puntos clave». Eso significa que **la cascada luz→telecomunicaciones ya
  se gestiona operativamente en Chile** — y que un producto que la cuantifique tiene
  destinatario concreto.
- No hay dato de **infraestructura fija** (fibra) en emergencia, sólo móvil.

---

## FICHA C2 ★ — SUBTEL · Datos abiertos (antenas y series estadísticas)

**1. Organismo y producto:** SUBTEL, publicaciones en el catálogo nacional.

**2. URL exacta**
`https://datos.gob.cl/api/3/action/package_search?q=telecomunicaciones` → **43 conjuntos**.
**VERIFICADO.** Los relevantes: **«Antenas»** y «antenas OCTUBRE» (XLSX),
«Series Estadísticas del Sector Telecomunicaciones» (XLSX), «Radio» (XLSX),
«Localidades Beneficiadas Proyecto IDCI» (XLSX).

**3. Qué publica:** el catastro de **antenas** (torres de telefonía móvil) y series
estadísticas del sector. **Contenido y campos NO VERIFICADOS** (no abrí los archivos:
está fuera de mi dominio, que es estado y no activo).

**4. Familia: ACTIVO.** **5. Formato:** XLSX. **6. Acceso:** anónimo — el catálogo
responde sin registro (**VERIFICADO** vía API CKAN).

**7. Granularidad:** presumiblemente por antena con ubicación. **NO VERIFICADO.**
**8.** Periódica. **9. Historia:** las series estadísticas son históricas; el catastro de
antenas es una foto.

**10. Condiciones de uso:** `datos.gob.cl` es el catálogo oficial de datos abiertos del
Estado; cada conjunto declara su licencia. Verificar la del conjunto puntual antes de usar.

**11. Para qué sirve:** completa el inventario de activos de telecomunicaciones, que hoy
falta. Es el equivalente telefónico de lo que Infotécnica es para la electricidad.

**12. Utilidad ★** desde mi dominio (es activo, no estado). **Se lo paso al agente que
cubre inventario de activos** — para ellos puede ser ★★★.

**13. Riesgos:** sin verificar; posible desactualización (el catálogo nacional tiende a
quedarse viejo, como se vio con el CUADRO 47 de la SISS).

---

# CONCLUSIONES DEL DOMINIO

## 1. El camino limpio es el regulador, no las empresas

Verifiqué que **las tres distribuidoras grandes publican sus cortes con excelente
detalle** (polígonos, clientes afectados, hora estimada de reposición). También verifiqué
que **CGE prohíbe expresamente reproducir ese contenido**, que Enel pone un cortafuegos
anti-bot, y que Edelmag pone otro. Dato visible no es dato libre.

Pero la SEC —organismo público, sin restricción de uso, con `robots.txt` permisivo—
**publica lo mismo, agregado por comuna y por hora, para todo el país y con seis años de
historia**. Es mejor dato para nuestro propósito (nacional, uniforme, comunal) y no tiene
problema legal.

**Recomendación: la fuente eléctrica del proyecto es la SEC. Punto.** Las empresas
quedan documentadas acá para el caso de que algún día haya convenio formal, y como
prueba de que el formato «polígono + clientes + hora de reposición» ya existe y funciona
en Chile — lo que hace muy razonable pedírselo al regulador de forma estandarizada.

## 2. La asimetría entre los tres servicios es enorme

| Servicio | ¿Hay estado en tiempo real? | ¿Hay historia? | ¿Por comuna? |
|---|---|---|---|
| **Electricidad** | Sí, nacional y consolidado (SEC) | **Sí, ≥ 6 años horarios** | **Sí** |
| **Agua urbana** | No consolidado (empresa por empresa) | Sólo indicador anual por empresa | No |
| **Agua rural (SSR)** | **No existe** | No | (inventario sí) |
| **Telecomunicaciones** | Sólo comunicados en prosa | Archivo de noticias | Inconsistente |

Esto tiene una consecuencia directa sobre el diseño: **la validación del modelo, en la
primera etapa, sólo se puede hacer con electricidad.** Y hay que decirlo así, sin
maquillarlo. Validar contra un solo servicio es una limitación real del instrumento, no
un detalle metodológico.

## 3. El eslabón que falta (y que ninguna fuente entrega)

Ninguna de las fuentes verificadas conecta **activo específico → falla → consecuencia**.
La SEC da la consecuencia (clientes sin luz por comuna) pero no la causa. El Coordinador
da el activo pero no su estado. Las empresas dan el polígono pero no qué instalación
falló.

Ese puente lo tiene que construir el proyecto, y **es su parte más frágil**. Conviene
declararlo desde el principio como supuesto explícito y no como resultado.
La ficha A9 (reportes de operación del Coordinador) es la candidata más prometedora a
cerrarlo: es el pendiente que más rendiría verificar.

## 4. Privacidad — lo que decidí no tocar

Tres de los mapas verificados permiten localizar a un cliente individual:
`buscar_cliente.php?id=` y `buscar_reclamo.php?id=` en CGE, `busquedaNroCliente` en
Saesa, y búsqueda por número de cliente en Enel. **No consulté ninguno, no guardé nada,
no aparecen datos de clientes en ningún archivo de este proyecto.**

Los archivos que sí descargué para verificar formato (el KMZ de CGE y el KML de Saesa)
contienen polígonos de sector y conteos agregados de clientes — **ningún dato que
identifique a una persona o un domicilio**. Están en el directorio temporal de la sesión
y no se incorporan al proyecto.

**Regla propuesta para el instrumento: la unidad mínima del proyecto es la comuna o el
activo. Nunca el cliente.** Y conviene que esté escrita en el documento que se entregue
a SENAPRED, porque es justamente la clase de proyecto en que la tentación existe.

## 5. Pendientes, en orden de rendimiento esperado

1. **Verificar los reportes de operación del Coordinador**
   (`https://www.coordinador.cl/reportes-y-estadisticas/`): si hay historia de fallas por
   instalación, cierra el eslabón que falta. **Máxima prioridad.**
2. **Georreferenciar las 1.269 subestaciones** del catastro. Sin coordenadas el catastro
   no se puede cruzar con ninguna amenaza. Sin esto, la ficha A3 vale la mitad.
3. **Pedir a la SEC, por Ley de Transparencia**, (a) el extracto histórico completo de
   clientes afectados por comuna/hora y (b) los datos base del SAIDI/SAIFI en planilla.
   Es más limpio, más rápido y más citable que consultar el endpoint miles de veces.
4. **Buscar si la SISS publica cortes de agua no programados** con fecha y comuna.
   Hoy es el hueco más grande del dominio del agua urbana.
5. **Verificar el campo `MXLOCATION`** del registro SSR: si es un identificador de
   localidad georreferenciable, resuelve de un saque la ubicación de los 2.293 sistemas
   rurales.
6. **Revisar los términos de uso de Grupo Saesa, Enel y Edelmag** antes de siquiera
   pensar en automatizar algo de ellos.
7. **Decidir cómo se pondera un «cliente»**: hoy un hospital y una casa cuentan igual.

---

## Anexo — Verificaciones realizadas (trazabilidad)

Todo lo marcado **VERIFICADO** en este documento proviene de peticiones HTTP reales
hechas el **15-ago-2026** desde este equipo, una o dos por fuente:

| Fuente | Prueba | Resultado |
|---|---|---|
| SEC `ClientesAfectados/Get` | POST `{}` | 200 · 12.777 b · serie horaria 08→15-ago-2026 |
| SEC `GetPorFecha` | POST `{2026,8,10,6}` | 200 · 140 filas comuna/clientes |
| SEC `GetPorFecha` | POST `{2026,7,17,12}` | 200 · 312 comunas · 421.263 clientes |
| SEC `GetPorFecha` | POST `{2024,8,1,12}` | 200 · 243 comunas · 446.892 clientes |
| SEC `GetPorFecha` | POST `{2020,1,15,12}` | 200 · 211 comunas · 18.188 clientes |
| Coordinador `/v1/subestaciones/` | GET | 200 · 507.255 b · 1.269 registros · 318 propietarios |
| Coordinador `/v1/subestaciones/199/` | GET | 200 · 355 b · sin coordenadas |
| CGE `mapa_cge.kmz` | GET | 200 · 23.706 b · 48 polígonos · generado 21:27 del mismo día |
| CGE `key.php?empresa=cge` | GET | 200 · hash de recarga |
| Saesa `mfallas.saesa.cl/outage.kml` | GET | 200 · 26.275 b · 39 placemarks con comuna |
| MOP `REGISTRO-DE-OPERADORES_V_MARZO_2026` | GET | 200 · 522.154 b · XLSX · ~2.293 sistemas |
| SISS CUADRO 47 (datos.gob.cl) | HEAD | 200 · XLSX · licencia CC-BY · metadato 2019 |
| SEC `Anuario-SEC-2023.pdf` | GET + extracción de texto | 2,6 MB · SAIDI presente, cifras en imagen |
| Enel `mapaemergencia.enel.com/mapa.html` | GET | 200 · 29.738 b · WAF Incapsula presente |
| Edelmag mapa de suministro | GET | 200 · 1,2 MB · indicios KML/KMZ · robots 403 |
| Chilquinta `mapainterrupciones` | GET ×3 | **falla TLS** · no verificable desde este equipo |
| robots.txt (7 sitios) | GET | ver cada ficha |

Archivos temporales de verificación: directorio scratchpad de la sesión. **No se
incorporó ningún dato descargado al proyecto**, salvo los conteos y estructuras citadas
en este documento.
