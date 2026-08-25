# Inventario georreferenciado: el estado real

Actualizado el 20-ago-2026.

Este documento es el estado del inventario de activos ubicados del proyecto. La
tabla del cierre se **genera** con `adaptadores/inventario_resumen.py`, no se
escribe a mano: si se escribiera a mano, empezaría a mentir la primera vez que
alguien vuelva a correr un adaptador y la fuente haya cambiado.

---

## 1 · De dónde veníamos

Hace cinco días el inventario georreferenciado del proyecto eran **39
subestaciones eléctricas capturadas a mano**. Elegidas, como dice el propio
`SUBPROYECTO_SUBMATRICES.md`, «no por importancia sino porque eran lo único
georreferenciado que había a mano».

Esta semana se les sumaron los 14.039 tramos viales y los 6.742 puentes del
Ministerio de Obras Públicas. Y esta tanda agrega el resto.

**La regla que lo hizo posible ya estaba escrita en el proyecto**, y funcionó:

> «Antes de capturar un solo dato a mano, buscar quién ya lo tiene. Chile tiene
> mucho más catastro público del que parece.»

No se capturó un solo dato a mano. Todo lo que sigue lo publica un organismo del
Estado chileno, gratis y sin credenciales.

---

## 2 · Los tres hallazgos que cambian el tamaño del problema

### 2.1 ★ El cuello de botella del Coordinador Eléctrico no existía

`SUBPROYECTO_SUBMATRICES.md` sección 5 lo daba por el problema central del
proyecto, textual:

> «El caso testigo es el Coordinador: tiene las 1.269 subestaciones y **ninguna
> trae coordenada**. Sin coordenada, un activo no se puede cruzar con ninguna
> amenaza — y el cruce es todo el proyecto.»

Eso era cierto **del endpoint que se había probado**, y falso del servicio.

`/v1/subestaciones/` no trae coordenadas. Pero la misma interfaz publica
`/v1/subestaciones/extended/`, y ahí cada subestación viene con `latitud` y
`longitud`. **1.226 de 1.273 (96,3 %)**. El servicio publica además su
especificación completa —176 rutas— en
`https://api-infotecnica.coordinador.cl/?format=openapi`, sin credenciales.

La lección vale más que el dato, y conviene dejarla escrita:

> **La conclusión «esta fuente no tiene el dato» se había sacado de UN endpoint,
> no del servicio. Antes de declarar que una fuente no tiene algo, hay que leerle
> el catálogo completo.**

**Y hay un segundo hallazgo dentro del primero.** Al poner las 39 subestaciones
del piloto al lado de las coordenadas del propio operador, 30 calzan por nombre
y su **desviación mediana es de 3,0 km**. La peor —«Subestación Crucero»— estaba
a **175 kilómetros** de donde está. Un cruce con amenaza hecho sobre esa
coordenada no habría dado un resultado impreciso: habría dado un resultado de
otra región. Es la validación independiente más contundente que ha tenido la
regla del proyecto de nunca aproximar en silencio.

### 2.2 ★ SENAPRED —el destinatario del proyecto— ya publica su propio inventario

`INTEGRACION_SENAPRED.md` fija que SENAPRED es el **destinatario**, no una
fuente. Eso sigue siendo cierto para las alertas. Pero SENAPRED mantiene un
servidor ArcGIS público, abierto, cuyo `robots.txt` dice `Disallow:` (o sea,
todo permitido), con **veintinueve capas de infraestructura crítica
georreferenciada de todo el país** bajo el nombre «Infraestructura Vulnerable»:
bomberos, carabineros, salud, educación, energía, telefonía, puertos, pasos
fronterizos, penitenciarios, residencias de infancia, establecimientos de adulto
mayor, municipios, supermercados.

No estaba en ningún catálogo de datos abiertos. Está en el servidor de su visor
de Gestión del Riesgo de Desastres, a la vista.

**Esto obliga a matizar una frase del proyecto.** `INTEGRACION_SENAPRED.md` dice:
«Chile sabe dónde está la amenaza y qué infraestructura importa; nadie cruza las
dos cosas». Hay que corregirlo: **SENAPRED tiene las dos mitades**. Y el
Ministerio de Salud va más lejos todavía: publica una capa
(`Amenazas2026_RedEstablecimientos`) con puntajes ponderados de amenaza volcánica,
tsunami, incendio y falla activa por establecimiento, y otra
(`EESS_Cortes_Electricos_Vista`) que cruza cada establecimiento con los cortes
eléctricos vigentes, actualizada cada hora.

Eso no hunde el proyecto; lo precisa. Lo que nadie hace es el cruce
**transversal, multi-sector y con una métrica común**. Salud cruza salud.
SENAPRED tiene el inventario y tiene las alertas, en servidores distintos y sin
una medida de qué pasa cuando un activo cae. Ese sigue siendo el hueco. Pero
conviene decirlo bien, porque presentarse diciendo «nadie hace esto» ante quien
sí hace una parte es la forma más rápida de perder una conversación.

### 2.3 ★ El agua potable rural mide la cascada energía → agua

Los 2.475 sistemas de agua potable rural del Ministerio de Obras Públicas traen
tres campos que ninguna otra capa del inventario tiene: si el sistema tiene
grupo electrógeno, si ese grupo alcanza, y si el sistema ya depende de camión
aljibe. Más la población servida por cada punto.

Con eso, la cadena de fallo deja de ser una hipótesis y pasa a ser un conteo:

| | sistemas | personas |
|---|---:|---:|
| con grupo electrógeno | 1.605 | 1.911.440 |
| **SIN grupo electrógeno** | **833** | **457.399** |
| sin dato del campo | 37 | 22.753 |
| ya dependen de camión aljibe | 115 | 104.828 |

**Un corte eléctrico prolongado deja sin agua a 833 sistemas y a unas 457.000
personas.** No es una estimación del proyecto: es el propio ministerio diciendo,
sistema por sistema, cuál tiene respaldo y cuál no.

Los 37 «sin dato» se cuentan aparte a propósito. Un campo vacío no es un «no»:
es un hueco, y el hueco se declara.

---

## 3 · La tabla del estado real

Genera con:

```bash
.venv-esa/bin/python infraestructura/adaptadores/inventario_resumen.py
.venv-esa/bin/python infraestructura/adaptadores/inventario_resumen.py --markdown
```

Cuenta como georreferenciada una fila con latitud y longitud presentes, legibles
como número y dentro de la caja de Chile (continental, insular y antártico).
Nada más. Una fila con coordenada vacía no cuenta; una con coordenada fuera de
Chile tampoco, y se cuenta aparte — un activo mal ubicado es peor que uno sin
ubicar, porque se cruza con la amenaza equivocada y nadie lo nota.

<!-- TABLA_PRINCIPAL -->

### Lo que NO se suma, y por qué

El total no es la suma de todos los archivos. Tres razones para excluir, cada
una declarada:

1. **Duplicación entre fuentes.** Los puentes del Ministerio de Obras Públicas y
   los de SENAPRED son mayormente los mismos. Los establecimientos de salud del
   Departamento de Estadísticas e Información de Salud y los de SENAPRED,
   también. Cuenta la fuente que manda; la otra queda de respaldo.
2. **Permiso versus activo.** Las antenas «autorizadas» de SUBTEL son actos
   administrativos, no fierros instalados. Cuentan las «en servicio».
3. **Proyecto versus activo.** Los contratos de servicios sanitarios rurales son
   obras en curso, no sistemas operando.

<!-- TABLA_EXCLUIDOS -->

---

## 4 · Las fuentes, una por una

| Organismo | Servicio | Formato | robots.txt | Licencia | Automatización |
|---|---|---|---|---|---|
| **MOP** · Obras Hidráulicas, Obras Portuarias, Aeropuertos | `rest-sit.mop.gob.cl/arcgis/rest/services` | esriJSON | **404** (sin restricciones) | no declarada en el servicio; la Red Vial del mismo ministerio es CC BY-NC | sí · **PENDIENTE** confirmar con la UGIT |
| **Coordinador Eléctrico Nacional** · Infotécnica | `api-infotecnica.coordinador.cl/v1/…` | JSON, con OpenAPI | **404** en el host de datos; el sitio institucional sí restringe (ver abajo) | no declarada | sí, con la salvedad de abajo · **PENDIENTE** confirmación escrita |
| **MINSAL** · DEIS, vía `datos.gob.cl` | CSV directo del recurso | CSV `;` UTF-8 | 200 · restringe `/api/`, `Crawl-Delay: 10` | ★ **CC-Zero** (dominio público) | sí · **la única sin pendientes** |
| **MINEDUC** · Centro de Estudios | `datosabiertos.mineduc.cl/…/Directorio-Oficial-EE-2025.rar` | RAR con CSV | **403** (cortafuegos, no hay robots legible) | CC-BY (declarada en el espejo de `datos.gob.cl`) | sí, una descarga por publicación |
| **SENAPRED** · SIIE | `visor-grd.senapred.gob.cl/arcgis/rest/services/SIIE` | esriJSON | 200 · `Disallow:` → **todo permitido** | no declarada (`copyrightText` vacío) | sí · **PENDIENTE** confirmación escrita |
| **SUBTEL** | `licancabur.subtel.gob.cl/server/rest/services` | esriJSON | **404** en el host de datos; el sitio institucional sólo bloquea `/wp-admin/` | no declarada (`copyrightText` vacío) | sí · **PENDIENTE** confirmación escrita |

### La salvedad del Coordinador Eléctrico, que hay que leer

`api-infotecnica.coordinador.cl/robots.txt` devuelve 404: el host de datos no
declara restricciones, y el servicio está documentado con OpenAPI y pensado para
acceso programático.

**Pero** `www.coordinador.cl/robots.txt` sí declara, y no es menor:
`Content-Signal: search=yes, ai-train=no, use=reference`, y `Disallow: /` para
una lista de agentes de inteligencia artificial. Es otro host y otro servicio,
pero la intención del organismo queda expresada. El adaptador la respeta: se
identifica con el agente del proyecto y no con un agente de rastreo de
inteligencia artificial, usa los datos como **referencia** en investigación —que
es lo que el sitio permite—, **no entrena ningún modelo** con ellos —que es lo
que prohíbe—, y hace una descarga completa en vez de rastreo continuo.

### Lo que se decidió NO usar, y por qué

**★ IGM · `gis.inv.igm.cl/vector/rest/services/SIIE_v3`.** El Instituto
Geográfico Militar expone, sin token y sin `robots.txt`, unas cuarenta capas
nacionales de infraestructura crítica —incluidos 79.178 grifos, que ninguna otra
fuente tiene— más capas de amenaza ya cruzables (inundación por tsunami, radio
de volcanes, vías de evacuación). Conceptualmente es la mejor fuente encontrada
en todo el barrido.

**No se bajó nada de ahí.** Tres razones, en orden de peso:

1. El **mismo servidor** aloja capas militares (`Unidades_militares`, `SIGINT`,
   `MineWarfare`, `AirMissile`) y una carpeta `250K` que **sí exige token**. O
   sea: el servidor sabe restringir. Que `SIIE_v3` esté abierta puede ser una
   configuración no intencionada, no una publicación de datos abiertos.
2. **No declara licencia** en ninguna capa.
3. La regla del proyecto dice «robots.txt y condiciones de uso **antes** de
   automatizar». Un `robots.txt` ausente no es un permiso; es la ausencia de una
   respuesta.

Queda como **pista documentada y trámite pendiente**: pedir autorización formal
al IGM. Es lo correcto, y probablemente abra un convenio en vez de un endpoint.

**Otras que se probaron y no sirven** (para que nadie repita el trabajo):

- `catalogo.ide.cl` y el catálogo GeoNetwork de la Infraestructura de Datos
  Geoespaciales de Chile: **no resuelven**. No existe hoy una interfaz de
  catálogo geoespacial nacional consultable programáticamente.
- `www.geoportal.cl/ArcGIS/rest/services`: **404**. El servidor ArcGIS nacional
  que citan casi todos los metadatos de `datos.gob.cl` está descontinuado, y con
  él quedaron rotos los enlaces de subestaciones, aeropuertos, unidades
  policiales, infraestructura portuaria y faenas mineras. `datos.gob.cl` sirve
  hoy como índice de qué existe y quién lo publica, no como fuente de descarga.
- `www.ide.cl` y `deis.minsal.cl`: su `robots.txt` declara `Disallow: /` para
  agentes automáticos de inteligencia artificial. **No se rastrearon.** En el
  caso del DEIS no hizo falta: el mismo dato está en `datos.gob.cl` con licencia
  CC-Zero. En el de la IDE queda un pendiente documentado (los archivos de
  división político-administrativa de SUBDERE), que un humano puede bajar desde
  el navegador sin problema.
- `sitport.directemar.cl` y `orion.directemar.cl`: la interfaz responde y da
  estado portuario en tiempo real (restricciones, avisos, meteorología por
  bahía), pero **sin coordenadas**. Es una buena fuente de *estado operacional*,
  no de inventario.
- **DGAC**: no se encontró ninguna fuente georreferenciada oficial de aeródromos.
  Lo publicado en `datos.gob.cl` es un CSV de la red aeroportuaria **al 31 de
  diciembre de 2013**, sin coordenadas. El sustituto usado es la Red
  Aeroportuaria Nacional de la Dirección de Aeropuertos del MOP (316 registros,
  con coordenada), que además trae la clasificación de aislamiento.

---

## 5 · Lo que quedó bloqueado

| Qué | Estado | Qué haría falta |
|---|---|---|
| **Trazado de las líneas de transmisión del Coordinador** | 40.643 km SIN geometría. El campo `coordenadas` viene vacío en los 2.926 tramos — comprobado uno por uno | Pedirlo al Coordinador. Mientras tanto, la capa `energia_lineal` de SENAPRED sí trae traza, pero es otro universo y hay que compararlas antes de fusionar |
| **47 subestaciones sin coordenada** | 3,7 % del total | Es una lista corta y con nombre. Pedirlas al operador es un trámite acotado, no un problema abierto |
| **219 derivaciones de línea (taps) sin coordenada** | sólo 58 de 277 ubicadas | Ídem |
| **Jardines infantiles JUNJI / INTEGRA / VTF** | están dentro de la capa de educación de SENAPRED (~4.300), pero mezclados con los colegios que ya aporta MINEDUC | Extraerlos filtrando por `dependencia`. Tarea chica, no hecha |
| **Fibra óptica nacional** | no existe capa consolidada. SUBTEL la publica fragmentada por operador y por oficio | Lo bajado son los 430 tramos del programa de conectividad digital, que **no** son la red completa |
| **Catastro nacional de albergues** | **no existe** como dato público. `geoportal.senapred.cl` no resuelve; el mapa de albergues de la cuenta ArcGIS Online de SENAPRED está en privado; en `datos.gob.cl` no hay ninguno | Los sustitutos son los 538 recintos deportivos y los 11.829 establecimientos educacionales. Ojo: la matrícula **no es capacidad de albergue certificada** |
| **IGM `SIIE_v3`** | no se usó, por decisión | Autorización formal del IGM |
| **Licencia de cuatro de las seis fuentes** | no declarada | Cartas. Es trámite de Alexis, no del proyecto |
| **Caletas pesqueras** | no se encontró catastro nacional con coordenadas | — |

---

## 6 · Reglas que este trabajo dejó comprobadas

**No paginar con `resultOffset` sin verificarlo primero.** El servidor del
Ministerio de Obras Públicas (ArcGIS 10.2.1) lo acepta sin protestar y lo
**ignora**, devolviendo siempre desde el principio: un inventario paginado así
sale con los primeros mil registros repetidos y nadie se entera. El de SUBTEL
(ArcGIS 11.5) sí lo respeta. La comprobación quedó dentro del adaptador de
SUBTEL, en `verificar_paginacion()`, para que vuelva a hacerse si el servidor
cambia. Donde no se respeta, se pagina por identificador: `returnIdsOnly=true`
primero y después lotes por `objectIds`, con lo que el conteo cierra por
construcción.

**Las coordenadas se verifican contra sí mismas.** Las capas de agua potable
rural y de aeropuertos traen la coordenada dos veces: en la geometría y en los
atributos `LATITUD`/`LONGITUD`. El adaptador pide la geometría reproyectada a
EPSG:4326 por el servidor —que sabe qué datum usó— y la contrasta con el
atributo. Discrepancias mayores a 100 m encontradas: **cero**. Que dé cero no
hace inútil la comprobación: la hace barata.

**Cada fuente tiene su trampa, y todas están anotadas en el módulo que la lee.**
El mismo estado escrito con dos capitalizaciones distintas en el archivo del
DEIS (210 establecimientos se pierden con un filtro por igualdad exacta). Las
coordenadas con coma decimal en MINEDUC. Los códigos comunales que pierden el
cero a la izquierda si se leen como número. La geometría entregada como texto en
algunas capas de SENAPRED. Un archivo RAR donde se esperaba un ZIP. Ninguna de
esas trampas falla a gritos: todas fallan callado, que es peor.

**No se recolectaron datos de personas.** Los campos `rut`, `MRUN`,
`RUT_SOSTENEDOR`, teléfonos, correos y nombres de responsables que traen algunas
fuentes **no se copiaron a ninguna tabla**. Quedan sólo en el archivo crudo, tal
como los entregó cada organismo, porque el crudo debe ser exactamente lo que se
recibió.

**Todo el crudo está guardado antes de procesar**, en
`datos/crudo/<fuente>/<fecha>/`, con su procedencia al lado: URL exacta,
cabeceras HTTP, cuántos registros declaró el servicio y cuántos llegaron.

---

## 7 · La advertencia que sigue vigente

`SUBPROYECTO_SUBMATRICES.md` sección 7 lo dijo antes de que esto empezara, y
sigue siendo lo más importante de este documento:

> «Este sub-proyecto es el que puede hundir al proyecto entero, y no por difícil
> sino por **tentador**. Poblar catastros es trabajo visible, medible y
> agradable: se ve la barra avanzar. Es muy fácil pasar seis meses poblando y
> llegar a fin de año con 200.000 activos ubicados y **ninguna respuesta** sobre
> si el método sirve.»

El inventario acaba de crecer de 20.820 activos a más de 130.000 en una noche.
La barra avanzó muchísimo. **Y el ancla de Copiapó sigue fallando**: la variable
mide rareza y no peligro, y mientras eso no se resuelva, cada activo nuevo es un
activo más al que el modelo le va a decir algo equivocado.

La Fase B —ingerir lo que ya existe— está esencialmente hecha, y salió más
barata de lo previsto. La Fase D —capturar a mano— sigue debiendo esperar a que
el método pase su validación.
