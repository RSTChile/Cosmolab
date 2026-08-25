# Fuente DGA — la verdad instrumental del río

**17-ago-2026.** Informe de lo que hay, cómo se accede, qué se bajó, qué **no**
se pudo y por qué.

Entregables de esta tanda:
`adaptadores/dga_hidrometria.py` · `datos/dga_estaciones.csv` ·
`datos/crudo/dga/2026-08-17/` (con su `PROCEDENCIA.txt`) · este documento.

---

## 0 · El veredicto, arriba de todo

> **Sí sirve.** La DGA da lo que al proyecto le faltaba: una medición
> instrumental de magnitud hidrológica, con 811 estaciones, caudal medio diario
> y **107 años de historia (feb-1913 → mar-2020)**, que reconoce el caso ancla
> de Copiapó 2015 con un peak de **14 veces la línea base**.
>
> **Pero no entrega un catálogo de desbordes hecho.** Entrega el ingrediente que
> falta —el *cuánto*— y deja dos cosas por resolver que no son culpa de la
> fuente: (a) caudal alto no prueba que el río se haya salido del cauce, y
> (b) el caudalímetro no distingue un desborde de un flujo de detritos.
> Ver §6, donde eso se desarrolla sin adornos.

---

## 1 · Qué publica la DGA, y por qué canal

La DGA declara **1.330 estaciones** transmitiendo en línea. Lo que encontramos,
canal por canal:

| # | Canal | Qué entrega | Estado |
|---|---|---|---|
| A | `dga.mop.gob.cl/estadisticas-estaciones-dga/` → xlsx | **3.810 estaciones activas** con coordenadas, cuenca, comuna, tipo | ✅ **bajado** |
| B | `snia.mop.gob.cl/sat/site/informes/mapas/mapas.xhtml` | **1.838 marcadores en línea (1.811 códigos únicos)**, lat-lon decimal + umbrales de la DGA | ✅ **bajado** |
| C | `snia.mop.gob.cl/BNAConsultas/reportes` (portal oficial de series) | series diarias / mensuales / anuales en XLS | ❌ **reCAPTCHA · no se automatiza** |
| D | `snia.mop.gob.cl/dgasat/…` (tiempo real detallado) | datos horarios por estación | ❌ **exige registro de usuario** |
| E | `dgasatel.mop.cl` | el sistema satelital antiguo | ❌ **el host ya no resuelve** |
| F | `datos.gob.cl` | — | ❌ **no publica hidrometría** (ver §3) |
| G | `lineasdebasepublicas.mma.gob.cl` | — | ❌ **no aporta** (ver §3) |
| H | `cr2.cl/datos-de-caudales/` (compilación (CR)2 del dato DGA) | **811 estaciones · caudal medio diario · 1913-2020** | ✅ **bajado, CC0** |

### A · El catálogo oficial de estaciones

`https://dga.mop.gob.cl/uploads/sites/13/2026/01/Informe-Nacional-09012026_124944.xlsx`

Es un enlace directo, sin formulario ni sesión. La DGA declara que está
actualizado al **15-ene-2026**. Trae: código, nombre, fecha de instalación,
región, provincia, comuna, cuenca / subcuenca / subsubcuenca, UTM WGS84,
lat-lon, altitud y tipo de estación.

Dos cosas que hay que saber antes de leerlo:

- **La cabecera está en la fila 17**, no en la 1. Arriba va el encabezado
  institucional y el resumen de la consulta que generó el informe. El adaptador
  busca la fila por contenido, no por número, porque la DGA regenera este
  archivo cada tanto.
- **Las coordenadas vienen en grados sexagesimales y SIN SIGNO** (`17°35'42''`).
  El signo lo pone el adaptador —Chile está entero al sur y al oeste— y lo deja
  declarado en el código, para que nadie lo deduzca después mirando el número.

### B · La red en línea, y el hallazgo escondido en el mapa

El mapa de tiempo real de la DGA trae **embebido en el HTML** un JSON con 1.838
marcadores de estaciones que transmiten —**1.811 códigos únicos**, hay 27
repetidos— en una llamada `initialize([...])`. Se extrae sin scraping de
interfaz: es un array JSON completo dentro de la página.

Aporta dos cosas que ningún otro canal da:

1. **lat-lon decimal de fábrica**, sin conversión de por medio. Sirvió para
   validar nuestra conversión desde sexagesimales (§4).
2. ★ **`umbral` y `nivelAlerta`** — que son **el umbral de crecida que la propia
   DGA le puso a la estación**. Es exactamente el dato que el proyecto necesita
   para no tener que inventar un percentil.

   **Y acá viene la mala noticia: sólo 4 de las 1.811 lo traen definido hoy.** El
   resto viene en `0.0`, que significa «no definido» y se guarda como sin dato,
   nunca como cero. **Vale la pena preguntárselo a la DGA por escrito**: si esos
   umbrales existen en su sistema y sólo no se publican, cambian el proyecto.

### C · ★ El portal oficial de series: por qué NO se automatiza

`snia.mop.gob.cl/BNAConsultas/reportes` es el canal oficial de las series
históricas. **No lo automatizamos, y no vamos a hacerlo**, por una razón que no
es técnica:

> El botón «Buscar» del formulario está **deshabilitado hasta pasar un
> reCAPTCHA v2** (sitekey `6LcRYtwqAAAAACZzq5MG06b68xaZpAhVHOJaLCFt`, función
> `desactivarbotonbusqueda()` disparada por `onload=desactivarBoton`).

Un reCAPTCHA es la forma estándar y explícita de decir *«esto se usa a mano»*.
Que además existan los topes declarados —4 años para datos diarios, 10 para
mensuales, 40 para anuales, máximo 10 estaciones por consulta (5 para altura y
caudal instantáneo)— apunta a lo mismo: el portal está dimensionado para
consultas humanas puntuales, no para barridos.

Se guarda el HTML en el crudo (`snia_bnaconsultas_reportes.html`) como evidencia
de por qué se tomó esta decisión, no como paso previo a saltarla.

**La vía formal existe y está escrita en el propio portal de la DGA:** solicitud
al amparo de la **Ley N° 20.285 de transparencia**, por la Plataforma SIAC del
MOP, con el `Formato-requerimiento2025.xls` que la misma página enlaza. Eso es
un trámite de Alexis, no del código, y es lo que hay que hacer si se quiere la
serie oficial completa y actualizada más allá de 2020.

### D-E · Tiempo real detallado: cerrado

`snia.mop.gob.cl/dgasat/pages/dgasat_main/dgasat_main.htm` responde, pero pide
usuario y contraseña (`check_login.asp`) o registro previo. Hay un botón de
«visita» que lleva a `dgasat_param.jsp`, pero es una interfaz de consulta
paramétrica pensada para navegación humana. `dgasatel.mop.cl` —el host que el
(CR)2 usaba en 2020— **ya no resuelve**: está muerto.

### H · De dónde salieron entonces las series

Del **(CR)2** (Centro de Ciencia del Clima y la Resiliencia, U. de Chile),
`https://www.cr2.cl/datos-de-caudales/`, archivo `cr2_qflxDaily_2020.zip`.

**Es dato DGA.** El propio archivo de descripción del (CR)2 lo declara:

> «*FUENTES: DGA web: Contiene los datos de 811 estaciones, descargados desde
> `http://snia.mop.gob.cl/BNAConsultas/reportes`. Estos registros se
> complementan con los datos de 256 estaciones automáticas descargados desde
> `http://dgasatel.mop.cl/index.asp`.*»

Y su licencia:

> «*DECLARACIÓN: Estos datos abiertos son publicados bajo licencia Creative
> Commons CC0 waiver. Han sido compilados para fines de investigación y
> docencia. No sustituyen los datos originales provistos por las instituciones
> responsables de los mismos.*»

Se bajaron tres archivos: el diario 2020 (811 est., feb-1913 → **mar-2020**), el
diario 2018 (809 est., → mar-2018) y el mensual 2018. El diario 2018 se guarda
además del 2020 **a propósito**: son compilaciones distintas del mismo archivo
histórico, y comparándolas se puede auditar si el compilador cambió valores del
pasado. Es la única forma de vigilar a un intermediario.

> ⚠ **Detalle que casi cuesta el dato:** el enlace rotulado «Cr2 QflxDaily 2019»
> en la página del (CR)2 aparece con tamaño 0,00 KB, como si estuviera roto.
> No lo está: entrega `cr2_qflxDaily_2020.zip`, la versión **más nueva** de
> todas. El nombre del enlace miente; el `content-disposition` de la respuesta
> dice la verdad. Está renombrado en el crudo a lo que realmente es.

---

## 2 · robots.txt y condiciones de uso (verificado el 17-ago-2026)

Antes de automatizar nada, como manda la regla del proyecto.

| Sitio | `robots.txt` | Lectura |
|---|---|---|
| `dga.mop.gob.cl` | `Disallow: /wp-admin/` y nada más | ✅ el xlsx y las páginas están permitidos |
| `snia.mop.gob.cl` | **404, no existe** | sin restricción declarada por esa vía — pero el reCAPTCHA sí es una restricción, y manda |
| `www.cr2.cl` | `Disallow: /wp-admin/` | ✅ permitido |
| `datos.gob.cl` | `Disallow: /api/` · `Crawl-Delay: 10` | ⚠ ver nota |
| `lineasdebasepublicas.mma.gob.cl` | `Disallow:` (vacío = todo permitido) | ✅ |

**Nota sobre `datos.gob.cl`:** su `robots.txt` bloquea `/api/`, que es el
boilerplate estándar de CKAN para que los buscadores no indexen respuestas JSON.
La API está **publicada como servicio** en el propio portal. Aun así se usó con
mano liviana: cuatro consultas, con 3-4 s entre una y otra, y no volveremos —
porque el resultado fue que **ahí no hay hidrometría** (§3).

### Condiciones de uso del MOP

De `https://www.mop.gob.cl/condiciones-de-uso/`, textual:

> «*Siempre que no sea para fines comerciales, el MOP autoriza la reproducción
> de los contenidos de este sitio, citando en forma clara su procedencia.*»
>
> «*No se permite copiar, duplicar, emitir, adaptar o cambiar en cualquier forma
> y por cualquier medio el contenido de las páginas web de mop.cl y su red de
> sitios asociados (Red MOP) **para fines comerciales**, a menos que exista un
> permiso escrito.*»

**Este proyecto es académico y no comercial → queda cubierto**, con la
obligación de citar «Dirección General de Aguas, MOP» cada vez.

Queda anotada **la misma salvedad que ya se anotó para el CSN**, y por la misma
razón: el destino declarado del instrumento es servir de insumo al COGRID y a
SENAPRED. El día que pase de investigación a apoyo operativo, «no comercial» ya
no es obviamente el encuadre correcto. **Pedir la autorización por escrito ahora
es barato**, y conviene pedirla junto con la solicitud de series por Ley 20.285:
es el mismo trámite y la misma ventanilla.

*(Dato menor pero real: `www.mop.cl` tiene el **certificado TLS vencido**. La
versión viva es `www.mop.gob.cl`. Se anota para que nadie pierda una hora con
eso.)*

---

## 3 · Lo que se probó y NO sirvió

Se probó, porque la tarea lo pedía. El resultado honesto:

**`datos.gob.cl` — no publica hidrometría de la DGA.** Cuatro búsquedas en su
API CKAN:

| consulta | resultados |
|---|---|
| `hidrométrica` | **0** |
| `fluviom` | **0** |
| `caudal` | 1, y es «Emisiones al agua» del Ministerio del Medio Ambiente |
| `DGA` | 3: derechos de aprovechamiento de aguas, la Mapoteca digital, y patentes de aguas de Tesorerías |

La DGA **sí** tiene organización en el portal, pero publica ahí sus registros
administrativos (derechos de agua) y cartografía. **Ni una serie de caudal.**

**`lineasdebasepublicas.mma.gob.cl` — no aporta.** Es una aplicación de página
única (Laravel + Vite) sin contenido en el HTML servido y sin API documentada
alcanzable desde fuera. Aunque publicara la red hidrométrica como capa
geográfica, sería la ubicación de las estaciones —que ya tenemos, y de primera
mano— y no las series.

**No hay API abierta de la DGA.** Ni REST, ni WFS, ni servicio de datos
abiertos. El canal oficial de series es el portal con captcha, y el canal formal
para pedir más es la Ley de Transparencia.

---

## 4 · Qué se bajó, y qué salió

Todo el crudo en `datos/crudo/dga/2026-08-17/`, con `PROCEDENCIA.txt` que
detalla URL, tamaño y fecha de cada archivo.

### `datos/dga_estaciones.csv` — 4.110 estaciones

Unión de las tres fuentes, cruzadas por código de estación normalizado a 8
dígitos (las tres lo escriben distinto: `01000005`, `08382002-0`, `1201005`; sin
unificarlos el cruce da cero).

```
estaciones distintas (unión de las 3 fuentes) : 4.110
    activas en el catálogo oficial 2026       : 3.769
    transmitiendo en línea hoy                : 1.811
  ★ con serie diaria de caudal                :   811
    con coordenadas                           : 4.110  (el 100 %)
    fluviométricas (miden caudal)             :   670
    con umbral de crecida definido por la DGA :     4
```

Cada fila lleva tres banderas —`activa_2026`, `transmite_en_linea`,
`con_serie_diaria`— que dicen **qué se le puede pedir a esa estación**. Ninguna
estación se descartó por estar en una sola fuente: las **297 que tienen serie
diaria pero ya no están en el catálogo activo** son estaciones muertas con
historia, y para validar un evento de 1987 valen igual que una viva.

### Validación cruzada de las coordenadas

1.762 estaciones aparecen en el xlsx (sexagesimales, convertidas por nosotros) y
en el JSON del SAT (decimales de fábrica). Comparadas:

| | |
|---|---|
| mediana de la diferencia | **0,000002°** (≈ 20 cm) |
| percentil 95 | 0,000006° |
| dentro de 0,01° (≈ 1,1 km) | **99,0 %** |

La conversión de sexagesimales está bien. Los 128 casos de la unión completa que
se contradicen por más de 0,01° **no se esconden**: van en la columna
`coord_discrepancia_grados` del CSV.

- 91 entre 0,01° y 0,05° — deriva menor entre bases
- 18 entre 0,05° y 0,10°
- 15 entre 0,10° y 1,00°
- 4 de 1° o más, y de ésos **dos tienen explicación**: el xlsx oficial de la DGA
  **trunca el dígito de las centenas en la longitud de Rapa Nui** —escribe
  `09°25'17''` donde va `109°25'17''` (estaciones `05610000` POIKE y `05610001`
  RAPA NUI)—. Es un defecto del archivo oficial, no de la conversión. El CSV usa
  la coordenada del SAT, que es la correcta.
- `11340001` POZO RODEO LOS PALOS difiere **exactamente 1°** entre fuentes y
  **ninguna de las dos permite arbitrar**. Queda declarado, sin resolver.

### Las series diarias

**811 estaciones · caudal medio diario · feb-1913 → mar-2020 · 99 cuencas.**
No hubo que muestrear: la compilación del (CR)2 viene entera, así que el crudo
tiene el país completo, no una selección nuestra.

Cobertura por franja, que era lo que la tarea pedía verificar:

| franja | estaciones con serie diaria |
|---|---|
| norte árido (> 30° S) | **146** |
| centro (30-36° S) | 304 |
| sur húmedo (36-42° S) | **252** |
| austral (< 42° S) | 109 |

Y por hasta cuándo llega el registro: **429 llegan a 2018 o después**, 65 a
2010-2017, 69 a 1990-2009, 248 se cortan antes de 1990.

---

## 5 · La prueba: el caso ancla

Copiapó, 24-25 de marzo de 2015. SENAPRED lo clasifica como «Inundación»; fue un
flujo de detritos; DesInventar lo nombra bien pero no lo mide. Esto es lo que
vieron los instrumentos de la DGA (`python adaptadores/dga_hidrometria.py --copiapo`):

| estación | base 23-mar | 24-mar | 25-mar | salto | después |
|---|---|---|---|---|---|
| Río Copiapó en Pastillo | 1,42 | 4,91 | **19,70** | **× 14** | 12,5 → 8,2 → 3,7 → 1,7 |
| Río Pulido en Vertedero | 0,95 | 2,48 | **12,20** | **× 13** | ★ **la estación se apaga** |
| Río Copiapó en La Puerta | 0,57 | **3,50** | — | × 6 | ★ **la estación se apaga** |
| Copiapó en Mal Paso Ab. | 0,001 | 0,001 | — | — | ★ **el registro TERMINA el 24-03-2015** |

Pastillo, que sobrevivió, dibuja el hidrograma completo: subida en un día, peak,
y recesión en cuatro. **Eso es una crecida medida, no un testimonio.**

Y la misma herramienta, corrida a ciegas sobre otras cuencas
(`--crecidas`), saca del ruido eventos que uno reconoce sin ayuda:

| estación | fecha | caudal | × sobre base |
|---|---|---|---|
| Río Mapocho en Los Almendros | **1993-05-03** | 158 m³/s | **× 49** |
| Río Mapocho en Los Almendros | 1986-06-16 | 232 m³/s | × 64 |
| Río Biobío en Rucalhue | 2001-05-28 | 4.668 m³/s | × 41 |
| Río Biobío en Rucalhue | 1991-05-28 | 4.179 m³/s | × 34 |
| Río Cautín en Rari-Ruca | 1991-05-28 | 603 m³/s | × 15 |

El 3 de mayo de 1993 es el aluvión de la Quebrada de Macul. Mayo de 1991 aparece
**simultáneamente en Biobío y en Cautín**, que es la firma de un temporal
regional. La herramienta no fue calibrada contra ninguno de estos eventos: sólo
mide el agua.

---

## 6 · ¿Sirve para construir un catálogo de desbordes? — evaluación honesta

### Sí, para tres cosas, y bien

1. **Da la magnitud absoluta que faltaba.** El diagnóstico del proyecto
   (`CORRECCION_RAREZA_PELIGRO.md`) fue que la variable *«mide RAREZA y no
   PELIGRO: falta magnitud absoluta»*. El caudal es magnitud absoluta, en
   unidades físicas, medida en el punto. Es el ingrediente que faltaba.
2. **Es independiente de las tres fuentes de eventos que ya hay.** ReTeRM,
   SENAPRED y DesInventar dependen todas de que alguien *reporte*. El
   caudalímetro registra aunque nadie mire. Eso corta la circularidad que ya
   mordió al proyecto con ReTeRM.
3. **Cubre el norte árido, que es donde las otras fuentes son más flacas.** 146
   estaciones sobre 30° S con serie diaria, contra los 23 eventos de ReTeRM en
   esa zona.

### No, todavía no, por tres razones que hay que resolver

**(a) Caudal alto ≠ desborde. Falta la capacidad a cauce lleno.**
Para decir «el río se salió» hace falta saber cuánta agua cabe en esa sección, y
la DGA no publica ese número junto con la serie. Lo único que se puede afirmar
sin inventar nada es *«caudal excepcional respecto de la propia historia de esta
estación»*, que es lo que calcula `crecidas()` y lo que el CSV rotula así.

Hay dos salidas, y ninguna es gratis: pedirle a la DGA los umbrales que su
propio sistema SAT ya tiene (§1-B: **existen, sólo 4 están publicados**), o
derivar la capacidad desde geometría de cauce, que es otro proyecto.

**(b) El caudalímetro no distingue desborde de flujo de detritos.**
Copiapó 2015 fue un aluvión y el instrumento lo registró igual de bien. Esto **no
es un defecto**: significa que el caudal aporta el *cuánto* y **el enrutador de
terreno sigue siendo el que decide el *cuál***. Es exactamente la arquitectura
que propone `ESTUDIO_VECTORES_DE_AMENAZA.md`. Pero hay que decirlo en voz alta:
**la DGA no resuelve por sí sola el problema de clasificación que tiene
SENAPRED.** Resuelve el de medición.

**(b-bis) El percentil de historia propia subestima los eventos recientes.**
En el mismo caso de Copiapó, `crecidas()` marca Pulido en Vertedero (× 16,6) y
**no** marca La Puerta: su peak de 3,50 m³/s queda bajo su propio percentil 99,
porque ese río tuvo caudales mucho mayores en las décadas de registro anteriores.
Una estación con historia larga y régimen cambiado **esconde** los eventos de
hoy. Está anotado en el código: hay que mirar `razon_base` sin filtrar por
percentil, o fijar el umbral contra un período de referencia declarado.

**(c) ★ Censura por arriba: los eventos peores rompen el instrumento.**
En Copiapó, **tres de cuatro estaciones dejaron de transmitir el día siguiente
al peak**, y una no volvió nunca. El sesgo va justo en contra de lo que
buscamos: *cuanto peor el evento, más probable que el dato falte*. Un catálogo
armado ingenuamente sobre esta serie **subrepresenta sistemáticamente los
extremos**.

Por eso `crecidas()` devuelve `apagon_posterior` como **señal, no como hueco**:
una estación que se calla justo después de una crecida está diciendo algo, y
tratarlo como dato faltante es tirar la información más valiosa que hay.

### El hueco temporal, que es de verdad

**La serie consolidada termina en marzo de 2020.** De ahí a hoy hay **seis años
sin serie**: sólo tiempo real en el portal SAT, sin archivo histórico
descargable. Los eventos recientes —incluido cualquier invierno que el proyecto
quiera analizar de 2020 en adelante— **no están cubiertos**. Esto no se arregla
con código: se arregla con la solicitud por Ley 20.285.

---

## 7 · Pendientes, en orden de valor

1. ★ **Solicitud formal a la DGA por Ley 20.285** (Plataforma SIAC, formato
   `Formato-requerimiento2025.xls`), pidiendo tres cosas en un mismo trámite:
   - serie de caudal medio diario **2020 → hoy** para las estaciones
     fluviométricas (cierra el hueco de seis años);
   - **los umbrales de crecida del sistema SAT** — existen, sólo 4 de 1.811
     están publicados, y son la respuesta directa al problema (a);
   - autorización escrita de uso, previendo el día en que el instrumento pase de
     académico a apoyo operativo del COGRID.
2. **Capacidad a cauce lleno** para las estaciones fluviométricas de las cuencas
   donde SENAPRED registra inundaciones. Sin eso, «desborde» sigue siendo
   «caudal excepcional».
3. **Cruzar `crecidas()` contra los eventos de SENAPRED y DesInventar** por
   cuenca y fecha. Es la prueba que sigue: si el instrumento ve una crecida
   grande el día que la prensa reporta una inundación, la etiqueta queda
   validada por una fuente que no la produjo.
4. **Comparar `cr2_qflxDaily_2018` contra `cr2_qflxDaily_2020`** para verificar
   que el compilador no alteró el pasado. Los dos crudos están guardados
   justamente para eso.
5. **Precipitación y nieve de la DGA.** Este adaptador sólo trae caudal. La red
   mide también precipitación, temperatura, humedad, altura de nieve y nieve
   equivalente en agua, y el (CR)2 publica la precipitación por la misma vía.
   Alimentaría `Ppers` y `Iso0` con estación real en vez de reanálisis, que es
   una deuda vieja del proyecto.

---

## 8 · Cómo se usa

```bash
PY=/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/.venv-esa/bin/python

# catálogo consolidado + informe en pantalla; --csv lo escribe a datos/
$PY adaptadores/dga_hidrometria.py --csv

# el caso ancla: Copiapó, marzo de 2015
$PY adaptadores/dga_hidrometria.py --copiapo

# crecidas sobre el percentil 99 de la propia estación
$PY adaptadores/dga_hidrometria.py --crecidas 03430003 08317001 09123001
```

Desde otro módulo:

```python
from adaptadores import dga_hidrometria as dga

filas, problema = dga.traer()                       # catálogo consolidado
series, aviso = dga.serie_diaria(["08317001"],      # caudal diario m³/s
                                 "1990-01-01", "2020-03-31")
eventos = dga.crecidas(series["08317001"])          # días excepcionales
```

El adaptador **lee sólo del crudo**: nunca vuelve a la red por su cuenta. Si
falta un archivo, degrada a «sin dato» y lo dice, como manda la regla 1 de
`FUENTES_DE_DATOS_NACIONALES.md`.

---

## 9 · Cómo citar

> Dirección General de Aguas, Ministerio de Obras Públicas de Chile. Red
> Hidrométrica Nacional. Catálogo de estaciones activas al 15-ene-2026 y red en
> línea al 17-ago-2026.
>
> Series de caudal medio diario compiladas por el Centro de Ciencia del Clima y
> la Resiliencia (CR)2, Universidad de Chile, a partir de datos de la DGA
> (`cr2_qflxDaily_2020`, licencia CC0), feb-1913 a mar-2020.
