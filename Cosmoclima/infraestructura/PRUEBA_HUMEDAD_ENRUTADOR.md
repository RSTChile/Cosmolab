# Prueba falsable del «enrutador de amenazas»: la humedad de suelo previa

**16-ago-2026.** Primera prueba falsable de la estructura propuesta en
`ESTUDIO_VECTORES_DE_AMENAZA.md` §6. Criterio fijado antes de calcular, no
tocado después.

# ► VEREDICTO: **NO PASA**

Y no pasa de la peor manera posible para la hipótesis: **el signo del efecto
depende de qué catálogo de remociones se use.** Con ReTeRM sale al revés de lo
predicho; con SENAPRED sale en el sentido predicho. Ninguno de los dos alcanza
significación. Con el dato disponible hoy, la pregunta no queda respondida ni
en un sentido ni en el otro.

---

## 1 · La hipótesis que se puso a prueba

> «Si hay lluvias persistentes, baja humedad del suelo, isoterma alta… es obvio
> que puede haber remociones en masa. Pero en el sur hay lluvias persistentes,
> pero alta humedad en el suelo… las remociones en masa son pocas… Lo que allí
> pasa es desborde de los ríos.» — Alexis, 16-ago-2026

El terreno no modula el peligro: **lo encamina.** Consecuencia técnica y
comprobable: la humedad del suelo previa entra con **signo opuesto** en las dos
familias de eventos.

| amenaza | el suelo previo debería venir… |
|---|---|
| Remoción en masa | **más seco** |
| Desborde fluvial | **más húmedo** |

## 2 · El criterio, copiado tal cual del estudio

Del §6 de `ESTUDIO_VECTORES_DE_AMENAZA.md`, escrito **antes** de bajar un solo
dato de humedad:

> **Criterio, fijado ahora:** la diferencia de humedad previa entre las dos
> familias debe ser significativa contra un nulo de fechas barajadas dentro del
> mismo punto (p < 0,01), y el signo debe ser el predicho — **no basta con que
> difieran**.

Traducido a números, también antes de calcular nada:

```
ESTADÍSTICO   D = media(z humedad | remoción) − media(z humedad | inundación)
PREDICCIÓN    D < 0
NULO          10.000 barajadas de la FECHA de cada evento, dentro de su MISMO
              punto y su MISMO mes calendario
APRUEBA       p bilateral < 0,01  Y  D < 0.  Las dos cosas, o no pasa.
```

## 3 · Método

**Humedad.** ESA CCI Soil Moisture COMBINED v09.2, malla 0,25°, diaria, por
OPeNDAP anónimo del CEDA. En vez de pedir celda por celda se bajó **la caja
completa de Chile un día a la vez** (157 × 49 celdas, lat −17,4 a −56,4, lon
−76,9 a −64,9): una petición por día devuelve el país entero. **3.684 días
completos (1-dic-2014 a 31-dic-2024), cero fallos de descarga, 5,4 minutos.**

**Humedad previa por evento.** Máximo de la humedad diaria en los 30 días
**anteriores** al evento — el día del evento excluido, para que la lluvia que lo
detona no contamine la medida. El máximo y no el promedio es regla heredada de
Cosmoclima: la serie tiene huecos y el promedio los arrastra. Se exigen ≥8 días
con dato en la ventana; si no, el evento queda **sin dato** y no se rellena.

**Normalización por punto (z).** El norte árido y el sur húmedo viven en rangos
de humedad distintos: comparar crudo mediría geografía, no encaminamiento. Cada
evento se expresa como `z = (V − μ_celda) / σ_celda`, con μ y σ de su propia
celda sobre 2015-2024. Lo que se compara es «¿venía esta celda más seca o más
húmeda **que su propia costumbre**?».

**Familias.**

| familia | fuente | ubicación | n bruto |
|---|---|---|---|
| remoción (principal) | ReTeRM SERNAGEOMIN, sólo detonante meteorológico | punto medido en terreno | 376 |
| inundación | SENAPRED: clase/tipo/sub-evento nombra inundación, desborde, anegamiento, crecida o aumento de caudal | centroide de la comuna | 863 |
| remoción (control) | SENAPRED, mismos criterios de texto | centroide de la comuna | 398 |

Las 47 filas **mixtas** (nombran inundación *y* remoción a la vez) se
descartaron: no pertenecen a ninguna familia pura. Un par (comuna, fecha) es un
evento, no treinta partes. Los eventos que caen en la **misma celda satelital el
mismo día** se cuentan una vez: comparten literalmente el mismo valor de
humedad, contarlos dos veces infla la n y estrecha el nulo (119 apartados).

**Control simétrico.** La prueba principal compara dos catálogos distintos,
ubicados de dos maneras distintas. Para saber si un resultado viene del fenómeno
o de esa asimetría, se repite todo con **remoción de SENAPRED contra inundación
de SENAPRED**: misma fuente, misma forma de ubicar, mismo sesgo de registro.

**Privacidad.** Del Excel de SENAPRED se leyeron **sólo las columnas 1 a 10**
(fecha, región, provincia, comuna, origen, clase, tipo, sub-eventos). La columna
44, «Antecedentes Observaciones», que contiene RUT y descripciones de personas
fallecidas, **no se abrió en ningún momento**. El CSV entregado no contiene
ninguna variable de personas.

## 4 · Resultado

### 4.1 · Las tres pruebas

| prueba | n remoción | n inundación | z remoción | z inundación | **D** | nulo (media ± sd) | **p** | signo | veredicto |
|---|---|---|---|---|---|---|---|---|---|
| **PRINCIPAL** ReTeRM vs SENAPRED | 44 | 441 | **+1,018** | **+0,750** | **+0,268** | +0,208 ± 0,079 | **0,222** | ✗ al revés | **NO PASA** |
| **CONTROL SIMÉTRICO** SENAPRED vs SENAPRED | 129 | 441 | +0,365 | +0,750 | **−0,385** | −0,365 ± 0,059 | **0,368** | ✓ predicho | **NO PASA** |
| SENSIBILIDAD principal, nulo de mes libre | 44 | 441 | +1,018 | +0,750 | +0,268 | +0,001 ± 0,159 | 0,091 | ✗ al revés | **NO PASA** |

En humedad cruda (máximo de 30 días previos, m³/m³): principal 0,378 remoción
vs 0,341 inundación; simétrica 0,288 remoción vs 0,341 inundación.

### 4.2 · Los dos hallazgos que importan más que el veredicto

**★ (a) El nulo se come casi todo el efecto.** Ésta es la razón real del
fracaso, y es un resultado en sí mismo. La diferencia observada entre familias
es casi idéntica a la que produce **barajar las fechas al azar**:

```
PRINCIPAL   D observado = +0,268   ·   nulo = +0,208 ± 0,079   →   +0,76 sd
SIMÉTRICA   D observado = −0,385   ·   nulo = −0,365 ± 0,059   →   −0,34 sd
```

Dicho en simple: **una vez que sabes DÓNDE y EN QUÉ MES**, enterarte además de
que ese día hubo un evento no te dice casi nada sobre la humedad del suelo. Toda
la separación aparente entre las dos familias es *dónde ocurren* y *en qué meses
ocurren* — geografía y estación — y no encaminamiento.

Y no es que la prueba fuera ciega: con un nulo de sd 0,079, un apartamiento de
0,20 z habría dado p<0,01. Lo que se observó fue 0,076. **La prueba tenía ojos
para ver un efecto de ese tamaño y no lo vio.**

**★ (b) La humedad previa está ALTA antes de las dos familias.** Los dos z son
claramente positivos (+1,02 y +0,75; +0,37 en la remoción de SENAPRED). Antes de
una remoción **y** antes de un desborde, el suelo viene más húmedo que su
costumbre para ese mes en ese lugar. Con este dato, la humedad de suelo se
comporta como un **indicador de peligro compartido**, no como un enrutador que
manda a una amenaza o a la otra.

### 4.3 · La contradicción de signo, que es lo que invalida la prueba

Cambiar sólo la fuente de las remociones **da vuelta el signo**:

```
remociones de ReTeRM   (punto en terreno)      →  D = +0,268   remoción MÁS húmeda
remociones de SENAPRED (centroide comunal)     →  D = −0,385   remoción MÁS seca
```

Con la misma familia de inundaciones, el mismo satélite, la misma ventana y el
mismo estadístico. **El signo lo está decidiendo el catálogo, no el terreno.**
Se probó si era artefacto de la ventana temporal: con humedad **antecedente** de
[−60, −31] días (que por construcción no puede contener la borrasca detonante),
la contradicción se repite igual — D = +0,363 con ReTeRM, D = −0,264 con
SENAPRED. No es la ventana.

Mientras esa contradicción no se explique, cualquier lectura de la prueba
principal —a favor o en contra— sería leer el sesgo de un catálogo.

### 4.4 · Control por lluvia

Sólo hay lluvia diaria ERA5 en los 91 puntos de ReTeRM, así que el control se
restringe a eventos en comunas que tienen uno de esos puntos, usándolo como
representante de la lluvia comunal (aproximación declarada: la comuna es grande
y el punto es uno).

La lluvia **no** es igual entre familias en la muestra principal: P48 medio 99,3
mm antes de las remociones ReTeRM contra 47,6 mm antes de las inundaciones. Por
estratos de lluvia comparable, D sigue positivo en la principal (+0,40 / +0,45 /
+0,15 / +0,27, o sea al revés de lo predicho en los cuatro) y cambia de signo
según el estrato en la simétrica (−0,55 / +0,02 / −0,20 / −0,10). El control por
lluvia **no rescata la hipótesis** en ninguna de las dos versiones.

## 5 · Cobertura lograda

**Cubo de humedad: 100 %** de lo que existe — 3.684 de 3.684 días, sin fallos.
Chile completo por día, no una muestra de puntos.

**Eventos con humedad utilizable:**

| familia | usables / brutos | % |
|---|---|---|
| ReTeRM remoción | **51 / 376** | 13,6 % |
| SENAPRED inundación | **441 / 863** | 51,1 % |
| SENAPRED remoción | **129 / 398** | 32,4 % |

De los 51 de ReTeRM, **44** entran en la prueba principal (los otros 7 no tienen
detonante meteorológico declarado).

Por qué se perdió el resto de ReTeRM, en orden:

- **190 — fecha fuera de 2015-2024.** ESA CCI v09.2 **termina el 31-dic-2024**
  (verificado en el catálogo del CEDA: no hay directorio 2025). Los 51 eventos
  de 2025 y los 138 de 2026 —la mitad del catálogo ReTeRM— **no tienen humedad
  satelital que consultar**. Se declara sin dato; no se aproximó con nada.
- **105 — menos de 8 días con dato satelital** en la ventana de 30 días.
- **25 — duplicados** de celda y fecha.
- **5 — celda sin climatología** suficiente para normalizar.

**Rango geográfico cubierto: 152 celdas ESA CCI distintas, de latitud −18,4
(Arica, norte árido) a −48,4 (Aysén, sur húmedo).** El requisito de cubrir norte
árido y sur húmedo se cumple.

## 6 · Limitaciones declaradas

1. **El satélite está más ciego justo donde ocurren las remociones.** En las
   celdas de las remociones hay dato el 70-74 % de los días; en las de las
   inundaciones, el 82 %. Y en las celdas de los **694 eventos descartados** por
   falta de dato, sólo el **11 %**. La pérdida **no es al azar**: ESA CCI
   enmascara pendiente fuerte, nieve y vegetación densa, que es exactamente el
   terreno de una remoción en masa. La muestra de remociones que sobrevive está
   sesgada hacia el terreno más suave, que es el menos característico.
2. **La celda mide ~25 km de lado; una remoción ocurre en una ladera.** La celda
   promedia ladera, valle y fondo de cuenca en un solo número.
3. **SENAPRED no tiene coordenadas.** Se usó el centroide ponderado por área de
   la comuna. En una comuna cordillerana grande, ese centroide puede caer lejos
   del río que se desbordó. Es la mejor ubicación disponible con el dato público;
   no se inventó una más fina.
4. **Las dos familias vienen de catálogos que se construyen distinto.** ReTeRM
   es levantamiento técnico en terreno; SENAPRED es registro de emergencia
   atendida — sesgado hacia donde hay gente que llama. Éste es el sospechoso
   número uno de la contradicción de signo del §4.3.
5. **n = 44 en la familia principal de remociones.** Suficiente para descartar un
   efecto de 0,2 z (§4.2), no para descartar uno más pequeño.
6. **`Iso0`, `Pend`, `Veg` y `Reg` no entraron.** La receta del §4.2 del estudio
   es una conjunción de seis condiciones; esta prueba aisló una sola. Que la
   humedad sola no separe no falsa la receta completa — falsa la afirmación
   fuerte de que **la humedad sola encamina**.
7. **Al sur de −53° no hay dato ESA CCI** en ninguna celda chilena. Magallanes
   queda fuera por completo.

## 7 · Qué queda sin resolver, y qué haría falta

1. **Explicar la contradicción de signo** antes que cualquier otra cosa. La vía
   más directa: geolocalizar las inundaciones con coordenada real (no centroide
   comunal) y repetir con las dos familias ubicadas igual. Si el signo se
   estabiliza, hay pregunta; si sigue dependiendo del catálogo, la pregunta no se
   puede responder con estos registros.
2. **Conseguir humedad 2025-2026** para recuperar la mitad de ReTeRM que hoy no
   tiene con qué compararse (v09.2 termina en 2024). Alternativas por verificar:
   ERA5-Land `volumetric_soil_water_layer_1` del CDS de Copernicus —exige
   registro de Alexis— o SMAP L3 de la NASA, ambos con latencia de días.
3. **Medir humedad a escala de ladera**, no de 25 km. Sin eso, la variable que
   la hipótesis necesita y la que el satélite entrega no son la misma variable.
4. **Probar la receta completa y no una condición suelta.** El enrutamiento que
   describe Alexis puede vivir en la *interacción* humedad × pendiente ×
   regolito, y esta prueba, por diseño, no podía verla.

**Lo que este resultado NO autoriza a concluir:** que la observación de Alexis
sea falsa. Lo que quedó falsado es la versión operativa más simple —«la humedad
de suelo satelital, sola, a 25 km y en el mes previo, separa las dos familias»—.
La observación de campo que la originó sigue en pie y sin poner a prueba de
verdad, porque el instrumento disponible no tiene la resolución que ella exige.

---

## 8 · Reproducir

```bash
.venv-esa/bin/python infraestructura/probar_humedad_enrutador.py --etapa bajar     # ~5 min
.venv-esa/bin/python infraestructura/probar_humedad_enrutador.py --etapa analizar   # ~2 min
```

Semilla fija `20260816`; los números de este informe se reproducen exactos.

**Archivos**

- `infraestructura/probar_humedad_enrutador.py` — el script.
- `infraestructura/datos/humedad_eventos.csv` — 1.637 eventos con su humedad
  previa, su z, si entró en la prueba y, si no entró, **por qué**.
- Cubo crudo ESA CCI de Chile (3.684 × 157 × 49, 113 MB): queda en el
  scratchpad de la sesión, no en el proyecto, para no chocar con otros agentes.
  Ruta en la constante `CACHE` del script; se puede mover a
  `infraestructura/datos/crudo/esacci/` si se quiere conservar la historia.
