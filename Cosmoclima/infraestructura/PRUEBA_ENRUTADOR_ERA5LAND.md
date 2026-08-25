> # ⚠️ ADVERTENCIA AÑADIDA EL 17-AGO-2026, ANTES DE QUE HAYA RESULTADOS
>
> **La clasificación de familias que esta prueba reutiliza NO SIRVE, y hay que
> resolverlo antes de leer cualquier número que produzca.**
>
> Verificado sobre el registro crudo: el **aluvión de Copiapó del 24-mar-2015**
> —22.111 afectados, 30 aislados, el caso ancla del proyecto— está en SENAPRED
> clasificado como
> `Clase: Precipitaciones · Tipo: Núcleo Frío en Altura · Sub evento: **Inundación**`.
>
> Es decir: **SENAPRED llama «Inundación» al aluvión**, que es justo el caso que
> la hipótesis usa como ejemplo de la familia CONTRARIA (suelo seco, flujo de
> detritos). Y en 50.457 eventos la palabra «aluvión» aparece **25 veces**,
> repartida entre cinco etiquetas distintas: `Aluvión`, `Sistema Frontal`,
> `Núcleo Frío en Altura`, `Flujo de Barro/Detritos (Aluvión)` y
> `Remoción en Masa`.
>
> **Consecuencia:** usar SENAPRED como clasificador de familias mete el caso
> ancla en la familia equivocada. El vigía va a disparar solo cuando el cupo se
> libere y va a producir un D y un p con estas etiquetas. **Esos números NO son
> el veredicto de la hipótesis** — miden un contraste entre dos bolsas mal
> armadas. Sirven como ensayo del programa, nada más.
>
> **Qué corresponde antes de dar veredicto:** para remociones, usar **ReTeRM**
> (SERNAGEOMIN, evaluado en terreno). Para la familia de desborde, buscar otra
> fuente — probablemente fluviométrica de la **DGA**. Y decidirlo con Alexis,
> porque cambia el conjunto de examen.
>
> *(El dato de humedad que baje el vigía sí sirve: es reutilizable con cualquier
> clasificación. Lo que no sirve es el contraste calculado sobre estas bolsas.)*

---

# Prueba del «enrutador de amenazas», segunda vuelta: humedad ERA5-Land

**16-ago-2026.** Repetición de `PRUEBA_HUMEDAD_ENRUTADOR.md` con un instrumento
mejor. Cambia **sólo la fuente de humedad**: la clasificación de los 1.637
eventos en familias se reutiliza tal cual de `datos/humedad_eventos.csv`, que ya
venía verificada.

# ► VEREDICTO: **NO PASA** (corrido el 19-ago-2026)

La captura quedó bloqueada el 16-ago por el cupo diario de Open-Meteo; el vigía
la completó la madrugada del 17 y el análisis se corrió el 19. **El criterio de
§1 no se tocó en ningún momento.**

**Resultado corto:** la prueba principal falla **por el signo** —la humedad
antecedente resulta *mayor* antes de las remociones, no menor— y falla
significativamente (D=+0,0988, p=0,0002). La replicación da el signo predicho
pero con efecto **cero** (D=−0,0086, p=0,23; en crudo 0,3828 vs 0,3830 m³/m³).
Los controles por lluvia y por región no la rescatan, y **el poder estaba**: la
prueba detectaba lo que necesitaba detectar.

**El detalle completo, con los controles y lo que este resultado NO falsa, está
al final del documento** (sección «RESULTADO — corrido el 19-ago-2026»).

---

## 1 · Criterio fijado antes

**Esta sección se escribió y se guardó ANTES de calcular un solo número con el
instrumento nuevo.** Está codificada en las constantes `CRITERIO_*` de
`probar_enrutador_era5land.py`, y sigue intacta.

### 1.1 · La predicción

> La humedad del suelo **antecedente** debe ser **MENOR** en los eventos de
> remoción en masa que en los eventos de inundación.

Porque el terreno no modula el peligro: lo **encamina**. Suelo seco y regolito
suelto → la lluvia arranca la ladera. Suelo ya saturado → la lluvia no infiltra,
escurre y desborda el cauce.

### 1.2 · El estadístico y el umbral

```
VARIABLE      humedad de suelo ERA5-Land, capa 7-28 cm (condición del terreno)
VENTANA       promedio de los días −30 a −3 respecto de la fecha del evento
              (el día del evento NO entra: si entrara se estaría midiendo la
               lluvia que causó el evento y la prueba sería circular)
z             (V − μ del punto) / σ del punto: «¿venía este punto más seco o
              más húmedo QUE SU PROPIA COSTUMBRE?»
ESTADÍSTICO   D = media(z | remoción) − media(z | inundación)
PREDICCIÓN    D < 0
NULO          10.000 barajadas de la FECHA de cada evento, dentro de su MISMO
              punto y su MISMO mes calendario
p             bilateral de permutación: 2 × min(P(nulo ≤ D), P(nulo ≥ D))
APRUEBA       D < 0  Y  p < 0,01.  Las dos, o NO PASA.
```

**Si falla, se reporta el fallo.** No se cambia el criterio después, no se
prueban variantes hasta que una acierte, no se ajusta ningún umbral.

### 1.3 · Qué es la prueba principal y qué es la replicación

El intento anterior mezcló catálogos —ReTeRM ubica con coordenada medida en
terreno, SENAPRED sólo con centroide comunal— y **eso solo dio vuelta el signo**.
Por eso, esta vez:

| | catálogos | ubicación | papel |
|---|---|---|---|
| **PRINCIPAL** | SENAPRED remoción vs SENAPRED inundación | centroide comunal en las dos | **la prueba** |
| **SECUNDARIA** | ReTeRM remoción vs SENAPRED inundación | terreno vs centroide | **replicación**, se reporta aparte |

Se reportan por separado y **no se mezclan**. El veredicto del criterio lo
dictamina la principal.

### 1.4 · Casos quemados, fuera del examen

Estos dos ya fueron mirados con **este mismo instrumento** antes de la prueba.
Eso fue diagnóstico, no prueba, y por la Regla 2 del banco no pueden formar
parte del examen:

- **el aluvión de Copiapó de marzo de 2015** (región de Atacama, 20-mar a
  5-abr-2015);
- **el temporal de julio de 2026** (todo el mes).

Quedan **excluidos del conjunto de examen** y sólo pueden aparecer, marcados
como tal, en una sección de ilustración.

> **Hallazgo colateral, ya comprobado:** del aluvión de Copiapó de marzo 2015
> **no hay ni un evento** en `humedad_eventos.csv`. Los eventos de Atacama de
> esa fecha no quedaron clasificados en ninguna de las dos familias por el
> criterio de texto que se usó (SENAPRED los rotula «aluvión», palabra que no
> entra ni en la familia inundación ni en remoción). O sea: **el caso ancla del
> proyecto no está en el conjunto de datos con que se prueba la hipótesis.**
> Eso no lo causó esta prueba, pero conviene saberlo. Del temporal de julio 2026
> sí hay 101 eventos ReTeRM, y quedan excluidos.

### 1.5 · Controles obligatorios, también fijados antes

1. **Por lluvia.** La diferencia no puede ser «llovió más». Se comparan las dos
   familias **dentro de cuartiles de `P48` comparable** (mayor acumulado de 48 h
   que toca al evento; aquí sí entra el día del evento, porque esto no es la
   variable puesta a prueba sino el control).
2. **Por región.** El efecto no puede ser «el sur es más húmedo». Se compara
   **dentro de cada región** y se promedia ponderado.
3. **Poder.** Se reporta qué tamaño de efecto habría sido detectable a p < 0,01
   con el n efectivamente logrado.

### 1.6 · Secundarias declaradas como descriptivas

La capa **0-7 cm** se reporta, pero **no decide nada**: la variable
pre-registrada es la de 7-28 cm.

---

## 2 · El instrumento y por qué debería ser mejor

**ERA5-Land vía la API de archivo de Open-Meteo**, malla de 0,1° (~9 km),
anónima, diaria, 1950 → hoy.

Los tres defectos que hundieron el intento anterior con ESA CCI, y qué pasa con
cada uno:

| defecto de ESA CCI | ERA5-Land |
|---|---|
| 235 de 398 eventos de remoción **sin ni un día** de dato en los 30 previos (el satélite enmascara pendiente fuerte, nieve y vegetación densa: justo el terreno de una remoción) | reanálisis: **no tiene huecos ni máscara** |
| píxel de 25 km | **~9 km** |
| serie termina el 31-dic-2024 (deja fuera medio ReTeRM) | **llega a 2026** |

**Lo que ERA5-Land no arregla:** sigue sin ser humedad de ladera. 9 km es mejor
que 25, pero una remoción ocurre en una ladera de cientos de metros. Y sigue
siendo un **modelo**, no una medición: ERA5-Land calcula la humedad con un
esquema de suelo alimentado por la lluvia del reanálisis. Donde ESA CCI erraba
por ceguera, ERA5-Land puede errar por suposición.

---

## 3 · Por qué no hay resultado: la aritmética del cupo

Este es el hallazgo operativo de la jornada, y corrige la lección heredada.

**Open-Meteo tiene DOS límites distintos y los dos muerden:**

| límite | mensaje | qué lo dispara |
|---|---|---|
| simultaneidad | `Too many concurrent requests` | 4-8 peticiones en vuelo a la vez. El cerrojo tarda ~2 min en soltarse |
| **peso** | `Hourly / Daily API request limit exceeded` | una petición **no cuenta como una**: cuenta como `puntos × (días/14) × (variables/10)` |

El cupo gratuito es de **10.000 «llamadas» al día** y 5.000 por hora. Una sola
petición de 50 puntos × 4 años × 3 variables pesa **≈ 1.566 llamadas**: un
tercio del cupo de una hora.

**Qué se gastó y en qué.** El primer diseño —bajar la serie diaria completa
2015-2026 de cada uno de los 403 puntos— pesaba **≈ 36.700 llamadas**, casi
cuatro veces el cupo del día. No se supo hasta chocar: la API primero devolvió
lentitud y tiempos de espera agotados, después `Too many concurrent requests`, y
sólo al final el mensaje de peso. Entre las pruebas de escalado y los dos
arranques del diseño malo se consumieron las ~10.000 del día. Desde las 23:17
la API responde `Daily API request limit exceeded. Please try again tomorrow.`
para cualquier petición, aunque sea de tres días y una variable.

**El diseño corregido cabe holgado.** No hace falta la serie completa de cada
punto: basta lo que la prueba usa.

| bloque | qué pide | peso |
|---|---|---|
| **A · ventanas** | para cada (año, mes) con eventos, los puntos que tuvieron evento ese mes, desde 33 días antes del día 1 hasta fin de mes, 3 variables | **1.843** |
| **B · climatología 7-28** | para cada mes calendario, los puntos que alguna vez tuvieron evento en ese mes, mismo tramo, 6 años (2019-2024), 1 variable | **2.676** |
| **C · climatología 0-7** | idem | **2.676** |
| | **total** | **7.195 de 10.000** |

392 peticiones, una a la vez. **Cabe en un día.** No cabía el diseño anterior, y
ése es exactamente el error que costó la jornada.

---

## 4 · Lo que sí quedó verificado

El programa completo se corrió **de punta a punta con un crudo sintético** que
imita byte a byte la forma de la respuesta de Open-Meteo. No valida ningún
resultado —los valores eran inventados a propósito— pero sí valida la máquina:

- **0 filas con el punto mal asignado.** Cuando se piden muchos puntos en una
  petición, la respuesta trae la coordenada de la malla, no la pedida; el
  manifiesto amarra cada posición de la respuesta a su punto. Verificado.
- **La urna del nulo queda con mediana de 186 fechas candidatas por evento**
  (mismo punto, mismo mes, 6 años). El intento anterior trabajaba con ~300; es
  del mismo orden.
- **El conjunto de examen que va a quedar**, que ya es dato real porque no
  depende de los valores de humedad sino de la geometría del plan:

| prueba | n remoción | n inundación | contra el intento anterior |
|---|---|---|---|
| **PRINCIPAL** SENAPRED vs SENAPRED | **393** | **845** | 129 vs 441 → **×3,0 y ×1,9** |
| **SECUNDARIA** ReTeRM vs SENAPRED | **175** | **845** | 44 vs 441 → **×4,0 y ×1,9** |

Exclusiones del examen: 143 duplicados de punto-fecha-familia, 101 eventos del
temporal de julio 2026 (**caso quemado**), 5 punto-fecha que aparecen en las dos
familias a la vez, 1 evento anterior a 2015 y las 48 remociones ReTeRM sin
detonante meteorológico declarado (sólo afectan a la secundaria).

**Ese salto de n es la mitad del argumento de esta segunda vuelta.** La otra
mitad —si la humedad separa o no las dos familias— sigue sin respuesta.

---

## 5 · Cómo se termina

Ya está andando: un vigía consulta la API cada 5 minutos y, en cuanto el cupo se
libere (Open-Meteo reinicia el contador diario a medianoche UTC, o sea a las
20:00 de Chile), **dispara solo** la captura y el análisis, y rellena el CSV.

> **Ojo:** el vigía rellena el CSV y deja el crudo, pero **este informe no se
> actualiza solo**. Los números —D, p, controles, poder— hay que traerlos de la
> corrida y escribirlos aquí, en la sección de resultados que hoy no existe.

Si hubiera que hacerlo a mano:

```bash
.venv-esa/bin/python infraestructura/probar_enrutador_era5land.py --etapa bajar     # ~30 min
.venv-esa/bin/python infraestructura/probar_enrutador_era5land.py --etapa analizar   # ~3 min
```

La etapa `bajar` es **reanudable**: lo que ya está en `datos/crudo/era5land/`
no se vuelve a pedir, y cada archivo se valida contra el manifiesto antes de
darlo por bueno. Semilla fija `20260816`.

**Cuando corra, el criterio del §1 no se toca.** Salga lo que salga.

---

## 6 · Entregables

- `infraestructura/probar_enrutador_era5land.py` — el programa. Dos etapas,
  `bajar` y `analizar`.
- `infraestructura/datos/humedad_era5land_eventos.csv` — hoy trae las 1.637
  filas del conjunto de examen (familia, fecha, comuna, punto de 0,1°,
  precisión de la ubicación, si es caso quemado) con las columnas de humedad
  **vacías** y el motivo declarado en `motivo_sin_dato`. Cuando la captura
  corra, el mismo archivo se rellena con la humedad, el z, la lluvia y qué
  prueba usó cada evento.
- `infraestructura/datos/crudo/era5land/2026-08-16/manifiesto.json` — el plan
  de captura completo: qué puntos, qué días y qué variables lleva cada
  petición. Es lo que amarra cada dato a su punto.
- Este informe.

---

## 7 · Lo que este resultado NO autoriza a concluir

**Nada sobre la hipótesis.** No es un «no pasa»: es un «no se midió». La
afirmación de Alexis —que el terreno encamina la amenaza y que la humedad entra
con signo opuesto en las dos familias— sigue exactamente donde estaba después
del intento con ESA CCI: sin poner a prueba con un instrumento que tenga la
resolución y la cobertura que ella exige.

---

# RESULTADO — corrido el 19-ago-2026

El vigía bajó el dato la madrugada del 17 y la etapa de análisis se corrió el 19.
El criterio de §1 **no se tocó**.

## ►► NO PASA — y esta vez el instrumento sí podía ver

| prueba | n remoción | n inundación | D | nulo | p | signo predicho |
|---|---|---|---|---|---|---|
| **PRINCIPAL** SENAPRED vs SENAPRED | 388 | 835 | **+0,0988** | −0,1341 ± 0,0375 (+6,20 sd) | 0,0002 | **NO** |
| SECUNDARIA ReTeRM vs SENAPRED | 172 | 835 | −0,0086 | −0,0893 ± 0,0671 (+1,20 sd) | 0,2338 | sí |
| SECUNDARIA capa 0-7 cm | 172 | 835 | −0,0001 | −0,0751 ± 0,0673 (+1,11 sd) | 0,2640 | sí |

**La principal falla por el signo, y falla significativamente**: la humedad
antecedente es *mayor* antes de las remociones, no menor. La secundaria da el
signo predicho pero el efecto es **cero**: en crudo, 0,3828 contra 0,3830 m³/m³.

## Los controles no rescatan la hipótesis

- **Por lluvia comparable:** D sigue positivo en 3 de los 4 estratos de la
  principal (+0,358 · +0,218 · +0,165) y sólo se vuelve negativo en el estrato
  más lluvioso (−0,168). No es un artefacto de «llovió más».
- **Por región:** D intra-región ponderado = **+0,1151** sobre 12 regiones. Sigue
  positivo. La dispersión entre regiones es enorme (Valparaíso +2,469, Tarapacá
  −1,147), lo que sugiere que la señal regional domina sobre la de familia.
- **Poder:** la principal detectaba |D−nulo| ≥ 0,098 z a p<0,01, y observó 6,20
  sd de apartamiento. **La prueba tenía con qué ver el efecto si existiera.**

## Lo que esto NO falsa

Tres cosas hay que dejar escritas para no sobreinterpretar:

1. **La receta de remoción del estudio tiene SEIS condiciones** (lluvia, humedad,
   pendiente, vegetación, regolito, isoterma). Esta prueba aisló **una**.
2. **Las etiquetas de familia son el sospechoso número uno.** La prueba principal
   compara dos bolsas armadas con etiquetas de SENAPRED, y ya está medido que
   SENAPRED clasifica el aluvión de Copiapó como «Inundación». Comparar bolsas
   parcialmente mal rotuladas puede producir exactamente este resultado.
3. **La climatología del z quedó corta**: sólo 208 de las 392 peticiones del plan
   llegaron a bajarse, y la base de referencia cubre **2022-2024** (7-28 cm) y
   2023-2024 (0-7 cm). Es poco para una normal. *Pero el resultado no depende de
   eso*: en valores crudos las dos familias son indistinguibles.

## Ilustración, fuera del examen

El temporal de julio de 2026 (n=101, excluido por quemado) llegó con el suelo
**seco**: humedad antecedente media 0,2729 m³/m³, z = **−0,604**, con 160,8 mm
de P48. Coincide con la intuición de campo, y por eso mismo no puede usarse como
prueba.

## Qué corresponde ahora

**No relanzar la medición.** Corresponde resolver primero **de dónde salen las
etiquetas de familia** — es lo que quedó pendiente el 17-ago y lo que esta prueba
vuelve urgente. Medir mejor sobre bolsas mal armadas produce números más
precisos de una comparación equivocada.
