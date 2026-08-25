# Variables y métricas propias del proyecto Infraestructura Crítica × Clima

**Regla de trabajo (Alexis, 16-ago-2026):** todo se desarrolla **acá**, dentro del
proyecto. Cuando esté validado se traspasa a los documentos oficiales del RMD —
Word primero, Excel después. Mientras tanto **no se toca el canon**.

Por eso cada ficha de este documento está escrita en el **formato canónico de la
sección 22 del Word** (los once campos de una ficha del anexo, verificados contra
la ficha de IVIC). Cuando llegue el momento del traspaso será copiar y pegar, no
reescribir.

Se agregan dos campos que el Word no tiene y el Excel sí: **Estado y validación**
y **Limitaciones conocidas**. Ningún número de este documento es una promesa: o
está medido y dice contra qué, o está marcado como pendiente.

**Siglas verificadas libres** contra las 348 en uso entre el Word (305 fichas) y
el Excel oficial (318 filas), el 16-ago-2026.

---

## Estado del conjunto

| Sigla | Variable | Estado |
|---|---|---|
| `PelPre` | Peligro de Precipitación | **implementada y validada** contra 328 eventos de terreno |
| `CClimP` | Coeficiente Climático por Precipitación | **propuesta** — regla declarada acá, sin calcular todavía |
| `FENef` | Fragilidad ante Eventos Naturales Efectiva | **propuesta** — depende de `CClimP` |
| `PFef` | Ponderación Final Efectiva | **propuesta y bloqueada** por H-01 |
| `CondTerr` | Condición del Terreno | **pendiente a propósito** — ver por qué |
| `TRec` | Tiempo de Recuperación | **pendiente** — hueco H-11, sin dato aún |
| `USCom` `USProv` | Unidad de Sistema comunal y provincial | **extensión del canon**, no variable nueva |

---

# 1 · PelPre — Peligro de Precipitación

**Estado: IMPLEMENTADA Y VALIDADA.** `normalizar.peligro()` ·
`normalizar.razon_contra_normal()` · `adaptadores/era5.py`

### Fórmula Numérica/Algebraica

```
PelPre = √( MagNac × ExcLoc )
```

### Donde

- **`P48`**: mayor lluvia acumulada en 48 horas dentro del mes, en milímetros,
  en el punto del activo. Es el evento del mes, no el total mensual: un temporal
  de dos días es lo que corta un camino, no que haya llovido repartido.
- **`MagNac`** (0 a 1): percentil de `P48` contra la distribución de **todo el
  país** (17.160 pares activo-mes, 1990-2026, excluidos los meses sin lluvia).
  Responde *¿es mucha agua en términos absolutos?*
- **`NormAnual`**: lluvia normal de un año en ese punto — suma de las doce
  normales mensuales del período 1991-2020.
- **`R = P48 / máx(NormAnual, 5 mm)`**: cuántos años de lluvia normal cayeron de
  golpe. El piso de 5 mm impide que un punto de normal casi nula produzca
  razones infinitas por un milímetro.
- **`ExcLoc`** (0 a 1): percentil de `R` contra la distribución nacional de
  razones. Responde *¿supera lo que este lugar aguanta?*

### Rango

Normalizado entre 0 y 1. 1 indica agua mucha en términos absolutos **y** muy por
encima de lo que ese lugar recibe.

### Definición

`PelPre` mide el peligro que la lluvia de un mes representa para la
infraestructura de un punto, como **conjunción** de dos condiciones
independientes: magnitud absoluta y excedencia respecto de la climatología
local. Se usa la media geométrica y no un promedio porque es una conjunción: si
cualquiera de las dos falta, no hay peligro. Un promedio dejaría que 8 mm
rarísimos en el desierto dieran «medio peligro», y no es medio peligro: es
ninguno.

### Pertinencia y Coherencia

La forma no es una elección libre: el propio canon resuelve las conjunciones con
raíz de producto — `ICSGS = √(FCN×FSS×FAS×FPI)`. Y las dos condiciones ya
estaban separadas en MACLIMA y las habíamos colapsado en una sola: `ANPrecip` es
la anomalía local, `InEvExtre` es el evento con su factor de exposición. La
definición canónica de aluvión (`EAL`) exige «precipitación intensa **más**
condición de suelo». `PelPre` implementa las dos primeras piezas; la tercera
está declarada como `CondTerr` y **no** está implementada.

### Descripción para un Lector No Técnico

Para que la lluvia haga daño tienen que pasar dos cosas al mismo tiempo: que sea
**mucha** agua, y que sea **más de la que ese lugar está hecho para aguantar**.
Ciento cuatro milímetros en dos días son peligrosos en cualquier parte del
mundo; ocho milímetros no lo son en ninguna. Pero trescientos milímetros en
Valdivia son un martes cualquiera, porque la ciudad tiene drenaje, alcantarillado
y techos para eso — y en Copiapó serían una catástrofe. No es que la lluvia sea
distinta: es que la ciudad no está hecha para ella. `PelPre` exige las dos cosas
juntas, y si falta una da cero.

### Fuentes Utilizadas para la Medición

- **ERA5 vía Open-Meteo Archive** (anónimo, 1990-2026, diario): precipitación
  por punto. Único con cobertura continua para 35 grados de latitud.
- **Normales 1991-2020** calculadas de la misma serie, por mes calendario.
- **ReTeRM de SERNAGEOMIN** (376 eventos de remoción en masa evaluados en
  terreno, 1996-2026): conjunto de validación independiente.
- **NASA POWER: descartada.** Para Copiapó el 24-25 de marzo de 2015 entrega
  2,7 y 6,6 mm donde ERA5 da 39,8 y 64,3 mm — borra el evento. No sirve para
  extremos en zona árida.

### Ejemplo Genérico

Copiapó, marzo de 2015: el aluvión que destruyó la ciudad. Cayeron **104,1 mm en
48 horas** sobre un punto cuya lluvia normal de un año es de 33 mm.

### Cálculo

```
P48       = 104,1 mm
MagNac    = 0,978                      (percentil nacional de 104,1 mm)
NormAnual = 33 mm
R         = 104,1 / 33 = 3,12          (3,12 años de lluvia de golpe)
ExcLoc    = 1,000                      (percentil nacional de esa razón)

PelPre    = √(0,978 × 1,000) = 0,9888
```

### Normalizado

`PelPre = 0,9888` — **percentil 99,81 %** del país en 36 años.

Los dos contraejemplos, el mismo mes y el mismo lugar en un mes tranquilo:

| caso | mm/48 h | R | PelPre | percentil |
|---|---|---|---|---|
| Copiapó, marzo 2015 (aluvión) | 104,1 | 3,12 | **0,9888** | 99,81 % |
| Punta Arenas, marzo 2015 | 22,9 | 0,03 | 0,5164 | 62,56 % |
| Copiapó, agosto 2015 (tranquilo) | 11,8 | 0,35 | 0,5805 | 68,56 % |

### Estado y validación

**Ancla (criterio fijado antes de calcular, en `CORRECCION_RAREZA_PELIGRO.md`):
los cuatro criterios se cumplen.** Copiapó en el 1 % superior; Punta Arenas y el
mes tranquilo de Copiapó fuera del 10 % superior; el desastre por encima del mes
tranquilo.

**Validación independiente:** 328 de 329 eventos ReTeRM detonados por lluvia,
ninguno de los cuales participó en el diseño.

| prueba | resultado | azar |
|---|---|---|
| posición mediana en la historia del propio punto | 98,0 % | 50 % |
| eventos en el decil superior | 72,0 % | 10 % |
| NULL-1 (meses al azar del mismo punto) | 10,0 % | 10 % |
| NULL-2 (eventos atribuidos a otros puntos) | 52,8 % | — |

**¿Aporta la normalización, o bastaba con los milímetros?** AUC sobre 152
meses-con-evento contra 39.888 sin evento, 91 puntos:

| | AUC |
|---|---|
| `PelPre` | **0,8370** |
| milímetros crudos en 48 h | 0,8197 |

**Señal no buscada:** de los ocho meses más peligrosos de 36 años que encuentra
el instrumento, **cuatro son julio de 2026** — el temporal que motivó el
proyecto. Nadie se lo dijo; lo encontró solo en la serie completa.

### Limitaciones conocidas

1. **En zonas secas no supera a los milímetros crudos.** AUC 0,8805 contra
   0,8810. Toda la ganancia está en los lugares húmedos (0,7847 contra 0,7580),
   que es **al revés** de lo que motivó la corrección.
2. **La mitad de la señal es «mes de temporal nacional»**, no destreza del
   lugar: el nulo de eventos atribuidos a otros puntos da 52,8 %.
3. **El conjunto de validación no puede probar el caso árido.** ReTeRM está
   dominado por el sur húmedo (Biobío 102, Los Ríos 56); el norte árido aporta
   23 de 329 eventos, el 7 %. SERNAGEOMIN registra donde va a mirar.
4. **Circularidad parcial:** 329 de 376 eventos tienen «lluvia» como detonante
   asignado por SERNAGEOMIN, así que «hay deslizamientos en meses lluviosos» es
   en parte tautológico. Lo que sí se establece es que la medida los ranquea en
   el decil alto de su lugar; lo que no, que prediga mejor que «llovió fuerte».
5. **Falta la condición del terreno** (`CondTerr`). Sin ella, en el rango alto
   la conjunción queda gobernada por la magnitud, y un temporal grande en zona
   húmeda le gana a uno mediano en zona árida.
6. **Hereda el límite de ERA5**: en la ronda 17 de Cosmoclima se comprobó que
   exagera los años secos en la zona de Illapel. Se usa igual porque no hay
   estaciones para 39 puntos en 35 grados de latitud, y en consecuencia el
   piloto prueba **si el método funciona**, no cuánto riesgo corre cada activo.

---

# 2 · CClimP — Coeficiente Climático por Precipitación

**Estado: PROPUESTA. La regla se declara acá, ANTES de calcular nada con ella.**

Ésta es la pieza que falta para que el proyecto tenga efecto sobre la matriz.
`PelPre` mide; `CClimP` traduce esa medida a la perilla que el canon ya tiene.

### Fórmula Numérica/Algebraica

```
              ⎧ 1,0   si  PelPre <  0,6501          (bajo el P75 nacional)
              ⎪ 1,2   si  0,6501 ≤ PelPre < 0,8292  (P75 – P90)
CClimP  =     ⎨ 1,4   si  0,8292 ≤ PelPre < 0,9658  (P90 – P99)
              ⎪ 1,6   si  PelPre ≥ 0,9658           (P99 y superior)
              ⎩ 1,0   forzado, si la confianza del dato es baja
```

### Donde

- Los **cortes son percentiles medidos**, no elegidos: salen de la distribución
  real de `PelPre` sobre los 17.160 pares activo-mes de 1990-2026.
- Los **valores del coeficiente son los del canon**, no inventados: MACC define
  0,8–1,2 ajuste leve, 1,2–1,4 moderado, 1,4–1,6 alto, y **1,0 = neutro**.
- El **bloqueo por confianza baja** también es del canon: MACC lo contempla
  explícitamente («no usar coeficiente si la confianza del dato es baja»). Se
  implementa con `normalizar.confianza()`.

### Rango

De 1,0 a 1,6.

**★ Decisión declarada: no se atenúa por debajo de 1,0.** MACC permite bajar
hasta 0,8, y aquí no se usa. Un mes seco no vuelve más resistente a un puente:
sólo no lo exige. Atenuar afirmaría que el activo es *menos frágil*, y eso no lo
podemos sostener con ningún dato. Si más adelante se quiere atenuación, tendrá
que justificarse por separado y validarse aparte.

### Definición

`CClimP` es la celda `Cᵢⱼ` de la Matriz de Coeficiente Climático (MACC /
MatCoefClim) correspondiente a la fila «elementos de la MICR» y la columna
«precipitación». Traduce el estado climático medido del punto y el mes a un
multiplicador auditable, sin reescribir ninguna fórmula del RMD.

### Pertinencia y Coherencia

MACC es explícitamente una estructura de ponderación exógena, tipo *overlay*,
que se aplica **después** de medir: «no es un índice nuevo que compita con los
índices del RMD». `CClimP` respeta eso — no toca la definición de FEN, la
modula. Y respeta la trazabilidad que el canon exige de cada coeficiente
(dato → regla → justificación → variable afectada): el dato es `P48`, la regla
está escrita arriba, la justificación es esta ficha, y la variable afectada es
`FEN`.

### Descripción para un Lector No Técnico

Es la perilla. El modelo mide primero la infraestructura como siempre; después
`PelPre` mira el clima de ese lugar y ese mes; y `CClimP` responde una sola
pregunta: *¿cuánto hay que subirle el volumen a la fragilidad de este activo,
este mes?* Si el mes es normal, la perilla queda en 1 y no cambia nada. Si el
mes está entre el 1 % peor del país en 36 años, la sube a 1,6.

### Fuentes Utilizadas para la Medición

- `PelPre` de esta misma ficha.
- **MACLIMA, sección MatCoefClim**: rangos del coeficiente, regla del neutro,
  regla de bloqueo por confianza.
- **Distribución nacional de `PelPre`**: 17.160 pares, `adaptadores/era5.py`.

### Ejemplo Genérico

Copiapó, marzo de 2015. `PelPre = 0,9888`, por encima del corte de 0,9658.

### Cálculo

```
PelPre = 0,9888  ≥  0,9658   →   CClimP = 1,6
```

Punta Arenas ese mismo mes: `PelPre = 0,5164`, bajo 0,6501 → `CClimP = 1,0`.
La perilla no se mueve a 2.900 km del evento.

### Normalizado

`CClimP = 1,6` (Copiapó, marzo 2015) · `CClimP = 1,0` (Punta Arenas, marzo 2015).

### Estado y validación

**Sin validar. Criterio de aprobación, fijado ahora:**

1. En un mes cualquiera (mediana del país), `CClimP` debe valer 1,0 en **más del
   70 %** de los pares activo-mes. Una perilla que se mueve siempre no informa.
2. En los meses de los eventos ReTeRM, `CClimP` debe ser > 1,0 en **más de la
   mitad** de los casos.
3. El ordenamiento de activos que produce `FENef` debe cambiar respecto del
   orden fijo de la matriz en al menos un mes de temporal documentado, y **no**
   cambiar en un mes tranquilo.

Se aprueba sólo si se cumplen las tres. Si falla alguna, se reporta el fallo y no
se ajustan los cortes para que pase.

### Limitaciones conocidas

- Los cortes salen de la distribución de **39 subestaciones**, que son muestra y
  no inventario (H-10). Al ampliar el universo habrá que recalcularlos, y eso
  hay que declararlo como recálculo, no como corrección.
- Los escalones producen saltos: un activo con `PelPre = 0,8290` y otro con
  `0,8295` reciben 1,2 y 1,4. Es el precio de usar las bandas del canon en vez
  de una función continua. Si molesta en el uso real, la alternativa continua se
  evalúa aparte, con su propia validación.

---

# 3 · FENef — Fragilidad ante Eventos Naturales, Efectiva

**Estado: PROPUESTA. Es la variable central del proyecto.**

### Fórmula Numérica/Algebraica

```
FENef(activo, lugar, mes) = mín( 1 ; FEN₄(tipo) × CClimP(lugar, mes) )
```

### Donde

- **`FEN₄(tipo)`**: la fragilidad base del tipo de elemento, en la escala de
  **cuatro** niveles adoptada el 16-ago-2026:

  | nivel | valor |
  |---|---|
  | Baja | 0,119 |
  | Moderada | 0,339 |
  | Alta | 0,661 |
  | Muy Alta | **0,881** |

- **`CClimP`**: el coeficiente de la ficha anterior, de 1,0 a 1,6.

### Rango

De 0,119 a 1,000.

### Definición

`FENef` es el `FEN` de la MICR dejado de ser una etiqueta fija. Hoy `FEN` es una
etiqueta por tipo de elemento, sin territorio y sin tiempo: una autopista tiene
la misma fragilidad en Arica que en Aysén, en año seco que en año de temporal.
`FENef` conserva esa fragilidad estructural como base y le aplica el estado
climático real del punto y del mes.

**Por qué importa:** `Pen` depende de `FEN` en un 60 % (0,5 directo más 0,2 vía
`FVT`). Volver `FEN` dinámico convierte la prioridad ante desastres en un
ranking que se mueve con el clima, en vez de una lista fija.

### Pertinencia y Coherencia

Dos correcciones al canon van declaradas dentro de esta variable:

**La escala de cuatro niveles.** La sección 22 del Word define `FEN` con tres
(Alta = 3, Media = 2, Baja = 1). SERNAGEOMIN publica **cuatro** niveles de
peligro (Baja, Moderada, Alta y **Muy Alta**), y con tres, «Alta» y «Muy Alta»
quedan pegadas — justo la distinción que más importa para priorizar. Adoptado
por Alexis el 16-ago-2026 (H-15) **para este proyecto**; la Matriz original
sigue sin tocarse.

**La anomalía con signo (H-07).** El canon aplica valor absoluto a `ANPrecip` y
con eso mete sequía e inundación en el mismo número. Para infraestructura es
inservible: una sequía no corta un camino y un temporal sí. `PelPre` no usa
valor absoluto en ningún punto de la cadena.

### Descripción para un Lector No Técnico

La matriz sabe qué cosas del país importan, pero no sabe **cuál, dónde ni
cuándo** está por fallar. Las 39 subestaciones del piloto comparten hoy
exactamente la misma fila —la misma fragilidad, la misma prioridad— desde Arica
hasta Punta Arenas, 35 grados de latitud. `FENef` toma esa fragilidad de fábrica
y le suma lo que está pasando de verdad en ese lugar ese mes.

### Fuentes Utilizadas para la Medición

- **MICR, sección 22 del Word**: `FEN` base por tipo de elemento, 835 ítems.
- **`CClimP`** de la ficha anterior.
- **Minuta Técnica de SERNAGEOMIN** (servicio ArcGIS, 119 zonas
  morfoclimáticas): los cuatro niveles de peligro y su vigencia. Se captura
  cuatro veces al día porque **se sobrescribe y no guarda historia**.

### Ejemplo Genérico

Subestación de Copiapó, marzo de 2015. En la matriz es el ítem 120, con
`FEN = Muy Alta` en la escala adoptada.

### Cálculo

```
FEN₄   = 0,881          (Muy Alta)
CClimP = 1,6            (PelPre = 0,9888, sobre el P99 nacional)

FENef  = mín(1 ; 0,881 × 1,6) = mín(1 ; 1,410) = 1,000
```

La misma subestación en agosto de 2015, mes tranquilo:

```
FEN₄   = 0,881
CClimP = 1,0            (PelPre = 0,5805, bajo el P75 nacional)

FENef  = 0,881          — sin cambio, que es lo correcto
```

### Normalizado

`FENef = 1,000` en el mes del aluvión · `0,881` en un mes tranquilo.

### Estado y validación

**Sin implementar.** Depende de que `CClimP` pase sus tres criterios.

### Limitaciones conocidas

**★ El techo satura, y satura justo donde más importa.** Un activo que ya está
en «Muy Alta» (0,881) sólo puede subir un 13,5 % antes de topar en 1,000. Con
`CClimP = 1,4` ya llega a 1,000, así que **1,4 y 1,6 son indistinguibles** para
los activos más frágiles — que son los que interesan. En el ejemplo de arriba
eso ya ocurre: 1,410 se corta a 1,000.

Hay una forma alternativa que no satura y conserva la lógica del multiplicador:

```
FENef = 1 − (1 − FEN₄) ^ CClimP
```

Con los mismos números da 0,9655 en vez de 1,000, nunca pasa de 1, y con
`CClimP = 1` devuelve exactamente `FEN₄`. **No la adopto por mi cuenta**: la
multiplicación simple es lo que MACC dice literalmente, y cambiarla es decisión
del director. Queda declarada la disyuntiva y sus consecuencias.

---

# 4 · PFef — Ponderación Final Efectiva

**Estado: PROPUESTA Y BLOQUEADA.** Es el final de la cadena — donde el clima
llega a mover el ranking de prioridad — y hoy no se puede calcular.

### Fórmula Numérica/Algebraica

```
FVTef = f(FENef, FANC, VT)          ← la regla real NO está escrita (H-01)
PFef  = IB × FVTef
```

### Por qué está bloqueada — H-01 medido sobre las 835 filas (16-ago-2026)

**La regla SÍ está escrita.** La sección 22 la da explícita, con ejemplo:
`FVT = (FEN + FANC + VT·3) / 3`, normalizada dividiendo por 3. Lo que el
documento admite dos líneas después es que da 0,933 donde la tabla dice 0,87, y
lo atribuye a redondeo.

**No es redondeo.** Aplicada a las 835 filas:

| | |
|---|---|
| error medio | **0,1084** |
| error máximo | 0,2700 |
| filas exactas (±0,005) | **3 de 835** |

**Y es peor que un error de fórmula: ninguna fórmula de esas columnas puede
reproducir la tabla.** `FVT` no es función determinista de (FEN, FANC, VT): de
35 combinaciones distintas, **22 aparecen con más de un `FVT`**. El caso
extremo, `(Alta, Alta, VT=0,7)`, aparece con **seis** valores distintos, de 0,63
a 0,87. Agregando `IB` quedan 18 combinaciones ambiguas; agregando el sector,
13. **La columna no se calculó: se asignó.**

### ★ Lo que apareció al buscar qué sí la explica

`FVT` sigue a la **importancia**, no a la vulnerabilidad:

| correlación con `FVT` | |
|---|---|
| `IB` (importancia) | **+0,671** |
| `VT` | +0,629 |
| `FANC` | +0,544 |
| `FEN` (fragilidad natural) | **+0,337** |

En el mejor ajuste lineal (R² = 0,838), `IB` pesa **0,317** y `FEN` **0,120**:
la importancia pesa casi el triple que la fragilidad ante desastres, dentro de
un índice que se llama *vulnerabilidad*.

Y como `PF = IB × FVT`, la importancia entra **dos veces**. Se mide en el
resultado: `PF` correlaciona **+0,935 con IB** y sólo **+0,349 con FEN**. El
ranking de prioridad de la matriz es, en los hechos, un ranking de importancia
con vulnerabilidad de adorno.

*(`PF = IB × FVT` sí se cumple: error medio 0,0023, 807 de 835 filas dentro de
±0,005. El eslabón roto es sólo `FVT`.)*

### Consecuencia directa para este proyecto

Nuestro coeficiente climático sube el `FEN` de un activo «Muy Alta» en 0,119 como
máximo. Propagado:

| regla usada para FVT | ΔFVT | ΔPF |
|---|---|---|
| la publicada (FEN pesa 1/3) | 0,0397 | **≈ 0,029** |
| la empírica (FEN pesa 0,120) | 0,0143 | ≈ 0,010 |

Es poco. **Pero la matriz está llena de empates:** 835 filas comparten sólo **35
valores distintos** de `PF`. Entonces ese empujón chico sí mueve:

- `ΔPF = +0,029` adelanta a un activo **9 puestos en promedio y hasta 240**.
- Hay **243 filas a menos de 0,03 del umbral `IRMD` Alto/Medio** (PF = 0,5). El
  empujón climático puede cruzar ese umbral para casi un tercio de la matriz.

O sea: el efecto es chico en magnitud y grande en consecuencia — pero eso mismo
dice que el ranking tiene poca resolución. Un orden con 35 valores para 835
elementos se reordena con cualquier cosa.

### Camino para desbloquear (decisión de Alexis)

1. **Usar la fórmula publicada** para recalcular `FVT` en los 39 activos del
   piloto. Es la canónica, es reproducible, y ahí `FEN` sí pesa un tercio.
   Implica declarar que el `FVT` de la tabla no se reproduce — hallazgo, no
   corrección.
2. **Definir un `FVT` propio del proyecto sin `IB` adentro**, para que
   vulnerabilidad e importancia dejen de contarse dos veces.
3. Dejar `PFef` en suspenso hasta que Alexis decida cuál.

Recomendación: **(1)**, porque respeta el canon escrito y no inventa nada; y
anotar (2) como propuesta de corrección al RMD para cuando toque la edición
inversa.

---

# 5 · Pendientes declarados

Estas tres **no llevan ficha canónica todavía**, a propósito: escribirle ficha a
algo que no medimos sería meter una promesa en el canon.

### `CondTerr` — Condición del Terreno

El tercer factor de la conjunción, el que el canon ya nombra: `EAL` =
«precipitación intensa **+ condición de suelo**». En una cuenca árida 104 mm
producen flujo de detritos, porque no hay vegetación que retenga y sí regolito
suelto; en Curicó 200 mm escurren por cauces hechos para eso.

**No se implementa todavía por una razón de método, no de datos:** agregar
aridez ayudaría específicamente a Copiapó, que es el caso con el que estamos
probando. Sería ajustar el modelo al examen. Corresponde: fijar criterio nuevo,
aplicar, y validar contra un evento **distinto** — Antofagasta 1991, Copiapó
2017, Atacama 2020, o julio de 2026.

Dos vías con dato disponible: aridez derivada de la climatología propia, o
humedad de suelo ESA CCI por OPeNDAP del CEDA (ya sabemos bajarla; falta
extenderla a los 39 puntos).

### `TRec` — Tiempo de Recuperación

**Es un hueco real del RMD, no una carencia nuestra (H-11).** La Ley 21.542
define el impacto sobre infraestructura crítica por cinco sub-criterios, y uno
es el tiempo de recuperación. La MICR no lo tiene, y **nadie en el canon lo
tapa**: `FRC` del MCSGS cubre resiliencia y `FAS`/`FPI` cubren interdependencia,
pero el tiempo de recuperación no lo cubre ninguna variable existente.

Es la variable nueva más justificada del proyecto. No se escribe todavía porque
no tenemos con qué medirla. Candidatos de dato ya localizados: las **6.141
emergencias viales del MOP** desde 2015 (si registran apertura y cierre) y los
datos de la **SEC**, que entrega clientes sin luz por comuna **y por hora** —
que es exactamente una curva de recuperación.

### `USCom` / `USProv` — Unidad de Sistema comunal y provincial

**No son variables nuevas: son una extensión del canon.** El MCSGS ya exige
declarar la Unidad de Sistema antes de calcular el `ICSGS`, con US-Nacional,
US-Regional y US-Global. El marco normativo obliga a este proyecto a trabajar en
cuatro niveles administrativos (comunal 346 → provincial 56 → regional 16 →
nacional 1), así que hay que extender esa escalera **hacia abajo**.

Se resuelve declarando dos niveles más de una lista que ya existe. No requiere
ficha de variable.

---

# 6 · Lo que sigue

1. Implementar `CClimP` y correr sus **tres criterios** de validación.
2. Si pasa, implementar `FENef` y resolver la disyuntiva del techo.
3. Reconstruir la regla real de `FVT` para desbloquear `PFef`.
4. Armar el conjunto de validación árido, con criterio fijado antes, para poder
   decidir sobre `CondTerr`.
5. Recién con todo eso validado, traspasar al Word y al Excel del RMD.
