# La corrección: de rareza a peligro

**Escrito el 16-ago-2026, ANTES de calcular la corrección.** Igual que el
protocolo de validación: si el criterio se fija después de ver el resultado, uno
termina acomodándolo sin darse cuenta.

---

## 1 · Qué salió mal

La variable que construí era la anomalía tipificada: cuántas desviaciones típicas
se apartó el mes de lo normal **para ese lugar**.

Eso mide **rareza**. Y la rareza no es peligro:

| | mm | rareza | ¿peligro real? | lo que daba |
|---|---|---|---|---|
| Copiapó, ago-2015 | ~8 mm | +3σ (rarísimo) | ninguno | **0,841** ✗ |
| Copiapó, mar-2015 | 104 mm/48h | +5σ | destruyó la ciudad | 0,995 ✓ |
| Punta Arenas, mar-2015 | moderado | +1,1σ | ninguno | **0,755** ✗ |

En Copiapó llueven unos 12 mm al año. Ocho milímetros son un evento rarísimo y no
le hacen nada a nadie. El instrumento los ponía casi al nivel del aluvión.

## 2 · Qué es el peligro, de verdad

Que la lluvia dañe infraestructura necesita **dos cosas a la vez**:

**(a) Que sea mucha agua, en términos absolutos.** Ciento cuatro milímetros en
48 horas son peligrosos en cualquier parte del mundo. Ocho milímetros no lo son
en ninguna.

**(b) Que supere lo que ese lugar aguanta.** Y acá la anomalía sí entra —pero no
como «clima raro» sino como **«más de aquello para lo que está construido»**.
Valdivia recibe 300 mm en un mes con normalidad: tiene drenaje, alcantarillado y
techos para eso. Los mismos 300 mm en Copiapó son una catástrofe. No es que la
lluvia sea distinta: es que la ciudad no está hecha para ella.

**La clave es que son una conjunción, no una suma.** Si falta cualquiera de las
dos, no hay peligro. Ocho milímetros rarísimos siguen siendo ocho milímetros.
Trescientos milímetros habituales en Valdivia siguen siendo un martes.

Mi error fue usar sólo (b) y llamarlo peligro.

## 3 · La corrección

```
peligro = √( magnitud_nacional × excedencia_local )
```

**`magnitud_nacional`** — el percentil del acumulado en 48 h contra la historia
de **todos los puntos del país**, no la del propio lugar. Responde: ¿es mucha
agua en términos absolutos? Es la pieza que faltaba.

**`excedencia_local`** — el percentil contra la historia del **propio punto**.
Responde: ¿supera lo que este lugar está acostumbrado a recibir?

**Por qué la media geométrica y no un promedio.** Un promedio deja que una
componente alta disimule a la otra en cero: 8 mm rarísimos darían «medio
peligro», y no es medio peligro, es ninguno. La media geométrica se va a cero si
cualquiera de las dos se va a cero, que es justo el comportamiento que
corresponde a una conjunción.

**Y no es una elección caprichosa:** la fórmula canónica del ICSGS ya combina
así sus factores — `√(FCN × FSS × FAS × FPI)`. El canon del RMD ya resuelve las
conjunciones con raíz de producto. Se sigue esa forma.

**Encaje con MACLIMA.** El canon ya tenía las dos piezas separadas y yo las había
colapsado en una: `ANPrecip` es la anomalía local (b), e `InEvExtre` es el evento
con su factor de exposición (a). La corrección no agrega nada al canon: lo
respeta mejor de lo que lo respetaba antes.

## 4 · Lo que NO incluye esta corrección, a propósito

**La condición del suelo.** El canon define el aluvión (`EAL`) como
«precipitación intensa **+** condición de suelo», y con razón: suelo árido y
suelto produce flujo de detritos, suelo saturado produce inundación. No lo
incluyo todavía porque sólo tenemos humedad de suelo satelital para un punto, no
para los 39.

Y hay una razón de método, más importante: agregar un factor de aridez ahora
**ayudaría específicamente a Copiapó**, que es justo el caso con el que estoy
probando. Sería ajustar el modelo al examen. Primero se prueba la corrección de
principio; si pasa, la condición de suelo se agrega después como paso propio y
con su propia validación.

## 5 · Criterio de aprobación, fijado ahora

El criterio anterior (`>0,90` y `<0,70`) era arbitrario y dependía de la escala.
Este es relativo y no se puede acomodar:

Sobre **todos** los pares (subestación, mes) de 1990-2026:

| # | Exigencia | Por qué |
|---|---|---|
| 1 | Copiapó mar-2015 en el **1% superior** del país | el aluvión tiene que estar entre lo más extremo que hubo en 36 años |
| 2 | Punta Arenas mar-2015 **fuera del 10% superior** | no puede alarmar a 2.900 km del evento |
| 3 | Copiapó ago-2015 **fuera del 10% superior** | es el caso que rompió la versión anterior |
| 4 | Copiapó mar-2015 **por encima** de Copiapó ago-2015 | el mínimo indispensable: distinguir el desastre del mes tranquilo |

**Se aprueba sólo si se cumplen las cuatro.** Si falla alguna, se reporta el
fallo y no se ajusta nada para que pase. Una sola corrección, un solo intento
declarado: probar variantes hasta que una acierte sería el mismo error con más
pasos.

---

# RESULTADO — 16-ago-2026

Universo: **17.160 pares (subestación, mes)** de 1990 a 2026.

## Lo que se arregló, que era el problema declarado

| caso | mm/48h | antes | ahora |
|---|---|---|---|
| Copiapó, marzo 2015 (aluvión) | 104,1 | 0,9946 | **0,9722** |
| Punta Arenas, marzo 2015 | 22,9 | 0,7552 | **0,5527** |
| Copiapó, agosto 2015 (tranquilo) | 11,8 | 0,8414 | **0,5536** |

La distancia entre el desastre y el mes tranquilo del mismo lugar pasó de
**0,15 a 0,42**: casi se triplicó. Los dos falsos positivos que rompían la
versión anterior quedaron claramente por debajo del 10% superior. **El error de
confundir rareza con peligro está corregido.**

## El criterio, sin embargo, NO se cumple entero

| # | Exigencia | Resultado |
|---|---|---|
| 1 | Copiapó mar-2015 en el 1% superior | **✗** 0,9722 vs umbral 0,9727 — quedó en el **98,98%** |
| 2 | Punta Arenas fuera del 10% superior | ✓ 0,5527 |
| 3 | Copiapó ago-2015 fuera del 10% superior | ✓ 0,5536 |
| 4 | Copiapó mar > Copiapó ago | ✓ 0,9722 vs 0,5536 |

Falla por **cinco milésimas**, dos puestos de 17.160. Y aun así se declara
**NO APRUEBA**, porque el criterio se fijó antes justamente para no discutir
esto después. Correrlo ahora sería el error que el método existe para evitar.

## Pero el fallo dice algo, y eso vale más que el fallo

Miré qué son los **173 pares que superan a Copiapó**. Casi todos están en zonas
húmedas: San Fernando, Puerto Aysén, Concepción, Quirihue, Curicó, Coyhaique.
Van de 85 a 247 mm en 48 h, con mediana de 130.

O sea: **el instrumento pone un temporal de invierno de 130 mm en Concepción por
encima de los 104 mm que destruyeron Copiapó.** En Concepción eso es un
temporal; en Copiapó, sobre una climatología de 12 mm al año, fue una catástrofe.

La causa es estructural y se ve en la fórmula: `excedencia_local` está acotada a
1 por construcción. Por más sin precedentes que sea el evento, no puede pasar de
1. Entonces, en el rango alto, la conjunción queda gobernada por la magnitud
nacional — y un evento grande en zona húmeda siempre le va a ganar a uno mediano
en zona árida.

**Falta el tercer factor, y es exactamente el que dejé afuera a propósito: la
condición del terreno.** El canon ya lo nombra: `EAL` = «precipitación intensa
**+ condición de suelo**». En una cuenca árida, 104 mm producen flujo de detritos
porque no hay vegetación que retenga y sí regolito suelto; en Curicó, 200 mm
escurren por cauces hechos para eso.

## Qué recomiendo (y no aplico)

Agregar la condición del terreno como tercer factor de la conjunción. Dos vías,
las dos con dato disponible:

1. **Aridez del punto** — derivable de la propia climatología que ya tenemos.
2. **Humedad de suelo satelital ESA CCI** — ya sabemos bajarla por OPeNDAP del
   CEDA; hoy la tenemos para un punto y habría que extenderla a los 39.

**No lo apliqué** por la misma razón que no lo incluí de entrada: agregar aridez
ayudaría específicamente a Copiapó, que es el caso con el que estoy probando.
Hacerlo ahora sería ajustar el modelo al examen. Corresponde: fijar criterio
nuevo, aplicar, y probar contra un evento **distinto** del que se usó para
diagnosticar. Candidatos: Antofagasta 1991, o el propio julio de 2026.

---

# VALIDACIÓN INDEPENDIENTE — 16-ago-2026

Con la corrección de Alexis (razón contra la normal del lugar en vez de
percentil), Copiapó sube del percentil 98,98% al **99,81%** y los cuatro
criterios pasan. Pero Copiapó fue el caso de diagnóstico, así que eso confirma la
implementación, no la generalización.

La validación de verdad son los **376 eventos ReTeRM** — remociones en masa que
SERNAGEOMIN evaluó en terreno entre 1996 y 2026, ninguna de las cuales participó
en el diseño de la medida. 329 fueron detonadas por lluvia; **264 quedaron
emparejadas con clima** (80% de cobertura; Open-Meteo limitó la descarga).

## Prueba 1 · ¿el peligro se concentra donde hubo deslizamientos?

| | resultado | azar |
|---|---|---|
| posición mediana en la historia de su propio punto | **97,0%** | 50% |
| en el decil superior | **67,8%** | 10% |
| en el cuartil superior | 81,4% | 25% |

NULL-1 (meses al azar del mismo punto) da **10,0%** — clava el azar, lo que
confirma que el nulo está bien construido. NULL-2 (eventos atribuidos a otros
puntos) da **51,5%**: o sea que **la mitad de la señal es «fue un mes de temporal
a nivel nacional»** y no destreza específica del lugar. El real le gana igual,
con p = 0,001 en ambos.

## Prueba 2 · ¿la normalización aporta, o bastaba con los milímetros?

Esta prueba nació de una incomodidad: **dentro de un mismo punto, ordenar por
`peligro` es idéntico a ordenar por milímetros**, porque las dos componentes son
monótonas con el lugar fijo. Entonces la prueba 1 confirma que llover fuerte
anticipa deslizamientos —cierto pero no novedoso— y no dice nada sobre la
normalización. Lo que la normalización promete es comparar ENTRE lugares.

AUC sobre 130 meses-con-evento contra 32.870 sin evento, todos los puntos en una
sola bolsa:

| | AUC |
|---|---|
| `peligro` normalizado | **0,8257** |
| milímetros crudos en 48 h | 0,8142 |
| **diferencia** | **+0,0115** |

**Aporta, pero poco.** Y al abrirlo por régimen de lluvia aparece algo incómodo:

| | peligro | mm crudos |
|---|---|---|
| lugares **secos** (56 eventos) | 0,8760 | **0,8780** |
| lugares **húmedos** (74 eventos) | **0,7648** | 0,7429 |

**En los lugares secos, los milímetros crudos ordenan apenas mejor.** La ganancia
está en los húmedos. Es justo al revés de lo que motivó la corrección.

## El límite que impide concluir

Los eventos de ReTeRM están abrumadoramente en el sur húmedo: Biobío 102, Los
Ríos 56, Valparaíso 42, Los Lagos 28. **El norte árido aporta 23 de 329: el 7%.**

O sea que **este conjunto de validación no puede probar bien el caso para el que
se hizo la corrección.** Con 56 eventos «secos» —y varios de ellos en Atacama,
no en el desierto absoluto— no hay poder estadístico para decidir si la razón
contra la normal local sirve donde debería servir.

**Lo que sí quedó establecido:**
- La medida separa meses con deslizamiento de meses sin, con AUC 0,83 sobre 264
  eventos independientes.
- El error de confundir rareza con peligro está corregido.
- La normalización no perjudica y aporta marginalmente.

**Lo que NO quedó establecido, y no se debe afirmar:**
- Que la normalización mejore la priorización en zona árida. El dato no alcanza.
- Que la medida prediga deslizamientos mejor que «llovió fuerte». La ventaja
  sobre los milímetros crudos es de 0,01 de AUC: real, pero chica.

**Qué haría falta:** un conjunto de eventos del norte árido. Candidatos:
Antofagasta 1991, Copiapó 2015 y 2017, Atacama 2020. Son pocos y catastróficos —
justamente el régimen que ReTeRM no cubre.

---

## Una señal buena, no buscada

Los ocho meses más peligrosos de los 36 años, según el instrumento corregido:

```
0.9998  2023-08  202 mm  Curicó          0.9991  2025-07  172 mm  Puerto Aysén
0.9996  2026-07  196 mm  La Ligua        0.9990  2026-07  169 mm  Valdivia
0.9994  2026-07  186 mm  La Serena       0.9986  2026-07  161 mm  Ovalle
0.9993  2023-08  181 mm  San Fernando    0.9986  2026-07  160 mm  Quillota
```

**Cinco de los ocho son julio de 2026** — el temporal que motivó este proyecto.
Nadie se lo dijo al instrumento: lo encontró solo, en la serie completa de 36
años. Es una confirmación independiente de que la medida está mirando en la
dirección correcta.
