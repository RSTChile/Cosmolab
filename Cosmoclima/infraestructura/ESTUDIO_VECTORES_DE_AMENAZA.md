# Estudio: los vectores de amenaza, y por qué la vulnerabilidad no es un número

**16-ago-2026.** Escrito a partir de dos observaciones de Alexis que cambian el
diseño completo del proyecto. No es un parche a la matriz: es la estructura que
debería reemplazarla.

---

## 1 · El diagnóstico medido, en tres líneas

Sobre las 835 filas de la MICR:

| columna | qué pasa |
|---|---|
| `FANC` | **802 de 835 dicen «Alta» — el 96 %.** Uno solo dice «Baja». Es una constante. |
| `FEN` | 71 % dice «Media». El 69,6 % de la matriz comparte la combinación `(Media, Alta)`. |
| `VT` | **La única que funciona**: TI 0,84 · digital 0,84 · energía 0,70 · vial 0,53 · hídrico 0,44 |

Si de cuatro insumos uno es constante y otro está apelmazado, lo único que queda
con información es `IB`. Por eso `PF` correlaciona **+0,935 con IB** y sólo
**+0,349 con FEN**. **El defecto de diseño y la patología que medimos son el
mismo hecho.**

Y `VT` funciona por una razón que enseña el resto: es la única de las tres que
describe **lo que la cosa es** —cuánta tecnología tiene encima— en vez de
pedirle un veredicto sobre un futuro incierto.

---

## 2 · La separación fundamental: naturales y deliberados

> «Separemos los vectores de amenaza en Naturales y Deliberados, eso es
> fundamental.» — Alexis, 16-ago-2026

No es una clasificación cómoda: las dos familias **obedecen lógicas opuestas**, y
mezclarlas es lo que produce columnas muertas.

| | **Naturales** | **Deliberados** |
|---|---|---|
| ¿elige blanco? | **no** — golpea lo que esté ahí | **sí** — busca el punto débil |
| ¿se adapta a la defensa? | no | **sí** — si blindas el frente, entra por atrás |
| ¿cuántos nodos a la vez? | **muchos, gratis** | pocos, y cada uno cuesta |
| ¿se puede prever? | **sí, con dato físico** | sólo con inteligencia y probabilidad |
| ¿qué la determina? | condiciones del ambiente | capacidad e intención de un actor |

**De ahí sale el 96 % de «Alta» en FANC.** Como el atacante elige y se adapta,
para *cualquier* elemento existe *algún* ataque no convencional que funciona.
Preguntar «¿es vulnerable a ataques no convencionales?» tiene una sola respuesta
honesta —sí— y la columna se llena de «Alta» y deja de informar.

> Un búnker es inmune a los disparos. Tírale un panal de avispas adentro y la
> situación cambia. **Las dos cosas son ciertas al mismo tiempo**, y un escalar
> sólo puede decir una.

La consecuencia de diseño es directa: **`FANC` no puede ser un número por
elemento.** Tiene que ser una función `(elemento, vector, capacidad del actor)`,
y agregarse recién cuando se declara contra qué actor se está evaluando.

**Y hay un detalle que el canon ya vio:** en el MCSGS, `FSS` mide la
sincronización — cuántos nodos caen a la vez. Un atacante tiene que *planificar*
esa sincronización; **un temporal la produce gratis**. Ésa es la ventaja
estructural de la amenaza natural y la razón por la que este proyecto existe.

---

## 3 · El principio que cambia todo: el terreno ENCAMINA la amenaza

> «Si hay lluvias persistentes, baja humedad del suelo, isoterma alta… es obvio
> que puede haber remociones en masa. Pero en el sur hay lluvias persistentes,
> pero alta humedad en el suelo, y aunque la isoterma esté alta y llueva y se
> derrita nieve, las remociones en masa son pocas porque esos suelos ya han sido
> ampliamente erosionados y removidos por miles de años de precipitaciones. Lo
> que allí pasa, como ha pasado, es desborde de los ríos, particularmente porque
> se ha construido en las riberas, se ha deforestado, etc.» — Alexis

El terreno no *modula* el peligro. **Lo encamina hacia una amenaza u otra.**

```
   MISMA METEOROLOGÍA          +   TERRENO SECO, REGOLITO SUELTO   →  REMOCIÓN EN MASA
   (lluvia persistente,
    isoterma alta, deshielo)   +   TERRENO SATURADO, YA ERODADO    →  DESBORDE FLUVIAL
                                   + ribera construida, deforestada
```

### ★ La consecuencia técnica: la humedad del suelo entra con SIGNO OPUESTO

| amenaza | necesita el suelo… |
|---|---|
| Remoción en masa | **seco** — regolito suelto, sin cohesión, sin vegetación que retenga |
| Desborde fluvial | **saturado** — ya no infiltra, todo lo que cae escurre al cauce |

**Esto explica un resultado que teníamos sin explicar.** En Cosmoclima, la
humedad de suelo ESA CCI «no decidía» y sólo acotaba el desacuerdo a 5 meses de
91. Claro: la estábamos usando como si tuviera un solo sentido, cuando su signo
depende de **cuál** de las dos amenazas se esté evaluando. Una variable que
entra con signo opuesto en dos fenómenos, promediada, se anula.

**También explica el fracaso del recálculo de hoy.** `ExpEstr` daba a Copiapó
0,128 —cuarto desde abajo— porque medía «cuán lluvioso es el lugar». Estaba
tratando de representar con un número dos amenazas que piden terrenos opuestos.

### La intervención humana es una condición, no un adorno

En el caso del desborde, Alexis nombra dos condiciones que **no son
meteorológicas ni geológicas**: se construyó en las riberas y se deforestó. Son
decisiones humanas acumuladas que **estrechan el cauce y aceleran el
escurrimiento**. Entran en la receta como cualquier otra condición, y son las
únicas que se pueden revertir.

---

## 4 · La estructura propuesta

### 4.1 · Vocabulario de condiciones

**Meteorológicas** (estado del mes, dato con historia):
`P48` lluvia máxima en 48 h · `Ppers` lluvia persistente (acumulado multi-día) ·
`Iso0` altura de la isoterma cero · `Tmax` · `DefHid` déficit hídrico acumulado

**De terreno** (estado del lugar, cambia lento):
`HumSuelo` humedad del suelo · `Pend` pendiente · `Veg` cobertura vegetal ·
`Cuenca` superficie aguas arriba · `Reg` regolito/material suelto disponible

**De intervención humana** (acumuladas, reversibles):
`Ribera` ocupación de riberas · `Defor` pérdida de bosque en la cuenca ·
`Imperm` impermeabilización del suelo urbano

### 4.2 · Los vectores naturales, cada uno con su receta

Cada vector es una **conjunción**: si falta una condición, no hay amenaza. Se
combinan con raíz de producto, igual que `ICSGS = √(FCN×FSS×FAS×FPI)`.

| vector | condiciones que exige |
|---|---|
| **Remoción en masa** | `P48` alta · `HumSuelo` **baja** · `Pend` alta · `Veg` baja · `Reg` alto · `Iso0` alta |
| **Desborde fluvial** | `Ppers` alta · `HumSuelo` **alta** · `Cuenca` grande · `Iso0` alta · `Ribera` · `Defor` |
| **Aluvión / flujo de detritos** | `P48` muy alta · zona árida · `Pend` alta · `Reg` alto |
| **Aislamiento por nieve** | precipitación · `Iso0` **baja** · acceso único |
| **Incendio forestal** | `DefHid` alto · `Tmax` alta · combustible · viento |

La misma lluvia entra en varias recetas; lo que decide cuál se activa es el
terreno. **Eso es exactamente lo que describió Alexis.**

### 4.3 · Los vectores deliberados, con otra lógica

No se calculan desde el ambiente sino desde el **encuentro entre lo que el
elemento expone y lo que el actor puede hacer**:

```
Vuln(elemento, vector, actor) =
      ¿el elemento expone la propiedad que el vector ataca?
    × ¿el actor declarado tiene la capacidad de ese vector?
    × ¿no hay redundancia que absorba la pérdida?
```

**Propiedades expuestas del elemento** (derivables de qué es la cosa, igual que
`VT` — que es la prueba de que se puede):
`dep_energia` · `dep_datos` · `dep_humana` · `exp_intemperie` · `extension`
(puntual / lineal / areal) · `confinamiento` · `redundancia` · `t_reposicion`

| vector | qué propiedad explota | carretera | data center | búnker |
|---|---|---|---|---|
| ciberataque | `dep_datos` | bajo | **alto** | bajo |
| corte eléctrico | `dep_energia` | bajo | **alto** | medio |
| sabotaje físico | `extension`, vigilancia | **alto** | bajo | bajo |
| disparos | blindaje ausente | medio | medio | **nulo** |
| contaminación interior | `confinamiento` + `dep_humana` | nulo | bajo | **alto** |

El búnker aparece con dos valores opuestos según el vector. **Eso es lo que un
escalar no puede representar, y es la razón de fondo del 96 %.**

### 4.4 · Qué pasa con FEN, FANC y VT

Dejan de ser columnas que se llenan y pasan a ser **resultados**:

```
FEN (elemento, lugar, mes)  = agregación sobre los vectores NATURALES
FANC(elemento, actor)       = agregación sobre los vectores DELIBERADOS
VT  (elemento)              = se queda: ya es una propiedad del elemento
```

`VT` sobrevive intacta porque ya estaba bien planteada. Es el modelo a seguir,
no la excepción.

---

## 5 · Qué dato real hay para cada condición (verificado al 16-ago-2026)

| condición | fuente | estado |
|---|---|---|
| `P48`, `Ppers` | ERA5 / Open-Meteo | ✅ **bajado**, 1990-2026, 39 + 91 puntos |
| `HumSuelo` | ESA CCI vía OPeNDAP del CEDA | 🟡 método probado en Cosmoclima, **1 punto**; falta extender |
| `Iso0` | Open-Meteo archivo | ❌ **no existe**: `freezing_level_height` y los niveles de presión vuelven vacíos (probado hoy) |
| `Iso0` (alternativa 1) | derivada de `Tmax` + elevación + gradiente estándar 6,5 °C/km | 🟢 **calculable hoy** con lo que ya tenemos; es aproximación y se declara |
| `Iso0` (alternativa 2) | ERA5 del Copernicus CDS, variable «zero degree level» | 🟡 existe, **exige registro de Alexis** |
| `Pend`, `Cuenca` | modelo de elevación (SRTM / Copernicus DEM) | 🟡 público, sin bajar |
| `Veg` | NDVI MODIS | 🟢 experiencia previa en Cosmoclima |
| `Reg` | geología SERNAGEOMIN | 🟡 por verificar |
| `Ribera`, `Defor` | catastro de uso de suelo CONAF | 🟡 por verificar |
| eventos de remoción para validar | **ReTeRM SERNAGEOMIN, 376 eventos en terreno** | ✅ **bajado** |
| eventos de inundación/desborde | SENAPRED 50.457 eventos por comuna | ✅ **bajado** |

---

## 6 · La primera prueba falsable de esta estructura

La afirmación de Alexis es **comprobable con dato que ya tenemos o sabemos
conseguir**, y hay que probarla antes de construir sobre ella:

> **Hipótesis:** en los eventos de remoción en masa, la humedad del suelo previa
> es *baja*; en los eventos de desborde fluvial, es *alta*. Y el contraste
> norte↔sur se explica por eso y no por la cantidad de lluvia.

**Cómo se prueba:** los 376 eventos ReTeRM (remoción) contra los eventos de
inundación de SENAPRED, cruzados con humedad ESA CCI del mes previo. Si la
humedad previa separa las dos familias, la estructura se sostiene. Si no las
separa, hay que revisarla antes de escribir una línea más.

**Criterio, fijado ahora:** la diferencia de humedad previa entre las dos
familias debe ser significativa contra un nulo de fechas barajadas dentro del
mismo punto (p < 0,01), y el signo debe ser el predicho — **no basta con que
difieran**.

---

## 7 · Lo que esto reemplaza del trabajo anterior

- **`PelPre` no se tira: se reubica.** Deja de ser «el peligro» y pasa a ser una
  condición (`P48`) dentro de dos recetas distintas. Su validación contra 328
  eventos sigue siendo válida para lo que ahora es: medir lluvia peligrosa.
- **`ExpEstr` se descarta.** Medía cuán lluvioso es un lugar y lo llamaba
  fragilidad. Falló el ancla de Copiapó por la razón que este estudio explica.
- **`CondTerr` deja de ser un factor y pasa a ser el enrutador.** No corrige el
  peligro: decide de qué amenaza estamos hablando.
- **La regla de no ajustar al examen sigue en pie.** La estructura se valida
  contra eventos que no participaron en su diseño, y el criterio se fija antes.
