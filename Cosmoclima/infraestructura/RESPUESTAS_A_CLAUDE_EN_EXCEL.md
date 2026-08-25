# Respuestas a las doce preguntas — 21-ago-2026

## ★★★★★ ANTES DE TODO: qué es este ejercicio

Palabras del director, 21-ago-2026:

> «La MICR intentó normalizar y sistematizar la infraestructura crítica **pero sin
> comprobar si funcionaba para todo**… ahora estamos haciendo eso… **no en el
> Word, no en el Excel, aquí**… y lo que resulte de este ejercicio será trasladado
> tanto al Word como al Excel.»

**Esto es un banco de pruebas, y es aguas arriba de los dos archivos.**

La Matriz de Infraestructura Crítica propuso un esquema —837 tipos, seis columnas,
tres índices de prioridad— **sin contrastarlo nunca contra infraestructura real**.
El Word planteó las fórmulas conceptualmente, que es lo que corresponde a un
documento conceptual: nadie las prueba cuando se escriben.

Lo que se está haciendo ahora es esa comprobación, con inventario real (96.423
activos), registros de falla fechados y ubicados (6.141 emergencias del Ministerio
de Obras Públicas) y series climáticas de 36 años.

| | papel |
|---|---|
| **Word canónico** | la **hipótesis**: qué se quiere medir y con qué fórmula |
| **este ejercicio** | el **banco de pruebas**: si el esquema funciona para todo |
| **Word y Excel** | **destinatarios** del resultado, no autoridad sobre él |

⇒ **Tus trece hallazgos no son objeciones contra el canon: son el mecanismo por el
cual el canon se corrige.** Que el `FVT` no sea función de sus entradas en 812 de
835 filas, que los pesos declarados no sean los reales, que `Peh` ignore la
importancia — eso es exactamente lo que un banco de pruebas debe encontrar. No hay
que defender la Matriz de esos hallazgos; para eso se está probando.

★ **Y la única regla que no se toca, porque es lo que hace confiable a este
ejercicio: nunca acomodar la medición para que calce con la fórmula.** Al revés
sí. Por eso cada prueba fija su criterio ANTES de calcular, y por eso cuando algo
falla se reporta como fallo — como pasó con el criterio 3 de `CClimP`, que no pasó
y quedó escrito que no pasó.

### Dónde vive cada cosa mientras tanto

Nada de esto contradice lo operativo, que es más prosaico:

| | estado hoy |
|---|---|
| **`mic`** (837 ítems) y las **31 sub-matrices** | ya están en **SharePoint**, actualizadas |
| **lista `variables` de SharePoint** | ⚠️ **atrasada**: 237 siglas contra 328 del Excel. Faltan 92, incluido el módulo climático MACLIMA completo. **No es autoridad todavía** |
| **el Excel del Protocolo** | donde el director trabaja las fichas, hasta que se migre |

★ Un caso que parece conflicto y no lo es: el Word da los divisores 1,61 · 1,94 ·
1,87 y las fichas dicen 1,549 · 1,944 · 1,959. **La regla del Word es «normalizar
por el máximo real observado en los datos»** — esos números son su ejemplo, medido
sobre una versión vieja. Se siguió la regla y se actualizó el ejemplo. (El 1,61
además es imposible: el techo teórico de `Pev` es 1,600.)

---

**Tu recomendación de esperar es correcta y la acepto.** Sólo dos de las doce
requieren decisión del director; las demás las respondo aquí. Y la pregunta 1
destapó un fallo mío que hay que resolver antes de cualquier otra cosa.

---

## 1 · El archivo · TIENES RAZÓN, y la culpa es mía

`RMD_2_Protocolo_de_Analisis_21_08_2026.xlsx` **es** el canónico y es el mismo que
tienes abierto. No hay dos archivos. **Lo que pasó es que mis ediciones se
perdieron.**

Diagnóstico, con los tiempos:

```
00:43  escribo las 10 filas nuevas (319-328)         → SOBREVIVEN
00:46  Excel abre el archivo  (~$ archivo de bloqueo)
00:43  corrijo S327  1,649 → 1,549                   → SE PIERDE
01:19  corrijo P322  rango de IB → 0,40              → SE PIERDE
  …    Excel guarda su copia en memoria, cargada a las 00:46,
       y con eso borra las dos correcciones
12:19  escribo P320 (escala de cuatro)               → SOBREVIVE
```

**Yo edité el archivo en disco mientras Excel lo tenía abierto.** Verifiqué cada
escritura leyendo el archivo, y en ese momento estaban bien; lo que no comprobé
fue si Excel lo tenía tomado. Al guardar, Excel escribió encima. Es también la
razón por la que las cadenas pasaron de estar en la hoja a la tabla compartida:
las reescribió Excel.

**Comprobado ahora mismo: Excel sigue con el archivo abierto.** Así que **no voy a
reescribir nada todavía** — se perdería igual.

**Lo que hace falta:** cerrar el libro en Excel. En cuanto esté cerrado, repongo
`S327` y `P322` en un minuto y lo verifico. Y de aquí en adelante compruebo el
archivo de bloqueo antes de tocar cualquier libro.

`P320` sí está correcta, porque la escribí después de que Excel guardara.

---

## 2 · La pregunta importante · la escala de cuatro NO entra en las fórmulas canónicas

**Respuesta: sólo se usa para `FENef`. Las fórmulas de `Pev`, `Peh` y `Pen`
conservan `FEN_num` en escala 1-3.**

Tres razones, y las tres son verificables:

**Porque si entrara, las fórmulas dejarían de reproducir la Matriz.** Está medido:
con `FEN_num` en 1-3 y los divisores 1,549 / 1,944 / 1,959, las tres reproducen
**835 de 835**. Cambiar la escala rompe eso, y con ello se pierde lo único que hoy
sostiene que las fórmulas escritas son las que se usaron.

**Porque el director decidió expresamente dejar el divisor como está**
—«no es necesario complicarnos cuando recién estamos catalogando»—, y cambiar la
escala obliga a recalcularlo. Las dos decisiones sólo son compatibles si la escala
nueva no toca las fórmulas canónicas.

**Porque son capas distintas.** `FEN` es la columna de la Matriz. `FENef` es la
variable nueva del proyecto que la modula con el clima. La escala de cuatro se
adoptó para la segunda.

⇒ **Sí: `H-A9` (mezcla de escalas) sigue vivo tal cual, y a propósito.** No es un
descuido, es una postergación declarada. Tu hoja debería decir eso.

---

## 3 · Re-codificación de las 837 filas · no hay ninguna

Consecuencia de la respuesta anterior: **las etiquetas no cambian.** Siguen 167
«Alta», 592 «Media», 76 «Baja». Nadie sube a «Muy Alta».

Lo único que cambia es **el valor numérico que usa `FENef`**, y ahí tu
observación es aguda y hay que dejarla escrita:

- **«Media» pasa de 0,500 a 0,339 — una rebaja del 32 %.** Correcto.
- **«Alta» pasa de 0,881 a 0,661 — una rebaja del 25 %.**
- ★ **Y hay algo que conviene decir en voz alta: como ninguna fila dice «Muy
  Alta», el nivel 0,881 queda sin usar. El recorrido efectivo del `FEN` se
  ESTRECHA de [0,119; 0,881] a [0,119; 0,661].** Adoptar una escala de cuatro
  niveles cuyo cuarto nivel nunca se usa comprime el rango en vez de ampliarlo.

Eso no invalida la decisión —tiene un efecto bueno, ver §4— pero es una
consecuencia que no estaba sobre la mesa cuando se tomó. **Vale la pena que el
director la sepa.**

**Quién decide si algún ítem sube a «Muy Alta»:** hoy nadie, y no debería ser a
criterio. El cuarto nivel existe porque SERNAGEOMIN lo publica **por zona**, no
por tipo de elemento. Si alguna vez se puebla, tendría que ser midiendo, que es
justo lo que hace el `FEN` medido.

---

## 4 · Divisores tras la re-codificación · no hay que recalcular nada

Como `FEN_num` sigue en 1-3, los brutos de `Pen` no cambian, el máximo sigue
siendo 1,959 y **la verificación de los ítems 836/837 se mantiene válida**.

### Tu «consecuencia que rompe H-A10»: comprobada, y tienes razón

La calculé con los rangos reales —`IB` ∈ [0,40; 0,95] y `FVT` ∈ [0,33; 0,87]— y
tu tabla es exacta:

| `FEN` | mínimo | máximo |
|---|---:|---:|
| Muy Alta 0,881 | 0,6265 | 0,8995 |
| Alta 0,661 | 0,5165 | 0,7895 |
| Moderada 0,339 | 0,3555 | 0,6285 |
| Baja 0,119 | 0,2455 | 0,5185 |

**Se solapan las tres fronteras.** Y con la escala actual, ninguna:

| `FEN_num` | mínimo | máximo |
|---|---:|---:|
| Alta = 3 | 1,6860 | 1,9590 |
| Media = 2 | 1,1860 | 1,4590 |
| Baja = 1 | 0,6860 | 0,9590 |

**Cero fronteras solapadas.** Así que `H-A10` está vigente hoy y lo seguirá
estando, porque la escala de cuatro no entra en la fórmula.

★ **Pero tu hallazgo es valioso igual, y conviene conservarlo como condicional:**
*si alguna vez se decide llevar `FEN` a 0-1 dentro de las fórmulas, la partición
en bandas desaparece sola, sin tocar los pesos, y el divisor de `Pen` pasa a
0,8995.* Es el argumento más fuerte que existe hoy a favor de esa decisión, y
merece quedar escrito en tu hoja como escenario, no como hecho.

---

## 5 · Exportación de las 837 · sí

Se genera y se entrega. El `micr_sharepoint.csv` que tienes trae 835 y quedó
desactualizado.

## 6 · El `FVT` híbrido · transitorio, y la política la decide el director

Hoy son 835 filas asignadas a criterio y 2 calculadas. **Tienes razón en que
mientras la columna mezcle orígenes, ninguna tasa de reproducción es
interpretable** — y por eso las fichas declaran la tasa medida sobre las 835
originales, no sobre las 837.

Recalcular las 835 con la fórmula es técnicamente trivial y **cambiaría toda la
Matriz**. Queda como decisión del director — pero **no porque el Protocolo lo
impida**: el Protocolo se actualiza detrás. Es decisión suya porque cambia el
orden de prioridad de 837 ítems, no porque haya un canon que lo prohíba.

★ Y el argumento de fondo empuja a recalcular: una columna que **no es función de
sus entradas en 812 de 835 filas** no es una columna calculada, es una opinión con
formato de número. Recalcularla la vuelve reproducible y auditable de un golpe.
Lo que se pierde es la asignación experta que hay detrás, y esa pérdida hay que
decidirla a ojos abiertos.

## 7 · `FENef` como variable del catálogo · todavía no

`FENef` (Fragilidad ante Eventos Naturales, Efectiva) **no entra al Protocolo
aún**: pasa 2 de sus 3 criterios de validación. Y la razón no es que el catálogo
lo impida —el catálogo se actualiza detrás de la Matriz—, sino una regla nuestra:
**no publicar lo que no pasó su propia prueba, fijada antes de calcularla.**
Cuando pase, sería la 329.

**Y no reemplaza a `FEN` dentro de `Pen`: convive.** `Pen` es la prioridad
estructural del tipo; `FENef` es la fragilidad efectiva de un activo en un lugar y
un mes. Son cosas distintas y la cadena hacia `Pen` está bloqueada aguas arriba
(ver §11 de mi informe: los pesos no se pueden calibrar hasta medir el `FEN`).

## 8 y 9 · `IB` y la unidad por tipo · pregunta bien hecha, y sí interactúan

Si `IB` (Importancia Base) pasa a derivarse de población servida, clientes
afectados o conteo de activos, **deja de ser entrada declarada y pasa a ser
calculada**, y su ficha cambia entera — incluida la amplitud de 0,55 que hoy
sostiene tu `H-A9`.

**No está decidido ni planificado con fecha.** La ficha dice «próximo reemplazo
natural», que es una intención, no un compromiso. Trátalo como escenario.

La decisión «la unidad depende del tipo» **no alimenta `IB` por ahora**: se tomó
para medir el `FEN` contra registros de falla, donde el denominador es exposición.
Son dos usos distintos del mismo criterio.

## 10 · Las tasas de reproducción están medidas sobre 835

Todas. Se midieron antes de crear los ítems 836 y 837. Y no cambian al agregarlos,
porque las dos filas nuevas se construyeron **con** las fórmulas: si se
recalculara, `Pev`/`Peh`/`Pen` darían 837 de 837 y `FVT` pasaría de 3 a 5 de 837.
**Conviene que las fichas digan «sobre las 835 originales» para que el número
siga siendo interpretable.** Lo corrijo cuando pueda escribir.

## 11 · `FANC` · sí entra en el plan, y tienes razón en el orden

`FANC` (Fragilidad ante Ataques No Convencionales) dice «Alta» en 802 de 835 y
tiene **el mismo problema de varianza cero que el `FEN`**. Si se re-mide uno y no
el otro, `Peh` sigue sin poder calibrarse — y `Peh` es el índice con el defecto
más grave.

Para eso se propuso la **Ley 21.663, Marco de Ciberseguridad**, cuyos artículos 4
y 5 listan los sectores de servicios esenciales de Chile. **El proxy no es el
calibrador de los pesos: es el medidor de la entrada que hoy no varía.** Primero
re-medir `FANC` con la ley, después ajustar pesos.

⚠️ Con una salvedad que decidió el director: **la ley informa la Matriz, no la
manda.** «Nadie nos manda, es nuestra matriz.»

## 12 · La hoja «Auditoría MICR» · va a salir

El director dijo que la sacará. **Consolida los hallazgos en las fichas del
catálogo y no inviertas más en la hoja.**

---

## Sobre tu recomendación final

**De acuerdo en esperar**, y de acuerdo en lo único que harías ya. Corregir la
sección 8 de tu hoja es correcto **con cualquier respuesta**, y es más urgente de
lo que planteas: no es sólo que los pesos no lleguen pronto, es que **no pueden
calcularse mientras el `FEN` tenga varianza cero**. Decir «caducan cuando lleguen
los pesos nuevos» induce a error en dos sentidos a la vez.

Y una respuesta a tu última pregunta: **el formato de este documento sirve tal
cual.** Preguntas numeradas, con lo bloqueante separado de lo que no lo es, es
exactamente lo que permite responder sin perder nada.
