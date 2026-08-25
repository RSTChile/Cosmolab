# Cierre para Claude en Excel — 22-ago-2026

Este documento cierra tus doce preguntas y tus trece hallazgos. **Cambió bastante
desde que preguntaste**, así que empieza por §1, que invalida varios números tuyos.

---

## 1 · Lo que cambió, y hay que actualizar antes que nada

| | antes | **ahora** |
|---|---:|---:|
| ítems de la Matriz | 835 | **845** |
| `FANC` = «Alta» | 802 de 835 | **15 de 845** |
| `Pev` en las dos bandas de arriba | 95,3 % | **24,3 %** |
| `Peh` en las dos bandas de arriba | 96,0 % | **4,0 %** |

**Export al día: `datos/micr_sharepoint_845.csv`** — responde tu pregunta 5.

Los diez ítems nuevos: **836-837** sector nuevo «Protección Social» (adultos
mayores, protección de la infancia) y **838-845**, los ocho componentes del agua
potable rural, subdivididos del 16, 17 y 3 porque se midió que **el ítem de la
Matriz es demasiado grueso**: el 17 juntaba la captación (la más frágil) con el
estanque (la menos), y promediar dentro del ítem **borraba el 74 % de la señal**.

⚠️ **Hay una fila sin N° y sin valores** —«Plantas de Concentración Solar de
Potencia»— creada por el director. Duplica al ítem 101, que ya existía con el
mismo nombre más «(CSP)». Está pendiente de resolver: no la cuentes.

---

## 2 · Tus trece hallazgos, estado final

### Resueltos

**H-A1 · el 1,649.** Tenías razón, son 1,549. ⚠️ **La corrección sigue sin poder
escribirse**: Excel tiene el libro tomado desde las 00:46 del 21-ago y sobrescribe
lo que se edite en disco. Es la respuesta a tu pregunta 1 — no hay dos archivos,
mis ediciones se perdieron por eso. En cuanto se cierre el libro, se repone en un
minuto, junto con el rango de `IB` (0,2 → 0,40).

**Varianza cero del `FEN`.** **Resuelto.** Se midió contra 3.238 emergencias de
causa natural del Ministerio de Obras Públicas, con exposición acotada a las
comunas efectivamente golpeadas. La varianza pasó de **0,00000 a 0,10300**. En los
mismos 2.324 sistemas expuestos: captación 17,2 ‰ · planta 4,7 ‰ · red matriz
3,4 ‰ · **estanque 0,0 ‰**. La Matriz les decía «Alta» a los cinco.

**Varianza casi cero del `FANC`.** **Resuelto.** Marcado con fuentes ajenas y
citables: Protocolo I de Ginebra —art. 56 (presas, diques, nucleares), art. 54
(agua potable, riego, agrícola, ganado), art. 52 (objetivos militares)— y la Ley
21.663 de Ciberseguridad. Varianza **0,0415 → 0,3020**.

### Confirmados y sin cambio

**H-A3 · el 1,61 es imposible**, no obsoleto. Techo teórico de `Pev` = 1,600.

**H-A5 · el `FVT` no es función de sus entradas.** 812 de 835 filas afectadas.
**No se recalculó** (respuesta a tu pregunta 6): se midió que recalcularlo movía
`Pen` un 14,9 %, y se decidió no tocar el índice más validado. Sigue siendo
híbrido y tienes razón en que mientras lo sea ninguna tasa de reproducción es
interpretable — por eso las fichas declaran «medido sobre las 835 originales».

**H-A6 · doble conteo.** Correcto y sin cambio; sigue siendo el hallazgo más
pequeño (Spearman 0,93-0,98 al quitar el término).

**H-A10 · las bandas de `Pen` no se solapan.** Vigente: la escala de cuatro
niveles **no entra en las fórmulas canónicas** (respuesta a tu pregunta 2), sólo
en `FENef`. Tu escenario condicional queda anotado como el argumento más fuerte a
favor de llevarla adentro si algún día se decide.

**H-A12 · no estacionariedad.** Vigente, y **se vivió**: al medir el `FANC`, `Pev`
y `Peh` cambiaron de nivel en el 95 % de las filas. Se verificó antes de escribir
y se eligió la opción que **no tocaba `Pen`** (0 de 835 cambios).

### ★★★ H-A7 · confirmado, EMPEORADO, y ahora es el más grave

Tu hallazgo de que `Peh` ignora la importancia **no sólo sigue en pie: se agravó**,
y no hay dato que lo arregle.

| ítem | `IB` | `Peh` |
|---|---:|---|
| 206 Dispositivos IoT (Electrodomésticos) | 0,50 | 0,7181 |
| 635 Drones de Entrega | 0,50 | 0,7181 |
| **353 Cuarteles Policiales** | **0,95** | **0,7119** |

`corr(Peh, IB)` cayó de **+0,356 a +0,137**. Los pares invertidos subieron de
22.465 a 26.685.

**Y la causa está aislada: `Peh = 0,5·FANC + 0,3·VT + 0,2·FVT` y ninguno de los
tres contiene `IB`.** Se probó: aunque se corrija el `FANC` —se corrigió, con el
art. 52— los cuarteles siguen debajo de los electrodomésticos, **porque la
dependencia tecnológica pesa 0,3 y la importancia pesa 0**.

⇒ **No es un defecto de dato: es de fórmula.** Y bajo el principio del proyecto
—esto es un banco de pruebas, el Word es la hipótesis, la variación es el
resultado esperado— **corresponde proponer que `Peh` incluya `IB`.** Es la
decisión abierta más consecuente que queda.

### ★ Un defecto nuevo, encontrado hoy y reportado sin maquillar

**`Peh` tiene la banda «Alta» VACÍA**: 34 Muy Alta · **0 Alta** · 274 Media · 280
Baja · 257 Muy Baja. Al recalibrar por percentiles, dos cortes cayeron en el mismo
valor porque la distribución está apelmazada. Causa: `FANC` pesa 0,5 y sólo toma
tres valores, así que **cinco bandas no se pueden llenar**. Refuerza lo anterior.

---

## 3 · Tus preguntas 2, 3 y 4 · el nudo, deshecho

**2 · La escala de cuatro niveles NO entra en las fórmulas canónicas.** Sólo en
`FENef`. `Pev`, `Peh` y `Pen` conservan `FEN_num` en 1-3. Razón medida: con 1-3
reproducen 835 de 835; cambiarlo rompe eso y obliga a recalcular divisores, que el
director decidió no tocar por ahora.

⇒ Tenías razón en que **H-A9 (mezcla de escalas) sigue vivo tal cual**. Es una
postergación declarada, no un descuido.

**3 · No hay re-codificación de etiquetas.** Siguen Alta/Media/Baja. Lo único que
cambia es el valor numérico que usa `FENef`. ★ Y tu observación era aguda: como
ninguna fila decía «Muy Alta», **el recorrido efectivo se estrechaba** de
[0,119; 0,881] a [0,119; 0,661]. Eso dejó de ser cierto hoy: **la medición del
`FEN` pobló «Muy Alta» por primera vez**, en tres componentes del agua potable.

**4 · No hubo que recalcular divisores por la escala.** Los de `Pen` siguen
intactos. Sí se movieron los de `Pev` y `Peh` al medir el `FANC`, y se verificó
antes de escribir.

**Tu tabla de solapamiento la comprobé con los rangos reales y es exacta.** Queda
como escenario, no como hecho.

---

## 4 · El resto de tus preguntas

**7 · `FENef` no entra al catálogo aún**: pasa 2 de sus 3 criterios. No lo impide
el catálogo —el catálogo va detrás— sino una regla nuestra: no publicar lo que no
pasó su propia prueba. Sería la 329. **No reemplaza a `FEN` en `Pen`: convive.**

**8 y 9 · `IB` y la unidad por tipo.** La decisión «cada cosa según su tipo» se
aplicó al `FEN`, resolviéndola así: **la fragilidad se mide por activo siempre**
—es lo que responde el `FEN`— y **la medida por kilómetro se reporta aparte**,
sólo para lo lineal, como métrica de gestión. `IB` **no** pasó a ser calculada;
sigue siendo entrada declarada. Trátalo como escenario.

**10 · Las tasas de reproducción están medidas sobre las 835 originales.** Con los
845 no cambian salvo `FVT`, que pasaría de 3 a 13 de 845 porque las diez filas
nuevas **sí** se construyeron con la fórmula. Conviene que las fichas digan «sobre
las 835 originales».

**11 · `FANC` entró en el plan y ya se midió.** Ver §2.

**12 · La hoja «Auditoría MICR» va a salir.** Consolida en las fichas.

---

## 5 · Lo que queda abierto, en orden de consecuencia

1. **Que `Peh` incluya `IB`.** Es el único hallazgo tuyo que no se puede resolver
   con dato, y hoy pone electrodomésticos por encima de cuarteles policiales.
2. **Reponer `S327` y `P322`** — bloqueado por el libro abierto en Excel.
3. **La fila solar sin número** — borrarla o completarla.
4. **`FVT` híbrido** — decidir si se recalculan las 835.
5. **H-A9, mezcla de escalas** — postergado a propósito.

## 6 · Y una nota sobre el marco, que te cambia el papel

Instrucción del director:

> «La MICR intentó normalizar y sistematizar la infraestructura crítica **pero sin
> comprobar si funcionaba para todo**… ahora estamos haciendo eso, **no en el
> Word, no en el Excel, aquí**… y lo que resulte será trasladado a los dos.»

Y el criterio con el que juzgar: **«la matriz debe ser utilizable y entregar
parámetros ponderados; no es un instrumento de precisión —eso es imposible— pero
permite saber qué cosa se dañará en función de la amenaza, natural o
voluntaria».**

⇒ **Tus hallazgos no son objeciones contra el canon: son el mecanismo por el cual
el canon se corrige.** Y el criterio para priorizarlos es claro: **importa lo que
rompe el orden, no lo que mueve el tercer decimal.** Bajo esa vara, H-A7 es el
primero de la lista y H-A6 el último.
