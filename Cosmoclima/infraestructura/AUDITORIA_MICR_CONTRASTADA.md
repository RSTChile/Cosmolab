# La auditoría de Claude en Excel, contrastada contra las 835 filas

**21-ago-2026.** Claude en Excel auditó las diez fichas de la Matriz de
Infraestructura Crítica (MICR) que se escribieron hoy en el Protocolo, y produjo
13 hallazgos **por vía algebraica**, sin acceso a los datos: el enlace a
SharePoint le devolvió `url_not_allowed` porque está tras autenticación del
tenant.

Yo sí tengo las 835 filas (`datos/micr_sharepoint.csv`, bajadas hoy por Microsoft
Graph). Este documento las contrasta una por una. **Resumen: la auditoría acierta
en todo lo que afirma, incluido un error mío; y sus tres hipótesis sin comprobar
quedan confirmadas —dos de ellas con consecuencias peores de lo que suponía.**

---

## 1 · H-A1 · El error aritmético es mío y está corregido

La ficha de `Pev` traía `0,5·0,95 + 0,3·3 + 0,2·0,87 = 1,649`. La suma correcta
es **1,549** (0,475 + 0,900 + 0,174). Escribí un 6 donde iba un 5.

**Y la observación que lo acompaña es más fina que el error:** 1,549 es
exactamente el divisor de normalización de `Pev`. No es coincidencia — significa
que **el ítem 133 (Reactores Nucleares) ES el máximo observado de las 835 filas**,
y por lo tanto su `Pev` normalizado vale 1,0000 exacto. Verificado.

Corregido en la celda `S327` del Protocolo, con esa observación añadida.
Respaldo: `RMD_2_Protocolo_de_Analisis_21_08_2026.RESPALDO_pre_correccion_HA1.xlsx`.

## 2 · H-A3 · El 1,61 del Word no está obsoleto: es imposible

Verificado, y la distinción importa:

| índice | máximo **teórico** | divisor del Word | máximo observado |
|---|---:|---:|---:|
| **`Pev`** | **1,600** | **1,61** ★ | 1,549 |
| `Peh` | 2,000 | 1,94 | 1,944 |
| `Pen` | 2,000 | 1,87 | 1,959 |

Con `IB ≤ 1`, `FANC_num ≤ 3` y `FVT ≤ 1`, el techo de `Pev` es
`0,5 + 0,9 + 0,2 = 1,600`. **El 1,61 del Word está por encima del máximo que la
fórmula puede alcanzar.** No es un dato viejo: es un número que no puede existir.
Yo lo había tratado como «desactualizado», que se queda corto. Los de `Peh` y
`Pen` sí son simplemente viejos.

## 3 · H-A7 · `Peh` ignora la importancia — y ahora hay lista de nombres

Confirmado algebraicamente: `Peh = 0,5·FANC + 0,3·VTic + 0,2·FVTic`, y `FVTic` no
contiene `IB`. **El peso efectivo de la importancia en `Peh` es cero.**

La auditoría lo dejó como hipótesis: *«un activo irrelevante pero muy tecnificado
supera a uno crítico poco tecnificado»*. Buscado en los datos, **existe, y son
seis**:

| `Peh` | nivel | `IB` | elemento |
|---:|---|---:|---|
| 0,975 | **Muy Alta** | 0,50 | **Dispositivos IoT (Electrodomésticos)** |
| 0,975 | **Muy Alta** | 0,50 | Drones de Entrega |
| 0,964 | **Muy Alta** | 0,50 | Inversores Residenciales |
| 0,964 | **Muy Alta** | 0,50 | Puntos de Acceso Wi-Fi (Domésticos) |
| 0,964 | **Muy Alta** | 0,50 | **Proyectores y Pizarras Digitales** |
| 0,929 | Alta | 0,50 | Paneles Solares (Uso Residencial) |

Los electrodomésticos conectados a internet y las pizarras digitales de los
colegios salen con **prioridad Muy Alta frente a sabotaje y ciberataque**.

**Matiz que corrige a la auditoría en un punto:** la inversión es de una sola
dirección. Buscado el caso simétrico —activo crítico (`IB ≥ 0,85`) hundido a
`Peh` Media o menos— hay **cero**. O sea, el defecto **infla lo irrelevante, no
entierra lo crítico**. La cabeza del ranking de `Peh` es sensata: Reactores
Nucleares, Centros de Datos, SCADA, Unidades de Cuidados Intensivos, Centros de
Comando y Control.

**★ Y hay algo peor que la inversión, que no está en la auditoría:** `Peh` tiene
**435 «Muy Alta» y 367 «Alta» de 835 — el 96 % en los dos niveles de arriba**, y
sólo 33 elementos por debajo. No es que `Peh` ordene mal: es que **casi no
ordena**. Los seis domésticos no son una anomalía puntual, son el síntoma
visible de una columna saturada, y la causa es la misma de siempre: `FANC` pesa
0,5 y dice «Alta» en 802 de 835 filas.

## 4 · H-A6 · El doble conteo es correcto, pero es el hallazgo más chico

La descomposición algebraica es exacta, la verifiqué término a término:

| índice | ef `FEN` | ef `FANC` | ef `IB` | ef `VTic` |
|---|---:|---:|---:|---:|
| `Pev` | 0,0222 | 0,3222 | 0,5000 | 0,0667 |
| `Peh` | 0,0222 | 0,5222 | **0** | 0,3667 |
| `Pen` | 0,5222 | 0,0222 | 0,3000 | 0,0667 |

**Pero hay que ponerle dos advertencias:**

**(a) La descomposición supone que `FVT = (FEN+FANC+VTic·3)/9`, y el `FVT`
publicado no sigue esa fórmula** (reproduce 3 de 835). Medido: el publicado tiene
media 0,699 y el de fórmula 0,805, y correlacionan +0,810 — o sea que **sólo dos
tercios de la varianza del `FVT` publicado son sus tres entradas**. El doble
conteo existe en esa proporción, no entero.

**(b) El término del `FVT` casi no mueve nada.** Medido sobre las 835 filas, su
desviación típica dentro de cada índice es **0,0134**, contra 0,0611 a 0,2642 de
los términos principales:

| índice | desv. total | términos, por cuánto mueven de verdad |
|---|---:|---|
| `Pev` | 0,0974 | `FANC` **0,0611** · `IB` 0,0456 · `FVT` 0,0134 |
| `Peh` | 0,1391 | `FANC` **0,1018** · `VTic` 0,0487 · `FVT` 0,0134 |
| `Pen` | 0,2790 | `FEN` **0,2642** · `IB` 0,0273 · `FVT` 0,0134 |

Y probado en el orden real: quitar por completo el término del `FVT` deja
Spearman **0,932 · 0,975 · 0,962**. El ranking casi no se mueve.

## 5 · H-A9 · La mezcla de escalas es el hallazgo de fondo, y se confirma con dato

Ésta sí es grave, y la tabla de arriba la demuestra sin álgebra: **en `Pev`, el
término con peso nominal 0,3 (`FANC`) mueve más que el de peso nominal 0,5
(`IB`)** — 0,0611 contra 0,0456. El término declarado dominante no es el
dominante, y eso pasa con `FANC` fijo en «Alta» en el 96 % de las filas; con una
`FANC` que variara de verdad la distorsión sería mucho mayor.

En `Pen` es extremo y coincide con lo que ya habíamos medido por otra vía: `FEN`
aporta 0,2642 de una desviación total de 0,2790 —el **95 %**— y correlaciona
**+0,992** con el índice. `Pen` **es** `FEN` con adornos. De ahí la partición en
bandas.

## 6 · ★ H-A12 · La no estacionariedad es el hallazgo más grave, y la auditoría se quedó corta

Normalizar por el máximo observado significa que **agregar una sola fila reescala
las 835**. La auditoría lo plantea como riesgo de comparabilidad entre informes.
Cuantificado, es peor:

| índice | si entra un ítem en el **máximo teórico** | si entra uno 5 % sobre el máximo actual |
|---|---:|---:|
| `Pev` | **153 de 835 cambian de nivel · 18,3 %** | 246 · 29,5 % |
| **`Peh`** | **418 de 835 · 50,1 %** | 444 · 53,2 % |
| `Pen` | 173 · 20,7 % | 247 · 29,6 % |

**Un solo elemento nuevo puede reclasificar la mitad de la matriz.** Y esto no es
hipotético: el sub-proyecto de sub-matrices existe justamente para hacer crecer
la Matriz. `Peh` es el más frágil porque su máximo observado (1,944) ya está casi
pegado al techo teórico (2,000), así que cualquier ítem nuevo con `FANC=Alta` y
`VTic` alto lo empuja.

La instrucción que yo dejé escrita —«recalcular el divisor y declararlo como
recálculo»— es necesaria pero **no basta**: no arregla que los informes viejos
dejen de ser comparables.

---

## 7 · Mi lectura de conjunto

**La auditoría acierta en todo lo verificable, incluido un error mío, y en dos
puntos se queda corta por prudencia** (H-A3 es peor que «obsoleto»; H-A12 es peor
que un problema de comparabilidad).

**Reordenaría la severidad así**, ahora con dato:

| | hallazgo | por qué |
|---|---|---|
| 1 | **H-A12 · no estacionariedad** | un ítem nuevo reclasifica hasta el 50 %, y el proyecto existe para agregar ítems |
| 2 | **H-A9 · mezcla de escalas** | los pesos declarados no son los reales; confirmado con desviaciones típicas |
| 3 | **saturación de `FANC` y de `Peh`** | 802 de 835 «Alta» · 96 % de `Peh` en los dos niveles de arriba |
| 4 | **H-A7 · `Peh` sin importancia** | 6 inversiones concretas, pero de una sola dirección |
| 5 | **H-A5 · `FVT` no reproducible** | 812 de 835 filas afectadas |
| 6 | H-A6 · doble conteo | correcto pero chico: Spearman 0,93-0,98 al quitarlo |
| 7 | H-A8 · contaminación cruzada | real pero de peso 0,0222 |

**Sobre la recomendación final de la auditoría —decidir primero las escalas y
después los pesos, normalizando las ordinales a 0-1 antes de ponderar— estoy de
acuerdo, y es exactamente lo que el proyecto ya venía haciendo por otro camino:**
la curva logística de `normalizar.py` (Baja 0,119 · Media 0,500 · Alta 0,881) es
la que se usó para el FEN medido con las 6.141 emergencias del Ministerio de
Obras Públicas.

**Añadiría una condición que la auditoría no menciona y que H-A12 vuelve
obligatoria: el divisor de normalización no puede ser el máximo observado.** Si
las entradas están todas en 0-1 y los pesos suman 1, el índice ya vive en 0-1 por
construcción y no hace falta dividir por nada. Eso resuelve H-A9 y H-A12 de una
vez, y hace que dos informes de distinta fecha sean comparables.

## 8 · Lo que NO se tocó

Sólo se corrigió el error aritmético de `S327`, que era mío e inequívoco.
**Las limitaciones nuevas verificadas en este documento —H-A7 con sus seis
ítems, H-A12 con sus porcentajes, la saturación de `Peh`— NO se escribieron en
las fichas del Protocolo**, porque cambiar el archivo oficial con hallazgos de
método es decisión del director.

Relacionado: `formalizar_micr_en_protocolo.py` ·
`FUENTE_SHAREPOINT_RMD.md` · `SUBMATRICES_Y_EL_FEN_CONSTANTE.md` ·
`FEN_MEDIDO.md`

---

# ANEXO · Segunda ronda: Claude en Excel sobre el Excel de la MICR (21-ago)

Ahora auditó el **Excel de la MICR** (la copia local, no SharePoint). Verificado
todo contra `datos/micr_sharepoint.csv`.

## A1 · Segundo error mío, corregido

La ficha de `IB` decía «valores observados entre 0,2 y 0,95». **El mínimo real es
0,40.** Medido: los once valores presentes son 0,40 · 0,50 · 0,55 · 0,60 · 0,65 ·
0,70 · 0,75 · 0,80 · 0,85 · 0,90 · 0,95. Corregido en `P322`, y se añadió la
consecuencia: **la amplitud real de `IB` es 0,55 y no 1,00**, lo que agrava
H-A9 — el término de peso nominal 0,5 tiene todavía menos recorrido del que
parecía. Él propagó ese 0,2 a su H-A11; con 0,40 el mínimo normalizado de `Pev`
sube de 0,301 a 0,365.

## A2 · Su mejor ejemplo de H-A7, verificado exacto

| ítem | elemento | `IB` | `Peh` |
|---:|---|---:|---:|
| 206 | Dispositivos IoT (Electrodomésticos) | 0,50 | **0,9753** |
| 635 | Drones de Entrega | 0,50 | **0,9753** |
| **353** | **Cuarteles Policiales** | **0,95** | **0,9691** |

Es mejor que el mío porque es una comparación directa contra un activo
genuinamente crítico. **Y corrige un matiz que yo había puesto:** yo dije que la
inversión «no entierra lo crítico» porque ningún activo con `IB ≥ 0,85` baja de
banda. Es cierto, pero incompleto — **dentro de la banda superior el orden sí
está invertido**, y ahí es donde se decide a quién se protege primero.

Correlaciones verificadas idénticas a las suyas: `corr(Peh, IB) = +0,356` ·
`corr(Peh, FANC) = +0,930` · `corr(Peh, VTic) = +0,715`.

★ **Discrepancia en un conteo, declarada:** él reporta **8.080** pares
invertidos; yo cuento **22.465** con la definición «`Peh` mayor y `IB` menor en
al menos 0,2», sobre pares ordenados. La diferencia es de criterio de conteo, no
de dirección: el hallazgo se sostiene con las dos cifras.

## A3 · ★ Su punto 6 (saturación) hay que corregirlo: `Pen` NO está saturada

Él calcula la saturación sobre **las columnas publicadas del Excel rancio**, y
para `Pen` esa columna es justamente la que está mal. Recalculado sobre
SharePoint:

| índice | en las dos bandas de arriba | reparto completo |
|---|---:|---|
| `Pev` | **796 de 835 · 95,3 %** | Muy Alta 81 · Alta 715 · Media 8 · Baja 30 · Muy Baja 1 |
| `Peh` | **802 de 835 · 96,0 %** | Muy Alta 435 · Alta 367 · Baja 32 · Muy Baja 1 |
| **`Pen`** | **167 de 835 · 20,0 %** | Muy Alta 100 · Alta 67 · Media 148 · **Baja 444 · Muy Baja 76** |

Él dice 759 de 835 para `Pen`; con el dato bueno son 167. **`Pen` es la única de
las tres que reparte de verdad**, y la que él daba por saturada. La saturación es
un problema de `Pev` y `Peh`, no de las tres — y su causa está identificada:
`FANC` pesa 0,5 en `Peh` y 0,3 en `Pev`, y dice «Alta» en 802 de 835 filas.
`Pen` se salva justamente porque **no** depende de `FANC`.

## A4 · Su punto 2 es correcto y señala un hueco de mi documentación

El Excel local está rancio en **las tres** columnas, no sólo en `Pen`:
`Pev` 53 filas · `Peh` 101 · `Pen` 680. Coincide exactamente con lo que yo había
medido ayer en `FUENTE_SHAREPOINT_RMD.md`. **Pero la advertencia sólo la escribí
en la ficha de `Pen` (`Q329`), no en las de `Pev` y `Peh`.** Es un hueco real de
documentación y conviene cerrarlo.

Su lectura de que «el catálogo dice 835/835 y el archivo da 782/734/155» tiene
una explicación simple: el 835/835 está medido contra **SharePoint**, no contra
el Excel. Las fórmulas están bien; lo rancio es el archivo.

## A5 · Lo que confirma de la auditoría interna

Reprodujo de forma independiente y le dio idéntico: `FVT` 3/835 · 35
combinaciones con 22 ambiguas · 812 filas afectadas · 95 combinaciones con 18
ambiguas al agregar `IB` · `PF` 807/835 con error máximo 0,0265 · `IRMD` 606/835
· `corr(FVT, IB) = 0,671` y `corr(FVT, FEN) = 0,337`.

Y **añade un dato que refuerza H-A5**: `corr(FVT, VTic) = 0,629`. Verificado.
O sea que el `FVT` se asignó pesando **importancia (0,671) y tecnificación
(0,629), y no fragilidad natural (0,337)** — que es exactamente lo contrario de
lo que su nombre promete.

## A6 · Su recomendación operativa, que comparto

*«Calibrar pesos sobre este Excel daría pesos que reproducen columnas rancias en
tres de sus salidas.»* Correcto. **La calibración tiene que hacerse sobre
`datos/micr_sharepoint.csv`**, que es la bajada de hoy por Microsoft Graph, no
sobre el Excel homologado.
