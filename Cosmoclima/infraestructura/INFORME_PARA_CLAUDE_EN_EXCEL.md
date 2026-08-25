# Informe para Claude en Excel — estado tras la sesión del 21-ago-2026

Este documento cierra el ciclo que abriste con la auditoría de la Matriz de
Infraestructura Crítica. **Va dirigido a ti**: contiene lo que se verificó de tus
hallazgos, lo que se corrigió por ellos, dos puntos donde te quedaste corto, uno
donde te equivocaste, y las decisiones que tomó el director. Al final está lo que
cambió en los archivos que auditaste, para que no trabajes sobre supuestos viejos.

---

## 1 · Lo primero, porque invalida conteos: la Matriz ya no tiene 835 filas

**Tiene 837.** Se creó un sector nuevo, **«Protección Social»**, con dos ítems que
no existían y que el catastro sí tenía:

| ítem | elemento | activos |
|---|---|---:|
| **836** | Establecimientos para Adultos Mayores (ELEAM y Centros Diurnos) | 881 |
| **837** | Establecimientos de Protección de la Infancia | 1.533 |

Valores: `FEN` Alta · `FANC` Media · `IB` 0,80 · `VT` 0,40 y 0,30 · `FVT` 0,69 y
0,66 · `PF` 0,55 y 0,52 · `IRMD` Alto · `Pev` Media · `Peh` Baja · **`Pen` Alta**.

★ Dos cosas que conviene que sepas de estas filas:

- **Su `FVT` va CALCULADO con la fórmula, no asignado.** Son las dos únicas filas
  reproducibles de las 837.
- **Se comprobó antes de escribirlas que ninguna supera los máximos de `Pev`,
  `Peh` ni `Pen`**, así que los divisores no cambiaron y **no se reclasificó
  ninguna de las 835 anteriores**. Tu hallazgo H-A12 estaba presente en la
  decisión.

⚠️ **La rejilla se rompió, a sabiendas.** La Matriz era 19 sectores × exactamente
44 ítems (Financiero 43), contigua y sin huecos. Ahora hay un vigésimo sector con
2. Se eligió esa opción sobre meterlos en Salud porque conserva la contigüidad por
sector, que es la propiedad que de verdad se usa.

---

## 2 · Tus hallazgos, verificados contra las 835 filas reales

Yo tenía la lista completa bajada por Microsoft Graph
(`datos/micr_sharepoint.csv`), que tú no pudiste alcanzar. Resultado: **acertaste
en todo lo verificable.**

### 2.1 · Corregí dos errores míos que tú encontraste

**H-A1 · el 1,649.** Tenías razón: la suma es **1,549**. Corregido en la celda
`S327`. Y añadí tu observación, que era mejor que el error: ese 1,549 es
exactamente el divisor porque **el ítem 133 ES el máximo observado**, así que su
`Pev` normalizado vale 1,0000.

**El rango de `IB` (Importancia Base).** Escribí «valores observados entre 0,2 y
0,95» y **el mínimo real es 0,40**. Corregido en `P322`, con los once valores
presentes listados. Consecuencia que agrava tu H-A9: **la amplitud real de `IB`
es 0,55, no 1,00**, así que el término con peso nominal 0,5 tiene todavía menos
recorrido del que suponías. Tu H-A11 usaba mi 0,2; con 0,40 el mínimo normalizado
de `Pev` sube de 0,301 a 0,365.

### 2.2 · Lo que confirmé exactamente igual

`FVT` reproduce 3 de 835 · 35 combinaciones con 22 ambiguas · 812 filas afectadas
· 95 combinaciones con 18 ambiguas al agregar `IB` · `PF` 807 de 835 con error
máximo 0,0265 · `IRMD` 606 de 835 · `corr(FVT, IB)` = 0,671 y `corr(FVT, FEN)` =
0,337. Tu dato añadido `corr(FVT, VT)` = 0,629 también da idéntico.

Y tus pesos efectivos de H-A6 son exactos, término a término.

### 2.3 · Dos puntos donde te quedaste corto

**H-A3 es peor que «obsoleto»: el 1,61 es IMPOSIBLE.** El techo teórico de `Pev`
es 0,5·1 + 0,3·3 + 0,2·1 = **1,600**. Un divisor de 1,61 está por encima de lo que
la fórmula puede alcanzar. Los de `Peh` (1,94 contra techo 2,000) y `Pen` (1,87
contra 2,000) sí son sólo viejos.

**H-A12 es más grave de lo que planteaste.** No es sólo comparabilidad entre
informes. Cuantificado: si entra un ítem en el máximo teórico, cambian de nivel
**153 filas en `Pev` (18,3 %), 418 en `Peh` (50,1 %) y 173 en `Pen` (20,7 %)**.
Un solo elemento nuevo puede reclasificar la mitad de la matriz.

### 2.4 · Un punto donde te equivocaste, y la causa

**`Pen` NO está saturada.** Tu punto 6 la calculó sobre la columna publicada del
Excel, que es justamente la que está rancia. Recalculado sobre SharePoint:

| índice | en las dos bandas de arriba |
|---|---:|
| `Pev` | 796 de 835 · 95,3 % |
| `Peh` | 802 de 835 · 96,0 % |
| **`Pen`** | **167 de 835 · 20,0 %** — con 444 en Baja y 76 en Muy Baja |

Dijiste 759 de 835 para `Pen`; son 167. **`Pen` es la única de las tres que
reparte de verdad**, y se salva precisamente porque no depende de `FANC`.

### 2.5 · Matiz a tu H-A7, y tu ejemplo era mejor que el mío

Confirmé tus tres ítems exactos: **206 Dispositivos IoT `IB` 0,50 → `Peh` 0,9753 ·
635 Drones de Entrega 0,50 → 0,9753 · 353 Cuarteles Policiales `IB` 0,95 → 0,9691**.
Correlaciones idénticas a las tuyas: `Peh` con `IB` +0,356, con `FANC` +0,930, con
`VT` +0,715.

Yo había dicho que la inversión «no entierra lo crítico» porque nadie con `IB` ≥
0,85 baja de banda. Es cierto pero incompleto: **dentro de la banda superior el
orden sí está invertido**, y ahí es donde se decide a quién se protege primero.
Tu ejemplo lo muestra y el mío no.

★ Discrepancia declarada de conteo: tú das **8.080** pares invertidos, yo cuento
**22.465** con la definición «`Peh` mayor e `IB` menor en al menos 0,2» sobre
pares ordenados. Es criterio de conteo, no dirección.

### 2.6 · Tu punto 2 señaló un hueco real de mi documentación

Tienes razón: el Excel está rancio en **las tres** columnas —`Pev` 53 filas, `Peh`
101, `Pen` 680—, y yo sólo puse la advertencia en la ficha de `Pen`. Tu lectura de
que «el catálogo dice 835/835 y el archivo da 782/734/155» se explica sola: **el
835/835 está medido contra SharePoint, no contra el Excel.** Las fórmulas están
bien; lo rancio es el archivo.

---

## 3 · El hallazgo nuevo, que cambia el plan de calibración

Se usó el invierno de 2026 como experimento natural: **1.503 emergencias del
Ministerio de Obras Públicas sólo en julio**, contra 29 en enero; 908 de causa
natural en junio-agosto, todas con coordenada, en 161 comunas.

Fragilidad medida, con exposición acotada a las comunas realmente golpeadas:

| tipo | fallas | expuestos | por mil |
|---|---:|---:|---:|
| Carpeta de rodadura | 532 | 8.021 tramos | **66,33** |
| Captación de agua potable rural | 11 | 1.388 | 7,93 |
| Puentes de carreteras | 24 | 3.681 | 6,52 |
| Planta de tratamiento | 2 | 1.388 | 1,44 |
| Red matriz de agua potable | 0 | 1.388 | **0,00** |

**El episodio los separa 46 a 1. La Matriz les dice `FEN` = «Alta» a los cinco.**

### ★★★ Y de ahí sale lo importante para tu hoja de auditoría

Varianza de las tres entradas de `Pen` entre esos cinco tipos:

| entrada | varianza |
|---|---|
| **`FEN`** | **0,00000 — cero** |
| `IB` | 0,00160 |
| `FVT` | 0,00122 |

**El peso de una variable sólo se puede estimar si la variable varía.** Con `FEN`
constante, su peso 0,5 es **inestimable — y conseguir más registros de falla no lo
arregla.** Tres de los cinco tipos tienen las tres entradas idénticas (Alta, 0,9,
0,76) y tasas reales de 7,93, 1,44 y 0,00 por mil.

⇒ **Los pesos NO se pueden calibrar todavía, y el orden obligado es: medir el
`FEN` primero, pesos después.** Si tu hoja de auditoría proyecta una calibración,
conviene que refleje esto.

---

## 4 · Las decisiones que tomó el director

| decisión | resuelta así |
|---|---|
| **Unidad de activo** | **depende del tipo**: un punto se cuenta por activo, una línea por kilómetro. «No se puede usar kilómetros para un puente.» |
| **Divisor de normalización** | **no uno fijo para cosas distintas**; y se deja como está por ahora: «no es necesario complicarnos cuando recién estamos catalogando». |
| **Escala del `FEN`** | **CUATRO niveles**, la oficial de SERNAGEOMIN: Baja 0,119 · Moderada 0,339 · **Alta 0,661** · Muy Alta 0,881 |
| **Forma de `FENef`** | **PRODUCTO**, `mín(1; FEN × CClimP)`. No potencia. |
| **Ítems 836/837 y sector «Protección Social»** | **crearlos** — hecho |
| **Ley 21.663 de Ciberseguridad** | **sólo informa la Matriz, no manda**: «nadie nos manda, es nuestra matriz» |

★ **Las dos primeras decisiones se combinan bien**, y vale la pena que lo tengas:
con la escala de cuatro, `FEN` «Alta» vale 0,661, así que con la forma producto la
saturación en 1,000 cae **del 24,9 % al 1,0 %** de los pares. La disyuntiva del
techo queda prácticamente disuelta sin cambiar la fórmula del canon.

---

## 5 · Qué cambió en los archivos que auditaste

**`RMD_2_Protocolo_de_Analisis_21_08_2026.xlsx`** (antes `..._18_06_2026`):

- **329 filas** en «Variables y Métricas Completas». Las diez últimas
  (Números 319-328, categoría `MICR`) son las variables de la Matriz, que no
  estaban: `FEN` · `FANC` · `IB` · `VTic` · `FVTic` · `PF` · `IRMD` · `Pev` ·
  `Peh` · `Pen`.
- ★ **`VTic` y `FVTic` llevan sufijo a propósito**: la hoja ya tenía una sigla
  `VT/FVT` (fila 316) que es **otra cosa** — «Vulnerabilidad Técnica / Funcional
  (Capacidad de Absorción)», del bloque de Presión Demográfica, con fórmula
  `1-(DemandaServicios/CapacidadServicios)`.
- `S327` corregido (1,649 → 1,549) y `P322` corregido (rango de `IB`).
- `P320` actualizado con la escala de cuatro niveles adoptada.
- Cada ficha lleva **su tasa de reproducción medida**, para que nadie use una
  fórmula creyendo que reproduce el dato cuando no lo hace.
- Hay tres respaldos con marca de fecha junto al archivo.

**La fuente de datos que manda es SharePoint, no el Excel de la Matriz.** Lista
`mic` del sitio RMD. Si necesitas las filas, la copia bajada está en
`datos/micr_sharepoint.csv` — **pero ya está desactualizada en el conteo: trae 835
y ahora son 837.**

**Nuevo en SharePoint, por si te sirve de contexto:** 31 sub-matrices con **96.423
activos georreferenciados**, verificados fila por fila contra su origen, cada uno
atado a su ítem de la Matriz por una columna de búsqueda.

---

## 6 · Lo que quedó abierto

- **Los pesos de `Pev`, `Peh` y `Pen` siguen sin calibrar**, por lo del punto 3.
- Para `Pev` y `Peh` se propusieron fuentes de proxy —Protocolo I Adicional a los
  Convenios de Ginebra para el conflicto convencional, Ley 21.663 para el
  irregular—, **sin implementar**. Y con una advertencia: el proxy sirve primero
  para **re-medir `FANC`**, que dice «Alta» en 802 de 835 y por eso tiene el mismo
  problema de varianza cero que el `FEN`.
- La hoja «Auditoría MICR» que agregaste sigue en el archivo. El director dijo que
  la sacará después.
