# Validación de CClimP y FENef — resultado

**20-ago-2026.** Corrido con `validar_cclimp.py --csv`. Los tres criterios
estaban escritos en `VARIABLES_Y_METRICAS_PROYECTO.md` §2 desde el 16-ago, antes
de calcular nada. Ninguno se movió para que pasara.

**Veredicto: 2 de 3. El criterio 3 NO PASA, y el fallo es de cómo escribí la
prueba, no del instrumento. Lo explico entero abajo, y no lo corrijo por mi
cuenta.**

Código: `cclimp.py` (implementación) · `validar_cclimp.py` (prueba).
Salida: `datos/fenef_39.csv`, 17.160 filas.

---

## 1 · Qué se implementó

```
PelPre  →  CClimP  →  FENef
(mide)     (perilla)  (fragilidad del activo, este mes, en este lugar)
```

En simple: la Matriz sabe que una subestación eléctrica es frágil, pero no sabe
que **ésta**, **este mes**, está bajo un temporal. `CClimP` es el volumen: queda
en 1 cuando el mes es normal y no cambia nada; sube a 1,6 cuando el mes está
entre el 1 % peor del país en 36 años. `FENef` es el resultado de subirle el
volumen a la fragilidad de fábrica.

**Añadido que faltaba y quedó declarado:** el canon MACC exige bloquear el
coeficiente «si la confianza del dato es baja», pero no dice dónde está el corte.
Se fija en **0,60**, que con la confianza base de ERA5 (0,70) equivale a exigir
al menos un 86 % de los días del mes con dato. Un mes al que le faltan cuatro
días puede haberse perdido justo el temporal. **22 pares activo-mes** habrían
movido la perilla y no se les creyó.

---

## 2 · Criterio 1 — la perilla se queda quieta · ✓ PASA

| | |
|---|---|
| `CClimP = 1,0` | **75,1 %** de los 17.160 pares activo-mes |
| mes mediano del país | **87,2 %** de las subestaciones quietas |
| exigido | > 70 % |

Reparto: `1,0` 12.895 (75,1 %) · `1,2` 2.559 (14,9 %) · `1,4` 1.533 (8,9 %) ·
`1,6` 173 (1,0 %).

**★ Honestidad sobre este criterio: pasa casi por aritmética.** Los cortes *son*
los percentiles 75/90/99 de la distribución nacional, así que que el 75 % quede
neutro no es un hallazgo, es la definición. Lo único que este criterio comprueba
de verdad es que la distribución **mes a mes** no se desarma — que no existan
meses en que todo el país se mueve a la vez. Eso sí lo mide la línea del mes
mediano, y da 87,2 %.

---

## 3 · Criterio 2 — se mueve donde hay deslizamientos · ✓ PASA

Sobre los 152 meses-con-deslizamiento del registro ReTeRM de SERNAGEOMIN, en 91
puntos que **no participaron en fijar los cortes**:

| | perilla movida |
|---|---|
| meses **con** deslizamiento | **118 de 152 · 77,6 %** |
| meses **sin** deslizamiento | 11.207 de 39.888 · **28,1 %** |
| exigido | > 50 % |

Reparto en los meses con evento: `1,0` 34 · `1,2` 33 · `1,4` 56 · `1,6` 29.

**El contraste es lo que le da valor**: 77,6 % contra 28,1 % es una razón de 2,8
a 1. Si la perilla se moviera igual con y sin evento, no estaría midiendo nada.

Y el traslado importa: los cortes se midieron sobre las 39 subestaciones y acá se
aplican a 91 puntos distintos. Eso convierte el criterio en una prueba y no en
una comprobación de sí mismo.

---

## 4 · Criterio 3 — reordenar en temporal, no reordenar en calma · ✗ NO PASA

El criterio pedía: *«el ordenamiento que produce `FENef` debe cambiar respecto
del orden fijo de la matriz en al menos un mes de temporal documentado, y **no**
cambiar en un mes tranquilo»*.

Las 39 subestaciones comparten hoy una sola fila de la Matriz (ítem 120), o sea
que el «orden fijo» es un **empate de 39**. Con esa base la medida es exacta:
de los 741 pares posibles, ¿cuántos dejan de estar empatados?

| mes | perilla movida | pares desempatados (escala de 3, potencia) |
|---|---|---|
| 2015-03 · aluvión de Copiapó | 9 de 39 | 297 de 741 |
| 2026-07 · el temporal del proyecto | 28 de 39 | 537 de 741 |
| **2015-08 · «mes tranquilo»** | **29 de 39** | **514 de 741** |

**Reordena en los dos meses de temporal (✓) y reordena también en el mes que
declaré tranquilo (✗). Falla.**

### Por qué falla: el mes que elegí no es tranquilo

Fijé `2015-08` como mes tranquilo antes de calcular, y la razón fue mala: lo
tomé de **el ancla de Copiapó**, donde agosto de 2015 es efectivamente el mes
sin nada (`PelPre = 0,5805`, `CClimP = 1,0`, medido). Pero el ancla es *un
lugar*, y el criterio se evalúa sobre *el país*. Agosto es pleno invierno en
Chile: en el centro-sur ese mes hubo temporal de verdad, y 29 de 39
subestaciones se movieron con razón.

**Confundí «tranquilo en Copiapó» con «tranquilo en Chile».**

### ★ El fondo del problema: en un país de 4.300 km, casi no hay meses tranquilos

Medido sobre los 440 meses de la serie:

| | |
|---|---|
| meses con las **39 quietas** | **49 de 440 · 11 %** |
| enero · febrero · diciembre | ~93 % de subestaciones quietas en promedio |
| junio · julio | **42 % y 52 %** |

O sea: el criterio **sí es comprobable**, pero sólo con un mes de verano. Un mes
de invierno nunca va a ser nacionalmente tranquilo, porque mientras el norte está
seco el sur está bajo agua. El criterio se escribió pensando en un punto y se
aplica a un país.

**No cambio el mes por mi cuenta.** Elegir otro mes después de ver el fallo es
exactamente lo que el protocolo prohíbe. La decisión es del director, y las
opciones están en §7.

---

## 5 · ★ Lo que el instrumento encontró solo, y que nadie le dijo

Esto no era parte de ningún criterio. Sale de mirar **qué** subestaciones se
mueven en cada mes.

**Marzo 2015 — el aluvión de Copiapó.** Se movieron 9 de 39, y son éstas:

| perilla | subestación | región |
|---|---|---|
| **1,6** | Copiapó (Urbana) | Atacama |
| **1,6** | Maitencillo (Rural) | Atacama |
| **1,6** | Escondida (Rural) | Antofagasta |
| 1,4 | La Serena · Ovalle | Coquimbo |
| 1,4 | Nueva Pozo Almonte | Tarapacá |
| 1,2 | Coyhaique · Puerto Aysén · Puerto Natales | Aysén / Magallanes |

**Las tres al máximo son exactamente la zona del aluvión**, y las de 1,4 son las
regiones vecinas que también recibieron el sistema frontal. Nadie le dijo al
instrumento dónde había sido: se lo encontró en la serie completa de 36 años.

**Julio 2026 — el temporal que motivó el proyecto.** Se movieron 28 de 39, y las
once que llegan a 1,6 son Valparaíso, Quillota, La Ligua, San Bernardo, Pirque,
Talca, Curicó, Illapel, Ovalle, La Serena y Maitencillo: **el centro de Chile**,
que es donde efectivamente ocurrió.

Los dos eventos quedan separados geográficamente, con la misma medida y sin
ningún ajuste entre uno y otro. Es la señal más fuerte de esta tanda.

---

## 6 · La disyuntiva del techo, con números

La ficha de FENef dejó abierta la forma, textualmente. Medido sobre los 17.160
pares:

| escala del FEN | forma | pares que topan en 1,000 | ¿distingue 1,4 de 1,6? |
|---|---|---|---|
| tres (FEN = 0,881) | producto | **4.265 · 24,9 %** | **no, indistinguibles** |
| tres (FEN = 0,881) | potencia | 0 · 0,0 % | sí, +0,0176 |
| cuatro (FEN = 0,661) | producto | 173 · 1,0 % | sí, +0,0749 |
| cuatro (FEN = 0,661) | potencia | 0 · 0,0 % | sí, +0,0428 |

- **`producto`** es `mín(1 ; FEN × CClimP)` — lo que MACC dice literalmente.
- **`potencia`** es `1 − (1 − FEN) ^ CClimP` — no satura nunca y con `CClimP = 1`
  devuelve exactamente el FEN base.

**★ Corrección a la ficha:** decía que el techo sólo mordía con FEN = 0,881.
Medido, con la escala de cuatro también satura — 0,661 × 1,6 = 1,058, que se
corta a 1,000. La saturación no depende de la escala, sólo de en qué escalón
empieza.

**Lo que se pierde con `producto` y la escala de tres es grave**: una cuarta
parte de todos los pares queda pegada en 1,000, y `1,4` y `1,6` se vuelven el
mismo número — o sea, el instrumento deja de distinguir «mes muy malo» de «mes
peor de la historia» justo en los activos más frágiles. En el mes del aluvión,
Copiapó marca 1,0000 igual que las tres subestaciones que sólo llegaron a 1,4.

---

## 7 · Las tres decisiones que quedan para el director

**Ninguna la tomo yo.** Las tres están implementadas con sus alternativas y se
pueden correr en cualquier momento.

### 7.1 · Qué hacer con el criterio 3

| opción | qué implica |
|---|---|
| **(a)** Dejar el fallo registrado y correr una segunda prueba con un mes tranquilo elegido por una regla independiente, declarada de antemano y etiquetada como **segunda prueba**, no como reemplazo | conserva la honestidad del protocolo; deja las dos pruebas visibles |
| **(b)** Reescribir el criterio: «no reordenar en calma» se evalúa **por activo**, no por mes del calendario | se cumple por construcción (`CClimP = 1` → `FENef = FEN`) y por eso mismo prueba poco |
| **(c)** Aceptar el fallo tal cual y no validar `CClimP` | lo más estricto; bloquea `FENef` y con él `PFef` |

### 7.2 · Qué escala usar para el FEN de la Matriz

La MICR trae **tres** niveles: `Media` 592 · `Alta` 167 · `Baja` 76, y **ni una
sola fila dice «Muy Alta»**. El ítem 120 dice `Alta`. La ficha de FENef habla de
cuatro (H-15) y su ejemplo calcula con 0,881 llamándolo «Muy Alta» — que es la
lectura de tres con el nombre de la de cuatro.

| lectura | Baja | Media | Alta | ítem 120 |
|---|---|---|---|---|
| por posición, escala de 3 (la que la MICR usa) | 0,119 | 0,500 | 0,881 | **0,881** |
| por nombre, escala de 4 (H-15) | 0,119 | 0,339 | 0,661 | **0,661** |

La escala de cuatro es de SERNAGEOMIN y describe el **peligro de un lugar**; la
de tres es de la MICR y describe la **fragilidad de un tipo de elemento**. Meter
la fragilidad en la escala del peligro le baja el valor a todo sin que ningún
dato lo respalde. Pero la decisión H-15 está tomada y es del director.

### 7.3 · La forma de FENef

`producto` (literal del canon, satura) o `potencia` (no satura, conserva el
neutro). Los números están en §6.

---

## 8 · Lo que esta validación NO probó

1. **La dimensión de tipo.** Las 39 subestaciones son todas del mismo tipo, así
   que comparten el mismo FEN base. Todo lo medido acá prueba que `FENef`
   distingue **lugares**; no puede probar que distinga **tipos**, porque no hay
   más de uno en la muestra. Hace falta un conjunto multi-sector con clima
   bajado, que hoy no existe.
2. **El universo grande.** Los cortes siguen saliendo de 39 subestaciones
   (limitación H-10). Anoche el inventario pasó a 1.226 subestaciones con
   coordenada y más de 130.000 activos. Recalcular los cortes sobre ese universo
   es un **recálculo declarado**, no una corrección, y cambia los tres números.
3. **Nada del norte árido.** Se hereda entera la limitación del conjunto ReTeRM:
   23 de 329 eventos son del norte, el 7 %.

Relacionado: `VARIABLES_Y_METRICAS_PROYECTO.md` · `CORRECCION_RAREZA_PELIGRO.md`
· `INVENTARIO_GEORREFERENCIADO.md`
