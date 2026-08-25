# El FEN recalculado con dato real, elemento por elemento

**20-ago-2026.** Instrucción de Alexis: *«simplemente hay que recalcular el FEN
para cada elemento de la submatriz, eso es lo que manda… el modelo se hizo en
abstracto, sin datos, sin localidades, sin infraestructura real»*.

Código: `recalcular_fen_medido.py` · Salida: `datos/fen_medido_por_elemento.csv`
Fuente: `datos/mop_emergencias_viales.csv`, **6.141 emergencias reales del
Ministerio de Obras Públicas, 2014-2026**.

**Las seis reglas se fijaron ANTES de calcular** y están escritas en la cabecera
del script. Aquí va el resultado, y también los tres defectos que encontré en mi
propia primera corrida y corregí antes de reportar.

---

## 1 · Qué es un FEN medido

```
tasa =  eventos de causa natural que le pasaron a ese tipo
        ────────────────────────────────────────────────
                cuántos activos de ese tipo existen
```

De cada mil sistemas de agua potable rural, ¿a cuántos les falló la captación
por causa del clima? ¿y el estanque? El que falla más es más frágil. No hay que
opinarlo.

**Y la escala pasa a significar algo.** Se traduce la tasa a la etiqueta con la
curva común del proyecto, tomando como referencia la mediana del grupo:

| tasa del elemento | etiqueta |
|---|---|
| 4 veces la mediana de su grupo | **Alta** (0,881) |
| igual a la mediana | **Media** (0,500) |
| un cuarto de la mediana | **Baja** (0,119) |

Es decir: **«Alta» pasa a querer decir «falla cuatro veces más que el elemento
típico de su grupo»**, y deja de querer decir «nos pareció que era alta».

---

## 2 · El resultado limpio: agua potable rural

Éste es el mejor experimento natural que tiene el registro, y por una razón
concreta: captación, red matriz, planta de tratamiento, estanque y arranque son
**componentes de los MISMOS 2.475 sistemas**, reportados por el **mismo
servicio**, en la **misma ventana de tiempo**. El denominador es idéntico para
todos y por lo tanto se cancela: la razón entre ellos es fragilidad y nada más.

| elemento | fallas | de causa natural | tasa ×1000 | **FEN medido** | Matriz dice |
|---|---:|---:|---:|---|---|
| **Captación** | 149 | **40** | 16,162 | **0,987 · Alta** | Alta |
| Planta de tratamiento (#17) | 28 | 11 | 4,444 | **0,921 · Alta** | Alta |
| Red matriz (#16) | 57 | 8 | 3,232 | **0,881 · Alta** | Alta |
| Red de distribución | 14 | 2 ⚠ | 0,808 | 0,500 · Media | Alta |
| Conducción | 11 | 2 ⚠ | 0,808 | 0,500 · Media | Alta |
| Arranque | 4 | 2 ⚠ | 0,808 | 0,500 · Media | Alta |
| Bocatoma | 5 | 1 ⚠ | 0,404 | 0,269 · Baja | Alta |
| **Estanque de regulación** | 4 | **0** ⚠ | 0,000 | **0,000 · Baja** | Alta |

⚠ = muestra fina (menos de 5 eventos naturales): la tasa se apoya en tan pocos
casos que uno más la cambiaría de etiqueta. Se marca, no se esconde.

**Cambian 5 de 8.** Y el resultado central no depende de ninguna normalización:
en los mismos 2.475 sistemas, **la captación acumuló 40 fallas por clima y el
estanque de regulación ninguna**. La Matriz dice que los dos son «Alta».

**En simple:** la parte del sistema que está metida en el río —la captación— es
la que el clima rompe. El estanque, que está en tierra firme y cerrado, no. Es
obvio una vez dicho, y sin embargo la Matriz no lo sabía, porque la Matriz se
escribió sin mirar ninguna falla real.

---

## 3 · Carreteras y puentes: la medida sí, la etiqueta no

| elemento | fallas | de causa natural | denominador | tasa ×1000 |
|---|---:|---:|---:|---:|
| Carpeta de rodadura (#616) | 2.528 | **1.581** | 14.039 tramos | 112,6 |
| Puente (#618) | 422 | **130** | 6.742 puentes | 19,3 |

**No se emite etiqueta para este grupo**, y la razón está en §5. Lo que sí se
puede afirmar es la razón entre los dos — y aquí aparece un problema conceptual
que vale más que el número:

| unidad elegida | la calzada falla … veces más que un puente |
|---|---|
| por **tramo** | **5,84 a 1** |
| por **100 km** | **100,67 a 1** |

**Un puente es un punto y una carretera es una línea.** «Un activo» no significa
lo mismo para los dos, y la razón entre ellos cambia diecisiete veces según lo
que se elija. El script no elige: muestra las dos. Es decisión del director, y
hay que tomarla antes de que este número entre en la Matriz.

---

## 4 · Lo que queda medido pero sin denominador

Existen las fallas; no existe el inventario de cuántos hay. Se reporta el conteo
y se marca, nunca se les inventa un denominador:

| elemento | fallas | naturales |
|---|---:|---:|
| Elementos de saneamiento (vialidad) | 175 | 96 |
| Pavimento de camino de acceso | 145 | 59 |
| Colector | 32 | 17 |
| **Enrocado** | 23 | **21 · el 91 %** |
| Cauce receptor | 12 | 8 |
| Gavión | 7 | 6 |
| Control aluvional | 4 | 4 |

★ Las obras de defensa fluvial —enrocados, gaviones, control aluvional— casi
sólo fallan por causa natural, que es exactamente lo que uno esperaría de algo
construido para contener un río. **Chile no tiene catastro nacional de defensas
fluviales**, así que no se les puede calcular tasa. Queda como hueco declarado.

---

## 5 · Los tres defectos que tuvo mi primera corrida

Los anoto porque el resultado publicado depende de haberlos corregido.

**5.1 · Un denominador que no correspondía.** Le había asignado a «pavimento de
camino de acceso» el inventario de la red vial enrolada (14.039 tramos). Los
caminos de acceso no están ahí adentro: el divisor quedaba inflado y el elemento
salía «Baja» por un error de contabilidad. Ahora va sin denominador.

**5.2 · La etiqueta era un artefacto en los grupos chicos.** Con la mediana del
grupo como referencia, si el grupo tiene tres elementos **el del medio siempre
sale «Media»**, por construcción y no por medición. Con uno solo, el resultado
es puro artefacto. En la primera corrida eso hizo que «Puente» saliera «Media» y
que la única estación de la DGA saliera «Media» también — las dos, sin que el
dato dijera nada. Regla nueva (R5-bis): **sólo se etiqueta un grupo con al menos
cinco elementos medibles.** En los demás se reporta la tasa y la razón, que sí
son medidas. Se prefiere no responder a responder con un artefacto.

**5.3 · El «techo» del 47 % era degenerado, y por culpa de la fuente.** Había
diseñado una cota superior repartiendo los casos sin causa. No funciona, y la
razón es un hallazgo sobre el registro: **su clasificación de causa junta «la
causa fue otra» con «no se anotó la causa» en un solo valor** (`otra_o_no_dice`,
2.903 de 6.141). Como no existe ni un solo caso de «causa declarada NO natural»,
el techo aritmético es «todas las fallas fueron naturales» — que no informa
nada. Se reporta como cota trivial y marcada, y el hueco del 47 % queda
**acotado, no resuelto**.

---

## 6 · Lo que esto NO cubre todavía

1. **Sólo infraestructura del Ministerio de Obras Públicas.** Subestaciones,
   hospitales, escuelas, relaves y telecomunicaciones no están en este registro
   y **siguen con el FEN heredado**. Para energía existe `sec_cortes.csv`
   (304.419 cortes con comuna y hora, 2019-2026): es otro registro, con otro
   denominador y otra unidad, y necesita su propio tratamiento y sus propias
   reglas fijadas antes.
2. **No se puede comparar entre grupos.** Vialidad reporta 4.544 emergencias y
   agua potable rural 948: eso mide cuánto reporta cada servicio, no cuánto
   falla cada cosa.
3. **El registro tiene huecos de tiempo declarados**: 2020 y 2021 no existen,
   2014 trae 2 casos y 2022 trae 3. Por eso no se calcula ninguna tasa «por
   año». La comparación entre tipos sigue siendo válida porque el hueco es el
   mismo para todos.
4. **El 38 % de las emergencias no identifica el elemento afectado** (vacío u
   «Otro Elemento»). Quedan fuera del cálculo y no se reparten.
5. **Sólo mide fragilidad ante lo natural.** El `FANC` —fragilidad ante ataques
   no convencionales, que dice «Alta» en el 96 % de la Matriz— sigue intacto y
   este registro no puede tocarlo: 21 de sus 22 modos de falla no anotan
   intención.

---

## 7 · Lo que habilita

Con un FEN que varía por tipo, `FENef = FEN(tipo) × CClimP(lugar, mes)` deja de
dar el mismo número para todo. Recién entonces la validación de `FENef` puede
probar la **dimensión de tipo**, que es justo lo que `VALIDACION_CCLIMP.md` §8
había declarado imposible de probar con un solo tipo de activo.

El camino corto para llegar ahí: los 2.475 sistemas de agua potable rural ya
tienen coordenada, ya tienen componentes con FEN medido y distinto, y ocupan
**1.176 celdas** de la malla climática fina (427 de la gruesa). Es el primer
conjunto del proyecto donde se puede probar tipo **y** lugar a la vez.

**No se cierra nada sin el director.**

Relacionado: `SUBMATRICES_Y_EL_FEN_CONSTANTE.md` · `VALIDACION_CCLIMP.md` ·
`VARIABLES_Y_METRICAS_PROYECTO.md` · `FUENTE_MOP.md`
