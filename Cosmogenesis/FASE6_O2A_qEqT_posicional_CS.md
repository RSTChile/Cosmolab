# O2-A — factorial q_E × q_T con instrumento POSICIONAL (post-`layout_resortes`): ¿deja huella el orden de formación?

**Fecha:** 11-ago-2026 · **Ola 2 del `FASE6_PLAN_EJECUCION_COMPLETA_CS.md`** · **No se corrió Phantom**
(todo se midió sobre la nube de posiciones ANTES del colapso gravitacional). No se declara cierre ni
veredicto — sólo se reportan números; la lectura final es de Alexis.

## Por qué existe esta tarea

`CS073_factorial_qE_qT_CS.md` (10-ago) cruzó los mismos dos ejes con el **espectro del laplaciano
topológico** y encontró que **q_T daba efecto exactamente CERO en las 75 celdas** — pero no como hallazgo
empírico, sino por identidad matemática: L = D − A se construye desde la adyacencia FINAL, que es una
función del *conjunto* de aristas y no tiene ninguna columna de "quién se conectó primero". El
instrumento era **ciego** al eje, no el eje inexistente. Ese mismo informe dejó anotado el camino no
tomado, que es exactamente esta tarea: repetir la grilla con un instrumento que **sí pueda ver el orden**
— las POSICIONES que devuelve `layout_resortes`, una relajación iterativa cuyo resultado depende del
camino. El precedente conocido era `null4_verificar_invarianza_orden.py`: el mismo conjunto de aristas
insertado en distinto orden daba posiciones con diferencias de hasta ~38% de la escala típica de
coordenada en el peor caso individual.

La pregunta operacional de O2-A: **cuando el instrumento puede ver q_T, ¿de qué tamaño es su huella
comparada con la de q_E?**

## Qué se corrió

| eje | valores | notas |
|---|---|---|
| **q_E** (identidad de aristas conservada) | {1.0, 0.75, 0.5, 0.0} — **4 niveles** | `factor_swaps = round(10·(1−q_E))` → 0, 2, 5, 10 intentos de double-edge-swap por arista, con el mismo filtro de longitud (`tol_relativa=0.2`). q_E=0.0 = NULL-3 puro. Se sacó el nivel 0.25 del factorial anterior. |
| **q_T** (orden de inserción conservado) | {1.0, 0.99, 0.9, 0.0} — **4 niveles** | fracción (1−q_T) de posiciones del orden que entran en un sorteo y se revuelven entre sí: **0, 49, 494 y 4945** posiciones tocadas de 4945. Espaciado NO uniforme y concentrado cerca de q_T=1 **decidido antes de correr**, para distinguir un ESCALÓN (saturación inmediata) de un GRADIENTE (dependencia del grado de desorden). El factorial anterior usaba {1.0, 0.5, 0.0}, que no separa esas dos posibilidades. |
| **semillas** | 5 (`seed_swap` = 6000–6004; `seed_orden` = 7000 + i·10 + j) | mismas familias que el factorial anterior; son la vara de ruido. |

**Grilla: 4 × 4 × 5 = 80 celdas** + 3 controles de techo = **83 layouts**. N = 2000 partículas,
malla causal REAL canónica (D=3, k=4, `seed_ejes=2000`), **4945 aristas**, `layout_resortes` con
`iters=100` y `seed_layout=12345` en todas las celdas. Costo real: 197.6 s de CPU por celda (~277 min de
CPU, ~35 min de reloj con 8 procesos). La grilla se recortó de 5×5×5 a 4×4×5 por presupuesto, con el
recorte documentado en el propio script antes de correr.

**Aislamiento de los ejes** (igual que en el factorial anterior): para cada `(q_E, semilla)` el swap de
aristas se hace UNA vez y sobre ESE MISMO conjunto de aristas se aplican los cuatro q_T. Un `assert` por
celda verifica que el barajado de orden no cambió el conjunto de aristas (nunca saltó).

### Observables posicionales medidos (14)

**(A) Distancia a la configuración de REFERENCIA** (la celda q_E=1.0, q_T=1.0). Los nodos conservan su
etiqueta y todos los layouts arrancan de la MISMA nube inicial, así que la comparación coordenada a
coordenada es directa y no hace falta alinear ni rotar.
- `rms_a_ref` — desplazamiento cuadrático medio por partícula; `rms_a_ref_rel` — el mismo, dividido por el
  radio de giro de la referencia (adimensional); `max_a_ref` — el peor desplazamiento individual.

**(B) Estadística de la nube** (sin referencia; describe la forma que quedó): `radio_giro`,
`anisotropia` (√(λ1/λ3) de la covarianza), `d_nn_media` / `d_nn_std` (vecino más cercano),
`d_knn8_media`, `long_arista_media` / `long_arista_std`.

**(C) Espectro laplaciano PONDERADO POR DISTANCIA** — el análogo del instrumento anterior pero **con
memoria de la geometría**: en vez de A_ij ∈ {0,1} se usa w_ij = exp(−d_ij²/2σ²), con σ = mediana de la
longitud de arista de la referencia, **fija para toda la grilla**. Se reportan `lambda2_w`,
`lambda_max_w`, `mean_eig_w`, `std_eig_w`.

## Chequeos de sanidad (los dos pasaron)

1. **Determinismo.** La celda (q_E=1.0, q_T=1.0) se corrió con las 5 semillas: `rms_a_ref` = **0.0
   exacto** en las cinco, y `lambda2_w` idéntico hasta el último decimal (0.004104). No hay ninguna
   fuente de azar sin contabilizar.
2. **Techo de decorrelación.** El MISMO grafo de referencia relajado desde 3 nubes iniciales distintas
   (`seed_layout` = 12346/12347/12348) da `rms_a_ref_rel` = **138.1%** del radio de giro (1.430 / 1.370 /
   1.343). Ese es el valor de "dos layouts completamente independientes" — y coincide con el valor
   analítico esperado para dos nubes isótropas sin relación entre sí (√2 = 141.4%). Es la vara contra la
   que se lee todo lo de abajo.
3. El grafo quedó conexo en las 80 celdas (`n_ceros_w` = 1 en todas).

## Resultado central — la descomposición de varianza

Descomposición de dos vías con réplicas sobre las 80 celdas (df = 3, 3, 9, 64), % de la suma de cuadrados
total. Los p-valores se agregan acá (no estaban en el CSV) para leer si una fracción chica es o no
distinguible del ruido entre semillas.

| observable | q_E % | **q_T %** | inter. % | error % | F_qE | F_qT (p) | F_int (p) |
|---|---|---|---|---|---|---|---|
| `rms_a_ref` | 98.14 | **0.074** | 0.223 | 1.56 | 1340.6 | 1.01 (p=0.40) | 1.01 (p=0.44) |
| `rms_a_ref_rel` | 98.14 | **0.074** | 0.223 | 1.56 | 1340.6 | 1.01 (p=0.40) | 1.01 (p=0.44) |
| **`max_a_ref`** | 97.29 | **0.301** | 0.858 | 1.55 | 1338.0 | **4.14 (p=0.0096)** | **3.93 (p=0.0005)** |
| `radio_giro` | 80.17 | 0.006 | 0.032 | 19.79 | 86.4 | 0.007 (p=0.999) | 0.01 (p≈1) |
| `anisotropia` | 22.25 | 0.050 | 0.038 | 77.66 | 6.1 | 0.014 (p=0.998) | 0.003 (p≈1) |
| `d_nn_media` | 33.48 | 0.633 | 1.764 | 64.12 | 11.1 | 0.21 (p=0.89) | 0.20 (p=0.99) |
| `d_nn_std` | 36.07 | 0.172 | 0.563 | 63.19 | 12.2 | 0.06 (p=0.98) | 0.06 (p≈1) |
| `d_knn8_media` | 72.60 | 0.142 | 0.287 | 26.97 | 57.4 | 0.11 (p=0.95) | 0.08 (p≈1) |
| `long_arista_media` | 71.41 | 0.005 | 0.043 | 28.54 | 53.4 | 0.004 (p≈1) | 0.01 (p≈1) |
| `long_arista_std` | 78.86 | 0.071 | 0.192 | 20.88 | 80.6 | 0.07 (p=0.97) | 0.07 (p≈1) |
| `lambda2_w` | 68.54 | 0.012 | 0.062 | 31.39 | 46.6 | 0.008 (p=0.999) | 0.01 (p≈1) |
| `lambda_max_w` | 4.22 | 0.057 | 0.074 | 95.65 | 0.94 (p=0.43) | 0.013 (p=0.998) | 0.005 (p≈1) |
| `mean_eig_w` | 69.58 | 0.001 | 0.041 | 30.37 | 48.9 | 0.0007 (p≈1) | 0.01 (p≈1) |
| `std_eig_w` | 32.08 | 0.038 | 0.189 | 67.69 | 10.1 | 0.012 (p=0.998) | 0.02 (p≈1) |

### El contraste con el 0% exacto del instrumento topológico

| | instrumento TOPOLÓGICO (10-ago) | instrumento POSICIONAL (esta tarea) |
|---|---|---|
| observable rector | `lambda2` de L = D − A | `rms_a_ref` / `rms_a_ref_rel` (post-`layout_resortes`) |
| **q_T, fracción de varianza** | **0.0% EXACTO** (identidad algebraica: las celdas daban el espectro idéntico bit a bit) | **0.074%** (rms) / **0.301%** (peor partícula) |
| ¿el efecto es distinguible de 0? | no, es 0 por construcción | **sí**: dentro del corte de grafo intacto, F = 1051.6, **p = 1.4×10⁻¹⁸** |
| q_E, fracción de varianza | 76.9% (sobre λ2) | 98.1% (sobre rms), 68.5% (sobre λ2 ponderado) |

**El 0% exacto dejó de ser 0.** Con el instrumento posicional el orden de formación **sí** deja huella
medible. Pero la fracción de varianza que se lleva en la grilla completa es de **décimas de punto
porcentual**, contra el 98% de q_E. Los dos números hay que leerlos juntos con la sección siguiente,
porque el 0.074% global **subestima** el efecto por una razón de diseño que se puede cuantificar.

## La forma del efecto de q_T: es un ESCALÓN, no un gradiente

`rms_a_ref_rel` (% del radio de giro), medias de las 5 semillas por celda:

| q_E \ q_T | 1.00 | 0.99 | 0.90 | 0.00 |
|---|---|---|---|---|
| **1.00** (grafo intacto) | **0.000** | **2.706** | **2.689** | **2.771** |
| 0.75 | 24.104 | 24.091 | 24.131 | 24.166 |
| 0.50 | 28.114 | 28.147 | 28.107 | 28.114 |
| 0.00 | 27.529 | 27.499 | 27.457 | 27.505 |

*(std entre semillas: 0.09–0.14 puntos en la fila q_E=1.0; 1.3–2.3 puntos en las demás.)*

En el corte **q_E = 1.0** (grafo intacto, el eje q_T puro):

- Tocar **49 posiciones de 4945** (el 1%) ya mueve el sistema de **0.000%** a **2.706%** del radio de giro.
- Tocar el 10% (494 posiciones) → 2.689%. Tocar el 100% (4945, = NULL-4 puro) → 2.771%.
- Entre esos tres niveles **no hay diferencia distinguible**: F = 0.79, **p = 0.48**.

Es decir: **el efecto satura de inmediato**. Esto era la hipótesis pre-registrada en el script (la que
motivó el espaciado no uniforme de la grilla, decidido antes de correr): si el orden dejara huella por
amplificación caótica del redondeo de punto flotante — `np.add.at` suma las contribuciones en distinto
orden, diferencia ~1e−16, y 100 iteraciones de una relajación no lineal la amplifican — bastaría tocar el
1% para saturar y la curva sería un escalón. **Es un escalón.** No hay ninguna evidencia de una
dependencia estructurada del *grado* de desorden.

Un matiz que apunta en dirección contraria y conviene no esconder: el **peor caso individual**
(`max_a_ref`) sí muestra una tendencia monótona creciente dentro del escalón — 2.256 → 2.392 → 2.597 al
pasar de q_T = 0.99 a 0.9 a 0.0 (+15%). Con 5 semillas esa tendencia no alcanza significancia entre los
tres niveles (F = 2.13, p = 0.16). Es el único indicio de que "más desorden" podría ser algo más que
"cualquier desorden", y queda sin resolver.

## ¿Hay interacción q_E × q_T?

**Formalmente, casi nada**: el término de interacción se lleva 0.22% de la varianza de `rms_a_ref`
(F = 1.01, p = 0.44). Sólo en `max_a_ref` la interacción es distinguible del ruido (0.86%, F = 3.93,
p = 0.0005), consistente con que el efecto de q_T se concentre en la fila q_E = 1.0.

**Conceptualmente, sin embargo, la tabla es puramente de interacción**: q_T mueve la aguja *sólo* cuando
el grafo está intacto. En las filas q_E < 1 los cuatro valores de q_T coinciden dentro del ruido
(ANOVA de una vía por fila: F ≈ 0.00, p ≈ 1.00 en las tres). La descomposición aditiva no lo refleja
porque 2.7 puntos porcentuales es una nadería al lado del salto de 24–28 puntos que produce q_E.

**Y esto NO se puede leer como "con el grafo alterado el orden deja de importar":** el diseño simplemente
pierde potencia ahí, y se puede mostrar con números. Si los desplazamientos de q_E y de q_T fueran
independientes, se sumarían en cuadratura:

| q_E | rms con orden intacto | predicho si se suma q_T en cuadratura | observado con q_T=0.0 | ruido entre semillas |
|---|---|---|---|---|
| 0.75 | 24.104% | 24.263% (+0.16) | 24.166% | ±1.33 |
| 0.50 | 28.114% | 28.250% (+0.14) | 28.114% | ±1.52 |
| 0.00 | 27.529% | 27.668% (+0.14) | 27.505% | ±2.26 |

El incremento esperado (~0.15 puntos) es **~10 veces menor que la dispersión entre semillas**. Con 5
semillas este observable no tiene ninguna posibilidad de verlo. Lo honesto es decir que en q_E < 1
**no se midió**, no que "no hay".

## Tamaño de la huella, con la vara del techo

| | `rms_a_ref_rel` | % del techo de decorrelación (138.1%) | `max_a_ref` (peor partícula) |
|---|---|---|---|
| **huella de q_T sola** (q_E=1.0, q_T=0.0) | 2.77% del radio de giro | **2.0%** | 2.60 = **27.5% del radio de giro** |
| **huella de q_E sola** (q_T=1.0, q_E=0.5) | 28.11% del radio de giro | **20.4%** | 13.06 = 138% del radio de giro |
| techo (mismo grafo, otra nube inicial) | 138.1% | 100% | 21.65 |

**La identidad de las aristas mueve la geometría ~10 veces más que el orden de formación** (28.1% vs
2.77% en desplazamiento típico por partícula). Y ninguno de los dos llega al techo: incluso destruir el
13% de las aristas deja la nube a un quinto de la distancia que separa dos relajaciones independientes.

El peor caso individual sí es notable: con el grafo **absolutamente intacto**, sólo por cambiar el orden
de inserción, hay al menos una partícula que termina a **27.5% del radio de giro** de donde habría
terminado. Es el mismo orden de magnitud que el ~38% que había reportado
`null4_verificar_invarianza_orden.py` sobre el caso extremo — este factorial reproduce ese precedente y
además muestra que **no hace falta el caso extremo**: con el 1% del orden tocado ya se llega ahí.

## Qué NO cambia con q_T: la forma estadística de la nube

Dentro del corte q_E = 1.0 (donde q_T es lo único que se mueve), ANOVA de una vía sobre los 4 niveles:

| observable | q_T=1.0 | q_T=0.99 | q_T=0.9 | q_T=0.0 | F (p) | rango / valor |
|---|---|---|---|---|---|---|
| `rms_a_ref_rel` | 0.00000 | 0.02706 | 0.02689 | 0.02771 | **1051.6 (1.4e−18)** | — |
| `max_a_ref` | 0.000 | 2.256 | 2.392 | 2.597 | **142.9 (9.2e−12)** | — |
| `radio_giro` | 9.4566 | 9.4554 | 9.4549 | 9.4558 | 2.59 (0.089) | 0.018% |
| `anisotropia` | 1.05208 | 1.05219 | 1.05230 | 1.05230 | 0.13 (0.94) | 0.021% |
| `d_nn_media` | 0.19577 | 0.19544 | 0.19677 | 0.19585 | 0.91 (0.46) | 0.68% |
| `d_knn8_media` | 0.64304 | 0.64526 | 0.64575 | 0.64431 | 3.81 (0.031) | 0.42% |
| `long_arista_media` | 3.15862 | 3.15835 | 3.15777 | 3.15893 | 0.14 (0.94) | 0.037% |
| `long_arista_std` | 2.09883 | 2.09443 | 2.09237 | 2.09456 | 6.53 (0.0043) | 0.31% |
| `lambda2_w` | 0.004104 | 0.004104 | 0.004094 | 0.004106 | 0.69 (0.57) | 0.40% |
| `lambda_max_w` | 10.293 | 10.259 | 10.261 | 10.274 | 1.90 (0.17) | 0.34% |
| `mean_eig_w` | 2.90456 | 2.90521 | 2.90556 | 2.90524 | 0.20 (0.90) | 0.034% |
| `std_eig_w` | 1.95729 | 1.95595 | 1.95526 | 1.95541 | 2.32 (0.11) | 0.10% |

Éste es, para mi lectura, **el resultado más limpio de la tanda**: el orden de formación cambia
**dónde quedó cada partícula concreta** (efecto masivo, p = 1e−18) y **no cambia prácticamente nada de la
forma agregada de la nube** — ni el tamaño (0.018%), ni el alargamiento (0.021%), ni la densidad local
(0.4–0.7%), ni la longitud de las aristas (0.037%), ni el espectro ponderado por distancia (0.03–0.4%).
Dos de esos observables rozan el umbral nominal de 0.05 (`long_arista_std` p=0.0043, `d_knn8_media`
p=0.031) pero con 12 observables corridos en paralelo y efectos de 0.3–0.4% de su propio valor, no los
leería como señal sin replicar.

Y notar que el **espectro laplaciano ponderado por distancia** —diseñado precisamente para "poder ver
q_T"— **casi no lo ve**: 0.012% de la varianza global, p = 0.999. Pesar las aristas por su longitud
geométrica no alcanzó: sigue siendo un resumen estadístico de la nube, y la huella del orden no está en
la estadística de la nube sino en la identidad de quién ocupa cada lugar.

## Cómo se comporta q_E (el eje de contraste), colapsando q_T

| q_E | factor_swaps | `rms_a_ref_rel` | `lambda2_w` | `long_arista_media` | `radio_giro` |
|---|---|---|---|---|---|
| 1.00 | 0 | 0.0% (exacto) | 0.00410 | 3.1584 | 9.4557 |
| 0.75 | 2 | 24.12% | 0.01233 | 3.1148 | 9.4685 |
| 0.50 | 5 | 28.12% | 0.01854 | 3.0856 | 9.4847 |
| 0.00 | 10 | 27.50% | 0.03075 | 3.0487 | 9.5078 |

`lambda2_w` crece de forma **monótona** con la destrucción de identidad de arista (misma dirección que ya
había encontrado CS085 con el laplaciano topológico). El **desplazamiento posicional**, en cambio,
**también satura**: sube de golpe a 24% con sólo 2 swaps por arista y después ya no crece (28.1% en
q_E=0.5, 27.5% en q_E=0.0, con std entre semillas de 1.5–2.3 puntos). Dos ejes distintos, la misma
firma de saturación temprana.

## Limitaciones reales de estos datos

1. **La zona q_T ∈ (0.99, 1.0) está sin explorar, y es justo la interesante.** El efecto ya está saturado
   en q_T=0.99 (49 posiciones de 4945). No sabemos si **una sola** transposición basta. Con la hipótesis
   caótica sería que sí, pero no está medido. Es la continuación natural más barata: q_T con 1, 2, 5, 10
   posiciones tocadas.
2. **Cobertura de la grilla recortada.** Se perdió q_E=0.25 y q_T=0.5 respecto del factorial anterior; y
   entre q_T=0.9 y q_T=0.0 no hay nada. Como el eje resultó ser un escalón plano, el hueco importa poco
   *ex post*, pero no se puede afirmar con esta grilla que no haya estructura ahí.
3. **5 semillas por celda.** Suficiente para el efecto de q_E (F ~ 1340) y para el escalón de q_T con
   grafo intacto (F ~ 1052), **insuficiente** para todo lo demás: la tendencia monótona de `max_a_ref`
   dentro del escalón (p=0.16) y cualquier efecto de q_T en q_E < 1 (donde el incremento esperado es 10×
   menor que el ruido entre semillas) están fuera del alcance de este n.
4. **Un solo `seed_layout` (12345) en toda la grilla.** Toda la comparación vive dentro de una única nube
   inicial. El techo con 3 semillas alternativas muestra que otra nube inicial da un resultado
   esencialmente independiente, así que no está garantizado que el escalón del 2.7% sea el mismo en otro
   basin. Replicar la fila q_E=1.0 con 2–3 `seed_layout` distintos costaría ~10 min y cerraría esa duda.
5. **Un solo N (2000) y un solo `iters` (100).** La amplificación caótica del redondeo depende del número
   de iteraciones: con 50 o 200 iteraciones el escalón podría estar en otro lado. Nada de esto se barrió.
6. **`rms_a_ref` es una distancia a UNA configuración de referencia**, no una distancia entre pares de
   celdas. Por eso no puede detectar q_T cuando q_E ya desplazó todo (ver la tabla de cuadratura). Un
   observable pareado — distancia entre las posiciones de dos celdas que comparten (q_E, semilla) y sólo
   difieren en q_T — sí tendría potencia ahí, y no requiere recorrer nada nuevo salvo guardar posiciones.
   **Las posiciones no se guardaron en disco**, así que ese análisis no se puede hacer sobre los datos
   actuales.
7. **Discrepancia de etiquetado con el factorial anterior (no afecta la comparación, pero conviene
   anotarla).** `cs073_factorial_qe_qt.csv` registra `n = 1599` en todas sus filas, y su informe lo repitió
   ("N=1599… `dens_bar.npy` del pool tiene 1599 partículas"). Al leer los datos verifiqué que
   `dens_bar.npy` del pool tiene **2000** partículas (mtime 1-ago, anterior a las dos corridas) y que la
   malla canónica tiene 4945 aristas sobre 2000 nodos; una lectura de esa misma malla con n=1599 daría
   3962 aristas, no 4945, y el propio `mean_eig` = 4.945 del CSV viejo es exactamente 2·4945/**2000**. O
   sea: **la corrida anterior usó la misma malla de 2000 nodos y 4945 aristas que ésta** — la columna `n`
   del CSV viejo está mal etiquetada. Los dos factoriales SÍ son comparables número a número; sólo queda
   pendiente auditar de dónde salió el 1599. **No modifiqué ningún archivo.**

## En simple, con la analogía

El informe anterior usaba **foto vs. video**. Se puede afinar así:

Imaginá que la malla causal es un grupo de gente tomada de la mano, y `layout_resortes` es soltarlos en
una plaza y dejar que se acomoden tironeándose hasta quedar cómodos. Hay dos cosas que podemos cambiar:
**quién le da la mano a quién** (q_E) y **en qué orden se fueron dando la mano** (q_T).

- El instrumento anterior era **una foto del organigrama**: una lista de "quién está agarrado de quién".
  Esa lista no tiene fecha. Cambiá el orden todo lo que quieras: la lista sale idéntica, letra por letra.
  Por eso q_T daba 0% — no porque el orden no hiciera nada, sino porque la foto no tiene dónde anotarlo.
- El instrumento de ahora es **la foto aérea de la plaza después de que todos se acomodaron**: dónde
  quedó parado cada uno.

Y lo que se ve en la foto aérea es esto: **si cambiás el orden en que se dieron la mano, la gente termina
parada en otro lado.** No es cero. Es real, y es tan fuerte estadísticamente como cualquier cosa que
hayamos medido (p = 1 en 10¹⁸). Pero es **chico**: cada persona se corre en promedio un 2.8% del tamaño
de la plaza, y la que más se corre se mueve un 27%. Si en cambio cambiás **con quién** se agarran, cada
persona se corre un 28% — **diez veces más**.

Dos detalles que valen la pena:

- **Basta un empujoncito.** Cambiar el orden de sólo **49 apretones de mano de 4945** produce exactamente
  el mismo desorden que cambiar los 4945. No hay "un poco de desorden da un poco de efecto": lo tocás y
  ya está todo el efecto. Es la firma típica de un sistema caótico — como el aleteo de la mariposa: una
  diferencia microscópica en el arranque (acá, literalmente, el orden en que la computadora suma unos
  números, diferencias del tamaño de 0.0000000000000001) se amplifica hasta ser visible después de 100
  rondas de acomodarse.
- **La plaza queda igual, la gente no.** El grupo ocupa el mismo tamaño, tiene la misma forma alargada,
  la misma densidad, los brazos igual de estirados — todos esos números cambian menos de medio punto
  porcentual y ninguno es distinguible del ruido. Lo único que cambió es **quién está en cada lugar**. El
  orden de formación deja una huella de **identidad**, no de **forma**.

Traducido a lo que importa para la línea de trabajo: si el observable con el que después se juzga el
sistema es un promedio o una estadística global de la nube (tamaño, densidad, espectro), **va a ser casi
ciego al orden de formación igual que lo era la foto topológica**. Para que el orden importe hace falta un
observable que dependa de dónde está cada cosa concreta.

## Archivos

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs073_factorial_qe_qt_posicional.py` — el script
  (importa, sin modificar, `cs073_factorial_qe_qt`, `null4_verificar_invarianza_orden`,
  `cs072_modulos.piezas.p_semilla_causal`, `null1_generar_ic`).
- `cs073_factorial_qe_qt_posicional.csv` — 83 filas (80 de grilla + 3 controles de techo), 30 columnas.
- `cs073_factorial_qe_qt_posicional_anova.csv` — descomposición de varianza, 14 observables.
- `cs073_factorial_qe_qt_posicional.png` — `rms_a_ref_rel`, `lambda2_w` y `d_nn_media` vs q_T, una curva
  por q_E, con la línea del techo.
- Antecedentes: `CS073_factorial_qE_qT_CS.md` (factorial topológico, el 0% exacto),
  `null4_verificar_invarianza_orden.py` (el precedente del ~38%), `FASE6_PLAN_EJECUCION_COMPLETA_CS.md`.
- Este informe. No se corrió Phantom, no se modificó ningún script ni CSV existente, no se hicieron
  commits, y no se declara cierre ni veredicto sobre O2-A ni sobre CS073.
