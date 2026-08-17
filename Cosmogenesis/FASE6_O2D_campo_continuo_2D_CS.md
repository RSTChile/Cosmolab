# FASE VI — O2-D: sustrato A0 genuinamente 2D (campo continuo, sin grafo)

**Fecha:** 11-ago-2026 · **Ejecuta:** CC (Claude) · **Plan:** `FASE6_PLAN_EJECUCION_COMPLETA_CS.md`, tarea O2-D
**Antecedentes directos:** `FASE6_O1C_cierre_A0_CS.md` (el grafo derivado de A0 fabrica su propio resultado:
test-retest 45.6%), `FASE5_A0_metricas_nativas_CS.md` (métricas nativas 1D), `FASE5_especificacion_universalidad_CS.md`
§1 (filtro P1-P5). Propuesta A4 del 3er analista del equipo.

No se declara cierre ni veredicto. Se reportan números; la lectura final es de Alexis. **No se corrió Phantom.
No se editó ningún script congelado** (`cs090_fase5_generador.py` sólo se importa, para reusar su lista de
patrones de coordenadas y su lista de nombres prohibidos). No se hizo ningún commit.

---

## 0. Qué se pedía y qué se hizo

La representación A0 del motor de Fase V es **un anillo 1D**. En 1D varias métricas geométricas son
degeneradas (el propio informe de métricas nativas descartó box-counting por eso). La pregunta de fondo:
**¿S>0 puede operar en un campo continuo sin grafo, o el grafo es condición NECESARIA para el régimen del
modelo?**

Se implementó un sustrato A0 en **2D** (grilla toroidal L×L) con dos familias de dinámica local, se verificó
P1-P5 de verdad sobre cada configuración, y se midió con seis instrumentos **sin construir ningún grafo
derivado**, cada uno contra su propio control NULL (mismo campo, posiciones barajadas). Se agregó un bloque
**3D** reducido.

| # | Qué | Resultado en una línea |
|---|---|---|
| 1 | Dos dinámicas 2D nuevas (FASE y reacción-difusión) | ambas implementadas y corridas: **45 corridas 2D** (15 configuraciones × 3 semillas, L=128) + **12 corridas 3D** (4 × 3, L=48) |
| 2 | Filtro P1-P5 verificado, no asumido | **2D: 15/20 admitidas** (FASE 10/10; RD 5/10 — los 3 controles puestos a propósito caen solos, más 2 caídas no anticipadas). **3D: 4/6** |
| 3 | Control NULL sobre todas las métricas | **12 de 12 métricas separan** el campo REAL del BARAJADO en 2D (Wilcoxon pareado p ≤ 2.1e-06, \|Cliff δ\| ≥ 0.66); el **control negativo** `std_valores` **no separa** (p=0.056, δ=-0.002), como tenía que pasar |
| 4 | Los 4 criterios de "estructura sin grafo" | conjunción completa: FASE **1/30** (versión pre-declarada) y **6/30** (versión simétrica); RD **4/15** y **9/15**. Dos de los cuatro umbrales resultan pasables por el propio barajado — auditoría en §10 |
| 5 | Bloque 3D | FASE 3D separa igual que en 2D (**9/9** corridas, p=3.9e-03); **RD 3D no es interpretable**: 2 de 3 semillas terminaron en un tablero de ajedrez a escala de red (k = Nyquist), artefacto numérico del integrador, no patrón |

---

## 1. Qué dinámica se implementó

Todo vive en una **grilla toroidal** L^d (d=2 o 3). Los vecinos de un sitio son un conjunto de
**desplazamientos fijos declarados una sola vez** (`np.roll` por eje) — exactamente la misma construcción que
el `left`/`right` del anillo 1D que el generador congelado ya acepta como P3-compatible, sólo que con d ejes
en vez de uno. Ningún sitio lee jamás su propia posición.

### Familia FASE — la generalización directa del A0 de 1D

Cada sitio tiene una fase en Z_K. Un sweep:

```
S(t+1) = [ (1-J)·S(t) + J·⟨fase de los vecinos de grilla⟩_circular ] mod K  + ruido gaussiano
```

Es literalmente la regla del anillo (`cs090_fase5_motor.dinamica_B0`, rama A0), con vecindad de 4 u 8 en vez
de 2. El promedio de vecinos es circular (vía el vector unitario e^{iθ}) para que no salte en el corte 0/K.
**10 configuraciones**, barriendo acople J ∈ [0.20, 0.90], ruido ∈ [0.02, 0.50], K ∈ {4,6,8}, vecindad ∈ {4,8},
400 sweeps cada una.

### Familia RD — reacción-difusión de Gray-Scott (campo continuo genuino, no un grafo disfrazado)

Dos especies U y V acopladas por la autocatálisis U + 2V → 3V:

```
U(t+1) = U + dt·( Du·∇²U − U·V²  + F·(1−U) )
V(t+1) = V + dt·( Dv·∇²V + U·V²  − (F+k)·V )
```

con ∇² el laplaciano discreto de la grilla (suma de vecinos menos n_vecinos·centro). **10 configuraciones**
barriendo el plano (F, k) de Gray-Scott, 4000 pasos cada una, incluidos **3 controles a propósito** (siembra
rala, alimentación alta, remoción alta) donde se esperaba que el patrón muriera — para ver si el filtro los
descarta solo, en vez de descartarlos a mano.

Estado inicial en las dos familias: **i.i.d. sitio a sitio, sin ninguna estructura espacial**. Toda estructura
que aparezca la produce la dinámica.

### Una calibración que hubo que hacer, documentada

Con la siembra inicial que se había elegido primero (2% de sitios sueltos con V=0.25) **el patrón de
Gray-Scott moría en las 5 configuraciones probadas** (V → 0.0000 exacto a t=1000, 4000 y 8000). Motivo: un
sitio de V aislado difunde su V antes de poder reproducirse. Con siembra ≈0.3 hay, por puro azar, suficientes
pares y tríos de sitios adyacentes para que la reacción nuclee. Se pasó a siembra=0.30 y **se dejó una
configuración con siembra=0.02 dentro del barrido como control**, para que el hecho quede en los datos y no
sólo en esta nota.

---

## 2. El filtro P1-P5, verificado de verdad

Se verificó cada P sobre cada configuración, antes de correr nada. Dos chequeos son **empíricos** (se corre la
dinámica chica de verdad) y dos reusan la lógica del generador congelado:

| P | Cómo se verificó | Reusa del congelado |
|---|---|---|
| **P1** persistencia | dinámica REAL encadenada vs. control SIN MEMORIA (mismo campo con los valores re-barajados); margen exigido >0.02, el mismo que usa el 1D. Con asentamiento previo (10% de los pasos) para no medir la memoria de un campo recién sorteado | criterio y margen de `GEN.chequear_P1_persistencia` |
| **P2** diferencia operable | FASE: fase en Z_K con K>1. RD: dos especies U,V distinguibles y acopladas | — |
| **P3** localidad sin coordenadas | inspección del **código** (sin docstrings ni comentarios) de las 9 funciones de sustrato y dinámica, buscando los mismos patrones de coordenadas | lista `["pos[", "posiciones[", "coord", "xy=", "(x, y)", "embedding"]` |
| **P4** reciprocidad | (a) el conjunto de desplazamientos es cerrado bajo negación (si i lee a j, j lee a i); (b) sólo RD: se perturba U en un sitio y se mide el cambio en V al paso siguiente, y viceversa | — |
| **P5** sin valores físicos horneados | nombres de parámetros contra la lista prohibida + descripción <500 caracteres | `GEN._PROHIBIDOS_P5` |

**Detalle de por qué P3 se chequea sobre el código y no sobre el texto:** los propios comentarios de este
sustrato hablan de "coordenadas" justamente para explicar que NO se usan, y eso daba un falso positivo del
chequeo consigo mismo (el generador congelado resuelve el mismo problema excluyendo su propia función de la
inspección). Lo que P3 exige es que la **ley de actualización** no lea coordenadas — eso vive en el código.

**Caveat honesto, del lado del instrumento:** la dinámica nunca usa (x,y,z), pero dos de las **métricas**
(box-counting y masa-radio) sí usan la geometría de la grilla. La distancia que usan es la distancia de la
propia adyacencia toroidal — la misma relación que define los vecinos de la dinámica — no una escala puesta a
mano. Queda anotado, no escondido.

---

## 3. Con qué se midió — y por qué esto NO es un grafo derivado

Seis instrumentos, todos leyendo el array del campo directamente, todos dimensión-genéricos (mismo código en
2D y 3D), todos aplicados de forma idéntica al campo REAL y al campo BARAJADO:

1. **Espectro de potencia radial P(k)** — pico y su prominencia sobre el fondo, nitidez (ancho relativo del
   pico), entropía espectral normalizada (1 = potencia repartida entre todos los modos = ruido blanco),
   exponente de ley de potencias.
2. **Espectro angular** — anisotropía en el anillo k ∈ [2, L/4]: ¿hay una dirección privilegiada?
3. **Dimensión por box-counting** del conjunto de nivel, en 4 ocupaciones (5%, 10%, 25%, 50%) — en 2D/3D sí es
   informativa, a diferencia de 1D.
4. **Entropía de configuración espacial** — entropía de los 16 patrones posibles de bloque 2×2 binarizado.
   (La entropía de los *valores* no serviría: el barajado la conserva exacta por construcción. Por eso se mide
   la de **bloques**, que sí ve el orden espacial.) Se mide en 5 instantes del tiempo.
5. **Percolación de conjuntos de nivel, con barrido de umbral** — 19 cuantiles de ocupación, no uno elegido.
   Ésa es la lección de O1-C. Componentes conexas por la **misma adyacencia que usa la dinámica** (4-conexo en
   2D, 6-conexo en 3D).
6. **Escalamiento masa-radio** desde los máximos locales del campo, con distancias toroidales.

Más **longitud de correlación ξ** por autocorrelación radial vía FFT.

**Por qué esto no es el grafo derivado de O1-C:** `_grafo_medicion_A0` conectaba cada sitio con 15 candidatos
elegidos **al azar en todo el anillo** si estaban en fase parecida — es decir, instalaba atajos de largo
alcance, que es la receta de Watts-Strogatz, que es exactamente la firma de mundo-pequeño que después decía
haber medido. Acá no se inventa ninguna arista: la única conectividad que se usa es la de la propia grilla.

**Control NULL:** el mismo campo con **las posiciones barajadas** — destruye toda la estructura espacial y
conserva exactamente la distribución de valores. Cada corrida es su propio control (emparejamiento exacto), así
que el test es un Wilcoxon **pareado**. Se incluye `std_valores` como **control negativo**: el barajado la
conserva por construcción, así que ahí NO debe aparecer diferencia; si apareciera, el NULL estaría mal hecho.

---

## 4. Qué entró y qué quedó fuera: el filtro P1-P5 en números

`cs090_fase6_o2d_2d_filtro_P1P5.csv` (20 filas) y `cs090_fase6_o2d_3d_filtro_P1P5.csv` (6 filas).

| bloque | admitidas | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|---|
| 2D FASE | **10/10** | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| 2D RD | **5/10** | 9/10 | 10/10 | 10/10 | **5/10** | 10/10 |
| 3D FASE | **3/3** | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 3D RD | **1/3** | 1/3 | 3/3 | 3/3 | **1/3** | 3/3 |

**El cuello de botella es P4, y siempre por el mismo motivo:** cuando el patrón de Gray-Scott muere, V→0
en todo el dominio, el término de acople U·V² se apaga, y la influencia deja de ir en los dos sentidos
(U→V = 0.00e+00 mientras V→U = 2.50e-03). El filtro no descarta "porque no me gustó el resultado": descarta
porque en un campo muerto **la reciprocidad literalmente no existe**. En un caso (RD08, alimentación alta) P1
cae también: persistencia real 0.938 vs sin-memoria 0.926, margen 0.012 < 0.02.

- **Los 3 controles puestos a propósito cayeron solos**: RD07 (siembra rala 0.02), RD08 (alimentación alta
  F=0.09), RD09 (remoción alta k=0.085). Eso era exactamente lo que se quería comprobar: que el filtro los
  detecte sin que nadie los tache a mano.
- **Dos caídas NO anticipadas**: RD01 (F=0.030, k=0.062, anotada como "laberinto — verificado vivo") y RD05
  (F=0.025, k=0.060, "cerca del borde del régimen"). Se anota el caveat honesto: **el filtro corre sobre una
  grilla chica de 32×32 y el barrido sobre 128×128**. Un patrón de Gray-Scott con longitud de onda ≈12 sitios
  entra unas 2-3 veces en una caja de 32 y unas 10 en una de 128; es perfectamente posible que RD01 estuviera
  viva en 128 y muerta en 32. Como el descarte ocurre **antes** del barrido, no hay dato para decidirlo. Queda
  como limitación conocida del diseño, no como resultado.
- P3 (localidad sin coordenadas) y P5 (sin valores horneados) pasan 26/26 en los dos bloques: cero patrones de
  coordenadas en el código de sustrato/dinámica, cero nombres prohibidos, descripciones de 128-211 caracteres
  contra el tope de 500.
- P1 en FASE: persistencia real 0.663-0.885 contra sin-memoria 0.484-0.503 — margen de 16 a 39 puntos, muy por
  encima del 0.02 exigido. En RD viva: 0.938-1.000 contra 0.495-0.926.

---

## 5. Control NULL primero: ¿las métricas ven algo, o miden ruido?

Esto va antes que cualquier interpretación, porque si el instrumento no distingue el campo real del mismo
campo con las posiciones barajadas, **nada de lo que siga vale**. Wilcoxon pareado (cada corrida contra su
propio barajado) sobre las 45 corridas 2D:

| métrica | REAL media | BARAJADO media | REAL>NULL | Wilcoxon p | Cliff δ | ¿separa? |
|---|---|---|---|---|---|---|
| prominencia del pico de P(k) | 1563.39 | 1.48 | 45/45 | 5.7e-14 | +0.995 | **SÍ** |
| entropía espectral | 0.553 | 0.997 | 0/45 | 5.7e-14 | −1.000 | **SÍ** |
| nitidez del pico (ancho relativo) | 1.251 | 6.214 | 6/45 | 2.1e-06 | −0.661 | **SÍ** |
| exponente de ley de potencias | −1.570 | +0.038 | 0/45 | 5.7e-14 | −0.991 | **SÍ** |
| anisotropía angular | 0.540 | 0.151 | 44/45 | 4.0e-13 | +0.922 | **SÍ** |
| box-dim (5% más alto) | 1.115 | 1.152 | 3/45 | 8.6e-07 | −0.864 | **SÍ** |
| box-dim (10%) | 1.329 | 1.385 | 3/45 | 1.2e-10 | −0.867 | **SÍ** |
| box-dim (25%) | 1.609 | 1.667 | 0/45 | 5.7e-14 | −1.000 | **SÍ** |
| box-dim (50%) | 1.816 | 1.849 | 0/45 | 5.7e-14 | −1.000 | **SÍ** |
| entropía de configuración espacial | 0.795 | 0.999 | 0/45 | 5.7e-14 | −1.000 | **SÍ** |
| exponente masa-radio | 1.841 | 1.972 | 2/45 | 1.9e-12 | −0.933 | **SÍ** |
| longitud de correlación ξ | 4.64 | 1.00 | 38/45 | 5.7e-08 | +0.844 | **SÍ** |
| *`std_valores` (control negativo)* | *0.2255* | *0.2255* | *1/45* | *0.056* | *−0.002* | *no (correcto)* |

**Las 12 métricas informativas separan, y las 12 separan en la dirección que la hipótesis de "hay estructura"
predecía.** El control negativo se comporta como debía: el barajado conserva la distribución de valores exacta,
y ahí no aparece diferencia (δ = −0.002, o sea nada). Ese renglón es la prueba de que el NULL está bien hecho
y de que las otras 12 diferencias no son un error de emparejamiento.

En analogía simple: si uno desarma un rompecabezas armado y tira las piezas en una caja, **las piezas son las
mismas** (eso es `std_valores`, que no cambia) pero **el dibujo desaparece** (eso son las otras doce, que
cambian todas). El instrumento ve el dibujo, no las piezas.

Por familia el patrón se sostiene y RD es aún más limpia que FASE: en RD las 12 métricas dan **Cliff δ = ±1.000**
(separación perfecta, sin un solo cruce en 15 corridas); en FASE la más débil es la nitidez del pico
(δ = −0.470, p = 9.1e-04) y aun así separa.

---

## 6. Espectro de potencia: dónde vive la escala — y una diferencia grande entre las dos familias

| familia | k del pico (rango) | longitud de onda | % de la potencia en k≤2 | % en k>8 | prominencia pico/fondo |
|---|---|---|---|---|---|
| FASE (2D) | 1-4, con **27/30 en k≤2** | ≥ 64 sitios (≥ media caja) | 4 - 96 % (mediana ≈ 50 %) | 1 - 83 % | 2 - 8300 |
| RD (2D) | **9-11 en 15/15** | ≈ 12-14 sitios | 0.2 - 0.4 % | 85 - 96 % | 1470 - 6180 |
| barajado | disperso (1-11, sin repetir) | — | — | — | **1.2 - 2.4** |

Éste es, a mi juicio, el número más informativo de todo el experimento y no estaba entre los cuatro criterios:

- **RD tiene una escala propia.** El pico se para en λ ≈ 12-14 sitios en las 15 corridas, con 3 semillas por
  configuración y 5 configuraciones distintas del plano (F,k). Esa longitud **no la puso nadie** —no hay ninguna
  distancia en el código, la unidad es el paso de grilla— y **no depende de la caja**: es la longitud de onda
  que sale del cociente entre las dos difusividades y las tasas de reacción. Eso es una escala característica
  emergente en un campo continuo, sin grafo.
- **FASE no tiene escala propia.** Su pico se para casi siempre en el modo más grande que la caja permite
  (k=1, λ = 128 = la caja entera; 27 de 30 corridas con k≤2). Un pico en k=1 no dice "el sistema eligió una
  longitud"; dice "el sistema está formando dominios cada vez más grandes y lo único que lo detiene es el
  borde". La longitud de correlación lo confirma desde el otro lado: ξ va de 1 (acople débil, mucho ruido) a
  22.7 sitios (FASE08, J=0.90, ruido=0.02), siempre **por debajo** de L=128. Analogía: RD es una tela con un
  estampado de tamaño fijo; FASE es un charco de tinta que se sigue expandiendo hasta chocar con el vaso.

La dependencia con los parámetros de FASE es fuerte y ordenada (Spearman sobre las 30 corridas):
entropía espectral vs acople J = **−0.902**, vs ruido = **+0.819**; prominencia del pico vs J = **+0.894**;
ξ vs ruido = **−0.915**. O sea: más acople y menos ruido ⇒ más orden, monótonamente. FASE09 (J=0.20,
ruido=0.50) es el extremo: prominencia 2.1 contra 1.3 del barajado, entropía espectral 0.993 contra 0.997 —
prácticamente indistinguible del ruido. El barrido cubrió de verdad los dos extremos del régimen.

**Anisotropía — con reserva.** REAL 0.540 vs barajado 0.151 (44/45, p=4.0e-13). Pero hay que decir que en las
corridas de FASE casi toda la potencia vive en 2-3 modos, y un perfil angular construido con 2-3 modos sale
disparejo *por construcción*, sin que eso signifique "hay una dirección privilegiada en la física". A favor de
que algo real haya: los ángulos del pico **no se pegan a los ejes de la red** (sólo 9 de 30 caen en 0° o 90°;
el resto se reparte por 10°, 25°, 35°, 60°, 105°, 150°, 160°...) y **cada corrida elige uno distinto**, que es
la firma de una dirección elegida espontáneamente y no impuesta por la grilla. Lo dejo como observación, no
como resultado: haría falta un test específico (por ejemplo, comparar contra un campo gaussiano con el mismo
P(k) radial pero fase aleatoria) para separar "pocos modos" de "dirección preferida".

---

## 7. Percolación: **el criterio pre-declarado salió al revés, y eso es el resultado**

Lo declarado de antemano fue: *"q_c real < 0.45"*, pensando en un campo agrupado que percola **antes** que el
azar (la ocupación crítica de percolación de sitios en la red cuadrada es 0.593; el barajado medido dio
**0.614**, rango 0.60-0.65, lo cual valida que la referencia era la correcta). La expectativa era razonable:
si los sitios altos están agrupados, la componente gigante debería aparecer con menos ocupación.

**Los datos dicen otra cosa, y dicen cosas distintas según la familia.**

| | q_c REAL medio | q_c barajado | antes / después / empate | Wilcoxon | AUC gigante REAL vs barajado |
|---|---|---|---|---|---|
| FASE (n=30) | **0.555** | 0.614 | **17 / 2 / 11** | p=3.6e-04, δ=−0.556 | 0.440 vs 0.357, **30/30**, p=1.9e-09 |
| RD (n=15) | **0.630** | 0.614 | **4 / 11 / 0** | p=0.73, δ=+0.467 | 0.387 vs 0.357, 6/15, p=0.89 |

- **FASE va en la dirección anticipada** (percola antes), pero flojo: el corrimiento típico es de **un solo paso
  de la grilla de cuantiles** (0.05), y sólo **1 de 30** corridas cruza la barra absoluta de 0.45. El criterio
  estaba puesto demasiado exigente para el tamaño del efecto que realmente hay.
- **RD va en la dirección CONTRARIA**: percola **más tarde** que el azar en 11 de 15 corridas. Ésa es la
  dirección opuesta al criterio pre-declarado, y **no se maquilla**: se reporta como salió.

**Por qué pasa, con los números que lo explican.** La clave está en el número de componentes conexas
(`cs090_fase6_o2d_2d_percolacion.csv`), que no formaba parte de ningún criterio:

| ocupación q | FASE real | FASE barajado | **RD real** | RD barajado |
|---|---|---|---|---|
| 0.15 | 913 | 1734 | **84** | 1750 |
| 0.25 | 1032 | 2131 | **68** | 2119 |
| 0.50 | 698 | 1131 | **49** | 1125 |
| 0.65 | 368 | 251 | **27** | 255 |

El campo de RD, al 25 % de ocupación, está hecho de **68 pedazos** donde el barajado tiene **2131**. Son
**treinta veces menos pedazos, y por lo tanto mucho más grandes**. Eso es *más* estructura, no menos. Y es
justamente por eso que percola tarde: la fracción gigante de RD sube tempranísimo (0.32 ya en q=0.35-0.50,
cuando el barajado está en 0.01-0.03) pero después **se queda planchada en ~0.32** y no cruza el 50 % hasta
q≈0.70, porque para que la mancha más grande se coma a las demás hay que rellenar los huecos que las separan.

Analogía simple: el criterio pre-declarado preguntaba *"¿el continente aparece antes?"*. En FASE sí, apenas.
En RD lo que hay no es un continente sino **un archipiélago de islas grandes**: cada isla es enorme comparada
con lo que produce el azar, pero están separadas por mar, y la isla mayor no llega a ser "más de la mitad de
toda la tierra" hasta que se rellena mucho. Preguntar "¿cuándo hay continente?" mide la forma del archipiélago,
no cuánta tierra hay.

**Consecuencia metodológica, dicha en voz alta:** el signo de q_c **no** es una medida de "cuánta estructura
hay" — es una medida de **morfología**. Las dos direcciones son igual de no-triviales; sólo una estaba
anticipada. Y dentro de RD la dirección **depende del régimen de Gray-Scott**: RD02 (F=0.040, k=0.060,
manchas densas/solitones) percola **antes** en las 3 semillas (q_c 0.30-0.35 contra 0.60); RD00, RD04 y RD06
(manchas más dispersas) percolan **después** en las 9; RD03 ("rayas") se parte entre semillas (2 después, 1
antes) — o sea que ahí hay dos atractores morfológicos y la semilla decide cuál. Con la misma dinámica, el
mismo filtro y las mismas métricas, la respuesta a "¿percola antes o después?" cambia de signo según la forma
del patrón.

**El observable que sí resiste** es la separación máxima de la curva completa respecto de su propio barajado:
**45 de 45 corridas** se desvían ≥0.10 en algún punto del barrido (40 de 45 se desvían ≥0.25); mediana 0.461
en FASE y **0.779** en RD. Es decir: **todas** las corridas percolan distinto del azar; lo que cambia es hacia
qué lado. El criterio simétrico agregado (|q_c real − q_c null| ≥ 0.10) recoge parte de eso — 6/30 en FASE,
9/15 en RD — pero sigue siendo tosco, porque la grilla de cuantiles avanza de a 0.05 y un corrimiento genuino
de un paso (el típico de FASE) es invisible para una barra de 0.10.

---

## 8. Las otras métricas geométricas

**Dimensión de box-counting.** En las cuatro ocupaciones el conjunto de nivel real es más "concentrado" que el
barajado, y en las ocupaciones medias la separación es **perfecta** (0/45 cruces, δ = −1.000): 1.609 vs 1.667
al 25 %, 1.816 vs 1.849 al 50 %. La diferencia absoluta es chica pero absolutamente sistemática. Por familia,
RD concentra más que FASE (box-dim al 25 %: 1.563 vs 1.632).
Hay **tres corridas donde el signo se invierte**, y son las tres semillas de FASE08 (J=0.90, ruido=0.02):
box-dim del 5 % más alto = 1.24-1.25 contra 1.15 del barajado. Tiene una lectura sencilla: FASE08 es la
configuración que forma **un solo dominio gigante**, y el 5 % más alto de un dominio gigante está *desparramado
por una región enorme*, así que llena más cajas que 5 % de puntos sueltos. Cuando la estructura es "una sola
cosa muy grande", box-counting deja de leerla como concentración. No es un fallo del NULL: es el límite del
instrumento, y se anota.

**Entropía de configuración espacial.** REAL 0.795 vs barajado 0.999 (0/45). Por familia: FASE 0.900,
RD **0.586**. RD está mucho más estructurada. El barajado da 0.999 en las 45 — o sea, exactamente lo que
predice la teoría para bloques 2×2 de sitios independientes, otra confirmación de que el control funciona.

**Masa-radio.** REAL 1.841 vs barajado 1.972 (2/45). El barajado da ≈2.0, que es lo que debe dar un campo
homogéneo en 2D (la masa crece como el área). El campo real da menos: la masa se concentra cerca de los
máximos. RD 1.745, FASE 1.889.

**Longitud de correlación ξ.** REAL 4.64 vs barajado 1.00 (38/45, p=5.7e-08). El barajado da 1.00 en las 45
corridas —el mínimo medible, sitios independientes— y ninguna corrida satura el rango. Caveat: ξ se devuelve
como número entero de sitios, así que en las configuraciones de acople débil (ξ real = 1) el instrumento
directamente **no tiene resolución** para distinguir; ésos son los 7 empates.

---

## 9. Persistencia temporal: ¿el campo se estructura o se homogeniza?

Entropía de configuración espacial en 5 instantes (2 %, 10 %, 25 %, 50 % y 100 % del presupuesto de pasos),
promediada sobre las 3 semillas. Delta negativo = se estructura.

| configuración | t inicial | t final | delta | lectura |
|---|---|---|---|---|
| FASE00 (J=0.80, r=0.05) | 0.796 | 0.846 | +0.050 | se homogeniza |
| FASE01 | 0.921 | 0.910 | −0.011 | se estructura (apenas) |
| FASE02 | 0.853 | 0.895 | +0.042 | se homogeniza |
| FASE03 | 0.966 | 0.937 | −0.029 | se estructura |
| FASE04 | 0.938 | 0.882 | **−0.056** | se estructura |
| FASE05 | 0.982 | 0.960 | −0.023 | se estructura |
| FASE06 | 0.893 | 0.936 | +0.044 | se homogeniza |
| FASE07 | 0.768 | 0.931 | **+0.163** | se homogeniza |
| FASE08 (J=0.90, r=0.02) | 0.570 | 0.701 | +0.131 | baja hasta 0.487 en t=40 y **después sube** |
| FASE09 (J=0.20, r=0.50) | 0.999 | 0.997 | −0.002 | estable (ruido, nunca se estructuró) |
| RD00 | 0.489 | 0.589 | +0.100 | se homogeniza |
| RD02 | 0.507 | 0.580 | +0.073 | se homogeniza |
| RD03 | 0.574 | 0.577 | +0.003 | estable (baja a 0.333 en t=400 y vuelve) |
| RD04 | 0.441 | 0.591 | +0.151 | se homogeniza |
| RD06 | 0.510 | 0.589 | +0.079 | se homogeniza |

**Ninguna de las dos familias muestra la entropía bajando monótonamente.** El patrón real es otro: la
estructura **aparece temprano** (ya en el primer instante medido, 2 % del tiempo, RD está en 0.44-0.57 contra
el 0.999 del ruido) y después **se queda o se erosiona un poco**. RD termina en 0.577-0.591 en las cinco
configuraciones: muy estructurada, y muy estable — el estampado ya está impreso y no cambia.

Dos caveats importantes del instrumento, no del campo:
1. **La entropía de bloques se calcula binarizando por la mediana del campo.** Cuando un solo dominio ocupa
   más de la mitad de la caja (FASE08), la mediana **cae adentro del dominio** y la binarización empieza a
   cortar ruido interno en vez de cortar el borde del dominio. Eso puede explicar por completo el "sube después
   de t=40" de FASE08 sin que el campo se haya desordenado. Es exactamente el mismo tipo de confound de umbral
   que O1-C encontró en el grafo derivado de A0, sólo que acá afecta a **una** métrica de siete y se puede ver.
2. Las trayectorias son medias de 3 semillas; con 3 semillas un delta de ±0.01 no significa nada. Los únicos
   deltas grandes son FASE07 (+0.163), FASE08 (+0.131), RD04 (+0.151) y FASE04 (−0.056).

---

## 10. Los 4 criterios pre-declarados: el marcador — y la auditoría del marcador

Marcador crudo, corrida por corrida (2D, 45 corridas):

| familia | n | pico nítido (>10× barajado) | H espacial < 0.90 | percolación (pre) | percolación (simétrico) | dim < 1.80 | **los 4 (pre)** | **los 4 (sim)** |
|---|---|---|---|---|---|---|---|---|
| FASE | 30 | 24 | 11 | **1** | 6 | 30 | **1** | **6** |
| RD | 15 | 15 | 15 | **4** | 9 | 15 | **4** | **9** |

Y ahora la parte que hay que decir aunque incomode: **dos de los cuatro criterios, tal como se pre-declararon,
también los pasa el propio campo barajado.** Se comprobó aplicándolos al NULL:

| criterio | ¿lo pasa el campo BARAJADO (2D)? | ¿discrimina? |
|---|---|---|
| pico nítido: prominencia real > 10× la del barajado | 0/45 (por construcción el cociente vale 1) | **sí** |
| entropía espacial < 0.90 | 0/45 (el barajado da 0.999) | **sí** |
| percolación q_c < 0.45 | 0/45 en 2D (el barajado da 0.60-0.65) | sí en 2D, **no en 3D** (ver §11) |
| dimensión box-counting del 5 % < 1.80 | **45/45** | **no: es un aprobado automático** |

El criterio de "dimensión no entera" está mal calibrado: al 5 % de ocupación, en una grilla de 128×128, el
box-counting da ≈1.15 **incluso para puntos completamente al azar** (porque 819 puntos sueltos no alcanzan a
llenar las cajas grandes). Poner la barra en 1.80 hace que lo pase todo el mundo. Lo que sí discrimina es la
**comparación pareada** contra su propio barajado (1.115 vs 1.152, 42/45, p=8.6e-07) — y eso es precisamente
lo que el diseño del NULL permitía hacer y el criterio absoluto desaprovechaba.

**Cómo leo el marcador, entonces:** la conjunción de los cuatro criterios (1/30 y 4/15 en la versión
pre-declarada) **no mide lo que el experimento quería medir**, porque uno de los criterios es un aprobado
automático, otro (percolación) resultó direccional cuando el fenómeno no lo es, y un tercero (entropía < 0.90)
tiene un umbral que en la práctica separa a RD de FASE más que a estructura de ruido. El número que **sí**
resiste es el de §5: **12 de 12 métricas separan REAL de BARAJADO, y el control negativo no**. Dejo los dos
marcadores a la vista, el crudo y esta auditoría, para que la lectura la haga Alexis con las dos cosas
delante y no con una sola.

---

## 11. Bloque 3D

12 corridas (4 configuraciones admitidas × 3 semillas, L=48). Se parte en dos historias muy distintas.

**FASE 3D funciona y repite lo de 2D.** Las mismas métricas separan REAL de BARAJADO con separación perfecta
sobre las 9 corridas (δ = −1.000 / +1.000, p = 3.9e-03, que es el p mínimo alcanzable con n=9 en un Wilcoxon):

| métrica | REAL | BARAJADO | REAL>NULL |
|---|---|---|---|
| prominencia del pico | 102.1 | 1.20 | 9/9 |
| entropía espectral | 0.596 | 0.999 | 0/9 |
| entropía espacial | 0.970 | 0.998 | 0/9 |
| box-dim 5 % | 1.958 | 1.978 | 0/9 |
| masa-radio | 2.878 | 2.974 | 0/9 |
| ξ | 1.00 | 1.00 | 0/9 (**no separa**: sin resolución) |

Y la percolación de FASE 3D es lo **más limpio** de todo el experimento en esa métrica: q_c real 0.25-0.30
contra 0.35 del barajado, **9 de 9 antes**, δ = −1.000, p = 3.9e-03, y la fracción gigante a q=0.25 es
**0.395 real contra 0.006 barajado**. Ahí sí el campo percola clarísimamente antes que el azar. Pero ojo:
**el criterio pre-declarado "q_c < 0.45" es inútil en 3D**, porque en la red cúbica el umbral de percolación
de sitios al azar es 0.312 y el barajado medido da 0.35-0.50 — o sea que el propio barajado pasa el criterio
en 10 de 12 corridas. Lo mismo con "dimensión < d − 0.20 = 2.80": el barajado lo pasa 10/10 (da 1.98). Los
umbrales absolutos de la §10 estaban calibrados para 2D y **no son transportables a 3D**; sólo sobrevive la
comparación pareada.

**RD 3D no es interpretable, y hay que decirlo así.** De las 3 semillas de R3D01, **2 terminaron en un tablero
de ajedrez a escala de red**: el pico de P(k) está en k=24, que con L=48 es exactamente el modo de Nyquist
(longitud de onda = 2 sitios, el valor alterna de un sitio al siguiente); la entropía espectral es
2.3e-94 (toda la potencia en un único modo); la entropía de bloques es exactamente 0; el barrido de
percolación da 55 296 componentes de un sitio cada una; y **las dos semillas dan `std_valores` idéntico hasta
el sexto decimal (0.132692)**, lo que confirma que no es un patrón sino un **atractor numérico del
integrador**, el mismo para cualquier condición inicial. El esquema explícito en 3D está al borde de su límite
de estabilidad (dt·Du·2d = 1·0.16·6 = 0.96, contra 1.0), y esas dos semillas lo cruzaron. La tercera semilla
(R3D01 s3) sí dio un patrón de Gray-Scott genuino: k_pico=4 (λ=12 sitios, la misma escala que en 2D),
prominencia 3779, entropía espacial **0.442**, box-dim al 25 % 2.460 vs 2.585 del barajado, y su entropía baja
en el tiempo (0.337 → 0.147, delta −0.190, la única trayectoria claramente descendente de todo el
experimento). Un dato aislado, con n=1, que apunta en la misma dirección que el 2D; nada más que eso.

Efecto secundario que conviene marcar: los dos tableros de ajedrez **pasan el criterio "entropía < 0.90"**
(dan 0.0) y por eso figuran en el marcador de §10 como 3/3 en ese criterio para RD 3D. Un criterio de "hay
mucha estructura" premiando un artefacto numérico es exactamente el tipo de cosa por la que conviene mirar los
campos y no sólo las tablas.

---

## 12. La pregunta de fondo: ¿S>0 puede operar sin grafo, o el grafo es NECESARIO?

Lo que los datos permiten decir, y lo que no.

**1) Un campo continuo, sin ningún grafo, sostiene estructura espacial que ninguna métrica confunde con
ruido.** Doce instrumentos independientes —espectrales, geométricos, topológicos y de correlación— separan el
campo real de su propio barajado, con emparejamiento exacto, en 45/45 corridas 2D y 9/9 corridas 3D de FASE,
y el control negativo se queda quieto. En el sentido literal de la pregunta: **el grafo no es necesario para
que haya estructura persistente medible**. Y esto se logró **evitando por diseño** el problema de O1-C: no se
construyó ninguna arista, así que no hay ningún atajo de largo alcance inventado por el instrumento que
después reaparezca como "hallazgo".

**2) La escala característica emergente aparece en la familia que NO es la generalización de A0.** RD elige
una longitud de onda propia (λ ≈ 12-14 sitios, reproducible en 15/15 corridas y 5 configuraciones, y la misma
en la única corrida 3D válida) que no está en ningún parámetro del código. FASE, que es literalmente la regla
del anillo A0 con d ejes en vez de uno, **no elige ninguna longitud**: su potencia se va al modo más grande
que la caja permite (k≤2 en 27/30). Dicho en simple: llevar A0 a 2D le dio geometría, pero no le dio una
**medida** propia; el que trae medida propia es Gray-Scott, que es una dinámica de otra familia.

**3) Lo que NO se puede concluir.** Que las métricas separen real de barajado dice *"acá pasó algo espacial"*,
que es una barra baja: cualquier proceso local de difusión la pasa. **No** dice que lo que pasó sea el régimen
que el modelo postula. La conjunción de criterios que iba a decidir eso (§10) resultó tener un criterio de
aprobado automático y otro direccional en un fenómeno que no lo es, así que **el experimento no queda en
condiciones de responder si el grafo es o no condición necesaria para el régimen del modelo** — sólo de
responder que no lo es para tener estructura. Es una respuesta a media pregunta, y prefiero decirlo así antes
que estirar el resultado.

**4) Una asimetría que sí queda planteada.** El grafo derivado de A0 (O1-C) producía mundo-pequeño porque el
instrumento instalaba atajos; acá, sin ningún atajo, lo que emerge son **dominios y manchas** — estructura
local, de vecino a vecino, sin nada parecido a un atajo de largo alcance. Si el régimen del modelo requiere
atajos, entonces todo lo que O1-C midió podría venir del instrumento, y este experimento estaría mostrando lo
que queda cuando se los saca. Es una hipótesis, no un resultado: para ponerla a prueba habría que medir en el
campo continuo el observable que en el grafo se leía como mundo-pequeño, sin volver a construir el grafo.

---

## 13. Limitaciones conocidas, en una lista

1. **El filtro corre en 32×32 y el barrido en 128×128** (§4). Dos configuraciones de RD pudieron caer por el
   tamaño de la grilla de chequeo, no por su dinámica. No hay dato para decidirlo porque el descarte precede
   al barrido.
2. **Dos de las siete métricas usan la geometría de la grilla** (box-counting y masa-radio). Es la geometría
   de la propia adyacencia que define los vecinos de la dinámica, no una escala puesta a mano — pero es del
   lado del instrumento, y ya estaba declarado en §2.
3. **3 semillas por configuración.** Alcanza para el Wilcoxon pareado agregado (n=45), no para hablar de una
   configuración suelta.
4. **Los umbrales absolutos de los 4 criterios están calibrados para 2D** y no sobreviven el cambio a 3D
   (§11); uno de ellos ni siquiera discrimina en 2D (§10).
5. **ξ se devuelve en sitios enteros**, sin resolución por debajo de 1 — de ahí que no separe en 3D.
6. **La entropía de bloques binariza por la mediana**, lo que la vuelve poco fiable cuando un dominio ocupa
   más de media caja (FASE08).
7. **El integrador de RD en 3D está al borde de su límite de estabilidad** y 2 de 3 semillas cayeron en el
   modo de tablero de ajedrez. El bloque de RD 3D queda con n=1 útil.
8. **La anisotropía no está controlada** contra un campo con el mismo P(k) radial y fase aleatoria, que es el
   control que haría falta para separar "pocos modos" de "dirección preferida".

---

## 14. Archivos

| archivo | qué tiene |
|---|---|
| `cs090_fase6_o2d_campo2d.py` | genera y mide (sustrato, filtro P1-P5, 7 instrumentos, barrido) |
| `cs090_fase6_o2d_analisis.py` | control NULL, criterios, persistencia temporal, figuras |
| `cs090_fase6_o2d_2d_raw.csv` | 45 corridas 2D × 58 columnas (cada métrica REAL y NULL lado a lado) |
| `cs090_fase6_o2d_2d_filtro_P1P5.csv` | 20 configuraciones con el resultado y el detalle de cada P |
| `cs090_fase6_o2d_2d_percolacion.csv` | curva completa (19 cuantiles) por configuración, semilla 1 |
| `cs090_fase6_o2d_2d_espectros.csv` | P(k) real y barajado, 64 valores de k por configuración, semilla 1 |
| `cs090_fase6_o2d_2d_campos.npy` | los campos REAL y BARAJADO de la semilla 1, para las figuras |
| `cs090_fase6_o2d_3d_*.csv` | lo mismo para el bloque 3D (12 corridas, 4 configuraciones) |
| `cs090_fase6_o2d_2d.log`, `_3d.log` | salida completa del barrido, corrida por corrida |

**Nota sobre las curvas guardadas:** `_percolacion.csv` y `_espectros.csv` guardan **sólo la semilla 1** de
cada configuración (así está escrito el barrido). Para R3D01 la semilla 1 es una de las dos que cayeron en el
tablero de ajedrez, así que esas dos curvas 3D de RD no representan a la configuración: la corrida útil
(semilla 3) está en `_3d_raw.csv` pero no tiene curva guardada.

No se declara cierre ni veredicto. La lectura final es de Alexis.
