# π CONTINGENTE — RE-CORRIDA CON LOS CONTROLES QUE FALTARON
## Re-corrida de `HALLAZGO_pi_contingente_y_rumbo_gravedad_cuantica_CS.md` (16-jul-2026)
### Director: Alexis López Tapia · Ejecuta: CC · 13-ago-2026 · **NO es cierre — son números y curvas**

---

## 0. PRE-REGISTRO (escrito ANTES de calcular, no se toca)

**Definición usada, textual del nodo, sin cambios:** en un grafo, la "circunferencia" de radio `r` es el
**nº de nodos a distancia geodésica EXACTA `r`** desde una fuente (la frontera de la bola, `|S(r)|`); el
"diámetro" es `2r`; y

    π_emergente(r) = |S(r)| / (2r)

Nada de coordenadas, nada de layout, nada de embebido euclidiano. Medida intrínseca al grafo.

**Qué se decide con el barajado (double-edge-swap, grados preservados) del mismo grafo mundo-pequeño:**

- **El barajado NO estalla** (π(r) acotado) → el estallido es firma de la estructura real; el hallazgo del
  16-jul **se sostiene y ahora tiene control**.
- **El barajado TAMBIÉN estalla** → el estallido es propiedad genérica de cualquier grafo no-reticular; el
  resultado **no distingue** y hay que decir que "π indefinido" es un **re-enunciado de mundo-pequeño**.
- **Intermedio** (estallan distinto, con pendientes separables) → se reporta la curva y la diferencia.

**Criterio de reproducción de las cuatro filas:** se considera reproducida la fila si π(r) medida cae dentro
de ±0,05 del valor publicado en el régimen sin efecto de borde (retículas) o si la curva sigue la misma forma
creciente sin cota (mundo-pequeño). Si NO se reproduce el 2,0 / 2,99 / 1,5 → **parar y reportar**: la
definición no estaba bien documentada, y eso es en sí un hallazgo.

**Guardas declaradas antes de correr:**
1. Buscar el resultado algebraico: si `|S(r)| ~ b^r`, entonces `π(r) = b^r/(2r)` estalla **por necesidad
   matemática**. Se calcula la tasa de crecimiento de bolas y se separa qué parte del estallido es eso.
2. Verificar con números que el barajado **no es isomorfo** al real (solapamiento de aristas, triángulos,
   clustering, diámetro).
3. Nunca usar coordenadas del layout.
4. Medir sobre la **componente gigante** (bug `_diam`), no desde el nodo 0 por índice.

*(El resto de este documento se escribió después de correr; el pre-registro de arriba es literal y no fue
modificado.)*

---

## 1. ¿SE REPRODUCEN LAS CUATRO FILAS? — SÍ, y las tres retículas salen EXACTAS

Reimplementado desde cero con la definición textual (`cs091_pi_contingente_rerun.py`), retículas de 121×121,
fuente en el centro, radios lejos del borde:

| sustrato | π publicado (16-jul) | π medido ahora | ¿reproduce? |
|---|---|---|---|
| retícula cuadrada | 2,0 fijo | **2,00** en r=1..12, sin una sola desviación | **SÍ, exacto** |
| retícula triangular | 2,99 | **3,00** en r=1..12 | **SÍ** (el 2,99 publicado es el mismo número con un borde rozado) |
| retícula hexagonal | 1,5 | **1,50** en r=1..12 | **SÍ, exacto** |
| mundo-pequeño | 2,5→48,2 en r=1..7 | 3,02 · 3,09 · 5,44 · 10,18 · 18,80 · 31,85 · 40,07 | **SÍ en forma y magnitud** |

**Por qué salen exactos, y por qué eso importa.** La frontera medida es literalmente `|S(r)| = 4r`, `6r`, `3r`.
Es la definición dividida por sí misma: `4r/(2r) = 2` para todo r. **No es un resultado empírico, es una
identidad.** Se construyó una retícula cuadrada y se obtuvo la constante de una retícula cuadrada. La
observación #2 del encargo ("es casi definicional") queda **confirmada con números**: no hay nada que medir ahí.

El mundo-pequeño publicado (`[2.5, 3.3, 5.8, 11.2, 18.1, 29.6, 48.2]`) corresponde a grado medio 5 y tasa de
ramificación ≈2,6 — un Watts-Strogatz N=2000, k=6, p=0,10 lo reproduce en forma y en orden de magnitud.

**No hay código del 16-jul en disco.** Se buscó (`grep` de la definición, de los valores, de los nombres) y no
existe script que haya producido esa tabla. La medida de hoy es una reimplementación desde el texto, no una
re-ejecución. Es reproducible: el texto bastaba.

---

## 2. EL CONTROL QUE DECIDE — **EL BARAJADO TAMBIÉN ESTALLA. Y estalla IGUAL.**

Sobre los grafos reales que ya estaban en disco (`grafos_f800/F5B_40pares`, línea A2-B0-C2, cargados con
`cs090_fase8_f800_grafos.py`, sello sha256 verificado), medidos sobre la **componente gigante** con 400 fuentes:

`A2-B0-C2-batch3-r0`, N=2000, E=2308 — π(r) para r=1..8:

| | r=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| **REAL** | 1,32 | 1,41 | 1,99 | 3,28 | 5,55 | 9,15 | 14,00 | 17,90 |
| **NULL barajado** (grados preservados) | 1,29 | 1,41 | 2,03 | 3,22 | 5,35 | 8,82 | 13,72 | 18,31 |
| **NULL Erdős–Rényi** (mismo N, mismo E) | 1,34 | 1,56 | 2,31 | 3,82 | 6,43 | 10,66 | 15,91 | 19,36 |

Con **20 réplicas barajadas** por grafo, el z del real contra la nube de sus barajados, radio a radio:

| grafo | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |
|---|---|---|---|---|---|---|---|---|
| batch3-r0 | +0,1 | −0,6 | −0,5 | +0,7 | +1,5 | **+1,8** | +1,5 | +0,1 |
| batch3-r100 | +0,1 | −1,1 | −0,3 | +0,4 | +1,1 | +1,2 | +0,9 | −1,6 |
| batch3-r104 | −0,0 | −0,9 | −0,0 | +0,9 | +1,5 | +0,7 | −4,0 | −2,4 |

En el régimen que el nodo publicó (r=1..7, la fase de crecimiento) **el real cae dentro de la nube barajada:
|z| ≤ 1,8**. El único |z| grande (−4,0 en batch3-r104, r=7) está ya en la zona de vuelta de la curva, donde lo
que se mide es el tamaño finito del grafo, no el estallido.

### → **Se cumple la SEGUNDA rama del pre-registro, literalmente.**

> *"El barajado TAMBIÉN estalla → el estallido es una propiedad genérica de cualquier grafo no-reticular;
> el resultado no distingue y hay que decir que 'π indefinido' es un re-enunciado de mundo-pequeño."*

Y todavía más fuerte que lo pre-registrado: **el ER también estalla**, con la misma forma y sólo un pelo más
arriba. Los tres brazos son la misma curva.

---

## 3. GUARDA 1 — CUÁNTO DEL ESTALLIDO ES PURA ARITMÉTICA: **todo**

`π(r) = |S(r)| / (2r)`. Si la frontera crece exponencialmente, `|S(r)| ≈ S(1)·b^(r−1)`, entonces

        π(r) ≈ S(1) · b^(r−1) / (2r)

que **crece sin cota para cualquier b > 1**, sin que ninguna estructura tenga que hacer absolutamente nada.
Medida la tasa de ramificación real (`b_r = |S(r)|/|S(r−1)|`, ajustada en r=2..4):

| sustrato | b ajustada | π(7) observada | π(7) que predice sólo el álgebra | fracción explicada |
|---|---|---|---|---|
| REAL corpus | 2,15 | 14,13 | 18,80 | **1,12** |
| NULL barajado | 2,14 | 13,80 | 18,26 | **1,12** |
| NULL ER | 2,28 | 16,83 | 26,88 | **1,18** |
| mundo-pequeño WS | 2,38 | 40,07 | 78,43 | **1,26** |

La fracción explicada es **≥ 1,00 en todos los casos**: la aritmética del crecimiento exponencial no sólo
explica el estallido, lo **sobre-explica** (predice más de lo observado; la diferencia es que el grafo es
finito y la bola se choca contra las paredes). **Queda cero residuo para atribuir a la estructura.**

En la retícula ocurre exactamente lo contrario y por la misma aritmética: ahí `b_r = r/(r−1)` → 1 (la frontera
crece **lineal**: 4, 8, 12, 16, 20…), y `4r/(2r)` es constante por construcción.

> **Traducción simple:** en la retícula, a cada paso se suman 4 baldosas más. En el mundo-pequeño, a cada
> paso el número de vecinos nuevos se **multiplica** por 2,15. Que 2,15⁷ sea un número grande no dice nada
> sobre el universo: dice que 2,15⁷ = 213.

### Y una consecuencia que el nodo no vio: **π(r) no "estalla", sube y baja**

La curva medida hasta r=12 tiene **joroba**: sube, llega a un máximo cerca de r ≈ diámetro/2, y cae a cero.
Tiene que hacerlo: `|S(r)| ≤ N`, así que `π(r) ≤ N/(2r)`. En `batch3-r104` la curva hace 1,78 → 41,4 → 23,4 →
… → cero. **La secuencia publicada `[2.5 … 48.2]` se cortó en r=7, justo antes de la vuelta.** "π indefinido"
no es divergencia: es la mitad ascendente del perfil de crecimiento de una bola en un grafo finito.

---

## 4. HALLAZGO METODOLÓGICO — "π constante" NO detecta geometría: detecta **dimensión 2**

Ésta es la objeción más seria y no estaba en el encargo. Se midió π(r) sobre una **retícula CÚBICA 3D**
construida a mano (31³ = 29.791 nodos, 6 vecinos, geometría perfecta, sin un solo atajo):

| r | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| π(r) retícula cúbica 3D | 3,00 | 4,50 | 6,33 | 8,25 | 10,20 | 12,17 | 14,14 | 16,12 |

**π(r) crece linealmente (≈ 2r) en un espacio geométrico impecable.** La razón es elemental: en dimensión d la
frontera de la bola crece como `r^(d−1)`, así que

        π(r) = |S(r)| / (2r) ∝ r^(d−2)

- d = 2 → π constante ✔ (las tres filas del nodo)
- d = 3 → π **crece lineal** ✘ (no es constante, y hay geometría de sobra)
- d = ∞ / exponencial → estalla

**Consecuencia dura para el nodo del 16-jul:** el criterio "π constante ⇒ hay geometría / π no constante ⇒ no
hay geometría" **es falso**. Es un detector de bidimensionalidad. Si el sustrato hubiera cuajado en la
geometría 3D que este proyecto busca desde el arco CS066-069, π(r) habría crecido igual — y con esta regla de
lectura se lo habría declarado "indefinido". El instrumento no puede distinguir "no cuajó ninguna geometría"
de "cuajó una geometría que no es 2D".

---

## 5. ¿HAY GEOMETRÍA EMERGENTE EN EL CORPUS? — se midió la celda que faltaba. **No la hay.**

**(a) Barrido de los 254 grafos guardados** (`grafos_f800/`, toda la línea A2-B0-C2 + O3A/O3B/O3C/O3D/O3E),
estadístico = coeficiente de variación de π(r) en r=2..5 (retícula: CV = 0,000):

| | CV de π(r) | b (ramificación) |
|---|---|---|
| retícula 2D (referencia) | **0,000** | → 1 |
| grafo del corpus **más constante** (`batch4-r49`) | 0,451 | 2,03 |
| mediana del corpus (n=254) | 0,759 | 2,95 |
| grafo **menos constante** (`r12v1fix` N4000) | 0,946 | 3,95 |

**Ninguno de los 254 se acerca a plano.** El más "reticular" del corpus está a medio camino entre una retícula
y un mundo-pequeño puro, y su ramificación es 2,03 — exponencial, no lineal.

**(b) El candidato fuerte, regenerado a propósito.** No hay grafos de CS066 guardados en disco, así que se
**regeneró** el brazo `local` de `cs066_localidad_geometrogenesis.py` (geometrogénesis por costo de
no-localidad, estilo Quantum Graphity) — el único sustrato del corpus del que se afirmó geometría emergente
(`cs066conf_exponentes.md`: *"locALMENTE 3D pero GLOBALMENTE compacto"*), N=2500:

| sustrato | π(r), r=1..8 | CV | b |
|---|---|---|---|
| CS066 `local` k=5 **EMERGENTE** | 3,36 · 9,83 · 9,50 · 14,87 · 24,70 · 31,67 · 31,80 · 24,89 | 0,42 | 2,87 |
| su barajado (grados preservados) | 3,60 · 18,24 · 85,44 · 159,41 · 42,26 · 0,69 · 0,00 | 0,70 | 5,00 |
| CS066 `sin_local` (blob, control) | 14,48 · 298,06 · 177,56 · 0,68 · 0 · 0 | 1,06 | 10,52 |

**Respuesta al encargo: sí existe un sustrato emergente que reclama geometría, se lo midió, y su π(r) tampoco
es constante — estalla.** Es el resultado esperable después del punto 4: CS066 reclama localidad **3D**, y en
3D π(r) crece por definición aunque la geometría sea perfecta. Así que este dato **no falsifica CS066**; lo
que muestra es que π(r) no sirve para juzgarlo.

Sí hay una diferencia real y grande entre `local` y su barajado (b: 2,87 vs 5,00; el barajado se agota a r=6 y
el real llega a r=12) — pero eso es **exactamente el resultado de diámetro-vs-barajado que CS066 ya había
publicado**, leído en otra coordenada. No es información nueva. (n=1 par de réplicas; dirección consistente
con lo ya establecido, no se le pone estadística.)

---

## 6. GUARDA 2 — el barajado NO es el mismo grafo renombrado (verificado con números)

Double-edge-swap con 20·E intentos (≈46.000–94.000 swaps efectivos por grafo). Comprobaciones:

| grafo | Jaccard aristas real∩barajado | grados idénticos | triángulos real→barajado | clustering real→barajado | diám. gigante real→barajado |
|---|---|---|---|---|---|
| batch3-r0 (N2000) | **0,00065** | sí | 27 → 2 | 0,0170 → 0,00067 | 19 → 21 |
| batch3-r100 | 0,00148 | sí | 26 → 0 | 0,0165 → 0,000 | 25 → 18 |
| batch3-r104 | 0,00135 | sí | 13 → 6 | 0,0047 → 0,0014 | 14 → 14 |
| batch3-r107 | 0,00092 | sí | 27 → 1 | 0,0098 → 0,00027 | 15 → 15 |
| batch3-r0 (N4000) | 0,00074 | sí | 28 → 1 | 0,0089 → 0,00017 | 20 → 21 |

El barajado comparte **menos del 0,15% de las aristas** con el real, destruye 90-100% de los triángulos, baja
el clustering un orden de magnitud, y mueve el diámetro. La secuencia de grados es idéntica (verificado
explícitamente). **No es el mismo grafo renombrado — y aun así da la misma π(r).** Ése es el punto: el NULL
funcionó, cambió todo lo que tenía que cambiar, y π no se enteró.

## 7. GUARDAS 3 y 4

- **Guarda 3 (sin coordenadas):** la medida es sólo BFS sobre listas de adyacencia. Nunca se llama a
  `layout_resortes` ni a ningún embebido. Se verifica leyendo `bfs_capas` en `cs091_pi_contingente_rerun.py`:
  no hay un solo número real de posición en todo el camino.
- **Guarda 4 (componente gigante):** `componente_gigante()` recorre TODAS las componentes y devuelve la mayor;
  las 400 fuentes de cada BFS se sortean dentro de ella, y el diámetro de control usa doble-BFS arrancando
  ahí. Nunca se arranca "desde el nodo 0 por índice". Las componentes gigantes son 1681-1939 de 2000 nodos
  (los grafos del corpus tienen resto fragmentado: medir desde el índice 0 habría sido el bug `_diam` otra vez).

---

## 8. QUÉ QUEDA EN PIE DEL NODO DEL 16-JUL, Y QUÉ HAY QUE REETIQUETAR

**Queda en pie (y se confirma):**
- Los cuatro números son correctos y reproducibles. No hay error de cálculo en ninguna parte.
- La mitad **"π distinto entre geometrías"** es real y sólida: 2,0 ≠ 3,0 ≠ 1,5. Una retícula cuadrada y una
  triangular tienen razones frontera/diámetro distintas. Eso es un hecho y no necesita NULL.

**Hay que reetiquetar (tres cosas):**
1. **"π indefinido donde no hay geometría" no está medido.** Está medido "π(r) crece donde la bola crece
   exponencialmente", que es una identidad algebraica y que el barajado y el ER cumplen igual. Como afirmación
   sobre el sustrato, es **un re-enunciado de "el sustrato es mundo-pequeño"** — cosa que el diámetro de
   CS068 ya había establecido, con menos pasos.
2. **"π constante donde hay geometría" es falso en general.** Vale sólo en d=2. La retícula cúbica 3D lo
   rompe. El instrumento no distingue "sin geometría" de "geometría no bidimensional".
3. **π(r) no diverge.** Sube y vuelve a bajar. La serie publicada se cortó justo antes de la joroba.

**La razón profunda, en una línea:** `π(r) = |S(r)|/(2r)` es una **transformación invertible del perfil de
crecimiento de bolas** `|S(r)|` — dividir por `2r` no agrega ni quita un solo bit. Todo lo que π(r) puede
decir ya lo dice `|S(r)|`, que es el diagnóstico que este proyecto viene usando desde el arco (d_s espectral,
diámetro-vs-N). **π(r) no es una medida nueva; es una unidad nueva para una medida vieja.**

## 9. EL MARCO DEL DIRECTOR, contra estos números

Alexis: *"π NO es algo metafísico, es simplemente la relación que persistió al filtro"*. La predicción tenía
tres partes separables, y quedan así:

| parte de la predicción | estado |
|---|---|
| π **estable dentro** de una geometría realizada | **cierto sólo en 2D.** En 3D es estable la ley (∝ r^(d−2)), no el número |
| π **distinto entre** geometrías | **SOSTENIDO.** 2,0 / 3,0 / 1,5 — y ahora se sabe que el discriminante es la **ley de crecimiento de la frontera**, que es el invariante honesto |
| π **indefinido donde no cuajó** ninguna | **NO DISTINGUE.** El barajado y el ER dan lo mismo |

La tesis del director **no queda tocada** por esto: que una constante geométrica sea huella de la geometría
que se realizó sigue siendo exactamente lo que muestran 2,0 vs 3,0 vs 1,5. Lo que cae es el **instrumento**
para la tercera parte, no la idea. Y el arreglo hacia adelante es directo: el observable que sí es
contingente-a-la-geometría y sí distingue es la **ley** `|S(r)| ∝ r^(d−1)` — su exponente `d` es lo que hay
que medir, y ése sí tiene un NULL con sentido (una retícula barajada pierde el exponente; una retícula no).

---

## 10. ARCHIVOS

| archivo | qué es |
|---|---|
| `cs091_pi_contingente_rerun.py` | la medida: BFS, π(r), retículas, mundo-pequeño, NULL barajado, NULL ER, anti-isomorfismo |
| `cs091_pi_analisis.py` | descomposición algebraica, 20 réplicas barajadas con z, barrido de los 254 grafos |
| `cs091_pi_geometria_emergente.py` | la celda que faltaba: retícula cúbica 3D + regeneración de CS066 `local` |
| `cs091_pi_figura.py` | la figura (lee los CSV, no recalcula) |
| `pi_contingente_rerun_curvas.csv` | **π(r) por sustrato y por radio** (324 filas) + `|S(r)|`, `|B(r)|`, `b(r)`, mediana e IQR entre fuentes |
| `pi_contingente_rerun_emergente.csv` | π(r) del calibrador 3D y de los sustratos CS066 (84 filas) |
| `pi_contingente_rerun_algebra.csv` | π observada vs π puramente algebraica, fracción explicada |
| `pi_contingente_rerun_barajado.csv` | real vs 20 réplicas barajadas, z por radio |
| `pi_contingente_rerun_controles.csv` | tabla anti-isomorfismo |
| `pi_contingente_rerun_corpus.csv` | los 254 grafos ordenados por constancia de π |
| `pi_contingente_rerun_curvas.png` | los tres paneles |

---

**NO ES CIERRE.** Son números y curvas. Ningún veredicto sobre el nodo del 16-jul es válido sin la
autorización del director. Lo que hay acá es: las cuatro filas reproducidas, el control que faltaba corrido,
la aritmética separada del hallazgo, y una objeción de método (el detector es de 2D) que no estaba prevista.

— CC, 13-ago-2026 🐝
