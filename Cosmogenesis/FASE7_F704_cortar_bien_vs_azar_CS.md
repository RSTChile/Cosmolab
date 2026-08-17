# F7-04 — ¿Importa *cuáles* relaciones se pierden, o sólo *cuántas*?

**Fase VII · 12 de agosto de 2026 · Preparado por CC (Claude Code) · Dirige: Alexis López Tapia**
**Continúa de:** `INFORME_EQUIPO_FASE6_11ago2026_CS.md` (Partes 3 y 3.bis) y
`FASE6_O3E_memoria_vs_sinmemoria_CS.md` (que dejó este control explícitamente pendiente).
**Se lee junto con:** `FASE7_F702_escalera_clustering_CS.md` — §6 de este informe resuelve, con números,
si los dos resultados se contradicen (no se contradicen).

> **No se declara cierre ni veredicto.** Todo lo de abajo son números, con sus archivos en disco.
> Ningún script congelado fue modificado. No se hicieron commits.

---

## 0. En simple, con analogía

A un pueblo ya construido hay que cerrarle exactamente 156 calles. Un brazo cierra las que el urbanista
habría cerrado. Otro cierra 156 sacadas de un sombrero. Otro cierra justo las 156 que el urbanista habría
defendido primero. Los cinco pueblos terminan con **exactamente el mismo número de calles** — se contó,
no se supuso. Después se llenan todos de arena igual y se los sacude igual.

**Resultado: el pueblo del urbanista no junta más arena que el pueblo sorteado.** Con el mismo déficit
exacto de calles, elegir *cuáles* cerrar con el criterio de C2 no se distingue de cerrarlas a ciegas.

Pero hay un segundo resultado, y es el que le da sentido al primero: **el criterio de C2, aplicado a ese
3% de calles, casi no cambia la forma del pueblo.** La perilla que F7-02 demostró que sí mueve la arena
—los "triangulitos"— acá se mueve **47 veces menos** de lo que F7-02 la movió. Este nulo no dice que la
estructura no importe; dice que **elegir cuáles aristas cortar es una palanca demasiado corta** para
mover la estructura que importa.

---

## 1. Qué se preguntó, y por qué esta pregunta y no otra

La Fase VI dejó la línea A2-B0-C2 con un problema incómodo. Casi todo el "efecto de clase" que veníamos
midiendo resultó ser **densidad de aristas**: descontándola, la ventaja de clase quedó indistinguible de
cero en las dos resoluciones probadas (p=0.22 y p=0.41). Lo único que sobrevivió fue un **residual chico
(~5%, O3-B)** correlacionado con el **clustering** (ρ=0.77) y **no** con la pendiente (ρ=0.04).

Ese residual admitía dos lecturas:

- **(a) importa CUÁLES relaciones se pierden** — C2 corta con criterio, y cortar con criterio deja una
  trama distinta de la que deja cortar a ciegas;
- **(b) importaba sólo CUÁNTAS** — y todo lo demás era la densidad disfrazada.

F7-04 separa (a) de (b) por construcción: **el mismo grafo de partida, el mismo número exacto de aristas
al final, y sólo cambia el criterio de qué se corta.**

---

## 2. Diseño — dónde se bifurca, exactamente

`cs090_fase5_motor.dinamica_B0` (congelado) corre 14 sweeps con recableo co-emergente cada 3 pasos y
`_enforce_kcap` cada 4, y **al final** aplica una poda global por costo: `_costo_y_podar` conserva las
aristas de costo ≤ percentil 70. Esa poda final es el único punto donde C2 elige de una sola vez.

`cs090_fase7_f704_brazos.py` re-ejecuta esa cadena llamando a las **mismas funciones del motor**, en el
mismo orden y con el mismo generador aleatorio, pero **se detiene justo antes de la poda final** y expone
`edges`, `flip_count`, `E_estado` y `triangles`. Desde ahí se abren los brazos.

### Los cinco brazos (todos parten del mismo grafo pre-poda y quitan el mismo `n_cut`)

| brazo | qué quita | qué representa |
|---|---|---|
| `c2` | las de **costo > P70** | lo que C2 corta de verdad |
| `azar` | `n_cut` al azar | "cortar cualquier cosa" |
| `anticosto` | las de **menor costo** | opuesto **literal** de la poda final de C2 |
| `soporte` | las de **menor soporte local** | el criterio de `_enforce_kcap` como poda (**conserva todos los triángulos**) |
| `antisoporte` | las de **mayor soporte local** | opuesto del criterio **estructural** de C2 (**destruye todos los triángulos**) |

**Cómo se definió "opuesto", y por qué las dos variantes.** C2 no tiene un criterio, tiene dos, y actúan
en momentos distintos: `_costo_y_podar` decide por *costo* (historia de parpadeo + conflicto de
holonomía) y `_enforce_kcap` decide por *soporte local* (vecinos compartidos = triángulos). Invertir sólo
el costo habría dejado intacto justo el eje que la Fase VI señaló como correlato del residual — el
clustering vive en el soporte local. Se hicieron los dos ejes, y el de soporte **con su par positivo**
(`soporte`), para que `antisoporte` tuviera contra qué compararse dentro del mismo eje.

Los tres brazos rankeados se aplican en el **mismo punto** (la poda final) para que el número de aristas
quede exactamente igual: intervenir `_enforce_kcap` dentro del bucle cambiaría la dinámica aguas abajo y
con ella el número final de aristas — se perdería justamente la variable que este experimento fija.

**Muestra:** 12 grafos base, 3 por cada uno de los 4 lotes de `seed_base` (271828 / 371828 / 471828 /
571828), tomando dentro de cada lote la pendiente corregida **mínima, mediana y máxima** (rango cubierto
0.479 – 1.255). **En ningún punto se usa "% Clase III"** como endpoint ni como criterio: los endpoints
son continuos (fracción de masa, clustering, pendiente).

**Phantom:** N=2000, masa total fija 18800, `rho_crit_cgs=1000`, `icreate_sinks=1`, `r_crit=0.6`,
`h_acc=0.3`, `tmax=0.500`, `dtmax=0.001` — los mismos de toda la jerarquía CS073, garantizados por
**reuso** del runner de Fase V-B (`cs090_fase5b_correr.main`), no por copia. Diámetro siempre con
`cs090_diam_corregido.diam_gigante`. 60 corridas (12 × 5), las 60 completas hasta `cosmog_00500`.

---

## 3. Verificaciones — el punto entero del experimento

**M idéntico en los cinco brazos: 12 de 12 grafos.** No se asumió: se comparó el número exacto de aristas
de los cinco, y en el análisis se volvió a leer desde el `meta_regla.json` de cada corrida de Phantom,
que es un camino independiente del script que las generó.

| grafo base | lote | kcap | M pre-poda | n_cut | % cortado | **M final (los 5 brazos)** | aristas con soporte ≥1 |
|---|---|---|---|---|---|---|---|
| A2-B0-C2-r9 | 271828 | 7 | 4031 | 92 | 2.28% | **3939** | 33 |
| A2-B0-C2-r19 | 271828 | 7 | 3835 | 156 | 4.07% | **3679** | 36 |
| A2-B0-C2-r14 | 271828 | 5 | 3418 | 122 | 3.57% | **3296** | 108 |
| A2-B0-C2-r2 | 371828 | 6 | 4219 | 125 | 2.96% | **4094** | 90 |
| A2-B0-C2-r28 | 371828 | 6 | 4184 | 237 | 5.66% | **3947** | 216 |
| A2-B0-C2-r20 | 371828 | 6 | 3883 | 177 | 4.56% | **3706** | 65 |
| A2-B0-C2-batch3-r9 | 471828 | 6 | 4254 | 185 | 4.35% | **4069** | 125 |
| A2-B0-C2-batch3-r21 | 471828 | 5 | 3432 | 124 | 3.61% | **3308** | 66 |
| A2-B0-C2-batch3-r100 | 471828 | 4 | 2457 | 83 | 3.38% | **2374** | 137 |
| A2-B0-C2-batch4-r18 | 571828 | 6 | 4191 | 205 | 4.89% | **3986** | 79 |
| A2-B0-C2-batch4-r43 | 571828 | 5 | 3388 | 119 | 3.51% | **3269** | 56 |
| A2-B0-C2-batch4-r51 | 571828 | 4 | 2592 | 109 | 4.21% | **2483** | 84 |

Otras verificaciones, **12/12 en todas**:

- **el brazo `c2` reproduce el motor congelado arista por arista** — se corrió aparte `MOT.dinamica_B0`
  de punta a punta y se comparó el conjunto de aristas: **0 diferencias**;
- **la réplica del vector de costos reproduce el `conservar` de `MOT._costo_y_podar`** (assert duro);
- **verificación cruzada contra `meta_regla.json` de Fase V-B**: mismo `rule_id`, mismo `seed` y mismo
  `n_aristas_grafo_final` que la corrida histórica de esa regla;
- los cinco grafos de cada bloque se **canonicalizan** antes de medir y antes de escribir la condición
  inicial, para que ninguna diferencia venga del orden accidental de los `set` de Python (lección O3-B).

### 3.1 Un hallazgo de método, de paso: la poda final de C2 corta 3%, no 30%

`_costo_y_podar` usa `costo <= percentil 70`. Suena a "tira el 30% más caro". **No lo hace.** El costo
toma sólo 14-40 valores distintos sobre 2.457-4.254 aristas, con el **92-97% empatadas en un único valor
modal**; como el corte es `<=`, todo el bloque empatado se conserva y la poda efectiva queda en
**2.3%-5.7%** (83-237 aristas). El soporte local es aún más degenerado: prácticamente **binario (0 ó 1)**
— con kcap 4-7 a N=2000 el grafo es casi sin triángulos y sólo 33-216 aristas cierran alguno.

Cuando el ranking no alcanza a llenar el cupo, el resto se completa **al azar dentro del grupo empatado**
(no por índice, que sesgaría hacia los nodos bajos), y cada brazo informa cuántas aristas salieron del
ranking estricto y cuántas del relleno. Por eso `anticosto` es una inversión **parcial** (~26% ranking
estricto) mientras `soporte`/`antisoporte` son inversiones **completas** en su eje: `antisoporte` saca el
100% de las aristas que cierran triángulo.

**Consecuencia para toda la línea:** cualquier lectura previa que haya supuesto que C2 "poda un tercio de
las relaciones" describe algo que el código no hace. Lo que efectivamente ralea el grafo en C2 es
`_enforce_kcap`, no `_costo_y_podar`.

### 3.2 Lo que este diseño **no** fija: la secuencia de grados (y cuánto se mueve)

A diferencia de F7-02 —que mueve el clustering con double-edge-swaps y por eso deja el grado de cada nodo
**clavado**— F7-04 **quita** aristas, y quitar aristas distintas cambia el grado de los nodos tocados.
Está medido (`cs090_fase7_f704_grados.py` → `cs090_fase7_f704_grados.csv`), no supuesto:

| comparación | nodos con grado distinto | media \|Δgrado\| | Spearman de las secuencias |
|---|---|---|---|
| c2 vs azar | 21.7% | 0.249 | 0.924 |
| c2 vs anticosto | 20.8% | 0.260 | 0.912 |
| c2 vs soporte | 22.0% | 0.254 | 0.922 |
| c2 vs antisoporte | 16.3% | 0.191 | 0.943 |

Lo que **sí** queda fijo: N=2000, el número de aristas (12/12 exacto), la suma de grados (=2M, idéntica
por construcción, verificada con assert) y por lo tanto **el grado medio (3.512, idéntico en los cinco
brazos)**. Y la *forma* de la distribución de grados prácticamente no se mueve: el desvío estándar del
grado es 1.5382 / 1.5339 / 1.5400 / 1.5343 / 1.5450 en c2 / azar / anticosto / soporte / antisoporte —
una diferencia del 0.7% entre extremos. **Los brazos difieren en qué nodo pierde el vecino, no en cómo se
reparten los grados en conjunto.** Esa es la diferencia de diseño con F7-02, y es la razón por la que los
dos experimentos pueden convivir (§6).

---

## 4. Que la manipulación estructural funcionó — y cuánto

Con **el mismo número de aristas y el mismo grado medio en los cinco brazos**, la trama sí quedó distinta.
El clustering abarca un rango de **82× en proporción** entre el brazo que conserva todos los triángulos y
el que no deja ninguno:

| brazo | aristas | grado medio | clustering local (media ± sd entre grafos) | pendiente corregida | **fracción de masa** | κ_V | nº sumideros |
|---|---|---|---|---|---|---|---|
| `soporte` | 3512.5 | 3.512 | **0.00903 ± 0.00745** (0.00182–0.02675) | 0.7366 ± 0.231 | **0.09742** | 0.5903 | 8.08 |
| `azar` | 3512.5 | 3.512 | 0.00819 ± 0.00675 (0.00143–0.02458) | 0.7217 ± 0.203 | 0.09729 | 0.5734 | 8.08 |
| `anticosto` | 3512.5 | 3.512 | 0.00397 ± 0.00325 (0.00048–0.01083) | 0.7362 ± 0.199 | 0.09663 | 0.5865 | 8.00 |
| `c2` | 3512.5 | 3.512 | 0.00511 ± 0.00463 (0.00100–0.01650) | 0.7348 ± 0.219 | 0.09638 | 0.5656 | 8.00 |
| `antisoporte` | 3512.5 | 3.512 | **0.00011 ± 0.00038** (0.00000–0.00133) | 0.7464 ± 0.226 | **0.09579** | 0.5719 | 8.00 |

**Rango de clustering DENTRO de cada grafo (el contraste que importa, porque es donde M está fijo):
media 0.00894**, mínimo 0.00182, máximo 0.02542. Rango de pendiente dentro de cada grafo: 0.105.

Guardar esos dos números: **0.0089 de rango en clustering** es la palanca real que tuvo este experimento.

---

## 5. Resultado — la pregunta central

### 5.1 C2 **no** le gana ni al azar ni al anti-C2

Comparación pareada grafo por grafo, endpoint primario (fracción de masa acretada en sumideros), n=12:

| contraste | Δ medio | Δ relativo | signos | p (signos) | p (Wilcoxon) |
|---|---|---|---|---|---|
| **c2 vs azar** | −0.00092 | **−0.94%** | 4/11 | 0.549 | 0.577 |
| **c2 vs anticosto** | −0.00025 | −0.26% | 5/12 | 0.774 | 1.000 |
| c2 vs soporte | −0.00104 | −1.07% | 4/12 | 0.388 | 0.339 |
| c2 vs antisoporte | +0.00058 | +0.61% | 7/12 | 0.774 | 0.424 |
| azar vs anticosto | +0.00067 | +0.69% | 6/10 | 0.754 | 0.332 |
| azar vs antisoporte | +0.00150 | +1.57% | 7/11 | 0.549 | 0.130 |
| **soporte vs antisoporte** | **+0.00163** | **+1.70%** | **8/10** | 0.109 | 0.160 |

**El orden predicho C2 > azar > anti-C2 se cumple en 2 de 12 grafos** con `anticosto` (lo esperado por
azar es 2.0) y en **1 de 12** con `antisoporte`. **Omnibus de Friedman** sobre los 5 brazos con bloques
por grafo: **χ²=4.64, p=0.326** en fracción de masa; χ²=4.00, p=0.406 en número de sumideros. **No hay
efecto de brazo detectable en la masa.**

### 5.2 El eje de los triángulos deja, aun así, una señal continua

Los brazos no se separan de a pares, pero la relación **continua** —cuánto clustering quedó vs cuánta
masa se acretó, **centrando dentro de cada grafo** para que el contraste sea sólo entre brazos del mismo
bloque, donde el nº de aristas está fijo por diseño— sí tiene señal:

- **clustering vs fracción de masa: ρ = +0.310**, y con el null correcto (permutar la masa **sólo dentro
  de cada grafo**, 20.000 permutaciones) **p = 0.012** a una cola;
- pendiente corregida vs masa: ρ = −0.277 (p=0.032); las dos variables **no** son la misma cosa acá
  (entre sí ρ = −0.116) y ambas sobreviven como parciales: clustering|pendiente r=+0.291 (p=0.024),
  pendiente|clustering r=−0.255 (p=0.049);
- por grafo, la correlación intra-bloque es **positiva en 9 de 12** (mediana ρ=+0.385), pero con n=12 el
  test de signos da **p=0.146** — no alcanza sola.

### 5.3 El límite de resolución, dicho de frente

Con masa total fija 18800 y N=2000, **un sumidero que acreta una partícula mueve la fracción de masa en
0.0005**. Los efectos que perseguimos son de 0.0010-0.0016: **2 a 3 partículas**. Estamos midiendo cerca
del grano del instrumento, y eso limita la potencia de los contrastes pareados.

---

## 6. ¿Contradice esto a F7-02? — No. Y los números dicen por qué

F7-02, con la **secuencia de grados exactamente idéntica nodo por nodo**, encontró que el clustering
mueve la masa causalmente (12/12 grafos, ρ=+0.965, Page L p=1.3×10⁻¹¹, +37% en todo el rango). F7-04, con
el número de aristas exactamente idéntico, no encuentra nada. La pregunta obligada es si los brazos de
F7-04 **difirieron o no en clustering**. Difirieron — pero en una escala completamente distinta:

| | rango de clustering recorrido | Δ masa observado |
|---|---|---|
| **F7-02** (escalera, grados fijos) | 0.000 → **0.418** (media del escalón e4) | +0.0394 (0.1059 → 0.1453) |
| **F7-04** (este trabajo, M fijo) | 0.00011 → **0.00903** | +0.00163 |

**La palanca de F7-04 es 47 veces más corta que la de F7-02** (0.0089 contra 0.418). Y el escalón *más
chico* de F7-02 —de e0 (C=0) a e1 (C=0.029)— ya es **3.3 veces más grande** que todo el rango que este
experimento pudo recorrer.

Ahora la prueba cuantitativa. Tomando la pendiente masa-vs-clustering que midió F7-02:

- pendiente **local** en la base de la escalera (e0→e1): **0.0716** de fracción de masa por unidad de
  clustering → predice, para el rango de F7-04 (0.00894): **+0.00064**;
- pendiente **global** (e0→e4): 0.0941 → predice **+0.00084**;
- **observado en F7-04** (`soporte` − `antisoporte`, que es exactamente ese rango): **+0.00163**.

**El efecto observado en F7-04 tiene el mismo signo que F7-02 y es 2-2,5× su predicción — es decir, cae
del lado grande de lo esperado, no del lado chico.** Traducido a partículas: F7-02 predice ~1,3
partículas de diferencia y se midieron ~3,3, sobre un instrumento cuyo grano es 1 partícula. Con 12
grafos, un efecto de ese tamaño no puede alcanzar significancia pareada: 8/10 signos, p=0.109 es
exactamente lo que se espera de una señal real de ese tamaño con esa muestra.

**Conclusión de la comparación: los dos experimentos son consistentes.** El nulo de F7-04 **no** es un
nulo sobre la estructura. Es un nulo sobre **la identidad de la arista**: cambiar *cuáles* aristas se
cortan, dentro del 3% que C2 corta, **casi no mueve las variables estructurales que sí importan**, y por
eso no mueve la masa. F7-02 movió esas variables 47 veces más y la masa respondió.

Dicho al revés, que es la forma útil: **si F7-04 hubiera encontrado un efecto grande, ése sí habría
estado en tensión con F7-02** —habría significado que la masa responde a algo que no es el clustering ni
el grado. No lo encontró.

---

## 7. Lectura (no un veredicto)

```
mismo grafo + mismo nº EXACTO de aristas (12/12 verificado) + mismo grado medio
        │
        ├─ criterio de C2 (costo) vs azar ──────────► NO se separa (−0.94%, p=0.55)
        │      · su inversión literal tampoco ───────► NO se separa (−0.26%, p=1.00)
        │      · omnibus de los 5 brazos ────────────► p=0.33
        │
        ├─ eje de triángulos (soporte local) ───────► +1.70% conservarlos vs destruirlos
        │      8/10 signos (p=0.11); relación continua ρ=+0.31, permutación por bloques p=0.012
        │      → consistente con F7-02: predicho +0.0006/+0.0008, observado +0.0016
        │
        └─ POR QUÉ el nulo: la palanca de clustering que da re-elegir el 3% de aristas
           es 47× más corta que la de F7-02 (0.0089 vs 0.418)
```

Lo que estos números sostienen, sin adornos:

1. **La respuesta a la pregunta tal como se formuló es negativa.** Con el mismo déficit exacto de
   aristas, **elegir con el criterio de C2 no rinde más que elegir al azar**. La parte del efecto que
   atribuíamos a "cortar bien" no aparece cuando se fija el número de cortes.
2. **Pero el nulo es sobre la identidad de la arista, no sobre la estructura.** Los cinco brazos quedaron
   estructuralmente muy parecidos en términos absolutos (rango de clustering 0.0089, grado medio
   idéntico, distribución de grados a 0.7% entre extremos). El experimento no tenía cómo mover la masa.
3. **El criterio de costo de C2 no es el vehículo del residual de O3-B.** El eje que sí deja huella
   —conservar o destruir los triángulos— es el que F7-02 ya estableció como causal. Y C2 toca ese eje sólo
   de refilón, a través de `_enforce_kcap`, cuyo cupo casi nunca se activa en estos grafos. Para la
   teoría esto es incómodo en un punto preciso: el criterio "caro" (historia de parpadeo + conflicto de
   holonomía), que es el que el marco pondría en el centro, **no es el que mueve la gravedad**.
4. **Aviso de método:** la poda "P70" de C2 corta el **3%**, no el 30% (§3.1).

### Lo que este experimento **no** dice

No dice que el criterio de C2 sea inútil: dice que, **aplicado al 3% final y con el número de aristas
fijo**, no cambia la masa acretada de forma detectable con 12 grafos a esta resolución. No dice que el
clustering no sea el mecanismo — al contrario, es consistente con que lo sea (§6), y F7-02 es la
evidencia fuerte de eso, no éste. Y no toca la densidad, que sigue siendo el efecto grande de la Fase VI:
acá está fijada por diseño, precisamente para sacarla del medio.

---

## 8. Qué se podría hacer después (sin cerrar nada)

- **Subir `n_cut` manteniendo M idéntico.** El déficit acá es 3-5% porque así corta C2. Un barrido de
  dosis (5%, 15%, 30%) con los mismos cinco criterios diría si el empate es real o si la palanca era
  corta. Es barato: mismo pipeline, sólo cambia `n_cut`. **Es el seguimiento más directo de este informe.**
- **Grafos con más triángulos.** Acá el soporte local es binario porque `kcap` 4-7 deja el grafo casi sin
  triángulos; con `kcap` mayor el eje de soporte tendría rango real en vez de 0/1, y `soporte` vs
  `antisoporte` recorrería una fracción mucho mayor de la escalera de F7-02.
- **Sólo `soporte` vs `antisoporte`, 30-40 grafos.** Con 8/10 signos, ese contraste está a un factor ~3
  de muestra de ser concluyente por sí mismo.
- **Resolución.** A N=2000 el endpoint está cuantizado en 0.0005 y perseguimos 0.0016. N=4000 duplicaría
  la resolución (O3-A ya midió que es practicable; N=8000 no lo es).

---

## Archivos

**Scripts (nuevos; ningún archivo existente fue modificado — todos sólo se importan):**
- `cs090_fase7_f704_brazos.py` — selección, fork del motor, los 5 brazos, verificaciones, métricas, IC.
- `cs090_fase7_f704_correr.py` — corre Phantom (importa el runner de Fase V-B, no lo reimplementa).
- `cs090_fase7_f704_analizar.py` — lectura de volcados (importa el lector de Fase V-B), pareados,
  Friedman, correlaciones parciales, permutación por bloques.
- `cs090_fase7_f704_grados.py` — divergencia de la secuencia de grados entre brazos (§3.2).

**Datos:**
- `cs090_fase7_f704_seleccion.csv` — los 12 grafos base y por qué se eligieron.
- `cs090_fase7_f704_estructura.csv` — **CSV crudo estructural**: por grafo base, los 5 brazos, todas las
  verificaciones (M idéntico, c2==motor, meta5b) y la degeneración de los criterios.
- `cs090_fase7_f704_phantom_crudo.csv` — **CSV crudo de Phantom**: una fila por corrida (60).
- `cs090_fase7_f704_pares.csv` — una fila por grafo base con los 5 brazos lado a lado.
- `cs090_fase7_f704_estadistica.csv` — los 30 contrastes pareados (3 métricas × 10 pares de brazos).
- `cs090_fase7_f704_grados.csv` — secuencias de grados comparadas brazo contra brazo.
- Logs: `cs090_fase7_f704_analisis.log`, `cs090_fase7_f704_shard*.log`,
  `cs090_fase7_f704_phantom_shard*.log`.
- Corridas de Phantom: `/Users/alexis/phantom_cs073/bateria_fase7_f704_cortar_bien/` (60 carpetas,
  prefijo `_f704_`, sin colisión con ninguna batería previa).
