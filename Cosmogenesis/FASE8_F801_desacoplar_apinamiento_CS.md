# FASE VIII · F8-01 — Desacoplar las cuatro medidas de apiñamiento: ¿cuál manda?

**Fecha:** 12-ago-2026 · **Ejecuta:** CC (Claude) · **Tarea:** F8-01, experimento #1 de Fase VIII
**Antecedente directo:** `FASE7_F703_grados_y_triangulos_fijos_CS.md` (+13.8%, 12/12, 28.7 partículas)
**Insumo en paralelo:** F8-00 (`cs090_fase8_f800_correlaciones.csv`, utilidad `cs090_fase8_f800_grafos.py`)
**Reglas de la fase:** `FASE8_PLAN_EJECUCION_CS.md` · **Phantom:** autorizado
**Grano del instrumento:** 1 partícula = 0.0005 de fracción de masa

> No se declara cierre ni veredicto. Ningún script congelado fue modificado (todos sólo se importan).
> No se hicieron commits de git.

---

## 0. En simple, con analogía

F7-03 dejó probado esto: con la misma cantidad de nudos, el mismo número de alambres en cada nudo y
**exactamente la misma cantidad de triangulitos**, la maqueta junta 13.8% más arena si esos triangulitos
están **apretados unos contra otros** que si están sueltos.

Pero "apretados" se puede medir de cuatro maneras, y en F7-03 las cuatro se movían juntas:

1. cuántos triangulitos se apoyan **en la misma varilla** (media),
2. cuán **desparejo** está el reparto entre postes (Gini),
3. qué fracción de varillas sostiene **más de un** triangulito (solapamiento),
4. **sobre cuántas varillas distintas** se apoya todo el conjunto (soporte).

Esta tarea fabrica maquetas donde se mueve **una** de esas cuatro y las otras quedan quietas, para ver
cuál es la que la arena realmente sigue.

**Lo que salió:** la arena sigue a **la varilla**, no al poste. Mover el reparto entre postes dejando las
varillas quietas **no mueve nada** (0.0 partículas). Y las "varillas supercargadas" —la cola— tampoco
agregan nada por encima del promedio: lo que manda es **sobre cuántas varillas distintas se reparte el
total**, que es lo mismo que decir cuántos triangulitos hay por varilla.

---

## 1. Lo primero: dos de las cuatro medidas **no son dos medidas**

Antes de gastar una sola corrida hay un hecho de álgebra. Con el nº de aristas `m` fijo y el nº de
triángulos `T` fijo, la carga total sobre las aristas es exactamente `Σ_e t_e = 3T`. Entonces, con
`D` = fracción de aristas que sostienen algún triángulo y `A` = media de triángulos por esas aristas:

> **A · D · m = 3T**, con `T` y `m` fijos por diseño → **A = 3T/(D·m)**.

`tri_por_arista_media` y `frac_aristas_en_triangulo` son **la misma variable y su inversa**. Ningún
experimento puede separarlas mientras grados y triángulos estén clavados. Eso explica, sin misterio, por
qué en F7-03 salieron con ρ casi especular (+0.776 y −0.781).

**Verificado numéricamente en los 40 grafos de esta batería:** el residuo `|A·D·m − 3T|` máximo es
**4.55e-13** (columna `identidad_A_D_resid` del CSV de estructura). Es cero de máquina.

F8-00 llegó a lo mismo desde los datos naturales (ρ=1.000 exacto entre pares equivalentes). **Coinciden
dos caminos independientes: son identidades, no hallazgos.**

Quedan entonces, como mucho, **tres** ejes:

| eje | qué es | medidas |
|---|---|---|
| **(1) soporte** | sobre cuántas aristas distintas se reparte el total | `D` ⇔ `A` (una sola cosa) |
| **(2) forma** | a soporte dado, ¿la carga extra está pareja o en pocas aristas? | `frac_aristas_multi_tri` (C), `tri_por_arista_max` (E) |
| **(3) nodo** | cuán desparejo es el reparto entre **nodos** | `gini_tri_nodo` (B), `frac_nodos_en_triangulo` |

Los ejes (1)+(2) son el "Eje 2" de F8-00 (apilamiento por arista); el eje (3) es parte de su "Eje 1".

Y dos cotas duras más, que limitan lo fabricable:

- **A ≥ 1 + C** (si una fracción C de las aristas del soporte lleva ≥2, la media no puede ser menor).
- **A ≤ E**, y **E ≤ kcap − 1 = 7**: una arista tiene tantos triángulos como vecinos comunes. La
  propuesta del equipo de "máximo 6 contra máximo 30" **no es fabricable** con estos grafos. Lo
  fabricable es "máximo 1 contra máximo 4", y así se hizo.

---

## 2. Qué se hizo, con qué archivos

| Archivo nuevo | Qué hace |
|---|---|
| `cs090_fase8_f801_desacople.py` | Piso común, cinco brazos (tres nuevos + dos importados de F7-03 tal cual), igualación exacta del nº de triángulos, verificación de grados nodo por nodo, medidas de organización + **la cola de la distribución por arista** (que F7-03 no medía), guardado del grafo, escritura de condiciones iniciales |
| `cs090_fase8_f801_correr.py` | Corre Phantom (mismo protocolo exacto de toda la línea) |
| `cs090_fase8_f801_analizar.py` | Verificación cruzada contra `meta_regla.json`, **matriz de desacople**, estadística pareada, correlaciones parciales, PNG |
| `cs090_fase8_f801_guardar_canonico.py` | Reescribe los 40 grafos en el formato canónico de F8-00 con su `sha256` |

| CSV / PNG de salida | Contenido |
|---|---|
| `cs090_fase8_f801_estructura_shard{0..7}.csv` | estructura y organización de cada brazo (crudo) |
| `cs090_fase8_f801_phantom_crudo.csv` | **una fila por corrida de Phantom** (CSV crudo pedido) |
| `cs090_fase8_f801_matriz_desacople.csv` | **cuánto se desacopló de verdad**, medida por medida y contraste por contraste |
| `cs090_fase8_f801_por_grafo.csv`, `_estadistica.csv`, `_correlaciones.csv` | pareados, pruebas, ρ |
| `cs090_fase8_f801_desacople.png` | el resultado dibujado (4 paneles) |

Batería en `/Users/alexis/phantom_cs073/bateria_fase8_f801_desacople/`, carpetas
`<rule_id>_s<seed>_f801_<brazo>` — **prefijo `f801` jamás usado antes**, con el **seed dentro del
nombre** (bug de colisión de nombres documentado en `FASE6_O3B` §2.1). 1.1 GB, 40 corridas.

**Scripts sólo importados, nunca tocados:** `cs090_fase7_f703_organizacion.py` (motor de brazos,
`replay`, `techo_conectado`, `medir_organizacion`, `_ok_disjunto`), `cs090_fase7_f702_escalera.py`
(swap elemental que conserva grados + selección de grafos base), `cs090_fase6_o3b_rewiring.py`,
`cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_analizar.py`, `cs090_fase7_f702_analizar.py`,
`cs090_diam_corregido.py`, `cs080_renormalizacion.py`, `cs090_fase8_f800_grafos.py`.

**Grafos guardados** (regla de la fase): cada carpeta tiene `grafo_f801.npz` (numpy: `aristas` E×2
int32 con i<j ordenado, `grados` N int32) **y** `grafo_f801.grafo.gz`, el formato canónico de F8-00 con
su `sha256` anotado dentro del `meta_regla.json` de la carpeta.

---

## 3. El diseño: cinco brazos, un piso común, el mismo T

**8 grafos base** (`F702.seleccionar_grafos(n_por_lote=2)`, la misma función que F7-02/F7-03): las 2
reglas de mayor pendiente corregida de cada uno de los 4 lotes. Todas Clase III del linaje A2-B0-C2.

De cada grafo se destruyen los triángulos con `F702.bajar_clustering` (misma función y dosis), y desde
**ese mismo piso** los cinco brazos vuelven a construir triángulos con el **mismo swap dirigido**
`(x-p),(y-q) → (x-y),(p-q)`, que **conserva el grado de los cuatro nodos**. Lo único distinto es el
criterio de aceptación:

| brazo | criterio | qué produce |
|---|---|---|
| `abanico` | **NUEVO.** Empaquetamiento disjunto en aristas (mismo `_ok_disjunto` de F7-03) **más** ápice sorteado de una bolsa de nodos ya calientes: los triángulos comparten **vértice** y **nunca arista** | molinos de viento: Gini alto con solapamiento clavado en 0 |
| `disp` | **IMPORTADO tal cual** de F7-03: cupo de triángulos por nodo que sube por capas | reparto parejo, también casi sin compartir aristas |
| `cola` | **NUEVO.** Todo crece disjunto salvo en poquísimas aristas "núcleo" (prob. 0.02 de abrir una), que sí pueden cargar hasta 8 | media baja con **cola larga**: pocas aristas muy cargadas |
| `malla` | **NUEVO.** Debe compartir arista con un triángulo existente, **con tope exacto de 2** por arista afectada | muchas aristas con exactamente 2, **sin cola** |
| `solap` | **IMPORTADO tal cual** de F7-03 | ancla: reproduce el brazo ganador de F7-03 |

Igualación del nº de triángulos: idéntica a F7-03 (lista de swaps aceptados + `replay` + `T* = mínimo
de los cinco techos`). **Todo lo demás idéntico:** N=2000, masa total fija 18800, mismo lado de caja,
**mismo `layout_resortes` con `seed_layout=12345` en los cinco brazos**, misma dilatación, misma
turbulencia, `icreate_sinks=1`, `rho_crit_cgs=1000`, `r_crit=0.6`, `h_acc=0.3`, `tmax=0.500`.

### 3.1 — Un detalle de método que hay que decir

En el piloto, el brazo `malla` chequeaba el tope sólo sobre cuatro aristas fijas y **se le escapaban**
las aristas `(a-w)` de los triángulos que cierra la segunda arista nueva: el tope era 2 y aparecían
aristas con 4. Se corrigió con `_aristas_afectadas`, que enumera **todas** las aristas cuya carga pudo
subir. Con el tope exacto, `malla` es más restrictivo y **es el brazo que fija T\***, que por eso bajó
respecto de F7-03 (34-1053, mediana 155, contra 57-1029 mediana 275).

---

## 4. Verificaciones hechas antes de mirar ningún resultado

- **Grados idénticos, nodo por nodo:** `np.array_equal(grados_original, grados_brazo)` sobre los 2000
  nodos, con `assert` que aborta el grafo si falla. Los 40 brazos pasaron.
- **Nº de triángulos: `dif_max = 0` en los 8 grafos.** No "casi igual": **exactamente el mismo número**
  en los cinco brazos de cada grafo (F7-03 había conseguido diferencia máxima 1).
- Mismo nº de aristas, sin bucles i-i, dos conteos independientes de triángulos coincidentes (`assert`).
- **Verificación cruzada contra `meta_regla.json`** de cada carpeta antes de la estadística: tarea
  declarada, brazo y `(rule_id, seed)` coincidentes con el nombre de la carpeta, carpeta declarada =
  carpeta real, `grados_identicos_al_original = true`, mismo nº de aristas / `seed_layout` / nº de
  triángulos entre los cinco brazos, 2000 partículas de gas iniciales. **0 avisos sobre 40 carpetas.**
- **Unión por `(rule_id, seed, brazo)`**, nunca por `rule_id` solo.
- **Identidad `A·D·m = 3T`** verificada en las 40 filas (residuo máx. 4.55e-13).
- 4 corridas se rescataron a mano porque la salvaguarda del runner ("el bloque de sumideros por defecto
  no coincide") saltó sobre carpetas cuyo `cosmog.in` había quedado a medio editar por un intento
  interrumpido. El rescate **verifica explícitamente** que el archivo tenga el bloque CS073 y
  `tmax=0.500`/`dtmax=0.001` antes de correr; no se editó nada a ciegas.

---

## 5. **Cuánto se desacopló de verdad** (la parte pedida por el punto 3 de la tarea)

Medias sobre los 8 grafos. Recordar: **mismo N, mismas aristas, mismos grados nodo por nodo y
exactamente los mismos 329.4 triángulos de media** en las cinco columnas.

| medida | `abanico` | `disp` | `cola` | `malla` | `solap` |
|---|---|---|---|---|---|
| **A** triángulos por arista del soporte | **1.0000** | 1.0288 | 1.0173 | 1.5618 | **2.3852** |
| **D** fracción de aristas con triángulo | **0.2746** | 0.2562 | 0.2635 | 0.1736 | **0.1052** |
| **C** aristas en >1 triángulo (solapamiento) | **0.0000** | 0.0288 | 0.0088 | 0.5618 | **0.8025** |
| **E** máximo de triángulos en una arista | **1.00** | 1.75 | **2.63** | 2.00 | 4.00 |
| **B** Gini de triángulos por nodo | 0.7543 | **0.6728** | 0.6991 | 0.8696 | **0.9352** |
| nodos que tocan algún triángulo | 0.3051 | 0.3628 | 0.3576 | 0.1673 | 0.0905 |
| máximo de triángulos en un nodo | 2.25 | 2.00 | 4.75 | 4.88 | 8.75 |
| clustering local medio (no explicativa) | 0.1022 | 0.1179 | 0.1127 | 0.0812 | 0.0662 |
| componente gigante | 1861.1 | 1860.3 | 1859.6 | 1860.0 | 1832.1 |

### 5.1 — Contraste por contraste: qué quedó fijo y qué se movió

|contraste|A|B (Gini)|C (multi)|D|E (max)|¿logrado?|
|---|---|---|---|---|---|---|
|**C1 `abanico`−`disp`**|1.000 / 1.029|**0.754 / 0.673**|0.000 / 0.029|0.275 / 0.256|1.00 / 1.75|**SÍ**: el eje de nodo se mueve con el de arista casi quieto|
|**C2 `cola`−`malla`**|1.017 / 1.562|0.699 / 0.870|0.009 / 0.562|0.264 / 0.174|**2.63 / 2.00**|**PARCIAL**: la cola se invierte (cola>malla) pero A y C no quedaron fijos|
|**C3 `abanico`−`malla`**|1.000 / 1.562|0.754 / 0.870|**0.000 / 0.562**|0.275 / 0.174|1.00 / 2.00|**PARCIAL**: mueve solapamiento, arrastra Gini +0.12|
|**C4 `solap`−`disp`** (ancla)|2.385 / 1.029|0.935 / 0.673|0.802 / 0.029|0.105 / 0.256|4.00 / 1.75|**se mueve TODO** — es el eje de F7-03|
|**C5 `cola`−`disp`** (emergente)|1.017 / 1.029|0.699 / 0.673|0.009 / 0.029|0.264 / 0.256|2.63 / 1.75|**casi gemelos**: sirve de piso de ruido|

**C1 es el contraste bueno, y es bueno por álgebra, no por suerte:** los dos brazos son
empaquetamientos casi disjuntos en aristas, así que con el mismo `T` tienen forzosamente casi el mismo
`A` (1.000 vs 1.029, **2% del rango entre brazos**), el mismo `C` (0.000 vs 0.029, **4% del rango**) y
el mismo `D` (0.275 vs 0.256, 11% del rango). Lo que se mueve es el **Gini por nodo: +0.082**, un 31%
del rango que las cinco organizaciones recorren.

**Lo que NO se pudo fabricar, y por qué:**

- **"Mismo solapamiento medio, distinto máximo (6 vs 30)":** imposible. `E ≤ kcap−1 = 7` (una arista no
  puede tener más triángulos que vecinos comunes). Lo conseguido es E = 2.63 contra 2.00 con A
  parecido... pero el brazo `cola` sólo despliega su cola cuando hay mucho material: con `T*` de 34-226
  casi no llegó a abrir núcleos y quedó **casi idéntico a `disp`**. La cola sólo se separó de verdad en
  el grafo con T*=1053 (A 1.086 vs 1.137, D 0.712 vs 0.680 —*iguales*— con **E=5 contra 3** y C=0.044
  contra 0.137). El contraste de cola **existe pero necesita muchos más triángulos de los que el brazo
  `malla` permite**; con este T\* quedó chico.
- **"Mismo Gini, distinto solapamiento":** no se logró separar. Apilar sobre aristas obliga a apilar
  sobre los nodos de esas aristas: C y B suben juntos siempre (C3 mueve C en 0.562 pero arrastra B en
  0.116). **Par ligado por construcción** en grafos de grado ≤8.
- **Modularidad / organización comunitaria global:** no se fabricó un contraste dedicado. Lo medido
  (ρ=−0.305, p=0.056, contra −0.576 en F7-03) sigue apuntando a que el "barrio" no es lo que manda.

---

## 6. El resultado: la masa sigue a la **arista**, no al **nodo**

Fracción de masa en sumideros. Una fila = un grafo; dentro de la fila **todo es idéntico salvo la
organización de los mismos triángulos**:

| regla | lote | T\* | `abanico` | `disp` | `cola` | `malla` | `solap` | rango (part.) |
|---|---|---|---|---|---|---|---|---|
| batch3-r0 | 471828 | 57 | **0.1520** | 0.1505 | 0.1510 | 0.1490 | 0.1480 | 8 |
| batch3-r60 | 471828 | 164 | 0.1215 | 0.1175 | 0.1170 | 0.1265 | **0.1300** | 26 |
| batch4-r10 | 571828 | 146 | 0.1210 | 0.1245 | 0.1250 | 0.1205 | **0.1320** | 23 |
| batch4-r36 | 571828 | 34 | 0.1485 | 0.1440 | **0.1495** | **0.1495** | 0.1450 | 11 |
| r14 | 271828 | 226 | 0.0965 | 0.0990 | 0.1015 | 0.1100 | **0.1155** | 38 |
| r17 | 271828 | 142 | 0.1175 | 0.1205 | 0.1190 | 0.1205 | **0.1285** | 22 |
| r20 | 371828 | 813 | 0.0945 | 0.1010 | 0.1035 | 0.1090 | **0.1265** | 64 |
| r39 | 371828 | 1053 | 0.0945 | 0.0890 | 0.0980 | 0.1010 | **0.1340** | 90 |
| **media** | | 329 | 0.1183 | 0.1183 | 0.1206 | 0.1232 | **0.1324** | **35.3** |

### 6.1 — Las pruebas (n=8 grafos, diseño pareado)

| contraste | Δ medio | en partículas (±EE) | signos | Wilcoxon |
|---|---|---|---|---|
| **C1 `abanico` − `disp`** (sólo Gini) | +0.00000 | **+0.0 ± 3.1** (IC95 −6.2, +6.2) | 4/8 | p = 0.95 |
| C2 `cola` − `malla` | −0.00269 | −5.4 ± 3.5 | 2/8 | p = 0.20 |
| C3 `abanico` − `malla` (solapamiento) | −0.00500 | **−10.0 ± 4.5** (IC95 −18.7, −1.3) | 2/8 | p = 0.055 |
| **C4 `solap` − `disp`** (ancla F7-03) | +0.01419 | **+28.4 ± 10.8** | 7/8 | **p = 0.023** |
| C5 `cola` − `disp` (gemelos) | +0.00231 | +4.6 ± 2.4 | 6/8 | p = 0.094 |
| `abanico` − `solap` | −0.01419 | −28.4 | 2/8 | p = 0.039 |
| `malla` − `solap` | −0.00919 | −18.4 | 2/8 | p = 0.055 |
| Friedman (5 brazos) | χ²=9.10 | — | — | p = 0.059 |

**El ancla reproduce F7-03 casi exacto: +28.4 partículas acá contra +28.7 allá.** El instrumento y el
pipeline están donde estaban; lo que cambió es qué se comparó.

### 6.2 — El número central: **C1 = 0.0 partículas**

Mover el Gini por nodo **+0.082** (31% del rango entre organizaciones) con las tres medidas de arista
prácticamente clavadas produce **cero cambio de masa: +0.0 ± 3.1 partículas**, 4/8 signos.

Y hay una predicción cuantitativa que se puede contrastar. Si el Gini fuera el motor, la pendiente del
ancla (28.4 partículas por Δgini=0.262) da **108 partículas por unidad de Gini**, y entonces C1 debería
haber dado **+8.8 partículas**. El intervalo de confianza del 95% observado (−6.2, +6.2) **excluye esa
predicción**. Con n=8 no es una demolición, pero va en contra.

### 6.3 — Las parciales dicen lo mismo, y desde otro lado

Spearman parcial sobre las 40 corridas, centrando cada valor en la media de su propio grafo:

| relación | ρ parcial | p |
|---|---|---|
| masa vs **D (soporte)**, descontando el Gini | **−0.525** | 0.0005 |
| masa vs **Gini**, descontando D | **−0.107** | 0.51 |
| masa vs **A**, descontando el Gini | +0.394 | 0.012 |
| masa vs Gini, descontando A | +0.276 | 0.085 |
| **masa vs A, descontando el máximo por arista (E)** | **+0.654** | 4.7e-06 |
| **masa vs E (la cola), descontando A** | **−0.005** | 0.975 |

Dos lecturas, las dos en el mismo sentido:

1. **Descontar el soporte mata al Gini** (−0.107, p=0.51); **descontar el Gini no mata al soporte**
   (−0.525, p=0.0005). El Gini viajaba de pasajero.
2. **La cola no agrega nada.** Descontando la carga media, el máximo por arista queda en ρ=−0.005
   (p=0.98); al revés, la carga media descontando el máximo queda en +0.654 (p=5e-06). La respuesta a
   "¿mandan unas pocas aristas supercargadas?" es **no**: manda la **media**, es decir el **tamaño del
   soporte**.

### 6.4 — Contra el grano, y el piso de ruido real

| | Δ | en partículas | contra el grano (1 part.) |
|---|---|---|---|
| C4 ancla (`solap`−`disp`) | +0.01419 | +28.4 | 28× el grano |
| C3 solapamiento (`abanico`−`malla`) | −0.00500 | −10.0 | 10× el grano |
| **C5 gemelos (`cola`−`disp`)** | +0.00231 | **+4.6 ± 2.4** | — |
| **C1 Gini (`abanico`−`disp`)** | +0.00000 | **+0.0 ± 3.1** | por debajo del grano |

**Advertencia de resolución que el propio experimento produjo:** `cola` y `disp` salieron casi idénticos
estructuralmente (ΔA=−0.012, Δgini=+0.026, ΔC=−0.020) y aun así su masa difiere en **+4.6 ± 2.4
partículas, 6/8 signos**. Es decir: **el piso práctico de un pareado de 8 grafos no es 1 partícula sino
~5**, porque dos grafos "casi iguales" no son el mismo grafo — cada aceptación de swap mueve aristas
distintas y el layout responde. C1 (0.0 ± 3.1) está por debajo de ese piso: es un cero **dentro de la
resolución de esta batería**, no un cero absoluto. Un efecto de Gini de 8.8 partículas, en cambio, sí
habría sido visible.

### 6.5 — Observables secundarios: el mismo patrón de siempre

| observable | `abanico` | `disp` | `cola` | `malla` | `solap` |
|---|---|---|---|---|---|
| nº de sumideros | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 |
| t del primer sumidero | 0.0314 | 0.0341 | 0.0331 | 0.0344 | **0.0307** |
| κ_V agregado | 0.851 | 0.857 | 0.861 | 0.932 | **1.001** |
| pendiente corregida | 0.769 | 0.752 | 0.790 | 0.839 | **0.906** |

**No se forman más grumos: los mismos 8 en los cinco brazos.** Lo que cambia es cuánto come cada uno.

---

## 7. Respuesta literal a la pregunta central de la tarea

> *¿Qué medida sigue la masa cuando las demás quedan fijas?*

**El tamaño del soporte en aristas** (`frac_aristas_en_triangulo`), equivalentemente **la carga media por
arista** (`tri_por_arista_media`) — que, por la identidad `A·D·m=3T`, **son la misma variable**.

- Cuando se mueve **sólo el Gini por nodo** con las aristas quietas: **0.0 ± 3.1 partículas** (C1).
- Cuando se mueve el **solapamiento/soporte**: −10.0 partículas (C3) y +28.4 (C4).
- Cuando se mueve **sólo la cola** (máximo por arista) descontando la media: **ρ = −0.005, p = 0.98**.

No empataron: **una gana**. Y la que gana es, además, la más barata de medir (contar aristas con
triángulo y dividir), lo que resuelve también la segunda mitad de la pregunta.

**Pares que resultaron inseparables por construcción:**

| par | por qué |
|---|---|
| `tri_por_arista_media` ↔ `frac_aristas_en_triangulo` | **identidad algebraica** `A·D·m = 3T` (residuo 4.6e-13). Imposible, no difícil |
| `frac_aristas_multi_tri` ↔ `tri_por_arista_media` | cota dura **A ≥ 1+C**: subir el solapamiento sube la media obligatoriamente |
| `frac_aristas_multi_tri` ↔ `gini_tri_nodo` | apilar en aristas apila en los nodos de esas aristas: C y B suben juntos (C3: ΔC=0.562 arrastra ΔB=0.116) |
| `tri_por_arista_max` ↔ techo de grado | `E ≤ kcap−1 = 7`. El "máximo 30" pedido no existe en este universo |

---

## 8. Lo que este experimento NO puede decidir

- **n=8 grafos.** El binomial mínimo alcanzable es 8/8 → p=0.008; con 7/8 el ancla llega a p=0.023. Los
  contrastes chicos (C2, C3) quedan en p≈0.06-0.20: **son direcciones, no pruebas**.
- **El cero de C1 es un cero a ~±3 partículas**, con un piso de ruido empírico de ~5 (§6.4). No dice
  "el Gini no hace nada"; dice "el Gini no hace ni 9 partículas cuando las aristas están quietas".
- **El contraste de cola quedó chico** porque `T*` lo fija el brazo `malla` (tope exacto de 2). Con un
  diseño que suelte esa restricción —o con grafos de más grado— la cola podría probarse de verdad. Hoy
  sólo hay **un** grafo (T*=1053) donde el contraste de cola salió limpio.
- **Régimen de triángulos fabricado, no natural.** Aviso de F8-00: los grafos naturales de esta línea
  tienen mediana 15 triángulos y 165 de 254 no tienen **ni una** arista con dos. Acá se trabajó con
  T*=34-1053 (mediana 155). **Todo lo de arriba vale en el régimen que F7-02/F7-03 fabricaron, no en el
  que el sistema produce solo.** Es la misma limitación de F7-03 y sigue abierta.
- **8 grafos, todos Clase III del mismo linaje A2-B0-C2.** No se sabe si vale fuera de ahí.
- No se probó **por qué** el soporte apretado junta más masa. Que κ_V suba y el primer sumidero llegue
  antes sigue sugiriendo que el layout hace nudos más compactos donde el soporte se concentra — pero eso
  es F8-02/F8-03 (manipular el pico local de densidad a propósito), no esto.

---

## 9. Costos

Piloto (1 grafo, 5 brazos, sin condiciones iniciales): 440 s — sirvió para detectar la fuga del tope de
`malla` y para confirmar que el desacople de C1 era fabricable antes de comprometer la batería.
Batería: 8 turnos en paralelo, ~55 min de reloj (la máquina estaba a carga media 150-300 por otros
trabajos; cada condición inicial tardó 130-385 s contra los ~71 s típicos). Phantom: ~20 s por corrida,
40 corridas en 5 turnos paralelos. Batería en disco: **1.1 GB**.

---

## Archivos

**Nuevos (esta tarea):** `cs090_fase8_f801_desacople.py`, `cs090_fase8_f801_correr.py`,
`cs090_fase8_f801_analizar.py`, `cs090_fase8_f801_guardar_canonico.py`,
`cs090_fase8_f801_estructura_shard{0..7}.csv`, `cs090_fase8_f801_estructura_piloto.csv`,
`cs090_fase8_f801_phantom_crudo.csv`, `cs090_fase8_f801_matriz_desacople.csv`,
`cs090_fase8_f801_por_grafo.csv`, `cs090_fase8_f801_estadistica.csv`,
`cs090_fase8_f801_correlaciones.csv`, `cs090_fase8_f801_desacople.png`,
`cs090_fase8_f801_shard{0..7}.log`, `cs090_fase8_f801_piloto.log`,
`cs090_fase8_f801_phantom_{abanico,disp,cola,malla,solap}.log`, y la batería
`/Users/alexis/phantom_cs073/bateria_fase8_f801_desacople/` (40 carpetas con su grafo guardado en los
dos formatos).

**Sólo importados, nunca modificados:** `cs090_fase7_f703_organizacion.py`,
`cs090_fase7_f702_escalera.py`, `cs090_fase7_f702_analizar.py`, `cs090_fase6_o3b_rewiring.py`,
`cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_analizar.py`, `cs090_fase5_generador/motor/
clasificador.py`, `cs090_diam_corregido.py`, `cs080_renormalizacion.py`, `cs090_fase8_f800_grafos.py`.

> Sin cierre, sin veredicto, sin commits.
