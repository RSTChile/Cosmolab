# F8-00 — Los grafos vuelven: 254 de 254 regenerados, medidos y guardados

**12 de agosto de 2026** · Tarea de infraestructura de la Ola 1 de Fase VIII. No corre Phantom.
Insumo directo de **F8-01** (desacoplar las medidas de apiñamiento), que corre en paralelo.

---

## 0. En una frase

Las 254 corridas de Phantom del dataset unificado tenían el apiñamiento medido en **24**; ahora lo
tienen las **254**, la reconstrucción coincide **exacto** con esas 24 previas, y los grafos quedaron
**escritos en disco con sello verificable** (3,6 MB en total) para que ningún experimento de Fase VIII
vuelva a perderlos.

**La analogía.** Cada corrida era una construcción de la que se anotó el resultado final y se tiraron
los planos. La receta para volver a levantarla, en cambio, había quedado escrita paso a paso. Se
levantaron las 254 otra vez, se las midió a todas con la misma cinta métrica, se archivaron los planos
y —para no creerle a la reconstrucción por fe— se comprobó contra las 24 que sí se habían medido en su
momento: dan lo mismo hasta la última cifra que la máquina puede distinguir.

---

## 1. Qué se pudo medir y qué no

| | filas | resultado |
|---|---|---|
| **Medidas con éxito** | **254 / 254** | ninguna quedó sin medir |
| Fallas de reconstrucción | 0 | — |

Se temía que las filas ajenas al linaje A2-B0-C2 no se pudieran reconstruir. **Sí se pudieron**, con su
propia receta:

| familia | filas | receta usada (copiada del runner original, sin tocarlo) |
|---|---|---|
| F5B_40pares, O3A_N4000, OUT_pendNEG, O3D(regla), O3B(orig) | 157 | `reconstruir_regla_a2b0c2(seed, N, n_sweeps=14)` |
| O3B_rewiring (gemelo) | 12 | el original + `barajar_aristas(..., seed=seed*9100+7, ×10)` |
| O3C_factorial (c1…c4) | 47 | `O3C._grafo_final_de_condicion(cond, p, N)` |
| O3E_memoria (mem/nomem) | 30 | `O3E.reconstruir(seed, ventana_memoria = None / 1)` |
| **O3D controles Erdős-Rényi** | 6 | `generar_grafo_erdos_renyi(N, aristas, seed)`, con `(aristas, seed)` leídos del propio `rule_id` |
| **O3D controles ER históricos** | 2 | ídem, con `seed_random = 1000·N + k`; **verificado** contra la cabecera del `cosmogenesis_ic.txt` guardado (declara `seed_random=2000001/2000002`, `n_aristas=4945`) |

Los 2 controles históricos venían con la columna `seed` **vacía** en el dataset: la semilla se dedujo
de la regla de generación (`grafo_random_masa_fija_generar.py`, línea 82) y **se comprobó contra el
archivo de condiciones iniciales de esa carpeta** antes de aceptarla. Si no hubiera coincidido, el
script aborta esa fila y la deja sin medir — no coincidió por casualidad, coincidió exacto.

Costo real: **12,1 s por grafo** de promedio (reconstruir + medir todo + las dos pendientes + guardar y
releer), 254 grafos en **219 s** con 14 procesos, sobre una máquina con carga muy alta.

---

## 2. Verificación de fidelidad — las 24 filas que ya tenían clustering

**Coinciden exacto.** Las 24 filas de O3-B (12 originales + 12 gemelos reconfigurados):

| medida | máxima diferencia sobre 24 filas |
|---|---|
| clustering local medio | **1,0 × 10⁻¹⁶** |
| transitividad global | **9,8 × 10⁻¹⁷** |
| nº de triángulos (conteo entero) | **0** |

Las diferencias de 10⁻¹⁶ son el último bit de un número de punto flotante: la máquina no puede
representar "más igual" que eso. El conteo de triángulos, que es un entero, coincide sin margen.
Detalle fila a fila en `cs090_fase8_f800_verificacion.csv`.

### 2b. Un hallazgo de método que salió de esta verificación: **hay dos pendientes, no una**

`cs080_renormalizacion.cajas_bfs` recorre `for v in adj[u]`. El **orden de iteración** de un conjunto
de Python depende de cómo se llenó, así que dos grafos con exactamente las mismas aristas pero armados
en distinto orden pueden caer en cajas distintas y mover la pendiente. Está documentado desde O3-B.
Este trabajo lo midió sobre las 254 y el resultado es nítido:

| origen de la pendiente histórica | filas | ¿coincide con la forma **nativa** (objeto tal cual sale del motor)? | ¿con la forma **canónica** (vecinos en orden creciente, la que se recupera del disco)? |
|---|---|---|---|
| F5B_40pares | 73 | **idéntica en las 73** | no |
| O3D_kcap | 32 | **idéntica en las 32** | no |
| O3C_factorial | 47 | idéntica en 39; máx. Δ 0,034 | no |
| O3E_memoria | 30 | idéntica en 21; máx. Δ 0,0007 | no |
| **O3B_rewiring** | 24 | **no** (máx. Δ 0,127) | **idéntica en las 24** (Δ = 2,2 × 10⁻¹⁶) |

Es decir: **O3-B midió su pendiente sobre la forma canónica y el resto de la línea sobre la nativa.**
No es un error de nadie —las dos son mediciones legítimas del mismo grafo— pero **no se pueden mezclar
en una misma regresión**: la diferencia llega a 0,127, del mismo orden que los residuales que persigue
Fase VIII. Por eso el CSV enriquecido publica **las dos columnas**, `f800_pendiente_nativa` y
`f800_pendiente_canon`, y quien las use tiene que elegir una sola y decir cuál.

Analogía: es como medir el largo de una habitación empezando desde la pared izquierda o desde la
derecha. La habitación es la misma; el número que sale, si el metro se apoya en distintos rincones, no
siempre. Lo que no se puede es promediar mediciones tomadas de las dos maneras y creer que la
diferencia es física.

---

## 3. Verificación 2 — el nº de aristas reconstruido contra el archivado (las 254)

**246 de 254 coinciden exacto.** Ocho no, y vale la pena mirar cuáles:

| experimento | filas | diferencia | sobre un total de |
|---|---|---|---|
| O3A_N4000 (3 de 26) | `batch3-r5`, `batch3-r76`, `batch4-r94` | +6, −1, +1 aristas | ~6.700–7.900 (0,01 %–0,08 %) |
| O3C_factorial (5 de 47) | `mec-r2/r6/r7/r11`, casi todos en la condición **azar** | −8, −1, −21, +21, −1 | ~3.700–5.300 (0,02 %–0,5 %) |

Dos cosas medidas, ninguna interpretada:

1. **La reconstrucción de hoy es determinista.** Se volvió a construir cada uno de esos grafos en
   procesos independientes: el sello sha256 salió **idéntico** las dos veces, y también idéntico al
   guardado en la corrida masiva. La variación no está en este script.
2. **Las 5 celdas de O3-C que discrepan son exactamente las 5 que el propio runner de O3-C ya había
   marcado** como no reproducibles entre sesiones (`reproduce_archivada = False` con
   `dif_vs_archivada > 10⁻³`: r6/c2 = 1,0e−3; r6/c4 = 1,5e−3; r7/c2 = 6,4e−3; r2/c2 = 2,0e−2;
   r11/c2 = 3,4e−2). La correspondencia es uno a uno. O3-C lo dejó anotado en su §3 como diagnóstico
   abierto; este trabajo lo confirma desde otro ángulo y agrega que también mueve el nº de aristas.

Queda **abierto** por qué esas celdas —y 3 de las 26 corridas a N=4000— no reproducen entre sesiones,
mientras las 131 filas del linaje base a N=2000 (F5B, O3D-regla, O3B-orig, OUT_pendNEG) reproducen el
nº de aristas **bit a bit, todas**. No se propone explicación acá.

---

## 4. Un dato colateral que conviene mirar: los 11 "outliers de pendiente negativa"

Las 11 filas de `OUT_pendNEG` traen en el dataset una pendiente **negativa** (−0,49 a −1,20) y un
**diámetro de 1 o 2**. Medidas con el diámetro corregido (`cs090_diam_corregido`, la medición oficial
vigente que la Fase VIII exige):

| | histórico | remedido F8-00 |
|---|---|---|
| diámetro | 1–2 | **12–28** (mediana 19) |
| pendiente | −0,49 … −1,20 | **+0,59 … +1,44** |

El nº de aristas coincide **exacto en las 11**, así que no es otro grafo: es la misma construcción
medida con otra cinta. Es la firma conocida del diámetro viejo colapsando a 1. Se reporta como número;
qué hacer con esas 11 filas (dejarlas, remedirlas aguas abajo, o tratarlas aparte) es decisión del
director, no de esta tarea.

---

## 5. Los grafos, guardados

`cs090_fase8_f800_grafos.py` — módulo nuevo, tres funciones y una autoprueba.

- **Formato** `.grafo.gz`: texto plano comprimido, una arista por línea, con cabecera
  `# cosmogenesis_grafo v1 N=… E=… sha256=… rule_id=… seed=… brazo=… receta=…`. Se lee con `zcat`.
- **Compacto**: 2,9 bytes por arista comprimidos. En la práctica **13,5 KB** por grafo de N=2000 y
  **26 KB** por grafo de N=4000. **Total de las 254: 3,66 MB.**
- **Verificable**: el `sha256` se calcula sobre la lista canónica de aristas, así que **no depende del
  orden en que el grafo se armó en memoria** (verificado en la autoprueba: se rearma al revés y da el
  mismo sello). `cargar_grafo` lo recalcula al leer y **falla si no coincide** (también verificado:
  se corrompe una línea a propósito y la detecta).
- **Ida y vuelta**: los 254 grafos se guardaron, se releyeron del disco y se comparó **arista por
  arista** con el original: **254 de 254 idénticos**.

Están en `grafos_f800/<experimento>/<rule_id>__s<seed>__<brazo>__N<N>.grafo.gz`, y el sello de cada uno
viaja también en la columna `f800_sha256` del CSV enriquecido.

### Cómo lo llama un runner futuro (dos líneas)

```python
import cs090_fase8_f800_grafos as G8

# donde el runner ya tiene m = MOT.medir(...) y su carpeta de salida:
G8.guardar_grafo(m["adj_final"], carpeta / "grafo_final.grafo.gz", N=N,
                 meta=dict(rule_id=rule_id, seed=seed, brazo=brazo, n_sweeps=14))

# y el sello dentro del meta_regla.json que ya escribía:
meta["grafo_sha256"] = G8.hash_grafo(m["adj_final"], N)      # ...o, si el meta ya está en disco:
G8.anotar_hash_en_meta(carpeta / "meta_regla.json", m["adj_final"], N,
                       archivo_grafo="grafo_final.grafo.gz")

# meses después, sin volver a correr el motor ni Phantom:
adj, N, meta = G8.cargar_grafo(carpeta / "grafo_final.grafo.gz")   # verifica el sello solo
```

`anotar_hash_en_meta` **no se corrió sobre las corridas históricas**: los `meta_regla.json` ya escritos
no se tocaron. Queda disponible para las corridas nuevas de Fase VIII.

### Para qué sirve el sello, más allá de detectar archivos rotos

Al comparar los 254 sellos aparece algo que hasta ahora no se podía ver: **las 254 filas contienen
sólo 239 grafos distintos**. 15 grafos entran dos veces:

- 12 pares **F5B ↔ O3B(orig)** — el brazo "original" de O3-B es, literalmente, la misma construcción
  que ya había pasado por Phantom en Fase V-B;
- 3 pares que involucran el brazo **mem** de O3-E (que por diseño reproduce la dinámica original), uno
  de ellos contra una fila de `OUT_pendNEG`.

Para F8-01 esto importa: **al contar grados de libertad, 254 filas no son 254 grafos independientes.**
La columna `f800_sha256` permite agrupar o excluir según convenga, por contenido y no por nombre —
que es justamente lo que el bug de colisión de nombres de Fase V-B enseñó a no hacer al revés.

---

## 6. Lo que se midió en las 254

Todas las variables pedidas, con el prefijo `f800_` en el CSV enriquecido:

- **triángulos por arista** (sobre las aristas que sostienen al menos uno): `tri_ar_media_sop`,
  `tri_ar_mediana_sop`, `tri_ar_max`, `tri_ar_p99_sop`; y `tri_ar_media_todas` sobre **todas** las
  aristas.
- **solapamiento de aristas entre triángulos**: `frac_aristas_multi_tri` (de las aristas con triángulo,
  cuántas están en ≥2).
- **Gini de concentración de triángulos por nodo**: `gini_tri_nodo`; más `tri_por_nodo_max`,
  `frac_nodos_en_triangulo`.
- **fracción de aristas que sostienen al menos un triángulo**: `frac_aristas_en_triangulo`.
- **cúmulos de triángulos**: `n_comp_tri` (unidos por nodo compartido, definición de F7-03),
  `n_comp_tri_arista` (unidos por **arista** compartida, el eje del solapamiento),
  `frac_mayor_comp_tri`, `tam_medio_comp_tri`, `modularidad_tri`; **distancia media entre triángulos**
  `dist_media_tri` y su vara `dist_media_azar`.
- **como dato, NO como variable explicativa** (F7-03: falla en el signo): `clustering`,
  `transitividad`, `n_triangulos`.
- **estructurales**: `n_aristas`, `grado_medio`, `giant`, `n_componentes`, `diam` (corregido),
  `asortatividad`, `pendiente_nativa`, `pendiente_canon`.

### Advertencia de escala que F8-01 necesita leer antes de elegir perilla

Estos grafos tienen **muy pocos triángulos**: mediana **15**, rango **0–59**, sobre ~3.000 aristas y
2.000 nodos. Consecuencias medidas:

- `tri_ar_max` vale 0 en 2 filas, **1 en 163**, 2 en 86 y 3 en sólo 3. Es decir: **en 165 de 254 grafos
  no existe ni una sola arista que sostenga dos triángulos.**
- `frac_aristas_multi_tri` tiene **mediana 0** y máximo 0,05.
- `tri_ar_mediana_sop` vale **1,000 en las 252 filas con triángulos** — es una constante, no aporta
  información y por eso sale NaN en toda la matriz de correlaciones.
- `gini_tri_nodo` vive entre 0,920 y 0,999: los triángulos siempre están concentradísimos, porque casi
  ningún nodo tiene alguno.

O sea: el "apiñamiento" que se puede manipular en esta familia es sobre todo **cuánto grafo llega a
tener triángulos**, y muy poco **cuántos triángulos se apilan sobre la misma arista** — ese segundo eje
existe, pero con tres niveles y cero en dos tercios de las filas.

---

## 7. La matriz de correlaciones (insumo directo de F8-01)

Sobre las 254 filas. Archivo completo par a par: `cs090_fase8_f800_correlaciones.csv`
(columnas `var_a, var_b, n, spearman, pearson, abs_spearman`, ordenado por |Spearman|).

Spearman entre las medidas nucleares de apiñamiento:

|  | tri_ar_media_sop | tri_ar_max | tri_ar_p99 | tri_ar_media_todas | frac_ar_en_tri | frac_ar_multi | gini_tri_nodo | tri_nodo_max | n_triangulos | transitiv. | clustering |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **tri_ar_media_sop** | 1 | 0,972 | 0,933 | 0,424 | 0,414 | **1,000** | −0,453 | 0,685 | 0,485 | 0,340 | 0,315 |
| **tri_ar_max** | 0,972 | 1 | 0,854 | 0,518 | 0,510 | 0,970 | −0,542 | 0,710 | 0,580 | 0,431 | 0,413 |
| **tri_ar_p99_sop** | 0,933 | 0,854 | 1 | 0,336 | 0,327 | 0,932 | −0,353 | 0,615 | 0,382 | 0,265 | 0,234 |
| **tri_ar_media_todas** | 0,424 | 0,518 | 0,336 | 1 | **1,000** | 0,422 | −0,963 | 0,638 | 0,936 | 0,964 | 0,950 |
| **frac_ar_en_triangulo** | 0,414 | 0,510 | 0,327 | **1,000** | 1 | 0,412 | −0,963 | 0,633 | 0,935 | 0,965 | 0,952 |
| **frac_ar_multi_tri** | **1,000** | 0,970 | 0,932 | 0,422 | 0,412 | 1 | −0,451 | 0,683 | 0,482 | 0,338 | 0,313 |
| **gini_tri_nodo** | −0,453 | −0,542 | −0,353 | −0,963 | −0,963 | −0,451 | 1 | −0,637 | −0,957 | −0,866 | −0,867 |
| **tri_por_nodo_max** | 0,685 | 0,710 | 0,615 | 0,638 | 0,633 | 0,683 | −0,637 | 1 | 0,691 | 0,557 | 0,536 |
| **n_triangulos** | 0,485 | 0,580 | 0,382 | 0,936 | 0,935 | 0,482 | −0,957 | 0,691 | 1 | 0,853 | 0,862 |
| **transitividad** | 0,340 | 0,431 | 0,265 | 0,964 | 0,965 | 0,338 | −0,866 | 0,557 | 0,853 | 1 | 0,970 |
| **clustering** | 0,315 | 0,413 | 0,234 | 0,950 | 0,952 | 0,313 | −0,867 | 0,536 | 0,862 | 0,970 | 1 |

### Lo que dice la matriz, en simple

Las once medidas **no son once perillas: son dos**, y una bisagra.

- **Eje 1 — "cuánto del grafo tiene triángulos"** (extensión): `frac_aristas_en_triangulo`,
  `tri_ar_media_todas`, `n_triangulos`, `transitividad`, `clustering`, `frac_nodos_en_triangulo`,
  `modularidad_tri`, `n_comp_tri`, `n_comp_tri_arista`, y `gini_tri_nodo` con el signo dado vuelta.
  **Entre ellas: |ρ| de 0,852 a 1,000 (mediana 0,953).**
- **Eje 2 — "cuántos triángulos se apilan sobre la misma arista"** (solapamiento):
  `tri_ar_media_sop`, `tri_ar_max`, `tri_ar_p99_sop`, `frac_aristas_multi_tri`.
  **Entre ellas: |ρ| de 0,854 a 1,000 (mediana 0,952).**
- **Entre los dos ejes: |ρ| de 0,234 a 0,580 (mediana 0,423).** Es el único desacople real que existe
  en estos datos.
- **La bisagra**: `tri_por_nodo_max` correlaciona ~0,64–0,71 con los dos ejes; `tam_medio_comp_tri`
  (0,726 con el eje 2) y `dist_media_tri` (0,669 con el eje 1) están cerca pero de un lado cada uno.

**Redundancias exactas que conviene no volver a estimar por separado** (ρ = 1,000):

| par | ρ Spearman | ρ Pearson | por qué |
|---|---|---|---|
| `tri_ar_media_sop` ~ `frac_aristas_multi_tri` | +1,000 | +0,993 | con máximo ≤ 2 triángulos por arista, la media sobre el soporte **es** 1 + la fracción multi. Es una identidad algebraica, no un hallazgo |
| `tri_ar_media_todas` ~ `frac_aristas_en_triangulo` | +1,000 | +1,000 | ídem: si casi toda arista con triángulo tiene exactamente uno, la media sobre todas las aristas **es** la fracción de aristas con triángulo |
| `frac_aristas_en_triangulo` ~ `modularidad_tri` | +1,000 | +1,000 | la modularidad de la partición por cúmulos, en un grafo tan pobre en triángulos, no agrega nada nuevo |
| `n_triangulos` ~ `n_comp_tri_arista` | +0,999 | +0,998 | casi ningún triángulo comparte arista con otro: hay tantos cúmulos como triángulos |
| `gini_tri_nodo` ~ `frac_nodos_en_triangulo` | −0,999 | −0,999 | el Gini acá sólo está midiendo cuántos nodos quedan en cero |
| `transitividad` ~ `clustering` | +0,970 | +0,977 | el par que F7-05 ya había reportado colineal |

**Las estructurales están razonablemente separadas del apiñamiento**: `n_aristas` (máx. |ρ| 0,554),
`grado_medio` (0,534), `asortatividad` (0,413), `diam` (0,517), `pendiente_nativa` (0,556),
`giant` (0,605) — todas contra `transitividad`, su vecina más cercana. Ninguna medida de apiñamiento es
un disfraz de la densidad.

**Consecuencia práctica para F8-01**: pedirle a los datos que separen `tri_ar_media_sop` de
`frac_aristas_multi_tri` (ρ = 1,000 por construcción) o `tri_ar_media_todas` de
`frac_aristas_en_triangulo` es imposible en esta muestra — hay que **fabricar** grafos que las
disocien. Lo que sí está disponible ya, sin fabricar nada, es el contraste **eje 1 vs eje 2**
(ρ ≈ 0,42), con la advertencia de la §6 sobre lo escaso que es el eje 2 en esta familia.

---

## 8. Archivos

| archivo | qué es |
|---|---|
| `cs090_fase8_f800_grafos.py` | **nuevo** · `guardar_grafo` / `cargar_grafo` / `hash_grafo` / `anotar_hash_en_meta` + autoprueba (`python3.9 cs090_fase8_f800_grafos.py`) |
| `cs090_fase8_f800_medir_254.py` | **nuevo** · runner: reconstruye, mide, guarda, verifica y arma el CSV enriquecido y la matriz |
| `cs090_fase8_f800_dataset_enriquecido.csv` | **254 filas × 89 columnas** — las 54 originales intactas + 35 columnas `f800_*` |
| `cs090_fase8_f800_correlaciones.csv` | matriz par a par (Spearman y Pearson) ordenada por |ρ| |
| `cs090_fase8_f800_verificacion.csv` | las 24 filas de O3-B, histórico vs. reconstruido, columna por columna |
| `cs090_fase8_f800_cache.jsonl` | una línea por corrida: métricas, sello, tiempos, receta usada (permite reanudar) |
| `cs090_fase8_f800_medir_254.log` | registro completo de la corrida y de las 4 verificaciones |
| `grafos_f800/<exp>/*.grafo.gz` | **254 grafos**, 3,66 MB en total |

`cs090_fase7_f705_dataset_unificado.csv` **no fue modificado**. Ningún script existente fue tocado:
`cs090_fase5b_phantom_adaptador`, `cs090_fase5_motor`, `cs090_fase5_generador`,
`cs090_fase6_o3b_rewiring`, `cs090_fase6_o3c_factorial_mecanistico`, `cs090_fase6_o3e_memoria`,
`cs090_fase7_f702_escalera`, `cs090_fase7_f703_organizacion`, `cs090_diam_corregido` y
`grafo_random_layout_generar_ic` sólo se **importan**. No se corrió Phantom. No hay commits.

---

## 9. Qué queda abierto (no se cierra nada acá)

1. Por qué **8 de 254** no reproducen el nº de aristas archivado —5 de ellas exactamente las que O3-C
   ya había marcado, 3 a N=4000— mientras las 154 restantes del linaje base (todas a N=2000)
   reproducen bit a bit.
2. Qué hacer con las **11 filas de `OUT_pendNEG`**, cuya pendiente negativa viene de un diámetro
   histórico de 1–2 que la medición oficial vigente no reproduce.
3. Si el análisis de Fase VIII usará la pendiente **nativa** o la **canónica** — hay que elegir una y
   declararla, porque O3-B está medido con la otra.
4. Que **239 grafos distintos** sostienen 254 filas: cómo contar los grados de libertad en F8-01.
