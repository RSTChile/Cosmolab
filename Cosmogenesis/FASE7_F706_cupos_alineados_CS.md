# F7-06 — Con la MISMA distribución de capacidades, ¿importa **a qué nodo** le toca cada cupo?

**Fecha:** 12-ago-2026 · Ejecuta: CC (Claude) · Tarea **F7-06 de Fase VII**, propuesta por **GPT-5.6 Sol**
y señalada también como pendiente explícito en `FASE6_O2C_kcap_capacidad_finita_CS.md` §10 (último
ítem) y §7.

**Antecedente directo.** O2-C repartió capacidad relacional desigual entre nodos con la misma media y
encontró dosis-respuesta con el CV del reparto (Spearman +0,745), pero midió que **el 86-88% de eso pasa
por un canal trivial**: los repartos desiguales desperdician cupo y la red termina con menos aristas.
Y dejó una anomalía sin explicar: **`HET-grado` es la única distribución que va en dirección contraria**,
y es también la única cuyo cupo está **correlacionado con el grado inicial del nodo**. De ahí la
hipótesis que O2-C marcó como sugerida-pero-no-testeada: lo que importaría no es la dispersión del cupo,
sino **si el cupo está alineado con la estructura preexistente**.

Es el análogo directo del hallazgo de `FASE6_O1D_factorial_trio_CS.md`: allí, con alineación el trío real
**ayuda** y sin alineación **estorba** (interacción z=−6,36).

**No se modificó ningún script existente** (`cs090_fase6_o2c_capacidad_finita.py`,
`cs090_fase5_mecanismo_aislado.py`, `cs090_fase5_motor.py` y el resto se **importan**, no se tocan).
**No se hicieron commits.** **No se declara cierre ni veredicto** — se reportan números; la lectura final
es de Alexis.

---

## 0. En simple, con analogía

Un pueblo donde cada persona tiene un **tope de amigos que puede sostener**. Hasta acá el proyecto había
probado *repartir los topes de distinta manera* (todos 6; unos 2 y otros 10; unos pocos con cientos). Lo
que **nunca** se había probado es lo siguiente: **repartir exactamente los mismos topes, pero cambiando
quién recibe cuál**.

Tres pueblos, con la misma lista de topes hasta el último número:

- **Alineado** — al que ya tenía muchos conocidos le toca el tope grande. (Es el `HET-grado` de O2-C.)
- **Permutado** — la misma lista de topes, repartida al azar entre la gente.
- **Anti-alineado** — al revés: el tope grande al que casi no conocía a nadie.

**El resultado, en tres frases.**

1. **Sí importa muchísimo a quién le toca cada cupo — pero por un camino bien aburrido.** Cuando el tope
   grande le toca al que ya tenía muchos conocidos, casi no hay que cortar ninguna amistad, y el pueblo
   termina con **43% más vínculos** que el permutado y **82% más** que el anti-alineado. En **12 de 12**
   pueblos, sin una sola excepción.
2. **Y esa diferencia de vínculos se lleva puesto todo lo demás.** La masa que el simulador junta en
   sumideros cae de 0,139 (anti) a 0,114 (permutado) a 0,083 (alineado): un −40%. Pero la masa y el
   número de vínculos van pegados con ρ=−0,96 en las 60 corridas: **eso no es "alineación", es densidad**.
3. **Cuando se igualan los vínculos exactamente, lo que queda es chico y no concluyente.** Con el mismo
   número exacto de aristas, el origen alineado deja **+6,6 partículas** de masa sobre el permutado
   (+2,3%): eso está **por encima del grano del instrumento** (1 partícula = 0,0005), pero el test
   pareado da p=0,065 (signos) / p=0,098 (Wilcoxon), y descontando la densidad por regresión en vez de
   por dilución **desaparece** (p=0,39). Además queda una asimetría que este diseño no puede separar
   (§7).

Analogía del punto 3: para comparar dos pueblos de distinto tamaño hay que sacarle gente a uno. Al
alineado hubo que sacarle 1 943 amistades y al permutado sólo 654. Después de eso el alineado sigue
juntando un poquito más de masa… pero **"sacar más amistades al azar" también sube la masa**, y en esta
corrida las dos cosas van siempre juntas. No se pueden separar con estos brazos.

---

## 1. El diseño: mismo multiconjunto, distinta asignación

Universo: los mismos 12 grafos base de F7-04 (los que ya pasaron por Phantom en Fase V-B, así que su
`meta_regla.json` existe en disco para la verificación cruzada), **3 por cada uno de los 4 lotes de
`seed_base`** (271828 / 371828 / 471828 / 571828), tomando la pendiente corregida mínima, mediana y
máxima de cada lote. N=2000, masa total fija 18800, `seed_layout`=12345, protocolo Phantom estándar de
toda la jerarquía CS073.

Todos los brazos parten del **mismo grafo Erdős-Rényi inicial**, consumen el **mismo flujo de números
aleatorios** en la dinámica (14 sweeps, recableo co-emergente cada 3 pasos, enforcement de cupo cada 4,
poda final por costo P70) y tienen el **mismo cupo medio** (el `p["kcap"]` de la regla, 4-7). Lo único
que cambia es **qué nodo recibe qué cupo**.

| brazo | qué es | ρ(cupo, grado inicial) | ¿va a Phantom? |
|---|---|---:|---|
| `unif` | cupo constante = `p["kcap"]`. **Control de fidelidad** | — (constante) | no (ya corrió en Fase V-B) |
| `alineado` | `MA._cupo_variable` **tal cual** = el `HET-grado` de O2-C | **+0,996** | sí |
| `permutado` | el **mismo multiconjunto**, barajado entre nodos | **−0,006** | sí |
| `anti` | el mismo multiconjunto, invertido (cupo alto ↔ grado bajo) | **−0,979** | sí |
| `alin_dil` | `alineado` diluido **al azar** hasta el nº de aristas de `anti` | +0,996 (origen) | sí |
| `perm_dil` | `permutado` diluido **al azar** hasta el nº de aristas de `anti` | −0,006 (origen) | sí |

Los dos últimos brazos no estaban en el pedido: se agregaron **después de medir el confound y antes de
correr Phantom** (§3). Los empates de grado en `anti` (masivos: el grado en un ER es un entero chico) se
rompen **al azar**, no por índice de nodo, que sesgaría hacia los nodos de índice bajo.

Cupos efectivos, para dar la escala: media = `kcap` de la regla (4-7), **CV entre 0,369 y 0,502**, mínimo
1 en los 12 grafos, máximo entre 10 y 20. Ese multiconjunto es **idéntico** en `alineado`, `permutado` y
`anti` — es el punto entero del experimento.

---

## 2. Verificaciones (corridas, no asumidas)

```
multiconjunto de cupos idéntico alineado/permutado/anti : 12/12 grafos
brazo `unif` reproduce MOT.dinamica_B0 arista por arista : 12/12
nº de aristas de `unif` == meta_regla.json de Fase V-B   : 12/12
densidad EXACTAMENTE igual en alin_dil/perm_dil/anti     : 12/12
ρ(cupo, grado inicial): alineado=+0,996  permutado=−0,006  anti=−0,979
```

- **El multiconjunto**: `np.array_equal(np.sort(cupo_alineado), np.sort(cupo_permutado))` y lo mismo
  contra `anti`, grafo por grafo. Misma distribución, misma media, mismo CV, mismo mínimo, mismo máximo.
  **Lo único que cambia es la asignación.**
- **La fidelidad del pipeline**: la dinámica con cupo-por-vector, alimentada con un vector constante,
  produce el grafo final **arista por arista idéntico** al del motor congelado con `_enforce_kcap`
  escalar, y ese nº de aristas coincide con el que Fase V-B escribió en disco para esa misma regla. O
  sea: la maquinaria de cupo variable no cambia nada cuando no debe cambiarlo.
- **La manipulación hizo lo que dice**: la correlación cupo↔grado pasa de +1 a 0 a −1 según el brazo.

---

## 3. El confound, medido ANTES de mandar nada a Phantom

Se corrió primero la parte estructural sola. El resultado obligó a rediseñar:

| brazo | aristas | grado medio | clustering | pendiente corregida | comp. gigante | triángulos |
|---|---:|---:|---:|---:|---:|---:|
| `unif` | 3 512,5 | 3,51 | 0,00511 | 0,7348 | 1 916,7 | 15,8 |
| **`alineado`** | **4 312,5** | 4,31 | 0,00388 | **0,5776** | 1 924,6 | 15,7 |
| **`permutado`** | **3 024,2** | 3,02 | 0,00815 | **0,8546** | 1 879,3 | 16,3 |
| **`anti`** | **2 369,9** | 2,37 | 0,00979 | **1,0352** | 1 781,0 | 12,3 |

Contrastes pareados (mismo grafo base, n=12):

| contraste | Δ aristas | signos | Wilcoxon | Δ pendiente | signos | Wilcoxon |
|---|---:|---|---:|---:|---|---:|
| alineado − permutado | **+1 288,3** | **12/12** | 0,00049 | **−0,277** | **0/12** | 0,00049 |
| alineado − anti | **+1 942,6** | **12/12** | 0,00049 | **−0,458** | **0/12** | 0,00049 |
| permutado − anti | **+654,3** | **12/12** | 0,00049 | **−0,181** | **0/12** | 0,00049 |
| alineado − unif | +800,0 | 12/12 | 0,00049 | −0,157 | 1/12 | 0,00098 |

**Unánime y ordenado: `alineado` > `permutado` > `anti`, en las 12 de 12, con Δ de 244 a 2 083 aristas.**
El mecanismo es transparente: si el cupo grande le toca justo al nodo que ya tenía muchos vecinos, casi
no hay que podar; si le toca a un nodo de grado bajo, el cupo se desperdicia y el vecino saturado pierde
aristas igual. **Éste es, medido de frente, el canal que O2-C había inferido**: `HET-grado` terminaba con
4,93 de grado alcanzado contra 4,03 del uniforme, y acá se ve que la causa es la alineación, no la
dispersión (la dispersión es idéntica en los tres brazos).

Con esa diferencia de densidad, comparar la masa cruda entre brazos mediría densidad, no alineación. Por
eso se agregaron `alin_dil` y `perm_dil`, **diluidos al azar hasta el nº exacto de aristas de `anti`**,
en el espíritu de F7-04 (mismo M, distinta estructura).

Jaccard de aristas entre brazos: `alineado`~`permutado` = 0,583, `alineado`~`anti` = 0,454,
`alineado`~`unif` = 0,695, `alin_dil`~`perm_dil` = 0,324. Los grafos son genuinamente distintos.

---

## 4. Phantom, contraste CRUDO — es densidad

60 corridas (12 grafos × 5 brazos), todas con `exit_run=0`, N=2000, masa fija 18800.

| brazo | aristas | frac. masa en sumideros | κ_V | nº sumideros |
|---|---:|---:|---:|---:|
| `alineado` | 4 312,5 | **0,08313** | 0,545 | 8,00 |
| `permutado` | 3 024,2 | **0,11388** | 0,751 | 8,08 |
| `anti` | 2 369,9 | **0,13850** | 1,153 | 8,08 |
| `alin_dil` | 2 369,9 | 0,14492 | 1,204 | 8,08 |
| `perm_dil` | 2 369,9 | 0,14162 | 1,185 | 8,08 |

```
masa vs aristas sobre las 60 corridas: Spearman ρ = −0,962  (p = 1,9·10⁻³⁴)
```

Contrastes pareados en el endpoint primario:

| contraste | Δ frac. masa | en partículas | % | signos | Wilcoxon |
|---|---:|---:|---:|---|---:|
| alineado − permutado | **−0,03075** | −61,5 | −27,0% | 2/12 | **0,0024** |
| alineado − anti | **−0,05537** | −110,8 | −40,0% | 1/12 | **0,0010** |
| permutado − anti | **−0,02462** | −49,3 | −17,8% | **0/12** | **0,0005** |

Y lo mismo en κ_V: alineado−anti = −0,609 (0/12, p=0,0005), permutado−anti = −0,403 (0/12, p=0,0005).

**El efecto crudo es enorme y perfectamente ordenado — y es el canal de la densidad.** Con ρ=−0,96
entre masa y aristas, cualquier manipulación que cambie el número de aristas produce exactamente esto.

**Descuento por regresión** (control secundario, sobre las 60 corridas):

```
frac_masa = +0,9183 − 0,1002 · log(aristas)      R² = 0,942
```

| residuo | Δ | en partículas | signos | Wilcoxon |
|---|---:|---:|---|---:|
| alineado − permutado | +0,00355 | +7,1 | 8/12 | 0,339 |
| alineado − anti | +0,00349 | +7,0 | 6/12 | 0,569 |
| permutado − anti | −0,00006 | −0,1 | 4/12 | 0,791 |

**Descontada la densidad por regresión, el contraste crudo no sobrevive**: el residuo cambia de signo
(de −61,5 a +7,1 partículas) y ninguno de los tres es distinguible de cero. La densidad no explica el
87% ni el 95% del efecto crudo: lo explica entero y de sobra.

---

## 5. Phantom, contraste LIMPIO — misma densidad exacta

Los tres brazos `alin_dil`, `perm_dil` y `anti` tienen, grafo por grafo, **el mismo número exacto de
aristas** (verificado 12/12). El contraste con el mismo tratamiento en ambos lados es
`alin_dil` vs `perm_dil`.

| contraste | medias | Δ | en partículas | % | signos | p signos | Wilcoxon |
|---|---|---:|---:|---:|---|---:|---:|
| **`alin_dil` − `perm_dil`** | 0,14492 / 0,14162 | **+0,00329** | **+6,6** | +2,3% | **9/11** | 0,065 | 0,098 |
| `alin_dil` − `anti` | 0,14492 / 0,13850 | +0,00642 | +12,8 | +4,6% | **12/12** | 0,0005 | 0,0005 |
| `perm_dil` − `anti` | 0,14162 / 0,13850 | +0,00312 | +6,3 | +2,3% | 8/11 | 0,227 | 0,028 |

Friedman sobre el trío de densidad igualada: **χ²=14,22, p=0,0008** — a densidad fija los tres brazos
**no** son intercambiables. En κ_V, en cambio, `alin_dil` − `perm_dil` = +0,020 (7/12, p=0,77): nada.

Detalle grafo por grafo del Δ limpio (`alin_dil` − `perm_dil`, en fracción de masa):

| regla | lote | kcap | M igualado | fm alin_dil | fm perm_dil | Δ |
|---|---|---:|---:|---:|---:|---:|
| A2-B0-C2-r9 | 271828 | 7 | 2 447 | 0,1405 | 0,1375 | +0,0030 |
| A2-B0-C2-r19 | 271828 | 7 | 2 237 | 0,1485 | 0,1470 | +0,0015 |
| A2-B0-C2-r14 | 271828 | 5 | 2 161 | 0,1535 | 0,1510 | +0,0025 |
| A2-B0-C2-r2 | 371828 | 6 | 2 575 | 0,1415 | 0,1385 | +0,0030 |
| A2-B0-C2-r28 | 371828 | 6 | 3 038 | 0,1360 | 0,1160 | **+0,0200** |
| A2-B0-C2-r20 | 371828 | 6 | 2 396 | 0,1435 | 0,1435 | 0,0000 |
| A2-B0-C2-batch3-r9 | 471828 | 6 | 2 828 | 0,1330 | 0,1245 | +0,0085 |
| A2-B0-C2-batch3-r21 | 471828 | 5 | 2 348 | 0,1445 | 0,1480 | **−0,0035** |
| A2-B0-C2-batch3-r100 | 471828 | 4 | 1 938 | 0,1560 | 0,1530 | +0,0030 |
| A2-B0-C2-batch4-r18 | 571828 | 6 | 2 717 | 0,1280 | 0,1305 | **−0,0025** |
| A2-B0-C2-batch4-r43 | 571828 | 5 | 1 906 | 0,1545 | 0,1525 | +0,0020 |
| A2-B0-C2-batch4-r51 | 571828 | 4 | 1 848 | 0,1595 | 0,1575 | +0,0020 |

Media +0,00329, **mediana +0,00225** (4,5 partículas), positivo en 9/12, un empate exacto, dos negativos.
**Un grafo (r28) aporta él solo +0,0200**; sacándolo, la media baja a +0,00177 (**3,5 partículas**). El
efecto no es un artefacto de ese punto —la mediana sigue en +4,5 partículas— pero tampoco es homogéneo.

---

## 6. Tamaño del efecto contra el grano del instrumento

`FASE7_F704_cortar_bien_vs_azar_CS.md` fijó la vara: la masa está cuantizada en partículas y
**1 partícula = 1/2000 = 0,0005 de fracción de masa**.

| | \|Δ\| medio | en partículas | Δ medio | \|Δ\|>1 partícula | IC95 de la media |
|---|---:|---:|---:|---|---|
| **LIMPIO** alin_dil−perm_dil | 0,00429 | **8,6** | +0,00329 (+6,6 part.) | **11/12 grafos** | [−0,00014, +0,00672] |
| CRUDO alineado−permutado | 0,03158 | 63,2 | −0,03075 (−61,5 part.) | 12/12 grafos | [−0,04094, −0,02056] |

**El efecto limpio NO está por debajo del grano del instrumento** — son 6,6 partículas de media y 4,5 de
mediana, contra un grano de 1, y supera 1 partícula en 11 de 12 grafos. Lo que pasa es otra cosa: es
**pequeño y de signo no unánime** (9/12), con el IC95 rozando el cero por abajo, y con la alternativa
explicativa de §7 sin descartar. No es un nulo por falta de resolución; es un efecto chico mal separado.

---

## 7. La asimetría que este diseño NO puede separar (lo incómodo)

Igualar la densidad tuvo un precio inevitable: para llegar al mismo nº de aristas hubo que quitarle
**mucho más** al brazo alineado que al permutado.

```
aristas quitadas al azar (media):   alin_dil = 1 943    perm_dil = 654    anti = 0
```

Y quitar aristas al azar **no es neutro** — es el brazo `azar` de F7-04, y en este sistema sube la masa.
Medido acá, centrando cada variable dentro de su grafo base (36 corridas a densidad igualada):

```
dosis de dilución vs fracción de masa:  Spearman ρ = +0,671  (p < 0,0001)
```

Y las tres medias caen exactamente en el orden de la dosis:

| brazo | dosis | frac. masa |
|---|---:|---:|
| `alin_dil` | 1 943 | 0,14492 |
| `perm_dil` | 654 | 0,14162 |
| `anti` | 0 | 0,13850 |

Los dos escalones son además del mismo tamaño: `alin_dil`−`perm_dil` = +6,6 partículas y
`perm_dil`−`anti` = +6,3 partículas. **El orden "alineado > permutado > anti" a densidad igualada es
colineal con el orden de la dosis de dilución, y este diseño no tiene ningún par con la MISMA dosis y
distinto origen.** Queda declarado como el límite principal.

Dos matices, en las dos direcciones:

- **A favor de la dosis**: la colinealidad es perfecta y ρ=+0,671 es fuerte.
- **En contra de que sea sólo dosis**: entre grafos, la diferencia de dosis **no** predice la diferencia
  de masa (Δdosis vs Δmasa limpio: ρ=−0,154, p=0,63). Si el efecto fuera puramente dosis, los grafos con
  mayor diferencia de dosis deberían mostrar mayor Δ, y no lo hacen.

El control que faltaría —y que no está en esta corrida— es un par con **dosis y densidad igualadas** y
distinto origen. Es geométricamente imposible construirlo diluyendo, porque los orígenes no arrancan con
el mismo número de aristas; habría que atacarlo por otro lado (por ejemplo, igualar la densidad **antes**
de la poda final, o generar el brazo permutado partiendo de un grafo previamente densificado).

---

## 8. Cruce con O1-D: no es el mismo patrón

`FASE6_O1D_factorial_trio_CS.md` encontró que la alineación **da vuelta el signo** del efecto: con
alineación el trío real ayuda (A−C = −0,105), sin alineación estorba (E−F = +0,057), interacción
z=−6,36. Ahí la magnitud no explicaba nada y la alineación explicaba todo.

Acá el patrón es **el opuesto**: la alineación tiene un efecto gigantesco y unánime, pero **enteramente
a través de la magnitud** (cuántas relaciones sobreviven), y lo que queda cuando se fija la magnitud es
chico, no unánime y no separable de la dosis de dilución. **Con estos datos, F7-06 no es un segundo caso
del patrón de Fase IV.** Lo que sí aporta es la medición directa del mecanismo que O2-C había inferido:
`HET-grado` iba en dirección contraria a las otras cinco distribuciones **porque la alineación con la
estructura preexistente es lo que hace que el cupo no se desperdicie**, y eso deja más aristas.

Reformulado para la teoría, con la cautela del caso: *la capacidad finita genera extensión a través de
cuántos vínculos se llegan a sostener de hecho (O2-C, R²=0,834), y **a qué nodo le toca cada cupo es una
de las manijas más fuertes para mover ese número** — pero, fijado ese número, la identidad del receptor
del cupo aporta poco y ese poco no queda aislado en esta corrida.*

---

## 9. Límites explícitos

- **El límite principal es §7**: a densidad igualada, "origen alineado" y "más dilución al azar" están
  perfectamente confundidos. El contraste limpio (+6,6 partículas, p=0,065/0,098) admite las dos
  lecturas.
- **n=12 grafos.** Con 9/12 signos, la potencia para un efecto de este tamaño es baja. El IC95 de la
  media del Δ limpio incluye el cero por muy poco.
- **Un grafo (r28) aporta +0,0200 de los +0,0033 de media.** Sacándolo el efecto baja a +3,5 partículas.
  No hay explicación de por qué ese grafo se comporta distinto.
- **Un solo eje** (A2-B0-C2), **un solo tamaño** (N=2000), **un solo `kcap` medio por regla** (4-7), **una
  sola forma de distribución** (la de `HET-grado`, CV≈0,37-0,50). No se probó si el resultado cambia con
  repartos de cola pesada (`HET-lognor`, `HET-potencia` de O2-C), donde la alineación podría pesar más.
- **La dilución es al azar uniforme**, no por ningún criterio. F7-04 mostró que *cuáles* aristas se
  pierden importa poco frente a *cuántas*, pero no es cero.
- `unif` no se mandó a Phantom (su número ya estaba en disco desde Fase V-B); se usó sólo como control de
  fidelidad estructural. No hay, por lo tanto, una medida de masa de `unif` tomada en esta misma tanda.
- El coarse-graining de cada brazo se hizo contra NULLs de **su propia** densidad, que es el trato
  estándar cuando los brazos no comparten nº de aristas — pero significa que las pendientes de la tabla
  de §3 no se comparan contra una vara única.

---

## 10. Archivos

| archivo | qué es |
|---|---|
| `cs090_fase7_f706_cupos.py` | script NUEVO: selección, fabricación de los 3 vectores de cupo, 4 brazos de dinámica + 2 diluidos, verificaciones, métricas y escritura de las IC de Phantom |
| `cs090_fase7_f706_correr.py` | script NUEVO: corre Phantom (importa el runner validado de Fase V-B) |
| `cs090_fase7_f706_analizar.py` | script NUEVO: lee los volcados, une por `(rule_id, seed)`, estadística pareada, descuento de densidad y control de dosis |
| `cs090_fase7_f706_seleccion.csv` | los 12 grafos base elegidos (3 por lote) |
| `cs090_fase7_f706_estructura.csv` | **CSV crudo estructural** consolidado (12 filas, un grafo por fila, todos los brazos y todas las verificaciones) |
| `cs090_fase7_f706_estructura_shard00..11.csv` | los shards originales, sin consolidar |
| `cs090_fase7_f706_phantom_crudo.csv` | **CSV crudo de Phantom**: 60 filas (12 grafos × 5 brazos) |
| `cs090_fase7_f706_pares.csv` | una fila por grafo, los 5 brazos lado a lado |
| `cs090_fase7_f706_estadistica.csv` | los contrastes pareados tabulados |
| `cs090_fase7_f706_analisis.log` | la corrida completa del analizador |
| `cs090_fase7_f706_shard00..11.log`, `cs090_fase7_f706_phantom_shard0..4.log` | logs de generación y de Phantom |
| `/Users/alexis/phantom_cs073/bateria_fase7_f706_cupos_alineados/` | 60 carpetas de Phantom (`*_f706_{alineado,permutado,anti,alin_dil,perm_dil}`), prefijo sin colisión |

**Números, no cierre.** La interpretación final es de Alexis.
