# FASE 6 · O2-F — N efectivo real de los 40 pares de Fase V-B

**Tarea:** resolver la alerta que dejó abierta `FASE6_O2B_genealogias_escaladas_CS.md`: si las 80 reglas
de los 40 pares salen de sólo **dos** `seed_base`, con ~40 reglas por lote y la ICC medida allí (0,033),
el efecto de diseño sería **deff ≈ 2,3** y el N efectivo caería a **≈17 pares de 40** — con lo cual habría
que recalcular todos los p-valores publicados de la línea.

**Tipo de tarea:** análisis estadístico sobre datos ya existentes. **No se corrió Phantom, no se generaron
reglas nuevas, no se modificó ningún script ni CSV previo.**

**Insumo:** `cs090_fase5b_TOTAL_40pares.csv` (80 filas = 40 pares I vs III).
**Código nuevo:** `fase6_o2f_n_efectivo_fase5b.py`.
**Salidas nuevas:** `FASE6_O2F_estructura_agrupamiento.csv`, `FASE6_O2F_resultados_neff.csv`.

---

## 0 — La pregunta, en lenguaje simple

Si mido la altura de 40 personas pero resulta que son **4 familias de 10 hermanos**, no tengo 40 medidas
independientes: los hermanos se parecen entre sí, así que la información real es menor que 40.

- La **ICC** mide cuánto se parecen los hermanos (0 = no se parecen más que dos desconocidos).
- El **deff** traduce ese parecido a "cuántas medidas verdaderamente independientes tengo":
  `deff = 1 + (m−1)·ICC`, con `m` el tamaño de familia. `N_eff = N / deff`.

O2-B sospechaba que los 40 pares eran **2 familias enormes** de ~40 hermanos cada una. Esta tarea abrió el
archivo y contó las familias de verdad.

---

## 1 — Estructura de agrupamiento REAL de los 80 datos

Reconstruida desde la columna `seed` (cada tanda de Fase V-B generó candidatas con un `seed_base` propio y
derivó las semillas individuales como `seed_base + intento·97 + 1`; los rangos no se solapan, así que la
semilla identifica el lote sin ambigüedad).

| nivel | cuenta |
|---|---|
| filas | 80 |
| pares | 40 |
| `seed` distintos | **76** (4 reglas reutilizadas en 2 pares cada una) |
| `rule_id` distintos | **76** |
| prefijos de `rule_id` | `A2-B0-C2` 18 · `A2-B0-C2-batch3` 34 · `A2-B0-C2-batch4` 28 |

### 1.1 — Lotes de generación (`seed_base`): son CUATRO, no dos

| `seed_base` | filas (reglas) | pares |
|---|---|---|
| 271828 (piloto v1) | 11 | 5 |
| 371828 (escala v2) | 7 | 3 |
| 471828 (escala v3) | 34 | 17 |
| 571828 (escala v4) | 28 | 14 |
| *par con lote mixto* | — | 1 |

**39 de 40 pares tienen las dos reglas del mismo lote.** El tamaño medio de "familia" a nivel de par es
**m̄ = 8-10**, no 40.

### 1.2 — Otras unidades candidatas de agrupamiento

| unidad | niveles | reparto (a nivel de par) |
|---|---|---|
| `kcap` del par | 4 útiles | 4:2 · 5:24 · 6:10 · 7:2 · (2 pares con kcap distinto entre roles) |
| `K` del par | 5 útiles | 4:3 · 5:12 · 6:12 · 7:8 · 8:4 · (1 par mixto) |
| tanda de generación | 3 | v1v2:8 · v3:12 · v4:20 |
| componentes por regla compartida | **36** | tamaños 4, 2, y 34 pares sueltos |
| lote × kcap | 11 | m̄ = 3,6 |

**`kcap` está balanceado dentro del par en 38/40 pares y `K` en 39/40** — como estaba diseñado (los pares
se armaron con `K = kcap` exacto dentro del par).

**Las 4 reglas reutilizadas** (`A2-B0-C2-r6`, `-r9`, `-r19`, `-r39`) generan sólo **2 racimos** de pares no
independientes (uno de 4 pares, uno de 2); los otros 34 pares no comparten corridas con nadie.

---

## 2 — La alerta de O2-B, contrastada con los datos

| escenario | G | m | ICC | deff | **N_eff (de 40)** |
|---|---|---|---|---|---|
| **hipótesis O2-B** (2 lotes × 40 reglas, ICC = 0,033) | 2 | 40 | 0,033 | 2,29 | **17,5** |
| la misma hipótesis pero contando PARES (2 lotes × 20 pares) | 2 | 20 | 0,033 | 1,63 | 24,6 |
| **estructura real**, con la ICC de O2-B | 4 | 10 | 0,033 | 1,30 | 30,8 |
| **estructura real, con la ICC medida acá sobre las Δ** | 4 | 10 | **0,058** | **1,52** | **26,3** |

**La alerta no se confirma en su forma fuerte.** Eran 4 lotes, no 2; el tamaño de conglomerado relevante
para el contraste pareado es ~10 pares, no ~40 reglas. El deff real está entre **1,4 y 2,0** según cómo se
resuelvan los 1-2 pares mixtos, no en 2,3.

Pero **tampoco es 1**: hay agrupamiento medible por lote y el N efectivo baja de 40 a **19-28** pares.

---

## 3 — ICC y deff medidos sobre las diferencias pareadas Δ

Unidad de análisis: **Δ = fracción de masa en sumideros (Clase III) − (Clase I)** por par. Es exactamente
lo que alimenta el test de signos y el Wilcoxon publicados.

### 3.1 — Los 40 pares originales

| unidad de agrupamiento | G | m̄ | ICC | IC 95 % ICC | p perm. | **deff** | **N_eff** |
|---|---|---|---|---|---|---|---|
| **lote (`seed_base`)** | 5 | 8,0 | **+0,058** | [−0,216 · 0,150] | 0,250 | **1,41** | **28,4** |
| lote, pares mixtos reasignados | 4 | 10,0 | +0,075 | [−0,138 · 0,175] | 0,173 | 1,68 | 23,9 |
| **`kcap` del par** | 5 | 8,0 | **−0,104** | [−0,732 · 0,202] | 0,781 | **1,00** | **40,0** |
| `K` del par | 6 | 6,7 | −0,107 | [−0,315 · −0,075] | 0,851 | 1,00 | 40,0 |
| tanda de generación | 3 | 13,3 | +0,025 | [−0,143 · 0,029] | 0,278 | 1,30 | 30,7 |
| componentes por regla compartida | 36 | 1,1 | −0,401 | [−1,913 · 0,061] | 0,739 | 1,00 | 40,0 |
| lote × kcap | 11 | 3,6 | −0,002 | [−0,360 · 0,562] | 0,469 | 1,00 | 40,0 |

### 3.2 — Los 37 pares válidos tras la corrección de diámetro

(se excluyen los 3 pares cuyo brazo "I" se re-etiquetó: `batch3-r100`, `batch4-r51`, `batch3-r143`)

| unidad de agrupamiento | G | m̄ | ICC | p perm. | **deff** | **N_eff** |
|---|---|---|---|---|---|---|
| **lote (`seed_base`)** | 5 | 7,4 | **+0,099** | 0,166 | **1,64** | **22,6** |
| lote, pares mixtos reasignados | 4 | 9,25 | +0,116 | 0,121 | **1,96** | **18,9** |
| `kcap` del par | 4 | 9,25 | −0,089 | 0,730 | 1,00 | 37,0 |
| `K` del par | 6 | 6,2 | −0,123 | 0,857 | 1,00 | 37,0 |
| tanda de generación | 3 | 12,3 | +0,017 | 0,322 | 1,20 | 30,9 |
| componentes por regla compartida | 33 | 1,1 | −0,382 | 0,739 | 1,00 | 37,0 |
| lote × kcap | 9 | 4,1 | +0,080 | 0,272 | 1,25 | 29,6 |

**Lectura:** la **única** unidad de agrupamiento con ICC positiva y no trivial es el **lote de generación**
(y, en menor medida, la tanda, que es casi lo mismo). `kcap`, `K`, y la reutilización de reglas dan **ICC
negativa** — es decir, los pares que comparten `kcap` se parecen *menos* entre sí que dos pares al azar; el
deff se trunca a 1 y no hay pérdida de N por esa vía.

Advertencia honesta: **el IC 95 % de la ICC por lote incluye el 0 en los dos subconjuntos**, y el p de
permutación es 0,12-0,25. Con 4-5 lotes no se puede estimar bien una ICC. El deff de 1,4-2,0 es la mejor
estimación puntual, no un número duro.

---

## 4 — Recálculo de la significancia

### 4.1 — Punto de partida (crudo, como está publicado)

| subconjunto | signos | p signos | Wilcoxon | media Δ | mediana Δ |
|---|---|---|---|---|---|
| 40 pares | **31/40** | 6,80×10⁻⁴ | W = 80, **9,17×10⁻⁶** | +0,00925 | +0,00725 |
| 37 válidos | **29/37** | 7,53×10⁻⁴ | W = 57, **8,87×10⁻⁶** | +0,00989 | +0,00750 |

(coinciden exactamente con `FASE5B_escala_40pares_CS.md` y `FASE6_adopcion_diam_corregido_CS.md`)

### 4.2 — Vía A: inflar el SE / bajar los grados de libertad según el deff (agrupando por lote)

| | 40 pares (deff 1,41) | 40 pares, sin mixto (deff 1,68) | 37 válidos (deff 1,64) | 37, sin mixto (deff 1,96) |
|---|---|---|---|---|
| N_eff | 28,4 | 23,9 | 22,6 | 18,9 |
| test de signos con N_eff | 22/28 · **p = 0,0037** | 19/24 · **p = 0,0066** | 18/23 · **p = 0,0106** | 15/19 · **p = 0,0192** |
| Wilcoxon con z deflactado por √deff | **1,84×10⁻⁴** | 6,10×10⁻⁴ | **5,11×10⁻⁴** | 1,50×10⁻³ |
| t de una muestra, SE × √deff, gl = N_eff−1 | **1,07×10⁻⁴** | 3,97×10⁻⁴ | **3,12×10⁻⁴** | 1,03×10⁻³ |
| t con gl = G−1 (ultra-conservador) | 0,0106 | 0,0255 | 0,0128 | 0,0297 |

### 4.3 — Vía B: tests que respetan el agrupamiento directamente (sin ajustar p a mano)

Agrupando por **lote**:

| test | 40 pares | 40, sin mixto | 37 válidos | 37, sin mixto |
|---|---|---|---|---|
| **SE cluster-robusto (CR1), gl = G−1** | **p = 0,0084** | p = 0,0192 | **p = 0,0141** | p = 0,0285 |
| ↳ deff *empírico* implícito (SE_CR / SE_ingenuo)² | 1,23 → N_eff 32,5 | 1,35 → 29,6 | 1,73 → 21,4 | 1,90 → 19,5 |
| **bootstrap de conglomerados enteros** (20 000 réps.) | **p = 5,0×10⁻⁴** | p < 10⁻⁴ | **p = 4,0×10⁻⁴** | p < 10⁻⁴ |
| ↳ IC 95 % de la media Δ | [+0,0055 · +0,0132] | [+0,0056 · +0,0137] | [+0,0053 · +0,0142] | [+0,0054 · +0,0145] |
| t sobre las G medias de lote | p = 0,0674 | p = 0,0406 | p = 0,0667 | p = 0,0409 |
| **sign-flip de lote ENTERO** | p = 0,125 | p = 0,125 | p = 0,125 | p = 0,125 |
| ↳ *p mínimo alcanzable con ese G* | 0,0625 | 0,125 | 0,0625 | 0,125 |
| sign-flip con estadístico centrado por lote | p ≤ 2×10⁻⁴ | p ≤ 5×10⁻⁵ | p ≤ 2×10⁻⁴ | p ≤ 5×10⁻⁵ |

Agrupando por **`kcap`** (la unidad que O2-B señaló como dominante):

| test | 40 pares | 37 válidos |
|---|---|---|
| CR1, gl = G−1 | p = 1,73×10⁻⁴ | p = 1,08×10⁻³ |
| bootstrap de conglomerados | p < 10⁻⁴ | p < 10⁻⁴ |
| t sobre las medias de `kcap` | p = 0,0090 | p = 0,0090 |
| sign-flip de `kcap` entero | p = 0,0625 *(piso 0,0625)* | p = 0,125 *(piso 0,125)* |

Agrupando por **componentes de reglas compartidas** (G = 36 / 33) y por **lote × kcap** (G = 11 / 9), todos
los tests quedan en 10⁻⁶ – 10⁻³.

### 4.4 — Sobre el sign-flip de conglomerado entero: no se puede leer como "no significativo"

Este test invierte el signo de TODAS las Δ de un lote a la vez, es decir supone que **el lote entero es un
solo dato**. Con 4-5 lotes sólo hay 16-32 reasignaciones posibles, así que **el p mínimo que ese test puede
producir es 0,0625 (G=5) o 0,125 (G=4)**, aunque el efecto fuera infinitamente fuerte. El p observado de
0,125 está en el piso o a un paso del piso: es un **límite de resolución del diseño**, no una medición de
ausencia de efecto. Y el supuesto que lo respalda (lote = un dato ⇔ ICC = 1) está muy lejos de la ICC
medida (0,06-0,12).

Se reporta por completitud y porque era literalmente lo pedido, pero **no es el número que resume la
tarea**. El análogo honesto con la misma lógica y sin el piso artificial es el bootstrap de conglomerados
(p ≈ 5×10⁻⁴) y el CR1 (p ≈ 0,008-0,029).

---

## 5 — La nota pedida: `kcap` está balanceado dentro del par, ¿contamina la diferencia?

**Medido, no asumido.** Misma variable (fracción de masa), dos formas de mirarla:

| | agrupando por `kcap` | agrupando por lote |
|---|---|---|
| **NIVELES** (80 fracciones individuales) | ICC = **+0,853** · η² = **0,771** · F = 85,2 · **p = 3,0×10⁻²⁴** | ICC = +0,157 · η² = 0,146 · F = 4,33 · p = 0,0071 |
| **DIFERENCIAS pareadas** (40 Δ) | ICC = **−0,104** · η² = 0,051 · F = 0,47 · **p = 0,761** | ICC = +0,058 · η² = 0,139 · F = 1,42 · p = 0,249 |

Δ media y aciertos por nivel de `kcap`:

| `kcap` | n pares | Δ media | aciertos III>I |
|---|---|---|---|
| 4 | 2 | +0,0045 | 2/2 |
| 5 | 24 | +0,0090 | 16/24 |
| 6 | 10 | +0,0082 | **10/10** |
| 7 | 2 | +0,0158 | 2/2 |
| mixto | 2 | +0,0160 | 1/2 |

**Conclusión numérica, con la claridad que pide la consigna:** `kcap` es, con mucha diferencia, el mayor
determinante del **nivel** de fracción de masa (explica el 77 % de la varianza entre las 80 corridas
individuales — consistente con el η² = 0,619 que O2-B reportó para la clase). Pero **al restar dentro del
par, ese efecto se cancela por completo**: la ICC de las diferencias por `kcap` es **negativa** (−0,10, es
decir, no hay ningún parecido extra entre pares que comparten `kcap`), el ANOVA da p = 0,76, y el deff se
trunca a **1,00 → N_eff = 40 de 40**.

O sea: **el agrupamiento por `kcap` NO afecta al contraste pareado.** El diseño de Fase V-B, al exigir
`K = kcap` exacto dentro de cada par, ya neutralizó por construcción la fuente de pseudorreplicación que
O2-B había identificado como dominante. Esto es lo contrario de "el resultado se cae": es "el agrupamiento
que más pesa está bloqueado por diseño".

Lo que **sí** queda como agrupamiento genuino para las diferencias es el **lote de generación**, con ICC
pequeña (0,058-0,116) e IC que incluye el 0.

Analogía: `kcap` es como la altura del piso donde se hicieron las mediciones — mueve muchísimo la lectura
absoluta del barómetro, pero si en cada piso mido la diferencia entre dos barómetros, la altura se va sola.

---

## 6 — ¿El efecto se replica lote por lote? (la prueba más informativa dado G pequeño)

Con conglomerados, la pregunta útil no es sólo "cuánto N pierdo", sino "¿aparece el efecto por separado en
cada conglomerado?". Si aparece en los cuatro, la dependencia interna deja de ser una amenaza: son
réplicas, no ecos del mismo dato.

| lote | n pares (40) | signos | p signos | Δ media | Wilcoxon p |
|---|---|---|---|---|---|
| 271828 | 5 | 4/5 | 0,375 | +0,0170 | 0,125 |
| 371828 | 3 | 3/3 | 0,250 | +0,0072 | 0,250 |
| 471828 | 17 | 13/17 | 0,049 | +0,0108 | 0,0017 |
| 571828 | 14 | 11/14 | 0,057 | +0,0058 | 0,0203 |

- **Los 4 lotes van en la misma dirección (Δ media > 0): 4/4.**
- **Fisher** combinando los 4 lotes independientes: χ² = 16,48, **p = 0,036** (37 válidos: 0,040).
- **Stouffer** ponderado por √n: z = 3,08, **p = 0,0021** (37 válidos: z = 3,03, p = 0,0024).

(Fisher y Stouffer usan sólo el test de signos dentro de cada lote, que es muy poco potente con n = 3-5;
con Wilcoxon dentro de lote los p individuales bajan a 0,0017 y 0,020 en los dos lotes grandes.)

---

## 7 — Observables secundarios (40 pares, agrupando por lote)

| observable | signos crudo | Wilcoxon crudo | ICC lote | deff | N_eff | signos con N_eff | Wilcoxon deflactado |
|---|---|---|---|---|---|---|---|
| masa acretada total | 31/40 · 6,80×10⁻⁴ | 9,46×10⁻⁶ | +0,058 | 1,41 | 28,4 | 22/28 · 0,0037 | 1,88×10⁻⁴ |
| κ_V agregado | 28/40 · 0,0166 | 3,20×10⁻³ | +0,057 | 1,40 | 28,6 | 20/29 · **0,061** | **0,0128** |

**κ_V es el observable que sí se mueve de zona.** Su test de signos ajustado por N_eff pasa de 0,017 a
**0,061** (cruza el 0,05 hacia arriba); el Wilcoxon ajustado queda en 0,013. La masa acretada total se
comporta igual que la fracción de masa: sobrevive con holgura.

---

## 8 — Veredicto numérico (sin cierre: son números, no un fallo)

**Estructura real:** 4 lotes de generación (`seed_base` 271828 / 371828 / 471828 / 571828) con 5 / 3 / 17 /
14 pares. No 2 lotes de 40 reglas. 39/40 pares tienen lote homogéneo; 38/40 tienen `kcap` homogéneo;
76 corridas distintas para 80 filas (4 reglas reutilizadas → 2 racimos de pares, 34 pares sueltos).

**deff / N_eff (unidad = lote, sobre las diferencias pareadas):**

| subconjunto | ICC | deff | **N_eff** |
|---|---|---|---|
| 40 pares | 0,058 | 1,41 | **28,4** |
| 40 pares, criterio estricto (pares mixtos reasignados) | 0,075 | 1,68 | 23,9 |
| 37 válidos (diámetro corregido) | 0,099 | 1,64 | **22,6** |
| 37 válidos, criterio estricto | 0,116 | 1,96 | 18,9 |

Es decir: **el N efectivo está entre 19 y 28 pares, no 17 (peor caso de O2-B) ni 40 (lo publicado).**

**¿Sobrevive el resultado principal (III acreta más masa que I)?**

| | crudo | ajustado por deff (signos) | Wilcoxon ajustado | cluster-robusto CR1 | bootstrap de lotes | combinación de lotes (Stouffer) |
|---|---|---|---|---|---|---|
| **40 pares** | 6,8×10⁻⁴ | **0,0037** | 1,8×10⁻⁴ | **0,0084** | 5,0×10⁻⁴ | 0,0021 |
| **37 válidos** | 7,5×10⁻⁴ | **0,0106** | 5,1×10⁻⁴ | **0,0141** | 4,0×10⁻⁴ | 0,0024 |
| 37 válidos, criterio estricto | — | 0,0192 | 1,5×10⁻³ | 0,0285 | <10⁻⁴ | — |

**Sí sobrevive.** Bajo todas las vías razonables el p queda en el rango **10⁻⁴ – 3×10⁻²**, siempre por
debajo de 0,05. Lo que se pierde es una a dos órdenes de magnitud de "certeza": el Wilcoxon publicado de
9×10⁻⁶ es demasiado optimista y debería reportarse como **~5×10⁻⁴ – 1,5×10⁻³** una vez descontada la
estructura de lotes; el test de signos pasa de ~7×10⁻⁴ a **0,004-0,019**.

El único test que no rechaza es el sign-flip de lote entero (p = 0,125), y ése **no puede** rechazar: su
piso es 0,0625-0,125 con 4-5 lotes (§4.4).

**Sobre `kcap`:** balanceado dentro del par, **no** es fuente de pseudorreplicación para el contraste
pareado (ICC de las Δ = −0,10, deff = 1,00, N_eff = 40/40), pese a dominar los niveles individuales
(η² = 0,77). La lectura correcta es **"el agrupamiento dominante no afecta al contraste pareado"**, no
"el resultado se cae".

---

## 9 — Caveats, y lo que esta tarea NO resuelve

1. **La ICC por lote tiene un IC 95 % que incluye el 0** en los dos subconjuntos (p de permutación
   0,12-0,25). Con 4 lotes no se puede estimar bien una ICC. El deff de 1,4-2,0 es una estimación puntual
   defendible, no un número cerrado; el rango honesto de N_eff va de ~19 a 40.
2. **Los lotes están muy desbalanceados** (17 y 14 pares contra 5 y 3). Los dos lotes grandes dominan casi
   toda la inferencia; los pequeños casi no aportan grados de libertad.
3. **Sólo 4 lotes.** Todos los tests que tratan el lote como unidad (sign-flip entero, t sobre medias de
   lote, CR1 con gl = 3-4) tienen potencia muy baja por construcción. La forma de sacarles el techo no es
   estadística sino experimental: **más `seed_base` distintos**, aunque sea con pocos pares cada uno.
4. **No se tocó la dependencia por reutilización de reglas** más allá de medirla: da ICC negativa y sólo
   afecta a 6 de los 40 pares, así que no cambia nada, pero conviene no repetir la práctica.
5. **κ_V queda en zona gris tras el ajuste** (signos 0,061; Wilcoxon 0,013). No es el observable principal
   de la línea, pero su lectura publicada (0,017 / 0,003) sí se debilita.
6. **Esto no es un cierre.** Son los números del ajuste por pseudorreplicación sobre datos existentes; qué
   se hace con ellos lo decide el director.

---

## 10 — Archivos

| archivo | qué es |
|---|---|
| `fase6_o2f_n_efectivo_fase5b.py` | script nuevo, autodescriptivo; reproduce todo lo de este informe |
| `FASE6_O2F_estructura_agrupamiento.csv` | los 40 pares con lote/kcap/K/tanda/componente de cada uno, Δ de los 3 observables y bandera `valido_diam` |
| `FASE6_O2F_resultados_neff.csv` | ICC, IC, deff, N_eff y los 10 p-valores por cada combinación (subconjunto × unidad de agrupamiento) |

Semilla del análisis: `20260811`. 20 000 réplicas para permutaciones y bootstrap; 4 000 para el IC de la
ICC. El sign-flip por conglomerado se enumera **exhaustivamente** cuando G ≤ 16.
