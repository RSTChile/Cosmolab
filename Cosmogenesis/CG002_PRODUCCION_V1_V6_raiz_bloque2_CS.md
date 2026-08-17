# CG002 — PRODUCCIÓN V1 · V2 · V3 · V6 (la raíz del Bloque 2)

**Fecha:** 2026-08-13 · **Dirige:** Alexis López Tapia · **Ejecuta:** Claude Code
**Protocolo pre-registrado:** `PROTOCOLO_CG002_ACOPLAMIENTO_ORIGINARIO.md` (29-jun-2026)
**Motor:** `cg002_acoplamiento.py` (reusado, no reescrito) · **Orquestador:** `cg002_produccion_v1v6.py`
**Datos:** `cg002_produccion_series.csv` · `cg002_produccion_theta_series.csv` · `cg002_produccion_resumen.json`

> **NO ES UN CIERRE.** Regla permanente del proyecto: ningún veredicto vale sin el director.
> Aquí van los números y la **lectura** de las reglas de decisión pre-registradas, no la sentencia.

---

## 0. Resumen en una página

| | Criterio textual del protocolo | REAL (α=1) | BARAJADO (α=1, fases barajadas) | α=0 (control G) | Lectura |
|---|---|---|---|---|---|
| **V1** | frac. semillas con ≥2 vivos a τ_cap, α=1 **>** α=0 | 0,77 / 0,93 / 1,00 (N=4/6/8) | 0,37 / 0,57 / 0,53 | 0,00 / 0,00 / 0,00 | **PASA contra los dos brazos.** Contra α=0 es artefacto de la regla de parada; **contra el barajado tiene contenido** (Fisher 1,9e−3 · 1,1e−3 · 8,4e−6) |
| **V2** | τ_final>0 en ≥83% con α=1; τ≈0 con α=0 | **1,00** (90/90) | **1,00** (90/90) | **0,00** (0/90), Δ_struct = 0,0 exacto | **PASA pero es casi tautológico** y **no separa** REAL de BARAJADO |
| **V3** | \|T̂\| y ρ con signo estable ≥83% entre semillas del mismo N; ρ no es input | literal 1,00 (**tautología**); no trivial 1,00/1,00/1,00 (N4) y 1,00/0,97/**0,63** (N8) | no trivial 0,65/0,65/0,65 (N4) y 0,50/0,53/0,53 (N8) | no definido | **La lectura literal es tautológica.** En la lectura no trivial REAL pasa en 5 de 6 celdas y el barajado **nunca** llega a 0,83 |
| **V6** | ∂̂S > 2,0 en ≥1 nodo en ≥50% semillas (α=1, N≥4) | **0,00** (0/90) | 0,00 (0/90) | 0,00 | **FALLA.** ∂̂S_max: mediana 1,09 · p90 1,67 · **máximo absoluto 1,83** en 270 corridas |
| V4 | eventos S→0 y W creciente sin reglas dedicadas | comp. 0,93 · fusión 0,99 · aniq. 0,68 | 0,94 · 0,99 · 0,78 | 0 · 0 · (1,00 espuria) | PASS, con una salvedad de medición (§7) |
| V5 | trayectorias no constantes; diám(G) crece en subconjunto | diám>0 en 0,90 | 0,49 | 0,00 | PASS |

**La pregunta central que planteó el director — ¿V1 pasa contra el brazo barajado o sólo contra α=0? — tiene respuesta: pasa contra el barajado.** Es el único de los cuatro veredictos que sobrevive al control que no destruye de más.

**Lo que NO pasa: V6.** Y el protocolo dice, textual, sobre V6: *«Eso cierra CG002»*. Bajo su propio criterio pre-registrado, **CG002 no cierra**.

---

## 1. Smoke — ¿reprodujo lo ya conocido?

**Sí.** `N=2, semillas 1–3, α∈{0,1}, k_max=1500`:

| semilla | ω | δ | c=cos(2πδ/8) | α=1: τ / k / vivos / aristas | α=0: τ / Δ_struct / aristas |
|---|---|---|---|---|---|
| 1 | [3,4] | 1 | +0,707 | 500 / 500 / 2 / 1 | 0 / 0,0 / 0 |
| 2 | [6,2] | 4 | −1,000 | 163 / 226 / 0 / 0 | 0 / 0,0 / 0 |
| 3 | [6,0] | 2 | **0,000** | 0 / 1375 / 0 / 0 | 0 / 0,0 / 0 |

- **V4 PASS · V5 PASS · control G limpio** (τ=0, Δ_struct=0,0, aristas=0 en las tres semillas α=0). Coincide con `INFORME_AVANCE_Cosmogenesis_29jun2026.md` §3.3. **El script no cambió: se pudo continuar.**
- Se verificó además que el motor quedó **bit a bit idéntico** tras el único cambio que le hice (§2): mismos τ, aristas, ∂̂S_max y ρ que antes de tocarlo.

**Dato del smoke que importa para V1 (y que el director pidió aislar):** con α=1 hay semillas que **mueren** y semillas donde el acoplamiento es **exactamente cero** (ω=[6,0] ⇒ δ=2 ⇒ cos(π/2)=0). O sea: la pregunta absoluta *«¿con α=1 sobreviven ≥2 nodos hasta τ_cap?»* **podía dar que no**, y de hecho da que no en 2 de 3 semillas con N=2. V1 no es tautológico en su parte absoluta.

---

## 2. El tercer brazo: qué se barajó y por qué así

### 2.1 Barajar UNA vez al inicio es inservible — demostrado con números

Todas las proto-distinciones arrancan con S₀=1,0 idéntico. Permutar el vector ω al inicio es, por lo tanto, **renombrar los nodos**: un isomorfismo exacto. Verificación directa (N=8, ω=[3,1,6,0,2,5,7,4] vs su permutación [3,0,2,4,6,7,5,1]):

| observable | real | barajado 1 vez |
|---|---|---|
| τ | 207 | 207 |
| k_micro | 226 | 226 |
| Δ_struct | 518,254009 | 518,254009 |
| ρ | 0,369398 | 0,369398 |
| n_vivos / aristas / diámetro / componentes | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| ∂̂S_max · φ · rango · dim_ef | 0,0 · 0,0 · 2,0 · 2,0 | 0,0 · 0,0 · 2,0 · 2,0 |

**Idénticos hasta la sexta cifra.** Es exactamente el NULL-que-era-el-mismo-grafo-renombrado contra el que este proyecto ya se estrelló. **Descartado.**

### 2.2 Lo que sí se usó: barajar **por paso**

α sigue en 1 (**hay acoplamiento**, con el mismo multiset de compatibilidades c_ij), pero la asignación **firma→nodo se re-permuta en cada micro-paso**, con un RNG independiente que no toca el sorteo de ω. Resultado: el acoplamiento existe y tiene la misma estadística, pero **la relación concreta no persiste**. Eso es lo único que separa *«hay acoplamiento»* de *«hay ESTE acoplamiento»* — que es justamente lo que afirma C-N2 (la relación como algo que **se sostiene**, no como un empujón instantáneo).

**Guarda de no-isomorfía (10 semillas, N=8):** 0/10 con S_final idéntico.

| semilla | τ real / baraj. | vivos real / baraj. | aristas real / baraj. | ρ real / baraj. |
|---|---|---|---|---|
| 1 | 500 / 500 | 5 / 3 | 8 / 3 | 1,000 / 0,250 |
| 2 | 500 / 395 | 4 / 0 | 6 / 0 | 1,000 / 0,090 |
| 3 | 500 / 500 | 4 / 2 | 6 / 1 | 1,000 / 0,224 |
| 4 | 500 / 500 | 6 / **8** | 15 / **28** | 1,000 / 1,000 |
| 5 | 500 / 500 | 5 / 3 | 10 / 3 | 1,000 / 0,316 |

La semilla 4 es importante: el barajado **puede ganarle** al real. No es un control que destruya monótonamente.

**Honestidad sobre el estado inicial:** en el **micro-paso 1** el barajado es idéntico al real por construcción — con todas las S iguales a S₀ el estado es intercambiable y permutar ω es un renombre (Δs = −0,05 en los 8 nodos, en ambos). **La divergencia empieza en el paso 2**, cuando las S ya son heterogéneas y la firma se despega de la historia acumulada del nodo. Lo que el brazo destruye no es la distribución de acoplamientos: es **su persistencia**.

---

## 3. Diseño de producción

- **N ∈ {4, 6, 8}** (informe pedía N≥4) · **30 semillas por brazo y por N** (informe pedía S_max≥30; el director pedía ≥12).
- **Bloque principal (θ_CP=0):** 3 N × 3 brazos × 30 = **270 corridas** + 270 sondas con presupuesto de micro-pasos igualado.
- **Bloque dirección (θ_CP≠0):** 2 N × 3 θ × 3 brazos × 30 = **540 corridas**.
- Parámetros del protocolo sin tocar: κₛ=1e−6, S₀=1,0, η=0,05, μ=0,01, K=8, ε_τ=1e−4, τ_cap=500, w_min=0,1, ∂_crit=2,0, k_max=1500.
- **Total: 810 corridas + 270 sondas.**

---

## 4. V1 — «el acoplamiento sostiene la persistencia relacional»

**Criterio textual:** *con α=1, fracción de semillas con ≥2 nodos vivos a τ_cap **>** α=0.*

La frase «a τ_cap» no es inocente, porque los brazos **no paran en el mismo lugar**. Reporto las tres lecturas posibles, crudas:

| N | lectura | REAL | BARAJADO | α=0 |
|---|---|---|---|---|
| 4 | ≥2 vivos al final de la corrida | **0,767** | 0,367 | 0,000 |
| 4 | ≥2 vivos **y** τ llegó a τ_cap | **0,767** | 0,367 | 0,000 |
| 4 | ≥2 vivos a **500 micro-pasos** (presupuesto igualado) | 0,867 | 0,400 | **1,000** |
| 6 | al final de la corrida | **0,933** | 0,567 | 0,000 |
| 6 | y τ llegó a τ_cap | **0,933** | 0,567 | 0,000 |
| 6 | a 500 micro-pasos | 0,933 | 0,567 | **1,000** |
| 8 | al final de la corrida | **1,000** | 0,533 | 0,000 |
| 8 | y τ llegó a τ_cap | **1,000** | 0,533 | 0,000 |
| 8 | a 500 micro-pasos | 1,000 | 0,733 | **1,000** |

**Tests (una cola, REAL > control):**

| N | Fisher vs BARAJADO | Fisher vs α=0 | Mann-Whitney sobre n_vivos, REAL vs BARAJADO |
|---|---|---|---|
| 4 | p = 1,88e−3 | p = 8,7e−11 | U=570, p=0,031 (medias 2,23 vs 1,30) |
| 6 | p = 1,06e−3 | p = 4,2e−15 | U=604, p=0,010 (3,80 vs 2,37) |
| 8 | p = 8,38e−6 | p = 8,5e−18 | U=712,5, p=4,1e−5 (5,00 vs 2,40) |

### 4.1 Contra α=0 el resultado es un artefacto de la regla de parada — declarado

Como verificó el director en el código: `s = (1−μ)·s` ocurre siempre y `_paso_acoplamiento` devuelve ceros si α≤0. Con α=0 **el sistema es aritmética pura**: S(k) = 0,99^k, que cruza κₛ=1e−6 en k=1375. Los 90 corridas α=0 dan **exactamente** k_micro=1375, n_vivos=0, Δ_struct=0,0 — sin una sola excepción, sin varianza entre semillas.

Pero el brazo α=1 **para en k=500** (llega a τ_cap). Así que la comparación «al final de la corrida» enfrenta 500 pasos contra 1375. **Con el presupuesto igualado a 500 micro-pasos, α=0 tiene el 100% de las semillas con ≥2 nodos vivos (S=0,99^500=6,6e−3 ≫ κₛ) y V1 se invierte: α=0 ≥ REAL en las tres N.**

Ninguna de las dos lecturas es neutral: con α=0 «vivo» no tiene punto fijo — es un decaimiento exponencial cuya supervivencia es **función del presupuesto de pasos y de nada más**. **Conclusión: V1 medido contra α=0 mide la regla de parada, no C-N2.** Está declarado como pidió el director: esa comparación estaba ganada de antemano y, mirada de otra forma, perdida de antemano.

### 4.2 Contra el barajado sí hay contenido

Ahí ambos brazos tienen α=1, paran por el mismo criterio y tienen la misma estadística de compatibilidades. **REAL supera al barajado en las tres N**, en fracción binaria y en la distribución completa de nodos vivos, y la separación **crece con N** (p de 3e−2 a 4e−5). La distribución cruda:

| N | brazo | media vivos | distribución (nº nodos vivos: nº semillas) |
|---|---|---|---|
| 4 | REAL | 2,23 | {0:7, 2:6, 3:13, 4:4} |
| 4 | BARAJADO | 1,30 | {0:**19**, 2:2, 3:1, 4:8} |
| 6 | REAL | 3,80 | {0:2, 3:8, 4:12, 5:6, 6:2} |
| 6 | BARAJADO | 2,37 | {0:**13**, 2:4, 3:2, 4:4, 5:1, 6:6} |
| 8 | REAL | 5,00 | {4:11, 5:12, 6:3, 7:4} — **ninguna semilla muere** |
| 8 | BARAJADO | 2,40 | {0:**14**, 2:3, 3:5, 4:1, 5:1, 6:2, 7:2, 8:2} |

El barajado es **bimodal**: o muere entero o sobrevive entero. El real es **unimodal y siempre parcial**. Esa diferencia de forma, no sólo de media, es lo que distingue una relación que persiste de un acoplamiento que se re-sortea.

**Traducción simple.** α=0 es apagar la conversación: todos se callan y se apagan, y sólo importa cuánto rato los dejes sentados. El barajado es una fiesta donde todos hablan pero cada segundo te cambian de interlocutor: hay tanto ruido como en la conversación real, pero **nadie termina de entenderse con nadie** — o la mesa entera se cae, o sobrevive de casualidad. El brazo real es la conversación donde **cada uno habla siempre con los mismos**: sobreviven menos que todos, pero sobreviven **casi siempre algunos**. C-N2 dice que lo que sostiene no es hablar: es hablar **con el mismo**.

---

## 5. V2 — «Δ_struct genera tiempo propio»

**Criterio textual:** *con α=1, τ_final>0 en ≥83% semillas; con α=0, τ≈0.*

| brazo | frac(τ>0) | τ mediana | τ medio | frac. que llega a τ_cap |
|---|---|---|---|---|
| REAL | **1,000** (90/90) | 500 | 470,2 | 0,900 |
| BARAJADO | **1,000** (90/90) | 478 | 388,6 | 0,489 |
| α=0 | **0,000** (0/90) | 0 | 0,0 | 0,000 |

**PASA. Y es casi una tautología — declarado.** Δ_step = Σ|F| + Σ|Δs| + Δ_topo, y con α=0 los tres sumandos son idénticamente cero por construcción (`if alpha <= 0: return ceros`). Δ_struct = 0,0 **exacto** en las 90 corridas: no es un número medido, es una identidad del código. τ es, literalmente, **un contador de pasos en que el acoplamiento hizo algo**. Que α=0 dé τ=0 no puede salir de otra manera.

**Y no separa REAL de BARAJADO: 1,00 vs 1,00.** El tiempo propio emerge de que **haya** acoplamiento, no de que sea **este** acoplamiento. La única diferencia con contenido es de grado: el real llega a τ_cap en el 90% de las semillas y el barajado en el 49% — o sea, el reloj del barajado **se para antes** porque el sistema se muere, no porque el tiempo se comporte distinto.

---

## 6. V3 — «la dirección emerge, no está dada»

**Criterio textual:** *con α=1, |T̂| y ρ tienen signo estable ≥83% entre semillas con mismo N; ρ no es input.*

### 6.1 La lectura literal del criterio es tautológica — declarado

- **|T̂| es un módulo** (`T_hat_mod = ‖flujo_neto‖`): es ≥0 **siempre**. Su «signo estable» es 100% por construcción.
- **ρ = Σmax(F,0)/Σ|F| ∈ [0,1]**: también ≥0 **siempre**. Su «signo estable» es 100% por construcción.

Verificado en las 12 celdas del bloque θ: `literal_frac_signo_T_hat = 1,000` y `literal_frac_signo_rho = 1,000` en **todas**, incluido el brazo barajado. **El criterio V3 tal como está escrito no puede fallar.** Es un 83% que se cumple al 100% sin que el sistema haga nada.

### 6.2 θ_CP activado — qué valor y por qué

Con θ_CP=0 el grafo es simétrico y el código devuelve `V3 = N/A`, como advirtió el director. Activé **θ_CP ∈ {0,1 · 0,3 · 0,5}**, que son **exactamente los tres valores del smoke pre-registrado** (`smoke_test()` §11 del protocolo, `theta_cps=[0.0, 0.1, 0.3, 0.5]`). No elegí un valor nuevo ni lo ajusté post-hoc: reporto la **curva completa** sobre los tres, como pidió el director (nada de umbrales que fabriquen categorías).

### 6.3 Lecturas no triviales (las que sí podían fallar)

`tracking_score` = proyección normalizada del flujo neto realizado sobre el predictor de orientación que induce θ_CP; su **signo sí puede variar** entre semillas. Y `ρ−0,5` = ¿el sistema cayó del lado cooperativo o del competitivo?

| celda | frac \|T̂\|≠0 | asimetría media | **signo estable tracking** | frac degenerada | **signo estable ρ−0,5** | ρ medio |
|---|---|---|---|---|---|---|
| N4 θ=0,1 REAL | 0,87 | 0,145 | **1,000** | 0,13 | 0,87 | — |
| N4 θ=0,1 BARAJADO | 0,87 | 0,075 | 0,654 | 0,13 | 0,70 | — |
| N4 θ=0,3 REAL | 0,87 | 0,506 | **1,000** | 0,13 | 0,80 | — |
| N4 θ=0,3 BARAJADO | 0,87 | 0,226 | 0,654 | 0,13 | 0,70 | — |
| N4 θ=0,5 REAL | 0,87 | 0,863 | **1,000** | 0,13 | 0,70 | 0,793 |
| N4 θ=0,5 BARAJADO | 0,87 | 0,381 | 0,654 | 0,13 | 0,70 | 0,313 |
| N8 θ=0,1 REAL | 1,00 | 0,144 | **1,000** | 0,00 | **1,00** | 0,985 |
| N8 θ=0,1 BARAJADO | 1,00 | 0,165 | 0,500 | 0,00 | 0,77 | 0,337 |
| N8 θ=0,3 REAL | 1,00 | 0,393 | **0,967** | 0,00 | **1,00** | — |
| N8 θ=0,3 BARAJADO | 1,00 | 0,484 | 0,533 | 0,00 | 0,77 | — |
| N8 θ=0,5 REAL | 1,00 | 0,788 | **0,633** | 0,00 | **1,00** | — |
| N8 θ=0,5 BARAJADO | 1,00 | 0,770 | 0,533 | 0,00 | 0,77 | — |

- **REAL alcanza el 0,83 en 5 de 6 celdas**; la excepción es N=8 con θ=0,5 (0,633), el θ más fuerte.
- **El barajado no llega a 0,83 en ninguna celda** (0,500–0,654, o sea: apenas por encima del 0,5 que es tirar una moneda).
- Nótese que **la asimetría del grafo es parecida en ambos brazos** (N8 θ=0,5: 0,788 vs 0,770). O sea: el barajado **también produce un grafo dirigido**; lo que no produce es una dirección que **coincida entre semillas**. La asimetría es del parámetro; la **dirección estable** es de la estructura.

### 6.4 «ρ no es input» — sólo se cumple a medias

| correlación (brazo REAL, n=90) | Spearman | p |
|---|---|---|
| ρ vs media de c_ij legible en el ω inicial | **0,601** | 3,7e−10 |
| ρ vs fracción de pares con c_ij>0 en ω inicial | 0,381 | 2,1e−4 |

ρ **no es una identidad** del input (no es 0,99 como pasó con κ_Δ y la masa), pero es **sustancialmente predecible** desde ω solo: ~36% de la varianza de rangos ya está en la condición inicial. La cláusula del protocolo «ρ no es input» **se cumple parcialmente**; ρ es un híbrido input/output y conviene no usarlo como si fuera puramente emergente.

### 6.5 Un número que no significa lo que parece

`|T̂|` medio va de 6,5e17 a 4,2e39 según la celda. **No es una magnitud interpretable**: es la divergencia exponencial de S (§8) propagada al flujo. Sólo el **signo/dirección** de T̂ tiene contenido; su módulo es una unidad arbitraria que crece con el número de pasos.

---

## 7. V6 — «la frontera operativa emerge» → **FALLA**

**Criterio textual:** *∂̂Sᵢ > ∂_crit (=2,0) en ≥1 nodo en ≥50% semillas (α=1, N≥4).*

| brazo (N≥4, n=90) | frac. C-N4 positivo | ∂̂S_max mediana | p90 | **máximo absoluto** |
|---|---|---|---|---|
| REAL | **0,000** | 1,088 | 1,667 | **1,826** |
| BARAJADO | 0,000 | 0,000 | 1,132 | 1,371 |
| α=0 | 0,000 | 0,000 | 0,000 | 0,000 |

En el bloque θ_CP≠0 (540 corridas más): REAL máx **1,753**, barajado máx 1,344, frac>2,0 = 0,000. **En 810 corridas el umbral 2,0 no se rozó nunca.**

### 7.1 ¿Podía dar otra cosa? Sí — es un FAIL genuino

∂̂Sᵢ = Bᵢ / mean(B | B>0). Si m nodos tienen borde activo, el máximo posible del cociente es **m**. Con m=1 vale exactamente 1,0 y con m=2 es <2 por álgebra: **superar 2,0 exige m≥3**.

- En el brazo REAL, **m≥3 en el 78,9% de las corridas** (m medio 3,62, hasta 8). O sea: la cota lo permitía en 71 de 90 corridas, con techo hasta 8,0.
- Entre esas 71 corridas, el ∂̂S_max observado nunca pasó de **1,826**, y ∂̂S_max/m promedia 0,331.

**No es una imposibilidad de diseño: es que las intensidades de borde salen casi uniformes.** Con acoplamiento simétrico y S divergiendo en bloque, todos los nodos vivos acumulan |F| de un orden parecido, y el cociente al promedio se queda pegado a 1. **No hay nodo que se destaque como frontera.**

### 7.2 Identidad algebraica sospechosa en ∂̂S — declarada

| par | Spearman | p |
|---|---|---|
| ∂̂S_max vs n_vivos | **0,810** | 3,8e−22 |
| ∂̂S_max vs n_aristas | **0,753** | 1,1e−17 |
| ∂̂S_max vs S_final_max | 0,567 | 5,8e−9 |
| Δ_struct vs τ_final | 0,520 | 1,5e−7 |
| ρ vs n_vivos | 0,462 | 4,5e−6 |

∂̂S_max **no es una medida independiente de frontera**: en dos tercios de su varianza de rangos es un reenunciado de *cuántos nodos sobrevivieron con enlaces*. Si algún día V6 diera positivo con estas correlaciones, habría que preguntarse si midió una frontera o volvió a medir la supervivencia.

---

## 8. Guardas adicionales (lecciones caras de esta semana)

1. **«Sobrevivir» aquí significa divergir.** No hay techo de S (`S_MAX_DEFAULT = None`; el propio script anota *«colapso duro mata coop; homeostasis en v0.1d»*). En el brazo REAL la S máxima final tiene **mediana 2,8e15 y llega a 1,9e42**. No hay homeostasis: los nodos «vivos» son nodos en explosión exponencial. Todo V1/V2/V6 se lee en ese régimen.
2. **El observable «aniquilación» de V4 no distingue nada.** Se marca cuando ≥2 nodos cruzan κₛ — y con α=0 eso pasa **siempre** por puro decaimiento (frac 1,00 en las 90 corridas α=0). Como evento emergente sólo vale leído junto con Δ_struct>0.
3. **Piso de ruido.** Con 30 semillas por brazo y por N, el p más chico alcanzable en una tabla 2×2 con separación total es **8,5e−18** (Fisher). Los p que informo para REAL vs BARAJADO van de 1,9e−3 a 8,4e−6: están **muy por encima del piso**, o sea no son artefactos del tamaño de muestra. En cambio los p de REAL vs α=0 (8,7e−11 a 8,5e−18) **están en el piso o cerca**: son separación total, es decir, la aritmética de §4.1, no una medición.
4. **Ningún umbral nuevo.** Se reportan fracciones crudas y distribuciones completas; los únicos umbrales usados (0,83 · 0,50 · ∂_crit=2,0 · κₛ · w_min) son los pre-registrados en junio.
5. **Cambio al motor:** una sola adición a `cg002_acoplamiento.py` — el campo `barajar_por_paso` en `CG002Config` y una línea en el bucle. Con el campo en `False` el motor da resultados **bit a bit idénticos** a los de antes del cambio (verificado sobre el smoke y sobre N=8 seed 1). El sorteo de ω no se tocó: el barajado usa un RNG aparte.

---

## 9. Lectura de las reglas de decisión pre-registradas (§9.2 del protocolo)

Las tres reglas, aplicadas como **lectura**:

| Regla pre-registrada | ¿Aplica? |
|---|---|
| *«Si V1 falla pero V2–V3 pasan → sustrato espurio, revisar implementación»* | **No aplica.** V1 no falla: pasa incluso contra el control que no destruye de más. |
| *«Si SÓLO V1 pasa → C-N1 prolongado, no C-N2»* | **No aplica.** Pasan V1, V2, V3 (lectura no trivial, 5/6 celdas), V4 y V5. |
| *«Si todos fallan en α=1 pero el control G muestra lo mismo → falla G, invalidar corrida»* | **No aplica.** El control G está limpio en las 90 corridas (τ=0, Δ_struct=0,0, aristas=0) y no todos fallan. |

**Lo que queda, dicho sin adorno:** la cadena C-N2 → C-N2.5 → C-N2.5.7 tiene evidencia — y en el caso de V1 y V3, evidencia que sobrevive al control barajado, que es el que importa. **El eslabón de cierre, C-N4 / V6, no la tiene: falla limpio, 0 de 810.** El protocolo dice sobre V6 «eso cierra CG002». Con este resultado, **CG002 no cierra bajo su propio criterio**, y lo que hay que discutir con el director es si el que falló es el sistema o el observable ∂̂S (que, según §7.2, en buena parte reenuncia la supervivencia y, según §8.1, se mide sobre un régimen sin homeostasis).

**Preguntas que dejo abiertas para el director, no resueltas por mi cuenta:**
- ¿Se re-corre V6 con techo de S (homeostasis, la «v0.1d» que el propio script anuncia) antes de dar por fallado C-N4?
- ¿Se adopta el brazo barajado como control canónico de CG002 en lugar de α=0, o se conservan los dos?
- ¿Se acepta θ_CP ∈ {0,1 · 0,3 · 0,5} como el régimen pre-registrado para V3, o se pre-registra uno solo?

---

*Datos crudos: `cg002_produccion_series.csv` (270 filas), `cg002_produccion_theta_series.csv` (540 filas), `cg002_produccion_resumen.json`. Reproducible con `./venv/bin/python cg002_produccion_v1v6.py` (~3,5 min).*
