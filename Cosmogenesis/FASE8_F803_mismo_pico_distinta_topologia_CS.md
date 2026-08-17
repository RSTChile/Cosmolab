# FASE VIII · F8-03 — Mismo pico local, distinta topología: ¿el apiñamiento actúa **a través** del pico o **además** de él?

**Fecha:** 12-ago-2026 · **Ejecuta:** CC (Claude) · **Tarea:** F8-03, Ola 2 de Fase VIII
**Antecedentes directos:** `FASE7_F703_grados_y_triangulos_fijos_CS.md` (+28.7 partículas),
`FASE8_F801_desacoplar_apinamiento_CS.md` (la variable que queda en pie es el **tamaño del soporte**;
el Gini queda excluido por intervención), `FASE7_F705_mediacion_nueva_CS.md` (lo único que sobrevive a
condicionar es el **pico local de densidad inicial**), `FASE8_F805_f703_solver_independiente_CS.md`
(usar el criterio de densidad de Phantom, no FoF laxo, y declararlo antes de correr)
**Reglas de la fase:** `FASE8_PLAN_EJECUCION_CS.md` · **Phantom:** autorizado
**Grano del instrumento:** 1 partícula = 0.0005 de fracción de masa · **piso práctico de un pareado: ~5 partículas** (F8-01)

> No se declara cierre ni veredicto. Ningún script congelado fue modificado (todos sólo se importan).
> No se hicieron commits de git.

---

## 0. En simple, con analogía

Sabemos que dos maquetas con los mismos nudos, los mismos alambres en cada nudo y **exactamente los
mismos triangulitos** juntan distinta cantidad de arena según si los triangulitos están **apretados**
(`solap`) o **repartidos** (`disp`): 28 partículas de diferencia.

También sabemos que cuando la maqueta se suelta en el espacio, la versión apretada arma un **montoncito
más alto** de gas al comienzo. La pregunta de esta tarea es cuál de estas dos historias es la verdadera:

- **A través:** el apiñamiento sirve *sólo porque* arma un montoncito más alto. Si dos maquetas muy
  distintas armaran **el mismo montoncito**, juntarían **la misma arena**.
- **Además:** el apiñamiento hace algo más, que no pasa por lo alto del montoncito.

Para responder no se tocó nada: se fabricaron **tres versiones distintas de cada maqueta** de cada tipo
(mismo criterio, distinta suerte) y se buscó, entre las nueve combinaciones posibles, **la pareja que por
casualidad armaba el mismo montoncito**. La elección se hizo mirando **sólo el montoncito**, antes de
tirar una sola partícula.

**Lo que salió:** el montoncito se lleva **tres cuartas partes** del efecto. Cuando el montoncito queda
igualado, de las 26.5 partículas quedan **entre 5 y 9**, justo en el borde de lo que este instrumento
puede ver. La cadena `topología → pico local → masa` explica casi todo, pero **no cierra del todo**:
queda un residuo pequeño, positivo y sistemático que esta batería no puede ni confirmar ni descartar.

---

## 1. Las tres lecturas, planteadas antes de mirar

| lectura | qué habría que ver | qué se vio |
|---|---|---|
| **A través (cadena cerrada)** | Δmasa ≈ 0 con los picos igualados | **no**: quedan +5 a +9 partículas |
| **Además (segundo canal)** | Δmasa sigue en ≈ +28 | **no**: se cae a un cuarto |
| **Los dos canales coexisten** | Δmasa intermedio; se reporta qué fracción sobrevive | **sí — sobrevive ~19-25%** |

---

## 2. Cómo se igualó el pico: **estrategia (a), por selección — sin ninguna intervención**

La tarea ofrecía dos caminos y pedía preferir el (a). Se hizo el (a) entero: **no se aplicó ninguna
transformación a ningún grafo, ni al layout**. La igualación salió de generar variedad y elegir.

De cada uno de los **12 grafos base** (`F702.seleccionar_grafos(n_por_lote=3)`, los mismos de F7-03) se
destruyeron los triángulos con el piso común de siempre, y desde ese mismo piso se construyeron
**3 realizaciones independientes de `solap` y 3 de `disp`** — mismo criterio de aceptación, mismo motor
de F7-03 importado tal cual, **distinta semilla de azar del brazo**. Las 6 se rebobinan al **mismo `T*`**
(el mínimo de los seis techos). Resultado: 6 grafos por base, **72 en total**, con

- todas las `solap` apiñadas por construcción (soporte chico),
- todas las `disp` repartidas por construcción (soporte grande),
- y **cada una cayendo en un pico distinto**, porque cada realización acomoda los mismos triángulos en
  otro lugar.

Eso da **9 emparejamientos posibles por grafo**. Se eligió el de **|Δpico| mínimo**, y esa elección se
hizo **antes de correr Phantom y sin haber visto ni una masa** (`cs090_fase8_f803_elegir_pares.py` lee
sólo los `meta_regla.json`, que no contienen ninguna salida del solver). El resto de las realizaciones
**también se corrió**: el promedio de las 3 `solap` contra el de las 3 `disp` es el **ancla sin
controlar** — el +28 de F7-03 rehecho en esta misma batería y sobre estos mismos grafos.

### 2.1 — El criterio de "igualado", declarado antes

El pico no tiene barra de error propia (es una medida determinista sobre una IC determinista), pero sí
tiene **dispersión entre realizaciones del mismo brazo**: dos maquetas armadas con el mismo criterio y
distinta suerte caen en picos distintos. Esa es la vara honesta:

> **σ_pico = 1.375** (desviación agrupada del pico entre las 3 realizaciones de un mismo brazo dentro de
> un mismo grafo; 24 grupos). Un par se declara **IGUALADO** si `|Δpico| ≤ σ_pico`.

### 2.2 — **Cuán bien quedó igualado: la igualación real, no la pretendida**

`pico` = `p90/mediana` de la densidad a 8 vecinos sobre `cosmogenesis_ic.txt` — **la misma vara
`geoIC_knn8_p90_med` de F7-05**, declarada antes de correr.

| grafo | T\* | var. elegidas | pico `solap` | pico `disp` | **Δpico** | Δpico si se emparejara por índice | ¿igualado? |
|---|---|---|---|---|---|---|---|
| batch3-r0 | 55 | solap1–disp2 | 11.777 | 11.818 | **−0.041** | 0.742 | **SÍ** |
| batch4-r36 | 31 | solap1–disp1 | 10.743 | 10.924 | **−0.181** | 0.553 | **SÍ** |
| batch3-r111 | 126 | solap2–disp0 | 12.709 | 11.510 | **+1.199** | 2.031 | **SÍ** |
| batch4-r10 | 118 | solap0–disp0 | 15.619 | 14.119 | +1.500 | 3.012 | no |
| batch3-r60 | 239 | solap2–disp1 | 16.378 | 14.159 | +2.219 | 3.686 | no |
| r19 | 939 | solap0–disp0 | 15.225 | 12.692 | +2.533 | 3.105 | no |
| batch4-r62 | 220 | solap0–disp2 | 12.708 | 9.920 | +2.788 | 5.001 | no |
| r14 | 250 | solap2–disp1 | 12.064 | 8.291 | +3.773 | 4.266 | no |
| r17 | 181 | solap0–disp1 | 16.868 | 12.980 | +3.888 | 5.419 | no |
| r20 | 562 | solap1–disp2 | 16.443 | 9.551 | +6.892 | 8.951 | no |
| r39 | 1100 | solap0–disp2 | 17.958 | 8.818 | +9.140 | 15.680 | no |
| r28 | 1149 | solap1–disp2 | 22.267 | 9.808 | +12.459 | 15.381 | no |

**La selección funcionó a medias, y hay que decirlo:** bajó la mediana de |Δpico| de **3.98 a 2.66** y en
**3 de 12** grafos llegó a igualar por debajo del ruido del propio pico. En los otros 9 **no alcanzó**,
por una razón estructural: el pico del brazo apiñado es **sistemáticamente** más alto (12/12 grafos,
media 16.27 contra 10.70), y la dispersión entre realizaciones (σ≈1.4, rango 0.14-5.3) **no llega a
cubrir** un desnivel sistemático que en los grafos de mucho triángulo vale 15 unidades. Nueve pares con
el pico **no igualado** es un control fallido en esos nueve, y por eso el número principal no se apoya
sólo en los tres que sí: se apoya en tres estimaciones independientes que coinciden (§6).

**Un sesgo del propio método, declarado:** los 3 grafos que sí se igualaron son también los de **menor
contraste topológico** (Δsoporte medio −0.042 contra −0.231 en los no igualados). El desnivel de pico
crece con `T`, así que "pico igualable" y "poco apiñamiento diferencial" vienen juntos. Por eso el
análisis no se queda en esos tres (§6.3 y §6.4 lo atacan por dos vías que no dependen de esa selección).

---

## 3. Qué se mantuvo fijo — verificado, no asumido

- **Secuencia de grados nodo por nodo:** `np.array_equal(grados_original, grados_variante)` sobre los
  2000 nodos, con `assert` que aborta el grafo si falla. **Las 72 variantes pasaron.**
- **Nº de aristas** idéntico al original en las 72 (`assert`), sin bucles i-i.
- **Nº de triángulos:** `dif_max = 0` en 5 de los 12 grafos y **`dif_max = 1`** en los otros 7
  (T ∈ {30,31}, {54,55}, {117,118}, {219,220}, {238,239}, {1099,1100}, {1148,1149}). Es la misma
  tolerancia que F7-03 (allí también 1); F8-01 había conseguido 0 con menos variantes por grafo. Se
  reporta grafo por grafo en vez de esconderlo: **un triángulo de 31 es 3%, de 1149 es 0.09%**.
- **Mismo layout y mismo θ en los dos brazos:** `layout_resortes` con `seed_layout=12345` en las 72
  variantes (regla dura de la fase), misma dilatación de 60 pasos, misma turbulencia Mach=3.
- **Misma masa total** (18800) y **2000 partículas de gas** en las 72 corridas.
- **Verificación cruzada contra `meta_regla.json`** antes de la estadística: tarea declarada, brazo,
  variante, `(rule_id, seed)` y carpeta coincidentes con el nombre real, `grados_identicos_al_original`,
  `seed_layout`, nº de aristas, masa total y `T*` idénticos entre variantes del mismo grafo. **Los
  únicos 7 avisos son los `dif_max=1` de arriba.**
- **Unión por `(rule_id, seed, variante)`**, nunca por `rule_id` solo.
- **Grafos guardados** (regla de la fase) en los dos formatos: `grafo_f803.npz` y el canónico
  `grafo_f803.grafo.gz` de F8-00, con su `sha256` anotado dentro del `meta_regla.json`.

**Observable principal, declarado antes de correr:** **fracción de masa en sumideros** de Phantom
(`icreate_sinks=1`, `rho_crit_cgs=1000`), leída con `cs090_fase5b_analizar.analizar_carpeta`. **No** se
usó FoF laxo (aviso de F8-05: el observable puede invertir el signo si se cambia la vara).

---

## 4. Cuánto difería el apiñamiento entre brazos

Medias sobre las 72 corridas. Recordar: mismo N, mismas aristas, mismos grados y el mismo `T*` dentro
de cada grafo.

| medida | `solap` (36) | `disp` (36) |
|---|---|---|
| **tamaño del soporte** `frac_aristas_en_triangulo` | **0.1262** | **0.3102** |
| carga media por arista `tri_por_arista_media` | 2.5302 | 1.0447 |
| solapamiento `frac_aristas_multi_tri` | 0.8383 | 0.0442 |
| Gini de triángulos por nodo | 0.9244 | 0.6079 |
| **pico local de la IC** (p90/mediana) | **16.274** | **10.703** |
| pendiente corregida | 0.7257 | 0.8331 |

En el **par elegido** el contraste topológico sigue entero: soporte **0.127 vs 0.311** (2.4×), carga
media **2.50 vs 1.04**. Es decir: **los pares comparados siguen siendo topológicamente muy distintos**;
lo que se les acercó es el pico.

---

## 5. El resultado, grafo por grafo

Fracción de masa en sumideros. **Ancla** = promedio de las 3 `solap` menos promedio de las 3 `disp`.
**Control** = el par elegido a ciegas por mínimo |Δpico|.

| grafo | T\* | masa `solap` | masa `disp` | **Δ ancla [part.]** | Δpico ancla | **Δ control [part.]** | Δpico control |
|---|---|---|---|---|---|---|---|
| batch4-r36 | 31 | 0.1487 | 0.1462 | +5.0 | 0.43 | **+3.0** | 0.18 |
| batch3-r0 | 55 | 0.1473 | 0.1472 | +0.4 | −0.11 | **+6.0** | 0.04 |
| batch4-r10 | 118 | 0.1352 | 0.1243 | +21.6 | 3.01 | +15.0 | 1.50 |
| batch3-r111 | 126 | 0.1225 | 0.1142 | +16.6 | 2.03 | **+18.0** | 1.20 |
| r17 | 181 | 0.1272 | 0.1130 | +28.4 | 5.42 | +29.0 | 3.89 |
| batch4-r62 | 220 | 0.1137 | 0.1018 | +23.6 | 5.00 | +16.0 | 2.79 |
| batch3-r60 | 239 | 0.1353 | 0.1253 | +20.0 | 3.69 | +14.0 | 2.22 |
| r14 | 250 | 0.1120 | 0.1028 | +18.4 | 4.27 | +12.0 | 3.77 |
| r20 | 562 | 0.1183 | 0.0978 | +41.0 | 8.95 | +35.0 | 6.89 |
| r19 | 939 | 0.1267 | 0.1157 | +22.0 | 3.11 | +17.0 | 2.53 |
| r39 | 1100 | 0.1238 | 0.0957 | +56.4 | 15.68 | +23.0 | 9.14 |
| r28 | 1149 | 0.1362 | 0.1038 | +64.6 | 15.38 | +57.0 | 12.46 |
| **media** | | 0.1289 | 0.1157 | **+26.5** | 5.57 | **+20.4** | 3.88 |

*(en negrita los 3 pares con el pico igualado por debajo de σ_pico)*

### 5.1 — Las pruebas (n=12 grafos, diseño pareado)

| contraste | Δ medio | **en partículas (±EE)** | IC95 | signos | Wilcoxon |
|---|---|---|---|---|---|
| **ANCLA** `solap`−`disp` (sin controlar el pico) | +0.01325 | **+26.5 ± 5.5** | (15.8, 37.2) | **12/12** | **p = 0.0005** |
| **CONTROL** par elegido (|Δpico| mínimo) | +0.01021 | **+20.4 ± 4.2** | (12.2, 28.6) | 12/12 | p = 0.0005 |
| **CONTROL sólo los 3 IGUALADOS** (|Δpico| ≤ 1.375) | +0.00450 | **+9.0 ± 4.6** | (0.02, 18.0) | 3/3 | p = 0.25 (n=3) |
| CONTROL sólo los 9 NO igualados | +0.01211 | +24.2 ± 4.8 | (14.8, 33.7) | 9/9 | p = 0.0039 |

**El ancla reproduce el efecto conocido: +26.5 partículas acá, +28.4 en F8-01, +28.7 en F7-03.** El
instrumento y el tubo están donde estaban; lo que cambió es qué se comparó.

---

## 6. El número de la tarea: cuánto sobrevive con el pico igualado

Tres estimaciones que **no dependen de las mismas suposiciones**, y las tres caen en el mismo lugar:

| vía | qué usa | Δmasa residual [partículas] | p |
|---|---|---|---|
| **6.1 · los 3 pares realmente igualados** | sólo los pares con \|Δpico\| ≤ σ_pico | **+9.0 ± 4.6** | 0.25 (n=3) |
| **6.2 · extrapolación a Δpico = 0** | los 12 pares, recta Δmasa vs Δpico | **+7.5 ± 3.0** | 0.031 |
| **6.3 · mediación sobre las 72 corridas** | todas, con efecto fijo de grafo | **+6.6 ± 1.2** | < 1e-4 |

**Fracción del efecto original que sobrevive: 6.6/26.5 = 25%** (con el pico en escala logarítmica,
4.8/26.5 = **18%**). O sea: **tres cuartas partes del apiñamiento actúan a través del pico local.**

### 6.4 — Los dos canales a la vez, a nivel de par (n=12)

Ajuste `Δmasa ~ Δpico + Δsoporte` sobre los 12 pares elegidos:

| término | coeficiente | EE | p |
|---|---|---|---|
| **Δpico** (partículas por unidad de p90/mediana) | **+3.78** | 0.98 | **0.0038** |
| **Δsoporte** (partículas por unidad de fracción de aristas) | +13.8 | 25.0 | 0.59 |
| **ordenada** (Δpico=0 **y** Δsoporte=0) | **+8.4** | 3.5 | 0.040 |

**Con el pico adentro, el tamaño del soporte deja de aportar** (p=0.59). Esto responde, del lado del
par, la misma pregunta que F8-01 respondió del lado de la topología: el soporte manda **porque mueve el
pico**.

### 6.5 — Lo mismo en rangos, sin suponer ninguna forma

Spearman parcial sobre las 72 corridas, centrando cada valor en la media de su propio grafo:

| relación | ρ parcial |
|---|---|
| masa vs **pico**, descontando el soporte | **+0.830** |
| masa vs **soporte**, descontando el pico | **−0.170** |

Y sin descontar nada, dentro de cada grafo: **ρ(pico, masa) = +0.973** (p=4e-46) contra
ρ(soporte, masa) = −0.912. **El pico local de la condición inicial predice la masa final casi
perfectamente**, corrida por corrida, dentro de un mismo grafo base.

### 6.6 — Robustez: el resultado **depende de qué se llame "pico"**

| vara del pico usada como mediador | residuo del brazo [part.] | fracción que sobrevive |
|---|---|---|
| **`p90/mediana`** (la de F7-05, declarada) | **6.6 ± 1.2** | 25% |
| `log(p90/mediana)` | 4.8 ± 1.6 | 18% |
| `máximo/mediana` | 20.6 ± 2.5 | 78% |
| `CV` de la densidad a 8 vecinos | 25.3 ± 2.5 | **96%** |

**Sólo la vara de F7-05 media.** El CV no media nada (el brazo sigue explicando el 96%): es que el CV
sale prácticamente igual en los dos brazos (**4.124 en `solap` contra 4.149 en `disp`**) — mide otra cosa. El máximo tampoco: es
una sola partícula, ruidosa. **Lo que media es el hombro alto de la distribución, no la cola ni la
dispersión global** — que es exactamente lo que F7-05 había aislado, y encaja con F8-01 (allí la cola por
arista tampoco agregaba nada sobre la media).

---

## 7. Contra el grano y contra el piso

| | Δ [partículas] | contra el grano (1 part.) | contra el piso práctico (~5 part., F8-01) |
|---|---|---|---|
| ANCLA sin controlar | +26.5 ± 5.5 | 27× | 5× |
| control, par elegido | +20.4 ± 4.2 | 20× | 4× |
| **residuo con el pico igualado** | **+6.6 a +9.0** | 7-9× | **1.3-1.8× — en el borde** |

El residuo es **positivo en las tres vías, con el mismo signo, y con 12/12 y 3/3 signos**, pero su
tamaño (5-9 partículas) está apenas por encima del piso empírico de ~5 partículas que F8-01 midió para
un pareado. **No es "cero" y no es "el efecto entero": es un resto chico que este instrumento apenas
resuelve.**

---

## 8. Observables secundarios (medias del par elegido)

| observable | `solap` | `disp` |
|---|---|---|
| **nº de sumideros** | 8.00 | 8.08 (11/12 pares con el mismo número exacto) |
| t del primer sumidero | **0.0308** | 0.0345 |
| κ_V agregado | **0.924** | 0.798 |
| tamaño del soporte | 0.127 | 0.311 |
| pico local de la IC | 15.06 | 11.22 |

**No se forman más grumos: se forman los mismos ~8 y más temprano.** El brazo apiñado enciende antes
(0.0308 vs 0.0345) y acreta más rápido (κ_V 0.92 vs 0.80). Es el mismo patrón de F7-03/F8-01, y encaja
con la lectura de F8-05: **la geometría inicial ordena los montoncitos, la dinámica ordena el encendido**.

---

## 9. Respuesta literal a la pregunta de la tarea

> *¿el apiñamiento de triángulos actúa **a través del** pico local de densidad, o **además de** él?*

**Las dos cosas, con proporciones muy desparejas: ~75-82% a través, ~18-25% además.**

- Con los picos igualados **la masa no se iguala**: quedan **+6.6 a +9.0 partículas** (todas las vías, el
  mismo signo).
- Pero **tampoco queda el efecto entero**: de +26.5 se cae a un cuarto.
- Y con el pico adentro del modelo, **el tamaño del soporte deja de predecir** (p=0.59; ρ parcial −0.17).

**La lectura que sostienen los datos es la tercera: los dos canales coexisten**, con el pico local
llevándose la parte del león. La cadena `topología → pico local → masa` **explica casi todo pero no
cierra**: el residuo es demasiado chico para afirmarlo y demasiado sistemático (12/12, 3/3, tres vías) para
descartarlo con esta batería.

---

## 10. Lo que este experimento NO puede decidir

- **La igualación quedó a medias:** sólo **3 de 12** pares bajaron de σ_pico. Los otros 9 son un control
  fallido, y lo que hay para ellos es extrapolación (6.2) o modelo (6.3), no medición directa.
- **Los 3 igualados son también los de menor contraste topológico** (Δsoporte −0.042 vs −0.231): la
  igualación del pico y la debilidad del contraste vienen juntas por construcción. Que su Δmasa por
  unidad de soporte sea **mayor** (206 vs 129 partículas por unidad) va en contra de leer su +9.0 como
  "efecto chico porque el contraste era chico", pero con n=3 eso es una dirección, no una prueba.
- **El residuo (5-9 partículas) está en el borde del piso práctico** de un pareado (~5, F8-01). Podría
  ser un segundo canal real, o podría ser que el pico **está medido con ruido** — y una variable medida
  con ruido nunca media el 100% aunque sea el único mecanismo. **Esta batería no puede separar esas dos
  explicaciones**, y es la limitación más importante del informe.
- **Sólo media la vara `p90/mediana`.** Con CV o con el máximo el efecto no se media casi nada. El
  resultado está atado a esa definición del pico, heredada de F7-05.
- **`dif_max = 1` triángulo en 7 de los 12 grafos** (0.09%-3% de `T*`). No debería mover 20 partículas,
  pero no es cero.
- **12 grafos, todos Clase III del linaje A2-B0-C2**, y en el **régimen fabricado** de triángulos
  (T\* = 31-1149, mediana 230), no en el que el sistema produce solo (mediana 15, aviso de F8-00). La
  misma limitación abierta desde F7-03.
- **No se probó por qué** un soporte apretado levanta el pico. Eso es geometría del layout, no dinámica:
  F8-02 (manipular el pico a propósito) es el que ataca ese eslabón.

---

## 11. Archivos

**Nuevos (esta tarea):**
`cs090_fase8_f803_mismo_pico.py` (generador: piso común, 3 realizaciones × 2 brazos, mismo `T*`,
medidas, IC, **pico**, guardado del grafo en los dos formatos),
`cs090_fase8_f803_elegir_pares.py` (elección del par **a ciegas de la masa**),
`cs090_fase8_f803_correr.py` (Phantom, protocolo estándar),
`cs090_fase8_f803_analizar.py` (verificación cruzada, ancla vs control, mediación, PNG),
`cs090_fase8_f803_robustez.py` (otras varas del pico, rangos, dos canales a nivel de par).

**CSV / PNG de salida:**
`cs090_fase8_f803_estructura_shard{0..11}.csv` (estructura cruda por variante),
`cs090_fase8_f803_phantom_crudo.csv` (**una fila por corrida de Phantom**, 72 filas),
`cs090_fase8_f803_emparejamientos.csv` (**los 9 candidatos de cada grafo**, con lo que se descartó),
`cs090_fase8_f803_pares_elegidos.csv`, `cs090_fase8_f803_pares_con_masa.csv`,
`cs090_fase8_f803_por_grafo.csv`, `cs090_fase8_f803_estadistica.csv`,
`cs090_fase8_f803_mediacion.csv`, `cs090_fase8_f803_robustez.csv`,
`cs090_fase8_f803_mismo_pico.png` (4 paneles),
`cs090_fase8_f803_shard{0..11}.log`, `cs090_fase8_f803_phantom*_{solap,disp}{0,1,2}.log`.

**Batería:** `/Users/alexis/phantom_cs073/bateria_fase8_f803_mismo_pico/`, carpetas
`<rule_id>_s<seed>_f803_<brazo><realización>` — **prefijo `f803` jamás usado antes**, con el **seed
dentro del nombre** (bug de colisión de nombres, `FASE6_O3B` §2.1). **72 corridas, 1.9 GB.**

**Sólo importados, nunca modificados:** `cs090_fase7_f703_organizacion.py`,
`cs090_fase7_f702_escalera.py`, `cs090_fase8_f801_desacople.py`, `cs090_fase8_f800_grafos.py`,
`cs090_fase6_o3b_rewiring.py`, `cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_analizar.py`,
`cs090_diam_corregido.py`, `cs080_renormalizacion.py`.

**Costos:** 12 procesos en paralelo, ~40 min de reloj para los 72 grafos + 72 condiciones iniciales (la
máquina estaba a carga 300-900 por otros trabajos: cada IC tardó 154-500 s contra los ~71 s típicos).
Phantom: 38 s por corrida, 72 corridas en 6 turnos paralelos, ~19 min. Total ~65 min.

> Sin cierre, sin veredicto, sin commits.
