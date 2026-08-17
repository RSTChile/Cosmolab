# Fase V-B — escalado a 40 pares (Clase III vs Clase I, A2-B0-C2, K=kcap exacto) + primer test estadístico formal

**Fecha:** 11-ago-2026 · Ejecuta: CC (Claude) · Sigue de `FASE5B_escala_20pares_CS.md` (20 pares, 17 con
match exacto K=kcap).

Alexis pidió escalar esta línea por cuarta vez, con razón explícita: *"Sigue escalando, a estas alturas
sólo estamos blindando el experimento contra críticas y por eso es correcto ser exhaustivos"* — el
objetivo es aumentar poder estadístico y robustez frente a objeciones, no explorar algo nuevo. No se
declara cierre ni veredicto sobre A2-B0-C2. Ningún script congelado de las tres tareas anteriores fue
modificado (`cs090_fase5_generador.py`, `cs090_fase5_motor.py`, `cs090_fase5_clasificador.py`,
`cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_generar_pares.py/_v2.py/_v3.py`,
`cs090_fase5b_correr.py/_v3.py`, `cs090_fase5b_analizar.py`, `cs090_fase5b_consolidar_20pares.py`). No se
hicieron commits de git.

## 0. En simple, con analogía

Ya teníamos 20 pares de "maquetas de alambre" (compacta=Clase I, extendida=Clase III, mismas reglas de
recableo, mismo tamaño de malla) llenas de arena y sacudidas por la gravedad en Phantom. Esta tarea agrega
20 pares más, para llegar a 40 en total.

**6 salieron gratis:** de la tanda anterior de 150 maquetas candidatas ya se habían encontrado 17 parejas
que calzaban exacto en tamaño de malla (K) y tope de vecinos (kcap), pero sólo se usaron 11 — quedaban 6
ya fabricadas y clasificadas, sólo faltaba llenarlas de arena y sacudirlas. Se hizo eso primero, sin
fabricar nada nuevo.

**Los otros 14 salieron de fabricar 220 maquetas nuevas al azar** y quedarse con las parejas que calzaban
exacto — de 220 maquetas nuevas salieron 25 parejas así de limpias (mejor suerte que las tandas
anteriores: 7.5%, 11.3%, ahora 11.4%), y se eligieron 14 priorizando que cubrieran todos los "tamaños de
malla" distintos disponibles, no sólo los más frecuentes.

**Lo nuevo de esta tarea** (a pedido explícito de Alexis, porque el n ya es "sustancial"): con 40 parejas
no basta con mirar "cuántas veces ganó la extendida" a ojo — se aplicó una prueba estadística formal, el
equivalente a preguntar *"si tirara una moneda justa 40 veces, ¿qué tan raro sería que salga cara 31 de
esas 40 veces sólo por azar?"* (test de signos) y una versión más exigente que además pesa *por cuánto*
ganó cada vez, no sólo si ganó (Wilcoxon). Los dos números salieron muy bajos — lo que significa que "es
pura casualidad que la extendida gane tan seguido" es poco creíble bajo estos datos. Eso NO es lo mismo
que "la teoría está confirmada": sólo dice que el patrón de quién-gana-y-por-cuánto no se parece a una
moneda justa. La lectura de qué tan lejos llega esa evidencia es de Alexis.

## 1. Convención de nombres — sin colisión (misma disciplina que v2/v3, extendida)

Mismo fix de dos capas que las tareas anteriores, ahora con un cuarto lote:

1. **`seed_base` nuevo: `571828`.** Sigue el patrón exacto de la línea: `271828` (piloto original) →
   `371828` (escala v2) → `471828` (escala v3) → `571828` (esta tarea). Cada rango individual de seeds ya
   usado (`271829`-`273672`, `371829`-`375612`, `471829`-`~485700`) queda muy lejos de `571828`+220, cero
   colisión numérica posible.
2. **Prefijo `rule_id` nuevo: `"A2-B0-C2-batch4-r{idx}"`.** Nunca usado antes (v1 sin prefijo `r0-r19`,
   v2 sin prefijo `r0-r39`, v3 `batch3-r0..r149` más fixes `r9v2fix`). Verificado por `assert
   f["rule_id"].startswith(PREFIJO_RULE_ID_V4)` sobre las 220 candidatas nuevas, igual que en v3.

**Chequeo adicional específico de esta tarea** (los 6 pares "sobrantes" de v3 son reglas de `batch3`, no
de `batch4`): antes de tocar nada, se verificó por script que los 12 `rule_id` que componen esos 6 pares
NO están en la lista de los 12 `rule_id` de `batch3` que la tarea anterior sí corrió
(`RULE_IDS_YA_CORRIDOS_V3`, comparación de conjuntos, intersección vacía confirmada) — para no correr por
segunda vez una regla que ya tiene resultado de Phantom.

## 2. De dónde salieron los 20 pares nuevos

### 2.1 — Los 6 pares "sobrantes" de v3 (gratis, sin generar nada nuevo)

De las 150 candidatas v3 (`cs090_fase5b_candidatas_v3.csv`, tarea anterior) se habían encontrado
**17 pares exactos** (K,kcap) pero sólo se corrieron 11 (más 1 recuperado del bug de v2 = 12 en total).
Se recalculó programáticamente la lista completa de los 17 pares exactos y se restaron los 12 ya corridos
(ver `cs090_fase5b_generar_pares_v4.py`, `PARES_SOBRANTES_V3`) — quedan estos 6, confirmados sin
ambigüedad:

| par sobrante | regla I (seed) | regla III (seed) | K | kcap |
|---|---|---|---|---|
| `batch3-r59` vs `batch3-r58` | 477552 | 477455 | 7 | 5 |
| `batch3-r107` vs `batch3-r71` | 482208 | 478716 | 6 | 5 |
| `batch3-r112` vs `batch3-r108` | 482693 | 482305 | 6 | 5 |
| `batch3-r120` vs `batch3-r111` | 483469 | 482596 | 6 | 5 |
| `batch3-r76` vs `batch3-r26` | 479201 | 474351 | 8 | 5 |
| `batch3-r143` vs `batch3-r70` | 485700 | 478619 | 4 | 5 |

Estos 6 ya tenían el motor relacional corrido y clasificado desde la tarea anterior (K, kcap, clase, seed
ya en `cs090_fase5b_candidatas_v3.csv`) — sólo faltaba generar la condición inicial de Phantom (grafo +
`layout_resortes`) y correr Phantom, que es lo que hizo esta tarea.

### 2.2 — Generación de 220 candidatas nuevas (`seed_base=571828`, prefijo `batch4`)

Se generaron y clasificaron 220 reglas nuevas con el generador y motor congelados (mismos parámetros que
toda la línea: N=2000, n_sweeps=14, escalas_b=(1,2,4,8,16), n_seeds_null_topo=3, ejes A2-B0-C2, filtro
P1-P5). Las 220 pasaron el filtro P1-P5 sin ninguna descartada. Distribución de clases: 125 Clase I, 73
Clase III, 13 Clase II, 6 "intermedio", 3 Clase IV. Tiempo: 695s (~11.6 min).

**Búsqueda de matches exactos (K,kcap) entre Clase I y Clase III dentro de las 220**: **25 pares exactos
encontrados** (11.4% — consistente con la tasa observada en v3, 11.3%), repartidos en 8 buckets (K,kcap)
distintos: (4,5), (5,5)×7, (5,6)×3, (6,5)×5, (6,6)×2, (7,5)×5, (8,4), (8,5). CSV completo:
`cs090_fase5b_candidatas_v4.csv`.

### 2.3 — Selección de 14 de los 25 para completar los 20 pares nuevos

Con 25 candidatos exactos disponibles y sólo 14 necesarios, se priorizó **diversidad de bucket (K,kcap)**
sobre volumen bruto (mismo criterio que v3): primero 1 par de cada uno de los 8 buckets distintos, luego
6 repeticiones adicionales en los buckets con más de un par disponible ((6,6), (6,5)×2, (5,6), (5,5),
(7,5)). Los 11 pares exactos restantes de las 220 candidatas **no se corrieron** — quedan disponibles sin
generar nada nuevo si Alexis pide escalar más adelante.

## 3. Los 20 pares nuevos — tabla completa

| par | regla I (seed) | regla III (seed) | K | kcap | origen |
|---|---|---|---|---|---|
| sobrante 1 | `batch3-r59` (477552) | `batch3-r58` (477455) | 7 | 5 | 6 sobrantes de v3 |
| sobrante 2 | `batch3-r107` (482208) | `batch3-r71` (478716) | 6 | 5 | 6 sobrantes de v3 |
| sobrante 3 | `batch3-r112` (482693) | `batch3-r108` (482305) | 6 | 5 | 6 sobrantes de v3 |
| sobrante 4 | `batch3-r120` (483469) | `batch3-r111` (482596) | 6 | 5 | 6 sobrantes de v3 |
| sobrante 5 | `batch3-r76` (479201) | `batch3-r26` (474351) | 8 | 5 | 6 sobrantes de v3 |
| sobrante 6 | `batch3-r143` (485700) | `batch3-r70` (478619) | 4 | 5 | 6 sobrantes de v3 |
| nuevo 1 | `batch4-r11` (572896) | `batch4-r0` (571829) | 6 | 6 | 220 candidatas v4 |
| nuevo 2 | `batch4-r31` (574836) | `batch4-r1` (571926) | 6 | 5 | 220 candidatas v4 |
| nuevo 3 | `batch4-r3` (572120) | `batch4-r9` (572702) | 5 | 6 | 220 candidatas v4 |
| nuevo 4 | `batch4-r23` (574060) | `batch4-r10` (572799) | 5 | 5 | 220 candidatas v4 |
| nuevo 5 | `batch4-r15` (573284) | `batch4-r62` (577843) | 8 | 5 | 220 candidatas v4 |
| nuevo 6 | `batch4-r51` (576776) | `batch4-r36` (575321) | 8 | 4 | 220 candidatas v4 |
| nuevo 7 | `batch4-r38` (575515) | `batch4-r72` (578813) | 4 | 5 | 220 candidatas v4 |
| nuevo 8 | `batch4-r57` (577358) | `batch4-r43` (576000) | 7 | 5 | 220 candidatas v4 |
| nuevo 9 | `batch4-r13` (573090) | `batch4-r94` (580947) | 6 | 6 | 220 candidatas v4 |
| nuevo 10 | `batch4-r41` (575806) | `batch4-r12` (572993) | 6 | 5 | 220 candidatas v4 |
| nuevo 11 | `batch4-r18` (573575) | `batch4-r19` (573672) | 5 | 6 | 220 candidatas v4 |
| nuevo 12 | `batch4-r39` (575612) | `batch4-r26` (574351) | 5 | 5 | 220 candidatas v4 |
| nuevo 13 | `batch4-r70` (578619) | `batch4-r47` (576388) | 7 | 5 | 220 candidatas v4 |
| nuevo 14 | `batch4-r44` (576097) | `batch4-r28` (574545) | 6 | 5 | 220 candidatas v4 |

(seeds verificados por script contra `cs090_fase5b_candidatas_v4.csv`, cada `rule_id` es único, sin
colisión — ver §5.)

Los 20 pares tienen K Y kcap exactamente iguales entre su regla I y su regla III.

## 4. Corridas de Phantom — 40 corridas nuevas (N=2000, masa fija=18800)

Mismos parámetros de Phantom que toda la jerarquía CS073: `rho_crit_cgs=1000`, `icreate_sinks=1`,
`r_crit=0.6`, `h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`. Las 40 corridas nuevas alcanzaron `tmax=0.500`
sin abortar (`exit_setup=0`, `exit_run=0` en las 40, `grep -c "exit_setup=0"` y `"exit_run=0"` sobre el
log dan 40/40, cero `AVISO` de fallo). `n_sumideros=8` en las 40 corridas sin excepción — mismo patrón de
saturación de conteo ya documentado en las tareas anteriores (no distingue nada a esta resolución; se
usan fracción de masa y κ_V, no el conteo).

**Costo real medido:**
- Generación de 220 candidatas (motor relacional, sin `layout_resortes`): 695s (~11.6 min).
- Generación de condiciones iniciales (grafo + `layout_resortes`) para las 40 reglas nuevas (20 pares ×
  2): 2770s (~46.2 min) — 43-94s por regla, algo más lento y más variable que en v3 (48-61s) por
  carga de la máquina durante una corrida tan larga, sin ningún patrón sistemático de alarma.
- Phantom puro: 441s (~7.4 min) para las 40 corridas.
- **Total del pipeline de generación+corrida: ~3909s (~65 min)**, dentro del presupuesto de 80-100 min
  pero cerca del límite superior — esta es la escalada más grande hasta ahora (20 pares nuevos vs 12 de
  la tarea anterior) y salió completa sin necesidad de recortar.

## 5. Verificación cruzada — sin colisiones, confirmado

Dos capas de verificación, `AssertionError` como salida si algo no cuadraba:

1. **Antes de generar cualquier IC**: los 20 pares se compararon contra el CSV de origen
   (`cs090_fase5b_candidatas_v3.csv` para los 6 sobrantes, `cs090_fase5b_candidatas_v4.csv` para los 14
   nuevos) — clase, K y kcap declarados debían coincidir con la fila real. **Las 20 verificaciones
   pasaron** (log: `[verificacion previa] 20 pares (6 sobrantes v3 + 14 nuevos v4) verificados contra CSV
   de origen -- OK, ningun mismatch antes de generar IC`). Capa adicional específica de esta tarea:
   `verificar_pares_sobrantes_v3()` confirmó que ninguno de los 12 `rule_id` de los 6 sobrantes ya estaba
   en la lista de los 12 corridos en v3 — sin solape.
2. **Después de generar cada `meta_regla.json` real**: su `seed`, `K` y `kcap` se compararon contra lo
   que el CSV decía que debían ser. **Las 40 generaciones de IC pasaron** (`-> ok, t=Ns, meta verificado
   contra CSV` × 40, sin ningún `AssertionError`).
3. **Chequeo cruzado #3 en el análisis final**: la clase leída del `meta_regla.json` real de cada
   corrida debía coincidir con el rol asignado (I/III) y con K/kcap del par — verificado para las 40
   filas de métricas sin excepción.

`grep -iE "error|traceback|assert" cs090_fase5b_correr_v4.log` → 0 resultados en las ~65 min de
ejecución.

## 6. Resultado agregado FINAL — n=40 pares totales (20 anteriores + 20 nuevos)

CSV consolidado: `cs090_fase5b_TOTAL_40pares.csv` (80 filas = 40 corridas de las tareas anteriores + 40
corridas nuevas). Comparación par-por-par (Clase III − Clase I):

| subconjunto | n pares | media Δfracción (III−I) | III>I en fracción | media Δκ_V (III−I) | III>I en κ_V |
|---|---|---|---|---|---|
| **TODOS (n=40)** | 40 | **+0.0092** | **31/40 (77.5%)** | **+0.0910** | **28/40 (70.0%)** |
| **SOLO match exacto K=kcap (n=37)** | 37 | **+0.0092** | **30/37 (81.1%)** | **+0.0860** | **26/37 (70.3%)** |
| Sólo los 20 pares nuevos de esta tarea | 20 | +0.0065 | 14/20 (70.0%) | +0.0596 | 12/20 (60.0%) |
| Sólo los 20 pares de la tarea anterior (referencia) | 20 | +0.0120 | 17/20 (85.0%) | +0.1225 | 16/20 (80.0%) |

**Lectura honesta, sin cerrar nada:** con n=40 (37 con emparejamiento exacto K=kcap), la dirección **Clase
III > Clase I** en fracción de masa se sostiene en 30/37 (81.1%) de los pares limpios y 31/40 (77.5%) del
total — sigue sin diluirse al escalar, aunque el subconjunto de los 20 pares NUEVOS de esta tarea muestra
un efecto algo más débil (70%/60%) que el de la tarea anterior (85%/80%) — dicho de otro modo, la
tendencia sigue siendo la misma pero esta tanda de 20 pares fue, por azar de qué reglas salieron exactas,
menos unánime que la anterior. El tamaño del efecto en fracción de masa sigue MODESTO (media +0.0092,
sobre fracciones típicas ~0.06-0.15).

## 7. Pruebas estadísticas formales (pedidas explícitamente por Alexis, dado que n ya es sustancial)

Sobre las 40 diferencias pareadas (Δfracción y Δκ_V, Clase III − Clase I), calculadas con
`scipy.stats` (`cs090_fase5b_estadistica_40pares.py`):

| métrica | subconjunto | n | test de signos (binomial exacta, 2 colas) | Wilcoxon signed-rank (2 colas) |
|---|---|---|---|---|
| fracción de masa | TODOS | 40 | 31/40 mismo signo, **p=0.00068** | W=80.00, **p=0.00001** |
| fracción de masa | SOLO exactos | 37 | 30/37 mismo signo, **p=0.00019** | W=59.00, **p=0.00001** |
| κ_V agregado | TODOS | 40 | 28/40 mismo signo, **p=0.01659** | W=195.00, **p=0.00320** |
| κ_V agregado | SOLO exactos | 37 | 26/37 mismo signo, **p=0.02007** | W=180.00, **p=0.00874** |

### Qué prueba exactamente cada test (y qué NO prueba)

- **Test de signos (binomial exacto):** la hipótesis nula es *"el signo de la diferencia por par (¿ganó
  III o ganó I?) es el resultado de una moneda justa, p=0.5 en cada par, independiente entre pares"*. El
  p-valor es la probabilidad de observar, bajo esa moneda justa, un resultado tan o más desbalanceado que
  el observado (31-9, ó 28-12, etc.) puramente por azar. **No** asume nada sobre el tamaño del efecto,
  sólo sobre la dirección.
- **Wilcoxon signed-rank:** la hipótesis nula es *"la distribución de las diferencias pareadas es
  simétrica alrededor de cero"* — una condición más exigente que el test de signos porque incorpora la
  *magnitud* de cada diferencia (ordena las diferencias por valor absoluto y sopesa los rangos), no sólo
  su signo. Un Wilcoxon significativo dice que ni el signo ni el tamaño de las diferencias son
  consistentes con ruido simétrico centrado en cero.
- **Lo que ninguno de los dos prueba:** ni el test de signos ni Wilcoxon establecen una relación causal
  entre "ser Clase III" y "acretar más masa" — ambos son pruebas sobre la estructura estadística de las
  40 (o 37) diferencias observadas EN ESTE DISEÑO PAREADO específico (mismo K, mismo kcap, mismo N=2000,
  mismo protocolo de Phantom). Tampoco descartan que el efecto observado provenga, parcial o totalmente,
  de algún confound no controlado en el diseño (ej. algo correlacionado con la clase que no sea "ser
  Clase III" en el sentido teórico de la Cosmosemiótica) — eso requeriría un diseño distinto, no un test
  estadístico sobre los datos ya generados. **Un p-valor bajo es evidencia en contra de la hipótesis nula
  puntual que cada test declara — no es lo mismo que "A2-B0-C2 está confirmado".** La interpretación
  final de qué tan lejos llega esta evidencia es de Alexis.

En simple: si alguien objeta "eso que ves es pura casualidad, en 40 tiradas de moneda a veces sale
desbalanceado así nomás" — el test de signos calcula exactamente qué tan raro sería eso bajo una moneda
justa (acá, muy raro: p<0.001 en fracción de masa), y Wilcoxon suma la objeción más fina "ya, pero además
mirá que cuando gana III gana por más" (también muy raro: p=0.00001). Ninguno de los dos dice "por qué"
gana III más seguido y por más — sólo dice que el patrón observado no se parece a ruido puro.

## 8. Archivos de esta tarea

- `cs090_fase5b_generar_pares_v4.py` — verifica los 6 pares sobrantes de v3 contra su CSV de origen +
  genera 220 candidatas nuevas (`seed_base=571828`, prefijo `batch4`) + busca pares exactos entre ellas.
  Incluye una reimplementación local de `generar_ic_para_regla` (idéntica en lógica a la de v3) apuntando
  al `BASE_SALIDA` de ESTE módulo — necesaria porque reusar la función de v3 tal cual habría escrito las
  condiciones iniciales nuevas dentro de la carpeta de la tarea anterior (`escala_v3`) por cierre sobre
  una constante ajena, no por bug de nombres. No modifica ningún script congelado.
- `cs090_fase5b_correr_v4.py` — verificación cruzada en 2 capas + genera condiciones iniciales + corre
  Phantom + analiza para los 20 pares nuevos (reusa `correr_una` de `cs090_fase5b_correr.py` y
  `analizar_carpeta` de `cs090_fase5b_analizar.py` sin modificarlos).
- `cs090_fase5b_consolidar_40pares.py` — junta los 20 pares anteriores (`cs090_fase5b_TOTAL_20pares.csv`)
  con los 20 nuevos en un CSV único y calcula la comparación agregada final.
- `cs090_fase5b_estadistica_40pares.py` — test de signos (binomial exacto, `scipy.stats.binomtest`) y
  Wilcoxon signed-rank (`scipy.stats.wilcoxon`) sobre las diferencias pareadas, con la interpretación
  honesta documentada en el propio código.
- `cs090_fase5b_candidatas_v4.csv` — las 220 reglas nuevas generadas y clasificadas (ground truth para
  la verificación cruzada de los 14 pares nuevos).
- `cs090_fase5b_escala_v4_metricas.csv` — las 40 corridas nuevas (20 pares), mismas columnas que las
  tareas anteriores más `par`, `rol`, `match_exacto_K_kcap`, `origen_par`.
- `cs090_fase5b_TOTAL_40pares.csv` — consolidado de las 80 filas (40 anteriores + 40 nuevas), con
  columna `origen_tarea` para distinguir de dónde vino cada fila.
- `cs090_fase5b_generar_pares_v4.log`, `cs090_fase5b_correr_v4.log`, `cs090_fase5b_consolidar_40pares.log`,
  `cs090_fase5b_estadistica_40pares.log` — logs completos, sin errores ni assertions fallidas.
- `/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v4/` — condiciones iniciales, dumps binarios
  y `.sink` de los 20 pares nuevos (6 sobrantes de `batch3` + 14 de `batch4`) corridos en esta tarea.
- Este informe.

Ningún script congelado de las tres tareas anteriores fue modificado. No se declaró cierre ni veredicto
sobre A2-B0-C2. No se hicieron commits de git.
