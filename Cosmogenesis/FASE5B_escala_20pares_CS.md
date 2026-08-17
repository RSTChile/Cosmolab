# Fase V-B — escalado a 20 pares (Clase III vs Clase I, A2-B0-C2, K=kcap exacto)

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude) · Sigue de `FASE5B_investigacion_8sumideros_y_escala_CS.md`
(8 pares corridos, 5 con match exacto K=kcap) y de `FASE5B_phantom_A2B0C2_piloto_CS.md` (piloto original,
3 pares).

Alexis autorizó Phantom para esta línea y pidió escalar tres veces en total ("Escala a más pares" →
piloto de 3 → 8 → esta tarea, "Más pares" → 20). No se declara cierre ni veredicto sobre A2-B0-C2 — se
reportan números crudos, la lectura final es de Alexis. Ningún script congelado de las dos tareas
anteriores fue modificado (`cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_generar_pares.py`,
`cs090_fase5b_generar_pares_v2.py`, `cs090_fase5b_correr.py`, `cs090_fase5b_analizar.py`). No se hicieron
commits de git.

## 0. En simple, con analogía

Ya teníamos 8 pares de "maquetas de alambre" (una compacta, Clase I; una extendida, Clase III, ambas
hechas con las mismas reglas de recableo) llenas de arena y sacudidas por la gravedad en Phantom. Esta
tarea agrega 12 pares más, buscando siempre que las dos maquetas de un mismo par tengan el mismo "tamaño
de malla" (K) y el mismo "tope de vecinos" (kcap) — para que la única diferencia real entre las dos sea
la clase (compacta vs extendida), no un parámetro residual colado. De los 12 pares nuevos, **1 salió
gratis**: era un par que la tarea anterior ya había calculado bien pero terminó corriendo por accidente
con la maqueta equivocada (el mismo bug de nombres repetidos que esa tarea ya había documentado) —
esta vez se reconstruyó la maqueta correcta y se aprovechó la otra mitad del par que ya estaba corrida.
Los otros 11 pares salieron de fabricar 150 maquetas nuevas al azar y quedarse con las parejas que
calzaban exacto en tamaño de malla y tope de vecinos — de 150 maquetas nuevas salieron 17 parejas así de
limpias (mejor suerte que la tanda anterior, que sacó sólo 3 de 40). Con eso alcanzó y sobró para los 11
que hacían falta.

## 1. Convención de nombres — sin colisión (lección de la tarea anterior)

La tarea anterior encontró un bug real: el generador de reglas (`cs090_fase5_generador.generar_regla`)
siempre etiqueta `rule_id = f"{eje_A}-{eje_B}-{eje_C}-r{idx}"`, con `idx` reiniciando en 0 en cada
llamada — **nunca incluye el `seed_base`** en el nombre. Dos lotes generados en momentos distintos con
el mismo rango de `idx` (`r0`...`r39`) colisionan en el nombre aunque sean reglas físicamente distintas
(distinto `seed`).

Esta tarea usó dos capas de protección, documentadas en el propio código
(`cs090_fase5b_generar_pares_v3.py`):

1. **`seed_base` nuevo: `471828`.** Sigue el patrón de la línea (`271828` → perfil original de
   `profundizar`, `371828` → escala v2), fuera de cualquier rango de seed ya usado
   (`271829`-`273672` y `371829`-`375612`) — no puede colisionar numéricamente con ningún seed anterior.
2. **Prefijo de `rule_id` sobrescrito de inmediato tras generar cada regla, antes de correr el motor**:
   `"A2-B0-C2-batch3-r{idx}"` en vez de `"A2-B0-C2-r{idx}"` — verificado por inspección de
   `cs090_fase5_motor.correr_regla_coarse` que `rule_id` sólo se usa para ETIQUETAR filas de salida
   (nunca para derivar semilla ni física: `rng = np.random.default_rng(p["seed"]*5000+N)`), así que
   renombrar antes de correr el motor es seguro. El script afirma explícitamente
   (`assert f["rule_id"].startswith(PREFIJO_RULE_ID_V3)`) que las 150 candidatas nuevas llevan el
   prefijo `batch3` — cero solape posible con `r0`-`r19` (perfil original), `r0`-`r39` (escala v2), ni
   con `r2v1fix`/`r12v1fix`/`r9v2fix` (nombres de fix usados en tareas anteriores y en ésta).

**Verificación cruzada obligatoria, hecha ANTES de generar ninguna condición inicial de Phantom**
(`cs090_fase5b_correr_v3.py::verificar_par_contra_csv`): cada uno de los 12 pares se comparó,
`rule_id` por `rule_id`, contra la fila real del CSV de origen (K, kcap, clase, seed) — si algo no
coincidía, el script aborta con `AssertionError` en vez de seguir en silencio. **Los 12 pares pasaron
esta verificación sin excepción** (ver log completo, §5). Una segunda capa de verificación
(`verificar_meta_contra_csv`) comparó, tras generar cada `meta_regla.json` real, sus valores contra lo
que el CSV decía ANTES de generar nada — exactamente el tipo de chequeo que hubiera detectado el bug de
la tarea anterior en el momento en que ocurrió, no después.

## 2. De dónde salieron los 12 pares nuevos

### 2.1 — Contado de pares gratis (sin generar reglas nuevas), como pedía el paso 1 de la tarea

Se revisaron las 18 reglas originales (`cs090_fase5_profundizar_a2b0c2_resumen.csv`) y las 40 de la
tarea anterior (`cs090_fase5b_candidatas_v2.csv`) buscando combinaciones (K,kcap) exactas entre Clase I
y Clase III que **no** estuvieran ya corridas. Resultado: **sólo 1 pareja disponible sin generar nada
nuevo** — y no era una regla nueva, sino la corrección de un bug: dentro de las 40 candidatas de la tarea
anterior, la propia `r9` de ese lote (seed=372702, I, K=5, kcap=6) tiene match exacto con la propia `r39`
de ese lote (seed=375612, III, K=5, kcap=6) — un sexto par exacto que existía en los datos pero nunca se
llegó a correr con el grafo correcto, porque el código de la tarea anterior reusó por error la carpeta ya
corrida de la `r9` del PILOTO original (seed=272702, un grafo distinto, K=5 pero kcap=7) en vez de
generar la condición inicial de la `r9` de ese lote. Se reconstruyó ahora bajo el nombre sin colisión
`A2-B0-C2-r9v2fix` y se reutilizó la corrida YA HECHA de `r39` (carpeta
`bateria_fase5b_a2b0c2_escala_v2/A2-B0-C2-r39_III`, no se regeneró ni se recorrió Phantom de nuevo sobre
ella).

Conclusión honesta de este paso: **no alcanzaban pares gratis para llegar a 12** (sólo 1 disponible) —
hacía falta generar reglas nuevas, como anticipaba la tarea.

### 2.2 — Generación de 150 candidatas nuevas (`seed_base=471828`, prefijo `batch3`)

Se generaron y clasificaron 150 reglas nuevas con el generador y motor congelados
(`cs090_fase5_generador.generar_reglas_clase`, `cs090_fase5_motor.correr_regla_coarse`,
`cs090_fase5_clasificador.clasificar_regla` — mismos parámetros que toda la línea: N=2000, n_sweeps=14,
escalas_b=(1,2,4,8,16), n_seeds_null_topo=3, mismos ejes A2-B0-C2, mismo filtro P1-P5 real). Las 150
pasaron el filtro P1-P5 sin ninguna descartada. Distribución de clases: 75 Clase I, 60 Clase III, 8
Clase II, 4 "intermedio", 3 Clase IV.

Tiempo: 315s (~5.25 min) — mucho más barato que generar condiciones iniciales de Phantom, porque esta
etapa no corre `layout_resortes` (el paso caro, ~50-60s/regla) — sólo corre el motor relacional puro.

**Búsqueda de matches exactos (K,kcap) entre Clase I y Clase III dentro de las 150**: **17 pares
exactos encontrados** — mejor rendimiento que la tarea anterior (3 de 40 = 7.5%) — 17 de 150 = 11.3%.
CSV completo: `cs090_fase5b_candidatas_v3.csv`.

### 2.3 — Selección de 11 de los 17 para completar los 12 nuevos

Con 17 candidatos exactos disponibles y sólo 11 necesarios (1 ya cubierto por el par recuperado), se
priorizó **diversidad de bucket (K,kcap)** sobre volumen bruto: se tomó 1 par de cada uno de los 8 buckets
(K,kcap) distintos presentes entre los 17 ((5,4), (7,6), (6,6), (7,5), (5,5), (6,5), (8,5), (4,5)), y 3
repeticiones adicionales de bucket para sumar observaciones donde había más de un par disponible ((7,6),
(7,5), (6,5) otra vez). Los 6 pares exactos restantes de las 150 candidatas **no se corrieron** —
quedan disponibles sin generar nada nuevo si Alexis pide escalar más adelante.

## 3. Los 12 pares nuevos — tabla completa

| par | regla I (seed) | regla III (seed) | K | kcap | origen |
|---|---|---|---|---|---|
| recuperado | `r9v2fix` (372702) | `r39` [v2, ya corrida] (375612) | 5 | 6 | bug de v2 corregido, gratis |
| nuevo 1 | `batch3-r100` (481529) | `batch3-r0` (471829) | 5 | 4 | 150 candidatas v3 |
| nuevo 2 | `batch3-r1` (471926) | `batch3-r69` (478522) | 7 | 6 | 150 candidatas v3 |
| nuevo 3 | `batch3-r5` (472314) | `batch3-r114` (482887) | 6 | 6 | 150 candidatas v3 |
| nuevo 4 | `batch3-r44` (476097) | `batch3-r10` (472799) | 7 | 5 | 150 candidatas v3 |
| nuevo 5 | `batch3-r86` (480171) | `batch3-r12` (472993) | 5 | 5 | 150 candidatas v3 |
| nuevo 6 | `batch3-r50` (476679) | `batch3-r21` (473866) | 6 | 5 | 150 candidatas v3 |
| nuevo 7 | `batch3-r48` (476485) | `batch3-r25` (474254) | 8 | 5 | 150 candidatas v3 |
| nuevo 8 | `batch3-r35` (475224) | `batch3-r31` (474836) | 4 | 5 | 150 candidatas v3 |
| nuevo 9 | `batch3-r9` (472702) | `batch3-r83` (479880) | 7 | 6 | 150 candidatas v3 |
| nuevo 10 | `batch3-r53` (476970) | `batch3-r23` (474060) | 7 | 5 | 150 candidatas v3 |
| nuevo 11 | `batch3-r104` (481917) | `batch3-r60` (477649) | 6 | 5 | 150 candidatas v3 |

Los 12 pares tienen K Y kcap exactamente iguales entre su regla I y su regla III (criterio "limpio" de
esta tarea). Nótese que `batch3-r9` (seed=472702) es una regla FÍSICAMENTE DISTINTA de la `r9` del
piloto original (seed=272702) y de la `r9v2fix` recuperada (también seed=372702, distinta de
`batch3-r9`) — tres reglas distintas, tres seeds distintos, tres nombres sin colisión, verificado.

## 4. Corridas de Phantom — 24 corridas nuevas (N=2000, masa fija=18800)

Mismos parámetros de Phantom que toda la jerarquía CS073: `rho_crit_cgs=1000`, `icreate_sinks=1`,
`r_crit=0.6`, `h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`. Las 24 corridas nuevas alcanzaron `tmax=0.500`
sin abortar (`exit_setup=0`, `exit_run=0` en las 24, ver log completo `cs090_fase5b_correr_v3.log`).
`n_sumideros=8` en las 24 corridas, sin excepción — consistente con el hallazgo de la tarea anterior
(el conteo de sumideros satura a esta resolución, no distingue nada; se usan fracción de masa y κ_V).

**Costo real medido:** ~50-60s por condición inicial (reconstrucción de grafo + `layout_resortes`,
21 IC nuevas generadas ya que 1 lado del par recuperado se reusó) = ~1265s (~21 min) de generación de IC,
más ~292s (~5 min) de Phantom puro (22 corridas nuevas de Phantom, 2 reusadas sin recorrer) — **~26 min
de principio a fin** para los 12 pares nuevos, dentro del presupuesto estimado (~70-90 min totales,
incluyendo generación de 150 candidatas ~5.25 min y overhead de verificación).

## 5. Verificación cruzada — sin colisiones, confirmado

Dos capas de verificación corrieron en el script (`cs090_fase5b_correr_v3.py`), ambas con
`AssertionError` como salida si algo no cuadraba (no un aviso silencioso):

1. **Antes de generar cualquier IC**: los 12 pares se compararon contra el CSV de origen
   (`cs090_fase5b_candidatas_v3.csv` para los 11 nuevos, `cs090_fase5b_candidatas_v2.csv` para el
   recuperado) — clase, K y kcap declarados deben coincidir con la fila real. **Las 12 verificaciones
   pasaron** (log: `[verificacion previa] 12 pares (1 recuperado + 11 v3) verificados contra CSV de
   origen -- OK, ningun mismatch antes de generar IC`).
2. **Después de generar cada `meta_regla.json` real**: su `seed`, `K` y `kcap` se compararon contra lo
   que el CSV decía que debían ser. **Las 21 generaciones de IC nuevas pasaron** (log: `-> ok, t=Ns,
   meta verificado contra CSV` × 21, sin ningún `AssertionError`).

No se registró ningún error, traceback ni assertion fallida en las ~26 min de ejecución (`grep -iE
"error|traceback|assert" cs090_fase5b_correr_v3.log` → 0 resultados).

## 6. Resultado agregado FINAL — n=20 pares totales (8 anteriores + 12 nuevos)

CSV consolidado: `cs090_fase5b_TOTAL_20pares.csv` (40 filas = 16 corridas de la tarea anterior + 24
corridas nuevas). Comparación par-por-par (Clase III − Clase I) en fracción de masa acretada en
sumideros y en κ_V agregado:

| subconjunto | n pares | media Δfracción (III−I) | III>I en fracción | media Δκ_V (III−I) | III>I en κ_V |
|---|---|---|---|---|---|
| **TODOS (n=20)** | 20 | **+0.0120** | **17/20** | **+0.1225** | **16/20** |
| **SOLO match exacto K=kcap (n=17)** | 17 | **+0.0123** | **16/17** | **+0.1170** | **14/17** |
| Sólo los 12 pares nuevos de esta tarea | 12 | +0.0116 | 11/12 | +0.1331 | 10/12 |
| Sólo los 8 pares de la tarea anterior (referencia) | 8 | +0.0126 | 6/8 | +0.1066 | 6/8 |

**Todos los 12 pares nuevos son match exacto** (K=kcap) por diseño de esta tarea — de los 3 no-exactos
que quedan en el total de 20, los 3 son heredados del piloto/tarea anterior (`PAR_piloto_B_r1_r17`,
`PAR_piloto_C_r6_r14`, `PAR_v2_H_r9_r39`), ya documentados como matches cercanos en esos informes.

Detalle par-por-par completo (20 filas) en la tabla del §3 de este informe y en
`cs090_fase5b_TOTAL_20pares.csv`.

**Lectura honesta, sin cerrar nada:** con el n ampliado a 20 pares (17 con emparejamiento exacto K=kcap,
más del doble que los 8 de la tarea anterior), la dirección **Clase III > Clase I** en fracción de masa
acretada en sumideros se sostiene en **16 de 17** de los pares limpios (94%) y en **17 de 20** de todos
los pares corridos hasta ahora en esta línea (85%) — la tendencia se mantiene consistente al escalar,
no se diluye ni se revierte. En κ_V agregado la dirección III>I se sostiene en 14 de 17 pares exactos
(82%) y 16 de 20 en total — algo menos unánime que en fracción de masa, pero también consistentemente
direccional. El tamaño del efecto sigue siendo MODESTO en fracción de masa (media +0.012, sobre
fracciones típicas de ~0.06-0.15) y bastante más variable en κ_V (rango observado de −0.10 a +0.39 según
el par individual) — sigue sin haber ninguna separación limpia tipo "todas las Clase III muy por encima
de todas las Clase I"; es una tendencia direccional consistente y ahora con más del doble de muestra que
antes, no una frontera dura. La interpretación de qué tan sólido es esto estadísticamente, y si vale la
pena escalar aún más, queda para Alexis.

## 7. Archivos de esta tarea

- `cs090_fase5b_generar_pares_v3.py` — recupera el par "escondido" del bug de v2 (`r9v2fix` vs `r39`) +
  genera 150 candidatas nuevas (`seed_base=471828`, prefijo `batch3`) + busca pares exactos entre ellas.
  No modifica ningún script congelado ni de tareas anteriores.
- `cs090_fase5b_correr_v3.py` — genera condiciones iniciales (con verificación cruzada obligatoria en 2
  capas) + corre Phantom (reusando `correr_una` de `cs090_fase5b_correr.py` sin modificarlo) + analiza
  (reusando `analizar_carpeta` de `cs090_fase5b_analizar.py` sin modificarlo) para los 12 pares nuevos.
- `cs090_fase5b_consolidar_20pares.py` — junta los 8 pares anteriores con los 12 nuevos en un CSV único
  y calcula la comparación agregada final (este informe reporta su salida).
- `cs090_fase5b_candidatas_v3.csv` — las 150 reglas nuevas generadas y clasificadas (ground truth
  usada para las dos capas de verificación cruzada).
- `cs090_fase5b_escala_v3_metricas.csv` — las 24 corridas nuevas (12 pares), con las mismas columnas
  que la tarea anterior (fracción en sumideros, κ_V, tiempo al primer sumidero, etc.) más `par`, `rol`,
  `match_exacto_K_kcap`, `origen_par`.
- `cs090_fase5b_TOTAL_20pares.csv` — consolidado de las 40 filas (16 anteriores + 24 nuevas), con
  columna `origen_tarea` para distinguir de dónde vino cada fila.
- `cs090_fase5b_correr_v3.log` — log completo de la corrida (verificación previa, 21 generaciones de IC
  con su verificación cruzada individual, 22 corridas de Phantom, análisis final) — sin errores ni
  assertions fallidas.
- `/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v3/` — condiciones iniciales, dumps
  binarios y `.sink` de las 11 reglas nuevas de `batch3` + `r9v2fix` corridas en esta tarea.
- Este informe.

Ningún script congelado de las dos tareas anteriores fue modificado. No se declaró cierre ni veredicto
sobre A2-B0-C2. No se hicieron commits de git.
