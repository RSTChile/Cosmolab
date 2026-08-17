# AUDITORÍA TRACK E — parámetros a mano / Shannon encubierto / barridos falsos / circularidad

**Fecha:** 2026-07-23
**Alcance:** SOLO modelamiento de factores físicos (campo/expansión/densidad/gravedad/masa). Topología (CG001/ANIMA/VSTCosmo) queda fuera — está cerrada.
**Punto de partida dado por sabido (no repetido aquí):** `HALLAZGO_ABIERTO_etapa7_v6_masa_es_linaje_CS.md` — en v6, `mass_obs` se construye con las mismas variables (`co_member_score`, `n_long_co_pairs`) que decide el juez de linaje → circular. Etapa 7 permanece ABIERTA.

Método: lectura línea por línea de cada archivo (código real, no docstrings), verificación cruzada código↔JSON de producción semilla a semilla, y un barrido de sensibilidad ejecutado en vivo (monkeypatch de constantes de módulo + rerun de `run_controls`) sobre los parámetros que gatean el veredicto final de v6, con evidencia documentada en `etapa7_trackE_auditoria_sensibilidad_v6.py` → `results/etapa7_trackE_auditoria/sensibilidad_v6_result.json`.

---

## 1. Tabla de hallazgos por archivo

### `cs074_rcruz.py`

| parámetro/línea | qué se verificó | evidencia | severidad |
|---|---|---|---|
| `D` medido (línea 139-149), `pasos_lavado` medido (188-220) | ¿son medidos del propio campo o puestos a mano? | Código mide D en un paso de difusión pura y calibra pasos hasta P<0.05 (mediana×1.15); confirmado en `cs074_rcruz_produccion_meta.txt` (mediana_lavado=5300→pasos=6095) | **limpio** |
| `P_LAVADO=0.05`, `MARGEN_LAVADO=1.15`, `control_r0_ok(P_max=0.15)` | ¿números arbitrarios que deciden el veredicto? | Son gates de validez de montaje, no del resultado físico; margen real observado (mean_P_r0=0.034) está 4-5× por debajo del umbral 0.15 → no fragible a ±30-50% | **limpio** (magic number, pero con margen amplio) |
| Eje r `{0,0.1,...,100}`, H=min(r·D,1) | ¿el barrido de r es real? | JSON `cs074_rcruz_produccion_resultado.json`: 10 valores de r × 8 eps, r cruza 1 de verdad, z sube monótono de 0.26→7.77 | **limpio, barrido genuino** |
| Robustez N=400 | ¿mismo umbral se sostiene con N distinto? | `RESUMEN_CS074_rcruz_robustez_N_PARA_CS.md`: max\|ΔP\| N200–N400 = 0.064 | **limpio** |

**Veredicto cs074_rcruz.py: limpio.** Es el archivo mejor construido del stack — D y pasos_lavado son medidos, no elegidos; el barrido de r es real y cruza el régimen crítico; el control de lavado tiene margen amplio.

---

### `TEST_RHO_DISPERSION.py` (Cosmogenesis-Web/codigo/test_rho_dispersion/)

| parámetro/línea | qué se verificó | evidencia | severidad |
|---|---|---|---|
| `STRETCH_RATIO_MAX=0.25`, `SMOOTH_WIDTH_MIN=2.0`, `RHO_SEP_THR=0.08`, `A_END_MIN=50.0` (líneas 50-53) | ¿derivados o a mano? | Comentario dice "pre-registrados; no 1/1836" — sin evidencia de derivación por barrido o histograma | sospechoso-sin-confirmar (magic numbers sin traza) |
| Sensibilidad de esos umbrales | ¿el PASS depende de que estén ajustados finamente? | Valores reales: A_phys_ratio REAL=0.0019 (vs umbral 0.25, 130× de margen); rho_contrast=0.66 (vs 0.08, 8× margen); a_final=403 (vs 50, 8× margen) | **limpio empíricamente** — magic numbers sin derivación documentada, pero con márgenes tan grandes que ±30-50% no cambia el veredicto |
| **Barrido / semillas** | ¿hay barrido de parámetros o de semillas? | El script corre **UNA sola vez**, `SEED=2025` fijo, 4 brazos (REAL + 3 NULL), **sin variar H_EXP, D0, W0, ni repetir con otra semilla** | **confirmado-problema** — no hay barrido de ningún tipo; es un run determinista único. No narra ser un "barrido" (no hay falsa venta), pero tampoco cumple el estándar del resto del stack ("todos los experimentos realizan barridos") |
| `pipeline.py` lee estos flags | ver hallazgo en motor_1a7 más abajo | — | ver sección motor_1a7 |

**Veredicto TEST_RHO_DISPERSION.py: deuda real — es el único experimento del stack que corre una sola semilla sin ningún barrido de robustez.** El resultado en sí no parece frágil (márgenes enormes), pero no hay evidencia de que se sostenga bajo otra semilla o parámetro.

---

### `suite_epocas_masa.py` (v1)

| parámetro/línea | qué se verificó | evidencia | severidad |
|---|---|---|---|
| `mass_obs` pre-E4 | ¿es 0 por construcción o por resultado? | Inicializado en 0.0, solo se calcula dentro de `if grav_active`; JSON confirma `mass_obs_max_pre_E4=0.0` exacto en 12/12 semillas | limpio — cero por construcción, declarado |
| Kill-switch `LEAK_SEP_MAX=0.05`, `LEAK_RATE_MAX=0.25` (detecta si el instrumento de "masa precoz" separa REAL/NULL antes de E4) | ¿funciona, se disparó alguna vez? | K1/K2: `rate_KILL=0.0` en ambos bloques — el kill-switch **se disparó y el resultado se reportó honestamente** como `MASS_E4_OK_BUT_LEAK_INSTRUMENT`, no se maquilló a PASS | **limpio (transparencia real)** |
| `K_MIN,K_MAX=4,14`; `F_CORE_MIN,MAX=0.15,0.75`; `COHESION_MIN,MAX=1.2,6.5` (banda de detección de "átomo" E3) | ¿derivadas de la distribución real o a priori? | Sin cálculo de percentil/histograma previo en el código; idénticas verbatim en v2→v6 (grep confirma) | sospechoso-sin-confirmar — persiste 6 versiones sin re-derivarse ni una vez |
| `VEV_RATIO_MIN=1.35` (línea 36) | ¿se mantiene el mismo valor en versiones siguientes? | **NO** — v2-v6 usan `1.3` embebido directamente en el código (ya no como constante nombrada), sin comentario que explique el cambio 1.35→1.3 | sospechoso-sin-confirmar — drift silencioso de un umbral de criterio E1(VEV) |
| Barridos (TC×R0, H_EXP, L×MIX0, G_GRAV) | ¿reales? | Confirmados en JSON: TC 6 valores×R0 5 valores; H_EXP 7 valores×3 seeds; L×MIX0 25 combinaciones; G_GRAV 9 valores en [0,0.4]×4 seeds — todos genuinos, múltiples valores distintos | **limpio, barridos genuinos** |
| `MASS_E4_SEP_MIN=0.08`, `DENS_ENHANCE_MIN=1.15`, `RATE_PASS=0.55` | sin evidencia de derivación | — | sospechoso-sin-confirmar (heredado sin cuestionar en las 6 versiones) |

**Veredicto v1: relativamente limpio.** El kill-switch anti-fuga funciona y se reporta con honestidad cuando falla.

---

### `suite_epocas_masa_v2_endurecido.py` (v2)

| parámetro/línea | qué se verificó | evidencia | severidad |
|---|---|---|---|
| `MASS_VS_SHUFFLE_MIN=1.4`, `MASS_VS_OFF_MIN=5.0` (líneas 55-56) | ¿se usan realmente en el gate de PASS? | Declaradas pero **nunca referenciadas** en el resto del archivo (`grep -n` solo devuelve la línea de declaración); el gate real usa `mass_E4<=eps` + `DENS_VS_SHUFFLE_MIN` | **confirmado-problema (menor)** — umbrales decorativos, documentación engañosa sobre qué se exige realmente |
| Nulls SHUFFLE/INVERT: `mass_obs=0.0` forzado por rama de código (líneas 371-380) | ¿por construcción o por resultado? | Rama explícita, comentada como diseño ("shuffle/invert no otorgan mass_obs") — legítimo, pero la síntesis `mass_nulls_clean=True` no distingue que 2/3 nulos son tautológicos y solo `off` es empírico | por construcción, declarado — **riesgo de cita fuera de contexto si se usa aguas abajo sin el matiz** |
| `dens_causal` / densidad como evidencia de gravedad→masa | ¿el resultado es honesto? | `mean_dens_real=7.967` vs `mean_dens_shuffle=7.905` (ratio≈1.008, casi idéntico); `dens_causal` rate=0.20/10; veredicto reportado = `E3_OK_E4_CAUSAL_WEAK` (no inflado a PASS) | **limpio (hallazgo físico negativo genuino, reportado sin maquillar)** |
| Barrido G (docstring "barrido amplio de G") | ¿real? | JSON: 9 valores en [0,0.4]×3 seeds = 27 filas genuinas | **limpio** |

**Veredicto v2: relativamente limpio**, con dos deudas puntuales: constantes muertas (MASS_VS_SHUFFLE_MIN, MASS_VS_OFF_MIN) y el matiz perdido de "nulls limpios" mezclando tautológico + empírico.

---

### `suite_epocas_masa_v3_atomic_nb.py` (v3)

| parámetro/línea | qué se verificó | evidencia | severidad |
|---|---|---|---|
| `SOFTENING=1.2`, `DT_NB=0.35` (integrador N-body) | ¿verificados por convergencia/estabilidad? | Sin comentario ni test de convergencia en el código; idénticos v3→v6 sin re-derivación | sospechoso-sin-confirmar |
| `GROUP_LINK_R=4.5` | ¿barrido o a ojo? | Sin barrido propio; usado simultáneamente como radio de "grupo" y de "par cercano" — doble uso sin discusión de si el mismo valor es correcto para ambos roles | sospechoso-sin-confirmar |
| `MASS_REAL_MIN=0.3`, `DENS_MIN=1.2` | ¿alguna vez deciden el veredicto? | JSON `suite_epocas_masa_v3_result.json`: mass_real (cuando≠0) va de 2308.7 a 9392.7 (umbral 0.3, ~1000× de margen); dens_real de 24.7 a 59.4 (umbral 1.2, ~25-50× margen) | **confirmado-problema — mismo patrón "gate decorativo" que v6, ya presente en v3** |
| `GYR_SHRINK_MAX=0.92` | ¿gatea el veredicto? | `gyr_causal` se calcula pero **no entra en ningún AND del veredicto final** (solo se reporta `rate_gyr_causal` como diagnóstico) | **confirmado-problema (gate decorativo, declarado como diagnóstico, no oculto)** |
| `BIND_VS_SHUFFLE_MIN=1.25` | ¿es el que realmente decide? | Verificado semilla-a-semilla contra `e4_causal` en las 10 seeds de control: coincidencia **exacta** 10/10 (7:1.057→F, 42:1.307→T, 99:0.904→F, 777:1.201→F, 2025:1.017→F, 3141:1.370→T, 8191:0.785→F, 99991:1.782→T, 12345:0.742→F, 54321:0.891→F) | **confirmado: único ingrediente real del veredicto**; `mean_bind_vs_shuffle=1.106` está a 12% del umbral 1.25 con rate=0.30/10 — resultado muy sensible a un umbral nunca barrido |
| Kill-switch de v1 (`LEAK_SEP_MAX`/`LEAK_RATE_MAX`) | ¿se sigue aplicando? | **Ausente** en v3 (grep confirma) — solo queda `mean_leak_v3_not_mass` como número reportado, sin gatear nada | **confirmado-problema (silencioso)**: el kill-switch anti-Shannon de v1 se abandonó desde v3 sin nota explícita |
| Barrido G, L | ¿reales? | G: 10 valores [0,0.45]×4 seeds; L: {24,28,32,40}×3 seeds — genuinos | **limpio** |

**Veredicto v3: deuda real** — gates de magnitud decorativos (MASS_REAL_MIN, DENS_MIN, GYR_SHRINK_MAX) y desaparición silenciosa del kill-switch anti-fuga; pero el veredicto reportado (`E3_OK_E4_PARTIAL_bind_sep_weak`) es honesto, no se infla.

---

### `suite_epocas_masa_v4_fusion_lineage.py` (v4)

| parámetro/línea | qué se verificó | evidencia | severidad |
|---|---|---|---|
| `FORCE_CUTOFF=8.0` (línea 57) | comentario dice "~2× radio de grupo" | 2×GROUP_LINK_R(4.5) = 9.0, **no** 8.0 — comentario y valor no coinciden | sospechoso-sin-confirmar (inconsistencia de documentación, no fraude confirmado) |
| `MUTUAL_MIN_STEPS=5` | ¿barrido de sensibilidad? | Ninguno en código ni JSON; heredado sin cambio v4→v6 | sospechoso-sin-confirmar |
| `MASS_REAL_MIN=0.3`, `DENS_MIN=1.2` | ¿deciden? | mass_real (cuando≠0): 788.8–5527.4 (umbral 0.3); dens_real: 30.1–66.9 (umbral 1.2) | **confirmado-problema — decorativos, igual patrón que v3/v6** |
| `MUTUAL_VS_SHUFFLE_MIN=1.25` | ¿es el que decide, y qué tan frágil? | Coincide exactamente con `e4_causal` seed-a-seed. **Dos semillas al borde**: seed 8191 mutual_vs_shuffle=1.222 (falla, 2.2% bajo el umbral) y seed 54321=1.214 (falla, 2.9% bajo) — verificado directamente en `suite_epocas_masa_v4_result.json` | **confirmado-problema de fragilidad** — 2/10 semillas a <3% del umbral nunca barrido |
| `mean_mutual_vs_shuffle` reportado en synthesis = 7,499,603,358,259 | higiene de métrica | Artefacto de dividir por `mutual_shuffle≈0` (seed 99) con épsilon=1e-12; verificado en JSON: fila seed 99 → `mutual_vs_shuffle=74996033582590.45` | **confirmado-problema (menor)** — número sin significado físico que aparece en el RESUMEN.md como si fuera informativo |
| `COMEM_VS_SHUFFLE_MIN=1.15` (criterio OR de linaje, diagnóstico en v4, primario desde v5) | ¿tan laxo que casi todo pasa? | `comem_vs_shuffle` real va de 0.70 a 2.03 en las 10 semillas — el OR de 3 condiciones (comem/fusión/n_long_co) hace que casi cualquier semilla lo cumpla por al menos una vía; `rate_lineage_causal=0.90` | sospechoso — anticipa el patrón que en v5/v6 termina decidiendo casi todo |
| Kill-switch v1 | ausente | igual que v3 | **confirmado-problema (silencioso)**, heredado de v3 |
| Barrido G, L | reales | mismos rangos que v3, genuinos | **limpio** |

**Veredicto v4: mismo patrón que v3, más pronunciado.** Umbrales de magnitud decorativos; el único ratio que decide (MUTUAL_VS_SHUFFLE_MIN) nunca se barrió y está al borde en 2/10 semillas; cambios v3→v4 declarados y motivados por el fallo previo (no oportunismo), pero ningún umbral nuevo se sometió a sensibilidad.

---

### `suite_epocas_masa_v5_linaje.py` (v5)

**Hallazgo central: v5 en sí mismo NO tiene la circularidad de v6.**

| parámetro/línea | qué se verificó | evidencia | severidad |
|---|---|---|---|
| `rate_e4_lineage_pass` | confirmar en JSON | 4/10 exacto (`True`: seeds 42, 8191, 99991, 54321) — coincide con narrativa 0.40 | limpio (transparente) |
| Coincidencia `e4_lineage_pass` vs `lineage_ok` semilla-a-semilla | ¿el gate de masa decide alguna vez, a diferencia de v6? | **Discrepan en 4/10 semillas** (7,777,3141,12345: linaje gana pero mass=0→falla igual; seed 99: mass alta pero linaje pierde→falla). En v6 la coincidencia es 10/10 (mass nunca decide). **En v5 el gate de masa (vía E_mutual) SÍ actúa como filtro real e independiente** | **v5: limpio de este patrón** — el problema nace exactamente en el cambio v5→v6 |
| `mass_obs ∝ (-E_mutual)×dens×gyr×n_groups` | ¿comparte variable con el juez de linaje? | `E_mutual` depende de pares con proximidad sostenida (`FORCE_CUTOFF`, `MUTUAL_MIN_STEPS`); `lineage_wins` depende de `co_member_score`/`fusion_events`/`n_long_co_pairs` (union-find con `GROUP_LINK_R`) — canales relacionados pero no la misma variable | limpio — divergen empíricamente (ver fila anterior) |
| Umbrales 1.15/1.25 | ¿derivados? | Sin evidencia; pero el protocolo documenta explícitamente "no se bajó COMEM_VS_SHUFFLE_MIN tras ver el dato" y el resultado (0.40, FAIL) lo confirma — **no se movió el umbral para forzar PASS** | sospechoso-sin-confirmar por origen, **pero conducta de uso honesta** |
| Barrido G / L declarados | ¿reales? | sweep_G: 10 valores [0,0.45]×4 seeds=40 filas; sweep_L: {24,28,32,40}×3 seeds=12 filas — coinciden exactamente con lo narrado | **limpio, barridos genuinos** |
| Nulls OFF/SHUFFLE/INVERT | ¿masa 0 por construcción, linaje real? | `mass_obs=0` forzado por rama si `grav_mode!="real"` (por construcción, declarado); pero `co_member_score`/`fusion_events` del brazo SHUFFLE sí se calculan empíricamente sobre dinámica con fuentes permutadas (no hardcodeado) | **por construcción (masa) + empírico (linaje)** — distinción que hay que preservar al citar "nulls limpios" |
| `rate_E3=1.0` en las 10 semillas | ¿bandas de átomo demasiado anchas? | Nunca falla — sugiere (no confirma) que K_MIN/MAX, F_CORE, COHESION son tan permisivas que no filtran nada | ver hallazgo transversal abajo (elevado a confirmado-problema) |

**Veredicto v5: limpio respecto al defecto de v6.** Es la versión donde el gate de masa todavía filtra de verdad y por eso falla honestamente (0.40<0.55, reportado sin maquillar). El propio RESUMEN_v5 documenta el fallo y propone exactamente el cambio (masa acoplada a linaje) que en v6 se vuelve circular.

---

### `suite_epocas_masa_v6_mass_linaje.py` (v6)

Circularidad ya documentada y dada por sabida. Hallazgos **adicionales** de esta auditoría (más allá de la circularidad ya confirmada):

| parámetro/línea | qué se verificó | evidencia | severidad |
|---|---|---|---|
| `gyr_causal` (líneas 862-868, 883) | ¿gatea `e4_lineage_pass`? | No — se calcula y se retorna como diagnóstico, pero está ausente del AND en `e4_lineage_pass` (líneas 837-844) | confirmado-problema (decorativo, declarado como diagnóstico) |
| **Sensibilidad real de `COMEM_VS_SHUFFLE_MIN=1.15`** (barrido ejecutado en esta auditoría) | ¿el veredicto flip-flopea con ±50%? | Smoke 3 seeds cercanas al umbral (54321 comem=1.183; 99 comem=1.003; 2025 comem=1.013): **baseline rate_e4_lineage_pass=0.33** (1/3) → **con COMEM_VS_SHUFFLE_MIN×0.5=0.575: rate=1.00 (3/3)**. Ver detalle abajo | **confirmado-problema de fragilidad extrema** — bajar el umbral no derivado en 50% invierte el veredicto de 2 de 3 semillas |
| Título auto-generado del RESUMEN (línea 1086 del script) | ¿coincide con la versión? | El script v6 genera internamente `"# SUITE ÉPOCAS MASA v5 — juez por linaje..."` (dice v5, no v6) — el archivo en disco tiene el título correcto "v6", lo que indica edición manual posterior al string generado por el propio script | confirmado-problema menor (higiene, no afecta el veredicto numérico) |

**Veredicto v6:** confirma y refuerza el hallazgo ya dado (circularidad mass_obs↔lineage_wins) y agrega evidencia nueva de que, además de ser circular, el criterio de linaje en sí es **frágil** ante el único umbral no derivado que lo gatea.

---

### `motor_1a7/pipeline.py` + `estado.py`

| parámetro/línea | qué se verificó | evidencia | severidad |
|---|---|---|---|
| Preferencia v6 sobre v5 si existe el JSON (línea 145) | ya documentado en HALLAZGO — confirmado en código | `if v6_json.exists(): ... usa v6` — reproduce exactamente el mecanismo narrado en el hallazgo abierto | dado por sabido, no se repite como hallazgo nuevo |
| **`_fill_v5_from_json` línea 214-220: `mass_pre_e4_zero`** | ¿el stage "5_orden_sin_masa" refleja datos reales? | ```python\nif rows:\n    estado.mass_pre_e4_zero = True  # by construction of suite\nelse:\n    estado.mass_pre_e4_zero = True\n```<br>**Ambas ramas asignan el mismo valor `True`, sin leer ningún campo del JSON.** El stage `5_orden_sin_masa` (que participa en `cierre_cadena`) es **estructuralmente incapaz de fallar** — no importa qué diga la suite v5/v6 sobre `mass_pre`, siempre pasa | **confirmado-problema** — gate del orquestador que nunca decide, análogo (a nivel de pipeline, no de suite) al patrón "gate decorativo" ya visto en v3/v4/v6 |
| **`stage_3_4_stretch_rho` línea 122-126: lectura de flags de TEST_RHO** | ¿stretch_ok y rho_ok reflejan sus flags individuales (`stretch_pure_ok`, `rho_effect_ok`)? | `v = d.get("verdict", "")` — pero `d["verdict"]` es un **dict**, no string; luego `estado.stretch_ok = "PASS" in str(v) or d.get("flags", {}).get("stretch_pure_ok")`. **`d.get("flags", {})` busca "flags" en el nivel raíz del JSON, pero en el archivo real "flags" está anidado dentro de `d["verdict"]["flags"]`** → siempre da `{}` → `.get("stretch_pure_ok")` siempre `None`. El único término que decide es `"PASS" in str(v)` — un substring-match sobre el dict completo, **igual para stretch_ok y rho_ok** | **confirmado-problema** — los stages 3 y 4 no están, de hecho, gateados por sus flags individuales (`stretch_pure_ok` vs `rho_effect_ok`); ambos colapsan al mismo booleano derivado de una búsqueda de texto sobre el veredicto global. Hoy no cambia el resultado (los 6 flags eran True), pero si algún flag individual fallara sin cambiar la etiqueta global, el pipeline no lo detectaría |
| `RATE_PASS=0.55` (umbral del pre-registro del motor) | ¿derivado o a mano? | `PROTOCOLO_1A7_PREREGISTRO.md` lo fija antes de correr — no se movió tras ver datos (excepto el cambio de juez v5→v6, ya documentado) | sospechoso-sin-confirmar por origen, uso honesto salvo el caso ya conocido |
| Timeline v5 FAIL → v6 PASS | confirmar en logs crudos | `pipeline_produccion.log` (v5): `chain_pass=False`, `rate_e4_lineage=0.4`. `pipeline_produccion_v6.log`: `chain_pass=True`, `rate_e4_lineage=0.8`. Ambos archivos existen en disco con timestamps distintos, consistente con el HALLAZGO | confirma lo ya dado, no es hallazgo nuevo |

**Veredicto motor_1a7: dos bugs de orquestación nuevos, no reportados en el HALLAZGO original**, ambos hacen que el pipeline declare "pass" en etapas que en realidad no está verificando con los datos que dice estar leyendo (stage 5 siempre True por diseño accidental; stages 3/4 colapsados a un solo substring-match que no distingue entre los 6 flags reales de TEST_RHO).

---

### Hallazgo transversal: el criterio E3 ("átomo") nunca falla en NINGUNA versión

Se verificó `rate_E3` / `E3_strict_rate` en el JSON de producción de las **6 versiones**:

| versión | rate E3 (n semillas) |
|---|---|
| v1 | (K1/K2, ver kill-switch — E3 no reportado por separado con el mismo nombre, pero `atom_ok` no es el cuello de botella en ningún bloque) |
| v2 | **1.0** (10 semillas) |
| v3 | **1.0** (10 semillas) |
| v4 | **1.0** (10 semillas) |
| v5 | **1.0** (10 semillas) |
| v6 | **1.0** (10 semillas) |

**En 50+ corridas de control (v2-v6) el criterio de átomo (`K_MIN<=k<=K_MAX`, `F_CORE_MIN<=f_core<=F_CORE_MAX`, `COHESION_MIN<=cohes<=COHESION_MAX`, con `K_MIN,K_MAX=4,14`; `F_CORE_MIN,MAX=0.15,0.75`; `COHESION_MIN,MAX=1.2,6.5`, idénticos y sin cambio en las 5 versiones) NUNCA rechazó una semilla.** Esto se eleva de "sospechoso" a **confirmado-problema**: o bien las bandas son tan anchas que el criterio "átomo" no es un filtro real (cualquier componente conexo típico del campo cae adentro), o bien fueron calibradas (aunque sin evidencia de cálculo explícito en el código) para garantizar ese resultado. No hay manera de distinguir ambas hipótesis sin ver la distribución completa de `k`, `f_core`, `cohes` de TODOS los componentes (no solo los que ya pasaron), lo cual el código no reporta — es en sí mismo un hueco de instrumentación: la suite nunca registra cuántos componentes NO son átomos ni por qué banda fallaron, así que es imposible auditar si el criterio alguna vez estuvo cerca de rechazar algo.

---

## 2. Barridos "vendidos": rango real vs. narrado

| archivo | narrado | rango real (JSON) | veredicto |
|---|---|---|---|
| cs074_rcruz | "r cruza 1" | r∈{0,0.1,0.3,0.5,1,2,5,10,30,100}, 8 eps, N=200/400 | **genuino** |
| v1 K2 | "barrido amplio" | TC: 6 val [0.35,0.80]; R0: 5 val [1.0,3.5] ×2 seeds | **genuino** |
| v1 E4 | "barrido G" | 9 val [0,0.4]×4 seeds | **genuino** |
| v2 | "barrido amplio de G" | 9 val [0,0.4]×3 seeds | **genuino** |
| v3 | "barrido G" | 10 val [0,0.45]×4 seeds | **genuino** |
| v4 | "Barrido G" | 10 val [0,0.45]×4 seeds | **genuino** |
| v5 | "Barrido G: 0…0.45" | 10 val [0,0.45]×4 seeds | **genuino** |
| v6 | (heredado de v5) | 10 val [0,0.45]×4 seeds | **genuino** |
| v1-v6 "L smoke" | {24,28,32,40} | confirmado idéntico en todas | **genuino** |
| **TEST_RHO_DISPERSION** | no narra explícitamente un "barrido" | **una sola corrida, SEED=2025 fijo, sin variar H_EXP/D0/W0, sin repetir semilla** | **no hay barrido — único experimento del stack sin ningún tipo de repetición/robustez** |

**Conclusión de esta sección: todos los barridos de G y L que SÍ se narran como barrido en los RESUMEN son genuinos** (múltiples valores reales, no un solo valor repetido). El problema no es "barrido maquillado" — es la ausencia total de barrido en TEST_RHO_DISPERSION, que nadie prometió pero que rompe la exigencia de "todos los experimentos realizan barridos sobre los rangos de análisis".

---

## 3. Circularidad: mapa completo

| versión | ¿masa comparte variable con el juez de PASS? | tipo |
|---|---|---|
| v1 | `dens_enhance` alimenta tanto `mass_obs` como el gate `dens_causal` | acoplamiento físico declarado y esperado ("masa nace de densificación"), no patológico |
| v2 | igual que v1 | igual |
| v3 | El único ingrediente real del veredicto es `bind_vs_shuffle` (comparación REAL/SHUFFLE genuina, no función trivial de sí misma) — la "masa" en miles es numéricamente irrelevante para el veredicto | **no es la circularidad de v6**, pero comparte el patrón madre: gates de magnitud decorativos + un solo ratio no barrido que gobierna todo |
| v4 | mismo patrón, vía `mutual_vs_shuffle` | igual que v3, con fragilidad confirmada (2 semillas <3% del umbral) |
| v5 | `mass_obs` (vía E_mutual) y `lineage_wins` (vía co_member/fusión) son canales relacionados pero **empíricamente distintos** — discrepan en 4/10 semillas | **limpio de circularidad** |
| v6 | `mass_obs` se construye directamente con `co_member_score` y `n_long_co_pairs`, las mismas variables de `lineage_wins` | **circular — ya documentado en HALLAZGO_ABIERTO** |

**No se encontró ninguna circularidad NUEVA del tipo exacto de v6** (observable construido con las variables del juez) en v1-v5. Lo que sí se encontró en v3/v4 es una variante más leve del mismo patrón madre: el "gate de masa" nominal (MASS_REAL_MIN, DENS_MIN) es decorativo — nunca decide porque los valores reales están 25-1000× por encima del umbral — y el veredicto real depende de un único ratio-vs-shuffle (`bind_vs_shuffle` en v3, `mutual_vs_shuffle` en v4) que nunca fue barrido ni justificado, y que en v4 está al borde (<3%) en 2 de 10 semillas.

---

## 4. Nulls: por construcción vs. por resultado empírico (para no citarlos mal)

| versión | null | tipo | nota |
|---|---|---|---|
| v1-v6 | `mode=="off"` (G=0) → mass=0 | **por construcción** (física trivial: sin gravedad no hay dinámica) | legítimo, declarado |
| v2-v6 | `mode=="shuffle"/"invert"` → `mass_obs=0` | **por construcción** (rama de código que solo calcula mass_obs si `grav_mode=="real"`) | legítimo si se declara así (v6 lo declara explícitamente en el protocolo); **riesgo si se cita "nulls limpios" sin decir que 2 de 3 son tautológicos y solo `off` es prueba física real** |
| v3-v6 | `co_member_score`, `fusion_events`, `n_long_co_pairs`, `E_mutual`, `bind_strength` del brazo SHUFFLE/INVERT | **empírico** — se calculan sobre dinámica real con fuentes de masa permutadas o fuerza invertida, no hardcodeados | correcto citarlos como evidencia real de contraste REAL vs NULL |

**Regla para lectura futura:** "mass_nulls_clean=True" en cualquier versión v2-v6 es **tautológico por diseño para 2/3 modos** (shuffle, invert) y **empírico solo para 1/3** (off). Nunca debe citarse como si los 3 modos fueran evidencia física independiente.

---

## 5. Barrido de sensibilidad ejecutado (v6, parámetros que gatean el veredicto final)

Script: `Cosmogenesis-Web/codigo/suite_epocas_masa/etapa7_trackE_auditoria_sensibilidad_v6.py`
Resultado: `results/etapa7_trackE_auditoria/sensibilidad_v6_result.json`

Método: monkeypatch de las constantes de módulo de `suite_epocas_masa_v6_mass_linaje.py` + rerun de `run_controls()`, usando 3 semillas de producción elegidas por estar **más cerca del umbral de decisión** (`comem_vs_shuffle` cercano a `COMEM_VS_SHUFFLE_MIN=1.15`): seed 54321 (comem=1.183, pasa por poco), seed 99 (comem=1.003, falla por poco), seed 2025 (comem=1.013, falla por poco).

**MASS_REAL_MIN (0.3) y DENS_MIN (1.2) se excluyeron del barrido en vivo** por decisión informada, no por omisión: la propia producción v6 (10 seeds) ya muestra `mass_real` en el rango 43.8–283.9 (100-1000× el umbral) y `dens_real` en 30.1–59.8 (25-50× el umbral) — un ±50% no puede acercarse a esos valores, no ameritan cómputo adicional.

Baseline (3 seeds cercanas al margen): **rate_e4_lineage_pass = 0.33** (1/3 — solo seed 54321 pasa).

| parámetro | valor base | ×0.5 | ×1.5 |
|---|---|---|---|
| **COMEM_VS_SHUFFLE_MIN** | 1.15 | **rate=1.00** (0.575 — las 3 semillas flipean a PASS) | **rate=0.00** (1.725 — la única que pasaba también flipea a FAIL) |
| GROUP_LINK_R | 4.5 | rate=0.33 (2.25 — sin cambio) | rate=0.33 (6.75 — sin cambio) |
| MUTUAL_MIN_STEPS | 5 | **rate=0.67** (2 — sube: una semilla adicional pasa) | rate=0.33 (8 — vuelve al baseline) |

**Lectura:**
1. **COMEM_VS_SHUFFLE_MIN es extremadamente frágil**: el rate barre TODO el rango posible (0.00 → 0.33 → 1.00) con solo ±50% de variación sobre un valor que nunca fue derivado ni barrido en ningún protocolo. Esto confirma, con evidencia empírica directa (no solo analítica), que el criterio "linaje gana" que gatea `e4_lineage_pass` en v5/v6 depende críticamente de la tercera cifra decimal de un número elegido a mano.
2. **GROUP_LINK_R es robusto** en el rango probado (±50%): no cambió el veredicto de ninguna de las 3 semillas — buena señal, no toda la cadena es frágil.
3. **MUTUAL_MIN_STEPS también muestra sensibilidad real, no monótona**: bajarlo a 2 pasos sube el rate 0.33→0.67 (más semillas cuentan como "co-miembros de largo plazo" con un umbral de persistencia más laxo); subirlo a 8 pasos lo devuelve a 0.33 (no a 0.00 — hay compensación parcial vía otras rutas del OR de `lineage_wins`). Que el rate se mueva en absoluto (no es plano) confirma que el parámetro no es inerte; que no sea monótono sugiere una interacción no trivial entre `MUTUAL_MIN_STEPS` y las otras condiciones del OR, no explorada ni documentada en ningún protocolo. Además, el mismo parámetro mueve `n_long_co_pairs`, que en v6 alimenta tanto `lineage_wins` como `mass_obs` — refuerza la circularidad ya documentada, porque un solo umbral no derivado mueve ambos lados del juicio a la vez.

Detalle completo (todas las semillas, todos los factores) en `results/etapa7_trackE_auditoria/sensibilidad_v6_result.json`.

---

## 6. Veredicto de conjunto

- **cs074_rcruz.py: limpio.** El mejor construido del stack — parámetros medidos del propio sistema, barrido de r genuino y decisivo, robustez a N confirmada.
- **TEST_RHO_DISPERSION.py: deuda de barrido.** Único experimento sin ningún tipo de repetición (ni semillas, ni parámetros); los umbrales tienen márgenes enormes hoy, pero nunca se probó su robustez.
- **suite_epocas_masa v1-v2: relativamente limpias.** Kill-switch honesto, fallos reportados sin maquillar, barridos genuinos. Deuda: bandas E3 (K_MIN/MAX, F_CORE, COHESION) sin trazabilidad de origen, heredadas sin cambio en 6 versiones; drift silencioso del umbral E1 (VEV_RATIO_MIN 1.35→1.3) entre v1 y v2.
- **v3-v4: deuda estructural real, del mismo tipo madre que v6 pero más leve.** Gates de magnitud (MASS_REAL_MIN, DENS_MIN) sistemáticamente decorativos; el veredicto real depende de un único ratio-vs-shuffle nunca barrido, al borde del umbral en varias semillas (v4: 2/10 a <3%). No hay circularidad tipo v6, pero sí el mismo patrón de fondo: "el número que se muestra (masa en miles) no es el que decide; el que decide es un umbral pequeño, nunca justificado, con poco margen." El kill-switch anti-fuga de v1 se abandonó silenciosamente desde v3.
- **v5: limpia de la circularidad de v6.** Es la versión donde el gate de masa todavía filtra de verdad (discrepa de "lineage_ok" en 4/10 semillas) y por eso falla honestamente (rate=0.40, reportado sin ajustar el umbral).
- **v6: la circularidad ya documentada se confirma y se le suma fragilidad demostrada por barrido en vivo** — el umbral COMEM_VS_SHUFFLE_MIN=1.15 nunca fue derivado ni barrido, y bajarlo 50% invierte el veredicto de las semillas más cercanas al margen.
- **motor_1a7/pipeline.py: dos bugs de orquestación nuevos** (no reportados en el HALLAZGO original): el stage "5_orden_sin_masa" es estructuralmente incapaz de fallar (ambas ramas de `_fill_v5_from_json` asignan `True`), y los stages "3_estiramiento"/"4_densidad" no leen realmente los flags individuales de TEST_RHO (`d.get("flags",{})` busca en el nivel equivocado del JSON) — ambos colapsan a un único substring-match sobre el veredicto global. Hoy no cambian el resultado final porque los datos subyacentes son todos True, pero son gates que **no están haciendo el trabajo que su nombre y estructura de código sugieren**.

**Respuesta a la pregunta de fondo:** el resto del stack (fuera de etapa 7/v6, ya adjudicado) **NO está limpio** — hay deuda real y confirmada en v3/v4 (mismo patrón madre de "gate decorativo + único ratio frágil no barrido" que en v6, aunque sin la circularidad exacta), en TEST_RHO_DISPERSION (ausencia total de barrido/robustez), y dos bugs de orquestación en motor_1a7/pipeline.py que hacen que dos de las siete etapas de la cadena no verifiquen realmente lo que dicen verificar. v1, v2, v5 y cs074_rcruz son los tramos más sólidos: reportan sus propios fallos sin maquillar y sus barridos declarados son genuinos.
