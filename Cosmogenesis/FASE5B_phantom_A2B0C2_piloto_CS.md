# Fase V-B — piloto Phantom sobre A2-B0-C2 (pares emparejados Clase I vs Clase III)

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude) · Sigue de `FASE5A_profundizar_A2B0C2_resultado_CS.md`
(el candidato A2-B0-C2, n=30, patrón bimodal 43.3% Clase I / 50.0% Clase III).

**Alexis autorizó explícitamente correr Phantom para esta línea de investigación** (10-ago-2026, "Sí,
corremos Phantom") — primera vez que se corre este pipeline concreto (grafo A2-B0-C2 → Phantom). Se
siguió la disciplina piloto-primero: correr 1 par, MEDIR el costo real, y sólo entonces decidir si
escalar. El costo medido resultó muy bajo (ver §2) — se escaló a los 3 pares completos dentro del mismo
presupuesto de tiempo.

No se declara cierre ni veredicto sobre si A2-B0-C2 "confirma" o "refuta" nada. Se reportan números
crudos; la lectura final es de Alexis. Ningún script congelado fue modificado
(`cs090_fase5_generador.py`, `cs090_fase5_motor.py`, `cs090_fase5_clasificador.py`,
`grafo_random_layout_generar_ic_masa_fija.py`, `real_extra_generar_ic.py`, `leer_volcado_phantom.py`).
No se hicieron commits de git.

## 0. En simple, con analogía

Fase V-A encontró que la "regla del juego" A2-B0-C2 (una red de relaciones que se recablea sola, sin
memoria de segundo orden, con un tope duro de cuántos vecinos puede tener cada nodo) produce, según la
tirada de parámetros, dos destinos muy distintos: la red se "aplasta" en un grupo compacto (Clase I,
disolución) o se "extiende" en una geometría amplia (Clase III). Fase V-A midió esto con matemática
pura de grafos (diámetro, componente gigante) — nunca le puso masa ni gravedad al grafo.

Esta tarea (Fase V-B) es la primera vez que a esas dos "tiradas" (una de cada clase, emparejadas para
que se parezcan en todo lo demás) se les pone MASA real y se las suelta en Phantom, el simulador de
gravedad de partículas que ya usó CS073 — como poner dos maquetas de alambre (una compacta, una
extendida) hechas con las MISMAS reglas de recableo, y ver si al llenarlas de arena y dejar que la
gravedad actúe, la arena se acumula distinto según la forma de la maqueta de origen.

## 1. Paso 1 — el adaptador construido

Archivo nuevo: **`cs090_fase5b_phantom_adaptador.py`**. Dos piezas, cada una reusa código ya validado
sin tocarlo:

- **`reconstruir_regla_a2b0c2(seed, N=2000, n_sweeps=14)`** — reconstruye BIT A BIT el grafo final de
  una regla A2-B0-C2 ya clasificada en `cs090_fase5_profundizar_a2b0c2_resumen.csv`, usando SÓLO su
  columna `seed` (`cs090_fase5_generador.generar_regla` sólo depende de `seed`, no del `idx` — el rng
  se construye como `np.random.default_rng(seed)`, verificado por inspección). Reproduce exactamente lo
  que hizo `cs090_fase5_motor.correr_regla_coarse()` (mismo `p`, mismo rng derivado de `seed*5000+N`,
  mismo `construir_A2`+`dinamica_B0`+`medir`). Verificado con un smoke test: la regla reconstruida con
  `seed=272702` (r9) dio K=5, J=0.493, noise=0.244, meandeg=4.25, kcap=7 — coincide EXACTO con la fila
  del CSV resumen.
- **`generar_ic_masa_fija_desde_grafo(...)`** — misma receta de MASA TOTAL FIJA
  (masa_particula=18800/N, lado de caja fijo=2000^(1/3), no depende de N) que
  `grafo_random_layout_generar_ic_masa_fija.generar_control_random_masa_fija` (congelado, sólo se
  reusan sus constantes/decisiones), pero en vez de generar un grafo Erdős-Rényi nuevo, recibe el grafo
  YA CONSTRUIDO por la pieza anterior (el grafo dinámico co-emergente A2-B0-C2, ya clasificado
  Clase I o Clase III). `layout_resortes` (mismo módulo congelado) se aplica sobre ese grafo, con la
  misma dilatación estática (`Expansion`, 60 pasos) y el mismo campo de turbulencia (Mach=3, seed=42)
  que toda la jerarquía CS073.

Archivo `cs090_fase5b_generar_pares.py` orquesta ambas piezas para las 6 reglas de los 3 pares elegidos
(§2) y escribe cada `cosmogenesis_ic.txt` en
`/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto/<rule_id>_<clase>/`.

## 2. Paso 2 — los 3 pares elegidos, y por qué

De las 20 reglas nuevas de `cs090_fase5_profundizar_a2b0c2_resumen.csv` se excluyen r0 (clase
"intermedio") y r15 (Clase II) — quedan 18 utilizables (8 Clase I, 10 Clase III). El Objetivo 2 de
`FASE5A_profundizar_A2B0C2_resultado_CS.md` ya había mostrado que NINGÚN parámetro separa limpiamente
Clase I de Clase III (solape total de rangos) — así que "emparejar" significa buscar, a mano, la
combinación con menor distancia en K/J/noise/meandeg/kcap entre una regla I y una III, sin reusar
ninguna regla dos veces:

| par | regla (clase) | K | J | noise | meandeg | kcap | seed |
|---|---|---|---|---|---|---|---|
| **A** | A2-B0-C2-r9 (I) | 5 | 0.493 | 0.244 | 4.25 | 7 | 272702 |
| **A** | A2-B0-C2-r19 (III) | 5 | 0.341 | 0.326 | 4.02 | 7 | 273672 |
| **B** | A2-B0-C2-r1 (I) | 5 | 0.659 | 0.234 | 7.63 | 6 | 271926 |
| **B** | A2-B0-C2-r17 (III) | 5 | 0.475 | 0.211 | 7.35 | 5 | 273478 |
| **C** | A2-B0-C2-r6 (I) | 7 | 0.632 | 0.119 | 5.38 | 5 | 272411 |
| **C** | A2-B0-C2-r14 (III) | 8 | 0.567 | 0.198 | 5.76 | 5 | 273187 |

PAR A es el más cercano de los 18 (K y kcap coinciden exactamente, meandeg casi idéntico: 4.25 vs 4.02).
PAR B comparte K y tiene meandeg/noise muy cercanos. PAR C comparte kcap y tiene J/noise/meandeg
cercanos, con K difiriendo en 1. `seed_layout=12345` fijo en las 6 reglas — la única diferencia entre
las dos reglas de un mismo par es el grafo de entrada (ya determinado por su propia dinámica A2-B0-C2),
no la realización del layout. N=2000 para las 6 (piso de resolución SPH válido, lección de CS073 —
`MISTERIO_N500_vs_N2000_CS.md`: N<1000 no es confiable).

## 3. Paso 3 — el piloto de 1 par, y el costo real medido

Se corrió Phantom PRIMERO sólo sobre PAR A (2 corridas), como pide la disciplina piloto-primero, con
los mismos parámetros de Phantom que toda la jerarquía CS073 (`rho_crit_cgs=1000`, `icreate_sinks=1`,
`r_crit=0.6`, `h_acc=0.3`, `tmax=0.500`, `dtmax=0.001`):

| etapa | tiempo medido |
|---|---|
| `phantomsetup_cosmogenesis_backup` (setup, por corrida) | 1.3-1.4 s |
| `phantom_cosmogenesis_backup` (corrida física, por corrida) | 14.4-15.0 s |
| **Total piloto (2 corridas, PAR A)** | **32.2 s** |

**Costo real: ~15-16 s por corrida completa de Phantom (setup+run), muy por debajo del umbral de
alarma (>20-30 min) y comparable al costo YA CONOCIDO de CS073 con masa fija a N=2000**
(`bateria_grafo_random_masa_fija/ic_masaFija_N2000_s1`: 9.25 s de wall time; `bateria_n2000/ic_real`
original: 31.45 s). El paso más caro del pipeline completo NO es Phantom — es la reconstrucción del
grafo + `layout_resortes` en Python (N=2000, Fruchterman-Reingold O(N²) por 100 iteraciones): 75-85 s
por regla. Con Phantom confirmado barato, se decidió **escalar de inmediato a los 3 pares completos**
(criterio del Paso 4: "si el costo es razonable, escalar").

**Costo total real de las 6 corridas de Phantom (los 3 pares completos): ~87 s de wall time de Phantom**
(6 × ~14.5 s), más ~8.2 min de generación de condiciones iniciales en Python (6 × ~85 s, dominado por
`layout_resortes`) — **~10 minutos de principio a fin para el piloto completo de 3 pares**, dentro del
presupuesto de la tarea sin necesidad de recortar a 1 solo par.

## 4. Resultados — las 6 corridas, número crudo por regla

Todas las 6 corridas alcanzaron `tmax=0.500` (`cosmog_00500`, 500 dumps) sin abortar
(`I_WILL_NOT_PUBLISH_CRAP` NO hizo falta, a diferencia de los pilotos fallidos de N=500/N=1000 masa fija
documentados en `bateria_grafo_random_masa_fija` — las 6 corridas de A2-B0-C2 a N=2000 fueron estables).

| regla | clase | par | masa total (IC) | masa en sumideros (final) | fracción en sumideros | n sumideros | t primer sumidero | κ_V agregado | κ_V medio (válidos) |
|---|---|---|---|---|---|---|---|---|---|
| r9  | I   | A | 18800.0 | 1513.4 | **0.0805** | 8 | 0.046 | **0.386** | 0.382 |
| r19 | III | A | 18800.0 | 1635.6 | **0.0870** | 8 | 0.038 | **0.436** | 0.445 |
| r1  | I   | B | 18800.0 | 1635.6 | **0.0870** | 8 | 0.038 | **0.414** | 0.425 |
| r17 | III | B | 18800.0 | 2274.8 | **0.1210** | 8 | 0.030 | **0.797** | 0.795 |
| r6  | I   | C | 18800.0 | 1889.4 | **0.1005** | 8 | 0.039 | **0.524** | 0.518 |
| r14 | III | C | 18800.0 | 1861.2 | **0.0990** | 8 | 0.042 | **0.515** | 0.508 |

(masa acretada total leída independientemente del `.sink` incremental coincide con la masa de sumideros
del dump binario final en las 6 filas, a redondeo — chequeo cruzado consistente. `n_kappa_indefinidos=0`
en las 6: ningún sumidero tuvo masa_primer_tercio=0, la razón κ_V está bien definida en los 24
sumideros combinados.)

## 5. Comparación honesta, par por par (números, sin veredicto)

| par | clase III − clase I (fracción en sumideros) | clase III − clase I (κ_V agregado) | dirección |
|---|---|---|---|
| A (r19−r9) | 0.0870 − 0.0805 = **+0.0065** | 0.436 − 0.386 = **+0.050** | III levemente mayor en ambas métricas |
| B (r17−r1) | 0.1210 − 0.0870 = **+0.0340** | 0.797 − 0.414 = **+0.383** | III claramente mayor en ambas métricas (κ_V casi el doble) |
| C (r14−r6) | 0.0990 − 0.1005 = **−0.0015** | 0.515 − 0.524 = **−0.009** | prácticamente empatado, I marginalmente mayor en ambas |

**Lectura honesta, sin cerrar nada:** en 2 de los 3 pares (A y B) la regla Clase III terminó con más
masa en sumideros y κ_V más alto que su pareja Clase I; en el par C (el que más difiere en K, 7 vs 8,
de los tres) el resultado está prácticamente empatado, con la Clase I levemente por delante. El efecto
más grande, por lejos, es el del par B (κ_V casi el doble: 0.797 vs 0.414) — pero es también el par con
mayor Δkcap (6 vs 5) entre los tres, así que no se puede aislar todavía si el efecto viene de la
clase (I/III) o de esa diferencia residual de kcap. El tiempo al primer sumidero (0.030-0.046, todas
las 6 corridas) NO muestra ninguna separación por clase — la formación temprana de sumideros parece
insensible a si el grafo de origen fue Clase I o III. n_sumideros=8 EN LAS 6 CORRIDAS, sin ninguna
excepción — la cantidad de sumideros que se forman no distingue nada aquí (mismo patrón "8 sumideros"
que venía dando `bateria_n2000` en corridas anteriores de CS073, sugiere que el número de sumideros
puede estar más determinado por la física común del pipeline —masa/caja/turbulencia fijas— que por el
grafo de origen). Con n=3 pares, ningún patrón de estos alcanza para hablar de tendencia robusta — es
la primera medición física de este candidato, no una batería.

## 6. Qué falta / recomendación de escala

El piloto de 3 pares se completó dentro del presupuesto (no fue necesario detenerse en 1 solo par: el
costo de Phantom resultó bajo desde la primera medición). Si Alexis decide escalar más allá de este
piloto:

- **Más pares**: cada regla adicional cuesta ~85 s de generación de IC (dominado por `layout_resortes`
  en Python, no por Phantom) + ~15 s de Phantom — **~100 s por regla, ~200 s (~3.3 min) por par
  adicional**. Escalar a, por ejemplo, 10 pares (20 corridas) costaría del orden de ~35 minutos, factible
  en una sola sesión.
- **Semillas de layout adicionales por regla** (como hizo `real_extra_generar_ic.py` con
  `seed_layout` para CS073): mismo costo por corrida (~100 s), permitiría separar "efecto de la clase"
  de "ruido de la realización del layout" — actualmente cada regla del piloto tiene una sola
  realización espacial.
- El **par C** (el más ambiguo, Δkcap=0 pero ΔK=1) sugiere que valdría la pena, si se escala, ELEGIR
  explícitamente algunos pares con kcap idéntico Y K idéntico (no sólo minimizar distancia combinada)
  para aislar mejor el efecto de clase de los parámetros residuales.

## 7. Archivos de esta tarea

- `cs090_fase5b_phantom_adaptador.py` — reconstrucción del grafo A2-B0-C2 + generación de IC de masa fija.
- `cs090_fase5b_generar_pares.py` — selección de los 3 pares + orquestación de generación de IC (6 reglas).
- `cs090_fase5b_correr.py` — corre Phantom sobre las carpetas generadas, mide wall time, salvaguarda de
  20 min por corrida, no recomputa si ya existe `cosmog_00500`.
- `cs090_fase5b_analizar.py` — extrae métricas (fracción en sumideros, tiempo al primer sumidero, masa
  acretada, κ_V) de los dumps binarios (`leer_volcado_phantom.py`) y de los `.sink`.
- `cs090_fase5b_metricas.csv` — las 6 filas de datos crudos (tabla completa de §4, con columnas
  adicionales: masa_gas_final, masa_sumideros_final, n_aristas_grafo_final, diam_grafo_final).
- `/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto/<rule_id>_<clase>/` (×6) — condiciones
  iniciales, `cosmog.in`, `setup.log`, `run.log`, dumps binarios `cosmog_00000..00500`, `cosmog01.sink`,
  `meta_regla.json` (parámetros de la regla + métricas del grafo antes de Phantom).
- `/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_piloto/pares_resumen.json` — resumen de las 6
  reglas generadas.
- Este informe.

Ningún script congelado fue modificado. No se declaró cierre ni veredicto sobre A2-B0-C2. No se
hicieron commits de git.
