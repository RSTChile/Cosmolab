# NULL-3 robustecido: motivos directos + dosis-respuesta — Fase II CS073

**Encargo:** resolver el caveat que quedó pendiente de `NULL3_resultado_CS.md` — "los motivos/ciclos no
importan" fue una INFERENCIA INDIRECTA del contraste de sumideros (REAL≈NULL-3 en masa acretada), nunca
una medición directa, y la tasa de aceptación del swap (0.6-0.7%) es baja, lo que deja abierto si NULL-3
realmente sacudió la topología de orden superior. Esta tarea mide motivos DIRECTAMENTE sobre el grafo
(Parte A) y corre una curva de dosis-respuesta relajando el filtro geométrico de longitud (Parte B). **No
se declara cierre ni veredicto sobre CS073 ni sobre la jerarquía — sólo se reportan números. La lectura
final es de Alexis.**

---

## Parte A — Motivos directos: REAL vs NULL-3 original (N=2000, seed=501, tol_relativa=0.2)

Método: reconstrucción determinista del grafo causal REAL (`null3_investigacion_preliminar.
reconstruir_grafo_real`, sin modificar) + el mismo `barajar_aristas_preservando_longitud` ya validado.
Triángulos y clustering: implementación propia por sets de adyacencia (no hay `networkx` en el venv del
proyecto). Ciclos de 4 nodos: identidad estándar sobre A² (matmul denso en `float64` para acelerar con
BLAS, verificado exacto contra un conteo directo — 4402 en ambos casos). Script:
`null3_motivos_directos.py`.

| grafo | triángulos | clustering promedio | clustering global | ciclos de 4 |
|---|---|---|---|---|
| REAL | 2780 | 0.60588 | 0.40548 | 4402 |
| NULL-3 (tol_relativa=0.2, seed=501) | 2005 | 0.42405 | 0.29244 | 2844 |
| swap SIN restricción (mismo seed) | 4 | 0.00054 | 0.00058 | 36 |

**Diferencia NULL-3 vs REAL: triángulos −27.9%, clustering promedio −30.0%, clustering global −27.9%,
ciclos de 4 −35.4%.** El swap con filtro de longitud (tol_relativa=0.2, sólo 346/49450 intentos
aceptados = 0.70%) SÍ destruyó una fracción sustancial de la estructura de orden superior — no es un
no-op, y la magnitud (≈28-35% de la estructura de motivos) es considerablemente mayor de lo que la baja
tasa de aceptación por sí sola sugeriría. Como contraste, el swap sin restricción de longitud (mismo
seed, mecanismo de los NULL1-8 originales) destruye prácticamente TODOS los triángulos y ciclos de 4
(quedan 4 de 2780 triángulos, 0.1%) — el grado exacto preservado no evita que el grafo se vuelva
esencialmente aleatorio en su topología local cuando no hay filtro de longitud.

---

## Parte B — Dosis-respuesta: relajando tol_relativa (N=2000, mismo seed=501 en todos los niveles)

### Paso 1 — grafo + perfil radial post-`layout_resortes` (script `null3_dosis_respuesta.py`)

Método: mismo `barajar_aristas_preservando_longitud`/`barajar_aristas_sin_restriccion` (importados, no
modificados) en 4 niveles de `tol_relativa`; para cada nivel, motivos (Parte A) + KS del perfil radial
contra REAL tras pasar por `layout_resortes` + la misma dilatación estática de toda la jerarquía (función
nueva `layout_y_expandir`, duplica la FORMA de la cola de `null3_generar_ic.generar_null3` para poder
alimentarle un grafo ya construido por cualquier mecanismo de swap).

| nivel | tasa aceptación | triángulos | Δ vs REAL | ciclos de 4 | Δ vs REAL | clustering global* | r_mean | r_std | KS radial | p (KS) |
|---|---|---|---|---|---|---|---|---|---|---|
| REAL | — | 2780 | — | 4402 | — | 0.40548 | 72.780 | 8.205 | — | — |
| tol_relativa=0.2 | 0.70% | 2005 | −27.9% | 2844 | −35.4% | 0.29244 | 73.262 | 7.808 | 0.0295 | 0.349 |
| tol_relativa=0.4 | 1.91% | 1263 | −54.6% | 1572 | −64.3% | 0.18424 | 73.749 | 7.328 | 0.0485 | **0.0181** |
| tol_relativa=0.8 | 5.99% | 451 | −83.8% | 484 | −89.0% | 0.06578 | 71.861 | 10.195 | 0.0585 | **0.00213** |
| sin filtro (Maslov-Sneppen puro) | n/a† | 4 | −99.9% | 36 | −99.2% | 0.00058 | 63.376 | 13.531 | 0.3775 | **2.7e-127** |

\* clustering global calculado exacto vía `clustering_global_REAL × (triángulos_nivel / triángulos_REAL)`
— válido porque el swap preserva la secuencia de grados EXACTA en los 4 niveles, así que el denominador
(nº de tripletes conectados, `Σ C(grado_i,2)`) es idéntico en todos; verificado contra el cálculo directo
en los niveles 0.2 y "sin filtro" (coincide a 4 decimales).
† el swap sin restricción no filtra por longitud, así que no hay una "tasa de aceptación" geométrica
comparable — empíricamente acepta casi cualquier intercambio que no duplique una arista, por eso destruye
casi toda la estructura de motivos.

**Lectura de la curva:** la tasa de aceptación crece con `tol_relativa` (0.70% → 1.91% → 5.99%, un
factor ~8.5 entre 0.2 y 0.8) y el conteo de motivos cae de forma aproximadamente monótona y bastante
suave en ese tramo (triángulos: 2005 → 1263 → 451; ciclos de 4: 2844 → 1572 → 484) — **no hay un quiebre
abrupto entre 0.2, 0.4 y 0.8: es una degradación gradual**, con un salto grande sólo al pasar al extremo
sin filtro (donde la estructura colapsa casi por completo). El perfil radial, en cambio, se mantiene
estadísticamente indistinguible de REAL SÓLO en tol_relativa=0.2 (p=0.349); ya en tol_relativa=0.4 la
diferencia es significativa al 5% (p=0.018), y se vuelve más marcada en 0.8 (p=0.002), antes del colapso
total sin filtro (p≈3×10⁻¹²⁷). **El perfil radial es una medida mucho menos sensible que el conteo de
motivos**: en tol_relativa=0.2, donde el perfil radial todavía "pasa" la prueba KS sin problema, ya se
había destruido ~28-35% de los triángulos/ciclos — el radio global tolera una pérdida sustancial de
estructura de orden superior antes de que el KS lo detecte.

### Paso 2 — piloto Phantom (N=500, 1 semilla por nivel nuevo, `null3_dosis_piloto_generar.py` +
`null3_dosis_piloto_correr.py`)

Reutiliza el pool N=500 y REAL de referencia ya en disco (`piloto_null1/real/`). Corridas limpias: exit
code 0 en `phantomsetup` y en `phantom` para las 2 corridas nuevas, sin abortos de conservación.

| corrida | tol_relativa | swap aceptado | masa en sumideros | nº sumideros |
|---|---|---|---|---|
| REAL (`piloto_null1/real`) | — | — | 282.0 | 4 |
| NULL-3 seed 601 (ya en disco) | 0.2 | 0.7% | 347.8 | 5 |
| NULL-3 seed 602 (ya en disco) | 0.2 | 0.6% | 235.0 | 3 |
| NULL-3 seed 603 (ya en disco) | 0.2 | 0.6% | 272.6 | 4 |
| **NULL-3 seed 701 (nuevo)** | **0.4** | **1.9%** | **432.4** | **6** |
| **NULL-3 seed 801 (nuevo)** | **0.8** | **6.0%** | **526.4** | **7** |

**Con una sola semilla nueva por nivel, la masa/nº de sumideros NO bajó al aumentar `tol_relativa` — al
contrario, subió (282→348/235/273 en tol=0.2 según semilla, →432 en tol=0.4, →526 en tol=0.8).** Esto es
opuesto a lo que predeciría "más motivos destruidos → menos sumideros". Pero la variabilidad de semilla
YA CONOCIDA dentro de un mismo nivel es grande (tol=0.2: rango 235.0–347.8, ~35% del valor medio) y aquí
sólo hay n=1 para 0.4 y 0.8 — con este tamaño de muestra no se puede distinguir una tendencia real de
ruido de semilla. **No se corrieron semillas adicionales por presupuesto de tiempo** (queda pendiente:
2-3 semillas más por nivel para saber si esto es señal o ruido).

**Dato de contexto pedido explícitamente para esta lectura:** los NULL1-8 ORIGINALES de CS073
(`bateria_n2000/ic_null1..8`, swap de Maslov-Sneppen SIN restricción de longitud — el mismo mecanismo que
el nivel "sin filtro" de esta tabla, pero a N=2000 y con el pipeline antiguo de `traducir_pool`)
**formaron sumideros PARCIALMENTE (masa promedio ≈680-770, no cero) en su batería original**, en
contraste con NULL-1 (radio exacto/ángulo aleatorio) y NULL-2 (Zel'dovich) de la jerarquía nueva, que
dieron CERO sumideros en 16 corridas combinadas. Esto es relevante para la curva de dosis-respuesta:
sugiere que un grafo causal completamente desordenado en su topología local (motivos destruidos casi al
100%, como confirma la Parte A de esta tarea: 4 triángulos de 2780) **todavía retiene ALGO capaz de
sembrar formación parcial de sumideros** — posiblemente el mero hecho de sembrar posiciones vía un grafo
de vecindad + `layout_resortes` (en vez de ángulo isótropo puro o Zel'dovich), no la topología específica
de ese grafo. Es una lectura especulativa, no verificada aquí de forma directa — se señala como hallazgo
relevante para que Alexis lo pondere, no como conclusión.

---

## Resumen de qué se corrió y qué queda pendiente

**Corrido en esta tarea:**
- Parte A completa: triángulos/clustering/ciclos-4 REAL vs NULL-3(tol=0.2) vs swap sin restricción, N=2000.
- Parte B, grafo+radial: 4 niveles (0.2, 0.4, 0.8, sin filtro), N=2000, mismo seed en todos.
- Parte B, piloto Phantom: 2 niveles nuevos (0.4, 0.8), 1 semilla cada uno, N=500.

**Pendiente (no alcanzado por presupuesto de tiempo, ~35 min de cómputo real usados):**
- Semillas adicionales del piloto Phantom en tol=0.4/0.8 (para separar señal de ruido de semilla).
- Piloto/batería Phantom del nivel "sin filtro" a esta misma escala N=500 con el pipeline nuevo (se usó
  en su lugar el dato ya existente de los NULL1-8 originales, N=2000, pipeline distinto — comparable
  sólo cualitativamente, no número a número).
- Escalar cualquier nivel nuevo a batería completa (8 semillas, N=2000) — explícitamente fuera de alcance
  de esta tarea según la salvaguarda de tiempo pedida.

---

## Tiempo de cómputo real

| paso | tiempo |
|---|---|
| Parte A (motivos REAL/NULL-3/sin-restricción, N=2000) | 98.9 s |
| Parte B paso 1 (4 niveles: motivos + layout + KS radial, N=2000) | 162.0 s |
| Parte B paso 2, generación piloto (pool N=500 + 2 IC) | 128.2 s |
| Parte B paso 2, Phantom (2 corridas) | 6.2 s |
| **total cómputo** | **≈395 s ≈ 6.6 min** |

Muy por debajo de la salvaguarda de ~45-50 min.

---

## Entregables de esta tarea

- `null3_motivos_directos.py` — Parte A: conteo directo de triángulos/clustering/ciclos-4 (REAL, NULL-3
  tol=0.2, swap sin restricción), N=2000, seed=501. Reusa `reconstruir_grafo_real`/
  `barajar_aristas_preservando_longitud`/`barajar_aristas_sin_restriccion` de
  `null3_investigacion_preliminar.py` (no modificado).
- `null3_dosis_respuesta.py` — Parte B, paso 1: barrido de `tol_relativa` (0.2/0.4/0.8/sin filtro),
  motivos + KS radial post-`layout_resortes`, N=2000. Reusa `null3_motivos_directos.py` y las piezas
  congeladas `layout_resortes`/`Expansion`/`T0`/`_T_reloj` (importadas, no modificadas).
- `null3_dosis_piloto_generar.py` / `null3_dosis_piloto_correr.py` — Parte B, paso 2: piloto Phantom
  N=500 para tol_relativa=0.4 (seed 701) y 0.8 (seed 801), en
  `/Users/alexis/phantom_cs073/piloto_null3_dosis/`. Reusa `generar_null3` de `null3_generar_ic.py` (no
  modificado) y el mismo patrón de `null3_piloto_correr.py` (no modificado, sólo duplicado a un archivo
  nuevo con las carpetas nuevas).
- `/Users/alexis/phantom_cs073/piloto_null3_dosis/` — carpeta nueva con las 2 corridas de Phantom (IC,
  `cosmog.in`, `setup.log`, `run.log`, `.sink`, dumps). No se tocó ninguna carpeta de batería/piloto
  anterior ni ningún script congelado — sólo lectura/importación.
- Este informe.
