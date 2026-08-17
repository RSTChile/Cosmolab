# Fase V-B — investigación del "8 sumideros reiterado" + escala de la batería de pares

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude) · Sigue de `FASE5B_phantom_A2B0C2_piloto_CS.md`.

Alexis vio que las 6 corridas del piloto dieron EXACTAMENTE 8 sumideros y dijo **"ese resultado de 8
reiterado es raro... me huele a fallo del instrumento"**. Esta tarea investiga esa sospecha (Parte 1)
ANTES de escalar la batería de pares (Parte 2), como se pidió. No se declara cierre ni veredicto sobre
A2-B0-C2 ni sobre si el "8" es fallo o física real — se reportan números, la lectura final es de
Alexis. Ningún script congelado del piloto anterior fue modificado
(`cs090_fase5b_phantom_adaptador.py`, `cs090_fase5b_generar_pares.py`, `cs090_fase5b_correr.py`,
`cs090_fase5b_analizar.py`). No se hicieron commits de git.

## 0. En simple, con analogía

Imaginá que tenés 8 moldes de gelatina idénticos (misma caja, misma cantidad de líquido) y que siempre,
sin importar qué fruta picada le pongas adentro (la fruta = el grafo de origen: malla causal REAL,
grafo random, A2-B0-C2 clase I o clase III), el líquido se cuaja en las mismas 8 bolsitas. La primera
sospecha razonable es "¿el molde tiene 8 compartimentos fijos por diseño?" (fallo de instrumento). Esta
tarea revisó los moldes (el código de Phantom) y no encontró ningún compartimento fijo de fábrica. Lo
que sí encontró es que el NÚMERO DE MOLDES cambia mucho si cambiás el TAMAÑO de la bandeja (la
resolución N: con más partículas hay muchas más bolsitas — 8 a N=2000, 29 a N=4000, 122 a N=8000, misma
masa total). Y cuando se agitó el líquido de formas distintas ANTES de cuajar (semillas de turbulencia
distintas, mismo grafo) las 8 bolsitas siguieron siendo 8, casi sin cambiar ni de tamaño. Así que el
"8" parece depender mucho de la bandeja (resolución) y muy poco de la fruta (grafo) o de cómo se agitó
(turbulencia) — pero el LÍQUIDO QUE TERMINÓ EN CADA BOLSITA (fracción de masa, κ_V) sí varía según la
fruta. Esa es la pista central: el conteo de sumideros parece "saturado" por la resolución; las métricas
continuas (fracción de masa, κ_V) siguen siendo las que distinguen algo real.

## PARTE 1 — investigación del "8 reiterado"

### 1.1 Tabla histórica: n_sumideros en TODAS las baterías con .sink en disco

Se escanearon las 64 carpetas de `/Users/alexis/phantom_cs073/` que tienen un archivo `.sink`
(cualquier N, cualquier tipo de grafo: REAL/malla causal, NULL-1 a NULL-5, grafo random, A2-B0-C2 I/III,
Sistema A/B de O-N7.7), contando sink IDs únicos (mismo método que
`cs090_fase5b_analizar.py::analizar_sink`). Agrupado por N (resolución):

| N (partículas) | n corridas | n_sumideros observados | rango | moda |
|---|---|---|---|---|
| 500 | 6 | 3, 4, 4, 5, 6, 7 | 3–7 | disperso |
| 1000 | 2 | 1, 3 | 1–3 | disperso |
| **2000** | **53** | **46× el valor 8; el resto: 4× el valor 7, 2× el valor 9, 1× el valor 10** | **7–10** | **8 (87% de las corridas)** |
| 4000 | 1 | 29 | — | — |
| 8000 | 1 | 122 | — | — |

**Dato clave 1 — escala con la resolución, a masa total fija:** las 3 corridas de `ON77_sistemaA_cierre`
(N=2000/4000/8000, MISMA masa total=18800, mismo grafo/método) dieron 8 → 29 → 122 sumideros. Esto es un
patrón de fragmentación numérica dependiente de resolución bien conocido en SPH (a más partículas
resolviendo la misma masa/caja, se resuelven más sitios de colapso independientes — no es exclusivo de
este proyecto). Este hallazgo por sí solo ya apunta a que la resolución (N), no el grafo de origen, es
un actor dominante en el CONTEO de sumideros.

**Dato clave 2 — a N=2000 fijo, el "8" NO es absolutamente rígido, pero sí muy angosto:** de 53 corridas
históricas a N=2000 (que incluyen: REAL, NULL-1..NULL-5, grafo random ×10 semillas, A2-B0-C2 clase I y
III ×6, Sistema-B de O-N7.7 con 5 valores de H, campo de velocidad heredado vs. turbulento), el 87% dio
exactamente 8, y el resto dio 7, 9 o (una vez) 10 — nunca fuera del rango 7-10. Es decir: el piloto de 6
corridas de Fase V-B dando "8 en las 6, sin excepción" fue, mirando el historial completo, más una
COINCIDENCIA DENTRO DE UNA DISTRIBUCIÓN MUY ANGOSTA que un valor literalmente fijo por config — pero la
distribución sigue siendo sospechosamente angosta para lo distinto que son esos grafos de entrada.

CSV completo: `cs090_fase5b_historico_n_sumideros_RAW.csv` (64 filas, todas las baterías/carpetas
escaneadas, bandera de error si alguna no se pudo leer).

### 1.2 ¿El seed de turbulencia (Mach=3, seed=42) es literalmente el mismo en TODO el pipeline?

**Sí, confirmado por grep exhaustivo.** `TURB_SEED = 42` aparece hardcodeado, sin excepción, en los 21
generadores de condiciones iniciales del proyecto (`null1_bateria_generar.py`,
`null2_bateria_generar.py`, ..., `null5_bateria_generar.py`, `grafo_random_bateria_generar.py`,
`grafo_random_masa_fija_generar.py`, `real_extra_generar_ic.py`, `ON77_sistemaA/B_*.py`,
`cs090_fase5b_phantom_adaptador.py`, etc.) — **nunca se varió en la historia del proyecto**, hasta el
test barato de esta tarea (§1.4). Esto confirma que el sospechoso #1 de la tarea (turbulencia siempre
igual) es real como HECHO — la pregunta es si ese hecho EXPLICA el "8".

### 1.3 ¿Hay un tope duro de sumideros en la configuración (sospechoso #2)?

**No se encontró ningún parámetro de tope en `cosmog.in`.** El bloque de sumideros que Phantom usa
(`icreate_sinks=1`, `rho_crit_cgs=1000`, `r_crit=0.6`, `h_acc=0.3`, `f_acc=0.8`) sólo tiene UMBRALES
FÍSICOS (densidad/radio) para decidir CUÁNDO nace un sumidero — Phantom no tiene ningún parámetro de
`.in` que limite CUÁNTOS sumideros puede haber. Se revisó también el código fuente de Phantom
(`config.F90`): existe un límite de COMPILACIÓN, `maxptmass = 1000` (arreglo estático), pero está muy
por encima de los 7-10 observados — no es el techo que se está tocando acá. **El sospechoso #2 (tope de
config) queda descartado como explicación del "8".**

### 1.4 Test barato de confirmación — MISMO grafo, turbulencia distinta

Se tomó el grafo YA generado de la regla `A2-B0-C2-r9` (Clase I, piloto original) y se generaron 2
condiciones iniciales NUEVAS con `cs090_fase5b_phantom_adaptador.generar_ic_masa_fija_desde_grafo`
(reusado, sin modificar — el parámetro `turb_seed` ya existía en su firma), cambiando SÓLO la semilla de
turbulencia (`seed=7` y `seed=99`), manteniendo TODO lo demás idéntico (mismo grafo, mismo
`seed_layout=12345`, mismo N=2000, misma masa fija). Script nuevo:
`cs090_fase5b_test_turbulencia.py`.

| turb_seed | n_sumideros | fracción masa en sumideros | κ_V agregado | t primer sumidero |
|---|---|---|---|---|
| 42 (original) | 8 | 0.0805 | 0.386 | 0.046 |
| 7 (nuevo) | 8 | 0.0805 | 0.328 | 0.046 |
| 99 (nuevo) | 8 | 0.0795 | 0.345 | 0.046 |

**Resultado — el más informativo de la Parte 1: cambiar la semilla de turbulencia NO cambió nada del
conteo, casi nada de la fracción de masa, y NADA del tiempo al primer sumidero (0.046 en las 3, igual
hasta el tercer decimal).** Esto es la evidencia MÁS DIRECTA contra la hipótesis "la turbulencia
domina" — si la turbulencia dominara la fragmentación, cambiar su semilla debería mover el número/tiempo
de colapso de forma apreciable, y no lo hizo. Sólo κ_V (que mide la FORMA de la curva de acreción a lo
largo de la vida del sumidero, no el conteo) mostró algo de variación (0.328-0.386, ~15%).

### 1.5 Conclusión honesta de la Parte 1 (sin cerrar nada)

Tres piezas de evidencia, combinadas:

1. **La resolución (N) domina el conteo de sumideros con un margen enorme** (8→29→122 al subir N a masa
   fija) — esto es un patrón de fragmentación numérica dependiente de resolución, conocido en SPH, no
   inventado para este proyecto.
2. **A N=2000 fijo, el "8" es una moda muy fuerte (87%) pero NO absolutamente rígida** (rango real
   observado: 7-10, en 53 corridas de grafos MUY distintos entre sí) — descarta un tope de config
   literal (§1.3 ya lo había descartado por inspección de código).
3. **Cambiar la semilla de turbulencia, con el MISMO grafo y la MISMA resolución, no movió el conteo ni
   el tiempo al primer sumidero** (§1.4) — esto pesa en contra de "la turbulencia es la que fija el 8",
   al menos para esta prueba puntual (1 grafo, 2 semillas alternativas).

**Lectura honesta, sin forzar:** la evidencia apunta más a que **la resolución (N=2000, caja y masa
fijas) satura el NÚMERO de sitios de colapso resolvibles**, y que ni el grafo de origen ni la
semilla de turbulencia mueven mucho esa cuenta en este régimen — no porque haya un tope de config, sino
porque la física numérica a esta resolución concreta converge a un puñado angosto de sitios (7-10)
casi sin importar qué se varíe. Esto NO invalida las métricas continuas (fracción de masa en sumideros,
κ_V), que sí variaron con el grafo de origen en el piloto original (0.08-0.12 en fracción, 0.39-0.80 en
κ_V) y con la turbulencia en este test (κ_V 0.33-0.39) — son ellas las que parecen cargar la información
que el conteo de sumideros ya no puede cargar a esta resolución. No se puede, con esta evidencia sola,
descartar del todo que a OTRA resolución o con OTRO grafo mucho más disímil el conteo sí discrimine algo
— sólo se probó 1 grafo × 2 semillas alternativas, no una grilla completa. La lectura final es de
Alexis.

## PARTE 2 — escalar la batería de pares (Clase III vs Clase I, kcap Y K idénticos)

### 2.1 Selección de los pares — y un bug real encontrado y corregido en el camino

De las 18 reglas ya clasificadas en `cs090_fase5_profundizar_a2b0c2_resumen.csv` (excluyendo r0
intermedio y r15 Clase II), sólo **2 pares nuevos** tienen K Y kcap EXACTAMENTE iguales (verificado
exhaustivamente sobre las 18 filas): `r6(I) vs r2(III)` (K=7,kcap=5) y `r12(I) vs r19(III)` (K=5,kcap=7)
— además del par `r9 vs r19` que YA estaba en el piloto original (K=5,kcap=7, el más cercano de los 3
pares del piloto). Para llegar a más pares con este criterio estricto, se generaron 40 reglas NUEVAS
adicionales con el generador congelado (`cs090_fase5_generador.generar_reglas_clase`, mismos ejes
A2-B0-C2, mismo filtro P1-P5, `seed_base=371828` nuevo), corridas con el motor congelado
(`cs090_fase5_motor.correr_regla_coarse`, N=2000, n_sweeps=14) y clasificadas con el clasificador
congelado (`cs090_fase5_clasificador.clasificar_regla`) — las 40 pasaron el filtro P1-P5 (16 Clase I, 21
Clase III, 3 Clase II). Se hallaron 3 pares NUEVOS con K+kcap exacto entre estas 40. Script:
`cs090_fase5b_generar_pares_v2.py` (nuevo, no modifica ningún archivo congelado).

**Bug real, detectado y corregido dentro de esta misma tarea:** el generador de reglas nuevas reutiliza
la misma convención de nombre (`A2-B0-C2-r0` … `r39`), que COLISIONA con los nombres `r0`-`r19` ya
usados por las 18 reglas clasificadas anteriormente (dos reglas FÍSICAMENTE DISTINTAS, con distinto
`seed`, terminaron compartiendo la etiqueta `A2-B0-C2-r2`, `r9` y `r12`). Un diccionario de búsqueda de
semillas sobrescribió sin darse cuenta la semilla "vieja" con la "nueva" para esos 3 nombres repetidos,
lo que generó 3 condiciones iniciales de Phantom con el grafo EQUIVOCADO (no el que el emparejamiento
por K/kcap había calculado). Se detectó comparando cada `meta_regla.json` real contra el CSV de origen
(dos corridas resultaron ser, sin querer, la MISMA simulación con etiqueta de clase distinta — señal
inequívoca del bug), y se corrigió regenerando las 2 reglas afectadas que sí valía la pena salvar
(`r2` y `r12` de la lista ORIGINAL de 18, bajo nombres sin colisión: `r2v1fix`, `r12v1fix`) y
volviendo a correr Phantom sobre ellas. La 3ª pareja afectada (`r9` de la lista de 40 nuevas, que
colisionaba con el `r9` original ya corrido en el piloto) se dejó tal cual quedó ejecutada (con el `r9`
original del piloto, K=5 kcap=7) y se relabeled honestamente como **match cercano, no exacto**
(kcap difiere 7 vs 6) en vez de descartarla. Se documenta este bug explícitamente porque hubiera
producido un resultado "escalado" sutilmente equivocado si no se hubiera verificado dato por dato contra
`meta_regla.json`.

**Los 8 pares finales corridos en Phantom** (16 corridas, N=2000, masa fija=18800, mismos parámetros de
Phantom que toda la jerarquía CS073):

| par | regla I | regla III | K | kcap | ¿match exacto? |
|---|---|---|---|---|---|
| A (piloto) | r9 | r19 | 5 | 7 | **sí** |
| B (piloto) | r1 | r17 | 5 | 6 vs 5 | no (kcap Δ1) |
| C (piloto) | r6 | r14 | 7 vs 8 | 5 | no (K Δ1) |
| D (nuevo) | r2 | r20 | 5 | 6 | **sí** |
| E (nuevo) | r12 | r28 | 6 | 6 | **sí** |
| F (nuevo, fix bug) | r6 | r2v1fix | 7 | 5 | **sí** |
| G (nuevo, fix bug) | r12v1fix | r19 | 5 | 7 | **sí** |
| H (nuevo) | r9 | r39 | 5 | 7 vs 6 | no (kcap Δ1, por el bug de colisión) |

5 de 8 pares tienen match exacto en K Y kcap (subconjunto más limpio, el que pidió Alexis); 3 son
matches cercanos heredados del piloto original (documentados igual, no descartados).

### 2.2 Resultados crudos — 16 corridas

CSV completo: `cs090_fase5b_escala_v2_metricas_FINAL.csv`. Todas las 16 corridas alcanzaron
`tmax=0.500` sin abortar. n_sumideros=8 en las 16 (consistente con la Parte 1 — a esta resolución, el
conteo sigue sin distinguir nada).

| par | clase III − clase I (fracción en sumideros) | clase III − clase I (κ_V agregado) | dirección | exacto |
|---|---|---|---|---|
| A (r19−r9) | +0.0065 | +0.050 | III>I | sí |
| B (r17−r1) | +0.0340 | +0.383 | III>I | no |
| C (r14−r6) | −0.0015 | −0.009 | I>III | no |
| D (r20−r2) | +0.0105 | −0.102 | III>I | sí |
| E (r28−r12) | +0.0075 | +0.058 | III>I | sí |
| F (r2v1fix−r6) | +0.0210 | +0.293 | III>I | sí |
| G (r19−r12v1fix) | +0.0250 | +0.092 | III>I | sí |
| H (r39−r9) | −0.0020 | +0.086 | I>III (frac) / III>I (κ_V) | no |

### 2.3 Comparación agregada — n=8 pares (todos) vs. n=5 pares (sólo match exacto)

| subconjunto | n pares | media Δfracción (III−I) | III>I en fracción | media Δκ_V (III−I) | III>I en κ_V |
|---|---|---|---|---|---|
| **Todos (n=8)** | 8 | **+0.0126** | **6/8** | **+0.1066** | **6/8** |
| **Sólo match exacto K=kcap (n=5)** | 5 | **+0.0141** | **5/5** | **+0.0784** | **4/5** |

**Lectura honesta, sin cerrar nada:** con el n ampliado (8 pares en vez de 3) y priorizando el
subconjunto mejor emparejado (5 pares con K y kcap idénticos, el que pidió Alexis para no confundir
clase con parámetros residuales), la dirección **Clase III > Clase I** en fracción de masa acretada en
sumideros se sostiene en **5 de 5** de esos pares limpios (antes era 2 de 3, con el par más "sucio"
siendo justo el que iba en contra). En κ_V agregado la dirección III>I se sostiene en 4 de 5 pares
exactos — el único que va al revés (par D, r20 vs r2) tiene Δfracción positivo pero Δκ_V negativo, así
que ese par por sí solo no es unánime en ambas métricas. El tamaño del efecto sigue siendo MODESTO en
fracción de masa (Δ≈0.01-0.03, sobre fracciones de ~0.08-0.12) y más variable en κ_V (de −0.10 a +0.38
según el par) — no hay ninguna separación limpia tipo "todas las Clase III muy por encima de todas las
Clase I"; es una tendencia direccional consistente, no una frontera dura. n=8 (o n=5 limpio) sigue
siendo poca muestra para hablar de solidez estadística fuerte — es más poder que el piloto de 3 pares,
no una batería masiva.

## 3. Archivos de esta tarea

- `cs090_fase5b_test_turbulencia.py` — test de 2 semillas de turbulencia nuevas sobre el mismo grafo
  (r9). Resultado: `cs090_fase5b_test_turbulencia_resultados.csv`.
- `cs090_fase5b_historico_n_sumideros_RAW.csv` (copiado del scratchpad al proyecto) — 64 filas, escaneo
  completo de `.sink` en `phantom_cs073/`, agrupable por N.
- `cs090_fase5b_generar_pares_v2.py` — generación de reglas nuevas + búsqueda de pares con match exacto
  K=kcap (incluye el bug de colisión de nombres, documentado en el propio docstring y en §2.1).
- `cs090_fase5b_candidatas_v2.csv` — las 40 reglas nuevas generadas y clasificadas (ground truth real
  usada para detectar y corregir el bug).
- `cs090_fase5b_escala_v2_metricas_FINAL.csv` — las 16 corridas finales (8 pares), con columna
  `match_exacto_K_kcap` y `nota` explicando cada caso.
- `/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v2/` — condiciones iniciales, dumps binarios
  y `.sink` de las 8 reglas nuevas corridas en esta tarea (`r2`, `r12`, `r20`, `r28`, `r39`, `r2v1fix`,
  `r12v1fix`, más las reutilizadas del piloto: `r9`, `r19`, `r1`, `r17`, `r6`, `r14`).
- `/Users/alexis/phantom_cs073/test_turbulencia_r9/` — las 2 corridas del test de turbulencia.
- Este informe.

Ningún script congelado del piloto anterior fue modificado. No se declaró cierre ni veredicto sobre si
A2-B0-C2 "confirma" o "refuta" nada, ni sobre si el "8" es fallo o física real. No se hicieron commits de
git.

## 4. Nota de presupuesto de tiempo

Esta tarea excedió el presupuesto original estimado (~70-90 min) — la Parte 1 (escaneo histórico +
verificación de código + test de turbulencia) y la Parte 2 (generación de 40 candidatas + detección y
corrección de un bug real de colisión de nombres + 8 pares × 2 corridas de Phantom) tomaron más tiempo
del previsto, en gran parte por el trabajo extra de detectar y arreglar el bug de colisión (sin ese
arreglo, 3 de los 8 pares hubieran quedado con datos física y silenciosamente equivocados). Se priorizó
terminar ambas partes de forma honesta y verificada antes que cortar temprano con datos sin auditar.
