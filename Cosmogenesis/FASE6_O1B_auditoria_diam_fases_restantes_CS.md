# Fase VI / O1-B — auditoría del bug de `_diam` en las fases que faltaban (Fase IV y línea CS07x-CS08x)

**Fecha:** 11-ago-2026 · Ejecuta: CC (Claude) · Tarea **O1-B** del `FASE6_PLAN_EJECUCION_COMPLETA_CS.md`.

**Antecedentes:** `FASE6_outliers_pendiente_negativa_CS.md` (dónde se encontró el bug) y
`FASE6_adopcion_diam_corregido_CS.md` (la corrección + la re-medición de Fase III Exp.1 y de toda la
Fase V). Ese informe dejó explícito, en su cola de pendientes, punto 5:

> *"esta tarea cubrió Fase III y toda la Fase V; **Fase IV y la línea CS07x-CS08x no se auditaron**, y
> usan el mismo `_diam`."*

Esta tarea audita ese resto. **No se editó ningún script congelado. No se corrió Phantom. No se hicieron
commits. No se declara cierre ni veredicto** — se reportan números.

---

## 0. En simple, con analogía

El problema conocido: la rutina vieja que medía el "diámetro" de una maqueta de alambre
(`_diam`, congelada en cs055) **apoyaba el metro en el primer alambre que encontraba por número de
nodo**. Si ese alambre estaba en un pedacito suelto de dos nodos colgando al costado, medía el
pedacito. *Apoyó el metro en el buzón de la vereda en vez del edificio: dio 30 cm.*

Se sabía que eso mordió 15 de 430 reglas de la línea A2-B0-C2 (Fase V). Faltaba mirar dos vecindarios:
**Fase IV** (los 4 sustratos de orden superior) y **la línea CS07x-CS08x** (dirección temporal,
gradientes, κ_V, C-N4, renormalización, poda, y los cinco experimentos espectrales).

**El resultado en una frase: el buzón no aparece en ninguno de los dos vecindarios.**

- En **Fase IV** ni siquiera hay metro: esos experimentos **no miden diámetro en ninguna parte** — su
  número es la holonomía. Y por si acaso se midió igual: sus 60 grafos están **enteros de una sola
  pieza**, sin un solo nodo suelto, así que ni con el metro viejo habría cambiado nada.
- En la **línea CS07x-CS08x**, de 15 scripts sólo **dos** miden diámetro: cs080 (ya auditado antes,
  0/54) y **cs081** (la poda dinámica, que faltaba). Se re-midió entero: **0 de 126** mediciones
  descarrilan, y los números viejo y corregido salen **idénticos** en las 126.
- Apareció una alarma menor en el CSV histórico de cs080 (dos series donde el diámetro sube de 3 a 4 al
  agrupar más grueso, algo que "no debería pasar"). Se investigó a fondo: **no es el bug**, es que las
  cajas de una escala y las de la otra se sortean por separado, y dos recubrimientos distintos del mismo
  grafo pueden diferir en ±1. Pasa el **21,6 %** de las veces sin que nada esté roto.
- Y `sarracen` (la librería que lee los volcados de Phantom) **nunca se había desinstalado**: está viva
  en el `venv/` del proyecto. El informe anterior la buscó en los intérpretes del sistema, donde
  efectivamente no está — pero el `venv/` es el que usan los scripts de Phantom. Verificado leyendo un
  volcado real: da exactamente el mismo número que ya estaba en el CSV.

---

## 1. PARTE 1 — El detector sobre las fases no auditadas

Scripts nuevos (ninguno toca código congelado):

| archivo | qué hace |
|---|---|
| `cs090_fase6_o1b_auditoria_diam_fases_restantes.py` | Partes A, B y C (abajo) |
| `cs090_fase6_o1b_remedir_cs081_poda.py` | Parte D: re-medición completa de Fase III Exp.2 (cs081) |
| `cs090_fase6_o1b_chequeo_monotonia_cs080.py` | seguimiento de la alarma de la Parte B |

CSV producidos: `cs090_fase6_o1b_auditoria_estatica.csv`,
`cs090_fase6_o1b_detector_csv_historicos.csv`, `cs090_fase6_o1b_fase4_grafos.csv`,
`cs090_fase6_o1b_remedicion_cs081.csv`, `cs090_fase6_o1b_monotonia_cs080.csv`.

### 1.A — ¿Quién mide diámetro, realmente? (auditoría estática por AST)

No se usó `grep` sino el **árbol sintáctico**: se buscó toda *llamada* (no mención en comentario) a
`_diam` / `diam_original` / `diam_gigante` / `diagnostico`, y a los envoltorios que lo llaman por dentro
(`metricas_escala`). Además se recorrió el **cierre transitivo de imports locales** de cada script, para
que "puede alcanzarlo por la cadena de imports" quede distinguido de "lo llama".

| script | fase | estado | llamadas propias |
|---|---|---|---|
| `cs076_direccion_temporal.py` | CS07x | **NO-TOCA** | — |
| `cs077_gradientes_atractores.py` | CS07x | **NO-TOCA** | — |
| `cs078_kappaV_permutacion.py` | CS07x | **NO-TOCA** | — |
| `cs079_delimitacion_cn4.py` | CS07x | **NO-TOCA** | — |
| `cs080_renormalizacion.py` | Fase III Exp.1 | **USA** | `_diam`, `metricas_escala()->_diam` |
| `cs081_poda_dinamica.py` | Fase III Exp.2 | **USA** | `_diam`, `metricas_escala()->_diam` |
| `cs082_fase4_4sustratos.py` | **Fase IV** | **NO-TOCA** | — |
| `cs083_fase4_robustecer.py` | **Fase IV** | **NO-TOCA** | — |
| `cs083b_fase4_control_local_global.py` | **Fase IV** | **NO-TOCA** | — |
| `cs084_espectro_laplaciano.py` | CS08x | IMPORTA-NO-USA | — |
| `cs085_espectro_jerarquia_cs073.py` | CS08x | IMPORTA-NO-USA | — |
| `cs086_espectro_renorm_poda.py` | CS08x | IMPORTA-NO-USA | — |
| `cs087_hodge_fase4.py` | CS08x/Fase IV | **NO-TOCA** | — |
| `cs088_espectro_proximidad_null12.py` | CS08x | IMPORTA-NO-USA | — |
| `cs089_on77_espectral.py` | CS08x | IMPORTA-NO-USA | — |

**De 15 scripts objetivo: 2 llaman al diámetro, 5 lo alcanzan por import sin llamarlo, 8 no lo tocan.**

Tres consecuencias que conviene dejar escritas:

1. **Fase IV entera (cs082 / cs083 / cs083b / cs087) es NO-TOCA.** No es que "importe `_diam` y no lo
   use": la cadena de imports de cs082 es `time` + `numpy` + `collections` y nada más — ni siquiera
   alcanza a cs055/cs057. Chequeo independiente por texto plano: la subcadena `diam` aparece **cero
   veces** en los tres archivos de Fase IV y **cero veces** en cs087. La métrica de Fase IV es la
   **holonomía de triángulos** (`_holonomia_triangulos`), más el conteo de ejes.
2. **La línea CS07x (cs076-cs079) es NO-TOCA.** Son experimentos de campo/partículas (skew y gradientes
   de φ, permutación de κ_V, delimitación C-N4 sobre volcados de Phantom): no construyen un grafo cuyo
   diámetro se mida.
3. **Los cinco espectrales (cs084/85/86/88/89) son IMPORTA-NO-USA.** Su número es espectral
   (λ_max, λ₂, dispersión, unfolding). Caso digno de mención: `cs086` sí llama a
   `C81.proceso066_instrumentado`, que **por dentro** ejecuta `C7._diam(adj, N)` en cada paso
   (cs081 línea 154) — pero ese valor se acumula en una lista local `D` que **nunca se devuelve ni se
   usa** (la función retorna `adj, V, flip_count`). Es código muerto: el diámetro se calcula y se tira,
   no llega a ninguna conclusión de cs086.

### 1.B — Detector barato sobre los CSV históricos que guardan diámetro por escala

Criterios aplicados a cada serie de escalas (b = 1, 2, 4, 8, 16, 32):

- **`descarrila_b1`** — el propuesto en el informe anterior: `diám(b=1) < diám(b=2)` **y** `diám(b=1) ≤ 3`.
- **`viola_monotonia`** — el criterio general, más sensible (el informe anterior ya anotó que "pendiente
  negativa" se le escapa un caso): existe algún `diám(b_k) < diám(b_{k+1})`.

| CSV histórico | qué es | series | `descarrila_b1` | `viola_monotonia` |
|---|---|---|---|---|
| `cs080_renormalizacion.csv` | Fase III Exp.1 — **control**, ya auditado | 9 | **0** | 2 |
| `cs081_poda_dinamica.csv` | Fase III Exp.2 (poda) — **no auditado antes** | 21 | **0** | **0** |

Las 2 no-monotonías son ambas del brazo `local_barajado` (semillas 80100 y 80200), con la misma serie
de diámetros `7 | 5 | 4 | 3 | 4 | 3`: el salto es de **3 en b=8 a 4 en b=16**, +1, lejos de b=1. Se
investigó aparte (§1.B-bis) y **no es el bug**.

*(Fase IV no aparece en esta tabla porque no existe ningún CSV suyo con columna de diámetro — coherente
con §1.A: nunca lo midió.)*

### 1.B-bis — Las 2 no-monotonías de cs080: azar de recubrimiento, no bug

`cajas_bfs` sortea semillas de caja y arma un recubrimiento **nuevo e independiente en cada escala**:
las cajas de b=16 **no contienen** a las de b=8. Contraer cajas conexas no puede alargar un camino
*respecto del grafo original*, pero comparar dos recubrimientos distintos entre sí no está protegido por
ese argumento. Prueba directa (`cs090_fase6_o1b_chequeo_monotonia_cs080.py`): se construyó una vez el
sustrato del caso señalado (arm=`local_barajado`, seed 80100, N=8000) y se repitió el recubrimiento
**40 veces por escala**, cambiando sólo la semilla del sorteo de cajas.

```
b=8    diám viejo: min=3 max=5 media=3.70   descarrilamientos=0/40   viejo!=corregido=0/40
b=16   diám viejo: min=3 max=4 media=3.58   descarrilamientos=0/40   viejo!=corregido=0/40
pares (b=8, b=16) cruzados: 1600 -> con diám(16) > diám(8): 345  (21,6 %)
descarrilamientos TOTALES en las 80 réplicas: 0
```

**Conclusión numérica:** el +1 de no-monotonía aparece en el **21,6 %** de las parejas de recubrimientos
sanos, con **cero** descarrilamientos y con viejo ≡ corregido en 80/80. Es ruido de recubrimiento. Nota
metodológica para el futuro: **`viola_monotonia` es un buen detector sólo en b=1→b=2**; entre escalas
gruesas tiene una tasa de falsos positivos alta (~20 % en este sustrato) y hay que acompañarlo del
diagnóstico de componentes.

### 1.C — Re-medición directa de los grafos de Fase IV

Aunque §1.A ya cierra el asunto formalmente ("no se mide diámetro, no puede estar afectado"), se hizo
la pregunta más fuerte: **¿los grafos de Fase IV son siquiera del tipo que puede descarrilar?** Se
reconstruyeron con las funciones propias de cs082 sin tocarlas (`construir_base`, `_linea_adyacencia`),
para las **20 semillas** de cs083/cs083b, los tres grafos sobre los que corre la dinámica.

| grafo | n | N medio | componentes: 1 / >1 | nodos aislados (medio) | descarrila | viejo ≠ corregido |
|---|---|---|---|---|---|---|
| `base_nodos` (ER N=110, p=0.09) | 20 | 110,0 | **20 / 0** | 0,00 | **0** | **0** |
| `linea_aristas_sust1` (line-graph de aristas) | 20 | 542,5 | **20 / 0** | 0,00 | **0** | **0** |
| `linea_trios_sust2y4` (line-graph de triángulos) | 20 | 161,6 | **20 / 0** | 0,00 | **0** | **0** |

**60 de 60 grafos de Fase IV son de una sola componente conexa, sin un solo nodo aislado.** El bug
requiere fragmentación para morder; acá no hay dónde morder. Los 60 diámetros viejo y corregido son
idénticos.

### 1.D — Re-medición completa de Fase III Exp.2 (cs081, la poda) — el único hueco real

Es el único script no auditado que **sí** llama a `_diam`, y su conclusión publicada
(`FASE3_poda_dinamica_resultado_CS.md`) **es** una pendiente de diámetro. Se reconstruyó la cadena
exacta de `cs081.corre_semilla` (mismo `proceso066_instrumentado`, mismo `costo_por_arista`, mismas 7
variantes, mismas 6 escalas) midiendo el diámetro de las dos maneras en cada escala.

**3 semillas × 7 variantes × 6 escalas = 126 mediciones.**

```
[resultado] escalas donde la medición vieja DESCARRILA: 0/126
[resultado] escalas donde viejo != corregido:           0/126
```

Pendientes log(diám) vs log(N_cajas), medianas sobre las 3 semillas:

| variante | pendiente **vieja** | pendiente **corregida** | diferencia máxima por semilla |
|---|---|---|---|
| `sin_poda` | +0,3873 | +0,3873 | 0,0 |
| `costo_P50` | +0,8279 | +0,8279 | 0,0 |
| `azar_P50` | +0,7006 | +0,7006 | 0,0 |
| `costo_P70` | +0,5556 | +0,5556 | 0,0 |
| `azar_P70` | +0,4700 | +0,4700 | 0,0 |
| `costo_P90` | +0,4481 | +0,4481 | 0,0 |
| `azar_P90` | +0,4510 | +0,4510 | 0,0 |

**Limitación declarada (no se disimula).** cs081 (igual que cs080) deriva semillas de rng con
`hash(str)`, aleatorizado por proceso salvo `PYTHONHASHSEED` fijo, valor que **no quedó registrado** en
la corrida histórica. Esta re-medición fija `PYTHONHASHSEED=0`: sus sustratos son del **mismo tipo**
pero **no la misma realización** que los del informe de Fase III. Por eso las medianas de arriba no
coinciden número a número con las publicadas (sin_poda **0,421**, azar_P50 **0,655**, costo_P50
**0,786**) — coinciden en **orden y magnitud** (0,39 / 0,70 / 0,83; costo_P50 > azar_P50 > sin_poda, con
la misma brecha costo-vs-azar de ~0,13). Lo que esta re-medición **sí** establece sin depender de la
realización es lo que se le preguntó: **este tipo de sustrato no descarrila la medición, en ninguna de
las 126 escalas, con ninguna de las 7 variantes.** Para comparar diámetro contra diámetro con el CSV
histórico se aplicó, en cambio, el detector barato **directo sobre el CSV histórico** (§1.B): 0/21.

### 1.E — Resumen de la Parte 1: ¿cuántos grafos descarrilan en cada fase?

| fase | qué se auditó | grafos / mediciones | **descarrilamientos** |
|---|---|---|---|
| **Fase IV** (cs082/083/083b/087) | no mide diámetro (AST) + re-medición de sus grafos | 60 grafos | **0** |
| **CS07x** (cs076-cs079) | no mide diámetro (AST) | — (no aplica) | **0** |
| **CS08x espectrales** (cs084/85/86/88/89) | no mide diámetro (AST); el `_diam` interno de cs086 es código muerto | — (no aplica) | **0** |
| **Fase III Exp.1** (cs080) | CSV histórico (detector) | 9 series | **0** *(ya sabido: 0/54 en el informe previo)* |
| **Fase III Exp.2** (cs081) | CSV histórico + re-medición completa | 21 series / 126 mediciones | **0** |
| *(control de la alarma)* cs080 recubrimientos | 80 réplicas b=8/b=16 | 80 | **0** |

**Total de descarrilamientos encontrados en esta auditoría: CERO.**

---

## 2. PARTE 2 — Impacto sobre las conclusiones publicadas

Como la Parte 1 no encontró **ningún** descarrilamiento, no hay nada que re-clasificar. Igual conviene
dejar por escrito, experimento por experimento, **por qué** cada conclusión queda donde estaba — porque
las razones no son todas iguales.

### 2.1 Fase IV — el diámetro no entra en la conclusión (con evidencia)

Ésta era la pregunta con más en juego, y la respuesta es la mejor posible.

| informe | conclusión publicada | ¿el diámetro entra? |
|---|---|---|
| `FASE4_orden_superior_resultado_CS.md` | sólo el 2-complejo con retroalimentación cara→arista se separa; holonomía ~5× menor | **No.** El observable es `_holonomia_triangulos` + conteo de ejes. |
| `FASE4_robustecido_CS.md` | 20 semillas, z=-34,8; descomposición 92 % / 8 % | **No.** Mismo observable, más semillas y un control fino. |
| `FASE4_control_local_global_CS.md` | el "92 %" es concentración en tríos, no dispersión global; NULL-GLOBAL indistinguible de ruido (p=0,52) | **No.** Mismo observable. |

Evidencia, en tres formas independientes:

1. **AST** (§1.A): cero llamadas a cualquier función de diámetro en cs082, cs083, cs083b y cs087; su
   cierre de imports locales ni siquiera alcanza cs055/cs057.
2. **Texto plano**: la subcadena `diam` aparece **0 veces** en los cuatro archivos de código y **0
   veces** en los tres informes publicados de Fase IV.
3. **Robustez adicional** (§1.C): incluso si el diámetro se hubiera usado, los 60 grafos son de una sola
   pieza — la medición vieja y la corregida dan el mismo entero en los 60.

**Ninguna conclusión de Fase IV cambia. No hay nada que re-correr en Fase IV por causa de `_diam`.**

### 2.2 Línea CS07x-CS08x

| experimento | su número de veredicto | ¿toca el diámetro? | ¿cambia? |
|---|---|---|---|
| CS076 dirección temporal | skew / asimetría de campo | no | no |
| CS077 gradientes-atractores | métricas del motor holístico | no | no |
| CS078 κ_V permutación | κ_V y su permutación | no | no |
| CS079 delimitación C-N4 | picos/curtosis sobre volcados Phantom | no | no |
| **CS080 renormalización** | pendiente diám vs N_cajas | **sí** | no *(0/54 en el informe previo; 0/9 series acá; la alarma de monotonía resuelta en §1.B-bis)* |
| **CS081 poda dinámica** | pendiente diám vs N_cajas | **sí** | **no** *(0/126 descarrilamientos, viejo ≡ corregido)* |
| CS084 espectro laplaciano | λ_max, dispersión, unfolding | no | no |
| CS085 espectro jerarquía CS073 | espectro | no | no |
| CS086 espectro renorm/poda | λ_max, λ₂ | no *(el `_diam` interno es código muerto)* | no |
| CS087 Hodge sobre Fase IV | descomposición de Hodge | no | no |
| CS088 espectro proximidad NULL-1/2 | espectro | no | no |
| CS089 O-N7.7 espectral | espectro | no | no |

Dos notas de detalle:

- **CS084 y CS086 citan el diámetro, pero como contraste con Fase III, no como medición propia.** Por
  ejemplo CS086: *"el diámetro sólo había mostrado una brecha de pendiente de 0,786 vs 0,655"*. Ese par
  de números es de Fase III Exp.2 — y §1.D los deja donde estaban (la corrección no mueve ninguna
  pendiente de cs081). Así que la frase de contraste sigue siendo correcta tal como está publicada.
- **La conclusión de Fase III Exp.2 que sí está en juego indirectamente**
  —`FASE3_poda_dinamica_resultado_CS.md`: "podar por costo relacional mueve la pendiente más que azar,
  reproducible pero modesto
  (costo_P50=0,786 vs azar_P50=0,655 vs sin_poda=0,421)"— queda **intacta**: la corrección da
  exactamente la misma pendiente en las 21 series, y el orden costo > azar > sin_poda se reproduce
  también en la realización nueva (0,83 > 0,70 > 0,39).

### 2.3 Lo que esta auditoría NO cubre (para que quede anotado, no para taparlo)

- La **realización histórica exacta** de cs080/cs081 no es reproducible bit a bit (`PYTHONHASHSEED` sin
  registrar). Se compensó aplicando el detector directamente al CSV histórico, que sí es el dato de la
  corrida original.
- Quedan **fuera del encargo** los scripts cs054-cs072 y la familia cg00x, que también contienen `_diam`
  (aparecen en el grep, no en la lista de objetivos O1-B). Varios de ellos (cs064/cs065/cs066) guardan
  `diam_fin` en CSV y podrían pasarse por el mismo detector barato en minutos, si Alexis lo quiere.
- Los grafos de Fase IV se re-midieron en su **estado inicial** (el que construye `construir_base`); la
  dinámica de cs082 no agrega ni quita aristas —sólo actualiza orientaciones Z_K sobre relaciones
  fijas—, así que la topología no cambia durante la corrida y la medición inicial es la relevante.

---

## 3. PARTE 3 — `sarracen` (infraestructura)

### 3.1 El diagnóstico anterior era correcto pero incompleto

`FASE6_outliers_pendiente_negativa_CS.md` reportó que `sarracen` "ya no está instalada en ningún
intérprete de la Mac". Verificado ahora, intérprete por intérprete:

| intérprete | `import sarracen` |
|---|---|
| `/usr/local/bin/python3` (3.13 del sistema) | ✗ `ModuleNotFoundError` |
| `/usr/bin/python3` (macOS) | ✗ `ModuleNotFoundError` |
| `/usr/local/bin/python3.9` | ✗ `ModuleNotFoundError` |
| `/usr/local/bin/python3.11` | ✗ `ModuleNotFoundError` |
| **`./venv/bin/python`** (venv del proyecto, Python 3.13.14) | **✓ `sarracen 1.3.1`** |

O sea: **no hacía falta reinstalarla — nunca se había ido del `venv/`.** Lo que faltaba era usar el
intérprete correcto. Es el mismo `venv/` que documenta el encabezado de `leer_volcado_phantom.py`
("Se instaló en el venv del proyecto (venv/)"), y trae además todo el resto de la cadena:
`numpy 2.3.5`, `pandas 3.0.3`, `scipy 1.18.0`, `numba 0.62.1`, `llvmlite 0.45.1`, `matplotlib 3.11.1`.
**No se instaló ni se cambió nada** en el entorno.

> **Nota operativa:** los scripts que leen volcados de Phantom hay que correrlos con
> `./venv/bin/python`, **no** con `python3`. Los scripts de grafos (cs080/cs081/cs090…) corren con
> `python3.9`, que es donde está la cadena vieja. Son dos entornos distintos y conviene no confundirlos.

### 3.2 Verificación sobre un volcado real

Volcado: `/Users/alexis/phantom_cs073/bateria_fase5b_a2b0c2_escala_v4/A2-B0-C2-batch3-r107_I/cosmog_00500`
(último dump de la corrida). Se ejecutó la cadena completa —`leer_volcado_phantom.leer_dump` →
`cs090_fase5b_analizar.analizar_carpeta`— y se comparó campo por campo contra la fila que ya estaba en
`cs090_fase5b_escala_v4_metricas.csv`:

| campo | CSV existente | leído ahora | ¿coincide? |
|---|---|---|---|
| `n_gas_inicial` | 2000 | 2000 | ✓ |
| `n_dump_final` | `cosmog_00500` | `cosmog_00500` | ✓ |
| `masa_gas_final` | 16797,8 | 16797,8 | ✓ |
| `masa_sumideros_final` | 2002,2000000000007 | 2002,2000000000007 | ✓ (bit a bit) |
| `masa_total_final` | 18800,0 | 18800,0 | ✓ |
| **`fraccion_masa_en_sumideros`** | **0,10650000000000004** | **0,10650000000000004** | **✓ (bit a bit)** |
| `n_sumideros` | 8 | 8 | ✓ |
| `t_primer_sumidero` | 0,031 | 0,031 | ✓ |
| `masa_acretada_total` | 2002,2 | 2002,2 | ✓ |

**Chequeo cruzado con la vía del `.sink`** (que es lo que pedía la tarea): las dos rutas son
independientes —`masa_sumideros_final` sale del **volcado binario** vía sarracen; `masa_acretada_total`
sale del **log incremental `cosmog01.sink`** sumando el `macc` final de cada sumidero— y dan
**2002,2000000000007** vs **2002,2** (diferencia 7×10⁻¹³, redondeo de punto flotante). La fracción de
masa reconstruida hoy con sarracen es **idéntica** a la registrada.

`sarracen` queda funcionando; `leer_volcado_phantom.py` vuelve a leer volcados reales sin cambios.

---

## 4. Archivos de esta tarea

**Scripts nuevos** (ninguno modifica código congelado):

- `cs090_fase6_o1b_auditoria_diam_fases_restantes.py` — Partes A (AST), B (detector sobre CSV
  históricos) y C (re-medición de los grafos de Fase IV).
- `cs090_fase6_o1b_remedir_cs081_poda.py` — re-medición completa de Fase III Exp.2 (126 mediciones).
- `cs090_fase6_o1b_chequeo_monotonia_cs080.py` — 80 recubrimientos para resolver la alarma de monotonía.

**CSV:**

- `cs090_fase6_o1b_auditoria_estatica.csv` — 15 filas: script, fase, estado, llamadas, conclusión publicada.
- `cs090_fase6_o1b_detector_csv_historicos.csv` — 30 filas: una por serie de escalas, con las dos banderas.
- `cs090_fase6_o1b_fase4_grafos.csv` — 60 filas: los grafos de Fase IV con las dos varas y el diagnóstico.
- `cs090_fase6_o1b_remedicion_cs081.csv` — 126 filas: cs081 escala por escala, viejo vs corregido.
- `cs090_fase6_o1b_monotonia_cs080.csv` — 80 filas: réplicas de recubrimiento b=8 / b=16.

**Logs:** `cs090_fase6_o1b_remedir_cs081.log`, `cs090_fase6_o1b_monotonia_cs080.log`.

---

## 5. Qué queda sobre la mesa (decide Alexis, acá no se cierra nada)

1. Pasar el mismo detector barato por los CSV de **cs064 / cs065 / cs065b / cs066** (columna `diam_fin`)
   y por la familia **cg004**: es lectura de CSV, minutos, y completaría el barrido de `_diam` en todo
   el proyecto.
2. Registrar `PYTHONHASHSEED=0` como convención para todo lo que use `hash(str)` en la derivación de
   semillas, para que las corridas futuras sí sean reproducibles bit a bit.
3. Anotar en la ficha del método que **`viola_monotonia` sólo es fiable en b=1→b=2**: entre escalas
   gruesas tiene ~20 % de falsos positivos por el sorteo independiente de cajas (§1.B-bis).
4. Anotar la nota operativa de los dos entornos (`./venv/bin/python` para Phantom, `python3.9` para
   grafos), que fue lo que produjo el falso "sarracen desapareció".
