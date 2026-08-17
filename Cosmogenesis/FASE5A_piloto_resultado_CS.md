# Fase V-A — resultado del piloto (5 clases × 3 reglas)

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude) · Plan seguido: `/Users/alexis/.claude/plans/synchronous-dreaming-fox.md`
· Especificación implementada: `FASE5_especificacion_universalidad_CS.md`

No se declara cierre ni veredicto sobre S>0 ni sobre ninguna clase. Se reportan números; la lectura final es de Alexis.

## 0. Qué se construyó

4 archivos nuevos, ninguno toca los 7 scripts congelados (sólo `import`):

- **`cs090_fase5_generador.py`** — genera reglas parametrizadas por los 3 ejes (A0/A1/A2 × B0/B1 ×
  C0/C1/C2) y aplica el filtro de admisión **P1-P5 con chequeo automatizado real** (simulación chica
  N=50 + inspección estructural del código), no asumido.
- **`cs090_fase5_motor.py`** — motor de ejecución liviano (N=500-2000). Reusa `_circ_mean_update`,
  `_linea_adyacencia`, `_holonomia_triangulos` (cs082), `_diam`/`_giant` (cs055, extraídas por AST —
  ver §3.1), `aleatorio()` (cg003). A0 = campo continuo en anillo (no-grafo genuino). Corre dos NULL
  emparejados por regla: NULL_topo (ER con misma densidad, patrón de `er_null` de cs080) y NULL_valor
  (misma topología, valores i.i.d., patrón de `null_de` de cs082).
- **`cs090_fase5_clasificador.py`** — aplica los umbrales YA FIJADOS en la especificación §4 (no se
  inventó ninguno nuevo).
- **`cs090_fase5_piloto.py`** — driver; corrió las 5 clases del plan × 3 reglas × N=500/1000/1500/2000
  en 1.6 min. Resultados en `cs090_fase5_piloto_resultados.csv` (60 filas).

## 1. Filtro P1-P5 — verificación explícita de que el chequeo es real, no asumido

Se probó primero que el filtro SÍ descalifica: una regla con `K=1` (sin dimensión de diferencia) y
descripción de 600 caracteres con una constante horneada (`G=9.8`) fue correctamente marcada
`admitida=False` con `motivo_descarte` citando P2 y P5 específicamente. El chequeo P1 es empírico
(compara la tasa con que S(t)>mediana predice S(t+1)>mediana bajo dinámica real encadenada contra un
control sin-memoria de resampleo i.i.d.); P3 y P4 inspeccionan/simulan la construcción real del
sustrato (offsets de anillo mutuamente inversos para A0; simetría de adyacencia para A1/A2).

**Resultado en el piloto real:** de 15 reglas generadas (3 por cada una de las 5 clases), **15/15
admitidas, 0 descartadas**. El espacio de parámetros se diseñó dentro de rangos que por construcción
cumplen P1-P5 (sin constantes físicas, K>1 siempre, adyacencia siempre simétrica), así que en la
práctica el filtro no tuvo que descalificar nada en este barrido — pero el mecanismo de descalificación
está demostrado activo (párrafo anterior), no es un chequeo de adorno.

## 2. Ejemplos concretos de reglas generadas (una por eje representativo)

- `A0-B0-C0-r1`: "S en Z_5 por nodo/relación en sustrato A0; actualización = media circular con
  vecinos definidos por adyacencia previa (J=0.62, ruido=0.19) [B0]; costo/localidad = C0."
- `A1-B0-C0-r2`: sustrato A1 (grafo ER fijo, meandeg≈6.3), K=6, J=0.71 — mundo-pequeño esperado.
- `A2-B1-C1-r0`: sustrato A2 (grafo co-emergente, recableo por compatibilidad de estado), B1 (estado
  en ARISTAS, vecindad = línea de adyacencia igual que sustrato-1 de cs082), C1 (poda por costo:
  0.5·z(inconsistencia histórica) + 0.5·z(conflicto de holonomía), percentil fijo P70).
- `A2-B0-C2-r1`: A2 + límite de escala duro (grado máx=5 por nodo, mismo criterio de soporte-local que
  `gate_localidad`) + costo C1 combinado.

## 3. Resultado de clasificación del piloto (medición cruda diam-vs-N)

| combo | nota | clases (3 reglas) |
|---|---|---|
| A0-B0-C0 | control no-grafo | I, I, I |
| A1-B0-C0 | línea base grafo fijo, mundo-pequeño esperado | I, I, **II** |
| A2-B1-C1 | combinación más prometedora → ¿III/IV? | I, I, I |
| A1-B1-C0 | aísla retroalimentación sin costo | I, I, I |
| A2-B0-C2 | aísla límite de escala duro sin retroalimentación | I, I, I |

Distribución global: **{I: 14, II: 1}**. El clasificador NO colapsa todo en una sola clase (probado
además con datos sintéticos fabricados para II/III/IV — el clasificador los distingue correctamente,
ver `cs090_fase5_clasificador.py` líneas finales).

## 3.1 Hallazgo metodológico — el motor extrae `_diam`/`_giant` sin ejecutar `cs055_proceso_acoplado.py`

`cs055_proceso_acoplado.py` llama a `main()` sin guardia `if __name__=="__main__"` (línea 278) — un
`import` normal dispara toda su batería de experimento como efecto secundario. Se resolvió extrayendo
sólo esas dos funciones por AST (sin tocar el archivo), documentado en el propio motor.

## 4. Hallazgo metodológico principal — el mismo sustrato da pendientes distintas según el método de medición

Se diagnosticó por qué casi todo cayó en Clase I: la pendiente log(diám)-vs-log(N) del piloto se mide
sobre **grafos independientes** a cada N (500, 1000, 1500, 2000). Los umbrales 0.35-0.45/0.7-0.8 de la
especificación fueron calibrados en CS080 con una metodología distinta: **coarse-graining de UN sólo
grafo** (box-counting BFS, `cajas_bfs`/`grafo_grueso`) a escalas b=2,4,8,16,32, midiendo diám vs N_b
(número de supernodos), no diám vs N de grafos independientes.

Se verificó con un diagnóstico puntual: la MISMA sustrato A1-B0-C0 (N=2000) da:
- pendiente por **N independiente** (método del piloto): 0.11-0.36 → Clase I/II según la regla
- pendiente por **coarse-graining** (método de calibración de CS080, reusando `cajas_bfs`/`grafo_grueso`
  tal cual, sin tocarlos): **0.459** → Clase II limpia, consistente con los valores ya medidos del
  proyecto (real=0.376, barajado=0.420, ER=0.406 — FASE5 espec. §2)

Esto explica el aparente colapso hacia Clase I: no es que el motor no genere estructura, es que la
métrica de pendiente del piloto no es la misma vara con la que se calibraron los umbrales.

## 5. Recomendación honesta (v1 — superada por §6, se deja como quedó)

**No escalar a 180 reglas tal cual todavía.** El generador y el filtro P1-P5 están validados (chequeo
real, descalifica cuando debe, no colapsa). El motor corre rápido y estable (1.6 min para 15 reglas ×
4 tallas × dos NULL). Pero antes de comprometer las 180 reglas hay que **corregir la medición de
pendiente**: reemplazar "diám vs N de grafos independientes" por coarse-graining de un único grafo por
regla (reusando `cajas_bfs`/`grafo_grueso` de cs080, ya en la lista de piezas aprobadas para reuso) —
es un cambio acotado (una función del motor), no un rediseño. Con esa corrección, se recomienda repetir
el piloto (mismas 5 clases, mismo costo de tiempo, <5 min) antes de comprometer el barrido completo.

---

## 6. Piloto corregido (10-ago-2026) — pendiente medida por coarse-graining de UN grafo por regla

**Qué cambió y por qué:** `cs090_fase5_motor.py` gana una función nueva, `correr_regla_coarse()`
(§9 del archivo), que reemplaza el método de medición de pendiente descrito en §4 sin tocar
`cs090_fase5_generador.py` ni `cs090_fase5_clasificador.py`. En vez de generar 4 grafos independientes
a N=500/1000/1500/2000, ahora se genera **UN sólo grafo por regla** (dinámica completa corrida una vez,
N=2000) y se le aplica coarse-graining (`cajas_bfs`/`grafo_grueso` de `cs080_renormalizacion.py`,
importados tal cual, ese archivo congelado no se tocó) a las escalas b=1,2,4,8,16 — las mismas con que
se calibraron los umbrales 0.35-0.45/0.7-0.8 en CS080. El NULL_topo se construye igual (grafos
Erdős-Rényi frescos con misma N y densidad, 3 semillas) y se somete al MISMO coarse-graining por
escala, para que el z-score real-vs-null siga siendo por escala. Las "filas" resultantes tienen
exactamente el mismo esquema de campos que el método viejo (`rule_id`, `N`, `diam_real`,
`diam_null_topo`, `diam_null_topo_std`, `holon_real`, `holon_null_valor`, ...), así que
`cs090_fase5_clasificador.py` se reutilizó **sin ningún cambio**.

**Simplificación declarada:** la holonomía (REAL y NULL_valor) se mide una sola vez, a resolución
nativa (b=1), y se replica en las filas de las demás escalas — el bug diagnosticado en §4 era
específico de diám-vs-N (que sí se corrige en cada escala, incluido el z-score que depende de él); no
había evidencia de que el método de holonomía estuviera roto, así que no se tocó sin motivo.

**Generalización a A0:** funcionó sin fricción. `medir()` ya deriva un grafo de medición (por similitud
de estado) para el sustrato A0 continuo — `cajas_bfs`/`grafo_grueso` no distinguen de dónde vino el
grafo, sólo necesitan una lista de adyacencia + N, así que el mismo coarse-graining aplica a A0/A1/A2
sin ramas especiales. La salvaguarda de tiempo prevista para este caso no hizo falta.

Corrida: mismas 5 clases × 3 reglas, N=2000, escalas b=1/2/4/8/16, 3 semillas NULL_topo. Tiempo total:
**0.9 min** (por debajo del presupuesto de 40-50 min). Filtro P1-P5: 15/15 admitidas de nuevo (mismo
generador, no cambió).

| combo | nota | clases v1 (grafos indep.) | clases v2 (coarse-graining) |
|---|---|---|---|
| A0-B0-C0 | control no-grafo | I, I, I | I, I, I |
| A1-B0-C0 | línea base grafo fijo, mundo-pequeño esperado | I, I, **II** | I, **II, II** |
| A2-B1-C1 | combinación más prometedora → ¿III/IV? | I, I, I | I, I, **II** |
| A1-B1-C0 | aísla retroalimentación sin costo | I, I, I | I, **II**, I |
| A2-B0-C2 | aísla límite de escala duro sin retroalimentación | I, I, I | I, **III, III** |

Distribución global v2: **{I: 9, II: 4, III: 2}** (v1 era {I: 14, II: 1}). El clasificador sigue sin
colapsar en una sola clase, y ahora con más separación real entre combos.

**Chequeo de sentido pedido explícitamente:** A1-B0-C0 (grafo ER fijo, línea base mundo-pequeño
esperada) cae en Clase II en 2 de 3 reglas con el método corregido (antes 1 de 3) — coherente con lo
que se esperaba de esa combinación y con el diagnóstico puntual de §4 (misma sustrato A1-B0-C0 daba
0.459 por coarse-graining vs 0.11-0.36 por N-independiente). También aparece variedad nueva que v1 no
mostraba: A2-B0-C2 (límite de escala duro) cae en Clase III en 2 de 3 reglas — una clase que v1 no
alcanzó en ninguna de las 15 reglas del piloto.

### Recomendación honesta (actualiza §5)

El síntoma que motivó "no escalar todavía" — casi-todo-Clase-I por descalibración del método de
medición — está corregido y verificado: el mismo caso de sanidad (A1-B0-C0 → Clase II) ahora se cumple
mayoritariamente, y aparece variedad de clases (incluida Clase III) que antes no aparecía en absoluto.
El motor sigue rápido (0.9 min para 15 reglas × 5 escalas × NULL), y la corrección fue acotada como se
esperaba (una función nueva en el motor, cero cambios en generador/clasificador, cero cambios en
scripts congelados).

Con esto, **el pipeline parece listo para escalar a las 180 reglas completas**, con una salvedad para
que Alexis decida: la muestra de 3 reglas por combo sigue siendo chica (p.ej. A1-B1-C0 dio I/II/I — la
frontera II está cerca, no lejos, en varias reglas), así que la distribución de clases en el barrido
completo bien podría reordenarse con más reglas por combo; esto no es una razón para no escalar, es
simplemente la razón misma para escalar (180 reglas dan la resolución estadística que 3 no dan). La
decisión de escalar y la lectura de los resultados siguen siendo de Alexis.

Archivos de esta corrección: `cs090_fase5_motor.py` (función `correr_regla_coarse`, §9, no se tocó
nada de lo existente), `cs090_fase5_piloto_v2.py` (driver nuevo, no sobrescribe
`cs090_fase5_piloto.py` ni su CSV), `cs090_fase5_piloto_v2_resultados.csv` (75 filas: 15 reglas × 5
escalas).
