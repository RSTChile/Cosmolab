# Profundizar A2-B0-C2 — ¿bimodal robusto, qué lo separa, y comparte la duda de A0?

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude) · Sigue de `FASE5A_completo_resultado_CS.md` §5, que
encontró el patrón más limpio del barrido de 180 reglas en la combinación **A2-B0-C2** (grafo dinámico
co-emergente, sin retroalimentación relación-sobre-relación, con límite de escala duro): 5/10 Clase III,
5/10 Clase I, 0/10 Clase II.

No se declara cierre ni veredicto. Se reportan números; la lectura final es de Alexis. No se corrió
Phantom. No se tocó ningún script congelado (`cs090_fase5_generador/motor/clasificador.py`,
`cs080_renormalizacion.py`, ni ninguno de los scripts de referencia anteriores).

## 0. Qué se construyó

Un archivo nuevo: **`cs090_fase5_profundizar_a2b0c2.py`**. Genera **20 reglas ADICIONALES** para
A2-B0-C2 con `cs090_fase5_generador.generar_reglas_clase()` (mismo método, filtro P1-P5 real, sin
modificarlo), las corre con `cs090_fase5_motor.correr_regla_coarse()` (motor corregido de coarse-graining,
N=2000, escalas b=1/2/4/8/16, exactamente los mismos parámetros que usó `cs090_fase5_completo.py` para
las 10 originales) y las clasifica con `cs090_fase5_clasificador.clasificar_regla()` sin tocar ningún
umbral. A diferencia del barrido original, este script sí guarda los parámetros concretos de cada regla
(K, J, noise, meandeg, kcap) en el CSV resumen, para poder responder el Objetivo 2.

**Tiempo real: 22 segundos** para las 20 reglas nuevas (el motor de coarse-graining resultó mucho más
rápido de lo estimado — muy por debajo del presupuesto de 30 min reservado). Filtro P1-P5: **20/20
admitidas, 0 descartadas**.

Salidas:
- `cs090_fase5_profundizar_a2b0c2_resultados.csv` — 100 filas (20 reglas × 5 escalas), dato crudo.
- `cs090_fase5_profundizar_a2b0c2_resumen.csv` — 30 filas (10 originales del barrido de 180 + 20 nuevas),
  una por regla, con clase + parámetros (vacíos para las 10 originales — el barrido de 180 no guardó
  parámetros por regla, ver limitación abajo).

## 1. Objetivo 1 — ¿el bimodal se sostiene con más muestra (n=30)?

| origen | n | I | II | III | intermedio |
|---|---|---|---|---|---|
| 10 originales (barrido de 180) | 10 | 5 | 0 | 5 | 0 |
| 20 nuevas (esta tarea) | 20 | 8 | 1 | 10 | 1 |
| **TOTAL combinado** | **30** | **13 (43.3%)** | **1 (3.3%)** | **15 (50.0%)** | **1 (3.3%)** |

**En términos generales, sí se sostiene: el patrón sigue siendo fuertemente bimodal I/III (28/30 = 93.3%
de las reglas), con Clase IV en 0/30.** La proporción global es prácticamente 50/50 (43.3% vs 50.0%),
igual que en el piloto original.

**Pero el "0/10 en Clase II" del barrido original NO se mantuvo perfecto con más muestra:** apareció
1 regla en Clase II (`A2-B0-C2-r15`: pendiente=0.426, justo en el rango mundo-pequeño 0.35-0.45) y 1 en
"intermedio — sin clase clara" (`A2-B0-C2-r0`: pendiente=**-0.854**, negativa). Revisando la fila cruda
de `r0`: el diámetro real NO es monótono en la escala — 1.0 (b=1, N=2000) → 13.0 (b=2) → 9.0 (b=4) → 6.0
(b=8) → 5.0 (b=16); el salto de diám=1.0 en la resolución nativa (N=2000, sin coarse-graining) hasta
diám=13.0 al agrupar una vez parece anómalo/degenerado, no un patrón limpio de disolución ni de
extensión. El clasificador, correctamente (por diseño anti-Shannon: no fuerza a una caja si el umbral no
lo sostiene), lo dejó como "intermedio" en vez de forzarlo a Clase I. No se investigó más a fondo el
mecanismo interno de esa anomalía (requeriría tocar o instrumentar el motor congelado, fuera de alcance
acá) — se documenta como hallazgo, no se esconde.

**Lectura honesta:** con n=30 el patrón sigue siendo "I o III, casi nunca la zona intermedia" (93.3%),
pero "casi nunca" ya no es "nunca" — con 3x más muestra apareció 1 caso de mundo-pequeño y 1 caso
anómalo. Esto es consistente con lo esperado estadísticamente (10 muestras es poco para afirmar 0% con
confianza) y no cambia la conclusión cualitativa, pero conviene que quede registrado con precisión en vez
de repetir "nunca cae en Clase II" sin matiz.

## 2. Objetivo 2 — ¿qué parámetro separa Clase I de Clase III?

**Limitación de partida:** el barrido original (`cs090_fase5_completo_resumen.csv`) sólo guardó
clase/pendiente/z_agg/holon_ratio por regla, no los parámetros generados (K, J, noise, meandeg, kcap) —
así que este análisis corre **sólo sobre las 20 reglas nuevas** (18 utilizables: 8 en Clase I, 10 en
Clase III, se excluyen la de Clase II y la "intermedio").

**Ningún parámetro separa limpiamente (rangos con solape total en los 5 parámetros)** — no hay un umbral
duro tipo "por debajo de X siempre disuelve". Pero SÍ aparece una correlación moderada, no despreciable
para n=18:

| parámetro | corr. con Clase III | media Clase I | media Clase III |
|---|---|---|---|
| **kcap** (límite de escala duro) | **-0.43** | 6.13 | 5.20 |
| **K** (alfabeto de fase) | **+0.45** | 5.38 | 6.50 |
| noise | +0.24 | 0.190 | 0.224 |
| meandeg | -0.16 | 6.31 | 5.93 |
| J | -0.13 | 0.539 | 0.502 |

Los dos con señal más consistente son **kcap** (a límite de escala MÁS estricto/bajo, más geometría
extensa) y **K** (a alfabeto de fase más grande, más geometría extensa):

- kcap≤5 (límite más estricto): 7/9 = 78% cae en Clase III.
- kcap≥6 (límite más laxo): 3/9 = 33% cae en Clase III.
- K≤5: 3/8 = 38% cae en Clase III. K≥6: 7/10 = 70% cae en Clase III.
- Combinando ambos (kcap≤5 **y** K≥6): 5/6 = 83% cae en Clase III; el resto (kcap≥6 o K≤5): 5/12 = 42%.

**Interpretación en simple, con analogía:** imaginá que kcap es "cuántos amigos cercanos puede tener cada
nodo como máximo" y K es "cuántos estados de ánimo distintos puede tener". La tendencia (no una regla
dura, sino una inclinación estadística con poca muestra) es: mientras MÁS estricto el límite de amigos
(kcap bajo) y MÁS variedad de estados de ánimo posibles (K alto), más veces la red termina "extendiéndose"
en vez de "aplastarse". Tiene sentido narrativamente — un límite duro de conexiones fuerza a la red a
mantenerse recableando en busca de compatibilidad en vez de asentarse en grupos densos, y con más
variedad de estados hay más margen para que ese recableo encuentre estructura genuina — pero **con n=18
y solape total de rangos, esto es una pista, no una explicación cerrada del bimodal.** No se encontró un
umbral único que explique el 50/50 de forma determinista; el resultado sigue viéndose parcialmente
estocástico incluso controlando por estos dos parámetros.

Nota adicional: las pendientes de Clase I (rango combinado 0.459–0.698) y de Clase III (rango combinado
0.720–1.079) quedan cerca pero sin cruzarse del umbral 0.7 del clasificador — es decir, la separación I/III
observada es sobre todo un fenómeno de la propia pendiente log-log agrupándose en dos bandas, no un
artefacto de que z_sostenido esté empujando reglas de pendiente baja hacia Clase III por otra vía.

## 3. Objetivo 3 — ¿A2-B0-C2 comparte la duda del "grafo de medición derivado" de A0?

**No — evaluación honesta por inspección de código, confirmada.** En `cs090_fase5_motor.py`, función
`medir(sustrato, p, rng)` (línea ~328):

- Para **A0**: llama a `_grafo_medicion_A0(sustrato["S"], p["K"], p["sim_thr_frac"], rng)` — construye un
  grafo DESDE CERO, por similitud de estado entre pares muestreados, que **nunca participó de la
  dinámica** (que en A0 es difusión local en anillo por offsets `left`/`right`, sin ningún objeto de
  adyacencia en ningún momento). El parámetro `sim_thr_frac` sólo existe para esta construcción post-hoc
  de medición.
- Para **A1/A2**: la rama `else` usa directamente `adj = sustrato["adj"]` — el **mismo objeto** de
  adyacencia que usó `dinamica_B0`/`dinamica_B1` para correr la dinámica real, incluida (para A2) la
  co-evolución por `_recablear_A2` (recableo por compatibilidad de estado, cada 3 sweeps) y la poda de
  C2 por `_enforce_kcap`. No se construye ningún grafo adicional sólo para medir.

Chequeo cruzado por grep: `sim_thr_frac` (el parámetro que activa la construcción del grafo derivado en
A0) **sólo aparece usado en `_grafo_medicion_A0`** en todo `cs090_fase5_motor.py` — para A2 el generador
igual lo genera (viene del espacio de parámetros genérico), pero el motor nunca lo lee. Es un parámetro
inerte para A2, no un artefacto activo.

**Conclusión:** el grafo que se mide en A2-B0-C2 (diám/giant/coarse-graining) ES el mismo grafo genuino
sobre el que corrió la dinámica y la poda de costo — no hay una "vara de medición" añadida por fuera del
sustrato. La duda de A0 (que el mundo-pequeño medido sea un artefacto de cómo se mide, no de cómo se
comporta el sustrato) **no aplica estructuralmente a A2-B0-C2**, porque no existe una construcción de
grafo equivalente a `_grafo_medicion_A0` en la rama A1/A2 de `medir()`.

## 4. Recomendación sobre Fase V-B (Phantom)

No corresponde a esta tarea declarar cierre, pero en base a los números: A2-B0-C2 sigue siendo el
candidato con la señal más limpia y de las más grandes del barrido (28/30 = 93.3% en I o III, 50% en III
específicamente, con n=30 acumuladas entre ambas tandas — supera el umbral ">10 reglas independientes"
del criterio "Fuerte" de la especificación). El chequeo del Objetivo 3 quita una duda metodológica
concreta que sí aplicaba a A0 (no aplica acá). El Objetivo 2 no encontró un umbral limpio que explique el
50/50 — sigue viéndose parcialmente estocástico incluso con más muestra, con una pista moderada (kcap
bajo + K alto → más Clase III) que valdría la pena que Phantom explore explícitamente variando esos dos
parámetros con más resolución si Alexis decide avanzar. **En criterio del ejecutor, los datos sostienen
pasar A2-B0-C2 a Fase V-B — la decisión final es de Alexis.**

## 5. Archivos de esta tarea

- `cs090_fase5_profundizar_a2b0c2.py` — script nuevo (único archivo de código de esta tarea).
- `cs090_fase5_profundizar_a2b0c2_resultados.csv` — 100 filas, dato crudo por regla/escala (20 reglas nuevas).
- `cs090_fase5_profundizar_a2b0c2_resumen.csv` — 30 filas (10 originales + 20 nuevas), clase + parámetros.
- Este informe.

Ningún script congelado fue modificado. No se corrió Phantom. No se hicieron commits de git.
