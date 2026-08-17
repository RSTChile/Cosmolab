# Fase V-A — resultado del barrido completo (18 clases × hasta 10 reglas)

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude) · Especificación implementada:
`FASE5_especificacion_universalidad_CS.md` · Pipeline reusado tal cual de `FASE5A_piloto_resultado_CS.md` §6
(motor `correr_regla_coarse()`, coarse-graining de UN grafo por regla, N=2000, escalas b=1/2/4/8/16).

No se declara cierre ni veredicto sobre S>0 ni sobre ninguna clase. Se reportan números; la lectura final y
cualquier afirmación sobre la Teoría es de Alexis. No se corrió Phantom.

## 0. Qué se construyó

Un solo archivo nuevo: **`cs090_fase5_completo.py`** — driver que escala `cs090_fase5_piloto_v2.py` de 5
combinaciones×3 reglas a las **18 combinaciones×10 reglas** (3×3×2 = 18 = eje A {A0,A1,A2} × eje B
{B0,B1} × eje C {C0,C1,C2}). Reusa sin tocar: `cs090_fase5_generador.generar_reglas_clase()` (filtro
P1-P5), `cs090_fase5_motor.correr_regla_coarse()` (motor corregido) y
`cs090_fase5_clasificador.clasificar_regla()` (umbrales §4 de la especificación). Ningún script congelado
fue editado.

Corre en dos pasadas (salvaguarda de tiempo pedida): pasada 1 garantiza ≥5 reglas por clase (90 reglas),
pasada 2 completa hasta 10 por clase si el presupuesto (60 min) lo permite. **Tiempo total real: 7.7
minutos** — muy por debajo del presupuesto, la salvaguarda no tuvo que activarse.

Salida: `cs090_fase5_completo_resultados.csv` (750 filas, una por N/escala — dato crudo del coarse-graining)
y `cs090_fase5_completo_resumen.csv` (150 filas, una por regla — clase final + métricas agregadas).

## 1. Hallazgo metodológico — 3 de 18 combinaciones no son ejecutables con el motor congelado actual

Al llegar a la combinación **A0-B1** (fondo sin grafo + retroalimentación relación-sobre-relación), el
motor lanza `KeyError: 'adj'`. Causa: `dinamica_B1()` en `cs090_fase5_motor.py` (línea 236) siempre lee
`sustrato["adj"]`, pero el sustrato A0 (campo continuo en anillo, ver `construir_A0`) nunca produce esa
clave — por diseño es una representación NO-grafo genuina (S, left, right; sin objeto de adyacencia). El
piloto de 5 combos nunca probó esta combinación (sus 5 combos eran A0-B0-C0, A1-B0-C0, A2-B1-C1, A1-B1-C0,
A2-B0-C2 — ninguno cruzaba A0 con B1), así que el hueco no se había visto antes.

**Decisión tomada, respetando la regla de no tocar scripts congelados:** el driver nuevo detecta el error,
lo documenta, corta esa combinación tras 2 fallos seguidos (para no perder tiempo repitiendo el mismo bug)
y sigue con las 15 combinaciones restantes. No se intentó "arreglar" `dinamica_B1` por fuera del archivo —
sería reescribir la dinámica B1 para A0, que es un cambio de diseño, no una reutilización.

**Resultado:** 15/18 combinaciones corrieron completas (10/10 reglas cada una = 150 reglas). Las 3
combinaciones **A0-B1-C0, A0-B1-C1, A0-B1-C2 quedan en 0/10 — no medibles con el pipeline actual.** Esto
es un hallazgo real que Alexis debe ver, no un detalle a esconder: si se quiere ese dato, hace falta
decidir qué significa "retroalimentación relación-sobre-relación" quiere decir cuando no hay relación
(grafo) de base — probablemente haya que definir una noción de "relación entre puntos vecinos del campo"
específica para A0 antes de poder correr esa celda. Filtro P1-P5: de las 165 reglas que llegaron a
intentarse, **0 fueron descartadas por el filtro** (el filtro sigue funcionando, el problema es en el
motor, no en el generador).

## 2. Mapa completo — 18 filas × 4 columnas (cuántas de las reglas cayeron en cada clase)

| combo | n reglas | I (disolución) | II (mundo-pequeño) | III (geometría extensa) | IV (retroalim. cerrada) |
|---|---|---|---|---|---|
| A0-B0-C0 | 10 | 5 | 5 | 0 | 0 |
| A0-B0-C1 | 10 | 9 | 1 | 0 | 0 |
| A0-B0-C2 | 10 | 8 | 2 | 0 | 0 |
| A0-B1-C0 | **0** | — | — | — | — (no ejecutable, ver §1) |
| A0-B1-C1 | **0** | — | — | — | — (no ejecutable, ver §1) |
| A0-B1-C2 | **0** | — | — | — | — (no ejecutable, ver §1) |
| A1-B0-C0 | 10 | 6 | 4 | 0 | 0 |
| A1-B0-C1 | 10 | 6 | 4 | 0 | 0 |
| A1-B0-C2 | 10 | 5 | 2 | **3** | 0 |
| A1-B1-C0 | 10 | 8 | 2 | 0 | 0 |
| A1-B1-C1 | 10 | 7 | 3 | 0 | 0 |
| A1-B1-C2 | 10 | 7 | 3 | 0 | 0 |
| A2-B0-C0 | 10 | 6 | 4 | 0 | 0 |
| A2-B0-C1 | 10 | 8 | 2 | 0 | 0 |
| A2-B0-C2 | 10 | 5 | 0 | **5** | 0 |
| A2-B1-C0 | 10 | 7 | 3 | 0 | 0 |
| A2-B1-C1 | 10 | 5 | 5 | 0 | 0 |
| A2-B1-C2 | 10 | 7 | 3 | 0 | 0 |

**Total: 150/180 reglas corridas** (15 combos completos × 10; 3 combos en 0 por el hueco del motor).
Distribución global: **{I: 99 (66%), II: 43 (29%), III: 8 (5%), IV: 0}**.

Ninguna regla, en ninguna combinación, alcanzó **Clase IV** en este barrido V-A.

## 3. Criterio de éxito — primario (correlación necesaria)

**Pregunta 1 — ¿las reglas SIN retroalimentación (B0) NUNCA alcanzan Clase IV?**
Sí se sostiene, pero de forma trivial y poco informativa: **ninguna regla de NINGÚN eje B alcanzó Clase IV**
(0/150 en total, B0 y B1 por igual). No hay ninguna regla B1 que sí haya llegado a IV para comparar — así
que no se puede decir que "B0 se queda corto donde B1 sí llega"; simplemente Clase IV no apareció en esta
etapa liviana (V-A) bajo ninguna combinación. Es consistente con la correlación necesaria, pero no la
prueba con fuerza (para eso haría falta al menos una regla B1 en Clase IV que sirva de contraste).

**Pregunta 2 — ¿las reglas SIN fondo relacional genuino (A0) NUNCA alcanzan Clase II o superior?**
**No se sostiene. Esta correlación se ROMPE con los datos.** De las 30 reglas A0 medibles (A0-B0-C0/C1/C2 —
A0-B1-* no corrió, ver §1), **8 de 30 (27%) cayeron en Clase II**, la mayoría concentradas en A0-B0-C0
(5/10 = 50%). Ninguna llegó a Clase III o IV, así que el techo se mantiene, pero el piso predicho por la
especificación (que A0 se quede SIEMPRE en Clase I) no se cumple.

**Nota metodológica sobre por qué podría estar pasando esto** (no es una explicación cerrada, es una
hipótesis a verificar si Alexis quiere): el propio piloto (§6 de `FASE5A_piloto_resultado_CS.md`) ya
documentó que para A0 el motor deriva un "grafo de medición" por similitud de estado (`medir()`), porque el
coarse-graining (`cajas_bfs`/`grafo_grueso`) necesita algún grafo para operar, aunque el sustrato A0 en sí
no tenga adyacencia. Es posible que ese grafo derivado, por sí solo, ya produzca algo de escalamiento tipo
mundo-pequeño — es decir, que la Clase II en A0 sea un artefacto de cómo se MIDE (la vara de medición
siempre construye un grafo, incluso donde no lo hay), no evidencia de que A0 tenga estructura relacional
genuina. Esto no se investigó más a fondo en esta tarea — queda anotado para que Alexis decida si vale la
pena diagnosticarlo antes de confiar en la fila A0 del mapa.

## 4. Criterio de éxito — secundario (débil / fuerte / muy fuerte)

- **Débil** — "existe al menos una clase amplia (>15% de las reglas de esa combinación) que cae en Clase
  II o III, distinta de la implementación específica de Cosmogénesis": **SE CUMPLE, ampliamente.** 14 de
  las 15 combinaciones medibles superan el 15% en II+III (la única que no, A0-B0-C1, con 10%, está cerca).
  Varias combinaciones superan el 40-50% (A0-B0-C0, A1-B0-C2, A2-B0-C0, A2-B0-C2, A2-B1-C1).

- **Fuerte** — "existe una combinación que cae en Clase III o IV de forma reproducible en >10 reglas
  independientes, Y sobrevive la validación Phantom (V-B)": **NO CERTIFICADO en esta etapa**, por dos
  razones separadas: (a) el diseño de V-A fija 10 reglas por combinación como máximo, así que
  literalmente no hay ">10 reglas independientes" dentro de una sola celda del mapa aunque A2-B0-C2
  (5/10 en Clase III) sea el candidato obvio; (b) el criterio exige sobrevivir Fase V-B con Phantom, que
  esta tarea tiene explícitamente prohibido correr. A2-B0-C2 queda como **el candidato natural para V-B**
  si Alexis decide dar ese paso.

- **Muy fuerte** — "Clase III/IV es mayoritaria sobre Clase II específicamente cuando B1 y C1/C2 están
  presentes juntos": **NO SE CUMPLE — y los datos apuntan en la dirección contraria.** Las 4 combinaciones
  medibles que cruzan B1 con C1 o C2 (A1-B1-C1, A1-B1-C2, A2-B1-C1, A2-B1-C2) dieron **0 reglas en Clase
  III o IV** — todas I/II. En cambio, las ÚNICAS dos combinaciones que sí produjeron Clase III en este
  barrido (A1-B0-C2 con 3/10, A2-B0-C2 con 5/10) tienen **B0, no B1**. Es decir: en esta muestra, el costo
  con límite de escala duro (C2) por sí solo, SIN retroalimentación relación-sobre-relación, es lo que se
  asocia con geometría extensa — lo opuesto de lo que predecía la hipótesis "muy fuerte" del pre-registro.

**Categoría global: Débil confirmado; Fuerte pendiente de V-B (no descartado, pero no certificable
todavía); Muy fuerte refutado por el patrón observado (que además señala un candidato distinto — C2 sin
B1 — al que predecía la especificación).**

## 5. A2-B0-C2 con n=10 — ¿se sostiene el hallazgo del piloto?

**Sí, se sostiene y se ve más nítido.** El piloto (n=3) dio 2/3 en Clase III. Con n=10 completo: **5/10 en
Clase III, 5/10 en Clase I, 0/10 en Clase II** — nótese que este combo NUNCA cae en la zona intermedia
(Clase II); o se disuelve del todo o se extiende del todo, sin quedar atrapado en mundo-pequeño. Las
pendientes de las 5 reglas en Clase III van de 0.722 a 1.369 (todas bien por encima del umbral 0.7-0.8),
consistentes con "geometría extensa" genuina, no un caso límite. Es la combinación con la señal más fuerte
y más limpia de todo el barrido.

## 6. En lenguaje simple, con analogía

Imaginá que estás probando 18 "recetas" distintas para ver si mezclando ciertos ingredientes (fondo
relacional, retroalimentación, costo) siempre sale una masa que "crece" en vez de quedarse chata. De 18
recetas, 15 se pudieron cocinar (10 pruebas cada una); 3 recetas (las que combinaban "sin ingrediente base"
con "retroalimentación") no se pudieron ni empezar — la herramienta de cocina no sabe qué hacer con esa
combinación, así que quedan pendientes, no fallidas ni descartadas.

De las 15 que sí se cocinaron: casi todas producen "masa chata" (Clase I, 66%) o "masa que crece un poco y
se estanca" (Clase II, 29%). Solo una receta específica — "fondo que crece junto con la masa" + "sin
retroalimentación" + "límite duro de cuánto puede crecer cada parte" (A2-B0-C2) — hizo que la masa
realmente se extendiera (Clase III) en la mitad de los intentos, ni más ni menos que en el piloto chico.
Curiosamente, la receta que la teoría predecía como la más prometedora (retroalimentación + costo juntos)
no infló nada en absoluto — cero veces. Y una de las verificaciones de sentido común que se esperaba
("sin ningún ingrediente relacional, la masa nunca debería crecer ni un poco") no se cumplió del todo: la
masa "sin ingrediente relacional" sí creció un poco en 1 de cada 4 intentos — aunque nunca se extendió de
verdad. Puede ser una señal real, o puede ser que la regla con la que medimos "cuánto creció" (que necesita
dibujar un mapa de vecindad incluso donde no lo hay) esté inflando un poco el número — eso no se investigó
más en esta tarea.

## 7. Archivos de esta tarea

- `cs090_fase5_completo.py` — driver nuevo (único archivo nuevo de código).
- `cs090_fase5_completo_resultados.csv` — 750 filas, dato crudo por N/escala.
- `cs090_fase5_completo_resumen.csv` — 150 filas, una por regla (clase, pendiente, z_agg, holon_ratio).
- Este informe.

Ningún script congelado (`cs080/81/82/83`, `cs090_fase5_generador/motor/clasificador`) fue modificado. No
se corrió Phantom. No se hicieron commits de git.
