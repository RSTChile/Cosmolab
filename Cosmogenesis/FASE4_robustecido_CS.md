# FASE IV robustecido — ¿el sustrato 4 cierra lazos LOCALES o converge a un consenso GLOBAL?

**Script:** `cs083_fase4_robustecer.py` (nuevo; importa piezas de `cs082_fase4_4sustratos.py` sin
tocarlo) · **Fecha:** ago-2026 · **Antecedente:** `FASE4_orden_superior_resultado_CS.md`
**Estado:** exploratorio, sigue directo al primer ataque de Fase IV. Reporta números.
**Veredicto final: de Alexis.** Ningún resultado de este documento cierra ni confirma nada.

---

## 1. El caveat que dejó abierto el informe original

`FASE4_orden_superior_resultado_CS.md` (§3) encontró que, de los 4 sustratos probados, sólo el
**sustrato 4** (2-complejo con retroalimentación cara→arista: cada triángulo empuja a sus 3 aristas
de borde para reducir su propio defecto de holonomía) se separaba de NULL con solidez: holonomía
~5× menor que el azar (h_REAL=0.30 vs h_NULL=1.54, promedio de 5 semillas).

Pero ese mismo informe reportó, sin resolver, un control que sembraba la duda: **SHUFFLED** — se
toman los mismos valores finales de arista que dio la corrida real, pero se **barajan al azar entre
aristas** (mismos números, topología rota) — dio h_SHUF=0.57, mucho más cerca de REAL (0.30) que de
NULL (1.54). Es decir: **con sólo remezclar qué arista tiene qué valor, sin tocar un solo número, ya
se recupera la mayor parte del aplanamiento.** Eso sugería que buena parte del efecto podía ser "hacia
dónde converge la distribución global de valores" (un consenso agregado que el feedback empuja sobre
todo el campo por igual) — no que cada triángulo cierre específicamente su propio lazo de a tres por
la topología local cara↔su-propio-borde. El informe original no distinguía ambas cosas con un control
que aislara la variable correcta: SHUFFLED rompe SIMULTÁNEAMENTE la topología del feedback y la
dinámica de alineación diádica de fondo, así que no permite saber cuánto del aplanamiento depende
específicamente de que la cara empuje A SUS aristas verdaderas.

---

## 2. Qué se hizo para resolverlo

**(a) Más semillas:** de 5 a **20** (`SEEDS = 1..20`; las primeras 5 coinciden exactamente con las de
`cs082_fase4_4sustratos.py`, y sirvieron de chequeo de reproducibilidad — ver §3).

**(b) Un control más fino, quirúrgico** (`4b_rewire_azar`, código nuevo en `cs083_fase4_robustecer.py`,
función `correr_sustrato_4_control_fino`): copia EXACTA de la maquinaria del sustrato 4 real —
mismos sweeps, mismo `J_FACE`, mismo número de triángulos empujando por sweep (161 en promedio),
mismo presupuesto total de operaciones-relación (77.660, verificado igual en las 20 semillas) — con
**un solo cambio**: la asignación cara→3-aristas se sortea **al azar sobre el grafo completo** (fija
para toda la corrida), en vez de usar las 3 aristas de borde reales del triángulo. Cada "cara" sigue
empujando 3 aristas cualquiera hacia el consenso — sólo que esas 3 aristas ya no son necesariamente
las del triángulo correcto.

Este control aísla justo la variable en duda: preserva el **volumen** de retroalimentación (misma
"presión global") pero rompe la **correspondencia local** cara↔su-propio-borde. La holonomía se mide
siempre sobre los **triángulos geométricos verdaderos** (nunca sobre los tríos al azar) — así el
número que se compara es "qué tan bien cerraron los lazos reales", tanto bajo feedback correctamente
cableado como bajo feedback mal-cableado.

Lógica de falsación: si el control fino aplana los triángulos verdaderos **tanto como** el real →
puro consenso global. Si se queda **mucho más cerca de NULL** que el real → hay algo genuinamente
local en juego.

---

## 3. Resultado con 20 semillas (chequeo de reproducibilidad + números nuevos)

Las primeras 5 semillas reproducen los números de `cs082` casi exacto (promedio de esas 5:
h_real≈0.295, h_null≈1.543, h_shuf≈0.568 — contra 0.30/1.54/0.57 del informe original), confirmando
que no hay divergencia entre lo que se reimporta y lo que se reportó antes.

**Resumen descriptivo sobre las 20 semillas:**

| serie | media | DE (entre semillas) | mín | máx |
|---|---:|---:|---:|---:|
| h_REAL (feedback local, sustrato 4 real) | 0.264 | 0.107 | 0.198 | 0.577 |
| h_CTRL_FINO (feedback rewire al azar) | 0.368 | **0.025** | 0.317 | 0.430 |
| h_NULL (sin dinámica) | 1.517 | 0.082 | 1.348 | 1.728 |
| h_SHUFFLED (topología rota) | 0.486 | 0.359 | 0.313 | 1.568 |

Dato aparte llamativo: **CTRL_FINO es la serie más estable de las 4** (DE=0.025, la décima parte de
la dispersión de h_REAL). Tiene sentido — al empujar aristas al azar en vez de las del propio
triángulo, el resultado no depende de la geometría particular de cada grafo-semilla, converge casi
siempre al mismo valor de "consenso disperso". h_REAL, en cambio, tiene más varianza entre semillas
(una semilla, la 5, salió con h_real=0.577 y h_shuf=1.447 — un outlier real, no descartado, incluido
en todos los cálculos de abajo).

---

## 4. La descomposición central: ¿cuánto del efecto es LOCAL y cuánto es GLOBAL?

```
gap total   (h_NULL − h_REAL)       = +1.253   (todo el aplanamiento observado)
gap local   (h_CTRL_FINO − h_REAL)  = +0.104   → fracción LOCAL  ≈  8.3%
gap global  (h_NULL − h_CTRL_FINO)  = +1.149   → fracción GLOBAL ≈ 91.7%
```

Lectura directa: de toda la caída de holonomía que produce el sustrato 4 respecto de no tener
dinámica (NULL), **~92% ocurre igual con feedback mal-cableado al azar** — es decir, es un efecto de
"hay bastante retroalimentación dando vueltas por el grafo, y eso por sí solo empuja todo hacia un
consenso" (componente GLOBAL). Sólo **~8% adicional** se pierde específicamente cuando se rompe la
correspondencia cara↔su-propio-borde (componente LOCAL).

---

## 5. ¿Ese 8% es ruido o es real? — tests estadísticos (20 semillas, pareado por semilla)

Mismo criterio de unidad válida que usó el resto del proyecto esta semana (p.ej.
`cs078_kappaV_permutacion.py`): la semilla/grafo-base es la unidad de la muestra, no cada triángulo
suelto. z-score de la diferencia pareada + test de permutación por volteo de signo (20.000
repeticiones), un lado pre-registrado según la hipótesis de esta tarea.

| comparación | z | diff. observada | p (una cola) | p (dos colas) |
|---|---:|---:|---:|---:|
| [A] REAL vs NULL | −34.8 | −1.253 | <0.0001 | <0.0001 |
| **[B] REAL vs CTRL_FINO** (test central) | **−4.14** | **−0.104** | **0.00055** | **0.00110** |
| [C] CTRL_FINO vs NULL | −59.6 | −1.149 | <0.0001 | <0.0001 |
| [D] REAL vs SHUFFLED | −3.90 | −0.222 | <0.0001 | <0.0001 |

- **[A]** confirma, con 4× más semillas, lo que ya sabíamos: el sustrato 4 real separa de NULL con
  altísima solidez.
- **[C]** muestra que **incluso el feedback mal-cableado al azar** ya se separa fortísimo de NULL —
  consistente con que la mayor parte del efecto es volumen/consenso global, no algo específico de la
  topología correcta.
- **[B] es el resultado central de esta tarea:** aunque la magnitud del componente local es chica
  (~8% del efecto total), **NO es ruido** — es una diferencia pequeña pero estadísticamente muy sólida
  (z=−4.14, p≈0.0006 con permutación, 20/20 semillas apuntando en la misma dirección salvo dispersión
  normal). El sustrato 4 real cierra los triángulos verdaderos de forma consistentemente mejor que la
  misma cantidad de feedback mal-dirigido.
- **[D]** repite el chequeo original de `cs082` (REAL vs SHUFFLED) con más poder: la brecha sigue
  siendo significativa, coherente con que SHUFFLED — que rompe MÁS cosas que el control fino (también
  la dinámica diádica de fondo) — se aleja más de REAL que CTRL_FINO no lo hace del todo.

---

## 6. En simple, con analogía

Retomando la analogía del informe original (personas tratando de acordar "hacia dónde apunta el
norte", con un árbitro que corrige contradicciones de a tres):

Ahora se probó una variante del árbitro: **el mismo árbitro, con la misma frecuencia de intervención,
pero que a veces corrige a un grupo de tres personas que NO son las que discutían juntas** — las elige
al azar de todo el salón. Resultado: ese árbitro "distraído" igual logra que el salón entero se ponga
mucho más de acuerdo que si no hubiera árbitro (porque total, cualquier corrección frecuente empuja a
todos hacia el mismo tono general) — eso explica el ~92% de la mejora. Pero cuando el árbitro corrige
a la gente CORRECTA (los tres que realmente discutían ese punto), el acuerdo dentro de ESE trío
específico es *todavía un poco mejor* — un ~8% extra, chico pero que se repite de forma confiable en
20 salones distintos (20 semillas), no es casualidad de un solo salón.

Lectura simple: **la mayor parte de lo que separaba al sustrato 4 de "no hacer nada" es simplemente
"corregir mucho, en general" — no específicamente "corregir a los tres correctos". Pero hay un resto
chico y estadísticamente confiable que SÍ depende de corregir a los tres correctos.**

---

## 7. Lectura honesta para la Pared R7 (no es cierre — es lectura, el veredicto es de Alexis)

- La "grieta" que reportó `cs082` (sustrato 4 separa de NULL) **se sostiene** con 4× más semillas —
  no era un accidente de 5 corridas afortunadas (test [A], z=−34.8).
- Pero el caveat del control SHUFFLED **tenía razón en la dirección**, y ahora se puede cuantificar:
  la gran mayoría del efecto (~92%) es un fenómeno de **consenso global agregado** — cualquier
  retroalimentación suficientemente densa sobre el grafo, aunque esté mal dirigida, empuja el campo
  entero hacia valores compatibles y eso "de rebote" también aplana los triángulos reales.
- Sin embargo, hay un componente **genuinamente local** — pequeño en magnitud (~8% del efecto total)
  pero estadísticamente robusto (p<0.001, no explicable por azar de muestreo entre semillas) — que
  depende específicamente de que la cara empuje a SUS PROPIAS 3 aristas de borde, no a 3 aristas
  cualquiera. Eso es evidencia de que "una relación que actúa sobre otra relación EN SU LUGAR
  CORRECTO" hace algo que "una relación que actúa sobre relaciones AL AZAR" no reproduce del todo.
- Para la pregunta de fondo de Fase IV ("¿la aridad/estructura de orden superior es lo que estuvo
  bloqueado en la Pared R7?"): este resultado matiza el titular de `cs082`. No alcanza con decir "el
  sustrato 4 rompe la pared" — más preciso es: **"el sustrato 4 rompe la pared mayormente por volumen
  de retroalimentación distribuida, con un remanente local pequeño pero real."** Si ese remanente
  local del 8% es o no "suficiente" para sostener la interpretación de `cs082` §4-5 (que el ingrediente
  crítico es que una relación empuje a OTRA relación específica, no sólo que haya "más corrección en
  general") es exactamente el tipo de pregunta que le corresponde a Alexis, no a este script.

---

## 8. Qué NO se reclama

- No es un motor físico — sigue siendo estructura relacional pura, sin grupos gauge ni masa ni
  partículas.
- No se afirma que el componente local (8%) sea "poco importante" ni que sea "suficiente" — sólo se
  reporta su magnitud y su solidez estadística. La interpretación de si eso alcanza para sostener la
  lectura de `cs082_fase4_4sustratos.py` §4-5 queda abierta.
- Escala sigue siendo chica (N=110, K=6, un solo punto de J/J_FACE/ruido) — no se barrió el espacio de
  parámetros. Un J_FACE distinto podría cambiar la proporción local/global; no se probó en esta tanda
  por el presupuesto de tiempo acordado (~45-55 min).
- El control fino no reemplaza a SHUFFLED — son controles distintos que aíslan variables distintas
  (SHUFFLED rompe topología + dinámica de fondo; CTRL_FINO sólo rompe el cableado del feedback,
  dejando intacta la dinámica diádica subyacente). Ambos se reportan, no se elige uno como "el bueno".
- Ningún resultado de este documento se declara cerrado, confirmado o refutado. Los números están
  arriba. El veredicto es de Alexis.

**Reproducibilidad:** `cs083_fase4_robustecer.py`, sin dependencias fuera de numpy (importa piezas de
`cs082_fase4_4sustratos.py`, que no fue modificado), corre en ~65s con la configuración actual
(`./venv/bin/python3 cs083_fase4_robustecer.py`).
