# PROTOCOLO — Track C: diagnóstico y reparación de E_mutual

**Timestamp de pre-registro:** 2026-07-23 05:01 -04 (ANTES de correr ningún barrido de este documento)
**Autor:** agente de diagnóstico (sesión Track C), a pedido del director Alexis López Tapia
**Motor base (NO se edita):** `codigo/suite_epocas_masa/suite_epocas_masa_v6_mass_linaje.py` (líneas 249-299 `nbody_step`, 544-570 bloque `E_mutual`)
**Hallazgo que se investiga:** en `suite_epocas_masa_v4` (10 semillas, G=0.20): `mutual_bind` REAL ≈ 32.6, SHUFFLE ≈ 43.9 → **R/S ≈ 0.74 < 1** (SHUFFLE gana). Confirmado también en v6 (semilla 42 aislada: mutual_bind REAL=74.90, coincide con el log v4 seed 42 = 74.903 → el mecanismo de E_mutual no cambió entre v4 y v6, solo dejó de gatear el veredicto final).

**Definición de `mutual_bind` en v6 (la que se audita):** `mutual_bind = max(0, -min_t(E_mutual(t)))` sobre los pasos E4, donde `E_mutual(t) = Σ_pares (edad≥5) [-0.5·G·(m_i·src_j + m_j·src_i)/r_soft]`. Es decir: **mínimo temporal (extremo) de una suma sobre pares que llevan ≥5 pasos consecutivos próximos (gate de persistencia)**.

---

## Disciplina anti-Shannon (compromiso antes de ver datos)

- El observable reparado sigue siendo energía física de pares (posiciones + masas + G), **nunca** función de `co_member_score`, `n_long_co_pairs` ni `fusion_events`.
- Semillas fijas: 7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321.
- Umbral de "arregla el instrumento" fijado AQUÍ, antes de correr: **para una configuración dada, "resuelve" si mean(R/S) sobre las semillas corridas ≥ 1.15 Y rate(R/S_seed > 1.0) ≥ 0.60** (6/10). Umbral de R/S coherente con `COMEM_VS_SHUFFLE_MIN=1.15` ya usado en el proyecto (v5/v6).
- Protocolo de escalado: cada barrido corre primero en **smoke** (subconjunto de semillas: 7, 42, 99, 777) sobre la grilla completa de parámetros; si **algún punto de la grilla** exhibe señal de inversión (R/S medio en smoke ≥ 1.05, o visiblemente > 1 en ≥3/4 semillas smoke), se **escala esa configuración** a las 10 semillas completas. Puntos sin señal en smoke no se escalan (ahorro de cómputo, no p-hacking: la grilla completa ya está fijada aquí, no se agregan puntos nuevos después de ver resultados).
- Prohibido: bajar el umbral 1.15 tras ver datos; ajustar parámetros solo en las semillas que fallan; usar `co_member`/`fusion`/`n_long_co` para "arreglar" el signo.

---

## H1 — Ventana de gravedad demasiado corta

**Mecanismo hipotetizado:** con `GRAV_START_FRAC=0.65` y `pasos=400`, la gravedad actúa ~140 pasos. Si eso no basta para que la segregación mass-dependiente (que exige que el REAL "sepa" distinguir masas reales) domine sobre la dinámica de corto plazo (donde SHUFFLE, al azar, puede generar pares fuertes por asignación aleatoria de fuente), más tiempo debería favorecer a REAL.

**Barrido pre-registrado (grilla fija):**
- `GRAV_START_FRAC` ∈ {0.35, 0.50, 0.65 (baseline), 0.80} con `pasos=400`
- `pasos` ∈ {400 (baseline), 800} con `GRAV_START_FRAC=0.65`
- combo ventana máxima: `GRAV_START_FRAC=0.35`, `pasos=800`

**Predicción si H1 correcta:** R/S de `mutual_bind` sube monótonamente (o al menos no baja) al aumentar la ventana de gravedad (más `pasos` a igual `GRAV_START_FRAC`, o `GRAV_START_FRAC` menor a igual `pasos`), y al menos un punto de la grilla cruza R/S≥1.15 con rate≥0.60.

**Predicción si H1 incorrecta:** R/S se mantiene <1 (o fluctúa sin tendencia) en toda la grilla — el problema no es de tiempo de exposición.

---

## H2 — Cutoff/softening demasiado permisivos

**Mecanismo hipotetizado:** `FORCE_CUTOFF=8` en una caja `L=28` es ~29% del lado de la caja — "todo atrae a todo" casi sin importar identidad, diluyendo cualquier señal de masa real. `SOFTENING=1.2` puede además aplanar el pozo de potencial a corta distancia, borrando la diferencia entre pares fuerte/débilmente ligados.

**Barrido pre-registrado (grilla fija):**
- `FORCE_CUTOFF` ∈ {3, 5, 8 (baseline), 12} con `SOFTENING=1.2`
- `SOFTENING` ∈ {0.4, 1.2 (baseline), 3.0} con `FORCE_CUTOFF=8`

**Predicción si H2 correcta:** al restringir el alcance (cutoff más chico) y/o endurecer el pozo (softening más chico), R/S sube porque solo sobreviven pares realmente próximos, donde la masa real (no la barajada) debería dominar la dinámica de formación. Softening muy grande (3.0) debería empeorar R/S (predicción de control interno).

**Predicción si H2 incorrecta:** R/S no responde de forma sistemática al cutoff/softening — el diluvión de "todo atrae a todo" no es la causa.

---

## H3 — Correlación espacial heredada (no gravitatoria)

**Mecanismo hipotetizado:** las posiciones de los átomos al entrar a E4 vienen del mismo proceso de congelamiento de campo (E0-E3) en las 4 ramas REAL/OFF/SHUFFLE/INVERT — el SHUFFLE de v6 solo permuta la **fuente de masa** (`perm`), nunca las **posiciones** iniciales. Si la proximidad ya estaba en los datos (estructura heredada del campo Φ, agrupada por el mismo camino determinista/estocástico-pero-idéntico de semilla), entonces SHUFFLE hereda la MISMA estructura espacial que REAL y el único grado de libertad que cambia es la fuente de masa — insuficiente para "romper" la proximidad heredada, y el ruido introducido por el shuffle de masa puede, además, generar pares aleatoriamente sobre-ligados (ver H4).

**Diseño del control (nuevo, no en v6):** variante `posrandom_e4_entry=True` — al momento en que los átomos estables entran por primera vez a la fase N-body de E4 (mismo punto donde v6 fija `nb_pos = pos.copy()`), se reemplazan las posiciones heredadas por posiciones **uniformemente aleatorias** en `[0, L)²` (mismo número de átomos, mismas masas/ids, mismo L — se preserva la distribución marginal pero se rompe la estructura espacial heredada del campo). A partir de ahí la dinámica N-body sigue igual (REAL/SHUFFLE).

**Predicción si H3 correcta:** con posiciones aleatorizadas, el R/S de `mutual_bind` cambia sustancialmente respecto al baseline heredado (idealmente sube ≥1.15 en REAL, porque ahora solo la masa real puede generar estructura, no una correlación heredada que SHUFFLE también aprovecha) — o al menos el patrón R/S<1 desaparece.

**Predicción si H3 incorrecta:** el R/S con posiciones aleatorizadas es indistinguible (dentro de ruido) del R/S con posiciones heredadas — la herencia espacial no es la causa.

**Semillas:** las 10 estándar directamente (el costo de este control es 1 config × 10 semillas × 2 modos, barato) — no requiere smoke previo dado que es un único punto, no una grilla.

---

## H4 — Auditoría de signo/normalización y sesgo de selección por supervivencia

**Dos sub-hipótesis, auditadas sobre las MISMAS corridas baseline (no requieren simulaciones nuevas — se recalculan post-hoc a partir de los pares registrados por paso):**

### H4a — Gate de persistencia (edad≥5) como sesgo de selección
`E_mutual` solo suma pares con ≥5 pasos consecutivos próximos. Bajo SHUFFLE, qué par recibe fuerza fuerte es esencialmente aleatorio (fuente de masa barajada); el subconjunto de pares que por azar quedan "sobre-ligados" y por eso sobreviven 5 pasos seguidos está **sesgado hacia el extremo alto** de la distribución de fuerza asignada al azar. Bajo REAL, el conjunto de pares persistentes refleja la red de fuerzas coherente (no un sorteo), pudiendo tener energías por-par más moderadas. Esto es un sesgo de selección (survivorship bias), no física real.

**Prueba:** recalcular `mutual_bind` con el gate de persistencia **quitado** (`E_mutual_instant`: suma sobre TODOS los pares próximos del paso, sin exigir edad≥5) y comparar R/S con la versión gateada (v6 original).

### H4b — Extremo temporal (mín sobre pasos) como estadístico ruidoso
`mutual_bind = max(0,-min_t(...))` toma el paso más extremo de toda la corrida E4 (~140-260 pasos), no un promedio. Un solo paso con alineación simultánea casual de pares fuertemente ligados (más probable bajo el ruido introducido por el shuffle de fuente) puede dominar el estadístico.

**Prueba:** recalcular usando **media temporal** (`mean_t`) en vez de mínimo, tanto para la versión gateada como la instantánea → 4 variantes totales por corrida: `min_gated` (=v6 original), `mean_gated`, `min_instant`, `mean_instant`. Adicionalmente, una 5ª variante normalizada por par: `mean_t(energía_por_par_promedio(t))` (saca el efecto de "más pares = suma mayor").

**Predicción si H4 correcta (a y/o b):** alguna de las variantes `mean_gated`, `min_instant`, `mean_instant` o `per_pair_mean` muestra R/S ≥ 1.15 con rate ≥ 0.60, mientras que `min_gated` (v6 original) se mantiene <1 — es decir, el signo negativo es un artefacto del estadístico extremo/gateado, no de la física de pares en sí.

**Predicción si H4 incorrecta:** todas las variantes (gated/instant × min/mean, y per-pair) mantienen R/S<1 de forma consistente — el problema no es de selección/estadístico, es más profundo (física de pares real no separa REAL de SHUFFLE en este juguete).

**Semillas:** las 10 estándar, calculadas post-hoc sobre las corridas baseline de H1/H3 (mismo costo computacional, cero corridas extra).

---

## Salidas esperadas de este protocolo

- `codigo/suite_epocas_masa/etapa7_trackC_mutual_engine.py` — motor instrumentado (copia+extensión de las funciones de v6 necesarias; v6 no se edita).
- `codigo/suite_epocas_masa/etapa7_trackC_mutual_run.py` — corredor de los 4 barridos (H1-H4).
- `results/etapa7_trackC_mutual/*.json` — resultados crudos por hipótesis, por semilla.
- `results/etapa7_trackC_mutual/RESUMEN_TRACKC_MUTUAL.md` — veredicto por hipótesis (RESUELVE / NO RESUELVE / PARCIAL) + veredicto global, escrito DESPUÉS de correr todo.

Este documento se sube tal cual quedó escrito antes de la primera corrida; cualquier desviación del plan (grilla, semillas, umbral) durante la ejecución se documentará explícitamente en el resumen final como desviación, no se edita este archivo retroactivamente.
