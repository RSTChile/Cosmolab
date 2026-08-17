# PRE-REGISTRO — Etapa 7 / Track A: segregación gravitacional por masa intrínseca

**Timestamp de escritura de este documento:** 2026-07-23 05:02 (hora local, -04:00), ANTES de
escribir/correr el motor de producción `etapa7_trackA_segregacion_01_engine.py`.

**Paso previo (ya corrido, resultado en `results/etapa7_trackA_segregacion/trackA_00_heterogeneidad_mass_proxy.json`):**
heterogeneidad de `mass_proxy` confirmada — CV pooled ≈ 0.35, rango ~4.6–18.5 sobre 239 átomos
estables (10 semillas), distribución con cola larga a la derecha. Hay varianza real → se procede
al diseño de la prueba de segregación.

---

## 1. Pregunta

¿Los átomos con `mass_proxy` (intrínseco, calculado del campo Φ propio del átomo, SIN linaje)
alto migran preferentemente hacia el centro de sus grupos gravitacionalmente ligados a lo largo
de la ventana E4, de forma que esto **NO** ocurra (o sea sustancialmente más débil) cuando la
fuente de la fuerza se baraja (SHUFFLE)?

Esto es un fenómeno **exclusivo de la gravedad real** (fricción dinámica / segregación de masa):
sin una relación causal masa→fuerza→posición, no hay razón para que el átomo intrínsecamente
más "pesado" (Φ propio) termine sistemáticamente más cerca del centro de su grupo.

## 2. Prohibiciones (cumplidas por diseño, no por disciplina post-hoc)

El motor de producción **no calcula ni importa** `co_member_score`, `n_long_co_pairs`, ni
`fusion_events`, ni ninguna función de ellos. No hay trackers de linaje en el código. El único
insumo de "masa" es `mass_proxy` local (idéntico a v6, línea ~189): `max(sum_phi,1e-6)*(1+f_core)`.
La única entrada de posición es la dinámica N-body sobre las posiciones reales de los átomos.

## 3. Diseño

### 3.1 Población fija al entrar a E4

Al primer paso en que `frozen and step >= grav_start` (idéntico umbral que v6:
`GRAV_START_FRAC=0.65`, `FREEZE_TNORM=0.40`, `RHO_FREEZE=0.05`), se toma el conjunto de átomos
**ya estables** (`age >= PERSIST_STEPS=4`) en ese instante como población FIJA para el resto de
la corrida. No se agregan átomos nuevos ni se re-detectan por campo después de ese punto — esto
es una decisión de diseño pre-registrada (no un ajuste post-hoc): evita el problema observado en
v3–v6 donde la dinámica N-body se reiniciaba cada vez que cambiaba el conteo de átomos estables,
impidiendo que la fricción dinámica se acumulara.

- `mass_intrinsic[id]` = `mass_proxy` del átomo en ese instante de congelamiento. **Nunca se
  vuelve a recalcular** (evita circularidad masa↔posición vía re-medición de campo influida por
  densidad local).
- posición inicial = centroide de campo en ese instante.
- Si una semilla no alcanza `len(población) >= 3` en ese instante, se descarta como "sin datos"
  para esa semilla (reportado, no forzado).

### 3.2 Dinámica E4 (idéntica físicamente a v6, sin reset de posición)

Para el resto de los pasos (`pasos - grav_start`), se integra `nbody_step` (mismo softening=1.2,
`DT_NB=0.35`, `FORCE_CUTOFF=8.0`, misma fórmula de fuerza) en modos:

- **REAL**: fuerza atractiva normal, `masses[i]*masses[j]`.
- **SHUFFLE**: la masa FUENTE se permuta una única vez al inicio de E4 (permutación fija por
  semilla, aplicada a todo el resto de la ventana) — igual que v6/v5/v4 en su definición de
  SHUFFLE. La masa RECEPTORA (inercia) no cambia. Esto rompe la correlación identidad↔atracción
  ejercida sin cambiar la intensidad total del campo de fuerzas.
- **OFF**: `G=0`, sin dinámica (diagnóstico: no debería haber segregación ni con REAL ni SHUFFLE
  porque no hay movimiento).
- **INVERT**: signo de fuerza invertido (repulsión) — diagnóstico: si REAL segrega hacia el
  centro, INVERT debería, si acaso, segregar hacia AFUERA (correlación positiva o nula).

### 3.3 Agrupamiento final y métrica

Al ÚLTIMO paso de la ventana E4, se calculan grupos por proximidad (union-find, radio de enlace
`GROUP_LINK_R=4.5`, idéntico a v6). Para cada grupo con `tamaño >= 3` átomos:

- se calcula el centroide toroidal del grupo,
- se calcula `dist_i` = distancia toroidal de cada átomo del grupo a ese centroide,
- se calcula el coeficiente de correlación de Pearson `r_grupo` entre `mass_intrinsic` y `dist_i`
  **dentro de ese grupo** (no pooled entre grupos, para no confundir escala de grupo con masa).

Por semilla y modo: `r_seed_modo` = promedio NO ponderado de `r_grupo` sobre todos los grupos
válidos (tamaño ≥ 3) de esa semilla. Si una semilla no tiene ningún grupo válido en un modo dado,
esa semilla queda "sin dato" para ese modo (excluida del numerador y denominador de la tasa, y
reportada explícitamente como tal — no se cuenta como fallo ni como éxito).

También se reporta la correlación de Spearman (rango) como robustez adicional (no gatea el
veredicto, solo diagnóstico de forma funcional).

## 4. Criterio de PASS (fijado AQUÍ, antes de correr producción)

Por semilla, **"seed_win" = True** si y solo si:

```
r_REAL <= -0.15          (REAL muestra segregación negativa no trivial: más masa, más cerca del centro)
  AND
r_REAL <= r_SHUFFLE - 0.10   (REAL es más segregante que SHUFFLE por un margen de seguridad)
```

- `rate = (# seed_win) / (# semillas con dato válido en REAL y SHUFFLE)`.
- **PASS** si `rate >= 0.55` sobre **al menos 10 semillas válidas** del set estándar
  `(7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321)`.
- **PARTIAL** si `0.30 <= rate < 0.55`.
- **FAIL** si `rate < 0.30`.
- **INCONCLUSO** si menos de 6 semillas tienen dato válido (muestra insuficiente).

Diagnóstico adicional (no gatea el veredicto):
- `rate_off_quiet`: fracción de semillas donde `|r_OFF| < 0.15` (se espera alta — sin dinámica no
  debería haber señal).
- `rate_invert_not_negative`: fracción de semillas donde `r_INVERT >= -0.05` (se espera alta si
  el mecanismo es específicamente gravitacional-atractivo).

Barrido de `G_GRAV`: se reporta `rate` por valor de G sobre una submuestra de 4 semillas
(`2025, 42, 777, 3141`, el mismo subconjunto usado como "smoke" en v5/v6) para valores
`G_GRAV ∈ {0.05, 0.10, 0.20, 0.30, 0.45}`. Esto es exploratorio/diagnóstico: si el efecto
aparece solo en un punto aislado de G y no en un rango, se reporta como evidencia débil, no como
PASS.

**Regla anti-Shannon explícita:** una vez fijados estos números (−0.15, −0.10, 0.55, el set de
semillas, el rango de G), NO se modifican después de ver el resultado de producción. Si el
resultado es FAIL o INCONCLUSO, se reporta como tal.

## 5. Plan de ejecución

1. Smoke test: 3–4 semillas (`2025, 42, 777`), `pasos` reducido (200) — solo para verificar que
   el código corre sin errores y que hay señal medible en absoluto (no para ajustar umbrales).
2. Si el smoke no revienta y produce números con sentido (grupos ≥3, correlaciones finitas), se
   corre producción completa: 10 semillas, `pasos=400` (igual que v6), G=0.20 base.
3. Barrido de G sobre las 4 semillas fijas, `pasos=400`.
4. Reporte final con números crudos por semilla, veredicto, y confirmación explícita de que no
   se usó `co_member_score` / `n_long_co_pairs` / `fusion_events`.

## 6. Archivos

- Motor: `codigo/suite_epocas_masa/etapa7_trackA_segregacion_01_engine.py` (nuevo, no edita v1–v6).
- Resultados: `results/etapa7_trackA_segregacion/`.
