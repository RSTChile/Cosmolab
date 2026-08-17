# PROTOCOLO F3-2 — PRE-REGISTRO
## "Enfriamiento como consecuencia medida, no impuesta: T leída del estado"

**Fecha de escritura:** 2026-07-24 (ANTES de correr `F3_2_T_emergente_motor.py`; este
archivo se congela con este mtime — no se edita tras ver resultados, T3).

**Ejecutor:** CC, experimento F3-2 de la batería `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`
(Enfoque 3, experimento 2). Corre en paralelo con otros 23 experimentos, cada uno con su
propio prefijo. Este protocolo cubre EXCLUSIVAMENTE el prefijo `F3_2_`. No se toca ningún
archivo fuera de `codigo/BATERIA_FUNDAMENTOS/F3_2_T_emergente/`.

---

## 1. Pregunta

F3-1 (hermano de este experimento) mide si el GRADIENTE de un salto térmico diseñado se
suaviza al expandirse. F3-2 pregunta algo distinto y más literal: **¿la TEMPERATURA —leída
directamente del propio estado del campo (varianza/energía de sus fluctuaciones), sin
ningún término de enfriamiento puesto a mano— cae con `a` de forma EMERGENTE?** Y si cae,
¿sigue una ley de potencia T(a) ~ a^(−n), midiendo n sin imponerlo?

Diferencia deliberada con F3-1 (evita T2, "observable ≠ juez"): F3-1 usa un salto (step)
diseñado tipo tanh y mide su abruptancia máxima en una banda. F3-2 usa un campo de **ruido
puro de banda ancha** (sin ninguna estructura macroscópica sembrada) y mide su **varianza
espacial** y su **energía cuadrática de gradiente**. Es un termómetro distinto sobre un
campo distinto — no reutiliza ni la forma inicial ni el observable de F3-1.

---

## 2. Código base (NO se edita)

`Cosmogenesis-Web/codigo/CF2_estiramiento/CF2_estiramiento_motor.py` — leído completo.
Se reutiliza SOLO su infraestructura de reloj genético (`t_g → a = exp(H_EXP·t_g)`,
`H_EXP=6.0`, `dtg=1/399` heredado de `TEST_RHO_DISPERSION.py`, mismo mecanismo de
checkpointing markoviano de una única trayectoria muestreada en los `t_g(a)` objetivo) y
el patrón de brazos REAL vs `NULL_RHO_FIXED` (misma trayectoria de `a(t)`, difiere solo la
densidad/difusividad). **No se reutiliza** el perfil inicial (tanh/step) ni el observable
(abruptancia máxima de banda) — éstos son propios de F3-2, ver §3-4.

No se toca `CF2_estiramiento_motor.py`, `F3_1_estiramiento_motor.py` ni ningún otro
archivo existente.

---

## 3. Sustrato propio de F3-2

- Malla `L×L` con `L=64` (mismo tamaño que CF2/F3-1, por comparabilidad de escala de
  ruido; no se elige para favorecer un resultado — T1).
- **Condición inicial:** `T0(x,y) = σ_ruido · N(0,1)` — ruido gaussiano blanco puro,
  media 0, SIN ninguna estructura macroscópica sembrada (a diferencia de CF2/F3-1 que
  siembran un salto tanh). Este es el campo "caliente" cuya temperatura se lee.
- **Sin clipping.** CF2/F3-1 recortan `T∈[0,1]` porque su campo representa una fracción
  física acotada. Aquí el campo es una variable de fluctuación sin ese significado; recortar
  introduciría un sesgo NO LINEAL que dependería de `σ_ruido` y contaminaría la propia
  medición de varianza/energía que se está intentando leer (violaría T2: el recorte sería
  una manipulación del observable, no un rasgo físico independiente). Se documenta esta
  desviación explícita de la convención heredada, justificada arriba, ANTES de correr.
- **Difusión:** laplaciano isótropo de 4 vecinos, mismo operador y mismo `DT=0.25`,
  `N_SUB=2` que CF2 (heredado sin retocar, T1).
- **Forzamiento estocástico dinámico (además de la condición inicial):** en cada subpaso
  de difusión se inyecta ruido gaussiano de amplitud `forcing_amp`, escalado
  `sqrt(dt_sub)` (incremento tipo Wiener):
  `T ← T + (dt_sub)·D·∇²T + forcing_amp · sqrt(dt_sub) · N(0,1)`.
  `forcing_amp` está atado a la MISMA dilución que ya gobierna D (no es un parámetro
  libre nuevo elegido para dar un resultado — T1):
  - **REAL:** `forcing_amp(a) = σ_ruido · DYN_FRAC · sqrt(ρ/ρ0) = σ_ruido·DYN_FRAC·a^(−1.5)`
  - **NULL_RHO_FIXED:** `forcing_amp = σ_ruido · DYN_FRAC` (constante, ρ≡ρ0)
  - `DYN_FRAC = 0.1`, constante técnica fija (análoga a `DT`/`N_SUB`/`W0` heredados de
    CF2 — NO se barre, se fija y documenta aquí, antes de ver resultados).
  **Por qué esto no es un "baño externo" (no viola el espíritu de F3-6):** un baño
  externo (prohibido salvo como control explícito en F3-6) empuja el campo HACIA un
  valor objetivo fijo T_baño (término tipo Newton `∝(T_baño−T)`). Este forzamiento no
  tiene objetivo: es ruido de media cero, sin punto de anclaje externo — es agitación
  intrínseca cuya amplitud se ata a la MISMA física de dilución ya presente (rho), no una
  fuente de calor externa a temperatura fija. Es la perturbación DINÁMICA que exige T7
  ("perturbar la dinámica, no solo la semilla") — sin ella, la ecuación de difusión es
  lineal y un barrido de amplitud de condición inicial sólo re-escalaría la curva sin
  cambiar su forma (ver razonamiento completo abajo).

**Por qué se necesita forzamiento dinámico aquí (razonamiento pre-registrado):** la
ecuación de difusión es LINEAL. Sin forzamiento en cada paso, escalar `σ_ruido` en la
condición inicial sólo re-escala toda la trayectoria linealmente (la pendiente log-log y
la monotonicidad NO cambiarían con `σ_ruido` — sería una perturbación cosmética, no
dinámica, el mismo defecto T7 que F3-1 identificó en CF2). El forzamiento dinámico rompe
esa linealidad trivial: en el NULL (D≡D0 y forzamiento constante para siempre) el sistema
alcanza un **equilibrio estadístico** (balance disipación↔forzamiento, varianza que deja
de caer y se estabiliza), mientras que en el REAL (D→0 y forzamiento→0 al diluirse) el
campo se "congela" y su energía de gradiente decae por puro estiramiento geométrico. Esta
es la predicción física pre-registrada — ver §6.

---

## 4. Observables (dos, independientes entre sí — T2)

Para cada `(modo, semilla, σ_ruido, a)` del barrido, sobre la malla completa (sin banda —
el campo es ruido homogéneo, no hay artefacto de borde localizado que evitar):

1. **`T_energy(a)`** — energía cuadrática de gradiente físico:
   `T_energy = mean[(∂x_comov T)² + (∂y_comov T)²] / a²`
   (diferencias centradas, laplaciano-compatible; el `/a²` es el análogo directo del
   `/a` de CF2/F3-1 pero para una cantidad cuadrática — gradiente físico al cuadrado).
2. **`T_var(a)`** — varianza espacial cruda del campo:
   `T_var = Var[T(x,y)]` (sin división por `a`: la varianza de VALORES no es una
   distancia que el estiramiento métrico afecte directamente en este marco; es la
   verificación cruzada — si el "enfriamiento" sólo aparece en `T_energy` y no en
   `T_var`, es un hallazgo honesto sobre CUÁL lectura de "temperatura" es la correcta,
   no un fallo del experimento).

Ninguno de los dos comparte variables con el criterio de PASS más allá de sí mismo
(ajuste log-log sobre la propia curva, ver §6) — evita T2.

---

## 5. Barrido pre-registrado

| Parámetro | Rango | Puntos | Espaciado |
|---|---|---|---|
| `a` (factor de expansión) | [1, 1e4] | 14 | log (`np.geomspace`) |
| `σ_ruido` (amplitud de ruido, IC + forzamiento) | [1e-4, 1e-1] | 8 | log (`np.geomspace`) |
| semillas | ver lista abajo | 16 | — |
| modo | {REAL, NULL_RHO_FIXED} | 2 | — |

Semillas (10 estándar del proyecto + 6 extra, mismas que F3-1 por consistencia entre
hermanos del mismo Enfoque, ≥12 exigidas):
`[7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 13, 271828, 161803, 31415, 90210, 20260724]`

Total de trayectorias físicas: 8 (σ_ruido) × 16 (semillas) × 2 (modo) = 256, cada una
evaluando los 14 checkpoints de `a` sobre una única integración markoviana (mismo método
de checkpointing que CF2/F3-1 — no se re-simula desde cero por punto de `a`).

---

## 6. NULL

`NULL_RHO_FIXED` — MISMA trayectoria `a(t)=exp(H_EXP·t_g)` (se usa igual para el eje x y
para la división `/a²` de `T_energy`), MISMA semilla, MISMA condición inicial de ruido;
la única diferencia física es `ρ≡ρ0` (sin dilución) → `D≡D0` constante y
`forcing_amp≡σ_ruido·DYN_FRAC` constante (sin diluirse). Es la misma convención de NULL
que CF2 y F3-1 ya usaron y que mordió — comparabilidad directa entre hermanos del mismo
Enfoque.

**Predicción física pre-registrada (falsable):** con D y forzamiento constantes para
siempre, el NULL alcanza un equilibrio estadístico disipación↔forzamiento — su
`T_energy`/`T_var` deberían dejar de caer (aplanarse) tras un transitorio inicial, en vez
de seguir una ley de potencia limpia hasta `a=1e4`. Ésta es la lectura literal de la
frase del documento madre: *"NULL: sin expansión (T no debe caer)"* — aquí "sin
expansión" se interpreta como "sin dilución de la física que apaga el ruido", igual que
`NULL_RHO_FIXED` en F3-1/CF2 (no como `a≡1` congelado, que dejaría el eje x sin sentido
físico y rompería la comparabilidad directa con F3-1). Se documenta esta interpretación
ANTES de correr. Si el NULL en cambio SÍ cae con pendiente similar al REAL, se reporta
como el NULL no mordiendo (T4), sin maquillaje.

---

## 7. Criterio de PASS (congelado, no se toca tras ver resultados)

Constantes heredadas sin retocar (T1): `MONO_TOL=1e-9`, `SLOPE_DIFF_MIN=0.05` (idénticas
a CF2/F3-1). Constante NUEVA propia de F3-2 (se requiere aquí porque es el primer
experimento de la batería que exige explícitamente verificar ley de potencia, no sólo
monotonicidad): `R2_MIN=0.70` — umbral razonado pre-registrado (no ajustado a los datos,
que aún no existen al escribir esto) para "sigue razonablemente una ley de potencia en
todo el rango barrido de `a`" (ajuste sobre TODO el barrido, no solo el tramo asintótico
— evita recortar a conveniencia, T5).

Por cada combinación `(σ_ruido, semilla)`, para CADA observable (`T_energy`, `T_var`)
por separado:

- `mono_REAL` = la curva REAL es no-creciente en `a` (tolerancia `MONO_TOL`).
- `slope_REAL`, `R2_REAL` = pendiente e intercepto de mínimos cuadrados de
  `ln(T)` vs `ln(a)`, y su R² sobre los 14 puntos.
- `mono_NULL`, `slope_NULL`, `R2_NULL` = ídem para NULL_RHO_FIXED.
- `slope_diff = |slope_NULL − slope_REAL|`.
- `punto_pass` = `mono_REAL AND (R2_REAL >= R2_MIN) AND (NOT mono_NULL OR slope_diff >= SLOPE_DIFF_MIN)`

**Curva de robustez (la pregunta central, no una tasa única):**
`P(σ_ruido) = fracción de semillas con punto_pass=True`, para cada uno de los 8 valores
de `σ_ruido`, reportada por separado para `T_energy` y `T_var`.

**PASS_F3-2 (veredicto principal, basado en `T_energy` — el observable con mecanismo
geométrico directo `/a²`):** si, promediando las 8 curvas `P(σ_ruido)`, la tasa global
`rate_T_energy >= PASS_RATE_MIN (0.55)` — Y el NULL muerde (`NOT mono_NULL OR
slope_diff>=0.05`) en al menos el 55% de las combinaciones.

Si `rate_T_energy < 0.55`, el veredicto es **FAIL** y se reporta como hallazgo (T3: no se
re-elige el umbral después de ver el número).

**Verificación cruzada obligatoria (T2, tres vías):**
(a) el NULL muerde (tasa reportada explícitamente, por `σ_ruido`);
(b) **`T_var` debe COINCIDIR cualitativamente con `T_energy`** (mismo signo de veredicto
    en la mayoría de combinaciones) — si NO coincide, se reporta la discrepancia como
    hallazgo honesto sobre qué lectura de "temperatura" es la que realmente cae con la
    expansión (no se fuerza el acuerdo, T3);
(c) auditoría en disco: código (`F3_2_T_emergente_motor.py`) + JSON crudo
    (`F3_2_T_emergente_produccion_result.json`) quedan en disco para quien NO escribió
    el código.

**Exponente reportado:** media ± desviación estándar de `slope_REAL` (T_energy) a través
de las 128 combinaciones `(σ_ruido, semilla)`, reportado TAL COMO SALGA — no se compara
contra un valor esperado para decidir PASS/FAIL (eso sería T1 disfrazado); se reporta
también, de forma puramente informativa, un ajuste solo sobre el tramo asintótico (últimos
60% de puntos de `a`) para ver si el exponente se estabiliza una vez el campo se congela,
etiquetado explícitamente como diagnóstico adicional, NO como criterio de PASS.

---

## 8. Qué NO se hace aquí

- No se toca `CF2_estiramiento_motor.py`, `F3_1_estiramiento_motor.py` ni ningún otro
  archivo existente fuera de este directorio.
- No se auto-adjudica "el enfriamiento adiabático es real" más allá de este experimento
  puntual — el veredicto lo da CS con la curva cruda.
- No se cambia este criterio después de correr el motor (T3). Si el resultado es FAIL o
  el NULL no muerde o `T_var` discrepa de `T_energy`, se reporta tal cual, sin suavizar.
- No topología, no commits (regla del director para este batch).
