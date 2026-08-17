# PROTOCOLO F3-3 — PRE-REGISTRO
## "Tasa de dilución: ¿es a⁻³ especial o solo monótona?"

**Batería:** BATERÍA DE FUNDAMENTOS F1–F4 (Enfoque 3 — enfriamiento adiabático).
**Experimento:** F3-3, ejecutado por CC en paralelo con otros 23 experimentos (prefijo propio
`F3_3_`, sin tocar código/resultados de otros experimentos).
**Fuente autoritativa:** `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección F3-3 (línea ~202).

**Este documento se escribe y congela ANTES de correr el motor de producción.** El motor
(`F3_3_exponente_dilucion_motor.py`) y sus resultados (`results/F3_3_exponente_dilucion/`) se
generan DESPUÉS de este archivo — verificar mtime (`ls -la` / `git log`).

---

## 1. Pregunta

La densidad diluye como ρ ∝ a⁻³ en el motor CF-2 heredado (3D real). Aquí se pregunta si ese
exponente concreto **n=3** tiene algo especial, o si **cualquier** tasa de dilución monótona
(n=1,2,3,4,5) produce el mismo tipo de destino cualitativo para el gradiente térmico —
simplemente con distinta velocidad. Explícitamente prohibido (T1): fijar n=3 a mano y no barrer.

## 2. Sustrato heredado (NO se retoca — mismo sello que CF2_estiramiento_motor.py)

Campo continuo T(x,y) en grilla L×L (L=64). Perfil inicial: salto tipo tanh de ancho comóvil
W0=1.2. Difusión isótropa de 4 vecinos, coeficiente D. Reloj de expansión a(t_g)=exp(H_EXP·t_g),
H_EXP=6.0. D0=0.12, DT=0.25, N_SUB=2, ORIGINAL_STEPS_PER_TG=399 — heredados de
`CF2_estiramiento_motor.py` (Cosmogenesis-Web/codigo/CF2_estiramiento/), que a su vez los hereda
de `TEST_RHO_DISPERSION.py`. Ningún valor del sello se cambia para favorecer el resultado (T1).

**Generalización propia de F3-3 (la única diferencia física con CF2):**

```
ρ(a) = ρ0 · a^(−n)          n ∈ {0, 1, 2, 3, 4, 5}
D(a) = D0 · (ρ(a)/ρ0) = D0 · a^(−n)
```

CF2 fijaba n=3 (D0/a³) como único brazo "REAL" contra un NULL de densidad fija. Aquí n=3 es
**un punto más del barrido**, sin privilegio alguno. `n=0` reproduce exactamente el
`NULL_RHO_FIXED` de CF2 (ρ y D constantes, sin dilución) y se usa como NULL de este experimento.

## 3. Barrido (T7 — nunca un punto/una semilla)

- **Exponente n:** `{0, 1, 2, 3, 4, 5}` — 5 valores "REAL" (n=1..5) + n=0 como NULL. Los 5
  valores físicos vienen dados por el enunciado del experimento (línea 205 del documento
  autoritativo); no se agregan ni se quitan para que el resultado salga de una forma u otra.
- **Factor de expansión `a`:** `np.geomspace(1.0, 1000.0, 7)` — **la misma grilla exacta** que
  `CF2_estiramiento_motor.py` (`A_GRID`), reutilizada por dos razones: (i) permite comparación
  directa punto a punto con el núcleo ya congelado de CF2, y (ii) evita introducir una grilla
  nueva ad-hoc que pudiera elegirse (consciente o inconscientemente) para favorecer la curva de
  n=3 (T1).
- **Semillas:** ≥12 exigidas por el documento autoritativo. Se usan las 10 semillas estándar del
  proyecto (`CF2_estiramiento_motor.py::SEEDS_STANDARD`) **más 2 semillas de extensión** basadas
  en dígitos de constantes matemáticas (no ajustadas a mano para dar un resultado):
  `271828` (dígitos de e) y `161803` (dígitos de φ). Total: 12 semillas —
  `[7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 271828, 161803]`.
  Cada semilla solo perturba la condición inicial (ruido gaussiano de amplitud 1e-4), igual que
  en CF2 — no es la fuente dominante de variación, es el control de robustez frente a ruido T7.

Total de corridas: 6 modos (n=0..5) × 12 semillas × 7 puntos de `a` = 504 evaluaciones del
observable, cada una tras integrar la PDE de difusión hasta el `t_g` correspondiente.

## 4. Brazos

- **n=0 → NULL_DENSIDAD_FIJA:** ρ≡ρ0, D≡D0 (idéntico a `NULL_RHO_FIXED` de CF2). Debe seguir
  erosionando el gradiente comóvil por difusión sostenida sin que la dilución lo frene.
- **n=1..5 → REAL(n):** ρ=ρ0·a⁻ⁿ, D=D0·a⁻ⁿ. A mayor n, la difusión se apaga más rápido al
  expandirse ⇒ predicción física cualitativa (a verificar, no asumir): el gradiente físico se
  acerca más al estiramiento geométrico puro (pendiente log-log → −1) cuanto mayor es n, porque
  la dilución apaga el mecanismo erosivo (difusión) antes.

## 5. Observables (T2 — no comparten variable con el juez; DOS métodos independientes)

Para cada (n, semilla, a):

```
∇_comov(a) = |∂T/∂x|   (banda central x ∈ [L/8, 7L/8], evita wrap-around periódico)
```

- **Observable primario — A_phys_max(a):** `max(∇_comov(a)) / a` (idéntico al observable de
  CF2 `A_phys`, máximo de la banda central).
- **Observable secundario — A_phys_rms(a):** `RMS(∇_comov(a)) / a` (raíz cuadrática media de la
  misma banda, en vez del máximo). Es un estadístico distinto del campo (sensible a la forma
  completa del perfil, no solo al pico) — sirve de verificación cruzada independiente: si el
  veredicto de "monótono en n" y "n=3 no singular" solo aparece en el máximo y no en el RMS, se
  reporta como discrepancia de método, no se oculta.

Ninguno de los dos observables usa variables de linaje/juez de otros experimentos de la batería.

## 6. Criterio de PASS (congelado, T3 — no se toca si falla; SLOPE_DIFF_MIN heredado de CF2)

Constantes pre-registradas:
- `MONO_TOL = 1e-9` (tolerancia de monotonicidad en `a`, igual que CF2).
- `SLOPE_DIFF_MIN = 0.05` (heredado literal de CF2, separación mínima REAL vs NULL).
- `MONO_N_TOL = 1e-6` (tolerancia de monotonicidad en `n`, deliberadamente más laxa: la variable
  barrida son 6 puntos discretos de n, no 7 de a en log; una tolerancia demasiado fina
  produciría falsos NO-monótonos por ruido numérico de punto flotante).
- `SINGULARITY_FACTOR = 3.0`, `SINGULARITY_FLOOR = 1e-3` (umbral generoso: solo se marca
  singularidad si la curvatura en n=3 es al menos 3× la curvatura de sus vecinos n=2 y n=4, con
  un piso absoluto para no disparar falsos positivos cuando la curva es casi perfectamente lineal
  y toda curvatura es ruido de orden 1e-3 o menor).

Por semilla `s`, sobre el observable primario `A_phys_max`:

1. **`mono(n,s)`**: `True` si `A_phys_max(a_{i+1}) ≤ A_phys_max(a_i)·(1+MONO_TOL)` para todos los
   pares consecutivos del barrido de `a`, para ese `n`.
2. **`slope(n,s)`**: pendiente OLS de `ln(A_phys_max)` vs `ln(a)` sobre los 7 puntos.
3. **`null_bites(s)`** (generaliza el T4 de CF2): `(not mono(0,s))` **o**
   `max_{n=1..5} |slope(n,s) − slope(0,s)| ≥ SLOPE_DIFF_MIN`. Si esto no muerde, el instrumento
   no distingue dilución de no-dilución para NINGÚN n — se reporta tal cual (T4).
4. **`mono_in_n(s)`** (la verificación central pedida por el documento — "el efecto debe ser
   monótono en n"): `True` si la secuencia `slope(1,s), slope(2,s), …, slope(5,s)` es monótona
   no-decreciente dentro de `MONO_N_TOL` (predicción física: a mayor n, pendiente menos negativa,
   más cerca de −1). Si NO es monótona, se reporta el punto exacto donde se rompe — no se
   suaviza ni se descarta.
5. **`n3_not_singular(s)`**: con `curv(n,s) = slope(n−1,s) − 2·slope(n,s) + slope(n+1,s)` para
   n=2,3,4 (curvatura discreta de la curva slope(n)), se marca **singular** si
   `|curv(3,s)| > SINGULARITY_FACTOR · max(|curv(2,s)|, |curv(4,s)|, SINGULARITY_FLOOR)`.
   `n3_not_singular(s) = not singular`.

**`seed_pass(s) = null_bites(s) AND mono_in_n(s) AND n3_not_singular(s)`**

**Verdict del experimento:** `rate = (#semillas con seed_pass) / 12`, `PASS_RATE_MIN = 0.55`
(idéntico umbral que CF2, pre-registrado antes de correr, no ajustado después de ver datos).

El observable secundario (`A_phys_rms`) se evalúa con el mismo procedimiento (puntos 1-5) de
forma independiente y se reporta su propia tasa `rate_rms`, como verificación cruzada de método.
Si `rate` (máximo) y `rate_rms` (RMS) difieren sustancialmente, se reporta la discrepancia sin
elegir cuál "vale más".

**Importante — qué NO significa "PASS" aquí:** a diferencia de CF2 (donde PASS = "el dato
distingue REAL de NULL"), en F3-3 `seed_pass` es un chequeo compuesto de tres afirmaciones
verificables (NULL muerde + monotonicidad en n + no-singularidad de n=3). El HALLAZGO real de
este experimento es la familia completa de curvas `slope(n)` y `A_phys(a; n)` — se reporta cruda
independientemente de si `seed_pass` da True o False. Ninguna lectura ("monótono en n" / "n=3 es
singular" / "el NULL no muerde para algún n") se privilegia sobre otra; las tres son hallazgo
válido (igual que F1-3, sección PASS de tres lecturas).

## 7. Qué NO es este experimento

- No decide si ρ∝a⁻³ es "la física correcta" del universo — eso no está en discusión aquí; se
  pregunta solo si, DENTRO del motor de campo-difusión de esta batería, n=3 se comporta distinto
  en tipo (no solo en velocidad) de n=1,2,4,5.
- No toca `CF2_estiramiento_motor.py`, `TEST_RHO_DISPERSION.py`, ni ningún resultado de otro
  experimento de la batería (F1, F2, F4, ni los otros F3).
- No se auto-adjudica el veredicto cosmológico — el motor entrega números crudos; la lectura la
  hace CS (Alexis) después.

## 8. Ruta de archivos

- Protocolo (este archivo):
  `Cosmogenesis-Web/codigo/BATERIA_FUNDAMENTOS/F3_3_exponente_dilucion/PROTOCOLO_F3-3_PREREGISTRO.md`
- Motor: `Cosmogenesis-Web/codigo/BATERIA_FUNDAMENTOS/F3_3_exponente_dilucion/F3_3_exponente_dilucion_motor.py`
- Resultados: `Cosmogenesis-Web/results/F3_3_exponente_dilucion/F3_3_exponente_dilucion_<modo>_result.json`

---

**Fecha/hora de este pre-registro:** 2026-07-24 (ver mtime del archivo — se congela antes de
generar el motor y cualquier resultado).
