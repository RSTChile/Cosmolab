# PROTOCOLO F4-3 — PRE-REGISTRO
## "Dilución y reabsorción: ¿la caída de densidad apaga la difusión?"

**Batería:** BATERÍA DE FUNDAMENTOS F1–F4 (Enfoque 4 — descenso de densidad por expansión).
**Experimento:** F4-3, ejecutado por CC en paralelo con otros 23 experimentos (prefijo propio
`F4_3_`, sin tocar código/resultados de otros experimentos).
**Fuente autoritativa:** `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección F4-3 (línea ~266).

**Este documento se escribe y congela ANTES de correr el motor de producción.** El motor
(`F4_3_dilucion_difusion_motor.py`) y sus resultados
(`Cosmogenesis-Web/results/BATERIA_FUNDAMENTOS/F4_3_dilucion_difusion/`) se generan DESPUÉS de
este archivo — verificar mtime.

---

## 1. Pregunta

"Menos densidad = menos encuentros = ¿menos re-homogeneización? ¿La difusión se frena sola al
diluirse el campo?" Enunciado exacto de la batería: **medir la difusividad EFECTIVA D como
función de la densidad ρ (alta, media, baja), sin tocar nada más del modelo — pregunta de
instrumento/mecanismo pura.**

Esto es explícitamente distinto de F4-2 (que desacopla ON/OFF expansión×dilución para juzgar
persistencia) y de F3-3 (que barre el exponente n de ρ∝a⁻ⁿ dentro del esquema de expansión de
CF2). Aquí **no hay expansión acoplada en absoluto** (`a≡1` fijo durante toda la corrida): se
parametriza ρ directamente como palanca única, aislada de cualquier otro mecanismo, y se mide si
el propio campo, al difundir, EXHIBE una difusividad efectiva que cae cuando ρ cae — no se lee el
coeficiente de la fórmula, se lo mide desde la dinámica del campo (T2: la cantidad medida ≠ la
variable que la juzga).

## 2. Sustrato heredado (NO se retoca — mismo sello que CF2_estiramiento_motor.py)

Campo continuo `T(x,y)` en grilla `L×L` (`L=64`). Perfil inicial: salto tipo tanh de ancho
comóvil `W0=1.2` + ruido gaussiano `1e-4·N(0,1)` por celda, clip a `[0,1]` (idéntico a
`initial_T()` + línea de ruido de `run_sweep()` en CF2). Difusión isótropa de 4 vecinos
(esquema de Euler explícito, `diffuse()` idéntica línea por línea a CF2):
`T ← T + (dt/n_sub)·D·lap(T)`, `n_sub=N_SUB=2` subiteraciones por paso, `dt=DT=0.25` por paso
externo. `RHO0=1.0`, `D0=0.12` — heredados sin cambio.

**Única diferencia física con CF2/F3-3:** no hay factor de expansión `a(t_g)` ni reloj genético.
La densidad `ρ` se fija como parámetro directo del barrido (no derivada de `a`), y
`D = D0 · (ρ/ρ0)` — la misma fórmula de acoplamiento densidad→difusividad que usa el brazo REAL
de CF2, pero aquí `ρ` es la variable barrida explícita, no una consecuencia de `a`.

## 3. Barrido (T7 — nunca un punto/una semilla)

- **Densidad `ρ/ρ0`:** `np.geomspace(1e-4, 2.0, 12)` — 12 puntos log, ~4.3 décadas, cubriendo
  "baja" (`1e-4`) a "alta" (`2.0`, por encima del valor de referencia). Valores exactos:
  `[1.000e-04, 2.460e-04, 6.053e-04, 1.489e-03, 3.664e-03, 9.016e-03, 2.218e-02, 5.458e-02,
  1.343e-01, 3.304e-01, 8.129e-01, 2.000e+00]`. El extremo superior (`ρ/ρ0=2.0` ⇒ `D=0.24`) se
  eligió para mantener el esquema explícito DENTRO del régimen estable de von Neumann
  (`D·dt_sub ≤ 0.25`; aquí `D·dt_sub_max = 0.24·0.125 = 0.03`, con margen ×8) — no se sube más
  porque el esquema numérico dejaría de ser fiable, no por conveniencia del resultado (T1).
- **Semillas:** ≥12 exigidas. Se usan las 10 semillas estándar del proyecto
  (`CF2_estiramiento_motor.py::SEEDS_STANDARD`) + 2 semillas de extensión (dígitos de e y φ,
  igual convención que F3-3): `[7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 271828,
  161803]`. Solo perturban la condición inicial (ruido gaussiano 1e-4), como en CF2/F3-3.
- **Duración de la corrida:** `N_STEPS=2400` pasos externos × `DT=0.25` = `T_total=600` unidades
  de tiempo. Fijo para todos los puntos de ρ (no se ajusta por punto — T1). Verificado antes de
  congelar este protocolo con la solución analítica de referencia (difusión de un escalón,
  modo espectral `k=1`, ver §5): con este `T_total`, el punto de mayor D (`D=0.24`) decae a
  `~25%` de su amplitud inicial (señal fuerte, no saturada) y el punto de menor D (`D=1.2e-5`)
  decae `~0.007%` (indistinguible de ruido — resultado ESPERADO en el extremo diluido, no un
  fallo del instrumento; ver §6 sobre rango resoluble).
- **Checkpoints:** 25 muestras del campo, equiespaciadas en pasos externos
  (`np.linspace(0, 2400, 25)` redondeado), incluyendo el estado inicial (`t=0`).

Total de corridas: 12 puntos de ρ × 12 semillas = **144 corridas de producción**, más una corrida
de control de cordura (ver §4) con `D≡0` para 1 semilla.

## 4. Control de cordura (no es NULL de persistencia — el enunciado dice NULL: "—")

El enunciado de F4-3 declara explícitamente que no aplica un NULL de persistencia (es medición
de instrumento). En su lugar, para satisfacer el punto 3 de las reglas generales de la batería
(alguna verificación tipo-(a) además del segundo método y la auditoría en disco), se corre un
**control de cordura de arnés**: 1 semilla, `D≡0` exacto (`ρ/ρ0=0`). Con `D=0`, `diffuse()` hace
`return T` sin modificar nada (línea 95-96 de CF2, heredada intacta) — el campo debe quedar
BIT-A-BIT idéntico en todos los checkpoints. Si no lo está, el arnés de medición (no el
mecanismo físico) tiene un error y se PARA y reporta a CS, no se seguirá con producción.

## 5. Observables — DOS métodos independientes, ninguno lee el coeficiente de entrada (T2)

Para cada `(ρ, semilla)`, en cada checkpoint `t`:

```
perfil(x, t) = mean_y T(x, y, t)      (promedio sobre filas; ver nota de exactitud abajo)
```

**Nota de exactitud:** para el operador discreto de CF2, el término de Laplaciano vertical
(`roll(·,±1,axis=0)`) se anula EXACTAMENTE al promediar sobre todas las filas de una columna
periódica (suma telescópica), de modo que `perfil(x,t)` evoluciona bajo la ecuación de difusión
1D pura en `x`, con la MISMA `D` que el campo 2D completo — no es una aproximación.

- **Método A (primario) — decaimiento espectral del modo fundamental:**
  `A1(t) = |FFT(perfil(x,t) − mean(perfil))[k=1]|`. Para difusión pura, `A1(t) = A1(0)·exp(−λ·t)`
  con `λ = D · eig(k=1)`, `eig(k=1) = 2 − 2·cos(2π/L) = 0.00963055` (autovalor EXACTO del operador
  discreto periódico usado por `diffuse()`, no la aproximación continua `k²`). Se ajusta
  `ln(A1(t))` vs `t` por mínimos cuadrados sobre los checkpoints con `A1(t) > 1e-10` (piso de
  punto flotante, pre-registrado) y con al menos 3 puntos; `D_eff_spectral = −pendiente / eig(k=1)`.
- **Método B (secundario, cross-check independiente) — ensanchamiento del frente en espacio real:**
  `peak_grad(t) = max` de `|∂perfil/∂x|` restringido a la banda central `[L/8, 7L/8]` (misma banda
  que `grad_metrics()` de CF2, evita wrap-around). `w(t) = 0.5 / peak_grad(t)` (ancho tipo-tanh:
  para `T=0.5(1−tanh(x/w))`, el gradiente pico es `0.5/w`). Se ajusta `w(t)²` vs `t` por mínimos
  cuadrados (difusión de un frente: `w(t)² ≈ w0² + 2·D·t`); `D_eff_front = pendiente / 2`.

Ninguno de los dos métodos usa `ρ` ni `D0` como entrada del ajuste — ambos derivan `D_eff`
exclusivamente de cómo evolucionó el campo. Son dos observables ortogonales (espectral vs
espacio real), igual que el par autocorrelación/información-mutua de F1-1/F1-2.

**Calidad de ajuste:** se reporta `R²` de cada regresión lineal (espectral: `ln(A1)` vs `t`;
frente: `w²` vs `t`). `R²_MIN = 0.8` (pre-registrado) — un punto `(ρ, semilla, método)` se marca
**resoluble** solo si `R² ≥ R²_MIN` Y hubo ≥3 checkpoints utilizables. Esto reemplaza a un NULL
externo: es el criterio honesto de "¿hay señal medible aquí?", separado de "¿la señal cae con
ρ?" — evita forzar un ajuste sobre ruido en el extremo diluido (T5: el criterio puede fallar).

## 6. Criterio de PASS (congelado, T3 — no se toca si falla)

Constantes pre-registradas:
- `R2_MIN = 0.8` (§5).
- `SLOPE_MIN = 0.5` (pendiente mínima de `ln(D_eff)` vs `ln(ρ/ρ0)` sobre los puntos resolubles,
  para leerse como "D cae claramente con ρ". La expectativa teórica del acoplamiento lineal
  `D=D0·ρ/ρ0` sería pendiente≈1; se usa un umbral MÁS LAXO (0.5) para no exigir el valor exacto
  de la fórmula de entrada — sería T2 si el juez fuera "¿la pendiente da exactamente 1?").
- `MONO_TOL = 1e-6` (tolerancia de monotonicidad no-decreciente de `D_eff_spectral(ρ)` sobre los
  puntos resolubles, ordenados por `ρ` creciente).
- `CORR_MIN = 0.8` (correlación de Pearson entre `ln(D_eff_spectral)` y `ln(D_eff_front)` sobre
  los puntos resolubles por AMBOS métodos — verificación cruzada de método, T2).
- `MIN_RESOLVABLE_PTS = 4` (mínimo de puntos de ρ resolubles para poder ajustar una pendiente
  log-log con sentido).
- `PASS_RATE_MIN = 0.55` (umbral estándar del proyecto, igual que CF2/F3-3).

Por semilla `s`:

1. Se calculan `D_eff_spectral(ρ,s)` y `D_eff_front(ρ,s)` para los 12 puntos de ρ.
2. `resoluble(ρ,s,método)` = `R²(ρ,s,método) ≥ R2_MIN` con ≥3 checkpoints usados.
3. Si el número de ρ resolubles por AMBOS métodos simultáneamente es `< MIN_RESOLVABLE_PTS`:
   `seed_pass(s) = None` (semilla **"sin rango resoluble"** — se excluye del denominador de
   `rate`, se reporta aparte, no se cuenta como FAIL disfrazado).
4. Si no, sobre esos puntos resolubles-por-ambos:
   - `cond1(s)`: pendiente `ln(D_eff_spectral)` vs `ln(ρ)` ≥ `SLOPE_MIN` **Y** pendiente
     `ln(D_eff_front)` vs `ln(ρ)` ≥ `SLOPE_MIN`.
   - `cond2(s)`: `D_eff_spectral` no-decreciente en ρ dentro de `MONO_TOL` sobre esos puntos.
   - `cond3(s)`: correlación de Pearson `ln(D_eff_spectral)` vs `ln(D_eff_front)` ≥ `CORR_MIN`.
   - `seed_pass(s) = cond1 AND cond2 AND cond3`.

**Verdict del experimento:**
- `n_validas` = semillas con `seed_pass ∈ {True, False}` (excluye "sin rango resoluble").
- `rate = (#seed_pass=True) / n_validas` si `n_validas > 0`.
- Si `n_validas = 0` (ninguna semilla tiene rango resoluble): veredicto explícito
  `"F4_3_SIN_SEÑAL_MEDIBLE"` — resultado honesto, no se fuerza un PASS/FAIL.
- Si `n_validas > 0`: `"F4_3_D_CAE_CON_RHO"` si `rate ≥ PASS_RATE_MIN`, si no
  `"F4_3_D_INDEPENDIENTE_O_NO_CONCLUYENTE"`.

**Diagnóstico adicional (no forma parte del PASS, se reporta siempre):** razón
`D_eff_spectral(ρ,s) / (D0·ρ/ρ0)` en los puntos resolubles — mide si el ajuste recupera el
coeficiente exactamente introducido en el código (calibración del instrumento en sí, distinto de
la pregunta física "¿cae con ρ?").

## 7. Qué NO es este experimento

- No decide si la dilución real del universo (acoplada a la expansión) apaga la difusión — eso
  es competencia de F4-2/F4-4/F4-6, que sí acoplan `a`. Aquí `ρ` es una palanca aislada, directa,
  sin expansión, precisamente para medir el INSTRUMENTO (la relación D↔ρ que el código ya
  codifica) de forma independiente de cómo se llega a esa ρ.
- No toca `CF2_estiramiento_motor.py`, `TEST_RHO_DISPERSION.py`, ni ningún archivo de otro
  experimento de la batería (F1, F2, F3, ni los otros F4).
- No se auto-adjudica el veredicto físico de la batería — el motor entrega números crudos; la
  lectura final es de CS (Alexis).

## 8. Ruta de archivos

- Protocolo (este archivo):
  `Cosmogenesis-Web/codigo/BATERIA_FUNDAMENTOS/F4_3_dilucion_difusion/PROTOCOLO_F4-3_PREREGISTRO.md`
- Motor:
  `Cosmogenesis-Web/codigo/BATERIA_FUNDAMENTOS/F4_3_dilucion_difusion/F4_3_dilucion_difusion_motor.py`
- Resultados:
  `Cosmogenesis-Web/results/BATERIA_FUNDAMENTOS/F4_3_dilucion_difusion/F4_3_dilucion_difusion_<modo>_result.json`

---

**Fecha/hora de este pre-registro:** 2026-07-24 (ver mtime del archivo — se congela antes de
generar el motor y cualquier resultado).
