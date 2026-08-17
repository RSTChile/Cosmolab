# PRE-REGISTRO — Etapa 7 / Track D: masa dinámica por recuperación cinemática tipo Kepler

**Escrito:** 2026-07-23 05:02 (hora local, antes de correr producción; smoke posterior a este archivo también cuenta como "antes de producción").
**Autor:** agente delegado, sesión Cosmogénesis (dirección: Alexis López Tapia).
**Motor:** copia nueva `etapa7_trackD_kepler_v1.py`, prefijo `etapa7_trackD_kepler_`. No se edita v1–v6 ni motor_1a7.

---

## 0. Objetivo

Canal de masa **independiente del linaje**: en vez de co-membresía/fusión (`co_member_score`,
`n_long_co_pairs`, `fusion_events` — **PROHIBIDOS como ingrediente**), recuperar una magnitud de
"masa" pura **cinemática** (tipo virial/Kepler: M ~ v²·r/G) a partir de separación y velocidad
relativa de pares de átomos en FORCE_CUTOFF, sin mirar nunca cuántos pasos llevan juntos ni con
quién comparten grupo.

`co_member_score`, `n_long_co_pairs`, `fusion_events` **no se calculan en este script** (no solo
no se usan: no existen en el código), para eliminar la posibilidad de fuga accidental. La única
variable "identidad" usada es el **id de track** (para saber que el par (i,j) del paso t es el
mismo par físico del paso t-1) — eso no es co-membresía ni linaje, es continuidad de objeto.

---

## 1. Observable primario: M_dyn (cinemático puro)

Para cada paso E4 (`step >= grav_start` y `frozen`), y cada par de átomos (i,j) cuya separación
toroidal esté dentro de `FORCE_CUTOFF=8.0`:

```
dy_t = toroidal_delta(pos_j_y(t), pos_i_y(t), L)
dx_t = toroidal_delta(pos_j_x(t), pos_i_x(t), L)
r_t  = hypot(dy_t, dx_t)
```

Si el mismo par (i,j) también estaba en FORCE_CUTOFF en el paso t-1 (serie consecutiva, sin huecos):

```
Δdy = toroidal_delta(dy_{t-1}, dy_t, L)
Δdx = toroidal_delta(dx_{t-1}, dx_t, L)
v_t   = hypot(Δdy, Δdx) / DT_NB          (DT_NB=0.35, igual al motor v6)
r_mid = 0.5 * (r_t + r_{t-1})
M_dyn_t = v_t² * r_mid / G_GRAV
```

`G_GRAV` es una constante de la simulación (parámetro conocido, no derivado del dato) — dividir
por ella no es un ingrediente prohibido, es solo el factor de escala orbital estándar
(M = v²r/G para órbita ligada tipo Kepler/virial).

Un par **califica** si acumula ≥ `PAIR_MIN_STEPS = 5` muestras `M_dyn_t` consecutivas dentro de
una misma racha en FORCE_CUTOFF (umbral de calidad de dato, no de linaje — es análogo en espíritu
a `MUTUAL_MIN_STEPS` de v4-v6 pero **calculado de cero aquí**, sin tocar esa variable).

Por par calificado:
- `mean_M`, `std_M` sobre su serie de `M_dyn_t`.
- `CV_pair = std_M / mean_M` (solo si `mean_M > 1e-9`).
- `mass_proxy_pair = mean(mass_proxy_i) + mean(mass_proxy_j)` — mass_proxy es la cantidad
  estructural por átomo ya existente en el motor (`sum_phi * (1+f_core)` del campo φ), **no**
  derivada de linaje/co-membresía. Se usa solo como variable de comparación (paso 3b de la
  misión), nunca dentro de la fórmula de M_dyn ni del criterio de pass.

Por semilla × modo (`real`, `shuffle`, `off`, `invert`):
- `CV_med` = mediana de `CV_pair` sobre todos los pares calificados (requiere ≥3 pares calificados
  con `CV_pair` finito; si no, `CV_med = None`).
- `rho_mass` = correlación de Spearman entre `mean_M` y `mass_proxy_pair` sobre pares calificados
  (requiere ≥5 pares; si no, `rho_mass = None`). Implementación: rango (promedio en empates) +
  Pearson sobre rangos (sin scipy).

---

## 2. Sub-tests por semilla (solo REAL vs SHUFFLE)

**T1 — estabilidad:** `CV_med_shuffle / CV_med_real >= CV_RATIO_MIN (1.15)`
(requiere ambos `CV_med` definidos). Lee: el par real es ≥15% más *coherente/estable* en el
tiempo — signo de órbita con masa fija — que el mismo cálculo aplicado a SHUFFLE (donde la fuente
de fuerza está desacoplada de la posición real).

**T2 — correlación con mass_proxy:** `rho_mass_real >= RHO_REAL_MIN (0.25)` **y**
`(rho_mass_real - rho_mass_shuffle) >= RHO_GAP_MIN (0.15)` (requiere ambos `rho_mass` definidos;
si `rho_mass_shuffle` es `None`, se trata como 0 para el gap).

`PASS_D(semilla) = T1 OR T2` (ambos calculados sin mirar el resultado agregado antes de fijar el
umbral; no se baja el umbral tras correr producción, ni se recalcula solo en semillas que fallan).

---

## 3. Controles nulos (reportados, no gatean el PASS salvo colapso)

- **OFF** (`G_GRAV=0` o modo off): `nbody_step` no actualiza posiciones ni genera `force_pairs`
  → **0 pares calificados por construcción**. Se reporta `frac_off_zero_pairs` sobre las 10
  semillas; si `< OFF_NULL_CLEAN_FRAC (0.90)` el motor tiene una fuga y el veredicto es INCONCLUSO
  independientemente de la tasa T1/T2.
- **INVERT** (signo de fuerza invertido, misma masa real): se espera *peor* T1/T2 que REAL
  (órbita se abre en vez de cerrarse) — reportado, no gatea.

---

## 4. Veredicto agregado (10 semillas estándar, G_GRAV=0.20 canónico)

```
rate_D = fracción de semillas con PASS_D=True (de las 10 semillas estándar)
```

- **PASS** si `rate_D >= SEED_RATE_PASS (0.55)` **y** `frac_off_zero_pairs >= 0.90`.
- **PARTIAL** si `0.30 <= rate_D < 0.55` (nulls limpios).
- **FAIL** si `rate_D < 0.30` (nulls limpios).
- **INCONCLUSO** si los nulls OFF no están limpios, o si <5 semillas tienen algún par calificado
  en REAL (dato insuficiente para evaluar T1/T2 en absoluto).

Umbrales (`CV_RATIO_MIN`, `RHO_REAL_MIN`, `RHO_GAP_MIN`, `PAIR_MIN_STEPS`, `SEED_RATE_PASS`,
`OFF_NULL_CLEAN_FRAC`) se fijan **en este archivo, antes de producción**, y no se tocan después de
ver el JSON de producción. Si el smoke revela que hay que ajustar algo estructural (p.ej. el
motor no genera ningún par calificado en ningún modo), el ajuste se hace **antes** de congelar
este documento — ver §6 para el registro de iteración smoke→pre-registro final.

---

## 5. Semillas y barrido

- **Semillas estándar (control principal, G=0.20):** 7, 42, 99, 777, 2025, 3141, 8191, 99991,
  12345, 54321 — las mismas 10 de v1-v6, por comparabilidad de régimen físico (no por necesidad
  circular: no se usa ningún output de linaje de esas corridas).
- **Barrido G_GRAV:** `{0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}` × subconjunto de 4
  semillas (`2025, 42, 777, 3141`, igual patrón que v6) — diagnóstico de si T1/T2 dependen de la
  intensidad de acople gravitatorio.
- **Smoke previo:** 4 semillas (`7, 42, 99, 777`), `pasos=200` (motor completo, no v1-v6), antes
  de la corrida de producción de 10 semillas × pasos=400 (default P, igual a v6).

---

## 6. Registro de iteración (llenar tras smoke, antes de producción)

- Hora de escritura de este documento: ver encabezado.
- Hora de la corrida de producción: **a llenar en el resumen final** (`RESUMEN_TRACKD_KEPLER.md`).
- Si el smoke obliga a cambiar algo de este documento, se anota aquí explícitamente qué cambió y
  por qué, ANTES de correr producción con los nuevos parámetros.
