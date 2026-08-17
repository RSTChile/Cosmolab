# PROTOCOLO F2-5 — Congelamiento bajo expansión no uniforme (historia temporal variable)

**Fecha de pre-registro:** 2026-07-24, antes de escribir/correr el motor.
**Autor:** CC (F2-5), sobre la base física de `cs074_rcruz.py` (CS074-rcruz, no editado).
**Pregunta del enfoque:** la expansión real no es a tasa constante — ¿el congelamiento
de la diferencia (persistencia) aguanta si la tasa H cambia en el tiempo, manteniendo
la misma expansión total integrada? ¿El veredicto depende del PERFIL H(t) o solo del
r efectivo integrado?

Este documento fija el observable, el NULL, el criterio de PASS y los rangos del
barrido ANTES de correr el motor. Si algo falla, se reporta el FAIL — este archivo
no se edita después de ver resultados (T3).

---

## 1. Física reusada de CS074-rcruz (sin tocar el archivo original)

- Campo continuo `phi` en anillo de N sitios; condición inicial = fondo=1 + mancha
  ε (5 modos de Fourier con fases aleatorias, normalizada a std=1).
- Difusión: `paso_difusion` — SOLO por aristas vivas (idéntica a CS074-rcruz,
  reimportada del módulo original, no reescrita).
- Expansión: `paso_expansion(activo, H, rng)` — cada arista viva se corta con
  probabilidad H en ese paso (Bernoulli), igual que CS074-rcruz. Aquí H puede
  variar paso a paso: H = H(t).
- D (difusividad) se MIDE igual que CS074-rcruz: fracción de contraste borrada en
  UN paso de difusión pura (H=0), promediada sobre semillas. No se impone.
- `pasos` (duración de la corrida) se CALIBRA igual que CS074-rcruz:
  `medir_pasos_lavado` — tiempo medido (mediana sobre semillas) para que a H=0 la
  persistencia caiga bajo P_LAVADO=0.05, con margen ×1.15. No se elige a mano.

## 2. Los perfiles H(t) — definición EXACTA (5 perfiles, ≥4 pedidos)

Cada perfil se define como una FORMA adimensional w(t), con media temporal discreta
`mean_t w(t) = 1` EXACTA por construcción (verificada en código, no solo en teoría).
El H(t) real usado es:

```
H(t) = clip( H_bar * w(t), 0, 1 )      donde H_bar = r_medio * D
```

`r_medio` es el eje barrido (igual rol que "r" en CS074-rcruz, pero aquí es el
promedio temporal nominal, no un valor constante impuesto en cada paso).

| Perfil | w(t), t=0..pasos-1 | Intuición |
|---|---|---|
| `constante` | w(t) = 1 | referencia (igual a CS074-rcruz original) |
| `acelerando` | w(t) = 2·(t+0.5)/pasos | empieza casi sin expandir, termina expandiendo fuerte (rampa lineal 0→2, media exacta 1) |
| `desacelerando` | w(t) = 2·(pasos−t−0.5)/pasos | espejo: empieza fuerte, termina casi sin expandir |
| `rafaga_lento` | primeros f·pasos pasos: w=5.0; resto: w=(1−f·5)/(1−f) | ráfaga inicial intensa (10% del tiempo, w=5) luego expansión lenta sostenida (90%, w≈0.556); f=0.1 |
| `lento_rafaga` | primeros (1−f)·pasos pasos: w≈0.556; últimos f·pasos: w=5.0 | espejo temporal de `rafaga_lento` — mismo total, ráfaga al final |

Verificación de construcción (T1): para cada perfil se registra en el JSON crudo
`w_mean_exacto` (media discreta real de w antes del clip) y `H_bar_objetivo` vs
`H_bar_realizado` (media real de H(t) tras el clip) — si el clip por saturación
(H→1) distorsiona la media nominal en más de 1%, se reporta explícitamente (no se
esconde) y esos puntos se marcan `clip_afectado=true`.

## 3. Barrido pre-registrado

- N = 200 (mismo tamaño que el modo `produccion` de CS074-rcruz).
- eps ∈ {0.0, 1e-3} — 0.0 es el control estricto (P debe ser 0 a todo r y todo
  perfil); 1e-3 es la amplitud de señal usada en la calibración de referencia de
  CS074-rcruz (`produccion`, `cal_ref`).
- r_medio ∈ {0.1, 0.3, 1.0, 3.0, 10.0, 30.0} — cruza el r*≈1 conocido de F2-1/rcruz,
  con puntos por debajo, en la transición y por encima.
- perfiles: los 5 de la tabla anterior.
- semillas: 16 por combinación (≥12 pedido), semillas 2000..2015 (offset propio,
  para no pisar el espacio de semillas de otros experimentos F2-x en paralelo).
- Total combinaciones REAL+NULL: 2 eps × 6 r_medio × 5 perfiles × 16 semillas × 2
  (real/null) = 1920 corridas del motor.
- D y `pasos` se miden una vez para (N=200, eps=1e-3) igual que CS074-rcruz
  `produccion`, y se reusan para todas las combinaciones (mismo criterio que el
  script base: pasos_fijo).

### 3b. Perturbación dinámica adicional (mini-estudio, T7)

El eje "perfil H(t)" YA ES una perturbación dinámica de la expansión (no es solo
cambiar semilla). Como refuerzo, se añade un jitter multiplicativo estocástico
sobre H(t): `H(t) → H(t) · ξ_t`, con `ξ_t ~ U(1−δ, 1+δ)` i.i.d. por paso,
resampleado con la misma rng de la corrida (no una rng aparte, para que sea
reproducible por semilla).

- δ ∈ {0.0, 0.3}.
- r_medio = 1.0 (la zona de transición, la más sensible).
- perfiles: `constante` y `rafaga_lento` (el más extremo de la tabla).
- eps = 1e-3.
- semillas: 16 (mismas 2000..2015).
- Total: 2 δ × 2 perfiles × 16 semillas × 2 (real/null) = 128 corridas extra.

## 4. Observables (dos métodos independientes, T2/verificación cruzada)

1. **P (persistencia, primario)** — idéntico a CS074-rcruz:
   `P = corr(phi, roll(phi,1))_{≥0} × (var(phi)/contraste0²)`. Forma × magnitud.
2. **std_ratio (secundario, independiente)** — `phi.std() / contraste0`, retención
   de amplitud del contraste SIN pasar por autocorrelación espacial. Ya estaba
   calculado en CS074-rcruz (`corrida()`); aquí se usa como segunda vía. Si P dice
   "persiste" y std_ratio dice "se borró", el veredicto de ese punto se marca en
   conflicto (no se promedia a favor).
3. **Auditoría en disco**: JSON crudo con TODAS las filas (una por combinación:
   perfil × eps × r_medio × [semilla implícita, agregado en media/std/z]) más los
   arrays de `frac_exp` realizado por semilla, para que quien audite pueda
   recalcular r_efectivo sin volver a correr el motor.

## 5. r efectivo integrado (la verificación cruzada central de F2-5)

Tras cada corrida se mide `frac_exp` = fracción de aristas cortadas al final. Se
define el r efectivo REALIZADO (equivalente-constante que habría dado la misma
supervivencia de aristas en `pasos` pasos):

```
r_efectivo = -ln(max(1 - frac_exp, 1e-12)) / pasos / D
```

Esto es lo que compara perfiles distintos en la MISMA vara: si dos perfiles con el
mismo r_medio nominal terminan con r_efectivo distinto (por el clip o por la
no-linealidad de "aristas ya cortadas no se pueden re-cortar"), el eje correcto
para comparar es r_efectivo, no r_medio nominal.

## 6. NULL

Barajado del campo (`rng.permutation(phi)`) DESPUÉS de toda la evolución —
idéntico al NULL de CS074-rcruz. Se corre por cada semilla, mismo camino de
expansión (misma secuencia de cortes) que su pareja REAL, solo se destruye el
orden espacial de phi al final. Esto es "NULL: barajado" como pide la ficha F2-5.

## 7. Criterio de PASS (fijado ahora, no se toca después)

**PASS pre-registrado:** persistencia robusta al perfil a r_efectivo fijo.
Operacionalización:

- (a) El NULL debe cumplir P_null ≈ 0 en todo el barrido (si no, T4: el NULL no
  muerde y el resto no se interpreta como señal).
- (b) Para cada bin de r_efectivo (agrupando corridas de perfiles distintos que
  caigan en el mismo bin log-ancho), la dispersión de P_real ENTRE PERFILES debe
  ser comparable a la dispersión ENTRE SEMILLAS dentro de un mismo perfil (razón
  ≤ 2×). Si la dispersión entre perfiles excede 2× la dispersión entre semillas
  en algún bin, ESE perfil/bin se reporta como ruptura de la robustez (T1: no se
  esconde bajo un promedio).
- (c) El observable secundario (std_ratio) debe dar el mismo veredicto cualitativo
  (mismo signo de tendencia con r_efectivo) que P en cada perfil.
- (d) eps=0 debe dar P≈0 (y P_null≈0) en TODOS los perfiles y r_medio — control
  estricto T1.
- (e) El mini-estudio de jitter (§3b) es informativo, no cambia el criterio (a)-(d);
  se reporta si el jitter cambia el orden relativo constante vs rafaga_lento.

**Lecturas posibles (las tres son hallazgo, ninguna se privilegia a priori):**
- Robusto al perfil → confirma que r efectivo integrado es la variable física real,
  el perfil es irrelevante (resultado esperado por la física de "solo importa
  cuánto se cortó en total").
- No robusto (algún perfil rompe el colapso) → se reporta CUÁL perfil, en qué
  rango de r_efectivo, y con qué magnitud — dato en contra de la hipótesis de
  "solo importa el total", posible mecanismo de "cuándo" además de "cuánto".
- Mixto (robusto en unos rangos de r_efectivo, no en otros) → se reporta el mapa
  completo, sin promediar la ruptura local.

## 8. Qué NO se hace aquí (fuera de alcance de F2-5)

- No se barre N (eso es F2-1/F2-2). Se usa N=200 fijo (modo `produccion` de
  CS074-rcruz) por presupuesto de cómputo; se anota como limitación.
- No se implementa el NULL alternativo de secuencia de cortes (eso es F2-6).
- No se toca topología ni el archivo `cs074_rcruz.py`.

---
*Fin del pre-registro. El motor (`F2_5_engine.py`) se escribe y corre DESPUÉS de
este archivo, sin modificarlo salvo error tipográfico anotado con fecha.*
