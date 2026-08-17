# PROTOCOLO E5.1-5 — Persistencia de exergía bajo expansión no monótona (historias H(t) variadas)

**Fecha de pre-registro:** 2026-07-24, antes de escribir/correr el motor.
**Autor:** CC (E5.1-5), sobre la base física de `cs074_rcruz.py` (CS074-rcruz, NO editado).
**Ficha origen:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, TEMA 1, E5.1-5.
**Referencia de diseño de perfiles H(t):** `BATERIA_FUNDAMENTOS/F2_5_expansion_no_uniforme/
F2_5_engine.py` + `PROTOCOLO_F2-5_PREREGISTRO.md` (Enfoque 2, experimento previo — leído,
NO editado; el mecanismo de perfiles se REUSA con extensión propia, el observable central
cambia de "persistencia P" a "exergía X_final").

Este documento fija el observable, el NULL, el criterio de PASS y los rangos del barrido
ANTES de correr el motor. Si algo falla, se reporta el FAIL — este archivo no se edita
después de ver resultados (T3), salvo error tipográfico anotado con fecha.

---

## 1. Pregunta

La ficha E5.1-5 pregunta: si la expansión acelera y frena (H(t) no constante en el
tiempo, distintas HISTORIAS), ¿la exergía (capacidad de hacer trabajo, medida como
desviación del equilibrio uniforme) aguanta igual? **PASS pre-registrado de la ficha:**
X depende del r efectivo INTEGRADO, no del perfil específico; si un perfil rompe eso,
se reporta explícitamente (no se esconde bajo un promedio).

## 2. Física reusada de CS074-rcruz (sin tocar el archivo original)

Se importa `cs074_rcruz.py` por ruta (`importlib.util`), sin editarlo:

- Campo continuo `phi` en anillo de N sitios; condición inicial = fondo=1 + mancha ε
  (5 modos de Fourier con fases aleatorias, normalizada a std=1) — `campo_inicial`.
- Difusión: `paso_difusion` — SOLO por aristas vivas, local (media de vecinos activos).
- Expansión: `paso_expansion(activo, H, rng)` — cada arista viva se corta con
  probabilidad H en ese paso (Bernoulli). Aquí H puede variar paso a paso: H = H(t).
- D (difusividad) se MIDE igual que CS074-rcruz: fracción de contraste borrada en UN
  paso de difusión pura (H=0), promediada sobre semillas. No se impone.
- `pasos` (duración de la corrida) se CALIBRA igual que CS074-rcruz:
  `medir_pasos_lavado` — tiempo medido (mediana sobre semillas) para que a H=0 la
  persistencia caiga bajo P_LAVADO=0.05, con margen ×1.15. No se elige a mano.
  Calibración de referencia: N=200, eps=1e-3, 16 semillas (idéntico a `produccion` de
  CS074-rcruz y a F2-5) → D y `pasos_fijo` medidos UNA vez y reusados en todo el grid
  (incluido eps=0), igual convención que F2-5.

### 2.1 Verificación previa de E1 (conservación) sobre la física reusada — HALLAZGO A REPORTAR

Antes de correr el grid completo se midió si `sum(phi)` (candidato a "E_total") se
conserva EXACTAMENTE bajo `paso_difusion` cuando hay aristas cortadas (conectividad
parcial). Prueba numérica (N=50, 40% aristas cortadas al azar, 20 pasos de difusión
pura): deriva relativa de `sum(phi)` ≈ 7.5e-5 tras 20 pasos (no cero exacto). Con
expansión progresiva (H=0.05/paso, 30 pasos): deriva relativa ≈ 5.0e-6.

**Causa:** en un sitio con un solo vecino activo, la regla `nuevo = phi + 0.5*(vecino
− phi)` NO es recíproca con la regla que aplica ese vecino si el vecino tiene 2 aristas
activas (mueve solo 1/4 hacia este sitio) — la conservación par-a-par exacta de
`paso_difusion` requiere simetría de grado, que se rompe en las fronteras de segmentos
aislados por la expansión. Es una propiedad ESTRUCTURAL de la física ya escrita en
`cs074_rcruz.py` (no un bug de este motor, no se toca el archivo base), pequeña en
magnitud (~1e-4 a ~1e-6 relativo en las escalas probadas) pero NO estrictamente cero.

**Decisión (T6, sin parar el experimento):** se declara E_total := sum(phi) como el
observable de conservación del axioma E1, y se AUDITA en cada corrida del grid completo
(no solo en la prueba previa): se registra `E_drift_rel_max` = máxima deriva relativa
de `sum(phi)` respecto del valor inicial, muestreada en puntos a lo largo de la corrida
(inicio, 25%, 50%, 75%, fin). Esto NO es un error ajeno que detenga el experimento —es
precisamente el tipo de caracterización que el TEMA 2 (E5.2-x) de esta batería está
diseñado para hacer en profundidad; aquí se reporta como auditoría lateral honesta (T1:
no se esconde) y queda anotado para quien corra E5.2-x.

## 3. La exergía — definición operacional (E5.1-5, ligada a la ficha E5.1-1)

La ficha E5.1-1 define X_final como "fracción de E que puede hacer trabajo (desviación
del equilibrio uniforme)". El motor de CS074-rcruz no calcula una "exergía" con ese
nombre, pero ya calcula `std_ratio = phi.std() / contraste0` (contraste0 = std inicial),
que es EXACTAMENTE esa cantidad operacionalizada: la fracción del contraste (dispersión
respecto del equilibrio uniforme, que es phi_i = mean(phi) ∀i) que sobrevive respecto
del contraste inicial. Se adopta:

```
X_final := std_ratio_final = std(phi_final) / std(phi_inicial)
```

Es adimensional, en [0, ~1] (puede superar 1 solo si la difusión reconcentrara
varianza, no esperado bajo esta física), X_final=0 en el equilibrio uniforme exacto,
X_final≈1 si nada se degrada. Definición CONGELADA antes de correr (T3).

**Observable secundario (cross-check independiente, T2):** persistencia P (idéntica a
CS074-rcruz: `corr(phi, roll(phi,1))_{≥0} × var(phi)/contraste0²`), que combina forma
(autocorrelación espacial) y magnitud. Si X_final dice "sobrevive" y P dice "se
descorreló", el punto se marca en conflicto (T2: el observable no es su propio juez, se
exige una segunda vía).

**Tercer observable de auditoría:** `frac_exp` (fracción de aristas cortadas al final)
→ se deriva `r_efectivo_realizado` (ver §5), y `E_drift_rel_max` (§2.1).

## 4. Los perfiles H(t) — 6 perfiles, extensión del método F2-5

Cada perfil es una forma adimensional w(t) con media temporal discreta EXACTA = 1 (se
verifica en código, no solo en teoría). H(t) = clip(H_bar · w(t), 0, 1), con
`H_bar = r_medio · D` (D medido). `r_medio` es el eje barrido (rol análogo a "r" en
CS074-rcruz, aquí como promedio temporal nominal).

| Perfil | w(t), t=0..pasos−1 | Intuición |
|---|---|---|
| `constante` | w(t)=1 | referencia (igual a CS074-rcruz / F2-5) |
| `acelerante` | w(t) = 2·(t+0.5)/pasos | empieza casi sin expandir, termina expandiendo fuerte |
| `frenante` | w(t) = 2·(pasos−t−0.5)/pasos | espejo: empieza fuerte, termina casi sin expandir |
| `rafaga_temprana` | primer 10% del tiempo w=5.0; resto w=(1−0.1·5)/(1−0.1)≈0.556 | una ráfaga intensa al inicio, luego expansión lenta sostenida |
| `rafaga_tardia` | espejo temporal de `rafaga_temprana` (ráfaga al final) | mismo total, orden invertido |
| `rafagas_multiples` | 5 ráfagas iguales (w=5.0) de ancho 2% del tiempo c/u, equiespaciadas, mismo presupuesto total de ráfaga (10%) que los perfiles de una sola ráfaga; fondo w≈0.556 igual que arriba | ráfagas PLURALES repetidas (pide la ficha: "acelerante, frenante, ráfagas") — separa "una ráfaga grande" de "muchas ráfagas chicas" al mismo r_medio nominal |

Los parámetros de FORMA (fracción en ráfaga=10%, amplitud de ráfaga=5.0, número de
sub-ráfagas=5) son elecciones de DISEÑO EXPERIMENTAL (qué tratamientos comparar), no
coeficientes físicos ajustados a un blanco — igual convención que F2-5 (T1 aplica a la
física/observable, no a qué condiciones de tratamiento se prueban).

Se registra por corrida: `w_mean_exacto` (media real de w antes del clip) y
`H_bar_realizado` (media real de H(t) tras el clip); si el clip distorsiona la media
nominal en más de 1%, se marca `clip_afectado=true` (no se esconde).

## 5. r efectivo integrado — verificación cruzada central

Tras cada corrida se mide `frac_exp` (fracción de aristas cortadas al final). Se define
el r efectivo REALIZADO (equivalente-constante que habría dado la misma supervivencia
de aristas en `pasos` pasos):

```
r_efectivo = -ln(max(1 - frac_exp, 1e-12)) / pasos / D
```

Esta es la vara común para comparar perfiles distintos: si dos perfiles con el mismo
`r_medio` nominal terminan con `r_efectivo` distinto (por el clip o por la no-linealidad
de "arista ya cortada no se puede re-cortar"), el eje correcto para el veredicto de
robustez es `r_efectivo`, no `r_medio` nominal.

## 6. Barrido pre-registrado (sobredimensionado, regla del director)

- N = 200 (modo `produccion` de CS074-rcruz / F2-5, por presupuesto de cómputo —
  limitación anotada, no se barre N aquí).
- eps ∈ {0.0, 1e-3} — 0.0 = control estricto (X_final y P deben ser 0 a todo r y todo
  perfil); 1e-3 = amplitud de señal de la calibración de referencia de CS074-rcruz.
- r_medio ∈ {1e-3, 1e-2, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 1000} — 11 puntos log,
  6 DÉCADAS completas (mismo rango sobredimensionado que E5.1-1: r∈[1e-3…1e3]), cruza
  r≈1 con densidad extra alrededor de la transición esperada.
- perfiles: los 6 de §4.
- semillas: 16 por combinación (≥12 pedido por la ficha), semillas 3000..3015 (offset
  propio E5.1-5, para no pisar el espacio de semillas de otros experimentos paralelos:
  CS074-rcruz usa 1000+; F2-5 usa 2000+).
- D y `pasos` se miden una vez (N=200, eps=1e-3, 16 semillas) y se reusan en todo el
  grid (pasos_fijo), igual criterio que F2-5.
- Total combinaciones REAL+NULL: 2 eps × 6 perfiles × 11 r_medio × 16 semillas × 2
  (real/null) = **4224 corridas** del motor.

### 6.1 Mini-estudio de jitter dinámico adicional (T7, refuerzo)

El eje "perfil H(t)" YA ES una perturbación dinámica (no es solo cambiar semilla). Como
refuerzo (igual método que F2-5): jitter multiplicativo `H(t) → H(t)·ξ_t`,
`ξ_t ~ U(1−δ, 1+δ)` i.i.d. por paso, misma rng de la corrida (reproducible por semilla).

- δ ∈ {0.0, 0.3}.
- r_medio = 1.0 (zona de transición, la más sensible).
- perfiles: `constante` y `rafagas_multiples` (el más extremo/oscilatorio de la tabla).
- eps = 1e-3, semillas: 16 (mismas 3000..3015).
- Total: 2×2×16×2 = 128 corridas extra.

## 7. NULL

Barajado del campo (`rng.permutation(phi)`) DESPUÉS de toda la evolución — idéntico al
NULL de CS074-rcruz y F2-5. Mismo camino de expansión (misma secuencia de cortes) que
su pareja REAL (misma semilla), solo se destruye el orden espacial de phi al final.

## 8. Criterio de PASS (fijado ahora, no se toca después)

**PASS pre-registrado de la ficha:** X_final depende del r_efectivo integrado, no del
perfil específico. Operacionalización:

- (a) NULL debe cumplir X_final_null ≈ 0 y P_null ≈ 0 en todo el barrido (T4: si el NULL
  no muerde, el resto no se interpreta como señal).
- (b) eps=0 debe dar X_final≈0 (y P≈0) en TODOS los perfiles y r_medio — control
  estricto (T1).
- (c) Para cada bin de r_efectivo (bins log-anchos), la dispersión de X_final ENTRE
  PERFILES (a r_efectivo fijo) debe ser comparable a la dispersión ENTRE SEMILLAS
  dentro de un mismo perfil (razón ≤ 2×). Si excede 2× en algún bin, ESE perfil/bin se
  reporta como ruptura de la robustez (T1: no se esconde bajo un promedio).
- (d) El observable secundario P debe dar el mismo veredicto cualitativo (mismo signo
  de tendencia con r_efectivo) que X_final en cada perfil.
- (e) Se reporta `E_drift_rel_max` (§2.1) en todo el grid — informativo, no cambia (a)-(d).
- (f) El mini-estudio de jitter (§6.1) es informativo, no cambia el criterio (a)-(d); se
  reporta si el jitter cambia el orden relativo constante vs rafagas_multiples.

**Lecturas posibles (las tres son hallazgo, ninguna se privilegia a priori):**
- Robusto al perfil → r_efectivo integrado es la variable física real, el perfil (la
  "historia") es irrelevante para la exergía final.
- No robusto → se reporta CUÁL perfil, en qué rango de r_efectivo, con qué magnitud —
  evidencia de que "cuándo" importa además de "cuánto".
- Mixto (robusto en unos rangos, no en otros) → se reporta el mapa completo, sin
  promediar la ruptura local.

## 9. Qué NO se hace aquí (fuera de alcance de E5.1-5)

- No se barre N (E5.1-3 lo hace en 2D). N=200 fijo por presupuesto — limitación anotada.
- No se caracteriza en profundidad la conservación de E1 (§2.1) — eso es Tema 2 completo
  (E5.2-1..5); aquí solo se AUDITA y reporta como hallazgo lateral.
- No se toca topología ni `cs074_rcruz.py` ni `F2_5_engine.py`.
- No se ajusta ningún coeficiente hacia el 4.9%/31.5% (no aplica a este tema, es Tema 3).

---
*Fin del pre-registro. El motor (`E5_1_5_engine.py`) se escribe y corre DESPUÉS de este
archivo, sin modificarlo salvo error tipográfico anotado con fecha.*

---

## ADENDA — Definición común de exergía (ARREGLO 3), 2026-07-25

**No se edita el texto original arriba (T3): esta sección se agrega, no reemplaza.**

Contexto: el director (Alexis) detectó que los ~30 experimentos de Enfoque 5 se
diseñaron por separado y cada uno terminó con su propia fórmula de "exergía", lo que
impide comparar curvas entre experimentos. Se definió una fórmula ÚNICA y homologada
(Arreglo 3, `BATERIA_ENFOQUE5/_observables_homologadas.py`, tomada verbatim de
E5.2-2, ya corrida y PASS):

```
Xh(phi) = (1/N) * sum_i (phi_i - 1)^2      -- exergía canónica, ABSOLUTA, phi_eq=1
```

E5.1-5 se re-corre desde cero **con el mismo diseño exacto** (mismos perfiles H(t),
mismo `r_medio_list` de 11 puntos, mismo `EPS_LIST`, mismas 16 semillas
(3000..3015), mismo NULL por barajado espacial, mismo `pasos`/`D` medidos por
calibración) — el barrido pre-registrado en §6 arriba **no cambia**. Lo único que
cambia es qué se mide, no qué se corre:

1. **`Xh_final`** se calcula EN PARALELO a `X_final` (std_ratio, vieja, sin tocar),
   sobre el MISMO `phi` final de cada corrida — tanto rama REAL como rama NULL. La
   definición vieja (`X_final = std(phi_final)/std(phi_inicial)`, §3 arriba) se
   conserva íntegra para la comparación lado a lado; no se reemplaza ni se borra.
2. Se propaga `z_Xh` (mismo cálculo de z-score que `z_X`, sobre `Xh_real`/`Xh_null`)
   y se guardan `Xh_final_real_mean/std`, `Xh_final_null_mean/std` en cada fila del
   barrido principal y del mini-estudio de jitter (§6.1).
3. `analisis_robustez()` (criterio §8(c) del pre-registro: razón dispersión
   entre-perfiles / entre-semillas ≤2x → robusto, mismos bins log de r_efectivo) se
   corre DOS VECES sobre el mismo `filas`: una vez con los campos `X_final_real_*`
   (como antes) y otra con `Xh_final_real_*` — misma función, parametrizada, sin
   duplicar lógica (`campo_mean`/`campo_std`).
4. Se guarda detalle crudo: para cada una de las 132 filas del barrido principal y
   las 4 filas del mini-estudio de jitter, el array `phi_final` (real y null) y
   `phi_inicial` de las 16 semillas — N=200 es chico, es barato guardarlo completo,
   no se recurre a muestreo.
5. NO se toca la física (`cs074_rcruz.py`, `evolucionar_perfil`, `paso_difusion`,
   `paso_expansion`), NI el mecanismo de perfiles H(t), NI la observable secundaria
   `P` (persistencia, cross-check, idéntica a antes), NI `E_drift_rel_max`.

**Sobre el Arreglo 2 (ruido calibrado) — verificado, NO aplica aquí:** el único
mecanismo estocástico dinámico por paso de este motor es el jitter multiplicativo
sobre H(t) del mini-estudio §6.1 (`Ht_ef = clip(Ht · U(1-δ,1+δ), 0, 1)`, ver
`evolucionar_perfil()`) — perturba la TASA de expansión (probabilidad de cortar una
arista en ese paso), no suma ruido aditivo a `phi` acumulado sobre muchos pasos. Es
un mecanismo estructuralmente distinto al bug de Arreglo 2 (`phi = phi + noise_amp *
randn()` por paso, con `noise_amp` constante sin escalar por `1/sqrt(pasos)`) —
confirmado leyendo `_ruido_calibrado.py` y el código de este motor antes de decidir
no tocar nada de Arreglo 2 aquí.

**Predicción pre-registrada para esta re-corrida (declarada ANTES de correr, T3):**
como `Xh` (absoluta, ref. fija `phi_eq=1`) y `X` (relativa, `std_ratio` normalizado
por el propio `std` inicial de esa corrida) miden la MISMA cantidad física
subyacente (dispersión respecto del equilibrio uniforme) con distinta normalización,
se espera que sean monótonamente relacionadas dentro de cada `(eps, perfil,
r_medio)` fijo y que el ORDEN CUALITATIVO entre perfiles en la zona de transición
(hallazgo insignia: ráfaga temprana > frenante > constante ≈ ráfagas múltiples >
ráfaga tardía > acelerante) se preserve bajo `Xh`. Si no se preserva, es un hallazgo
genuino y se reporta tal cual, no se fuerza el resultado viejo.

Antes de sobrescribir, `E5_1_5_resultado_crudo.json` (definición vieja únicamente)
se conserva en disco como `E5_1_5_resultado_crudo_DEFINICION_VIEJA_pre_ARREGLO3.json`
para auditoría, no se borra.

No se corre nada de esta re-corrida hasta que esta ADENDA esté guardada en disco.
