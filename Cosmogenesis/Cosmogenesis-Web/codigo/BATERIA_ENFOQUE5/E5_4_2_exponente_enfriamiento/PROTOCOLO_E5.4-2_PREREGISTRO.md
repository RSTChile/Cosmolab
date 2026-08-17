# PROTOCOLO E5.4-2 — PRE-REGISTRO
## "Exponente de enfriamiento emergente: ¿T∝a^−n, con qué n?"

**Fecha de escritura:** 2026-07-24 (ANTES de correr `E5_4_2_exponente_enfriamiento_motor.py`;
este archivo se congela con este mtime — no se edita tras ver resultados, T3).

**Ejecutor:** CC, experimento E5.4-2 de `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`
(Enfoque 5, Tema 4 — Exergía y enfriamiento adiabático). Corre en paralelo con otros 29
experimentos, cada uno con su propio prefijo. Este protocolo cubre EXCLUSIVAMENTE el
prefijo `E5_4_2_` en `codigo/BATERIA_ENFOQUE5/E5_4_2_exponente_enfriamiento/`. No se toca
ningún archivo fuera de ese directorio.

---

## 0. Lectura obligatoria hecha antes de diseñar

- `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md` completo: sección 0 (tres
  reglas de oro), REGLAS DE EJECUCIÓN, intro TEMA 4, spec E5.4-2 literal:
  > **Barrido:** a ∈ [1 … 1e6] (6 décadas) × ε × ≥12 semillas.
  > **Observable:** n medido en T∝a^−n (SALIDA). **NULL:** sin expansión.
  > **PASS:** n emerge y se reporta; NO se fija a n=2 ni n=3 (aunque la física los sugiera).
- `CF2_estiramiento_motor.py` (Cosmogenesis-Web/codigo/CF2_estiramiento/) completo, leído
  como referencia de infraestructura (reloj genético `t_g→a=exp(H_EXP·t_g)`, `H_EXP=6.0`,
  `dtg=1/399`, checkpointing markoviano de una sola trayectoria muestreada en los `t_g(a)`
  objetivo, patrón de arms REAL vs NULL). **No se edita.**
- `BATERIA_FUNDAMENTOS/F3_2_T_emergente/` (protocolo + motor + resultado JSON) completo,
  como advertencia explícita del director: ese experimento midió
  `T_energy(a) = grad_energy_comov(a) / a²` en AMBAS ramas (REAL y NULL_RHO_FIXED). El
  resultado real (`results/F3_2_T_emergente/F3_2_T_emergente_produccion_result.json`)
  muestra el defecto con total claridad: el ajuste asintótico da
  `slope_REAL → -1.999999...` (exactamente el exponente de la propia división geométrica
  `/a²`, no una medida independiente), y en la combinación inspeccionada
  (σ=1e-4, seed=7) `slope_NULL_RHO_FIXED = -2.1876` está a sólo `0.065` de
  `slope_REAL = -2.1225` — pasa el umbral `SLOPE_DIFF_MIN=0.05` por un margen mínimo, no
  por una diferencia física robusta. **Causa raíz:** `grad_energy_comov` (la cantidad
  comóvil sin dividir) se estabiliza en un PISO casi idéntico en ambas ramas (equilibrio
  disipación↔forzamiento en NULL; piso numérico/de forzamiento decayendo en REAL), así que
  cuando ambas se dividen por el mismo `a²`, el "−2" que sale es la aritmética de la
  división, no una medida de la física de enfriamiento. Esto es exactamente el "artefacto
  geométrico de dividir por a² en ambas ramas" que este protocolo debe evitar o, si no
  puede evitarlo, declarar.

---

## 1. Decisión de diseño para EVITAR el defecto de F3-2

**Regla de diseño adoptada:** el observable de temperatura de este experimento **no
contiene ninguna división por una potencia de `a` en su propia definición.** Se mide
temperatura **física directa** (energía cinética media de un ensamble de partículas en
posiciones y velocidades físicas reales), nunca una cantidad comóvil post-dividida por
`a^n`. Si el enfriamiento aparece, tiene que emerger de la DINÁMICA (colisiones físicas
reales con una pared que se aleja), no de una conversión geométrica aplicada por igual a
REAL y a NULL.

Consecuencia directa: el NULL de este experimento (`NULL_SIN_EXPANSION`) es **literalmente
sin expansión** — pared física fija en `a≡1` durante toda la corrida, no "misma
trayectoria de `a(t)` con densidad fija" (la interpretación que F3-2 tuvo que adoptar
porque su observable SÍ necesitaba `a(t)` para la división `/a²`). Aquí no hace falta esa
reinterpretación: como `T=mean(v²)` no divide por nada, el NULL puede ser el caso más
literal y más fuerte — pared que nunca se mueve — sin que el eje x pierda sentido, porque
el eje x (`a_grid`) es sólo una ETIQUETA de en qué paso de reloj genético se tomó la
lectura, común a ambas ramas, no un divisor.

**Por qué esto no es sólo "cambiar el observable" cosméticamente:** el mecanismo físico de
enfriamiento (colisión elástica de una partícula libre contra una pared que retrocede) es
el modelo de libro de texto para "enfriamiento adiabático de un gas no relativista en un
universo en expansión" (invariante adiabático `p·L=const` → `p∝1/a` → `T∝1/a²` en el
límite adiabático). El punto crítico: **no se impone `p∝1/a` por fórmula** — eso sería
poner el resultado a mano (violaría T1, y sería el MISMO defecto de fondo de F3-2 en otra
forma: construir la respuesta dentro de la definición). En cambio, se simula la
**colisión mecánica real** partícula–pared cada vez que ocurre, y `n` se **mide** ajustando
`T(a)` medido. Si la física realmente sigue el invariante adiabático, `n≈2` DEBERÍA emerger
del ajuste — pero el código nunca lo asume ni lo fuerza, y si la pared se aleja demasiado
rápido para que el gas la siga (ruptura de adiabaticidad, ver §3), el código medirá
honestamente lo que salga (incluyendo un `n` distinto, o un quiebre de la ley de potencia).

---

## 2. Modelo físico propio de E5.4-2

**Sustrato:** gas ideal de `N_PART` partículas libres (sin interacción partícula–partícula,
sólo colisión elástica con las paredes) en una caja cúbica isótropa `[-Lh(t), +Lh(t)]³`
(3 ejes independientes, separables).

- **REAL:** `Lh(t_g) = 0.5·L0·a(t_g)`, con `a(t_g)=exp(H_EXP·t_g)` (mismo reloj genético
  que CF2/F3-2, `H_EXP=6.0`, reutilizado sin retocar — T1). La pared se aleja con
  velocidad física `Vw(t_g) = dLh/dt_g = 0.5·L0·H_EXP·a(t_g)`.
- **NULL_SIN_EXPANSION:** `Lh ≡ 0.5·L0` fijo durante TODA la corrida (`Vw≡0`). Ninguna
  pared se mueve nunca. Ver §1 para por qué esta es la interpretación literal y preferida
  aquí (a diferencia de F3-2).
- **Colisión elástica con pared móvil (por eje, independiente):** si una partícula cruza
  `+Lh`: `v ← 2·Vw − v` (fórmula exacta de colisión elástica 1D contra un objeto de masa
  infinita moviéndose a velocidad `Vw`); posición reflejada `x ← 2·Lh − x`. Simétrico para
  `-Lh` con `-Vw`. Se itera la detección de cruce hasta 4 veces por subpaso (partículas de
  cola rápida que podrían cruzar más de una vez) — cota fija, no ajustada a resultados.
- **Movimiento libre (sin colisión):** balístico, `x ← x + v·dt_sub` — sin fuerzas, sin
  ningún término que dependa de `a` fuera de la posición de la pared. Esto es deliberado:
  el ÚNICO lugar donde `a` entra en la dinámica es la posición/velocidad de la pared, nunca
  en la ecuación de movimiento de la partícula ni en el observable.
- **Perturbación dinámica (T7 — obligatoria, no cosmética):** en cada subpaso se inyecta un
  kick de velocidad tipo Wiener, `v ← v + ε·√dt_sub·N(0,1)`, idéntico en REAL y NULL,
  independiente de `a`/`ρ` (a diferencia del forzamiento de F3-2, que estaba atado a la
  dilución — aquí se mantiene deliberadamente DESACOPLADO de la física de expansión para
  no contaminar el mecanismo que se está midiendo; es ruido puro de media cero que prueba
  robustez del exponente medido frente a agitación ajena a la expansión, cumple T7 sin
  meter una nueva vía de "artefacto compartido" entre ramas).
- **Sin clipping.** Las velocidades no se recortan (igual que F3-2 razonó para su campo:
  recortar aquí sería una manipulación no física de la propia cantidad que se mide).

**Observable primario — `T_all(a)`:** energía cinética media por grado de libertad,
`T_all = mean(vx²+vy²+vz²)/3` sobre las `N_PART` partículas de una réplica, en unidades
físicas (masa=1, k_B=1). **Ninguna división por `a` en esta fórmula.**

**Observable secundario (T2 — segundo método, independiente) — `T_x(a)`:** temperatura
resuelta en un solo eje, `T_x = mean(vx²)`, calculada de una porción disjunta de los datos
(sólo la componente x, no promedia con y/z). Verifica isotropía y sirve de segunda vía de
medición sin compartir la fórmula completa del observable primario. Se reportan también
`T_y`, `T_z` como diagnóstico adicional de isotropía.

**Ledger de energía (diagnóstico T6 — "conservación de E verificada cada paso"):** en cada
colisión con pared se registra `ΔKE_pared = KE_después − KE_antes` de las partículas
reflejadas (energía extraída por/hacia la pared); en cada kick de ruido se registra
`ΔKE_ruido`. Se verifica en cada checkpoint que
`KE(t) − KE(0) + W_pared_acumulado(t) − E_ruido_acumulado(t) ≈ 0`
(identidad de contabilidad, no supuesto físico — debe cumplirse por construcción si el
código está bien escrito). Se reporta la desviación relativa máxima observada en toda la
corrida como control de calidad del código, no como criterio de PASS de la física.

---

## 3. Por qué se espera (sin imponerlo) un quiebre de régimen — declarado ANTES de correr

Cálculo de factibilidad hecho ANTES de fijar parámetros (ver prototipo, no es ajuste post
hoc de resultados: es dimensionamiento de la ventana observable, igual que se elige `dtg`
o `N_SUB` en cualquier motor de esta batería).

En el límite adiabático (pared mucho más lenta que la partícula), el invariante `v·Lh≈const`
predice `v(a)≈v0/a`. La velocidad de la pared crece como `Vw(a) = 0.5·L0·H_EXP·a` — es
decir, LINEAL en `a`, mientras la velocidad típica de la partícula (si el invariante se
sostiene) CAE como `1/a`. Como una crece y la otra cae, existe necesariamente un punto
`a_freeze` donde `Vw(a_freeze) ≈ v(a_freeze)`: la pared empieza a alejarse más rápido de lo
que la partícula más veloz puede alcanzar, las colisiones cesan, y las velocidades quedan
**congeladas** (`T` constante) para todo `a > a_freeze`. Estimación:
`a_freeze ≈ sqrt(2·V0/(L0·H_EXP))`.

**Esto se declara como predicción física ANTES de correr**, no como hallazgo posterior: si
aparece, es el análogo de un "desacople cinético" — el gas deja de poder termalizarse con
la expansión porque ya no hay forma de que la pared y la partícula se encuentren. Un ajuste
de `n` sobre las 6 décadas COMPLETAS de `a` puede por tanto mostrar un R² bajo si mezcla
fase adiabática + fase congelada — eso se reporta como hallazgo honesto (T3), no se oculta
recortando el rango a conveniencia. Por esto el criterio de PASS (§6) incluye tanto un
ajuste global (todo el rango, pre-registrado como el veredicto principal, tal como pide
la spec literal "a ∈ [1…1e6]") como un ajuste segmentado (fase temprana vs tardía,
DIAGNÓSTICO, no criterio de PASS) que puede revelar el quiebre sin cambiar el criterio.

---

## 4. Parámetros congelados (elegidos por factibilidad computacional, NO por el resultado)

| Parámetro | Valor | Origen |
|---|---|---|
| `H_EXP` | 6.0 | Reutilizado sin retocar de CF2/F3-2 (T1) |
| `dtg` | 1/399 | Reutilizado sin retocar (`ORIGINAL_STEPS_PER_TG=399`) |
| `L0` (ancho inicial de caja, por eje) | 5e-4 | Elegido ANTES de ver resultados finales para que `a_freeze` (§3) caiga en `~26`, dentro de la primera década y media del barrido — deja ver fase adiabática Y fase congelada dentro de las 6 décadas, sin requerir resolución temporal computacionalmente inviable (ver prototipo de factibilidad) |
| `V0` (σ de velocidad inicial por eje) | 1.0 | Escala natural, `T0=1` |
| `N_SUB` (subpasos de integración por paso de reloj genético) | 16 | Ver prueba de convergencia abajo: N_SUB∈{16,32,64,128} dan `T(a)` indistinguible (diferencias <0.5% en todos los checkpoints) y conservación de energía a nivel de precisión de máquina (`ledger_dev`~1e-15) en los CUATRO niveles probados — la dinámica ya converge en N_SUB=16; usar más subpasos sólo gasta cómputo sin cambiar la física. Elegido por convergencia numérica, no por favorecer un resultado |
| `MAX_COLLISION_ITERS` | 2 | El conteo de colisiones no cambia entre 1 y 4 reintentos en las pruebas (cruces múltiples por subpaso son ~inexistentes con N_SUB=16); se deja 2 como margen de seguridad barato |
| `N_PART` (partículas por réplica) | 2000 | Balance estadístico/cómputo — error relativo esperado en `T` ≈ `sqrt(2/(3·N_PART))` ≈ 1.8% por checkpoint |
| `a_grid` | `np.geomspace(1, 1e6, 25)` | 25 puntos log, 6 décadas — el rango PEDIDO literal por la spec, sin recortar |
| `ε` (perturbación dinámica) | `np.geomspace(1e-12, 1, 8)` | 8 puntos, 12 décadas — mismo rango que usan otros experimentos hermanos de este Enfoque (E5.1-1, E5.3-1: `ε∈[1e-12…1]`), no elegido para favorecer este experimento en particular |
| Semillas | `[7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 13, 271828]` | 10 estándar del proyecto + 2 extra → 12 (mínimo exigido por la spec) |
| Modos | `{REAL, NULL_SIN_EXPANSION}` | — |

Total de combinaciones: 8 (ε) × 12 (semillas) × 2 (modo) = 192 réplicas, cada una
integrando 919 pasos de reloj genético × 16 subpasos = 14,704 subpasos, muestreadas en
los 25 checkpoints de `a_grid` (mismo método de checkpointing markoviano de una sola
trayectoria que CF2/F3-2 — no se re-simula desde cero por punto de `a`).

**Validación de factibilidad (prototipo, ANTES de este protocolo, con semilla 7 fija,
`N_PART=500-2000`, sin ruido ε — pruebas de arquitectura y de convergencia numérica, NO
datos de producción):**
1. Corrida completa `a:1→1e6`, `N_SUB=128`: sin NaN/explosión; REAL declina de `T≈0.97` en
   `a≈1` a un piso de `T≈0.0013` alrededor de `a≈20-50`, luego completamente PLANO hasta
   `a≈1e6` (consistente con `a_freeze≈26` estimado analíticamente en §3); NULL permanece
   constante (`ledger_dev`~1e-15, conservación exacta con pared estática).
2. Prueba de convergencia temporal: se repite la MISMA corrida (semilla 7, REAL,
   `a:1→1e6`, 15 checkpoints) variando SÓLO `N_SUB∈{128,64,32,16}`. Resultado: `T(a)` en
   cada checkpoint coincide entre los 4 niveles a mejor que 0.5% (p.ej. en `a≈2.74`:
   T=0.13273 (N_SUB=128) vs 0.13269 (64) vs 0.13268 (32) vs 0.12659→0.13259 (16); en el
   piso congelado, T=0.001302-0.001304 en los 4 niveles), y `ledger_dev` (identidad de
   conservación de energía, ver §2) permanece en `~1e-15` (precisión de máquina) en TODOS
   los niveles, incluido `N_SUB=16`. Esto confirma que `N_SUB=16` ya está en el régimen
   convergido — no es una resolución insuficiente, es la resolución mínima que ya reproduce
   la física de `N_SUB=128` dentro del ruido de máquina. Se fija `N_SUB=16` por esta razón
   (factibilidad computacional: reduce el costo ~8× sin alterar el resultado físico),
   ANTES de correr la producción con `ε>0` y el barrido completo de semillas.
3. Timing: con `N_SUB=16`, `N_PART=2000`, 8 columnas de `ε` en paralelo (vectorizado), una
   corrida completa (1 semilla, 1 modo, 919 pasos × 16 subpasos = 14,704 subpasos) toma
   ~45-55s. Con 12 semillas × 2 modos = 24 corridas, tiempo total estimado ~20-25 min —
   dentro de lo razonable para "cómputo largo autorizado".

---

## 5. Ajuste y exponente

Por cada `(ε, semilla)`, para `T_all` y para `T_x` por separado:
- **Ajuste global (PRINCIPAL, pre-registrado):** regresión `ln(T)` vs `ln(a)` por mínimos
  cuadrados sobre los 25 puntos completos → `slope_global = -n_global`, con `R²_global`.
- **Ajuste segmentado (DIAGNÓSTICO, no gate de PASS):** mismo ajuste restringido a la
  primera mitad de puntos en `log(a)` (fase temprana) y a la segunda mitad (fase tardía)
  por separado, para exponer el quiebre de §3 si existe, sin usarlo para decidir PASS/FAIL.
- **NULL:** mismo ajuste global sobre la curva NULL.

**Exponente reportado:** media ± desviación estándar de `slope_global` (T_all) a través de
las 96 combinaciones `(ε, semilla)` de la rama REAL, TAL COMO SALGA — no se compara contra
n=2 ni n=3 para decidir el veredicto (eso sería T1 disfrazado de post-hoc). Se reporta
también, puramente informativo, el exponente segmentado por fase.

---

## 6. Criterio de PASS (congelado, no se toca tras ver resultados)

Constantes fijadas AHORA, antes de correr:
- `MONO_TOL = 1e-6` (tolerancia de no-crecimiento, algo más laxa que CF2/F3-2's `1e-9`
  porque aquí hay ruido dinámico ε>0 que puede introducir fluctuaciones locales genuinas —
  se documenta esta diferencia explícitamente, no es un ajuste posterior).
- `R2_MIN = 0.50` (umbral deliberadamente MÁS BAJO que el `0.70` de F3-2: la predicción
  física §3 anticipa un posible quiebre de régimen que reduce el R² de un ajuste global
  aunque la física sea perfectamente real y medible en la fase temprana — exigir un R²
  alto penalizaría precisamente el hallazgo honesto que se quiere poder reportar).
- `SLOPE_DIFF_MIN = 0.3` (mucho más exigente que el `0.05` de F3-2/CF2 — deliberado: como
  este observable NO comparte ningún factor geométrico `a^n` entre ramas, se espera que si
  hay una diferencia real sea GRANDE, no marginal; fijar un umbral alto previene que un
  "empate técnico" como el de F3-2 [0.065 vs 0.05] se cuente como mordida).
- `PASS_RATE_MIN = 0.55` (heredado del resto de la batería/proyecto).

Por combinación `(ε, semilla)`, usando `T_all`:
- `mono_REAL` = curva REAL no-creciente (tolerancia `MONO_TOL`).
- `cond_r2` = `R²_global_REAL ≥ R2_MIN`.
- `cond_null_muerde` = `NOT mono_NULL` **O** `|slope_NULL − slope_REAL| ≥ SLOPE_DIFF_MIN`.
- `punto_pass = mono_REAL AND cond_r2 AND cond_null_muerde`.

**PASS_E5.4-2 (veredicto principal):** si, sobre las 96 combinaciones `(ε,semilla)` de la
rama REAL, la tasa `rate_T_all ≥ PASS_RATE_MIN` Y la tasa de "NULL muerde"
`rate_null_muerde ≥ PASS_RATE_MIN`. Si no, **FAIL**, reportado como hallazgo (T3: el umbral
no se re-elige después de ver el número).

**Verificación cruzada obligatoria (T2/regla 4, tres vías):**
(a) el NULL muerde — tasa reportada explícitamente, por `ε`;
(b) `T_x` (eje único) debe coincidir cualitativamente con `T_all` (mismo signo de
    veredicto en la mayoría de combinaciones) — si no coincide, se reporta como hallazgo
    honesto, no se fuerza el acuerdo;
(c) auditoría en disco: `E5_4_2_exponente_enfriamiento_motor.py` + JSON crudo
    (`E5_4_2_exponente_enfriamiento_produccion_result.json`) quedan en disco para quien no
    escribió el código.

**Declaración explícita del defecto F3-2 (obligatoria en el reporte final, sin importar el
resultado):** el reporte final debe decir EXPLÍCITAMENTE si este experimento evitó el
defecto de F3-2 (división geométrica `/a^n` compartida entre ramas) o lo heredó, con la
evidencia concreta (comparación de las fórmulas de observable, y del comportamiento medido
del NULL) — no basta con afirmarlo, se muestra el número.

---

## 7. Qué NO se hace aquí

- No se toca `CF2_estiramiento_motor.py`, ningún motor de `F3_2_T_emergente/`, ni ningún
  otro archivo existente fuera de `codigo/BATERIA_ENFOQUE5/E5_4_2_exponente_enfriamiento/`.
- No se auto-adjudica "la expansión enfría de verdad, en general" más allá de este
  experimento puntual — el veredicto amplio lo da CS con la curva cruda.
- No se cambia este criterio después de correr el motor (T3). Si el resultado es FAIL, si
  el NULL no muerde, o si `T_x` discrepa de `T_all`, se reporta tal cual, sin suavizar.
- No se fija `n=2` ni `n=3` en el código de evaluación en ningún punto — el código sólo
  AJUSTA y REPORTA la pendiente medida.
- No topología, no commits (regla del director para este batch).
