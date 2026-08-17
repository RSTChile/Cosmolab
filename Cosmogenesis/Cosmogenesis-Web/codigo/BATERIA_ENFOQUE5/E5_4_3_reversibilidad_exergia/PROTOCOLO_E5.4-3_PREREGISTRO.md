# PROTOCOLO E5.4-3 — PRE-REGISTRO
## "Reversibilidad: si se detiene la expansión, ¿la exergía se re-degrada?"

**Batería:** ENFOQUE 5 — Energía · Exergía · Entropía (`BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`),
Tema 4 — Exergía y enfriamiento adiabático.
**Experimento:** E5.4-3.
**Ejecuta:** CC (agente paralelo, prefijo `E5_4_3_`), carpeta
`codigo/BATERIA_ENFOQUE5/E5_4_3_reversibilidad_exergia/`. No toca `CF2_estiramiento_motor.py`,
`F3_4_reversibilidad_termica_motor.py`, ni ninguna otra carpeta `E5_4_*` de los agentes en
paralelo (solo lectura si se consultan, y a fecha de este pre-registro E5.4-1/E5.4-2 aún no
habían publicado archivos — este protocolo es autocontenido, no depende de sus definiciones).

**Este documento se escribe y congela ANTES de correr el motor de producción.** El motor
(`E5_4_3_reversibilidad_exergia_motor.py`) y los resultados
(`results/E5_4_3_reversibilidad_exergia/`) se generan DESPUÉS — verificar mtime.

---

## 1. Pregunta

Si se detiene la expansión (`a` queda fijo desde ese instante en adelante) y solo se deja
correr difusión (+ eventual ruido dinámico) después, ¿la exergía ganada durante la expansión
se re-degrada (vuelve hacia el equilibrio uniforme) o queda congelada? Concretamente: ¿existe
un **tiempo de no-retorno** — un `t_g` de parada más allá del cual, incluso dándole a la
difusión tanto tiempo como tuvo la expansión completa, la exergía ya no puede re-degradarse
de forma apreciable?

**Predicción pre-registrada (antes de ver datos), heredada por analogía directa de
F3_4_reversibilidad_termica (mismo sustrato físico, distinto observable):** parar temprano
(`D` todavía grande, poca expansión acumulada) debería permitir que la difusión erosione la
exergía ganada; parar tarde (`D=D0/a³` ya casi nulo) debería dejarla congelada — con una
transición en algún punto intermedio del barrido de `a` de parada. Se pre-registra también,
como hipótesis secundaria motivada por la física del diseño (ver §2.3), que el ruido dinámico
(que NO se apaga con `a`, a diferencia de la difusión) puede impedir que exista un punto de
no-retorno limpio cuando su amplitud es grande frente a la exergía inicial (parametrizada por
`ε`, §2.2) — si esto ocurre, se reporta como hallazgo honesto, no se oculta ni se re-escala
el ruido para "arreglarlo" (T3).

## 2. Sustrato y definiciones (construidas para este experimento, T1: nada retocado tras ver datos)

### 2.1 Campo y dinámica (heredado del sustrato compartido CF2/F3_4, mismo motor físico)

Mismo campo continuo `T(x,y)` en grilla `L×L` (`L=64`), mismo reloj de expansión
`a(t_g)=exp(H_EXP·t_g)` con `H_EXP=6.0`, misma ley de dilución REAL (idéntica a CF2/F3_4, no
re-derivada aquí): `ρ=ρ0/a³`, `D=D0·(ρ/ρ0)=D0/a³` con `D0=0.12`. Difusión de 4 vecinos,
`DT=0.25`, `N_SUB=2`, `ORIGINAL_STEPS_PER_TG=399` (`dtg=1/399`) — sellos idénticos a
`CF2_estiramiento_motor.py`/`F3_4_reversibilidad_termica_motor.py`, reusados por convención de
proyecto (comparabilidad entre experimentos de la misma familia), no copiados como código.
Perfil inicial: salto tanh de ancho comóvil `W0=1.2` (idéntico a CF2/F3_4).

**Perturbación dinámica pre-registrada (T7):** en cada sub-paso de difusión se añade ruido
gaussiano aditivo de amplitud `σ` (escalado Euler-Maruyama por `√(dt/n_sub)`), barrida en
`RUIDO_DINAMICO_GRID = {0.0, 1e-3, 5e-3, 1e-2}` (4 puntos; `0.0` reproduce exactamente la
física sin ruido). Tras cada paso el campo se recorta a `[0,1]`.

### 2.2 Parámetro `ε` — amplitud de la diferencia inicial (NO es una cantidad hallada, es un
parámetro de diseño barrido, T1)

El documento autoritativo usa `ε` en todos los temas como la escala de la diferencia/estructura
inicial (S>0). Aquí se define de forma explícita y auditable:

```
perfil(x)      = 0.5·(1 − tanh(x/W0))                    # salto tanh, en [0,1]
T0(x,y)        = 0.5 + ε·(perfil(x) − 0.5) + ruido_semilla
```

`ε=1` reproduce el salto completo de CF2/F3_4 (contraste máximo T≈0↔T≈1). `ε→0` reproduce el
límite de "casi sin diferencia" (Tema 5, muerte térmica) — el campo arranca casi uniforme en
`T≈0.5`. `ruido_semilla = 1e-4·N(0,1)` (idéntico a CF2/F3_4, no escalado por `ε`) para no volver
el estado inicial perfectamente degenerado incluso en `ε=0`.

**Barrido pre-registrado:** `EPS_GRID = np.geomspace(1e-6, 1, 9)` — 9 puntos, 6 décadas. Es
deliberadamente sobredimensionado (regla del director, §0.1 del documento autoritativo): no se
centra donde se espera la transición (que, por construcción lineal de la difusión, se predice
INDEPENDIENTE de `ε` salvo por dos efectos no lineales que si son el objeto de barrido: (a) el
recorte `[0,1]` cerca de `ε≈1`, y (b) el piso de ruido dinámico (§2.3) dominando la señal cuando
`ε` es pequeño frente a `σ`).

### 2.3 Por qué `ε` importa pese a que la difusión es lineal (justificación física, no ajuste)

La ecuación de difusión (sin ruido) es lineal: si `T0 = μ + ε·δ0`, entonces `T(t) − μ =
ε·evoluciona(δ0, t)` exactamente (el operador laplaciano discreto y el operador identidad son
lineales), de modo que el COCIENTE de re-degradación (§3, definido como fracción relativa)
sería exactamente independiente de `ε` en ausencia de recorte y de ruido dinámico. El barrido de
`ε` existe precisamente para exponer los dos quiebres de esa linealidad — el recorte `[0,1]` (no
lineal, activo cerca de `ε≈1`) y el ruido dinámico de amplitud fija `σ` (no escala con `ε`, por
lo que domina cuando `ε` es chico) — sin fijar a mano dónde ocurre el quiebre.

### 2.4 Exergía — dos observables independientes (T2, y la "tercera verificación" de las REGLAS
DE EJECUCIÓN, §5 del documento: segundo método)

**Método 1 (primario, tipo "energía disponible" / APE — cuadrático, termodinámico):**
```
μ        = mean(T)                    # referencia de equilibrio (uniforme, misma E_total)
X_var    = Σ_(x,y) (T(x,y) − μ)²       # exergía-varianza: capacidad de trabajo, desviación
                                        # cuadrática del equilibrio uniforme
```
`X_var → 0` cuando el campo se homogeneiza; `X_var` grande cuando hay estructura (gradiente)
lejos del equilibrio.

**Método 2 (secundario, informacional — independiente, no usa derivadas ni resta cuadrática):**
```
S_local(x,y) = −T·ln(T) − (1−T)·ln(1−T)     # entropía binaria por celda (T como parámetro
                                              # de orden en [0,1], clip a [1e-12,1−1e-12])
X_info       = Σ_(x,y) [ln(2) − S_local(x,y)]   # exergía informacional: déficit de entropía
                                                  # respecto del máximo (T=0.5 en cada celda)
```
`X_info → 0` en el equilibrio uniforme `T≈0.5`; positivo cuando el campo se aparta de 0.5 en
cualquier dirección. Es una medida distinta en naturaleza (entrópica, no cuadrática) — sirve de
verificación cruzada de §2.4-Método 1, en el mismo espíritu que el par
`reaplan_∇`/`reaplan_Var` de F3_4.

### 2.5 Guardián E1/E2 (axiomas de la batería, §0 del documento autoritativo — verificados, no
supuestos)

`E1` (conservación del presupuesto total): `E_total = Σ_(x,y) T(x,y)`. La difusión de 4 vecinos
con condiciones periódicas conserva `Σ T` EXACTAMENTE (la suma de laplacianos discretos sobre
una grilla periódica es cero); la única fuente posible de deriva es el recorte `[0,1]` tras cada
paso (si algún valor se sale de rango) y el ruido dinámico (que sí puede mover la suma, al no
ser de divergencia nula por celda tras el recorte). Se mide `|E_total(t) − E_total(0)|` en
**cada** checkpoint de **cada** rama (T6: "toda etapa puede fallar") y se reporta la deriva
máxima observada por combo — no se asume conservación, se verifica.

`E2` (la expansión redistribuye, no crea, exergía): estructuralmente, `a` y `D` solo entran en
la ecuación de evolución como el coeficiente de difusión — nunca como fuente/sumidero de
`Σ T` — de modo que si `E1` se verifica (la suma se conserva), `E2` queda satisfecho por
construcción: toda la dinámica de `X_var`/`X_info` proviene de REDISTRIBUIR el mismo `E_total`
entre configuración estructurada (alta `X`) y uniforme (baja `X`), nunca de crear o destruir
`E_total`. Esto se reporta como verificación, no como supuesto no examinado.

## 3. Diseño experimental — bifurcación STOP vs NULL (metodología de F3_4, aplicada a `X`)

Para cada `(a_parada, ε, σ_ruido, semilla)`:

1. **Fase común de expansión** (REAL): se integra desde `t_g=0` con `D=D0/a³`, muestreando
   (checkpointing markoviano, mismo truco que CF2/F3_4) el campo EXACTO en cada punto del
   barrido de `a` de parada. Ambas ramas de abajo parten del mismo campo exacto en el instante
   de la bifurcación — la única diferencia es lo que pasa DESPUÉS.
2. **Rama STOP:** desde el checkpoint, `a` queda fijo en `a_parada` (`D` fijo en
   `D0/a_parada³`) durante una ventana `POST_STOP_TG` (idéntica para todos los puntos del
   barrido, §4), corre solo difusión (+ ruido dinámico).
3. **Rama NULL — "nunca parar"** (control pre-registrado por el documento autoritativo,
   E5.4-3: "NULL=nunca parar"): desde el MISMO checkpoint, la expansión CONTINÚA
   (`D=D0/a(t_g)³` sigue cayendo) durante la MISMA ventana `POST_STOP_TG`. Es la pregunta
   contrafactual exacta: "¿qué habría pasado si no hubiéramos parado?", todo lo demás idéntico.

Ambas ramas usan generadores de ruido independientes y deterministas,
`np.random.default_rng([seed, idx_checkpoint, idx_eps, idx_noise, codigo_rama])`.

**Observable central — re-degradación:**
```
redegrad_X(a_parada, rama) = (X_parada − X_final_rama) / X_parada
```
calculado para ambos métodos (`X_var`, `X_info`). `redegrad ≈ 1` ⇒ la exergía ganada se
re-degradó casi del todo; `≈ 0` ⇒ quedó congelada (sobrevivió). Cuando `X_parada` es
numéricamente despreciable (< `1e-12`, típico en `ε` muy chico donde el ruido de semilla domina
sobre la señal) se reporta `NaN` explícito — no se sustituye por 0 ni se excluye en silencio
(T3); cualquier condición de PASS que involucre un NaN se evalúa `False` (falla honesta, no
falso positivo).

## 4. Barrido (T7 — regla del director: sobredimensionado)

- **`a` de parada:** `np.geomspace(1.0, 1000.0, 10)` — 10 puntos, 3 décadas (mismo rango que
  CF2/F3_4; cumple el mínimo `≥10 puntos` exigido por el documento autoritativo para E5.4-3).
  `t_g de parada = ln(a_parada)/H_EXP`.
- **`ε` (amplitud de la diferencia inicial):** `EPS_GRID = np.geomspace(1e-6, 1, 9)` — 9 puntos,
  6 décadas (§2.2-2.3).
- **Ruido dinámico:** `RUIDO_DINAMICO_GRID = {0.0, 1e-3, 5e-3, 1e-2}` — 4 puntos.
- **Semillas:** las 10 semillas estándar del proyecto (`SEEDS_STANDARD_PROJECT`, idénticas a
  CF2/F3_4) más las 2 semillas de extensión ya usadas por F3-3/F3-4 (`271828`, `161803`) — total
  **12** (cumple `≥12 semillas`).
- **Ventana post-parada `POST_STOP_TG`:** `ln(1000)/H_EXP` — el mismo `t_g` que duró TODA la
  fase de expansión (mismo criterio que F3_4: se le da a la difusión, en cada punto del barrido,
  tanto tiempo como duró la expansión completa; constante única para los 10 puntos de parada,
  para que la comparación entre puntos sea justa).
- **Total de combos evaluados:** 10 (a_parada) × 9 (ε) × 4 (ruido) × 12 (semillas) = **4320**
  puntos de barrido, cada uno con 2 ramas (STOP, NULL) y 2 métodos de exergía (`X_var`,
  `X_info`) = 4320 × 2 = 8640 evaluaciones de re-degradación; más las 9×4×12=432 fases comunes de
  expansión (una por ε×ruido×semilla).

## 5. Criterio de PASS (congelado, T3 — no se toca si falla)

Por `(ε, σ_ruido, semilla)`, sobre el observable primario `redegrad_Xvar` de la rama STOP,
evaluado en los 10 puntos del barrido de `a_parada` (orden ascendente):

1. **`cond_a` — monotonicidad no-creciente** (tolerancia `MONO_TOL=0.05`, laxa por el ruido
   dinámico estocástico): `redegrad[i+1] ≤ redegrad[i] + MONO_TOL` para todos los pares
   consecutivos. Predicción: parar más tarde nunca re-degrada MÁS que parar más temprano.
2. **`cond_b` — el punto más temprano SÍ se re-degrada** (`REDEGRAD_EARLY_MIN = 0.5`):
   `redegrad[0] ≥ 0.5` en STOP. Si falla, ni parar casi al inicio re-degrada apreciablemente en
   la ventana dada — se reporta tal cual (T4, falsable, sin ajustar el umbral después).
3. **`cond_c` — el punto más tardío queda congelado** (`REDEGRAD_LATE_MAX = 0.1`):
   `redegrad[9] ≤ 0.1` en STOP.
4. **`cond_d` — el NULL muerde** (`DIFF_MIN = 0.1`): en el punto más temprano,
   `redegrad_STOP[0] − redegrad_NULL[0] ≥ 0.1` — parar debe re-degradar claramente MÁS que
   seguir expandiendo (la expansión continua sigue apagando `D`, protegiendo más la exergía). Si
   STOP y NULL no difieren, el experimento no discrimina — se reporta como hallazgo T4, no se
   descarta.

`seed_pass = cond_a AND cond_b AND cond_c AND cond_d` (evaluado con NaN-safe: cualquier NaN en
la curva ⇒ condición `False` ⇒ `seed_pass=False`, reportado aparte como "punto degenerado", no
mezclado silenciosamente con un fallo físico).

**Verdict global:** `rate = (#combos con seed_pass) / (9×4×12=432)`, `PASS_RATE_MIN = 0.55`
(mismo umbral que CF2/F3_4, no ajustado aquí). Se reporta también `rate` desglosado por `ε` y
por `σ_ruido` (no solo el agregado), porque §2.3 predice que el resultado puede depender
fuertemente de ambos.

**Punto de no-retorno (descriptivo, T5 — se reporta siempre la curva completa, no es un gate
adicional):** por combo, el primer `a_parada` (ascendente) donde `redegrad_STOP ≤
REDEGRAD_LATE_MAX` de forma SOSTENIDA (no vuelve a subir por encima del umbral en ningún punto
posterior). Si nunca se cumple: `no_retorno = None` ("todo el rango probado se re-degrada"). Si
se cumple desde el primer punto: `no_retorno = a_parada[0]` ("ya congelado desde el arranque del
barrido"). Se calcula también con `redegrad_Xinfo` (Método 2) como verificación cruzada — si
diverge cualitativamente del calculado con `X_var`, se reporta como hallazgo, no se descarta uno
de los dos (T3).

Si `rate < 0.55`, o si `cond_d` falla sistemáticamente (NULL no muerde), o si el punto de
no-retorno no existe o existe desde el inicio en la mayoría de los combos: se reporta el
FAIL/hallazgo con los números crudos. No se cambia el juez, no se sustituyen los observables, no
se ajustan los umbrales después de ver los datos (T3).

## 6. Qué NO es este experimento

- No mide masa, Higgs, ni linaje. Solo la reversibilidad del par exergía↔enfriamiento adiabático
  al DETENER la expansión (Tema 4, E5.4-3).
- No re-litiga si `D=D0/a³` es correcto — heredado como presupuesto (idéntico a CF2/F3_4), no
  re-derivado aquí.
- No define la eficiencia estructura/total (Tema 3) ni la descomposición de 3 vías `{X,
  degradada, ligada}` de E5.2-4 — este experimento solo necesita verificar conservación de
  `E_total` (guardián E1, §2.5) y medir `X`, no producir el desglose completo del presupuesto.
- No se auto-adjudica el veredicto de la hipótesis más amplia de la batería (Tema 4 completo,
  ni Enfoque 5) — eso lo hace CS con los números crudos.
- No toca `CF2_estiramiento_motor.py`, `F3_4_reversibilidad_termica_motor.py`, ni ninguna carpeta
  `E5_4_1/2/4/5_*` de los agentes en paralelo.
- La definición de `ε`, `X_var`, `X_info` es propia de este experimento (autocontenida, T1); no
  se asume que coincida con la que usen E5.4-1/E5.4-2 para "producción de exergía" — la
  reconciliación entre definiciones, si hace falta, es tarea de CS, no de este agente.

---

**Fecha/hora de este pre-registro:** ver mtime del archivo (se congela antes de generar
`E5_4_3_reversibilidad_exergia_motor.py` y cualquier resultado en
`results/E5_4_3_reversibilidad_exergia/`).
