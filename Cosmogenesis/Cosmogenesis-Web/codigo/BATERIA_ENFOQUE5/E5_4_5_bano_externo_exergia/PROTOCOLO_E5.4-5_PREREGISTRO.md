# PROTOCOLO E5.4-5 — PRE-REGISTRO
## "Control negativo: enfriamiento CON baño externo (lo prohibido) vs adiabático — y su exergía X"

**Fecha de escritura:** 2026-07-24 (ANTES de correr `E5_4_5_bano_externo_exergia_motor.py`; este
archivo se congela con este mtime — no se edita tras ver resultados, T3).

**Ejecutor:** CC, experimento E5.4-5 de `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`
(TEMA 4 — Exergía y enfriamiento adiabático, experimento 5, el último del tema). Corre en
paralelo con otros 29 experimentos, cada uno con su propio prefijo. Este protocolo cubre
EXCLUSIVAMENTE el prefijo `E5_4_5_`. No se toca ningún archivo fuera de
`E5_4_5_bano_externo_exergia/`.

---

## 1. Contexto y motivo

Todo el TEMA 4 (E5.4-1…E5.4-4) afirma que la exergía X aparece por **enfriamiento adiabático**
vía expansión: el gradiente comóvil se re-escala geométricamente por `a` y la difusividad se
diluye (`D=D0/a³`), sin ningún término de enfriamiento impuesto a mano. Esa afirmación solo es
creíble si el arnés experimental sabe distinguir ese mecanismo de la alternativa prohibida: que
el modelo estuviera, sin darnos cuenta, simulando un **baño térmico externo** (acoplamiento a un
reservorio a temperatura fija, como un termostato de laboratorio).

E5.4-5 es el control negativo pre-registrado por la spec (TEMA 4, experimento 5): mete **a
propósito** un baño externo explícito — algo que NO debe estar en el modelo principal (T0/T1) —
y muestra que el resultado es **cualitativamente distinto** del caso adiabático, tanto en `T(a)`
como en el observable central de este Enfoque, la **exergía X(a)**. Si no lo fuera, la
producción de exergía "emergente" de E5.4-1…E5.4-4 sería sospechosa de estar producida por un
baño encubierto.

**Este experimento no intenta pasar una hipótesis positiva sobre el campo físico.** Es un
chequeo de método: ¿el arnés experimental distingue bien enfriamiento geométrico de baño
térmico, EN AMBOS observables (T y X)?

Se reutiliza la metodología que tuvo éxito claro en el experimento casi idéntico previo
`Cosmogenesis-Web/codigo/BATERIA_FUNDAMENTOS/F3_6_bano_externo/` (PASS limpio, adiabático vs
baño se distinguieron con claridad en `T_fis`), leído completo y NO editado, extendiendo su
término de relajación tipo Newton hacia un reservorio a T fija con el observable adicional que
pide E5.4-5: la exergía X, no solo T.

---

## 2. Código base (NO se edita)

`Cosmogenesis-Web/codigo/CF2_estiramiento/CF2_estiramiento_motor.py` — leído completo, no se
toca. Se reutiliza tal cual su sello físico: campo `T` en malla `L×L` (L=64), salto tanh
inicial, difusión con laplaciano de 5 puntos (roll), reloj genético `t_g → a=exp(H_EXP·t_g)`,
brazo adiabático `ρ=ρ0/a³, D=D0·ρ/ρ0=D0/a³` (dilución geométrica, siempre presente, T1 prohíbe
quitarlo). También se lee completo (no se edita)
`Cosmogenesis-Web/codigo/BATERIA_FUNDAMENTOS/F3_6_bano_externo/F3_6_bano_externo_motor.py`, cuya
metodología de baño (término de relajación tipo Newton hacia objetivo comóvil `a·T_BANO`) se
reutiliza EXACTAMENTE.

El motor nuevo (`E5_4_5_bano_externo_exergia_motor.py`, archivo propio, prefijo `E5_4_5_`)
reimplementa esta física sin alterar ninguno de los dos originales, y añade lo que pide E5.4-5
que F3-6 no medía: el observable de **exergía X**.

No se toca `CF2_estiramiento_motor.py`, `F3_6_bano_externo_motor.py`, ni ningún archivo de otro
experimento de la batería (Enfoque 5 ni Fundamentos).

---

## 3. Modelo físico del baño externo (idéntico a F3-6)

El campo se mantiene internamente en representación **comóvil** `T_c(x,y)`. En cada subpaso de
difusión (`dt_sub = DT/N_SUB = 0.125`, idéntico a CF2/F3-6), además del laplaciano de difusión ya
existente, se aplica un término de relajación tipo Newton hacia un reservorio a temperatura
física FIJA `T_BANO=0.5` (idéntico valor que F3-6, T1: reutilizado, no re-elegido):

```
T_c ← T_c + dt_sub·D(a)·∇²T_c − dt_sub·κ·(T_c − a·T_BANO)
```

Con `κ=0` el término desaparece exactamente y el motor reduce **bit-a-bit** a la dinámica
adiabática de CF2/F3-6 (mismo laplaciano, mismo `D(a)=D0/a³`, misma condición inicial con ruido
1e-4, sin clip — ver razón declarada en F3-6 §3, heredada aquí sin cambios: el objetivo comóvil
`a·T_BANO` supera 1 al crecer `a`, recortar destruiría la física del baño).

**Estabilidad numérica — elección del `κ_max` (sobredimensionado respecto a F3-6):** el término
de baño es una contracción lineal exacta: `(T_c − objetivo)` se multiplica por `(1−κ·dt_sub)` en
cada subpaso. F3-6 usó `κ_max=1.0` (`κ·dt_sub=0.125`, "fuerte" pero con margen amplio). Aquí,
siguiendo la regla de oro de barrido sobredimensionado de ENFOQUE 5 (§0.1 de la spec: "rango
mucho mayor que donde se espera"), se sube el extremo superior a `κ_max = 1/dt_sub = 8.0`
exactamente: en ese punto `κ·dt_sub=1.0`, el factor de contracción es `(1−1.0)=0` — el campo
salta EXACTAMENTE al objetivo del baño en un solo subpaso (acople perfecto, el caso "fuerte" más
extremo que tiene sentido físico sin cruzar a oscilación de signo, que empezaría en
`κ·dt_sub>1`, es decir `κ>8.0`). Con `κ·dt_sub=1.0 ≪ 2` (umbral de estabilidad de Euler explícito
para una relajación lineal), no hay riesgo de blow-up. La difusión ya cumple el CFL de sobra
(`D0·dt_sub=0.015 ≪ 0.25`, idéntico a CF2).

---

## 4. Observables

**T_fis(a) = mean(T_comov)/a** — idéntico al observable primario de F3-6 (media espacial
GLOBAL, análogo a la dilución cosmológica `T∝1/a`).

**X_fis(a) = Var_espacial(T_comov)/a²  — OBSERVABLE NUEVO de E5.4-5, exergía.**
Justificación (T1: ningún coeficiente puesto a mano): en el régimen lineal cerca del
equilibrio, la energía libre disponible (capacidad de hacer trabajo, "exergía") de un campo de
temperatura escala a segundo orden con la varianza espacial del campo respecto a su media —
resultado estándar de termodinámica de fluctuaciones (la energía libre de una fluctuación
gaussiana ΔT es ∝(ΔT)²). Un campo perfectamente uniforme (Var=0) no puede hacer trabajo — cero
exergía, coincide con la definición operacional que usa E5.1-1 de la misma spec ("fracción de E
que puede hacer trabajo, desviación del equilibrio uniforme"). Se aplica el mismo convenio
físico/comóvil = campo/a que ya usan CF2/F3-6 para sus observables, pero elevado al cuadrado
(`/a²`) porque la varianza tiene unidades cuadráticas del campo. Se calcula sobre el campo
GLOBAL completo (L×L, sin banda), igual que `T_fis`: la varianza no sufre el artefacto de
wrap-around periódico que sí afecta al máximo del gradiente (por eso ESE observable sí usa
banda central), así que no hace falta restringir el dominio.

**Predicción pre-registrada (antes de correr):**
- Adiabático (`κ=0`): la difusión frena rápido porque `D=D0/a³→0`; la varianza COMÓVIL casi se
  congela (queda ≈constante), así que `X_fis` decae dominado por el re-escalamiento `/a²` —
  pendiente log-log esperada cerca de `−2`.
- Con baño fuerte (`κ grande`): el término `−κ(T_c−objetivo)` es una contracción IDÉNTICA en
  cada punto del espacio hacia el mismo objetivo, así que además de mover la media también
  aplasta la dispersión: `Var(T_c−objetivo)` decae extra por un factor `(1−κ·dt_sub)²` por
  subpaso, ADEMÁS del `/a²`. Consecuencia: mientras `T_fis` con baño fuerte se APLANA (converge
  a `T_BANO`, deja de caer), `X_fis` con baño fuerte debería CAER MÁS RÁPIDO que en el caso
  adiabático (pendiente más negativa, o colapso directo a ~0 numérico) — el baño no solo cambia
  el nivel medio, BORRA la estructura que sostiene la capacidad de hacer trabajo. Firma
  cualitativa distinta por observable: T se aplana, X se derrumba. Se reporta tal cual salga.

**Secundario, verificación cruzada (T2, observable ≠ juez):**
`grad_fis(a) = max_{banda central}|∂_x T_c|/a` — idéntico al de CF2/F3-6. No gatea el veredicto.

**Tercero, verificación adicional (liga con el axioma E1 de la batería, informativo, no
gatea):** `E_comov_sum(a) = sum(T_c)` — bajo difusión pura de 5 puntos en malla periódica, la
suma total del campo comóvil se conserva EXACTAMENTE (el laplaciano de `roll` suma cero). Con
`κ=0` esta cantidad debe permanecer prácticamente constante en toda la corrida (verificación de
que el brazo adiabático respeta el axioma E1 declarado por la batería: "el presupuesto se
conserva"). Con `κ>0` el baño bombea/drena activamente el campo hacia `a·T_BANO`, así que
`E_comov_sum` NO se conservará — eso ES la firma física de "lo prohibido": un baño externo viola
E1 por construcción (inyecta/retira energía sin pagarla dentro del sistema), mientras que la
expansión adiabática (E2: solo redistribuye) no. Se reporta la deriva relativa
`|E_comov_sum(a)−E_comov_sum(a=1)| / |E_comov_sum(a=1)|` por `κ`, sin que gatee el veredicto
principal (que es sobre T y X, según pide la spec).

---

## 5. Barrido pre-registrado (sobredimensionado)

| Parámetro | Rango | Puntos | Espaciado |
|---|---|---|---|
| `κ` (acople_bano) | {0} ∪ [1e-3, 8.0] | 8 | 0 exacto + 7 log (geomspace) — `κ_max` 8× mayor que F3-6 |
| `a` (factor de expansión) | [1, 1e4] | 9 | log (geomspace) — 4 décadas, sobredimensionado respecto a F3-6 (3 décadas), en línea con el rango de E5.4-1 de esta misma spec (`a∈[1…1e4]`) |
| semillas | ver lista abajo | 12 | — (10 estándar del proyecto + 2 nuevas) |

`KAPPA_GRID = [0.0] + geomspace(1e-3, 8.0, 7)` (8 puntos, incluye 0 exacto).
`A_GRID = geomspace(1.0, 1e4, 9)` (9 puntos, 4 décadas).

Semillas: `[7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 4242, 161803]`
(10 estándar del proyecto, T1: reutilizadas sin re-elegir + 2 nuevas propias de este
experimento, distintas de las que usó F3-6).

Total de trayectorias físicas: 8 (κ) × 12 (semillas) = 96, cada una integrada de forma
markoviana y muestreada en los 9 checkpoints de `a` (idéntico método de checkpointing de
CF2/F3-6: una sola trayectoria por (κ, semilla), sin re-simular desde cero por punto de `a`).

**NULL:** no aplica — declarado igual que F3-6 (control de método, no una hipótesis que
necesite barajado). El propio brazo `κ=0` (adiabático puro) es el control central.

---

## 6. Criterio de PASS (congelado, no se toca tras ver resultados)

Por cada semilla, se ajusta `slope(κ) = d(ln·)/d(ln a)` (mínimos cuadrados) y se lee el
`valor_final(κ)` en `a=a_max=1e4`, para CADA uno de los 8 `κ` (curva entera, T5), tanto para
`T_fis` como para `X_fis`.

**Umbrales para T_fis (idénticos en espíritu a F3-6, no re-elegidos sin razón):**
- `SLOPE_T_ADIABATICO_MAX = -0.5` → `cond_adiabatico_T`: `slope_T(κ=0) <= -0.5`.
- `SLOPE_T_FLATTENING_MIN = 0.3` → `cond_bano_aplana_T`: `slope_T(κ_max) - slope_T(κ=0) >= 0.3`.
- `cond_convergencia_T`: `|valor_final_T(κ_max) - T_BANO| < |valor_final_T(κ=0) - T_BANO|`.

**Umbrales para X_fis (nuevos, derivados de la predicción §4, con margen amplio):**
- `SLOPE_X_ADIABATICO_MAX = -1.0` → `cond_adiabatico_X`: `slope_X(κ=0) <= -1.0` (se espera
  ≈−2 por el re-escalamiento `/a²`; se deja margen amplio bajo 0, igual de conservador que el
  margen que usó F3-6 para T: umbral a mitad de camino entre 0 y el valor teórico).
- `SLOPE_X_STEEPENING_MIN = 0.3` → `cond_bano_derrumba_X`: `slope_X(κ=0) - slope_X(κ_max) >= 0.3`
  (nótese el signo INVERTIDO respecto al de T: aquí el baño debe hacer la pendiente MÁS
  negativa — X colapsa más rápido, no se aplana).
- `X_RATIO_BANO_ADIABATICO_MAX = 0.5` → `cond_bano_agota_X`:
  `valor_final_X(κ_max) <= 0.5 · valor_final_X(κ=0)` (el baño debe dejar, en `a=a_max`, como
  máximo la mitad de la exergía que sobrevive en el brazo adiabático).

`seed_pass = cond_adiabatico_T AND cond_bano_aplana_T AND cond_convergencia_T AND
cond_adiabatico_X AND cond_bano_derrumba_X AND cond_bano_agota_X`

**Veredicto del experimento:** `PASS_E5_4_5` si `rate = fracción de semillas con
seed_pass=True >= PASS_RATE_MIN = 0.55` (umbral reutilizado de CF-2/F3-1/F3-6, T1: no
re-elegido). Si `rate < 0.55`, el veredicto es `FAIL_E5_4_5` — se reporta tal cual (T3).

**Lectura de la spec (E5.4-5, verbatim):** "PASS: el caso adiabático (acople=0) difiere
claramente del caso con baño en X y en T(a); confirma que no hay baño encubierto." — el
criterio anterior es la traducción operacional exacta: dos observables, dos firmas
cualitativas DISTINTAS (T se aplana, X se derrumba), ambas deben cumplirse.

---

## 6-bis. Enmienda de calibración (ANTES de producción, documentada por transparencia)

Durante el smoke test (previo a la corrida de producción, mismo día 2026-07-24), se detectó
que el ajuste `slope(κ)` por mínimos cuadrados log-log, tal como estaba escrito arriba, se
rompe en el extremo `κ_max=8.0`: con `κ·dt_sub=1.0` el factor de contracción es EXACTAMENTE 0,
así que el campo salta bit-a-bit al valor uniforme del objetivo del baño en un solo subpaso, y
`Var_comov` (y por tanto `X_fis`) y `grad_fis` caen a **exactamente 0.0** en punto flotante. El
ajuste original recortaba esos ceros a `1e-300` antes de tomar logaritmo, lo que introduce
`log(1e-300)≈-690` como un outlier de leverage extremo en un ajuste de solo 8-9 puntos,
produciendo pendientes sin sentido físico (se observó `slope_X(κ_max)=+26` en el smoke, cuando
la física predice más colapso, no menos).

**Esto es un defecto del estimador de pendiente, no de la física ni del criterio de PASS**: el
colapso exacto a cero ES la señal de derrumbe de exergía más fuerte posible — no existe una
"pendiente de ley de potencia" que ajustar cuando la cantidad es idénticamente cero en varios
puntos consecutivos.

**Corrección aplicada (antes de correr producción, sin tocar ningún umbral numérico del §6):**
se añadió `loglog_slope_robusto()`, que filtra del ajuste los puntos cuyo valor absoluto cae
por debajo de `1e-18` del máximo de su propia curva (separa señal física de ruido de redondeo
tras un colapso exacto), y si sobreviven menos de 2 puntos reporta `colapso_total=True`
explícitamente (bandera, no una pendiente inventada). En `evaluate_seed`, `colapso_total_X` en
`κ_max` se trata como "más empinado que cualquier caso no colapsado" — satisface trivialmente
`cond_bano_derrumba_X`, que es la lectura físicamente correcta (colapso total = derrumbe
máximo). Ninguno de los cinco umbrales numéricos del §6
(`SLOPE_T_ADIABATICO_MAX`, `SLOPE_T_FLATTENING_MIN`, `SLOPE_X_ADIABATICO_MAX`,
`SLOPE_X_STEEPENING_MIN`, `X_RATIO_BANO_ADIABATICO_MAX`) se modificó. Se aplica la misma
corrección a `evaluate_seed_grad` (observable secundario, no gatea) por la misma razón —
`grad_fis` de un campo bit-a-bit uniforme es exactamente 0.

Esta enmienda se hizo ANTES de correr el barrido de producción completo (T3: no se tocó nada
después de ver resultados de producción); se documenta aquí por transparencia total del
proceso, no para ocultar el ajuste.

---

## 7. Verificación cruzada (tres vías, obligatorias)

(a) **NULL / control de método:** el brazo `κ=0` reduce exactamente a la dinámica de CF2/F3-6
    (mismo laplaciano, mismo `D(a)`, misma condición inicial). Se verifica que su `slope_T` y
    forma de curva son consistentes con lo reportado por F3-6.
(b) **Segundo observable independiente:** `grad_fis(a)` (banda central, idéntico a CF2/F3-6)
    reporta su propia curva `slope(κ)`/`valor_final(κ)`. No gatea el veredicto.
(c) **Tercera verificación, axioma E1:** `E_comov_sum(a)` por `κ` — deriva ≈0 en `κ=0`,
    deriva creciente con `κ` (la firma de "baño = violación de E1"). Informativa, no gatea.
(d) **Auditoría en disco:** código (`E5_4_5_bano_externo_exergia_motor.py`) + JSON crudo
    (`E5_4_5_bano_externo_exergia_produccion_result.json`) quedan en disco para que alguien que
    NO escribió el código pueda re-verificar sin depender del reporte narrado.

---

## 8. Qué NO se hace aquí

- No se toca `CF2_estiramiento_motor.py`, `F3_6_bano_externo_motor.py`, ni ningún archivo de
  otro experimento (Enfoque 5 ni Fundamentos).
- No se auto-adjudica ninguna conclusión sobre el TEMA 4 completo — E5.4-5 solo verifica que el
  arnés distingue enfriamiento adiabático de baño externo, en T y en X; no prueba ni refuta la
  producción de exergía "emergente" que reportan E5.4-1…E5.4-4 por sí misma.
- No se cambia este criterio después de correr el motor (T3). Si el resultado es FAIL, o si
  `κ=0` y `κ>0` NO difieren claramente en alguno de los dos observables, se reporta como
  hallazgo — sería una alerta seria sobre el arnés del TEMA 4 completo. Se PARA y se reporta a
  CS, no se re-interpreta a mano.
- No topología, no commits (regla del director para este batch).
- No se auto-adjudica el veredicto final de la batería de 30 — CS decide.
