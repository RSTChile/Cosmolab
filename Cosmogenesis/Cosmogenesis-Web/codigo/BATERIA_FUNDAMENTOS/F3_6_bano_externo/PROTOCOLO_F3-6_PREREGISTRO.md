# PROTOCOLO F3-6 — PRE-REGISTRO
## "Control negativo: enfriamiento CON baño externo (lo prohibido)"

**Fecha de escritura:** 2026-07-24, 05:35 (ANTES de correr `F3_6_bano_externo_motor.py`;
este archivo se congela con este mtime — no se edita tras ver resultados, T3).

**Ejecutor:** CC, experimento F3-6 de la batería `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`
(Enfoque 3, experimento 6, el último del enfoque). Corre en paralelo con otros 23
experimentos, cada uno con su propio prefijo. Este protocolo cubre EXCLUSIVAMENTE el
prefijo `F3_6_`. No se toca ningún archivo fuera de `F3_6_bano_externo/`.

---

## 1. Contexto y motivo (por qué existe este experimento)

Todo el Enfoque 3 (F3-1 … F3-5) afirma que el campo se enfría **sin** ningún término de
enfriamiento impuesto: la temperatura cae solo porque el gradiente comóvil se re-escala
geométricamente por `a` (`∇_fis = ∇_comov/a`) y porque la difusión se apaga al diluirse
la densidad (`D = D0/a³`). Esa afirmación solo es creíble si demostramos que sabemos
distinguir ese mecanismo de la alternativa obvia y prohibida: **que el modelo estuviera,
sin darnos cuenta, simulando un baño térmico externo** (acoplamiento a un reservorio a
temperatura fija, como un termostato de laboratorio).

F3-6 es el control negativo: mete **a propósito** un baño externo explícito — algo que
NO debe estar en el modelo principal (T0/T1) — y muestra que el resultado es
**cualitativamente distinto** del caso adiabático. Si no lo fuera (si con o sin baño se
obtuviera la misma curva T(a)), el enfriamiento "emergente" de F3-1…F3-5 sería
sospechoso de ser, en realidad, un baño encubierto.

**Este experimento no intenta pasar una hipótesis positiva sobre el campo físico.**
Es un chequeo de método: ¿el arnés experimental distingue bien las dos física?

---

## 2. Código base (NO se edita)

`Cosmogenesis-Web/codigo/CF2_estiramiento/CF2_estiramiento_motor.py` — leído completo
(no se toca). Se reutiliza tal cual su sello físico: campo `T` en malla `L×L` (L=64),
salto tanh inicial, difusión con laplaciano de 5 puntos (roll), reloj genético
`t_g → a = exp(H_EXP·t_g)`, brazo adiabático `ρ=ρ0/a³, D=D0·ρ/ρ0=D0/a³` (dilución
geométrica, siempre presente — es el "ingrediente físico" ya validado, T1 prohíbe
quitarlo, no solo prohíbe agregar cosas a mano). El motor nuevo
(`F3_6_bano_externo_motor.py`, archivo propio, prefijo `F3_6_`) reimplementa esta
física sin alterar el original y **añade exactamente un ingrediente nuevo**: el
acoplamiento a un reservorio externo a temperatura fija, con intensidad `κ`
(`intensidad_acople_bano`) que se barre desde 0 (= adiabático puro, reduce
exactamente al caso CF-2) hasta un valor "fuerte".

No se toca `CF2_estiramiento_motor.py`, `TEST_RHO_DISPERSION.py`, ni ningún archivo de
otro experimento de la batería (F3_1…F3_5, F1, F2, F4).

---

## 3. Modelo físico del baño externo

El campo se mantiene internamente en representación **comóvil** `T_c(x,y)`, igual que
CF2. En cada subpaso de difusión (mismo `dt_sub = DT/N_SUB = 0.125` que CF2), además
del laplaciano de difusión ya existente, se aplica un término de relajación tipo Newton
hacia un reservorio a temperatura física FIJA `T_BANO` (constante, no depende de `a`,
elegida una sola vez, `T_BANO = 0.5` — el punto medio del rango natural del campo
`[0,1]`; no se ajusta por ensayo y error, T1):

```
T_c ← T_c + dt_sub·D(a)·∇²T_c − dt_sub·κ·(T_c − a·T_BANO)
```

**Por qué el objetivo comóvil es `a·T_BANO` y no `T_BANO` directo:** el observable
reportado (§4) es siempre la cantidad FÍSICA `T_fis = T_c / a` (idéntica convención
`físico = comóvil / a` que usa CF2 para su gradiente). Para que un baño a temperatura
física fija `T_BANO` efectivamente empuje `T_fis → T_BANO` (independiente de `a`), el
forzamiento debe actuar sobre el objetivo comóvil `a·T_BANO`, porque
`d(T_c/a)/dt ⊃ −κ(T_c/a − T_BANO)` es equivalente, en la representación comóvil que ya
usa el motor, a `dT_c/dt ⊃ −κ(T_c − a·T_BANO)`. Esto es álgebra de la definición
física/comóvil ya aceptada en CF2, no un ajuste nuevo.

Con `κ=0` el término desaparece exactamente y el motor reduce **bit-a-bit** a la
dinámica adiabática de CF2 (mismo laplaciano, mismo `D(a)=D0/a³`, misma condición
inicial con ruido 1e-4).

**Diferencia declarada respecto a CF2 (T1: se documenta, no se esconde):** CF2 aplica
`np.clip(T,0,1)` después de cada paso, porque su campo representa una fracción
acotada. Aquí, para `κ>0`, el objetivo comóvil `a·T_BANO` puede superar `1` a medida
que `a` crece (p.ej. `a=1000, T_BANO=0.5 → objetivo=500`) — eso es esperado y correcto:
la representación comóvil de un baño a temperatura física fija y constante DEBE crecer
en unidades comóviles a medida que el espacio se expande. Recortar a `[0,1]` destruiría
por construcción la física que se quiere probar. Por eso **se omite el `clip` en este
motor**, en ambos brazos (`κ=0` y `κ>0`), para no introducir una discontinuidad de
comportamiento justo en `κ=0`. Se verifica en el JSON de salida que, para `κ=0`, la
diferencia frente a CF2 (que sí recorta) es numéricamente despreciable (el campo nunca
se sale de `[0,1]` de forma apreciable bajo difusión pura con esta amplitud de ruido).

**Estabilidad numérica:** el término de baño es una contracción lineal exacta hacia el
objetivo: `(T_c − objetivo)` se multiplica por `(1 − κ·dt_sub)` en cada subpaso. Con
`κ_max=1.0` y `dt_sub=0.125`, `κ·dt_sub=0.125 ≪ 2` (umbral de estabilidad de Euler
explícito para una relajación lineal) — sin riesgo de blow-up. La difusión ya cumple el
CFL de sobra en CF2 (`D0·dt_sub=0.015 ≪ 0.25`).

---

## 4. Observable exacto

**Primario — "temperatura física del campo":**
`T_fis(a) = mean(T_c) / a` — media espacial de TODO el campo comóvil (L×L, sin
restringir a banda; es una estadística global, no depende de dónde está el frente),
dividida por `a`. Análogo directo a la dilución cosmológica estándar `T∝1/a` de un gas
sin interacción (p.ej. temperatura de radiación del CMB). Bajo difusión pura con
condiciones periódicas, la media espacial del campo se conserva (el laplaciano de
`roll` suma cero sobre el dominio periódico), así que en el brazo adiabático (`κ=0`)
`T_fis(a) ≈ T_fis(1)/a` — ley de potencia limpia, exponente medido, no impuesto.

**Secundario — verificación cruzada, observable DISTINTO (T2):**
`grad_fis(a) = max_{banda central} |∂_x T_c| / a` — la misma abruptancia física de
gradiente que usa CF2/F3-1 (banda central `[L/8, 7L/8)` anti-wrap-around). Es una
estadística de forma/espacial (máximo de un gradiente local), no de nivel/media global
— independiente del observable primario por construcción. Se reporta su propia curva
`slope(κ)`/`valor_final(κ)` como chequeo cualitativo; NO define el veredicto principal.

---

## 5. Barrido pre-registrado

| Parámetro | Rango | Puntos | Espaciado |
|---|---|---|---|
| `κ` (intensidad_acople_bano) | {0} ∪ [1e-3, 1] | 8 | 0 exacto + 7 log (geomspace) |
| `a` (factor de expansión) | [1, 1000] | 7 | log (geomspace) — idéntico a `A_GRID` de CF2, reutilizado tal cual (T1: no se re-elige) |
| semillas | ver lista abajo | 12 | — (10 estándar del proyecto + 2 nuevas, ≥12 exigidas) |

`κ_grid = [0.0] + geomspace(1e-3, 1.0, 7)` (8 puntos totales, incluye 0 exacto). El
extremo superior `κ=1.0` da `κ·dt_sub=0.125`: en cada subpaso el 12.5% de la distancia
al objetivo del baño se recorre — "fuerte" en el sentido literal de la spec (arrastra
el campo de forma apreciable en la escala de tiempo de la simulación).

Semillas: `[7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 13, 271828]`
(10 estándar del proyecto + 2 nuevas, mismo patrón que usó F3-1).

Total de trayectorias físicas: 8 (κ) × 12 (semillas) = 96. Cada trayectoria evalúa
los 7 checkpoints de `a` a lo largo de UNA integración markoviana (idéntico método de
checkpointing de CF-2/F3-1: no se re-simula desde cero por punto de `a`).

**NULL:** no aplica (declarado en la spec F3-6: "es control de método", no una
hipótesis que necesite un barajado). El propio brazo `κ=0` (adiabático puro) hace de
control — es la comparación central del experimento.

---

## 6. Criterio de PASS (congelado, no se toca tras ver resultados)

Por cada semilla, se ajusta `slope(κ) = d(ln T_fis)/d(ln a)` (mínimos cuadrados sobre
los 7 puntos de `a`) y se lee `valor_final(κ) = T_fis` en `a=a_max=1000`, para CADA uno
de los 8 valores de `κ` (curva entera, T5 — no solo los extremos).

Predicción pre-registrada (sección 6 de la spec, verbatim):
> con baño, T→T_baño (no ley de potencia de a); sin baño, T∝a^(−n).

Se traduce a booleanos verificables:

- `cond_adiabatico`: `slope(κ=0) <= SLOPE_ADIABATICO_MAX = -0.5` (caída claramente en
  ley de potencia negativa; el valor teórico esperado por conservación de la media
  bajo difusión periódica es `slope≈-1`; se deja margen amplio, umbral fijado ANTES de
  correr, no ajustado después).
- `cond_bano_aplana`: `slope(κ=κ_max) − slope(κ=0) >= SLOPE_FLATTENING_MIN = 0.3`
  (la curva con baño fuerte debe ser claramente MENOS negativa — más plana — que la
  adiabática, por al menos 0.3 en pendiente log-log).
- `cond_convergencia`: `|valor_final(κ=κ_max) − T_BANO| < |valor_final(κ=0) − T_BANO|`
  (el brazo con baño fuerte debe terminar, en `a=a_max`, más cerca de `T_BANO=0.5` que
  el brazo adiabático).

`seed_pass = cond_adiabatico AND cond_bano_aplana AND cond_convergencia`.

**Veredicto del experimento:** `PASS_F3-6` si `rate = fracción de semillas con
seed_pass=True >= PASS_RATE_MIN = 0.55` (umbral reutilizado de CF-2/F3-1, T1: no se
re-elige para este experimento). Si `rate < 0.55`, el veredicto es `FAIL_F3-6` — se
reporta tal cual, no se cambia el criterio (T3).

**Adicionalmente (reporte descriptivo, no gatea el veredicto — T5: toda la curva, no
solo un punto):** se reporta `slope(κ)` y `valor_final(κ)` para los 8 puntos de `κ`,
promediados sobre semillas con su dispersión (std), para poder ver si la transición
adiabático→baño es monótona y gradual o abrupta.

---

## 7. Verificación cruzada (tres vías, obligatorias)

(a) **NULL / control de método:** el brazo `κ=0` reduce exactamente a la dinámica de
    CF-2 (mismo laplaciano, mismo `D(a)`, misma condición inicial). Se verifica en el
    JSON que su `slope` y forma de curva son consistentes con lo ya reportado por CF-2
    (caída monótona con `a`).
(b) **Segundo observable independiente:** `grad_fis(a)` (gradiente de banda central,
    idéntico al de CF2/F3-1) se calcula y se reporta su propia curva `slope(κ)` /
    `valor_final(κ)`. Si el aplanamiento con `κ` creciente aparece también en esta
    estadística de forma (no solo en el nivel medio), es evidencia independiente de
    que el baño está operando como se espera, no un artefacto del observable primario.
(c) **Auditoría en disco:** código (`F3_6_bano_externo_motor.py`) + JSON crudo
    (`F3_6_bano_externo_produccion_result.json`) quedan en disco para que alguien que
    NO escribió el código pueda re-verificar sin depender de este reporte narrado.

---

## 8. Qué NO se hace aquí

- No se toca `CF2_estiramiento_motor.py`, `TEST_RHO_DISPERSION.py`, ni ningún archivo
  de otro experimento de la batería.
- No se auto-adjudica ninguna conclusión sobre el Enfoque 3 completo — F3-6 solo
  verifica que el arnés distingue adiabático de baño-externo; no prueba ni refuta el
  enfriamiento "emergente" que reportan F3-1…F3-5 por sí mismo.
- No se cambia este criterio después de correr el motor (T3). Si el resultado es FAIL,
  o si `κ=0` y `κ>0` NO difieren claramente, se reporta como hallazgo — sería en sí
  mismo una alerta seria: significaría que el arnés experimental de todo el Enfoque 3
  no puede distinguir enfriamiento geométrico de un baño térmico, y habría que
  revisarlo (se PARA y se reporta a CS, no se re-interpreta a mano).
- No topología, no commits (regla del director para este batch).
