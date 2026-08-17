# RE-ANÁLISIS — C-N2.5.5 (las continuas detrás del booleano) y C-N2.5.6 (¿reversible por construcción?)

**Fecha:** 13-ago-2026 · **Encargo:** Alexis López Tapia · **Tipo:** re-análisis de datos existentes
(+ una re-corrida determinista autorizada, sólo para recuperar el espectro que `_juzga()` descartaba).

**Este documento NO cierra nada.** Reporta números, identidades de código y contrafácticos. La decisión de
qué hacer con ellos es del director (`nota-permanente-no-cerrar-experimentos.md`).

**Archivos producidos**
- `REANALISIS_CS070_continuas.csv` — 96 filas, una por corrida, con las continuas + el espectro completo.
- `REANALISIS_CS070_continuas.png` — 4 paneles (histogramas, n_ejes, plano de las dos condiciones, contrafáctico).

---

# TAREA 1 — C-N2.5.5: qué había debajo de "dirección real = 0,000"

## 1.1 Los datos crudos SÍ estaban guardados

`cs070_tanda_resultados.json` (96 registros) ya traía `pico_medio`, `n_ejes`, `PR`, `gap_interno`,
`frac_picados`, `certificado` y `peso_semilla` por corrida. Lo único perdido era el **espectro** `ev` del
tensor de orientación (`_juzga()` lo consume y lo tira). Se recuperó re-corriendo la tanda con las mismas
semillas, interceptando `SM.tensor_orientacion` sin tocar ningún archivo del proyecto.

> **Verificación de determinismo (guarda de la casa):** la re-corrida reproduce la tanda original en
> **0/96 discrepancias** (`n_ejes` idéntico, `pico_medio` idéntico a <1e-12). No hay azar no controlado.

## 1.2 Las dos condiciones, medidas por separado

`direccion_real(r) = bool(r["certificado"] and r["n_ejes"] > 1)`

| condición | corridas | fracción |
|---|---|---|
| `certificado` (pico_medio > 0.85) | **18/96** | 0.188 |
| `n_ejes > 1` | **4/96** | 0.042 |
| **ambas a la vez** (`direccion_real`) | **0/96** | 0.000 |

**Distribución de `pico_medio`** (96 corridas): mín **0,4497** · mediana **0,7681** · máx **0,9775**.
Es una variable continua bien extendida que cruza el umbral 0,85 en 18 corridas — no una cantidad pegada al piso.

**Distribución de `n_ejes`**: `{0: 87, 1: 5, 5: 1, 6: 2, 7: 1}`.

**Medianas de `pico_medio` por brazo × N** (rango entre corchetes):

| brazo | N=900 | N=1500 | N=2500 | certificado |
|---|---|---|---|---|
| semilla_coherente | 0,714 [0,667–0,880] | 0,774 [0,722–**0,977**] | 0,751 [0,654–0,811] | 4/24 |
| semilla_barajada | 0,744 [0,450–0,763] | 0,731 [0,669–0,859] | 0,736 [0,574–0,821] | 1/24 |
| sin_semilla | 0,830 [0,765–0,876] | **0,850** [0,727–0,884] | 0,749 [0,675–0,909] | **9/24** |
| semilla_sustrato_local | 0,819 [0,765–0,955] | 0,810 [0,667–0,910] | 0,769 [0,726–0,846] | 4/24 |

En la variable continua, el brazo que más "certifica" sigue siendo **sin_semilla** (9/24), como ya notaba el
informe original. La semilla no sube `pico_medio`; el orden es sin_semilla ≈ sustrato_local > coherente > barajada.

## 1.3 La pregunta que importa: ¿ninguna se acercó, o una sí y la otra nunca?

**Ninguna de las dos lecturas propuestas es la correcta. La respuesta es una tercera:**

1. **Las dos condiciones se cumplen por separado, con frecuencia distinta** (18/96 y 4/96), y
2. **están ANTI-correlacionadas**: Spearman(`pico_medio`, `n_pobl`) = **−0,369, p = 2,1×10⁻⁴** sobre las 96
   corridas (`n_pobl` = nº de ejes de vacío que retienen ≥2% de los nodos). Cuanto más se concentra el
   consenso (sube la confianza por nodo), menos ejes sobreviven. Dentro de las 9 corridas donde el contador
   de ejes está siquiera activo, la separación es total y sin solape:

   | | `pico_medio` |
   |---|---|
   | corridas con `n_ejes > 1` (n=4) | 0,667 · 0,699 · 0,765 · 0,833 — **ninguna cruza 0,85** |
   | corridas con `n_ejes = 1` (n=5) | 0,682 · 0,732 · **0,922** · **0,955** · **0,978** |

3. **El AND nunca tuvo poder estadístico.** Con esas marginales, el número esperado de corridas con ambas
   condiciones bajo **independencia** es 96 × 0,188 × 0,042 = **0,75 corridas**. Poisson(λ=0,75) da
   **P(cero éxitos) = 0,47**. Fisher exacto sobre la tabla 2×2: **p = 1,0**.

> **Conclusión de esta parte:** "0,000 en las 96 corridas, sin excepción" suena a 96 oportunidades falladas.
> No lo es. Con las marginales realmente observadas se esperaba **menos de una** coincidencia, y ver cero es
> el resultado **más probable** aun si las dos condiciones fueran independientes. El número no discrimina
> entre "el nodo falla" y "el diseño no podía verlo". **No es evidencia sobre C-N2.5.5.**

## 1.4 `cuenta_ejes_gap`: la guarda trasplantada que apaga el contador en 87/96 corridas

El código (`cs067_habitacion_completa.py:130-156`):

```python
def cuenta_ejes_gap(ev, c=1.6, g=3.0, piso=0.02):
    ...
    pobl = p[p >= piso]; n_pobl = len(pobl)
    if n_pobl == D:
        return 0, PR, 0.0, r_thr        # rango PLENO = isotropía, sin estructura (esfera 8D -> 0)
    if n_pobl <= 1:
        return n_pobl, PR, 0.0, r_thr
    ...
```

Su docstring declara la premisa explícitamente:

> *"con K_sorteado < D=8 hay dimensiones del embedding en CERO EXACTO, y el mayor salto λ_r/λ_{r+1} cae
> trivialmente en el BORDE de rango"*

**Esa premisa se cumple en CS067 y NO se cumple en CS070.**

| | CS067 (donde se escribió la guarda) | CS070 (donde se reusó) |
|---|---|---|
| construcción de V | `C9._spins(N, DMAX_INT, rng)` → **D = 8** (`cs067_habitacion_completa.py:262`) | `C9._spins(N, K, rng)` → **D = K** (`cs070_semilla.py:116`) |
| `K_sorteado` | 4–8 | 4–8 |
| dimensiones no usadas del embedding | 8 − K (en cero exacto) | **ninguna** |
| qué significa `n_pobl == D` | los 8 ejes del embedding poblados = isotropía 8D genuina | **los K vacíos vivos con ≥2% de nodos cada uno** |

En CS070 `V` post-Potts es one-hot en la base canónica de dimensión K, así que el tensor de orientación es
**diagonal K×K con las fracciones de nodos por vacío**. "Rango pleno" ya no es isotropía en un embedding
sobredimensionado: **es el resultado MÁS multidireccional posible** — todos los vacíos sobreviven — y la
guarda lo devuelve como `n_ejes = 0`.

**Verificado en los datos:**
- `n_pobl` (vacíos con ≥2% de los nodos): mediana **6**, rango 2–8.
- **87/96 corridas están en "rango pleno"** → `n_ejes = 0` por esa rama, sin llegar nunca al cálculo del gap.
- **Las 18 corridas certificadas tienen `n_pobl ≥ 2`; 17 de 18 tienen `n_pobl ≥ 4`.** O sea: la certificación
  NO viene de colapso a un eje. Es exactamente lo que el comentario de CC en `cs070_smoke.py:46-50` describía
  ("pico_medio>0,85 pero n_ejes=0") e interpretó como isotropía — pero el espectro dice que son 4–8 vacíos
  poblados con alta confianza local, no una esfera isótropa.
- Sólo **11/96** corridas tuvieron `K = 8`, el único caso en que el trasplante es inocuo.

### ¿Era `n_ejes > 1` inalcanzable por diseño?

**No literalmente: 4/96 corridas lo alcanzaron.** No puedo afirmar "el booleano nunca podía dar True".
Pero sí, con el código en la mano:

- para que el contador se active hace falta que **al menos un vacío quede casi extinto** (<2% de los nodos),
  condición que se dio en 9/96 corridas;
- en las otras 87 el contador está **desactivado por una guarda cuya premisa declarada no se cumple** en este reuso;
- y las dos condiciones del AND están anti-correlacionadas (§1.3), así que la ventana conjunta es todavía
  más angosta que el producto de las marginales.

**Lectura honesta: no es "inalcanzable por diseño", es "prácticamente inalcanzable con 96 corridas".**
Que es, a efectos de leer el 0,000 como refutación, el mismo problema.

## 1.5 Contrafáctico: el mismo juez, en el embedding para el que fue escrito

Recomputando `cuenta_ejes_gap` sobre **el mismo espectro**, rellenado con ceros hasta D=8 (es decir,
reproduciendo el embedding de CS067 sin cambiar ni un dato de la simulación):

| | juez tal cual (D=K) | contrafáctico (D=8) |
|---|---|---|
| `n_ejes > 1` | 4/96 | **63/96** |
| `direccion_real` | 0/96 | **6/96** |

Por brazo, `direccion_real` en el contrafáctico: **sin_semilla 5/24 · semilla_barajada 1/24 ·
semilla_coherente 0/24 · semilla_sustrato_local 0/24.**

> **Esto no rescata C-N2.5.5.** La cuerda decisiva del diseño era COHERENTE > BARAJADA; en el contrafáctico
> la coherente sigue en cero y el que enciende es el brazo **sin semilla**. La semilla sigue sin amplificar nada.
> Lo que el contrafáctico sí muestra es que **el 0,000 medía el embedding, no el nodo**: bastó devolver el
> juez a su premisa para que 63/96 corridas pasaran de "cero ejes" a "múltiples ejes".

## 1.6 Una segunda guarda de código, ya documentada por CC, que conviene tener a la vista

`_juzga()` (`cs070_semilla.py:123-128`) define `pico_medio = np.mean(conf)` — la confianza del voto Potts —
**no** `picado_por_nodo(V)`, que es la GUARDA 1 del arco. La razón está escrita en
`cs067_habitacion_completa.py:337-339`: post-Potts `V` es one-hot por construcción, así que
`picado_por_nodo(V) ≡ 1,0` y certificaría el 100% de las corridas trivialmente. La sustitución es correcta y
fue hecha a sabiendas — pero implica que **el "certificado" de CS070 no es el mismo certificado que el resto
del arco**, y no debería compararse con él como si lo fuera.

---

# TAREA 2 — C-N2.5.6: `cg001_field.py` es reversible por construcción

## 2.1 La dinámica, escrita como operador

De `_paso()` (`cg001_field.py:67-99`), con `G` = `gaussian_filter(·, sigma, mode="wrap")`:

```
a    = (I − G) φ
m'   = decay·m + |a|                        # decay = 0.97
Λ    = diag( lam / (1 + gamma·m') )         # lam = 0.50  ← usa el m YA actualizado
φ'   = φ − Λ·(I − G) φ = [I − Λ(I−G)] φ
```

## 2.2 El inverso, derivado a mano

El detalle decisivo es de orden de líneas: **`m` se actualiza ANTES de calcular `lam_eff`** (líneas 77 y 80).
Por lo tanto `Λ` depende sólo de `m'`, que **forma parte del estado nuevo**. Dado `(φ', m')`:

1. `Λ = lam / (1 + gamma·m')` — se conoce exactamente.
2. `φ = [I − Λ(I−G)]⁻¹ φ'` — por punto fijo `φ_{k+1} = φ' + Λ·((I−G)φ_k)`.
3. `a = (I−G)φ` ; `m = (m' − |a|) / decay`.

**La inversa existe siempre y está bien condicionada:** `G` es un suavizador circulante simétrico PSD con
autovalores en (0,1] ⇒ `I−G` tiene autovalores en [0,1) ⇒ `‖Λ(I−G)‖ ≤ lam = 0,5 < 1`. Serie de Neumann
convergente; número de condición ≤ (1+0,5)/(1−0,5) = **3**.

## 2.3 Pasos que destruyan información: NO hay ninguno en la dinámica

Auditados uno por uno (grep de `np.maximum` / `np.minimum` / `np.clip` / `np.round` / `np.where` / `floor` /
`ceil` sobre `cg001_field.py` — **no aparece ninguno dentro de `_paso`**):

| operación | ¿destruye información? | por qué |
|---|---|---|
| `abs_a = np.abs(a)` (l.71) | **No** | se aplica a una cantidad **derivada**; `a = (I−G)φ` se recupera de φ, que es estado. El signo nunca se pierde del estado. |
| `m = decay·m + abs_a` (l.77) | **No** | contracción invertible (÷0,97). "Olvido" en sentido de amortiguación, no de pérdida de biyectividad. |
| `lam_eff = lam/(1 + gamma·m)` (l.80) | **No** | `m ≥ 0` siempre ⇒ el denominador nunca es ≤0. No hay división por cero. |
| `thr = np.quantile(m, 0.999)` ; `mask_nicho = m > thr` (l.84-85) | **No** | **umbral duro, pero SÓLO medición**: alimenta `convertido` y `n_nicho`. Nunca toca `φ` ni `m`. |
| `disipado`, `exergia`, `entropia_acum` | **No** | acumuladores de medición, no realimentan. |
| `hist` con `log_every=10` (l.129) | **No** (para el estado) | submuestreo del **log**, no de la dinámica. |
| `gaussian_filter(mode="wrap")` | **No** | convolución circular lineal; el núcleo gaussiano tiene coeficientes de Fourier estrictamente positivos. (Ni hace falta invertirlo: el inverso lo esquiva.) |

**No hay máximo, ni umbral realimentado, ni redondeo, ni saturación en la actualización del estado.**

## 2.4 Verificación numérica: evolucionar k pasos e invertir

Script: `cs076_test_reversibilidad.py` (scratchpad). Se evoluciona desde `_inicializar_phi`, se aplica el
inverso k veces y se compara con el estado inicial.

| k pasos (ida y vuelta) | ‖φ_recuperado − φ₀‖∞ | error relativo | factor por paso |
|---|---|---|---|
| **1** | **5,19×10⁻¹⁶** | 1,06×10⁻¹⁶ | — |
| 2 | 1,54×10⁻¹⁵ | 3,16×10⁻¹⁶ | 2,97 |
| 8 | 1,10×10⁻¹³ | 2,26×10⁻¹⁴ | 2,03 |
| 16 | 1,40×10⁻¹¹ | 2,86×10⁻¹² | 1,83 |
| 24 | 1,27×10⁻⁰⁹ | 2,61×10⁻¹⁰ | 1,73 |
| 32 | 2,50×10⁻⁰⁷ | 5,13×10⁻⁰⁸ | 1,84 |
| 40 | 2,32×10⁻⁰⁵ | 4,76×10⁻⁰⁶ | 1,76 |
| 60 | desborda | — | — |

(ε de máquina float64 = 2,22×10⁻¹⁶. También verificado en L=48/50 pasos y en el régimen anti-Shannon
`gamma=0` de CS076: mismo comportamiento.)

**Lectura:** en **un paso** el estado vuelve exacto a **2–6 ε de máquina**. El crecimiento posterior es
geométrico con factor **1,7–2,0 por paso**, que es exactamente la cota de condicionamiento del inverso
`1/(1−lam) = 2`. **No es pérdida de información: es redondeo de float64 amplificado por la contracción.**

## 2.5 Matiz importante: biyectiva PERO disipativa

`log|det[I − Λ(I−G)]| ≈ −173` sobre 4096 celdas ≈ **−0,042 por celda** (estimador de Hutchinson, tras 10 pasos).
El determinante es **negativo en logaritmo pero distinto de cero**: el flujo **contrae volumen de fase** y es
biyectivo a la vez. Es un flujo de relajación con atractor (φ → constante, m → 0).

Esto separa dos cosas que conviene no confundir:

- **Flecha macroscópica: SÍ existe y es medible.** La exergía Σ|a| decae monótonamente y el volumen de fase
  se contrae. Es exactamente lo que C-N3/CS009 ya había confirmado ("el desorden agregado sólo sube").
- **Flecha microscópica estocástica: NO está definida en este sustrato.** Fijada la semilla, la trayectoria
  es **una sola órbita determinista**. Σ_T = log(P[Γ]/P[Γ†]) compara dos medidas de probabilidad sobre
  trayectorias; acá ambas son deltas sobre una órbita. No hay kernel de transición que violar.

El propio informe de CS076 lo declaraba ("*es determinista y no expone un kernel de transición estocástico
explícito*") y aun así reportó el resultado como si midiera el mundo.

## 2.6 Hallazgo adicional: el NULL decisivo de CS076 también está casi degenerado

CS076 ya cazó que el NULL de orden barajado es **una identidad** para la skewness (momento marginal,
invariante al orden) y avisó que no se reusara. Pero el estadístico que se declaró **decisivo** — el KL de
balance detallado contra ese mismo NULL — sufre una versión más sutil del mismo problema.

`null_orden_barajado` (`cs076_direccion_temporal.py:192-200`) reconstruye
`x_null = x[0] + cumsum(permutación de los incrementos)`. Eso **conserva `x[0]` y `x[-1]` EXACTAMENTE**
(la suma de una permutación es la misma suma). Y como el campo relaja, **87–89% de los incrementos comparten
signo**, así que la serie barajada también es casi monótona sobre el mismo recorrido ⇒ el histograma 2D de
`(x_t, x_{t+1})` queda casi idéntico ⇒ el KL contra su transpuesta queda casi idéntico.

Medido (200 permutaciones, L=32, 10 celdas — **parámetros propios, no los de CS076**, así que esto reproduce
la estructura del problema, no su z exacto):

| régimen | KL real | KL NULL (media ± sd) | el NULL reproduce | z |
|---|---|---|---|---|
| gamma = 8,0 | 0,004212 | 0,004198 ± 3,1×10⁻⁵ | **99,65%** del KL real | +0,47 |
| gamma = 0 | 0,009121 | 0,008907 ± 9,7×10⁻⁴ | **97,65%** del KL real | +0,22 |

En gamma=8 el **máximo** de las 200 permutaciones (0,004212) coincide con el valor real: el NULL está pegado
al techo. Con ~0,35% de margen disponible, un z≈0 estaba prácticamente forzado.

## 2.7 Veredicto de la Tarea 2 (no-cierre)

`cg001_field.py` **es reversible por construcción**: biyección analítica del estado completo (φ, m),
verificada a 5×10⁻¹⁶ en un paso. **La única irreversibilidad disponible es el redondeo de la máquina**,
amplificado ~1,8× por paso por el condicionamiento del inverso. Medir "flecha temporal microscópica" ahí
mide el error de punto flotante amplificado.

**C-N2.5.6 debe pasar de "refutado" a "NO TESTEABLE EN ESTE SUSTRATO"** — el mismo movimiento que se hizo
con κ_V. El resultado nulo de CS076 es un resultado **sobre el instrumento**, no sobre el mundo, y por partida
doble: (a) el sustrato no tiene kernel estocástico que pueda violar balance detallado, y (b) el único control
que aislaba el orden temporal reproduce el 97–99% del estadístico real.

**Lo que sí queda en pie y no depende de nada de esto:** la flecha **macroscópica** (exergía monótona
decreciente, contracción de volumen de fase) es real, está en el código, y ya la había confirmado C-N3/CS009.

---

# GUARDAS APLICADAS

**1. ¿El número podía salir distinto?**
- **CS070 / `direccion_real = 0,000`:** técnicamente sí (4/96 corridas alcanzaron `n_ejes>1`), pero con las
  marginales observadas el conjunto esperado bajo independencia es **0,75 corridas** y P(cero) = 0,47.
  **Con 96 corridas el diseño nunca tuvo poder para ver el AND.** No es una medición del nodo.
- **CS076 / `z = −0,02`:** el NULL decisivo reproduce el 99,65% del estadístico real. El z estaba acotado a
  ser ≈0 por construcción del control.

**2. Identidades algebraicas y guardas de código encontradas (tres):**
- `cuenta_ejes_gap`: la rama `n_pobl == D → 0` fue escrita para D=8 con K<8 (CS067) y se reusó donde **D=K**
  (CS070), invirtiendo su semántica. Apaga el contador en **87/96** corridas.
- `picado_por_nodo(V) ≡ 1,0` post-Potts por construcción (one-hot) — obligó a sustituir la GUARDA 1 del arco
  por `mean(conf)`. Ya documentado por CC; se re-anota porque cambia qué significa "certificado" en CS070.
- `null_orden_barajado` conserva `x[0]` y `x[-1]` exactamente y la envolvente monótona — el NULL decisivo de
  CS076 está casi degenerado también para el KL, no sólo para la skewness (que CS ya había cazado).

**3. No se inventó nada.** Todas las continuas salen del JSON crudo guardado en julio; el espectro salió de
una re-corrida **bit-a-bit idéntica** (0/96 discrepancias). El contrafáctico D=8 es una recomputación del
mismo juez sobre el mismo espectro, sin re-simular. El z=+0,47 de §2.6 usa parámetros propios (L=32,
1 semilla, 10 celdas) y **no** pretende reproducir el z=−0,02 de CS076: lo que se afirma es la degeneración
estructural del control, no su valor numérico exacto.

---

# QUÉ QUEDA ABIERTO (si el director lo autoriza)

- **C-N2.5.5:** re-correr CS070 con el embedding D=8 (o con un contador de ejes cuya guarda de rango se
  reformule para D=K) y **más semillas**, para que el AND tenga poder. Nota: el contrafáctico ya sugiere que
  el veredicto sobre la SEMILLA no cambiaría (enciende sin_semilla, no coherente).
- **C-N2.5.6:** si se quiere medir flecha microscópica, hace falta un sustrato con **kernel estocástico
  explícito** (ruido por paso), no `cg001_field.py`. Alternativamente, medir en `cg001_field.py` lo que sí es
  medible: la **tasa de contracción de volumen de fase** (−0,042/celda/paso), que es un observable de flecha
  legítimo para un flujo determinista disipativo.

— re-análisis CS · 13-ago-2026
