# PROTOCOLO E5.4-4 — PRE-REGISTRO
## "Exergía por escalas espectrales: ¿qué longitudes de onda la retienen?"

**Fecha de escritura:** 2026-07-25, 00:42 UTC (ANTES de correr
`E5_4_4_exergia_espectral_motor.py`; este archivo se congela con este mtime —
no se edita tras ver resultados, T3).

**Ejecutor:** CC, experimento E5.4-4 de
`BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md` (Tema 4, experimento 4).
Corre en paralelo con otros 29 experimentos, cada uno con su propio prefijo. Este
protocolo cubre EXCLUSIVAMENTE el prefijo `E5_4_4_`. No se toca ningún archivo fuera de
`codigo/BATERIA_ENFOQUE5/E5_4_4_exergia_espectral/` y
`results/BATERIA_ENFOQUE5/E5_4_4_exergia_espectral/`.

---

## 1. Contexto y motivo

E5.4-4 pregunta: *"¿la exergía se congela antes en estructuras grandes o chicas?"*
Barrido pedido: `a × banda espectral completa × ≥12 semillas`. Observable: *exergía por
escala vs a*. NULL: *densidad fija*. PASS: *espectro de retención reportado; se compara
con "escalas grandes primero" sin imponerlo.*

El propio documento de batería indica reutilizar la metodología del experimento espectral
previo casi idéntico: `Cosmogenesis-Web/codigo/BATERIA_FUNDAMENTOS/F3_5_espectral/`
(FFT por bandas log del campo T, detector de punto-de-no-retorno/congelamiento por banda,
correlación de Spearman entre centro de banda y `a` de congelamiento). F3-5 tuvo éxito
claro (orden de congelamiento perfecto, escalas grandes primero) usando como observable la
**potencia espectral cruda** `P_banda(a) = Σ_{n∈banda} |FFT(T)|²`.

E5.4-4 pide explícitamente **exergía**, no potencia cruda. Este protocolo define una
cantidad de exergía por banda, derivada de la misma FFT pero con una interpretación
termodinámica explícita (capacidad de hacer trabajo, no solo "energía en el modo"), y la
verifica por un SEGUNDO método independiente (reconstrucción en espacio real) antes de
usarla como observable de congelamiento.

**Predicción externa a comparar, NO a imponer** (tomada de F3-5, coherente con el resto de
la batería — Tema 4 pide compararla, no forzarla):
> Las escalas GRANDES (λ grande, k comóvil pequeño) retienen/congelan su exergía PRIMERO
> —a un `a` más chico—; las escalas CHICAS (λ pequeña, k grande) siguen liberando/perdiendo
> exergía hasta un `a` mayor.
>
> Si el orden observado es el inverso, o no hay orden, es un **dato en contra** y se
> reporta como tal — no se reinterpreta a posteriori (T3).

---

## 2. Código base (NO se edita)

`Cosmogenesis-Web/codigo/CF2_estiramiento/CF2_estiramiento_motor.py` — leído completo.
El motor de E5.4-4 **importa directamente por ruta de archivo** (no copia, no edita) el
sello físico original, igual que hizo F3-5:

- Constantes: `L, H_EXP, RHO0, D0, W0, DT, N_SUB, ORIGINAL_STEPS_PER_TG`.
- Funciones: `initial_T(L, w0)`, `diffuse(T, D, dt, n_sub)`.

Esto garantiza que la dinámica del campo `T` es EXACTAMENTE la misma que CF-2/F3-1/F3-5 —
la lente de exergía mira la MISMA física, no una réplica que pudo haber divergido.

No se toca `CF2_estiramiento_motor.py`, `F3_5_espectral_motor.py`, ni ningún otro archivo
existente. El agrupamiento en 6 bandas espectrales logarítmicas (B0..B5) se reutiliza
**idéntico** al de F3-5 (mismo `BAND_DEFS`, definido aquí de nuevo por valor — no por
import cruzado a un archivo de otro prefijo en ejecución paralela — para no crear una
dependencia entre carpetas de agentes distintos corriendo a la vez).

---

## 3. Observable exacto: EXERGÍA por banda (no potencia cruda)

### 3.1 Definición termodinámica de la exergía por banda

Para un campo escalar `T(x,y)` que difunde bajo una dinámica que conserva la suma total
(la difusión con condiciones periódicas solo redistribuye, no crea ni destruye `ΣT`), el
estado de equilibrio (muerte térmica local) es el campo espacialmente uniforme igual a la
media `T̄`. La exergía de una distribución de temperatura no uniforme, a orden cuadrático
en la desviación del equilibrio (aproximación estándar de disponibilidad termodinámica
para fluctuaciones pequeñas alrededor de un estado de referencia, coherente con **E5.6-2**
de esta batería: `X ≈ E − T·S_ent`, cuyo término dominante en `T` casi uniforme es
cuadrático en `(T−T̄)`), es:

```
X_total(a) = (1 / (2·T_ref)) · Σ_{i,j} (T_ij(a) − T̄_ref)²
```

`T_ref` = media espacial del campo en el checkpoint inicial `a=1` de esa corrida (semilla,
modo), **medida, no impuesta a mano** (T1). Se usa fija (estado muerto de referencia fijo,
la convención estándar de exergía) en vez de la media instantánea `T̄(a)` porque la
difusión periódica conserva `ΣT` (se verifica empíricamente en la sección 3.4 como
diagnóstico, con el único quiebre posible siendo el `clip(0,1)` del motor original — se
mide y reporta, no se oculta).

Por el teorema de Parseval, la parte de esa suma que corresponde a los modos de Fourier
`n=1..L/2` (excluyendo el modo `n=0`, que ES la media `T̄` — no aporta exergía, es el
propio estado de referencia) es exactamente proporcional a la potencia espectral por
banda ya usada en F3-5. Esto hace la banda de exergía:

```
X_banda(a) = P_banda(a) / (2·T_ref)
```

con `P_banda(a) = Σ_{n∈banda} P(n,a)`, `P(n,a) = ⟨|rFFT(T fila)|²⟩_filas` — misma
construcción exacta que F3-5 (`band_power`), reutilizada aquí y reinterpretada
explícitamente como exergía dividiendo por `2·T_ref` (constante por corrida, medida).

**Nota T1 honesta:** dividir por una constante fija por corrida no cambia ninguna curva de
retención relativa `R_X_banda(a) = X_banda(a)/X_banda(a=1)` frente a la de potencia cruda
de F3-5 — son la misma curva (la constante se cancela). Esto es esperado y se declara
explícitamente ANTES de correr: la novedad científica de E5.4-4 sobre F3-5 no está en la
curva de retención por sí sola, sino en (a) los valores absolutos de exergía con
interpretación termodinámica y su suma total como presupuesto, (b) la fracción de exergía
total que sostiene cada banda a lo largo de `a` (`frac_banda(a)`, no reportada en F3-5),
(c) la verificación por un segundo método independiente (§3.2) y (d) el NULL y el marco de
comparación puestos explícitamente en términos de exergía, según pide E5.4-4.

### 3.2 Segundo método (independiente, T2: verificación cruzada)

Para no depender de una sola vía de cómputo, se reconstruye cada banda en **espacio real**
por filtrado pasa-banda (FFT inversa manteniendo solo los índices `n` de esa banda) y se
calcula la varianza cuadrática directamente sobre el campo filtrado:

```
T_banda_filtrado(a) = irFFT( rFFT(T(a)) · mascara_banda )   [por fila]
X_banda_realspace(a) = (1/(2·T_ref)) · Σ_{i,j} T_banda_filtrado(a)_ij²  · (factor Parseval)
```

Debe coincidir con `X_banda(a)` de §3.1 hasta precisión numérica (identidad de Parseval).
Se reporta la discrepancia relativa máxima observada en toda la corrida como diagnóstico
de correctitud del código (no es un hallazgo científico, es una prueba de que el cómputo
en frecuencia no tiene un error de implementación).

### 3.3 Fracción de exergía total por banda (segundo observable científico)

```
frac_banda(a) = X_banda(a) / X_total(a)     donde X_total(a) = Σ_bandas X_banda(a)
```

Muestra cómo se REPARTE la exergía disponible entre escalas a medida que `a` crece —
complementario a la retención relativa (que solo mira cada banda contra sí misma en
`a=1`). Ambas curvas se entregan completas (T5: curva entera, no gate binario).

### 3.4 Diagnóstico de conservación (axioma E1, honestidad de la referencia fija)

Se mide y reporta, por corrida, la deriva `|T̄(a) − T̄_ref| / T̄_ref` a lo largo del
barrido de `a` (debería ser ≈0 si la difusión periódica conserva la media; el único
mecanismo de ruptura es el `clip(T,0,1)` heredado de CF2, que no se retoca). Esto NO es un
criterio de PASS/FAIL de este experimento — es documentación honesta de cuán válida es la
convención de referencia fija usada en §3.1.

### 3.5 Detección de congelamiento (idéntica en forma a F3-5, aplicada a `R_X_banda`)

Igual algoritmo que F3-5 (pendiente log-log local, tolerancia `FREEZE_SLOPE_TOL`,
clasificación `frozen_preserved` vs `frozen_depleted` vía `R_FLOOR`), aplicado a
`R_X_banda(a) = X_banda(a)/X_banda(a=1)` (idéntica en valor a la retención de potencia de
F3-5 por la cancelación de constante ya declarada en §3.1 — no se pretende que sea un
resultado nuevo, se declara igual desde antes de correr, T3).

- `FREEZE_SLOPE_TOL = 0.02` (mismo valor que F3-5, no elegido ad hoc para este experimento
  — T1).
- `R_FLOOR = 0.05` (ídem).
- `freeze_a(banda)` = primer `a` desde el cual la pendiente log-log de `R_X_banda` se
  mantiene `< FREEZE_SLOPE_TOL` en valor absoluto hasta el final del barrido. `NaN` si
  nunca se cumple (censura, se reporta, no se fuerza).

---

## 4. Barrido pre-registrado

| Parámetro | Rango | Puntos | Espaciado |
|---|---|---|---|
| `a` (factor de expansión) | [1, 1e4] | 30 | log (geomspace) |
| banda espectral | B0..B5 (idéntica tabla que F3-5, L=64, Nyquist=32) | 6 | todo el espectro accesible |
| semillas | ver lista abajo | 16 | ≥12 exigidas por spec |
| amplitud de ruido dinámico `σ_din` (perturbación dinámica, T7) | {0.0, 1e-3, 1e-2} | 3 | veredicto principal en `σ_din=0`, robustez reportada en los otros dos |
| modo | {REAL, NULL_RHO_FIXED} | 2 | NULL = densidad fija (ρ≡ρ0, D≡D0), tal como pide la spec de E5.4-4 |

Semillas (10 estándar del proyecto + 6 adicionales, mismo criterio que F3-5):
`[7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 13, 271828, 161803, 31415, 90210, 20260724]`

Ruido dinámico: `N(0, σ_din·sqrt(dt_sub·N_SUB))` inyectado tras cada paso de difusión,
mismo mecanismo Wiener que F3-5/F3-1, idéntico en REAL y NULL.

Total de trayectorias físicas: 3 (σ_din) × 16 (semillas) × 2 (modo) = 96, cada una
evaluada en 30 checkpoints de `a` × 6 bandas.

Bandas espectrales (idéntico a F3-5, L=64, Nyquist=32):

| Banda | n (armónicos) | rótulo |
|---|---|---|
| B0 | 1 | escala más grande (fundamental) |
| B1 | 2 | |
| B2 | 3–4 | |
| B3 | 5–8 | |
| B4 | 9–16 | |
| B5 | 17–32 | escala más chica (Nyquist) |

---

## 5. NULL

`NULL_RHO_FIXED`: densidad fija `ρ≡ρ0`, `D≡D0` constante (tal como especifica la spec de
E5.4-4: "NULL=densidad fija"), aunque `a` se sigue definiendo por el mismo reloj genético.
Sin el freno `D∝a⁻³`, la exposición difusiva acumulada crece sin límite mientras dure la
simulación — la exergía por banda debería seguir cayendo (no congelarse en un valor
preservado) mientras el barrido continúe. Si el NULL muestra el MISMO orden de
congelamiento que el REAL, el hallazgo se reporta como artefacto del método, no del
mecanismo físico (T4).

---

## 6. Criterio de PASS (congelado, no se toca tras ver resultados)

Este experimento tiene, por spec, un PASS de entrega ("espectro de retención de exergía
reportado; se compara con 'escalas grandes primero' SIN imponerlo") más débil que un gate
binario estricto — coherente con T5. Aun así, se aplica un criterio cuantitativo
pre-registrado para la comparación, idéntico en forma al de F3-5 (mismo umbral, no
re-elegido a mano — T1):

Por cada combinación `(σ_din, semilla)`, en REAL y NULL por separado:
- `freeze_a(banda)` para las 6 bandas (puede ser `NaN`).
- Sobre bandas no censuradas (≥3 requeridas), `ρ_orden` = Spearman entre centro de banda
  (proxy de `k`, ascendente B0→B5) y `freeze_a(banda)`.
- `orden_REAL` = `ρ_orden(REAL) ≥ RHO_MIN` con `RHO_MIN = 0.6`.
- `orden_NULL` = `ρ_orden(NULL) ≥ RHO_MIN`.
- `combo_pass` = `orden_REAL AND (NOT orden_NULL)`.

`PASS_RATE_MIN = 0.55` (mismo valor que CF-2/F3-1/F3-5, T1). `rate` se calcula sobre
combos no indeterminados, para `σ_din=0` (variante principal) y cada `σ_din` (robustez).

> **PASS_E5.4-4 (comparación con "grandes primero")** si `rate(σ_din=0) ≥ 0.55` Y mediana
> de `ρ_orden(REAL)` en las 16 semillas de `σ_din=0` es positiva y ≥ `RHO_MIN` Y el NULL
> NO alcanza `RHO_MIN` en la mayoría de semillas (tasa `orden_NULL` < 0.45).
>
> Si la mediana de `ρ_orden(REAL)` es **negativa**: `FAIL_INVERSO — dato en contra` (T3,
> no se reinterpreta).
>
> Si el NULL reproduce el mismo orden: `FAIL_NULL_NO_MUERDE` (T4), sin importar cuán
> limpio sea el orden REAL.
>
> El **espectro de retención completo** (`R_X_banda(a)`, `frac_banda(a)`, `freeze_a` por
> banda, para las 96 trayectorias) se entrega SIEMPRE, independientemente del veredicto
> anterior — es el entregable mínimo pedido explícitamente por la spec de E5.4-4.

---

## 7. Verificación cruzada (tres vías, obligatorias)

(a) **NULL muerde**: se reporta `orden_NULL` junto a `orden_REAL` por cada `σ_din`.
(b) **Segundo método independiente**: reconstrucción en espacio real (§3.2), discrepancia
    relativa máxima frente al cómputo en frecuencia, reportada explícitamente.
(c) **Auditoría en disco**: código (`E5_4_4_exergia_espectral_motor.py`) + JSON crudo
    (`E5_4_4_exergia_espectral_{smoke,produccion}_result.json`) quedan en disco para
    revisión por quien no escribió este código.

---

## 8. Orden de ejecución

1. **Smoke** (`python E5_4_4_exergia_espectral_motor.py smoke`): 2 semillas (7, 42), 8
   puntos de `a` en [1,100], `σ_din=0` únicamente — verifica que la FFT, la exergía por
   banda, el cross-check en espacio real y la detección de `freeze_a` corren sin error y
   dan números con sentido físico ANTES de la producción completa.
2. **Producción completa** (`python E5_4_4_exergia_espectral_motor.py produccion`):
   barrido íntegro de la sección 4.
3. Reporte crudo a CS — sin adjudicar el hallazgo más allá de este experimento puntual.

---

## 9. Qué NO se hace aquí

- No se toca `CF2_estiramiento_motor.py`, `F3_5_espectral_motor.py`, ni ningún otro
  archivo existente fuera de este prefijo.
- No se auto-adjudica el veredicto final de la batería — eso es de CS/Alexis.
- No se cambia este criterio después de correr el motor (T3). Si el resultado es FAIL,
  FAIL_INVERSO o FAIL_NULL_NO_MUERDE, se reporta tal cual, sin suavizar.
- No topología, no commits (regla del director para este batch).
- Si se detecta un error en el código base (`CF2_estiramiento_motor.py`) durante la
  lectura o el uso, se PARA y se reporta a CS con la línea exacta — no se "arregla" a
  criterio propio.
