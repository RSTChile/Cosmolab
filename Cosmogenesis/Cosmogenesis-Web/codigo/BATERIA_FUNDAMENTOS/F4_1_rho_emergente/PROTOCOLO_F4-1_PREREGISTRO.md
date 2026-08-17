# PROTOCOLO F4-1 — PRE-REGISTRO
## "Densidad emergente: ¿ρ cae sola al expandir, sin imponerlo?"

**Fecha de escritura:** 2026-07-24 (ANTES de correr `F4_1_rho_emergente_motor.py`; este
archivo se congela con este mtime — no se edita tras ver resultados, T3).

**Ejecutor:** CC, experimento F4-1 de la batería `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`
(Enfoque 4, experimento 1). Corre en paralelo con otros 23 experimentos, cada uno con su
propio prefijo. Este protocolo cubre EXCLUSIVAMENTE el prefijo `F4_1_`. No se toca ningún
archivo fuera de `codigo/BATERIA_FUNDAMENTOS/F4_1_rho_emergente/` ni de
`results/BATERIA_FUNDAMENTOS/F4_1_rho_emergente/`.

---

## 1. Contexto y motivo

`CF2_estiramiento_motor.py` (leído completo, NO editado) impone directamente en código
`rho = RHO0 / (a**3)` dentro del brazo REAL. Eso es exactamente lo que F4-1 prohíbe: la
ley que se quiere probar (ρ∝a⁻³) está puesta a mano como fórmula, no medida. F4-1 debe
medir la densidad efectiva **a partir del propio estado del campo** (su contenido de
masa/energía, tal como evoluciona bajo la dinámica) y sólo entonces preguntar qué
exponente sale al compararla contra el factor de expansión `a` — sin que ese exponente
esté escrito en ninguna línea de la simulación.

**Distinción crítica con CF-2/F3-1/F4-2:** en CF-2 y F3-1 la cantidad medida es la forma
del GRADIENTE (`∇_fis = ∇_comov/a`, geometría pura de un salto de temperatura). Aquí la
cantidad medida es el CONTENIDO INTEGRADO del campo (su "masa/energía" total), que es una
magnitud conservada (o casi) bajo la dinámica interna, no una forma espacial. F4-2 (otro
prefijo, no tocado aquí) desacopla expansión y dilución para ver si ρ es causal propia;
F4-1 es previo a eso: sólo pregunta si ρ, definida honestamente desde el estado, CAE con
`a` y con qué exponente — sin la palanca de F4-2 de encender/apagar dilución aparte.

---

## 2. Definición EXACTA del medidor de densidad efectiva (el corazón del pre-registro)

### 2.1 Motor físico (heredado de CF2, sin la imposición de ρ)

Campo escalar `T` en malla comóvil `L×L` (`L=64`, igual "sello" que CF2/F3-1), inicializado
con el mismo salto `tanh` de CF2 (`initial_T`, ancho `W0=1.2`) más ruido gaussiano inicial
de amplitud `1e-4` (idéntico a CF2). `T` se interpreta como una densidad de masa/energía
LOCAL en unidades comóviles (convención declarada: valores de `T` ≈ contenido de
masa-energía por celda comóvil; no es literalmente temperatura aquí, es el mismo campo
reinterpretado para este experimento).

Evolución por difusión de 5 puntos (idéntica función `diffuse()` de CF2: laplaciano
`roll(-1)+roll(1)+roll(-1,0)+roll(1,0)-4·T`, `DT=0.25`, `N_SUB=2` subpasos), con **una
diferencia deliberada respecto de CF2**: el coeficiente de difusión `D` se mantiene
**FIJO en `D0=0.12`** durante TODA la corrida, para AMBOS brazos (REAL y NULL). Esto es
intencional y es la corrección central de T1/T2 frente a CF-2: si `D` dependiera de `ρ`
(como en CF-2, `D=D0·ρ/ρ0`), estaríamos usando la propia ley que queremos medir para
mover la dinámica que después mediría esa ley — circular. Aquí la dinámica interna del
campo (difusión + ruido) es **independiente de `a`**; sólo la CONVERSIÓN a volumen físico
depende de `a` (ver 2.2). Esto separa limpiamente "cómo evoluciona el campo" de "cómo se
mide su densidad física".

**Ruido dinámico (perturbación T7, no cosmético de semilla):** en cada subpaso de
difusión se inyecta ruido gaussiano de amplitud `sigma_ruido` escalado por
`sqrt(dt_sub)` (tipo Wiener, misma convención que usó F3-1), ANTES de recortar. Después
de cada paso completo, `T` se recorta a `[0, 1]` (igual que CF2). Este recorte es la
única fuente de no-conservación exacta de la masa comóvil total — es numérica y real,
no se disimula.

### 2.2 Observable primario — Método A: densidad global por conteo de contenido

En cada paso se mide, DEL ESTADO ACTUAL (no de una fórmula de `a`):

```
M_comov(t) = Σ_i T_i(t)          # "masa/energía" comóvil total (suma sobre las L² celdas)
```

`M_comov` es una cantidad MEDIDA en cada checkpoint, nunca asumida constante ni impuesta.
Bajo difusión pura con condiciones periódicas, `Σ lap(T)=0` exactamente (es una divergencia
discreta que telescopa a cero sobre el dominio periódico), así que `M_comov` DEBERÍA
conservarse salvo por: (a) el recorte a `[0,1]`, y (b) el ruido dinámico inyectado (que no
tiene media exactamente cero en una realización finita). Ambas son fuentes REALES de deriva,
no artificios — y es precisamente esa deriva (o su ausencia) lo que hace que el exponente
medido pueda diferir de −3, en vez de ser una certeza matemática.

La conversión a densidad FÍSICA usa el **único** ingrediente geométrico declarado
explícitamente (paralelo exacto a `∇_fis=∇_comov/a` de F3-1/CF2, que la batería ya trata
como conversión cinemática permitida, no como "la ley bajo prueba"): el volumen físico de
la malla comóvil crece como `V_fis(a) = L² · a³` (exponente `d=3`, declarado — NO
ajustado a posteriori para forzar −3; es la misma convención dimensional que el propio
documento de la batería usa para decir "−3 esperado en 3D"). Entonces:

```
ρ_eff_A(a) = M_comov(a) / (L² · a³)
```

**Por qué esto no es T1 disfrazado:** el exponente `d=3` del denominador es un factor de
conversión geométrico fijo, análogo al `/a` de `∇_fis`, NO una fórmula de la densidad en
función de `a` (no se escribe `ρ=f(a)` en ningún lado; se escribe `V_fis=L²a³`, un hecho
sobre cómo crece el volumen, y se DIVIDE por él el contenido MEDIDO). Si `M_comov(a)`
fuera perfectamente constante, el ajuste log-log daría exactamente pendiente −3 por pura
aritmética — y ESO ES EXACTAMENTE LO QUE SE REPORTA COMO POSIBLE RESULTADO, no se
esconde: si sale −3 "trivial" porque la masa se conserva casi perfectamente, ese es el
hallazgo honesto (conservación + geometría ⇒ ley de potencia), y la desviación medida
respecto de −3 exacto (por recorte + ruido) es la parte genuinamente empírica.

### 2.3 Observable secundario — Método B: densidad local de bloque central (verificación cruzada independiente)

Igual que CF2/F3-1 evitan artefactos de borde con una "banda central", aquí se mide la
densidad SÓLO en el bloque central de la malla (evita medir el mismo agregado global dos
veces con otro nombre — es un subconjunto espacial genuinamente distinto):

```
banda = T[L/4 : 3L/4, L/4 : 3L/4]                 # bloque central, 1/4 del área
M_banda(t) = Σ banda_i(t)
ρ_eff_B(a) = M_banda(a) / ( (L/2)² · a³ )
```

Si la estructura espacial que desarrolla el campo (difusión + ruido) concentra o vacía
contenido de forma no uniforme, `ρ_eff_B` puede diverger de `ρ_eff_A` — la COINCIDENCIA
(o no) de ambos exponentes es la verificación cruzada exigida por la Sección 2, regla 3
del documento madre (T2: un observable no define al otro).

### 2.4 NULL — sin expansión

Dinámica interna IDÉNTICA (mismo `T(t)`, misma semilla, mismo `sigma_ruido`, mismo
número de pasos) — la única diferencia es la conversión de volumen: `a≡1` fijo siempre
(no crece), así que:

```
ρ_eff_A_NULL(t) = M_comov(t) / L²
ρ_eff_B_NULL(t) = M_banda(t) / (L/2)²
```

Como la dinámica es la misma para REAL y NULL (no depende de `a`), **se corre UNA sola
trayectoria por (semilla, sigma_ruido)** y se computan ambas series (REAL con `a(t)³`
creciente, NULL con divisor fijo en 1) a partir del mismo `M_comov(t)`/`M_banda(t)`
medido — no son dos simulaciones físicas distintas, son dos formas de LEER el mismo
estado medido, lo cual hace el contraste REAL vs NULL más limpio (aísla exactamente el
efecto de la conversión de volumen, sin confundirlo con diferencias de semilla/ruido
entre brazos).

**Predicción pre-registrada del NULL (T4):** `ρ_eff_A_NULL(t)` y `ρ_eff_B_NULL(t)` deben
permanecer ~planas (sin tendencia monótona de caída) a lo largo del tiempo, porque no hay
ninguna dilución de volumen aplicada — sólo la deriva de `M_comov` por recorte/ruido, que
se espera pequeña. Si el NULL también cae con pendiente apreciable, el NULL no muerde y
se reporta como tal (no se disimula).

---

## 3. Reloj genético / mapeo a→t (idéntico a CF2/F3-1)

`tg = paso · dtg`, `dtg = 1/399` (mismo `ORIGINAL_STEPS_PER_TG=399` de CF2, sin retocar),
`a(tg) = exp(H_EXP · tg)`, `H_EXP=6.0` (mismo valor de CF2, T1: no se re-elige). Se
muestrea el estado en los `tg` objetivo de cada punto del barrido de `a`, igual método de
checkpointing de CF2/F3-1 (una sola trayectoria markoviana, sin re-simular desde cero por
punto).

---

## 4. Barrido pre-registrado

| Parámetro | Rango | Puntos | Espaciado |
|---|---|---|---|
| `a` (factor de expansión) | [1, 1e4] | 12 | log (geomspace) |
| `sigma_ruido` (amplitud de ruido dinámico) | [1e-5, 1e-1] | 8 | log (geomspace) |
| semillas | ver lista abajo | 16 | — |

Semillas (10 estándar del proyecto + 6 adicionales, ≥12 exigidas por la spec):
`[7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 13, 271828, 161803, 31415, 90210, 20260724]`

Total de trayectorias físicas: 8 (sigma) × 16 (semillas) = 128. Cada trayectoria se lee
dos veces (REAL y NULL, misma `M_comov(t)`/`M_banda(t)` medida) en los 12 checkpoints de
`a`.

---

## 5. Ajuste del exponente (verificación cruzada, sin imponerlo)

Por cada (semilla, sigma_ruido), para REAL y para NULL, y para Métodos A y B por
separado:

```
slope = pendiente por mínimos cuadrados de ln(ρ_eff) vs ln(a)      [para REAL]
slope = pendiente por mínimos cuadrados de ln(ρ_eff) vs ln(a_grid) [para NULL, mismo eje x
                                                                     que REAL para comparar
                                                                     aunque a_NULL≡1 en la
                                                                     dinámica misma]
```

El exponente reportado NUNCA se fija a −3; se calcula y se compara. La comparación con
−3 es POST-HOC, con incertidumbre (desviación estándar entre las 16 semillas, por cada
`sigma_ruido`), nunca al revés.

---

## 6. Criterio de PASS (congelado, no se toca tras ver resultados)

`MONO_TOL = 1e-6` (más laxo que CF2 porque aquí puede haber ruido de conteo/recorte
genuino, a diferencia del gradiente casi-determinista de CF2 — se declara ANTES de correr).

Por cada (semilla, sigma_ruido):
- `mono_REAL_A` = `ρ_eff_A` REAL es no-creciente en `a` dentro de `MONO_TOL`.
- `mono_NULL_A` = ídem para `ρ_eff_A_NULL`.
- `slope_REAL_A`, `slope_NULL_A` = pendientes log-log (Método A).
- `punto_pass_A` = `mono_REAL_A AND (NOT mono_NULL_A OR |slope_NULL_A - slope_REAL_A| >= 0.3)`
  (`0.3` = umbral de separación de pendiente, análogo al `SLOPE_DIFF_MIN` de CF2/F3-1,
  fijado aquí ANTES de ver resultados, no se re-elige, T1/T3).
- Igual construcción para Método B (`punto_pass_B`).
- `punto_pass = punto_pass_A AND punto_pass_B` (ambos métodos deben coincidir en el
  veredicto cualitativo — si uno pasa y el otro no, se reporta como discrepancia, no se
  promedia ni se elige el que convenga).

**PASS_F4-1** (curva completa, no una tasa única): se reporta `P(sigma_ruido)` = fracción
de semillas con `punto_pass=True`, para cada uno de los 8 valores de `sigma_ruido`. El
experimento se considera **robusto** si `P(sigma_ruido) >= 0.55` en TODOS los puntos del
barrido de ruido; si cae por debajo en algún punto, se reporta **FAIL_INESTABLE** en esa
región del ruido — no se suaviza el umbral después de ver el resultado (T3).

El exponente medido (media ± desviación estándar entre semillas, por `sigma_ruido`) se
reporta SIEMPRE, independientemente del veredicto PASS/FAIL — es el resultado central
que responde a la pregunta de F4-1, no un efecto secundario del gate.

---

## 7. Verificación cruzada (tres vías, obligatorias)

(a) **NULL muerde**: se reporta la tasa de "NULL muerde" (`NOT mono_NULL_A OR
    slope_diff>=0.3`) por `sigma_ruido`, Métodos A y B por separado.
(b) **Segundo observable independiente**: Método B (bloque central) contra Método A
    (suma global) — coincidencia de exponente medido, no sólo del booleano de PASS.
(c) **Auditoría en disco**: código (`F4_1_rho_emergente_motor.py`) + JSON crudo
    (`F4_1_rho_emergente_produccion_result.json`) quedan en disco para que alguien que
    NO escribió el código pueda re-verificar sin depender de este reporte narrado.

---

## 8. Qué NO se hace aquí

- No se impone `ρ=RHO0/a³` en ninguna línea del motor (a diferencia de CF-2). La única
  fórmula con `a` es la conversión de VOLUMEN (`V=L²a³`), declarada como convención
  geométrica, no como la densidad misma.
- No se acopla `D` (difusión) a `ρ` ni a `a` — eso es exactamente lo que F4-2 (otro
  prefijo) va a probar por separado; mezclarlo aquí sería contaminar la pregunta de F4-1
  con la de F4-2.
- No se toca `CF2_estiramiento_motor.py` ni ningún archivo de otro prefijo (`F3_*`,
  `F4_2_*`, etc.).
- No se cambia este criterio después de correr el motor (T3). Si el resultado es FAIL,
  inestable, o el exponente medido no es −3, se reporta tal cual — ésa es la pregunta
  que hace el experimento, no un fracaso.
- No topología, no commits (regla del director para este batch).
- No se auto-adjudica el veredicto de la batería más allá de este experimento puntual;
  la adjudicación final es de CS.
