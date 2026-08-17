# PROTOCOLO CF-4b — PRE-REGISTRO
## "¿Existe un régimen donde la masa-ligadura domina sobre los constituyentes?" — barrido de γ

**Escrito:** 2026-07-24 (hora local, ANTES de escribir `CF4b_barrido_acoplamiento.py` y antes de
correr nada). Este archivo se congela aquí. Cualquier desviación del código respecto a lo
aquí escrito se reporta explícitamente, no se corrige en silencio.

**Autor de la corrida:** agente CC, bajo dirección de Alexis López Tapia (director) y diseño de
Claude Science (CS). **Batería:** CF (Cosmo-Física) — CF-4b, corrige CF-4.

**Instrucción autoritativa (no reinterpretada, seguida tal cual):**
`/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/INSTRUCCION_CF-4b_masa_ligadura_barrido_acoplamiento_PARA_CC_y_Grok.md`

**Código base (leído completo, NO editado, NO importado con efectos — solo reutilizado):**
`Cosmogenesis-Web/codigo/CF4_ligadura/CF4_confinamiento.py`

**Resultado y hallazgo que motivan esta corrida:**
`Cosmogenesis-Web/results/CF4_ligadura/CF4_produccion_result.json` y `CF4_RESUMEN.md`
(CF-4 FAIL: ratio_lig mediana máxima ~0.20 vs umbral 5.0, con `D_PHI=0.05, R0=2.0, U=0.5`
hardcodeados, nunca barridos — sesgo estructural hacia el FAIL, violación accidental de T1).
`/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/HALLAZGO_ABIERTO_etapa7_v6_masa_es_linaje_CS.md`
(por qué `co_member`/`linaje`/`n_long_co`/`fusion_events` están PROHIBIDOS en cualquier
observable de este arco).

---

## 0. Qué NO cambia respecto a CF-4 (anti-T2, anti-T3)

- **Observable de masa** (idéntico, sin tocar):
  - `m1` = suma de `(r_field·Φ² + U·Φ⁴) − V_min` sobre los nodos del cierre, "como si"
    estuvieran libres (masa de constituyentes libres, medida respecto al vacío).
  - `m2_real` = suma de `D_eff·(Φ_i − Φ_j)²` sobre los enlaces INTERNOS vivos del cierre
    (energía de ligadura = trabajo para separarlo).
  - `m2_null` = mismo cierre (mismos nodos, mismo número de enlaces internos), enlaces
    internos RE-CONECTADOS al azar entre esos mismos nodos (topología barajada).
  - `ratio_lig = m2_real / m1` (cierres con k≥2). `ratio_null = m2_real / m2_null`
    (cierres con k≥3, k=2 excluido por degenerado — único par posible).
- **NO se usa** `co_member_score`, `n_long_co_pairs`, `fusion_events` ni ningún tracking
  de linaje/persistencia entre pasos. Cada medición es una instantánea independiente.
- **Cierres** = componentes conexos del grafo de enlaces `ar`/`ad` vivos (BFS), sin
  criterio de átomo (`K_MIN/K_MAX/F_CORE/COHESION`). `k` emerge, no se impone (T0).
- **Umbrales de PASS, heredados de CF-4, CONGELADOS — no se tocan pase lo que pase:**
  - `ratio_lig ≥ THRESH_BIG = 5.0`
  - `ratio_null ≥ THRESH_NULL = 1.25`
  - banda **estable**: ≥3 puntos contiguos de γ (no un punto aislado) con ambos umbrales
    cumplidos, sobre ≥3 semillas evaluadas en ese punto.
- **Física del campo Φ** (evolución, corte de enlaces `weighted_cut`, BFS de cierres,
  `null_bind_energy`): reutilizada **por import** de `CF4_confinamiento.py` sin editar ese
  archivo (funciones `medium_norm`, `weighted_cut`, `find_closures`, `null_bind_energy`,
  la dataclass `P`, y las constantes `THRESH_BIG`/`THRESH_NULL`, importadas directamente
  para garantizar que son las mismas, no retipeadas). El bucle `simulate()` se **reimplementa**
  (copiado, no importa la función `simulate` de CF4) únicamente para agregar
  instrumentación de estabilidad numérica (sección 3) — la física dentro es idéntica.

---

## 1. Qué SÍ cambia: γ es el eje del barrido (corrige T1)

**Definición:** `γ = D_PHI / (R0·U)`, con `R0=2.0` y `U=0.5` **fijos** (idénticos a CF-4,
`R0·U = 1.0`), de modo que numéricamente `γ = D_PHI` en este diseño — pero se calcula y
reporta como razón explícita para que el significado físico (acoplamiento de ligadura
frente al pozo de potencial) quede claro, no como coincidencia numérica.

**Valor de referencia (CF-4, el que falló):** `D_PHI = 0.05` → `γ_CF4 = 0.05`.

**Rango de γ barrido (producción), decidido ANTES de correr, varias décadas a ambos lados
de γ_CF4, log-espaciado, 16 puntos:**

```
D_PHI_SWEEP = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
               0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
```

Esto cubre ~2 décadas por debajo de γ_CF4 (hasta 0.0005, 100× más chico) y ~3 décadas por
encima (hasta 50, 1000× más grande) — 5 décadas de span total, asimétrico hacia arriba
porque la hipótesis física (ligadura domina cuando el acoplamiento es fuerte) predice que
si existe un régimen, está por ENCIMA de γ_CF4, no por debajo. Se reporta el rango
completo igual, sin recortar.

`R0` y `U` **NO se barren** en esta corrida (la instrucción autoritativa permite barrer
los tres, pero el resumen operativo pactado con el director fija "vía escalar D_PHI/D_eff
manteniendo R0 y U fijos" — se sigue esa versión, más simple y suficiente para responder
la pregunta con un solo eje limpio; si CS pide después barrer R0/U también, es una
corrida nueva).

---

## 2. H_TOPO elegido (barrido secundario) — con criterio, de la tabla `per_seed_H_table` de CF-4

Se leyó `CF4_produccion_result.json` → `summary.by_H_TOPO` (agregado de
`per_seed_H_table`). Resumen relevante:

| H_TOPO | n_joint_pop (k≥3) | mean_k | median ratio_lig | median ratio_null |
|---|---|---|---|---|
| 0.002 | 130 | 733.2 | 0.196 | 0.994 |
| 0.004 | 130 | 582.4 | 0.188 | 0.991 |
| 0.007 | 130 | 243.2 | 0.179 | 0.999 |
| 0.010 | 183 | 89.3 | 0.025 | 1.006 |
| 0.020 | 2084 | 10.7 | 0.019 | 1.037 |
| **0.040** | **5232** | **2.85** | 0.060 | 1.054 |
| 0.070 | 3789 | 1.68 | 0.069 | 1.053 |
| **0.100** | 2432 | 1.40 | 0.066 | **1.072** |

**Se descartan H_TOPO ≤ 0.01:** `mean_k` de 89 a 733 indica un cierre casi-percolante
(prácticamente toda la red es una sola componente gigante) — no es un "cierre" análogo a
un hadrón, es un artefacto de la casi-ausencia de cortes, y además `n_joint_pop` es bajo
(130-183, pocas instancias independientes de cierre por corrida).

**Se eligen dos valores, ambos ya con estadística sólida en CF-4:**

- **`H_TOPO = 0.04`** — el de **mayor `n_joint_pop` (5232)** de todo el barrido de CF-4, y
  `mean_k ≈ 2.85`, el más cercano a una escala "pocos cuerpos" (análogo a 2-3 quarks). Es
  el punto con mejor estadística para medir `ratio_lig`/`ratio_null` en cierres pequeños
  y bien poblados.
- **`H_TOPO = 0.10`** — el que tuvo la **`median_ratio_null` más alta de todo CF-4 (1.072)**,
  el más cercano al umbral 1.25 de los ocho valores barridos. Es el punto más prometedor
  para ver si el NULL cede al variar γ, y sirve de cruce independiente frente a 0.04.

Ambos se barren contra el rango completo de γ (sección 1) en producción.

---

## 3. Estabilidad numérica (instrumentación nueva, pedida explícitamente por el director)

La actualización de Φ es Euler explícito: `Φ += DT_PHI·(−dV + D_eff·lap) + ruido`, con
`DT_PHI = 0.08` fijo (idéntico a CF-4). El límite de estabilidad estándar de un esquema
explícito de difusión 2D (stencil de 5 puntos, paso de malla=1) es aproximadamente
`D_eff · DT_PHI ≤ 0.25` ⇒ `D_eff ≤ ~3.1`. Con `D_eff = D_PHI · rho_hat_c` y
`rho_hat_c ≤ 1` (máximo al inicio de la corrida, decae después), valores de `D_PHI` por
encima de ~3-4 son candidatos a inestabilidad numérica pura (divergencia del esquema, NO
señal física de "ligadura fuerte"). El rango de γ barrido llega hasta 50, muy por encima
de ese límite — **a propósito**, para encontrar dónde ocurre y reportarlo, no para
evitarlo.

**Criterio de divergencia (aplicado por corrida, no oculta nada):**

1. Toda la aritmética de la actualización de Φ (Laplaciano, `dV`, `D_eff·lap`, suma final)
   se ejecuta bajo `np.errstate(over="raise", invalid="raise")`. Si ocurre overflow o una
   operación inválida (`inf−inf`, etc.), se captura `FloatingPointError`, se marca esa
   corrida `status="diverged"` con el `step` exacto, y se **detiene esa corrida ahí**
   (no se sigue integrando sobre un campo ya roto).
2. Verificación adicional por paso: si `Phi` deja de ser finito (`np.isfinite`) sin haber
   disparado `FloatingPointError` (posible con ciertos flags de NumPy), se marca igual.
3. Se registra `max_abs_phi` (máximo de `|Φ|` alcanzado en toda la corrida, mientras siguió
   finito) como diagnóstico cuantitativo de qué tan cerca estuvo del borde, incluso en
   corridas que NO divergieron.
4. Corridas que divergen contribuyen **0 registros** a las curvas `ratio_lig(γ)` /
   `ratio_null(γ)` (no hay observable válido que extraer de un campo roto) pero se listan
   explícitamente en el JSON de salida (`stability_table`), con `D_PHI`, `H_TOPO`, `seed`,
   `diverged_step`. **No se ocultan ni se promedian con las corridas válidas.**

Esto se corre primero en el smoke (sección 4) para confirmar que el mecanismo de detección
funciona antes de fiarse de los puntos de γ alto en producción.

---

## 4. Semillas

- **Smoke:** `(7, 42)` — 2 semillas, subconjunto de las 10 estándar del proyecto
  (`CF4_confinamiento.SEEDS`).
- **Producción:** `(7, 42, 99, 777)` — 4 semillas, también subconjunto de las 10 estándar.
  Se usan 3-4 semillas por punto de γ (no 10) porque, por la lección de CF-2 (el campo es
  una PDE casi determinista — "10/10 semillas" resultó ser el mismo resultado repetido, no
  10 confirmaciones independientes), la robustez real de este experimento viene de **barrer
  γ**, que sí perturba la dinámica, no de multiplicar semillas. Se reporta la dispersión
  real entre semillas en cada punto de γ (media, min, max), no solo un promedio.

---

## 5. Tamaño de grilla y pasos

- **Smoke:** `L=16, pasos=120` (idéntico a los defaults de smoke de CF-4) — 3 puntos de γ
  × 2 `H_TOPO` × 2 semillas = 12 corridas, solo para validar mecánica + disparo del
  detector de divergencia. **No decide PASS.**
  - `D_PHI_SMOKE = (0.005, 0.05, 5.0)` — un punto muy por debajo de γ_CF4, el propio
    γ_CF4, y un punto muy por encima (candidato a inestable) para verificar que el
    detector de divergencia efectivamente se activa antes de escalar.
- **Producción:** `L=28, pasos=400` (idéntico a CF-4) — 16 puntos de γ × 2 `H_TOPO` × 4
  semillas = 128 corridas.

---

## 6. Qué se entrega (idéntico a lo pedido en la instrucción autoritativa, §7)

- Este protocolo (fechado, antes del motor).
- `CF4b_barrido_acoplamiento.py`.
- `CF4b_smoke_result.json`, `CF4b_produccion_result.json` (curva completa `ratio_lig(γ)`,
  `ratio_null(γ)`, histograma de `k`, dispersión entre semillas, tabla de estabilidad).
- `CF4b_RESUMEN.md` (crudo, sin adjudicar).
- Tiempo de corrida y pico de RAM.

**No se adjudica "existe régimen" o no en este documento ni en el código.** El código
aplica mecánicamente el criterio congelado (sección 0) para poder reportar bandas
candidatas, pero la lectura final ("¿hay mecanismo?") la hace CS con la curva completa a
la vista, verificando en disco.

---

## 7. Reglas de ejecución (heredadas de la batería CF, sin cambio)

Pre-registro antes de correr · barrido de rango + semillas, nada de un punto · NULL que
muerde, se verifica que cae · el observable ≠ su juez · todo gate puede fallar · quien
corre no cambia el código a criterio propio (si ve un error ajeno, PARA y reporta la línea
exacta a CS) · ejecución completa · verificación cruzada en disco, no de palabra · se
entrega crudo a CS, sin adjudicar.
