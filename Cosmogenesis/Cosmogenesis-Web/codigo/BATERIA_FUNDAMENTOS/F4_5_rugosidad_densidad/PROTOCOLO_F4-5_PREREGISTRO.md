# PROTOCOLO F4-5 — PRE-REGISTRO
## "Rugosidad de densidad: ¿importa el gradiente de ρ, no solo su valor medio?"

**Escrito:** 2026-07-24 05:35 (hora local, ANTES de escribir o correr
`F4_5_rugosidad_motor.py`).
**Autor de la corrida:** agente F4-5 (Claude Code / CC), bajo dirección de
Alexis López Tapia (CS diseñó la batería completa).
**Batería:** BATERIA_FUNDAMENTOS, Enfoque 4 (descenso de densidad por
expansión), experimento F4-5 — spec exacta en
`BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección "F4-5".

Este documento se congela ANTES de escribir el motor de simulación.
Cualquier desviación del código respecto a lo aquí escrito debe reportarse
explícitamente, no corregirse en silencio. El criterio de PASS/FAIL no se
toca después de ver resultados (T3). Ningún veredicto final se adjudica
aquí — el director/CS adjudica con la curva cruda.

---

## 0. Pregunta exacta (spec del documento madre)

> "El fondo no es uniforme — ¿que unas zonas sean más densas que otras
> cambia dónde persiste la diferencia?" Sembrar un campo de densidad con
> estructura espacial (rugosidad, no uniforme) y medir si la persistencia
> de una diferencia física se correlaciona con las zonas más densas vs.
> las diluidas. NULL = rugosidad de densidad barajada (misma estadística,
> sin estructura espacial). Si la persistencia correlaciona con la
> estructura REAL y no con la barajada, la rugosidad de ρ es causal.

Esto es distinto de F4-3/F4-4 (que barren el VALOR MEDIO de ρ). Aquí ρ
tiene la MISMA media global en todo el barrido — lo que se barre es cuánta
ESTRUCTURA ESPACIAL (varianza espacialmente correlacionada, "parches")
tiene ρ alrededor de esa media.

---

## 1. Por qué el diseño evita T0/T1/T2 (leer antes del código)

**Riesgo de circularidad que este diseño evita explícitamente:** si el
observable de "persistencia por celda" se comparara directamente contra
el valor de ρ EN ESA MISMA CELDA, el resultado sería trivial por
construcción (D(x) se define a partir de ρ(x), así que por supuesto la
celda con más ρ difunde más — eso no prueba que la ESTRUCTURA ESPACIAL
importe, solo que D depende de ρ puntual, lo cual es una entrada del
modelo, no un hallazgo).

**Lo que este experimento mide de verdad:** el campo T difunde con un
operador de vecinos (Laplaciano), así que su dinámica en cualquier celda
depende de los valores de D en un VECINDARIO, no solo del de esa celda.
Por eso el observable se agrega a nivel de ZONA (bloque B×B, no celda) y
se compara entre:

- **REAL:** ρ tiene estructura espacialmente correlacionada (parches
  suaves — "rugosidad" genuina, con longitud de correlación fija).
- **NULL:** exactamente los mismos valores de ρ (mismo histograma, misma
  media, misma varianza, mismos percentiles — "misma estadística"), pero
  RE-UBICADOS al azar celda por celda, destruyendo la coherencia espacial
  (parches → ruido blanco).

Si la correlación zona-a-zona entre densidad y persistencia sobrevive en
REAL y colapsa en NULL, es la ESTRUCTURA (no solo el valor de ρ) la que
importa — eso es lo que T0/T2 exige medir: "la estructura de densidad se
mide contra su propio barajado, no se impone a mano".

---

## 2. Física del motor (autocontenido, no importa CF2/CF4)

### 2.1 Densidad con rugosidad, REAL

Campo base de ruido blanco `w(x,y) ~ N(0,1)`, suavizado con un kernel
gaussiano de sigma FIJO `SIGMA_SUAVIZADO = 3.0` celdas (no se barre — es
la escala física de "tamaño de parche", constante en todo el experimento,
para que el único eje barrido sea la AMPLITUD, no la escala). El suavizado
se normaliza a media 0 / std 1 después de suavizar. La densidad:

```
rho(x,y) = RHO0 * (1 + amplitud_rugosidad * w_suave_normalizado(x,y))
rho(x,y) = clip(rho, RHO0 * RHO_FLOOR_RATIO, None)   # evita rho<=0
```

`amplitud_rugosidad = 0` ⇒ `rho ≡ RHO0` (uniforme, caso trivial de
control, incluido en el barrido).

### 2.2 Densidad barajada, NULL

`rho_null = np.random.permutation(rho.flatten()).reshape(L, L)` — permuta
las celdas de la MISMA matriz `rho` ya generada (idéntico histograma,
media, varianza, cuantiles — "misma estadística"), destruyendo la
coherencia espacial de parches. Se aplica la MISMA permutación fija
(elegida con el rng de la semilla) tanto a `rho_init` como para derivar
`rho(t)` en cada paso (la dilución temporal es un factor multiplicativo
uniforme, así que conmuta con la permutación — no hace falta re-barajar
en cada paso).

### 2.3 Dilución temporal (Enfoque 4: ρ∝a⁻³, ligada a Enfoque 2: r=H/D)

```
a(tg)    = exp(H * tg),           tg ∈ [0, TG_MAX]
rho(x,y,tg) = rho_init(x,y) / a(tg)**3
D(x,y,tg)   = D0 * clip(rho(x,y,tg)/RHO0, D_FLOOR_RATIO, None)
```

`D0` = difusividad de referencia a densidad media RHO0 sin diluir.
`H` se parametriza como `r = H / D0` (mismo eje `r` que Enfoque 2 —
competencia expansión/reabsorción), barrido explícito. La dilución es
UNIFORME multiplicativa sobre todo el campo (mismo factor `1/a³` en cada
celda), así que el ORDEN relativo de zonas densas/diluidas NO cambia con
el tiempo — solo su escala absoluta. Esto es intencional: aísla el eje de
"estructura espacial" (amplitud_rugosidad) del eje de "cuánta dilución
total ocurrió" (r), sin mezclarlos.

### 2.4 Campo T (la "diferencia" cuya persistencia se mide)

`T_init(x,y)` = ruido gaussiano de media 0, `std = T_NOISE_STD = 0.10`
FIJO (no se barre — la amplitud de la perturbación no es el eje de este
experimento), generado con un RNG INDEPENDIENTE del que generó `rho`
(misma semilla base pero stream derivado distinto: `seed_rho = seed`,
`seed_T = seed + 10_000_000`), para que no exista correlación impuesta a
mano entre dónde arranca T y dónde está la densidad — cualquier
correlación final entre persistencia y densidad tiene que emerger de la
DINÁMICA (difusión modulada por D(x,y,t)), no de la siembra.

Evolución (difusión con D espacialmente variable, esquema de volúmenes
finitos con D promediada en cada cara, conservativo — NO el Laplaciano de
D uniforme de CF2/CF4, porque aquí D varía por celda):

```
flujo_x[i,j] = 0.5*(D[i,j]+D[i,j+1]) * (T[i,j+1]-T[i,j])   # cara derecha
flujo_y[i,j] = 0.5*(D[i,j]+D[i+1,j]) * (T[i+1,j]-T[i,j])   # cara abajo
T[i,j] += dt * ( flujo_x[i,j] - flujo_x[i,j-1]
               + flujo_y[i,j] - flujo_y[i-1,j] )
        + sigma_dinamico * ruido_normal()   # forzamiento térmico leve, ver 2.5
```
(fronteras periódicas, `np.roll`, igual convención que CF2/CF4).

### 2.5 Forzamiento dinámico leve (no cosmético de semilla)

Para no depender únicamente de la aleatoriedad de la condición inicial
(lección T7 de la batería), en cada subpaso se añade ruido térmico de
amplitud FIJA `SIGMA_DINAMICO = 0.01` (no se barre — esa es la tarea de
F1-5, no de F4-5; aquí se incluye solo para que la dinámica no sea
puramente determinista dado el campo D). Este término es idéntico en REAL
y NULL (mismo rng derivado), así que no puede introducir sesgo entre
ambos brazos.

### 2.6 Parámetros fijos (no barridos, iguales en REAL y NULL)

```
L = 64                    # tamaño de grilla LxL
B = 8                     # tamaño de bloque/zona (8x8 = 64 zonas)
RHO0 = 1.0
D0 = 0.12                 # difusividad de referencia (mismo orden que CF2)
D_FLOOR_RATIO = 0.05      # piso de rho/D relativo (evita D=0 o rho<=0)
SIGMA_SUAVIZADO = 3.0     # celdas, longitud de correlación de la rugosidad REAL
T_NOISE_STD = 0.10        # amplitud fija de la siembra de T
SIGMA_DINAMICO = 0.01     # ruido térmico fijo por subpaso (sec. 2.5)
TG_MAX = 1.0              # reloj genético normalizado (igual convención que CF2)
N_STEPS = 240             # pasos de tg entre 0 y TG_MAX
N_SUB = 2                 # subiteraciones de difusión por paso de tg
DT_SUB = 0.125            # dt físico por subiteración (idéntico orden a CF2: DT=0.25/N_SUB)
```

**Verificación de estabilidad numérica (antes de correr, no ajustada
después):** `D_max ≈ D0*(1+amplitud_max) = 0.12*(1+3.0) = 0.48`. Límite
de estabilidad explícito 2D: `dt*D <= 0.25` (por dimensión, esquema de
5 puntos). `DT_SUB*D_max = 0.125*0.48 = 0.06 < 0.25` — margen amplio, sin
necesidad de reducir `DT_SUB` para ningún punto del barrido.

---

## 3. Observables (dos métodos independientes, T2)

Por zona `z` (bloque B×B, 64 zonas en total, `L/B=8` por lado):

```
rho_zona(z)        = media espacial de rho_init dentro del bloque z
persistencia(z)    = media espacial de |T_final| dentro del bloque z
```

`rho_init` (no `rho(t_final)`) se usa como identificador de "qué tan
densa es la zona z", porque la dilución es un factor multiplicativo
uniforme (sec. 2.3) — el ORDEN relativo de las zonas por densidad es
invariante en el tiempo, así que usar `rho_init` es equivalente a usar
`rho(t)` para efectos de ranking, y es más limpio de auditar.

### Método A (primario) — correlación de Spearman zona-a-zona

```
corr_A = spearman(rho_zona[1..64], persistencia[1..64])
```
Predicción física direccional: `D` crece con `rho` (sec. 2.3) ⇒ zonas más
densas difunden/erosionan MÁS ⇒ persiste MENOS ⇒ se espera
`corr_A(REAL) < 0` (anti-correlación), y `corr_A(NULL) ≈ 0`.

### Método B (independiente) — brecha de cuartiles extremos

Ordenar las 64 zonas por `rho_zona`; tomar el cuartil superior (16 zonas
más densas) y el inferior (16 zonas más diluidas):

```
brecha_B = media(persistencia | cuartil_diluido) - media(persistencia | cuartil_denso)
```
Predicción: `brecha_B(REAL) > 0` (zonas diluidas persisten más que las
densas); `brecha_B(NULL) ≈ 0`.

Método A (correlación de rango, usa las 64 zonas) y Método B (diferencia
de medias entre grupos extremos) son estadísticos DISTINTOS — ninguno se
define en términos del otro (T2). Ambos deben coincidir en signo y en
comportamiento cualitativo frente al barrido de `amplitud_rugosidad` para
que el hallazgo se considere robusto.

### Tercera verificación — auditoría en disco

El JSON de salida incluye, por cada combinación
`(amplitud_rugosidad, r, seed, modo)`, los 64 valores de `rho_zona` y los
64 de `persistencia` CRUDOS (no solo los escalares `corr_A`/`brecha_B`),
para que alguien que NO escribió este código pueda re-calcular ambos
observables de forma independiente sin confiar en el resumen.

---

## 4. Barrido pre-registrado

```
amplitud_rugosidad ∈ {0.0, 0.05, 0.15, 0.30, 0.60, 1.00, 1.80, 3.00}   # 8 puntos, más fino cerca de 0
r = H/D0 ∈ {0.1, 0.3, 1.0, 3.0, 10.0, 30.0}                            # 6 puntos, cruza r≈1 (régimen de Enfoque 2)
SEEDS = (7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 161803, 271828)  # 12 semillas: las 10 estándar del proyecto + 2 (φ, e en miles, arbitrarias, fijadas ANTES de correr)
MODOS = REAL, NULL_BARAJADO
```

Total: 8 × 6 × 12 × 2 = 1152 corridas. Grilla L=64 con difusión vectorizada
en numpy — costo esperado bajo (se reporta runtime real en el resultado;
autorización de cómputo largo vigente si el costo real resulta mayor al
estimado).

**Nota T1 (nada elegido para dar el resultado):** la grilla de
`amplitud_rugosidad` es más fina cerca de 0 (para resolver si hay un
umbral) y el grid de `r` cruza 1 (régimen crítico ya encontrado en
Enfoque 2) — ambas elecciones de RESOLUCIÓN, no de VALOR objetivo; no se
excluye ningún punto que pudiera dar FAIL.

---

## 5. NULL adicional de sanidad (control ε=0 de este experimento)

En `amplitud_rugosidad = 0.0`, `rho` es literalmente uniforme, así que
`rho_zona` es constante en las 64 zonas por construcción — `corr_A` no
está definida (varianza cero) y `brecha_B` debe ser ≈0 por definición
(cuartiles indistinguibles salvo ruido de discretización). Este punto es
el control trivial de la batería (T6): si en `amplitud_rugosidad=0` se
observara una correlación fuerte y no-nula, sería señal de un error de
implementación (persistencia correlacionando con algo que no es densidad,
por ejemplo con la posición absoluta en la grilla) — se reporta
explícitamente si esto ocurre, no se descarta el punto.

---

## 6. Criterio de PASS (congelado ANTES de correr)

Por cada combinación `(amplitud_rugosidad, r)`, agregando sobre las 12
semillas:

```
bite(amplitud, r) = fracción de semillas donde
    signo(corr_A_REAL) coincide con la predicción física (negativo)  AND
    |corr_A_REAL| > |corr_A_NULL| + MARGEN_CORR            AND
    signo(brecha_B_REAL) coincide con la predicción física (positivo) AND
    brecha_B_REAL > brecha_B_NULL + MARGEN_BRECHA
```

```
MARGEN_CORR   = 0.10   # separación mínima en |Spearman rho|
MARGEN_BRECHA = 0.01   # separación mínima en unidades de T (T_NOISE_STD=0.10, así que 0.01 = 10% de la amplitud sembrada)
PASS_RATE_MIN = 0.55   # mismo umbral que el resto de la batería (CF2/CF4)
```

`(amplitud, r)` se marca **PASS** si `bite(amplitud, r) ≥ PASS_RATE_MIN`.

**Veredicto del experimento (tres lecturas posibles, pre-registradas,
ninguna se privilegia):**

1. **Rugosidad causal (scale-free o con umbral):** existe un rango de
   `amplitud_rugosidad > 0` con PASS consistente y la tasa de PASS NO cae
   con `amplitud_rugosidad` creciente (o cae solo cerca de 0, por falta de
   señal, lo cual es esperable) — se reporta la curva completa
   `bite(amplitud)` por cada `r`.
2. **Rugosidad sin efecto propio:** `bite(amplitud, r) < PASS_RATE_MIN`
   en todo el barrido, o `corr_A_NULL`/`brecha_B_NULL` no colapsan
   respecto a REAL (el NULL no muerde, T4 — se reporta así explícitamente,
   sin mover el umbral).
3. **Métodos A y B discrepan:** si el signo o la tendencia de A y B no
   coinciden en una fracción sustancial del barrido, se reporta como
   hallazgo de fragilidad del observable, no se descarta ninguno de los
   dos a posteriori.

**No se auto-adjudica** cuál de las tres lecturas aplica en el reporte
final más allá de describir los números — la lectura definitiva es de
CS/director.

---

## 7. Plan de ejecución

1. **Smoke** (valida mecánica, no decide PASS): `L=32`, `N_STEPS=60`,
   `amplitud_rugosidad ∈ {0.0, 0.3, 3.0}`, `r ∈ {0.3, 3.0}`,
   `seeds = (7, 42)` → 3×2×2×2 = 24 corridas. Verifica: `rho>0` siempre,
   sin NaN/Inf en T, `corr_A` definido para amplitud>0, barajado
   efectivamente destruye la estructura (verificar autocorrelación
   espacial de `rho_null` ≈ 0 vs `rho_real` > 0), tiempo de ejecución
   razonable.
2. **Producción:** parámetros de la sección 4 completos (1152 corridas).
3. **Análisis:** tabla `bite(amplitud, r)`, curvas `corr_A`/`brecha_B`
   REAL vs NULL vs `amplitud_rugosidad` (una curva por `r`), dispersión
   entre semillas (percentiles, no solo media), verificación T4/T5/T6
   explícita.
4. Reporte crudo a CS — sin adjudicar veredicto final de la batería.

---

## 8. Prohibiciones explícitas / alcance

- No se edita `CF2_estiramiento_motor.py` ni ningún otro motor existente
  de la batería (T1/T2 se verifican de forma autocontenida, sin heredar
  código de otros experimentos).
- No se implementa dimensión/topología fuera de la grilla 2D periódica
  estándar del proyecto (T0).
- No se hacen commits de git. No se toca ningún archivo fuera de
  `codigo/BATERIA_FUNDAMENTOS/F4_5_rugosidad_densidad/` y
  `results/BATERIA_FUNDAMENTOS/F4_5_rugosidad_densidad/`.
- El criterio de PASS (sección 6) no se mueve después de correr (T3). Si
  al correr se encuentra un defecto de mecánica (como el m1 negativo de
  CF-4), se documenta como adenda fechada ANTES de producción, nunca
  después de ver el resultado final.
- Si se detecta un error en código AJENO (de otro experimento de la
  batería), se PARA y se reporta a CS — no se edita.
