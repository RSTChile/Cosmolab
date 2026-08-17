# PROTOCOLO F4-6 — PRE-REGISTRO
## "Control de dilución: ρ∝a⁻ⁿ barrido + doble NULL"

**Batería:** BATERÍA DE FUNDAMENTOS F1–F4 (Enfoque 4 — ¿la densidad tiene efecto causal propio?).
**Experimento:** F4-6, ejecutado por CC en paralelo con otros 23 experimentos (prefijo propio
`F4_6_`, sin tocar código/resultados de ningún otro experimento).
**Fuente autoritativa:** `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección F4-6 (línea ~293).

**Este documento se escribe y congela ANTES de correr el motor de producción.** El motor
(`F4_6_control_dilucion_motor.py`) y sus resultados
(`results/BATERIA_FUNDAMENTOS/F4_6_control_dilucion/`) se generan DESPUÉS — verificar mtime.

Fecha/hora de congelamiento: **2026-07-24 05:33 (-04)**, verificable con `ls -la` de este
archivo comparado con el mtime del motor `.py`.

---

## 1. Pregunta

F4-6 es el cross-check de F3-3: F3-3 barre n en ρ∝a⁻ⁿ midiendo el **gradiente físico** y su
pendiente log-log. Aquí se barre el MISMO n∈{1,2,3,4,5}, pero el observable declarado es
**la PERSISTENCIA de la diferencia** (no el gradiente crudo), y en vez de un solo NULL
(densidad fija) se exige **DOS NULLs independientes** corriendo en paralelo sobre los mismos
casos. Pregunta operativa: ¿el efecto de la densidad sobre la persistencia de la diferencia
sobrevive a ambos nulos, para todo n, sin que n=3 sea un punto singular no explicado?

## 2. Sustrato heredado (NO se retoca — mismo sello que CF2_estiramiento_motor.py)

Campo continuo T(x,y) en grilla L×L (L=64), condiciones periódicas. Perfil inicial: salto tipo
tanh de ancho comóvil W0=1.2, con ruido gaussiano de amplitud 1e-4 en la condición inicial
(idéntico a CF2). Difusión isótropa de 4 vecinos, coeficiente D, subpaso DT=0.25, N_SUB=2
subiteraciones. Reloj de expansión a(t_g)=exp(H_EXP·t_g), H_EXP=6.0. RHO0=1.0, D0=0.12,
ORIGINAL_STEPS_PER_TG=399. Ningún valor del sello se cambia para favorecer el resultado (T1).
Se sigue el mismo truco de CF2 de muestrear checkpoints de una única trayectoria markoviana en
vez de re-simular desde cero por cada `a` (idéntico resultado, mucho más barato).

**Generalización física propia de F4-6 (única diferencia dinámica con CF2):**

```
ρ(a) = ρ0 · a^(−n)         n ∈ {1, 2, 3, 4, 5}   (REAL, un brazo por cada n)
D(a) = D0 · (ρ(a)/ρ0) = D0 · a^(−n)
```

No se añade ruido dinámico por paso además del de condición inicial: se reutiliza la misma
justificación que el núcleo CF2 y el experimento hermano F3-3 (mismo sustrato sellado, T1) —
la diversidad entre semillas (12, cada una con su propio ruido de condición inicial) es la
perturbación exigida por la regla general T7 de la sección 1 del documento autoritativo. No se
introduce un eje de barrido adicional (amplitud de ruido dinámico) porque el bullet específico
de F4-6 no lo pide (`n × a × ≥12 semillas, con NULL-barajado-acople y NULL-densidad-fija`) y
añadirlo unilateralmente rompería la comparabilidad directa con F3-3, que usa exactamente el
mismo sustrato sin ruido por paso.

## 3. Barrido (T7 — nunca un punto/una semilla)

- **Exponente n:** `{1, 2, 3, 4, 5}` — los 5 valores del enunciado, ninguno privilegiado (T1).
- **Factor de expansión `a`:** `np.geomspace(1.0, 1000.0, 7)` — **la misma grilla exacta** que
  `CF2_estiramiento_motor.py` y que el experimento hermano F3-3, reutilizada para comparabilidad
  directa punto a punto y para no introducir una grilla ad-hoc que pudiera (consciente o
  inconscientemente) favorecer algún n.
- **Semillas:** ≥12 exigidas. Se usan las 10 semillas estándar de CF2 más las 2 semillas de
  extensión ya congeladas por F3-3 (dígitos de e y de φ), para máxima comparabilidad cruzada
  entre los dos experimentos que cruzan sobre el mismo n:
  `[7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 271828, 161803]` — 12 semillas.

Total de corridas físicas: brazo REAL = 5(n) × 12(semillas) = 60 trayectorias completas;
brazo NULL_DENSIDAD_FIJA = 12(semillas) trayectorias (no depende de n — ver §4); el
NULL_BARAJADO_ACOPLE se calcula in-situ sobre los campos ya generados (ver §5), no requiere
trayectorias físicas adicionales. Total: 72 integraciones de la PDE hasta t_g(a=1000).

## 4. Brazos y los DOS NULLs (T4 — deben poder no morder, y se reporta si no muerden)

- **REAL(n):** ρ=ρ0·a⁻ⁿ, D=D0·a⁻ⁿ, para n=1..5. Es el brazo con dilución física.

- **NULL_DENSIDAD_FIJA (control físico, independiente de n):** ρ≡ρ0, D≡D0 para todo `a`.
  Idéntico al `NULL_RHO_FIXED` de CF2 y al `n=0` de F3-3. Al no depender de n, se corre **una
  sola vez por semilla** (12 trayectorias) y se reutiliza como referencia compartida por los 5
  valores de n — correrlo 5 veces daría la misma trayectoria (determinista dada la semilla), así
  que repetirlo sería solo gasto de cómputo, no rigor adicional.
  Predicción a verificar (no asumida): sin dilución, la difusión sigue erosionando la
  estructura → la persistencia de NULL_DENSIDAD_FIJA debe decaer más rápido que la de REAL(n)
  para n>0, si la densidad tiene efecto causal propio.

- **NULL_BARAJADO_ACOPLE (control estadístico, permutación espacial):** en el sustrato de
  campo continuo de esta batería no existe una estructura discreta de "aristas/acople" que
  cortar (eso es propio del modelo de grafo del Enfoque 2). La interpretación congelada aquí —
  explícita para que quede auditable — es la MISMA que usa F1-1 para su NULL de persistencia:
  en cada checkpoint de `a`, se toma el campo `T` ya generado por la dinámica (REAL o NULL de
  densidad fija) y se baraja aleatoriamente el valor de sus L² celdas (permutación uniforme).
  Esto destruye la FORMA/coherencia espacial (el "acople" entre celdas vecinas) preservando
  exactamente el histograma de valores (misma varianza). Se promedian `N_PERM=5` permutaciones
  independientes por checkpoint (semilla de permutación derivada determinísticamente de
  `(seed, n, índice_de_checkpoint)`, ver motor) para tener una media y desvío estables del NULL,
  no una sola muestra ruidosa.
  Predicción a verificar: la persistencia de un campo barajado debe caer a ~0 (no hay
  coherencia espacial en un campo permutado), muy por debajo de REAL — si no cae, T4 se reporta.

## 5. Observable primario — PERSISTENCIA (T2: no comparte variable con el juez)

Para cualquier snapshot de campo `T` (L×L), se define el score de persistencia — forma×magnitud,
igual a la receta de F1-1, generalizado del anillo 1D a la grilla 2D nativa de este motor:

```
Tc          = T − mean(T)
Var(T)      = mean(Tc²)                                            # magnitud
autocorr_nn = [mean(Tc · roll(Tc,-1,eje_x)) + mean(Tc · roll(Tc,-1,eje_y))] / (2·Var(T))
Π(T)        = autocorr_nn(T) · Var(T)                               # forma × magnitud
```

`Π(T)` es alto cuando la estructura (el salto sembrado) sigue presente, espacialmente coherente
y con amplitud apreciable; cae a ~0 tanto si el campo se homogeneiza (Var→0) como si pierde
coherencia espacial aunque conserve varianza (autocorr_nn→0, caso del barajado).

Por cada (n, semilla, a) se calculan tres cantidades:

```
Π_REAL(n,s,a)                              # persistencia del campo REAL(n)
Π_NULL_A_mean(n,s,a), Π_NULL_A_std(n,s,a)  # media/std de 5 permutaciones del campo REAL(n) en ese punto
Π_NULL_B(s,a)                              # persistencia del campo NULL_DENSIDAD_FIJA (no depende de n)
```

Ninguna de estas cantidades usa variables de linaje/juez de otros experimentos de la batería.

### Observable secundario (verificación cruzada de método, §1 regla general "(b) segundo método")

Se registra también, por cada checkpoint, el gradiente físico `A_phys(a) = max(|∂T/∂x|_banda) / a`
— idéntico a `grad_metrics()` de CF2/F3-3 — como segunda vía de verificación independiente
(observable ortogonal: gradiente puntual vs. coherencia espacial global). No decide el PASS;
sirve para chequear que el veredicto cualitativo (REAL > ambos NULL, monótono en n) no es un
artefacto exclusivo de `Π`.

## 6. Criterio de PASS (congelado, T3 — no se toca si falla)

Constantes pre-registradas:

- `Z_NULL_A = 3.0` — condición de "muerde" para el NULL de permutación: se exige que `Π_REAL`
  supere a `Π_NULL_A_mean + Z_NULL_A · Π_NULL_A_std` (criterio de 3-sigma sobre la distribución
  de permutaciones, no un épsilon arbitrario — T1).
- `PASS_RATE_MIN = 0.55` — heredado literal de CF2 y F3-3, no ajustado a posteriori.
- `SINGULARITY_FACTOR = 3.0`, `SINGULARITY_FLOOR = 1e-4` — mismo criterio de curvatura que
  F3-3 (adaptado de `slope(n)` a `gap(n)`, ver abajo), umbral generoso pre-registrado.

Por semilla `s`, para cada `n`, en el punto final `a_final = 1000` (régimen de congelamiento,
el punto donde "persiste o no" se decide; las curvas completas 1..1000 se reportan igual, T5):

```
bite_A(n,s)   = Π_REAL(n,s,a_final) > Π_NULL_A_mean(n,s,a_final) + Z_NULL_A · Π_NULL_A_std(n,s,a_final)
bite_B(n,s)   = Π_REAL(n,s,a_final) > Π_NULL_B(s,a_final)
seed_pass(n,s) = bite_A(n,s) AND bite_B(n,s)
```

`rate(n) = (#semillas con seed_pass) / 12`. `verdict(n) = PASS si rate(n) ≥ PASS_RATE_MIN, si no FAIL.`

**Nota de frontera esperada (no es bug):** en `a=1` (checkpoint inicial, t_g=0, antes del primer
paso de difusión) REAL(n) y NULL_DENSIDAD_FIJA parten del MISMO campo inicial para una semilla
dada (aún no actuó ninguna dilución) → `Π_REAL(n,s,a=1) == Π_NULL_B(s,a=1)` exactamente, y
`bite_B` es falso por empate en ese punto para todo n. Esto es físicamente correcto (sin
expansión transcurrida no puede haber efecto de densidad todavía) y se documenta aquí para que
no se lea como fallo del instrumento; el juicio de PASS se hace en `a_final`, no en `a=1`.

**Verificación central pedida por el enunciado — "el efecto debe sobrevivir a AMBOS nulos":**
se reporta, además del `seed_pass` compuesto, las tasas `rate_A(n)` (solo bite_A) y `rate_B(n)`
(solo bite_B) por separado, para poder ver si el efecto sobrevive a uno y no al otro (resultado
frágil) o a ambos (robusto) — sin ocultar la diferencia si la hay.

**Chequeo de no-singularidad de n=3:** con `gap(n,s) = Π_REAL(n,s,a_final) − Π_NULL_B(s,a_final)`
y `gap(n) = mean_s(gap(n,s))`, se define la curvatura discreta
`curv(n) = gap(n−1) − 2·gap(n) + gap(n+1)` para n=2,3,4. Se marca **singular** si
`|curv(3)| > SINGULARITY_FACTOR · max(|curv(2)|, |curv(4)|, SINGULARITY_FLOOR)`. Se reporta el
resultado tal cual salga — n=3 NO se privilegia ni se fuerza a no ser singular (T1/T3).

**Qué NO significa "PASS" aquí:** igual que en F3-3, `seed_pass` es un chequeo compuesto (ambos
nulos muerden). El hallazgo real de este experimento es la familia completa de curvas
`Π_REAL(a; n)`, `Π_NULL_A(a; n)` y `Π_NULL_B(a)` — se reporta cruda independientemente de si
`seed_pass` da True o False, junto con `rate_A`, `rate_B`, `rate` compuesto, la curva `gap(n)` y
el resultado de singularidad de n=3. Ninguna lectura se privilegia sobre otra.

## 7. Qué NO es este experimento

- No decide si ρ∝a⁻³ es "la física correcta" del universo.
- No toca `CF2_estiramiento_motor.py`, ni ningún archivo o resultado de F3-3 ni de ningún otro
  experimento de la batería (F1, F2, F3, ni los otros F4). No hay coordinación en vivo con F3-3;
  la comparabilidad viene de reutilizar deliberadamente su misma grilla de `a` y sus mismas 12
  semillas, ya congeladas por ese experimento hermano cuando este protocolo se escribió.
- No se auto-adjudica el veredicto de la batería — el motor entrega números crudos; la lectura
  final la hace CS (Alexis) después. Si se detecta un error ajeno (p.ej. en el motor CF2
  heredado o en el protocolo de F3-3), se PARA y se reporta, no se corrige por cuenta propia.

## 8. Ruta de archivos

- Protocolo (este archivo):
  `Cosmogenesis-Web/codigo/BATERIA_FUNDAMENTOS/F4_6_control_dilucion/PROTOCOLO_F4-6_PREREGISTRO.md`
- Motor:
  `Cosmogenesis-Web/codigo/BATERIA_FUNDAMENTOS/F4_6_control_dilucion/F4_6_control_dilucion_motor.py`
- Resultados:
  `Cosmogenesis-Web/results/BATERIA_FUNDAMENTOS/F4_6_control_dilucion/F4_6_control_dilucion_<modo>_result.json`

---

**Fecha/hora de este pre-registro:** 2026-07-24 05:33 (-04) — ver mtime del archivo, congelado
antes de generar el motor y cualquier resultado.
