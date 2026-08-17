# κ_P, κ_Δ y κ_V remedidos en un instrumento CON MORTALIDAD (CG002)

**Fecha:** 13-ago-2026 · **Instrumento:** `cg002_acoplamiento.py` (motor v0.1c, no reescrito) ·
**Motivo:** los tres κ medidos sobre sumideros de Phantom se cayeron
(`VALIDACION_kappaP_kappaDelta_controles_con_malla_CS.md`): κ_P era un reloj de nacimiento (r=−0,97 con
t_nac; 280/280 sumideros llegan al final → mortalidad cero), κ_Δ era la masa (r=+0,997), κ_V casi copia
de la masa. **El problema era el instrumento, no el control.**

**Estado: ⚠️ NO ES UN CIERRE.** Se reportan números y curvas. Ningún veredicto es válido sin el director.

---

## 0 · PRE-REGISTRO

**Escrito y guardado ANTES de correr una sola simulación. No se modificó después.**
(Verificable: este archivo se creó antes que `kappas_mortalidad_barrido.py` y que todos los CSV/PNG.)

### 0.1 · Por qué CG002 y no Phantom

CG002 tiene un **umbral literal de muerte**: `_vivo(s) = s > KAPPA_S` (KAPPA_S=1e−6). Los nodos **sí**
mueren. Hay, por tanto, un piso de persistencia que buscar. En Phantom no lo había.

### 0.2 · Derivación analítica declarada ANTES de medir

El micro-paso de CG002 es, textualmente (líneas 331–352 de `cg002_acoplamiento.py`):

```
s ← (1−MU)·s                                        # decaimiento
Δs_i = ETA·α·Σ_{j≠i} g_ij·√(s_i·s_j)                # acoplamiento
s ← s + Δs
```

con `g_ij = cos(2π(ω_i−ω_j)/K + θ_CP)`, MU=0,01, ETA=0,05.

Dos consecuencias **algebraicas**, no empíricas, que se declaran de antemano:

**(a) El mapa es homogéneo de grado 1 en S.** `√(s_i s_j)` escala igual que `s`. Si se multiplica todo S
por λ, la trayectoria entera se multiplica por λ. Por lo tanto **la escala de S no puede fijar ningún
piso**: cualquier "S mínimo" que aparezca al bajar S0 tiene que venir de las **constantes absolutas
puestas a mano** (KAPPA_S, W_MIN=0,1·S0, EPS_TAU), no de la dinámica. → *Test de tautología obligatorio.*

**(b) Sí existe un umbral no trivial, pero está en el COCIENTE acoplamiento/decaimiento, no en S.**
Poniendo x_i=√s_i, la suma total crece como
`S_tot' = (1−MU)·[ xᵀx + ETA·α·xᵀGx ]` con G_ij=g_ij, diag(G)=0. El cociente de Rayleigh da:

> **α_c = (MU/(1−MU)) / (ETA · λ_max(G))**

Con MU=0,01, ETA=0,05: **α_c = 0,20202 / λ_max(G)**. Como G = C − I con C=cos(θ_i−θ_j) de rango 2 y
traza N, se predice λ_max(G) ∈ [N/2−1, N−1] → para N=8, **α_c ≈ 0,029–0,067**, y **depende de la
configuración ω concreta** (o sea, de la estructura).

**Predicción de estructura declarada de antemano:** λ_max(G) ≥ media de sumas de fila = (N−1)·ḡ
(Rayleigh con el vector uniforme). El brazo BARAJADO re-permuta ω en cada paso, de modo que su
acoplamiento efectivo es el **promedio** ḡ, no el máximo. Por lo tanto:

> **α_c(REAL) ≤ α_c(BARAJADO)**, con igualdad sólo si el vector uniforme ya es el autovector dominante.
> Si ḡ < 0, BARAJADO no persiste a **ningún** α.

### 0.3 · Qué se barre

| Barrido | Parámetro | Rango | Para qué |
|---|---|---|---|
| **A** (κ_P) | α (acoplamiento) | 1e−6 … 1e+1, **7 décadas**, log | ¿hay α_c nítido? |
| **A′** (κ_P) | MU (decaimiento) | 1e−5 … 1e−1, **4 décadas**, log | ¿el umbral es el cociente y no α? |
| **C** (κ_P) | S0 (persistencia inicial) | 1e−12 … 1e+3, **15 décadas** | test de tautología (a) |
| **B** (κ_Δ) | δ (diferencia de fase entre dos nodos) | 1e−20 … 1e0, **20 décadas**, log | ¿hay diferencia mínima operable? |
| **B′** (κ_Δ) | EPS_TAU (umbral de operabilidad) | 1e−12 … 1e−1 | ¿el piso de κ_Δ es la constante puesta a mano? |

Todos los barridos, **en los tres brazos**: REAL (α>0, ω fija) · BARAJADO (α>0, ω re-permutada por paso) ·
ALFA_0 (α=0, control G del protocolo).

### 0.4 · Criterios de decisión (pre-registrados, no se cambian)

**κ_P:**
- **Umbral nítido y en el MISMO lugar en las tres precisiones** (float64 · float80 · mpmath 50 dígitos)
  → κ_P existe como piso, y es un piso de **cociente acoplamiento/decaimiento**, no de S.
- **Umbral que se mueve al cambiar el flotante** → **artefacto numérico** (lección F1-3).
- **Caída suave sin umbral** → no hay piso; κ_P queda como cota inferior trivial (>0).
- **El umbral coincide con la fórmula de §0.2 dentro de la resolución del barrido** → el piso es
  **analítico**, no emergente: se reportará como tal (un piso analítico NO es un descubrimiento
  empírico, es una propiedad del modelo).

**κ_Δ:**
- **δ_min operable estable en las tres precisiones** → κ_Δ existe.
- **δ_min operable ∝ ε_máquina** (baja ×1000 al pasar de float64 a float80, y sigue bajando con mpmath)
  → **artefacto numérico**, no piso.
- **δ_min operable ∝ EPS_TAU / W_MIN** → el piso es la constante puesta a mano, es decir **tautología**.

**κ_V:** primero se determina si es medible; si no lo es, se documenta por qué y se deja como
"no aplicable a este nivel de descripción" (como κ_O / κ_LF / κ_H). **No se fuerza.**

### 0.5 · Guardas que se reportan sí o sí

1. **Tautología** — ¿podía el número salir distinto? Explícito para cada κ.
2. **Identidades algebraicas** — correlación de cada κ con `n_vivos_fin`, `S_final_max`, `delta_struct`,
   `n_aristas`. r>0,9 ⇒ es esa variable reescalada (fue lo que hundió a κ_Δ en Phantom).
3. **Piso de ruido** — con n semillas por brazo, cuál es el p mínimo alcanzable.
4. **No isomorfía del barajado** — barajar una vez al inicio es un renombre de nodos (isomorfo exacto,
   ya verificado el 13-ago); se baraja **por paso** y se vuelve a verificar.
5. **Réplica del motor** — el barrido usa un núcleo parametrizado por precisión; se verifica contra
   `cg002_acoplamiento.correr` original antes de usarlo.

---

<!-- RESULTADOS SE AÑADEN DEBAJO DE ESTA LÍNEA, DESPUÉS DE CORRER -->

# RESULTADOS

**Corrida:** 13-ago-2026, 943 s. `kappas_mortalidad_barrido.py` (barridos A/A'/B/C + guardas),
`kappaV_sustrato_cg002.py` (κ_V), `kappaDelta_refino.py` (arreglo de la guarda de precisión en κ_Δ),
`kappas_mortalidad_lectura.py` (lectura, read-only).
**Datos:** `kappas_mortalidad_curva_alpha.csv` (1585 filas) · `kappas_mortalidad_curva_S0.csv` (640) ·
`kappas_mortalidad_curva_delta.csv` (883) · `kappas_mortalidad_resumen.json` ·
`kappaV_sustrato_cg002.json` · `kappaDelta_refino.json` · `kappas_mortalidad_curvas.png`.

---

## 0 · EN CASTELLANO SIMPLE (antes de los números)

**La analogía de la fogata.** κ_P pregunta: *¿existe una cantidad mínima de leña por debajo de la
cual el fuego no se sostiene?*

En Phantom la pregunta no tenía sentido porque **ningún fuego se apagaba nunca**: las 280 fogatas
llegaban encendidas al final. Medir "cuánto duran" ahí es medir "a qué hora las prendieron" — que es
exactamente lo que pasó (r=−0,97 con la hora de nacimiento).

En CG002 los fuegos **sí se apagan**: en la producción del 13-ago murió el **38,7 %** de los nodos en
REAL, el **66,3 %** en BARAJADO y el **100 %** en el control α=0. Acá la pregunta se puede hacer.

La respuesta tiene una forma que no esperábamos:

> **El mínimo no está en la cantidad de leña. Está en la relación entre lo que cada fuego recibe de
> los otros y lo que pierde por su cuenta.**

Da igual si arrancás con una brasa o con un tronco. Bajamos el tamaño inicial **quince órdenes de
magnitud** y el resultado no cambió **ni en un nodo, ni en un paso**. No hay piso en el tamaño.

Sí hay un piso, nítido, en la **relación**. Y —esto es lo interesante— **ese piso depende de cómo
estén ordenados los fuegos**: barajando quién está al lado de quién (mismos fuegos, misma energía,
mismo acoplamiento) el piso sube, y en 6 de 10 configuraciones **sube a infinito**: barajado, no se
sostiene con ninguna cantidad de acoplamiento.

**Y la advertencia de siempre.** El piso **es en buena parte una cuenta, no un descubrimiento**: sale
de despejar tres líneas del propio motor. Está reportado como tal en §2.4.

---

## 1 · κ_V — **NO ES MEDIBLE en CG002.** Y el motivo no es falta de datos.

**Canon:** `A_sys-env ≥ κ_V > 0` — *"el acoplamiento con el entorno no puede caer a cero sin ruptura."*

Para medir eso hace falta que el modelo **entregue** una partición sistema/entorno. CG002 no la tiene:

1. **El estado es una lista cerrada de N nodos** (`s`, `omega`, `w_link`, `f_signed`). No hay ningún
   grado de libertad que no sea un nodo.
2. **El único término no-nodal es MU** (`s ← (1−MU)·s`). Es un sumidero **sin estado y sin
   retroacción**: no acumula lo que se lleva, no devuelve nada, no cambia. Medir `κ_V = MU` sería
   leer de vuelta un parámetro de entrada — la tautología más pura posible.
3. **Cualquier otra partición hay que dibujarla a mano.** Y entonces el número es el corte, no el sistema.

Esto último se convirtió en dato (`kappaV_sustrato_cg002.py/.json`): las **254 biparticiones** de los
8 nodos, en 10 corridas REAL completas, con `A_sys-env` = flujo de acoplamiento que cruza el corte /
flujo total.

| | resultado |
|---|---|
| corridas con **alguna** bipartición de A_sys-env **exactamente 0** | **10 de 10** |
| fracción media de biparticiones con A = 0 | **6,9 %** (≈17 de 254 por corrida) |
| **corridas rotas** (0 nodos vivos) | **0 de 10** |
| rango de A dentro de una misma corrida | 0,000 – 0,68 (mediana ≈ 0,52) |

**Lectura:** en toda corrida sana existen cortes con acoplamiento cruzado exactamente cero, y el
sistema **no se rompe** (esos cortes aíslan un nodo muerto o desconectado). `A_sys-env ≥ κ_V > 0` **no
es falsable** en CG002: el valor de A es una propiedad del corte que uno eligió.

> **κ_V queda como "no aplicable a este nivel de descripción"**, igual que κ_O, κ_LF y κ_H.
> Eso es un resultado, no una deuda. **No se forzó.**

**Dónde sí tiene sustrato:** en Phantom hay una pareja sistema/entorno genuina — sumidero ↔ gas, con
estado a los dos lados y transferencia bidireccional (acreción). Por eso allí sí se pudo definir
(CS078). Y allí ya falló: ρ=+0,823 / r=+0,810 con la masa (§5 de la validación del 13-ago; ρ=0,902 en
la medición previa), z=1,33 contra el NULL original y **z=−0,54 contra NULL-3**.
**El sustrato de κ_V existe en el proyecto, pero no en CG002; y donde existe, el observable ya cayó.**

---

## 2 · κ_P — **SÍ hay un piso. No está en S: está en el cociente acoplamiento/decaimiento.**

### 2.1 · Primero, el test de tautología: **no hay piso EN S** (barrido C, 15 décadas)

Se barrió S0 de 10⁻¹² a 10³ en dos modos. `frac_vivo` = fracción de semillas con ≥1 nodo vivo;
`k_micro` = pasos que duró la corrida; `n_vivos` = nodos vivos al final (medias sobre 10 semillas).

| S0 | REAL, pisos **escalados** | REAL, pisos **absolutos** |
|---|---|---|
| 10⁻¹² | 1,00 · k=500,0 · 5,00 | **0,00 · k=0,0 · 0,00** |
| 10⁻⁹ | 1,00 · k=500,0 · 5,00 | **0,00 · k=0,0 · 0,00** |
| 10⁻⁶ | 1,00 · k=500,0 · 5,00 | **0,00 · k=0,0 · 0,00** |
| 10⁻⁵ | 1,00 · k=500,0 · 5,00 | 1,00 · k=500,0 · 5,00 |
| 1 | 1,00 · k=500,0 · 5,00 | 1,00 · k=500,0 · 5,00 |
| 10³ | 1,00 · k=500,0 · 5,00 | 1,00 · k=500,0 · 5,00 |

En modo escalado el resultado es **idéntico hasta el último dígito en las 16 escalas y en los dos
brazos** (BARAJADO: 0,60 · k=676,9 · 2,90 en las 16, sin variación). En modo absoluto hay un escalón,
**y cae exactamente en S0 = KAPPA_S = 10⁻⁶**: por debajo, el nodo ya nace muerto (`s > KAPPA_S` es
falso en el paso 0) y la corrida dura **cero pasos**.

> **El único "piso en S" que aparece es la constante KAPPA_S leída de vuelta.** Es la tautología en
> estado puro. Y desaparece en cuanto se escalan las unidades, porque el mapa es homogéneo de grado 1.
> **κ_P = inf(S_viable) = 0** en CG002: cota inferior trivial, sin contenido.

### 2.2 · El piso real: la curva λ(α) completa, 7 décadas (barrido A)

λ = tasa de crecimiento por paso, medida sin pisos puestos a mano (operador puro, renormalizado).
λ<0 = el sistema se apaga; λ>0 = se sostiene. **REAL, float64, 10 semillas:**

| α | λ medio | frac λ>0 | frac "vivo a k=1500" |
|---|---|---|---|
| 10⁻⁶ … 10⁻⁴ | −0,010050 | 0,00 | 0,00 |
| 5,6×10⁻⁴ | −0,010048 | 0,00 | **1,00** ← el reloj ya miente acá |
| 10⁻² | −0,009521 | 0,00 | 1,00 |
| 0,0431 | −0,004379 | 0,00 | 1,00 |
| 0,0514 | −0,003145 | 0,10 | 1,00 |
| 0,0730 | −0,000094 | 0,20 | 1,00 |
| **0,0870** | **+0,001821** | **0,90** | 1,00 |
| 0,1038 | +0,004095 | **1,00** | 1,00 |
| 1 | +0,118320 | 1,00 | 1,00 |
| 10 | +0,846571 | 1,00 | 1,00 |

**La transición es nítida:** 0,00 → 1,00 entre α=0,043 y α=0,104 (factor 2,4), y **por semilla es un
salto exacto** (por eso se bisecta: §2.3). Por debajo del umbral λ tiende a −MU/(1−MU) = −0,010050
exactamente, que es el decaimiento puro: el acoplamiento no aporta nada.

**Y el hallazgo colateral más importante:** la columna "vivo a k=1500" vale **1,00 desde α=5,6×10⁻⁴**,
donde λ es negativo y el sistema se está apagando. **El criterio de horizonte finito da "vivo" en dos
décadas y media donde el sistema muere.** Es el mismo error de Phantom con otra ropa: mide cuánto
tarda en cruzar el umbral, no si se sostiene. Panel (b) de la figura.

### 2.3 · El umbral por bisección, en **tres precisiones** (guarda F1-3)

Bisección del signo de λ, 45 iteraciones (30 en mpmath), sin binarizar por ningún umbral nuevo:

| semilla | λ_max(G) | fórmula §0.2 | **float64** | **float80** | mpmath-50 (k=400) | float64 (k=400) | BARAJADO f64 | BARAJADO f80 |
|---|---|---|---|---|---|---|---|---|
| 1 | 4,0000 | 0,050505 | **0,0771956** | **0,0771956** | 0,078034 | 0,094793 | **∞** | **∞** |
| 2 | 4,4142 | 0,045766 | **0,0837458** | **0,0837458** | 0,089992 | 0,121282 | **∞** | **∞** |
| 3 | 4,5811 | 0,044098 | **0,0784867** | **0,0784867** | 0,092895 | 0,124246 | **∞** | **∞** |
| 4 | 5,2361 | 0,038582 | **0,0444116** | **0,0444116** | 0,044949 | 0,049347 | 0,0903049 | 0,0903049 |
| 5 | 3,7071 | 0,054495 | **0,0772403** | **0,0772403** | 0,078794 | 0,094932 | **∞** | **∞** |

> ### **Desplazamiento relativo float64 → float80: 0,000×10⁰ — EXACTAMENTE CERO.**
> Con ε de máquina 2048 veces más chico (2,22×10⁻¹⁶ → 1,08×10⁻¹⁹), **el umbral no se movió en
> ninguna cifra de ninguna semilla, ni en REAL ni en BARAJADO.** Según el pre-registro §0.4, eso
> descarta el artefacto numérico: **el piso no es de la máquina.**

**Sobre mpmath:** difiere hasta un 18 %, pero **no por precisión**: esa columna corre a k=400 y
promedia una ventana distinta. La columna de control "float64 (k=400)" muestra que a horizonte 400
el float64 da 0,095 y el mpmath 0,078 — o sea, la dispersión es del **estimador de λ cerca de λ=0**
(donde converge lento), no de la aritmética. La comparación limpia, mismo código y misma ventana, es
float64 vs float80: **cero**.

### 2.4 · ¿Es un descubrimiento o una cuenta? — **Es en buena parte una cuenta, y hay que decirlo.**

La fórmula pre-registrada `α_c = 0,20202/λ_max(G)` predice 0,039–0,054; lo medido es 0,044–0,084.
**No coincide, y la desviación (hasta 45 %) es informativa:** el cociente de Rayleigh sin restricción
usa el autovector dominante de G, que tiene componentes negativas — pero el estado del sistema es
`x = √s ≥ 0`. El umbral real lo fija el **máximo de xᵀGx sobre el cono no negativo**, que es menor que
λ_max(G), y por eso α_c medido es **mayor** que la fórmula. La fórmula es una **cota inferior** de α_c,
no una identidad.

> **Honestidad requerida:** el piso de κ_P es una propiedad **analítica** del motor (un cambio de signo
> de un cociente de Rayleigh restringido), no un fenómeno emergente. Que la simulación lo confirme con
> desplazamiento cero entre precisiones **no lo vuelve un hallazgo empírico: lo vuelve una verificación
> de que el código hace lo que dice el papel.** Se reporta como tal.

### 2.5 · El piso depende de la ESTRUCTURA, no sólo de la energía

Éste es el resultado que **no** es una cuenta trivial, porque contrasta dos brazos con la misma
energía, la misma α, el mismo N y **el mismo multiset de firmas ω**:

| | REAL | BARAJADO |
|---|---|---|
| α_c (5 semillas bisectadas) | 0,0444 – 0,0837 | **∞ en 4 de 5**; 0,0903 en la restante |
| α_c predicho por ḡ (10 semillas, §2.6) | 0,0386 – 0,0545 | **∞ en 6 de 10**; 0,090 – 0,352 en las otras 4 |
| frac λ>0 a α=1 | **1,00** | 0,30 |
| frac λ>0 a α=10 | **1,00** | 0,20 |
| λ medio a α=1 | +0,118320 | +0,005544 |
| nodos muertos (producción 13-ago) | 38,7 % | 66,3 % |

**Barajar quién se relaciona con quién no sube el piso: en la mayoría de las configuraciones lo manda
a infinito.** Con la compatibilidad media ḡ<0 (6 de 10 semillas), el sistema barajado **no se sostiene
con ningún acoplamiento**, ni con α=10. No es que le cueste más: no puede.

### 2.6 · Tabla estructural por semilla (cuenta cerrada, sin simular)

| semilla | ω | λ_max(G) | ḡ | α_c REAL | α_c BARAJADO |
|---|---|---|---|---|---|
| 1 | 3 4 6 7 0 1 6 7 | 4,000 | −0,036 | 0,05051 | **∞** |
| 2 | 6 2 0 2 3 6 3 0 | 4,414 | −0,101 | 0,04577 | **∞** |
| 3 | 6 0 1 1 1 6 6 4 | 4,581 | −0,049 | 0,04410 | **∞** |
| 4 | 5 7 7 4 7 7 7 0 | 5,236 | +0,321 | 0,03858 | 0,0898 |
| 5 | 5 6 0 6 3 4 5 2 | 3,707 | −0,011 | 0,05450 | **∞** |
| 6 | 3 4 4 2 7 2 5 2 | 3,707 | +0,082 | 0,05450 | 0,3524 |
| 7 | 7 5 5 7 4 6 6 1 | 3,707 | +0,162 | 0,05450 | 0,1782 |
| 8 | 5 2 1 7 1 2 5 6 | 5,121 | −0,132 | 0,03945 | **∞** |
| 9 | 3 6 7 2 0 4 5 6 | 3,707 | −0,082 | 0,05450 | **∞** |
| 10 | 6 7 2 1 6 6 4 1 | 4,581 | −0,091 | 0,04410 | **∞** |

### 2.7 · El umbral está en el COCIENTE, no en α (barrido A′, MU en 4 décadas)

Con α fijo = 0,05 se barrió MU de 10⁻⁵ a 10⁻¹. El umbral aparece donde predice el cociente:

| MU | λ medio REAL | frac λ>0 REAL | λ medio BARAJADO | frac λ>0 BARAJADO |
|---|---|---|---|---|
| 10⁻⁵ | +0,006695 | 1,00 | +0,000098 | 0,30 |
| 10⁻³ | +0,005704 | 1,00 | −0,000893 | 0,30 |
| 6,3×10⁻³ | +0,000375 | **0,60** | −0,006222 | 0,00 |
| 10⁻² | −0,003345 | **0,10** | −0,009943 | 0,00 |
| 10⁻¹ | −0,098656 | 0,00 | −0,105253 | 0,00 |

**El mismo piso se cruza moviendo el decaimiento en vez del acoplamiento**, y BARAJADO lo cruza
**tres veces más temprano** (entre 10⁻⁴ y 6×10⁻³, contra 6×10⁻³–10⁻² de REAL). Confirma que el
observable con contenido es el cociente, no ninguna de las dos perillas por separado.

---

## 3 · κ_Δ — **Δ_struct no sirve (es la masa). El único piso que sobrevive es de tamaño ≈1 letra del alfabeto.**

### 3.1 · El estimador obvio está muerto antes de empezar

Aplicado a la producción CG002 existente (`cg002_produccion_series.csv`, 90 corridas REAL):

| candidato | contra | Spearman | Pearson | lectura |
|---|---|---|---|---|
| **κ_Δ = Δ_struct** | `S_final_suma` | **+0,999** | **+1,000** | 🔴 es la misma cantidad |
| **κ_Δ = Δ_struct** | `S_final_max` | **+0,998** | **+1,000** | 🔴 es la misma cantidad |
| **κ_Δ = Δ_struct** | `n_aristas` | +0,937 | +0,242 | 🔴 |

**Δ_struct reproduce exactamente el fracaso de Phantom** (allí r=+0,997 con la masa). Y es algebraico:
`Δ_struct = Σ_k(Σ|f| + Σ|Δs| + Δ_topo)`, y `|f|`,`|Δs|` escalan con S. **En un mapa homogéneo de grado 1,
toda suma de magnitudes es la masa disfrazada.** Descartado antes de mirar ningún z.

### 3.2 · El estimador adimensional: la mínima diferencia de fase δ que el sistema puede operar

Se inyecta δ en la fase de **un** nodo y se compara contra la corrida sin perturbar (barrido B, 20
décadas; refinado en `kappaDelta_refino.py`, que **corrige un defecto del primer barrido**: la
perturbación se sumaba en float64 antes de entrar al motor, así que float80 y mpmath heredaban el piso
de float64 y la guarda de precisión era falsa. En el refino ω y δ se construyen en la precisión de
destino).

**(a) "Distinguible" (cambia algún observable continuo) — sigue a la máquina, no hay piso:**

| precisión | ε de máquina | ulp(ω≈7) | δ mínimo distinguible (5 semillas) |
|---|---|---|---|
| float64 | 2,22×10⁻¹⁶ | 8,9×10⁻¹⁶ | 10⁻¹⁵ (4/5) · 10⁻¹⁴ (1/5) |
| float80 | 1,08×10⁻¹⁹ | 4,3×10⁻¹⁹ | **10⁻¹⁷ · 10⁻¹⁶ · 10⁻¹⁵ · 10⁻¹⁴** |

El piso de float64 **coincide con su propio ulp**, y al pasar a float80 **baja hasta 100×**.
Según el pre-registro §0.4: **artefacto numérico. Para "distinguible" no hay κ_Δ** — sólo el sustrato.
*(Limitación declarada: el brazo mpmath da 10⁻⁸–10⁻¹⁰ y **no se mueve** entre 30 y 60 dígitos, pero ese
brazo compara una salida truncada a float64, así que su piso es del lector, no de la aritmética. No se
usa para decidir.)*

**(b) "Operable" (cambia el conjunto de aristas, el nº de vivos o τ) — hay piso, y no es de la máquina
ni de las constantes a mano:**

| semilla | δ mínimo operable con EPS_TAU=10⁻¹² | con 10⁻⁴ | con 10⁻¹ |
|---|---|---|---|
| 1 | **1,72** | 1,72 | 1,72 |
| 2 | **>2** (no ocurre) | >2 | >2 |
| 3 | **0,80** | 0,80 | 0,80 |
| 4 | **0,14** | 0,14 | 0,14 |
| 5 | **1,00** | 1,00 | 1,00 |

> **El piso operable es idéntico dígito a dígito moviendo EPS_TAU once órdenes de magnitud**, e
> idéntico en float64 y float80. **No es la constante puesta a mano** (que era la hipótesis de
> tautología del pre-registro) **ni el ε de máquina.** Es de tamaño **≈1 unidad de ω**, o sea
> **una letra del alfabeto ℤ_K** — el mismo orden que el cuanto del alfabeto, 1−cos(2π/8) = **0,2929**.

**Lectura provisoria:** κ_Δ, si existe en CG002, **no es un número universal: es la granularidad del
alfabeto de firmas**. Depende de K (→0 como 2π²/K² al agrandar el alfabeto) y varía por configuración
(0,14 a >2 en cinco semillas). **Es el resultado que más trabajo pide y el que menos cerrado está.**

---

## 4 · GUARDAS (las cinco, reportadas pase lo que pase)

### 4.1 · Tautologías — **el brazo ALFA_0 no informa sobre el piso. Es un reloj.**

Con α=0 el motor se reduce a `s(k) = S0·(1−MU)^k`. El nodo cruza KAPPA_S en

> k\* = ln(S0/KAPPA_S)/(−ln(1−MU)) = ln(10⁶)/0,010050 = **1374,6 pasos**

y el presupuesto del protocolo es `K_MAX_MIN = 1500`. **El control G muere por aritmética pura, con un
margen del 8 %.** Con presupuesto 1300 —cifra igual de arbitraria— el mismo control habría dado
"sobrevive" y V1/V2 se habrían dado vuelta. El propio código lo dice sin sacar la consecuencia:
`K_MAX_MIN = 1500  # μ no extingue antes ~1400 pasos`.

**Consecuencia metodológica:** el contraste que informa sobre estructura es **REAL vs BARAJADO**, no
REAL vs ALFA_0. ALFA_0 se reporta porque está pre-registrado, no porque diga algo del piso.
*(En el barrido A, ALFA_0: λ = −0,010050 y frac λ>0 = 0,00 — exactamente el decaimiento puro.)*

### 4.2 · Identidades algebraicas

**Sobre la producción existente** (§3.1): `Δ_struct` = la masa (Pearson +1,000). **Descartado.**

**Sobre λ, el observable de este informe** (1585 filas del barrido A, brazo REAL, float64):

| λ contra | Spearman | Pearson |
|---|---|---|
| `alpha` | +0,938 | +0,980 |
| `S_final_max` | **+0,952** | +0,333 |
| `delta_struct` | **+0,940** | +0,333 |
| `tau_final` | +0,586 | +0,164 |
| `n_aristas` | +0,458 | +0,243 |
| `n_vivos_fin` | +0,064 | +0,059 |
| `k_micro` | −0,593 | −0,192 |

**Sí, λ está ligado monótonamente a la masa final (ρ=0,95) — y tiene que estarlo:** `S_final ≈ e^{λk}`.
Es la *misma* cantidad en escala logarítmica, no una coincidencia sospechosa. Por eso **el observable
que se usa no es λ sino el CAMBIO DE SIGNO de λ**, que es invariante bajo cualquier reescalado
monótono: ningún cambio de unidades, ni la exponencial, puede mover un cero. Ésa es la diferencia con
κ_Δ en Phantom, donde el número reportado *era* la masa reescalada y el z heredaba su significado.

*(`α_c` vs `λ_max(G)`: Pearson −0,742, Spearman −0,200 con n=5 — negativo como predice la fórmula, pero
débil, por lo dicho en §2.4: el umbral lo fija el cono no negativo, no λ_max.)*

### 4.3 · Piso de ruido

10 semillas por brazo y por punto. El p más chico alcanzable en una 2×2 con separación total (10-0 /
0-10) es **p = 5,41×10⁻⁶**. Ninguna fracción de este informe puede tener un p menor. **No se binarizó
nada por umbrales nuevos:** los α_c salen de bisectar el **signo de λ**, que es la frontera natural
entre crecer y decaer, no una convención.

### 4.4 · El barajado no es isomorfo al real

Repetida sobre el núcleo nuevo (10 semillas, k=300): fracción de semillas con `S_final` ordenado
idéntico entre REAL y BARAJADO = **0,0**. (Barajar **una sola vez** al inicio sí es un renombre de
nodos y da identidad exacta — por eso se baraja **por paso**, como el 13-ago.)

### 4.5 · El núcleo replica el motor original

30 comparaciones (N∈{4,6,8} × 5 semillas × k∈{25,60}) contra `cg002_acoplamiento.correr`:
error relativo máximo en `S_final` = **5,14×10⁻¹⁶** (una ulp de float64), en `delta_struct` =
**3,55×10⁻¹⁶**, y **100 % de coincidencia** en los tres observables discretos (τ, vivos, aristas).
El núcleo no cambia el modelo; sólo lo deja correr en otra precisión.

**Advertencia de horizonte (no es un defecto, es física del modelo):** por encima del umbral la
dinámica es exponencialmente inestable, así que a horizonte largo dos sumas de los mismos términos en
distinto orden divergen. **Por eso ningún observable de estado detallado es reproducible a k grande —
y por eso el observable elegido es la TASA, no el estado.** La tasa es lo único que sobrevive.

---

## 5 · RESPUESTAS DIRECTAS A LAS CUATRO PREGUNTAS DEL ENCARGO

**1. ¿κ_V es medible acá? — NO, y por una razón estructural, no por falta de datos.**
CG002 no tiene entorno: el estado es una lista cerrada de nodos y el único término no-nodal (MU) es un
sumidero sin estado ni retroacción. Toda partición sistema/entorno hay que imponerla, y entonces
A_sys-env es el corte, no el sistema: en **10 de 10** corridas sanas existen biparticiones con
A_sys-env **exactamente 0** (6,9 % de las 254) **sin ninguna ruptura**. → **no aplicable a este nivel
de descripción**, como κ_O/κ_LF/κ_H. Su sustrato real está en Phantom (sumidero↔gas), donde ya cayó.

**2. ¿Aparece un piso para κ_P? — SÍ, pero no donde el canon lo pone.**
`κ_P = inf(S_viable) = 0`: la escala de S no fija nada (15 décadas, resultado idéntico dígito a
dígito), y el único "piso en S" que aparece es la constante KAPPA_S leída de vuelta. El piso real está
en el **cociente acoplamiento/decaimiento**, `α_c ≈ 0,044–0,084` para N=8, y es **nítido**: por debajo,
λ = −MU/(1−MU) exacto (el acoplamiento no aporta nada); por encima, λ>0 en el 100 % de las semillas.

**3. ¿Sobrevive el cambio de precisión? — SÍ para κ_P, NO para la lectura "distinguible" de κ_Δ.**
- κ_P: desplazamiento float64→float80 **exactamente 0,000e+00** en las 5 semillas y los 2 brazos, con
  ε de máquina 2048× más chico. **No es artefacto numérico.**
- κ_Δ "distinguible": el piso **es** el ulp del sustrato (10⁻¹⁵ en float64 ≈ su ulp; baja hasta 10⁻¹⁷
  en float80). **Artefacto.** No hay κ_Δ por esa lectura.
- κ_Δ "operable": **0,14 / 0,80 / 1,00 / 1,72 / >2** según la semilla, **idéntico dígito a dígito** con
  EPS_TAU movido **once órdenes de magnitud** y en las dos precisiones. **Ni máquina ni constante a
  mano.** Es el único candidato vivo, y su tamaño es ≈1 letra del alfabeto ℤ_K.

**4. ¿El piso depende de la estructura o sólo de la energía? — DE LA ESTRUCTURA, y de forma brutal.**
Mismo N, misma α, misma energía, **mismo multiset de firmas**: lo único que cambia es quién se
relaciona con quién. α_c pasa de 0,044–0,084 (REAL) a **infinito en 4 de 5 semillas bisectadas** (6 de
10 por la cuenta cerrada): barajado, el sistema **no se sostiene con ningún acoplamiento, ni α=10**.
En las semillas donde sí puede, el piso está 2,0×–6,5× más arriba. A α=1, λ_REAL = +0,1183 contra
λ_BARAJADO = +0,0055 (21×).

---

## 6 · LO QUE QUEDA ABIERTO (no se cerró nada)

1. **α_c medido ≠ fórmula (hasta 45 %).** La explicación propuesta —el máximo de xᵀGx sobre el cono no
   negativo, no λ_max(G)— **no se verificó numéricamente**. Es la primera tarea de continuación:
   calcular el Rayleigh restringido y ver si predice α_c a 3-4 cifras. Si lo hace, κ_P en CG002 es
   **enteramente analítico** y hay que decirlo con todas las letras.
2. **κ_Δ operable varía 0,14 → >2 entre semillas** y no se entiende qué configuración lo fija. La
   hipótesis (distancia al cambio de signo de la compatibilidad) se codificó mal y dio 0 en las 5
   semillas: **hay que rehacerla**.
3. **El brazo mpmath de κ_Δ compara una salida truncada a float64.** Hay que reescribir la comparación
   en aritmética mp para que la guarda de precisión sea limpia en las tres precisiones, no en dos.
4. **Sólo N=8 y θ_CP=0.** Falta ver si α_c escala con N como predice λ_max ∈ [N/2−1, N−1], y si θ_CP≠0
   (el grafo orientado) mueve el piso.
5. **BARAJADO como control:** funciona, pero conviene un tercer control que preserve ḡ y destruya sólo
   λ_max, para separar "estructura" de "compatibilidad media".

---

## 7 · ARCHIVOS

| archivo | qué es |
|---|---|
| `kappas_mortalidad_barrido.py` | barridos A/A′/B/C, núcleo multi-precisión, guardas |
| `kappas_mortalidad_curva_alpha.csv` | 1585 filas: λ(α) y λ(MU) por brazo, semilla y precisión |
| `kappas_mortalidad_curva_S0.csv` | 640 filas: S0 en 15 décadas, modos absoluto/escalado |
| `kappas_mortalidad_curva_delta.csv` | 883 filas: δ en 20 décadas × 3 precisiones × 3 EPS_TAU |
| `kappas_mortalidad_resumen.json` | umbrales α_c, κ_Δ, y las cinco guardas |
| `kappas_mortalidad_curvas.png` | 4 paneles: λ(α) · reloj vs asintótico · S0 · δ. **El panel (d) es el barrido δ ANTERIOR al arreglo de §3.2** (δ inyectado en float64 en los tres brazos); los números válidos de κ_Δ son los de las tablas §3.2, no los de ese panel |
| `kappaV_sustrato_cg002.py/.json` | las 254 biparticiones, 10 corridas |
| `kappaDelta_refino.py/.json` | κ_Δ con δ inyectado en la precisión de destino + rejilla fina |
| `kappas_mortalidad_lectura.py` | lectura read-only de los CSV (tablas de este informe) |

---

⚠️ **NO ES UN CIERRE.** Se reportan números y curvas. Ningún veredicto es válido sin el director.
