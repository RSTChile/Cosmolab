# PROTOCOLO CF-4 — PRE-REGISTRO
## "¿El 99% de la masa es energía de ligadura?" (confinamiento, época correcta, pre-átomo)

**Escrito:** 2026-07-23 17:22 (hora local, antes de escribir o correr `CF4_confinamiento.py`).
**Autor de la corrida:** agente CF-4 (Claude), bajo dirección de Alexis López Tapia (CS).
**Batería:** CF (Cosmo-Física) — experimento CF-4, prioridad más alta de la batería ahora mismo.

Este documento se congela ANTES de escribir el motor de simulación. Cualquier
desviación del código respecto a lo aquí escrito debe reportarse explícitamente,
no corregirse en silencio. El criterio de PASS/FAIL no se toca después de ver
resultados.

---

## 0. Por qué existe CF-4 (resumen del error que corrige)

El Modelo Estándar dice que ~99% de la masa del protón nace en el
**confinamiento** (transición QCD, ~10⁻⁵ s): cuando los quarks quedan
atrapados en un cierre, la energía de ligadura ES la masa. Eso ocurre
MUCHO ANTES de que existan átomos (~10¹¹ años antes).

El motor v1–v6 de `suite_epocas_masa` (ver
`HALLAZGO_ABIERTO_etapa7_v6_masa_es_linaje_CS.md`) medía "masa" **después**
del átomo, con criterios E3 (`K_MIN/K_MAX/F_CORE/COHESION` de
`components_strict`), y en v6 la construyó con las mismas variables de
linaje (`co_member_score`, `n_long_co_pairs`, `fusion_events`) que luego
la juzgaban — circularidad. **Eso está prohibido en CF-4.**

CF-4 mide la masa en su época física correcta: durante el confinamiento,
en la fase caliente, **antes** de que se apliquen criterios de átomo.
No se usa `components_strict`, ni `K_MIN/K_MAX/F_CORE/COHESION`, ni
tracking de linaje/persistencia de ningún tipo.

---

## 1. Pregunta y diseño (fijado por el director, no se cambia aquí)

Cuando los "quarks" (nodos del campo Φ) se confinan en un cierre (componente
conectado por enlaces `ar`/`ad` que sobreviven el corte), ¿el cierre "pesa"
mucho más que la suma de sus partes libres — porque la masa es la energía
que cuesta mantenerlos juntos, no la materia de los nodos?

**Barrido:** intensidad de confinamiento (`H_TOPO`) × tamaño de cierre `k`
(MEDIDO, nunca impuesto) × ≥8 semillas (usamos las 10 estándar del proyecto).

---

## 2. Física reutilizada (de `suite_epocas_masa_v6_mass_linaje.py`, sin editar ese archivo)

### 2.1 Evolución del campo Φ (reutilizada de las líneas ~430-443 de v6)

Igual forma funcional que v6, **con una simplificación explícita y documentada**:
CF-4 no modela el campo `phi` (medio de mezcla) porque ese campo solo existe
en v6 para la lógica de "átomo" (lado/f_core), que aquí está prohibida. Por
tanto el término de acoplamiento a densidad de `phi` (`G_RHO·(rho_hat−1)`)
se omite; `r_field` depende solo de la temperatura:

```
r_field = R0 * (Tnorm - TC)                      # [SIMPLIFICACIÓN vs v6: sin G_RHO·(rho_hat-1), no hay campo phi en CF-4]
lap     = roll(Phi,-1,x) + roll(Phi,1,x) + roll(Phi,-1,y) + roll(Phi,1,y) - 4*Phi   # Laplaciano completo del toro, igual que v6
dV      = 2*r_field*Phi + 4*U*Phi**3             # idéntico a v6 (potencial Φ⁴)
D_eff   = D_PHI * rho_hat_c                       # idéntico a v6 (rho_hat_c = 1/a^3, cosmológico, no depende de phi)
sig     = SIGMA0 * sqrt(max(Tnorm,1e-6) * max(rho_hat_c,1e-12))   # idéntico a v6
Phi    += DT_PHI * (-dV + D_eff*lap) + sig * ruido_normal          # idéntico a v6
```

Cosmología idéntica a v6: `a = exp(H_EXP*tg)`, `Tnorm = exp(-H_EXP*tg)`,
`tg = step/(pasos-1)`, `rho_c = RHO0/a**3`, `rho_hat_c = rho_c/RHO0`,
`frozen = (Tnorm < FREEZE_TNORM) or (rho_c < RHO_FREEZE)`.

### 2.2 Corte de enlaces (confinamiento) — reutilizado de `weighted_cut()` de v6

Se reimplementa (no se importa del archivo v6) la misma lógica: cada paso
se cortan `nc = round(H_TOPO * sqrt(Tnorm+1e-12) * enlaces_totales)` enlaces,
con sesgo `(1 - borde_Φ + 1e-3) ** ALPHA_CUT` hacia zonas de campo débil
(igual fórmula que v6). `ALPHA_CUT = 2.5` fijo (valor por defecto de v6, no
tocado). Los enlaces `ar`/`ad` empiezan **todos True** (todo conectado) y
solo se cortan, nunca se reconectan — igual que v6.

**Parámetro barrido = intensidad de confinamiento = `H_TOPO`.**
Semántica: **`H_TOPO` MÁS CHICO → menos cortes por paso → los enlaces
sobreviven más → confinamiento MÁS FUERTE** (más resistencia a separar).
`H_TOPO` MÁS GRANDE → más cortes → confinamiento MÁS DÉBIL.

Rango barrido (8 valores, decidido antes de correr, cubre débil↔fuerte
alrededor del default de v6 que era 0.01):

```
H_TOPO ∈ {0.002, 0.004, 0.007, 0.01, 0.02, 0.04, 0.07, 0.10}
```

### 2.3 Cierres (closures) — algoritmo reutilizado de `components_strict()` de v6, SIN el filtro de átomo

Se reimplementa el esqueleto algorítmico de v6 (unión de nodos vecinos por
los arrays de enlace `ar`/`ad`, BFS/flood-fill) pero:

- **NO** se usa el criterio de "mismo lado" (`phi >= media`) porque no existe
  campo `phi` en CF-4.
- **NO** se aplican `K_MIN/K_MAX/F_CORE/COHESION/PERSIST_STEPS/VEV_POST_MIN/
  PHI_CORE_THR` — esos son criterios de ÁTOMO (etapa posterior, prohibidos
  aquí por diseño explícito del director).
- Un "cierre" = simplemente un componente conexo del grafo de enlaces
  `ar`/`ad` vivos en ese paso. Tamaño `k` = número de nodos del componente.
  **`k` emerge del barrido, nunca se impone** (T0).

### 2.4 Ventana de medición temporal

Se mide en cada paso con `frozen == False` (fase caliente/pre-freeze), cada
5 pasos (`MEASURE_STRIDE = 5`, decidido antes de correr, por costo
computacional — no afecta el criterio de PASS, solo la densidad de muestreo).
No se mide nunca después de `frozen == True`. No hay noción de "átomo" en
todo CF-4.

---

## 3. Observable de masa (núcleo de CF-4)

Para cada cierre (componente conexo) con nodos `{i}` y enlaces internos
`{(i,j)}` (los pares de nodos del cierre unidos por un `ar`/`ad` vivo):

### m₁ — "masa de constituyentes libres" (sin ligadura)

```
m1(cierre) = Σ_{i ∈ cierre} V(Φ_i) = Σ_i [ r_field * Φ_i**2 + U * Φ_i**4 ]
```
(el mismo potencial `V(Φ)` que ya define `dV` en la dinámica, evaluado nodo
por nodo COMO SI cada nodo estuviera aislado — sin el término de enlace).

### m₂ — "energía de ligadura" = observable de masa de CF-4

```
m2(cierre) = Σ_{(i,j) enlace interno del cierre} D_eff * (Φ_i - Φ_j)**2
```
Esta es exactamente la energía que el término `D_eff·lap` de la dinámica
ya representa (el Laplaciano es `-∂E_acople/∂Φ` para
`E_acople = (D_eff/2)·Σ_enlaces (Φ_i-Φ_j)²`; usamos el coeficiente sin el
factor 1/2 por enlace, ver nota de convención abajo — es una elección de
normalización, no un ajuste para forzar el resultado). Es la energía que
se PIERDE si se corta cada enlace del cierre, sumada — literalmente "el
trabajo para separarlo".

**Nota de convención:** se usa `D_eff·(Φ_i-Φ_j)²` (sin 1/2) por enlace,
tal como está escrito literalmente en el enunciado de la misión. Esto es
una constante multiplicativa global (factor 2) que no puede cambiar
ninguna comparación de razones (m2/m1, m2_REAL/m2_NULL) — no es un grado
de libertad que se ajuste después de ver resultados.

**Confirmación T2 explícita:** `m1` y `m2` se calculan ÚNICAMENTE a partir
de `Φ`, `r_field`, `U`, `D_eff` y la topología viva de `ar`/`ad` en ESE
paso. CF-4 no usa `co_member_score`, `n_long_co_pairs`, `fusion_events`,
tracking de persistencia (`match_persist`/`tracks`) ni ninguna cantidad de
"linaje" de v1-v6. No hay identidad de cierre a través del tiempo: cada
paso medido es una instantánea independiente.

### NULL — mismo cierre, enlaces internos barajados

Mismos nodos, mismo tamaño `k`, mismo número de enlaces internos `m_edges`
(preserva tamaño y grado agregado del cierre), pero las `m_edges` conexiones
se resamplean **uniformemente al azar entre pares de nodos del mismo
cierre** (no necesariamente vecinos de la retícula) — "topología de
confinamiento real" reemplazada por conexión aleatoria del mismo tamaño.
Se usa el mismo `Φ` (no se re-simula nada), solo cambia qué pares se suman:

```
m2_NULL(cierre) = Σ_{(i,j) ∈ pares aleatorios, |pares|=m_edges} D_eff * (Φ_i - Φ_j)**2
```

Se repite el barajado `R = 5` veces por cierre por paso medido (decidido
antes de correr, para reducir varianza de una sola muestra aleatoria) y se
promedia → `m2_NULL_mean`.

**Caso degenerado k=2:** un cierre de 2 nodos solo tiene 1 par posible, así
que el "barajado" reproduce el mismo enlace con probabilidad 1 (no hay
otra topología posible). Por eso **k=2 se excluye de la comparación
REAL vs NULL** (queda en el histograma de `k` y en las estadísticas de
m1/m2_REAL, pero no en `ratio_null`). Para `k≥3` el barajado sí tiene
soporte real (para k=3, hay 3 pares posibles y se eligen 2 → 1/3 de chance
de reproducir la topología real, el resto de las veces difiere).

---

## 4. Criterio de PASS (congelado ANTES de correr, sin mirar resultados)

Dos condiciones por instancia de cierre medido (k≥3 para la condición b;
k≥2 para la condición a, con k=1 excluido de ambas por no tener m2):

- **(a) "ligadura grande":** `ratio_lig = m2_REAL / max(m1, 1e-9) ≥ THRESH_BIG = 5.0`
- **(b) "REAL supera NULL":** `ratio_null = m2_REAL / max(m2_NULL_mean, 1e-9) ≥ THRESH_NULL = 1.25`

**Justificación de `THRESH_BIG = 5.0` (decidida ANTES de correr, por
razonamiento de órdenes de magnitud, no por mirar el resultado):** los
coeficientes de `m1` (`r_field ~ O(0.1–1)`, `U = 0.5`) son de orden 1,
mientras que `D_eff = D_PHI · rho_hat_c` con `D_PHI = 0.05` es a priori
PEQUEÑO — nada en el diseño garantiza que `m2 ≫ m1`; de hecho un observador
ingenuo esperaría lo contrario si el conteo de enlaces internos no compensa.
Fijar el umbral en 5× es una barra real, no un trivial "≥1", y dado que ni
`m1` ni `m2` fueron ajustados para el resultado (mismos coeficientes que
v6, no tocados), esto dejará casos a ambos lados del umbral según el
barrido (T5).

**Justificación de `THRESH_NULL = 1.25`:** valor estándar del proyecto
(igual a `BIND_VS_SHUFFLE_MIN`/`MUTUAL_VS_SHUFFLE_MIN` usados en v5/v6),
consistente con el resto de la batería, no elegido ad hoc para CF-4.

**PASS conjunto por instancia de cierre** = (a) AND (b).

**Tasa de PASS del experimento** (pre-registrada):
```
rate_pass = (# instancias cierre con k≥3 que cumplen (a) AND (b)) / (# instancias cierre con k≥3)
```
reportada global y por `H_TOPO`. **No se fija un umbral de "éxito global"
único con valor mágico tipo 99%** — se reporta la razón real como curva y
distribución, tal como exige el director. Como referencia de lectura (no
como pass/fail binario adicional) se usa el mismo `RATE_PASS = 0.55` que
usan v5/v6 del proyecto para decir si la mayoría del barrido confinado
cumple: si `rate_pass_global ≥ 0.55` se reporta como "mayoría PASS",
si no, se reporta el número real sin maquillar.

**T4 (el NULL debe caer de verdad):** se reporta explícitamente si
`mean(m2_NULL) < mean(m2_REAL)` a nivel agregado. Si el NULL NO cae, se
reporta así, sin ocultarlo ni ajustar el umbral después.

**T6 (nada por construcción):** ni `m1` ni `m2` ni el NULL están
garantizados de antemano a dar un resultado particular — dependen de la
trayectoria real de `Φ` y de qué enlaces sobreviven, que a su vez depende
de `H_TOPO`, semilla y ruido térmico. El experimento puede fallar.

---

## 5. Parámetros fijos (idénticos a los defaults de v6, no tocados por CF-4)

```
L = 28
pasos = 400
H_EXP = 6.0
TC = 0.55
R0 = 2.0
U = 0.5
D_PHI = 0.05
DT_PHI = 0.08
SIGMA0 = 0.10
RHO0 = 1.0
FREEZE_TNORM = 0.40
RHO_FREEZE = 0.05
ALPHA_CUT = 2.5
MEASURE_STRIDE = 5     # nuevo en CF-4, solo afecta densidad de muestreo
NULL_REPEATS = 5        # nuevo en CF-4, promedio de barajados por cierre/paso
```

## 6. Semillas

```
SEEDS = (7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321)
```
(10 semillas, set estándar del proyecto, ≥8 requeridas.)

## 7. Plan de ejecución

1. **Smoke** (validar mecánica, no decide PASS): `L=16, pasos=120`,
   `H_TOPO ∈ {0.005, 0.02, 0.08}`, semillas `{7, 42}` → 6 corridas.
   Verifica: cierres se forman, `k` varía, `m1`/`m2` no son NaN/0
   triviales, NULL difiere de REAL para al menos algún cierre, tiempo de
   ejecución razonable.
2. **Producción**: parámetros de la sección 5, `H_TOPO` (8 valores) ×
   `SEEDS` (10) = 80 corridas completas.
3. **Análisis**: agregación de `ratio_lig`, `ratio_null`, histograma de
   `k`, `rate_pass` global y por `H_TOPO`, verificación T4/T5/T6.
4. Reporte crudo a CS — **sin adjudicar** "la masa nace aquí".

---

## 7bis. ADENDA post-smoke, pre-producción (2026-07-23, mismo día, antes de correr producción)

Al correr el smoke (sección 7.1) se encontró que `m1` tal como estaba escrito
literalmente en la sección 3 (`Σ V(Φ_i)` absoluto) puede ser **negativo**
cuando `r_field < 0` (fase de simetría rota, `Tnorm < TC`): el mínimo del
potencial `V_min = -r_field²/(4U)` puede ser muy negativo, y sumar `V(Φ)`
absoluto da un "m1" sin piso, a veces negativo — una "masa de constituyente
libre" negativa no es una cantidad físicamente honesta (el potencial `V(Φ)`
tiene un aditivo libre — invariante de norma — mientras que `m2`, al
depender solo de diferencias `Φ_i-Φ_j`, no tiene esa ambigüedad).

**Corrección aplicada (antes de producción, no después de verla fallar):**
```
m1(cierre) = Σ_i [ V(Φ_i) - V_min(r_field) ],  V_min = -r_field²/(4U) si r_field<0, si no 0
```
Esta es la definición estándar de teoría de campos: la masa es la energía
de excitación SOBRE el vacío (siempre ≥0 por construcción del mínimo), no
el valor absoluto del potencial (que depende de una constante aditiva
arbitraria). Esto **no** es un ajuste de coeficiente para forzar el
resultado (T1): no cambia la forma funcional de `m2` ni del NULL, no toca
ningún umbral de PASS, y es una corrección de bien-definición aplicada
UNIFORMEMENTE (misma fórmula en todo el barrido) antes de mirar resultados
de producción — exactamente lo que el smoke test existe para cazar.

No se cambió ningún otro elemento del protocolo tras esta adenda.

---

## 8. Prohibiciones explícitas (recordatorio)

- No se usa `co_member_score`, `n_long_co_pairs`, `fusion_events`, ni
  tracking de linaje/persistencia de ningún tipo (T2).
- No se usa `components_strict` de v6 ni sus criterios de átomo
  (`K_MIN/K_MAX/F_CORE/COHESION/PERSIST_STEPS/VEV_POST_MIN/PHI_CORE_THR`).
- No se ajusta ningún coeficiente físico para forzar el resultado (T1) —
  todos los coeficientes de la dinámica de Φ y del corte son los defaults
  de v6, sin tocar.
- El criterio de PASS (sección 4) no se mueve después de correr (T3).
- No se edita `suite_epocas_masa_v6_mass_linaje.py` ni ningún archivo v1-v6
  ni `motor_1a7`. No se toca topología CG001/ANIMA/VSTCosmo. No se hacen
  commits de git.
