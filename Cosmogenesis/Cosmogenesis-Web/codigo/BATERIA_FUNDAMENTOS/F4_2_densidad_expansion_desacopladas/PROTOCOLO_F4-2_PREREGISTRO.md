# PROTOCOLO F4-2 — PRE-REGISTRO
## "Densidad vs expansión DESACOPLADAS: ¿efecto causal propio de ρ?" ★ (el clave, 24/24)

**Fecha/hora de escritura (UTC):** 2026-07-24T09:32:41Z — ANTES de escribir el motor
(`F4_2_motor.py`, que se creará después con mtime posterior a este archivo).
**Ejecutor:** CC (agente F4-2, batería paralela de 24). **Diseño de la batería:** CS.
**No se edita este archivo tras ver resultados (T3).** Si el criterio falla, se reporta
el FAIL con los números; no se cambia el juez.

---

## 1. LA PREGUNTA

En el modelo actual (CF2_estiramiento_motor.py, cs074_rcruz.py) la densidad ρ y la
expansión `a` están ATADAS: ρ = ρ0/a³, D = D0·(ρ/ρ0). Nunca se ha probado si ρ hace
algo POR SÍ SOLA sobre el destino de la diferencia (persistencia P), o si todo lo que
observamos es la expansión (corte topológico de aristas / aislamiento) disfrazada de
"densidad". F4-2 desacopla artificialmente las dos palancas para responder eso.

---

## 2. MOTOR BASE (heredado, sin tocar los archivos originales)

Se reutiliza el motor de campo continuo de `cs074_rcruz.py` (campo φ en anillo de N
celdas, difusión SOLO por aristas activas, expansión = corte de aristas Bernoulli(H)
por paso, persistencia P = autocorrelación_lag1_clip≥0 × razón_de_varianza). Ese motor
YA separaba "expansión" (corte de aristas = H) de "difusión" (D), pero no tenía ninguna
noción de densidad/dilución de D en el tiempo — eso es lo que F4-2 añade, como una
palanca ADICIONAL e independiente.

No se importa ni edita `cs074_rcruz.py` ni `CF2_estiramiento_motor.py`. Las funciones
`campo_inicial`, `paso_difusion`, `medir_D`, `medir_pasos_lavado`, `persistencia` se
REESCRIBEN idénticas (mismo álgebra, verificado) dentro de `F4_2_motor.py` porque F4-2
necesita inyectar el forzamiento dinámico (§4) y el `D(t)` variable (§3) dentro del
bucle de evolución, algo que las funciones originales no aceptan como parámetro.

---

## 3. CÓMO SE DESACOPLA (el corazón del diseño)

Se define una tasa nominal de expansión `H = min(r_target · D0, 1)` — igual que en
cs074_rcruz — donde `D0` es la difusividad MEDIDA (no puesta a mano) a la condición de
referencia (ver §6). `H` alimenta DOS mecanismos que en el modelo natural (ρ∝a⁻³)
estarían atados, y que aquí se activan/desactivan por separado:

- **Interruptor EXPANSIÓN (corte topológico):**
  - ON → cada arista activa se corta con probabilidad Bernoulli(H) en cada paso
    (idéntico a `paso_expansion` de cs074_rcruz).
  - OFF → ninguna arista se corta jamás; la topología queda 100% conectada todo el run.

- **Interruptor DILUCIÓN (densidad → difusividad):**
  - ON → `D(t) = D0 · exp(−3·H·t)`, con `t` = número de pasos transcurridos. Esto
    reproduce exactamente ρ(t)/ρ0 = a(t)⁻³ con a(t) = exp(H·t) — la MISMA ley de CF2
    (ρ∝a⁻³, n=3 fijo aquí; barrer n es tarea de F3-3/F4-6, no de F4-2) — pero usando
    el `H` de esta fila del barrido como si fuera la tasa de Hubble nominal, SIN
    necesidad de que las aristas se corten de verdad.
  - OFF → `D(t) ≡ D0` (constante — la densidad se mantiene artificialmente fija,
    "compensada").

**Las cuatro ramas de la grilla 2×2:**

| rama | expansión | dilución | qué significa físicamente |
|---|---|---|---|
| `00` | OFF | OFF | control base: nada pasa (ni corte ni dilución) |
| `a`  | ON  | OFF | expandir manteniendo ρ FIJA (compensación artificial) |
| `b`  | OFF | ON  | bajar ρ SIN expandir (a fijo, dilución "a mano") |
| `c`  | ON  | ON  | ambas juntas — el caso NATURAL, ρ∝a⁻³ |

`r = H/D0` es el eje de acople reportado (igual convención que cs074_rcruz). Se usa
`D0` de referencia (no el D(t) instantáneo) para que `r` sea comparable ENTRE ramas —
si se usara el D instantáneo, `r` cambiaría con el tiempo solo en la rama de dilución
y dejaría de ser un eje común.

---

## 4. PERTURBACIÓN DINÁMICA (T7 — lección de CF-2)

Cada paso de difusión recibe un forzamiento aditivo gaussiano de amplitud FIJA
`SIGMA_DYN = 1e-3` (sobre campo φ∈[0,1]), extraído del `rng` propio de cada semilla
(no solo la condición inicial depende de la semilla — la trayectoria completa
también). La amplitud NO se barre en F4-2 (eso es el objeto de F1-5/F3-1); aquí solo
se documenta que el ruido dinámico está PRESENTE y es el mismo para las 4 ramas y para
REAL/NULL, de modo que no sesga la comparación entre ramas. Además, las 16 semillas
(§6) dan 16 trayectorias dinámicas distintas, no solo 16 condiciones iniciales
distintas (el corte de aristas Bernoulli(H) también usa el mismo rng por paso).

---

## 5. NULL (T4)

Por cada rama y cada punto (r, semilla): NULL = barajar (`rng.permutation`) el campo φ
final, DESPUÉS de correr la dinámica completa de esa rama — "barajado del acople"
(destruye la estructura espacial/correlación entre vecinos, conserva el histograma de
valores). Se aplica el NULL POR RAMA (no un NULL global compartido): cada una de las 4
ramas tiene su propio REAL y su propio NULL, corridos con la MISMA semilla y el MISMO
φ final antes de barajar.

---

## 6. BARRIDO Y CALIBRACIÓN (congelados antes de correr)

- **N = 200** (malla principal; igual que `cs074_rcruz.py modo=produccion`).
- **eps = 1e-2** (canónico, tomado de la lista propia de `cs074_rcruz.py`; régimen
  claramente por encima del piso de ruido de inicialización). Gate adicional barato:
  `eps=0` en las 4 ramas → debe dar P=0 exacto (la función `persistencia` ya lo
  garantiza por construcción: contraste0=0 ⇒ P=0). Se verifica, no se incluye en el
  presupuesto de la grilla principal.
- **r_target ∈ {0, 0.1, 0.3, 0.5, 1, 2, 5, 10, 30, 100}** — idéntico eje pre-registrado
  de `cs074_rcruz.py` (cruza r≈1 fino).
- **Semillas (16, ≥12 exigidas):**
  `[7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 271828, 161803, 500500,
  31415, 27182, 141421]` — las primeras 10 son el banco estándar del proyecto (mismas
  de `CF2_estiramiento_motor.py`); las 6 adicionales se fijan aquí, antes de correr,
  para llegar a 16 sin tocar el banco original.
- **D0**: media de `medir_D(N=200, eps=1e-2, seed)` sobre las 16 semillas (difusión
  pura, un paso, sin ruido dinámico — mide la propiedad del campo, no del forzamiento).
- **pasos_fijo**: calibrado UNA vez en la rama `00` (D≡D0 constante, sin cortes) con
  `medir_pasos_lavado`-equivalente: mediana (sobre 16 semillas) del primer paso en que
  P(t) < 0.05, multiplicado por MARGEN_LAVADO=1.15 — igual criterio que
  `cs074_rcruz.py`. Ese mismo `pasos_fijo` se usa en las 4 ramas × 10 r × 16 semillas
  para que la comparación sea a tiempo total igual.
- **N=400 confirmatorio (best-effort):** si el tiempo de cómputo lo permite tras la
  grilla principal, se repite la grilla completa a N=400 como chequeo de robustez de
  escala (no cambia el veredicto pre-registrado de §7, es verificación cruzada
  adicional). Se reporta si se corrió o no.

---

## 7. SEGUNDO OBSERVABLE INDEPENDIENTE (regla de verificación múltiple, obligatoria)

Además de P (autocorrelación×varianza), se mide **información mutua espacial
antipodal**: se empareja la celda `i` con la celda `i+N/2` (i=0..N/2−1) del mismo
campo φ final, se discretizan ambas series en B=10 bins uniformes sobre [0,1], y se
calcula la información mutua discreta `MI = Σ p_ij·log(p_ij/(p_i·p_j))` del histograma
conjunto. Es un método distinto (información, no correlación de vecinos) sobre el
mismo φ. Su mapa (rama, r) debe COINCIDIR cualitativamente con el de P — si difieren,
se reporta como hallazgo (uno de los dos podría ser artefacto del observable, T2).

**Tercera verificación:** auditoría en disco — el JSON crudo incluye el campo φ FINAL
completo (REAL y NULL) de cada (rama, r, semilla), no solo P y MI, para que quien no
escribió el código pueda re-derivar cualquier estadístico de forma independiente.

---

## 8. CRITERIO DE PASS / LECTURA — LAS TRES LECTURAS PRE-REGISTRADAS

Para cada rama x ∈ {a, b, c} y cada r, con P̄ = media sobre 16 semillas:

- **Efecto vs baseline:** ΔP_x(r) = P̄_x_real(r) − P̄_00_real(r)
- **Significancia vs su propio NULL:** z_x(r) = (P̄_x_real − P̄_x_null) / σ_pooled(r),
  σ_pooled = sqrt((var_real+var_null)/2) con piso 1/√16 (misma fórmula que
  `cs074_rcruz.py::barrido_rcruz`).
- Rama x "congela de forma significativa" en r si: `z_x(r) ≥ Z_THR=2.0` **Y**
  `ΔP_x(r) ≥ DELTA_MIN=0.05`.

**Banda de decisión:** r∈{5,10,30,100} (régimen "congelado" donde cs074_rcruz mostró
separación clara REAL/NULL). Se promedia ΔP_x y el flag de significancia sobre esos 4
puntos de r para la decisión; la curva COMPLETA (los 10 puntos de r) se reporta de
todos modos (T5 — no se recorta a un bin).

Con `RATIO_bc = ΔP_b / ΔP_c` y `RATIO_ac = ΔP_a / ΔP_c` (promediados en la banda):

1. **DENSIDAD CAUSAL PROPIA** si la rama `b` es significativa Y `RATIO_bc ≥ 0.5`
   (la dilución sola logra ≥ la mitad del efecto natural).
2. **DENSIDAD = PROXY DE EXPANSIÓN** si la rama `b` NO es significativa (ΔP_b≈0,
   indistinguible de su propio NULL y del baseline) Y la rama `a` sí es significativa
   con `RATIO_ac ≥ 0.8` (la expansión sola explica ≥ 80% del efecto natural).
3. **INTERACCIÓN DE AMBAS** en cualquier otro caso (ninguna de las dos por sí sola
   cubre el umbral, o `ΔP_a+ΔP_b` no reconstruye linealmente `ΔP_c` — evidencia de
   sinergia o sustitución parcial).

Los umbrales (Z_THR=2.0, DELTA_MIN=0.05, 0.5, 0.8) se fijan AQUÍ, antes de correr, como
números redondos pre-registrados — no se ajustan después de ver el resultado (T1/T3).

**Quién adjudica:** este documento fija CÓMO se decide con los números. La ejecución
(F4_2_motor.py) calcula los números y el booleano de cada lectura, pero el reporte
final NO se auto-adjudica en prosa grandilocuente cuál "ganó" — se entregan los tres
booleanos + los números exactos; el cierre de interpretación es de CS/Alexis.

---

## 9. QUÉ SE ENTREGA

- Este protocolo (verbatim, con su timestamp real de creación).
- `F4_2_motor.py` (motor, mtime posterior a este archivo).
- JSON crudo con: por rama × r × semilla → P_real, P_null, MI_real, MI_null, z, ΔP,
  φ_final (REAL y NULL), D0, pasos_fijo, H, y los tres booleanos de lectura con sus
  números de soporte.
- Resumen impreso (stdout + log) con tiempos de corrida.

**Nada de esto se edita después de correr.** Si algo falla o sale raro, se reporta
como está — no se "arregla" el criterio (T3) ni el código sin pasar por CS.
