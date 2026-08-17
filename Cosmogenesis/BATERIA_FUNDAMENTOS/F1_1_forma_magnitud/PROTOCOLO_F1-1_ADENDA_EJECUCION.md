# ADENDA DE EJECUCIÓN — F1-1 (fechada, posterior al pre-registro)

**No modifica** el observable, el NULL, ni el criterio de PASS del
`PROTOCOLO_F1-1_PREREGISTRO.md` (eso permanece congelado, T3). Esta adenda
documenta dos decisiones tomadas DURANTE la ejecución, ambas por razones de
cómputo, con la evidencia de validación adjunta.

**Fecha:** 2026-07-24 (misma sesión, tras el smoke test y antes de producción).

---

## 1. Hallazgo: escalado O(N²) hace el barrido literal inviable en una noche

Medido directamente (no estimado): `pasos` (calibrados por lavado, igual que
hace `cs074_rcruz.py`) escala aproximadamente como N²:

| N | pasos calibrados (medido) |
|---|---|
| 100 | 1.553 |
| 200 | 6.095 |
| 400 | 24.380 |
| 800 | 97.520 |
| 1600 | 390.080 (extrapolado por la misma ley, confirmado por benchmark) |

Con el bucle literal (una corrida de `base.corrida()` por cada punto
(ε,r,N,semilla) × REAL/NULL), el costo total para el grid completo
pre-registrado (13 ε × 34 r × 12 semillas × 2) medido/extrapolado:

| N | tiempo total estimado (bucle literal, sin optimizar) |
|---|---|
| 200 | ~93 min |
| 400 | ~403 min (6.7 h) |
| 800 | ~2.619 min (43.6 h) |
| 1600 | ~14.516 min (242 h ≈ 10 días) |

Esto excede por mucho "una noche". Se optimizó la EJECUCIÓN (no el modelo).

## 2. Optimización de implementación (validada como IDÉNTICA, no aproximada)

Se implementó un motor vectorizado (`F1_1_motor.py`, funciones `*_batch`) que:

1. **Comparte la trayectoria REAL y NULL**: en `cs074_rcruz.py`,
   `evolucionar(..., null=True)` con una semilla dada produce el MISMO φ final
   pre-permutación que `null=False` (mismo rng fresco, mismo consumo), y solo
   agrega UNA permutación al final. Confirmado por identidad matemática y
   validado numéricamente (`valida_batch.py`): correr la dinámica UNA vez y
   derivar P_real y P_null de esa misma trayectoria reproduce EXACTO
   (diff=0.0) el resultado de llamar `corrida(null=False)` y
   `corrida(null=True)` por separado.
2. **Comparte la secuencia aleatoria entre valores de r** (a ε y semilla
   fijos): `base.corrida()` crea un `rng` fresco con la MISMA semilla para
   cada r; el patrón de consumo de esa rng (fases si ε>0, luego draws por
   paso) es idéntico para cualquier r que caiga en el régimen "0<H<1". Esto
   permite vectorizar los 33-34 valores de r de ese régimen en un solo array
   (R,N) que comparte el mismo draw `u` por paso — identidad matemática, NO
   aproximación (mismo generador, misma secuencia).
3. **Régimen especial detectado y corregido**: `paso_expansion()` en
   `cs074_rcruz.py` retorna TEMPRANO sin consumir ningún número aleatorio
   cuando H≤0 o H≥1 (líneas 116-121 de `cs074_rcruz.py`). Si no se separa
   este caso, el batching rompe la correspondencia con `base.corrida()`
   (se detectó exactamente este error en la primera versión de la
   validación — ver `valida_batch2.py`, fila r=0 con diff_null=4.21e-02).
   Se corrigió separando el barrido en 3 regímenes (`zero`: H≤0, `full`: H≥1,
   `mid`: 0<H<1), cada uno con su propio manejo de la rng, exacto para los
   tres.

**Validación (script `valida_batch3.py`, ejecutado y verificado por CC en esta
sesión):** 3 casos de prueba (D moderado sin full; D grande con los 3
regímenes mezclados; ε=0 con D=0 puro zero+full) — TODOS los puntos
comparados contra `base.corrida()` dan diferencia ≤ 6.66e-16 en P_real y
P_null (precisión de punto flotante double, es decir, IDÉNTICO). Los 3 casos:
**PASA**.

**Benchmark de costo real tras la optimización** (medido, no estimado, bajo
la carga del sistema compartido en el momento de la medición — ver §3):

| N | pasos | tiempo total FULL grid (13 ε×34 r×12 semillas), motor batched |
|---|---|---|
| 200 | 6.095 | ~13.4 min |
| 400 | 24.380 | ~74.7 min (1.24 h) |
| 800 | 97.520 | ~16.9 h |
| 1600 | 390.080 | ~122.1 h (≈5.1 días) |

**Por qué la optimización no da más:** el perfilado mostró que a N grande el
costo por paso escala LINEAL con el número de filas del batch (no hay ahorro
de "overhead" de Python que explotar — el costo real es ancho de banda de
memoria: ~12 pasadas de array completo por paso de difusión, cada una de
tamaño R×N). El batching evita trabajo REDUNDANTE (RNG duplicada, trayectoria
NULL recomputada) pero no reduce el trabajo físico real de difusión, que
domina a N grande. Se probó además una variante más liviana (activo en
float64 en vez de bool, evitando el cast por paso): dio solo 1.18× de mejora,
no significativo — no se adoptó por el riesgo/beneficio.

## 3. Decisión de reducción para N=800 y N=1600 (con motivo y flag explícito)

Con el motor batched validado, N=200 y N=400 corren el grid COMPLETO
pre-registrado (13 ε × 34 r × 12 semillas) sin ninguna reducción.

Para N=800 y N=1600, el costo medido (16.9 h y 122.1 h respectivamente)
excede lo razonable para "una noche", y se agrava porque la máquina está
compartida con las otras ~23 corridas paralelas de esta batería (load average
medido ≈47 en una máquina de 16 hilos al momento de decidir esto — cualquier
estimación de tiempo aquí es optimista si la contención sube).

Siguiendo la instrucción explícita del coordinador ("si el costo es inviable,
reduce el barrido de N más alto — p.ej. omite 1600 o usa menos semillas
ahí — y repórtalo como limitación honesta, no lo escondas"), se aplica:

- **N=800:** semillas reducidas de 12 → **8** (grid ε×r completo, 13×34 sin
  cambios). Reducción ×1.5 → tiempo estimado ~11.3 h.
- **N=1600:** semillas reducidas de 12 → **4**; r reducido de 34 → **15**
  puntos (al MÍNIMO pre-registrado, conservando la resolución fina cerca de
  r≈1: se usan los 15 puntos {0, 0.1, 0.3, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25,
  1.5, 2, 5, 10, 30, 100}); ε se mantiene COMPLETO (13 puntos, incluye el
  control ε=0). Reducción combinada ×9.1 → tiempo estimado ~13.4 h.

**Esto es una desviación EXPLÍCITA del pre-registro para N=800/1600
únicamente** (que pedía ≥12 semillas y ≥15 puntos de r en TODOS los N). Se
reporta así, sin disimularlo:

- La dispersión entre semillas de N=1600 (n=4) es una muestra MÁS DÉBIL que
  la de N=200/400 (n=12) y N=800 (n=8) — CS debe ponderarlo así al leer la
  curva.
- El eje r de N=1600 tiene 15 puntos (mínimo pre-registrado) en vez de 34 —
  sigue siendo un barrido real (no un punto, T7), con resolución fina cerca
  de r≈1, pero menos denso que N=200/400/800.
- El criterio de PASS mecánico (congelado en el pre-registro) se evalúa
  IGUAL para todos los N con los datos disponibles; no se cambia el juez.

## 4. Qué NO se cambió

- El observable (`persistencia()`), la física (`campo_inicial`,
  `paso_difusion`, `paso_expansion`), y el NULL siguen siendo exactamente las
  funciones de `cs074_rcruz.py`, verificadas idénticas (§2).
- El criterio de PASS del pre-registro no se tocó.
- N=200 y N=400 corren el grid pre-registrado COMPLETO, sin reducción.

---
*Fin de la adenda. Archivos de validación en
`/private/tmp/.../scratchpad/valida_batch.py`, `valida_batch2.py`,
`valida_batch3.py`, `benchmark_batch.py` (locales a la sesión de CC; el
código de producción con la misma lógica validada vive en `F1_1_motor.py`
dentro de esta carpeta).*
