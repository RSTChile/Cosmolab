# PROTOCOLO F3-5 — PRE-REGISTRO
## "Enfriamiento en el espectro: ¿qué escalas se congelan primero?"

**Fecha de escritura:** 2026-07-24, 09:33 UTC (ANTES de correr `F3_5_espectral_motor.py`;
este archivo se congela con este mtime — no se edita tras ver resultados, T3).

**Ejecutor:** CC, experimento F3-5 de la batería `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`
(Enfoque 3, experimento 5). Corre en paralelo con otros 23 experimentos, cada uno con su
propio prefijo. Este protocolo cubre EXCLUSIVAMENTE el prefijo `F3_5_`. No se toca ningún
archivo fuera de `codigo/BATERIA_FUNDAMENTOS/F3_5_espectral/` y
`results/BATERIA_FUNDAMENTOS/F3_5_espectral/`.

---

## 1. Contexto y motivo

F3-1 (`F3_1_estiramiento_motor.py`, en paralelo) mide el enfriamiento por estiramiento
con el observable **gradiente**: `∇_fis(a) = ∇_comov(a)/a`. F3-5 pide explícitamente una
**lente de verificación TOTALMENTE distinta**: análisis de Fourier del campo, para ver
qué longitudes de onda dejan de difundir (se congelan) al crecer `a`. Es la forma en que
esta batería evita T7 en este experimento puntual — no barriendo más semillas del mismo
método, sino cambiando el MÉTODO de medición sobre la misma dinámica física.

**Predicción física pre-registrada (a firmar, no se toca tras correr):**
> Las escalas GRANDES (longitud de onda λ grande, número de onda comóvil k pequeño) se
> congelan PRIMERO — a un `a` más chico — porque no alcanzan a re-homogeneizarse a
> través del horizonte creciente. Las escalas CHICAS (λ pequeña, k grande) siguen
> cambiando (difundiendo) hasta un `a` más grande antes de congelarse.
>
> Si el orden observado es el inverso (chicas se congelan antes que grandes, o no hay
> orden), es un **dato en contra** de la predicción y se reporta como tal — no se
> reinterpreta a posteriori (T3).

---

## 2. Código base (NO se edita)

`Cosmogenesis-Web/codigo/CF2_estiramiento/CF2_estiramiento_motor.py` — leído completo.
El motor de F3-5 **importa directamente** (no copia, no edita) las siguientes piezas del
sello físico original para garantizar que la dinámica subyacente es EXACTAMENTE la misma
que CF-2/F3-1 (comparabilidad real entre métodos — la lente espectral debe mirar la
MISMA física, no una réplica que pueda haber divergido):

- Constantes: `L, H_EXP, RHO0, D0, W0, DT, N_SUB, ORIGINAL_STEPS_PER_TG`.
- Funciones: `initial_T(L, w0)`, `diffuse(T, D, dt, n_sub)`.

Lo que F3-5 **NO reutiliza** de CF-2/F3-1 es el observable: no se calcula `grad_metrics`
en ningún punto del motor nuevo. El observable de F3-5 es exclusivamente espectral
(sección 3). Esto es lo que hace de F3-5 un método independiente y no una repetición de
F3-1 con otro nombre.

No se toca `CF2_estiramiento_motor.py`, `F3_1_estiramiento_motor.py` ni ningún otro
archivo existente.

---

## 3. Observable exacto (espectral, NO gradiente)

1. En cada checkpoint de `a` (misma técnica de muestreo markoviano que CF-2/F3-1: una
   sola trayectoria por semilla, muestreada en los `t_g(a)` objetivo — no se re-simula
   desde cero por punto de `a`), se toma el campo comóvil `T` (malla `L×L`).
2. Se calcula la FFT real 1D a lo largo del eje `x` (la dirección del salto de
   temperatura, `axis=1`) para CADA fila de la malla, y se promedia la potencia
   `|FFT(fila)|²` sobre las `L` filas para obtener un espectro de potencia promedio
   `P(n, a)`, con `n = 0..L/2` el índice de armónico (número de onda comóvil
   `k_n = 2π·n/L`, longitud de onda comóvil `λ_n = L/n` celdas). Se excluye `n=0` (nivel
   medio, no es una escala).
3. Los `n=1..32` (L=64, Nyquist=32) se agrupan en **6 bandas logarítmicas** ("banda de
   escalas" pedida por la spec, cubriendo TODO el espectro):

   | Banda | n (armónicos) | λ comóvil (celdas) | rótulo |
   |---|---|---|---|
   | B0 | 1 | 64 | escala más grande (fundamental) |
   | B1 | 2 | 32 | |
   | B2 | 3–4 | 16–21 | |
   | B3 | 5–8 | 8–12.8 | |
   | B4 | 9–16 | 4–7.1 | |
   | B5 | 17–32 | 2–3.76 (Nyquist) | escala más chica |

   Potencia de banda: `P_banda(a) = Σ_{n∈banda} P(n,a)`. Centro de banda para el test de
   orden: media aritmética de `n` dentro de la banda (proxy monótono de `k`).
4. **Fracción retenida** (normalizada a la potencia inicial de la propia banda, misma
   semilla/modo): `R_banda(a) = P_banda(a) / P_banda(a=1)`.
5. **Detección de congelamiento**: entre checkpoints consecutivos `a_i, a_{i+1}` del
   barrido, se calcula la pendiente log-log local
   `slope_i = [ln R(a_{i+1}) − ln R(a_i)] / [ln a_{i+1} − ln a_i]`.
   Una banda se considera **congelada en el checkpoint `i*`** = el primer índice tal que
   `|slope_j| < FREEZE_SLOPE_TOL` para TODOS los `j ≥ i*` hasta el final del barrido
   (punto de no-retorno de la caída espectral — análogo al criterio de no-retorno de
   F3-4 pero aplicado a la tasa de decaimiento espectral, no a reversibilidad temporal).
   `freeze_a(banda) = a_{i*}`. Si nunca se cumple dentro del rango barrido, se registra
   `freeze_a = NaN` ("no congelada dentro del rango probado" — se reporta como censura,
   no se fuerza un valor).
   `FREEZE_SLOPE_TOL = 0.02` (menos de 2% de cambio relativo de log-potencia por e-fold
   de `a`; valor fijado aquí, no se ajusta tras ver resultados — T1/T3).
6. **Diagnóstico auxiliar** (para distinguir congelamiento "preservado" de congelamiento
   trivial por piso numérico): `R_frozen(banda) = R_banda(a_{i*})`.
   `R_FLOOR = 0.05`: si `R_frozen < R_FLOOR`, la banda se marca `frozen_depleted` (ya
   estaba casi en cero, "congelada" solo porque no queda nada que perder — freeze
   trivial); si `R_frozen ≥ R_FLOOR`, se marca `frozen_preserved` (retiene estructura
   real — el freeze interesante que predice la hipótesis de expansión). Ambos tipos se
   reportan; el test de orden (sección 6) usa `freeze_a` tal cual, sin descartar ningún
   tipo — la clasificación es solo para interpretar el POR QUÉ, no para filtrar datos.

**Segundo observable de verificación cruzada interna:** además de `freeze_a` por banda,
se reporta la curva completa `R_banda(a)` para las 6 bandas — permite inspección visual
directa del "espectro de congelamiento vs a" pedido por la spec, independiente del
umbral de detección.

---

## 4. Barrido pre-registrado

| Parámetro | Rango | Puntos | Espaciado |
|---|---|---|---|
| `a` (factor de expansión) | [1, 1e4] | 30 | log (geomspace) — resolución fina para localizar `freeze_a` con precisión (el checkpointing es gratis: misma trayectoria markoviana, no cuesta recomputar) |
| banda espectral | B0..B5 (tabla §3) | 6 | todo el espectro accesible en L=64 |
| semillas | ver lista abajo | 16 | — |
| amplitud de ruido dinámico `σ_din` | {0.0, 1e-3, 1e-2} | 3 | perturbación DINÁMICA por paso (regla general §2 de la batería, T7) — no reemplaza el pre-registro literal de F3-5 (que no la pide explícitamente), se reporta como capa adicional de robustez, con el veredicto principal basado en `σ_din=0` (dinámica sin forzar, la más comparable a CF-2/F3-1) |
| modo | {REAL, NULL_RHO_FIXED} | 2 | — |

Semillas (10 estándar del proyecto + 6 nuevas para llegar a 16, ≥12 exigidas):
`[7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 13, 271828, 161803, 31415, 90210, 20260724]`

Ruido dinámico: se inyecta `N(0, σ_din·sqrt(dt_sub))` en cada subpaso de difusión
(idéntico mecanismo Wiener que la extensión documentada en F3-1), igual en REAL y NULL.

Total de trayectorias físicas: 3 (σ_din) × 16 (semillas) × 2 (modo) = 96, cada una
evaluada en 30 checkpoints de `a` y 6 bandas espectrales.

---

## 5. NULL

`NULL_RHO_FIXED`: idéntico al de CF-2/F3-1 — densidad `ρ≡ρ0` (sin dilución), `D≡D0`
constante, aunque `a` se sigue definiendo por el mismo reloj genético (no se detiene el
avance de `a` en el NULL — el NULL no tiene expansión FÍSICA sobre `D`, no anula `a`
como variable de bookkeeping).

**Predicción del NULL:** sin el freno `D∝a⁻³`, la exposición difusiva acumulada
`∫D dt` NO converge a un valor finito al crecer `a` (crece sin límite mientras dure la
simulación) — cada banda debería seguir perdiendo potencia mientras el barrido continúe,
en vez de asentarse en un valor preservado. Si el NULL muestra el MISMO patrón de
congelamiento ordenado que el REAL, el "congelamiento por expansión" sería un artefacto
del método de detección o del corte de la simulación, no del mecanismo físico — y se
reporta así (T4).

---

## 6. Criterio de PASS (congelado, no se toca tras ver resultados)

Por cada combinación `(σ_din, semilla)`, en el modo REAL y en el NULL por separado:

- Se calcula `freeze_a(banda)` para las 6 bandas (puede haber `NaN` = censurada).
- Sobre las bandas NO censuradas (`freeze_a` finito), se calcula la correlación de
  rango de Spearman `ρ_orden` entre el centro de banda (proxy de `k`, ascendente de B0 a
  B5) y `freeze_a(banda)`. Se requieren ≥3 bandas no censuradas para calcular `ρ_orden`;
  si hay menos, la combinación queda `INDETERMINADA` (ni pasa ni falla — se reporta la
  tasa de indeterminación por separado, no se cuenta como fallo silencioso).

- `orden_REAL` = `ρ_orden(REAL) ≥ RHO_MIN` con `RHO_MIN = 0.6` (correlación positiva
  fuerte: bandas de k mayor —escala chica— se congelan en `a` mayor que las de k
  menor —escala grande—, confirmando "grandes primero").
- `orden_NULL` = `ρ_orden(NULL) ≥ RHO_MIN` (si el NULL TAMBIÉN ordena así, no muerde).
- `combo_pass` = `orden_REAL AND (NOT orden_NULL)` — el NULL debe FALLAR en reproducir
  el mismo orden para que el hallazgo se atribuya a la expansión, no al método (T4).

**PASS_RATE_MIN = 0.55** (mismo umbral que CF-2/F3-1, no se re-elige a mano — T1).
Se calcula `rate = combos_con_combo_pass / combos_totales_no_indeterminados` para
`σ_din=0` (variante principal) y por separado para cada `σ_din` (curva de robustez a la
perturbación dinámica, igual que F3-1).

> **PASS_F3-5** si `rate(σ_din=0) ≥ 0.55` Y la mediana de `ρ_orden(REAL)` a través de
> las 16 semillas en `σ_din=0` es positiva y ≥ `RHO_MIN` Y el NULL NO alcanza `RHO_MIN`
> en la mayoría de semillas (`orden_NULL` tasa < 0.45, es decir el NULL muerde en la
> mayoría de los casos).
>
> Si `ρ_orden(REAL)` mediano es **negativo** (orden inverso: chicas primero), se reporta
> explícitamente como **FAIL_INVERSO — dato en contra de la predicción física**, no se
> reinterpreta ni se cambia el umbral (T3).
>
> Si el NULL reproduce el mismo orden que el REAL (`orden_NULL` alto), se reporta como
> **FAIL_NULL_NO_MUERDE** (T4) independientemente de qué tan limpio sea el orden en REAL.

---

## 7. Verificación cruzada (tres vías, obligatorias)

(a) **NULL muerde**: se reporta la tasa `orden_NULL` por cada `σ_din`, explícitamente,
    junto a `orden_REAL`, para que la comparación sea auditable directamente en el JSON.
(b) **Curva espectral completa como segundo observable**: además del booleano
    `freeze_a`, se entregan las curvas crudas `R_banda(a)` para las 6 bandas × 16
    semillas × 3 `σ_din` × 2 modos — permite reconstruir el "espectro de congelamiento
    vs a" pedido por la spec sin depender del umbral `FREEZE_SLOPE_TOL` elegido aquí.
(c) **Auditoría en disco**: código (`F3_5_espectral_motor.py`) + JSON crudo
    (`F3_5_espectral_{smoke,produccion}_result.json`) quedan en disco para que alguien
    que NO escribió este código pueda re-verificar sin depender del reporte narrado.

---

## 8. Orden de ejecución

1. **Smoke de dominio pequeño** (`python F3_5_espectral_motor.py smoke`): 2 semillas
   (7, 42), 8 puntos de `a` en [1, 100], `σ_din=0` únicamente, para verificar que la
   FFT, el agrupamiento en bandas y la detección de `freeze_a` corren sin error y dan
   números con sentido físico ANTES de lanzar la producción completa.
2. **Producción completa** (`python F3_5_espectral_motor.py produccion`): barrido
   íntegro de la sección 4.
3. Reporte crudo a CS — sin adjudicar el hallazgo más allá de este experimento puntual.

---

## 9. Qué NO se hace aquí

- No se toca `CF2_estiramiento_motor.py`, `F3_1_estiramiento_motor.py` ni ningún otro
  archivo existente fuera de este prefijo.
- No se auto-adjudica "el fundamento persiste" ni se decide el veredicto final de la
  batería — eso es de CS/Alexis.
- No se cambia este criterio después de correr el motor (T3). Si el resultado es FAIL,
  FAIL_INVERSO o FAIL_NULL_NO_MUERDE, se reporta tal cual, sin suavizar.
- No topología, no commits (regla del director para este batch).
- Si se detecta un error en el código base (`CF2_estiramiento_motor.py`) durante la
  lectura o el uso, se PARA y se reporta a CS con la línea exacta — no se "arregla" a
  criterio propio.
