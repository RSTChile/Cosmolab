# PROTOCOLO F4-4 — PRE-REGISTRO
### "Densidad crítica de congelamiento: ¿hay un ρ bajo el cual la diferencia se bloquea?"

**Fecha/hora de escritura:** 2026-07-24 05:34 (America/Santiago, -04) — ANTES de escribir
el motor (`F4_4_densidad_critica_motor.py`, mtime posterior verificable en disco).
**Ejecutor:** CC (subagente F4-4, batería de 24 experimentos, Cosmogénesis).
**Fuente del encargo:** `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección F4-4 (líneas 275-282).
**No se edita este archivo tras ver resultados (T3).** Si el motor falla el PASS,
se reporta el FAIL — no se cambia el criterio aquí.

---

## 1. PREGUNTA

¿Existe una densidad ρ_c del campo por debajo de la cual la difusión (reabsorción)
se vuelve tan débil que la diferencia sembrada queda congelada — persiste alto —
incluso en el mismo tiempo físico en que a densidad de referencia se habría lavado?
Y si existe, ¿es estable al variar N (tamaño del sistema) y la perturbación dinámica
(ruido), o es un artefacto de una sola configuración?

Esto extiende el mecanismo de `cs074_rcruz.py` (persistencia vs. r=H/D, congelamiento
por AISLAMIENTO cuando la expansión gana a la difusión) añadiendo un eje de densidad ρ
que actúa sobre la propia difusión — enlace físico ya usado en `CF2_estiramiento_motor.py`
(D ∝ ρ/ρ0) — y preguntando si ρ por sí sola, a **tiempo físico fijo**, puede congelar.

---

## 2. MECANISMO Y ENLACE FÍSICO (no elegido para forzar el resultado)

- Sustrato: mismo campo continuo en anillo de N celdas y misma dinámica de
  `cs074_rcruz.py` (difusión solo por aristas vivas + expansión = corte Bernoulli
  de aristas con probabilidad H por paso).
- **Densidad → difusión:** la tasa de difusión por paso se escala por ρ/ρ0:
  `nuevo = phi + RATE0·(ρ/ρ0)·(media − phi)`, con RATE0=0.5 (la tasa original,
  sin retocar, de `cs074_rcruz.py`) y ρ0=1.0 (densidad de referencia, misma
  convención `RHO0=1.0` que `CF2_estiramiento_motor.py`). Este es el mismo enlace
  físico ya usado en el código base (no un coeficiente inventado para F4-4).
- **D se MIDE, no se impone** (T1): para cada ρ, D(ρ) se mide empíricamente con
  el mismo método de un paso (`medir_D`) que usa `cs074_rcruz.py`, aplicado con la
  tasa escalada por ρ. Como la tasa escala linealmente, D(ρ) debería salir
  aproximadamente ∝ ρ — pero se mide, no se asume, y se reporta el valor medido.
- **Expansión:** H = min(r_objetivo · D(ρ), 1.0) — mismo mecanismo r=H/D de
  `cs074_rcruz.py`, ahora con D dependiente de ρ.
- **Reloj físico FIJO (clave del experimento):** el número de pasos de la corrida
  NO se recalibra por cada ρ. Se calibra UNA sola vez, a ρ=ρ0 (referencia), con el
  mismo método de `medir_pasos_lavado` (umbral P<0.05, margen 1.15×), y ese mismo
  número de pasos se usa para TODOS los ρ del barrido. Si se recalibrara por ρ,
  el efecto de congelamiento se auto-cancelaría por construcción (T2: el juez no
  puede ajustarse a la variable que se está juzgando) — este es el punto de diseño
  más importante del protocolo.
- **ε (amplitud de la mancha sembrada):** fijo en ε=1e-3, el mismo valor de
  referencia usado en la calibración de producción de `cs074_rcruz.py` (no elegido
  ad hoc para F4-4; heredado del experimento predecesor).
- **Ruido dinámico (T7):** en cada paso se suma al campo ruido gaussiano de media 0
  y amplitud `sigma_ruido` (parámetro barrido, no cosmético de semilla).
- **Observable primario:** P = c·v (autocorrelación a primer vecino × varianza
  normalizada), idéntico a `cs074_rcruz.persistencia()` — mismo observable
  validado en CS074/F1-1, no inventado aquí.
- **Observable secundario (diagnóstico, mismo cómputo, sin costo extra):**
  v_solo = phi.var()/contraste0² (solo el factor de varianza, sin autocorrelación).
  Sirve de chequeo cualitativo de que el patrón de congelamiento no depende de la
  autocorrelación por sí sola.

---

## 3. BARRIDO PRE-REGISTRADO

| Eje | Rango | Puntos |
|---|---|---|
| ρ/ρ0 | geomspace(1e-6, 1.0) | **15** (6 décadas) |
| r objetivo | {0, 0.1, 0.5, 1, 3, 10, 30} | 7 (cruza r=1) |
| N | {200, 400} | 2 (chequeo de estabilidad T7/verificación-cruzada-b) |
| σ_ruido dinámico | {0, 1e-4, 1e-3, 1e-2} | 4 (chequeo de estabilidad T7/verificación-cruzada-b) |
| semillas | 7,42,99,777,2025,3141,8191,99991,12345,54321,271828,161803 | **12** |

Total combinaciones (ρ×r×N×σ) = 15×7×2×4 = 840 celdas × 12 semillas × 2 ramas
(REAL/NULL) = 20,160 corridas de campo. Ejecutado en paralelo (multiprocessing,
núcleos disponibles) — cómputo largo autorizado por el director.

---

## 4. NULL

**Barajado del acople**, operacionalizado como permutación del campo φ al final de
la corrida (idéntico a `cs074_rcruz.py`: `phi = rng.permutation(phi)`). Esto destruye
el orden espacial inducido por la dinámica de acople (difusión+cortes) preservando el
histograma de valores — es la misma operacionalización ya validada en CS074-rcruz.
Se reporta explícitamente aquí para que quede escrito ANTES de correr (T3).

---

## 5. CRITERIO DE PASS / LECTURAS POSIBLES (T5: curva, no gate binario)

No hay un único "PASS/FAIL". Se pre-registran **dos lecturas posibles, ambas
hallazgo**, tal como exige la sección F4-4 del documento madre:

- **Lectura A — existe ρ_c:** la curva P_real(ρ) (a r y N y σ fijos) muestra una
  transición — P alto (congelado) a ρ bajo, P bajo (lavado) a ρ alto — con un punto
  de cruce ρ_c identificable. **Se reporta ESTABLE** solo si ρ_c no se mueve más de
  ~1 orden de magnitud al cambiar N∈{200,400} y σ_ruido∈{0,1e-4,1e-3,1e-2}, y si la
  dispersión entre semillas (≥12) en ese punto es menor que la separación real-NULL.
- **Lectura B — no existe ρ_c (scale-free o plano):** P_real(ρ) no muestra
  transición clara (plano, o domina el ruido de semilla) — se reporta la curva
  completa igual.

**ρ_c operacional (si existe):** primer valor de ρ (recorriendo de alto a bajo)
donde P_real cruza el punto medio entre el piso medido a ρ=ρ0 (más lavado) y el
techo medido al ρ mínimo del barrido (más congelado), por interpolación lineal en
log(ρ). Es una lectura DERIVADA de la curva completa, no un gate que decide
pasa/no-pasa — la curva entera se reporta siempre.

**Verificación cruzada obligatoria (según F4-4):**
- (a) el NULL debe caer (P_null bajo y estable, separado de P_real donde P_real
  esté congelado; si REAL=NULL en algún tramo, se reporta tal cual — es hallazgo).
- (b) el ρ_c (si existe) debe ser estable al variar N y la perturbación dinámica
  (σ_ruido) — NO de una sola semilla ni de una sola combinación.
- (c) auditoría en disco: código + JSON crudo completo, sin resumir de palabra.

**Control de validez (heredado de cs074_rcruz):** a ρ=ρ0 (referencia) y r=0, la
difusión debe lavar (P_real bajo) — si no lava a la densidad de referencia, el
reloj físico fijo está mal calibrado y el resto no se interpreta como cruce válido.

---

## 6. QUÉ NO SE HACE

- No se recalibra el número de pasos por ρ (ver §2 — anularía el efecto).
- No se elige a mano ningún ρ_c objetivo; sale de la curva medida.
- No se auto-adjudica el veredicto final ("existe"/"no existe" ρ_c a nivel de
  batería) — eso lo decide CS con la curva cruda. Este script solo mide y reporta.
- No se toca `cs074_rcruz.py` ni `CF2_estiramiento_motor.py` (se leen, no se editan).
- No hay topología ni commits — fuera de alcance de este experimento.

---

## 7. ARCHIVOS DE SALIDA

- Código: `F4_4_densidad_critica_motor.py` (mismo directorio, mtime posterior a este).
- Resultado crudo: `Cosmogenesis-Web/results/F4_4_densidad_critica/F4_4_resultado_produccion.json`
  (una fila por combinación ρ×r×N×σ, agregada sobre 12 semillas, con dispersión
  real entre semillas incluida).
- Log de corrida: `Cosmogenesis-Web/results/F4_4_densidad_critica/F4_4_run.log`.
