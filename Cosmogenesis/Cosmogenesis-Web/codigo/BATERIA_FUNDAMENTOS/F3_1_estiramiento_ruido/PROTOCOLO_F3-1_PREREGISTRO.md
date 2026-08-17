# PROTOCOLO F3-1 — PRE-REGISTRO
## "Estiramiento geométrico del gradiente, barrido amplio de a + ruido dinámico"

**Fecha de escritura:** 2026-07-24 (ANTES de correr `F3_1_estiramiento_motor.py`; este
archivo se congela con este mtime — no se edita tras ver resultados, T3).

**Ejecutor:** CC, experimento F3-1 de la batería `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`
(Enfoque 3, experimento 1). Corre en paralelo con otros 23 experimentos, cada uno con su
propio prefijo. Este protocolo cubre EXCLUSIVAMENTE el prefijo `F3_1_`.

---

## 1. Contexto y motivo (por qué existe este experimento)

CF-2 (`CF2_estiramiento_motor.py`, resultado en
`Cosmogenesis-Web/results/CF2_estiramiento/CF2_estiramiento_produccion_result.json`)
dio **PASS** (rate=1.0, 10/10 semillas) en la pregunta "¿el estiramiento geométrico
∇_fis = ∇_comov/a suaviza el gradiente físico al expandirse el espacio?". Pero al
inspeccionar las 10 corridas se descubrió que son **casi idénticas entre semillas**:
la PDE de difusión que gobierna el campo es cuasi-determinista (el ruido solo entra
como condición inicial de amplitud fija 1e-4, y se disuelve rápido bajo difusión), así
que variar la semilla NO perturbaba la dinámica de forma apreciable. Un "10/10" en ese
régimen es **T7 disfrazado de robustez** (barrer solo semilla, no la dinámica).

**F3-1 corrige exactamente eso:** en vez de fijar la amplitud del ruido inicial en
1e-4, se BARRE esa amplitud en ≥6 puntos log entre 1e-4 y 1e-1 — la perturbación
dinámica que T7 exige — y se exige que el veredicto REAL≻NULL sea **estable a lo largo
de esa curva**, no solo repetible al cambiar de semilla.

---

## 2. Código base (NO se edita)

`Cosmogenesis-Web/codigo/CF2_estiramiento/CF2_estiramiento_motor.py` — leído completo.
Se reutiliza tal cual su núcleo físico (campo T en malla LxL, salto tanh, difusión con
laplaciano de 5 puntos, reloj genético t_g→a=exp(H_EXP·t_g), brazo REAL con
ρ=ρ0/a³,D=D0·ρ/ρ0 vs brazo NULL_RHO_FIXED con ρ≡ρ0,D≡D0, observable
∇_fis=∇_comov/a). El motor nuevo (`F3_1_estiramiento_motor.py`, prefijo propio, archivo
separado) **importa/reimplementa la física sin alterar el original**; la única
diferencia funcional es que la amplitud del ruido inicial deja de ser la constante
`1e-4` y pasa a ser un parámetro barrido explícito `sigma_ruido`.

No se toca `CF2_estiramiento_motor.py` ni ningún otro archivo existente.

---

## 3. Observable exacto

`∇_fis(a) = ∇_comov(a) / a`, donde `∇_comov(a) = max_{banda central} |∂_x T|` es la
abruptancia comóvil del salto de temperatura T en la banda central de la malla
(evita artefactos de wrap-around periódico; banda = columnas [L/8, 7L/8)).
Es geometría pura del campo T y de a — no depende de ninguna variable de linaje o
juez de otro experimento (T2).

**Segundo observable independiente (verificación cruzada):** pendiente log-log
`slope = d(ln ∇_fis)/d(ln a)` ajustada por mínimos cuadrados sobre TODO el barrido de
a de una corrida. La predicción de "estiramiento puro" (T comóvil aprox. constante,
solo se re-escala por a) es `slope_REAL ≈ −1`. Este es un observable DISTINTO del
booleano de monotonicidad — mide la FORMA de la caída, no solo su signo.

---

## 4. Barrido pre-registrado

| Parámetro | Rango | Puntos | Espaciado |
|---|---|---|---|
| `a` (factor de expansión) | [1, 1e4] | 13 | log (geomspace) |
| `sigma_ruido` (amplitud de ruido inicial) | [1e-4, 1e-1] | 8 | log (geomspace) |
| semillas | ver lista abajo | 16 | — |
| modo | {REAL, NULL_RHO_FIXED} | 2 | — |

Semillas (10 estándar del proyecto + 6 nuevas para llegar a 16, ≥12 exigidas):
`[7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321, 13, 271828, 161803, 31415, 90210, 20260724]`

Total de trayectorias físicas: 8 (sigma) × 16 (semillas) × 2 (modo) = 256.
Cada trayectoria evalúa los 13 checkpoints de `a` a lo largo de una única integración
markoviana (idéntico al método de checkpointing de CF-2 — no se re-simula desde cero
por punto de `a`, es la misma trayectoria muestreada en los instantes t_g(a) objetivo).

**Ruido dinámico añadido en CADA PASO (no solo condición inicial):** además de
sembrar el campo inicial con `sigma_ruido`, se inyecta ruido gaussiano de la MISMA
amplitud `sigma_ruido` en cada subpaso de difusión, escalado por `sqrt(dt_sub)` (ruido
tipo Wiener), para que la perturbación sea genuinamente dinámica y no se disuelva de
inmediato bajo el operador de difusión. Esto es superset de lo pedido por la spec
(que pide barrer la amplitud del ruido INICIAL); se documenta la extensión aquí y se
reporta también la variante sólo-inicial como submuestra de control.

Nota de honestidad: la spec F3-1 pide explícitamente "amplitud de ruido inicial"; el
forzamiento por paso es una extensión propia (para blindar T7 con más margen, siguiendo
el espíritu de F1-5). Se reporta CLARAMENTE cuál curva corresponde a cuál variante en
el JSON de salida (`ruido_solo_inicial` vs `ruido_inicial_y_dinamico`), y el veredicto
principal se basa en la variante literal pedida por la spec: **ruido en la condición
inicial**, barriendo su amplitud. La variante con ruido dinámico por paso se reporta
como verificación adicional, no reemplaza al pre-registro literal.

---

## 5. NULL

`NULL_RHO_FIXED`: idéntico al de CF-2 — densidad ρ≡ρ0 (sin dilución), D≡D0 constante.
El gradiente comóvil bajo el NULL debe seguir erosionándose por difusión pura sin el
freno de D→D0/a³; a diferencia del REAL, el gradiente FÍSICO del NULL no se beneficia
del factor geométrico 1/a combinado con difusión congelada, así que su caída con `a`
debe ser cualitativamente distinta (más lenta / no monotónica de la misma forma) — es
el mismo NULL que CF-2 ya usó y que mordió (rate=1.0 con separación de pendiente).

---

## 6. Criterio de PASS (congelado, no se toca tras ver resultados)

Por cada combinación (sigma_ruido, semilla):
- `mono_REAL` = ∇_fis(a) del REAL es no-creciente en a (tolerancia MONO_TOL=1e-9,
  igual que CF-2).
- `mono_NULL` = ídem para NULL_RHO_FIXED.
- `slope_REAL`, `slope_NULL` = pendiente log-log ajustada por mínimos cuadrados.
- `slope_diff = |slope_NULL - slope_REAL|`.
- `punto_pass` = `mono_REAL AND (NOT mono_NULL OR slope_diff >= SLOPE_DIFF_MIN)`
  (SLOPE_DIFF_MIN = 0.05, idéntico umbral de CF-2 — no se re-elige a mano, T1).
- `slope_cerca_de_menos_uno` = `abs(slope_REAL - (-1.0)) <= SLOPE_TOL` con
  `SLOPE_TOL = 0.15` (banda pre-registrada alrededor de −1; el −1 teórico es
  estiramiento puro, T≈cte comóvil).

**Curva de robustez (la pregunta central de F3-1, NO una tasa única):**
`P(sigma_ruido) = fracción de semillas con punto_pass=True`, calculada para cada uno
de los 8 valores de sigma_ruido. El PASS pre-registrado del experimento completo es:

> **PASS_F3-1** si, para TODOS los `sigma_ruido` del barrido, `P(sigma_ruido) >=
> PASS_RATE_MIN (0.55)` — es decir, el veredicto no colapsa en ningún punto del rango
> de amplitud de ruido — Y la pendiente media de REAL a través de todas las
> combinaciones cae dentro de la banda `[-1.15, -0.85]` (estiramiento ≈ −1) — Y el
> NULL muerde (mono_NULL=False o slope_diff>=0.05) en al menos el 70% de las
> combinaciones.

Si `P(sigma_ruido)` cae por debajo de 0.55 en ALGÚN punto del barrido de ruido, el
veredicto es **FAIL_INESTABLE** (el PASS de CF-2 no era robusto a la dinámica) y se
reporta como hallazgo, no se suaviza ni se re-elige el umbral (T3).

---

## 7. Verificación cruzada (tres vías, obligatorias)

(a) **NULL muerde**: se reporta explícitamente la tasa de "NULL muerde" por punto de
    sigma_ruido (fracción de semillas donde `NOT mono_NULL OR slope_diff>=0.05`).
(b) **Pendiente ≈ −1 en REAL**: se reporta `slope_REAL` completo (media, desviación
    estándar entre semillas, por cada sigma_ruido) contra el −1 teórico.
(c) **Auditoría en disco**: código (`F3_1_estiramiento_motor.py`) + JSON crudo
    (`F3_1_estiramiento_produccion_result.json`) quedan en disco para que alguien que
    NO escribió el código pueda re-verificar sin depender de este reporte narrado.

---

## 8. Qué NO se hace aquí

- No se toca `CF2_estiramiento_motor.py` ni `TEST_RHO_DISPERSION.py`.
- No se auto-adjudica "el fundamento persiste" más allá de este experimento puntual.
- No se cambia este criterio después de correr el motor (T3). Si el resultado es FAIL
  o inestable, se reporta tal cual.
- No topología, no commits (regla del director para este batch).
