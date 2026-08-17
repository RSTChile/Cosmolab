# PROTOCOLO F1-1 — PRE-REGISTRO (fechado, congelado antes de correr el motor)

**Experimento:** F1-1 · "Persistencia por autocorrelación de forma contra NULL barajado"
**Enfoque:** 1 — ¿persiste una diferencia ínfima en un campo continuo caliente?
**Ejecutor:** CC (agente de esta corrida)
**Fecha/hora de congelamiento:** 2026-07-24 05:30 (hora local del sistema, America/Santiago -04)
**Documento fuente:** `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección "F1-1"
**Código base (NO editado, solo importado/leído):** `cs074_rcruz.py`

Este documento se escribe y se congela ANTES de ejecutar el motor de producción.
No se edita después de ver resultados (regla T3 de la batería). Si algo falla, se
reporta el FAIL tal cual.

---

## 1. Hipótesis / pregunta

¿La forma espacial de una diferencia (mancha) ínfima sembrada en un campo continuo
caliente sobrevive a la dinámica (difusión + expansión) y le gana al azar (NULL
barajado), y esa ventaja aparece en la banda de expansión (r grande) mientras se
anula en r=0 y en ε=0?

## 2. Observable exacto (congelado)

`P = corr(φ, roll(φ,1)) · [var(φ) / var(φ₀)]`

es decir: **autocorrelación a primer vecino (recortada a ≥0) × razón de varianza
final/inicial** ("forma × magnitud"). Es la función `persistencia()` ya implementada
en `cs074_rcruz.py` (líneas 152-160), reutilizada sin modificación — se importa el
módulo, no se copia a mano, para que el observable sea idéntico byte a byte al del
código base.

El observable NO es función de ε, r o N directamente (no es circular, T2): se mide
sobre el campo φ final, cualesquiera que hayan sido los parámetros de la corrida.

## 3. Física (idéntica a cs074_rcruz.py, sin cambios)

- Campo inicial: fondo=1 + ε·(perturbación multi-modo Fourier m=1..5, fases
  aleatorias, normalizada a std=1) — `campo_inicial()`.
- Difusión: promedio con vecinos por aristas vivas, vectorizado — `paso_difusion()`.
- Expansión: corte Bernoulli por arista viva con probabilidad H por paso —
  `paso_expansion()`.
- D = fracción de contraste borrada en UN paso de difusión pura (H=0), medida del
  propio campo — `medir_D()`.
- H(r) = min(r·D, 1.0) — r es la razón interna H/D, no un número puesto a mano (T1).
- pasos = calibrados por lavado: tiempo medido (a H=0) para que P caiga bajo
  P_LAVADO=0.05, con margen 1.15× — `medir_pasos_lavado()`. Se calibra UNA vez por
  N (en ε=1e-3, igual que el modo "produccion" del código base) y se aplica fijo a
  todo el barrido de ese N — esto es lo que hace el código base; se documenta el
  valor medido de `pasos` por N en el JSON de salida.

## 4. Barrido (congelado, todos los ejes)

| Eje | Rango | Puntos | Nota |
|---|---|---|---|
| ε | 1e-12 … 1 (log) | 12 puntos log-espaciados + 1 punto ε=0 (control) = 13 | `np.logspace(-12,0,12)` ∪ {0} |
| r = H/D | 0 … 100 | 34 puntos, FINOS cerca de r≈1 (paso 0.05 entre 0.75 y 1.3) | ver lista exacta en el motor |
| N | {200, 400, 800, 1600} | 4 | |
| semillas | 12 | ≥12 pre-registrado | seeds 1000..1011 (independiente por combinación) |

Total combinaciones (ε×r×N) = 13 × 34 × 4 = 1768 puntos de grid, cada uno con 12
semillas × (REAL + NULL) = 24 corridas → 42.432 corridas totales.

**Perturbación dinámica:** F1-1 no pide barrido de ruido dinámico (eso es F1-5); la
perturbación de robustez de F1-1 es el barrido de r/ε/N mismo, con 12 semillas
independientes por punto (perturbación de condición inicial, lección CF-2 aplicada
vía cobertura amplia de grid, no una sola semilla "de más").

## 5. NULL (congelado)

Permutación del campo φ al final de la dinámica (`rng.permutation(phi)`,
`null=True` en `evolucionar()` del código base) — destruye la forma espacial,
conserva el histograma exacto de valores. Mismo φ inicial y misma secuencia de
pasos que el REAL, solo difiere en la permutación final.

## 6. Controles (congelados)

- **Control r=0 (H=0):** a ε>0, la difusión debe lavar P_real hacia ~0 (gate de
  validez del cruce, igual que `control_r0_ok()` del código base, umbral
  P_max=0.15).
- **Control ε=0:** a ε=0 no hay diferencia sembrada; P_real debe ser ≈0 a TODO r
  (no solo r=0). Si P_real(ε=0) > umbral en algún r, es FALLO del control, se
  reporta tal cual.

## 7. Criterio de PASS (congelado, tres lecturas — no se cambia tras ver datos)

1. **NULL cae:** en la banda r≫1 (r≥10), P_real ≫ P_null (z-score = (P_real_mean −
   P_null_mean) / sd_combinada ≥ 3 en al menos el 50% de los puntos ε>1e-6, r≥10).
2. **Control ε=0 → P≈0:** P_real(ε=0) < 0.05 en ≥95% de los puntos r del barrido a
   ese ε (si no, se reporta como violación del control, no se re-interpreta el
   resto).
3. **Control r=0 → lava:** P_real(r=0, ε>0) < 0.15 en promedio (mismo gate que
   `control_r0_ok()` del código base).

**Veredicto PASS del experimento F1-1 (pre-registrado):** persiste si (1) y (3) se
cumplen simultáneamente y (2) no se viola. Si el NULL nunca se separa del REAL
incluso en r≫1, es un NULO/hallazgo negativo — se reporta como tal, sin forzar
lectura positiva (regla del documento madre: "cualquier NEGATIVO es un hallazgo").

**No se ajusta este criterio después de correr.** El veredicto final de lectura
(qué significa para la batería) lo da CS con la curva cruda; este documento solo
fija el gate mecánico de PASS/FAIL de ESTE experimento.

## 8. Verificación cruzada (tres vías, T obligatorias)

(a) NULL — descrito arriba, es parte del criterio de PASS.
(b) Segundo observable — NO es responsabilidad de F1-1 (eso es F1-2, información
    mutua); F1-1 se valida contra su propio control ε=0 como segunda vía interna
    (además del NULL barajado, que es una vía distinta de "azar").
(c) Auditoría en disco — todos los JSON crudos (por N) + este protocolo + log de
    ejecución con timestamps quedan en
    `BATERIA_FUNDAMENTOS/F1_1_forma_magnitud/resultados/`.

## 9. Qué puede fallar (T6 — todo gate debe poder fallar)

- El NULL podría NO caer (P_real ≈ P_null en toda la banda) → hallazgo negativo.
- El control ε=0 podría dar P>0 espurio → indicaría fuga del observable, se reporta.
- El control r=0 podría no lavar (como pasó en el cs074 original antes del fix
  rcruz) → invalida la lectura del cruce, se reporta y NO se interpreta el resto.
- r* podría depender fuertemente de N → se reporta la dependencia, no se oculta.

## 10. Archivos de salida

- `resultados/F1_1_smoke_resultado.json` — corrida pequeña de validación (no es el
  resultado final).
- `resultados/F1_1_produccion_N{200,400,800,1600}_resultado.json` — barrido completo
  por N.
- `resultados/F1_1_produccion_resumen.json` — agregados + evaluación del criterio de
  PASS por punto y global.
- `resultados/F1_1_log_ejecucion.txt` — log con timestamps de inicio/fin de cada
  fase.

---
*Congelado. No editar después de este punto salvo para anotar FAIL explícito de
algún control (T3: no se cambia el juez tras el resultado).*
