# PROTOCOLO F1-5 — Robustez frente a ruido dinámico (no cosmético de semilla)

**Fechado:** 2026-07-24T09:32:25Z (antes de escribir/ejecutar el motor)
**Ejecutor:** CC (agente F1_5, batería paralela de 24 experimentos)
**Base física (NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Documento madre:** `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección F1-5 (líneas 101-108)

Este protocolo se congela ANTES de correr el motor. Si el experimento falla, se reporta
el FAIL — no se edita este documento después de ver resultados (regla T3).

---

## 1. Pregunta

¿La persistencia de una diferencia ínfima (ε) en el campo continuo sobrevive si, además
de la condición inicial, se inyecta **forzamiento estocástico en CADA PASO** de la
dinámica (no solo al sembrar el campo)? Esta es la prueba de robustez dinámica que
CF-1/CF-2 no tenían — ahí "10/10 semillas" resultó ser el mismo resultado casi
determinista repetido (lección de CF-2, sección 0 y 1 del documento madre).

## 2. Reutilización del núcleo validado (sin editar cs074_rcruz.py)

Se importan sin modificar las siguientes primitivas de `cs074_rcruz.py`:
`campo_inicial`, `paso_difusion`, `paso_expansion`, `medir_D`, `medir_pasos_lavado`,
`temperatura_fisica`, `detectar_cuantizacion`.

**Física añadida (propia de F1-5, no toca el archivo base):** en cada paso de la
dinámica, después de difusión + expansión, se suma ruido blanco gaussiano al campo:

```
phi = phi + amplitud_ruido * rng.standard_normal(N)
```

Esto perturba la DINÁMICA en cada paso (no solo la semilla ni la condición inicial),
que es exactamente lo que T7 exige y CF-1/CF-2 no hacían.

## 3. Observable

Se reutiliza literalmente la fórmula de `persistencia()` de rcruz:
`P = corr(φ, roll(φ,1)) · var(φ_final)/contraste0²`, con `contraste0 = std(φ)` en t=0.

**Advertencia metodológica pre-registrada (para no violar T6 con un control decorativo):**
cuando ε=0, `contraste0=0` por construcción, y la fórmula de P de rcruz devuelve 0.0
SIEMPRE por definición (línea `if contraste0 <= 0: return 0.0`), sin importar qué genere
el ruido dinámico. Eso volvería el control "ruido con ε=0" decorativo (no puede fallar).
Para que ese control SÍ pueda fallar, se registra ADEMÁS —para todos los puntos, no solo
ε=0— el coeficiente de autocorrelación solo:

```
c = corrcoef(φ_final, roll(φ_final,1))   (acotado [0,1], no normaliza por contraste0)
```

- Para ε>0: el veredicto primario usa **P** (comparable con el resto de la batería F1-x).
- Para ε=0 (control "ruido sin señal"): el veredicto usa **c** (P es 0 por definición y
  no sirve de control real). Se reporta P también, marcado como no-informativo en ε=0.

Esto se declara AQUÍ, antes de correr, precisamente para que quede registrado como
decisión de método y no como ajuste post-hoc del juez (T3).

## 4. Barrido (T7: perturbar la dinámica, no solo la semilla)

| Eje | Valores | Puntos |
|---|---|---|
| amplitud_ruido | `np.logspace(-6, -1, 8)` → 1e-6, 5.18e-6, 2.68e-5, 1.39e-4, 7.20e-4, 3.73e-3, 1.93e-2, 1e-1 | 8 (log, cumple ≥8 del documento madre) |
| ε | 0.0 (control obligatorio "ruido sin señal"), 1e-3, 1e-1 | 3 |
| r = H/D | 0.0, 0.3, 1.0, 3.0, 10.0 (cruza r≈1, subconjunto de R_TARGETS de rcruz) | 5 |
| semillas | 0..15 | 16 (cumple ≥16 del documento madre) |

N fijo = 200 (modo "producción" de rcruz). **No forma parte del barrido pedido por
F1-5** (el documento madre no lo incluye en el eje de F1-5, a diferencia de F1-1); se
fija por tratabilidad computacional y se declara aquí explícitamente — no se elige N
para producir un resultado (T1), es el mismo N que la corrida de producción de rcruz.

`pasos` fijo, calibrado UNA vez (sin ruido dinámico, igual que rcruz) con
`medir_pasos_lavado(N=200, eps=1e-2, semillas=12)` y margen 1.15×, y reutilizado en
TODA la grilla (mismo criterio "producción" de rcruz). Se calibra con ε=1e-2 (punto
medio de la grilla) para no privilegiar el extremo chico ni el grande.

Total de corridas: 8 × 3 × 5 × 16 × 2 (real+NULL) = 3840 corridas de dinámica.

## 5. NULL y control (dos verificaciones independientes, no confundir)

1. **NULL = permutación:** al final de la dinámica, `φ_null = rng.permutation(φ)`
   (destruye forma, conserva histograma). Se corre con la MISMA semilla que el REAL
   correspondiente (mismo camino estocástico hasta el barajado final), igual que rcruz.
2. **Control "ruido sin señal":** rama ε=0.0 del mismo barrido — ruido dinámico
   presente, sin diferencia ε sembrada. Juzgado por `c` (ver §3), no por `P`.

## 6. Criterio de PASS pre-registrado (congelado, tres lecturas — no se elige a posteriori)

a. **NULL:** P_real(ε>0) debe superar claramente a P_null en la banda con r que congela
   (r≥1), igual que en el resto de la batería. Si P_real ≈ P_null en toda la grilla, es
   NULO — se reporta como hallazgo, no se re-interpreta.
b. **Decaimiento suave:** P(amplitud_ruido) a (ε,r) fijos debe decaer de forma continua
   al subir la amplitud de ruido — SIN salto artificial entre puntos log consecutivos.
   Se define operacionalmente: salto = |ΔP| entre puntos consecutivos > 3× la dispersión
   inter-semilla (std entre las 16 semillas) en ese punto. Si hay salto, se REPORTA tal
   cual (no se aplica ningún suavizado/filtro a la curva cruda).
c. **Auditoría en disco:** JSON crudo con las 3840 corridas (P, c, z, std por semilla)
   entregado sin curar, para que quien no escribió el código pueda re-verificar.
d. **Control ε=0:** c_real(ε=0) ≈ c_null(ε=0) en TODA la grilla de amplitud_ruido — si el
   ruido dinámico por sí solo genera autocorrelación espuria que gana a su propio NULL,
   se reporta como contaminación del método (falsea el resto de la batería F1-x), no se
   descarta silenciosamente.

**PASS pre-registrado (documento madre, verbatim):** "existe un rango de ruido bajo el
cual la persistencia se mantiene; se reporta la curva P(amplitud_ruido) entera." Un
resultado sin ese rango (decae a NULL desde el primer punto, o nunca decae) es también
un hallazgo válido y se reporta como tal — no se auto-adjudica PASS/FAIL, el veredicto
lo da CS con la curva cruda.

## 7. Qué NO se hace

- No se edita `cs074_rcruz.py`.
- No se cambia este protocolo tras ver resultados.
- No se suaviza/filtra la curva P(amplitud_ruido) para "arreglar" un salto.
- No se auto-adjudica el veredicto final del experimento.
- No se toca topología ni se hacen commits.

## 8. Salidas en disco

- `PROTOCOLO_F1-5_PREREGISTRO.md` (este archivo).
- `F1_5_ruido_dinamico_motor.py` (motor, escrito DESPUÉS de este protocolo).
- `F1_5_ruido_dinamico_resultado.json` (crudo, las 3840 corridas + metadatos + timestamps).
