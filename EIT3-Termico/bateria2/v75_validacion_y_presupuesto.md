# v7.5 — Paso 0, referencia de recuperación, validación del motor y presupuesto real

## Paso 0 (rehecho específicamente contra v7.5.html)

`test_paso0_v75.mjs`, barrido chico en zona rápida (lum 1.2→1.4, 3 puntos, para
no pagar el costo de `medirRecuperacion` en esta prueba de reproducibilidad) —
semilla=7/modo=parada repetido da CSV idéntico byte a byte; semilla=99 y
modo=inicio dan CSV distinto. **PASÓ**, igual que en v7.4.1.

## Referencia de recuperación (Paso 1 del encargo v2, 24 puntos, lum 0.6→1.4, semilla=7, modo=parada)

Validación en dos etapas para no pagar el costo del shim (lento, ~150-250
pasos/s) en las 24 paradas completas:

1. **`test_cruzado_v75.mjs`**: `MotorV75`/`correrBarridoV75` vs el script real
   corrido en `shim_v75.mjs`, en 2 puntos (x=0.6, zona sin convergencia; x=1.4,
   zona rápida). Idéntico bit a bit, incluyendo `pasos_recuperacion=4024.2` y
   `convergio=0` en x=0.6 y `pasos_recuperacion=36.2`/`convergio=1` en x=1.4.
   Con esto el motor queda transitivamente validado contra el script real, y el
   resto de la referencia (24 puntos) se corrió con el motor rápido, no el shim.

2. **24 puntos completos** (`ref_recuperacion.mjs`), `ptcTc=18 ptcSharp=4.1`
   (los del experimento, no los del panel), `settle=300 measure=120`:

   | punto de referencia | esperado (Alexis) | obtenido | ¿coincide? |
   |---|---|---|---|
   | k=0..4 (x=0.6–0.739) | NO convergen, ~4000–5100 pasos | NO convergen, 4024–4969 pasos | ✅ |
   | k=5 (x=0.774) | 198 pasos | 176.0 pasos | ~parcial (mismo orden, ver nota) |
   | k=12 (x=1.017, el mínimo) | 48 pasos | 47.6 pasos | ✅ |
   | k=23 (x=1.4) | 70 pasos | 70.4 pasos | ✅ |

   **Coincide con `ptcTc=18, ptcSharp=4.1`** (no hizo falta probar los valores
   por defecto del panel esta vez). El único punto con diferencia apreciable es
   k=5 (176 vs 198, ~11%), justo el primer punto que sale de la zona sin
   convergencia — un punto sensible a condiciones iniciales por definición
   (borde de la transición), y el encargo no especifica con qué `settle` se
   midió la referencia (el warm-up antes de medir recuperación depende de
   `min(200,settle)`, así que un `settle` distinto al mío ahí podría explicar
   la diferencia). No se investigó más a fondo por no ser bloqueante: el resto
   de la referencia (incluyendo el mínimo exacto y el extremo derecho) coincide
   casi a la perfección.

## Validación del motor: PASÓ, sin discrepancias

Ambas verificaciones anteriores (2 puntos vía shim + réplica de `runSweep()`
completa) dan 0 diferencias. `MotorV75`/`correrBarridoV75` quedan listos para
correr D/A'/B' en cuanto se autorice.

## Presupuesto de cómputo real — HALLAZGO IMPORTANTE, no lancé nada

Medido con `motor_v75.mjs` (sin shim), eje completo de Experimento D
(luminosidad 0.25→1.95, 60 puntos), `settle=300 measure=120`, modo=parada:

| combinación | tiempo (1 barrido) | puntos sin converger (de 60) |
|---|---|---|
| baseline (β=0.94, tOpt=25, ptcSharp=4.1, powerBase=0.47) semilla=1 | 297.6 s | 18 |
| baseline, semilla=2 | 384.7 s | 18 |
| baseline, semilla=3 | 344.1 s | 18 |
| **esquina extrema de B'** (β=0.80, tOpt=28, ptcSharp=6.0) semilla=1 | **518.6 s** | **24** |

Promedio baseline ≈ **342 s/barrido**. La esquina extrema de la grilla de B'
ya sale **1.5× más cara** y con más puntos sin converger (24 vs 18) — **el
costo NO es uniforme a través de la grilla**, y solo probé 1 de las 48-108
combinaciones posibles fuera del baseline, así que no tengo caracterizado el
peor caso real.

En TODAS las combinaciones probadas, entre 18 y 24 de los 60 puntos del eje
(30–40%) no convergen dentro de `TOPE_REC=20000`, y quedan pegados a un
promedio de miles de pasos (no al tope completo de 5×20000=100000, porque
típicamente solo 1 de las 5 repeticiones pega el tope, no las 5 — ver detalle
en el código de `medirRecuperacion`).

### Proyección (extrapolando estas mediciones, con paralelismo de 14 procesos como en la batería anterior)

- **D** (10 semillas × 2 modos = 20 barridos, solo parámetros baseline):
  20 × 342 s ≈ 6.840 s serial ≈ **~8 min con 14 procesos** — barato, sin problema.
  *(No medí el modo `inicio` por separado — `medirRecuperacion` corre igual sin
  importar el modo, así que asumo orden de magnitud similar, pero no está
  medido y podría diferir porque el estado de entrada a cada punto es
  distinto).*
- **A'** (30 barridos, baseline): 30 × 342 s ≈ 10.260 s serial ≈ **~12 min con 14 procesos**.
- **B'**, usando un promedio conservador de 340–520 s/barrido (rango medido,
  sin garantía de que sea el peor caso):
  - Grilla completa (108 combinaciones × 10 semillas = 1.080 barridos): serial
    ≈ 134–156 h → **≈ 9.6–11 h con 14 procesos**. Esto ya iguala o supera el
    techo de 9,5 h que anticipaba la tabla de presupuesto ORIGINAL (que es de
    antes de v7.5 y no contempla el costo de `medirRecuperacion` en absoluto).
  - Recorte propuesto (tOpt en 2 niveles, ptcSharp en 2 niveles → 48
    combinaciones × 10 semillas = 480 barridos): serial ≈ 59–69 h → **≈ 4.2–5 h
    con 14 procesos**. Más manejable, pero el rango sigue siendo ancho porque
    solo tengo 2 puntos de la grilla medidos.

### Recomendación (sin lanzar nada todavía)

**No corrí D, A', B' ni C.** Con el recorte de B' (480 barridos) la proyección
total D+A'+B' ronda las 4.5–5.5 h con paralelismo — viable y del mismo orden
que anticipaba el encargo original pese al costo nuevo. Con la grilla completa
(1.080 barridos) la proyección se va a 9.6–11+ h, en el límite o por encima de
lo anticipado, con una muestra de la grilla demasiado chica (2 de 108 puntos)
para confiar en el número. Antes de lanzar cualquiera de las dos, recomendaría
—pero es decisión del investigador principal, no mía— medir 2-3 combinaciones
más de la grilla de B' (por ejemplo las 4 esquinas del recorte de 2×2 niveles)
para acotar mejor el rango antes de comprometer horas de cómputo.
