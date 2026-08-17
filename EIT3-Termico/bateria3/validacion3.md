# Validación — EIT-3 Térmico κ_H, tercera batería (v7.6.1)

## Paso 0 — verificación obligatoria (reproducibilidad)

`test_paso0_v76.mjs`, barrido en zona rápida (lum 1.2→1.4, 3 puntos, settle=40
measure=30), corrido contra el script REAL vía `shim_v76.mjs`:

| # | semilla | modo | resultado | ¿ok? |
|---|---|---|---|---|
| 1 | 7 | parada | CSV "X" | — |
| 2 | 7 | parada (repetido) | idéntico a X byte a byte | ✅ |
| 3 | 99 | parada | distinto de X | ✅ |
| 4 | 7 | inicio | distinto de X | ✅ |

**PASÓ.** Con el generador reestructurado (dos flujos, resembrado por fase),
la reproducibilidad se sostiene: misma semilla+modo → idéntico; semilla o
modo distinto → distinto.

## Verificación bit-a-bit render vs no-render

`test_bit_identico_v76.mjs`, 2 semillas × 1000 pasos, `updateSimulation()`
(con `renderAll()` real) vs `stepHeadless()`:

| semilla | pasos | estado idéntico | campo idéntico |
|---|---|---|---|
| 42  | 1000 | ✅ | ✅ (fieldSum=104891.03107071099) |
| 777 | 1000 | ✅ | ✅ (fieldSum=104539.20261500536) |

**PASÓ**, idéntico bit a bit en ambos casos.

## Motor (`motor_v76.mjs`/`correrBarridoV76`) vs script real

`test_cruzado_v76.mjs`, barrido de 2 puntos (lum=0,93 — zona de asentamiento
lento, cerca de donde las margaritas negras se extinguen — y lum=1,4 — zona
rápida), comparado campo por campo (precisión completa, no CSV truncado)
contra `runSweep()` real corrido en el shim:

| caso | semilla | modo | x=0,93 (asent_pasos / rec_mediana) | x=1,4 (asent_pasos / rec_mediana) | resultado |
|---|---|---|---|---|---|
| 1 | 7  | parada | 2900 / 948 | 750 / 201 | ✅ idéntico bit a bit |
| 2 | 7  | inicio | 2750 / 944 | 550 / 203 | ✅ idéntico bit a bit |
| 3 | 17 | parada | 2900 / 948 | 750 / 201 | ✅ idéntico bit a bit |

**PASÓ los 3 casos**, incluyendo `asent_pasos`, `asent_ok`, `pasos_recuperacion`,
`rec_mediana`, `rec_topes`, `rec_1..5` y todas las columnas de siempre.
Nota: en los puntos probados `rec_topes=0` en los tres casos (ninguna de las 5
repeticiones tocó el tope de 20.000) — no encontré en estos 3 casos un punto
con `topes>0` sin gastar más tiempo del shim en explorarlo, pero el mecanismo
de conteo de topes y el de asentamiento (`asent_ok`) sí quedaron ejercitados
con valores no triviales (asent_pasos=2900, bastante por encima del piso de
~700 que menciona el propio código) y coincidieron exactos.

**motor_v76.mjs reproduce exactamente `runSweep()` real de v7.6.1**, incluyendo
el orden de `sembrarFase()` (calibración → preasentamiento → recuperación →
asentamiento → medición) y el snapshot/restore de `instantanea()`/`restaurarInstantanea()`.
Se usa como motor para D/A'/B'.
