# Validación — EIT-3 Térmico κ_H, segunda batería (v7.4.1)

## Tarea 2 · Paso 0 — verificación obligatoria (shim del script real)

Método: `shim_html2.mjs` ejecuta el `<script>` REAL de v7.4.1.html en una sandbox
`vm` de Node (DOM mínimo falso), incluyendo captura real de los CSV que produce
`exportSweepCSV()` (mismo `Blob`/`URL.createObjectURL`/`a.click()` que en el
navegador, interceptados). `test_paso0.mjs` corre `runSweep()` de verdad (no una
reimplementación) con `desde=0.6 hasta=1.4 puntos=8 settle=40 measure=30 trazas=0`,
`powerBase=0.47 beta=0.94 sigma=6.8 noise=0.0079 band=1.105 tOpt=25 ptcTc=18
ptcSharp=4.1`, día/noche apagado.

| # | semilla | modo | resultado | esperado | ok |
|---|---|---|---|---|---|
| 1 | 7 | parada | CSV "X" (sha256 3e4c5123…) | — | — |
| 2 | 7 | parada (proceso nuevo) | idéntico a X, sha256 3e4c5123… | idéntico byte a byte | ✅ |
| 3 | 99 | parada | distinto de X, sha256 4eee570d… | distinto de X | ✅ |
| 4 | 7 | inicio | distinto de X, sha256 a29c2365… | distinto de X | ✅ |

**Paso 0 PASÓ**: la semilla es reproducible byte a byte, y tanto la semilla como
el modo de reinicio afectan el resultado (no hay una fuente de azar sin sembrar,
ni un modo que no haga nada).

## Tarea 3 · referencia numérica de Alexis

Con los mismos parámetros de la Tarea 2, se buscó la fila de `footprint` mínimo
en los CSV de semilla=7/modo=parada y semilla=7/modo=inicio.

- **Intento 1** (`ptcTc=18, ptcSharp=4.1`, los valores que declara el Paso 2 del
  encargo): parada → k=4 (x=1.057); inicio → k=3 (x=0.943). **No coincide** con
  la referencia (parada k=5 x≈1.171 / inicio k=4 x≈1.057).
- **Intento 2** (`ptcTc=25, ptcSharp=8`, los valores **por defecto** del HTML):
  parada → k=5 (x=1.1714); inicio → k=4 (x=1.0571). **Coincide exactamente** con
  la referencia de Alexis.

No es un defecto del instrumento: fue una ambigüedad de lectura de mi parte (el
encargo no repite los parámetros dentro de la tabla del Paso 0, y asumí que
usaba los del Paso 2 por estar en el párrafo de al lado). La verificación de
reproducibilidad de Paso 0 se hizo con los valores por defecto del panel, no con
los del experimento. **A partir de Paso 1 se usan los valores que sí pide el
encargo explícitamente para los experimentos: `ptcTc=18, ptcSharp=4.1`.**

## Tarea 4 · verificación bit-a-bit render vs no-render

`test_bit_identico2.mjs`: misma semilla, 2000 pasos, `updateSimulation()` (con
`renderAll()` real vía el shim) vs `stepHeadless()` (solo `pasoFisica()`).

| semilla | pasos | estado idéntico | campo idéntico | fieldSum |
|---|---|---|---|---|
| 42  | 2000 | ✅ | ✅ | 102471.95466430501 |
| 777 | 2000 | ✅ | ✅ | 102650.02782439598 |

Idéntico bit a bit en ambos casos. Los `fieldSum` coinciden exactamente con los
de la primera batería (mismos parámetros de prueba, misma física) — confirma
que la física no se tocó entre v7.3 y v7.4.1, tal como dice el encargo.

## Tarea 5 · motor2.mjs vs shim (runSweep real completo)

`test_cruzado2.mjs` corre `runSweep()` real (8 puntos, settle=40, measure=30,
`ptcTc=18 ptcSharp=4.1`) y compara CADA fila y CADA campo (en precisión
completa, leyendo `sweepRows` crudo del sandbox, no el CSV truncado a 4-5
decimales) contra `correrBarrido2()` de `motor2.mjs`.

| caso | semilla | modo | resultado |
|---|---|---|---|
| 1 | 1  | parada | ✅ idéntico (0 diferencias, 8 filas × 17 campos) |
| 2 | 1  | inicio | ✅ idéntico |
| 3 | 17 | parada | ✅ idéntico |

**motor2.mjs reproduce exactamente `runSweep()` real de v7.4.1**, incluyendo la
lógica nueva de reinicio condicionado por modo en AMBAS pasadas (calibración y
medición). Se usa como motor para Paso 1 y quedará listo para D/A'/B' cuando se
autorice correrlos.

## Velocidad medida

Con `motor2.mjs` (sin shim), barrido de 20 puntos, modo=parada:

| settle | tiempo (20 puntos) | pasos/s aprox |
|---|---|---|
| 150  | 4.30 s  | ~1600 |
| 300  | 6.37 s  | ~1570 |
| 600  | 11.44 s | ~1400 |
| 1200 | 19.14 s | ~1460 |
| 2400 | 35.55 s | ~1460 |
| 4800 (extra, ver Paso 1) | 63.1 s | ~1520 |

~1400-1600 pasos/s en esta máquina (comparable a los ~1146 pasos/s medidos con
`motor.mjs` en la primera batería; el encargo asume 2624 pasos/s de otra
máquina — ver Tarea 7 / presupuesto en `paso1_sensibilidad_settle.md`).
