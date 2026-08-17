# Re-verificación rápida tras bajar TOPE_EQ/TOPE_REC (sin rehacer el shim)

Con `TOPE_EQ=6.000` y `TOPE_REC=3.000` (antes 20.000/20.000), reproduje los
mismos 3 casos de `test_cruzado_v76.mjs` (semilla 7/parada, 7/inicio,
17/parada, puntos x=0,93 y x=1,4) directo con `correrBarridoV76` (sin el
shim, que es caro y no hacía falta rehacer):

| caso | x | asent_pasos (antes → ahora) | rec_mediana (antes → ahora) |
|---|---|---|---|
| 7/parada | 0,93 | 2900 → 2900 | 948 → 948 |
| 7/parada | 1,40 | 750 → 750  | 201 → 201 |
| 7/inicio | 0,93 | 2750 → 2750 | 944 → 944 |
| 7/inicio | 1,40 | 550 → 550  | 203 → 203 |
| 17/parada| 0,93 | 2900 → 2900 | 948 → 948 |
| 17/parada| 1,40 | 750 → 750  | 201 → 201 |

**Idénticos.** Esperable: ninguno de estos valores se acerca a los topes
nuevos (2900 y 948 están muy por debajo de 6.000 y 3.000 respectivamente).
Confirma que bajar los topes no afectó estos puntos ya validados contra el
script real — el motor queda listo para D/A'/B'/C con los topes nuevos.
