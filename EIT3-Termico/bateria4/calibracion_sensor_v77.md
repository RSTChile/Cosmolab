# Calibración del sensor PTC para la física real (v7.7) — Paso 0 de esta batería

Con la física Stefan-Boltzmann, la zona fértil del eje 0,6-1,4 opera con Tf
entre ~29°C y ~35°C (medido en barrido de prueba, ver
`recalibracion_fisica_v77.md`). Los valores `tc_ptc=18, exponente_ptc=4,1`
usados en las baterías 2 y 3 estaban calibrados para la banda angosta vieja
(17-48°C) y no se reusan a ciegas.

## Hallazgo previo a la exploración: el cambio a Kelvin comprime la sensibilidad

`ptcResponse()` pasó de `ratio=temp/ptcTc` (Celsius) a `ratio=(temp+273)/(ptcTc+273)`
(Kelvin) en v7.7, para evitar que temperaturas negativas degeneraran el
sensor. Efecto colateral: con valores absolutos grandes (~273-373 K), un
mismo `exponente_ptc` da MUCHA menos discriminación que antes — un salto de
6°C alrededor de 300K es un cambio de ratio de apenas ~2%, contra ~24% que
daba el mismo salto alrededor de 25°C en Celsius. Hacía falta re-explorar,
no solo elegir un `tc_ptc` nuevo con el `exponente_ptc` viejo.

## Exploración (barridos de 24 puntos, eje 0,6→1,4, modo=parada, semilla=1)

**Saturación** (`ptc_saturado=1`), 9 combinaciones medidas:

| tc_ptc | exponente_ptc | % saturado | extintos |
|---|---|---|---|
| 20 | 4,1 | 8,3% | 0 |
| 20 | 8 | 12,5% | 0 |
| 20 | 12 | 12,5% | 1 |
| 20 | 16 | 12,5% | 0 |
| 25 | 4,1 | 12,5% | 1 |
| 25 | 8 | 12,5% | 0 |
| 25 | 12 | 12,5% | 1 |
| 25 | 16 | 12,5% | 0 |
| 30 | 4,1 | 12,5% | 0 |

Todas las combinaciones dan saturación baja (8-13%, muy por debajo del 44%
que dejó `tc_ptc=18/exponente_ptc=6,0` en la batería 2) — pero saturación
baja no confirma que el sensor discrimine bien, solo que no está pegado en
los topes duros (1,2 o 0,05).

**Discriminación real** (rango de `powerLive`/`mult` en el mismo barrido):

| tc_ptc | exponente_ptc | powerLive [min,max] | mult [min,max] | rango de mult |
|---|---|---|---|---|
| 20 | 4,1 | [0,296, 0,564] | [0,009, 0,087] | 0,078 |
| 20 | 8 | [0,198, 0,564] | [0,026, 0,180] | 0,155 |
| **20** | **16** | **[0,091, 0,564]** | **[0,030, 0,296]** | **0,266** |
| 25 | 8 | [0,224, 0,564] | [0,006, 0,154] | 0,148 |
| 30 | 8 | [0,253, 0,564] | [0,000, 0,125] | 0,125 |

`exponente_ptc` más alto da más discriminación (esperable, dado el
achatamiento por Kelvin) sin empeorar la saturación (se mantiene en 12,5%
igual que `exponente_ptc=8`). **`tc_ptc=20, exponente_ptc=16`** da el rango
más ancho de los 5 candidatos comparados en detalle (factor >6× en `mult`,
el doble que `exponente_ptc=8`), con 0 extinciones y saturación igual de baja
que el resto.

## Decisión

**`tc_ptc=20, exponente_ptc=16`** para D, A' y para el nivel central de la
grilla de B'. `exponente_ptc=16` es el máximo actual del slider — no se subió
más allá porque el control no lo permite; si en el futuro se quisiera más
discriminación aún, ampliar el rango del slider (como ya se hizo con
`ptcTc`) sería el paso siguiente, documentado acá para no perderlo.

## Grilla de B' — los 3 niveles de exponente_ptc de las baterías anteriores ya no sirven

`{3,0 · 4,1 · 6,0}` (batería 2/3) fueron calibrados para el sensor viejo en
Celsius. Con Kelvin, los tres quedarían en la zona de MUY baja discriminación
(equivalente al `4,1` de la tabla de arriba, la fila con menor rango de
`mult`). **Nueva grilla de `exponente_ptc` para B': `{8 · 12 · 16}`** — cubre
desde discriminación moderada hasta la calibración elegida como centro,
manteniendo el resto del diseño de B' sin cambios (persistencia β, t_óptima,
potencia_base con los mismos niveles que bateria2/bateria3).
