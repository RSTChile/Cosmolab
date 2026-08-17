# v7.6.1 — validación y presupuesto real (DETENIDO antes de lanzar D/A'/B')

## Validación: PASÓ todo, sin defectos bloqueantes

Ver `validacion3.md` para el detalle completo:
- Paso 0 (reproducibilidad byte a byte): PASÓ.
- Bit-a-bit render vs no-render: PASÓ (2 semillas × 1000 pasos, idéntico).
- `motor_v76.mjs`/`correrBarridoV76` vs script real (`runSweep()` completo vía
  shim), 3 casos incluyendo un punto de asentamiento lento (lum=0,93,
  asent_pasos=2900): PASÓ, 0 diferencias en todos los campos, incluyendo el
  orden exacto de `sembrarFase()` (calibración → preasentamiento →
  recuperación → asentamiento → medición) y el snapshot/restore de
  `instantanea()`/`restaurarInstantanea()`.

El motor está listo para D/A'/B'/C. Lo que sigue es solo presupuesto de
cómputo — no lancé la batería completa.

## Costo real medido (eje 0,60→1,40, 60 puntos, settle=300, measure=120, modo=parada)

| caso | tiempo (1 barrido) | puntos sin converger recuperación (de 60) |
|---|---|---|
| baseline (β=0,94, tOpt=25, ptcSharp=4,1, powerBase=0,47) semilla=1 | 630,2 s | 8 |
| baseline, semilla=2 | 704,2 s | 8 |
| esquina de B' (β=0,80, tOpt=28, ptcSharp=6,0) semilla=1 | 826,1 s | 10 |

Promedio baseline ≈ **667 s/barrido** — **~1,9× más caro que el baseline
equivalente de v7.5 (342 s/barrido)**, consistente con que ahora se suma el
costo de `asentarHastaEquilibrio` (rango medido: 300 a 3.600 pasos) ANTES de
`medirRecuperacion`, que antes no existía. La esquina de grilla probada sale
1,24× más cara que el baseline (menos dispersión relativa que en la ronda
anterior de v7.5, donde la esquina extrema llegó a 1,5×, pero la base ya es
más cara de por sí).

`asent_ok=0` no apareció en ninguno de los tres barridos completos (0/60 en
los dos casos donde lo miré) — el asentamiento sí llega a converger dentro del
tope de 20.000 en este eje más angosto (0,60→1,40), a diferencia de lo que
podría pasar en un eje más amplio.

## Proyección para D + A' + B' (grilla completa) + C

- **D** (20 barridos, baseline): 20 × 667 s ≈ 13.340 s serial.
- **A'** (30 barridos, baseline): 30 × 667 s ≈ 20.010 s serial.
- **B'** (1.080 barridos, grilla completa): usando un promedio conservador
  entre 667 y 826 s/barrido (~740 s), 1.080 × 740 ≈ 799.200 s serial.
- **Total serial** ≈ 832.550 s ≈ **231 horas**.
- **Con 14 procesos en paralelo** (sin contar throttling térmico, que la
  ronda anterior mostró que puede aparecer con cargas sostenidas de varias
  horas): ≈ **16,5 horas**.

**Esto supera el umbral de ~6 horas que fijaste para seguir sin volver a
preguntar.** Con el antecedente de la ronda anterior (proyección optimista de
9,6-11h que terminó en ~12,7h real por throttling, después de haber tocado un
peor caso teórico de 25-35h), un presupuesto que ya arranca en ~16,5h optimista
podría terminar bastante más arriba si el throttling se repite.

**No lancé D, A' ni B'.** Todo lo demás (motor validado, shim, scripts de
medición) está listo en `bateria3/` para correr en cuanto se autorice —
incluido el diseño de jobs (mismo patrón que `bateria2/orquestador_v75.mjs`,
solo cambiaría el eje a 0,60→1,40 y el motor importado).
