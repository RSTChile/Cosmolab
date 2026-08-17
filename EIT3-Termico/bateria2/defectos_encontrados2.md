# Defectos y decisiones — segunda batería EIT-3 Térmico κ_H (v7.4.1)

## 1. El defecto reportado sobre v7.4 (modo `ninguno` = `inicio`) está resuelto en v7.4.1

Confirmado leyendo el código: `runSweep()` ahora condiciona el reset inicial a
`modoReinicio!=='ninguno'` (antes era incondicional), y además la pasada de
calibración global ahora también respeta el modo (`if(modoReinicio==='parada')`,
algo que no pasaba en v7.3/v7.4). Verificado empíricamente en Paso 0
(`validacion2.md`, Tarea 2): semilla=7 en modo `parada` e `inicio` dan
resultados distintos y cada uno es reproducible por separado.

## 2. Ambigüedad de parámetros en Paso 0 (no es un defecto, ya resuelta)

La tabla de verificación del Paso 0 no repite los parámetros físicos (ptcTc,
ptcSharp, etc.), y estaban justo antes en el párrafo de Paso 2. Probé primero
con `ptcTc=18, ptcSharp=4.1` (los del experimento) y no coincidió con la
referencia (parada k=5 / inicio k=4); coincidió exactamente con los valores
**por defecto del HTML** (`ptcTc=25, ptcSharp=8`). Ver detalle en
`validacion2.md`, Tarea 3. A partir de Paso 1 en adelante se usan los valores
que el encargo pide explícitamente para los experimentos (`ptcTc=18,
ptcSharp=4.1`), no los del panel.

## 3. Hallazgo del Paso 1 (no es un bug de cálculo — ver detalle completo en `paso1_sensibilidad_settle.md`)

La zona donde la huella colapsa (x≈0.79–1.06, exactamente la frontera que mide
Experimento D) no estabiliza dentro de ningún settle probado (hasta 4800),
mientras que el resto del eje sí converge razonablemente rápido. Es consistente
con enlentecimiento crítico cerca de una bifurcación, no con un error de
fórmula (las funciones están validadas bit a bit contra el script real, ver
`validacion2.md`). **No se recomienda un settle para Experimento D / A' / B'
por esta razón — ver el reporte completo antes de decidir cómo seguir.**

## 4. Sin defectos de cálculo nuevos (v7.4.1)

Las cuatro verificaciones bit-a-bit (Paso 0 completo, render-vs-no-render,
motor2.mjs-vs-script-real en modo parada e inicio) pasaron todas sin
discrepancias. No se tocó ni se necesitó tocar el HTML — v7.4.1 se usó tal cual
se recibió.

---

# v7.5 — segunda ronda (D + A' + B' + C completos)

## 5. La predicción del encargo sobre D salió al revés — y hay una explicación de código, no un bug

El encargo predecía: "en modo `inicio` la posición del mínimo no se mueve
entre semillas; en modo `parada` sí se mueve". El resultado real
(`resumen_descriptivo2.md`): **ninguno de los dos modos se mueve** — la
posición del mínimo de huella es idéntica a precisión de punto flotante en
las 10 semillas de `parada` (x=0,8551) y en las 10 de `inicio` (x=1,3449).
Se repite en A' (30 semillas, modo=parada): mismo x=0,8551 exacto.

No parece ser un defecto de cálculo (las verificaciones bit-a-bit de v7.5
pasaron, ver `v75_validacion_y_presupuesto.md`) sino una propiedad del código:
`computeDaisyworld()` — la función que gobierna el crecimiento y muerte de
`black`/`white` (la vegetación) — **no llama al generador de azar (`rng()`)
en ningún punto**. Es una recursión determinística que depende solo de
`luminosity`, `tOpt` y `noise` (como parámetro fijo de la tasa de muerte, no
como término aleatorio) y de su propio estado anterior. Con `modo=parada`,
cada punto arranca de `black=0.18, white=0.14` fijos, así que la trayectoria
de vegetación en cada punto del eje es EXACTAMENTE igual entre semillas —
lo único que varía por semilla es el término estocástico de `Tf` y el ruido
del campo, que se promedian sobre la ventana de 120 pasos y aparentemente no
alcanzan a mover la huella lo suficiente como para cambiar qué punto del eje
de 60 paradas queda como mínimo. Esto no decide si el fenómeno es un
"hallazgo" o no — solo explica por qué la posición no se mueve en ninguno de
los dos modos, al revés de lo esperado.

## 6. Correlación huella↔entropía cambió de signo respecto a la primera batería

A' (modo=parada corregido): r=0,375±0,039. Primera batería (con el defecto de
arrastre): r=−0,236±0,073. No solo cambió la magnitud, cambió el signo. Dato
para el investigador principal, sin conclusión de mi parte.

## 7. Saturación alta en la grilla de B' — concentrada en `ptcSharp=6,0`

48 de 108 combinaciones (44%) superan el 10% de puntos saturados. Al filtrar
esas combinaciones de los promedios (como pide el encargo), el nivel
`ptcSharp=6,0` de la grilla queda prácticamente sin combinaciones limpias para
calcular su fila en la tabla de frontera-por-nivel — ver
`resumen_descriptivo2.md`. Las filas crudas de esas combinaciones siguen en
`experimento_Bprima_multivariable.csv`, marcadas, para poder auditarlas.

## 8. Throttling térmico sostenido durante la corrida de B'

`pmset -g therm` mostró `CPU_Speed_Limit=35` durante buena parte de la
corrida (14 procesos en paralelo, varias horas seguidas). El tiempo total
real (~12,7 h) terminó siendo mejor que la proyección pesimista (25-35 h)
pero peor que la optimista original (9,6-11 h) — dato operativo, no de
cálculo: no afecta la validez de los resultados (0 fallidos, 0 NaN/Infinity
en 1.130 barridos), solo el tiempo de pared.

## 9. Sin defectos de cálculo en v7.5

Todas las corridas (1.130 barridos) terminaron sin errores y sin
`NaN`/`Infinity`. El motor (`motor_v75.mjs`) sigue validado bit a bit contra
el script real (ver `v75_validacion_y_presupuesto.md`). No se tocó el HTML.
