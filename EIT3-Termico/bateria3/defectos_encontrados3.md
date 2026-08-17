# Defectos y decisiones — tercera batería EIT-3 Térmico κ_H (v7.6.1)

## 1. Límite de validación de las dos baterías anteriores: el motor headless nunca recortó el eje

v7.6.1 declara formalmente `RANGO_EJE` (luminosidad limitada a 0,60–1,40, el
mismo límite que el `<input type="range" min="0.6" max="1.4">` de la
luminosidad tuvo SIEMPRE en el HTML, en todas las versiones). En el navegador
real, pedirle al barrido un valor fuera de ese rango hacía que el input lo
recortara en silencio — el CSV anotaba el valor pedido, no el que realmente
corrió.

`motor2.mjs`/`motor_v75.mjs` (bateria2, primera y segunda ronda de esta
segunda batería) usaban `state.luminosity = v` directo, sin ningún objeto
`<input>` real de por medio — nunca recortaron, así que corrieron la física
de verdad en luminosidad 0,25 y 1,95 en ambas baterías anteriores (bateria/ y
bateria2/). Tampoco lo habría detectado la validación cruzada contra el shim:
el shim tampoco usa elementos `<input>` reales con min/max (son objetos
planos), así que motor y shim coincidían entre sí sin coincidir con el
comportamiento real del navegador en este punto puntual.

Esto no invalida los resultados anteriores matemáticamente (el motor corrió
la física real en esos valores, y eso es válido como experimento), pero sí
significa que esas dos baterías corrieron un rango de luminosidad que el
instrumento interactivo real nunca permitió explorar directamente. Para esta
tercera batería no hace falta corregir nada — Alexis decidió usar el rango
0,60–1,40 que v7.6.1 ya declara como válido, así que el recorte nunca se
activaría de todas formas.

## 2. Sin defectos de cálculo en v7.6.1

Las tres verificaciones bit-a-bit (Paso 0, render-vs-no-render, motor vs
script real en 3 casos incluyendo un punto de asentamiento lento) pasaron sin
discrepancias. Detalle en `validacion3.md`. No se tocó el HTML.

## 3. Costo real ~1,9× más caro que v7.5 — detenido antes de lanzar D/A'/B'

`asentarHastaEquilibrio` (nuevo en v7.6.1) se suma AL COSTO de
`medirRecuperacion`, antes de él, en cada punto de cada barrido. Medido:
~667 s/barrido baseline (vs ~342 s/barrido en v7.5). Proyección total para
D+A'+B'(grilla completa) ≈ 16,5 h con 14 procesos en paralelo — por encima
del umbral de 6h para seguir sin preguntar. Detenido, reportado antes de
correr la parte cara. Ver `v76_validacion_y_presupuesto.md` para el detalle
completo y la proyección.

## 4. TOPE_EQ y TOPE_REC bajados para esta batería — decisión deliberada, NO un cambio al instrumento

`correr_barrido_v76.mjs` corre con `TOPE_EQ=6.000` y `TOPE_REC=3.000` en vez
de los 20.000/20.000 del HTML real (`ET3-Termico_v7.6.1.html`, sin tocar). Es
una decisión de esta batería únicamente, tomada con respaldo de datos, no un
defecto ni una corrección al instrumento:

- Muestreé 8 combinaciones de la grilla de B' (variando tOpt/ptcSharp/beta,
  incluida una esquina que combina varios extremos a la vez: tOpt=28,
  ptcSharp=6,0, β=0,80, potencia_base=0,30), barrido completo de 60 puntos
  cada una, guardando la distribución CRUDA (480 `asent_pasos`, 2.400 `reps`
  individuales de `medirRecuperacion`), no solo el resumen.
- `asent_ok=0` (no asentó ni con el tope actual de 20.000): 0/480. El máximo
  genuino de asentamiento fue 4.000 pasos. `TOPE_EQ=6.000` da 0 casos
  reclasificados en la muestra, con 50% de margen sobre ese máximo.
- De los 2.400 reps, 300 (12,5%) topan a 20.000 (dato válido de bifurcación,
  no error). De los que sí convergen, el máximo fue 2.101 pasos.
  `TOPE_REC=3.000` da 0 casos reclasificados, con 43% de margen.
- Ningún punto de la muestra —ni la esquina más extrema armada a propósito—
  mostró un caso genuinamente lento-pero-convergente por encima de esos
  umbrales. El detalle completo (tablas de reclasificación por candidato,
  percentiles) está en `topes_investigacion.md`.
- Riesgo residual reconocido: la muestra cubre 8 de las 108 combinaciones de
  B'. No hay garantía absoluta de que ninguna de las 100 combinaciones no
  muestreadas tenga un caso más lento que el máximo observado — el margen de
  43-50% es la mitigación, no una prueba exhaustiva. Alexis aceptó ese riesgo
  con esa información.
- Verificado antes de lanzar la batería completa (no se rehizo la validación
  cruzada entera): los 3 casos de `test_cruzado_v76.mjs` y el punto de
  asentamiento lento (lum≈0,93, `asent_pasos`=2.900) siguen dando los mismos
  valores con los topes nuevos, porque ninguno se acerca a 6.000/3.000 — ver
  `validacion3_topes_bajados.md`.

## 5. Resultado real: 1.130/1.130 barridos, 0 fallidos, 15,41 h

La corrida completa de D+A'+B' terminó sin errores (`TERMINADO_v76`,
`orquestador_v76.log`). Tardó 15,41 h con 14 procesos — más que la proyección
optimista de ~6,1h (los topes bajados sí redujeron el costo por barrido, pero
`CPU_Speed_Limit` volvió a caer a 39% por throttling térmico sostenido,
confirmando el mismo fenómeno de las dos baterías anteriores). Sin
reclasificaciones detectables por los topes bajados: 0 filas con `NaN`/`Infinity`
en los 64.800+1.800+1.200 registros.

## 6. Saturación en B' mucho menor que en `bateria2`: 2/108 combinaciones (1,9%)

Con el eje angosto (0,60-1,40, el que v7.6.1 declara válido), solo 2 de las
108 combinaciones de B' superan el 10% de puntos con `saturacion_sensor=1` —
muy por debajo del 44,4% (48/108) que se vio en `bateria2` con el eje ancho
(0,25-1,95). Consistente con que el eje angosto es el rango donde el sensor
PTC efectivamente responde bien; el eje ancho anterior empujaba el sistema a
extremos que saturan el sensor con más frecuencia. No es un defecto — es
esperable dado el cambio de eje, y vale la pena tenerlo presente al comparar
ambas baterías.

## 7. Correlación huella↔entropía y su barajado: mucho más débiles que en `bateria2`

En D, A' y B' de esta batería, la correlación de Pearson entre huella y
entropía de la conducta sale sistemáticamente más chica y con signo menos
consistente que en `bateria2` (A': 0,008±0,134 acá vs. 0,375±0,039 en
`bateria2`), y el barajado (Experimento C) muestra que el r real de A' es en
promedio indistinguible del azar (percentil ~52) — lo opuesto al patrón de
`bateria2` (percentil ~99,7). Dato para el investigador principal, sin
conclusión de mi parte sobre a qué se debe el cambio (candidatos obvios que
NO investigué: el eje distinto, o el efecto acumulado de los tres cambios de
mecanismo de v7.6.1 sobre la trayectoria térmica). Detalle completo en
`resumen_descriptivo3.md`.
