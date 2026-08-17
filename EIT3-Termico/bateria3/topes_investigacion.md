# Investigación de TOPE_EQ / TOPE_REC — v7.6.1 (antes de decidir grilla de B')

**No se aplicó ningún cambio.** Esto es solo la medición pedida, para decidir
entre bajar los topes, recortar la grilla, o combinar ambas.

## Método

8 combinaciones del espacio de B' (variando tOpt, ptcSharp y beta, los
parámetros con más chance de mover la zona lenta — más una esquina que
combina varios extremos a la vez), cada una con un barrido COMPLETO de 60
puntos en el eje 0,60→1,40, semilla=1, modo=parada, usando `motor_v76.mjs` ya
validado. Guardé la distribución cruda completa: los 60 `asent_pasos` y los
60×5=300 `reps` de `medirRecuperacion` de cada barrido (no solo el resumen).

Combinaciones: baseline (tOpt=25,ptcSharp=4,1,β=0,94,pB=0,47), tOpt=22,
tOpt=28, ptcSharp=3,0, ptcSharp=6,0, β=0,80, β=0,98, y una esquina combinada
(tOpt=28, ptcSharp=6,0, β=0,80, pB=0,30). Total: 480 puntos, 2.400 reps.

## TOPE_EQ (asentarHastaEquilibrio) — margen amplio, seguro bajarlo

`asent_ok=0` (no asentó ni con el tope actual de 20.000): **0 de 480** en las
8 combinaciones. De los que sí asentaron: **mínimo 300, máximo 4.000, media
1.769, p99=3.600 pasos.**

| candidato TOPE_EQ | casos reclasificados (lento-pero-convergente > tope) |
|---|---|
| 3.000 | 23/480 (4,8%) |
| 3.500 | 10/480 (2,1%) |
| **4.000** | **0/480 (0,0%) — es exactamente el máximo observado** |
| 5.000 – 15.000 | 0/480 |

**No encontré NINGÚN caso de asentamiento genuinamente lento (por encima de
4.000 pasos) en 480 puntos de la grilla, incluida la esquina más extrema
combinada.** Bajar `TOPE_EQ` a 4.000 ya da cero reclasificación en esta
muestra; con margen de seguridad (la muestra es de 8 combinaciones, no las
108), recomiendo **TOPE_EQ = 6.000** (50% por encima del máximo observado).

## TOPE_REC (medirRecuperacion) — margen todavía más amplio

De los 2.400 reps individuales: **300 (12,5%) tocan el tope actual de
20.000** — esto es dato válido de ralentización crítica cerca de la
bifurcación, no error de medición (tal como señalaste). De los que SÍ
convergen: **mínimo 33, máximo 2.101, media 503, p99=1.798 pasos.**

| candidato TOPE_REC | casos reclasificados (convergente genuino > tope) | ya topaban a 20.000 (sin cambio) |
|---|---|---|
| 2.000 | 10/2.400 (0,42%) | 300 (12,5%) |
| **2.101** | **0/2.400 — es exactamente el máximo observado** | 300 |
| 2.500 – 15.000 | 0/2.400 | 300 |

**Tampoco encontré reps genuinamente lentos-pero-convergentes por encima de
2.101 pasos, en ninguna de las 8 combinaciones.** El 12,5% que topa lo sigue
haciendo igual con un tope más bajo (es la misma clasificación: no convergió),
solo que más barato de detectar. Recomiendo **TOPE_REC = 3.000** (43% de
margen sobre el máximo observado).

## Impacto en el costo

`asentarHastaEquilibrio` no cambia de costo con `TOPE_EQ=6.000` (ningún caso
en la muestra se acerca a ese tope, así que sigue corriendo igual de rápido
que ahora). El ahorro real viene de `TOPE_REC=3.000`: los pasos totales de
recuperación en la muestra bajan de 7.056.085 a 1.956.085 (**27,7% del
actual**) — el 12,5% de reps que hoy queman 20.000 pasos cada uno pasarían a
quemar 3.000.

Como `asentarHastaEquilibrio`+`medirRecuperacion` juntos son ~97% de los
pasos de un barrido completo (settle+measure+calibración son solo ~30.000
pasos de los ~988.000 que promedia cada barrido en esta muestra), el barrido
completo bajaría a **~37% del tiempo actual** con estos dos topes nuevos.

### Proyección recalculada para D+A'+B' (grilla completa, 1.130 barridos)

- Con los topes actuales: ~16,5 h optimista (14 procesos, sin throttling) —
  ya reportado la ronda anterior.
- **Con TOPE_EQ=6.000 y TOPE_REC=3.000: ~16,5 h × 0,37 ≈ 6,1 h optimista.**
  Justo en el borde del umbral de 6h. Con el antecedente de throttling
  térmico sostenido (confirmado otra vez en esta misma investigación:
  `CPU_Speed_Limit` volvió a caer a 35% corriendo las 8 muestras en paralelo
  durante ~24 minutos), el tiempo real probablemente supere ese óptimo, pero
  la reducción relativa (~63%) debería sostenerse aunque el throttling
  aparezca, porque throttling y cantidad de pasos son factores independientes
  que se multiplican.

## Recomendación

Los datos respaldan bajar ambos topes con reclasificación cero medida (en 8
de 108 combinaciones, con la esquina más extrema del espacio incluida). No es
una certeza absoluta para las 100 combinaciones no muestreadas, pero el
patrón es consistente y con margen holgado (50% y 43% respectivamente) sobre
el máximo observado en la muestra más adversarial que pude armar.

Dado que aun con los topes bajados la proyección (~6,1h optimista) queda
justo en el límite y el historial de throttling es real, **mi lectura es que
la opción (c) — combinar ambas — es la más prudente**: bajar los topes (dan
margen real y gratis) Y de todas formas recortar la grilla de B' al esquema
de siempre (tOpt en 2 niveles, ptcSharp en 2 niveles → 480 barridos en vez de
1.080), lo que dejaría la proyección optimista en el orden de 3-3.5h y con
más colchón frente al throttling. Pero esto ya es una decisión de diseño, no
un hallazgo de los datos — quedo a la espera de qué combinación prefieren.
