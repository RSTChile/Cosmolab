# Cierre de investigación — confound huella/acoplamiento y material para κ_H

Análisis sobre datos YA EXISTENTES (`bateria/`, `bateria2/`, `bateria3/` —
2.770 series/barridos en total, ~135.000 filas). No se corrió ninguna
simulación nueva. Script: `cierre_investigacion.mjs`. Solo estadística
descriptiva, sin concluir si hay "hallazgo".

## Tarea 1 — ¿huella y acoplamiento son casi-identidad, o se separan?

**No es una identidad algebraica forzada.** La correlación de Pearson entre
`huella` y `acoplamiento`, calculada por serie (barrido completo a lo largo
del eje de luminosidad), varía muchísimo según el régimen:

- **media = 0,458 ± 0,317** sobre las 2.770 series, **rango de −0,392 a 0,998**.
- 1.676 de 2.770 series (60,5%) tienen |r| < 0,50 — más de la mitad de los
  barridos corridos NO muestran la correlación fuerte que motivó la pregunta.
- Solo 197 series (7,1%) superan |r| > 0,98 (la zona de casi-identidad).

**Dónde se separan casi por completo (r ≈ 0):** las 20 series con |r| más
bajo son TODAS de `bateria3/Bprima`, y TODAS comparten `tOpt = 28` (la
temperatura óptima más alta de la grilla), con distintas combinaciones de
beta/potencia_base/ptcSharp. `tOpt=28` parece ser el factor dominante que
separa las dos referencias.

**Dónde son casi-identidad (r ≈ 0,998):** las series con |r| más alto
también se concentran en un régimen específico: `tOpt=22` (la temperatura
óptima más baja) con `beta=0,80`.

Esto sugiere (sin concluir) que **tOpt es el parámetro que decide si huella
y acoplamiento son redundantes o no** — con tOpt bajo son casi la misma
variable; con tOpt alto se separan casi por completo. El detalle fila por
fila (una fila por serie, con sus parámetros) está en
`confound_huella_acoplamiento_por_serie.csv`.

### Candidato de variable derivada: `brecha = huella − 8×(1−acoplamiento)`

**Limitación de datos, importante:** ninguna de las tres baterías exportó
`Tf` ni `targetTf` como columnas — solo las derivadas `huella=|Tf−abioticTf|`
y `acoplamiento=max(0,1−|Tf−targetTf|/8)`. Por eso **no se puede reconstruir
`targetTf−abioticTf` con signo** desde los CSV existentes; solo se puede
invertir la fórmula del acoplamiento para obtener la MAGNITUD
`|Tf−targetTf| = 8×(1−acoplamiento)` (válido mientras acoplamiento>0). La
`brecha` calculada acá es entonces `|Tf−abioticTf| − |Tf−targetTf|`
(diferencia de dos magnitudes), no la diferencia con signo de las dos
referencias — es lo más cercano reconstruible con lo que hay.

**Resultado: esta brecha NO es menos redundante — es, en promedio, MÁS
redundante con huella que el propio acoplamiento** (r media = 0,959 ± 0,039,
rango 0,827 a 1,000 — nunca baja de 0,83, mientras que acoplamiento solo por
sí solo llega a −0,39). Tiene sentido algebraicamente: `brecha` es
`huella` menos otra cosa, así que hereda buena parte de la varianza propia
de huella. **No la recomiendo como reemplazo** — queda documentada la razón
por la que no funciona, para no repetir el intento.

## Tarea 2 — material descriptivo para definir κ_H (sin proponer el valor)

Por cada batería/experimento, cada serie se normalizó a su propio rango de
huella (0=su mínimo/colapso, 1=su máximo/más vivo). "Colapsado" = filas en el
10% inferior de esa serie; "vivo" = filas en el 50% superior. Tabla completa
(media, desviación, min, p25, mediana, p75, max de `H_absLocal`, `H_rel`,
`H_noiseLocal` para cada zona × batería) en `distribucion_H_vivo_vs_colapsado.csv`.

Resumen (medianas, todas las baterías):

| batería/exp | H_absLocal vivo | H_absLocal colapsado | H_rel vivo | H_rel colapsado | H_noiseLocal vivo | H_noiseLocal colapsado |
|---|---|---|---|---|---|---|
| bateria/A | 0,960 | 1,656 | 4,229 | 4,266 | 4,334 | 4,339 |
| bateria/B | 0,954 | 1,641 | 4,212 | 4,266 | 4,332 | 4,339 |
| bateria2/D | 0,971 | 0,941 | 4,233 | 4,212 | 4,336 | 4,341 |
| bateria2/A' | 0,971 | 0,966 | 4,242 | 4,161 | 4,335 | 4,334 |
| bateria2/B' | 0,954 | 0,966 | 4,212 | 4,171 | 4,335 | 4,334 |
| bateria3/D | 0,966 | 0,934 | 4,250 | 4,213 | 4,332 | 4,340 |
| bateria3/A' | 0,966 | 0,954 | 4,245 | 4,223 | 4,336 | 4,339 |
| bateria3/B' | 0,966 | 0,960 | 4,241 | 4,232 | 4,334 | 4,338 |

Con datos: en `bateria/` (huella no monótona, colapso en V marcado) las
medianas de `H_absLocal` sí difieren notablemente entre vivo (~0,95) y
colapsado (~1,64-1,66). En `bateria2/` y `bateria3/` (con reinicio corregido
por parada) las medianas de las tres variables quedan muy cercanas entre
"vivo" y "colapsado" en casi todos los experimentos — la separación es mucho
más chica que en la primera batería. `H_noiseLocal` es prácticamente
constante (~4,33) en todas las zonas y baterías — no parece discriminar entre
vivo y colapsado en ningún caso.

No propongo ningún umbral de κ_H — la tabla completa (con p25/p75/min/max,
no solo medianas) está en el CSV para que el investigador principal decida.
