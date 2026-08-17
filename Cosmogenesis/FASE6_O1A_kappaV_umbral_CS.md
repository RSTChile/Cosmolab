# Fase VI · O1-A — κ_V como métrica puente, y recalibración del umbral de Clase III

**Fecha:** 11-ago-2026 · Ejecuta: CC (Claude) · Tarea **O1-A** del
`FASE6_PLAN_EJECUCION_COMPLETA_CS.md` (propuestas #15 y #17 del equipo).

**No se corrió Phantom.** **No se generaron reglas nuevas.** **No se modificó ningún script existente**
(todo el código nuevo está en `cs090_fase6_o1a_kappav_umbral.py`). **No se hicieron commits.**
**No se declara cierre ni veredicto** — se reportan los números; la lectura es de Alexis.

Toda medición de diámetro/pendiente usada acá viene de `cs090_diam_corregido.py` vía
`cs090_fase6_remedicion_430.csv`, conforme a la regla adoptada en `FASE6_adopcion_diam_corregido_CS.md`.

---

## 0. En simple, con analogías

Dos preguntas, las dos sobre datos que ya estaban en disco.

**La primera (κ_V).** Tenemos dos formas de mirar cada regla. Una es **antes** de la gravedad: se mira
la forma del tejido de relaciones y se le saca un número, la *pendiente* (cuánto se encoge el "diámetro"
del tejido cuando lo miramos con lupa más gruesa). Eso es barato: 1,3 segundos por regla. La otra es
**después**: se corre Phantom, cae la gravedad, se forman los sumideros, y de ahí sale κ_V. Eso es caro.

La pregunta del equipo era: *¿κ_V nos está contando lo mismo que la forma del tejido?* Si sí, podría
servir de métrica única.

La analogía: la pendiente es **medirle las piernas al corredor antes de la carrera**; κ_V es
**tomarle el pulso al llegar**. La respuesta que dan los números es que **el pulso casi no habla de las
piernas** (correlación de rangos 0,48, y baja a 0,33 si se controla la perilla del diseño); en cambio el
pulso es **casi una copia del tiempo de la carrera** (correlación 0,90 con la masa en sumideros; 0,96
dentro de un bloque homogéneo). Y hay algo aún más básico: **para tomar el pulso hay que correr la
carrera.** κ_V se produce *dentro* de Phantom. Como atajo para ahorrar cómputo no puede funcionar por
construcción: el ahorro estaría en la dirección contraria (usar la pendiente, que es gratis, y no correr
Phantom).

**La segunda (el umbral).** La Clase III se define hoy como "pendiente mayor que 0,7". Ese 0,7 se fijó
antes, sin datos de gravedad. Ahora hay 80 corridas de Phantom que dicen cuánta masa terminó en los
sumideros, o sea una **verdad de campo física**. ¿Dónde habría que poner la raya para que separe mejor?

La analogía: es la **raya de altura mínima en un juego de parque de diversiones**. Uno puede ir subiendo
la raya y ver si los que entran se divierten más que los que quedan afuera. El resultado es doble y hay
que decir las dos mitades:

1. Subir la raya de 0,70 a **0,88** separa **bastante mejor** (la separación pasa de 0,43 a 0,73 en la
   medida usada), y la mejora **sobrevive** cuando se elige la raya con la mitad de los datos y se la
   prueba en la otra mitad (gana en el 95,3 % de las particiones).
2. **Pero no hay ninguna raya.** Cuando se pregunta "¿el escalón explica algo que la rampa no explique?",
   la respuesta es que **la recta continua sobre la pendiente le gana a todos los escalones**
   (R² = 0,663 la recta, contra 0,540 el mejor escalón y 0,182 el escalón oficial de 0,70). O sea:
   quien es más alto se divierte más *gradualmente*; poner una raya en cualquier lado es tirar
   información a la basura. El "0,88 óptimo" es simplemente **dónde conviene cortar si a uno lo obligan a
   cortar** — y encima está en una zona con pocos datos (5 de 76 corridas a ±0,05; percentil 89).

---

## 1. Datos, unión y decisiones de método

### 1.1 De dónde sale cada número

| fuente | qué aporta |
|---|---|
| `cs090_fase5b_TOTAL_40pares.csv` | las **80 corridas de Phantom** (40 pares): `fraccion_masa_en_sumideros`, `kappa_v_agregado`, `kcap`, `seed` |
| `cs090_fase6_remedicion_430.csv` | la **pendiente CORREGIDA** (y la vieja, para contraste) y los **diámetros por escala** medidos sobre la componente gigante |
| `cs090_fase6_outliers_phantom_metricas.csv` | **11 corridas de Phantom adicionales** (las reglas que descarrilaban), usadas **sólo** como chequeo externo |

### 1.2 Dos decisiones que cambian los números y por eso se declaran

**(a) La llave de unión es `seed`, no `rule_id`.** En el CSV de los 40 pares hay 3 reglas con sufijo
`v1fix`/`v2fix` cuyo `rule_id` no existe en el CSV de re-medición, pero cuya semilla sí. Y la semilla
determina íntegramente la regla (`GEN.generar_regla("A2","B0","C2", idx=0, seed=seed)`). Uniendo por
`rule_id` se perderían 3 corridas y — peor — `A2-B0-C2-r9` (seed 272702) y `A2-B0-C2-r9v2fix`
(seed 372702) son **grafos distintos con nombres casi iguales**. Uniendo por semilla: **80 de 80
emparejan**, cero pérdidas.

**(b) Hay 80 filas pero sólo 76 corridas distintas.** Cuatro reglas (`r6`, `r9`, `r19`, `r39`)
participan en dos pares cada una y su corrida de Phantom está **copiada** en el CSV. Para el diseño
pareado eso es correcto; para una correlación son **4 duplicados exactos que inflan el n**. Se reporta
todo por duplicado: **n=80** (el CSV tal cual) y **n=76** (una fila por corrida). Las conclusiones no
cambian entre las dos; el texto usa n=76, que es el número honesto de puntos independientes.

**Diámetro usado:** `diam_corregido[b=1]`, o sea el diámetro de la **componente gigante** del grafo a
escala fina (N=2000), tal como manda `cs090_diam_corregido.py`. Rango observado: 10 a 25.

---

## 2. ANÁLISIS 1 — ¿κ_V es un proxy de la geometría del grafo?

### 2.1 La tabla central (n=76 corridas independientes)

IC95 % del rho por bootstrap percentil (4000 remuestreos). "parcial | kcap" = correlación de Spearman
descontando la variable de diseño `kcap` (ver §2.3).

| par de variables | Spearman rho | IC95 % | p | Pearson r | p | parcial \| kcap |
|---|---|---|---|---|---|---|
| **κ_V vs pendiente CORREGIDA** (geometría) | **+0,480** | [+0,25, +0,67] | 1,1×10⁻⁵ | **+0,767** | 7,0×10⁻¹⁶ | **+0,201** (p=0,082) |
| κ_V vs pendiente vieja (con el bug) | +0,365 | [+0,10, +0,60] | 1,2×10⁻³ | −0,173 | 0,135 | +0,239 (p=0,038) |
| **κ_V vs diámetro corregido** (b=1) | **+0,509** | [+0,30, +0,69] | 2,7×10⁻⁶ | **+0,730** | 7,1×10⁻¹⁴ | **+0,057** (p=0,63) |
| **κ_V vs fracción de masa en sumideros** | **+0,902** | [+0,81, +0,96] | 1,1×10⁻²⁸ | **+0,934** | 1,0×10⁻³⁴ | **+0,758** (p=2×10⁻¹⁵) |
| *(referencia)* pendiente corr. vs masa | +0,629 | [+0,42, +0,78] | 1,1×10⁻⁹ | +0,814 | 3,7×10⁻¹⁹ | +0,467 (p=2×10⁻⁵) |
| *(referencia)* diámetro corr. vs masa | +0,629 | [+0,42, +0,78] | 1,1×10⁻⁹ | +0,767 | 7,0×10⁻¹⁶ | +0,215 (p=0,062) |
| *(referencia)* pendiente corr. vs diámetro corr. | +0,790 | [+0,66, +0,88] | 2,1×10⁻¹⁷ | +0,879 | 1,9×10⁻²⁵ | +0,714 (p=5×10⁻¹³) |

La tabla completa (n=80 y n=76, con IC y p de las dos correlaciones) está en
`cs090_fase6_o1a_kappav_correlaciones.csv`.

### 2.2 El criterio del analista (r > 0,6 con la geometría): se cumple, pero por 4 puntos

El criterio propuesto era *"si r > 0,6 con la geometría, κ_V sirve como métrica unificada"*. Con Pearson
sobre las 76, **r = 0,767 > 0,6: pasa.** Con Spearman, **rho = 0,480 < 0,6: no pasa.** Cuando las dos
medidas discrepan tanto, casi siempre hay **puntos de palanca**: unos pocos datos muy alejados en las
dos variables a la vez que estiran la recta. Se comprobó:

| muestra | n | Pearson r | Spearman rho |
|---|---|---|---|
| todas | 76 | **+0,767** | +0,480 |
| sin las 4 corridas con `kcap=4` | 72 | **+0,553** | +0,389 |
| sólo el bloque `kcap=5` (el más poblado) | 48 | **+0,536** | +0,329 |

Las 4 corridas con `kcap=4` están apartadas del resto en las dos variables (pendiente 1,09-1,26 contra
una mediana de 0,72; κ_V 1,29-1,41 contra una mediana de ~0,52). **Sacándolas, el r cae por debajo del
0,6 del criterio.** Con lo cual: por el criterio literal, κ_V pasa; con el criterio aplicado a un
subconjunto homogéneo, no pasa.

Para comparar, **κ_V vs masa dentro de ese mismo bloque `kcap=5`: r = +0,961, rho = +0,907.** Esa sí es
una relación que no depende de puntos de palanca.

### 2.3 El confound que atraviesa todo: `kcap`

`kcap` es una **perilla del generador de reglas** (el tope de grado/cupo). Correlaciona fortísimo con
las tres variables a la vez:

| variable | Spearman con `kcap` |
|---|---|
| fracción de masa en sumideros | **−0,844** |
| κ_V | **−0,749** |
| pendiente corregida | −0,486 |

Reparto de las 76 corridas: `kcap=4` → 4 corridas (masa media 0,150) · `kcap=5` → 48 (0,105) ·
`kcap=6` → 21 (0,083) · `kcap=7` → 3 (0,077). El diseño está **muy desbalanceado**.

Al descontar `kcap`, la relación **κ_V ↔ diámetro se evapora** (rho parcial +0,057, p=0,63) y la
relación **κ_V ↔ pendiente queda al borde de no distinguirse** (+0,201, p=0,082). En cambio
**κ_V ↔ masa sobrevive intacta** (+0,758, p=2×10⁻¹⁵). Dicho de otro modo: buena parte del vínculo
aparente entre κ_V y la geometría es que **las dos siguen a `kcap`**.

### 2.4 ¿Qué información tiene κ_V que la pendiente no tenga? (y al revés)

Correlaciones parciales de Spearman entre las tres, n=76:

| pregunta | rho parcial | p |
|---|---|---|
| κ_V ↔ masa, **descontando la pendiente** | **+0,879** | 1,5×10⁻²⁵ |
| pendiente ↔ masa, **descontando κ_V** | +0,517 | 1,7×10⁻⁶ |
| κ_V ↔ pendiente, **descontando la masa** | **−0,259** | 0,024 |

Y en regresión lineal sobre la masa (n=76):

| modelo | R² |
|---|---|
| sólo pendiente corregida | 0,663 |
| sólo κ_V | **0,872** |
| las dos juntas | 0,895 (la pendiente agrega +0,024 sobre κ_V; κ_V agrega +0,232 sobre la pendiente) |

La lectura literal de la tercera fila de la tabla de parciales: **descontada la masa, lo que queda de
κ_V va ligeramente en contra de la pendiente** (−0,26). No es que κ_V mida "geometría más algo": es que
κ_V mide, casi exclusivamente, **la respuesta gravitacional** — la misma cosa que la fracción de masa,
con la que forma prácticamente una recta (ver panel derecho de `cs090_fase6_o1a_fig1_kappav.png`).

### 2.5 El punto que decide la pregunta de "ahorrar cómputo"

Aunque κ_V correlacionara alto con la pendiente, **no ahorraría nada**: κ_V se calcula sobre el volumen
de Phantom *después* de la gravedad. Para tenerlo hay que haber corrido la simulación. La única
dirección en la que hay ahorro real es la contraria, y ya está medida: **la pendiente corregida, que
cuesta 1,3 s por regla, explica el 66 % de la varianza de la masa en sumideros** (R²=0,663; rho=+0,629),
que es lo que costaba ~11 s de Phantom por corrida más la generación de condiciones iniciales.

---

## 3. ANÁLISIS 2 — recalibrar el umbral de Clase III

### 3.1 Qué métrica se usa para "separa bien" y por qué

Para cada umbral candidato `u` se parten las 76 corridas en dos grupos (pendiente ≥ u vs < u) y se mide
cuánto se separan sus **fracciones de masa en sumideros**, con tres medidas a la vez:

| medida | qué es | por qué se incluye | defecto |
|---|---|---|---|
| **r punto-biserial** (`r_pb`) — **primaria** | Pearson entre el grupo 0/1 y la masa | es la única de las tres que **incorpora el balance del corte** (r_pb = d·√(p·q)): un corte que aísla 1 sola corrida extrema no puede ganar por serlo | menos interpretable "a ojo" |
| **AUC** | probabilidad de que una corrida del grupo alto tenga más masa que una del bajo (Mann-Whitney U / n₁n₀) | escala-libre, no supone normalidad, interpretable en una frase | **premia los cortes minúsculos**: aislar el punto más extremo da AUC=1,0 sin que signifique nada |
| **Cohen's d** | diferencia de medias / desvío agrupado | tamaño de efecto clásico, comparable con la literatura | igual defecto que AUC, y supone dispersiones parecidas |

También se reporta el p de Welch, pero **el p-valor no sirve para elegir umbral**: como se ve en la
tabla, *todos* los umbrales entre 0,58 y 0,88 dan p entre 10⁻⁴ y 10⁻⁶. El p dice "hay señal", no "acá
está el corte".

Se exige un mínimo de 8 corridas por grupo (más abajo, §3.5(a), se repite sin ese mínimo).

### 3.2 El barrido (n=76, `cs090_fase6_o1a_barrido_umbral.csv`)

| umbral | n alto | n bajo | masa media alto | masa media bajo | AUC | **r_pb** | Cohen d | p (Welch) |
|---|---|---|---|---|---|---|---|---|
| 0,58 | 65 | 11 | 0,1031 | 0,0807 | 0,890 | +0,442 | +1,38 | 2,0×10⁻⁵ |
| 0,60 | 62 | 14 | 0,1039 | 0,0820 | 0,888 | +0,476 | +1,38 | 2,2×10⁻⁶ |
| 0,62 | 60 | 16 | 0,1040 | 0,0843 | 0,836 | +0,451 | +1,22 | 9,1×10⁻⁶ |
| 0,64 | 55 | 21 | 0,1043 | 0,0884 | 0,743 | +0,399 | +0,96 | 8,1×10⁻⁵ |
| 0,66 | 52 | 24 | 0,1045 | 0,0899 | 0,707 | +0,380 | +0,87 | 1,8×10⁻⁴ |
| 0,68 | 45 | 31 | 0,1062 | 0,0907 | 0,729 | +0,425 | +0,94 | 3,8×10⁻⁵ |
| **0,70 (oficial)** | **41** | **35** | **0,1069** | **0,0916** | **0,720** | **+0,427** | **+0,93** | **7,6×10⁻⁵** |
| 0,72 | 38 | 38 | 0,1084 | 0,0914 | 0,759 | +0,477 | +1,07 | 1,8×10⁻⁵ |
| 0,74 | 32 | 44 | 0,1105 | 0,0922 | 0,783 | +0,507 | +1,18 | 2,4×10⁻⁵ |
| 0,76 | 26 | 50 | 0,1140 | 0,0925 | 0,826 | +0,573 | +1,45 | 1,2×10⁻⁵ |
| 0,78 | 16 | 60 | 0,1229 | 0,0937 | 0,897 | +0,668 | +2,17 | 2,1×10⁻⁵ |
| 0,80 | 14 | 62 | 0,1256 | 0,0941 | 0,921 | +0,687 | +2,41 | 2,6×10⁻⁵ |
| 0,82 | 11 | 65 | 0,1292 | 0,0949 | 0,926 | +0,677 | +2,58 | 1,4×10⁻⁴ |
| 0,84 | 10 | 66 | 0,1328 | 0,0949 | 0,980 | +0,720 | +3,03 | 2,8×10⁻⁵ |
| 0,86 | 9 | 67 | 0,1341 | 0,0953 | 0,977 | +0,704 | +3,03 | 8,6×10⁻⁵ |
| **0,88 (óptimo)** | **8** | **68** | **0,1381** | **0,0954** | **1,000** | **+0,735** | **+3,48** | **1,3×10⁻⁵** |

Los umbrales por debajo de 0,58 no aparecen porque la pendiente mínima observada es 0,479: cortar más
abajo deja menos de 8 corridas en el grupo bajo (a 0,40-0,47 el grupo bajo es literalmente vacío).

**Óptimo dentro de la muestra: 0,88 por las tres medidas a la vez** (r_pb +0,735 vs +0,427 del 0,70,
ganancia +0,308; AUC 1,000 vs 0,720; d 3,48 vs 0,93).

### 3.3 ¿Cae el óptimo en zona bien cubierta por datos? **No.**

| umbral | corridas dentro de ±0,05 | percentil del corte | hueco local entre vecinos |
|---|---|---|---|
| 0,70 | **23 de 76 (30 %)** | 46 % (parte la muestra casi por la mitad) | 0,0077 |
| 0,88 | **5 de 76 (7 %)** | **89 %** | 0,0184 |

El 0,88 está en **la cola derecha**: sólo 8 corridas quedan por encima, y el histograma
(`cs090_fase6_o1a_fig2_umbral.png`, panel superior derecho) muestra la masa de datos concentrada entre
0,55 y 0,85. **El 0,70 sí está en el corazón de la distribución.**

Y hay una razón de diseño para esto que conviene decir: **las 80 corridas no son una muestra al azar del
espacio de pendientes.** Se eligieron como pares "Clase I vs Clase III", o sea deliberadamente a un lado
y otro del 0,70 — por construcción están **densas alrededor de 0,70 y ralas arriba de 0,85**. La zona
donde cae el óptimo es, precisamente, la que este diseño menos exploró.

### 3.4 Validación — ¿la mejora sobrevive, o es sobreajuste?

Cuatro pruebas, porque una sola no alcanza.

**VALIDACIÓN 1 — mitades repetidas (2000 repeticiones, 1424 útiles).** En cada repetición se parte al
azar en dos mitades, se elige el umbral que maximiza r_pb en la **mitad A**, y se lo evalúa en la
**mitad B, nunca vista**; se lo compara con evaluar el 0,70 oficial en esa misma B.

| resultado fuera de muestra | valor |
|---|---|
| diferencia de r_pb (umbral optimizado − 0,70), media | **+0,220** |
| ídem, mediana | +0,233 |
| ídem, IC95 % | **[−0,050, +0,402]** ← **incluye el cero** |
| % de particiones en que el optimizado gana | **95,3 %** |
| umbral elegido en entrenamiento: mediana / IC95 % | **0,80** / [0,60, 0,84] |

Lectura honesta: **la mejora sobrevive en dirección** (gana 19 de cada 20 veces, con una ventaja media
grande), **pero no en precisión**: el intervalo del 95 % roza el cero, y **la posición del umbral es
inestable** (con la mitad de los datos el óptimo se elige entre 0,60 y 0,84; sólo el 43 % de las veces
cae a ±0,02 de la mediana).

**VALIDACIÓN 2 — dejar-uno-afuera (LOO).** Sacando cada una de las 76 corridas y re-optimizando:
**0,88 en 67 casos, 0,84 en 9.** O sea: el óptimo **no depende de un solo punto** — es robusto a
quitar cualquier corrida individual. (Esto mide estabilidad frente a una perturbación chica; la
Validación 1, que quita la mitad, es la prueba dura, y ahí sí se mueve.)

**VALIDACIÓN 3 — permutación (¿cuánto regala la libertad de elegir umbral?).** Se baraja la masa 5000
veces y se re-optimiza el umbral sobre puro ruido:

| | r_pb |
|---|---|
| máximo real barriendo umbrales | **+0,735** |
| máximo con la masa barajada: mediana | +0,124 |
| ídem, percentil 95 | +0,277 |
| ídem, máximo en 5000 | +0,491 |
| **p corregido por la búsqueda** | **0,0002** |
| para comparar, el umbral **fijo** 0,70 (no gasta grados de libertad) | r_pb +0,427, **p = 0,0004** |

O sea: **elegir umbral sobre ruido regala típicamente r_pb≈0,12 y como mucho 0,49** — bastante menos que
el 0,735 observado. La señal no es un artefacto de la búsqueda. Nótese que los dos p son casi iguales:
tanto el umbral optimizado como el fijo separan la masa mucho más de lo que separaría el azar.

**VALIDACIÓN 4 — chequeo externo con las 11 corridas de Phantom adicionales.** *Advertencia fuerte:*
esas 11 son justamente **las reglas que descarrilaban**, todas con pendiente corregida alta (0,594 a
1,443) y masa alta (0,077 a 0,157). **No son una muestra al azar** y no sirven para estimar tamaños de
efecto; sirven para ver si el ordenamiento se mantiene fuera del lote original. Dentro de esas 11 solas,
pendiente vs masa da **rho = +0,820 (p=0,002)**. Uniéndolas a las 76 (n=87): con umbral 0,70,
AUC=0,781 / r_pb=+0,498 / d=+1,15; con umbral 0,88, AUC=0,999 / r_pb=+0,834 / d=+3,85. El orden de
mérito entre los dos umbrales **se repite**, con la advertencia de sesgo puesta arriba de la mesa.

### 3.5 Tres controles que cambian cómo hay que leer el "óptimo"

**(a) ¿Es un máximo de verdad, o el borde de donde dejamos de buscar?** Repitiendo el barrido hasta 1,24
sin exigir tamaño mínimo de grupo (`cs090_fase6_o1a_barrido_umbral_sin_minimo.csv`): r_pb sube hasta
**+0,735 en 0,88** y **después baja** (+0,703 en 0,90; +0,693 en 0,92; +0,659 en 1,00; +0,325 en 1,10).
**Es un máximo interior, no un borde.** Pero es un **pico ancho**: 4 umbrales (0,84-0,90) están dentro
del 95 % del máximo. No hay un punto privilegiado, hay una meseta.

**(b) ¿Cuán preciso es el 0,88?** Bootstrap de la muestra (2000 remuestreos, re-optimizando cada vez):
mediana **0,84**, IC95 % **[0,60, 0,88]** — y ese 0,88 del extremo superior **es el techo de la
búsqueda** (con mínimo de 8 por grupo no se puede cortar más arriba), así que el intervalo real por
arriba es más ancho que lo que dice el número. El óptimo cae por debajo de 0,72 en el 4,0 % de los
remuestreos.

**(c) ¿Está cortando por pendiente o está cortando por `kcap`?** Como `kcap` correlaciona −0,84 con la
masa, cortar por pendiente podría ser cortar por `kcap` disfrazado. Control: repetir todo **dentro del
bloque `kcap=5`** (48 corridas, la perilla fija):

- Spearman pendiente-masa dentro del bloque: **+0,449 (p=0,0014)** — la relación **no desaparece**.
- El barrido dentro del bloque:

| umbral | n alto | masa media alto | masa media bajo | r_pb |
|---|---|---|---|---|
| 0,66 | 35 | 0,1060 | 0,1006 | +0,242 |
| **0,70** | 26 | 0,1087 | 0,0995 | **+0,468** |
| 0,74 | 23 | 0,1086 | 0,1007 | +0,401 |
| 0,78 | 12 | 0,1140 | 0,1013 | +0,559 |
| 0,80 | 10 | 0,1160 | 0,1015 | +0,601 |
| 0,84 | 6 | 0,1216 | 0,1021 | +0,658 |
| **0,88** | 4 | 0,1264 | 0,1025 | **+0,673** |

El mismo patrón (sube hacia arriba, el 0,70 no es el mejor) se reproduce con la perilla fija, con grupos
ya muy chicos. **El efecto no es sólo `kcap`**, aunque `kcap` amplifica todo.

### 3.6 El resultado que más importa: **el escalón pierde contra la rampa**

Modelos lineales sobre la fracción de masa (n=76):

| modelo | R² |
|---|---|
| **recta sobre la pendiente continua** | **0,663** |
| sólo el escalón en 0,70 (el clasificador oficial) | 0,182 |
| sólo el escalón en el óptimo 0,88 | 0,540 |
| recta + escalón óptimo | 0,704 (el escalón agrega **+0,041** sobre la recta) |

Traducido: **cualquier umbral tira información.** El escalón oficial en 0,70 conserva el 27 % de lo que
conserva la recta (0,182 / 0,663); el mejor escalón posible conserva el 81 %; y una vez que la recta está
en el modelo, agregarle el escalón sólo suma 4 puntos de R². Esto es coherente con lo que ya había
señalado `FASE6_reanalisis_azar_continuo_CS.md` — **la pendiente se comporta como una variable continua,
y la Clase III como una etiqueta impuesta sobre una rampa** — y ahora se lo puede cuantificar con la
pendiente corregida y con la verdad de campo física al lado.

---

## 4. Limitaciones declaradas

1. **El diseño no es una muestra al azar del espacio de pendientes.** Las 80 corridas son pares
   I-vs-III elegidos alrededor del 0,70; la zona > 0,85, donde cae el óptimo, está ralamente poblada
   **por construcción del diseño**, no por casualidad.
2. **`kcap` está muy desbalanceado** (4/48/21/3) y correlaciona −0,84 con la masa. El control dentro de
   `kcap=5` mitiga pero no elimina la duda; con `kcap=4` y `kcap=7` no hay potencia para nada.
3. **4 de las 80 filas son duplicados** de corridas ya presentes. Se trabajó con 76; con 80 los números
   cambian en el tercer decimal.
4. **Las 11 corridas externas son una muestra sesgada** (todas eran reglas descarriladas). Se usaron
   sólo como chequeo de ordenamiento, nunca para estimar efectos.
5. **La permutación responde "¿hay señal?", no "¿está el corte en 0,88?"**. Para lo segundo la evidencia
   es el bootstrap y las mitades, y ambos dicen que la posición es imprecisa.
6. **Nada de esto vuelve a correr la física.** Si Alexis quisiera cerrar la pregunta del umbral con
   datos propios, haría falta un lote de Phantom **muestreado uniformemente en pendiente** (por ejemplo
   0,45 a 1,30 en pasos parejos), no pares construidos alrededor del 0,70.

---

## 5. Archivos de esta tarea

**Script nuevo** (no modifica nada existente):

- `cs090_fase6_o1a_kappav_umbral.py` — todo el análisis: unión por semilla, correlaciones con bootstrap y
  parciales, barrido de umbrales con las tres medidas, las 4 validaciones, los 3 controles y los
  gráficos. Trae un autotest que verifica que la fórmula rápida de r_pb coincide bit a bit con
  `scipy.stats.pointbiserialr` antes de usarla en los bucles de remuestreo.

**Datos:**

- `cs090_fase6_o1a_datos_unidos.csv` — las 80 filas unidas (Phantom + re-medición corregida), con la
  marca `dup` de las 4 repetidas.
- `cs090_fase6_o1a_kappav_correlaciones.csv` — las 20 correlaciones del Análisis 1 (n=80 y n=76).
- `cs090_fase6_o1a_barrido_umbral.csv` — el barrido 0,58-0,88 con las tres medidas.
- `cs090_fase6_o1a_barrido_umbral_sin_minimo.csv` — el barrido completo 0,48-1,26 sin mínimo de grupo.
- `cs090_fase6_o1a_validacion_umbral.csv` — una fila con todos los resultados de validación.

**Gráficos:**

- `cs090_fase6_o1a_fig1_kappav.png` — κ_V contra pendiente, contra diámetro y contra masa.
- `cs090_fase6_o1a_fig2_umbral.png` — barrido, cobertura de datos, la rampa con las 11 externas, y la
  estabilidad de la elección del umbral.

No se corrió Phantom. No se editó ningún script congelado. No se hicieron commits. No se declaró cierre
ni veredicto: la lectura final es de Alexis.
