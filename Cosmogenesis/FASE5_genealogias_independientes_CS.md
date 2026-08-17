# ¿El 45% de A2-B0-C2 depende de la familia de reglas, o del día en que se la mira? — genealogías independientes

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude) · Paso 4 del roadmap del equipo para A2-B0-C2 ("réplicas de
genealogías A2 verdaderamente independientes, no sólo semillas de la misma genealogía"). Continúa la línea
F5-C2-C (`FASE5_presupuesto_emergente_CS.md` → `..._soporte_local_CS.md` → `..._mecanismo_aislado_CS.md` →
`..._matriz_2x2_completa_CS.md` → `FASE5_control_azar_elastico_CS.md`), sin tocar ninguno de esos 5 archivos
ni los módulos congelados (`cs090_fase5_generador.py`, `cs090_fase5_motor.py`, `cs090_fase5_clasificador.py`,
`cs090_fase5_mecanismo_aislado.py`). El único archivo de código de esta tarea es
`cs090_fase5_genealogias_independientes.py` (ya escrito, no modificado por este informe). No se corrió
Phantom. No se hicieron commits de git.

## 0. La pregunta, en simple: ¿otra red social, o el mismo día distinto?

Todas las tareas anteriores de la línea F5-C2-C generaron su lote de 20 reglas desde el **mismo punto de
partida del generador aleatorio** (`seed_base`): variaban las 20 semillas individuales dentro de ese lote
(20 "días" distintos de la misma red social), pero nunca cambiaron la red social entera. Es la misma
distinción que el proyecto ya usó con CS073: variar `seed_layout` de una topología fija no es lo mismo que
generar topologías genuinamente independientes.

**Analogía:** una "genealogía" acá es como tomar 4 redes sociales completamente distintas y separadas —
cada una con su propio conjunto de 20 familias de parámetros (K, J, ruido, grado medio, tope de amigos,
semilla) generadas desde cero. Una "realización" son los distintos días/estados dentro de UNA MISMA red
social (las 20 semillas ya conocidas dentro de un `seed_base` fijo, que ya varían: unos días caen en Clase
I, otros en Clase III). Esta tarea corre 4 redes sociales genuinamente distintas y mide si el ~45% de "días
extendidos" (Clase III) se sostiene entre redes, o si es un rasgo de una red social particular.

## 1. Las 4 genealogías y por qué se consideran independientes

| genealogía | seed_base | origen |
|---|---|---|
| G0_original_90210 | 90210 | la ya usada en la línea F5-C2-C (documentada con 45.0% Clase III en `FASE5_control_azar_elastico_CS.md`) |
| G1_471829 | 471829 | nueva, nunca usada en ningún script de este proyecto |
| G2_823001 | 823001 | nueva, nunca usada |
| G3_156644 | 156644 | nueva, nunca usada |

Verificación por `grep` sobre `cs090_fase5_*.py` (documentada en el docstring del script): G1/G2/G3 no
aparecen en ningún archivo previo. Las tres nuevas semillas están bien separadas entre sí y de G0 en
magnitud/dígitos, y no guardan relación de múltiplo/submúltiplo entre ellas. Lo que garantiza independencia
real, más allá de la separación numérica (que es sólo una precaución adicional), es que
`generar_reglas_clase()` deriva cada parámetro individual de una cadena `seed = seed_base + intento*97 + 1`
— ninguna de las 4 genealogías comparte ni un solo `seed` individual con otra, por construcción aritmética.

**Una salvedad honesta sobre G0, encontrada al auditar los CSVs (no estaba en el docstring del script):**
el script pasa `seed_base=90210` directamente para G0, con la intención declarada de "recalcular con este
mismo pipeline" el número ya documentado (45.0%). Pero la línea F5-C2-C anterior usó `SEED_BASE=90210`
**sólo para los pilotos de 3 semillas** — el lote de 20 reglas que dio 45.0% se generó con
`seed_base=SEED_BASE+1=90211` (verificado en `cs090_fase5_control_azar_elastico.py` línea 398 y en
`cs090_fase5_presupuesto_emergente.py` línea 404). Con la fórmula `seed = seed_base + intento*97 + 1`, eso
significa que el G0 de esta tarea (semilla individual de partida 90211) **no es bit-a-bit el mismo lote**
que el que dio 45.0% (semilla individual de partida 90212) — es un lote adyacente, generado un "paso" más
temprano en la misma cadena. Confirmado directamente comparando parámetros: la regla `r0` de G0 acá tiene
`K=6, J=0.656, noise=0.292, meandeg=6.83, kcap=4` (seed=90211), mientras que `r0` de la corrida original
tiene `K=4, J=0.404, noise=0.234, meandeg=7.64, kcap=5` (seed=90212) — reglas distintas en todos sus
parámetros, no una reproducción exacta. Por eso G0 aquí dio 50.0% (10/20) en vez de 45.0% (9/20): no es una
falla de reproducibilidad del motor (que sigue siendo determinista, como siempre), es que el "control de
consistencia" terminó corriendo, sin que el script lo advirtiera, un **quinto lote cuasi-independiente**
más que una repetición exacta del cuarto. Esto no invalida el diseño — si acaso, da un punto de datos
adicional gratis sobre sensibilidad a la realización — pero significa que ninguna de las 4 filas de este
informe es una reproducción bit-exacta del 45.0% ya publicado; las cuatro son, en rigor, genealogías
nuevas o cuasi-nuevas.

## 2. Verificación honesta del filtro P1-P5 en cada genealogía

Se recalculó `cs090_fase5_generador.generar_reglas_clase("A2","B0","C2", n_reglas=20, seed_base=…)` para
las 4 genealogías (llamada de solo lectura al generador congelado, sin volver a correr el motor ni
Phantom) para confirmar cuántas de las primeras 20 propuestas pasaron el filtro:

| genealogía | seed_base | admitidas | descartadas | intentos totales |
|---|---|---|---|---|
| G0_original_90210 | 90210 | 20/20 | 0 | 20 |
| G1_471829 | 471829 | 20/20 | 0 | 20 |
| G2_823001 | 823001 | 20/20 | 0 | 20 |
| G3_156644 | 156644 | 20/20 | 0 | 20 |

Las 4 genealogías admitieron el 100% de sus primeras 20 propuestas sin ningún descarte por P1-P5. El filtro
no es más permisivo ni más estricto con ninguna de las 4 semillas base — no hubo que "buscar" reglas extra
en ninguna genealogía para completar el lote de 20.

## 3. Corrida

Brazos corridos en las 4 genealogías: **C2-hard** (`MOT.correr_regla_coarse`, el brazo estricto+uniforme de
toda la línea) y **C2-hibrido** (`MA.correr_regla_coarse_hibrido(modo="soporte")`, estricto+variable). El
piloto (3 semillas × 2 brazos sobre G1, ver `cs090_fase5_genealogias_independientes_piloto_resumen.csv`)
confirmó que ambos brazos corren sin fallos con `seed_base` arbitrario antes de lanzar el barrido completo.
No se corrió C0 ni las variantes de presupuesto elástico (fuera del alcance de esta tarea, que se concentra
en los dos brazos con el patrón bimodal más marcado).

Totales en disco: 4 genealogías × 20 reglas × 2 brazos = 160 filas en
`cs090_fase5_genealogias_independientes_resumen.csv` (161 líneas con encabezado, verificado) y 160 × 5
escalas de coarse-graining (b=1,2,4,8,16) = 800 filas en
`cs090_fase5_genealogias_independientes_resultados.csv` (801 líneas con encabezado, verificado). Ninguna
"salvaguarda de tiempo" recortó ninguna genealogía (las 4 tienen exactamente 20/20 filas en ambos brazos).

## 4. Resultado — tabla por genealogía

| genealogía | seed_base | brazo | n | I | II | III | IV | otro | **%Clase III** | grado medio (b=1) | diám medio | pendiente media | pendiente mediana |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G0_original_90210 | 90210 | C2-hard | 20 | 10 | 0 | **10** | 0 | 0 | **50.0%** | 3.33 | 14.10 | 0.710 | 0.747 |
| G1_471829 | 471829 | C2-hard | 20 | 9 | 1 | **9** | 0 | 1 | **45.0%** | 3.59 | 13.30 | 0.690 | 0.648 |
| G2_823001 | 823001 | C2-hard | 20 | 4 | 3 | **13** | 0 | 0 | **65.0%** | 3.49 | 13.95 | 0.680 | 0.778 |
| G3_156644 | 156644 | C2-hard | 20 | 5 | 0 | **15** | 0 | 0 | **75.0%** | 3.25 | 15.30 | 0.806 | 0.818 |
| G0_original_90210 | 90210 | C2-hibrido | 20 | 10 | 2 | **8** | 0 | 0 | **40.0%** | 3.85 | 11.85 | 0.733 | 0.606 |
| G1_471829 | 471829 | C2-hibrido | 20 | 12 | 1 | **5** | 0 | 2 | **25.0%** | 4.14 | 9.85 | 0.508 | 0.560 |
| G2_823001 | 823001 | C2-hibrido | 20 | 11 | 3 | **5** | 0 | 1 | **25.0%** | 4.21 | 10.00 | 0.492 | 0.577 |
| G3_156644 | 156644 | C2-hibrido | 20 | 8 | 2 | **10** | 0 | 0 | **50.0%** | 3.89 | 11.25 | 0.635 | 0.680 |

Sin ninguna fila Clase IV en las 160 (a diferencia de la línea F5-C2-C anterior, donde la misma regla `r16`
caía sistemáticamente en IV). Las 4 filas "otro" (intermedio, sin clase clara) están repartidas en reglas
distintas de genealogías distintas (`G1-r1`, `G1-r10`, `G1-r15`, `G2-r19`) — no hay un caso repetido tipo
`r16` esta vez, esperable porque cada genealogía tiene un lote de parámetros completamente distinto.

**Rango entre genealogías:** C2-hard va de 45.0% a 75.0% (30 puntos porcentuales); C2-hibrido va de 25.0% a
50.0% (25 puntos porcentuales). Los dos brazos mantienen el mismo orden relativo entre sí en las 4
genealogías (hard siempre por encima de hibrido, igual que en toda la línea F5-C2-C), y en las 4 genealogías
C2-hard sigue por encima de 45% — nunca cae a los niveles bajos (10%, 5%, 0%) que dieron los brazos elástico
o azar en `FASE5_control_azar_elastico_CS.md`.

## 5. Varianza ENTRE genealogías vs. varianza YA CONOCIDA dentro de una genealogía

Dos maneras de comparar, una categórica (fracción Clase III) y una continua (pendiente).

**(a) Categórica — %Clase III entre las 4 genealogías, vs. el ruido de muestreo que produciría CUALQUIER
lote de sólo 20 semillas de una única genealogía, aunque la tasa real fuera fija:**

| brazo | %III por genealogía | media entre-genealogías | **std entre-genealogías** | SE binomial esperado por genealogía (n=20, con su propio p) |
|---|---|---|---|---|
| C2-hard | 50.0 / 45.0 / 65.0 / 75.0 | 58.75% | **11.92 pp** | 11.18 / 11.12 / 10.67 / 9.68 pp (prom. ≈10.66 pp) |
| C2-hibrido | 40.0 / 25.0 / 25.0 / 50.0 | 35.0% | **10.61 pp** | 10.95 / 9.68 / 9.68 / 11.18 pp (prom. ≈10.37 pp) |

El desvío estándar observado ENTRE las 4 genealogías (11.92 pp hard, 10.61 pp hibrido) es prácticamente del
mismo tamaño que el error de muestreo binomial que ya se esperaría dentro de una sola genealogía si sólo se
mirara otro lote de 20 semillas al azar (≈10.4-10.7 pp en ambos brazos). Es decir: la dispersión que se ve
entre "redes sociales distintas" no es mayor de lo que ya se esperaría entre "distintos lotes de 20 días
tomados de la misma red social".

**(b) Continua — pendiente media entre genealogías, vs. la dispersión de la pendiente ya conocida
dentro de cada lote de 20 reglas:**

| brazo | pendiente media por genealogía | std ENTRE genealogías (de las 4 medias) | std INTRA genealogía (promedio, de las 20 reglas c/u) | SE esperado de la media si n=20 fueran muestras de una sola población (std_intra/√20) |
|---|---|---|---|---|
| C2-hard | 0.710 / 0.690 / 0.680 / 0.806 | **0.050** | 0.345 | 0.077 |
| C2-hibrido | 0.733 / 0.508 / 0.492 / 0.635 | **0.098** | 0.444 | 0.099 |

En C2-hard, la dispersión de las 4 medias entre genealogías (0.050) es incluso **más chica** que lo que la
sola dispersión intra-lote ya predeciría por puro muestreo (0.077). En C2-hibrido, la dispersión entre
genealogías (0.098) prácticamente **coincide** con lo que predeciría el mismo cálculo (0.099).

Con sólo 4 genealogías (3 grados de libertad) esto es una comparación de baja potencia — no alcanza para
afirmar con confianza que las genealogías son estadísticamente indistinguibles entre sí, sólo que **no hay
evidencia, en estos números, de que la genealogía agregue dispersión por encima de la que ya aporta la
variación seed-a-seed dentro de un mismo lote**. Con 4 puntos no se puede descartar un efecto de genealogía
moderado que quede enmascarado por el ruido de n=20.

## 6. Lectura honesta: ¿el bimodal ~45% es robusto a la genealogía, o específico de la que ya se venía usando?

- El **patrón cualitativo** (bimodal I/III sin apenas Clase II, hard por encima de hibrido, ninguna
  genealogía cerca de 0% o 100%) se sostiene en las 4 genealogías: C2-hard nunca baja de 45%, C2-hibrido
  nunca baja de 25%. No apareció ninguna genealogía "muerta" (0% Clase III) ni ninguna "saturada" (100%).
- El **número puntual 45.0%** documentado en `FASE5_control_azar_elastico_CS.md` no se reprodujo tal cual
  en ninguna de las 4 filas (ni siquiera en G0, por el desfase de `seed_base` explicado en §1) — los 8
  valores de esta tarea van de 25.0% a 75.0%, un rango amplio.
- La comparación cuantitativa de §5 no encuentra que la varianza entre genealogías exceda la varianza que
  ya se esperaría por puro muestreo de 20 semillas — en ese sentido, **no hay evidencia todavía de que el
  45% (o el 58.75% promedio observado acá) sea una propiedad frágil, específica de una genealogía
  particular**; tampoco hay evidencia firme de lo contrario, dado el bajo n=4 de genealogías.
- Dicho de otro modo: con los datos de hoy, no se puede distinguir "el mecanismo A2-B0-C2 produce ~45-60%
  Clase III en cualquier red social que se le dé" de "cada red social tiene su propio número verdadero
  (25% a 75% según cuál) y todavía no alcanzamos a medirlo con precisión porque 20 semillas por red social
  dejan mucho ruido". Separar esas dos lecturas pediría más genealogías (para ganar grados de libertad
  entre-grupo) y/o más semillas por genealogía (para achicar el ruido intra-grupo) — ninguna de las dos
  se hizo en esta tarea.

No se declara cierre ni veredicto sobre cuál de las dos lecturas es la correcta — los números están arriba,
la síntesis es de Alexis.

## 7. Lecturas alternativas y caveats honestos

- **Sólo 4 genealogías es poca muestra para hablar de "varianza entre genealogías" con solidez estadística**
  (3 grados de libertad). Los números de §5 son una primera mirada cuantitativa, no una prueba de hipótesis
  con potencia adecuada.
- **G0 no es una reproducción bit-exacta del 45.0% ya publicado** (ver §1) — es, en la práctica, una quinta
  muestra cuasi-independiente más que un control de consistencia limpio. Esto no compromete la comparación
  entre-genealogías de §5 (las 4 filas son igualmente válidas como muestras independientes entre sí), pero
  sí significa que este informe no puede decir "confirmamos que G0 reproduce el 45.0%" — no lo reprodujo,
  y ahora se sabe por qué.
- **C2-hibrido tuvo más filas "intermedio, sin clase clara" (4/160) que C2-hard (0/160)**, repartidas en 3
  de las 4 genealogías — consistente con el patrón ya visto en la línea F5-C2-C de que el brazo hibrido es
  algo más ruidoso/ambiguo que el hard, ahora replicado en genealogías nuevas.
- **No se corrieron los brazos elásticos** (`C2-presupuesto-variable`, `C2-presupuesto-variable-azar`, C0)
  en estas 4 genealogías — quedan como extensión posible si se quisiera ver si el patrón "el criterio deja
  de importar" de `FASE5_control_azar_elastico_CS.md` también es estable entre genealogías.
- **El grado medio y el diámetro medio (b=1)** también varían entre genealogías (grado medio 3.25-3.59 en
  hard, 3.85-4.21 en hibrido; diámetro medio 13.3-15.3 en hard, 9.85-11.85 en hibrido) en un rango
  comparable, en términos relativos, al de la pendiente — no se calculó formalmente su varianza
  entre/intra por brevedad, pero están en la tabla de §4 para quien quiera revisarlos.

## 8. Archivos de esta tarea

- `cs090_fase5_genealogias_independientes.py` — script ya existente, no modificado por este informe.
- `cs090_fase5_genealogias_independientes_resultados.csv` — 800 filas, dato crudo (4 genealogías × 20 reglas × 2 brazos × 5 escalas).
- `cs090_fase5_genealogias_independientes_resumen.csv` — 160 filas, una por genealogía×regla×brazo (clase + observables + parámetros).
- `cs090_fase5_genealogias_independientes_por_genealogia.csv` — 8 filas, resumen agregado por genealogía×brazo (la tabla de §4).
- `cs090_fase5_genealogias_independientes_piloto_raw.csv` / `_piloto_resumen.csv` — piloto de 3 semillas sobre G1, conservado.
- Este informe.

Ningún script congelado ni ninguno de los 5 archivos de la línea F5-C2-C fue modificado. No se corrió
Phantom. No se hicieron commits de git. No se declara cierre ni veredicto — los números de §4-§6 y los
caveats de §7 están arriba; la lectura final es de Alexis.
