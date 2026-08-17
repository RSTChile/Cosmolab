# O1-D — Factorial sistemático del trío (n=30): qué era el "hallazgo raro" de Fase IV

Tarea O1-D del `FASE6_PLAN_EJECUCION_COMPLETA_CS.md`. Sigue directo a `FASE4_control_local_global_CS.md`
(cs083b). Script nuevo: `cs083c_fase4_factorial_trio.py`. Datos crudos: `cs083c_resultados.csv` (30
semillas × 9 brazos) y `cs083c_resultados_piloto.csv` (5 semillas). Gráfico:
`cs083c_factorial_trio.png`.

No toca `cs082_fase4_4sustratos.py`, `cs083_fase4_robustecer.py` ni `cs083b_fase4_control_local_global.py`
— los importa. No se corrió Phantom (Fase IV nunca lo usó). **No hay cierre ni veredicto acá: son
números. La lectura final es de Alexis.**

---

## 1. Qué se pidió

`FASE4_control_local_global_CS.md` dejó un resultado sin explicar:

> El trío "coherente pero equivocado" (NULL-LOCAL-ROTO: las 3 aristas de OTRO triángulo real) aplana
> **menos** que el trío "totalmente suelto" (NULL-REWIRE: 3 aristas sueltas al azar). z=−15.62, en la
> dirección opuesta a la intuición.

Y dejó una hipótesis, marcada explícitamente como no comprobada: que la causa fuera una diferencia de
**cobertura de aristas** — que LOCAL-ROTO concentrara los empujones en las zonas densas del grafo y
dejara aristas periféricas sin tocar, mientras REWIRE los repartiera parejo.

La tarea pidió: factorial de 4-5 condiciones con n=30, midiendo cobertura (nº de aristas distintas
empujadas + Gini/entropía del reparto) y, si era barato, la variación intra-trío.

## 2. Lo primero que apareció fue una demostración, no un número

Leyendo el código de cs083b **antes de correr nada** apareció algo que cambia el diseño del factorial:

**El destino de LOCAL-ROTO se elige con un *derangement* (permutación sin puntos fijos) sobre los
triángulos. Una permutación es una biyección.** Cada triángulo real es destino de exactamente un
empujón por sweep — igual que en REAL. Por lo tanto **el reparto de empujones por arista es IDÉNTICO,
arista por arista, entre REAL y LOCAL-ROTO**.

Verificado numéricamente en el piloto y de nuevo en la corrida completa:

```
Gini de cobertura:  A(REAL)=0.5445   E(LOCAL-ROTO)=0.5445   ← idénticos hasta el último decimal
aristas tocadas:    A=323.3          E=323.3
```

Segunda consecuencia, más fuerte: **si en LOCAL-ROTO el defecto empujado se calculara sobre el trío
destino en vez del propio, el resultado sería REAL bit a bit** (la corrección se acumula y se aplica al
final del sweep, así que el orden dentro del sweep no importa; una biyección sólo reordena). También
verificado numéricamente (`_verificar_biyeccion_equivale_a_real`, sale "sí").

O sea: **lo que separa LOCAL-ROTO de REAL no es "el trío equivocado". Es la DESALINEACIÓN** entre de
dónde sale el defecto (las 3 aristas propias de T) y adónde va el empujón (las 3 de T'). Ese factor no
estaba cruzado en el diseño de cs083b. Este factorial lo cruza.

## 3. Los 9 brazos

Todos comparten N=110, mismo grafo base por semilla, K=6, J=0.6, J_FACE=0.5, ruido=0.25,
COMPUTE_BUDGET=60 000 → **110 sweeps, DoF=706, 483 eventos-empuje/sweep en los 9 brazos** (verificado en
las 30 semillas). Lo único que cambia es a qué aristas va el empujón y de qué trío sale el defecto.

| | brazo | destino del empujón | defecto calculado sobre | rol |
|---|---|---|---|---|
| **A** | trío CORRECTO | sus propias 3 aristas | el destino | = REAL (cs082) — **pedido 1** |
| **B** | trío CASI-correcto | 1 arista propia + 2 sueltas | el destino | **condición NUEVA — pedido 2** |
| **C** | trío AL AZAR | 3 aristas sueltas al azar | el destino | = NULL-REWIRE (cs083) — **pedido 3** |
| **D** | SIN TRÍOS | todas las aristas, cada sweep | media global | = NULL-GLOBAL (cs083b) — **pedido 4** |
| **E** | trío EQUIVOCADO | las 3 de otro triángulo real | **el trío propio** | = NULL-LOCAL-ROTO — **pedido 5, el raro** |
| F | azar DESALINEADO | 3 aristas sueltas al azar | **el trío propio** | diagnóstico: cierra el 2×2 |
| G | trío real AZAR c/reempl. | trío de un triángulo real al azar | el destino | diagnóstico: coherencia sin biyección |
| H | sueltas-DE-triángulo desalin. | 3 sueltas, sorteadas **sólo entre aristas de triángulo** | el trío propio | diagnóstico: puntería igualada a E |
| I | sueltas-DE-triángulo alineado | ídem H | el destino | diagnóstico: puntería igualada a A |

Los brazos F, G, H, I son extra (el presupuesto lo permitía de sobra: 9 brazos × 30 semillas corrieron
en una sola tanda). Por qué H e I: en C y F las aristas-destino se sortean sobre TODO el grafo, y **el
40.7% de las aristas no pertenece a ningún triángulo** — o sea, ~41% de esos empujones cae donde la
holonomía ni siquiera mide. En A y E el 100% cae sobre aristas de triángulo. Esa diferencia de
*puntería* es la otra variable de la familia "cobertura" y estaba confundida con la coherencia del trío.

**Verificación bit a bit** (piloto, seed 1): el motor genérico reproduce EXACTAMENTE las funciones ya
auditadas — A = `cs082.correr_sustrato_4_2complejo`, C = `cs083.correr_sustrato_4_control_fino`,
E = `cs083b.correr_sustrato_4_null_local_roto`, D = `cs083b.correr_sustrato_4_null_global`. Las cuatro
dan "OK (idéntico)". Nada de lo que sigue depende de una reimplementación distinta.

## 4. Ordenamiento en holonomía |h| (n=30; más bajo = más aplanado)

| # | brazo | h media | DE | h mediana | >REAL? | pedido |
|---|---|---|---|---|---|---|
| 1 | **A trío CORRECTO (REAL)** | **0.2627** | 0.1099 | 0.2253 | 0/30 | sí |
| 2 | G trío real azar c/reempl. | 0.2988 | 0.1517 | 0.2731 | 25/30 | extra |
| 3 | I sueltas-de-triáng. alineado | 0.3507 | 0.0285 | 0.3491 | 27/30 | extra |
| 4 | **B trío CASI-correcto** | **0.3601** | 0.0219 | 0.3572 | 27/30 | sí |
| 5 | **C trío AL AZAR (REWIRE)** | **0.3680** | 0.0268 | 0.3710 | 27/30 | sí |
| 6 | H sueltas-de-triáng. desalin. | 0.4387 | 0.0392 | 0.4331 | 27/30 | extra |
| 7 | F azar DESALINEADO | 0.4433 | 0.0426 | 0.4379 | 27/30 | extra |
| 8 | **E trío EQUIVOCADO (LOC.ROTO)** | **0.5006** | 0.0411 | 0.4931 | 27/30 | sí |
| 9 | **D SIN TRÍOS (GLOBAL)** | **1.5907** | 0.8788 | 1.6860 | 29/30 | sí |
| | NULL (ruido puro) | 1.5108 | 0.0827 | | | ref. |
| | SHUFFLED | 0.4923 | 0.3904 | | | ref. |

**El ordenamiento de las 5 condiciones pedidas es: A (0.263) < B (0.360) < C (0.368) < E (0.501) <
D (1.591).**

Reproducciones de resultados previos que se sostienen a n=30:
- **D SIN TRÍOS sigue siendo indistinguible de ruido puro**: z=+0.49, p=0.63, 13/30 signos. Idéntico
  al p=0.52 de cs083b.
- **REAL sigue separando de todos los controles con trío**: A−C z=−5.08, A−E z=−11.39, A−B z=−4.89,
  todos p≤0.0001, 27/30 signos.
- La fracción "8% de cierre local genuino" de cs083 se reproduce otra vez: A−C = 8.4% del gap total
  (era 8.3% en cs083b con 20 semillas, ~8% en cs083).

## 5. El hallazgo raro SE SOSTIENE — casi con el mismo tamaño

| | n | z | diff. observada | signos | p (2 colas) |
|---|---|---|---|---|---|
| cs083b (informe previo) | 20 | −15.62 | −0.138 | — | <0.0001 |
| **cs083c (esta corrida)** | **30** | **−15.54** | **−0.1326** | **30/30** | **<0.00005** |

C (trío suelto) aplana más que E (trío equivocado) en las **30 de 30 semillas**. No se disuelve con más
muestra: se reproduce con un z prácticamente idéntico. Lo que cambia es que ahora sabemos de qué está
hecho.

## 6. La cobertura NO lo explica

Esta era la hipótesis a poner a prueba. Los tres frentes la contradicen:

**(a) Prueba directa.** A y E tienen la MISMA cobertura arista por arista (Gini 0.5445 los dos, 323.3
aristas tocadas los dos, por la biyección) y sin embargo difieren en h por 0.238 (z=−11.39). C y F usan
el mismo esquema de sorteo (Gini 0.5461 vs 0.5453, |Δ|=0.0008) y difieren por 0.075 (z=−8.33). **La
cobertura está igualada exactamente en los dos pares donde el efecto es más grande.**

**(b) Correlaciones.** Spearman de cada descriptor contra h, sobre las 270 filas brazo×semilla:

| variable | ρ (9 brazos) | ρ (sin D) | ρ (sólo A,C,E,F) |
|---|---|---|---|
| Gini de cobertura | −0.362 | −0.199 | **−0.038** |
| entropía de cobertura | +0.360 | +0.197 | +0.044 |
| % de aristas tocadas | +0.352 | +0.187 | +0.043 |
| % de aristas con cero empujones | −0.352 | −0.187 | −0.043 |
| puntería (% de empujones sobre aristas de triángulo) | −0.305 | −0.163 | +0.021 |
| solape de nodos del trío destino | +0.057 | −0.219 | −0.006 |
| **variación intra-trío (destino)** | **+0.710** | **+0.681** | **+0.673** |
| **variación intra-trío (triángulos reales)** | +0.245 | **+0.615** | **+0.694** |

Los ρ de la familia cobertura son moderados sólo mientras D esté en la muestra — y D es un brazo
distinto en todo, no sólo en cobertura. Restringido al 2×2 limpio (A,C,E,F) la cobertura **cae a
ρ≈0.04**, o sea nada. El descriptor que sí correlaciona fuerte y estable es la **variación intra-trío**.

**(c) Puntería.** Igualarla no mueve casi nada: F−H (misma condición, puntería 59.5% vs 100%) da
diff=+0.0046, z=+0.40, p=0.69 — **no significativo**. Con alineación, C−I da +0.0173 (z=+2.66), un
efecto pequeño y en la dirección contraria a la que haría falta para explicar el raro.

## 7. Lo que sí lo explica — el 2×2 y una inversión de signo

Cruzando **destino** (trío real ajeno / 3 aristas sueltas) × **alineación** (el defecto sale del trío que
se empuja / de otro trío):

| | destino = trío real ajeno | destino = 3 sueltas | efecto del destino |
|---|---|---|---|
| **alineado** | A = 0.2627 | C = 0.3680 | **A−C = −0.105** (z=−5.08) — el trío real **ayuda** |
| **desalineado** | E = 0.5006 | F = 0.4433 | **E−F = +0.057** (z=+5.70, 4/30) — el trío real **estorba** |
| efecto de la alineación | A−E = −0.238 (z=−11.39) | C−F = −0.075 (z=−8.33) | **interacción z=−6.36** |

**El signo del "efecto trío real" se da vuelta según si la fuerza está alineada o no.** Cuando el
empujón trae el defecto del trío que va a corregir, ser un trío real ayuda. Cuando trae el defecto de
otro, ser un trío real *empeora las cosas* respecto de 3 aristas sueltas. Eso es exactamente lo que
generaba el "hallazgo raro": comparaba una celda alineada (C) contra una desalineada (E), mezclando los
dos factores.

**Descomposición aditiva del raro** (h(E) − h(C) = +0.1326), con el camino C → F → H → E:

| paso | qué cambia | Δh | % del raro |
|---|---|---|---|
| (1) C → F | **desalineación** fuente↔destino | +0.0753 | **56.8%** |
| (2) F → H | **puntería** (dejar de desperdiciar empujones fuera de los triángulos) | −0.0046 | **−3.5%** (nulo) |
| (3) H → E | **coherencia** del trío destino, ya desalineado | +0.0619 | **46.7%** |
| | **total C → E** | **+0.1326** | 100% |

O sea: **~57% del hallazgo raro es desalineación, ~47% es coherencia del trío destino, y ~0% es
cobertura/puntería.** El paso (3) tiene su propio test: H−E = −0.0619, z=−5.66, 24/30 signos, p<0.0001 —
con la puntería igualada al 100% en ambos, meterle ruido ajeno a un **trío coherente** hace más daño que
meterlo a 3 aristas sueltas. Ese segundo pedazo sí es un efecto de estructura del trío, pero opera en el
sentido de **fragilidad**, no de imitación: un trío cerrado es más sensible al ruido incoherente.

Y encaja con la métrica que el analista sugirió: la **variación intra-trío** al final de la corrida es
sistemáticamente mayor en la familia desalineada (0.0293) que en la alineada (0.0235), y es el
descriptor con la correlación más alta y más estable contra h (ρ=0.67-0.71 en todos los cortes). La
lectura "estrés estructural" es compatible con los datos; la lectura "cobertura" no.

## 8. Las condiciones pedidas, una por una

- **B trío CASI-correcto (1 arista propia + 2 sueltas)** — la condición nueva que llenaba el hueco:
  h=0.3601, prácticamente **idéntica a C (trío al azar, 0.3680)**: B−C diff=−0.0079, z=−1.18, p=0.25,
  15/30 signos. **Tener una sola de las tres aristas correctas no compra nada medible.** El aplanamiento
  extra de REAL aparece sólo cuando las 3 aristas son las propias, no de a poco.
- **D SIN TRÍOS**: sigue siendo ruido (z=+0.49 vs NULL). Confirma el hallazgo (a) de cs083b a n=30.
- **G trío real azar con reemplazo** (coherente y alineado, pero sin biyección → cobertura mucho más
  dispareja: Gini 0.703 vs 0.545, 56.8% de aristas sin tocar vs 40.7%): h=0.2988, **no se distingue de
  REAL** (A−G z=−1.01, p=0.19). Es decir: **triplicar la desigualdad de cobertura no rompe el
  aplanamiento** mientras el mecanismo siga alineado y sobre tríos reales. Otro golpe a la hipótesis de
  cobertura, desde el lado opuesto.

## 9. Lo que queda abierto y las cosas incómodas

- **REAL es bimodal entre semillas.** 27 de las 30 semillas dan h≈0.228 ± 0.030; **3 semillas (5, 13,
  28) dan 0.544-0.604**, casi el valor de E. Por eso la DE de A (0.1099) es 3-5× la de los otros brazos
  y su mediana (0.2253) está muy por debajo de su media. Excluyendo esas 3, todos los contrastes se
  agrandan mucho (A−C z=−18.0, A−E z=−29.1). **No hay explicación acá de por qué en esas 3 semillas el
  sustrato real falla en aplanar.** Es un dato para perseguir, no un ruido que convenga barrer.
- **G tiene su propio caso raro** (semilla 30, h=1.095, contra 0.271 ± 0.020 en las otras 29); ese único
  punto es lo que hace que G−I dé z=−1.87 pese a que G queda por debajo de I en 29/30 semillas.
- Los porcentajes de la descomposición (57% / −3% / 47%) suponen que los tres pasos son aditivos. Con
  interacción z=−6.36 ya demostrada entre dos de los factores, esa aditividad es una aproximación de
  contabilidad, no una propiedad del sistema.
- Todo esto es sobre UN sustrato (el 4), UN tamaño (N=110), UN presupuesto (60 000). Nada dice que el
  patrón sobreviva a otro régimen de J_FACE o de ruido.

## 10. En simple, con analogía

Volvamos a la ronda de gente en círculo tratando de ponerse de acuerdo en qué mano levantar, en tríos
de tres personas que se conocen.

- **A (REAL)**: cada trío se mira a sí mismo, ve qué tan desalineado está, y se corrige a sí mismo.
- **C (trío al azar)**: se arman tríos de tres desconocidos; cada trío se mira a sí mismo y se corrige
  a sí mismo. **Funciona casi igual de bien.**
- **E (trío equivocado)**: cada trío mira SU propio desorden… y le grita la corrección a OTRO trío, que
  no tenía ese problema. **Funciona bastante peor.**

La sospecha del informe anterior era que E funcionaba peor porque los gritos caían siempre sobre la
misma gente y otros quedaban sin escuchar nada. **Eso resultó falso, y de la manera más limpia posible:
la matemática de la permutación garantiza que en E cada persona recibe exactamente la misma cantidad de
gritos que en A.** El reparto es idéntico. No hay nadie desatendido.

Lo que pasa es otra cosa, y son dos cosas sumadas:

1. **La corrección va dirigida a quien no le corresponde** (57% del efecto). No es que se grite poco o
   mal repartido: es que el mensaje que llega no tiene nada que ver con el problema de quien lo recibe.
   Es ruido con forma de consejo.
2. **Un grupo que ya está bien coordinado es más frágil a ese ruido que tres desconocidos** (47%). A
   tres personas sueltas, un consejo equivocado las descoloca poco, cada una por su lado. A un trío que
   ya se había alineado entre sí, el mismo consejo equivocado le rompe la coordinación que había
   logrado. Por eso "casi correcto" no es "casi tan bueno": en esto, parecerse al arreglo correcto sin
   serlo es peor que no parecerse en nada.

Y un detalle nuevo que salió al pasar: tener **una sola** de las tres aristas correctas (brazo B) no
sirve para nada — mide igual que tener las tres al azar. El beneficio del trío correcto no llega de a
poco; o están las tres, o no está.

## 11. Archivos generados

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs083c_fase4_factorial_trio.py` — script nuevo
  (modos `pilot` / `full` / `replot`). No modifica cs082/cs083/cs083b: los importa y verifica bit a bit
  contra ellos.
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs083c_resultados.csv` — 270 filas (30 semillas × 9
  brazos), 22 columnas: holonomía, referencias NULL/SHUFFLED, los 7 descriptores de cobertura, solape de
  nodos, las 2 variaciones intra-trío y los metadatos de equiparación (DoF, sweeps, eventos/sweep).
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs083c_resultados_piloto.csv` — piloto de 5 semillas.
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs083c_factorial_trio.png` — 4 paneles: ordenamiento
  de los brazos, cobertura vs holonomía, puntería vs holonomía, variación intra-trío vs holonomía.

Sin cierre. Números arriba; la interpretación final queda para Alexis.
