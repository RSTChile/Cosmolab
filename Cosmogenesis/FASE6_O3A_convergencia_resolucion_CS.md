# Fase VI · O3-A — ¿el efecto de Fase V-B sobrevive al subir la resolución?

**Fecha:** 11-ago-2026 · **Ejecuta:** CC (Claude) · **Origen:** `FASE6_PLAN_EJECUCION_COMPLETA_CS.md`,
tarea O3-A (marcada por el equipo — GPT-5.6 Sol, VI-B/F6-02 — como la OBLIGATORIA y más urgente de la
serie). Sigue directamente de `FASE5B_escala_40pares_CS.md` y de
`FASE5B_investigacion_8sumideros_y_escala_CS.md`.

Phantom estaba autorizado explícitamente por Alexis. **No se declara cierre ni veredicto**: se reportan
números. Ningún script congelado fue modificado — todos se importan tal cual. No se hicieron commits de
git.

> **ESTADO: batería completa a N=2000 y N=4000 — 13 pares × 2 reglas × 2 resoluciones = 52 corridas, de
> las cuales 26 son nuevas (las de N=4000; las de N=2000 se heredan del CSV congelado de Fase V-B sin
> recomputar). N=8000 sondeado y abortado por costo, con el costo medido reportado en §8.** Números, no
> veredicto.
>
> **Los tres titulares, en una línea cada uno:**
> 1. La diferencia **no se va a cero ni se invierte en promedio**: Δmasa media pasa de **+0,0095**
>    (N=2000) a **+0,0175** (N=4000) sobre los MISMOS 13 pares; en magnitud absoluta se **duplica**
>    (0,0110 → 0,0239, ×2,18, Wilcoxon pareado p=0,022).
> 2. Pero el **signo par-a-par se conserva sólo en 9 de 13**, y la correlación del ordenamiento entre
>    resoluciones es **débil (Spearman +0,38)**: lo que sobrevive es la tendencia del conjunto, no el
>    ranking. Los 4 pares que se dieron vuelta son los de Δ chico a N=2000 (los 4 de |Δ|≥0,012 conservan
>    los 4 su signo).
> 3. El observable secundario **κ_V se da vuelta entero**: media +0,063 (9/13 a favor de III) a N=2000
>    contra **−0,062 (2/13)** a N=4000.
>
> Y el hallazgo que más pesa para leer todo lo anterior, en §6: al descontar la diferencia de **densidad
> del grafo** (aristas por nodo) entre las dos reglas de cada par, la ventaja residual de la Clase III
> queda en +0,004 con p=0,22 (N=2000) y p=0,41 (N=4000) — **indistinguible de cero en las dos
> resoluciones** —, mientras que la **pendiente** del efecto de densidad casi se triplica con N.

## 0. En simple, con analogía

Fase V-B midió una carrera entre dos tipos de "reglas de tejido": la Clase III (la que sigue creciendo
con el tamaño) contra la Clase I (la que se estanca), emparejadas de a dos con exactamente los mismos
parámetros de fuerza (mismo K, mismo kcap). Cada regla arma una nube de partículas, se la deja caer bajo
gravedad en Phantom, y se mide cuánta masa termina adentro de "sumideros" (grumos colapsados). En 31 de
40 parejas ganó la Clase III.

El problema: **todas esas carreras se corrieron en la misma "cámara de baja definición" — 2000
partículas.** Y en la tarea anterior se descubrió que el NÚMERO de grumos que se forman no lo decide el
tejido sino la definición de la cámara: con la misma cantidad de materia, 2000 partículas dan ~8 grumos,
4000 dan ~29 y 8000 dan ~122. Si el número de grumos es un artefacto de la definición, ¿la ventaja en
MASA también lo será?

La analogía: fotografiar el mismo paisaje con 2, 4 y 8 megapíxeles. Nadie espera que las tres fotos
tengan los mismos píxeles. Lo que sí se exige es que **si en la foto chica el árbol de la izquierda se
veía más alto que el de la derecha, en la foto grande siga siendo el más alto.** Si al subir la
definición los árboles se emparejan o se dan vuelta, "el árbol más alto" era un efecto de la mala
resolución, no del paisaje. Eso es exactamente lo que esta tarea mide: no la altura absoluta, sino el
SIGNO y el ORDEN de la diferencia.

## 1. Diseño

### 1.1 Qué se mantiene idéntico y qué no (la sutileza honesta)

Se mantienen **exactamente** iguales entre resoluciones:

| Cosa | Valor |
|---|---|
| Regla (semilla y parámetros) | la misma `seed`; mismos K, J, noise, meandeg, kcap, sim_thr_frac |
| Masa física total | 18800 (no escala con N — el fix del confound caja/masa∝N de CS073) |
| Lado de la caja | fijo, 2000^(1/3) = 12.599210, **no** depende de N |
| Semilla de layout | 12345 |
| Semilla de turbulencia | 42, Mach=3, espectro Burgers k^-2 |
| Sweeps del motor relacional | 14 |
| Iteraciones de `layout_resortes` | 100 (**no** se bajó a 25 como hizo `ON77_sistemaA_cierre.py`: bajarlas cambiaría el protocolo respecto del punto N=2000 con el que se compara) |
| Protocolo de Phantom | `icreate_sinks=1`, `rho_crit_cgs=1000`, `r_crit=0.600`, `h_acc=0.300`, `f_acc=0.800`, `tmax=0.500`, `dtmax=0.001` |

Lo que **no puede** mantenerse literalmente idéntico, y hay que decirlo con todas las letras: en este
pipeline **los nodos del grafo SON las partículas SPH**. `reconstruir_regla_a2b0c2(seed, N)` construye el
grafo con N nodos y `rng = default_rng(seed*5000 + N)`. Subir N no es reetiquetar un grafo fijo con más
partículas: es **reconstruir la MISMA REGLA sobre un grafo más grande**, otra realización de la misma ley
generativa. La invariante entre resoluciones es la REGLA, no la lista de aristas. Es el mismo sentido de
"subir resolución" que usó `ON77_sistemaA_cierre.py` en el barrido 2000/4000/8000 que produjo el
8→29→122, así que la serie es comparable con ese antecedente. Un test alternativo — congelar el grafo de
2000 nodos y sólo partir cada partícula SPH en varias — mediría otra cosa (resolución del fluido a
tejido congelado) y **queda como pendiente explícito**, no se hizo acá.

Consecuencia que hay que vigilar y que este informe reporta en vez de suponer: la etiqueta de clase
(I/III) se la pone el clasificador a partir de métricas del grafo medidas a una N dada. Por eso cada
corrida guarda las métricas del grafo a su N (n_aristas, grado medio, diámetro de la componente gigante
con la medición **oficial** `cs090_diam_corregido.diam_gigante`, holonomía) — §6.1 las compara.

### 1.2 Selección de pares (criterio escrito ANTES de mirar nada nuevo)

Script: `cs090_fase6_o3a_seleccionar_pares.py`. Criterio mecánico, sin cherry-picking:

1. Se parte de los 40 pares de `cs090_fase5b_TOTAL_40pares.csv` (Fase V-B, N=2000).
2. Se filtran los **37 pares limpios** — `match_exacto_K_kcap == True`, o sea Clase III y Clase I con
   exactamente el mismo K y el mismo kcap (prioridad pedida por el equipo: los 3 pares "sucios" mezclan
   la diferencia de clase con una diferencia de parámetros).
3. Se ordenan por Δmasa = fracción_masa(III) − fracción_masa(I) a N=2000, de menor a mayor.
4. Se toman **12 posiciones equiespaciadas en ese ranking** (índices 0, 3, 7, 10, 13, 16, 20, 23, 26, 29,
   33, 36). El primero es el par **más invertido** (el peor caso para la hipótesis) y el último el más
   favorable.
5. Un empate exacto en Δ=+0.0085 hizo que se corriera también `batch4-r51_vs_batch4-r36`, marcado como
   `extra_por_empate` — se reporta como par #13 en vez de tirar el cómputo; no sesga nada porque los dos
   empatados tienen el MISMO Δ a N=2000.

Resultado: **13 pares**, de los cuales 2 con Δ<0 (ganaba la Clase I), 1 prácticamente en cero
(Δ=+0.0005) y 10 con Δ>0, cubriendo todo el rango de −0.0060 a +0.0325.

### 1.3 Verificación cruzada obligatoria

Lección del bug de colisión de nombres de la tarea anterior. Dos capas:

- **Nombre de carpeta con las tres cosas** — `bateria_fase6_o3a_resolucion/N<N>/<rule_id>_<clase>`: es
  imposible que una corrida de N=4000 pise la de N=2000 de la misma regla.
- **`verificar_meta_contra_csv`** — antes de aceptar cualquier corrida se relee
  `cs090_fase5b_TOTAL_40pares.csv` y se comprueba que `seed`, `clase`, `K` y `kcap` del
  `meta_regla.json` recién escrito coinciden con la fila de esa regla en el CSV de origen. Si no
  coinciden, `AssertionError` y esa corrida no entra en el análisis.
- Además, `cs090_fase6_o3a_verificar_ic.py` chequea la integridad de cada `cosmogenesis_ic.txt`
  (N+2 líneas exactas, 7 números finitos por partícula, masa total = 18800) porque varios layouts se
  generaron en procesos paralelos.

## 2. Costo REAL medido (esto decidió el alcance de la tarea)

La tarea pedía explícitamente cronometrar antes de comprometer la batería. Se hizo, y el resultado
cambió el plan. **El cuello de botella no es Phantom: es `layout_resortes`**, que es O(N²) por iteración
× 100 iteraciones y además aloca arreglos N×N×3 (a N=4000 son ~900 MB de temporales POR ITERACIÓN, así
que el proceso queda limitado por ancho de banda de memoria, no por CPU).

| Etapa | N=2000 (histórico, Fase V-B) | N=4000 (medido acá) | N=8000 (medido acá) |
|---|---|---|---|
| Motor relacional (reconstruir el grafo) | incluido abajo | **3.7 – 10.6 s** | — |
| `layout_resortes` + escribir la IC | 43 – 94 s por grafo (total con el motor) | **656 s un grafo solo; 756 – 807 s con 5-8 grafos en paralelo** | **abortado a los 16.6 min sin terminar** |
| Phantom (setup + run hasta tmax=0.5) | ~11 s por corrida (441 s / 40) | **25 – 43 s por corrida** | no se llegó a correr |
| Disco por corrida (500 dumps) | ~27 MB | ~54 MB (se podan los intermedios, ver abajo) | ~108 MB |

**Decisión sobre N=8000, documentada:** se lanzó un grafo a N=8000 en paralelo con el benchmark de
N=4000. A los **16 min 36 s** todavía no había terminado su layout, con **5.1 GB de RSS**, mientras la
máquina estaba en **15.6 GB de 16 GB de swap usado** (y con el disco al 99%, 7.8 GiB libres). El costo
esperado por grafo, escalando el 656 s medido a N=4000 por el factor 4 de O(N²), es de **≥ 44 min por
grafo**, o sea **≥ 1 h 30 min por par** — y eso antes de Phantom, que en el antecedente de
`ON77_sistemaA_cierre` a N=8000 tuvo que abortarse por colapso del paso de tiempo cerca de los 122
sumideros que se forman a esa resolución. **Se abortó el grafo de N=8000 para proteger la batería de
N=4000**, siguiendo la instrucción explícita de la tarea ("es preferible una serie N=2000/4000 completa
y bien medida que una serie N=8000 a medias"). **No hay ningún punto de N=8000 en este informe.**

> **Nota de la segunda sesión (23:09–00:00):** ese primer benchmark se corrió con la máquina saturada
> por ~20 agentes en paralelo. Ya libre la máquina, se volvió a medir el costo de N=8000 en condiciones
> limpias y con la memoria real disponible (64 GB, no 16). El resultado de esa segunda medición —que
> confirma el orden de magnitud pero por una razón distinta— está en **§8**, y tampoco produjo ningún
> punto de N=8000. La tabla de arriba se deja tal cual quedó medida, sin retocar.

**Poda de dumps:** cada corrida escribe 500 dumps (`tmax=0.500`, `dtmax=0.001`) y el disco de la Mac
estaba al 99%. El análisis sólo lee el PRIMER dump (gas inicial), el ÚLTIMO (masa en sumideros vs. gas)
y el `.sink` (κ_V, t del primer sumidero); los 498 del medio no se abren nunca. Se borran DESPUÉS de que
la corrida ya fue analizada y su `resultado_o3a.json` está escrito (`cs090_fase6_o3a_correr_phantom.sh`).

## 3. Qué se corrió, en números

| | N=2000 | N=4000 | N=8000 |
|---|---|---|---|
| Reglas con corrida completa | 26 (heredadas de Fase V-B, sin recomputar) | **26** | 0 |
| Pares completos (Clase I + Clase III) | **13** | **13** | 0 |
| Corridas de Phantom nuevas de esta tarea | 0 | **26** | 0 |
| Corridas abortadas por timeout | 0 | **0** | — |
| Dump final alcanzado | — | `cosmog_00500` en las 26 (ninguna se quedó corta) | — |
| Sumideros por corrida (media) | 8,00 | **28,65** | — |

Las 26 corridas de N=4000 pasaron la verificación cruzada `verificar_meta_contra_csv` (seed, rol, K y
kcap del `meta_regla.json` contra la fila del CSV de Fase V-B) y el chequeo de integridad de la IC. Las
13 primeras se generaron en la sesión anterior; las 13 restantes, en ésta.

### 3.1 Los 13 pares son representativos de los 37 — el sesgo de selección está descartado

Antes de leer nada, el control que pidió la tarea. Comparación de Δmasa a N=2000:

| Grupo | n | Δ medio | pares con III>I |
|---|---|---|---|
| Los 37 pares limpios de Fase V-B | 37 | +0,00918 | 30/37 (Wilcoxon p=0,00001) |
| **Los 13 elegidos para O3-A** | 13 | **+0,00950** | 11/13 |
| Los 24 NO elegidos | 24 | +0,00900 | 19/24 |

Kolmogorov-Smirnov entre elegidos y no elegidos: **p = 1,000**. La muestra de 13 no está sesgada
respecto del universo del que salió.

### 3.2 El aviso preliminar era correcto: los 3 primeros pares SÍ engañaban

La lectura preliminar (con los 3 únicos pares que habían llegado a N=4000) decía Δ ≈ −0,0003 a N=2000 y
+0,0215 a N=4000: un salto espectacular desde cero. Con los 13, ese salto se desinfla:

| Muestra | Δ medio a N=2000 | Δ medio a N=4000 |
|---|---|---|
| Los 3 que alcanzaron a correr primero | −0,00033 | +0,02150 |
| **Los 13 completos** | **+0,00950** | **+0,01752** |

La razón es exactamente la que se sospechaba: los 3 primeros eran los pares de las posiciones 0, 3 y 7
del ranking ordenado por Δ ascendente — o sea, **el fondo del ranking** (los dos únicos con Δ<0 y el que
estaba prácticamente en cero). No eran una muestra al azar de los 13: eran los tres peores casos para la
hipótesis. Con la muestra completa el efecto a N=2000 ya no arranca de cero, y por lo tanto el
crecimiento hasta N=4000 es un factor ~1,8 en la media con signo, no un salto desde la nada.

## 4. El resultado principal: Δ masa contra N

Sobre los **mismos 13 pares** en las dos resoluciones (la única comparación pareada válida):

| | N=2000 | N=4000 |
|---|---|---|
| n pares | 13 | 13 |
| **Δmasa medio (III − I)** | **+0,00950** | **+0,01752** |
| Δmasa mediano | +0,00850 | +0,02050 |
| error estándar de la media | 0,00303 | 0,00671 |
| Δ medio en valor absoluto | 0,01096 | **0,02387** |
| Δ RELATIVO medio (Δ / fracción de la Clase I) | +11,3 % | **+18,0 %** |
| fracción de masa media, Clase I | 0,0974 | 0,1411 |
| fracción de masa media, Clase III | 0,1069 | 0,1587 |
| pares con III > I | 11/13 | 9/13 |
| test de signos (binomial dos colas) | p = 0,022 | p = 0,267 |
| Wilcoxon de rangos con signo | p = 0,0051 | **p = 0,0398** |
| **pares que conservan el signo respecto de N=2000** | 13/13 (trivial) | **9/13** |
| Pearson del Δ par-a-par contra N=2000 | 1,00 (trivial) | **+0,377** |
| Spearman del Δ par-a-par contra N=2000 | 1,00 (trivial) | **+0,379** |

Tres lecturas separadas, porque contestan tres preguntas distintas:

**(a) ¿Se va a cero o se invierte?** No. La media con signo **sube** (+0,0095 → +0,0175) y la magnitud
absoluta media **se duplica** (0,0110 → 0,0239, factor **2,18**). El test pareado sobre |Δ| da
**Wilcoxon p = 0,0215**: el aumento de magnitud es más de lo que se esperaría por azar. Sobre el Δ CON
SIGNO, en cambio, el mismo test pareado da **p = 0,376** — o sea, el aumento de la media con signo no se
distingue del ruido, porque los pares que se dan vuelta se comen buena parte de la ganancia. Las dos
cosas son ciertas a la vez y hay que decir las dos.

**(b) ¿Se mantiene el signo?** En **9 de 13**. Y el fallo no es homogéneo: depende del tamaño del efecto
a baja resolución.

| pares agrupados por la MAGNITUD de Δ a N=2000 | n | conservan el signo |
|---|---|---|
| magnitud ≥ 0,012 (los de señal grande) | 4 | **4/4** |
| entre 0,008 y 0,012 | 3 | 2/3 |
| magnitud < 0,008 (los de empate práctico) | 6 | 3/6 |

Los cuatro pares que se dan vuelta tenían Δ a N=2000 de −0,0060, +0,0005, +0,0070 y +0,0105: todos en la
zona de empate. Los cuatro de señal grande (+0,0125, +0,0195, +0,0250, +0,0325) mantienen los cuatro su
signo. Leído en la analogía de la foto: **los árboles claramente más altos siguen siendo los más altos
al subir los megapíxeles; los que estaban casi empatados se barajan.** Eso es compatible tanto con "el
efecto es real pero chico y el ruido de una sola corrida por regla lo tapa cuando |Δ| es pequeño" como
con "sólo hay señal en la cola" — este experimento, con una corrida por regla y n=13, **no separa esas
dos**.

**(c) ¿Se conserva el ORDEN?** Débilmente. Spearman +0,379 y Pearson +0,377 entre el Δ de N=2000 y el de
N=4000. Es positivo, pero muy lejos de 1: **el que ganaba no necesariamente sigue ganando por el mismo
margen**. Es el número más incómodo de la tabla y no hay que suavizarlo.

### 4.1 La marea sube para los dos

Un control obligatorio antes de festejar el +0,0175: la fracción de masa ABSOLUTA sube con la resolución
en las dos clases (Clase I 0,0974 → 0,1411; Clase III 0,1069 → 0,1587, ×1,45 y ×1,48). Es la misma marea
que ya se conocía (a más partículas, la misma masa se resuelve en más sitios de colapso: 8 → 28,7
sumideros). Por eso la columna importante no es el Δ absoluto sino el **Δ RELATIVO**, que descuenta la
marea: **+11,3 % → +18,0 %**. Sigue subiendo, pero el crecimiento real es de un factor ~1,6, no ~2.

## 5. El observable secundario κ_V se DA VUELTA

| | N=2000 | N=4000 |
|---|---|---|
| Δκ_V medio (III − I) | **+0,0626** | **−0,0619** |
| Δκ_V mediano | +0,0440 | −0,0766 |
| pares con κ_V(III) > κ_V(I) | 9/13 | **2/13** |
| Spearman del Δκ_V par-a-par contra N=2000 | 1,00 | +0,412 |

Esto es un cambio de signo completo, no una atenuación. A N=2000 la Clase III tendía a tener κ_V más
alto; a N=4000 la Clase I lo tiene más alto en 11 de 13 pares. **En el observable κ_V, el criterio de
falsación que declaró el equipo ("si se invierte al subir resolución, era artefacto de baja
resolución") se cumple literalmente.** No se declara cierre: se deja el número escrito y se señala que
las dos métricas de la misma batería apuntan en direcciones opuestas — la de masa se mantiene, la de
κ_V se invierte.

## 6. Auditoría del grafo: qué es realmente lo que distingue a los pares

### 6.1 A N=4000 el par sigue siendo el par, pero la etiqueta de clase es sobre todo una etiqueta de densidad

El grado medio del grafo reconstruido a cada resolución, promediado sobre los 13 pares:

| | grado medio Clase I | grado medio Clase III | Δaristas medio (III − I) | pares con III con MENOS aristas |
|---|---|---|---|---|
| N=2000 | 3,53 | 3,29 | −243 | 12/13 |
| N=4000 | 3,53 | 3,31 | −430 | 10/13 |

La brecha de densidad es **una propiedad estable de la regla**: el Δ de aristas por nodo entre las dos
resoluciones correlaciona **Pearson +0,976** (Spearman +0,862). O sea, la regla Clase III de cada par
teje sistemáticamente un grafo **más ralo** que su pareja Clase I, y lo sigue haciendo cuando se la
reconstruye con el doble de nodos.

### 6.2 Y esa densidad, sola, explica el Δmasa

| | correlación Δaristas ↔ Δmasa | recta ajustada (Δaristas por nodo) | intercepto = Δmasa a densidad IGUAL |
|---|---|---|---|
| N=2000 | Pearson **−0,632** · Spearman −0,681 | Δmasa = +0,0041 **−0,0444**·Δaristas/nodo | **+0,0041 ± 0,0032 · p = 0,221** |
| N=4000 | Pearson **−0,791** · Spearman −0,769 | Δmasa = +0,0045 **−0,1209**·Δaristas/nodo | **+0,0045 ± 0,0053 · p = 0,409** |

Dos cosas:

1. **A las dos resoluciones, la ventaja residual de la Clase III cuando las dos reglas tienen la misma
   densidad es indistinguible de cero** (+0,004, p=0,22 y p=0,41). Casi todo el Δmasa que mide esta
   batería es la sombra de que la Clase III teje más ralo. *Salvedad honesta:* el intercepto es una
   extrapolación — casi todos los pares tienen Δaristas negativo (rango −0,55 a +0,05 aristas por nodo a
   N=2000), así que "densidad igual" es el borde de los datos, no su centro. Con n=13 el intercepto
   tiene poca potencia: este número **no prueba que no haya efecto**, dice que esta batería no puede
   verlo separado de la densidad.
2. **Lo que crece con la resolución es la PENDIENTE, no el residuo**: −0,0444 → −0,1209, casi ×2,7. La
   misma diferencia de densidad entre dos reglas produce, con más partículas, una diferencia de masa
   final casi tres veces mayor. Ésa es la forma precisa en que "el efecto escala" en estos datos: **no
   escala una propiedad exclusiva de la Clase III; escala la sensibilidad del colapso a la densidad del
   tejido.**

Esto encaja pieza por pieza con `FASE6_O3D_barrido_kcap_phantom_CS.md`, donde controles Erdős-Rényi
emparejados en número de aristas caen sobre la misma recta masa-vs-pendiente que las reglas
estructuradas, y el grado medio se lleva la significancia de `kcap`. Acá se llega al mismo lugar por otro
camino y con otro diseño (pares emparejados en K y kcap, no barrido de kcap), y además se agrega que
**esa dominancia de la densidad se REFUERZA al subir la resolución** (r pasa de −0,63 a −0,79).

## 7. ¿Escala la geometría inicial o la física del colapso?

La pregunta la abre `FASE6_O4A_solver_independiente_CS.md`: el observable medido sobre las condiciones
iniciales, **sin integrar un solo paso**, predice el resultado de Phantom con r = +0,98. Si eso es así,
un Δmasa más grande a N=4000 podría ser simplemente que la nube de la Clase III **nace** más apelotonada
a esa resolución, sin que la gravedad tenga nada que ver.

Se midió directamente (`cs090_fase6_o3a_geometria_ic.py`): friends-of-friends sobre las IC a t=0, con la
longitud de enlace en unidades de la separación media entre partículas (b = 0,20 / 0,30 / 0,50 — la
convención estándar, y la única que hace comparables dos resoluciones cuando la caja y la masa total son
fijas) y umbral de grumo en masa física fija (47,0 = 5 partículas de N=2000). Se reusa el `fof_masa` de
O4-A tal cual, para que las dos tareas usen la misma vara.

| | Δ FoF b=0,20 | Δ FoF b=0,30 | Δ FoF b=0,50 | pares con III más apelotonada (b=0,20) |
|---|---|---|---|---|
| N=2000 | **+0,0636** | +0,0460 | +0,0380 | 12/13 |
| N=4000 | **+0,0440** | +0,0301 | +0,0295 | 10/13 |

**La ventaja geométrica inicial de la Clase III NO crece con la resolución: baja ~30 %.** Y sin embargo:

| correlación Δgeometría(t=0) ↔ Δmasa final de Phantom | N=2000 | N=4000 |
|---|---|---|
| FoF b=0,20 | Pearson +0,459 | **+0,720** |
| FoF b=0,30 | Pearson +0,494 | **+0,773** |
| FoF b=0,50 | Pearson +0,515 · Spearman +0,468 | **+0,787 · Spearman +0,786** |

O sea, las dos cosas a la vez: **la ventaja de partida se achica, y al mismo tiempo predice mucho mejor
el resultado final.** El cociente Δmasa/Δgeometría pasa de 0,149 a 0,398 (×2,7 — el mismo factor que la
pendiente de §6.2, lo cual no es casualidad: la geometría del layout la fija la densidad del grafo).

La lectura que estos números permiten, sin ir más lejos que ellos: a N=4000 el resultado de Phantom está
**más determinado** por cómo nació la nube que a N=2000 (Pearson +0,79 contra +0,52), y la gravedad
**amplifica más** una ventaja de partida que en realidad es más chica. No se puede decidir con esta
batería si eso es "física del colapso que escala" o "la resolución dejando ver mejor una geometría que
siempre estuvo ahí": las dos predicen lo mismo acá. Lo que sí queda medido es que **no es la geometría
inicial la que crece** — crece la conversión de geometría en masa.

## 8. Costo REAL de N=8000, medido dos veces

### 8.1 El exponente real de `layout_resortes`, medido

El cuello de botella no es Phantom (**19–62 s** por corrida a N=4000, medido sobre las 26): es
`layout_resortes`, que se llevó **618–861 s por grafo**. Se cronometró
en limpio, con la máquina libre, la misma regla y los mismos 100 iters
(`logs_o3a/bench_escala_layout.log`):

| N | tiempo del motor relacional | **tiempo de `layout_resortes`** |
|---|---|---|
| 1000 | 1,2 s | 23,3 s |
| 1414 | 2,0 s | 58,2 s |
| 2000 | 3,2 s | 98,4 s |
| 2828 | 4,0 s | 188,0 s |
| 4000 (medido antes, un grafo solo) | 3,7 s | **656 s** |

*(Honestidad sobre las condiciones: las cuatro filas de 1000 a 2828 se cronometraron mientras corría el
sondeo de N=8000, así que están si acaso INFLADAS por contención. Eso hace que el exponente calculado
abajo sea conservador —un t(2000) inflado achica el cociente 4000/2000—, no optimista.)*

El exponente entre 1000 y 2828 es **2,01** (O(N²) limpio, como decía la teoría). Pero entre 2000 y 4000
el exponente medido es **2,74**: el algoritmo se sale de O(N²) al cruzar el tamaño en que los arreglos
temporales N×N×3 dejan de caber en la jerarquía de caché y el proceso pasa a estar limitado por ancho de
banda de memoria — coherente con lo observado en §8.2, donde los procesos de N=8000 se quedaban en 40-90 %
de CPU en vez de saturar su núcleo. Extrapolando con el exponente **medido en el tramo relevante** (2,74,
no el teórico 2,0):

**t(N=8000) ≈ 656 s × 2^2,74 ≈ 4 380 s ≈ 73 minutos por grafo, corriendo solo.**

### 8.2 La segunda corrida de N=8000, con la máquina libre

El primer intento (§2) se abortó con la máquina saturada por ~20 agentes en paralelo y 15,6 GB de swap
en uso, y en ese momento se atribuyó el fracaso a falta de RAM. Al volver, la máquina estaba libre y se
verificó que tiene **64 GB de RAM, no 16** — así que la explicación anterior era incorrecta y había que
volver a medir. Se relanzaron **4 grafos de N=8000 en paralelo** (los dos pares extremos del ranking:
`batch4-r23/r10`, el más favorable a N=2000, y `batch4-r57/r43`, el más invertido) a las 23:09:39.

Comportamiento observado, con la memoria ya no siendo el límite:

| | valor medido |
|---|---|
| RSS por proceso | 4,9 – 7,2 GB (28 GB entre los 4; nunca cerca del techo de 64 GB) |
| CPU por proceso | 40 % al principio (compitiendo con la tanda de N=4000), **~90 % una vez sola** |
| tiempo transcurrido al abortar | **17 min 51 s** (23:09:39 → 23:27:30) |
| layouts terminados | **0 de 4** |
| costo esperado por grafo, del exponente medido en §8.1 | ~73 min solo · ~90–120 min con 4 en paralelo |

**Decisión, y por qué:** completar los 4 grafos habría llevado el sondeo hasta pasada la 01:00, y después
todavía faltaba Phantom a N=8000 — que en el antecedente de `ON77_sistemaA_cierre` tuvo que abortarse por
colapso del paso de tiempo cerca de los ~122 sumideros que se forman a esa resolución, o sea que ni
siquiera está garantizado que devuelva un número. Se aplicó la instrucción explícita de la tarea: *es
preferible una serie N=2000/4000 completa y bien medida que una serie N=8000 a medias.* **Se abortaron
los 4 grafos y se entrega la serie de dos puntos con los 13 pares completos.**

Lo que hace falta para que N=8000 sea viable, dicho para la próxima vez: **no es más RAM, es cambiar
`layout_resortes`**. Un layout con árbol de Barnes-Hut o con vecinos por celdas (O(N log N) en vez de
O(N²)) bajaría los 73 min a menos de 2. Pero eso **cambia el protocolo** respecto de los puntos de
N=2000 y N=4000 con los que hay que comparar, así que habría que revalidar la serie entera con el layout
nuevo, no mezclar. Queda como pendiente explícito y NO se hizo acá.

## 9. Contra el criterio de falsación que declaró el equipo

El equipo escribió: *si la diferencia se va a 0 o se invierte al subir resolución, el efecto de Fase V-B
sería artefacto de baja resolución; si se mantiene o crece, es efecto físico que escala.* Contra esa
vara, y **sólo con los dos puntos que hay**:

| Observable | ¿se va a 0? | ¿se invierte? | ¿se mantiene o crece? |
|---|---|---|---|
| **Fracción de masa (métrica principal)** | no | no, en media | **crece**: +0,0095 → +0,0175 (relativo +11,3 % → +18,0 %) |
| **Fracción de masa, par a par** | — | **sí en 4 de 13** | se mantiene en 9/13; en los 4 de señal grande, 4/4 |
| **κ_V (métrica secundaria)** | no | **sí, entero**: +0,063 → −0,062 | no |

O sea: **la vara no da un veredicto único, porque las dos métricas de la misma batería la responden
distinto.** Es exactamente la situación en la que la regla del proyecto (no cerrar sin Alexis) tiene
sentido: hay una lectura defendible en que el efecto escala (masa) y otra defendible en que no (κ_V y el
36 % de pares que se dan vuelta), y elegir entre ellas no es una decisión que salga de estos números.

Y hay una tercera lectura, la de §6, que no estaba en la vara original porque el contexto llegó después:
puede que la vara esté midiendo la cosa equivocada. Si la diferencia entre Clase III y Clase I es, en lo
esencial, una diferencia de **densidad del grafo** (§6.1: grado medio 3,29–3,31 contra 3,53 en las dos
resoluciones, Δaristas conservado entre resoluciones con Pearson +0,976), y si al descontar esa densidad
la ventaja de clase queda en +0,004 con p=0,22 y p=0,41 (§6.2), entonces "¿el efecto de clase escala?"
puede no ser la pregunta bien planteada. Lo que sí escala, medido, es **la sensibilidad del colapso a la
densidad del tejido**: la pendiente pasa de −0,044 a −0,121.

### 9.1 Cómo se conecta con lo que llegó después de que esta tarea se lanzara

- **`FASE6_O3D_barrido_kcap_phantom_CS.md`** (controles Erdős-Rényi emparejados en aristas caen sobre la
  misma recta masa-vs-pendiente que las reglas estructuradas; la densidad domina). O3-A llega al mismo
  lugar con un diseño independiente — pares emparejados en K y kcap, no barrido de kcap — y **agrega una
  dimensión nueva: esa dominancia de la densidad no se diluye al subir la resolución, se REFUERZA**
  (r de −0,63 a −0,79). Si O3-D dice "la densidad explica la separación", O3-A dice "y la explica cada
  vez mejor cuanta más resolución hay".
- **`FASE6_O4A_solver_independiente_CS.md`** (el observable sobre las condiciones iniciales, sin integrar
  nada, predice a Phantom con r=0,98). §7 mide qué escala: **no la geometría inicial** — la ventaja
  geométrica de partida de la Clase III BAJA de +0,064 a +0,044 al doblar N — **sino la conversión de
  geometría en masa**, y el poder predictivo de la IC sobre el resultado final (de r=+0,52 a r=+0,79).
  Dicho de otro modo: a más resolución, Phantom se parece más a una función determinista de su condición
  inicial, y el margen para "física del colapso" que no estuviera ya escrita en la IC se **achica**, no
  crece.

## 10. Limitaciones que hay que tener a la vista al leer esto

1. **Dos puntos, no tres.** "Crece" con N=2000 y N=4000 es una recta entre dos puntos: no distingue
   crecimiento real de una tendencia que se dé vuelta en N=8000. La serie 8→29→122 sumideros muestra que
   entre 4000 y 8000 pasa algo cualitativo (el número de sumideros más que se cuadruplica), así que
   extrapolar desde dos puntos es especialmente frágil acá.
2. **Una corrida por regla.** No hay repeticiones con otra `seed_layout` ni con otra semilla de
   turbulencia, así que **no hay una barra de error interna por regla** y no se puede decir cuánto de los
   4 signos que se dan vuelta es ruido de una sola realización. Es la limitación más seria del diseño y
   es heredada de Fase V-B.
3. **n=13.** Suficiente para el Wilcoxon de la métrica principal (p=0,040) pero insuficiente para el test
   de signos (p=0,267) y para estimar bien el intercepto de §6.2. Los 24 pares limpios restantes están
   disponibles y correrlos a N=4000 costaría ~11 min de layout por tanda de 10 en paralelo más ~30 s de
   Phantom por corrida — es la ampliación más barata que queda sobre la mesa.
4. **Subir N reconstruye la regla, no reetiqueta un grafo.** Ya está dicho en §1.1 y es una decisión
   deliberada (es el mismo sentido de "resolución" del antecedente ON77), pero significa que parte de la
   variación entre N=2000 y N=4000 es **otra realización de la misma ley**, no sólo más partículas. El
   test complementario —congelar el grafo de 2000 nodos y partir cada partícula SPH— mide la otra mitad
   de la pregunta y sigue sin hacerse.
5. **El intercepto de §6.2 es una extrapolación al borde de los datos**, no una medición en el centro:
   casi todos los pares tienen Δaristas negativo. Decir "a densidad igual no hay efecto" es más fuerte de
   lo que estos datos aguantan; lo que aguantan es "esta batería no puede separar el efecto de clase del
   de densidad".
6. **κ_V se invierte y no se explica.** El informe registra el número y no ofrece mecanismo. Con 8
   sumideros por corrida a N=2000 contra 28,7 a N=4000, es plausible que κ_V agregado esté midiendo cosas
   distintas en las dos resoluciones (el promedio sobre 8 objetos frente al promedio sobre 29), pero eso
   es una hipótesis no testeada.

## 11. Archivos

**Scripts (los 6 primeros ya existían y se reusaron sin tocar; el último es nuevo de esta sesión):**

- `cs090_fase6_o3a_convergencia_resolucion.py` — worker (grafo → IC → Phantom → métricas) y modo `tabla`.
- `cs090_fase6_o3a_seleccionar_pares.py` — el criterio mecánico de selección de §1.2.
- `cs090_fase6_o3a_correr_phantom.sh` — Phantom serial + poda de dumps intermedios.
- `cs090_fase6_o3a_grafos_por_N.py` · `cs090_fase6_o3a_verificar_ic.py` — generación en paralelo y
  chequeo de integridad de las IC.
- `cs090_fase6_o3a_analizar.py` — tabla por par×N, resumen estadístico y figura.
- **`cs090_fase6_o3a_geometria_ic.py`** (nuevo) — el FoF sobre las condiciones iniciales de §7; importa
  `fof_masa` de `cs090_fase6_o4a_observable_comun.py` sin modificarlo.

**Datos:**

- `cs090_fase6_o3a_resolucion_crudo.csv` — 106 filas, una por (regla, N).
- `cs090_fase6_o3a_pares_por_N.csv` — 26 filas, una por (par, N): Δmasa, Δκ_V, sumideros, aristas, diámetro.
- `cs090_fase6_o3a_resumen_estadistico.csv` — la tabla de §4.
- `cs090_fase6_o3a_geometria_ic.csv` — 52 filas, FoF y dispersión de densidad de cada IC.
- `cs090_fase6_o3a_delta_vs_N.png` — Δ absoluto, Δ relativo y Δκ_V contra N, una línea por par.
- `cs090_fase6_o3a_pares_seleccionados.json` — los 13 pares y su criterio.

**Corridas de Phantom:** `/Users/alexis/phantom_cs073/bateria_fase6_o3a_resolucion/N4000/` (26 carpetas,
cada una con `meta_regla.json`, `resultado_o3a.json`, el primer y el último dump y el `.sink`; los 498
dumps intermedios podados). El directorio `N8000/` quedó vacío a propósito: las 4 carpetas del sondeo
abortado se borraron para que no queden restos parciales que un análisis futuro pueda tomar por datos.
Las 26 IC de N=4000 pasan `cs090_fase6_o3a_verificar_ic.py` sin una sola observación
(N=4000 exactas, masa por partícula 4,7, masa total 18800 en las 26).

**Logs:** `logs_o3a/ic_N4000.log`, `logs_o3a/ic_ronda2_N4000.log`, `logs_o3a/phantom_N4000.log`,
`logs_o3a/phantom_ronda2_N4000.log`, `logs_o3a/ic_sonda_N8000.log`, `logs_o3a/bench_N4000.log`,
`logs_o3a/bench_escala_layout.log`.

**No se declara cierre ni veredicto. No se hicieron commits de git. Ningún script congelado fue
modificado.**
