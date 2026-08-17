# FASE VI · O4-A — Réplica con un integrador gravitacional INDEPENDIENTE de Phantom

**Checkpoint final de la serie** · propuesta de GPT-5.6 Sol (F6-08) · 11-ago-2026
Entorno: `./venv/bin/python`, `numpy`/`scipy`/`sarracen 1.3.1`. **No se corrió Phantom
nuevo.** No se modificó ningún archivo previo: todo lo de esta tarea es nuevo.

---

## 0. Qué se preguntó y cómo se lo atacó, en simple

Todo el resultado de la Fase V-B descansa en un solo motor: Phantom. La pregunta de
O4-A es si el efecto sobrevive a **cambiar de motor**: si un simulador construido con
otra física ordena los mismos pares en el mismo sentido, el orden es de la receta y no
del simulador.

En vez de instalar Gadget o AREPO (horas de compilación, y una madriguera de
dependencias), se escribió un motor propio, chico y completamente independiente:

| | Phantom | El motor de esta tarea |
|---|---|---|
| física | SPH: fluido con presión, viscosidad de choque, gravedad propia | **gravedad y nada más** |
| gravedad | árbol (tree, `tree_accuracy=0.5`), suavizado adaptativo ligado a `h` | **suma directa O(N²)**, suavizado de Plummer fijo |
| integración | pasos individuales por partícula, jerárquicos | **leapfrog KDK de paso global fijo** |
| grumos | **sumideros**: nacen por umbral de densidad y acretan vecinas | no hay: los grumos son grumos, nada desaparece |
| implementación | Fortran, ~10⁵ líneas, 20 años de desarrollo | Python/numpy, ~250 líneas, escrito hoy |

Analogía: Phantom es la olla de sopa simulada al detalle — burbujas, remolinos, grumos
que se tragan a sus vecinos. Este motor es la misma olla pensada como **canicas que
sólo se atraen**. No van a dar el mismo número. La pregunta es si dan el mismo
**ranking**.

---

## 1. El motor: `cs090_fase6_o4a_nbody.py`

- **Fuerza**: `a_i = G Σ_j m_j (x_j − x_i) / (|x_j − x_i|² + ε²)^{3/2}` — suma directa,
  todos contra todos, con suavizado de Plummer. Implementada con el truco
  `Σ_j w_ij (x_j − x_i) = (W·X)_i − (Σ_j w_ij)·x_i`, que convierte las tres componentes
  en un solo producto matriz-matriz (BLAS).
- **Potencial**: `U_ij = −G m_i m_j / (r² + ε²)^{1/2}`, que es el par **consistente** con
  esa fuerza. Esto importa: si uno mezcla la fuerza suavizada con el potencial sin
  suavizar, la "conservación de energía" deja de ser una prueba de nada.
- **Integrador**: velocity-Verlet / leapfrog patada-deriva-patada, **paso fijo**. Se
  eligió paso fijo a propósito: el leapfrog de paso fijo es simpléctico, así que la
  energía **oscila pero no deriva**, y entonces "¿se conservó la energía?" es un
  termómetro honesto de si el paso alcanza.
- **Unidades y fronteras**: las mismas de Phantom en estas corridas — G = 1, `umass` =
  `udist` = 1, masa total 18800 en 2000 partículas de 9.4. Los volcados **no traen
  `xmin/xmax`**, o sea que Phantom corrió con **frontera aislada** (no periódica): por
  eso acá también la gravedad es aislada, sin sumas de Ewald.

### 1.1 Un dato que justifica que "gravedad sola" no sea un disparate

En estas corridas de Phantom la presión es energéticamente irrelevante. Del `run.log`
de cualquier corrida (ejemplo `A2-B0-C2-r1_I`):

```
Etot=-2.815E+06, Ekin= 2.538E+04, Etherm= 4.500E-01, Epot=-2.840E+06
```

`Etherm/|Epot| ≈ 1.6 × 10⁻⁷`. El gas es frío hasta el ridículo (`ieos=1` isotermo con
`polyk = 0.3`). Es decir: **en el balance energético, estas simulaciones de Phantom ya
son un problema de gravedad pura**. Lo que el motor de esta tarea deja afuera no es la
presión (que no pesa) sino tres cosas concretas: **(a)** la disipación por viscosidad
artificial en los choques, **(b)** la creación y acreción de sumideros, y **(c)** el
suavizado gravitatorio adaptativo de SPH (que se ajusta a la densidad local) frente al
ε fijo de acá. Ver §6.

---

## 2. Validación del integrador (sin esto los números no valen nada)

`./venv/bin/python cs090_fase6_o4a_nbody.py` corre la batería. Resultados:

| prueba | qué mide | resultado |
|---|---|---|
| **Dos cuerpos, órbita circular** | dos masas iguales, un período completo `T = 2π√(d³/(G·2m))`; la órbita debe cerrarse | error máximo de posición **1.03 × 10⁻⁷**, de velocidad **1.46 × 10⁻⁷**; deriva de energía **−2.9 × 10⁻¹⁵** |
| **Dos cuerpos, elipse e = 0.5** | lanzada en el apoastro por vis-viva; prueba más dura porque la velocidad varía mucho | error máximo de posición **2.8 × 10⁻⁹**, de velocidad **1.5 × 10⁻⁹**; deriva **1.9 × 10⁻¹³** |
| **Esfera de Plummer en virial** | 500 cuerpos en equilibrio; no debe colapsar ni explotar en un tiempo de cruce | radio de media masa cambia **−5.1 %**; cociente virial final **0.521** (el esperado es 0.5); deriva **−2.7 × 10⁻⁹** |
| **Momento lineal** | en gravedad aislada se conserva exacto (acción-reacción) | se verifica dentro del redondeo en las corridas reales |

Las dos primeras prueban la **física** (reproduce órbitas con solución analítica
cerrada), la tercera prueba que el motor **no inventa ni disipa estructura** (una
configuración en equilibrio se queda en equilibrio), y la cuarta prueba que la suma
directa no tiene errores de signo o de índice.

### 2.1 Convergencia en paso de tiempo — cómo se eligió `dt`

Script: `cs090_fase6_o4a_correr.py convergencia` · salida
`cs090_fase6_o4a_convergencia_dt.csv`. Se corrieron dos sistemas reales del proyecto
(`A2-B0-C2-r1` y `A2-B0-C2-r17`) hasta `t = 0.5` con cuatro pasos distintos:

| dt | pasos | deriva de energía (r1 / r17) | observable final (r1 / r17) |
|---|---|---|---|
| 4 × 10⁻³ | 125 | −6.78e−6 / −1.52e−5 | **0.1800 / 0.2950** |
| 2 × 10⁻³ | 250 | −1.70e−6 / −3.81e−6 | **0.1800 / 0.2950** |
| 1 × 10⁻³ | 500 | −4.24e−7 / −9.54e−7 | **0.1800 / 0.2950** |
| 5 × 10⁻⁴ | 1000 | −1.06e−7 / −2.38e−7 | **0.1800 / 0.2950** |

La deriva cae **exactamente por un factor 4 cada vez que se parte el paso en dos** — que
es la firma `dt²` del leapfrog de segundo orden, o sea que el integrador se comporta
como dice la teoría — y el observable **no se mueve ni un dígito** en cuatro pasos de
tiempo que abarcan un factor 8.
Se adoptó **dt = 2 × 10⁻³** (250 pasos) para la tanda principal. La decisión también
tiene un motivo práctico declarado: la máquina estaba con carga media ~500 (decenas de
agentes en paralelo), así que cada evaluación de fuerza costaba ~0.9 s en vez de ~0.05 s.

---

## 3. La muestra: 10 pares, en dos estratos, elegidos antes de mirar nada

De los **37 pares que siguen siendo contraste válido** tras la corrección de diámetro
(`cs090_fase6_reanalisis_40pares_corregido.csv`, estado `valido`) se tomaron 10 por un
criterio mecánico, no por conveniencia:

- **estrato SEÑAL FUERTE** — los 5 con mayor |Δfracción| en Phantom (0.025 – 0.034)
- **estrato EMPATE** — los 5 con menor |Δfracción| en Phantom (0.001 – 0.002)

**Por qué en estratos.** En los empates el orden de Phantom es prácticamente ruido de
redondeo: `Δfracción = 0.001` son **2 partículas de 2000**. Ningún motor puede
"acertar" eso, y mezclarlos con los pares de señal fuerte para reportar un único
porcentaje escondería exactamente la información que interesa. Separados, la predicción
es explícita: **si el efecto es real, el estrato fuerte debería concordar y el de empate
debería salir cerca de cara-o-cruz**.

Se reusaron los archivos `cosmogenesis_ic.txt` originales de
`/Users/alexis/phantom_cs073/bateria_fase5b_*` — **no se regeneró ninguna condición
inicial**. Se verificó que las 20 reglas dan 20 archivos distintos (md5) y que cada
regla tiene una sola carpeta candidata (sin colisiones de nombre, la lección del bug
anterior).

---

## 4. El observable análogo (y por qué NO es la misma cantidad)

Phantom mide **fracción de masa en sumideros**. Acá no hay sumideros, así que el
análogo es **fracción de masa en las regiones más densas al final de la integración**,
medida con *friends-of-friends*: cada partícula "se da la mano" con toda vecina a menos
de `ell`, los grupos son las cadenas de manos, y el observable es la fracción de masa en
grupos de al menos 5 partículas.

- **elección principal declarada de antemano**: `ε = 0.6` (= `r_crit` de Phantom, el
  radio que decide si dos concentraciones son un sumidero o dos), `ell = 1.0`,
  `n_min = 5`, medido en `t = 0.5` (el mismo `tmax` de Phantom).
- **escala de referencia**: la separación media entre partículas es
  `97.6 / 2000^{1/3} ≈ 7.8`. Enlazar a `ell = 1.0` sólo agrupa material que de verdad
  colapsó — no es un umbral generoso.
- **robustez**: se reporta la grilla completa `ell ∈ {0.6, 1.0, 2.0} × n_min ∈ {3, 5, 10}`
  más una variante por densidad local del 8.º vecino con umbral 100× y 1000× la media.

**Reconocimiento explícito**: *no es la misma cantidad que mide Phantom.* Un sumidero de
Phantom se traga partículas y las saca del gas; un grupo FoF sólo dice "estas están
juntas". Los valores absolutos no son comparables. **La comparación válida es de ORDEN.**

### 4.1 Dos controles que se agregaron y que resultaron decisivos

1. **El mismo observable sobre las condiciones iniciales (`t = 0`, cero dinámica).**
   Es el control barato y obligatorio: si el orden ya está en la geometría de partida,
   entonces "los dos motores coinciden" no habla de la dinámica, habla de que ambos
   heredaron el mismo punto de partida.
2. **La MISMA vara aplicada al estado final de Phantom**
   (`cs090_fase6_o4a_observable_comun.py`): se reconstruye el estado final de Phantom
   como nube de masas puntuales — gas vivo (9.4 c/u) **más** los 8 sumideros con la masa
   que acretaron — y se le corre el mismo FoF, con criterio de tamaño en **masa**
   (≥ 5 × 9.4 = 47) en vez de en número de miembros, porque un sumidero es un solo punto
   que ya vale por decenas de partículas. Esto cierra la rendija "¿y si el acuerdo o el
   desacuerdo viene sólo de usar dos reglas distintas?".

---

## 5. Resultados

20 corridas (10 pares), dt = 2 × 10⁻³, 250 pasos, ε = 0.6, t = 0.5.
Salud numérica: deriva de energía entre **−3.8 × 10⁻⁶ y −7.8 × 10⁻⁷** en las 20;
1076 s de reloj en total con 6 procesos.
Crudo: `cs090_fase6_o4a_corridas_nbody.csv`.

### 5.1 La comparación central — par por par

`cs090_fase6_o4a_comparacion_pares.csv`. Observable principal `FoF ell=1.0, n_min=5`.

| par | estrato | Phantom Δ(III−I) | motor propio Δ(III−I) | gana Phantom | gana motor propio | ¿mismo orden? |
|---|---|---|---|---|---|---|
| piloto_B r1 vs r17 | fuerte | +0.0340 | +0.1150 | III | III | **sí** |
| batch4 r23 vs r10 | fuerte | +0.0325 | +0.0140 | III | III | **sí** |
| batch3 r120 vs r111 | fuerte | +0.0295 | +0.0175 | III | III | **sí** |
| batch3 r104 vs r60 | fuerte | +0.0270 | +0.0260 | III | III | **sí** |
| v2_G r12v1fix vs r19 | fuerte | +0.0250 | +0.1505 | III | III | **sí** |
| batch3 r59 vs r58 | empate | −0.0010 | +0.0205 | I | III | no |
| batch4 r39 vs r26 | empate | +0.0010 | −0.0060 | III | I | no |
| piloto_C r6 vs r14 | empate | −0.0015 | +0.0235 | I | III | no |
| batch4 r38 vs r72 | empate | +0.0020 | +0.0045 | III | III | **sí** |
| v2_H r9 vs r39 | empate | −0.0020 | −0.0340 | I | I | **sí** |

- **estrato SEÑAL FUERTE: 5/5** (binomial bilateral p = 0.0625)
- **estrato EMPATE: 2/5** (p = 1.0) — cara-o-cruz, exactamente lo predicho
- total 7/10 (p = 0.34)

**Y el 5/5 del estrato fuerte se sostiene en las 11 definiciones de "región densa"
probadas** (grilla `ell × n_min` completa + las dos variantes de densidad por vecino):
`cs090_fase6_o4a_robustez_grilla.csv`, columna `coinc_fuerte_fin` = 5 en 10 de 11 filas
y 4 en la restante (`ell=0.6, n_min=3`, la más ruidosa). Los empates oscilan entre 1/5 y
4/5 según la definición, que es lo que uno espera de ruido.

Correlación entre el observable de Phantom (fracción en sumideros) y el propio, a través
de las 20 corridas: **Pearson +0.806, Spearman +0.753** (p = 1 × 10⁻⁴); según la
variante del observable, Spearman va de **+0.753 a +0.934**.

### 5.2 La misma vara en las dos mesas

`cs090_fase6_o4a_observable_comun.py` · `cs090_fase6_o4a_observable_comun.csv` ·
`cs090_fase6_o4a_ordenes_ell{0.6,1.0,2.0}.csv`.

Reconstruyendo el estado final de Phantom como nube de masas puntuales (gas vivo + los
8 sumideros con su masa) y corriéndole **el mismo FoF**:

| longitud de enlace | Phantom vs motor propio: Pearson (20 corridas) | sesgo medio (propio − Phantom) | orden coincidente |
|---|---|---|---|
| ell = 0.6 | **+0.956** | +0.045 | **9/10** (fuerte 5/5, empate 4/5) · p = 0.021 |
| ell = 1.0 | **+0.975** | −0.038 | **9/10** (fuerte 5/5, empate 4/5) · p = 0.021 |
| ell = 2.0 | **+0.989** | −0.105 | 8/10 (fuerte 5/5, empate 3/5) · p = 0.109 |

Los valores absolutos quedan asombrosamente cerca — por ejemplo `A2-B0-C2-r12v1fix` a
ell = 1.0: Phantom 0.107, motor propio 0.106; `A2-B0-C2-r1`: 0.188 vs 0.180. El sesgo
sistemático aparece recién a ell = 2.0, y va en el sentido esperado: **Phantom aglomera
más a escala grande** (−0.105), porque su viscosidad de choque disipa energía cinética y
mis canicas no disipan nada, así que rebotan y se dispersan más.

### 5.3 El control que cambia la lectura: la geometría inicial ya lo sabía

Medido **sobre las condiciones iniciales, sin integrar un solo paso**:

| relación (20 corridas) | Pearson | Spearman |
|---|---|---|
| Phantom (fracción en sumideros) ↔ FoF de la IC en t = 0, ell = 0.6 | **+0.980** | **+0.959** |
| Phantom (fracción en sumideros) ↔ motor propio (final) | +0.844 | +0.785 |
| motor propio (final) ↔ FoF de la IC en t = 0 | +0.855 | — |
| **parcial: Phantom ↔ motor propio, controlando la IC** | **+0.062** (p = 0.80) | — |

Y en concordancia de orden, el t = 0 puro da **5/5 en el estrato fuerte** — igual que la
integración completa.

Traducido: **el ordenamiento que comparten los dos motores ya está escrito en las
condiciones iniciales.** Una vez que se descuenta cuán apelotonada nació cada nube, lo
que agrega correr el motor de gravedad no explica nada extra de lo que hizo Phantom
(correlación parcial +0.06, p = 0.80).

Consistente con eso, el observable **incremental** (final menos inicial — "lo que agregó
la dinámica") va **en contra**: 0/5 en el estrato fuerte, con correlación **negativa**
(Pearson −0.83 sobre los 10 Δ). Lectura probable: es techo/saturación, no anti-señal —
las nubes Clase III arrancan más apelotonadas, así que les queda **menos margen** para
subir en un observable acotado entre 0 y 1. No se testeó esa explicación; queda como
hipótesis.

---

## 6. Limitaciones — dicho sin maquillaje

1. **No es el mismo experimento, es otra física.** Sin sumideros, sin viscosidad de
   choque, sin presión, sin suavizado adaptativo. Los sumideros de Phantom, además,
   **cambian la gravedad**: una vez que 20 partículas se convierten en un punto, el
   campo local ya no es el mismo. Que el orden coincida es informativo; si no hubiera
   coincidido, la primera explicación a descartar sería la diferencia de física, no
   "Phantom se equivoca".
2. **La muestra es chica y está estratificada a propósito.** 10 pares de 37. El 5/5 del
   estrato fuerte tiene p = 0.0625 por binomial — no cruza 0.05 ni con generosidad. Lo
   que lo hace fuerte no es ese p sino que **se repite en 11 definiciones distintas del
   observable y en las dos varas**, y que va acompañado del 2/5 predicho en los empates.
3. **El observable análogo no es el de Phantom** (§4), y aunque §5.2 arregla mucho de
   eso usando la misma vara en las dos mesas, el estado final de Phantom que uso ahí
   trata cada sumidero como una masa puntual en su centro — que es una idealización.
4. **El hallazgo de §5.3 le pone un techo a todo lo demás.** La concordancia entre
   motores es real y muy alta (r ≈ 0.96–0.99), pero es en buena medida concordancia
   *heredada*: ambos motores recibieron el mismo punto de partida y ese punto de partida
   ya contenía el ranking. Un test de "motor independiente" prueba menos de lo que
   parece cuando la respuesta está en el input. **Este control no estaba pedido en la
   consigna y es, para mi gusto, el número más importante de la tarea.**
5. **ε = 0.6 es una elección.** No se barrió ε. Un ε distinto cambia cuánto colapsan los
   grumos; sí se verificó que el resultado es estable frente al paso de tiempo y frente
   a la definición de "región densa", pero no frente al suavizado.
6. **Máquina muy cargada** (load ~500–680 por decenas de agentes en paralelo). Eso no
   afecta los números — el integrador es determinista — pero explica los tiempos
   (250 pasos ≈ 4–5 min por corrida en vez de ~15 s) y por qué la tanda es de 10 pares
   y no de 20.

---

## 7. Las dos lecturas, ambas sobre la mesa

**Lectura A — la réplica salió bien.** Un motor escrito desde cero, con otra física,
otro algoritmo de gravedad, otro integrador y otro lenguaje, reproduce el ordenamiento
de Phantom en los 5 pares donde Phantom tiene señal, con las 11 maneras de medir; y
cuando se usa la misma vara en las dos mesas, los valores absolutos coinciden con
r ≈ 0.96–0.99 y el orden en 9/10 pares. El efecto de la Fase V-B no es un artefacto del
código de Phantom.

**Lectura B — pero prueba menos de lo que parece.** La misma tanda muestra que la
fracción de masa en sumideros de Phantom está predicha con r = +0.98 por la
aglomeración que las condiciones iniciales **ya tenían antes de simular nada**, y que
controlando por eso el motor independiente no aporta información adicional (parcial
+0.06, p = 0.80). O sea: lo que O4-A confirma es que **la geometría inicial ordena** y
que **dos motores distintos la respetan** — no que la dinámica gravitatoria de Phantom
esté generando el ordenamiento.

Las dos son compatibles. Cuál pesa más depende de qué se estaba afirmando: si la
afirmación es "el contraste Clase III/Clase I sobrevive al cambio de motor", esta tanda
la apoya; si la afirmación es "el proceso dinámico convierte la diferencia topológica en
diferencia de masa colapsada", esta tanda **no la apoya y sugiere mirar el punto de
partida**.

No declaro cierre. La interpretación es de Alexis.

---

## 8. Archivos de esta tarea (todos nuevos)

**Código**
- `cs090_fase6_o4a_nbody.py` — el integrador + la batería de validación
  (`./venv/bin/python cs090_fase6_o4a_nbody.py` la corre).
- `cs090_fase6_o4a_correr.py` — selección de pares, lectura de las IC de Phantom,
  tanda principal y modo `convergencia`.
- `cs090_fase6_o4a_analizar.py` — comparación de orden y correlaciones (fin / ini / delta).
- `cs090_fase6_o4a_observable_comun.py` — la misma vara aplicada al estado final de Phantom.

**Datos crudos**
- `cs090_fase6_o4a_corridas_nbody.csv` — 20 corridas × (diagnóstico + 11 observables en
  t = 0 y en t = 0.5).
- `cs090_fase6_o4a_convergencia_dt.csv` — 8 corridas del test de paso de tiempo.
- `cs090_fase6_o4a_comparacion_pares.csv` — la tabla de §5.1.
- `cs090_fase6_o4a_robustez_grilla.csv` — la cuenta repetida en 11 observables × 3 modos.
- `cs090_fase6_o4a_observable_comun.csv` + `cs090_fase6_o4a_ordenes_ell{0.6,1.0,2.0}.csv`.

**Registros**
- `cs090_fase6_o4a_correr.log`, `cs090_fase6_o4a_convergencia.log`,
  `cs090_fase6_o4a_analisis.log`, `cs090_fase6_o4a_observable_comun.log`,
  `cs090_fase6_o4a_correlaciones_extra.log`.

Nada de Phantom se re-ejecutó; los `cosmogenesis_ic.txt` se leyeron tal cual, sin
regenerarlos (20 reglas → 20 md5 distintos, sin colisiones de carpeta). Sin commits.
