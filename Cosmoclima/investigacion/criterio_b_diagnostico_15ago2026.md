# EIT-3 · Diagnóstico del criterio (b) — 15-ago-2026

> **Estado: HALLAZGO + RECOMENDACIÓN. Nada aplicado, nada cerrado.**
> No se tocó `sim-cosmoclima.html`, ni el motor, ni ningún archivo del proyecto.
> Todo el cálculo se hizo en un directorio temporal aparte, leyendo el motor
> publicado. Ningún veredicto acá es válido sin autorización de Alexis.

**Control de que estoy midiendo lo mismo que se publicó:** volví a correr la
física completa (1.357.800 ticks, 7,9 min) reproduciendo la variante *cruda* —
la adoptada— y el CSV año-por-año coincide con
`motor/r17_cruda_por_anio.csv` en **62 años de 62, cero diferencias**. Además mi
recálculo del criterio (d) da **ρ = +0,217** contra la lluvia independiente del
sqlite: el mismo número exacto que reporta el evaluador oficial. Todo lo que
sigue está medido sobre la corrida publicada, no sobre una réplica aproximada.

---

## 1. Qué es exactamente el criterio (b)

### Dónde está escrito

| Qué | Archivo | Líneas |
|---|---|---|
| Definición y cálculo | `Web/prueba_de_concepto/motor/evaluar_contra_ground_truth.js` | 219-264 |
| Regla literal de aprobado/reprobado | mismo archivo | 410 |
| Impresión en consola | mismo archivo | 399-409 |
| Dato de contraste (ground truth) | `investigacion/fuentes/curva_empirica_lluvia_floracion_gyriosomus.csv` | 26 filas |
| Texto publicado | `Web/prueba_de_concepto/informe-cosmoclima.html` y la copia bajada de la web `investigacion/publicado_15ago2026/informe-cosmoclima.html` | 383, 395-407 |

### Definición literal, del propio código

```js
// evaluar_contra_ground_truth.js:219-222
// Métrica (b): floración real (años con floracion_documentada=1) debe
// dar JARDIN_FERTIL_pct mayor que los años control (=0).
```

Y la regla de aprobado, en una sola línea (410):

```js
console.log(`[${pasaFalla(floracion.media_JARDIN_FERTIL_floracion > floracion.media_JARDIN_FERTIL_control)}] umbral: media(floración) > media(control)`);
```

### Qué calcula, contra qué, con qué n

- **Variable del modelo:** `JARDIN_FERTIL_pct` — el **porcentaje de los ticks del
  año** que el clasificador puso en la zona Jardín Fértil. Un año tiene 365×60 =
  21.900 ticks.
- **Contra qué:** la columna `floracion_documentada` del CSV empírico: 1 =
  floración documentada, 0 = año control, vacío = sin dato.
- **n:** **13 años de floración** (1983, 1991, 1997, 2000, 2002, 2005, 2011,
  2012, 2015, 2017, 2021, 2022, 2024) contra **10 controles** (1989, 1990, 1996,
  2003, 2008, 2013, 2016, 2018, 2019, 2020). 2023 y 2025 quedan fuera (sin
  etiqueta).
- **Estadístico:** una resta de dos promedios, con **desigualdad estricta y sin
  ninguna tolerancia**. No hay test, no hay error estándar, no hay n en la
  fórmula. Si el promedio de floración supera al de control **aunque sea por
  0,001 pp**, aprueba; si empata o queda debajo, reprueba.

### El número adoptado

En la corrida adoptada (*cruda*): **floración 38,185 % vs control 38,631 %
→ (b) = −0,446 pp** (`motor/r17_cruda_por_anio_evaluacion.json`).

> ⚠️ **Inconsistencia detectada en la página publicada.** La tabla del informe
> (línea 383) y el párrafo que la explica (395-397) dicen *"38,19 vs 38,25 …
> falla por seis centésimas de punto"*. Esos son los números de la variante
> **ERA5**, no de la *cruda* que se adoptó. Las otras cuatro filas de la misma
> tabla (23,1 vs 40,9; ρ=+0,217; 42,9 vs 22,3) sí son de la *cruda*. Es decir:
> **una fila de la tabla publicada viene de una corrida distinta a las otras
> cuatro.** El valor correcto para la serie adoptada es 38,19 vs 38,63 (−0,45 pp),
> no −0,06. Lo dejo señalado; **no lo corregí**.

---

## 2. Por qué da negativo — la cadena, medida paso a paso

### 2.1 El dato de entrada SÍ separa los dos grupos

Primero hay que descartar la explicación que está escrita hoy en la página
("es un límite del dato de contraste"). **No lo es, o al menos no es eso lo que
domina.** La lluvia que efectivamente entra al motor en el punto-reloj distingue
muy bien los dos grupos:

| Variable de entrada | años floración | años control | AUC | p (permutación) |
|---|---|---|---|---|
| lluvia anual del motor | 259,8 mm | 120,5 mm | **0,831** | **0,0085** |
| pico mensual del motor | 133,1 mm | 63,0 mm | 0,808 | 0,0070 |
| lluvia anual NASA POWER (la columna del propio ground truth) | 78,9 mm | 33,3 mm | 0,854 | 0,0019 |

*AUC = probabilidad de que un año de floración tomado al azar tenga más lluvia
que un control tomado al azar. 0,5 es una moneda; 1,0 es separación perfecta.*

**La señal entra al instrumento con AUC 0,83.** El etiquetado del ground truth no
está roto: los años que la literatura llama de floración son, de verdad, los años
que llovieron en el punto.

### 2.2 Dónde se pierde

Seguí la señal por toda la cadena de cálculo, midiendo el AUC en cada eslabón:

| # | Eslabón | flo | ctl | **AUC** | p |
|---|---|---|---|---|---|
| 0 | lluvia anual que entra (mm) | 259,8 | 120,5 | **0,831** | 0,0085 |
| 1 | floración media del año `f` | 0,163 | 0,124 | 0,746 | 0,121 |
| 1b | floración **máxima** del año | 0,296 | 0,217 | **0,785** | 0,044 |
| 2 | `LF` media del año = 4·f·(1−f) | 0,495 | 0,419 | 0,654 | 0,257 |
| 3 | **% del año ACTIVO** (LF ≥ κ_LF = 0,35) | 64,11 | 64,77 | **0,465** | 0,956 |
| 4 | % del año VIABLE | 61,83 | 63,29 | 0,412 | 0,509 |
| 5 | **% del año JARDÍN FÉRTIL = (b)** | 38,19 | 38,63 | **0,454** | 0,957 |

**El salto letal está entre el renglón 2 y el 3.** La señal sobrevive intacta
hasta el último invariante continuo (LF media, AUC 0,654; LF p90, AUC 0,777) y
muere exactamente en el momento en que se convierte en un **sí/no** y se cuenta
"cuánto del año". Pasa de 0,654 a 0,465: de "algo mejor que una moneda" a "peor
que una moneda".

### 2.3 Por qué muere ahí: la compuerta está abierta casi siempre

`clasificarCierre()` (`sim-cosmoclima.html:850-882`) declara ACTIVO cuando
`LF ≥ κ_LF`, con `KAPPA_LF_INFIMO = 0.35` (línea 811). Y `LF` es, en el modo B
adoptado (línea 1675-1677, 1707):

```js
const senalLF = 4*state.floracion*(1-state.floracion);
state.LF = clamp(senalLF, 0, 1);
```

Despejando: **LF ≥ 0,35 equivale a floración ≥ 0,0969.**

Ahora la parte que importa. La floración que el motor produce en 62 años:

| estadístico de `floracion` | valor |
|---|---|
| media | 0,1471 |
| mediana | 0,1317 |
| p90 | 0,2737 |
| p99 | 0,3949 |
| **máximo en 62 años** | **0,5092** |
| tope duro del código (`clamp(...,0,0.9)`, línea 1605) | 0,90 |

O sea: **la compuerta se abre en 0,0969, que es el 19 % de la floración máxima
que el instrumento llegó a producir jamás.** Resultado medido: **el 66,6 % de
todos los ticks de los 62 años están por encima del umbral**, y el 36,5 % de los
ticks "activos" tienen una floración residual (f < 0,15), no una floración.

Y como la floración se apaga despacio a propósito (166 días, la duración real de
un evento según Chávez et al. 2019), una sola lluvia deja la compuerta abierta
medio año largo:

- decaimiento teórico de f = 0,30 hasta f = 0,0969: **188 días**
- medido en la corrida: **49 rachas continuas de "ACTIVO" en 62 años, mediana de
  260 días cada una** (con κ_LF = 0,65 serían 46 rachas de 112 días)

**La analogía.** Es como querer distinguir un año de lluvias fuertes de un año
seco usando un sensor de humedad que se enciende con la primera gota y tarda
nueve meses en apagarse. Da igual si cayeron 250 mm o 25: el sensor pasa
encendido más o menos el mismo tiempo. Y como sólo tenemos 12 meses por año, dos
años distintos terminan con la misma cuenta. Eso es exactamente lo que dice el
renglón 3 de la tabla: 64,11 % contra 64,77 %.

Dicho en una frase: **`JARDIN_FERTIL_pct` no mide qué tan grande fue la
floración; mide cuántos meses quedó humedad residual.** El desierto florido es un
evento de *magnitud* (un pulso), no de *permanencia*.

### 2.4 El cómplice: la viabilidad sigue ciega

El otro eje del AND tampoco ayuda. `A_sys_env` en este mismo test:

- floración 0,7260 vs control 0,7275 → **diferencia −0,0015, AUC 0,477**
- barriendo κ_V de 0,50 a 0,85, el AUC de (b) nunca pasa de 0,569

Es la misma ceguera diagnosticada el 11-ago (ρ = −0,055 contra lluvia real), que
sigue en pie. **Pero no es la causa principal de (b):** el renglón 3 muestra que
la señal ya estaba destruida en la activación, antes del AND. La viabilidad
diluye; no mata.

---

## 3. ¿Es un quinto caso de los cuatro patrones ya diagnosticados?

| # | Patrón ya conocido | ¿(b) lo repite? | Evidencia medida |
|---|---|---|---|
| (i) | **LF medía distancia a un slider** | **No** — pero hay una trampa dormida | LF ya no es `\|powerLive−powerBase\|`. Sin embargo `4f(1−f)` es una **U invertida**: crece hasta f=0,5 y después **baja**. Hoy no muerde porque el motor nunca pasa de f=0,5092: sólo **608 ticks de 1.357.800 (0,045 %)** caen en la rama descendente. Si el modelo alguna vez produjera floraciones grandes, el instrumento volvería a premiar la floración mediana por sobre la máxima. Es la misma forma que ya nos mordió una vez, hoy inofensiva sólo por suerte de rango. |
| (ii) | **Variable ciega que devuelve ruido (A_sys_env)** | **Sí, confirmado — pero no es nuevo ni es la causa** | AUC 0,477; diferencia −0,0015. Sigue exactamente como el 11-ago. Cómplice, no autor. |
| (iii) | **Cota mirada de un solo lado (e_R)** | **Parcialmente, y sin consecuencia numérica** | `LF ≥ κ_LF` es una sola cota. Como LF es una U, esa única cota define en realidad una **banda en floración: f ∈ [0,0969 , 0,9031]**, así que la forma canónica de dos extremos ("cierre total suena muerto, apertura total colapsa") sí está representada. El borde de arriba nunca se alcanza (f_max = 0,509), así que arreglarlo no cambiaría ni un número. |
| (iv) | **Constante en el plano equivocado (κ_Δ)** | **SÍ — éste es el quinto caso** | κ_Δ fue "inventado como percentil". κ_LF = 0,35 **no** fue inventado: viene del canon (Bloque 28, E1 "Procesador de Audio EIT-3", zona fértil en LF = 0,35), y así está documentado en `sim-cosmoclima.html:803-811`. El problema es el otro lado de la misma moneda: **el número es correcto en aquel instrumento y nunca se volvió a verificar el plano en éste.** Allá LF era el eje directo, de 0 a 1. Acá LF es `4f(1−f)`, una transformación que comprime el rango real de floración (0 → 0,509) contra todo el rango 0 → 1 de LF. El mismo 0,35 aterriza en **f = 0,0969**, y en su propia distribución queda alrededor del **percentil 33** (p25 = 0,290, p50 = 0,457): una compuerta abierta dos tercios del tiempo. Es el **error de plano O-N16.2d otra vez**, en versión espejo. |

**Conclusión del punto 3:** sí, hay un quinto caso, y es de la familia (iv).

---

## 4. Diagnóstico con número

**No es límite del dato. No es límite del modelo. Es un bug de medición: el
criterio mide una cosa distinta de la que dice medir.**

### Evidencia a favor de "bug"

1. **La señal entra y no sale.** AUC 0,831 en la lluvia de entrada → 0,785 en la
   floración máxima del modelo → 0,654 en LF media → **0,454 en el número que se
   publica**. Un límite de dato no puede explicar una pérdida que ocurre *dentro*
   del instrumento, después de que la señal ya entró.
2. **La información sigue ahí, y es abundante.** Si se recuenta el mismo año con
   la compuerta puesta en otro lugar del *mismo* invariante, sin tocar nada más
   de la física:

   | κ_LF | equivale a f ≥ | % ticks activos | **(b)** | p | AUC |
   |---|---|---|---|---|---|
   | 0,20 | 0,053 | 87,1 % | −1,49 pp | 0,797 | 0,462 |
   | **0,35 (publicado)** | **0,097** | **66,6 %** | **−0,44 pp** | **0,957** | **0,454** |
   | 0,50 | 0,146 | 43,6 % | +7,05 pp | 0,232 | 0,662 |
   | 0,65 | 0,204 | 24,4 % | **+15,89 pp** | **0,0032** | **0,858** |
   | 0,80 | 0,276 | 9,7 % | +9,31 pp | 0,025 | 0,746 |
   | 0,90 | 0,342 | 3,3 % | +4,28 pp | 0,104 | 0,654 |

   No es un punto de suerte: es una **meseta ancha** (todo el tramo 0,50-0,90 da
   diferencia positiva) y aguanta el sub-análisis que saca la megasequía de los
   dos grupos (κ=0,65: **+17,86 pp, p = 0,0020**, n = 10 vs 8) y el
   dejando-uno-fuera (**+14,95 a +17,67**, nunca cambia de signo).

3. **El límite estadístico existe, pero es secundario.** Tal como está escrito,
   (b) compara dos promedios con desviación combinada de **18,05 pp**, así que
   con n = 13 vs 10 el **efecto mínimo detectable** (α = 0,05, potencia 80 %) es
   de **21,28 pp** — un tercio del rango entero de la serie (0,00 a 67,40). Con
   esa vara, la diferencia observada de 0,44 pp no es "casi": es indistinguible
   de cero (p = 0,957). **Pero** el barrido muestra que un instrumento bien
   escalado produce +15,9 pp, que **sí** es detectable con esos mismos 23 años.
   O sea: el n no es el cuello de botella; el escalado sí.

4. **El veredicto actual cuelga de un solo año, y encima de uno que no es de
   floración.** Los tres valores publicados de (b) —ERA5 −0,06, cruda −0,45,
   ajustada +0,13— **no vienen de los años de floración**: los 13 años de
   floración dan `JARDIN_FERTIL_pct` **idéntico en las tres series** (media
   38,1854 en las tres, hasta el cuarto decimal). La única diferencia está en el
   grupo control, y en la práctica en **un año: 2020** (24,93 / 21,10 / 19,18 %).
   Ese único año, dividido por los 10 controles, explica **exactamente** los tres
   números:
   - 38,631 − 24,93/10 + 21,10/10 = 38,248 ✔ (ERA5)
   - 38,631 − 24,93/10 + 19,18/10 = 38,056 ✔ (ajustada)

   Y el dejando-uno-fuera del criterio publicado va de **−4,52 pp (sin 2013)** a
   **+2,75 pp (sin 2018)**: cualquiera de los 23 años puede dar vuelta el signo.
   **Un veredicto que se decide con una desigualdad estricta sobre una
   diferencia de 0,45 pp, cuando quitar un año la mueve 3-4 pp, no es un
   veredicto: es ruido leído con lupa.**

### Lo que el diagnóstico NO dice

- La explicación de la página ("años en conflicto" + "parches localizados a
  decenas de kilómetros") **no es falsa, pero no es lo que domina**. Sí hay dos
  conflictos genuinos del ground truth: **2021** (documentado como floración con
  33,5 mm en el punto y `JARDIN_FERTIL_pct` = 0,00) y **2008** (control con
  184,2 mm). Pero con la lluvia separando los grupos a AUC 0,831, esos casos
  sueltos no alcanzan a explicar una caída hasta AUC 0,454.
- Tampoco es límite del modelo: la física **sí** produce la diferencia. La
  floración máxima del año separa los grupos con AUC 0,785, p = 0,044. Lo que
  falla es la regla que la lee.

---

## 5. Qué pasaría si se corrigiera — cálculo, NO aplicado

Cuatro lecturas alternativas del **mismo invariante**, sin tocar la física, todas
recalculadas sobre la corrida verificada. **Ninguna está aplicada ni recomendada
como adoptada.**

| Variante | (b) | p | AUC | (a) tuplas | (c) megasequía vs global | (d) ρ vs lluvia independiente | (e) Cierre megasequía vs global |
|---|---|---|---|---|---|---|---|
| **PUBLICADO** `4f(1−f) ≥ 0,35` (f ≥ 0,0969) | −0,44 | 0,957 | 0,454 | **61/62** | 23,13 vs 40,91 ✓ | **+0,217** ✓ | 42,90 vs 22,30 ✓ |
| **A** leer el punto fértil sobre la magnitud: `f ≥ 0,35` | +3,60 | 0,188 | 0,654 | 34/62 ✗ | 0,00 vs 1,73 (degenerado) | +0,528 | 66,03 vs 61,48 |
| **B** LF monótona `f/0,9`, umbral 0,35 (f ≥ 0,315) | +5,30 | 0,060 | 0,692 | 36/62 ✗ | 0,00 vs 2,78 (degenerado) | +0,628 | 66,03 vs 60,42 |
| **C** LF monótona `f/f_max_real`, umbral canónico 0,35 | **+14,31** | **0,0080** | 0,838 | 58/62 | 8,54 vs 16,80 ✓ | **+0,760** | 57,49 vs 46,40 ✓ |
| **D** `4f(1−f) ≥ 0,65` (f ≥ 0,204) — **hallado por barrido** | **+15,89** | **0,0032** | 0,858 | 56/62 | 6,58 vs 13,40 ✓ | **+0,805** | 59,45 vs 49,81 ✓ |

Lecturas:

- **A y B se descartan solos.** Aplicar el 0,35 directamente sobre la floración
  deja el Jardín Fértil en ~2 % del tiempo y rompe el criterio (a) —el más fuerte
  del experimento, el de las trayectorias propias— de 61/62 a 34-36/62.
- **C y D dan lo mismo por caminos distintos** y mejoran los cinco criterios a la
  vez, incluido ρ contra una fuente **totalmente externa** (la lluvia regional del
  sqlite, 61 años, 130-270 estaciones, que no toca el motor): de **+0,217 a
  +0,76 / +0,805**.
- **Advertencias honestas, las tres:**
  1. **D salió de un barrido.** Adoptar un umbral porque el barrido lo premia es
     exactamente el error (iv) otra vez, con número nuevo. **No lo propongo como
     adopción.** Lo muestro sólo como prueba de que la información existe.
  2. **C usa `f_max_real`, que es un dato de la propia muestra.** Es más
     defendible que D (no mueve el número canónico 0,35, sólo devuelve la
     variable a su plano), pero sigue teniendo un pie dentro de la muestra. Una
     versión limpia usaría el máximo *alcanzable* por la curva empírica
     (`objetivoFloracionEmpirico` + las tasas de 90/166 días), no el observado.
     **Ese cálculo queda pendiente; no lo hice.**
  3. Con umbrales altos `JARDIN_FERTIL_pct` se parece cada vez más a un
     termómetro de lluvia (ρ = 0,805). Hay que vigilar que el instrumento no
     pierda justo lo que lo hace valioso —distinguir *detenido* de *muerto*—.
     Medido: con κ = 0,65 el Cierre en megasequía sigue dominando (59,45 vs
     49,81), así que esa distinción **sobrevive**; pero es lo primero que habría
     que revisar en cualquier propuesta.

---

## 6. Recomendación (sin aplicar, para decisión de Alexis)

1. **Cambiar la frase de la página**, en algún momento y con tu visto bueno.
   Hoy dice "es un límite del dato de contraste, no del modelo". Lo medido dice
   lo contrario: el dato de contraste separa los grupos a AUC 0,831 y la pérdida
   ocurre adentro del instrumento. Lo honesto sería: *"criterio no resuelto — la
   señal existe en el dato y en la física, pero la regla de conteo tal como está
   calibrada no la puede leer"*.
2. **Corregir la fila inconsistente de la tabla publicada** (38,19 vs 38,25 es de
   la variante ERA5; la adoptada da 38,19 vs 38,63). Es un dato, no una
   interpretación.
3. **Revisar κ_LF = 0,35 como problema de plano, no de valor.** El número es
   canónico y no hay por qué tocarlo; lo que hay que revisar es sobre qué
   variable se aplica. La pregunta para la Teoría es: *¿el 0,35 del E1 de audio
   se aplica al eje LF normalizado del instrumento, o a la magnitud física?* Si
   la respuesta es "al eje normalizado", entonces `4f(1−f)` con f llegando sólo a
   0,509 no es el eje normalizado de este instrumento, y ahí está el arreglo.
4. **Cambiar el estadístico de (b), aparte de todo lo anterior.** Una desigualdad
   estricta entre dos promedios, sin test, con un mínimo detectable de 21 pp, no
   puede aprobar ni reprobar nada. Sugerencia: reportar **AUC + p de
   permutación**, que es la misma pregunta ("¿un año de floración tiende a dar
   más Jardín Fértil que un control?") pero con la incertidumbre a la vista.
5. **A_sys_env sigue ciega** (AUC 0,477 acá). Es un pendiente abierto desde el
   11-ago, independiente de (b).
6. **No cerrar nada.** Esto es un hallazgo, no un veredicto.

---

## Anexo · Detalle año por año (variante *cruda*, la adoptada)

| año | grupo | lluvia motor (mm) | pico mes | ACTIVO % | VIABLE % | **JF %** |
|---|---|---|---|---|---|---|
| 1997 | FLO | 537,1 | 252,3 | 53,97 | 70,96 | 32,60 |
| 2002 | FLO | 466,9 | 178,0 | 98,08 | 59,45 | 57,53 |
| 2017 | FLO | 404,4 | 257,0 | 78,91 | 59,73 | 40,55 |
| 2000 | FLO | 325,0 | 159,0 | 84,66 | 51,78 | 46,03 |
| 2024 | FLO | 279,1 | 138,3 | 57,80 | 65,48 | 30,95 |
| 1983 | FLO | 243,5 | 127,0 | 100,00 | 53,97 | 53,97 |
| 1991 | FLO | 237,0 | 125,5 | 61,65 | 65,48 | 36,71 |
| 2015 | FLO | 232,1 | 115,0 | 52,05 | 67,12 | 40,27 |
| 2011 | FLO | 214,7 | 143,0 | 78,91 | 59,46 | 47,95 |
| 2016 | ctl | 205,6 | 79,8 | 100,00 | 59,72 | 59,72 |
| 2005 | FLO | 188,1 | 54,4 | 80,82 | 59,72 | 42,46 |
| 2008 | ctl | 184,2 | 58,2 | 59,72 | 55,62 | 28,77 |
| 1989 | ctl | 167,5 | 82,5 | 73,15 | 59,72 | 40,54 |
| 2022 | FLO | 146,1 | 120,2 | 42,46 | 67,39 | 30,95 |
| 2020 | ctl | 129,8 | 111,6 | 46,03 | 67,40 | 24,93 |
| 1996 | ctl | 129,5 | 59,0 | 80,56 | 67,40 | 47,95 |
| 2018 | ctl | 103,7 | 65,2 | 92,33 | 67,40 | **67,40** |
| 2003 | ctl | 101,4 | 57,0 | 99,73 | 57,80 | 57,53 |
| 1990 | ctl | 93,5 | 52,5 | 63,56 | 61,38 | 49,87 |
| 2013 | ctl | 73,7 | 53,2 | 24,93 | 69,05 | 1,92 |
| 2012 | FLO | 70,1 | 43,6 | 44,12 | 61,64 | 36,44 |
| 2021 | FLO | 33,5 | 16,5 | 0,00 | 61,64 | **0,00** |
| 2019 | ctl | 15,9 | 11,4 | 7,68 | 67,40 | 7,68 |

Mirando la columna ACTIVO se ve el problema a ojo desnudo: **1997, el año más
lluvioso de los 62 (537 mm), queda en 53,97 % de activación; 2003 y 2016
—controles con 101 y 206 mm— llegan a 99,73 % y 100 %.** La compuerta no ordena
los años por magnitud de floración; los ordena por cuántos meses hubo humedad
residual, que es otra cosa.

---

### Reproducibilidad

Scripts de este diagnóstico (temporales, fuera del proyecto):
`…/scratchpad/diag_b.js` (re-corrida de la física con volcado de invariantes tick
a tick), `analisis_b.py`, `analisis_b2.py`, `analisis_ticks.py`,
`analisis_barrido.py`, `analisis_robustez.py`.
Los p-valores son de permutación (20.000 remuestreos, semilla fija); el AUC es
Mann-Whitney con empates a 0,5. Ningún archivo del proyecto fue modificado.
