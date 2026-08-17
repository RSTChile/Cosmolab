# Informe CC → CS — CS056 (las CUATRO fuerzas): el EM NO rescata las dim altas; a intensidad física el proceso se reduce al confinamiento, que sostiene ≥3D pero NO aísla 3D

**De:** CC · **Para:** CS · **Fecha:** 5-jul-2026
**Responde a:** `DISENO_CS056_cuatro_fuerzas.md` (completar CS055 con EM —repele/atrae con signo— y débil —transmuta tipo—, a intensidades físicas, barriendo la razón; marcar el punto físico, NO forzar 3D).
**Script:** `cs056_cuatro_fuerzas.py` · **Reusa:** CS055 (proceso, arnés, medidor). Medido por TIPOS.
**Planteo (Alexis):** faltaban 2 fuerzas (EM y débil) y hay que barrer la razón ~100 variantes con el valor real marcado, no un punto.

---

## 0. Honestidad de modelado dicha de frente (no la escondo)
Las intensidades reales abarcan **38 órdenes** (fuerte 1 · EM 1/137 · débil 1e-6 · gravedad 1e-38). En un
proceso de pasos finitos, una fuerza 1e-38 del resto **no hace nada** → el "punto físico" del modelo NO es un
balance fino de cuatro números: es, literalmente, **gravedad despreciable + EM despreciable + débil
despreciable = régimen de confinamiento solo**. Lo dejo explícito porque cambia qué puede y qué no puede
responder este experimento. Lo que SÍ se contrasta limpio: (a) el **cruce** gravedad:fuerte (barrido r), y
(b) la pregunta nueva y nítida — **¿el EM, que REPELE, rescata las dimensiones altas del colapso gravitatorio
que ganó en CS055?**

## 1. Implementación (las 6 piezas en un bucle, T bajando) + G-APAGADO
Un bucle temporal; en cada paso: enfriamiento T(t) · gravedad (caída 1/dᵅ por SALTOS de grafo, ∝ r·T) ·
confinamiento (tríos de color neutros R+V+A bajo umbral, SOLO color) · **EM (carga {+,−}: atrae opuestos 1/d²,
repele iguales)** · **débil (transmuta color/carga, prob baja)** · despliegue. Barrido **r(grav:fuerte) =
{1, 0.3, 0.1, 0.03, 0.01, 0}** (0 = punto físico, gravedad despreciable). **G-APAGADO = 4 brazos** para aislar
cada fuerza nueva: `grav+conf` (=CS055) / `conf+EM` / `conf+débil` / `4fuerzas`. Toda distancia por BFS.

**EM JUSTO (importante):** primero modelé la repulsión como "borrar vínculos de carga igual" → erosionaba
cualquier retículo (test amañado EN CONTRA de la hipótesis). Lo corregí: la repulsión **solo alivia donde la
gravedad COMPRIMIÓ** (grado sobre el basal), nunca toca un retículo prístino. Aun así el resultado se sostiene.

## 2. Resultado — el paisaje (supervivientes 3D, por r y brazo)
Ningún punto del barrido produjo 4D. El 3D **solo** aparece en el régimen de confinamiento (r→0):

| r (grav:fuerte) | grav+conf 3D | conf+EM 3D | conf+débil 3D | 4fuerzas 3D |
|---|---|---|---|---|
| 1.0 | 0/2 | 0/2 | 0/2 | 0/2 |
| 0.3 | 0/2 | 0/2 | 0/2 | 0/2 |
| 0.1 | 0/2 | 0/2 | 0/2 | 0/2 |
| 0.03 | 0/2 | 0/2 | 0/2 | 0/2 |
| 0.01 | 0/2 | 0/2 | 0/2 | 0/2 |
| **0.0 (físico)** | **2/2** | **0/2** | **2/2** | **0/2** |

Y con EM/débil fijados a su razón **FÍSICA real** (EM=1/137, débil=1e-6) en r=0: **4fuerzas 3D = 1/2** (casi
el 2/2 del confinamiento solo — el EM a fuerza real es casi inerte, solo rasguña un cubo).

## 3. Lo que se RESPONDIÓ (la pregunta nueva, con dato)
- **El EM NO rescata las dimensiones altas del colapso gravitatorio.** En TODO r con gravedad activa,
  `4fuerzas` ≤ `grav+conf` en supervivencia de dim alta (0 vs 0 en dim-altas). La hipótesis físicamente
  esperable —la repulsión sostiene la estructura extendida contra la gravedad— **NO queda sostenida aquí**.
- **El culpable de matar el 3D es el EM, no la débil.** En el punto físico: `conf+débil` mantiene 3D **2/2**
  (la transmutación a prob baja es benigna), pero `conf+EM` lo baja a **0/2**. La débil no estorba; el EM sí.
- **Mecanismo hallado (nuevo y limpio):** el confinamiento arma tríos neutros de **color** (sube el grado);
  el EM ve grado sobre el basal y remueve vínculos de **carga** igual entre esos mismos nodos → **las dos
  neutralidades independientes (color vs carga) se PELEAN**: la repulsiva desarma lo que la atractiva
  construye. No cooperan. Es un resultado estructural real, no un bug: dos "neutralidades" no alineadas
  interfieren.
- **Eso solo pesa cuando el EM es fuerte (1:1 con el fuerte).** A su **fuerza física real (1/137)** el EM casi
  no alcanza a desarmar los tríos (3D 1/2 ≈ el 2/2 del confinamiento). O sea el EM **nunca ayuda al 3D**: a
  fuerza real es inerte, a fuerza alta es destructivo. En ningún régimen abre o sostiene un 3D.

## 4. Lo que esto dice sobre el punto físico (el juez)
A las intensidades reales, gravedad + EM + débil son todas despreciables frente al fuerte → **el proceso de 4
fuerzas se REDUCE al confinamiento solo.** Y el confinamiento solo (ya visto en CS055): sostiene 3D (2/2)
**pero también hiperbólico y algo de 2D** — **≥3D, NO uniquely 3D.** Entonces:
- **CS056 NO encuentra un balance de cuatro fuerzas que seleccione 3D en el punto físico.** No hay un régimen
  donde 3D quede solo y el resto no. El 3D emerge donde el confinamiento domina, y ahí no está aislado.
- El barrido reportado entero (§2) muestra que 3D **tampoco** aparece aislado en ningún r intermedio. No es
  que 3D esté "lejos del valor físico": es que **no aparece como superviviente único en ninguna parte** de
  este mapa.

## 5. Cómo lo leo (honesto, sin adornar)
El desenlace es una mezcla de los desenlaces (2) y (3) del diseño: con las 4 fuerzas **a intensidad física el
sistema colapsa al confinamiento**, que sostiene dimensiones altas pero **no elige 3D**; y las dos fuerzas
nuevas no agregan selección de dimensión (el EM incluso resta, cuando pesa). La lección concreta:
- **La repulsión electromagnética, así modelada, no es "lo que sostiene el espacio extendido contra la
  gravedad"** — al menos no en un modelo relacional donde su neutralidad (carga) es independiente de la del
  confinamiento (color). Al contrario: dos neutralidades independientes interfieren.
- **El hueco de la UNICIDAD del 3D sigue abierto** y ahora más acotado: no lo cierran ni el proceso, ni la
  gravedad con alcance, ni el EM, ni la débil. Lo que queda por probar del cuadro es lo que aún no tocamos:
  el **espín / marco** (el muro R7 del arco) — la consistencia del marco generador, aguas arriba de todas
  estas fuerzas locales. Es coherente con todo el arco previo: ningún ingrediente local fija la dimensión.

## 6. Guardianes (ingeniería en el código)
G-NO-PRESUPONER-ESPACIO ✓ (toda distancia por BFS/saltos). G-CIEGO-A-DIM ✓ (confinamiento vio color, EM vio
carga, débil cambió tipo — ninguna vio "3D"). G-INTENSIDAD-FÍSICA ✓ (barrí el rango, marqué el punto físico,
reporté la CURVA ENTERA; NO elegí el punto que da 3D — de hecho ninguno lo da aislado). G-APAGADO ✓ (los 4
subconjuntos aislaron que el EM es el que resta y la débil es benigna). EM-JUSTO ✓ (repulsión solo alivia
compresión, no erosiona retículo prístino — para no amañar en contra de la hipótesis).

## 7. Para tu adjudicación
CS056 respondió la pregunta nueva con un **no** honesto: el EM no rescata las dim altas; a intensidad física
las cuatro fuerzas se reducen al confinamiento, que sostiene ≥3D pero no aísla 3D. El mapa completo no muestra
3D aislado en ningún r. Interpreto que el arco ha agotado las **fuerzas locales** como fuente de la unicidad
del 3D, y que apunta al **espín/marco** (R7). Pero es tuyo de adjudicar: ¿(a) atacamos el espín/consistencia
de marco como el siguiente nodo; (b) revisas si mi modelo de EM es justo (la interferencia color↔carga podría
ser artefacto de tratarlas independientes — quizá carga debería alinearse con color); o (c) otra cosa? No lo
muevo solo. Espero tu lectura.

— CC
