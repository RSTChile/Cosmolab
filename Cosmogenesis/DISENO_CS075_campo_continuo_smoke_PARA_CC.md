# DISEÑO CS075 — Campo continuo estigmérgico: cinco problemas del protocolo y su arreglo

**Diseña:** Claude Science · **Director:** Alexis López Tapia · 29-jul-2026
**Base:** PROTOCOLO EXPERIMENTAL CS075 (versión canónica del director) + arquitectura real
verificada en `VSTCosmo/Célula_Madre/campo/VST_Celula_Madre_001.py` (clase `Hemisferio`, l.520).
**Estado:** diseño para pasar a CC. **Nada corrido. Nada cerrado.**

---

## 0. Qué se verificó antes de diseñar

- La arquitectura existe: clase `Hemisferio` (l.520) con `Phi` (campo), `Phi_vel`, `W`
  (pesos plásticos), `Phi_int_historia` (atractor de historia), laplaciano local, reacción
  no lineal `Phi*(1-Phi²)`, forzamiento de borde en los nodos extremos, y todas las
  constantes `CAMPO_*` (l.460-500).
- **CORRECCIÓN (29-jul, el director aportó los archivos): esta afirmación mía era FALSA.**
  Escribí que "los dos archivos que cita el protocolo no existen con ese nombre". Los dos
  existen:
  - `VST_Celula_Madre_001.py` **existe con exactamente ese nombre** en
    `VSTCosmo/Célula_Madre/campo/`, y **yo mismo lo había leído** — cité su clase `Hemisferio`
    (l.520) y le copié las constantes `CAMPO_*`. Afirmé que no existía habiéndolo leído.
  - `Célula_Madre_Funcional_001.py` existe y el director lo aportó. Mi búsqueda no lo encontró
    bajo `Cosmolab/`, y de "no lo encuentro donde busqué" concluí "no existe" — que es
    justamente lo que el pacto prohíbe: no dictaminar falso sin refutar.
- **El campo real es 1D de 32 nodos**, no 3D. Esto no es un detalle: portarlo a cosmología
  es un cambio de dimensión, no un cambio de parámetros (ver §2, problema 4).

---

## 1. Los cinco problemas del diseño, y el arreglo de cada uno

### Problema 1 — El NULL propuesto no puede decidir nada (el más grave)

El protocolo §3 propone como control: *"las mismas ecuaciones de campo actualizadas mediante
un pipeline secuencial por turnos"*.

**Por qué falla.** "Simultáneo" y "por turnos" no son dos físicas distintas: son **dos
esquemas de integración numérica distintos** del mismo sistema de ecuaciones. La diferencia
entre ellos es error de truncamiento del integrador, no un efecto del universo. Un revisor
diría, con razón, que el experimento mide su propio solver.

Lo medí sobre un campo de juguete con la misma forma que el de ANIMA (laplaciano + reacción
`Phi(1-Phi²)` + forzamiento de borde): la divergencia entre simultáneo y por turnos es
grande (norma 1,21; el 15,6 % de los nodos termina en atractor de signo opuesto) y **no
disminuye al reducir dt** de 0,02 a 0,00125. Es decir: es persistente, pero eso no la vuelve
física — el campo es no lineal con dos atractores (±1), así que cualquier diferencia
minúscula de trayectoria se amplifica hasta cambiar de atractor. Un integrador distinto
basta para eso.

**Corrección de método:** en la primera pasada escribí que la diferencia "cae con dt" cuando
la tabla mostraba que se queda en 1,2. Los datos decían lo contrario de lo que afirmé. Queda
asentado.

**El arreglo.** El NULL tiene que destruir **el mecanismo que la hipótesis dice que produce
el efecto**, no el orden de las cuentas. La hipótesis del protocolo §2.D es que la inercia
emerge del repertorio de pesos plásticos `W` y del atractor `Phi_int_historia`. Entonces:

| brazo | qué es | qué se espera |
|---|---|---|
| **REAL** | campo con plasticidad viva (`campo_W=True`, `campo_atractor=True`) | grumos con inercia |
| **NULL-A (memoria congelada)** | `W` se calcula igual pero **se congela** en su valor inicial: el término plástico sigue actuando, con la misma magnitud, sin aprender nada | si la inercia viene de la *historia*, desaparece |
| **NULL-B (memoria barajada)** | `W` evoluciona, pero en cada paso sus filas se permutan al azar: misma energía, misma norma, correlación con el campo destruida | control de que no sea solo "magnitud del término" |

Los tres brazos comparten integrador, dt, semilla de inicialización y forzamiento. **La
única diferencia es la memoria.** Eso sí es una pregunta sobre el modelo.

### Problema 2 — No hay observable numérico ni umbral pre-inscrito

El protocolo §3 pide medir *"si el modelo logra auto-organizar grumos estables que
desarrollen resistencia a la expansión (inercia emergente) de manera autónoma"*. No hay
número, no hay umbral, no hay criterio de decisión.

Ese es exactamente lo que hundió a cs074D: el umbral existía (z>2) pero el instrumento no
podía alcanzarlo, y no se supo hasta gastar 61 horas.

**El arreglo — inercia medida como coeficiente de arrastre, no como atributo.** Un grumo con
inercia es uno que **se retrasa** respecto del forzamiento de expansión. Se mide así:

1. **Identificar grumos** sin distancias N×N (respetando §2.B): regiones conexas donde
   `|Phi| > 0.5`, por componentes conexas de la malla (vecinos inmediatos). Es local, es el
   mismo criterio que un FoF pero sobre celdas de campo.
2. **Medir el retraso.** El forzamiento de expansión impone una dilatación conocida
   `a(t)`. Para cada grumo se registra su tamaño característico `L_grumo(t)`. Un grumo sin
   inercia sigue la expansión: `L_grumo ∝ a(t)`. Un grumo con inercia se queda atrás.
   **Observable:** `beta = d(ln L_grumo) / d(ln a)`, ajustado sobre la segunda mitad de la
   corrida. `beta = 1` es arrastre total (sin inercia); `beta = 0` es resistencia total.
3. **Estadístico:** `z = (beta_REAL − beta_NULL) / desviación_conjunta` sobre las semillas.

**Umbral pre-inscrito: z > 2**, igual que cs074D. Pero — y esto es la lección de las 61
horas — **con un calibrador obligatorio previo** (§3, S1) que verifique que el NULL puede
alcanzar ese umbral. Si no lo alcanza, no se corre el experimento.

### Problema 3 — El observable "estabilidad de grumos" puede saturar

Si casi todos los grumos sobreviven, `beta` no discrimina, y el problema es invisible hasta
el final. Se registran por eso **dos observables secundarios en la misma corrida** (cuestan
casi nada y evitan otro cs074D): número de grumos al final, y fracción del campo con
`|Phi|>0.5`. Si `beta` satura, hay con qué reanalizar sin volver a correr — el error que
cometí al afirmar que el Camino B se podía recalcular con el JSON de cs074D, que no guardaba
el observable del brazo NULL.

### Problema 4 — 1D a 3D no es un port, es un rediseño

El campo de ANIMA es 32 nodos en línea, con forzamiento en los dos extremos (`forzamiento[0]`
y `forzamiento[-1]`, un dipolo). En cosmología no hay "extremos": el forzamiento de expansión
es **volumétrico y sin eje**.

**El arreglo:** malla 3D periódica (sin bordes), laplaciano de 7 puntos, y la expansión
entrando como término homogéneo de dilución `−3·H(t)·Phi` más el escalado de la métrica de la
malla — no como forzamiento en celdas de borde. Se mantiene la prohibición de N×N: cada celda
solo lee sus 6 vecinos.

**Malla del smoke: 16³ = 4096 celdas.** Suficiente para que quepan varios grumos y barato
(~0,8-2 s por corrida). La producción, si llega, sube a 32³.

### Problema 5 — Contradicción interna del protocolo

§2.A prohíbe el bucle secuencial de fuerzas como algo estructuralmente equivocado. §3 lo usa
como brazo de control. Si el secuencial es el NULL, el experimento está probando *"mi
integrador es mejor que el otro"* — una afirmación sobre software, no sobre el universo.

**El arreglo:** la actualización simultánea se declara **decisión de implementación**
(justificada, sostenida, no negociable), no hipótesis a testear. La hipótesis a testear es la
del §2.D: **que la inercia emerge de la memoria del campo.** Eso es falsable y es sobre
física del modelo.

---

## 2. Los tres smoke tests — 1,2 horas en total

Presupuesto calculado sobre malla 16³, T=20, dt=0,01 (~0,8-2 s por corrida; se toma 12 s por
corrida como margen conservador de 6-15×).

### S1 — CALIBRADOR DEL NULL (**bloqueante**: si falla, no se sigue)

**4 configuraciones × 6 semillas × 3 brazos = 72 corridas ≈ 0,24 h**

Se eligen 4 configuraciones donde el campo **sí** forma grumos (verificado a ojo en la primera
corrida, es un smoke). Se corren los tres brazos: REAL, NULL-A, NULL-B.

**Criterio de paso:** al menos una configuración con **z > 2** entre REAL y NULL-A.

- **Si pasa:** el NULL tiene fuerza. Seguir a S2.
- **Si no pasa:** **parar y reportar.** El NULL no puede detectar lo que se le pide. No se
  corre S2 ni S3, y no se gastan 61 horas para descubrirlo. Se rediseña el control.

Esta es la pieza que cs074D no tuvo, y es la razón por la que este diseño la pone primera y
bloqueante.

### S2 — INERCIA EMERGENTE (el experimento propiamente dicho, en chico)

**8 configuraciones × 6 semillas × 2 brazos = 96 corridas ≈ 0,32 h**

Rangos anclados en lo ya medido, **no inventados**:

| variable | rango | de dónde sale |
|---|---|---|
| amplitud inicial de perturbación | 0,05 – 3,8 (log) | los tres regímenes de cs074A: meseta <0,5, fragmentación 0,9-2,3, colapso >3,8 |
| tasa de expansión | 0,0030 – 0,05 (log) | **arranca 15 % sobre el piso 0,0026** medido en cs074D FULL |
| tasa de aprendizaje del campo | 0,002 – 0,2 (log) | barre ×10 en torno a `CAMPO_ETA_HEBB = 0,02` de ANIMA |

Arrancar la expansión sobre el piso recupera el **17,6 %** del cómputo que en cs074D se fue
en configuraciones que no formaban ni un átomo.

**Lectura pre-inscrita, tres salidas, cualquiera válida:**
1. **INERCIA EMERGE:** hay configuraciones con `z > 2` y `beta_REAL < beta_NULL` (el campo con
   memoria resiste más la expansión). Se describe en qué región.
2. **NO EMERGE:** ninguna configuración con `z > 2`, **y el calibrador S1 pasó** (esto es lo
   que le faltó a cs074D para poder afirmar algo). Entonces sí se puede decir: en este
   modelo, la memoria del campo no produce inercia.
3. **SATURADO:** más del 50 % con `z > 2` → el efecto es trivial, hay que endurecer el NULL.

### S3 — BARRIDO MÍNIMO 2D (solo si S2 da la lectura 1)

**24 configuraciones × 4 semillas × 2 brazos = 192 corridas ≈ 0,64 h**

Malla 6×4 en (amplitud inicial × tasa de aprendizaje), expansión fija en el mejor valor de S2.
Contesta si los aciertos forman **región conexa** o están dispersos — la pregunta de banda
estrecha que cs074D no pudo contestar. Método de conectividad: el mismo del protocolo cs074D
§5 (vecinos-hit contra etiquetas barajadas, 1000 permutaciones), que estaba bien diseñado; lo
que falló ahí fue el NULL, no esta métrica.

---

## 3. Lo que CC debe guardar en disco (para que se pueda adjudicar)

Aprendido de cs074D, donde no se pudo verificar la conservación porque el campo no existía en
el JSON:

- **por corrida individual, no solo promedios:** `beta`, número de grumos, fracción de campo
  activo, y `beta` de **ambos** brazos
- balance de energía del campo por corrida (para poder verificar conservación de verdad)
- `dt`, semilla, malla y todos los `CAMPO_*` usados, en el propio JSON
- las corridas fallidas con su **motivo escrito** (en cs074D las 353 fallidas no traían nota)

---

## 4. Reglas que este diseño mantiene del pacto

- **Ningún número estructural puesto a mano.** Los rangos salen de valores medidos en cs074A
  y cs074D o de constantes que ya existen en ANIMA, barridas en ×10 alrededor.
- **El motor no se toca improvisando.** Si algo que este diseño pide no está en la
  arquitectura, CC **para y reporta** — como hizo, correctamente, en cs074D.
- **El experimento no se cierra** sin autorización explícita del director.
- **Nada de esto reformula la Teoría.** Es un experimento sobre un modelo.

---

## 5. Decisión que necesito del director

1. ¿Se aprueba el NULL por **memoria** (congelada / barajada) en reemplazo del NULL por
   pipeline secuencial? Es el cambio de fondo de este rediseño.
2. ¿Se aprueba que la actualización simultánea pase de hipótesis a decisión de
   implementación?
3. ¿Se aprueba `beta` (retraso respecto de la expansión) como observable de inercia?

Con esos tres sí, el documento se pasa a CC tal cual, y **S1 devuelve en 15 minutos** si el
control sirve o no.
