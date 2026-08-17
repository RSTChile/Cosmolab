# DISEÑO CS057 — El PAISAJE completo: barrido de TODAS las fuerzas (0→1) + sector oscuro EMERGENTE + brazo sincrónico/asincrónico. ¿Qué combinaciones estabilizan un universo en expansión — de cualquier tipo?

**Número:** CS057 (secuencia CS) · **Dimensión técnica:** de "¿emerge 3D?" a "¿qué combinaciones de las
seis fuerzas ESTABILIZAN un universo persistente en expansión, de cualquier dimensión?". Barrido del espacio
completo de intensidades (0→1) con la distancia modulando cada fuerza según su alcance efectivo; sector
oscuro (materia/energía) medido como SALIDA emergente, nunca insertado; brazo sincrónico vs asincrónico como
falsación del "es un proceso". Juez: estabilidad + expansión medidas CIEGAS; nuestro universo (3D-plano-
expansión) es UN punto del mapa, no el objetivo.
**Planteo (Alexis):** "Probar múltiples variantes de inicio y ver qué combinaciones se estabilizan y
permiten un universo de cualquier tipo, con cualquier dimensión, en expansión... incluir materia y energía
oscuras NO como algo dado, sino como probabilidad de algo cuando todas las fuerzas actúan variando sus
valores, sincrónica y asincrónicamente (para falsar). Las fuerzas se afectan mutuamente de manera sincrónica
—a menos que pueda existir más de un tiempo a la vez."
**Diseño:** Claude Science (CS) · **Planteo físico y método:** Alexis López Tapia. · **Estado:** DISEÑO, a
codear CC. · **Fecha:** 5-jul-2026 · Reusa CS055/CS056 (proceso, arnés, medidor). Subsume CS056-v2 (alcances
distintos) como uno de los ejes.

---

## 0. EL CAMBIO DE PREGUNTA (lo más importante — y es de Alexis)
Dejamos de preguntar "¿emerge 3D?" (privilegia nuestra respuesta = riesgo de circularidad). Preguntamos:
**"¿QUÉ combinaciones de fuerzas estabilizan un universo persistente en expansión —de la dimensión que
sea?"** Se mapea el paisaje entero de universos posibles; nuestro universo (3D, plano, en expansión) es UN
punto del mapa. Esto es epistemológicamente más limpio: el poder falsador está en ver TODO el paisaje y
localizar dónde cae el real, no en cazar 3D.

## 1. EL BARRIDO COMPLETO (todas las fuerzas 0→1, no un par)
Seis fuerzas, cada una con un peso variable en [0,1]: gravedad · fuerte(confinamiento) · electromagnetismo ·
débil · despliegue/expansión · (enfriamiento como reloj, siempre presente). En vez de barrer un solo par
(CS055/CS056), se MUESTREA el espacio de combinaciones:
- **Muestreo del volumen:** ~algunos miles de puntos (Latin-hypercube o al azar) en el hipercubo de pesos
  [0,1]^k, cubriendo el espacio — NO malla completa (100^6 es imposible; restricción de ingeniería honesta).
- **El punto físico REAL marcado:** las intensidades del mundo (fuerte 1 / EM 1/137 / débil 1e-6 / gravedad
  1e-38, con sus alcances) se corren y se SEÑALAN en el mapa. El valor físico es un punto localizado, no el
  objetivo del barrido.
- Reportar el PAISAJE entero: para cada punto, qué tipo de universo salió (dim, curvatura, estable?,
  expande?). No se elige el punto que da 3D — se muestra el mapa y dónde cae lo real.

## 2. LA DISTANCIA MODULA CADA FUERZA SEGÚN SU ALCANCE (subsume CS056-v2)
Misma ley `1/d²` para las de largo/medio alcance, pero ALCANCE EFECTIVO distinto por física (hallazgo del
turno de Alexis sobre EM vs gravedad):
- Gravedad: alcance LARGO (se acumula, nunca se cancela).
- EM: alcance CORTO (se cancela por neutralidad de carga a escala).
- Fuerte/confinamiento: alcance ULTRA-CORTO (solo vecindad inmediata, como en la realidad).
- Débil: alcance ULTRA-CORTO (transmutación local).
La distancia (saltos de grafo) modula TODO, como pidió Alexis. Los alcances se fijan por física ANTES de
correr (no se afinan).

## 3. EL CRITERIO — ESTABILIDAD + EXPANSIÓN, medidos CIEGOS (la trampa cerrada)
"Estable y en expansión" se define afilado y ciego a los valores de las fuerzas, o se satisface trivial:
- **ESTABLE:** estructura conexa PERSISTENTE (componente gigante que no se disuelve ni colapsa a punto) con
  geometría MEDIBLE (dim y curvatura no degeneradas) que se mantiene en el tiempo tardío. Un blob (colapso)
  NO es estable; un gas (fragmentado) NO es estable.
- **EN EXPANSIÓN:** el diámetro (en saltos de grafo) CRECE con el tiempo. Un universo estático o que colapsa
  NO expande.
- **Ambos medidos POR TIPOS y por trayectoria, ciegos a los pesos de las fuerzas** (G-CIEGO). El clasificador
  de dim se lee por TIPOS de retículo, no por el contador roto (regla de CS054-v2).
- Un universo "viable" = estable Y en expansión. Se cuenta cuántos puntos del barrido lo logran, de qué
  dimensión son, y DÓNDE cae el punto físico real.

## 4. EL SECTOR OSCURO — EMERGENTE, jamás insertado (la disciplina dura de Alexis)
Materia y energía oscuras NO entran como términos. Se buscan como SALIDAS emergentes del interjuego de
fuerzas:
- **Candidato a ENERGÍA OSCURA = expansión que se ACELERA sola.** Si en algún punto del barrido el diámetro
  no solo crece sino que crece cada vez MÁS RÁPIDO —sin que se haya metido ningún término de aceleración, con
  el despliegue a tasa fija— eso es un candidato a energía oscura. Debe ser SORPRESA de salida.
- **Candidato a MATERIA OSCURA = gravitación extra sin fuente visible.** Si la estructura se comporta como
  si hubiera más masa de la que aportan los nodos con color/carga (más contracción/estructura que la que las
  fuerzas visibles explican), eso es un candidato a materia oscura.
- **G-NO-INSERTAR-OSCURO (assert de código):** ningún término se llama "oscuro", ninguno se ajusta para
  calzar con la aceleración observada. El sector oscuro solo puede APARECER como propiedad medida de la
  trayectoria, nunca como entrada. Si lo metemos, nos copiamos la respuesta. (Esto es la regla anti-Shannon
  de Alexis aplicada al sector oscuro — ingeniería, no supuesto.)
- La lectura honesta: si NINGÚN punto produce expansión acelerada emergente → el modelo no genera un análogo
  de energía oscura con estas fuerzas (negativo informativo, dice qué falta). Si ALGUNO lo produce → mapear
  dónde, y si coincide con universos estables-expandiendo.

## 5. EL BRAZO SINCRÓNICO vs ASINCRÓNICO (la falsación del "es un proceso" — de Alexis)
La prueba directa de la tesis "es un proceso, no una sucesión", ahora al nivel de las fuerzas:
- **SINCRÓNICO:** todas las fuerzas actúan en CADA paso, a la vez (el default físico). Es como corrió CS055/
  CS056.
- **ASINCRÓNICO (brazo NULO):** las fuerzas actúan por TURNOS, una tras otra en fases separadas (gravedad un
  tramo, luego EM, luego confinamiento...). Emula "una sucesión de sucesos" en vez de un proceso.
- **La falsación:** si sincrónico y asincrónico dan el MISMO paisaje → el acoplamiento simultáneo no importa,
  y la tesis "es un proceso" queda FALSADA (sería solo una sucesión). Si el sincrónico estabiliza universos
  que el asincrónico NO → la simultaneidad es esencial, y la tesis queda PROBADA al nivel de las fuerzas.
- Fundamento físico (de Alexis): el mundo es sincrónico porque hay UN tiempo; el asincrónico solo sería
  físico si existiera más de un tiempo a la vez. El asincrónico es, por eso, el brazo de contraste correcto.

## 6. GUARDIANES (ingeniería del código)
1. **G-NO-PRESUPONER-ESPACIO:** toda distancia por saltos de grafo (BFS), jamás coordenada. Assert.
2. **G-CIEGO:** estabilidad, expansión, dimensión, sector oscuro — TODOS medidos ciegos a los pesos de las
   fuerzas; ninguna medida recibe "3D" ni un objetivo. Assert.
3. **G-NO-INSERTAR-OSCURO:** ningún término "oscuro" de entrada; el sector oscuro solo como salida medida.
   Assert.
4. **G-ALCANCE-FISICO:** los alcances (gravedad largo / EM corto / fuerte-débil ultracorto) se fijan por
   física ANTES; no se afinan.
5. **G-MUESTREO-REPORTADO:** se reporta el PAISAJE entero del barrido (todos los puntos), no solo los que
   dan universos viables ni solo el que da 3D. El punto físico se marca.
6. **G-NULL-ASINCRÓNICO:** el brazo asincrónico es el contraste de la tesis del proceso.
7. **G-NO-TUNE:** nada se re-afina buscando un resultado; los valores físicos son datos del mundo.

## 7. LOS DESENLACES (pre-escritos, honestos — el mapa informa pase lo que pase)
- **Muchas combinaciones estabilizan universos en expansión, de dimensiones variadas, y el punto físico cae
  entre los que dan 3D-plano-expansión** → nuestro universo es uno de los viables, y las fuerzas reales lo
  producen. Confirmación fuerte de la tesis del proceso.
- **Solo el sincrónico estabiliza universos; el asincrónico no** → la simultaneidad de las fuerzas es
  esencial ("es un proceso" probado), independiente de si sale 3D.
- **Aparece expansión acelerada emergente en alguna región** → candidato a energía oscura SIN insertarla;
  mapear si coincide con universos viables.
- **El punto físico NO cae entre los viables, o ninguna combinación estabiliza** → las fuerzas locales (aun
  todas juntas, barridas, con distancia) no bastan; el paisaje dice qué falta (apunta al espín/marco R7,
  aguas arriba de las fuerzas locales). Negativo informativo.

## 7-bis. LA ESCALA EXHAUSTIVA (explícito — esto NO es una simulación de 2 variables)
Instrucción de Alexis: "miles de variaciones; no me importa si se demora hasta mañana o más. Fuera la idea
de resolver el Big Bang con 2 variables y tres copas de vino." El barrido se especifica a escala, con
números concretos y su justificación — para que NO se corra en miniatura:

**A. Dimensión del espacio de búsqueda.** k = 6 pesos de fuerza continuos en [0,1] (gravedad, fuerte, EM,
débil, expansión/despliegue, + tasa de enfriamiento) × 3 ejes de alcance (largo/corto/ultracorto, ya
fijados por física pero con 1 eje de razón-de-alcances libre para mapear) ≈ 7 dimensiones efectivas.

**B. Muestreo — SOBOL, no al azar ingenuo.** Secuencia de Sobol (baja discrepancia) sobre el hipercubo
[0,1]^k → cobertura uniforme del volumen sin huecos ni cúmulos. **N_puntos = 4096** (2^12, potencia de 2
para que Sobol sea balanceado). Es el piso; si el paisaje sale rugoso, subir a 16384 (2^14) en una segunda
tanda. Esto es "miles de variaciones" de verdad, no una malla gruesa.

**C. Réplicas por punto.** Cada punto del hipercubo se corre con **S = 8 semillas** distintas (condición
inicial del ensemble + estocasticidad del proceso) → se reporta media y dispersión de cada métrica. Un
punto no es "viable" por una corrida afortunada: tiene que serlo en la mayoría de sus 8 semillas
(robustez). → 4096 × 8 = **32.768 corridas** en la tanda base.

**D. El ensemble inicial dentro de cada corrida.** Cada corrida arranca de un ensemble simétrico de
geometrías/dimensiones (d≈1..4+, plana/curva±) — NINGUNA privilegiada. Así, un punto de fuerzas se evalúa
sobre TODAS las dimensiones de partida a la vez (¿cuáles sobreviven bajo esas fuerzas?).

**E. Los dos brazos (sync/async) DUPLICAN la tanda.** Sincrónico + asincrónico = 2 × 32.768 = **65.536
corridas**. Más los brazos G-APAGADO (subconjuntos de fuerzas) y G-NULL (color/carga barajados) sobre una
submuestra representativa (~512 puntos × 8 semillas × 2 brazos ≈ 8.192 corridas de control).

**F. El punto físico REAL, resuelto fino.** Alrededor del punto físico (fuerte 1 / EM 1/137 / débil 1e-6 /
gravedad 1e-38, con sus alcances) se hace un sub-barrido DENSO (una malla local fina, ~256 puntos × 8
semillas) para resolver bien la vecindad del universo real — porque ahí es donde la respuesta importa y no
basta la densidad de Sobol global.

**G. Presupuesto total y tiempo.** Orden de ~75.000-90.000 corridas del proceso. Cada corrida es el bucle
temporal de CS055/CS056 (numpy puro, N≈300-450 nodos, ~cientos de pasos). A ~1-3 s/corrida en un núcleo,
son ~25-75 horas mono-núcleo → **paralelizable por puntos** (cada punto es independiente): en 8-16 núcleos,
una noche o dos. Aceptado explícitamente por Alexis ("hasta mañana o más"). Se guarda checkpoint incremental
(cada punto escribe su fila al terminar) para reanudar sin recomenzar — como el experimento del iPad.

**H. Qué se guarda por corrida (para el paisaje).** Fila con: los k pesos, el eje de alcance, la semilla, el
brazo (sync/async/control), y las métricas CIEGAS de salida — estable? (0/1), expande? (0/1), aceleración de
expansión (para el candidato energía-oscura), dim por TIPOS, curvatura, %gigante, diámetro final,
gravitación-sin-fuente (candidato materia-oscura). → un CSV de decenas de miles de filas = EL PAISAJE, sobre
el que se hacen los mapas 2D/3D de "dónde se estabilizan universos en expansión" y "dónde cae el punto
físico real".

**I. Lo que NO se hace (anti-miniatura, anti-horneado).** No se recorta el barrido a un par de fuerzas; no
se corre 1 semilla; no se elige el punto que da 3D; no se ajusta ningún alcance ni peso para calzar. Se
reporta el CSV entero. La escala es el punto: solo con decenas de miles de combinaciones el paisaje es real
y la posición del universo físico en él es significativa.

## 8. RESUMEN OPERATIVO PARA CC
- **Escala EXHAUSTIVA (no negociable — ver §7-bis):** Sobol 4096 puntos × 8 semillas × 2 brazos (sync/
  async) ≈ 65.536 corridas base + controles + sub-barrido denso del punto físico. Orden de ~75-90k
  corridas. Paralelizar por puntos (independientes), checkpoint incremental por fila para reanudar. Días de
  cómputo aceptados. NO correr en miniatura, NO 1 semilla, NO 2 fuerzas.
- Muestrear con SOBOL (baja discrepancia) el hipercubo de pesos [0,1] de las 6 fuerzas + eje de razón de
  alcances; marcar y sub-barrer DENSO el punto físico real. NO malla completa (100^6 imposible).
- Cada fuerza modulada por su ALCANCE efectivo físico (gravedad largo / EM corto / fuerte-débil ultracorto),
  misma ley 1/d² donde aplica, distancia por saltos de grafo.
- Correr el proceso; medir CIEGO: estable? (conexo persistente, geometría medible), expande? (diámetro
  crece), dim POR TIPOS, curvatura. Buscar sector oscuro como SALIDA: expansión acelerada (energía oscura),
  gravitación sin fuente visible (materia oscura) — NUNCA insertados.
- Dos brazos: SINCRÓNICO (todas por paso) vs ASINCRÓNICO (por turnos) — falsación del "es un proceso".
- Reportar el PAISAJE entero (CSV de decenas de miles de filas) + FIGURAS del paisaje: mapa de dónde se
  estabilizan universos en expansión (proyecciones 2D del hipercubo), coloreado por dimensión emergente, con
  el punto físico real MARCADO; histograma de dimensiones viables; sync vs async lado a lado; región (si la
  hay) de expansión acelerada emergente. Traer CSV + figuras + los brazos a CS. Registrar CS057. Sig: CS058.

— Diseño CS057 por Claude Science. El planteo entero (barrer todas las fuerzas 0→1; criterio de estabilidad+
expansión de cualquier dimensión en vez de cazar 3D; sector oscuro emergente no insertado; brazo sincrónico/
asincrónico como falsación del proceso; la distancia modulando todo) es de Alexis López Tapia. La
formalización, los guardianes y las definiciones ciegas de estable/expande, míos. El experimento puede dar
universos viables con las fuerzas reales, o no; puede mostrar el sector oscuro emergente, o no — se corre
para MAPEAR el paisaje de universos posibles y ver dónde cae el nuestro, no para forzar un resultado.
