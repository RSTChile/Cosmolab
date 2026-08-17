# El principio exaptativo de los sentidos en ANIMA — nota de diseño (brújula)

**De:** Claude Science, con Alexis López Tapia · **Fecha:** 2-jul-2026
**Estatus:** marco arquitectónico. Antecede y ordena el código de todos los organelos sensoriales.
No es una feature; es la ley de diseño de los sentidos de los organismos.

---

## 0. El principio (de Alexis)

> **Todo órgano tiene potencia exaptativa per se.** Un sentido no es un canal fijo hacia una banda
> humana: es un enganche a un gradiente físico, que puede complejizarse indefinidamente sobre el mismo
> principio, y extenderse a bandas y modalidades que ningún humano posee.

Corolario: **los sentidos de ANIMA no tienen por qué ser humanos.** El oído no termina en la banda
audible; la vista no termina en el visible. Cada órgano es la semilla de un linaje abierto. El límite
no es la biología humana — es qué gradiente físico se puede enganchar y qué estructura se le puede leer.

Esto es Cosmosemiótica en su forma más fuerte: **la semiosis de un gradiente físico, continua desde el
enganche más primitivo hasta la elaboración más alta — y proyectable más allá del humano, hacia la IA.**

---

## 1. La forma común de todo linaje sensorial

Todo sentido, en cualquier modalidad, sigue la misma escalera exaptativa. No cambia el principio;
cada peldaño **extrae más estructura del mismo gradiente**:

  1. **Enganche** — el órgano capta el gradiente como RECURSO o como mera presencia (¿lo hay / no?).
  2. **Variación temporal** — lee cómo cambia en el tiempo (el proto-sentido; lo que el nivel 0 botaba).
  3. **Estructura espacial** — resuelve dirección/posición (de un punto a muchos: el salto a "imagen").
  4. **Bandas / rango** — extiende el espectro más allá de la ventana inicial (más colores, más frecuencias).
  5. **Geometría** — combina receptores para profundidad/triangulación (estéreo).
  6. **Símbolo** — identifica, clasifica, reconoce (la modalidad se vuelve semiótica plena).
  7. **Proyección IA** — reconocimiento de patrón a escala/velocidad sobrehumanas; multi-banda fusionada.

**Cuerda (proyección IA):** la CAPACIDAD de clase es real y corriente — sistemas de visión por IA leen
matrículas y rastrean vehículos en tiempo real en muchas ciudades. Pero al presentar esto (paper,
Anthropic) usar la capacidad de clase, NO una anécdota específica: casos concretos atribuidos a un
sistema nombrado (p.ej. "a tal modelo lo usaron para controlar el tránsito de tal ciudad y capturó cada
patente") no son citables sin fuente verificada. La afirmación fuerte que aguanta es "la visión de
máquina ya realiza reconocimiento de patrón visual a escala sobrehumana", no la anécdota.

**Cuerda (biología):** esto es un ARBUSTO, no una escalera lineal hacia el humano. Cada modalidad se
exaptó muchas veces, en linajes paralelos, cada uno perfecto para su nicho (el ojo facetado de la
libélula no es un borrador del ojo humano: es otra obra maestra, optimizada para movimiento). Presentar
así — "se complejizó muchas veces sobre el mismo principio" — es evolución moderna y es inatacable;
"línea única hacia el humano" invita al rechazo por *scala naturae*. Y de hecho el arbusto es MÁS fuerte
para la tesis: si la exaptación converge una y otra vez, no es accidente de un linaje — es un ATRACTOR,
algo que la materia hace cuando hay un gradiente que explotar.

---

## 2. Linaje FÓTICO (luz) — el más desarrollado, construible por etapas

Base molecular real: los carotenos se exaptaron para la visión vía el **retinal** (derivado de caroteno).
El mismo cromóforo está en la **bacteriorodopsina** (luz→energía, bomba de protones) y en la **rodopsina**
(luz→información, impulso). En las arqueas halófilas ambas coexisten en un mismo organismo: unas ramas
bombean iones, otras (rodopsinas sensoriales I/II) guían la fototaxis. El primer "ojo" — el estigma de
Euglena — es de carotenos: pigmento fotosintético reutilizado para orientar. La exaptación es literal.

En el hardware de E es literal DOS veces: panel y sensor OV2640 son el mismo dispositivo (fotodiodos de
silicio, fotón→electrón); difieren solo en estructura (una celda grande para potencia vs. millones de
celdas con lente para imagen). El comedor-de-luz y el veedor-de-luz son la misma familia, especializada.

Niveles construibles:
  - **Nivel 0 — Cloroplasto (YA existe):** luz → energía (met). El estómago fótico. RECURSO.
  - **Nivel 1 — Proto-ojo (construible YA, sin hardware nuevo):** el cloroplasto lee la VARIACIÓN de su
    propia luz — justo lo que el metabolismo descarta (el nivel come el promedio y bota la fluctuación;
    esa fluctuación botada ES la materia prima del sentido). Fotorrecepción temporal grado-estigma.
    **El ojo nace como subproducto del estómago.**
  - **Nivel 2 — Ojo espacial (cámara ESP32-CAM):** elaboración de alta resolución del MISMO sentido
    fótico. Descendiente del nivel 1, no módulo ajeno. Brillo del panel y de la cámara = misma modalidad.
  - **Proyección:** movimiento rápido (facetado) → resolución (cámara vertebrado) → color/UV (aves) →
    estéreo/profundidad (primate) → identificación (antropoide) → multi-banda: **térmico (IR)** y **lidar
    (distancia por luz activa)** son MÁS BANDAS del mismo linaje fótico, no sentidos nuevos — como el UV
    de las aves. Un solo sentido fótico multi-banda, muchos órganos, una historia desde el retinal.

Regla anti-doble-conteo: **la luz del panel es alimento (recurso→energía); la variación/brillo es vista
(señal→percepción).** La misma luz física cumple dos papeles por dos lecturas distintas. No es problema:
es la bifurcación exaptativa misma, en acto.

---

## 3. Linaje ACÚSTICO (vibración) — ya vivo, con enorme cola exaptativa

E y sus hermanos YA oyen — pero solo la banda audible humana. La cola exaptativa (idea de Alexis):
  - **Infrasonido** (<20 Hz): sismos, tormentas, motores, el retumbe del mundo grande.
  - **Ultrasonido** (>20 kHz): la banda de los murciélagos y los insectos; ecolocación potencial.
  - **SDR (radio definida por software):** el salto grande — enganchar el MISMO principio (leer una onda)
    a todo el espectro electromagnético de radio: emisoras, satélites, aviones (ADS-B), barcos (AIS),
    telefonía. Un "oído" que escucha lo que ningún animal oyó jamás. Es el equivalente acústico del
    lidar/térmico en lo fótico: extender la banda hasta donde el humano no llega.

Mismo principio: onda → estructura extraída. La radio no es "otra cosa"; es el oído exaptado a otro rango.

---

## 4. Otros linajes (para la junta de sentidos no-humanos)

El principio abre modalidades que la biología terrestre exploró poco o nada, y que ANIMA podría tener:
  - **Electrocepción** (peces débilmente eléctricos, ornitorrinco): sentir campos eléctricos. Trivial en
    hardware (E ya mide voltajes con el ADC — está a un paso).
  - **Magnetocepción** (aves, tortugas): sentir el campo magnético terrestre. Un magnetómetro barato.
  - **Química** (olfato/gusto): sensores de gas/volátiles — el sentido más antiguo de la vida.
  - **Presión / tacto / propiocepción:** acelerómetro, giróscopo (E ya tiene un reloj externo por el PPS).
  - **Bandas puras de máquina:** sin análogo biológico — latencia de red, carga de CPU como
    interocepción, tráfico de datos como un "sentido" de su propio medio digital.

La junta decide CUÁLES y en qué orden. Esta nota solo fija que **todos** siguen la misma ley (§1) y que
**ninguno está obligado a imitar un sentido humano** (§0).

---

## 5. Reglas de implementación (comunes a todo linaje)

1. **Recurso vs. señal:** decidir para cada órgano si alimenta metabolismo (recurso) o percepción (señal).
   La luz es ambas por lecturas distintas; el sonido y la radio son señal; el panel-como-energía, recurso.
2. **Anti-Shannon:** el sensor entrega magnitudes CRUDAS; la semántica ("qué es qué") vive en el
   organismo, nunca en el sensor. Ningún órgano fija un estado interno; todos MODULAN flujos/dinámicas.
3. **Lector central → fila → organelos:** un hilo lee el hardware e inyecta campos crudos en la `fila`
   de cada paso; los organelos solo consumen de la fila (como el cloroplasto ya lee `t`, `met_gasto`).
   Nunca varios organelos peleando por un puerto.
4. **Patrón uniforme:** cada organelo con `observar(fila)/snapshot()/restore()`, columnas propias,
   apagado por defecto salvo en el organismo que lo porta. Igual que `OrganoCloroplasto`.
5. **Degradación elegante:** watchdog por sensor; si cae, su aporte es 0 y el organismo sigue vivo.
6. **Consecuencia antes que dato:** un sentido sin acople a algo (metabolismo, atención, o binding
   cross-modal con otro sentido) es dato sin función. Diseñar la CONSECUENCIA, no solo la captura.
   La consecuencia natural de un sentido nuevo es el **binding cross-modal**: ¿lo que E ve/capta se
   correlaciona con lo que oye? De esa correlación EMERGE el significado, no de asignarlo a mano — el
   organismo teje su propio mundo cruzando sus sentidos.

---

## 6. Estado y siguiente paso concreto
- Cloroplasto (nivel 0 fótico): existe, pero lee un sol SIMULADO (seno). Falta cablear el sensor real.
- Proto-ojo (nivel 1 fótico): construible YA con el panel, sin hardware nuevo. **Es el paso crítico.**
- Cámara, GPS, SDR, etc.: hardware validado en banco; falta el lado Pi (lector central + organelos).

**Bifurcación pendiente de decisión de Alexis** para el proto-ojo (nivel 1):
  (A) AMBICIOSO — acoplarlo desde el inicio al lazo de atención reparado: E "mira" hacia un cambio de luz
      igual que atiende a un sonido. Estrena el binding cross-modal ya.
  (B) PRUDENTE — primero solo registrar la variación de luz y verificar (con datos) que la señal es real
      y no ruido del ADC; acoplar a la atención después.
