# FASE IV — Relación de orden superior: 4 sustratos vs la Pared R7

**Script:** `cs082_fase4_4sustratos.py` (nuevo, no toca ningún script existente) · **Fecha:** ago-2026
**Estado:** exploratorio, primer ataque de Fase IV (roadmap de 5 analistas, 5-ago-2026). Reporta números.
**Veredicto final: de Alexis.** Ningún resultado de este documento cierra ni confirma nada.

---

## 1. Qué intentó la Pared R7 original y por qué falló (resumen, para no repetir el camino)

La Pared R7 es un hallazgo **ya cerrado** de CG002 (30-jun-2026), no un arco aparte. El motor de
juguete usa una regla de compatibilidad **pareada** `c_ij` — un número por cada PAR de nodos, como un
producto interno de a dos. Con esa regla:

- **Lo que es "de a dos" en física real funcionó:** color SU(3) (r7a), carga eléctrica U(1) (r7d),
  generaciones (r7e) — todos se pueden escribir como un producto interno de a pares, y el motor produjo
  polarización emergente real (magnitud reproducible + signo por historia, sin trampa inyectada).
- **Lo que es "de a tres" en física real quedó bloqueado:** el vértice del gluón (r7b, un quark cambia
  de color emitiendo un gluón — 3 actores a la vez) y el vértice de Yukawa/Higgs (r7f, dos fermiones +
  un bosón escalar — también 3 a la vez). Con `c_ij` pareado, ambos colapsaron: los gluones quedaron
  100% inertes (nunca se aniquilaban entre sí) y el Higgs quedó desacoplado (Σ|c_ij|=0, nunca se
  enganchó a nada). **No son dos fallos — es un solo fallo, tocado dos veces**: un vértice de 3 puntos
  no se reduce a un escalar de 2 puntos.

**Ya se intentó extender esto una vez (CS032/R7g, 30-jun-2026) — y no repetimos ese camino.** Grok
sumó un término extra de 3 cuerpos (`v_ijk·m_j·m_k`) ENCIMA de la misma estructura de grafo pareado.
El resultado, tras auditoría (retractado el titular inicial "la pared se abre"): **"la pared se movió,
no cayó"** — apareció acoplamiento donde antes había cero, pero el octeto de gluones seguía muriendo
(~0-1% supervivencia), la fracción de Higgs no cambió realmente (f≈0.30 sin mejora), y la razón de masa
quedó en 1.001 (sin jerarquía — ninguna señal). La lección que dejó ese intento: **sumar una fuerza de
3 cuerpos a un objeto que sigue siendo de 2 (una arista con 2 extremos) no alcanza.** Hace falta un
objeto-relación cuya aridad NATIVA sea 3 o más, o una relación que actúe sobre otra relación — no un
parche de fuerza sobre la misma estructura.

Esa es exactamente la instrucción del roadmap de Fase IV, y por eso los 4 sustratos de abajo cambian la
**estructura de datos** de la relación, no sólo la fórmula de la fuerza.

---

## 2. Los 4 sustratos y el control de equiparación

Los 4 comparten: mismos N=110 nodos, mismo grafo aleatorio base (mismo seed → misma adyacencia), mismo
alfabeto de orientación Z₆ (la convención "sextante" ya usada en `cs052_v1_coemergencia.py` y
`cg004f3_cinta_eisenstein.py` — no es física importada, es sólo la unidad angular del proyecto), misma
constante de acople J=0.6, mismo ruido por paso, y el mismo **presupuesto total de cómputo**
(~55.000-75.000 operaciones-relación) — el número de "sweeps" se ajusta por sustrato para que ninguno
gane sólo por correr más pasos. **Guardián no-hornear:** en ningún punto del código hay una coordenada
(x,y) — sólo adyacencia y estado relacional (verificable por inspección directa del archivo).

| Sustrato | Objeto-relación | Qué mide/hace de nuevo respecto al anterior |
|---|---|---|
| **1. Grafo diádico** (línea base) | arista (2 nodos) | orientación Z₆ que se alinea con las aristas vecinas (comparten 1 nodo) — el límite pareado de siempre |
| **2. Hipergrafo** | triángulo tratado como UN objeto que toca 3 nodos A LA VEZ | la relación misma tiene aridad 3 nativa (no son 3 aristas, es 1 hiperarista) |
| **3. Complejo simplicial** | aristas (igual que 1) + cara que MIDE la holonomía de sus 3 aristas de borde | una relación (cara) observa a otras 3 relaciones (aristas), pero no actúa sobre ellas — control "sólo estructura, sin retroalimentación" |
| **4. 2-complejo con feedback** | igual que 3, pero la cara EMPUJA de vuelta a sus 3 aristas para reducir su propio defecto | una relación actúa causalmente sobre otras relaciones — la forma más completa de "relación de relaciones" |

**Tabla de control de equiparación** (una corrida, N=110, 5 seeds):

| sustrato | grados de libertad (Z₆) | sweeps | operaciones-relación totales | runtime prom. |
|---|---:|---:|---:|---:|
| 1 grafo diádico | 545 | 106 | 57.770 | 1.3 s |
| 2 hipergrafo | 161 | 328 | 52.808 | 1.3 s |
| 3 simplicial | 706 | 106 | 74.836 | 1.4 s |
| 4 2-complejo feedback | 706 | 106 | 74.836 | 1.4 s |

Los grados de libertad **difieren por diseño** — es la variable manipulada (aridad/estructura), no un
descuido escondido: se reportan explícitos para que cualquiera pueda auditar que la comparación no está
trucada. Lo que sí se igualó fue el presupuesto de cómputo, el alfabeto, el acople, el ruido y la
topología base.

---

## 3. Resultado comparativo

**Holonomía de triángulo** (`|suma de las 3 orientaciones de borde| mod 6`, centrada — misma fórmula
para los 4, aplicada al mismo conjunto de triángulos, promedio sobre 5 seeds):

| sustrato | h_REAL | h_NULL | h_SHUFFLED | REAL/NULL | ejes_REAL | ejes_NULL | ¿separa de NULL? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 grafo diádico | 2.13 | 1.54 | 2.17 | 1.38 (RUGOSA) | 1.0 | 3.8 | **no** |
| 2 hipergrafo | 1.35 | 1.46 | 1.34 | 0.93 (aplana débil) | 1.0 | 4.2 | **no** |
| 3 simplicial | 1.20 | 1.54 | 1.19 | 0.78 (aplana débil) | 1.0 | 3.8 | **no** |
| 4 2-complejo feedback | **0.30** | 1.54 | 0.57 | **0.19 (APLANA fuerte)** | 1.2 | 3.8 | **sí** |

("separa de NULL" = el efecto |h_REAL−h_NULL| supera 2× la variabilidad entre las 5 semillas — el
mismo criterio de robustez que usa el resto del proyecto.)

- **Sustrato 1 (diádico):** NO se distingue de NULL con robustez estadística (y si acaso, el signo es
  hacia MÁS frustración, no menos). Alinear de a pares no cierra los lazos de 3 — coherente con el
  diagnóstico teórico de la Pared R7.
- **Sustrato 2 (hipergrafo, aridad 3 nativa pero SIN feedback):** aplana apenas (ratio 0.93), pero el
  efecto es del tamaño del ruido entre semillas — **no separa de NULL**. Tener un objeto-relación de
  aridad 3 **por sí solo, sin que se relacione con otras relaciones, no alcanzó** en esta batería.
- **Sustrato 3 (simplicial pasivo, cara que sólo mide):** mismo resultado que el diádico en la práctica
  (aplana un poco por el mismo motivo que el 2, no separa de NULL) — confirma que **agregar una capa de
  medición sin retroalimentación no cambia nada**, un control negativo limpio.
- **Sustrato 4 (2-complejo CON feedback cara→arista):** es el único que separa de NULL con claridad —
  holonomía ~5× menor que el azar, y el efecto se repite entre las 5 semillas (desviación
  entre-semillas la más chica de los 4, 0.14 contra un promedio de 0.30 — es reproducible, no ruido de
  una corrida suelta).

**Conexión independiente de la métrica:** por construcción ningún sustrato usa coordenadas — la única
información disponible es "quién-con-quién" (adyacencia) y el estado Z₆ de cada relación. La
estabilidad del sustrato 4 entre semillas (misma magnitud de aplanamiento en las 5 corridas
independientes) es la evidencia operativa de que el efecto es una propiedad relacional repetible, no
ruido de una realización particular.

**Ejes múltiples:** los 4 sustratos convergen a **un solo eje dominante** (ejes_REAL≈1.0-1.2) contra
~4 "ejes" espurios de NULL (que son sólo el ruido de un histograma sobre valores al azar, no ejes
reales). Este metro distingue "hubo dinámica" de "no hubo dinámica" en los 4 por igual — no distinguió
entre sustratos. No hubo evidencia de **multi-eje genuino** en ningún sustrato con estos parámetros.

**Advertencia honesta (posible confound, no escondida):** en el sustrato 4, el control SHUFFLED (mismos
valores finales, topología barajada) da 0.57 — mucho más cerca de REAL (0.30) que de NULL (1.54). Esto
dice que **parte** del aplanamiento es un efecto de "hacia dónde se concentra la distribución de
valores" (un consenso global que el feedback empuja) y no puramente de que CADA triángulo cierre su
propio lazo por su topología local. El efecto sigue siendo real (SHUFFLED también queda muy por debajo
de NULL), pero es más débil y más mixto de lo que parece mirando sólo REAL vs NULL. Esto es análogo al
"medio B se desenrosca" que ya describió `cs052_v1_coemergencia.py` — hay que leer con cuidado antes de
festejar.

---

## 4. ¿Algún sustrato de aridad 3+ logró algo que el diádico no?

**Sí — pero no el que el roadmap podría esperar de entrada.** No fue el hipergrafo "puro" (aridad 3,
sin feedback) el que superó al diádico — ese quedó estadísticamente indistinguible de NULL, igual que
el diádico. El que sí separó de NULL con claridad fue el **sustrato 4 (2-complejo con retroalimentación
cara→arista)** — que además de tener aridad 3 en sus caras, tiene algo extra: una relación (la cara)
que **actúa causalmente** sobre otras relaciones (sus 3 aristas de borde), no sólo las mide.

Esto matiza el resultado en una dirección específica y auditable: en esta batería mínima, **la aridad-3
sola no bastó — hizo falta que una relación pudiera empujar a otra relación** (feedback entre niveles),
no sólo tocar más nodos a la vez. Es el mismo patrón de "co-emergencia" que ya encontró
`cs052_v1_coemergencia.py` con el patrón A=0, B=0, C-discrimina: ni la entidad sola ni la conexión
libre generaban curvatura — sólo el vínculo ATADO. Acá: ni el grafo pareado (1) ni el objeto de aridad 3
aislado (2) ni la medición pasiva (3) bastaron — sólo la relación que se ata y empuja a otras (4).

---

## 5. En simple, con analogía

Imaginate un grupo de personas tratando de ponerse de acuerdo sobre "hacia dónde apunta el norte",
usando sólo relaciones locales, sin brújula compartida ni mapa:

- **Grafo diádico:** cada persona sólo puede susurrarle a UN vecino a la vez, de a pares. Se van
  copiando entre vecinos, pero nadie nunca chequea si tres personas que se conocen entre sí (un
  "triángulo" de conocidos) terminaron de acuerdo LOS TRES a la vez. Resultado: no mejor que el azar.
- **Hipergrafo:** ahora hay reuniones de tres personas hablando al mismo tiempo (en vez de a-pares).
  Pero cada reunión es sorda a las demás — no hay un árbitro que compare reuniones. Resultado: casi
  igual que antes, tampoco mejor que el azar con claridad.
- **Complejo simplicial:** aparece un árbitro que SÍ anota cuando tres personas de un triángulo quedan
  contradichas entre sí — pero no le dice nada a nadie, sólo lo anota en su libreta. Resultado: igual
  que antes, la anotación sola no cambia nada si nadie corrige.
- **2-complejo con feedback:** el mismo árbitro, pero ahora SÍ interviene: cuando ve una contradicción
  entre los tres, les dice "corrijan, no cierran" — y las tres personas ajustan un poco su postura.
  Resultado: es el único caso donde el grupo termina más de acuerdo que si hubiera apuntado al azar, y
  ese resultado se repite en 5 corridas independientes.

La lectura simple: **no alcanza con juntar a tres a la vez (aridad) — hace falta que exista alguien
(una relación de nivel superior) que pueda corregir a los que están en desacuerdo.** Eso es justo lo
que faltaba en el motor pareado de la Pared R7 original, y es lo primero, en esta batería mínima
y de escala chica, que mostró una separación robusta de un control NULL.

---

## 6. Qué NO se reclama

- No es un motor físico (no hay grupos gauge, ni masa, ni partículas reales) — es estructura relacional
  pura, como pidió el roadmap.
- El efecto del sustrato 4 tiene el confound parcial de distribución global señalado en §3 — no se
  reclama holonomía "geométrica" limpia como la de `cg004f3_cinta_eisenstein.py` (esa usa una
  triangulación fija {3,q} con giros cuantizados exactos; acá el campo es un consenso dinámico sobre un
  grafo aleatorio, con Eisenstein exacto NO aplicable).
- Escala chica (N=110, 5 semillas) — presupuesto de tiempo de esta tanda. Un barrido con más semillas,
  más N, y variando J/ruido sistemáticamente daría un veredicto más firme.
- Ningún resultado de este documento se declara cerrado, confirmado o refutado. Los números están
  arriba. El veredicto es de Alexis.

**Reproducibilidad:** `cs082_fase4_4sustratos.py`, sin dependencias fuera de numpy, corre en ~25-30s con
la configuración actual (`./venv/bin/python3 cs082_fase4_4sustratos.py`).
