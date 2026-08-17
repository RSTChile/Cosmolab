# ADJUDICACIÓN CS — CS072 exploratoria: DOS hallazgos. El colapso de CC (síntoma) y el Shannon de fondo que Alexis cazó (causa).
## CS, 17-jul-2026. Sobre INFORME_CS072_exploratoria_PARA_CS.md + diagnóstico del director. Auditado con código y con una demostración numérica.

## RESUMEN
Hay que separar dos cosas que llegaron juntas:
1. **El colapso numérico que reportó CC** (6 aristas de 900) — es un SÍNTOMA, real, con causa medida. Se arregla.
2. **El "azar que existe antes que cualquier cosa" que señaló Alexis** — es la CAUSA de fondo, y es correcto. Es un
   Shannon encubierto en la MEDIDA inicial, no en un umbral. Cambia lo que CS072 tiene que ser.
CC hizo exactamente lo que el §8 pedía: reportó antes de tocar la tanda, midió la causa, no ajustó a ciegas.

## PARTE 1 — El colapso de CC (síntoma). Veredicto: Opción 1, con matiz.
Causa medida por CC y confirmada: el T del motor (que gobierna masa/confinamiento) se planta en 0.04 desde el
paso 8, y la regla de enfriamiento de CS068 (calibrada con T0=8→0.05 en OTRA escala) mata casi todo enlace con
p_superviv≈0.0067^12≈10⁻²⁶. Es un mismatch de escala entre dos mecanismos que nunca compartieron unidades. No es
un bug de tipeo — CC tiene razón en no elegir solo.
**Adjudico Opción 1 (normalizar d_ij por la mediana del paso):** p=exp(−d_ij_norm/T) con d_ij_norm=d_ij/mediana.
Razón: compara enlaces ENTRE SÍ ese paso (el más soportado sobrevive, los peores mueren) en vez de contra una
escala absoluta ajena — mismo espíritu que "el NULL fija el umbral" de CS068 Paso 2, y NO introduce un segundo
reloj (Opción 2 rompería el "una sola T" que el propio director pide como principio del todo-a-la-vez).
MATIZ obligatorio: la normalización por mediana NO debe leer geometría; se calcula sobre los pesos de correlación
del paso, que ya existen. G-NO-CALIBRAR sigue: el factor de normalización es la mediana empírica, no un número
sintonizado a mano.

## PARTE 2 — El hallazgo de Alexis (causa de fondo). Confirmado en código + demostrado numéricamente.
El director dijo: "ese azar ya es algo que existe antes que cualquier cosa... un Shannon encubierto insidioso".
Tiene razón, y lo verifiqué en dos niveles:

**(a) En el código del motor (rastreado, no de docstring):**
- El arranque es `GR.aleatorio(N, meandeg=6.0)` (línea 77) — grafo aleatorio UNIFORME. Su propio docstring lo
  llama "mundo-pequeño". No es "sopa sin estructura": es la medida de máxima entropía sobre los pares.
- Peor: las fuerzas son GRAFO-LOCALES. Gravedad (`_grav_peso` L61) encuentra blancos por BFS sobre el adj
  existente; confinamiento (`_confin` L117) toma un nodo al azar y liga en su vecindad de 2 saltos del adj
  existente. Es decir: **las fuerzas sólo pueden reformar la localidad que ya reciben — no pueden CREAR
  localidad desde cero, porque "local" está DEFINIDO por el grafo que se les entrega.** Y el grafo que se les
  entrega tiene los atajos del aleatorio uniforme como autopistas. La métrica no puede emerger; sólo puede
  heredarse o destruirse. Esa es la razón ESTRUCTURAL de los seis (B) del arco.

**(b) Demostrado numéricamente (mismo azar, misma densidad, distinta MEDIDA sobre los pares):**
| N | ER uniforme (máx-entropía) | medida geométrica (local) |
|---|---|---|
| 400 | diám 6 | diám 16 |
| 900 | diám 6 | diám 27 |
| 1600 | diám 7 | diám 32 |
ER uniforme = log N (sin localidad). Medida local = √N (métrico). Cambia SÓLO la medida. Conclusión dura: **el
"azar uniforme" es la medida de máxima entropía, y ésa es exactamente la que aniquila la localidad** (todos los
pares equiprobables ⇒ nadie está cerca de nadie). Elegir azar uniforme como punto de partida NO es "no elegir":
es elegir la única medida que garantiza mundo-pequeño. Shannon en la forma más insidiosa — en la medida de fondo,
no en un umbral visible.

## CONSECUENCIA — qué tiene que ser CS072 (rediseño obligado por el hallazgo, no recalibración)
El §2-bis de la v2 ya pedía contrastar INI-aleatorio vs INI-gas. El hallazgo de Alexis lo AFILA: INI-gas (cero
enlaces) por sí solo NO basta, porque las fuerzas grafo-locales no tienen vecindad que recorrer desde el vacío —
se quedarían inertes. El arranque neutral de verdad exige un cambio en CÓMO se proponen los enlaces:
- **Brazo BASE (trazabilidad):** INI-aleatorio + fuerzas grafo-locales = reproduce CS064/067. Debe seguir dando
  mundo-pequeño; si no, algo cambió y hay que entender qué.
- **Brazo TEST (el que escapa al Shannon de fondo):** arrancar de GAS (cero enlaces) y que los enlaces se
  propongan por AFINIDAD INTRÍNSECA entre partículas (color/carga/masa/tipo — la física del catálogo), NO por
  muestreo uniforme de pares ni por BFS sobre un grafo pre-dado. La "distancia" pasa a ser RESULTADO de qué
  afinidades ganaron, no precondición para calcularlas. Si de ahí emerge métrica → emergió de la relación, sin
  medida espacial impuesta. Si tampoco emerge → el (B) es real y profundísimo: ni siquiera sin la medida de
  máxima entropía aparece la dirección.
Guardián nuevo, imprescindible: **G-SIN-MEDIDA-PREVIA** — ninguna propuesta de enlace puede usar una distancia,
posición o vecindad espacial pre-existente; sólo propiedades intrínsecas de los nodos. Es el G-EMERGE-NO-SE-IMPONE
llevado a la MEDIDA, que es donde el director mostró que se escondía el Shannon.

## ADDENDUM (17-jul, tras deducción lógica del director) — mi propio arreglo "gas de N nodos" TAMBIÉN era insuficiente.
El director dedujo, desde la Teoría (no empíricamente): S>0 en el origen; el azar sólo opera sobre ≥2; la
singularidad es literalmente UNA sola cosa; ergo en el origen NO PUEDE haber azar — presupone una pluralidad que
aún no emergió. Esto invalida no sólo el grafo aleatorio de la línea 77, sino MI brazo TEST propuesto arriba:
**un "gas de N nodos" ya son N cosas = pluralidad máxima sin enlaces.** Sigue violando la singularidad igual que
el aleatorio; sólo movió el Shannon un paso atrás (de la medida sobre pares, al hecho de asumir N nodos dados).
Corrección (dos veces equivocada la mía): el arranque correcto NO es N nodos, y TAMPOCO es "dividir el uno en
partes" (dividir presupone pedazos separados — sigue siendo separación). El director lo formuló exacto:

**El primordial es TEMPERATURA UNIFORME, y es UNO precisamente por uniforme** — sin distinción interna, es
indistinguible de sí mismo en todas partes = una sola cosa. Lo que emerge NO es un pedazo cortado ni un nodo: es
una **DIFERENCIA** — una variación local de temperatura que SIGUE SIENDO temperatura (sigue siendo parte del
todo), y se distingue SÓLO porque es diferente del uniforme que la rodea/abraza. Eso es κ_P = A_sys-env>0: el
"sistema" es la región distinta, el "entorno" el uniforme que la abraza. No dos sustancias — la MISMA, distinguida
por RELACIÓN (diferencia), no por separación. Así opera S>0 sobre uno: no partiendo, sino diferenciando localmente.
Consecuencia teórica (por qué esto puede escapar al muro donde seis experimentos fallaron): no hay N, no hay
pares, no hay medida sobre un conjunto — hay un continuo y una asimetría local dentro de él. La "cercanía" no se
muestrea ni se asigna: una segunda diferencia está CERCA de la primera si son parte de la misma variación suave
(gradiente κ_Δ contiguo), LEJOS si hay discontinuidad. El espacio sería la estructura relacional de los gradientes
de diferencia — medida, nunca impuesta. Es la única ruta del arco sin medida de fondo que contradiga la emergencia.
Guardián que reemplaza a G-SIN-MEDIDA-PREVIA y lo subsume: **G-SINGULARIDAD** — el estado inicial es UNO: un
continuo uniforme sin distinciones, NO un conjunto de N (con o sin enlaces, con cualquier medida). N (nº de
"cosas") deja de ser parámetro de entrada y pasa a ser RESULTADO de cuántas diferencias se sostuvieron. Cualquier
experimento que empiece fijando N —o que separe en vez de diferenciar— viola la Teoría en el origen.
TENSIÓN TÉCNICA ABIERTA (la trampa Shannon de esta reformulación, a resolver ANTES de codear): un continuo
simulado numéricamente sobre una grilla RE-IMPONE el espacio (la grilla ES una métrica) = Shannon de vuelta. El
reto de diseño es representar "uniforme + diferencia local" SIN grilla espacial previa — con la localidad definida
por contigüidad de gradiente (κ_Δ), no por coordenada. Esto NO lo resuelvo yo solo: es decisión de Teoría, la
confirma el director antes de que CC codee.
NOTA: esto es rediseño de FONDO, no recalibración — CS072 pasa de "el todo sobre un sustrato dado" a "una
diferencia local emergiendo en un continuo uniforme, y el espacio como estructura de esas diferencias".

## EL ARRANQUE, EN TÉRMINOS FÍSICOS (director, 17-jul) — el Big Bang como lo dicta la Teoría.
Hubo una explosión ("Gran Explosión") y se expandió como cualquier explosión. Lo ÚNICO que había era
TEMPERATURA — literalmente toda la energía del Universo, sin quarks, sin gluones, sin partículas. Es UNO porque
preguntado "¿qué eres?" siempre responde lo mismo: temperatura. La expansión inicial NO fue simétrica: un área —o
muchas, no se sabe— tuvo una temperatura DIFERENTE del resto. Esa es la primera diferencia. No es una cosa nueva:
sigue siendo temperatura, la MISMA sustancia, distinguida sólo por ser diferente del todo del que es parte
(vistas en conjunto, muchas áreas diferentes seguían siendo "lo mismo": una diferencia respecto del todo).

**Esto DISUELVE la tensión técnica que yo había planteado ("¿cuál es el dónde antes del espacio?").** Mal
planteada: yo asumía que el "dónde" debe preexistir. NO preexiste. El "área" NO es una coordenada dada — es la
EXTENSIÓN de la propia diferencia: un "área" es, por definición, allí-donde-la-temperatura-difiere. La diferencia
ES el primer lugar. El "al lado de" nace de que dos zonas de temperatura distinta son contiguas — la contigüidad
de la variación, no una casilla en un tablero. Primero la diferencia; el "dónde" es su sombra, leída DESPUÉS.

**Consecuencia dura para el diseño (mata el Shannon de raíz):** el estado inicial de CS072 NO es un conjunto de N
posiciones/nodos con una medida encima. Es UN CAMPO DE TEMPERATURA (energía) y una ASIMETRÍA en su expansión.
Nunca se introduce un conjunto de posiciones; el espacio es lo que se LEE de la estructura de las diferencias de
temperatura, a posteriori. La grilla numérica que yo temía era el error de siempre —asumir el "dónde" antes que
la diferencia—; el director lo invierte, como la Teoría exige.
**TENSIÓN TÉCNICA QUE PERMANECE (honesta, a resolver antes de codear):** simular "un campo de temperatura con
asimetría" en un computador tienta a usar un array indexado = una grilla = una métrica recontrabandeada por la
puerta de atrás (el índice del array ES una coordenada). El reto de implementación es representar temperatura +
diferencia SIN que el sustrato de datos imponga vecindad: la vecindad debe DERIVARSE de la contigüidad de valores
de temperatura (κ_Δ: dos parcelas están "al lado" si su diferencia es suave/continua; separadas si hay salto), no
del layout de memoria. Esto es decisión de Teoría+implementación conjunta — CS lo lleva a CC como restricción
dura (G-DONDE-ES-SOMBRA: ninguna vecindad puede venir del índice del array; sólo de la relación entre temperaturas),
NO como algo que CC resuelva solo.

## LA MAGNITUD DE LA DIFERENCIA (director, 17-jul) — S>0 la fija como CONDICIÓN, no como número a sintonizar.
Formalización del director: el todo es **1** (normalizado — la singularidad, temperatura uniforme). **S>0**
significa que ese 1 no es perfectamente homogéneo: una parte de lo que lo compone —aunque sea la millonésima,
aunque sea infinitesimal— está a temperatura un poco menor. La asimetría basta que sea una **fracción
infinitesimal ε del 1, con ε>0**: ε∈(0,1), y puede ser ε→0⁺.
Lectura de los extremos (por qué es exactamente S>0 y no otra cosa): ε=0 ⇒ el 1 perfecto, homogéneo, sin
distinción ⇒ sin universo (S=0, nada persiste, nada se distingue). ε≥1 ⇒ la "parte" ya no es parte del todo, es
otra cosa ⇒ deja de ser UNO. Sólo 0<ε<1 respeta la singularidad Y la rompe: una parte que sigue siendo del 1,
distinguida por ser un poco menor.
**Por qué esto es anti-Shannon POR CONSTRUCCIÓN (no por guardián añadido):** la magnitud de la semilla deja de
ser un parámetro libre. No se sintoniza — sólo debe cumplir 0<ε<1, y cuanto MÁS PEQUEÑA, MÁS FUERTE la afirmación.
Si con ε=1e-6 emerge estructura direccional, nadie puede alegar que "inyectamos" la dirección: una millonésima NO
puede CONTENER la dirección del universo, sólo puede DESENCADENARLA. La diferencia infinitesimal no aporta
información sobre el resultado; sólo rompe la homogeneidad para que la relación I⟷E empiece a operar. Eso es lo
contrario exacto de hornear la respuesta. → Diseño CS072: barrer ε en escala log (1e-6, 1e-4, 1e-2) y reportar si
el resultado es INVARIANTE a ε (si lo es, la dirección vino de la relación, no del tamaño de la semilla).
**Por qué esto explica que CS070 fallara y esto no es repetirlo:** en CS070 la semilla se puso SOBRE un sustrato
ya construido (mundo-pequeño) y se lavó. Aquí la semilla ES el origen — es S>0 mismo, lo PRIMERO que existe tras
el 1. No hay sustrato previo que la lave porque el sustrato aún no existe; la relación se construye A PARTIR de la
diferencia, no encima de un grafo dado. Es el mismo ingrediente (asimetría primordial, C-N2.5.5) puesto por fin
en el lugar que la Teoría le asigna: el origen, no un añadido tardío.

## LA ASIMETRÍA ES PREMISA, NO OBJETIVO (director, 17-jul) — qué responde CS072 y qué NO.
El director fijó el límite exacto de la pregunta: "todo emergió de esa única o de las muchas diferencias de
temperatura iniciales porque la expansión de la explosión en sí misma no fue simétrica... ¿por qué? NO SÉ: SÓLO
SÉ QUE SI NO HUBIESE SIDO ASÍ NO HABRÍA UNIVERSO."
- **El "por qué hubo asimetría" NO es la pregunta de CS072.** Se toma como CONDICIÓN DE CONTORNO, dada — igual
  que la física estándar toma como dado el exceso materia-antimateria (la asimetría es LA razón de que la materia
  sobreviviera y no se anulara del todo con la antimateria) sin derivar de primeros principios por qué. Exigirle
  al experimento que EXPLIQUE la asimetría obligaría a inventarle una causa = hornear. Prohibido.
- **La pregunta de CS072 SÍ es:** dado que hubo una diferencia (ε>0), ¿EMERGE el espacio / la dirección / lo
  demás, de esa diferencia a través de la relación I⟷E? Eso es falsable y medible. La premisa es "hubo
  diferencia"; el objeto de prueba es "qué emerge de ella", nunca "por qué la hubo".
- **"Una o muchas es irrelevante" → regla de diseño (no parámetro a sintonizar):** el número de focos de asimetría
  es una MOLESTIA de la que hay que demostrar INDEPENDENCIA. Correr con 1, con pocas, con muchas, y reportar que
  el resultado NO depende del número. Si dependiera, se habría metido información por la cantidad; si no depende,
  se confirma que lo único relevante era ε>0. (Junto con el barrido de ε: dos invarianzas que blindan el anti-
  Shannon — ni el tamaño ni el número de la semilla deben cargar el resultado; sólo el hecho de que exista.)

## PRECISIÓN CANÓNICA (director, 17-jul) — S = I⟷E es RELACIÓN, no multiplicación.
La forma es **S = I⟷E**: información y energía en acoplamiento bidireccional que se constituyen mutuamente — NO
S = I·E (producto). La diferencia es de fondo, no notacional: un producto colapsa I y E en un solo número,
simétrico y sin memoria de que hubo dos términos (si uno es 0, todo es 0); una RELACIÓN conserva ambos polos y su
vínculo, y S ES ese vínculo, no el resultado de operarlos. Esto explica ESTRUCTURALMENTE por qué el motor dio (B)
seis veces: multiplicaba/muestreaba MAGNITUDES escalares (pesos, correlaciones, temperaturas) — nunca representó
la RELACIÓN I⟷E. La diferencia primordial no es "I por E" en un punto: es I y E en intercambio mutuo, y es ESE
intercambio lo que se distingue del uniforme. Consecuencia para CS072: el estado no puede ser un campo escalar de
"cantidad de S" por sitio; tiene que llevar el acoplamiento I⟷E como relación, o se re-cae en el producto. (Los
registros históricos del arco que escriben "S=I·E" NO se reescriben — son fechados; esta precisión rige de aquí en
adelante.)

## NOTA de honestidad (mía, otra vez)
Este Shannon de fondo estuvo en TODOS los experimentos desde CS064 y no lo vi hasta que Alexis lo nombró — yo
había afirmado (mal, desde un docstring) que el motor "arrancaba sin estructura". Arrancaba de la medida de máxima
entropía, que es lo contrario de neutral. El hallazgo es del director; yo sólo lo confirmé con el código y una
demostración. Queda registrado así.

## PARA CC (dos pasos, en orden)
1. Aplica Opción 1 (normalización por mediana) para que el motor NO colapse — es prerequisito técnico, no el
   experimento.
2. Implementa el brazo TEST (GAS + afinidad intrínseca, G-SIN-MEDIDA-PREVIA) junto al brazo BASE. Ese contraste
   ES CS072. Si la afinidad-primero es O(N²) prohibitiva, repórtalo y decidimos una versión acotada ANTES de la
   tanda — nunca a mitad.
NO corras la tanda de veredicto hasta que ambos brazos arranquen sanos en exploratoria y me lo reportes.

## EN UNA LÍNEA
El colapso de CC se arregla normalizando el enfriamiento por la mediana del paso (una sola T, sin segundo reloj).
Pero el hallazgo grande es de Alexis: el "azar uniforme" del arranque es la medida de máxima entropía, la única
que garantiza mundo-pequeño, y las fuerzas grafo-locales sólo la heredan — por eso el espacio nunca pudo emerger
en seis experimentos. CS072 se rediseña para arrancar SIN medida previa: gas de nodos + enlaces por afinidad
intrínseca, con G-SIN-MEDIDA-PREVIA. Es la primera vez que el todo se enciende sin el Shannon escondido en el
punto de partida.

— CS 🐝
