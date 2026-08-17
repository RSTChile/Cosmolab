# DISEÑO CS073 (cierre) — Transición Átomos → Masa → Gravedad General → Primera Estrella

**De:** Claude Science (CS) — diseño para que CC lo implemente en el motor real y lo corra a escala.
**Fecha:** 19-jul-2026. **Regla:** toca física → CC implementa y corre; CS no toca el motor.
Este es el experimento que CIERRA Cosmogénesis de verdad (con acuerdo de Alexis para cerrar).

---

## Por qué es un experimento nuevo (no "más de lo mismo a más escala")
Ya está establecido y medido:
- El motor basal (S>0 → átomos) está validado.
- La gravedad que el motor tiene HOY (`Bgrav`) es **relacional-cuántica**: umbral de proximidad
  térmica → hub de densidad de red **0.500 INVARIANTE** en 4 escalas (23→213 átomos). Es hub por
  construcción, escalarla no da estructura. Y es CORRECTO para su régimen (pre-masa).
- **Nunca corrimos el régimen de gravedad GENERAL-clásica** (masa-atrae-masa), porque no opera antes
  de acumular masa/densidad — un régimen posterior que el motor no modela.

La estructura (y con ella distancia/dirección clásicas medibles) es un fenómeno del régimen de
gravedad general. Modelar ese régimen ES el experimento de cierre.

## Diferencia física clave — las dos gravedades
| | relacional-cuántica (HOY, `Bgrav`) | general-clásica (A CONSTRUIR) |
|---|---|---|
| régimen | pre-métrico, sin posición | métrico, masa con posición |
| regla | UMBRAL binario (proximidad térmica) | FUERZA continua ∝ m_i·m_j que ACUMULA |
| con escala | hub invariante (0.500) | debe transicionar a estructura |
| activación | siempre | sólo con masa/densidad suficientes |

El punto: la gravedad general **no es un umbral, es una dinámica de acumulación** — la masa concentra
masa, la densidad crece localmente, y donde supera la masa de Jeans → colapsa → primera estrella.

## PRINCIPIO RECTOR (Alexis) — TODAS las fuerzas simultáneas, o no resulta
**No funciona por partes porque nunca fue por partes.** La primera estrella NO emerge de la densidad,
ni de la gravedad, ni de la expansión, ni del enfriamiento por turnos — emerge de las cuatro TENSÁNDOSE
ENTRE SÍ, AL MISMO TIEMPO, en la ventana entre el primer átomo y la primera estrella. La expansión
separa mientras la gravedad junta mientras el enfriamiento baja M_J mientras las semillas (fluct.
cuánticas #23) se amplifican — esa COMPETENCIA SIMULTÁNEA es la que fragmenta el gas en estructuras
separadas en vez de un blob. Sacar una pieza rompe el equilibrio que produce el fenómeno.

**Prueba negativa acumulada (4 prototipos aislados de CS, todos fallan por aislar):** densidad sola →
un grumo; red Bgrav sola → hub invariante; masa sola → ρ=T, Jeans imposible; expansión sola → semillas
demasiado suaves (δ_rms 0.23 « δ_c 1.686). Cada pieza aislada falla; ninguna es refutación de la física
— son refutación del método reduccionista. **CONSECUENCIA DE MÉTODO: ningún prototipo aislado (ni de CS
ni en sandbox) puede dar este veredicto. Sólo el motor completo con todo operando junto.**

## El experimento (para el motor real, a cargo de CC) — HOLÍSTICO, no por barridos de una variable
**[CORREGIDO tras la observación de CC — 19-jul: el Paso A (posiciones) es prerrequisito EXPLÍCITO,
NO proximidad térmica.]** CC cazó una contradicción real en la v1 de este diseño: yo había escrito
"usar la localidad térmica que el motor ya tiene", pero el prototipo #2 YA falsificó eso (REAL=NULL —
un escalar 1D no codifica vecindad 3D; una fuerza continua sobre proximidad térmica reproduce el mismo
REAL=NULL, sólo más caro). La corrección es de fondo: **la gravedad GENERAL opera en el régimen
MÉTRICO, sobre POSICIONES 3D reales — nunca sobre proximidad térmica (que es la gravedad RELACIONAL,
régimen pre-métrico).** Desplegar posiciones ES lo que distingue el régimen general del relacional.

1. **Correr el motor basal completo** (S>0 → átomos) a **escala grande** (con números pequeños NO
   opera, es obvio). Los átomos ya tienen masa real (`masa_trio`) y densidad real (#23) en el estado.
2. **PASO A — desplegar la métrica fosilizada como POSICIONES 3D** (prerrequisito, no opcional). Fuente
   = **distancias de grafo de la malla causal** (`_malla_causal`, la que escapa del mundo-pequeño —
   NO `Bgrav`, que es hub), embebidas a coordenadas (MDS/landmark-MDS por costo). Gate de despliegue =
   `dimension_acoplada` finito (el motor ya define ahí "dejó de ser mundo-pequeño"). SIN posiciones no
   hay "vecindad 3D" → no hay gravedad general posible. Este paso NO estaba en la v1; es el que CC exigió.
3. **PASO B — gravedad general sobre las POSICIONES**: fuerza continua de acumulación ∝ m_i·m_j entre
   átomos **espacialmente cercanos EN 3D** (no en temperatura), que crece con el tiempo. Es un régimen
   NUEVO, no `Bgrav`.
4. **Enfriamiento → masa de Jeans cae** (M_J ∝ T^(3/2)/√ρ; el motor ya enfría). Donde la masa local
   3D supera M_J → colapso → estructura ligada. La supernube se fragmenta al caer M_J (jerarquía).
5. **Observable de cierre:** ¿emerge la PRIMERA ESTRUCTURA LIGADA (proto-estrella)? ¿La transición
   hub→estructura aparece al escalar, donde la gravedad relacional daba hub invariante?

**REGLA DE SIMULTANEIDAD (crítica):** los pasos 2-4 NO se corren por separado ni en secuencia — son
UN solo bucle temporal donde en CADA paso de tiempo actúan a la vez: expansión (separa+diluye),
gravedad general (concentra ∝m·m sobre posiciones), enfriamiento (baja M_J), y las semillas #23
amplificándose. El colapso local es el resultado de la COMPETENCIA entre expansión y gravedad en cada
instante. Correr una pieza y luego otra = volver al error de los 4 prototipos aislados. El veredicto
sólo cuenta si TODO opera junto en la ventana átomo→estrella.

**Nota sobre la salvedad A.4 (marco físico de los ejes):** el despliegue debe usar coordenadas con
referente físico, no D barajadas del mismo escalar (eso era el hueco CG004/CG005). La resolución
conceptual acordada: los ejes/posiciones NACEN con el colapso (decoherencia) — la métrica de la malla
causal da la semilla de distancias, el colapso gravitacional le da posición clásica. A verificar en
la implementación: que las coordenadas desplegadas NO sean el `_ejes_independientes` barajado.

## Guardianes anti-Shannon (todos heredados del arco)
- **G-DIFERENCIA-INTERNA:** toda diferencia es el campo consigo mismo. NULL = campo #23 real barajado
  (misma distribución, coherencia destruida), enfriamiento idéntico.
- **G-SIN-ENERGIA-NUEVA:** M_J depende sólo de T y ρ, herencia de la Singularidad. Si el motor
  necesitara inyectar energía para colapsar, es Shannon.
- **G-DOS-GRAVEDADES:** la gravedad general es un régimen NUEVO, no un reajuste de `Bgrav`. No mezclar.
  La relacional queda para el régimen pre-masa; la general para el post-masa.
- **G-SIN-SIEMBRA:** cero centros de colapso impuestos; emergen de las sobredensidades #23 reales.
- **G-UMBRAL-FISICO:** el colapso lo decide δ_c=1.686 / M_J, no un umbral a ojo.
- **G-PARAMETROS-ESTRUCTURALES:** los parámetros (índice espectral, tasa de enfriamiento, etc.) se
  derivan de la física del motor o se BARREN exigiendo que la señal sobreviva — no se fijan a ojo
  (`no_parametros_solo_estructurales`).
- **G-DISCRIMINANTE-PRE-REGISTRADO:** fijar el observable de estructura ANTES de correr (candidato:
  transición hub→estructura con escala, o nacimiento de la 1ª región Jeans-inestable ligada).

## Discriminante de veredicto (pre-registrar antes de correr)
La firma inequívoca de que la gravedad general opera y la relacional no:
- **relacional (`Bgrav`)**: densidad de red 0.500 invariante con N (hub) — YA medido.
- **general-clásica**: la topología DEBE cambiar con N — fragmentar en estructuras ligadas discretas,
  con al menos una que supere el umbral de Jeans (proto-estrella). Si NO cambia con N → la gravedad
  general tampoco forma estructura y el resultado es negativo honesto.

## Estado y coste
Diseño de CS. Requiere: (a) CC implemente el régimen de gravedad general en el motor (pieza nueva
`p02b_gravedad_general` o extensión), (b) corra el basal a escala grande (O(N²) hoy — evaluar costo /
optimización / corrida en segundo plano). Motor CONGELADO hasta acuerdo. Cierra el experimento cuando
Alexis lo diga (nota permanente vigente).
