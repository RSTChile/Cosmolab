# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCCIÓN v3 (CC) — ESCALAR + ANCLAR A NÚMEROS FÍSICOS REALES (Alexis, 19-jul)
# ═══════════════════════════════════════════════════════════════════════════════
# CONFIRMADO: CC escala. Discriminante SIGUE siendo REAL vs NULL a cada escala (no "cruzó Jeans en
# absoluto" — ese fue el artefacto retractado). z=10.26 ya ganó a f=5; escalar prueba si el MISMO
# mecanismo alcanza el umbral ABSOLUTO de Jeans, no fabrica señal.
#
# LO QUE IMPORTA A ALEXIS — blindar contra "es que usaron simulaciones discretas":
# NO basta con "cruzó Jeans a N grande". Hay que ANCLAR el discreto a escalas FÍSICAS REALES por
# ORDEN DE MAGNITUD (exponenciales, no cálculo full — la máquina no da para 10⁶⁰ partículas). Plan:
# 1. Correr una SERIE de escalas (p.ej. N = 250, 500, 1000, 2000, 4000...) — todas REAL vs NULL.
# 2. MEDIR la ley de escala del discriminante: cómo crece con N (a) el z-score REAL/NULL, (b) la razón
#    masa_cluster / M_J_local (cuán cerca de cruzar). Ajustar EXPONENTE (potencia/log), NO forzar.
# 3. EXTRAPOLAR por potencias hacia el N físico de una nube de Jeans real Pop III (~10⁶²-10⁶³ átomos de H,
#    M_J~10⁵-10⁶ M☉). Si la ley medida predice masa_cluster/M_J ≥ 1 en el N físico → el discreto es un
#    MUESTREO GRUESO de un continuo que SÍ enciende la estrella; la discretización es resolución, no
#    artefacto. Argumento de convergencia, como en cualquier simulación N-cuerpos cosmológica seria.
# 4. GUARDIA ANTI-SHANNON: la ley de escala se MIDE de la serie, NO se ajusta para que cruce. Reportar
#    el exponente con su error y el R² del ajuste. Si la extrapolación NO cruza, es un dato honesto
#    (mecanismo real pero subcrítico incluso en el límite) — no se retoca.
# 5. Anclar unidades por orden de magnitud: mapear el N discreto y las masas adimensionales a M☉ / años
#    luz vía las relaciones de Jeans (sólo para el EJE de lectura, no para la física; declararlo).
#
# Entregable: figura z(N) y (masa/M_J)(N) en log-log con la ley ajustada + la extrapolación al N físico
# marcada. Eso es lo que desactiva la crítica "simulación discreta" por adelantado.
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# ADICIÓN FORMAL v2 — lo que FALTA al basal: la ASIMETRÍA DE DISTRIBUCIÓN
# (reemplaza la hipótesis W_ij, que CC probó ausente del motor. Ver más abajo el diseño original.)
# ═══════════════════════════════════════════════════════════════════════════════

## Qué falta, con precisión (tres caminos convergentes lo señalan)
El motor genera asimetría de CANTIDAD (#23 = fluctuaciones cuánticas, distribución lognormal marginal)
pero NO asimetría de DISTRIBUCIÓN (correlación espacial: qué regiones vecinas son densas juntas).
Malla causal (Paso A), N-cuerpos y ausencia de W_ij: los tres chocan con el mismo muro. NO es problema
de medición — el ingrediente está AUSENTE del sustrato. Hay que añadirlo formalmente y correr de nuevo.

## El mecanismo NO se inventa — descansa en una premisa de Alexis (anti-Shannon)
**Cita verbatim de Alexis (premisa del gradiente térmico):** "La expansión ocurrió tan rápido que el
sistema no tuvo tiempo de volver a homogeneizarse; las diferencias quedaron preservadas y aumentaron."
**Encuadre de CS (síntesis, NO cita de Alexis):** leo esa premisa como la distinción cantidad vs
distribución — el mecanismo físico es la EXPANSIÓN SUPERLUMÍNICA CONGELANDO las correlaciones del
gradiente térmico antes de que se homogeneicen (horizonte causal). El motor congela CANTIDAD pero nunca
aplicó ese congelamiento a un CAMPO ESPACIALMENTE CORRELACIONADO. Eso es lo que se añade.
**Nota:** el fondo rugoso NO es un hecho asentado a invocar, es justo lo que este experimento debe hacer
EMERGER y verificar vs NULL (regla de Alexis: "el fondo rugoso debe surgir o no se muestra"). Si la
correlación no emerge del congelamiento, no se pinta.

## CORRECCIÓN de ubicación (CC, verificado en código — mi error): el mecanismo es POST-átomo
CS escribió "en el basal, pre-átomo, sobre un gradiente térmico". ERROR: el mecanismo de dos fases
(horizonte causal) vive en `_malla_causal` (proceso_sucesivo.py) y opera DESPUÉS del átomo, sobre las
densidades de los átomos ya formados. NO existe versión pre-átomo en catalogo.py/estado.py — pedirla
sería INVENTAR física nueva (prohibido). Ubicación corregida abajo.

## Módulo del puente — malla causal como SEMILLA DINÁMICA (reusa código validado, no inventa)
1. Átomos reales YA formados (como siempre).
2. Correr `_malla_causal` sobre sus densidades reales (el MISMO código de dimension_acoplada) → grafo de
   qué átomos quedaron causalmente cerca bajo el horizonte de las dos fases.
3. Usar ESE grafo como SEMILLA DINÁMICA de posición (layout por resortes: pares conectados arrancan
   próximos), NO un MDS estático de golpe. ← distinción real con Paso A: la malla causal como semilla que
   la expansión DESPLIEGA, no como fotografía final.
4. Desde ahí, los 4 módulos ya validados (expansión, gravedad general, CDM, H2) evolucionan el bucle único.

Tercer camino genuino: ni el umbral térmico de Bgrav, ni el MDS estático del Paso A. No toca nada
congelado (catalogo.py/estado.py/nucleo.py intactos).

**La correlación espacial DEBE EMERGER del despliegue por expansión de la semilla causal** (ritmo de dos
fases YA fijo en el motor), NO ajustarse. Si hay que tunear el ritmo para que salga correlación = Shannon.

## Correr COMPLETO de nuevo (regla holística de Alexis)
Basal + p_gradiente_correlacionado + los 4 módulos de cierre (gravedad general, expansión, H2, CDM),
TODO operando simultáneamente en un bucle, ventana S>0 → átomo → estructura. Primero el todo; si falla,
se aísla el módulo. NO por partes.

## NULL y observable (pre-registrados)
- **NULL = conexiones de la malla causal barajadas** (preservando distribución de grado/peso) → destruye
  la coherencia relacional del grafo preservando su distribución marginal. (Corrección de CC: "fases
  aleatorizadas" era lenguaje de Fourier que no encaja con un grafo; el barajado de aristas es el NULL
  correcto en este dominio, el mismo tipo del resto del arco.) G-DIFERENCIA-INTERNA: el grafo desordenado.
- **Observable:** ¿nacen estructuras múltiples y separadas que cruzan Jeans, MÁS que en el NULL (z-score,
  ≥5 semillas × ≥8 NULL)? ¿El campo desplegado tiene P(k) con potencia a gran escala que el NULL no tiene?

## Tres resultados pre-inscritos
- **(A) POSITIVO:** el congelamiento causal genera correlación espacial emergente; REAL gana al NULL;
  nacen estructuras. → EL SUSTRATO SÍ CARGA DISTRIBUCIÓN cuando se incluye el mecanismo de congelamiento.
  El experimento estaba incompleto, no el sustrato. Cierra Cosmogénesis en positivo.
- **(B) NEGATIVO:** el congelamiento barajado da lo mismo → ni con el mecanismo formal emerge distribución.
  → CUARTO camino: el sustrato no carga correlación espacial ni con el congelamiento causal. Cierre robusto.
- **(C) PARCIAL:** correlación emerge pero no basta para Jeans → mecanismo real pero débil; falta escala o
  un ingrediente adicional. Dato fino.

## Guardianes
G-DIFERENCIA-INTERNA (NULL = fase barajada). G-SIN-SIEMBRA (correlación EMERGE del congelamiento, no
pintada). G-SIN-ENERGIA-NUEVA. G-EXPANSION-ISOTROPA (el congelamiento no impone dirección/rejilla).
G-PARAMETROS-ESTRUCTURALES (P(k) emerge del ritmo de dos fases YA fijo, no tuneado). G-CORRELACION-EMERGE
-NO-PINTADA (el espectro sale del proceso causal, jamás asignado a mano).

## Costo
O(N²). Escala grande → entorno de CC / segundo plano, no kernel de CS. Motor CONGELADO hasta acuerdo.

---
# DISEÑO ORIGINAL v1 (hipótesis W_ij — SUPERADA: CC probó que W_ij no existe en el motor. Se conserva
# como registro del razonamiento. La adición formal v2 de arriba es la vigente.)

# DISEÑO — Experimento PUENTE (CS073): del resultado cuántico a la primera estrella
**De:** CS (diseño + adjudicación). **Para:** CC (implementación + corrida). Regla del pacto: CC corre,
no rediseña; un desacuerdo es dato a coordinar. Nota permanente: no se cierra hasta que Alexis lo diga.

## Dónde estamos (las dos piezas ya probadas)
1. **La maquinaria funciona** (control positivo ✓): dada una nube favorable, gravedad+expansión+
   enfriamiento la colapsan en estructura ligada. Gravedad es la causa (G=0 → no liga).
2. **El sustrato #23 da CANTIDAD, no DISTRIBUCIÓN ESPACIAL** (negativos convergentes): dos mecanismos
   independientes (malla causal Paso A + N-cuerpos) chocaron con el mismo muro. REAL ≈ NULL.

## La hipótesis del puente (intuición de Alexis, precisada por CS)
"Hay algún elemento o propiedad que no estamos considerando, o que no está actuando sobre el resultado
cuántico que ya tenemos." **CANDIDATO CONCRETO:** el motor produce átomos con un GRAFO DE CORRELACIONES
W_ij (qué distinciones están correlacionadas con cuáles) — coherencia RELACIONAL. Cuando CS asignó
posiciones 3D INDEPENDIENTES de la densidad (defecto Q3), tiró W_ij junto con la densidad. Medimos la
coherencia de distribución MARGINAL (negativa), pero **NUNCA probamos si la coherencia RELACIONAL (W_ij),
desplegada por la expansión, produce correlación ESPACIAL.** Ese es el elemento que no está actuando.

## Distinción CRÍTICA con el Paso A (que ya falló — para no repetir)
- **Paso A (falló):** embedding GEOMÉTRICO ESTÁTICO — MDS sobre distancias del grafo → posiciones. Negativo
  sólido (z<1). El embedding estático no extrajo geometría 3D del grafo relacional.
- **Puente (nuevo):** despliegue DINÁMICO por EXPANSIÓN — no se embebe el grafo de golpe; se deja que la
  expansión ESTIRE las correlaciones en el tiempo (átomos correlacionados en W_ij arrancan cerca y la
  expansión los separa a un ritmo que PRESERVA la correlación como estructura espacial de gran escala).
  Es la idea de Alexis: la expansión convierte la métrica cuántica (correlación relacional, adimensional)
  en espacio MACROSCÓPICO dimensionable. Proceso, no fotografía.

## El experimento (motor real, CC)
Sobre el motor basal (S>0 → átomos con densidad #23 Y grafo W_ij):
1. **Semilla de posición desde W_ij (no uniforme, no MDS):** layout dirigido por fuerzas donde el peso de
   arista = W_ij real. Átomos muy correlacionados arrancan próximos. SIN coordenadas plantadas a mano.
2. **Expansión despliega:** el mismo p_expansion, mismo reloj T(t). La expansión separa; las correlaciones
   fuertes resisten la separación (quedan como sobredensidades de gran escala EMERGENTES).
3. **Gravedad+presión+H2+CDM** (la maquinaria ya validada) actúan sobre ese campo desplegado.
4. **Observable:** ¿nacen estructuras MÚLTIPLES Y SEPARADAS que superan Jeans, MÁS que en el NULL?

## NULL y discriminante (pre-registrados, anti-Shannon)
- **NULL = W_ij BARAJADO** (mismas correlaciones, reasignadas al azar entre pares). Destruye la coherencia
  relacional preservando su distribución marginal. G-DIFERENCIA-INTERNA: el campo consigo mismo, desordenado.
- **Discriminante pre-registrado:** nº de estructuras ligadas separadas que cruzan Jeans, REAL vs NULL,
  z-score sobre ≥5 semillas × ≥8 NULL. Y espectro P(k) del campo desplegado: ¿tiene potencia a gran
  escala (correlación) que el NULL no tiene?
- **El espectro P(k) debe EMERGER del despliegue, NO ajustarse.** Si hay que tunear el ritmo de expansión
  para que salga correlación = Shannon. El ritmo de expansión es el mismo del resto del motor, fijo.

## Guardianes
- G-DIFERENCIA-INTERNA (NULL = W_ij barajado). G-SIN-SIEMBRA (posiciones nacen de W_ij, no plantadas).
- G-SIN-ENERGIA-NUEVA (M_J sólo de T y ρ). G-EXPANSION-ISOTROPA (expansión no impone dirección/rejilla).
- G-PARAMETROS-ESTRUCTURALES (adimensional; ritmo de expansión heredado, no tuneado).
- G-DESPLIEGUE-NO-EMBEDDING (el puente es dinámica temporal, no MDS de golpe — distinto del Paso A).

## Los tres resultados posibles (todos informativos, pre-inscritos)
- **(A) POSITIVO:** el despliegue de W_ij por expansión produce campo espacialmente correlacionado; REAL
  gana al NULL; nacen estructuras múltiples. → EL PUENTE EXISTE: la coherencia relacional del sustrato,
  desplegada por expansión, ES la semilla de la estructura. Cierra Cosmogénesis en positivo.
- **(B) NEGATIVO:** W_ij barajado da lo mismo → la coherencia relacional TAMPOCO tiene estructura espacial
  extraíble. → confirma por TERCER camino independiente que el sustrato da cantidad, no distribución.
  El puente requiere un ingrediente que el sustrato no tiene (cierre en negativo, robusto).
- **(C) PARCIAL:** correlación espacial emerge pero no basta para cruzar Jeans → hay coherencia relacional
  real pero débil; el puente existe pero incompleto. Dato fino sobre qué falta.

## Costo
O(N²) por el layout y la gravedad. Escala grande → entorno de CC / segundo plano, no kernel de CS.
Motor CONGELADO hasta acuerdo.