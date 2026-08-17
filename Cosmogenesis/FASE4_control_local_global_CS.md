# CS083b — Fase IV: afinando el 92%/8% con dos controles nuevos (local-roto y global)

Sigue directo a `FASE4_orden_superior_resultado_CS.md` (Fase IV original) y
`FASE4_robustecido_CS.md` (el 92%/8%). Script: `cs083b_fase4_control_local_global.py`. Datos crudos:
`cs083b_resultados.csv` (20 semillas completas) y `cs083b_resultados_piloto.csv` (piloto de 5 semillas
usado para verificar que el código corría antes de escalar). No toca `cs082_fase4_4sustratos.py` ni
`cs083_fase4_robustecer.py` — los importa tal cual.

No hay veredicto ni cierre acá. Son números. La lectura final es de Alexis.

---

## 1. Qué se le pidió a este script

El sustrato 4 (2-complejo con retroalimentación cara→arista: cada triángulo "empuja" a sus 3 aristas de
borde para achicar su propia holonomía) achica la holonomía ~5x respecto de NULL. El informe robustecido
(`FASE4_robustecido_CS.md`) usó UN control fino — "rewire al azar" — para descomponer esa caída en
~92% "consenso global" (el mero volumen de retroalimentación, sin importar dónde cae) y ~8% "cierre
local genuino" (z=-4.14, p≈0.0006 — que caiga exactamente en el triángulo correcto).

El equipo pidió afinar ese resultado con DOS controles más específicos, comparados entre sí:

- **NULL-LOCAL-ROTO**: ¿importa la correspondencia geométrica EXACTA (mi triángulo empuja MIS 3
  aristas), o alcanza con que la retroalimentación esté organizada en tríos coherentes de 3 aristas
  (aunque sean las de OTRO triángulo)?
- **NULL-GLOBAL**: ¿importa que la retroalimentación esté concentrada en tríos de 3 aristas a la vez
  (aunque sean los tríos equivocados), o el efecto sobrevive igual si se DILUYE parejo sobre TODAS las
  aristas del grafo, sin ningún trío?

## 2. Cómo se construyeron los 2 controles nuevos (fórmulas concretas)

Los 4 brazos comparten TODO lo demás: mismo N=110, mismo grafo base por semilla, mismo K=6, J=0.6,
J_FACE=0.5, ruido=0.25, mismo presupuesto de cómputo (110 sweeps, DoF=706), 20 semillas (1-20, las
mismas que cs083). El control de equiparación quedó verificado: mismos sweeps en los 4 brazos, las 20
semillas.

### NULL-LOCAL-ROTO ("el trío correcto de forma, pero de OTRO triángulo")

En el sustrato REAL, cada triángulo T calcula su propio defecto de holonomía h_T a partir de SUS 3
aristas, y empuja esas mismas 3 aristas. En NULL-LOCAL-ROTO, T sigue calculando h_T con SUS propias 3
aristas (exactamente igual que el real — "la retroalimentación que le corresponde a esa cara" no
cambia), pero el empujón resultante se REDIRIGE a las 3 aristas de OTRO triángulo real T' del mismo
grafo (T'≠T siempre, por una permutación sin puntos fijos —"derangement"— sorteada una vez al arrancar
la corrida y fija durante toda ella).

La diferencia con el control "rewire al azar" ya existente (cs083): ahí las 3 aristas de destino se
sortean sueltas de CUALQUIER parte del grafo (típicamente ni siquiera comparten un nodo entre sí). Acá
el trío-destino sigue siendo un trío GENUINO — 3 aristas que sí forman un triángulo real en otra parte
del grafo — sólo que es el triángulo equivocado.

**Honestidad sobre la aproximación**: esto NO es un bootstrap de grado formalmente controlado. No
garantiza que el triángulo T' tenga, nodo a nodo, los mismos 3 grados que T. Es una redirección a otro
trío real de la MISMA población ("tríos de arista que sí forman un triángulo genuino en este grafo"),
lo cual en espíritu se parece más al trío original que 3 aristas sueltas sin relación entre sí — pero
no es una garantía exacta de "mismo grado de nodos involucrados" como pedía la tarea al pie de la
letra.

### NULL-GLOBAL ("sin ningún trío: toda arista, cada sweep, hacia el promedio del campo")

En vez de que cada cara empuje 3 aristas (las suyas o las de otro triángulo), acá se calcula en cada
sweep la media circular de TODO el campo de aristas (μ), y CADA una de las ~545 aristas del grafo recibe
el mismo tipo de empujón —con la MISMA constante de fuerza por evento (J_FACE/3) que usa cada arista en
el sustrato real— pero dirigido hacia μ (el consenso global) en vez de hacia el defecto de SU propio
triángulo. No hay ningún trío en este mecanismo: es la versión más diluida posible del mismo tipo de
corrección.

**Honestidad sobre la aproximación**: se preservó la constante de fuerza por evento (J_FACE/3, idéntica
a la del sustrato real), pero NO se intentó igualar el conteo exacto evento-por-evento del sustrato real
(esto habría exigido acoplar paso a paso las dos simulaciones, lo que mezclaría sus trayectorias
estocásticas y volvería la comparación menos limpia). En la práctica, NULL-GLOBAL toca ~545
aristas/sweep (todas) contra ~483 eventos/sweep del sustrato real (3×n_tri, concentrados en tríos) — un
volumen total de "toques" del mismo orden de magnitud, reportado explícitamente en la tabla de control
para que sea auditable.

## 3. Resultado de los 4 brazos (20 semillas)

Holonomía promedio |h| (más bajo = más "aplanado"; NULL≈ruido puro ≈1.52 es el techo de referencia):

| brazo | h media | DE | vs NULL (ruido) |
|---|---|---|---|
| **REAL** (sustrato 4 sin cambios) | 0.264 | 0.107 | — |
| NULL-REWIRE (control ya existente, cs083) | 0.368 | 0.025 | muy por debajo de NULL |
| **NULL-LOCAL-ROTO** (nuevo) | 0.506 | 0.037 | por debajo de NULL |
| **NULL-GLOBAL** (nuevo) | 1.528 | 0.865 | prácticamente = NULL |
| NULL (ruido puro, referencia) | 1.517 | 0.082 | — |
| SHUFFLED (referencia cs082) | 0.486 | 0.359 | por debajo de NULL |

Tests pareados por semilla (n=20, z-score + permutación sign-flip, 20 000 repeticiones):

| comparación | z | diff. observada | p (una cola) |
|---|---|---|---|
| REAL vs NULL-REWIRE | −4.14 | −0.105 | 0.0004 |
| REAL vs NULL-LOCAL-ROTO | −9.43 | −0.243 | <0.0001 |
| REAL vs NULL-GLOBAL | −6.55 | −1.265 | <0.0001 |
| **NULL-LOCAL-ROTO vs NULL-GLOBAL** | **−5.27** | **−1.022** | **0.0001** |
| NULL-REWIRE vs NULL-LOCAL-ROTO | −15.62 | −0.138 | <0.0001 |
| NULL-REWIRE vs NULL-GLOBAL | −5.96 | −1.160 | 0.00005 |
| NULL-LOCAL-ROTO vs NULL (ruido) | −55.94 | −1.011 | <0.0001 |
| **NULL-GLOBAL vs NULL (ruido)** | **+0.06** | **+0.011** | **0.52 (NO significativo)** |

Descomposición (misma lectura que cs083 — "cuánto del aplanamiento total NULL−REAL se pierde con cada
control", sobre gap_total = h_NULL − h_REAL = 1.253):

| control | h_control − h_REAL | fracción del gap "perdida" | fracción que SOBREVIVE de REAL |
|---|---|---|---|
| NULL-REWIRE (referencia cs083) | +0.105 | 8.3% | 91.7% |
| NULL-LOCAL-ROTO (nuevo) | +0.243 | 19.3% | 80.7% |
| NULL-GLOBAL (nuevo) | +1.265 | 100.9% | ≈0% (no significativo) |

## 4. Lectura honesta (sin cerrar el experimento)

Tres hallazgos, sin forzar ninguna interpretación:

**(a) NULL-GLOBAL es indistinguible de ruido puro.** Diluir la MISMA constante de fuerza sobre TODAS
las aristas, sin ningún trío, dio z=+0.06 contra NULL (p=0.52) — cero aplanamiento medible. Esto es
sorprendente respecto de cómo se venía leyendo el "92% consenso global" en `FASE4_robustecido_CS.md`:
si el efecto fuera realmente "cualquier volumen de retroalimentación disperso empuja todo el campo hacia
consenso", una versión maximalmente dispersa (ésta) debería aplanar TAMBIÉN, aunque sea un poco. No lo
hizo. Esto sugiere que lo que el control "rewire al azar" de cs083 estaba capturando bajo el nombre de
"consenso global" no era en realidad difusión pareja sobre el grafo — era la CONCENTRACIÓN en tríos de
3 aristas (aunque desordenados), no la dispersión total.

**(b) La concentración en tríos —incluso mal cableados— explica la mayor parte del aplanamiento.**
NULL-REWIRE (tríos de 3 aristas sueltas, sin relación geométrica entre sí) preserva 91.7% del
aplanamiento de REAL. NULL-LOCAL-ROTO (tríos coherentes, pero del triángulo equivocado) preserva 80.7%.
Ambos números están muy por encima de lo que aportaría NULL-GLOBAL (≈0%). O sea: "empujar de a tríos"
importa mucho más que "empujar sobre todo el grafo parejo" — independientemente de si el trío es
geométricamente correcto.

**(c) Contraintuitivo: el trío "geométricamente parecido pero equivocado" (LOCAL-ROTO) aplana MENOS
que el trío "totalmente suelto" (REWIRE)** — z=−15.62, muy significativo, y en la dirección opuesta a
lo que se podría esperar si "parecerse más a un triángulo real" ayudara a imitar al REAL. Una lectura
posible (no comprobada acá, requeriría un experimento aparte): REWIRE sortea sus 3 aristas de destino
de MANERA INDEPENDIENTE para cada triángulo sobre TODO el conjunto de aristas, lo que en la práctica
termina tocando casi todas las ~545 aristas del grafo al menos una vez (cobertura amplia). LOCAL-ROTO,
en cambio, redirige por ÍNDICE DE TRIÁNGULO (una permutación de los ~161 triángulos), y como las aristas
que pertenecen a MUCHOS triángulos (zonas densas del grafo) tienen más chance de ser el destino de una
redirección, el patrón de "quién recibe empuje" queda más concentrado en un subconjunto de aristas
—posiblemente dejando aristas "puente"/periféricas sin tocar— que en REWIRE. Si esa lectura es correcta,
lo que separaría a REAL de sus controles no sería sólo "trío correcto sí/no" sino también "qué tan
pareja es la cobertura de aristas que toca la retroalimentación" — una variable que este diseño no
aisló limpiamente y que Alexis puede decidir perseguir o no.

**¿Esto cambia el 92%/8% ya reportado?** No lo reemplaza — lo AFINA en una dirección específica: el
"8% local genuino" de cs083 sigue ahí (reproducido casi exactamente: 8.3% en este re-run de 20 semillas
independientes, muy cerca del 8% original). Pero el "92%" que antes se llamaba en bloque "consenso
global" ahora se ve como dos cosas distintas de magnitud muy distinta: (i) concentración-en-tríos
(aunque errados) ≈ 80-92% del efecto, y (ii) dispersión pura sin trío ≈ 0% del efecto. La etiqueta
"consenso global" del informe anterior probablemente describía mejor a (i) que a una difusión literal
sobre todo el grafo — que es justamente lo que NULL-GLOBAL prueba por separado acá, y que no aplana
nada.

## 5. En simple, con analogía

Pensemos en el sustrato 4 como una ronda de gente en círculo tratando de ponerse de acuerdo en qué
mano levantar (izquierda/derecha/etc., 6 opciones — como las horas de un reloj de 6 números). Cada
trío de 3 personas que se conocen entre sí (un "triángulo") se manda mensajitos para corregirse mutuamente
si están muy desalineados.

- **REAL**: cada trío se corrige A SÍ MISMO — cada persona escucha a SUS DOS compañeros de trío real.
- **NULL-REWIRE** (control viejo): cada trío igual manda su corrección, pero a 3 personas SUELTAS
  sacadas al azar de toda la ronda (ni se conocen entre sí).
- **NULL-LOCAL-ROTO** (nuevo): cada trío manda su corrección a OTRO trío real de la ronda (3 personas
  que SÍ se conocen entre sí, pero no son las suyas).
- **NULL-GLOBAL** (nuevo): en vez de mensajitos de a 3, hay un altavoz único que le dice a TODA la
  ronda, a la vez, "muévanse un poquito hacia el promedio de la sala".

Resultado: el altavoz único (NULL-GLOBAL) NO logra que la ronda se ordene — es como si no hubiera
hecho nada, casi idéntico a dejarlos en total desorden. En cambio, los mensajitos de a tríos —incluso
mandados a la gente equivocada (REWIRE y LOCAL-ROTO)— sí logran bastante orden, aunque no tanto como
cuando cada trío se corrige a sí mismo (REAL). Es decir: lo que ordena a la ronda no es "hablar más
fuerte para todos" — es "hablar en grupos chicos de a tres", aunque sea con el grupo equivocado. Hablar
con TU PROPIO grupo (REAL) todavía suma un poco más de orden por encima de eso (~8-19% extra, según
cómo se mida) — ese es el "cierre local genuino" que venía reportando el equipo.

## 6. Archivos generados

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs083b_fase4_control_local_global.py` — script nuevo
  (no modifica cs082 ni cs083, los importa).
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs083b_resultados.csv` — datos crudos, 20 semillas,
  columnas por brazo (h_real, h_rewire, h_local_roto, h_global, h_null, h_shuf + metadatos de
  equiparación DoF/sweeps/eventos por semilla).
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs083b_resultados_piloto.csv` — piloto de 5 semillas
  (usado para validar el código antes de escalar).

Sin cierre. Números arriba; la interpretación final queda para Alexis.
