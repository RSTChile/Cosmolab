# PROTOCOLO E5.3-1 — PRE-REGISTRO (firmado antes de correr el motor)

**Experimento:** E5.3-1 · "Eficiencia estructura/total barriendo ε de 12 décadas (0% a 100% posible)"
**Tema:** 3 — Eficiencia de conversión emergente ★ (ancla contra 4.9%/31.5%, junto con E5.3-5)
**Agente:** E5.3-1 (batería Enfoque 5, 30 experimentos en paralelo)
**Timestamp de pre-registro (UTC):** 2026-07-25T00:41:03Z
**Documento autoritativo:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md` (sección 0 + REGLAS DE EJECUCIÓN + intro TEMA 3 + spec E5.3-1)
**Regla de oro aplicada:** barrido sobredimensionado — ε en 12 décadas, r en 6 décadas — la eficiencia puede
caer en CUALQUIER punto de [0,1]; el barrido NO se centra en 4.9%/31.5%.

Este protocolo es la definición **CANÓNICA** de E_ligada para todo el TEMA 3. Los experimentos
E5.3-2, E5.3-3, E5.3-4 y E5.3-5 (otros agentes) están instruidos a reutilizar esta definición leyendo
este archivo — no a redefinirla desde cero. Cualquier cambio de definición debe declararse
explícitamente en el reporte del experimento que se aparte, no editando este archivo.

---

## 1. Por qué esto y no otra cosa (justificación conceptual, ANTES de ver resultados)

El código base `cs074_rcruz.py` (leído, no editado, no ejecutado — solo referencia conceptual
compartida por la batería: campo φ en anillo, difusión solo por aristas vivas, expansión que
corta aristas con probabilidad H, r=H/D) establece la convención de campo+conectividad que usan
varios experimentos del Enfoque 5 (r=H/D, ε=amplitud de perturbación estructurada). E5.3-1 reutiliza
ESA convención de dinámica (campo en anillo, difusión local, expansión que corta enlaces) porque es
el sustrato compartido de la batería, pero define un observable PROPIO y NUEVO para el Tema 3:
E_ligada. No se copia física de masa/confinamiento de `CF4_confinamiento.py`; de ese archivo solo se
toma la idea general (functional de ligadura sobre el campo), no su implementación.

**Idea física de E_ligada:** de la energía estructural con la que arranca el campo, una fracción queda
"atrapada" en dominios que la expansión aisló (les cortó TODAS las conexiones con el resto del anillo)
antes de que la difusión alcanzara a homogeneizarlos. Esa energía ya no puede intercambiarse con el
resto (no es exergía global disponible, X, que se define contra el equilibrio de TODO el sistema) ni
se ha perdido en el termalizado general (no es entropía/degradada); quedó **encerrada** dentro de una
estructura persistente y aislada. Esto es distinto de X (E5.1-x: desviación del equilibrio uniforme,
disponible para trabajo con el resto) y distinto de la entropía (homogeneización dentro de una región
aún conectada). E_ligada es la tercera pata de la contabilidad {X, degradada, ligada} que E5.2-4 exige
que sume E_total.

---

## 2. Definición EXACTA de E_ligada (congelada, SALIDA no ajustada)

### 2.1 Campo y dinámica

- Anillo de **N=200** sitios (fijo; el barrido en N es tarea de E5.6-3, no de aquí).
- Campo inicial: φ_i(0) = 1 + ε·pert_i, con `pert` = suma de 5 modos seno de frecuencia
  1..5 con fase aleatoria (rng), centrada y normalizada a std=1 antes de escalar por ε
  (idéntico método genérico de perturbación estructurada usado en el resto de la batería;
  no es física de masa, es una forma estándar de generar una perturbación con estructura
  espacial no trivial).
- ε barrido en **12 décadas**: {1e-12, 1e-11, ..., 1e0} (13 puntos, `np.logspace(-12,0,13)`),
  MÁS un punto de control ε=0 (vacío exacto, caso degenerado, ver §2.4).
- Aristas activas (bonds) entre vecinos i, i+1 (mod N); todas activas al inicio.
- Cada paso:
  1. **Difusión** local SOLO por aristas activas (promedio con vecinos vivos), igual en
     espíritu al operador de `cs074_rcruz.py::paso_difusion` (reescrito aquí, no importado).
  2. **Ruido dinámico** (T7 — perturbación dinámica además de semilla, NO cosmética):
     tras difundir, cada arista activa intercambia (swap) los valores de φ en sus dos
     extremos con probabilidad `p_swap=0.02` (constante metodológica fija, análoga a
     `MARGEN_LAVADO`/`P_LAVADO` de cs074 — NO es un ajuste hacia el blanco). Un swap es una
     PERMUTACIÓN de valores existentes: redistribuye estructura espacialmente sin crear ni
     destruir energía (preserva exactamente Σ_i(φ_i−1)² del anillo completo). Se eligió swap
     en vez de ruido aditivo gaussiano precisamente para que el presupuesto declarado E1 no
     pueda violarse por construcción (ver §2.9) — un aditivo gaussiano habría podido inflar
     artificialmente la energía en el régimen ε→1e-12, rompiendo la cota E_ligada≤E_total.
  3. **Expansión**: cada arista activa se corta con probabilidad Bernoulli H = min(r·D_eps, 1)
     (mismo mecanismo que `paso_expansion` de cs074: corte independiente por arista, válido
     también para H pequeño).
- r barrido en **6 décadas**: {1e-3, ..., 1e3} (13 puntos, `np.logspace(-3,3,13)`).
- Pasos por corrida: **fijo POR ε, NO por r** —
  `pasos(ε) = clip(ceil(5.0 / D_eps), 100, 3000)`, una ventana de observación de ≈5 "tiempos
  de difusión" propios del campo (constante metodológica, no apunta a ningún blanco). r NO
  entra en esta fórmula: r solo determina H=r·D_eps, la velocidad de corte DENTRO de esa
  misma ventana fija compartida por todos los r de ese ε. **Corrección de diseño (detectada
  en prueba de humo, antes de correr producción):** una primera versión calibraba
  `pasos=ceil(5/H)` (dependiente de r), lo que forzaba a CUALQUIER r>0 a converger siempre a
  ≈99.3% de aristas cortadas (survival≈e^-5 sea cual sea H) — anulando el propósito del
  barrido de r. Con `pasos(ε)` fijo, la fracción de aristas cortadas al final (`frac_exp`)
  varía naturalmente con r dentro de la misma ventana: frac_exp≈1−e^(−5r) en el régimen
  típico, dando el rango completo de fragmentación (casi nula en r≪1, transición cerca de
  r≈0.2–1, saturada en r≫1) que el barrido sobredimensionado busca observar.
- Semillas: **20** (≥16 exigidas) por cada (ε,r), independientes, generan tanto la
  condición inicial como el ruido dinámico y el proceso de corte.

### 2.2 D_eps (difusividad medida, no impuesta)

Igual método que cs074: `D_eps` = fracción de contraste (std) que borra UN paso de difusión pura
(H=0), medida sobre el propio campo inicial, promediada sobre las semillas. Para ε=0 no hay
contraste que medir → D_eps se fija por convención a 0 (ver §2.4).

### 2.3 E_total (presupuesto declarado, axioma E1)

    E_total(ε) := Σ_i (φ_i(0) − 1)²     [suma sobre las N sitios, con la MISMA condición
                                          inicial φ(0) de esa corrida — semilla incluida]

Es el presupuesto de energía estructural con el que arranca esa corrida particular. Por
axioma E1 (declarado, NO física real que conserve globalmente) este es el total que se
rastrea: toda la energía que "aparece" en cualquier categoría de la contabilidad tiene que
salir de aquí, nunca inventarse.

### 2.4 Dominios (estructura emergente, T0: nada discreto puesto a mano)

Al final de la corrida (tras `pasos` pasos), se calculan las **componentes conexas** del
grafo de aristas activas remanentes (BFS/recorrido de anillo con las aristas vivas al
final). Analogía con percolación: el dominio de mayor tamaño (la "componente gigante") se
interpreta como el remanente todavía conectado — el trasfondo que sigue intercambiando por
difusión, NO estructura atrapada. Los demás dominios (los que quedaron aislados y son
MINORÍA frente al remanente) son la estructura "ligada".

Un dominio D_k es "**ligado**" (bound) si y solo si:
  - 1 ≤ |D_k| < N, **Y**
  - D_k NO es el dominio de mayor tamaño de esa corrida (si hay empate en el tamaño
    máximo, se excluye solo UNO, el de menor índice de sitio inicial — convención
    determinista, documentada, sin efecto material salvo en el caso de empate exacto).

**Corrección de diseño (detectada en prueba de humo):** una primera versión marcaba como
"ligado" a TODO dominio con tamaño<N. Eso funciona bien cuando la red apenas se fragmenta,
pero en cuanto aparece más de un dominio, la definición original marcaba TODOS los
dominios (sin excepción) como ligados — es decir, el 100% del anillo pasaba a contar como
"estructura atrapada" apenas se cortaba una sola arista, sin importar que un dominio
siguiera siendo casi todo el anillo. Eso rompía el NULL (T4): al sumar sobre
prácticamente todos los N sitios, la suma es invariante a la permutación (una permutación
conserva la suma total), así que REAL≈NULL siempre, no una prueba real. Excluir la
componente gigante (el remanente) corrige esto: ahora "ligado" es una MINORÍA genuina de
sitios, y el NULL sí puede morder (compara si esos sitios específicos, aislados temprano
por la expansión, concentran más energía estructural que un subconjunto aleatorio del
mismo tamaño tomado de la MISMA corrida).

### 2.5 E_ligada (observable, SALIDA)

    E_ligada := Σ_{i ∈ dominios aislados} (φ_i(final) − 1)²

es decir, la energía estructural (misma métrica que E_total, pero evaluada con los valores
FINALES del campo) que quedó contenida dentro de dominios que la expansión aisló del resto.
Se usa el valor final (no el inicial) porque dentro de un dominio aislado la difusión interna
puede seguir operando (parte de esa energía puede seguir degradándose puertas adentro) — lo
que mide E_ligada es cuánta energía estructural quedó ENCERRADA, no cuánta se conservó intacta.

### 2.6 Eficiencia (el observable de E5.3-1)

    eficiencia(ε, r, semilla) := E_ligada / E_total     (∈ [0,1] por construcción; ver §2.4)

Si E_total = 0 (solo ocurre en ε=0, ver §2.7), la eficiencia se define como 0/0 → se reporta
por separado como "indefinida" y se excluye del histograma/curva (no se imputa un valor).

### 2.9 Guardián de conservación (T6, verificado cada celda no cada paso individual por costo,
    pero garantizado por construcción del operador)

Con el swap como único mecanismo de ruido dinámico, y dado que la difusión (promedio local)
es contractiva (no puede aumentar Σ_i(φ_i−1)² del anillo completo) y el corte de aristas no
toca valores de φ, se cumple SIEMPRE: Σ_i(φ_i(t)−1)² ≤ Σ_i(φ_i(0)−1)² = E_total, para todo t.
Como E_ligada es una suma parcial (solo sobre sitios en dominios aislados) de esa misma
cantidad no-creciente, se sigue que **E_ligada(t) ≤ E_total siempre**, por construcción — la
eficiencia nunca puede exceder 1 ni depender de un ajuste externo. Esto se verifica también
numéricamente en el motor (assert por celda) como doble chequeo, no solo se asume.

### 2.7 Caso de control ε=0

φ(0) es exactamente plano (=1 en todos los sitios). E_total=0 por definición. No hay
estructura que atrapar. Se corre igual (para confirmar que E_ligada=0 también, por
construcción, ya que φ(final) también queda en 1 sin ruido más allá del dinámico) y se
reporta aparte como punto de control, NO como parte del barrido logarítmico de 12 décadas.

### 2.8 NULL (T4 — debe morder)

Para cada corrida REAL se genera un NULL: se permuta aleatoriamente (misma rng, tirada
adicional) el campo final φ(final) ANTES de calcular E_ligada — la topología de dominios
(quién quedó conectado con quién) es la MISMA que en REAL (no se re-simula la expansión),
pero los valores de energía en cada sitio quedan barajados. Esto pone a prueba si los
dominios reales concentran MÁS energía estructural que una asignación aleatoria de esos
mismos valores a esos mismos dominios — si REAL ≈ NULL, la "ligadura" no es más que el
tamaño del dominio (proporción trivial), no una concentración real de estructura.

---

## 3. Juez, PASS/FAIL (congelado ANTES de correr, T3)

- **Observable:** curva completa eficiencia(ε,r) — media y dispersión sobre 20 semillas —
  sobre las 13×13 celdas del grid log (169 celdas), más histograma de todos los valores
  obtenidos (169 celdas × 20 semillas = 3380 corridas REAL, + 3380 NULL).
- **NO hay un umbral de aprobar/reprobar único** — T5 exige la curva entera, no un gate
  binario. El "PASS" de este experimento es: (a) la curva se produjo en su totalidad sin
  errores de conservación (E_ligada ≤ E_total siempre, por construcción — se verifica
  explícitamente como guardián T6), (b) el NULL se calculó y se compara sin excepción, (c) se
  reporta, SIN AJUSTAR NADA, si algún régimen (ε,r) cae cerca de 4.9% o 31.5% por azar del
  barrido — "cerca" definido post-hoc en el reporte como |eficiencia−blanco| < 0.02 (2 puntos
  porcentuales), umbral de reporte, NO de selección (no se usa para decidir qué mostrar, se
  aplica a TODA la curva y se reporta cuántas celdas caen ahí, sean muchas o ninguna).
- **Prohibido:** mover D_eps, el factor 5.0 de calibración de pasos, la amplitud de ruido
  0.01, o cualquier otro coeficiente hacia 4.9%/31.5%. Estos quedan fijados aquí, antes de
  ver un solo resultado del motor.

## 4. Verificaciones (regla 4 de ejecución)

1. NULL (arriba, §2.8).
2. Segundo observable/método: se reporta también `frac_exp` (fracción de aristas cortadas al
   final) y el número/tamaño de dominios por celda — para diagnosticar si la eficiencia
   obtenida es "real" (dominios múltiples y parciales) o degenerada (0 dominios aislados, o
   todo el anillo fragmentado en singletons).
3. Auditoría en disco: JSON crudo con las 169×20×2 corridas queda en
   `E5_3_1_resultado_crudo.json` para que otro agente/CS pueda re-verificar sin re-correr.

## 5. Archivos que produce este experimento

- `PROTOCOLO_E5.3-1_PREREGISTRO.md` (este archivo)
- `E5_3_1_motor.py` (motor, no se edita tras esta firma salvo error ajeno detectado — T3/regla 7)
- `E5_3_1_resultado_crudo.json` (salida cruda del motor)
- `E5_3_1_resumen.json` (agregados: curva media±std por celda, histograma, distancias a blancos)
- `E5_3_1_REPORTE.md` (reporte final para CS)

---
*Firmado (pre-registro) antes de escribir una sola línea del motor. Cualquier desviación de
lo aquí escrito, si ocurre por necesidad técnica durante la implementación, se declara
explícitamente en el reporte final con el motivo — nunca en silencio.*
