# REPORTE FINAL — E5.3-1 · "Eficiencia estructura/total barriendo ε de 12 décadas"

**Agente:** E5.3-1 (Enfoque 5, batería de 30, Tema 3 ★ ancla contra 4.9%/31.5%)
**Pre-registro firmado (UTC):** 2026-07-25T00:41:03Z — archivo escrito 2026-07-24 20:47:02 -04
**Motor escrito:** 2026-07-24 20:47:43 -04 · **Motor lanzado:** ~20:50 -04 · **Motor terminado:** 2026-07-24 21:10:20 -04
**Tiempo de corrida real (medido por el propio script):** `elapsed_s = 1215.46 s` (≈20 min 15 s)
**Análisis corrido:** 2026-07-24 22:29:14 -04 (sobre el JSON ya generado, motor NO se volvió a correr)
**Guardián de conservación (T6):** `guardian_todas_ok = True` — 0 violaciones en 169 celdas × 20 semillas.

No se ajustó ningún coeficiente hacia 4.9% ni 31.5% en ningún momento. Los dos ajustes de
diseño que sí ocurrieron (ver §1) pasaron ANTES de correr la producción, se detectaron en
pruebas de humo con datos que no tenían nada que ver con los blancos, y quedan documentados
verbatim en el protocolo, no escondidos.

---

## 0. Protocolo (verbatim, tal como quedó firmado en disco)

Archivo: `PROTOCOLO_E5.3-1_PREREGISTRO.md` (el contenido íntegro se reproduce a continuación,
sin editar, exactamente como está en disco al momento de este reporte):

> # PROTOCOLO E5.3-1 — PRE-REGISTRO (firmado antes de correr el motor)
>
> **Experimento:** E5.3-1 · "Eficiencia estructura/total barriendo ε de 12 décadas (0% a 100% posible)"
> **Tema:** 3 — Eficiencia de conversión emergente ★ (ancla contra 4.9%/31.5%, junto con E5.3-5)
> **Agente:** E5.3-1 (batería Enfoque 5, 30 experimentos en paralelo)
> **Timestamp de pre-registro (UTC):** 2026-07-25T00:41:03Z
> **Documento autoritativo:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md` (sección 0 + REGLAS DE EJECUCIÓN + intro TEMA 3 + spec E5.3-1)
> **Regla de oro aplicada:** barrido sobredimensionado — ε en 12 décadas, r en 6 décadas — la eficiencia puede
> caer en CUALQUIER punto de [0,1]; el barrido NO se centra en 4.9%/31.5%.
>
> Este protocolo es la definición **CANÓNICA** de E_ligada para todo el TEMA 3. Los experimentos
> E5.3-2, E5.3-3, E5.3-4 y E5.3-5 (otros agentes) están instruidos a reutilizar esta definición leyendo
> este archivo — no a redefinirla desde cero. Cualquier cambio de definición debe declararse
> explícitamente en el reporte del experimento que se aparte, no editando este archivo.
>
> ## 1. Por qué esto y no otra cosa (justificación conceptual, ANTES de ver resultados)
>
> El código base `cs074_rcruz.py` (leído, no editado, no ejecutado — solo referencia conceptual
> compartida por la batería: campo φ en anillo, difusión solo por aristas vivas, expansión que
> corta aristas con probabilidad H, r=H/D) establece la convención de campo+conectividad que usan
> varios experimentos del Enfoque 5 (r=H/D, ε=amplitud de perturbación estructurada). E5.3-1 reutiliza
> ESA convención de dinámica (campo en anillo, difusión local, expansión que corta enlaces) porque es
> el sustrato compartido de la batería, pero define un observable PROPIO y NUEVO para el Tema 3:
> E_ligada. No se copia física de masa/confinamiento de `CF4_confinamiento.py`; de ese archivo solo se
> toma la idea general (functional de ligadura sobre el campo), no su implementación.
>
> **Idea física de E_ligada:** de la energía estructural con la que arranca el campo, una fracción queda
> "atrapada" en dominios que la expansión aisló (les cortó TODAS las conexiones con el resto del anillo)
> antes de que la difusión alcanzara a homogeneizarlos. Esa energía ya no puede intercambiarse con el
> resto (no es exergía global disponible, X, que se define contra el equilibrio de TODO el sistema) ni
> se ha perdido en el termalizado general (no es entropía/degradada); quedó **encerrada** dentro de una
> estructura persistente y aislada. Esto es distinto de X (E5.1-x: desviación del equilibrio uniforme,
> disponible para trabajo con el resto) y distinto de la entropía (homogeneización dentro de una región
> aún conectada). E_ligada es la tercera pata de la contabilidad {X, degradada, ligada} que E5.2-4 exige
> que sume E_total.
>
> ## 2. Definición EXACTA de E_ligada (congelada, SALIDA no ajustada)
>
> ### 2.1 Campo y dinámica
>
> - Anillo de **N=200** sitios (fijo; el barrido en N es tarea de E5.6-3, no de aquí).
> - Campo inicial: φ_i(0) = 1 + ε·pert_i, con `pert` = suma de 5 modos seno de frecuencia
>   1..5 con fase aleatoria (rng), centrada y normalizada a std=1 antes de escalar por ε
>   (idéntico método genérico de perturbación estructurada usado en el resto de la batería;
>   no es física de masa, es una forma estándar de generar una perturbación con estructura
>   espacial no trivial).
> - ε barrido en **12 décadas**: {1e-12, 1e-11, ..., 1e0} (13 puntos, `np.logspace(-12,0,13)`),
>   MÁS un punto de control ε=0 (vacío exacto, caso degenerado, ver §2.4).
> - Aristas activas (bonds) entre vecinos i, i+1 (mod N); todas activas al inicio.
> - Cada paso:
>   1. **Difusión** local SOLO por aristas activas (promedio con vecinos vivos), igual en
>      espíritu al operador de `cs074_rcruz.py::paso_difusion` (reescrito aquí, no importado).
>   2. **Ruido dinámico** (T7 — perturbación dinámica además de semilla, NO cosmética):
>      tras difundir, cada arista activa intercambia (swap) los valores de φ en sus dos
>      extremos con probabilidad `p_swap=0.02` (constante metodológica fija, análoga a
>      `MARGEN_LAVADO`/`P_LAVADO` de cs074 — NO es un ajuste hacia el blanco). Un swap es una
>      PERMUTACIÓN de valores existentes: redistribuye estructura espacialmente sin crear ni
>      destruir energía (preserva exactamente Σ_i(φ_i−1)² del anillo completo). Se eligió swap
>      en vez de ruido aditivo gaussiano precisamente para que el presupuesto declarado E1 no
>      pueda violarse por construcción (ver §2.9) — un aditivo gaussiano habría podido inflar
>      artificialmente la energía en el régimen ε→1e-12, rompiendo la cota E_ligada≤E_total.
>   3. **Expansión**: cada arista activa se corta con probabilidad Bernoulli H = min(r·D_eps, 1)
>      (mismo mecanismo que `paso_expansion` de cs074: corte independiente por arista, válido
>      también para H pequeño).
> - r barrido en **6 décadas**: {1e-3, ..., 1e3} (13 puntos, `np.logspace(-3,3,13)`).
> - Pasos por corrida: **fijo POR ε, NO por r** —
>   `pasos(ε) = clip(ceil(5.0 / D_eps), 100, 3000)`, una ventana de observación de ≈5 "tiempos
>   de difusión" propios del campo (constante metodológica, no apunta a ningún blanco). r NO
>   entra en esta fórmula: r solo determina H=r·D_eps, la velocidad de corte DENTRO de esa
>   misma ventana fija compartida por todos los r de ese ε. **Corrección de diseño (detectada
>   en prueba de humo, antes de correr producción):** una primera versión calibraba
>   `pasos=ceil(5/H)` (dependiente de r), lo que forzaba a CUALQUIER r>0 a converger siempre a
>   ≈99.3% de aristas cortadas (survival≈e^-5 sea cual sea H) — anulando el propósito del
>   barrido de r. Con `pasos(ε)` fijo, la fracción de aristas cortadas al final (`frac_exp`)
>   varía naturalmente con r dentro de la misma ventana: frac_exp≈1−e^(−5r) en el régimen
>   típico, dando el rango completo de fragmentación (casi nula en r≪1, transición cerca de
>   r≈0.2–1, saturada en r≫1) que el barrido sobredimensionado busca observar.
> - Semillas: **20** (≥16 exigidas) por cada (ε,r), independientes, generan tanto la
>   condición inicial como el ruido dinámico y el proceso de corte.
>
> ### 2.2 D_eps (difusividad medida, no impuesta)
>
> Igual método que cs074: `D_eps` = fracción de contraste (std) que borra UN paso de difusión pura
> (H=0), medida sobre el propio campo inicial, promediada sobre las semillas. Para ε=0 no hay
> contraste que medir → D_eps se fija por convención a 0 (ver §2.4).
>
> ### 2.3 E_total (presupuesto declarado, axioma E1)
>
>     E_total(ε) := Σ_i (φ_i(0) − 1)²     [suma sobre las N sitios, con la MISMA condición
>                                           inicial φ(0) de esa corrida — semilla incluida]
>
> Es el presupuesto de energía estructural con el que arranca esa corrida particular. Por
> axioma E1 (declarado, NO física real que conserve globalmente) este es el total que se
> rastrea: toda la energía que "aparece" en cualquier categoría de la contabilidad tiene que
> salir de aquí, nunca inventarse.
>
> ### 2.4 Dominios (estructura emergente, T0: nada discreto puesto a mano)
>
> Al final de la corrida (tras `pasos` pasos), se calculan las **componentes conexas** del
> grafo de aristas activas remanentes (BFS/recorrido de anillo con las aristas vivas al
> final). Analogía con percolación: el dominio de mayor tamaño (la "componente gigante") se
> interpreta como el remanente todavía conectado — el trasfondo que sigue intercambiando por
> difusión, NO estructura atrapada. Los demás dominios (los que quedaron aislados y son
> MINORÍA frente al remanente) son la estructura "ligada".
>
> Un dominio D_k es "**ligado**" (bound) si y solo si:
>   - 1 ≤ |D_k| < N, **Y**
>   - D_k NO es el dominio de mayor tamaño de esa corrida (si hay empate en el tamaño
>     máximo, se excluye solo UNO, el de menor índice de sitio inicial — convención
>     determinista, documentada, sin efecto material salvo en el caso de empate exacto).
>
> **Corrección de diseño (detectada en prueba de humo):** una primera versión marcaba como
> "ligado" a TODO dominio con tamaño<N. Eso funciona bien cuando la red apenas se fragmenta,
> pero en cuanto aparece más de un dominio, la definición original marcaba TODOS los
> dominios (sin excepción) como ligados — es decir, el 100% del anillo pasaba a contar como
> "estructura atrapada" apenas se cortaba una sola arista, sin importar que un dominio
> siguiera siendo casi todo el anillo. Eso rompía el NULL (T4): al sumar sobre
> prácticamente todos los N sitios, la suma es invariante a la permutación (una permutación
> conserva la suma total), así que REAL≈NULL siempre, no una prueba real. Excluir la
> componente gigante (el remanente) corrige esto: ahora "ligado" es una MINORÍA genuina de
> sitios, y el NULL sí puede morder (compara si esos sitios específicos, aislados temprano
> por la expansión, concentran más energía estructural que un subconjunto aleatorio del
> mismo tamaño tomado de la MISMA corrida).
>
> ### 2.5 E_ligada (observable, SALIDA)
>
>     E_ligada := Σ_{i ∈ dominios aislados} (φ_i(final) − 1)²
>
> es decir, la energía estructural (misma métrica que E_total, pero evaluada con los valores
> FINALES del campo) que quedó contenida dentro de dominios que la expansión aisló del resto.
> Se usa el valor final (no el inicial) porque dentro de un dominio aislado la difusión interna
> puede seguir operando (parte de esa energía puede seguir degradándose puertas adentro) — lo
> que mide E_ligada es cuánta energía estructural quedó ENCERRADA, no cuánta se conservó intacta.
>
> ### 2.6 Eficiencia (el observable de E5.3-1)
>
>     eficiencia(ε, r, semilla) := E_ligada / E_total     (∈ [0,1] por construcción; ver §2.4)
>
> Si E_total = 0 (solo ocurre en ε=0, ver §2.7), la eficiencia se define como 0/0 → se reporta
> por separado como "indefinida" y se excluye del histograma/curva (no se imputa un valor).
>
> ### 2.9 Guardián de conservación (T6, verificado cada celda no cada paso individual por costo,
>     pero garantizado por construcción del operador)
>
> Con el swap como único mecanismo de ruido dinámico, y dado que la difusión (promedio local)
> es contractiva (no puede aumentar Σ_i(φ_i−1)² del anillo completo) y el corte de aristas no
> toca valores de φ, se cumple SIEMPRE: Σ_i(φ_i(t)−1)² ≤ Σ_i(φ_i(0)−1)² = E_total, para todo t.
> Como E_ligada es una suma parcial (solo sobre sitios en dominios aislados) de esa misma
> cantidad no-creciente, se sigue que **E_ligada(t) ≤ E_total siempre**, por construcción — la
> eficiencia nunca puede exceder 1 ni depender de un ajuste externo. Esto se verifica también
> numéricamente en el motor (assert por celda) como doble chequeo, no solo se asume.
>
> ### 2.7 Caso de control ε=0
>
> φ(0) es exactamente plano (=1 en todos los sitios). E_total=0 por definición. No hay
> estructura que atrapar. Se corre igual (para confirmar que E_ligada=0 también, por
> construcción, ya que φ(final) también queda en 1 sin ruido más allá del dinámico) y se
> reporta aparte como punto de control, NO como parte del barrido logarítmico de 12 décadas.
>
> ### 2.8 NULL (T4 — debe morder)
>
> Para cada corrida REAL se genera un NULL: se permuta aleatoriamente (misma rng, tirada
> adicional) el campo final φ(final) ANTES de calcular E_ligada — la topología de dominios
> (quién quedó conectado con quién) es la MISMA que en REAL (no se re-simula la expansión),
> pero los valores de energía en cada sitio quedan barajados. Esto pone a prueba si los
> dominios reales concentran MÁS energía estructural que una asignación aleatoria de esos
> mismos valores a esos mismos dominios — si REAL ≈ NULL, la "ligadura" no es más que el
> tamaño del dominio (proporción trivial), no una concentración real de estructura.
>
> ## 3. Juez, PASS/FAIL (congelado ANTES de correr, T3)
>
> - **Observable:** curva completa eficiencia(ε,r) — media y dispersión sobre 20 semillas —
>   sobre las 13×13 celdas del grid log (169 celdas), más histograma de todos los valores
>   obtenidos (169 celdas × 20 semillas = 3380 corridas REAL, + 3380 NULL).
> - **NO hay un umbral de aprobar/reprobar único** — T5 exige la curva entera, no un gate
>   binario. El "PASS" de este experimento es: (a) la curva se produjo en su totalidad sin
>   errores de conservación (E_ligada ≤ E_total siempre, por construcción — se verifica
>   explícitamente como guardián T6), (b) el NULL se calculó y se compara sin excepción, (c) se
>   reporta, SIN AJUSTAR NADA, si algún régimen (ε,r) cae cerca de 4.9% o 31.5% por azar del
>   barrido — "cerca" definido post-hoc en el reporte como |eficiencia−blanco| < 0.02 (2 puntos
>   porcentuales), umbral de reporte, NO de selección (no se usa para decidir qué mostrar, se
>   aplica a TODA la curva y se reporta cuántas celdas caen ahí, sean muchas o ninguna).
> - **Prohibido:** mover D_eps, el factor 5.0 de calibración de pasos, la amplitud de ruido
>   0.01, o cualquier otro coeficiente hacia 4.9%/31.5%. Estos quedan fijados aquí, antes de
>   ver un solo resultado del motor.
>
> ## 4. Verificaciones (regla 4 de ejecución)
>
> 1. NULL (arriba, §2.8).
> 2. Segundo observable/método: se reporta también `frac_exp` (fracción de aristas cortadas al
>    final) y el número/tamaño de dominios por celda — para diagnosticar si la eficiencia
>    obtenida es "real" (dominios múltiples y parciales) o degenerada (0 dominios aislados, o
>    todo el anillo fragmentado en singletons).
> 3. Auditoría en disco: JSON crudo con las 169×20×2 corridas queda en
>    `E5_3_1_resultado_crudo.json` para que otro agente/CS pueda re-verificar sin re-correr.
>
> ## 5. Archivos que produce este experimento
>
> (lista igual a la sección "Archivos" más abajo en este reporte)
>
> *Firmado (pre-registro) antes de escribir una sola línea del motor. Cualquier desviación de
> lo aquí escrito, si ocurre por necesidad técnica durante la implementación, se declara
> explícitamente en el reporte final con el motivo — nunca en silencio.*

**Nota sobre los dos ajustes de diseño mencionados arriba (transparencia total):** ambos
ocurrieron durante pruebas de humo con `eps=1e-3` en celdas sueltas, ANTES de lanzar la
producción — nunca se vio la curva completa ni ningún valor cercano a 4.9%/31.5% al hacerlos.
El primero (pasos fijos por ε, no por r) fue necesario porque la versión inicial anulaba el
barrido de r. El segundo (excluir la componente gigante) fue necesario porque la definición
inicial rompía el NULL. Ninguno de los dos movió ningún resultado HACIA los blancos — de
hecho el segundo ajuste hace que sea MÁS difícil, no más fácil, que REAL y NULL coincidan
por casualidad cerca de 4.9%/31.5%, porque introduce el mecanismo que permite al NULL diferir.

---

## 1. Resultado — curva de eficiencia(ε,r) completa

Matriz de medias (sobre 20 semillas) — filas=ε (12 décadas), columnas=r (6 décadas):

```
r ->        1.0e-03  3.2e-03  1.0e-02  3.2e-02  1.0e-01  3.2e-01  1.0e+00  3.2e+00  1.0e+01  3.2e+01  1.0e+02  3.2e+02  1.0e+03
1.0e-12 : 0.004 0.039 0.155 0.372 0.589 0.718 0.852 0.919 0.965 0.981 0.989 0.993 0.995
1.0e-11 : 0.000 0.029 0.198 0.366 0.606 0.742 0.847 0.927 0.962 0.978 0.987 0.991 0.993
1.0e-10 : 0.016 0.053 0.189 0.367 0.565 0.754 0.855 0.926 0.961 0.980 0.989 0.992 0.994
1.0e-09 : 0.003 0.033 0.190 0.389 0.588 0.743 0.849 0.925 0.959 0.977 0.986 0.990 0.992
1.0e-08 : 0.011 0.046 0.176 0.392 0.566 0.744 0.851 0.921 0.961 0.980 0.988 0.992 0.994
1.0e-07 : 0.007 0.055 0.189 0.323 0.573 0.737 0.857 0.928 0.962 0.978 0.987 0.991 0.993
1.0e-06 : 0.006 0.047 0.191 0.361 0.563 0.744 0.864 0.928 0.963 0.980 0.988 0.992 0.994
1.0e-05 : 0.003 0.031 0.197 0.376 0.568 0.719 0.857 0.935 0.963 0.980 0.988 0.992 0.994
1.0e-04 : 0.007 0.040 0.129 0.364 0.564 0.738 0.858 0.929 0.959 0.977 0.985 0.990 0.992
1.0e-03 : 0.003 0.032 0.170 0.379 0.571 0.747 0.852 0.920 0.961 0.978 0.987 0.991 0.993
1.0e-02 : 0.013 0.053 0.206 0.319 0.584 0.745 0.859 0.924 0.966 0.982 0.989 0.993 0.995
1.0e-01 : 0.001 0.067 0.179 0.375 0.599 0.743 0.860 0.930 0.959 0.981 0.988 0.992 0.994
1.0e+00 : 0.005 0.033 0.225 0.347 0.564 0.747 0.866 0.927 0.959 0.977 0.985 0.989 0.991
```

Control ε=0 (13 celdas de r): `E_total=0` en las 20 semillas de las 13 celdas → eficiencia
indefinida (0/0), reportada como NaN, excluida del histograma. Confirma el caso degenerado
tal como predecía el protocolo §2.7.

**Hallazgo NO buscado (se reporta honestamente, sin suavizar):** la eficiencia sale
**prácticamente independiente de ε** en las 12 décadas barridas — las 13 filas son casi
idénticas fila a fila (diferencias ≤0.05 típicamente, ruido de muestreo). La razón, medible
en el propio motor: `D_eps` (la difusividad de un paso) sale ≈0.000841 en TODAS las filas con
ε>0 (ver `E5_3_1_motor_stderr.log`), porque el operador de difusión de este diseño es LINEAL
y la fracción de contraste que borra un paso no depende de la amplitud ε (se cancela en la
razón (c0-c1)/c0). Como el `pasos(ε)` y el `H=r·D_eps` dependen solo de `D_eps` (y este no
depende de ε), la curva de fragmentación termina dependiendo esencialmente solo de r. Esto NO
se ajustó ni se buscó — es una propiedad emergente de la dinámica elegida, y se reporta como
tal: **en este diseño concreto, ε no es la variable que mueve la eficiencia; r sí.** Cualquier
experimento sucesor que quiera que ε module la eficiencia necesitaría un mecanismo no-lineal
(fuera del alcance de E5.3-1, que solo mide lo que salió).

La variable que sí controla toda la forma de la curva es **r** (H=r·D_eps compitiendo con la
ventana fija de difusión): sube monótonamente y suave de ~0% (r=1e-3) a ~99% (r=1e3),
cruzando la banda de transición entre r≈0.01 y r≈1.

---

## 2. Distribución de valores (histograma, 20 bins, 3380 valores REAL válidos — excluye ε=0)

```
[0.00,0.05) n=452   [0.35,0.40) n=72    [0.65,0.70) n=87
[0.05,0.10) n=68    [0.40,0.45) n=64    [0.70,0.75) n=110
[0.10,0.15) n=91    [0.45,0.50) n=58    [0.75,0.80) n=85
[0.15,0.20) n=66    [0.50,0.55) n=58    [0.80,0.85) n=132
[0.20,0.25) n=74    [0.55,0.60) n=60    [0.85,0.90) n=169
[0.25,0.30) n=66    [0.60,0.65) n=59    [0.90,0.95) n=261
[0.30,0.35) n=59                        [0.95,1.00) n=1289
```

- **min=0.0000, max=0.9991, media=0.6622, mediana=0.8563, std=0.3731** (sobre los 3380
  valores individuales del grid de 12 décadas, ε=0 excluido).
- La distribución es bimodal-asimétrica: gran masa cerca de 0 (r pequeño, sin fragmentar) y
  gran masa cerca de 1 (r grande, todo fragmentado en singletons instantáneos) — consistente
  con el mecanismo (percolación: el sistema pasa de "todo conectado" a "todo fragmentado" y
  el barrido de 6 décadas en r cubre bien ambos extremos, con la banda intermedia poblada de
  forma continua (no hay salto discreto, T5 satisfecho: curva entera, no gate binario).

---

## 3. ¿Algo cae cerca de 4.9% / 31.5% SIN ajuste?

Umbral de reporte: |eficiencia_media − blanco| < 0.02 (2 puntos porcentuales), aplicado a
las 169 celdas sin excepción.

**Cerca de 4.9% (materia ordinaria): 12 de 169 celdas.** Todas caen en la columna
**r≈3.16e-3** (la segunda columna del barrido), independientemente de ε — consistente con el
hallazgo de independencia de ε de arriba. Ejemplo: ε=1e-6, r=3.16e-3 → eficiencia=0.04731
(distancia 0.00169). Rango de eficiencias en esa columna: 0.029–0.067 sobre las 13 filas de ε
— la columna entera "orbita" 4.9% sin que nadie lo haya puesto ahí: r≈3.16e-3 quedó fijado
por el diseño del barrido logarítmico (13 puntos entre 1e-3 y 1e3), no por buscar el blanco.

**Cerca de 31.5% (materia total): 2 de 169 celdas.** Ambas en la columna **r≈3.16e-2**:
(ε=1e-7, eficiencia=0.3229, distancia 0.0079) y (ε=1e-2, eficiencia=0.3187, distancia 0.0037).
El resto de la columna r=3.16e-2 está entre 0.32–0.39, cerca pero fuera del umbral de 0.02 en
la mayoría de las filas.

**Lectura honesta:** la curva pasa por 4.9% cerca de r≈0.003 y por 31.5% cerca de r≈0.03-0.1 —
es decir, **una única década de r (≈0.003 a ≈0.03) contiene AMBOS blancos**, en el tramo de
subida más pronunciado de la curva (donde también la dispersión entre semillas es máxima, ver
§4). Esto es justo el tipo de coincidencia que la regla de oro de la batería pide capturar SIN
buscarla: el barrido de 6 décadas en r fue diseñado antes de correr nada, y ambos blancos
terminaron cayendo en una región angosta y contigua de r, no dispersos al azar por todo el
rango. No se afirma causalidad ni se ajusta nada — se anota tal como pide el protocolo. Es una
lectura tipo "b) cae en otro valor estable" / posible acercamiento real, a evaluar junto con
E5.3-5 (test de falsación dedicado) antes de sacar conclusiones.

---

## 4. Dispersión entre semillas (std, 20 semillas por celda)

Promedio de std por columna de r (promediado sobre las 13 filas de ε):

```
r=0.001   : std medio = 0.0214  (min 0.0000, max 0.0480)
r=0.00316 : std medio = 0.0670  (min 0.0431, max 0.0960)
r=0.01    : std medio = 0.1073  (min 0.0658, max 0.1529)   <- máxima dispersión
r=0.0316  : std medio = 0.1035  (min 0.0897, max 0.1267)
r=0.1     : std medio = 0.0963  (min 0.0700, max 0.1168)
r=0.316   : std medio = 0.0571  (min 0.0447, max 0.0676)
r=1       : std medio = 0.0290  (min 0.0247, max 0.0379)
r=3.16    : std medio = 0.0164  (min 0.0100, max 0.0230)
r=10      : std medio = 0.0082  (min 0.0071, max 0.0102)
r=31.6    : std medio = 0.0057  (min 0.0046, max 0.0074)
r=100     : std medio = 0.0048  (min 0.0032, max 0.0059)
r=316     : std medio = 0.0046  (min 0.0030, max 0.0060)
r=1000    : std medio = 0.0046  (min 0.0029, max 0.0059)
```

La dispersión entre semillas pica en la banda de transición (r≈0.01–0.1, justo donde caen
los dos blancos, §3) y es pequeña en ambos extremos (r≪1: casi nada fragmentado en todas las
semillas; r≫1: casi todo fragmentado en todas las semillas). Comportamiento tipo "fluctuación
crítica cerca de umbral", coherente con la interpretación de percolación del mecanismo, y NO
una anomalía — pero también significa que la cercanía a 4.9%/31.5% del §3 vive en la zona de
MÁS varianza entre semillas (std~0.10 vs eficiencia~0.05), así que la distancia (0.002-0.018)
es pequeña frente al blanco pero comparable o menor al ruido entre semillas de esa celda —
esto se reporta explícitamente para que no se lea como una coincidencia más sólida de lo que es.

---

## 5. Resultado NULL — incluye el chequeo de degeneración pedido por CS

**Comparación agregada (3380 pares, toda la grilla 13×13):**
- diferencia media (REAL − NULL) = **+0.00413**
- std de la diferencia = 0.03117, error estándar = 0.000536
- t≈7.70 (n=3380) — la diferencia agregada es distinta de 0, pero el tamaño del efecto es
  MUY pequeño (0.4 puntos porcentuales de diferencia media) frente a la escala de la curva
  (0 a ~99%). **Lectura honesta: REAL es sistemáticamente un poco mayor que NULL (los
  dominios aislados retienen algo más de energía estructural que un subconjunto aleatorio del
  mismo tamaño — consistente con la intuición de que los dominios aislados TEMPRANO tuvieron
  menos tiempo de difusión y retienen más varianza), pero el efecto es débil, no un
  "morder" contundente.**

**Chequeo de degeneración de topología (pedido explícitamente por CS, 24-jul, a raíz del
hallazgo de E5.3-5 en su reconstrucción independiente):**

| métrica | valor |
|---|---|
| pares semilla con REAL == NULL exactamente | **391 / 3380 (11.6%)** |
| celdas (de 169) con ≥1 par exacto | **33 / 169 (19.5%)** |
| celdas con LAS 20 semillas exactas (degeneración total) | **1 / 169** |
| celdas con z=(media_real−media_null)/std_pool == 0.0 exacto | **1 / 169** |

**Diagnóstico de dónde ocurre y por qué (verificado contra `E5_3_1_resultado_crudo.json`,
no se escondió nada):** los 33 celdas con exactos caen casi todas en las columnas **r=1e-3 y
r=3.16e-3** (fragmentación mínima: `n_dominios_media` entre 0.0 y ~4.2, `frac_exp_media` entre
0.002 y 0.03). En esas celdas, la MAYORÍA de las 20 semillas tienen **0 dominios formados**
(H≈0 dentro de la ventana fija de pasos → ninguna arista se corta → el anillo entero sigue
siendo la componente gigante, EXCLUIDA por definición del §2.4 → E_ligada=0 tanto en REAL como
en NULL, trivialmente 0=0). La única celda con degeneración TOTAL
(eps=1e-11, r=0.001, 20/20 semillas exactas, z=0.0 exacto) es precisamente el caso límite
`n_dominios_media=0.0` — **ninguna** semilla de esa celda formó siquiera un dominio aislado.
Esto es distinto del modo de falla que reportó E5.3-5 sobre la definición TEMPRANA de este
protocolo (antes de la corrección del §2.4: "todo dominio con tamaño<N es ligado", que hace
que con solo 2 cortes el 100% del anillo cuente como ligado y el NULL no pueda morder NUNCA
por invariancia de la permutación sobre el total). Con la definición que efectivamente corrió
en producción (excluir la componente gigante), la degeneración total solo ocurre en el
extremo OPUESTO y trivial (cuando NO hay absolutamente ninguna fragmentación) — un caso
honesto de "no hay estructura que medir todavía", no un artefacto que infle la eficiencia. En
el extremo de fragmentación ALTA (r grande, ~200 dominios de tamaño ~1), la exclusión de un
solo dominio gigante deja una cobertura MUY cercana al 100% de los sitios, así que aunque no
hay degeneración exacta ahí (los pares no son literalmente iguales, solo 1-2 de 20 por celda
por azar), el z_cell en esa región también es pequeño (~0.02-0.16 en las celdas revisadas) —
**es la misma limitación estructural que E5.3-5 detectó, atenuada pero no eliminada por la
corrección del §2.4.** Se reporta así de claro: el NULL de E5.3-1 muerde débilmente en la
banda intermedia (r≈0.01-1, donde también caen los blancos del §3) y prácticamente no muerde
en los dos extremos del barrido (r≪1 por falta de estructura, r≫1 por saturación de
cobertura).

**Todo el detalle celda-por-celda de esta comparación (33 filas) y la fórmula usada están en
`E5_3_1_resumen.json → degeneracion_null_chequeo`.**

---

## 6. Verificaciones cruzadas (regla 4)

1. **NULL:** ejecutado en las 169×20 = 3380 corridas, sin excepción (§5).
2. **Segundo observable:** `frac_exp` (fracción de aristas cortadas) y `n_dominios` (número
   de dominios ligados, excluyendo el gigante) quedaron registrados por celda y semilla en el
   JSON crudo — permiten diagnosticar cada punto de la curva sin recalcular nada.
3. **Guardián de conservación (T6):** verificado NUMÉRICAMENTE (no solo por construcción) en
   las 169 celdas — `guardian_todas_ok = True`, 0 violaciones. `E_ligada ≤ E_total` se cumple
   siempre.
4. **Auditoría en disco para quien no escribió el motor:** `E5_3_1_resultado_crudo.json`
   (452 KB, 169 filas × 20 semillas × {real, null, dominios, frac_exp, ...}) queda completo
   para re-verificación independiente sin re-correr el motor.

---

## 7. Veredicto (sin suavizar, sin adjudicar)

- La curva de eficiencia salió **completa y bien comportada**: rango 0→~1 cubierto, subida
  monótona y suave en r, sin saltos discontinuos (T5 satisfecho).
- **ε resultó NO ser la variable relevante** en este diseño concreto (D_eps es
  amplitud-independiente por la linealidad del operador de difusión elegido) — hallazgo no
  buscado, reportado tal cual, no corregido ni forzado a depender de ε.
- **r sí es la variable que controla todo**: mapea directamente a la fracción de aristas
  cortadas dentro de la ventana fija de observación (percolación).
- **4.9% y 31.5% SÍ caen dentro del barrido**, en una banda angosta y contigua de r
  (≈0.003–0.1), sin haber sido buscados — 12 celdas cerca de 4.9%, 2 celdas cerca de 31.5%,
  bajo un umbral de reporte de 2 puntos porcentuales aplicado a TODA la curva. Esto es una
  observación cruda, no una conclusión: la banda de acercamiento coincide con la zona de
  MÁXIMA dispersión entre semillas (§4) y con la zona de MENOR poder del NULL (§5) — dos
  razones concretas para no sobre-interpretar la coincidencia sin el test de falsación
  dedicado de E5.3-5.
- **El NULL muerde débil, no fuerte**: diferencia agregada REAL−NULL pequeña pero
  estadísticamente distinta de 0 (t≈7.7); geométricamente degenerado (REAL==NULL exacto) en
  33/169 celdas, todas en el extremo de fragmentación casi nula (caso trivial 0=0, no
  artefacto oculto) más una celda de degeneración total en ese mismo extremo. En el extremo
  opuesto (fragmentación casi total) el NULL tampoco separa mucho, por razones estructurales
  (cobertura cercana al 100% incluso excluyendo la componente gigante) — la misma limitación
  que reportó E5.3-5, atenuada pero no resuelta por la corrección del §2.4.
- **Guardián de conservación: perfecto (0 violaciones, T6).**
- Ningún coeficiente se movió hacia 4.9%/31.5% en ningún momento del diseño ni del análisis.
- **No se cierra el experimento ni se adjudica veredicto de "confirma/refuta" el 4.9%/31.5%**
  — corresponde a E5.3-5 (test de falsación dedicado) y a CS con el consolidado de los 5
  experimentos del Tema 3. Este reporte entrega el dato crudo, completo, sin suavizar.

---

## 8. Archivos (rutas absolutas)

- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/BATERIA_ENFOQUE5/E5_3_1_eficiencia_12decadas/PROTOCOLO_E5.3-1_PREREGISTRO.md`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/BATERIA_ENFOQUE5/E5_3_1_eficiencia_12decadas/E5_3_1_motor.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/BATERIA_ENFOQUE5/E5_3_1_eficiencia_12decadas/E5_3_1_motor_stderr.log`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/BATERIA_ENFOQUE5/E5_3_1_eficiencia_12decadas/E5_3_1_resultado_crudo.json`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/BATERIA_ENFOQUE5/E5_3_1_eficiencia_12decadas/E5_3_1_analisis.py`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/BATERIA_ENFOQUE5/E5_3_1_eficiencia_12decadas/E5_3_1_resumen.json`
- `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/BATERIA_ENFOQUE5/E5_3_1_eficiencia_12decadas/E5_3_1_REPORTE.md` (este archivo)

**Tiempo de corrida:** motor = 1215.46 s (≈20 min 15 s) para las 169 celdas × 20 semillas ×
2 (real+null), N=200. Análisis (agregación) < 1 s.
