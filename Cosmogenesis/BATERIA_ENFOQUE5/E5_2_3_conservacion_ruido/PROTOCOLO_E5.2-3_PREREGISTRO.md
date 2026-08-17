# E5.2-3 · Conservación bajo forzamiento estocástico: ¿el ruido rompe el balance?

**Pre-registro fechado.** Redactado ANTES de correr el motor: 2026-07-24T20:44:00Z (UTC).
Regla T3: si algo falla, se reporta — no se edita esto después.

**Estado de E5.2-1 al momento de escribir esto:** directorio
`BATERIA_ENFOQUE5/E5_2_1_balance_deriva/` existe pero está VACÍO (verificado dos veces,
20:40 y 20:44 UTC) — su protocolo aún no está en disco. No hay definición de E_total que
heredar de él.

**Definición reutilizada de E5.2-2 (sí está en disco):** `E5_2_2_anticorrelacion_X_S/
E5_2_2_PROTOCOLO_PREREGISTRO.md`, ya pre-registrado por otro agente del mismo TEMA 2, define:

```
X(t) = (1/N) · Σ_i (φ_i(t) − 1)²        [exergía, momento cuadrático vs equilibrio FIJO φ_eq=1]
```

φ_eq=1 es el fondo uniforme con el que arranca `campo_inicial` (`fondo = np.ones(N)`) antes de
sumar la perturbación ε·pert. Esta definición se REUTILIZA VERBATIM aquí (misma fórmula, mismo
N=200, mismo φ_eq=1) para que X(t) sea comparable entre E5.2-2 y E5.2-3. Como E5.2-1 no estaba
disponible, esta es la definición más cercana ya pre-registrada en la familia TEMA 2 y se declara
como la base de comparabilidad, tal como pide el encargo.

---

## 1. Definición de E_total (propia, extiende X(t) de E5.2-2 a un presupuesto conservado)

Identidad algebraica EXACTA (no es física, es álgebra — vale para cualquier φ):

```
E_total(t) := (1/N) · Σ_i φ_i(t)²  =  1 + 2·D̄(t) + X(t)
```
donde `D̄(t) = mean(φ(t)) − 1` (desplazamiento del promedio espacial respecto al equilibrio) y
`X(t)` es EXACTAMENTE la fórmula de E5.2-2 de arriba. Verificación algebraica:
`(1/N)Σ(1+δ_i)² = 1 + (2/N)Σδ_i + (1/N)Σδ_i²` con `δ_i=φ_i−1`, `(2/N)Σδ_i = 2·D̄`,
`(1/N)Σδ_i² = X`. □

En t=0, `campo_inicial` construye la perturbación con `pert -= pert.mean()` (media exactamente
cero por construcción) ⟹ `D̄(0) = 0` exacto ⟹ `E_total(0) = 1 + X(0)`.

**Axioma E1 (declarado, no derivado):** E_total(t) se declara CONSTANTE = E_total(0). Esto es
una elección de diseño de la batería (el rango 0 del documento madre), no una propiedad
garantizada del código: se PONE A PRUEBA, no se asume.

**Axioma E2 (declarado):** la dinámica (difusión + expansión + ruido) debe REDISTRIBUIR entre
`D̄` y `X` sin cambiar `E_total` — nunca crear ni destruir presupuesto.

**deriva(t) = |E_total(t) − E_total(0)| / E_total(0)** — observable primario, igual estructura
que pide E5.2-1/E5.2-3 en el documento madre.

**Nota de honestidad (T2, observable≠juez):** X(t) ya es observable independiente de E5.2-2;
D̄(t) es una cantidad NUEVA (media espacial, no usada como tal en E5.2-2). El juez de E5.2-3 es
`deriva(t)` contra el umbral, no X ni D̄ por separado.

**Resultado algebraico previo verificado (por qué esta E_total es la elección correcta para
este código, no arbitraria):** en el anillo COMPLETAMENTE conectado (H=0, ninguna arista
cortada), `paso_difusion` promedia cada nodo con sus 2 vecinos vivos; por simetría del anillo
`Σ_i media_i = Σ_i φ_i` exactamente (cada φ_j aparece dos veces del lado derecho, una vez como
vecino izquierdo y otra como vecino derecho de nodos distintos) ⟹ la difusión pura preserva
`mean(φ)` EXACTAMENTE, es decir preserva `D̄` exactamente, mientras homogeneiza (reduce) `X`.
Por eso, SIN ruido y SIN expansión (H=0), se espera deriva(t)≈0 en punto flotante — ese es el
control de referencia contra el que se mide el efecto del ruido.

---

## 2. Mecanismos de ruido dinámico (T7: perturbación dinámica, no solo semilla)

Aplicados DESPUÉS de cada paso de difusión+expansión (física exacta de `cs074_rcruz.py`,
importado sin editar: `campo_inicial`, `paso_difusion`, `paso_expansion`, `medir_D`).

### (a) ADITIVO — forzamiento externo tipo baño
```
φ_i ← φ_i + amplitud · η_i,   η_i ~ N(0,1) i.i.d. por nodo y por paso
```
Representa una fuerza estocástica externa genuina (no ligada a la topología de aristas vivas).
Predicción analítica pre-registrada: como `mean(η)` tiene esperanza 0 pero varianza no nula,
`D̄(t)` hace una CAMINATA ALEATORIA (no un sesgo sistemático): `E[D̄(t)]=0` para todo t (es
martingala), pero `SD[D̄(t)] ≈ (amplitud/√N)·√t`. Consecuencia: `E[E_total(t)] ≈ E_total(0)`
en promedio sobre semillas (E1 se cumple EN EXPECTACIÓN), pero la dispersión típica de
`deriva(t)` entre semillas debe crecer ~`amplitud·√t` (E1 falla POR REALIZACIÓN, no en el
agregado). Esta es la predicción que T6 pide falsar/confirmar, no solo narrar.

### (b) INTERCAMBIO — redistribución interna (control positivo del axioma E2)
```
para cada arista viva i↔(i+1): δ_i ~ N(0,amplitud²); φ_i ← φ_i − δ_i ; φ_{i+1} ← φ_{i+1} + δ_i
```
(vectorizado: `φ ← φ − δ·activo + roll(δ·activo, 1)`). Transfiere masa SOLO entre vecinos
conectados por arista viva (misma restricción topológica que la difusión). Por construcción,
`Σφ_i` (y por tanto `mean(φ)`, y por tanto `D̄`) se conserva EXACTO a precisión de punto
flotante en CADA transferencia, sin importar la amplitud. Predicción pre-registrada: `deriva(t)`
bajo este mecanismo debe permanecer en el piso numérico (~1e-14..1e-10) para TODA amplitud —
es el control positivo que opera el axioma E2 tal como está escrito ("redistribuye, no crea").
Si (b) también derivara con la amplitud, la conclusión sería que el "balance" no es la cantidad
correcta o hay un bug — se reportaría, no se escondería.

---

## 3. Barrido (sobredimensionado, regla del director)

- **N = 200** (misma escala que `cs074_rcruz.py modo=produccion` y que E5.2-2).
- **ε = 1e-3** fijo (perturbación inicial moderada; no afecta E_total(0)=1+X(0) salvo vía X(0),
  documentado — el foco de este experimento es el ruido dinámico, no el barrido de ε, que ya
  cubren E5.1-x/E5.3-x).
- **amplitud_ruido** ∈ {0} ∪ logspace(1e-6, 1, 19) — 20 valores, 6 décadas tal como pide la
  regla de oro, más el control exacto amplitud=0.
- **pasos_max = 100 000** ("pasos_largos"), con E_total registrado en checkpoints log-espaciados
  dentro de la MISMA trayectoria: t ∈ {10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,
  100000} — 13 puntos. Esto da la curva deriva(t) completa por corrida sin recorridas
  redundantes, y localiza (T6) el paso exacto donde se cruza el umbral, si se cruza.
- **Semillas:** 12 por celda (amplitud, mecanismo), semillas base 20000..20011.
- **Mecanismos:** {aditivo, intercambio} — grid principal con **H=0** (sin expansión, aísla la
  pregunta "¿el ruido rompe el balance?" del efecto de la expansión, que ya tiene su propio
  hueco en E5.2-1/E5.2-4/E5.4-x).
- **Grid principal:** 20 amplitudes × 12 semillas × 2 mecanismos = 480 corridas × 13
  checkpoints.
- **Suplementario (robustez a H≠0):** mismo grid de amplitud reducido a 7 puntos
  (logspace(1e-6,1,7)), pasos_max=10 000, checkpoints {10,30,100,300,1000,3000,10000},
  12 semillas, mecanismo=aditivo únicamente, con H = H(r=1) = min(1·D_medido, 1) (D medido con
  `medir_D(N,ε,seed)` del propio módulo base, r=1 = régimen de transición, el más relevante del
  documento madre). Objetivo: ¿la propia expansión (corte de aristas, que rompe la simetría del
  anillo usada en la prueba algebraica de la Sección 1) ya introduce deriva por sí sola, incluso
  sin ruido, y el ruido la agrava o no?
- **Testigo de trayectoria fina (localización extra, T6):** para el mecanismo (a) a la amplitud
  máxima (1.0) y la amplitud mínima no nula (1e-6), 1 semilla cada una, registrar deriva(t) en
  TODOS los pasos (no solo checkpoints) para poder señalar el paso exacto de cruce de umbral si
  existe.

## 4. NULL

No aplica un NULL de barajado clásico aquí (T2 no compara contra permutación espacial; la
pregunta es de conservación temporal, no de estructura espacial). El control es interno y
doble: (i) amplitud=0 (sin ruido, física base sola) como línea de base, y (ii) mecanismo (b)
como control positivo de "redistribución sin creación" contra el mismo axioma que (a) pone a
prueba. Ambos están pre-registrados en la Sección 2, no post-hoc.

## 5. Juez y PASS (congelado antes de correr)

- **Umbral de deriva:** UMBRAL = 1e-6, tomado literalmente de la regla del documento madre para
  E5.2-1 ("deriva < 1e-6 en toda la corrida"), reutilizado aquí como el mismo estándar de
  presupuesto para toda la familia TEMA 2.
- **PASS estricto (por celda amplitud×mecanismo, sobre las 12 semillas, en TODO checkpoint,
  T5 curva entera no gate binario):** deriva(t) < UMBRAL para todo t.
- **T6 — si se rompe, LOCALIZAR:** para cada celda que no pasa, se reporta (a) el primer
  checkpoint t donde deriva cruza UMBRAL (mediana entre semillas) y (b) la amplitud mínima a la
  que eso ocurre para el pasos_max dado. NO se promedia sobre toda la corrida para esconder el
  cruce.
- **Cruce con la predicción analítica (T2, segundo método):** para el mecanismo (a), se compara
  la dispersión ENTRE SEMILLAS de deriva(t) contra la predicción `amplitud·√(t)/√N` (ver Sección
  2) — si coincide en orden de magnitud, el "rompimiento" (cuando ocurra) es el efecto esperado
  de un baño estocástico genuino, no un bug; si no coincide, se reporta la discrepancia sin
  ajustar la teoría al dato.
- **Veredicto de la pregunta central:** "el ruido redistribuye, no crea" (E2) se considera
  SOSTENIDO si (b) pasa en TODA amplitud Y (a) pasa SOLO en el agregado/promedio de semillas
  (no por realización) — es decir, el axioma es cierto EN EXPECTACIÓN para forzamiento aditivo
  genuino, pero cada trayectoria individual sí se aleja del balance declarado. Si (b) también
  falla, el veredicto es NEGATIVO (el balance declarado no es robusto a ningún ruido dinámico,
  ni siquiera al diseñado para conservar por construcción) y se reporta así, sin suavizar.

## 6. Qué se entrega crudo a CS

- Curva deriva vs amplitud_ruido (mediana y banda entre semillas) para (a) y (b), a
  pasos_max=100000 (checkpoint final) y en checkpoints intermedios.
- Curvas deriva(t) completas (13 checkpoints) para al menos 3 amplitudes representativas
  (mínima no nula, mediana, máxima) por mecanismo.
- Trayectorias testigo paso-a-paso (Sección 3) con el punto exacto de cruce de umbral marcado.
- Comparación dispersión-medida vs dispersión-predicha (mecanismo a).
- Grid suplementario H=H(r=1): deriva vs amplitud, comparado contra su propio control
  amplitud=0.
- Dispersión entre semillas reportada explícitamente (std, no solo media) en cada punto.
- Veredicto sin suavizar por celda y agregado.

## 7. Archivos

- Motor: `E5_2_3_motor.py` (importa `cs074_rcruz.py` sin editarlo).
- Resultados crudos: `E5_2_3_resultado.json`.
- Log de corrida: `E5_2_3_run.log`.
- Este pre-registro: `PROTOCOLO_E5.2-3_PREREGISTRO.md`.

**Firmado (pre-registro, antes de correr):** agente E5.2-3, 2026-07-24T20:44:00Z UTC.

---

## ADENDA — calibración de cómputo (ANTES de correr el grid completo, DESPUÉS de escribir
## la Sección 3 de arriba)

Timestamp: 2026-07-24T20:52:00Z UTC. Se midió el throughput real del motor (no el juicio,
no el umbral, no el diseño): a N=200, un paso (difusión+expansión+ruido+medición) tarda
≈275 µs. `pasos_max=100000` para las 480 corridas del grid principal ⟹ ≈3.7 h de cómputo,
excesivo para el turno. Se ajusta **pasos_max del grid principal y de los testigos de
100000 → 20000** (el grid suplementario ya usaba 10000, sin cambio). `CHECKPOINTS_MAIN` se
recorta a `[10,20,50,100,200,500,1000,2000,5000,10000,20000]` (se quitan 50000 y 100000).
20000 pasos sigue dentro del rango "pasos_largos" que pide el documento madre (su propio
rango de referencia para E5.2-1/E5.2-3 es 1e2..1e5) y las pruebas de humo (Sección de abajo,
hechas ANTES de este ajuste, a 4 amplitudes de 8 posibles y solo 1 semilla) ya muestran la
dinámica completa (saturación a amplitudes altas, crecimiento lento a amplitudes bajas)
resuelta bien antes de 5000 pasos — no se ajustó mirando el resultado del grid completo
(no existía aún), solo el tiempo de reloj de una corrida de calibración. El umbral
(UMBRAL_DERIVA=1e-6), el juez (deriva(t)<umbral en todo checkpoint) y el diseño de los dos
mecanismos NO se tocaron.

**Hallazgo de la prueba de humo que corrige una predicción de la Sección 2 (T3: se reporta,
no se esconde):** la predicción pre-registrada decía que el mecanismo (b) intercambio
debía mantener deriva en el piso numérico para TODA amplitud. La prueba de humo lo refuta:
D̄(t) SÍ se mantiene exactamente en 0 (confirmado, ~1e-16, como predije), pero X(t) —y por
tanto E_total, que es 1+2D̄+X— SÍ crece con la amplitud incluso bajo (b), porque cualquier
intercambio aleatorio de masa entre vecinos aumenta la suma de cuadrados (⟨Δ(φ_i²+φ_j²)⟩ =
2δ² > 0 por transferencia, en valor esperado) aunque conserve la suma lineal exactamente.
Confundí "conserva Σφ" con "conserva Σφ²=E_total" — son cantidades distintas. Se deja la
Sección 2 tal cual (T3, no se edita el pre-registro original) y se reporta esta corrección
aquí, de forma visible, junto con los resultados finales.

