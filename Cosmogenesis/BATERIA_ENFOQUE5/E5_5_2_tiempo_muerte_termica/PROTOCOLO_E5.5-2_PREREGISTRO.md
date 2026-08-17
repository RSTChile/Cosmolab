# PROTOCOLO E5.5-2 — Tiempo a la muerte térmica: ¿cuánto tarda X→0 según ε y r?

**Congelado (pre-registro):** 2026-07-24 20:48 (America/Santiago, UTC-4)
**Ejecutor:** CC (agente E5.5-2, batería Enfoque 5, corrida en paralelo con 29 agentes más)
**Base de código leída (NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Documento madre:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, sección "E5.5-2"
**Protocolos hermanos leídos en disco (para heredar la definición de X):**
`BATERIA_ENFOQUE5/E5_1_1_supervivencia_exergia/PROTOCOLO_E5.1-1_PREREGISTRO.md` y
`E5_1_1_engine.py`. **E5.5-1 NO estaba en disco al momento de este pre-registro**
(no existe carpeta `E5_5_1_*` ni protocolo con ese nombre en `BATERIA_ENFOQUE5/`) — se
verificó explícitamente antes de escribir esto; por tanto la definición de X se hereda
de E5.1-1 únicamente (documento madre confirma que ambos deben ser consistentes con la
misma fórmula base `persistencia()` de cs074_rcruz.py, así que no hay conflicto posible).

Este documento se escribe y congela ANTES de tocar el motor. Cualquier desviación
respecto de lo aquí escrito se reporta como desviación explícita, no se edita
retroactivamente (T3).

---

## 1. Pregunta

¿Cuántos pasos tarda la exergía X en caer bajo un umbral pre-registrado de "muerte
térmica" (estructura no explotable, equilibrio uniforme para fines prácticos), en
función de ε (amplitud de la diferencia inicial) y r=H/D (razón expansión/difusión),
en la zona donde r **no alcanza a congelar** (r pequeño, expansión insuficiente para
aislar regiones antes de que la difusión las lave)?

## 2. Modelo (idéntico en física a cs074_rcruz.py y a E5.1-1, reimplementado bajo este
prefijo — NO se importa el archivo base, se reimplementa fielmente, mismo método que
usó E5.1-1)

- Campo escalar φ en anillo de N=200 sitios (mismo N que producción de la base y que
  E5.1-1).
- Fondo φ=1 + ε·(suma de 5 armónicos con fase aleatoria, normalizada a desviación
  estándar 1) — idéntico a `campo_inicial`.
- **Difusión:** relajación local hacia el promedio de vecinos, SOLO por aristas vivas
  (idéntica fórmula a `paso_difusion`: nuevo = φ + 0.5·(media_vecinos−φ)).
- **Expansión:** cada arista viva se corta con probabilidad de Bernoulli H por paso
  (idéntica a `paso_expansion`, corrección r-cruz: Bernoulli por arista, válido también
  para H·N≪1). El corte es IRREVERSIBLE (una arista cortada no se reconecta).
- **D** = fracción de contraste (desviación estándar) borrada en UN paso de difusión
  pura (H=0), MEDIDA del propio campo, igual que `medir_D`. **Optimización de cómputo
  declarada:** D es una propiedad del operador lineal de difusión sobre la topología
  (anillo, N=200), NO depende de ε (la difusión es lineal: la razón (c0−c1)/c0 es
  invariante a la amplitud). Se mide UNA vez, promediando 20 semillas a ε=1e-3 de
  referencia, y se reutiliza para todo el barrido — evita medir D redundantemente en
  cada una de las 10 celdas de ε (ahorro de cómputo documentado, no cambia el método:
  ya usa exactamente `medir_D` sobre datos reales).
- **r** = H/D es el eje del barrido en la zona sub-congelamiento. H se fija como
  H = min(r_target·D, 1.0) — D se mide primero, H emerge de esa medida (T0/T1: nada
  puesto a mano).
- **Ruido dinámico (T7):** en CADA paso de evolución se añade al campo ruido gaussiano
  de amplitud NOISE_REL·ε (NOISE_REL=0.02, idéntico valor que E5.1-1, constante
  congelada aquí, jamás ajustada a posteriori). Con ε=0 el ruido dinámico es
  exactamente 0.

## 3. Axiomas declarados (E1/E2, NO física real)

- **E1 (conservación declarada):** E_decl = Σφ se declara conservada por el mecanismo
  de difusión. Se AUDITA (no se fuerza): se mide E_decl al inicio y en el instante en
  que se detecta la muerte térmica (o al llegar al tope de pasos), y se reporta la
  deriva relativa por celda. No se renormaliza el campo.
- **E2 (redistribución por expansión):** la expansión no crea energía; aísla regiones
  y con ello puede congelar gradientes que la difusión de otro modo borraría. Marco
  interpretativo de por qué r creciente debería alargar (o impedir) la muerte térmica.
  No se fuerza en el motor.

## 4. Observable — Exergía X (heredada literal de E5.1-1 / `persistencia()` de la base)

    c = corr(φ, roll(φ,1))   (autocorrelación a un paso; clip a ≥0)
    v = Var(φ_actual) / Var(φ_inicial)     (fracción de varianza retenida)
    X = c · v

Idéntica fórmula a `persistencia()` en cs074_rcruz.py y a `exergia()` en
`E5_1_1_engine.py`. Se evalúa a lo largo del tiempo (no solo al final): en cada paso
verificado se recalcula X(t) contra Var(φ_inicial) fija de esa corrida.

**Umbral de muerte térmica (X_UMBRAL) — pre-registrado, NO inventado a mano:**
X_UMBRAL = **0.05**, el mismo valor P_LAVADO=0.05 que cs074_rcruz.py y E5.1-1 ya usan
como criterio de "lavado" (estructura ya no distinguible del equilibrio uniforme, al
5% de significancia de forma-persistencia). Se reutiliza el umbral EXISTENTE en el
código base — no se elige un número nuevo para este experimento (T1).

**Tiempo a la muerte térmica (observable primario):**
t_muerte(ε,r,semilla) = primer paso t (múltiplo de CHECK_EVERY=50, igual cadencia que
`medir_pasos_lavado` de la base) en que X(t) < X_UMBRAL. Si no se cruza dentro del tope
de pasos simulados (MAX_STEPS, ver §5), la corrida se marca **censurada** (t_muerte =
None, se registra t_muerte_censurado = MAX_STEPS) — esto es precisamente lo que permite
detectar divergencia (T5: se reporta la curva completa, incluida la censura, nunca solo
un gate binario).

**Juez ≠ observable (T2):** el veredicto de "diverge / no diverge" se basa en la
fracción de semillas censuradas por celda (r,ε) y en la curva t_muerte(r) completa, no
en un único número.

## 5. Barrido (ε sobredimensionado por regla del director; r acotado a la zona
sub-congelamiento por el diseño de ESTE experimento — así lo pide el documento madre
para E5.5-2)

| Eje | Rango | Puntos |
|---|---|---|
| ε | {0} ∪ {1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0} | 11 (control 0 + 10 décadas 1e-9…1, "regla de oro" del director) |
| r = H/D | {0} ∪ logspace(1e-3, 1, 13) | 14 (control r=0 "sin expansión en absoluto" — ancla, igual que E5.1-1 — + 13 puntos log en la zona [1e-3…1] pre-registrada por el documento madre para E5.5-2, ~4.3 pts/década, cruzando el umbral de congelamiento conocido ~0.1 con margen a ambos lados: incluye explícitamente puntos cerca de 0.056, 0.1, 0.178) |
| semillas | 0..11 | 12 (mínimo pre-registrado por el documento madre para E5.5-2: "≥12 semillas") |
| ruido dinámico | NOISE_REL=0.02·ε, aplicado cada paso | fijo, declarado (T7) |
| N | 200 (fijo, igual que producción de la base y E5.1-1) | — |
| D | medido UNA vez (20 semillas, ε=1e-3), reusado en todo el barrido (ver §2) | — |
| CHECK_EVERY | 50 pasos (igual cadencia que `medir_pasos_lavado` de la base) | — |
| MAX_STEPS | calibrado de forma medida (NO puesto a mano): MAX_STEPS = ceil(mediana_lavado_r0_ref × CAP_MULT), con mediana_lavado_r0_ref = mediana de t_muerte a r=0, ε=1e-3 sobre 16 semillas (mismo método que `medir_pasos_lavado` de la base) y **CAP_MULT=10** (constante de diseño declarada aquí, ANTES de correr — margen de un orden de magnitud sobre el lavado puro, suficiente para separar "lento pero finito" de "censurado/divergente" sin costo computacional prohibitivo; NO se ajusta después de ver resultados) | — |

Nota sobre ε=0: por construcción, contraste0=Var(φ_inicial)=0 ⇒ X≡0 desde t=0 (no hay
diferencia que evolucionar). Se registra analíticamente t_muerte=0 en todas las celdas
de ε=0 sin simular (ahorro de cómputo trivial y honesto: la fórmula de `persistencia()`
ya devuelve 0 cuando contraste0≤0, idéntico en la base).

Combinaciones simuladas = 10 (ε>0) × 14 (r) × 12 semillas = 1680 corridas de evolución
(+ 14 filas analíticas de ε=0, + calibración de D y de MAX_STEPS).

**Optimización de cómputo declarada (no cambia el método, sólo la implementación):**
las 12 semillas de cada celda (ε,r) se evolucionan EN PARALELO como un arreglo (12,N)
con numpy (misma física por fila, draws de aleatoriedad independientes por fila —
`rng.random((12,N))` genera un flujo independiente por entrada); la corrida de la celda
se detiene en cuanto TODAS las 12 semillas cruzan X<X_UMBRAL o se alcanza MAX_STEPS.
Esto es aritméticamente idéntico a 12 corridas 1D separadas, solo evita el overhead de
bucle Python por semilla — no se cambia la física ni el umbral.

## 6. NULL

**Ninguno (declarado explícitamente por el documento madre para E5.5-2: "NULL: —").**
Este experimento no es una prueba de falsación REAL-vs-NULL; es una caracterización de
t_muerte(ε,r). La verificación de validez viene de los controles internos (§7) y de la
auditoría de conservación E1.

## 7. PASS / criterios de lectura (congelados antes de correr)

- **ε=0 → t_muerte=0 a todo r** (ya muerto desde el inicio, control trivial).
- **r=0 (sin expansión) → t_muerte finito y del orden de la calibración de lavado**
  (control de validez: reproduce el comportamiento ya medido en cs074_rcruz.py y
  E5.1-1 para difusión pura).
- **r≪0.1 (dentro de la zona [1e-3…1]) → t_muerte finito, similar al de r=0** (la
  expansión es demasiado débil para alterar el lavado dentro del horizonte de MAX_STEPS).
- **PASS central (pre-registrado en el documento madre):** t_muerte(ε,r) **debe
  divergir** (no cruzar X<X_UMBRAL dentro de MAX_STEPS → censura sistemática) al cruzar
  r por encima del umbral de congelamiento ya conocido (~0.1, documentado en
  `cs074_rcruz_produccion_resultado.json`: a r_target=0.1, P_real=0.62 con z=4.9 frente
  a NULL, ya claramente separado del régimen no congelado). Si la fracción de semillas
  censuradas NO sube con r cerca de ese cruce, es un **dato en contra** — se reporta tal
  cual, sin reinterpretar.
- **Dispersión entre semillas:** se reporta t_muerte por semilla individual (no solo la
  media/mediana), y su dispersión, para cada celda (ε,r).
- Si CUALQUIERA de estos falla, se reporta como tal — no se reinterpreta ni se ajusta el
  motor después de ver los datos (T3, regla de ejecución #1).

## 8. Verificación cruzada (regla de ejecución #4)

1. Segundo observable/método reportado en paralelo: `std_ratio` = φ.std()/φ_inicial.std()
   en el instante de la muerte térmica (o al tope), para separar si la censura viene
   del factor de autocorrelación c o ya está en la varianza cruda.
2. Auditoría de conservación E1 (deriva de Σφ inicio→instante de muerte/tope) reportada
   en cada fila para revisión externa en disco (JSON crudo).
3. Consistencia contra el umbral de congelamiento ya reportado por cs074_rcruz.py
   (r_target=0.1 con z=4.9) y contra la definición de X de E5.1-1 (misma fórmula) —
   auditable por quien no escribió este motor.

## 9. Salidas

- `E5_5_2_engine.py` — motor (este archivo, escrito DESPUÉS de este pre-registro).
- `E5_5_2_resultado_crudo.json` — filas completas del barrido (r, eps, H, D, MAX_STEPS,
  t_muerte por semilla, fracción censurada, std_ratio en el instante final,
  deriva_E_decl, frac_exp).
- `E5_5_2_run.log` — timestamps de inicio/fin y progreso.

## 10. Trampas explícitamente evitadas

- T0: nada discreto puesto a mano — N, X_UMBRAL (=P_LAVADO heredado) y MAX_STEPS vienen
  de la base/medición, no de ajustar-para-que-cruce.
- T1: NOISE_REL=0.02 (idéntico a E5.1-1) y CAP_MULT=10 son constantes de diseño
  declaradas ANTES de correr, no ajustadas para acercar el resultado a nada esperado.
- T2: t_muerte (observable) es un cruce de umbral fijo; el veredicto lo da la curva
  completa + fracción censurada, no un único número.
- T3: este documento se congela antes de escribir el motor; cualquier desviación se
  reporta, no se edita retroactivamente.
- T4: no aplica (este experimento no tiene NULL, declarado explícitamente en el
  documento madre).
- T5: se reporta la curva t_muerte(r) entera para cada ε, incluida la censura, no un
  gate binario.
- T6: se audita conservación E1 en cada celda (inicio → instante de muerte/tope).
- T7: ruido dinámico presente en cada paso, además de 12 semillas.

No se corre nada del motor hasta que este archivo esté guardado en disco.
