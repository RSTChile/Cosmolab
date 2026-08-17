# PROTOCOLO E5.6-3 — Invariancia de X a la escala del sistema (N barrido amplio)

**Congelado (pre-registro):** 2026-07-24 20:5x (America/Santiago, UTC-4)
**Ejecutor:** CC (agente E5.6-3, batería Enfoque 5, corrida en paralelo con 29 agentes más)
**Base de código leída (NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Definición de X reutilizada de:** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/BATERIA_ENFOQUE5/E5_1_1_supervivencia_exergia/E5_1_1_engine.py`
(agente E5.1-1, ya en disco — misma fórmula, mismo motor físico, SIN reimportar código, reimplementado aquí bajo mi prefijo).
**Documento madre:** `../../BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, sección "E5.6-3"

Este documento se escribe y congela ANTES de tocar el motor de producción. Cualquier
desviación respecto de lo aquí escrito se reporta como desviación explícita, no se edita
retroactivamente (T3). Se permitió, ANTES de este congelado, una fase de **sondeo de
factibilidad computacional** (medir D(N) y tiempos de lavado en N=64,128,256,512 con topes
de tiempo) para dimensionar el barrido de forma realista — esa fase NO toca el observable X
ni ningún criterio PASS, solo el presupuesto de cómputo. Se documenta en la sección 6.

---

## 1. Pregunta

¿La exergía por unidad de sitio (X/N) es una cantidad **intensiva** (no depende del tamaño
del sistema N), o es un artefacto de tamaño finito / efecto de borde que decae o crece con
N?

## 2. Modelo (idéntico en física a cs074_rcruz.py y a E5.1-1, motor propio bajo mi prefijo)

Mismo campo escalar φ en anillo de **N sitios variable** (el eje que se barre aquí,
E5.1-1 lo fijó en N=200):
- Fondo φ=1 + perturbación ε·(suma de 5 armónicos con fase aleatoria, normalizada a
  desviación estándar 1) — idéntico a `campo_inicial` de la base y de E5.1-1.
- **Difusión:** relajación local hacia el promedio de vecinos, SOLO por aristas vivas
  (idéntica fórmula `paso_difusion`: nuevo = φ + 0.5·(media_vecinos−φ)).
- **Expansión:** cada arista viva se corta con probabilidad de Bernoulli H por paso
  (idéntica a `paso_expansion`).
- **D(N)** = fracción de contraste borrada en UN paso de difusión pura (H=0), MEDIDA del
  propio campo, para cada N (no puesta a mano) — igual método que `medir_D`.
- **r = H/D** es la razón expansión/difusión; H = min(r_target·D(N), 1.0).
- **Ruido dinámico (T7):** en cada paso de evolución se añade al campo ruido gaussiano de
  amplitud NOISE_REL·ε (NOISE_REL=0.02, idéntico a E5.1-1, constante congelada, jamás
  ajustada a posteriori).

## 3. Axiomas declarados (E1/E2, NO física real) — idénticos a E5.1-1

- **E1:** presupuesto declarado E_decl=Σφ, se AUDITA (deriva inicio→fin reportada por fila,
  no se fuerza ni se renormaliza).
- **E2:** la expansión redistribuye E latente en exergía, no la crea (marco interpretativo,
  no mecanismo verificado aquí).

## 4. Observable — Exergía X (REUTILIZADA verbatim de E5.1-1, NO redefinida)

**X_final** = fracción de la energía (estructura) capaz de hacer trabajo:

    c = corr(φ, roll(φ,1))   (autocorrelación a un paso; clip a ≥0)
    v = Var(φ_final) / Var(φ_inicial)
    X_final = c · v

Esta es la MISMA fórmula que `exergia()` en `E5_1_1_engine.py` (a su vez `persistencia()`
en `cs074_rcruz.py`). No se cambia ni un signo. Se reutiliza precisamente porque el
propósito de E5.6-3 es preguntar si ESTA MISMA cantidad, medida igual, escala con N.

**Observable primario de ESTE experimento: X/N** (exergía dividida por el tamaño del
sistema — candidato a cantidad intensiva) y, en paralelo, **X_final crudo** (para poder
distinguir "X/N constante" de "X constante" de "X/N con ley de potencia X/N ~ N^α").

**Juez ≠ observable (T2):** el veredicto (intensiva / no intensiva) se basa en el
comportamiento de la curva X/N vs N completa (ajuste de pendiente log-log α = d log(X/N)/d
log N, comparado contra α≈0 dentro de la dispersión entre semillas), no en un número único.

## 5. Barrido de N (regla de oro: sobredimensionado)

**N ∈ {64, 128, 256, 512, 1024, 2048, 4096}** — 7 puntos, **6 duplicaciones exactas**
desde 64 (64→128→256→512→1024→2048→4096), cubriendo dos órdenes de magnitud, tal como
pide el documento madre y la instrucción del director ("64→4096 aprox").

## 6. Presupuesto de cómputo — por qué el diseño NO es un grid uniforme (documentado, T1)

**Hallazgo de factibilidad (sondeo previo, con topes de tiempo, ANTES de congelar el
diseño):** D(N) mide la fracción de varianza borrada en UN paso de difusión. Se midió:

| N | D medido | D·N² (control de escala) |
|---|---|---|
| 64 | 8.086e-3 | 33.1 |
| 128 | 2.049e-3 | 33.6 |
| 256 | 5.139e-4 | 33.7 |

D(N) escala como **N⁻² casi perfecto** (razón ≈3.95-4.00 por duplicación — la difusión
local es un proceso de relajación de modo largo, cuyo tiempo característico crece con el
cuadrado de la longitud de onda, aquí ∝N). El "tiempo de lavado" (pasos hasta X<P_LAVADO
con H=0) también escala ∝N² (medido con tope de tiempo: t_hit=543, 2172, 8688 para
N=64,128,256 — factor ≈4.0 por duplicación, y el producto **t_hit·D ≈ 4.39–4.46 es
prácticamente CONSTANTE** entre N — validación empírica directa de que el proceso es
auto-similar bajo el reescalamiento de N: el mismo número de "tiempos de difusión locales"
lava el campo, sea cual sea N).

**Consecuencia de costo:** correr el MISMO número relativo de pasos (pasos ∝ 1/D(N) ∝ N²)
con un costo por paso ∝N (arreglos vectorizados) da un costo TOTAL por corrida ∝N²
(empíricamente, no N³, porque a N pequeño domina el overhead fijo de las llamadas numpy,
no el tamaño del arreglo — medido: N=64→0.49s/corrida, N=4096→~1550s/corrida completa a
pasos_fijo(N), con velocidad de pasos/s casi constante ~1300-2000). Un grid uniforme
(mismas celdas ε×r×semillas en los 7 N) costaría **~40-70 horas** de cómputo, dominado
casi enteramente por N=2048 y N=4096. Esto se declara ANTES de correr (T1/T3): el diseño
se adapta reduciendo el número de CELDAS (ε,r) en el tramo caro, **nunca reduciendo
semillas por debajo del mínimo pre-registrado de 8** para ninguna celda estocástica, y
**nunca cambiando la fórmula de X, el método de cálculo de D, ni el umbral de lavado**
(P_LAVADO=0.05, MARGEN=1.15, idénticos a E5.1-1).

**Cálculo de pasos_fijo(N) sin fuerza bruta cara:** se calibra K_ref = mediana(t_hit·D)
midiendo por fuerza bruta SOLO en N∈{64,128,256} (barato, <10s total), y se deriva
pasos_fijo(N) = ceil(K_ref·MARGEN_LAVADO / D(N)) para TODO N — incluidos 1024, 2048, 4096
— usando la D(N) medida (barata, un solo paso) en vez de re-simular el lavado completo por
fuerza bruta en los N caros. Esto es la MISMA lógica de linealidad que usa E5.1-1 para
generalizar la calibración de pasos a través de ε (aquí se generaliza a través de N,
validado empíricamente arriba, con error <2% entre las 3 réplicas de calibración).

**Diseño de celdas (ε, r) resultante — MISMA celda "señal" (ε=1.0, r=100.0) presente en
TODOS los N (curva primaria X/N vs N con cobertura completa, 8 semillas):**

| Tramo de N | Celdas (ε, r) | Semillas |
|---|---|---|
| N ≤ 1024 (64,128,256,512,1024) | (1.0, 0.0) · (1.0, 1.0) · (1.0, 100.0) · (1.0, 1000.0) · (0.0, 0.0) — 5 celdas | 8 cada una |
| N ∈ {2048, 4096} | (1.0, 100.0) — señal principal, 8 semillas · (1.0, 0.0) — control de lavado, 2 semillas · (0.0, 0.0) — control cero, 1 semilla | ver detalle |

Justificación de la reducción en el tramo caro (declarada, no post-hoc):
- **(0.0, 0.0)** con ε=0 es **determinista por construcción del código**: `campo_inicial`
  con ε≤0 no invoca al rng en absoluto (retorna el fondo constante sin usar `rng.uniform`)
  y el ruido dinámico es NOISE_REL·ε=0. Se confirma empíricamente con 8 semillas en el
  tramo barato (N≤1024) que el resultado es idéntico entre semillas (std=0 exacto); en el
  tramo caro se corre 1 semilla como verificación puntual de que la propiedad se sostiene
  también a N grande — no una repetición de estadística que no puede variar.
- **(1.0, 0.0)** (control de lavado, sin expansión) se reduce a 2 semillas en el tramo caro
  como verificación puntual barata de que la difusión sigue lavando el campo a N grande
  (ya está garantizado en parte por construcción: pasos_fijo(N) se calibra PRECISAMENTE
  para que r=0 lave a P<0.05, así que este control es, por diseño de la calibración,
  parcialmente tautológico — se reporta igual, con menos semillas, como chequeo de
  sanidad, no como hallazgo estadístico independiente).
- La celda de señal **(1.0, 100.0)** SÍ mantiene las 8 semillas completas en TODO N, porque
  es la única cantidad estocástica no trivial cuya dispersión entre semillas es
  necesaria para el criterio PASS (sección 8).

**Total de corridas de evolución:** 5×8×5 (tramo barato) + (8+2+1)×2 (tramo caro) =
200 + 22 = **222 corridas de evolución** (cada una da REAL+NULL sin costo adicional, ver
sección 7).

**Costo estimado ANTES de correr (documentado, T1, no se ajusta después):**

| N | pasos_fijo(N) (derivado) | s/corrida (medido) | corridas | tiempo estimado |
|---|---|---|---|---|
| 64 | ~624 | ~0.49 | 40 | ~20 s |
| 128 | ~2498 | ~1.6 | 40 | ~64 s |
| 256 | ~9991 | ~5.35 | 40 | ~214 s |
| 512 | ~39965 | ~20.7 | 40 | ~830 s |
| 1024 | ~159859 | ~77.2 | 40 | ~3088 s |
| 2048 | ~639237 | ~346.5 | 11 | ~3812 s |
| 4096 | ~2557747 | ~1550 | 11 | ~17050 s |

**Total estimado: ≈25,080 s ≈ 6.97 horas.** Autorizado explícitamente por el director
("Cómputo largo autorizado... documenta el costo real"). El tiempo REAL medido se reporta
en el resultado crudo (no se trunca ni se recorta el barrido de N si el tiempo real excede
esta estimación — solo se documenta la desviación).

## 7. NULL

Permutar φ al final de la evolución (idéntico a E5.1-1 y a `evolucionar(...,null=True)` de
la base), MISMA semilla y MISMA H/ε que su pareja REAL — optimización de cómputo
documentada (mismo argumento que E5.1-1: la trayectoria REAL y NULL son idénticas hasta el
paso final, difieren solo en el barajado, así que se deriva del mismo φ_final con una
permutación independiente, mitad del cómputo). Reportado por celda: X_real vs X_null y
z-score.

## 8. PASS / criterios de lectura (congelados antes de correr)

- **ε=0 → X_final=0 a todo N** (control determinista, no hay estructura inicial que
  sobreviva a ningún tamaño).
- **Intensividad (hipótesis a testear, NO forzada):** X/N para la celda de señal
  (ε=1.0, r=100.0) se ajusta en log-log contra N: pendiente α = d log(X/N)/d log(N).
  - **PASS "intensiva":** |α| < 0.1 (prácticamente plano) dentro de la dispersión entre
    semillas (IC bootstrap o ±1 SEM de la pendiente cruza 0).
  - **PASS "no intensiva, ley de potencia reportada":** si α se aleja claramente de 0
    (fuera de la dispersión entre semillas), se reporta α, el signo, y si X (no X/N) es lo
    que en realidad es constante (α≈−1 en X/N ⟺ X constante, efecto de borde puro) o si
    ninguna de las dos lo es.
  - **Ambos resultados son hallazgo válido** (regla del documento madre): no hay ajuste
    posterior de umbral para forzar "PASS=intensiva".
- **NULL debe permanecer bajo/plano** en todo N (T4): si el NULL también escala con N de
  forma similar a REAL, el hallazgo de (no)intensividad sería artefacto de la métrica, no
  de la física — se reporta explícitamente.
- **Conservación E1** auditada por fila (deriva de Σφ), reportada, no forzada.
- Si cualquiera de estos criterios no puede evaluarse con los datos (p.ej. dispersión
  insuficiente), se reporta como tal — no se reinterpreta ni se ajusta el motor después de
  ver los datos (T3).

## 9. Verificación cruzada (regla de ejecución #4)

1. NULL propio (permutación), por celda y N.
2. Segundo observable/método: `std_ratio` = φ.std()/√Var(φ_inicial) (varianza retenida
   sola, sin el factor de autocorrelación), reportado en paralelo — ídem E5.1-1.
3. Auditoría de conservación E1 (deriva de Σφ inicio→fin) en cada fila, para revisión
   externa en disco (JSON crudo).
4. Validación de la derivación de pasos_fijo(N): se reporta K_ref medido (mediana y
   dispersión entre N=64,128,256) y se compara contra el N donde SÍ hay lavado por fuerza
   bruta (los 3 baratos), documentando el error relativo de la extrapolación.

## 10. Salidas

- `E5_6_3_engine.py` — motor (este archivo, escrito DESPUÉS de este pre-registro).
- `E5_6_3_resultado_crudo.json` — filas completas del barrido (N, eps, r_target, H, D(N),
  pasos_fijo(N), X_real por semilla, X_null por semilla, z, std_ratio, deriva_E, frac_exp,
  tiempo de corrida real por celda).
- `E5_6_3_run.log` — log de ejecución con timestamps de inicio/fin por N (para reportar
  costo real).

## 11. Trampas explícitamente evitadas

- T0: N y pasos_fijo(N) vienen de medición (D(N)) + calibración (K_ref), no puestos a mano.
- T1: NOISE_REL=0.02, P_LAVADO=0.05, MARGEN_LAVADO=1.15 idénticos a E5.1-1, declarados
  aquí, no ajustados después de ver resultados. La reducción de celdas en el tramo caro de
  N está declarada y justificada ANTES de correr (esta misma sección 6), no es un ajuste
  posterior a un resultado incómodo.
- T2: X_final (observable) es fórmula fija reutilizada; el veredicto de intensividad lo da
  la pendiente log-log de la curva completa contra NULL, no un observable ad-hoc.
- T3: pre-registro congelado antes del motor; desviaciones se reportan, no se editan.
- T4: NULL corrido en cada celda para descartar que la métrica misma module con N.
- T5: se reporta la curva X/N(N) entera (7 puntos) para la celda de señal, no un gate
  binario a un solo N.
- T6: se audita conservación E1 cada corrida (inicio/fin).
- T7: ruido dinámico presente en cada paso, además de semillas independientes.

No se corre nada del motor hasta que este archivo esté guardado en disco.

---

## ADENDA — Arreglo 2 (ruido calibrado) + Arreglo 3 (definición común de exergía), 2026-07-25

**No se edita el texto original arriba (T3): esta sección se agrega, no reemplaza.**
Congelada ANTES de lanzar la re-corrida final (mandato: `INSTRUCCION_recorrer_5_definicion_comun_PARA_CC.md`).

### Contexto: este experimento DETECTÓ el bug de ruido

E5.6-3 fue el experimento que reveló, en su propia corrida original (`E5_6_3_resultado_crudo.json`,
ya en disco, conservado sin tocar), que el mecanismo de ruido dinámico por paso (regla T7 del
pre-registro, sección 2: "ruido gaussiano de amplitud NOISE_REL·ε") estaba mal calibrado: la
amplitud era **constante por paso** sin importar cuántos pasos corriera la simulación
(`noise_amp = NOISE_REL * eps` en `evolucionar_con_ruido()`). Como `pasos_fijo(N)` crece con
1/D(N) (~N², sección 6), la varianza acumulada del ruido crecía sin control con N. Evidencia en
el JSON original, celda de señal (ε=1.0, r=100.0):

| N | z (real vs NULL) | deriva_E_max (conservación) |
|---|---|---|
| 64 | 6.79 | 11.6% |
| 128 | 7.50 | 27.1% |
| 256 | 3.05 | 42.0% |
| 512 | 0.95 | 41.6% |
| 1024 | 1.99 | 49.1% |
| 2048 | **-0.06** | 55.3% |
| 4096 | **-0.28** | 71.1% |

A N≥2048 el NULL deja de discriminar (\|z\|≈0) y la deriva de conservación E1 llega a >70% —
el hallazgo insignia del experimento (pendiente log-log α de X/N vs N, que dio **α=-0.567
(SE≈0.122)**, lejos del criterio de intensividad \|α\|<0.1) queda contaminado por este artefacto
en el tramo caro de N, exactamente donde más importa (N=2048, 4096 son la mitad del rango
barrido en órdenes de magnitud).

### ARREGLO 2 — ruido calibrado (módulo compartido, no reimplementado)

Se importa `ruido_por_paso()` de `../_ruido_calibrado.py` (módulo aditivo, verificado
independientemente en `../_verificacion_arreglo2_N_sweep.py`, corrido hoy: a N=2048, ruido viejo
da z=0.04/deriva 44%, ruido nuevo da z=98.5/deriva 0.03%). En `evolucionar_con_ruido()` de este
motor, la única línea que cambia es:

    noise_amp = NOISE_REL * eps                              # ANTES (bug)
    noise_amp = ruido_por_paso(NOISE_REL, eps, pasos)         # AHORA (Arreglo 2)

que escala la amplitud por paso con `1/sqrt(pasos_fijo)`, dejando la varianza acumulada total
sobre `pasos_fijo` pasos constante (≈(NOISE_REL·ε)², independiente de N). No cambia NOISE_REL
(sigue siendo 0.02, la misma constante congelada del pre-registro original), ni ningún otro
parámetro de diseño (N_LIST, celdas, semillas quedan intactos — T1/T3).

### ARREGLO 3 — definición común de exergía (módulo compartido, no reimplementado)

Se importa `exergia_X()` de `../_observables_homologadas.py` (Xh = (1/N)·Σ(φᵢ-1)², definición
canónica de la batería, verbatim de E5.2-2). Se calcula **en paralelo** al observable original
de este experimento (`X = c·v`, familia persistencia, heredado de E5.1-1) — ninguno reemplaza al
otro. `corrida_celda()` ahora devuelve `Xh_real`/`Xh_null` además de `X_real`/`X_null`, sobre el
MISMO φ_f/φ_null; `correr_celda()` agrega `z_h`, `Xh_real_mean/std`, `Xh_null_mean/std` y
`Xh_por_N_real_mean` (análogo a `X_por_N_real_mean`); `main()` repite el ajuste log-log de
intensividad también sobre `Xh_por_N`, guardado como `analisis_intensividad_celda_senal_Xh`
junto al ya existente `analisis_intensividad_celda_senal` (que con esta corrida queda recalculado
con el ruido YA arreglado).

### Propiedad estructural detectada ANTES de correr (declarada aquí, T3): `z_h` es
### trivialmente 0 en TODA celda, por construcción matemática — no es un bug

Al escribir el smoke test de esta adenda se detectó que `Xh` (exergía canónica) es una función
**simétrica/permutation-invariant** de φ: `Xh(φ) = (1/N)·Σᵢ(φᵢ-1)²` no depende del orden espacial
de los sitios, solo del multiconjunto de valores. El NULL de este experimento (heredado de
E5.1-1, sección 7 del pre-registro) se construye permutando el propio φ_final
(`phi_null = rng.permutation(phi_f)`). Como la permutación no cambia el multiconjunto de
valores, **Xh_real y Xh_null son numéricamente idénticos en cada corrida individual**, así que
`z_h = 0.0` exacto (hasta punto flotante) en cada fila, para cualquier N y cualquier celda —
verificado con un smoke test a N=64 antes de lanzar la corrida completa. Esto NO afecta el
análisis de intensividad primario (`analisis_intensividad_celda_senal_Xh`, que compara
`Xh_por_N_real_mean` REAL contra N, no contra NULL), pero significa que `z_h` no sirve, con esta
definición y este NULL, como criterio de "REAL vs NULL discrimina" — se reporta igual por
completitud y para que quede documentado el porqué, no se omite el campo ni se cambia el diseño
del NULL (eso sería un cambio de diseño no autorizado, fuera del alcance de esta adenda).

### Guardado de detalle (para reconstrucción futura sin re-simular)

Cada corrida individual guarda `sum_phi_real/null` y `sum_phi2_real/null` (junto con N, permite
reconstruir E canónica y X canónica de cualquier definición futura). Para la celda de señal
(ε=1.0, r=100.0) — la más importante, con cobertura completa en los 7 N — se guarda además el
array `φ` crudo completo (real y NULL) de cada semilla (`phi_real_arr_per_seed`,
`phi_null_arr_per_seed`); las demás celdas (controles) solo guardan las sumas, por tamaño de
archivo.

### Qué NO cambia (T1)

N_LIST, N_CALIB_BRUTE, CELDAS_BARATO, CELDAS_CARO, número de semillas por celda, P_LAVADO,
MARGEN_LAVADO, NOISE_REL, el método de cálculo de D(N)/K_ref/pasos_fijo(N), y la definición de
`X` (persistencia, c·v) permanecen exactamente como en el pre-registro original. Solo cambia (1)
la calibración de la amplitud de ruido por paso y (2) la adición de la medición canónica Xh en
paralelo.

### Archivo de referencia pre-arreglos

El resultado crudo original (con el bug de ruido activo, arreglo 3 ausente) se copia sin tocar a
`E5_6_3_resultado_crudo_DEFINICION_VIEJA_pre_ARREGLOS_2_3.json` antes de sobrescribir
`E5_6_3_resultado_crudo.json` con la nueva corrida.

### Pregunta que responde esta re-corrida

(a) ¿La pendiente log-log de X/N vs N (definición vieja, persistencia) mejora una vez arreglado
SOLO el ruido? (b) ¿La pendiente log-log de Xh/N vs N (definición canónica) es distinta de la de
X/N con el ruido ya arreglado — aislando el efecto de la DEFINICIÓN sola, con el ruido ya
controlado? Ambas se reportan con su error estándar; el criterio de intensividad sigue siendo
\|α\|<0.1, idéntico al pre-registro original. No se adjudica ni cierra el experimento en esta
adenda — el veredicto se compara honestamente contra el original una vez corrido.

No se corre la re-simulación completa hasta que esta adenda esté guardada en disco.
