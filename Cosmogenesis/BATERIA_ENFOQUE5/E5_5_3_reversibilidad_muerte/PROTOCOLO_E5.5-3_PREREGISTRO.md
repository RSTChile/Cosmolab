# PROTOCOLO E5.5-3 — Reversibilidad de la muerte térmica: ¿re-inyectar ε la revierte?

**Congelado (pre-registro):** 2026-07-24 20:45 (America/Santiago, UTC-4)
**Ejecutor:** CC (agente E5.5-3, batería Enfoque 5, corrida en paralelo con 29 agentes más)
**Base de código leída (NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Documento madre:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, sección "E5.5-3"
**Definición de X reutilizada de:** `BATERIA_ENFOQUE5/E5_1_1_supervivencia_exergia/PROTOCOLO_E5.1-1_PREREGISTRO.md`
(única pieza de Tema 5/1 en disco al momento de congelar esto; E5.5-1/E5.5-2 no estaban
en disco — no hay nada más que reutilizar de esos).

Este documento se escribe y congela ANTES de tocar el motor. Cualquier desviación
respecto de lo aquí escrito se reporta como desviación explícita, no se edita
retroactivamente (T3).

---

## ADENDA — Corrección post-hoc (ARREGLO 1), 2026-07-25

**No se edita el texto original arriba (T3): esta sección se agrega, no reemplaza.**

El director (Alexis) detectó, en lenguaje simple ("¿de dónde sale energía extra si no
se crea ni se destruye?"), que el mecanismo de re-inyección descrito en §4.6 y §8
(`φ_reinyectado = φ_muerto + amplitud_reinyectada · patrón_nuevo`) **mete energía nueva
al sistema** — aunque el §8 original ya lo admitía honestamente ("aquí Σφ SÍ cambia"),
el diseño violaba el axioma de presupuesto cerrado del proyecto (ninguna energía entra
desde afuera de la singularidad declarada). Además, y más grave para el observable de
ESTE experimento: el patrón inyectado tiene media≈0 (Σφ casi no cambia) pero **std=1
por construcción**, así que la variable que realmente se manufactura de la nada es
Σφ² / la varianza — exactamente la cantidad de la que depende X (v = Var(φ)/Var(φ₀)).
La auditoría de Σφ del §8 original NO detecta este tipo de creación de energía, porque
mide la cantidad equivocada para este observable.

**Corrección aplicada (ver el motor corregido `E5_5_3_engine.py`, función
`redistribuir_energia()`):** la re-inyección deja de ser
una SUMA de un patrón externo y pasa a ser una **permutación parcial** de los valores
que YA existen en φ_muerto: se eligen `frac · N` sitios al azar y se reordenan sus
valores entre sí (mismo multiconjunto de valores, otro orden espacial). Esto conserva
Σφ Y Σφ² **exactamente** (hasta error de punto flotante) por construcción algebraica,
no por verificación externa — no hay forma de que "cree" energía porque nunca importa
valores de fuera del propio φ_muerto.

El eje `amplitud_reinyectada` (logspace 1e-6…1) se reinterpreta como
`fraccion_redistribuida` (mismo rango numérico [1e-6, 1], mismo número de puntos —
la grilla pre-registrada no cambia de forma, solo de significado: ya no es "cuánta
perturbación externa se mete" sino "qué fracción del campo ya existente se reordena").

**Pregunta corregida que responde esta versión:** ¿se puede recuperar exergía
redistribuyendo lo que YA hay en el sistema, sin meter nada de afuera? Predicción
pre-registrada para ESTA corrección (declarada aquí, ANTES de re-correr — T3): dado que
el estado muerto es casi homogéneo (todos los φᵢ ≈ el mismo valor), reordenar valores
casi idénticos entre sí debería producir un revival ≈0 en casi todo el rango de
`fraccion_redistribuida` — si aun así X revive de forma clara y separada del NULL, es
un hallazgo genuino y sorprendente que hay que reportar tal cual, no descartar.

Este experimento se re-corre desde cero con esta corrección (mandato explícito del
director: "Este experimento (E5.5-3) se vuelve a correr desde cero con esta
corrección"). El resultado crudo anterior (buggy) se conserva en disco como
`E5_5_3_resultado_crudo_PRE_ARREGLO1_BUGGY.json` para auditoría, no se borra.

---

## ADENDA 2 — Definición común de X (ARREGLO 3), 2026-07-25

**No se edita el texto original arriba (T3): esta sección se agrega.**

Por decisión del director (`INSTRUCCION_recorrer_5_definicion_comun_PARA_CC.md`): además
del Arreglo 1 (arriba), este motor ahora calcula, EN PARALELO, la definición canónica de
exergía (`Xh`, de `BATERIA_ENFOQUE5/_observables_homologadas.py`,
`Xh = (1/N)·Σ(φᵢ-1)²`) junto a la definición vieja heredada de E5.1-1 (`X`, familia
persistencia, `c·v`). Nada más cambia: mismo diseño, mismo barrido de 8 momentos × 13
fracciones × 16 semillas, mismo NULL, misma física con el Arreglo 1 ya aplicado. El
motor guarda ambas curvas (`curva_X_post` y `curva_Xh_post`) y además el φ crudo
(`phi_post_iny`, `phi_final`) en cada fila, para que la comparación vieja-vs-nueva se
pueda auditar directamente y para que si hace falta una tercera definición en el futuro
no haya que re-correr. El resultado con Arreglo 1 pero SIN esta adenda (definición
vieja únicamente) se conserva como
`E5_5_3_resultado_crudo_ARREGLO1_SOLO_pre_ARREGLO3.json`.

---

## 1. Pregunta

Una vez que el sistema alcanzó la muerte térmica (equilibrio, X≈0), ¿una nueva
perturbación ε re-inyectada revive la exergía, o el equilibrio es un estado absorbente
que ya no responde? ¿Importa cuánto tiempo lleva muerto (momento de re-inyección) o
cuán grande es la re-inyección (amplitud)?

## 2. Modelo (heredado de cs074_rcruz.py; motor propio bajo mi prefijo)

Mismas primitivas exactas que la base, importadas sin modificación desde
`cs074_rcruz.py` (`campo_inicial`, `paso_difusion`, `persistencia`):

- Campo escalar φ en un anillo de N=200 sitios. Fondo φ=1 + perturbación
  ε·(suma de 5 armónicos con fase aleatoria, normalizada a desviación estándar 1).
- **Difusión:** relajación local hacia el promedio de vecinos (`paso_difusion`,
  idéntica fórmula: nuevo = φ + 0.5·(media_vecinos−φ)).
- **Régimen de equilibrio bajo prueba: H=0 (r=0), SIN expansión, en TODA la corrida.**
  Justificación: la "muerte térmica" de la Tema 5 (E=máx, X=0, equilibrio uniforme
  real) es el límite de difusión pura sobre un grafo SIEMPRE conectado — con H>0 el
  mecanismo de Tema 1 (aislamiento por corte de aristas) puede CONGELAR estructura
  para siempre, lo cual es un canal de supervivencia YA estudiado por E5.1-1/E5.1-2,
  no el "morir de verdad y ver si revive" que pide E5.5-3. Con H=0, cada paso de
  `paso_difusion` sobre el anillo completo preserva exactamente Σφ (promedio local
  lineal sobre grafo regular conectado) — se AUDITA esta conservación (E1), no se
  asume.
- No se llama `paso_expansion` (con H=0 es un no-operación exacta; se omite para
  simplicidad, `activo` queda fijo en todo-True).

## 3. Definición de X (idéntica a E5.1-1, reutilizada tal cual)

    c = corr(φ, roll(φ,1))         (autocorrelación a un paso; clip a ≥0)
    v = Var(φ_final) / Var(φ_inicial_de_la_corrida_completa)
    X = c · v

`φ_inicial_de_la_corrida_completa` es el campo en t=0 de CADA corrida completa (antes
de que empiece a morir), NO el campo en el momento de re-inyección. Esto mide "cuánta
de la capacidad ORIGINAL se recuperó", que es la pregunta de E5.5-3 ("cuánta X se
recupera"). Se reporta también un segundo observable (`std_ratio`, ver §7) para
verificación cruzada (regla de ejecución #4).

`persistencia()` se importa literalmente de `cs074_rcruz.py` — no se reimplementa
(evita transcripción divergente).

## 4. Protocolo experimental (tres fases por corrida)

**Fase A — vida y muerte (una vez por semilla):**
1. `φ(0)` con ε_inicial=1.0 (declarado, NO barrido — ver §9), H=0.
2. Evolucionar paso a paso, midiendo X(t) cada `CHECK_EVERY=50` pasos.
3. Calibración (una vez, con semillas piloto, ANTES del barrido principal, método
   idéntico en espíritu a `medir_pasos_lavado` de la base pero con el umbral propio
   de X): medir `t_muerte` = primer t (múltiplo de 50) donde X(t) < `THR_MUERTE=0.02`
   Y se mantiene por debajo en las siguientes 3 verificaciones (evita declarar muerte
   por una fluctuación transitoria).
4. Cada corrida real evoluciona hasta `T_MAX = 32 × t_muerte_cal` (el mayor momento de
   re-inyección de la grilla, ver §5) para tener el campo disponible en todos los
   momentos de la grilla, GUARDANDO snapshots de φ exactamente en los pasos de la
   grilla de momentos (§5). Determinista dado H=0 (sin aleatoriedad en la evolución
   misma, solo en la condición inicial) → un solo pase hacia adelante por semilla basta.

**Fase B — re-inyección (por semilla × momento × amplitud):**
5. Se toma el snapshot φ_muerto guardado en el paso `t_muerte_cal × momento_factor`.
6. Se genera una perturbación NUEVA (misma familia funcional que `campo_inicial`: 5
   armónicos, fases aleatorias FRESCAS — no las mismas de t=0 — normalizada a std=1),
   escalada por `amplitud_reinyectada`, y se SUMA a φ_muerto:
   `φ_reinyectado = φ_muerto + amplitud_reinyectada · patrón_nuevo`.
7. Se mide `X_boost` = X inmediatamente tras la inyección (antes de evolucionar más).

**Fase C — ¿la re-inyección también muere? (por semilla × momento × amplitud):**
8. Se continúa evolucionando con H=0 por `post_pasos = 2 × t_muerte_cal` pasos más,
   registrando X en checkpoints fraccionarios `{0, 0.05, 0.1, 0.25, 0.5, 1.0} ×
   post_pasos` (curva completa, T5 — no un gate binario).
9. **NULL de este experimento:** para cada (semilla, momento), UNA corrida gemela SIN
   re-inyección (amplitud=0): se deja φ_muerto seguir evolucionando bajo la misma H=0
   por los mismos `post_pasos`, midiendo X en los mismos checkpoints. Confirma que el
   equilibrio es estable (no revive solo) y da la línea base contra la que se compara
   la re-inyección real.

## 5. Barrido (sobredimensionado, regla del director + regla de oro explícita del
encargo)

| Eje | Rango | Puntos |
|---|---|---|
| **amplitud_reinyectada** | logspace(1e-6, 1, 13) — 6 décadas, 2 puntos/década (regla de oro literal del encargo: [1e-6…1]) | 13 |
| **momento_reinyeccion** (tiempo de permanencia MUERTO antes de inyectar, en múltiplos de t_muerte_cal) | {0, 0.5, 1, 2, 4, 8, 16, 32} × t_muerte_cal | 8 |
| **semillas** | 0..15 | 16 (≥12 pedido) |
| ε_inicial (Fase A, fijo, no barrido — ver §9) | 1.0 | — |
| H (fijo, régimen bajo prueba) | 0 | — |
| N | 200 | — |
| post_pasos (fijo, derivado de calibración) | 2 × t_muerte_cal | — |

Total combinaciones reales: 13 amplitudes × 8 momentos × 16 semillas = **1664 corridas
de re-inyección**, cada una con 6 checkpoints post-inyección en Fase C.
Más el NULL: 8 momentos × 16 semillas = **128 corridas NULL** (sin re-inyección),
mismos checkpoints.
Más 16 corridas de Fase A (una por semilla, hasta T_MAX) que generan los snapshots.
**Total: 1664 + 128 + 16 = 1808 trayectorias evolutivas.**

## 6. NULL

Descrito en §4.9: continuar SIN re-inyección desde el mismo φ_muerto, misma semilla,
mismo momento, mismos pasos post-momento. Si el NULL también "revive" (X sube sin
inyección), el hallazgo real se descarta como artefacto numérico/de la métrica, no de
la física (T4: el NULL debe morder).

## 7. Segundo observable / método (regla de ejecución #4, cruce)

`std_ratio = φ.std() / φ_inicial(t=0).std()` — la varianza retenida CRUDA, sin el
factor de autocorrelación c. Reportado en paralelo a X en cada checkpoint. Si el
hallazgo de "revive/no revive" solo aparece en X pero no en std_ratio (o viceversa),
se reporta la discrepancia explícitamente — no se elige el observable que "se ve
mejor" después de correr (T2).

## 8. Auditoría de conservación (E1, T6)

En cada corrida se registra Σφ en: t=0 (Fase A), momento de inyección (antes y
después de sumar la perturbación — aquí Σφ SÍ cambia, porque la perturbación nueva no
tiene media exactamente cero por construcción numérica; se reporta esa inyección de
"masa" explícitamente, no se oculta) y al final de Fase C. Se reporta la deriva de Σφ
durante la evolución pura (que si H=0 y el grafo es regular, debe conservarse exacto
salvo error de punto flotante) por separado de la inyección deliberada de amplitud.

## 9. Constantes declaradas ANTES de correr (T1 — nada tocado después de ver datos)

- `EPS_INICIAL = 1.0` — no es un barrido de este experimento (el barrido de ε_inicial
  ya es E5.1-1/E5.5-1); se fija al valor MÁS ALTO permitido en la base (estructura
  inicial inequívoca) para que la muerte observada sea indiscutible y no un artefacto
  de partir casi-plano.
- `THR_MUERTE = 0.02` — umbral de X para declarar "muerto" (declarado antes de
  calibrar; NO ajustado después de ver la curva de decaimiento).
- `CHECK_EVERY = 50`, `MAX_CAL_STEPS = 200000` — iguales a los defaults de
  `medir_pasos_lavado` en la base (reutilizados por consistencia, no inventados).
- `MOMENTO_FACTORS = [0, 0.5, 1, 2, 4, 8, 16, 32]` — grilla de "tiempo muerto antes de
  inyectar", elegida para cubrir desde "recién llegado al umbral" hasta "muerto 32×
  más tiempo del que tardó en morir" (sobredimensionado en el eje temporal también,
  en el espíritu de la regla del director, aunque el encargo solo exige
  sobredimensionar la amplitud explícitamente).
- `CHECKPOINTS_POST = [0, 0.05, 0.1, 0.25, 0.5, 1.0]` (fracciones de post_pasos).
- Semilla de patrón de re-inyección: `rng_reiny = np.random.default_rng(90_000 +
  1000*seed + 10*momento_idx + amp_idx)` — determinista, declarado, no tocado tras
  correr.

## 10. Predicción pre-registrada (T3 — se compara después, no se ajusta)

Bajo difusión lineal pura (H=0), el estado "muerto" es φ≈constante (uniforme) en todo
el anillo, indistinguible — para la dinámica futura — de partir de cero con ese mismo
nivel de base. La predicción ingenua es que el sistema es MARKOVIANO: la re-inyección
debería comportarse igual que una condición inicial fresca de esa amplitud,
INDEPENDIENTE del momento (no hay "memoria" de haber estado muerto), y el X recuperado
debería depender solo de `amplitud_reinyectada` (curva similar a X_final(ε) de
E5.1-1/E5.5-1 restringida a r=0), no de `momento_reinyeccion`. Si el momento SÍ importa
(p. ej. degradación numérica acumulada, deriva de Σφ, etc.), es un hallazgo y se
reporta como tal — no se fuerza a encajar en la predicción.

## 11. PASS / criterios de lectura (congelados antes de correr)

- **NULL debe quedarse en X≈0** en todos los momentos y semillas (T4) — confirma que
  la muerte es estable y que cualquier subida de X en las corridas reales es atribuible
  a la re-inyección, no a ruido numérico o revival espontáneo.
- **Absorbente (si se observa):** X_boost y/o X tras post_pasos permanecen ≈0
  (indistinguibles del NULL) para TODAS las amplitudes probadas — el equilibrio no
  responde a la perturbación.
- **Recuperable (si se observa):** X_boost sube con `amplitud_reinyectada` de forma
  medible y separado del NULL, y (Fase C) decae de nuevo hacia 0 con una escala de
  tiempo — se reporta si esa escala coincide con la de la Fase A original (mismo
  mecanismo, sin memoria) o es distinta (el sistema "recuerda" haber muerto).
- **Punto de no retorno (si existe):** se reporta si `momento_reinyeccion` grande
  reduce sistemáticamente la capacidad de recuperación (X_boost más bajo, o decaimiento
  más rápido) respecto a momento=0 — de lo contrario, se reporta que el momento NO
  importa (memoryless), tal como predice §10, y eso también es un resultado válido.
- Se reporta la curva ENTERA (X vs amplitud, por momento, con dispersión entre
  semillas) — T5, sin colapsar a un solo número ni a un gate binario.
- Ningún resultado se auto-adjudica como veredicto final de la batería — se entrega
  crudo a CS (regla de ejecución #9).

## 12. Salidas

- `E5_5_3_engine.py` — motor (escrito DESPUÉS de este pre-registro).
- `E5_5_3_resultado_crudo.json` — calibración, snapshots meta, filas completas
  (semilla, momento, amplitud, X_boost, X_post por checkpoint, std_ratio, Σφ en cada
  etapa, filas NULL).
- Este archivo (`PROTOCOLO_E5.5-3_PREREGISTRO.md`).

## 13. Trampas explícitamente evitadas

- T0: N, THR_MUERTE, CHECK_EVERY vienen de la base o se declaran aquí, no se ajustan
  para que "cruce".
- T1: EPS_INICIAL, THR_MUERTE, grillas — todo declarado en §9 antes de calibrar/correr.
- T2: X es una fórmula fija (idéntica a E5.1-1); el veredicto lo da la curva completa
  vs NULL, no el observable mismo. Segundo observable (`std_ratio`) para cruce.
- T3: predicción declarada en §10 ANTES de correr; si falla, se reporta como falla.
- T4: NULL explícito (sin re-inyección) en cada celda (momento×semilla).
- T5: curva entera X(amplitud) por momento, y X(t) dentro de Fase C — no gate binario.
- T6: Σφ auditado en cada etapa (t=0, pre/post-inyección, final de Fase C).
- T7: la perturbación de re-inyección usa fases aleatorias FRESCAS por cada
  (semilla, momento, amplitud) — no es la misma mancha reciclada; y el barrido de
  16 semillas × condición inicial ya trae variabilidad dinámica de fondo.

No se corre nada del motor hasta que este archivo esté guardado en disco.
