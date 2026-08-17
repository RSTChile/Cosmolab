# PROTOCOLO E5.2-4 — Presupuesto por componentes: ¿en qué se reparte E a lo largo del barrido?

**Congelado (pre-registro):** 2026-07-24 20:42 (America/Santiago, UTC-4) / 2026-07-25T00:42:24Z
**Ejecutor:** CC (agente E5.2-4, batería Enfoque 5, corrida en paralelo con 29 agentes más)
**Base de código leída (NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Documento madre:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, sección "E5.2-4"

Este documento se escribe y congela ANTES de tocar el motor. Cualquier desviación
respecto de lo aquí escrito se reporta como desviación explícita, no se edita
retroactivamente (T3).

**Verificación de reuso (regla del director):** se buscó en disco
`Cosmogenesis/BATERIA_ENFOQUE5/E5_2_1_balance_deriva/` — el directorio existe pero está
VACÍO (`find` sin resultados, confirmado dos veces, la segunda vez tras un aviso explícito
del coordinador de revisar de nuevo). Ningún otro directorio de la batería (revisado por
completo con `find … -maxdepth 2`) contiene una definición de E_total previa a reutilizar
para este tema. Por tanto se define E_total aquí, siguiendo el MISMO principio declarado
en la sección 0 del documento madre (E1: presupuesto declarado conservado; E2: la
expansión redistribuye, no crea). Si E5.2-1 aparece en disco más tarde con una definición
distinta, se reporta la discrepancia a CS — no se reconcilia unilateralmente.

---

## 1. Pregunta

A lo largo del barrido de r = H/D (expansión/difusión), ¿cómo se reparte el presupuesto
de energía inicial E_total entre tres destinos — exergía todavía útil (X), energía ya
degradada (disipada irreversiblemente) y energía ligada (atrapada en estructura congelada)
— y varían esas tres fracciones de forma medible con r?

## 2. Modelo (heredado de cs074_rcruz.py, motor propio bajo mi prefijo, SOLO importado)

Campo escalar φ en un anillo de N=200 sitios (misma física que CS074-rcruz, funciones
`campo_inicial`, `paso_difusion`, `paso_expansion` importadas SIN modificar el archivo):
- Fondo φ=1 + perturbación ε·(suma de 5 armónicos con fase aleatoria, normalizada a
  desviación estándar 1).
- **Difusión:** relajación local hacia el promedio de vecinos, SOLO por aristas vivas.
- **Expansión:** cada arista viva se corta con probabilidad de Bernoulli H por paso;
  H≥1 corta todas; H=0 no corta ninguna. Monotónica: una arista cortada NUNCA se
  restaura (ratchet, igual que la base).
- **D** = fracción de contraste borrada en un paso de difusión pura (H=0), MEDIDA
  (`medir_D` de la base), no puesta a mano.
- **r** = H/D, eje primario. H = min(r_target·D, 1.0).

## 3. Definición de E_total y las tres componentes (declaración ANTES de correr)

**E_total** (presupuesto declarado, axioma E1) = Var(φ₀) — varianza del campo en t=0
(la "capacidad" inicial disponible, análoga al c0² usado como ancla en toda la base:
`contraste0 = phi.std()` en `evolucionar`). Se fija UNA vez por corrida, al inicio, y NUNCA
se renormaliza — si algo no cuadra contra este ancla, es hallazgo, no se oculta.

En cada paso, la difusión (`paso_difusion`) es la ÚNICA operación que cambia φ — la
expansión (`paso_expansion`) sólo modifica qué aristas están vivas, nunca toca φ. Esto
permite una contabilidad EXACTA por partición:

- **X (exergía / útil):** al final de la corrida, el campo queda partido en componentes
  conexas por las aristas que sobrevivieron (arcos del anillo). Dentro de cada componente
  viva, la varianza interna (ANOVA: Var_within = Σ_c n_c·Var(φ|c) / N) sigue siendo
  potencialmente relajable — la difusión interna del arco AÚN podría seguir borrándola si
  la corrida continuara. Definimos **X = Var_within(φ_final, activo_final)**: estructura
  que retiene capacidad de cambio futuro (interpretación operacional de "capaz de hacer
  trabajo", igual espíritu que la fórmula de E5.1-1: desviación del equilibrio que no ha
  terminado de relajar).
- **Ligada (atrapada en estructura):** la diferencia ENTRE los niveles medios de las
  componentes aisladas ya NO puede mezclarse nunca (las aristas que las separan no se
  restauran). Definimos **Ligada = Var_between(φ_final, activo_final) = Σ_c n_c·(media_c −
  media_global)² / N** — variación estructural permanentemente congelada por el
  aislamiento (E2: la expansión no crea esta cantidad, la REDISTRIBUYE fuera del alcance
  de la difusión).
- **Degradada:** lo que realmente se perdió de la varianza total desde t=0 —
  **Degradada = E_total − Var(φ_final)** — disipación irreversible por mezcla (entropía
  de la difusión). Se mide de DOS formas independientes en el código (T2, ejecución #4):
  (a) residuo algebraico E_total − X − Ligada; (b) suma telescópica paso a paso de
  Var(φ_antes) − Var(φ_después) en CADA paso de difusión, acumulada durante toda la
  corrida. Ambas deben coincidir a precisión de punto flotante — la comparación entre
  ambas es la auditoría T6 de conservación en cada corrida (no sólo al final: se registra
  en checkpoints intermedios a lo largo de la corrida, no un único punto).

**Identidad exacta por construcción:** X + Ligada + Degradada = Var_within + Var_between +
(E_total − Var_final) = Var_final + E_total − Var_final = **E_total**, para TODO t
(incluye t=0, donde Ligada=0, X=E_total, Degradada=0, consistente con "nada se ha
repartido todavía").

Esta NO es la misma definición de E1 usada en E5.1-1 (que audita Σφ, conservación del
promedio bajo difusión lineal) — aquí E1 se aplica a la varianza como presupuesto de
"capacidad organizada", que es la cantidad relevante para exergía/degradación/atrapamiento.
Se declara la diferencia explícitamente para que CS no la lea como inconsistencia entre
agentes.

## 4. NULL

**Barajado (igual principio que toda la batería):** al final de la corrida, se permuta φ
(`rng.permutation`, MISMA función que `evolucionar(..., null=True)` de la base) mientras se
CONSERVA el patrón de aristas vivas (`activo_final`) tal cual quedó de la física real. Se
recalculan X_null, Ligada_null, Degradada_null sobre el φ barajado con las MISMAS
componentes conexas. Esto prueba si la partición X/Ligada realmente refleja estructura
espacial real (la correlación entre "qué valores" y "qué arco quedó aislado con ellos") o
si un reparto equivalente aparecería solo por azar de agrupar valores en arcos de tamaños
dados. **No se re-corre la física para el NULL** — se deriva del mismo estado final, ahorro
válido porque el NULL de esta batería siempre se define como barajado post-hoc, nunca como
una física distinta.

## 5. Barrido (sobredimensionado, regla del director)

| Eje | Rango | Puntos |
|---|---|---|
| r = H/D | logspace(1e-3, 1e3) | 13 (6 décadas, medio-décadas) |
| ε | {1e-3, 1e-2, 0.1, 0.3, 1.0} | 5 |
| semillas | 0..11 | 12 (mínimo pedido) |
| N | 200 (fijo, igual que modo "produccion" de la base) | — |
| pasos | calibrado UNA vez con `medir_pasos_lavado(N=200, eps=1e-3, semillas=6)` × margen 1.15, reusado en toda la grilla (mismo método que `pasos_fijo` de la base) | — |
| checkpoints de conservación | 7 puntos uniformes en pasos (incluye t final) | — |

Total combinaciones (r,ε) = 13×5 = 65. Cada combinación: 12 semillas × 1 corrida física
(REAL; NULL se deriva del mismo estado sin correr física aparte) = **780 corridas de
evolución**, cada una auditada en 7 checkpoints → 5460 puntos de verificación de suma.

ε=0 NO se incluye (Var(φ₀)=0 hace E_total=0 y las tres fracciones quedan indefinidas 0/0;
se declara como caso degenerado fuera del barrido, no un fallo).

## 6. PASS / criterios de lectura (congelados antes de correr)

- **T6 — suma:** |X + Ligada + Degradada − E_total| / E_total < 1e-9 en TODOS los
  checkpoints de TODAS las corridas (no sólo el final). Si falla en cualquier fila, se
  reporta la fila exacta — no se oculta ni se promedia.
- **Curvas enteras, no un punto (T5):** las tres fracciones X/E_total, Ligada/E_total,
  Degradada/E_total deben variar de forma medible con r (rango del barrido en cada
  fracción claramente mayor que la dispersión entre semillas) para declarar PASS en la
  parte "varían con r". Lectura esperada (no forzada): r≪1 → difusión domina antes de que
  la expansión aísle nada → Ligada≈0, Degradada alta, X bajo; r≫1 → aislamiento rápido
  → Ligada alta, Degradada baja (poco tiempo de mezcla antes de aislar), X intermedio o
  bajo según cuánto arco sobrevive completo. Si el patrón real no es este, se reporta tal
  cual — no se reinterpreta la lectura esperada como la única aceptable.
- **NULL (T4, debe morder):** Ligada_null se compara contra Ligada_real. Si el NULL
  reproduce la misma partición X/Ligada (porque la varianza between/within de una
  partición en arcos de tamaños fijos ya captura algo de estructura incluso barajada),
  se reporta honestamente — la interpretación de "Ligada = estructura espacial real"
  quedaría debilitada y se dice así, no se disimula.
- Si CUALQUIERA de estos falla, se reporta como tal — no se reinterpreta ni se ajusta el
  motor después de ver los datos (T3, regla de ejecución #1).

## 7. Verificación cruzada (regla de ejecución #4)

1. NULL propio (barajado sobre el mismo estado final), por celda.
2. Segundo método para Degradada: residuo algebraico vs suma telescópica paso a paso
   (T2 — ambos derivan de la misma identidad pero por caminos de código distintos, sirven
   de auditoría de bugs/precisión numérica, se declara honestamente que NO son física
   independiente).
3. Observables auxiliares de la base para contexto (no como juez): `persistencia()` y
   `std_ratio` de `cs074_rcruz.py` calculados sobre el mismo φ_final, reportados junto a
   X/E_total para ver si covarían con la definición propia de esta pieza.
4. Auditoría en disco: JSON crudo con TODAS las semillas individuales (no sólo medias),
   para que otro agente pueda re-verificar sin re-correr.

## 8. Salidas

- `E5_2_4_motor.py` — motor (escrito DESPUÉS de este pre-registro, importa `cs074_rcruz.py`
  sin editarlo).
- `E5_2_4_resultado_crudo.json` — filas completas del barrido: r, eps, D, H, pasos, y por
  semilla: X_real, Ligada_real, Degradada_real (ambos métodos), X_null, Ligada_null,
  Degradada_null, checkpoints intermedios, persistencia/std_ratio auxiliares, deriva de
  conservación máxima observada.
- Reporte final a CS (no archivo separado): protocolo verbatim + timestamps, curvas
  {X,Ligada,Degradada}(r) completas, verificación de suma, resultado NULL, dispersión
  entre semillas, veredicto sin suavizar.

## 9. Trampas explícitamente evitadas

- T0/T1: N, pasos y checkpoints vienen del modelo base y de calibración medida; ningún
  coeficiente puesto a mano para acercar el resultado a nada esperado.
- T2: Degradada tiene juez independiente de su definición (dos caminos de cómputo).
- T3: este archivo se congela ANTES de escribir el motor.
- T4: el NULL se diseña para poder morder (puede reproducir la partición si la métrica es
  débil — eso se reportaría como debilidad, no se escondería).
- T5: se reportan las tres curvas completas vs r, no un umbral binario.
- T6: conservación auditada en 7 checkpoints por corrida, 780 corridas → 5460
  verificaciones, no una sola al final.
- T7: el barrido cruza 6 décadas de r y 5 valores de ε; la aleatoriedad de la expansión
  (Bernoulli por arista, no determinista) actúa como perturbación dinámica en cada paso,
  no sólo semilla inicial.

No se corre nada del motor hasta que este archivo esté guardado en disco.
