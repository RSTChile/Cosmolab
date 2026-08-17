# PROTOCOLO F3-4 — PRE-REGISTRO
## "Reversibilidad térmica: ¿se re-homogeneiza si se detiene la expansión?"

**Batería:** BATERÍA DE FUNDAMENTOS (Enfoque 3 — enfriamiento adiabático).
**Experimento:** F3-4 (`BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección Enfoque 3).
**Ejecuta:** CC, prefijo `F3_4_`, carpeta
`codigo/BATERIA_FUNDAMENTOS/F3_4_reversibilidad_termica/`. No toca `CF2_estiramiento_motor.py`
ni ningún otro archivo existente (solo lectura).

**Este documento se escribe y congela ANTES de correr el motor de producción.** El motor
(`F3_4_reversibilidad_termica_motor.py`) y los resultados (`results/F3_4_reversibilidad_termica/`)
se generan DESPUÉS — verificar mtime.

---

## 1. Pregunta

Si se detiene la expansión (`a` queda fijo desde ese instante en adelante) y solo se deja correr
difusión después, ¿el gradiente térmico se re-aplana (la difusión alcanza a re-homogeneizar) o
queda congelado? Concretamente: ¿existe un **tiempo de no-retorno** — un `t_g` de parada más allá
del cual, incluso dándole a la difusión tanto tiempo como tuvo la expansión completa, el gradiente
ya no puede re-aplanarse de forma apreciable?

**Predicción pre-registrada (antes de ver datos):** parar temprano (poca expansión acumulada, `D`
todavía grande) permite re-homogeneizar; parar tarde (mucha expansión acumulada, `D` ya casi nulo
porque `D=D0/a³`) no permite re-homogeneizar — el gradiente queda congelado. Debe existir una
transición entre ambos regímenes en algún punto intermedio del barrido de `t_g` de parada.

## 2. Sustrato (heredado de CF2/F3-1, no se retoca — T1)

Mismo campo continuo `T(x,y)` en grilla `L×L` (`L=64`), mismo perfil inicial (salto tanh de ancho
comóvil `W0=1.2`), misma difusión de 4 vecinos, mismo reloj de expansión
`a(t_g) = exp(H_EXP·t_g)` con `H_EXP=6.0`, mismo `D0=0.12`, `DT=0.25`, `N_SUB=2`,
`ORIGINAL_STEPS_PER_TG=399` (⇒ `dtg = 1/399`). Ley de dilución REAL (idéntica a CF2, no
re-derivada aquí): `ρ=ρ0/a³`, `D=D0·(ρ/ρ0)=D0/a³` — el resultado de CF2/F3-1 (el transporte se
apaga al expandirse) es el PRESUPUESTO de este experimento, no lo que se re-litiga.

**Extensión pre-registrada de F3-4 (perturbación dinámica, regla general T7 — sección 2.2 del
documento autoritativo, "barridos extensos + perturbación dinámica"):** en cada sub-paso de
difusión se añade opcionalmente ruido gaussiano aditivo de amplitud `σ` (escalado por `√(dt/n_sub)`,
discretización Euler-Maruyama estándar), barrida en `RUIDO_DINAMICO_GRID = {0.0, 1e-3, 5e-3}`
(3 puntos; `0.0` reproduce exactamente la física de CF2 sin modificarla, los otros dos son
perturbaciones modestas de la dinámica — no solo de la semilla). Después de cada paso completo el
campo se recorta a `[0,1]`, igual que CF2.

## 3. Diseño experimental (la variable nueva de F3-4)

Para cada `(t_g de parada, semilla, σ_ruido)`:

1. **Fase común de expansión** (idéntica a CF2 REAL): se integra desde `t_g=0` con
   `a(t_g)=exp(H_EXP·t_g)`, `D=D0/a³`, muestreando (checkpointing markoviano, igual truco que
   CF2) el estado del campo exactamente en cada `t_g` de parada del barrido. Esto garantiza que
   las ramas STOP y NULL de abajo parten del **mismo campo exacto** en el instante de la
   bifurcación — la única diferencia entre ramas es lo que pasa DESPUÉS de parar.
2. **Rama STOP (parar la expansión):** desde el checkpoint, `a` queda fijo en `a_parada`
   (⇒ `D` queda fijo en `D_parada = D0/a_parada³`) durante una ventana de reloj genético
   `POST_STOP_TG` (idéntica para TODOS los puntos del barrido — ver §4), y solo corre difusión.
3. **Rama NULL — "nunca parar" (control pre-registrado por el documento autoritativo):** desde el
   MISMO checkpoint, la expansión CONTINÚA sin frenar (`a(t_g)=exp(H_EXP·t_g)` sigue creciendo,
   `D=D0/a³` sigue cayendo) durante la MISMA ventana `POST_STOP_TG`. Es la pregunta contrafactual
   exacta: "¿qué habría pasado si no hubiéramos parado?" con todo lo demás idéntico (semilla,
   ruido, campo de partida).

Ambas ramas usan generadores de ruido independientes y deterministas,
`np.random.default_rng([seed, idx_checkpoint, codigo_rama])`, para que la bifurcación sea
reproducible sin reusar el mismo stream de azar en las dos ramas.

## 4. Barrido (T7 — ≥8 puntos de parada × ≥12 semillas, exigido por el documento autoritativo)

- **`a` de parada:** `np.geomspace(1.0, 1000.0, 9)` — 9 puntos, 3 décadas (mismo rango que el
  barrido de CF2/F3-1; `a=1` es el caso límite "parar antes de expandir nada"). `t_g de parada
  = ln(a_parada)/H_EXP`.
- **Semillas:** las 10 semillas estándar del proyecto (`CF2_estiramiento_motor.py::SEEDS_STANDARD`)
  más las 2 semillas de extensión ya usadas por F3-3 (`271828` dígitos de e, `161803` dígitos de
  φ) — total 12, mismo criterio que el resto de la batería para poder comparar entre experimentos.
- **Ruido dinámico:** `RUIDO_DINAMICO_GRID = {0.0, 1e-3, 5e-3}` (§2).
- **Ventana post-parada `POST_STOP_TG`:** `ln(1000)/H_EXP` (el mismo `t_g` que tardó TODA la fase
  de expansión de CF2 en ir de `a=1` a `a=1000`) — se le da a la difusión, en cada punto del
  barrido, tanto tiempo como duró la expansión completa. Es una constante única, la MISMA para
  los 9 puntos de parada (necesario para que la comparación entre puntos sea justa — no se le da
  más tiempo a los puntos tardíos).
- Total de evaluaciones: 9 puntos de parada × 12 semillas × 3 ruidos × 2 ramas = 648 integraciones
  (más las 36 fases comunes de expansión, una por semilla×ruido).

## 5. Observables (T2 — no comparten variable con el juez de otro experimento)

Dos métodos independientes, medidos en el checkpoint de parada y al final de la ventana
`POST_STOP_TG` de cada rama:

**Método 1 — gradiente comóvil máximo (banda central, evita wrap-around), igual definición que
CF2:**
```
∇_comov = max |∂T/∂x|   (banda central)
reaplanamiento_∇(t_g_parada, rama) = (∇_comov_parada − ∇_comov_final) / ∇_comov_parada
```
`reaplanamiento_∇ ≈ 1` ⇒ el gradiente se borró casi del todo; `≈ 0` ⇒ quedó congelado.

**Método 2 — varianza espacial global del campo (medidor completamente distinto — no usa
derivadas ni banda central):**
```
Var(T) = varianza de T sobre toda la grilla
reaplanamiento_Var(t_g_parada, rama) = (Var_parada − Var_final) / Var_parada
```
Si un campo se homogeneiza, su varianza espacial tiende a 0 independientemente de si el
gradiente máximo local ya cayó. Verificación cruzada: el mapa de `reaplanamiento_Var` debe
coincidir cualitativamente con el de `reaplanamiento_∇` (ambos ≈1 o ambos ≈0 en el mismo régimen).
Si divergen, se reporta como hallazgo, no se descarta uno de los dos observables (T3).

## 6. Criterio de PASS (congelado, T3 — no se toca si falla)

Por `(semilla, σ_ruido)`, sobre el observable primario `reaplanamiento_∇` de la rama STOP, evaluado
en los 9 puntos del barrido de `a` de parada (ordenados de menor a mayor):

1. **`cond_a` — monotonicidad no-creciente** (tolerancia `MONO_TOL=0.05`, más laxa que la de CF2
   porque aquí hay ruido dinámico estocástico posible): `reaplanamiento_∇[i+1] ≤
   reaplanamiento_∇[i] + MONO_TOL` para todos los pares consecutivos. Predicción: parar más tarde
   nunca re-homogeneiza MÁS que parar más temprano.
2. **`cond_b` — el punto más temprano SÍ re-homogeneiza** (`REHOMOG_EARLY_MIN = 0.5`):
   `reaplanamiento_∇[0] ≥ 0.5` en la rama STOP. Si esto falla, ni siquiera parar casi al inicio
   permite re-aplanar en la ventana dada — se reporta tal cual (falsable, no se ajusta el umbral).
3. **`cond_c` — el punto más tardío queda congelado** (`REHOMOG_LATE_MAX = 0.1`):
   `reaplanamiento_∇[8] ≤ 0.1` en la rama STOP.
4. **`cond_d` — el NULL muerde** (`DIFF_MIN = 0.1`): en el punto más temprano,
   `reaplanamiento_∇_STOP[0] − reaplanamiento_∇_NULL[0] ≥ 0.1` — parar debe re-homogeneizar
   claramente MÁS que seguir expandiendo (la expansión continua sigue apagando `D`); si STOP y
   NULL no difieren, el experimento no discrimina y se reporta como hallazgo T4.

**`seed_pass = cond_a AND cond_b AND cond_c AND cond_d`.**

**Verdict del experimento:** `rate = (#combos (semilla,ruido) con seed_pass) / 36`,
`PASS_RATE_MIN = 0.55` (mismo umbral que el resto de la batería CF/F3, no ajustado aquí).

**Punto de no-retorno (descriptivo, T5 — no es un gate binario adicional, se reporta siempre la
curva completa):** por combo, el primer `a_parada` de la grilla (ascendente) donde
`reaplanamiento_∇_STOP ≤ REHOMOG_LATE_MAX` de forma sostenida (no vuelve a subir por encima del
umbral en ningún punto posterior del barrido). Si nunca se cumple, se reporta `no_retorno = None`
("todo el rango probado re-homogeneiza"). Si se cumple desde el primer punto, se reporta
`no_retorno = a_parada[0]` ("ya está congelado desde el arranque del barrido").

Si `rate < 0.55`, o si `cond_d` falla sistemáticamente (NULL no muerde), o si el punto de
no-retorno no existe o existe desde el inicio: se reporta el FAIL/hallazgo con los números crudos.
No se cambia el juez, no se sustituyen los observables, no se ajustan los umbrales después de ver
los datos (T3).

## 7. Qué NO es este experimento

- No mide masa, Higgs, ni linaje. Solo la reversibilidad del enfriamiento adiabático estudiado en
  F3-1/F3-2 (el "¿enfriar es expandir?" de CF2), específicamente su comportamiento al DETENER la
  expansión.
- No re-litiga si `D=D0/a³` es correcto — ese es el resultado heredado que este experimento asume
  como sustrato para preguntar sobre reversibilidad, no sobre la ley de dilución en sí (eso es
  F3-3/F4-6).
- No se auto-adjudica el veredicto de la hipótesis más amplia de la batería — eso lo hace CS con
  los números crudos.
- No toca `CF2_estiramiento_motor.py`, ni ningún archivo de `F3_1_estiramiento_ruido/` o
  `F3_3_exponente_dilucion/` (carpetas de otros agentes en paralelo, solo lectura si se consultan).

---

**Fecha/hora de este pre-registro:** ver mtime del archivo (se congela antes de generar
`F3_4_reversibilidad_termica_motor.py` y cualquier resultado en
`results/F3_4_reversibilidad_termica/`).
