# INSTRUCCIÓN PARA CC — Barrido de la asimetría fundacional ε (re-fundación del motor)
**De:** CS · **Fecha:** 20-jul-2026 · **Regla:** corres y mides, NO ajustas para que salga algo. Un resultado nulo es un HALLAZGO, no un error a corregir.

## Por qué existe este experimento
La auditoría (verificada contra el motor, por ti y por CS) mostró dos cosas que hay que reparar juntas, porque una depende de la otra:
1. **El confinamiento no confina:** `_detecta_trios` (nucleo.py:28-58) calcula `ligado = Bq>1.5*b0` y nunca lo consulta. Los bariones salen por estequiometría de población (nq/3), no por ligadura real. Apagar la fuerza fuerte deja bariones=100 (falso).
2. **La densidad #23 es un alias del orden de construcción:** el campo de `catalogo()` correlaciona con el índice (quarks primero → cuartil bajo). Es Shannon en el cimiento.

La decisión del director (Alexis): **la única diferencia legítima en el estado pre-físico es UN escalar — la asimetría fundacional ε — que NO se elige, se BARRE de 0 a 1 reportando la curva entera.** No hay campo de densidad asignado en t=0; todo contraste, si existe, debe PRODUCIRLO la dinámica (fuerzas + expansión + enfriamiento). Si no emerge a ningún ε, ese es el resultado: Mundo B se extiende.

## Qué construir (módulos aislados, no toques los viejos in situ)
### 1. ε como escalar de población — SIN campo de densidad inicial
- Parametriza la asimetría como un conteo: para un pool total T = N_q + N_q̄ y un ε dado,
  `N_q = round(T*(1+ε)/2)`, `N_q̄ = round(T*(1-ε)/2)`. Así `ε = (N_q − N_q̄)/(N_q + N_q̄)` EXACTO. ε es UN número, no un vector, no una distribución.
- El catálogo entra **sin `densidad` #23** (o con densidad constante = 1.0 para todas las partículas). Prohibido asignar un campo de contraste por partícula en t=0. Ésta es la reparación de raíz del alias de índice: si no hay campo, no hay alias posible.
- Guardián **G-EPSILON-UNICO-ESCALAR** y **G-SIN-CAMPO-DENSIDAD-INICIAL**.

### 2. Confinamiento real (Opción B) — el conteo debe exigir ligadura
- Reescribe `_detecta_trios` para que forme un trío SOLO si los tres quarks forman un **triángulo cerrado** en el grafo de ligadura (`ligado[i,j] AND ligado[j,k] AND ligado[i,k]`), colores distintos, no-anti, vivos.
- **Desempate cuando hay varios triángulos candidatos: por una MAGNITUD FÍSICA EMERGENTE, nunca por índice.** Como ya NO hay campo de densidad inicial, el desempate debe usar algo que la física produjo — p.ej. el **peso de ligadura acumulado** `Bq` (la suma de los tres enlaces del triángulo, o su enlace más débil `min`). El más ligado se forma primero. Guardián **G-DESEMPATE-FISICO-NO-INDICE**.
- Verifica admisibilidad: con la fuerza fuerte apagada, este núcleo DEBE dar 0 bariones (a diferencia del viejo, que da 100). Reporta ambos lado a lado como control de que la reparación funciona.

### 3. El barrido (el experimento central)
- ε en escala logarítmica, fino cerca de cero: **`0, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0`**.
- Para cada ε, corre el motor COMPLETO (los 23, expansión y enfriamiento incluidos, pasos equilibrados) y registra: `n_trios` (Opción B), `bariones`, `p/n`, `H`, `He`, `diametro_red`, y si emerge algún `contraste_de_densidad` (varianza de la densidad emergente vs 0).
- **Control NULL por cada ε:** compara el observable REAL contra su barajado (misma cantidad de aristas/partículas). Reporta z = (REAL−NULL)/sd_NULL. Guardián **G-NULL-MISMA-MAGNITUD**.

## GUARDIANES (si uno se viola, PARA y reporta — no sigas, no ajustes)
- **G-EPSILON-UNICO-ESCALAR**: ε es un solo número por corrida. Prohibido un campo/vector/distribución de ε.
- **G-SIN-CAMPO-DENSIDAD-INICIAL**: cero densidad #23 asignada en t=0. Todo contraste debe emerger de la dinámica.
- **G-CONFINAMIENTO-REAL**: bariones = triángulos cerrados en el grafo de ligadura, no estequiometría.
- **G-DESEMPATE-FISICO-NO-INDICE**: desempates por magnitud física emergente (peso Bq), jamás por posición en el array.
- **G-BARRIDO-COMPLETO**: se reporta la CURVA entera (todos los ε). Prohibido elegir un ε "porque da bonito".
- **G-NULO-ES-HALLAZGO**: si a ningún ε emerge contraste/estructura por encima de su NULL, ESE es el resultado. No se toca nada para recuperarlo.
- **G-NO-TOCAR-PROHIBIDAS**: no tocar I_WILL_NOT_PUBLISH_CRAP, tolv, dt, criterio de conservación. No re-cablear el 20.0/freeze-out (muerto, irrelevante).

## Lectura pre-inscrita (antes de correr — para no acomodarse al resultado)
- **ε=0** → sin diferencia, debe dar **0 bariones / 0 estructura**. Si el núcleo NUEVO da >0 en ε=0, hay un flanco que revisar. Si el núcleo VIEJO da 100 en ε=0 (ya verificado), eso confirma que el Shannon era total.
- **ε ∈ [1e-9, 1e-6]** → si emergen los primeros triángulos cerrados aquí, hay umbral de emergencia genuino.
- **ε ∈ [1e-3, 0.1]** → régimen que usábamos; producción (si la hay) debería ser estable.
- **ε → 1** → saturación / sólo materia.
- **Si la curva es plana en 0 a todo ε** → no emerge estructura de la sola asimetría: Mundo B se extiende, la contingencia se confirma. Resultado legítimo, se reporta tal cual.

## Entregable
Una tabla `ε | n_trios | bariones | p/n | H | He | diámetro | contraste | z_vs_NULL` para los 9 valores, más una figura `bariones-vs-ε` (log x) y `contraste-vs-ε`. Y una línea de veredicto CRUDO, sin interpretación cosmológica. El motor contesta; CS adjudica después.

## Nota de nomenclatura (para el paper, cuando llegue)
Esto NO se llama "densidad del quark" (error de categoría; un quark puntual no porta densidad). ε es la **asimetría bariónica fundacional** (exceso de materia sobre antimateria, un escalar). Cualquier contraste que emerja se llama **densidad de energía local / contraste emergente**, medido a posteriori, nunca asignado.
