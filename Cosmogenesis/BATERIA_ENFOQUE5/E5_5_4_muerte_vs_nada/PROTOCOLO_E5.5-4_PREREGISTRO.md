# PROTOCOLO E5.5-4 — Muerte térmica vs Nada operativa: caracterización de los dos estados

**Congelado (pre-registro):** 2026-07-24 16:41 (America/Santiago, UTC-4)
**Ejecutor:** CC (agente E5.5-4, batería Enfoque 5, corrida en paralelo con 29 agentes más)
**Base de código leída (NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Documento madre:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, sección "E5.5-4"

Este documento se escribe y congela ANTES de tocar el motor. Cualquier desviación
respecto de lo aquí escrito se reporta como desviación explícita, no se edita
retroactivamente (T3).

**Auditoría de reutilización (regla del director, cumplida ANTES de definir nada):**
Se revisó el disco en `BATERIA_ENFOQUE5/` buscando protocolos ya congelados de
E5.2-1 y E5.5-1 para reutilizar sus definiciones de E/X/S_ent:
- `E5_2_1_balance_deriva/` — carpeta existe, **vacía** (el agente de E5.2-1 aún no
  escribió su preregistro/motor en el momento de este congelamiento).
- `E5_5_1_*` — la carpeta **ni siquiera existe** todavía en disco.
- Único sibling con contenido real al momento de congelar: `E5_1_1_supervivencia_exergia/
  PROTOCOLO_E5.1-1_PREREGISTRO.md`. Se reutiliza de ahí la fórmula de **X** (exergía),
  que es la definición de `persistencia()` heredada literalmente de la base de código
  (no inventada por E5.1-1 tampoco). Para **E** y **S_ent**, que E5.1-1 no necesitó
  definir con ese nombre, este documento los define aquí de forma explícita y
  consistente con los axiomas E1/E2 del documento madre, para que cualquier agente
  posterior (E5.2-1, E5.5-1, E5.5-2/3/5) pueda a su vez reutilizarlos de vuelta.
- Si en el momento de ejecutar el motor (después de este congelamiento) aparecieran
  archivos nuevos de E5.2-1/E5.5-1 en disco, **no se reabre este documento** (T3): se
  anota la comparación a posteriori en el reporte final, no se reescribe el protocolo.

---

## 1. Pregunta

¿El equilibrio de muerte térmica (difusión total, sin expansión que aísle nada) es un
estado que **tiene energía (E>0) pero no puede hacer nada (X=0, S_ent=máx)** — y es esto
**medible y distinguible** de la Nada operativa (∅, E=0, el campo no existe)? La consigna
explícita del director es: **confirmar empíricamente, no dar por sentado.**

## 2. Modelo (heredado de cs074_rcruz.py, motor propio bajo mi prefijo)

Campo escalar φ en un anillo de N=200 sitios, física idéntica a la base:
- Fondo φ=1 + perturbación ε·(suma de 5 armónicos con fase aleatoria, normalizada a
  desviación estándar 1) — `campo_inicial()`.
- **Difusión:** relajación local hacia el promedio de vecinos, solo por aristas vivas
  (`paso_difusion()`, sin modificar).
- **Expansión:** corte de aristas vivas por Bernoulli con probabilidad H por paso
  (`paso_expansion()`, sin modificar). H = min(r·D, 1.0); D medido del propio campo
  (`medir_D()`).
- **Rama primaria (muerte térmica):** r=0 → H=0 → sin expansión, difusión pura corriendo
  hasta equilibrio. Este es el escenario físico correcto de "muerte térmica": la
  expansión (r≫1) AÍSLA y CONGELA estructura — es lo opuesto del equilibrio térmico. El
  equilibrio de máxima entropía requiere específicamente que NADA lo aísle.
- **Rama de control/discriminación (no es la pregunta central, pero valida que el
  observable no sea trivial):** r=1000 (aislamiento extremo, sobredimensionado sobre el
  r≈1 esperado) — mismo número de pasos que la rama r=0, para confirmar que el vector
  (E,X,S_ent) del equilibrio térmico NO aparece automáticamente para cualquier r; debe
  requerir específicamente difusión sin aislamiento.
- **Comparador "Nada" (∅):** NO es una corrida del motor — es un estado de referencia
  construido explícitamente, φ≡0 en los N sitios (ausencia total de campo, no "campo en
  su valor mínimo"). Se mide el mismo vector (E,X,S_ent) sobre este estado por las MISMAS
  fórmulas, para contrastar numéricamente contra el equilibrio térmico. Es la
  operacionalización literal de "Nada": no hay presupuesto que repartir.

## 3. Axiomas declarados (E1/E2, NO física real)

- **E1 (conservación declarada):** E_decl = mean(φ) se declara conservado por la difusión
  (promedio local lineal en grafo regular). Se AUDITA cada corrida (inicio vs fin), nunca
  se renormaliza el campo. Deriva relativa reportada; umbral de PASS de conservación
  heredado de la convención de E5.2-1 en el documento madre (<1e-6) aunque ese protocolo
  aún no estaba en disco al congelar este documento — es el umbral que el propio
  documento madre fija en su sección E5.2-1, así que se adopta por coherencia con el
  texto madre, no por lectura de un archivo ajeno.
- **E2 (redistribución por expansión):** la expansión no crea energía, solo aísla
  regiones y con ello impide que la difusión termine de lavar sus gradientes. Es el
  argumento de por qué la rama r=1000 debe diferir de la rama r=0 al mismo número de
  pasos.

## 4. Definiciones operacionales del vector (E, X, S_ent) — CONGELADAS

### E — energía total declarada
    E = mean(φ)          (intensiva; E_total_extensiva = Σφ = N·E si se prefiere extensiva)
Justificación: bajo E1, la difusión (promedio local lineal) preserva Σφ exactamente en
un grafo regular; la expansión (cortar aristas) no mueve masa del campo, solo aristas.
Por construcción fondo=1 y la perturbación ε·pert tiene media 0 (normalizada en
`campo_inicial`), así que E≈1 en el estado inicial para CUALQUIER ε, incluido ε=0. Esto
es intencional: permite aislar el efecto de ε sobre X y S_ent sin confundirlo con un
cambio trivial en el presupuesto de energía. Se AUDITA la deriva real (no se asume).

### X — exergía (capacidad de hacer trabajo), reutilizada de E5.1-1/base
    c = corr(φ_final, roll(φ_final,1))   (autocorrelación a un paso; clip a ≥0)
    v = Var(φ_final) / Var(φ_inicial)
    X = c · v
Fórmula IDÉNTICA a `persistencia()` de la base y a la definición congelada por E5.1-1
(única sibling con contenido en disco al momento de congelar). X=0 si Var(φ_inicial)=0
(ε=0 exacto, sin diferencia que evolucionar) — caso manejado explícitamente sin división
por cero.

### S_ent — entropía (segundo observable, método INDEPENDIENTE de X — T2)
Se define sobre cómo el presupuesto de energía se **distribuye** entre los N sitios, no
sobre el histograma de valores de φ (que daría la lectura opuesta: un φ perfectamente
uniforme tiene un solo valor repetido, que sería entropía de valores ≈0, la lectura
termodinámicamente incorrecta). La entropía correcta de máxima-al-equilibrar es la de
Gibbs/Shannon sobre la distribución normalizada de energía por sitio:
    p_i = φ_i / Σφ_j     (requiere φ_i ≥ 0 en todo i — verificado, ver §8)
    S_ent = −Σ p_i·log(p_i) / log(N)     ∈ [0,1]
S_ent=1 (máximo) exactamente cuando p_i=1/N para todo i, es decir φ uniforme — el
equilibrio de muerte térmica. S_ent=0 (mínimo) cuando toda la energía está concentrada en
un solo sitio. Esta fórmula es funcionalmente DISTINTA de X (Shannon de la distribución
normalizada vs. autocorrelación×varianza-retenida): no son la misma cantidad disfrazada
(T2 — el observable no es su propio juez), aunque ambas midan homogeneización, son
métodos independientes y se reportan por separado, incluyendo su correlación cruzada
como diagnóstico (si correlacionan >0.99 en todo el barrido, se reporta como hallazgo:
ambas fórmulas miden lo mismo en este modelo — no se oculta si eso ocurre).

### W_res — capacidad de trabajo residual (tercer observable, complementa X)
    W_res = mean(|φ_i − mean(φ)|)      (desviación media absoluta, unidades de φ)
Proxy directo de "cuánta energía es extraíble contra el estado muerto" (análogo a la
definición clásica de exergía = desviación del estado-muerto ambiental). W_res=0 solo si
φ es exactamente uniforme. Reportado en unidades absolutas (no fracción) para responder
literalmente "capacidad de trabajo residual" que pide el enunciado de E5.5-4.

## 5. Barrido (sobredimensionado, regla del director)

| Eje | Rango | Puntos |
|---|---|---|
| ε (eje primario, límite ε→0) | {0, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 1.0} | 9 (0 exacto + 15 décadas de acercamiento a 0, sobredimensionado sobre el rango [0…1e-2] que pide la sección E5.5-1 del doc madre, más 2 anclas de contraste 0.1/1.0 donde SÍ se espera estructura) |
| r (rama muerte térmica vs control aislado) | {0, 1000} | 2 (0 = sin expansión/equilibrio térmico; 1000 = aislamiento extremo, control de discriminación, sobredimensionado sobre r≈1) |
| semillas | 0..19 | 20 (≥16 pedido) |
| pasos | calibrado (lavado a P<0.05, N=200, eps=1e-3, H=0) × margen 1.15, y ADEMÁS ×4 extra como corrida "sobredimensionada" de verificación de convergencia (dos duraciones, se reporta si el vector ya no cambia entre ambas — confirma equilibrio genuino, no corte a medio camino) | 2 duraciones |
| N | 200 (igual que motor "produccion" de la base) | fijo |
| comparador Nada (∅) | φ≡0, sin evolución (estado de referencia, no depende de ε/r/semilla/pasos) | 1 fila de referencia |

Total corridas del motor: 9 ε × 2 r × 20 semillas × 2 duraciones = **720 corridas** de
evolución (más 1 fila de referencia ∅, calculada, no simulada, y la calibración de
lavado).

## 6. NULL

El documento madre marca explícitamente **NULL: —** para E5.5-4 (es caracterización, no
detección contra barajado). Se respeta esa indicación. El comparador operativo de esta pieza NO
es un NULL estadístico sino el estado ∅ literal descrito en §2 y §5 — es la pregunta
misma del experimento, no un control de significancia.

## 7. PASS / criterios de lectura (congelados antes de correr)

Para la rama muerte térmica (r=0, todas las ε, ambas duraciones de pasos):
- **PASS_E:** |E_final − E_inicial| / E_inicial < 1e-6 (conservación, umbral de E5.2-1
  adoptado por coherencia con el documento madre) — **Y** E_final > 0 (trivialmente
  cierto si la conservación se cumple, pero se mide, no se asume).
- **PASS_X:** X_final < 0.05 (umbral P_LAVADO de la base, mismo criterio que
  `control_r0_ok`).
- **PASS_Sent:** S_ent_final > 0.99 (cerca del máximo normalizado =1).
- **PASS_vector (el veredicto central del experimento):** PASS_E ∧ PASS_X ∧ PASS_Sent
  simultáneamente, en la MISMA corrida, reportado por semilla y agregado — no basta que
  cada condición se cumpla en promedio; se cuenta cuántas semillas cumplen las tres a la
  vez.
- **Confirmación de "no es automático":** en la rama r=1000 (mismo pasos, mismas ε>0), se
  espera PASS_X FALSE (X_final alto, estructura congelada) — si en cambio r=1000 también
  da X≈0, el observable no discrimina y se reporta como tal (no se reinterpreta).
- **Distinción vs Nada (∅):** se reporta la comparación numérica directa E_muerte vs
  E_nada(=0 por construcción) — la lectura pre-registrada es que son categóricamente
  distintos (E_muerte>0 medido, E_nada=0 por definición del estado ∅, no por evolución).
  Esto es en parte definicional (∅ no tiene campo), pero el punto empírico real es que
  **el estado de muerte térmica, alcanzado por evolución real del campo, retiene E
  medible pese a X≈0 y S_ent≈máx** — eso es lo que se confirma o no con datos, no la
  comparación trivial contra ∅.
- **ε=0 exacto:** caso trivial de control interno — Var(φ_inicial)=0, así que X=0 por
  definición matemática (no por dinámica) y S_ent=1 exacto (φ uniforme desde el inicio,
  sin necesidad de evolución). Se reporta por separado, no se mezcla con las filas donde
  X→0 POR EVOLUCIÓN (que es el resultado sustantivo).
- Si cualquiera de estos falla, se reporta como tal — no se reinterpreta ni se ajusta el
  motor después de ver los datos (T3, regla de ejecución #1).

## 8. Verificación cruzada (regla de ejecución #4)

1. Segundo observable/método: S_ent (Shannon de energía-por-sitio) es independiente de X
   (autocorrelación×varianza) — correlación cruzada reportada.
2. Tercer observable: W_res (desviación media absoluta), en unidades absolutas.
3. Auditoría de conservación E1 (deriva de mean(φ) inicio→fin) reportada en cada fila.
4. **Validez de S_ent (positividad de φ):** se verifica en cada corrida que min(φ) ≥ 0
   antes de calcular p_i; con el rango de ε de este barrido (≤1.0) y fondo=1, se espera
   que φ no cruce 0 salvo quizás en ε=1.0 con la perturbación en su pico — si ocurre, se
   clip a 0 y se reporta CUÁNTAS veces y en qué filas ocurrió el clip (no se oculta).
5. Confirmación de equilibrio genuino: comparación entre las dos duraciones de pasos
   (calibrada×1.15 vs ×4 extra) — si el vector (E,X,S_ent) no cambia entre ambas, el
   equilibrio es real, no un corte a medio camino.

## 9. Salidas

- `E5_5_4_engine.py` — motor (escrito DESPUÉS de este pre-registro).
- `E5_5_4_resultado_crudo.json` — filas completas del barrido (ε, r, semilla, duración,
  E, X, S_ent, W_res, deriva_E, min_phi, clip_aplicado) + fila de referencia ∅.
- Reporte final verbatim a CS con protocolo, timestamps, vector medido con
  incertidumbre, dispersión entre semillas, veredicto sin suavizar.

## 10. Trampas explícitamente evitadas

- T0: N=200 y fórmulas heredadas de la base/E5.1-1, no puestas a mano para este experimento.
- T1: umbrales (0.05 para X, 0.99 para S_ent, 1e-6 para deriva) declarados AQUÍ, antes de
  correr, tomados de convenciones ya presentes en la base/documento madre — no ajustados
  a posteriori.
- T2: S_ent y W_res son observables calculados por fórmula fija, independientes de X;
  el veredicto lo da la conjunción de las tres condiciones medidas, no una sola métrica
  que se autojustifique.
- T3: este documento se congela ANTES de escribir el motor; no se reabre tras ver datos.
- T5: se reporta la curva completa (por ε, r, duración), no un gate binario único.
- T6: se audita conservación de E cada corrida.
- T7: ≥16 semillas (uso 20) + dos duraciones de evolución como perturbación estructural
  del barrido (no es ruido dinámico per sé como en E5.1-1/E5.1-4, pero cumple el espíritu
  de T7 para esta pieza específica: no un solo punto ni solo semilla).

No se corre nada del motor hasta que este archivo esté guardado en disco.
