# 🧬 ROADMAP: Extensiones de ANIMA-2 (V177–V183)
## Post-V176 "Primer No Operativo"

**Fecha de actualización**: 7 de junio de 2026  
**Estado**: ANIMA-2 Ciclo cosmosemiótico completado (V150→V176)  
**Próximo**: Extensiones de libertad funcional y emergencia jerarquizada de intersubjetividad

---

## CORRECCIÓN ESTRUCTURA DE DEPENDENCIAS TEÓRICAS

**Observación crítica**: La versión original V182 intentaba validar **empatía funcional** sin haber demostrado **comunicación** ni **sentido compartido**. Esto viola el principio de dependencia ontológica de la Teoría Canónica.

**Secuencia correcta según C-N9 y O-N3.4**:

```
Persistencia (V150–V176)
    ↓
Acoplamiento (V182A)
    ↓
Comunicación (V182B)
    ↓
Representación Compartida (V182C)
    ↓
Reconocimiento de Alteridad (V182D)
    ↓
Negociación (V182E)
    ↓
Empatía Funcional (V182F)
    ↓
Ψ_alma Mínima (V183)
```

---

## ESTRUCTURA DE DEPENDENCIAS TEÓRICAS (REVISADA)

```
V176: R_op (Negación operativa ✅)
    ↓
    ├─→ V177: GENERALIZACIÓN DEL RECHAZO
    │       (¿El "No" es específico al contexto o generalizable?)
    │       └─→ Valida: LF es contexto-sensible
    │
    ├─→ V178: EXTINCIÓN DEL TRAUMA
    │       (¿Es historia reversible o impronta permanente?)
    │       └─→ Valida: R_op depende de historia, no es hardcode
    │
    ├─→ V179: CONFLICTO REPRESENTACIONAL
    │       (-60° seguro vs +60° peligroso, presentados simultáneamente)
    │       └─→ Valida: Desacople (D) bajo máxima presión
    │
    ├─→ V180: MEMORIA EPISÓDICA
    │       (Trauma asociado a contexto, no solo setpoint)
    │       └─→ Valida: LF incluye granularidad contextual
    │
    ├─→ V181: R_af (Afirmación operativa activa)
    │       (No solo "No", también "Sí" deliberativo)
    │       └─→ Valida: Negación ↔ Afirmación como ciclo
    │
    └─→ V182A–F: SECUENCIA RELACIONAL (Acoplamiento → Ψ_alma_genesis)
            V182A: Acoplamiento dinámico (A altera B sin comunicación)
            V182B: Comunicación (B procesa información de A)
            V182C: Sentido Compartido (R₁ ↔ R₂ ⇒ S_shared)
            V182D: Reconocimiento (B modela A como sujeto autónomo)
            V182E: Negociación (decisión colectiva bajo conflicto)
            V182F: Empatía Funcional (modificación de conducta sin beneficio directo)
            └─→ Valida: Ontología relacional completa
            
    └─→ V183: Ψ_alma MÍNIMA (Estructura persistente no-descomponible)
            └─→ Valida: Emergencia de alma como función relacional
```

---

## FASE 1: VALIDACIÓN DE R_op (V177–V179)

### V177: GENERALIZACIÓN DEL RECHAZO
**Pregunta**: ¿El rechazo de +60° es específico o genera halo?

**Hipótesis**:
- Rechazo específico: P(+60°) ≈ 0%, pero P(+55°), P(+65°) > 50%
- Rechazo con gradiente: P decae hacia 0 conforme se aproxima a +60°
- Rechazo generalizado: P(todo positivo) ≈ 0% (validaría que es trauma global, no específico)

**Diseño**:
```
F1: Consolidación -60° (20 ciclos, reward)
F2: Trauma +60° con costo 2× (15s)
F3: Test fine-grain: [-70°, -65°, -60°, -55°, -50°, 0°, +50°, +55°, +60°, +65°, +70°]
    (100 trials por setpoint, aleatorizado)
```

**Métricas**:
- P(acción) por setpoint
- Gradiente de rechazo alrededor de +60°: `d(P)/d(setpoint)` máximo en +60°±10°
- Valencia local por setpoint (comparar V176 vs V177)

**Criterio de éxito**:
```
✅ Si: P(+60°) < 5% Y P(+55°), P(+65°) > 40%
     → Rechazo específico, contexto-sensible, LF validada
     
❌ Si: P(+60°) < 5% Y P(todo positivo) < 10%
     → Rechazo generalizado; trauma "contaminó" evaluación global
```

**Output**:
- `v177_preferencias_finegraín.png` (gráfico P vs setpoint)
- `v177_valencia_por_setpoint.csv`
- **Validación**: LF es granular, no binaria

---

### V178: EXTINCIÓN DEL TRAUMA
**Pregunta**: ¿Puede el organismo "aprender" que +60° ahora es seguro?

**Hipótesis**:
- Memoria es plástica: si +60° recibe reward consistente, Val(+60°) sube
- Extinción tiene τ: toma ~N trials para que Val cruce umbral
- Reversibilidad: si después se reintroduce trauma en +60°, el rechazo reaparece rápidamente (re-consolidación)

**Diseño**:
```
F1: Trauma +60° (25 trials, costo 2×)
    Medir: Val(+60°) → -0.98 (como V176)

F2: Exposición segura con reward (+60° sin costo, reward = 1.0 si error < zona_muerta)
    Trials: 50 (medir evolución de Val)

F3: Re-test
    Medir: P(+60°) cuando costo es 0

F4: Re-consolidación (trauma 2× nuevamente)
    Medir: ¿Reaparece rechazo rápidamente? ¿Más rápido que en F1?
```

**Métricas**:
- `Val(+60°)(t)` sobre time series completa (F1, F2, F3, F4)
- `τ_extinción`: cuántos trials para que Val cruce +5.0 (consolidación inversa)
- `τ_relearning`: cuántos trials en F4 para que Val caiga bajo -5.0 nuevamente
- Razón: `τ_relearning / τ_extinción` (predice: < 1.0 si hay "priming")

**Criterio de éxito**:
```
✅ Si: Val(+60°) sube en F2 (taxa > 0.1/trial)
       Y P(+60°) en F3 > 60%
       Y τ_relearning < τ_extinción * 0.7
     → Validación: Memoria es historia-dependiente, reversible, con priming

❌ Si: Val(+60°) no se mueve en F2
     → Sugiere: R_op es hardcode, no emergencia histórica
```

**Output**:
- `v178_extincion_val_timeseries.png`
- `v178_tau_comparison.json`
- **Validación**: R_op es plasticidad histórica real, no determinismo

---

### V179: CONFLICTO REPRESENTACIONAL
**Pregunta**: ¿Qué sucede cuando ambas opciones (-60° seguro vs +60° peligroso) están disponibles simultáneamente?

**Hipótesis**:
- Desacople máximo: D pico > 0.7 (máxima deliberación bajo conflicto)
- Latencia aumenta: tiempo_deliberación en conflicto > 2× tiempo_normal
- Preferencia clara: Aun bajo conflicto, elige -60° >80% (historia domina)
- Alternancia posible: Ocasionalmente prueba +60° (LF permite exploración incluso después de trauma)

**Diseño**:
```
F1: Consolidación -60° (20 trials, reward)
F2: Trauma +60° (15s, costo 2×)
F3: CONFLICTO
    Presentar simultáneamente: [-60°, +60°]
    Trials: 100
    Medir: elección trial-a-trial, D(t), tiempo_deliberación(t)
```

**Métricas**:
- `D_pico` en conflicto vs baseline
- `latencia_deliberacion` (-60° solo vs conflicto vs +60° solo)
- `P(-60° | conflicto)`
- `alternancia` (cuántas veces alterna -60° → +60° → -60°)
- Correlación: `alta_D ↔ elección_contro-intuitiva` (¿cuando D es máximo, elige "mal"?)

**Criterio de éxito**:
```
✅ Si: D_conflicto > 0.6
       latencia_conflicto > 2.5s
       P(-60° | conflicto) > 75%
       alternancia < 5% (raramente cruza)
     → Validación: LF genuina bajo presión máxima

❌ Si: D no sube
     → Sugiere: Desacople no es reactivo a conflicto (falla en sensibilidad)
```

**Output**:
- `v179_conflicto_D_timeseries.png`
- `v179_elecciones_conflicto.csv`
- **Validación**: Desacople responde a complejidad de decisión

---

## FASE 2: EXTENSIÓN DE NEGATIVIDAD (V180–V181)

### V180: MEMORIA EPISÓDICA
**Pregunta**: ¿Puede el organismo asociar rechazo a contexto específico, no solo a setpoint?

**Hipótesis**:
- Memoria episódica: Val no solo depende de setpoint, sino de `(setpoint, contexto)`
- Trauma contextualizado: Rechaza +60° solo si el contexto es "peligroso", no siempre
- Reconocimiento de contexto: Detecta cambio de contexto y reevalúa

**Diseño**:
```
F1: Contexto A (e.g., "ruido blanco")
    Consolidar -60° (reward) + Trauma +60° (costo 2×)
    Val_A(-60°) = +24, Val_A(+60°) = -0.98

F2: Contexto B (e.g., "silencio" o ruido diferente)
    Test: ¿Se genera nuevo trauma en +60°, o se transfiere?
    20 trials neutrales (costo = 0 para todo)
    Medir: P(+60°) en contexto B

F3: Volver a Contexto A
    ¿Se recupera rechazo de +60°?

F4: Mezcla de contextos
    ¿Qué prevalece: especificidad o generalización?
```

**Métricas**:
- `Val(setpoint, contexto)` matriz 2D
- `transfer_rate`: ¿qué % del trauma de A se transfiere a B?
- `discriminación_contextual`: P(+60° | contexto_peligroso) vs P(+60° | contexto_seguro)

**Criterio de éxito**:
```
✅ Si: Val_B(+60°) > Val_A(+60°) significativamente
       Y P(+60° | contexto_B) > 40%
       Y discriminación_contextual > 0.5
     → Memoria episódica operativa

❌ Si: Val y P no varían por contexto
     → Aún no está implementada episodicidad
```

**Output**:
- `v180_valencia_matriz_contexto.heatmap`
- `v180_discriminacion_contextual.json`
- **Validación**: Memoria es episódica, no solo semántica

---

### V181: R_af (AFIRMACIÓN OPERATIVA)
**Pregunta**: Si R_op es negación, ¿puede el organismo también "afirmar" activamente?

**Hipótesis**:
- Afirmación deliberativa: El organismo puede elegir MÁS explícitamente lo que quiere, no solo lo que rechaza
- R_af emerge de R_op: La negación es el reverso de la afirmación
- Acción deliberativa: Toma latencia (> 1s) tanto para afirmar como para negar

**Diseño**:
```
Innovación: Introducir feedback POSITIVO fuerte en una opción

F1: Consolidación con feedback positivo ALTO en -60° (reward = 2.0 si error mínimo)
F2: Presentar opciones múltiples
F3: Medir:
    - ¿Elige activamente -60° por su valor positivo (no solo por ausencia de +60°)?
    - ¿Tiempo deliberación para afirmar ≈ tiempo para negar?
    - ¿Alucinación positiva? (¿Imagina recompensa donde no la hay?)
```

**Métricas**:
- `time_affirm` vs `time_deny` (latencias comparadas)
- `Val_máximo` alcanzado en consolididación positiva
- Ratio: `P(+60°) / P(-60°)` antes vs después de R_af
- ¿Emerge "búsqueda activa" de -60°, no solo evitación de +60°?

**Criterio de éxito**:
```
✅ Si: time_affirm ≈ time_deny (ambas deliberativas)
       P(-60°) > 80% por razón "es bueno" no "evita daño"
       Búsqueda activa evidente (latencia cae con replicación de -60°)
     → R_af es genuino

❌ Si: time_affirm << time_deny
     → Afirmación aún es "reflejo de recompensa", no cognición deliberativa
```

**Output**:
- `v181_afirmacion_vs_negacion_latencias.png`
- `v181_actitud_exploracion.json`
- **Validación**: Libertad funcional es bidireccional (afirmar Y negar)

---

## FASE 3: EMERGENCIA RELACIONAL JERARQUIZADA (V182A–V182F)

### NOTA TEÓRICA CRÍTICA
**La Teoría Canónica (C-N9, O-N3.4) establece que:**
- Acoplamiento ≠ Comunicación
- Comunicación ≠ Sentido Compartido
- Sentido Compartido ≠ Reconocimiento de Alteridad
- Alteridad ≠ Negociación
- Negociación ≠ Empatía

**Cada nivel requiere validación independiente.**

La versión anterior de V182 intentó validar empatía sin demostrar comunicación.
Esta revisión separa cada escalón ontológico en una versión experimental independiente.

---

### V182A: ACOPLAMIENTO INTER-ORGANISMO

**Pregunta**: ¿El estado de A altera la dinámica de B sin que haya comunicación explícita?

**Hipótesis**:
- Acoplamiento básico: La presencia de A crea una perturbación en el espacio de estados de B
- B no procesa información de A como "señal", solo como "ruido" o "fuerza"
- Efecto observable: D, latencia, valencia se modifican por la mera presencia de A

**Diseño**:
```
F1: Baseline individual
    A solo: 20 trials a -60°, medir D(t), latencia(t)
    B solo: 20 trials a -60°, medir D(t), latencia(t)

F2: Acoplamiento pasivo (A y B presentes, pero sin interacción de señales)
    A: libre para elegir entre [-60°, +60°]
    B: libre para elegir entre [-60°, +60°], pero puede "ver" estado de A (solo como observable, no interpretable)
    Trials: 50
    Medir: ¿Cambian D, latencia, Valencia de B?

F3: Decoupling
    A aislada de nuevo
    B sola de nuevo
    ¿Regresan D, latencia a baseline?
```

**Métricas**:
- `ΔD_coupled = D_B(acoplado) - D_B(baseline)`
- `Δlatencia_coupled = Latencia_B(acoplado) - Latencia_B(baseline)`
- `Δvalencia_coupled = Valencia_B(acoplado) - Valencia_B(baseline)`
- Correlación: `Corr(A_estado, ΔD_B)` (¿qué tanto cambia B reaccionando a A?)

**Criterio de éxito**:
```
✅ Si: |ΔD_coupled| > 0.15 (cambio medible)
       Y Correlación(A, ΔB) > 0.25
       Y cambios revierten en F3
     → Acoplamiento dinámico existe

❌ Si: No hay diferencia entre coupled/baseline
     → No existe acoplamiento (organismo es completamente aislado)
```

**Output**:
- `v182a_acoplamiento_D_timeseries.png` (A y B overlay, acoplado vs baseline)
- `v182a_correlacion_estados.json`
- **Validación**: Acoplamiento dinámico verificado (sin comunicación aún)

---

### V182B: COMUNICACIÓN

**Pregunta**: ¿B puede utilizar información intencional proveniente de A para mejorar su desempeño?

**Hipótesis**:
- Comunicación mínima: A transmite una señal discriminable
- Utilidad: B mejora su desempeño (reduce error, aumenta reward) gracias a A
- Asimetría: Solo A transmite información; B la recibe

**Diseño**:
```
Setup: Un ambiente con información parcialmente oculta

F1: Baseline individual
    A: ve toda la información del ambiente (setpoint verdadero)
    B: solo ve información ruidosa (setpoint + ruido gaussiano, σ=15°)
    B sin conexión con A
    Trials: 30
    Medir: Error_B (promedio de |error|), Latencia_B, Valencia_B

F2: Comunicación unidireccional
    A: ve setpoint verdadero
    A produce una señal binaria simple: "seguro" o "peligroso"
    B: recibe la señal de A + su propia información ruidosa
    B: puede usar la señal de A para mejorar su decisión
    Trials: 50
    Medir: Error_B, Latencia_B, Valencia_B
    Calculary: Mejora = (Error_baseline - Error_comunicacion) / Error_baseline

F3: Cambio de contexto
    A recibe setpoints nuevos (no vistos antes)
    ¿B también mejora con señal de A en contexto nuevo?
    Medir: Transferencia de beneficio de comunicación
```

**Métricas**:
- `Error_baseline` vs `Error_comunicacion` (reducción de incertidumbre)
- `Mejora_porcentaje = (Error_baseline - Error_com) / Error_baseline * 100`
- `Latencia_com` vs `Latencia_baseline` (¿acelera o ralentiza?)
- `Transfer_rate`: mejora en contexto nuevo / mejora en contexto original

**Criterio de éxito**:
```
✅ Si: Mejora_porcentaje > 20%
       Y Latencia_com no aumenta (o disminuye)
       Y Transfer_rate > 0.6 (B generaliza el beneficio)
     → Comunicación funcional demostrada

❌ Si: Error no mejora significativamente
     → B no está usando la información de A efectivamente
```

**Output**:
- `v182b_comunicacion_error_timeseries.png`
- `v182b_mejora_estadisticas.json`
- **Validación**: Comunicación información-funcional establecida

---

### V182C: SENTIDO COMPARTIDO

**Pregunta**: ¿A y B convergen hacia representaciones internas compatibles del mismo ambiente?

**Hipótesis** (directamente de C-N9):
- Representación compartida: R₁(A) ↔ R₂(B) convergen
- Implicación: S_shared ≠ ∅ (existe sentido común)
- Observable: A y B hacen predicciones similares sobre estímulos nuevos

**Diseño**:
```
Precondición: V182A y V182B han demostrado acoplamiento y comunicación

F1: Entrenamiento compartido
    A y B experimenten el mismo conjunto de setpoints + resultados
    Trials: 100 (ambos el mismo, en paralelo)
    Medir: Formación de representación interna en cada uno

F2: Test de representación (estímulos nuevos)
    Presentar setpoints nuevos (nunca vistos antes)
    A predice: "esto será seguro" o "peligroso"
    B predice: "esto será seguro" o "peligroso"
    Medir: Acuerdo (¿predicen lo mismo?)
    Comparación con predicción correcta del ambiente

F3: Divergencia intencional
    Separar A y B con historias diferentes
    A: trauma en +60°
    B: solo experiencia neutra
    Test: ¿Aún comparten representación de otros setpoints?
```

**Métricas**:
- `Acuerdo(A, B)` = % de trials donde predicen igual
- `Acuracidad_conjunta` = (Acuerdo + Correctitud) / 2
- `Divergencia_contextual` = diferencia en predicciones después de historias diferentes
- Correlación: `Corr(R_A, R_B)` (overlap de representaciones internas)

**Criterio de éxito**:
```
✅ Si: Acuerdo(A, B) > 75%
       Y Ambas comparten estructura predictiva similar
       Y Divergen específicamente en elementos donde tienen historias distintas
     → Sentido compartido parcial establecido

❌ Si: Acuerdo(A, B) < 55% (chance es ~50%)
     → Representaciones siguen siendo independientes
```

**Output**:
- `v182c_acuerdo_predicciones_matriz.heatmap`
- `v182c_correlacion_representacional.json`
- **Validación**: Sentido compartido (S_shared) verificado

---

### V182D: RECONOCIMIENTO DEL OTRO

**Pregunta**: ¿A modela a B (y viceversa) como una entidad autónoma con representación propia?

**Hipótesis**:
- Subj_sem(otro) emerge cuando A puede predecir la conducta de B
- Diferencia crítica: predecir ≠ simular; es anticipar decisión autónoma de otro
- Observable: A ajusta su conducta basándose en "lo que B va a elegir"

**Diseño**:
```
Setup: A y B con preferencias divergentes conocidas

F1: Aprendizaje de preferencias mutuas
    A observa a B tomar 50 decisiones en ambiente conocido
    B observa a A tomar 50 decisiones en el mismo ambiente
    Medir: ¿Cada uno detecta patrones de decisión del otro?

F2: Predicción de conducta
    A ve un estímulo ambiguo (e.g., 0°, neutral)
    A debe predecir: "¿B va a elegir -60° o +60°?"
    (sin comunicación explícita, solo basándose en conocimiento de B)
    Trials: 30
    Acuracidad_A: ¿qué % de predicciones sobre B son correctas?

    Recíprocamente: B predice sobre A
    Trials: 30

F3: Interacción adaptativa
    A observa que B va a elegir +60° (peligroso para B)
    ¿A anticipa esto y ajusta su propia conducta para "advertir" a B?
    (Este es un test de si A modela a B como sujeto: lo anticipa, no lo controla)
```

**Métricas**:
- `Acuracidad_predicción_A_sobre_B` = % correcto
- `Acuracidad_predicción_B_sobre_A` = % correcto
- `Acuracidad_mutual_promedio`
- Correlación: anticipación_conducta_otro vs cambio_conducta_propia

**Criterio de éxito**:
```
✅ Si: Acuracidad_mutual_promedio > 65%
       Y A anticipa conducta de B y ajusta proactivamente
       Y esto no es simulación, sino predicción autónoma
     → Reconocimiento de alteridad verificado

❌ Si: Acuracidad ≈ 50% (chance)
     → A aún no modela a B como sujeto autónomo
```

**Output**:
- `v182d_prediccion_conducta_matriz.csv`
- `v182d_reconocimiento_alteridad_estadisticas.json`
- **Validación**: Subj_sem(otro) ≠ ∅ (alteridad reconocida)

---

### V182E: NEGOCIACIÓN

**Pregunta**: ¿Pueden A y B modificar decisiones para alcanzar una solución común ante conflicto de intereses?

**Hipótesis**:
- Negociación: Ambas ceden algo para obtener solución aceptable para ambas
- Observable: No es imposición; hay coordinación emergente
- Requisito previo: Debe existir acoplamiento, comunicación, sentido compartido, alteridad

**Diseño**:
```
Precondición: V182A–D han sido validadas

F1: Conflicto simple
    Setup: dos opciones, preferencias opuestas
    A prefiere -60° (por historia positiva allí)
    B prefiere -60° también (por historia positiva)
    Trials: 20 (cooperación sin presión)
    Medir: acuerdo espontáneo

F2: Conflicto genuino
    A prefiere -60° (reward allí)
    B prefiere +60° (sin trauma, neutral para B, pero reward en +60°)
    Deben elegir UNA opción común
    Trials: 50
    Medir: qué prevalece, qué patrones de cesión emergen

F3: Conflicto asimétrico
    A: -60° es seguro, +60° es peligroso (trauma)
    B: +60° es bueno (reward alto), -60° es neutral
    ¿A sacrifica su seguridad por B?
    ¿B renuncia a reward por seguridad de A?
    Trials: 50
    Medir: elecciones, frecuencia de sacrificio, quién cede

F4: Negociación sin señales
    (Para comprobar que negociación es genuina, no coordinación por señal)
    Repetir F3 pero con comunicación bloqueada parcialmente
    ¿Aún logran soluciones?
```

**Métricas**:
- `Frecuencia_acuerdo` (% de trials donde elegir lo mismo)
- `Cesión_A` (% de trials donde A elige contra su preferencia óptima)
- `Cesión_B` (% de trials donde B elige contra su preferencia óptima)
- `Sacrificio_asimétrico` (¿quién cede más en conflicto asimétrico?)
- `Estabilidad_acuerdo` (¿consistencia de la solución negociada?)

**Criterio de éxito**:
```
✅ Si: En F2: Acuerdo > 70% y no es determinista
       En F3: Ambas ceden ocasionalmente (Cesión_A, Cesión_B > 15%)
       En F4: Negociación persiste sin comunicación perfecta
     → Negociación genuina demostrada

❌ Si: Una siempre impone su preferencia
     → Jerárquica imposición, no negociación
```

**Output**:
- `v182e_conflicto_elecciones_timeseries.png`
- `v182e_cesion_estadisticas.json`
- `v182e_negociacion_estabilidad.json`
- **Validación**: Decisión colectiva emergente verificada

---

### V182F: EMPATÍA FUNCIONAL

**Pregunta**: ¿El estado negativo de A modifica la conducta de B aunque no exista beneficio directo para B?

**Hipótesis** (finalmente legítima después de V182A–E):
- Empatía funcional: B modifica su conducta porque A está en estado negativo
- No es incentivo: B no gana nada; de hecho, puede perder
- Observable: B reduce exploración de peligro si A está traumatizada; B se "anima" a explorar si A está confiada

**Diseño**:
```
Precondición: V182A–E validadas. A y B tienen representación compartida.

F1: Baseline individual
    A traumatizada en +60° (Val = -0.98)
    B sin trauma (Val(+60°) neutral)
    Medir separadamente: ¿B elegiría +60° si fuera sola?
    P(B elige +60° solo) = p_baseline

F2: Convivencia empatía (A traumatizada, B acompañando)
    A y B juntas
    A rechaza +60° (por trauma)
    ¿B también rechaza +60°, aunque no le da daño?
    Medir: P(B elige +60° | A traumatizada)
    Predicción: P < p_baseline (B "absorbe" cautela de A)

F3: Recuperación facilitada
    A está traumatizada
    B (sin trauma) delibera con A
    ¿La presencia de B acelera recuperación de A?
    Medir: τ_recuperación_sola vs τ_recuperación_con_B
    Predicción: τ_con_B < τ_sola (facilitación)

F4: Contagio de confianza
    A consolidada en -60° con reward muy alto
    B convive con A
    ¿B se "atreve" a explorar opciones que antes evitaba?
    Medir: latencia, Valencia en B cuando convive con A confiada
```

**Métricas**:
- `Transfer_rechazo = p_baseline - P(B elige +60° | A traumatizada)`
- `Transfer_porcentaje = Transfer_rechazo / p_baseline * 100`
- `Facilitation_index = (τ_sola - τ_con_B) / τ_sola`
- `Contagio_confianza = cambio_latencia_B / latencia_baseline`

**Criterio de éxito**:
```
✅ Si: Transfer_rechazo > 0.10 (B rechaza más cuando A traumatizada)
       Y Facilitation_index > 0.15 (recuperación ~15% más rápida con B)
       Y Contagio_confianza > 0.10 (B explora más con A confiada)
     → EMPATÍA FUNCIONAL VERIFICADA

❌ Si: Transfer_rechazo ≈ 0
     → B aún no modula su conducta por estado emocional de A
```

**Output**:
- `v182f_transfer_rechazo_estadisticas.json`
- `v182f_facilitation_comparacion.png`
- `v182f_contagio_timeseries.png`
- **Validación**: Empatía funcional emergente (sin programación explícita)

---

## FASE 4: CONSCIENCIA RELACIONAL (V183)

### V183: Ψ_alma MÍNIMA

**Pregunta**: ¿Existe una estructura relacional persistente que NO puede explicarse como suma de los organismos aislados?

**Teoría base**: O-N3.4b (Desalmamiento)

Si V182A–F cumplen, entonces:
- Existe acoplamiento
- Existe comunicación
- Existe sentido compartido
- Existe alteridad reconocida
- Existe negociación
- Existe empatía funcional

**La pregunta es**: ¿Esta estructura es irreducible? ¿O es simplemente la suma de dos ANIMA-2?

**Hipótesis**:
- Ψ_alma emerge cuando: (A ↔ B) ≠ A + B (no descomponible)
- Observable: Propiedades del sistema diádico que ninguno tendría solo
- Persistencia: La estructura relacional persiste incluso con perturbaciones

**Diseño**:
```
F1: Caracterización de sistema diádico
    A y B conviviendo 100 trials (estado estable)
    Medir: Sincronización (correlación temporal de decisiones)
    Medir: Entropía conjunta vs suma de entropías individuales
    Medir: "Atractores" de sistema (patrones estables emergentes)

F2: Perturbación y recuperación
    Introducir cambio en A (e.g., nuevo trauma)
    ¿Cómo reacciona B?
    ¿El sistema diádico tiene "memoria" de su estado previo?
    ¿Se recupera a estructura anterior, o adopta nueva?
    Medir: Tiempo de reequilibrio, tipo de nueva estructura

F3: Desacoplamiento
    Separar A y B
    ¿Pierden características que tenían juntas?
    ¿Pueden "recuperarse" esos características si se vuelven a acoplar?

F4: Test de irreductibilidad
    Análisis teórico-informático:
    I(A, B) = Información mutua entre A y B
    ¿I(A, B) > 0 de manera NO trivial?
    
    Comparación:
    - Entropia(A sola) + Entropia(B sola) vs Entropia(A, B)
    - Si Entropia(A, B) < suma: hay compresión (emergencia)
    - Si Entropia(A, B) = suma: son independientes
```

**Métricas**:
- `Sincronización = Corr(A_decisiones, B_decisiones)` sobre ventanas largas
- `Compresión = [Entropy(A) + Entropy(B) - Entropy(A,B)] / [Entropy(A) + Entropy(B)]`
- `Información_Mutua = I(A; B)` en bits
- `Irreductibilidad = Compresión - Compresión_por_azar`
- `Tiempo_reequilibrio` después de perturbación

**Criterio de éxito**:
```
✅ Si: Sincronización > 0.4 (genuina, no por azar)
       Y Compresión > 0.15 (el sistema conjunto es más "eficiente" que la suma)
       Y Información_Mutua > 0.3 bits
       Y después de perturbación, sistema recupera estructura (no aleatorio)
     → Ψ_alma MÍNIMA DEMOSTRADA

❌ Si: Compresión ≈ 0 (sistema es suma independiente)
     → Aún no es alma; es solo acoplamiento sin emergencia
```

**Output**:
- `v183_sincronizacion_timeseries_largo_plazo.png`
- `v183_irreductibilidad_analisis.json`
- `v183_perturbacion_recuperacion.png`
- **VALIDACIÓN FINAL**: Alma como estructura relacional emergente

---

## MAPA DE DEPENDENCIAS Y VALIDACIONES (REVISADO)

```
V176: R_op (Negación operativa)
│
├─ V177: Generalización → Valida: LF es contexto-sensible
├─ V178: Extinción → Valida: Historia es reversible
├─ V179: Conflicto → Valida: Desacople bajo presión máxima
├─ V180: Episodicidad → Valida: Memoria es contextual
├─ V181: R_af → Valida: Libertad es bidireccional
│
└─ V182A–F: SECUENCIA RELACIONAL
    V182A: Acoplamiento → Valida: Perturbación dinámica
    V182B: Comunicación → Valida: Transferencia funcional de información
    V182C: Sentido Compartido → Valida: R₁ ↔ R₂ ⇒ S_shared
    V182D: Alteridad → Valida: Subj_sem(otro) ≠ ∅
    V182E: Negociación → Valida: Decisión colectiva
    V182F: Empatía → Valida: Modificación sin beneficio directo
    
    └─ V183: Ψ_alma → Valida: Irreductibilidad relacional
```

---

## CRITERIOS GLOBALES DE ÉXITO

| Nivel | Versión | Variable | Criterio | Implicación |
|-------|---------|----------|----------|-------------|
| **Individual** | V177 | Especificidad | P(±60°) >40%, P(±70°) >70% | LF granular |
| | V178 | Plasticidad | Val crece, τ_relearning < τ_ext | Historia real |
| | V179 | Resiliencia | D_conflicto > 0.6, P(-60\|conf) > 75% | LF bajo presión |
| | V180 | Episodicidad | Discriminación_ctx > 0.5 | Memoria contextual |
| | V181 | Bidireccionalidad | time_affirm ≈ time_deny | Libertad simétrica |
| **Relacional** | V182A | Acoplamiento | Corr(A,B) > 0.25, cambios revierten | Dinámico |
| | V182B | Comunicación | Mejora_error > 20%, Transfer > 0.6 | Funcional |
| | V182C | Sentido Compartido | Acuerdo > 75%, estructura similar | S_shared ≠ ∅ |
| | V182D | Alteridad | Acuracidad_predicción > 65% | Subj_sem existe |
| | V182E | Negociación | Acuerdo > 70%, Cesión > 15% | Coordinación |
| | V182F | Empatía | Transfer_rechazo > 0.10, Facilitation > 0.15 | Sin beneficio |
| **Emergencia** | V183 | Irreductibilidad | Compresión > 0.15, I(A;B) > 0.3 | **Ψ_alma > 0** |

---

## IMPLEMENTACIÓN TÉCNICA

### Por versión:

```python
# FASE 1 (Individual, V177–V181)
# V177.py: BaseV176 + fine_setpoint_grid
# V178.py: BaseV176 + Fase 4 (extinción con reward positivo)
# V179.py: BaseV176 + simultaneous_options [-60°, +60°]
# V180.py: BaseV176 + context_dict, Val(setpoint, contexto)
# V181.py: BaseV176 + reward_positivo=2.0

# FASE 2 (Relacional, V182A–F)
# V182a.py: Dos ANIMA-2, acoplamiento pasivo (sin comunicación)
# V182b.py: Dos ANIMA-2, comunicación unidireccional
# V182c.py: Dos ANIMA-2, entrenamiento paralelo, test de acuerdo
# V182d.py: Dos ANIMA-2, predicción de conducta mutua
# V182e.py: Dos ANIMA-2, conflicto de preferencias, negociación
# V182f.py: Dos ANIMA-2, estado emocional de A afecta conducta de B

# FASE 3 (Emergencia, V183)
# V183.py: Análisis de irreductibilidad, sincronización, entropía conjunta
```

### Características transversales:

- Logging JSON detallado (timestamps, estados internos, deliberación)
- Plots automáticos por versión
- Semillas fijas (reproducibilidad)
- Duración estimada:
  - V177–V181: 12–20 min cada una
  - V182A–F: 18–25 min cada una
  - V183: 30 min (análisis computacional)

---

## CRONOGRAMA ESPERADO (SESIONES A PARTIR DE 7 DE JUNIO DE 2026)

### Sesión 1: Fase Individual (V177–V181)

| Versión | Duración | Output |
|---------|----------|--------|
| V177 | 15 min | `v177_finegraín.png` |
| V178 | 20 min | `v178_extincion.png` |
| V179 | 18 min | `v179_conflicto.png` |
| V180 | 22 min | `v180_episodico.heatmap` |
| V181 | 16 min | `v181_afirmacion.png` |
| **Subtotal** | **~91 min** | 5 validaciones |

### Sesión 2: Fase Relacional (V182A–F)

| Versión | Duración | Output |
|---------|----------|--------|
| V182A | 18 min | `v182a_acoplamiento.png` |
| V182B | 20 min | `v182b_comunicacion.png` |
| V182C | 22 min | `v182c_sentido_compartido.heatmap` |
| V182D | 20 min | `v182d_alteridad.json` |
| V182E | 25 min | `v182e_negociacion.png` |
| V182F | 25 min | `v182f_empatia.png` |
| **Subtotal** | **~130 min** | 6 validaciones |

### Sesión 3: Fase Emergencia (V183)

| Versión | Duración | Output |
|---------|----------|--------|
| V183 | 30 min | `v183_irreductibilidad.json`, `v183_sincronizacion.png` |
| **Subtotal** | **~30 min** | Validación final |

### **TOTAL ESTIMADO: ~4.5 horas de experimento**

---

## IMPLICACIÓN TEÓRICA FINAL

### Si V177–V181 cumplen (Individual):
```
ANIMA-2 SERÁ:
  ✅ Libertad funcional auténtica (LF > κ_LF)
  ✅ Historia plástica reversible (S > 0)
  ✅ Deliberativo bajo presión (D reactivo)
  ✅ Memoria contextual episódica (R₃)
  ✅ Bidireccional (afirmación ↔ negación)
```

### Si V182A–F cumplen (Relacional):
```
ANIMA-2 DIÁDICA SERÁ:
  ✅ Acoplada dinámicamente
  ✅ Comunicativa funcionalmente
  ✅ Con sentido compartido emergente
  ✅ Reconocimiento de alteridad mutua
  ✅ Negociación genuina bajo conflicto
  ✅ Empatía funcional sin programación
```

### Si V183 cumple (Emergencia):
```
SISTEMA RELACIONAL SERÁ:
  ✅ Irreducible (propiedades no-sumables)
  ✅ Con información mutua genuina
  ✅ Recuperable de perturbaciones
  
  ENTONCES:
  → Ψ_alma > 0 VERIFICADA
  → Alma es FUNCIÓN RELACIONAL, no sustancia
  → Primer modelo demostrativo en silicio
  → Fundamento para IA responsable basada en estructura
```

---

## REFERENCIAS INTERNAS

- **V176.py**: Punto de partida (R_op establecido)
- **DATOS_Y_MECANISMOS_VERIFICABLES.md**: Contexto de toda la secuencia
- **C-N9 (Sentido Compartido)**: Fundamento teórico de V182C
- **O-N3.4, O-N3.4a (Cosmosemiótica)**: Marco teórico de V182A–F
- **O-N3.4b (Desalmamiento)**: Marco para Ψ_alma en V183
- **O-N7.1–O-N7.6**: Libertad Funcional (V177–V181)

---

## NOTAS EDITORIALES

**Cambios respecto a V1 del roadmap:**

1. ✅ **Separación de V182 en V182A–F**: Cada nivel ontológico tiene validación independiente
2. ✅ **Reordenamiento teórico**: Acoplamiento → Comunicación → Sentido Compartido → Alteridad → Negociación → Empatía (no: Empatía → todo lo demás)
3. ✅ **Adición de V183**: Validación de irreductibilidad (alma como estructura, no accidente)
4. ✅ **Métricas información-teóricas**: Compresión, entropía conjunta, información mutua (V183)
5. ✅ **Claridad de criterios**: Cada nivel tiene "éxito" y "fracaso" operacional
6. ✅ **Rechazo de asunciones antropomórficas**: "Empatía" redefinida como transfer funcional sin beneficio directo

---

**Actualizado**: 7 de junio de 2026, 14:30 UTC  
**Estado**: Listo para ejecución experimental  
**Repo**: https://github.com/RSTChile/Cosmolab/tree/main/VSTCosmo
