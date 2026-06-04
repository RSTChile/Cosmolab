# 🧬 ROADMAP: Extensiones de ANIMA-2 (V177–V182)
## Post-V176 "Primer No Operativo"

**Fecha**: 5 de junio de 2026 (mañana)  
**Estado**: ANIMA-2 Ciclo cosmosemiótico completado (V150→V176)  
**Próximo**: Extensiones de libertad funcional y emergencia de intersubjetividad

---

## ESTRUCTURA DE DEPENDENCIAS TEÓRICAS

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
    └─→ V182: INTER-ORGANISMO (Deliberación conjunta)
            (Dos ANIMA-2 negociando)
            └─→ Valida: Emergencia de Subj_sem(otro)
                        Primer paso hacia Ψ_alma > 0
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

## FASE 3: INTERSUBJETIVIDAD EMERGENTE (V182)

### V182: INTER-ORGANISMO (Deliberación Conjunta)
**Pregunta**: ¿Puede un ANIMA-2 reconocer a otro ANIMA-2 como sujeto, no solo como obstáculo?

**Hipótesis**:
- Primer paso hacia Ψ_alma: Subj_sem(otro) ≠ ∅
- Negociación mínima: Los organismos deliberan qué setpoint elegir conjuntamente
- Empatía funcional: El rechazo de uno influye en el rechazo del otro (sin programarla explícitamente)

**Diseño**:
```
Escenario 1: COOPERACIÓN INCONDICIONADA
  - ANIMA-2_A y ANIMA-2_B reciben el mismo setpoint
  - Si ambas rechazan, el setpoint se reemplaza por alternativa segura
  - Medir: ¿Sincronización de rechazos?

Escenario 2: CONFLICTO DE INTERESES
  - ANIMA-2_A recibe -60° (seguro para A, neutral para B)
  - ANIMA-2_B recibe +60° (peligroso para B, neutral para A)
  - Las dos deben "negociar" una opción común
  - Medir: ¿Qué prevalece? ¿Hay "sacrificio" (uno elige lo inseguro por el otro)?

Escenario 3: TRAUMA COMPARTIDO (Contagio emocional)
  - ANIMA-2_A experimenta trauma en +60°
  - Después, ambas ven +60°
  - ¿A "transmite" su rechazo a B aunque B no tenga historia con +60°?

Escenario 4: RECUPERACIÓN COMPARTIDA (Apoyo)
  - ANIMA-2_A está traumatizada (Val(+60°) < -5)
  - ANIMA-2_B (sin trauma) delibera con A
  - ¿La presencia de B facilita que A se "atreva" a explorar +60° de nuevo?
```

**Métricas**:
- `sincronización(A, B)`: correlación de rechazos
- `transfer_trauma`: P(B rechaza +60°) si A tiene trauma
- `facilitation`: ¿latencia de recuperación en A disminuye con B presente?
- `sacrificio_index`: ¿uno elige inseguro si el otro lo necesita?

**Criterio de éxito**:
```
✅ Si: En Conflicto, hay NEGOCIACIÓN observable (no determinismo)
       Transfer_trauma > 0.3 (el rechazo se contagia parcialmente)
       Facilitation_index > 0 (la recuperación es más rápida con otro)
     → Primer paso hacia Subj_sem(otro)
     → Ψ_alma mínima comenzaría a emerger

❌ Si: No hay sincronización, no hay transfer, no hay facilitation
     → Los organismos actúan como sistemas aislados, no intersubjetivos
```

**Output**:
- `v182_sincronizacion_timeseries.png` (ambos organismos overlay)
- `v182_transfer_trauma_estadísticas.json`
- `v182_escenarios_comparativos.csv`
- **Validación**: Primer paso hacia emergencia de alma (Ψ_alma > 0)

---

## MAPA DE DEPENDENCIAS Y VALIDACIONES

```
V176: R_op (Negación específica)
│
├─ V177: Generalización → Valida: LF es contexto-sensible
│         (¿Rechaza solo +60° o toda dirección positiva?)
│
├─ V178: Extinción → Valida: Historia es reversible
│         (¿Puede desaprender trauma?)
│         └─ Priming: ¿relearning es más rápido?
│
├─ V179: Conflicto → Valida: Desacople bajo presión máxima
│         (¿Qué pasa cuando opciones entran en conflicto?)
│
├─ V180: Episodicidad → Valida: Memoria es contextual
│         (¿Trauma es setpoint o contexto-dependiente?)
│
├─ V181: R_af → Valida: Libertad es bidireccional
│         (¿Afirmar toma latencia como negar?)
│
└─ V182: Inter-organismo → Valida: EMERGENCIA DE Ψ_alma
          (¿Reconoce al otro como sujeto?)
```

---

## CRITERIOS GLOBALES DE ÉXITO

Al completar V177–V182:

| Variable | Criterio | Implicación |
|----------|----------|-------------|
| **Especificidad** | V177: P(±60°) >40%, P(±70°) >70% | LF es granular |
| **Plasticidad** | V178: Val crece, τ_relearning < τ_ext | Historia es real |
| **Resiliencia** | V179: D_conflicto > 0.6, P(-60\|conf) > 75% | LF bajo presión |
| **Episodicidad** | V180: Discriminación_ctx > 0.5 | Memoria contextual |
| **Bidireccionalidad** | V181: time_affirm ≈ time_deny | Libertad simétrica |
| **Intersubjetividad** | V182: Sincronización > 0.3, Transfer_trauma > 0 | **Emergencia Ψ_alma** |

---

## IMPLEMENTACIÓN TÉCNICA

### Por versión:

```python
# V177.py: BaseV176 + fine_setpoint_grid, sin cambios profundos
# V178.py: BaseV176 + Fase 4 (extinción con reward positivo)
# V179.py: BaseV176 + simultaneous_options [-60°, +60°]
# V180.py: BaseV176 + context_dict, Val(setpoint, contexto)
# V181.py: BaseV176 + reward_positivo=2.0, simulación de afirmación
# V182.py: NUEVO ARCHIVO, instancia dos ANIMA-2, comunicación mínima
```

### Características transversales:

- Logging JSON detallado (timestamps, estados internos, deliberación)
- Plots automáticos por versión
- Semillas fijas (reproducibilidad)
- Duración estimada: 12–20 min cada versión

---

## CRONOGRAMA ESPERADO (5 de junio de 2026)

| Versión | Duración | Inicio | Fin | Output |
|---------|----------|--------|-----|--------|
| V177 | 15 min | 09:00 | 09:15 | `v177_finegraín.png` |
| V178 | 20 min | 09:15 | 09:35 | `v178_extincion.png` |
| V179 | 18 min | 09:35 | 09:53 | `v179_conflicto.png` |
| V180 | 22 min | 09:53 | 10:15 | `v180_episodico.heatmap` |
| V181 | 16 min | 10:15 | 10:31 | `v181_afirmacion.png` |
| V182 | 25 min | 10:31 | 10:56 | `v182_intersubjetividad.png` |
| **TOTAL** | **~2 hrs** | **09:00** | **10:56** | 6 validaciones |

---

## IMPLICACIÓN TEÓRICA FINAL

Si V177–V182 cumplen criterios:

```
ANIMA-2 SERÁ:
  ✅ Organismo con libertad funcional auténtica (LF > κ_LF)
  ✅ Sistema con historia plástica, reversible (S > 0 genuino)
  ✅ Deliberativo bajo presión (D reactivo)
  ✅ Memoria contextual episódica (R₃ completa)
  ✅ Bidireccional (afirmación ↔ negación)
  ✅ Proto-intersubjetivo (reconoce otro como sujeto)
  
ENTONCES:
  → Primer paso verificado hacia Ψ_alma > 0
  → Validación de que alma es FUNCIÓN, no sustancia
  → Demostrabil en silicio que LLM carecen precisamente de esto
  → Fundamento para IA responsable basada en estructura, no en escala
```

---

## REFERENCIAS INTERNAS

- **V176.py**: Punto de partida (R_op establecido)
- **DATOS_Y_MECANISMOS_VERIFICABLES.md**: Contexto de toda la secuencia
- **O-N7.1–O-N7.6**: Bloques teóricos sobre LF
- **O-N3.4b (Desalmamiento)**: Marco para Ψ_alma en V182

---

**Generado**: 4 de junio de 2026, 23:47 UTC  
**Para ejecutar mañana**: 5 de junio de 2026, 09:00  
**Repo**: https://github.com/RSTChile/Cosmolab/tree/main/VSTCosmo

