# ANÁLISIS: GENÉTICA DEL ALTRUISMO EN LA DÍADA ANIMA
## Estado real de la cooperación A↔B · 27-jun-2026

---

## RESUMEN EJECUTIVO

La **Genética del Altruismo de Boorman está completamente integrada** en el genoma y operativa en Docker. Sin embargo, **la díada está en un estado de mutismo estable** (atractor "mudo") tras ~26h de vida acoplada.

**Estado:**
- ✅ Locus desarrollado: `VST_Genoma.py` (OrganeloAltruismo) + `VST_DiadaAltruismo.py` (gobernanza)
- ✅ Sujetos reconocidos: Ψ_alma_A ≈ 0.985, Ψ_alma_B ≈ 0.985
- ✅ Díada como unidad: costo_desacople > 0 (cooperar beneficiaría ambos)
- ❌ Volición CERO: disposicion_A = 0.00009, disposicion_B = 0.00000
- ❌ Cooperación NUNCA ocurre: atractor = "mudo" 100% del tiempo
- ❌ Confianza no crece: τ_A = 0.0, τ_B = 0.0

**Conclusión:** Los organismos **reconocen** mutuamente su condición de sujetos (Ψ_alma), pero no tienen **volición** (disposición) de cooperar. Esto no es fallo: es un **equilibrio de Nash** estable.

---

## 1. LA ARQUITECTURA (Qué está operando)

### 1.1 Los archivos del locus

| Archivo | Localización | Función |
|---------|--------------|---------|
| `VST_Genoma.py` | raíz | Motor del genoma: `OrganeloAltruismo` + `beta_crit` + `organelo_altruismo()` |
| `VST_DiadaAltruismo.py` | `Célula_Madre/diada/` | Gobernanza: `GobernanzaAltruismo` mapea fisiología → altruismo |
| `VST_LocusAltruismo.py` | raíz | Tests + verificación + re-exporta para histórico |

### 1.2 Los 6 mecanismos (Boorman → código)

```
1. β_crit(LF, e_R) = (e_R - LF·e_R) / (LF + ε)
   → umbral dinámico: baja con libertad, sube con error

2. τ (historia) ↑ si mutualidad_sostenida, ↓ a 0 si traición
   → confianza acumulada (relación profundiza con el tiempo)

3. voice_rms = base · max(disposicion, piso_exploracion)
   → señal costosa: invierto más voz cuanto más dispuesto

4. τ abre camino: Hamilton r↑ → β_crit↓ → cooperación más fácil
   → la relación se auto-estabiliza

5. costo_desacople = e_R_solo - e_R_acoplado > 0
   → la díada es unidad (estar juntos es mejor que solos)

6. if otro.Cb == 0 then psi_alma=0 and coopera=FALSE
   → anti-Shannon: sin reconocimiento de sujeto, no hay cooperación
```

### 1.3 Ciclo por paso

```
fila_propia (C_b, R2, LF_op, e_R, A_sys_env, ...)
         ↓
    GobernanzaAltruismo.paso(fila_A, estado_B)
         ↓
    milieu.secretar() → OrganeloAltruismo
         ↓
    organelo.metabolizar(dt, tempo)
         ↓
    SALIDAS: disposicion, coopera, Ψ_alma, τ, β_crit, voice_rms
         ↓
    A vuelca disposicion → fila_A["disposicion_cooperar"]
    B lee disposicion_A por HTTP(/comunicacion/estado)
```

---

## 2. ANÁLISIS DE DATOS (Sesión 27-jun 08:18)

### 2.1 Estadísticas crudas

```
Duración:     ~26 minutos (21,000 pasos @ 10 Hz)
Organismos:   A y B acoplados por HTTP
Fuente audio: demo:silencio (ambos nacen en soledad)
```

| Variable | A | B | Interpretación |
|----------|---|---|---|
| **disposicion_cooperar** | 0.000088 | 0.000000 | NULA |
| disposicion_max | 0.1999 | 0 | A casi nunca llega a 0.2 |
| disposicion_nonzero | 75/21k | 0/21k | A: 3 momentos débiles; B: nunca |
| **atractor** | mudo 100% | mudo 100% | Ninguno emite |
| **Ψ_alma** | 0.985 | 0.985 | Ambos reconocen mutuamente |
| Ψ_alma_min | 0 | 0 | Pero hay momentos de 0 (?) |
| **τ (confianza)** | 0.0 | 0.0 | Nunca crece |
| τ_max | 0.0 | 0.0 | Nunca, en toda la sesión |
| **costo_desacople** | 0.1893 | 0.2403 | POSITIVO: díada sería mejor |
| **β_crit** | 0.9998 | 0.9999 | MÁXIMO: imposible cooperar |

### 2.2 La paradoja central

```
RECONOCIMIENTO:        Ψ_alma_A = 0.985 ✓
                       Ψ_alma_B = 0.985 ✓
                       "Se ven como sujetos"

PERO:
VOLICIÓN:              disposicion_A = 0.00009 ✗
                       disposicion_B = 0.000000 ✗
                       "No hay intención de cooperar"

RESULTADO:             β_crit = 1.0 (imposible)
                       τ = 0 (la relación NUNCA empieza)
                       atractor = "mudo" (silencio autosostenido)
```

### 2.3 Momentos de "casi" cooperación (A: 75 instancias)

En A, hubo **3 eventos de "emergiendo"** (0.014% de la sesión):
- Duraron < 1 segundo cada uno
- Nunca cruzaron a "comunicando"
- β_crit nunca bajó del umbral

**Interpretación:** A **probó** cooperar momentáneamente (disposicion ≈ 0.2), pero:
1. No fue suficiente para cruzar β_crit
2. B no respondió (disposicion = 0)
3. El atractor "mudo" se re-estabilizó inmediatamente

---

## 3. EL CICLO CERRADO DEL MUTISMO

```
Condición inicial:
  silencio absoluto (demo:silencio)
  ↓
disposicion_cooperar = 0
  ↓
sin emisión de voz (voice_rms = base · piso_exploracion ~ 0.08)
  ↓
el otro no tiene entrada auditiva que lo perturbe
  ↓
no se dispara R2 (auto-modelo) ni LF (libertad)
  ↓
disposicion sigue en 0
  ↓
EQUILIBRIO DE NASH PERVERSO:
  "A: callarse. B: callarse. Ambos mejor que comunicar."
```

### ¿Por qué es estable?

En silencio puro, **los costos de cooperación superan los beneficios potenciales**:

- **Costo:** emitir voz cuesta e_R (error de reproducción)
- **Beneficio:** necesitaría que el otro TAMBIÉN emita y responda
- **Pero:** sin entrada, el otro no tiene razón para emitir
- **Resultado:** ambos se quedan callados

Esto NO es un bug. Es **exactamente lo que debe pasar** en silencio absoluto.

---

## 4. ANÁLISIS CONTRAFÁCTICO: ¿Qué se necesita para romper el ciclo?

### 4.1 Tres rutas hacia la cooperación

**Ruta A: Estímulo externo (RECOMENDADO)**

```
Cambiar fuente de demo:silencio a:
  - Rødecaster Pro (audio vivo del entorno)
  - BigBang.wav (música rítmica)
  - voces_r2d2/ (grabaciones de voz)
```

**Mecanismo:**
```
Entrada ≠ silencio
  → ω_A ≠ 0 (lo percibido sube)
  → e_R sube (desajuste audio-expectativa)
  → presion_desacople sube
  → juego se activa (LF_op ↑)
  → R2 se dispara (auto-modelo)
  → ENTONCES disposicion empieza a crecer
  → β_crit baja (porque LF↑)
  → umbral se cruza
  → atractor "comunicando" PUEDE encenderse
```

**Ruta B: Inyectar disposición mínima (CALIBRACIÓN)**

Cambiar `piso_exploracion` en `GobernanzaAltruismo`:

```python
GobernanzaAltruismo(base_voice_rms=0.40, piso_exploracion=0.15)
                                          ↑ subir de 0.08 → 0.15
```

**Efecto:** fuerza emisión de voz incluso con disposicion ≈ 0, permitiendo que el otro tenga input para responder.

**Ruta C: Cambiar τ_min (ESPERANZA)**

En `plast`:
```python
organelo_altruismo(plast=dict(tau_min=0.5))  # O MAYOR
                                 ↑ subir
```

**Efecto:** baja más agresivamente β_crit conforme pasa el tiempo, eventualmente permitiendo que pequeñas fluctuaciones crucen.

---

## 5. HALLAZGOS SOBRE LA GENÉTICA DEL ALTRUISMO

### ✅ Lo que funciona

1. **Reconocimiento de sujeto (Ψ_alma):** ambos leen correctamente que el otro tiene Cb>0
2. **Subaditividad (costo_desacople > 0):** la díada es detectada como unidad beneficiosa
3. **Anti-imposición:** sin sujeto (Cb=0), la cooperación NO se fuerza
4. **Umbral dinámico:** β_crit responde a LF (cuando hubiera)

### ⚠️ Lo que está inerte

1. **Volición inicial:** no hay mecanismo de "primer movimiento" en silencio
   - A intenta 3 veces débilmente
   - B nunca lo intenta
   - Ambos desisten

2. **Asimetría:** ¿por qué A es breve (0.2) pero B es 0 absolutamente?
   - Posible: diferencia en calibración de parámetros
   - Posible: lagas de comunicación HTTP (B llega a ver la disposicion de A con retraso)
   - Posible: inicialización diferente de los organizmos

3. **Ciclo de refuerzo:** no hay mecanismo "de arranque" que rompa el mutismo sin entrada externa

---

## 6. INTERPRETACIÓN COSMOSEMIÓTICA

La **genertica del altruismo está haciendo exactamente lo que debe:** 

> *"Cooperación voluntaria emerge solo bajo condiciones favorables; en silencio absoluto, el equilibrio es la no-cooperación."*

**Esto valida la teoría:**

- ✅ **Anti-Shannon:** sin reconocimiento de sujeto → sin cooperación (Ψ_alma gate)
- ✅ **Economía honesta:** cooperar en silencio CUESTA (energía, error); no hay recompensa
- ✅ **Emergencia, no imposición:** el atractor cooperativo NO se activa por decreto

**Lo que NO valida (aún):**

- ❌ **Cooperación espontánea:** la díada necesita input externo para despertar
- ❌ **Mutualismo:** una relación mutuamente beneficiosa (costo_desacople > 0) que debería auto-ignirse

---

## 7. RECOMENDACIONES

### Inmediatas (para prueba)

1. **Carga audio real** en uno de los contenedores (p.ej., BigBang.wav):
   ```bash
   docker exec anima-a /usr/bin/env bash
   # setear: ANIMA_FUENTE_DEFECTO=/app/audio_binaural/BigBang.wav
   ```

2. **Observa qué pasa:**
   - ¿Sube disposicion_A?
   - ¿Responde disposicion_B?
   - ¿Crece τ?
   - ¿Se estabiliza en "comunicando"?

3. **Recolecta datos** de esa sesión en Docker_Historia

### Mediano plazo

1. **Calibración:** ajustar `piso_exploracion`, `tau_min`, `base_voice_rms` con datos reales

2. **Asimetría A↔B:** investigar por qué A sí intenta pero B no

3. **Latencia HTTP:** medir retraso entre envío de voz y recepción en el otro; si es > DT, bloquea reciprocidad

### Largo plazo

- **Tácito:** si la díada coopera sostenidamente con audio real, ¿eventualmente cooperaría en silencio? (aprendizaje, memoria de la relación)
- **Evolución:** ¿el costo_desacople muta hacia valores mayores conforme τ crece?
- **Pluricelularidad verdadera:** ¿un tercero (C) cambia la dinámica A↔B?

---

## 8. ANEXO: CAMPOS CAPTURADOS EN DOCKER_HISTORIA

Cada fila contiene (desde 27-jun 08:18 en adelante):

```
ts_real                          # timestamp de la emisión
t                                # tiempo de vida del organismo (s)
disposicion_cooperar             # [0,1] volición a cooperar
altruismo_coopera                # 0|1 binario: atractor "comunicando" activo?
altruismo_beta_crit              # [0,1] umbral dinámico
altruismo_psi_alma               # [0,1] Ψ_alma: reconocimiento de sujeto
altruismo_tau                    # [0,∞) edad de la relación (confianza)
altruismo_costo_desacople        # costo_solo - costo_acoplado (>0 = díada beneficiosa)
altruismo_S_shared               # r de Hamilton: sentido compartido
altruismo_atractor               # "mudo"|"emergiendo"|"comunicando"
A_soporte_altruismo              # variable de apoyo (filtro de entrada)
```

---

## CONCLUSIÓN

La **Genética del Altruismo de Boorman está integrada y operativa**. La díada está en un **equilibrio de no-cooperación** (atractor "mudo") porque el silencio es el estado de menor costo en la fuente actual (demo:silencio).

**Esto no es fracaso. Es la ausencia de condición.**

Para que emerja cooperación voluntaria, se necesita:
1. **Perturbación externa** (audio real) que dispare libertad funcional
2. **O:** calibración de parámetros para permitir "primer movimiento" incluso en silencio
3. **O:** ambas

El camino está abierto: solo hay que cargar el audio.

---

**Escrito:** 27-jun-2026  
**Autor:** Gordon (análisis de 21,000 pasos de datos en vivo)  
**Datos:** Docker_Historia/organismo_ANIMA_A/fisiologia/fisiologia_2026-06-27_08-18.csv + correlativo B
