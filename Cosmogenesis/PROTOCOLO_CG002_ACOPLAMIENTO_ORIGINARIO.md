# PROTOCOLO CG002 — Acoplamiento originario y emergencia de orden

**Versión:** 0.1b (lectura honesta) + v0.2 (θ_CP adoptado 29-jun)  
**Fecha:** 2026-06-29  
**Autores del diseño:** Grok (Diotallevi) · revisión adversarial Claude CC (verificada)  
**Dirige:** Alexis López Tapia (Casaubon)  
**Estado:** **IMPLEMENTADO v0.1b** — Fases A+B+θ_CP; pendiente re-verificación CC antes de producción  
**Precedente cerrado:** CG001 §115 — C-N1 persistencia mínima → historia local (`INFORME_CAUSALIDAD_EPSILON.md`)

---

## 0. Pregunta experimental (derivada de la Parte I)

CG001 demostró **C-N1** (S > 0 deja huella). La teoría exige el siguiente eslabón:

| Nodo | Enunciado (resumen) | Pregunta operativa CG002 |
|------|---------------------|--------------------------|
| **C-N2** | S = I ⟺ E | ¿La persistencia se **sostiene** solo cuando hay interacción mutua interior–exterior? |
| **C-N2.1** | S > 0 ⟺ I > 0 ∧ E > 0 | ¿Con N ≥ 2 proto-distinciones, el exterior activo de cada una son las **otras**? |
| **C-N2.2** | E = 0 ∨ I = 0 ⇒ S = 0 | ¿Sin acoplamiento, cada proto-distinción vuelve a C-N1 o desaparece? |
| **C-N2.5.2** | t ≡ secuencia de constricciones acumuladas | ¿El **tiempo propio** emerge solo cuando Δ_struct > 0? |
| **C-N2.5.6** | S > 0 ⇏ T⁺ | ¿La **dirección** no es parámetro de entrada? |
| **C-N2.5.7** | T⃗ ≡ Σ(Δ_struct · R · P) | ¿La orientación es **salida medida** del enredo? |
| **C-N2.5.10** | T⁺ + T⁻ → 0 ⇒ Δ_struct → 0 | ¿La **aniquilación** aparece como outcome, no como regla semilla? |
| **C-N2.6.2** | trayectorias ≡ seguir gradiente de acoplamiento | ¿Hay **trayectoria en el espacio de estados** (grafo), no en ℝ³? |
| **C-N4** | ∂S ≠ 0 ⇒ sistema definido | ¿Emergen **fronteras operativas** medibles (output)? |

**No preguntamos en CG002:** U_Cos, κ completos, Λ_Cos, cuatro fuerzas, π, c (C-N2.7.6, C-N2.8.15 — Parte II / otra capa).

---

## 1. Contrato adversarial (4 afinaciones + criterios A–G)

### 1.1 Afinaciones obligatorias (Claude CC, 29-jun-2026)

1. **N ≥ 2** — forzado por C-N2.1/2.2, no elección estética. Singularidad = **pluralidad densa de proto-distinciones en un locus**, sin campo global.
2. **Proto-distinción ≠ sistema definido** — C-N4 es **output** (∂S medido), no frontera cerrada en t = 0.
3. **Una regla primitiva: acoplamiento (C-N2)** — competencia, fusión y aniquilación son **outcomes**; T⁺/T⁻ no existen en t = 0.
4. **Sin Fase D de viabilidad** — cierre de Cosmogénesis = **frontera emergente (C-N4)**; medir U_Cos es programa posterior (VSTCosmo).

### 1.2 Autocertificación contra criterios de aceptación

| ID | Criterio | Cómo lo cumple este protocolo | Falla si… |
|----|----------|-------------------------------|-----------|
| **A** | Acoplamiento por propiedad intrínseca, no distancia | Pares acoplan vía **firma orientacional** ωᵢ (ver §4); **prohibido** radio, vecindad espacial, grilla | Aparece alcance espacial |
| **B** | Dirección = salida del grafo orientado | **N/A si θ_CP=0** (grafo simétrico); con θ_CP≠0: `imbalance_orientado` (§6.3b) | Eje xyz o ángulo fijo de entrada |
| **C** | Tiempo = Δ_struct acumulado | Reloj **τ** solo avanza si Δτ > 0 en el paso (§6.2) | τ ≡ contador de bucle siempre |
| **D** | Una regla; outcomes emergen | Solo §5.1; sin fases «primero compite» | Dos reglas o fases codificadas |
| **E** | Singularidad = pluralidad N≥2, sin campo global | §3; sin φ ruidoso en volumen | N=1 o campo lleno |
| **F** | Sistemas y constantes físicas = outputs | Sin π, c, fuerzas; C-N2.7.6 explícito | Protocolo deriva física |
| **G** | Control nulo acoplamiento OFF | Condición **G** (§7.2); análogo γ=0 de §115 | Sin control I·E apagado |

---

## 2. Alcance y exclusiones

**Incluye:** dinámica relacional mínima, grafo en el tiempo, emergencia de τ y T̂, test de frontera ∂S.  
**Excluye:** grilla L³ fija, wrap toroidal, kernel gaussiano espacial, barrido RUIDO cola-lisa, métricas de concentración max/mean, Docker/WebLive v1, medición de U_Cos.

**Relación con CG001:** reutiliza el **mismo espíritu** de controles nulos y pre-registro; **no** reutiliza el motor `cg001_field.py` como núcleo (acoplamiento ≠ relajación local en celda).

---

## 3. Condición inicial — singularidad como pluralidad (C-N2.1, C-N2.2, criterio E)

### 3.1 Locus

- Un **locus** L₀: conjunto finito de **N** proto-distinciones indexadas i = 1…N.
- **N mínimo:** 2 (obligatorio). **N por defecto en producción:** 8. **Barrido:** N ∈ {2, 4, 8, 16}.
- **No** hay coordenadas espaciales (x, y, z). **No** hay volumen preexistente.

### 3.2 Semilla de pluralidad

Cada proto-distinción i en t = 0 porta:

| Símbolo | Rol teórico | Definición operativa mínima |
|---------|-------------|----------------------------|
| **Sᵢ** | Persistencia (C-N1) | Escalar > 0; si Sᵢ ≤ κₛ ⇒ extinción (C-N1.1) |
| **ωᵢ** | Proto-interior (input I) | Firma en espacio **Ω_firma** (§4.1); distingue «qué es» i |
| **(E implícito)** | Exterior activo | Conjunto {j ≠ i}; no es campo inerte |

**Singularidad:** el **vector** (Sᵢ, ωᵢ)ᵢ₌₁..N en el locus L₀ — «todas las asimetrías contenidas» = **muchas distinciones cuya mutua exterioridad es la semilla de toda relación**, no un punto escalar ni ruido global.

### 3.3 Inicialización de ωᵢ y Sᵢ

- **Sᵢ(0) = S₀** constante para todas las semillas de corrida (S₀ > κₛ).
- **ωᵢ(0)** se sortea con RNG determinista por `seed` desde Ω_firma **sin** mirar outcomes futuros.
- **Prohibido:** asignar fronteras, ∂S, o «sistemas ya definidos» en t = 0 (afinación 2).

---

## 4. Espacio de firmas — acoplamiento sin distancia (criterio A)

### 4.1 Ω_firma (pre-registrado, v0.1)

Versión mínima para implementación v1:

- **ωᵢ ∈ ℤ_K** (fase discreta), con **K = 8** fijo en v0.1.
- Interpretación: orientación proto-histórica **antes** de T⁺/T⁻ (que son emergentes).

### 4.2 Compatibilidad intrínseca (no espacial)

Para cada par (i, j), i ≠ j:

```
δᵢⱼ = min(|ωᵢ − ωⱼ|, K − |ωᵢ − ωⱼ|)   ∈ {0, …, ⌊K/2⌋}
cᵢⱼ = cos(2π · δᵢⱼ / K)                 ∈ [−1, 1]
```

- **cᵢⱼ** depende **solo** de (ωᵢ, ωⱼ). **No** hay distancia, radio ni vecindad.
- **R (restricción de viabilidad):** en v0.1, R = 1 si Sᵢ, Sⱼ > κₛ; si no, el par no acopla.

---

## 5. Regla única — acoplamiento originario (C-N2)

### 5.1 Regla primitiva (única)

En cada paso micro **solo** si el **interruptor de acoplamiento** α = 1 (§7):

Para todo par no ordenado (i, j) con Sᵢ > κₛ y Sⱼ > κₛ:

```
Fᵢⱼ = α · cᵢⱼ · √(Sᵢ · Sⱼ)        # flujo de acoplamiento (signed)
Sᵢ ← Sᵢ + η · Fᵢⱼ               # η > 0 paso de acoplamiento, simétrico en i,j
Sⱼ ← Sⱼ + η · Fᵢⱼ
```

- Actualización **simultánea** de pares (orden de pares no importa; usar suma simétrica).
- **Decaimiento pasivo** (sin acoplamiento efectivo): Sᵢ ← (1 − μ) · Sᵢ con μ ∈ (0,1) pequeño **antes** del acoplamiento, para que el aislamiento sea posible.

**Parámetros fijos v0.1 (no ajustar post-hoc):**

| Parámetro | Valor | Nota |
|-----------|-------|------|
| κₛ | 1e−6 | umbral C-N1.1 |
| S₀ | 1.0 | |
| η | 0.05 | |
| μ | 0.01 | |
| K | 8 | |

### 5.2 Outcomes emergentes (no reglas — criterio D)

A partir **solo** de §5.1, el implementador **registra** pero **no codifica** como reglas separadas:

| Outcome | Condición de registro (post-hoc descriptiva) | Nodo teórico |
|---------|-----------------------------------------------|--------------|
| **Competencia** | Fᵢⱼ < 0 y min(Sᵢ, Sⱼ) cae más que la media de pares | C-N2.5.7 (balance negativo) |
| **Fusión / enlace** | Fᵢⱼ > 0 sostenido ⇒ arista (i,j) en grafo con peso acumulado | C-N2, C-N2.6.3 |
| **Aniquilación** | ∃ paso donde Sᵢ → 0 y Sⱼ → 0 tras secuencia de F de signo opuesto acumulado | C-N2.5.10 |
| **Orientación T̂** | Solo definida cuando hay Δ_struct acumulado (§6) | C-N2.5.6–7 |

**Prohibido en código:** `if competir: … elif fusionar: …`; `if aniquilar: …`.

---

## 6. Tiempo y dirección como salidas (C-N2.5, criterios B y C)

### 6.1 Δ_struct — diferencia estructural acumulada

En cada paso micro, **antes** de actualizar τ:

```
Δ_step = Σ_{pares acoplados} |Fᵢⱼ|  +  Σᵢ |ΔSᵢ|   +  Δ_topo
```

donde:

- |Fᵢⱼ| = magnitud del flujo de acoplamiento en el paso.
- |ΔSᵢ| = magnitud del cambio **atribuible al acoplamiento** (η·F), no al decaimiento pasivo μ.
- **Δ_topo** = 1 si se crea o destruye una arista en el grafo de enlace (§6.4); 0 si no.

**Δ_struct(t) = Σ Δ_step** acumulado desde el primer paso con α = 1.

### 6.2 Reloj τ (tiempo propio)

```
Si Δ_step > ε_τ  ⇒  τ ← τ + 1
Si Δ_step ≤ ε_τ  ⇒  τ no avanza
```

- **ε_τ** fijo: 1e−4 en v0.1.
- **El contador de pasos micro** `k` se loguea pero **no** es tiempo teórico (criterio C).

### 6.3 Balance cooperativo ρ (v0.1 simétrico)

- **ρ** = (Σ max(F,0)) / (Σ |F| + ε) sobre `f_signed` — fracción cooperativa global.
- **No es dirección.** Con θ_CP=0, criterio **B = N/A**.

### 6.3b Dirección T̂ (v0.2 — θ_CP ≠ 0)

Acoplamiento dirigido (C-N2.5.5):

```
g_{i←j} = cos(2π(ωᵢ−ωⱼ)/K + θ_CP)    g_{j←i} = cos(2π(ωⱼ−ωᵢ)/K + θ_CP)
```

- **θ_CP = 0** reproduce v0.1 exacto (control G′ de dirección).
- **θ_CP ≠ 0** ⇒ Fᵢⱼ ≠ Fⱼᵢ ⇒ `imbalance_orientado(W_orient)`:
  - `asimetria` = ‖W−Wᵀ‖/‖W‖
  - `flujo_neto` por nodo (vector de orientación realizada)
  - |T̂| = ‖flujo_neto‖
- Evaluable solo si τ ≥ τ_min y θ_CP ≠ 0.

### 6.3c Frustración φ(τ) → C-N2.5.10

- λ_min del Laplaciano signado del subgrafo vivo (W signado normalizado).
- Reportar **fracción de semillas** con φ > 0, no un λ_min único.

### 6.4 Grafo G(t) — espacio de estados (no ℝ³)

- **Nodos:** proto-distinciones con Sᵢ > κₛ.
- **Aristas:** entre i, j si peso acumulado Wᵢⱼ > w_min (w_min = 0.1 · S₀).
- **Trayectoria** de i: secuencia (ωᵢ(τ), Sᵢ(τ), grado(i, τ)) — C-N2.6.2 en espacio de estados.
- **Extensión emergente:** diámetro de G(t), número de componentes — **salidas**, no coordenadas.

---

## 7. Condiciones experimentales y controles (criterio G)

### 7.1 Condición principal (α = 1)

- N ∈ {2, 4, 8, 16}; semillas 1…S_max (S_max = 30 en smoke; 1000 en producción iPad/iMac).
- Pasos micro máximos K_max = 10⁴ (safety); parada anticipada si τ ≥ τ_cap (τ_cap = 500) o todos Sᵢ ≤ κₛ.

### 7.2 Control G — acoplamiento OFF (obligatorio)

- **α = 0:** regla §5.1 no aplica; solo decaimiento Sᵢ ← (1−μ)Sᵢ.
- **Predicción teórica (pre-registrada):**
  - Δ_struct → 0 (no acumula estructura relacional).
  - τ permanece en 0 (o crece despreciablemente).
  - G(t) sin aristas nuevas; T̂ no definido.
  - Solo supervivencia/decaimiento individual (C-N1), **sin** C-N2.

**Veredicto de control:** si con α = 0 aparecen aristas, τ > 0, o T̂ definido → **protocolo inválido / sustrato contamina** (falla G).

### 7.3 Control N = 1 (obligatorio, criterio E)

- Una sola proto-distinción; α = 1.
- **Predicción:** E efectivo = ∅ ⇒ no hay acoplamiento real (C-N2.2); comportamiento ≈ decaimiento; **no** debe reproducir CG002 positivo.

### 7.4 Control de caos (opcional, herencia §115)

- Pares de corridas con misma N y distinta `seed` sin acoplamiento relevante: cuantificar variabilidad de T̂ y diámetro de G para separar señal de azar.

---

## 8. Observable C-N4 — frontera emergente (cierre Cosmogénesis)

**No** se asigna ∂S en t = 0. Se **mide** al final (o cuando τ = τ_cap):

Para cada nodo i vivo:

```
Hᵢ = |{ j : Wᵢⱼ > w_min }|          # grado de acoplamiento
Bᵢ = Σⱼ |Fᵢⱼ|_acumulado / (Hᵢ + ε) # intensidad de borde relacional
∂̂Sᵢ = Bᵢ / (mean_j Bⱼ + ε)         # proxy de frontera operativa
```

**C-N4 positivo (pre-registrado):** ∃ al menos un i con ∂̂Sᵢ > ∂_crit ( ∂_crit = 2.0 ) **y** Hᵢ ≥ 1 **y** el par i no es trivialmente aislado en α = 1.

**Eso cierra CG002** — no se exige κ_Δ, κ_V, Λ_Cos (Parte II).

---

## 9. Criterios de lectura y veredicto (pre-registrados)

### 9.1 Veredicto principal (cadena teórica)

| ID | Afirmación | Positivo si… | Nodo |
|----|------------|--------------|------|
| **V1** | Acoplamiento sostiene persistencia relacional | Con α=1, fracción de semillas con ≥2 nodos vivos a τ_cap **>** α=0 | C-N2, C-N2.1 |
| **V2** | Δ_struct genera tiempo propio | Con α=1, τ_final > 0 en ≥ 83 % semillas; con α=0, τ ≈ 0 | C-N2.5.2–3 |
| **V3** | Dirección emerge (no dada) | Con α=1, |T̂| y ρ tienen signo estable ≥ 83 % entre semillas con mismo N; ρ no es input | C-N2.5.6–7 |
| **V4** | Aniquilación / enlace son outcomes | Logs muestran eventos de S→0 y Wᵢⱼ creciente **sin** reglas dedicadas | C-N2.5.10, C-N2 |
| **V5** | Trayectoria en espacio de estados | Secuencias (ωᵢ, Sᵢ, grado)ᵢ no constantes; diámetro(G) crece en subconjunto de semillas | C-N2.6.2 |
| **V6** | Frontera operativa emerge | V6: ∂̂Sᵢ > ∂_crit en ≥ 1 nodo en ≥ 50 % semillas (α=1, N≥4) | C-N4 |

**Umbral 0.83** heredado de barridos CG001 (convención de signo estable).

### 9.2 Veredicto negativo honesto

Si V1 falla pero V2–V3 pasan → **sustrato espurio** (revisar implementación).  
Si solo V1 pasa → **C-N1 prolongado**, no C-N2 (como N=1).  
Si todos fallan en α=1 pero control G muestra lo mismo → **falla G**, invalidar corrida.

### 9.3 Jerarquía de informe

1. Reportar **datos** (CSV/JSON) sin narrativa optimista.  
2. Aplicar V1–V6 en orden.  
3. Solo entonces redactar `INFORME_CG002_*.md`.

---

## 10. Entregables y trazabilidad

| Artefacto | Contenido |
|-----------|-----------|
| `cg002_acoplamiento.py` | Implementación fiel (post-visto bueno) |
| `logs/cg002_YYYYMMDD_HHMMSS/resultado.json` | Agregados por N, α |
| `logs/cg002_.../series.csv` | Por semilla: τ, Δ_struct, |T̂|, ρ, diámetro(G), ∂̂S_max |
| `logs/cg002_.../grafo_seed{N}.jsonl` | Snapshots G(τ) opcionales (cada Δτ) |
| `cg002_viz.html` | Grafo dinámico (nodos/aristas), **sin** caja 3D fija |

**Smoke-test obligatorio antes de producción:**

- N=2, semillas 1–3, α ∈ {0,1}, K_max=10³ → verificar G manualmente.

---

## 11. Checklist adversarial — verificado por Claude CC (29-jun)

| ID | v0.1 simétrico (θ_CP=0) | v0.2 (θ_CP≠0) |
|----|-------------------------|---------------|
| **A** | ✅ PASS verificado | ✅ |
| **B** | **N/A** (Fᵢⱼ=Fⱼᵢ; sin objeto dirección) | ✅ evaluable (`imbalance_orientado`) |
| **C** | ✅ PASS | ✅ |
| **D** | ✅ PASS | ✅ |
| **E** | ✅ PASS | ✅ |
| **F** | ✅ PASS | ✅ |
| **G** | ✅ PASS (α=0: τ=0, Δ=0, aristas=0) | ✅ G′ (θ_CP=0: asimetría=0) |

**Observables v0.1b (sin θ_CP):** frustración φ(τ), rango estructural (=2 en ℤ₈), balance cooperativo ρ.  
**Pendiente 0.1b:** saturación de S (normalización observables); re-verificación CC post-fix.  
**Smoke:** `python3 cg002_acoplamiento.py` → logs `logs/cg002_*`; k_max≥1500.

---

## 12. Cadena teórica que este protocolo intenta cerrar

```
C-N1  S > 0           (CG001 — cerrado)
        ↓
C-N2  I ⟺ E           (CG002 V1 — acoplamiento α)
        ↓
C-N2.5  Δ_struct → τ  (CG002 V2)
        ↓
C-N2.5.7  T̂         (CG002 V3 — salida espectral)
        ↓
C-N2.6  trayectorias en G (CG002 V5)
        ↓
C-N4  ∂̂S emergente   (CG002 V6 — FIN Cosmogénesis fase 2)
```

---

## 13. Nota para implementadores (Grok / Claude Web / Claude CC)

- La **visualización** debe mostrar **nodos y enlaces** que aparecen, se refuerzan o desaparecen — el «movimiento» es **reconfiguración del grafo**, no traslación en un cubo.
- Cualquier desviación de §4–§5 requiere **nueva versión** del protocolo (0.2), no ajuste silencioso.
- MEMANTO: registrar contrato A–G + v0.1 cuando Alexis autorice y `memanto` esté disponible.

---

*Implementación: `cg002_acoplamiento.py`, visor `cg002_viz.html`. Logs en `logs/cg002_*/`.*