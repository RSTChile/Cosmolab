# PROPUESTA CG002 v0.2 — Dirección, frustración y geometría como observables reales

**De:** Claude Web · **Para:** Grok (Diotallevi), equipo Cosmogénesis · **Dirige:** Alexis López Tapia (Casaubon)
**Fecha:** 2026-06-29 · **Estado:** propuesta pre-registrada para integración (no parche silencioso — protocolo §13 exige v0.2)
**Verificación:** las cuatro piezas corridas en `cg002_observables_v02.py` (autotest PASS)
**Coordinación:** capa de lectura la verifica CC contra código; firma/θ_CP las decide Alexis; esto define los observables y el mecanismo.

---

## 0. En una línea

v0.1 no puede mostrar dirección **por construcción, antes de cualquier bug**: el grafo es simétrico (Fᵢⱼ = Fⱼᵢ) y la firma ℤ_K es de rango 2. Esta propuesta separa lo que v0.1 **sí** puede medir (frustración, geometría 2D) de lo que necesita un mecanismo nuevo (dirección), e introduce ese mecanismo como **θ_CP** — la violación CP cosmosemiótica (C-N2.5.5), con θ_CP = 0 como control nulo que reproduce v0.1 exacto.

---

## 1. Diagnóstico verificado (raíz estructural, no de implementación)

| Hecho verificado | Consecuencia | Nodo afectado |
|---|---|---|
| cᵢⱼ = cos(2π·δ/K), δ simétrica, update simétrico ⇒ **Fᵢⱼ = Fⱼᵢ** | grafo **no dirigido** ⇒ no existe T⁺/T⁻ | C-N2.5.7 sin sustrato |
| C(ω) = ⟨uᵢ,uⱼ⟩ con u en círculo ⇒ **rango 2 para todo N, K** | embedding emergente topeado a **2D** | C-N2.6 capada |
| K=8: los 16 triángulos reales **balanceados**, λ_min=0 | frustración triádica **inexistente** a N=3 en K=8 | C-N2.5.10 escondida |
| Frustración **sí aparece** a N≥4 en K=8 (peor λ_min=0.414); y a N=3 con K∈{6,12,16} | la aniquilación tiene sustrato, pero no en la unidad mínima de K=8 | C-N2.5.10 |

**Lectura:** el `lap.sum()==0` que cazaron CC y Grok es real y hay que arreglarlo, pero **arreglarlo no resucita la dirección** — abajo no hay flecha que medir mientras el grafo sea simétrico. La pregunta "¿de dónde sale la flecha?" es, en el marco de Alexis, "¿cuál es la violación CP de CG002?". Hoy: ninguna. Eso es §3 de esta propuesta.

---

## 2. Cuatro integraciones, ordenadas

Todas implementadas y probadas en `cg002_observables_v02.py`. Funciones puras (leen estado, no lo escriben). Las firmas exactas están en el módulo; aquí va el qué y la predicción pre-registrada.

### 2.1 — Capa de lectura: renombrar honestamente *(territorio de CC)*

No tocar acá; CC ya especificó el fix (f_signed, sacar ρ del return temprano). Solo una corrección de **nombre**, no de código: lo que el código llama `T̂` sobre un grafo simétrico **no es dirección**, es un escalar de cooperación-vs-competencia (ρ). Mientras θ_CP = 0, reportar ρ como *balance cooperativo*, nunca como *orientación*. Marcar B en el protocolo como **N/A en v0.1** (no PASS, no FAIL: no hay objeto), y mover B a v0.2.

### 2.2 — Observable de FRUSTRACIÓN φ(τ) → C-N2.5.10

`laplaciano_signado_lambda_min(W_signed)` → λ_min ≥ 0. **W debe ser signado** (si se usa |F|, ciego).
- λ_min = 0 ⟺ grafo balanceado · λ_min > 0 ⟺ frustración.

**Predicción pre-registrada (C-N2.5.10):** seguir φ(τ) = λ_min del subgrafo vivo en el tiempo.
- ✅ confirma si **φ(τ) → 0 monótono** (el clúster sobreviviente se balancea: la aniquilación ES frustración resolviéndose).
- ❌ refuta si φ(τ) queda atrapado en > 0 (frustración persistente sin aniquilación).
- ⊘ no evaluable a N≤3 con K=8 (sin sustrato). **Exponer la unidad mínima con K=12, N=3** — una sola triada que se resuelve, más diagnóstico que N=16 ciego.

### 2.3 — Observable de EMBEDDING → C-N2.6 (curvatura sin coordenadas)

`embedding_espectral(W_abs, n_dim)` → posiciones emergentes (autovectores bajos del Laplaciano no signado). `rango_estructural(C)` → techo duro. `dimension_efectiva(C)` → dimensión usada.

**Predicción pre-registrada:** en v0.1, `rango_estructural` = **2** para todo N, K (verificado). Reportarlo no como bug sino como **medida del cuello de botella**: la "extensión emergente" de §6.4 está topeada a un plano. Sube solo con §2.5.

### 2.4 — θ_CP: acoplamiento dirigido = violación CP cosmosemiótica → C-N2.5.5 / 2.5.7 / 2.5.10

`paso_acoplamiento_dirigido(S, ω, K, η, θ_cp)` — drop-in del paso §5.1 con dirección. Un solo parámetro nuevo:

```
g_{i←j} = cos(2π(ωᵢ−ωⱼ)/K + θ_CP)      g_{j←i} = cos(2π(ωⱼ−ωᵢ)/K + θ_CP)
```

- **θ_CP = 0 ⇒ Fᵢⱼ = Fⱼᵢ ⇒ v0.1 EXACTO.** Es el control nulo G' de la dirección.
- **θ_CP ≠ 0 ⇒ grafo orientado genuino.** `imbalance_orientado(W)` devuelve `asimetria = ‖W−Wᵀ‖/‖W‖` (verificado: 0.000 en θ=0, 1.039 en θ=0.5) y un `flujo_neto` por nodo que **sí** es vector de orientación, no escalar.

Esto ancla C-N2.5.5 a un mecanismo: θ_CP **es** la asimetría primordial. Sin él, materia/antimateria (T⁺/T⁻) se cancelan total (C-N2.5.10 trivial, universo vacío); con θ_CP pequeño, sobrevive un exceso orientado. Recién entonces T̂ tiene sustrato y B es evaluable.

**Decisión de Alexis (no mía):** si la cadena C-N2.5.5→2.5.10 debe leerse como *requiriendo* θ_CP ≠ 0 (la flecha no es gratis, cuesta una asimetría), o si la flecha debe emerger sin inyectarla. Las dos lecturas son coherentes con los nodos; la primera es la que el paralelo CP sugiere.

### 2.5 — Firma multicomponente (esfera S^{d-1}) → sube el techo *(opcional, decisión de Alexis)*

`matriz_compat_esfera(U)` con U:(N,d) → rango hasta d (verificado: d=3→3, d=5→5). **Ortogonal a θ_CP**: d controla *cuánta estructura relacional* (frustración más rica, embedding hasta d dimensiones); θ_CP controla *si hay flecha*. Variar K **no** sube el rango (sigue 2). Si C-N2.6 quiere geometría >2D emergente, el lever es d, no K.

---

## 3. Qué NO se afirma (frenos)

- Todo lo de §1 y los techos de §2 es **combinatoria estática de C(ω)** — computable casi sin dinámica. Lo que **no sé** y necesita correr: si la dinámica lleva el clúster sobreviviente a balance (φ→0) o lo deja frustrado. Ese es el test de 2.2.
- El paralelo con Wheeler-DeWitt / violación CP / Page-Wootters es **estructural, no derivación** (C-N2.7.6). θ_CP **no es** la fase CP del Modelo Estándar; es su análogo cosmosemiótico. No introducir constantes físicas en el código por esta vía.
- θ_CP y la firma-esfera son **v0.2**: cambios de mecanismo, no de lectura. Requieren bump de protocolo (§13), no ajuste silencioso.

---

## 4. Orden de integración sugerido y ruteo

1. **CC** — arreglar capa de lectura (su spec) + renombrar B como N/A en v0.1 (§2.1).
2. **Grok** — integrar 2.2 (frustración) y 2.3 (embedding/rango) como observables sobre las corridas **simétricas actuales**. No requieren cambiar la física. Correr el test φ(τ) a N≥4 (K=8) y a N=3 (K=12).
3. **Alexis decide** — si v0.2 incorpora θ_CP (§2.4) y/o firma-esfera (§2.5). Ambas listas como funciones completas.
4. **Grok** — si Alexis aprueba θ_CP: integrar `paso_acoplamiento_dirigido`, correr barrido θ_CP ∈ {0, 0.1, 0.3, 0.6} con θ_CP=0 como control. Recién ahí B (dirección) pasa a evaluable.
5. **CC** — re-verificar B contra el test discriminante con θ_CP ≠ 0.

Módulo: `cg002_observables_v02.py` (completo, autotest PASS). No toco interfaz (Alexis) ni MEMANTO (CC).
