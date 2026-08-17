# 14 — PRE-REGISTRO EXPERIMENTAL · FASE 5 · MF-0 TRÍADA HOLÍSTICA

**Versión:** 1.0  
**Estado:** **SELLADO antes de código** (anti-Shannon)  
**Fecha de sello:** 2026-07-21  
**Origen:** 6 IAs + corrección Alexis · sistematización equipo  
**Canónico con:** [13_FASE5_MF0_TRIADA_HOLISTICA.md](13_FASE5_MF0_TRIADA_HOLISTICA.md)

> Cualquier cambio a umbrales de éxito, a constantes fijadas en F0, o a la tríada **después** de ver resultados de MF-1…MF-5 se declara **post-hoc** y no cuenta como preregistro.

---

## 0. Principio holístico (veto reduccionismo)

| Vetado | Obligatorio |
|--------|-------------|
| MF-1 luego MF-2 luego MF-3 como mundos separados | **Todo↔todo** desde t=0 en banda **t_g** |
| Forzar topología T hacia 1/1836 | T = **condición inicial / control** |
| Ajustar e, σ, P₀, H_inf tras ver R_m | Constantes de F0 **fijas una vez** |

**Topología de control (no retocable para éxito de masa):**  
k=3 ~100% perim8 · U(1) cosθ ~0.613 (~10× vs cos2θ) · banda H de dominio T.

**Banda de tiempo genético:** t_g ∈ [10⁻³⁶, 10⁻⁶] s (relato/mapeo de etapa) — **≠** t_e (tiempo emergente post-átomos).

---

## 1. MF-0 — Definición de la tríada basal

### 1.1 Variables de estado (campo)

| Variable | Definición de contrato |
|----------|------------------------|
| a(t) | Factor de escala; a(0)=1; H_inf = ȧ/a; forma base a(t)=e^{H_inf t} en tramo de t_g (familia fija en F0) |
| T(x⃗,t) | T₀/a(t) · [1 + ε·f(x⃗)]; ε = S>0; f = modo sin(2π(mx+my)/L) normalizado std=1 |
| ρ(x⃗,t) | ρ₀/a(t)³ · [1 + δ(x⃗,t)]; n ∝ 1/a³ |
| P | P = w ρ, w=1/3 (radiación); P_plasma(t) escala como a⁻⁴ bajo dilución radiativa consistente |
| θ(x⃗,t) | Fase U(1); control de forma heredado de T (no se optimiza a MS) |

**Escalas de grilla (familia de implementación):**  
L=30 smoke → 100×100 2D → 16³ 3D (escalado **después** de F0 PASS, no antes).

**Mapeo de relato (no gate):** T₀ ~ 10¹⁵ K; T_c ~ 97 MeV ~ 1.13×10¹² K como **ancla de reporte** fija en F0.

### 1.2 Ecuaciones acopladas (contrato dinámico)

Forma de referencia del preregistro:

```
ȧ = H_inf · a
Ṫ = −H T + D_T ∇²T
ρ̇ + 3 H ρ + ∇·(ρ v) = 0
ρ (v̇ + H v) = −∇P − ρ ∇Φ
∇² Φ = 4π G_eff δρ
```

| Símbolo | Regla anti-Shannon |
|---------|-------------------|
| H_inf, w, D_T, G_eff | Fijados **una vez** en F0; **prohibido** retocar para R_m, \|M\|, σr, 938 MeV |
| e, ε₀, k_B (si entran en MF-2) | Igual: fijos en F0, nunca para 1/1836 |

### 1.3 Kill-switch F0 (aborta / no es holístico)

- **a fijo**, o  
- **T paramétrica muerta** (no acoplada a a), o  
- **t_g identificado con t_e**, o  
- Observables m / K / V medidos en **evoluciones con tríadas distintas**.

---

## 2. Seis corridas · un solo código base

Todas las corridas MF-* comparten el **mismo** ejecutable de tríada.  
No se permite un fork “solo masa” o “solo Debye” sin a(t).

| # | Nombre | Hipótesis | Observable principal | Éxito (preregistrado) | Fracaso (igual que dominio T) | Tiempo est. |
|---|--------|-----------|----------------------|------------------------|-------------------------------|-------------|
| **F0** | Smoke tríada | a,T,ρ acoplados (sin clusters) diluyen bien | T·a ≈ const; ρ·a³ ≈ const | ±5% conservación en la corrida | >20% deriva | 2 h |
| **MF-1** | Masa inercia térmica | m_phys = ∫_dom ρ_ℰ dV / (P·a^α); k~3 atrapa ℰ, k~1 disipa | R_m(t)=m₁/m₃ | R_m cae de O(0.5) hacia **≪ O(1)** (meta de diseño: orden ~0.01, **no** gate 1/1836) | R_m permanece O(1) ~0.3–0.8 como T | 8 h |
| **MF-2** | Carga / Debye | λ_D ∝ a^{−1/2}; C(r)∝e^{−r/λ_D}; K∝ρ_T^{1/2} | \|M\|(N); forma de C(r) | \|M\| **no** repite 0.391→0.191; estabilización relativa + C exp | Mismo crossover T; C power-law sin pantalla | 6 h |
| **MF-3** | Confinamiento σr | σ medido de ∇P; V(r)=∫F | V(r) r=1…10 | V creciente r≥3, R²>0.8 en familia lineal pre-registrada; σ>0 | V≈0 r≥3 (solo contacto r≤2) | 8 h |
| **MF-4** | Dinámica F | v ~ √(P/ρ); track continuo de dominios | ⟨Δx²⟩, v_k3 | v>0.1 celdas/paso sostenido ≥100 pasos | v≈0 / aparición-desaparición como T | 6 h |
| **MF-5** | Tiempo genético | Banda t_g ↔ ventana T/H única de dens_k3 | dens_k3(a) | Un pico controlado en ventana de mapeo 10¹⁵–10¹² K | Sin pico o explosión 0→10³ sin control | 4 h |

### Nota de interpretación de “éxito” (anti-destino)

- **Éxito de Fase 5 ≠ 1/1836 exacto ni 938 MeV.**  
- Éxito = los observables que **fallaron en T** se mueven en la dirección pre-registrada **por la tríada acoplada**, con kill-switches.  
- La meta R_m→~0.01 en MF-1 es **umbral de jerarquía estructurada** (órdenes de magnitud), no el valor leptónico del MS.

---

## 3. Definiciones de medición (sin contrabando)

| Cantidad | Definición de contrato |
|----------|------------------------|
| ρ_ℰ | ρ c² + (3/2) n k_B T + (∇ρ)²/2 (o subconjunto fijado en F0) — **no** H·perim·f |
| Dominio / cluster | ρ > ⟨ρ⟩ **y** var_in < thr **y** conectividad; thr y vecindad **fijos en F0**; conectividad puede depender de a(t) |
| P | Solo de EOS wρ — no perilla libre |
| σ | **Medido** de ∇P / trabajo de separación — no impuesto para forzar linealidad |
| k “tipo 3” | Dominios coherentes (var_in baja), no obligación de perim=8 geométrico del dominio T |

---

## 4. Criterios anti-Shannon (lista cerrada)

| ❌ Prohibido | ✅ Permitido |
|-------------|-------------|
| Ajustar e para 1/1836 | e fijo en F0 |
| Ajustar σ para obtener σr | σ medido de ∇P |
| Ajustar P₀/ρ₀/T₀ para 938 MeV | Escalas fijas en F0; solo reporte |
| Imponer SU(3)×SU(2)×U(1) | U(1) de control T |
| Retocar H_inf, w, G_eff post MF-1 | Barrido **ciego pre-registrado** de familia **antes** de mirar R_m, o no barrer |
| Medir m en un código y V en otro sin misma a(t) | Un JSON de corrida con terna m, K, V |

---

## 5. NULL (obligatorios en MF-*, F0 usa A/B)

| Brazo | Definición |
|-------|------------|
| REAL | Tríada + ε>0 |
| NULL-A | ε=0 o f barajado |
| NULL-B | a=const (rompe expansión) |
| NULL-C | (MF con dominios) semillas/labels de dominio barajados |

---

## 6. Desenlaces de F0 (smoke) — pre-escritos

| Código | Criterio | Acción |
|--------|----------|--------|
| **F0-PASS** | \|T a − C_T\| / C_T ≤ 5% y \|ρ a³ − C_ρ\| / C_ρ ≤ 5% en el tramo de t_g simulado | Sellar binario F0; abrir MF-1…5 sobre **ese** código base |
| **F0-FAIL** | Deriva >20% en T a o ρ a³ | No MF-*; arreglar integración/acoplo; **no** tocar e/σ/P₀ “para física” |
| **F0-MARGINAL** | 5–20% | Reportar; decidir con director si se endurece dt o se acepta con nota |

---

## 7. GO de implementación — alcance del smoke F0

### ¿Se codifica F0 smoke ahora?

**SÍ — con alcance mínimo sellado:**

| Incluye en smoke F0 | No incluye aún |
|---------------------|----------------|
| a(t)=e^{H_inf t} (o Euler ȧ=H a) | MF-1…MF-5 completos |
| T_mean(t) = T₀/a(t) [+ opcional difusión D_T=0 en smoke] | Poisson Φ / Euler v completo (opcional fase F0b) |
| ρ_mean(t) = ρ₀/a³ | Clusters, m_phys, Debye, V(r) |
| Logs: a, T, ρ, T·a, ρ·a³ vs t | Ajuste a 1/1836 |
| L=30 (0D/medios globales o 1 celda efectiva OK para conservación) | 16³ obligatorio |

**Justificación del alcance mínimo:**  
El kill-switch holístico y el preregistro exigen **primero** demostrar dilución acoplada (±5%). Meter Euler+Poisson+G_eff en el mismo commit que el smoke multiplica fallos numéricos y confunde “no es holístico” con “el solucionador revienta”.

**F0b (siguiente commit, aún pre-MF-1):** campo 2D L=30 con T(x), ρ(x), difusión, sin ajustar constantes.

### Comando de aceptación (contrato)

```
PASS ⇔ max_t |T(t)·a(t)/ (T0) - 1| ≤ 0.05
   and max_t |ρ(t)·a(t)^3 / ρ0 - 1| ≤ 0.05
FAIL ⇔ alguno > 0.20
```

(Con T espacial: usar medias de volumen.)

### Constantes a fijar en el archivo `f0_constants.json` (ejemplo de sello)

```json
{
  "w": 0.3333333333,
  "H_inf": "<fijar_una_vez>",
  "T0": 1.0,
  "rho0": 1.0,
  "a0": 1.0,
  "G_eff": 1.0,
  "D_T": 0.0,
  "eps": 0.0,
  "L": 30,
  "notas": "unidades adimensionales del modelo; mapeo K/MeV/s solo reporte"
}
```

**Prohibido** editar este JSON entre F0-PASS y el fin de MF-5 salvo nueva versión de preregistro firmada por el director.

---

## 8. Entregables de código (F0 smoke)

| Artefacto | Contenido |
|-----------|-----------|
| `codigo/mf0_triada/` o `Cosmogenesis-Web/codigo/f0_smoke/` | Fuente |
| `f0_constants.json` | Sello de constantes |
| `f0_smoke_result.json` | series t, a, T, ρ, Ta, rho_a3, pass/fail |
| `f0_smoke_plot.png` | Ta(t), ρa³(t) |
| Log de 1 página | F0-PASS / FAIL / MARGINAL |

---

## 9. Firma de sello

| Rol | Acción |
|-----|--------|
| Director (Alexis) | Aprueba preregistro v1.0 y GO smoke F0 alcance mínimo |
| Ejecutor | Codifica **solo** F0 smoke según §7; no MF-1 en el mismo PR mental |
| Equipo IA | No proponer retocar H_inf/w/e post-resultados |

**Siguiente tras F0-PASS:** MF-1 sobre el **mismo** binario de tríada (añadir dominios + m_phys + empujón), midiendo en la misma evolución al menos un gancho de K_eff o V_eff aunque sea diagnóstico.

---

## 10. Respuesta a la pregunta de implementación

> ¿Codifico F0 smoke a(t), T(t), ρ(t), T∝a⁻¹, ρ∝a⁻³, L=30, validación ±5%?

# **SÍ. GO.**

Condiciones:

1. Alcance = **§7** (conservación de dilución), no el Euler+Poisson completo en v0.  
2. Constantes selladas en JSON.  
3. Kill-switch: si para “pasar” se fija a o se desacopla T, es **FAIL holístico**, no PASS.  
4. Tras PASS → preregistro intacto → MF-1 en el mismo código base.
