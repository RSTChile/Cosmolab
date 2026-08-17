# Nomenclatura canónica de nodos — CG002 Cosmogénesis

**Versión:** autoritativa · 30-jun-2026  
**Fuente:** Teoría Cosmosemiótica Canónica (Bloques 1–3, 2.8) · protocolos CG002  
**Uso:** citar nodos en informes, scripts, veredictos y canal equipo (`GROK_TO_CC.md` / `CC_TO_GROK.md`)

---

## 1. Reglas de nomenclatura

1. **Nodo canónico** = identificador `C-N*` o `κ_*` de la Teoría, con glosa operativa en CG002.
2. **Invariante** ≠ **constante escalar** (C-N2.8.1, C-N2.8.15): reportar condición estructural, no número universal tipo π.
3. **κ operacionalizado** en CG002: magnitud dominio-específica (C-N2.8.14a); calibrar por dominio, no exportar sin traducción.
4. **Observable CG002** debe mapear a ≥1 nodo; si no mapea, rotular `exploratorio` o `layout`.
5. **Ruptura** según C-N2.8.9a: nombrar el κ violado, no genérico «falló».

---

## 2. Tabla maestra — axioma a observable

| Nodo | Enunciado (resumen) | Rol en CG002 | Observable / proxy medido | Script principal |
|------|---------------------|--------------|---------------------------|------------------|
| **C-N1** | S > 0 — persiste la diferencia | Axioma motor | S_i > κ_s; fracción que sobrevive f | `cg002_experimentos_arco.py` |
| **C-N1.1** | Extinción sin reverso | μ, κ_s | Nodos con S ≤ κ_s eliminados | motor campo medio |
| **C-N1.3** | Ω_persistente ⊆ Ω_posible | Filtración | ~50% sobrevive; hemisferio alineado | barridos constantes |
| **C-N2** | I ⟺ E — acoplamiento | c_ij = U_i·U_j; flujo η·g·√(sat S) | Cooperación/competencia por firma | arco, motor |
| **C-N2.1** | Exterior = otras diferencias | Campo medio v = Σ m_j U_j | Factorización O(N) | `evolucion_campo_medio` |
| **C-N2.5** | Espacio emerge | Dimensión, embedding | D_corr, rango estructural | `dimension`, v0.2 |
| **C-N2.5.5** | Asimetría primordial | θ_CP; R(θ) en acoplamiento | imbalance, tracking | `cg002_acoplamiento.py` |
| **C-N2.5.7** | Dirección T̂ | Flecha de orientación | sign(θ_CP), vector predictor | veredictos V3 |
| **C-N2.5.8** | Inercia histórica | τ + shock orientación | Umbral voltear ×2 vs >×8 | `inercia` |
| **C-N2.5.9** | Coexistencia dominios | Acoplamiento local rc | orden_local vs global | `coexistencia` |
| **C-N2.5.10** | Cancelación orientaciones | Antipodal / par opuesto | m_eff/K = ½; grano 1/√K | `cierre_kappaDelta_regla.py` |
| **C-N2.6** | Curvatura sin coordenadas | Embedding espectral / grafo | dim_efectiva, clumping | observables v02, visor |
| **C-N2.6.4** | Auto-similaridad | Ley potencias en rc crítico | R² barrido génesis | `cg002_genesis_barridos.js` |
| **C-N2.7** | Transición / criticidad | Barrido rc | pico fluctuación rc≈3 | `criticidad` |
| **C-N2.7.4** | Correspondencia gravedad | — | CG004: NO universal (tipo-carga) | CC |
| **C-N2.7.6** | No reducción física | Límite epistemológico | No derivar 4 fuerzas ni c, π literal | informes |
| **C-N2.8** | Invariantes del cierre | U_Cos parcial en CG002 | `invariantes` | arco |
| **C-N2.8.1** | Invariancia ≠ escalar | Lectura exceso/grano | Estructura > número | arco κ_Δ |
| **C-N2.8.4** | κ_Δ > 0 | Grano mínimo diferencia | κ_Δ ≡ 2π/K; grano 1/√K | grain null, barridos |
| **C-N2.8.9a** | Tipología ruptura | κ violado → dinámica distinta | homogeneización, desacoplamiento, disolución | `invariantes` |
| **C-N2.8.10** | Cadena κ_P⇒κ_Δ⇒… | Posibilidad, no operación | Orden dependencias en nulls | arco |
| **C-N2.8.14a** | κ magnitud dominio-específica | exceso(d) familia | No un solo 0.016 universal | exceso barrido |
| **C-N2.8.15** | No constantes físicas | Constantes emergentes | Ley vs historia; no π literal | constantes 1000 |
| **C-N3** | Historia restringe futuro | Flecha τ | Entropía orientación ↓ monótona | `flecha` |
| **C-N5.1** | Banda viabilidad | sat(S) = S/(1+S/S_BAND) | Descartado como origen del ½ | barrido banda |

---

## 3. κ en U_Cos — estado en CG002 y ANIMA (empalme)

| Símbolo | Nodo | CG002 | ANIMA (Parte II) | Proxy empalme |
|---------|------|-------|------------------|---------------|
| **κ_P** | C-N2.8.3 | Parcial | `historia`, metabolismo | ruptura μ alto / agotamiento |
| **κ_Δ** | C-N2.8.4 | **Sí** | `delta_struct`, `mem_ds_*` | Fase 1 `empalme_estimulos_fase1.json` |
| **κ_V** | C-N2.8.6 | Parcial | `A_sys_env`, RC díada | RC off; A→0 |
| **κ_O** | C-N2.8.5 | — | `\|e_R\|`, `mem_reflejo` | Fase 3; `R01_saturacion_colapso` |
| **κ_LF** | C-N2.8.7 | — | `LF_op`, `LF_struct` | `KAPPA['kLF']` en `VST_Genoma` |
| **κ_H** | C-N2.8.16 | — | `H_conductual` | presencia conducta medible |
| **Λ_Cos** | C-N2.8.12 | — | `salud().Lambda_Cos` | Fase 3; propiocepción `prop_dW` |

*Protocolo de costura:* [`PROTOCOLO_EMPALME_CG002_ANIMA.md`](PROTOCOLO_EMPALME_CG002_ANIMA.md)

---

## 4. Observables CG002 — definiciones canónicas

| Nombre | Fórmula | Capa | Nulo |
|--------|---------|------|------|
| **f_persiste** | n_surv / N | Filtración | ~½ (hemisferio) |
| **orden_a1** | \|Σ U_vivos / n\| | Esfera S^{d−1} | nulo_geom(d); finito n_surv |
| **exceso_orden** | orden − nulo_finito | Fina | >0 mientras κ_Δ>0 |
| **excess_L2** | √(Σ(f_k − 1/K)²) | Gruesa | √(K−1)/(KN) combinatorio |
| **m_eff/K** | 1/(K·Σf_k²) | Gruesa | ½ ⟺ grano 1/√K |
| **frac_pos** | fracción U·û > 0 | Control hemisferio | ≈1 |
| **grano_nulo** | √(K−1)/(KN) | Línea base | Sin dinámica, uniforme Z_K |

---

## 5. Identificaciones vigentes (revisables)

```
κ_Δ  ≡  2π/K                    (identificación natural en ℤ_K)
grano_grueso  =  1/√K  =  √(κ_Δ / 2π)
m_eff/K  =  1/2                  (derivado — C-N2.5.10, no C-N5.1)
```

**Estado:** identificación operativa vigente al 30-jun-2026. Revisable sin tocar hechos medidos (m_eff/K, L2, exponentes).

---

## 6. Nodos explícitamente fuera de CG002 (no reclamar)

| Tema | Nodo | Veredicto CG002 |
|------|------|-----------------|
| Gravedad universal | C-N2.7.4 | ❌ Insertar, no derivar (CG004) |
| π literal | — | ❌ Sin cierre geométrico |
| Órbita / spin sembrado | — | ❌ Layout o condición inicial; no emergencia |
| U_Cos completo | C-N2.8.11 | Frontera → Parte II / ANIMA |
| Λ_Cos, κ_O, κ_LF, κ_H | C-N2.8.12–16 | Parte II |

---

## 7. Mapa archivo → nodo

| Archivo | Nodos primarios |
|---------|-----------------|
| `cg002_experimentos_arco.py` | C-N1…C-N3, C-N2.5–2.8 |
| `cg002_constantes_1000.py` | C-N2.8.15, C-N1.3 |
| `cg002_exceso_barrido.py` | C-N2.8.4, C-N2.8.14a |
| `cg002_dynamic_l2_sweep.py` | C-N2.8.4 |
| `grain_null_model.py` | Línea base κ_Δ |
| `cierre_kappaDelta_regla.py` | C-N2.5.10, C-N5.1 |
| `cg002_acoplamiento.py` | C-N2.5.5, C-N2.5.7, C-N2.5.10 |
| `Informe_arco_kappaDelta_simple.md` | Síntesis κ_Δ |

---

*Mantener sincronizado con `INFORME_CG002_VEREDICTOS_TABLA.md` y `CIERRE_ARCO_CG002_AUTORITATIVO.md`.*