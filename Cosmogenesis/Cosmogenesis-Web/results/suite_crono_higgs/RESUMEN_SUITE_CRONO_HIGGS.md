# SUITE CRONO_HIGGS — resultados

**Fecha:** 2026-07-22  
**Código:** `Cosmogenesis-Web/codigo/suite_crono_higgs/suite_crono_higgs.py`  
**JSON:** `results/suite_crono_higgs/suite_crono_higgs_result.json`

---

## Principio (cronología del relato → contrato del juguete)

| Época (relato) | Condición | Qué NO puede haber |
|----------------|-----------|---------------------|
| Antes ~10⁻¹² s, T ≳ 10¹⁵ K | Campo simétrico, V~0 | Masa por VEV, bosón como grumo del vacío |
| ~10⁻¹² s, T ~ T_c | Ruptura / VEV | — |
| Después | Medio congelado; partículas “sienten arrastre” | Claim de Higgs **antes** de la ruptura |

**Regla de la suite:** sin pasar A1–A3 (orden), el claim tipo Higgs queda **suspendido**.  
Barridos **amplios** (no sintonía a 1/1836). CI: Φ ~ ruido pequeño (**sin VEV impuesto**).

Umbrales pre-registrados: `VEV_PRE_MAX=0.20`, `VEV_POST_MIN=0.12`, `SEP_THR=0.06`, `SEP_PRE_MAX=0.05`, `RATE_PASS=0.55`.

---

## Veredicto global

### `CHRONO_FAIL_HIGGS_CLAIM_SUSPENDED`

En el sentido estricto del contrato de la suite:

- **0 / 5** bloques de barrido **A** (familias que definen la ruptura) superan `rate_chrono ≥ 0.55` en **todo** el rango amplio.
- Por tanto: **no** se declara cronología universalmente admisible → **claim Higgs de la línea v3/v4 queda suspendido** hasta acotar la isla de parámetros donde el orden se cumple, o rediseñar para matar masa pre-SSB.

Eso **no** borra los hallazgos locales (baseline e isla media). Los separa del claim global.

---

## Bloque A — orden cronológico (barridos amplios)

| bloque | rate_chrono | A1 VEV | A2 masa post | A3 fluc post | early_mass_fail | broad_pass |
|--------|-------------|--------|--------------|--------------|-----------------|------------|
| A1_sweep_TC | 0.37 | 0.56 | 0.52 | 0.56 | **0.41** | no |
| A1_sweep_R0 | 0.42 | 0.75 | 0.54 | 0.75 | **0.42** | no |
| A1_sweep_U | 0.29 | 0.79 | 0.36 | 0.79 | **0.57** | no |
| A2_sweep_H_EXP | 0.48 | 1.00 | 0.48 | 1.00 | 0.38 | no |
| A3_sweep_freeze | 0.33 | 0.40 | 0.93 | 0.40 | 0.00 | no |

### Hallazgo crítico: masa **antes** de la ruptura

`a2_fail_early` = separación de masa REAL−NULL ya grande **pre-T_c**.

- En A1_TC / R0 / U: **~40–57%** de corridas fallan por masa temprana.
- Cuando falla: `pre_sep` típico **0.10–0.22** (sobre umbral).
- Cuando pasa: `pre_sep` típico **0.01–0.06**, `post_sep` **0.12–0.15**.

Eso es exactamente lo que la cronología del Higgs **prohíbe**:  
no puede haber “arrastre de masa tipo VEV” mientras el campo aún está en la fase simétrica de relato.

### Islas donde el orden sí emerge (no es sintonía a 1/1836; es mapa del fallido)

**TC (temperatura crítica del juguete):**

| TC | rate_chrono |
|----|-------------|
| 0.25–0.475 | **0.00** |
| 0.55–0.85 | **0.67** |

→ Si la ruptura ocurre “demasiado tarde” en el enfriamiento (TC bajo), el orden se rompe o no hay ventana limpia.

**R0 (profundidad del potencial):**

| R0 | rate_chrono |
|----|-------------|
| 0.5–1.0 | 0.00 |
| 1.5–2.5 | ~0.67 |
| 3.0–4.0 | 0.33–0.67 (mixto) |

---

## Bloque B — otros enfoques + ablaciones

| bloque | rate_chrono | broad_pass | nota |
|--------|-------------|------------|------|
| B2_transport (D_PHI × SIGMA0) | 0.23 | no | extremos de ruido/difusión matan VEV limpio |
| B2_G_RHO | 0.50 | no | frontera de la frontera |
| **B3_medium** (ALPHA × MIX) | **0.60** | **sí** | con potencial baseline, el medio tiene isla amplia |
| **B4_L** | **0.61** | **sí** | L=16…40 estable en baseline |
| **B6_seeds** | **0.75** | **sí** | 9/12 seeds baseline |

### Ablaciones (4 seeds c/u) — lectura causal

| ablación | rate_chrono | rate_A2 (masa post) | mean post_sep |
|----------|-------------|---------------------|---------------|
| **full** | **0.75** | **0.75** | **0.131** |
| no_medium | 0.00 | **0.00** | 0.033 |
| blind_cuts_only | 0.00 | **0.00** | 0.033 |
| rho_fixed | 0.75 | 0.75 | 0.128 |
| no_freeze_T | 0.75 | 0.75 | 0.129 |
| no_freeze_rho | 0.75 | 0.75 | 0.131 |
| **no_freeze_any** | **0.00** | **0.00** | 0.019 |

Lecturas:

1. **Sin medio** (o cortes ciegos): **no hay masa post-SSB** → el germen de masa **no** es geometría pura.
2. **Sin ningún freeze**: cronología y masa se caen → hace falta congelar el orden en algún momento.
3. Freeze solo-T o solo-ρ, o ρ fija: en baseline, **casi indistinguibles** del full (como v4).
4. **medium_matters_vs_blind = True** (full A2 0.75 vs no_medium 0.00).

---

## Relación con Higgs v3 / v4 previos

| Hecho previo | Relectura con la suite |
|--------------|-------------------------|
| v3/v4 PARTIAL robusto | Usaban CI con **dominios ya sembrados** (`0.6·sign`) ≈ VEV/muros **desde t=0** → **saltan** la fase simétrica del relato |
| Suite (esta) | CI **simétrica** (ruido 0.08) → la ruptura debe **emerger**; el claim es más exigente y **más fiel** a tu cronología |
| Estiramiento ρ | Sigue compatible; no fabrica la masa (ablación rho_fixed) |
| Conclusión | El PARTIAL v3/v4 **no queda certificado** como “Higgs post-EWSB” hasta resolver **masa pre-SSB** y/o partir siempre de fase simétrica |

---

## Qué queda admitido vs suspendido

| Claim | Estado |
|-------|--------|
| “En una isla baseline, VEV crece al enfriar y hay masa post con medio ON” | **Admitido localmente** (B6 0.75, ablación full 0.75) |
| “Cronología estable en **todo** barrido amplio de potencial/transporte/freeze” | **Rechazado** (0/5 bloques A) |
| “Masa / Higgs antes de la ruptura no ocurre” | **Falso en ~40% del barrido A** → problema abierto prioritario |
| “El medio es necesario para la masa post” | **Admitido** (ablación) |
| “1/1836 / SM Higgs 125 GeV” | **Ni buscado ni reclamado** |

---

## Próximos experimentos naturales (capítulo abierto)

1. **Matar masa pre-SSB** — factor de masa solo activo si `Tnorm < TC` y/o `|Φ|` del dominio > umbral emergente (no número SM); barrer esa puerta como **hipótesis de cronología**, no de jerarquía.
2. **Prohibir CI con sign(Φ)** en toda la línea Higgs — solo ruido simétrico.
3. **Barrido fino solo en isla** TC∈[0.55,0.85], R0∈[1.5,2.5] **después** de que A2 early_fail < 0.1.
4. **Proxy bosón** más limpio: espectro de fluctuaciones de Φ **sobre el VEV** solo en `post_SSB_frozen`, NULL = misma red pre-SSB.
5. Re-correr v4 **solo** si (1)+(2) pasan en barrido amplio.

---

## Artefactos

- `codigo/suite_crono_higgs/suite_crono_higgs.py`
- `results/suite_crono_higgs/suite_crono_higgs_result.json`
- `results/suite_crono_higgs/suite_run.log`
- este resumen

**Tiempo de suite:** ~68 s · **~250+** corridas · L≈28 · 320 pasos · sin sintonía a 1/1836.
