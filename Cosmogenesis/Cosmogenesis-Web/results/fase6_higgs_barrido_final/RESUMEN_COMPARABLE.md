# Fase 6 — Higgs barrido fino A/B (Mac)

**Zip:** `/Users/alexis/Downloads/fase6_higgs_barrido_final.zip`  
**Código:** `codigo/fase6_higgs_barrido_final/`  
**Logs:** `results/fase6_higgs_barrido_final/*.log`  
**Fecha:** 2026-07-22  
**EXIT:** A=0, B=0

---

## A — Scalar Φ (barrido r0 × u)

Criterio VIVO del script: `<|Phi|> > 0.15` y k1>10 y k3>5.

| r0 | u | ⟨\|Φ\|⟩ | ratio | k1 | k3 | flag |
|---:|--:|--------:|------:|---:|---:|------|
| 0.5 | 0.1 | 0.009 | 0.309 | 177 | 19 | **APAGADO** |
| 0.5 | 0.3 | 0.008 | 0.304 | 180 | 25 | APAGADO |
| 0.5 | 0.7 | 0.009 | 0.339 | 171 | 24 | APAGADO |
| 1.0 | 0.1 | 0.007 | 0.293 | 191 | 23 | APAGADO |
| 1.0 | 0.3 | 0.007 | 0.350 | 171 | 26 | APAGADO |
| 1.0 | 0.7 | 0.007 | 0.364 | 174 | 13 | APAGADO |
| 2.0 | 0.1 | 0.006 | 0.443 | 180 | 20 | APAGADO |
| 2.0 | 0.3 | 0.006 | 0.343 | 166 | 25 | APAGADO |
| 2.0 | 0.7 | 0.006 | 0.305 | 175 | 27 | APAGADO |
| 3.0 | 0.1 | 0.005 | 0.295 | 181 | 24 | APAGADO |
| 3.0 | 0.3 | 0.005 | 0.346 | 181 | 18 | APAGADO |
| 3.0 | 0.7 | 0.006 | 0.361 | 188 | 22 | APAGADO |

**Mac A:** **0/12 VIVO**. ⟨\|Φ\|⟩ ∈ 0.005–0.009 (como README: “apagado <0.05”).  
Clusters **sí** se sostienen (k1~170–190, k3~13–27).  
ratio **0.29–0.44 = O(1)** (no 0.00054).

---

## B — Fricción fase (barrido D × K0 × α)

Criterio VIVO: `0 < ratio < 0.9` y `eta_k1 > 1.1 * eta_k3`.

### Ventana VIVA (Mac) — coincide con sintonía Meta

| D | K0 | α | ratio | η_k1 | η_k3 | k1 | k3 | flag |
|--:|---:|--:|------:|-----:|-----:|---:|---:|------|
| 0.10 | 2.0 | 0.5 | **0.36971** | 0.2241 | 0.1460 | 195 | 19 | **VIVO** |
| 0.10 | 1.0 | 1.0 | **0.37574** | 0.5182 | 0.2509 | 185 | 28 | **VIVO** |
| 0.01 | 2.0 | 0.5 | **0.40261** | 1.0905 | 0.9700 | 174 | 20 | **VIVO** |
| 0.05 | 1.0 | 0.5 | **0.45525** | 0.9155 | 0.5402 | 173 | 26 | **VIVO** |

### Conteo barrido completo B (27 celdas)

| | n |
|--|--:|
| **VIVO** | **18** |
| GEOM/APAG | 9 |

Mejores ratios VIVO (más bajos, aún O(1)):

- D=0.10 K0=2.0 α=0.5 → **0.370** (η1>η3)
- D=0.10 K0=1.0 α=1.0 → **0.376**
- D=0.10 K0=1.0 α=0.5 → **0.411**

---

## Veredicto Mac (alineado con README_SINTONIA)

| Brazo | Estado |
|-------|--------|
| **A** | **VEV apagado** en todo el grid r0≤3, u≤0.7. Clusters vivos; masa O(1). Hace falta r0≫3 o más anclaje/ruido (README: r0>5 o Tc<0.2). |
| **B** | **Ventana VIVA real** D∈[0.01,0.1], K0∈[1,2], α∈[0.5,1]. η_k1 ≳ η_k3. ratio **0.37–0.55** O(1), no 0.00054. |
| Jerarquía 1/1836 | **No** en este barrido |
| Anti-Shannon | Fórmula única en B; A usa y0 global. No se retocó a 1/1836 |

---

## Comparación con sintonía Meta (README)

| Claim README | Mac |
|--------------|-----|
| A: ⟨Φ⟩ 0.005–0.009 APAGADO | **Sí** (max 0.009) |
| B: D=0.10 K0=2 α=0.5 ratio=0.36971 | **Exacto 0.36971** |
| B: D=0.10 K0=1 α=1.0 ratio=0.37574 | **Exacto 0.37574** |
| B: D=0.01 K0=2 α=0.5 ratio=0.40261 | **Exacto 0.40261** |
| B: D=0.05 K0=1 α=0.5 ratio=0.45525 | **Exacto 0.45525** |

**Meta ↔ Mac: 100% alineados en las celdas de sintonía reportadas.**
